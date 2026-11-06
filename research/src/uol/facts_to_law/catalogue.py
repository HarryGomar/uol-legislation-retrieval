"""Chunked article views of the candidate catalogue for dense encoders (WP2).

Chunk 0 of every article is a text whose ``'passage: '``-prefixed tokenization,
truncated at ``max_tokens``, equals that of the full ``instrument_article_body``,
so FirstP over chunk 0 reproduces the October truncated dense system. Later
chunks repeat the instrument title and ``Artículo <label> (continuación)`` before
a word-aligned body window that shares at least ``overlap`` tokens with the
previous window; each chunk fits ``max_tokens`` including prefix and special
tokens. Windows are cut at character offsets of one tokenization of the full
text, never by decoding token ids, so chunk texts keep the source characters.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from uol.facts_to_law import spec
from uol.facts_to_law.spec import FORMAT_VERSION, ROOT, read_jsonl, sha256_file, write_json, write_jsonl

CHUNKER_VERSION = "ftl-chunks-1"
PASSAGE_PREFIX = "passage: "
WORD_START = "▁"
CONTINUATION = "Artículo {label} (continuación)"
MAX_SNAP = 16
BATCH = 256
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "sentencepiece.bpe.model", "special_tokens_map.json")

Tokens = tuple[list[int], list[tuple[int, int]], list[bool]]


class TokenView:
    """Ids, character offsets and word starts from a fast tokenizer with offset mapping.

    A token starts a word when its string begins with ``WORD_START`` (SentencePiece
    Metaspace). Counts include special tokens; ``n_prefix`` is the token cost of
    ``prefix`` in front of text.
    """

    def __init__(self, tokenizer: Any, *, prefix: str = PASSAGE_PREFIX, max_tokens: int = 512):
        self.tokenizer = tokenizer
        self.prefix = prefix
        self.max_tokens = max_tokens
        self._word_start: dict[int, bool] = {}
        self.n_special = len(tokenizer([""], verbose=False)["input_ids"][0])
        self.n_prefix = self.count([prefix + "x"])[0] - self.count(["x"])[0]

    def tokens(self, texts: Sequence[str]) -> list[Tokens]:
        encoded = self.tokenizer(list(texts), add_special_tokens=False, return_offsets_mapping=True, verbose=False)
        out = []
        for ids, offsets in zip(encoded["input_ids"], encoded["offset_mapping"]):
            unknown = sorted({i for i in ids if i not in self._word_start})
            if unknown:
                for i, token in zip(unknown, self.tokenizer.convert_ids_to_tokens(unknown)):
                    self._word_start[i] = token.startswith(WORD_START)
            out.append((list(ids), [(int(a), int(b)) for a, b in offsets], [self._word_start[i] for i in ids]))
        return out

    def count(self, texts: Sequence[str]) -> list[int]:
        """Untruncated encoder input lengths, special tokens included."""
        return [len(ids) for ids in self.tokenizer(list(texts), verbose=False)["input_ids"]]

    def encoder_ids(self, texts: Sequence[str]) -> list[list[int]]:
        """Prefixed ids truncated at ``max_tokens``, as the October encoder built them."""
        encoded = self.tokenizer([self.prefix + t for t in texts], truncation=True, max_length=self.max_tokens, verbose=False)
        return [list(ids) for ids in encoded["input_ids"]]


def chunk_id(candidate_id: str, chunk_index: int) -> str:
    return f"{candidate_id}::chunk={chunk_index}"


def title_of(row: dict[str, Any]) -> str:
    """Instrument title: the first line of ``instrument_article_body``."""
    return row["instrument_article_body"].split("\n", 1)[0].rstrip("\r")


def chunk_article(row: dict[str, Any], view: TokenView, *, overlap: int = 64, cap: int = 32,
                  tokens: Tokens | None = None, full_count: int | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Chunk rows and a coverage record for one catalogue article.

    The record holds token windows ``[start, end)`` over one tokenization of the
    full text (window 0 is what the October encoder saw), the chunk-0 mode and
    the untouched remainder when ``cap`` applies.
    """
    if overlap < 0 or cap < 1:
        raise ValueError("require overlap >= 0 and cap >= 1")
    candidate_id, text = row["candidate_id"], row["instrument_article_body"]
    ids, offsets, starts = tokens if tokens is not None else view.tokens([text])[0]
    if full_count is None:
        full_count = view.count([view.prefix + text])[0]
    n = len(ids)
    info: dict[str, Any] = {"candidate_id": candidate_id, "text_tokens": n, "october_truncated": full_count > view.max_tokens}
    if full_count <= view.max_tokens:
        info.update(chunk0_mode="whole", windows=[[0, n]], remainder_tokens=0, capped=False)
        return [_row(candidate_id, 0, text, full_count, False)], info
    first = min(view.max_tokens - view.n_special - view.n_prefix, n)
    chunk0, mode, n0 = _first_chunk(text, offsets, starts, first, view)
    title = title_of(row)
    header = f"{title}\n{CONTINUATION.format(label=row['article_label'])}"
    header_tokens = view.count([header])[0] - view.n_special
    budget = view.max_tokens - view.n_special - view.n_prefix - header_tokens
    if budget <= overlap + MAX_SNAP:
        raise ValueError(f"window budget {budget} cannot hold overlap {overlap}: {candidate_id}")
    body_start = next((i for i, (_, end) in enumerate(offsets) if end > len(title)), n)
    texts, counts, windows = [chunk0], [n0], [[0, first]]
    while windows[-1][1] < n and len(windows) < cap:
        prev_start, prev_end = windows[-1]
        start = _window_start(starts, prev_start, prev_end, body_start, overlap)
        end, chunk_text, count = _fit_window(text, offsets, starts, start, prev_end, budget, header, view)
        texts.append(chunk_text)
        counts.append(count)
        windows.append([start, end])
    covered = windows[-1][1]
    capped = covered < n
    info.update(chunk0_mode=mode, windows=windows, remainder_tokens=n - covered, capped=capped,
                header_tokens=header_tokens, window_budget=budget)
    return [_row(candidate_id, i, t, c, capped) for i, (t, c) in enumerate(zip(texts, counts))], info


def _row(candidate_id: str, index: int, text: str, n_tokens: int, capped: bool) -> dict[str, Any]:
    return {"chunk_id": chunk_id(candidate_id, index), "candidate_id": candidate_id, "chunk_index": index,
            "text": text, "n_tokens": n_tokens, "capped": capped}


def _first_chunk(text: str, offsets: list[tuple[int, int]], starts: list[bool], first: int,
                 view: TokenView) -> tuple[str, str, int]:
    """Shortest verified chunk-0 text: cut after token ``first``, else at the end of its word, else the full text."""
    target = view.encoder_ids([text])[0]
    n = len(offsets)
    candidates = []
    if 0 < first < n:
        candidates.append(("cut", text[:offsets[first - 1][1]]))
        end = first
        while end < n and not starts[end]:
            end += 1
        if end > first:
            candidates.append(("word_end", text[:offsets[end - 1][1]]))
    for mode, candidate in candidates:
        if view.encoder_ids([candidate])[0] == target:
            return candidate, mode, len(target)
    return text, "full_text", len(target)


def _window_start(starts: list[bool], prev_start: int, prev_end: int, body_start: int, overlap: int) -> int:
    """Start ``overlap`` tokens before ``prev_end``, moved back to a word start by at most ``MAX_SNAP`` tokens."""
    low = max(prev_start + 1, min(body_start, prev_end))
    start = max(prev_end - overlap, low)
    for candidate in range(start, max(low, start - MAX_SNAP) - 1, -1):
        if starts[candidate]:
            return candidate
    return start


def _fit_window(text: str, offsets: list[tuple[int, int]], starts: list[bool], start: int, prev_end: int,
                budget: int, header: str, view: TokenView) -> tuple[int, str, int]:
    """Largest word-aligned end within ``budget`` tokens whose prefixed chunk fits ``max_tokens``."""
    n = len(offsets)
    limit = min(n, start + budget)
    while True:
        end = limit
        if end < n:
            boundary = end
            while boundary > prev_end and not starts[boundary]:
                boundary -= 1
            if boundary > prev_end:
                end = boundary
        chunk_text = f"{header}\n{text[offsets[start][0]:offsets[end - 1][1]].strip()}"
        count = view.count([view.prefix + chunk_text])[0]
        if count <= view.max_tokens:
            return end, chunk_text, count
        limit = end - 1
        if limit <= prev_end:
            raise ValueError("no window after the previous chunk fits max_tokens")


def _batches(rows: Iterable[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def chunk_catalogue(articles_jsonl: Path, tokenizer: Any, max_tokens: int = 512, overlap: int = 64, cap: int = 32,
                    *, prefix: str = PASSAGE_PREFIX) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chunk rows in catalogue order and one coverage record per article."""
    view = TokenView(tokenizer, prefix=prefix, max_tokens=max_tokens)
    rows: list[dict[str, Any]] = []
    infos: list[dict[str, Any]] = []
    for batch in _batches(read_jsonl(Path(articles_jsonl)), BATCH):
        texts = [r["instrument_article_body"] for r in batch]
        for row, tokens, count in zip(batch, view.tokens(texts), view.count([prefix + t for t in texts])):
            article_rows, info = chunk_article(row, view, overlap=overlap, cap=cap, tokens=tokens, full_count=count)
            rows.extend(article_rows)
            infos.append(info)
    return rows, infos


def build_chunks(articles_jsonl: Path, tokenizer: Any, max_tokens: int = 512, overlap: int = 64, cap: int = 32,
                 *, out_dir: Path | None = None, prefix: str = PASSAGE_PREFIX) -> list[dict[str, Any]]:
    """Chunk rows ``{chunk_id, candidate_id, chunk_index, text, n_tokens, capped}`` in catalogue order.

    With ``out_dir`` (which must not exist) also writes ``chunks.jsonl`` and ``manifest.json``.
    """
    if out_dir is not None and Path(out_dir).exists():
        raise FileExistsError(f"chunk view exists; choose a new directory: {out_dir}")
    rows, infos = chunk_catalogue(articles_jsonl, tokenizer, max_tokens, overlap, cap, prefix=prefix)
    if out_dir is not None:
        write_chunk_view(Path(out_dir), rows, infos, articles_jsonl=Path(articles_jsonl), tokenizer=tokenizer,
                         params={"max_tokens": max_tokens, "overlap": overlap, "cap": cap, "prefix": prefix})
    return rows


def summarize(rows: Sequence[dict[str, Any]], infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Counts, token totals, chunk-0 modes, overlaps and capped articles of a chunk view."""
    n_tokens = [r["n_tokens"] for r in rows]
    per_article = [len(i["windows"]) for i in infos]
    overlaps = [a[1] - b[0] for i in infos for a, b in zip(i["windows"], i["windows"][1:])]
    headers = [i["header_tokens"] for i in infos if "header_tokens" in i]
    budgets = [i["window_budget"] for i in infos if "window_budget" in i]
    capped = [{"candidate_id": i["candidate_id"], "text_tokens": i["text_tokens"], "covered_tokens": i["windows"][-1][1],
               "remainder_tokens": i["remainder_tokens"]} for i in infos if i["capped"]]
    return {
        "articles": len(infos),
        "chunks": len(rows),
        "chunks_per_article": {"max": max(per_article, default=0), "mean": sum(per_article) / max(len(per_article), 1),
                               "histogram": {str(k): v for k, v in sorted(Counter(per_article).items())}},
        "october_truncated_articles": sum(bool(i["october_truncated"]) for i in infos),
        "chunk0_modes": dict(sorted(Counter(i["chunk0_mode"] for i in infos).items())),
        "n_capped": len(capped),
        "capped_articles": capped,
        "encoder_input_tokens": sum(n_tokens),
        "max_n_tokens": max(n_tokens, default=0),
        "catalogue_text_tokens": sum(i["text_tokens"] for i in infos),
        "uncovered_text_tokens": sum(i["remainder_tokens"] for i in infos),
        "overlap_tokens": {"min": min(overlaps), "max": max(overlaps), "mean": sum(overlaps) / len(overlaps)} if overlaps else None,
        "header_tokens": {"min": min(headers), "max": max(headers)} if headers else None,
        "window_budget": {"min": min(budgets), "max": max(budgets)} if budgets else None,
    }


def tokenizer_identity(tokenizer: Any) -> dict[str, Any]:
    """Tokenizer class and SHA-256 of its local files, when it was loaded from a directory."""
    name = getattr(tokenizer, "name_or_path", None)
    directory = Path(name) if name else None
    files = {}
    if directory is not None and directory.is_dir():
        files = {f: sha256_file(directory / f) for f in TOKENIZER_FILES if (directory / f).is_file()}
    return {"class": type(tokenizer).__name__, "name_or_path": str(name) if name else None, "files": files}


def code_hashes() -> dict[str, str]:
    paths = (Path(__file__).resolve(), Path(spec.__file__).resolve())
    return {relative_path(p): sha256_file(p) for p in paths}


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_chunk_view(out_dir: Path, rows: Sequence[dict[str, Any]], infos: Sequence[dict[str, Any]], *,
                     articles_jsonl: Path, tokenizer: Any, params: dict[str, Any]) -> dict[str, Any]:
    """Write ``chunks.jsonl`` and ``manifest.json`` into a new directory (staged, then renamed)."""
    out_dir = Path(out_dir)
    staging = out_dir.with_name(f"{out_dir.name}.partial-{os.getpid()}")
    if out_dir.exists() or staging.exists():
        raise FileExistsError(f"chunk view exists; choose a new directory: {out_dir}")
    staging.mkdir(parents=True)
    write_jsonl(staging / "chunks.jsonl", rows)
    release_manifest = articles_jsonl.parent.parent / "manifest.json"
    release: dict[str, Any] = {"manifest_path": None}
    if release_manifest.is_file():
        data = json.loads(release_manifest.read_text(encoding="utf-8"))
        release = {"manifest_path": str(release_manifest), "manifest_sha256": sha256_file(release_manifest),
                   "release_id": data.get("release_id"),
                   "articles_sha256_in_release_manifest": data.get("files", {}).get("candidates/articles.jsonl")}
    catalogue_sha256 = sha256_file(articles_jsonl)
    manifest = {
        "format_version": FORMAT_VERSION,
        "kind": "facts_to_law_v2_chunk_view",
        "chunker_version": CHUNKER_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "catalogue": {"path": str(articles_jsonl), "sha256": catalogue_sha256, "articles": len(infos),
                      "text_field": "instrument_article_body"},
        "candidate_release": release,
        "tokenizer": tokenizer_identity(tokenizer),
        "params": {**params, "max_snap": MAX_SNAP, "continuation": CONTINUATION, "separator": "\n"},
        "rules": {
            "chunk_id": "<candidate_id>::chunk=<chunk_index>",
            "chunk_0": "prefix+text truncated at max_tokens equals prefix+instrument_article_body truncated at max_tokens; "
                       "modes: whole (untruncated), cut (after the last kept token), word_end (to the end of the split word, "
                       "removed again by truncation), full_text (fallback)",
            "later_chunks": "title line + '\\n' + continuation heading + '\\n' + stripped body window; windows are word-aligned "
                            "spans of one tokenization of the full text; next start = previous end - overlap, moved back "
                            "to a word start by at most max_snap tokens; never re-enters the title",
            "n_tokens": "encoder input length: prefix + text with special tokens, after truncation at max_tokens",
            "capped": "every chunk of an article whose windows stop at cap before the end of its text",
        },
        "counts": summarize(rows, infos),
        "files": {"chunks.jsonl": sha256_file(staging / "chunks.jsonl")},
        "code": code_hashes(),
    }
    write_json(staging / "manifest.json", manifest)
    staging.rename(out_dir)
    return manifest
