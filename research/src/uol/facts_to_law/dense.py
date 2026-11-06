"""Offline E5-small encoding of chunked article views and chunk-to-article scores (WP2).

Encoding reuses ``uol.orchestration.embedding_usefulness.encoder.OfflineEncoder``
unchanged: SHA-256-verified installed files, ``local_files_only``, eager attention,
``'query: '``/``'passage: '`` prefixes, truncation at 512 tokens, attention-mask mean
pooling, L2 normalisation, float32 on CPU. Its ``encode`` is called directly; its
``cache`` is not, because it binds one array to one text list and cannot resume
shards. ``encode`` keeps input order, so chunk shards are sorted by token length
before encoding and restored before saving.

Command line: ``python -B -m uol.facts_to_law.dense {chunks,check,encode-chunks,parity-full,encode-queries}``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.facts_to_law import catalogue, spec
from uol.facts_to_law.catalogue import PASSAGE_PREFIX, TokenView, relative_path
from uol.facts_to_law.spec import FORMAT_VERSION, ROOT, load_candidate_ids, read_jsonl, sha256_file, sha256_text, write_json
from uol.orchestration.embedding_usefulness import encoder as october_encoder

QUERY_PREFIX = "query: "
PREFIXES = {"query": QUERY_PREFIX, "passage": PASSAGE_PREFIX}
MODES = ("maxp", "firstp", "meanp")
MAX_TOKENS = 512
MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MODEL_DIR = Path(r"C:\uol_legacy_home\.cache\retrieval_benchmarks_20260914\models\intfloat_multilingual-e5-small_614241f6")
MODEL_FILES = {
    "config.json": (655, "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959"),
    "model.safetensors": (470641600, "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477"),
    "modules.json": (387, "c6e29747481e8b5dd2b58401966aeac910de39092f90cda9a704b1545f902b04"),
    "sentence_bert_config.json": (57, "948201d8329907aae938fa62f9ceeed53f5694dacc2b87b9f3b78b37ee986529"),
    "sentencepiece.bpe.model": (5069051, "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865"),
    "special_tokens_map.json": (167, "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7"),
    "tokenizer.json": (17082730, "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"),
    "tokenizer_config.json": (443, "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b"),
}
MODEL_BINDING = {"model": MODEL_ID, "declared_revision": MODEL_REVISION, "path": str(MODEL_DIR),
                 "files": {name: {"bytes": size, "path": str(MODEL_DIR / name), "sha256": digest}
                           for name, (size, digest) in MODEL_FILES.items()}}
RECIPE = {"max_tokens": MAX_TOKENS, "truncation": "truncation=True, max_length=512, prefix and special tokens included",
          "pooling": "attention-mask mean followed by L2", "dtype": "float32", "device": "cpu", "attention": "eager",
          "padding": "longest in batch"}
V10_ARTICLES = ROOT / "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates/articles.jsonl"
OCTOBER_C01 = ROOT / "experiments/runs/embedding_usefulness_20261001/c01_v9_restricted_core"
RUN_KIND = "facts_to_law_v2_chunk_vectors"
OUTPUT_FILES = ("chunk_vectors.npy", "article_index.npy", "chunk_ordinal.npy", "ids.json")
PARITY_SALT = "ftl-v2-firstp-parity-20261004"
SMOKE_SEED = 20261004


class EncodingStopped(RuntimeError):
    """Raised between encoder batches once the run's stop file exists."""


class _StopCheck:
    def __init__(self) -> None:
        self.stop_file: Path | None = None

    def __call__(self) -> None:
        if self.stop_file is not None and self.stop_file.exists():
            raise EncodingStopped(f"stop file present: {self.stop_file}")


_CHECK = _StopCheck()
_ENCODER: Any = None


def configure_runtime(threads: int) -> dict[str, Any]:
    """October CPU runtime (offline flags, deterministic algorithms) with ``threads`` intra-op threads."""
    env = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
           "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads)}
    os.environ.update(env)
    import torch
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    return {"env": env, "torch": torch.__version__, "intra_op_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": True}


def load_tokenizer() -> Any:
    """Installed E5-small fast tokenizer from local files, after SHA-256 verification."""
    for name in catalogue.TOKENIZER_FILES:
        if sha256_file(MODEL_DIR / name) != MODEL_FILES[name][1]:
            raise ValueError(f"installed tokenizer bytes changed: {name}")
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)


def get_encoder() -> Any:
    """The October ``OfflineEncoder`` over the bound installed model, loaded once per process."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = october_encoder.OfflineEncoder(MODEL_BINDING, device="cpu", check=_CHECK)
    return _ENCODER


def code_hashes() -> dict[str, str]:
    paths = (Path(__file__), Path(catalogue.__file__), Path(spec.__file__), Path(october_encoder.__file__))
    return {relative_path(p.resolve()): sha256_file(p) for p in paths}


def encode_texts(texts: Sequence[str], kind: str = "passage", batch_size: int = 8, threads: int = 16,
                 *, encoder: Any = None) -> np.ndarray:
    """L2-normalised float32 vectors [n, 384] in input order, with the October recipe."""
    return _encode(texts, kind=kind, batch_size=batch_size, threads=threads, encoder=encoder)[0]


def _encode(texts: Sequence[str], *, kind: str, batch_size: int, threads: int, encoder: Any) -> tuple[np.ndarray, dict]:
    if kind not in PREFIXES:
        raise ValueError(f"kind must be one of {sorted(PREFIXES)}")
    if encoder is None:
        configure_runtime(threads)
        encoder = get_encoder()
    values, cost = encoder.encode(list(texts), prefix=PREFIXES[kind], batch_size=batch_size, max_tokens=MAX_TOKENS)
    return _validated(values, len(texts)), cost


def _encode_sorted(encoder: Any, texts: Sequence[str], n_tokens: np.ndarray, *, batch_size: int) -> tuple[np.ndarray, dict]:
    """Passage vectors in input order, encoded in stable ascending ``n_tokens`` order."""
    order = np.argsort(np.asarray(n_tokens), kind="stable")
    values, cost = encoder.encode([texts[i] for i in order], prefix=PASSAGE_PREFIX, batch_size=batch_size, max_tokens=MAX_TOKENS)
    values = _validated(values, len(texts))
    restored = np.empty_like(values)
    restored[order] = values
    return restored, cost


def _validated(values: Any, n: int) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.float32)
    if values.ndim != 2 or len(values) != n or not np.isfinite(values).all() \
            or not np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5):
        raise ValueError("encoder output must be finite unit-norm rows, one per text")
    return values


def _encoder_identity(encoder: Any) -> dict[str, Any]:
    binding = MODEL_BINDING if encoder is None else getattr(encoder, "binding", None) or {"model": type(encoder).__name__}
    return {"model": binding.get("model"), "declared_revision": binding.get("declared_revision"),
            "files": {name: row["sha256"] for name, row in binding.get("files", {}).items()}}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _peak_rss() -> int | None:
    try:
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or info.rss)
    except Exception:
        return None


def _cpu_percent() -> float | None:
    try:
        import psutil
        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


class _Log:
    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        line = f"{_utc()} {message}"
        print(line, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(line + "\n")


def _shard_records(out_dir: Path) -> dict[int, dict[str, Any]]:
    path = out_dir / "progress.jsonl"
    return {row["shard"]: row for row in read_jsonl(path)} if path.is_file() else {}


def encode_chunks(chunks_jsonl: Path, out_dir: Path, shard_size: int = 4096, threads: int = 16, resume: bool = True,
                  *, batch_size: int = 16, encoder: Any = None, log_path: Path | None = None,
                  evidence: Sequence[Path] = ()) -> dict[str, Any]:
    """Encode chunk texts into float32 shards, then consolidate; returns the final manifest.

    On resume a shard is skipped when its recorded SHA-256 matches its file; the
    chunks, encoder and recipe must be unchanged. Creating ``<out_dir>/STOP``
    stops the run between batches; rerunning the same call resumes it.
    """
    chunks_jsonl, out_dir = Path(chunks_jsonl), Path(out_dir)
    if shard_size < 1 or batch_size < 1:
        raise ValueError("shard_size and batch_size must be positive")
    log = _Log(log_path)
    rows = list(read_jsonl(chunks_jsonl))
    texts = [r["text"] for r in rows]
    n_tokens = np.asarray([r["n_tokens"] for r in rows], dtype=np.int64)
    chunk_manifest = chunks_jsonl.parent / "manifest.json"
    config = {
        "format_version": FORMAT_VERSION,
        "kind": RUN_KIND,
        "chunks": {"path": str(chunks_jsonl), "sha256": sha256_file(chunks_jsonl), "rows": len(rows),
                   "manifest_path": str(chunk_manifest), "manifest_sha256": sha256_file(chunk_manifest)},
        "encoder": _encoder_identity(encoder),
        "recipe": {**RECIPE, "prefix": PASSAGE_PREFIX, "batch_size": batch_size, "shard_size": shard_size,
                   "order_within_shard": "stable ascending n_tokens, restored before saving"},
    }
    if out_dir.exists():
        if not resume:
            raise FileExistsError(f"vector run exists; resume it or choose a new directory: {out_dir}")
        config_path = out_dir / "run_config.json"
        if not config_path.is_file():
            raise FileExistsError(f"not a chunk-vector run directory: {out_dir}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if {key: previous.get(key) for key in config} != config:
            raise ValueError("resume requires the same chunks, encoder and recipe")
        if (out_dir / "manifest.json").is_file():
            log(f"complete run found; verifying {out_dir}")
            return load_chunk_matrix(out_dir).manifest
    else:
        out_dir.mkdir(parents=True)
        write_json(out_dir / "run_config.json", {**config, "created_utc": _utc(),
                                                 "evidence": {str(p): sha256_file(Path(p)) for p in evidence}})
    session = {"utc": _utc(), "pid": os.getpid(), "threads": threads, "batch_size": batch_size, "code": code_hashes(),
               "python": sys.version.split()[0], "cpu_count": os.cpu_count()}
    if encoder is None:
        session["runtime"] = configure_runtime(threads)
    _append_jsonl(out_dir / "sessions.jsonl", session)
    _CHECK.stop_file = out_dir / "STOP"
    shard_dir = out_dir / "shards"
    shard_dir.mkdir(exist_ok=True)
    records = _shard_records(out_dir)
    n_shards = max(1, math.ceil(len(rows) / shard_size))
    total_tokens = int(n_tokens.sum())
    started, session_tokens = time.perf_counter(), 0
    log(f"session start pid={os.getpid()} threads={threads} batch_size={batch_size} shards={n_shards} "
        f"chunks={len(rows)} tokens={total_tokens} out={out_dir}")
    try:
        for index in range(n_shards):
            begin, end = index * shard_size, min(len(rows), (index + 1) * shard_size)
            path = shard_dir / f"shard_{index:05d}.npy"
            record = records.get(index)
            if record and (record["begin"], record["end"]) == (begin, end) and path.is_file() \
                    and sha256_file(path) == record["sha256"]:
                log(f"shard {index + 1}/{n_shards} verified complete; skipped")
                continue
            if path.exists():
                log(f"shard {index + 1}/{n_shards} has no matching record; re-encoding")
            _CHECK()
            active = encoder if encoder is not None else get_encoder()
            _cpu_percent()
            t0 = time.perf_counter()
            values, cost = _encode_sorted(active, texts[begin:end], n_tokens[begin:end], batch_size=batch_size)
            seconds = time.perf_counter() - t0
            partial = path.with_name(f"{path.stem}.partial.npy")
            np.save(partial, values)
            os.replace(partial, path)
            tokens = int(n_tokens[begin:end].sum())
            record = {"shard": index, "begin": begin, "end": end, "rows": end - begin, "dimension": int(values.shape[1]),
                      "sha256": sha256_file(path), "seconds": seconds, "tokens": tokens,
                      "chunks_per_second": (end - begin) / seconds if seconds > 0 else None,
                      "tokens_per_second": tokens / seconds if seconds > 0 else None,
                      "truncated_inputs": len(cost.get("truncated_positions", [])), "threads": threads,
                      "batch_size": batch_size, "pid": os.getpid(), "peak_rss_bytes": _peak_rss(),
                      "system_cpu_percent": _cpu_percent(), "utc": _utc()}
            _append_jsonl(out_dir / "progress.jsonl", record)
            records[index] = record
            session_tokens += tokens
            done_rows = sum(r["rows"] for r in records.values())
            done_tokens = sum(r["tokens"] for r in records.values())
            rate = session_tokens / max(time.perf_counter() - started, 1e-9)
            finish = datetime.now(timezone.utc) + timedelta(seconds=(total_tokens - done_tokens) / rate)
            rss = (record["peak_rss_bytes"] or 0) / 2**30
            log(f"shard {index + 1}/{n_shards} rows {begin}-{end - 1}: {seconds:.1f}s, "
                f"{record['chunks_per_second']:.2f} chunks/s, {record['tokens_per_second']:.0f} tokens/s; "
                f"done {done_rows}/{len(rows)} chunks, {100 * done_tokens / max(total_tokens, 1):.1f}% tokens; "
                f"eta {finish.isoformat(timespec='minutes')} ({finish.astimezone().isoformat(timespec='minutes')}); "
                f"peak_rss {rss:.2f} GiB; system_cpu {record['system_cpu_percent']}%")
    except EncodingStopped:
        log("stop file found; stopped between batches; rerun the same command to resume")
        raise
    except BaseException as error:
        log(f"failed: {error!r}; completed shards are kept; rerun the same command to resume")
        raise
    manifest = _consolidate(out_dir, rows, config, records, n_shards)
    log(f"complete: {manifest['rows']} chunks x {manifest['dimension']}; "
        f"chunk_vectors.npy sha256 {manifest['files']['chunk_vectors.npy']['sha256']}")
    return manifest


def _consolidate(out_dir: Path, rows: Sequence[dict[str, Any]], config: dict[str, Any],
                 records: dict[int, dict[str, Any]], n_shards: int) -> dict[str, Any]:
    """Join verified shards into ``chunk_vectors.npy`` with catalogue-order indexes and the final manifest."""
    if sorted(records) != list(range(n_shards)):
        raise ValueError("shard records do not cover the run")
    chunk_manifest = json.loads(Path(config["chunks"]["manifest_path"]).read_text(encoding="utf-8"))
    articles = Path(chunk_manifest["catalogue"]["path"])
    if sha256_file(articles) != chunk_manifest["catalogue"]["sha256"]:
        raise ValueError("catalogue bytes differ from the chunk manifest binding")
    candidate_ids = load_candidate_ids(articles)
    position = {cid: i for i, cid in enumerate(candidate_ids)}
    unknown = sorted({r["candidate_id"] for r in rows} - position.keys())
    if unknown:
        raise ValueError(f"{len(unknown)} chunk candidates are not in the catalogue, e.g. {unknown[0]}")
    article_index = np.asarray([position[r["candidate_id"]] for r in rows], dtype=np.int32)
    ordinal = np.asarray([r["chunk_index"] for r in rows], dtype=np.int32)
    first = np.r_[True, article_index[1:] != article_index[:-1]]
    if (np.diff(article_index) < 0).any() or ordinal[first].any() or int(first.sum()) != len(candidate_ids):
        raise ValueError("chunks must follow catalogue order, start at chunk 0 and cover every article")
    shards = [records[i] for i in range(n_shards)]
    dimension = shards[0]["dimension"]
    partial = out_dir / "chunk_vectors.partial.npy"
    matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=(len(rows), dimension))
    for record in shards:
        path = out_dir / "shards" / f"shard_{record['shard']:05d}.npy"
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"shard changed after encoding: {path}")
        matrix[record["begin"]:record["end"]] = np.load(path)
    matrix.flush()
    del matrix
    os.replace(partial, out_dir / "chunk_vectors.npy")
    np.save(out_dir / "article_index.npy", article_index)
    np.save(out_dir / "chunk_ordinal.npy", ordinal)
    write_json(out_dir / "ids.json", {"candidate_ids": candidate_ids, "chunk_ids": [r["chunk_id"] for r in rows]})
    run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
    sessions = list(read_jsonl(out_dir / "sessions.jsonl"))
    encode_seconds = sum(r["seconds"] for r in shards)
    tokens = sum(r["tokens"] for r in shards)
    shapes = {"chunk_vectors.npy": [len(rows), dimension], "article_index.npy": [len(rows)], "chunk_ordinal.npy": [len(rows)]}
    manifest = {
        **config,
        "status": "complete",
        "created_utc": run_config["created_utc"],
        "completed_utc": _utc(),
        "catalogue": {"path": str(articles), "sha256": chunk_manifest["catalogue"]["sha256"], "articles": len(candidate_ids)},
        "rows": len(rows),
        "dimension": dimension,
        "files": {name: {"sha256": sha256_file(out_dir / name), "shape": shapes.get(name)} for name in OUTPUT_FILES},
        "index": {"article_index.npy": "int32 position of each chunk's candidate in catalogue (articles.jsonl) order",
                  "chunk_ordinal.npy": "int32 chunk_index within its article", "ids.json": "candidate_ids, chunk_ids"},
        "shards": shards,
        "timing": {"encode_seconds": encode_seconds, "sessions": len(sessions),
                   "chunks_per_second": len(rows) / encode_seconds if encode_seconds else None,
                   "tokens_per_second": tokens / encode_seconds if encode_seconds else None},
        "tokens": {"encoder_input_tokens": tokens, "truncated_inputs": sum(r["truncated_inputs"] for r in shards)},
        "resources": {"peak_rss_bytes": max((r["peak_rss_bytes"] or 0) for r in shards),
                      "threads": sorted({r["threads"] for r in shards})},
        "sessions": sessions,
        "evidence": run_config.get("evidence", {}),
        "code": code_hashes(),
        "downloads": 0,
        "network_calls": 0,
    }
    write_json(out_dir / "manifest.json", manifest)
    return manifest


@dataclass
class ChunkMatrix:
    """Chunk vectors with each chunk's article position in ``articles.jsonl`` order."""

    vectors: np.ndarray
    article_index: np.ndarray
    chunk_ordinal: np.ndarray
    chunk_ids: list[str]
    candidate_ids: list[str]
    manifest: dict[str, Any]

    @property
    def n_articles(self) -> int:
        return len(self.candidate_ids)


def load_chunk_matrix(out_dir: Path, *, verify: bool = True) -> ChunkMatrix:
    """Memory-mapped chunk vectors and an index aligned to the candidate order of articles.jsonl."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete chunk-vector run: {out_dir}")
    if verify:
        for name, meta in manifest["files"].items():
            if sha256_file(out_dir / name) != meta["sha256"]:
                raise ValueError(f"output changed after completion: {name}")
    ids = json.loads((out_dir / "ids.json").read_text(encoding="utf-8"))
    matrix = ChunkMatrix(np.load(out_dir / "chunk_vectors.npy", mmap_mode="r"), np.load(out_dir / "article_index.npy"),
                         np.load(out_dir / "chunk_ordinal.npy"), ids["chunk_ids"], ids["candidate_ids"], manifest)
    if matrix.vectors.shape != (manifest["rows"], manifest["dimension"]) or len(matrix.article_index) != manifest["rows"] \
            or len(matrix.chunk_ids) != manifest["rows"]:
        raise ValueError("chunk-vector outputs disagree with their manifest")
    return matrix


def article_scores(query_vecs: np.ndarray, chunk_vecs: np.ndarray, chunk_article_index: np.ndarray, n_articles: int,
                   mode: str = "maxp", batch: int = 256) -> np.ndarray:
    """Article scores float32 [n_queries, n_articles] from chunk dot products, in query blocks of ``batch``.

    ``maxp`` takes each article's best chunk, ``firstp`` its first chunk in array
    order (chunk 0 in chunks.jsonl order) and ``meanp`` the mean over its chunks.
    Every article in ``range(n_articles)`` must own at least one chunk.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if batch < 1:
        raise ValueError("batch must be positive")
    queries = np.asarray(query_vecs, dtype=np.float32)
    owners = np.asarray(chunk_article_index, dtype=np.int64)
    if queries.ndim != 2 or chunk_vecs.ndim != 2 or queries.shape[1] != chunk_vecs.shape[1] \
            or owners.shape != (chunk_vecs.shape[0],):
        raise ValueError("expected queries [q, d], chunks [c, d] and one article index per chunk")
    if len(owners) and (owners.min() < 0 or owners.max() >= n_articles):
        raise ValueError("article index outside range(n_articles)")
    order = np.argsort(owners, kind="stable")
    sorted_owners = owners[order]
    starts = np.flatnonzero(np.r_[True, sorted_owners[1:] != sorted_owners[:-1]]) if len(owners) else np.zeros(0, np.int64)
    if len(starts) != n_articles:
        raise ValueError("every article needs at least one chunk")
    chunks = np.asarray(chunk_vecs) if np.array_equal(order, np.arange(len(owners))) else np.asarray(chunk_vecs)[order]
    chunks = chunks.astype(np.float32, copy=False)
    if mode == "firstp":
        chunks = np.ascontiguousarray(chunks[starts])
    counts = np.diff(np.r_[starts, len(owners)]).astype(np.float64)
    out = np.empty((len(queries), n_articles), dtype=np.float32)
    for begin in range(0, len(queries), batch):
        block = queries[begin:begin + batch] @ chunks.T
        if mode == "firstp":
            out[begin:begin + batch] = block
        elif mode == "maxp":
            out[begin:begin + batch] = np.maximum.reduceat(block, starts, axis=1)
        else:
            out[begin:begin + batch] = np.add.reduceat(block, starts, axis=1, dtype=np.float64) / counts
    return out


def encode_queries(release_dir: Path, out_path: Path, *, threads: int = 16, batch_size: int = 8,
                   encoder: Any = None) -> dict[str, Any]:
    """Encode ``queries.jsonl`` texts in release order ('query: ' prefix, truncation at 512, as October).

    Writes ``out_path`` (.npy, float32 [n, 384]) and a receipt beside it (.json); neither may exist.
    """
    release_dir, out_path = Path(release_dir), Path(out_path)
    receipt_path = out_path.with_suffix(".json")
    if out_path.suffix != ".npy":
        raise ValueError("out_path must end with .npy")
    if out_path.exists() or receipt_path.exists():
        raise FileExistsError(f"query vectors exist; choose a new path: {out_path}")
    queries_path = release_dir / "queries.jsonl"
    rows = list(read_jsonl(queries_path))
    for row in rows:
        if "text_sha256" in row and sha256_text(row["text"]) != row["text_sha256"]:
            raise ValueError(f"query text differs from its text_sha256: {row['query_id']}")
    started = time.perf_counter()
    values, cost = _encode([r["text"] for r in rows], kind="query", batch_size=batch_size, threads=threads, encoder=encoder)
    seconds = time.perf_counter() - started
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, values)
    release_manifest = release_dir / "manifest.json"
    receipt = {
        "format_version": FORMAT_VERSION,
        "kind": "facts_to_law_v2_query_vectors",
        "created_utc": _utc(),
        "release": {"dir": str(release_dir),
                    "manifest_sha256": sha256_file(release_manifest) if release_manifest.is_file() else None},
        "queries": {"path": str(queries_path), "sha256": sha256_file(queries_path), "rows": len(rows)},
        "query_ids": [r["query_id"] for r in rows],
        "text_sha256": [sha256_text(r["text"]) for r in rows],
        "encoder": _encoder_identity(encoder),
        "recipe": {**RECIPE, "prefix": QUERY_PREFIX, "batch_size": batch_size, "order": "release order"},
        "threads": threads,
        "seconds": seconds,
        "cost": {key: value for key, value in cost.items() if key != "token_lengths"},
        "array": {"path": str(out_path), "sha256": sha256_file(out_path), "shape": list(values.shape), "dtype": "float32"},
        "code": code_hashes(),
        "downloads": 0,
        "network_calls": 0,
    }
    write_json(receipt_path, receipt)
    return receipt


def _october(october_dir: Path) -> tuple[dict[str, Any], np.ndarray]:
    """October article-vector receipt and vectors, after recipe and byte checks."""
    receipt = json.loads((october_dir / "articles.json").read_text(encoding="utf-8"))
    signature = receipt["signature"]
    if (signature["model_files"] != {name: digest for name, (_, digest) in MODEL_FILES.items()}
            or signature["revision"] != MODEL_REVISION or signature["prefix"] != PASSAGE_PREFIX
            or signature["max_tokens"] != MAX_TOKENS or signature["pooling"] != RECIPE["pooling"]
            or signature["dtype"] != "float32"):
        raise ValueError("October article vectors use a different encoder recipe")
    if sha256_file(october_dir / "articles.npy") != receipt["array_sha256"]:
        raise ValueError("October article vector bytes changed")
    return receipt, np.load(october_dir / "articles.npy", mmap_mode="r")


def _unchanged_bodies(articles_jsonl: Path, receipt: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Catalogue articles whose full text hash equals the October input hash for the same candidate."""
    signature = receipt["signature"]
    october_row = {cid: i for i, cid in enumerate(signature["ids"])}
    truncated = set(receipt["cost"]["truncated_positions"])
    unchanged, total = [], 0
    for position, row in enumerate(read_jsonl(articles_jsonl)):
        total += 1
        index = october_row.get(row["candidate_id"])
        if index is not None and signature["text_sha256"][index] == sha256_text(row["instrument_article_body"]):
            unchanged.append({"candidate_id": row["candidate_id"], "position": position, "october_row": index,
                              "october_truncated": index in truncated, "text": row["instrument_article_body"]})
    return unchanged, total


def _compare(vectors: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    diff = np.abs(np.asarray(vectors, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    cosine = (np.asarray(vectors, dtype=np.float64) * np.asarray(reference, dtype=np.float64)).sum(axis=1)
    return {"rows": len(vectors), "max_abs_diff": float(diff.max()) if len(diff) else None,
            "mean_abs_diff": float(diff.mean()) if len(diff) else None,
            "min_cosine": float(cosine.min()) if len(cosine) else None}


def firstp_parity(chunks_dir: Path, *, october_dir: Path = OCTOBER_C01, n_truncated: int = 150, n_whole: int = 150,
                  salt: str = PARITY_SALT, threads: int = 16, batch_size: int = 16, encoder: Any = None,
                  tokenizer: Any = None) -> dict[str, Any]:
    """Encode chunk 0 of sampled unchanged-body articles and compare with the saved October vectors."""
    chunks_dir, october_dir = Path(chunks_dir), Path(october_dir)
    chunk_manifest = json.loads((chunks_dir / "manifest.json").read_text(encoding="utf-8"))
    receipt, october_vectors = _october(october_dir)
    unchanged, total = _unchanged_bodies(Path(chunk_manifest["catalogue"]["path"]), receipt)
    key = lambda a: sha256_text(f"{salt}:{a['candidate_id']}")
    picked = (sorted((a for a in unchanged if a["october_truncated"]), key=key)[:n_truncated]
              + sorted((a for a in unchanged if not a["october_truncated"]), key=key)[:n_whole])
    wanted = {a["candidate_id"] for a in picked}
    first = {r["candidate_id"]: r for r in read_jsonl(chunks_dir / "chunks.jsonl") if r["chunk_index"] == 0 and r["candidate_id"] in wanted}
    texts = [first[a["candidate_id"]]["text"] for a in picked]
    tokens = np.asarray([first[a["candidate_id"]]["n_tokens"] for a in picked], dtype=np.int64)
    if encoder is None:
        configure_runtime(threads)
    active = encoder if encoder is not None else get_encoder()
    tokenizer = tokenizer if tokenizer is not None else getattr(active, "tokenizer", None)
    token_check = None
    if tokenizer is not None:
        view = TokenView(tokenizer, max_tokens=MAX_TOKENS)
        equal = sum(view.encoder_ids([t])[0] == view.encoder_ids([a["text"]])[0] for t, a in zip(texts, picked))
        token_check = {"checked": len(picked), "equal": int(equal)}
    started = time.perf_counter()
    vectors, _ = _encode_sorted(active, texts, tokens, batch_size=batch_size)
    seconds = time.perf_counter() - started
    reference = october_vectors[[a["october_row"] for a in picked]]
    groups = np.asarray([a["october_truncated"] for a in picked])
    comparison = _compare(vectors, reference)
    worst = np.argsort(-np.abs(vectors - reference).max(axis=1))[:5]
    return {
        "kind": "facts_to_law_v2_firstp_parity",
        "created_utc": _utc(),
        "chunks": {"dir": str(chunks_dir), "manifest_sha256": sha256_file(chunks_dir / "manifest.json"),
                   "chunks_sha256": sha256_file(chunks_dir / "chunks.jsonl")},
        "october": {"dir": str(october_dir), "articles_npy_sha256": receipt["array_sha256"],
                    "receipt_sha256": sha256_file(october_dir / "articles.json"),
                    "recipe": {k: receipt["signature"][k] for k in ("prefix", "max_tokens", "pooling", "dtype", "attention", "batch_size", "revision")}},
        "eligible": {"catalogue_articles": total, "unchanged_body": len(unchanged), "changed_or_absent": total - len(unchanged),
                     "unchanged_october_truncated": sum(a["october_truncated"] for a in unchanged)},
        "sample": {"salt": salt, "rule": "sha256(salt:candidate_id) ascending within October-truncated and untruncated",
                   "truncated": int(groups.sum()), "whole": int((~groups).sum()),
                   "candidate_ids": [a["candidate_id"] for a in picked]},
        "token_ids_equal": token_check,
        **comparison,
        "by_group": {"october_truncated": _compare(vectors[groups], reference[groups]),
                     "untruncated": _compare(vectors[~groups], reference[~groups])},
        "worst": [{"candidate_id": picked[i]["candidate_id"], "max_abs_diff": float(np.abs(vectors[i] - reference[i]).max())} for i in worst],
        "threads": threads,
        "batch_size": batch_size,
        "encode_seconds": seconds,
        "encoder": _encoder_identity(encoder),
        "code": code_hashes(),
    }


def firstp_parity_full(vectors_dir: Path, *, october_dir: Path = OCTOBER_C01) -> dict[str, Any]:
    """Compare every chunk-0 vector of a completed run with the October vector of the same unchanged body."""
    matrix = load_chunk_matrix(vectors_dir)
    receipt, october_vectors = _october(Path(october_dir))
    unchanged, total = _unchanged_bodies(Path(matrix.manifest["catalogue"]["path"]), receipt)
    first = np.flatnonzero(matrix.chunk_ordinal == 0)
    chunk0 = first[np.argsort(matrix.article_index[first], kind="stable")]
    vectors = np.asarray(matrix.vectors[chunk0[[a["position"] for a in unchanged]]])
    reference = np.asarray(october_vectors[[a["october_row"] for a in unchanged]])
    groups = np.asarray([a["october_truncated"] for a in unchanged])
    worst = np.argsort(-np.abs(vectors - reference).max(axis=1))[:5]
    return {
        "kind": "facts_to_law_v2_firstp_parity_full",
        "created_utc": _utc(),
        "vectors": {"dir": str(vectors_dir), "chunk_vectors_sha256": matrix.manifest["files"]["chunk_vectors.npy"]["sha256"]},
        "october": {"dir": str(october_dir), "articles_npy_sha256": receipt["array_sha256"]},
        "eligible": {"catalogue_articles": total, "unchanged_body": len(unchanged), "changed_or_absent": total - len(unchanged)},
        **_compare(vectors, reference),
        "by_group": {"october_truncated": _compare(vectors[groups], reference[groups]),
                     "untruncated": _compare(vectors[~groups], reference[~groups])},
        "worst": [{"candidate_id": unchanged[i]["candidate_id"], "max_abs_diff": float(np.abs(vectors[i] - reference[i]).max())} for i in worst],
        "code": code_hashes(),
    }


def smoke_throughput(chunks_jsonl: Path, *, n: int = 300, seed: int = SMOKE_SEED, threads: int = 16,
                     batch_sizes: Sequence[int] = (16,), encoder: Any = None) -> dict[str, Any]:
    """Encode a seeded random chunk sample once per batch size; throughput and full-run estimates."""
    chunks_jsonl = Path(chunks_jsonl)
    rows = list(read_jsonl(chunks_jsonl))
    pick = np.sort(np.random.default_rng(seed).choice(len(rows), size=min(n, len(rows)), replace=False))
    texts = [rows[i]["text"] for i in pick]
    tokens = np.asarray([rows[i]["n_tokens"] for i in pick], dtype=np.int64)
    runtime = configure_runtime(threads) if encoder is None else None
    active = encoder if encoder is not None else get_encoder()
    warm = np.argsort(tokens, kind="stable")[:8]
    _encode_sorted(active, [texts[i] for i in warm], tokens[warm], batch_size=8)
    results = []
    for batch_size in batch_sizes:
        _cpu_percent()
        t0 = time.perf_counter()
        _encode_sorted(active, texts, tokens, batch_size=batch_size)
        seconds = time.perf_counter() - t0
        results.append({"batch_size": batch_size, "seconds": seconds, "chunks_per_second": len(texts) / seconds,
                        "tokens_per_second": int(tokens.sum()) / seconds, "system_cpu_percent": _cpu_percent()})
    best = max(results, key=lambda r: r["tokens_per_second"])
    total_tokens = sum(r["n_tokens"] for r in rows)
    return {
        "kind": "facts_to_law_v2_dense_smoke",
        "created_utc": _utc(),
        "chunks": {"path": str(chunks_jsonl), "sha256": sha256_file(chunks_jsonl), "rows": len(rows), "tokens": total_tokens},
        "sample": {"n": len(pick), "seed": seed, "tokens": int(tokens.sum()), "positions": pick.tolist()},
        "threads": threads,
        "runtime": runtime,
        "results": results,
        "estimate": {"batch_size": best["batch_size"], "seconds_by_chunks": len(rows) / best["chunks_per_second"],
                     "seconds_by_tokens": total_tokens / best["tokens_per_second"]},
        "peak_rss_bytes": _peak_rss(),
        "encoder": _encoder_identity(encoder),
        "code": code_hashes(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -B -m uol.facts_to_law.dense", description="Facts-to-law v2 dense encoding (WP2).")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("chunks", help="build a chunk view with the installed E5-small tokenizer")
    p.add_argument("--articles", type=Path, default=V10_ARTICLES)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--cap", type=int, default=32)
    p = sub.add_parser("check", help="FirstP parity with October vectors and a throughput smoke test")
    p.add_argument("--chunks", type=Path, required=True, help="chunk view directory")
    p.add_argument("--out", type=Path, required=True, help="new directory for firstp_parity.json and smoke_throughput.json")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--smoke-batch-sizes", default="16")
    p.add_argument("--smoke-n", type=int, default=300)
    p.add_argument("--n-truncated", type=int, default=150)
    p.add_argument("--n-whole", type=int, default=150)
    p = sub.add_parser("encode-chunks", help="encode a chunk view into resumable shards")
    p.add_argument("--chunks", type=Path, required=True, help="chunks.jsonl")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--log", type=Path, default=None)
    p.add_argument("--evidence", type=Path, action="append", default=[])
    p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("parity-full", help="compare all chunk-0 vectors of a completed run with October vectors")
    p.add_argument("--vectors", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("encode-queries", help="encode a release's queries.jsonl in release order")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "chunks":
        manifest_path = args.out / "manifest.json"
        catalogue.build_chunks(args.articles, load_tokenizer(), args.max_tokens, args.overlap, args.cap, out_dir=args.out)
        counts = json.loads(manifest_path.read_text(encoding="utf-8"))["counts"]
        print(json.dumps({k: counts[k] for k in ("articles", "chunks", "n_capped", "chunk0_modes", "encoder_input_tokens")}))
    elif args.command == "check":
        args.out.mkdir(parents=True, exist_ok=False)
        configure_runtime(args.threads)
        active = get_encoder()
        parity = firstp_parity(args.chunks, n_truncated=args.n_truncated, n_whole=args.n_whole, threads=args.threads,
                               batch_size=args.batch_size, encoder=active)
        write_json(args.out / "firstp_parity.json", parity)
        print(json.dumps({k: parity[k] for k in ("max_abs_diff", "min_cosine", "token_ids_equal")}))
        sizes = [int(s) for s in args.smoke_batch_sizes.split(",")]
        smoke = smoke_throughput(args.chunks / "chunks.jsonl", n=args.smoke_n, threads=args.threads, batch_sizes=sizes, encoder=active)
        smoke["runtime"] = {"threads": args.threads, "configured_by": "check command"}
        write_json(args.out / "smoke_throughput.json", smoke)
        print(json.dumps({"results": smoke["results"], "estimate": smoke["estimate"]}))
    elif args.command == "encode-chunks":
        try:
            encode_chunks(args.chunks, args.out, args.shard_size, args.threads, not args.no_resume,
                          batch_size=args.batch_size, log_path=args.log, evidence=args.evidence)
        except EncodingStopped:
            return 3
    elif args.command == "parity-full":
        if args.out.exists():
            raise FileExistsError(f"report exists: {args.out}")
        report = firstp_parity_full(args.vectors)
        write_json(args.out, report)
        print(json.dumps({k: report[k] for k in ("eligible", "max_abs_diff", "min_cosine")}))
    elif args.command == "encode-queries":
        receipt = encode_queries(args.release, args.out, threads=args.threads, batch_size=args.batch_size)
        print(json.dumps({"rows": receipt["queries"]["rows"], "array": receipt["array"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
