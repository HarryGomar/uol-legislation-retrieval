"""bge-m3 multi-vector (ColBERT-style) late interaction as a label-free reranking feature (agent EP-delta).

No network was available, so the head is implemented from the BGE-M3 / FlagEmbedding architecture.
Assumptions, all checked where a check is possible:

- ``colbert_linear.pt`` is ``Linear(1024, 1024)``: ``weight`` (1024, 1024) and ``bias`` (1024,), stored fp16, used in fp32.
- Token vectors are ``colbert_linear(last_hidden_state[:, 1:])``: the CLS position is dropped, the final ``</s>``
  position is kept (FlagEmbedding keeps ``colbert_vecs[:tokens_num - 1]``), padding is removed, and each vector is
  L2-normalised (BGE-M3 ``normlized=True``). The vectors are stored as float16 after the fp32 normalisation.
- ``score(q, d) = (1 / |q|) * sum over query vectors of max over document vectors of the dot product`` where
  ``|q|`` counts every kept query vector, ``</s>`` included (FlagEmbedding ``colbert_score``).
- No instruction prefix. Queries are truncated at 512 tokens. Articles are chunked exactly as in
  ``uol.facts_to_law.bge_gpu`` (<= 1,024 tokens, title repeat, 64 overlap, <= 32 chunks); the CLS vector of the same
  forward pass must reproduce the frozen ``bge_m3_chunks_v1`` vector (checked on every batch).
- Multi-chunk articles: the primary ``score`` is MaxP, the maximum over chunks of the chunk score. The variant
  ``*_union`` lets every query vector pick its best document vector across all chunks of the article.
- Query-vector weightings (extras, all label-free): ``uniform`` (the standard definition), ``nomask`` (drops the
  ``[CITA_OCULTA]`` placeholder tokens) and ``idf`` (token IDF over all chunks, placeholder dropped).

Pool: per query the union of the top-100 by BM25 (``uol.facts_to_law.lexical``, k1 1.2, b .75, computed here) and the
top-100 by frozen bge-m3 MaxP. Nothing in this module reads a label to build a feature; labels are read only by the
dev diagnostics (``eval``), never for a selection that feeds a score.

Stages (each restartable; GPU stages hold ``common.gpu_lock`` per job)::

    pool -> chunks -> alloc -> encode-docs (shards) -> encode-queries -> score -> assemble -> eval
    long-encode -> long-scores            (optional dense_bge_long)
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np

from uol.embedding_push import common

OWNER = "EP-delta"
OUT = common.RUN / "colbert"
DOCS = OUT / "docs"
QUERIES = OUT / "queries"
LONG = OUT / "long"
PAIRS_DIR = common.SCORES / "colbert_bge"
LONG_ID = "dense_bge_long"
FROZEN = common.DERIVED / "bge_m3_chunks_v1"
TOP_K = 100
BM25_K1, BM25_B = 1.2, 0.75
DOC_MAX_TOKENS, QUERY_MAX_TOKENS, OVERLAP, CAP = 1024, 512, 64, 32
LONG_MAX_TOKENS = 8192
DIM = 1024
PLACEHOLDER = re.compile(r"\[CITA_OCULTA\]")
WEIGHTINGS = ("uniform", "nomask", "idf")
AGGREGATIONS = ("maxp", "union")
VARIANTS = tuple(f"{w}_{a}" for w in WEIGHTINGS for a in AGGREGATIONS)  # uniform_maxp is the primary score
PRIMARY = "uniform_maxp"


# ----------------------------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------------------------

def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    line = f"{utc()} {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "progress.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def record_stage(stage: str, seconds: float, **extra: Any) -> None:
    """Append one stage record (wall time, device, notes) to stages.jsonl for the manifest."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "stages.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"stage": stage, "seconds": round(seconds, 1), "finished_utc": utc(), **extra}) + "\n")


def search_log(entry: dict[str, Any]) -> None:
    """Every configuration tried, with its dev metrics (PROTOCOL / brief rule 3)."""
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "search_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"utc": utc(), **entry}, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def free_disk_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


# ----------------------------------------------------------------------------------------------------------------
# the late-interaction kernel (torch; CPU or GPU)
# ----------------------------------------------------------------------------------------------------------------

def late_interaction(q, d, chunk_lengths, article_chunks, weights):
    """Late interaction of one query against a block of whole articles.

    ``q`` is [Lq, H] and ``d`` is [T, H], both with unit rows (float32). ``d`` holds consecutive chunks of consecutive
    articles: ``chunk_lengths`` [C] gives the vectors per chunk and ``article_chunks`` [A] the chunks per article.
    ``weights`` is [Lq, V]; each column is a non-negative weighting of the query vectors (columns need positive sums).

    Returns ``maxp`` [A, V] (maximum over an article's chunks of the weighted mean over query vectors of the best
    in-chunk dot product) and ``union`` [A, V] (the best dot product over all chunks of the article, then the same
    weighted mean). A single-chunk article gives identical values.
    """
    import torch

    sim = d @ q.T                                                       # [T, Lq]
    per_chunk = torch.segment_reduce(sim, "max", lengths=chunk_lengths, axis=0)      # [C, Lq]
    norm = weights / weights.sum(dim=0, keepdim=True)
    chunk_scores = per_chunk @ norm                                     # [C, V]
    maxp = torch.segment_reduce(chunk_scores, "max", lengths=article_chunks, axis=0)  # [A, V]
    per_article = torch.segment_reduce(per_chunk, "max", lengths=article_chunks, axis=0)  # [A, Lq]
    union = per_article @ norm                                          # [A, V]
    return maxp, union


def late_interaction_reference(q: np.ndarray, d_chunks: Sequence[np.ndarray], weights: np.ndarray | None = None) -> tuple[float, float]:
    """Plain numpy definition of the same two scores for one query and one article (used by the tests)."""
    q = np.asarray(q, dtype=np.float64)
    w = np.ones(len(q)) if weights is None else np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    best = [(q @ np.asarray(c, dtype=np.float64).T).max(axis=1) for c in d_chunks]    # per chunk, per query vector
    maxp = max(float(w @ b) for b in best)
    union = float(w @ np.max(np.stack(best), axis=0))
    return maxp, union


def query_weightings(token_ids: np.ndarray, placeholder: np.ndarray, idf: np.ndarray) -> np.ndarray:
    """[Lq, 3] weights in WEIGHTINGS order for one query's kept token vectors (CLS excluded, ``</s>`` included)."""
    keep = (~placeholder).astype(np.float32)
    uniform = np.ones(len(token_ids), dtype=np.float32)
    weighted = idf[token_ids].astype(np.float32) * keep
    if weighted.sum() <= 0:          # degenerate query (only placeholders / zero-idf tokens)
        weighted = keep if keep.sum() > 0 else uniform
    if keep.sum() <= 0:
        keep = uniform
    return np.stack([uniform, keep, weighted], axis=1)


def block_articles(arts: np.ndarray, art_tokens: np.ndarray, max_tokens: int) -> list[list[int]]:
    """Group whole articles into blocks of at most ``max_tokens`` vectors (one larger article forms its own block)."""
    blocks: list[list[int]] = []
    current: list[int] = []
    used = 0
    for position, article in enumerate(arts):
        size = int(art_tokens[article])
        if current and used + size > max_tokens:
            blocks.append(current)
            current, used = [], 0
        current.append(position)
        used += size
    if current:
        blocks.append(current)
    return blocks


# ----------------------------------------------------------------------------------------------------------------
# stage: pool (CPU)
# ----------------------------------------------------------------------------------------------------------------

def stage_pool(k: int = TOP_K, batch: int = 256) -> dict[str, Any]:
    """Per query: union of BM25 top-k and frozen bge-m3 MaxP top-k, with both raw scores for every pool pair."""
    from uol.facts_to_law import lexical

    t0 = time.time()
    data = common.release()
    n_q, n_c = len(data.queries), len(data.candidate_ids)
    found = sorted(p.name for p in common.SCORES.glob("*")) if common.SCORES.exists() else []
    qv = np.load(FROZEN / "query_vectors.npy")
    cv = np.load(FROZEN / "chunk_vectors.npy")
    owner = common.chunk_article_index(FROZEN / "units.jsonl")
    log(f"pool: fitting BM25 k1={BM25_K1} b={BM25_B} on {n_c} candidates (scores dir had: {found})")
    index = lexical.LexicalIndex(data.candidate_texts, "bm25", {"k1": BM25_K1, "b": BM25_B})
    texts = [q.text for q in data.queries]
    bm_top = np.empty((n_q, k), np.int32)
    de_top = np.empty((n_q, k), np.int32)
    qs, cs, flags, bms, des = [], [], [], [], []
    for start in range(0, n_q, batch):
        stop = min(start + batch, n_q)
        _, dense = next(iter(common.maxp_scores(qv[start:stop], cv, owner, n_c, batch=batch)))
        bm25 = index.score(texts[start:stop])
        for r in range(stop - start):
            bt = np.argsort(-bm25[r], kind="stable")[:k]
            dt = np.argsort(-dense[r], kind="stable")[:k]
            bm_top[start + r], de_top[start + r] = bt, dt
            union = np.union1d(bt, dt)
            flag = np.isin(union, bt).astype(np.int8) | (np.isin(union, dt).astype(np.int8) << 1)
            qs.append(np.full(len(union), start + r, np.int32))
            cs.append(union.astype(np.int32))
            flags.append(flag)
            bms.append(bm25[r, union])
            des.append(dense[r, union])
        if (start // batch) % 4 == 0:
            log(f"pool: {stop}/{n_q} queries, {time.time() - t0:.0f}s")
    qidx, cidx = np.concatenate(qs), np.concatenate(cs)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "pool.npz", qidx=qidx, cidx=cidx, flag=np.concatenate(flags), bm25=np.concatenate(bms),
             dense=np.concatenate(des), bm25_top=bm_top, dense_top=de_top)
    sizes = np.bincount(qidx, minlength=n_q)
    stats = {"top_k": k, "queries": n_q, "pairs": int(len(qidx)), "unique_articles": int(len(np.unique(cidx))),
             "pool_size_mean": float(sizes.mean()), "pool_size_min": int(sizes.min()), "pool_size_max": int(sizes.max()),
             "bm25": {"kind": "uol.facts_to_law.lexical.LexicalIndex bm25", "k1": BM25_K1, "b": BM25_B,
                      "why": "no complete scores/bm25* or scores/L_star existed when the pool was built; the brief's fallback",
                      "scores_dir_listing": found},
             "dense": "frozen bge_m3_chunks_v1 MaxP cosine (common.maxp_scores)", "tie_rule": "score desc, candidate index asc"}
    stats["dev_recall"] = pool_recall(qidx, cidx, bm_top, de_top)
    (OUT / "pool_stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    record_stage("pool", time.time() - t0, device="cpu")
    log(f"pool: done pairs={stats['pairs']} unique_articles={stats['unique_articles']} mean={stats['pool_size_mean']:.1f}")
    return stats


def pool_recall(qidx: np.ndarray, cidx: np.ndarray, bm_top: np.ndarray, de_top: np.ndarray) -> dict[str, Any]:
    """Diagnostic only (dev rows, P and P_spec): positives inside BM25-100, dense-100 and the union pool."""
    data = common.release()
    n_q = len(data.queries)
    ptr = np.searchsorted(qidx, np.arange(n_q + 1))
    out: dict[str, Any] = {}
    for name in ("P_spec", "P"):
        rows = common.population_rows(name, common.dev_rows())
        positives = data.specific_idx if name == "P_spec" else data.positive_idx
        agg = {"bm25_top": [], "dense_top": [], "union": []}
        anyhit = {key: [] for key in agg}
        for r in rows:
            pos = np.asarray(positives[int(r)], dtype=np.int64)
            sets = {"bm25_top": bm_top[r], "dense_top": de_top[r], "union": cidx[ptr[r]:ptr[r + 1]]}
            for key, members in sets.items():
                hit = np.isin(pos, members)
                agg[key].append(hit.mean())
                anyhit[key].append(hit.any())
        out[name] = {"rows": int(len(rows)),
                     "mean_recall": {key: round(float(np.mean(v)), 4) for key, v in agg.items()},
                     "share_with_any_positive": {key: round(float(np.mean(v)), 4) for key, v in anyhit.items()}}
    return out


# ----------------------------------------------------------------------------------------------------------------
# stage: chunks (CPU; tokenizer only)
# ----------------------------------------------------------------------------------------------------------------

def stage_chunks() -> dict[str, Any]:
    """Chunk the pool articles exactly as the frozen run did, verify against units.jsonl, and measure token counts."""
    from transformers import AutoTokenizer

    from uol.facts_to_law import bge_gpu

    t0 = time.time()
    data = common.release()
    pool = np.load(OUT / "pool.npz")
    pool_articles = set(np.unique(pool["cidx"]).tolist())
    tokenizer = AutoTokenizer.from_pretrained(common.BGE_DIR, local_files_only=True)
    articles_path = Path(data.candidate_binding["articles_path"])
    receipt = json.loads((FROZEN / "receipt.json").read_text(encoding="utf-8"))
    if common.sha256_file(articles_path) != receipt["inputs"]["articles"]["sha256"]:
        raise ValueError("the bound articles differ from the frozen bge-m3 run's input")
    rows = bge_gpu.read_jsonl(articles_path)
    units = read_jsonl(FROZEN / "units.jsonl")
    first_unit: dict[str, int] = {}
    n_units: dict[str, int] = {}
    for u, rec in enumerate(units):
        cid = rec["candidate_id"]
        if cid not in first_unit:
            first_unit[cid] = u
        if rec["chunk_index"] != n_units.get(cid, 0):
            raise ValueError("frozen units are not consecutive per article")
        n_units[cid] = n_units.get(cid, 0) + 1
    keep_unit, keep_cand, keep_ci, keep_text = [], [], [], []
    all_chunks = mismatches = 0
    all_texts: list[str] = []
    for a, row in enumerate(rows):
        cid = row["candidate_id"]
        if cid != data.candidate_ids[a]:
            raise ValueError("catalogue order differs from the release candidate order")
        chunks = bge_gpu.chunk_article(row, tokenizer, DOC_MAX_TOKENS, OVERLAP, CAP)
        if first_unit[cid] != all_chunks:
            raise ValueError("frozen units are not in catalogue order")
        all_chunks += len(chunks)
        if len(chunks) != n_units[cid]:
            mismatches += 1
        all_texts.extend(chunks)
        if a in pool_articles:
            for ci, text in enumerate(chunks):
                keep_unit.append(first_unit[cid] + ci)
                keep_cand.append(a)
                keep_ci.append(ci)
                keep_text.append(text)
        if a % 5000 == 0:
            log(f"chunks: article {a}/{len(rows)} {time.time() - t0:.0f}s")
    if mismatches or all_chunks != len(units):
        raise ValueError(f"chunking does not reproduce the frozen units: {mismatches} articles differ, {all_chunks} vs {len(units)} chunks")
    # token ids of every chunk: lengths for the pool chunks, document frequencies for the IDF weighting
    vocab = len(tokenizer)
    df = np.zeros(max(vocab, 250002), dtype=np.int64)
    pool_len = {}
    pool_units = set(keep_unit)
    for start in range(0, len(all_texts), 2000):
        ids = tokenizer(all_texts[start:start + 2000], truncation=True, max_length=DOC_MAX_TOKENS)["input_ids"]
        for offset, one in enumerate(ids):
            np.add.at(df, np.unique(one), 1)
            if start + offset in pool_units:
                pool_len[start + offset] = len(one)
    order = np.argsort(np.asarray(keep_unit))
    unit = np.asarray(keep_unit, dtype=np.int64)[order]
    n_tok = np.asarray([pool_len[int(u)] for u in unit], dtype=np.int32)
    n_vec = n_tok - 1
    offset = np.concatenate([[0], np.cumsum(n_vec, dtype=np.int64)[:-1]])
    texts = [keep_text[i] for i in order]
    np.savez(OUT / "chunks.npz", unit=unit, cand=np.asarray(keep_cand, dtype=np.int32)[order],
             ci=np.asarray(keep_ci, dtype=np.int32)[order], n_tok=n_tok, n_vec=n_vec.astype(np.int32), offset=offset)
    with (OUT / "chunk_texts.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for text in texts:
            handle.write(json.dumps(text, ensure_ascii=False) + "\n")
    n_chunks_total = len(all_texts)
    idf = np.log1p((n_chunks_total - df + 0.5) / (df + 0.5)).astype(np.float32)
    np.save(OUT / "idf.npy", idf)
    total = int(n_vec.sum())
    est = {"pool_articles": len(pool_articles), "pool_chunks": int(len(unit)), "all_chunks": n_chunks_total,
           "pool_token_vectors": total, "doc_store_gb": round(total * DIM * 2 / 1e9, 2),
           "free_disk_gb": round(free_disk_gb(OUT), 1), "mean_vectors_per_chunk": round(float(n_vec.mean()), 1),
           "max_vectors_per_chunk": int(n_vec.max()), "chunking_matches_frozen_units": True}
    (OUT / "chunk_stats.json").write_text(json.dumps(est, indent=1), encoding="utf-8")
    record_stage("chunks", time.time() - t0, device="cpu", **est)
    log(f"chunks: {est}")
    return est


def stage_queries_meta() -> dict[str, Any]:
    """Token ids, placeholder flags and token counts for the 5,980 queries (CPU); allocates nothing yet."""
    from transformers import AutoTokenizer

    t0 = time.time()
    data = common.release()
    tokenizer = AutoTokenizer.from_pretrained(common.BGE_DIR, local_files_only=True)
    texts = [q.text for q in data.queries]
    ids_all, flags_all, lengths = [], [], []
    for start in range(0, len(texts), 1000):
        batch = texts[start:start + 1000]
        enc = tokenizer(batch, truncation=True, max_length=QUERY_MAX_TOKENS, return_offsets_mapping=True)
        for text, ids, spans in zip(batch, enc["input_ids"], enc["offset_mapping"]):
            holes = [(m.start(), m.end()) for m in PLACEHOLDER.finditer(text)]
            kept_ids = np.asarray(ids[1:], dtype=np.int32)                        # CLS dropped, </s> kept
            hit = np.zeros(len(kept_ids), dtype=bool)
            for position, (lo, hi) in enumerate(spans[1:]):
                if hi > lo and any(lo < b and hi > a for a, b in holes):
                    hit[position] = True
            ids_all.append(kept_ids)
            flags_all.append(hit)
            lengths.append(len(kept_ids))
    n_vec = np.asarray(lengths, dtype=np.int32)
    offset = np.concatenate([[0], np.cumsum(n_vec, dtype=np.int64)[:-1]])
    np.savez(OUT / "queries_meta.npz", offset=offset, n_vec=n_vec, token_ids=np.concatenate(ids_all),
             placeholder=np.concatenate(flags_all))
    info = {"queries": len(texts), "token_vectors": int(n_vec.sum()), "store_gb": round(int(n_vec.sum()) * DIM * 2 / 1e9, 2),
            "placeholder_tokens": int(np.concatenate(flags_all).sum()), "queries_with_placeholder": int(sum(f.any() for f in flags_all)),
            "truncated_at_512": int((n_vec + 1 >= QUERY_MAX_TOKENS).sum())}
    record_stage("queries_meta", time.time() - t0, device="cpu", **info)
    log(f"queries meta: {info}")
    return info


def stage_alloc(shards: int = 8) -> dict[str, Any]:
    """Allocate the float16 stores and fix the shard plan (balanced by token vectors, article-aligned boundaries)."""
    t0 = time.time()
    chunks = np.load(OUT / "chunks.npz")
    qmeta = np.load(OUT / "queries_meta.npz")
    n_docs, n_queries = int(chunks["n_vec"].sum()), int(qmeta["n_vec"].sum())
    need = (n_docs + n_queries) * DIM * 2 / 1e9
    free = free_disk_gb(OUT)
    if free < need * 1.2 + 20:
        raise RuntimeError(f"not enough disk: need {need:.1f} GB, free {free:.1f} GB")
    DOCS.mkdir(parents=True, exist_ok=True)
    QUERIES.mkdir(parents=True, exist_ok=True)
    for path, rows in ((DOCS / "doc_tokens.npy", n_docs), (QUERIES / "query_tokens.npy", n_queries)):
        if path.exists():
            raise FileExistsError(path)
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(rows, DIM))
        del mm
    cum = np.cumsum(chunks["n_vec"], dtype=np.int64)
    targets = np.linspace(0, cum[-1], shards + 1)[1:-1]
    cand = chunks["cand"]
    cuts = []
    for t in targets:
        i = int(np.searchsorted(cum, t))
        while 0 < i < len(cand) and cand[i] == cand[i - 1]:     # keep an article's chunks in one shard
            i += 1
        cuts.append(min(i, len(cand)))
    bounds = [0, *sorted(set(cuts)), len(cand)]
    plan = [{"shard": s, "chunks": [int(bounds[s]), int(bounds[s + 1])],
             "vectors": int(chunks["n_vec"][bounds[s]:bounds[s + 1]].sum())} for s in range(len(bounds) - 1)]
    (DOCS / "plan.json").write_text(json.dumps({"shards": plan, "token_budget": 8192}, indent=1), encoding="utf-8")
    (DOCS / "progress.json").write_text(json.dumps({"done": {}}, indent=1), encoding="utf-8")
    (QUERIES / "progress.json").write_text(json.dumps({"done": False}, indent=1), encoding="utf-8")
    record_stage("alloc", time.time() - t0, device="cpu", doc_vectors=n_docs, query_vectors=n_queries,
                 stores_gb=round(need, 2), shards=len(plan))
    log(f"alloc: docs {n_docs} vectors, queries {n_queries} vectors, {need:.1f} GB, {len(plan)} shards")
    return {"plan": plan, "gb": need}


# ----------------------------------------------------------------------------------------------------------------
# the encoder (torch; the GPU stages construct it inside common.gpu_lock)
# ----------------------------------------------------------------------------------------------------------------

class Encoder:
    """bge-m3 backbone plus the ColBERT head; fp32. ``encode`` yields CLS and token vectors per input text."""

    def __init__(self, device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device = torch, device
        self.tokenizer = AutoTokenizer.from_pretrained(common.BGE_DIR, local_files_only=True)
        self.model = AutoModel.from_pretrained(common.BGE_DIR, local_files_only=True, dtype=torch.float32).to(device).eval()
        head = torch.load(common.BGE_DIR / "colbert_linear.pt", map_location="cpu", weights_only=True)
        if tuple(head["weight"].shape) != (DIM, DIM) or tuple(head["bias"].shape) != (DIM,):
            raise ValueError(f"unexpected colbert_linear shapes {[tuple(v.shape) for v in head.values()]}")
        self.weight = head["weight"].to(device=device, dtype=torch.float32)
        self.bias = head["bias"].to(device=device, dtype=torch.float32)

    def lengths(self, texts: Sequence[str], max_tokens: int) -> np.ndarray:
        ids = self.tokenizer(list(texts), add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"]
        return np.asarray([len(x) for x in ids], dtype=np.int64)

    def encode(self, texts: Sequence[str], max_tokens: int, token_budget: int, sink: Callable[[int, np.ndarray, np.ndarray], None],
               lengths: np.ndarray | None = None, tokens: bool = True) -> None:
        """Length-sorted dynamic batches under ``token_budget``; ``sink(i, cls[1024], token_vectors[n, 1024] fp16)``."""
        torch = self.torch
        F = torch.nn.functional
        lengths = self.lengths(texts, max_tokens) if lengths is None else lengths
        order = np.argsort(lengths, kind="stable")[::-1]
        i = 0
        with torch.inference_mode():
            while i < len(order):
                size = max(1, token_budget // int(lengths[order[i]]))
                batch = order[i:i + size]
                enc = self.tokenizer([texts[j] for j in batch], padding=True, truncation=True, max_length=max_tokens,
                                     return_tensors="pt").to(self.device)
                hidden = self.model(**enc).last_hidden_state
                cls = F.normalize(hidden[:, 0], dim=-1).cpu().numpy()
                counts = (enc["attention_mask"].sum(dim=1) - 1).cpu().numpy()
                if tokens:
                    vec = F.normalize(F.linear(hidden[:, 1:], self.weight, self.bias), dim=-1).half().cpu().numpy()
                for r, j in enumerate(batch):
                    if int(counts[r]) != int(lengths[j]) - 1:
                        raise RuntimeError(f"token count mismatch for item {int(j)}: {int(counts[r])} vs {int(lengths[j]) - 1}")
                    sink(int(j), cls[r], vec[r, :counts[r]] if tokens else None)
                i += size

    def free(self) -> None:
        del self.model
        self.torch.cuda.empty_cache()


@contextlib.contextmanager
def gpu_job(name: str) -> Iterator[float]:
    """The shared GPU lock around one job; yields the seconds spent waiting for it."""
    t = time.time()
    log(f"{name}: waiting for the GPU lock")
    with common.gpu_lock(owner=OWNER):
        waited = time.time() - t
        log(f"{name}: lock acquired after {waited:.0f}s")
        yield waited
    log(f"{name}: lock released")


def stage_encode_docs(shards: "int | Sequence[int]", device: str = "cuda", limit: int = 0) -> dict[int, dict[str, Any]]:
    """Pool chunks of one or more shards: token vectors into the float16 store; CLS checked against the frozen vectors.

    The listed shards form one job: one GPU-lock acquisition, one model load, progress saved after every shard.
    ``device="cpu"`` with ``limit`` is a smoke test only: it writes the first ``limit`` chunks of each shard and records nothing.
    """
    wanted = [int(shards)] if isinstance(shards, (int, np.integer)) else [int(x) for x in shards]
    chunks = {k: v for k, v in np.load(OUT / "chunks.npz").items()}
    plan = json.loads((DOCS / "plan.json").read_text(encoding="utf-8"))
    progress = json.loads((DOCS / "progress.json").read_text(encoding="utf-8"))
    todo = [x for x in wanted if limit or str(x) not in progress["done"]]
    if not todo:
        log(f"encode-docs: shards {wanted} already done")
        return {x: progress["done"][str(x)] for x in wanted}
    spans = {x: tuple(plan["shards"][x]["chunks"]) for x in todo}
    if limit:
        spans = {x: (lo, min(hi, lo + limit)) for x, (lo, hi) in spans.items()}
    texts_by: dict[int, list[str]] = {x: [] for x in todo}
    last = max(hi for _, hi in spans.values())
    with (OUT / "chunk_texts.jsonl").open(encoding="utf-8") as handle:       # CPU-side preparation, before the lock
        for i, line in enumerate(handle):
            if i >= last:
                break
            for x, (lo, hi) in spans.items():
                if lo <= i < hi:
                    texts_by[x].append(json.loads(line))
                    break
    frozen = np.load(FROZEN / "chunk_vectors.npy", mmap_mode="r")
    store = np.load(DOCS / "doc_tokens.npy", mmap_mode="r+")
    results: dict[int, dict[str, Any]] = {}
    with (gpu_job(f"encode-docs shards {todo}") if device == "cuda" else contextlib.nullcontext(0.0)) as waited:
        enc = Encoder(device)
        for x in todo:
            lo, hi = spans[x]
            n_vec, offset, unit = chunks["n_vec"][lo:hi], chunks["offset"][lo:hi], chunks["unit"][lo:hi]
            cosines = np.full(hi - lo, np.nan, dtype=np.float64)

            def sink(i: int, cls: np.ndarray, vec: np.ndarray, n_vec=n_vec, offset=offset, unit=unit, cosines=cosines) -> None:
                store[offset[i]:offset[i] + n_vec[i]] = vec
                cosines[i] = float(cls @ frozen[unit[i]])

            t1 = time.time()
            enc.encode(texts_by[x], DOC_MAX_TOKENS, plan["token_budget"], sink, lengths=n_vec.astype(np.int64) + 1)
            store.flush()
            peak = enc.torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
            result = {"shard": x, "chunks": hi - lo, "vectors": int(n_vec.sum()), "waited_s": round(waited, 1),
                      "job_shards": todo, "encode_s": round(time.time() - t1, 1), "cos_min": float(cosines.min()),
                      "cos_mean": float(cosines.mean()), "peak_mib": round(peak), "finished_utc": utc()}
            if result["cos_min"] < 0.999:
                raise RuntimeError(f"CLS vectors do not reproduce the frozen vectors: {result}")
            results[x] = result
            if limit:
                log(f"encode-docs: smoke test on {hi - lo} chunks of shard {x} ({device}) ok {result}")
                continue
            progress = json.loads((DOCS / "progress.json").read_text(encoding="utf-8"))
            progress["done"][str(x)] = result
            (DOCS / "progress.json").write_text(json.dumps(progress, indent=1), encoding="utf-8")
            record_stage(f"encode_docs_shard{x}", time.time() - t1, device="cuda GTX 1070 fp32", **result)
            log(f"encode-docs: shard {x} done {result}")
        if device == "cuda":
            enc.free()
    del store
    return results


def run_all_shards(courtesy_s: int = 0, group: int = 4) -> None:
    """Run the query job and the outstanding shards in jobs of ``group`` shards, each job its own process.

    One process per job releases all GPU memory between jobs. When a job had to wait for the lock, another agent is
    probably queueing, so pause ``courtesy_s`` (> the 60 s poll) after releasing, letting a waiter take the lock.
    """
    plan = json.loads((DOCS / "plan.json").read_text(encoding="utf-8"))
    done = json.loads((DOCS / "progress.json").read_text(encoding="utf-8"))["done"]
    outstanding = [x for x in range(len(plan["shards"])) if str(x) not in done]

    def queries_done() -> dict[str, Any] | None:
        progress = json.loads((QUERIES / "progress.json").read_text(encoding="utf-8"))
        return progress if progress.get("done") else None

    def shards_done(ids: list[int]) -> dict[str, Any] | None:
        finished = json.loads((DOCS / "progress.json").read_text(encoding="utf-8"))["done"]
        return finished.get(str(ids[-1])) if all(str(i) in finished for i in ids) else None

    jobs: list[tuple[str, list[str], Callable[[], dict[str, Any] | None]]] = [("queries", ["encode-queries"], queries_done)]
    for k in range(0, len(outstanding), group):
        ids = outstanding[k:k + group]
        jobs.append((f"shards {ids}", ["encode-docs", "--shards", ",".join(map(str, ids))], lambda ids=ids: shards_done(ids)))
    for position, (name, args, finished) in enumerate(jobs):
        if finished() is not None:
            continue
        subprocess.run([sys.executable, "-B", "-m", "uol.embedding_push.colbert", *args], check=True)
        waited = (finished() or {}).get("waited_s", 0)
        if waited > 5 and position + 1 < len(jobs):
            log(f"{name}: the lock was held by someone else for {waited:.0f}s; pausing {courtesy_s}s so a waiting agent can take it")
            time.sleep(courtesy_s)


def stage_encode_queries(device: str = "cuda", limit: int = 0) -> dict[str, Any]:
    """All 5,980 queries (<= 512 tokens) into the float16 store; CLS checked against the frozen query vectors.

    ``device="cpu"`` with ``limit`` is a smoke test only (first ``limit`` queries, nothing recorded).
    """
    qmeta = np.load(OUT / "queries_meta.npz")
    progress = json.loads((QUERIES / "progress.json").read_text(encoding="utf-8"))
    if progress.get("done") and not limit:
        log("encode-queries: already done")
        return progress
    texts = [q.text for q in common.release().queries]
    n_vec, offset = qmeta["n_vec"], qmeta["offset"]
    if limit:
        texts, n_vec, offset = texts[:limit], n_vec[:limit], offset[:limit]
    frozen = np.load(FROZEN / "query_vectors.npy", mmap_mode="r")
    store = np.load(QUERIES / "query_tokens.npy", mmap_mode="r+")
    cosines = np.full(len(texts), np.nan, dtype=np.float64)

    def sink(i: int, cls: np.ndarray, vec: np.ndarray) -> None:
        store[offset[i]:offset[i] + n_vec[i]] = vec
        cosines[i] = float(cls @ frozen[i])

    t0 = time.time()
    with (gpu_job("encode-queries") if device == "cuda" else contextlib.nullcontext(0.0)) as waited:
        t1 = time.time()
        enc = Encoder(device)
        enc.encode(texts, QUERY_MAX_TOKENS, 8192, sink, lengths=n_vec.astype(np.int64) + 1)
        peak = enc.torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
        if device == "cuda":
            enc.free()
    store.flush()
    del store
    result = {"done": True, "queries": len(texts), "vectors": int(n_vec.sum()), "waited_s": round(waited, 1),
              "encode_s": round(time.time() - t1, 1), "cos_min": float(cosines.min()), "cos_mean": float(cosines.mean()),
              "peak_mib": round(peak), "finished_utc": utc()}
    if result["cos_min"] < 0.999:
        raise RuntimeError(f"query CLS vectors do not reproduce the frozen vectors: {result}")
    if limit:
        log(f"encode-queries: smoke test on {len(texts)} queries ({device}) ok {result}")
        return result
    (QUERIES / "progress.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    record_stage("encode_queries", time.time() - t0, device="cuda GTX 1070 fp32", **result)
    log(f"encode-queries: done {result}")
    return result


# ----------------------------------------------------------------------------------------------------------------
# stage: score (CPU or GPU)
# ----------------------------------------------------------------------------------------------------------------

def article_index(chunks: Any, n_c: int) -> dict[str, np.ndarray]:
    """Per candidate: first chunk row, chunk count, first vector offset and vector count in the document store.

    ``chunks`` is the ``chunks.npz`` content (rows sorted by frozen unit, hence by article), so an article's vectors
    are contiguous in the store.
    """
    cand, n_vec, offset = np.asarray(chunks["cand"]), np.asarray(chunks["n_vec"]), np.asarray(chunks["offset"])
    unique, start = np.unique(cand, return_index=True)
    first = np.full(n_c, -1, dtype=np.int64)
    first[unique] = start
    art_off = np.zeros(n_c, dtype=np.int64)
    art_off[unique] = offset[start]
    return {"first": first, "n_ch": np.bincount(cand, minlength=n_c).astype(np.int64),
            "art_vec": np.bincount(cand, weights=n_vec, minlength=n_c).astype(np.int64), "art_off": art_off,
            "chunk_len": n_vec.astype(np.int64)}


def score_query(q_vecs: np.ndarray, weights: np.ndarray, arts: np.ndarray, index: dict[str, np.ndarray], docs: Any,
                block_tokens: int, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """(maxp, union), each [len(arts), V], of one query against its pool articles read from the document store."""
    import torch

    dev = torch.device(device)
    q = torch.from_numpy(np.ascontiguousarray(q_vecs)).to(dev).float()
    w = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(dev)
    maxp = np.empty((len(arts), w.shape[1]), dtype=np.float32)
    union = np.empty_like(maxp)
    off, nvec, first, nch, clen = index["art_off"], index["art_vec"], index["first"], index["n_ch"], index["chunk_len"]
    for block in block_articles(arts, nvec, block_tokens):
        members = arts[block]
        d = torch.from_numpy(np.concatenate([docs[off[a]:off[a] + nvec[a]] for a in members])).to(dev).float()
        lengths = torch.from_numpy(np.concatenate([clen[first[a]:first[a] + nch[a]] for a in members])).to(dev)
        per_article = torch.from_numpy(nch[members]).to(dev)
        m, u = late_interaction(q, d, lengths, per_article, w)
        maxp[block], union[block] = m.cpu().numpy(), u.cpu().numpy()
    return maxp, union


def stage_score(device: str = "cpu", block_tokens: int = 49152, limit: int = 0, checkpoint_every: int = 500) -> dict[str, Any]:
    """Late-interaction scores for every pool pair, all variants, query-major with checkpoints."""
    t0 = time.time()
    pool = np.load(OUT / "pool.npz")
    qidx, cidx = pool["qidx"], pool["cidx"]
    chunks = {k: v for k, v in np.load(OUT / "chunks.npz").items()}
    qm = {k: v for k, v in np.load(OUT / "queries_meta.npz").items()}
    idf = np.load(OUT / "idf.npy")
    docs = np.load(DOCS / "doc_tokens.npy", mmap_mode="r")
    qtok = np.load(QUERIES / "query_tokens.npy", mmap_mode="r")
    n_q = int(qidx.max()) + 1
    ptr = np.searchsorted(qidx, np.arange(n_q + 1))
    index = article_index(chunks, len(common.release().candidate_ids))
    if (index["first"][cidx] < 0).any():
        raise ValueError("a pool article has no encoded chunks")
    scores = np.full((len(VARIANTS), len(qidx)), np.nan, dtype=np.float32)
    ckpt = OUT / "score_checkpoint.npz"
    start_q = 0
    if ckpt.exists() and not limit:
        saved = np.load(ckpt)
        scores, start_q = saved["scores"], int(saved["next_q"])
        log(f"score: resuming at query {start_q}")
    stop_q = min(n_q, limit) if limit else n_q
    last = time.time()
    for q in range(start_q, stop_q):
        lo, hi = ptr[q], ptr[q + 1]
        span = slice(int(qm["offset"][q]), int(qm["offset"][q] + qm["n_vec"][q]))
        weights = query_weightings(qm["token_ids"][span], qm["placeholder"][span], idf)
        maxp, union = score_query(np.asarray(qtok[span]), weights, cidx[lo:hi], index, docs, block_tokens, device)
        scores[0::2, lo:hi] = maxp.T                        # rows follow VARIANTS: weighting-major, maxp then union
        scores[1::2, lo:hi] = union.T
        if (q + 1) % checkpoint_every == 0 or q + 1 == stop_q:
            if not limit:
                np.savez(ckpt, scores=scores, next_q=q + 1)
            log(f"score: {q + 1}/{n_q} queries, {time.time() - t0:.0f}s ({(time.time() - last) / checkpoint_every:.3f}s/query)")
            last = time.time()
    if limit:
        return {"scores": scores[:, :ptr[stop_q]], "queries": stop_q}
    if not np.isfinite(scores).all():
        raise ValueError("unscored pool pairs remain")
    np.save(OUT / "pair_scores.npy", scores)
    record_stage("score", time.time() - t0, device=device, pairs=int(len(qidx)), variants=list(VARIANTS))
    log(f"score: done {len(qidx)} pairs in {time.time() - t0:.0f}s on {device}")
    return {"pairs": int(len(qidx))}


# ----------------------------------------------------------------------------------------------------------------
# stage: assemble pairs.npz and its manifest
# ----------------------------------------------------------------------------------------------------------------

def _model_hashes() -> dict[str, str]:
    return {name: common.sha256_file(common.BGE_DIR / name) for name in ("pytorch_model.bin", "colbert_linear.pt", "config.json",
                                                                          "tokenizer.json", "sentencepiece.bpe.model")}


def _stage_times() -> dict[str, float]:
    path = OUT / "stages.jsonl"
    if not path.exists():
        return {}
    totals: dict[str, float] = {}
    for rec in read_jsonl(path):
        key = "encode_docs" if rec["stage"].startswith("encode_docs_shard") else rec["stage"]
        totals[key] = round(totals.get(key, 0.0) + rec["seconds"], 1)
    return totals


def _stage_device(stage: str) -> str:
    found = [rec for rec in read_jsonl(OUT / "stages.jsonl") if rec["stage"] == stage]
    return found[-1].get("device", "unknown") if found else "unknown"


def stage_assemble() -> dict[str, Any]:
    """pairs.npz (qidx, cidx, score plus the extra variants and the pool provenance) and manifest.json."""
    t0 = time.time()
    pool = np.load(OUT / "pool.npz")
    scores = np.load(OUT / "pair_scores.npy")
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {"qidx": pool["qidx"].astype(np.int32), "cidx": pool["cidx"].astype(np.int32),
                                     "score": scores[VARIANTS.index(PRIMARY)].astype(np.float32)}
    for i, name in enumerate(VARIANTS):
        if name != PRIMARY:
            arrays[f"score_{name}"] = scores[i].astype(np.float32)
    arrays.update(flag=pool["flag"].astype(np.int8), bm25=pool["bm25"].astype(np.float32), dense=pool["dense"].astype(np.float32))
    np.savez(PAIRS_DIR / "pairs.npz", **arrays)
    pool_stats = json.loads((OUT / "pool_stats.json").read_text(encoding="utf-8"))
    chunk_stats = json.loads((OUT / "chunk_stats.json").read_text(encoding="utf-8"))
    manifest = {
        "method_id": "colbert_bge", "kind": "sparse rerank features (pairs.npz)", "agent": OWNER,
        "created_utc": utc(),
        "description": "bge-m3 multi-vector (ColBERT-style) late interaction scores for every (query, pool candidate) pair; label-free",
        "arrays": {"qidx": "release query row (sorted)", "cidx": "candidate index in the bound catalogue (ascending within a query)",
                   "score": "PRIMARY: uniform query weights, MaxP over chunks (the standard BGE-M3 colbert_score)",
                   **{f"score_{v}": f"extra variant {v}" for v in VARIANTS if v != PRIMARY},
                   "flag": "1 = in BM25 top-100, 2 = in frozen bge-m3 MaxP top-100, 3 = both",
                   "bm25": "BM25 raw score (k1 1.2, b .75) of the pair", "dense": "frozen bge-m3 MaxP cosine of the pair"},
        "pairs": int(len(arrays["qidx"])), "queries": int(arrays["qidx"].max()) + 1,
        "definition": {
            "token_vectors": "colbert_linear(last_hidden_state[:, 1:]), CLS dropped, </s> kept, padding removed, L2-normalised per token, "
                             "stored float16 after fp32 normalisation",
            "score": "mean over query vectors of the maximum dot product with a document vector (FlagEmbedding colbert_score)",
            "multi_chunk": "MaxP (max of chunk scores) for score; *_union takes each query vector's best vector across all chunks",
            "weightings": {"uniform": "all kept query vectors", "nomask": "drops [CITA_OCULTA] placeholder tokens",
                           "idf": "BM25-style IDF of the token id over all 41,030 chunks, placeholder dropped"},
            "doc_chunking": f"uol.facts_to_law.bge_gpu.chunk_article: {DOC_MAX_TOKENS} tokens, title repeat, overlap {OVERLAP}, cap {CAP}; "
                            "reproduces the 41,030 frozen units (verified)",
            "query": f"masked Hechos text, truncated at {QUERY_MAX_TOKENS} tokens, no instruction prefix",
            "precision": "fp32 forward and scoring; token vectors stored float16",
        },
        "pool": pool_stats, "chunks": chunk_stats,
        "label_use": "none. Pool, encoding and scoring read no label; every row (dev and lockbox) comes from the same fixed procedure",
        "selection_evidence": "see dev_check in this manifest (dev rows only; written by the eval stage)",
        "seeds": "none (deterministic)",
        "input_sha256": {"release_manifest": common.sha256_file(common.RELEASE / "manifest.json"),
                         "split": common.sha256_file(common.SPLIT_PATH),
                         "frozen_receipt": common.sha256_file(FROZEN / "receipt.json"),
                         "frozen_query_vectors": common.sha256_file(FROZEN / "query_vectors.npy"),
                         "frozen_chunk_vectors": common.sha256_file(FROZEN / "chunk_vectors.npy"),
                         "frozen_units": common.sha256_file(FROZEN / "units.jsonl"),
                         "pool_npz": common.sha256_file(OUT / "pool.npz"), "chunks_npz": common.sha256_file(OUT / "chunks.npz"),
                         "queries_meta_npz": common.sha256_file(OUT / "queries_meta.npz"), "idf_npy": common.sha256_file(OUT / "idf.npy"),
                         "model_files": _model_hashes()},
        "output_sha256": {"pairs.npz": common.sha256_file(PAIRS_DIR / "pairs.npz")},
        "code_sha256": {str(Path(__file__)): common.sha256_file(Path(__file__)),
                        str(Path(common.__file__)): common.sha256_file(Path(common.__file__))},
        "wall_time_s": _stage_times(),
        "device": {"encoding": "NVIDIA GeForce GTX 1070, fp32", "scoring": _stage_device("score")},
        "bm25_choice": "plain BM25 k1=1.2 b=.75 from uol.facts_to_law.lexical, computed here: no complete scores/bm25* or scores/L_star "
                       "existed at pool-build time",
    }
    (PAIRS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")
    record_stage("assemble", time.time() - t0, device="cpu")
    log(f"assemble: wrote {PAIRS_DIR / 'pairs.npz'} ({manifest['pairs']} pairs)")
    return manifest


def _update_manifest(path: Path, key: str, value: Any) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest[key] = value
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------
# stage: dev check (dev rows only)
# ----------------------------------------------------------------------------------------------------------------

def comp_macro(values: Sequence[float], comps: Sequence[str]) -> float:
    """Mean over components of the mean over each component's queries."""
    labels = np.asarray(list(comps))
    _, inverse = np.unique(labels, return_inverse=True)
    sums = np.bincount(inverse, weights=np.asarray(values, dtype=np.float64))
    return float((sums / np.bincount(inverse)).mean())


def per_component(values_by_row: dict[int, float], rows: Sequence[int]) -> tuple[list[str], np.ndarray]:
    data = common.release()
    comps = [data.queries[int(r)].component_id for r in rows]
    ordered = sorted(set(comps))
    index = {c: i for i, c in enumerate(ordered)}
    sums, counts = np.zeros(len(ordered)), np.zeros(len(ordered))
    for r, c in zip(rows, comps):
        sums[index[c]] += values_by_row[int(r)]
        counts[index[c]] += 1
    return ordered, sums / counts


def crossfit_choice(per_w: dict[Any, dict[int, float]], rows: Sequence[int]) -> dict[str, Any]:
    """Pick one parameter per outer fold using only that fold's training rows; return the out-of-fold per-query values."""
    data = common.release()
    roles = common.roles()
    comp = {int(r): data.queries[int(r)].component_id for r in rows}
    chosen: dict[int, Any] = {}
    oof: dict[int, float] = {}
    for k in range(5):
        test = [int(r) for r in rows if roles[r] == f"dev_fold{k}"]
        train = [int(r) for r in rows if roles[r] not in ("lockbox", f"dev_fold{k}")]
        best, best_value = None, -1.0
        for key, values in per_w.items():
            value = comp_macro([values[r] for r in train], [comp[r] for r in train])
            if value > best_value + 1e-12:
                best, best_value = key, value
        chosen[k] = best
        oof.update({r: per_w[best][r] for r in test})
    return {"chosen": chosen, "oof": oof, "macro": comp_macro([oof[int(r)] for r in rows], [comp[int(r)] for r in rows])}


def stage_eval(partners: str = "auto") -> dict[str, Any]:
    """Dev-only quick check: (a) ColBERT rerank of the pool, (b) robust-z fusion with L_star (or the BM25 proxy)."""
    from uol.facts_to_law import fusion, lexical, stats

    t0 = time.time()
    data = common.release()
    n_c = len(data.candidate_ids)
    pairs = np.load(PAIRS_DIR / "pairs.npz")
    qidx, cidx = pairs["qidx"], pairs["cidx"]
    ptr = np.searchsorted(qidx, np.arange(len(data.queries) + 1))
    dev = common.dev_rows()
    rows = common.population_rows("P", dev)               # P_spec rows are a subset of the P rows
    pops = {"P_spec": common.population_rows("P_spec", dev), "P": rows}
    qv = np.load(FROZEN / "query_vectors.npy")
    cv = np.load(FROZEN / "chunk_vectors.npy")
    owner = common.chunk_article_index(FROZEN / "units.jsonl")
    dense = np.empty((len(rows), n_c), dtype=np.float32)
    for start, block in common.maxp_scores(qv[rows], cv, owner, n_c):
        dense[start:start + len(block)] = block
    del cv
    results: dict[str, Any] = {"rows": {name: int(len(r)) for name, r in pops.items()}}
    per_query: dict[str, dict[str, dict[int, float]]] = {}      # config -> population -> row -> ndcg@10
    reports: dict[str, dict[str, Any]] = {}                      # config -> population -> evaluate() result

    def run(config: str, matrix: np.ndarray, info: dict[str, Any]) -> None:
        reports[config] = {}
        per_query[config] = {}
        for pop in ("P_spec", "P"):
            res = common.evaluate(matrix, rows, pop)
            reports[config][pop] = {"macro": res["macro"], "n_queries": res["n_queries"], "n_components": res["n_components"]}
            per_query[config][pop] = {int(r): float(q["ndcg_at_10"]) for r, q in zip(res["rows"], res["per_query"])}
            search_log({"stage": "dev_check", "config": config, "population": pop, **info,
                        **{k: round(float(v), 5) for k, v in res["macro"].items()}, "n_queries": res["n_queries"],
                        "n_components": res["n_components"]})
        log("eval: {:<44s} P_spec nDCG@10 {:.4f}  P {:.4f}".format(config, reports[config]["P_spec"]["macro"]["ndcg_at_10"],
                                                                  reports[config]["P"]["macro"]["ndcg_at_10"]))

    def pool_slice(i: int) -> slice:
        r = int(rows[i])
        return slice(ptr[r], ptr[r + 1])

    def rerank(values: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        """Pool pairs ranked by ``values`` above every non-pool candidate, which keep the order of ``fallback``."""
        out = np.array(fallback, dtype=np.float32, copy=True)
        for i in range(len(rows)):
            sl = pool_slice(i)
            out[i, cidx[sl]] = 10.0 + values[sl]
        return out

    # controls
    run("control_dense_maxp_frozen", dense, {"kind": "control"})
    log("eval: fitting BM25 for the full-catalogue control")
    index = lexical.LexicalIndex(data.candidate_texts, "bm25", {"k1": BM25_K1, "b": BM25_B})
    bm25 = np.empty((len(rows), n_c), dtype=np.float32)
    for start in range(0, len(rows), 256):
        bm25[start:start + 256] = index.score([data.queries[int(r)].text for r in rows[start:start + 256]])
    del index
    run("control_bm25", bm25, {"kind": "control"})
    # (a) ColBERT rerank of the pool, non-pool below in frozen dense order
    for name in VARIANTS:
        values = pairs["score" if name == PRIMARY else f"score_{name}"]
        run(f"colbert_{name}", rerank(values, dense), {"kind": "colbert_rerank", "variant": name,
                                                       "pool": "BM25-100 union dense-100", "non_pool_order": "frozen dense MaxP"})
    # (b) fusion with L_star when it exists, else the BM25 proxy (stated)
    lstar_path = common.score_path("L_star")
    use_lstar = partners in ("auto", "lstar") and lstar_path.exists() and (lstar_path.parent / "manifest.json").exists()
    if use_lstar:
        lex_rows = np.asarray(common.open_scores("L_star")[rows])
        lex_name = "L_star"
        run("control_L_star", lex_rows, {"kind": "control"})
    else:
        lex_rows, lex_name = bm25, "bm25_proxy"
        log("eval: scores/L_star not available; fusing with plain BM25 (k1 1.2, b .75) as the proxy")
    results["lexical_partner"] = lex_name
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 1e6]                  # weight of the ColBERT channel; 0 = lexical alone, 1e6 = ColBERT alone
    grid_full = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]

    def fused_matrix(values: np.ndarray, partner_rows: np.ndarray, w: float) -> np.ndarray:
        """Pool style: both channels robust-z'd within the pool and fused; every non-pool candidate ranks below the pool."""
        out = np.empty((len(rows), n_c), dtype=np.float32)
        for i in range(len(rows)):
            sl = pool_slice(i)
            a = fusion.robust_z(partner_rows[i, cidx[sl]])
            b = fusion.robust_z(values[sl])
            pooled = a + np.float32(w) * b if w < 1e5 else b
            fallback = partner_rows[i] - partner_rows[i].max() - 1.0     # all below zero
            fallback[cidx[sl]] = pooled - pooled.min()                    # pool >= 0
            out[i] = fallback
        return out

    def fused_full_matrix(values: np.ndarray, partner_rows: np.ndarray, w: float) -> np.ndarray:
        """Full style: robust-z of the partner over the whole catalogue plus w * ColBERT z (within the pool);
        a non-pool candidate takes the pool's lowest ColBERT z, so strong partner hits outside the pool keep their place."""
        out = np.empty((len(rows), n_c), dtype=np.float32)
        for i in range(len(rows)):
            sl = pool_slice(i)
            z_full = fusion.robust_z(partner_rows[i])
            z_c = fusion.robust_z(values[sl])
            row = z_full + np.float32(w) * z_c.min()
            row[cidx[sl]] = z_full[cidx[sl]] + np.float32(w) * z_c
            out[i] = row
        return out

    fusion_variants = [PRIMARY]
    best_by_dev = max(VARIANTS, key=lambda v: reports[f"colbert_{v}"]["P_spec"]["macro"]["ndcg_at_10"])
    if best_by_dev != PRIMARY:
        fusion_variants.append(best_by_dev)
    results["best_variant_by_dev_P_spec"] = best_by_dev
    families: list[tuple[str, list[float]]] = []
    for name in fusion_variants:
        values = pairs["score" if name == PRIMARY else f"score_{name}"]
        for w in grid:
            run(f"fuse_{lex_name}+colbert_{name}_w{w:g}", fused_matrix(values, lex_rows, w),
                {"kind": "fusion", "style": "pool", "partner": lex_name, "variant": name, "w_colbert": w, "z": "robust_z within the pool"})
        families.append((f"fuse_{lex_name}+colbert_{name}", grid))
        for w in grid_full:
            run(f"fusefull_{lex_name}+colbert_{name}_w{w:g}", fused_full_matrix(values, lex_rows, w),
                {"kind": "fusion", "style": "full", "partner": lex_name, "variant": name, "w_colbert": w,
                 "z": "partner robust_z over the catalogue, ColBERT robust_z within the pool, non-pool at the pool minimum"})
        families.append((f"fusefull_{lex_name}+colbert_{name}", grid_full))
    # ColBERT + frozen dense (pool z), primary variant only
    for w in grid:
        run(f"fuse_dense+colbert_{PRIMARY}_w{w:g}", fused_matrix(pairs["score"], dense, w),
            {"kind": "fusion", "style": "pool", "partner": "dense_maxp_frozen", "variant": PRIMARY, "w_colbert": w,
             "z": "robust_z within the pool"})
    families.append((f"fuse_dense+colbert_{PRIMARY}", grid))
    # three channels: partner + w_d * frozen dense + w_c * ColBERT, each robust-z'd within the pool (ColBERT's marginal value
    # on top of "partner + dense"); w_c = 0 is the matching no-ColBERT comparator
    grid3 = [(wd, wc) for wd in (0.0, 0.5, 1.0, 2.0) for wc in (0.0, 0.5, 1.0, 2.0)]
    z_partner = [fusion.robust_z(lex_rows[i, cidx[pool_slice(i)]]) for i in range(len(rows))]
    z_dense = [fusion.robust_z(dense[i, cidx[pool_slice(i)]]) for i in range(len(rows))]
    triple_families: list[tuple[str, str]] = []
    for name in fusion_variants:
        values = pairs["score" if name == PRIMARY else f"score_{name}"]
        z_colbert = [fusion.robust_z(values[pool_slice(i)]) for i in range(len(rows))]
        for wd, wc in grid3:
            out = np.empty((len(rows), n_c), dtype=np.float32)
            for i in range(len(rows)):
                sl = pool_slice(i)
                pooled = z_partner[i] + np.float32(wd) * z_dense[i] + np.float32(wc) * z_colbert[i]
                fallback = lex_rows[i] - lex_rows[i].max() - 1.0
                fallback[cidx[sl]] = pooled - pooled.min()
                out[i] = fallback
            run(f"fuse3_{lex_name}+dense{wd:g}+colbert_{name}{wc:g}", out,
                {"kind": "fusion", "style": "pool3", "partner": lex_name, "variant": name, "w_dense": wd, "w_colbert": wc,
                 "z": "robust_z within the pool"})
        triple_families.append((name, f"fuse3_{lex_name}+dense{{wd:g}}+colbert_{name}{{wc:g}}"))
    # cross-fitted choice of w (training rows of each outer fold only), per population
    xfit: dict[str, Any] = {}
    for name, template in triple_families:
        for label, subgrid in ((f"fuse3_{lex_name}+dense+colbert_{name}", grid3),
                               (f"fuse3_{lex_name}+dense_only(no colbert)_{name}", [g for g in grid3 if g[1] == 0.0])):
            per_query[f"{label}|crossfit"] = {}
            for pop in ("P_spec", "P"):
                per_w = {g: per_query[template.format(wd=g[0], wc=g[1])][pop] for g in subgrid}
                choice = crossfit_choice(per_w, list(pops[pop]))
                xfit[f"{label}|{pop}"] = {"chosen_wd_wc_by_fold": {k: list(v) for k, v in choice["chosen"].items()},
                                          "oof_ndcg_at_10": round(choice["macro"], 5)}
                per_query[f"{label}|crossfit"][pop] = choice["oof"]
                search_log({"stage": "dev_check", "config": f"{label}|crossfit_w", "population": pop,
                            "chosen_wd_wc_by_fold": {k: list(v) for k, v in choice["chosen"].items()},
                            "ndcg_at_10": round(choice["macro"], 5)})
                log(f"eval: {label}|crossfit {pop} oof nDCG@10 {choice['macro']:.4f} (wd, wc) by fold {choice['chosen']}")
    for tag, weights_grid in families:
        per_query[f"{tag}|crossfit"] = {}
        for pop in ("P_spec", "P"):
            per_w = {w: per_query[f"{tag}_w{w:g}"][pop] for w in weights_grid}
            choice = crossfit_choice(per_w, list(pops[pop]))
            xfit[f"{tag}|{pop}"] = {"chosen_w_by_fold": choice["chosen"], "oof_ndcg_at_10": round(choice["macro"], 5)}
            per_query[f"{tag}|crossfit"][pop] = choice["oof"]
            search_log({"stage": "dev_check", "config": f"{tag}|crossfit_w", "population": pop,
                        "chosen_w_by_fold": choice["chosen"], "ndcg_at_10": round(choice["macro"], 5)})
            log(f"eval: {tag}|crossfit {pop} oof nDCG@10 {choice['macro']:.4f} w by fold {choice['chosen']}")
    results["crossfit_w"] = xfit
    # summary table and paired contrasts (component bootstrap, dev rows)
    table: dict[str, dict[str, float]] = {}
    for config, byp in reports.items():
        table[config] = {pop: round(float(byp[pop]["macro"]["ndcg_at_10"]), 4) for pop in ("P_spec", "P")}
        table[config].update({f"{pop}_recall_at_100": round(float(byp[pop]["macro"]["recall_at_100"]), 4) for pop in ("P_spec", "P")})
    results["table"] = table

    def contrast(config_a: str, config_b: str, pop: str) -> dict[str, Any]:
        comps_a, a = per_component(per_query[config_a][pop], list(pops[pop]))
        comps_b, b = per_component(per_query[config_b][pop], list(pops[pop]))
        assert comps_a == comps_b
        out = stats.contrast(a, b)
        return {"pop": pop, "a": config_a, "b": config_b, "effect": round(out["effect"], 4),
                "ci95": [round(x, 4) for x in out["ci95"]], "verdict": out["verdict"]}

    ref = "control_L_star" if use_lstar else "control_bm25"
    tag = f"fuse_{lex_name}+colbert_{PRIMARY}"
    pairs_to_test = [(f"colbert_{PRIMARY}", "control_dense_maxp_frozen"), (f"colbert_{PRIMARY}", "control_bm25"), (f"{tag}|crossfit", ref),
                     (f"fusefull_{lex_name}+colbert_{PRIMARY}|crossfit", ref),
                     (f"fuse_dense+colbert_{PRIMARY}|crossfit", "control_dense_maxp_frozen"),
                     (f"fuse3_{lex_name}+dense+colbert_{PRIMARY}|crossfit", f"fuse3_{lex_name}+dense_only(no colbert)_{PRIMARY}|crossfit"),
                     (f"fuse3_{lex_name}+dense_only(no colbert)_{PRIMARY}|crossfit", ref)]
    results["contrasts"] = [contrast(a, b, pop) for a, b in pairs_to_test for pop in ("P_spec", "P")]
    results["seconds"] = round(time.time() - t0, 1)
    (OUT / "dev_check.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    _update_manifest(PAIRS_DIR / "manifest.json", "dev_check", {k: results[k] for k in (
        "rows", "table", "lexical_partner", "best_variant_by_dev_P_spec", "crossfit_w", "contrasts")})
    record_stage("eval", time.time() - t0, device="cpu")
    log(f"eval: done in {results['seconds']}s")
    return results


# ----------------------------------------------------------------------------------------------------------------
# optional: dense_bge_long (single CLS vector of up to 8,192 tokens for the multi-chunk articles)
# ----------------------------------------------------------------------------------------------------------------

def multi_chunk_articles() -> np.ndarray:
    owner = common.chunk_article_index(FROZEN / "units.jsonl")
    counts = np.bincount(owner, minlength=len(common.release().candidate_ids))
    return np.flatnonzero(counts > 1)


def stage_long_encode(max_tokens: int = LONG_MAX_TOKENS, device: str = "cuda", limit: int = 0) -> dict[str, Any]:
    """Single CLS vector per multi-chunk article, up to ``max_tokens`` tokens. ``device="cpu"`` + ``limit`` is a smoke test."""
    LONG.mkdir(parents=True, exist_ok=True)
    out = LONG / "long_cls.npy"
    if out.exists() and not limit:
        log("long-encode: already done")
        return json.loads((LONG / "long_info.json").read_text(encoding="utf-8"))
    data = common.release()
    multi = multi_chunk_articles()
    if limit:
        multi = multi[:limit]
    texts = [data.candidate_texts[int(a)] for a in multi]
    vectors = np.zeros((len(multi), DIM), dtype=np.float32)
    seen = [0]

    def sink(i: int, cls: np.ndarray, vec: np.ndarray) -> None:
        vectors[i] = cls
        seen[0] += 1

    t0 = time.time()
    with (gpu_job("long-encode") if device == "cuda" else contextlib.nullcontext(0.0)) as waited:
        t1 = time.time()
        enc = Encoder(device)
        lengths = enc.lengths(texts, max_tokens)
        enc.encode(texts, max_tokens, max_tokens, sink, lengths=lengths, tokens=False)
        peak = enc.torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else 0.0
        if device == "cuda":
            enc.free()
    if seen[0] != len(multi):
        raise RuntimeError("not every multi-chunk article was encoded")
    if limit:
        first = common.chunk_article_index(FROZEN / "units.jsonl")
        frozen = np.load(FROZEN / "chunk_vectors.npy", mmap_mode="r")
        cos = [float(vectors[i] @ frozen[int(np.searchsorted(first, a))]) for i, a in enumerate(multi)]
        log(f"long-encode: smoke test on {len(multi)} articles ({device}), lengths {lengths.tolist()}, "
            f"cosine of the long vector with the chunk-0 vector {np.round(cos, 3).tolist()}")
        return {"lengths": lengths.tolist(), "cos_with_chunk0": cos}
    np.save(out, vectors)
    np.save(LONG / "long_articles.npy", multi)
    info = {"articles": int(len(multi)), "max_tokens": max_tokens, "token_mean": float(lengths.mean()), "token_max": int(lengths.max()),
            "truncated_at_max": int((lengths >= max_tokens).sum()), "waited_s": round(waited, 1), "encode_s": round(time.time() - t1, 1),
            "peak_mib": round(peak), "finished_utc": utc()}
    (LONG / "long_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    record_stage("long_encode", time.time() - t0, device="cuda GTX 1070 fp32", **info)
    log(f"long-encode: done {info}")
    return info


def stage_long_scores() -> dict[str, Any]:
    """dense_bge_long: frozen CLS for single-chunk articles, the long single-vector encoding for the others."""
    t0 = time.time()
    data = common.release()
    n_c = len(data.candidate_ids)
    cv = np.load(FROZEN / "chunk_vectors.npy")
    qv = np.load(FROZEN / "query_vectors.npy")
    owner = common.chunk_article_index(FROZEN / "units.jsonl")
    first = np.full(n_c, -1, dtype=np.int64)
    for u in range(len(owner) - 1, -1, -1):
        first[owner[u]] = u                                  # first unit of each article (chunk 0)
    articles = cv[first].astype(np.float32)
    multi = np.load(LONG / "long_articles.npy")
    articles[multi] = np.load(LONG / "long_cls.npy")
    if not np.allclose(np.linalg.norm(articles, axis=1), 1.0, atol=1e-3):
        raise ValueError("article vectors are not unit length")
    matrix = common.create_scores(LONG_ID)
    for start in range(0, len(qv), 512):
        matrix[start:start + 512] = qv[start:start + 512] @ articles.T
    matrix.flush()
    info = json.loads((LONG / "long_info.json").read_text(encoding="utf-8"))
    manifest = {"kind": "full matrix (dense cosine)", "agent": OWNER,
                "description": "bge-m3 dense CLS; articles with more than one frozen chunk are encoded as one sequence of up to "
                               f"{info['max_tokens']} tokens (no chunking); every other article reuses its frozen chunk vector; "
                               "queries are the frozen query vectors",
                "multi_chunk_articles": int(len(multi)), "long_encoding": info, "label_use": "none (every row from one fixed procedure)",
                "selection_evidence": "see dev_check in this manifest", "seeds": "none (deterministic)", "device": "GTX 1070 fp32",
                "wall_time_s": _stage_times(),
                "input_sha256": {"frozen_chunk_vectors": common.sha256_file(FROZEN / "chunk_vectors.npy"),
                                 "frozen_query_vectors": common.sha256_file(FROZEN / "query_vectors.npy"),
                                 "long_cls": common.sha256_file(LONG / "long_cls.npy"), "model_files": _model_hashes()}}
    record = common.finish_scores(LONG_ID, manifest, code_files=[Path(__file__), Path(common.__file__)])
    record_stage("long_scores", time.time() - t0, device="cpu")
    log("long-scores: wrote the full matrix")
    return record


def stage_long_eval() -> dict[str, Any]:
    t0 = time.time()
    data = common.release()
    n_c = len(data.candidate_ids)
    dev = common.dev_rows()
    rows = common.population_rows("P", dev)
    pops = {"P_spec": common.population_rows("P_spec", dev), "P": rows}
    qv = np.load(FROZEN / "query_vectors.npy")
    cv = np.load(FROZEN / "chunk_vectors.npy")
    owner = common.chunk_article_index(FROZEN / "units.jsonl")
    dense = np.empty((len(rows), n_c), dtype=np.float32)
    for start, block in common.maxp_scores(qv[rows], cv, owner, n_c):
        dense[start:start + len(block)] = block
    long_rows = np.asarray(common.open_scores(LONG_ID)[rows])
    out: dict[str, Any] = {"rows": {k: int(len(v)) for k, v in pops.items()}}
    results = {}
    for name, matrix in (("maxp_frozen", dense), ("long_single_vector", long_rows)):
        for pop in ("P_spec", "P"):
            res = common.evaluate(matrix, rows, pop)
            results[(name, pop)] = res
            search_log({"stage": "dev_check_long", "config": name, "population": pop,
                        **{k: round(float(v), 5) for k, v in res["macro"].items()}, "n_queries": res["n_queries"]})
            out[f"{name}|{pop}"] = {k: round(float(v), 4) for k, v in res["macro"].items()}
    for pop in ("P_spec", "P"):
        con = common.contrast(results[("long_single_vector", pop)], results[("maxp_frozen", pop)])
        out[f"contrast_long_minus_maxp|{pop}"] = {k: con[k] for k in ("effect", "ci95", "ci90", "verdict", "mean_a", "mean_b")}
    multi = set(np.load(LONG / "long_articles.npy").tolist())
    out["P_queries_with_a_multi_chunk_positive"] = sum(1 for r in rows if any(p in multi for p in data.positive_idx[int(r)]))
    (OUT / "long_dev_check.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    _update_manifest(common.score_path(LONG_ID).parent / "manifest.json", "dev_check", out)
    record_stage("long_eval", time.time() - t0, device="cpu")
    log("long-eval: " + json.dumps({k: v for k, v in out.items() if k.endswith("P_spec") or k.endswith("|P")}, default=str))
    return out


# ----------------------------------------------------------------------------------------------------------------
# autonomous tail of the pipeline
# ----------------------------------------------------------------------------------------------------------------

def encodes_finished() -> bool:
    plan = json.loads((DOCS / "plan.json").read_text(encoding="utf-8"))
    done = json.loads((DOCS / "progress.json").read_text(encoding="utf-8"))["done"]
    queries = json.loads((QUERIES / "progress.json").read_text(encoding="utf-8")).get("done")
    return bool(queries) and all(str(s) in done for s in range(len(plan["shards"])))


def stage_run_rest(poll_s: int = 30, long_dense: bool = True) -> None:
    """Wait for the query and shard encodes, then: score (CPU), assemble, eval, then the optional long-dense GPU job and its scores.

    The core deliverable (pairs.npz and its dev check) needs no GPU, so it comes first. Stops at the first failure.
    Every stage is restartable, so a second invocation resumes where this one stopped.
    """
    while not encodes_finished():
        time.sleep(poll_s)
    log("run-rest: encodes finished")
    if not (OUT / "pair_scores.npy").exists():
        stage_score("cpu")
    stage_assemble()
    stage_eval()
    if long_dense:
        subprocess.run([sys.executable, "-B", "-m", "uol.embedding_push.colbert", "long-encode"], check=True)
        if not common.score_path(LONG_ID).exists():
            stage_long_scores()
        stage_long_eval()
    log("run-rest: all stages done")


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name in ("pool", "chunks", "qmeta", "assemble", "eval", "long-scores", "long-eval", "run-rest"):
        sub.add_parser(name)
    p = sub.add_parser("long-encode")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N multi-chunk articles, no record")
    p.add_argument("--max-tokens", type=int, default=LONG_MAX_TOKENS)
    p = sub.add_parser("alloc")
    p.add_argument("--shards", type=int, default=8)
    p = sub.add_parser("encode-docs")
    p.add_argument("--shard", type=int, help="one shard")
    p.add_argument("--shards", help="comma-separated shards forming one GPU job")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N chunks of each shard, no record")
    p = sub.add_parser("encode-docs-all")
    p.add_argument("--group", type=int, default=3, help="shards per GPU job")
    p = sub.add_parser("encode-queries")
    p.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    p.add_argument("--limit", type=int, default=0, help="smoke test: first N queries, no record")
    p = sub.add_parser("score")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--block-tokens", type=int, default=49152)
    args = parser.parse_args(argv)
    if args.stage == "pool":
        stage_pool()
    elif args.stage == "chunks":
        stage_chunks()
    elif args.stage == "qmeta":
        stage_queries_meta()
    elif args.stage == "alloc":
        stage_alloc(args.shards)
    elif args.stage == "encode-docs":
        ids = [args.shard] if args.shard is not None else [int(x) for x in args.shards.split(",")]
        stage_encode_docs(ids, args.device, args.limit)
    elif args.stage == "encode-docs-all":
        run_all_shards(group=args.group)
    elif args.stage == "encode-queries":
        stage_encode_queries(args.device, args.limit)
    elif args.stage == "score":
        if args.device == "cuda":
            with gpu_job("score"):
                stage_score("cuda", args.block_tokens)
        else:
            stage_score("cpu", args.block_tokens)
    elif args.stage == "assemble":
        stage_assemble()
    elif args.stage == "eval":
        stage_eval()
    elif args.stage == "long-encode":
        stage_long_encode(args.max_tokens, args.device, args.limit)
    elif args.stage == "run-rest":
        stage_run_rest()
    elif args.stage == "long-scores":
        stage_long_scores()
    elif args.stage == "long-eval":
        stage_long_eval()


if __name__ == "__main__":
    main()
