"""C01/C02 lexical controls on the frozen N06-MP SJF release.

The method phase consumes only ``method/queries.jsonl`` and
``method/candidates.jsonl`` (development rows are selected before ranking).
The all-split evaluation reference is streamed only after rankings are
frozen; rows whose query id is not in the development allowlist are discarded
immediately and never retained or emitted.  No test query or target is used.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from uol.retrieval_benchmarks.semantic_v2_common import CAMPAIGN_ID, digest, write_json


RELEASE = Path("data/releases/uol_semantic_benchmarks_20260916_v2/n06_mp_sjf_luna_a_v1")
TOP_K = 100
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260916
VECTOR_DIM = 4096
PROJECTION_DIM = 256
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
QUERY_INSTRUCTION = (
    "Dada una descripción publicada de hechos, recupera el criterio jurídico "
    "publicado que pertenece documentalmente a la misma ficha."
)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip(): yield json.loads(line)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rank_texts(query_texts: Sequence[str], candidate_ids: Sequence[str], candidate_texts: Sequence[str], *, analyzer: str, top_k: int = TOP_K) -> list[list[dict[str, Any]]]:
    if len(candidate_ids) != len(candidate_texts) or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique and aligned")
    if not query_texts or not candidate_texts or top_k <= 0:
        raise ValueError("queries, candidates and top_k must be non-empty/positive")
    vectorizer = TfidfVectorizer(analyzer=analyzer, ngram_range=(1, 2) if analyzer == "word" else (3, 5),
                                 strip_accents="unicode", lowercase=True, sublinear_tf=True,
                                 token_pattern=r"(?u)\b\w+\b" if analyzer == "word" else None,
                                 max_features=300000)
    matrix = vectorizer.fit_transform(candidate_texts)
    queries = vectorizer.transform(query_texts)
    ids = np.asarray([str(x) for x in candidate_ids], dtype=object)
    depth = min(top_k, len(ids)); output = []
    for row in queries @ matrix.T:
        scores = np.asarray(row.toarray()).ravel()
        order = np.lexsort((ids, -scores))[:depth]
        output.append([{"candidate_id": str(ids[i]), "rank": rank, "score": float(scores[i])} for rank, i in enumerate(order, 1)])
    return output


def _load_dev_references(path: Path, development_ids: set[str]) -> list[dict[str, Any]]:
    """Read only development rows into memory from the custodial reference.

    The file is an evaluator-side all-split artifact.  Non-development rows
    are discarded before any field other than ``query_id`` is inspected.
    """
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            query_id = json.loads(line).get("query_id")
            if query_id not in development_ids:
                continue
            row = json.loads(line)
            result.append({"query_id": str(row["query_id"]), "group_id": str(row["group_id"]),
                           "positive_candidate_ids": [str(x) for x in row["positive_candidate_ids"]]})
    return result


def evaluate(rankings: Mapping[str, Sequence[Mapping[str, Any]]], references: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[float]]]:
    details: list[dict[str, Any]] = []
    for ref in references:
        values = [str(row["candidate_id"]) for row in rankings.get(str(ref["query_id"]), ())]
        positives = set(str(x) for x in ref["positive_candidate_ids"])
        hits = [i + 1 for i, cid in enumerate(values) if cid in positives]
        relevant_at_10 = sum(rank <= 10 for rank in hits)
        relevant_at_20 = sum(rank <= 20 for rank in hits)
        relevant_at_100 = sum(rank <= 100 for rank in hits)
        ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(positives), 10)))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in hits if rank <= 10)
        details.append({"query_id": str(ref["query_id"]), "group_id": str(ref["group_id"]), "positive_count": len(positives),
                        "hits_at_10": relevant_at_10, "hits_at_20": relevant_at_20, "hits_at_100": relevant_at_100,
                        "recall_at_10": relevant_at_10 / len(positives) if positives else 0.0,
                        "recall_at_20": relevant_at_20 / len(positives) if positives else 0.0,
                        "recall_at_100": relevant_at_100 / len(positives) if positives else 0.0,
                        "ndcg_at_10": dcg / ideal if ideal else 0.0,
                        "first_positive_rank": min(hits) if hits else None})
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details: groups[row["group_id"]].append(row)
    metrics = {"queries": len(details), "groups": len(groups), "multipositive_queries": sum(r["positive_count"] > 1 for r in details),
               "mean_positive_count": float(np.mean([r["positive_count"] for r in details])) if details else 0.0,
               "recall_at_10": float(np.mean([r["recall_at_10"] for r in details])) if details else 0.0,
               "recall_at_20": float(np.mean([r["recall_at_20"] for r in details])) if details else 0.0,
               "recall_at_100": float(np.mean([r["recall_at_100"] for r in details])) if details else 0.0,
               "ndcg_at_10": float(np.mean([r["ndcg_at_10"] for r in details])) if details else 0.0,
               "group_macro_recall_at_20": float(np.mean([np.mean([r["recall_at_20"] for r in rows]) for rows in groups.values()])) if groups else 0.0,
               "group_macro_ndcg_at_10": float(np.mean([np.mean([r["ndcg_at_10"] for r in rows]) for rows in groups.values()])) if groups else 0.0}
    return metrics, details, {group: [float(row["recall_at_20"]) for row in rows] for group, rows in groups.items()}


def paired_bootstrap(group_values: Mapping[str, Sequence[float]], other_values: Mapping[str, Sequence[float]], *, seed: int = BOOTSTRAP_SEED, iterations: int = BOOTSTRAP) -> dict[str, Any]:
    groups = sorted(set(group_values) & set(other_values))
    if not groups: return {"iterations": iterations, "groups": 0, "delta": None, "ci95": [None, None], "seed": seed}
    rng = np.random.default_rng(seed); deltas = np.asarray([np.mean(other_values[g]) - np.mean(group_values[g]) for g in groups], dtype=float); observed = float(np.mean(deltas))
    sampled = rng.choice(deltas, size=(iterations, len(deltas)), replace=True).mean(axis=1)
    return {"iterations": iterations, "groups": len(groups), "delta_other_minus_first": observed, "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))], "seed": seed, "unit": "component/group"}


def fixed_projection(*, seed: str = "n06_mp_c02_projection_v1", input_dim: int = VECTOR_DIM, output_dim: int = PROJECTION_DIM) -> np.ndarray:
    """Deterministic untrained 4,096d-to-256d control."""
    if input_dim <= 0 or output_dim <= 0: raise ValueError("projection dimensions must be positive")
    value = int(digest({"seed": seed, "input_dim": input_dim, "output_dim": output_dim})[:16], 16)
    rng = np.random.default_rng(value)
    return rng.standard_normal((input_dim, output_dim), dtype=np.float64) / math.sqrt(output_dim)


def fit_ridge(train_queries: np.ndarray, train_candidates: np.ndarray, projection: np.ndarray, *, alpha: float = 1.0) -> np.ndarray:
    """Reference regression head; only caller-supplied train rows are used."""
    if projection.shape[0] != train_queries.shape[1] or train_candidates.shape[1] != train_queries.shape[1] or train_candidates.shape[0] != train_queries.shape[0] or alpha <= 0:
        raise ValueError("incompatible ridge inputs")
    q = train_queries / np.maximum(np.linalg.norm(train_queries, axis=1, keepdims=True), 1e-12)
    k = train_candidates / np.maximum(np.linalg.norm(train_candidates, axis=1, keepdims=True), 1e-12)
    z = q @ projection
    return np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ k)


def fit_low_rank_residual(train_queries: np.ndarray, candidate_vectors: np.ndarray, positives: Sequence[Sequence[int]], projection: np.ndarray, *, rank: int = 16, epochs: int = 2, learning_rate: float = .03, negatives: int = 8, seed: int = BOOTSTRAP_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Train a genuinely rank-constrained residual q' = q(I+UVᵀ).

    ``rank`` changes U/V shapes and therefore capacity.  The sampled hinge is
    multi-positive: every known positive is scored against the same negative
    set, while no unjudged item is promoted to a positive label.
    """
    if projection.shape[1] != PROJECTION_DIM or rank <= 0 or rank > PROJECTION_DIM or epochs <= 0: raise ValueError("invalid residual configuration")
    q = train_queries / np.maximum(np.linalg.norm(train_queries, axis=1, keepdims=True), 1e-12); q = q @ projection
    c = candidate_vectors / np.maximum(np.linalg.norm(candidate_vectors, axis=1, keepdims=True), 1e-12); c = c @ projection
    rng = np.random.default_rng(seed)
    # Small non-zero factors permit learning while keeping the initial map
    # close to identity; unlike a full matrix, capacity is exactly rank.
    u = rng.normal(0.0, 1e-3, (PROJECTION_DIM, rank)); v = rng.normal(0.0, 1e-3, (PROJECTION_DIM, rank)); all_i = np.arange(len(c))
    for _ in range(epochs):
        for qi in rng.permutation(len(q)):
            pos = sorted(set(int(i) for i in positives[qi]));
            if not pos: continue
            blocked = set(pos); pool = all_i[[int(i) not in blocked for i in all_i]]; neg = rng.choice(pool, size=min(negatives, len(pool)), replace=False)
            transform = np.eye(PROJECTION_DIM) + u @ v.T; z = q[qi] @ transform
            pos_scores = z @ c[pos].T; pos_mean = float(np.mean(pos_scores))
            neg_scores = z @ c[neg].T
            for ni, neg_score in zip(neg, neg_scores):
                if 1.0 - pos_mean + float(neg_score) > 0:
                    grad = np.outer(q[qi], np.mean(c[pos], axis=0) - c[int(ni)]) / len(pos)
                    u += learning_rate * (grad @ v); v += learning_rate * (grad.T @ u)
    return u, v


def _rank_vectors(query_vectors: np.ndarray, candidate_vectors: np.ndarray, candidate_ids: Sequence[str], *, top_k: int = TOP_K) -> list[list[dict[str, Any]]]:
    """Dense ranking with deterministic ID tie-breaks and no extra candidates."""
    if query_vectors.ndim != 2 or candidate_vectors.ndim != 2 or query_vectors.shape[1] != candidate_vectors.shape[1]:
        raise ValueError("dense matrices must be 2-D with the same dimension")
    ids = np.asarray([str(x) for x in candidate_ids], dtype=object)
    if len(set(ids)) != len(ids): raise ValueError("candidate IDs must be unique")
    depth = min(top_k, len(ids)); output = []
    import torch
    torch.set_num_threads(min(16, torch.get_num_threads()))
    scores_matrix = torch.from_numpy(np.asarray(query_vectors, dtype=np.float32)) @ torch.from_numpy(np.asarray(candidate_vectors, dtype=np.float32)).T
    for values in scores_matrix.numpy():
        order = np.lexsort((ids, -values))[:depth]
        output.append([{"candidate_id": str(ids[i]), "rank": rank, "score": float(values[i])} for rank, i in enumerate(order, 1)])
    return output


def fuse_rrf(char_rows: Sequence[Mapping[str, Any]], dense_rows: Sequence[Mapping[str, Any]], *, k: int = 60, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Fuse exactly the char∪dense candidate pool; word-only rows cannot enter."""
    scores: dict[str, float] = defaultdict(float)
    for row in char_rows: scores[str(row["candidate_id"])] += 1.0 / (k + int(row["rank"]))
    for row in dense_rows: scores[str(row["candidate_id"])] += 1.0 / (k + int(row["rank"]))
    ids = np.asarray(list(scores), dtype=object)
    order = np.lexsort((ids, -np.asarray([scores[x] for x in ids])))[:top_k]
    return [{"candidate_id": str(ids[i]), "rank": rank, "score": float(scores[str(ids[i])])} for rank, i in enumerate(order, 1)]


def fuse_calibrated(char_rows: Sequence[Mapping[str, Any]], dense_rows: Sequence[Mapping[str, Any]], *, alpha: float = .5, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """Unsupervised robust score fusion over only the char∪dense pool."""
    if not 0 <= alpha <= 1: raise ValueError("alpha must be in [0,1]")
    def normalize(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        values = {str(row["candidate_id"]): float(row["score"]) for row in rows}
        arr = np.asarray(list(values.values()), dtype=float); med = float(np.median(arr)) if len(arr) else 0.0
        mad = float(np.median(np.abs(arr - med))) if len(arr) else 0.0; scale = 1.4826 * mad or (float(np.std(arr)) or 1.0)
        return {key: (value - med) / scale for key, value in values.items()}
    char = normalize(char_rows); dense = normalize(dense_rows); ids = np.asarray(sorted(set(char) | set(dense)), dtype=object)
    scores = np.asarray([alpha * char.get(x, 0.0) + (1 - alpha) * dense.get(x, 0.0) for x in ids])
    order = np.lexsort((ids, -scores))[:top_k]
    return [{"candidate_id": str(ids[i]), "rank": rank, "score": float(scores[i])} for rank, i in enumerate(order, 1)]


def pool_recall(references: Sequence[Mapping[str, Any]], char_by_query: Mapping[str, Sequence[Mapping[str, Any]]], dense_by_query: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Recall of the un-reordered char∪dense pool, separate from fused order."""
    values = []
    for ref in references:
        qid = str(ref["query_id"]); positives = set(str(x) for x in ref["positive_candidate_ids"])
        pool = {str(row["candidate_id"]) for row in char_by_query.get(qid, ())} | {str(row["candidate_id"]) for row in dense_by_query.get(qid, ())}
        values.append(len(positives & pool) / len(positives) if positives else 0.0)
    return {"queries": len(values), "pool_recall_at_100": float(np.mean(values)) if values else 0.0,
            "multipositive": sum(1 for ref in references if len(ref["positive_candidate_ids"]) > 1),
            "pool_definition": "char top-100 union dense top-100; no word-only candidates"}


def build_train_targets(train_queries: Sequence[Mapping[str, Any]], train_references: Sequence[Mapping[str, Any]], candidate_vectors: np.ndarray, candidate_ids: Sequence[str]) -> np.ndarray:
    """Align one multi-positive target row to each query id, independent of ref order."""
    position = {str(cid): i for i, cid in enumerate(candidate_ids)}; references = {str(row["query_id"]): row for row in train_references}
    targets = []
    for query in train_queries:
        ref = references.get(str(query["query_id"]))
        indices = [position[cid] for cid in ref["positive_candidate_ids"] if cid in position] if ref else []
        if not indices: raise ValueError(f"missing train positives for {query['query_id']}")
        targets.append(np.mean(candidate_vectors[indices], axis=0))
    return np.asarray(targets)


def _post_json(route: str, payload: Mapping[str, Any], *, timeout: int = 90) -> dict[str, Any]:
    base = os.environ.get("VLLM_EMBEDDINGS_BASE_URL", "http://localhost:8000/embeddings/v1")
    headers = {"Content-Type": "application/json"}; key = os.environ.get("VLLM_EMBEDDINGS_API_KEY", os.environ.get("VLLM_API_KEY", ""))
    if key: headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(base.rstrip("/") + route, data=json.dumps(dict(payload), ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response: return json.load(response)


def embed_qwen_cached(cache_dir: Path, kind: str, texts: Sequence[str], *, instruction: str | None, model: str = EMBEDDING_MODEL, dimension: int = VECTOR_DIM, batch_size: int = 128, workers: int = 1) -> tuple[np.ndarray, dict[str, Any]]:
    """Generate/reuse Qwen vectors keyed by text, instruction, model and dimension."""
    cache_dir.mkdir(parents=True, exist_ok=True); manifest_path = cache_dir / f"{kind}_cache_manifest.json"
    input_texts = [f"Instruct: {instruction}\nQuery:{text}" if instruction else str(text) for text in texts]
    text_hash = digest(input_texts); operation = {"kind": kind, "text_sha256": text_hash, "instruction": instruction, "model": model, "dimension": dimension, "normalization": "L2", "batch_size": batch_size}
    op_hash = digest(operation); array_path = cache_dir / f"{kind}_{op_hash[:20]}.npy"; metadata_path = cache_dir / f"{kind}_{op_hash[:20]}.json"
    if array_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(array_path, mmap_mode="r")
        if metadata.get("operation_sha256") != op_hash or tuple(values.shape) != (len(texts), dimension): raise ValueError("embedding cache identity/shape mismatch")
        return np.asarray(values), {**metadata, "cache_hit": True}
    if workers < 1 or workers > 8: raise ValueError("workers must be between 1 and 8")
    chunks: list[np.ndarray] = []; requests = 0; reported_tokens = 0; failures = 0; started = time.monotonic(); chunk_dir = cache_dir / f"{kind}_{op_hash[:20]}_chunks"; chunk_dir.mkdir(exist_ok=True)
    jobs = []
    for begin in range(0, len(input_texts), batch_size):
        batch = input_texts[begin:begin + batch_size]; chunk_path = chunk_dir / f"{begin:06d}.npy"; chunk_meta = chunk_path.with_suffix(".json"); jobs.append((batch, chunk_path, chunk_meta))

    def load_or_fetch(job: tuple[list[str], Path, Path]) -> tuple[np.ndarray, int, int]:
        batch, chunk_path, chunk_meta = job
        if chunk_path.is_file() and chunk_meta.is_file():
            cached = json.loads(chunk_meta.read_text(encoding="utf-8")); values = np.load(chunk_path)
            if cached.get("text_sha256") != digest(batch) or tuple(values.shape) != (len(batch), dimension): raise ValueError("embedding chunk identity/shape mismatch")
            # These are cumulative generation receipts, not merely calls made
            # by the current resumed process.  Preserve already completed
            # endpoint work in the final metadata.
            return values, 1, int((cached.get("usage") or {}).get("total_tokens", 0))
        else:
            response = _post_json("/embeddings", {"model": model, "input": batch, "encoding_format": "float"})
            rows = sorted(response["data"], key=lambda row: int(row["index"]))
            if [int(row["index"]) for row in rows] != list(range(len(batch))): raise ValueError("embedding indices are not contiguous")
            values = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            if values.shape != (len(batch), dimension) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) == 0).any(): raise ValueError("invalid Qwen embedding response")
            values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            temp = chunk_path.with_suffix(".npy.tmp")
            with temp.open("wb") as stream: np.save(stream, values, allow_pickle=False)
            temp.replace(chunk_path); chunk_meta.write_text(json.dumps({"text_sha256": digest(batch), "array_sha256": _sha256(chunk_path), "usage": response.get("usage")}, sort_keys=True), encoding="utf-8")
            return values, 1, int((response.get("usage") or {}).get("total_tokens", 0))

    if workers == 1:
        completed = map(load_or_fetch, jobs)
    else:
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qwen-embedding")
        completed = executor.map(load_or_fetch, jobs)
    try:
        for values, new_requests, new_tokens in completed:
            chunks.append(values); requests += new_requests; reported_tokens += new_tokens
    finally:
        if workers != 1: executor.shutdown(wait=True, cancel_futures=True)
    result = np.concatenate(chunks); temp = array_path.with_suffix(".npy.tmp")
    with temp.open("wb") as stream: np.save(stream, result, allow_pickle=False)
    temp.replace(array_path); metadata = {"operation_sha256": op_hash, **operation, "array_sha256": _sha256(array_path), "requests": requests, "reported_tokens": reported_tokens, "failures": failures, "workers": workers, "wall_seconds": time.monotonic() - started, "cache_hit": False}; metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"); manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    return result, metadata


def run_dense_and_c02(run_dir: Path, *, release_root: Path = RELEASE, stage_name: str = "dense_and_c02", cache_root: Path | None = None) -> dict[str, Any]:
    """Continue after the lexical freeze in a separate, idempotent stage."""
    run_dir = run_dir.resolve(); stage = run_dir / stage_name; stage.mkdir(parents=True, exist_ok=True); cache_root = (cache_root or (stage / "embedding_cache")).resolve()
    result_path = stage / "result.json"
    if result_path.is_file(): return json.loads(result_path.read_text(encoding="utf-8"))
    method_q = release_root.resolve() / "method/queries.jsonl"; method_c = release_root.resolve() / "method/candidates.jsonl"; ref_path = release_root.resolve() / "evaluation/reference.jsonl"
    raw = _jsonl(method_q); allowed = [r for r in raw if r.get("split") in {"train", "development"}]; dev = [r for r in allowed if r.get("split") == "development"]; train = [r for r in allowed if r.get("split") == "train"]
    candidates = _jsonl(method_c); candidate_ids = [str(r["candidate_id"]) for r in candidates]; candidate_texts = [str(r["text"]) for r in candidates]; dev_ids = [str(r["query_id"]) for r in dev]; train_ids = {str(r["query_id"]) for r in train}
    if len(dev) != 1344 or len(train) != 6095 or len(candidates) != 9584: raise ValueError("unexpected N06-MP method population")
    # Train positives are admitted only for optimization; test rows are never retained.
    train_refs: list[dict[str, Any]] = []
    with ref_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line); qid = str(row.get("query_id"))
            if qid not in train_ids: continue
            train_refs.append({"query_id": qid, "group_id": str(row["group_id"]), "positive_candidate_ids": [str(x) for x in row["positive_candidate_ids"]]})
    if len(train_refs) != len(train): raise ValueError("train positives do not align with method queries")
    query_order = train + dev; query_texts = [str(r["text"]) for r in query_order]; query_ids = [str(r["query_id"]) for r in query_order]; query_vectors, query_meta = embed_qwen_cached(cache_root, "queries", query_texts, instruction=QUERY_INSTRUCTION); candidate_vectors, candidate_meta = embed_qwen_cached(cache_root, "candidates", candidate_texts, instruction=None)
    np.save(stage / "query_vectors.npy", query_vectors, allow_pickle=False); np.save(stage / "candidate_vectors.npy", candidate_vectors, allow_pickle=False)
    qpos = {qid: i for i, qid in enumerate(query_ids)}; dev_slice = np.asarray([qpos[qid] for qid in dev_ids]); q_dev = query_vectors[dev_slice]; train_q = query_vectors[:len(train)]; c = candidate_vectors
    lexical: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(run_dir / "rankings.jsonl"):
        if row.get("method") in {"char_tfidf", "word_tfidf"}: lexical[str(row["method"])].append(row)
    def by_query(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows: output[str(row["query_id"])].append(dict(row))
        return output
    char_by = by_query(lexical["char_tfidf"])
    reusable_dense = run_dir / "dense_and_c02" / "rankings_dense_and_controls.jsonl"
    if reusable_dense.is_file():
        cached_dense = [row for row in _iter_jsonl(reusable_dense) if row.get("method") == "dense_identity_4096d"]
        dense_by = by_query(cached_dense)
        if set(dense_by) != set(dev_ids) or any(len(dense_by[qid]) != TOP_K for qid in dev_ids): raise ValueError("reusable dense ranking is incomplete")
    else:
        dense_ranked = _rank_vectors(q_dev, c, candidate_ids); dense_by = {qid: rows for qid, rows in zip(dev_ids, dense_ranked)}
    rankings: list[dict[str, Any]] = []
    treatment_by_query: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for qid, rows in dense_by.items(): treatment_by_query[qid]["dense_identity_4096d"] = rows
    for qid, rows in char_by.items():
        drows = dense_by[qid]; treatment_by_query[qid]["rrf_char_dense"] = fuse_rrf(rows, drows); treatment_by_query[qid]["calibrated_char_dense"] = fuse_calibrated(rows, drows)
    projection = fixed_projection(); q_proj = q_dev @ projection; c_proj = c @ projection; q_proj /= np.maximum(np.linalg.norm(q_proj, axis=1, keepdims=True), 1e-12); c_proj /= np.maximum(np.linalg.norm(c_proj, axis=1, keepdims=True), 1e-12)
    for qid, rows in zip(dev_ids, _rank_vectors(q_proj, c_proj, candidate_ids)): treatment_by_query[qid]["projection_fixed_256d"] = rows
    train_q_proj = train_q @ projection; train_q_proj /= np.maximum(np.linalg.norm(train_q_proj, axis=1, keepdims=True), 1e-12)
    candidate_position = {cid: i for i, cid in enumerate(candidate_ids)}
    train_pos = []
    pos_map = {str(r["query_id"]): [candidate_position[cid] for cid in r["positive_candidate_ids"] if cid in candidate_position] for r in train_refs}
    for row in train: train_pos.append(pos_map.get(str(row["query_id"]), []))
    train_targets = build_train_targets(train, train_refs, c, candidate_ids)
    ridge = fit_ridge(train_q, train_targets, projection)
    import torch
    torch.set_num_threads(min(16, torch.get_num_threads()))
    # Associate as (candidate·Wᵀ) first: same exact 4,096d ridge score, far
    # less work than forming (query·W)·candidateᵀ for the full pool.
    ridge_candidates = torch.from_numpy(c.astype(np.float32)) @ torch.from_numpy(ridge.astype(np.float32)).T
    ridge_scores = (torch.from_numpy(q_proj.astype(np.float32)) @ ridge_candidates.T).numpy()
    ridge_ranked = []
    for values in ridge_scores:
        order = np.lexsort((np.asarray(candidate_ids, dtype=object), -values))[:TOP_K]; ridge_ranked.append([{"candidate_id": candidate_ids[i], "rank": rank, "score": float(values[i])} for rank, i in enumerate(order, 1)])
    for qid, rows in zip(dev_ids, ridge_ranked): treatment_by_query[qid]["ridge_256_to_4096"] = rows
    residual_meta: dict[str, Any] = {}
    for rank in (8, 16):
        u, v = fit_low_rank_residual(train_q, c, train_pos, projection, rank=rank, epochs=1, negatives=2, seed=BOOTSTRAP_SEED + rank); transform = np.eye(PROJECTION_DIM) + u @ v.T; transformed = q_proj @ transform; values = (torch.from_numpy(transformed.astype(np.float32)) @ torch.from_numpy(c_proj.astype(np.float32)).T).numpy(); output = []
        for scores in values:
            order = np.lexsort((np.asarray(candidate_ids, dtype=object), -scores))[:TOP_K]; output.append([{"candidate_id": candidate_ids[i], "rank": pos, "score": float(scores[i])} for pos, i in enumerate(order, 1)])
        name = f"residual_low_rank_{rank}"; residual_meta[name] = {"rank": rank, "u_shape": list(u.shape), "v_shape": list(v.shape), "loss": "sampled multi-positive hinge; train only"}
        for qid, rows in zip(dev_ids, output): treatment_by_query[qid][name] = rows
    method_metrics: dict[str, Any] = {}; method_groups: dict[str, dict[str, list[float]]] = {}
    def treatment_rows() -> Iterable[dict[str, Any]]:
        for qid, per_q in treatment_by_query.items():
            for method, rows in per_q.items():
                for row in rows: yield {"method": method, "query_id": qid, **row}
    # Method phase ends here: persist and hash all dense/control rankings before
    # opening the evaluator-side development reference.
    _write_jsonl(stage / "rankings_dense_and_controls.jsonl", treatment_rows())
    ranking_hash = _sha256(stage / "rankings_dense_and_controls.jsonl")
    write_json(stage / "ranking_freeze.json", {"schema_version": 1, "ranking_sha256": ranking_hash, "query_count": len(dev), "candidate_count": len(candidates), "test_opened": False, "evaluation_started": False})
    refs_all = _load_dev_references(ref_path, set(dev_ids))
    for method in sorted({name for per_q in treatment_by_query.values() for name in per_q}):
        current = {qid: per_q[method] for qid, per_q in treatment_by_query.items()}; metrics, details, groups = evaluate(current, refs_all); method_metrics[method] = metrics; method_groups[method] = groups
        _write_jsonl(stage / f"metrics_per_query_{method}.jsonl", details)
    freeze = json.loads((stage / "ranking_freeze.json").read_text(encoding="utf-8")); freeze["evaluation_started"] = True; write_json(stage / "ranking_freeze.json", freeze)
    pool = pool_recall(refs_all, char_by, dense_by); bootstraps = {method: paired_bootstrap(method_groups["dense_identity_4096d"], method_groups[method]) for method in sorted(method_groups) if method != "dense_identity_4096d"}
    metrics_output = {"primary_metric": "group_macro_ndcg_at_10", "secondary_metric": "recall_at_20", "methods": method_metrics, "char_dense_pool": pool, "paired_bootstrap_vs_dense": bootstraps}
    write_json(stage / "metrics_dense_and_controls.json", metrics_output); write_json(stage / "embedding_metadata.json", {"model": EMBEDDING_MODEL, "dimension": VECTOR_DIM, "query_instruction": QUERY_INSTRUCTION, "query_cache": query_meta, "candidate_cache": candidate_meta, "train_queries": len(train), "development_queries": len(dev), "candidate_count": len(candidates)})
    write_json(stage / "resources.json", {"model_calls": int(query_meta.get("requests", 0) + candidate_meta.get("requests", 0)), "new_embedding_inputs": len(query_order) + len(candidates), "reported_tokens": int(query_meta.get("reported_tokens", 0) + candidate_meta.get("reported_tokens", 0)), "test_opened": False, "remote_endpoint": os.environ.get("VLLM_EMBEDDINGS_BASE_URL", "http://localhost:8000/embeddings/v1")})
    contract = {"schema_version": 2, "task": "N06-MP C01/C02", "release_id": "n06_mp_sjf_luna_a_v1", "model": EMBEDDING_MODEL, "dimension": VECTOR_DIM, "query_instruction": QUERY_INSTRUCTION, "pool": "all 9,584 candidates", "methods": sorted(method_metrics), "char_dense_union_only": True, "pool_recall_reported_separately": True, "same_candidate_pool_for_rerankers": True, "train_only_optimization": True, "multipositive_loss": residual_meta, "test_opened": False, "reranker_instruct": "not_run; separate budget required", "input_hashes": {"method_queries": _sha256(method_q), "method_candidates": _sha256(method_c), "lexical_rankings": _sha256(run_dir / "rankings.jsonl")}}
    contract["contract_sha256"] = digest(contract); write_json(stage / "contract.json", contract)
    result = {"schema_version": 2, "status": "dense_and_c02_completed_development", "test_opened": False, "metrics": metrics_output, "resources": json.loads((stage / "resources.json").read_text(encoding="utf-8")), "output_hashes": {name: _sha256(stage / name) for name in ("contract.json", "metrics_dense_and_controls.json", "rankings_dense_and_controls.jsonl", "embedding_metadata.json", "resources.json")}}
    write_json(result_path, result); return result


def finalize_dense_stage(stage: Path, *, release_root: Path = RELEASE) -> dict[str, Any]:
    """Finalize a ranking-complete stage without reranking or reopening method inputs."""
    stage = stage.resolve(); release_root = release_root.resolve(); freeze = json.loads((stage / "ranking_freeze.json").read_text(encoding="utf-8"))
    if not freeze.get("ranking_sha256") or not freeze.get("evaluation_started"): raise ValueError("dense ranking freeze is missing")
    method_q = release_root / "method/queries.jsonl"; ref_path = release_root / "evaluation/reference.jsonl"; dev_ids = {str(row["query_id"]) for row in _iter_jsonl(method_q) if row.get("split") == "development"}; refs = _load_dev_references(ref_path, dev_ids)
    metrics: dict[str, Any] = {}; groups: dict[str, dict[str, list[float]]] = {}; method_names = [p.name[len("metrics_per_query_"):-len(".jsonl")] for p in stage.glob("metrics_per_query_*.jsonl")]
    for method in sorted(method_names):
        details = _jsonl(stage / f"metrics_per_query_{method}.jsonl"); per_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in details: per_group[str(row["group_id"])].append(row)
        metrics[method] = {"queries": len(details), "groups": len(per_group), "multipositive_queries": sum(r.get("positive_count", 0) > 1 for r in details), "mean_positive_count": float(np.mean([r.get("positive_count", 0) for r in details])) if details else 0.0, "recall_at_10": float(np.mean([r["recall_at_10"] for r in details])) if details else 0.0, "recall_at_20": float(np.mean([r["recall_at_20"] for r in details])) if details else 0.0, "recall_at_100": float(np.mean([r["recall_at_100"] for r in details])) if details else 0.0, "ndcg_at_10": float(np.mean([r["ndcg_at_10"] for r in details])) if details else 0.0, "group_macro_recall_at_20": float(np.mean([np.mean([r["recall_at_20"] for r in rows]) for rows in per_group.values()])) if per_group else 0.0, "group_macro_ndcg_at_10": float(np.mean([np.mean([r["ndcg_at_10"] for r in rows]) for rows in per_group.values()])) if per_group else 0.0}; groups[method] = {group: [float(row["recall_at_20"]) for row in rows] for group, rows in per_group.items()}
    char_by: dict[str, list[dict[str, Any]]] = defaultdict(list); dense_by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(stage.parent / "rankings.jsonl"):
        if row.get("method") == "char_tfidf": char_by[str(row["query_id"])].append(row)
    for row in _iter_jsonl(stage / "rankings_dense_and_controls.jsonl"):
        if row.get("method") == "rrf_char_dense": continue
        if row.get("method") == "char_tfidf": char_by[str(row["query_id"])].append(row)
        elif row.get("method") == "dense_identity_4096d": dense_by[str(row["query_id"])].append(row)
    pool = pool_recall(refs, char_by, dense_by); bootstraps = {method: paired_bootstrap(groups["dense_identity_4096d"], groups[method]) for method in sorted(groups) if method != "dense_identity_4096d"}
    output = {"primary_metric": "group_macro_ndcg_at_10", "secondary_metric": "recall_at_20", "methods": metrics, "char_dense_pool": pool, "paired_bootstrap_vs_dense": bootstraps}; write_json(stage / "metrics_dense_and_controls.json", output)
    metadata_path = stage / "embedding_metadata.json"; metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {"model": EMBEDDING_MODEL, "dimension": VECTOR_DIM, "query_instruction": QUERY_INSTRUCTION, "train_queries": 6095, "development_queries": 1344, "candidate_count": 9584, "cache_reused": True, "cache_note": "verified chunk cache keyed by text/instruction/model/dimension/batch"}; write_json(metadata_path, metadata); resources = {"model_calls": int(metadata.get("query_cache", {}).get("requests", 0) + metadata.get("candidate_cache", {}).get("requests", 0)), "new_embedding_inputs": int(metadata.get("train_queries", 0) + metadata.get("development_queries", 0) + metadata.get("candidate_count", 0)), "reported_tokens": int(metadata.get("query_cache", {}).get("reported_tokens", 0) + metadata.get("candidate_cache", {}).get("reported_tokens", 0)), "test_opened": False, "finalized_without_reranking": True}; write_json(stage / "resources.json", resources)
    contract = {"schema_version": 2, "task": "N06-MP C01/C02", "release_id": "n06_mp_sjf_luna_a_v1", "model": EMBEDDING_MODEL, "dimension": VECTOR_DIM, "query_instruction": QUERY_INSTRUCTION, "pool": "all 9,584 candidates", "methods": sorted(metrics), "char_dense_union_only": True, "pool_recall_reported_separately": True, "same_candidate_pool_for_rerankers": True, "train_only_optimization": True, "multipositive_loss": {"residual_low_rank_8": "sampled multi-positive hinge", "residual_low_rank_16": "sampled multi-positive hinge"}, "test_opened": False, "reranker_instruct": "not_run; separate budget required", "input_hashes": {"method_queries": _sha256(method_q), "lexical_rankings": _sha256(stage.parent / "rankings.jsonl"), "ranking_dense_controls": _sha256(stage / "rankings_dense_and_controls.jsonl")}}; contract["contract_sha256"] = digest(contract); write_json(stage / "contract.json", contract)
    result = {"schema_version": 2, "status": "dense_and_c02_completed_development", "test_opened": False, "metrics": output, "resources": resources, "output_hashes": {name: _sha256(stage / name) for name in ("contract.json", "metrics_dense_and_controls.json", "rankings_dense_and_controls.jsonl", "embedding_metadata.json", "resources.json")}}; write_json(stage / "result.json", result); return result


def run(run_dir: Path, *, release_root: Path = RELEASE) -> dict[str, Any]:
    started = time.monotonic(); release_root = release_root.resolve(); run_dir = run_dir.resolve()
    if (run_dir / "result.json").is_file(): return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    method_q = release_root / "method/queries.jsonl"; method_c = release_root / "method/candidates.jsonl"; evaluation_ref = release_root / "evaluation/reference.jsonl"
    if not all(p.is_file() for p in (method_q, method_c, evaluation_ref)): raise FileNotFoundError("N06-MP release inputs missing")
    raw_queries = _jsonl(method_q); queries = [r for r in raw_queries if r.get("split") == "development"]
    candidates = _jsonl(method_c); ids = [str(r["candidate_id"]) for r in candidates]; texts = [str(r["text"]) for r in candidates]
    if len(queries) != 1344 or len(candidates) != 9584: raise ValueError("unexpected N06-MP population")
    all_rankings: list[dict[str, Any]] = []; query_texts = [str(r["text"]) for r in queries]; query_ids = [str(r["query_id"]) for r in queries]
    for name, analyzer in (("word_tfidf", "word"), ("char_tfidf", "char")):
        ranked = rank_texts(query_texts, ids, texts, analyzer=analyzer)
        for qid, rows in zip(query_ids, ranked): all_rankings.extend({"method": name, "query_id": qid, **row} for row in rows)
    run_dir.mkdir(parents=True, exist_ok=True); _write_jsonl(run_dir / "rankings.jsonl", all_rankings)
    rank_hash = _sha256(run_dir / "rankings.jsonl")
    write_json(run_dir / "ranking_freeze.json", {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "task": "N06-MP", "ranking_sha256": rank_hash, "query_count": len(queries), "candidate_count": len(candidates), "test_opened": False, "evaluation_started": False})
    dev_ids = set(query_ids); refs = _load_dev_references(evaluation_ref, dev_ids)
    metrics, details, group_values = {}, {}, {}
    for method in ("word_tfidf", "char_tfidf"):
        by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rankings:
            if row["method"] == method: by_query[str(row["query_id"])].append(row)
        metrics[method], detail, group = evaluate(by_query, refs); details[method] = detail; group_values[method] = group
        _write_jsonl(run_dir / f"metrics_per_query_{method}.jsonl", detail)
    metrics["bootstrap_word_minus_char_recall_at_20"] = paired_bootstrap(group_values["char_tfidf"], group_values["word_tfidf"])
    freeze = json.loads((run_dir / "ranking_freeze.json").read_text(encoding="utf-8")); freeze["evaluation_started"] = True; write_json(run_dir / "ranking_freeze.json", freeze)
    write_json(run_dir / "metrics.json", metrics)
    resources = {"wall_seconds": time.monotonic() - started, "remote_calls": 0, "model_calls": 0, "new_embedding_inputs": 0, "candidate_count": len(candidates), "development_queries": len(queries), "evaluation_reference_rows_retained": len(refs), "money_usd": None}
    write_json(run_dir / "resources.json", resources)
    contract = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "task": "N06-MP", "methods": ["word_tfidf", "char_tfidf"], "release_id": "n06_mp_sjf_luna_a_v1", "candidate_count": len(candidates), "development_query_count": len(queries), "pool": "all 9,584 candidates", "primary_metric": "group_macro_recall_at_20", "multipositive_evaluation": True, "bootstrap_iterations": BOOTSTRAP, "bootstrap_seed": BOOTSTRAP_SEED, "method_inputs": ["method/queries.jsonl", "method/candidates.jsonl"], "evaluation_phase": "development rows filtered from custodial evaluation/reference.jsonl after ranking freeze", "test_opened": False, "input_hashes": {"method_queries": _sha256(method_q), "method_candidates": _sha256(method_c)}, "ranking_sha256": rank_hash,
                "dense_and_adapter_controls": {"status": "not_run_incompatible_cache", "required": ["identity_4096d", "projection_fixed", "ridge", "low_rank_residual_multipositive"], "reason": "old N06 vectors have 885 development queries/9,595 candidates; N06-MP has 1,344 development queries/9,584 candidates", "new_embedding_inputs": 0, "reranker_instruct": "not_run; no compatible cached query vectors"}}
    contract["contract_sha256"] = digest(contract); write_json(run_dir / "contract.json", contract)
    result = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, "task": "N06-MP", "status": "development_completed", "test_opened": False, "independent_confirmation": False, "legal_truth": False, "metrics": metrics, "resources": resources, "output_hashes": {name: _sha256(run_dir / name) for name in ("contract.json", "rankings.jsonl", "ranking_freeze.json", "metrics.json", "resources.json")}}
    write_json(run_dir / "result.json", result); return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", type=Path, required=True); parser.add_argument("--release-root", type=Path, default=RELEASE); args = parser.parse_args(argv)
    print(json.dumps(run(args.run_dir, release_root=args.release_root), ensure_ascii=False, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
