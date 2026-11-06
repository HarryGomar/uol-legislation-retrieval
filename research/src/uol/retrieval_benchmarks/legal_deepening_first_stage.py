"""First-stage retrieval for repaired R02/R03.

The scorer reads only method-safe query views and the shared 36,887-candidate
article universe.  References are opened after rankings are frozen.
"""

from __future__ import annotations

import argparse
import collections
import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import psutil
# On this Windows CUDA runtime, Arrow's bundled native libraries loaded first
# can make SentencePiece crash during DLL initialisation.  Establish the
# tokenizer runtime before importing Arrow; the inverse order is covered by a
# recorded capability failure in the study ledger.
try:  # the shared CPU environment intentionally does not require this package
    import sentencepiece  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - exercised by the CPU test runtime
    sentencepiece = None  # type: ignore[assignment]
import pyarrow.parquet as pq
from scipy import sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from uol.retrieval_benchmarks.common import (
    ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_bytes,
    digest,
    sha256_file,
    utc_now,
)
from uol.retrieval_benchmarks.legal_deepening import (
    CacheIdentity,
    DEFAULT_RELEASE,
    METHOD_SPLITS,
    MethodSafeAdapter,
    multipositive_metrics,
    normalize_legal_input_v1,
)


PROGRAM_ID = "legal_retrieval_deepening_20260917"
RUN_ROOT = ROOT / "experiments/runs/legal_retrieval_deepening_20260917"
DEFAULT_RUN = RUN_ROOT / "r02_r03_first_stage_v1"
DEFAULT_CACHE = ROOT / ".cache/legal_retrieval_deepening_20260917"
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_DIMENSION = 4096
QWEN_TOKENIZER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
QWEN_DISCOVERY_SHA256 = "f367256fa0ff86fef2bcfb8e198b21bcef46bb559f25b912a54b64459c1c82f4"
QWEN_REVISION = (
    f"endpoint-discovery:{QWEN_DISCOVERY_SHA256}:weights-undisclosed;"
    f"tokenizer:{QWEN_TOKENIZER_REVISION}"
)
QWEN_INSTRUCTION = (
    "Dada una consulta jurídica en español, recupera el artículo legal unido "
    "por la relación documental publicada. La cita puede estar oculta. "
    "Distingue instrumento y número de artículo."
)
GEMMA_MODEL = "wilfredomartel/embeddinggemma-300m-legal-spanish-420k-v2"
GEMMA_REVISION = "47fd23052f7df1ad415b0e082280a1144077a8fe"
EULAW_MODEL = "quanfire-ai/embed-eulaw-multi"
EULAW_REVISION = "b4b940408a883de7ff0b05e887c426aa2000d293"
EULAW_BASE = "intfloat/multilingual-e5-small"
EULAW_BASE_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
TOP_K = 200
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260917
CLAUSE_CONTRACT = "title-prefixed-legal-clause-v1:max900:min80"
ARTICLE_CONTRACT = "whole-article-title-plus-body-v1"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _candidate_rows(release: Path) -> list[dict[str, Any]]:
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    spec = manifest["candidate_universe"]
    path = release / spec["path"]
    if path.stat().st_size != int(spec["bytes"]) or sha256_file(path) != spec["sha256"]:
        raise ValueError("shared candidate universe receipt mismatch")
    rows = pq.read_table(path).to_pylist()
    if len(rows) != 36887 or len({row["candidate_id"] for row in rows}) != 36887:
        raise ValueError("the full 36,887-candidate universe is required")
    return rows


def article_text(row: Mapping[str, Any]) -> str:
    return "\n".join(
        value.strip()
        for value in (str(row.get("title") or ""), str(row.get("text") or ""))
        if value.strip()
    )


_CLAUSE_BOUNDARY = re.compile(r"(?<=[.;:!?])\s+|\n+")


def clause_chunks(row: Mapping[str, Any]) -> list[str]:
    title = str(row.get("title") or "").strip()
    body = str(row.get("text") or "").strip()
    pieces = [piece.strip() for piece in _CLAUSE_BOUNDARY.split(body) if piece.strip()]
    chunks: list[str] = []
    current = ""
    for piece in pieces or [body]:
        if current and len(current) + 1 + len(piece) > 900:
            chunks.append(current)
            current = piece
        else:
            current = f"{current} {piece}".strip()
        if len(current) >= 900:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    if len(chunks) > 1 and len(chunks[-1]) < 80:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()
    return [f"{title}\n{chunk}".strip() for chunk in chunks if chunk] or [title]


def candidate_representation(
    candidates: Sequence[Mapping[str, Any]], representation: str
) -> tuple[list[str], np.ndarray]:
    if representation == "article":
        return [article_text(row) for row in candidates], np.arange(len(candidates), dtype=np.int32)
    if representation != "clause":
        raise ValueError(f"unknown representation: {representation}")
    texts: list[str] = []
    owners: list[int] = []
    for index, row in enumerate(candidates):
        chunks = clause_chunks(row)
        texts.extend(chunks)
        owners.extend([index] * len(chunks))
    return texts, np.asarray(owners, dtype=np.int32)


def _aggregate_chunk_scores(values: np.ndarray, owners: np.ndarray, candidate_count: int) -> np.ndarray:
    if len(owners) == candidate_count and np.array_equal(owners, np.arange(candidate_count)):
        return values.astype(np.float32, copy=False)
    output = np.full((values.shape[0], candidate_count), -np.inf, dtype=np.float32)
    for query_index in range(values.shape[0]):
        np.maximum.at(output[query_index], owners, values[query_index])
    return output


def _top_rankings(
    scores: np.ndarray,
    query_rows: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    method: str,
    representation: str,
    top_k: int = TOP_K,
) -> list[dict[str, Any]]:
    candidate_array = np.asarray(candidate_ids, dtype=str)
    if scores.shape != (len(query_rows), len(candidate_array)):
        raise ValueError("score matrix shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("non-finite retrieval scores")
    rows: list[dict[str, Any]] = []
    depth = min(top_k, len(candidate_array))
    for query, values in zip(query_rows, scores, strict=True):
        pool = np.argpartition(-values, depth - 1)[:depth]
        order = pool[np.lexsort((candidate_array[pool], -values[pool]))]
        rows.extend(
            {
                "query_id": str(query["query_id"]),
                "candidate_id": str(candidate_array[index]),
                "rank": rank,
                "score": float(values[index]),
                "method": method,
                "representation": representation,
                "split": str(query["split"]),
            }
            for rank, index in enumerate(order, start=1)
        )
    return rows


def _reference_metrics(
    rankings: Sequence[Mapping[str, Any]], references: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_query: dict[str, list[str]] = collections.defaultdict(list)
    for row in sorted(rankings, key=lambda item: (str(item["query_id"]), int(item["rank"]))):
        by_query[str(row["query_id"])].append(str(row["candidate_id"]))
    details = []
    for reference in references:
        query_id = str(reference["query_id"])
        values = multipositive_metrics(
            reference["positive_candidate_ids"], by_query.get(query_id, []), k=10
        )
        details.append(
            {
                "query_id": query_id,
                "component_id": str(reference["component_id"]),
                "positive_count": len(reference["positive_candidate_ids"]),
                **values,
            }
        )
    return (
        {
            "queries": len(details),
            "independent_components": len({row["component_id"] for row in details}),
            "multipositive_queries": sum(row["positive_count"] > 1 for row in details),
            "component_macro_ndcg_at_10": float(
                np.mean([row["ndcg_at_10"] for row in details])
            ),
            "recall_at_20": float(np.mean([row["recall_at_20"] for row in details])),
            "recall_at_100": float(np.mean([row["recall_at_100"] for row in details])),
            "mrr": float(np.mean([row["mrr"] for row in details])),
            "strict_source_id_ndcg_at_10": float(
                np.mean([row["ndcg_at_10"] for row in details])
            ),
        },
        details,
    )


def _paired_interval(
    baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {str(row["component_id"]): float(row["ndcg_at_10"]) for row in baseline}
    right = {str(row["component_id"]): float(row["ndcg_at_10"]) for row in treatment}
    groups = sorted(set(left) & set(right))
    deltas = np.asarray([right[group] - left[group] for group in groups], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(deltas, size=(BOOTSTRAP_DRAWS, len(deltas)), replace=True).mean(axis=1)
    return {
        "baseline": "char_tfidf_article",
        "delta": float(deltas.mean()),
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "draws": BOOTSTRAP_DRAWS,
        "independent_components": len(groups),
        "seed": BOOTSTRAP_SEED,
    }


def _peak_rss() -> int:
    return int(psutil.Process().memory_info().rss)


def _lexical_scores(
    query_texts: Sequence[str],
    representation_texts: Sequence[str],
    owners: np.ndarray,
    candidate_count: int,
    family: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    if family in {"char_tfidf", "word_tfidf"}:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb" if family == "char_tfidf" else "word",
            ngram_range=tuple(config["ngram_range"]),
            min_df=int(config.get("min_df", 1)),
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
            token_pattern=None if family == "char_tfidf" else r"(?u)\b\w+\b",
            max_features=300_000,
            dtype=np.float32,
        )
        documents = vectorizer.fit_transform(representation_texts)
        queries = vectorizer.transform(query_texts)
        raw = (queries @ documents.T).toarray().astype(np.float32, copy=False)
    elif family == "bm25":
        vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
            max_features=300_000,
            dtype=np.float32,
        )
        counts = vectorizer.fit_transform(representation_texts).tocsr().astype(np.float32)
        n_documents = counts.shape[0]
        document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
        inverse = np.log1p(
            (n_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        ).astype(np.float32)
        lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
        average = max(float(lengths.mean()), 1e-9)
        row_ids = np.repeat(np.arange(n_documents), np.diff(counts.indptr))
        tf = counts.data
        k1, b = float(config["k1"]), float(config["b"])
        normalizer = k1 * (1.0 - b + b * lengths[row_ids] / average)
        counts.data = inverse[counts.indices] * (tf * (k1 + 1.0)) / (tf + normalizer)
        queries = vectorizer.transform(query_texts).astype(np.float32)
        queries.data[:] = 1.0
        raw = (queries @ counts.T).toarray().astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown lexical family: {family}")
    size = int(documents.data.nbytes + documents.indices.nbytes + documents.indptr.nbytes) if family != "bm25" else int(counts.data.nbytes + counts.indices.nbytes + counts.indptr.nbytes)
    return _aggregate_chunk_scores(raw, owners, candidate_count), size


def _query_bundle(release: Path, task_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        adapter = MethodSafeAdapter(release, task_id)
        for split in METHOD_SPLITS:
            for row in adapter.load_queries(split):
                rows.append({**row, "task_id": task_id})
    return rows


def _fold_validation_ids(release: Path, task_id: str) -> set[str]:
    return {
        str(row["query_id"])
        for row in _jsonl(release / f"tasks/{task_id}/training/component_folds.jsonl")
        if row["role"] == "internal_validation"
    }


def _evaluate_rows(
    release: Path,
    task_ids: Sequence[str],
    rankings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    output: dict[str, Any] = {}
    details_by_key: dict[str, list[dict[str, Any]]] = {}
    for task_id in task_ids:
        adapter = MethodSafeAdapter(release, task_id)
        task_rankings = [row for row in rankings if row.get("task_id", task_id) == task_id]
        for split in METHOD_SPLITS:
            split_rankings = [row for row in task_rankings if row["split"] == split]
            references = adapter.load_references(split)
            metrics, details = _reference_metrics(split_rankings, references)
            key = f"{task_id}:{split}"
            output[key] = metrics
            details_by_key[key] = details
        validation_ids = _fold_validation_ids(release, task_id)
        train_refs = [
            row for row in adapter.load_references("train") if row["query_id"] in validation_ids
        ]
        validation_rankings = [
            row for row in task_rankings if row["query_id"] in validation_ids
        ]
        output[f"{task_id}:internal_validation"], details_by_key[
            f"{task_id}:internal_validation"
        ] = _reference_metrics(validation_rankings, train_refs)
    return output, details_by_key


def _evaluate_methods(
    release: Path,
    task_ids: Sequence[str],
    rankings: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in rankings:
        grouped[str(row["method"])].append(row)
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for method, rows in sorted(grouped.items()):
        metrics[method], details[method] = _evaluate_rows(release, task_ids, rows)
    return metrics, details


def run_lexical(
    release: Path, run_dir: Path, cache_dir: Path, task_ids: Sequence[str]
) -> dict[str, Any]:
    del cache_dir
    stage = run_dir / "lexical"
    result_path = stage / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    stage.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(release)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    queries = _query_bundle(release, task_ids)
    query_texts = [str(row["text"]) for row in queries]
    trial_grid = {
        "char_tfidf": [
            {"ngram_range": [3, 5], "min_df": 2},
            {"ngram_range": [4, 6], "min_df": 2},
        ],
        "word_tfidf": [
            {"ngram_range": [1, 1], "min_df": 1},
            {"ngram_range": [1, 2], "min_df": 1},
        ],
        "bm25": [{"k1": 0.8, "b": 0.6}, {"k1": 1.2, "b": 0.9}],
    }
    rankings_all: list[dict[str, Any]] = []
    resources = []
    selection = {}
    peak_rss = _peak_rss()
    for representation in ("article", "clause"):
        texts, owners = candidate_representation(candidates, representation)
        for family, configs in trial_grid.items():
            trials = []
            trial_payloads = []
            for config in configs:
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                scores, index_size = _lexical_scores(
                    query_texts, texts, owners, len(candidates), family, config
                )
                trial_rows = _top_rankings(
                    scores, queries, candidate_ids, family, representation
                )
                for row, query in zip(
                    trial_rows,
                    (q for q in queries for _ in range(min(TOP_K, len(candidates)))),
                    strict=True,
                ):
                    row["task_id"] = query["task_id"]
                metrics, _ = _evaluate_rows(release, task_ids, trial_rows)
                selection_score = float(
                    np.mean(
                        [
                            metrics[f"{task}:internal_validation"][
                                "component_macro_ndcg_at_10"
                            ]
                            for task in task_ids
                        ]
                    )
                )
                trials.append(
                    {
                        "config": config,
                        "selection_score": selection_score,
                        "metrics": metrics,
                    }
                )
                trial_payloads.append((selection_score, canonical_bytes(config), trial_rows, scores))
                resources.append(
                    {
                        "method": family,
                        "representation": representation,
                        "config": config,
                        "wall_seconds": time.perf_counter() - started_wall,
                        "cpu_seconds": time.process_time() - started_cpu,
                        "peak_rss_bytes": _peak_rss(),
                        "index_bytes": index_size,
                        "candidate_count": len(candidates),
                        "representation_rows": len(texts),
                        "candidate_coverage": 1.0,
                        "gpu_seconds": 0.0,
                        "peak_vram_mib": 0.0,
                        "money": None,
                        "unknown_costs": ["local electricity"],
                    }
                )
                peak_rss = max(peak_rss, _peak_rss())
                del scores
                gc.collect()
            best = max(trial_payloads, key=lambda item: (item[0], bytes(255 - b for b in item[1])))
            best_config = next(
                trial["config"] for trial in trials if trial["selection_score"] == best[0]
            )
            method_id = f"{family}_{representation}"
            for row in best[2]:
                row["method"] = method_id
            rankings_all.extend(best[2])
            selection[method_id] = {
                "selected_config": best_config,
                "selection_metric": "mean R02/R03 internal-validation component-macro nDCG@10",
                "trials": trials,
            }
    atomic_write_jsonl(stage / "rankings.jsonl", rankings_all)
    metrics, details = _evaluate_methods(release, task_ids, rankings_all)
    result = {
        "schema_version": 1,
        "status": "completed",
        "stage": "lexical",
        "candidate_count": len(candidates),
        "all_candidates_represented": True,
        "selection": selection,
        "metrics": metrics,
        "resources": resources,
        "peak_rss_bytes": peak_rss,
        "rankings_sha256": sha256_file(stage / "rankings.jsonl"),
    }
    atomic_write_json(stage / "details.json", details)
    atomic_write_json(result_path, result)
    return result


def _write_cache_contract(
    path: Path,
    texts: Sequence[str],
    *,
    role: str,
    instruction: str,
    model_id: str,
    revision: str,
    dimension: int,
    dtype: str,
    chunking_contract: str,
) -> str:
    aggregate_hash = hashlib.sha256(
        b"".join(_sha_text(normalize_legal_input_v1(text)).encode("ascii") for text in texts)
    ).hexdigest()
    identity = CacheIdentity(
        source_text_sha256=aggregate_hash,
        role=role,
        instruction=instruction,
        model_id=model_id,
        revision=revision,
        dimension=dimension,
        normalization="l2",
        dtype=dtype,
        chunking_contract=chunking_contract,
    )
    payload = {"identity": identity.__dict__, "cache_key": identity.key(), "rows": len(texts)}
    atomic_write_json(path, payload)
    return identity.key()


def _save_float16(path: Path, values: np.ndarray) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as stream:
        np.save(stream, np.asarray(values, dtype=np.float16), allow_pickle=False)
    os.replace(temp, path)
    return path.stat().st_size


def _dense_scores(
    query_vectors: np.ndarray,
    representation_vectors: np.ndarray,
    owners: np.ndarray,
    candidate_count: int,
    *,
    batch_size: int = 32,
) -> np.ndarray:
    documents = np.asarray(representation_vectors, dtype=np.float32)
    output = np.empty((len(query_vectors), candidate_count), dtype=np.float32)
    for begin in range(0, len(query_vectors), batch_size):
        raw = np.asarray(query_vectors[begin : begin + batch_size], dtype=np.float32) @ documents.T
        output[begin : begin + len(raw)] = _aggregate_chunk_scores(
            raw, owners, candidate_count
        )
    return output


def _qwen_embeddings(
    cache: Path,
    kind: str,
    texts: Sequence[str],
    *,
    instruction: str | None,
    batch_size: int = 64,
    workers: int = 1,
) -> tuple[np.ndarray, dict[str, Any]]:
    from uol.retrieval_benchmarks.semantic_v2_c01_c02_mp import embed_qwen_cached

    return embed_qwen_cached(
        cache,
        kind,
        texts,
        instruction=instruction,
        model=QWEN_MODEL,
        dimension=QWEN_DIMENSION,
        batch_size=batch_size,
        workers=workers,
    )


def _qwen_generation_receipt(
    cache: Path, kind: str, current: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the original endpoint-generation receipt even on a cache hit.

    A resumed scoring pass must not make endpoint use disappear from the
    resource ledger merely because the vectors were generated in an earlier
    invocation.  The endpoint helper preserves the original manifest beside
    the array; validate its operation identity before using it.
    """
    path = cache / f"{kind}_cache_manifest.json"
    if not path.is_file():
        return dict(current)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    expected = current.get("operation_sha256")
    observed = persisted.get("operation_sha256")
    if expected and observed != expected:
        raise ValueError(f"Qwen cache receipt identity mismatch for {kind}")
    return persisted


def _truncate_qwen_inputs(
    texts: Sequence[str], *, max_tokens: int
) -> tuple[list[str], dict[str, Any]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        QWEN_MODEL, revision=QWEN_TOKENIZER_REVISION
    )
    output: list[str] = []
    truncated = 0
    original_token_total = 0
    retained_token_total = 0
    maximum = 0
    for text in texts:
        token_ids = tokenizer.encode(str(text), add_special_tokens=False)
        original_token_total += len(token_ids)
        maximum = max(maximum, len(token_ids))
        if len(token_ids) > max_tokens:
            truncated += 1
            token_ids = token_ids[:max_tokens]
        retained_token_total += len(token_ids)
        output.append(tokenizer.decode(token_ids, skip_special_tokens=True))
    return output, {
        "tokenizer_revision": QWEN_TOKENIZER_REVISION,
        "max_tokens": max_tokens,
        "rows": len(texts),
        "truncated_rows": truncated,
        "truncation_rate": truncated / len(texts) if texts else 0.0,
        "original_token_total": original_token_total,
        "retained_token_total": retained_token_total,
        "maximum_original_tokens": maximum,
    }


def run_qwen(
    release: Path, run_dir: Path, cache_dir: Path, task_ids: Sequence[str]
) -> dict[str, Any]:
    stage = run_dir / "qwen"
    result_path = stage / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    stage.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(release)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    queries = _query_bundle(release, task_ids)
    query_texts = [str(row["text"]) for row in queries]
    query_contract = _write_cache_contract(
        stage / "query_cache_contract.json",
        query_texts,
        role="query",
        instruction=QWEN_INSTRUCTION,
        model_id=QWEN_MODEL,
        revision=QWEN_REVISION,
        dimension=QWEN_DIMENSION,
        dtype="float16-storage-float32-score",
        chunking_contract="query-whole-v1:qwen-max7600",
    )
    qwen_query_texts, query_truncation = _truncate_qwen_inputs(
        query_texts, max_tokens=7600
    )
    query_vectors, query_meta = _qwen_embeddings(
        cache_dir / "qwen", f"query_{query_contract[:16]}_t7600", qwen_query_texts, instruction=QWEN_INSTRUCTION
    )
    query_index_bytes = _save_float16(stage / "query_vectors.f16.npy", query_vectors)
    rankings_all: list[dict[str, Any]] = []
    resources = []
    for representation in ("article", "clause"):
        texts, owners = candidate_representation(candidates, representation)
        contract_name = (
            ARTICLE_CONTRACT if representation == "article" else CLAUSE_CONTRACT
        ) + ":qwen-max8000"
        key = _write_cache_contract(
            stage / f"candidate_{representation}_cache_contract.json",
            texts,
            role="candidate",
            instruction="none",
            model_id=QWEN_MODEL,
            revision=QWEN_REVISION,
            dimension=QWEN_DIMENSION,
            dtype="float16-storage-float32-score",
            chunking_contract=contract_name,
        )
        qwen_texts, truncation = _truncate_qwen_inputs(texts, max_tokens=8000)
        atomic_write_json(stage / f"{representation}_truncation.json", truncation)
        started_wall, started_cpu = time.perf_counter(), time.process_time()
        endpoint_batch_size = 64 if representation == "article" else 128
        cache_kind = f"candidate_{representation}_{key[:16]}_t8000"
        if representation == "clause":
            cache_kind += "_b128"
        endpoint_cache = cache_dir / "qwen"
        vectors, metadata = _qwen_embeddings(
            endpoint_cache,
            cache_kind,
            qwen_texts,
            instruction=None,
            batch_size=endpoint_batch_size,
        )
        generation = _qwen_generation_receipt(endpoint_cache, cache_kind, metadata)
        index_bytes = _save_float16(stage / f"candidate_{representation}.f16.npy", vectors)
        scores = _dense_scores(query_vectors, vectors, owners, len(candidates), batch_size=16)
        rows = _top_rankings(scores, queries, candidate_ids, f"qwen_{representation}", representation)
        for row, query in zip(
            rows, (q for q in queries for _ in range(TOP_K)), strict=True
        ):
            row["task_id"] = query["task_id"]
        rankings_all.extend(rows)
        resources.append(
            {
                "method": f"qwen_{representation}",
                "wall_seconds": time.perf_counter() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu,
                "peak_rss_bytes": _peak_rss(),
                "index_bytes": index_bytes,
                "query_index_bytes": query_index_bytes,
                "candidate_count": len(candidates),
                "representation_rows": len(texts),
                "candidate_coverage": 1.0,
                "endpoint_inputs": len(texts),
                "endpoint_requests": generation.get("requests", 0),
                "endpoint_reported_tokens": generation.get("reported_tokens", 0),
                "endpoint_cache_hit_current": bool(metadata.get("cache_hit")),
                "endpoint_requests_this_invocation": (
                    0 if metadata.get("cache_hit") else generation.get("requests", 0)
                ),
                "endpoint_revision": QWEN_REVISION,
                "endpoint_batch_size": endpoint_batch_size,
                "truncation": truncation,
                "gpu_seconds": None,
                "peak_vram_mib": None,
                "money": None,
                "unknown_costs": ["endpoint billing", "endpoint GPU time", "local electricity"],
            }
        )
        del vectors, scores
        gc.collect()
    atomic_write_jsonl(stage / "rankings.jsonl", rankings_all)
    metrics, details = _evaluate_methods(release, task_ids, rankings_all)
    result = {
        "schema_version": 1,
        "status": "completed",
        "stage": "qwen",
        "model": QWEN_MODEL,
        "served_revision": None,
        "endpoint_discovery_sha256": QWEN_DISCOVERY_SHA256,
        "revision_limitation": "the endpoint does not certify a weights or tokenizer revision",
        "instruction": QWEN_INSTRUCTION,
        "query_cache": query_meta,
        "query_truncation": query_truncation,
        "candidate_count": len(candidates),
        "all_candidates_represented": True,
        "metrics": metrics,
        "resources": resources,
        "rankings_sha256": sha256_file(stage / "rankings.jsonl"),
    }
    atomic_write_json(stage / "details.json", details)
    atomic_write_json(result_path, result)
    return result


def _encode_sentence_transformer(
    model_id: str,
    revision: str,
    texts: Sequence[str],
    *,
    role: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from sentence_transformers import SentenceTransformer

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = SentenceTransformer(
        model_id,
        revision=revision,
        device="cuda",
        model_kwargs={"dtype": torch.float32},
    )
    model.max_seq_length = max_length
    encoder: Callable[..., Any] = model.encode_query if role == "query" else model.encode_document
    vectors = encoder(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    torch.cuda.synchronize()
    metadata = {
        "gpu_seconds_wall": time.perf_counter() - started,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "finite": bool(np.isfinite(vectors).all()),
        "dtype": "float32",
        "max_length": max_length,
    }
    del model
    torch.cuda.empty_cache()
    if not metadata["finite"]:
        raise ValueError(f"{model_id} returned non-finite vectors")
    return np.asarray(vectors, dtype=np.float32), metadata


def _encode_eulaw(texts: Sequence[str], *, batch_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    from multilingual_embedding.embedding.neural.adapter import load_adapter
    from uol.retrieval_benchmarks import legal_deepening_first_stage as this_module
    from scripts.probe_legal_retrieval_models import pinned_eulaw_adapter

    del this_module
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    encoder, _ = load_adapter(pinned_eulaw_adapter(), device="cuda", local_files_only=True)
    chunks = []
    for begin in range(0, len(texts), batch_size):
        chunks.append(np.asarray(encoder.encode_batch(list(texts[begin : begin + batch_size]))))
    vectors = np.concatenate(chunks).astype(np.float32, copy=False)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    torch.cuda.synchronize()
    metadata = {
        "gpu_seconds_wall": time.perf_counter() - started,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "finite": bool(np.isfinite(vectors).all()),
        "dtype": "float32",
        "base_model": EULAW_BASE,
        "base_revision": EULAW_BASE_REVISION,
    }
    del encoder
    torch.cuda.empty_cache()
    if not metadata["finite"]:
        raise ValueError("EU-law adapter returned non-finite vectors")
    return vectors, metadata


def _load_or_encode_local(
    cache_path: Path,
    contract_path: Path,
    texts: Sequence[str],
    encode: Callable[[Sequence[str]], tuple[np.ndarray, dict[str, Any]]],
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    key = _write_cache_contract(contract_path, texts, **contract)
    meta_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and meta_path.is_file():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        values = np.load(cache_path, mmap_mode="r")
        if metadata.get("cache_key") != key or len(values) != len(texts):
            raise ValueError("local embedding cache identity mismatch")
        return np.asarray(values, dtype=np.float32), {**metadata, "cache_hit": True}
    values, metadata = encode(texts)
    index_bytes = _save_float16(cache_path, values)
    payload = {**metadata, "cache_key": key, "cache_hit": False, "index_bytes": index_bytes}
    atomic_write_json(meta_path, payload)
    return values, payload


def run_local_dense(
    release: Path, run_dir: Path, cache_dir: Path, task_ids: Sequence[str]
) -> dict[str, Any]:
    stage = run_dir / "local_dense"
    result_path = stage / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    stage.mkdir(parents=True, exist_ok=True)
    cache_root = cache_dir / "local_dense"
    cache_root.mkdir(parents=True, exist_ok=True)
    candidates = _candidate_rows(release)
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    queries = _query_bundle(release, task_ids)
    query_texts = [str(row["text"]) for row in queries]
    models = [
        {
            "short": "embeddinggemma",
            "id": GEMMA_MODEL,
            "revision": GEMMA_REVISION,
            "dimension": 768,
            "encode": lambda texts, role: _encode_sentence_transformer(
                GEMMA_MODEL,
                GEMMA_REVISION,
                texts,
                role=role,
                batch_size=16,
                max_length=512 if role == "candidate" else 384,
            ),
            "scope": "Spanish legal dense primary; FP32 required on GTX 1070",
        },
        {
            "short": "eulaw",
            "id": EULAW_MODEL,
            "revision": EULAW_REVISION,
            "dimension": 384,
            "encode": lambda texts, role: _encode_eulaw(texts, batch_size=32),
            "scope": "out-of-domain EU-law cross-lingual provision-transfer control",
        },
    ]
    rankings_all = []
    resources = []
    failures = []
    for spec in models:
        try:
            query_cache = cache_root / f"{spec['short']}_queries.f16.npy"
            query_contract = {
                "role": "query",
                "instruction": "query encoder" if spec["short"] == "embeddinggemma" else "shared EU-law encoder",
                "model_id": spec["id"],
                "revision": spec["revision"],
                "dimension": spec["dimension"],
                "dtype": "float32-compute-float16-storage",
                "chunking_contract": "query-whole-v1",
            }
            query_vectors, query_meta = _load_or_encode_local(
                query_cache,
                stage / f"{spec['short']}_query_contract.json",
                query_texts,
                lambda texts, fn=spec["encode"]: fn(texts, "query"),
                query_contract,
            )
            for representation in ("article", "clause"):
                texts, owners = candidate_representation(candidates, representation)
                contract_name = ARTICLE_CONTRACT if representation == "article" else CLAUSE_CONTRACT
                candidate_cache = cache_root / f"{spec['short']}_{representation}.f16.npy"
                candidate_contract = {
                    "role": "candidate",
                    "instruction": "document encoder" if spec["short"] == "embeddinggemma" else "shared EU-law encoder",
                    "model_id": spec["id"],
                    "revision": spec["revision"],
                    "dimension": spec["dimension"],
                    "dtype": "float32-compute-float16-storage",
                    "chunking_contract": contract_name,
                }
                started_wall, started_cpu = time.perf_counter(), time.process_time()
                candidate_vectors, candidate_meta = _load_or_encode_local(
                    candidate_cache,
                    stage / f"{spec['short']}_{representation}_contract.json",
                    texts,
                    lambda values, fn=spec["encode"]: fn(values, "candidate"),
                    candidate_contract,
                )
                scores = _dense_scores(
                    query_vectors, candidate_vectors, owners, len(candidates), batch_size=32
                )
                method = f"{spec['short']}_{representation}"
                rows = _top_rankings(scores, queries, candidate_ids, method, representation)
                for row, query in zip(
                    rows, (q for q in queries for _ in range(TOP_K)), strict=True
                ):
                    row["task_id"] = query["task_id"]
                rankings_all.extend(rows)
                resources.append(
                    {
                        "method": method,
                        "model_id": spec["id"],
                        "revision": spec["revision"],
                        "scope": spec["scope"],
                        "wall_seconds": time.perf_counter() - started_wall,
                        "cpu_seconds": time.process_time() - started_cpu,
                        "gpu_seconds_wall": (
                            0.0 if candidate_meta.get("cache_hit") else candidate_meta.get("gpu_seconds_wall")
                        ),
                        "peak_vram_mib": candidate_meta.get("peak_vram_mib"),
                        "peak_rss_bytes": _peak_rss(),
                        "index_bytes": candidate_meta.get("index_bytes", candidate_cache.stat().st_size),
                        "candidate_count": len(candidates),
                        "representation_rows": len(texts),
                        "candidate_coverage": 1.0,
                        "query_cache": query_meta,
                        "candidate_cache": candidate_meta,
                        "money": None,
                        "unknown_costs": ["local electricity"],
                    }
                )
                del candidate_vectors, scores
                gc.collect()
        except Exception as error:
            failures.append(
                {
                    "model_id": spec["id"],
                    "error_type": type(error).__name__,
                    "error": str(error)[:2000],
                    "attempted_batch": 16 if spec["short"] == "embeddinggemma" else 32,
                    "fallbacks_remaining": ["batch_1", "shorter_sequence", "CPU"],
                }
            )
    if rankings_all:
        atomic_write_jsonl(stage / "rankings.jsonl", rankings_all)
        metrics, details = _evaluate_methods(release, task_ids, rankings_all)
        rankings_hash = sha256_file(stage / "rankings.jsonl")
        atomic_write_json(stage / "details.json", details)
    else:
        metrics, rankings_hash = {}, None
    result = {
        "schema_version": 1,
        "status": "completed" if not failures else "completed_with_failures",
        "stage": "local_dense",
        "candidate_count": len(candidates),
        "all_candidates_represented": bool(rankings_all),
        "metrics": metrics,
        "resources": resources,
        "failures": failures,
        "rankings_sha256": rankings_hash,
    }
    atomic_write_json(result_path, result)
    return result


def _rankings_by_method(run_dir: Path) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for relative in ("lexical/rankings.jsonl", "qwen/rankings.jsonl", "local_dense/rankings.jsonl"):
        path = run_dir / relative
        if path.is_file():
            for row in _jsonl(path):
                output[str(row["method"])].append(row)
    return output


def _robust_z(values: Mapping[str, float]) -> dict[str, float]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    scale = max(1.4826 * mad, 1e-8)
    return {key: (float(value) - median) / scale for key, value in values.items()}


def run_fusion(release: Path, run_dir: Path, task_ids: Sequence[str]) -> dict[str, Any]:
    stage = run_dir / "fusion"
    result_path = stage / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    stage.mkdir(parents=True, exist_ok=True)
    by_method = _rankings_by_method(run_dir)
    required = {"char_tfidf_article", "word_tfidf_article", "bm25_article", "qwen_article"}
    missing = sorted(required - set(by_method))
    if missing:
        raise ValueError(f"fusion requires completed first stages: {missing}")
    methods = sorted(by_method)
    by_query_method: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    query_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for method, rows in by_method.items():
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            key = (str(row["task_id"]), str(row["query_id"]))
            grouped[key].append(row)
            query_meta[key] = row
        for key, values in grouped.items():
            by_query_method[(key[0], key[1], method)] = sorted(values, key=lambda row: int(row["rank"]))

    def fuse(weights: Mapping[str, float], label: str, mode: str) -> list[dict[str, Any]]:
        output = []
        for (task_id, query_id), meta in sorted(query_meta.items()):
            candidate_union = sorted(
                {
                    str(row["candidate_id"])
                    for method in methods
                    for row in by_query_method.get((task_id, query_id, method), [])
                }
            )
            scores = {candidate_id: 0.0 for candidate_id in candidate_union}
            for method in methods:
                weight = float(weights.get(method, 0.0))
                if weight == 0.0:
                    continue
                rows = by_query_method.get((task_id, query_id, method), [])
                if mode == "rrf":
                    values = {str(row["candidate_id"]): 1.0 / (60.0 + int(row["rank"])) for row in rows}
                    floor = 0.0
                else:
                    values = _robust_z({str(row["candidate_id"]): float(row["score"]) for row in rows})
                    floor = min(values.values(), default=0.0) - 1.0
                for candidate_id in candidate_union:
                    scores[candidate_id] += weight * values.get(candidate_id, floor)
            ordered = sorted(candidate_union, key=lambda candidate_id: (-scores[candidate_id], candidate_id))[:TOP_K]
            output.extend(
                {
                    "task_id": task_id,
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "rank": rank,
                    "score": scores[candidate_id],
                    "method": label,
                    "representation": "common_top200_union",
                    "split": meta["split"],
                }
                for rank, candidate_id in enumerate(ordered, start=1)
            )
        return output

    train_candidates = ["char_tfidf_article", "word_tfidf_article", "bm25_article", "qwen_article"]
    weight_trials = []
    for qwen_weight in (0.25, 0.5, 0.6, 0.75):
        lexical_weight = (1.0 - qwen_weight) / 3.0
        weights = {method: lexical_weight for method in train_candidates[:3]}
        weights["qwen_article"] = qwen_weight
        rows = fuse(weights, "robust_z_tuning", "robust_z")
        metrics, _ = _evaluate_rows(release, task_ids, rows)
        score = float(np.mean([metrics[f"{task}:internal_validation"]["component_macro_ndcg_at_10"] for task in task_ids]))
        weight_trials.append((score, qwen_weight, weights))
    selected = max(weight_trials, key=lambda item: (item[0], -item[1]))
    robust_rows = fuse(selected[2], "robust_z_convex_article", "robust_z")
    equal_weights = {method: 1.0 / len(methods) for method in methods}
    rrf_rows = fuse(equal_weights, "rrf_all_first_stages", "rrf")
    rankings = robust_rows + rrf_rows
    atomic_write_jsonl(stage / "rankings.jsonl", rankings)
    metrics, details = _evaluate_methods(release, task_ids, rankings)
    result = {
        "schema_version": 1,
        "status": "completed",
        "candidate_union_depth_per_method": TOP_K,
        "methods": methods,
        "robust_z": {
            "selected_weights": selected[2],
            "selection_score": selected[0],
            "trials": [
                {"selection_score": score, "qwen_weight": weight, "weights": weights}
                for score, weight, weights in weight_trials
            ],
        },
        "rrf": {"constant": 60.0, "weights": equal_weights},
        "metrics": metrics,
        "rankings_sha256": sha256_file(stage / "rankings.jsonl"),
    }
    atomic_write_json(stage / "details.json", details)
    atomic_write_json(result_path, result)
    return result


def summarize(run_dir: Path, release: Path, task_ids: Sequence[str]) -> dict[str, Any]:
    by_method = _rankings_by_method(run_dir)
    fusion_path = run_dir / "fusion/rankings.jsonl"
    if fusion_path.is_file():
        for row in _jsonl(fusion_path):
            by_method[str(row["method"])].append(row)
    method_metrics: dict[str, Any] = {}
    details: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for method, rankings in sorted(by_method.items()):
        metrics, method_details = _evaluate_rows(release, task_ids, rankings)
        method_metrics[method] = metrics
        details[method] = method_details
    baseline = details.get("char_tfidf_article", {})
    intervals = {}
    for method, method_details in details.items():
        if method == "char_tfidf_article":
            continue
        intervals[method] = {}
        for task_id in task_ids:
            key = f"{task_id}:development"
            if key in baseline and key in method_details:
                intervals[method][task_id] = _paired_interval(
                    baseline[key], method_details[key]
                )
    resources = []
    for stage in ("lexical", "qwen", "local_dense"):
        path = run_dir / stage / "result.json"
        if path.is_file():
            resources.extend(json.loads(path.read_text(encoding="utf-8")).get("resources", []))
    summary = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "status": "completed",
        "release": str(release),
        "release_manifest_sha256": sha256_file(release / "manifest.json"),
        "candidate_count": 36887,
        "all_article_candidates_embedded": any(row["method"] == "qwen_article" for row in resources),
        "methods": method_metrics,
        "paired_component_bootstrap_vs_char_article": intervals,
        "resources": resources,
        "scientific_status": "exploratory; R02/R03 source qrels and historical test were exposed",
        "test_scored": False,
        "money": None,
        "unknown_costs": ["MIT endpoint billing", "local electricity", "assistant compute"],
    }
    atomic_write_json(run_dir / "first_stage_summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("lexical", "qwen", "local", "fusion", "summarize"))
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--tasks", nargs="+", default=["R02", "R03"])
    args = parser.parse_args(argv)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM"):
        os.environ.setdefault(name, "1" if name != "TOKENIZERS_PARALLELISM" else "false")
    release, run_dir, cache_dir = args.release.resolve(), args.run_dir.resolve(), args.cache_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "lexical":
        result = run_lexical(release, run_dir, cache_dir, args.tasks)
    elif args.stage == "qwen":
        result = run_qwen(release, run_dir, cache_dir, args.tasks)
    elif args.stage == "local":
        result = run_local_dense(release, run_dir, cache_dir, args.tasks)
    elif args.stage == "fusion":
        result = run_fusion(release, run_dir, args.tasks)
    else:
        result = summarize(run_dir, release, args.tasks)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
