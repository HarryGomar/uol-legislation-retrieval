"""Exploratory retrieval experiments for the structured C01 statute pilot.

The promotion gate for this release did not pass.  This runner therefore
uses only train labels for method selection, evaluates the already public
development split, and never opens the sealed test references.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from uol.retrieval_benchmarks.common import ROOT, sha256_file, utc_now


RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"
RUN = ROOT / "experiments/runs/c01_statute_graph_retrieval_20260918/exploratory_v3"
CACHE = ROOT / ".cache/c01_statute_graph_retrieval_20260918/qwen"
ARTICLE_CACHE = ROOT / (
    ".cache/legal_retrieval_deepening_20260917/qwen/"
    "candidate_article_c98204d1d98058b2_t8000_375d744ce0f5875368fc.npy"
)
ARTICLE_CACHE_SHA256 = "b5655a78dbbee7e3d39882abb36695bba6ba7fc03b8cd4b0362ae1b1533c405c"
ARTICLE_TEXT_SHA256 = "9f06ee338c7c7536d659b5506327fca74fbd81382aa24b55580ad7551216c8e4"
QUERY_KIND = "structured_facts_default_t7600"
QUERY_INSTRUCTION = (
    "Dada una narración de hechos judiciales mexicanos, recupera el artículo "
    "federal que después aparece como cita documental en el análisis jurídico. "
    "La identidad de la ley y del artículo puede estar oculta; distingue la "
    "materia, el mecanismo jurídico, el instrumento y el número de artículo."
)
TOP_K = 200
BOOTSTRAP_DRAWS = 5000
BOOTSTRAP_SEED = 20260918
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"
QWEN_DIMENSION = 4096
QWEN_TOKENIZER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
QWEN_DISCOVERY_SHA256 = "f367256fa0ff86fef2bcfb8e198b21bcef46bb559f25b912a54b64459c1c82f4"
QWEN_REVISION = (
    f"endpoint-discovery:{QWEN_DISCOVERY_SHA256}:weights-undisclosed;"
    f"tokenizer:{QWEN_TOKENIZER_REVISION}"
)


def _qwen_embeddings(
    cache: Path,
    kind: str,
    texts: Sequence[str],
    *,
    instruction: str | None,
    batch_size: int,
    workers: int,
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
    path = cache / f"{kind}_cache_manifest.json"
    if not path.is_file():
        return dict(current)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    if persisted.get("operation_sha256") != current.get("operation_sha256"):
        raise ValueError("Qwen query cache receipt identity mismatch")
    return persisted


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _component_is_internal_validation(component_id: str) -> bool:
    value = int(hashlib.sha256(component_id.encode("utf-8")).hexdigest()[:8], 16)
    return value % 5 == 0


def _robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    median = np.median(values, axis=1, keepdims=True)
    mad = np.median(np.abs(values - median), axis=1, keepdims=True)
    scale = 1.4826 * mad
    std = np.std(values, axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0))
    return ((values - median) / scale).astype(np.float32, copy=False)


def _bm25_document_matrix(counts: sp.csr_matrix, k1: float = 1.2, b: float = 0.75) -> sp.csr_matrix:
    counts = counts.astype(np.float32, copy=True).tocsr()
    count_documents = counts.shape[0]
    document_frequency = np.asarray((counts > 0).sum(axis=0)).ravel()
    inverse_document_frequency = np.log1p(
        (count_documents - document_frequency + 0.5) / (document_frequency + 0.5)
    ).astype(np.float32)
    lengths = np.asarray(counts.sum(axis=1)).ravel().astype(np.float32)
    average_length = float(lengths.mean()) if len(lengths) else 1.0
    row_ids = np.repeat(np.arange(counts.shape[0]), np.diff(counts.indptr))
    term_frequency = counts.data
    normalizer = k1 * (1.0 - b + b * lengths[row_ids] / max(average_length, 1e-9))
    counts.data = inverse_document_frequency[counts.indices] * (
        term_frequency * (k1 + 1.0)
    ) / (term_frequency + normalizer)
    return counts


def _sparse_scores(
    candidate_texts: Sequence[str], query_texts: Sequence[str], method: str
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    if method == "char_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=250_000,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        documents = vectorizer.fit_transform(candidate_texts).tocsr()
        queries = vectorizer.transform(query_texts).tocsr()
    elif method == "word_tfidf":
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            max_features=250_000,
            lowercase=True,
            strip_accents="unicode",
            sublinear_tf=True,
            norm="l2",
            dtype=np.float32,
        )
        documents = vectorizer.fit_transform(candidate_texts).tocsr()
        queries = vectorizer.transform(query_texts).tocsr()
    elif method == "bm25":
        vectorizer = CountVectorizer(
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
            max_features=250_000,
            lowercase=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        counts = vectorizer.fit_transform(candidate_texts).tocsr()
        documents = _bm25_document_matrix(counts)
        queries = vectorizer.transform(query_texts).astype(np.float32).tocsr()
        queries.data[:] = 1.0
    else:
        raise ValueError(f"unknown sparse method: {method}")
    scores = np.empty((len(query_texts), len(candidate_texts)), dtype=np.float32)
    for begin in range(0, len(query_texts), 32):
        end = min(begin + 32, len(query_texts))
        scores[begin:end] = (queries[begin:end] @ documents.T).toarray().astype(
            np.float32, copy=False
        )
    receipt = {
        "method": method,
        "features": len(vectorizer.vocabulary_),
        "candidate_matrix_nnz": int(documents.nnz),
        "candidate_matrix_bytes": int(
            documents.data.nbytes + documents.indices.nbytes + documents.indptr.nbytes
        ),
        "wall_seconds": time.perf_counter() - started,
    }
    return scores, receipt


def _dense_scores(query_vectors: np.ndarray, candidate_vectors: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full Qwen scoring pass")
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    candidates = torch.tensor(
        np.asarray(candidate_vectors), dtype=torch.float32, device="cuda"
    )
    output = np.empty((len(query_vectors), len(candidate_vectors)), dtype=np.float32)
    for begin in range(0, len(query_vectors), 32):
        end = min(begin + 32, len(query_vectors))
        queries = torch.as_tensor(
            np.asarray(query_vectors[begin:end]), dtype=torch.float32, device="cuda"
        )
        output[begin:end] = (queries @ candidates.T).cpu().numpy()
        del queries
    receipt = {
        "method": "qwen3_embedding_8b_cosine",
        "wall_seconds": time.perf_counter() - started,
        "peak_vram_mib": float(torch.cuda.max_memory_allocated() / 2**20),
    }
    del candidates
    torch.cuda.empty_cache()
    return output, receipt


def _instrument_centroids(
    article_vectors: np.ndarray,
    parent_positions: np.ndarray,
    instrument_count: int,
) -> np.ndarray:
    result = np.zeros((instrument_count, article_vectors.shape[1]), dtype=np.float32)
    for instrument_position in range(instrument_count):
        positions = np.flatnonzero(parent_positions == instrument_position)
        if not len(positions):
            raise ValueError(f"instrument without articles: {instrument_position}")
        result[instrument_position] = np.asarray(article_vectors[positions]).mean(axis=0)
    result /= np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1e-12)
    return result


def _aggregate_article_scores(
    article_scores: np.ndarray, parent_positions: np.ndarray, instrument_count: int
) -> np.ndarray:
    output = np.full((len(article_scores), instrument_count), -np.inf, dtype=np.float32)
    for instrument_position in range(instrument_count):
        positions = np.flatnonzero(parent_positions == instrument_position)
        output[:, instrument_position] = np.max(article_scores[:, positions], axis=1)
    return output


def _top_order(values: np.ndarray, candidate_ids: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    keep = min(top_k, len(candidate_ids))
    if keep == len(candidate_ids):
        return np.lexsort((candidate_ids, -values))
    pool = np.argpartition(-values, keep - 1)[:keep]
    return pool[np.lexsort((candidate_ids[pool], -values[pool]))]


def _evaluate(
    scores: np.ndarray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    references: Mapping[str, Sequence[str]],
    components: Mapping[str, str],
    *,
    task: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    ids = np.asarray(candidate_ids, dtype=str)
    positions = {value: index for index, value in enumerate(ids)}
    cutoff = 5 if task == "C01-INSTRUMENT" else 10
    recall_cutoffs = (1, 3, 5) if task == "C01-INSTRUMENT" else (20, 100)
    details: list[dict[str, Any]] = []
    rankings: list[dict[str, Any]] = []
    for row_index, query_id in enumerate(query_ids):
        values = scores[row_index]
        order = _top_order(values, ids)
        positives = {value for value in references[query_id] if value in positions}
        positive_positions = [positions[value] for value in positives]
        exact_ranks = []
        for position in positive_positions:
            score = values[position]
            rank = 1 + int(np.sum(values > score))
            if np.any(values == score):
                rank += int(np.sum((values == score) & (ids < ids[position])))
            exact_ranks.append(rank)
        ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(positives), cutoff)))
        dcg = sum(1.0 / math.log2(rank + 1) for rank in exact_ranks if rank <= cutoff)
        detail = {
            "query_id": query_id,
            "component_id": components[query_id],
            "positive_count": len(positives),
            f"ndcg_at_{cutoff}": dcg / ideal if ideal else 0.0,
            "mrr": 1.0 / min(exact_ranks) if exact_ranks else 0.0,
        }
        for value in recall_cutoffs:
            detail[f"recall_at_{value}"] = (
                sum(rank <= value for rank in exact_ranks) / len(positives) if positives else 0.0
            )
        details.append(detail)
        rankings.extend(
            {
                "query_id": query_id,
                "candidate_id": str(ids[position]),
                "rank": rank,
                "score": float(values[position]),
            }
            for rank, position in enumerate(order, 1)
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        grouped[str(row["component_id"])].append(row)
    metric_names = [f"ndcg_at_{cutoff}", "mrr"] + [f"recall_at_{value}" for value in recall_cutoffs]
    component_rows = []
    for component_id, rows in sorted(grouped.items()):
        component_rows.append(
            {"component_id": component_id, **{name: float(np.mean([row[name] for row in rows])) for name in metric_names}}
        )
    metrics = {
        "queries": len(details),
        "components": len(component_rows),
        **{f"query_macro_{name}": float(np.mean([row[name] for row in details])) for name in metric_names},
        **{f"component_macro_{name}": float(np.mean([row[name] for row in component_rows])) for name in metric_names},
    }
    return metrics, details, rankings


def _rrf_scores(first: np.ndarray, second: np.ndarray, candidate_ids: Sequence[str]) -> np.ndarray:
    ids = np.asarray(candidate_ids, dtype=str)
    output = np.zeros_like(first, dtype=np.float32)
    for row_index in range(len(first)):
        for values in (first[row_index], second[row_index]):
            order = _top_order(values, ids)
            output[row_index, order] += 1.0 / (60.0 + np.arange(1, len(order) + 1))
    return output


def _graph_expand(
    base_scores: np.ndarray, edges: Sequence[tuple[int, int]], weight: float
) -> np.ndarray:
    output = np.asarray(base_scores, dtype=np.float32).copy()
    normalized = _robust_z(base_scores)
    for source, target in edges:
        contribution = np.maximum(normalized[:, source], 0.0) * weight
        output[:, target] += contribution
    return output


def _primary_metric(task: str) -> str:
    return "component_macro_ndcg_at_5" if task == "C01-INSTRUMENT" else "component_macro_ndcg_at_10"


def _component_means(details: Sequence[Mapping[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in details:
        grouped[str(row["component_id"])].append(float(row[metric]))
    return {component: float(np.mean(values)) for component, values in grouped.items()}


def _paired_bootstrap(
    first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]], metric: str
) -> dict[str, Any]:
    # Average repeated query rows within each component before pairing.
    left = _component_means(first, metric)
    right = _component_means(second, metric)
    keys = sorted(set(left) & set(right))
    deltas = np.asarray([right[key] - left[key] for key in keys], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.choice(deltas, size=(BOOTSTRAP_DRAWS, len(deltas)), replace=True).mean(axis=1)
    return {
        "first": "first_argument",
        "second": "second_argument",
        "delta_second_minus_first": float(deltas.mean()),
        "ci95": [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))],
        "components": len(keys),
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
    }


def prepare_embeddings() -> dict[str, Any]:
    result_path = RUN / "embedding_receipt.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    RUN.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    compatibility = json.loads(
        (ROOT / "research/state/c01_statute_graph_retrieval_20260918/qwen_candidate_cache_compatibility.json").read_text(encoding="utf-8")
    )
    if not compatibility.get("array_hash_match") or not compatibility.get("candidate_text_hash_match"):
        raise ValueError("the frozen Qwen article cache is not compatible")
    if sha256_file(ARTICLE_CACHE) != ARTICLE_CACHE_SHA256:
        raise ValueError("the Qwen article array hash changed")
    queries = [
        row
        for row in _jsonl(RELEASE / "method/queries.jsonl")
        if row.get("split") in {"train", "development"}
    ]
    texts = [str(row["text"]) for row in queries]
    prior_receipt_path = RUN.parent / "exploratory_v1/embedding_receipt.json"
    if prior_receipt_path.is_file():
        prior_receipt = json.loads(prior_receipt_path.read_text(encoding="utf-8"))
        truncation = dict(prior_receipt["query_truncation"])
        if truncation.get("rows") != len(texts) or truncation.get("truncated_rows") != 0:
            raise ValueError("the pinned no-truncation query contract changed")
    else:
        if any(len(text) > 30_000 for text in texts):
            raise ValueError("query requires tokenizer-qualified truncation")
        truncation = {
            "max_tokens": 7600,
            "rows": len(texts),
            "truncated_rows": 0,
            "truncation_rate": 0.0,
            "verification": "conservative_character_bound; tokenizer audit required before endpoint generation",
        }
    vectors, metadata = _qwen_embeddings(
        CACHE,
        QUERY_KIND,
        texts,
        instruction=QUERY_INSTRUCTION,
        batch_size=64,
        workers=4,
    )
    if vectors.shape != (len(queries), QWEN_DIMENSION):
        raise ValueError("incomplete query embedding array")
    descriptors_path = RUN / "query_descriptors.jsonl"
    _write_jsonl(
        descriptors_path,
        (
            {
                "query_id": row["query_id"],
                "component_id": row["component_id"],
                "split": row["split"],
                "text_sha256": hashlib.sha256(str(row["text"]).encode("utf-8")).hexdigest(),
            }
            for row in queries
        ),
    )
    generation = _qwen_generation_receipt(CACHE, QUERY_KIND, metadata)
    receipt = {
        "schema_version": 1,
        "status": "completed",
        "created_utc": utc_now(),
        "role": "exploratory structured-only coverage-pilot query embeddings",
        "model": QWEN_MODEL,
        "served_revision": QWEN_REVISION,
        "endpoint_discovery_sha256": QWEN_DISCOVERY_SHA256,
        "tokenizer_revision": QWEN_TOKENIZER_REVISION,
        "instruction": QUERY_INSTRUCTION,
        "query_count": len(queries),
        "dimension": QWEN_DIMENSION,
        "query_generation": generation,
        "query_truncation": truncation,
        "query_descriptors_sha256": sha256_file(descriptors_path),
        "candidate_array": str(ARTICLE_CACHE.relative_to(ROOT)).replace("\\", "/"),
        "candidate_array_sha256": ARTICLE_CACHE_SHA256,
        "candidate_text_sha256": ARTICLE_TEXT_SHA256,
        "candidate_count": 36887,
        "test_query_embedded": False,
        "test_reference_opened": False,
        "endpoint_money": "unknown",
    }
    _write_json(result_path, receipt)
    return receipt


def run_exploratory() -> dict[str, Any]:
    result_path = RUN / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text(encoding="utf-8"))
    started = time.perf_counter()
    embedding_receipt = prepare_embeddings()
    queries = {
        str(row["query_id"]): row
        for row in _jsonl(RELEASE / "method/queries.jsonl")
        if row.get("split") in {"train", "development"}
    }
    descriptors = _jsonl(RUN / "query_descriptors.jsonl")
    query_position = {str(row["query_id"]): index for index, row in enumerate(descriptors)}
    query_meta = embedding_receipt["query_generation"]
    query_array = CACHE / f"{QUERY_KIND}_{str(query_meta['operation_sha256'])[:20]}.npy"
    query_vectors = np.load(query_array, mmap_mode="r")
    article_vectors = np.load(ARTICLE_CACHE, mmap_mode="r")
    instruments = _jsonl(RELEASE / "candidates/instruments.jsonl")
    articles = _jsonl(RELEASE / "candidates/articles.jsonl")
    instrument_ids = [str(row["instrument_id"]) for row in instruments]
    article_ids = [str(row["candidate_id"]) for row in articles]
    instrument_position = {value: index for index, value in enumerate(instrument_ids)}
    parent_positions = np.asarray(
        [instrument_position[str(row["instrument_id"])] for row in articles], dtype=np.int32
    )
    components = {query_id: str(row["component_id"]) for query_id, row in queries.items()}
    instrument_refs_rows = _jsonl(RELEASE / "method/instrument_references_train_development.jsonl")
    article_refs_rows = _jsonl(RELEASE / "method/article_references_train_development.jsonl")
    references_by_task = {
        "C01-INSTRUMENT": {str(row["query_id"]): list(map(str, row["positive_ids"])) for row in instrument_refs_rows},
        "C01-ARTICLE": {str(row["query_id"]): list(map(str, row["positive_ids"])) for row in article_refs_rows},
    }
    query_ids_by_task: dict[str, list[str]] = {}
    partitions_by_task: dict[str, dict[str, list[str]]] = {}
    for task, references in references_by_task.items():
        internal = sorted(
            query_id
            for query_id in references
            if queries[query_id]["split"] == "train"
            and _component_is_internal_validation(components[query_id])
        )
        development = sorted(
            query_id for query_id in references if queries[query_id]["split"] == "development"
        )
        partitions_by_task[task] = {"internal_validation": internal, "development": development}
        query_ids_by_task[task] = internal + development
    union_query_ids = sorted(set(query_ids_by_task["C01-INSTRUMENT"]) | set(query_ids_by_task["C01-ARTICLE"]))
    union_vectors = np.asarray(query_vectors[[query_position[value] for value in union_query_ids]], dtype=np.float32)
    union_texts = [str(queries[value]["text"]) for value in union_query_ids]
    union_position = {value: index for index, value in enumerate(union_query_ids)}

    resources: dict[str, Any] = {}
    article_dense_union, resources["article_qwen"] = _dense_scores(union_vectors, article_vectors)
    centroids = _instrument_centroids(article_vectors, parent_positions, len(instrument_ids))
    instrument_dense_union = union_vectors @ centroids.T
    resources["instrument_centroid_bytes"] = int(centroids.nbytes)

    article_texts = [str(row["instrument_article_body"]) for row in articles]
    instrument_texts = [
        "\n".join([str(row["title"])] * 2 + list(map(str, row.get("observed_aliases", []))))
        for row in instruments
    ]
    instrument_eval_ids = query_ids_by_task["C01-INSTRUMENT"]
    instrument_eval_texts = [str(queries[value]["text"]) for value in instrument_eval_ids]
    instrument_sparse = {}
    for method in ("bm25", "char_tfidf", "word_tfidf"):
        instrument_sparse[method], resources[f"instrument_{method}"] = _sparse_scores(
            instrument_texts, instrument_eval_texts, method
        )
    instrument_dense = instrument_dense_union[[union_position[value] for value in instrument_eval_ids]]
    instrument_article_max = _aggregate_article_scores(
        article_dense_union[[union_position[value] for value in instrument_eval_ids]],
        parent_positions,
        len(instrument_ids),
    )

    article_eval_ids = query_ids_by_task["C01-ARTICLE"]
    article_eval_texts = [str(queries[value]["text"]) for value in article_eval_ids]
    article_sparse = {}
    for method in ("bm25", "char_tfidf", "word_tfidf"):
        article_sparse[method], resources[f"article_{method}"] = _sparse_scores(
            article_texts, article_eval_texts, method
        )
    article_dense = article_dense_union[[union_position[value] for value in article_eval_ids]]

    method_scores: dict[str, dict[str, np.ndarray]] = {
        "C01-INSTRUMENT": {
            **instrument_sparse,
            "qwen_article_centroid": instrument_dense,
            "qwen_article_max": instrument_article_max,
        },
        "C01-ARTICLE": {**article_sparse, "qwen_direct": article_dense},
    }
    selections: dict[str, Any] = {}
    all_metrics: dict[str, Any] = {"C01-INSTRUMENT": {}, "C01-ARTICLE": {}}
    all_details: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    development_rankings: list[dict[str, Any]] = []

    def evaluate_method(
        task: str,
        method: str,
        values: np.ndarray,
        selected_partitions: Sequence[str] = ("internal_validation", "development"),
    ) -> None:
        ids = instrument_ids if task == "C01-INSTRUMENT" else article_ids
        qids = query_ids_by_task[task]
        for partition, selected_ids in partitions_by_task[task].items():
            if partition not in selected_partitions:
                continue
            selected_positions = [qids.index(value) for value in selected_ids]
            metrics, details, rankings = _evaluate(
                values[selected_positions],
                selected_ids,
                ids,
                references_by_task[task],
                components,
                task=task,
            )
            all_metrics[task].setdefault(method, {})[partition] = metrics
            all_details[(task, method, partition)] = details
            if partition == "development":
                for row in rankings:
                    development_rankings.append({"task_id": task, "method": method, **row})

    for task, methods in method_scores.items():
        for method, values in methods.items():
            evaluate_method(task, method, values)
        lexical_methods = ("bm25", "char_tfidf", "word_tfidf")
        primary = _primary_metric(task)
        best_lexical = max(
            lexical_methods,
            key=lambda method: all_metrics[task][method]["internal_validation"][primary],
        )
        dense_candidates = (
            ("qwen_article_centroid", "qwen_article_max")
            if task == "C01-INSTRUMENT"
            else ("qwen_direct",)
        )
        dense_method = max(
            dense_candidates,
            key=lambda method: all_metrics[task][method]["internal_validation"][primary],
        )
        best_weight = None
        best_value = -np.inf
        for lexical_weight in (0.1, 0.25, 0.5, 0.75):
            values = (
                lexical_weight * _robust_z(methods[best_lexical])
                + (1.0 - lexical_weight) * _robust_z(methods[dense_method])
            )
            name = f"fusion_robust_z_w{lexical_weight:.2f}"
            evaluate_method(task, name, values, ("internal_validation",))
            score = all_metrics[task][name]["internal_validation"][primary]
            if score > best_value:
                best_value = score
                best_weight = lexical_weight
        assert best_weight is not None
        selected_fusion = f"fusion_robust_z_w{best_weight:.2f}"
        methods["fusion_selected"] = (
            best_weight * _robust_z(methods[best_lexical])
            + (1.0 - best_weight) * _robust_z(methods[dense_method])
        )
        evaluate_method(task, "fusion_selected", methods["fusion_selected"])
        methods["rrf_lexical_qwen"] = _rrf_scores(
            methods[best_lexical], methods[dense_method],
            instrument_ids if task == "C01-INSTRUMENT" else article_ids,
        )
        evaluate_method(task, "rrf_lexical_qwen", methods["rrf_lexical_qwen"])
        selections[task] = {
            "best_lexical": best_lexical,
            "dense_method": dense_method,
            "fusion_lexical_weight": best_weight,
            "selected_on": "component-disjoint internal_validation carved from train",
        }

    article_law_positions = [instrument_eval_ids.index(value) for value in article_eval_ids]
    predicted_law_scores = method_scores["C01-INSTRUMENT"]["fusion_selected"][article_law_positions]
    direct_scores = method_scores["C01-ARTICLE"]["fusion_selected"]
    primary = _primary_metric("C01-ARTICLE")
    best_beta = None
    best_value = -np.inf
    for beta in (0.1, 0.25, 0.5, 1.0):
        hierarchical = _robust_z(direct_scores) + beta * _robust_z(predicted_law_scores)[:, parent_positions]
        name = f"predicted_law_hierarchical_b{beta:.2f}"
        method_scores["C01-ARTICLE"][name] = hierarchical
        evaluate_method("C01-ARTICLE", name, hierarchical, ("internal_validation",))
        score = all_metrics["C01-ARTICLE"][name]["internal_validation"][primary]
        if score > best_value:
            best_value = score
            best_beta = beta
    assert best_beta is not None
    hierarchical_selected = f"predicted_law_hierarchical_b{best_beta:.2f}"
    method_scores["C01-ARTICLE"]["hierarchical_selected"] = method_scores["C01-ARTICLE"][hierarchical_selected]
    evaluate_method("C01-ARTICLE", "hierarchical_selected", method_scores["C01-ARTICLE"]["hierarchical_selected"])
    selections["C01-ARTICLE"]["hierarchical_beta"] = best_beta

    article_position = {value: index for index, value in enumerate(article_ids)}
    graph_edges = []
    for row in _jsonl(RELEASE / "graph/exact_direct_article_edges.jsonl"):
        source = article_position.get(str(row["source_id"]))
        target = article_position.get(str(row["target_id"]))
        if source is not None and target is not None:
            graph_edges.append((source, target))
    best_gamma = None
    best_value = -np.inf
    for gamma in (0.05, 0.1, 0.2):
        expanded = _graph_expand(direct_scores, graph_edges, gamma)
        name = f"exact_reference_expansion_g{gamma:.2f}"
        method_scores["C01-ARTICLE"][name] = expanded
        evaluate_method("C01-ARTICLE", name, expanded, ("internal_validation",))
        score = all_metrics["C01-ARTICLE"][name]["internal_validation"][primary]
        if score > best_value:
            best_value = score
            best_gamma = gamma
    assert best_gamma is not None
    graph_selected = f"exact_reference_expansion_g{best_gamma:.2f}"
    method_scores["C01-ARTICLE"]["graph_expansion_selected"] = method_scores["C01-ARTICLE"][graph_selected]
    evaluate_method("C01-ARTICLE", "graph_expansion_selected", method_scores["C01-ARTICLE"]["graph_expansion_selected"])
    selections["C01-ARTICLE"]["graph_expansion_weight"] = best_gamma

    oracle = _robust_z(direct_scores).copy()
    for row_index, query_id in enumerate(article_eval_ids):
        positive_laws = {
            str(candidate).split("#articulo=", 1)[0]
            for candidate in references_by_task["C01-ARTICLE"][query_id]
        }
        for instrument_id in positive_laws:
            oracle[row_index, parent_positions == instrument_position[instrument_id]] += 1.0
    method_scores["C01-ARTICLE"]["oracle_parent_law_diagnostic"] = oracle
    evaluate_method("C01-ARTICLE", "oracle_parent_law_diagnostic", oracle)

    comparisons = {}
    for task in ("C01-INSTRUMENT", "C01-ARTICLE"):
        metric = "ndcg_at_5" if task == "C01-INSTRUMENT" else "ndcg_at_10"
        lexical = selections[task]["best_lexical"]
        dense = selections[task]["dense_method"]
        comparisons[f"{task}:qwen_minus_lexical"] = _paired_bootstrap(
            all_details[(task, lexical, "development")],
            all_details[(task, dense, "development")],
            metric,
        )
        comparisons[f"{task}:fusion_minus_lexical"] = _paired_bootstrap(
            all_details[(task, lexical, "development")],
            all_details[(task, "fusion_selected", "development")],
            metric,
        )
    comparisons["C01-ARTICLE:hierarchical_minus_direct"] = _paired_bootstrap(
        all_details[("C01-ARTICLE", "fusion_selected", "development")],
        all_details[("C01-ARTICLE", "hierarchical_selected", "development")],
        "ndcg_at_10",
    )
    comparisons["C01-ARTICLE:graph_minus_direct"] = _paired_bootstrap(
        all_details[("C01-ARTICLE", "fusion_selected", "development")],
        all_details[("C01-ARTICLE", "graph_expansion_selected", "development")],
        "ndcg_at_10",
    )

    rankings_path = RUN / "rankings_development.jsonl"
    _write_jsonl(rankings_path, development_rankings)
    compact_metrics = {
        task: {
            method: values
            for method, values in methods.items()
            if method
            in {
                "bm25",
                "char_tfidf",
                "word_tfidf",
                "qwen_article_centroid",
                "qwen_article_max",
                "qwen_direct",
                "fusion_selected",
                "rrf_lexical_qwen",
                "hierarchical_selected",
                "graph_expansion_selected",
                "oracle_parent_law_diagnostic",
            }
        }
        for task, methods in all_metrics.items()
    }
    result = {
        "schema_version": 1,
        "status": "completed_exploratory_coverage_pilot",
        "created_utc": utc_now(),
        "release_id": "uol_c01_statutes_20260918_v1_candidate",
        "candidate_counts": {"C01-INSTRUMENT": len(instrument_ids), "C01-ARTICLE": len(article_ids)},
        "partitions": {
            task: {name: {"queries": len(values), "components": len({components[q] for q in values})} for name, values in partitions.items()}
            for task, partitions in partitions_by_task.items()
        },
        "selections": selections,
        "metrics": compact_metrics,
        "comparisons": comparisons,
        "resources": resources,
        "embedding_receipt": str((RUN / "embedding_receipt.json").relative_to(ROOT)).replace("\\", "/"),
        "rankings": str(rankings_path.relative_to(ROOT)).replace("\\", "/"),
        "rankings_sha256": sha256_file(rankings_path),
        "graph_edges_used": len(graph_edges),
        "test_query_embedded": False,
        "test_reference_opened": False,
        "confirmatory": False,
        "limitation": "Both tasks missed the pre-registered 3000-component promotion gate; development results are exploratory.",
        "wall_seconds": time.perf_counter() - started,
    }
    _write_json(result_path, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("embed", "run"))
    args = parser.parse_args(argv)
    result = prepare_embeddings() if args.command == "embed" else run_exploratory()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
