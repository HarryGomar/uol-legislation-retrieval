"""Red input audit for the C01 representation-refinement programme.

The audit recomputes every pinned hash, shape, order, ownership relation, and
graph/fusion observation before any fitting happens.  A mismatch is a
capability block: the caller must stop rather than repair inputs by calling an
endpoint.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from typing import Any

import numpy as np

from uol.retrieval_benchmarks.common import atomic_write_json, sha256_file, utc_now

from . import contracts as C
from . import safeio
from .metrics import evaluate_scores, paired_bootstrap, robust_z_rows


def _sha256(path, chunk_size: int = 1 << 20) -> str:
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            value.update(chunk)
    return value.hexdigest()


def _is_internal_validation(component_id: str) -> bool:
    value = int(hashlib.sha256(component_id.encode("utf-8")).hexdigest()[:8], 16)
    return value % 5 == 0


def _clause_representation_from_parquet():
    # On this Windows CUDA runtime, Arrow's bundled native libraries loaded
    # first can crash SentencePiece during DLL initialisation.  Establish the
    # tokenizer runtime before importing Arrow, matching the recorded contract
    # in legal_deepening_first_stage.
    try:
        import sentencepiece  # noqa: F401
    except ModuleNotFoundError:
        sentencepiece = None  # type: ignore[assignment]
    import pyarrow.parquet as pq

    from uol.retrieval_benchmarks.legal_deepening_first_stage import candidate_representation

    rows = pq.read_table(C.PROVISIONS).to_pylist()
    if len(rows) != C.ARTICLE_ROWS:
        raise ValueError("provisions parquet row count changed")
    return rows, candidate_representation(rows, "clause")


def _article115_observation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Token-count the Constitution Article 115 cached input locally."""
    try:
        from transformers import AutoTokenizer

        snapshot = (
            r"C:\uol_legacy_home\user\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-8B"
            r"\snapshots\1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
        )
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        matches = [
            row
            for row in rows
            if str(row.get("candidate_id", "")).endswith("CPEUM.pdf#articulo=115")
        ]
        if len(matches) != 1:
            return {
                "status": "failed",
                "reason": f"expected one Constitution Article 115 row, found {len(matches)}",
            }
        from uol.retrieval_benchmarks.legal_deepening_first_stage import article_text

        text = article_text(matches[0])
        tokens = len(tokenizer(text)["input_ids"])
        return {
            "status": "verified_with_pinned_tokenizer",
            "candidate_id": str(matches[0]["candidate_id"]),
            "tokens": tokens,
            "below_8000_limit": tokens < 8000,
            "about_3600": abs(tokens - 3600) <= 400,
        }
    except Exception as error:  # pragma: no cover - environment-dependent
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"}


def _fusion_reproduction(inputs: safeio.SafeInputs) -> dict[str, Any]:
    """Recompute the archived fusion from pinned caches before ranking."""
    import torch

    started = time.perf_counter()
    baseline = json.loads(C.BASELINE_RESULT.read_text(encoding="utf-8"))
    selections = baseline["selections"]["C01-ARTICLE"]
    if (
        selections["best_lexical"] != C.ARCHIVED_FUSION["lexical_method"]
        or selections["fusion_lexical_weight"] != C.ARCHIVED_FUSION["lexical_weight"]
    ):
        raise ValueError("archived fusion selection does not match the pinned operator")
    queries = inputs.public_queries()
    query_ids = [str(row["query_id"]) for row in queries]
    query_position = {value: index for index, value in enumerate(query_ids)}
    articles = inputs.articles()
    article_ids = [str(row["candidate_id"]) for row in articles]
    article_texts = [str(row["instrument_article_body"]) for row in articles]
    components = {str(row["query_id"]): str(row["component_id"]) for row in queries}
    references = {
        str(row["query_id"]): list(map(str, row["positive_ids"]))
        for row in inputs.article_references()
    }
    partitions: dict[str, list[str]] = {"internal_validation": [], "development": []}
    query_split = {str(row["query_id"]): str(row["split"]) for row in queries}
    for query_id in references:
        split = query_split[query_id]
        if split == "development":
            partitions["development"].append(query_id)
        elif split == "train" and _is_internal_validation(components[query_id]):
            partitions["internal_validation"].append(query_id)
    for values in partitions.values():
        values.sort()

    from sklearn.feature_extraction.text import TfidfVectorizer

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
    documents = vectorizer.fit_transform(article_texts).tocsr()
    lexical_started = time.perf_counter()
    lexical = np.empty((len(queries), len(article_ids)), dtype=np.float32)
    for begin in range(0, len(queries), 32):
        end = min(begin + 32, len(queries))
        chunk = vectorizer.transform([str(row["text"]) for row in queries[begin:end]]).tocsr()
        lexical[begin:end] = (chunk @ documents.T).toarray().astype(np.float32, copy=False)
    lexical_seconds = time.perf_counter() - lexical_started
    del documents

    article_vectors = np.asarray(inputs.article_array(), dtype=np.float32)
    query_vectors = np.asarray(inputs.query_array(), dtype=np.float32)
    dense_started = time.perf_counter()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            index = torch.from_numpy(article_vectors).to(device=device, dtype=torch.float32)
            dense = np.empty((len(queries), len(article_ids)), dtype=np.float32)
            for begin in range(0, len(queries), 64):
                end = min(begin + 64, len(queries))
                block = torch.from_numpy(query_vectors[begin:end]).to(
                    device=device, dtype=torch.float32
                )
                dense[begin:end] = (block @ index.T).cpu().numpy()
            del index, block
            peak_vram_mib = float(torch.cuda.max_memory_allocated() / 2**20)
        torch.cuda.empty_cache()
    else:
        dense = (query_vectors @ article_vectors.T).astype(np.float32, copy=False)
        peak_vram_mib = 0.0
    dense_seconds = time.perf_counter() - dense_started
    lexical_path = C.STATE / "lexical_char_tfidf_public.npy"
    dense_path = C.STATE / "dense_qwen_public.npy"
    C.STATE.mkdir(parents=True, exist_ok=True)
    np.save(lexical_path, lexical, allow_pickle=False)
    np.save(dense_path, dense, allow_pickle=False)

    weight = float(C.ARCHIVED_FUSION["lexical_weight"])
    fusion = (weight * robust_z_rows(lexical) + (1.0 - weight) * robust_z_rows(dense)).astype(
        np.float32, copy=False
    )
    reproduced: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    archived_metrics = baseline["metrics"]["C01-ARTICLE"]
    for partition, selected in partitions.items():
        positions = [query_position[value] for value in selected]
        metrics, details, _ = evaluate_scores(
            fusion[positions], selected, article_ids, references, components
        )
        _, details_direct, _ = evaluate_scores(
            dense[positions], selected, article_ids, references, components
        )
        archived = archived_metrics["fusion_selected"][partition]
        reproduced[partition] = {
            name: {
                "reproduced": metrics[name],
                "archived": archived[name],
                "absolute_difference": abs(metrics[name] - archived[name]),
            }
            for name in (
                "component_macro_ndcg_at_10",
                "component_macro_mrr",
                "component_macro_recall_at_20",
                "component_macro_recall_at_100",
            )
        }
        comparisons[f"{partition}:fusion_minus_qwen_direct"] = paired_bootstrap(
            details_direct, details, "ndcg_at_10", draws=2000
        )
    tolerance = 5e-4
    max_difference = max(
        row["absolute_difference"]
        for partition in reproduced.values()
        for row in partition.values()
    )
    return {
        "status": "passed" if max_difference <= tolerance else "failed",
        "tolerance_absolute": tolerance,
        "max_absolute_difference": max_difference,
        "operator": {
            "lexical_method": C.ARCHIVED_FUSION["lexical_method"],
            "normalization": C.ARCHIVED_FUSION["normalization"],
            "lexical_weight": C.ARCHIVED_FUSION["lexical_weight"],
            "dense_channel": "qwen_direct",
            "matrix_scope": "full_universe_before_ranking",
            "candidate_count": len(article_ids),
        },
        "reproduced_metrics": reproduced,
        "spot_comparisons": comparisons,
        "resources": {
            "lexical_wall_seconds": lexical_seconds,
            "dense_wall_seconds": dense_seconds,
            "dense_peak_vram_mib": peak_vram_mib,
            "wall_seconds": time.perf_counter() - started,
        },
        "derived_matrices": {
            "lexical": {
                "path": safeio.relative_str(lexical_path),
                "sha256": sha256_file(lexical_path),
                "shape": list(lexical.shape),
            },
            "dense": {
                "path": safeio.relative_str(dense_path),
                "sha256": sha256_file(dense_path),
                "shape": list(dense.shape),
            },
        },
    }


def _component_coverage(
    inputs: safeio.SafeInputs,
    components: dict[str, str],
    endpoints: set[str],
) -> dict[str, dict[str, int]]:
    """Count components whose positive set intersects reference endpoints."""
    query_split = {
        str(row["query_id"]): str(row["split"]) for row in inputs.public_queries()
    }
    seen: dict[str, set[str]] = {}
    connected: dict[str, set[str]] = {}
    for row in inputs.article_references():
        query_id = str(row["query_id"])
        component_id = components[query_id]
        split = query_split[query_id]
        category = (
            "internal_validation"
            if split == "train" and _is_internal_validation(component_id)
            else "fitting"
            if split == "train"
            else split
        )
        seen.setdefault(category, set()).add(component_id)
        if any(str(value) in endpoints for value in row["positive_ids"]):
            connected.setdefault(category, set()).add(component_id)
    return {
        category: {
            "components": len(components_seen),
            "reference_connected": len(connected.get(category, set())),
        }
        for category, components_seen in seen.items()
    }


def run_audit() -> dict[str, Any]:
    """Execute the red input audit and write the compatibility receipt."""
    started = time.perf_counter()
    C.STATE.mkdir(parents=True, exist_ok=True)
    inputs = safeio.SafeInputs()
    checks: dict[str, Any] = {}

    observed = _sha256(C.RELEASE / "manifest.json")
    checks["release_manifest"] = {
        "path": safeio.relative_str(C.RELEASE / "manifest.json"),
        "sha256_observed": observed,
        "sha256_expected": C.RELEASE_MANIFEST_SHA256,
        "match": observed == C.RELEASE_MANIFEST_SHA256,
    }

    arrays = {
        "article": (C.ARTICLE_ARRAY, C.ARTICLE_ARRAY_SHA256, (C.ARTICLE_ROWS, C.DIMENSION)),
        "clause": (C.CLAUSE_ARRAY, C.CLAUSE_ARRAY_SHA256, (C.CLAUSE_ROWS, C.DIMENSION)),
        "query": (C.QUERY_ARRAY, C.QUERY_ARRAY_SHA256, (C.PUBLIC_QUERY_ROWS, C.DIMENSION)),
    }
    for name, (path, expected, shape) in arrays.items():
        observed_hash = _sha256(path)
        array = np.load(path, mmap_mode="r")
        sample = np.asarray(array[:64])
        norms = np.linalg.norm(sample, axis=1)
        checks[f"{name}_array"] = {
            "path": safeio.relative_str(path),
            "sha256_observed": observed_hash,
            "sha256_expected": expected,
            "match": observed_hash == expected,
            "shape_observed": list(array.shape),
            "shape_expected": list(shape),
            "shape_match": tuple(array.shape) == shape,
            "finite": bool(np.isfinite(sample).all()),
            "sample_l2_norm_range": [float(norms.min()), float(norms.max())],
        }

    provision_rows, (clause_texts, clause_owners) = _clause_representation_from_parquet()
    articles = inputs.articles()
    release_ids = [str(row["candidate_id"]) for row in articles]
    parquet_ids = [str(row["candidate_id"]) for row in provision_rows]
    clause_digest = safeio.normalized_text_digest(clause_texts)
    owners = np.asarray(clause_owners, dtype=np.int64)
    checks["candidate_order"] = {
        "release_rows": len(release_ids),
        "parquet_rows": len(parquet_ids),
        "order_identical": release_ids == parquet_ids,
    }
    checks["clause_contract"] = {
        "contract": C.CLAUSE_CONTRACT,
        "rows": len(clause_texts),
        "rows_expected": C.CLAUSE_ROWS,
        "normalized_source_text_sha256": clause_digest,
        "normalized_source_text_sha256_expected": C.CLAUSE_TEXT_SHA256,
        "match": clause_digest == C.CLAUSE_TEXT_SHA256,
    }
    checks["clause_ownership"] = {
        "covered_candidates": int(np.unique(owners).size),
        "expected_candidates": len(release_ids),
        "full_coverage": int(np.unique(owners).size) == len(release_ids),
        "contiguous_owner_blocks": bool(np.all(np.diff(owners) >= 0)),
    }

    queries = inputs.public_queries()
    instruction_texts = [
        f"Instruct: {C.QUERY_INSTRUCTION}\nQuery:{row['text']}" for row in queries
    ]
    from uol.retrieval_benchmarks.common import digest

    query_digest = digest(instruction_texts)
    expected_query_digest = "cde5b0fe6cdb105ac046d737714b05fc9908349e2538b152e8fbddbd4d982425"
    checks["query_order"] = {
        "public_rows": len(queries),
        "instruction_text_sha256": query_digest,
        "expected": expected_query_digest,
        "match": query_digest == expected_query_digest,
        "splits": {
            name: sum(str(row["split"]) == name for row in queries)
            for name in ("train", "development")
        },
        "sealed_test_query_rows": sum(
            str(row["split"]) == "test" for row in inputs.queries()
        ),
    }
    checks["constitution_article_115"] = _article115_observation(provision_rows)

    edges = inputs.direct_article_edges()
    endpoints: set[str] = set()
    for row in edges:
        endpoints.add(str(row["source_id"]))
        endpoints.add(str(row["target_id"]))
    position = {value: index for index, value in enumerate(release_ids)}
    edge_pairs = [
        (position[str(row["source_id"])], position[str(row["target_id"])])
        for row in edges
    ]
    coverage = _component_coverage(inputs, {str(r["query_id"]): str(r["component_id"]) for r in queries}, endpoints)
    graph_checks = {
        "rows": len(edges),
        "rows_expected": C.EXPECTED_DIRECT_EDGES,
        "rows_match": len(edges) == C.EXPECTED_DIRECT_EDGES,
        "distinct_endpoints": len(endpoints),
        "distinct_endpoints_expected": C.EXPECTED_DIRECT_ENDPOINTS,
        "distinct_endpoints_match": len(endpoints) == C.EXPECTED_DIRECT_ENDPOINTS,
        "endpoint_share_of_universe": round(len(endpoints) / len(release_ids), 6),
        "self_loops": int(sum(source == target for source, target in edge_pairs)),
        "unknown_candidate_ids": int(
            sum(
                str(row["source_id"]) not in position or str(row["target_id"]) not in position
                for row in edges
            )
        ),
        "component_coverage": {
            name: {"components": value["components"], "reference_connected": value["reference_connected"]}
            for name, value in coverage.items()
        },
        "expected_component_coverage": C.EXPECTED_GRAPH_COVERAGE,
        "graph_support_gate": {
            "minimum_fitting": C.GRAPH_SUPPORT_GATE["minimum_fitting"],
            "minimum_internal_validation": C.GRAPH_SUPPORT_GATE["minimum_internal_validation"],
            "fitting_pass": coverage.get("fitting", {}).get("reference_connected", 0)
            >= C.GRAPH_SUPPORT_GATE["minimum_fitting"],
            "internal_validation_pass": coverage.get("internal_validation", {}).get(
                "reference_connected", 0
            )
            >= C.GRAPH_SUPPORT_GATE["minimum_internal_validation"],
        },
    }
    atomic_write_json(C.STATE / "graph_coverage.json", graph_checks)
    checks["graph_rows"] = graph_checks
    checks["custody"] = safeio.custody_denial_test()
    free_gib = shutil.disk_usage(str(C.ROOT)).free / 2**30
    checks["storage"] = {
        "free_gib": free_gib,
        "required_free_gib": C.FREE_STORAGE_LIMIT_GIB,
        "pass": free_gib >= C.FREE_STORAGE_LIMIT_GIB,
    }
    checks["specification_sha256"] = _sha256(C.SPEC_PATH)

    hard_failures: list[str] = []
    for name, value in checks.items():
        if isinstance(value, dict) and (
            value.get("match") is False
            or value.get("order_identical") is False
            or value.get("full_coverage") is False
            or value.get("rows_match") is False
            or value.get("shape_match") is False
            or value.get("status") == "failed"
        ):
            hard_failures.append(name)
    if checks["clause_contract"]["match"] is False:
        hard_failures.append("clause_contract")
    if checks["custody"]["status"] != "passed":
        hard_failures.append("custody")
    if not checks["storage"]["pass"]:
        hard_failures.append("storage")

    fusion = _fusion_reproduction(inputs)
    if fusion["status"] != "passed":
        hard_failures.append("fusion_reproduction")
    checks["fusion_reproduction"] = fusion
    fusion_audit = {
        "schema_version": 1,
        "spec_version": C.SPEC_VERSION,
        "created_utc": utc_now(),
        "status": fusion["status"],
        "provenance": "independently recomputed by refinement audit",
        **fusion,
    }
    atomic_write_json(C.STATE / "fusion_audit.json", fusion_audit)

    receipt = {
        "schema_version": 1,
        "spec_version": C.SPEC_VERSION,
        "created_utc": utc_now(),
        "status": "passed" if not hard_failures else "failed",
        "hard_failures": sorted(set(hard_failures)),
        "checks": checks,
        "endpoint_calls": 0,
        "embedding_calls": 0,
        "sealed_test_opened": False,
        "wall_seconds": time.perf_counter() - started,
    }
    atomic_write_json(C.STATE / "input_compatibility.json", receipt)
    return receipt
