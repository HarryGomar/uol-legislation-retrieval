"""Custody-safe foundations for the legal retrieval deepening programme.

The release builder repairs R02/R03 query equivalence without changing the
historical v4 release.  Method code receives only explicitly allowed split
views.  Evaluation references remain separate from model inputs.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from uol.retrieval_benchmarks.common import (
    ROOT,
    atomic_write_json,
    atomic_write_jsonl,
    canonical_bytes,
    digest,
    iter_jsonl,
    read_json,
    sha256_file,
    utc_now,
)
from uol.retrieval_benchmarks.quality_gate import audit_release


PROGRAM_ID = "legal_retrieval_deepening_20260917"
NORMALIZATION_VERSION = "uol-legal-input-nfkc-casefold-ws-v1"
SOURCE_RELEASE = ROOT / "data/releases/uol_retrieval_benchmarks_20260914_v4"
DEFAULT_RELEASE = (
    ROOT / "data/releases/uol_retrieval_benchmarks_20260917_r02_r03_repaired_v1"
)
TASK_IDS = ("R02", "R03")
METHOD_SPLITS = ("train", "development")


class ForbiddenPartitionAccess(PermissionError):
    """Raised before any path operation for a partition outside the allowlist."""


def normalize_legal_input_v1(value: Any) -> str:
    """Return the declared exact-input equivalence form for R02/R03."""

    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _rows(path: Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _receipt(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _all_receipts(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): _receipt(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def deterministic_rank(
    candidate_ids: Sequence[str], scores: Sequence[float]
) -> list[tuple[str, float]]:
    """Rank finite scores descending with candidate identity as the tie-break."""

    if len(candidate_ids) != len(scores):
        raise ValueError("candidate_ids and scores have different lengths")
    pairs: list[tuple[str, float]] = []
    for candidate_id, score in zip(candidate_ids, scores, strict=True):
        value = float(score)
        if not math.isfinite(value):
            raise ValueError(f"non-finite score for {candidate_id}")
        pairs.append((str(candidate_id), value))
    return sorted(pairs, key=lambda item: (-item[1], item[0]))


def multipositive_metrics(
    positives: Iterable[str], ranking: Sequence[str], *, k: int = 10
) -> dict[str, float]:
    """Score a deterministic ranking against a binary multi-positive target set."""

    relevant = {str(value) for value in positives}
    if not relevant:
        raise ValueError("at least one positive is required")
    seen: set[str] = set()
    ordered = []
    for value in ranking:
        value = str(value)
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, candidate_id in enumerate(ordered[:k], start=1)
        if candidate_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(k, len(relevant)) + 1)
    )
    first = next(
        (rank for rank, candidate_id in enumerate(ordered, start=1) if candidate_id in relevant),
        None,
    )

    def recall(depth: int) -> float:
        return len(relevant & set(ordered[:depth])) / len(relevant)

    return {
        f"ndcg_at_{k}": dcg / ideal,
        "mrr": 0.0 if first is None else 1.0 / first,
        "recall_at_20": recall(20),
        "recall_at_100": recall(100),
    }


@dataclass(frozen=True)
class CacheIdentity:
    source_text_sha256: str
    role: str
    instruction: str
    model_id: str
    revision: str
    dimension: int
    normalization: str
    dtype: str
    chunking_contract: str

    def validate(self) -> None:
        for key, value in asdict(self).items():
            if key == "dimension":
                if int(value) <= 0:
                    raise ValueError("dimension must be positive")
            elif not str(value):
                raise ValueError(f"cache identity field is empty: {key}")
        if len(self.source_text_sha256) != 64:
            raise ValueError("source_text_sha256 must be a full SHA-256 digest")

    def key(self) -> str:
        self.validate()
        return hashlib.sha256(canonical_bytes(asdict(self))).hexdigest()


class MethodSafeAdapter:
    """Load only method-authorized views from one repaired task.

    A forbidden reference request is rejected from the in-memory policy before
    the requested partition is converted to a path or opened.
    """

    def __init__(self, release: Path, task_id: str):
        self.release = Path(release).resolve()
        self.task_id = task_id
        policy_path = self.release / "method" / task_id / "access_policy.json"
        self._policy = read_json(policy_path)

    def _allowed_relative(self, kind: str, split: str) -> str:
        allowed = self._policy.get(kind, {})
        relative = allowed.get(split)
        if not isinstance(relative, str) or not relative:
            raise ForbiddenPartitionAccess(
                f"{self.task_id}:{split}:{kind} is outside the method allowlist"
            )
        return relative

    def load_queries(self, split: str) -> list[dict[str, Any]]:
        relative = self._allowed_relative("queries", split)
        return _rows(self.release / relative)

    def load_references(self, split: str) -> list[dict[str, Any]]:
        relative = self._allowed_relative("references", split)
        return _rows(self.release / relative)

    def candidate_path(self) -> Path:
        relative = self._policy.get("candidate_path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("candidate_path is absent from the access policy")
        return self.release / relative


def _source_defect_census(task_dir: Path) -> dict[str, Any]:
    """Reproduce the v4 conflict census without calling the shared quality gate."""

    queries = _rows(task_dir / "queries.jsonl")
    targets: dict[str, set[str]] = collections.defaultdict(set)
    for row in _rows(task_dir / "qrels.jsonl"):
        targets[str(row["query_id"])].add(str(row["candidate_id"]))
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in queries:
        buckets[normalize_legal_input_v1(row.get("text"))].append(row)
    conflicts = []
    for normalized, members in sorted(buckets.items()):
        target_sets = {
            tuple(sorted(targets[str(member["query_id"])])) for member in members
        }
        if len(target_sets) > 1:
            conflicts.append(
                {
                    "normalized_text_sha256": text_sha256(normalized),
                    "query_ids": sorted(str(row["query_id"]) for row in members),
                    "target_sets": [list(values) for values in sorted(target_sets)],
                }
            )
    return {
        "queries": len(queries),
        "unique_normalized_inputs": len(buckets),
        "conflicting_buckets": len(conflicts),
        "conflict_evidence": conflicts,
    }


def reproduce_source_defects(source_release: Path = SOURCE_RELEASE) -> dict[str, Any]:
    source_release = Path(source_release).resolve()
    result = {
        "schema_version": 1,
        "normalization_version": NORMALIZATION_VERSION,
        "source_release": str(source_release),
        "source_manifest_sha256": sha256_file(source_release / "manifest.json"),
        "tasks": {
            task_id: _source_defect_census(source_release / "tasks" / task_id)
            for task_id in TASK_IDS
        },
    }
    expected = {"R02": 4, "R03": 54}
    result["status"] = (
        "passed"
        if all(
            result["tasks"][task_id]["conflicting_buckets"] == count
            for task_id, count in expected.items()
        )
        else "failed"
    )
    return result


def _component_rows(
    task_id: str,
    task_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    queries = _rows(task_dir / "queries.jsonl")
    qrels = _rows(task_dir / "qrels.jsonl")
    evidence = _rows(task_dir / "qrel_evidence.jsonl")
    qrels_by_query: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in qrels:
        qrels_by_query[str(row["query_id"])].append(row)
    evidence_by_query: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in evidence:
        evidence_by_query[str(row["query_id"])].append(row)
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in queries:
        normalized = normalize_legal_input_v1(row.get("text"))
        if not normalized:
            raise ValueError(f"empty normalized input: {task_id}:{row.get('query_id')}")
        buckets[normalized].append(row)

    repaired_queries: list[dict[str, Any]] = []
    repaired_qrels: list[dict[str, Any]] = []
    repaired_evidence: list[dict[str, Any]] = []
    lineage_queries: list[dict[str, Any]] = []
    lineage_qrels: list[dict[str, Any]] = []
    for normalized, members in sorted(buckets.items()):
        splits = {str(row["split"]) for row in members}
        if len(splits) != 1:
            raise ValueError(
                f"normalized component crosses source splits: {task_id}:{sorted(splits)}"
            )
        split = next(iter(splits))
        component_hash = text_sha256(normalized)
        component_id = f"{task_id.lower()}-component:{component_hash[:32]}"
        query_id = f"{task_id.lower()}-query:{component_hash[:32]}"
        originals = sorted(members, key=lambda row: str(row["query_id"]))
        original_ids = [str(row["query_id"]) for row in originals]
        positive_ids = sorted(
            {
                str(qrel["candidate_id"])
                for original_id in original_ids
                for qrel in qrels_by_query[original_id]
            }
        )
        if not positive_ids:
            raise ValueError(f"component without positives: {component_id}")
        repaired_queries.append(
            {
                "task_id": task_id,
                "query_id": query_id,
                "component_id": component_id,
                "group_id": component_id,
                "split": split,
                "text": normalized,
                "normalization_version": NORMALIZATION_VERSION,
                "normalized_text_sha256": component_hash,
                "original_query_count": len(originals),
                "positive_count": len(positive_ids),
            }
        )
        for row in originals:
            lineage_queries.append(
                {
                    **row,
                    "original_query_id": str(row["query_id"]),
                    "repaired_query_id": query_id,
                    "repaired_component_id": component_id,
                    "normalization_version": NORMALIZATION_VERSION,
                    "normalized_text_sha256": component_hash,
                }
            )
        emitted: set[str] = set()
        for original_id in original_ids:
            for qrel in qrels_by_query[original_id]:
                candidate_id = str(qrel["candidate_id"])
                lineage_qrels.append(
                    {
                        **qrel,
                        "original_query_id": original_id,
                        "repaired_query_id": query_id,
                        "repaired_component_id": component_id,
                    }
                )
                if candidate_id not in emitted:
                    repaired_qrels.append(
                        {
                            "query_id": query_id,
                            "candidate_id": candidate_id,
                            "relevance": max(
                                int(item.get("relevance", 1))
                                for member_id in original_ids
                                for item in qrels_by_query[member_id]
                                if str(item["candidate_id"]) == candidate_id
                            ),
                            "reference_class": "union_of_observed_documentary_relations",
                            "component_id": component_id,
                        }
                    )
                    emitted.add(candidate_id)
            for ordinal, row in enumerate(evidence_by_query[original_id], start=1):
                repaired_evidence.append(
                    {
                        **row,
                        "query_id": query_id,
                        "original_query_id": original_id,
                        "repaired_component_id": component_id,
                        "original_evidence_ordinal": ordinal,
                        "original_evidence_sha256": hashlib.sha256(
                            canonical_bytes(row)
                        ).hexdigest(),
                    }
                )
    return (
        repaired_queries,
        repaired_qrels,
        repaired_evidence,
        lineage_queries,
        lineage_qrels,
    )


def _write_trec(path: Path, qrels: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        f"{row['query_id']} 0 {row['candidate_id']} {int(row.get('relevance', 1))}\n"
        for row in sorted(qrels, key=lambda row: (str(row["query_id"]), str(row["candidate_id"])))
    ]
    path.write_text("".join(lines), encoding="utf-8", newline="\n")


def _fold_id(component_id: str) -> int:
    return int(digest(["legal-deepening-fold-v1", component_id])[:8], 16) % 5


def build_repaired_release(
    destination: Path = DEFAULT_RELEASE,
    source_release: Path = SOURCE_RELEASE,
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    source_release = Path(source_release).resolve()
    if destination.exists():
        raise FileExistsError(f"immutable destination already exists: {destination}")
    source_manifest_before = sha256_file(source_release / "manifest.json")
    defect_census = reproduce_source_defects(source_release)
    if defect_census["status"] != "passed":
        raise ValueError("source defect census did not reproduce 4 and 54 buckets")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=destination.name + ".", dir=destination.parent))
    try:
        (staging / "entities").mkdir()
        (staging / "tasks").mkdir()
        (staging / "method").mkdir()
        shutil.copy2(
            source_release / "entities/provisions.parquet",
            staging / "entities/provisions.parquet",
        )
        preview = source_release / "entities/provisions_preview.jsonl"
        if preview.is_file():
            shutil.copy2(preview, staging / "entities/provisions_preview.jsonl")
        atomic_write_json(staging / "source_defect_census.json", defect_census)

        task_summaries: dict[str, Any] = {}
        for task_id in TASK_IDS:
            source_task = source_release / "tasks" / task_id
            task_dir = staging / "tasks" / task_id
            method_dir = staging / "method" / task_id
            lineage_dir = task_dir / "lineage"
            evaluation_dir = task_dir / "evaluation"
            training_dir = task_dir / "training"
            for path in (task_dir, method_dir, lineage_dir, evaluation_dir, training_dir):
                path.mkdir(parents=True, exist_ok=True)
            (
                queries,
                qrels,
                evidence,
                lineage_queries,
                lineage_qrels,
            ) = _component_rows(task_id, source_task)
            qrels_by_query: dict[str, list[str]] = collections.defaultdict(list)
            for row in qrels:
                qrels_by_query[str(row["query_id"])].append(str(row["candidate_id"]))
            atomic_write_jsonl(task_dir / "queries.jsonl", queries)
            atomic_write_jsonl(task_dir / "qrels.jsonl", qrels)
            atomic_write_jsonl(task_dir / "qrel_evidence.jsonl", evidence)
            atomic_write_jsonl(lineage_dir / "original_queries.jsonl", lineage_queries)
            atomic_write_jsonl(lineage_dir / "original_qrels.jsonl", lineage_qrels)
            atomic_write_jsonl(
                task_dir / "groups.jsonl",
                (
                    {
                        "group_id": row["component_id"],
                        "component_id": row["component_id"],
                        "split": row["split"],
                        "query_id": row["query_id"],
                        "original_query_count": row["original_query_count"],
                        "positive_count": row["positive_count"],
                    }
                    for row in queries
                ),
            )
            _write_trec(task_dir / "qrels.trec", qrels)
            source_contract = read_json(source_task / "contract.json")
            atomic_write_json(
                task_dir / "contract.json",
                {
                    **source_contract,
                    "status": "repaired_exploratory_release",
                    "normalization_version": NORMALIZATION_VERSION,
                    "equivalence_policy": "one normalized input per component",
                    "positive_policy": "union every observed source positive in the component",
                    "split_policy": "preserve source split only when every component member agrees",
                    "scientific_scope": "exploratory; v4 qrels were exposed before this repair",
                },
            )
            shutil.copy2(source_task / "exclusions.json", task_dir / "exclusions.json")
            atomic_write_json(
                task_dir / "exposure.json",
                {
                    "historical_v4_qrels_exposed": True,
                    "confirmation_eligible": False,
                    "method_allowed_splits": list(METHOD_SPLITS),
                    "test_method_access": False,
                },
            )
            split_counts = collections.Counter(str(row["split"]) for row in queries)
            examples = []
            for row in queries:
                if row["split"] == "development" and len(examples) < 3:
                    examples.append(
                        {
                            "query_id": row["query_id"],
                            "text": row["text"],
                            "positive_candidate_ids": qrels_by_query[row["query_id"]],
                        }
                    )
            atomic_write_json(task_dir / "examples.json", examples)
            candidate_receipt = _receipt(staging / "entities/provisions.parquet")
            atomic_write_json(
                task_dir / "candidate_set.json",
                {
                    "path": "entities/provisions.parquet",
                    "rows": 36887,
                    **candidate_receipt,
                },
            )

            folds = [
                {
                    "query_id": row["query_id"],
                    "component_id": row["component_id"],
                    "fold_id": _fold_id(str(row["component_id"])),
                    "role": "internal_validation"
                    if _fold_id(str(row["component_id"])) == 0
                    else "fit",
                }
                for row in queries
                if row["split"] == "train"
            ]
            atomic_write_jsonl(training_dir / "component_folds.jsonl", folds)
            atomic_write_jsonl(
                training_dir / "qrels.jsonl",
                (row for row in qrels if split_counts and next(
                    query["split"] for query in queries if query["query_id"] == row["query_id"]
                ) == "train"),
            )
            query_by_id = {str(row["query_id"]): row for row in queries}
            query_paths: dict[str, str] = {}
            reference_paths: dict[str, str] = {}
            for split in METHOD_SPLITS:
                safe_queries = [
                    {"query_id": row["query_id"], "text": row["text"], "split": split}
                    for row in queries
                    if row["split"] == split
                ]
                references = [
                    {
                        "query_id": row["query_id"],
                        "component_id": row["component_id"],
                        "positive_candidate_ids": sorted(qrels_by_query[row["query_id"]]),
                    }
                    for row in queries
                    if row["split"] == split
                ]
                query_file = method_dir / f"queries_{split}.jsonl"
                reference_file = evaluation_dir / f"{split}_reference.jsonl"
                atomic_write_jsonl(query_file, safe_queries)
                atomic_write_jsonl(reference_file, references)
                query_paths[split] = query_file.relative_to(staging).as_posix()
                reference_paths[split] = reference_file.relative_to(staging).as_posix()
            atomic_write_json(
                method_dir / "access_policy.json",
                {
                    "schema_version": 1,
                    "task_id": task_id,
                    "queries": query_paths,
                    "references": reference_paths,
                    "candidate_path": "entities/provisions.parquet",
                    "forbidden_reference_splits": ["test"],
                    "policy": "deny unless a split has an explicit reference path",
                },
            )
            atomic_write_json(
                task_dir / "manifest.json",
                {
                    "schema_version": 1,
                    "program_id": PROGRAM_ID,
                    "task_id": task_id,
                    "status": "released_exploratory_repaired",
                    "counts": {
                        "queries": len(queries),
                        "qrels": len(qrels),
                        "qrel_evidence": len(evidence),
                        "original_queries": len(lineage_queries),
                        "original_qrels": len(lineage_qrels),
                        "splits": dict(sorted(split_counts.items())),
                        "candidates": 36887,
                    },
                },
            )
            task_summaries[task_id] = read_json(task_dir / "manifest.json")["counts"]

        manifest = {
            "schema_version": 1,
            "release_id": destination.name,
            "program_id": PROGRAM_ID,
            "status": "immutable_repaired_exploratory_release",
            "immutable": True,
            "created_utc": utc_now(),
            "information_cutoff_utc": read_json(source_release / "manifest.json").get(
                "information_cutoff_utc"
            ),
            "normalization_version": NORMALIZATION_VERSION,
            "released_tasks": list(TASK_IDS),
            "source_release": source_release.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": source_manifest_before,
            "candidate_universe": {
                "path": "entities/provisions.parquet",
                "rows": 36887,
                **_receipt(staging / "entities/provisions.parquet"),
                "shared_by": list(TASK_IDS),
            },
            "task_counts": task_summaries,
            "custody": {
                "v4_qrels_exposed": True,
                "confirmation_eligible": False,
                "method_splits": list(METHOD_SPLITS),
                "test_method_access": False,
                "future_prospective_cohort_required": True,
            },
            "outputs": _all_receipts(staging),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        if sha256_file(source_release / "manifest.json") != source_manifest_before:
            raise RuntimeError("source release manifest changed during repair")
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return read_json(destination / "manifest.json")


def validate_repaired_release(release: Path = DEFAULT_RELEASE) -> dict[str, Any]:
    release = Path(release).resolve()
    manifest = read_json(release / "manifest.json")
    failures: list[str] = []
    for relative, receipt in manifest.get("outputs", {}).items():
        path = release / relative
        if not path.is_file():
            failures.append(f"missing:{relative}")
        elif _receipt(path) != receipt:
            failures.append(f"receipt:{relative}")
    source = ROOT / manifest["source_release"]
    if sha256_file(source / "manifest.json") != manifest["source_manifest_sha256"]:
        failures.append("source_manifest_changed")
    candidate = release / manifest["candidate_universe"]["path"]
    if _receipt(candidate) != {
        "bytes": manifest["candidate_universe"]["bytes"],
        "sha256": manifest["candidate_universe"]["sha256"],
    }:
        failures.append("candidate_universe_receipt")

    for task_id in TASK_IDS:
        task_dir = release / "tasks" / task_id
        queries = _rows(task_dir / "queries.jsonl")
        qrels = _rows(task_dir / "qrels.jsonl")
        evidence = _rows(task_dir / "qrel_evidence.jsonl")
        lineage_queries = _rows(task_dir / "lineage/original_queries.jsonl")
        lineage_qrels = _rows(task_dir / "lineage/original_qrels.jsonl")
        query_by_id = {str(row["query_id"]): row for row in queries}
        if len(query_by_id) != len(queries):
            failures.append(f"duplicate_query:{task_id}")
        if any(
            normalize_legal_input_v1(row["text"]) != row["text"] for row in queries
        ):
            failures.append(f"noncanonical_input:{task_id}")
        component_splits: dict[str, set[str]] = collections.defaultdict(set)
        for row in queries:
            component_splits[str(row["component_id"])].add(str(row["split"]))
        if any(len(values) != 1 for values in component_splits.values()):
            failures.append(f"component_split:{task_id}")
        positive_pairs = {
            (str(row["query_id"]), str(row["candidate_id"])) for row in qrels
        }
        mapped_pairs = {
            (str(row["repaired_query_id"]), str(row["candidate_id"]))
            for row in lineage_qrels
        }
        if positive_pairs != mapped_pairs:
            failures.append(f"positive_union:{task_id}")
        if len(evidence) != len(_rows(source / "tasks" / task_id / "qrel_evidence.jsonl")):
            failures.append(f"evidence_count:{task_id}")
        if any(str(row["query_id"]) not in query_by_id for row in evidence):
            failures.append(f"evidence_mapping:{task_id}")
        source_query_count = len(_rows(source / "tasks" / task_id / "queries.jsonl"))
        if len(lineage_queries) != source_query_count:
            failures.append(f"query_lineage:{task_id}")
        try:
            MethodSafeAdapter(release, task_id).load_references("test")
            failures.append(f"test_reference_opened:{task_id}")
        except ForbiddenPartitionAccess:
            pass

    strict = audit_release(release)
    if strict["status"] != "passed":
        failures.extend(
            f"strict:{row['check']}:{row.get('task_id', '')}"
            for row in strict["failures"]
        )
    return {
        "schema_version": 1,
        "validated_utc": utc_now(),
        "release": str(release),
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "strict_cross_release_gate": strict,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--source", type=Path, default=SOURCE_RELEASE)
    build.add_argument("--destination", type=Path, default=DEFAULT_RELEASE)
    build.add_argument("--output", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    validate.add_argument("--output", type=Path)
    audit = subparsers.add_parser("audit-source")
    audit.add_argument("--source", type=Path, default=SOURCE_RELEASE)
    audit.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_repaired_release(args.destination, args.source)
    elif args.command == "validate":
        result = validate_repaired_release(args.release)
    else:
        result = reproduce_source_defects(args.source)
    if args.output:
        atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status", "").startswith(("passed", "immutable")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
