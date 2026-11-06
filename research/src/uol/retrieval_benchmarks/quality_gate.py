"""Cross-release quality gate for retrieval datasets.

Release-specific validators prove lineage and schema details.  This module
checks properties that must hold for every benchmark presented to the same
text-only ranking interface: unambiguous labels, leakage-safe splits, strict
JSON, and durable metadata dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

from uol.retrieval_benchmarks.common import ROOT, utc_now


class StrictJsonError(ValueError):
    """A control or dataset file is not strict JSON."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StrictJsonError(f"duplicate_key:{key}")
        result[key] = value
    return result


def _finite_number(value: str) -> None:
    raise StrictJsonError(f"nonfinite_number:{value}")


def strict_loads(payload: str, *, source: str = "<string>") -> Any:
    if payload.startswith("\ufeff"):
        raise StrictJsonError(f"bom_not_allowed:{source}")
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_finite_number,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise StrictJsonError(f"invalid_json:{source}:{exc}") from exc


def strict_read_json(path: Path) -> Any:
    try:
        payload = path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise StrictJsonError(f"unreadable_json:{path}:{exc}") from exc
    return strict_loads(payload, source=str(path))


def strict_iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise StrictJsonError(f"unreadable_jsonl:{path}:{exc}") from exc
    with stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                raise StrictJsonError(f"blank_jsonl_row:{path}:{number}")
            value = strict_loads(line, source=f"{path}:{number}")
            if not isinstance(value, dict):
                raise StrictJsonError(f"jsonl_row_not_object:{path}:{number}")
            yield value


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def audit_task_inputs(
    task_id: str,
    queries: Iterable[dict[str, Any]],
    qrels: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    query_rows = list(queries)
    qrel_rows = list(qrels)
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    query_by_id: dict[str, dict[str, Any]] = {}
    for row in query_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            failures.append({"check": "query_id", "task_id": task_id})
            continue
        if query_id in query_by_id:
            failures.append(
                {"check": "duplicate_query_id", "task_id": task_id, "query_id": query_id}
            )
        query_by_id[query_id] = row

    targets_by_query: dict[str, set[str]] = defaultdict(set)
    qrel_keys: set[tuple[str, str]] = set()
    for row in qrel_rows:
        query_id = row.get("query_id")
        candidate_id = row.get("candidate_id")
        if not isinstance(query_id, str) or not query_id:
            failures.append({"check": "qrel_query_id", "task_id": task_id})
            continue
        if query_id not in query_by_id:
            failures.append(
                {"check": "qrel_unknown_query", "task_id": task_id, "query_id": query_id}
            )
        if not isinstance(candidate_id, str) or not candidate_id:
            failures.append(
                {"check": "qrel_candidate_id", "task_id": task_id, "query_id": query_id}
            )
        else:
            key = (query_id, candidate_id)
            if key in qrel_keys:
                failures.append(
                    {
                        "check": "duplicate_qrel",
                        "task_id": task_id,
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                    }
                )
            qrel_keys.add(key)
            targets_by_query[query_id].add(candidate_id)

    by_text: dict[str, list[dict[str, Any]]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    for row in query_rows:
        query_id = str(row.get("query_id"))
        text = _normalise_text(row.get("text"))
        if not text:
            failures.append(
                {"check": "empty_model_input", "task_id": task_id, "query_id": query_id}
            )
        by_text[text].append(row)
        group_id = row.get("group_id")
        split = row.get("split")
        if not isinstance(group_id, str) or not group_id:
            failures.append(
                {"check": "query_group_id", "task_id": task_id, "query_id": query_id}
            )
        elif not isinstance(split, str) or not split:
            failures.append(
                {"check": "query_split", "task_id": task_id, "query_id": query_id}
            )
        else:
            groups[group_id].add(split)
        if not targets_by_query.get(query_id):
            failures.append(
                {"check": "query_without_qrel", "task_id": task_id, "query_id": query_id}
            )

    split_leak_groups = sorted(group for group, splits in groups.items() if len(splits) > 1)
    if split_leak_groups:
        failures.append(
            {
                "check": "group_crosses_splits",
                "task_id": task_id,
                "count": len(split_leak_groups),
                "examples": split_leak_groups[:5],
            }
        )

    duplicate_inputs = 0
    conflicting_inputs = 0
    cross_split_inputs = 0
    conflict_examples: list[dict[str, Any]] = []
    cross_split_examples: list[dict[str, Any]] = []
    for text, rows in by_text.items():
        if len(rows) < 2:
            continue
        duplicate_inputs += 1
        target_sets = {
            tuple(sorted(targets_by_query[str(row.get("query_id"))])) for row in rows
        }
        splits = {str(row.get("split")) for row in rows}
        if len(target_sets) > 1:
            conflicting_inputs += 1
            if len(conflict_examples) < 5:
                conflict_examples.append(
                    {
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "query_ids": [str(row.get("query_id")) for row in rows[:8]],
                        "target_sets": [
                            {"count": len(items), "candidate_ids": list(items[:8])}
                            for items in sorted(target_sets)[:8]
                        ],
                    }
                )
        if len(splits) > 1:
            cross_split_inputs += 1
            if len(cross_split_examples) < 5:
                cross_split_examples.append(
                    {
                        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "query_ids": [str(row.get("query_id")) for row in rows[:8]],
                        "splits": sorted(splits),
                    }
                )

    if conflicting_inputs:
        failures.append(
            {
                "check": "identical_model_input_conflicting_qrels",
                "task_id": task_id,
                "count": conflicting_inputs,
                "examples": conflict_examples,
            }
        )
    if cross_split_inputs:
        failures.append(
            {
                "check": "identical_model_input_crosses_splits",
                "task_id": task_id,
                "count": cross_split_inputs,
                "examples": cross_split_examples,
            }
        )
    if duplicate_inputs and not conflicting_inputs and not cross_split_inputs:
        warnings.append(
            {
                "check": "duplicate_model_input_weighting",
                "task_id": task_id,
                "count": duplicate_inputs,
            }
        )

    return {
        "task_id": task_id,
        "queries": len(query_rows),
        "qrels": len(qrel_rows),
        "unique_model_inputs": len(by_text),
        "duplicate_model_inputs": duplicate_inputs,
        "conflicting_model_inputs": conflicting_inputs,
        "cross_split_model_inputs": cross_split_inputs,
        "failures": failures,
        "warnings": warnings,
    }


def _cache_references(value: Any, trail: str = "$") -> Iterator[dict[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _cache_references(child, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _cache_references(child, f"{trail}[{index}]")
    elif isinstance(value, str):
        normal = value.replace("\\", "/")
        if normal.startswith(".cache/") or "/.cache/" in normal:
            yield {"trail": trail, "path": value}


def audit_release(release: Path) -> dict[str, Any]:
    release = release.resolve()
    failures: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []
    manifest_path = release / "manifest.json"
    try:
        manifest = strict_read_json(manifest_path)
    except StrictJsonError as exc:
        return {
            "release": str(release),
            "status": "failed",
            "failure_count": 1,
            "failures": [{"check": "strict_manifest", "detail": str(exc)}],
            "warnings": [],
            "tasks": [],
        }
    cache_refs = list(_cache_references(manifest))
    if cache_refs:
        failures.append(
            {
                "check": "nondurable_release_dependency",
                "count": len(cache_refs),
                "examples": cache_refs[:8],
            }
        )

    task_root = release / "tasks"
    if task_root.is_dir():
        for task_dir in sorted(path for path in task_root.iterdir() if path.is_dir()):
            query_path = task_dir / "queries.jsonl"
            qrel_path = task_dir / "qrels.jsonl"
            if not query_path.is_file() or not qrel_path.is_file():
                continue
            try:
                result = audit_task_inputs(
                    task_dir.name,
                    strict_iter_jsonl(query_path),
                    strict_iter_jsonl(qrel_path),
                )
            except StrictJsonError as exc:
                failures.append(
                    {"check": "strict_task_jsonl", "task_id": task_dir.name, "detail": str(exc)}
                )
                continue
            task_results.append(result)
            failures.extend(result["failures"])
            warnings.extend(result["warnings"])

    return {
        "release": str(release),
        "release_id": manifest.get("release_id") if isinstance(manifest, dict) else None,
        "status": "passed" if not failures else "failed",
        "failure_count": len(failures),
        "failures": failures,
        "warnings": warnings,
        "tasks": task_results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", action="append", type=Path, required=True)
    args = parser.parse_args(argv)
    results = [audit_release(path) for path in args.release]
    payload = {
        "schema_version": 1,
        "audited_utc": utc_now(),
        "status": "passed" if all(row["status"] == "passed" for row in results) else "failed",
        "releases": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
