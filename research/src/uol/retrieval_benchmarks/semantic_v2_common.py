"""Common, leak-resistant interfaces for the second semantic campaign."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CAMPAIGN_ID = "semantic_benchmarks_sol_luna_20260916_v2"
SPLITS = {"train", "development", "test"}


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class QueryView:
    query_id: str
    text: str
    split: str
    component_id: str
    information_cutoff: str | None = None
    scope_id: str | None = None
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.query_id or not self.text or not self.component_id:
            raise ValueError("query_id, text and component_id are required")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split: {self.split}")
        forbidden = {"qrels", "positive_ids", "target_id", "target_ids", "member_ids"}
        exposed = forbidden.intersection(self.public_metadata)
        if exposed:
            raise PermissionError(f"query metadata exposes evaluation fields: {sorted(exposed)}")

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CandidateView:
    candidate_id: str
    original_text: str
    source_id: str
    version_id: str | None = None
    public_hierarchy: Mapping[str, str] = field(default_factory=dict)
    allowed_spans: tuple[tuple[int, int], ...] = ()
    public_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_id:
            raise ValueError("candidate_id and source_id are required")
        for begin, end in self.allowed_spans:
            if begin < 0 or end <= begin or end > len(self.original_text):
                raise ValueError(f"invalid candidate span: {(begin, end)}")
        forbidden = {"relevance", "is_positive", "qrels", "query_ids", "target_for"}
        exposed = forbidden.intersection(self.public_metadata)
        if exposed:
            raise PermissionError(f"candidate metadata exposes evaluation fields: {sorted(exposed)}")


@dataclass(frozen=True)
class TrainingRelation:
    source_id: str
    target_id: str
    relation_type: str
    split: str
    provenance: Mapping[str, Any]
    weak_label: bool = False

    def __post_init__(self) -> None:
        if self.split != "train":
            raise PermissionError("training relations must be train-only")
        if not self.source_id or not self.target_id or not self.relation_type:
            raise ValueError("training relation identifiers and type are required")


@dataclass(frozen=True)
class EvaluationReference:
    query_id: str
    candidate_id: str
    component_id: str
    relation_type: str
    source_id: str
    source_field: str
    source_span: tuple[int, int] | None
    witness_sha256: str
    documentary_identity_verified: bool
    legal_truth: bool | None
    exposure: str

    def __post_init__(self) -> None:
        if self.source_span is not None:
            begin, end = self.source_span
            if begin < 0 or end <= begin:
                raise ValueError("evaluation source span is invalid")
        if self.documentary_identity_verified and not self.witness_sha256:
            raise ValueError("verified documentary identity requires a witness hash")
        if self.exposure not in {"unopened", "historically_exposed", "custodian_only"}:
            raise ValueError(f"unknown exposure state: {self.exposure}")


@dataclass(frozen=True)
class RunContract:
    contract_id: str
    task_id: str
    release_ids: tuple[str, ...]
    candidate_set_sha256: str
    component_set_sha256: str
    primary_metric: str
    query_fields: tuple[str, ...]
    candidate_fields: tuple[str, ...]
    controls: tuple[str, ...]
    max_searches: int | None = None
    max_unique_read_tokens: int | None = None
    test_opened_before_freeze: bool = False

    def __post_init__(self) -> None:
        if self.test_opened_before_freeze:
            raise PermissionError("cannot freeze a contract after opening test")
        if not self.release_ids or not self.controls:
            raise ValueError("release_ids and controls are required")
        if self.max_searches is not None and self.max_searches < 1:
            raise ValueError("max_searches must be positive")
        if self.max_unique_read_tokens is not None and self.max_unique_read_tokens < 1:
            raise ValueError("max_unique_read_tokens must be positive")

    def frozen(self) -> dict[str, Any]:
        payload = {"schema_version": 1, "campaign_id": CAMPAIGN_ID, **asdict(self)}
        payload["contract_sha256"] = digest(payload)
        return payload


def verify_contract(value: Mapping[str, Any]) -> None:
    expected = str(value["contract_sha256"])
    payload = {key: item for key, item in value.items() if key != "contract_sha256"}
    if digest(payload) != expected:
        raise ValueError("contract hash mismatch")
    if value.get("campaign_id") != CAMPAIGN_ID:
        raise ValueError("wrong campaign identity")
    if value.get("test_opened_before_freeze") is not False:
        raise PermissionError("test boundary was not preserved")


def method_bundle(
    queries: Sequence[QueryView],
    candidates: Sequence[CandidateView],
    training_relations: Sequence[TrainingRelation] = (),
) -> dict[str, Any]:
    """Return only method-visible objects. Evaluation references are a separate type."""
    return {
        "campaign_id": CAMPAIGN_ID,
        "queries": [asdict(row) for row in queries],
        "candidates": [asdict(row) for row in candidates],
        "training_relations": [asdict(row) for row in training_relations],
    }


def evaluation_bundle(references: Sequence[EvaluationReference]) -> dict[str, Any]:
    return {
        "campaign_id": CAMPAIGN_ID,
        "references": [asdict(row) for row in references],
    }


def assert_disjoint_bundles(method: Mapping[str, Any], evaluation: Mapping[str, Any]) -> None:
    if "references" in method or "queries" in evaluation or "candidates" in evaluation:
        raise PermissionError("method and evaluation bundles are not separated")
    method_text = stable_json(method).casefold()
    for forbidden in ('"qrels"', '"positive_ids"', '"target_id"', '"member_ids"'):
        if forbidden in method_text:
            raise PermissionError(f"method bundle exposes forbidden field: {forbidden}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_span_coverage(
    references: Iterable[tuple[int, int]], selected_spans: Iterable[tuple[int, int]]
) -> dict[str, float | int]:
    """Measure exact and fractional reference coverage using unique source positions."""
    selected: set[int] = set()
    for begin, end in selected_spans:
        if begin < 0 or end <= begin:
            raise ValueError("selected span is invalid")
        selected.update(range(begin, end))
    total = covered = exact = count = 0
    for begin, end in references:
        if begin < 0 or end <= begin:
            raise ValueError("reference span is invalid")
        positions = set(range(begin, end))
        hit = len(positions.intersection(selected))
        total += len(positions)
        covered += hit
        exact += int(hit == len(positions))
        count += 1
    return {
        "references": count,
        "fully_covered": exact,
        "reference_recall": exact / count if count else 0.0,
        "fractional_coverage": covered / total if total else 0.0,
        "unique_source_positions_read": len(selected),
    }


def source_span_coverage_by_source(
    references: Iterable[tuple[str, int, int]],
    selected_spans: Iterable[tuple[str, int, int]],
) -> dict[str, Any]:
    """Measure spans independently per source so equal offsets never collide."""
    refs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    selected: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for source_id, begin, end in references:
        refs[source_id].append((begin, end))
    for source_id, begin, end in selected_spans:
        selected[source_id].append((begin, end))
    per_source = {source_id: source_span_coverage(values, selected.get(source_id, ())) for source_id, values in refs.items()}
    total_references = sum(int(row["references"]) for row in per_source.values())
    total_full = sum(int(row["fully_covered"]) for row in per_source.values())
    total_positions = sum(sum(end - begin for begin, end in refs[source_id]) for source_id in refs)
    covered_positions = sum(float(row["fractional_coverage"]) * sum(end - begin for begin, end in refs[source_id])
                            for source_id, row in per_source.items())
    return {
        "sources": len(per_source),
        "references": total_references,
        "fully_covered": total_full,
        "reference_recall": total_full / total_references if total_references else 0.0,
        "fractional_coverage": covered_positions / total_positions if total_positions else 0.0,
        "unique_source_positions_read": sum(int(row["unique_source_positions_read"]) for row in per_source.values()),
        "per_source": per_source,
    }
