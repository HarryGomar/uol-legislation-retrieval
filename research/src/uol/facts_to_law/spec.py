"""Shared file formats, identifiers and helpers for the facts-to-law v2 program.

All modules in ``uol.facts_to_law`` exchange data only through the formats
defined here. Changing a format requires a new FORMAT_VERSION and a new release.

Release directory (``data/releases/<release_id>/``)::

    manifest.json            Manifest (see ``Manifest``)
    queries.jsonl            one ``QueryRecord`` per line, release order
    references.jsonl         one ``ReferenceRecord`` per line, same order
    target_types.jsonl       one ``TargetTypeRecord`` per cited candidate
    folds.json               {"salt": str, "k": int, "components": {component_id: fold}}
    audit/                   audit samples and policy comparisons (not model inputs)

Candidates are NOT copied: the manifest binds an existing candidate release
(articles.jsonl / instruments.jsonl) by path and SHA-256.

Run directory (``experiments/runs/<run_id>/``)::

    run_lock.json            frozen systems, grids, seeds, code hashes
    scores/<system>.npy      optional float16 memmap [n_queries, n_candidates]
    details/<system>.jsonl   one ``DetailRecord`` per query (positive ranks, top100)
    metrics/<system>.json    component-macro metrics per stratum
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

FORMAT_VERSION = "facts-to-law-v2-format-1"
ROOT = Path(__file__).resolve().parents[3]
RELEASES = ROOT / "data" / "releases"
RUNS = ROOT / "experiments" / "runs"

# Query strata (declared before scoring)
S1_NO_CITATION = "S1_no_citation"          # facts name no statute; no mask applied
S2_MASKED_CITATION = "S2_masked_citation"  # facts named >=1 statute/article; masked
STRATA = (S1_NO_CITATION, S2_MASKED_CITATION)

# Reference resolution status per query
FULL = "full"        # every article expression in Criterio/Justificacion resolved
PARTIAL = "partial"  # >=1 resolved positive, >=1 unresolved expression
RESOLUTION_STATUSES = (FULL, PARTIAL)

# Target types (analysis-only; never a model input)
GENERIC = "generic"
SPECIFIC = "specific"

# Fitting-fold popularity bands (computed per outer fold from fitting folds only)
UNSEEN, RARE, FREQUENT = "unseen", "rare", "frequent"  # 0 / 1-4 / 5+ fitting components

MASK_TOKEN = "[CITA_OCULTA]"


@dataclass
class QueryRecord:
    query_id: str
    record_id: str
    component_id: str
    fold: int
    source_split: str                 # provenance split in the predecessor (e.g. v1_train, v1_dev)
    text: str                         # target-blind masked facts (model input)
    text_sha256: str
    stratum: str                      # S1_no_citation | S2_masked_citation
    resolution_status: str            # full | partial
    n_unresolved: int
    flags: dict[str, bool] = field(default_factory=dict)  # quote_bearing, residual_statute_name, residual_number_after_mask, version_cue
    materias: list[str] = field(default_factory=list)
    year: int | None = None
    tipo: str | None = None           # Jurisprudencia | Aislada
    instancia: str | None = None


@dataclass
class Positive:
    candidate_id: str
    rule_ids: list[str]               # resolver rules that produced this identity
    section: str                      # criterio | justificacion
    span: list[int]                   # [start, end) in the section text
    surface_sha256: str


@dataclass
class Unresolved:
    reason: str
    section: str
    span: list[int]
    surface_sha256: str


@dataclass
class ReferenceRecord:
    query_id: str
    positives: list[Positive]
    unresolved: list[Unresolved] = field(default_factory=list)

    @property
    def positive_ids(self) -> list[str]:
        seen: list[str] = []
        for p in self.positives:
            if p.candidate_id not in seen:
                seen.append(p.candidate_id)
        return seen


@dataclass
class TargetTypeRecord:
    candidate_id: str
    target_type: str                  # generic | specific
    rule: str
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetailRecord:
    query_id: str
    component_id: str
    fold: int
    positive_ids: list[str]
    positive_ranks: list[int]         # 1-based full-catalogue ranks, same order as positive_ids
    top100: list[str]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fold_of(component_id: str, salt: str, k: int = 5) -> int:
    """Deterministic component-level fold assignment."""
    return int(sha256_text(f"{salt}:{component_id}"), 16) % k


def read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    """Write rows (dicts or dataclasses) as UTF-8 JSONL with LF endings; returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            if hasattr(row, "__dataclass_fields__"):
                row = asdict(row)
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
    return n


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def load_queries(release_dir: Path) -> list[QueryRecord]:
    return [QueryRecord(**row) for row in read_jsonl(Path(release_dir) / "queries.jsonl")]


def load_references(release_dir: Path) -> list[ReferenceRecord]:
    out = []
    for row in read_jsonl(Path(release_dir) / "references.jsonl"):
        out.append(ReferenceRecord(
            query_id=row["query_id"],
            positives=[Positive(**p) for p in row["positives"]],
            unresolved=[Unresolved(**u) for u in row.get("unresolved", [])],
        ))
    return out


def load_target_types(release_dir: Path) -> dict[str, TargetTypeRecord]:
    return {row["candidate_id"]: TargetTypeRecord(**row) for row in read_jsonl(Path(release_dir) / "target_types.jsonl")}


def load_candidate_ids(articles_jsonl: Path) -> list[str]:
    return [row["candidate_id"] for row in read_jsonl(articles_jsonl)]
