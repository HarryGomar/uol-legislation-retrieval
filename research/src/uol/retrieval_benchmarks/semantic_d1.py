"""First, source-grounded D1 route for the SJF Bicentenario views release.

This module deliberately builds *documentary views*, not legal gold.  The
Parquet release contains literal citation candidates, while its relations are
currently unresolved.  Empty ``target_ids`` are therefore retained as audit
data and never promoted to targets.  The route is streaming and resumable:
``iter_records`` consumes the existing portable Parquet reader in source order
and accepts a limit/offset for bounded continuation.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import re
import unicodedata
from typing import Any, Iterable, Iterator, Mapping

from uol.datasets.sjf_bicentenario_portable import _rows_from_parquet
from uol.datasets.canonical import strict_json_loads


DEFAULT_ROOT = Path(
    "data/releases/uol_mexico_tasks_20260915/"
    "sjf_bicentenario_content_views_v2"
)
SECTION_RE = re.compile(
    r"(?im)(?P<start>^|\r?\n)[ \t]*(?P<label>Hechos|Criterio\s+jur[ií]dico|Justificaci[oó]n)\s*:\s*"
)
PERIOD_SECTION_RE = re.compile(
    r"(?im)(?P<start>^|\r?\n)[ \t]*(?P<label>Hechos|Criterio\s+jur[ií]dico|Justificaci[oó]n)\s*[:.]\s*"
)
VERIFIED_PERIOD_SECTION_IDS = frozenset(
    {"2028323", "2022565", "2022337", "2022229", "2022075"}
)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]+", re.UNICODE)


@dataclass(frozen=True)
class SourceSpan:
    block_id: str
    start: int
    end: int


@dataclass(frozen=True)
class Section:
    name: str
    text: str
    span: SourceSpan


@dataclass(frozen=True)
class CitationCandidate:
    citation_id: str
    field: str
    surface: str
    span: SourceSpan
    instrument: str | None
    instrument_literal: str | None
    articles: tuple[str, ...]
    fractions: tuple[str, ...]
    paragraphs: tuple[str, ...]
    incisos: tuple[str, ...]
    status: str
    resolution_status: str
    target_ids: tuple[str, ...]
    target_status: str
    legal_truth: bool

    @property
    def is_gold(self) -> bool:
        """Whether an explicitly resolved target may be consumed as gold.

        This is intentionally false for empty target lists and unresolved or
        deferred relations.  A citation candidate itself remains useful for
        coverage diagnostics without becoming a relevance label.
        """
        return bool(
            self.target_ids
            and self.target_status not in {"unresolved", "candidate", "candidate_unresolved"}
            and self.resolution_status not in {"unresolved", "deferred", "candidate"}
            and self.legal_truth
        )


@dataclass(frozen=True)
class CriterionCandidate:
    target_id: str
    text: str
    span: SourceSpan
    provenance: str = "published_ficha_criterion"


@dataclass(frozen=True)
class D1Record:
    record_id: str
    document_id: str
    unit_id: str
    ordinal: int
    source_sha256: str
    hechos: Section | None
    criterio_juridico: Section | None
    justificacion: Section | None
    rubro: str | None
    precedentes: Any
    materias: tuple[Any, ...]
    nota_publica: str | None
    tesis_metadata: Mapping[str, Any]
    citations: tuple[CitationCandidate, ...]
    fact_group_id: str
    procedure_group_id: str
    common_group_id: str

    @property
    def criterion_candidate(self) -> CriterionCandidate | None:
        if self.criterio_juridico is None:
            return None
        return CriterionCandidate(
            target_id=f"{self.unit_id}:criterio_juridico",
            text=self.criterio_juridico.text,
            span=self.criterio_juridico.span,
        )

    @property
    def n06_candidate(self) -> CriterionCandidate | None:
        """The K target is eligible for N06 only when the H query is present."""
        return self.criterion_candidate if self.hechos is not None else None

    @property
    def query_fields(self) -> Mapping[str, Section]:
        """Fields allowed for the H query; side metadata is intentionally absent."""
        return {"hechos": self.hechos} if self.hechos is not None else {}

    @property
    def citation_gold(self) -> tuple[str, ...]:
        """Resolved citation targets only; empty/deferred relations yield none."""
        return tuple(target for citation in self.citations for target in citation.target_ids if citation.is_gold)

    @property
    def note_candidate_ids(self) -> tuple[str, ...]:
        """Notes never define contenders; retained for an explicit audit hook."""
        return ()


def _normalise(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = _PUNCT_RE.sub(" ", text)
    return _SPACE_RE.sub(" ", text).strip()


def _group_id(prefix: str, value: Any, fallback: str) -> str:
    normal = _normalise(value)
    if not normal:
        normal = fallback
    digest = hashlib.sha256(normal.encode("utf-8")).hexdigest()[:20]
    return f"d1:{prefix}:{digest}"


def split_sections(
    text: str,
    block_id: str,
    *,
    allow_verified_period_headings: bool = False,
) -> dict[str, Section]:
    """Extract H/K/J sections with offsets into the original ``texto`` field.

    The historical parser accepts colon headings. Five inspected modern SJF
    records use a terminal period instead. Callers must opt into that narrow
    variant after checking the record identity against
    ``VERIFIED_PERIOD_SECTION_IDS``. This keeps arbitrary older prose outside
    the structured H/K/J route.
    """
    pattern = PERIOD_SECTION_RE if allow_verified_period_headings else SECTION_RE
    matches = list(pattern.finditer(text))
    sections: dict[str, Section] = {}
    names = {"hechos": "hechos", "criterio juridico": "criterio_juridico", "justificacion": "justificacion"}
    for index, match in enumerate(matches):
        key = names[_normalise(match.group("label"))]
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        # Trim only boundary whitespace while preserving offsets in source.
        raw = text[content_start:content_end]
        left_trim = len(raw) - len(raw.lstrip())
        right_trimmed = raw.rstrip()
        start = content_start + left_trim
        end = start + len(right_trimmed)
        if end > start:
            sections[key] = Section(key, text[start:end], SourceSpan(block_id, start, end))
    return sections


def _metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    value = strict_json_loads(document.get("published_json", "{}"))
    return value if isinstance(value, dict) else {}


def _rows_for_document(iterator: Iterator[dict[str, Any]], document_id: str, first: dict[str, Any] | None) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    current = first
    while current is not None and current.get("document_id") == document_id:
        rows.append(current)
        current = next(iterator, None)
    return rows, current


def _citation(row: Mapping[str, Any], relation: Mapping[str, Any] | None, block_id: str) -> CitationCandidate:
    relation = relation or {}
    targets = tuple(str(item) for item in (relation.get("target_ids") or ()) if item)
    return CitationCandidate(
        citation_id=str(row["citation_id"]), field=str(row["field"]), surface=str(row["surface"]),
        span=SourceSpan(block_id, int(row["start"]), int(row["end"])),
        instrument=row.get("instrument"), instrument_literal=row.get("instrument_literal"),
        articles=tuple(row.get("articles") or ()), fractions=tuple(row.get("fractions") or ()),
        paragraphs=tuple(row.get("paragraphs") or ()), incisos=tuple(row.get("incisos") or ()),
        status=str(row.get("status") or ""), resolution_status=str(row.get("resolution_status") or ""),
        target_ids=targets, target_status=str(relation.get("status") or "unresolved"),
        legal_truth=bool(relation.get("legal_truth", False)),
    )


def iter_records(
    root: Path | str = DEFAULT_ROOT,
    *,
    offset: int = 0,
    limit: int | None = None,
    allow_verified_period_headings: bool = False,
) -> Iterator[D1Record]:
    """Yield D1 records using the caller's explicitly versioned parser lane."""
    root = Path(root)
    documents = _rows_from_parquet(root / "documents.parquet")
    fields = _rows_from_parquet(root / "fields.parquet")
    citations = _rows_from_parquet(root / "citations.parquet")
    relations = _rows_from_parquet(root / "relations.parquet")
    field_next = next(fields, None)
    citation_next = next(citations, None)
    relation_next = next(relations, None)
    yielded = 0
    for document in documents:
        document_id = str(document["document_id"])
        field_rows, field_next = _rows_for_document(fields, document_id, field_next)
        citation_rows: list[dict[str, Any]] = []
        while citation_next is not None and int(citation_next.get("ordinal", -1)) < int(document["ordinal"]):
            citation_next = next(citations, None)
        while citation_next is not None and int(citation_next.get("ordinal", -1)) == int(document["ordinal"]):
            citation_rows.append(citation_next)
            citation_next = next(citations, None)
        relation_map: dict[str, dict[str, Any]] = {}
        # Release order is aligned, but this bounded map also tolerates a small
        # source-order perturbation without loading the relation table.
        for citation_row in citation_rows:
            while relation_next is not None and int(relation_next.get("source_sequence", 0)) < int(citation_row.get("source_sequence", 0)):
                relation_next = next(relations, None)
            if relation_next is not None and relation_next.get("citation_id") == citation_row.get("citation_id"):
                relation_map[str(citation_row["citation_id"])] = relation_next
                relation_next = next(relations, None)
        metadata = _metadata(document)
        text_row = next((row for row in field_rows if row.get("source_field") == "texto"), None)
        text = str(text_row.get("text") or "") if text_row else ""
        block_id = str(text_row.get("block_id") if text_row else f"sjf:{document['requested_id']}:published:texto")
        record_id = str(document["requested_id"])
        sections = split_sections(
            text,
            block_id,
            allow_verified_period_headings=(
                allow_verified_period_headings
                and record_id in VERIFIED_PERIOD_SECTION_IDS
            ),
        )
        hechos = sections.get("hechos")
        procedure = metadata.get("precedentes")
        fact_group = _group_id("facts", hechos.text if hechos else "", document_id)
        procedure_group = _group_id("procedure", procedure, document_id)
        common_group = _group_id("common", f"{fact_group}\0{procedure_group}", document_id)
        record = D1Record(
            record_id=record_id, document_id=document_id,
            unit_id=str(next((row["unit_id"] for row in field_rows if row.get("unit_id")), f"sjf:tesis:{document['requested_id']}")),
            ordinal=int(document["ordinal"]), source_sha256=str(document.get("body_sha256") or ""),
            hechos=hechos, criterio_juridico=sections.get("criterio_juridico"), justificacion=sections.get("justificacion"),
            rubro=metadata.get("rubro"), precedentes=procedure,
            materias=tuple(metadata.get("materias") or ()), nota_publica=metadata.get("notaPublica"),
            tesis_metadata={key: metadata[key] for key in ("idTesis", "tipoTesis", "epoca", "instancia", "fuente", "anio") if key in metadata},
            citations=tuple(_citation(row, relation_map.get(str(row["citation_id"])), block_id) for row in citation_rows),
            fact_group_id=fact_group, procedure_group_id=procedure_group, common_group_id=common_group,
        )
        if yielded >= offset:
            yield record
            if limit is not None and yielded - offset + 1 >= limit:
                return
        yielded += 1


def load_records(
    root: Path | str = DEFAULT_ROOT,
    *,
    offset: int = 0,
    limit: int | None = None,
    allow_verified_period_headings: bool = False,
) -> list[D1Record]:
    return list(
        iter_records(
            root,
            offset=offset,
            limit=limit,
            allow_verified_period_headings=allow_verified_period_headings,
        )
    )


def iter_n06_pairs(records: Iterable[D1Record]) -> Iterator[dict[str, Any]]:
    """Yield source-grounded H→K pairs for N06, without creating citation gold."""
    for record in records:
        target = record.n06_candidate
        if target is None or record.hechos is None:
            continue
        yield {
            "query_id": f"{record.unit_id}:hechos",
            "target_id": target.target_id,
            "query": record.hechos.text,
            "target": target.text,
            "query_span": asdict(record.hechos.span),
            "target_span": asdict(target.span),
            "fact_group_id": record.fact_group_id,
            "procedure_group_id": record.procedure_group_id,
            "common_group_id": record.common_group_id,
            "provenance": target.provenance,
            "gold_kind": "editorial_association",
        }


def assign_splits(records: Iterable[D1Record], *, seed: str = "semantic_d1_v1") -> dict[str, list[D1Record]]:
    """Assign common connected groups to train/development/test.

    Fact groups and procedure groups are unioned, so repeated H text cannot
    leak across splits even when its procedure metadata differs.  Hashing is
    stable and requires no random state or mutable checkpoint.
    """
    records = list(records)
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left, right = find(left), find(right)
        if left != right:
            parent[right] = left

    for record in records:
        union(record.fact_group_id, record.procedure_group_id)
        union(record.fact_group_id, record.common_group_id)
    groups: dict[str, list[D1Record]] = {}
    for record in records:
        groups.setdefault(find(record.fact_group_id), []).append(record)
    output: dict[str, list[D1Record]] = {"train": [], "development": [], "test": []}
    for group_id, members in sorted(groups.items()):
        fraction = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 2**32
        split = "train" if fraction < 0.8 else "development" if fraction < 0.9 else "test"
        output[split].extend(members)
    for rows in output.values():
        rows.sort(key=lambda row: row.ordinal)
    return output


def funnel(records: Iterable[D1Record]) -> dict[str, Any]:
    """Return auditable counts, explicitly distinguishing candidates/gold."""
    rows = list(records)
    citations = [citation for row in rows for citation in row.citations]
    criterion_sections = [row for row in rows if row.criterion_candidate is not None]
    criterion = [row for row in rows if row.n06_candidate is not None]
    return {
        "records": len(rows), "records_with_hechos": sum(row.hechos is not None for row in rows),
        "records_with_criterio_juridico": len(criterion_sections), "records_with_justificacion": sum(row.justificacion is not None for row in rows),
        "citation_candidates": len(citations), "citation_candidates_with_nonempty_targets": sum(bool(c.target_ids) for c in citations),
        "citation_gold": sum(c.is_gold for c in citations), "criterion_sections": len(criterion_sections),
        "criterion_candidates": len(criterion), "n06_pairs": len(criterion),
        "records_with_rubro": sum(row.rubro is not None for row in rows), "records_with_precedentes": sum(bool(row.precedentes) for row in rows),
        "records_with_materias": sum(bool(row.materias) for row in rows), "records_with_nota_publica": sum(bool(row.nota_publica) for row in rows),
        "fact_groups": len({row.fact_group_id for row in rows}), "procedure_groups": len({row.procedure_group_id for row in rows}),
        "common_groups": len({row.common_group_id for row in rows}),
        "limits": [
            "citation relations are unresolved; empty target_ids are not gold",
            "criterion targets are editorial associations from each ficha, not citation resolution",
            "API ficha is metadata/text and not an original judgment or legal-validity label",
            "notaPublica, rubro, precedentes and materias are side metadata and never contenders",
        ],
    }


def to_jsonable(record: D1Record) -> dict[str, Any]:
    """Serialize one record for a small checkpoint/artifact without losing spans."""
    return asdict(record)


__all__ = [
    "DEFAULT_ROOT", "SourceSpan", "Section", "CitationCandidate", "CriterionCandidate", "D1Record",
    "split_sections", "iter_records", "load_records", "iter_n06_pairs", "assign_splits", "funnel", "to_jsonable",
]
