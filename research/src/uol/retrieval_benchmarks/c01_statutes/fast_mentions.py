"""Fast regex-only article mention extraction for well-structured SJF text."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import re
import time
from typing import Any, Iterable, Mapping

from uol.retrieval_benchmarks.body_backed_citations.common import normalize_article_label


_LABEL = r"\d+(?:[a-z]|o\.|º|°)?(?:\s+(?:bis|ter|qu[aá]ter))?"
_ARTICLE_BLOCK = re.compile(
    rf"(?i)\b(?:art(?:i|í)culos?|arts?\.?|numerales?)\s+"
    rf"(?P<labels>{_LABEL}(?:\s*(?:,|;|y|e|al|a|-)\s*{_LABEL})*)"
)
_LABEL_TOKEN = re.compile(_LABEL, re.IGNORECASE)
_RANGE = re.compile(rf"(?i)^\s*(?P<left>{_LABEL})\s*(?:al|a|-)\s*(?P<right>{_LABEL})\s*$")
_CLAUSE_BOUNDARY = re.compile(r"(?:\r?\n)+|(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ0-9])")


@dataclass(frozen=True)
class FastMention:
    start: int
    end: int
    surface: str
    article_labels: tuple[str, ...]
    instrument_id: str | None
    status: str


def is_strict_hkj(record: Any) -> bool:
    """Require non-empty H, K, J spans in source order without overlap."""
    h, k, j = record.hechos, record.criterio_juridico, record.justificacion
    return bool(
        h and k and j
        and h.text.strip() and k.text.strip() and j.text.strip()
        and h.span.start < h.span.end <= k.span.start < k.span.end <= j.span.start < j.span.end
    )


def parse_article_labels(value: str, *, maximum_range: int = 100) -> tuple[str, ...]:
    """Parse witnessed labels and expand only bounded plain-numeric ranges."""
    match = _RANGE.match(value)
    if match:
        left = normalize_article_label(match.group("left"))
        right = normalize_article_label(match.group("right"))
        if left.isdigit() and right.isdigit():
            begin, end = int(left), int(right)
            if begin <= end and end - begin <= maximum_range:
                return tuple(str(number) for number in range(begin, end + 1))
    output: list[str] = []
    for token in _LABEL_TOKEN.finditer(value):
        label = normalize_article_label(token.group(0))
        if label and label not in output:
            output.append(label)
    return tuple(output)


class FastArticleExtractor:
    """One compiled exact-title matcher plus one article grammar."""

    def __init__(self, instruments: Iterable[Mapping[str, Any]], aliases: Mapping[str, Iterable[str]] | None = None) -> None:
        identity: dict[str, set[str]] = {}
        variants: dict[str, str] = {}
        aliases = aliases or {}
        for row in instruments:
            instrument_id = str(row["instrument_id"])
            values = [str(row["title"]), *[str(value) for value in aliases.get(instrument_id, ())]]
            for value in values:
                key = value.casefold().strip()
                if len(key) < 8:
                    continue
                identity.setdefault(key, set()).add(instrument_id)
                variants[value] = key
        self.identity = {key: next(iter(values)) for key, values in identity.items() if len(values) == 1}
        accepted = [value for value, key in variants.items() if key in self.identity]
        self.title_pattern = re.compile(
            r"(?<!\w)(?:" + "|".join(re.escape(value) for value in sorted(accepted, key=len, reverse=True)) + r")(?!\w)",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> tuple[FastMention, ...]:
        output: list[FastMention] = []
        cursor = 0
        for boundary in _CLAUSE_BOUNDARY.finditer(text):
            output.extend(self._clause(text, cursor, boundary.start()))
            cursor = boundary.end()
        output.extend(self._clause(text, cursor, len(text)))
        return tuple(output)

    def _clause(self, text: str, start: int, end: int) -> list[FastMention]:
        clause = text[start:end]
        article_matches = list(_ARTICLE_BLOCK.finditer(clause))
        if not article_matches:
            return []
        instrument_ids = {
            self.identity[match.group(0).casefold().strip()]
            for match in self.title_pattern.finditer(clause)
            if match.group(0).casefold().strip() in self.identity
        }
        instrument_id = next(iter(instrument_ids)) if len(instrument_ids) == 1 else None
        status = "exact_clause_binding" if instrument_id else "ambiguous_clause_instrument" if instrument_ids else "unbound_no_exact_title"
        return [
            FastMention(
                start=start + match.start(),
                end=start + match.end(),
                surface=match.group(0),
                article_labels=parse_article_labels(match.group("labels")),
                instrument_id=instrument_id,
                status=status,
            )
            for match in article_matches
        ]


def benchmark_extractor(records: Iterable[Any], extractor: FastArticleExtractor) -> dict[str, Any]:
    records = list(records)
    started = time.perf_counter()
    mentions = 0
    bound = 0
    by_field = {"criterio_juridico": 0, "justificacion": 0}
    frozen_mentions = 0
    frozen_overlap = 0
    frozen_label_overlap = 0
    examples: list[dict[str, Any]] = []
    for record in records:
        for field, section in (("criterio_juridico", record.criterio_juridico), ("justificacion", record.justificacion)):
            extracted = extractor.extract(section.text)
            absolute = [
                (section.span.start + mention.start, section.span.start + mention.end, set(mention.article_labels))
                for mention in extracted
            ]
            frozen = [
                citation for citation in record.citations
                if section.span.start <= citation.span.start and citation.span.end <= section.span.end
            ]
            frozen_mentions += len(frozen)
            for citation in frozen:
                overlaps = [row for row in absolute if row[0] < citation.span.end and citation.span.start < row[1]]
                if overlaps:
                    frozen_overlap += 1
                    expected = {normalize_article_label(value) for value in citation.articles}
                    observed = set().union(*(row[2] for row in overlaps))
                    if expected.intersection(observed):
                        frozen_label_overlap += 1
            for mention in extracted:
                mentions += 1
                by_field[field] += 1
                bound += mention.instrument_id is not None
                if len(examples) < 100:
                    examples.append({"record_id": record.record_id, "field": field, **asdict(mention)})
    elapsed = time.perf_counter() - started
    return {
        "records": len(records),
        "fields_scanned": 2 * len(records),
        "article_mentions": mentions,
        "exact_clause_bound_mentions": bound,
        "unbound_or_ambiguous_mentions": mentions - bound,
        "frozen_citation_mentions": frozen_mentions,
        "frozen_span_overlap": frozen_overlap,
        "frozen_span_overlap_rate": frozen_overlap / frozen_mentions if frozen_mentions else None,
        "frozen_label_overlap": frozen_label_overlap,
        "frozen_label_overlap_rate": frozen_label_overlap / frozen_mentions if frozen_mentions else None,
        "mentions_by_field": by_field,
        "wall_seconds": elapsed,
        "records_per_second": len(records) / elapsed if elapsed else None,
        "examples": examples,
        "role": "fast proposal and audit path; exact joins still decide labels",
    }
