"""Shared, model-free contracts for body-backed citation releases.

The resolver accepts exact documentary identities only. Text normalization is
used to compare already-declared identifiers and article labels. It is not a
similarity rule and never proposes a new identity.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MASK = "[CITA_OCULTA]"
SCHEMA_VERSION = "body-backed-citations-1.0"
SPLIT_SEED = "body-backed-citations-20260917-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def normalize_identity(value: Any) -> str:
    """Fold an explicit identity or alias without fuzzy matching."""
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.replace("�", "")
    return " ".join(re.sub(r"[^\w:/#.?=&%+-]+", " ", text).split())


_ARTICLE_PREFIX = re.compile(
    r"^(?:(?:los?\s+)?art(?:i|í)culos?|arts?\.?|numeral(?:es)?)\s*[:.\-]?\s*",
    re.IGNORECASE,
)
_ORDINAL_SUFFIX = re.compile(r"(?<=\d)\s*(?:o\.?|º|°)$", re.IGNORECASE)


def normalize_article_label(value: Any) -> str:
    """Normalize one witnessed label without expanding ranges or composites."""
    text = unicodedata.normalize("NFKC", str(value or "")).strip().casefold()
    text = _ARTICLE_PREFIX.sub("", text)
    text = text.replace("–", "-").replace("—", "-")
    text = _ORDINAL_SUFFIX.sub("", text)
    text = re.sub(r"\s*([,;:/-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .")
    return text


def normalize_body(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


@dataclass(frozen=True)
class Resolution:
    status: str
    candidate_ids: tuple[str, ...] = ()
    tier: str | None = None
    reason: str | None = None
    body_equivalence_id: str | None = None


class CandidateIndex:
    """Exact instrument-and-label index over the fixed provision universe."""

    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.rows = tuple(dict(row) for row in rows)
        self.by_id = {str(row["candidate_id"]): dict(row) for row in self.rows}
        if len(self.by_id) != len(self.rows):
            raise ValueError("duplicate candidate_id in provision universe")
        self._exact: dict[tuple[str, str], set[str]] = defaultdict(set)
        self._alias_tier: dict[tuple[str, str], str] = {}
        alias_map = aliases or {}
        for row in self.rows:
            candidate_id = str(row["candidate_id"])
            instrument_id = str(row.get("instrument_id") or "")
            title = str(row.get("instrument_title") or "")
            label = normalize_article_label(row.get("article_label"))
            if not instrument_id or not label:
                continue
            values = [(instrument_id, "exact_instrument_id"), (title, "exact_instrument_title")]
            values.extend((alias, "frozen_alias_map") for alias in alias_map.get(instrument_id, ()))
            for identity, tier in values:
                key = (normalize_identity(identity), label)
                if not key[0]:
                    continue
                self._exact[key].add(candidate_id)
                previous = self._alias_tier.get(key)
                if previous is None or previous == "frozen_alias_map":
                    self._alias_tier[key] = tier

    @classmethod
    def from_parquet(
        cls,
        path: Path | str,
        *,
        aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> "CandidateIndex":
        import pyarrow.parquet as pq

        return cls(pq.read_table(path).to_pylist(), aliases=aliases)

    def resolve(self, instrument: Any, article_label: Any) -> Resolution:
        identity = normalize_identity(instrument)
        label = normalize_article_label(article_label)
        if not identity:
            return Resolution("excluded", reason="missing_instrument_identity")
        if not label:
            return Resolution("excluded", reason="missing_article_label")
        ids = tuple(sorted(self._exact.get((identity, label), ())))
        if not ids:
            return Resolution("excluded", reason="zero_body_join")
        equivalence = {digest(normalize_body(self.by_id[candidate_id].get("text"))) for candidate_id in ids}
        if len(equivalence) != 1:
            return Resolution("excluded", reason="ambiguous_non_equivalent_body_join")
        return Resolution(
            "resolved",
            candidate_ids=ids,
            tier=self._alias_tier[(identity, label)],
            body_equivalence_id=next(iter(equivalence)),
        )

    def candidate_views(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidates: list[dict[str, Any]] = []
        clauses: list[dict[str, Any]] = []
        for row in sorted(self.rows, key=lambda item: str(item["candidate_id"])):
            cid = str(row["candidate_id"])
            body = str(row.get("text") or "")
            article_body = body.strip()
            instrument_body = "\n".join(
                value for value in (
                    str(row.get("instrument_title") or "").strip(),
                    f"Artículo {row.get('article_label')}" if row.get("article_label") else "",
                    article_body,
                ) if value
            )
            eq_id = digest(normalize_body(body))
            candidates.append({
                "candidate_id": cid,
                "instrument_id": row.get("instrument_id"),
                "article_label": row.get("article_label"),
                "article_body": article_body,
                "instrument_article_body": instrument_body,
                "body_equivalence_id": eq_id,
                "source_unit_id": row.get("source_unit_id"),
            })
            for ordinal, (start, end) in enumerate(clause_spans(body)):
                clauses.append({
                    "candidate_id": cid,
                    "clause_id": f"{cid}#clause={ordinal}",
                    "ordinal": ordinal,
                    "start": start,
                    "end": end,
                    "text": body[start:end],
                })
        return candidates, clauses


def clause_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic, source-preserving paragraph/clause spans."""
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for match in re.finditer(r"(?:\r?\n{2,}|(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ0-9]))", text):
        start, end = cursor, match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
        cursor = match.end()
    start, end = cursor, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if end > start:
        spans.append((start, end))
    return spans


def _literal_intervals(text: str, literal: str, reason: str) -> list[dict[str, Any]]:
    if not literal or literal == MASK:
        return []
    return [
        {"start": match.start(), "end": match.end(), "reason": reason, "surface": match.group(0)}
        for match in re.finditer(re.escape(literal), text, re.IGNORECASE)
    ]


def _fold_with_map(text: str) -> tuple[str, list[int], list[int]]:
    """Return identity-folded text plus original offsets for every character."""
    output: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    pending_space: tuple[int, int] | None = None
    for original_index, original in enumerate(text):
        expanded = unicodedata.normalize("NFKD", original.casefold())
        for char in expanded:
            if unicodedata.combining(char) or char == "�":
                continue
            value = char if re.match(r"[\w:/#.?=&%+-]", char) else " "
            if value.isspace():
                if output and output[-1] != " ":
                    pending_space = (original_index, original_index + 1)
                continue
            if pending_space is not None:
                output.append(" ")
                starts.append(pending_space[0])
                ends.append(pending_space[1])
                pending_space = None
            output.append(value)
            starts.append(original_index)
            ends.append(original_index + 1)
    if output and output[-1] == " ":
        output.pop(); starts.pop(); ends.pop()
    return "".join(output), starts, ends


def _normalized_intervals(
    text: str,
    literal: str,
    reason: str,
    *,
    folded_map: tuple[str, list[int], list[int]] | None = None,
) -> list[dict[str, Any]]:
    needle = normalize_identity(literal)
    if not needle:
        return []
    folded, starts, ends = folded_map or _fold_with_map(text)
    pattern = re.compile(r"(?<!\w)" + re.escape(needle).replace(r"\ ", r"\s+") + r"(?!\w)")
    result: list[dict[str, Any]] = []
    for match in pattern.finditer(folded):
        start = starts[match.start()]
        end = ends[match.end() - 1]
        result.append({"start": start, "end": end, "reason": reason, "surface": text[start:end]})
    return result


def _article_intervals(text: str, label: str) -> list[dict[str, Any]]:
    if not label:
        return []
    escaped = re.escape(str(label).strip()).replace(r"\ ", r"\s+")
    pattern = re.compile(rf"\bart(?:i|í)culos?\s+(?:n[uú]mero\s+)?{escaped}(?![\w-])", re.IGNORECASE)
    return [
        {"start": match.start(), "end": match.end(), "reason": "target_article_label", "surface": match.group(0)}
        for match in pattern.finditer(text)
    ]


def mask_query(
    text: str,
    *,
    aliases: Sequence[str] = (),
    article_labels: Sequence[str] = (),
    urls: Sequence[str] = (),
    canonical_ids: Sequence[str] = (),
    citation_spans: Sequence[Mapping[str, Any] | tuple[int, int]] = (),
) -> tuple[str, list[dict[str, Any]]]:
    """Mask target identity intervals and return an original-offset ledger."""
    if not text or text == MASK:
        return text, []
    events: list[dict[str, Any]] = []
    folded_map = _fold_with_map(text) if aliases or canonical_ids else None
    for span in citation_spans:
        if isinstance(span, Mapping):
            start, end = int(span.get("start", 0)), int(span.get("end", 0))
        else:
            start, end = int(span[0]), int(span[1])
        if 0 <= start < end <= len(text):
            events.append({"start": start, "end": end, "reason": "source_citation_span", "surface": text[start:end]})
    alias_events: list[dict[str, Any]] = []
    for alias in sorted(set(filter(None, aliases)), key=len, reverse=True):
        alias_events.extend(_normalized_intervals(text, alias, "target_alias", folded_map=folded_map))
    events.extend(alias_events)
    for url in sorted(set(filter(None, urls)), key=len, reverse=True):
        events.extend(_literal_intervals(text, url, "target_url"))
    for identifier in sorted(set(filter(None, canonical_ids)), key=len, reverse=True):
        events.extend(_normalized_intervals(text, identifier, "target_canonical_id", folded_map=folded_map))
    article_events = [
        event
        for label in sorted(set(filter(None, article_labels)), key=len, reverse=True)
        for event in _article_intervals(text, label)
    ]
    # A bare number is not an identity. Mask it only inside the witnessed
    # citation span or when the same citation phrase also contains a target
    # alias. Commas and sentence punctuation break that attachment so a
    # same-number citation to another law survives.
    anchors = [event for event in events if event["reason"] in {"source_citation_span", "target_alias", "target_url", "target_canonical_id"}]
    for article in article_events:
        attached = False
        for anchor in anchors:
            if article["start"] < anchor["end"] and anchor["start"] < article["end"]:
                attached = True
                break
            left = min(article["end"], anchor["end"])
            right = max(article["start"], anchor["start"])
            between = text[left:right]
            if len(between) <= 120 and not re.search(r"[;,.:\r\n]", between):
                attached = True
                break
        if attached:
            events.append(article)
    if not events:
        return text, []
    events.sort(key=lambda row: (row["start"], -row["end"], row["reason"]))
    merged: list[dict[str, Any]] = []
    for event in events:
        if merged and event["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], event["end"])
            merged[-1]["reasons"] = sorted(set(merged[-1]["reasons"] + [event["reason"]]))
            merged[-1]["surface"] = text[merged[-1]["start"]:merged[-1]["end"]]
        else:
            merged.append({**event, "reasons": [event["reason"]]})
    pieces: list[str] = []
    cursor = 0
    ledger: list[dict[str, Any]] = []
    for event in merged:
        pieces.append(text[cursor:event["start"]])
        transformed_start = sum(len(piece) for piece in pieces)
        pieces.append(MASK)
        ledger.append({
            "original_start": event["start"],
            "original_end": event["end"],
            "transformed_start": transformed_start,
            "transformed_end": transformed_start + len(MASK),
            "reasons": event["reasons"],
            "surface_sha256": hashlib.sha256(event["surface"].encode("utf-8")).hexdigest(),
        })
        cursor = event["end"]
    pieces.append(text[cursor:])
    masked = re.sub(rf"(?:\s*{re.escape(MASK)}\s*)+", f" {MASK} ", "".join(pieces)).strip()
    return masked, ledger


def leakage_reasons(
    text: str,
    *,
    aliases: Sequence[str] = (),
    article_labels: Sequence[str] = (),
    urls: Sequence[str] = (),
    canonical_ids: Sequence[str] = (),
) -> list[str]:
    failures: list[str] = []
    folded_map = _fold_with_map(text) if aliases or urls or canonical_ids else None
    for value in (*aliases, *urls, *canonical_ids):
        needle = normalize_identity(value)
        if needle and _normalized_intervals(text, str(value), "identity", folded_map=folded_map):
            failures.append(f"identity:{needle}")
    identity_values = [value for value in (*aliases, *urls, *canonical_ids) if normalize_identity(value)]
    alias_ranges = [
        event
        for value in identity_values
        for event in _normalized_intervals(text, str(value), "identity", folded_map=folded_map)
    ]
    for label in article_labels:
        for article in _article_intervals(text, str(label)):
            if any(
                len(text[min(article["end"], alias["end"]):max(article["start"], alias["start"])]) <= 120
                and not re.search(r"[;,.:\r\n]", text[min(article["end"], alias["end"]):max(article["start"], alias["start"])])
                for alias in alias_ranges
            ):
                failures.append(f"article:{normalize_article_label(label)}")
                break
    return sorted(set(failures))


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_holdout: bool = False,
) -> dict[str, str]:
    """Build source components; add target/body edges for the holdout view."""
    query_ids = [str(row["query_id"]) for row in rows]
    uf = _UnionFind(query_ids)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in rows:
        qid = str(row["query_id"])
        query_hash = str(row.get("normalized_query_sha256") or digest(normalize_body(row.get("text"))))
        for kind, value in (
            ("source", row.get("source_id") or row.get("source_document_id") or row.get("source_work_id")),
            ("query", query_hash),
            ("derivation", row.get("derivation_family")),
        ):
            if value:
                buckets[(kind, str(value))].append(qid)
        if target_holdout:
            for value in row.get("positive_body_equivalence_ids") or ():
                buckets[("body", str(value))].append(qid)
            for value in row.get("positive_candidate_ids") or ():
                buckets[("target", str(value))].append(qid)
    for values in buckets.values():
        for other in values[1:]:
            uf.union(values[0], other)
    roots: dict[str, list[str]] = defaultdict(list)
    for qid in query_ids:
        roots[uf.find(qid)].append(qid)
    output: dict[str, str] = {}
    for members in roots.values():
        component_id = "component:" + digest(sorted(members))[:24]
        output.update({member: component_id for member in members})
    return output


def assign_splits(
    components: Mapping[str, str],
    *,
    seed: str = SPLIT_SEED,
) -> dict[str, str]:
    assignment: dict[str, str] = {}
    for component_id in sorted(set(components.values())):
        value = int(hashlib.sha256(f"{seed}:{component_id}".encode()).hexdigest()[:16], 16) / 2**64
        assignment[component_id] = "train" if value < 0.75 else "development" if value < 0.85 else "test_sealed"
    return {query_id: assignment[component_id] for query_id, component_id in components.items()}


def split_violations(rows: Sequence[Mapping[str, Any]], component_field: str = "component_id") -> list[dict[str, Any]]:
    seen: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        seen[str(row[component_field])].add(str(row["split"]))
    return [
        {"component_id": component_id, "splits": sorted(splits)}
        for component_id, splits in sorted(seen.items()) if len(splits) > 1
    ]


def semantic_hashes(root: Path | str) -> dict[str, str]:
    """Hash JSON/JSONL semantically and all other files by bytes."""
    root = Path(root)
    output: dict[str, str] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if path.suffix == ".json":
            output[relative] = digest(json.loads(path.read_text(encoding="utf-8")))
        elif path.suffix == ".jsonl":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            output[relative] = digest(rows)
        else:
            output[relative] = sha256_file(path)
    return output
