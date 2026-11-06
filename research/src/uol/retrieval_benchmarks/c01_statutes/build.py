"""Build and validate the linked C01 statute retrieval releases."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import random
import shutil
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from uol.retrieval_benchmarks.semantic_d1 import iter_records
from uol.retrieval_benchmarks.body_backed_citations.common import (
    CandidateIndex,
    digest,
    leakage_reasons,
    mask_query,
    normalize_body,
    normalize_identity,
    semantic_hashes,
    sha256_file,
)

from .config import (
    EXPECTED,
    INSTRUMENT_CATALOGUE,
    PROGRAM_ID,
    PROVISIONS,
    RELEASE,
    REPORT,
    RUNS,
    SCHEMA_VERSION,
    SJF_CONTENT,
    SJF_ROOT,
    SPLIT_SEED,
    STAGING,
    STATE,
)
from .graph import exact_direct_references
from .fast_mentions import FastArticleExtractor, benchmark_extractor, is_strict_hkj
from .legacy import audit_legacy
from .precedents import parse_precedents


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _load_source_metadata() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    malformed = 0
    mojibake = 0
    with SJF_CONTENT.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            published = row.get("published") or {}
            text = str(published.get("texto") or "")
            if "Hechos:" not in text:
                continue
            record_id = str(published.get("idTesis") or row.get("requested_id"))
            if "�" in text or "�" in str(published.get("precedentes") or ""):
                mojibake += 1
            if record_id in metadata:
                malformed += 1
            metadata[record_id] = {
                "record_id": record_id,
                "court": published.get("organoJuris") or published.get("instancia") or "",
                "instance": published.get("instancia") or "",
                "precedentes": published.get("precedentes") or "",
            }
    return metadata, {"facts_records": len(metadata), "duplicate_ids": malformed, "mojibake_records": mojibake}


def _component_map(source: Mapping[str, Mapping[str, Any]], facts: Mapping[str, str]) -> tuple[dict[str, str], dict[str, Any], list[dict[str, Any]]]:
    record_ids = sorted(source)
    uf = UnionFind(record_ids)
    fact_buckets: dict[str, list[str]] = defaultdict(list)
    proceeding_buckets: dict[str, list[str]] = defaultdict(list)
    proceeding_rows: list[dict[str, Any]] = []
    counts = Counter()
    for record_id in record_ids:
        fact_buckets[digest(normalize_body(facts.get(record_id, "")))].append(record_id)
        source_row = source[record_id]
        mentions = parse_precedents(source_row.get("precedentes"), source_row.get("court"))
        counts[len(mentions)] += 1
        for ordinal, mention in enumerate(mentions):
            proceeding_buckets[mention.qualified_id].append(record_id)
            proceeding_rows.append({
                "record_id": record_id,
                "ordinal": ordinal,
                **asdict(mention),
                "source": mention.source,
            })
    for bucket in (*fact_buckets.values(), *proceeding_buckets.values()):
        for other in bucket[1:]:
            uf.union(bucket[0], other)
    root_members: dict[str, list[str]] = defaultdict(list)
    for record_id in record_ids:
        root_members[uf.find(record_id)].append(record_id)
    component_map: dict[str, str] = {}
    for members in root_members.values():
        component_id = "component:" + digest(sorted(members))[:24]
        for member in members:
            component_map[member] = component_id
    sizes = Counter(component_map.values())
    mention_count = len(proceeding_rows)
    identities = len(proceeding_buckets)
    component_census = {
        "precedent_mentions": mention_count,
        "precedent_zero_records": counts[0],
        "precedent_one_records": counts[1],
        "precedent_multiple_records": sum(value for key, value in counts.items() if key > 1),
        "precedent_identities": identities,
        "source_precedent_components": len(sizes),
        "multi_record_components": sum(1 for size in sizes.values() if size > 1),
        "records_in_multi_record_components": sum(size for size in sizes.values() if size > 1),
        "component_size_distribution": dict(sorted(Counter(sizes.values()).items())),
    }
    return component_map, component_census, proceeding_rows


def _split(component_id: str) -> str:
    value = int(hashlib.sha256(f"{SPLIT_SEED}:{component_id}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < 0.75 else "development" if value < 0.85 else "test_sealed"


def _instrument_universe(provisions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, list[str]]]:
    grouped: dict[str, dict[str, Any]] = {}
    exact: dict[str, set[str]] = defaultdict(set)
    candidate_ids: dict[str, list[str]] = defaultdict(list)
    for row in provisions:
        instrument_id = str(row.get("instrument_id") or "")
        title = str(row.get("instrument_title") or "")
        if not instrument_id or not title:
            continue
        grouped.setdefault(instrument_id, {"instrument_id": instrument_id, "title": title})
        exact[normalize_identity(instrument_id)].add(instrument_id)
        exact[normalize_identity(title)].add(instrument_id)
        candidate_ids[instrument_id].append(str(row["candidate_id"]))
    rows = []
    for instrument_id in sorted(grouped):
        rows.append({
            **grouped[instrument_id],
            "observed_aliases": [grouped[instrument_id]["title"]],
            "article_candidate_ids": sorted(candidate_ids[instrument_id]),
        })
    unambiguous = {key: next(iter(values)) for key, values in exact.items() if len(values) == 1}
    ambiguous = {key: sorted(values) for key, values in exact.items() if len(values) > 1}
    return rows, unambiguous, ambiguous


def _field_name(record: Any, citation: Any) -> str | None:
    start, end = citation.span.start, citation.span.end
    for name, section in (("criterio_juridico", record.criterio_juridico), ("justificacion", record.justificacion)):
        if section is not None and section.span.start <= start and end <= section.span.end:
            return name
    return None


def _resolve_instrument(citation: Any, exact: Mapping[str, str]) -> tuple[str | None, str | None]:
    witnessed = [citation.instrument, citation.instrument_literal]
    resolved = {exact[normalize_identity(value)] for value in witnessed if normalize_identity(value) in exact}
    if len(resolved) == 1:
        return next(iter(resolved)), "exact_instrument_identity"
    if len(resolved) > 1:
        return None, "conflicting_exact_identities"
    return None, "zero_exact_instrument_join"


def _make_browser(path: Path, title: str, rows: list[Mapping[str, Any]]) -> None:
    body = [f"<h1>{html.escape(title)}</h1>", "<table><thead><tr>"]
    keys = list(rows[0]) if rows else []
    body.extend(f"<th>{html.escape(key)}</th>" for key in keys)
    body.append("</tr></thead><tbody>")
    for row in rows:
        body.append("<tr>")
        body.extend(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys)
        body.append("</tr>")
    body.append("</tbody></table>")
    css = "body{font-family:system-ui;margin:2rem}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.3rem;vertical-align:top}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>{''.join(body)}", encoding="utf-8")


def _gate(census: Mapping[str, Any]) -> dict[str, Any]:
    instrument_pass = census["instrument_components"] >= 3000 and census["resolved_instruments"] >= 50
    article_pass = (
        census["article_components"] >= 3000
        and census["resolved_articles"] >= 500
        and census["article_instruments"] >= 50
    )
    return {
        "C01-INSTRUMENT": {"passed": instrument_pass, "disposition": "confirmatory_eligible" if instrument_pass else "coverage_pilot"},
        "C01-ARTICLE": {"passed": article_pass, "disposition": "confirmatory_eligible" if article_pass else "coverage_pilot"},
        "all_passed": instrument_pass and article_pass,
    }


def _target_holdout_rows(queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Connect primary components that share an instrument, article, or body."""
    primary_ids = sorted({str(row["component_id"]) for row in queries})
    uf = UnionFind(primary_ids)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in queries:
        component_id = str(row["component_id"])
        for target in row["instrument_positive_ids"]:
            buckets[("instrument", str(target))].append(component_id)
        for target in row["article_positive_ids"]:
            buckets[("article", str(target))].append(component_id)
        for target in row["positive_body_equivalence_ids"]:
            buckets[("body", str(target))].append(component_id)
    for values in buckets.values():
        for other in values[1:]:
            uf.union(values[0], other)
    members: dict[str, list[str]] = defaultdict(list)
    for component_id in primary_ids:
        members[uf.find(component_id)].append(component_id)
    holdout_by_primary: dict[str, str] = {}
    for group in members.values():
        holdout_id = "target-component:" + digest(sorted(group))[:24]
        for component_id in group:
            holdout_by_primary[component_id] = holdout_id
    return [
        {
            "query_id": row["query_id"],
            "primary_component_id": row["component_id"],
            "target_holdout_component_id": holdout_by_primary[str(row["component_id"])],
            "target_holdout_split": _split(holdout_by_primary[str(row["component_id"])]),
        }
        for row in queries
    ]


def build_release() -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    for path in (STATE, STAGING, RELEASE, RUNS, REPORT):
        path.mkdir(parents=True, exist_ok=True)

    provisions = pq.read_table(PROVISIONS).to_pylist()
    if len(provisions) != EXPECTED["articles"]:
        raise ValueError(f"expected {EXPECTED['articles']} provisions, found {len(provisions)}")
    instruments, exact_instruments, ambiguous_titles = _instrument_universe(provisions)
    if len(instruments) != EXPECTED["instruments"]:
        raise ValueError(f"expected {EXPECTED['instruments']} instruments, found {len(instruments)}")
    legacy_audit = audit_legacy(instruments)
    accepted_aliases = legacy_audit["accepted_aliases_by_instrument"]
    candidate_index = CandidateIndex(provisions, aliases=accepted_aliases)
    for instrument_id, aliases in accepted_aliases.items():
        for alias in aliases:
            exact_instruments[normalize_identity(alias)] = instrument_id
    _write_json(STATE / "legacy_audit.json", legacy_audit)

    source_metadata, source_census = _load_source_metadata()
    facts_records = [record for record in iter_records(SJF_ROOT) if record.hechos is not None]
    records = [record for record in facts_records if is_strict_hkj(record)]
    structured_ids = {record.record_id for record in records}
    source_metadata = {record_id: row for record_id, row in source_metadata.items() if record_id in structured_ids}
    facts = {record.record_id: record.hechos.text for record in records if record.hechos is not None}
    component_map, precedent_census, proceeding_rows = _component_map(source_metadata, facts)
    _write_jsonl(STATE / "proceeding_mentions.jsonl", proceeding_rows)
    fast_audit = benchmark_extractor(records, FastArticleExtractor(instruments, accepted_aliases))
    fast_audit.update({
        "facts_records_seen": len(facts_records),
        "strict_hkj_records": len(records),
        "excluded_non_strict_hkj_records": len(facts_records) - len(records),
    })
    _write_json(STATE / "fast_regex_article_mentions.json", fast_audit)
    proceeding_ids_by_record: dict[str, list[str]] = defaultdict(list)
    for proceeding_row in proceeding_rows:
        proceeding_ids_by_record[str(proceeding_row["record_id"])].append(str(proceeding_row["qualified_id"]))

    instrument_title = {row["instrument_id"]: row["title"] for row in instruments}
    article_parent = {str(row["candidate_id"]): str(row["instrument_id"]) for row in provisions}
    article_rows_by_id = {str(row["candidate_id"]): row for row in provisions}
    queries: list[dict[str, Any]] = []
    occurrence_audit: list[dict[str, Any]] = []
    raw_counts = Counter()
    exclusion_counts = Counter()
    for record in records:
        instrument_positive: set[str] = set()
        article_positive: set[str] = set()
        body_equivalence: set[str] = set()
        aliases: set[str] = set()
        article_labels: set[str] = set()
        fields_seen: set[str] = set()
        for citation in record.citations:
            field = _field_name(record, citation)
            if field is None:
                continue
            raw_counts[f"{field}_mentions"] += 1
            fields_seen.add(field)
            instrument_id, instrument_reason = _resolve_instrument(citation, exact_instruments)
            resolved_article_ids: set[str] = set()
            resolution_reason = instrument_reason
            if instrument_id is not None:
                instrument_positive.add(instrument_id)
                aliases.update(value for value in (citation.instrument, citation.instrument_literal, instrument_title[instrument_id]) if value)
                for label in citation.articles:
                    resolution = candidate_index.resolve(instrument_id, label)
                    if resolution.status == "resolved":
                        resolved_article_ids.update(resolution.candidate_ids)
                        body_equivalence.add(str(resolution.body_equivalence_id))
                        article_labels.add(str(label))
                    else:
                        exclusion_counts[str(resolution.reason)] += 1
                article_positive.update(resolved_article_ids)
                if resolved_article_ids:
                    resolution_reason = "exact_instrument_article_body_join"
                elif citation.articles:
                    resolution_reason = "instrument_resolved_article_unresolved"
                else:
                    resolution_reason = "instrument_only"
            else:
                exclusion_counts[instrument_reason or "unresolved"] += 1
            occurrence_audit.append({
                "record_id": record.record_id,
                "citation_id": citation.citation_id,
                "field": field,
                "surface": citation.surface,
                "instrument": citation.instrument,
                "instrument_literal": citation.instrument_literal,
                "articles": list(citation.articles),
                "instrument_id": instrument_id,
                "article_candidate_ids": sorted(resolved_article_ids),
                "status": resolution_reason,
            })
        for field in fields_seen:
            raw_counts[f"{field}_records"] += 1
        if not instrument_positive:
            continue
        parent_ids = {article_parent[candidate_id] for candidate_id in article_positive}
        if not parent_ids.issubset(instrument_positive):
            raise AssertionError("article parent instrument invariant failed")
        masked, ledger = mask_query(
            record.hechos.text,
            aliases=sorted(aliases),
            article_labels=sorted(article_labels),
            canonical_ids=sorted(instrument_positive),
        )
        leak = leakage_reasons(
            masked,
            aliases=sorted(aliases),
            article_labels=sorted(article_labels),
            canonical_ids=sorted(instrument_positive),
        )
        component_id = component_map[record.record_id]
        queries.append({
            "query_id": f"{record.unit_id}:hechos",
            "record_id": record.record_id,
            "source_document_id": record.document_id,
            "text": masked,
            "unmasked_text_sha256": digest(record.hechos.text),
            "normalized_query_sha256": digest(normalize_body(record.hechos.text)),
            "mask_ledger": ledger,
            "leakage_reasons": leak,
            "component_id": component_id,
            "split": _split(component_id),
            "instrument_positive_ids": sorted(instrument_positive),
            "article_positive_ids": sorted(article_positive),
            "positive_body_equivalence_ids": sorted(body_equivalence),
            "label_fields": sorted(fields_seen),
            "proceeding_ids": sorted(set(proceeding_ids_by_record.get(record.record_id, ()))),
        })

    if any(row["leakage_reasons"] for row in queries):
        raise AssertionError("post-mask leakage audit failed")

    instrument_queries = queries
    article_queries = [row for row in queries if row["article_positive_ids"]]
    instrument_components = len({row["component_id"] for row in instrument_queries})
    article_components = len({row["component_id"] for row in article_queries})
    resolved_instruments = {value for row in instrument_queries for value in row["instrument_positive_ids"]}
    resolved_articles = {value for row in article_queries for value in row["article_positive_ids"]}
    article_instruments = {article_parent[value] for value in resolved_articles}
    invariant_failures = [
        row["query_id"] for row in article_queries
        if not {article_parent[value] for value in row["article_positive_ids"]}.issubset(set(row["instrument_positive_ids"]))
    ]
    census = {
        **source_census,
        **precedent_census,
        "strict_hkj_records": len(records),
        "excluded_non_strict_hkj_records": len(facts_records) - len(records),
        "criterio_records": raw_counts["criterio_juridico_records"],
        "justificacion_records": raw_counts["justificacion_records"],
        "criterio_mentions": raw_counts["criterio_juridico_mentions"],
        "justificacion_mentions": raw_counts["justificacion_mentions"],
        "label_records": sum(1 for record in records if any(_field_name(record, c) for c in record.citations)),
        "label_mentions": raw_counts["criterio_juridico_mentions"] + raw_counts["justificacion_mentions"],
        "instrument_queries": len(instrument_queries),
        "article_queries": len(article_queries),
        "instrument_only_queries": len(instrument_queries) - len(article_queries),
        "instrument_components": instrument_components,
        "article_components": article_components,
        "resolved_instruments": len(resolved_instruments),
        "resolved_articles": len(resolved_articles),
        "article_instruments": len(article_instruments),
        "candidate_instruments": len(instruments),
        "candidate_articles": len(provisions),
        "ambiguous_exact_instrument_titles": len(ambiguous_titles),
        "cross_task_invariant_failures": len(invariant_failures),
        "exclusion_reasons": dict(sorted(exclusion_counts.items())),
    }
    gates = _gate(census)

    method = RELEASE / "method"
    custody = RELEASE / "custody/test"
    candidates = RELEASE / "candidates"
    review = RELEASE / "review"
    graph = RELEASE / "graph"
    for path in (method, custody, candidates, review, graph):
        path.mkdir(parents=True, exist_ok=True)

    public_queries = [{key: value for key, value in row.items() if key not in {"instrument_positive_ids", "article_positive_ids", "positive_body_equivalence_ids"}} for row in queries]
    _write_jsonl(method / "queries.jsonl", public_queries)
    _write_jsonl(method / "target_holdout_assignments.jsonl", _target_holdout_rows(queries))
    _write_jsonl(method / "instrument_references_train_development.jsonl", (
        {"query_id": row["query_id"], "split": row["split"], "positive_ids": row["instrument_positive_ids"]}
        for row in instrument_queries if row["split"] != "test_sealed"
    ))
    _write_jsonl(method / "article_references_train_development.jsonl", (
        {"query_id": row["query_id"], "split": row["split"], "positive_ids": row["article_positive_ids"], "body_equivalence_ids": row["positive_body_equivalence_ids"]}
        for row in article_queries if row["split"] != "test_sealed"
    ))
    _write_jsonl(custody / "instrument_references.jsonl", (
        {"query_id": row["query_id"], "positive_ids": row["instrument_positive_ids"]}
        for row in instrument_queries if row["split"] == "test_sealed"
    ))
    _write_jsonl(custody / "article_references.jsonl", (
        {"query_id": row["query_id"], "positive_ids": row["article_positive_ids"], "body_equivalence_ids": row["positive_body_equivalence_ids"]}
        for row in article_queries if row["split"] == "test_sealed"
    ))
    _write_jsonl(candidates / "instruments.jsonl", instruments)
    candidate_views, clauses = candidate_index.candidate_views()
    _write_jsonl(candidates / "articles.jsonl", candidate_views)
    _write_jsonl(candidates / "clauses.jsonl", clauses)
    _write_jsonl(STATE / "occurrence_resolution_audit.jsonl", occurrence_audit)

    containment = [
        {"source_id": str(row["candidate_id"]), "target_id": str(row["instrument_id"]), "relation_type": "article_contained_in_instrument"}
        for row in sorted(provisions, key=lambda item: str(item["candidate_id"]))
    ]
    _write_jsonl(graph / "containment_edges.jsonl", containment)
    direct_instrument, direct_article = exact_direct_references(provisions, candidate_index)
    _write_jsonl(graph / "exact_direct_instrument_edges.jsonl", direct_instrument)
    _write_jsonl(graph / "exact_direct_article_edges.jsonl", direct_article)
    train_cocitation = []
    for row in queries:
        if row["split"] != "train":
            continue
        positives = row["article_positive_ids"]
        for index, left in enumerate(positives):
            for right in positives[index + 1:]:
                train_cocitation.append({"source_id": left, "target_id": right, "relation_type": "training_cocitation", "query_id": row["query_id"]})
    _write_jsonl(graph / "training_cocitation_edges.jsonl", train_cocitation)

    rng = random.Random(20260918)
    review_pool = occurrence_audit.copy()
    rng.shuffle(review_pool)
    review_rows = []
    for row in review_pool[:400]:
        review_rows.append({
            **row,
            "reviewer": None,
            "instrument_correct": None,
            "article_correct": None,
            "clause_binding_correct": None,
            "exclusion_correct": None,
            "rationale": None,
        })
    _write_jsonl(review / "resolution_sample_400.jsonl", review_rows)
    test_rows = [row for row in queries if row["split"] == "test_sealed"]
    rng.shuffle(test_rows)
    _write_jsonl(review / "sealed_error_analysis_sample.jsonl", (
        {"query_id": row["query_id"], "reviewer": None, "error_class": None, "rationale": None}
        for row in test_rows[:200]
    ))
    _make_browser(review / "resolution_sample_400.html", "C01 resolution audit sample", review_rows)

    discrepancy = {key: {"expected": EXPECTED.get(key), "observed": census.get(key)} for key in (
        "facts", "criterio_records", "justificacion_records", "label_records", "label_mentions",
        "precedent_mentions", "precedent_one_records", "precedent_multiple_records",
        "precedent_identities", "precedent_components", "precedent_multi_components", "precedent_multi_records",
    ) if EXPECTED.get(key) != census.get(key)}
    # Map names used by the prompt to the explicit census fields.
    discrepancy.update({
        "facts": {"expected": EXPECTED["facts"], "observed": census["facts_records"]},
        "precedent_components": {"expected": EXPECTED["precedent_components"], "observed": census["source_precedent_components"]},
        "precedent_multi_components": {"expected": EXPECTED["precedent_multi_components"], "observed": census["multi_record_components"]},
        "precedent_multi_records": {"expected": EXPECTED["precedent_multi_records"], "observed": census["records_in_multi_record_components"]},
    })
    discrepancy = {key: value for key, value in discrepancy.items() if value["expected"] != value["observed"]}
    _write_json(STATE / "raw_to_release_funnel.json", census)
    _write_json(STATE / "source_census_discrepancies.json", discrepancy)
    _write_json(STATE / "promotion_gates.json", gates)
    _write_json(STATE / "cross_task_invariant.json", {"failures": invariant_failures, "passed": not invariant_failures})

    manifest = {
        "release_id": RELEASE.name,
        "program_id": PROGRAM_ID,
        "schema_version": SCHEMA_VERSION,
        "created_utc": "2026-09-18T00:00:00Z",
        "status": "candidate_confirmatory_eligible" if gates["all_passed"] else "coverage_pilot",
        "inputs": {
            "sjf_content": {"path": str(SJF_CONTENT.relative_to(SJF_CONTENT.parents[4])).replace("\\", "/"), "sha256": sha256_file(SJF_CONTENT)},
            "provisions": {"path": str(PROVISIONS.relative_to(PROVISIONS.parents[4])).replace("\\", "/"), "sha256": sha256_file(PROVISIONS)},
            "instrument_catalogue": {"path": str(INSTRUMENT_CATALOGUE.relative_to(INSTRUMENT_CATALOGUE.parents[4])).replace("\\", "/"), "sha256": sha256_file(INSTRUMENT_CATALOGUE) if INSTRUMENT_CATALOGUE.exists() else None},
        },
        "census": census,
        "promotion_gates": gates,
        "source_census_discrepancies": discrepancy,
        "human_review_completed": 0,
        "sealed_labels_location": "custody/test",
    }
    _write_json(RELEASE / "manifest.json", manifest)
    scientific_hashes = semantic_hashes(RELEASE)
    scientific_hashes.pop("scientific_hashes.json", None)
    _write_json(RELEASE / "scientific_hashes.json", scientific_hashes)

    result = {
        "status": manifest["status"],
        "release": str(RELEASE),
        "census": census,
        "promotion_gates": gates,
        "source_census_discrepancies": discrepancy,
        "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
    }
    _write_json(REPORT / "build_result.json", result)
    return result


def validate_release() -> dict[str, Any]:
    manifest_path = RELEASE / "manifest.json"
    if not manifest_path.exists():
        return {"status": "invalid", "reason": "missing_manifest"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if manifest.get("census", {}).get("candidate_instruments") != EXPECTED["instruments"]:
        errors.append("instrument_candidate_coverage")
    if manifest.get("census", {}).get("candidate_articles") != EXPECTED["articles"]:
        errors.append("article_candidate_coverage")
    if manifest.get("census", {}).get("cross_task_invariant_failures"):
        errors.append("cross_task_invariant")
    queries = [json.loads(line) for line in (RELEASE / "method/queries.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("leakage_reasons") for row in queries):
        errors.append("query_leakage")
    seen: dict[str, set[str]] = defaultdict(set)
    for row in queries:
        seen[str(row["component_id"])].add(str(row["split"]))
    if any(len(splits) != 1 for splits in seen.values()):
        errors.append("component_split_leakage")
    current_hashes = semantic_hashes(RELEASE)
    current_hashes.pop("scientific_hashes.json", None)
    frozen_hashes = json.loads((RELEASE / "scientific_hashes.json").read_text(encoding="utf-8"))
    frozen_hashes.pop("scientific_hashes.json", None)
    if current_hashes != frozen_hashes:
        errors.append("scientific_hash_mismatch")
    result = {"status": "valid" if not errors else "invalid", "errors": errors, "manifest_status": manifest.get("status")}
    _write_json(REPORT / "validation.json", result)
    return result
