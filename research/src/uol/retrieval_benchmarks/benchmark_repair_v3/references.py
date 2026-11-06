"""Version 3 documentary references: exclude article numbers inside law titles.

The v2 parser correctly resolves many compound article expressions, but treats
"Ley Reglamentaria de los Artículos 103 y 107 de la Constitución ..." as two
independent Constitution citations. This adapter filters only article spans
within that formal statute-title construction. All other parsing remains bound
to the immutable v2 implementation, including its source and catalogue checks.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from uol.retrieval_benchmarks.benchmark_repair import references as legacy


PARSER_VERSION = "c01-documentary-repair-3.0"
REQUIRED_V2_SHA256 = "9caac373905669b4e5db5bac17928df9a178f87c1628da97e23082b43408d160"
if hashlib.sha256(Path(legacy.__file__).read_bytes()).hexdigest() != REQUIRED_V2_SHA256:
    raise RuntimeError("v3 requires the pinned v2 reference parser")

# This identifies a formal law name, not a generic mention of constitutional
# articles. A true independent article citation before or after the title lies
# outside its measured span and survives. The narrow pattern is deliberate:
# additional title classes need their own witnessed regression before adoption.
TITLE_LEAD = re.compile(r"\bley(?:\s+de\s+amparo\s*,)?\s+reglamentaria\s+de\s+los\s+articulos\s+")
CONSTITUTION_END = re.compile(
    r"\s+de\s+la\s+(?:constitucion\s+politica\s+de\s+los\s+estados\s+unidos\s+mexicanos|"
    r"constitucion\s+general\s+de\s+la\s+republica|constitucion\s+general|constitucion\s+federal)\b"
)


def _title_spans(source: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for section in source["sections"]:
        if section["field"] not in legacy.LATER_FIELDS:
            continue
        text = section["text"]
        normalized, offsets = legacy.fold(text)
        for lead in TITLE_LEAD.finditer(normalized):
            expression = legacy.parse_expression(normalized, lead.end())
            if not expression["entries"] or expression["reason"] is not None:
                continue
            ending = CONSTITUTION_END.match(normalized, expression["end"])
            if ending is None:
                continue
            left = offsets[lead.start()]
            right = offsets[ending.end() - 1] + 1
            article_prefix = normalized.rfind("articulos", lead.start(), lead.end())
            assert article_prefix >= lead.start()
            article_left = offsets[article_prefix]
            span = {"field": section["field"],
                    "start": int(section["start"]) + left,
                    "end": int(section["start"]) + right,
                    "article_start": int(section["start"]) + article_left,
                    "text": text[left:right],
                    "reason": "article_numbers_inside_formal_statute_title"}
            assert source["raw_published_text"][span["start"]:span["end"]] == span["text"]
            spans.append(span)
    return spans


def _within_title(span: dict, titles: list[dict]) -> bool:
    return any(t["start"] <= span["start"] and span["end"] <= t["end"] for t in titles)


def extract_source(source: dict, instruments: list[dict], candidate_index: dict) -> dict:
    """Return v2-compatible rows plus explicit title exclusions.

    Only a title-contained occurrence is removed. A citation to the same
    Constitution article elsewhere in the source remains an independent
    reference. The source and its offsets are never rewritten.
    """
    result = legacy.extract_source(source, instruments, candidate_index)
    titles = _title_spans(source)
    excluded_references = [row for row in result["references"]
                           if _within_title(row["literal_article_span"], titles)]
    excluded_occurrences = [row for row in result["occurrences"]
                            if _within_title(row["article_span"], titles)]
    excluded_unresolved = [row for row in result["unresolved"]
                           if _within_title(row["article_span"], titles)]
    result["references"] = [row for row in result["references"]
                            if not _within_title(row["literal_article_span"], titles)]
    result["occurrences"] = [row for row in result["occurrences"]
                             if not _within_title(row["article_span"], titles)]
    result["unresolved"] = [row for row in result["unresolved"]
                            if not _within_title(row["article_span"], titles)]
    result["positive_ids"] = sorted({row["candidate_id"] for row in result["references"]})
    result["article_prefix_count"] -= sum(row["declaration_type"] == "article_prefix"
                                          for row in excluded_occurrences)
    assert result["article_prefix_count"] >= 0
    reasons = Counter(row["reason"] for row in result["unresolved"])
    if not result["sections"]:
        reasons["missing_later_sections"] += 1
    if not result["positive_ids"]:
        reasons["no_exact_reference"] += 1
    result["unresolved_reasons"] = dict(reasons)
    result["strict_eligible"] = not reasons
    result["parser_version"] = PARSER_VERSION
    result["formal_statute_title_spans"] = titles
    result["excluded_title_references"] = [
        {"candidate_id": row["candidate_id"], "instrument_id": row["instrument_id"],
         "article_label": row["article_label"], "field": row["field"],
         "literal_article_span": row["literal_article_span"],
         "witness_span": row["witness_span"],
         "reason": "article_numbers_inside_formal_statute_title"}
        for row in excluded_references
    ]
    result["excluded_title_unresolved"] = excluded_unresolved
    result["title_filter_applied_after_pinned_v2_parse"] = True
    return result


def build_reference_rows(sources: list[dict], instruments: list[dict], articles: list[dict]) -> tuple[list[dict], dict]:
    """Produce the same row/summary interface as v2 for a new release build."""
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for article in articles:
        index[(article["instrument_id"], article["article_label"])].append(article)
    rows = [extract_source(source, instruments, index) for source in sources]
    reasons = Counter(reason for row in rows for reason in row["unresolved_reasons"])
    summary = {"parser_version": PARSER_VERSION, "queries": len(rows),
               "strict_eligible_queries": sum(row["strict_eligible"] for row in rows),
               "queries_with_exact_references": sum(bool(row["positive_ids"]) for row in rows),
               "distinct_query_candidate_pairs": sum(len(row["positive_ids"]) for row in rows),
               "literal_reference_occurrences": sum(len(row["references"]) for row in rows),
               "unresolved_expression_count": sum(len(row["unresolved"]) for row in rows),
               "excluded_title_reference_occurrences": sum(len(row["excluded_title_references"]) for row in rows),
               "queries_by_exclusion_reason": dict(reasons),
               "old_qrels_used_for_extraction": False,
               "protected_references_opened": 0, "model_calls": 0, "network_calls": 0}
    return rows, summary
