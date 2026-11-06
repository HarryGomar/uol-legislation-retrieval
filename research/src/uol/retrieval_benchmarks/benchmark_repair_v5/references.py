"""V5 numeric-token adapter over the immutable v2/v3 documentary parser.

A grouped numeral such as ``artículo 1,245`` denotes one atomic article token.
It is accepted only as the sole article identity immediately following an
explicit singular article designator. Plural and elided grouped expressions
remain unresolved because a comma can instead separate an article list.

The adapter replaces three lexical seams in isolated function namespaces. It
does not monkeypatch imported modules or change source text. All law binding,
span mapping, title exclusions and other abstention rules retain their pinned
implementations. The differential command writes engineering evidence only;
it never creates a release or runs retrieval scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from types import FunctionType, SimpleNamespace

from uol.retrieval_benchmarks.benchmark_repair import references as v2
from uol.retrieval_benchmarks.benchmark_repair_v3 import references as v3


PARSER_VERSION = "c01-documentary-repair-5.0"
REQUIRED_V2_SHA256 = "9caac373905669b4e5db5bac17928df9a178f87c1628da97e23082b43408d160"
REQUIRED_V3_SHA256 = "399e16b965a9c90b8d7608c79028e5d2a2c0c1f1c802dbcf1408a0378bf020f6"
REQUIRED_SOURCE_SHA256 = "b2a49a54c1563f47f19ca452133caa52ba61a0fe1a6b98b8ac4059ea93360f8e"
REQUIRED_V4_MANIFEST_SHA256 = "1b7260a9506b12c9cce902d38acc65f7c27148f0f5bd2743f7bd565f637b1d75"
REQUIRED_V4_ROWS_SHA256 = "2d1c9a6ae4fa3d88f1e52ecf0fcbd7bcfafdd1da9b1c544f0fe93db199eb0f74"
REQUIRED_ARTICLES_SHA256 = "a2b2c496a9c8b59439e8c5ff9596db57f4fbecf172af4cac7a44b969aa9d61fd"
REQUIRED_INSTRUMENTS_SHA256 = "8064daed8562281ccabfa53871debe7a4818244ac26d5402ebb3276117001f20"
ROOT = Path(__file__).resolve().parents[4]
IMPORTED_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for _module, _expected in ((v2, REQUIRED_V2_SHA256), (v3, REQUIRED_V3_SHA256)):
    if file_sha(Path(_module.__file__)) != _expected:
        raise RuntimeError("V5 requires unchanged pinned predecessor parser bytes")

_OLD_BASE = r"(?P<base>\d{1,5})"
_NEW_BASE = r"(?P<base>(?:[1-9]\d{0,1},\d{3}|\d{1,5}))"
if v2.ARTICLE_NUMBER.pattern.count(_OLD_BASE) != 1:
    raise RuntimeError("Pinned numeric-token seam changed")
ARTICLE_NUMBER = re.compile(
    v2.ARTICLE_NUMBER.pattern.replace(_OLD_BASE, _NEW_BASE)
    .replace(r"(?![\w-])", r"(?![\w-]|,\d{3})")
)
SINGULAR_DESIGNATOR = re.compile(r"\b(?:articulo|art\.)\s+$")
GROUPED_NUMERAL = re.compile(r"(?<![\d,.])\d{1,3}(?:,\d{3})+(?![\d,])")


def canonical_article(label: str) -> str:
    """Canonicalize one complete token, including a validated grouping comma.

    This lexical helper does not establish source-context eligibility. That
    separate check occurs in parse_expression before any label is accepted.
    """
    value = v2.fold(label)[0].strip()
    match = ARTICLE_NUMBER.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported article identity: {label!r}")
    base = str(int(match.group("base").replace(",", "")))
    tokens = re.findall(r"[a-zñ]+|\d+", match.group("tail"))
    return "-".join([base, *[str(int(token)) if token.isdigit() else token for token in tokens]])


def _adapt(function, **overrides):
    """Give a pinned pure function a private namespace; leave imports intact."""
    namespace = dict(function.__globals__)
    namespace.update(overrides)
    adapted = FunctionType(function.__code__, namespace, function.__name__, function.__defaults__, function.__closure__)
    adapted.__kwdefaults__ = function.__kwdefaults__
    adapted.__module__ = __name__
    return adapted


_parse_tokens = _adapt(v2.parse_expression, ARTICLE_NUMBER=ARTICLE_NUMBER, canonical_article=canonical_article)


def parse_expression(text: str, start: int = 0) -> dict:
    """Parse folded text, retaining every original source-character position."""
    parsed = _parse_tokens(text, start)
    grouped = [entry for entry in parsed["entries"] if "," in text[entry["start"]:entry["end"]]]
    if grouped and (len(parsed["entries"]) != 1 or SINGULAR_DESIGNATOR.search(text[:start]) is None):
        parsed["reason"] = parsed["reason"] or "comma_group_ambiguous_with_article_list"
    return parsed


_extract_section = _adapt(v2.extract_section, ARTICLE_NUMBER=ARTICLE_NUMBER, parse_expression=parse_expression)
_extract_v2_source = _adapt(v2.extract_source, extract_section=_extract_section)
_extract_with_title_filter = _adapt(v3.extract_source, legacy=SimpleNamespace(extract_source=_extract_v2_source))


def extract_source(source: dict, instruments: list[dict], candidate_index: dict) -> dict:
    """Return the v3 row schema with atomic, context-bound grouped numerals."""
    result = _extract_with_title_filter(source, instruments, candidate_index)
    result["parser_version"] = PARSER_VERSION
    result["grouped_numeral_policy"] = "atomic full token; one identity after explicit singular article designator; ambiguous lists abstain"
    result["grouped_numeral_reference_occurrences"] = sum(
        "," in row["literal_article_span"]["text"] for row in result["references"]
    )
    raw = source["raw_published_text"]
    for row in result["references"]:
        for name in ("literal_article_span", "witness_span"):
            span = row[name]
            if raw[span["start"]:span["end"]] != span["text"] or v2.digest(span["text"]) != span["sha256"]:
                raise ValueError("V5 reference span no longer addresses unchanged source text")
    return result


def build_reference_rows(sources: list[dict], instruments: list[dict], articles: list[dict]) -> tuple[list[dict], dict]:
    """Keep the v3 integration API; candidate body changes do not alter IDs."""
    index = defaultdict(list)
    for article in articles:
        index[(article["instrument_id"], article["article_label"])].append(article)
    rows = [extract_source(source, instruments, index) for source in sources]
    reasons = Counter(reason for row in rows for reason in row["unresolved_reasons"])
    return rows, {
        "parser_version": PARSER_VERSION, "queries": len(rows),
        "strict_eligible_queries": sum(row["strict_eligible"] for row in rows),
        "queries_with_exact_references": sum(bool(row["positive_ids"]) for row in rows),
        "distinct_query_candidate_pairs": sum(len(row["positive_ids"]) for row in rows),
        "literal_reference_occurrences": sum(len(row["references"]) for row in rows),
        "unresolved_expression_count": sum(len(row["unresolved"]) for row in rows),
        "excluded_title_reference_occurrences": sum(len(row["excluded_title_references"]) for row in rows),
        "grouped_numeral_reference_occurrences": sum(row["grouped_numeral_reference_occurrences"] for row in rows),
        "queries_by_exclusion_reason": dict(reasons),
        "old_qrels_used_for_extraction": False, "protected_references_opened": 0,
        "model_calls": 0, "network_calls": 0,
    }


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def differential(output: Path) -> dict:
    """Parse all 2,339 safe sources against pinned v4 data, without a release."""
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("Preserve existing attempts; use a new differential directory")
    state = ROOT / "research/state/benchmark_repair_20260929_v4"
    source_path = ROOT / "research/state/benchmark_validation_20260926/safe_training_sources.jsonl"
    reference_path = state / "all_reference_dispositions.jsonl"
    articles_path, instruments_path = state / "candidates/articles.jsonl", state / "candidates/instruments.jsonl"
    manifest = ROOT / "data/releases/uol_c01_statutes_20260929_v4_development/manifest.json"
    expected = {
        source_path: REQUIRED_SOURCE_SHA256, reference_path: REQUIRED_V4_ROWS_SHA256,
        articles_path: REQUIRED_ARTICLES_SHA256, instruments_path: REQUIRED_INSTRUMENTS_SHA256,
        manifest: REQUIRED_V4_MANIFEST_SHA256,
        Path(v2.__file__): REQUIRED_V2_SHA256, Path(v3.__file__): REQUIRED_V3_SHA256,
        Path(__file__): IMPORTED_CODE_SHA256,
    }
    if any(file_sha(path) != digest for path, digest in expected.items()):
        raise ValueError("Pinned differential inputs/code changed")
    sources = _read(source_path)
    if len(sources) != 2339 or any(source["split"] != "train" for source in sources):
        raise ValueError("Differential requires all 2,339 original TRAIN safe sources")
    before = {row["query_id"]: row for row in _read(reference_path)}
    if set(before) != {source["query_id"] for source in sources}:
        raise ValueError("V4/source population mismatch")
    articles, instruments = _read(articles_path), _read(instruments_path)
    after_rows, summary = build_reference_rows(sources, instruments, articles)
    differences = []
    compare_keys = ("positive_ids", "references", "occurrences", "unresolved", "excluded_title_references",
                    "excluded_title_unresolved", "formal_statute_title_spans", "version_flags", "sections", "article_prefix_count")
    source_map = {source["query_id"]: source for source in sources}
    for after in after_rows:
        prior = before[after["query_id"]]
        prior_labels, next_labels = set(prior["positive_ids"]), set(after["positive_ids"])
        strict_after = bool(after["strict_eligible"] and not after["version_flags"] and not prior["source_review_quarantine"])
        changed = [key for key in compare_keys if prior[key] != after[key]]
        if after["strict_eligible"] != prior["parser_strict_eligible_before_version_policy"]:
            changed.append("parser_strict_eligible")
        source = source_map[after["query_id"]]
        grouped_spans = []
        for section in source["sections"]:
            if section["field"] not in v2.LATER_FIELDS:
                continue
            for match in GROUPED_NUMERAL.finditer(section["text"]):
                start, end = section["start"] + match.start(), section["start"] + match.end()
                grouped_spans.append({"field": section["field"], "start": start, "end": end,
                                      "text": source["raw_published_text"][start:end],
                                      "sha256": v2.digest(source["raw_published_text"][start:end])})
        differences.append({
            "query_id": after["query_id"], "record_id": source["record_id"],
            "source_content_sha256": source["source_content_sha256"],
            "changed_parser_fields": changed,
            "positive_ids_before": prior["positive_ids"], "positive_ids_after": after["positive_ids"],
            "labels_removed": sorted(prior_labels - next_labels), "labels_added": sorted(next_labels - prior_labels),
            "v4_strict_before": prior["strict_eligible"],
            "v5_parser_strict": after["strict_eligible"],
            "strict_after_unchanged_v4_version_and_source_policy": strict_after,
            "v4_unresolved_reasons": prior["unresolved_reasons"],
            "v5_parser_unresolved_reasons": after["unresolved_reasons"],
            "unchanged_v4_source_review_quarantine": prior["source_review_quarantine"],
            "raw_grouped_numeral_spans": grouped_spans,
            "references_before": prior["references"] if changed else None,
            "references_after": after["references"] if changed else None,
            "unresolved_before": prior["unresolved"] if changed else None,
            "unresolved_after": after["unresolved"] if changed else None,
        })
    if any(file_sha(path) != digest for path, digest in expected.items()):
        raise RuntimeError("Pinned code or inputs changed during differential")
    output.mkdir(parents=True)
    _write_rows(output / "reference_dispositions.jsonl", after_rows)
    _write_rows(output / "all_source_differential.jsonl", differences)
    _write_json(output / "reference_summary.json", summary)
    changed_rows = [row for row in differences if row["changed_parser_fields"]]
    result = {
        "utc": datetime.now(timezone.utc).isoformat(), "status": "completed_pre_release_reference_differential",
        "parser_version": PARSER_VERSION, "sources": len(sources), "candidate_articles": len(articles),
        "changed_source_records": len(changed_rows), "changed_query_ids": [row["query_id"] for row in changed_rows],
        "label_pairs_removed_all_sources": sum(len(row["labels_removed"]) for row in differences),
        "label_pairs_added_all_sources": sum(len(row["labels_added"]) for row in differences),
        "strict_queries_before": sum(row["v4_strict_before"] for row in differences),
        "strict_queries_after_unchanged_v4_policy": sum(row["strict_after_unchanged_v4_version_and_source_policy"] for row in differences),
        "strict_queries_added": [row["query_id"] for row in differences if not row["v4_strict_before"] and row["strict_after_unchanged_v4_version_and_source_policy"]],
        "strict_queries_removed": [row["query_id"] for row in differences if row["v4_strict_before"] and not row["strict_after_unchanged_v4_version_and_source_policy"]],
        "strict_label_pairs_added": sum(len(row["labels_added"]) for row in differences if row["v4_strict_before"]),
        "strict_label_pairs_removed": sum(len(row["labels_removed"]) for row in differences if row["v4_strict_before"]),
        "inputs_before_and_after_sha256": {path.relative_to(ROOT).as_posix(): digest for path, digest in expected.items()},
        "outputs_sha256": {name: file_sha(output / name) for name in ("reference_dispositions.jsonl", "all_source_differential.jsonl", "reference_summary.json")},
        "elapsed_seconds": time.perf_counter() - started,
        "release_built": False, "retrieval_scores_computed": False, "full_pytest_launched": False,
        "protected_references_opened": False, "v4_phase2_reviewer_ledgers_opened": False,
        "network_calls": 0, "model_calls": 0, "gpu_jobs": 0,
        "scope": "Engineering differential only; independent documentary correctness and final-test eligibility remain separate gates",
    }
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("differential")
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(differential(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
