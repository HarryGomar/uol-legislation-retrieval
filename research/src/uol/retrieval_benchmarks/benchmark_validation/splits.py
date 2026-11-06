"""Independent public-metadata split audit; never opens protected answers.

Sealed query rows contribute only allowlisted IDs and opaque hashes. Query
text is decoded only for original train/development rows. Near-duplicate
signals warrant grouping review and are not treated as proof of leakage.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import time
import unicodedata
from typing import Any, Mapping, Sequence

from uol.paths import project_root
from uol.retrieval_benchmarks.common import atomic_write_json, atomic_write_jsonl, sha256_file, utc_now

ROOT = project_root()
RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"
STATE = ROOT / "research/state/benchmark_validation_20260926/splits"
REPORT = ROOT / "research/reports/studies/benchmark_validation_20260926/SPLITS_AND_CUSTODY.md"
PROTOCOL = ROOT / "research/programs/benchmark_validation_20260926/protocol.json"
SPLIT_SEED = "c01-statutes-source-precedent-20260918-v1"
TUNE_SALT = "c01-refinement-suite-20260926-fitting-tune-v1"
METADATA = ("query_id", "record_id", "source_document_id", "component_id", "split", "proceeding_ids", "unmasked_text_sha256", "normalized_query_sha256")
DECODER = json.JSONDecoder()


def field(line: str, key: str) -> Any:
    matches = list(re.finditer(r'(?<!\\)"' + re.escape(key) + r'"\s*:\s*', line))
    if len(matches) != 1:
        raise ValueError(f"expected one allowlisted field: {key}")
    return DECODER.raw_decode(line, matches[0].end())[0]


def project_query(line: str) -> dict[str, Any]:
    result = {key: field(line, key) for key in METADATA}
    if result["split"] not in {"train", "development", "test_sealed"}:
        raise ValueError("unknown query split")
    if not all(isinstance(value, str) and value for key, value in result.items() if key != "proceeding_ids") or not isinstance(result["proceeding_ids"], list):
        raise ValueError("invalid query identity metadata")
    if not all(isinstance(value, str) and re.fullmatch(r"proceeding:[0-9a-f]{24}", value) for value in result["proceeding_ids"]):
        raise ValueError("invalid opaque proceeding identity")
    if result["split"] in {"train", "development"}:
        result["text"] = field(line, "text")
        if not isinstance(result["text"], str):
            raise ValueError("public query text must be a string")
    return result


def project_train_reference(line: str) -> dict[str, Any] | None:
    split = field(line, "split")
    if split == "development":
        return None
    if split != "train":
        raise ValueError("unexpected split in public train/development references")
    result = {key: field(line, key) for key in ("query_id", "positive_ids", "body_equivalence_ids")}
    if not isinstance(result["query_id"], str) or not result["query_id"]:
        raise ValueError("invalid public training reference identity")
    if any(not isinstance(result[key], list) or not result[key] or not all(isinstance(value, str) and value for value in result[key]) for key in ("positive_ids", "body_equivalence_ids")):
        raise ValueError("invalid public training reference values")
    result["split"] = "train"
    return result


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def derived_role(component: str, split: str) -> str:
    if split != "train":
        return split
    if int(hashlib.sha256(component.encode()).hexdigest()[:8], 16) % 5 == 0:
        return "internal_validation"
    value = int(hashlib.sha256(f"{TUNE_SALT}:{component}".encode()).hexdigest()[:8], 16)
    return "fit_tune" if value % 5 == 0 else "fit_train"


def expected_split(component: str) -> str:
    value = int(hashlib.sha256(f"{SPLIT_SEED}:{component}".encode()).hexdigest()[:16], 16) / 2**64
    return "train" if value < .75 else "development" if value < .85 else "test_sealed"


def overlap_buckets(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    buckets = defaultdict(list)
    for row in rows:
        value = row[key]
        for token in value if isinstance(value, list) else [value]:
            if token:
                buckets[str(token)].append(row)
    duplicated, split_crossings, component_crossings = [], [], []
    for value, members in sorted(buckets.items()):
        if len(members) < 2:
            continue
        item = {"value": value, "query_ids": sorted(row["query_id"] for row in members), "splits": sorted({row["split"] for row in members}), "components": sorted({row["component_id"] for row in members})}
        duplicated.append(item)
        if len(item["splits"]) > 1:
            split_crossings.append(item)
        if len(item["components"]) > 1:
            component_crossings.append(item)
    return {"unique_values": len(buckets), "duplicate_buckets": len(duplicated), "cross_split_buckets": len(split_crossings),
            "cross_component_buckets": len(component_crossings), "split_crossings": split_crossings, "component_crossings": component_crossings}


def shingles(text: str, size: int = 5) -> frozenset[tuple[str, ...]]:
    tokens = re.findall(r"\w+", normalize(text), flags=re.UNICODE)
    return frozenset(tuple(tokens[start:start + size]) for start in range(len(tokens) - size + 1))


def near_duplicate_pairs(rows: Sequence[Mapping[str, Any]], threshold: float = .8) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Complete Jaccard threshold search using deterministic prefix filtering."""
    if not 0 < threshold <= 1:
        raise ValueError("invalid Jaccard threshold")
    ordered = sorted(rows, key=lambda row: row["query_id"])
    if any(row["split"] not in {"train", "development"} for row in ordered):
        raise ValueError("near-duplicate search cannot consume protected query text")
    sets = [shingles(row["text"]) for row in ordered]
    frequencies = Counter(value for values in sets for value in values)
    token_order = {value: rank for rank, value in enumerate(sorted(frequencies, key=lambda value: (frequencies[value], value)))}
    postings = defaultdict(list)
    candidates = set()
    for index, values in enumerate(sets):
        if not values:
            continue
        prefix_length = len(values) - math.ceil(threshold * len(values)) + 1
        for value in sorted(values, key=token_order.__getitem__)[:prefix_length]:
            for previous in postings[value]:
                if min(len(values), len(sets[previous])) >= threshold * max(len(values), len(sets[previous])):
                    candidates.add((previous, index))
            postings[value].append(index)
    result = []
    for left, right in sorted(candidates):
        union = len(sets[left] | sets[right])
        similarity = len(sets[left] & sets[right]) / union
        if similarity + 1e-15 < threshold:
            continue
        first, second = ordered[left], ordered[right]
        roles = [derived_role(row["component_id"], row["split"]) for row in (first, second)]
        result.append({"query_ids": [first["query_id"], second["query_id"]], "splits": [first["split"], second["split"]],
                       "components": [first["component_id"], second["component_id"]], "derived_roles": roles,
                       "jaccard_word5": similarity, "shingle_counts": [len(sets[left]), len(sets[right])],
                       "cross_split": first["split"] != second["split"], "cross_component": first["component_id"] != second["component_id"],
                       "cross_derived_role": roles[0] != roles[1], "normalized_masked_text_exact": normalize(first["text"]) == normalize(second["text"])} )
    receipt = {"rows": len(ordered), "possible_pairs": len(ordered) * (len(ordered) - 1) // 2, "prefix_candidates_verified": len(candidates),
               "short_text_rows_without_five_word_shingles": [row["query_id"] for row, values in zip(ordered, sets, strict=True) if not values],
               "normalization": "NFKC, casefold, whitespace collapse, Unicode word tokens; set of five-token shingles",
               "search": "Complete threshold search by global rarity-ordered prefix filtering plus exact Jaccard; no LSH sampling", "threshold": threshold}
    return result, receipt


def grouping_implications(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    adjacency = defaultdict(set)
    for pair in pairs:
        left, right = pair["components"]
        if left != right:
            adjacency[left].add(right)
            adjacency[right].add(left)
    groups, seen = [], set()
    for node in sorted(adjacency):
        if node in seen:
            continue
        pending, group = [node], []
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            pending.extend(sorted(adjacency[current] - seen))
        groups.append(sorted(group))
    return {"candidate_component_merges": groups, "components_affected": len(seen), "groups": len(groups),
            "interpretation": "Similarity flags possible common-source or template dependence. Review these groups before a successor split; no historical split is altered and similarity alone is not proof of leakage."}


def custody_path_audit() -> dict[str, Any]:
    from uol.retrieval_benchmarks.c01_statutes.refinement.safeio import _resolve_inside_release, SealedReferenceAccess
    paths = ("custody/test/article_references.jsonl", "custody/test/instrument_references.jsonl", "review/sealed_error_analysis_sample.jsonl", "review/resolution_sample_400.jsonl", "review/resolution_sample_400.html")
    results = []
    for path in paths:
        try:
            _resolve_inside_release(path)
            denied = False
        except SealedReferenceAccess:
            denied = True
        results.append({"path": path, "denied_before_open": denied, "payload_opened": False})
    return {"status": "fail" if any(not row["denied_before_open"] for row in results) else "pass", "checks": results,
            "payload_files_opened": 0, "actual_historical_access": "unknown",
            "code_observation": "C01 builder exports an unfiltered all-record occurrence audit and draws resolution_sample_400 from that pool; body-backed release code exports positive IDs in all-split audit/query receipts. Their actual historical use is not inferred from path presence."}


def run() -> dict[str, Any]:
    started = time.monotonic()
    if (STATE / "result.json").exists() or (STATE / "audit_lock.json").exists():
        raise FileExistsError("split audit already prepared or completed; preserve its evidence")
    paths = {"queries": RELEASE / "method/queries.jsonl", "train_references": RELEASE / "method/article_references_train_development.jsonl", "articles": RELEASE / "candidates/articles.jsonl", "manifest": RELEASE / "manifest.json"}
    input_hashes = {key: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for key, path in paths.items()}
    source_paths = [Path(__file__), ROOT / "src/uol/retrieval_benchmarks/common.py",
                    ROOT / "tests/test_benchmark_validation_splits.py",
                    ROOT / "src/uol/retrieval_benchmarks/c01_statutes/build.py",
                    ROOT / "src/uol/retrieval_benchmarks/c01_statutes/precedents.py",
                    ROOT / "src/uol/retrieval_benchmarks/c01_statutes/refinement/safeio.py",
                    ROOT / "src/uol/retrieval_benchmarks/c01_statutes/refinement/contracts.py",
                    ROOT / "src/uol/retrieval_benchmarks/body_backed_citations/release.py",
                    ROOT / "src/uol/retrieval_benchmarks/body_backed_citations/validate.py",
                    ROOT / "research/reports/studies/representation_suite_20260926/VALIDITY_ASSESSMENT.md",
                    PROTOCOL]
    source_hashes = {path.relative_to(ROOT).as_posix(): sha256_file(path) for path in source_paths}
    atomic_write_json(STATE / "audit_lock.json", {"created_utc": utc_now(), "input_hashes": input_hashes,
                     "source_hashes": source_hashes, "thresholds": [.8, .9, .95], "word_shingle_size": 5,
                     "near_text_scope": ["original_train", "original_development"], "protected_payload_reads": 0})
    rows = [project_query(line) for line in paths["queries"].read_text(encoding="utf-8").splitlines() if line]
    references, skipped_refs = {}, 0
    for line in paths["train_references"].read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = project_train_reference(line)
        if row is None:
            skipped_refs += 1
        elif row["query_id"] in references:
            raise ValueError("duplicate public training reference ID")
        else:
            references[row["query_id"]] = row
    articles = []
    for line in paths["articles"].read_text(encoding="utf-8").splitlines():
        if line:
            articles.append({key: field(line, key) for key in ("candidate_id", "instrument_id", "body_equivalence_id")})
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    by_query = {row["query_id"]: row for row in rows}
    duplicate_ids = len(rows) - len(by_query)
    splits = {}
    for split in ("train", "development", "test_sealed"):
        selected = [row for row in rows if row["split"] == split]
        splits[split] = {"queries": len(selected), "components": len({row["component_id"] for row in selected}), "proceedings": len({value for row in selected for value in row["proceeding_ids"]}),
                         "without_proceeding_ids": sum(not row["proceeding_ids"] for row in selected), "text_decoded": split != "test_sealed"}
    overlaps = {key: overlap_buckets(rows, key) for key in ("component_id", "record_id", "source_document_id", "proceeding_ids", "unmasked_text_sha256", "normalized_query_sha256")}
    masked_rows = [dict(row, masked_hash=hashlib.sha256(normalize(row["text"]).encode()).hexdigest()) for row in rows if "text" in row]
    overlaps["normalized_masked_text"] = overlap_buckets(masked_rows, "masked_hash")
    assignment_mismatches = [row["query_id"] for row in rows if expected_split(row["component_id"]) != row["split"]]
    pairs, search = near_duplicate_pairs(masked_rows)
    summaries = {str(threshold): {"pairs": sum(pair["jaccard_word5"] >= threshold for pair in pairs), "cross_component_pairs": sum(pair["jaccard_word5"] >= threshold and pair["cross_component"] for pair in pairs), "cross_original_split_pairs": sum(pair["jaccard_word5"] >= threshold and pair["cross_split"] for pair in pairs), "cross_derived_role_pairs": sum(pair["jaccard_word5"] >= threshold and pair["cross_derived_role"] for pair in pairs), "within_original_train_cross_derived_role_pairs": sum(pair["jaccard_word5"] >= threshold and pair["cross_derived_role"] and pair["splits"] == ["train", "train"] for pair in pairs)} for threshold in (.8, .9, .95)}
    candidate_ids = {row["candidate_id"] for row in articles}
    positive_ids = {value for row in references.values() for value in row["positive_ids"]}
    training_queries = {row["query_id"] for row in rows if row["split"] == "train"}
    training_components = {by_query[query]["component_id"] for query in references if query in by_query}
    coverage = {"candidates": len(articles), "unique_candidate_ids": len(candidate_ids), "candidate_instruments": len({row["instrument_id"] for row in articles}), "candidate_body_classes": len({row["body_equivalence_id"] for row in articles}), "public_train_reference_queries": len(references), "public_train_reference_components": len(training_components), "public_train_unique_positive_articles": len(positive_ids), "train_queries_without_article_reference": len(training_queries - set(references)), "unknown_candidate_references": sorted(positive_ids - candidate_ids), "nontraining_reference_ids": sorted(set(references) - training_queries), "frozen_release_article_components_all_splits": manifest["census"]["article_components"], "original_3000_component_gate": manifest["census"]["article_components"] >= 3000}
    custody = custody_path_audit()
    cohort = {"certified_untouched_eligible_article_queries": 0, "certified_untouched_eligible_article_components": 0,
              "sealed_query_metadata_candidates": splits["test_sealed"]["queries"], "sealed_component_metadata_candidates": splits["test_sealed"]["components"],
              "verdict": "unknown_not_established", "sealed_article_eligibility": "Individual positive-label eligibility not inspected; public queries include instrument-only rows",
              "reason": "Original train was used for fitting and repeated internal evaluation; original development was evaluated in archived studies. Sealed paths coexist with unfiltered audit/review exports and no complete access ledger or custodian disposition establishes non-exposure. Repartitioning exposed rows cannot certify an untouched cohort.",
              "next_action": "Before final estimation, a custodian must issue label-free eligible IDs with source/proceeding/text exclusions and historical-exposure disposition. Existing known exposed rows support only exploratory comparisons."}
    changed = [str(path.relative_to(ROOT)) for key, path in paths.items() if sha256_file(path) != input_hashes[key]["sha256"]]
    changed += [str(path.relative_to(ROOT)) for path in source_paths if sha256_file(path) != source_hashes[path.relative_to(ROOT).as_posix()]]
    if changed:
        raise RuntimeError(f"audit input or source changed during execution: {changed}")
    result = {"status": "completed", "completed_utc": utc_now(), "input_hashes": input_hashes, "source_hashes": source_hashes,
              "hashes_rechecked_after_processing": True, "implementation_sha256": sha256_file(Path(__file__)), "splits": splits,
              "duplicate_query_ids": duplicate_ids, "assignment_mismatches": assignment_mismatches, "overlaps": overlaps, "near_duplicate_search": search,
              "near_duplicate_summary": summaries, "grouping_implications": grouping_implications(pairs), "coverage": coverage, "custody": custody, "untouched_cohort": cohort,
              "gates": {"declared_split_identity": "pass" if not duplicate_ids and not assignment_mismatches and all(value["cross_split_buckets"] == 0 for key, value in overlaps.items() if key != "normalized_masked_text") else "fail",
                        "public_masked_exact_grouping": "review_required" if overlaps["normalized_masked_text"]["cross_component_buckets"] else "pass_within_tested_text_scope",
                        "public_near_duplicate_grouping": "review_required" if summaries["0.8"]["cross_component_pairs"] else "pass_within_tested_text_scope", "custody_implementation": custody["status"], "historical_protected_answer_access": "unknown", "untouched_final_cohort": "unknown_not_established", "original_coverage_gate": "pass" if coverage["original_3000_component_gate"] else "fail"},
              "access": {"sealed_reference_files_opened": 0, "sealed_query_text_decoded": 0, "mixed_audit_payloads_opened": 0, "source_raw_sections_opened": 0, "public_train_reference_rows_decoded": len(references), "development_reference_rows_skipped_before_value_decode": skipped_refs, "limit": "Public mixed-file bytes traversed for hashes and allowlisted field extraction; protected answers and sealed query text were not decoded."},
              "limits": ["Proceeding coverage is conditional on retained parser IDs; missing or differently normalized real dockets are not independently verified.", "Near duplicates use historical masked public queries. Unmasked exact identity uses opaque exported hashes; sealed or unmasked near-text comparisons are not performed.", "Similarity alone does not establish shared proceeding or leakage, and no frozen split is modified.", "Candidate count is not independent evaluation sample size. Manifest-wide coverage counts remain builder-reported, not independently re-read protected references."],
              "resources": {"wall_seconds": time.monotonic() - started, "model_calls": 0, "endpoint_calls": 0, "gpu_calls": 0, "source_acquisition": 0}}
    atomic_write_jsonl(STATE / "near_duplicate_pairs.jsonl", pairs)
    atomic_write_jsonl(STATE / "query_identity_metadata.jsonl", [{key: row[key] for key in METADATA} for row in rows])
    atomic_write_json(STATE / "result.json", result)
    write_report(result)
    return result


def write_report(result: Mapping[str, Any]) -> None:
    lines = ["# Split, coverage and custody audit", "", "The declared metadata splits can be tested independently. An actually untouched final article-retrieval cohort cannot be certified from the inspected records. This is not evidence that every sealed record was accessed; historical access remains unknown.", "",
             "| Original split | Query metadata rows | Components | Missing proceeding IDs | Query text decoded |", "|---|---:|---:|---:|---|"]
    for name, value in result["splits"].items():
        lines.append(f"| {name} | {value['queries']} | {value['components']} | {value['without_proceeding_ids']} | {value['text_decoded']} |")
    lines += ["", f"Duplicate query IDs: {result['duplicate_query_ids']}. Independent split-hash assignment mismatches: {len(result['assignment_mismatches'])}.", "",
              "| Identity field | Cross-split repeated buckets | Cross-component repeated buckets |", "|---|---:|---:|"]
    for name, value in result["overlaps"].items():
        lines.append(f"| {name} | {value['cross_split_buckets']} | {value['cross_component_buckets']} |")
    lines += ["", "The builder unions exact normalized facts and retained court-qualified proceeding identities. Its grouping does not include near-duplicate text. The following deterministic audit uses five-word shingle Jaccard on train/development masked query text only. Every prefix-filter candidate is checked exactly. Similarity is a grouping-review signal, not automatic proof of leakage.", "",
              "| Jaccard threshold | Pairs | Across components | Across original train/development | Across any derived role, including development | Within original train, across fitting/tuning/internal |", "|---|---:|---:|---:|---:|---:|"]
    for threshold, value in result["near_duplicate_summary"].items():
        lines.append(f"| {threshold} | {value['pairs']} | {value['cross_component_pairs']} | {value['cross_original_split_pairs']} | {value['cross_derived_role_pairs']} | {value['within_original_train_cross_derived_role_pairs']} |")
    grouping = result["grouping_implications"]
    lines += ["", f"The .80 signals connect {grouping['components_affected']} recorded components in {grouping['groups']} possible merge groups. A successor should review these groups and assign any accepted grouping before splitting. This audit does not alter historical assignments. Raw unmasked and sealed near-text comparisons are outside this inspection.", "",
              f"The retained candidate metadata contains {result['coverage']['candidates']:,} articles in {result['coverage']['candidate_instruments']} instruments. Public training references cover {result['coverage']['public_train_reference_queries']} queries in {result['coverage']['public_train_reference_components']} components and {result['coverage']['public_train_unique_positive_articles']} distinct articles. {result['coverage']['train_queries_without_article_reference']} public train queries lack an article reference. The frozen manifest reports {result['coverage']['frozen_release_article_components_all_splits']} article components across all splits, below the original 3,000-component gate. That gate fails for the broader planned coverage claim, while a bounded exploratory pilot remains useful.", "",
              "The historical custody implementation fails the alternate-artifact guard check. The legacy safe loader rejects both custody reference paths but permits the review sample paths. Code inspection shows that the resolution sample and its HTML browser were produced from an unfiltered occurrence pool. The all-record occurrence audit and body-backed query receipts likewise contain answer-bearing fields outside the protected qrel directory. This audit tested path resolution and inspected writer/validator code only; it opened none of those payloads. Actual historical access remains unknown.", "",
              "No certified untouched cohort is established. Train and development have documented development use. Sealed public metadata provides potential source identities, but it neither certifies article-label eligibility nor proves that alternate answer artifacts remained unaccessed. A custodian needs to issue a label-free eligibility/exclusion manifest and an exposure disposition before any prospective final estimation. A newly frozen comparison on reused rows would still be exploratory.", "",
              "| Gate | Verdict |", "|---|---|"]
    lines += [f"| {key} | {value} |" for key, value in result["gates"].items()]
    lines += ["", "Thesis usefulness: the existing benchmark can support an explicitly retrospective, exploratory study of observed citation recovery and a documented benchmark-audit contribution. These checks do not authorize a confirmatory superiority claim, legal applicability claim, or untouched-test claim.", "",
              f"[Pre-execution lock]({(STATE / 'audit_lock.json').as_posix()}) · [Measured receipt]({(STATE / 'result.json').as_posix()}) · [Near-duplicate identity pairs]({(STATE / 'near_duplicate_pairs.jsonl').as_posix()}) · [Metadata-only identity manifest]({(STATE / 'query_identity_metadata.jsonl').as_posix()})", ""]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "gates": result["gates"], "output": str(STATE)}))
