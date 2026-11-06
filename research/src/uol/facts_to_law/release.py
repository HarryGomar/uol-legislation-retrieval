"""Build the facts-to-law v2 release (C01 v11 stratified).

Inputs: the structured SJF source export (``uol.facts_to_law.sources``) and a
candidate release bound by path and SHA-256 (not copied). Steps: resolver v2
references with graded eligibility, the unchanged target-blind v3 masker,
declared strata and flags, dependence components (merge_groups plus shared
proceedings), salted component folds and analysis-only target typing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from uol.retrieval_benchmarks.benchmark_repair.grouping import merge_groups
from uol.retrieval_benchmarks.benchmark_repair.masking import VERSION as MASK_VERSION, CatalogueMasker

from . import references_v2, strata, targets
from .spec import (FORMAT_VERSION, PARTIAL, ROOT, Positive, QueryRecord, ReferenceRecord, Unresolved, fold_of,
                   sha256_file, sha256_text, write_json, write_jsonl)

RELEASE_ID = "uol_c01_statutes_20261004_v11_stratified"
SOURCES = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"
CANDIDATES = ROOT / "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates"
FOLD_SALT = "facts-to-law-v2-20261004"
SNAPSHOT_DATE = "2024-09-15"  # latest reform note in the candidate texts
CFPC136_NOTE = "CFPC.pdf#articulo=136 carries a non-standard body_equivalence_id in v9/v10 (benchmark_repair_v6/documentary.py)"


def _load(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _materias(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            return [str(v) for v in parsed] if isinstance(parsed, (list, tuple)) else [value]
        except (ValueError, SyntaxError):
            return [value]
    return []


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def _components(eligible: list[dict], masked: dict[str, str]) -> tuple[dict[str, str], dict]:
    """merge_groups (v1 components, exact and Jaccard facts) unioned with shared proceeding ids."""
    mapping, receipt = merge_groups([{"query_id": s["query_id"], "component_id": s["component_id"], "raw_hechos": s["raw_hechos"]} for s in eligible], masked)
    parent = {q: q for q in mapping}

    def find(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    def union(a, b):
        a, b = sorted((find(a), find(b)))
        parent[b] = a

    first: dict[str, str] = {}
    for query, component in sorted(mapping.items()):
        if component in first:
            union(query, first[component])
        else:
            first[component] = query
    proceeding_links = 0
    holder: dict[str, str] = {}
    for source in sorted(eligible, key=lambda s: s["query_id"]):
        for proceeding in source.get("proceeding_ids") or []:
            if proceeding in holder:
                if find(holder[proceeding]) != find(source["query_id"]):
                    proceeding_links += 1
                union(source["query_id"], holder[proceeding])
            else:
                holder[proceeding] = source["query_id"]
    members = defaultdict(list)
    for query in sorted(mapping):
        members[find(query)].append(query)
    final = {}
    for group in members.values():
        component = "ftl-component:" + hashlib.sha256("\n".join(group).encode()).hexdigest()[:24]
        for query in group:
            final[query] = component
    return final, {"merge_groups_components": receipt["repaired_components"], "proceeding_unions": proceeding_links,
                   "final_components": len(members), "merge_links": len(receipt["links"])}


def build_release(sources_path: Path = SOURCES, candidate_dir: Path = CANDIDATES, out_dir: Path | None = None, *,
                  fold_salt: str = FOLD_SALT, k: int = 5) -> dict:
    out = Path(out_dir or ROOT / "data/releases" / RELEASE_ID)
    if out.exists():
        raise FileExistsError(f"{out} exists; releases are immutable")
    instruments = _load(Path(candidate_dir) / "instruments.jsonl")
    articles = _load(Path(candidate_dir) / "articles.jsonl")
    catalogue = references_v2.build_catalogue(instruments, articles)
    masker = CatalogueMasker(instruments)
    titles = {r["instrument_id"]: [t for t in (_fold(x) for x in [r["title"], *(r.get("safe_aliases") or [])]) if len(t) >= 6] for r in instruments}
    sources = _load(sources_path)

    exclusions = Counter()
    eligible, resolutions, masked_out = [], {}, {}
    for source in sources:
        resolution = references_v2.resolve_source(source, catalogue)
        if resolution.status is None:
            exclusions["no_resolved_positive"] += 1
            continue
        eligible.append(source)
        resolutions[source["query_id"]] = resolution
        masked_out[source["query_id"]] = masker.mask(source["raw_hechos"])
    components, grouping = _components(eligible, {q: m["text"] for q, m in masked_out.items()})

    queries, references, typing_rows = [], [], []
    for source in sorted(eligible, key=lambda s: s["query_id"]):
        qid = source["query_id"]
        resolution, masked = resolutions[qid], masked_out[qid]
        positives = sorted({p["candidate_id"] for p in resolution.positives})
        stratum, flags = strata.query_strata(
            source["raw_hechos"], masked, version_cue=bool(resolution.version_flags),
            answer_named=strata.answer_named_in_facts(source["raw_hechos"], [{"candidate_id": c} for c in positives], titles))
        date = source.get("publication_date")
        flags["post_catalogue_snapshot"] = bool(date and date > SNAPSHOT_DATE)
        materias = _materias(source.get("source_materias"))
        year = int(source["source_year"]) if str(source.get("source_year") or "").isdigit() else None
        missing = resolution.missing
        queries.append(QueryRecord(
            query_id=qid, record_id=str(source["record_id"]), component_id=components[qid],
            fold=fold_of(components[qid], fold_salt, k), source_split=source["split"], text=masked["text"],
            text_sha256=sha256_text(masked["text"]), stratum=stratum, resolution_status=resolution.status,
            n_unresolved=len(missing), flags=flags, materias=materias, year=year,
            tipo=source.get("source_tipo_tesis"), instancia=source.get("source_instancia")))
        row = ReferenceRecord(
            query_id=qid,
            positives=[Positive(p["candidate_id"], p["rule_ids"], p["section"], p["span"], p["surface_sha256"]) for p in resolution.positives],
            unresolved=[Unresolved(u["reason"], u["section"], u["span"], u["surface_sha256"]) for u in missing])
        references.append({"query_id": row.query_id,
                           "positives": [vars(p) for p in row.positives],
                           "unresolved": [vars(u) for u in row.unresolved],
                           "out_of_scope": [{"reason": u["reason"], "section": u["section"], "span": u["span"]} for u in resolution.unresolved if u["disposition"] == references_v2.OUT_OF_SCOPE],
                           "ignorable": [{"reason": u["reason"], "section": u["section"], "span": u["span"]} for u in resolution.unresolved if u["disposition"] == references_v2.IGNORABLE],
                           "version_flags": resolution.version_flags})
        typing_rows.append({"component_id": components[qid], "materias": materias, "positive_ids": positives})
    target_rows = targets.type_targets(typing_rows)

    out.mkdir(parents=True)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "references.jsonl", references)
    write_jsonl(out / "target_types.jsonl", target_rows)
    write_json(out / "folds.json", {"salt": fold_salt, "k": k, "components": {c: fold_of(c, fold_salt, k) for c in sorted(set(components.values()))}})
    audit = out / "audit"
    write_json(audit / "resolver_summary.json", references_v2.summarize(resolutions.values()))
    write_json(audit / "grouping.json", grouping)

    generic = {r.candidate_id for r in target_rows if r.target_type == "generic"}
    primary = [q for q in queries if strata.in_primary_population(q.stratum, q.resolution_status, q.flags)]
    refs_by_query = {r["query_id"]: r for r in references}
    primary_spec = [q for q in primary if any(p["candidate_id"] not in generic for p in refs_by_query[q.query_id]["positives"])]
    count = lambda rows, key: dict(sorted(Counter(key(r) for r in rows).items()))
    manifest = {
        "release_id": out.name, "format_version": FORMAT_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pre-registered exploratory development population (exposed sources); v1 sealed excluded",
        "sources": {"path": str(sources_path), "sha256": sha256_file(sources_path)},
        "candidates": {"articles": {"path": str(Path(candidate_dir) / "articles.jsonl"), "sha256": sha256_file(Path(candidate_dir) / "articles.jsonl")},
                       "instruments": {"path": str(Path(candidate_dir) / "instruments.jsonl"), "sha256": sha256_file(Path(candidate_dir) / "instruments.jsonl")},
                       "count": len(articles), "known_quirks": [CFPC136_NOTE]},
        "policies": {"resolver": references_v2.RESOLVER_VERSION, "masker": MASK_VERSION, "strata": "S1 no mask span / S2 >=1 mask span",
                     "typing": target_rows[0].rule if target_rows else None, "typing_sensitivity": "law_v1 (CPEUM, LAmp)",
                     "fold_salt": fold_salt, "k": k, "snapshot_date": SNAPSHOT_DATE,
                     "primary_population": "S1 & full & !version_cue & !residual_statute_name & !quote_bearing"},
        "counts": {
            "sources": len(sources), "queries": len(queries), "components": len(set(components.values())),
            "exclusions": dict(exclusions),
            "stratum_x_status": count(queries, lambda q: f"{q.stratum}|{q.resolution_status}"),
            "stratum_x_status_x_split": count(queries, lambda q: f"{q.stratum}|{q.resolution_status}|{q.source_split}"),
            "post_catalogue_snapshot": count(queries, lambda q: q.flags["post_catalogue_snapshot"]),
            "flags_true": {f: sum(q.flags[f] for q in queries) for f in queries[0].flags} if queries else {},
            "folds_queries": count(queries, lambda q: q.fold),
            "primary_P": {"queries": len(primary), "components": len({q.component_id for q in primary})},
            "primary_P_spec": {"queries": len(primary_spec), "components": len({q.component_id for q in primary_spec})},
            "generic_targets": len(generic), "cited_targets": len(target_rows),
            "positive_pairs": sum(len({p["candidate_id"] for p in r["positives"]}) for r in references),
        },
        "code": {name: sha256_file(Path(__file__).with_name(name)) for name in ("release.py", "references_v2.py", "strata.py", "targets.py", "spec.py")},
    }
    manifest["files"] = {name: sha256_file(out / name) for name in ("queries.jsonl", "references.jsonl", "target_types.jsonl", "folds.json")}
    write_json(out / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=SOURCES)
    parser.add_argument("--candidates", type=Path, default=CANDIDATES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    manifest = build_release(args.sources, args.candidates, args.out)
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
