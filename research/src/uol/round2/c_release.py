"""Workstream C: run the identical v11 release pipeline on confirmation-set source rows.

Steps are those of ``uol.facts_to_law.release.build_release`` (resolver v2, target-blind v3
masker, strata and flags, post-snapshot flag, component grouping by merge_groups plus shared
proceedings), with two fixed choices so the new rows match v11 exactly:

* generic targets are the frozen v11 ``target_types.jsonl`` set (breadth_v1 was fitted on v11);
* components are computed jointly over v11 eligible rows plus the new rows; a new row's component
  id follows the v11 recipe over the new members of its joint component. A new row whose joint
  component also contains v11 rows is an *overlap* row and is flagged (``overlap_v11``) with the
  v11 components and their roles. Overlap rows are scored exactly like the others (frozen memory =
  all dev rows, which then contains the twin component); the pre-registration decides drop/flag.

Outputs per set (``experiments/runs/round2_20261004/C/<set>/release``): queries.jsonl,
references.jsonl, rows_meta.jsonl (publication date, encoder-release flag, overlap, population
membership), grouping.json, manifest.json. ``extend`` appends sets to a copy of v11 for scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from uol.facts_to_law import references_v2, strata
from uol.facts_to_law import release as v11
from uol.facts_to_law.spec import (Positive, QueryRecord, ReferenceRecord, Unresolved, fold_of, sha256_file, sha256_text,
                                   write_json, write_jsonl)
from uol.retrieval_benchmarks.benchmark_repair.masking import CatalogueMasker

from .c_sources import RUN

V11_DIR = v11.ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified"
SPLIT = v11.ROOT / "research/state/embedding_push_20261004/SPLIT.json"
QWEN_RELEASE = "2025-06-05"   # Qwen3-Embedding release date (RESULTS.md contamination check)
EXT_DIR = RUN / "release_ext"


def _load(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class Pipeline:
    """Catalogue, masker and titles exactly as ``build_release`` builds them (v10 reconciled candidates)."""

    def __init__(self) -> None:
        cand = v11.CANDIDATES
        self.instruments = _load(cand / "instruments.jsonl")
        self.articles = _load(cand / "articles.jsonl")
        self.catalogue = references_v2.build_catalogue(self.instruments, self.articles)
        self.masker = CatalogueMasker(self.instruments)
        self.titles = {r["instrument_id"]: [t for t in (v11._fold(x) for x in [r["title"], *(r.get("safe_aliases") or [])]) if len(t) >= 6]
                       for r in self.instruments}

    def resolve(self, sources: list[dict]):
        eligible, res, masked, excl = [], {}, {}, Counter()
        for s in sources:
            r = references_v2.resolve_source(s, self.catalogue)
            if r.status is None:
                excl["no_resolved_positive"] += 1
                continue
            eligible.append(s)
            res[s["query_id"]] = r
            masked[s["query_id"]] = self.masker.mask(s["raw_hechos"])
        return eligible, res, masked, excl


def v11_state():
    queries = _load(V11_DIR / "queries.jsonl")
    by_q = {q["query_id"]: q for q in queries}
    sources = [s for s in _load(v11.SOURCES) if s["query_id"] in by_q]
    if len(sources) != len(queries):
        raise ValueError("v11 sources do not cover every v11 query")
    generic = {r["candidate_id"] for r in _load(V11_DIR / "target_types.jsonl") if r["target_type"] == "generic"}
    roles = json.loads(SPLIT.read_text(encoding="utf-8"))["assignment"]
    return queries, by_q, sources, generic, roles


def build_set(name: str, sources_path: Path, *, check_v11: bool = True) -> dict:
    out = RUN / name / "release"
    if out.exists():
        raise FileExistsError(f"{out} exists; set releases are immutable (remove deliberately to rebuild)")
    pipe = Pipeline()
    v11_queries, v11_by_q, v11_sources, generic, roles = v11_state()
    new_sources = _load(sources_path)
    clash = {s["query_id"] for s in new_sources} & set(v11_by_q)
    if clash:
        raise ValueError(f"{len(clash)} new rows are already v11 rows, e.g. {sorted(clash)[:3]}")
    checks = {}
    v11_masked = {q: v11_by_q[q]["text"] for q in v11_by_q}
    if check_v11:
        # determinism: masker reproduces v11 texts and v11-only grouping reproduces v11 components
        sample = v11_sources[::40]
        bad = [s["query_id"] for s in sample if pipe.masker.mask(s["raw_hechos"])["text"] != v11_masked[s["query_id"]]]
        comps_v11, _ = v11._components(v11_sources, v11_masked)
        diff = sum(comps_v11[q] != v11_by_q[q]["component_id"] for q in v11_by_q)
        if bad or diff:
            raise ValueError(f"v11 pipeline not reproduced: masked mismatches {bad[:3]}, component mismatches {diff}")
        checks = {"v11_masked_text_sample_identical": len(sample), "v11_components_reproduced": len(v11_by_q)}
    eligible, res, masked, excl = pipe.resolve(new_sources)
    joint_masked = {**v11_masked, **{q: m["text"] for q, m in masked.items()}}
    joint, grouping = v11._components(v11_sources + eligible, joint_masked)
    members = defaultdict(list)
    for q, c in joint.items():
        members[c].append(q)
    new_ids = {s["query_id"] for s in eligible}
    v11_bridged = 0
    comp_of, overlap = {}, {}
    for c, qs in members.items():
        olds = sorted({v11_by_q[q]["component_id"] for q in qs if q in v11_by_q})
        news = sorted(q for q in qs if q in new_ids)
        if len(olds) > 1:
            v11_bridged += 1
        if not news:
            continue
        cid = "ftl-component:" + hashlib.sha256("\n".join(sorted(news)).encode()).hexdigest()[:24]
        for q in news:
            comp_of[q] = cid
        if olds:
            role_set = sorted({("lockbox" if roles[q] == "lockbox" else "dev") for q in qs if q in v11_by_q})
            for q in news:
                overlap[q] = {"v11_components": olds, "v11_roles": role_set,
                              "v11_queries": sorted(q2 for q2 in qs if q2 in v11_by_q)[:20]}
    queries, references, meta = [], [], []
    for s in sorted(eligible, key=lambda s: s["query_id"]):
        qid = s["query_id"]
        r, m = res[qid], masked[qid]
        positives = sorted({p["candidate_id"] for p in r.positives})
        stratum, flags = strata.query_strata(
            s["raw_hechos"], m, version_cue=bool(r.version_flags),
            answer_named=strata.answer_named_in_facts(s["raw_hechos"], [{"candidate_id": c} for c in positives], pipe.titles))
        date = s.get("publication_date")
        flags["post_catalogue_snapshot"] = bool(date and date > v11.SNAPSHOT_DATE)
        materias = v11._materias(s.get("source_materias"))
        year = int(s["source_year"]) if str(s.get("source_year") or "").isdigit() else None
        comp = comp_of[qid]
        q = QueryRecord(query_id=qid, record_id=str(s["record_id"]), component_id=comp, fold=fold_of(comp, v11.FOLD_SALT, 5),
                        source_split=s["split"], text=m["text"], text_sha256=sha256_text(m["text"]), stratum=stratum,
                        resolution_status=r.status, n_unresolved=len(r.missing), flags=flags, materias=materias, year=year,
                        tipo=s.get("source_tipo_tesis"), instancia=s.get("source_instancia"))
        queries.append(q)
        row = ReferenceRecord(query_id=qid,
                              positives=[Positive(p["candidate_id"], p["rule_ids"], p["section"], p["span"], p["surface_sha256"]) for p in r.positives],
                              unresolved=[Unresolved(u["reason"], u["section"], u["span"], u["surface_sha256"]) for u in r.missing])
        references.append({"query_id": qid, "positives": [vars(p) for p in row.positives], "unresolved": [vars(u) for u in row.unresolved],
                           "out_of_scope": [{"reason": u["reason"], "section": u["section"], "span": u["span"]} for u in r.unresolved if u["disposition"] == references_v2.OUT_OF_SCOPE],
                           "ignorable": [{"reason": u["reason"], "section": u["section"], "span": u["span"]} for u in r.unresolved if u["disposition"] == references_v2.IGNORABLE],
                           "version_flags": r.version_flags})
        in_p = strata.in_primary_population(stratum, r.status, flags)
        has_spec = any(c not in generic for c in positives)
        s2_full = stratum == "S2_masked_citation" and r.status == "full" and not any(flags.get(f) for f in ("version_cue", "residual_statute_name", "quote_bearing"))
        meta.append({"query_id": qid, "record_id": q.record_id, "component_id": comp, "set": name,
                     "publication_date": date, "publication_kind": s.get("publication_kind"),
                     "pre_qwen3_embedding_release": (date < QWEN_RELEASE) if date else None,
                     "epoca": s.get("source_epoca"), "instancia": s.get("source_instancia"),
                     "overlap_v11": overlap.get(qid), "P": in_p, "P_spec": in_p and has_spec, "S2_spec": s2_full and has_spec})
    out.mkdir(parents=True)
    write_jsonl(out / "queries.jsonl", queries)
    write_jsonl(out / "references.jsonl", references)
    write_jsonl(out / "rows_meta.jsonl", meta)
    write_json(out / "grouping.json", {**grouping, "v11_components_bridged_by_new_rows": v11_bridged,
                                       "overlap_rows": len(overlap)})
    manifest = {"set": name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "sources": {"path": str(sources_path), "sha256": sha256_file(sources_path), "rows": len(new_sources)},
                "pipeline": "uol.facts_to_law.release (v11) steps; frozen v11 generic set; joint components with v11",
                "v11_release_manifest_sha256": sha256_file(V11_DIR / "manifest.json"), "checks": checks,
                "exclusions": dict(excl), "queries": len(queries),
                "files": {n: sha256_file(out / n) for n in ("queries.jsonl", "references.jsonl", "rows_meta.jsonl", "grouping.json")},
                "code": {"c_release.py": sha256_file(Path(__file__)), "release.py": sha256_file(Path(v11.__file__))}}
    write_json(out / "manifest.json", manifest)
    return manifest


def extend(sets: list[str], out: Path = EXT_DIR) -> dict:
    """Copy v11 and append the given sets' rows (release order: v11 rows first, then each set in order).

    ``SPLIT_ext.json`` keeps every v11 assignment and assigns new rows the role ``lockbox`` so the
    frozen code treats them like lockbox rows (scored with models trained on all dev rows); the
    set of each row is recorded under ``new_sets``.
    """
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    queries = _load(V11_DIR / "queries.jsonl")
    refs = _load(V11_DIR / "references.jsonl")
    folds = json.loads((V11_DIR / "folds.json").read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    new_sets = {}
    for name in sets:
        d = RUN / name / "release"
        for q in _load(d / "queries.jsonl"):
            queries.append(q)
            new_sets[q["query_id"]] = name
            folds["components"].setdefault(q["component_id"], q["fold"])
            split["assignment"][q["query_id"]] = "lockbox"
        refs += _load(d / "references.jsonl")
    with (out / "queries.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for q in queries:
            f.write(json.dumps(q, ensure_ascii=False, sort_keys=True) + "\n")
    with (out / "references.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    write_json(out / "folds.json", folds)
    shutil.copyfile(V11_DIR / "target_types.jsonl", out / "target_types.jsonl")
    manifest = json.loads((V11_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["release_id"] = "round2_C_release_ext"
    manifest["extended_from"] = {"path": str(V11_DIR), "manifest_sha256": sha256_file(V11_DIR / "manifest.json"), "sets": sets}
    manifest["files"] = {n: sha256_file(out / n) for n in ("queries.jsonl", "references.jsonl", "target_types.jsonl", "folds.json")}
    write_json(out / "manifest.json", manifest)
    split["new_sets"] = new_sets
    split["note"] = "round2 C extension: v11 assignment unchanged; new rows carry role 'lockbox' (models trained on all dev rows)"
    write_json(out / "SPLIT_ext.json", split)
    return {"rows": len(queries), "new_rows": len(new_sets), "sets": dict(Counter(new_sets.values()))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["build", "extend"])
    p.add_argument("--set", dest="name")
    p.add_argument("--sources", type=Path)
    p.add_argument("--sets", nargs="*", default=[])
    p.add_argument("--no-check", action="store_true")
    a = p.parse_args()
    if a.command == "build":
        m = build_set(a.name, a.sources, check_v11=not a.no_check)
        print(json.dumps({k: m[k] for k in ("set", "queries", "exclusions", "checks")}, indent=1))
    else:
        print(json.dumps(extend(a.sets), indent=1))


if __name__ == "__main__":
    main()
