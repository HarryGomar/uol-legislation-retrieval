"""Counts of the whole benchmark, the 46 test cases published from 25 September to 23 October 2026 included (v25, 3 Nov 2026).

The v11 counts in thesis/evidence/numbers*.tex stop before the later batch. This script adds the batch read on 27 Oct 2026
step by step and writes thesis/evidence/numbers_benchmark_full.tex, one \\renewcommand per macro with its source. It reads
only release files and the SJF census; it scores nothing.

    py -3.12 scripts/benchmark_full_counts.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V11 = ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified"
EXT = ROOT / "experiments/runs/footing_20261101/new/release_ext"
NEW = ROOT / "experiments/runs/new_cases_20261026/NEW/release"
COLLECT = ROOT / "experiments/runs/new_cases_20261026/collect/collect_result_20261027T020549Z.json"
TEST = ROOT / "experiments/runs/footing_20261101/test/per_case_combo.parquet"
TEX = ROOT / "thesis/evidence/numbers_benchmark_full.tex"
N_CATALOGUE = 39178

# v11 values as printed (numbers.tex, numbers_ch4.tex, numbers_app.tex); the batch is added to them.
V11_COUNTS = {"nSJFRecords": 311965, "nStructured": 9129, "nReleased": 5980, "nReleasedTest": 1168, "nNoCatalogue": 3149,
              "nFactsClean": 3337, "nFactsCleanTest": 727, "nNamed": 2643, "nResolved": 2532, "nResolvedTest": 572,
              "nUnresolved": 805, "nClean": 2240, "nCleanTest": 511, "nFlagged": 292, "nGenericOnly": 670,
              "nEval": 1569, "nEvalTest": 352}
N_GROUPS_EVAL_DEV = 1092  # numbers_ch4.tex \nGroupsEvalDev, unchanged


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def pair_counts(release: Path) -> tuple[int, int, int]:
    """Cited pairs, distinct cited articles, and pairs whose every mention the explicit rule (R0) binds."""
    pairs: dict[tuple[str, str], list[list[str]]] = {}
    for r in jsonl(release / "references.jsonl"):
        for p in r["positives"]:
            pairs.setdefault((r["query_id"], p["candidate_id"]), []).append(p["rule_ids"])
    explicit = sum(all(ids == ["R0"] for ids in v) for v in pairs.values())
    return len(pairs), len({k[1] for k in pairs}), explicit


def main() -> None:
    census = json.loads(COLLECT.read_text(encoding="utf-8"))["result"]
    new_records = census["new_ids_vs_census"]
    assert census["count"] == V11_COUNTS["nSJFRecords"] + new_records

    q = jsonl(NEW / "queries.jsonl")
    meta = {m["query_id"]: m for m in jsonl(NEW / "rows_meta.jsonl")}
    released = len(q)
    clean_facts = [x for x in q if x["stratum"].startswith("S1")]
    resolved = [x for x in clean_facts if x["resolution_status"] == "full"]
    # post_catalogue_snapshot marks every later case and removes none; the other flags remove a case
    clean = [x for x in resolved if not any(v for k, v in x["flags"].items() if k != "post_catalogue_snapshot")]
    assert all(meta[x["query_id"]]["P"] for x in clean)
    evaluated = [x for x in clean if meta[x["query_id"]]["P_spec"]]

    batch = {"nSJFRecords": new_records, "nStructured": new_records, "nReleased": released, "nReleasedTest": released,
             "nNoCatalogue": new_records - released, "nFactsClean": len(clean_facts), "nFactsCleanTest": len(clean_facts),
             "nNamed": released - len(clean_facts), "nResolved": len(resolved), "nResolvedTest": len(resolved),
             "nUnresolved": len(clean_facts) - len(resolved), "nClean": len(clean), "nCleanTest": len(clean),
             "nFlagged": len(resolved) - len(clean), "nGenericOnly": len(clean) - len(evaluated),
             "nEval": len(evaluated), "nEvalTest": len(evaluated)}
    new = {k: V11_COUNTS[k] + batch[k] for k in V11_COUNTS}

    ext_q = jsonl(EXT / "queries.jsonl")
    assert len(ext_q) == new["nReleased"]
    assert pair_counts(V11) == (12893, 2734, 9489)  # the v11 macros, reproduced by the same code
    n_pairs, n_articles, n_explicit = pair_counts(EXT)
    component = {x["query_id"]: x["component_id"] for x in ext_q}
    test_ids = pd.read_parquet(TEST)["query_id"].unique()
    assert len(test_ids) == new["nEvalTest"]
    test_groups = len({component[i] for i in test_ids})

    source = {"nSJFRecords": "census 27 Oct 2026 (collect_result count)", "nGroups": "release_ext component_id",
              "nPairs": "release_ext references positives", "nCitedArticles": "release_ext references positives",
              "pctCitedArticles": "nCitedArticles / 39,178", "nPairsExplicit": "pairs whose every mention has rule_ids [R0]",
              "pctPairsExplicit": "nPairsExplicit / nPairs", "nGroupsEvalTest": "component_id of the scored test cases",
              "nGroupsEval": "nGroupsEvalDev + nGroupsEvalTest"}
    values = {k: f"{v:,}" for k, v in new.items()}
    values.update({"nGroups": f"{len({x['component_id'] for x in ext_q}):,}", "nPairs": f"{n_pairs:,}",
                   "nCitedArticles": f"{n_articles:,}", "pctCitedArticles": f"{100 * n_articles / N_CATALOGUE:.1f}\\%",
                   "nPairsExplicit": f"{n_explicit:,}", "pctPairsExplicit": f"{100 * n_explicit / n_pairs:.1f}\\%",
                   "nGroupsEvalTest": f"{test_groups:,}", "nGroupsEval": f"{N_GROUPS_EVAL_DEV + test_groups:,}"})
    lines = ["% The whole benchmark, the later test cases included (scripts/benchmark_full_counts.py; do not edit by hand).",
             "% Read after numbers_final.tex. Development counts are unchanged."]
    for k, v in values.items():
        note = source[k] if k in source else f"v11 {V11_COUNTS[k]:,} + later batch {batch[k]:,} (new_cases_20261026 release)"
        lines.append(f"\\renewcommand{{\\{k}}}{{{v}}}% {note}")
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
