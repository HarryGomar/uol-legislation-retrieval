"""Where the language model alone falls short, and how far the catalogue's texts have moved (v29, 3 Nov 2026).

Descriptive, development cases only, from stored lists: no model call, no refit, nothing scored on the test set.

* GPT-6.1 Sol alone (``gpt61sol_alone``): how many articles it names per case (its own names carry scores of at least
  999,000 in the stored list; the rest is the ranker's fill), the cases in which it names no cited article, and the share
  of cited articles it names by how often their law is cited in the development cases. Against it, the ranker
  (``ft_lean``) and the system (``gpt61sol_combo_fwd_on_ft_lean``) on the same cases.
* The catalogue: reform notes (``... DOF dd-mm-yyyy``) per year in the candidate texts, and the comparison of the
  catalogue's texts with the consolidated texts of September 2026 (``experiments/runs/round2_20261004/A/p14_reform.json``).

    py -3.12 -m uol.final.model_limits        writes research/programs/why_models_20261103/limits.json
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs"
P14 = ROOT / "experiments" / "runs" / "round2_20261004" / "A" / "p14_reform.json"
OUT = ROOT / "research" / "programs" / "why_models_20261103" / "limits.json"
ALONE, RANKER, SYSTEM = "gpt61sol_alone", "rt_lm_tesis", "gpt61sol_rk_system"
NAMED = 999_000          # the model's own names in the stored list
BAND = 100               # a law cited more than this many times in the development cases counts as often cited
NOTE = re.compile(r"DOF\s*(\d{2})[-/](\d{2})[-/](\d{4})")


def _tops(run: str) -> pd.DataFrame:
    return pd.read_parquet(RUNS / run / "DEV" / "top200.parquet")


def _first(t: pd.DataFrame, k: int) -> dict[str, set[int]]:
    return {q: set(g.loc[g["rank"] <= k, "candidate_idx"].astype(int)) for q, g in t.groupby("query_id")}


def model_alone() -> dict:
    from uol.embedding_push import common as ep
    d = ep.release()
    qi = d.query_index if isinstance(d.query_index, dict) else {q: i for i, q in enumerate(d.query_index)}
    law = {i: cid.split("#")[0] for i, cid in enumerate(d.candidate_ids)}
    cases = pd.read_parquet(RUNS / RANKER / "DEV" / "per_case.parquet")["query_id"].tolist()
    alone = _tops(ALONE)
    named = {q: set(g.loc[g["score"] >= NAMED, "candidate_idx"].astype(int)) for q, g in alone.groupby("query_id")}
    rank, system = _tops(RANKER), _tops(SYSTEM)
    r10, r100, s10 = _first(rank, 10), _first(rank, 100), _first(system, 10)
    gold = {q: {int(a) for a in d.specific_idx[qi[q]]} for q in cases}
    cites = collections.Counter(law[a] for q in cases for a in gold[q])
    n_named = [len(named.get(q, ())) for q in cases]
    blind = [q for q in cases if not gold[q] & named.get(q, set())]
    pairs = [(law[a], a in named.get(q, set())) for q in cases for a in gold[q]]
    often = [hit for l, hit in pairs if cites[l] > BAND]
    rest = [hit for l, hit in pairs if cites[l] <= BAND]
    return {"cases": len(cases), "named_median": float(np.median(n_named)), "named_max": int(max(n_named)),
            "cited_pairs": len(pairs), "pairs_named": float(np.mean([h for _, h in pairs])),
            "blind_cases": len(blind), "blind_share": len(blind) / len(cases),
            "blind_ranker_first100": float(np.mean([bool(gold[q] & r100[q]) for q in blind])),
            "blind_ranker_first10": float(np.mean([bool(gold[q] & r10[q]) for q in blind])),
            "blind_system_first10": float(np.mean([bool(gold[q] & s10[q]) for q in blind])),
            "band": BAND, "named_often_cited_laws": float(np.mean(often)), "named_other_laws": float(np.mean(rest)),
            "pairs_often_cited_laws": len(often), "pairs_other_laws": len(rest)}


def catalogue() -> dict:
    from uol.embedding_push import common as ep
    d = ep.release()
    per_year = collections.Counter()
    for t in d.candidate_texts:
        for y in {int(y) for _, _, y in NOTE.findall(t)}:
            per_year[y] += 1
    years = [y for y in range(2015, 2024)]          # whole years before the snapshot of 15 September 2024
    p = json.loads(P14.read_text(encoding="utf-8"))
    cls = p["targets"]["change_class"]
    n = p["targets"]["n"]
    q = {s: p["splits"][s]["queries"]["version_mismatch"] for s in ("dev", "lockbox")}
    return {"articles_with_note_per_year": {str(y): per_year[y] for y in sorted(per_year)},
            "articles_reformed_per_year_mean": float(np.mean([per_year[y] for y in years])), "years": [years[0], years[-1]],
            "cited_articles": n, "cited_unchanged_share": cls["unchanged"] / n,
            "cited_substantive_share": cls["substantive (new DOF note, similarity < .95)"] / n,
            "later_cases": q["lockbox"]["flagged"]["n_queries"] + q["lockbox"]["unflagged"]["n_queries"],
            "later_cases_other_wording": q["lockbox"]["flagged"]["n_queries"],
            "dev_cases_other_wording": q["dev"]["flagged"]["n_queries"],
            "dev_cases_p14": q["dev"]["flagged"]["n_queries"] + q["dev"]["unflagged"]["n_queries"]}


def main() -> int:
    res = {"model_alone": model_alone(), "catalogue": catalogue()}
    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
