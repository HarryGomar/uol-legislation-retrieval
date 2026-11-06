"""How complete are the labels, and do the comparisons depend on it? (examiner point, 2 Nov 2026)

A label is what one court cited. Where the courts published near-identical facts more than once, the cases fall in one
group of related cases (Chapter 4), and the citations of the other cases of the group are a second answer from the
courts for the same facts. On the development-period cases only (the test labels are not read):

* agreement: for each ordered pair of labelled cases in a group, the share of the related case's articles that this
  case also cites; the share of the group's union that each case cites; both averaged within a group, then across
  groups, as every measure of the thesis;
* union labels: the main systems re-scored on the development cases with each case's labels widened to every article
  cited in its group (an article cited for the same facts by a related case counts as found), compared in one family
  with the record's contrast arithmetic (``audit_d181_common.dev_family``: paired bootstrap over groups, margin .02).

Groups are never split across folds, so no ranker trained on a related case of the case it scores.

    py -3.12 scripts/label_completeness.py        writes experiments/runs/footing_20261101/dev/label_completeness.json
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uol.final import audit_d181_common as A, footing as F  # noqa: E402
from uol.embedding_push import common as ep  # noqa: E402

RUNS = ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs"
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "label_completeness.json"
SYSTEMS = {"system": "gpt61sol_combo_fwd", "alone": "gpt61sol_alone", "ranker": "S0+qwen3_8b"}
PAIRS = [("system", "alone"), ("system", "ranker"), ("alone", "ranker")]


def ndcg(ranked: list[int], gold: set[int], generic: set[int]) -> float:
    keep = list(dict.fromkeys(i for i in ranked if i not in generic))[:10]
    gains = sum(1 / np.log2(k + 2) for k, i in enumerate(keep) if i in gold)
    idcg = sum(1 / np.log2(k + 2) for k in range(min(len(gold), 10)))
    return gains / idcg if idcg else 0.0


def main() -> int:
    F.D.activate(F.STORE)
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = [int(r) for r in ep.dev_rows()]
    Y = {r: {int(x) for x in d.specific_idx[r]} for r in rows}
    groups: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if Y[r]:
            groups[d.queries[r].component_id].append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}

    pair_share, union_share, n_pairs = [], [], 0
    for v in multi.values():
        vals = [len(Y[a] & Y[b]) / len(Y[b]) for a, b in itertools.permutations(v, 2)]
        n_pairs += len(vals)
        pair_share.append(float(np.mean(vals)))
        U = set().union(*(Y[r] for r in v))
        union_share.append(float(np.mean([len(Y[r]) / len(U) for r in v])))
    same = sum(all(Y[r] == Y[v[0]] for r in v) for v in multi.values())
    agreement = {"labelled_dev_period_cases": sum(map(len, groups.values())), "groups": len(groups), "groups_with_related": len(multi),
                 "cases_in_them": sum(map(len, multi.values())), "ordered_pairs": n_pairs,
                 "pair_share_mean": float(np.mean(pair_share)), "union_share_mean": float(np.mean(union_share)),
                 "groups_all_same_labels": same}

    row_of = {d.queries[r].query_id: r for r in rows}
    frames: dict[str, dict[str, pd.DataFrame]] = {"own": {}, "union": {}}
    widened = None
    for key, sid in SYSTEMS.items():
        top = pd.read_parquet(RUNS / sid / "DEV" / "top200.parquet").sort_values(["query_id", "rank"])
        lists = top.groupby("query_id", sort=False)["candidate_idx"].apply(lambda s: [int(x) for x in s]).to_dict()
        own, uni = [], []
        for qid, ranked in lists.items():
            r = row_of[qid]
            U = set().union(*(Y[x] for x in groups.get(d.queries[r].component_id, [r]))) | Y[r]
            base = {"query_id": qid, "row": r, "component_id": d.queries[r].component_id}
            own.append({**base, "ndcg_at_10": ndcg(ranked, Y[r], generic)})
            uni.append({**base, "ndcg_at_10": ndcg(ranked, U, generic), "widened": len(U) > len(Y[r])})
        frames["own"][key], frames["union"][key] = pd.DataFrame(own), pd.DataFrame(uni)
        widened = int(frames["union"][key]["widened"].sum())
        stored = pd.read_parquet(RUNS / sid / "DEV" / "per_case.parquet").set_index("query_id")["ndcg_at_10"]
        diff = (frames["own"][key].set_index("query_id")["ndcg_at_10"] - stored).abs().max()
        assert diff < 1e-9, (sid, diff)                    # the own-label scores reproduce the store

    def group_mean(f: pd.DataFrame) -> float:
        return float(f.groupby("component_id")["ndcg_at_10"].mean().mean())

    out = {"step": "label_completeness", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "systems": SYSTEMS,
           "agreement": agreement, "eval_cases": len(frames["own"]["system"]), "eval_cases_widened": widened, "scores": {}, "families": {}}
    wide = set(frames["union"]["system"].loc[frames["union"]["system"]["widened"], "query_id"])
    for lab in ("own", "own_widened", "union_widened", "union"):
        src = frames[lab.split("_")[0]]
        sub = {k: (f[f["query_id"].isin(wide)] if lab.endswith("widened") else f) for k, f in src.items()}
        out["scores"][lab] = {k: group_mean(f) for k, f in sub.items()}
        out["families"][lab] = A.dev_family(f"labels_{lab}", [(a, sub[a], b, sub[b]) for a, b in PAIRS])
    out["groups_widened"] = int(frames["union"]["system"].loc[frames["union"]["system"]["widened"], "component_id"].nunique())
    OUT.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"agreement": agreement, "eval_cases_widened": widened, "scores": out["scores"],
                      "contrasts": {lab: {k: {x: v.get(x) for x in ("delta", "ci95", "adjusted_lower", "class")}
                                          for k, v in out["families"][lab].get("contrasts", out["families"][lab]).items() if isinstance(v, dict)}
                                    for lab in out["families"]}}, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
