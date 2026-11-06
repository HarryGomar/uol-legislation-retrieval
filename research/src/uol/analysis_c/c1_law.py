"""C1: law and article for every system (ANALYSIS_PLAN_C §2; DEV, descriptive).

Per case, from the saved per-case and top-200 files: law@1, law@10, success@1, nDCG@10 (stored), and the conditional
within-law@1 (success@1 among cases with law@1 = 1). success@1 = mean(law@1) x mean(within-law@1 | law@1) is reported
per system on pooled case means (where it is an identity, because a cited top article implies a cited law) and on
component-macro means (where it is approximate), with the stored unconditional within-law@1
(``success_at_1_in_law``) beside it.

The law task ranks the laws found in the system's top 200 by (a) the best article score (first appearance; the stored
law@k), (b) the sum of the law's top-3 article scores, (c) the count of the law's articles in the top 50, ties broken
by (a). Laws absent from the top 200 are unranked (reciprocal rank 0). Baselines rank all 264 laws by popularity
(DEV cases of the other four folds citing the law) and by 1/size (articles per law, so the smallest law first); ties
by law code.

Counterfactual re-orderings (no refit), on the saved top 200: the law oracle keeps only the cited laws' articles
(order kept); gate 1 / gate 3 put the articles of the system's own top-1 / top-3 laws (by (a)) first, each block in the
system's order. Cited articles beyond the top 200 stay unretrieved, so the oracle is a lower bound of the headroom.

Choices where the plan is silent (recorded in the output): means are component-macro (as every stored summary); the
ideal list of nDCG@10 is the strict one (all cited specific articles); (b) sums raw scores on each system's own scale.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

ARROWS = [("e5_small", "ft_e5_q"), ("qwen3_0p6b", "qwen3_4b"), ("qwen3_4b", "qwen3_8b"), ("qwen3_0p6b", "qwen3_0p6b_ft"),
          ("bge_m3_base", "bge_m3_legal"), ("bge_m3", "bge_m3_ft"), ("lstar_headings", "S0"), ("S0", "S_full_ext"),
          ("S_full_ext", "llm_rerank_S_full_ext")]
CASE_MEASURES = ("law_at_1", "law_at_10", "success_at_1", "ndcg_at_10", "within_law_at_1_cond", "success_at_1_in_law",
                 "law_a_mrr", "law_a_at_1", "law_a_at_3", "law_b_mrr", "law_b_at_1", "law_b_at_3",
                 "law_c_mrr", "law_c_at_1", "law_c_at_3", "oracle_ndcg_at_10", "oracle_success_at_1",
                 "gate1_ndcg_at_10", "gate1_success_at_1", "gate3_ndcg_at_10", "gate3_success_at_1")
TOP_COUNT = 50


def _first_rank(order: np.ndarray, cited: np.ndarray) -> int:
    """1-based rank of the first cited law in ``order`` (0 when none is ranked)."""
    hit = np.flatnonzero(np.isin(order, cited))
    return int(hit[0]) + 1 if hit.size else 0


def law_orders(idx: np.ndarray, score: np.ndarray, law: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three law rankings (a), (b), (c) of one case's list."""
    ok = idx >= 0
    ids, sc = idx[ok], score[ok]
    if not ids.size:
        e = np.zeros(0, dtype=np.int64)
        return e, e, e
    L = law[ids].astype(np.int64)
    uniq, first = np.unique(L, return_index=True)
    order_a = uniq[np.argsort(first, kind="stable")]
    o = np.argsort(L, kind="stable")
    Ls = L[o]
    start = np.r_[0, np.flatnonzero(np.diff(Ls)) + 1]
    grp = np.repeat(start, np.diff(np.r_[start, Ls.size]))
    occ = np.empty_like(o)
    occ[o] = np.arange(o.size) - grp
    m = occ < 3
    s3 = np.bincount(L[m], weights=np.nan_to_num(sc[m]), minlength=int(L.max()) + 1)[uniq]
    order_b = uniq[np.lexsort((first, -s3))]
    cnt = np.bincount(L[:TOP_COUNT], minlength=int(L.max()) + 1)[uniq]
    order_c = uniq[np.lexsort((first, -cnt))]
    return order_a, order_b, order_c


def _rank_measures(rank: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    r = np.asarray(rank, dtype=np.float64)
    with np.errstate(divide="ignore"):
        mrr = np.where(r > 0, 1.0 / np.where(r > 0, r, 1), 0.0)
    return {f"{prefix}_mrr": mrr, f"{prefix}_at_1": ((r >= 1) & (r <= 1)).astype(float),
            f"{prefix}_at_3": ((r >= 1) & (r <= 3)).astype(float)}


def _reorder(hits: np.ndarray, first: np.ndarray, keep_rest: bool = True) -> np.ndarray:
    """Stable re-order: rows of ``first`` (bool) come first; the rest follow (or are dropped)."""
    order = np.argsort(~first, axis=1, kind="stable")
    h = np.take_along_axis(hits & (first | keep_rest), order, axis=1)
    return h


def system_cases(store: C.RunStore, cat: C.Catalogue, system: str) -> pd.DataFrame:
    """Per-case C1 measures of one system (label order; absent cases NaN)."""
    lab = store.labels
    f = store.cases(system)
    present = f["present"].to_numpy()
    idx, sc = store.top(system)
    hits = C.membership(idx, lab.positives, cat.n)
    lawm = np.where(idx >= 0, cat.law[np.maximum(idx, 0)], -1)
    cited = [np.unique(cat.law[p]) for p in lab.positives]
    keys = np.unique(np.concatenate([i * len(cat.law_ids) + c for i, c in enumerate(cited)]))
    in_cited = np.isin(np.arange(lab.n)[:, None] * len(cat.law_ids) + lawm, keys) & (idx >= 0)
    nrel = np.asarray([len(p) for p in lab.positives])
    ranks = {k: np.zeros(lab.n, dtype=np.int64) for k in "abc"}
    top3 = np.full((lab.n, 3), -2, dtype=np.int64)
    for i in range(lab.n):
        if not present[i]:
            continue
        oa, ob, oc = law_orders(idx[i], sc[i], cat.law)
        ranks["a"][i], ranks["b"][i], ranks["c"][i] = (_first_rank(o, cited[i]) for o in (oa, ob, oc))
        top3[i, :min(3, oa.size)] = oa[:3]
    out: dict[str, Any] = {"query_id": lab.query_id, "component_id": lab.component, "fold": lab.fold, "present": present}
    for m in ("law_at_1", "law_at_10", "success_at_1", "ndcg_at_10", "success_at_1_in_law"):
        out[m] = f[m].to_numpy(dtype=np.float64)
    out["within_law_at_1_cond"] = np.where(out["law_at_1"] == 1, out["success_at_1"], np.nan)
    for k in "abc":
        out.update(_rank_measures(ranks[k], f"law_{k}"))
    out["recomputed_ndcg_at_10"] = C.ndcg10(hits.astype(float), nrel)
    oracle = _reorder(hits, in_cited, keep_rest=False)
    gate1 = _reorder(hits, (lawm == lawm[:, :1]) & (idx >= 0))
    gate3 = _reorder(hits, (lawm[:, :, None] == top3[:, None, :]).any(-1) & (idx >= 0))
    for name, h in (("oracle", oracle), ("gate1", gate1), ("gate3", gate3)):
        out[f"{name}_ndcg_at_10"] = C.ndcg10(h.astype(float), nrel)
        out[f"{name}_success_at_1"] = h[:, 0].astype(float)
    frame = pd.DataFrame(out)
    cols = [c for c in frame.columns if c not in ("query_id", "component_id", "fold", "present")]
    frame.loc[~present, cols] = np.nan
    return frame


def baselines(store: C.RunStore, cat: C.Catalogue) -> pd.DataFrame:
    """Law popularity (fold-local, cases) and 1/size baselines of the law task."""
    lab = store.labels
    nl = len(cat.law_ids)
    cited = [np.unique(cat.law[p]) for p in lab.positives]
    by_fold = {}
    for k in np.unique(lab.fold):
        cnt = np.zeros(nl, dtype=np.int64)
        for i in np.flatnonzero(lab.fold != k):
            cnt[cited[i]] += 1
        by_fold[int(k)] = np.lexsort((np.arange(nl), -cnt))
    size_order = np.lexsort((np.arange(nl), cat.law_size))
    pop = np.asarray([_first_rank(by_fold[int(lab.fold[i])], cited[i]) for i in range(lab.n)])
    inv = np.asarray([_first_rank(size_order, cited[i]) for i in range(lab.n)])
    out = {"query_id": lab.query_id, "component_id": lab.component, "fold": lab.fold}
    out.update(_rank_measures(pop, "law_popularity"))
    out.update(_rank_measures(inv, "law_inv_size"))
    out["uniform_within_law_at_1"] = np.asarray(
        [np.mean([1.0 / cat.law_size[c] for c in cited[i]]) if cited[i].size else np.nan for i in range(lab.n)])
    return pd.DataFrame(out)


def summarise(frame: pd.DataFrame, boot: C.Boot, measures: Sequence[str]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for m in measures:
        d = boot.macro(frame[m].to_numpy(dtype=np.float64))
        row[m], row[f"{m}_lo"], row[f"{m}_hi"] = d["mean"], d["lo"], d["hi"]
    return row


def decomposition(frame: pd.DataFrame) -> dict[str, float]:
    p = frame[frame["present"].astype(bool)]
    l1, s1 = p["law_at_1"].to_numpy(float), p["success_at_1"].to_numpy(float)
    within = s1[l1 == 1].mean() if (l1 == 1).any() else np.nan
    return {"pooled_law_at_1": float(l1.mean()), "pooled_within_law_at_1_cond": float(within),
            "pooled_product": float(l1.mean() * within), "pooled_success_at_1": float(s1.mean()),
            "n_cases": int(len(p))}


def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", systems: Sequence[str] | None = None,
        draws: int = C.DRAWS, seed: int = C.SEED) -> dict[str, Any]:
    out = Path(out)
    store = C.RunStore(runs, population)
    cat = C.load_catalogue(str(data))
    boot = C.Boot(store.labels.component, draws, seed)
    names = list(systems) if systems else store.systems
    rows, checks = [], []
    for s in names:
        f = system_cases(store, cat, s)
        row = {"system": s, "family": C.family(s), "n_cases": int(f["present"].sum())}
        row.update(summarise(f, boot, CASE_MEASURES))
        dec = decomposition(f)
        row.update(dec)
        row["macro_product"] = row["law_at_1"] * row["within_law_at_1_cond"]
        p = f[f["present"].astype(bool)]
        checks.append({"system": s, "law_at_1_mismatch_cases": int((p["law_a_at_1"] != p["law_at_1"]).sum()),
                       "ndcg_max_abs_diff": float(np.nanmax(np.abs(p["recomputed_ndcg_at_10"] - p["ndcg_at_10"])))
                       if len(p) else float("nan")})
        rows.append(row)
    table = pd.DataFrame(rows)
    base = baselines(store, cat)
    brow = {"system": "baseline", "n_cases": store.labels.n}
    brow.update(summarise(base, boot, [c for c in base.columns if c.startswith(("law_", "uniform_"))]))
    arrows = []
    by = table.set_index("system")
    skipped_arrows = []
    for a, b in ARROWS:
        if a in by.index and b in by.index:
            arrows.append({"from": a, "to": b, "x0": by.at[a, "law_at_1"], "y0": by.at[a, "within_law_at_1_cond"],
                           "x1": by.at[b, "law_at_1"], "y1": by.at[b, "within_law_at_1_cond"],
                           "d_law_at_1": by.at[b, "law_at_1"] - by.at[a, "law_at_1"],
                           "d_within_law_at_1_cond": by.at[b, "within_law_at_1_cond"] - by.at[a, "within_law_at_1_cond"],
                           "d_success_at_1_in_law": by.at[b, "success_at_1_in_law"] - by.at[a, "success_at_1_in_law"]})
        else:
            skipped_arrows.append(f"{a} -> {b}")
    C.write_csv(out / "c1_systems.csv", table)
    C.write_csv(out / "c1_scatter.csv", table[["system", "family", "law_at_1", "law_at_1_lo", "law_at_1_hi",
                                               "within_law_at_1_cond", "within_law_at_1_cond_lo",
                                               "within_law_at_1_cond_hi", "success_at_1", "ndcg_at_10"]])
    C.write_csv(out / "c1_decomposition.csv", table[["system", "law_at_1", "within_law_at_1_cond", "macro_product",
                                                     "success_at_1", "success_at_1_in_law", "pooled_law_at_1",
                                                     "pooled_within_law_at_1_cond", "pooled_product",
                                                     "pooled_success_at_1", "n_cases"]])
    C.write_csv(out / "c1_baselines.csv", [brow])
    C.write_csv(out / "c1_arrows.csv", arrows)
    C.write_csv(out / "c1_checks.csv", checks)
    notes = [f"Reference labels: {store.labels.reference}; {store.labels.n} cases in {boot.G} groups.",
             "Means are component-macro; the pooled decomposition is exact (success@1 = law@1 x within-law@1 | law@1).",
             "Law task: laws outside the top 200 are unranked; (b) sums raw scores on the system's own scale.",
             "Oracle and gates re-order the saved top 200 only; cited articles beyond rank 200 stay unretrieved.",
             "Not run: the optional 264-class law classifier (ANALYSIS_PLAN_C §2)."]
    if skipped_arrows:
        notes.append("Arrows skipped (a system is absent from this store): " + ", ".join(skipped_arrows) + ".")
    if store.label_mismatch:
        notes.append(f"Systems whose stored labels differ from the reference on some cases: {store.label_mismatch}.")
    ev = C.Evidence("C1")
    named = [s for s in _named(store) if s in by.index]
    for s in named:
        r = by.loc[s]
        src = store.source(s)
        ci = lambda m: C.fmt_ci({"mean": r[m], "lo": r[f"{m}_lo"], "hi": r[f"{m}_hi"]})
        ev.add(f"{s} on {population}", ["law@1", "within-law@1 (conditional)", "success@1", "nDCG@10",
                                        "within-law@1 (stored, unconditional)"],
               [ci("law_at_1"), ci("within_law_at_1_cond"), ci("success_at_1"), ci("ndcg_at_10"),
                ci("success_at_1_in_law")], src)
        ev.add(f"{s} on {population}, law task", ["law MRR (a) best article", "law@1 (b) top-3 sum", "law@1 (c) top-50 count",
                                                  "law@3 (a)"],
               [ci("law_a_mrr"), ci("law_b_at_1"), ci("law_c_at_1"), ci("law_a_at_3")], src.replace("per_case", "top200"))
        ev.add(f"{s} on {population}, re-orderings", ["law-oracle nDCG@10", "gate-1 nDCG@10", "gate-3 nDCG@10",
                                                      "law-oracle success@1"],
               [ci("oracle_ndcg_at_10"), ci("gate1_ndcg_at_10"), ci("gate3_ndcg_at_10"), ci("oracle_success_at_1")],
               src.replace("per_case", "top200"))
        ev.add(f"{s} on {population}, decomposition (pooled cases)", ["law@1", "within-law@1 | law@1", "product", "success@1"],
               [C.fmt(r["pooled_law_at_1"]), C.fmt(r["pooled_within_law_at_1_cond"]), C.fmt(r["pooled_product"]),
                C.fmt(r["pooled_success_at_1"])], src)
    bci = lambda m: C.fmt_ci({"mean": brow[m], "lo": brow[f"{m}_lo"], "hi": brow[f"{m}_hi"]})
    ev.add(f"law-task baselines on {population}", ["popularity law MRR", "popularity law@1", "1/size law MRR", "1/size law@1",
                                                   "uniform pick within the cited law (1/size)"],
           [bci("law_popularity_mrr"), bci("law_popularity_at_1"), bci("law_inv_size_mrr"), bci("law_inv_size_at_1"),
            bci("uniform_within_law_at_1")], f"runs/*/{population}/per_case.parquet (labels) + catalogue")
    for a in arrows:
        ev.add(f"arrow {a['from']} -> {a['to']} on {population}", ["change in law@1", "change in within-law@1 (conditional)"],
               [f"{a['d_law_at_1']:+.3f}", f"{a['d_within_law_at_1_cond']:+.3f}"],
               f"{store.source(a['from'])}; {store.source(a['to'])}")
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C1", "plan": "ANALYSIS_PLAN_C §2", "population": population, "runs": str(runs),
           "reference_labels": store.labels.reference, "cases": store.labels.n, "groups": boot.G,
           "draws": draws, "seed": seed, "systems": len(names), "notes": notes, "checks": checks,
           "baselines": brow, "arrows": arrows, "skipped_arrows": skipped_arrows,
           "label_mismatch": store.label_mismatch}
    C.write_json(out / "c1.json", rec)
    return rec


def _named(store: C.RunStore) -> list[str]:
    from uol.analysis_c.c3_gain import SYSTEMS as C3_SYSTEMS
    want = [x for a in ARROWS for x in a] + list(C3_SYSTEMS) + list(C.answer_systems().values())
    seen, out = set(), []
    for s in want:
        if s not in seen and store.has(s):
            seen.add(s)
            out.append(s)
    return out
