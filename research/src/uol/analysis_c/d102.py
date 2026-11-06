"""D102 analyses (ANALYSIS_PLAN_C §12, version 4; descriptive): each stage's ceiling and the law-first alternative.

**Stage ceilings.** For every system with a ``top200.parquet``, on every population folder it has: the oracle
nDCG@10 of a perfect reorder of its top k (k = 30, 100): the relevant articles found within the top k move first (in
their list order), the rest of the list keeps its order; strict ideal DCG over all of the case's relevant articles
(the population's own ``positive_idx``). Pool recall from the receipts' fit read-out, as in the recall sweep
(``recall_sweep.pool_recall``), when present.

For the base rankers S0, S_full and S_full_ext, beside each second stage over the same base: the stage's stored
nDCG@10, the base's stored nDCG@10 and the base's oracle@30 on the same cases (query ids both have). Stages scored on
DEV-A300 (claude_rerank_S_full_ext, jev_S_full_ext, jev_rrf_S_full_ext) are read against the base's DEV per-case and
top 200 restricted to the same query ids. Headroom used = (mean stage - mean base) / (mean oracle@30 - mean base) on
group-macro means (a per-case ratio is undefined where the oracle equals the base); intervals from the same cluster
bootstrap draws (2,000, seed 20261023, over the DEV groups present).

**Law-first.** From the base's top 200 per case, one law is chosen and only that law's articles are listed in the
base's order, nothing after them. Primary rule: the law of the top-ranked article. Secondary rule: the law with the
largest sum of its three best article scores in the top 200 (C1 rule (b), ties by first appearance). Reported:
nDCG@10 (strict ideal) and success@10 of each rule, paired differences against the base (recomputed from the same
top 200, equal to the stored values) with cluster-bootstrap intervals, and law@1 under each rule (the chosen law is
a cited law).

Stand-ins: RUN-A has none of the plan's base rankers, so a run may name other systems (``stages`` and
``law_base``); every row then says which plan system it stands in for.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C
from uol.analysis_c.c1_law import law_orders
from uol.analysis_c.recall_sweep import pool_recall, populations

KS = (30, 100)
A300 = "DEV-A300"
STAGES: dict[str, list[tuple[str, str]]] = {
    "S0": [("llm_rerank_S0", "DEV")],
    "S_full": [],
    "S_full_ext": [("llm_rerank_S_full_ext", "DEV"), ("ce_trained", "DEV"), ("rerank_bge", "DEV"),
                   ("rerank_bge_alone", "DEV"), ("rerank_qwen3_4b", "DEV"), ("rerank_qwen3_4b_alone", "DEV"),
                   ("claude_rerank_S_full_ext", A300), ("jev_S_full_ext", A300), ("jev_rrf_S_full_ext", A300)],
}
LAW_BASE = "S_full_ext"


def oracle_ndcg(idx: np.ndarray, positives: Sequence[np.ndarray], n_articles: int, k: int) -> np.ndarray:
    """Per case, nDCG@10 after moving the relevant articles of the top k to the front (stable)."""
    hits = C.membership(idx, positives, n_articles)
    front = hits & (np.arange(idx.shape[1]) < k)[None, :]
    order = np.argsort(~front, axis=1, kind="stable")
    h = np.take_along_axis(hits, order, axis=1)
    nrel = np.asarray([len(p) for p in positives])
    return C.ndcg10(h.astype(float), nrel)


def law_first(idx: np.ndarray, score: np.ndarray, positives: Sequence[np.ndarray], law: np.ndarray,
              n_articles: int) -> dict[str, np.ndarray]:
    """Per case: nDCG@10, success@10 and law@1 of the base and of the two law-first rules."""
    n = idx.shape[0]
    hits = C.membership(idx, positives, n_articles)
    lawm = np.where(idx >= 0, law[np.maximum(idx, 0)], -1)
    nrel = np.asarray([len(p) for p in positives])
    chosen = {"primary": np.full(n, -2), "secondary": np.full(n, -2)}
    for i in range(n):
        if idx[i, 0] < 0:
            continue
        oa, ob, _ = law_orders(idx[i], score[i], law)
        chosen["primary"][i], chosen["secondary"][i] = oa[0], ob[0]
    out: dict[str, np.ndarray] = {}
    empty = idx[:, 0] < 0
    out["base_ndcg_at_10"] = np.where(empty, np.nan, C.ndcg10(hits.astype(float), nrel))
    out["base_success_at_10"] = np.where(empty, np.nan, hits[:, :10].any(axis=1).astype(float))
    cited = [set(np.unique(law[p]).tolist()) for p in positives]
    out["base_law_at_1"] = np.where(empty, np.nan, [float(lawm[i, 0] in cited[i]) for i in range(n)])
    for rule, lw in chosen.items():
        keep = (lawm == lw[:, None]) & (idx >= 0)
        order = np.argsort(~keep, axis=1, kind="stable")
        h = np.take_along_axis(hits & keep, order, axis=1)
        out[f"{rule}_ndcg_at_10"] = np.where(empty, np.nan, C.ndcg10(h.astype(float), nrel))
        out[f"{rule}_success_at_10"] = np.where(empty, np.nan, h[:, :10].any(axis=1).astype(float))
        out[f"{rule}_law_at_1"] = np.where(empty, np.nan, [float(lw[i] in cited[i]) for i in range(n)])
    return out


def _ci(mean: float, boot: np.ndarray) -> dict[str, float]:
    lo, hi = C.interval(boot)
    return {"value": mean, "ci_low": lo, "ci_high": hi}


def _restrict(store: C.RunStore, system: str, population: str, query_ids: set[str] | None) -> np.ndarray:
    """Per case (label order) the stored nDCG@10 of ``system`` on ``population``; NaN outside ``query_ids``."""
    f = pd.read_parquet(store.runs / system / population / "per_case.parquet").drop_duplicates("query_id")
    s = f.set_index("query_id")["ndcg_at_10"].reindex(store.labels.query_id).to_numpy(float)
    if query_ids is not None:
        s = np.where(np.isin(store.labels.query_id, list(query_ids)), s, np.nan)
    return s


def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", systems: Sequence[str] | None = None,
        draws: int = C.DRAWS, seed: int = C.SEED, receipts: Path | None = None,
        stages: dict[str, list[tuple[str, str]]] | None = None, law_base: str = LAW_BASE,
        stand_in: dict[str, str] | None = None) -> dict[str, Any]:
    runs, out = Path(runs), Path(out)
    receipts = Path(receipts) if receipts else runs.parent / "receipts"
    stages = STAGES if stages is None else stages
    stand_in = stand_in or {}
    cat = C.load_catalogue(str(data))
    notes: list[str] = []
    ev = C.Evidence("D102")
    # ------------------------------------------------------------------ (1a, b) oracle@k and pool recall, every system
    orc_rows: list[dict[str, Any]] = []
    for pop, names in populations(runs, systems).items():
        frames = {s: pd.read_parquet(runs / s / pop / "per_case.parquet").drop_duplicates("query_id").sort_values("query_id")
                  for s in names}
        boot = C.Boot(sorted({str(c) for f in frames.values() for c in f["component_id"]}), draws, seed)
        for s in names:
            f = frames[s]
            qids = f["query_id"].to_numpy(dtype=str)
            tops = pd.read_parquet(runs / s / pop / "top200.parquet", columns=["query_id", "rank", "candidate_idx"])
            pos_map = {q: i for i, q in enumerate(qids)}
            r = tops["query_id"].map(pos_map)
            tops = tops[r.notna()]
            idx = np.full((len(qids), C.TOP_K), -1, dtype=np.int64)
            idx[r[r.notna()].to_numpy(np.int64), tops["rank"].to_numpy(np.int64) - 1] = tops["candidate_idx"].to_numpy()
            pos = [C.parse_idx(v) for v in f["positive_idx"]]
            g = boot.group_index(f["component_id"].astype(str))
            vals = {"ndcg_at_10": f["ndcg_at_10"].to_numpy(float)}
            for k in KS:
                vals[f"oracle_at_{k}"] = oracle_ndcg(idx, pos, cat.n, k)
            for m, v in vals.items():
                gm = np.bincount(g, weights=np.nan_to_num(v), minlength=boot.G)
                cnt = np.bincount(g, weights=np.isfinite(v).astype(float), minlength=boot.G)
                ok = cnt > 0
                gm = gm[ok] / cnt[ok]
                W = boot.W[:, ok]
                orc_rows.append({"system": s, "population": pop, "measure": m,
                                 **_ci(float(gm.mean()), (W @ gm) / W.sum(axis=1)), "note": ""})
            pr = pool_recall(receipts, s, pop) if receipts.exists() else None
            if pr is not None:
                orc_rows.append({"system": s, "population": pop, "measure": "pool_recall", "value": pr["value"],
                                 "ci_low": None, "ci_high": None, "note": pr["basis"]})
    if not receipts.exists():
        notes.append(f"No receipts folder at {receipts.as_posix()}: pool recall rows skipped.")
    # ------------------------------------------------------------------ (1c) stage gains against the oracle
    store = C.RunStore(runs, population)
    boot = C.Boot(store.labels.component, draws, seed)
    stage_rows: list[dict[str, Any]] = []
    for base, lst in stages.items():
        if not store.has(base):
            notes.append(f"Base {base} absent: its stage rows are skipped.")
            continue
        idx, _ = store.top(base)
        orc30 = oracle_ndcg(idx, store.labels.positives, cat.n, 30)
        base_nd = store.cases(base)["ndcg_at_10"].to_numpy(float)
        for stage, spop in lst:
            sp = runs / stage / spop / "per_case.parquet"
            if not sp.exists():
                notes.append(f"Stage {stage} on {spop} absent: skipped.")
                continue
            qids = set(pd.read_parquet(sp, columns=["query_id"])["query_id"].astype(str))
            st = _restrict(store, stage, spop, qids)            # DEV-A300 rows read against the base's DEV rows
            ok = np.isfinite(st) & np.isfinite(base_nd) & np.isin(store.labels.query_id, list(qids))
            if not ok.any():
                notes.append(f"Stage {stage} shares no case with {base}: skipped.")
                continue
            m = lambda v: boot.macro_boot(np.where(ok, v, np.nan))
            (mb, bb), (ms, bs), (mo, bo) = m(base_nd), m(st), m(orc30)
            with np.errstate(invalid="ignore", divide="ignore"):
                used, used_b = (ms - mb) / (mo - mb), (bs - bb) / (bo - bb)
            row = {"base": base, "stands_in_for": stand_in.get(base, base), "stage": stage, "stage_population": spop,
                   "cases": int(ok.sum()), "base_ndcg_at_10": mb, "stage_ndcg_at_10": ms, "oracle_at_30": mo}
            for name, pt, bt in (("gain", ms - mb, bs - bb), ("headroom", mo - mb, bo - bb), ("headroom_used", used, used_b)):
                lo, hi = C.interval(bt)
                row.update({name: pt, f"{name}_lo": lo, f"{name}_hi": hi})
            stage_rows.append(row)
            ev.add(f"{stage} over {base}{' (stand-in for ' + stand_in[base] + ')' if base in stand_in else ''} on "
                   f"{spop} cases", ["gain nDCG@10", "oracle@30 headroom", "headroom used"],
                   [C.fmt_ci({"mean": row["gain"], "lo": row["gain_lo"], "hi": row["gain_hi"]}),
                    C.fmt_ci({"mean": row["headroom"], "lo": row["headroom_lo"], "hi": row["headroom_hi"]}),
                    C.fmt_ci({"mean": row["headroom_used"], "lo": row["headroom_used_lo"], "hi": row["headroom_used_hi"]})],
                   f"{store.source(base, 'top200.parquet')}; runs/{stage}/{spop}/per_case.parquet",
                   scope=f"{spop}, descriptive (ANALYSIS_PLAN_C §12 v4)")
    # ------------------------------------------------------------------ (2) law-first
    lf_rows: list[dict[str, Any]] = []
    if store.has(law_base):
        idx, sc = store.top(law_base)
        present = store.cases(law_base)["present"].to_numpy()
        lf = law_first(idx, sc, store.labels.positives, cat.law, cat.n)
        lf = {k: np.where(present, v, np.nan) for k, v in lf.items()}
        role = stand_in.get(law_base, law_base)
        for rule in ("base", "primary", "secondary"):
            row: dict[str, Any] = {"base": law_base, "stands_in_for": role, "rule": rule}
            for meas in ("ndcg_at_10", "success_at_10", "law_at_1"):
                pt, bt = boot.macro_boot(lf[f"{rule}_{meas}"])
                lo, hi = C.interval(bt)
                row.update({meas: pt, f"{meas}_lo": lo, f"{meas}_hi": hi})
                if rule != "base" and meas != "law_at_1":
                    pt, bt = boot.macro_boot(lf[f"{rule}_{meas}"] - lf[f"base_{meas}"])
                    lo, hi = C.interval(bt)
                    row.update({f"diff_{meas}": pt, f"diff_{meas}_lo": lo, f"diff_{meas}_hi": hi})
            lf_rows.append(row)
        stored = store.cases(law_base)["ndcg_at_10"].to_numpy(float)
        notes.append(f"Law-first base {law_base}" + (f" (stand-in for {role})" if role != law_base else "")
                     + f"; recomputed base nDCG@10 vs stored, max |diff| "
                     f"{np.nanmax(np.abs(lf['base_ndcg_at_10'] - stored)):.2g}.")
        tag = f"{law_base}" + (f" (stand-in for {role})" if role != law_base else "")
        for r in lf_rows[1:]:
            ev.add(f"law-first {r['rule']} rule over {tag} on {population}",
                   ["nDCG@10", "difference vs base", "success@10", "difference vs base", "law@1 (chosen law cited)"],
                   [C.fmt_ci({"mean": r["ndcg_at_10"], "lo": r["ndcg_at_10_lo"], "hi": r["ndcg_at_10_hi"]}),
                    C.fmt_ci({"mean": r["diff_ndcg_at_10"], "lo": r["diff_ndcg_at_10_lo"], "hi": r["diff_ndcg_at_10_hi"]}),
                    C.fmt_ci({"mean": r["success_at_10"], "lo": r["success_at_10_lo"], "hi": r["success_at_10_hi"]}),
                    C.fmt_ci({"mean": r["diff_success_at_10"], "lo": r["diff_success_at_10_lo"], "hi": r["diff_success_at_10_hi"]}),
                    C.fmt_ci({"mean": r["law_at_1"], "lo": r["law_at_1_lo"], "hi": r["law_at_1_hi"]})],
                   store.source(law_base, "top200.parquet"), scope=f"{population}, descriptive (ANALYSIS_PLAN_C §12 v4)")
    else:
        notes.append(f"Law-first base {law_base} absent: skipped.")
    orc = pd.DataFrame(orc_rows)
    focus = [s for s in dict.fromkeys(list(stages) + [law_base] + list(C.answer_systems().values()))]
    for s in focus:
        sub = orc[(orc["system"] == s) & (orc["population"] == population)].set_index("measure") if len(orc) else orc
        if len(sub):
            tag = s + (f" (stand-in for {stand_in[s]})" if s in stand_in else "")
            ev.add(f"{tag} on {population}, stage ceilings", ["nDCG@10", "oracle nDCG@10 of top 30", "oracle nDCG@10 of top 100"],
                   [C.fmt_ci({"mean": sub.at[m, "value"], "lo": sub.at[m, "ci_low"], "hi": sub.at[m, "ci_high"]})
                    for m in ("ndcg_at_10", "oracle_at_30", "oracle_at_100")], store.source(s, "top200.parquet"),
                   scope=f"{population}, descriptive (ANALYSIS_PLAN_C §12 v4)")
    C.write_csv(out / "d102_ceilings.csv", orc)
    C.write_csv(out / "d102_stage_gain.csv", stage_rows)
    C.write_csv(out / "d102_law_first.csv", lf_rows)
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "D102", "plan": "ANALYSIS_PLAN_C §12 (version 4)", "runs": str(runs), "population": population,
           "stages": {b: [list(x) for x in v] for b, v in stages.items()}, "law_base": law_base, "stand_in": stand_in,
           "stage_gain": stage_rows, "law_first": lf_rows, "notes": notes, "draws": draws, "seed": seed}
    C.write_json(out / "d102.json", rec)
    return rec
