"""Label-noise simulation (ANALYSIS_PLAN_C §13a, D109; 6.8; descriptive).

The X11 design of 15 Oct on the final rankers: cited (case, article) pairs of the population are removed at random at
1 %, 3.3 % and 5 % of all its cited pairs, 200 draws per rate (seed 20261024); a case left without a cited article
drops out of that draw, as evaluation drops it. nDCG@10 is recomputed from each system's saved top 200 with the
remaining labels, with no refit. Each contrast is the component-macro mean of the per-case differences (the measure
of the main tables), over the cases both systems scored. Reported per rate: the contrast without noise, mean, 2.5 and
97.5 percentiles and lowest draw, and the number of draws below the margin .02.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.analysis_c import common as C

RATES = (0.01, 0.033, 0.05)
DRAWS = 200
SEED = 20261024
MARGIN = 0.02
CONTRASTS = (("S_full", "S0"), ("S_full_ext", "S0"), ("qwen3_8b_vectors", "lstar_headings"))


def macro(values: np.ndarray, comp_inv: np.ndarray, n_comp: int) -> float:
    ok = np.isfinite(values)
    s = np.bincount(comp_inv[ok], weights=values[ok], minlength=n_comp)
    c = np.bincount(comp_inv[ok], minlength=n_comp)
    return float((s[c > 0] / c[c > 0]).mean()) if (c > 0).any() else float("nan")


def simulate(hits: dict[str, np.ndarray], case: np.ndarray, n_cases: int, comp_inv: np.ndarray,
             contrasts: Sequence[tuple[str, str]], rates: Sequence[float] = RATES, draws: int = DRAWS,
             seed: int = SEED) -> dict[str, Any]:
    """``hits[s]``: [P, 10] boolean, pair p's article at rank r of system s (NaN-free; a missing case has none);
    ``case``: the case of each pair. Returns the contrast distributions per rate."""
    P = len(case)
    n_comp = int(comp_inv.max()) + 1 if len(comp_inv) else 0
    disc = C.DISC

    def ndcg(keep: np.ndarray) -> dict[str, np.ndarray]:
        nrel = np.bincount(case[keep], minlength=n_cases)
        idcg = C.IDCG[np.minimum(nrel, 10)]
        out = {}
        for s, h in hits.items():
            if s.startswith("__"):
                continue
            dcg = np.bincount(case[keep], weights=(h[keep] * disc).sum(axis=1), minlength=n_cases)
            with np.errstate(invalid="ignore", divide="ignore"):
                v = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), np.nan)
            out[s] = np.where(hits["__present__" + s], v, np.nan)
        return out

    def contrast_values(nd: dict[str, np.ndarray]) -> dict[str, float]:
        return {f"{a} - {b}": macro(nd[a] - nd[b], comp_inv, n_comp) for a, b in contrasts}

    full = contrast_values(ndcg(np.ones(P, dtype=bool)))
    rng = np.random.default_rng(seed)
    res: dict[str, Any] = {"none": {k: {"value": v} for k, v in full.items()}}
    for rate in rates:
        m = int(round(rate * P))
        vals: dict[str, list[float]] = {k: [] for k in full}
        for _ in range(draws):
            keep = np.ones(P, dtype=bool)
            keep[rng.choice(P, size=m, replace=False)] = False
            for k, v in contrast_values(ndcg(keep)).items():
                vals[k].append(v)
        res[f"{rate:g}"] = {k: {"pairs_removed": m, "mean": float(np.mean(v)), "p2_5": float(np.percentile(v, 2.5)),
                                 "p97_5": float(np.percentile(v, 97.5)), "lowest": float(np.min(v)),
                                 "draws_below_margin": int(np.sum(np.asarray(v) < MARGIN)), "draws": draws}
                            for k, v in vals.items()}
    return res


def run(runs: Path, out: Path, data: Path | None = None, population: str = "DEV", draws: int = DRAWS,
        seed: int = SEED, contrasts: Sequence[tuple[str, str]] = CONTRASTS) -> dict[str, Any]:
    store = C.RunStore(runs, population)
    lab = store.labels
    wanted = sorted({s for c in contrasts for s in c})
    have, missing = C.present(store, wanted)
    contrasts = [c for c in contrasts if c[0] in have and c[1] in have]
    case, art = lab.pair_arrays()
    hits: dict[str, np.ndarray] = {}
    check = {}
    for s in sorted({x for c in contrasts for x in c}):
        idx, _ = store.top(s, 10)
        hits[s] = idx
    for s, idx in list(hits.items()):
        top = idx[case]                                             # [P, 10] the system's first ten for each pair's case
        hits[s] = (top == art[:, None]).astype(np.float64)
        hits["__present__" + s] = (idx >= 0).any(axis=1)
        saved = store.cases(s)["ndcg_at_10"].to_numpy(dtype=np.float64) if "ndcg_at_10" in store.cases(s) else None
        if saved is not None:                                       # the recomputed scores reproduce the saved ones
            nrel = np.bincount(case, minlength=lab.n)
            dcg = np.bincount(case, weights=(hits[s] * C.DISC).sum(axis=1), minlength=lab.n)
            with np.errstate(invalid="ignore", divide="ignore"):
                mine = np.where(nrel > 0, dcg / C.IDCG[np.minimum(nrel, 10)].clip(1e-12), np.nan)
            ok = np.isfinite(saved) & hits["__present__" + s]
            check[s] = float(np.nanmax(np.abs(mine[ok] - saved[ok]))) if ok.any() else None
    _, comp_inv = np.unique(lab.component, return_inverse=True)
    res = simulate(hits, case, lab.n, comp_inv, contrasts, draws=draws, seed=seed)
    rec = {"item": "noise", "scope": C.SCOPE, "population": population, "cases": lab.n, "cited_pairs": int(len(case)),
           "rates": list(RATES), "draws": draws, "seed": seed, "margin": MARGIN, "contrasts": [f"{a} - {b}" for a, b in contrasts],
           "missing_systems": missing, "reproduction_max_abs_diff": check, "label_reference": lab.reference,
           "results": res, "notes": [f"missing systems: {missing}"] if missing else []}
    out.mkdir(parents=True, exist_ok=True)
    C.write_json(out / "noise.json", rec)
    rows = [{"rate": r, "contrast": k, **v} for r, d in res.items() for k, v in d.items()]
    C.write_csv(out / "noise.csv", rows)
    return rec
