"""Recall sweep (ANALYSIS_PLAN_C §12, 23 Oct 2026 amendment, version 3; descriptive).

For every system with a ``top200.parquet``, on every population folder it has:

* article level, k = 1..200: success@k (a relevant article in the top k) and Recall@k (share of the case's relevant
  articles in the top k); the relevant set is the case's ``positive_idx`` in the same population's per-case file;
* law level, k = 1..50: law@k, laws ranked by their best article's rank (first appearance in the list), a hit if any
  law of a relevant article is among the first k laws; laws absent from the top 200 are unranked;
* pool (learned rankers): pool recall from the receipts' fit read-out ``folds[].fit.pool_recall``, pooled over folds
  (relevant articles inside the pool / relevant articles when the read-out carries counts, otherwise the case-weighted
  or plain mean of the fold values, as stated in the row's note). Receipts are looked up in ``<runs>/../receipts``;
  when absent (RUN-A's receipts lack the field) the pool row is skipped with a note.

Values are group-macro means (cases averaged within ``component_id``, then over groups). Intervals: cluster bootstrap
over the population's groups (2,000 draws, seed 20261023; one draw matrix per population over the union of its
groups, so systems are paired) at k = 10, 30, 100 only (law level: 10, 30); the pool row has no interval (fold
aggregates only). Cases without a relevant article are left out of Recall@k and success@k (NaN), as in every measure.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

K_ARTICLE = 200
K_LAW = 50
MARKED = (10, 30, 100)


def populations(runs: Path, systems: Sequence[str] | None = None) -> dict[str, list[str]]:
    """population -> systems with both per_case and top200 files there."""
    out: dict[str, list[str]] = {}
    for p in sorted(Path(runs).glob("*/*/top200.parquet")):
        sysname, pop = p.parent.parent.name, p.parent.name
        if (p.parent / "per_case.parquet").exists() and (systems is None or sysname in systems):
            out.setdefault(pop, []).append(sysname)
    return out


def curves(top: np.ndarray, positives: Sequence[np.ndarray], law: np.ndarray, n_laws: int) -> dict[str, np.ndarray]:
    """Per case [n, k]: success@k, Recall@k (k = 1..200) and law@k (k = 1..50); NaN rows for cases without labels."""
    n = top.shape[0]
    n_art = int(max(law.size, top.max(initial=-1) + 1))
    hits = C.membership(top, positives, n_art).astype(np.float64)
    cum = np.cumsum(hits, axis=1)
    nrel = np.asarray([len(p) for p in positives], dtype=np.float64)
    ok = nrel > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(ok[:, None], cum / np.where(ok, nrel, 1)[:, None], np.nan)
    success = np.where(ok[:, None], (cum > 0).astype(float), np.nan)
    law_rank = np.zeros(n, dtype=np.int64)
    for i in range(n):
        row = top[i][top[i] >= 0]
        if not row.size or not ok[i]:
            continue
        L = law[row]
        _, first = np.unique(L, return_index=True)
        seq = L[np.sort(first)]
        hit = np.flatnonzero(np.isin(seq, np.unique(law[positives[i]])))
        law_rank[i] = hit[0] + 1 if hit.size else 0
    ks = np.arange(1, K_LAW + 1)
    law_k = ((law_rank[:, None] >= 1) & (law_rank[:, None] <= ks[None, :])).astype(float)
    law_k[~ok] = np.nan
    pad = K_ARTICLE - recall.shape[1]
    if pad > 0:                                       # shorter lists: the curve stays flat beyond the last rank
        recall = np.concatenate([recall, np.repeat(recall[:, -1:], pad, axis=1)], axis=1)
        success = np.concatenate([success, np.repeat(success[:, -1:], pad, axis=1)], axis=1)
    return {"success": success, "recall": recall, "law": law_k}


def macro_matrix(values: np.ndarray, group: np.ndarray, G: int) -> tuple[np.ndarray, np.ndarray]:
    """Group means [G, K] (NaN cases dropped per column) and the per-group 'has a value' mask [G, K]."""
    ok = np.isfinite(values)
    s = np.zeros((G, values.shape[1]))
    c = np.zeros((G, values.shape[1]))
    np.add.at(s, group, np.where(ok, values, 0.0))
    np.add.at(c, group, ok.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1), np.nan), c > 0


def pool_recall(receipts: Path, system: str, population: str) -> dict[str, Any] | None:
    p = Path(receipts) / f"{system}__{population}.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    vals, hits, tots, ns = [], [], [], []
    for f in rec.get("folds") or []:
        pr = ((f or {}).get("fit") or {}).get("pool_recall")
        if pr is None:
            continue
        if isinstance(pr, dict):
            h = next((pr[k] for k in ("hits", "found", "positives_in_pool", "in_pool") if k in pr), None)
            t = next((pr[k] for k in ("n", "total", "positives", "relevant", "n_relevant") if k in pr), None)
            v = next((pr[k] for k in ("recall", "pool_recall", "mean", "value") if k in pr), None)
            if h is not None and t:
                hits.append(float(h))
                tots.append(float(t))
            if v is not None:
                vals.append(float(v))
                ns.append(float(pr.get("cases", pr.get("n_cases", 0)) or 0))
        else:
            vals.append(float(pr))
    if hits and len(hits) == len(rec.get("folds") or []) or (hits and not vals):
        return {"value": sum(hits) / sum(tots), "basis": f"pooled counts over {len(hits)} folds"}
    if vals and ns and all(ns) and len(ns) == len(vals):
        return {"value": float(np.average(vals, weights=ns)), "basis": f"case-weighted mean of {len(vals)} fold values"}
    if vals:
        return {"value": float(np.mean(vals)), "basis": f"plain mean of {len(vals)} fold values (no counts recorded)"}
    return None


def run(runs: Path, out: Path, *, data: Path = C.DATA, systems: Sequence[str] | None = None, draws: int = C.DRAWS,
        seed: int = C.SEED, receipts: Path | None = None) -> dict[str, Any]:
    runs, out = Path(runs), Path(out)
    receipts = Path(receipts) if receipts else runs.parent / "receipts"
    cat = C.load_catalogue(str(data))
    rows: list[dict[str, Any]] = []
    notes: list[str] = []
    ev = C.Evidence("Recall sweep")
    pool_missing: list[str] = []
    pops = populations(runs, systems)
    for pop, names in pops.items():
        frames = {s: pd.read_parquet(runs / s / pop / "per_case.parquet").drop_duplicates("query_id").sort_values("query_id")
                  for s in names}
        comps = sorted({str(c) for f in frames.values() for c in f["component_id"]})
        boot = C.Boot(comps, draws, seed)
        for s in names:
            f = frames[s]
            store_top = _top(runs / s / pop / "top200.parquet", f["query_id"].to_numpy(dtype=str))
            pos = [C.parse_idx(v) for v in f["positive_idx"]]
            cv = curves(store_top, pos, cat.law, len(cat.law_ids))
            group = boot.group_index(f["component_id"].astype(str))
            vals: dict[tuple[str, int], tuple[float, float, float]] = {}
            for level, mat, K, kname in (("article", cv["success"], K_ARTICLE, "success"),
                                         ("article", cv["recall"], K_ARTICLE, "recall"),
                                         ("law", cv["law"], K_LAW, "law")):
                gm, has = macro_matrix(mat, group, boot.G)
                for k in range(1, K + 1):
                    col, ok = gm[:, k - 1], has[:, k - 1]
                    v = float(col[ok].mean()) if ok.any() else float("nan")
                    lo = hi = None
                    if k in MARKED and ok.any():
                        W = boot.W[:, ok]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            lo, hi = C.interval((W @ col[ok]) / W.sum(axis=1))
                    rows.append({"system": s, "population": pop, "level": level, "measure": kname, "k": k,
                                 "value": v, "ci_low": lo, "ci_high": hi})
                    if k in MARKED:
                        vals[(kname, k)] = (v, lo, hi)
            pr = pool_recall(receipts, s, pop) if receipts.exists() else None
            if pr is not None:
                rows.append({"system": s, "population": pop, "level": "pool", "measure": "pool_recall", "k": None,
                             "value": pr["value"], "ci_low": None, "ci_high": None, "note": pr["basis"]})
            elif C.family(s) == "ranker":
                pool_missing.append(f"{s}/{pop}")
            ci = lambda key: C.fmt_ci({"mean": vals[key][0], "lo": vals[key][1], "hi": vals[key][2]}) if key in vals else "n/a"
            meas = ["success@10", "Recall@10", "success@30", "Recall@30", "Recall@100", "law@10", "law@30"]
            keys = [("success", 10), ("recall", 10), ("success", 30), ("recall", 30), ("recall", 100), ("law", 10), ("law", 30)]
            values = [ci(k) for k in keys]
            if pr is not None:
                meas.append("pool recall")
                values.append(f"{C.fmt(pr['value'])} ({pr['basis']})")
            ev.add(f"{s} on {pop}", meas, values, f"runs/{s}/{pop}/top200.parquet; runs/{s}/{pop}/per_case.parquet",
                   scope=f"{pop}, descriptive (ANALYSIS_PLAN_C §12)")
    table = pd.DataFrame(rows)
    if "note" not in table:
        table["note"] = None
    table = table[["system", "population", "level", "measure", "k", "value", "ci_low", "ci_high", "note"]]
    C.write_csv(out / "recall_sweep.csv", table)
    if not receipts.exists():
        notes.append(f"No receipts folder at {receipts.as_posix()}: pool rows skipped.")
    if pool_missing and receipts.exists():
        notes.append(f"Pool recall absent from the receipts (no folds[].fit.pool_recall) for {len(pool_missing)} "
                     "learned-ranker runs; their pool rows are skipped.")
    notes.append("Intervals only at k = 10, 30, 100 (law level: 10, 30); the CSV leaves them empty elsewhere.")
    notes.append("Column 'measure' separates success@k and Recall@k inside level 'article'.")
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "recall_sweep", "plan": "ANALYSIS_PLAN_C §12 (version 3)", "runs": str(runs),
           "receipts": str(receipts), "populations": {p: len(v) for p, v in pops.items()}, "rows": len(table),
           "draws": draws, "seed": seed, "notes": notes, "pool_missing": pool_missing}
    C.write_json(out / "recall_sweep.json", rec)
    return rec


def _top(path: Path, query_ids: np.ndarray, k: int = K_ARTICLE) -> np.ndarray:
    t = pd.read_parquet(path, columns=["query_id", "rank", "candidate_idx"])
    t = t[t["rank"] <= k]
    pos = t["query_id"].map({q: i for i, q in enumerate(query_ids)})
    t = t[pos.notna()]
    idx = np.full((len(query_ids), k), -1, dtype=np.int64)
    idx[pos[pos.notna()].to_numpy(dtype=np.int64), t["rank"].to_numpy(dtype=np.int64) - 1] = t["candidate_idx"].to_numpy()
    return idx
