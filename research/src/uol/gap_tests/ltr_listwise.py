"""Test 6 (D56): listwise LambdaMART (LightGBM ``lambdarank``) for both rankers on the frozen feature table (dev only).

Same pools and HGB feature columns as the frozen rankers (``fusion_ltr.make_variants``: ``noemb`` and ``emb``), same
outer groups (the five dev folds; the lockbox group is never built), the same 2-inner-fold selection rule and the same
assembly of held-out scores (``fusion_ltr.write_pooled`` rule: pooled candidates keep model scores, the rest sit below
in L* order). Training lists: every pooled specific positive plus up to ``N_NEG`` sampled pool negatives per training
query (generic candidates removed, queries without a pooled positive dropped), as the frozen HGB but with a larger
negative sample, since a listwise loss needs lists.

Read-only on the frozen table: the cached ``fusion/features`` arrays are memory-mapped and the source list is checked
against the table's key, so nothing is rebuilt. ``fusion_ltr.flog`` is redirected so no frozen log is written.

    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.ltr_listwise smoke    # fold 0, noemb, one config, 50 trees
    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.ltr_listwise run      # both families, five folds
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import base, common as ep, fusion_ltr as fl
from uol.gap_tests import common as gc

TEST = "t6_listwise"
N_NEG = 300
GRID = [{"learning_rate": lr, "num_leaves": nl} for lr in (0.05, 0.1) for nl in (15, 63)]
FIXED = {"objective": "lambdarank", "metric": "ndcg", "n_estimators": 300, "min_child_samples": 40,
         "colsample_bytree": 0.5, "reg_lambda": 1.0, "max_bin": 127, "n_jobs": 8, "deterministic": True,
         "force_row_wise": True, "verbose": -1}
SEED = fl.SEED


def load_features(ctx: base.Ctx) -> fl.Features:
    """The frozen feature table, memory-mapped; refuses to proceed if the discovered sources do not match its key."""
    fl.flog = lambda msg: gc.log(TEST, f"[fusion_ltr] {msg}")
    meta = json.loads((fl.FEATDIR / "meta.json").read_text(encoding="utf-8"))
    wanted = [s["id"] for s in meta["sources"]]
    found = {s.id: s for s in fl.discover()}
    sources = [found[i] for i in wanted]
    if fl.features_key(sources) != meta["key"]:
        raise RuntimeError("the discovered sources do not match the frozen feature table; nothing is rebuilt")
    pq, pc, pmask = (np.load(fl.FEATDIR / f"{n}.npy") for n in ("pq", "pc", "pmask"))
    X = np.load(fl.FEATDIR / "X.npy", mmap_mode="r")
    F = fl.finish_features(ctx, sources, pq, pc, pmask, X, meta["columns"], meta["full_ids"])
    F.table_key = meta["key"]
    return F


def training_lists(ctx, F, v, rows, rng, n_neg=N_NEG) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = fl.row_pairs(v, rows, True, F)
    y = F.y_spec[idx]
    r = F.pq[idx]
    has_pos = np.isin(r, np.unique(r[y > 0]))
    idx, y, r = idx[has_pos], y[has_pos], r[has_pos]
    neg = np.flatnonzero(y == 0)
    order = np.lexsort((rng.random(len(neg)), r[neg]))
    neg, rn = neg[order], r[neg][order]
    first = np.searchsorted(rn, rn, side="left")
    keep = np.sort(np.concatenate([np.flatnonzero(y > 0), neg[(np.arange(len(neg)) - first) < n_neg]]))
    idx, y = idx[keep], y[keep]
    _, counts = np.unique(F.pq[idx], return_counts=True)       # idx sorted by row, so groups are contiguous
    return idx, y.astype(np.int32), counts


def matrix(F, v, idx) -> np.ndarray:
    return np.asarray(F.X[idx[:, None], np.asarray(v.hgb_cols)[None, :]], dtype=np.float32)


def fit(F, v, idx, y, groups, cfg, seed, n_estimators=None):
    import lightgbm as lgb
    params = {**FIXED, **cfg, "random_state": seed}
    if n_estimators:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMRanker(**params)
    model.fit(matrix(F, v, idx), y, group=groups)
    return model


def predict(F, v, model, idx, chunk=400_000) -> np.ndarray:
    return np.concatenate([model.predict(matrix(F, v, idx[s:s + chunk])) for s in range(0, len(idx), chunk)]).astype(np.float32)


def heldout_ndcg(ctx, F, v, model, rows) -> np.ndarray:
    va_idx = fl.row_pairs(v, rows, True, F)
    pad = fl.pad_pairs(F, va_idx)
    S = np.full((len(pad.rows), pad.Pmax), -np.inf, dtype=np.float32)
    S[pad.seg, pad.pos] = predict(F, v, model, va_idx)
    nd = dict(zip(pad.rows.tolist(), fl.ndcg_padded(S, pad).tolist()))
    return np.asarray([nd.get(int(r), 0.0) for r in rows])


def fit_group(ctx, F, v, g, grid: Sequence[dict[str, Any]] = GRID, n_estimators=None) -> tuple[dict, np.ndarray, np.ndarray]:
    seed = SEED + 1000 * g.idx
    results = []
    for cfg in grid:
        rows_all, vals_all = [], []
        for inner in g.train_folds[:fl.INNER_FOLDS]:
            rng = np.random.default_rng(seed + 17 * inner)
            tr = fl.train_rows_of(ctx, g, [f for f in g.train_folds if f != inner])
            idx, y, groups = training_lists(ctx, F, v, tr, rng)
            t = time.time()
            model = fit(F, v, idx, y, groups, cfg, seed, n_estimators)
            va = fl.spec_rows_of(ctx, [inner])
            rows_all.append(va)
            vals_all.append(heldout_ndcg(ctx, F, v, model, va))
            gc.log(TEST, f"{v.name} {g.role} cfg {cfg} inner {inner}: {len(idx):,} pairs, fit {time.time() - t:.0f}s")
        results.append({"config": cfg, "objective": fl.macro_rows(ctx, np.concatenate(rows_all), np.concatenate(vals_all))})
    best = base.argbest([r["objective"] for r in results]) if len(results) > 1 else 0
    cfg = grid[best]
    rng = np.random.default_rng(seed + 999)
    idx, y, groups = training_lists(ctx, F, v, fl.train_rows_of(ctx, g), rng)
    model = fit(F, v, idx, y, groups, cfg, seed, n_estimators)
    out_idx = fl.row_pairs(v, g.rows, False, F)
    return {"grid": results, "chosen": cfg, "final_train_pairs": int(len(idx))}, out_idx, predict(F, v, model, out_idx)


def assemble(F, R, parts, lstar) -> np.ndarray:
    """[len(R), N] scores for dev P_spec rows from (pair idx, scores) of the outer groups (write_pooled rule)."""
    idx = np.concatenate([p[0] for p in parts])
    sc = np.concatenate([p[1] for p in parts])
    order = np.argsort(F.pq[idx], kind="stable")
    idx, sc = idx[order], sc[order]
    r = F.pq[idx]
    N = lstar.shape[1]
    out = np.empty((len(R), N), dtype=np.float32)
    for i, row in enumerate(R):
        lo, hi = np.searchsorted(r, row), np.searchsorted(r, row, side="right")
        cand, s = F.pc[idx[lo:hi]], sc[lo:hi]
        o = np.argsort(-lstar[i], kind="stable")
        rank = np.empty(N)
        rank[o] = np.arange(1, N + 1)
        floor = float(s.min()) if len(s) else 0.0
        row_out = ((floor - 1.0) - rank / (N + 1.0)).astype(np.float32)
        row_out[cand] = s
        out[i] = row_out
    return out


def run(smoke: bool = False) -> int:
    t0 = time.time()
    ctx = base.Ctx()
    F = load_features(ctx)
    variants = fl.make_variants(F)
    groups = [g for g in ctx.groups if not g.is_lockbox]
    gc.assert_dev(np.concatenate([g.rows for g in groups]))
    if smoke:
        g = groups[0]
        v = variants["noemb"]
        ev, idx, sc = fit_group(ctx, F, v, g, grid=GRID[:1], n_estimators=50)
        gc.log(TEST, f"smoke: fold 0 noemb, 1 config, 50 trees, {time.time() - t0:.0f}s total; {ev}; "
                     f"peak {gc.peak_gb():.1f} GB (no score reported)")
        return 0
    R = gc.dev_pspec_rows()
    lstar = np.asarray(ep.open_scores("L_star")[R], dtype=np.float32)
    res: dict[str, Any] = {"families": {}, "contrasts": {}}
    evals = {}
    for fam, frozen in (("noemb", "LTR_noemb_hgb"), ("emb", "LTR_emb_hgb")):
        v = variants[fam]
        parts, evid = [], {}
        for g in groups:
            ev, idx, sc = fit_group(ctx, F, v, g)
            parts.append((idx, sc))
            evid[g.role] = ev
        M = assemble(F, R, parts, lstar)
        evals[fam] = ep.evaluate(M, R, "P_spec")
        evals[frozen] = ep.evaluate(np.asarray(np.load(fl.PIPES / frozen / "scores.npy", mmap_mode="r")[R]), R, "P_spec")
        res["families"][fam] = {"listwise_ndcg_at_10": evals[fam]["macro"]["ndcg_at_10"],
                                "pointwise_frozen_ndcg_at_10": evals[frozen]["macro"]["ndcg_at_10"], "selection": evid}
        gc.write_json(gc.OUT / TEST / "result_partial.json", res)
        del M
    for name, (a, b) in {"emb listwise - emb pointwise": ("emb", "LTR_emb_hgb"),
                         "noemb listwise - noemb pointwise": ("noemb", "LTR_noemb_hgb"),
                         "gap listwise (emb - noemb)": ("emb", "noemb"),
                         "gap pointwise (emb - noemb)": ("LTR_emb_hgb", "LTR_noemb_hgb")}.items():
        c = ep.contrast(evals[a], evals[b])
        res["contrasts"][name] = {"effect": c["effect"], "ci95": c["ci95"]}
        gc.log(TEST, f"{name}: {gc.fmt(c)}")
    gl = evals["emb"]["per_component"]["ndcg_at_10"] - evals["noemb"]["per_component"]["ndcg_at_10"]
    gp = evals["LTR_emb_hgb"]["per_component"]["ndcg_at_10"] - evals["LTR_noemb_hgb"]["per_component"]["ndcg_at_10"]
    from uol.facts_to_law import stats
    c = stats.contrast(gl, gp)
    res["contrasts"]["gap listwise - gap pointwise"] = {"effect": c["effect"], "ci95": c["ci95"]}
    res["cpu_seconds"] = time.time() - t0
    res["peak_gb"] = gc.peak_gb()
    gc.write_json(gc.OUT / TEST / "result.json", res)
    return 0


if __name__ == "__main__":
    sys.exit(run(smoke=(sys.argv[1:2] == ["smoke"])))
