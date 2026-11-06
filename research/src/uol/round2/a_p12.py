"""Round 2 / A: P12 co-citation graph features with a degree-preserving null (EXPLORATORY, dev cross-fitting only).

Design (PLAN rule 2):
* Graph: article-article co-citation from dev training-fold labels only. A component contributes one unit edge per pair
  of distinct articles in the union of its positives. Leak-free nesting: for outer group k the scoring rows (fold k) use
  the graph of the four other folds; a training row of fold j uses the graph of the three folds other than j and k.
* Seeds: the query's top-5 candidates of ``K_lex`` (out-of-fold score file; label use is fold-local).
* Features per (query, pooled candidate): log1p co-citation count with the seeds, cosine-normalised co-citation
  (count / sqrt(strength_s * strength_c)), and personalised-PageRank mass from the seeds (restart 0.3, 30 steps; the
  restart term is removed so the seeds' own rank feature is not duplicated).
* Null: the same features on degree-preserving rewired graphs (stub-level double-edge swaps on the unit-edge multigraph,
  10 x |E| swaps, no self-loops, so every article keeps its co-citation strength). Popularity-only control: the same three
  features from the configuration-model expectation (strength_s * strength_c / 2m), i.e. degree alone.
* Model: ``LTR_emb_hgb`` feature set + the 3 columns, refit per dev group with the frozen per-group hyper-parameters;
  baseline = identical refit without the columns (must reproduce dev P_spec nDCG@10 of ``LTR_emb_hgb`` within .002).
* Lockbox: never touched (no lockbox group is fitted or scored).

Run: ``py -3.12 -m uol.round2.a_p12 --nulls 5``.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from typing import Any

import numpy as np
import scipy.sparse as sp

from uol.embedding_push import base, common, final
from uol.embedding_push import fusion_ltr as fl
from uol.round2 import a_common as A

SEED = 20261004
N_SEEDS = 5
PPR_RESTART = 0.3
PPR_STEPS = 30
SWAP_FACTOR = 10
FEATS = ("cc_log", "cc_cos", "ppr")
OUT = A.OUT / "p12"


# ----------------------------------------------------------------------------- features table (memory-mapped)


def load_features(ctx: base.Ctx) -> fl.Features:
    meta = json.loads((fl.FEATDIR / "meta.json").read_text(encoding="utf-8"))
    by_id = {s.id: s for s in fl.discover()}
    sources = [by_id[s["id"]] for s in meta["sources"]]
    for s, m in zip(sources, meta["sources"]):
        if s.sha != m["sha256"]:
            raise RuntimeError(f"source {s.id} changed since the frozen feature table")
    if fl.features_key(sources) != meta["key"]:
        raise RuntimeError("feature key mismatch")
    pq, pc, pmask = (np.load(fl.FEATDIR / f"{n}.npy") for n in ("pq", "pc", "pmask"))
    X = np.load(fl.FEATDIR / "X.npy", mmap_mode="r")
    F = fl.finish_features(ctx, sources, pq, pc, pmask, X, meta["columns"], meta["full_ids"])
    F.table_key = meta["key"]
    return F


# ----------------------------------------------------------------------------- graphs


def unit_edges(ctx: base.Ctx, folds: tuple[int, ...]) -> np.ndarray:
    """[E, 2] unit co-citation edges (i < j) from the components of the given dev folds."""
    d = ctx.data
    comps: dict[str, set[int]] = {}
    for r in ctx.dev:
        if int(ctx.fold[r]) in folds:
            comps.setdefault(d.queries[int(r)].component_id, set()).update(d.positive_idx[int(r)])
    edges = []
    for s in comps.values():
        a = np.asarray(sorted(s))
        if len(a) < 2:
            continue
        i, j = np.triu_indices(len(a), 1)
        edges.append(np.stack([a[i], a[j]], 1))
    return np.concatenate(edges).astype(np.int64)


def rewire(edges: np.ndarray, seed: int, factor: int = SWAP_FACTOR) -> np.ndarray:
    """Degree(strength)-preserving double-edge swaps on the unit-edge multigraph; self-loops rejected."""
    rng = np.random.default_rng(seed)
    e = edges.copy()
    m = len(e)
    n_try = factor * m
    a_idx = rng.integers(0, m, n_try)
    b_idx = rng.integers(0, m, n_try)
    flip = rng.random(n_try) < 0.5
    done = 0
    for t in range(n_try):
        a, b = a_idx[t], b_idx[t]
        if a == b:
            continue
        u, v = e[a]
        x, y = e[b]
        if flip[t]:
            x, y = y, x
        # (u,v),(x,y) -> (u,y),(x,v)
        if u == y or x == v:
            continue
        e[a, 0], e[a, 1] = u, y
        e[b, 0], e[b, 1] = x, v
        done += 1
    return e


def adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    w = np.ones(len(edges), dtype=np.float32)
    W = sp.coo_matrix((np.r_[w, w], (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]])), shape=(n, n)).tocsr()
    W.sum_duplicates()
    return W


def graph_features(W: sp.csr_matrix | None, strength: np.ndarray, seeds: np.ndarray, popularity_only: bool = False) -> np.ndarray:
    """[B, N, 3] float32: features of every candidate for B queries with seed sets ``seeds`` [B, k]."""
    n = len(strength)
    B = len(seeds)
    S = sp.csr_matrix((np.ones(seeds.size, dtype=np.float32), (np.repeat(np.arange(B), seeds.shape[1]), seeds.ravel())), shape=(B, n))
    two_m = float(strength.sum())
    out = np.zeros((B, n, 3), dtype=np.float32)
    if popularity_only:
        ds = np.asarray(S @ strength).ravel()                      # sum of seed strengths
        exp = (ds[:, None] * strength[None, :]) / max(two_m, 1.0)
        out[:, :, 0] = np.log1p(exp)
        sq = np.asarray(S @ np.sqrt(strength)).ravel()
        out[:, :, 1] = (sq[:, None] * np.sqrt(strength)[None, :]) / max(two_m, 1.0)
        out[:, :, 2] = (strength / max(two_m, 1.0))[None, :] * (1 - PPR_RESTART)
        return out
    C = (S @ W).toarray()                                          # co-citation counts with the seeds
    out[:, :, 0] = np.log1p(C)
    inv = np.where(strength > 0, 1.0 / np.sqrt(np.maximum(strength, 1e-12)), 0.0).astype(np.float32)
    Wn = sp.diags(inv) @ W @ sp.diags(inv)
    out[:, :, 1] = (S @ Wn).toarray()
    # personalised PageRank on the articles that have co-citation edges: p = r e + (1 - r) P^T p, P row-normalised;
    # the restart vector is uniform over the query's seeds that have edges (no such seed: all zero)
    deg = np.asarray(W.sum(axis=1)).ravel()
    active = np.flatnonzero(deg > 0)
    Wa = W[active][:, active]
    Pa = (sp.diags(1.0 / deg[active]) @ Wa).T.tocsr().astype(np.float32)
    Sa = S[:, active].toarray().T.astype(np.float32)               # [N', B]
    cnt = Sa.sum(axis=0, keepdims=True)
    E = np.divide(Sa, cnt, out=np.zeros_like(Sa), where=cnt > 0)
    p = E.copy()
    for _ in range(PPR_STEPS):
        p = PPR_RESTART * E + (1 - PPR_RESTART) * (Pa @ p)
    out[:, active, 2] = (p - PPR_RESTART * E).T
    return out


# ----------------------------------------------------------------------------- per-group feature columns


def seeds_for(rows: np.ndarray) -> np.ndarray:
    mm = common.open_scores("K_lex")
    block = fl.sanitize(np.asarray(mm[np.sort(rows)], dtype=np.float32))
    return np.argpartition(-block, N_SEEDS, axis=1)[:, :N_SEEDS]  # aligned with np.sort(rows)


def group_columns(ctx: base.Ctx, F: fl.Features, v: fl.Variant, g: base.Group, kind: str, rep: int, edge_cache: dict) -> np.ndarray:
    """[len(v.idx), 3] columns for the dev rows of the variant pool (NaN elsewhere) for outer group g."""
    n = ctx.N
    cols = np.full((len(F.pq), 3), np.nan, dtype=np.float32)
    k = int(g.role.replace("dev_fold", ""))
    for j in range(5):
        rows = np.sort(ctx.dev[ctx.fold[ctx.dev] == j])
        folds = tuple(f for f in range(5) if f != j and (f != k or j == k))   # fold k rows: other 4 folds; fold j rows: 3 folds
        key = (folds, kind, rep)
        if key not in edge_cache:
            e = unit_edges(ctx, folds)
            if kind == "rewired":
                e = rewire(e, SEED + 7919 * rep + 101 * sum(1 << f for f in folds))
            edge_cache[key] = e
        e = edge_cache[key]
        W = adjacency(e, n)
        strength = np.asarray(W.sum(axis=1)).ravel().astype(np.float32)
        seeds = seeds_for(rows)
        for s0 in range(0, len(rows), 256):
            rr = rows[s0:s0 + 256]
            feats = graph_features(W, strength, seeds[s0:s0 + 256], popularity_only=(kind == "popularity"))
            for b, r in enumerate(rr):
                sl = slice(F.row_ptr[r], F.row_ptr[r + 1])
                cols[sl] = feats[b, F.pc[sl]]
    return cols


# ----------------------------------------------------------------------------- HGB refit with optional extra columns


def fit_predict(ctx: base.Ctx, F: fl.Features, v: fl.Variant, g: base.Group, cfg: dict[str, Any], base_cache: dict,
                extra: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingClassifier

    seed = fl.SEED + 1000 * g.idx
    if "tr" not in base_cache:
        rng = np.random.default_rng(seed + 999)
        tr_idx, tr_y = fl.hgb_training_pairs(ctx, F, v, fl.train_rows_of(ctx, g), rng, "spec")
        cols = np.asarray(v.hgb_cols)
        base_cache["tr"] = (tr_idx, tr_y, np.asarray(F.X[tr_idx[:, None], cols[None, :]]))
        idx_out = fl.row_pairs(v, g.rows, False, F)
        base_cache["out"] = (idx_out, np.concatenate([np.asarray(F.X[idx_out[s:s + 500_000, None], cols[None, :]])
                                                      for s in range(0, len(idx_out), 500_000)]))
    tr_idx, tr_y, Xtr = base_cache["tr"]
    idx_out, Xout = base_cache["out"]
    if extra is not None:
        Xtr = np.hstack([Xtr, extra[tr_idx]])
        Xout = np.hstack([Xout, extra[idx_out]])
    model = HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"], random_state=seed, **fl.HGB_FIXED)
    model.fit(Xtr, tr_y)
    return idx_out, model.decision_function(Xout).astype(np.float32)


# ----------------------------------------------------------------------------- evaluation


def evaluate_variant(ctx: base.Ctx, F: fl.Features, preds: dict[str, tuple[np.ndarray, np.ndarray]], matrix: np.ndarray,
                     lstar: np.ndarray) -> dict[str, dict]:
    for g in ctx.groups:
        if g.role in preds:
            idx, sc = preds[g.role]
            fl.write_pooled(ctx, F, matrix, g.rows, idx, sc, lstar)
    out = {}
    for pop in ("P_spec", "P"):
        rows = common.population_rows(pop, common.dev_rows())
        res = common.evaluate(np.asarray(matrix[rows]), rows, pop, top_k=0)
        for rec in res["per_query"]:
            rec.pop("topk", None)
        out[pop] = res
    return out


def summarize(name: str, res: dict, ref: dict | None) -> dict[str, Any]:
    s = {"variant": name, "P_spec": res["P_spec"]["macro"]["ndcg_at_10"], "P": res["P"]["macro"]["ndcg_at_10"],
         "P_spec_R20": res["P_spec"]["macro"]["recall_at_20"]}
    if ref is not None:
        for pop in ("P_spec", "P"):
            c = common.contrast(res[pop], ref[pop])
            s[f"delta_{pop}"] = c["effect"]
            s[f"ci95_{pop}"] = c["ci95"]
        s["bands_P_spec"] = final.band_table(res["P_spec"], ref["P_spec"], A.band_labels)
    return s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nulls", type=int, default=5)
    ap.add_argument("--reproduce-only", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ctx = base.Ctx()
    F = load_features(ctx)
    v = fl.make_variants(F)["emb"]
    man = json.loads((fl.pipeline_path("LTR_emb_hgb").parent / "manifest.json").read_text(encoding="utf-8"))
    cfgs = {role: rec["config"] for role, rec in man["configuration"].items()}
    if len(v.hgb_cols) != man["n_hgb_features"]:
        raise RuntimeError("variant feature count differs from the frozen LTR_emb_hgb")
    A.log(f"P12: features mapped ({len(F.pq):,} pairs, {len(v.hgb_cols)} hgb cols); frozen cfgs {cfgs}")
    dev_groups = [g for g in ctx.groups if not g.is_lockbox]
    variants = ["baseline"] if args.reproduce_only else (["baseline", "real", "popularity"] + [f"rewired{r}" for r in range(args.nulls)])
    preds: dict[str, dict[str, tuple]] = {name: {} for name in variants}
    edge_cache: dict = {}
    graph_stats = {}
    for g in dev_groups:
        tg = time.perf_counter()
        base_cache: dict = {}
        for name in variants:
            ckpt = OUT / f"pred_{name}_{g.role}.npz"
            if ckpt.exists():
                z = np.load(ckpt)
                preds[name][g.role] = (z["idx"], z["score"])
                continue
            if name == "baseline":
                extra = None
            else:
                kind = "real" if name == "real" else "popularity" if name == "popularity" else "rewired"
                rep = int(name.replace("rewired", "")) if kind == "rewired" else 0
                extra = group_columns(ctx, F, v, g, kind, rep, edge_cache)
            idx, sc = fit_predict(ctx, F, v, g, cfgs[g.role], base_cache, extra)
            np.savez(ckpt, idx=idx, score=sc)
            preds[name][g.role] = (idx, sc)
            A.log(f"P12 {g.role} {name}: fitted ({time.perf_counter() - tg:.0f}s into group)")
        k = int(g.role.replace("dev_fold", ""))
        e = edge_cache.get((tuple(f for f in range(5) if f != k), "real", 0))
        if e is not None:
            graph_stats[g.role] = {"unit_edges": int(len(e)), "articles_with_edges": int(len(np.unique(e)))}
        del base_cache
    lstar = common.open_scores("L_star")
    matrix = np.full((len(ctx.data.queries), ctx.N), np.nan, dtype=np.float32)
    results = {}
    for name in variants:
        results[name] = evaluate_variant(ctx, F, preds[name], matrix, lstar)
        A.log(f"P12 {name}: dev P_spec {results[name]['P_spec']['macro']['ndcg_at_10']:.4f}  P {results[name]['P']['macro']['ndcg_at_10']:.4f}")
    ref = results["baseline"]
    frozen = json.loads((fl.FUSION / "dev_results.json").read_text(encoding="utf-8"))["summaries"]["LTR_emb_hgb"]["P_spec"]["ndcg_at_10"]
    repro = {"frozen_dev_P_spec": frozen, "refit_dev_P_spec": ref["P_spec"]["macro"]["ndcg_at_10"],
             "abs_diff": abs(frozen - ref["P_spec"]["macro"]["ndcg_at_10"])}
    repro["within_002"] = repro["abs_diff"] <= 0.002
    A.log(f"P12 reproduction: {repro}")
    summary = {"reproduction": repro, "graph_stats": graph_stats, "variants": {}}
    for name in variants:
        summary["variants"][name] = summarize(name, results[name], None if name == "baseline" else ref)
    nulls = [n for n in variants if n.startswith("rewired")]
    if "real" in results and nulls:
        real_vs_null = {}
        for n in nulls:
            c = common.contrast(results["real"]["P_spec"], results[n]["P_spec"])
            real_vs_null[n] = {"effect": c["effect"], "ci95": c["ci95"]}
        d_real = summary["variants"]["real"]["delta_P_spec"]
        d_null = [summary["variants"][n]["delta_P_spec"] for n in nulls]
        summary["real_vs_null"] = {"per_null": real_vs_null, "delta_real": d_real, "delta_null_mean": float(np.mean(d_null)),
                                   "delta_null_max": float(np.max(d_null)), "rank_of_real_among_real_and_nulls": int(1 + sum(x >= d_real for x in d_null)),
                                   "n_nulls": len(nulls)}
        c_pop = common.contrast(results["real"]["P_spec"], results["popularity"]["P_spec"])
        summary["real_vs_popularity"] = {"effect": c_pop["effect"], "ci95": c_pop["ci95"]}
    summary["wall_seconds"] = time.perf_counter() - t0
    A.save_json("p12_summary.json", summary)
    A.log(f"P12 done in {summary['wall_seconds']:.0f}s")
    return 0 if repro["within_002"] else 2


if __name__ == "__main__":
    sys.exit(main())


def md_summary() -> str:
    """Markdown tables from p12_summary.json (used for the report)."""
    s = json.loads((A.OUT / "p12_summary.json").read_text(encoding="utf-8"))
    v = s["variants"]
    lines = ["| variant | dev P_spec nDCG@10 | Δ vs refit [95% CI] | dev P nDCG@10 | Δ P [95% CI] | R@20 Δ frequent | rare | unseen |",
             "|---|---:|---|---:|---|---:|---:|---:|"]
    nulls = [k for k in v if k.startswith("rewired")]
    for name in ["baseline", "real", "popularity"] + nulls:
        r = v[name]
        if name == "baseline":
            lines.append(f"| baseline refit (= frozen LTR_emb_hgb) | {r['P_spec']:.4f} | – | {r['P']:.4f} | – | – | – | – |")
            continue
        b = r["bands_P_spec"]
        lines.append(f"| {name} | {r['P_spec']:.4f} | {r['delta_P_spec']:+.4f} {A.ci(r['ci95_P_spec'], 4)} | {r['P']:.4f} | "
                     f"{r['delta_P']:+.4f} {A.ci(r['ci95_P'], 4)} | {b['frequent']['diff']:+.3f} | {b['rare']['diff']:+.3f} | {b['unseen']['diff']:+.3f} |")
    return "\n".join(lines)
