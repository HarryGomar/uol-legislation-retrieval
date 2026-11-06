"""D58 steps 3-4: heading-path (V2) article text for the dense source and for both learned rankers (dev only, no new search).

Plan: research/programs/gap_tests_20261019/PLAN.md, section D58. Development rows only: every score block is sliced by
dev row (asserted), the feature table holds dev rows only, and the lockbox group is never built. Frozen score files,
vectors, selections and pipelines are read only; new source blocks ([dev rows, N] float32) live in
``experiments/runs/gap_tests_20261019/d58/sources``.

Rebuilt sources (frozen per-fold configurations throughout):
* ``L_star``: the fold's frozen L* selection (stemmed BM25 + character n-grams, frozen w_char) on V2 text;
* ``docexp_bm25``: frozen configuration (cap 2000, all positives, k1 .9, b .4, concat) with V2 article text;
* ``K_lex``, ``K_lex_ext``, ``LP``, ``H_bge``, ``K_qwen8b``: D + c * (z(L*_new) - z(L*_frozen)), c = the frozen weight on
  the robust-z L* channel (exact for these linear fusions; the remainder's row median is checked);
* ``H_qwen8b`` = (1 - w) z(L*) + w z(dense) with the frozen w; ``KH_qwen8b`` = D + c * (z(H_new) - z(H_frozen));
* arm b: ``dense_qwen8b`` = MaxP cosine of the frozen query vectors (variant c) against the V2 article vectors.

    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.d58_refit dense
    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.d58_refit sources
    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.d58_refit refit control a b
"""
from __future__ import annotations

import json
import sys
import time
from typing import Any, Callable

import numpy as np

from uol.embedding_push import base, common as ep, fusion_ltr as fl, qwen_endpoint as qe
from uol.facts_to_law import fusion, lexical, metrics
from uol.gap_tests import common as gc
from uol.gap_tests.lexical_grid import variant_texts

TEST = "d58"
D = gc.OUT / TEST
SRC = D / "sources"
V2DIR = ep.ROOT / "data/derived/embedding_push_qwen8b_headings_20261019"
QSEL = ep.RUN / "qwen/selection"
BSEL = ep.RUN / "base/selection"
DOCEXP_CFG = {"cap": 2000, "pos_mode": "all", "k1": 0.9, "b": 0.4}
ARM_A = ("L_star", "docexp_bm25", "K_lex", "K_lex_ext", "LP", "H_bge", "K_qwen8b", "H_qwen8b", "KH_qwen8b")
ARMS = {"control": {}, "a": {s: f"a/{s}" for s in ARM_A},
        "b": {**{s: f"a/{s}" for s in ARM_A}, "dense_qwen8b": "b/dense_qwen8b", "H_qwen8b": "b/H_qwen8b",
              "KH_qwen8b": "b/KH_qwen8b"}}
FROZEN_PIPE = {"noemb": "LTR_noemb_hgb", "emb": "LTR_emb_hgb"}

DEV = gc.assert_dev(ep.dev_rows())
DEV_POS = np.full(len(ep.release().queries), -1, dtype=np.int64)
DEV_POS[DEV] = np.arange(len(DEV))


def log(msg: str) -> None:
    gc.log(TEST, msg)


def frozen(sid: str, rows: np.ndarray) -> np.ndarray:
    rows = gc.assert_dev(rows)
    return np.asarray(ep.open_scores(sid)[rows], dtype=np.float32)


def save_src(name: str, M: np.ndarray) -> None:
    p = SRC / f"{name}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.save(p, M.astype(np.float32, copy=False))


def new_src(name: str) -> np.ndarray:
    p = SRC / f"{name}.npy"
    p.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(p, mode="w+", dtype=np.float32, shape=(len(DEV), len(ep.release().candidate_ids)))


def load_src(name: str) -> np.ndarray:
    return np.load(SRC / f"{name}.npy", mmap_mode="r")


def getter(arm: str) -> Callable[[str, np.ndarray], np.ndarray]:
    repl = ARMS[arm]

    def get(sid: str, rows: np.ndarray) -> np.ndarray:
        rows = gc.assert_dev(rows)
        if sid in repl:
            return np.asarray(load_src(repl[sid])[DEV_POS[rows]], dtype=np.float32)
        return frozen(sid, rows)
    return get


def dev_groups(ctx: base.Ctx) -> list[base.Group]:
    return [g for g in ctx.groups if not g.is_lockbox]


# ----------------------------------------------------------------------------- step 3: dense alone

def v2_dense(q: np.ndarray) -> np.ndarray:
    A = np.load(V2DIR / "article_vectors.npy", mmap_mode="r")
    owner = np.load(V2DIR / "article_index.npy")
    N = len(ep.release().candidate_ids)
    out = np.empty((len(q), N), dtype=np.float32)
    for s, block in ep.maxp_scores(q, np.asarray(A), owner, N):
        out[s:s + len(block)] = block
    return out


def query_c() -> np.ndarray:
    variant = json.loads((QSEL / "dense.json").read_text(encoding="utf-8"))["selected"]
    ids = json.loads((qe.DERIVED / "query_ids.json").read_text(encoding="utf-8"))
    if ids != [q.query_id for q in ep.release().queries]:
        raise ValueError("query vectors out of release order")
    return np.load(qe.DERIVED / f"query_vectors_{variant}.npy", mmap_mode="r")


def never_cited_hits(S: np.ndarray, rows: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Per (row, unseen specific positive) pair: hit@k under the condensed ranking, and the pair's row."""
    from uol.round3 import common as C
    d = ep.release()
    roles = ep.roles()
    positives = [d.specific_idx[int(r)] for r in rows]
    res = metrics.evaluate_matrix(np.asarray(S, dtype=np.float64), positives, exclude_idx=d.generic_idx, top_k=0,
                                  tie_order=ep.tie_order())
    hits, owner = [], []
    for i, r in enumerate(rows):
        band = C.band_array(str(roles[int(r)]))
        for p, rank in zip(positives[i], res[i]["positive_ranks"]):
            if band[p] == "unseen":
                hits.append(float(rank <= k))
                owner.append(i)
    return np.asarray(hits), np.asarray(owner)


def dense() -> int:
    t0 = time.time()
    R = gc.dev_pspec_rows()
    Q = np.asarray(query_c()[DEV], dtype=np.float32)
    S2 = v2_dense(Q)
    save_src("b/dense_qwen8b", S2)
    log(f"V2 dense scored for {len(DEV)} dev rows ({time.time() - t0:.0f}s)")
    rpos = DEV_POS[R]
    e2 = ep.evaluate(S2[rpos], R)
    e0 = ep.evaluate(frozen("dense_qwen8b", R), R)
    c = ep.contrast(e2, e0)
    h2, o2 = never_cited_hits(S2[rpos], R)
    h0, o0 = never_cited_hits(frozen("dense_qwen8b", R), R)
    assert np.array_equal(o2, o0)
    comp = np.asarray([ep.release().queries[int(R[i])].component_id for i in o2])
    res = {"ndcg_at_10": {"V2": e2["macro"]["ndcg_at_10"], "V0": e0["macro"]["ndcg_at_10"]},
           "recall_at_20": {"V2": e2["macro"]["recall_at_20"], "V0": e0["macro"]["recall_at_20"]},
           "V2_minus_V0_ndcg": {"effect": c["effect"], "ci95": c["ci95"]},
           "never_cited_pairs": int(len(h2)), "never_cited_recall_at_20": {"V2": float(h2.mean()), "V0": float(h0.mean())},
           "never_cited_V2_minus_V0": gc.contrast(h2, h0, comp), "rows": int(len(R)), "seconds": time.time() - t0,
           "peak_gb": gc.peak_gb()}
    gc.write_json(D / "dense_result.json", res)
    log(f"dense alone: V2 {res['ndcg_at_10']['V2']:.4f} vs V0 {res['ndcg_at_10']['V0']:.4f}, {gc.fmt(res['V2_minus_V0_ndcg'])}; "
        f"never-cited R@20 {res['never_cited_recall_at_20']['V2']:.3f} vs {res['never_cited_recall_at_20']['V0']:.3f} "
        f"({len(h2)} pairs), {gc.fmt(res['never_cited_V2_minus_V0'])}")
    return 0


# ----------------------------------------------------------------------------- step 4a: rebuilt sources

def lstar_rows(stem: lexical.LexicalIndex, char: lexical.LexicalIndex, sel: base.LSel, texts: list[str]) -> np.ndarray:
    if sel.kind != "fuse" or sel.base != "bm25_stem":
        raise ValueError(f"unexpected frozen L* selection {sel}")
    bm = fusion.robust_z(stem.score(texts, {"k1": sel.k1, "b": sel.b}))
    ch = fusion.robust_z(char.score(texts))
    return fusion.linear_fusion([bm, ch], base.pair(sel.w_char), normalize="none")


def build_lstar(ctx: base.Ctx, texts_c: list[str], name: str | None, rows_limit: int | None = None) -> np.ndarray:
    lsel = json.loads((BSEL / "lexical.json").read_text(encoding="utf-8"))
    stem, char = lexical.LexicalIndex(texts_c, kind="bm25_stem"), lexical.LexicalIndex(texts_c, kind="char")
    out = new_src(name) if name else np.zeros((len(DEV), len(texts_c)), dtype=np.float32)
    for g in dev_groups(ctx):
        sel = base.LSel.from_json(lsel[g.role]["selected"])
        rows_g = g.rows if rows_limit is None else g.rows[:rows_limit]
        for s in range(0, len(rows_g), 256):
            rows = rows_g[s:s + 256]
            out[DEV_POS[rows]] = lstar_rows(stem, char, sel, [ctx.texts[int(r)] for r in rows])
        if rows_limit is not None:
            break
    if name:
        out.flush()
    return out


def build_docexp(ctx: base.Ctx, texts_c: list[str], rows_limit: int | None = None) -> np.ndarray:
    from uol.embedding_push import memory_proto as mp
    d = ep.release()
    original = d.candidate_texts
    out = np.zeros((len(DEV), len(texts_c)), dtype=np.float32) if rows_limit is not None else new_src("a/docexp_bm25")
    try:
        d.candidate_texts = texts_c
        for g in dev_groups(ctx):
            grp = mp.DocExpGroup(g.train_rows)
            rows = g.rows if rows_limit is None else g.rows[:rows_limit]
            for s in range(0, len(rows), 512):
                r = rows[s:s + 512]
                out[DEV_POS[r]] = mp.docexp_block(grp, r, DOCEXP_CFG["cap"], DOCEXP_CFG["pos_mode"], "concat",
                                                  DOCEXP_CFG["k1"], DOCEXP_CFG["b"])
            if rows_limit is not None:
                break
    finally:
        d.candidate_texts = original
    return out


def channel_weights() -> dict[str, dict[str, float]]:
    """Frozen per-fold coefficient on the robust-z L* (or H) channel of each linear fusion."""
    nested = json.loads((BSEL / "nested.json").read_text(encoding="utf-8"))
    lexc = json.loads((QSEL / "lex_control.json").read_text(encoding="utf-8"))
    kq = json.loads((QSEL / "K_qwen8b.json").read_text(encoding="utf-8"))
    kh = json.loads((QSEL / "KH_qwen8b.json").read_text(encoding="utf-8"))
    hq = json.loads((QSEL / "H_qwen8b.json").read_text(encoding="utf-8"))
    out: dict[str, dict[str, float]] = {k: {} for k in ("K_lex", "K_lex_ext", "LP", "H_bge", "K_qwen8b", "KH_qwen8b",
                                                         "H_qwen8b_w_dense")}
    for k in range(5):
        r = f"dev_fold{k}"
        out["K_lex"][r] = 1 - nested[r]["K_lex"]["w_knn"]
        out["K_lex_ext"][r] = 1 - lexc[r]["fused"]["w"]
        out["LP"][r] = 1.0
        h = nested[r]["H_bge"]["config"]
        if h["type"] != "linear":
            raise ValueError("H_bge RRF fold: not covered by the delta rule")
        out["H_bge"][r] = 1 - h["w_dense"]
        out["K_qwen8b"][r] = 1 - kq[r]["w"]
        fam = kh["pick"][r]["family"]
        out["KH_qwen8b"][r] = 1 - kh["best"][fam][r]["w"]
        out["H_qwen8b_w_dense"][r] = hq[r]["w_dense"]
    return out


def sources() -> int:
    t0 = time.time()
    ctx = base.Ctx()
    d = ctx.data
    v0 = list(d.candidate_texts)
    v2 = variant_texts(v0)[0]["V2_hierarchy_path"]
    # gate: the L* and docexp rebuilds reproduce the frozen files on V0 text (sample rows)
    gate_rows = dev_groups(ctx)[0].rows[:64]
    lstar_v0 = build_lstar(ctx, v0, None, rows_limit=64)
    diff_l = float(np.abs(lstar_v0[DEV_POS[gate_rows]] - frozen("L_star", gate_rows)).max())
    dex_v0 = build_docexp(ctx, v0, rows_limit=64)
    diff_d = float(np.abs(dex_v0[DEV_POS[gate_rows]] - frozen("docexp_bm25", gate_rows)).max())
    log(f"rebuild gate on V0: L* max |diff| {diff_l:.2e}; docexp max |diff| {diff_d:.2e}")
    if diff_l > 1e-4 or diff_d > 1e-3:
        raise SystemExit("rebuild gate failed")
    del lstar_v0, dex_v0
    L_new = build_lstar(ctx, v2, "a/L_star")
    log(f"L* V2 built ({time.time() - t0:.0f}s)")
    build_docexp(ctx, v2).flush()
    log(f"docexp V2 built ({time.time() - t0:.0f}s)")
    w = channel_weights()
    checks: dict[str, Any] = {"gate_V0": {"L_star_max_abs_diff": diff_l, "docexp_max_abs_diff": diff_d}, "remainder_median": {}}
    out = {s: new_src(f"a/{s}") for s in ("K_lex", "K_lex_ext", "LP", "H_bge", "K_qwen8b")}
    Hq = {arm: new_src(f"{arm}/H_qwen8b") for arm in ("a", "b")}
    KH = {arm: new_src(f"{arm}/KH_qwen8b") for arm in ("a", "b")}
    dense_b = load_src("b/dense_qwen8b")
    hcheck = 0.0
    for g in dev_groups(ctx):
        for s in range(0, len(g.rows), 256):
            rows = g.rows[s:s + 256]
            pos = DEV_POS[rows]
            zl_old = fusion.robust_z(frozen("L_star", rows))
            zl_new = fusion.robust_z(L_new[pos])
            for sid in out:
                c = np.float32(w[sid][g.role])
                Df = frozen(sid, rows)
                rest = Df - c * zl_old
                med = float(np.abs(np.median(rest, axis=1)).max())
                checks["remainder_median"][sid] = max(checks["remainder_median"].get(sid, 0.0), med)
                out[sid][pos] = Df + c * (zl_new - zl_old)
            wd = float(w["H_qwen8b_w_dense"][g.role])                 # python float: pair(w) exactly as stage_h
            zd0 = fusion.robust_z(frozen("dense_qwen8b", rows))
            H_old_re = fusion.linear_fusion([zl_old, zd0], base.pair(float(wd)), normalize="none")
            H_f = frozen("H_qwen8b", rows)
            hcheck = max(hcheck, float(np.abs(H_old_re - H_f).max()))
            zh_old = fusion.robust_z(H_f)
            ck = np.float32(w["KH_qwen8b"][g.role])
            KH_f = frozen("KH_qwen8b", rows)
            rest = KH_f - ck * zh_old
            checks["remainder_median"]["KH_qwen8b"] = max(checks["remainder_median"].get("KH_qwen8b", 0.0),
                                                           float(np.abs(np.median(rest, axis=1)).max()))
            for arm, zd in (("a", zd0), ("b", fusion.robust_z(np.asarray(dense_b[pos])))):
                Hn = fusion.linear_fusion([zl_new, zd], base.pair(float(wd)), normalize="none")
                Hq[arm][pos] = Hn
                KH[arm][pos] = KH_f + ck * (fusion.robust_z(Hn) - zh_old)
        log(f"derived sources: {g.role} done ({time.time() - t0:.0f}s)")
    checks["H_qwen8b_recompute_max_abs_diff"] = hcheck
    log(f"checks: {json.dumps(checks)}")
    for M in list(out.values()) + list(Hq.values()) + list(KH.values()):
        M.flush()
    gc.write_json(D / "sources_checks.json", checks)
    if hcheck > 1e-3 or max(checks["remainder_median"].values()) > 1e-3:
        raise SystemExit("a fusion formula check failed; derived sources are not valid")
    ev = {}
    R = gc.dev_pspec_rows()
    for name in ("a/L_star", "a/K_lex", "a/H_qwen8b", "b/H_qwen8b", "a/docexp_bm25"):
        ev[name] = ep.evaluate(np.asarray(load_src(name)[DEV_POS[R]]), R)["macro"]["ndcg_at_10"]
    for sid in ("L_star", "K_lex", "H_qwen8b", "docexp_bm25"):
        ev[f"frozen/{sid}"] = ep.evaluate(frozen(sid, R), R)["macro"]["ndcg_at_10"]
    checks["dev_ndcg_at_10"] = ev
    checks["weights"] = w
    checks["seconds"] = time.time() - t0
    checks["peak_gb"] = gc.peak_gb()
    gc.write_json(D / "sources_result.json", checks)
    log(f"sources written; dev nDCG@10 {json.dumps(ev)}")
    return 0


# ----------------------------------------------------------------------------- step 4b: dev-only feature table and refit

def build_features(ctx: base.Ctx, arm: str) -> fl.Features:
    fl.flog = lambda msg: log(f"[fusion_ltr] {msg}")
    meta = json.loads((fl.FEATDIR / "meta.json").read_text(encoding="utf-8"))
    found = {s.id: s for s in fl.discover()}
    sources = [found[s["id"]] for s in meta["sources"]]
    full = [s for s in sources if s.kind == "full"]
    if [s.id for s in full] != meta["full_ids"]:
        raise RuntimeError("full-source order differs from the frozen table")
    get = getter(arm)
    N, n_rows = ctx.N, len(ctx.data.queries)
    t0 = time.time()
    keys, flags = [], []
    cache = D / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(full):
        tag = ARMS[arm].get(s.id, "frozen").replace("/", "_")
        cp = cache / f"top{fl.TOPK}_{s.id}_{tag}.npy"
        if cp.exists():
            top = np.load(cp)
        else:
            top = np.empty((len(DEV), fl.TOPK), dtype=np.int32)
            for a in range(0, len(DEV), 512):
                top[a:a + 512] = fl.topk_block(fl.sanitize(get(s.id, DEV[a:a + 512])))
            np.save(cp, top)
        keys.append(np.repeat(DEV.astype(np.int64), fl.TOPK) * N + top.ravel())
        flags.append(np.full(keys[-1].shape, np.uint64(1) << np.uint64(i), dtype=np.uint64))
    keys, flags = np.concatenate(keys), np.concatenate(flags)
    order = np.argsort(keys, kind="stable")
    keys, flags = keys[order], flags[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ukeys, pmask = keys[starts], np.bitwise_or.reduceat(flags, starts)
    del keys, flags, order
    pq, pc = (ukeys // N).astype(np.int32), (ukeys % N).astype(np.int32)
    row_ptr = np.searchsorted(pq, np.arange(n_rows + 1))
    cols = [c for s in sources for c in fl.source_columns(s)] + fl.EXTRA_COLUMNS
    log(f"{arm}: pool {len(pq):,} pairs ({len(pq) / len(DEV):.0f} per dev row) ({time.time() - t0:.0f}s)")
    xp, done = D / f"X_{arm}.npy", D / f"X_{arm}.done"
    if done.exists() and json.loads(done.read_text())["n_pairs"] == len(pq):
        X = np.load(xp)                                         # completed earlier for this exact pool: reuse, in RAM
        log(f"{arm}: feature table reused from {xp.name}")
        F = fl.finish_features(ctx, sources, pq, pc, pmask, X, cols, meta["full_ids"])
        F.table_key = f"d58_{arm}"
        return F
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32, shape=(len(pq), len(cols)))
    table, cand_instr = fl.instrument_tables(ctx)
    perm, istarts, segment = fl.instrument_layout(cand_instr)
    col0 = 0
    for s in sources:
        if s.kind == "full":
            for a in range(0, len(DEV), 128):
                rows = DEV[a:a + 128]
                lo, hi = int(row_ptr[rows[0]]), int(row_ptr[rows[-1] + 1])
                if hi == lo:
                    continue
                block = fl.sanitize(get(s.id, rows))
                local = DEV_POS[pq[lo:hi]] - a
                cc = pc[lo:hi]
                raw = block[local, cc]
                med = np.median(block, axis=1)
                mad = np.median(np.abs(block - med[:, None]), axis=1)
                std = block.std(axis=1)
                scale = (1.4826 * mad).astype(np.float32)
                scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0)).astype(np.float32)
                z = (raw - med[local].astype(np.float32)) / scale[local]
                ordered = np.sort(block, axis=1)
                ptr = np.searchsorted(local, np.arange(len(rows) + 1))
                lr = np.empty(hi - lo, dtype=np.float32)
                for i in range(len(rows)):
                    s0, e0 = ptr[i], ptr[i + 1]
                    if e0 > s0:
                        n_le = np.searchsorted(ordered[i], raw[s0:e0], side="right")
                        lr[s0:e0] = np.log((N - n_le + 1).astype(np.float32))
                imax = np.maximum.reduceat(block[:, perm], istarts, axis=1)
                imaxz = (imax[local, segment[cand_instr[cc]]] - med[local].astype(np.float32)) / scale[local]
                X[lo:hi, col0], X[lo:hi, col0 + 1], X[lo:hi, col0 + 2] = z, lr, imaxz
        else:
            pairs_features_dev(s, pq, pc, X, col0, N)
        col0 += len(fl.source_columns(s))
    log_len = np.log1p(np.asarray([len(t) for t in ctx.data.candidate_texts], dtype=np.float32))   # V0 text length
    gid = np.where(ctx.fold >= 0, ctx.fold, len(ctx.groups) - 1)
    log_qlen = np.log1p(np.asarray([len(t) for t in ctx.texts], dtype=np.float32))
    X[:, col0], X[:, col0 + 1], X[:, col0 + 2] = log_len[pc], table[gid[pq], pc], log_qlen[pq]
    X.flush()
    done.write_text(json.dumps({"n_pairs": int(len(pq)), "cols": len(cols)}))
    del X
    X = np.load(xp)                                             # in RAM for the fits (random row gathers)
    log(f"{arm}: features built ({time.time() - t0:.0f}s)")
    F = fl.finish_features(ctx, sources, pq, pc, pmask, X, cols, meta["full_ids"])
    F.table_key = f"d58_{arm}"
    return F


def pairs_features_dev(src, pq, pc, X, col0, n_cand) -> None:
    """``fusion_ltr.pairs_features`` restricted to dev rows (other rows are dropped on load, never used)."""
    import pandas as pd
    z = np.load(src.folder / "pairs.npz")
    q = z["qidx"].astype(np.int64)
    keep = np.isin(q, DEV)
    q, c, s = q[keep], z["cidx"].astype(np.int64)[keep], z["score"].astype(np.float32)[keep]
    del z
    frame = pd.DataFrame({"q": q, "c": c, "s": s}).drop_duplicates(["q", "c"], keep="last")
    grp = frame.groupby("q")["s"]
    frame["med"] = grp.transform("median")
    frame["mad"] = (frame["s"] - frame["med"]).abs().groupby(frame["q"]).transform("median")
    frame["std"] = grp.transform("std").fillna(0.0)
    scale = 1.4826 * frame["mad"]
    frame["scale"] = np.where(scale > 1e-8, scale, np.where(frame["std"] > 1e-8, frame["std"], 1.0))
    frame["z"] = (frame["s"] - frame["med"]) / frame["scale"]
    frame["rank"] = grp.rank(method="min", ascending=False)
    frame["key"] = frame["q"] * n_cand + frame["c"]
    frame = frame.sort_values("key")
    skeys = frame["key"].to_numpy()
    pkeys = pq.astype(np.int64) * n_cand + pc
    pos = np.minimum(np.searchsorted(skeys, pkeys), len(skeys) - 1)
    present = skeys[pos] == pkeys
    for j, name in enumerate(("s", "z", "rank")):
        vals = frame[name].to_numpy(dtype=np.float32)[pos]
        if name == "rank":
            vals = np.log(vals)
        X[:, col0 + j] = np.where(present, vals, np.nan)
    X[:, col0 + 3] = present.astype(np.float32)


def assemble(F: fl.Features, R: np.ndarray, parts, lstar_R: np.ndarray) -> np.ndarray:
    idx = np.concatenate([p[0] for p in parts])
    sc = np.concatenate([p[1] for p in parts])
    order = np.argsort(F.pq[idx], kind="stable")
    idx, sc = idx[order], sc[order]
    r = F.pq[idx]
    N = lstar_R.shape[1]
    out = np.empty((len(R), N), dtype=np.float32)
    for i, row in enumerate(R):
        lo, hi = np.searchsorted(r, row), np.searchsorted(r, row, side="right")
        cand, s = F.pc[idx[lo:hi]], sc[lo:hi]
        o = np.argsort(-lstar_R[i], kind="stable")
        rank = np.empty(N)
        rank[o] = np.arange(1, N + 1)
        floor = float(s.min()) if len(s) else 0.0
        row_out = ((floor - 1.0) - rank / (N + 1.0)).astype(np.float32)
        row_out[cand] = s
        out[i] = row_out
    return out


def refit(arms: list[str]) -> int:
    import faulthandler
    trace = open(D / "stack_dumps.txt", "a")
    faulthandler.dump_traceback_later(600, repeat=True, file=trace)
    ctx = base.Ctx()
    R = gc.dev_pspec_rows()
    results = json.loads((D / "refit_result.json").read_text(encoding="utf-8")) if (D / "refit_result.json").exists() else {}
    for arm in arms:
        t0 = time.time()
        F = build_features(ctx, arm)
        variants = fl.make_variants(F)
        lstar_R = getter(arm)("L_star", R)
        out: dict[str, Any] = {"pool": {}, "ndcg_at_10": {}, "configs": {}}
        for fam, pipe in FROZEN_PIPE.items():
            if arm not in ("control", "a") and fam == "noemb":
                continue                                        # no EMB source changes in the NO-EMB ranker (= control or arm a)
            man = json.loads((fl.PIPES / pipe / "manifest.json").read_text(encoding="utf-8"))
            v = variants[fam]
            parts = []
            for g in dev_groups(ctx):
                cfg = man["configuration"][g.role]["config"]
                tg = time.time()
                m = fl.fit_group_hgb(ctx, F, v, g, fixed_cfg=cfg)
                log(f"{arm}/{fam} {g.role}: fit {cfg} ({time.time() - tg:.0f}s, {m.evidence.get('final_train_pairs')} pairs)")
                parts.append((m.pair_idx, m.pair_scores))
                out["configs"][f"{fam}/{g.role}"] = cfg
            M = assemble(F, R, parts, lstar_R)
            np.save(D / f"scores_{arm}_{fam}_pspec.npy", M)
            e = ep.evaluate(M, R)
            out["ndcg_at_10"][fam] = e["macro"]["ndcg_at_10"]
            out["pool"][fam] = fl.pool_stats(ctx, F, v)
            log(f"{arm}/{fam}: nDCG@10 {e['macro']['ndcg_at_10']:.4f}; pool {out['pool'][fam]} ({time.time() - t0:.0f}s)")
            del M
        out["seconds"] = time.time() - t0
        out["peak_gb"] = gc.peak_gb()
        results[arm] = out
        gc.write_json(D / "refit_result.json", results)
        del F, variants
        if arm == "control":
            ok = abs(out["ndcg_at_10"]["noemb"] - 0.4081) <= 0.003 and abs(out["ndcg_at_10"]["emb"] - 0.4721) <= 0.003
            log(f"control gate: {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise SystemExit("control arm does not reproduce the frozen rankers; D58 refit stops")
    return 0


def contrasts() -> int:
    """Paired component-bootstrap contrasts between arms (from the saved dev P_spec score matrices)."""
    R = gc.dev_pspec_rows()
    ev = {}
    for arm, fam in (("control", "noemb"), ("control", "emb"), ("a", "noemb"), ("a", "emb"), ("b", "emb")):
        p = D / f"scores_{arm}_{fam}_pspec.npy"
        if p.exists():
            ev[f"{arm}/{fam}"] = ep.evaluate(np.load(p), R)
    ev["b/noemb"] = ev.get("a/noemb")
    out: dict[str, Any] = {"ndcg_at_10": {k: v["macro"]["ndcg_at_10"] for k, v in ev.items() if v}, "contrasts": {}}
    from uol.facts_to_law import stats
    pc = {k: v["per_component"]["ndcg_at_10"] for k, v in ev.items() if v}
    for arm in ("control", "a", "b"):
        if f"{arm}/emb" in pc:
            c = stats.contrast(pc[f"{arm}/emb"], pc[f"{arm}/noemb"])
            out["contrasts"][f"gain {arm} (emb - noemb)"] = {"effect": c["effect"], "ci95": c["ci95"]}
    for arm in ("a", "b"):
        for fam in ("noemb", "emb"):
            if f"{arm}/{fam}" in pc and not (arm == "b" and fam == "noemb"):
                c = stats.contrast(pc[f"{arm}/{fam}"], pc[f"control/{fam}"])
                out["contrasts"][f"{arm} - control ({fam})"] = {"effect": c["effect"], "ci95": c["ci95"]}
        if f"{arm}/emb" in pc:
            ga = pc[f"{arm}/emb"] - pc[f"{arm}/noemb"]
            g0 = pc["control/emb"] - pc["control/noemb"]
            c = stats.contrast(ga, g0)
            out["contrasts"][f"gain {arm} - gain control"] = {"effect": c["effect"], "ci95": c["ci95"]}
    gc.write_json(D / "contrasts.json", out)
    for k, v in out["contrasts"].items():
        log(f"{k}: {gc.fmt(v)}")
    log(f"nDCG@10: {json.dumps(out['ndcg_at_10'])}")
    return 0




def diag() -> int:
    """One-fold diagnostic: training-pair counts, NaN/inf shares and one timed HGB fit (control arm)."""
    ctx = base.Ctx()
    F = build_features(ctx, "control")
    v = fl.make_variants(F)["noemb"]
    g = dev_groups(ctx)[0]
    rng = np.random.default_rng(fl.SEED + 1000 * g.idx + 999)
    idx, y = fl.hgb_training_pairs(ctx, F, v, fl.train_rows_of(ctx, g), rng)
    Xt = np.asarray(F.X[idx[:, None], np.asarray(v.hgb_cols)[None, :]])
    log(f"diag: train rows {len(fl.train_rows_of(ctx, g))}, pairs {len(idx)}, positives {int(y.sum())}, cols {Xt.shape[1]}, "
        f"nan share {np.isnan(Xt).mean():.4f}, inf {int(np.isinf(Xt).sum())}, abs max {np.nanmax(np.abs(Xt)):.3g}")
    per_col = {F.cols[c]: (float(np.isnan(Xt[:, j]).mean()), float(np.nanmin(Xt[:, j])), float(np.nanmax(Xt[:, j])))
               for j, c in enumerate(v.hgb_cols)}
    gc.write_json(D / "diag_cols.json", per_col)
    from sklearn.ensemble import HistGradientBoostingClassifier
    for it in (10,):
        t = time.time()
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_depth=6, random_state=0, **{**fl.HGB_FIXED, "max_iter": it})
        m.fit(Xt, y)
        log(f"diag: HGB {it} iterations in {time.time() - t:.1f}s")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "refit":
        sys.exit(refit(sys.argv[2:]))
    sys.exit({"dense": dense, "sources": sources, "contrasts": contrasts, "diag": diag}[cmd]())
