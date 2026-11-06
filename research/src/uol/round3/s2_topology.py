"""Round 3 arm S2: statute topology, two-pass neighbour features over the label-free graph G2 (PLAN §3 S2). Dev only.

Graph (``phase2/statute_graph``; xref_ext dropped at lift 1.0): adj1/adj2 (document order within an instrument),
sibling (same full heading path) and chapter (same path up to CAPÍTULO) as clique groups, xref_int (directed, used
undirected: outgoing and incoming). Edge sets: ``hier_adj`` (sibling, chapter, adj1, adj2), ``xref`` (xref_int),
``both``, ``both_cocit`` (both, plus two extra columns from fold co-citation).

First pass (seeds). For outer group k, ``features.seed_scores(k, ·)`` gives out-of-fold LTR scores over each row's pool
(fold-k rows: frozen out-of-fold pipeline; fold-j rows: ``nested_ltr(k, j)``, trained without folds j and k), or the
label-free dense Qwen3-8B cosine (``dense_qwen8b``). The seed set is the row's top K of the pool; scores are z-scored
within the top K.

Features per candidate c (3 columns; ``both_cocit`` adds 2):
1. max z-score of a top-K neighbour of c (c itself excluded; NaN without one);
2. number of the top-10 seeds that are neighbours of c;
3. chapter softmax mass: Σ softmax(z) over the top K inside c's chapter, c itself excluded (NaN without a chapter);
4-5. the same as 1-2 with fold co-citation neighbours (dev components of the folds other than the row's own and
   the outer fold, dated strictly before the query; ``s3_cocitation`` machinery).

Nulls (≥ 50 draws each): ``chapter_perm`` (heading paths permuted among the located articles of each instrument, so
sibling and chapter group sizes are kept), ``order_perm`` (document order permuted within each instrument, adj1/adj2
rebuilt), ``xref_rewire`` (degree-preserving directed target swaps of xref_int within each instrument, no self-loops or
duplicates, 10 × |E| attempts), ``popularity`` (training-fold citation counts, nested like the seeds, replace the seed
score).

    python -m uol.round3.s2_topology run [--nulls 50] [--kinds chapter_perm,order_perm,xref_rewire,popularity]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache

import numpy as np

from uol.round3 import common as C
from uol.round3 import s3_cocitation as S3
from uol.round3 import statute_graph as SG

ARM = "S2"
KEPT = ("adj1", "adj2", "sibling", "chapter", "xref_int")
EDGE_SETS = {"hier_adj": ("adj1", "adj2", "sibling", "chapter"), "xref": ("xref_int",),
             "both": ("adj1", "adj2", "sibling", "chapter", "xref_int"),
             "both_cocit": ("adj1", "adj2", "sibling", "chapter", "xref_int")}
NULL_KINDS = ("chapter_perm", "order_perm", "xref_rewire", "popularity")
GROUP_MAX = 400                                        # statute_graph.build clique rule
REWIRE_FACTOR = 10
DENSE = "dense_qwen8b"


# ----------------------------------------------------------------------------- graph structures

@lru_cache(maxsize=1)
def base_graph() -> dict:
    g = SG.load_graph()
    paths = json.loads((SG.OUT / "paths.json").read_text(encoding="utf-8"))
    n = len(g["inst"])
    path_of = [None] * n
    for i, p in paths.items():
        path_of[int(i)] = tuple(p)
    return {"inst": g["inst"].astype(np.int64), "order": g["order"].astype(np.int64), "path": path_of,
            "xref_int": g["e_xref_int"].astype(np.int64), "e": {k: g[f"e_{k}"] for k in KEPT}, "n": n}


def chapter_key(p: tuple) -> tuple:
    return tuple(h for h in p if SG.fold(h).split()[0] in ("libro", "parte", "titulo", "capitulo"))


def group_ids(inst: np.ndarray, path_of: list) -> tuple[np.ndarray, np.ndarray]:
    """Per article: sibling group id and chapter group id (-1 when absent), as statute_graph.build defines them."""
    n = len(inst)
    sib, chap = np.full(n, -1, np.int64), np.full(n, -1, np.int64)
    sk, ck = {}, {}
    for i in range(n):
        p = path_of[i]
        if not p:
            continue
        sib[i] = sk.setdefault((int(inst[i]), p), len(sk))
        c = chapter_key(p)
        if c:
            chap[i] = ck.setdefault((int(inst[i]), c), len(ck))
    return sib, chap


def members(gid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ptr, member list, size per article's group) for clique groups; groups of size 1 or > GROUP_MAX are dropped."""
    ng = int(gid.max()) + 1 if (gid >= 0).any() else 0
    sz = np.bincount(gid[gid >= 0], minlength=ng)
    ok = (sz > 1) & (sz <= GROUP_MAX)
    g2 = np.where(gid >= 0, gid, -1)
    g2 = np.where((g2 >= 0) & ok[np.maximum(g2, 0)], g2, -1)
    order = np.argsort(g2, kind="stable")
    order = order[g2[order] >= 0]
    ptr = np.r_[0, np.cumsum(np.bincount(g2[order], minlength=ng))]
    return ptr, order, g2


def adjacency_edges(inst: np.ndarray, order: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """adj1 and adj2 edge arrays from document order (statute_graph.build rule)."""
    loc = np.flatnonzero(order >= 0)
    srt = loc[np.lexsort((order[loc], inst[loc]))]
    same1 = inst[srt[1:]] == inst[srt[:-1]]
    same2 = inst[srt[2:]] == inst[srt[:-2]]
    return (np.stack([srt[:-1][same1], srt[1:][same1]], 1), np.stack([srt[:-2][same2], srt[2:][same2]], 1))


def csr_undirected(edges: list[np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    e = np.concatenate([x.reshape(-1, 2) for x in edges]) if edges else np.zeros((0, 2), np.int64)
    a = np.r_[e[:, 0], e[:, 1]]
    b = np.r_[e[:, 1], e[:, 0]]
    key = np.unique(a * n + b)
    a, b = key // n, key % n
    ptr = np.r_[0, np.cumsum(np.bincount(a, minlength=n))]
    return ptr, b


def rewire_xref(e: np.ndarray, inst: np.ndarray, seed: int, factor: int = REWIRE_FACTOR) -> np.ndarray:
    """Directed target swaps within each instrument: (u→v, x→y) → (u→y, x→v); out- and in-degrees are kept."""
    rng = np.random.default_rng(seed)
    e = e.copy()
    present = set(map(tuple, e.tolist()))
    by_inst: dict[int, list[int]] = {}
    for t, u in enumerate(e[:, 0]):
        by_inst.setdefault(int(inst[u]), []).append(t)
    for ids in by_inst.values():
        m = len(ids)
        if m < 2:
            continue
        ids = np.asarray(ids)
        pa = ids[rng.integers(0, m, factor * m)]
        pb = ids[rng.integers(0, m, factor * m)]
        for a, b in zip(pa.tolist(), pb.tolist()):
            if a == b:
                continue
            u, v = int(e[a, 0]), int(e[a, 1])
            x, y = int(e[b, 0]), int(e[b, 1])
            if u == y or x == v or (u, y) in present or (x, v) in present:
                continue
            present.discard((u, v))
            present.discard((x, y))
            present.add((u, y))
            present.add((x, v))
            e[a, 1], e[b, 1] = y, v
    return e


class Topo:
    """Neighbour structures for one (possibly null) graph."""

    def __init__(self, inst, order, path_of, xref, n):
        self.n = n
        sib, chap = group_ids(inst, path_of)
        self.sib_ptr, self.sib_mem, self.sib = members(sib)
        self.chap_ptr, self.chap_mem, self.chap = members(chap)
        self.chap_all = chap                                  # chapter id for the mass feature (any size)
        adj1, adj2 = adjacency_edges(inst, order)
        self.edges = {"adj1": adj1, "adj2": adj2, "xref_int": xref}
        self.n_chap = int(chap.max()) + 1 if (chap >= 0).any() else 0
        self._csr = {}
        self._nb = {}

    def explicit(self, types: tuple[str, ...]):
        key = tuple(t for t in types if t in self.edges)
        if key not in self._csr:
            self._csr[key] = csr_undirected([self.edges[t] for t in key], self.n)
        return self._csr[key]

    def neighbours(self, a: int, types: tuple[str, ...]) -> np.ndarray:
        key = (a, types)
        got = self._nb.get(key)
        if got is None:
            got = self._nb[key] = self._neighbours(a, types)
        return got

    def _neighbours(self, a: int, types: tuple[str, ...]) -> np.ndarray:
        parts = []
        ptr, ind = self.explicit(types)
        parts.append(ind[ptr[a]:ptr[a + 1]])
        if "sibling" in types and self.sib[a] >= 0:
            s = self.sib[a]
            parts.append(self.sib_mem[self.sib_ptr[s]:self.sib_ptr[s + 1]])
        if "chapter" in types and self.chap[a] >= 0:
            s = self.chap[a]
            parts.append(self.chap_mem[self.chap_ptr[s]:self.chap_ptr[s + 1]])
        x = np.unique(np.concatenate(parts))
        return x[x != a]


@lru_cache(maxsize=1)
def real_topo() -> Topo:
    b = base_graph()
    return Topo(b["inst"], b["order"], b["path"], b["xref_int"], b["n"])


_NULL_TOPO: dict = {}


def null_topo(kind: str, d: int) -> Topo:
    key = (kind, d)
    if _NULL_TOPO.get("key") == key:
        return _NULL_TOPO["topo"]
    b = base_graph()
    rng = np.random.default_rng(C.SEED + 7919 * d + 31 * NULL_KINDS.index(kind))
    inst, order, path_of, xref = b["inst"], b["order"], list(b["path"]), b["xref_int"]
    if kind == "chapter_perm":
        for k in np.unique(inst):
            idx = [i for i in np.flatnonzero(inst == k) if path_of[i]]
            perm = rng.permutation(len(idx))
            vals = [path_of[i] for i in idx]
            for t, i in enumerate(idx):
                path_of[i] = vals[perm[t]]
    elif kind == "order_perm":
        order = order.copy()
        for k in np.unique(inst):
            idx = np.flatnonzero((inst == k) & (order >= 0))
            order[idx] = order[idx][rng.permutation(len(idx))]
    elif kind == "xref_rewire":
        xref = rewire_xref(xref, inst, int(rng.integers(0, 2**31)))
    else:
        raise ValueError(kind)
    t = Topo(inst, order, path_of, xref, b["n"])
    _NULL_TOPO.clear()
    _NULL_TOPO.update(key=key, topo=t)
    return t


# ----------------------------------------------------------------------------- per-row features

def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def row_features(topo: Topo, types: tuple[str, ...], pool: np.ndarray, seeds: np.ndarray, z: np.ndarray,
                 buf: np.ndarray, cnt: np.ndarray) -> np.ndarray:
    """[P, 3] for one row. ``seeds`` are the top-K articles sorted by score (desc), ``z`` their z-scores.

    ``buf`` (float64, NaN-filled) and ``cnt`` (int32, zero) are length-N scratch arrays returned clean."""
    out = np.full((len(pool), 3), np.nan, dtype=np.float32)
    # 1. max neighbour z: seeds in descending order, the first assignment wins
    touched = []
    for s, zs in zip(seeds.tolist(), z.tolist()):
        nb = topo.neighbours(s, types)
        nb = nb[np.isnan(buf[nb])]
        buf[nb] = zs
        touched.append(nb)
    out[:, 0] = buf[pool]
    for nb in touched:
        buf[nb] = np.nan
    # 2. number of top-10 seeds that neighbour the candidate
    touched = []
    for s in seeds[:10].tolist():
        nb = topo.neighbours(s, types)
        cnt[nb] += 1
        touched.append(nb)
    out[:, 1] = cnt[pool]
    for nb in touched:
        cnt[nb] = 0
    # 3. chapter softmax mass, the candidate itself excluded
    p = np.exp(z - z.max())
    p /= p.sum()
    ch = topo.chap_all[seeds]
    ok = ch >= 0
    mass = np.bincount(ch[ok], weights=p[ok], minlength=topo.n_chap)
    buf[seeds] = p
    own = np.nan_to_num(buf[pool], nan=0.0)
    buf[seeds] = np.nan
    pc_ch = topo.chap_all[pool]
    out[:, 2] = np.where(pc_ch >= 0, mass[np.maximum(pc_ch, 0)] - own, np.nan)
    return out


def align(pool: np.ndarray, values: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-pool values reordered to the pair-table candidates ``target`` of the same row.

    The seed pool can be a subset of the pair-table row (the NO-EMB pipeline pools fewer sources); target candidates
    outside the pool get NaN (they are not among that pipeline's pairs). Every pool candidate must be in the row."""
    if np.array_equal(pool, target):
        return values
    sorter = np.argsort(target, kind="stable")
    at = np.searchsorted(target, pool, sorter=sorter)
    pos = sorter[np.minimum(at, len(target) - 1)]
    if not np.array_equal(target[pos], pool):
        raise ValueError("seed pool candidate missing from the pair-table row")
    out = np.full((len(target),) + values.shape[1:], np.nan, dtype=values.dtype)
    out[pos] = values
    return out


def cocit_features(st: dict, i: int, pool: np.ndarray, seeds: np.ndarray, z: np.ndarray) -> np.ndarray:
    """[P, 2]: max z of a co-cited top-K seed; number of co-cited top-10 seeds (obs > 0, self excluded)."""
    obs = st["obs"][i] > 0
    obs &= seeds[:, None] != pool[None, :]
    zz = np.where(obs, z[:, None], -np.inf).max(axis=0)
    out = np.empty((len(pool), 2), np.float32)
    out[:, 0] = np.where(np.isfinite(zz), zz, np.nan)
    out[:, 1] = obs[:10].sum(axis=0)
    return out


# ----------------------------------------------------------------------------- seeds

_CACHE: dict = {}


def seed_table(k: int, variant: str, seed: str) -> dict:
    """{row: (pool, seed score over the pool)} for every dev row, for outer group k."""
    from uol.round3 import features as FT
    key = (k, variant, seed)
    st = _CACHE.setdefault("seeds", {})
    if key in st:
        return st[key]
    for old in [x for x in st if x[0] != k or x[1] != variant]:
        del st[old]
    dev = C.dev_rows()
    if seed == "ltr":
        st[key] = FT.seed_scores(k, dev, variant)
    elif seed == "dense":
        ctx, F = FT.context()
        M = FT.ep.open_scores(DENSE)
        out = {}
        for r in dev:
            pc = F.pc[int(F.row_ptr[r]):int(F.row_ptr[r + 1])]
            out[int(r)] = (pc, np.asarray(M[int(r)], np.float32)[pc])
        st[key] = out
    elif seed == "popularity":
        st[key] = popularity_seeds(k)
    else:
        raise ValueError(seed)
    return st[key]


def popularity_seeds(k: int) -> dict:
    """Null: log1p of distinct training-component citation counts, nested exactly like the LTR seeds."""
    from uol.facts_to_law import prior as prior_mod
    from uol.round3 import features as FT
    ctx, F = FT.context()
    d = C.release()
    dev = C.dev_rows()
    fold = C.fold_of(dev)
    out = {}
    for j in range(5):
        fit_folds = [f for f in range(5) if f not in {j, k}]
        ids = [d.queries[int(r)].query_id for r in dev[np.isin(fold, fit_folds)]]
        cnt = prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)
        pop = np.log1p(cnt).astype(np.float32)
        for r in dev[fold == j]:
            pc = F.pc[int(F.row_ptr[r]):int(F.row_ptr[r + 1])]
            # tiny deterministic tie-break by catalogue index so the top K is well defined
            out[int(r)] = (pc, pop[pc] - 1e-7 * (pc / len(d.candidate_ids)).astype(np.float32))
    return out


def top_k(pool: np.ndarray, score: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(-np.asarray(score, np.float64), kind="stable")[:K]
    return np.asarray(pool[o], np.int64), zscore(np.asarray(score, np.float64)[o])


def cocit_stats(k: int, variant: str, seed: str, K: int, folds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> dict:
    """{row: [P, 2] fold co-citation features} for every dev row (pool order of the seed table), cached per key.

    Units: dev components of the folds other than the row's own and k, dated strictly before the query."""
    key = (k, variant, seed, K, folds)
    cs = _CACHE.setdefault("cocit", {})
    if key in cs:
        return cs[key]
    for old in [x for x in cs if x[0] != k or x[1] != variant]:
        del cs[old]
    seeds = seed_table(k, variant, seed)
    u = S3.all_units()
    n_art = len(C.release().candidate_ids)
    dev = C.dev_rows()
    fold = C.fold_of(dev)
    res = {}
    for j in folds:
        rows = dev[fold == j]
        pools, sids, zs = [], [], []
        for r in rows:
            pc, sc = seeds[int(r)]
            s, z = top_k(np.asarray(pc, np.int64), sc, K)
            pools.append(np.asarray(pc, np.int64))
            sids.append(s)
            zs.append(z)
        sub = u.select(S3.source_mask(u, "folds", S3.excluded_for(k, j)))
        cuts = np.searchsorted(sub.date, S3.query_days(rows), side="left")
        for i, obs in S3.cooc_rows(sub, n_art, rows, cuts, sids, pools):
            res[int(rows[i])] = cocit_features({"obs": [obs]}, 0, pools[i], sids[i], zs[i])
    cs[key] = res
    return res


# ----------------------------------------------------------------------------- columns_for (arm_runner contract)

CONFIGS = [{"K": K, "seed": s, "edges": e} for K in (20, 50, 100) for s in ("ltr", "dense")
           for e in ("hier_adj", "xref", "both", "both_cocit")]


def columns_for(cfg: dict, g, h, draw=None) -> np.ndarray:
    k = int(g.role.replace("dev_fold", ""))
    kind = cfg.get("_null")
    topo = real_topo()
    seed = cfg["seed"]
    if draw is not None:
        d = int(draw) % 1000
        if kind == "popularity":
            seed = "popularity"
        else:
            topo = null_topo(kind, d)
    return compute_columns(k, h.variant, cfg, topo, seed, h)


def compute_columns(k: int, variant: str, cfg: dict, topo: Topo, seed: str, h=None, rows_only=None) -> np.ndarray:
    types = EDGE_SETS[cfg["edges"]]
    cocit = cfg["edges"] == "both_cocit"
    width = 5 if cocit else 3
    seeds = seed_table(k, variant, seed)
    only = tuple(sorted(set(C.fold_of(rows_only).tolist()))) if rows_only is not None else (0, 1, 2, 3, 4)
    cs = cocit_stats(k, variant, seed, cfg["K"], only) if cocit else None
    n = topo.n
    buf = np.full(n, np.nan)
    cnt = np.zeros(n, np.int32)
    dev = C.dev_rows() if rows_only is None else np.asarray(rows_only)
    if h is not None:
        cols = h.empty_columns(width)
    else:
        cols = {}
    for r in dev:
        pc, sc = seeds[int(r)]
        pool = np.asarray(pc, np.int64)
        s, z = top_k(pool, sc, cfg["K"])
        f = row_features(topo, types, pool, s, z, buf, cnt)
        if cocit:
            f = np.hstack([f, cs[int(r)]])
        if h is not None:
            sl, hpc = h.pair_slice(int(r))
            cols[sl] = align(pool, f, np.asarray(hpc, np.int64))
        else:
            cols[int(r)] = (pool, f)
    return cols


# ----------------------------------------------------------------------------- subgroups, standalone

def _union_neighbours(topo: Topo, a: int) -> np.ndarray:
    return topo.neighbours(a, KEPT)


@lru_cache(maxsize=1)
def _ltr_ranks() -> dict:
    from uol.round3 import features as FT
    M = FT.frozen_matrix("LTR_emb_hgb")
    rows = C.population_rows("P_spec", C.dev_rows())
    return {int(r): np.argsort(-np.asarray(M[int(r)], np.float64), kind="stable")[:50] for r in rows}


def sg_two_gold_one_chapter(rows) -> np.ndarray:
    ch = real_topo().chap_all
    out = []
    for g in C.gold_rows(rows, "spec"):
        c = ch[np.asarray(g, np.int64)]
        c = c[c >= 0]
        out.append(len(c) > len(np.unique(c)))
    return np.asarray(out)


def sg_gold_11_50_with_top10_neighbour(rows) -> np.ndarray:
    topo = real_topo()
    ranks = _ltr_ranks()
    out = []
    for r, g in zip(rows, C.gold_rows(rows, "spec")):
        top = ranks[int(r)]
        top10, mid = set(top[:10].tolist()), set(top[10:50].tolist())
        hit = any(a in mid and top10 & set(_union_neighbours(topo, int(a)).tolist()) for a in g)
        out.append(bool(hit))
    return np.asarray(out)


def _band_sg(band: str):
    def fn(rows) -> np.ndarray:
        return np.asarray([any(C.band_of_row(int(r))[a] == band for a in g) for r, g in zip(rows, C.gold_rows(rows, "spec"))])
    fn.__name__ = f"sg_{band}"
    return fn


SUBGROUPS = {"two_gold_in_one_chapter": sg_two_gold_one_chapter,
             "gold_rank11_50_with_top10_neighbour": sg_gold_11_50_with_top10_neighbour,
             "rare_band_gold": _band_sg("rare"), "unseen_band_gold": _band_sg("unseen")}


def standalone(configs, variant: str = "emb") -> dict:
    """Arm alone: column 1 (max neighbour z) ranks the pool, ties by column 2 then 3; fold-k rows of outer group k."""
    from uol.embedding_push import common as ep
    from uol.round3 import arm_runner
    rows = C.population_rows("P_spec", C.dev_rows())
    fold = C.fold_of(rows)
    n_art = len(C.release().candidate_ids)
    got_all = {arm_runner.cfg_key(c): {} for c in configs}
    for k in range(5):
        rk = rows[fold == k]
        idx = np.flatnonzero(fold == k)
        for cfg in configs:
            got = compute_columns(k, variant, cfg, real_topo(), cfg["seed"], None, rows_only=rk)
            M = got_all[arm_runner.cfg_key(cfg)]
            for t, r in zip(idx, rk):
                pool, f = got[int(r)]
                sc = np.nan_to_num(f[:, 0], nan=-50.0) + 1e-3 * f[:, 1] + 1e-6 * np.nan_to_num(f[:, 2], nan=0.0)
                M[int(t)] = (pool, sc)
    out = {}
    for key, per_row in got_all.items():
        M = np.full((len(rows), n_art), -1e12, dtype=np.float32)       # finite: below every pooled candidate
        for t, (pool, sc) in per_row.items():
            M[t, pool] = sc
        out[key] = ep.evaluate(M, rows, "P_spec")["macro"]["ndcg_at_10"]
        del M
    return {"P_spec_ndcg_at_10_by_config": out,
            "definition": "max-neighbour z ranks the pooled candidates (ties: top-10 neighbour count, then chapter "
                          "mass); seeds nested exactly as in the refit columns"}


def describe() -> dict:
    t = real_topo()
    deg = {}
    for name, types in EDGE_SETS.items():
        if name == "both_cocit":
            continue
        sample = np.arange(0, t.n, 7)
        deg[name] = float(np.mean([len(t.neighbours(int(a), types)) for a in sample]))
    return {"edges_kept": list(KEPT), "edges_npz": {k: int(len(v)) for k, v in base_graph()["e"].items()},
            "mean_undirected_degree_sampled": deg,
            "sibling_groups_used": int(len(t.sib_ptr) - 1), "chapter_groups_used": int(len(t.chap_ptr) - 1)}


def popularity_null() -> dict:
    """The popularity null is deterministic (no draw changes it), so it is refit once, not 50 times (proposed DECISION).

    Uses the per-fold selected configurations of the existing RESULT.json; reports Δ vs the baseline refit and the
    paired contrast real − popularity."""
    import gc

    from uol.round3 import arm_runner as AR
    from uol.round3 import features as FT
    res = json.loads((C.ARMS / ARM / "RESULT.json").read_text(encoding="utf-8"))
    per_fold_cfg = {int(k): {kk: v for kk, v in cfg.items()} for k, cfg in res["selected_config"].items()}
    choice = {k: AR.cfg_key(c) for k, c in per_fold_cfg.items()}
    h = FT.Harness("emb", "spec")
    ckpt = C.ARMS / ARM / "ckpt"
    base = AR._eval(h.matrix(AR.baseline(h)))
    real = AR._eval(h.matrix(AR.assemble(h, AR.refit_configs(h, CONFIGS, columns_for, ckpt, per_fold_cfg=per_fold_cfg),
                                         choice)))
    pn = AR.refit_configs(h, CONFIGS, lambda cfg, g, hh, dr: columns_for({**cfg, "_null": "popularity"}, g, hh, dr),
                          ckpt, draw=9000, per_fold_cfg=per_fold_cfg)
    null = AR._eval(h.matrix(AR.assemble(h, pn, choice)))
    b = base["P_spec"]["macro"]["ndcg_at_10"]
    out = {"n": 1, "deterministic": True, "delta": null["P_spec"]["macro"]["ndcg_at_10"] - b,
           "real_delta": real["P_spec"]["macro"]["ndcg_at_10"] - b,
           "real_minus_null": FT.dev_contrasts(real, null)["P_spec"]}
    out["real_beats_null"] = out["real_delta"] > out["delta"]
    del h
    _CACHE.clear()
    gc.collect()
    return out


def auto_kinds() -> list[str]:
    """Null kinds that can change the per-fold selected configurations' columns (from the existing RESULT.json).

    chapter_perm always applies (chapter mass is in every edge set) and so does popularity; order_perm only when a
    selected edge set has adjacency; xref_rewire only when it has xref_int. An inapplicable null would reproduce the
    real columns exactly (identical refit), so it is not run; NOTES.md records which were skipped."""
    res = json.loads((C.ARMS / ARM / "RESULT.json").read_text(encoding="utf-8"))
    sets = {cfg["edges"] for cfg in res["selected_config"].values()}
    kinds = ["chapter_perm"]
    if sets & {"hier_adj", "both", "both_cocit"}:
        kinds.append("order_perm")
    if sets & {"xref", "both", "both_cocit"}:
        kinds.append("xref_rewire")
    kinds.append("popularity")
    return kinds


def main(argv=None) -> int:
    from uol.round3 import arm_runner
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["describe", "run", "prep_noemb"])
    ap.add_argument("--nulls", type=int, default=0)
    ap.add_argument("--kinds", default=",".join(NULL_KINDS))
    a = ap.parse_args(argv)
    if a.command == "describe":
        print(json.dumps(describe(), indent=1))
        return 0
    if a.command == "prep_noemb":
        from uol.round3 import features as FT
        for k in range(5):
            for j in range(5):
                if j != k:
                    t = time.perf_counter()
                    FT.nested_ltr(k, j, "noemb")
                    print(f"nested noemb k{k} j{j}: {time.perf_counter() - t:.0f}s", flush=True)
        return 0
    t0 = time.perf_counter()
    slot = C.cpu_slot()                                 # the whole arm (feature builds and refits) holds one slot
    slot.__enter__()
    sa = standalone(CONFIGS, "emb")
    _CACHE.clear()
    kinds = (auto_kinds() if a.kinds == "auto" else [x for x in a.kinds.split(",") if x]) if a.nulls else []
    extra = {"graph": describe(), "standalone_seconds": round(time.perf_counter() - t0)}
    if "popularity" in kinds:
        kinds = [x for x in kinds if x != "popularity"]
        extra["null_popularity"] = popularity_null()
    notes = ("S2 statute topology two-pass features over G2 (xref_ext dropped); seeds out-of-fold LTR (nested) or "
             "dense_qwen8b; fold co-citation dated before the query and nested. See arms/S2/NOTES.md.")
    arm_runner.run(ARM, CONFIGS, columns_for, n_nulls=a.nulls, null_kinds=kinds, subgroups=SUBGROUPS, standalone=sa,
                   notes=notes, extra_result=extra)
    slot.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
