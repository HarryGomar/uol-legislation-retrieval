"""Round 3 arm S3: degree-corrected, time-respecting co-citation (PLAN §3 S3; PREREG_DRAFT S3). Dev only.

Bipartite source matrix ("units" × articles). A unit is either
* a census tesis from round-2 B ``memory.jsonl`` (84,459 pre-2020-08-14 tesis with resolved article citations; v11
  sources and tesis sharing a v11 proceeding are already excluded upstream; any record id equal to a release query is
  dropped here as well), or
* a dev component (union of its rows' full positives, as ``a_p12.unit_edges``), dated at the **latest** publication date
  of its rows (proposed DECISION: the component's citations are only treated as known once all its rows are public).

Nesting (a_p12 rule). For outer group k, fold-k scoring rows use dev components of the folds other than k; a fold-j
training row uses the folds other than j and k. Time rule: every query only sees units dated strictly before its own
publication date. Lockbox rows never enter a unit, a seed or a column.

Association of a seed article s with a candidate c over the units visible to the query:
* ``count``: obs(s, c), the number of units citing both;
* ``pmi``: log(obs·T / (n_s n_c)) when obs > 0, else 0 (n = units citing the article, T = units);
* ``resz``: Pearson residual (obs − E)/√E of the configuration model on the projection, E = k_s k_c / 2m, with
  k = Σ_{units ∋ a} (|u| − 1) and 2m = Σ_u |u|(|u| − 1); 0 when E = 0.
Seeds are the query's out-of-fold LTR top 5 or 10 (``features.seed_scores``). Aggregation over the seeds (a candidate is
never its own seed): ``max`` or ``wsum`` (weights = softmax of the seeds' LTR scores). One column; NaN when the
candidate has no visible citation or no seed has one.

Null: curveball randomisation of the bipartite matrix (row and column totals kept), stratified by source block
(census or dev fold) × publication year, 100 draws.

    python -m uol.round3.s3_cocitation run [--nulls 100]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from uol.round3 import common as C

CENSUS = C.ROUND2 / "B" / "memory.jsonl"
CENSUS_CUTOFF = "2020-08-14"
ARM = "S3"
TOP_SEEDS = 10
BATCH = 128
CURVEBALL_FACTOR = 5


def day(date: str) -> int:
    return int(np.datetime64(date[:10], "D").astype(np.int64))


# ----------------------------------------------------------------------------- units (bipartite rows)

class Units:
    """Date-sorted bipartite matrix: ``indptr``/``indices`` (CSR over articles), ``date`` (days), ``block``
    (-1 census, f = dev fold), ``year``, ``comp`` (component id for dev units, '' for census)."""

    def __init__(self, indptr, indices, date, block, year, comp, rid):
        self.indptr, self.indices = np.asarray(indptr, np.int64), np.asarray(indices, np.int32)
        self.date, self.block, self.year = np.asarray(date, np.int64), np.asarray(block, np.int8), np.asarray(year, np.int16)
        self.comp, self.rid = np.asarray(comp, dtype=object), np.asarray(rid, dtype=object)

    @property
    def n(self) -> int:
        return len(self.date)

    def sizes(self) -> np.ndarray:
        return np.diff(self.indptr)

    def with_indices(self, indices: np.ndarray) -> "Units":
        return Units(self.indptr, indices, self.date, self.block, self.year, self.comp, self.rid)

    def select(self, mask: np.ndarray) -> "Units":
        idx = np.flatnonzero(mask)
        sz = self.sizes()[idx]
        indptr = np.r_[0, np.cumsum(sz)]
        parts = [self.indices[self.indptr[i]:self.indptr[i + 1]] for i in idx]
        ind = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        return Units(indptr, ind, self.date[idx], self.block[idx], self.year[idx], self.comp[idx], self.rid[idx])

    def csr(self, n_art: int) -> sp.csr_matrix:
        return sp.csr_matrix((np.ones(len(self.indices), np.float32), self.indices, self.indptr), shape=(self.n, n_art))


@lru_cache(maxsize=1)
def census_records() -> list[dict]:
    d = C.release()
    query_rids = {str(q.record_id) for q in d.queries}
    out = []
    with CENSUS.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if str(r["record_id"]) in query_rids:
                continue                                   # never a release query (dev, lockbox or other)
            cited = sorted(set(int(a) for a in r["cited"]))
            if not cited or not r.get("date"):
                continue
            out.append({"rid": str(r["record_id"]), "date": r["date"][:10], "cited": cited})
    return out


@lru_cache(maxsize=1)
def all_units() -> Units:
    """Census units plus one unit per dev component (lockbox rows excluded), sorted by date (stable)."""
    d = C.release()
    pdates = C.publication_dates()
    dev = C.dev_rows()
    fold = C.fold_of(dev)
    comps: dict[str, dict] = {}
    for r, f in zip(dev, fold):
        q = d.queries[int(r)]
        rec = comps.setdefault(q.component_id, {"fold": int(f), "arts": set(), "date": "0000"})
        assert rec["fold"] == int(f)
        rec["arts"].update(int(a) for a in d.positive_idx[int(r)])
        rec["date"] = max(rec["date"], pdates[int(r)])
    rows = [(c["date"], -1, c["cited"], "", c["rid"]) for c in census_records()]
    rows += [(v["date"], v["fold"], sorted(v["arts"]), cid, "") for cid, v in comps.items() if v["arts"]]
    rows.sort(key=lambda x: (x[0], x[1], x[3], x[4]))
    sizes = np.asarray([len(x[2]) for x in rows])
    indptr = np.r_[0, np.cumsum(sizes)]
    indices = np.concatenate([np.asarray(x[2], np.int32) for x in rows])
    return Units(indptr, indices, [day(x[0]) for x in rows], [x[1] for x in rows], [int(x[0][:4]) for x in rows],
                 [x[3] for x in rows], [x[4] for x in rows])


def source_mask(u: Units, source: str, excluded: tuple[int, ...]) -> np.ndarray:
    """Units admissible for rows whose own fold and outer fold are ``excluded``."""
    dev_ok = (u.block >= 0) & ~np.isin(u.block, list(excluded))
    if source == "folds":
        return dev_ok
    if source == "census_folds":
        return dev_ok | (u.block == -1)
    raise ValueError(source)


def excluded_for(k: int, j: int) -> tuple[int, ...]:
    return (k,) if j == k else tuple(sorted((j, k)))


# ----------------------------------------------------------------------------- curveball null

def curveball(u: Units, seed: int, factor: int = CURVEBALL_FACTOR) -> Units:
    """Curveball trades within (block, year) strata: every unit keeps its size, every article keeps its count per
    stratum (so in every source subset), and dates stay with units."""
    rng = np.random.default_rng(seed)
    sets = [set(u.indices[u.indptr[i]:u.indptr[i + 1]].tolist()) for i in range(u.n)]
    key = u.block.astype(np.int64) * 10000 + u.year
    for s in np.unique(key):
        members = np.flatnonzero(key == s)
        m = len(members)
        if m < 2:
            continue
        n_trade = factor * m
        a_i = members[rng.integers(0, m, n_trade)]
        b_i = members[rng.integers(0, m, n_trade)]
        for a, b in zip(a_i.tolist(), b_i.tolist()):
            if a == b:
                continue
            A, B = sets[a], sets[b]
            a_only = A - B
            if not a_only:
                continue
            b_only = B - A
            if not b_only:
                continue
            pool = list(a_only | b_only)
            rng.shuffle(pool)
            na = len(a_only)
            new_a, new_b = pool[:na], pool[na:]
            A.difference_update(a_only)
            A.update(new_a)
            B.difference_update(b_only)
            B.update(new_b)
    indices = np.concatenate([np.asarray(sorted(s), np.int32) for s in sets])
    return u.with_indices(indices)


_NULL_CACHE: dict = {}


def null_units(draw: int) -> Units:
    if _NULL_CACHE.get("draw") != draw:
        _NULL_CACHE.clear()
        _NULL_CACHE["draw"] = draw
        _NULL_CACHE["units"] = curveball(all_units(), C.SEED + 104729 * (draw % 1000) + 17)
    return _NULL_CACHE["units"]


# ----------------------------------------------------------------------------- co-citation statistics per row

def query_days(rows) -> np.ndarray:
    pd = C.publication_dates()
    return np.asarray([day(pd[int(r)]) for r in rows], dtype=np.int64)


def degree_snapshots(u: Units, n_art: int, cuts: np.ndarray):
    """For each distinct cut (number of visible units, date-sorted), yield (cut, n_vec, k_vec, T, two_m)."""
    sizes = u.sizes()
    n_vec = np.zeros(n_art, np.float64)
    k_vec = np.zeros(n_art, np.float64)
    T = 0
    two_m = 0.0
    pos = 0
    for cut in np.unique(cuts):
        if cut > pos:
            a, b = u.indptr[pos], u.indptr[cut]
            ind = u.indices[a:b]
            w = np.repeat(sizes[pos:cut] - 1, sizes[pos:cut]).astype(np.float64)
            n_vec += np.bincount(ind, minlength=n_art)
            k_vec += np.bincount(ind, weights=w, minlength=n_art)
            T += int(cut - pos)
            two_m += float((sizes[pos:cut] * (sizes[pos:cut] - 1)).sum())
            pos = int(cut)
        yield int(cut), n_vec, k_vec, T, two_m


def cooc_rows(u: Units, n_art: int, rows: np.ndarray, cuts: np.ndarray, seeds: list[np.ndarray],
              pools: list[np.ndarray]):
    """Yield (i, obs [len(seeds[i]), len(pools[i])]) with obs = units (among the first cuts[i]) citing seed and candidate."""
    B = u.csr(n_art)
    csc = B.tocsc()
    csc.sort_indices()
    for s0 in range(0, len(rows), BATCH):
        idxs = range(s0, min(len(rows), s0 + BATCH))
        parts, rptr = [], [0]
        for i in idxs:
            for s in seeds[i]:
                col = csc.indices[csc.indptr[s]:csc.indptr[s + 1]]
                col = col[:np.searchsorted(col, cuts[i], side="left")]
                parts.append(col)
                rptr.append(rptr[-1] + len(col))
        ind = np.concatenate(parts) if parts else np.zeros(0, np.int32)
        A = sp.csr_matrix((np.ones(len(ind), np.float32), ind, np.asarray(rptr)), shape=(len(rptr) - 1, u.n))
        Cm = (A @ B).tocsr()
        r0 = 0
        for i in idxs:
            ns = len(seeds[i])
            yield i, np.asarray(Cm[r0:r0 + ns][:, pools[i]].toarray(), dtype=np.float32)
            r0 += ns


def row_stats(u: Units, n_art: int, rows: np.ndarray, seeds: list[np.ndarray], pools: list[np.ndarray]) -> dict:
    """Per row: obs [ns, P], n_s, n_c, k_s, k_c, T, two_m, all over the units dated strictly before the query."""
    cuts = np.searchsorted(u.date, query_days(rows), side="left")
    out = {"obs": [None] * len(rows), "n_s": [None] * len(rows), "n_c": [None] * len(rows), "k_s": [None] * len(rows),
           "k_c": [None] * len(rows), "T": np.zeros(len(rows)), "two_m": np.zeros(len(rows)), "cut": cuts}
    order = np.argsort(cuts, kind="stable")
    snaps = degree_snapshots(u, n_art, cuts[order])
    cur = next(snaps)
    for i in order:
        while cur[0] != cuts[i]:
            cur = next(snaps)
        _, n_vec, k_vec, T, two_m = cur
        out["n_s"][i], out["n_c"][i] = n_vec[seeds[i]].astype(np.float32), n_vec[pools[i]].astype(np.float32)
        out["k_s"][i], out["k_c"][i] = k_vec[seeds[i]].astype(np.float32), k_vec[pools[i]].astype(np.float32)
        out["T"][i], out["two_m"][i] = T, two_m
    for i, obs in cooc_rows(u, n_art, rows, cuts, seeds, pools):
        out["obs"][i] = obs
    return out


# ----------------------------------------------------------------------------- association and aggregation

def association(obs, n_s, n_c, k_s, k_c, T, two_m, kind: str) -> np.ndarray:
    obs = np.asarray(obs, np.float64)
    if kind == "count":
        return obs
    if kind == "pmi":
        den = np.outer(n_s, n_c)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log(obs * T / den)
        return np.where((obs > 0) & (den > 0), v, 0.0)
    if kind == "resz":
        E = np.outer(k_s, k_c) / two_m if two_m > 0 else np.zeros_like(obs)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (obs - E) / np.sqrt(E)
        return np.where(E > 0, v, 0.0)
    raise ValueError(kind)


def aggregate(vals: np.ndarray, seed_ids: np.ndarray, seed_sc: np.ndarray, pool: np.ndarray, agg: str) -> np.ndarray:
    """[P] aggregation over seeds; a candidate's own seed row is excluded."""
    self_mask = seed_ids[:, None] == pool[None, :]
    if agg == "max":
        v = np.where(self_mask, -np.inf, vals)
        out = v.max(axis=0)
        return np.where(np.isfinite(out), out, np.nan)
    if agg == "wsum":
        w = np.exp(seed_sc - seed_sc.max())
        w = w / w.sum()
        return (np.where(self_mask, 0.0, vals) * w[:, None]).sum(axis=0)
    raise ValueError(agg)


def config_column(st: dict, i: int, n_seeds: int, assoc: str, agg: str, seed_ids: np.ndarray, seed_sc: np.ndarray,
                  pool: np.ndarray) -> np.ndarray:
    ns = n_seeds
    vals = association(st["obs"][i][:ns], st["n_s"][i][:ns], st["n_c"][i], st["k_s"][i][:ns], st["k_c"][i],
                       st["T"][i], st["two_m"][i], assoc)
    col = aggregate(vals, seed_ids[:ns], seed_sc[:ns], pool, agg)
    undefined = (st["n_c"][i] == 0) | (st["n_s"][i][:ns].max() == 0)
    return np.where(undefined, np.nan, col).astype(np.float32)


# ----------------------------------------------------------------------------- columns_for (arm_runner contract)

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


CONFIGS = [{"source": s, "assoc": a, "seeds": n, "agg": g}
           for s in ("folds", "census_folds") for a in ("count", "pmi", "resz") for n in (5, 10) for g in ("max", "wsum")]

_CACHE: dict = {}


def seeds_for_group(k: int, variant: str) -> dict:
    """{row: (pool, seed scores over the pool)} for every dev row, nested for outer group k (cached per k)."""
    from uol.round3 import features as FT
    key = ("seeds", k, variant)
    if _CACHE.get("seed_key") != key:
        _CACHE.pop("seeds", None)
        _CACHE["seeds"] = FT.seed_scores(k, C.dev_rows(), variant)
        _CACHE["seed_key"] = key
    return _CACHE["seeds"]


def group_stats(k: int, variant: str, source: str, draw: int | None) -> dict:
    """{fold j: (rows, seeds, seed scores, pools, stats)} for outer group k (cached per (k, variant, source, draw))."""
    key = (k, variant, source, draw)
    store = _CACHE.setdefault("stats", {})
    if key in store:
        return store[key]
    for old in [x for x in store if x[0] != k or x[1] != variant or x[3] != draw]:
        del store[old]
    seeds = seeds_for_group(k, variant)
    u = all_units() if draw is None else null_units(draw)
    n_art = len(C.release().candidate_ids)
    dev = C.dev_rows()
    fold = C.fold_of(dev)
    res = {}
    for j in range(5):
        rows = dev[fold == j]
        sub = u.select(source_mask(u, source, excluded_for(k, j)))
        pools, sids, ssc = [], [], []
        for r in rows:
            pc, sc = seeds[int(r)]
            o = np.argsort(-np.asarray(sc, np.float64), kind="stable")[:TOP_SEEDS]
            pools.append(np.asarray(pc, np.int64))
            sids.append(np.asarray(pc[o], np.int64))
            ssc.append(np.asarray(sc[o], np.float64))
        res[j] = (rows, sids, ssc, pools, row_stats(sub, n_art, rows, sids, pools))
    store[key] = res
    return res


def columns_for(cfg: dict, g, h, draw=None) -> np.ndarray:
    k = int(g.role.replace("dev_fold", ""))
    if draw is not None and cfg.get("_null") != "curveball":
        raise ValueError(f"unknown null {cfg.get('_null')}")
    gs = group_stats(k, h.variant, cfg["source"], draw)
    cols = h.empty_columns(1)
    for j, (rows, sids, ssc, pools, st) in gs.items():
        for i, r in enumerate(rows):
            sl, pc = h.pair_slice(int(r))
            col = config_column(st, i, cfg["seeds"], cfg["assoc"], cfg["agg"], sids[i], ssc[i], pools[i])
            cols[sl, 0] = align(pools[i], col, np.asarray(pc, np.int64))
    return cols


# ----------------------------------------------------------------------------- subgroup, standalone, run

@lru_cache(maxsize=1)
def census_counts() -> np.ndarray:
    n = len(C.release().candidate_ids)
    c = np.zeros(n, np.int64)
    for r in census_records():
        c[r["cited"]] += 1
    return c


def mid_frequency(rows) -> np.ndarray:
    """Pre-registered subgroup (proposed DECISION): >= 1 specific gold article with 5-50 census citations."""
    cc = census_counts()
    return np.asarray([any(5 <= cc[a] <= 50 for a in g) for g in C.gold_rows(rows, "spec")])


def standalone(configs, variant: str = "emb") -> dict:
    """Arm alone: each config's column as the ranking score over the pool (fold-k rows from outer group k), dev P_spec."""
    from uol.embedding_push import common as ep
    from uol.round3 import arm_runner
    rows = C.population_rows("P_spec", C.dev_rows())
    fold = C.fold_of(rows)
    n_art = len(C.release().candidate_ids)
    got_all = {arm_runner.cfg_key(c): {} for c in configs}
    for k in range(5):
        for cfg in configs:
            rows_k, sids, ssc, pools, st = group_stats(k, variant, cfg["source"], None)[k]
            at = {int(r): i for i, r in enumerate(rows_k)}
            M = got_all[arm_runner.cfg_key(cfg)]
            for t in np.flatnonzero(fold == k):
                i = at[int(rows[t])]
                col = config_column(st, i, cfg["seeds"], cfg["assoc"], cfg["agg"], sids[i], ssc[i], pools[i])
                M[int(t)] = (pools[i], np.where(np.isfinite(col), col, -1e9))
    out = {}
    for key, per_row in got_all.items():
        M = np.full((len(rows), n_art), -1e12, dtype=np.float32)       # finite: below every pooled candidate
        for t, (pool, sc) in per_row.items():
            M[t, pool] = sc
        out[key] = ep.evaluate(M, rows, "P_spec")["macro"]["ndcg_at_10"]
        del M
    return {"P_spec_ndcg_at_10_by_config": out,
            "definition": "column alone ranks the pooled candidates (NaN below every scored candidate; candidates "
                          "outside the pool last); seeds and sources nested exactly as in the refit columns"}


def describe() -> dict:
    u = all_units()
    cen = u.block == -1
    return {"units": int(u.n), "census_units": int(cen.sum()), "dev_component_units": int((~cen).sum()),
            "census_max_date": str(np.datetime64(int(u.date[cen].max()), "D")),
            "census_pairs": int(sum(len(r["cited"]) * (len(r["cited"]) - 1) // 2 for r in census_records())),
            "dev_units_by_fold": {int(f): int((u.block == f).sum()) for f in range(5)}}


def main(argv=None) -> int:
    from uol.round3 import arm_runner
    from uol.round3 import features as FT
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["describe", "run"])
    ap.add_argument("--nulls", type=int, default=0)
    a = ap.parse_args(argv)
    if a.command == "describe":
        print(json.dumps(describe(), indent=1))
        return 0
    t0 = time.perf_counter()
    slot = C.cpu_slot()                                 # the whole arm (feature builds and refits) holds one slot
    slot.__enter__()
    sa = standalone(CONFIGS, "emb")
    _CACHE.clear()
    notes = ("S3 time-respecting co-citation; census = round-2 B memory.jsonl (pre-2020-08-14) + dev components "
             "(nested folds, dated at the latest row date); strict date < query; curveball null stratified by "
             "source block x year. See arms/S3/NOTES.md.")
    res = arm_runner.run(ARM, CONFIGS, columns_for, n_nulls=a.nulls, null_kinds=["curveball"] if a.nulls else [],
                         subgroups={"mid_frequency_gold_5_50": mid_frequency}, standalone=sa, notes=notes,
                         extra_result={"graph": describe(), "standalone_seconds": round(time.perf_counter() - t0)})
    slot.__exit__(None, None, None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
