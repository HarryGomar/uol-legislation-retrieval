"""M1: tuned historical citation memory (PLAN §3 M1; PREREG_DRAFT M1), feature route through ``arm_runner``.

Memory. The 84,459 round-2 B historical tesis (``experiments/runs/round2_20261004/B/memory.jsonl``: record id, date,
resolved catalogue articles). All were published before 2020-08-14, i.e. strictly before every v11 query (dev rows
start 2020-08-14); the date rule is still enforced per (query, entry). No release record (dev, lockbox or any other
split) is a memory entry; this is checked on load. Dev training-fold rows are NOT memory entries (proposed DECISION,
NOTES.md): the frozen LTR already carries ~30 training-fold kNN memory sources, so M1 tunes the *historical* memory
that round 2 measured (+.028 untuned). With no dev row in memory, the columns are label-free with respect to dev and
identical for every outer group, so the a_p12 nesting rule holds trivially.

Keys. ``dense``: cosine of the frozen Qwen3-Embedding-8B variant-c query vectors to the cached round-2 memory vectors
(rubro + texto, first 2,000 Qwen tokens); no endpoint call. ``lex``: BM25 (k1 1.2, b .75) of the masked facts text
against rubro + texto (round-2 ``b_mem_bm25``). The top ``KMAX`` = 100 neighbours per dev row are cached once.

Grid (24): k in {10, 25, 50, 100} x decay {none, 5-year, 10-year half-life} x key {dense, lex}. Neighbours are the
top-k by similarity; the decay only re-weights them: w = max(sim, 0) * 0.5 ** (age_years / half_life), normalised per
query. Columns per (query, candidate): ``vote`` = sum_n w_n * 1[a in cited_n] / |cited_n| (the ``knn_votes`` rule),
``count`` = number of the k neighbours (with w > 0) citing a, ``logrank`` = log(1 + #articles with a larger vote).

Null (``_null`` = ``shuffle_year``): the memory's citation lists are permuted among entries of the same publication
year (neighbours and weights unchanged).

    python -m uol.round3.m1_memory neighbours          # cache the top-100 neighbours (dense and lex)
    python -m uol.round3.m1_memory standalone          # arm-alone dev scores of all 24 configurations
    python -m uol.round3.m1_memory run [--nulls 50]    # arm_runner (refits); RESULT.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy import sparse

from uol.round3 import common as C

ARM = "M1"
OUT = C.ARMS / ARM
CACHE = OUT / "cache"
B_DIR = C.ROUND2 / "B"
MEMORY_JSONL = B_DIR / "memory.jsonl"
HIST_JSONL = B_DIR / "historical_tesis.jsonl"
KMAX = 100
KS = (10, 25, 50, 100)
DECAYS = {"none": None, "hl5": 5.0, "hl10": 10.0}
KEYS = ("dense", "lex")
CONFIGS = [{"key": key, "decay": dec, "k": k} for key in KEYS for dec in DECAYS for k in KS]
COLUMNS = ("vote", "count", "logrank")
NULL_SEED = C.SEED + 31


def log(msg: str) -> None:
    print(f"[{C.utc()}] {msg}", flush=True)


# ----------------------------------------------------------------------------- memory and leakage guards

@lru_cache(maxsize=1)
def memory() -> dict:
    """{'record_id', 'date' (datetime64[D]), 'year', 'cited' (list of tuples)} in memory.jsonl order."""
    rec = [json.loads(l) for l in MEMORY_JSONL.open(encoding="utf-8") if l.strip()]
    mem = {"record_id": np.asarray([str(r["record_id"]) for r in rec]),
           "date": np.asarray([r["date"] for r in rec], dtype="datetime64[D]"),
           "cited": [tuple(sorted(set(int(a) for a in r["cited"]))) for r in rec]}
    mem["year"] = mem["date"].astype("datetime64[Y]").astype(int) + 1970
    leaked = set(mem["record_id"].tolist()) & release_record_ids()
    if leaked:
        raise AssertionError(f"{len(leaked)} release records (dev/lockbox) are memory entries, e.g. {sorted(leaked)[:3]}")
    if any(not c for c in mem["cited"]):
        raise ValueError("memory entry without citations")
    return mem


def record_id_of(query_id: str) -> str:
    """'sjf:tesis:<rid>:hechos' -> '<rid>'."""
    return str(query_id).split(":")[2]


@lru_cache(maxsize=1)
def release_record_ids() -> set[str]:
    return {record_id_of(q.query_id) for q in C.release().queries}


def lockbox_record_ids() -> set[str]:
    d = C.release()
    return {record_id_of(d.queries[int(r)].query_id) for r in C.lockbox_rows()}


@lru_cache(maxsize=1)
def dev_dates() -> np.ndarray:
    """Publication date of each dev row (C.dev_rows() order) as datetime64[D]; every dev row is dated."""
    pd = C.publication_dates()
    dates = [pd[int(r)] for r in C.dev_rows()]
    if any(not x for x in dates):
        raise ValueError("undated dev row")
    return np.asarray(dates, dtype="datetime64[D]")


# ----------------------------------------------------------------------------- neighbours (computed once per key)

def _topk(sims: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k columns per row by similarity (descending; ties by memory order), with their similarities."""
    part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    ps = np.take_along_axis(sims, part, axis=1)
    order = np.lexsort((part, -ps), axis=1)
    idx = np.take_along_axis(part, order, axis=1)
    return idx.astype(np.int32), np.take_along_axis(sims, idx, axis=1).astype(np.float32)


def _date_mask(sims: np.ndarray, q_dates: np.ndarray, m_dates: np.ndarray) -> None:
    """Entries not published strictly before the query are removed (-inf), in place."""
    bad = m_dates[None, :] >= q_dates[:, None]
    if bad.any():
        sims[bad] = -np.inf


def neighbours_path(key: str) -> Path:
    return CACHE / f"neighbours_{key}.npz"


def build_neighbours(key: str, block: int = 256) -> dict:
    t0 = time.perf_counter()
    mem = memory()
    dev = C.dev_rows()
    qd = dev_dates()
    n_mem = len(mem["record_id"])
    usable = np.ones(n_mem, dtype=bool)
    if key == "dense":
        from uol.embedding_push import qwen_endpoint as Q
        from uol.round2 import b_embed
        texts = b_embed.memory_texts(Q.load_tokenizer())
        vec_path, idx_path = b_embed.STORE / "vectors.f32", b_embed.STORE / "index.jsonl"
        complete = vec_path.stat().st_size // (Q.DIM * 4)
        rows = {}
        with idx_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    break
                if e["row"] < complete:
                    rows[e["k"]] = e["row"]
        keys = [Q.content_key(t) for t in texts]
        usable = np.asarray([k in rows for k in keys])
        mm = np.memmap(vec_path, dtype=np.float32, mode="r", shape=(complete, Q.DIM))
        sel = np.flatnonzero(usable)
        mvec = Q.unit_rows(np.asarray(mm[np.asarray([rows[keys[i]] for i in sel], dtype=np.int64)]))
        del mm
        qv = np.ascontiguousarray(Q.query_vectors("c")[dev], dtype=np.float32)

        def score(s: int, e: int) -> np.ndarray:
            return qv[s:e] @ mvec.T
    elif key == "lex":
        from uol.facts_to_law.lexical import LexicalIndex
        from uol.round2.b_resolve import compose
        want = set(mem["record_id"].tolist())
        hist = {}
        with HIST_JSONL.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if str(r["record_id"]) in want:
                    hist[str(r["record_id"])] = compose(r)
        usable = np.asarray([rid in hist for rid in mem["record_id"]])
        sel = np.flatnonzero(usable)
        index = LexicalIndex([hist[mem["record_id"][i]] for i in sel], kind="bm25")
        d = C.release()
        texts = [d.queries[int(r)].text for r in dev]

        def score(s: int, e: int) -> np.ndarray:
            return np.asarray(index.score(texts[s:e]), dtype=np.float32)
    else:
        raise ValueError(key)
    m_dates = mem["date"][sel]
    out_idx = np.empty((len(dev), KMAX), dtype=np.int32)
    out_sim = np.empty((len(dev), KMAX), dtype=np.float32)
    for s in range(0, len(dev), block):
        e = min(s + block, len(dev))
        sims = score(s, e)
        _date_mask(sims, qd[s:e], m_dates)
        li, ls = _topk(sims, KMAX)
        out_idx[s:e] = sel[li]                                  # back to memory.jsonl positions
        out_sim[s:e] = ls
    CACHE.mkdir(parents=True, exist_ok=True)
    meta = {"key": key, "rows": "C.dev_rows() order", "n_dev": int(len(dev)), "kmax": KMAX,
            "memory_entries": int(n_mem), "memory_usable": int(usable.sum()), "memory_missing": int((~usable).sum()),
            "seconds": round(time.perf_counter() - t0), "utc": C.utc()}
    np.savez(neighbours_path(key), rows=dev.astype(np.int64), idx=out_idx, sim=out_sim, meta=json.dumps(meta))
    log(f"neighbours {key}: {meta}")
    return meta


@lru_cache(maxsize=2)
def neighbours(key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(neighbours_path(key))
    rows = z["rows"]
    if not np.array_equal(rows, C.dev_rows()):
        raise RuntimeError("neighbour cache rows differ from the dev rows")
    return rows, z["idx"], z["sim"]


# ----------------------------------------------------------------------------- votes

def year_shuffle(years: np.ndarray, draw: int) -> np.ndarray:
    """perm[i] = memory entry whose citation list entry i receives; permutation within publication year."""
    rng = np.random.default_rng(NULL_SEED + 7919 * int(draw))
    perm = np.arange(len(years))
    for y in np.unique(years):
        pos = np.flatnonzero(years == y)
        perm[pos] = pos[rng.permutation(len(pos))]
    return perm


def weights(sim: np.ndarray, age_years: np.ndarray, half_life: float | None) -> np.ndarray:
    w = np.maximum(np.nan_to_num(sim.astype(np.float64), neginf=0.0), 0.0)
    if half_life is not None:
        w = w * np.power(0.5, age_years / half_life)
    tot = w.sum(axis=1, keepdims=True)
    return np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)


def cited_matrices(cited: Sequence[Sequence[int]], n_cand: int) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """(mass-one positive matrix, binary matrix), [n_memory, n_cand]."""
    lens = np.fromiter((len(c) for c in cited), dtype=np.int64, count=len(cited))
    rows = np.repeat(np.arange(len(cited)), lens)
    cols = np.fromiter((a for c in cited for a in c), dtype=np.int64, count=int(lens.sum()))
    mass = np.repeat(1.0 / lens, lens).astype(np.float32)
    shape = (len(cited), n_cand)
    return (sparse.csr_matrix((mass, (rows, cols)), shape=shape),
            sparse.csr_matrix((np.ones(len(cols), dtype=np.float32), (rows, cols)), shape=shape))


def votes_from(idx: np.ndarray, sim: np.ndarray, q_dates: np.ndarray, m_dates: np.ndarray,
               cited: Sequence[Sequence[int]], k: int, half_life: float | None, n_cand: int,
               perm: np.ndarray | None = None) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """(vote, count) sparse [n_rows, n_cand] from cached neighbours; ``perm`` re-assigns citation lists (null)."""
    idx, sim = idx[:, :k], sim[:, :k]
    nb_dates = m_dates[idx]
    if (nb_dates >= q_dates[:, None]).any():
        raise AssertionError("a memory neighbour does not predate its query")
    age = (q_dates[:, None] - nb_dates).astype(np.float64) / 365.25
    w = weights(sim, age, half_life)
    n = len(idx)
    r = np.repeat(np.arange(n), k)
    W = sparse.csr_matrix((w.ravel().astype(np.float32), (r, idx.ravel())), shape=(n, len(m_dates)))
    B = sparse.csr_matrix(((w.ravel() > 0).astype(np.float32), (r, idx.ravel())), shape=(n, len(m_dates)))
    if perm is not None:
        cited = [cited[int(j)] for j in perm]
    P, Pb = cited_matrices(cited, n_cand)
    return (W @ P).tocsr(), (B @ Pb).tocsr()


_VOTES: "OrderedDict[tuple, tuple]" = OrderedDict()


def config_votes(cfg: dict, draw: int | None = None) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    key = (cfg["key"], cfg["decay"], int(cfg["k"]), cfg.get("_null"), draw)
    if key in _VOTES:
        _VOTES.move_to_end(key)
        return _VOTES[key]
    mem = memory()
    _, idx, sim = neighbours(cfg["key"])
    perm = None
    if draw is not None:
        if cfg.get("_null") != "shuffle_year":
            raise ValueError(f"unknown null {cfg.get('_null')!r}")
        perm = year_shuffle(mem["year"], draw)
    out = votes_from(idx, sim, dev_dates(), mem["date"], mem["cited"], int(cfg["k"]), DECAYS[cfg["decay"]],
                     len(C.release().candidate_ids), perm)
    _VOTES[key] = out
    while len(_VOTES) > 30:
        _VOTES.popitem(last=False)
    return out


def row_columns(v: np.ndarray, c: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """[len(pc), 3] (vote, count, logrank) of one query row from its dense vote/count rows."""
    nz = np.sort(v[v > 0])
    larger = len(nz) - np.searchsorted(nz, v[pc], side="right")
    return np.stack([v[pc], c[pc], np.log1p(larger.astype(np.float32))], axis=1).astype(np.float32)


def columns_for(cfg: dict, g, h, draw: int | None = None) -> np.ndarray:
    """arm_runner hook. Same columns for every dev outer group (memory holds no dev row); lockbox pairs stay NaN."""
    if getattr(g, "is_lockbox", False) or not str(g.role).startswith("dev_fold"):
        raise PermissionError("M1 columns are built for dev groups only")
    V, Cn = config_votes(cfg, draw)
    cols = h.empty_columns(len(COLUMNS))
    for i, r in enumerate(C.dev_rows()):
        sl, pc = h.pair_slice(int(r))
        cols[sl] = row_columns(V.getrow(i).toarray().ravel(), Cn.getrow(i).toarray().ravel(), pc)
    return cols


# ----------------------------------------------------------------------------- standalone and subgroups

STANDALONE = OUT / "standalone.json"


def standalone_scores() -> dict:
    """Arm-alone dev nDCG@10 (vote as the ranking score; count breaks ties) for every configuration."""
    from uol.embedding_push import common as ep
    from uol.round3.arm_runner import cfg_key
    dev = C.dev_rows()
    out = {}
    for cfg in CONFIGS:
        V, Cn = config_votes(cfg)
        M = (V.toarray() + 1e-6 * Cn.toarray()).astype(np.float32)
        rec = {}
        for pop in ("P_spec", "P"):
            rows = ep.population_rows(pop, dev)
            pos = np.searchsorted(dev, rows)
            rec[pop] = ep.evaluate(M[pos], rows, pop)["macro"]["ndcg_at_10"]
        out[cfg_key(cfg)] = {"config": cfg, **rec}
        log(f"standalone {cfg}: {rec}")
        del M
    C.write_json(STANDALONE, {"utc": C.utc(), "by_config": out,
                              "note": "vote matrix as the ranking (count x 1e-6 tie-break); dev rows only"})
    return out


def band_flags(band: str):
    """Query-level subgroup: the row has >= 1 specific gold article in ``band`` (fold-local training counts)."""
    def fn(rows: Sequence[int]) -> np.ndarray:
        gold = C.gold_rows(rows, "spec")
        return np.asarray([any(C.band_of_row(int(r))[a] == band for a in g) for r, g in zip(rows, gold)])
    return fn


def run(n_nulls: int) -> dict:
    from uol.round3 import arm_runner
    st = json.loads(STANDALONE.read_text(encoding="utf-8")) if STANDALONE.exists() else None
    metas = {k: json.loads(str(np.load(neighbours_path(k))["meta"])) for k in KEYS}
    notes = ("M1 tuned historical citation memory: 84,459 pre-2020-08-14 census tesis (round-2 B); columns vote/count/"
             "logrank on the frozen pool (no pool extension); dev training-fold rows not in memory (proposed DECISION); "
             f"null = citation lists shuffled within publication year ({n_nulls} draws). See arms/M1/NOTES.md.")
    return arm_runner.run(
        ARM, CONFIGS, columns_for, n_nulls=n_nulls, null_kinds=["shuffle_year"] if n_nulls else [],
        subgroups={"rare": band_flags("rare"), "frequent": band_flags("frequent"), "unseen": band_flags("unseen")},
        standalone=st, notes=notes,
        extra_result={"memory": {"source": str(MEMORY_JSONL.relative_to(C.ROOT)), "entries": len(memory()["record_id"]),
                                 "max_date": str(memory()["date"].max()), "neighbours": metas,
                                 "columns": list(COLUMNS)},
                      "subgroup_definition": "query has >= 1 specific gold article in the band (C.band_of_row)",
                      "code_sha256_m1": C.sha256_file(Path(__file__))},
        log=log)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["neighbours", "standalone", "run"])
    ap.add_argument("--keys", nargs="*", default=list(KEYS))
    ap.add_argument("--nulls", type=int, default=0)
    a = ap.parse_args()
    if a.command == "neighbours":
        for k in a.keys:
            build_neighbours(k)
    elif a.command == "standalone":
        standalone_scores()
    else:
        run(a.nulls)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
