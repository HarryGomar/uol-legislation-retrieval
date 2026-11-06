"""Whole-method time per new case on the run computer (D137): every stage from the facts to the top-10, per case.

The cost record timed only the learned ranker's lookup of precomputed source scores (``cost.latency``: 0.136 ms median
for S_full_ext). This job times the whole method for about 200 development cases treated as new: the facts go in,
the frozen stores, models and indexes are loaded once (load and fit times reported apart), and every per-case
quantity is computed from scratch: the query encoders (GPU job ``gpu_jobs/timing_encode.py``, and e5-small and the
LSA projections here on the CPU), the 50 first-stage sources and the ColBERT pairs (``timing_sources``), the
candidate pool, the 157 frozen feature columns, the topic columns (LSA, LDA similarity, LDA document model,
doc2vec), the M1 historical memory, the ranker's decision function and the top-10. The LLM stage is not run: its
recorded time is a separate row.

Systems: S_full_ext (the integrated method), S0 (the ranker without embeddings) and lstar (tuned lexical matching),
fold-0 models of the record, cases ``score_rows[:200]`` of DEV fold 0 (the cases of ``cost.latency``), 8 threads.

Each case is scored twice. The *stored* pass reads the stored inputs: it is the reproduction check of the per-case
path; the topic columns there are the ones the fold-0 fit computed (LDA and doc2vec inference is sequential and
seeded per call, so a single-case call differs slightly; the *fresh* pass shows by how much). The *fresh* pass is
the new-case path whose times are reported: encoders run on the case's text and their vectors replace the stored
ones; topic columns come from a single-case call; M1 neighbours from the fresh Qwen3-8B vector.

The gate (T2, 26 Oct 2026, after the second and third RUN-B jobs stopped on it) holds the per-case path to what the
timing must show. lstar's per-case top-10 must equal the record's. A learned ranker's per-case top-10, ranked on the
recomputed sources, must equal its own fold-0 refit's batch output (``LTRRefit.score``, the record's path), so a
source drift that changes a ranking stops the job there. The refit itself cannot reproduce the record's top-10: its
topic members LDA (``LdaMulticore``) and doc2vec are trained with several gensim workers, which is not reproducible
from one process to the next (the record's own S0 and S_full_ext fits, run side by side, already differ in their
doc2vec columns), so the refit's and the per-case agreement with the record are reported, not gated. Every source row
is compared with its frozen file and reported (lowest top-200 overlap, largest |diff|, the cases that differ); a
source stops the job only on a gross failure (missing, or a top-200 overlap below ``GATE``'s floor). The sources whose
small models are trained when the first stage loads (``RETRAINED``: the case-similarity adapters and the probe) are
retrained on the run computer; in the stored pass their frozen rows are used, like the stored vectors. The first stage
reads the labels its frozen files were built with (``first_stage_labels``: v11, while the bundle profile corrects
three cases' labels for the rankers; timing C stopped on the label-using sources this moved). A failed gate after the
warm-up cases stops the job (exit 2) with the report and every warm-up record (``warmup_cases.jsonl``) written.

The warm-up cases go through both passes like the timed ones, so the GPU encoders save their vectors too, marked
``warmup``; the run checks before any fit that every case of both passes has its encodings (T2, 26 Oct 2026: the
first RUN-B job stopped on the fresh warm-up pass after the fits, because the encoders had saved only the timed rows).
Warm-up seconds enter no statistic.

    python -m uol.final.timing stage-data --out <dir>                 (thesis PC: the inputs the bundle lacks)
    python -m uol.final.timing encode --out <dir>                     (run computer: GPU encoders, own processes)
    python -m uol.final.timing run --out <dir> [--cases 200]          (run computer: load, fit, two passes per case)
    python -m uol.final.timing summary --out <dir>                    (medians and p95 per stage and system)
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

SYSTEMS = ("S_full_ext", "S0", "lstar")
RANKERS = ("S_full_ext", "S0")
N_CASES, N_WARMUP = 200, 5
THREADS = 8                      # cost.LATENCY_THREADS
FIT_THREADS = 4                  # the record's fits ran with 4 threads (receipts S0__DEV, S_full_ext__DEV)
POPULATION, FOLD = "DEV", 0
RECORD = "store_dev_20261025b"
# GPU encoders per case (gpu_jobs/timing_encode.py) and the CPU encoders / transforms run here
GPU_MODELS = ("bge", "qwen", "e5proxy")
# the warm-up gate's stop conditions (T2, 26 Oct 2026, after timing C): what the timing must prove, plus a gross
# failure of a source; every source's agreement with its frozen file is reported, not gated
GATE = {"lstar_top10": 1.0, "case_top10_refit": 0.95, "source_overlap_floor": 0.9, "retrained_overlap_floor": 0.5}
# the labels the first stage reads: "v11", the labels the frozen source files were built with (the record ranked those
# files; its rankers were fitted on the corrected labels); None reads the release as loaded (timing C on RUN-B)
FIRST_STAGE_LABELS: str | None = "v11"
# sources whose small models are trained when the first stage loads (timing_sources._load_adapters, _load_probe):
# retrained on the run computer, they may drift from the frozen files trained on the thesis PC
RETRAINED = ("knn_adapt_spec", "knn_adapt_all", "knn_lsaadapt_spec", "knn_lsaadapt_all", "knn_lsacadapt_spec",
             "knn_lsacadapt_all", "proto_adapt", "proto_lsaadapt", "proto_lsacadapt", "proto_probe")
STAGE_ORDER = ("encode", "transform", "sources", "pool", "features", "topic", "m1", "rank", "top10")


# --------------------------------------------------------------------------------------------- first-stage labels


_V11: dict[str, Any] | None = None


def v11_labels() -> dict[str, Any]:
    """{query_id: (references, positive_idx, specific_idx)} of the release files for every query whose labels the
    active profile has changed (``context.apply_label_corrections``: the bundle's corrected labels); empty otherwise."""
    global _V11
    from uol.embedding_push import common as ep
    info = getattr(ep.release, "cache_info", None)
    if info is not None and not info().currsize:      # nothing loaded, so nothing corrected
        return {}
    d = ep.release()
    if not getattr(d, "_uol_final_corrected", False):
        return {}
    if _V11 is None:
        from uol.facts_to_law import crossfit
        raw = crossfit.load_release(ep.RELEASE)
        _V11 = {q.query_id: (list(raw.references[q.query_id]), list(raw.positive_idx[i]), list(raw.specific_idx[i]))
                for i, q in enumerate(raw.queries) if list(raw.references[q.query_id]) != list(d.references[q.query_id])}
        del raw
    return _V11


@contextlib.contextmanager
def first_stage_labels() -> Iterator[dict[str, Any]]:
    """The release's labels as the frozen first-stage files were built (``FIRST_STAGE_LABELS``), for the first stage's
    load and per-case scoring only, restored afterwards. The bundle profile corrects three queries' labels in place
    before anything runs; the frozen source files the record ranked were built on the v11 labels, so a first stage
    built on the corrected ones moves every label-using source (memories, priors, prototypes, adapters, the probe,
    document expansion, the Qwen memories) away from its file (timing C on RUN-B). The rankers keep the corrected
    labels the record fitted them on. A no-op in the local profile (v11)."""
    from uol.embedding_push import common as ep
    swap = v11_labels() if FIRST_STAGE_LABELS == "v11" else {}
    d = ep.release() if swap else None
    keep = {}
    for qid, (refs, pos, spec) in swap.items():
        i = d.query_index[qid]
        keep[qid] = (d.references[qid], d.positive_idx[i], d.specific_idx[i])
        d.references[qid], d.positive_idx[i], d.specific_idx[i] = refs, pos, spec
    try:
        yield {"labels": FIRST_STAGE_LABELS if swap else "as loaded", "queries": sorted(swap)}
    finally:
        for qid, (refs, pos, spec) in keep.items():
            i = d.query_index[qid]
            d.references[qid], d.positive_idx[i], d.specific_idx[i] = refs, pos, spec


# --------------------------------------------------------------------------------------------- clock and stats


class Clock:
    """Wall (``perf_counter``) and process CPU (``process_time``, all threads) seconds per named stage of one case."""

    def __init__(self) -> None:
        self.wall: dict[str, float] = {}
        self.cpu: dict[str, float] = {}

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        w, c = time.perf_counter(), time.process_time()
        try:
            yield
        finally:
            self.wall[name] = self.wall.get(name, 0.0) + time.perf_counter() - w
            self.cpu[name] = self.cpu.get(name, 0.0) + time.process_time() - c

    def add(self, name: str, wall: float, cpu: float = 0.0) -> None:
        self.wall[name] = self.wall.get(name, 0.0) + float(wall)
        self.cpu[name] = self.cpu.get(name, 0.0) + float(cpu)


def stats(values: Sequence[float]) -> dict[str, float | int | None]:
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if not len(v):
        return {"n": 0, "median": None, "p95": None, "mean": None, "max": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)), "mean": float(v.mean()),
            "max": float(v.max())}


def stage_of(name: str) -> str:
    """Report stage of a clock entry (``source:<id>`` entries are the first stage)."""
    if name.startswith("source:"):
        return "sources"
    return name.split(":", 1)[0]


def system_stages(system: str, fs_sources: dict[str, list[str]]) -> list[str]:
    """Clock entries that make up one system's per-case time (as if it ran alone; shared sources count for each)."""
    from uol.final import timing_sources as TS
    if system == "lstar":
        return ["lstar:score", "top10:lstar"]
    srcs = fs_sources[system]
    out = [f"source:{s}" for s in srcs]
    if "colbert_bge" in srcs or any(s in srcs for s in ("dense_bge_maxp", "dense_bge_firstp", "dense_bge_long", "H_bge")):
        out += ["encode:bge"]
    if any(s.endswith("qwen8b") for s in srcs) or "K_lex_ext" in srcs or system == "S_full_ext":
        out += ["encode:qwen"]
    if any(s in srcs for s in ("dense_e5_maxp", "knn_e5", "proto_e5", "knn_fused", "knn_x_spec", "knn_x_all",
                                "knn_xc_spec", "knn_xc_all", "proto_probe") + TS.CASESIM + ("ft_e5_q",)):
        out += ["encode:e5"]
    if "ft_e5_q" in srcs:
        out += ["encode:ft_e5_q"]
    if any(s in srcs for s in TS.CASESIM):
        out += ["encode:casesim_e5"]
    if any(s in srcs for s in ("knn_lsaadapt_spec", "knn_lsaadapt_all", "proto_lsaadapt", "knn_lsacadapt_spec",
                                "knn_lsacadapt_all", "proto_lsacadapt")):
        out += ["transform:lsa", "transform:clsa"]
    out += [f"pool:{system}", f"features:{system}", f"topic:{system}", f"rank:{system}", f"top10:{system}"]
    if system == "S_full_ext":
        out += ["m1:neighbours"]
    return out


# --------------------------------------------------------------------------------------------- profile


def activate(data: str | None, store: str | None) -> Any:
    from uol.final import context
    data = data or os.environ.get("UOL_FINAL_DATA")
    if data:
        return context.activate_bundle(Path(data), store=Path(store) if store else None)
    return context.activate_local("v11", store=Path(store) if store else None)


def case_rows(n: int = N_CASES, warmup: int = N_WARMUP) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """(timed rows, warm-up rows, population, fold): the first ``n`` DEV fold-0 score rows (``cost.latency``), then the
    next ``warmup`` rows of the same fold for warming the caches."""
    from uol.final import context, guard, populations
    pop = populations.load(POPULATION, context.active().store)
    f = next(x for x in pop.folds() if x.k == FOLD)
    rows = guard.dev_rows_only(np.asarray(f.score_rows[:n]), "timing rows")
    warm = guard.dev_rows_only(np.asarray(f.score_rows[n:n + warmup]), "timing warm-up rows")
    if len(warm) < warmup:                         # a smaller population: warm up on the first timed rows instead
        warm = rows[:warmup]
    return rows, warm, pop, f


_VIEW: tuple[np.ndarray | None, np.ndarray | None] | None = None


def eval_view() -> tuple[np.ndarray | None, np.ndarray | None]:
    """(excluded candidates, tie key) of the record's evaluator in P_spec: the generic articles leave the list and ties
    go by candidate-ID order (``ep.evaluate``)."""
    global _VIEW
    if _VIEW is None:
        from uol.embedding_push import common as ep
        data = ep.release()
        _VIEW = (data.generic_idx if len(data.generic_idx) else None, ep.tie_order())
    return _VIEW


def top10(row_scores: np.ndarray, view: tuple[np.ndarray | None, np.ndarray | None] | None = None) -> list[int]:
    """Top 10 as the record's evaluator lists it (``metrics.evaluate_matrix`` under ``eval_view``)."""
    from uol.facts_to_law import metrics
    exclude, key = eval_view() if view is None else view
    res = metrics.evaluate_matrix(np.asarray(row_scores, dtype=np.float64)[None], [[]], exclude_idx=exclude,
                                  top_k=10, tie_order=key)
    return [int(c) for c in res[0]["topk"]]


# --------------------------------------------------------------------------------------------- ranker per case


@contextlib.contextmanager
def _one_row_table(pc: np.ndarray, row: int) -> Iterator[None]:
    """Point ``e2_adapter.pair_table`` at one case's pool (row ``row``), so ``e2.full_columns`` gives its columns with
    the frozen arithmetic."""
    from uol.round3 import e2_adapter as e2
    orig = e2.pair_table
    row_ptr = np.zeros(int(row) + 2, dtype=np.int64)
    row_ptr[int(row) + 1] = len(pc)
    e2.pair_table = lambda: (np.full(len(pc), int(row), dtype=np.int32), np.asarray(pc, dtype=np.int32), row_ptr)
    try:
        yield
    finally:
        e2.pair_table = orig


def full_cols(score_row: np.ndarray, pc: np.ndarray, row: int) -> np.ndarray:
    """[len(pc), 3] robust z, log rank and instrument-max z of one full score row (the frozen pair-table recipe)."""
    from uol.round3 import e2_adapter as e2
    with _one_row_table(pc, row):
        _, cols = e2.full_columns(np.asarray(score_row, dtype=np.float32)[None, :], [int(row)])
    return cols


def pairs_cols(cidx: np.ndarray, score: np.ndarray, pc: np.ndarray) -> np.ndarray:
    """[len(pc), 4] raw, z, log rank and presence of a pair source for one case (``d58_refit.pairs_features_dev``
    restricted to one query)."""
    import pandas as pd
    frame = pd.DataFrame({"q": np.zeros(len(cidx), dtype=np.int64), "c": np.asarray(cidx, dtype=np.int64),
                          "s": np.asarray(score, dtype=np.float32)}).drop_duplicates(["q", "c"], keep="last")
    out = np.full((len(pc), 4), np.nan, dtype=np.float32)
    out[:, 3] = 0.0
    if not len(frame):
        return out
    grp = frame.groupby("q")["s"]
    frame["med"] = grp.transform("median")
    frame["mad"] = (frame["s"] - frame["med"]).abs().groupby(frame["q"]).transform("median")
    frame["std"] = grp.transform("std").fillna(0.0)
    scale = 1.4826 * frame["mad"]
    frame["scale"] = np.where(scale > 1e-8, scale, np.where(frame["std"] > 1e-8, frame["std"], 1.0))
    frame["z"] = (frame["s"] - frame["med"]) / frame["scale"]
    frame["rank"] = grp.rank(method="min", ascending=False)
    frame = frame.sort_values("c")
    skeys = frame["c"].to_numpy()
    pkeys = np.asarray(pc, dtype=np.int64)
    pos = np.minimum(np.searchsorted(skeys, pkeys), len(skeys) - 1)
    present = skeys[pos] == pkeys
    for j, name in enumerate(("s", "z", "rank")):
        vals = frame[name].to_numpy(dtype=np.float32)[pos]
        if name == "rank":
            vals = np.log(vals)
        out[:, j] = np.where(present, vals, np.nan)
    out[:, 3] = present.astype(np.float32)
    return out


class CaseRanker:
    """One fitted learned ranker (``adapters.ltr.LTRRefit`` after its fold-0 fit) applied to one case at a time."""

    def __init__(self, sc: Any, fold_k: int = FOLD):
        from uol.embedding_push import fusion_ltr as fl
        self.sc, self.h, self.v = sc, sc.h, sc.h.v
        self.by_id = {s.id: s for s in self.h.F.sources}
        self.model, self.Xout, self.idx_out = self.h.last_fit
        ctx = self.h.ctx
        self.N = ctx.N
        self.table, _ = fl.instrument_tables(ctx)
        self.gid = np.where(ctx.fold >= 0, ctx.fold, len(ctx.groups) - 1)
        self.log_len = np.log1p(np.asarray([len(t) for t in ctx.data.candidate_texts], dtype=np.float32))
        self.log_qlen = np.log1p(np.asarray([len(t) for t in ctx.texts], dtype=np.float32))
        self.extra_ids = [m.sid for m in sc.extra]
        self.m1 = getattr(sc, "extension", None) == "m1"
        ptr = np.searchsorted(self.h.F.pq[self.idx_out], np.arange(len(self.h.F.row_ptr)))
        self._fit_ptr = ptr

    @property
    def sources(self) -> list[str]:
        return list(dict.fromkeys(list(self.v.pool_ids) + list(self.v.feature_ids)))

    def pool(self, src: dict[str, Any]) -> np.ndarray:
        """Candidates of the case's pool: the union of the top 200 of every pool source (sorted, as the pair table)."""
        from uol.embedding_push import fusion_ltr as fl
        tops = [fl.topk_block(fl.sanitize(np.asarray(src[s], dtype=np.float32)[None, :]))[0] for s in self.v.pool_ids]
        return np.unique(np.concatenate(tops)).astype(np.int64)

    def features(self, row: int, src: dict[str, Any], pc: np.ndarray) -> np.ndarray:
        """The frozen columns of the variant for the case's pool (sources in feature order, then the 3 shared ones)."""
        from uol.embedding_push import fusion_ltr as fl
        cols = []
        for sid in self.v.feature_ids:
            s = self.by_id[sid]
            if s.kind == "full":
                cols.append(full_cols(fl.sanitize(np.asarray(src[sid], dtype=np.float32)[None, :])[0], pc, row))
            else:
                cols.append(pairs_cols(src[sid]["cidx"], src[sid]["score"], pc))
        shared = np.stack([self.log_len[pc], self.table[self.gid[row], pc], np.full(len(pc), self.log_qlen[row])], axis=1)
        return np.hstack(cols + [shared.astype(np.float32)]).astype(np.float32)

    def extras(self, row: int, extra_rows: dict[str, np.ndarray], pc: np.ndarray, m1_cols: np.ndarray | None) -> np.ndarray:
        blocks = [full_cols(extra_rows[m], pc, row) for m in self.extra_ids]
        if self.m1:
            blocks.append(np.asarray(m1_cols, dtype=np.float32))
        return np.hstack(blocks).astype(np.float32) if blocks else np.empty((len(pc), 0), dtype=np.float32)

    def decide(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(X).astype(np.float32)

    def pooled_row(self, pc: np.ndarray, sc: np.ndarray, lstar_row: np.ndarray) -> np.ndarray:
        """``fusion_ltr.write_pooled`` for one row: pooled candidates keep their scores, the rest sit below in L* order."""
        lrow = np.asarray(lstar_row)
        order = np.argsort(-lrow, kind="stable")
        rank = np.empty(len(lrow), dtype=np.float64)
        rank[order] = np.arange(1, len(lrow) + 1)
        floor = float(sc.min()) if len(sc) else 0.0
        out = ((floor - 1.0) - rank / (len(lrow) + 1.0)).astype(np.float32)
        out[pc] = sc
        return out

    def fit_rows(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """(candidates, feature rows) of ``row`` in the fold-0 fit (``Harness.last_fit``)."""
        sl = slice(int(self._fit_ptr[row]), int(self._fit_ptr[row + 1]))
        return np.asarray(self.h.F.pc[self.idx_out[sl]], dtype=np.int64), np.asarray(self.Xout[sl])


# --------------------------------------------------------------------------------------------- M1 per case


class M1Case:
    """M1 historical citation memory for one case: Qwen3-8B (variant c) neighbours among the 84,459 older tesis
    published before it, then the fold-0 configuration's votes (``round3.m1_memory``, the dense key)."""

    def __init__(self, cfg: dict):
        from uol.embedding_push import common as ep
        from uol.final import timing_sources as TS
        from uol.round3 import common as R3, m1_memory as M1
        if cfg["key"] != "dense":
            raise ValueError(f"M1 key {cfg['key']!r}: only the dense key is timed")
        self.cfg, self.M1 = cfg, M1
        self.mem = M1.memory()
        z = np.load(TS.staged_path("m1/dense_memory.npz"))
        self.sel = np.asarray(z["sel"], dtype=np.int64)
        self.mvec_t = np.ascontiguousarray(TS.load_array(TS.staged_path("m1/dense_memory_vectors.npy")).T)
        if self.mvec_t.shape[1] != len(self.sel):
            raise ValueError("M1 memory vectors do not match their selection")
        self.m_dates_sel = self.mem["date"][self.sel]
        self.P, self.Pb = M1.cited_matrices(self.mem["cited"], len(ep.release().candidate_ids))
        pd = R3.publication_dates()
        self.dates = {int(r): np.datetime64(pd[int(r)], "D") for r in ep.dev_rows()}
        rows, idx, sim = M1.neighbours("dense")
        self.cached = {int(r): (idx[i], sim[i]) for i, r in enumerate(rows)}

    def neighbours(self, q: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
        sims = (np.asarray(q, dtype=np.float32)[None, :] @ self.mvec_t)
        qd = np.asarray([self.dates[int(row)]], dtype="datetime64[D]")
        self.M1._date_mask(sims, qd, self.m_dates_sel)
        li, ls = self.M1._topk(sims, self.M1.KMAX)
        return self.sel[li[0]], ls[0]

    def votes(self, idx: np.ndarray, sim: np.ndarray, row: int) -> tuple[np.ndarray, np.ndarray]:
        """``m1_memory.votes_from`` for one row with the citation matrices built once."""
        from scipy import sparse
        k = int(self.cfg["k"])
        idx, sim = np.asarray(idx)[None, :k], np.asarray(sim)[None, :k]
        nb_dates = self.mem["date"][idx]
        qd = np.asarray([self.dates[int(row)]], dtype="datetime64[D]")
        if (nb_dates >= qd[:, None]).any():
            raise AssertionError("a memory neighbour does not predate its query")
        age = (qd[:, None] - nb_dates).astype(np.float64) / 365.25
        w = self.M1.weights(sim, age, self.M1.DECAYS[self.cfg["decay"]])
        r = np.repeat(np.arange(1), k)
        n_mem = len(self.mem["date"])
        W = sparse.csr_matrix((w.ravel().astype(np.float32), (r, idx.ravel())), shape=(1, n_mem))
        B = sparse.csr_matrix(((w.ravel() > 0).astype(np.float32), (r, idx.ravel())), shape=(1, n_mem))
        return (W @ self.P).toarray().ravel(), (B @ self.Pb).toarray().ravel()

    def columns(self, q: np.ndarray, row: int, pc: np.ndarray, cached: bool = False) -> tuple[np.ndarray, dict]:
        idx, sim = self.cached[int(row)] if cached else self.neighbours(q, row)
        v, c = self.votes(idx, sim, row)
        info = {}
        if not cached:
            ci, cs = self.cached[int(row)]
            k = int(self.cfg["k"])
            info = {"same_top_k": bool(np.array_equal(np.asarray(idx[:k]), np.asarray(ci[:k]))),
                    "sim_max_abs_diff": float(np.max(np.abs(np.asarray(sim[:k]) - np.asarray(cs[:k]))))}
        return self.M1.row_columns(v, c, pc), info


# --------------------------------------------------------------------------------------------- CPU encoders


class CpuEncoders:
    """e5-small query encoder (the October CPU recipe) and the word / character LSA projections (refitted on the
    development texts with the frozen recipe; ``round2.c_vectors``)."""

    def __init__(self, threads: int = THREADS, lsa: bool = True, e5: bool = True):
        self.threads, self.lsa, self.e5 = threads, {}, None
        if e5:
            from uol.facts_to_law import dense
            point_e5_model()
            dense.configure_runtime(threads)
            self.e5 = dense.get_encoder()
        if lsa:
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer
            from uol.embedding_push import common as ep, memory_proto as mp
            from uol.facts_to_law.lexical import CHAR_SETTINGS, TOKEN_PATTERN
            t = mp.texts()
            dev = ep.dev_rows()
            for name, make in (("lsa", lambda: TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, strip_accents="unicode",
                                                               sublinear_tf=True, ngram_range=(1, 2), min_df=3,
                                                               max_features=200_000, dtype=np.float32)),
                               ("clsa", lambda: TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32))):
                vectorizer = make()
                vectorizer.fit([t[int(r)] for r in dev])
                svd = TruncatedSVD(n_components=512, algorithm="randomized", n_iter=5, random_state=0)
                svd.fit(vectorizer.transform([t[int(r)] for r in dev]))
                self.lsa[name] = (vectorizer, svd)

    def encode_e5(self, text: str) -> np.ndarray:
        from uol.facts_to_law import dense
        return dense.encode_texts([text], kind="query", batch_size=16, threads=self.threads, encoder=self.e5)[0]

    def transform(self, name: str, text: str) -> np.ndarray:
        from uol.embedding_push import memory_proto as mp
        vectorizer, svd = self.lsa[name]
        return mp._unit(svd.transform(vectorizer.transform([text])).astype(np.float32))[0]


def point_e5_model() -> None:
    """The e5-small snapshot of the run computer (``<work>/models``) when the hard-coded October path is absent."""
    from uol.facts_to_law import dense
    from uol.final import context, rebuild
    if Path(dense.MODEL_DIR).exists():
        return
    folder = context.active().work / "models" / rebuild.MODELS["E5_DIR"][2]
    if not folder.exists():
        raise FileNotFoundError(f"e5-small model folder missing: {folder}")
    dense.MODEL_DIR = folder
    b = dict(dense.MODEL_BINDING)
    b["path"] = str(folder)
    b["files"] = {n: {**v, "path": str(folder / n)} for n, v in b.get("files", {}).items()}
    dense.MODEL_BINDING = b


# --------------------------------------------------------------------------------------------- reference top-10


def reference_top10(path: Path | None = None) -> dict[str, dict[str, list[int]]]:
    """{system: {query_id: top-10 candidate indices}} of the record (staged ``reference_top10.json``)."""
    from uol.final import timing_sources as TS
    if path is None and TS.staged_root() is None:              # thesis PC: read the registered record itself
        return ref_from_record(None)["systems"]
    p = Path(path) if path else TS.staged_path("reference/reference_top10.json")
    return json.loads(p.read_text(encoding="utf-8"))["systems"]


# --------------------------------------------------------------------------------------------- the run


class Run:
    def __init__(self, out: Path, n: int, warmup: int, encodings: Path | None, colbert: str, systems: Sequence[str],
                 sources_only: bool = False, sources: Sequence[str] | None = None, warmup_only: bool = False):
        self.out, self.n, self.warmup = Path(out), int(n), int(warmup)
        self.warmup_only = warmup_only
        self.encodings = Path(encodings) if encodings else None
        self.colbert, self.systems = colbert, list(systems)
        self.sources_only = sources_only
        self.source_list = list(sources) if sources else None
        self.out.mkdir(parents=True, exist_ok=True)
        self.load_s: dict[str, float] = {}
        self.fit_s: dict[str, float] = {}
        self.cases_path = self.out / "cases.jsonl"

    def log(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
        print(line, flush=True)
        with (self.out / "timing.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def _t(self, name: str, fn, bucket: dict | None = None):
        t0 = time.perf_counter()
        v = fn()
        (self.load_s if bucket is None else bucket)[name] = round(time.perf_counter() - t0, 3)
        self.log(f"{name}: {time.perf_counter() - t0:.1f}s")
        return v

    # ------------------------------------------------------------------ load and fit
    def setup(self) -> None:
        from uol.final import cost, registry
        from uol.final import timing_sources as TS
        self.rows, self.warm, self.pop, self.fold = case_rows(self.n, self.warmup)
        self.log(f"cases: {len(self.rows)} timed (DEV fold {FOLD} score rows), {len(self.warm)} warm-up")
        self.gpu = self._load_encodings()
        self.check_encodings(self.source_list or [])
        self.scorers: dict[str, Any] = {}
        self.replay: dict[str, dict[str, dict[int, np.ndarray]]] = {}
        want = set(int(r) for r in np.concatenate([self.rows, self.warm]))
        for sid in [s for s in self.systems if s in RANKERS] + [s for s in self.systems if s not in RANKERS]:
            if sid in RANKERS and self.sources_only:
                continue
            sc = registry.make(sid)
            self._t(f"prepare:{sid}", lambda: sc.prepare(self.pop))
            if sid in RANKERS:
                self.check_encodings(list(sc.h.v.pool_ids) + list(sc.h.v.feature_ids))     # before its fit
                self.replay[sid] = {}
                for m in sc.extra:                      # keep the topic rows the fit computes for the timed cases
                    self.replay[sid][m.sid] = {}
                    m.score = _recording(m.score, self.replay[sid][m.sid], want)
                from uol.final.adapters import topic
                topic._MODELS.clear()                   # each fit starts like a fresh process (the record's)
                cost.set_threads(FIT_THREADS)
                self._t(f"fit:{sid}", lambda: sc.fit(self.fold), self.fit_s)
            else:
                cost.set_threads(THREADS)
                self._t(f"fit:{sid}", lambda: sc.fit(self.fold), self.fit_s)
            self.scorers[sid] = sc
        cost.set_threads(THREADS)
        self.rankers = {s: CaseRanker(self.scorers[s]) for s in RANKERS if s in self.scorers and not self.sources_only}
        need: list[str] = []
        self.sys_sources: dict[str, list[str]] = {}
        for s in RANKERS:
            if s in self.scorers:
                v = self.scorers[s].h.v
                self.sys_sources[s] = list(dict.fromkeys(list(v.pool_ids) + list(v.feature_ids)))
                need += self.sys_sources[s]
        if self.source_list is not None:
            need += self.source_list
        need = list(dict.fromkeys(need)) or ["L_star"]
        self.check_encodings(need)
        self.fs = TS.FirstStage("dev_fold0", sources=need, colbert=self.colbert if "colbert_bge" in need else "off")
        if self.encodings is not None and (self.encodings / "bge_chunks.npz").exists():
            self.fs.colbert_chunks = self.encodings / "bge_chunks.npz"
        with first_stage_labels() as lab:
            self._t("first_stage", self.fs.load)
        self.first_stage_labels = lab
        self.log(f"first stage labels: {lab['labels']} (queries restored: {lab['queries']})")
        self.load_s.update({f"first_stage:{k}": v for k, v in self.fs.load_seconds.items()})
        lsa_needed = any(s in need for s in ("knn_lsaadapt_spec", "knn_lsaadapt_all", "proto_lsaadapt", "knn_lsacadapt_spec",
                                             "knn_lsacadapt_all", "proto_lsacadapt", "proto_probe"))
        self.cpu = self._t("cpu_encoders", lambda: CpuEncoders(THREADS, lsa=lsa_needed, e5="e5" in self.fs.stored))
        self.m1 = None
        if "S_full_ext" in self.rankers:
            sc = self.scorers["S_full_ext"]
            self.m1 = self._t("m1_memory", lambda: M1Case(_m1_cfg()))
        self.ref = reference_top10()

    def _load_encodings(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.encodings is None:
            return out
        for m in GPU_MODELS:
            p = self.encodings / f"{m}.npz"
            if p.exists():
                z = np.load(p, allow_pickle=False)
                out[m] = {k: z[k] for k in z.files}
                meta = self.encodings / f"{m}.json"
                if meta.exists():
                    out[m]["meta"] = json.loads(meta.read_text(encoding="utf-8"))
        return out

    def check_encodings(self, sources: Sequence[str]) -> None:
        """Every case of both passes has its entry in every loaded GPU encoding (timed rows as timed, warm-up rows as
        warm-up), and the bge tokens exist when ``colbert_bge`` reads the token store. Raises before the fits, not
        after them."""
        miss = []
        for m in self.gpu:
            miss += [f"{m}: timed row {int(r)}" for r in self.rows if self.gpu_row(m, int(r), warmup=False) is None]
            miss += [f"{m}: warm-up row {int(r)}" for r in self.warm if self.gpu_row(m, int(r), warmup=True) is None]
        if "colbert_bge" in sources and self.colbert == "store" and "tok_off" not in self.gpu.get("bge", {}):
            miss.append("bge: no query token vectors (encode/bge.npz), which colbert_bge reads in the fresh pass")
        if miss:
            raise RuntimeError(f"GPU encodings missing for {len(miss)} case(s): {miss[:10]}")

    def gpu_row(self, model: str, row: int, warmup: bool = False) -> dict[str, Any] | None:
        """The case's entry in a GPU encoding: the timed entry, or with ``warmup`` the warm-up entry (an encoding
        without the ``warmup`` array holds timed entries only)."""
        g = self.gpu.get(model)
        if g is None:
            return None
        flags = np.asarray(g["warmup"], dtype=bool) if "warmup" in g else np.zeros(len(g["rows"]), dtype=bool)
        pos = np.flatnonzero((np.asarray(g["rows"]) == int(row)) & (flags == bool(warmup)))
        if not len(pos):
            return None
        i = int(pos[0])
        rec: dict[str, Any] = {"seconds": float(g["seconds"][i]), "warmup": bool(warmup)}
        if "vectors" in g:
            rec["vector"] = np.asarray(g["vectors"][i], dtype=np.float32)
        if "tok_off" in g:
            a, b = int(g["tok_off"][i]), int(g["tok_off"][i + 1])
            rec["tokens"] = np.asarray(g["tokens"][a:b], dtype=np.float32)
        for k in ("ft_seconds", "casesim_seconds"):
            if k in g:
                rec[k] = float(g[k][i])
        return rec

    # ------------------------------------------------------------------ one case
    def score_case(self, row: int, fresh: bool, warmup: bool = False) -> dict[str, Any]:
        from uol.embedding_push import common as ep
        from uol.final import timing_sources as TS
        clock = Clock()
        text = ep.release().queries[int(row)].text
        rec: dict[str, Any] = {"row": int(row), "query_id": ep.release().queries[int(row)].query_id,
                               "pass": "fresh" if fresh else "stored", "warmup": bool(warmup)}
        self.fs.reset_vectors([row])
        qtok = None
        qc = self.fs.vectors["Qc"][int(row)] if "Qc" in self.fs.vectors else None
        fid: dict[str, float] = {}
        if fresh:
            inj: dict[str, np.ndarray] = {}
            got = {m: self.gpu_row(m, row, warmup) for m in self.gpu}
            if any(v is None for v in got.values()):          # check_encodings ran at setup: not expected here
                raise RuntimeError(f"row {row} ({'warm-up' if warmup else 'timed'}) missing from "
                                   f"{[m for m, v in got.items() if v is None]}")
            b = got.get("bge")
            if b is not None:
                clock.add("encode:bge", b["seconds"])
                inj["bge"] = b["vector"]
                qtok = {"tokens": b["tokens"]}
                fid["bge_cos"] = float(np.dot(b["vector"] / np.linalg.norm(b["vector"]), self.fs.stored["bge"][row]))
            q = got.get("qwen")
            if q is not None and "Qc" in self.fs.vectors:
                clock.add("encode:qwen", q["seconds"])
                inj["Qc"] = q["vector"]
                qc = q["vector"] / np.linalg.norm(q["vector"])
                fid["qwen_cos"] = float(np.dot(qc, self.fs.stored["Qc"][row]))
            p = got.get("e5proxy")
            if p is not None:
                clock.add("encode:ft_e5_q", p.get("ft_seconds", 0.0))
                clock.add("encode:casesim_e5", p.get("casesim_seconds", 0.0))
            if self.cpu.e5 is not None:
                with clock.stage("encode:e5"):
                    e = self.cpu.encode_e5(text)
                inj["e5"] = e
                fid["e5_cos"] = float(np.dot(e, self.fs.stored["e5"][row]))
            for name in self.cpu.lsa:
                with clock.stage(f"transform:{name}"):
                    v = self.cpu.transform(name, text)
                inj[name] = v
                fid[f"{name}_cos"] = float(np.dot(v, self.fs.unit[name][row]))
            self.fs.inject(row, inj)
        elif self.colbert == "store" and "colbert_bge" in self.fs.sources:
            qtok = self.stored_qtok(row)
        with first_stage_labels():
            src = self.fs.case(row, qtok=qtok, clock=clock)
        rec["fidelity"] = fid
        if not fresh:
            rec["sources"] = {}
            for sid, val in src.items():
                rec["sources"][sid] = TS.compare_pairs(row, val) if sid == "colbert_bge" else TS.compare_row(sid, row, val)
                if sid in RETRAINED:                  # stored inputs: the frozen row, as for the stored vectors
                    rec["sources"][sid]["retrained"] = True
                    src[sid] = TS.frozen_row(sid, row)
        rec["systems"] = {}
        for sid, rk in self.rankers.items():
            rec["systems"][sid] = self.rank_case(sid, rk, row, src, clock, fresh, qc)
        if "lstar" in self.scorers:
            with clock.stage("lstar:score"):
                S = self.scorers["lstar"].score(np.asarray([row]))[0]
            with clock.stage("top10:lstar"):
                t10 = top10(S)
            rec["systems"]["lstar"] = {"top10": t10}
        for sid, r in rec["systems"].items():
            ref = self.ref.get(sid, {}).get(rec["query_id"])
            r["record_top10_equal"] = None if ref is None else (r["top10"] == list(ref))
        rec["wall"] = {k: round(v, 6) for k, v in clock.wall.items()}
        rec["cpu"] = {k: round(v, 6) for k, v in clock.cpu.items()}
        return rec

    def stored_qtok(self, row: int) -> dict[str, np.ndarray] | None:
        """The case's stored ColBERT query token vectors (``<work>/colbert_full/store/query_tokens.npy``)."""
        from uol.final import timing_sources as TS
        if not hasattr(self, "_qtok"):
            store = TS.colbert_store()
            self._qtok = (np.load(store / "query_tokens.npy", mmap_mode="r"),
                          np.load(self.fs.colbert_chunks)["q_off"] if getattr(self.fs, "colbert_chunks", None) else None)
        tokens, q_off = self._qtok
        if q_off is None:
            return None
        return {"tokens": np.asarray(tokens[int(q_off[row]):int(q_off[row + 1])], dtype=np.float32)}

    def rank_case(self, sid: str, rk: CaseRanker, row: int, src: dict, clock: Clock, fresh: bool, qc) -> dict[str, Any]:
        out: dict[str, Any] = {}
        with clock.stage(f"pool:{sid}"):
            pc = rk.pool(src)
        with clock.stage(f"features:{sid}"):
            X = rk.features(row, src, pc)
        extra_rows: dict[str, np.ndarray] = {}
        with clock.stage(f"topic:{sid}"):
            for m in rk.sc.extra:
                if fresh:
                    extra_rows[m.sid] = np.asarray(m.score.__wrapped__(np.asarray([row])), dtype=np.float32)[0]
                else:
                    extra_rows[m.sid] = self.replay[sid][m.sid][int(row)]
        m1_cols = None
        if rk.m1:
            with clock.stage("m1:neighbours"):
                m1_cols, info = self.m1.columns(qc, row, pc, cached=False)
            out["m1"] = info
        with clock.stage(f"features:{sid}"):
            X = np.hstack([X, rk.extras(row, extra_rows, pc, m1_cols)])
        with clock.stage(f"rank:{sid}"):
            sc = rk.decide(X)
            S = rk.pooled_row(pc, sc, src["L_star"])
        with clock.stage(f"top10:{sid}"):
            out["top10"] = top10(S)
        fit_pc, fit_X = rk.fit_rows(row)
        out["batch_top10"] = top10(rk.sc.score([row])[0])        # the refit's own batch output for this row
        out["pool_size"] = int(len(pc))
        out["pool_equal_fit"] = bool(np.array_equal(fit_pc, pc))
        if out["pool_equal_fit"]:
            d = np.abs(np.nan_to_num(X, nan=1e9) - np.nan_to_num(fit_X, nan=1e9))
            out["feature_max_abs_diff"] = float(d.max()) if d.size else 0.0
            ncol = X.shape[1]
            nbase = len(rk.v.hgb_cols)
            out["feature_max_abs_diff_base"] = float(d[:, :nbase].max()) if d.size else 0.0
            out["feature_max_abs_diff_extra"] = float(d[:, nbase:].max()) if ncol > nbase and d.size else 0.0
            out["fit_top10"] = top10(rk.pooled_row(fit_pc, rk.decide(fit_X), src["L_star"]))
        return out

    # ------------------------------------------------------------------ driver
    def gate(self, recs: list[dict]) -> dict[str, Any]:
        """Stored-pass checks (module docstring). It stops on what the timing must prove: lstar's per-case top-10
        equals the record's, and each learned ranker's per-case top-10 equals its own refit's batch output with the
        recomputed sources as its inputs, so a source drift that changes a ranking stops it there. A source stops it
        only on a gross failure: missing from a case, or a top-200 overlap with its frozen file below
        ``GATE['source_overlap_floor']`` (``retrained_overlap_floor`` for the sources retrained at load). Every
        source's agreement (lowest overlap, largest |diff|, the cases that differ) and the rankers' agreement with the
        record are reported."""
        stored = [r for r in recs if r["pass"] == "stored"]
        g: dict[str, Any] = {"cases": len(stored), "first_stage_labels": getattr(self, "first_stage_labels", None)}

        def ok(v: dict) -> bool:
            return bool(v.get("top200_equivalent", v.get("pool_identical", True)))

        def share(xs: list) -> float | None:
            return float(np.mean(xs)) if xs else None
        g["source_top200"] = share([all(ok(v) for s, v in r["sources"].items() if s not in RETRAINED) for r in stored])
        g["source_top200_retrained"] = share([ok(v) for r in stored for s, v in r["sources"].items() if s in RETRAINED])
        g["sources_differing"] = sorted({s for r in stored for s, v in r["sources"].items() if not ok(v)})
        fs = getattr(self, "fs", None)
        expected = sorted({s for r in stored for s in r["sources"]} | set(fs.order() if hasattr(fs, "order") else []))
        per: dict[str, dict[str, Any]] = {}
        gross: list[str] = []
        for sid in expected:
            vals = [(r.get("row", r.get("query_id")), r["sources"].get(sid)) for r in stored]
            missing = [row for row, v in vals if v is None]
            got = [(row, v) for row, v in vals if v is not None]
            ov = [v["top200_overlap"] / 200 for _, v in got if v.get("top200_overlap") is not None]
            d = [v["max_abs_diff"] for _, v in got if v.get("max_abs_diff") is not None]
            per[sid] = {"min_top200_overlap": min(ov) if ov else None, "max_abs_diff": max(d) if d else None,
                        "cases_differing": [row for row, v in got if not ok(v)], "retrained": sid in RETRAINED}
            floor = GATE["retrained_overlap_floor" if sid in RETRAINED else "source_overlap_floor"]
            if missing:
                per[sid]["cases_missing"] = missing
                gross.append(f"{sid} missing in rows {missing}")
            elif ov and min(ov) < floor:
                gross.append(f"{sid} top-200 overlap {min(ov):.3f} < {floor}")
            elif sid == "colbert_bge" and any(v.get("n_pairs") == 0 for _, v in got):
                gross.append(f"{sid} gave no pairs")
        g["sources"] = per
        for sid in self.systems:
            sy = [(r["query_id"], r["systems"][sid]) for r in stored if sid in r["systems"]]
            ref = self.ref.get(sid, {})
            g[f"{sid}:case_top10_stored"] = share([x["record_top10_equal"] for _, x in sy if x["record_top10_equal"] is not None])
            if sid in RANKERS:
                g[f"{sid}:case_top10_refit"] = share([x["top10"] == x["batch_top10"] for _, x in sy])
                g[f"{sid}:refit_top10"] = share([x["batch_top10"] == list(ref[q]) for q, x in sy if q in ref])
                fd = [x["feature_max_abs_diff_base"] for _, x in sy if x.get("feature_max_abs_diff_base") is not None]
                g[f"{sid}:feature_max_abs_diff_base"] = max(fd) if fd else None
        fails = []
        if "lstar" in self.systems and g.get("lstar:case_top10_stored") not in (None, GATE["lstar_top10"]):
            fails.append("lstar top-10 differs from the record")
        for sid in RANKERS:
            v = g.get(f"{sid}:case_top10_refit")
            if sid in self.rankers and v is not None and v < GATE["case_top10_refit"]:
                fails.append(f"{sid}: per-case top-10 (stored inputs) agrees with the refit's own output on {v:.3f} "
                             f"< {GATE['case_top10_refit']}")
        if gross:
            fails.append(f"gross source failure: {gross}")
        g["fails"] = fails
        g["pass"] = not fails
        return g

    def run(self) -> int:
        from uol.final import cost
        t_all = time.perf_counter()
        self.setup()
        cost.set_threads(THREADS)
        self.log("warm-up")
        warm_recs = [self.score_case(int(r), fresh, warmup=True) for r in self.warm for fresh in (False, True)]
        with (self.out / "warmup_cases.jsonl").open("w", encoding="utf-8") as f:     # kept apart, never summarised
            f.writelines(json.dumps(rec, default=float) + "\n" for rec in warm_recs)
        g = self.gate(warm_recs)
        self.write_json("gate_warmup.json", g)
        if self.warmup_only:
            self.log(f"warm-up only: gate {'pass' if g['pass'] else 'failed'}")
            self.write_json("RESULT.json", {"status": "warmup_only", "gate_warmup": g, "load_seconds": self.load_s,
                                            "fit_seconds": self.fit_s})
            return 0 if g["pass"] else 2
        if not g["pass"]:
            self.log(f"GATE FAILED after warm-up: {g['fails']}")
            self.write_json("RESULT.json", {"status": "gate_failed", "gate_warmup": g, "load_seconds": self.load_s,
                                            "fit_seconds": self.fit_s})
            return 2
        self.cases_path.write_text("", encoding="utf-8")
        recs = []
        for i, r in enumerate(self.rows):
            for fresh in (False, True):
                rec = self.score_case(int(r), fresh)
                recs.append(rec)
                with self.cases_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
            if (i + 1) % 20 == 0:
                self.log(f"{i + 1}/{len(self.rows)} cases")
        gate = self.gate(recs)
        res = summarize(recs, self.sys_sources, self.load_s, self.fit_s, gate, self.gpu)
        res["wall_seconds_total"] = round(time.perf_counter() - t_all, 1)
        res["status"] = "done" if gate["pass"] else "gate_failed"
        res["machine"] = machine()
        self.write_json("RESULT.json", res)
        self.log(f"done: {res['status']}")
        return 0 if gate["pass"] else 2

    def write_json(self, name: str, obj: Any) -> None:
        (self.out / name).write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")


def _recording(fn, sink: dict[int, np.ndarray], want: set[int]):
    """``fn(rows)`` that also keeps the output rows of the timed and warm-up cases (``__wrapped__`` is the original)."""
    def wrapped(rows):
        S = fn(rows)
        for i, r in enumerate(np.asarray(rows)):
            if int(r) in want:
                sink[int(r)] = np.array(S[i], dtype=np.float32)
        return S
    wrapped.__wrapped__ = fn
    return wrapped


def _m1_cfg() -> dict:
    from uol.round3 import common as R3
    sel = json.loads((R3.ARMS / "M1" / "RESULT.json").read_text(encoding="utf-8"))["selected_config"]
    return sel[str(FOLD)]


def machine() -> dict[str, Any]:
    import platform
    out: dict[str, Any] = {"platform": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(),
                           "threads": THREADS}
    try:
        import psutil
        out["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
        out["peak_rss_gb"] = round(psutil.Process().memory_info().rss / 1e9, 2)
    except Exception:                                      # pragma: no cover
        pass
    return out


# --------------------------------------------------------------------------------------------- summary


def summarize(recs: list[dict], sys_sources: dict[str, list[str]], load_s: dict, fit_s: dict, gate: dict,
              gpu: dict | None = None) -> dict[str, Any]:
    """Per stage and per system: median and p95 of wall and CPU seconds per case (fresh pass), plus reproduction.
    Warm-up records never count."""
    recs = [r for r in recs if not r.get("warmup")]
    fresh = [r for r in recs if r["pass"] == "fresh"]
    stored = [r for r in recs if r["pass"] == "stored"]
    names = sorted({k for r in fresh for k in r["wall"]})
    per_entry = {k: {"wall": stats([r["wall"].get(k) for r in fresh]), "cpu": stats([r["cpu"].get(k) for r in fresh])}
                 for k in names}
    by_stage: dict[str, dict] = {}
    for st in STAGE_ORDER:
        ks = [k for k in names if stage_of(k) == st]
        if ks:
            by_stage[st] = {"entries": ks}
    systems: dict[str, Any] = {}
    for sid in sorted({s for r in fresh for s in r["systems"]}):
        ks = [k for k in system_stages(sid, sys_sources) if any(k in r["wall"] for r in fresh)]
        tot_w = [sum(r["wall"].get(k, 0.0) for k in ks) for r in fresh]
        tot_c = [sum(r["cpu"].get(k, 0.0) for k in ks) for r in fresh]
        gpu_keys = [k for k in ks if k in ("encode:bge", "encode:qwen", "encode:ft_e5_q", "encode:casesim_e5")]
        tot_g = [sum(r["wall"].get(k, 0.0) for k in gpu_keys) for r in fresh]
        st_tot: dict[str, Any] = {}
        for st in STAGE_ORDER:
            sk = [k for k in ks if stage_of(k) == st]
            if sk:
                st_tot[st] = {"wall": stats([sum(r["wall"].get(k, 0.0) for k in sk) for r in fresh]),
                              "cpu": stats([sum(r["cpu"].get(k, 0.0) for k in sk) for r in fresh])}
        eq_s = [r["systems"][sid]["record_top10_equal"] for r in stored if sid in r["systems"]]
        eq_f = [r["systems"][sid]["record_top10_equal"] for r in fresh if sid in r["systems"]]
        systems[sid] = {"total_wall": stats(tot_w), "total_cpu": stats(tot_c), "gpu_seconds": stats(tot_g),
                        "by_stage": st_tot, "entries": ks,
                        "top10_equal_record": {"stored_pass": _share(eq_s), "fresh_pass": _share(eq_f)}}
        if sid in RANKERS:                           # the gate's reference: the refit's own batch output
            systems[sid]["top10_equal_refit"] = {
                p: _share([r["systems"][sid]["top10"] == r["systems"][sid].get("batch_top10") for r in rs if sid in r["systems"]])
                for p, rs in (("stored_pass", stored), ("fresh_pass", fresh))}
    fid = {k: stats([r["fidelity"].get(k) for r in fresh]) for k in sorted({k for r in fresh for k in r["fidelity"]})}
    src = {}
    for sid in sorted({s for r in stored for s in r.get("sources", {})}):
        vals = [r["sources"][sid] for r in stored]
        src[sid] = {"top200_identical": _share([v.get("top200_identical", v.get("pool_identical")) for v in vals]),
                    "top200_equivalent": _share([v.get("top200_equivalent", v.get("pool_identical")) for v in vals]),
                    "max_abs_diff": max((v.get("max_abs_diff") or 0.0) for v in vals), "retrained": sid in RETRAINED}
    gpu_meta = {m: g.get("meta") for m, g in (gpu or {}).items()}
    return {"cases": len(fresh), "per_entry": per_entry, "stages": by_stage, "systems": systems, "fidelity": fid,
            "sources_vs_frozen": src, "gate": gate, "load_seconds": load_s, "fit_seconds": fit_s, "gpu_jobs": gpu_meta,
            "llm_stage": "not run: the LLM stage keeps its recorded time (separate row)",
            "note": ("fresh pass: encoders run on the case text; topic columns from a single-case call; M1 neighbours "
                     "from the fresh Qwen3-8B vector. ft_e5_q and the case-similarity encoder: their fold-0 weights "
                     "were not kept, so their query vectors are the stored outputs and their encode time is an "
                     "e5-small forward pass of the same architecture and length (timing_encode e5proxy).")}


def _share(v: Sequence[Any]) -> dict[str, Any]:
    v = [x for x in v if x is not None]
    return {"n": len(v), "equal": int(sum(bool(x) for x in v)), "share": (float(np.mean(v)) if v else None)}


# --------------------------------------------------------------------------------------------- stage-data (thesis PC)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


BUNDLE_JSON = Path(r"bundle\UOL_FINAL_RUN\data\BUNDLE.json")


def stage_data(out: Path, record: Path | None = None, bundle_json: Path = BUNDLE_JSON) -> dict[str, Any]:
    """The inputs the bundle lacks, as development-row arrays in bundle row order (``BUNDLE.json`` row map), packed
    when large, with a MANIFEST of sha256. Local profile (the thesis PC), read only."""
    from uol.embedding_push import colbert, common as ep
    from uol.final import packing
    activate(None, None)
    out = Path(out)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    row_map = np.asarray(ep.dev_rows(), dtype=np.int64)
    if Path(bundle_json).exists():
        b = json.loads(Path(bundle_json).read_text(encoding="utf-8"))
        if not np.array_equal(np.asarray(b["row_map"], dtype=np.int64), row_map):
            raise RuntimeError("the bundle row map differs from the local development rows")
    files: dict[str, Any] = {}

    def target(rel: str) -> Path:
        p = data / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(rel: str, arr: np.ndarray) -> None:
        p = target(rel)
        np.save(p, arr)
        if p.stat().st_size > 8_000_000:
            packing.pack(p)
            packed = p.with_name(p.name + packing.SUFFIX)
            files[rel + packing.SUFFIX] = {"sha256": sha256(packed), "unpacked": rel, "unpacked_sha256": sha256(p)}
            p.unlink()
        else:
            files[rel] = {"sha256": sha256(p)}

    def put_file(rel: str, src: Path) -> None:
        p = target(rel)
        p.write_bytes(Path(src).read_bytes())
        files[rel] = {"sha256": sha256(p)}

    for f in sorted((ep.RUN / "memory_proto" / "state").glob("*.dev_fold0.json")):
        put_file(f"memory_proto/state/{f.name}", f)
    for name in ("lsa_dev_v1.npy", "clsa_dev_v1.npy"):
        put(f"memory_proto/{name}", np.load(ep.RUN / "memory_proto" / name)[row_map])
    for e in (4, 6):
        put(f"memory_proto/casesim/e5_dev_fold0/B/epoch{e}.npy",
            np.load(ep.RUN / "memory_proto" / "casesim" / "e5_dev_fold0" / "B" / f"epoch{e}.npy")[row_map])
    z = np.load(ep.RUN / "finetune_dense" / "cache" / "e5-8b32e14f" / "dev_fold0.npz")
    inv = {int(v): i for i, v in enumerate(row_map)}
    rel = "finetune_dense/cache/e5-8b32e14f/dev_fold0.npz"
    np.savez(target(rel), qvecs=z["qvecs"], rows=np.asarray([inv[int(r)] for r in z["rows"]], dtype=np.int64))
    files[rel] = {"sha256": sha256(data / rel)}
    for name in ("long_articles.npy", "long_cls.npy"):
        put(f"colbert/long/{name}", np.load(colbert.LONG / name))
    sel, mvec = m1_dense_memory()
    np.savez(target("m1/dense_memory.npz"), sel=sel)
    files["m1/dense_memory.npz"] = {"sha256": sha256(data / "m1" / "dense_memory.npz")}
    put("m1/dense_memory_vectors.npy", mvec)
    put_ref(target("reference/reference_top10.json"), record)
    files["reference/reference_top10.json"] = {"sha256": sha256(data / "reference" / "reference_top10.json")}
    code = stage_code(out)
    man = {"code": code, "what": "uol.final.timing inputs the bundle lacks (D137)", "rows": "bundle row order (the development rows)",
           "row_map": "BUNDLE.json row_map (checked equal to the local development rows)", "files": files,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


LAUNCHER = '''"""Run uol.final.timing from this overlay folder: the run computer's code tree stays as its MANIFEST lists it."""
import runpy
import sys
from pathlib import Path

import uol.final

uol.final.__path__.append(str(Path(__file__).resolve().parent / "src" / "uol" / "final"))
sys.argv = ["uol.final.timing"] + sys.argv[1:]
runpy.run_module("uol.final.timing", run_name="__main__")
'''
CODE_FILES = ("timing.py", "timing_sources.py", "gpu_jobs/timing_encode.py")


def stage_code(out: Path) -> dict[str, str]:
    """The timing modules as an overlay (``<out>/src/uol/final``) and the launcher ``run_timing.py``; sha256 each."""
    here = Path(__file__).resolve().parent
    got: dict[str, str] = {}
    for rel in CODE_FILES:
        dst = Path(out) / "src" / "uol" / "final" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((here / rel).read_bytes())
        got[f"src/uol/final/{rel}"] = sha256(dst)
    (Path(out) / "run_timing.py").write_text(LAUNCHER, encoding="utf-8")
    got["run_timing.py"] = sha256(Path(out) / "run_timing.py")
    return got


def m1_dense_memory() -> tuple[np.ndarray, np.ndarray]:
    """(memory.jsonl positions of the usable entries, their unit Qwen3-8B vectors), as ``build_neighbours("dense")``."""
    from uol.embedding_push import qwen_endpoint as Q
    from uol.round2 import b_embed
    from uol.round3 import m1_memory as M1
    mem = M1.memory()
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
    if len(usable) != len(mem["record_id"]):
        raise RuntimeError("memory texts and memory.jsonl differ in length")
    mm = np.memmap(vec_path, dtype=np.float32, mode="r", shape=(complete, Q.DIM))
    sel = np.flatnonzero(usable)
    mvec = Q.unit_rows(np.asarray(mm[np.asarray([rows[keys[i]] for i in sel], dtype=np.int64)]))
    return sel.astype(np.int64), np.ascontiguousarray(mvec, dtype=np.float32)


def ref_from_record(record: Path | None) -> dict[str, Any]:
    """Top-10 of S_full_ext, S0 and lstar per query id from the record's ``top200.parquet`` (DEV)."""
    import pandas as pd
    root = Path(record) if record else Path(__file__).resolve().parents[3] / "experiments" / "runs" / "final_20261025" / RECORD
    out: dict[str, dict[str, list[int]]] = {}
    for sid in SYSTEMS:
        df = pd.read_parquet(root / "runs" / sid / POPULATION / "top200.parquet", columns=["query_id", "rank", "candidate_idx"])
        df = df[df["rank"] <= 10].sort_values(["query_id", "rank"])
        out[sid] = {q: [int(c) for c in g["candidate_idx"]] for q, g in df.groupby("query_id")}
    return {"record": str(root.name), "population": POPULATION, "systems": out}


def put_ref(path: Path, record: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref_from_record(record)), encoding="utf-8")


# --------------------------------------------------------------------------------------------- encode (GPU job driver)


def encode(out: Path, n: int, warmup: int, models: Sequence[str] = GPU_MODELS) -> int:
    """Run the GPU encoders, each in its own process (their pinned stacks differ), on the timed and warm-up cases."""
    rows, warm, _, _ = case_rows(n, warmup)
    enc = Path(out) / "encode"
    enc.mkdir(parents=True, exist_ok=True)
    (enc / "rows.json").write_text(json.dumps({"timed": [int(r) for r in rows], "warmup": [int(r) for r in warm]}),
                                   encoding="utf-8")
    job = Path(__file__).resolve().parent / "gpu_jobs" / "timing_encode.py"
    rc = 0
    for m in models:
        cmd = [sys.executable, str(job), "--model", m, "--rows", str(enc / "rows.json"), "--out", str(enc)]
        print(" ".join(cmd), flush=True)
        r = subprocess.run(cmd)
        if r.returncode:
            print(f"{m}: exit {r.returncode}", flush=True)
            rc = r.returncode
    return rc


def verify(folder: Path) -> dict[str, Any]:
    """Check every sha256 of the staged folder's MANIFEST (code and data), then unpack the packed arrays in place
    (``packing.unpack`` checks the unpacked bytes). Nothing is deleted."""
    from uol.final import packing
    folder = Path(folder)
    man = json.loads((folder / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["code"].items() if sha256(folder / rel) != h]
    bad += [rel for rel, rec in man["files"].items() if sha256(folder / "data" / rel) != rec["sha256"]]
    if bad:
        raise SystemExit(f"sha256 mismatch: {bad}")
    unpacked = []
    for rel, rec in man["files"].items():
        if "unpacked" in rec:
            dst = folder / "data" / rec["unpacked"]
            if not dst.exists():
                packing.unpack(folder / "data" / rel)
            if sha256(dst) != rec["unpacked_sha256"]:
                raise SystemExit(f"unpacked sha256 mismatch: {rec['unpacked']}")
            unpacked.append(rec["unpacked"])
    return {"checked": len(man["code"]) + len(man["files"]), "unpacked": unpacked, "pass": True}


def results_manifest(folder: Path) -> dict[str, Any]:
    """``RESULTS_MANIFEST.json``: sha256 and size of every file of a timing output folder."""
    folder = Path(folder)
    files = {p.relative_to(folder).as_posix(): {"sha256": sha256(p), "bytes": p.stat().st_size}
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != "RESULTS_MANIFEST.json"}
    man = {"folder": folder.name, "files": files, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (folder / "RESULTS_MANIFEST.json").write_text(json.dumps(man, indent=1), encoding="utf-8")
    return man


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.timing", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--store", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stage-data")
    p.add_argument("--out", required=True)
    p.add_argument("--record", default=None)
    for name in ("encode", "run"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--cases", type=int, default=N_CASES)
        p.add_argument("--warmup", type=int, default=N_WARMUP)
        if name == "encode":
            p.add_argument("--models", default=",".join(GPU_MODELS))
        else:
            p.add_argument("--encodings", default=None, help="folder of the encode step (default <out>/encode)")
            p.add_argument("--colbert", default="store", choices=["store", "frozen", "off"])
            p.add_argument("--systems", default=",".join(SYSTEMS))
            p.add_argument("--sources-only", action="store_true", help="dry run: sources and lstar, no ranker fit")
            p.add_argument("--sources", default=None, help="with --sources-only: source ids (default all 50 + colbert_bge)")
            p.add_argument("--warmup-only", action="store_true", help="stop after the warm-up gate (no timed cases)")
    p = sub.add_parser("summary")
    p.add_argument("--out", required=True)
    p = sub.add_parser("verify")
    p.add_argument("--dir", required=True)
    p = sub.add_parser("results-manifest")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "stage-data":
        print(json.dumps(stage_data(Path(a.out), Path(a.record) if a.record else None)["files"], indent=1))
        return 0
    if a.cmd == "verify":
        print(json.dumps(verify(Path(a.dir))))
        return 0
    if a.cmd == "results-manifest":
        print(len(results_manifest(Path(a.out))["files"]), "files")
        return 0
    activate(a.data, a.store)
    if a.cmd == "encode":
        return encode(Path(a.out), a.cases, a.warmup, a.models.split(","))
    if a.cmd == "run":
        enc = Path(a.encodings) if a.encodings else Path(a.out) / "encode"
        from uol.final import timing_sources as TS
        srcs = a.sources.split(",") if a.sources else (TS.FULL_ORDER + TS.PAIR_SOURCES if a.sources_only else None)
        r = Run(Path(a.out), a.cases, a.warmup, enc if enc.exists() else None, a.colbert, a.systems.split(","),
                sources_only=a.sources_only, sources=srcs, warmup_only=a.warmup_only)
        return r.run()
    recs = [json.loads(x) for x in (Path(a.out) / "cases.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    res = json.loads((Path(a.out) / "RESULT.json").read_text(encoding="utf-8"))
    print(json.dumps({s: {"median_s": v["total_wall"]["median"], "p95_s": v["total_wall"]["p95"]}
                      for s, v in res["systems"].items()}, indent=1))
    return 0 if recs else 1


if __name__ == "__main__":
    sys.exit(main())
