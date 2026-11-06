"""EP-alpha phase 1: cross-fitted base score files for the embedding push.

Method ids written to ``scores/<id>/scores.npy`` (float32 ``[5980, 39178]``, ``common.create_scores``):

``L_star`` (tuned lexical, label-free), ``prior``, ``knn_lex_votes``, ``K_lex``, ``LP``,
``dense_bge_maxp``, ``dense_bge_firstp``, ``dense_e5_maxp`` (frozen dense, label-free) and ``H_bge``.

Every scoring group of ``common.scoring_groups()`` (five dev folds and the lockbox) is scored by a model that
used only that group's training rows. Hyper-parameters are selected by nested inner cross-validation: the inner
folds are the group's training folds, each inner fold is scored with labels from the other training folds, and the
objective is the pooled component-macro P_spec nDCG@10 (ties: lowest grid index, tolerance 1e-12) exactly as in
``uol.facts_to_law.crossfit``. Systems and grids replicate facts-to-law v2 (``uol.facts_to_law.run.default_systems``
and ``systems.bge_systems``); the lockbox never trains, selects or gets a metric.

Run: ``python -m uol.embedding_push.base lexical|dense|nested|summary|all [--force]``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common
from uol.facts_to_law import fusion, knn, lexical, metrics
from uol.facts_to_law import prior as prior_mod
from uol.facts_to_law import run as ftl_run
from uol.facts_to_law import spec as ftl_spec

OUT = common.RUN / "base"
SEARCH_LOG = OUT / "search_log.jsonl"
RUN_LOG = OUT / "run.log"
SELECTION_DIR = OUT / "selection"

METRIC_KEYS = metrics.METRICS            # ndcg_at_10, mrr, recall_at_20, recall_at_100
BM25_CFG = [(float(c["k1"]), float(c["b"])) for c in lexical.bm25_grid()]       # k1 major, b minor (6)
W_CHAR, W_PRIOR, W_DENSE, KNN_K, W_KNN = ftl_run.W_CHAR, ftl_run.W_PRIOR, ftl_run.W_DENSE, ftl_run.KNN_K, ftl_run.W_KNN
RRF_K = ftl_run.RRF_K
TOL = 1e-12
BLOCK = 256
BASE_METHODS = ("L_star", "prior", "knn_lex_votes", "K_lex", "LP", "dense_bge_maxp", "dense_bge_firstp",
                "dense_e5_maxp", "H_bge")
CODE_FILES = [Path(__file__), Path(common.__file__)] + [Path(m.__file__) for m in (lexical, knn, prior_mod, fusion, metrics, ftl_run)]

BGE_DIR = common.DERIVED / "bge_m3_chunks_v1"
E5_DIR = common.DERIVED / "e5_small_chunks_v1"
E5_QUERIES = common.DERIVED / "e5_small_queries_v11/query_vectors.npy"


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc()}] {message}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def search_log(entry: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with SEARCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def reset_search_log(stage: str) -> None:
    """Drop earlier entries of one stage (re-runs), keeping the other stages' entries."""
    if not SEARCH_LOG.exists():
        return
    keep = [line for line in SEARCH_LOG.read_text(encoding="utf-8").splitlines() if json.loads(line).get("stage") != stage]
    SEARCH_LOG.write_text("".join(line + "\n" for line in keep), encoding="utf-8")


def pair(w: float) -> list[float]:
    """Two-channel weights (1 - w, w) exactly as ``uol.facts_to_law.run._pairs``."""
    return [round(1.0 - float(w), 10), float(w)]


def argbest(values: Sequence[float]) -> int:
    """Lowest index among values within TOL of the maximum (crossfit tie rule)."""
    best = max(values)
    return min(i for i, v in enumerate(values) if v >= best - TOL)


# --------------------------------------------------------------------------- context


@dataclass
class Group:
    idx: int
    role: str
    rows: np.ndarray
    train_rows: np.ndarray
    train_folds: tuple[int, ...]

    @property
    def is_lockbox(self) -> bool:
        return self.role == "lockbox"


class Ctx:
    """Release, split, populations and per-query metric helpers (dev rows only)."""

    def __init__(self) -> None:
        d = self.data = common.release()
        self.N = len(d.candidate_ids)
        assert np.array_equal(common.tie_order(), np.arange(self.N)), "index order must equal candidate-ID order"
        roles = common.roles()
        self.fold = np.asarray([-1 if r == "lockbox" else int(r.replace("dev_fold", "")) for r in roles])
        qfold = np.asarray([q.fold for q in d.queries])
        dev = self.fold >= 0
        assert (qfold[dev] == self.fold[dev]).all(), "dev folds must equal the v11 folds"
        self.dev, self.lock = common.dev_rows(), common.lockbox_rows()
        comp_ids = sorted({q.component_id for q in d.queries})
        comp_index = {c: i for i, c in enumerate(comp_ids)}
        self.comp = np.asarray([comp_index[q.component_id] for q in d.queries])
        self.ncomp = len(comp_ids)
        self.U = common.population_rows("P", self.dev)          # dev rows of population P (P_spec is a subset)
        self.nU = len(self.U)
        self.foldU = self.fold[self.U]
        self.cU = self.comp[self.U]
        self.pos_in_U = {int(r): i for i, r in enumerate(self.U)}
        self.groups = []
        for i, (role, rows, train) in enumerate(common.scoring_groups()):
            self.groups.append(Group(i, role, np.asarray(rows), np.asarray(train),
                                     tuple(sorted(set(int(f) for f in self.fold[train])))))
        self.texts = [q.text for q in d.queries]
        self.generic = d.generic_idx if len(d.generic_idx) else None
        self._lex: dict[str, lexical.LexicalIndex] = {}

    # lexical indexes (label-free, built once)
    def lex(self, kind: str) -> lexical.LexicalIndex:
        if kind not in self._lex:
            t = time.perf_counter()
            self._lex[kind] = lexical.LexicalIndex(self.data.candidate_texts, kind=kind)
            log(f"lexical index {kind} built in {time.perf_counter() - t:.1f}s")
        return self._lex[kind]

    def query_metrics(self, scores: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        """``[len(rows), 2, 4]`` per-query metrics: view 0 full reference, view 1 specific condensed list.

        Dev rows only. The specific view is NaN for rows without a specific positive. Ranking follows the shared
        score-descending, candidate-index-ascending rule (identical to the candidate-ID order, asserted above).
        """
        rows = np.asarray(rows, dtype=np.int64)
        if np.isin(rows, self.lock).any():
            raise PermissionError("lockbox rows never receive metrics")
        d = self.data
        out = np.full((len(rows), 2, 4), np.nan)
        res = metrics.evaluate_matrix(scores, [d.positive_idx[int(r)] for r in rows], exclude_idx=None, top_k=0)
        out[:, 0, :] = [[x[m] for m in METRIC_KEYS] for x in res]
        loc = [j for j, r in enumerate(rows) if d.specific_idx[int(r)]]
        if loc:
            res = metrics.evaluate_matrix(scores[loc], [d.specific_idx[int(rows[j])] for j in loc],
                                          exclude_idx=self.generic, top_k=0)
            out[loc, 1, :] = [[x[m] for m in METRIC_KEYS] for x in res]
        return out

    def pooled(self, values_u: np.ndarray, mask_u: np.ndarray) -> float:
        """Component-macro mean of per-query values over U positions in ``mask_u`` (NaN values dropped)."""
        ok = mask_u & np.isfinite(values_u)
        if not ok.any():
            return float("nan")
        codes = self.cU[ok]
        sums = np.bincount(codes, weights=values_u[ok], minlength=self.ncomp)
        cnt = np.bincount(codes, minlength=self.ncomp)
        m = cnt > 0
        return float((sums[m] / cnt[m]).mean())

    def objective(self, table: np.ndarray, train_u: np.ndarray, population: str = "P_spec", metric: int = 0) -> float:
        view = 1 if population == "P_spec" else 0
        return self.pooled(table[:, view, metric], train_u)

    def train_mask_u(self, g: Group) -> np.ndarray:
        return np.isin(self.foldU, g.train_folds)

    def blocks(self, n: int, size: int = BLOCK):
        for s in range(0, n, size):
            yield slice(s, min(s + size, n))


def inputs_hashes() -> dict[str, str]:
    return {"split_sha256": common.sha256_file(common.SPLIT_PATH),
            "release_manifest_sha256": common.sha256_file(common.RELEASE / "manifest.json")}


# --------------------------------------------------------------------------- lexical stage


@dataclass(frozen=True)
class LSel:
    """One L* choice: a BM25 variant, character TF-IDF, or their robust-z fusion at ``w_char``."""

    kind: str                 # bm25 | bm25_stem | char | fuse
    base: str | None = None   # bm25 | bm25_stem (the BM25 channel of a fusion, or the kind itself)
    k1: float | None = None
    b: float | None = None
    w_char: float | None = None

    @property
    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @staticmethod
    def from_json(row: dict[str, Any]) -> "LSel":
        return LSel(**row)


def lstar_block(ctx: Ctx, sel: LSel, texts: list[str]) -> np.ndarray:
    """L* scores of query texts under a selection (float32, same operators as facts-to-law v2)."""
    if sel.kind in ("bm25", "bm25_stem"):
        return ctx.lex(sel.kind).score(texts, {"k1": sel.k1, "b": sel.b})
    if sel.kind == "char":
        return ctx.lex("char").score(texts)
    bm = fusion.robust_z(ctx.lex(sel.base).score(texts, {"k1": sel.k1, "b": sel.b}))
    ch = fusion.robust_z(ctx.lex("char").score(texts))
    return fusion.linear_fusion([bm, ch], pair(sel.w_char), normalize="none")


def lstar_rows(ctx: Ctx, sel: LSel, rows: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    res = np.empty((len(rows), ctx.N), dtype=np.float32) if out is None else out
    for sl in ctx.blocks(len(rows)):
        res[sl] = lstar_block(ctx, sel, [ctx.texts[int(r)] for r in rows[sl]])
    return res


def lexical_tables(ctx: Ctx, max_rows: int | None = None) -> dict[Any, np.ndarray]:
    """Per-query metric tables over U for 12 BM25 configs, char and 60 fused (BM25 config x w_char) variants.

    ``max_rows`` (smoke tests only) limits the U positions that are scored; the rest stay NaN.
    """
    keys: list[Any] = [(kind, k1, b) for kind in ("bm25", "bm25_stem") for (k1, b) in BM25_CFG] + ["char"]
    keys += [("fuse", kind, k1, b, w) for kind in ("bm25", "bm25_stem") for (k1, b) in BM25_CFG for w in W_CHAR]
    tables = {k: np.full((ctx.nU, 2, 4), np.nan) for k in keys}
    t0 = time.perf_counter()
    for sl in ctx.blocks(ctx.nU if max_rows is None else min(max_rows, ctx.nU)):
        rows = ctx.U[sl]
        texts = [ctx.texts[int(r)] for r in rows]
        char = ctx.lex("char").score(texts)
        tables["char"][sl] = ctx.query_metrics(char, rows)
        zc = fusion.robust_z(char)
        for kind in ("bm25", "bm25_stem"):
            index = ctx.lex(kind)
            for (k1, b) in BM25_CFG:
                raw = index.score(texts, {"k1": k1, "b": b})
                tables[(kind, k1, b)][sl] = ctx.query_metrics(raw, rows)
                zr = fusion.robust_z(raw)
                for w in W_CHAR:
                    fused = fusion.linear_fusion([zr, zc], pair(w), normalize="none")
                    tables[("fuse", kind, k1, b, w)][sl] = ctx.query_metrics(fused, rows)
        log(f"lexical tables: rows {sl.stop}/{ctx.nU} ({time.perf_counter() - t0:.0f}s)")
    return tables


def select_lstar(ctx: Ctx, tables: dict[Any, np.ndarray], g: Group) -> tuple[LSel, dict[str, Any]]:
    """Hierarchical L* selection of facts-to-law v2 (L_bm25, L_bm25_stem -> L_bm25_best; L_fuse; L_star)."""
    train = ctx.train_mask_u(g)
    obj = lambda t: ctx.objective(t, train)  # noqa: E731
    ev: dict[str, Any] = {}
    chosen: dict[str, tuple[Any, float]] = {}
    for kind in ("bm25", "bm25_stem"):
        vals = [obj(tables[(kind, k1, b)]) for (k1, b) in BM25_CFG]
        i = argbest(vals)
        chosen[kind] = ((kind,) + BM25_CFG[i], vals[i])
        ev[f"L_{kind}"] = {"grid": [{"k1": k1, "b": b, "objective": v} for (k1, b), v in zip(BM25_CFG, vals)], "chosen": i}
    pick = argbest([chosen["bm25"][1], chosen["bm25_stem"][1]])
    base_key = chosen["bm25" if pick == 0 else "bm25_stem"][0]
    ev["L_bm25_best"] = {"chosen": ["bm25", "bm25_stem"][pick], "objectives": [chosen["bm25"][1], chosen["bm25_stem"][1]]}
    fvals = [obj(tables[("fuse",) + base_key + (w,)]) for w in W_CHAR]
    fi = argbest(fvals)
    ev["L_fuse"] = {"grid": [{"w_char": w, "objective": v} for w, v in zip(W_CHAR, fvals)], "chosen": fi}
    char_v = obj(tables["char"])
    options = [chosen["bm25"][1], chosen["bm25_stem"][1], char_v, fvals[fi]]
    star = argbest(options)
    ev["L_star"] = {"options": dict(zip(["L_bm25", "L_bm25_stem", "L_char", "L_fuse"], options)), "chosen": star}
    kind, k1, b = base_key
    sel = [LSel("bm25", "bm25", *chosen["bm25"][0][1:]), LSel("bm25_stem", "bm25_stem", *chosen["bm25_stem"][0][1:]),
           LSel("char"), LSel("fuse", kind, k1, b, W_CHAR[fi])][star]
    return sel, ev


def stage_lexical(force: bool = False) -> dict[str, Any]:
    ctx = Ctx()
    path = SELECTION_DIR / "lexical.json"
    if path.exists() and common.score_path("L_star").exists() and not force:
        log("lexical stage already complete")
        return json.loads(path.read_text(encoding="utf-8"))
    t0 = time.perf_counter()
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    reset_search_log("lexical")
    tables = lexical_tables(ctx)
    save_tables(tables, OUT / "tables_lexical.npz")
    selection: dict[str, Any] = {}
    for g in ctx.groups:
        sel, ev = select_lstar(ctx, tables, g)
        selection[g.role] = {"selected": asdict(sel), "evidence": ev, "train_folds": list(g.train_folds)}
        log(f"{g.role}: L* = {sel}")
    # search log: every configuration, pooled over all dev rows (label-free, so no fold dependence) and per group objective
    allu = np.ones(ctx.nU, dtype=bool)
    for key, table in tables.items():
        search_log({"stage": "lexical", "config": list(key) if isinstance(key, tuple) else key,
                    "dev_P_spec_ndcg10": ctx.objective(table, allu), "dev_P_ndcg10": ctx.objective(table, allu, "P"),
                    "dev_P_spec_mrr": ctx.objective(table, allu, "P_spec", 1), "dev_P_spec_r20": ctx.objective(table, allu, "P_spec", 2),
                    "n_P_spec_dev": int(np.isfinite(table[:, 1, 0]).sum())})
    if common.score_path("L_star").parent.exists():
        shutil.rmtree(common.score_path("L_star").parent)
    matrix = common.create_scores("L_star")
    for g in ctx.groups:
        sel = LSel.from_json(selection[g.role]["selected"])
        lstar_rows(ctx, sel, g.rows, out=_MemmapRows(matrix, g.rows))
        log(f"L_star written for {g.role} ({len(g.rows)} rows) {time.perf_counter() - t0:.0f}s")
    matrix.flush()
    del matrix
    manifest = {"method": "L_star", "family": "NO-EMB", "label_free": True,
                "description": "tuned lexical retrieval: per scoring group the best of BM25 grid, stemmed BM25, character TF-IDF "
                               "and their robust-z fusion (w_char grid), selected by inner CV on the group's training dev rows",
                "training_data": "none (label-free scoring); hyper-parameters selected on the group's training rows only "
                                 "(pooled component-macro P_spec nDCG@10; lockbox group: all dev rows)",
                "configuration": {g: selection[g]["selected"] for g in selection},
                "selection_evidence": f"{path}", "grids": {"bm25": [list(c) for c in BM25_CFG], "w_char": list(W_CHAR)},
                "seeds": None, "input_sha256": inputs_hashes(), "wall_seconds": time.perf_counter() - t0, "device": "cpu"}
    common.finish_scores("L_star", manifest, CODE_FILES)
    path.write_text(json.dumps(selection, indent=1), encoding="utf-8")
    log(f"L_star complete in {time.perf_counter() - t0:.0f}s")
    return selection


class _MemmapRows:
    """Assignable view: ``view[sl] = block`` writes rows ``rows[sl]`` of a memmap."""

    def __init__(self, matrix: np.ndarray, rows: np.ndarray) -> None:
        self.matrix, self.rows = matrix, np.asarray(rows)
        self.shape = (len(rows), matrix.shape[1])

    def __setitem__(self, sl: slice, block: np.ndarray) -> None:
        self.matrix[self.rows[sl]] = block


def save_tables(tables: dict[Any, np.ndarray], path: Path) -> None:
    np.savez_compressed(path, **{json.dumps(k if not isinstance(k, tuple) else list(k)): v for k, v in tables.items()})


# --------------------------------------------------------------------------- dense stage


def _write_dense(method_id: str, producer, manifest_extra: dict[str, Any], force: bool) -> None:
    if common.score_path(method_id).exists():
        if not force and (common.score_path(method_id).parent / "manifest.json").exists():
            log(f"{method_id} already complete")
            return
        shutil.rmtree(common.score_path(method_id).parent)
    t0 = time.perf_counter()
    matrix = common.create_scores(method_id)
    producer(matrix)
    matrix.flush()
    del matrix
    manifest = {"method": method_id, "label_free": True, "training_data": "none (frozen encoder, label-free)",
                "seeds": None, "input_sha256": inputs_hashes(), "wall_seconds": time.perf_counter() - t0, "device": "cpu (BLAS)",
                **manifest_extra}
    common.finish_scores(method_id, manifest, CODE_FILES)
    log(f"{method_id} complete in {time.perf_counter() - t0:.0f}s")


def stage_dense(force: bool = False) -> None:
    n = len(common.release().candidate_ids)
    bge_q = np.load(BGE_DIR / "query_vectors.npy")
    bge_c = np.load(BGE_DIR / "chunk_vectors.npy")
    bge_owner = common.chunk_article_index(BGE_DIR / "units.jsonl")
    units = [json.loads(line) for line in (BGE_DIR / "units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    chunk0 = np.asarray([u["chunk_index"] for u in units])
    bge_files = {p.name: common.sha256_file(p) for p in (BGE_DIR / "chunk_vectors.npy", BGE_DIR / "query_vectors.npy", BGE_DIR / "units.jsonl")}

    def maxp(q, c, owner):
        def producer(matrix):
            for start, block in common.maxp_scores(q, c, owner, n):
                matrix[start:start + len(block)] = block
        return producer

    _write_dense("dense_bge_maxp", maxp(bge_q, bge_c, bge_owner),
                 {"family": "EMB", "description": "bge-m3 CLS vectors, max over 1,024-token chunks of cosine similarity",
                  "configuration": {"aggregation": "maxp", "vectors": str(BGE_DIR)}, "frozen_vector_sha256": bge_files}, force)

    order = np.argsort(bge_owner, kind="stable")
    first_pos = order[np.flatnonzero(np.r_[True, bge_owner[order][1:] != bge_owner[order][:-1]])]
    assert (chunk0[first_pos] == 0).all() and len(first_pos) == n and (bge_owner[first_pos] == np.arange(n)).all()

    def firstp(matrix):
        vecs = bge_c[first_pos]
        for s in range(0, len(bge_q), 512):
            matrix[s:s + 512] = bge_q[s:s + 512] @ vecs.T

    _write_dense("dense_bge_firstp", firstp,
                 {"family": "EMB", "description": "bge-m3 chunk 0 (first 1,024 tokens) cosine similarity",
                  "configuration": {"aggregation": "firstp", "vectors": str(BGE_DIR)}, "frozen_vector_sha256": bge_files}, force)
    del bge_c

    e5_c = np.load(E5_DIR / "chunk_vectors.npy")
    e5_q = np.load(E5_QUERIES)
    ids = json.loads((E5_DIR / "ids.json").read_text(encoding="utf-8"))["candidate_ids"]
    assert ids == common.release().candidate_ids, "E5 article_index must index the bound catalogue order"
    e5_owner = np.load(E5_DIR / "article_index.npy").astype(np.int64)
    assert e5_c.shape[0] == len(e5_owner) and e5_q.shape == (len(common.release().queries), e5_c.shape[1])
    e5_norm = (float(np.abs(np.linalg.norm(e5_c[:2000], axis=1) - 1).max()), float(np.abs(np.linalg.norm(e5_q, axis=1) - 1).max()))
    _write_dense("dense_e5_maxp", maxp(e5_q, e5_c, e5_owner),
                 {"family": "EMB", "description": "multilingual-e5-small, max over chunks of cosine similarity",
                  "configuration": {"aggregation": "maxp", "vectors": str(E5_DIR), "article_index": "catalogue position of each chunk (verified against ids.json)",
                                    "max_norm_deviation_chunks_queries": e5_norm},
                  "frozen_vector_sha256": {p.name: common.sha256_file(p) for p in (E5_DIR / "chunk_vectors.npy", E5_DIR / "article_index.npy", E5_QUERIES)}},
                 force)


# --------------------------------------------------------------------------- nested (label-using) stage


class Nested:
    """Memories, priors and similarity for scoring rows with labels from a set of fit folds (cached)."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self._mem: dict[tuple[int, ...], tuple[knn.Memory, lexical.LexicalIndex]] = {}
        self._prior: dict[tuple[int, ...], np.ndarray] = {}

    def fit_query_ids(self, fit: tuple[int, ...]) -> list[str]:
        d = self.ctx.data
        return [q.query_id for q, f in zip(d.queries, self.ctx.fold) if f in fit]

    def memory(self, fit: tuple[int, ...]):
        if fit not in self._mem:
            d = self.ctx.data
            mem = knn.build_memory(self.fit_query_ids(fit), d.component_by_query, d.references, d.candidate_index)
            texts = [d.queries[d.query_index[q]].text for q in mem.query_ids]
            self._mem[fit] = (mem, lexical.LexicalIndex(texts, kind="bm25"))
        return self._mem[fit]

    def prior(self, fit: tuple[int, ...]) -> np.ndarray:
        if fit not in self._prior:
            d = self.ctx.data
            self._prior[fit] = prior_mod.fitting_prior(d.references, d.component_by_query, self.fit_query_ids(fit),
                                                        self.ctx.N, d.candidate_index)
        return self._prior[fit]

    def votes(self, rows: np.ndarray, fit: tuple[int, ...], ks: Sequence[int]) -> dict[int, np.ndarray]:
        mem, index = self.memory(fit)
        d = self.ctx.data
        sim = index.score([self.ctx.texts[int(r)] for r in rows])
        comps = [d.queries[int(r)].component_id for r in rows]
        return {int(k): knn.knn_votes(sim, mem.positives, int(k), self.ctx.N, "sim", eval_components=comps,
                                      fit_components=list(mem.components)) for k in ks}


class UStore:
    """Raw and robust-z L* / dense scores of the U rows (dev population P) for each distinct L* selection."""

    def __init__(self, ctx: Ctx) -> None:
        self.ctx = ctx
        self.raw: dict[str, np.ndarray] = {}
        self.z: dict[str, np.ndarray] = {}
        self._dense: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def lstar(self, sel: LSel) -> tuple[np.ndarray, np.ndarray]:
        if sel.key not in self.raw:
            t = time.perf_counter()
            raw = lstar_rows(self.ctx, sel, self.ctx.U)
            self.raw[sel.key] = raw
            self.z[sel.key] = fusion.robust_z(raw)
            log(f"L* U-rows for {sel} in {time.perf_counter() - t:.0f}s")
        return self.raw[sel.key], self.z[sel.key]

    def dense(self, method_id: str) -> tuple[np.ndarray, np.ndarray]:
        if method_id not in self._dense:
            raw = np.asarray(common.open_scores(method_id)[self.ctx.U])
            self._dense[method_id] = (raw, fusion.robust_z(raw))
        return self._dense[method_id]


def h_tables(ctx: Ctx, store: UStore, sel: LSel, dense_id: str = "dense_bge_maxp") -> dict[str, np.ndarray]:
    """Label-free H tables over U for one L* selection: 'rrf' and 'lin<j>' for each w_dense."""
    lraw, lz = store.lstar(sel)
    draw, dz = store.dense(dense_id)
    tabs = {"rrf": np.full((ctx.nU, 2, 4), np.nan)}
    tabs.update({f"lin{j}": np.full((ctx.nU, 2, 4), np.nan) for j in range(len(W_DENSE))})
    for sl in ctx.blocks(ctx.nU):
        rows = ctx.U[sl]
        rrf = fusion.rrf([lraw[sl], draw[sl]], k=RRF_K).astype(np.float32)
        tabs["rrf"][sl] = ctx.query_metrics(rrf, rows)
        for j, w in enumerate(W_DENSE):
            tabs[f"lin{j}"][sl] = ctx.query_metrics(fusion.linear_fusion([lz[sl], dz[sl]], pair(w), normalize="none"), rows)
    return tabs


def select_h(ctx: Ctx, tabs: dict[str, np.ndarray], g: Group) -> tuple[dict[str, Any], dict[str, Any]]:
    train = ctx.train_mask_u(g)
    rrf_v = ctx.objective(tabs["rrf"], train)
    lin_v = [ctx.objective(tabs[f"lin{j}"], train) for j in range(len(W_DENSE))]
    j = argbest(lin_v)
    pick = argbest([rrf_v, lin_v[j]])       # channel order of H_star: (H_rrf, H_lin)
    ev = {"rrf": rrf_v, "lin": [{"w_dense": w, "objective": v} for w, v in zip(W_DENSE, lin_v)], "chosen_lin": j,
          "pick": ["rrf", "lin"][pick]}
    return ({"type": "rrf", "k": RRF_K} if pick == 0 else {"type": "linear", "w_dense": W_DENSE[j]}), ev


def stage_nested(force: bool = False) -> None:
    ctx = Ctx()
    t_all = time.perf_counter()
    lex_sel = json.loads((SELECTION_DIR / "lexical.json").read_text(encoding="utf-8"))
    lsel = {g.role: LSel.from_json(lex_sel[g.role]["selected"]) for g in ctx.groups}
    nested, store = Nested(ctx), UStore(ctx)
    ng, nk, nw = len(ctx.groups), len(KNN_K), len(W_KNN)
    # inner tables: [group, config..., U, 2, 4]
    tv = np.full((ng, nk, ctx.nU, 2, 4), np.nan)
    tk = np.full((ng, nk * nw, ctx.nU, 2, 4), np.nan)
    tlp = np.full((ng, len(W_PRIOR), ctx.nU, 2, 4), np.nan)
    for g in ctx.groups:
        sel = lsel[g.role]
        _, zl_all = store.lstar(sel)
        for inner in g.train_folds:
            t = time.perf_counter()
            fit = tuple(f for f in g.train_folds if f != inner)
            idx = np.flatnonzero(ctx.foldU == inner)
            rows = ctx.U[idx]
            votes = nested.votes(rows, fit, KNN_K)
            zl = zl_all[idx]
            for ki, k in enumerate(KNN_K):
                tv[g.idx, ki, idx] = ctx.query_metrics(votes[k], rows)
                zv = fusion.robust_z(votes[k])
                for wi, w in enumerate(W_KNN):
                    fused = fusion.linear_fusion([zl, zv], pair(w), normalize="none")
                    tk[g.idx, ki * nw + wi, idx] = ctx.query_metrics(fused, rows)
            zp = fusion.robust_z(nested.prior(fit))
            for wi, w in enumerate(W_PRIOR):
                tlp[g.idx, wi, idx] = ctx.query_metrics(fusion.add_prior(zl, zp, w, normalize="none") if w else zl, rows)
            log(f"inner task {g.role}: fold {inner} scored with fit folds {fit} ({len(rows)} rows, {time.perf_counter() - t:.0f}s)")
    np.savez_compressed(OUT / "tables_nested.npz", votes=tv, k_lex=tk, lp=tlp)

    # label-free H tables per distinct L* selection
    h_by_sel = {}
    for g in ctx.groups:
        if lsel[g.role].key not in h_by_sel:
            h_by_sel[lsel[g.role].key] = h_tables(ctx, store, lsel[g.role])
            log(f"H tables built for {lsel[g.role]}")

    selection: dict[str, Any] = {}
    reset_search_log("nested")
    for g in ctx.groups:
        train = ctx.train_mask_u(g)
        # knn_lex_votes alone: k by votes-alone objective
        v_obj = [ctx.objective(tv[g.idx, ki], train) for ki in range(nk)]
        v_best = argbest(v_obj)
        # K_lex: (k, w) jointly, k major
        k_obj = [ctx.objective(tk[g.idx, ci], train) for ci in range(nk * nw)]
        k_best = argbest(k_obj)
        lp_obj = [ctx.objective(tlp[g.idx, wi], train) for wi in range(len(W_PRIOR))]
        lp_best = argbest(lp_obj)
        h_cfg, h_ev = select_h(ctx, h_by_sel[lsel[g.role].key], g)
        selection[g.role] = {
            "train_folds": list(g.train_folds), "L_star": asdict(lsel[g.role]),
            "knn_lex_votes": {"k": KNN_K[v_best], "grid": [{"k": k, "objective": v} for k, v in zip(KNN_K, v_obj)]},
            "K_lex": {"k": KNN_K[k_best // nw], "w_knn": W_KNN[k_best % nw],
                      "grid": [{"k": KNN_K[ci // nw], "w_knn": W_KNN[ci % nw], "objective": k_obj[ci],
                                "objective_P": ctx.objective(tk[g.idx, ci], train, "P")} for ci in range(nk * nw)]},
            "LP": {"w_prior": W_PRIOR[lp_best], "grid": [{"w_prior": w, "objective": v} for w, v in zip(W_PRIOR, lp_obj)]},
            "H_bge": {"config": h_cfg, "evidence": h_ev}}
        for name, grid in (("knn_lex_votes", selection[g.role]["knn_lex_votes"]["grid"]), ("K_lex", selection[g.role]["K_lex"]["grid"]),
                           ("LP", selection[g.role]["LP"]["grid"])):
            for cfg in grid:
                search_log({"stage": "nested", "method": name, "group": g.role, "config": {k: v for k, v in cfg.items() if k != "objective"},
                            "inner_P_spec_ndcg10": cfg["objective"]})
        search_log({"stage": "nested", "method": "H_bge", "group": g.role, "evidence": h_ev})
        log(f"{g.role}: votes k={KNN_K[v_best]}; K_lex k={KNN_K[k_best // nw]} w={W_KNN[k_best % nw]}; "
            f"LP w={W_PRIOR[lp_best]}; H_bge {h_cfg}")
    (SELECTION_DIR / "nested.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    del tv, tk, tlp, store

    # outer scoring of all rows of each group with the selected configurations
    methods = ("prior", "knn_lex_votes", "K_lex", "LP", "H_bge")
    for m in methods:
        if common.score_path(m).parent.exists():
            shutil.rmtree(common.score_path(m).parent)
    mats = {m: common.create_scores(m) for m in methods}
    lstar_file = common.open_scores("L_star")
    dense_file = common.open_scores("dense_bge_maxp")
    for g in ctx.groups:
        sel = selection[g.role]
        fit = g.train_folds
        pv = nested.prior(fit)
        zp = fusion.robust_z(pv)
        mats["prior"][g.rows] = pv[None, :]
        ks = sorted({sel["knn_lex_votes"]["k"], sel["K_lex"]["k"]})
        for sl in ctx.blocks(len(g.rows), 512):
            rows = g.rows[sl]
            votes = nested.votes(rows, fit, ks)
            lraw = np.asarray(lstar_file[rows])
            zl = fusion.robust_z(lraw)
            mats["knn_lex_votes"][rows] = votes[sel["knn_lex_votes"]["k"]]
            zv = fusion.robust_z(votes[sel["K_lex"]["k"]])
            mats["K_lex"][rows] = fusion.linear_fusion([zl, zv], pair(sel["K_lex"]["w_knn"]), normalize="none")
            wp = sel["LP"]["w_prior"]
            mats["LP"][rows] = fusion.add_prior(zl, zp, wp, normalize="none") if wp else zl
            draw = np.asarray(dense_file[rows])
            if sel["H_bge"]["config"]["type"] == "rrf":
                mats["H_bge"][rows] = fusion.rrf([lraw, draw], k=RRF_K).astype(np.float32)
            else:
                mats["H_bge"][rows] = fusion.linear_fusion([zl, fusion.robust_z(draw)],
                                                           pair(sel["H_bge"]["config"]["w_dense"]), normalize="none")
        log(f"outer scoring {g.role}: {len(g.rows)} rows done ({time.perf_counter() - t_all:.0f}s)")
    for m in methods:
        mats[m].flush()
    del mats
    wall = time.perf_counter() - t_all
    common_manifest = {"seeds": None, "input_sha256": inputs_hashes(), "wall_seconds": wall, "device": "cpu",
                       "selection_evidence": str(SELECTION_DIR / "nested.json"), "search_log": str(SEARCH_LOG),
                       "inner_cv": "inner folds = the group's training dev folds; each inner fold scored with labels from the other "
                                   "training folds; objective pooled component-macro P_spec nDCG@10 (ties: lowest grid index)"}
    descr = {
        "prior": ("NO-EMB", "log1p of the number of distinct training-row components citing each article (query-independent)",
                  {"fit": "training folds of the group"}),
        "knn_lex_votes": ("NO-EMB", "citation votes from the k most BM25-similar training components (memory = one entry per training "
                          "component, representative query text, union of the component's positives; sim-weighted)",
                          {g: selection[g]["knn_lex_votes"]["k"] for g in selection}),
        "K_lex": ("NO-EMB", "robust-z fusion of L_star and kNN lexical votes; (k, w_knn) by inner CV",
                  {g: {"k": selection[g]["K_lex"]["k"], "w_knn": selection[g]["K_lex"]["w_knn"]} for g in selection}),
        "LP": ("NO-EMB", "robust-z L_star + w_prior * robust-z prior; w_prior by inner CV",
               {g: selection[g]["LP"]["w_prior"] for g in selection}),
        "H_bge": ("EMB", "better of RRF(L_star, dense_bge_maxp; k=60) and robust-z linear fusion (w_dense grid), by inner CV",
                  {g: selection[g]["H_bge"]["config"] for g in selection})}
    for m, (family, text, config) in descr.items():
        manifest = {"method": m, "family": family, "label_free": False, "description": text,
                    "training_data": "labels of the group's training dev rows only (lockbox group: all dev rows); lockbox never trains",
                    "configuration": config, **common_manifest}
        if m == "H_bge":
            manifest["label_free"] = True
            manifest["training_data"] = "none (L_star and frozen dense are label-free); fusion choice selected on the group's training rows"
            manifest["depends_on"] = ["L_star", "dense_bge_maxp"]
        common.finish_scores(m, manifest, CODE_FILES)
    log(f"nested stage complete in {wall:.0f}s")


# --------------------------------------------------------------------------- extras: extra NO-EMB citation-memory sources


EXTRA_METHODS = ("knn_lex_votes_k5", "knn_lex_votes_k50", "knn_lex_instr_votes", "knn_char_votes")


def stage_extras(force: bool = False) -> None:
    """Fixed-parameter NO-EMB feature sources for the learned fusion (no hyper-parameter selection, labels from training folds only).

    ``knn_lex_votes_k5`` / ``_k50``: BM25-key memory votes at k = 5 and 50 (the base ``knn_lex_votes`` uses the inner-CV k);
    ``knn_lex_instr_votes``: the k = 50 vote mass aggregated per instrument and given to each of its articles;
    ``knn_char_votes``: memory votes at k = 20 with a character TF-IDF key (``uol.facts_to_law.lexical`` ``char`` settings).
    """
    from scipy import sparse

    if not force and all((common.score_path(m).parent / "manifest.json").exists() for m in EXTRA_METHODS):
        log("extras already complete")
        return
    t0 = time.perf_counter()
    ctx = Ctx()
    d, nested = ctx.data, Nested(ctx)
    instr = [row["instrument_id"] for row in ftl_spec.read_jsonl(Path(d.candidate_binding["articles_path"]))]
    code = {v: i for i, v in enumerate(sorted(set(instr)))}
    cand_instr = np.asarray([code[v] for v in instr])
    incidence_t = sparse.csr_matrix((np.ones(ctx.N, np.float32), (cand_instr, np.arange(ctx.N))), shape=(len(code), ctx.N))
    for m in EXTRA_METHODS:
        if common.score_path(m).parent.exists():
            shutil.rmtree(common.score_path(m).parent)
    mats = {m: common.create_scores(m) for m in EXTRA_METHODS}
    for g in ctx.groups:
        mem, bm25 = nested.memory(g.train_folds)
        mem_texts = [ctx.texts[d.query_index[q]] for q in mem.query_ids]
        char_index = lexical.LexicalIndex(mem_texts, kind="char")
        for sl in ctx.blocks(len(g.rows), 512):
            rows = g.rows[sl]
            texts = [ctx.texts[int(r)] for r in rows]
            comps = [d.queries[int(r)].component_id for r in rows]
            sim = bm25.score(texts)
            kwargs = {"eval_components": comps, "fit_components": list(mem.components)}
            v5 = knn.knn_votes(sim, mem.positives, 5, ctx.N, "sim", **kwargs)
            v50 = knn.knn_votes(sim, mem.positives, 50, ctx.N, "sim", **kwargs)
            mats["knn_lex_votes_k5"][rows] = v5
            mats["knn_lex_votes_k50"][rows] = v50
            mats["knn_lex_instr_votes"][rows] = np.asarray((incidence_t @ v50.T).T, dtype=np.float32)[:, cand_instr]
            mats["knn_char_votes"][rows] = knn.knn_votes(char_index.score(texts), mem.positives, 20, ctx.N, "sim", **kwargs)
        log(f"extras: {g.role} done ({time.perf_counter() - t0:.0f}s)")
    for m in EXTRA_METHODS:
        mats[m].flush()
    del mats
    text = {"knn_lex_votes_k5": "citation votes from the 5 most BM25-similar training components",
            "knn_lex_votes_k50": "citation votes from the 50 most BM25-similar training components",
            "knn_lex_instr_votes": "k = 50 BM25-key vote mass aggregated per instrument, assigned to every article of the instrument",
            "knn_char_votes": "citation votes (k = 20) from training components most similar under character TF-IDF"}
    for m in EXTRA_METHODS:
        common.finish_scores(m, {"method": m, "family": "NO-EMB", "label_free": False, "description": text[m],
                                 "training_data": "labels of the group's training dev rows only (lockbox group: all dev rows); lockbox never trains",
                                 "configuration": {"memory": "one entry per training component: representative query text, union of positives",
                                                   "weighting": "sim", "fixed_parameters": True, "selection": "none (feature source for the learned fusion)"},
                                 "seeds": None, "input_sha256": inputs_hashes(), "wall_seconds": time.perf_counter() - t0, "device": "cpu"},
                             CODE_FILES)
    log(f"extras complete in {time.perf_counter() - t0:.0f}s")


# --------------------------------------------------------------------------- summary


REFERENCE = {"L_star": 0.189, "H_bge": 0.219, "K_lex": 0.327, "LP": 0.239, "dense_bge_maxp": 0.156}


def dev_metrics(method_id: str, ctx: Ctx | None = None, populations: Sequence[str] = ("P_spec", "P", "S2_spec", "all")) -> dict[str, Any]:
    """Dev out-of-fold component-macro metrics of a score file (dev rows only; the lockbox guard stays on)."""
    dev = common.dev_rows()
    scores = np.asarray(common.open_scores(method_id)[dev])
    out = {}
    for pop in populations:
        res = common.evaluate(scores, dev, pop)
        out[pop] = {"n_queries": res["n_queries"], "n_components": res["n_components"],
                    **{m: res["macro"][m] for m in ("ndcg_at_10", "mrr", "recall_at_20")}}
    return out


def stage_summary(methods: Sequence[str] = BASE_METHODS + EXTRA_METHODS) -> dict[str, Any]:
    results = {}
    for m in methods:
        if (common.score_path(m).parent / "manifest.json").exists():
            results[m] = dev_metrics(m)
            log(f"dev metrics {m}: P_spec nDCG@10 {results[m]['P_spec']['ndcg_at_10']:.4f}")
    lines = ["# Base methods: dev out-of-fold metrics (EP-alpha phase 1)", "",
             "Component-macro means over the 4,812 dev rows, each row scored by a model fitted without its fold "
             "(lockbox excluded from all training and selection; no lockbox metric was computed). "
             "Condensed specific lists for P_spec and S2_spec, full reference for P and all.", "",
             "| method | family | P_spec nDCG@10 | MRR | R@20 | P nDCG@10 | MRR | R@20 | S2_spec nDCG@10 | MRR | R@20 | all nDCG@10 | MRR | R@20 |",
             "|---|---|" + "---:|" * 12]
    for m, r in results.items():
        fam = json.loads((common.score_path(m).parent / "manifest.json").read_text(encoding="utf-8")).get("family", "")
        cells = [f"{r[p][k]:.3f}" for p in ("P_spec", "P", "S2_spec", "all") for k in ("ndcg_at_10", "mrr", "recall_at_20")]
        lines.append(f"| {m} | {fam} | " + " | ".join(cells) + " |")
    n = next(iter(results.values()))
    lines += ["", f"Queries / components: " + "; ".join(f"{p} {n[p]['n_queries']}/{n[p]['n_components']}" for p in n), "",
              "## Reference check (facts-to-law v2, all-fold cross-fit incl. lockbox rows, P_spec nDCG@10)", "",
              "| system | brief reference | this run (dev only) | difference |", "|---|---:|---:|---:|"]
    for m, ref in REFERENCE.items():
        if m in results:
            v = results[m]["P_spec"]["ndcg_at_10"]
            lines.append(f"| {m} | {ref:.3f} | {v:.3f} | {v - ref:+.3f} |")
    lines += ["", "Differences arise because the lockbox is excluded from training and from the evaluated rows."]
    (OUT / "DEV_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "dev_summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=("lexical", "dense", "nested", "extras", "summary", "all"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("lexical", "all"):
        stage_lexical(args.force)
    if args.stage in ("dense", "all"):
        stage_dense(args.force)
    if args.stage in ("nested", "all"):
        stage_nested(args.force)
    if args.stage in ("extras", "all"):
        stage_extras(args.force)
    if args.stage in ("summary", "all"):
        stage_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
