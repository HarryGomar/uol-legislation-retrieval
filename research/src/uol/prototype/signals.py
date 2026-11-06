"""Deployment signals: one float32 score row over the 39,178-article catalogue per signal and query.

Every operator mirrors the research source it replaces (same code paths or a line-for-line copy):

* ``L_star``         ``embedding_push.base.lstar_block`` (BM25 with Spanish stemming fused with character TF-IDF);
* ``prior``          ``facts_to_law.prior.fitting_prior`` (log1p of distinct training components citing the article);
* ``knn_lex_votes``  ``Nested.votes``: BM25 key over the training components' representative facts, ``sim`` votes;
* ``dense_qwen8b``   ``qwen_endpoint.dense_block``: MaxP cosine to the cached Qwen3-Embedding-8B article windows;
* ``knn_qwen8b``     ``FitSet.dense_sim`` + ``FitSet.votes``: Qwen cosine key over the same memory;
* ``proto_qwen8b``   ``FitSet.prototypes`` / ``proto_scores``: cosine to label-weighted article prototypes;
* ``census_qwen8b``  ``round3.m1_memory`` (dense key, k 10, 5-year half-life) over the 84k pre-2021 census tesis.

A ``Fit`` holds what the labels of a set of dev training folds give (memory, prior, prototypes, instrument prior). The
dev table, vectors and fits come from the bundle, which holds dev rows only.
"""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from uol.facts_to_law import fusion, knn, lexical
from uol.prototype import config as K

def unit_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Stopwatch:
    """Accumulates seconds per named stage."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}

    def add(self, stage: str, t0: float) -> float:
        now = time.perf_counter()
        self.seconds[stage] = self.seconds.get(stage, 0.0) + (now - t0)
        return now


# ----------------------------------------------------------------------------- dev table


@dataclass
class Dev:
    """Dev rows only (bundle ``dev.jsonl``): facts text, component, fold and label lists as candidate indices."""

    rows: list[dict[str, Any]]
    root: Path

    @staticmethod
    def load(root: Path = K.BUNDLE) -> "Dev":
        return Dev(read_jsonl(root / "dev.jsonl"), root)

    def __len__(self) -> int:
        return len(self.rows)

    @cached_property
    def fold(self) -> np.ndarray:
        return np.asarray([r["fold"] for r in self.rows], dtype=np.int64)

    @cached_property
    def texts(self) -> list[str]:
        return [r["text"] for r in self.rows]

    @cached_property
    def components(self) -> list[str]:
        return [r["component"] for r in self.rows]

    @cached_property
    def by_query(self) -> dict[str, int]:
        return {r["query_id"]: i for i, r in enumerate(self.rows)}

    def positions(self, folds: Sequence[int]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.fold, list(folds)))

    def vectors(self, variant: str) -> np.ndarray:
        """Unit Qwen query vectors of the dev rows (bundle order) for a query variant."""
        if variant not in self._vec:
            self._vec[variant] = np.load(self.root / f"dev_vectors_{variant}.npy", mmap_mode="r")
        return self._vec[variant]

    @cached_property
    def _vec(self) -> dict[str, np.ndarray]:
        return {}


# ----------------------------------------------------------------------------- catalogue


class Catalogue:
    """Article metadata, lexical indexes and the Qwen article windows (label-free)."""

    def __init__(self, root: Path = K.BUNDLE, *, load_vectors: bool = True) -> None:
        self.root = root
        self.meta = read_jsonl(root / "catalogue.jsonl")
        self.N = len(self.meta)
        self.instrument = np.asarray([m["instrument"] for m in self.meta], dtype=np.int64)
        self.generic = np.asarray([bool(m["generic"]) for m in self.meta])
        self.log_len = np.log1p(np.asarray([m["text_len"] for m in self.meta], dtype=np.float32))
        perm = np.argsort(self.instrument, kind="stable")
        ordered = self.instrument[perm]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        segment = np.zeros(int(self.instrument.max()) + 1, dtype=np.int64)
        segment[ordered[starts]] = np.arange(len(starts))
        self.layout = (perm, starts, segment)
        self._lex: dict[str, lexical.LexicalIndex] = {}
        self._units: tuple[np.ndarray, np.ndarray] | None = None
        if load_vectors:
            self.units()

    def lex(self, kind: str) -> lexical.LexicalIndex:
        if kind not in self._lex:
            with (self.root / f"lex_{kind}.pkl").open("rb") as fh:
                self._lex[kind] = pickle.load(fh)
        return self._lex[kind]

    def units(self) -> tuple[np.ndarray, np.ndarray]:
        """(unit vectors [U, 4096] float32 in RAM, reduceat starts per article); units are sorted by article."""
        if self._units is None:
            owner = np.load(K.QWEN / "article_index.npy")
            if not (np.diff(owner) >= 0).all() or len(np.unique(owner)) != self.N:
                raise ValueError("article windows must be sorted by article and cover the catalogue")
            starts = np.flatnonzero(np.r_[True, owner[1:] != owner[:-1]])
            self._units = (np.load(K.QWEN / "article_vectors.npy"), starts)
        return self._units

    # label-free signals ---------------------------------------------------------------------------------

    def lstar(self, texts: Sequence[str], sel: dict[str, Any]) -> np.ndarray:
        """``base.lstar_block`` with this catalogue's indexes."""
        kind = sel["kind"]
        if kind in ("bm25", "bm25_stem"):
            return self.lex(kind).score(texts, {"k1": sel["k1"], "b": sel["b"]})
        if kind == "char":
            return self.lex("char").score(texts)
        w = float(sel["w_char"])
        bm = fusion.robust_z(self.lex(sel["base"]).score(texts, {"k1": sel["k1"], "b": sel["b"]}))
        ch = fusion.robust_z(self.lex("char").score(texts))
        return fusion.linear_fusion([bm, ch], [round(1.0 - w, 10), w], normalize="none")

    def dense(self, q: np.ndarray) -> np.ndarray:
        """MaxP cosine of unit query vectors over each article's windows (``common.maxp_scores``)."""
        vectors, starts = self.units()
        return np.maximum.reduceat(np.asarray(q, dtype=np.float32) @ vectors.T, starts, axis=1).astype(np.float32, copy=False)


# ----------------------------------------------------------------------------- fits (labels of training folds)


@dataclass
class Memory:
    """``knn.build_memory`` over dev positions: one entry per component, lowest query id as representative."""

    rep: np.ndarray                         # dev position of each entry's representative row
    components: tuple[str, ...]
    positives: tuple[tuple[int, ...], ...]


def build_memory(dev: Dev, positions: Sequence[int]) -> Memory:
    rep: dict[str, int] = {}
    union: dict[str, set[int]] = {}
    for p in positions:
        row = dev.rows[int(p)]
        c = row["component"]
        if c not in rep or row["query_id"] < dev.rows[rep[c]]["query_id"]:
            rep[c] = int(p)
        union.setdefault(c, set()).update(row["positives"])
    comps = sorted(c for c in union if union[c])
    return Memory(np.asarray([rep[c] for c in comps], dtype=np.int64), tuple(comps), tuple(tuple(sorted(union[c])) for c in comps))


@dataclass
class Fit:
    """Everything the labels of ``folds`` give: memory, its keys, prior, prototypes and the instrument prior."""

    group: str
    cfg: K.GroupConfig
    folds: tuple[int, ...]
    memory: Memory
    mem_lex: lexical.LexicalIndex
    mem_keys: dict[str, np.ndarray]                    # variant -> unit vectors of the memory representatives
    prior: np.ndarray                                  # [N]
    instr_prior: np.ndarray                            # [N]
    protos: tuple[np.ndarray, np.ndarray]              # (unit prototypes [n_seen, D], candidate index of each)
    mem_records: list[str] = field(default_factory=list)


def build_fit(dev: Dev, cat: Catalogue, group: str) -> Fit:
    cfg = K.group_config(group)
    folds = K.training_folds(group)
    pos = dev.positions(folds)
    mem = build_memory(dev, pos)
    mem_lex = lexical.LexicalIndex([dev.texts[int(p)] for p in mem.rep], kind="bm25")
    keys = {cfg.knn_variant: np.ascontiguousarray(dev.vectors(cfg.knn_variant)[mem.rep], dtype=np.float32)}
    # prior: distinct training components citing each article (all positives, as the research references)
    by_comp: dict[str, set[int]] = {}
    by_comp_instr: set[tuple[str, int]] = set()
    for p in pos:
        row = dev.rows[int(p)]
        by_comp.setdefault(row["component"], set()).update(row["positives"])
        by_comp_instr.update((row["component"], int(cat.instrument[a])) for a in row["positives"])
    counts = np.zeros(cat.N, dtype=np.int64)
    for arts in by_comp.values():
        counts[np.fromiter(arts, dtype=np.int64, count=len(arts))] += 1
    prior = np.log1p(counts).astype(np.float32)
    icount = np.zeros(int(cat.instrument.max()) + 1)
    for _, i in by_comp_instr:
        icount[i] += 1
    instr_prior = np.log1p(icount).astype(np.float32)[cat.instrument]
    protos = prototypes(dev, pos, cfg.proto_variant, cfg.proto_mode, cfg.proto_cn)
    records = [dev.rows[int(p)]["record_id"] for p in mem.rep]
    return Fit(group, cfg, folds, mem, mem_lex, keys, prior, instr_prior, protos, records)


def prototypes(dev: Dev, positions: np.ndarray, variant: str, mode: str, cn: int) -> tuple[np.ndarray, np.ndarray]:
    """``qwen_endpoint.FitSet.prototypes``: each fit row adds its vector to every cited article with weight 1/|positives|
    (``specific``: 1/|specific| for its specific positives); ``cn`` divides by the row's component size in the fit."""
    comps = [dev.rows[int(p)]["component"] for p in positions]
    size: dict[str, int] = {}
    for c in comps:
        size[c] = size.get(c, 0) + 1
    ri, ai, wt = [], [], []
    for j, p in enumerate(positions):
        row = dev.rows[int(p)]
        pos, spec = row["positives"], row["specific"]
        if not pos:
            continue
        sset = set(spec)
        for a in pos:
            w = 1.0 / len(spec) if mode == "specific" and spec and a in sset else 1.0 / len(pos)
            if cn:
                w /= size[comps[j]]
            ri.append(j)
            ai.append(a)
            wt.append(w)
    seen = np.unique(np.asarray(ai, dtype=np.int64))
    column = {int(a): i for i, a in enumerate(seen)}
    W = sparse.csr_matrix((np.asarray(wt, dtype=np.float32), (np.asarray(ri), np.asarray([column[a] for a in ai]))),
                          shape=(len(positions), len(seen)))
    S = np.asarray(W.T @ np.asarray(dev.vectors(variant)[np.asarray(positions)], dtype=np.float32), dtype=np.float32)
    return unit_rows(S), seen


def positive_matrix(positives: Sequence[Sequence[int]], n: int) -> sparse.csr_matrix:
    """Row i spreads mass one uniformly over entry i's positives (``nonparametric.positive_matrix``)."""
    lens = np.fromiter((len(p) for p in positives), dtype=np.int64, count=len(positives))
    rows = np.repeat(np.arange(len(positives)), lens)
    cols = np.fromiter((a for p in positives for a in p), dtype=np.int64, count=int(lens.sum()))
    return sparse.csr_matrix((np.repeat(1.0 / np.maximum(lens, 1), lens).astype(np.float32), (rows, cols)),
                             shape=(len(positives), n))


# ----------------------------------------------------------------------------- census memory


class Census:
    """84k pre-2021 census tesis (round-2 B memory) with cached Qwen vectors (rubro + texto, float16 storage)."""

    def __init__(self, root: Path = K.BUNDLE) -> None:
        self.meta = read_jsonl(root / "census.jsonl")
        # float16 on disk, float32 in RAM (1.4 GB): the conversion is exact and per-query conversion cost ~1.5 s
        self.vectors = np.load(root / "census_vectors_f16.npy").astype(np.float32)
        if len(self.vectors) != len(self.meta):
            raise ValueError("census vectors and metadata differ in length")
        dates = np.asarray([m["date"] for m in self.meta], dtype="datetime64[D]")
        # M1 weights are max(sim, 0) * 0.5 ** (age / half-life), normalised per query: the query date cancels, so the
        # entry's age is measured from a fixed reference that every census entry predates.
        self.ref_date = np.datetime64("2020-08-14")
        if dates.max() >= self.ref_date:
            raise ValueError("a census entry does not predate the first dev query")
        self.age_years = (self.ref_date - dates).astype(np.float64) / 365.25
        self.cited = [tuple(m["cited"]) for m in self.meta]

    def sims(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float32) @ self.vectors.T).astype(np.float32, copy=False)

    def neighbours(self, q: np.ndarray, k: int = K.CENSUS_CFG["k"]) -> tuple[np.ndarray, np.ndarray]:
        """Top-k entries per query (similarity descending, memory order on ties: ``m1_memory._topk``)."""
        sims = self.sims(q)
        part = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(sims, part, axis=1)
        order = np.lexsort((part, -ps), axis=1)
        idx = np.take_along_axis(part, order, axis=1)
        return idx, np.take_along_axis(sims, idx, axis=1)

    def votes(self, idx: np.ndarray, sim: np.ndarray, n: int, half_life: float = K.CENSUS_CFG["half_life"]) -> np.ndarray:
        w = np.maximum(sim.astype(np.float64), 0.0) * np.power(0.5, self.age_years[idx] / half_life)
        tot = w.sum(axis=1, keepdims=True)
        w = np.divide(w, tot, out=np.zeros_like(w), where=tot > 0)
        out = np.zeros((len(idx), n), dtype=np.float32)
        for i in range(len(idx)):
            for j, e in enumerate(idx[i]):
                c = self.cited[int(e)]
                out[i, list(c)] += np.float32(w[i, j] / len(c))
        return out


# ----------------------------------------------------------------------------- one block of queries


@dataclass
class Evidence:
    """Neighbours behind the memory votes (for provenance): memory entry index and similarity per query."""

    dev_idx: np.ndarray | None = None       # [n, k] entries of fit.memory
    dev_sim: np.ndarray | None = None
    census_idx: np.ndarray | None = None    # [n, k] entries of the census
    census_sim: np.ndarray | None = None


def knn_rows(sim: np.ndarray, positives: Sequence[Sequence[int]], k: int, n: int,
             eval_components: Sequence[str] | None, fit_components: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``knn.knn_votes`` (sim weighting) plus the neighbour indices and similarities it used."""
    votes = knn.knn_votes(sim, positives, k, n, "sim", eval_components=eval_components,
                          fit_components=list(fit_components) if eval_components is not None else None)
    kk = min(int(k), sim.shape[1])
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :kk]
    return votes, idx, np.take_along_axis(sim, idx, axis=1)


def compute(cat: Catalogue, fit: Fit, census: Census | None, texts: Sequence[str], vecs: dict[str, np.ndarray] | None,
            components: Sequence[str] | None = None, signals: Sequence[str] = K.SIGNALS,
            watch: Stopwatch | None = None) -> tuple[dict[str, np.ndarray], Evidence]:
    """Score rows ``{signal: [n, N]}`` for a block of queries. ``vecs`` maps query variant -> unit vectors ``[n, 4096]``
    (None: no embedding, only the non-embedding signals may be asked for)."""
    watch = watch or Stopwatch()
    cfg, n, out, ev = fit.cfg, cat.N, {}, Evidence()
    want = set(signals)
    if vecs is None and want & K.EMB_SIGNALS:
        raise ValueError("embedding signals need query vectors")
    t = time.perf_counter()
    if "L_star" in want:
        out["L_star"] = cat.lstar(texts, cfg.lstar).astype(np.float32, copy=False)
        t = watch.add("lexical", t)
    if "dense_qwen8b" in want:
        out["dense_qwen8b"] = cat.dense(vecs[K.DENSE_VARIANT])
        t = watch.add("dense", t)
    if "knn_lex_votes" in want:
        sim = fit.mem_lex.score(texts)
        out["knn_lex_votes"], _, _ = knn_rows(sim, fit.memory.positives, cfg.knn_lex_k, n, components, fit.memory.components)
    if "knn_qwen8b" in want:
        sim = (np.asarray(vecs[cfg.knn_variant], dtype=np.float32) @ fit.mem_keys[cfg.knn_variant].T).astype(np.float32)
        out["knn_qwen8b"], ev.dev_idx, ev.dev_sim = knn_rows(sim, fit.memory.positives, cfg.knn_k, n, components,
                                                             fit.memory.components)
    t = watch.add("memory", t)
    if "census_qwen8b" in want:
        ev.census_idx, ev.census_sim = census.neighbours(vecs[K.CENSUS_CFG["variant"]])
        out["census_qwen8b"] = census.votes(ev.census_idx, ev.census_sim, n)
        t = watch.add("census", t)
    if "proto_qwen8b" in want:
        protos, seen = fit.protos
        s = np.asarray(vecs[cfg.proto_variant], dtype=np.float32) @ protos.T
        row = np.empty((len(s), n), dtype=np.float32)
        row[:] = s.min(axis=1, keepdims=True)
        row[:, seen] = s
        out["proto_qwen8b"] = row
    if "prior" in want:
        out["prior"] = np.broadcast_to(fit.prior, (len(texts), n)).astype(np.float32)
    watch.add("prototypes_prior", t)
    return {s: out[s] for s in signals}, ev
