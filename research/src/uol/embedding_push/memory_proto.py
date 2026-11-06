"""EP-gamma CPU methods: citation-memory key variants, dense article prototypes, lexical document expansion.

Every label-using quantity of a scored row comes from that row's group training rows only
(``common.scoring_groups()``); the lockbox never trains anything. Memory semantics follow
``uol.facts_to_law.knn`` exactly (one entry per fitting component: its lowest-query-id
representative and the union of the component's fitting positives; the top-k entries vote with
normalized weights; ties keep memory order). Selection of k / alpha / weighting uses inner
cross-validation over the group's own training folds (each inner fold is scored by a memory of
the other training folds) with a pooled component-macro nDCG@10 objective, as the facts-to-law
cross-fitting does. Prototype and document-expansion variants are chosen on dev out-of-fold
results (PROTOCOL section 4) and applied unchanged to the lockbox group. Every configuration
tried is appended to ``memory_proto/search_log*.jsonl``.

Command line::

    python -B -m uol.embedding_push.memory_proto knn <method_id>...    # vote files (nested selection); ids in GRIDS
    python -B -m uol.embedding_push.memory_proto proto <bge|e5|tfidf>   # dense (EMB) or TF-IDF (NO-EMB) prototypes
    python -B -m uol.embedding_push.memory_proto docexp <docexp_bm25|docexp_facts_bm25>
    python -B -m uol.embedding_push.memory_proto report <method_id>... [--fusion]   # dev OOF metrics, L_star fusion
    python -B -m uol.embedding_push.memory_proto stack <method_id>... --tag T       # greedy dev z-stack on L_star
    python -B -m uol.embedding_push.memory_proto contrast <tagA> <tagB>             # paired bootstrap of two saved stacks

Learned keys (adapters, probes, fine-tuned encoders) live in ``uol.embedding_push.case_sim`` and reuse this module.
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse

from uol.embedding_push import common
from uol.facts_to_law import fusion
from uol.facts_to_law import knn as ftl_knn
from uol.facts_to_law.lexical import MAX_FEATURES, TOKEN_PATTERN, LexicalIndex

OWNER = "EP-gamma"
WORK = common.RUN / "memory_proto"
STATE = WORK / "state"
KS = (5, 10, 20, 50, 100)
KS_X = (10, 20, 50, 100)
ALPHAS = (0.25, 0.5, 0.75)
CODE_FILE = Path(__file__)
TIE = 1e-12

VECTOR_FILES: dict[str, Path] = {
    "bge": common.DERIVED / "bge_m3_chunks_v1/query_vectors.npy",
    "e5": common.DERIVED / "e5_small_queries_v11/query_vectors.npy",
}


# ---------------------------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------------------------

def _utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"{_utc()} [{os.getpid()}] {message}"
    print(line, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "progress.log").open("a", encoding="utf-8", newline="\n") as f:
        f.write(line + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)}")


def search_log(record: dict[str, Any]) -> None:
    """Append one configuration result. One file per process avoids interleaved writes; see ``merge_logs``."""
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / f"search_log.{os.getpid()}.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps({"utc": _utc(), **record}, ensure_ascii=False, default=_jsonable) + "\n")


def _pid_alive(pid: int) -> bool:
    try:
        import psutil

        return psutil.pid_exists(pid)
    except ImportError:                                  # pragma: no cover - psutil is optional
        import subprocess

        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out


def merge_logs() -> int:
    """Fold the per-process search logs into ``search_log.jsonl`` (sorted by time) and remove the parts.

    Parts of other processes that are still running are left alone, so concurrent jobs never lose records.
    """
    parts = sorted(p for p in WORK.glob("search_log.*.jsonl")
                   if p.name.split(".")[1] == str(os.getpid()) or not _pid_alive(int(p.name.split(".")[1])))
    rows = [json.loads(line) for p in parts for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    main = WORK / "search_log.jsonl"
    if main.exists():
        rows = [json.loads(line) for line in main.read_text(encoding="utf-8").splitlines() if line.strip()] + rows
    rows.sort(key=lambda r: r.get("utc", ""))
    main.write_text("".join(json.dumps(r, ensure_ascii=False, default=_jsonable) + "\n" for r in rows), encoding="utf-8")
    for p in parts:
        p.unlink()
    return len(rows)


# ---------------------------------------------------------------------------------------------
# release-level cached views
# ---------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def texts() -> list[str]:
    return [q.text for q in common.release().queries]


@lru_cache(maxsize=1)
def components() -> np.ndarray:
    return np.asarray([q.component_id for q in common.release().queries])


@lru_cache(maxsize=1)
def query_ids() -> np.ndarray:
    return np.asarray([q.query_id for q in common.release().queries])


_VEC: dict[str, np.ndarray] = {}


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def qvec(name: str) -> np.ndarray:
    """Unit-norm float32 query vectors in release row order (frozen sets, or learned sets once registered)."""
    if name not in _VEC:
        if name not in VECTOR_FILES:
            raise KeyError(f"unknown vector set {name!r}; register it with register_vectors() or register_array()")
        _VEC[name] = _unit(np.load(VECTOR_FILES[name]))
    return _VEC[name]


def register_vectors(name: str, path: Path) -> None:
    VECTOR_FILES[name] = Path(path)
    _VEC.pop(name, None)


def register_array(name: str, array: np.ndarray) -> None:
    """Register in-memory vectors ([n_release_rows, d]) under ``name`` (replacing any earlier set of that name)."""
    if len(array) != len(common.release().queries):
        raise ValueError("vector sets are indexed by release row")
    _VEC[name] = _unit(array)


def n_candidates() -> int:
    return len(common.release().candidate_ids)


def inner_splits(train_rows: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(held-out dev fold, validation rows, fitting rows) for every dev fold present in ``train_rows``."""
    labels = common.roles()[train_rows]
    return [(name, train_rows[labels == name], train_rows[labels != name]) for name in sorted(set(labels.tolist()))]


def group_by_role(role: str) -> tuple[np.ndarray, np.ndarray]:
    for name, eval_rows, train_rows in common.scoring_groups():
        if name == role:
            return eval_rows, train_rows
    raise KeyError(role)


# ---------------------------------------------------------------------------------------------
# citation memory (facts-to-law semantics)
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Mem:
    """One entry per fitting component, in component-id order."""

    rows: np.ndarray                      # representative release row (lowest query id)
    components: tuple[str, ...]
    members: tuple[np.ndarray, ...]       # all fitting rows of each component
    positives: tuple[tuple[int, ...], ...]
    pos_mode: str = "all"

    def matrix(self) -> sparse.csr_matrix:
        """Row i spreads mass one uniformly over entry i's positives (``nonparametric.positive_matrix``)."""
        from uol.retrieval_benchmarks.refinement_suite.nonparametric import positive_matrix
        return positive_matrix(self.positives, n_candidates())


def build_mem(fit_rows: Sequence[int], pos_mode: str = "all") -> Mem:
    """Memory of ``fit_rows``: ``all`` (every reference) or ``specific`` (generic targets dropped).

    ``all`` equals ``uol.facts_to_law.knn.build_memory`` over the same query ids. In ``specific`` mode
    components without a specific positive are dropped, so every entry votes for specific articles only.
    """
    data = common.release()
    qids, comp = query_ids(), components()
    by_comp: dict[str, list[int]] = {}
    for r in sorted((int(x) for x in fit_rows), key=lambda x: qids[x]):
        by_comp.setdefault(str(comp[r]), []).append(r)
    source = data.positive_idx if pos_mode == "all" else data.specific_idx
    names, reps, members, positives = [], [], [], []
    for c in sorted(by_comp):
        rows = by_comp[c]
        union = sorted({p for r in rows for p in source[r]})
        if not union:
            continue
        names.append(c)
        reps.append(rows[0])                       # rows sorted by query id: the lowest id first
        members.append(np.asarray(rows, dtype=np.int64))
        positives.append(tuple(union))
    return Mem(np.asarray(reps, dtype=np.int64), tuple(names), tuple(members), tuple(positives), pos_mode)


def mem_vectors(name: str, mem: Mem, rep: str = "rep") -> np.ndarray:
    """Memory-entry vectors: the representative query's vector (``rep``) or the component mean (``mean``)."""
    v = qvec(name)
    if rep == "rep":
        return v[mem.rows]
    out = np.stack([v[m].mean(axis=0) for m in mem.members]).astype(np.float32)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def sim_lexical(eval_rows: Sequence[int], mem: Mem, params: dict[str, float] | None = None) -> np.ndarray:
    """BM25 of the evaluation texts against the memory representatives (October defaults k1 1.2, b .75)."""
    t = texts()
    index = LexicalIndex([t[int(r)] for r in mem.rows], kind="bm25")
    return index.score([t[int(r)] for r in eval_rows], params)


def sim_char(eval_rows: Sequence[int], mem: Mem) -> np.ndarray:
    """Character TF-IDF cosine (the L_char settings, vocabulary fitted on the memory representatives)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    from uol.facts_to_law.lexical import CHAR_SETTINGS

    t = texts()
    vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
    memory = vectorizer.fit_transform([t[int(r)] for r in mem.rows])
    queries = vectorizer.transform([t[int(r)] for r in eval_rows])
    return (queries @ memory.T).toarray().astype(np.float32, copy=False)


def sim_dense(eval_rows: Sequence[int], mem: Mem, name: str, rep: str = "rep") -> np.ndarray:
    v = qvec(name)
    return (v[np.asarray(eval_rows, dtype=np.int64)] @ mem_vectors(name, mem, rep).T).astype(np.float32, copy=False)


def zrow(x: np.ndarray) -> np.ndarray:
    """Row-wise z-score over memory entries."""
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    return (x - mu) / np.maximum(sd, 1e-6)


def fuse_key(z_lex: np.ndarray, z_dense: np.ndarray, alpha: float) -> np.ndarray:
    return (np.float32(alpha) * z_lex + np.float32(1.0 - alpha) * z_dense).astype(np.float32, copy=False)


@dataclass
class Neighbours:
    """Top-``kmax`` memory entries per evaluation row, in the stable order of ``knn_votes``."""

    index: np.ndarray
    value: np.ndarray


def neighbours(sim: np.ndarray, kmax: int = max(KS)) -> Neighbours:
    if not np.isfinite(sim).all():
        raise ValueError("non-finite memory similarity")
    kmax = min(kmax, sim.shape[1])
    index = np.argsort(-sim, axis=1, kind="stable")[:, :kmax]
    return Neighbours(index, np.take_along_axis(sim, index, axis=1).astype(np.float64))


def votes(nb: Neighbours, mem_matrix: sparse.csr_matrix, k: int, weighting: str = "sim",
          temperature: float = 0.05) -> sparse.csr_matrix:
    """Sparse citation votes ``[n_eval, n_candidates]`` from the k best entries (``knn_votes`` rule)."""
    k = min(int(k), nb.index.shape[1])
    top = nb.value[:, :k]
    weights = ftl_knn._weights(top, weighting, temperature)
    rows = np.repeat(np.arange(top.shape[0]), k)
    neighbour_matrix = sparse.csr_matrix((np.asarray(weights, dtype=np.float32).ravel(), (rows, nb.index[:, :k].ravel())),
                                         shape=(top.shape[0], mem_matrix.shape[0]), dtype=np.float32)
    return (neighbour_matrix @ mem_matrix).tocsr()


def dense_block(votes_sparse: sparse.csr_matrix) -> np.ndarray:
    return votes_sparse.toarray().astype(np.float32, copy=False)


# ---------------------------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------------------------

def pspec_rows(rows: Sequence[int]) -> np.ndarray:
    return common.population_rows("P_spec", rows)


def per_query_ndcg(block: np.ndarray, rows: np.ndarray, population: str = "P_spec") -> tuple[np.ndarray, list[str]]:
    """nDCG@10 of every admitted query of ``rows`` (dev rows only): (values, component ids)."""
    ev = common.evaluate(block, rows, population)
    comp = [str(components()[r]) for r in ev["rows"]]
    return np.asarray([q["ndcg_at_10"] for q in ev["per_query"]], dtype=np.float64), comp


def macro(values: Sequence[float], comps: Sequence[str]) -> float:
    """Component-macro mean (queries within a component first)."""
    ordered = sorted(set(comps))
    index = {c: i for i, c in enumerate(ordered)}
    codes = np.fromiter((index[c] for c in comps), dtype=np.int64, count=len(comps))
    sums = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=len(ordered))
    return float((sums / np.bincount(codes, minlength=len(ordered))).mean())


def pooled(parts: Sequence[tuple[np.ndarray, list[str]]]) -> float:
    return macro(np.concatenate([v for v, _ in parts]), [c for _, cs in parts for c in cs])


def choose(objectives: Sequence[float]) -> int:
    """Highest pooled objective; values within 1e-12 tie and the lowest grid index wins (crossfit rule)."""
    best = max(objectives)
    return min(i for i, v in enumerate(objectives) if v >= best - TIE)


def dev_metrics(scores: np.ndarray, populations: Sequence[str] = ("P_spec", "P", "S2_spec")) -> dict[str, Any]:
    """Component-macro dev metrics of a score matrix [5980, n] (dev rows only; the lockbox is never touched)."""
    out = {}
    for pop in populations:
        rows = common.population_rows(pop, common.dev_rows())
        ev = common.evaluate(np.asarray(scores[rows]), rows, pop)
        out[pop] = {"n_queries": ev["n_queries"], "n_components": ev["n_components"], **ev["macro"]}
    return out


def input_hashes(*names: str) -> dict[str, str]:
    return {n: common.sha256_file(VECTOR_FILES[n]) for n in names}


# ---------------------------------------------------------------------------------------------
# score-file plumbing
# ---------------------------------------------------------------------------------------------

def write_rows(method_id: str, rows: np.ndarray, block: np.ndarray) -> None:
    matrix = common.open_scores(method_id, "r+")
    matrix[np.asarray(rows)] = np.asarray(block, dtype=np.float32)
    matrix.flush()
    del matrix


def ensure_scores(method_id: str) -> None:
    """Create the NaN-filled score file once; later calls resume into the existing file."""
    if not common.score_path(method_id).exists():
        matrix = common.create_scores(method_id)
        matrix.flush()
        del matrix


def state_path(method_id: str) -> Path:
    return STATE / f"{method_id}.json"


def load_state(method_id: str) -> dict[str, Any]:
    path = state_path(method_id)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"groups": {}}


def save_group_state(method_id: str, role: str, info: dict[str, Any]) -> None:
    """Record a finished group. Written per (method, role) file so concurrent workers never clobber each other."""
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / f"{method_id}.{role}.json").write_text(json.dumps(info, indent=1, ensure_ascii=False, default=_jsonable),
                                                    encoding="utf-8")


def group_states(method_id: str) -> dict[str, Any]:
    out = {}
    for role, _, _ in common.scoring_groups():
        p = STATE / f"{method_id}.{role}.json"
        if p.exists():
            out[role] = json.loads(p.read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------------------------
# kNN citation memory: configurations, nested selection, scoring
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Cfg:
    """One memory configuration: key (lex | <dense> | fuse:<dense>:<alpha>), k, weighting, tau, memory positives."""

    key: str
    k: int
    weighting: str = "sim"
    tau: float = 0.05
    pos_mode: str = "all"

    def label(self) -> str:
        w = self.weighting if self.weighting != "softmax" else f"softmax{self.tau:g}"
        return f"{self.pos_mode}|{self.key}|{w}|k{self.k}"


def fused(dense: str, alphas: Sequence[float] = ALPHAS) -> list[str]:
    return [f"fuse:{dense}:{a:g}" for a in alphas]


def _grid(keys: Sequence[str], ks: Sequence[int], weightings: Sequence[tuple[str, float]], pos_mode: str) -> list[Cfg]:
    return [Cfg(key, k, w, tau, pos_mode) for key in keys for (w, tau) in weightings for k in ks]


def extended_grid(pos_mode: str, dense: Sequence[str] = ("bge", "e5")) -> list[Cfg]:
    """Spec grid plus softmax weighting (temperatures on the key's own scale) and specific-only memory."""
    out = _grid(["lex"], KS_X, [("sim", 0.0), ("rank", 0.0)], pos_mode)
    for d in dense:
        out += _grid([d], KS_X, [("sim", 0.0), ("softmax", 0.02), ("softmax", 0.05), ("softmax", 0.1)], pos_mode)
        out += _grid(fused(d), KS_X, [("sim", 0.0), ("softmax", 0.5), ("softmax", 1.0), ("softmax", 2.0)], pos_mode)
    return out


def lex_control_grid(pos_mode: str) -> list[Cfg]:
    """NO-EMB control for the extended grids: the BM25 key alone (raw or z-scored) with every weighting the dense keys get."""
    return (_grid(["lex"], KS_X, [("sim", 0.0), ("rank", 0.0)], pos_mode)
            + _grid(["z:lex"], KS_X, [("softmax", 0.5), ("softmax", 1.0), ("softmax", 2.0)], pos_mode))


def lexc_grid(pos_mode: str) -> list[Cfg]:
    """Strongest NO-EMB memory control: BM25 and character TF-IDF keys (alone or averaged), every weighting."""
    return (lex_control_grid(pos_mode)
            + _grid(["z:char", "mix:lex+char:0.5,0.5"], KS_X, [("softmax", 0.5), ("softmax", 1.0), ("softmax", 2.0)], pos_mode)
            + _grid(["char"], KS_X, [("sim", 0.0), ("rank", 0.0)], pos_mode))


def xc_grid(pos_mode: str) -> list[Cfg]:
    """Frozen dense keys on top of the strongest lexical key: BM25 + char TF-IDF + bge or e5 (lexical-only keys stay in the grid)."""
    out = lexc_grid(pos_mode)
    for dense in ("bge", "e5"):
        out += _grid([f"mix:lex+char+{dense}:{w}" for w in ("0.25,0.25,0.5", "0.2,0.2,0.6", "0.3,0.3,0.4")], (50, 100),
                     [("softmax", 1.0), ("softmax", 2.0)], pos_mode)
    return out


GRIDS: dict[str, dict[str, Any]] = {
    # delivered, exactly the brief's grids ('sim' weighting, all-positive memory, objective P_spec)
    "knn_bge": {"grid": _grid(["bge"], KS, [("sim", 0.0)], "all"), "objective": "P_spec"},
    "knn_e5": {"grid": _grid(["e5"], KS, [("sim", 0.0)], "all"), "objective": "P_spec"},
    "knn_fused": {"grid": _grid(fused("bge") + fused("e5"), KS, [("sim", 0.0)], "all"), "objective": "P_spec"},
    # extended grids: weighting and specific-only memory are searched as well
    "knn_x_spec": {"grid": extended_grid("specific"), "objective": "P_spec"},
    "knn_x_all": {"grid": extended_grid("all"), "objective": "P"},
    # NO-EMB controls: the strongest lexical-key memory (BM25 / character TF-IDF keys) under the same selection protocol
    "knn_lexc_spec": {"grid": lexc_grid("specific"), "objective": "P_spec"},
    "knn_lexc_all": {"grid": lexc_grid("all"), "objective": "P"},
    "knn_xc_spec": {"grid": xc_grid("specific"), "objective": "P_spec"},
    "knn_xc_all": {"grid": xc_grid("all"), "objective": "P"},
    # parity only (never delivered; agent alpha owns knn_lex_votes): the facts-to-law K_lex grid
    "knn_lex": {"grid": _grid(["lex"], (5, 10, 20, 50), [("sim", 0.0)], "all"), "objective": "P_spec", "deliver": False},
}


def key_names(spec: str) -> set[str]:
    """The similarity sets a key specification reads (``lex``, ``char``, dense vector sets)."""
    if spec.startswith("fuse:"):
        return {"lex", spec.split(":")[1]}
    if spec.startswith("mix:"):
        return set(spec.split(":")[1].split("+"))
    if spec.startswith("z:"):
        return {spec[2:]}
    return {spec}


class KeyCache:
    """Lazily computed similarities, z-scores, fused keys and neighbour lists for one (rows, memory) pair."""

    def __init__(self, eval_rows: Sequence[int], mem: Mem, kmax: int = max(KS)):
        self.rows, self.mem, self.kmax = np.asarray(eval_rows, dtype=np.int64), mem, kmax
        self._sim: dict[str, np.ndarray] = {}
        self._z: dict[str, np.ndarray] = {}
        self._nb: dict[str, Neighbours] = {}

    def sim(self, name: str) -> np.ndarray:
        if name not in self._sim:
            if name == "lex":
                self._sim[name] = sim_lexical(self.rows, self.mem)
            elif name == "char":
                self._sim[name] = sim_char(self.rows, self.mem)
            else:
                self._sim[name] = sim_dense(self.rows, self.mem, name)
        return self._sim[name]

    def z(self, name: str) -> np.ndarray:
        if name not in self._z:
            self._z[name] = zrow(self.sim(name))
        return self._z[name]

    def key(self, spec: str) -> np.ndarray:
        if spec.startswith("fuse:"):
            _, dense, alpha = spec.split(":")
            return fuse_key(self.z("lex"), self.z(dense), float(alpha))
        if spec.startswith("z:"):                         # z:lex -> the row z-scored similarity itself
            return self.z(spec[2:])
        if spec.startswith("mix:"):                       # mix:lex+bge+cs:0.4,0.3,0.3 -> weighted sum of row z-scores
            _, names, weights = spec.split(":")
            parts, ws = names.split("+"), [float(x) for x in weights.split(",")]
            return sum(np.float32(w) * self.z(n) for n, w in zip(parts, ws)).astype(np.float32, copy=False)
        return self.sim(spec)

    def nb(self, spec: str) -> Neighbours:
        if spec not in self._nb:
            self._nb[spec] = neighbours(self.key(spec), self.kmax)
        return self._nb[spec]

    def votes(self, cfg: Cfg, mem_matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        return votes(self.nb(cfg.key), mem_matrix, cfg.k, cfg.weighting, cfg.tau)

    def fork(self, volatile: str = "cs") -> "KeyCache":
        """A cache sharing every similarity, z-score and neighbour list that does not involve the vector set ``volatile``."""
        other = KeyCache(self.rows, self.mem, self.kmax)
        other._sim = {k: v for k, v in self._sim.items() if k != volatile}
        other._z = {k: v for k, v in self._z.items() if k != volatile}
        other._nb = {k: v for k, v in self._nb.items() if volatile not in k}
        return other


def knn_inner_table(train_rows: np.ndarray, grid: Sequence[Cfg], objective: str) -> list[dict[str, Any]]:
    """Pooled inner-CV objective of every configuration: each training fold scored by a memory of the others."""
    parts: list[list[tuple[np.ndarray, list[str]]]] = [[] for _ in grid]
    per_fold: list[dict[str, float]] = [{} for _ in grid]
    for name, val_rows, fit_rows in inner_splits(train_rows):
        ev_rows = common.population_rows(objective, val_rows)
        for pos_mode in sorted({c.pos_mode for c in grid}):
            mem = build_mem(fit_rows, pos_mode)
            matrix = mem.matrix()
            cache = KeyCache(ev_rows, mem)
            for i, cfg in enumerate(grid):
                if cfg.pos_mode != pos_mode:
                    continue
                part = per_query_ndcg(dense_block(cache.votes(cfg, matrix)), ev_rows, objective)
                parts[i].append(part)
                per_fold[i][name] = macro(*part)
    return [{"config": asdict(cfg), "label": cfg.label(), "objective": pooled(parts[i]), "per_inner_fold": per_fold[i]}
            for i, cfg in enumerate(grid)]


def knn_outer_block(eval_rows: np.ndarray, train_rows: np.ndarray, cfg: Cfg) -> np.ndarray:
    """Votes of every ``eval_rows`` row from the memory of ``train_rows``: float32 [n_eval, n_candidates]."""
    mem = build_mem(train_rows, cfg.pos_mode)
    if set(mem.rows.tolist()) & set(np.asarray(eval_rows).tolist()):
        raise AssertionError("an evaluation row is in the memory")
    if set(mem.components) & set(map(str, components()[eval_rows])):
        raise AssertionError("an evaluation component is in the memory")
    return dense_block(KeyCache(eval_rows, mem).votes(cfg, mem.matrix()))


def knn_group_job(role: str, method_ids: Sequence[str], resume: bool = True) -> dict[str, Any]:
    """Nested selection and outer scoring of one group for several vote methods (a worker-process job)."""
    eval_rows, train_rows = group_by_role(role)
    summary = {}
    for method_id in method_ids:
        spec = GRIDS[method_id]
        deliver = spec.get("deliver", True)
        if deliver and resume and (STATE / f"{method_id}.{role}.json").exists():
            log(f"{method_id} {role}: already done")
            continue
        started = stamp()
        table = knn_inner_table(train_rows, spec["grid"], spec["objective"])
        best = choose([t["objective"] for t in table])
        cfg = spec["grid"][best]
        for t in table:
            search_log({"stage": "inner_cv", "method": method_id, "role": role, "objective_population": spec["objective"], **t})
        log(f"{method_id} {role}: inner CV selected {cfg.label()} objective {table[best]['objective']:.4f} "
            f"({len(spec['grid'])} configs, {stamp() - started:.0f}s)")
        block = knn_outer_block(eval_rows, train_rows, cfg)
        info = {"role": role, "selected": asdict(cfg), "label": cfg.label(), "inner_objective": table[best]["objective"],
                "objective_population": spec["objective"], "n_configs": len(table), "train_rows": int(len(train_rows)),
                "eval_rows": int(len(eval_rows)), "seconds": stamp() - started}
        if deliver:
            write_rows(method_id, eval_rows, block)
            save_group_state(method_id, role, info)
        else:
            if role != "lockbox":                     # parity diagnostics are dev-only and never stored as a score file
                info["dev_oof"] = dev_block_metrics(block, eval_rows)
            save_group_state(method_id, role, info)
        summary[method_id] = info
    return summary


def dev_block_metrics(block: np.ndarray, rows: np.ndarray) -> dict[str, float]:
    out = {}
    for pop in ("P_spec", "P"):
        sel = common.population_rows(pop, rows)
        pos = {int(r): i for i, r in enumerate(rows)}
        ev = common.evaluate(block[[pos[int(r)] for r in sel]], sel, pop)
        out[pop] = ev["macro"]["ndcg_at_10"]
    return out


def stamp() -> float:
    return time.perf_counter()


def run_knn(method_ids: Sequence[str], workers: int = 3, roles: Sequence[str] | None = None) -> None:
    """Create the score files, run all groups (in parallel worker processes) and finish the manifests."""
    started = stamp()
    delivered = [m for m in method_ids if GRIDS[m].get("deliver", True)]
    for m in delivered:
        ensure_scores(m)
    chosen = list(roles) if roles else [r for r, _, _ in common.scoring_groups()]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {role: pool.submit(knn_group_job, role, list(method_ids)) for role in chosen}
            for role, fut in futures.items():
                fut.result()
    else:
        for role in chosen:
            knn_group_job(role, list(method_ids))
    merge_logs()
    for m in method_ids:
        if GRIDS[m].get("deliver", True):
            finish_knn(m, stamp() - started)
        else:
            summarize_parity(m)


def finish_knn(method_id: str, wall: float) -> dict[str, Any]:
    spec = GRIDS[method_id]
    states = group_states(method_id)
    dense_names = sorted({n for c in spec["grid"] for n in key_names(c.key)} & set(VECTOR_FILES))
    manifest = {
        "method": method_id, "family": "EMB" if dense_names else "NO-EMB", "kind": "kNN citation-memory votes (votes only)",
        "training_data": "per group: memory and selection from that group's training rows only (dev folds != fold; all dev for lockbox)",
        "configuration": {"memory": "one entry per training component: lowest-query-id text/vector, union of component positives",
                          "pos_mode": sorted({c.pos_mode for c in spec["grid"]}),
                          "grid": [c.label() for c in spec["grid"]], "dense_vectors": dense_names,
                          "key_normalization": "row z-score over memory entries before fusing lexical (BM25 k1 1.2 b .75) and dense cosine",
                          "selected_per_group": {r: s["label"] for r, s in states.items()}},
        "selection": {"protocol": "inner CV over the group's training dev folds (each inner fold scored by a memory of the others); "
                                  "pooled component-macro nDCG@10; ties -> lowest grid index",
                      "objective_population": spec["objective"],
                      "inner_objective_per_group": {r: s["inner_objective"] for r, s in states.items()},
                      "search_log": str(WORK / "search_log.jsonl")},
        "seeds": {"randomness": "none (deterministic)"}, "input_sha256": input_hashes(*dense_names),
        "wall_seconds": wall, "device": "cpu",
    }
    manifest["dev_oof"] = dev_metrics(common.open_scores(method_id))
    record = common.finish_scores(method_id, manifest, [CODE_FILE])
    search_log({"stage": "final_dev_oof", "method": method_id, "dev_oof": manifest["dev_oof"]})
    log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")
    return record


def summarize_parity(method_id: str) -> None:
    """Dev-only votes-alone parity diagnostics (pooled over the five dev folds are recomputed by ``report``)."""
    states = group_states(method_id)
    log(f"{method_id} parity: selected per group { {r: s['label'] for r, s in states.items()} }")


# ---------------------------------------------------------------------------------------------
# dense article prototypes
# ---------------------------------------------------------------------------------------------

def proto_link(train_rows: Sequence[int], pos_mode: str, comp_norm: bool) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Sparse [n_candidates, n_train] prototype weights and the sorted indices of the articles cited in training.

    Weight of training query j for article a: 1 / |positives| (``all``: every positive; ``specific``: specific
    positives only, generic articles keep the ``all`` weighting), divided by the size of j's training component
    when ``comp_norm``.
    """
    data, nc = common.release(), n_candidates()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    comp = components()
    labels, counts = np.unique(comp[train_rows], return_counts=True)
    csize = dict(zip(labels.tolist(), counts.tolist()))
    generic = set(data.generic_idx.tolist())
    rows_i, cols_i, weights = [], [], []
    for j, r in enumerate(train_rows):
        allp = data.positive_idx[int(r)]
        if not allp:
            continue
        specp = data.specific_idx[int(r)]
        scale = 1.0 / csize[str(comp[r])] if comp_norm else 1.0
        for a in allp:
            if pos_mode == "specific" and a not in generic:
                w = 1.0 / len(specp)
            else:
                w = 1.0 / len(allp)
            rows_i.append(a)
            cols_i.append(j)
            weights.append(w * scale)
    link = sparse.csr_matrix((weights, (rows_i, cols_i)), shape=(nc, len(train_rows)), dtype=np.float32)
    return link, np.unique(rows_i)


def proto_block(train_rows: Sequence[int], eval_rows: Sequence[int], name: str, pos_mode: str = "specific",
                comp_norm: bool = True) -> np.ndarray:
    """Cosine(query, article prototype) for every article cited in training; others get the row minimum.

    The prototype of article a is the normalized weighted sum of the training queries citing it (weights:
    ``proto_link``).
    """
    nc = n_candidates()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    v = qvec(name)
    link, cited = proto_link(train_rows, pos_mode, comp_norm)
    proto = np.asarray(link[cited] @ v[train_rows])
    proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
    sims = v[np.asarray(eval_rows, dtype=np.int64)] @ proto.T
    out = np.empty((len(eval_rows), nc), dtype=np.float32)
    out[:] = sims.min(axis=1, keepdims=True)
    out[:, cited] = sims
    return out


def proto_sparse_block(train_rows: Sequence[int], eval_rows: Sequence[int], kind: str = "word", pos_mode: str = "specific",
                       comp_norm: bool = True) -> np.ndarray:
    """NO-EMB control of ``proto_block``: cosine of TF-IDF vectors (fitted on the training facts only) to article centroids.

    ``word``: word unigrams (BM25 tokenizer), sublinear tf, min_df 2; ``char``: the L_char settings (char_wb 3-5).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    from uol.facts_to_law.lexical import CHAR_SETTINGS

    nc, t = n_candidates(), texts()
    train_rows = np.asarray(train_rows, dtype=np.int64)
    if kind == "char":
        vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
    else:
        vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                     min_df=2, max_features=MAX_FEATURES, norm="l2", dtype=np.float32)
    x_train = vectorizer.fit_transform([t[int(r)] for r in train_rows]).tocsr()
    x_eval = vectorizer.transform([t[int(r)] for r in eval_rows]).tocsr()
    link, cited = proto_link(train_rows, pos_mode, comp_norm)
    centroids = normalize((link[cited] @ x_train).tocsr(), norm="l2", axis=1)
    sims = (x_eval @ centroids.T).toarray().astype(np.float32)
    out = np.empty((len(eval_rows), nc), dtype=np.float32)
    out[:] = sims.min(axis=1, keepdims=True)
    out[:, cited] = sims
    return out


PROTO_VARIANTS = [("all", False), ("all", True), ("specific", False), ("specific", True)]


def run_proto(name: str) -> None:
    """Choose the prototype variant on dev OOF P_spec, then write all six groups."""
    method_id = f"proto_{name}"
    started = stamp()
    parts: dict[tuple[str, bool], list[tuple[np.ndarray, list[str]]]] = {v: [] for v in PROTO_VARIANTS}
    for role, eval_rows, train_rows in common.scoring_groups()[:5]:
        ps = pspec_rows(eval_rows)
        for variant in PROTO_VARIANTS:
            parts[variant].append(per_query_ndcg(proto_block(train_rows, ps, name, *variant), ps, "P_spec"))
    table = []
    for variant in PROTO_VARIANTS:
        obj = pooled(parts[variant])
        table.append({"pos_mode": variant[0], "comp_norm": variant[1], "dev_oof_P_spec_ndcg10": obj})
        search_log({"stage": "dev_oof_variant", "method": method_id, "pos_mode": variant[0], "comp_norm": variant[1],
                    "dev_oof_P_spec_ndcg10": obj})
    best = choose([t["dev_oof_P_spec_ndcg10"] for t in table])
    pos_mode, comp_norm = PROTO_VARIANTS[best]
    log(f"{method_id}: variant {pos_mode} comp_norm={comp_norm} chosen on dev OOF {table[best]['dev_oof_P_spec_ndcg10']:.4f}")
    ensure_scores(method_id)
    for role, eval_rows, train_rows in common.scoring_groups():
        write_rows(method_id, eval_rows, proto_block(train_rows, eval_rows, name, pos_mode, comp_norm))
        save_group_state(method_id, role, {"role": role, "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows))})
    manifest = {
        "method": method_id, "family": "EMB", "kind": "dense article prototypes",
        "training_data": "per group: queries of the group's training rows citing each article",
        "configuration": {"vectors": name, "score": "cosine(query, normalized weighted sum of citing training queries)",
                          "weights": "1/|positives| (specific mode: 1/|specific positives|, generic articles keep 1/|positives|), "
                                     "divided by training-component size when comp_norm",
                          "pos_mode": pos_mode, "comp_norm": comp_norm, "unseen_articles": "row minimum"},
        "selection": {"protocol": "variant chosen on dev OOF P_spec nDCG@10 (4 variants)", "table": table,
                      "search_log": str(WORK / "search_log.jsonl")},
        "seeds": {"randomness": "none (deterministic)"}, "input_sha256": input_hashes(name),
        "wall_seconds": stamp() - started, "device": "cpu",
    }
    manifest["dev_oof"] = dev_metrics(common.open_scores(method_id))
    common.finish_scores(method_id, manifest, [CODE_FILE])
    log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


SPARSE_VARIANTS = [(kind, pos_mode) for kind in ("word", "char") for pos_mode in ("all", "specific")]


def run_proto_sparse() -> None:
    """NO-EMB control for the dense prototypes: TF-IDF centroids. Variant chosen on dev OOF P_spec, then all six groups."""
    method_id = "proto_tfidf"
    started = stamp()
    parts: dict[tuple[str, str], list[tuple[np.ndarray, list[str]]]] = {v: [] for v in SPARSE_VARIANTS}
    for role, eval_rows, train_rows in common.scoring_groups()[:5]:
        ps = pspec_rows(eval_rows)
        for kind, pos_mode in SPARSE_VARIANTS:
            parts[(kind, pos_mode)].append(per_query_ndcg(proto_sparse_block(train_rows, ps, kind, pos_mode, True), ps, "P_spec"))
        log(f"{method_id}: dev fold {role} evaluated")
    table = []
    for kind, pos_mode in SPARSE_VARIANTS:
        obj = pooled(parts[(kind, pos_mode)])
        table.append({"kind": kind, "pos_mode": pos_mode, "comp_norm": True, "dev_oof_P_spec_ndcg10": obj})
        search_log({"stage": "dev_oof_variant", "method": method_id, "kind": kind, "pos_mode": pos_mode, "comp_norm": True,
                    "dev_oof_P_spec_ndcg10": obj})
    best = choose([t["dev_oof_P_spec_ndcg10"] for t in table])
    kind, pos_mode = SPARSE_VARIANTS[best]
    log(f"{method_id}: variant {kind}/{pos_mode} chosen on dev OOF {table[best]['dev_oof_P_spec_ndcg10']:.4f}")
    ensure_scores(method_id)
    for role, eval_rows, train_rows in common.scoring_groups():
        write_rows(method_id, eval_rows, proto_sparse_block(train_rows, eval_rows, kind, pos_mode, True))
        save_group_state(method_id, role, {"role": role, "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows))})
    manifest = {
        "method": method_id, "family": "NO-EMB", "kind": "TF-IDF article centroids (the NO-EMB analogue of the dense prototypes)",
        "training_data": "per group: TF-IDF vocabulary and centroids from the group's training facts only",
        "configuration": {"tfidf": kind, "pos_mode": pos_mode, "comp_norm": True,
                          "score": "cosine(query tf-idf, normalized weighted sum of the citing training queries' tf-idf)",
                          "unseen_articles": "row minimum"},
        "selection": {"protocol": "variant chosen on dev OOF P_spec nDCG@10 (4 variants)", "table": table,
                      "search_log": str(WORK / "search_log.jsonl")},
        "seeds": {"randomness": "none (deterministic)"}, "input_sha256": text_hashes(), "wall_seconds": stamp() - started,
        "device": "cpu",
    }
    manifest["dev_oof"] = dev_metrics(common.open_scores(method_id))
    common.finish_scores(method_id, manifest, [CODE_FILE])
    log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


def text_hashes() -> dict[str, str]:
    data = common.release()
    return {"queries.jsonl": data.hashes["queries.jsonl"], "candidate_catalogue": str(data.candidate_binding.get("sha256"))}


LSA_PATH = WORK / "lsa_dev_v1.npy"


def ensure_lsa(n_components: int = 512) -> Path:
    """LSA vectors of every release row (NO-EMB): word 1-2-gram TF-IDF and truncated SVD fitted on the dev query texts only."""
    if not LSA_PATH.exists():
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        t = texts()
        dev = common.dev_rows()
        vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN, lowercase=True, strip_accents="unicode", sublinear_tf=True,
                                     ngram_range=(1, 2), min_df=3, max_features=200_000, dtype=np.float32)
        vectorizer.fit([t[int(r)] for r in dev])
        svd = TruncatedSVD(n_components=n_components, algorithm="randomized", n_iter=5, random_state=0)
        svd.fit(vectorizer.transform([t[int(r)] for r in dev]))
        vec = svd.transform(vectorizer.transform(t)).astype(np.float32)
        LSA_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(LSA_PATH, _unit(vec))
        (WORK / "lsa_dev_v1.json").write_text(json.dumps({
            "kind": "LSA (NO-EMB)", "tfidf": {"ngram_range": [1, 2], "min_df": 3, "max_features": 200000, "sublinear_tf": True},
            "svd": {"n_components": n_components, "algorithm": "randomized", "n_iter": 5, "random_state": 0,
                    "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())},
            "fitted_on": "dev query texts only (no labels, no lockbox text); all rows transformed",
            "sha256": common.sha256_file(LSA_PATH)}, indent=1), encoding="utf-8")
    VECTOR_FILES["lsa"] = LSA_PATH
    _VEC.pop("lsa", None)
    return LSA_PATH


CLSA_PATH = WORK / "clsa_dev_v1.npy"


def ensure_clsa(n_components: int = 512) -> Path:
    """Character LSA vectors of every release row (NO-EMB): char_wb 3-5 TF-IDF (the L_char settings) and truncated SVD, dev texts only."""
    if not CLSA_PATH.exists():
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        from uol.facts_to_law.lexical import CHAR_SETTINGS

        t = texts()
        dev = common.dev_rows()
        vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
        vectorizer.fit([t[int(r)] for r in dev])
        svd = TruncatedSVD(n_components=n_components, algorithm="randomized", n_iter=5, random_state=0)
        svd.fit(vectorizer.transform([t[int(r)] for r in dev]))
        vec = svd.transform(vectorizer.transform(t)).astype(np.float32)
        CLSA_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(CLSA_PATH, _unit(vec))
        (WORK / "clsa_dev_v1.json").write_text(json.dumps({
            "kind": "character LSA (NO-EMB)", "tfidf": "uol.facts_to_law.lexical.CHAR_SETTINGS (char_wb 3-5, min_df 2, sublinear tf)",
            "svd": {"n_components": n_components, "algorithm": "randomized", "n_iter": 5, "random_state": 0,
                    "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum())},
            "fitted_on": "dev query texts only (no labels, no lockbox text); all rows transformed",
            "sha256": common.sha256_file(CLSA_PATH)}, indent=1), encoding="utf-8")
    VECTOR_FILES["clsa"] = CLSA_PATH
    _VEC.pop("clsa", None)
    return CLSA_PATH


# ---------------------------------------------------------------------------------------------
# lexical document expansion (NO-EMB control)
# ---------------------------------------------------------------------------------------------

_TOKEN = re.compile(TOKEN_PATTERN)


@lru_cache(maxsize=1)
def token_counts() -> np.ndarray:
    """Number of BM25 tokens (lower-cased ``\\b\\w+\\b``) of every query text."""
    return np.asarray([len(_TOKEN.findall(t.lower())) for t in texts()])


def expansion_links(train_rows: Sequence[int], cap: int | None, pos_mode: str = "all") -> sparse.csr_matrix:
    """Sparse [n_candidates, n_queries] 0/1: training facts appended to each cited article.

    Per article the citing training queries are taken round-robin over components (lowest query id first)
    until their token count reaches ``cap`` (the last fact may cross it; ``None``: every citing query).
    """
    data, nc = common.release(), n_candidates()
    qids, comp, ntok = query_ids(), components(), token_counts()
    source = data.positive_idx if pos_mode == "all" else data.specific_idx
    cited: dict[int, list[int]] = {}
    for r in sorted((int(x) for x in train_rows), key=lambda x: qids[x]):
        for a in source[r]:
            cited.setdefault(a, []).append(r)
    rows_i, cols_i = [], []
    for a, citing in cited.items():
        by: dict[str, list[int]] = {}
        for r in citing:
            by.setdefault(str(comp[r]), []).append(r)
        lists = [by[c] for c in sorted(by)]
        order = [x[depth] for depth in range(max(len(x) for x in lists)) for x in lists if depth < len(x)]
        used = 0
        for r in order:
            if cap is not None and used >= cap:
                break
            rows_i.append(a)
            cols_i.append(r)
            used += int(ntok[r])
    return sparse.csr_matrix((np.ones(len(rows_i), dtype=np.float32), (rows_i, cols_i)), shape=(nc, len(texts())))


class DocExpGroup:
    """Counts of the catalogue articles and of the training facts under one vocabulary (per group)."""

    def __init__(self, train_rows: Sequence[int]):
        from sklearn.feature_extraction.text import CountVectorizer

        data = common.release()
        self.train_rows = np.asarray(train_rows, dtype=np.int64)
        t = texts()
        self.vectorizer = CountVectorizer(token_pattern=TOKEN_PATTERN, min_df=1, max_features=MAX_FEATURES, lowercase=True,
                                          strip_accents="unicode", dtype=np.float32)
        self.vectorizer.fit(list(data.candidate_texts) + [t[int(r)] for r in self.train_rows])
        self.articles = self.vectorizer.transform(data.candidate_texts).tocsr()
        train_counts = self.vectorizer.transform([t[int(r)] for r in self.train_rows]).tocsr()
        # release-row indexed fact counts: only training rows are non-empty, so evaluation facts can never be appended
        selector = sparse.csr_matrix((np.ones(len(self.train_rows), dtype=np.float32),
                                      (self.train_rows, np.arange(len(self.train_rows)))),
                                     shape=(len(t), len(self.train_rows)))
        self.facts = (selector @ train_counts).tocsr()

    def expansion(self, cap: int | None, pos_mode: str) -> sparse.csr_matrix:
        return (expansion_links(self.train_rows, cap, pos_mode) @ self.facts).tocsr()

    def queries(self, eval_rows: Sequence[int]) -> sparse.csr_matrix:
        t = texts()
        q = self.vectorizer.transform([t[int(r)] for r in eval_rows]).astype(np.float32).tocsr()
        q.data[:] = 1.0
        return q


def docexp_block(group: DocExpGroup, eval_rows: Sequence[int], cap: int | None, pos_mode: str, mode: str, k1: float, b: float,
                 weight: float = 1.0, expansion: sparse.csr_matrix | None = None) -> np.ndarray:
    """BM25 (``LexicalIndex`` semantics) of the evaluation texts against expanded article documents.

    ``concat``: article text plus ``weight`` x the appended training facts; ``facts``: only the appended
    training facts, indexed for the cited articles (others score 0).
    """
    from uol.retrieval_benchmarks.c01_statutes.experiments import _bm25_document_matrix

    nc = n_candidates()
    expansion = group.expansion(cap, pos_mode) if expansion is None else expansion
    queries = group.queries(eval_rows)
    if mode == "concat":
        counts = (group.articles + np.float32(weight) * expansion).tocsr()
        docs_t = _bm25_document_matrix(counts, k1=k1, b=b).T.tocsr()
        return (queries @ docs_t).toarray().astype(np.float32, copy=False)
    cited = np.flatnonzero(np.diff(expansion.indptr) > 0)
    docs_t = _bm25_document_matrix(expansion[cited], k1=k1, b=b).T.tocsr()
    out = np.zeros((len(eval_rows), nc), dtype=np.float32)
    out[:, cited] = (queries @ docs_t).toarray()
    return out


DOCEXP_METHODS: dict[str, dict[str, Any]] = {
    "docexp_bm25": {"mode": "concat",
                    "grid": [{"cap": cap, "pos_mode": pm, "k1": k1, "b": b}
                             for cap in (2000,) for pm in ("all", "specific")
                             for k1 in (0.5, 0.7, 0.9, 1.2) for b in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)]},
    "docexp_facts_bm25": {"mode": "facts",
                          "grid": [{"cap": cap, "pos_mode": pm, "k1": k1, "b": b}
                                   for cap in (2000, None) for pm in ("all", "specific")
                                   for k1 in (0.9, 1.2, 1.5) for b in (0.2, 0.4, 0.6, 0.75)]},
}


def run_docexp(method_id: str) -> None:
    """Choose the expansion configuration on dev OOF P_spec (all five dev folds), then write all six groups."""
    spec = DOCEXP_METHODS[method_id]
    grid, mode = spec["grid"], spec["mode"]
    started = stamp()
    parts: list[list[tuple[np.ndarray, list[str]]]] = [[] for _ in grid]
    for role, eval_rows, train_rows in common.scoring_groups()[:5]:
        g = DocExpGroup(train_rows)
        ps = pspec_rows(eval_rows)
        cache: dict[tuple[Any, str], sparse.csr_matrix] = {}
        for i, cfg in enumerate(grid):
            key = (cfg["cap"], cfg["pos_mode"])
            if key not in cache:
                cache[key] = g.expansion(*key)
            block = docexp_block(g, ps, cfg["cap"], cfg["pos_mode"], mode, cfg["k1"], cfg["b"], expansion=cache[key])
            parts[i].append(per_query_ndcg(block, ps, "P_spec"))
        log(f"{method_id}: dev fold {role} evaluated {len(grid)} configs")
    table = []
    for i, cfg in enumerate(grid):
        obj = pooled(parts[i])
        table.append({**cfg, "mode": mode, "dev_oof_P_spec_ndcg10": obj})
        search_log({"stage": "dev_oof_variant", "method": method_id, **cfg, "mode": mode, "dev_oof_P_spec_ndcg10": obj})
    best = choose([t["dev_oof_P_spec_ndcg10"] for t in table])
    cfg = grid[best]
    log(f"{method_id}: chose {cfg} dev OOF {table[best]['dev_oof_P_spec_ndcg10']:.4f}")
    ensure_scores(method_id)
    for role, eval_rows, train_rows in common.scoring_groups():
        g = DocExpGroup(train_rows)
        write_rows(method_id, eval_rows, docexp_block(g, eval_rows, cfg["cap"], cfg["pos_mode"], mode, cfg["k1"], cfg["b"]))
        save_group_state(method_id, role, {"role": role, "train_rows": int(len(train_rows)), "eval_rows": int(len(eval_rows))})
        log(f"{method_id}: wrote {role}")
    manifest = {
        "method": method_id, "family": "NO-EMB",
        "kind": "lexical document expansion with training facts" + (" (facts-only index)" if mode == "facts" else ""),
        "training_data": "per group: masked facts of the group's training rows appended to the articles they cite",
        "configuration": {**cfg, "mode": mode, "bm25": "facts-to-law BM25 (word tokens, accent folding, IDF log1p, binary query terms)",
                          "expansion_order": "round-robin over components, lowest query id first, until the token cap"},
        "selection": {"protocol": "configuration chosen on dev OOF P_spec nDCG@10 over the five dev folds", "table": table,
                      "search_log": str(WORK / "search_log.jsonl")},
        "seeds": {"randomness": "none (deterministic)"}, "input_sha256": {}, "wall_seconds": stamp() - started, "device": "cpu",
    }
    manifest["dev_oof"] = dev_metrics(common.open_scores(method_id))
    common.finish_scores(method_id, manifest, [CODE_FILE])
    log(f"{method_id}: finished; dev OOF P_spec {manifest['dev_oof']['P_spec']['ndcg_at_10']:.4f} P {manifest['dev_oof']['P']['ndcg_at_10']:.4f}")


# ---------------------------------------------------------------------------------------------
# reporting: dev OOF metrics and quick fusion with L_star
# ---------------------------------------------------------------------------------------------

FUSION_WEIGHTS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)


def fusion_check(method_ids: Sequence[str], base: str = "L_star", weights: Sequence[float] = FUSION_WEIGHTS) -> dict[str, Any]:
    """Dev OOF P_spec / P of robust_z(base) + w * robust_z(method) per weight (one global weight; no label use)."""
    if not (common.SCORES / base / "manifest.json").exists():
        raise FileNotFoundError(f"{base} has no manifest yet")
    rows = common.population_rows("P", common.dev_rows())
    base_z = fusion.robust_z(np.asarray(common.open_scores(base)[rows]))
    out: dict[str, Any] = {"base": base, "base_alone": {p: common.evaluate(base_z, rows, p)["macro"]["ndcg_at_10"] for p in ("P_spec", "P")}}
    for m in method_ids:
        z = fusion.robust_z(np.asarray(common.open_scores(m)[rows]))
        curve = {}
        for w in weights:
            f = base_z + np.float32(w) * z
            curve[w] = {p: common.evaluate(f, rows, p)["macro"]["ndcg_at_10"] for p in ("P_spec", "P")}
        best_spec = max(curve, key=lambda w: curve[w]["P_spec"])
        best_p = max(curve, key=lambda w: curve[w]["P"])
        out[m] = {"curve": curve, "best_w_P_spec": best_spec, "best_P_spec": curve[best_spec],
                  "best_w_P": best_p, "best_P": curve[best_p]}
        search_log({"stage": "fusion_check", "method": m, **out[m]})
        log(f"fusion {m}: best w(P_spec)={best_spec} -> P_spec {curve[best_spec]['P_spec']:.4f} P {curve[best_spec]['P']:.4f}; "
            f"best w(P)={best_p} -> P {curve[best_p]['P']:.4f}")
    return out


def stack_check(components_: Sequence[str], base: str = "L_star", grid: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0),
                passes: int = 2, tag: str = "", objective_population: str = "P_spec") -> dict[str, Any]:
    """Greedy robust-z stack ``z(base) + sum_i w_i z(component_i)``; weights by coordinate ascent on dev OOF P_spec (or P).

    One global weight vector is tuned on all dev rows (no cross-fitting), so absolute numbers are optimistic; the
    comparison of two stacks under the same procedure is the intended use (e.g. NO-EMB stack versus the same stack
    with embedding signals added).
    """
    rows = common.population_rows("P", common.dev_rows())
    spec = common.population_rows(objective_population, rows)
    spec_pos = np.flatnonzero(np.isin(rows, spec))
    z = {m: fusion.robust_z(np.asarray(common.open_scores(m)[rows])) for m in [base, *components_]}
    zs = {m: z[m][spec_pos] for m in z}
    weights = {m: 0.0 for m in components_}

    def fused(zz: dict[str, np.ndarray]) -> np.ndarray:
        out = zz[base].copy()
        for m, w in weights.items():
            if w:
                out += np.float32(w) * zz[m]
        return out

    def objective() -> float:
        return common.evaluate(fused(zs), rows[spec_pos], objective_population)["macro"]["ndcg_at_10"]

    best = objective()
    for _ in range(passes):
        changed = False
        for m in components_:
            keep = weights[m]
            for w in grid:
                if w == keep:
                    continue
                weights[m] = w
                value = objective()
                if value > best + 1e-9:
                    best, keep, changed = value, w, True
            weights[m] = keep
        if not changed:
            break
    final = fused(z)
    ev_spec, ev_p = common.evaluate(final, rows, "P_spec"), common.evaluate(final, rows, "P")
    result = {"tag": tag, "base": base, "components": list(components_), "weights": dict(weights),
              "objective_population": objective_population,
              "P_spec": ev_spec["macro"]["ndcg_at_10"], "P": ev_p["macro"]["ndcg_at_10"]}
    search_log({"stage": "stack_check", **result})
    stacks = WORK / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", tag or "_".join(components_))[:80]
    (stacks / f"{slug}.json").write_text(json.dumps({
        **result, "components_P_spec": ev_spec["components"], "per_component_P_spec_ndcg10": ev_spec["per_component"]["ndcg_at_10"],
        "components_P": ev_p["components"], "per_component_P_ndcg10": ev_p["per_component"]["ndcg_at_10"]}, default=_jsonable),
        encoding="utf-8")
    log(f"stack {tag or components_}: P_spec {result['P_spec']:.4f} P {result['P']:.4f} weights {weights}")
    return result


def stack_cv(components_: Sequence[str], base: str = "L_star", grid: Sequence[float] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0),
             passes: int = 2, tag: str = "", objective_population: str = "P_spec") -> dict[str, Any]:
    """Cross-fitted version of ``stack_check``: the weights scoring dev fold f are tuned on the other four dev folds only.

    The pooled dev metrics are therefore free of weight-selection optimism (the components themselves are already OOF),
    and the per-component values are saved under ``stacks/`` for ``stack_contrast`` (tag suffixed ``[cv]``).
    """
    from uol.facts_to_law import metrics

    rows = common.population_rows("P", common.dev_rows())
    fold_of = common.roles()[rows]
    spec_mask = np.isin(rows, common.population_rows(objective_population, rows))
    z = {m: fusion.robust_z(np.asarray(common.open_scores(m)[rows])) for m in [base, *components_]}
    pooled_q: dict[str, list[dict[str, Any]]] = {"P_spec": [], "P": []}
    pooled_c: dict[str, list[str]] = {"P_spec": [], "P": []}
    weights_by_fold: dict[str, dict[str, float]] = {}
    for f in range(5):
        name = f"dev_fold{f}"
        train = np.flatnonzero((fold_of != name) & spec_mask)
        test = np.flatnonzero(fold_of == name)
        zt = {m: z[m][train] for m in z}
        weights = {m: 0.0 for m in components_}

        def fused(zz: dict[str, np.ndarray]) -> np.ndarray:
            out = zz[base].copy()
            for m, w in weights.items():
                if w:
                    out += np.float32(w) * zz[m]
            return out

        def objective() -> float:
            return common.evaluate(fused(zt), rows[train], objective_population)["macro"]["ndcg_at_10"]

        best = objective()
        for _ in range(passes):
            changed = False
            for m in components_:
                keep = weights[m]
                for w in grid:
                    if w == keep:
                        continue
                    weights[m] = w
                    value = objective()
                    if value > best + 1e-9:
                        best, keep, changed = value, w, True
                weights[m] = keep
            if not changed:
                break
        del zt
        weights_by_fold[name] = dict(weights)
        fused_test = fused({m: z[m][test] for m in z})
        for pop in ("P_spec", "P"):
            ev = common.evaluate(fused_test, rows[test], pop)
            pooled_q[pop] += ev["per_query"]
            pooled_c[pop] += [str(components()[r]) for r in ev["rows"]]
        log(f"stack_cv {tag or components_}: fold {f} weights {weights}")
    out: dict[str, Any] = {"tag": tag + (" [cv]" if objective_population == "P_spec" else f" [cv {objective_population}]"), "base": base,
                           "components": list(components_), "weights_by_fold": weights_by_fold, "objective_population": objective_population}
    saved: dict[str, Any] = {}
    for pop in ("P_spec", "P"):
        macro_, per_component, ordered = metrics.component_macro(pooled_q[pop], pooled_c[pop])
        out[pop] = macro_["ndcg_at_10"]
        saved[f"components_{pop}"] = ordered
        saved[f"per_component_{pop}_ndcg10"] = per_component["ndcg_at_10"]
    search_log({"stage": "stack_cv", **out})
    stacks = WORK / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", out["tag"])[:80]
    (stacks / f"{slug}.json").write_text(json.dumps({**out, **saved}, default=_jsonable), encoding="utf-8")
    log(f"stack_cv {tag or components_}: cross-fitted P_spec {out['P_spec']:.4f} P {out['P']:.4f}")
    return out


def stack_contrast(tag_a: str, tag_b: str, population: str = "P_spec") -> dict[str, Any]:
    """Paired component bootstrap (10,000 draws, seed 20261004) of stack a - stack b on dev rows (saved by ``stack_check``)."""
    from uol.facts_to_law import stats

    def load(tag: str) -> dict[str, Any]:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", tag)[:80]
        return json.loads((WORK / "stacks" / f"{slug}.json").read_text(encoding="utf-8"))

    a, b = load(tag_a), load(tag_b)
    key = "P" if population == "P" else "P_spec"
    if a[f"components_{key}"] != b[f"components_{key}"]:
        raise ValueError("stacks were evaluated on different components")
    out = {**stats.contrast(np.asarray(a[f"per_component_{key}_ndcg10"]), np.asarray(b[f"per_component_{key}_ndcg10"])),
           "a": tag_a, "b": tag_b, "population": population, "components": len(a[f"components_{key}"])}
    search_log({"stage": "stack_contrast", **out})
    log(f"stack contrast {population}: {out}")
    return out


def report(method_ids: Sequence[str]) -> dict[str, Any]:
    out = {}
    for m in method_ids:
        out[m] = dev_metrics(common.open_scores(m))
        log(f"{m}: dev OOF " + " ".join(f"{p}={v['ndcg_at_10']:.4f}" for p, v in out[m].items()))
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -B -m uol.embedding_push.memory_proto")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("knn")
    p.add_argument("methods", nargs="+", choices=sorted(GRIDS))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--roles", nargs="*", default=None)
    p = sub.add_parser("proto")
    p.add_argument("name")
    p = sub.add_parser("docexp")
    p.add_argument("method", choices=sorted(DOCEXP_METHODS))
    p = sub.add_parser("report")
    p.add_argument("methods", nargs="+")
    p.add_argument("--fusion", action="store_true")
    p = sub.add_parser("stack")
    p.add_argument("components", nargs="+")
    p.add_argument("--tag", default="")
    p.add_argument("--objective", default="P_spec", choices=("P_spec", "P"))
    p = sub.add_parser("stackcv")
    p.add_argument("components", nargs="+")
    p.add_argument("--tag", default="")
    p.add_argument("--objective", default="P_spec", choices=("P_spec", "P"))
    p = sub.add_parser("contrast")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--population", default="P_spec", choices=("P_spec", "P"))
    p = sub.add_parser("merge-logs")
    args = parser.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    if args.command == "knn":
        run_knn(args.methods, args.workers, args.roles)
    elif args.command == "proto":
        run_proto_sparse() if args.name == "tfidf" else run_proto(args.name)
        merge_logs()
    elif args.command == "docexp":
        run_docexp(args.method)
        merge_logs()
    elif args.command == "report":
        report(args.methods)
        if args.fusion:
            print(json.dumps(fusion_check(args.methods), default=_jsonable, indent=1))
    elif args.command == "stack":
        print(json.dumps(stack_check(args.components, tag=args.tag, objective_population=args.objective), default=_jsonable, indent=1))
        merge_logs()
    elif args.command == "stackcv":
        print(json.dumps(stack_cv(args.components, tag=args.tag, objective_population=args.objective), default=_jsonable, indent=1))
        merge_logs()
    elif args.command == "contrast":
        print(json.dumps(stack_contrast(args.a, args.b, args.population), default=_jsonable, indent=1))
    elif args.command == "merge-logs":
        print(merge_logs())
    return 0


if __name__ == "__main__":
    sys.exit(main())
