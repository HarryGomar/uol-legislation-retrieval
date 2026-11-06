"""Training a ranker from the benchmark.

Step 1 builds a table of out-of-fold columns for every development case: the case's sources are those of the group
of its own fold, fitted without it, so a training case looks to the ranker as a new case does. The table also keeps
each case's 100 nearest older *tesis* for each kind of similarity.

Step 2 fits one group: its training cases (every fold but its own), the older-tesis votes with the group's setting,
the cited articles as targets (corrected labels, generic articles removed) and LightGBM LambdaMART. It also scores
the group's own fold, which gives the out-of-fold development lists without building a ranker per case. ``select``
repeats the choice of the LightGBM setting on the group's inner folds.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from uol import evaluate, ranker as RK
from uol.data import ALL, N_FOLDS, Benchmark
from uol.dense import Vectors
from uol.history import CitationMemory, Neighbours
from uol.system import HISTORY, SETTINGS, FirstStage, history_query


@dataclass
class Table:
    rows: np.ndarray                      # development case rows, ascending
    ptr: np.ndarray                       # the pairs of rows[i] are ptr[i]:ptr[i + 1]
    candidates: np.ndarray                # article of each pair
    X: np.ndarray                         # source and pair columns of each pair
    near: dict[str, tuple[np.ndarray, np.ndarray]]    # per similarity: [rows, KMAX] memory positions and similarities

    def save(self, path: str | Path) -> None:
        near = {f"{k}_{part}": v[i] for k, v in self.near.items() for i, part in enumerate(("idx", "sim"))}
        np.savez(path, rows=self.rows, ptr=self.ptr, candidates=self.candidates, X=self.X, **near)

    @classmethod
    def load(cls, path: str | Path) -> "Table":
        z = np.load(path)
        keys = {name.rsplit("_", 1)[0] for name in z.files if name.endswith("_idx")}
        return cls(z["rows"], z["ptr"], z["candidates"], z["X"], {k: (z[f"{k}_idx"], z[f"{k}_sim"]) for k in keys})


def build_table(bench: Benchmark, setup: str, vectors: Vectors | None = None, log=print) -> Table:
    rows = bench.train_rows(ALL)
    case_vectors = (lambda r: {v: vectors.cases[v][r] for v in vectors.cases}) if vectors else (lambda r: None)
    cands, cols = {}, {}
    for fold in range(N_FOLDS):
        t = time.perf_counter()
        first = FirstStage(bench, setup, fold, vectors)
        for r in rows[bench.fold[rows] == fold]:
            cands[r], cols[r] = first.columns(bench.texts[r], case_vectors(r))
        log(f"fold {fold}: columns of {int((bench.fold[rows] == fold).sum())} cases in {time.perf_counter() - t:.0f} s")
    near = {}
    for key in sorted({h[0] for h in HISTORY[setup].values()}):
        t = time.perf_counter()
        find = Neighbours(bench, key, vectors)
        found = [find(history_query(key, bench.texts[r], case_vectors(r)), bench.dates[r]) for r in rows]
        near[key] = (np.stack([f[0] for f in found]).astype(np.int32), np.stack([f[1] for f in found]))
        log(f"older tesis by {key}: {len(rows)} cases in {time.perf_counter() - t:.0f} s")
    sizes = np.asarray([len(cands[r]) for r in rows])
    return Table(rows, np.r_[0, np.cumsum(sizes)], np.concatenate([cands[r] for r in rows]),
                 np.vstack([cols[r] for r in rows]), near)


def group_matrix(bench: Benchmark, table: Table, setup: str, group: int, rows: np.ndarray,
                 memory: CitationMemory) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs of ``rows`` without generic articles: their position in the table, their full columns with the group's
    older-tesis votes, and the case of each pair."""
    key, k, half_life = HISTORY[setup][group]
    at = np.searchsorted(table.rows, rows)
    generic = np.zeros(len(bench.article_ids), dtype=bool)
    generic[bench.generic] = True
    pairs, votes, case = [], [], []
    for j, i in enumerate(at):
        p = np.arange(table.ptr[i], table.ptr[i + 1])
        p = p[~generic[table.candidates[p]]]
        near = (table.near[key][0][i], table.near[key][1][i])
        votes.append(memory.columns(near, bench.dates[table.rows[i]], table.candidates[p], k, half_life))
        pairs.append(p)
        case.append(np.full(len(p), j))
    pairs = np.concatenate(pairs)
    return pairs, np.hstack([table.X[pairs], np.vstack(votes)]), np.concatenate(case)


def fit_rows(bench: Benchmark, table: Table, setup: str, group: int, train: np.ndarray, setting: dict,
             rng: np.random.Generator, memory: CitationMemory, threads: int = 4):
    """LightGBM on the pooled pairs of ``train``: every cited article, at most 300 others per case."""
    pairs, X, case = group_matrix(bench, table, setup, group, train, memory)
    specific, n = bench.specific(corrected=True), len(bench.article_ids)
    cited = np.fromiter((j * n + a for j, r in enumerate(train) for a in specific.get(int(r), [])), dtype=np.int64)
    y = np.isin(case * n + table.candidates[pairs], cited).astype(np.int32)
    keep, yy, counts = RK.training_lists(case, y, rng)
    return RK.fit(X[keep], yy, counts, setting, RK.seed_of(group), threads), len(counts), len(keep)


def lists(bench: Benchmark, table: Table, setup: str, group: int, model, rows: np.ndarray, memory: CitationMemory,
          k: int = 200) -> dict[int, list[int]]:
    """The ranked list of each of ``rows`` (generic articles removed)."""
    pairs, X, case = group_matrix(bench, table, setup, group, rows, memory)
    score = model.predict(X)
    out = {}
    for j, r in enumerate(rows):
        m = case == j
        out[int(r)] = [int(a) for a in table.candidates[pairs[m]][np.argsort(-score[m], kind="stable")][:k]]
    return out


def fit(bench: Benchmark, table: Table, setup: str, group: int, memory: CitationMemory | None = None,
        setting: dict | None = None, threads: int = 4, log=print):
    """The group's model, on every development fold but its own, with its chosen setting unless one is given."""
    memory = memory or CitationMemory(bench)
    train = table.rows[bench.fold[table.rows] != group]
    rng = np.random.default_rng(RK.seed_of(group) + 999)
    t = time.perf_counter()
    model, cases, pairs = fit_rows(bench, table, setup, group, train, setting or SETTINGS[setup][group], rng, memory,
                                   threads)
    log(f"{setup} group {group}: {cases} training cases, {pairs} pairs, fitted in {time.perf_counter() - t:.0f} s")
    return model


def fold_lists(bench: Benchmark, table: Table, setup: str, group: int, model, memory: CitationMemory | None = None,
               k: int = 200) -> dict[int, list[int]]:
    """The ranked list of every development case of the group's fold."""
    rows = table.rows[bench.fold[table.rows] == group]
    return lists(bench, table, setup, group, model, rows, memory or CitationMemory(bench), k)


def select(bench: Benchmark, table: Table, setup: str, group: int, memory: CitationMemory | None = None,
           threads: int = 4, log=print) -> tuple[dict, list[float]]:
    """The setting of ``ranker.GRID`` a group chooses: the first two of its training folds are held out in turn, and
    the setting with the best group-macro nDCG@10 on their development evaluation cases wins (ties to the first)."""
    memory = memory or CitationMemory(bench)
    folds = [f for f in range(N_FOLDS) if f != group][:2]
    evaluation = bench.evaluation_rows("DEV")
    objective = []
    for setting in RK.GRID:
        scores = {}
        for inner in folds:
            train = table.rows[(bench.fold[table.rows] != group) & (bench.fold[table.rows] != inner)]
            rng = np.random.default_rng(RK.seed_of(group) + 17 * inner)
            model, _, _ = fit_rows(bench, table, setup, group, train, setting, rng, memory, threads)
            held = evaluation[bench.fold[evaluation] == inner]
            scores |= evaluate.per_case(bench, lists(bench, table, setup, group, model, held, memory))
        objective.append(float(np.mean(list(evaluate.by_group(bench, scores).values()))))
        log(f"{setup} group {group}: {setting} -> {objective[-1]:.4f}")
    return RK.GRID[int(np.argmax(objective))], objective
