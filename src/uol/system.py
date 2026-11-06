"""The two rankers of the thesis, each answering one case at a time from its facts.

    noenc    the ranker without the encoder: five lexical and citation sources, votes of the older tesis by BM25
    ranker   the ranker with Qwen3-Embedding-8B: eight sources, votes of the older tesis by vector similarity

The full system (``uol.llm``) merges the ranker's list with GPT-6.1 Sol's and reorders the first 30.

A ranker is built for a group: an outer fold 0-4 answers the development cases of that fold, and group 5 (``ALL``),
trained on every development fold, answers new cases. Per-group settings were chosen on the group's training folds.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from uol import features as FX, ranker as RK, sources as S
from uol.data import ALL, Benchmark
from uol.dense import DenseSources, Vectors
from uol.history import CitationMemory, Neighbours

SETUPS = ("noenc", "ranker")
SOURCES = {"noenc": S.LEXICAL, "ranker": S.LEXICAL + S.DENSE}

# older-tesis votes per group: similarity (BM25 "lex" or Qwen3 "dense"), neighbours, half-life in years
HISTORY = {
    "noenc": {0: ("lex", 10, 5.0), 1: ("lex", 10, 10.0), 2: ("lex", 25, 10.0), 3: ("lex", 10, 5.0), 4: ("lex", 10, 5.0),
              ALL: ("lex", 10, 5.0)},
    "ranker": {0: ("dense", 10, 5.0), 1: ("dense", 10, 10.0), 2: ("lex", 25, 10.0), 3: ("dense", 10, 5.0),
               4: ("dense", 10, 5.0), ALL: ("dense", 10, 5.0)},
}
# LightGBM learning rate and number of leaves per group
SETTINGS = {
    "noenc": {g: dict(learning_rate=0.05, num_leaves=15) for g in range(ALL + 1)},
    "ranker": {0: dict(learning_rate=0.05, num_leaves=63), **{g: dict(learning_rate=0.05, num_leaves=15) for g in (1, 2, 3, 4)},
               ALL: dict(learning_rate=0.1, num_leaves=15)},
}
HISTORY_VARIANT = "c"             # case vectors used to find similar older tesis


def law_prior(bench: Benchmark, group: int) -> np.ndarray:
    """log(1 + number of training groups citing any article of the article's law), from the corrected labels."""
    pos, law = bench.positives(corrected=True), bench.law_codes
    seen = {(bench.components[r], int(law[a])) for r in bench.train_rows(group) for a in pos.get(int(r), [])}
    count = np.zeros(law.max() + 1)
    for _, i in seen:
        count[i] += 1
    return np.log1p(count)[law].astype(np.float32)


class FirstStage:
    """The sources of one group and the columns they give a case: the pooled candidates and, for each, three columns
    per source and three describing the pair. The older-tesis votes are added by the ranker."""

    def __init__(self, bench: Benchmark, setup: str, group: int, vectors: Vectors | None = None) -> None:
        self.bench, self.sources = bench, SOURCES[setup]
        self.lexical = S.Group(bench, group)
        self.dense = DenseSources(bench, group, vectors) if setup == "ranker" else None
        self.law_prior = law_prior(bench, group)
        self.article_lengths = np.log1p(np.asarray([len(t) for t in bench.article_texts], dtype=np.float32))

    def scores(self, facts: str, case_vectors: Mapping[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        out = self.lexical.score(facts)
        if self.dense is not None:
            out |= self.dense.score(case_vectors)
        return out

    def columns(self, facts: str, case_vectors: Mapping[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        src = self.scores(facts, case_vectors)
        pc = FX.pool(src, self.sources)
        source = np.hstack([FX.source_columns(src[s], pc, self.bench.law_codes) for s in self.sources])
        return pc, np.hstack([source, FX.pair_columns(self.article_lengths, self.law_prior, facts, pc)])


def history_query(key: str, facts: str, case_vectors: Mapping[str, np.ndarray] | None):
    return facts if key == "lex" else case_vectors[HISTORY_VARIANT]


class Ranker:
    """One setup and group: its first stage, the older-tesis votes and a fitted model."""

    def __init__(self, bench: Benchmark, setup: str, group: int, model, vectors: Vectors | None = None,
                 memory: CitationMemory | None = None, neighbours: Neighbours | None = None) -> None:
        self.bench, self.model = bench, model
        self.key, self.k, self.half_life = HISTORY[setup][group]
        self.first = FirstStage(bench, setup, group, vectors)
        self.memory = memory or CitationMemory(bench)
        self.neighbours = neighbours or Neighbours(bench, self.key, vectors)

    def features(self, facts: str, date, case_vectors: Mapping[str, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
        pc, X = self.first.columns(facts, case_vectors)
        near = self.neighbours(history_query(self.key, facts, case_vectors), date)
        return pc, np.hstack([X, self.memory.columns(near, date, pc, self.k, self.half_life)])

    def rank(self, facts: str, date, case_vectors: Mapping[str, np.ndarray] | None = None) -> list[int]:
        pc, X = self.features(facts, date, case_vectors)
        return RK.order(self.model, X, pc, self.bench.generic)
