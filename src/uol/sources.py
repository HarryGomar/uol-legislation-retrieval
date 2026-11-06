"""The first-stage sources. Each scores every catalogue article for one case's facts.

A ``Group`` holds what the sources learn from the training cases of one outer fold (or of every development fold, for
new cases): the case memories, the citation prior, the prototypes and the expanded article documents. Sources learn
from the labels as released; the three corrections apply only to the ranker's targets and to scoring.

    lexical   L_star          BM25 over Spanish stems fused with character n-gram TF-IDF
              docexp_bm25     BM25 over each article plus the facts of training cases that cite it
    citations knn_lex_votes   articles cited by the most similar training cases (BM25 between facts)
              prior           how many training groups cite each article
              proto_tfidf     similarity to each article's prototype: the mean of the facts that cite it
    dense     dense_qwen8b    cosine between the facts and the article (best window), Qwen3-Embedding-8B
              knn_qwen8b      knn_lex_votes with cosine between case vectors
              proto_qwen8b    proto_tfidf with case vectors
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize

from uol import text as T
from uol.data import Benchmark

LEXICAL = ("L_star", "docexp_bm25", "knn_lex_votes", "prior", "proto_tfidf")
DENSE = ("dense_qwen8b", "knn_qwen8b", "proto_qwen8b")

# settings chosen on each group's training folds (inner folds); group 5 is every development fold
L_STAR_W_CHAR = {0: 0.75, 1: 0.5, 2: 0.75, 3: 0.5, 4: 0.5, 5: 0.5}
KNN_LEX_K = {0: 20, 1: 50, 2: 50, 3: 20, 4: 20, 5: 20}
KNN_DENSE = {0: ("c", 10), 1: ("c", 20), 2: ("c", 10), 3: ("c", 10), 4: ("b", 10), 5: ("c", 10)}   # query vectors, k
PROTO_DENSE = {0: ("all", False), 1: ("specific", True), 2: ("specific", True), 3: ("specific", False),
               4: ("specific", False), 5: ("specific", False)}                                    # weights, per group
L_STAR_BM25 = dict(k1=0.9, b=0.75)
DOCEXP = dict(cap=2000, k1=0.9, b=0.4)


@dataclass(frozen=True)
class Memory:
    """One entry per training group of cases: its lowest ``query_id`` represents it, and it cites the union."""
    rows: tuple[int, ...]
    components: tuple[str, ...]
    cited: tuple[tuple[int, ...], ...]


def case_memory(bench: Benchmark, rows: Sequence[int], positives: dict[int, list[int]]) -> Memory:
    rep: dict[str, int] = {}
    union: dict[str, set[int]] = {}
    qid = [c["query_id"] for c in bench.cases]
    for r in rows:
        c = bench.components[r]
        if c not in rep or qid[r] < qid[rep[c]]:
            rep[c] = r
        union.setdefault(c, set()).update(positives.get(r, []))
    comps = sorted(c for c in union if union[c])
    return Memory(tuple(rep[c] for c in comps), tuple(comps), tuple(tuple(sorted(union[c])) for c in comps))


def spread(cited: Sequence[Sequence[int]], n: int) -> sparse.csr_matrix:
    """Row i gives mass one, split evenly over the articles entry i cites."""
    rows, cols, w = [], [], []
    for i, a in enumerate(cited):
        rows += [i] * len(a)
        cols += list(a)
        w += [1.0 / len(a)] * len(a)
    return sparse.csr_matrix((w, (rows, cols)), shape=(len(cited), n), dtype=np.float32)


def votes(sim: np.ndarray, cited: sparse.csr_matrix, k: int) -> np.ndarray:
    """Citation votes of the k most similar memory entries, weighted by their (non-negative) similarity."""
    sim = np.asarray(sim, dtype=np.float32)[None, :]
    k = min(k, sim.shape[1])
    idx = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    top = np.maximum(np.take_along_axis(sim, idx, axis=1).astype(np.float64), 0.0)
    total = top.sum(axis=1, keepdims=True)
    w = np.divide(top, total, out=np.zeros_like(top), where=total > 0)
    W = sparse.csr_matrix((w.astype(np.float32).ravel(), (np.zeros(k, dtype=np.int64), idx.ravel())), shape=sim.shape,
                          dtype=np.float32)
    return (W @ cited).toarray().astype(np.float32, copy=False)[0]


def prototype_links(bench: Benchmark, rows: np.ndarray, positives: dict[int, list[int]], specific_weights: bool,
                    by_group: bool) -> tuple[sparse.csr_matrix, np.ndarray]:
    """[articles, training cases] weights: case j gives each article it cites 1 / (number it cites); with
    ``specific_weights`` a non-generic article gets 1 / (number of non-generic articles cited); with ``by_group`` the
    weight is divided by the number of training cases in j's group. Returns the links and the cited articles."""
    generic = set(bench.generic.tolist())
    comps = np.asarray(bench.components, dtype=object)
    labels, counts = np.unique(comps[rows].astype(str), return_counts=True)
    size = dict(zip(labels.tolist(), counts.tolist()))
    ri, ci, w = [], [], []
    for j, r in enumerate(rows):
        allp = positives.get(int(r), [])
        if not allp:
            continue
        spec = [a for a in allp if a not in generic]
        scale = 1.0 / size[str(comps[r])] if by_group else 1.0
        for a in allp:
            ri.append(a)
            ci.append(j)
            w.append((1.0 / len(spec) if specific_weights and a not in generic else 1.0 / len(allp)) * scale)
    n = len(bench.article_ids)
    return sparse.csr_matrix((w, (ri, ci)), shape=(n, len(rows)), dtype=np.float32), np.unique(ri)


def fill_uncited(sims: np.ndarray, cited: np.ndarray, n: int) -> np.ndarray:
    """Articles no training case cites get the lowest prototype similarity of the case."""
    out = np.full(n, sims.min(), dtype=np.float32)
    out[cited] = sims
    return out


class Group:
    """Everything the lexical and citation sources learn from one group's training cases."""

    def __init__(self, bench: Benchmark, group: int) -> None:
        self.bench, self.group = bench, group
        self.n = len(bench.article_ids)
        self.train = bench.train_rows(group)
        pos = bench.positives(corrected=False)
        docs = bench.article_texts
        # L_star
        self.stems = T.BM25(docs, stems=True, **L_STAR_BM25)
        self.char = T.CharTfidf(docs)
        self.w_char = L_STAR_W_CHAR[group]
        # case memory, shared by the citation votes of both kinds of similarity
        self.memory = case_memory(bench, self.train, pos)
        self.memory_cited = spread(self.memory.cited, self.n)
        self.memory_bm25 = T.BM25([bench.texts[r] for r in self.memory.rows])
        # prior: log(1 + number of training groups citing the article)
        count = np.zeros(self.n, dtype=np.int64)
        for cited in self.memory.cited:
            count[list(cited)] += 1
        self.prior = np.log1p(count).astype(np.float32)
        # prototypes over character TF-IDF of the training facts
        self.proto_vec = T.TfidfVectorizer(**T.CHAR, dtype=np.float32)
        x_train = self.proto_vec.fit_transform([bench.texts[r] for r in self.train]).tocsr()
        link, self.proto_cited = prototype_links(bench, self.train, pos, specific_weights=True, by_group=True)
        self.proto_t = normalize((link[self.proto_cited] @ x_train).tocsr(), norm="l2", axis=1).T.tocsr()
        # expanded documents
        self._docexp(pos)

    def _docexp(self, pos: dict[int, list[int]]) -> None:
        b, train = self.bench, self.train
        self.dx_vec = T.word_vectorizer().fit(list(b.article_texts) + [b.texts[r] for r in train])
        articles = self.dx_vec.transform(b.article_texts).tocsr()
        facts = self.dx_vec.transform([b.texts[r] for r in train]).tocsr()
        qid = [c["query_id"] for c in b.cases]
        cited: dict[int, list[int]] = {}
        for j in sorted(range(len(train)), key=lambda j: qid[train[j]]):
            for a in pos.get(int(train[j]), []):
                cited.setdefault(a, []).append(j)
        ri, ci = [], []
        for a, citing in cited.items():
            by: dict[str, list[int]] = {}
            for j in citing:
                by.setdefault(str(b.components[train[j]]), []).append(j)
            lists = [by[c] for c in sorted(by)]
            used = 0
            for j in (x[d] for d in range(max(map(len, lists))) for x in lists if d < len(x)):     # round robin
                if used >= DOCEXP["cap"]:
                    break
                ri.append(a)
                ci.append(j)
                used += T.n_words(b.texts[train[j]])
        links = sparse.csr_matrix((np.ones(len(ri), dtype=np.float32), (ri, ci)), shape=(self.n, len(train)))
        counts = (articles + np.float32(1.0) * (links @ facts)).tocsr()
        self.dx_t = T.bm25_documents(counts, DOCEXP["k1"], DOCEXP["b"]).T.tocsr()

    def score(self, facts: str) -> dict[str, np.ndarray]:
        """The five lexical and citation sources for one case."""
        lstar = (np.float32(round(1.0 - self.w_char, 10)) * T.robust_z(self.stems.score([facts]))
                 + np.float32(self.w_char) * T.robust_z(self.char.score([facts])))[0]
        docexp = (T.binary_queries(self.dx_vec, [facts]) @ self.dx_t).toarray().astype(np.float32, copy=False)[0]
        knn = votes(self.memory_bm25.score([facts])[0], self.memory_cited, KNN_LEX_K[self.group])
        sims = (self.proto_vec.transform([facts]).tocsr() @ self.proto_t).toarray().astype(np.float32)[0]
        return {"L_star": lstar, "docexp_bm25": docexp, "knn_lex_votes": knn, "prior": self.prior,
                "proto_tfidf": fill_uncited(sims, self.proto_cited, self.n)}
