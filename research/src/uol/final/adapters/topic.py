"""Topic and latent models (new systems, RUN_PLAN §3.3): LSA, LDA topic similarity, the LDA document model of
Wei & Croft (2006) and doc2vec PV-DBOW.

All four are unsupervised. Per outer fold each is fitted on the catalogue article texts (body text, the frozen
``V0``) **plus the masked facts of that fold's training cases** (text only; no label), with the project's Spanish
analyser (``facts_to_law.lexical.SpanishAnalyzer``: stopwords, Snowball stems, accent folding); the scored facts of
the held-out fold never enter the fit. The model size is chosen on the inner grouped folds of the outer fold
(``adapters.base.select``) with the outer-fold model.

* LSA: sublinear TF-IDF (min_df 2) -> truncated SVD (randomized, seed 20261004) -> cosine; 256 / 512 / 1,024 dims.
* LDA similarity: gensim ``LdaMulticore`` (symmetric alpha 50/K, eta 0.01, 3 passes, seed 20261004); query topic
  mixture by variational inference; score = Bhattacharyya coefficient sum_k sqrt(theta_q,k theta_d,k)
  (= 1 - squared Hellinger distance); K = 100 / 200 / 400.
* LDA document model (Wei & Croft 2006): P(w|d) = lam (tf(w,d) + mu P(w|C)) / (|d| + mu)
  + (1 - lam) sum_z P(w|z) P(z|d); score = sum_{w in q} c(w,q) log P(w|d); lam .7, mu 1000 as in the paper; it
  reuses the LDA model of the same K and fold.
* doc2vec PV-DBOW: gensim ``Doc2Vec(dm=0)`` (window 5, min_count 2, negative 5, 20 epochs, seed 20261004);
  query vectors by ``infer_vector`` (50 epochs); cosine; 100 / 300 dims.
* word2vec (D88, added after the freeze): gensim skip-gram (min_count 2, negative 5, 10 epochs, seed 20261004); a text
  is the IDF-weighted mean of its word vectors; cosine; vector size {100, 300} x window {5, 10}.

Multi-threaded LDA, doc2vec and word2vec training are not bit-reproducible run to run; the thread count is in the receipt.
Fitted models are cached in ``<work>/topic/`` by (family, size, catalogue hash, training-facts hash).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context
from uol.final.adapters.base import Scorer, select
from uol.final.adapters.lexical import article_texts, query_texts

SEED = 20261004
THREADS = int(os.environ.get("UOL_FINAL_THREADS", "8"))
GRIDS = {"lsa": [256, 512, 1024], "lda_sim": [100, 200, 400], "lda_dm": [100, 200, 400], "doc2vec": [100, 300]}
_ANALYZER: dict[str, Any] = {}
_MODELS: dict[str, Any] = {}


def analyzer():
    if "a" not in _ANALYZER:
        from uol.facts_to_law.lexical import SpanishAnalyzer
        _ANALYZER["a"] = SpanishAnalyzer()
    return _ANALYZER["a"]


def tokens(texts: Sequence[str]) -> list[list[str]]:
    a = analyzer()
    return [a(t) for t in texts]


def text_key(texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _cache_dir() -> Path:
    p = context.active().work / "topic"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.where(n > 0, n, 1.0)).astype(np.float32)


# --------------------------------------------------------------------------- models (fit on articles + training facts)


class LSAModel:
    def __init__(self, articles: list[str], extra: list[str], dim: int):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(analyzer=analyzer(), sublinear_tf=True, min_df=2, dtype=np.float32)
        X = self.vec.fit_transform(list(articles) + list(extra))
        dim = int(min(dim, X.shape[1] - 1, X.shape[0] - 1))
        self.svd = TruncatedSVD(n_components=dim, algorithm="randomized", n_iter=7, random_state=SEED)
        Z = self.svd.fit_transform(X)
        self.A = _l2(Z[:len(articles)])
        self.info = {"vocabulary": int(X.shape[1]), "dim": dim, "fit_docs": int(X.shape[0]),
                     "explained_variance": float(self.svd.explained_variance_ratio_.sum())}

    def score(self, q_texts: list[str]) -> np.ndarray:
        Q = _l2(self.svd.transform(self.vec.transform(q_texts)))
        return (Q @ self.A.T).astype(np.float32)


def deprior(gamma: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Topic mixture from the posterior Dirichlet parameter with the prior mass removed: n = max(gamma - alpha, 0) +
    1e-6, theta = n / sum(n) (REVIEW_results_20261022 §3(b), Option B; D78)."""
    n = np.maximum(np.asarray(gamma, dtype=np.float64) - np.asarray(alpha, dtype=np.float64)[None, :], 0.0) + 1e-6
    return (n / n.sum(1, keepdims=True)).astype(np.float32)


class LDAModel:
    def __init__(self, articles: list[str], extra: list[str], k: int, passes: int = 3):
        from gensim import corpora, models
        toks = tokens(list(articles) + list(extra))
        self.dct = corpora.Dictionary(toks)
        self.dct.filter_extremes(no_below=2, no_above=0.5, keep_n=100_000)
        self.dct.compactify()
        corpus = [self.dct.doc2bow(t) for t in toks]
        kw = dict(num_topics=k, id2word=self.dct, alpha=[50.0 / k] * k, eta=0.01, passes=passes, iterations=100,
                  random_state=SEED, chunksize=4000, eval_every=None)
        if THREADS > 1:
            self.lda = models.LdaMulticore(corpus, workers=max(1, THREADS - 1), **kw)
        else:
            self.lda = models.LdaModel(corpus, **kw)
        art = corpus[:len(articles)]
        gamma, _ = self.lda.inference(art)
        self.theta = (gamma / gamma.sum(1, keepdims=True)).astype(np.float32)
        V = len(self.dct)
        self.counts = _bow_matrix(art, V)                         # csc [articles, V]
        self.doc_len = np.asarray(self.counts.sum(1)).ravel().astype(np.float64)
        coll = np.asarray(self.counts.sum(0)).ravel().astype(np.float64)
        self.p_coll = (coll + 1e-9) / max(coll.sum(), 1.0)
        self.phi = self.lda.get_topics().astype(np.float64)        # [K, V], rows sum to 1
        self.info = {"topics": int(k), "vocabulary": int(V), "passes": passes, "alpha": 50.0 / k, "eta": 0.01,
                     "fit_docs": len(corpus), "workers": max(1, THREADS - 1) if THREADS > 1 else 1}

    def query_theta(self, q_texts: list[str]) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
        bows = [self.dct.doc2bow(t) for t in tokens(q_texts)]
        gamma, _ = self.lda.inference(bows)
        return (gamma / gamma.sum(1, keepdims=True)).astype(np.float32), bows

    def similarity(self, q_texts: list[str]) -> np.ndarray:
        """lda_sim: Bhattacharyya coefficient of the topic mixtures, read out as expected topic counts (the posterior
        Dirichlet parameter minus the prior alpha; D78, changed after the freeze, CHANGES_AFTER_FREEZE.md). Normalising
        gamma directly kept the prior mass of 50 pseudo-counts in every mixture, which flattened them and favoured short
        articles (.010, long articles never retrieved). ``self.theta`` (the normalised gamma) is unchanged and still
        feeds lda_dm."""
        bows = [self.dct.doc2bow(t) for t in tokens(q_texts)]
        gamma, _ = self.lda.inference(bows)
        th = deprior(gamma, self.lda.alpha)
        return (np.sqrt(th) @ np.sqrt(self.article_theta_sim()).T).astype(np.float32)

    def article_theta_sim(self) -> np.ndarray:
        """Article mixtures for lda_sim (expected topic counts), from the fitted model and the stored article counts,
        so cached models stay valid; inference starts from a fixed seed so the read-out is reproducible."""
        if getattr(self, "_theta_sim", None) is None:
            rows = self.counts.tocsr()
            bows = [list(zip(rows[i].indices.tolist(), rows[i].data.tolist())) for i in range(rows.shape[0])]
            state = self.lda.random_state
            self.lda.random_state = np.random.RandomState(SEED)
            try:
                gamma, _ = self.lda.inference(bows)
            finally:
                self.lda.random_state = state
            self._theta_sim = deprior(gamma, self.lda.alpha)
        return self._theta_sim

    def doc_model(self, q_texts: list[str], lam: float = 0.7, mu: float = 1000.0) -> np.ndarray:
        """Wei & Croft LDA-based document model log-likelihood of each query under every article."""
        _, bows = self.query_theta(q_texts)
        out = np.empty((len(bows), self.theta.shape[0]), dtype=np.float32)
        denom = self.doc_len + mu
        th = self.theta.astype(np.float64)
        for i, bow in enumerate(bows):
            if not bow:
                out[i] = 0.0
                continue
            w = np.asarray([b[0] for b in bow], dtype=np.int64)
            c = np.asarray([b[1] for b in bow], dtype=np.float64)
            tf = self.counts[:, w].toarray().astype(np.float64)                  # [articles, |q|]
            p_dir = (tf + mu * self.p_coll[w][None, :]) / denom[:, None]
            p = lam * p_dir + (1.0 - lam) * (th @ self.phi[:, w])
            out[i] = (np.log(np.maximum(p, 1e-300)) @ c).astype(np.float32)
        return out


def _bow_matrix(corpus, V: int):
    from scipy import sparse
    rows, cols, vals = [], [], []
    for i, bow in enumerate(corpus):
        for j, c in bow:
            rows.append(i)
            cols.append(j)
            vals.append(c)
    return sparse.csc_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(corpus), V))


class Doc2VecModel:
    def __init__(self, articles: list[str], extra: list[str], size: int, epochs: int = 20):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument
        toks = tokens(list(articles) + list(extra))
        docs = [TaggedDocument(t, [i]) for i, t in enumerate(toks)]
        self.model = Doc2Vec(docs, dm=0, vector_size=size, window=5, min_count=2, sample=1e-4, negative=5,
                             epochs=epochs, workers=THREADS, seed=SEED)
        self.A = _l2(np.asarray([self.model.dv[i] for i in range(len(articles))], dtype=np.float32))
        self.info = {"vector_size": size, "epochs": epochs, "workers": THREADS, "dm": 0, "fit_docs": len(docs)}

    def score(self, q_texts: list[str]) -> np.ndarray:
        self.model.random.seed(SEED)
        Q = _l2(np.asarray([self.model.infer_vector(t, epochs=50) for t in tokens(q_texts)], dtype=np.float32))
        return (Q @ self.A.T).astype(np.float32)


class Word2VecModel:
    """D88 (23 Oct 2026, added after the freeze): gensim skip-gram word vectors fitted on the same corpus and tokens as
    the other models of this module; a text is the IDF-weighted mean of its in-vocabulary token vectors (IDF over the
    fit documents, ``log(N / df)``); score = cosine. Articles are scored whole (no MaxP: they are short)."""

    def __init__(self, articles: list[str], extra: list[str], value: str, epochs: int = 10):
        from gensim.models import Word2Vec
        size, window = (int(x) for x in str(value).split("w"))
        toks = tokens(list(articles) + list(extra))
        model = Word2Vec(toks, sg=1, vector_size=size, window=window, min_count=2, sample=1e-4, negative=5,
                         epochs=epochs, workers=THREADS, seed=SEED)
        self.index = dict(model.wv.key_to_index)
        self.vectors = np.asarray(model.wv.vectors, dtype=np.float32)
        df = np.zeros(len(self.index), dtype=np.float64)
        for t in toks:
            for j in {self.index[w] for w in t if w in self.index}:
                df[j] += 1
        self.idf = np.log(len(toks) / np.maximum(df, 1.0)).astype(np.float32)
        self.A = self._embed(toks[:len(articles)])
        self.info = {"vector_size": size, "window": window, "epochs": epochs, "sg": 1, "min_count": 2,
                     "workers": THREADS, "fit_docs": len(toks), "vocabulary": len(self.index),
                     "text_vector": "IDF-weighted mean of word vectors (IDF log(N/df) over the fit documents)"}

    def _embed(self, toks: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(toks), self.vectors.shape[1]), dtype=np.float32)
        for i, t in enumerate(toks):
            idx = np.asarray([self.index[w] for w in t if w in self.index], dtype=np.int64)
            if len(idx):
                w = self.idf[idx]
                if w.sum() > 0:
                    out[i] = (w @ self.vectors[idx]) / w.sum()
        return _l2(out)

    def score(self, q_texts: list[str]) -> np.ndarray:
        return (self._embed(tokens(q_texts)) @ self.A.T).astype(np.float32)


KINDS = {"lsa": (LSAModel, "dim"), "lda_sim": (LDAModel, "topics"), "lda_dm": (LDAModel, "topics"),
         "doc2vec": (Doc2VecModel, "size"), "word2vec": (Word2VecModel, "size")}
GRIDS["word2vec"] = [{"size": s, "window": w} for s in (100, 300) for w in (5, 10)]


def param_value(kind: str, params: dict) -> int | str:
    """The model's cache value for one grid point: an int, or ``<size>w<window>`` for word2vec."""
    if kind == "word2vec":
        return f"{int(params['size'])}w{int(params['window'])}"
    return int(params[KINDS[kind][1]])


def model_key(kind: str, value: int | str, articles: list[str], extra: list[str]) -> tuple[str, str]:
    """(family, cache key) of a fitted model; ``<work>/topic/<key>.pkl`` is its file (also read by the cost record)."""
    fam = "lda" if kind.startswith("lda") else kind             # both LDA systems share one fitted model per (K, fold)
    return fam, f"{fam}_{value}_{text_key(articles)}_{text_key(extra)}"


def model_for(kind: str, value: int | str, articles: list[str], extra: list[str]) -> Any:
    cls, _ = KINDS[kind]
    fam, key = model_key(kind, value, articles, extra)
    if key in _MODELS:
        return _MODELS[key]
    path = _cache_dir() / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f:
            _MODELS[key] = pickle.load(f)
        return _MODELS[key]
    t0 = time.perf_counter()
    m = cls(articles, extra, value)
    m.info["fit_seconds"] = round(time.perf_counter() - t0, 1)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("wb") as f:
        pickle.dump(m, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    for k in [k for k in _MODELS if k.startswith(fam + "_")]:     # keep one fitted model per family in memory
        del _MODELS[k]
    _MODELS[key] = m
    return m


class TopicScorer(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        a = self.spec["adapter_args"]
        self.kind = a["kind"]
        self.articles = article_texts(a.get("text", "V0"))
        self.use_facts = bool(a.get("training_facts", True))
        name = KINDS[self.kind][1]
        grid = a.get("grid") or GRIDS[self.kind]
        self.grid = [dict(v) if isinstance(v, dict) else
                     {name: v, **({"lam": a.get("lam", 0.7), "mu": a.get("mu", 1000.0)} if self.kind == "lda_dm" else {})}
                     for v in grid]
        self.pname = name
        self.extra: list[str] = []

    def _scores(self, params: dict, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        m = model_for(self.kind, param_value(self.kind, params), self.articles, self.extra)
        q = query_texts(rows)
        if self.kind == "lda_sim":
            return m.similarity(q)
        if self.kind == "lda_dm":
            return m.doc_model(q, params["lam"], params["mu"])
        return m.score(q)

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        self.extra = query_texts(self.rows(fold.train_rows, "topic training facts")) if self.use_facts else []
        self.params, hist = select(self.grid, self._scores, fold, self.pop.eval_population)
        m = model_for(self.kind, param_value(self.kind, self.params), self.articles, self.extra)
        return {"params": self.params, "selection": hist, "model": m.info, "training_facts": len(self.extra)}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        return self._scores(self.params, rows)

    def inputs(self) -> dict[str, Any]:
        return {"catalogue_text_sha16": text_key(self.articles), "threads": THREADS,
                "analyzer": json.loads(json.dumps(analyzer().receipt, default=str))}
