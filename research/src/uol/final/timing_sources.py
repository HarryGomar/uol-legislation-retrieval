"""First-stage sources of the frozen rankers recomputed for single cases, through the original producers (D137).

Every one of the 50 full-catalogue sources of the frozen pair table and the ColBERT pair source is computed for one
release row of an outer development group from its query-side inputs, with that group's frozen configuration: the
selections, memory-proto states and Qwen selections of the frozen embedding-push run, the memories, priors,
prototypes and adapters of the group's training rows. The code is ``uol.round2.c_score`` (which did the same for the
lockbox group on new rows and reproduced the frozen files to 1.3e-6), generalised from the lockbox to any group, with
one change: everything that depends only on the group (indexes over the catalogue or the memory, memories, priors,
prototypes, adapters, the probe, the document-expansion counts) is built once in ``FirstStage.load`` and reused,
because a deployed system builds it once; ``KNN_FITS`` lists the memory-proto fits that are hoisted this way, and
``tests/final/test_timing_20261026.py`` checks that the hoisted versions equal the originals.

Query-side inputs are read by row from in-memory arrays (``FirstStage.vectors``): the stored vectors of the frozen
run, or, for a case, the vectors its encoders just produced (``inject``). Learned per-group maps that the frozen run
kept only as outputs (the linear adapters, the probe classes, the case-similarity encoder and the fine-tuned e5 query
tower of the group) are read from their stored outputs; their encoders are timed by the encoder job.

Nothing here reads a label of a scored row: memories, priors and prototypes use the group's training rows only.
"""
from __future__ import annotations

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.final import context

FULL_ORDER = [
    "L_star", "dense_bge_maxp", "dense_bge_firstp", "dense_e5_maxp", "prior", "knn_lex_votes", "K_lex", "LP", "H_bge",
    "knn_lex_votes_k5", "knn_lex_votes_k50", "knn_lex_instr_votes", "knn_char_votes",
    "knn_bge", "knn_e5", "knn_fused", "knn_x_spec", "knn_x_all", "knn_lexc_spec", "knn_lexc_all", "knn_xc_spec", "knn_xc_all",
    "proto_bge", "proto_e5", "proto_tfidf", "docexp_bm25", "docexp_facts_bm25",
    "knn_adapt_spec", "proto_adapt", "knn_adapt_all", "knn_lsaadapt_spec", "proto_lsaadapt", "knn_lsaadapt_all",
    "knn_lsacadapt_spec", "proto_lsacadapt", "knn_lsacadapt_all", "proto_probe",
    "dense_qwen8b", "H_qwen8b", "knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b", "K_qwen8b", "KH_qwen8b", "K_lex_ext",
    "dense_bge_long", "ft_e5_q", "knn_casesim_e5", "knn_casesim_e5_all", "proto_casesim_e5"]
PAIR_SOURCES = ["colbert_bge"]
KNN_METHODS = ("knn_bge", "knn_e5", "knn_fused", "knn_x_spec", "knn_x_all", "knn_lexc_spec", "knn_lexc_all", "knn_xc_spec",
               "knn_xc_all")
ADAPT = {"knn_adapt_spec": "knn", "knn_adapt_all": "knn", "knn_lsaadapt_spec": "knn", "knn_lsaadapt_all": "knn",
         "knn_lsacadapt_spec": "knn", "knn_lsacadapt_all": "knn", "proto_adapt": "proto", "proto_lsaadapt": "proto",
         "proto_lsacadapt": "proto"}
CASESIM = ("knn_casesim_e5", "knn_casesim_e5_all", "proto_casesim_e5")
# the sources each derived source reads (computed first, from the same case)
DERIVED = {"K_lex": ["L_star"], "LP": ["L_star"], "H_bge": ["L_star", "dense_bge_maxp"], "H_qwen8b": ["L_star", "dense_qwen8b"],
           "K_qwen8b": ["L_star"], "KH_qwen8b": ["H_qwen8b"], "K_lex_ext": ["L_star"], "colbert_bge": ["dense_bge_maxp"]}
# query-side vector sets read by row, and the sources that read them
VECTOR_SETS = ("bge", "e5", "Qc", "lsa", "clsa")
# per-case CPU transforms of the facts (no GPU): the LSA projections refitted on the development texts
CPU_TRANSFORMS = ("lsa", "clsa")
KNN_FITS = ("memory_proto.build_mem", "memory_proto.sim_lexical (BM25 index over the memory)",
            "memory_proto.sim_char (character TF-IDF over the memory)", "memory_proto.proto_link",
            "memory_proto.proto_block (article prototypes)", "memory_proto.proto_sparse_block (TF-IDF over the training facts)",
            "memory_proto.docexp_block (expanded-article BM25 matrix)", "common.maxp_scores (chunk sort)")
COLBERT_K = 100
PROBE_THREADS = 16
LOCKBOX_GROUP = 5           # the all-development group (training folds 0-4) of base.Ctx / qwen_endpoint.Ctx


def _json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ep():
    from uol.embedding_push import common
    return common


class MaxP:
    """``common.maxp_scores`` with the chunk sort done once (the sort depends on the catalogue only)."""

    def __init__(self, chunk_vecs: np.ndarray, chunk_article: np.ndarray, n_articles: int):
        order = np.argsort(chunk_article, kind="stable")
        self.vecs_t = np.ascontiguousarray(np.asarray(chunk_vecs, dtype=np.float32)[order].T)
        sorted_art = np.asarray(chunk_article)[order]
        self.starts = np.flatnonzero(np.r_[True, sorted_art[1:] != sorted_art[:-1]])
        self.owners = sorted_art[self.starts]
        self.n = int(n_articles)

    def __call__(self, q: np.ndarray) -> np.ndarray:
        sims = np.asarray(q, dtype=np.float32)[None, :] @ self.vecs_t
        out = np.full((1, self.n), -np.inf, dtype=np.float32)
        out[:, self.owners] = np.maximum.reduceat(sims, self.starts, axis=1)
        return out[0]


# --------------------------------------------------------------------------------------------- staged inputs


def staged_root() -> Path | None:
    """Folder of the inputs the bundle does not carry (``timing stage-data``); None in the local profile."""
    import os
    p = os.environ.get("UOL_TIMING_DATA")
    return Path(p) if p else None


def staged_path(rel: str) -> Path:
    """An input of the frozen embedding-push run: from the staged folder (run computer) or its original place."""
    root = staged_root()
    if root is not None:
        return root / rel
    local = {"memory_proto/state": ep().RUN / "memory_proto" / "state",
             "memory_proto/lsa_dev_v1.npy": ep().RUN / "memory_proto" / "lsa_dev_v1.npy",
             "memory_proto/clsa_dev_v1.npy": ep().RUN / "memory_proto" / "clsa_dev_v1.npy",
             "colbert/idf.npy": ep().RUN / "colbert" / "idf.npy",
             "colbert/long/long_articles.npy": ep().RUN / "colbert" / "long" / "long_articles.npy",
             "colbert/long/long_cls.npy": ep().RUN / "colbert" / "long" / "long_cls.npy"}
    if rel in local:
        return local[rel]
    if rel.startswith("memory_proto/casesim/"):
        return ep().RUN / rel
    if rel.startswith("finetune_dense/"):
        return ep().RUN / rel
    if rel.startswith("m1/"):
        return context.active().work / "timing" / rel
    raise KeyError(rel)


def load_array(path: Path) -> np.ndarray:
    """``.npy``, or its packed ``.npy.zsh`` twin (unpacked next to it once, sha256 checked by ``packing``)."""
    from uol.final import packing
    path = Path(path)
    if not path.exists() and path.with_name(path.name + packing.SUFFIX).exists():
        packing.unpack(path.with_name(path.name + packing.SUFFIX))
    return np.load(path)


def release_rows_array(arr: np.ndarray) -> np.ndarray:
    """A staged development-row array as a release-row array (identity in the bundle, whose rows are the development
    rows; in the local profile the staged arrays are the original release-row files)."""
    if len(arr) != len(ep().release().queries):
        raise ValueError(f"array of {len(arr)} rows for a release of {len(ep().release().queries)} rows")
    return arr


# --------------------------------------------------------------------------------------------- hoisted fits


def _rows_key(rows: Sequence[int]) -> str:
    return hashlib.sha1(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest()


def install_hoisting() -> dict[str, Any]:
    """Memoise the group-only fits inside ``memory_proto`` (same arithmetic; the fitted object is kept per memory or
    training-row set). Returns the caches for inspection."""
    from uol.embedding_push import memory_proto as mp
    from uol.facts_to_law.lexical import CHAR_SETTINGS, LexicalIndex
    if getattr(mp, "_uol_timing_hoisted", None):
        return mp._uol_timing_hoisted
    caches: dict[str, dict] = {"mem": {}, "matrix": {}, "lex": {}, "char": {}, "link": {}, "sparse": {}, "proto": {}}
    orig = {"build_mem": mp.build_mem, "proto_link": mp.proto_link, "sim_lexical": mp.sim_lexical, "sim_char": mp.sim_char,
            "proto_sparse_block": mp.proto_sparse_block, "proto_block": mp.proto_block}

    def build_mem(fit_rows, pos_mode="all"):
        key = (_rows_key(fit_rows), pos_mode)
        if key not in caches["mem"]:
            m = orig["build_mem"](fit_rows, pos_mode)
            caches["mem"][key] = m
            matrix = m.matrix()
            caches["matrix"][id(m)] = matrix
            object.__setattr__(m, "matrix", lambda matrix=matrix: matrix)   # frozen dataclass: the same sparse matrix
        return caches["mem"][key]

    def sim_lexical(eval_rows, mem, params=None):
        t = mp.texts()
        key = _rows_key(mem.rows)
        if key not in caches["lex"]:
            caches["lex"][key] = LexicalIndex([t[int(r)] for r in mem.rows], kind="bm25")
        return caches["lex"][key].score([t[int(r)] for r in eval_rows], params)

    def sim_char(eval_rows, mem):
        from sklearn.feature_extraction.text import TfidfVectorizer
        t = mp.texts()
        key = _rows_key(mem.rows)
        if key not in caches["char"]:
            vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
            memory = vectorizer.fit_transform([t[int(r)] for r in mem.rows])
            caches["char"][key] = (vectorizer, memory.T.tocsr())
        vectorizer, memory_t = caches["char"][key]
        queries = vectorizer.transform([t[int(r)] for r in eval_rows])
        return (queries @ memory_t).toarray().astype(np.float32, copy=False)

    def proto_link(train_rows, pos_mode, comp_norm):
        key = (_rows_key(train_rows), pos_mode, bool(comp_norm))
        if key not in caches["link"]:
            caches["link"][key] = orig["proto_link"](train_rows, pos_mode, comp_norm)
        return caches["link"][key]

    def proto_sparse_block(train_rows, eval_rows, kind="word", pos_mode="specific", comp_norm=True):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        nc, t = mp.n_candidates(), mp.texts()
        train_rows = np.asarray(train_rows, dtype=np.int64)
        key = (_rows_key(train_rows), kind, pos_mode, bool(comp_norm))
        if key not in caches["sparse"]:
            if kind == "char":
                vectorizer = TfidfVectorizer(**CHAR_SETTINGS, dtype=np.float32)
            else:
                vectorizer = TfidfVectorizer(token_pattern=mp.TOKEN_PATTERN, lowercase=True, strip_accents="unicode",
                                             sublinear_tf=True, min_df=2, max_features=mp.MAX_FEATURES, norm="l2",
                                             dtype=np.float32)
            x_train = vectorizer.fit_transform([t[int(r)] for r in train_rows]).tocsr()
            link, cited = proto_link(train_rows, pos_mode, comp_norm)
            centroids = normalize((link[cited] @ x_train).tocsr(), norm="l2", axis=1)
            caches["sparse"][key] = (vectorizer, centroids.T.tocsr(), cited)
        vectorizer, centroids_t, cited = caches["sparse"][key]
        x_eval = vectorizer.transform([t[int(r)] for r in eval_rows]).tocsr()
        sims = (x_eval @ centroids_t).toarray().astype(np.float32)
        out = np.empty((len(eval_rows), nc), dtype=np.float32)
        out[:] = sims.min(axis=1, keepdims=True)
        out[:, cited] = sims
        return out

    def proto_block(train_rows, eval_rows, name, pos_mode="specific", comp_norm=True):
        nc = mp.n_candidates()
        train_rows = np.asarray(train_rows, dtype=np.int64)
        v = mp.qvec(name)
        key = (_rows_key(train_rows), name, id(v), pos_mode, bool(comp_norm))
        if key not in caches["proto"]:                 # prototypes read training rows only (never an injected row)
            link, cited = proto_link(train_rows, pos_mode, comp_norm)
            proto = np.asarray(link[cited] @ v[train_rows])
            proto /= np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
            caches["proto"][key] = (np.ascontiguousarray(proto.T), cited)
        proto_t, cited = caches["proto"][key]
        sims = v[np.asarray(eval_rows, dtype=np.int64)] @ proto_t
        out = np.empty((len(eval_rows), nc), dtype=np.float32)
        out[:] = sims.min(axis=1, keepdims=True)
        out[:, cited] = sims
        return out

    mp.build_mem, mp.sim_lexical, mp.sim_char, mp.proto_link, mp.proto_sparse_block, mp.proto_block = (
        build_mem, sim_lexical, sim_char, proto_link, proto_sparse_block, proto_block)
    mp._uol_timing_hoisted = caches
    mp._uol_timing_original = orig
    return caches


# --------------------------------------------------------------------------------------------- the first stage


class FirstStage:
    """The 50 full sources and the ColBERT pairs of one outer development group, or of the all-development group
    (role ``lockbox``, the models of the new cases, E13), for single release rows."""

    def __init__(self, role: str = "dev_fold0", sources: Sequence[str] | None = None, colbert: str = "store"):
        self.role = role
        self.k = LOCKBOX_GROUP if role == "lockbox" else int(role.replace("dev_fold", ""))
        self.fit = tuple(range(5)) if role == "lockbox" else tuple(f for f in range(5) if f != self.k)
        self.sources = list(sources) if sources is not None else list(FULL_ORDER)
        # store (token store of the GPU job) | frozen (dry run) | given (pairs scored elsewhere, ``given_pairs``) | off
        self.colbert_mode = colbert
        self.colbert_chunks: Path | None = None    # chunk plan of the token store (timing_encode bge: bge_chunks.npz)
        self.vectors: dict[str, np.ndarray] = {}
        self.stored: dict[str, np.ndarray] = {}
        # E13: query-side vectors of the rows after the stored ones (the new cases of an extended release), appended
        # to the stored arrays at load: bge, e5, lsa, clsa, Qc, casesim_e<epoch>; ft: (rows, query vectors)
        self.append: dict[str, np.ndarray] = {}
        self.append_ft: tuple[np.ndarray, np.ndarray] | None = None
        self.given_pairs: dict[int, dict[str, np.ndarray]] = {}
        # E13: the group's evaluation rows restricted to these rows (the new cases), so that no original test-set row
        # is touched, not even by the load's warm-up calls or the ft_e5_q coverage check
        self.eval_only: np.ndarray | None = None
        self.load_seconds: dict[str, float] = {}
        self.S: dict[str, Any] = {}

    def _with_append(self, name: str, arr: np.ndarray) -> np.ndarray:
        """``arr`` with the appended rows of ``name`` after it (unchanged when nothing is appended)."""
        if name not in self.append:
            return arr
        extra = np.asarray(self.append[name], dtype=np.float32)
        if extra.ndim != 2 or extra.shape[1] != arr.shape[1]:
            raise ValueError(f"appended {name} vectors of shape {extra.shape} do not fit {arr.shape}")
        out = np.concatenate([np.asarray(arr, dtype=np.float32), extra], axis=0)
        if len(out) != len(ep().release().queries):
            raise ValueError(f"{name}: {len(out)} rows after appending, release has {len(ep().release().queries)}")
        return out

    # ------------------------------------------------------------------ load (timed apart)
    def _timed(self, name: str, fn: Callable[[], Any]) -> Any:
        t0 = time.perf_counter()
        out = fn()
        self.load_seconds[name] = round(self.load_seconds.get(name, 0.0) + time.perf_counter() - t0, 3)
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} first stage load {name}: {time.perf_counter() - t0:.1f}s", flush=True)
        return out

    def needs(self, *names: str) -> bool:
        return any(n in self.sources for n in names)

    def load(self) -> dict[str, float]:
        from uol.embedding_push import base, memory_proto as mp
        c = ep()
        install_hoisting()
        self.eval_rows, self.train_rows = mp.group_by_role(self.role)
        if self.eval_only is not None:
            only = np.asarray(self.eval_only, dtype=np.int64)
            if not np.isin(only, self.eval_rows).all():
                raise ValueError(f"eval_only rows are not evaluation rows of {self.role}")
            self.eval_rows = only
        S = self.S
        S["ctx"] = self._timed("base_context", base.Ctx)
        S["g"] = S["ctx"].groups[self.k]
        assert S["g"].role == self.role and S["g"].train_folds == self.fit, (S["g"].role, S["g"].train_folds)
        S["nested"] = base.Nested(S["ctx"])
        S["lexical_sel"] = _json(c.RUN / "base" / "selection" / "lexical.json")[self.role]["selected"]
        S["nested_sel"] = _json(c.RUN / "base" / "selection" / "nested.json")[self.role]
        # vectors read by row (writable copies, so a case's fresh vectors can be written in): ``vectors`` as stored
        # (read by base.py and the Qwen code), ``unit`` as memory_proto registers them (``_unit`` of the stored file)
        self.stored["bge"] = self._with_append("bge", np.load(c.DERIVED / "bge_m3_chunks_v1" / "query_vectors.npy").astype(np.float32))
        self.stored["e5"] = self._with_append("e5", np.load(c.DERIVED / "e5_small_queries_v11" / "query_vectors.npy").astype(np.float32))
        lsa = release_rows_array(self._with_append("lsa", load_array(staged_path("memory_proto/lsa_dev_v1.npy"))))
        clsa = release_rows_array(self._with_append("clsa", load_array(staged_path("memory_proto/clsa_dev_v1.npy"))))
        self.stored["lsa"], self.stored["clsa"] = lsa.astype(np.float32), clsa.astype(np.float32)
        if self.needs("dense_qwen8b", "H_qwen8b", "knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b", "K_qwen8b", "KH_qwen8b",
                      "K_lex_ext"):
            from uol.embedding_push import qwen_endpoint as qe
            self.stored["Qc"] = self._with_append("Qc", np.asarray(
                self._timed("qwen_query_vectors", lambda: qe.query_vectors("c")), dtype=np.float32))
        self.reset_vectors()
        mp.ensure_lsa = lambda n_components=512: staged_path("memory_proto/lsa_dev_v1.npy")
        mp.ensure_clsa = lambda n_components=512: staged_path("memory_proto/clsa_dev_v1.npy")
        # article-side vectors
        if self.needs("dense_bge_maxp", "dense_bge_firstp", "dense_bge_long", "H_bge", "colbert_bge"):
            d = c.DERIVED / "bge_m3_chunks_v1"
            S["bge_c"] = self._timed("bge_chunks", lambda: load_array(d / "chunk_vectors.npy").astype(np.float32))
            S["bge_owner"] = c.chunk_article_index(d / "units.jsonl")
            units = [json.loads(x) for x in (d / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
            chunk0 = np.asarray([u["chunk_index"] for u in units])
            owner = S["bge_owner"]
            order = np.argsort(owner, kind="stable")
            first_pos = order[np.flatnonzero(np.r_[True, owner[order][1:] != owner[order][:-1]])]
            assert (chunk0[first_pos] == 0).all()
            S["bge_first"] = np.ascontiguousarray(S["bge_c"][first_pos])
            n_c = len(c.release().candidate_ids)
            first = np.full(n_c, -1, dtype=np.int64)
            for u in range(len(owner) - 1, -1, -1):
                first[owner[u]] = u
            articles = S["bge_c"][first].astype(np.float32)
            articles[np.load(staged_path("colbert/long/long_articles.npy"))] = np.load(staged_path("colbert/long/long_cls.npy"))
            S["bge_long"] = articles
        if self.needs("dense_e5_maxp", "ft_e5_q"):
            d = c.DERIVED / "e5_small_chunks_v1"
            S["e5_c"] = self._timed("e5_chunks", lambda: load_array(d / "chunk_vectors.npy").astype(np.float32))
            S["e5_owner"] = np.load(d / "article_index.npy").astype(np.int64)
        if "bge_c" in S:
            S["bge_maxp"] = self._timed("bge_maxp_sort", lambda: MaxP(S["bge_c"], S["bge_owner"], len(c.release().candidate_ids)))
        if "e5_c" in S:
            S["e5_maxp"] = self._timed("e5_maxp_sort", lambda: MaxP(S["e5_c"], S["e5_owner"], len(c.release().candidate_ids)))
        # base: catalogue indexes, memory, prior, the extras' character index
        sel = base.LSel.from_json(S["lexical_sel"])
        kinds = {sel.kind} if sel.kind in ("bm25", "bm25_stem", "char") else {sel.base, "char"}
        for kind in sorted(kinds):
            self._timed(f"catalogue_index_{kind}", lambda kind=kind: S["ctx"].lex(kind))
        S["lsel"] = sel
        self._timed("memory_bm25", lambda: S["nested"].memory(self.fit))
        S["prior"] = self._timed("prior", lambda: S["nested"].prior(self.fit))
        if self.needs("knn_lex_instr_votes", "knn_char_votes"):
            self._timed("extras_indexes", self._load_extras)
        if self.needs(*KNN_METHODS, *ADAPT, *CASESIM, "proto_bge", "proto_e5", "proto_tfidf"):
            self._timed("memory_proto_memories", self._warm_memory_proto)
        if self.needs("docexp_bm25", "docexp_facts_bm25"):
            S["docexp"] = self._timed("docexp_counts", lambda: mp.DocExpGroup(self.train_rows))
            for m in ("docexp_bm25", "docexp_facts_bm25"):
                if m in self.sources:
                    self._timed("docexp_documents", lambda m=m: self.docexp_documents(m))
        if self.needs(*ADAPT):
            self._timed("adapters", self._load_adapters)
        if self.needs("proto_probe"):
            self._timed("probe", self._load_probe)
        if self.needs(*CASESIM):
            S["casesim"] = {}
            for m in CASESIM:
                e = self.state(m)["epoch"]
                S["casesim"][m] = mp._unit(release_rows_array(self._with_append(f"casesim_e{e}", load_array(
                    staged_path(f"memory_proto/casesim/e5_{self.role}/B/epoch{e}.npy")))))
        if self.needs("ft_e5_q"):
            self._timed("ft_e5_q_vectors", self._load_ft)
        if any(s.endswith("qwen8b") for s in self.sources) or self.needs("K_lex_ext"):
            self._timed("qwen_fit", self._load_qwen)
        if "colbert_bge" in self.sources and self.colbert_mode != "off":
            self._timed("colbert", self._load_colbert)
        return self.load_seconds

    unit: dict[str, np.ndarray]

    def reset_vectors(self, rows: Sequence[int] | None = None) -> None:
        """Query-side arrays back to the stored vectors (all, or the given rows; in place, so the references the
        producers hold stay valid)."""
        from uol.embedding_push import memory_proto as mp
        if not hasattr(self, "unit"):
            self.unit = {}
        for k, v in self.stored.items():
            if k not in self.vectors:
                self.vectors[k] = v.copy()
                if k != "Qc":
                    self.unit[k] = mp._unit(v)
            elif rows is None:
                self.vectors[k][:] = v
                if k != "Qc":
                    self.unit[k][:] = mp._unit(v)
            else:
                r = np.asarray(rows, dtype=np.int64)
                self.vectors[k][r] = v[r]
                if k != "Qc":
                    self.unit[k][r] = mp._unit(v[r])
        for k, u in self.unit.items():
            mp._VEC[k] = u

    def inject(self, row: int, fresh: dict[str, np.ndarray]) -> None:
        """Write one case's freshly encoded query-side vectors (unit norm) into the arrays the producers read."""
        for k, v in fresh.items():
            if k not in self.vectors:
                continue
            v = np.asarray(v, dtype=np.float32).ravel()
            v = v / max(float(np.linalg.norm(v)), 1e-12)
            self.vectors[k][int(row)] = v
            if k in self.unit:
                self.unit[k][int(row)] = v

    def state(self, method: str) -> dict:
        return _json(staged_path("memory_proto/state") / f"{method}.{self.role}.json")

    def manifest_cfg(self, method: str) -> Any:
        return _json(ep().SCORES / method / "manifest.json")["configuration"]

    def qsel(self, name: str) -> dict:
        from uol.embedding_push import qwen_endpoint as qe
        return _json(qe.SELECTION_DIR / f"{name}.json")

    def _load_extras(self) -> None:
        from scipy import sparse
        from uol.facts_to_law import lexical
        from uol.facts_to_law import spec as ftl_spec
        S = self.S
        ctx = S["ctx"]
        d = ctx.data
        instr = [row["instrument_id"] for row in ftl_spec.read_jsonl(Path(d.candidate_binding["articles_path"]))]
        code = {v: i for i, v in enumerate(sorted(set(instr)))}
        S["cand_instr"] = np.asarray([code[v] for v in instr])
        S["incidence_t"] = sparse.csr_matrix((np.ones(ctx.N, np.float32), (S["cand_instr"], np.arange(ctx.N))),
                                             shape=(len(code), ctx.N))
        mem, _ = S["nested"].memory(self.fit)
        S["char_mem"] = lexical.LexicalIndex([ctx.texts[d.query_index[q]] for q in mem.query_ids], kind="char")

    def _warm_memory_proto(self) -> None:
        """Build every memory, index and prototype link the group's memory-proto sources use (hoisted fits)."""
        from uol.embedding_push import memory_proto as mp
        r = self.eval_rows[:1]
        for m in [m for m in KNN_METHODS if m in self.sources]:
            mp.knn_outer_block(r, self.train_rows, mp.Cfg(**self.state(m)["selected"]))
        for name in ("bge", "e5"):
            if f"proto_{name}" in self.sources:
                cfg = self.manifest_cfg(f"proto_{name}")
                mp.proto_block(self.train_rows, r, name, cfg["pos_mode"], cfg["comp_norm"])
        if "proto_tfidf" in self.sources:
            cfg = self.manifest_cfg("proto_tfidf")
            mp.proto_sparse_block(self.train_rows, r, cfg["tfidf"], cfg["pos_mode"], cfg["comp_norm"])

    def _load_adapters(self) -> None:
        import torch
        from uol.embedding_push import case_sim as cs
        self.S["adapters"] = {}
        torch.set_num_threads(4)                  # the frozen adapters were trained with 4 CPU threads (c_score)
        for m in [m for m in ADAPT if m in self.sources]:
            st = self.state(m)
            key = json.dumps(st["recipe"], sort_keys=True)
            if key not in self.S["adapters"]:
                r = dict(st["recipe"])
                r["epochs"] = tuple(r["epochs"])
                self.S["adapters"][key] = cs.train_adapter(cs.AdaptRecipe(**r), self.train_rows)
            self.adapter_vectors(st)

    def adapter_vectors(self, st: dict) -> np.ndarray:
        """Unit rows of one adapter epoch (``register_array`` semantics), made once."""
        key = ("adapter_unit", json.dumps(st["recipe"], sort_keys=True), int(st["epoch"]))
        if key not in self.S:
            from uol.embedding_push import memory_proto as mp
            self.S[key] = mp._unit(self.S["adapters"][key[1]][st["epoch"]])
        return self.S[key]

    def _load_probe(self) -> None:
        import torch
        from uol.embedding_push import case_sim as cs
        st = self.state("proto_probe")
        r = dict(st["recipe"])
        r["epochs"] = tuple(r["epochs"])
        recipe = cs.ProbeRecipe(**r)
        before = torch.get_num_threads()
        # the frozen probe ran with torch's default 16 threads on the thesis PC (run_probe workers); at 4 threads its
        # 40-epoch training drifts by 0.03, at 16 it reproduces to 1e-7
        torch.set_num_threads(PROBE_THREADS)
        try:
            cited, classes = cs.train_probe(recipe, self.train_rows)[st["epoch"]]
        finally:
            torch.set_num_threads(before)
        self.S["probe"] = (recipe, cited, classes)

    def _load_ft(self) -> None:
        if self.eval_only is not None:              # E13: only the new cases' vectors from the group's own tower
            rows, q = np.asarray(self.append_ft[0]), np.asarray(self.append_ft[1], dtype=np.float32)
        else:
            z = np.load(staged_path(f"finetune_dense/cache/e5-8b32e14f/{self.role}.npz"))
            rows, q = np.asarray(z["rows"]), np.asarray(z["qvecs"], dtype=np.float32)
            if self.append_ft is not None:          # E13: the new cases' vectors from the group's own tower
                rows = np.concatenate([rows, np.asarray(self.append_ft[0])])
                q = np.concatenate([q, np.asarray(self.append_ft[1], dtype=np.float32)])
        pos = {int(r): i for i, r in enumerate(rows)}
        if set(pos) != set(int(r) for r in self.eval_rows):
            raise ValueError("ft_e5_q cached query vectors do not cover the group's evaluation rows")
        self.S["ft_q"], self.S["ft_pos"] = q, pos

    def _load_qwen(self) -> None:
        from uol.embedding_push import qwen_endpoint as qe
        qe._VEC["Qc"] = self.vectors["Qc"]
        ctx = qe.Ctx()
        g = ctx.groups[self.k]
        assert g.role == self.role and tuple(g.train_folds) == self.fit
        self.S["qe_ctx"], self.S["qe_fit"] = ctx, ctx.fit(g.train_folds)
        # warm the memory index and the two prototype sets the sources read
        sims = qe.Sims(self.S["qe_fit"], self.eval_rows[:1])
        for cfg in self._qwen_configs():
            qe.config_block(sims, cfg)

    def _qwen_configs(self) -> list[str]:
        out = [self.qsel(m)[self.role]["alone"]["config"] for m in ("knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b")]
        out += [self.qsel("K_qwen8b")[self.role]["config"], self.qsel("KH_qwen8b")["pick"][self.role]["config"],
                self.qsel("lex_control")[self.role]["fused"]["config"]]
        return out

    def _load_colbert(self) -> None:
        from uol.facts_to_law import lexical
        data = ep().release()
        S = self.S
        S["colbert_bm25"] = lexical.LexicalIndex(data.candidate_texts, "bm25", {"k1": 1.2, "b": 0.75})
        if self.colbert_mode == "store":
            from uol.embedding_push import colbert
            store = colbert_store()
            if self.colbert_chunks is None:
                raise FileNotFoundError("colbert_bge needs the token store's chunk plan (timing encode --models bge)")
            chunks = dict(np.load(self.colbert_chunks))
            S["colbert_index"] = colbert.article_index(chunks, len(data.candidate_ids))
            S["colbert_docs"] = np.load(store / "doc_tokens.npy", mmap_mode="r")
            if int(chunks["n_vec"].sum()) != S["colbert_docs"].shape[0]:
                raise ValueError("the chunk plan does not match the ColBERT token store")
        elif self.colbert_mode == "given":          # E13: pairs scored on the run computer, checked against the pool
            pass
        else:                                       # dry run without the token store: the frozen pair scores
            z = np.load(ep().SCORES / "colbert_bge" / "pairs.npz")
            S["colbert_frozen"] = {k: np.asarray(z[k]) for k in ("qidx", "cidx", "score")}

    # ------------------------------------------------------------------ per case
    def score(self, sid: str, row: int, done: dict[str, np.ndarray], qtok: dict[str, np.ndarray] | None = None) -> Any:
        """[N] float32 scores of source ``sid`` for release row ``row`` (pairs dict for colbert_bge)."""
        return getattr(self, "_" + sid if hasattr(self, "_" + sid) else "_generic")(sid, int(row), done, qtok)

    def _generic(self, sid, row, done, qtok):
        from uol.embedding_push import memory_proto as mp
        r = np.asarray([row], dtype=np.int64)
        if sid in KNN_METHODS:
            return mp.knn_outer_block(r, self.train_rows, mp.Cfg(**self.state(sid)["selected"]))[0]
        if sid in ADAPT:
            st = self.state(sid)
            mp._VEC["cs"] = self.adapter_vectors(st)
            if ADAPT[sid] == "knn":
                return mp.knn_outer_block(r, self.train_rows, mp.Cfg(**st["selected"]))[0]
            return mp.proto_block(self.train_rows, r, "cs", "specific", True)[0]
        if sid in CASESIM:
            mp._VEC["cs"] = self.S["casesim"][sid]                   # unit rows (register_array semantics)
            if sid.startswith("proto_"):
                return mp.proto_block(self.train_rows, r, "cs", "specific", True)[0]
            return mp.knn_outer_block(r, self.train_rows, mp.Cfg(**self.state(sid)["selected"]))[0]
        if sid in ("knn_lex_votes_k5", "knn_lex_votes_k50", "knn_lex_instr_votes", "knn_char_votes"):
            return self._lex_extras(sid, row)
        raise KeyError(sid)

    # base.py sources
    def _L_star(self, sid, row, done, qtok):
        from uol.embedding_push import base
        return base.lstar_rows(self.S["ctx"], self.S["lsel"], np.asarray([row]))[0]

    def _dense_bge_maxp(self, sid, row, done, qtok):
        return self.S["bge_maxp"](self.vectors["bge"][row])

    def _dense_bge_firstp(self, sid, row, done, qtok):
        return (self.vectors["bge"][row][None, :] @ self.S["bge_first"].T).astype(np.float32)[0]

    def _dense_bge_long(self, sid, row, done, qtok):
        return (self.vectors["bge"][row][None, :] @ self.S["bge_long"].T).astype(np.float32)[0]

    def _dense_e5_maxp(self, sid, row, done, qtok):
        return self.S["e5_maxp"](self.vectors["e5"][row])

    def _ft_e5_q(self, sid, row, done, qtok):
        return self.S["e5_maxp"](self.S["ft_q"][self.S["ft_pos"][row]])

    def _prior(self, sid, row, done, qtok):
        return np.asarray(self.S["prior"], dtype=np.float32)

    def _votes(self, row: int, ks: Sequence[int]) -> dict[int, np.ndarray]:
        return self.S["nested"].votes(np.asarray([row]), self.fit, sorted(set(int(k) for k in ks)))

    def _knn_lex_votes(self, sid, row, done, qtok):
        k = self.S["nested_sel"]["knn_lex_votes"]["k"]
        return self._votes(row, [k])[k][0]

    def _K_lex(self, sid, row, done, qtok):
        from uol.embedding_push import base
        from uol.facts_to_law import fusion
        sel = self.S["nested_sel"]["K_lex"]
        zl = fusion.robust_z(done["L_star"][None, :])
        zv = fusion.robust_z(self._votes(row, [sel["k"]])[sel["k"]])
        return fusion.linear_fusion([zl, zv], base.pair(sel["w_knn"]), normalize="none")[0]

    def _LP(self, sid, row, done, qtok):
        from uol.facts_to_law import fusion
        wp = self.S["nested_sel"]["LP"]["w_prior"]
        zl = fusion.robust_z(done["L_star"][None, :])
        zp = fusion.robust_z(self.S["prior"])
        return (fusion.add_prior(zl, zp, wp, normalize="none") if wp else zl)[0]

    def _H_bge(self, sid, row, done, qtok):
        from uol.embedding_push import base
        from uol.facts_to_law import fusion
        cfg = self.S["nested_sel"]["H_bge"]["config"]
        lraw, draw = done["L_star"][None, :], done["dense_bge_maxp"][None, :]
        if cfg["type"] == "rrf":
            return fusion.rrf([lraw, draw], k=base.RRF_K).astype(np.float32)[0]
        return fusion.linear_fusion([fusion.robust_z(lraw), fusion.robust_z(draw)], base.pair(cfg["w_dense"]),
                                    normalize="none")[0]

    def _lex_extras(self, sid: str, row: int) -> np.ndarray:
        from uol.facts_to_law import knn
        S = self.S
        ctx = S["ctx"]
        mem, bm25 = S["nested"].memory(self.fit)
        texts = [ctx.texts[row]]
        kw = {"eval_components": [ctx.data.queries[row].component_id], "fit_components": list(mem.components)}
        if sid == "knn_char_votes":
            return knn.knn_votes(S["char_mem"].score(texts), mem.positives, 20, ctx.N, "sim", **kw)[0]
        sim = bm25.score(texts)
        if sid == "knn_lex_votes_k5":
            return knn.knn_votes(sim, mem.positives, 5, ctx.N, "sim", **kw)[0]
        v50 = knn.knn_votes(sim, mem.positives, 50, ctx.N, "sim", **kw)
        if sid == "knn_lex_votes_k50":
            return v50[0]
        return np.asarray((S["incidence_t"] @ v50.T).T, dtype=np.float32)[:, S["cand_instr"]][0]

    # memory_proto prototypes and document expansion
    def _proto_bge(self, sid, row, done, qtok):
        from uol.embedding_push import memory_proto as mp
        cfg = self.manifest_cfg(sid)
        return mp.proto_block(self.train_rows, np.asarray([row]), sid.split("_", 1)[1], cfg["pos_mode"], cfg["comp_norm"])[0]

    _proto_e5 = _proto_bge

    def _proto_tfidf(self, sid, row, done, qtok):
        from uol.embedding_push import memory_proto as mp
        cfg = self.manifest_cfg(sid)
        return mp.proto_sparse_block(self.train_rows, np.asarray([row]), cfg["tfidf"], cfg["pos_mode"], cfg["comp_norm"])[0]

    def docexp_documents(self, sid: str) -> tuple[Any, np.ndarray | None]:
        """The BM25 document matrix of ``memory_proto.docexp_block`` (it depends on the group only), built once."""
        key = ("docexp_docs", sid)
        if key not in self.S:
            from uol.retrieval_benchmarks.c01_statutes.experiments import _bm25_document_matrix
            cfg, group = self.manifest_cfg(sid), self.S["docexp"]
            expansion = group.expansion(cfg["cap"], cfg["pos_mode"])
            if cfg["mode"] == "concat":
                counts = (group.articles + np.float32(1.0) * expansion).tocsr()
                self.S[key] = (_bm25_document_matrix(counts, k1=cfg["k1"], b=cfg["b"]).T.tocsr(), None)
            else:
                cited = np.flatnonzero(np.diff(expansion.indptr) > 0)
                self.S[key] = (_bm25_document_matrix(expansion[cited], k1=cfg["k1"], b=cfg["b"]).T.tocsr(), cited)
        return self.S[key]

    def _docexp_bm25(self, sid, row, done, qtok):
        from uol.embedding_push import memory_proto as mp
        docs_t, cited = self.docexp_documents(sid)
        queries = self.S["docexp"].queries([row])
        if cited is None:
            return (queries @ docs_t).toarray().astype(np.float32, copy=False)[0]
        out = np.zeros((1, mp.n_candidates()), dtype=np.float32)
        out[:, cited] = (queries @ docs_t).toarray()
        return out[0]

    _docexp_facts_bm25 = _docexp_bm25

    def _proto_probe(self, sid, row, done, qtok):
        from uol.embedding_push import case_sim as cs
        recipe, cited, classes = self.S["probe"]
        return cs.probe_block(recipe, cited, classes, np.asarray([row]))[0]

    # Qwen3-Embedding-8B
    def _qsims(self, row: int):
        from uol.embedding_push import qwen_endpoint as qe
        key = ("qe_sims", row, id(self.vectors["Qc"]), self._qc_version)
        if self.S.get("qe_sims_key") != key:
            self.S["qe_sims"], self.S["qe_sims_key"] = qe.Sims(self.S["qe_fit"], np.asarray([row])), key
        return self.S["qe_sims"]

    _qc_version = 0

    def _qcfg(self, row: int, cfg: str) -> np.ndarray:
        from uol.embedding_push import qwen_endpoint as qe
        return qe.config_block(self._qsims(row), cfg)

    def _dense_qwen8b(self, sid, row, done, qtok):
        from uol.embedding_push import qwen_endpoint as qe
        assert self.qsel("dense")["selected"] == "c"
        return qe.dense_block(self.vectors["Qc"][[row]])[0]

    def _H_qwen8b(self, sid, row, done, qtok):
        from uol.embedding_push import qwen_endpoint as qe
        from uol.facts_to_law import fusion
        w = self.qsel("H_qwen8b")[self.role]["w_dense"]
        return fusion.linear_fusion([fusion.robust_z(done["L_star"][None, :]), fusion.robust_z(done["dense_qwen8b"][None, :])],
                                    qe.pair(w), normalize="none")[0]

    def _q_alone(self, sid, row, done, qtok):
        return self._qcfg(row, self.qsel(sid)[self.role]["alone"]["config"])[0]

    _knn_qwen8b = _knn_fused_qwen8b = _proto_qwen8b = _q_alone

    def _q_fused(self, row: int, base_row: np.ndarray, sel: dict) -> np.ndarray:
        from uol.embedding_push import qwen_endpoint as qe
        from uol.facts_to_law import fusion
        return fusion.linear_fusion([fusion.robust_z(base_row[None, :]), fusion.robust_z(self._qcfg(row, sel["config"]))],
                                    qe.pair(sel["w"]), normalize="none")[0]

    def _K_qwen8b(self, sid, row, done, qtok):
        return self._q_fused(row, done["L_star"], self.qsel("K_qwen8b")[self.role])

    def _KH_qwen8b(self, sid, row, done, qtok):
        return self._q_fused(row, done["H_qwen8b"], self.qsel("KH_qwen8b")["pick"][self.role])

    def _K_lex_ext(self, sid, row, done, qtok):
        return self._q_fused(row, done["L_star"], self.qsel("lex_control")[self.role]["fused"])

    # ColBERT pairs (bge-m3 late interaction over the BM25 / bge pool)
    def _colbert_bge(self, sid, row, done, qtok):
        from uol.embedding_push import colbert
        data = ep().release()
        S = self.S
        bm25 = S["colbert_bm25"].score([data.queries[row].text])[0]
        bt = np.argsort(-bm25, kind="stable")[:COLBERT_K]
        dt = np.argsort(-done["dense_bge_maxp"], kind="stable")[:COLBERT_K]
        union = np.union1d(bt, dt)
        if self.colbert_mode == "frozen":
            f = S["colbert_frozen"]
            m = f["qidx"] == row
            pos = {int(c): float(s) for c, s in zip(f["cidx"][m], f["score"][m])}
            score = np.asarray([pos.get(int(c), np.nan) for c in union], dtype=np.float32)
            return {"cidx": union.astype(np.int32), "score": score, "basis": "frozen pair scores (dry run)"}
        if self.colbert_mode == "given":
            g = self.given_pairs[int(row)]
            if not np.array_equal(np.sort(np.asarray(g["cidx"], dtype=np.int64)), union.astype(np.int64)):
                raise ValueError(f"row {row}: the given ColBERT pairs are not this case's pool (BM25 top 100 + bge top 100)")
            pos = {int(c): float(s) for c, s in zip(g["cidx"], g["score"])}
            return {"cidx": union.astype(np.int32), "score": np.asarray([pos[int(c)] for c in union], dtype=np.float32),
                    "basis": "given (run computer token store)"}
        if qtok is None:
            raise ValueError("colbert_bge needs the case's query token vectors")
        weights = np.ones((len(qtok["tokens"]), 1), dtype=np.float32)        # PRIMARY uniform_maxp: uniform weights
        maxp, _ = colbert.score_query(np.asarray(qtok["tokens"], dtype=np.float32), weights, union, S["colbert_index"],
                                      S["colbert_docs"], 49152, "cpu")
        return {"cidx": union.astype(np.int32), "score": maxp[:, 0].astype(np.float32), "basis": "token store"}

    # ------------------------------------------------------------------ one case
    def case(self, row: int, qtok: dict | None = None, clock=None) -> dict[str, Any]:
        """Every source of this stage for one row: {sid: [N] scores} (colbert_bge: pairs), timed per source."""
        done: dict[str, Any] = {}
        for sid in self.order():
            if clock is not None:
                with clock.stage(f"source:{sid}"):
                    done[sid] = self.score(sid, row, done, qtok)
            else:
                done[sid] = self.score(sid, row, done, qtok)
        return done

    def order(self) -> list[str]:
        """Sources in dependency order (derived ones after their inputs)."""
        want = list(self.sources)
        out: list[str] = []

        def add(s: str) -> None:
            if s in out:
                return
            for d in DERIVED.get(s, []):
                add(d)
            out.append(s)
        for s in want:
            add(s)
        return out


# --------------------------------------------------------------------------------------------- reproduction checks


def frozen_row(sid: str, row: int) -> np.ndarray:
    return np.asarray(np.load(ep().SCORES / sid / "scores.npy", mmap_mode="r")[int(row)], dtype=np.float32)


@lru_cache(maxsize=1)
def frozen_pairs() -> dict[str, np.ndarray]:
    z = np.load(ep().SCORES / "colbert_bge" / "pairs.npz")
    return {k: np.asarray(z[k]) for k in ("qidx", "cidx", "score")}


def compare_row(sid: str, row: int, got: np.ndarray, topk: int = 200) -> dict[str, Any]:
    """Recomputed vs frozen source row: max |diff|, whether the top-200 (the pool contribution) is identical, and
    whether it is equivalent: every member that differs scores within ``1e-5`` x scale of the frozen 200th score, a
    boundary tie that float rounding may break either way."""
    from uol.embedding_push import fusion_ltr as fl
    ref = frozen_row(sid, row)
    a, b = fl.sanitize(np.asarray(got, dtype=np.float32)[None, :]), fl.sanitize(ref[None, :])
    diff = np.abs(a - b)
    ia, ib = fl.topk_block(a, topk)[0], fl.topk_block(b, topk)[0]
    ta, tb = set(ia.tolist()), set(ib.tolist())
    scale = float(np.nanmax(np.abs(b))) or 1.0
    kth = float(np.min(b[0, ib]))
    tie = all(abs(float(b[0, c]) - kth) <= 1e-5 * scale for c in ta ^ tb)
    return {"max_abs_diff": float(np.nanmax(diff)), "rel": float(np.nanmax(diff)) / scale,
            "top200_identical": ta == tb, "top200_equivalent": ta == tb or tie, "top200_overlap": len(ta & tb)}


def compare_pairs(row: int, got: dict[str, Any]) -> dict[str, Any]:
    f = frozen_pairs()
    m = f["qidx"] == int(row)
    fc, fs = f["cidx"][m], f["score"][m]
    same_pool = set(fc.tolist()) == set(np.asarray(got["cidx"]).tolist())
    pos = {int(c): float(s) for c, s in zip(fc, fs)}
    d = np.asarray([abs(float(s) - pos[int(c)]) for c, s in zip(got["cidx"], got["score"]) if int(c) in pos])
    return {"pool_identical": same_pool, "n_pairs": int(len(got["cidx"])), "n_frozen": int(m.sum()),
            "max_abs_diff": float(d.max()) if len(d) else None, "mean_abs_diff": float(d.mean()) if len(d) else None}


def colbert_store() -> Path:
    """The ColBERT token store of the run computer's ``colbert_full`` GPU job (``<work>/colbert_full/store``)."""
    import os
    p = os.environ.get("UOL_TIMING_COLBERT_STORE")
    if p:
        return Path(p)
    prof = context.active()
    root_work = prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work
    return root_work / "colbert_full" / "store"
