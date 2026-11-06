"""EP-alpha phase 2: candidate pool, features, hand fusions and learned fusion over all delivered score files.

``python -m uol.embedding_push.fusion_ltr --rebuild`` re-runs everything from the files present in ``scores/``
(a source is COMPLETE when its ``manifest.json`` exists; ``scores.npy`` = full float32 matrix, ``pairs.npz`` = sparse
``qidx/cidx/score`` rerank features) and rewrites ``fusion/DEV_RESULTS.md``.

Design
------
* Pool (label-free): per query the union of the top-300 of every full source of the family (NO-EMB pools use only
  NO-EMB sources, so the NO-EMB pipelines never see an embedding-retrieved candidate). Positives are never injected;
  a positive outside the pool ranks below every pooled candidate, ordered by L_star.
* Features per (query, pooled candidate): for each source the raw score, per-query robust-z over the full candidate
  row and log (min-)rank; sparse sources add raw / z / log-rank over the supplied pairs and a presence indicator;
  plus log body length, a fold-local instrument citation prior and log query length.
* Models, trained per scoring group on that group's training rows only, from their out-of-fold features:
  (a) linear fusion of robust-z features (coordinate ascent on training-row P_spec nDCG@10);
  (b) pointwise HistGradientBoostingClassifier (positives + 50 sampled pool negatives per query), learning rate and
  depth chosen by inner CV (inner folds = the group's training folds).
* Stacking caveat: a training row's features are its own out-of-fold features, computed by models that saw the group's
  held-out fold; the held-out fold's labels therefore reach the group's model only indirectly (second order). The
  lockbox group is trained on all dev rows and is unaffected.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import base, common
from uol.facts_to_law import fusion, spec
from uol.facts_to_law import prior as prior_mod

FUSION = common.RUN / "fusion"
PIPES = FUSION / "pipelines"
CACHE = FUSION / "cache"
FEATDIR = FUSION / "features"
FLOG = FUSION / "run.log"
TOPK = 200
FEATURE_VERSION = "4"
NOEMB_IDS = {"L_star", "prior", "knn_lex_votes", "K_lex", "LP", "knn_lex_votes_k5", "knn_lex_votes_k50", "knn_lex_instr_votes", "knn_char_votes", "K_lex_ext"}
DERIVED_IDS = {"K_lex", "LP", "H_bge", "H_qwen8b", "K_qwen8b", "K_lex_ext"}          # linear / rank fusions of other sources (left out of the linear model)
W_PRIOR2 = (0.0, 0.1, 0.2, 0.3, 0.5)
HAND_W_DENSE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
HAND_VERSION = "2"
LINEAR_VERSION = "3"
LINEAR_THREADS = 3
HGB_VERSION = "2"
HGB_GRID = [{"learning_rate": lr, "max_depth": d} for lr in (0.05, 0.1) for d in (3, 6)]
HGB_FIXED = {"max_iter": 150, "min_samples_leaf": 40, "l2_regularization": 1.0, "early_stopping": False,
             "max_features": 0.5, "max_bins": 127}
INNER_FOLDS = 2          # hyper-parameter selection: the first two training folds are held out in turn
N_NEG = 50
SEED = 20261004
LIN_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0,
            12.0, 16.0, 24.0, 32.0)
LIN_REFINE = (0.6, 0.8, 0.9, 1.1, 1.25, 1.6)
THREADS = 8
CODE_FILES = [Path(__file__), Path(base.__file__), Path(common.__file__)]
LOG2 = 1.0 / np.log2(np.arange(2, 12))
IDCG = np.concatenate([[0.0], np.cumsum(LOG2[:10])])


def flog(message: str) -> None:
    line = f"[{base.utc()}] {message}"
    print(line, flush=True)
    FUSION.mkdir(parents=True, exist_ok=True)
    with FLOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


# ----------------------------------------------------------------------------- sources


@dataclass
class Source:
    id: str
    kind: str                # full | pairs
    family: str              # NO-EMB | EMB
    group: str               # feature group (ablation unit)
    derived: bool
    sha: str
    manifest: dict[str, Any] = field(default_factory=dict)
    lsa: bool = False        # built on LSA vectors (NO-EMB by the lead's ruling D4; a sensitivity analysis drops these)

    @property
    def folder(self) -> Path:
        return common.SCORES / self.id


EMB_PATTERN = re.compile(r"(?<![0-9a-z])(bge|e5|qwen[0-9a-z]*)(?![0-9a-z])|embed|colbert|encoder|fine.?tun|sentence.?transformer|transformer")
GROUP_RULES = [(r"qwen", "qwen8b"), (r"(^|_)(ft|finetuned|finetune|tuned)(_|$)", "finetuned_dense"), (r"colbert|multivec|late_inter", "colbert"),
               (r"^proto", "prototypes"), (r"case|learned|lcs", "learned_case_similarity"), (r"^knn_", "dense_memory"),
               (r"^(dense|H_bge)", "frozen_dense")]


def family_of(sid: str, manifest: dict[str, Any]) -> str:
    """NO-EMB only for the lexical sources of this study and for files that declare NO-EMB with no embedding vocabulary."""
    declared = str(manifest.get("family") or "").upper()
    if sid in NOEMB_IDS:
        return "EMB" if declared == "EMB" else "NO-EMB"
    text = json.dumps({k: manifest.get(k) for k in ("method", "kind", "description", "configuration", "selection", "training_data")},
                      default=str).lower().replace("embedding_push", "")
    if declared != "EMB" and (sid.startswith("docexp_") or declared == "NO-EMB") and not EMB_PATTERN.search(text):
        return "NO-EMB"
    return "EMB"


def group_of(sid: str, manifest: dict[str, Any]) -> str:
    """Ablation unit: the manifest's ``feature_group`` when given, else a concept group derived from the id."""
    if manifest.get("feature_group"):
        return str(manifest["feature_group"])
    for pattern, name in GROUP_RULES:
        if re.search(pattern, sid):
            return name
    return re.sub(r"_(maxp|firstp|meanp|pairs|votes|scores)$", "", sid)


def is_lsa(sid: str, manifest: dict[str, Any]) -> bool:
    """Sources built on LSA vectors: 'lsa' in the id, in the kind/description, or declared as the adapter input."""
    if "lsa" in sid.lower():
        return True
    text = json.dumps({k: manifest.get(k) for k in ("method", "kind", "description")}, default=str).lower()
    config = json.dumps(manifest.get("configuration"), default=str).lower()
    return bool(re.search(r"(?<![a-z])lsa(?![a-z])", text) or re.search(r'"inputs": "lsa"', config))


def discover(exclude: Sequence[str] = ()) -> list[Source]:
    """COMPLETE sources of ``scores/`` in a deterministic order (full before pairs, then by id)."""
    out: list[Source] = []
    n_rows, n_cand = len(common.release().queries), len(common.release().candidate_ids)
    for d in sorted(common.SCORES.iterdir()):
        manifest_path = d / "manifest.json"
        if not d.is_dir() or not manifest_path.exists() or d.name in exclude:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (d / "scores.npy").exists():
            mm = np.load(d / "scores.npy", mmap_mode="r")
            if mm.shape != (n_rows, n_cand) or mm.dtype != np.float32:
                flog(f"skipping {d.name}: unexpected shape/dtype {mm.shape} {mm.dtype}")
                continue
            kind, sha = "full", str(manifest.get("scores_sha256") or common.sha256_file(d / "scores.npy"))
        elif (d / "pairs.npz").exists():
            kind, sha = "pairs", str(manifest.get("pairs_sha256") or common.sha256_file(d / "pairs.npz"))
        else:
            continue
        family = family_of(d.name, manifest)
        out.append(Source(d.name, kind, family, group_of(d.name, manifest) if family == "EMB" else d.name, d.name in DERIVED_IDS, sha, manifest,
                          family == "NO-EMB" and is_lsa(d.name, manifest)))
    out.sort(key=lambda s: (s.kind != "full", s.id))
    return out


# ----------------------------------------------------------------------------- helpers on score rows


def sanitize(block: np.ndarray) -> np.ndarray:
    """Replace non-finite entries by (row minimum finite value - 1) so ranks and z-scores stay defined."""
    bad = ~np.isfinite(block)
    if not bad.any():
        return block
    block = block.copy()
    lowest = np.where(bad, np.inf, block).min(axis=1)
    lowest = np.where(np.isfinite(lowest), lowest, 0.0).astype(np.float32)
    block[bad] = np.broadcast_to((lowest - 1.0)[:, None], block.shape)[bad]
    return block


def topk_block(block: np.ndarray, k: int = TOPK) -> np.ndarray:
    """Per row the k best candidates (score descending, index ascending on ties); exact boundary tie handling."""
    n = block.shape[1]
    kth = np.partition(block, n - k, axis=1)[:, n - k]
    out = np.empty((block.shape[0], k), dtype=np.int32)
    for i in range(block.shape[0]):
        row = block[i]
        above = np.flatnonzero(row > kth[i])
        tied = np.flatnonzero(row == kth[i])[: k - len(above)]
        out[i] = np.concatenate([above, tied])
    return out


def parallel_blocks(n: int, block: int, fn, threads: int = THREADS) -> list:
    spans = [(a, min(a + block, n)) for a in range(0, n, block)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        return list(pool.map(lambda ab: fn(*ab), spans))


def top_indices(src: Source) -> np.ndarray:
    """``[rows, TOPK]`` top-candidate indices of a full source (cached by the file's sha256)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"top{TOPK}_{src.id}_{src.sha[:12]}.npy"
    if path.exists():
        return np.load(path)
    mm = common.open_scores(src.id)
    n_rows = mm.shape[0]
    top = np.empty((n_rows, TOPK), dtype=np.int32)

    def work(a: int, b: int) -> None:
        top[a:b] = topk_block(sanitize(np.asarray(mm[a:b], dtype=np.float32)))

    parallel_blocks(n_rows, 256, work)
    np.save(path, top)
    return top


# ----------------------------------------------------------------------------- feature table


@dataclass
class Features:
    pq: np.ndarray            # int32 [n] query row
    pc: np.ndarray            # int32 [n] candidate index
    pmask: np.ndarray         # uint64 [n] bit i set when source i (full sources, in order) has the pair in its top-K
    X: np.ndarray             # float32 [n, ncols]
    cols: list[str]
    row_ptr: np.ndarray       # [rows + 1] pair range per query row
    sources: list[Source]
    full_ids: list[str]       # bit order
    y_spec: np.ndarray        # int8 [n] (dev rows only; 0 elsewhere)
    y_full: np.ndarray
    generic: np.ndarray       # bool [n]
    nspec: np.ndarray         # int [rows] number of specific positives (dev rows only)
    table_key: str = ""       # identifies the global pair table: pair indices in checkpoints are only valid for this table

    def col(self, name: str) -> int:
        return self.cols.index(name)

    def bits(self, ids: Sequence[str]) -> np.uint64:
        out = np.uint64(0)
        for sid in ids:
            if sid in self.full_ids:
                out |= np.uint64(1) << np.uint64(self.full_ids.index(sid))
        return out


def features_key(sources: Sequence[Source]) -> str:
    payload = json.dumps({"v": FEATURE_VERSION, "topk": TOPK, "release": common.sha256_file(common.RELEASE / "manifest.json"),
                          "sources": [(s.id, s.kind, s.sha) for s in sources]}, sort_keys=True)
    return spec.sha256_text(payload)


def source_columns(src: Source) -> list[str]:
    """Full sources: per-query robust z, log (min-)rank, robust z of the best article of the candidate's instrument (raw scores are not stored: memory)."""
    if src.kind == "full":
        return [f"{src.id}:z", f"{src.id}:lr", f"{src.id}:imaxz"]
    return [f"{src.id}:raw", f"{src.id}:z", f"{src.id}:lr", f"{src.id}:present"]


EXTRA_COLUMNS = ["log_len", "instr_prior", "log_qlen"]


def build_pool(sources: Sequence[Source], n_rows: int, n_cand: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    full = [s for s in sources if s.kind == "full"]
    if len(full) > 63:
        raise ValueError("at most 63 full sources fit the pool bit mask")
    keys, flags = [], []
    for i, s in enumerate(full):
        top = top_indices(s)
        keys.append((np.repeat(np.arange(n_rows, dtype=np.int64), TOPK) * n_cand + top.ravel()))
        flags.append(np.full(keys[-1].shape, np.uint64(1) << np.uint64(i), dtype=np.uint64))
    keys, flags = np.concatenate(keys), np.concatenate(flags)
    order = np.argsort(keys, kind="stable")
    keys, flags = keys[order], flags[order]
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
    ukeys, uflags = keys[starts], np.bitwise_or.reduceat(flags, starts)
    return (ukeys // n_cand).astype(np.int32), (ukeys % n_cand).astype(np.int32), uflags, [s.id for s in full]


def instrument_tables(ctx: base.Ctx) -> tuple[np.ndarray, np.ndarray]:
    """(``[groups, N]`` log1p of distinct training components citing any article of the candidate's instrument, instrument ids)."""
    d = ctx.data
    instr = [row["instrument_id"] for row in spec.read_jsonl(Path(d.candidate_binding["articles_path"]))]
    code = {v: i for i, v in enumerate(sorted(set(instr)))}
    cand_instr = np.asarray([code[v] for v in instr])
    table = np.zeros((len(ctx.groups), ctx.N), dtype=np.float32)
    nested = base.Nested(ctx)
    for g in ctx.groups:
        seen: set[tuple[str, int]] = set()
        for qid in nested.fit_query_ids(g.train_folds):
            comp = d.component_by_query[qid]
            for cid in d.references[qid]:
                seen.add((comp, int(cand_instr[d.candidate_index[cid]])))
        count = np.zeros(len(code))
        for _, i in seen:
            count[i] += 1
        table[g.idx] = np.log1p(count)[cand_instr]
    return table, cand_instr


def instrument_layout(cand_instr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(permutation grouping candidates by instrument, segment starts in that order, segment index of each instrument id)."""
    perm = np.argsort(cand_instr, kind="stable")
    ordered = cand_instr[perm]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    segment = np.zeros(int(cand_instr.max()) + 1, dtype=np.int64)
    segment[ordered[starts]] = np.arange(len(starts))
    return perm, starts, segment


def full_features(src: Source, pq: np.ndarray, pc: np.ndarray, row_ptr: np.ndarray, X: np.ndarray, col0: int, n_cand: int,
                  layout: tuple[np.ndarray, np.ndarray, np.ndarray], cand_instr: np.ndarray) -> None:
    mm = common.open_scores(src.id)
    perm, starts, segment = layout

    def work(a: int, b: int) -> None:
        lo, hi = int(row_ptr[a]), int(row_ptr[b])
        if hi == lo:
            return
        block = sanitize(np.asarray(mm[a:b], dtype=np.float32))
        ql, cc = pq[lo:hi] - a, pc[lo:hi]
        raw = block[ql, cc]
        med = np.median(block, axis=1)
        mad = np.median(np.abs(block - med[:, None]), axis=1)
        std = block.std(axis=1)
        scale = (1.4826 * mad).astype(np.float32)
        scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0)).astype(np.float32)
        z = (raw - med[ql].astype(np.float32)) / scale[ql]
        ordered = np.sort(block, axis=1)
        ptr = np.searchsorted(ql, np.arange(b - a + 1))
        lr = np.empty(hi - lo, dtype=np.float32)
        for i in range(b - a):
            s, e = ptr[i], ptr[i + 1]
            if e > s:
                n_le = np.searchsorted(ordered[i], raw[s:e], side="right")
                lr[s:e] = np.log((n_cand - n_le + 1).astype(np.float32))
        imax = np.maximum.reduceat(block[:, perm], starts, axis=1)
        imaxz = (imax[ql, segment[cand_instr[cc]]] - med[ql].astype(np.float32)) / scale[ql]
        X[lo:hi, col0] = z
        X[lo:hi, col0 + 1] = lr
        X[lo:hi, col0 + 2] = imaxz

    parallel_blocks(len(row_ptr) - 1, 128, work)


def pairs_features(src: Source, pq: np.ndarray, pc: np.ndarray, X: np.ndarray, col0: int, n_cand: int) -> None:
    import pandas as pd

    z = np.load(src.folder / "pairs.npz")
    q, c, s = z["qidx"].astype(np.int64), z["cidx"].astype(np.int64), z["score"].astype(np.float32)
    frame = pd.DataFrame({"q": q, "c": c, "s": s}).drop_duplicates(["q", "c"], keep="last")
    grp = frame.groupby("q")["s"]
    frame["med"] = grp.transform("median")
    frame["mad"] = (frame["s"] - frame["med"]).abs().groupby(frame["q"]).transform("median")
    frame["std"] = grp.transform("std").fillna(0.0)
    scale = 1.4826 * frame["mad"]
    frame["scale"] = np.where(scale > 1e-8, scale, np.where(frame["std"] > 1e-8, frame["std"], 1.0))
    frame["z"] = (frame["s"] - frame["med"]) / frame["scale"]
    frame["rank"] = grp.rank(method="min", ascending=False)
    frame["key"] = frame["q"] * n_cand + frame["c"]
    frame = frame.sort_values("key")
    skeys = frame["key"].to_numpy()
    pkeys = pq.astype(np.int64) * n_cand + pc
    pos = np.searchsorted(skeys, pkeys)
    pos_c = np.minimum(pos, len(skeys) - 1)
    present = skeys[pos_c] == pkeys
    for j, name in enumerate(("s", "z", "rank")):
        vals = frame[name].to_numpy(dtype=np.float32)[pos_c]
        if name == "rank":
            vals = np.log(vals)
        X[:, col0 + j] = np.where(present, vals, np.nan)
    X[:, col0 + 3] = present.astype(np.float32)


def build_features(ctx: base.Ctx, sources: list[Source]) -> Features:
    key = features_key(sources)
    FEATDIR.mkdir(parents=True, exist_ok=True)
    meta_path = FEATDIR / "meta.json"
    n_rows, n_cand = len(ctx.data.queries), ctx.N
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")).get("key") == key:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        pq, pc, pmask = (np.load(FEATDIR / f"{n}.npy") for n in ("pq", "pc", "pmask"))
        X = np.load(FEATDIR / "X.npy")
        flog(f"features loaded from cache: {len(pq):,} pairs x {X.shape[1]} columns")
        F = finish_features(ctx, sources, pq, pc, pmask, X, meta["columns"], meta["full_ids"])
        F.table_key = key
        return F
    t0 = time.perf_counter()
    pq, pc, pmask, full_ids = build_pool(sources, n_rows, n_cand)
    row_ptr = np.searchsorted(pq, np.arange(n_rows + 1))
    cols = [c for s in sources for c in source_columns(s)] + EXTRA_COLUMNS
    X = np.full((len(pq), len(cols)), np.nan, dtype=np.float32)
    table, cand_instr = instrument_tables(ctx)
    layout = instrument_layout(cand_instr)
    flog(f"pool built: {len(pq):,} pairs ({len(pq) / n_rows:.0f} per query), {len(cols)} columns ({time.perf_counter() - t0:.0f}s)")
    col0 = 0
    for s in sources:
        t = time.perf_counter()
        if s.kind == "full":
            full_features(s, pq, pc, row_ptr, X, col0, n_cand, layout, cand_instr)
        else:
            pairs_features(s, pq, pc, X, col0, n_cand)
        col0 += len(source_columns(s))
        flog(f"features for {s.id} ({s.kind}) in {time.perf_counter() - t:.0f}s")
    texts = ctx.data.candidate_texts
    log_len = np.log1p(np.asarray([len(t) for t in texts], dtype=np.float32))
    gid = np.where(ctx.fold >= 0, ctx.fold, len(ctx.groups) - 1)
    log_qlen = np.log1p(np.asarray([len(t) for t in ctx.texts], dtype=np.float32))
    X[:, col0] = log_len[pc]
    X[:, col0 + 1] = table[gid[pq], pc]
    X[:, col0 + 2] = log_qlen[pq]
    np.save(FEATDIR / "pq.npy", pq)
    np.save(FEATDIR / "pc.npy", pc)
    np.save(FEATDIR / "pmask.npy", pmask)
    np.save(FEATDIR / "X.npy", X)
    meta_path.write_text(json.dumps({"key": key, "columns": cols, "full_ids": full_ids,
                                     "sources": [{"id": s.id, "kind": s.kind, "sha256": s.sha, "family": s.family, "group": s.group} for s in sources],
                                     "n_pairs": int(len(pq)), "topk": TOPK, "built_utc": base.utc()}, indent=1), encoding="utf-8")
    flog(f"features built in {time.perf_counter() - t0:.0f}s")
    F = finish_features(ctx, sources, pq, pc, pmask, X, cols, full_ids)
    F.table_key = key
    return F


def finish_features(ctx: base.Ctx, sources, pq, pc, pmask, X, cols, full_ids) -> Features:
    """Attach dev-only labels (lockbox rows get none), generic flags and the row pointer."""
    d = ctx.data
    n_rows = len(d.queries)
    row_ptr = np.searchsorted(pq, np.arange(n_rows + 1))
    n_cand = ctx.N
    pkey = pq.astype(np.int64) * n_cand + pc

    def labels(positive_lists) -> np.ndarray:
        keys = np.sort(np.fromiter((int(r) * n_cand + int(c) for r in ctx.dev for c in positive_lists[int(r)]), dtype=np.int64))
        return np.isin(pkey, keys, assume_unique=False).astype(np.int8)

    y_spec, y_full = labels(d.specific_idx), labels(d.positive_idx)
    generic = np.zeros(n_cand, dtype=bool)
    generic[d.generic_idx] = True
    nspec = np.zeros(n_rows, dtype=np.int64)
    for r in ctx.dev:
        nspec[int(r)] = len(d.specific_idx[int(r)])
    return Features(pq, pc, pmask, X, list(cols), row_ptr, sources, list(full_ids), y_spec, y_full, generic[pc], nspec)


# ----------------------------------------------------------------------------- variants, padding, nDCG


@dataclass
class Variant:
    name: str
    pool_ids: list[str]        # full sources whose top-K define the pool
    feature_ids: list[str]     # sources whose columns are features
    idx: np.ndarray = field(default=None, repr=False)      # pair indices (sorted by row)
    ptr: np.ndarray = field(default=None, repr=False)      # row pointer into idx
    hgb_cols: list[int] = field(default_factory=list)
    lin_cols: list[int] = field(default_factory=list)
    anchor: int = 0
    lin_names: list[str] = field(default_factory=list)


def make_variant(F: Features, name: str, pool_ids: Sequence[str], feature_ids: Sequence[str]) -> Variant:
    bits = F.bits(pool_ids)
    idx = np.flatnonzero((F.pmask & bits) != 0)
    rows = F.pq[idx]
    ptr = np.searchsorted(rows, np.arange(len(F.row_ptr)))
    by_id = {s.id: s for s in F.sources}
    hgb, lin, lin_names = [], [], []
    for sid in feature_ids:
        for cname in source_columns(by_id[sid]):
            hgb.append(F.col(cname))
        if not by_id[sid].derived:
            lin.append(F.col(f"{sid}:z"))
            lin_names.append(sid)
    hgb += [F.col(c) for c in EXTRA_COLUMNS]
    v = Variant(name, list(pool_ids), list(feature_ids), idx, ptr, hgb, lin, lin_names.index("L_star") if "L_star" in lin_names else 0, lin_names)
    return v


def row_pairs(v: Variant, rows: np.ndarray, drop_generic: bool, F: Features) -> np.ndarray:
    """Pair indices of the variant's pool for ``rows`` (sorted by row), optionally without generic candidates."""
    parts = [v.idx[v.ptr[r]:v.ptr[r + 1]] for r in np.asarray(rows)]
    out = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    return out[~F.generic[out]] if drop_generic else out


@dataclass
class Padded:
    rows: np.ndarray          # row id per segment
    seg: np.ndarray           # segment of each pair
    pos: np.ndarray           # position of each pair inside its segment
    Pmax: int
    rel: np.ndarray           # bool [S, Pmax]
    valid: np.ndarray
    nrel: np.ndarray          # specific positives per segment (all, not only pooled)
    Z: np.ndarray | None = None


def pad_pairs(F: Features, idx: np.ndarray, cols: Sequence[int] | None = None) -> Padded:
    r = F.pq[idx]
    rows, start, counts = np.unique(r, return_index=True, return_counts=True)
    seg = np.repeat(np.arange(len(rows)), counts)
    pos = np.arange(len(idx)) - np.repeat(start, counts)
    pmax = int(counts.max()) if len(counts) else 0
    rel = np.zeros((len(rows), pmax), dtype=bool)
    valid = np.zeros((len(rows), pmax), dtype=bool)
    rel[seg, pos] = F.y_spec[idx] > 0
    valid[seg, pos] = True
    pad = Padded(rows, seg, pos, pmax, rel, valid, F.nspec[rows])
    if cols is not None:
        Z = np.zeros((len(rows), pmax, len(cols)), dtype=np.float32)
        Z[seg, pos] = np.nan_to_num(F.X[idx[:, None], np.asarray(cols)[None, :]], nan=0.0)
        pad.Z = Z
    return pad


def ndcg_padded(S: np.ndarray, pad: Padded) -> np.ndarray:
    """nDCG@10 of each segment given pooled scores ``S [segments, Pmax]`` (specific positives, IDCG over all of them)."""
    S = np.where(pad.valid, S, -np.inf)
    k = min(10, pad.Pmax)
    top = np.argpartition(-S, k - 1, axis=1)[:, :k]
    ts = np.take_along_axis(S, top, 1)
    top = np.take_along_axis(top, np.argsort(-ts, axis=1, kind="stable"), 1)
    hit = np.take_along_axis(pad.rel, top, 1) & np.take_along_axis(pad.valid, top, 1)
    dcg = (hit * LOG2[:k]).sum(axis=1)
    ideal = IDCG[np.minimum(pad.nrel, 10)]
    return np.divide(dcg, ideal, out=np.zeros_like(dcg), where=ideal > 0)


def macro_rows(ctx: base.Ctx, rows: np.ndarray, values: np.ndarray) -> float:
    """Component-macro mean of per-row values."""
    codes = ctx.comp[rows]
    sums = np.bincount(codes, weights=values, minlength=ctx.ncomp)
    cnt = np.bincount(codes, minlength=ctx.ncomp)
    m = cnt > 0
    return float((sums[m] / cnt[m]).mean())


def spec_rows_of(ctx: base.Ctx, folds: Sequence[int]) -> np.ndarray:
    """Dev P_spec rows of the given folds."""
    rows = common.population_rows("P_spec", common.dev_rows())
    return rows[np.isin(ctx.fold[rows], list(folds))]


def eval_pairs(ctx: base.Ctx, F: Features, idx: np.ndarray, pair_scores: np.ndarray, rows: np.ndarray) -> float:
    """Component-macro P_spec nDCG@10 over ``rows``; rows without pooled pairs count 0. ``idx``/scores aligned."""
    pad = pad_pairs(F, idx)
    S = np.full((len(pad.rows), pad.Pmax), -np.inf, dtype=np.float32)
    S[pad.seg, pad.pos] = pair_scores
    nd = dict(zip(pad.rows.tolist(), ndcg_padded(S, pad).tolist()))
    return macro_rows(ctx, rows, np.asarray([nd.get(int(r), 0.0) for r in rows]))


# ----------------------------------------------------------------------------- linear model


def coordinate_ascent(ctx: base.Ctx, pad: Padded, rows_all: np.ndarray, anchor: int, grid: Sequence[float] = LIN_GRID,
                      sweeps: int = 3, refine_sweeps: int = 2) -> tuple[np.ndarray, float, list[float]]:
    """Non-negative weights over (standardised) robust-z features maximising component-macro nDCG@10; anchor weight fixed at 1.

    Coarse sweeps try every grid value per coordinate; refinement sweeps try multiplicative nudges of the non-zero weights.
    A move is accepted only on strict improvement, so ties keep the sparser / earlier solution.
    """
    Z = pad.Z
    f_count = Z.shape[2]
    w = np.zeros(f_count, dtype=np.float32)
    w[anchor] = 1.0
    S = Z[:, :, anchor].copy()
    row_pos = {int(r): i for i, r in enumerate(pad.rows)}
    seg_of = np.asarray([row_pos.get(int(r), -1) for r in rows_all])

    def objective(scores: np.ndarray) -> float:
        nd = ndcg_padded(scores, pad)
        values = np.where(seg_of >= 0, nd[np.maximum(seg_of, 0)], 0.0)
        return macro_rows(ctx, rows_all, values)

    cur = objective(S)
    history = [cur]

    def sweep(candidates) -> bool:
        nonlocal S, cur
        improved = False
        for f in range(f_count):
            if f == anchor:
                continue
            best_v, best_obj = float(w[f]), cur
            for v in candidates(float(w[f])):
                if v == w[f]:
                    continue
                obj = objective(S + np.float32(v - w[f]) * Z[:, :, f])
                if obj > best_obj + 1e-9:
                    best_v, best_obj = float(v), obj
            if best_v != w[f]:
                S += np.float32(best_v - w[f]) * Z[:, :, f]
                w[f], cur, improved = best_v, best_obj, True
        history.append(cur)
        return improved

    for _ in range(sweeps):
        if not sweep(lambda current: grid):
            break
    for _ in range(refine_sweeps):
        if not sweep(lambda current: [current * m for m in LIN_REFINE] if current > 0 else []):
            break
    return w, cur, history


# ----------------------------------------------------------------------------- HGB


def hgb_training_pairs(ctx: base.Ctx, F: Features, v: Variant, rows: np.ndarray, rng: np.random.Generator,
                       view: str = "spec") -> tuple[np.ndarray, np.ndarray]:
    """Positives plus up to N_NEG sampled pool negatives per row (rows with a pooled positive only).

    ``view='spec'``: specific positives, generic candidates removed (the P_spec objective); ``'full'``: all positives, generics kept.
    """
    idx = row_pairs(v, rows, view == "spec", F)
    y = (F.y_spec if view == "spec" else F.y_full)[idx]
    r = F.pq[idx]
    has_pos = np.isin(r, np.unique(r[y > 0]))
    idx, y, r = idx[has_pos], y[has_pos], r[has_pos]
    neg = np.flatnonzero(y == 0)
    keys = rng.random(len(neg))
    order = np.lexsort((keys, r[neg]))
    neg, rn = neg[order], r[neg][order]
    first = np.searchsorted(rn, rn, side="left")
    keep_neg = neg[(np.arange(len(neg)) - first) < N_NEG]
    keep = np.sort(np.concatenate([np.flatnonzero(y > 0), keep_neg]))
    return idx[keep], y[keep]


def fit_hgb(F: Features, v: Variant, idx: np.ndarray, y: np.ndarray, cfg: dict[str, Any], seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"], random_state=seed, **HGB_FIXED)
    model.fit(F.X[idx[:, None], np.asarray(v.hgb_cols)[None, :]], y)
    return model


def predict_hgb(F: Features, v: Variant, model, idx: np.ndarray, chunk: int = 500_000) -> np.ndarray:
    cols = np.asarray(v.hgb_cols)[None, :]
    return np.concatenate([model.decision_function(F.X[idx[s:s + chunk, None], cols]) for s in range(0, len(idx), chunk)]).astype(np.float32) \
        if len(idx) else np.empty(0, dtype=np.float32)


@dataclass
class GroupModel:
    kind: str                         # linear | hgb
    config: dict[str, Any]
    evidence: dict[str, Any]
    pair_idx: np.ndarray = field(default=None, repr=False)
    pair_scores: np.ndarray = field(default=None, repr=False)


def train_rows_of(ctx: base.Ctx, g: base.Group, folds: Sequence[int] | None = None) -> np.ndarray:
    use = g.train_folds if folds is None else tuple(folds)
    return ctx.dev[np.isin(ctx.fold[ctx.dev], use)]


def fit_group_linear(ctx: base.Ctx, F: Features, v: Variant, g: base.Group) -> GroupModel:
    """Weights on the raw per-query robust-z features (anchor L_star = 1).

    No standardisation or clipping: a handful of queries have a near-zero MAD, so their z-scores reach 1e5-1e6 and any
    spread-based rescaling would be dominated by them; the weight grid is log-spaced instead.
    """
    rows = spec_rows_of(ctx, g.train_folds)
    idx = row_pairs(v, rows, True, F)
    cols = np.asarray(v.lin_cols)[None, :]
    pad = pad_pairs(F, idx, v.lin_cols)
    w, obj, hist = coordinate_ascent(ctx, pad, rows, v.anchor)
    del pad
    idx_out = row_pairs(v, g.rows, False, F)
    Z = np.nan_to_num(F.X[idx_out[:, None], cols], nan=0.0)
    return GroupModel("linear", {"weights_z": dict(zip(v.lin_names, map(float, w)))},
                      {"train_objective": obj, "history": hist, "train_rows": int(len(rows))}, idx_out, (Z @ w).astype(np.float32))


def fit_group_hgb(ctx: base.Ctx, F: Features, v: Variant, g: base.Group, fixed_cfg: dict[str, Any] | None = None,
                  view: str = "spec") -> GroupModel:
    if view != "spec" and fixed_cfg is None:
        raise ValueError("the full-label variant reuses hyper-parameters selected on the specific-label objective")
    seed = SEED + 1000 * g.idx
    evidence: dict[str, Any] = {}
    if fixed_cfg is None:
        results = []
        for ci, cfg in enumerate(HGB_GRID):
            per_fold = []
            all_vals: list[np.ndarray] = []
            all_rows: list[np.ndarray] = []
            for inner in g.train_folds[:INNER_FOLDS]:
                rng = np.random.default_rng(seed + 17 * inner)
                tr_rows = train_rows_of(ctx, g, [f for f in g.train_folds if f != inner])
                tr_idx, tr_y = hgb_training_pairs(ctx, F, v, tr_rows, rng)
                model = fit_hgb(F, v, tr_idx, tr_y, cfg, seed)
                va_rows = spec_rows_of(ctx, [inner])
                va_idx = row_pairs(v, va_rows, True, F)
                pad = pad_pairs(F, va_idx)
                S = np.full((len(pad.rows), pad.Pmax), -np.inf, dtype=np.float32)
                S[pad.seg, pad.pos] = predict_hgb(F, v, model, va_idx)
                nd = dict(zip(pad.rows.tolist(), ndcg_padded(S, pad).tolist()))
                all_rows.append(va_rows)
                all_vals.append(np.asarray([nd.get(int(r), 0.0) for r in va_rows]))
                per_fold.append({"inner_fold": inner, "n_train_pairs": int(len(tr_idx))})
            rows_cat, vals_cat = np.concatenate(all_rows), np.concatenate(all_vals)
            results.append({"config": cfg, "objective": macro_rows(ctx, rows_cat, vals_cat), "folds": per_fold})
        best = base.argbest([r["objective"] for r in results])
        cfg = HGB_GRID[best]
        evidence = {"grid": results, "chosen": best}
    else:
        cfg = fixed_cfg
        evidence = {"fixed": cfg}
    rng = np.random.default_rng(seed + 999)
    tr_idx, tr_y = hgb_training_pairs(ctx, F, v, train_rows_of(ctx, g), rng, view)
    model = fit_hgb(F, v, tr_idx, tr_y, cfg, seed)
    idx_out = row_pairs(v, g.rows, False, F)
    return GroupModel("hgb", dict(cfg), {**evidence, "final_train_pairs": int(len(tr_idx)), "label_view": view}, idx_out,
                      predict_hgb(F, v, model, idx_out))


# ----------------------------------------------------------------------------- pipeline files


def pipeline_path(name: str) -> Path:
    return PIPES / name / "scores.npy"


def create_pipeline(name: str) -> np.ndarray:
    path = pipeline_path(name)
    if path.parent.exists():
        shutil.rmtree(path.parent)
    path.parent.mkdir(parents=True)
    data = common.release()
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(data.queries), len(data.candidate_ids)))
    matrix[:] = np.nan
    return matrix


def finish_pipeline(name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    path = pipeline_path(name)
    matrix = np.load(path, mmap_mode="r")
    missing = [int(i) for i in range(matrix.shape[0]) if not np.isfinite(matrix[i]).any()]
    if missing:
        raise ValueError(f"{name}: {len(missing)} rows never written")
    record = {**manifest, "pipeline": name, "shape": list(matrix.shape), "dtype": "float32", "scores_sha256": common.sha256_file(path),
              "split_sha256": common.sha256_file(common.SPLIT_PATH),
              "release_manifest_sha256": common.sha256_file(common.RELEASE / "manifest.json"),
              "code_sha256": {str(p): common.sha256_file(p) for p in CODE_FILES}, "written_utc": base.utc()}
    (path.parent / "manifest.json").write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    return record


def write_pooled(ctx: base.Ctx, F: Features, matrix: np.ndarray, rows: np.ndarray, idx: np.ndarray, pair_scores: np.ndarray,
                 lstar: np.ndarray) -> None:
    """Pooled candidates keep their model scores; every other candidate sits below the pool in L_star order."""
    r = F.pq[idx]
    ptr = np.searchsorted(r, np.arange(len(F.row_ptr)))
    for row in rows:
        row = int(row)
        sl = slice(ptr[row], ptr[row + 1])
        cand, sc = F.pc[idx[sl]], pair_scores[sl]
        lrow = lstar[row]
        order = np.argsort(-lrow, kind="stable")
        rank = np.empty(len(lrow), dtype=np.float64)
        rank[order] = np.arange(1, len(lrow) + 1)
        floor = float(sc.min()) if len(sc) else 0.0
        out = ((floor - 1.0) - rank / (len(lrow) + 1.0)).astype(np.float32)
        out[cand] = sc
        matrix[row] = out


def unique_sorted_rows(*arrays) -> np.ndarray:
    return np.unique(np.concatenate(arrays))


# ----------------------------------------------------------------------------- hand pipelines with their own inner CV


def code_version() -> str:
    return common.sha256_file(Path(__file__))[:16]


def stage_hand(ctx: base.Ctx, force: bool = False) -> None:
    """``K_lex_prior`` (NO-EMB) and ``K_lex_H_bge_prior`` (EMB): K_lex plus dense and/or prior, weights by inner CV."""
    names = ("K_lex_prior", "K_lex_H_bge_prior")
    dep = {m: json.loads((common.score_path(m).parent / "manifest.json").read_text(encoding="utf-8"))["scores_sha256"]
           for m in ("L_star", "K_lex", "prior", "dense_bge_maxp")}

    def up_to_date(n: str) -> bool:
        path = pipeline_path(n).parent / "manifest.json"
        if not path.exists():
            return False
        rec = json.loads(path.read_text(encoding="utf-8"))
        return rec.get("depends_on_sha256") == dep and rec.get("hand_version") == HAND_VERSION

    if all(up_to_date(n) for n in names) and not force:
        flog("hand pipelines up to date")
        return
    t0 = time.perf_counter()
    lex = json.loads((base.SELECTION_DIR / "lexical.json").read_text(encoding="utf-8"))
    nest = json.loads((base.SELECTION_DIR / "nested.json").read_text(encoding="utf-8"))
    lsel = {g.role: base.LSel.from_json(lex[g.role]["selected"]) for g in ctx.groups}
    nested, store = base.Nested(ctx), base.UStore(ctx)
    ng, nd, npw = len(ctx.groups), len(HAND_W_DENSE), len(W_PRIOR2)
    t_kp = np.full((ng, npw, ctx.nU, 2, 4), np.nan)
    t_kh = np.full((ng, nd * npw, ctx.nU, 2, 4), np.nan)
    for g in ctx.groups:
        kc = nest[g.role]["K_lex"]
        _, zl_all = store.lstar(lsel[g.role])
        _, zd_all = store.dense("dense_bge_maxp")
        for inner in g.train_folds:
            fit = tuple(f for f in g.train_folds if f != inner)
            idx = np.flatnonzero(ctx.foldU == inner)
            rows = ctx.U[idx]
            votes = nested.votes(rows, fit, [kc["k"]])[kc["k"]]
            zk = fusion.robust_z(fusion.linear_fusion([zl_all[idx], fusion.robust_z(votes)], base.pair(kc["w_knn"]), normalize="none"))
            zp = fusion.robust_z(nested.prior(fit))
            zd = zd_all[idx]
            for wi, wp in enumerate(W_PRIOR2):
                t_kp[g.idx, wi, idx] = ctx.query_metrics(zk + np.float32(wp) * zp if wp else zk, rows)
            for di, wd in enumerate(HAND_W_DENSE):
                h = fusion.linear_fusion([zk, zd], base.pair(wd), normalize="none")
                for wi, wp in enumerate(W_PRIOR2):
                    t_kh[g.idx, di * npw + wi, idx] = ctx.query_metrics(h + np.float32(wp) * zp if wp else h, rows)
        flog(f"hand inner CV {g.role} done ({time.perf_counter() - t0:.0f}s)")
    selection: dict[str, Any] = {}
    base.reset_search_log("hand")
    for g in ctx.groups:
        train = ctx.train_mask_u(g)
        kp = [ctx.objective(t_kp[g.idx, i], train) for i in range(npw)]
        kh = [ctx.objective(t_kh[g.idx, i], train) for i in range(nd * npw)]
        i, j = base.argbest(kp), base.argbest(kh)
        selection[g.role] = {"K_lex_prior": {"w_prior": W_PRIOR2[i], "grid": [{"w_prior": w, "objective": v} for w, v in zip(W_PRIOR2, kp)]},
                             "K_lex_H_bge_prior": {"w_dense": HAND_W_DENSE[j // npw], "w_prior": W_PRIOR2[j % npw],
                                                   "grid": [{"w_dense": HAND_W_DENSE[c // npw], "w_prior": W_PRIOR2[c % npw], "objective": v}
                                                            for c, v in enumerate(kh)]}}
        for name in names:
            for cfg in selection[g.role][name]["grid"]:
                base.search_log({"stage": "hand", "method": name, "group": g.role,
                                 "config": {k: v for k, v in cfg.items() if k != "objective"}, "inner_P_spec_ndcg10": cfg["objective"]})
        flog(f"{g.role}: K_lex_prior w_prior={W_PRIOR2[i]}; K_lex_H_bge_prior w_dense={HAND_W_DENSE[j // npw]} w_prior={W_PRIOR2[j % npw]}")
    (base.SELECTION_DIR / "hand.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    mats = {n: create_pipeline(n) for n in names}
    kfile, dfile = common.open_scores("K_lex"), common.open_scores("dense_bge_maxp")
    for g in ctx.groups:
        zp = fusion.robust_z(nested.prior(g.train_folds))
        cfg = selection[g.role]
        for sl in ctx.blocks(len(g.rows), 512):
            rows = g.rows[sl]
            zk = fusion.robust_z(np.asarray(kfile[rows]))
            zd = fusion.robust_z(np.asarray(dfile[rows]))
            wp = cfg["K_lex_prior"]["w_prior"]
            mats["K_lex_prior"][rows] = zk + np.float32(wp) * zp if wp else zk
            h = fusion.linear_fusion([zk, zd], base.pair(cfg["K_lex_H_bge_prior"]["w_dense"]), normalize="none")
            wp2 = cfg["K_lex_H_bge_prior"]["w_prior"]
            mats["K_lex_H_bge_prior"][rows] = h + np.float32(wp2) * zp if wp2 else h
    wall = time.perf_counter() - t0
    for n in names:
        mats[n].flush()
    del mats
    for n, fam, text in (("K_lex_prior", "NO-EMB", "robust-z K_lex + w_prior * robust-z prior"),
                         ("K_lex_H_bge_prior", "EMB", "robust-z (1 - w_dense) K_lex + w_dense dense_bge_maxp, plus w_prior * robust-z prior")):
        finish_pipeline(n, {"method": n, "family": fam, "kind": "hand fusion", "description": text, "depends_on_sha256": dep,
                            "hand_version": HAND_VERSION, "code_version": code_version(), "configuration": {g: selection[g][n] for g in selection},
                            "training_data": "labels of the group's training dev rows only (inner CV over its training folds); lockbox group: all dev",
                            "selection_evidence": str(base.SELECTION_DIR / "hand.json"), "seeds": None, "wall_seconds": wall, "device": "cpu"})
    flog(f"hand pipelines complete in {wall:.0f}s")


# ----------------------------------------------------------------------------- learned fusion


def variant_key(F: Features, v: Variant, kind: str) -> str:
    by_id = {s.id: s for s in F.sources}
    payload = {"name": v.name, "pool": [(i, by_id[i].sha) for i in v.pool_ids], "features": [(i, by_id[i].sha) for i in v.feature_ids],
               "kind": kind, "model_version": LINEAR_VERSION if kind == "linear" else HGB_VERSION, "topk": TOPK,
               "feature_version": FEATURE_VERSION}
    return spec.sha256_text(json.dumps(payload, sort_keys=True))


def pool_stats(ctx: base.Ctx, F: Features, v: Variant) -> dict[str, float]:
    rows = common.population_rows("P_spec", common.dev_rows())
    idx = row_pairs(v, rows, True, F)
    total = float(sum(len(ctx.data.specific_idx[int(r)]) for r in rows))
    hit_rows = np.unique(F.pq[idx][F.y_spec[idx] > 0])
    return {"mean_pool_size_all_rows": float(len(v.idx) / len(F.row_ptr[:-1])),
            "mean_pool_size_dev_P_spec": float(len(idx) / len(rows)),
            "specific_positive_recall_in_pool_dev_P_spec": float(F.y_spec[idx].sum() / total),
            "share_P_spec_queries_with_pooled_positive": float(len(hit_rows) / len(rows))}


def group_cache_file(key: str, role: str, view: str, fixed: dict | None) -> Path:
    digest = spec.sha256_text(json.dumps({"key": key, "role": role, "view": view, "fixed": fixed}, sort_keys=True, default=str))[:24]
    return CACHE / f"group_{digest}.npz"


def load_group(path: Path) -> GroupModel | None:
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return GroupModel(meta["kind"], meta["config"], meta["evidence"], z["pair_idx"], z["pair_scores"])


def save_group(path: Path, model: GroupModel) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, pair_idx=model.pair_idx, pair_scores=model.pair_scores,
             meta=np.array(json.dumps({"kind": model.kind, "config": model.config, "evidence": model.evidence}, default=float)))
    tmp.replace(path)


def fit_variant(ctx: base.Ctx, F: Features, v: Variant, kind: str, groups: Sequence[base.Group],
                fixed: dict[str, dict] | None = None, view: str = "spec") -> dict[str, GroupModel]:
    """Per-group models, checkpointed per group (a restart resumes). Linear fits run in a few threads; HGB uses all cores itself."""
    key = variant_key(F, v, kind) + "|" + F.table_key
    models: dict[str, GroupModel] = {}
    todo = []
    for g in groups:
        cached = load_group(group_cache_file(key, g.role, view, None if fixed is None else fixed.get(g.role)))
        if cached is not None:
            models[g.role] = cached
            flog(f"{v.name}/{kind} {g.role}: loaded from checkpoint")
        else:
            todo.append(g)

    def fit(g: base.Group) -> GroupModel:
        if kind == "linear":
            return fit_group_linear(ctx, F, v, g)
        return fit_group_hgb(ctx, F, v, g, None if fixed is None else fixed[g.role], view)

    def done(g: base.Group, model: GroupModel, seconds: float) -> None:
        models[g.role] = model
        save_group(group_cache_file(key, g.role, view, None if fixed is None else fixed.get(g.role)), model)
        flog(f"{v.name}/{kind} {g.role}: {model.config} ({seconds:.0f}s)")

    if kind == "linear" and todo:
        def timed(g: base.Group) -> tuple[base.Group, GroupModel, float]:
            t = time.perf_counter()
            return g, fit(g), time.perf_counter() - t

        with ThreadPoolExecutor(max_workers=LINEAR_THREADS) as pool:
            for g, model, seconds in pool.map(timed, todo):
                done(g, model, seconds)
    else:
        for g in todo:
            t = time.perf_counter()
            done(g, fit(g), time.perf_counter() - t)
    return {g.role: models[g.role] for g in groups}


VARIANT_FAMILY = {"noemb": "NO-EMB", "noemb_nolsa": "NO-EMB-noLSA", "emb": "EMB"}


def make_variants(F: Features) -> dict[str, Variant]:
    """Pools and features per family; ``noemb_nolsa`` (sensitivity: NO-EMB without any LSA source) exists only when LSA sources do."""
    sources = F.sources
    out = {"noemb": make_variant(F, "noemb", [s.id for s in sources if s.family == "NO-EMB" and s.kind == "full"],
                                 [s.id for s in sources if s.family == "NO-EMB"])}
    if any(s.lsa for s in sources):
        out["noemb_nolsa"] = make_variant(F, "noemb_nolsa", [s.id for s in sources if s.family == "NO-EMB" and s.kind == "full" and not s.lsa],
                                          [s.id for s in sources if s.family == "NO-EMB" and not s.lsa])
    out["emb"] = make_variant(F, "emb", [s.id for s in sources if s.kind == "full"], [s.id for s in sources])
    return out


def stage_ltr(ctx: base.Ctx, F: Features, force: bool = False, skip_hgb: bool = False) -> dict[str, dict]:
    sources = F.sources
    variants = make_variants(F)
    lstar = np.asarray(common.open_scores("L_star"))
    info: dict[str, dict] = {}
    for fam, v in variants.items():
        stats = pool_stats(ctx, F, v)
        flog(f"pool {fam}: {stats}")
        for kind in ("linear", "hgb", "hgb_full"):
            name = f"LTR_{fam}_{kind}"
            if skip_hgb and kind != "linear":
                continue
            key = variant_key(F, v, kind)
            mpath = pipeline_path(name).parent / "manifest.json"
            if mpath.exists() and not force and json.loads(mpath.read_text(encoding="utf-8")).get("variant_key") == key:
                flog(f"{name} up to date")
                info[name] = json.loads(mpath.read_text(encoding="utf-8"))
                continue
            t0 = time.perf_counter()
            if kind == "hgb_full":
                spec_cfg = info[f"LTR_{fam}_hgb"]["configuration"]
                models = fit_variant(ctx, F, v, "hgb", ctx.groups, {role: spec_cfg[role]["config"] for role in spec_cfg}, "full")
            else:
                models = fit_variant(ctx, F, v, kind, ctx.groups)
            matrix = create_pipeline(name)
            for g in ctx.groups:
                m = models[g.role]
                write_pooled(ctx, F, matrix, g.rows, m.pair_idx, m.pair_scores, lstar)
            matrix.flush()
            del matrix
            family = "EMB" if fam == "emb" else "NO-EMB"
            label = ("all positives (generic articles kept); hyper-parameters reused from the specific-label model" if kind == "hgb_full"
                     else "specific positives; generic candidates removed from training pools (P_spec objective)")
            manifest = {"method": name, "family": family, "variant": VARIANT_FAMILY[fam], "kind": f"learned fusion ({kind})", "variant_key": key,
                        "pool": {"top_k_per_source": TOPK, "sources": v.pool_ids, **stats},
                        "feature_sources": v.feature_ids, "linear_features": v.lin_names, "n_hgb_features": len(v.hgb_cols),
                        "source_sha256": {s.id: s.sha for s in sources if s.id in v.feature_ids},
                        "configuration": {g: {"config": models[g].config, "evidence": models[g].evidence} for g in models},
                        "training_data": "labels of the group's training dev rows only (their out-of-fold features); lockbox group: all dev rows",
                        "label": label,
                        "stacking_caveat": "training rows' out-of-fold features were produced by models that saw the group's held-out fold "
                                           "(indirect, second-order); the lockbox group is unaffected",
                        "seeds": {"hgb_and_sampling": SEED}, "n_negatives_per_query": N_NEG if kind != "linear" else None,
                        "hgb_fixed": HGB_FIXED if kind != "linear" else None, "hgb_grid": HGB_GRID if kind == "hgb" else None,
                        "wall_seconds": time.perf_counter() - t0, "device": "cpu"}
            info[name] = finish_pipeline(name, manifest)
            flog(f"{name} complete in {time.perf_counter() - t0:.0f}s")
    return info


# ----------------------------------------------------------------------------- K_lex stacked hand fusions (partners discovered automatically)

PARTNER_PATTERN = re.compile(r"^(H_|K_(?!lex)|dense_|knn_fused|knn_.*fused)")


def partner_ids(sources: Sequence[Source]) -> list[str]:
    """EMB full sources worth fusing with K_lex one at a time: hybrids (H_*), dense scores and fused-key memories (no first-chunk twins)."""
    return [s.id for s in sources if s.kind == "full" and s.family == "EMB" and PARTNER_PATTERN.search(s.id) and "firstp" not in s.id]


def make_hand_variant(F: Features, name: str, pool_variant: Variant, lin_ids: Sequence[str], anchor_id: str = "K_lex") -> Variant:
    v = Variant(name, list(pool_variant.pool_ids), list(lin_ids), pool_variant.idx, pool_variant.ptr)
    v.lin_cols = [F.col(f"{i}:z") for i in lin_ids]
    v.lin_names = list(lin_ids)
    v.anchor = list(lin_ids).index(anchor_id)
    v.hgb_cols = []
    return v


def hand_specs(F: Features) -> list[tuple[str, str, Variant, list[str]]]:
    variants = make_variants(F)
    sources = F.sources
    specs: list[tuple[str, str, Variant, list[str]]] = []
    for p in partner_ids(sources):
        specs.append((f"KP_{p}", "EMB", variants["emb"], ["K_lex", p, "prior"]))
    emb_full = [s.id for s in sources if s.family == "EMB" and s.kind == "full"]
    specs.append(("KP_emb_all", "EMB", variants["emb"], ["K_lex"] + emb_full + ["prior"]))
    noemb_full = [s.id for s in sources if s.family == "NO-EMB" and s.kind == "full" and not s.derived and s.id != "prior"]
    specs.append(("KP_noemb_all", "NO-EMB", variants["noemb"], ["K_lex"] + noemb_full + ["prior"]))
    if "noemb_nolsa" in variants:
        specs.append(("KP_noemb_nolsa_all", "NO-EMB-noLSA", variants["noemb_nolsa"], ["K_lex"] + [i for i in noemb_full if not next(
            s for s in sources if s.id == i).lsa] + ["prior"]))
    return specs


def stage_partner_hand(ctx: base.Ctx, F: Features, force: bool = False) -> dict[str, dict]:
    """Restricted linear fusions with K_lex as anchor: K_lex + one EMB partner + prior, and K_lex + all sources of a family.

    Weights come from coordinate ascent on the group's training rows (their out-of-fold features; same stacking caveat as the
    learned models). Pool: the family's pool. The NO-EMB analogue (K_lex + every NO-EMB source) keeps the comparison symmetric.
    """
    lstar = np.asarray(common.open_scores("L_star"))
    info: dict[str, dict] = {}
    by_id = {s.id: s for s in F.sources}
    for name, fam, pool_v, lin_ids in hand_specs(F):
        hv = make_hand_variant(F, name, pool_v, lin_ids)
        key = variant_key(F, hv, "linear")
        mpath = pipeline_path(name).parent / "manifest.json"
        if mpath.exists() and not force and json.loads(mpath.read_text(encoding="utf-8")).get("variant_key") == key:
            info[name] = json.loads(mpath.read_text(encoding="utf-8"))
            continue
        t0 = time.perf_counter()
        models = fit_variant(ctx, F, hv, "linear", ctx.groups)
        matrix = create_pipeline(name)
        for g in ctx.groups:
            m = models[g.role]
            write_pooled(ctx, F, matrix, g.rows, m.pair_idx, m.pair_scores, lstar)
        matrix.flush()
        del matrix
        info[name] = finish_pipeline(name, {
            "method": name, "family": "EMB" if fam == "EMB" else "NO-EMB", "variant": fam,
            "kind": "stacked hand fusion (K_lex anchored, restricted linear)", "variant_key": key,
            "pool": {"top_k_per_source": TOPK, "sources": pool_v.pool_ids}, "feature_sources": list(lin_ids), "linear_features": list(lin_ids),
            "source_sha256": {i: by_id[i].sha for i in lin_ids},
            "configuration": {g: {"config": models[g].config, "evidence": models[g].evidence} for g in models},
            "training_data": "labels of the group's training dev rows only (their out-of-fold features); lockbox group: all dev rows",
            "stacking_caveat": "training rows' out-of-fold features were produced by models that saw the group's held-out fold "
                               "(indirect, second-order); the lockbox group is unaffected",
            "seeds": None, "wall_seconds": time.perf_counter() - t0, "device": "cpu"})
        flog(f"{name} complete in {time.perf_counter() - t0:.0f}s")
    return info


# ----------------------------------------------------------------------------- evaluation and ablation


FAMILY_PIPELINES = {"NO-EMB": ["L_star", "LP", "K_lex", "K_lex_prior", "LTR_noemb_linear", "LTR_noemb_hgb", "LTR_noemb_hgb_full"],
                    "NO-EMB-noLSA": ["L_star", "LP", "K_lex", "K_lex_prior", "LTR_noemb_nolsa_linear", "LTR_noemb_nolsa_hgb", "LTR_noemb_nolsa_hgb_full"],
                    "EMB": ["H_bge", "K_lex_H_bge_prior", "LTR_emb_linear", "LTR_emb_hgb", "LTR_emb_hgb_full"]}
POPS = ("P_spec", "P", "S2_spec", "all")
STATIC_FAMILY_PIPELINES = {fam: list(names) for fam, names in FAMILY_PIPELINES.items()}


def pipeline_file(name: str) -> Path:
    return common.score_path(name) if common.score_path(name).exists() else pipeline_path(name)


def evaluate_matrix(matrix: np.ndarray, rows: np.ndarray | None = None) -> tuple[dict[str, Any], dict[str, dict]]:
    """Dev out-of-fold component-macro metrics (4 populations) and the P_spec / P evaluation records for contrasts."""
    dev = common.dev_rows() if rows is None else rows
    scores = np.asarray(matrix[dev])
    summary, keep = {}, {}
    for pop in POPS:
        res = common.evaluate(scores, dev, pop)
        summary[pop] = {"n_queries": res["n_queries"], "n_components": res["n_components"],
                        **{m: float(res["macro"][m]) for m in ("ndcg_at_10", "mrr", "recall_at_20")}}
        if pop in ("P_spec", "P"):
            keep[pop] = res
    return summary, keep


def evaluate_pipelines(names: Sequence[str]) -> tuple[dict[str, Any], dict[str, dict]]:
    summaries, records = {}, {}
    for n in names:
        if not pipeline_file(n).exists():
            continue
        t = time.perf_counter()
        summaries[n], records[n] = evaluate_matrix(np.load(pipeline_file(n), mmap_mode="r"))
        flog(f"dev metrics {n}: P_spec {summaries[n]['P_spec']['ndcg_at_10']:.4f} P {summaries[n]['P']['ndcg_at_10']:.4f} ({time.perf_counter() - t:.0f}s)")
    return summaries, records


def choose_best(summaries: dict[str, Any]) -> dict[str, dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for fam, names in FAMILY_PIPELINES.items():
        have = [n for n in names if n in summaries]
        best[fam] = {pop: max(have, key=lambda n: (round(summaries[n][pop]["ndcg_at_10"], 12), -have.index(n))) for pop in ("P_spec", "P")}
    return best


def dev_contrasts(records: dict[str, dict], best: dict[str, dict[str, str]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def add(label: str, a: str, b: str, pop: str) -> None:
        if a in records and b in records:
            out[label] = {"a": a, "b": b, **common.contrast(records[a][pop], records[b][pop])}

    for pop in ("P_spec", "P"):
        add(f"BEST_EMB - BEST_NOEMB ({pop} choice, evaluated on {pop})", best["EMB"][pop], best["NO-EMB"][pop], pop)
    add("BEST_EMB(P_spec choice) - BEST_NOEMB(P_spec choice) on P", best["EMB"]["P_spec"], best["NO-EMB"]["P_spec"], "P")
    if "NO-EMB-noLSA" in best:
        for pop in ("P_spec", "P"):
            add(f"SENSITIVITY BEST_EMB - BEST_NOEMB without LSA sources ({pop} choice, evaluated on {pop})", best["EMB"][pop], best["NO-EMB-noLSA"][pop], pop)
        add("LTR_noemb_hgb - LTR_noemb_nolsa_hgb (what the LSA controls add to NO-EMB)", "LTR_noemb_hgb", "LTR_noemb_nolsa_hgb", "P_spec")
    add("LTR_emb_hgb - LTR_noemb_hgb", "LTR_emb_hgb", "LTR_noemb_hgb", "P_spec")
    add("LTR_emb_hgb_full - LTR_noemb_hgb_full (on P)", "LTR_emb_hgb_full", "LTR_noemb_hgb_full", "P")
    add("LTR_emb_linear - LTR_noemb_linear", "LTR_emb_linear", "LTR_noemb_linear", "P_spec")
    add("K_lex_H_bge_prior - K_lex_prior", "K_lex_H_bge_prior", "K_lex_prior", "P_spec")
    add("KP_emb_all - KP_noemb_all (stacked hand fusions)", "KP_emb_all", "KP_noemb_all", "P_spec")
    add("H_bge - L_star", "H_bge", "L_star", "P_spec")
    return out


def ablate(ctx: base.Ctx, F: Features, info: dict[str, dict], summaries: dict[str, Any]) -> dict[str, Any]:
    """Remove one EMB feature group at a time (pool and features) from the better LTR_emb model; dev groups only."""
    sources = F.sources
    kinds = [k for k in ("linear", "hgb") if f"LTR_emb_{k}" in summaries]
    if not kinds:
        return {}
    kind = max(kinds, key=lambda k: summaries[f"LTR_emb_{k}"]["P_spec"]["ndcg_at_10"])
    full_name = f"LTR_emb_{kind}"
    cfg_full = info.get(full_name, {}).get("configuration", {})
    fixed = {role: cfg_full[role]["config"] for role in cfg_full} if kind == "hgb" and cfg_full else None
    groups = sorted({s.group for s in sources if s.family == "EMB"})
    dev_groups = [g for g in ctx.groups if not g.is_lockbox]
    lstar = np.asarray(common.open_scores("L_star"))
    matrix = np.full((len(ctx.data.queries), ctx.N), np.nan, dtype=np.float32)
    out: dict[str, Any] = {"model": full_name, "reference_P_spec": summaries[full_name]["P_spec"]["ndcg_at_10"],
                           "reference_P": summaries[full_name]["P"]["ndcg_at_10"], "removed": {}}
    for grp in groups:
        drop = {s.id for s in sources if s.group == grp and s.family == "EMB"}
        v = make_variant(F, f"emb-minus-{grp}", [s.id for s in sources if s.kind == "full" and s.id not in drop],
                         [s.id for s in sources if s.id not in drop])
        t = time.perf_counter()
        models = fit_variant(ctx, F, v, kind, dev_groups, fixed)
        for g in dev_groups:
            m = models[g.role]
            write_pooled(ctx, F, matrix, g.rows, m.pair_idx, m.pair_scores, lstar)
        summary, _ = evaluate_matrix(matrix, common.dev_rows())
        out["removed"][grp] = {"sources": sorted(drop), "P_spec": summary["P_spec"]["ndcg_at_10"], "P": summary["P"]["ndcg_at_10"],
                               "S2_spec": summary["S2_spec"]["ndcg_at_10"], "all": summary["all"]["ndcg_at_10"],
                               "delta_P_spec": summary["P_spec"]["ndcg_at_10"] - out["reference_P_spec"],
                               "delta_P": summary["P"]["ndcg_at_10"] - out["reference_P"]}
        flog(f"ablation without {grp}: P_spec {summary['P_spec']['ndcg_at_10']:.4f} ({time.perf_counter() - t:.0f}s)")
    return out


def add_one(ctx: base.Ctx, F: Features, info: dict[str, dict], summaries: dict[str, Any], kind: str | None = None) -> dict[str, Any]:
    """NO-EMB pipeline plus one EMB feature group at a time (pool and features), same model type and hyper-parameters as LTR_emb."""
    sources = F.sources
    kinds = [k for k in ("linear", "hgb") if f"LTR_emb_{k}" in summaries and f"LTR_noemb_{k}" in summaries]
    if not kinds:
        return {}
    kind = kind or max(kinds, key=lambda k: summaries[f"LTR_emb_{k}"]["P_spec"]["ndcg_at_10"])
    cfg_emb = info.get(f"LTR_emb_{kind}", {}).get("configuration", {})
    fixed = {role: cfg_emb[role]["config"] for role in cfg_emb} if kind == "hgb" and cfg_emb else None
    base_name = f"LTR_noemb_{kind}"
    noemb_full = [s.id for s in sources if s.family == "NO-EMB" and s.kind == "full"]
    noemb_all = [s.id for s in sources if s.family == "NO-EMB"]
    dev_groups = [g for g in ctx.groups if not g.is_lockbox]
    lstar = np.asarray(common.open_scores("L_star"))
    matrix = np.full((len(ctx.data.queries), ctx.N), np.nan, dtype=np.float32)
    out: dict[str, Any] = {"model": f"LTR_<family>_{kind}", "reference_noemb_P_spec": summaries[base_name]["P_spec"]["ndcg_at_10"],
                           "reference_noemb_P": summaries[base_name]["P"]["ndcg_at_10"],
                           "reference_emb_P_spec": summaries[f"LTR_emb_{kind}"]["P_spec"]["ndcg_at_10"],
                           "reference_emb_P": summaries[f"LTR_emb_{kind}"]["P"]["ndcg_at_10"], "added": {}}
    for grp in sorted({s.group for s in sources if s.family == "EMB"}):
        add = [s.id for s in sources if s.group == grp and s.family == "EMB"]
        v = make_variant(F, f"noemb-plus-{grp}", noemb_full + [i for i in add if any(x.id == i and x.kind == "full" for x in sources)],
                         noemb_all + add)
        t = time.perf_counter()
        models = fit_variant(ctx, F, v, kind, dev_groups, fixed)
        for g in dev_groups:
            m = models[g.role]
            write_pooled(ctx, F, matrix, g.rows, m.pair_idx, m.pair_scores, lstar)
        summary, _ = evaluate_matrix(matrix, common.dev_rows())
        out["added"][grp] = {"sources": sorted(add), "P_spec": summary["P_spec"]["ndcg_at_10"], "P": summary["P"]["ndcg_at_10"],
                             "delta_P_spec": summary["P_spec"]["ndcg_at_10"] - out["reference_noemb_P_spec"],
                             "delta_P": summary["P"]["ndcg_at_10"] - out["reference_noemb_P"]}
        flog(f"NO-EMB plus {grp}: P_spec {summary['P_spec']['ndcg_at_10']:.4f} ({time.perf_counter() - t:.0f}s)")
    return out


def dev_bands(ctx: base.Ctx, records: dict[str, dict], best: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Dev pair-level Recall@20 by fold-local popularity band for BEST_EMB vs BEST_NO-EMB (bands from each outer fold's training rows)."""
    from uol.embedding_push import final

    d = common.release()
    by_fold = {}
    for k in range(5):
        ids = [d.queries[int(r)].query_id for r in common.dev_rows() if ctx.fold[int(r)] != k]
        counts = prior_mod.component_counts(d.references, d.component_by_query, ids, ctx.N, d.candidate_index)
        by_fold[k] = np.where(counts == 0, "unseen", np.where(counts <= prior_mod.RARE_MAX, "rare", "frequent"))
    out = {}
    for pop in ("P_spec", "P"):
        a, b = best["EMB"][pop], best["NO-EMB"][pop]
        if a in records and b in records:
            out[pop] = {"a": a, "b": b, "table": final.band_table(records[a][pop], records[b][pop], lambda row: by_fold[int(ctx.fold[row])], draws=2000)}
    return out


# ----------------------------------------------------------------------------- report


def fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def selected_configs_text() -> list[str]:
    lines: list[str] = []
    lex = json.loads((base.SELECTION_DIR / "lexical.json").read_text(encoding="utf-8"))
    nest = json.loads((base.SELECTION_DIR / "nested.json").read_text(encoding="utf-8"))
    hand_path = base.SELECTION_DIR / "hand.json"
    hand = json.loads(hand_path.read_text(encoding="utf-8")) if hand_path.exists() else {}
    lines += ["| scoring group | L_star | knn k | K_lex (k, w_knn) | LP w_prior | H_bge | K_lex_prior w_prior | K_lex_H_bge_prior (w_dense, w_prior) |",
              "|---|---|---|---|---|---|---|---|"]
    for role, sel in nest.items():
        ls = lex[role]["selected"]
        lstar = f"{ls['kind']}:{ls.get('base') or ''} k1={ls.get('k1')} b={ls.get('b')} w_char={ls.get('w_char')}"
        h = sel["H_bge"]["config"]
        hh = "rrf k=60" if h["type"] == "rrf" else f"linear w_dense={h['w_dense']}"
        hp = hand.get(role, {})
        lines.append(f"| {role} | {lstar} | {sel['knn_lex_votes']['k']} | ({sel['K_lex']['k']}, {sel['K_lex']['w_knn']}) | {sel['LP']['w_prior']} | {hh} | "
                     f"{hp.get('K_lex_prior', {}).get('w_prior', '')} | ({hp.get('K_lex_H_bge_prior', {}).get('w_dense', '')}, "
                     f"{hp.get('K_lex_H_bge_prior', {}).get('w_prior', '')}) |")
    return lines


def write_dev_results(sources: list[Source], F: Features, info: dict[str, dict], summaries: dict[str, Any], best: dict, contrasts: dict,
                      ablation: dict, stats: dict[str, dict], addition: dict | None = None, bands: dict | None = None) -> None:
    L = ["# Fusion and learning-to-rank: dev results (EP-alpha phase 2)", "",
         f"Generated {base.utc()} from {len(sources)} COMPLETE sources; code version `{code_version()}`. All numbers are dev out-of-fold, "
         "component-macro; no lockbox metric exists. Each dev row is scored by a model trained only on its group's training folds; "
         "the lockbox rows were written by models trained on all dev rows and are not evaluated here. "
         "Dev contrasts are selection-biased (pipelines and hyper-parameters were chosen on these numbers).", "",
         "## Sources", "", "Family rule: NO-EMB only for this study's lexical sources, and for files that declare NO-EMB while naming no embedding "
         "resource (bge, e5, embedding, ColBERT, encoder, prototype, fine-tuning); everything else is EMB. A mismatch with the declared family is shown.", "",
         "| id | kind | family | declared | feature group | derived | sha256 |", "|---|---|---|---|---|---|---|"]
    for s in sources:
        declared = str(s.manifest.get("family") or "")
        L.append(f"| {s.id} | {s.kind} | {s.family} | {declared}{' (differs)' if declared and declared.upper() != s.family else ''} | {s.group} | "
                 f"{'yes' if s.derived else ''} | {s.sha[:10]} |")
    L += ["", "## Candidate pools (top-K of each full source of the family; positives never injected)", "",
          "| family | pooled sources | mean pool size (dev P_spec rows) | specific-positive recall in pool (dev P_spec) | P_spec queries with a pooled positive |",
          "|---|---|---:|---:|---:|"]
    for fam, st in stats.items():
        L.append(f"| {fam} | {', '.join(st['sources'])} | {st['mean_pool_size_dev_P_spec']:.0f} | "
                 f"{st['specific_positive_recall_in_pool_dev_P_spec']:.3f} | {st['share_P_spec_queries_with_pooled_positive']:.3f} |")
    L += ["", "## Pipelines (dev out-of-fold nDCG@10 / MRR / Recall@20)", "",
          "| family | pipeline | P_spec nDCG@10 | MRR | R@20 | P nDCG@10 | MRR | R@20 | S2_spec nDCG@10 | all nDCG@10 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fam, names in FAMILY_PIPELINES.items():
        for n in names:
            if n not in summaries:
                continue
            r = summaries[n]
            L.append(f"| {fam} | {n} | {fmt(r['P_spec']['ndcg_at_10'])} | {fmt(r['P_spec']['mrr'])} | {fmt(r['P_spec']['recall_at_20'])} | "
                     f"{fmt(r['P']['ndcg_at_10'])} | {fmt(r['P']['mrr'])} | {fmt(r['P']['recall_at_20'])} | "
                     f"{fmt(r['S2_spec']['ndcg_at_10'])} | {fmt(r['all']['ndcg_at_10'])} |")
    first = next(iter(summaries.values()))
    L += ["", "Rows / components: " + "; ".join(f"{p} {first[p]['n_queries']}/{first[p]['n_components']}" for p in POPS), "",
          "## Best pipeline per family (dev)", "", "| family | by dev P_spec nDCG@10 | by dev P nDCG@10 |", "|---|---|---|"]
    for fam in best:
        L.append(f"| {fam} | {best[fam]['P_spec']} ({fmt(summaries[best[fam]['P_spec']]['P_spec']['ndcg_at_10'])}) | "
                 f"{best[fam]['P']} ({fmt(summaries[best[fam]['P']]['P']['ndcg_at_10'])}) |")
    if "NO-EMB-noLSA" in best:
        L += ["", "Sensitivity (lead ruling D4): `NO-EMB-noLSA` repeats the NO-EMB pipelines without any LSA-built source "
              f"({', '.join(s.id for s in sources if s.lsa)}); the primary contrast is reported with BEST_NO-EMB from `NO-EMB` and from `NO-EMB-noLSA`.", ""]
    L += ["", "## Dev contrasts (paired component bootstrap, 10,000 draws, seed 20261004; nDCG@10; unadjusted, selection-biased)", "",
          "| contrast | a | b | mean a | mean b | effect | 95% CI | 90% CI | components | verdict |", "|---|---|---|---:|---:|---:|---|---|---:|---|"]
    for label, c in contrasts.items():
        L.append(f"| {label} | {c['a']} | {c['b']} | {fmt(c['mean_a'])} | {fmt(c['mean_b'])} | {c['effect']:+.4f} | "
                 f"[{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}] | [{c['ci90'][0]:+.4f}, {c['ci90'][1]:+.4f}] | {c['components']} | {c['verdict']} |")
    if ablation:
        L += ["", f"## Ablation: one EMB feature group removed from {ablation['model']} (pool and features; same hyper-parameters)", "",
              f"Reference (all EMB groups): P_spec {fmt(ablation['reference_P_spec'])}, P {fmt(ablation['reference_P'])}.", "",
              "| removed group | sources | P_spec nDCG@10 | delta | P nDCG@10 | delta |", "|---|---|---:|---:|---:|---:|"]
        for grp, r in ablation["removed"].items():
            L.append(f"| {grp} | {', '.join(r['sources'])} | {fmt(r['P_spec'])} | {r['delta_P_spec']:+.4f} | {fmt(r['P'])} | {r['delta_P']:+.4f} |")
    if addition:
        L += ["", "## Forward addition: NO-EMB pipeline plus one EMB feature group (same model type and hyper-parameters as LTR_emb)", "",
              f"References: NO-EMB P_spec {fmt(addition['reference_noemb_P_spec'])}, P {fmt(addition['reference_noemb_P'])}; "
              f"all EMB groups P_spec {fmt(addition['reference_emb_P_spec'])}, P {fmt(addition['reference_emb_P'])}.", "",
              "| added group | sources | P_spec nDCG@10 | delta vs NO-EMB | P nDCG@10 | delta vs NO-EMB |", "|---|---|---:|---:|---:|---:|"]
        for grp, r in addition["added"].items():
            L.append(f"| {grp} | {', '.join(r['sources'])} | {fmt(r['P_spec'])} | {r['delta_P_spec']:+.4f} | {fmt(r['P'])} | {r['delta_P']:+.4f} |")
    if bands:
        L += ["", "## Dev pair-level Recall@20 by fold-local popularity band (BEST_EMB vs BEST_NO-EMB; bands from each fold's training rows)", ""]
        for pop, rec in bands.items():
            L += [f"### {pop}: a = {rec['a']}, b = {rec['b']}", "", "| band | pairs | a | b | a - b | 95% CI |", "|---|---:|---:|---:|---:|---|"]
            for band in ("frequent", "rare", "unseen", "all"):
                r = rec["table"][band]
                L.append(f"| {band} | {int(r['pairs'])} | {fmt(r['a'])} | {fmt(r['b'])} | {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
            L.append("")
    L += ["", "## Selected configurations per scoring group (hand pipelines)", ""] + selected_configs_text()
    kp = [n for n in info if n.startswith("KP_")]
    if kp:
        L += ["", "## Stacked hand fusions anchored on K_lex: weights per scoring group (per-query robust z; K_lex = 1)", ""]
        for n in kp:
            L += [f"### {n}", "", "| group | weights | train P_spec nDCG@10 |", "|---|---|---:|"]
            for role, rec in info[n]["configuration"].items():
                w = ", ".join(f"{k}={v:g}" for k, v in rec["config"]["weights_z"].items() if v)
                L.append(f"| {role} | {w} | {fmt(rec['evidence']['train_objective'], 4)} |")
            L.append("")
    L += ["", "## Learned models: per-group configuration", ""]
    for n in ("LTR_noemb_linear", "LTR_noemb_nolsa_linear", "LTR_emb_linear"):
        if n in info:
            L += [f"### {n}: coordinate-ascent weights on per-query robust-z features (L_star anchored at 1)", "", "| group | weights | train P_spec nDCG@10 |", "|---|---|---:|"]
            for role, rec in info[n]["configuration"].items():
                w = ", ".join(f"{k}={v:g}" for k, v in rec["config"]["weights_z"].items() if v)
                L.append(f"| {role} | {w} | {fmt(rec['evidence']['train_objective'], 4)} |")
            L.append("")
    for n in ("LTR_noemb_hgb", "LTR_noemb_nolsa_hgb", "LTR_emb_hgb"):
        if n in info and info[n]["configuration"]:
            L += [f"### {n}: HistGradientBoosting (inner-CV grid on learning rate and max depth)", "", "| group | chosen | inner objective by config |", "|---|---|---|"]
            for role, rec in info[n]["configuration"].items():
                grid = rec["evidence"].get("grid", [])
                L.append(f"| {role} | lr={rec['config']['learning_rate']}, depth={rec['config']['max_depth']} | "
                         + "; ".join(f"{r['config']['learning_rate']}/{r['config']['max_depth']}: {r['objective']:.4f}" for r in grid) + " |")
            L.append("")
    L += ["## Notes", "",
          "- Stacking caveat: learned models are trained on the training rows' own out-of-fold features. Those features come from models that "
          "saw the group's held-out fold, so the held-out labels reach the learned model only indirectly (second order). The lockbox group is "
          "unaffected because it trains on all dev rows.",
          "- Hyper-parameters of every pipeline were chosen per group by inner CV over its training folds (objective: pooled P_spec nDCG@10). "
          "Best-per-family choices use the dev numbers above and are therefore optimistic.",
          "- NO-EMB pools and features contain no embedding-derived source; EMB pools and features contain every source. "
          "The instrument citation prior, log body length and log query length are available to both families.",
          "- Re-run with `python -m uol.embedding_push.fusion_ltr --rebuild` after new score files arrive (cached pipelines whose "
          "inputs are unchanged are reused)."]
    (FUSION / "DEV_RESULTS.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def run_rebuild(force: bool = False, skip_hgb: bool = False, no_ablation: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    FUSION.mkdir(parents=True, exist_ok=True)
    ctx = base.Ctx()
    stage_hand(ctx, force)
    sources = discover()
    flog("sources: " + ", ".join(f"{s.id}[{s.kind},{s.family},{s.group}]" for s in sources))
    F = build_features(ctx, sources)
    info = stage_ltr(ctx, F, force, skip_hgb)
    hand_info = stage_partner_hand(ctx, F, force)
    info.update(hand_info)
    for fam in list(STATIC_FAMILY_PIPELINES):
        FAMILY_PIPELINES[fam] = list(STATIC_FAMILY_PIPELINES[fam])
    if not any(s.lsa for s in sources):
        FAMILY_PIPELINES.pop("NO-EMB-noLSA", None)
    for fam, ids in (("NO-EMB", ["K_lex_ext"]), ("NO-EMB-noLSA", ["K_lex_ext"]), ("EMB", ["K_qwen8b", "H_qwen8b"])):
        for i in ids:
            if fam in FAMILY_PIPELINES and any(x.id == i for x in sources) and i not in FAMILY_PIPELINES[fam]:
                FAMILY_PIPELINES[fam].insert(1, i)
    anchors = {"NO-EMB": "LTR_noemb_linear", "NO-EMB-noLSA": "LTR_noemb_nolsa_linear", "EMB": "LTR_emb_linear"}
    for name, fam, _pool, _ids in hand_specs(F):
        FAMILY_PIPELINES[fam].insert(FAMILY_PIPELINES[fam].index(anchors[fam]), name)
    names = list(dict.fromkeys(n for fam in FAMILY_PIPELINES.values() for n in fam))
    summaries, records = evaluate_pipelines(names)
    best = choose_best(summaries)
    contrasts = dev_contrasts(records, best)
    stats = {}
    for fam, v in make_variants(F).items():
        stats[VARIANT_FAMILY[fam]] = {"sources": v.pool_ids, **pool_stats(ctx, F, v)}
    ablation = {} if no_ablation else ablate(ctx, F, info, summaries)
    addition = {} if no_ablation else add_one(ctx, F, info, summaries)
    bands = dev_bands(ctx, records, best)
    write_dev_results(sources, F, info, summaries, best, contrasts, ablation, stats, addition, bands)
    serial = {"generated_utc": base.utc(), "code_version": code_version(), "summaries": summaries, "best": best,
              "contrasts": contrasts, "ablation": ablation, "addition": addition, "bands": bands, "pool_stats": stats,
              "sources": [{"id": s.id, "kind": s.kind, "family": s.family, "group": s.group, "derived": s.derived, "sha256": s.sha} for s in sources],
              "pipeline_manifests": {n: {"scores_sha256": json.loads((pipeline_file(n).parent / "manifest.json").read_text(encoding="utf-8")).get("scores_sha256")}
                                     for n in names if pipeline_file(n).exists()}}
    (FUSION / "dev_results.json").write_text(json.dumps(serial, indent=1, default=float), encoding="utf-8")
    flog(f"rebuild complete in {time.perf_counter() - t0:.0f}s")
    return serial


# ----------------------------------------------------------------------------- lockbox ablation score rows (no metric is computed here)

ABL_DIR = FUSION / "ablation_lockbox"


def write_lockbox_ablation_scores() -> dict[str, Any]:
    """Score the lockbox rows (models trained on all dev rows, frozen hyper-parameters) for LTR_emb_hgb with each EMB group removed
    and for LTR_noemb_hgb plus each EMB group. Only score rows are written; the metrics are computed by ``final.py`` after approval."""
    ctx = base.Ctx()
    F = build_features(ctx, discover())
    sources = F.sources
    results = json.loads((FUSION / "dev_results.json").read_text(encoding="utf-8"))
    if [(s.id, s.sha) for s in sources] != [(s["id"], s["sha256"]) for s in results["sources"]]:
        raise RuntimeError("score files changed since the last --rebuild")
    manifest = json.loads((pipeline_file("LTR_emb_hgb").parent / "manifest.json").read_text(encoding="utf-8"))
    fixed = {role: rec["config"] for role, rec in manifest["configuration"].items()}
    lock = [g for g in ctx.groups if g.is_lockbox]
    lstar = np.asarray(common.open_scores("L_star"))
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    noemb_full = [s.id for s in sources if s.family == "NO-EMB" and s.kind == "full"]
    noemb_all = [s.id for s in sources if s.family == "NO-EMB"]
    full_ids = {s.id for s in sources if s.kind == "full"}
    out: dict[str, Any] = {}
    jobs: list[tuple[str, Variant]] = []
    for grp in sorted({s.group for s in sources if s.family == "EMB"}):
        grp_ids = [s.id for s in sources if s.group == grp and s.family == "EMB"]
        drop = set(grp_ids)
        jobs.append((f"emb_minus_{grp}", make_variant(F, f"emb-minus-{grp}", [i for i in full_ids if i not in drop], [s.id for s in sources if s.id not in drop])))
        jobs.append((f"noemb_plus_{grp}", make_variant(F, f"noemb-plus-{grp}", noemb_full + [i for i in grp_ids if i in full_ids], noemb_all + grp_ids)))
    for name, v in jobs:
        models = fit_variant(ctx, F, v, "hgb", lock, fixed)
        matrix = np.full((len(ctx.data.queries), ctx.N), np.nan, dtype=np.float32)
        for g in lock:
            m = models[g.role]
            write_pooled(ctx, F, matrix, g.rows, m.pair_idx, m.pair_scores, lstar)
        path = ABL_DIR / f"{name}.npy"
        np.save(path, matrix[lock[0].rows])
        out[name] = {"path": str(path), "sha256": common.sha256_file(path), "rows": "lockbox rows in release order", "variant": name,
                     "feature_sources": list(v.feature_ids), "pool_sources": list(v.pool_ids)}
        flog(f"lockbox ablation scores written: {name}")
    record = {"written_utc": base.utc(), "model": "LTR hgb (specific labels), frozen per-group hyper-parameters of LTR_emb_hgb",
              "files": out, "no_metric_computed": True}
    (ABL_DIR / "manifest.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


# ----------------------------------------------------------------------------- phase 3: freeze record (lead_approved stays false)


def write_freeze() -> dict[str, Any]:
    """Record the selected pipelines, their configurations, hashes and dev results; the lead approves by editing ``lead_approved``."""
    results = json.loads((FUSION / "dev_results.json").read_text(encoding="utf-8"))
    current = [(s.id, s.sha) for s in discover()]
    if current != [(s["id"], s["sha256"]) for s in results["sources"]]:
        raise RuntimeError("score files changed since the last --rebuild; rebuild before freezing")
    if common.FREEZE_PATH.exists():
        raise FileExistsError(f"{common.FREEZE_PATH} exists; the freeze is written once")
    chosen: dict[str, dict[str, Any]] = {}
    needed: set[str] = set()
    for fam, picks in results["best"].items():
        chosen[fam] = {}
        for crit, name in picks.items():
            manifest = json.loads((pipeline_file(name).parent / "manifest.json").read_text(encoding="utf-8"))
            chosen[fam][crit] = {"pipeline": name, "scores_path": str(pipeline_file(name)), "scores_sha256": common.sha256_file(pipeline_file(name)),
                                 "manifest_sha256": common.sha256_file(pipeline_file(name).parent / "manifest.json"),
                                 "configuration": manifest.get("configuration"), "family": manifest.get("family"),
                                 "dev": results["summaries"][name], "feature_sources": manifest.get("feature_sources"),
                                 "pool_sources": (manifest.get("pool") or {}).get("sources")}
            needed |= set(manifest.get("feature_sources") or [])
    qwen_pipes: dict[str, Any] = {}
    for name in results["summaries"]:
        mpath = pipeline_file(name).parent / "manifest.json"
        if not mpath.exists():
            continue
        m = json.loads(mpath.read_text(encoding="utf-8"))
        used = set(m.get("feature_sources") or []) | set((m.get("pool") or {}).get("sources") or [])
        if "qwen" in name.lower() or any("qwen" in i.lower() for i in used):
            qwen_pipes[name] = {"path": str(pipeline_file(name)), "sha256": common.sha256_file(pipeline_file(name))}
    qwen_sources = {s["id"]: {"path": str(common.SCORES / s["id"] / "scores.npy"), "sha256": common.sha256_file(common.SCORES / s["id"] / "scores.npy")}
                    for s in results["sources"] if "qwen" in s["id"].lower() and s["kind"] == "full"}
    upstream = {}
    for s in results["sources"]:
        path = common.SCORES / s["id"] / ("scores.npy" if s["kind"] == "full" else "pairs.npz")
        upstream[s["id"]] = {"path": str(path), "sha256": common.sha256_file(path), "kind": s["kind"], "family": s["family"],
                             "manifest_sha256": common.sha256_file(common.SCORES / s["id"] / "manifest.json")}
    record = {"lead_approved": False, "written_utc": base.utc(), "protocol_sha256": common.sha256_file(common.ROOT / "research/programs/embedding_push_20261004/PROTOCOL.md"),
              "split_sha256": common.sha256_file(common.SPLIT_PATH),
              "selection": "per family, the pipeline with the highest dev P_spec nDCG@10 (primary) and, separately, the highest dev P nDCG@10",
              "primary_contrast": "BEST_EMB(P_spec choice) - BEST_NOEMB(P_spec choice), lockbox P_spec nDCG@10, paired component bootstrap (10,000 draws, seed 20261004)",
              "secondary_lockbox_analyses": [
                  "primary pair also on P, S2_spec and all (nDCG@10), and P_spec MRR / Recall@20",
                  "P-choice pair on P",
                  "sensitivity: BEST_EMB vs BEST_NO-EMB without any LSA-built source (NO-EMB-noLSA family), P_spec and P",
                  "pair-level Recall@20 by popularity band (frequent 5+, rare 1-4, unseen 0 dev components), bands from all dev rows",
                  "contamination check: Qwen-containing pipelines and sources on lockbox rows before vs on/after 2025-06-05 "
                  "(empty 'before' side: all lockbox rows are dated 2025-06-27 or later) and on the earlier vs later half of the lockbox by publication date",
                  "per-family ablations on the lockbox: LTR_emb with each EMB feature group removed and NO-EMB plus one group "
                  "(Qwen-only = NO-EMB + qwen8b group, ft_e5-only = NO-EMB + finetuned_dense), computed from the frozen dev configurations"],
              "lockbox_ablation_scores": json.loads((ABL_DIR / "manifest.json").read_text(encoding="utf-8")) if (ABL_DIR / "manifest.json").exists() else None,
              "dev_ablation_groups": sorted({r["group"] for r in results["sources"] if r["family"] == "EMB"}),
              "pipelines": chosen, "upstream_score_files": upstream,
              "contamination_check": {"cutoff": "2025-06-05", "pipelines": qwen_pipes, "qwen_sources": qwen_sources,
                                      "note": "lockbox rows published before vs on/after the Qwen3-Embedding release; Qwen-containing pipelines and sources"},
              "code_sha256": {str(p): common.sha256_file(p) for p in CODE_FILES + [Path(common.__file__).with_name("final.py")] if Path(p).exists()},
              "dev_results": {"summaries": results["summaries"], "contrasts": results["contrasts"], "pool_stats": results["pool_stats"],
                              "ablation": results["ablation"]},
              "dev_results_file_sha256": common.sha256_file(FUSION / "dev_results.json"),
              "note": "Lockbox score rows already exist (models trained on all dev rows); no lockbox metric has been computed. "
                      "final.py refuses to run until lead_approved is set to true."}
    common.STATE.mkdir(parents=True, exist_ok=True)
    common.FREEZE_PATH.write_text(json.dumps(record, indent=1, default=float), encoding="utf-8")
    flog(f"FREEZE.json written (lead_approved=false): {({fam: {c: v['pipeline'] for c, v in d.items()} for fam, d in chosen.items()})}")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true", help="rebuild pool, features, pipelines and DEV_RESULTS.md from scores/")
    parser.add_argument("--force", action="store_true", help="refit even when cached pipelines are up to date")
    parser.add_argument("--skip-hgb", action="store_true", help="debug: linear models only")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--lockbox-ablation-scores", action="store_true", help="write lockbox score rows for the ablation variants (no metric)")
    parser.add_argument("--freeze", action="store_true", help="phase 3: write FREEZE.json with lead_approved=false from dev_results.json")
    args = parser.parse_args(argv)
    if args.lockbox_ablation_scores:
        write_lockbox_ablation_scores()
        return 0
    if args.freeze:
        write_freeze()
        return 0
    if not args.rebuild:
        parser.error("pass --rebuild (or --freeze)")
    run_rebuild(args.force, args.skip_hgb, args.no_ablation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
