"""E2 and E3 (representation experiments; DEVIATIONS E0.2, E0.3): supervised low-rank adapters on the frozen
Qwen3-Embedding-8B vectors (E2) and the same adapters over graph-smoothed article vectors (E3), measured dense alone
and as features over the rebuilt ``LTR_emb_hgb`` through the refit harness (``arm_runner``, D3.1 selection).

Data. Article vectors: the cached Qwen3-Embedding-8B windows (``data/derived/embedding_push_qwen8b_v1``), one vector
per article (mean of its windows, renormalised; only 8 articles have more than one window). Query vectors: variant c,
the one ``dense_qwen8b`` uses. No endpoint call, no GPU.

Adapters (``form``):
* ``qres``: query-side residual low-rank map q' = norm(q + B A q); B starts at 0 (the frozen geometry), articles fixed.
* ``proj``: two-tower low-rank projection q' = norm(P_q q), a' = norm(P_a a); both towers start at the top-r right
  singular vectors of the article matrix.
Loss: multi-positive softmax (-log of the softmax mass on the row's specific positives) at temperature ``tau`` over a
sampled candidate set: per row ``N_HARD`` hard negatives drawn from its top ``HARD_DEPTH`` non-cited candidates under an
out-of-fold LTR_emb_hgb refit, every other candidate of the batch, and ``N_RANDOM`` uniform non-generic articles; the
row's other cited articles are masked and generic articles never enter. AdamW (lr 1e-3, weight decay .01), batch 64,
at most 8 epochs; the kept epoch (0 = the initial map allowed) is the best inner-validation P_spec nDCG@10 on 10% of
the training components.

Strict cross-fitting. An adapter is keyed by the dev folds it excludes, E. It trains on dev rows of the other folds.
The hard negatives of a training row in fold i come from an LTR_emb_hgb refit trained on folds outside E ∪ {i}
(``features.nested_ltr`` for |E| = 1; ``dn_ltr`` on the two remaining folds for |E| = 2). Outer adapters (E = {k})
score fold k; nested adapters (E = {j, k}) give the columns of the fold-j training rows of outer group k (a_p12 rule).
Lockbox rows are never loaded as training, validation, negative or scored rows.

Columns (feature route): three per (query, pooled candidate), computed as ``fusion_ltr.full_features`` does for a
full score source: per-query robust z, log (min-)rank and the robust z of the best article of the candidate's
instrument, from the adapter's full-catalogue cosine row.

E3. Article vectors are smoothed over the label-free statute graph G2 before the adapter: Â = D^-1/2 (A + I) D^-1/2
over the chosen edge types (binary, symmetrised); SGC X̃ = Â² X; APPNP Z ← (1 − α) Â Z + α X, 10 steps, α = .1; rows
renormalised. Queries are never graph nodes. Per outer fold the adapter configuration is the one E2 selected, so E2's
assembled arm is the capacity-matched no-edge control.

    python -m uol.round3.e2_adapter prep                       # article matrix, SVD init, dn_ltr hard-negative refits
    python -m uol.round3.e2_adapter smoke                      # timing check: one short training (training loss only)
    python -m uol.round3.e2_adapter train --arm E2 [--seeds 1]
    python -m uol.round3.e2_adapter run --arm E2               # dense alone + arm_runner (refits); RESULT.json
    python -m uol.round3.e2_adapter seeds --arm E2             # selected configs at seeds 2 and 3
    python -m uol.round3.e2_adapter compare                    # E3 vs the E2 no-edge control
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.round3 import common as C

QWEN = C.ROOT / "data/derived/embedding_push_qwen8b_v1"
ARM_DIR = {"E2": C.ARMS / "E2_adapter", "E3": C.ARMS / "E3_graph"}
ARM_NAME = {"E2": "E2_adapter", "E3": "E3_graph"}
SHARED = C.ARMS / "E2_adapter" / "cache"
FOLDS = (0, 1, 2, 3, 4)

N_HARD, HARD_DEPTH, N_RANDOM = 16, 50, 1024
BATCH, MAX_EPOCHS, LR, WD, VAL_SHARE = 64, 8, 1e-3, 0.01, 0.10
THREADS = 8
SEEDS = (1, 2, 3)
SGC_K, APPNP_K, APPNP_ALPHA = 2, 10, 0.1
COLUMNS = ("z", "lr", "imaxz")

E2_GRID = [{"form": f, "rank": r, "tau": t} for f in ("qres", "proj") for r in (256, 512) for t in (0.02, 0.05)]
EDGE_SETS = {"hier": ("sibling", "chapter"), "adj": ("adj1", "adj2"), "xref": ("xref_int",),
             "all": ("sibling", "chapter", "adj1", "adj2", "xref_int")}
E3_GRID = [{"smooth": m, "edges": e} for m in ("sgc", "appnp") for e in EDGE_SETS]


def log(msg: str) -> None:
    print(f"[{C.utc()}] {msg}", flush=True)


# ----------------------------------------------------------------------------- release helpers

@lru_cache(maxsize=1)
def fold_array() -> np.ndarray:
    """Per release row: dev fold 0-4, -1 for lockbox (and any non-dev role)."""
    roles = C.roles()
    return np.asarray([int(str(r).replace("dev_fold", "")) if str(r).startswith("dev_fold") else -1 for r in roles])


@lru_cache(maxsize=1)
def generic_mask() -> np.ndarray:
    d = C.release()
    m = np.zeros(len(d.candidate_ids), dtype=bool)
    if len(d.generic_idx):
        m[np.asarray(d.generic_idx)] = True
    return m


def spec_positives(r: int) -> np.ndarray:
    g = generic_mask()
    return np.asarray(sorted(int(a) for a in C.release().specific_idx[int(r)] if not g[int(a)]), dtype=np.int64)


def cited(r: int) -> np.ndarray:
    return np.asarray(sorted(set(int(a) for a in C.release().positive_idx[int(r)])), dtype=np.int64)


def rows_of_folds(folds: Sequence[int]) -> np.ndarray:
    f = fold_array()
    rows = C.dev_rows()
    return rows[np.isin(f[rows], list(folds))]


def train_rows(E: frozenset) -> np.ndarray:
    """Dev rows outside the excluded folds with at least one specific (non-generic) positive."""
    rows = rows_of_folds([k for k in FOLDS if k not in E])
    out = np.asarray([int(r) for r in rows if len(spec_positives(int(r)))], dtype=np.int64)
    if np.isin(out, C.lockbox_rows()).any() or (len(out) and np.isin(fold_array()[out], list(E)).any()):
        raise PermissionError("adapter training rows must be dev rows outside the excluded folds")
    return out


def ekey(E: frozenset) -> str:
    return "".join(str(k) for k in sorted(E))


def eseed(E: frozenset, seed: int) -> int:
    """Distinct RNG seed per (excluded folds, seed): fold bitmask, so {1} and {0, 1} never collide."""
    return C.SEED + 7919 * int(seed) + 101 * sum(1 << int(k) for k in E)


# ----------------------------------------------------------------------------- article and query matrices

@lru_cache(maxsize=1)
def query_matrix() -> np.ndarray:
    ids = json.loads((QWEN / "query_ids.json").read_text(encoding="utf-8"))
    if ids != [q.query_id for q in C.release().queries]:
        raise ValueError("stored query vectors do not follow the release row order")
    return np.load(QWEN / "query_vectors_c.npy", mmap_mode="r")


def article_path(base: str) -> Path:
    return SHARED / f"articles_{base}.npy"


def build_raw_articles() -> Path:
    path = article_path("raw")
    if path.exists():
        return path
    v = np.load(QWEN / "article_vectors.npy")
    owner = np.load(QWEN / "article_index.npy").astype(np.int64)
    n = len(C.release().candidate_ids)
    if set(np.unique(owner).tolist()) != set(range(n)):
        raise ValueError("every catalogue article must have at least one vector")
    order = np.argsort(owner, kind="stable")
    starts = np.flatnonzero(np.r_[True, owner[order][1:] != owner[order][:-1]])
    out = np.add.reduceat(v[order].astype(np.float64), starts, axis=0)
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    SHARED.mkdir(parents=True, exist_ok=True)
    np.save(path, out.astype(np.float32))
    C.write_json(SHARED / "articles_raw.json", {"articles": n, "units": int(len(v)),
                                                 "multi_window_articles": int((np.bincount(owner) > 1).sum()),
                                                 "rule": "mean of an article's window vectors, L2-normalised",
                                                 "source_sha256": C.sha256_file(QWEN / "article_vectors.npy")})
    return path


def adjacency(edges: str):
    from scipy import sparse
    from uol.round3 import statute_graph
    g = statute_graph.load_graph()
    n = len(g["inst"])
    pairs = np.concatenate([g[f"e_{t}"] for t in EDGE_SETS[edges]]).astype(np.int64)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    a = np.concatenate([pairs[:, 0], pairs[:, 1]])
    b = np.concatenate([pairs[:, 1], pairs[:, 0]])
    A = sparse.csr_matrix((np.ones(len(a), dtype=np.float32), (a, b)), shape=(n, n))
    A.data[:] = 1.0                                         # binary after duplicate merge
    A = A + sparse.identity(n, dtype=np.float32, format="csr")
    deg = np.asarray(A.sum(axis=1)).ravel()
    dinv = sparse.diags((1.0 / np.sqrt(deg)).astype(np.float32))
    return (dinv @ A @ dinv).tocsr(), int(len(pairs))


def build_smoothed(method: str, edges: str) -> Path:
    base = f"{method}_{edges}"
    path = article_path(base)
    if path.exists():
        return path
    t0 = time.perf_counter()
    X = np.load(build_raw_articles())
    Ahat, n_edges = adjacency(edges)
    if method == "sgc":
        Z = X
        for _ in range(SGC_K):
            Z = Ahat @ Z
    elif method == "appnp":
        Z = X
        for _ in range(APPNP_K):
            Z = (1 - APPNP_ALPHA) * (Ahat @ Z) + APPNP_ALPHA * X
    else:
        raise ValueError(method)
    Z = np.asarray(Z, dtype=np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    np.save(path, Z)
    moved = float(np.mean(np.sum(Z * X, axis=1)))
    C.write_json(SHARED / f"articles_{base}.json", {"method": method, "edges": list(EDGE_SETS[edges]),
                                                    "undirected_edges_in": n_edges, "mean_cos_to_raw": moved,
                                                    "seconds": round(time.perf_counter() - t0, 1), "labels_used": False})
    log(f"smoothed {base}: mean cos to raw {moved:.3f}")
    return path


def article_matrix(base: str) -> np.ndarray:
    if base == "raw":
        return np.load(build_raw_articles(), mmap_mode="r")
    method, edges = base.split("_", 1)
    return np.load(build_smoothed(method, edges), mmap_mode="r")


_TENSOR: dict[str, Any] = {}


def article_tensor(base: str):
    """The article matrix as an in-memory torch tensor (one base kept per process)."""
    import torch
    if base not in _TENSOR:
        _TENSOR.clear()
        _TENSOR[base] = torch.from_numpy(np.ascontiguousarray(article_matrix(base), dtype=np.float32))
    return _TENSOR[base]


def svd_init(base: str) -> np.ndarray:
    """Top-512 right singular vectors [512, D] of the (uncentred) article matrix, by eigh of X^T X."""
    path = SHARED / f"svd512_{base}.npy"
    if path.exists():
        return np.load(path)
    X = np.asarray(article_matrix(base), dtype=np.float64)
    w, V = np.linalg.eigh(X.T @ X)
    P = V[:, np.argsort(-w)[:512]].T.astype(np.float32)
    np.save(path, P)
    return P


# ----------------------------------------------------------------------------- hard negatives (strictly cross-fitted)

@lru_cache(maxsize=1)
def pair_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from uol.embedding_push import fusion_ltr as fl
    pq = np.load(fl.FEATDIR / "pq.npy", mmap_mode="r")
    pc = np.load(fl.FEATDIR / "pc.npy", mmap_mode="r")
    row_ptr = np.searchsorted(pq, np.arange(len(C.release().queries) + 1))
    return pq, pc, row_ptr


DN_DIR = SHARED / "dn_ltr"


def dn_path(T: frozenset) -> Path:
    return DN_DIR / f"emb_T{ekey(T)}.npz"


def dn_ltr(T: frozenset) -> tuple[np.ndarray, np.ndarray]:
    """Pooled LTR_emb_hgb scores for the rows of the three folds in T from a refit on the two other dev folds.

    Frozen hyper-parameters and seeds of outer group min(T) (the ``features.nested_ltr`` recipe, one level deeper)."""
    if len(T) != 3:
        raise ValueError("dn_ltr needs three excluded folds")
    path = dn_path(T)
    if path.exists():
        z = np.load(path)
        return z["idx"], z["score"]
    from uol.round3 import features as FT
    h = FT.Harness("emb", "spec")
    k = min(T)
    g = next(x for x in h.groups if x.role == f"dev_fold{k}")
    folds, scored = dn_folds(T)
    rows_T = h.ctx.dev[np.isin(h.ctx.fold[h.ctx.dev], scored)]
    tr = h.ctx.dev[np.isin(h.ctx.fold[h.ctx.dev], folds)]
    gg = replace(g, rows=rows_T, train_rows=tr, train_folds=folds)
    with C.cpu_slot():
        idx, sc = h.fit_group(gg, None)
    DN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, idx=idx, score=sc)
    return idx, sc


def hard_source(E: frozenset, i: int) -> tuple[np.ndarray, np.ndarray]:
    from uol.round3 import features as FT
    if i in E:
        raise ValueError("training rows never come from an excluded fold")
    if not E:                 # lockbox group (F0.3): the frozen out-of-fold pipeline, whose fold-i rows never saw fold i
        from uol.embedding_push import fusion_ltr as fl
        pq, pc, row_ptr = pair_table()
        M = np.load(fl.pipeline_path("LTR_emb_hgb"), mmap_mode="r")
        rows = rows_of_folds([i])
        idx = np.concatenate([np.arange(int(row_ptr[r]), int(row_ptr[r + 1])) for r in rows])
        sc = np.concatenate([np.asarray(M[int(r)], np.float32)[np.asarray(pc[int(row_ptr[r]):int(row_ptr[r + 1])])]
                             for r in rows])
        return idx, sc
    if len(E) == 1:
        (k,) = tuple(E)
        path = FT.NESTED_DIR / f"emb_k{k}_j{i}.npz"
        if not path.exists():
            raise FileNotFoundError(f"nested_ltr cache missing: {path}")
        z = np.load(path)
        return z["idx"], z["score"]
    return dn_ltr(frozenset(E | {i}))


def hard_lists(E: frozenset, rows: Sequence[int]) -> dict[int, np.ndarray]:
    """{row: top-HARD_DEPTH non-cited, non-generic candidates under the cross-fitted LTR}."""
    pq, pc, _ = pair_table()
    gm = generic_mask()
    f = fold_array()
    out: dict[int, np.ndarray] = {}
    for i in sorted(set(int(f[r]) for r in rows)):
        idx, sc = hard_source(E, i)
        q = np.asarray(pq[idx])
        order = np.lexsort((-sc, q))
        idx, sc, q = idx[order], sc[order], q[order]
        for r in (int(x) for x in rows if f[x] == i):
            a, b = np.searchsorted(q, r), np.searchsorted(q, r, side="right")
            if b == a:
                raise RuntimeError(f"row {r} has no hard-negative scores")
            cand = np.asarray(pc[idx[a:b]], dtype=np.int64)
            bad = gm[cand] | np.isin(cand, cited(r))
            out[r] = cand[~bad][:HARD_DEPTH]
    return out


# ----------------------------------------------------------------------------- the adapter

def adapter_key(cfg: dict, base: str, E: frozenset, seed: int) -> str:
    return f"{cfg['form']}_r{cfg['rank']}_t{cfg['tau']}__{base}__E{ekey(E)}__s{seed}".replace(".", "p")


def _model(cfg: dict, dim: int, base: str):
    import torch

    class Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            r = int(cfg["rank"])
            if cfg["form"] == "qres":
                self.A = torch.nn.Linear(dim, r, bias=False)
                self.B = torch.nn.Linear(r, dim, bias=False)
                torch.nn.init.normal_(self.A.weight, std=1.0 / np.sqrt(dim))
                torch.nn.init.zeros_(self.B.weight)
            elif cfg["form"] == "proj":
                P = torch.from_numpy(svd_init(base)[:r].copy())
                self.Pq = torch.nn.Linear(dim, r, bias=False)
                self.Pa = torch.nn.Linear(dim, r, bias=False)
                with torch.no_grad():
                    self.Pq.weight.copy_(P)
                    self.Pa.weight.copy_(P)
            else:
                raise ValueError(cfg["form"])

        def q(self, x):
            if cfg["form"] == "qres":
                return torch.nn.functional.normalize(x + self.B(self.A(x)), dim=-1)
            return torch.nn.functional.normalize(self.Pq(x), dim=-1)

        def a(self, x):
            if cfg["form"] == "qres":
                return x
            return torch.nn.functional.normalize(self.Pa(x), dim=-1)

    return Adapter()


def score_rows(model, X, rows: Sequence[int], block: int = 256) -> np.ndarray:
    """Full-catalogue cosine scores [len(rows), N] (float32)."""
    import torch
    Qall = query_matrix()
    with torch.no_grad():
        Aall = model.a(X) if hasattr(model, "Pa") else X
        out = np.empty((len(rows), Aall.shape[0]), dtype=np.float32)
        for s in range(0, len(rows), block):
            q = torch.from_numpy(np.asarray(Qall[np.asarray(rows[s:s + block])], dtype=np.float32))
            out[s:s + block] = (model.q(q) @ Aall.T).numpy()
    return out


def train_adapter(cfg: dict, base: str, E: frozenset, seed: int, max_epochs: int = MAX_EPOCHS,
                  max_steps: int = 0) -> tuple[Any, dict]:
    import torch
    torch.set_num_threads(THREADS)
    torch.manual_seed(eseed(E, seed))
    t0 = time.perf_counter()
    X = article_tensor(base)
    Qall = query_matrix()
    rows = train_rows(E)
    comps = C.component_ids(rows)
    rng = np.random.default_rng(eseed(E, seed))
    uc = sorted(set(comps.tolist()))
    val_comps = set(rng.choice(uc, max(1, int(round(VAL_SHARE * len(uc)))), replace=False).tolist())
    is_val = np.isin(comps, list(val_comps))
    fit, val = rows[~is_val], rows[is_val]
    val_spec = C.population_rows("P_spec", val)
    hard = hard_lists(E, fit)
    pos = {int(r): spec_positives(int(r)) for r in fit}
    other = {int(r): np.setdiff1d(cited(int(r)), pos[int(r)]) for r in fit}
    nongeneric = np.flatnonzero(~generic_mask())
    model = _model(cfg, X.shape[1], base)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    tau = float(cfg["tau"])

    def val_ndcg() -> float:
        model.eval()
        S = score_rows(model, X, val_spec)
        return float(ep.evaluate(S, val_spec, "P_spec")["macro"]["ndcg_at_10"])

    history = [{"epoch": 0, "val_ndcg": val_ndcg()}]
    best = (history[0]["val_ndcg"], 0, {k: v.clone() for k, v in model.state_dict().items()})
    steps = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(len(fit))
        total, nb = 0.0, 0
        te = time.perf_counter()
        for s in range(0, len(perm), BATCH):
            rb = [int(fit[j]) for j in perm[s:s + BATCH]]
            hn = [rng.choice(hard[r], min(N_HARD, len(hard[r])), replace=False) for r in rb]
            rand = rng.choice(nongeneric, N_RANDOM, replace=False)
            cand = np.unique(np.concatenate([*(pos[r] for r in rb), *hn, rand]))
            posm = np.zeros((len(rb), len(cand)), dtype=bool)
            outm = np.zeros((len(rb), len(cand)), dtype=bool)
            for i, r in enumerate(rb):
                posm[i, np.searchsorted(cand, pos[r])] = True
                o = other[r][np.isin(other[r], cand)]
                outm[i, np.searchsorted(cand, o)] = True
            q = torch.from_numpy(np.asarray(Qall[np.asarray(rb)], dtype=np.float32))
            logits = (model.q(q) @ model.a(X[torch.from_numpy(cand)]).T) / tau
            logits = logits.masked_fill(torch.from_numpy(outm), float("-inf"))
            pos_l = logits.masked_fill(~torch.from_numpy(posm), float("-inf"))
            loss = (torch.logsumexp(logits, dim=1) - torch.logsumexp(pos_l, dim=1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss) * len(rb)
            nb += len(rb)
            steps += 1
            if max_steps and steps >= max_steps:
                break
        v = val_ndcg()
        history.append({"epoch": epoch, "train_loss": total / max(nb, 1), "val_ndcg": v,
                        "epoch_seconds": round(time.perf_counter() - te, 1)})
        if v > best[0] + 1e-9:
            best = (v, epoch, {k: t.clone() for k, t in model.state_dict().items()})
        if max_steps and steps >= max_steps:
            break
    model.load_state_dict(best[2])
    model.eval()
    meta = {"cfg": cfg, "base": base, "excluded_folds": sorted(E), "seed": seed, "train_rows": int(len(fit)),
            "val_rows": int(len(val)), "val_pspec_rows": int(len(val_spec)), "best_epoch": best[1],
            "best_val_ndcg": best[0], "history": history, "seconds": round(time.perf_counter() - t0, 1),
            "fixed": {"n_hard": N_HARD, "hard_depth": HARD_DEPTH, "n_random": N_RANDOM, "batch": BATCH,
                      "max_epochs": max_epochs, "lr": LR, "weight_decay": WD, "val_share": VAL_SHARE}}
    return model, meta


# ----------------------------------------------------------------------------- feature columns (frozen recipe)

@lru_cache(maxsize=1)
def instrument_layout() -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    from uol.embedding_push import fusion_ltr as fl
    from uol.facts_to_law import spec as ftl_spec
    d = C.release()
    instr = [row["instrument_id"] for row in ftl_spec.read_jsonl(Path(d.candidate_binding["articles_path"]))]
    code = {v: i for i, v in enumerate(sorted(set(instr)))}
    cand_instr = np.asarray([code[v] for v in instr])
    return cand_instr, fl.instrument_layout(cand_instr)


def full_columns(S: np.ndarray, rows: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """(pair-table positions, [n, 3] z / log-rank / instrument-max z) for ``rows`` from their full score rows S."""
    _, pc, row_ptr = pair_table()
    cand_instr, (perm, starts, segment) = instrument_layout()
    n_cand = S.shape[1]
    block = np.asarray(S, dtype=np.float32)
    med = np.median(block, axis=1)
    mad = np.median(np.abs(block - med[:, None]), axis=1)
    std = block.std(axis=1)
    scale = (1.4826 * mad).astype(np.float32)
    scale = np.where(scale > 1e-8, scale, np.where(std > 1e-8, std, 1.0)).astype(np.float32)
    ordered = np.sort(block, axis=1)
    imax = np.maximum.reduceat(block[:, perm], starts, axis=1)
    pos_all, cols_all = [], []
    for i, r in enumerate(rows):
        p = np.arange(int(row_ptr[r]), int(row_ptr[r + 1]))
        cc = np.asarray(pc[p], dtype=np.int64)
        raw = block[i, cc]
        z = (raw - med[i]) / scale[i]
        n_le = np.searchsorted(ordered[i], raw, side="right")
        lr = np.log((n_cand - n_le + 1).astype(np.float32))
        iz = (imax[i, segment[cand_instr[cc]]] - med[i]) / scale[i]
        pos_all.append(p)
        cols_all.append(np.stack([z, lr, iz], axis=1).astype(np.float32))
    return np.concatenate(pos_all), np.concatenate(cols_all)


# ----------------------------------------------------------------------------- adapters on disk

def adapter_dir(arm: str) -> Path:
    return ARM_DIR[arm] / "adapters"


def ensure_adapter(arm: str, cfg: dict, base: str, E: frozenset, seed: int) -> Path:
    """Train (if missing) one adapter and save its served-row columns, its meta and, for outer adapters, the dense-alone
    per-query records of the scored fold and the weights."""
    key = adapter_key(cfg, base, E, seed)
    out = adapter_dir(arm) / f"{key}.npz"
    if out.exists():
        return out
    model, meta = train_adapter(cfg, base, E, seed)
    X = article_tensor(base)
    served = rows_of_folds(sorted(E))
    if np.isin(served, C.lockbox_rows()).any():
        raise PermissionError("adapters serve dev rows only")
    pos_l, col_l = [], []
    dense: dict[str, list] = {"P_spec": [], "P": []}
    pspec = set(C.population_rows("P_spec", served).tolist())
    pfull = set(C.population_rows("P", served).tolist())
    for s in range(0, len(served), 256):
        rb = served[s:s + 256]
        S = score_rows(model, X, rb)
        p, c = full_columns(S, rb)
        pos_l.append(p)
        col_l.append(c)
        if len(E) == 1:
            for pop, keep in (("P_spec", pspec), ("P", pfull)):
                sel = [i for i, r in enumerate(rb) if int(r) in keep]
                if sel:
                    res = ep.evaluate(S[sel], rb[sel], pop)
                    for r_, rec in zip(res["rows"], res["per_query"]):
                        rec.pop("topk", None)
                        dense[pop].append({"row": int(r_), **{k: v for k, v in rec.items()}})
    adapter_dir(arm).mkdir(parents=True, exist_ok=True)
    extra = {}
    if len(E) == 1:
        sd = {k: v.numpy() for k, v in model.state_dict().items()}
        extra = {f"w_{k}": v for k, v in sd.items()}
        C.write_json(adapter_dir(arm) / f"{key}.dense.json", dense)
    meta["served_rows"] = int(len(served))
    meta["utc"] = C.utc()
    C.write_json(adapter_dir(arm) / f"{key}.json", meta)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez(tmp, rows=served, pos=np.concatenate(pos_l), cols=np.concatenate(col_l), **extra)
    tmp.replace(out)
    log(f"{arm} adapter {key}: best epoch {meta['best_epoch']} val {meta['best_val_ndcg']:.4f} "
        f"({meta['seconds']:.0f}s)")
    return out


def all_dev_adapter_path(arm: str, cfg: dict, base: str, seed: int) -> Path:
    return adapter_dir(arm) / f"{adapter_key(cfg, base, frozenset(), seed)}.weights.npz"


def ensure_all_dev_adapter(arm: str, cfg: dict, base: str, seed: int) -> Path:
    """The lockbox-group adapter (F0.3): the frozen recipe trained on all dev rows (E = {}); weights only.

    Training reads dev rows only; lockbox rows are scored by ``score_with_weights`` inside the batch."""
    out = all_dev_adapter_path(arm, cfg, base, seed)
    if out.exists():
        return out
    model, meta = train_adapter(cfg, base, frozenset(), seed)
    adapter_dir(arm).mkdir(parents=True, exist_ok=True)
    np.savez(out, **{f"w_{k}": v.numpy() for k, v in model.state_dict().items()})
    meta["utc"] = C.utc()
    C.write_json(out.with_suffix("").with_suffix(".json"), meta)
    log(f"{arm} all-dev adapter {out.name}: best epoch {meta['best_epoch']} val {meta['best_val_ndcg']:.4f}")
    return out


def score_with_weights(path: Path, cfg: dict, base: str, rows: Sequence[int]) -> np.ndarray:
    """Full-catalogue cosine rows of a saved adapter (outer or all-dev) for ``rows``."""
    import torch
    torch.set_num_threads(THREADS)
    X = article_tensor(base)
    model = _model(cfg, X.shape[1], base)
    z = np.load(path)
    model.load_state_dict({k[2:]: torch.from_numpy(z[k]) for k in z.files if k.startswith("w_")})
    model.eval()
    return score_rows(model, X, np.asarray(rows))


def excluded_sets(k: int) -> list[frozenset]:
    return [frozenset({k})] + [frozenset({j, k}) for j in FOLDS if j != k]


def adapter_for(k: int, fold: int) -> frozenset:
    """Excluded folds of the adapter that supplies the columns of a fold-``fold`` row inside outer group k."""
    return frozenset({k}) if fold == k else frozenset({fold, k})


def dn_folds(T: frozenset) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """(training folds, scored folds) of the doubly nested hard-negative refit for excluded set T."""
    return tuple(f for f in FOLDS if f not in T), tuple(sorted(T))


@lru_cache(maxsize=1)
def e2_selected() -> dict[int, dict]:
    res = json.loads((ARM_DIR["E2"] / "RESULT.json").read_text(encoding="utf-8"))
    return {int(k): {kk: v for kk, v in c.items() if kk in ("form", "rank", "tau")}
            for k, c in res["selected_config"].items()}


def plan(arm: str, cfg: dict, k: int) -> tuple[dict, str]:
    """(adapter config, article base) of arm config ``cfg`` inside outer group k."""
    if arm == "E2":
        return {x: cfg[x] for x in ("form", "rank", "tau")}, "raw"
    return e2_selected()[k], f"{cfg['smooth']}_{cfg['edges']}"


def grid(arm: str) -> list[dict]:
    return E2_GRID if arm == "E2" else E3_GRID


def train(arm: str, seeds: Sequence[int] = (1,), configs: Sequence[dict] | None = None,
          per_fold: dict[int, dict] | None = None) -> None:
    todo = []
    for seed in seeds:
        for k in FOLDS:
            cfgs = [per_fold[k]] if per_fold is not None else list(configs or grid(arm))
            for cfg in cfgs:
                acfg, base = plan(arm, cfg, k)
                for E in excluded_sets(k):
                    item = (adapter_key(acfg, base, E, seed), acfg, base, E, seed)
                    if item[0] not in {t[0] for t in todo}:
                        todo.append(item)
    log(f"{arm}: {len(todo)} adapters needed ({sum((adapter_dir(arm) / f'{t[0]}.npz').exists() for t in todo)} done)")
    for i, (key, acfg, base, E, seed) in enumerate(todo):
        ensure_adapter(arm, acfg, base, E, seed)
        if (i + 1) % 10 == 0:
            log(f"{arm}: {i + 1}/{len(todo)} adapters")


def load_adapter(arm: str, acfg: dict, base: str, E: frozenset, seed: int) -> dict:
    path = adapter_dir(arm) / f"{adapter_key(acfg, base, E, seed)}.npz"
    if not path.exists():
        raise FileNotFoundError(f"adapter not trained: {path.name}")
    z = np.load(path)
    return {"rows": z["rows"], "pos": z["pos"], "cols": z["cols"]}


def columns_for_arm(arm: str):
    def columns_for(cfg: dict, g, h, draw: int | None = None) -> np.ndarray:
        if getattr(g, "is_lockbox", False) or not str(g.role).startswith("dev_fold"):
            raise PermissionError("E2/E3 columns are built for dev groups only")
        if draw is not None:
            raise ValueError("no null draws are registered for E2/E3")
        k = int(str(g.role).replace("dev_fold", ""))
        seed = int(cfg.get("seed", 1))
        acfg, base = plan(arm, cfg, k)
        cols = h.empty_columns(len(COLUMNS))
        f = fold_array()
        pq, _, row_ptr = pair_table()
        for E in excluded_sets(k):
            a = load_adapter(arm, acfg, base, E, seed)
            row_fold = f[np.asarray(pq[a["pos"]])]
            keep = np.asarray([adapter_for(k, int(x)) == E for x in np.unique(row_fold)])
            keep = np.isin(row_fold, np.unique(row_fold)[keep])
            cols[a["pos"][keep]] = a["cols"][keep]
        dev = C.dev_rows()
        lo, hi = row_ptr[dev], row_ptr[dev + 1]
        dev_pos = np.concatenate([np.arange(a_, b_) for a_, b_ in zip(lo, hi)])
        if np.isnan(cols[dev_pos]).any():
            raise RuntimeError(f"{arm} group {k}: dev pairs without adapter columns")
        return cols
    return columns_for


# ----------------------------------------------------------------------------- dense alone

def dense_records(arm: str, acfg: dict, base: str, k: int, seed: int) -> dict[str, list]:
    return json.loads((adapter_dir(arm) / f"{adapter_key(acfg, base, frozenset({k}), seed)}.dense.json")
                      .read_text(encoding="utf-8"))


def macro_from_records(recs: list[dict], pop: str = "P_spec") -> dict:
    from uol.facts_to_law import metrics
    recs = sorted(recs, key=lambda r: r["row"])
    comps = C.component_ids([r["row"] for r in recs])
    macro, per_comp, ordered = metrics.component_macro(recs, comps)
    return {"macro": macro, "per_component": per_comp, "components": ordered, "rows": [r["row"] for r in recs],
            "per_query": recs}


def unseen_recall(res: dict, k: int = 20) -> dict:
    """Pair-level Recall@k on fold-local band targets (round-2 A definition), per band, with component tallies."""
    out = {}
    comps = C.component_ids(res["rows"])
    for band in ("unseen", "rare", "frequent"):
        hit = tot = 0
        by: dict[str, list[int]] = {}
        for r, q, c in zip(res["rows"], res["per_query"], comps):
            lab = C.band_of_row(int(r))
            for p, rk in zip(q["positive_idx"], q["positive_ranks"]):
                if lab[p] == band:
                    tot += 1
                    hit += int(rk <= k)
                    by.setdefault(c, [0, 0])
                    by[c][0] += int(rk <= k)
                    by[c][1] += 1
        out[band] = {"recall": hit / max(tot, 1), "pairs": tot, "by_component": by}
    return out


def recall_contrast(a: dict, b: dict, band: str = "unseen", draws: int = 10_000) -> dict:
    """Component bootstrap of the difference of pooled pair recalls (same components and pairs)."""
    ca, cb = a[band]["by_component"], b[band]["by_component"]
    keys = sorted(set(ca) & set(cb))
    A = np.asarray([ca[x] for x in keys], dtype=np.float64)
    B = np.asarray([cb[x] for x in keys], dtype=np.float64)
    rng = np.random.default_rng(C.SEED)
    diffs = []
    for _ in range(draws):
        i = rng.integers(0, len(keys), len(keys))
        diffs.append(A[i, 0].sum() / A[i, 1].sum() - B[i, 0].sum() / B[i, 1].sum())
    point = A[:, 0].sum() / A[:, 1].sum() - B[:, 0].sum() / B[:, 1].sum()
    return {"delta": float(point), "ci95": [float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))],
            "pairs": int(A[:, 1].sum()), "components": len(keys)}


@lru_cache(maxsize=1)
def frozen_dense() -> dict:
    rows = C.population_rows("P_spec", C.dev_rows())
    M = ep.open_scores("dense_qwen8b")
    res = ep.evaluate(np.asarray(M[rows], dtype=np.float32), rows, "P_spec")
    for rec in res["per_query"]:
        rec.pop("topk", None)
    return res


def dense_alone(arm: str, per_fold: dict[int, dict] | None = None, seed: int = 1) -> dict:
    """Dense-alone dev P_spec readout per configuration and the D3.1 assembly (selection on dense alone), or, with
    ``per_fold``, the assembly of given per-fold configurations."""
    from uol.round3 import arm_runner, stats
    ref = frozen_dense()
    ref_u = unseen_recall(ref)
    configs = grid(arm)

    def assemble(choice: dict[int, dict]) -> dict:
        recs = []
        for k in FOLDS:
            acfg, base = plan(arm, choice[k], k)
            recs += dense_records(arm, acfg, base, k, seed)["P_spec"]
        return macro_from_records(recs)

    def summary(res: dict) -> dict:
        con = stats.contrast(res["per_component"]["ndcg_at_10"], ref["per_component"]["ndcg_at_10"])
        u = unseen_recall(res)
        return {"ndcg_at_10": res["macro"]["ndcg_at_10"], "recall_at_20": res["macro"].get("recall_at_20"),
                "vs_frozen_qwen": {k_: con[k_] for k_ in ("delta", "ci95", "t_ci95", "p_randomization", "class")},
                "unseen_recall_at_20": u["unseen"]["recall"], "unseen_pairs": u["unseen"]["pairs"],
                "rare_recall_at_20": u["rare"]["recall"], "frequent_recall_at_20": u["frequent"]["recall"],
                "unseen_vs_frozen": recall_contrast(u, ref_u, "unseen")}

    if per_fold is not None:
        return summary(assemble(per_fold))
    per_cfg, fold_scores = {}, {}
    for cfg in configs:
        res = assemble({k: cfg for k in FOLDS})
        per_cfg[arm_runner.cfg_key(cfg)] = {"config": cfg, **summary(res)}
        nd = {int(r): q["ndcg_at_10"] for r, q in zip(res["rows"], res["per_query"])}
        fold_scores[arm_runner.cfg_key(cfg)] = _fold_macro(nd)
    choice = {}
    for k in FOLDS:
        other = {key: np.mean([v for f, v in fs.items() if f != k]) for key, fs in fold_scores.items()}
        best = max(other.values())
        choice[k] = next(c for c in configs if other[arm_runner.cfg_key(c)] >= best - 1e-12)
    return {"reference": {"system": "dense_qwen8b (frozen, MaxP)", "ndcg_at_10": ref["macro"]["ndcg_at_10"],
                          "unseen_recall_at_20": ref_u["unseen"]["recall"], "unseen_pairs": ref_u["unseen"]["pairs"],
                          "rare_recall_at_20": ref_u["rare"]["recall"],
                          "frequent_recall_at_20": ref_u["frequent"]["recall"]},
            "by_config": per_cfg, "selected_by_fold": {str(k): c for k, c in choice.items()},
            "selection": "per outer fold, best dense-alone P_spec nDCG@10 on the other four folds (D3.1 rule)",
            "assembled": summary(assemble(choice)), "seed": seed}


def _fold_macro(nd: dict[int, float]) -> dict[int, float]:
    rows = sorted(nd)
    f = fold_array()
    comps = C.component_ids(rows)
    by: dict[int, dict[str, list[float]]] = {}
    for r, c in zip(rows, comps):
        by.setdefault(int(f[r]), {}).setdefault(c, []).append(nd[r])
    return {k: float(np.mean([np.mean(v) for v in d.values()])) for k, d in by.items()}


def smoothed_frozen_dense() -> dict:
    """E3 descriptive: frozen queries against smoothed article vectors (no adapter), dense alone, dev P_spec."""
    from uol.round3 import stats
    ref = frozen_dense()
    rows = C.population_rows("P_spec", C.dev_rows())
    Q = np.asarray(query_matrix()[rows], dtype=np.float32)
    out = {}
    for base in ["raw"] + [f"{c['smooth']}_{c['edges']}" for c in E3_GRID]:
        S = Q @ np.asarray(article_matrix(base), dtype=np.float32).T
        res = ep.evaluate(S, rows, "P_spec")
        con = stats.contrast(res["per_component"]["ndcg_at_10"], ref["per_component"]["ndcg_at_10"])
        out[base] = {"ndcg_at_10": res["macro"]["ndcg_at_10"],
                     "vs_frozen_qwen": {k: con[k] for k in ("delta", "ci95", "class")},
                     "unseen_recall_at_20": unseen_recall(res)["unseen"]["recall"]}
    return out


# ----------------------------------------------------------------------------- arm driver

def band_flags(band: str):
    def fn(rows: Sequence[int]) -> np.ndarray:
        gold = C.gold_rows(rows, "spec")
        return np.asarray([any(C.band_of_row(int(r))[a] == band for a in g) for r, g in zip(rows, gold)])
    return fn


SUBGROUPS = {"unseen": band_flags("unseen"), "rare": band_flags("rare")}


def run(arm: str) -> dict:
    from uol.round3 import arm_runner
    configs = grid(arm)
    dense = dense_alone(arm)
    extra: dict[str, Any] = {"dense_alone": dense, "registered": "DEVIATIONS E0.2 (E2) / E0.3 (E3)",
                             "code_sha256_arm": C.sha256_file(Path(__file__))}
    if arm == "E3":
        extra["adapter_by_fold_from_E2"] = {str(k): v for k, v in e2_selected().items()}
        extra["smoothed_no_adapter_dense"] = smoothed_frozen_dense()
        extra["smoothing"] = {"sgc_k": SGC_K, "appnp_k": APPNP_K, "appnp_alpha": APPNP_ALPHA,
                              "edge_sets": {k: list(v) for k, v in EDGE_SETS.items()}}
    notes = ("E2: low-rank adapters on frozen Qwen3-Embedding-8B vectors (forms qres/proj), strictly cross-fitted hard "
             "negatives; 3 columns (z, log-rank, instrument max z) per pooled pair." if arm == "E2" else
             "E3: E2 adapters (per-fold E2 selection) over G2-smoothed article vectors (SGC K=2 / APPNP K=10 a=.1); "
             "capacity-matched control = E2's assembled arm.")
    return arm_runner.run(ARM_NAME[arm], configs, columns_for_arm(arm), noemb=True, full_view=True,
                          subgroups=SUBGROUPS, standalone={"P_spec": {"dense_alone_assembled": dense["assembled"]}},
                          notes=notes, extra_result=extra, log=log)


def seeds(arm: str, extra_seeds: Sequence[int] = (2, 3)) -> dict:
    """Retrain the per-fold selected configurations at seeds 2 and 3; refit, evaluate and add to RESULT.json."""
    from uol.round3 import arm_runner, stats
    from uol.round3 import features as FT
    path = ARM_DIR[arm] / "RESULT.json"
    res = json.loads(path.read_text(encoding="utf-8"))
    sel = {int(k): {kk: v for kk, v in c.items() if kk != "seed"} for k, c in res["selected_config"].items()}
    for s in extra_seeds:
        train(arm, seeds=(s,), per_fold=sel)
    h = FT.Harness("emb", "spec")
    base = FT.evaluate_dev(h.matrix(arm_runner.baseline(h, log)))
    ckpt = ARM_DIR[arm] / "ckpt"
    per_seed, comp_vals = {}, {}
    for s in (1, *extra_seeds):
        pf = {k: (sel[k] if s == 1 else {**sel[k], "seed": s}) for k in FOLDS}
        preds = arm_runner.refit_configs(h, [], columns_for_arm(arm), ckpt, per_fold_cfg=pf, log=log)
        choice = {k: arm_runner.cfg_key(pf[k]) for k in FOLDS}
        ev = FT.evaluate_dev(h.matrix(arm_runner.assemble(h, preds, choice)))
        con = FT.dev_contrasts(ev, base)
        dense = dense_alone(arm, per_fold=sel, seed=s)
        per_seed[str(s)] = {"vs_EMB": con, "dense_alone_selected_configs": dense}
        comp_vals[s] = np.asarray(ev["P_spec"]["per_component"]["ndcg_at_10"])
        np.save(ARM_DIR[arm] / f"pspec_per_component_seed{s}.npy", comp_vals[s])
        log(f"{arm} seed {s}: Δ vs EMB {con['P_spec']['delta']:+.4f} {con['P_spec']['ci95']} {con['P_spec']['class']}")
    np.save(ARM_DIR[arm] / "pspec_per_component_base.npy", np.asarray(base["P_spec"]["per_component"]["ndcg_at_10"]))
    mean = np.mean([comp_vals[s] for s in comp_vals], axis=0)
    seed_mean = stats.contrast(mean, base["P_spec"]["per_component"]["ndcg_at_10"])
    res["seeds"] = {"per_seed": per_seed, "seed_mean_vs_EMB_P_spec": seed_mean,
                    "rule": "grid at seed 1 (D3.1 per-fold selection); the selected per-fold configurations retrained "
                            "and refit at seeds 2 and 3; primary = seed 1, seed mean reported beside it (E0.2)"}
    C.write_json(path, res)
    return res["seeds"]


def compare() -> dict:
    """E3 vs its capacity-matched control (E2's assembled arm, identical adapters without edges), per seed."""
    from uol.round3 import stats
    out = {}
    for s in SEEDS:
        a, b = ARM_DIR["E3"] / f"pspec_per_component_seed{s}.npy", ARM_DIR["E2"] / f"pspec_per_component_seed{s}.npy"
        if a.exists() and b.exists():
            out[f"seed{s}"] = stats.contrast(np.load(a), np.load(b))
    if all(f"seed{s}" in out for s in SEEDS):
        out["seed_mean"] = stats.contrast(np.mean([np.load(ARM_DIR["E3"] / f"pspec_per_component_seed{s}.npy") for s in SEEDS], 0),
                                          np.mean([np.load(ARM_DIR["E2"] / f"pspec_per_component_seed{s}.npy") for s in SEEDS], 0))
    e3 = json.loads((ARM_DIR["E3"] / "RESULT.json").read_text(encoding="utf-8"))
    e2 = json.loads((ARM_DIR["E2"] / "RESULT.json").read_text(encoding="utf-8"))
    sel2 = {int(k): {kk: v for kk, v in c.items() if kk in ("form", "rank", "tau")} for k, c in e2["selected_config"].items()}
    dense_e2_ctrl = dense_alone("E2", per_fold=sel2)
    out["dense_alone"] = {"E3_assembled": e3["dense_alone"]["assembled"], "E2_same_adapters_no_edges": dense_e2_ctrl}
    e3["vs_E2_control"] = out
    C.write_json(ARM_DIR["E3"] / "RESULT.json", e3)
    return out


def prep() -> None:
    build_raw_articles()
    svd_init("raw")
    for T in sorted({frozenset({i, j, k}) for i in FOLDS for j in FOLDS for k in FOLDS if len({i, j, k}) == 3},
                    key=ekey):
        t0 = time.perf_counter()
        if not dn_path(T).exists():
            dn_ltr(T)
            log(f"dn_ltr T={ekey(T)}: {time.perf_counter() - t0:.0f}s")
    C.write_json(SHARED / "PREP.json", {"utc": C.utc(), "dn_ltr": sorted(p.name for p in DN_DIR.glob("*.npz"))})


def prep_e3() -> None:
    for c in E3_GRID:
        build_smoothed(c["smooth"], c["edges"])
        b = f"{c['smooth']}_{c['edges']}"
        svd_init(b)


def smoke() -> dict:
    """Timing check only: a few steps per form on outer fold 0 (training loss and timing; no dev metric is read)."""
    import torch
    out = {}
    for form in ("qres", "proj"):
        cfg = {"form": form, "rank": 512, "tau": 0.02}
        t0 = time.perf_counter()
        model, meta = train_adapter(cfg, "raw", frozenset({0}), 1, max_epochs=1, max_steps=10)
        out[form] = {"seconds_10_steps_plus_2_val": round(time.perf_counter() - t0, 1),
                     "loss": meta["history"][-1].get("train_loss"), "threads": torch.get_num_threads()}
        log(f"smoke {form}: {out[form]}")
    C.write_json(SHARED / "SMOKE.json", out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prep", "prep-e3", "smoke", "train", "run", "seeds", "compare"])
    ap.add_argument("--arm", default="E2", choices=["E2", "E3"])
    ap.add_argument("--seeds", default="1")
    a = ap.parse_args()
    t0 = time.perf_counter()
    job = f"{a.arm}_{a.command}"
    C.ledger({"kind": "job_start", "job_id": job})
    if a.command == "prep":
        prep()
    elif a.command == "prep-e3":
        prep_e3()
    elif a.command == "smoke":
        smoke()
    elif a.command == "train":
        train(a.arm, seeds=tuple(int(x) for x in a.seeds.split(",")))
    elif a.command == "run":
        run(a.arm)
    elif a.command == "seeds":
        seeds(a.arm)
    else:
        compare()
    C.ledger({"kind": "job_end", "job_id": job, "exit_code": 0, "wall_seconds": round(time.perf_counter() - t0)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
