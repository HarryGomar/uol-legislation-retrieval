"""Per-system caches on the dev P_spec rows, built from the saved score files with the project's evaluator.

Conventions are those of ``uol.embedding_push.common.evaluate``: population P_spec, condensed specific lists
(generic positives dropped, generic candidates removed from the ranked list), candidate-ID tie order, exact ranks,
component-macro means. Only dev rows are ever indexed: score matrices are memory-mapped and sliced by dev row, and
labels are read through ``specific_idx`` for those rows only. Lockbox and prospective rows are never touched.

Per system the cache holds, for each dev P_spec row:
* ``pos_rank``: exact condensed rank of every specific positive (flat, aligned to ``Meta.pos_flat``);
* ``pos_inst_rank``: the positive's rank among the condensed candidates of its own instrument;
* ``top_idx`` / ``top_score``: the condensed top ``TOPN`` (catalogue indices, best first) and their scores;
* ``inst_max``: per instrument, the maximum condensed score (law-level aggregation by max);
* ``ndcg``: the per-query nDCG@10 from ``metrics.evaluate_matrix`` (for the acceptance check).

    python -m uol.eval_lenses.core build [names...]
    python -m uol.eval_lenses.core accept
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.facts_to_law import metrics

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/runs/eval_lenses_20261015"
CACHE = OUT / "cache"
ROUND3 = ROOT / "experiments/runs/round3_best_case_20261013"
GRAPH = ROUND3 / "phase2/statute_graph/graph.npz"
QWEN = ROOT / "data/derived/embedding_push_qwen8b_v1"
TOPN = 1000
POP = "P_spec"

# name -> (kind, source). kind: "score" (scores/<id>), "pipeline" (fusion/pipelines/<id>),
# "pspec" (a [dev P_spec rows, N] matrix saved by a Round 3 arm), "c_hybrid" (rebuilt below, no refit).
SYSTEMS: dict[str, tuple[str, str]] = {
    "popularity": ("score", "prior"),
    "L_star": ("score", "L_star"),
    "dense_e5": ("score", "dense_e5_maxp"),
    "dense_bge": ("score", "dense_bge_maxp"),
    "dense_qwen8b": ("score", "dense_qwen8b"),
    "ft_e5_q": ("score", "ft_e5_q"),
    "K_lex": ("score", "K_lex"),
    "knn_e5": ("score", "knn_e5"),
    "knn_bge": ("score", "knn_bge"),
    "knn_qwen8b": ("score", "knn_qwen8b"),
    "proto_e5": ("score", "proto_e5"),
    "proto_bge": ("score", "proto_bge"),
    "proto_qwen8b": ("score", "proto_qwen8b"),
    "K_qwen8b": ("score", "K_qwen8b"),
    "LTR_noemb_hgb": ("pipeline", "LTR_noemb_hgb"),
    "LTR_emb_hgb": ("pipeline", "LTR_emb_hgb"),
    "C_stage2": ("pspec", "arms/C_hybrid/stage2_pspec_matrix.npy"),
    "C_hybrid": ("c_hybrid", "arms/C_hybrid"),
}
LADDER = ("popularity", "L_star", "dense_e5", "dense_bge", "dense_qwen8b", "ft_e5_q", "K_lex", "knn_qwen8b",
          "proto_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid")
LABELS = {"popularity": "Popularity prior", "L_star": "Tuned lexical (L*)", "dense_e5": "Dense e5-small",
          "dense_bge": "Dense bge-m3", "dense_qwen8b": "Dense Qwen3-8B", "ft_e5_q": "e5-small fine-tuned",
          "K_lex": "Lexical citation memory", "knn_e5": "kNN memory (e5)", "knn_bge": "kNN memory (bge)",
          "knn_qwen8b": "kNN memory (Qwen)", "proto_e5": "Prototypes (e5)", "proto_bge": "Prototypes (bge)",
          "proto_qwen8b": "Prototypes (Qwen)", "K_qwen8b": "Memory + dense (Qwen)",
          "LTR_noemb_hgb": "Ranker without embeddings", "LTR_emb_hgb": "Ranker with embeddings",
          "C_stage2": "Ranker + memory + LLM features", "C_hybrid": "+ LLM reranking (hybrid)"}
OFFICIAL = {"L_star": 0.184, "dense_qwen8b": 0.274, "LTR_noemb_hgb": 0.408, "LTR_emb_hgb": 0.4721}
TOLERANCE = 0.003


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ----------------------------------------------------------------------------- dev rows, labels, catalogue structure

@dataclass
class Meta:
    rows: np.ndarray            # dev P_spec release rows (sorted)
    fold: np.ndarray            # outer dev fold of each row
    component: np.ndarray       # component id of each row
    pos_off: np.ndarray         # row i's positives are pos_flat[pos_off[i]:pos_off[i+1]]
    pos_flat: np.ndarray        # specific positive catalogue indices, in specific_idx order
    n_generic_pos: np.ndarray   # generic positives dropped from each row (full minus specific)
    inst: np.ndarray            # instrument code of each catalogue article
    instruments: list[str]      # instrument URL per code
    generic: np.ndarray         # bool per catalogue article: generic (removed from condensed lists)
    tie: np.ndarray             # candidate-ID tie order

    def positives(self, i: int) -> np.ndarray:
        return self.pos_flat[self.pos_off[i]:self.pos_off[i + 1]]

    @property
    def n(self) -> int:
        return len(self.rows)


@lru_cache(maxsize=1)
def meta() -> Meta:
    d = ep.release()
    rows = ep.population_rows(POP, ep.dev_rows())
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("dev helper returned a lockbox row")
    roles = ep.roles()
    fold = np.asarray([int(str(roles[r]).replace("dev_fold", "")) for r in rows])
    comp = np.asarray([d.queries[int(r)].component_id for r in rows])
    pos = [np.asarray(d.specific_idx[int(r)], dtype=np.int64) for r in rows]
    off = np.concatenate([[0], np.cumsum([len(p) for p in pos])]).astype(np.int64)
    ngen = np.asarray([len(d.positive_idx[int(r)]) - len(d.specific_idx[int(r)]) for r in rows])
    inst_url = [c.split("#", 1)[0] for c in d.candidate_ids]
    instruments = sorted(set(inst_url))
    code = {u: i for i, u in enumerate(instruments)}
    inst = np.asarray([code[u] for u in inst_url], dtype=np.int32)
    generic = np.zeros(len(d.candidate_ids), dtype=bool)
    generic[d.generic_idx] = True
    return Meta(rows, fold, comp, off, np.concatenate(pos), ngen, inst, instruments, generic, ep.tie_order())


# ----------------------------------------------------------------------------- score loading (dev rows only)

def _c_hybrid_rows() -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the C_hybrid out-of-fold scores from saved Stage-2 scores and saved Luna orders (no refit, no calls).

    Same arithmetic as ``uol.round3.c_hybrid.evaluate``: arm score = 1 / mean position over the 3 samples x 2 passes,
    fused per row by ``features.fuse_rows`` with the cross-fitted weight recorded per fold in RESULT.json.
    """
    from uol.round3 import c_hybrid, features
    res = json.loads((c_hybrid.OUT / "RESULT.json").read_text(encoding="utf-8"))
    w_by_fold = {int(k): float(v) for k, v in res["stages"]["w_by_fold"].items()}
    rows, A = c_hybrid.arm_matrix(int(res["stages"]["samples"]))
    S2 = np.load(c_hybrid.OUT / "stage2_pspec_matrix.npy", mmap_mode="r")
    roles = ep.roles()
    out = np.empty((len(rows), A.shape[1]), dtype=np.float32)
    for i, r in enumerate(rows):
        w = w_by_fold[int(str(roles[int(r)]).replace("dev_fold", ""))]
        out[i] = features.fuse_rows(np.asarray(S2[i:i + 1], dtype=np.float64), A[i:i + 1], w)[0].astype(np.float32)
    del A
    return np.asarray(rows), out


def load_dev_scores(name: str) -> np.ndarray:
    """[len(meta().rows), N] float32 scores of a system on the dev P_spec rows."""
    m = meta()
    kind, src = SYSTEMS[name]
    if kind in ("score", "pipeline"):
        path = ep.score_path(src) if kind == "score" else ep.RUN / "fusion/pipelines" / src / "scores.npy"
        mm = np.load(path, mmap_mode="r")
        return np.asarray(mm[m.rows], dtype=np.float32)
    if kind == "pspec":
        rows = np.load(ROUND3 / src.replace("matrix", "rows"))
        mm = np.load(ROUND3 / src, mmap_mode="r")
        if not np.array_equal(rows, m.rows):
            raise ValueError(f"{name}: saved rows differ from the dev P_spec rows")
        return np.asarray(mm, dtype=np.float32)
    if kind == "c_hybrid":
        rows, M = _c_hybrid_rows()
        if not np.array_equal(rows, m.rows):
            raise ValueError("C_hybrid rows differ from the dev P_spec rows")
        return M
    raise ValueError(kind)


# ----------------------------------------------------------------------------- cache build

def cache_path(name: str) -> Path:
    return CACHE / f"{name}.npz"


def build(name: str, force: bool = False) -> Path:
    path = cache_path(name)
    if path.exists() and not force:
        return path
    m = meta()
    t0 = time.time()
    S = load_dev_scores(name)
    keep = np.flatnonzero(~m.generic)
    kinst = m.inst[keep]
    ktie = m.tie[keep]
    n_inst = len(m.instruments)
    # instrument blocks of the condensed list for the max-aggregation
    order = np.argsort(kinst, kind="stable")
    starts = np.flatnonzero(np.r_[True, np.diff(kinst[order]) != 0])
    present = kinst[order][starts]
    positives = [m.positives(i).tolist() for i in range(m.n)]
    res = metrics.evaluate_matrix(S, positives, exclude_idx=m.generic, top_k=TOPN, tie_order=m.tie)
    pos_rank = np.concatenate([np.asarray(r["positive_ranks"], dtype=np.int32) for r in res])
    if len(pos_rank) != len(m.pos_flat):
        raise ValueError("a specific positive was dropped by the condensed list")
    top_idx = np.full((m.n, TOPN), -1, dtype=np.int32)
    top_score = np.full((m.n, TOPN), np.nan, dtype=np.float32)
    inst_max = np.full((m.n, n_inst), -np.inf, dtype=np.float32)
    inst_rank = np.empty(len(m.pos_flat), dtype=np.int32)
    for i in range(m.n):
        t = np.asarray(res[i]["topk"], dtype=np.int32)
        top_idx[i, :len(t)] = t
        top_score[i, :len(t)] = S[i, t]
        row = S[i, keep].astype(np.float64)
        inst_max[i, present] = np.maximum.reduceat(row[order], starts).astype(np.float32)
        for j, p in enumerate(m.positives(i)):
            same = kinst == m.inst[p]
            sp = float(S[i, p])
            inst_rank[m.pos_off[i] + j] = 1 + np.count_nonzero(same & (row > sp)) + \
                np.count_nonzero(same & (row == sp) & (ktie < m.tie[p]))
    ndcg = np.asarray([r["ndcg_at_10"] for r in res], dtype=np.float64)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, pos_rank=pos_rank, pos_inst_rank=inst_rank, top_idx=top_idx, top_score=top_score,
                        inst_max=inst_max, ndcg=ndcg)
    del S, res
    gc.collect()
    log(f"cache {name}: {time.time() - t0:.0f}s, nDCG@10 macro {component_macro(ndcg):.4f}")
    return path


@lru_cache(maxsize=32)
def load(name: str) -> dict[str, np.ndarray]:
    z = np.load(cache_path(name))
    return {k: z[k] for k in z.files}


# ----------------------------------------------------------------------------- component-macro helpers

def component_macro(values: Sequence[float], mask: np.ndarray | None = None) -> float:
    """Queries averaged within components, then components (over rows where ``mask``)."""
    return float(component_values(values, mask)[1].mean())


def component_values(values: Sequence[float], mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    m = meta()
    v = np.asarray(values, dtype=np.float64)
    comp = m.component
    if mask is not None:
        v, comp = v[mask], comp[mask]
    ids, codes = np.unique(comp, return_inverse=True)
    sums = np.bincount(codes, weights=v, minlength=len(ids))
    counts = np.bincount(codes, minlength=len(ids))
    return ids, sums / counts


def per_query_first_rank(name: str) -> np.ndarray:
    m, c = meta(), load(name)
    return np.asarray([c["pos_rank"][m.pos_off[i]:m.pos_off[i + 1]].min() for i in range(m.n)])


# ----------------------------------------------------------------------------- acceptance (step 0)

def accept() -> dict[str, Any]:
    out = {}
    for name, target in OFFICIAL.items():
        build(name)
        val = component_macro(load(name)["ndcg"])
        out[name] = {"reproduced": val, "official": target, "abs_diff": abs(val - target),
                     "pass": abs(val - target) <= TOLERANCE}
    m = meta()
    summary = {"population": POP, "dev_rows": int(m.n), "components": int(len(np.unique(m.component))),
               "positives": int(len(m.pos_flat)), "systems": out, "all_pass": all(v["pass"] for v in out.values()),
               "tolerance": TOLERANCE}
    write_json(OUT / "step0_acceptance.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "accept"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "accept":
        s = accept()
        print(json.dumps(s, indent=1))
        return 0 if s["all_pass"] else 1
    for name in a.names or list(SYSTEMS):
        build(name, force=a.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
