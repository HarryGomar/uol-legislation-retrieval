"""Round 3 arm entry into the frozen LTR (prompt §4): the feature route and the fusion route.

**Feature route** (arms that cover every dev row). k extra columns, aligned to the frozen pooled pair table, are
appended to the frozen ``LTR_emb_hgb`` (or ``LTR_noemb_hgb``) feature set. The model is refit per dev group with the
frozen per-group hyper-parameters and the same sampling seeds as ``fusion_ltr.fit_group_hgb`` (the
``uol.round2.a_p12.fit_predict`` recipe), so a refit with no extra columns reproduces the rebuilt pipeline exactly.
The columns for an outer group must be out-of-fold for that group's scoring rows *and* for its training rows; that
is the arm's responsibility, and ``Harness.refit`` asks for them per group.

**Fusion route** (arms that cover only the LTR top K of dev P_spec rows). Inside each row's top K the final score is
z(LTR) + w·z(arm), with both z-scores taken over the K scored candidates; the rest of the ranking is unchanged. w is
chosen from W_GRID on the other four folds (component-macro P_spec nDCG@10; ties go to the smaller w) and applied to
the held-out fold, so the estimate is out-of-fold. ``w_all`` (fit on all dev rows) is what the lockbox uses.

Lockbox rows are never fitted, scored or evaluated here.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.embedding_push import base, common as ep
from uol.embedding_push import fusion_ltr as fl

W_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
PIPELINES = {("emb", "spec"): "LTR_emb_hgb", ("emb", "full"): "LTR_emb_hgb_full",
             ("noemb", "spec"): "LTR_noemb_hgb", ("noemb", "full"): "LTR_noemb_hgb_full"}
CHUNK = 500_000

_SHARED: dict[str, Any] = {}


def context() -> tuple[base.Ctx, fl.Features]:
    """The release context and the memory-mapped rebuilt feature table (loaded once per process)."""
    if "F" not in _SHARED:
        from uol.round2 import a_p12
        ctx = base.Ctx()
        _SHARED["ctx"], _SHARED["F"] = ctx, a_p12.load_features(ctx)
    return _SHARED["ctx"], _SHARED["F"]


def frozen_matrix(name: str) -> np.ndarray:
    return np.load(fl.pipeline_path(name), mmap_mode="r")


def frozen_dev_summary(name: str) -> dict:
    return json.loads((fl.FUSION / "dev_results.json").read_text(encoding="utf-8"))["summaries"][name]


def evaluate_dev(matrix: np.ndarray, populations: Sequence[str] = ("P_spec", "P")) -> dict[str, dict]:
    """Component-macro dev metrics of a full [release rows, N] score matrix (per-query records trimmed)."""
    out = {}
    for pop in populations:
        rows = ep.population_rows(pop, ep.dev_rows())
        res = ep.evaluate(np.asarray(matrix[rows]), rows, pop, top_k=0)
        for rec in res["per_query"]:
            rec.pop("topk", None)
        out[pop] = res
    return out


class Harness:
    """Refits one frozen HGB pipeline per dev group with optional extra feature columns."""

    def __init__(self, variant: str = "emb", view: str = "spec"):
        if (variant, view) not in PIPELINES:
            raise ValueError(f"unknown pipeline {(variant, view)}")
        self.ctx, self.F = context()
        self.variant, self.view = variant, view
        self.name = PIPELINES[(variant, view)]
        self.v = fl.make_variants(self.F)[variant]
        man = json.loads((fl.pipeline_path(self.name).parent / "manifest.json").read_text(encoding="utf-8"))
        self.cfgs = {role: rec["config"] for role, rec in man["configuration"].items()}
        if len(self.v.hgb_cols) != man["n_hgb_features"]:
            raise RuntimeError(f"{self.name}: variant has {len(self.v.hgb_cols)} columns, manifest {man['n_hgb_features']}")
        self.groups = [g for g in self.ctx.groups if not g.is_lockbox]
        self._base: dict[str, tuple] = {}

    # ------------------------------------------------------------------ column helpers

    @property
    def n_pairs(self) -> int:
        return len(self.F.pq)

    def empty_columns(self, k: int) -> np.ndarray:
        return np.full((self.n_pairs, k), np.nan, dtype=np.float32)

    def pair_slice(self, row: int) -> tuple[slice, np.ndarray]:
        """Pair-table positions of one release row and the candidate index of each pair."""
        sl = slice(int(self.F.row_ptr[row]), int(self.F.row_ptr[row + 1]))
        return sl, self.F.pc[sl]

    def fill_from_matrix(self, cols: np.ndarray, j: int, rows: Sequence[int], M: np.ndarray) -> None:
        """cols[pairs of row r, j] = M[i, candidate] for each rows[i]; M is [len(rows), N] (NaN allowed)."""
        for i, r in enumerate(rows):
            sl, pc = self.pair_slice(int(r))
            cols[sl, j] = M[i, pc]

    # ------------------------------------------------------------------ refit

    def _base_matrices(self, g: base.Group) -> tuple:
        if g.role not in self._base:
            seed = fl.SEED + 1000 * g.idx
            rng = np.random.default_rng(seed + 999)
            tr_idx, tr_y = fl.hgb_training_pairs(self.ctx, self.F, self.v, fl.train_rows_of(self.ctx, g), rng, self.view)
            cols = np.asarray(self.v.hgb_cols)
            Xtr = np.asarray(self.F.X[tr_idx[:, None], cols[None, :]])
            idx_out = fl.row_pairs(self.v, g.rows, False, self.F)
            Xout = np.concatenate([np.asarray(self.F.X[idx_out[s:s + CHUNK, None], cols[None, :]])
                                   for s in range(0, len(idx_out), CHUNK)])
            self._base = {g.role: (tr_idx, tr_y, Xtr, idx_out, Xout)}       # keep one group in memory
        return self._base[g.role]

    def fit_group(self, g: base.Group, extra: np.ndarray | None, seed_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(pair indices of the group's scoring rows, decision scores). ``extra`` is [n_pairs, k] or None."""
        from sklearn.ensemble import HistGradientBoostingClassifier

        tr_idx, tr_y, Xtr, idx_out, Xout = self._base_matrices(g)
        if extra is not None:
            if extra.shape[0] != self.n_pairs:
                raise ValueError("extra columns must cover the whole pair table")
            Xtr = np.hstack([Xtr, extra[tr_idx]])
            Xout = np.hstack([Xout, extra[idx_out]])
        cfg = self.cfgs[g.role]
        model = HistGradientBoostingClassifier(learning_rate=cfg["learning_rate"], max_depth=cfg["max_depth"],
                                               random_state=fl.SEED + 1000 * g.idx + seed_offset, **fl.HGB_FIXED)
        model.fit(Xtr, tr_y)
        self.last_fit = (model, Xout, idx_out)             # kept for uol.final's permutation importance (C2, D93)
        return idx_out, model.decision_function(Xout).astype(np.float32)

    def refit(self, columns_for: Callable[[base.Group], np.ndarray | None] | None, *, ckpt_dir: Path | None = None,
              tag: str = "baseline", log: Callable[[str], None] = print) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Refit every dev group; ``columns_for(g)`` returns that group's extra columns (or None for the baseline).

        Per-group predictions are checkpointed as ``{ckpt_dir}/{tag}_{role}.npz`` and reused when present.
        """
        preds = {}
        for g in self.groups:
            ck = None if ckpt_dir is None else Path(ckpt_dir) / f"{tag}_{self.name}_{g.role}.npz"
            if ck is not None and ck.exists():
                z = np.load(ck)
                preds[g.role] = (z["idx"], z["score"])
                continue
            t = time.perf_counter()
            extra = None if columns_for is None else columns_for(g)
            preds[g.role] = self.fit_group(g, extra)
            if ck is not None:
                ck.parent.mkdir(parents=True, exist_ok=True)
                np.savez(ck, idx=preds[g.role][0], score=preds[g.role][1])
            log(f"{self.name} {tag} {g.role}: refit in {time.perf_counter() - t:.0f}s")
        return preds

    def matrix(self, preds: dict[str, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Full [release rows, N] matrix: dev rows from ``preds`` (pooled, L_star below the pool), NaN elsewhere."""
        lstar = ep.open_scores("L_star")
        out = np.full((len(self.ctx.data.queries), self.ctx.N), np.nan, dtype=np.float32)
        for g in self.groups:
            idx, sc = preds[g.role]
            fl.write_pooled(self.ctx, self.F, out, g.rows, idx, sc, lstar)
        return out

    def reproduce(self, ckpt_dir: Path | None = None, tol: float = 0.002) -> dict[str, Any]:
        """Baseline refit vs the rebuilt pipeline's saved dev result; must agree within ``tol`` (a_p12 rule)."""
        res = evaluate_dev(self.matrix(self.refit(None, ckpt_dir=ckpt_dir)))
        pop = "P_spec" if self.view == "spec" else "P"
        saved = frozen_dev_summary(self.name)[pop]["ndcg_at_10"]
        got = res[pop]["macro"]["ndcg_at_10"]
        return {"pipeline": self.name, "population": pop, "saved": saved, "refit": got, "abs_diff": abs(got - saved),
                "pass": abs(got - saved) <= tol, "result": res}


# ----------------------------------------------------------------------------- fusion route

def _z(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def fuse_rows(ltr: np.ndarray, arm: np.ndarray, w: float) -> np.ndarray:
    """Rerank each row's arm-scored set (which must be that row's LTR top K) by z(LTR) + w·z(arm).

    ``ltr`` and ``arm`` are [rows, N]; ``arm`` is NaN outside the scored set. Rows without arm scores are unchanged.
    """
    out = np.array(ltr, dtype=np.float64, copy=True)
    for i in range(len(out)):
        s = np.flatnonzero(np.isfinite(arm[i]))
        if not len(s):
            continue
        k = len(s)
        top = np.argsort(-out[i], kind="stable")[:k]
        if set(top.tolist()) != set(s.tolist()):
            raise ValueError(f"row {i}: the arm-scored set is not the LTR top {k}")
        comb = _z(out[i, s]) + w * _z(np.asarray(arm[i, s], dtype=np.float64))
        out[i, s] = out[i].max() + 1.0 + comb - comb.min()          # the set stays above every unscored candidate
    return out


def ltr_topk(ltr_name: str, rows: Sequence[int], k: int) -> np.ndarray:
    """[len(rows), k] candidate indices of each row's LTR top k, under the evaluator's tie rule (index ascending)."""
    M = frozen_matrix(ltr_name)
    return np.stack([np.argsort(-np.asarray(M[int(r)], dtype=np.float64), kind="stable")[:k] for r in rows])


def fusion_route(ltr_name: str, arm_rows: np.ndarray, arm_scores: np.ndarray, *, grid: Sequence[float] = W_GRID,
                 population: str = "P_spec", ltr_matrix: np.ndarray | None = None) -> dict[str, Any]:
    """Cross-fitted fusion weight. ``arm_rows`` are dev release rows, ``arm_scores`` [len(arm_rows), N].

    Returns the out-of-fold fused matrix rows (aligned to ``arm_rows``), the w chosen per held-out fold, the per-w
    dev nDCG@10 by fold, and ``w_all`` fitted on all dev rows (for the lockbox).
    """
    arm_rows = np.asarray(arm_rows, dtype=np.int64)
    if np.isin(arm_rows, ep.lockbox_rows()).any():
        raise PermissionError("the fusion route is fitted on dev rows only")
    ltr = (np.asarray(frozen_matrix(ltr_name)[arm_rows], dtype=np.float64) if ltr_matrix is None
           else np.asarray(ltr_matrix, dtype=np.float64))                     # rows aligned to arm_rows
    folds = np.asarray([int(str(ep.roles()[r]).replace("dev_fold", "")) for r in arm_rows])
    fused = {w: fuse_rows(ltr, arm_scores, w) for w in grid}
    pop_rows = set(ep.population_rows(population, arm_rows).tolist())
    keep = np.asarray([int(r) in pop_rows for r in arm_rows])

    def score(w: float, mask: np.ndarray) -> float:
        m = mask & keep
        return ep.evaluate(fused[w][m], arm_rows[m], population)["macro"]["ndcg_at_10"]

    table = {f: {w: score(w, folds != f) for w in grid} for f in sorted(set(folds.tolist()))}
    chosen = {f: _pick(table[f], grid) for f in table}
    out = np.empty_like(ltr)
    for f, w in chosen.items():
        out[folds == f] = fused[w][folds == f]
    all_scores = {w: score(w, np.ones(len(arm_rows), dtype=bool)) for w in grid}
    return {"oof_matrix": out.astype(np.float32), "rows": arm_rows, "w_by_fold": chosen,
            "inner_ndcg_by_fold": table, "dev_ndcg_by_w": all_scores, "w_all": _pick(all_scores, grid)}


def _pick(scores: dict[float, float], grid: Sequence[float]) -> float:
    best = max(scores.values())
    return min(w for w, s in scores.items() if s >= best - 1e-12)


# ----------------------------------------------------------------------------- contrasts for RESULT.json

def dev_contrasts(arm: dict[str, dict], ref: dict[str, dict], family_size: int = 1) -> dict[str, dict]:
    """Per population: ``stats.contrast`` of arm - ref over aligned components (``evaluate_dev`` outputs)."""
    from uol.round3 import stats

    out = {}
    for pop in arm:
        if pop not in ref:
            continue
        a, b = arm[pop], ref[pop]
        if a["components"] != b["components"]:
            raise ValueError(f"{pop}: components are not aligned")
        out[pop] = {**stats.contrast(a["per_component"]["ndcg_at_10"], b["per_component"]["ndcg_at_10"],
                                     family_size=family_size),
                    "arm_ndcg_at_10": a["macro"]["ndcg_at_10"], "ref_ndcg_at_10": b["macro"]["ndcg_at_10"]}
    return out


def _main() -> int:
    import argparse
    import sys

    from uol.round3 import common as C

    ap = argparse.ArgumentParser(description="Refit-harness reproduction check (dev only)")
    ap.add_argument("command", choices=["reproduce"])
    ap.add_argument("--variant", default="emb", choices=["emb", "noemb"])
    ap.add_argument("--view", default="spec", choices=["spec", "full"])
    a = ap.parse_args()
    h = Harness(a.variant, a.view)
    job = f"phase0_harness_reproduce_{h.name}"
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": job, "command": " ".join(sys.argv)})
    rep = h.reproduce(ckpt_dir=C.RUN / "phase0/harness" / h.name)
    rep.pop("result")
    C.write_json(C.RUN / "phase0/harness" / f"reproduce_{h.name}.json", {**rep, "utc": C.utc()})
    C.ledger({"kind": "job_end", "job_id": job, "start_utc": start, "end_utc": C.utc(), "exit_code": 0 if rep["pass"] else 2,
              "result": rep})
    print(json.dumps(rep, indent=1))
    return 0 if rep["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(_main())


# ----------------------------------------------------------------------------- nested out-of-fold LTR seeds

NESTED_DIR = __import__("uol.round3.common", fromlist=["RUN"]).RUN / "phase2" / "nested_ltr"


def nested_ltr(k: int, j: int, variant: str = "emb") -> tuple[np.ndarray, np.ndarray]:
    """Pooled LTR scores for the fold-j rows from a model trained on the dev folds other than j and k.

    The frozen hyper-parameters of outer group k and its seeds are used. For j == k this is the ordinary out-of-fold
    refit (train on the four folds other than k). Cached in phase2/nested_ltr. Use these as first-pass seeds for the
    training rows (fold j) of outer group k so that no fold-k label reaches group k's features (a_p12 nesting rule).
    """
    from dataclasses import replace

    from uol.round3 import common as C

    path = NESTED_DIR / f"{variant}_k{k}_j{j}.npz"
    if path.exists():
        z = np.load(path)
        return z["idx"], z["score"]
    h = Harness(variant, "spec")
    g = next(x for x in h.groups if x.role == f"dev_fold{k}")
    folds = tuple(f for f in range(5) if f not in (j, k))
    rows_j = h.ctx.dev[h.ctx.fold[h.ctx.dev] == j]
    gg = replace(g, rows=rows_j, train_rows=h.ctx.dev[np.isin(h.ctx.fold[h.ctx.dev], folds)], train_folds=folds)
    with C.cpu_slot():
        idx, sc = h.fit_group(gg, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, idx=idx, score=sc)
    return idx, sc


def seed_scores(k: int, rows: np.ndarray, variant: str = "emb") -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{row: (candidate indices, scores)} over the pooled set: fold-k rows from the frozen out-of-fold pipeline,
    fold-j rows (j != k) from ``nested_ltr(k, j)``."""
    ctx, F = context()
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    fold = ctx.fold
    M = frozen_matrix(PIPELINES[(variant, "spec")])
    for j in sorted(set(int(fold[r]) for r in rows)):
        rj = [int(r) for r in rows if fold[r] == j]
        if j == k:
            for r in rj:
                sl = slice(int(F.row_ptr[r]), int(F.row_ptr[r + 1]))
                pc = F.pc[sl]
                out[r] = (pc, np.asarray(M[r], dtype=np.float32)[pc])
            continue
        idx, sc = nested_ltr(k, j, variant)
        q = F.pq[idx]
        order = np.argsort(q, kind="stable")
        idx, sc, q = idx[order], sc[order], q[order]
        for r in rj:
            a, b = np.searchsorted(q, r), np.searchsorted(q, r, side="right")
            out[r] = (F.pc[idx[a:b]], sc[a:b])
    return out
