"""Learned rankers (Blocks C and D): refits of the frozen pipelines through ``round3.features.Harness``.

* Feature table. ``table: "frozen"`` uses the rebuilt frozen pair table of embedding_push (local profile only; it
  holds test-set pairs that are never fitted or scored). ``table: "dev"`` (always in the bundle) rebuilds a
  development-only table from the frozen source score files with ``gap_tests.d58_refit.build_features`` (its
  "control" arm, which reproduced .4081 / .4721 exactly on 19 Oct) and injects it into ``features.context()``.
* Method. ``hgb``: ``Harness.fit_group`` with the frozen per-group hyper-parameters and seeds (the a_p12 recipe);
  ``linear``: ``fusion_ltr.fit_group_linear`` (its own coordinate ascent on the training folds); ``lambdamart``:
  ``gap_tests.ltr_listwise.fit_group`` (LightGBM lambdarank, inner-fold grid as in D57 test 6).
* Extra columns (the ladder). For each system in ``extra`` the three label-free columns of the scale-curve recipe
  (robust z, log rank, instrument-max z; ``round3.e2_adapter.full_columns``) are appended for every development
  pair. Members are fitted on the outer fold first (their own inner-fold choices) and scored on all development rows
  of the pair table; label-free members only, so the a_p12 nesting rule holds.
* ``extension: "m1"`` appends the M1 historical citation-memory columns (``round3.m1_memory.columns_for`` with each
  fold's selected configuration), the "historical extension" of P5.
* ``extension: "columns"`` appends an alternative's saved columns (``footing.arm_columns``, D195).
* Seeds. ``seed_offset`` changes the tree seed (Block G: three seeds); the pair sampling seed stays frozen.

Below-pool candidates keep the frozen L* order (``fusion_ltr.write_pooled``).
"""
from __future__ import annotations

import contextlib
import json
from collections import Counter
from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard
from uol.final.adapters.base import Scorer, SystemUnavailable

_TABLES: dict[str, Any] = {}


def dev_feature_table():
    """The development-only pair table (built once per process; cached on disk in <work>/d58)."""
    if "F" not in _TABLES:
        from uol.embedding_push import base
        prof = context.active()
        d58 = context.redirect_d58(prof)
        ctx = base.Ctx()
        _TABLES["ctx"], _TABLES["F"] = ctx, d58.build_features(ctx, "control")
        _TABLES["dir"] = d58.D
    return _TABLES["ctx"], _TABLES["F"]


def use_table(kind: str) -> str:
    from pathlib import Path
    from uol.round3 import features as FT
    prof = context.active()
    if kind == "frozen" and prof.kind == "local" and prof.release_id == "v11":
        return "frozen"
    ctx, F = dev_feature_table()
    FT._SHARED["ctx"], FT._SHARED["F"] = ctx, F
    # the nested out-of-fold refits (features.nested_ltr: E2 adapter hard negatives, R2 seeds) store pair indices of
    # this table, so their cache sits beside it, one per sources / labels / variant profile (21 Oct 2026)
    if "dir" in _TABLES:
        FT.NESTED_DIR = Path(_TABLES["dir"]) / "nested_ltr"
        prof.patches["uol.round3.features.NESTED_DIR"] = str(FT.NESTED_DIR)
    return "dev"


class _RowSink:
    """``matrix[row] = values`` target for ``fusion_ltr.write_pooled`` that stores only the scored rows."""

    def __init__(self, rows: np.ndarray, n: int):
        self.pos = {int(r): i for i, r in enumerate(rows)}
        self.out = np.empty((len(rows), n), dtype=np.float32)

    def __setitem__(self, row, values):
        self.out[self.pos[int(row)]] = values


@contextlib.contextmanager
def _train_rows_override(rows: np.ndarray | None):
    """DEV-T: training rows that are not a union of outer folds (``fusion_ltr.train_rows_of`` is looked up at call
    time by the Harness; restored afterwards)."""
    from uol.embedding_push import fusion_ltr as fl
    if rows is None:
        yield
        return
    orig = fl.train_rows_of
    fl.train_rows_of = lambda ctx, g, folds=None: np.asarray(rows) if folds is None else orig(ctx, g, folds)
    try:
        yield
    finally:
        fl.train_rows_of = orig


@contextlib.contextmanager
def _pair_table(F, table: str):
    """Point ``e2_adapter.pair_table`` at the injected development table while extra columns are built."""
    from uol.round3 import e2_adapter as e2
    if table == "frozen":
        yield
        return
    orig = e2.pair_table
    e2.pair_table = lambda: (F.pq, F.pc, F.row_ptr)
    try:
        yield
    finally:
        e2.pair_table = orig


class LTRRefit(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        from uol.final import registry
        from uol.round3 import features as FT
        a = self.spec["adapter_args"]
        self.variant = a.get("variant", "emb")
        self.method = a.get("method", "hgb")
        self.seed_offset = int(a.get("seed_offset", 0))
        self.table = use_table(a.get("table", "frozen"))
        self.h = FT.Harness(self.variant, "spec")
        add = [s for s in a.get("add_sources", []) if s in self.h.F.full_ids or any(x.id == s for x in self.h.F.sources)]
        if add:                                   # S0 + X: the encoder's own frozen sources from the same pair table
            from uol.embedding_push import fusion_ltr as fl
            v = self.h.v
            pool = list(dict.fromkeys(v.pool_ids + [s for s in add if s in self.h.F.full_ids]))
            feats = list(dict.fromkeys(v.feature_ids + add))
            self.h.v = fl.make_variant(self.h.F, f"{self.variant}+{'+'.join(add)}", pool, feats)
        self.added_sources = add
        drop = [s for s in a.get("drop_sources", []) if s in self.h.v.feature_ids or s in self.h.v.pool_ids]
        if a.get("drop_sources") and not drop:
            raise ValueError(f"{self.sid}: drop_sources {a['drop_sources']} are not in the {self.variant} features")
        if drop:                                  # leave-one-out (D79): the same refit without these sources' columns
            from uol.embedding_push import fusion_ltr as fl
            v = self.h.v
            self.h.v = fl.make_variant(self.h.F, f"{v.name}-{'-'.join(drop)}", [p for p in v.pool_ids if p not in drop],
                                       [f for f in v.feature_ids if f not in drop])
        self.dropped_sources = drop
        for name in a.get("drop_columns", []):    # addendum 12: a shared column, not a source (only with _base_matrices' refit)
            col = self.h.F.col(name)
            if col not in self.h.v.hgb_cols:
                raise ValueError(f"{self.sid}: column {name} is not in the {self.variant} features")
            self.h.v.hgb_cols = [c for c in self.h.v.hgb_cols if c != col]
        self.extra = [registry.make(x) for x in a.get("extra", [])]
        for m in self.extra:
            m.prepare(pop)
        self.extension = a.get("extension")
        self.extension_key = a.get("extension_key")        # C2 (D93): the extension without its Qwen3-8B key
        self.importance = a.get("importance")              # C2 (D93): grouped permutation importance
        from uol.embedding_push import common as ep
        self.lstar = ep.open_scores("L_star")
        self.N = self.h.ctx.N

    def _group(self, fold):
        if fold.role.startswith("dev_fold"):
            return next(g for g in self.h.groups if g.role == fold.role), None
        base_g = self.h.groups[0]
        cfg = Counter(json.dumps(c, sort_keys=True) for c in self.h.cfgs.values()).most_common(1)[0][0]
        self.h.cfgs["time"] = json.loads(cfg)
        g = replace(base_g, role="time", rows=np.asarray(fold.score_rows), train_rows=np.asarray(fold.train_rows),
                    train_folds=tuple(range(5)))
        return g, np.asarray(fold.train_rows)

    def _extra_columns(self, fold, g) -> tuple[np.ndarray | None, dict]:
        info: dict[str, Any] = {}
        blocks = []
        if self.extra:
            from uol.embedding_push import common as ep
            from uol.round3 import e2_adapter as e2
            dev = guard.dev_rows_only(ep.dev_rows(), "pair-table rows")
            for m in self.extra:
                info[m.sid] = m.fit(fold)
                cols = self.h.empty_columns(3)
                with _pair_table(self.h.F, self.table):
                    for s in range(0, len(dev), 256):
                        rows = dev[s:s + 256]
                        pos, c = e2.full_columns(m.score(rows), rows)
                        cols[pos] = c
                blocks.append(cols)
        if self.extension in ("m1", "m1+columns"):
            if not g.role.startswith("dev_fold"):
                raise SystemUnavailable("M1 columns are defined for the five outer folds only")
            from uol.round3 import common as R3, m1_memory
            res = R3.ARMS / "M1" / "RESULT.json"
            if not res.exists():
                raise SystemUnavailable(f"M1 selection {res} is not available in this profile")
            sel = json.loads(res.read_text(encoding="utf-8"))["selected_config"]
            k = str(int(g.role.replace("dev_fold", "")))
            key = getattr(self, "extension_key", None)
            cfg = {**sel[k], "key": key} if key else sel[k]
            if context.active().variant == "devb":           # D113: recovered DEV-B rows have no cached neighbours
                from uol.final import m1_rows
                m1_cols, m1_info = m1_rows.columns_for(cfg, g, self.h)
                info["m1_rows"] = m1_info
            else:
                m1_cols = m1_memory.columns_for(cfg, g, self.h)
            blocks.append(np.asarray(m1_cols, dtype=np.float32))
            info["m1"] = cfg
        if self.extension in ("columns", "m1+columns"):         # D195: an alternative's columns, by case and candidate
            from uol.final import footing
            arm = self.spec["adapter_args"]["arm"]
            cols, info["arm_columns"] = footing.arm_columns(arm, int(g.role.replace("dev_fold", "")), self.h)
            blocks.append(cols)
        return (np.hstack(blocks).astype(np.float32) if blocks else None), info

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        from uol.embedding_push import fusion_ltr as fl
        g, train_override = self._group(fold)
        extra, info = self._extra_columns(fold, g)
        with _train_rows_override(train_override):
            if self.method == "hgb":
                idx, sc = self.h.fit_group(g, extra, seed_offset=self.seed_offset)
                info["config"] = self.h.cfgs[g.role]
                # read-outs for the thesis (EVIDENCE_FINAL, 23 Oct): columns the model saw, candidate pairs per case
                info["feature_columns"] = int(len(self.h.v.hgb_cols) + (extra.shape[1] if extra is not None else 0))
                info["pool_pairs_per_case"] = round(len(idx) / max(len(g.rows), 1), 2)
                # recall sweep (ANALYSIS_PLAN_C v3): share of the group's specific positives inside the candidate pool
                n_pos = float(np.sum(self.h.F.nspec[np.asarray(g.rows)]))
                info["pool_recall"] = round(float(np.sum(self.h.F.y_spec[idx])) / n_pos, 6) if n_pos else None
                if getattr(self, "importance", None):
                    info["importance"] = self._importance(fold, self.importance)
            elif self.method == "linear":
                if extra is not None:
                    raise SystemUnavailable("the linear ranker uses the frozen columns only")
                m = fl.fit_group_linear(self.h.ctx, self.h.F, self.h.v, g)
                idx, sc, info["config"] = m.pair_idx, m.pair_scores, m.config
            elif self.method == "lambdamart":
                if extra is not None:
                    raise SystemUnavailable("LambdaMART uses the frozen columns only")
                from uol.gap_tests import ltr_listwise as LW
                ev, idx, sc = LW.fit_group(self.h.ctx, self.h.F, self.h.v, g)
                info["config"] = ev["chosen"]
                info["grid"] = ev["grid"]
            elif self.method == "rank_test":                    # research/programs/ranker_tests_20261104/PLAN.md
                from uol.final import ranker_tests as RT
                a = self.spec["adapter_args"]
                ev, idx, sc = RT.fit_group(self.h.ctx, self.h.F, self.h.v, g, extra, a["learner"], a["settings"],
                                            seed_offset=self.seed_offset, train_filter=a.get("train_filter"))
                info.update({"config": ev["chosen"], "grid": ev["grid"], "final_fit_seconds": ev["final_fit_seconds"]})
            else:
                raise ValueError(self.method)
        rows = guard.dev_rows_only(fold.score_rows, "ranker scored rows")
        sink = _RowSink(rows, self.N)
        fl.write_pooled(self.h.ctx, self.h.F, sink, rows, idx, sc, self.lstar)
        self._out = sink
        self.h._base = {}                                       # free the group matrices
        info.update({"table": self.table, "variant": self.variant, "method": self.method,
                     "seed_offset": self.seed_offset, "pairs_scored": int(len(idx))})
        return info

    def column_groups(self) -> dict[str, list[int]]:
        """Column positions (in the fitted matrix: the sources' columns, the shared columns, then the extra blocks)
        of each C2 importance group."""
        from uol.embedding_push import fusion_ltr as fl
        from uol.final import c2
        by_id = {s.id: s for s in self.h.F.sources}
        pos: dict[str, list[int]] = {}
        at = 0
        for sid in self.h.v.feature_ids:
            n = len(fl.source_columns(by_id[sid]))
            pos[sid] = list(range(at, at + n))
            at += n
        at += len(fl.EXTRA_COLUMNS)
        pos["topic"] = list(range(at, at + 3 * len(self.extra)))
        at += 3 * len(self.extra)
        if self.extension == "m1":
            pos["extension"] = list(range(at, at + 3))
        noemb = [s.id for s in self.h.F.sources if s.family == "NO-EMB" and s.id in self.h.v.feature_ids]
        return {name: sorted(i for s in members for i in pos.get(s, []))
                for name, members in c2.importance_groups(noemb).items()}

    def _importance(self, fold, cfg: dict) -> dict[str, Any]:
        """Grouped permutation importance (ANALYSIS_PLAN_C §3): one group's columns permuted jointly across each
        held-out case's pooled candidates; the loss is nDCG@10(base) - nDCG@10(permuted) per case, mean over repeats."""
        from uol.embedding_push import common as ep, fusion_ltr as fl
        from uol.final import measures
        model, X, idx = self.h.last_fit
        rows = guard.dev_rows_only(fold.score_rows, "importance rows")
        prow = self.h.F.pq[idx]

        def ndcg(scores: np.ndarray) -> dict[int, float]:
            sink = _RowSink(rows, self.N)
            fl.write_pooled(self.h.ctx, self.h.F, sink, rows, idx, scores, self.lstar)
            res = ep.evaluate(sink.out, rows, self.pop.eval_population, top_k=measures.TOP_K)
            return {int(r): float(q["ndcg_at_10"]) for r, q in zip(res["rows"], res["per_query"])}

        base = ndcg(model.decision_function(X).astype(np.float32))
        keep = sorted(base)
        out: dict[str, Any] = {"rows": keep, "base_ndcg": [base[r] for r in keep], "repeats": int(cfg["repeats"]),
                               "seed": int(cfg["seed"]), "loss": {}, "columns": {}}
        for name, cols in self.column_groups().items():
            out["columns"][name] = len(cols)
            if not cols:
                continue
            tot = np.zeros(len(keep))
            for rep in range(int(cfg["repeats"])):
                rng = np.random.default_rng(int(cfg["seed"]) + 1000 * int(fold.k) + rep)
                # a random permutation inside each case, whatever the order of the pairs
                same = np.lexsort((np.arange(len(idx)), prow))
                shuffled = np.lexsort((rng.random(len(idx)), prow))
                take = np.empty(len(idx), dtype=np.int64)
                take[same] = shuffled
                Xp = X.copy()
                Xp[:, cols] = X[take][:, cols]
                perm = ndcg(model.decision_function(Xp).astype(np.float32))
                tot += np.asarray([base[r] - perm[r] for r in keep])
            out["loss"][name] = [round(float(x), 6) for x in tot / int(cfg["repeats"])]
        return out

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return self._out.out[[self._out.pos[int(r)] for r in rows]]

    def inputs(self) -> dict[str, Any]:
        from uol.embedding_push import fusion_ltr as fl
        man = fl.pipeline_path(self.h.name).parent / "manifest.json"
        return {"pipeline_manifest": str(man), "table": self.table, "n_pairs": int(self.h.n_pairs),
                "extra": {m.sid: m.inputs() for m in self.extra}, "extension": self.extension}
