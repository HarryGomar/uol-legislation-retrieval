"""Can the ranker be improved at the same running cost? (the author, 4 Nov 2026; research/programs/ranker_tests_20261104/PLAN.md)

Listwise learners on the ranker's eight sources without the topic scores, with or without the votes of the older tesis,
fitted through the record's own harness (``adapters.ltr``, method ``rank_test``). Settings are chosen inside each outer
fold on its first two training folds, as ``gap_tests.ltr_listwise`` does. Development cases only; run ids are new.

    py -3.12 -m uol.final.ranker_tests fit --id rt_check --go
    py -3.12 -m uol.final.ranker_tests score
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

SEED = 20261104
N_SETTINGS = 24
LM_THREADS = 8
XGB_THREADS = 12

LM_SPACE = {"learning_rate": (0.02, 0.03, 0.05, 0.08, 0.1, 0.15), "num_leaves": (7, 15, 31, 63, 127), "n_estimators": (150, 300, 600, 1000),
            "min_child_samples": (10, 20, 40, 80, 160), "colsample_bytree": (0.3, 0.5, 0.7, 1.0), "reg_lambda": (0.0, 1.0, 5.0, 20.0),
            "subsample": (0.7, 1.0), "lambdarank_truncation_level": (10, 20, 30), "n_neg": (100, 300)}
XGB_SPACE = {"learning_rate": (0.02, 0.03, 0.05, 0.08, 0.1, 0.15), "max_leaves": (7, 15, 31, 63, 127), "n_estimators": (150, 300, 600, 1000),
             "min_child_weight": (0.1, 1.0, 5.0, 20.0), "colsample_bytree": (0.3, 0.5, 0.7, 1.0), "reg_lambda": (0.0, 1.0, 5.0, 20.0),
             "subsample": (0.7, 1.0), "lambdarank_num_pair_per_sample": (10, 20, 30), "n_neg": (100, 300)}
XGB_GRID = [{"learning_rate": lr, "max_leaves": nl} for lr in (0.05, 0.1) for nl in (15, 63)]


def drawn(space: dict[str, tuple], first: dict[str, Any], key: int) -> list[dict[str, Any]]:
    """``N_SETTINGS`` distinct settings, the stored default first."""
    rng = np.random.default_rng(SEED + key)
    out, seen = [first], {json.dumps(first, sort_keys=True)}
    while len(out) < N_SETTINGS:
        c = {k: (v[int(rng.integers(len(v)))]) for k, v in space.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        s = json.dumps(c, sort_keys=True)
        if s not in seen:
            seen.add(s)
            out.append(c)
    return out


def settings(learner: str, kind: str) -> list[dict[str, Any]]:
    from uol.gap_tests import ltr_listwise as LW
    if learner == "lgbm":
        if kind == "stored":
            return list(LW.GRID)
        return drawn(LM_SPACE, {"learning_rate": 0.05, "num_leaves": 63, "n_estimators": 300, "min_child_samples": 40, "colsample_bytree": 0.5,
                                "reg_lambda": 1.0, "subsample": 1.0, "lambdarank_truncation_level": 30, "n_neg": 300}, 1)
    if kind == "stored":
        return list(XGB_GRID)
    return drawn(XGB_SPACE, {"learning_rate": 0.05, "max_leaves": 63, "n_estimators": 300, "min_child_weight": 1.0, "colsample_bytree": 0.5,
                             "reg_lambda": 1.0, "subsample": 1.0, "lambdarank_num_pair_per_sample": 20, "n_neg": 300}, 2)


def _matrix(F, v, idx, extra) -> np.ndarray:
    from uol.gap_tests import ltr_listwise as LW
    X = LW.matrix(F, v, idx)
    return X if extra is None else np.hstack([X, np.asarray(extra[idx], dtype=np.float32)])


def _fit(learner: str, cfg: dict[str, Any], X, y, counts, seed: int):
    cfg = {k: v for k, v in cfg.items() if k != "n_neg"}
    if learner == "lgbm":
        import lightgbm as lgb
        from uol.gap_tests import ltr_listwise as LW
        params = {**LW.FIXED, "n_jobs": LM_THREADS, **cfg, "random_state": seed}
        if params.get("subsample", 1.0) < 1.0:
            params["subsample_freq"] = 1
        m = lgb.LGBMRanker(**params)
        m.fit(X, y, group=counts)
        return m
    import xgboost as xgb
    params = {"objective": "rank:ndcg", "tree_method": "hist", "grow_policy": "lossguide", "max_depth": 0, "max_bin": 127, "n_estimators": 300,
              "min_child_weight": 1.0, "colsample_bytree": 0.5, "reg_lambda": 1.0, "lambdarank_pair_method": "topk", "lambdarank_num_pair_per_sample": 20,
              "n_jobs": XGB_THREADS, "verbosity": 0, **cfg, "random_state": seed}
    m = xgb.XGBRanker(**params)
    m.fit(X, y, qid=np.repeat(np.arange(len(counts)), counts))
    return m


def _predict(model, F, v, idx, extra, chunk: int = 400_000) -> np.ndarray:
    return np.concatenate([model.predict(_matrix(F, v, idx[s:s + chunk], extra)) for s in range(0, len(idx), chunk)]).astype(np.float32)


def kept_rows(ctx, rows: np.ndarray, train_filter: str | None) -> np.ndarray:
    """H042 (research/programs/rank_rerun_20261105/PLAN.md): training rows without the cases whose facts named a provision
    (``no_masked``) or a cited article (``no_answer_named``). Held-out cases are never touched."""
    if not train_filter:
        return rows
    from uol.embedding_push import common as ep
    root = Path(__file__).resolve().parents[3]
    d = ep.release()
    with open(root / "data" / "releases" / "uol_c01_statutes_20261004_v11_stratified" / "queries.jsonl", encoding="utf-8") as f:
        info = {q["query_id"]: q for q in map(json.loads, f)}
    def keep(r: int) -> bool:
        q = info[d.queries[int(r)].query_id]
        if train_filter == "no_masked":
            return not q["stratum"].startswith("S2")
        return not q["flags"]["answer_named_in_facts"]
    return np.asarray([r for r in rows if keep(r)], dtype=rows.dtype)


def fit_group(ctx, F, v, g, extra, learner: str, kind: str, log=print, seed_offset: int = 0, train_filter: str | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """As ``ltr_listwise.fit_group``, with extra columns, either learner and any list of settings. ``seed_offset`` moves the
    tree seed only, the sampling of pairs stays frozen."""
    from uol.embedding_push import base, fusion_ltr as fl
    from uol.gap_tests import ltr_listwise as LW
    seed = fl.SEED + 1000 * g.idx
    grid = settings(learner, kind)
    results = []
    for n, cfg in enumerate(grid):
        rows_all, vals_all, t0 = [], [], time.time()
        for inner in g.train_folds[:fl.INNER_FOLDS]:
            rng = np.random.default_rng(seed + 17 * inner)
            tr = kept_rows(ctx, fl.train_rows_of(ctx, g, [f for f in g.train_folds if f != inner]), train_filter)
            idx, y, counts = LW.training_lists(ctx, F, v, tr, rng, n_neg=cfg.get("n_neg", LW.N_NEG))
            model = _fit(learner, cfg, _matrix(F, v, idx, extra), y, counts, seed + seed_offset)
            va = fl.spec_rows_of(ctx, [inner])
            va_idx = fl.row_pairs(v, va, True, F)
            pad = fl.pad_pairs(F, va_idx)
            S = np.full((len(pad.rows), pad.Pmax), -np.inf, dtype=np.float32)
            S[pad.seg, pad.pos] = _predict(model, F, v, va_idx, extra)
            nd = dict(zip(pad.rows.tolist(), fl.ndcg_padded(S, pad).tolist()))
            rows_all.append(va)
            vals_all.append(np.asarray([nd.get(int(r), 0.0) for r in va]))
        obj = fl.macro_rows(ctx, np.concatenate(rows_all), np.concatenate(vals_all))
        results.append({"config": cfg, "objective": obj, "seconds": round(time.time() - t0, 1)})
        log(f"{g.role} setting {n + 1}/{len(grid)} objective {obj:.4f} ({time.time() - t0:.0f} s) {cfg}")
    best = base.argbest([r["objective"] for r in results]) if len(results) > 1 else 0
    cfg = grid[best]
    rng = np.random.default_rng(seed + 999)
    tr_all = kept_rows(ctx, fl.train_rows_of(ctx, g), train_filter)
    log(f"{g.role} training cases {len(tr_all)}")
    idx, y, counts = LW.training_lists(ctx, F, v, tr_all, rng, n_neg=cfg.get("n_neg", LW.N_NEG))
    t0 = time.time()
    model = _fit(learner, cfg, _matrix(F, v, idx, extra), y, counts, seed + seed_offset)
    fit_seconds = round(time.time() - t0, 1)
    out_idx = fl.row_pairs(v, g.rows, False, F)
    return {"grid": results, "chosen": cfg, "final_train_pairs": int(len(idx)), "final_fit_seconds": fit_seconds}, out_idx, _predict(model, F, v, out_idx, extra)


ARMS = {"rt_check": ("lgbm", "stored", False), "rt_lm_tesis": ("lgbm", "stored", True), "rt_xgb": ("xgb", "stored", False), "rt_xgb_tesis": ("xgb", "stored", True),
        "rt_lm_wide": ("lgbm", "wide", False), "rt_lm_wide_tesis": ("lgbm", "wide", True), "rt_xgb_wide_tesis": ("xgb", "wide", True)}
FAMILY = [a for a in ARMS if a != "rt_check"]
BASELINE = "ft_lean"


def specs() -> list[dict]:
    from uol.final import lean_rerun as LR
    base = LR.A._systems()["S0+qwen3_8b"]
    out = []
    for sid, (learner, kind, tesis) in ARMS.items():
        s = copy.deepcopy(base)
        s["id"], s["notes"] = sid, f"ranker tests (4 Nov 2026): {learner}, {kind} settings, {'with' if tesis else 'without'} the older tesis votes"
        s["adapter_args"].update({"drop_sources": list(LR.FT.LEAN_DROP), "extra": [], "method": "rank_test", "learner": learner, "settings": kind})
        if tesis:
            s["adapter_args"]["extension"] = "m1"
        s.pop("answer", None)
        out.append(s)
    return out


def fit(sid: str) -> int:
    import os
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0")
    from uol.final import harness, lean_rerun as LR
    LR.A.install_systems(specs(), dry=False)
    LR.D.activate(LR.STORE)
    LR._overlay_stat_glob()
    t0 = time.perf_counter()
    s = harness.run(sid, "DEV", store=LR.STORE)
    print(json.dumps({"system": sid, "status": s.get("status"), "ndcg_at_10": s.get("ndcg_at_10"), "n": s.get("n_cases"), "seconds": round(time.perf_counter() - t0),
                      "reason": s.get("reason")}), flush=True)
    return 0 if s.get("status") == "done" else 1


def score() -> int:
    import pandas as pd
    from uol.final import lean_rerun as LR
    A, runs = LR.A, Path(LR.STORE) / "runs"
    frame = lambda s: pd.read_parquet(runs / s / "DEV" / "per_case.parquet")
    summ = lambda s: json.loads((runs / s / "DEV" / "summary.json").read_text(encoding="utf-8"))
    have = [a for a in FAMILY if (runs / a / "DEV" / "per_case.parquet").exists()]
    if len(have) != len(FAMILY):
        raise SystemExit(f"missing runs: {sorted(set(FAMILY) - set(have))}")
    base = frame(BASELINE)
    fam = A.dev_family("ranker_tests", [(a, frame(a), BASELINE, base) for a in FAMILY])
    beside = {b: A.dev_family(f"ranker_tests_vs_{b}", [(a, frame(a), b, frame(b)) for a in FAMILY]) for b in ("lf_tesis", "lf_listwise", "S_full_ext")}
    res = {"plan": "research/programs/ranker_tests_20261104/PLAN.md", "baseline": BASELINE, "family": fam,
           "beside": {b: x["contrasts"] for b, x in beside.items()},
           "summaries": {s: {k: summ(s).get(k) for k in ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_100", "unseen_recall_at_20", "success_at_1", "success_at_10")}
                         for s in [BASELINE, "lf_plain", "lf_listwise", "lf_tesis", "S_full_ext", *ARMS]},
           "check": {"rt_check": summ("rt_check")["ndcg_at_10"], "lf_listwise": summ("lf_listwise")["ndcg_at_10"]}}
    out = Path(A.ROOT) / "research" / "programs" / "ranker_tests_20261104" / "results.json"
    A.write_json(out, res)
    print(json.dumps(res["check"]), flush=True)
    for k, c in fam["contrasts"].items():
        print(f"{k:32s} {c['mean_a']:.4f} delta {c['delta']:+.3f} ci95 [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] adj_lo {c['ci_adjusted'][0]:+.3f} {c['class']}  never-cited R@20 {res['summaries'][c['a']]['unseen_recall_at_20']:.3f}", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["fit", "score", "show"])
    ap.add_argument("--id", choices=list(ARMS))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "score":
        return score()
    if a.step == "show" or not a.go:
        for sid, (learner, kind, tesis) in ARMS.items():
            print(sid, learner, kind, "tesis" if tesis else "", len(settings(learner, kind)))
        return 0
    return fit(a.id)


if __name__ == "__main__":
    raise SystemExit(main())
