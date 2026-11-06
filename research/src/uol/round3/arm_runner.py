"""Shared feature-route arm runner (prompt §4; PLAN §5): configs, cross-fitted selection, nulls, subgroups, RESULT.json.

An arm supplies ``columns_for(cfg, group, harness, draw=None) -> [n_pairs, k] float32`` (NaN where undefined): the
extra columns for one dev outer group, computed so that they are out-of-fold for the group's scoring rows AND its
training rows (the a_p12 nesting rule). ``draw`` is None for the real arm and an int for null draws.

Selection (DECISION D3.1). Every configuration is refit on all five dev groups (frozen per-group hyper-parameters and
seeds, ``features.Harness``). For outer fold k the configuration with the best component-macro dev P_spec nDCG@10 on
the *other four folds* is chosen and its fold-k predictions are kept; the arm's dev estimate is this out-of-fold
assembly. The best single configuration's score is also reported (optimistic). Nulls rerun the per-fold selected
configurations with null columns. Refits run group-outer so each group's base matrices are built once.

Heavy CPU work takes one of ``CPU_SLOTS`` slots (``common.cpu_slot``) so concurrent arms cannot exhaust RAM.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.round3 import common as C
from uol.round3 import features as FT
from uol.round3 import stats

BASELINE_DIR = C.RUN / "phase0" / "harness"


def cfg_key(cfg: dict) -> str:
    return "c_" + "_".join(f"{k}-{cfg[k]}" for k in sorted(cfg)).replace(".", "p").replace(" ", "")


def _eval(matrix: np.ndarray) -> dict:
    return FT.evaluate_dev(matrix)


def _per_fold_scores(res: dict, rows_fold: dict[int, int]) -> dict[int, float]:
    """Component-macro P_spec nDCG@10 restricted to each fold (components are fold-pure)."""
    pq = res["P_spec"]["per_query"]
    by_fold: dict[int, dict[str, list[float]]] = {}
    comps = C.component_ids(res["P_spec"]["rows"])
    for r, rec, comp in zip(res["P_spec"]["rows"], pq, comps):
        by_fold.setdefault(rows_fold[int(r)], {}).setdefault(comp, []).append(rec["ndcg_at_10"])
    return {f: float(np.mean([np.mean(v) for v in d.values()])) for f, d in by_fold.items()}


def baseline(harness: FT.Harness, log=print) -> dict:
    """Baseline refit predictions (cached under phase0/harness/<pipeline>)."""
    return harness.refit(None, ckpt_dir=BASELINE_DIR / harness.name, tag="baseline", log=log)


def refit_configs(harness: FT.Harness, configs: Sequence[dict], columns_for: Callable, ckpt: Path,
                  draw: int | None = None, per_fold_cfg: dict[int, dict] | None = None, log=print) -> dict:
    """{cfg_key: {role: (idx, score)}} for all configs (or, with ``per_fold_cfg``, only each fold's chosen one)."""
    out: dict[str, dict] = {}
    for g in harness.groups:
        k = int(g.role.replace("dev_fold", ""))
        todo = [per_fold_cfg[k]] if per_fold_cfg is not None else list(configs)
        for cfg in todo:
            tag = cfg_key(cfg) + ("" if draw is None else f"_null{draw}")
            path = Path(ckpt) / f"{tag}_{harness.name}_{g.role}.npz"
            if path.exists():
                z = np.load(path)
                out.setdefault(cfg_key(cfg), {})[g.role] = (z["idx"], z["score"])
                continue
            t = time.perf_counter()
            with C.cpu_slot():
                extra = columns_for(cfg, g, harness, draw)
                pred = harness.fit_group(g, extra)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, idx=pred[0], score=pred[1])
            out.setdefault(cfg_key(cfg), {})[g.role] = pred
            log(f"{harness.name} {tag} {g.role}: {time.perf_counter() - t:.0f}s")
    return out


def assemble(harness: FT.Harness, preds_by_cfg: dict, choice: dict[int, str]) -> dict:
    return {g.role: preds_by_cfg[choice[int(g.role.replace("dev_fold", ""))]][g.role] for g in harness.groups}


def run(arm: str, configs: Sequence[dict], columns_for: Callable, *, task: str = "T1", n_nulls: int = 0,
        null_kinds: Sequence[str] = (), subgroups: dict[str, Callable[[Sequence[int]], np.ndarray]] | None = None,
        noemb: bool = True, full_view: bool = True, standalone: dict | None = None, notes: str = "",
        extra_result: dict | None = None, log=print) -> dict:
    """Run a feature-route arm end to end and write arms/<arm>/RESULT.json (plus ckpt/ and NOTES.md)."""
    if len(configs) > 24:
        raise ValueError("at most 24 configurations per arm")
    out_dir = C.ARMS / arm
    ckpt = out_dir / "ckpt"
    t0 = time.perf_counter()
    C.ledger({"kind": "job_start", "job_id": f"arm_{arm}", "configs": len(configs)})
    h = FT.Harness("emb", "spec")
    rows_fold = {int(r): int(str(C.roles()[int(r)]).replace("dev_fold", "")) for r in C.dev_rows()}
    base = _eval(h.matrix(baseline(h, log)))
    preds = refit_configs(h, configs, columns_for, ckpt, log=log)
    per_cfg = {}
    fold_scores = {}
    for cfg in configs:
        res = _eval(h.matrix(preds[cfg_key(cfg)]))
        per_cfg[cfg_key(cfg)] = res
        fold_scores[cfg_key(cfg)] = _per_fold_scores(res, rows_fold)
    choice = {}
    for k in range(5):
        def other(key):
            return np.mean([v for f, v in fold_scores[key].items() if f != k])
        best = max(other(cfg_key(c)) for c in configs)
        choice[k] = next(cfg_key(c) for c in configs if other(cfg_key(c)) >= best - 1e-12)
    sel = _eval(h.matrix(assemble(h, preds, choice)))
    dev = {"vs_EMB": FT.dev_contrasts(sel, base)}
    cfg_by_key = {cfg_key(c): c for c in configs}
    per_fold_cfg = {k: cfg_by_key[v] for k, v in choice.items()}
    if noemb:
        hn = FT.Harness("noemb", "spec")
        bn = _eval(hn.matrix(baseline(hn, log)))
        pn = refit_configs(hn, configs, columns_for, ckpt, per_fold_cfg=per_fold_cfg, log=log)
        dev["vs_NOEMB"] = FT.dev_contrasts(_eval(hn.matrix(assemble(hn, pn, choice))), bn)
        del hn
    if full_view:
        hf = FT.Harness("emb", "full")
        bf = _eval(hf.matrix(baseline(hf, log)))
        pf = refit_configs(hf, configs, columns_for, ckpt, per_fold_cfg=per_fold_cfg, log=log)
        dev["vs_EMB_full"] = FT.dev_contrasts(_eval(hf.matrix(assemble(hf, pf, choice))), bf)
        del hf
    if standalone:
        dev["standalone"] = standalone
    nulls = {}
    real_delta = dev["vs_EMB"]["P_spec"]["delta"]
    for kind_i, kind in enumerate(null_kinds):
        deltas = []
        for d in range(n_nulls):
            draw = 1000 * kind_i + d
            pnull = refit_configs(h, configs, lambda cfg, g, hh, dr, kind=kind: columns_for({**cfg, "_null": kind}, g, hh, dr),
                                  ckpt, draw=draw, per_fold_cfg=per_fold_cfg, log=log)
            rnull = _eval(h.matrix(assemble(h, pnull, choice)))
            deltas.append(rnull["P_spec"]["macro"]["ndcg_at_10"] - base["P_spec"]["macro"]["ndcg_at_10"])
        nulls[kind] = {"n": len(deltas), "deltas": deltas, "rank": int(1 + sum(x >= real_delta for x in deltas)),
                       "p95": float(np.quantile(deltas, 0.95)) if deltas else None,
                       "real_beats_p95": bool(deltas) and real_delta > float(np.quantile(deltas, 0.95))}
    sg = {}
    if subgroups:
        a, b = sel["P_spec"], base["P_spec"]
        assert a["rows"] == b["rows"]
        comps = C.component_ids(a["rows"])
        av = [q["ndcg_at_10"] for q in a["per_query"]]
        bv = [q["ndcg_at_10"] for q in b["per_query"]]
        for name, fn in subgroups.items():
            flags = np.asarray(fn(a["rows"]), dtype=bool)
            sg[name] = stats.subgroup_contrast(av, bv, comps, flags)
    files = sorted(ckpt.glob("*.npz"))
    result = {"arm": arm, "task": task, "route": "feature",
              "configs_tried": [{"config": c, "dev_score": per_cfg[cfg_key(c)]["P_spec"]["macro"]["ndcg_at_10"]}
                                for c in configs],
              "selected_config": {str(k): cfg_by_key[v] for k, v in choice.items()},
              "selection": "per outer fold, best on the other four folds (D3.1)",
              "best_single_config_dev": max(per_cfg[cfg_key(c)]["P_spec"]["macro"]["ndcg_at_10"] for c in configs),
              "dev": dev, "nulls": nulls, "subgroups": sg, "pool_recall": {},
              "compute": {"wall_seconds": round(time.perf_counter() - t0), "gpu_seconds": 0, "endpoint_inputs": 0,
                          "luna_calls": 0},
              "score_files": [{"path": str(f.relative_to(C.ROOT)), "sha256": C.sha256_file(f)} for f in files],
              "code_sha256": {p.name: C.sha256_file(p) for p in (Path(__file__), Path(FT.__file__))},
              "notes": notes, **(extra_result or {})}
    C.write_json(out_dir / "RESULT.json", result)
    C.ledger({"kind": "job_end", "job_id": f"arm_{arm}", "exit_code": 0, "wall_seconds": result["compute"]["wall_seconds"],
              "dev_delta_P_spec": real_delta})
    log(f"{arm}: dev P_spec Δ vs EMB {real_delta:+.4f} {dev['vs_EMB']['P_spec']['ci95']} class {dev['vs_EMB']['P_spec']['class']}")
    return result


# ----------------------------------------------------------------------------- fusion-route arms

def run_fusion(arm: str, configs: dict[str, tuple[np.ndarray, np.ndarray]], *, task: str = "T1",
               controls: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
               subgroups: dict[str, Callable[[Sequence[int]], np.ndarray]] | None = None, notes: str = "",
               extra_result: dict | None = None, log=print) -> dict:
    """Fusion-route arm over dev P_spec rows. ``configs``: name -> (rows, A [len(rows), N] arm scores, NaN outside each
    row's scored LTR top K). Per config the weight w is cross-fitted (``features.fusion_route``); per outer fold the
    config with the best other-fold score is kept (D3.1). Controls are scored the same way (single config each)."""
    t0 = time.perf_counter()
    out_dir = C.ARMS / arm
    ref_name = "LTR_emb_hgb"
    names = list(configs)
    if len(names) > 24:
        raise ValueError("at most 24 configurations per arm")
    rows0 = np.asarray(configs[names[0]][0])
    for n in names:
        if not np.array_equal(np.asarray(configs[n][0]), rows0):
            raise ValueError("all configurations must score the same rows")
    pspec = set(ep.population_rows("P_spec", rows0).tolist())
    keep = np.asarray([int(r) in pspec for r in rows0])
    rows = rows0[keep]
    folds = C.fold_of(rows)
    ltr = np.asarray(FT.frozen_matrix(ref_name)[rows])
    base = ep.evaluate(ltr, rows, "P_spec")
    comps = C.component_ids(rows)
    fused, per_fold, wsel = {}, {}, {}
    for n in names:
        fr = FT.fusion_route(ref_name, rows0, configs[n][1])
        M = fr["oof_matrix"][keep]
        fused[n] = M
        wsel[n] = {"w_by_fold": fr["w_by_fold"], "w_all": fr["w_all"]}
        res = ep.evaluate(M, rows, "P_spec")
        nd = {int(r): q["ndcg_at_10"] for r, q in zip(res["rows"], res["per_query"])}
        per_fold[n] = {}
        for f in sorted(set(folds.tolist())):
            by = {}
            for r, c, ff in zip(rows, comps, folds):
                if ff == f and int(r) in nd:
                    by.setdefault(c, []).append(nd[int(r)])
            per_fold[n][f] = float(np.mean([np.mean(v) for v in by.values()]))
    choice = {}
    for f in sorted(set(folds.tolist())):
        other = {n: np.mean([v for ff, v in per_fold[n].items() if ff != f]) for n in names}
        best = max(other.values())
        choice[f] = next(n for n in names if other[n] >= best - 1e-12)
    S = np.empty_like(ltr)
    for f, n in choice.items():
        S[folds == f] = fused[n][folds == f]
    sel = ep.evaluate(S, rows, "P_spec")
    dev = {"vs_EMB": {"P_spec": {**stats.contrast(sel["per_component"]["ndcg_at_10"], base["per_component"]["ndcg_at_10"]),
                                 "arm_ndcg_at_10": sel["macro"]["ndcg_at_10"], "ref_ndcg_at_10": base["macro"]["ndcg_at_10"]}}}
    alone = {}
    for n in names:
        A = np.asarray(configs[n][1])[keep].astype(np.float64)
        A = np.where(np.isfinite(A), A, -1e9)
        alone[n] = ep.evaluate(A, rows, "P_spec")["macro"]["ndcg_at_10"]
    dev["standalone"] = {"P_spec": {"ndcg_at_10_by_config": alone,
                                    "note": "arm scores alone over the scored top K (replacement ranking; negative control)"}}
    ctrl = {}
    for cname, (crows, CA) in (controls or {}).items():
        crows = np.asarray(crows)
        ck = np.asarray([int(r) in pspec for r in crows])
        fr = FT.fusion_route(ref_name, crows, CA)
        cres = ep.evaluate(fr["oof_matrix"][ck], crows[ck], "P_spec")
        cb = ep.evaluate(np.asarray(FT.frozen_matrix(ref_name)[crows[ck]]), crows[ck], "P_spec")
        ctrl[cname] = {**stats.contrast(cres["per_component"]["ndcg_at_10"], cb["per_component"]["ndcg_at_10"]),
                       "w_by_fold": fr["w_by_fold"], "rows": int(ck.sum())}
    sg = {}
    if subgroups:
        av = {int(r): q["ndcg_at_10"] for r, q in zip(sel["rows"], sel["per_query"])}
        bv = {int(r): q["ndcg_at_10"] for r, q in zip(base["rows"], base["per_query"])}
        rr = [r for r in sel["rows"] if r in bv]
        for name, fn in subgroups.items():
            sg[name] = stats.subgroup_contrast([av[r] for r in rr], [bv[r] for r in rr], C.component_ids(rr),
                                               np.asarray(fn(rr), dtype=bool))
    result = {"arm": arm, "task": task, "route": "fusion",
              "configs_tried": [{"config": n, "dev_score": float(np.mean(list(per_fold[n].values())))} for n in names],
              "selected_config": {str(f): n for f, n in choice.items()}, "fusion_weights": wsel,
              "selection": "per outer fold, best on the other folds (D3.1); w cross-fitted per config",
              "dev": dev, "controls": ctrl, "nulls": {}, "subgroups": sg, "pool_recall": {},
              "compute": {"wall_seconds": round(time.perf_counter() - t0)}, "score_files": [],
              "code_sha256": {p.name: C.sha256_file(p) for p in (Path(__file__), Path(FT.__file__))},
              "notes": notes, **(extra_result or {})}
    C.write_json(out_dir / "RESULT.json", result)
    d = dev["vs_EMB"]["P_spec"]
    log(f"{arm}: dev P_spec Δ vs EMB {d['delta']:+.4f} {d['ci95']} class {d['class']}")
    return result
