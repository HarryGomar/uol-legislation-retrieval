"""C_hybrid: the full hybrid system (DECISION D7.0; author request "make it good").

Stage 1  the frozen EMB candidate pool (pool extension checked and rejected: +0.5% recall).
Stage 2  joint LTR refit (feature route, frozen hyper-parameters/seeds): frozen features + M1 memory (each fold's
         selected config) + L2 (variants a+b) [+ L4 synthetic memory] [+ R1 zero-shot reranker scores], at most 4
         configurations, cross-fitted selection (D3.1). Nothing here uses held-out rows.
Stage 3  Luna listwise reranking (L1 protocol: window 20, stride 10, two passes) of the Stage-2 top 30, run for
         3 samples; arm score = 1 / central rank over all 6 orders; fused with the Stage-2 score by a cross-fitted
         weight from W_EXT.

    python -m uol.round3.heavy uol.round3.c_hybrid stage2
    python -m uol.round3.c_hybrid stage3 [--samples 3]          # Luna (light)
    python -m uol.round3.heavy uol.round3.c_hybrid evaluate
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from uol.round3 import arm_runner, common as C, features, stats

ARM2 = "C_hybrid_ltr"
OUT = C.ARMS / "C_hybrid"
W_EXT = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
TOP = 30


def _available() -> dict[str, bool]:
    return {"l4": (C.ARMS / "L4" / "patterns_kept.jsonl").exists(),
            "r1": (C.RUN / "gpu_jobs/r1_qwen3_rr_4b_bf16_pspec_s1/remote/outputs/scores.npy").exists()}


def configs() -> list[dict]:
    av = _available()
    out = [{"parts": "m1+l2"}]
    if av["l4"]:
        out.append({"parts": "m1+l2+l4"})
    if av["r1"]:
        out.append({"parts": "m1+l2+r1"})
    if av["l4"] and av["r1"]:
        out.append({"parts": "m1+l2+l4+r1"})
    return out


_R1: dict = {}


def _r1_columns(h) -> np.ndarray:
    if "cols" not in _R1:
        from uol.round3 import r1_zeroshot
        cols = h.empty_columns(2)
        for j, (m, (rows, A, _)) in enumerate(sorted(r1_zeroshot.matrices().items())):
            for i, r in enumerate(rows):
                sl, pc = h.pair_slice(int(r))
                cols[sl, j] = A[i, pc]
        _R1["cols"] = cols
    return _R1["cols"]


def columns_for(cfg: dict, g, h, draw=None) -> np.ndarray:
    from uol.round3 import l2_features, m1_memory
    k = str(int(g.role.replace("dev_fold", "")))
    sel = json.loads((C.ARMS / "M1" / "RESULT.json").read_text(encoding="utf-8"))["selected_config"]
    parts = cfg["parts"].split("+")
    cols = []
    if "m1" in parts:
        cols.append(np.asarray(m1_memory.columns_for(sel[k], g, h)))
    if "l2" in parts:
        cols.append(np.asarray(l2_features.columns_for({"sources": "a+b"}, g, h)))
    if "l4" in parts:
        from uol.round3 import l4_luna
        cols.append(np.asarray(l4_luna.columns_for({"cols": "max+mean"}, g, h)))
    if "r1" in parts:
        cols.append(np.asarray(_r1_columns(h)))
    return np.hstack(cols).astype(np.float32)


def stage2() -> dict:
    res = arm_runner.run(ARM2, configs(), columns_for, noemb=False, full_view=True,
                         notes="C_hybrid Stage 2: joint LTR refit with M1 + L2 [+ L4] [+ R1] (D7.0)")
    h = features.Harness("emb", "spec")
    ck = C.ARMS / ARM2 / "ckpt"
    preds = {}
    for g in h.groups:
        cfg = res["selected_config"][g.role.replace("dev_fold", "")]
        z = np.load(ck / f"{arm_runner.cfg_key(cfg)}_{h.name}_{g.role}.npz")
        preds[g.role] = (z["idx"], z["score"])
    M = h.matrix(preds)
    rows = C.population_rows("P_spec", C.dev_rows())
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / "stage2_pspec_rows.npy", rows)
    np.save(OUT / "stage2_pspec_matrix.npy", np.asarray(M[rows], dtype=np.float32))
    return res


ARM2_R1 = "C_hybrid_ltr_r1"


def stage2_r1() -> dict:
    """Dev only (DEVIATIONS F0.1): the frozen stage 2 (m1+l2+l4) with and without the zero-shot R1 columns.

    R1 has no lockbox scores and needs GPU inference (5080 retired; the 1070 is not used), so this variant is never
    frozen: it measures what R1 would add to the frozen stage 2. The m1+l2+l4 refits are copied from C_hybrid_ltr,
    the selection between the two configurations is cross-fitted (D3.1), and the result is compared with the v1
    stage-2 ranking per component."""
    import shutil
    base_cfg = {"parts": "m1+l2+l4"}
    src, dst = C.ARMS / ARM2 / "ckpt", C.ARMS / ARM2_R1 / "ckpt"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob(f"{arm_runner.cfg_key(base_cfg)}_LTR_emb_hgb_dev_fold*.npz"):
        if not (dst / f.name).exists():
            shutil.copy2(f, dst / f.name)
    cfgs = [base_cfg, {"parts": "m1+l2+l4+r1"}]
    res = arm_runner.run(ARM2_R1, cfgs, columns_for, noemb=False, full_view=False,
                         notes="DEV ONLY: frozen C_hybrid stage 2 (m1+l2+l4) with/without zero-shot R1 4B+8B columns")
    h = features.Harness("emb", "spec")
    preds = {}
    for g in h.groups:
        cfg = res["selected_config"][g.role.replace("dev_fold", "")]
        z = np.load(dst / f"{arm_runner.cfg_key(cfg)}_{h.name}_{g.role}.npz")
        preds[g.role] = (z["idx"], z["score"])
    rows = C.population_rows("P_spec", C.dev_rows())
    M = h.matrix(preds)
    a = C.ep.evaluate(np.asarray(M[rows], dtype=np.float32), rows, "P_spec")
    v1 = C.ARMS / "C_hybrid" / "v1"
    vr = np.load(v1 / "stage2_pspec_rows.npy")
    if not np.array_equal(vr, rows):
        raise RuntimeError("v1 stage-2 rows differ")
    b = C.ep.evaluate(np.load(v1 / "stage2_pspec_matrix.npy"), rows, "P_spec")
    res["vs_stage2_v1"] = stats.contrast(a["per_component"]["ndcg_at_10"], b["per_component"]["ndcg_at_10"])
    res["label"] = "dev only; not frozen (R1 needs GPU inference for lockbox rows)"
    C.write_json(C.ARMS / ARM2_R1 / "RESULT.json", res)
    return res


def stage2_top() -> tuple[np.ndarray, np.ndarray]:
    rows = np.load(OUT / "stage2_pspec_rows.npy")
    M = np.load(OUT / "stage2_pspec_matrix.npy", mmap_mode="r")
    top = np.stack([np.argsort(-np.asarray(M[i], dtype=np.float64), kind="stable")[:TOP] for i in range(len(rows))])
    return rows, top


def stage3(samples: int = 3, workers: int = 8) -> dict:
    from uol.round3 import l1_luna
    rows, top = stage2_top()
    out = {}
    for s in range(samples):
        out[s] = l1_luna.run([int(r) for r in rows], "main", workers, "L1h", sample=s, top=top,
                             out_name=f"../C_hybrid/orders_s{s}.json")
    return out


def arm_matrix(samples: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rows = np.load(OUT / "stage2_pspec_rows.npy")
    n = len(C.release().candidate_ids)
    A = np.full((len(rows), n), np.nan, dtype=np.float32)
    orders = [json.loads((OUT / f"orders_s{s}.json").read_text(encoding="utf-8")) for s in range(samples)]
    for i, r in enumerate(rows):
        rk: dict[int, list[int]] = {}
        for o in orders:
            for p in o["passes"]:
                for pos, c in enumerate(o["orders"][p][str(r)], 1):
                    rk.setdefault(int(c), []).append(pos)
        for c, v in rk.items():
            A[i, c] = 1.0 / float(np.mean(v))
    return rows, A


def evaluate(samples: int = 3) -> dict:
    rows, A = arm_matrix(samples)
    S2 = np.load(OUT / "stage2_pspec_matrix.npy")
    fr = features.fusion_route("stage2", rows, A, grid=W_EXT, ltr_matrix=S2)
    ep = C.ep
    base = ep.evaluate(np.asarray(features.frozen_matrix("LTR_emb_hgb")[rows]), rows, "P_spec")
    st2 = ep.evaluate(S2, rows, "P_spec")
    full = ep.evaluate(fr["oof_matrix"], rows, "P_spec")
    l1r = json.loads((C.ARMS / "L1" / "RESULT.json").read_text(encoding="utf-8"))
    res = {"system": "C_hybrid", "stages": {"stage2_configs": configs(), "samples": samples, "w_grid": list(W_EXT),
                                            "w_by_fold": fr["w_by_fold"], "w_all": fr["w_all"],
                                            "dev_ndcg_by_w": fr["dev_ndcg_by_w"]},
           "dev_P_spec": {"LTR_emb_hgb": base["macro"]["ndcg_at_10"], "stage2": st2["macro"]["ndcg_at_10"],
                          "C_hybrid": full["macro"]["ndcg_at_10"],
                          "protocol_combination_L1": l1r["dev"]["vs_EMB"]["P_spec"]["arm_ndcg_at_10"]},
           "contrasts": {
               "C_hybrid_vs_EMB": stats.contrast(full["per_component"]["ndcg_at_10"], base["per_component"]["ndcg_at_10"]),
               "stage2_vs_EMB": stats.contrast(st2["per_component"]["ndcg_at_10"], base["per_component"]["ndcg_at_10"]),
               "C_hybrid_vs_stage2": stats.contrast(full["per_component"]["ndcg_at_10"], st2["per_component"]["ndcg_at_10"])},
           "notes": "dev only; Luna components get held-out claims only on the post-cutoff lockbox and the cohort (PLAN)"}
    C.write_json(OUT / "RESULT.json", res)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["stage2", "stage2-r1", "stage3", "evaluate"])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    C.ledger({"kind": "job_start", "job_id": f"C_hybrid_{a.command}", "command": " ".join(sys.argv)})
    if a.command == "stage2":
        r = stage2()
        print(json.dumps(r["dev"]["vs_EMB"]["P_spec"], default=float)[:600])
    elif a.command == "stage2-r1":
        r = stage2_r1()
        print(json.dumps({k: r["vs_stage2_v1"][k] for k in ("delta", "ci95", "class")}, default=float))
    elif a.command == "stage3":
        print(stage3(a.samples, a.workers))
    else:
        r = evaluate(a.samples)
        print(json.dumps({k: v for k, v in r.items() if k != "contrasts"}, indent=1, default=float))
        print(json.dumps({k: {kk: v[kk] for kk in ("delta", "ci95", "class")} for k, v in r["contrasts"].items()}, indent=1))
    C.ledger({"kind": "job_end", "job_id": f"C_hybrid_{a.command}", "exit_code": 0})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
