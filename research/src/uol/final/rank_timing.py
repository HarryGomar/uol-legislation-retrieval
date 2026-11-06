"""What the chosen ranker adds to the answer time of a case (D209, research/programs/rank_rerun_20261105/PLAN.md).

The same 200 development cases and fresh encoder vectors as ``timing_ranker`` (``footing_20261101/timing``). Its other
timed steps are unchanged, so two are measured here and combined by ``scripts/final_results.py``:

* ``memory``: the older-tesis votes for one new case: nearest tesis by the fresh Qwen3-8B vector, then their citations
  (``timing.M1Case``, the configuration of a new case).
* ``predict``: LightGBM lambdarank scoring a case's pool, with the model fitted on fold 0's training cases at the
  chosen settings, against the stored pointwise step.
* ``cpu_encode``: the Qwen3-8B query vector on a processor, on the remote worker (H071), posted and read back.

Run while no refit holds the processor.

    py -3.12 -m uol.final.rank_timing memory --go
    py -3.12 -m uol.final.rank_timing predict --go
    py -3.12 -m uol.final.rank_timing cpu_encode --go
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TIMING = ROOT / "experiments" / "runs" / "footing_20261101" / "timing"
OUT = TIMING / "rk"
CPU_JOB = "timing_encode_cpu_rk1"
CPU_CASES = 50
CPU_THREADS = 8


def rows_and_vectors() -> tuple[list[int], dict[int, np.ndarray]]:
    z = np.load(TIMING / "encode" / "encode.npz")
    rows = [int(r) for r, w in zip(z["rows"], z["warmup"]) if not w]
    return rows, {int(r): z["vectors"][i] for i, (r, w) in enumerate(zip(z["rows"], z["warmup"])) if not w}


def spread(v: list[float]) -> dict:
    w = sorted(v)
    return {"n": len(v), "median": statistics.median(v), "p95": w[int(np.ceil(0.95 * len(w))) - 1], "max": w[-1]}


def memory() -> int:
    import psutil
    from uol.final import new_cases_fit as NF, timing as T, timing_ranker as TR
    TR.activate()
    me = psutil.Process()
    t0 = time.perf_counter()
    m1 = T.M1Case(dict(NF.M1_CFG))
    load = time.perf_counter() - t0
    rss = me.memory_info().rss / 1e9
    rows, vec = rows_and_vectors()
    pc = np.arange(1488, dtype=np.int64)
    wall, cpu, same = [], [], []
    for r in rows:
        w, c = time.perf_counter(), time.process_time()
        _, info = m1.columns(vec[r], r, pc, cached=False)
        wall.append(time.perf_counter() - w)
        cpu.append(time.process_time() - c)
        same.append(bool(info.get("same_top_k")))
    mem = me.memory_info()
    res = {"step": "older-tesis votes", "config": dict(NF.M1_CFG), "threads": os.environ.get("UOL_FINAL_THREADS"), "load_seconds": round(load, 2), "wall": spread(wall), "cpu": spread(cpu),
           "ram_gb_loaded": round(rss, 2), "ram_gb_peak": round(getattr(mem, "peak_wset", 0) / 1e9, 2), "same_neighbours_as_stored": sum(same), "cases": len(rows),
           "memory_vectors_bytes": (Path(os.environ["UOL_TIMING_DATA"]) / "m1" / "dense_memory_vectors.npy").stat().st_size}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "memory.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    return 0


def predict() -> int:
    from uol.embedding_push import fusion_ltr as fl
    from uol.final import lean_rerun as LR, populations, ranker_tests as RT, registry
    from uol.gap_tests import ltr_listwise as LW
    LR.D.activate(LR.STORE)
    LR._overlay_stat_glob()
    pop = populations.load("DEV", None)
    sc = registry.make("rt_lm_tesis")
    sc.prepare(pop)
    fold = next(f for f in pop.folds() if f.k == 0)
    g, _ = sc._group(fold)
    extra, _ = sc._extra_columns(fold, g)
    cfg = {"learning_rate": 0.05, "num_leaves": 63}          # chosen in fold 0
    rng = np.random.default_rng(fl.SEED + 999)
    idx, y, counts = LW.training_lists(sc.h.ctx, sc.h.F, sc.h.v, fl.train_rows_of(sc.h.ctx, g), rng)
    RT.LM_THREADS = 4
    model = RT._fit("lgbm", cfg, RT._matrix(sc.h.F, sc.h.v, idx, extra), y, counts, fl.SEED)
    rows, _ = rows_and_vectors()
    wall, cpu, pairs = [], [], []
    X = None
    for r in rows:
        pi = fl.row_pairs(sc.h.v, np.asarray([r]), False, sc.h.F)
        X = RT._matrix(sc.h.F, sc.h.v, pi, extra)
        w, c = time.perf_counter(), time.process_time()
        model.predict(X)
        wall.append(time.perf_counter() - w)
        cpu.append(time.process_time() - c)
        pairs.append(len(pi))
    res = {"step": "LambdaMART scoring a case's pool", "settings": cfg, "threads": 4, "wall": spread(wall), "cpu": spread(cpu), "pairs_median": statistics.median(pairs), "inputs": int(X.shape[1])}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predict.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1), flush=True)
    return 0


def cpu_encode() -> int:
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.final import timing_ranker as TR
    from uol.round3 import gpu_remote as G
    from uol.scale_curve import legal
    TR.activate()
    d = ep.release()
    rows, _ = rows_and_vectors()
    rows = rows[:CPU_CASES]
    jobs = Path(G.__file__).parent / "gpu_jobs"
    OUT.mkdir(parents=True, exist_ok=True)
    task = OUT / "cpu_encode_task.json"
    model = {**legal.GATE["qwen3_8b_gate"], "key": "qwen3_8b_timing_cpu", "precision": "fp32"}
    task.write_text(json.dumps({"rows": rows, "threads": CPU_THREADS, "model": model, "inputs": {str(r): qe.query_input("c", d.queries[r].text) for r in rows}}, ensure_ascii=False), encoding="utf-8")
    G.submit(CPU_JOB, Path(__file__).parent / "gpu_jobs" / "timing_encode_cpu.py", {"task.json": task},
             helpers=[jobs / "legal_encode.py", jobs / "rerank_bench.py", ROOT / "src" / "uol" / "scale_curve" / "store.py"], expected_outputs=["encode_cpu.json"], timeout=8 * 3600)
    out = G.LOCAL_JOBS / CPU_JOB / "remote" / "outputs" / "encode_cpu.json"
    (OUT / "encode_cpu.json").write_bytes(out.read_bytes())
    print(out.read_text(encoding="utf-8"), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["memory", "predict", "cpu_encode"])
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step}")
        return 0
    return {"memory": memory, "predict": predict, "cpu_encode": cpu_encode}[a.step]()


if __name__ == "__main__":
    sys.exit(main())
