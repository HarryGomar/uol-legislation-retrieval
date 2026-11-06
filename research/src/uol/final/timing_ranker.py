"""Whole-method time of the frozen ranker S0+qwen3_8b for one new case (D196; D137's method, ``uol.final.timing``).

About 200 development cases (the first score rows of DEV fold 0, the cases of ``timing_20261026d``) are treated as
new: the facts go in, the top list comes out. The stores, indexes and the saved all-development model load once (load
times reported apart). Per case, from scratch: the Qwen3-8B query vector (GPU, ``gpu_jobs/timing_encode_ranker.py`` on
the remote worker; its seconds are added and its vector injected), the two LSA projections, the 28 first-stage sources
(``timing_sources.FirstStage``), the topic columns, the pool, the features, the decision function and the top 10.

The first stage is the fold-0 group's (the group the stored source rows of these cases came from, so its rows can be
checked against the frozen files); its memories hold four fifths of the development cases, against all of them for a
truly new case. Not timed: the per-case forward pass of the adapters over the LSA vectors, whose outputs the first
stage holds per row (``timing_sources`` docstring).

Each case then runs twice more, untimed: the first stage on the stored Qwen3-8B vector, compared source by source with
the frozen files, and ``NewCaseRanker.rank`` on the frozen sources (``test_set.rank``'s path); the top 10 of each is
compared with the timed one. ``FirstStage._qsims`` keys its memo on the row, so it is cleared before every pass (in
``timing.Run`` the fresh pass reused the stored pass's similarities).

    py -3.12 -m uol.final.timing_ranker post --go                 send the encoder job (once)
    py -3.12 -m uol.final.timing_ranker pull --go                 wait for it, verify, copy to <out>/encode
    py -3.12 -m uol.final.timing_ranker run --go [--cases 200]    the timed cases (stored vectors if no encode output)
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from uol.final import new_cases_fit as NF

ROOT = Path(__file__).resolve().parents[3]
SYSTEM = "S0+qwen3_8b"
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "timing"
MODELS = ROOT / "experiments" / "runs" / "audit_d181_20261028" / "new" / "models"
WORK = ROOT / ".cache" / "timing_ranker_20261102"
STAGE = ROOT / ".cache" / "timing_stage_20261026c" / "data"     # the verified staged inputs, bundle row order (T2)
JOB = "timing_encode_s0q8b_j2"
N_CASES, N_WARMUP = 200, 5
WAIT_S = 10 * 3600


def activate() -> None:
    from uol.final import new_cases_steps as NS
    os.environ["UOL_TIMING_DATA"] = str(STAGE)
    NS.link_fit_features(WORK)
    NF.activate(WORK)


def case_rows(n: int, warmup: int) -> tuple[list[int], list[int]]:
    """The first ``n`` score rows of DEV fold 0 (``timing.case_rows``) and the next ``warmup``."""
    from uol.final import guard
    f = next(x for x in NF.dev_population().folds() if x.k == 0)
    rows = guard.dev_rows_only(np.asarray(f.score_rows[:n + warmup]), "timing rows")
    return [int(r) for r in rows[:n]], [int(r) for r in rows[n:]]


def post(n: int, warmup: int) -> int:
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    activate()
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.round3 import gpu_remote as G
    from uol.scale_curve import legal
    timed, warm = case_rows(n, warmup)
    d = ep.release()
    jobs = Path(G.__file__).parent / "gpu_jobs"
    OUT.mkdir(parents=True, exist_ok=True)
    task = OUT / "encode_task.json"
    task.write_text(json.dumps({"timed": timed, "warmup": warm, "model": {**legal.GATE["qwen3_8b_gate"], "key": "qwen3_8b_timing"},
                                "inputs": {str(r): qe.query_input("c", d.queries[r].text) for r in timed + warm}}, ensure_ascii=False), encoding="utf-8")
    job = G.pack(JOB, Path(__file__).parent / "gpu_jobs" / "timing_encode_ranker.py", {"task.json": task},
                 helpers=[jobs / "legal_encode.py", jobs / "rerank_bench.py", ROOT / "src" / "uol" / "scale_curve" / "store.py"],
                 expected_outputs=["encode.npz", "encode.json"])
    G.push(job)
    print(f"posted {JOB}: {len(timed)} timed, {len(warm)} warm-up", flush=True)
    return 0


def pull() -> int:
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    from uol.round3 import gpu_remote as G
    job = G.LOCAL_JOBS / JOB
    rc = G.run(job, timeout=WAIT_S)
    G.pull(job)
    (OUT / "encode").mkdir(parents=True, exist_ok=True)
    for p in (job / "remote" / "outputs").iterdir():
        shutil.copy2(p, OUT / "encode" / p.name)
    shutil.copy2(job / "remote" / "receipt.json", OUT / "encode" / "receipt.json")
    print(f"{JOB}: exit {rc}", flush=True)
    return rc


def run(n: int, warmup: int, out: Path) -> int:
    import psutil
    from uol.embedding_push import common as ep
    from uol.final import timing as T, timing_sources as TS
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    load: dict[str, float] = {}

    def timed_load(name: str, fn):
        t0 = time.perf_counter()
        v = fn()
        load[name] = round(time.perf_counter() - t0, 2)
        print(f"{time.strftime('%H:%M:%S')} load {name}: {load[name]} s", flush=True)
        return v

    timed_load("profile", activate)
    me = psutil.Process()
    d = ep.release()
    timed_rows, warm = case_rows(n, warmup)
    info = json.loads((MODELS / SYSTEM / "info.json").read_text(encoding="utf-8"))
    for name, h in info["files"].items():
        if NF.sha256(MODELS / SYSTEM / name) != h:
            raise RuntimeError(f"{SYSTEM}/{name}: sha256 differs from info.json")

    def ranker():
        with (MODELS / SYSTEM / "model.pkl").open("rb") as f:
            model = pickle.load(f)
        return NF.NewCaseRanker(NF.prepared(SYSTEM, NF.dev_population()), model, NF.load_topic_models(MODELS, info), None)

    rk = timed_load("ranker", ranker)
    fs = TS.FirstStage("dev_fold0", sources=rk.sources, colbert="off")
    with T.first_stage_labels():
        timed_load("first_stage", fs.load)
    load.update({f"first_stage:{k}": v for k, v in fs.load_seconds.items()})
    enc = timed_load("lsa_projections", lambda: T.CpuEncoders(int(os.environ["UOL_FINAL_THREADS"]), lsa=True, e5=False))
    gpu, gpu_meta, z = {}, None, None
    if (out / "encode" / "encode.npz").exists():
        z = np.load(out / "encode" / "encode.npz")
        z = {k: z[k] for k in z.files}
        gpu = {(int(r), bool(w)): i for i, (r, w) in enumerate(zip(z["rows"], z["warmup"]))}
        gpu_meta = json.loads((out / "encode" / "encode.json").read_text(encoding="utf-8"))
    rss_loaded = me.memory_info().rss / 1e9
    print(f"{time.strftime('%H:%M:%S')} loaded, {rss_loaded:.1f} GB; encodings: {'fresh' if gpu else 'none (stored vectors, no GPU seconds)'}", flush=True)

    def sources(row: int, inject: dict | None, clock=None) -> dict:
        fs.reset_vectors([row])
        fs.S.pop("qe_sims_key", None)
        if inject:
            fs.inject(row, inject)
        with T.first_stage_labels():
            return fs.case(row, clock=clock)

    def one(row: int, warmup_case: bool) -> dict:
        text, group = d.queries[row].text, int(rk.h.ctx.fold[row])
        clock = T.Clock()
        rec: dict = {"row": row, "query_id": d.queries[row].query_id, "warmup": warmup_case, "fidelity": {}}
        inj: dict = {}
        g = gpu.get((row, warmup_case))
        if g is not None:
            clock.add("encode:qwen", float(z["seconds"][g]), float(z["cpu_seconds"][g]))
            inj["Qc"] = np.asarray(z["vectors"][g], dtype=np.float32)
            rec["fidelity"]["qwen_cos"] = float(np.dot(inj["Qc"], fs.stored["Qc"][row]) / np.linalg.norm(inj["Qc"]) / np.linalg.norm(fs.stored["Qc"][row]))
        psutil.cpu_percent(None)
        w0, c0 = time.perf_counter(), time.process_time()
        for name in enc.lsa:
            with clock.stage(f"transform:{name}"):
                inj[name] = enc.transform(name, text)
        src = sources(row, inj, clock)
        with clock.stage("topic"):
            tr = rk.topic_rows(text)
        with clock.stage("pool"):
            pc = rk.pool(src)
        with clock.stage("features"):
            X = np.hstack([rk.features(text, src, pc, group), rk.extras(tr, pc, None)]).astype(np.float32)
        with clock.stage("rank"):
            S = T.CaseRanker.pooled_row(rk, pc, rk.model.decision_function(X).astype(np.float32), np.asarray(src["L_star"]))
        with clock.stage("top10"):
            top = T.top10(S)
        rec["local_wall"], rec["local_cpu"] = time.perf_counter() - w0, time.process_time() - c0
        rec["machine_cpu_percent"] = psutil.cpu_percent(None)
        rec["rss_gb"] = me.memory_info().rss / 1e9
        rec["wall"], rec["cpu"] = {k: round(v, 6) for k, v in clock.wall.items()}, {k: round(v, 6) for k, v in clock.cpu.items()}
        # untimed checks: the same path on the stored vector and on the frozen source rows
        stored = rk.rank(text, NF.frozen_sources(rk.sources, row), group, topic_rows=tr)["row"]
        rec["same_path_top10_equal"] = T.top10(rk.rank(text, src, group, topic_rows=tr)["row"]) == top
        again = sources(row, None)
        rec["sources"] = {s: TS.compare_row(s, row, again[s]) for s in rk.sources}
        rec["top10"], stored_top = top, T.top10(stored)
        rec["top10_equal_stored_sources"] = top == stored_top
        rec["recomputed_top10_equal_stored_sources"] = T.top10(rk.rank(text, again, group, topic_rows=tr)["row"]) == stored_top
        return rec

    for r in warm:
        one(r, True)
    print(f"{time.strftime('%H:%M:%S')} warm-up done", flush=True)
    recs = []
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for i, r in enumerate(timed_rows):
            recs.append(one(r, False))
            f.write(json.dumps(recs[-1], default=float) + "\n")
            f.flush()
            if (i + 1) % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + 1}/{len(timed_rows)}", flush=True)

    enc_w = [r["wall"].get("encode:qwen", 0.0) for r in recs]
    enc_c = [r["cpu"].get("encode:qwen", 0.0) for r in recs]
    names = sorted({k for r in recs for k in r["wall"]})
    stages = sorted({T.stage_of(k) for k in names})
    share = lambda key: float(np.mean([bool(r[key]) for r in recs]))
    mem = me.memory_info()
    res = {"system": SYSTEM, "status": "done", "cases": len(recs), "first_stage": "dev_fold0", "threads": int(os.environ["UOL_FINAL_THREADS"]),
           "query_vector": "fresh (remote worker)" if gpu else "stored: no GPU seconds in these totals",
           "whole_method_seconds": T.stats([a + r["local_wall"] for a, r in zip(enc_w, recs)]),
           "local_seconds": T.stats([r["local_wall"] for r in recs]),
           "gpu_seconds": T.stats(enc_w) if gpu else None,
           "cpu_seconds": {"local": T.stats([r["local_cpu"] for r in recs]), "encoder_host": T.stats(enc_c) if gpu else None,
                           "total": T.stats([a + r["local_cpu"] for a, r in zip(enc_c, recs)])},
           "stages_wall": {st: T.stats([sum(v for k, v in r["wall"].items() if T.stage_of(k) == st) for r in recs]) for st in stages},
           "entries_wall_median": {k: float(np.median([r["wall"].get(k, 0.0) for r in recs])) for k in names},
           "ram_gb": {"process_peak_including_load": round(getattr(mem, "peak_wset", 0) / 1e9, 2), "process_peak_committed": round(getattr(mem, "peak_pagefile", 0) / 1e9, 2),
                      "after_load": round(rss_loaded, 2),
                      "largest_during_cases": round(max(r["rss_gb"] for r in recs), 2),
                      "encoder_process_peak": gpu_meta.get("peak_process_ram_gb") if gpu_meta else None},
           "gpu_memory_gb": {k: gpu_meta.get(k) for k in ("peak_allocated_gb", "peak_reserved_gb", "gpu", "card_used_before_load", "card_used_after_cases")} if gpu_meta else None,
           "agreement": {"top10_timed_equal_stored_sources": share("top10_equal_stored_sources"),
                         "top10_recomputed_equal_stored_sources": share("recomputed_top10_equal_stored_sources"),
                         "same_path": share("same_path_top10_equal"), "qwen_cos": T.stats([r["fidelity"].get("qwen_cos") for r in recs])},
           "sources_vs_frozen": {s: {"min_top200_overlap": min(r["sources"][s]["top200_overlap"] for r in recs),
                                     "max_abs_diff": max(r["sources"][s]["max_abs_diff"] for r in recs)} for s in rk.sources},
           "machine_cpu_percent": T.stats([r["machine_cpu_percent"] for r in recs]), "machine": T.machine(), "load_seconds": load,
           "encoder_load_seconds": gpu_meta.get("load_seconds") if gpu_meta else None, "wall_seconds_total": round(time.perf_counter() - t_all, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "RESULT.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("whole_method_seconds", "cpu_seconds", "ram_gb", "gpu_seconds", "agreement")}, indent=1, default=float), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["post", "pull", "run"])
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--cases", type=int, default=N_CASES)
    ap.add_argument("--warmup", type=int, default=N_WARMUP)
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step} {SYSTEM}")
        return 0
    if a.step == "post":
        return post(a.cases, a.warmup)
    if a.step == "pull":
        return pull()
    return run(a.cases, a.warmup, a.out)


if __name__ == "__main__":
    sys.exit(main())
