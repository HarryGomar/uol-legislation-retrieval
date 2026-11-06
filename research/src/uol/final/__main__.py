"""Command line of the final benchmark harness.

    py -3.12 -m uol.final systems                         # (re)write systems.json from the registry code
    py -3.12 -m uol.final populations                     # build and save the population row lists
    py -3.12 -m uol.final run --system lstar --population DEV [--folds 0] [--max-rows 50] [--force]
    py -3.12 -m uol.final g1 [--route cached|recompute|all]
    py -3.12 -m uol.final report
    py -3.12 -m uol.final cost --system lstar             # latency (200 DEV cases, 8 threads) + cost record
    py -3.12 -m uol.final contrast --a S0 --b ltr_noemb_refit [--population DEV]
    py -3.12 -m uol.final export-top --system S_full_ext --k 30
    py -3.12 -m uol.final bundle --out DIR [--smoke]      # development-only portable bundle
    py -3.12 -m uol.final plan --out jobs/plan.json       # the overnight DAG
    py -3.12 -m uol.final overnight --plan jobs/plan.json # the overnight runner (run computer)
    py -3.12 -m uol.final job --id <job id> --plan ...    # one plan job (what the runner launches)
    py -3.12 -m uol.final assignments --plan ... --out ...  # the second runner's ASSIGNMENTS.json (D85)
    py -3.12 -m uol.final merge-store --from <raw mirror>  # the run computer's results into this store (D86)

Global options: ``--data DIR`` (a bundle's data folder; also env UOL_FINAL_DATA), ``--release v11`` (local profile),
``--store DIR`` (results store; default experiments/runs/final_20261021 locally).
Heavy CPU commands on the thesis PC go through ``py -3.12 -m uol.round3.heavy uol.final ...``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _profile(a) -> None:
    from uol.final import context
    data = a.data or os.environ.get("UOL_FINAL_DATA")
    if data:
        context.activate_bundle(Path(data), store=Path(a.store) if a.store else None)
    else:
        context.activate_local(a.release, store=Path(a.store) if a.store else None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final", description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None)
    ap.add_argument("--release", default="v11")
    ap.add_argument("--store", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("systems")
    p = sub.add_parser("populations")
    p.add_argument("--names", default=None)
    p = sub.add_parser("run")
    p.add_argument("--system", required=True)
    p.add_argument("--population", default="DEV")
    p.add_argument("--folds", default=None)
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("rebuild-stage")
    p.add_argument("stage", nargs=argparse.REMAINDER)
    p = sub.add_parser("g1")
    p.add_argument("--route", default="cached", choices=["cached", "recompute", "all", "rebuild"])
    p.add_argument("--only", default=None)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=None)
    p = sub.add_parser("cost")
    p.add_argument("--system", required=True)
    p.add_argument("--no-latency", action="store_true")
    sub.add_parser("cost-summary")                            # LLM usage per system + fine-tune records (llm_cost)
    p = sub.add_parser("contrast")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--population", default="DEV")
    p = sub.add_parser("export-top")
    p.add_argument("--system", required=True)
    p.add_argument("--population", default="DEV")
    p.add_argument("--k", type=int, default=30)
    p = sub.add_parser("bundle")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true", help="small bundle: a few sources, no large vectors")
    p.add_argument("--no-compress", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("--out", required=True)
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("overnight")
    p.add_argument("--plan", required=True)
    p.add_argument("--drive-root", default=None)
    p.add_argument("--ram-share", type=float, default=0.75)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--vram-gb", type=float, default=None,
                   help="declared-VRAM budget of GPU jobs sharing the card (default: card minus 6 GB)")
    p.add_argument("--local-root", default=None)
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument("--results-root", default=None,
                   help="second runner: Drive folder (name or full path) whose results/ receives this runner's results")
    p.add_argument("--jobs-file", default=None, help="run only these jobs: ids file or ASSIGNMENTS.json (active ones)")
    p = sub.add_parser("job")
    p.add_argument("--id", required=True)
    p.add_argument("--plan", required=True)
    p = sub.add_parser("reset-jobs")                          # explicit invalidation of finished jobs (runner stopped)
    p.add_argument("--plan", required=True)
    p.add_argument("--ids-file", required=True, help="text file, one job id per line (# comments allowed)")
    p.add_argument("--tag", required=True)
    sub.add_parser("features")
    p = sub.add_parser("ingest")
    p.add_argument("--key", required=True)
    sub.add_parser("gate-e0")
    p = sub.add_parser("unpack")
    p.add_argument("--root", required=True)
    p = sub.add_parser("gpu-prep")
    p.add_argument("--kind", required=True)
    p.add_argument("--fold", default=None)
    p.add_argument("--smoke", action="store_true")
    sub.add_parser("codex-smoke")
    p = sub.add_parser("llm")
    p.add_argument("--job", required=True, choices=["llm_alone", "hyde", "llm_rerank", "agent", "devl_rewrite"])
    p.add_argument("--base", default=None)
    p.add_argument("--condition", default=None)
    p.add_argument("--rows", type=int, default=None, help="smoke: the first N rows only")
    p.add_argument("--samples", type=int, default=3)
    p.add_argument("--variant", default="main", choices=["main", "blank"], help="llm_rerank: C5 blank-text control")
    p.add_argument("--population", default=None, help="agent: score this population's rows (DEV-A300, D96)")
    p.add_argument("--smoke", action="store_true")
    p = sub.add_parser("ce")
    p.add_argument("--step", required=True, choices=["prep", "fuse", "zs-prep", "zs-fuse", "smoke-prep"])
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--system", default=None)
    sub.add_parser("rebuild-copy-frozen")
    sub.add_parser("models")
    p = sub.add_parser("variant")
    p.add_argument("--build", default=None)
    p.add_argument("--encode-inputs", default=None)
    p.add_argument("--vectors", default=None)
    p.add_argument("--dense-queries", default=None)
    sub.add_parser("devb-labels")
    sub.add_parser("families")
    p = sub.add_parser("merge-store")
    p.add_argument("--from", dest="src", required=True, help="the run computer's results\\raw mirror (D86)")
    p = sub.add_parser("pack")
    p.add_argument("--only", default=None)
    p = sub.add_parser("new-cases", help="E13 (D161, D162): py -3.12 -m uol.final new-cases --step <step> [options]")
    p.add_argument("--step", required=True)
    p.add_argument("rest", nargs=argparse.REMAINDER, help="options of uol.final.new_cases")
    p = sub.add_parser("external")
    p.add_argument("--kind", required=True, choices=["claude", "jev", "sonnet", "haiku", "gpt6sol"],
                   help="C10 (D101): fuse the thesis PC's outputs; sonnet, haiku, gpt6sol: the LLM sweep (D174)")
    p = sub.add_parser("assignments")
    p.add_argument("--plan", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--done-status", default=None, help="RUN-A's STATUS.json: write the D86 fallback assignments")
    p.add_argument("--reset", action="append", default=[], help="reset list whose jobs count as not done (repeatable)")
    p = sub.add_parser("bundle-new")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    if a.cmd == "new-cases":                                  # E13: each step activates its own profile
        from uol.final import new_cases
        return new_cases.main([a.step, *a.rest])
    if a.cmd == "bundle-new":
        from uol.final import new_cases
        return new_cases.build_bundle(Path(a.out))
    if a.cmd == "unpack":                                     # setup step: restore packed arrays, check sha256
        from uol.final import setup_files
        return setup_files.unpack_all(Path(a.root))

    if a.cmd == "bundle":                                     # always built from the local profile
        from uol.final import bundle, context
        context.activate_local(a.release)
        print(json.dumps(bundle.build(Path(a.out), smoke=a.smoke, compress=not a.no_compress), indent=1, default=str)[:4000])
        return 0
    if a.cmd == "assignments":                                # RUN-B job lists (D85)
        from uol.final import assign
        print(json.dumps(assign.write(Path(a.plan), Path(a.out), Path(a.done_status) if a.done_status else None,
                                      [Path(x) for x in a.reset]), indent=1))
        return 0
    if a.cmd == "reset-jobs":
        from uol.final import overnight
        ids = [x.split("#", 1)[0].strip() for x in Path(a.ids_file).read_text(encoding="utf-8").splitlines()]
        print(json.dumps(overnight.reset_jobs(Path(a.plan), [x for x in ids if x], a.tag), indent=1))
        return 0
    if a.cmd == "overnight":
        from uol.final import overnight
        return overnight.main(Path(a.plan), drive_root=a.drive_root, ram_share=a.ram_share, max_hours=a.max_hours,
                              poll=a.poll, retry_failed=a.retry_failed,
                              local_root=Path(a.local_root) if a.local_root else None, skip_preflight=a.skip_preflight,
                              results_root=a.results_root, jobs_file=Path(a.jobs_file) if a.jobs_file else None,
                              vram_gb=a.vram_gb)
    _profile(a)
    from uol.final import context
    prof = context.active()
    if a.cmd == "systems":
        from uol.final import registry
        out = registry.write_systems()
        print(f"{out['n_systems']} systems -> {registry.SYSTEMS_JSON}")
    elif a.cmd == "populations":
        from uol.final import populations
        names = a.names.split(",") if a.names else list(populations.SPECS)
        print(json.dumps(populations.save_all(prof.store, names), indent=1, default=str))
    elif a.cmd == "run":
        from uol.final import harness
        folds = [int(x) for x in a.folds.split(",")] if a.folds else None
        for sid in a.system.split(","):
            r = harness.run(sid, a.population, folds=folds, max_rows=a.max_rows, force=a.force)
            print(json.dumps({k: r.get(k) for k in ("system", "population", "status", "ndcg_at_10", "n_cases",
                                                     "n_groups", "reason", "partial")}, default=str))
    elif a.cmd == "rebuild-stage":
        from uol.final import rebuild
        return rebuild.run_stage(a.stage)
    elif a.cmd == "g1" and a.route == "rebuild":
        from uol.final import g1
        rec = g1.rebuild_route(force=a.force)
        print((prof.store / "g1" / "G1_rebuild.md").read_text(encoding="utf-8"))
        return 0 if rec["pass"] else 2
    elif a.cmd == "g1":
        from uol.final import g1
        routes = ("cached", "recompute") if a.route == "all" else (a.route,)
        rec = g1.run(routes, force=a.force, only=a.only.split(",") if a.only else None)
        print((prof.store / "g1" / "G1.md").read_text(encoding="utf-8"))
        return 0 if rec["pass"] else 2
    elif a.cmd == "report":
        from uol.final import report
        print(report.write(out=Path(a.out) if a.out else None))
    elif a.cmd == "cost":
        from uol.final import cost
        lat = None if a.no_latency else cost.latency(a.system)
        print(json.dumps(cost.record(a.system, latency_rec=lat), indent=1, default=str))
    elif a.cmd == "cost-summary":
        from uol.final import llm_cost
        print(json.dumps(llm_cost.write_all(), indent=1, default=str))
    elif a.cmd == "contrast":
        from uol.final import harness, stats
        c = stats.contrast(prof.store, a.a, a.b, a.population)
        harness.write_json(prof.store / "contrasts" / f"{a.a}__vs__{a.b}__{a.population}.json", c)
        print(json.dumps({k: c[k] for k in ("a", "b", "population", "delta", "ci95", "class")}, default=str))
    elif a.cmd == "export-top":
        from uol.final.adapters import llm
        print(llm.export_top(prof.store, a.system, a.population, a.k))
    elif a.cmd == "plan":
        from uol.final import plan
        print(plan.write(Path(a.out), smoke=a.smoke))
    elif a.cmd == "job":
        from uol.final import overnight
        return overnight.run_job(Path(a.plan), a.id)
    elif a.cmd == "gpu-prep":
        from uol.final import gpu_prep
        print(gpu_prep.write(a.kind, a.fold, smoke=a.smoke))
    elif a.cmd == "codex-smoke":
        from uol.final import codex, harness
        rec = codex.smoke()
        harness.write_json(prof.store / "llm" / "codex_smoke.json", rec)
        print(json.dumps(rec, default=str))
        return 0 if rec["ok"] else 2
    elif a.cmd == "llm":
        from uol.final import llm_jobs
        from uol.embedding_push import common as ep
        first = (lambda rows: rows[:a.rows]) if a.rows else (lambda rows: None)
        if a.job == "llm_rerank":
            print(llm_jobs.llm_rerank(a.base, samples=a.samples, limit=a.rows, smoke=a.smoke, variant=a.variant))
        elif a.job == "agent":
            if a.population:
                from uol.final import populations
                rows = populations.load(a.population, prof.store).rows
            else:
                rows = first(llm_jobs.dev_eval_rows())
            print(llm_jobs.agent(a.condition, rows=rows))
        elif a.job == "devl_rewrite":                          # C6 (D93): DEV-L lay texts
            from uol.final import devl
            print(json.dumps(devl.rewrite(rows=first(llm_jobs.dev_eval_rows())), indent=1))
        elif a.job == "hyde":
            print(llm_jobs.hyde(rows=first(ep.dev_rows())))
        else:
            print(llm_jobs.llm_alone(rows=first(llm_jobs.dev_eval_rows())))
    elif a.cmd == "ce":
        from uol.final import rerank_jobs
        print({"prep": lambda: rerank_jobs.prep(a.fold), "fuse": rerank_jobs.fuse,
               "zs-prep": lambda: rerank_jobs.zs_prep(a.system), "zs-fuse": lambda: rerank_jobs.zs_fuse(a.system),
               "smoke-prep": lambda: rerank_jobs.smoke_inputs(a.system)}[a.step]())
    elif a.cmd == "models":
        from uol.final import rebuild
        print(json.dumps(rebuild.model_download(prof.work / "models"), default=str))
    elif a.cmd == "rebuild-copy-frozen":
        from uol.final import rebuild
        print(json.dumps(rebuild.copy_frozen_gpu_trained()))
    elif a.cmd == "variant":
        from uol.final import variants
        if a.build == "devt":
            print(json.dumps(variants.build_devt(), default=str))
        elif a.build == "devb":
            print(json.dumps(variants.build_devb(variants.variant_root("devb") / "g4.pkl"), default=str))
        elif a.encode_inputs == "devb":
            print([str(p) for p in variants.devb_encode_inputs()])
        elif a.vectors == "devb":
            print(json.dumps(variants.devb_vectors()))
        elif a.dense_queries == "devb":
            print(json.dumps(variants.devb_dense_queries(), default=str))
    elif a.cmd == "devb-labels":
        from uol.final import devb
        print(devb.run_g4())
    elif a.cmd == "families":
        from uol.final import stats
        print(json.dumps(stats.all_families(prof.store)))
    elif a.cmd == "external":
        from uol.final import external
        print(json.dumps([str(x) for x in external.fuse(a.kind)]))
    elif a.cmd == "merge-store":
        from uol.final import merge
        print(json.dumps(merge.merge_store(Path(a.src), prof.store), indent=1))
    elif a.cmd == "pack":
        from uol.final import pack
        print(pack.write())
    elif a.cmd == "features":
        from uol.final.adapters import ltr
        ctx, F = ltr.dev_feature_table()
        print(json.dumps({"pairs": int(len(F.pq)), "columns": len(F.cols)}))
    elif a.cmd == "ingest":
        from uol.final.adapters import gpu
        inputs = prof.work / "gpu_inputs" / a.key                     # prepared on the run computer
        if not inputs.exists() and prof.data_dir:
            inputs = prof.data_dir / "gpu_inputs" / a.key              # prepared at bundle time
        print(gpu.ingest(a.key, inputs, prof.work / "gpu_jobs" / a.key / "outputs"))
    elif a.cmd == "gate-e0":
        from uol.final import harness
        from uol.final.adapters import gpu
        base = prof.data_dir / "gpu_inputs" if prof.kind == "bundle" else prof.work / "gpu_inputs"
        rec = gpu.gate_e0(base / "qwen3_8b_gate", prof.work / "gpu_jobs" / "qwen3_8b_gate" / "outputs",
                          base / "qwen3_8b_gate" / "gate_e0_reference.npz")
        harness.write_json(prof.store / "gpu" / "gate_e0.json", rec)
        print(json.dumps(rec))
        return 0 if rec["pass"] else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
