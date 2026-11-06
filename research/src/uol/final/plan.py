"""The overnight DAG (``jobs/plan.json``) of RUN_PLAN v2 §4 and §7, and the post-freeze plan (``plan_new_cases.json``).

Three queues run side by side (``uol.final.overnight``):
* **CPU**: admitted while the declared threads fit the logical cores (32) and the declared peak RAM fits 72 GB; every
  job's thread cap is exported (OMP, MKL, OpenBLAS, LightGBM / HGB through OMP, ``UOL_FINAL_THREADS``);
* **GPU**: admitted by declared VRAM on the 16 GB card, so 2-3 small encoders share it; ``gpu_exclusive`` jobs (8B int8,
  bge_m3_ft, ce_trained, the agent's embedding tool) run alone; priority order 1-6 of §7;
* **LLM**: Codex jobs (``kind: llm``), at most ``UOL_FINAL_LLM_JOBS`` (default 2) at a time, each with adaptive
  concurrency 6 -> 12 (``codex.run_adaptive``); they start as soon as their inputs exist; all depend on ``codex_smoke``.

Profiles per job (``env``): ``UOL_FINAL_LABELS`` (v11 for G1, corrected otherwise), ``UOL_FINAL_SOURCES`` (frozen copies
or ``rebuilt`` from code), ``UOL_FINAL_VARIANT`` (devt, devb, devn). Gates: ``g1`` blocks everything; ``g1_rebuild``
blocks the jobs on rebuilt sources (their frozen-source twins still run); ``gate_e0`` and ``fidelity_int8`` block the
int8 query encodes; ``codex_smoke`` blocks the LLM block. ``experiment`` names the RUN_PLAN experiment (E1-E13).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from uol.final import registry

UF = ["{python}", "-m", "uol.final", "--data", "{data}", "--store", "{store}"]
V11 = {"UOL_FINAL_LABELS": "v11"}
RB = {"UOL_FINAL_SOURCES": "rebuilt"}
RB11 = {**RB, **V11}
ROBUST_SYSTEMS = ("popularity", "lstar_headings", "bm25_stem", "qwen3_8b", "lsa", "S0", "S0+qwen3_4b", "S_full",
                  "S_full_ext")
COST_SYSTEMS = ("lstar_headings", "bm25_stem", "lsa", "lda_sim", "lda_dm", "doc2vec", "qwen3_0p6b", "qwen3_4b", "qwen3_8b",
                "bge_m3_colbert",
                "qwen3_8b_mrl256_int8", "qwen3_8b_binary", "rrf__lstar_headings__qwen3_0p6b", "linear__lstar_headings__qwen3_0p6b",
                "S0", "S0+qwen3_4b", "S0+qwen3_0p6b", "S_full", "S_full_ext",
                "qwen3_0p6b_ft")                                                                      # D89
EXPERIMENT_OF_BLOCK = {"A": "E1", "B": "E4", "C": "E5", "D": "E6", "E": "E9", "G": "E11", "G1": "G1",
                       "C2": "E5", "C5": "E9", "C9": "E1", "C10": "E9"}    # D93, D101 additions (ANALYSIS_PLAN_C)
RANKER_ADAPTERS = ("ltr",)
# D77 (22 Oct 2026, the author): g1_rebuild failed narrowly (46/47 CPU sources exact; src_knn_lsaadapt_spec .000264 off; the
# rebuilt NO-EMB ranker .4031 vs .4081). The gate stays as it is and keeps gating the @rebuilt twins (their report is
# the reproducibility evidence); everything downstream of the answer rankers (LLM rerank, cross-encoder, zero-shot
# rerankers, exports, cost) uses the frozen-source rankers run__S0__DEV / run__S_full_ext__DEV, which reproduce G1.
DOWNSTREAM_ENV: dict = {}                 # frozen sources, corrected labels (no UOL_FINAL_SOURCES=rebuilt)


def _job(jid: str, exp: str, argv: list[str], *, deps: list[str] | None = None, ram: float = 4, threads: int = 2,
         hours: float = 0.2, kind: str = "cpu", env: dict | None = None, **extra: Any) -> dict[str, Any]:
    return {"id": jid, "kind": kind, "experiment": exp, "argv": argv, "deps": deps or [], "peak_ram_gb": float(ram),
            "threads": int(threads), "est_hours": float(hours), "env": env or {}, **extra}


def _run(sid: str, pop: str, exp: str, deps: list[str], spec: dict[str, Any], env: dict, suffix: str = "") -> dict[str, Any]:
    hours = {"loaded": 0.05, "rebuilt": 0.3, "new": 0.6, "gpu_pending": 0.2}.get(spec["mode"], 0.3)
    if spec["adapter"] == "ltr":
        hours = 0.4                       # one refit of five folds took about 6 min on the thesis PC (D58 control)
        if sid.startswith(("S0", "S_full")):
            hours = 1.3                   # with their sources' features: 54-95 min each on the run computer (23 Oct)
    if spec["adapter"] == "topic":
        hours = 1.5
    ram = float(spec.get("peak_ram_gb", 6)) * (1.6 if pop == "DEV-ALL" else 1.0)
    tag = "@rebuilt" if env.get("UOL_FINAL_SOURCES") == "rebuilt" else ""
    return _job(f"run__{sid}{tag}__{pop}{suffix}", exp, UF + ["run", "--system", sid, "--population", pop], deps=deps,
                ram=ram, threads=int(spec.get("threads", 4)), hours=hours, env=env,
                outputs=[f"store:runs/{sid}{tag}/{pop}", f"store:receipts/{sid}{tag}__{pop}.json"])


def gpu_jobs() -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Encoders (smoke + full + ingest), Gate E0, the int8 query encodes, bge_m3_ft and ce_trained."""
    from uol.final.adapters import gpu
    jobs, ready = [], {}
    specs = gpu.specs()
    env = {"UOL_WORKER_SCRATCH": "{work}/gpu_scratch", "PYTHONPATH": "{code_src}"}
    for key in gpu.ORDER:
        s = specs[key]
        script = f"{{code_src}}/{'uol/round3/gpu_jobs/legal_encode.py' if s['script'] == 'legal_encode' else 'uol/final/gpu_jobs/' + s['script'] + '.py'}"
        vram = gpu.VRAM_GB.get(key, 8)
        excl = vram >= 12
        deps = ["g1"] + (["gate_e0", "fidelity_int8"] if key in ("euler_legal", "qwen3_8b_int8path") else [])
        models = [{"repo": s["model"], "revision": s["revision"], "licence": s.get("licence")}]
        tag = (["--store-tag", gpu.STORE_TAGS[key]] if key in gpu.STORE_TAGS else [])
        sm = _job(f"gpu_smoke__{key}", "E1/E2", ["{python}", script, "--inputs", f"{{data}}/gpu_inputs/{key}__smoke",
                                                "--out", f"{{work}}/gpu_jobs/{key}__smoke/outputs"] + tag,
                  deps=deps, ram=8, threads=2, hours=0.1, kind="gpu", vram_gb=vram, gpu_exclusive=excl, env=env,
                  priority=gpu.PRIORITY.get(key, 3), models=models)
        full = _job(f"gpu__{key}", "E1/E2", ["{python}", script, "--inputs", f"{{data}}/gpu_inputs/{key}",
                                            "--out", f"{{work}}/gpu_jobs/{key}/outputs"] + tag,
                    deps=[sm["id"]] + (["gpu__jina_v3_docs"] if key == "jina_v3" else []), ram=12, threads=4,
                    hours=gpu.GPU_HOURS.get(key, 1.0), kind="gpu", vram_gb=vram, gpu_exclusive=excl, env=env,
                    priority=gpu.PRIORITY.get(key, 3), models=models,
                    fallback={"batch_tokens": max(2048, int(s["batch_tokens"]) // 4)},
                    outputs=[f"work:gpu_jobs/{key}/outputs"])
        jobs += [sm, full]
        if key == "qwen3_8b_gate":
            jobs.append(_job("gate_e0", "fidelity", UF + ["gate-e0"], deps=[full["id"]], ram=4, threads=1, hours=0.05,
                             outputs=["store:gpu/gate_e0.json"]))
            continue
        ing = _job(f"ingest__{key}", "E1", UF + ["ingest", "--key", key], deps=[full["id"]], ram=10, threads=2,
                   hours=0.1, outputs=[f"data:vectors_new/{key}"])
        jobs.append(ing)
        ready[key] = ing["id"]
    # int8 query-side encodes (Qwen3-8B int8 path), behind Gate E0; the fidelity check gates the dependants
    q8 = {"UOL_WORKER_SCRATCH": "{work}/gpu_scratch", "PYTHONPATH": "{code_src}"}
    LE = "{code_src}/uol/round3/gpu_jobs/legal_encode.py"

    def int8(key: str, deps: list[str], prep_env: dict | None = None, prep_deps: list[str] | None = None) -> None:
        jobs.append(_job(f"prep__{key}", "E8", UF + ["gpu-prep", "--kind", key], deps=(prep_deps or []) + ["g1"], ram=6,
                         threads=2, hours=0.05, env=prep_env or {}))
        jobs.append(_job(f"gpu__{key}", "E8", ["{python}", LE, "--inputs", f"{{work}}/gpu_inputs/{key}", "--out",
                                               f"{{work}}/gpu_jobs/{key}/outputs"], deps=[f"prep__{key}"] + deps,
                         ram=12, threads=4, hours=0.3, kind="gpu", vram_gb=16, gpu_exclusive=True, env=q8, priority=3,
                         models=[{"repo": "Qwen/Qwen3-Embedding-8B", "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"}]))
        jobs.append(_job(f"ingest__{key}", "E8", UF + ["ingest", "--key", key], deps=[f"gpu__{key}"], ram=6, threads=2,
                         hours=0.05, env=prep_env or {}))
        ready[key] = f"ingest__{key}"
    int8("qwen_int8_dev", ["gate_e0"])
    jobs.append(_job("fidelity_int8", "fidelity", UF + ["run", "--system", "qwen3_8b_int8q", "--population", "DEV"],
                     deps=["ingest__qwen_int8_dev"], ram=8, threads=4, hours=0.1, gate_check="qwen3_8b_int8q within .003 of .2742"))
    int8("query2doc", ["fidelity_int8"])
    int8("hyde", ["fidelity_int8"], prep_deps=["llm__hyde"])
    # bge_m3_ft, one model per fold (priority 4, alone on the card)
    for k in range(5):
        jobs.append(_job(f"prep__bge_m3_ft_f{k}", "E2", UF + ["gpu-prep", "--kind", "bge_m3_ft", "--fold", str(k)],
                         deps=["g1"], ram=8, threads=2, hours=0.1))
        jobs.append(_job(f"gpu__bge_m3_ft_f{k}", "E2", ["{python}", "{code_src}/uol/final/gpu_jobs/bge_m3_ft.py", "--inputs",
                                                        f"{{work}}/gpu_inputs/bge_m3_ft_f{k}", "--out",
                                                        f"{{work}}/gpu_jobs/bge_m3_ft_f{k}/outputs"],
                         deps=[f"prep__bge_m3_ft_f{k}"], ram=16, threads=4, hours=0.8, kind="gpu", vram_gb=16,
                         gpu_exclusive=True, priority=4, env=env,
                         models=[{"repo": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}]))
        jobs.append(_job(f"ingest__bge_m3_ft_f{k}", "E2", UF + ["ingest", "--key", f"bge_m3_ft_f{k}"],
                         deps=[f"gpu__bge_m3_ft_f{k}"], ram=8, threads=2, hours=0.05))
    # a fold-model system needs all five folds ingested (23 Oct: it waited for fold 4 only)
    ready["bge_m3_ft"] = [f"ingest__bge_m3_ft_f{k}" for k in range(5)]
    # D89: Qwen3-Embedding-0.6B fully fine-tuned per fold with the bge_m3_ft recipe and script; its inputs need the
    # base qwen3_0p6b run (the query variant each fold chose) and the base vectors (hard negatives)
    for k in range(5):
        jobs.append(_job(f"prep__qwen3_0p6b_ft_f{k}", "E2", UF + ["gpu-prep", "--kind", "qwen3_0p6b_ft", "--fold", str(k)],
                         deps=["g1", "run__qwen3_0p6b__DEV"], ram=10, threads=2, hours=0.1))
        jobs.append(_job(f"gpu__qwen3_0p6b_ft_f{k}", "E2", ["{python}", "{code_src}/uol/final/gpu_jobs/bge_m3_ft.py",
                                                            "--inputs", f"{{work}}/gpu_inputs/qwen3_0p6b_ft_f{k}", "--out",
                                                            f"{{work}}/gpu_jobs/qwen3_0p6b_ft_f{k}/outputs"],
                         deps=[f"prep__qwen3_0p6b_ft_f{k}"], ram=16, threads=4, hours=0.8, kind="gpu", vram_gb=16,
                         gpu_exclusive=True, priority=4, env=env,
                         models=[{"repo": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"}]))
        jobs.append(_job(f"ingest__qwen3_0p6b_ft_f{k}", "E2", UF + ["ingest", "--key", f"qwen3_0p6b_ft_f{k}"],
                         deps=[f"gpu__qwen3_0p6b_ft_f{k}"], ram=8, threads=2, hours=0.05))
    ready["qwen3_0p6b_ft"] = [f"ingest__qwen3_0p6b_ft_f{k}" for k in range(5)]
    # D79: late interaction over the whole catalogue (frozen colbert_bge recipe), alone on the card; writes the score
    # file the bge_m3_colbert system reads (no separate ingest)
    jobs.append(_job("gpu__colbert_full", "E1", ["{python}", "{code_src}/uol/final/gpu_jobs/colbert_full.py", "--out",
                                                 "{work}/gpu_jobs/colbert_full/outputs"],
                     deps=["g1", "models"], ram=24, threads=8, hours=1.5, kind="gpu", vram_gb=16, gpu_exclusive=True,
                     priority=4, env=env, disk_gb=22,
                     models=[{"repo": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181"}],
                     outputs=["data:vectors_new/bge_m3_colbert", "work:gpu_jobs/colbert_full/outputs"],
                     note="token store 17.5 GB fp16 + queries 1.5 GB in <work>/colbert_full/store; fidelity gate on the "
                          "frozen colbert_bge pairs"))
    ready["bge_m3_colbert"] = "gpu__colbert_full"
    return jobs, ready


DEVN_SYSTEMS = {"qwen3_8b_vectors": "qwen8b", "qwen3_0p6b": "qwen3_0p6b", "qwen3_4b": "qwen3_4b",
                "e5_large_legal_en": "e5_large_legal", "e5_small_re": "e5_small_re", "qwen3_8b_int8_base": "qwen3_8b_int8path",
                "bge_m3_base": "bge_m3_base", "e5_base": "e5_base", "e5_large": "e5_large", "e5_large_instruct": "e5_large_instruct",
                "gte_multilingual": "gte_multilingual", "jina_v3": "jina_v3", "arctic_l_v2": "arctic_l_v2",
                "alia_mrbert": "alia_mrbert", "gemma_legal": "gemma_legal", "bge_m3_legal": "bge_m3_legal",
                "euler_legal": "euler_legal"}


def devn_encode_jobs(ready: dict[str, str], v: str = "devn", targets: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    """DEV-N: the 854 clause-dropped queries re-encoded by every dense model (queries only, plus 50 controls). The
    same jobs for DEV-L (``v`` "devl", C6): the lay-worded DEV facts, after the rewrite job."""
    from uol.final.adapters import gpu
    specs = gpu.specs()
    env_v = {"UOL_FINAL_VARIANT": v}
    genv = {"UOL_WORKER_SCRATCH": "{work}/gpu_scratch", "PYTHONPATH": "{code_src}"}
    jobs = []
    for target, skey in gpu.DEVN_ENCODERS.items():
        if targets is not None and target not in targets:
            continue
        s = specs[skey]
        script = "{code_src}/uol/round3/gpu_jobs/legal_encode.py" if s["script"] == "legal_encode" else \
            f"{{code_src}}/uol/final/gpu_jobs/{s['script']}.py"
        vram = gpu.VRAM_GB.get(skey, 8)
        pre_deps = ["g1"] + ([ready[target]] if target in ready else []) + (["fidelity_int8"] if s["precision"] == "int8" else []) \
            + (["llm__devl_rewrite"] if v == "devl" else [])
        jobs.append(_job(f"prep__{v}__{target}", "E11", UF + ["gpu-prep", "--kind", f"{v}__{target}"], deps=pre_deps,
                         ram=6, threads=2, hours=0.05, env=env_v))
        argv = ["{python}", script, "--inputs", f"{{work}}/variants/{v}/gpu_inputs/{v}__{target}", "--out",
                f"{{work}}/variants/{v}/gpu_jobs/{v}__{target}/outputs"] + \
            (["--store-tag", gpu.STORE_TAGS[target]] if target in gpu.STORE_TAGS else [])
        models = [{"repo": s["model"], "revision": s["revision"]}]
        if s.get("device") == "cpu":            # same job id and outputs; fp32 on the CPU (qwen3_4b_q, 22 Oct 2026)
            # run_last (RUN-B, 23 Oct): a check, not the main result; it holds 28 GB and 16 cores for 1-2 h, so it starts
            # after every other job except its own dependants; hours from the run computer's DEV-N encode (100 min)
            jobs.append(_job(f"gpu__{v}__{target}", "E11", argv + ["--device", "cpu"], deps=[f"prep__{v}__{target}"],
                             ram=float(s.get("cpu_ram_gb", 24)), threads=int(s.get("cpu_threads", 16)),
                             hours=1.7 if v == "devn" else 1.0, kind="cpu", env=genv, models=models,
                             note=s.get("device_reason", ""), run_last=True, priority=9))
        else:
            jobs.append(_job(f"gpu__{v}__{target}", "E11", argv, deps=[f"prep__{v}__{target}"], ram=10, threads=4,
                             hours=0.1, kind="gpu", vram_gb=vram, gpu_exclusive=vram >= 12, priority=3, env=genv,
                             models=models))
        jobs.append(_job(f"ingest__{v}__{target}", "E11", UF + ["ingest", "--key", f"{v}__{target}"],
                         deps=[f"gpu__{v}__{target}"], ram=6, threads=2, hours=0.05, env=env_v))
    return jobs


def rebuild_chain(prefix: str, env: dict, deps: list[str], gpu_stages: bool = True) -> list[dict[str, Any]]:
    from uol.final import rebuild
    out = []
    for jid, module, args, kind, ram, threads, hours, d in rebuild.STAGES:
        if kind == "gpu" and not gpu_stages:
            continue
        spec = dict(ram=ram, threads=threads, hours=hours)
        if kind == "gpu":
            spec.update(kind="gpu", vram_gb=8, gpu_exclusive=False, priority=3)
        extra = ["models"] if module in ("finetune_dense", "case_sim") and gpu_stages else []
        out.append(_job(f"{prefix}{jid}", "sources", UF + ["rebuild-stage", module] + args,
                        deps=deps + extra + [f"{prefix}{x}" for x in d], env=env, **spec))
    if not gpu_stages:
        out.append(_job(f"{prefix}rb_copy_gpu_trained", "sources", UF + ["rebuild-copy-frozen"], deps=deps, env=env,
                        ram=4, threads=1, hours=0.05,
                        note="GPU-trained sources (ft_e5_q, case-similarity keys) copied from the frozen run in this chain"))
    return out


def build(smoke: bool = False) -> dict[str, Any]:
    if smoke:
        return smoke_plan()
    S = registry.systems()
    jobs: list[dict[str, Any]] = []
    # ---- gates
    jobs.append(_job("features_v11", "G1", UF + ["features"], ram=14, threads=8, hours=0.6, env=V11))
    jobs.append(_job("g1", "G1", UF + ["g1", "--route", "all"], deps=["features_v11"], ram=20, threads=8, hours=1.0,
                     gate=True, env=V11, outputs=["store:g1"]))
    jobs += rebuild_chain("v11_", RB11, ["g1"], gpu_stages=False)
    last_v11 = [j["id"] for j in jobs if j["id"].startswith("v11_")]
    jobs.append(_job("features_rb_v11", "G1", UF + ["features"], deps=last_v11, ram=14, threads=8, hours=0.6, env=RB11))
    jobs.append(_job("g1_rebuild", "G1", UF + ["g1", "--route", "rebuild"], deps=["features_rb_v11"], ram=20, threads=8,
                     hours=0.8, env=RB11, outputs=["store:g1/G1_rebuild.json"], soft_gate=True,
                     note="soft gate: a FAIL shows in STATUS gate_failed and blocks only its @rebuilt dependants (D77)"))
    jobs.append(_job("codex_smoke", "E9", UF + ["codex-smoke"], deps=["g1"], ram=1, threads=1, hours=0.02, kind="llm",
                     gate_check="gpt-5.6-luna answers valid JSON"))
    # ---- main chain on corrected labels: rebuilt sources and their pair table
    jobs.append(_job("features", "E5", UF + ["features"], deps=["g1"], ram=14, threads=8, hours=0.6))
    jobs.append(_job("models", "sources", UF + ["models"], deps=["g1"], ram=4, threads=2, hours=0.2,
                     models=[{"repo": r, "revision": v} for r, v, _ in __import__("uol.final.rebuild", fromlist=["x"]).MODELS.values()],
                     note="pinned snapshots for the GPU-trained rebuild stages (ft_e5_q, case-similarity keys)"))
    jobs += rebuild_chain("", RB, ["g1"], gpu_stages=True)
    main_rb = [j["id"] for j in jobs if j["id"].startswith("rb_")]
    jobs.append(_job("features_rb", "E5", UF + ["features"], deps=main_rb + ["g1_rebuild"], ram=14, threads=8, hours=0.6,
                     env=RB, soft_gate_ok=["g1_rebuild"]))           # D93: needed by the S_full_ext seeds
    gjobs, ready = gpu_jobs()
    jobs += gjobs

    def deps_for(spec: dict[str, Any], rebuilt: bool) -> list[str]:
        # the E2 adapter reads the development pair table that the "features" job builds (21 Oct 2026)
        d = ["g1"] + (["features_rb", "g1_rebuild"] if rebuilt else
                      (["features"] if spec["adapter"] in ("ltr", "qwen_adapter") else []))
        a = spec.get("adapter_args", {})
        for x in [spec["id"]] + list(a.get("members", [])) + list(a.get("extra", [])):
            key = S.get(x, {}).get("adapter_args", {}).get("key") if x in S else None
            for k in (x, key):
                if k in ready:
                    d.extend(ready[k] if isinstance(ready[k], list) else [ready[k]])
            if x in registry.TOPIC + registry.WORD_VECTORS and x != spec["id"]:
                d.append(f"run__{x}__DEV")
        if spec["adapter"] == "llm_matrix":
            key = a.get("key", "")
            base = key[:-len("_alone")] if key.endswith("_alone") else key
            d.append("ce_fuse" if base == "ce_trained" else (f"zs_fuse__{base}" if base.startswith("rerank_") else f"llm__{key}"))
        if spec["id"] == "hyde":
            d.append("ingest__hyde")
        if spec["id"] == "qwen3_8b_int8q":
            d.append("ingest__qwen_int8_dev")
        return sorted(set(d))

    runnable = {sid: s for sid, s in S.items() if s["adapter"] not in ("not_wired", "llm_export")}
    for sid, s in runnable.items():
        blocks = [b for b in s["blocks"] if b not in ("G1",)]
        if not blocks or set(s["blocks"]) <= {"G"} or sid.startswith(("agent_", "qwen3_8b_devn")) or "C10" in blocks:
            continue
        exp = next((b for b in blocks if b.startswith("E") and b[1:].isdigit()), EXPERIMENT_OF_BLOCK.get(blocks[0], "E1"))
        uses_sources = s["adapter"] in RANKER_ADAPTERS or sid.startswith("src_") or sid.startswith("rrf_all")
        jobs.append(_run(sid, "DEV", exp, deps_for(s, False), s, {}))
        if uses_sources:
            jobs.append(_run(sid, "DEV", exp, deps_for(s, True), s, RB))
    for sid, s in runnable.items():
        if set(s["blocks"]) == {"G"} and not sid.startswith("qwen3_8b_devn"):                 # seeds
            jobs.append(_run(sid, "DEV", "E11", deps_for(s, True), s, RB))
            if sid.startswith("S_full_ext_seed"):
                # D93: the S_full_ext seeds run on the rebuilt sources although g1_rebuild failed (its verdict
                # stands); every other gated row stays blocked by the soft gate
                jobs[-1]["soft_gate_ok"] = ["g1_rebuild"]
    # ---- robustness populations (E11)
    for sid in ROBUST_SYSTEMS:
        if sid not in runnable:
            continue
        s = runnable[sid]
        env = RB if s["adapter"] == "ltr" else {}
        for pop in ("DEV-G", "DEV-ALL"):
            jobs.append(_run(sid, pop, "E11", deps_for(s, bool(env)), s, env))
            if env:
                # D106 (ANALYSIS_PLAN_C v5): the four rankers on DEV-G / DEV-ALL run on the rebuilt sources although
                # g1_rebuild failed, as the D93 seeds do; every other gated row stays blocked by the soft gate
                jobs[-1]["soft_gate_ok"] = ["g1_rebuild"]
    jobs += variant_jobs(runnable, deps_for, ready)
    # ---- LLM queue (E9, E12) and the reranker (E9)
    llm = {"llm__llm_alone": (UF + ["llm", "--job", "llm_alone"], ["codex_smoke"]),
           "llm__hyde": (UF + ["llm", "--job", "hyde"], ["codex_smoke"]),
           "llm__llm_rerank_S0": (UF + ["llm", "--job", "llm_rerank", "--base", "S0"], ["codex_smoke", "run__S0__DEV"]),
           "llm__llm_rerank_S_full_ext": (UF + ["llm", "--job", "llm_rerank", "--base", "S_full_ext"],
                                          ["codex_smoke", "run__S_full_ext__DEV"]),
           "llm__agent_words": (UF + ["llm", "--job", "agent", "--condition", "words"], ["codex_smoke"]),
           # C5 (D93): the blank-text control of P6, one sample (fused per fold like the main rerank)
           "llm__llm_rerank_S_full_ext_blank": (UF + ["llm", "--job", "llm_rerank", "--base", "S_full_ext", "--variant",
                                                      "blank", "--samples", "1"],
                                                ["codex_smoke", "run__S_full_ext__DEV"])}
    for jid, (argv, deps) in llm.items():
        jobs.append(_job(jid, "E12" if "agent" in jid else "E9", argv, deps=deps, ram=6, threads=2,
                         hours=1.5 if jid.endswith("_blank") else 4.0, kind="llm",
                         env=dict(DOWNSTREAM_ENV) if "rerank" in jid else {}))
    # D96: the embedding-tool agent on the fixed DEV-A300 sample (agent_words keeps its DEV run, read on the same rows)
    jobs.append(_job("llm__agent_embedding", "E12", UF + ["llm", "--job", "agent", "--condition", "embedding",
                                                         "--population", "DEV-A300"],
                     deps=["codex_smoke", "fidelity_int8"], ram=20, threads=2, hours=1.5, kind="gpu", vram_gb=16,
                     gpu_exclusive=True, priority=9, uses_llm=True))
    for k in range(5):
        jobs.append(_job(f"ce_prep_f{k}", "E9", UF + ["ce", "--step", "prep", "--fold", str(k)],
                         deps=["run__S_full_ext__DEV"], ram=20, threads=8, hours=0.5, env=dict(DOWNSTREAM_ENV)))
        jobs.append(_job(f"gpu__ce_trained_f{k}", "E9", ["{python}", "{code_src}/uol/round3/gpu_jobs/r2_train.py", "--inputs",
                                                         f"{{work}}/round3_run/arms/R2/inputs/ce_trained_f{k}_s1", "--out",
                                                         f"{{work}}/gpu_jobs/ce_trained_f{k}_s1/outputs"],
                         deps=[f"ce_prep_f{k}"], ram=16, threads=4, hours=1.1, kind="gpu", vram_gb=16, gpu_exclusive=True,
                         priority=6, env={"UOL_WORKER_SCRATCH": "{work}/gpu_scratch", "PYTHONPATH": "{code_src}"},
                         models=[{"repo": "Qwen/Qwen3-Reranker-0.6B", "revision": "e61197ed45024b0ed8a2d74b80b4d909f1255473"}]))
    jobs.append(_job("ce_fuse", "E9", UF + ["ce", "--step", "fuse"], deps=[f"gpu__ce_trained_f{k}" for k in range(5)],
                     ram=12, threads=4, hours=0.2, env=dict(DOWNSTREAM_ENV)))
    for sid, vram in (("rerank_bge", 5), ("rerank_qwen3_4b", 14)):
        jobs.append(_job(f"zs_prep__{sid}", "E9", UF + ["ce", "--step", "zs-prep", "--system", sid],
                         deps=["run__S_full_ext__DEV"], ram=8, threads=2, hours=0.1, env=dict(DOWNSTREAM_ENV)))
        jobs.append(_job(f"gpu__{sid}", "E9", ["{python}", "{code_src}/uol/round3/gpu_jobs/r1_score.py", "--inputs",
                                               f"{{work}}/gpu_inputs/{sid}", "--out", f"{{work}}/gpu_jobs/{sid}/outputs"],
                         deps=[f"zs_prep__{sid}"], ram=12, threads=4, hours=0.6 if "qwen" in sid else 0.3, kind="gpu",
                         vram_gb=vram, gpu_exclusive=vram >= 12, priority=6,
                         env={"UOL_WORKER_SCRATCH": "{work}/gpu_scratch", "PYTHONPATH": "{code_src}"},
                         models=[{"repo": "pinned in round3/gpu_jobs/rerank_bench.MODELS", "revision": "pinned"}]))
        jobs.append(_job(f"zs_fuse__{sid}", "E9", UF + ["ce", "--step", "zs-fuse", "--system", sid], deps=[f"gpu__{sid}"],
                         ram=12, threads=4, hours=0.1, env=dict(DOWNSTREAM_ENV)))
    for sid, pop in (("agent_words", "DEV"), ("agent_embedding", "DEV-A300")):
        jobs.append(_run(sid, pop, "E12", [f"llm__{sid}"], runnable[sid], {}))
    # C10 (D101): the thesis PC's Claude orders and Jev probabilities (UOL_RUN_B/inputs -> <work>/external) fused here
    for kind, sids in (("claude", ("claude_rerank_S_full_ext",)), ("jev", ("jev_S_full_ext", "jev_rrf_S_full_ext"))):
        jobs.append(_job(f"ext__{kind}", "E9", UF + ["external", "--kind", kind], deps=["run__S_full_ext__DEV"], ram=12,
                         threads=4, hours=0.2, env=dict(DOWNSTREAM_ENV),
                         note="needs <work>/external/" + {"claude": "claude_orders_s0.json", "jev": "jev_p_yes.jsonl"}[kind]))
        for sid in sids:
            if sid in runnable:
                jobs.append(_run(sid, "DEV-A300", "E9", [f"ext__{kind}"], runnable[sid], {}))
    # ---- cost (E10): latency measured with nothing else on the CPU
    for sid in COST_SYSTEMS:
        if sid in runnable:                  # rankers: the frozen-source runs (D77), like every other system
            # late interaction: index size, encode time and query latency come from the GPU job's run.json
            nolat = ["--no-latency"] if sid == "bge_m3_colbert" else []
            jobs.append(_job(f"cost__{sid}", "E10", UF + ["cost", "--system", sid] + nolat, deps=[f"run__{sid}__DEV"],
                             ram=float(runnable[sid].get("peak_ram_gb", 6)), threads=8, hours=0.2, exclusive=True,
                             env=dict(DOWNSTREAM_ENV), outputs=[f"store:cost/{sid}.json"]))
    # ---- exports for later and the thesis pack
    for sid in ("S_full_ext", "S0"):
        jobs.append(_job(f"export__{sid}", "E9", UF + ["export-top", "--system", sid, "--k", "30"],
                         deps=[f"run__{sid}__DEV"], ram=2, threads=1, hours=0.02, env=dict(DOWNSTREAM_ENV)))
    jobs.append(_job("contrasts", "stats", UF + ["families"], deps=[], ram=8, threads=4, hours=0.3, final=True))
    # LLM calls / tokens / seconds per DEV case of every LLM system (none has a cost job) and the fine-tunes' GPU-hours,
    # into <store>/cost/<system>.json; final, so it runs after every LLM job whatever their outcome, before the pack
    # (22 Oct 2026). It records failures per system and does not fail itself.
    jobs.append(_job("cost_summary", "E10", UF + ["cost-summary"], deps=[], ram=2, threads=1, hours=0.05, final=True,
                     outputs=["store:llm/cost_summary.json"]))
    jobs.append(_job("thesis_pack", "pack", UF + ["pack"], deps=["contrasts", "cost_summary"], ram=8, threads=4, hours=0.3,
                     final=True))
    jobs = smoke_stage(jobs, S)
    ids = [j["id"] for j in jobs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate job ids: {sorted(dup)[:5]}")
    known = set(ids)
    for j in jobs:
        j["deps"] = [d for d in dict.fromkeys(j["deps"]) if d in known]
    return {"plan": "uol.final overnight DAG (RUN_PLAN v2)", "n_jobs": len(jobs), "gates": ["g1", "g1_rebuild", "gate_e0",
            "fidelity_int8", "codex_smoke"], "queues": {"cpu_ram_gb": 72, "cpu_threads": "logical cores",
            "gpu_vram_gb": 16, "llm_jobs": 2, "llm_concurrency": "adaptive 6-12"},
            "not_in_plan": {"E13": "new cases and TEST-R: jobs/plan_new_cases.json, after the freeze"}, "jobs": jobs}


def smoke_stage(jobs: list[dict[str, Any]], S: dict[str, dict]) -> list[dict[str, Any]]:
    """The first stage (coordinator, 21 Oct): every job kind runs once on a tiny slice (one fold, a few cases) into a
    separate store and work folder (``{store}_smoke``, ``{work}_smoke``), and the full jobs of a kind depend on that
    kind's smoke passing. GPU encoders keep their own per-model smoke jobs; Gate E0 is the int8 path's smoke."""
    US = ["{python}", "-m", "uol.final", "--data", "{data}", "--store", "{store}_smoke"]
    SW = {"UOL_FINAL_WORK": "{work}_smoke"}
    one = ["--population", "DEV", "--folds", "0", "--max-rows", "20"]
    genv = {"UOL_WORKER_SCRATCH": "{work}_smoke/gpu_scratch", "PYTHONPATH": "{code_src}"}
    sm = [
        _job("smoke__run", "smoke", US + ["run", "--system", "bm25_stem"] + one, deps=["g1"], ram=8, threads=4, hours=0.1),
        _job("smoke__topic", "smoke", US + ["run", "--system", "lsa"] + one, deps=["g1"], ram=12, threads=8, hours=0.3),
        _job("smoke__dense", "smoke", US + ["run", "--system", "qwen3_0p6b"] + one, deps=["g1"], ram=8, threads=4, hours=0.1),
        _job("smoke__fusion", "smoke", US + ["run", "--system", "rrf__lstar_headings__qwen3_0p6b"] + one, deps=["g1"],
             ram=12, threads=4, hours=0.2),
        _job("smoke__ltr", "smoke", US + ["run", "--system", "ltr_noemb_refit"] + one, deps=["g1", "features"], ram=18,
             threads=4, hours=0.2),
        _job("smoke__rebuild", "smoke", US + ["rebuild-stage", "base", "all"], deps=["g1"], ram=12, threads=8, hours=0.3,
             env={**RB, **SW}, note="the first original stage, complete, in a separate work folder"),
        _job("smoke__llm", "smoke", US + ["llm", "--job", "llm_alone", "--rows", "3"], deps=["codex_smoke"], ram=4,
             threads=1, hours=0.1, kind="llm", env=SW),
        _job("smoke__hyde", "smoke", US + ["llm", "--job", "hyde", "--rows", "20"], deps=["codex_smoke"], ram=4, threads=1,
             hours=0.1, kind="llm", env=SW),
        _job("smoke__llm_rerank", "smoke", US + ["llm", "--job", "llm_rerank", "--base", "bm25_stem", "--rows", "2",
                                                 "--samples", "1", "--smoke"], deps=["codex_smoke", "smoke__run"], ram=4,
             threads=1, hours=0.1, kind="llm", env=SW),
        _job("smoke__agent_words", "smoke", US + ["llm", "--job", "agent", "--condition", "words", "--rows", "2"],
             deps=["codex_smoke"], ram=6, threads=1, hours=0.2, kind="llm", env=SW),
        _job("smoke__agent_embedding", "smoke", US + ["llm", "--job", "agent", "--condition", "embedding", "--rows", "2"],
             deps=["codex_smoke", "fidelity_int8"], ram=20, threads=2, hours=0.3, kind="gpu", vram_gb=16, gpu_exclusive=True,
             uses_llm=True, priority=2, env=SW),
        _job("smoke__prep_bge_m3_ft", "smoke", US + ["gpu-prep", "--kind", "bge_m3_ft", "--fold", "0", "--smoke"],
             deps=["g1"], ram=8, threads=2, hours=0.1, env=SW),
        _job("smoke__bge_m3_ft", "smoke", ["{python}", "{code_src}/uol/final/gpu_jobs/bge_m3_ft.py", "--inputs",
                                           "{work}_smoke/gpu_inputs/bge_m3_ft_smoke", "--out",
                                           "{work}_smoke/gpu_jobs/bge_m3_ft_smoke/outputs"],
             deps=["smoke__prep_bge_m3_ft"], ram=12, threads=4, hours=0.2, kind="gpu", vram_gb=16, gpu_exclusive=True,
             priority=2, env=genv),
        # D89: the Qwen3-0.6B fine-tune path on a few rows (it reads the base run's fold-0 variant)
        _job("smoke__prep_qwen3_0p6b_ft", "smoke", US + ["gpu-prep", "--kind", "qwen3_0p6b_ft", "--fold", "0", "--smoke"],
             deps=["g1", "run__qwen3_0p6b__DEV"], ram=10, threads=2, hours=0.1, env=SW),
        _job("smoke__qwen3_0p6b_ft", "smoke", ["{python}", "{code_src}/uol/final/gpu_jobs/bge_m3_ft.py", "--inputs",
                                               "{work}_smoke/gpu_inputs/qwen3_0p6b_ft_smoke", "--out",
                                               "{work}_smoke/gpu_jobs/qwen3_0p6b_ft_smoke/outputs"],
             deps=["smoke__prep_qwen3_0p6b_ft"], ram=12, threads=4, hours=0.2, kind="gpu", vram_gb=16, gpu_exclusive=True,
             priority=2, env=genv),
    ]
    for kind, script in (("ce", "r2_train.py"), ("zs", "r1_score.py")):
        sm.append(_job(f"smoke__prep_{kind}", "smoke", US + ["ce", "--step", "smoke-prep", "--system", kind], deps=["g1"],
                       ram=6, threads=2, hours=0.05, env=SW))
        sm.append(_job(f"smoke__{kind}", "smoke", ["{python}", f"{{code_src}}/uol/round3/gpu_jobs/{script}", "--inputs",
                                                   f"{{work}}_smoke/gpu_inputs/smoke_{kind}", "--out",
                                                   f"{{work}}_smoke/gpu_jobs/smoke_{kind}/outputs"],
                       deps=[f"smoke__prep_{kind}"], ram=12, threads=4, hours=0.2, kind="gpu", vram_gb=12,
                       gpu_exclusive=True, priority=2, env=genv))
    sm.append(_job("smoke__cost", "smoke", US + ["cost", "--system", "bm25_stem"], deps=["smoke__run"], ram=8, threads=8,
                   hours=0.1, exclusive=True))
    sm.append(_job("smoke__pack", "smoke", US + ["pack"], deps=["smoke__run", "smoke__dense", "smoke__topic"], ram=4,
                   threads=2, hours=0.05))

    def system_of(j: dict[str, Any]) -> dict[str, Any] | None:
        a = j["argv"]
        if "run" in a and "--system" in a:
            return S.get(a[a.index("--system") + 1])
        return None

    for j in jobs:
        jid, extra = j["id"], []
        spec = system_of(j)
        if jid.startswith("run__") and spec:
            ad = spec["adapter"]
            extra = {"topic": ["smoke__topic"], "ltr": ["smoke__ltr"], "dense_vectors": ["smoke__dense"],
                     "fusion": ["smoke__fusion"], "fold_dense": ["smoke__dense"]}.get(ad, ["smoke__run"])
            if j.get("env", {}).get("UOL_FINAL_SOURCES") == "rebuilt":
                extra.append("smoke__rebuild")
        elif "rebuild-stage" in j["argv"] or jid.endswith("_features") or jid in ("features_rb", "features_rb_v11"):
            extra = ["smoke__rebuild"]
        elif jid == "llm__llm_alone":
            extra = ["smoke__llm"]
        elif jid == "llm__hyde":
            extra = ["smoke__hyde"]
        elif jid.startswith("llm__llm_rerank"):
            extra = ["smoke__llm_rerank"]
        elif jid.startswith("llm__agent_"):
            extra = [f"smoke__agent_{jid.split('_')[-1]}"]
        elif jid.startswith("gpu__bge_m3_ft_f"):
            extra = ["smoke__bge_m3_ft"]
        elif jid.startswith("gpu__qwen3_0p6b_ft_f"):
            extra = ["smoke__qwen3_0p6b_ft"]
        elif jid.startswith("gpu__ce_trained_f"):
            extra = ["smoke__ce"]
        elif jid.startswith("gpu__rerank_"):
            extra = ["smoke__zs"]
        elif jid.startswith("cost__"):
            extra = ["smoke__cost"]
        elif jid in ("contrasts", "thesis_pack"):
            extra = ["smoke__pack"]
        j["deps"] = list(dict.fromkeys(j["deps"] + extra))
    return sm + jobs


def variant_jobs(runnable: dict[str, dict], deps_for, ready: dict[str, str]) -> list[dict[str, Any]]:
    """DEV-T, DEV-B and DEV-N: their own release, sources rebuilt from code, pair table and refits (E11)."""
    jobs = []
    pops = {"devt": "DEV-T", "devb": "DEV-B", "devn": "DEV-N"}
    for v, pop in pops.items():
        env = {"UOL_FINAL_VARIANT": v}
        renv = {**RB, **env}
        build_deps = ["g1"] + (["devb_vectors"] if v == "devb" else [])
        if v in ("devt", "devb"):
            jobs.append(_job(f"{v}_build", "E11", UF + ["variant", "--build", v], deps=["g1"], ram=8, threads=2, hours=0.3))
        if v == "devb":
            jobs.append(_job("devb_g4", "E11", UF + ["devb-labels"], deps=["g1"], ram=8, threads=2, hours=0.5,
                             note="gap-review resolver patches P1,P2,P3,P5,P6,P7 over development sources"))
            jobs[-2]["deps"].append("devb_g4")
            jobs.append(_job("prep__devb", "E11", UF + ["variant", "--encode-inputs", "devb"], deps=["devb_build"], ram=6,
                             threads=2, hours=0.05, env=env))
            for key in ("devb_q8_a", "devb_q8_b", "devb_q8_c"):
                jobs.append(_job(f"gpu__{key}", "E11", ["{python}", "{code_src}/uol/round3/gpu_jobs/legal_encode.py",
                                                        "--inputs", f"{{work}}/variants/devb/gpu_inputs/{key}", "--out",
                                                        f"{{work}}/variants/devb/gpu_jobs/{key}/outputs"],
                                 deps=["prep__devb", "fidelity_int8"], ram=12, threads=4, hours=0.1, kind="gpu", vram_gb=16,
                                 gpu_exclusive=True, priority=3, env={"UOL_WORKER_SCRATCH": "{work}/gpu_scratch",
                                                                       "PYTHONPATH": "{code_src}"}))
                jobs.append(_job(f"ingest__{key}", "E11", UF + ["ingest", "--key", key], deps=[f"gpu__{key}"], ram=4,
                                 threads=1, hours=0.02, env=env))
            jobs.append(_job("devb_vectors", "E11", UF + ["variant", "--vectors", "devb"],
                             deps=[f"ingest__devb_q8_{x}" for x in "abc"], ram=8, threads=2, hours=0.1, env=env))
            # the recovered rows' bge-m3 / e5-small query vectors: the original encoder code on the CPU, 50 controls
            # against the frozen vectors (cos >= .99), with the pinned snapshots of the "models" job (22 Oct 2026)
            jobs.append(_job("devb_dense_queries", "E11", UF + ["variant", "--dense-queries", "devb"],
                             deps=["devb_build", "models"], ram=10, threads=8, hours=0.1, env=env,
                             note="facts_to_law.bge_gpu.encode / facts_to_law.dense on the CPU; fails below cos .99"))
        chain_deps = [f"{v}_build"] if v in ("devt", "devb") else ["g1"]
        if v == "devb":
            chain_deps += ["devb_vectors", "devb_dense_queries"]
        if v == "devn":
            jobs += devn_encode_jobs(ready)
        chain = rebuild_chain(f"{v}_", renv, chain_deps, gpu_stages=False)
        jobs += chain
        jobs.append(_job(f"{v}_features", "E11", UF + ["features"], deps=[j["id"] for j in chain], ram=14, threads=8,
                         hours=0.6, env=renv))
        systems = ("lstar_headings", "bm25_stem", "lsa", "S0", "S_full", "S_full_ext") if v != "devn" else \
            ("lstar_headings", "bm25_stem", "lsa", "lda_dm", "doc2vec", "S0") + tuple(DEVN_SYSTEMS)
        for sid in systems:
            if sid not in runnable:
                continue
            s = runnable[sid]
            e = renv if s["adapter"] == "ltr" else env
            d = chain_deps + ([f"{v}_features"] if s["adapter"] == "ltr" else [])
            if v == "devn" and sid in DEVN_SYSTEMS:
                d = ["g1", f"ingest__devn__{DEVN_SYSTEMS[sid]}"] + ([ready[DEVN_SYSTEMS[sid]]] if DEVN_SYSTEMS[sid] in ready else [])
            jobs.append(_run(sid, pop, "E11", d, s, e))
    jobs += devl_jobs(runnable, ready)
    jobs += devs_jobs(runnable, ready)
    return jobs


# C6 (D93; ANALYSIS_PLAN_C §6a): systems scored on DEV-L. Only systems whose query side is recomputed from the lay
# text and that are not retrained on it: no ranker (its sources and refit would be rebuilt on lay facts) and no row
# whose 8B column is a frozen score file (S0+qwen3_8b_dense, the qwen3_8b hybrids)
DEVL_SYSTEMS = ("lstar_headings", "bm25_stem", "lsa", "rrf__lstar_headings__qwen3_0p6b", "linear__lstar_headings__qwen3_0p6b")
# the dense encoders whose catalogue vectors RUN-B has (bundle) or rebuilds anyway: DEV-L adds query encodes only
# (the other DEV-N encoders would need about 5-6 GPU-h of catalogue re-encoding on RUN-B; ANALYSIS_PLAN_C §12)
DEVL_DENSE = ("qwen3_8b_vectors", "qwen3_0p6b", "qwen3_4b", "qwen3_8b_int8_base", "e5_large_legal_en", "bge_m3_base",
              "bge_m3_legal")


def devl_jobs(runnable: dict[str, dict], ready: dict[str, str]) -> list[dict[str, Any]]:
    """DEV-L: the lay rewrite (Codex), the dense encoders' lay query re-encodes, and the runs on DEV-L."""
    env = {"UOL_FINAL_VARIANT": "devl"}
    jobs = [_job("llm__devl_rewrite", "E11", UF + ["llm", "--job", "devl_rewrite"], deps=["codex_smoke"], ram=6,
                 threads=2, hours=1.0, kind="llm", outputs=["work:variants/devl/lay_texts.jsonl"],
                 note="about 61 Codex calls (20 cases per call); 20 rewrites for the author's check in lay_check_20.csv")]
    jobs += devn_encode_jobs(ready, "devl", tuple(DEVN_SYSTEMS[s] for s in DEVL_DENSE))
    for sid in DEVL_SYSTEMS + DEVL_DENSE:
        if sid not in runnable:
            continue
        if sid in DEVN_SYSTEMS:
            d = ["g1", f"ingest__devl__{DEVN_SYSTEMS[sid]}"] + ([ready[DEVN_SYSTEMS[sid]]] if DEVN_SYSTEMS[sid] in ready else [])
        else:
            d = ["g1", "llm__devl_rewrite"] + (["ingest__devl__qwen3_0p6b"] if "qwen3_0p6b" in sid else [])
        jobs.append(_run(sid, "DEV-L", "E11", d, runnable[sid], env))
    return jobs


# D109 (24 Oct; ANALYSIS_PLAN_C §6b): systems scored on the shortened facts. The DEV-L design without the rewrite: the
# cut is deterministic (no LLM job), the rankers are not retrained, and the dense side re-encodes only the queries of the
# two encoders the earlier X10 lens read (8B) or that the fusions use (0.6B)
DEVS_SYSTEMS = DEVL_SYSTEMS
DEVS_DENSE = ("qwen3_8b_vectors", "qwen3_0p6b")


def devs_jobs(runnable: dict[str, dict], ready: dict[str, str]) -> list[dict[str, Any]]:
    """DEV-S25, DEV-S50, DEV-S100: the query re-encodes of the cut facts and the runs on each length."""
    from uol.final import devs
    jobs = []
    for v, pop in devs.POPULATIONS.items():
        env = {"UOL_FINAL_VARIANT": v}
        jobs += devn_encode_jobs(ready, v, tuple(DEVN_SYSTEMS[s] for s in DEVS_DENSE))
        for sid in DEVS_SYSTEMS + DEVS_DENSE:
            if sid not in runnable:
                continue
            if sid in DEVN_SYSTEMS:
                d = ["g1", f"ingest__{v}__{DEVN_SYSTEMS[sid]}"] + ([ready[DEVN_SYSTEMS[sid]]] if DEVN_SYSTEMS[sid] in ready else [])
            else:
                d = ["g1"] + ([f"ingest__{v}__qwen3_0p6b"] if "qwen3_0p6b" in sid else [])
            jobs.append(_run(sid, pop, "E11", d, runnable[sid], env))
    return jobs


def smoke_plan() -> dict[str, Any]:
    """Dry run on the smoke bundle: G1 on the cached sources it carries, then 1-fold runs of a few real systems."""
    jobs = [_job("g1", "G1", UF + ["g1", "--route", "cached", "--only", "lstar_frozen,qwen3_8b,ltr_noemb_frozen,ltr_emb_frozen"],
                 ram=4, threads=2, hours=0.05, gate=True, env=V11, outputs=["store:g1"])]
    for sid in ("popularity", "random", "lstar_frozen", "src_prior", "bm25_stem", "lsa"):
        jobs.append(_job(f"run__{sid}__DEV", "E1", UF + ["run", "--system", sid, "--population", "DEV", "--folds", "0",
                                                         "--max-rows", "40"],
                         deps=["g1"], ram=6, threads=2, hours=0.05, outputs=[f"store:runs/{sid}/DEV"]))
    jobs.append(_job("thesis_pack", "pack", UF + ["pack"], deps=[], ram=2, threads=1, hours=0.01, final=True))
    return {"plan": "uol.final smoke plan (dry run)", "n_jobs": len(jobs), "gates": ["g1"], "jobs": jobs}


def new_cases_plan() -> dict[str, Any]:
    """E13 (RUN_PLAN v2 §4, §8): the post-freeze plan, started on the run computer on or after 31 Oct with its own
    small bundle (``uol.final bundle-new``, built only after FREEZE.json exists). NEW: collect with the existing
    collector, build the sources, resolve and mask with the frozen resolver and masker, encode the queries through the
    fidelity-gated Qwen3-8B int8 path, score the frozen answer systems (models trained on every development case) and
    run the LLM stage. TEST-R: test-period cases recovered by the fixed linking rules, selected from resolver output
    only (never scores or rankings), scored once with NEW."""
    U = ["{python}", "-m", "uol.final", "--data", "{data}", "--store", "{store}"]
    j = [
        _job("new_collect", "E13", ["{python}", "-m", "uol.round2.c_collect", "collect", "--max-pages", "3"], ram=4, threads=2,
             hours=0.5, note="Selenium + Chrome against the SJF repository (the collector's own throttling)"),
        _job("new_build", "E13", ["{python}", "-m", "uol.round2.c_collect", "build", "--start", "2026-09-12", "--cutoff",
                                  "{today}"], deps=["new_collect"], ram=4, threads=2, hours=0.1),
        _job("new_release", "E13", U + ["new-cases", "--step", "release"], deps=["new_build"], ram=8, threads=2, hours=0.2,
             note="frozen resolver (references_v2) + masker (CatalogueMasker v3), release format of v11"),
        _job("testr_select", "E13", U + ["new-cases", "--step", "testr"], ram=8, threads=2, hours=0.3,
             note="test-period sources re-resolved with the gap-review patches; selection by resolver output only"),
        _job("new_encode_prep", "E13", U + ["new-cases", "--step", "encode-inputs"], deps=["new_release", "testr_select"],
             ram=4, threads=2, hours=0.05),
        _job("new_gate_e0", "E13", U + ["gate-e0"], deps=["new_encode_prep"], ram=4, threads=1, hours=0.3, kind="gpu",
             vram_gb=16, gpu_exclusive=True),
        _job("new_encode", "E13", ["{python}", "{code_src}/uol/round3/gpu_jobs/legal_encode.py", "--inputs",
                                   "{work}/gpu_inputs/new_cases", "--out", "{work}/gpu_jobs/new_cases/outputs"],
             deps=["new_gate_e0"], ram=12, threads=4, hours=0.2, kind="gpu", vram_gb=16, gpu_exclusive=True),
        _job("new_score", "E13", U + ["new-cases", "--step", "score"], deps=["new_encode"], ram=24, threads=8, hours=0.5,
             note="P1-P6, P5+CE, P3+L frozen (FREEZE.json); population NEW and TEST-R, scored once"),
        _job("new_llm", "E13", U + ["new-cases", "--step", "llm"], deps=["new_score"], ram=6, threads=2, hours=1.0, kind="llm"),
        _job("new_pack", "E13", U + ["pack", "--only", "E13"], deps=["new_llm"], ram=4, threads=2, hours=0.1, final=True),
    ]
    return {"plan": "uol.final post-freeze plan (E13: NEW and TEST-R)", "n_jobs": len(j), "requires": "FREEZE.json",
            "status": "prepared; its bundle and the new-cases steps are built after the freeze", "jobs": j}


def write(path: Path, smoke: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(build(smoke), indent=1), encoding="utf-8")
    tmp.replace(path)
    if not smoke:
        (path.parent / "plan_new_cases.json").write_text(json.dumps(new_cases_plan(), indent=1), encoding="utf-8")
    return path
