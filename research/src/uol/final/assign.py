"""Assignments of the second runner, RUN-B (D85, 23 Oct 2026).

    py -3.12 -m uol.final assignments --plan jobs/plan.json --out <Drive>/UOL_RUN_B/ASSIGNMENTS.json

RUN-B installs from the same bundle as the run computer but writes to its own Drive folder (``UOL_RUN_B``) and runs
only the jobs of the ``active`` assignments (``overnight --results-root UOL_RUN_B --jobs-file ASSIGNMENTS.json``).
Each assignment lists its plan job ids; the runner refuses a list whose dependencies are neither listed nor already
done on that machine. Later assignments are added only after a decision row (LEAD confirms the author's answer).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FOLDER = "UOL_RUN_B"
ACCEPTANCE = ("features_v11", "g1")
CORE = ("run__S0__DEV", "run__S_full__DEV", "run__S_full_ext__DEV")
# compared by RUN-B after its jobs, one contrast at a time (the run computer computes them inside family F1)
CORE_CONTRASTS = (("S_full", "S0"), ("S_full_ext", "S0"))
G1_FROZEN = {"tuned lexical": .183763, "Qwen3-8B": .274169, "ranker without embeddings": .408136,
             "ranker with embeddings": .472101, "LLM-stage hybrid": .555788}


def closure(plan: dict[str, Any], ids: list[str] | tuple[str, ...]) -> list[str]:
    """The ids and everything they depend on, in plan order."""
    jobs = {j["id"]: j for j in plan["jobs"]}
    seen: set[str] = set()
    stack = list(ids)
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        if x not in jobs:
            raise KeyError(f"not in the plan: {x}")
        seen.add(x)
        stack += jobs[x].get("deps", [])
    return [j["id"] for j in plan["jobs"] if j["id"] in seen]


def run_b(plan: dict[str, Any], plan_sha256: str) -> dict[str, Any]:
    jobs = {j["id"]: j for j in plan["jobs"]}
    acc = closure(plan, ACCEPTANCE)
    core = [j for j in closure(plan, CORE) if j not in acc]
    py, uf = "<Root>\\venv\\Scripts\\python.exe", "-m uol.final --data <Root>\\data --store <Root>\\run\\final_20261021"
    hours = lambda ids: round(sum(float(jobs[i].get("est_hours", 0)) for i in ids), 1)
    return {
        "runner": "RUN-B (D85)",
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "plan_sha256": plan_sha256,
        "results_folder": f"{FOLDER}\\results (never UOL_FINAL_RUN\\results)",
        "install": ("Until the clean repository and data pack exist (P2, P3): setup from the bundle exactly as the run "
                    "computer did, but with its own results folder: powershell -ExecutionPolicy Bypass -File "
                    "\"<Drive>\\UOL_FINAL_RUN\\setup\\setup_runner.ps1\" -ResultsRoot UOL_RUN_B (<Root> is the root it "
                    "prints; setup.log goes to UOL_RUN_B\\results). Fixes arrive as on "
                    "the run computer: UOL_FINAL_RUN\\FIXES_READY.json, applied by re-running setup."),
        "start_runner": (f"Start-Process -FilePath {py} -ArgumentList '-m','uol.final','overnight','--plan',"
                         "'<Root>\\jobs\\plan.json','--results-root','UOL_RUN_B','--jobs-file','<Drive>\\UOL_RUN_B\\"
                         "ASSIGNMENTS.json' -RedirectStandardOutput <Root>\\logs\\overnight.log -RedirectStandardError "
                         "<Root>\\logs\\overnight.err -WindowStyle Minimized"),
        "rules": ["Run only the jobs of assignments whose status is active; restart the runner when this file changes.",
                  "Never change code, data or plan; failures go to UOL_RUN_B\\results\\OPERATOR_REQUESTS.md with the log tail.",
                  "LLM jobs only through Codex signed in by the author on this machine; paid APIs are excluded."],
        "assignments": [
            {"id": "A1", "status": "active", "title": "Acceptance",
             "before_runner": ["setup_runner.ps1 passes: every MANIFEST sha256, the GPU check, the expected test count "
                               "of FIXES_READY.json",
                               f"$env:UOL_FINAL_LABELS = 'v11'; {py} {uf} g1 --route cached  (must print PASS)",
                               "write UOL_RUN_B\\results\\MACHINE.md (nvidia-smi, RAM, free disk, OS, Python) and "
                               "results\\G1.md (the 9 values)"],
             "jobs": acc, "est_job_hours": hours(acc),
             "pass": {"g1": "9 of 9 frozen values within 1e-4", "frozen_values": G1_FROZEN},
             "on_fail": "stop; the runner blocks every job when g1 fails (results\\G1_FAILED.md)"},
            {"id": "A2", "status": "active", "title": "Independent reproduction of the core rankers",
             "jobs": core, "est_job_hours": hours(core),
             "after_runner": [f"{py} {uf} contrast --a {a} --b {b}" for a, b in CORE_CONTRASTS],
             "compare": ("For S0, S_full and S_full_ext: nDCG@10, n_cases and n_groups of runs\\<system>\\DEV\\summary.json "
                         "against UOL_FINAL_RUN\\results\\raw\\runs\\<system>\\DEV\\summary.json once the run computer has "
                         "them, and the two contrasts against its contrasts\\. Record every difference above 1e-4 in "
                         "results\\REPRO.md; if the run computer has no value yet, record RUN-B's value and say so."),
             "systems": [c.split("__")[1] for c in CORE]},
            {"id": "A3", "status": "awaiting_decision", "title": "Robustness behind the failed rebuild gate",
             "jobs": [],
             "note": ("ACTION_PLAN_20261022 §5 decision 2 (fix sequence 10: DEV-G, DEV-ALL, ranker seeds and their "
                      "rebuilt sources). Added here only after LEAD confirms the author's answer in DECISIONS.md.")},
        ],
    }


def descendants(plan: dict[str, Any], roots: set[str]) -> set[str]:
    """The roots and every job that depends on one of them."""
    out = set(roots)
    changed = True
    while changed:
        changed = False
        for j in plan["jobs"]:
            if j["id"] not in out and set(j.get("deps", [])) & out:
                out.add(j["id"])
                changed = True
    return out


def cost_gpu_jobs(plan: dict[str, Any], specs: dict[str, dict[str, Any]]) -> set[str]:
    """GPU jobs whose run records a cost job reads on its own machine (``cost._gpu_runs`` and ``cost.query_encoding``:
    ``gpu__<job|key>``, ``gpu__devn__<key>``, per-fold ``_f<k>``), for every cost job of the plan and its members."""
    import re
    ids = {j["id"] for j in plan["jobs"]}
    names: set[str] = set()

    def add(sid: str, depth: int = 0) -> None:
        s = specs.get(sid) or {}
        a = s.get("adapter_args", {}) or {}
        names.update(x for x in (sid, a.get("job"), a.get("key")) if x)
        if str(a.get("source", "")).startswith("dense_"):
            names.add(str(a["source"])[len("dense_"):])
        if depth == 0:
            for m in list(a.get("members", []) or []) + list(a.get("extra", []) or []):
                add(m, 1)

    for i in ids:
        if i.startswith("cost__"):
            add(i[len("cost__"):])
    pat = re.compile(r"gpu__(devn__)?(" + "|".join(re.escape(n) for n in sorted(names)) + r")(_f\d+(_s\d+)?)?")
    return {i for i in ids if pat.fullmatch(i)} if names else set()


def run_b_fallback(plan: dict[str, Any], plan_sha256: str, done_a: set[str],
                   specs: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """D86: RUN-B takes the remaining run. Every job RUN-A had not finished, except the gated robustness jobs (the soft
    gate g1_rebuild and its dependants, decision 2), plus everything they depend on (RUN-A's work folder is not on
    Drive), plus the finished HyDE rows (one machine's HyDE texts) and the GPU jobs the cost records read."""
    doc = run_b(plan, plan_sha256)
    jobs = {j["id"]: j for j in plan["jobs"]}
    soft = {j["id"] for j in plan["jobs"] if j.get("soft_gate")}
    gated = descendants(plan, soft)
    remaining = {i for i in jobs if i not in done_a}
    hyde = descendants(plan, {"llm__hyde"}) if "llm__hyde" in jobs else set()     # every HyDE row from RUN-B's texts
    cost_gpu = cost_gpu_jobs(plan, specs or {})
    want = (remaining - gated) | (hyde - gated) | (cost_gpu - gated)
    full = set(closure(plan, sorted(want)))
    if full & gated:
        raise ValueError(f"the RUN-B list would need gated jobs: {sorted(full & gated)[:5]}")
    a1 = set(closure(plan, ACCEPTANCE))
    a2 = set(closure(plan, CORE)) - a1
    llm = {i for i in full if jobs[i].get("kind") == "llm"}
    final = {i for i in full if jobs[i].get("final")}
    a5 = (descendants(plan, llm) & full) - a1 - a2 - final
    a5 |= {i for i in final if i == "cost_summary"}             # it summarises the LLM calls: after them
    a6 = final - a5                                              # contrasts, thesis_pack: after merge-store
    # C10 (D101): the external second stages wait for the thesis PC's outputs in UOL_RUN_B/inputs
    a7 = descendants(plan, {i for i in full if i.startswith("ext__")}) & full - a6
    a5 -= a7
    a4 = full - a1 - a2 - a5 - a6 - a7
    order = [j["id"] for j in plan["jobs"]]
    lst = lambda s: [i for i in order if i in s]
    hours = lambda s: round(sum(float(jobs[i].get("est_hours", 0)) for i in s), 1)
    # D93: the gated jobs that may pass the failed soft gate (the S_full_ext seeds) run, with the rebuild stages
    # they need; every other gated job stays in A3
    seeds = sorted(i for i in gated & remaining if jobs[i].get("soft_gate_ok") and i.startswith("run__"))
    a3s = set(closure(plan, seeds)) - full if seeds else set()
    a3 = lst(set(closure(plan, sorted(gated & remaining))) - full - a3s)  # with the rebuild stages it needs here
    py, uf = "<Root>\\venv\\Scripts\\python.exe", "-m uol.final --data <Root>\\data --store <Root>\\run\\final_20261021"
    doc["runner"] = "RUN-B (D85), taking the remaining final run (D86)"
    doc["rules"].append("Assignments that are not active are switched on by RESULTS only; restart the runner (with "
                        "--retry-failed) whenever this file changes.")
    doc["counts"] = {"run_a_done": len(done_a), "remaining": len(remaining), "gated_inactive": len(a3),
                     "gated_active_seeds": len(a3s), "recomputed_from_run_a": len((full | a3s) & done_a),
                     "run_b_total": len(full | a3s)}
    doc["reproduction"] = ("RUN-B recomputes finished RUN-A jobs. For every runs\\<system>\\<population>\\summary.json "
                           "written here that UOL_FINAL_RUN\\results\\raw also has, record nDCG@10, n_cases and n_groups of "
                           "both in results\\REPRO.md, and every difference above 1e-4 also in OPERATOR_REQUESTS.md. "
                           "HyDE rows are exempt (new texts by design, D86); record them without comparing. Also record "
                           "the NVIDIA driver version beside RUN-A's (its receipts: machine.gpu).")
    a1_doc, a2_doc = doc["assignments"][0], doc["assignments"][1]
    a1_doc["est_job_hours"], a2_doc["est_job_hours"] = hours(a1), hours(a2)
    doc["assignments"] = [
        a1_doc, a2_doc,
        {"id": "A3", "status": "awaiting_decision", "title": "Robustness behind the failed rebuild gate",
         "jobs": a3, "est_job_hours": hours(a3),
         "note": ("ACTION_PLAN_20261022 §5 decision 2: the soft gate g1_rebuild and its dependants. RUN-A's G1-rebuild "
                  "verdict stands (merged into this store by merge-store). Activated only after LEAD confirms the author's "
                  "answer in DECISIONS.md, with the fix it needs.")},
        {"id": "A3s", "status": "active", "title": "The S_full_ext seeds on rebuilt sources (D93)",
         "jobs": lst(a3s), "est_job_hours": hours(a3s),
         "note": ("The two S_full_ext seed rows and the rebuild stages they need; they pass the failed g1_rebuild "
                  "(soft_gate_ok), which runs here again as part of the chain and keeps blocking every other gated "
                  "row. The seed spread is the noise yardstick of C2 (ANALYSIS_PLAN_C §10).")},
        {"id": "A4", "status": "active", "title": "The rest of the remaining run, without the LLM",
         "jobs": lst(a4), "est_job_hours": hours(a4),
         "note": ("Remaining jobs in the runner's order (GPU first, then the longest CPU jobs, cost timing alone on the "
                  "machine), with every finished RUN-A job they need recomputed here (RUN-A's work folder is not on Drive) "
                  "and the GPU runs the cost records read.")},
        {"id": "A5", "status": "awaiting_codex", "title": "LLM stages and everything that needs them",
         "jobs": lst(a5), "est_job_hours": hours(a5),
         "note": ("Codex smoke, LLM reranking over S0 and S_full_ext (P3+L, P6), HyDE (all HyDE rows recomputed here, "
                  "one machine's texts), the embedding-tool agent, the fusions and costs that use them, cost_summary. "
                  "Activated by RESULTS once A2 reproduces .408136 / .472101 and the author has signed in to Codex here.")},
        {"id": "A7", "status": "awaiting_inputs", "title": "External second stages: Claude and Jev (C10, D101)",
         "before_runner": ["copy UOL_RUN_B\\inputs\\claude_orders_s0.json and jev_p_yes.jsonl (whichever exist) into "
                           "<Root>\\work\\external\\"],
         "jobs": lst(a7), "est_job_hours": hours(a7),
         "note": "Activated by RESULTS when the thesis PC has written the inputs (after A2's S_full_ext); before A6."},
        {"id": "A6", "status": "awaiting_merge", "title": "Contrasts and the thesis pack over all results",
         "before_runner": [f"stop the runner; then {py} {uf} merge-store --from \"<Drive>\\UOL_FINAL_RUN\\results\\raw\""],
         "jobs": lst(a6), "est_job_hours": hours(a6),
         "note": "Activated by RESULTS after every job of A1, A2, A4 and A5 is done or failed."},
    ]
    return doc


def reset_ids(paths: list[Path]) -> set[str]:
    """Job ids of reset lists (``reset-jobs`` files: one id per line, ``#`` comments)."""
    out: set[str] = set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            i = line.split("#", 1)[0].strip()
            if i:
                out.add(i)
    return out


def write(plan_path: Path, out: Path, done_status: Path | None = None, resets: list[Path] | None = None) -> dict[str, Any]:
    raw = Path(plan_path).read_bytes()
    plan = json.loads(raw.decode("utf-8"))
    if done_status:
        from uol.final import registry
        # a job a reset list invalidated is not done, whatever an older receipt says (23 Oct: RUN-A stopped before
        # applying reset_r6, and its pre-reset receipts made 49 reset jobs look finished)
        done = set(json.loads(Path(done_status).read_text(encoding="utf-8"))["done"]) - reset_ids(resets or [])
        doc = run_b_fallback(plan, hashlib.sha256(raw).hexdigest(), done, registry.systems())
    else:
        doc = run_b(plan, hashlib.sha256(raw).hexdigest())
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"out": str(out), "assignments": {a["id"]: (a["status"], len(a["jobs"])) for a in doc["assignments"]}}
