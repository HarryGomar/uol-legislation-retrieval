"""Spanish legal encoders on the scale-curve units, zero-shot (PLAN research/programs/legal_encoders_20261020; dev, descriptive).

Each model is encoded on the RTX 5080 worker by ``round3/gpu_jobs/legal_encode.py`` (through ``round3.gpu_remote``),
ingested into its scale-curve vector store, assembled in the scale-curve layout and read out with
``scale_curve.evaluate`` (dense alone, then the NO-EMB + source refit). Only dev rows travel; the units file is the
scale-curve export (dev queries only), and every query owner is checked again with ``assert_dev``.

    python -m uol.scale_curve.legal inputs --model alia_mrbert --subset smoke      # write job inputs only (no GPU)
    python -m uol.scale_curve.legal submit --model alia_mrbert --subset smoke      # blocks; run in the background
    python -m uol.scale_curve.legal submit --model qwen3_8b_gate --subset validation
    python -m uol.scale_curve.legal gate-e0
    python -m uol.scale_curve.legal submit --model bge_m3_base --subset full
    python -m uol.scale_curve.legal ingest --model bge_m3_base [--job-id ...]      # store + assemble
    python -m uol.scale_curve.legal dense                                          # light
    python -m uol.round3.heavy uol.scale_curve.legal refit --model bge_m3_legal
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.scale_curve import common as C
from uol.scale_curve import evaluate as EV

PROGRAM = "legal_encoders_20261020"
RUN = C.ROOT / "experiments/runs" / PROGRAM
EVAL = RUN / "eval"
INPUTS = RUN / "inputs"
LEDGER = C.ROOT / "research/state" / PROGRAM / "LEDGER.jsonl"
PLAN = C.ROOT / "research/programs" / PROGRAM / "PLAN.md"
JOBS = Path(__file__).resolve().parents[1] / "round3" / "gpu_jobs"
ENTRY = JOBS / "legal_encode.py"
HELPERS = (JOBS / "rerank_bench.py", Path(__file__).resolve().parent / "store.py")
LABEL = "dev, descriptive"

# key: (repo, revision, dim, max tokens, {group: prompt rule}, precision, batch tokens, params, base for the paired contrast)
SPECS: dict[str, dict[str, Any]] = {
    "alia_mrbert": dict(model="SINAI/ALIA-MrBERT-es-legal-administrative-embeddings",
                        revision="dc6fcf1a36c3e75ff6d1d0866aca85f781035eed", dim=768, max_tokens=2048,
                        prompts={"articles": "model", "queries_a": "model"}, precision="fp32", batch_tokens=32768,
                        params="~150M", base=None, name="ALIA-MrBERT legal-administrative embeddings (~150M)"),
    "gemma_legal": dict(model="wilfredomartel/embeddinggemma-300m-legal-spanish-420k-v2",
                        revision="47fd23052f7df1ad415b0e082280a1144077a8fe", dim=768, max_tokens=2048,
                        prompts={"articles": "model", "queries_a": "model"}, precision="fp32", batch_tokens=32768,
                        params="300M", base=None, name="EmbeddingGemma-300m legal Spanish (300M)"),
    "bge_m3_legal": dict(model="wilfredomartel/BGE-M3-Legal-Spanish", revision="2cc48e7fea93caa531032a1fa2f099c6efd45e5e",
                         dim=1024, max_tokens=8192, prompts={"articles": "model", "queries_a": "model"}, precision="fp32",
                         batch_tokens=32768, params="568M", base="bge_m3_base", name="BGE-M3 Legal Spanish (568M)"),
    "bge_m3_base": dict(model="BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181", dim=1024,
                        max_tokens=8192, prompts={"articles": "model", "queries_a": "model"}, precision="fp32",
                        batch_tokens=32768, params="568M", base=None, name="bge-m3, re-encoded here (568M)"),
    "euler_legal": dict(model="Mira190/Euler-Legal-Embedding-V1", revision="492c75735b5e6be22a8b792229b4ef2e22f071ed",
                        dim=4096, max_tokens=8192, prompts={"articles": "none", "queries_a": "model", "queries_c": "none"},
                        precision="int8", batch_tokens=16384, params="8B", base="qwen3_8b",
                        name="Euler-Legal-Embedding-V1 (Qwen3-Embedding-8B + legal LoRA, int8)"),
}
# Gate E0 only: the base 8B through the same code path, compared with the cached vectors (never a source).
GATE = {"qwen3_8b_gate": dict(model="Qwen/Qwen3-Embedding-8B", revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af",
                              dim=4096, max_tokens=8192, prompts={"articles": "none", "queries_c": "none"},
                              precision="int8", batch_tokens=16384)}
ORDER = ("alia_mrbert", "gemma_legal", "bge_m3_base", "bge_m3_legal", "euler_legal")
REFERENCES = ("e5_small", "bge_m3", "qwen3_0p6b", "qwen3_4b", "qwen3_8b")
GATE_L0_TOL = .015
GATE_E0 = {"cos_mean": .99, "cos_min": .97}
REFIT_FLOOR = .186                                       # Qwen3-Embedding-0.6B dense alone (scale curve)

for _k, _s in SPECS.items():                             # register as scale-curve vector sources (in-process only)
    C.MODELS.setdefault(_k, C.ModelSpec(_k, _s["model"], _s["revision"], _s["dim"],
                                        C.ROOT / "data/derived" / f"legal_{_k}_v1", "remote", _s["params"]))
    EV.VECTOR_SOURCES.setdefault(_k, _s["name"])


def ledger(row: dict[str, Any]) -> None:
    C.append_jsonl(LEDGER, {"utc": C.utc(), **row})


def spec(key: str) -> dict[str, Any]:
    return SPECS[key] if key in SPECS else GATE[key]


# --------------------------------------------------------------------------- job inputs


def unit_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The scale-curve units file, hash-checked against its manifest (dev queries only)."""
    man = json.loads((C.UNITS_DIR / "units_manifest.json").read_text(encoding="utf-8"))
    path = C.UNITS_DIR / "units.jsonl"
    if ep.sha256_file(path) != man["sha256"] or not man.get("dev_only") or man.get("lockbox_rows_included"):
        raise RuntimeError("units.jsonl does not match its dev-only manifest")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return rows, man


def select(rows: list[dict[str, Any]], man: dict[str, Any], groups: Sequence[str], subset: str) -> list[dict[str, Any]]:
    """Items {"i", "g", "o", "w", "text"} of ``groups``: all ("full"), the fixed validation set, or a 20-unit smoke set
    (16 article units including the longest, 4 queries per query group)."""
    by_group: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        by_group.setdefault(r["g"], []).append(i)
    pick: list[int] = []
    for g in groups:
        idx = by_group[g]
        if subset == "full":
            pick += idx
        elif subset == "validation":
            pos = man["validation"]["article_positions"] if g == "articles" else man["validation"]["query_c_positions"]
            pick += [idx[p] for p in pos]
        elif subset == "smoke":
            if g == "articles":
                longest = max(idx, key=lambda i: rows[i]["t"])
                pick += sorted(set(idx[:15]) | {longest})
            else:
                pick += idx[:4]
        else:
            raise ValueError(subset)
    items = [{"i": i, "g": rows[i]["g"], "o": rows[i]["o"], "w": rows[i]["w"], "text": rows[i]["text"]} for i in pick]
    C.assert_dev([it["o"] for it in items if it["g"] != "articles"])
    return items


def build_inputs(key: str, subset: str, job_id: str, budget_seconds: float | None = None) -> dict[str, Path]:
    s = spec(key)
    rows, man = unit_rows()
    items = select(rows, man, list(s["prompts"]), subset)
    d = INPUTS / job_id
    d.mkdir(parents=True, exist_ok=True)
    with (d / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    task = {"key": key, "model": s["model"], "revision": s["revision"], "max_tokens": s["max_tokens"],
            "prompts": s["prompts"], "precision": s["precision"], "batch_tokens": s["batch_tokens"], "limit": None,
            "budget_seconds": budget_seconds,
            "subset": subset, "units_sha256": man["sha256"], "plan_sha256": ep.sha256_file(PLAN)}
    C.write_json(d / "task.json", task)
    return {"items.jsonl": d / "items.jsonl", "task.json": d / "task.json"}


def submit(key: str, subset: str, job_id: str) -> dict[str, Any]:
    from uol.round3 import gpu_remote
    inputs = build_inputs(key, subset, job_id)
    ledger({"kind": "job_start", "job_id": job_id, "model": key, "subset": subset, "gpu": "remote RTX 5080"})
    receipt = gpu_remote.submit(job_id, ENTRY, inputs, helpers=HELPERS,
                                expected_outputs=["vectors.npy", "keys.json", "run.json"])
    ledger({"kind": "job_end", "job_id": job_id, "model": key, "exit_code": receipt.get("exit_code"),
            "gpu_seconds": receipt.get("seconds")})
    return receipt


def job_outputs(job_id: str) -> Path:
    from uol.round3 import gpu_remote
    return gpu_remote.LOCAL_JOBS / job_id / "remote" / "outputs"


def default_job(key: str, subset: str) -> str:
    return f"legal_{key}_{subset}_s1"


# --------------------------------------------------------------------------- ingest, assemble, gates


def load_job(job_id: str, dim: int) -> tuple[dict[int, list[str]], dict[str, np.ndarray], dict[str, Any]]:
    out = job_outputs(job_id)
    meta = json.loads((out / "keys.json").read_text(encoding="utf-8"))
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    V = np.load(out / "vectors.npy")
    if V.shape != (len(meta["keys"]), dim) or run["dim"] != dim:
        raise ValueError(f"{job_id}: vectors {V.shape}, expected ({len(meta['keys'])}, {dim})")
    if not np.isfinite(V).all():
        raise ValueError(f"{job_id}: non-finite vectors")
    return {int(i): ks for i, ks in meta["items"]}, dict(zip(meta["keys"], V)), run


def ingest(key: str, job_id: str) -> dict[str, Any]:
    """Store the job's vectors and assemble the scale-curve vector files for ``key`` (full subset only)."""
    s, ms = SPECS[key], C.MODELS[key]
    item_keys, vecs, run = load_job(job_id, s["dim"])
    if run["task"]["subset"] != "full" or run.get("incomplete"):
        raise ValueError("only a complete full job is assembled")
    store = C.VectorStore(ms.store_dir, ms.dim)
    keys = list(vecs)
    store.add(keys, np.stack([vecs[k] for k in keys]))
    rows, _ = unit_rows()
    by_unit = {(rows[i]["g"], rows[i]["o"], rows[i]["w"]): ks for i, ks in item_keys.items()}
    units = C.build_units(check=True)
    variants = tuple(g.split("_", 1)[1] for g in s["prompts"] if g.startswith("queries_"))

    def key_of(it: C.Item) -> list[str]:
        return by_unit[(it.group, it.owner, it.window)]

    extra = {"program": PROGRAM, "plan_sha256": ep.sha256_file(PLAN), "job_id": job_id,
             "encoder": {"repo": s["model"], "revision": s["revision"], "precision": s["precision"],
                         "max_tokens": s["max_tokens"], "prompts": run["prompts"], "model_prompts": run.get("model_prompts"),
                         "windows": "units above max_tokens cut into title-prefixed windows (64-token overlap), MaxP",
                         "queries_truncated": run["queries_truncated"], "articles_split": run["articles_split"],
                         "library": "sentence-transformers 6.1.0, transformers 5.17.0"},
             "compute": {k: run[k] for k in ("encode_seconds", "load_seconds", "tokens_encoded", "peak_vram_mib", "device")}}
    man = C.assemble(ms, units, extra, variants=variants, key_of=key_of)
    ledger({"kind": "assembled", "model": key, "job_id": job_id, "store_vectors": man["counts"]["store_vectors"]})
    return man


def gate_e0(job_id: str = default_job("qwen3_8b_gate", "validation")) -> dict[str, Any]:
    """Base 8B (int8, this job's code path) vs the cached 8B vectors on the scale-curve validation set."""
    item_keys, vecs, run = load_job(job_id, GATE["qwen3_8b_gate"]["dim"])
    rows, man = unit_rows()
    units = C.build_units(check=False)
    A8 = np.load(C.QWEN8B / "article_vectors.npy", mmap_mode="r")
    Q8 = np.load(C.QWEN8B / "query_vectors_c.npy", mmap_mode="r")
    art_index = {(it.owner, it.window): p for p, it in enumerate(units["articles"])}
    cos = {"articles": [], "queries": []}
    for i, ks in item_keys.items():
        r = rows[i]
        if len(ks) != 1:
            raise ValueError("an 8B unit was split; the gate needs unsplit units")
        v = vecs[ks[0]]
        ref = A8[art_index[(r["o"], r["w"])]] if r["g"] == "articles" else Q8[r["o"]]
        cos["articles" if r["g"] == "articles" else "queries"].append(float(v @ np.asarray(ref, dtype=np.float32)))
    allc = np.asarray(cos["articles"] + cos["queries"])
    out = {"job_id": job_id, "label": "fidelity gate (PLAN Gate E0)", "n": int(len(allc)), "utc": C.utc(),
           "cos_mean": float(allc.mean()), "cos_min": float(allc.min()),
           "articles": {"n": len(cos["articles"]), "mean": float(np.mean(cos["articles"])), "min": float(np.min(cos["articles"]))},
           "queries": {"n": len(cos["queries"]), "mean": float(np.mean(cos["queries"])), "min": float(np.min(cos["queries"]))},
           "rule": GATE_E0, "compute": {k: run.get(k) for k in ("encode_seconds", "peak_vram_mib", "device")}}
    out["pass"] = bool(out["cos_mean"] >= GATE_E0["cos_mean"] and out["cos_min"] >= GATE_E0["cos_min"])
    C.write_json(EVAL / "gate_e0.json", out)
    ledger({"kind": "gate_e0", "pass": out["pass"], "cos_mean": out["cos_mean"], "cos_min": out["cos_min"]})
    return out


# --------------------------------------------------------------------------- readouts


def run_dense() -> dict[str, Any]:
    from uol.round3 import e2_adapter as e2
    from uol.round3 import stats

    summary: dict[str, Any] = {"label": LABEL, "program": PROGRAM, "plan_sha256": ep.sha256_file(PLAN),
                               "metric": "dev P_spec component-macro nDCG@10 (MaxP dense alone)", "utc": C.utc(),
                               "sources": {}, "contrasts": {}}
    results: dict[str, dict] = {}
    for key in REFERENCES + ORDER:
        if not EV.available(key):
            summary["sources"][key] = {"available": False}
            continue
        info, res = EV.evaluate_source(key)
        results[key] = res[info["selected_variant"]]
        summary["sources"][key] = {"available": True, "reference": key in REFERENCES,
                                   "name": EV.SAVED.get(key, (None, EV.VECTOR_SOURCES.get(key)))[1], **info}
    pairs = [(k, "qwen3_8b") for k in ORDER] + [(k, SPECS[k]["base"]) for k in ORDER if SPECS[k]["base"]]
    for a, b in dict.fromkeys(pairs):
        if a in results and b in results and a != b:
            con = stats.contrast(results[a]["per_component"]["ndcg_at_10"], results[b]["per_component"]["ndcg_at_10"])
            summary["contrasts"][f"{a}_minus_{b}"] = {
                "ndcg_delta": con["delta"], "ndcg_ci95": con["ci95"], "class": con["class"],
                "unseen_recall_delta": e2.recall_contrast(e2.unseen_recall(results[a]), e2.unseen_recall(results[b]), "unseen")}
    if "bge_m3_base" in results and "bge_m3" in results:
        mine, saved = results["bge_m3_base"]["macro"]["ndcg_at_10"], results["bge_m3"]["macro"]["ndcg_at_10"]
        summary["gate_l0"] = {"re_encoded": mine, "saved_file": saved, "abs_diff": abs(mine - saved),
                              "tolerance": GATE_L0_TOL, "pass": abs(mine - saved) <= GATE_L0_TOL}
    summary["refit_candidates"] = refit_candidates(summary)
    C.write_json(EVAL / "dense_alone.json", summary)
    ledger({"kind": "dense", "sources": {k: v.get("variants", {}).get(v.get("selected_variant"), {}).get("ndcg_at_10")
                                         for k, v in summary["sources"].items() if v.get("available")}})
    return summary


def refit_candidates(summary: dict[str, Any]) -> list[str]:
    """PLAN readout 3: dense alone >= the Qwen 0.6B level, or above its base (lower bound > 0); Euler if encoded."""
    out = []
    for k in ORDER:
        src = summary["sources"].get(k, {})
        if not src.get("available") or k == "bge_m3_base":
            continue
        nd = src["variants"][src["selected_variant"]]["ndcg_at_10"]
        base = SPECS[k]["base"]
        con = summary["contrasts"].get(f"{k}_minus_{base}") if base else None
        if k == "euler_legal" or nd >= REFIT_FLOOR or (con and con["ndcg_ci95"][0] > 0):
            out.append(k)
    return out


def run_refit(key: str) -> dict[str, Any]:
    out = EV.run_refit(key, eval_dir=EVAL, ckpt_root=RUN / "refit")
    ledger({"kind": "refit", "model": key, "P_spec": out["arm"]["P_spec"]})
    return out


SMALL = ("alia_mrbert", "gemma_legal", "bge_m3_base", "bge_m3_legal")
CAP_SMALL_S, CAP_EULER_S = 2 * 3600, 8 * 3600


def pipeline(skip_done: bool = True) -> dict[str, Any]:
    """The PLAN order with its stop rules: smoke, the four small models (cap 2 h), Gate E0, Euler (cap 8 h), dense.

    Each job blocks until the worker returns it; a finished job (``remote/outputs/run.json`` present) is not resubmitted.
    Refits are separate heavy steps (``refit --model``), run afterwards for ``refit_candidates``."""
    log: dict[str, Any] = {"utc": C.utc(), "steps": []}

    def job(key: str, subset: str, budget: float | None = None) -> dict[str, Any]:
        jid = default_job(key, subset)
        if skip_done and (job_outputs(jid) / "run.json").exists():
            rec = {"job": jid, "skipped": "done"}
        else:
            from uol.round3 import gpu_remote
            inputs = build_inputs(key, subset, jid, budget)
            ledger({"kind": "job_start", "job_id": jid, "model": key, "subset": subset, "gpu": "remote RTX 5080"})
            receipt = gpu_remote.submit(jid, ENTRY, inputs, helpers=HELPERS,
                                        expected_outputs=["vectors.npy", "keys.json", "run.json"])
            ledger({"kind": "job_end", "job_id": jid, "model": key, "exit_code": receipt.get("exit_code"),
                    "gpu_seconds": receipt.get("seconds")})
            rec = {"job": jid, "exit_code": receipt.get("exit_code"), "seconds": receipt.get("seconds")}
        log["steps"].append(rec)
        C.write_json(RUN / "pipeline.json", log)
        if rec.get("exit_code") not in (None, 0, 3):
            raise RuntimeError(f"{jid} failed; see its job.log")
        return rec

    job("alia_mrbert", "smoke")
    used = 0.0
    for key in SMALL:
        if used >= CAP_SMALL_S:
            log["steps"].append({"model": key, "stopped": "small-model GPU cap reached"})
            continue
        rec = job(key, "full", budget=CAP_SMALL_S - used)
        used += float(rec.get("seconds") or 0)
        if rec.get("exit_code") == 3:
            log["steps"].append({"model": key, "stopped": "small-model GPU cap reached mid-encode"})
            continue
        ingest(key, default_job(key, "full"))
    job("qwen3_8b_gate", "validation")
    gate = gate_e0()
    log["gate_e0"] = {"pass": gate["pass"], "cos_mean": gate["cos_mean"], "cos_min": gate["cos_min"]}
    if gate["pass"]:
        job("euler_legal", "smoke")
        rec = job("euler_legal", "full", budget=CAP_EULER_S - float(gate.get("compute", {}).get("encode_seconds") or 0))
        if rec.get("exit_code") == 3:
            log["steps"].append({"model": "euler_legal", "stopped": "Euler GPU cap reached mid-encode"})
        else:
            ingest("euler_legal", default_job("euler_legal", "full"))
    else:
        log["steps"].append({"model": "euler_legal", "stopped": "Gate E0 failed (int8 infeasible on a 16 GB card)"})
    log["dense"] = {"refit_candidates": run_dense()["refit_candidates"]}
    C.write_json(RUN / "pipeline.json", log)
    return log


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spanish legal encoders, zero-shot (dev, descriptive)")
    ap.add_argument("command", choices=["inputs", "submit", "ingest", "gate-e0", "dense", "refit", "pipeline"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--subset", default="full", choices=["smoke", "validation", "full"])
    ap.add_argument("--job-id", default=None)
    a = ap.parse_args(argv)
    if a.command in ("inputs", "submit", "ingest", "refit") and not a.model:
        ap.error("--model is required")
    job = a.job_id or (default_job(a.model, a.subset) if a.model else None)
    if a.command == "inputs":
        print({k: str(v) for k, v in build_inputs(a.model, a.subset, job).items()})
    elif a.command == "submit":
        print(json.dumps(submit(a.model, a.subset, job), indent=1)[:2000])
    elif a.command == "ingest":
        print(json.dumps(ingest(a.model, job)["counts"]))
    elif a.command == "gate-e0":
        r = gate_e0(a.job_id or default_job("qwen3_8b_gate", "validation"))
        print(json.dumps(r, indent=1))
        return 0 if r["pass"] else 2
    elif a.command == "pipeline":
        print(json.dumps(pipeline(), indent=1, default=float))
    elif a.command == "dense":
        s = run_dense()
        print(json.dumps({"contrasts": s["contrasts"], "refit_candidates": s["refit_candidates"]}, indent=1, default=float))
    else:
        for key in a.model.split(","):
            run_refit(key)
    return 0


if __name__ == "__main__":
    sys.exit(main())
