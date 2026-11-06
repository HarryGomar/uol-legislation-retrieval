"""R1: zero-shot Qwen3-Reranker 4B (bf16) and 8B (NF4, D0.9) over the LTR_emb_hgb top 50 (PLAN §3 R1; Phase 4.1).

Pairs: every dev P_spec row x its LTR_emb_hgb top 50 (evaluator tie rule), plus one blank-query pair per candidate
(the blank-query prior control). Document text: "<law title> › <heading path> › Art. <label>: <body>" with heading
paths from the label-free G2 (phase2/statute_graph/paths.json); MaxP over 1,024-token chunks (≤ 4). English
instruction as the model card recommends. Dev rows only; no labels are shipped.

    python -m uol.round3.r1_zeroshot submit --model qwen3_rr_4b                  # blocks; run in the background
    python -m uol.round3.r1_zeroshot submit --model qwen3_rr_8b --quant 4bit
    python -m uol.round3.r1_zeroshot evaluate                                     # fusion route, depth sweep, controls
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from uol.round3 import common as C
from uol.round3 import features, gpu_remote, rerank_setup

OUT = C.ARMS / "R1"
INPUTS = OUT / "inputs"
ENTRY = Path(__file__).resolve().parent / "gpu_jobs" / "r1_score.py"
HELPER = Path(__file__).resolve().parent / "gpu_jobs" / "rerank_bench.py"
DEPTH = 50
DEPTHS = (20, 30, 50)


def doc_text(i: int, paths: dict) -> str:
    d = C.release()
    title, _, body = d.candidate_texts[i].partition("\n")
    label = d.candidate_ids[i].split("#articulo=")[-1]
    path = " › ".join(re.sub(r"\s+", " ", p) for p in paths.get(str(i), []))
    head = f"{title.strip()} › {path} › Art. {label}" if path else f"{title.strip()} › Art. {label}"
    return f"{head}: {body.strip()}"


def build_inputs() -> dict:
    rows = C.population_rows("P_spec", C.dev_rows())
    if np.isin(rows, C.lockbox_rows()).any():
        raise PermissionError("dev rows only")
    top = features.ltr_topk("LTR_emb_hgb", rows, DEPTH)
    paths = json.loads((C.RUN / "phase2/statute_graph/paths.json").read_text(encoding="utf-8"))
    d = C.release()
    cands = np.unique(top)
    INPUTS.mkdir(parents=True, exist_ok=True)
    with (INPUTS / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"qid": int(r), "text": d.queries[int(r)].text}, ensure_ascii=False) + "\n")
    with (INPUTS / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps({"did": int(c), "text": doc_text(int(c), paths)}, ensure_ascii=False) + "\n")
    pairs = np.concatenate([np.stack([np.repeat(rows, DEPTH), top.ravel()], 1),
                            np.stack([np.full(len(cands), -1), cands], 1)]).astype(np.int32)
    np.save(INPUTS / "pairs.npy", pairs)
    C.write_json(INPUTS / "index.json", {"rows": rows.tolist(), "depth": DEPTH, "n_pairs": int(len(pairs)),
                                         "n_blank": int(len(cands))})
    return {"queries.jsonl": INPUTS / "queries.jsonl", "docs.jsonl": INPUTS / "docs.jsonl", "pairs.npy": INPUTS / "pairs.npy"}


def submit(model: str, quant: str | None, job_id: str) -> dict:
    inputs = build_inputs()
    key = f"r1_{model}_{quant or 'bf16'}_pspec_top{DEPTH}"
    task = {"model": model, "quant": quant, "instruction": rerank_setup.INSTRUCTION, "max_query_tokens": 384,
            "chunk_tokens": 1024, "max_chunks": 4, "batch_size": 8, "key": key}
    tp = INPUTS / f"task_{key}.json"
    tp.write_text(json.dumps(task, indent=1), encoding="utf-8")
    inputs["task.json"] = tp
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": job_id, "gpu": "remote RTX 5080", "task": task})
    receipt = gpu_remote.submit(job_id, ENTRY, inputs, helpers=[HELPER], expected_outputs=["scores.npy"])
    C.ledger({"kind": "job_end", "job_id": job_id, "start_utc": start, "end_utc": C.utc(),
              "exit_code": receipt.get("exit_code"), "gpu_seconds": receipt.get("seconds")})
    return receipt


JOBS = {"4B": "r1_qwen3_rr_4b_bf16_pspec_s1", "8B": "r1_qwen3_rr_8b_4bit_pspec_s1"}


def matrices() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """model -> (rows, A [rows, N] top-50 scores, blank-query score per candidate [N] (NaN where unscored))."""
    d = C.release()
    idx = json.loads((INPUTS / "index.json").read_text(encoding="utf-8"))
    rows = np.asarray(idx["rows"])
    pairs = np.load(INPUTS / "pairs.npy")
    out = {}
    for m, job in JOBS.items():
        f = gpu_remote.LOCAL_JOBS / job / "remote" / "outputs" / "scores.npy"
        if not f.exists():
            continue
        sc = np.load(f)
        A = np.full((len(rows), len(d.candidate_ids)), np.nan, dtype=np.float32)
        blank = np.full(len(d.candidate_ids), np.nan, dtype=np.float32)
        pos = {int(r): i for i, r in enumerate(rows)}
        for (q, c), v in zip(pairs, sc):
            if q < 0:
                blank[int(c)] = v
            else:
                A[pos[int(q)], int(c)] = v
        out[m] = (rows, A, blank)
    return out


def at_depth(rows: np.ndarray, A: np.ndarray, depth: int) -> np.ndarray:
    top = features.ltr_topk("LTR_emb_hgb", rows, depth)
    B = np.full_like(A, np.nan)
    for i, t in enumerate(top):
        B[i, t] = A[i, t]
    return B


def evaluate() -> dict:
    from uol.round3 import arm_runner
    mats = matrices()
    configs, controls = {}, {}
    for m, (rows, A, blank) in mats.items():
        for depth in DEPTHS:
            configs[f"{m}_top{depth}"] = (rows, at_depth(rows, A, depth))
        Bq = np.where(np.isfinite(A), blank[None, :], np.nan).astype(np.float32)
        controls[f"blank_query_prior_{m}_top50"] = (rows, Bq)
    return arm_runner.run_fusion("R1", configs, controls=controls,
                                 subgroups={"low_overlap": C.low_overlap,
                                            "rare_band": lambda rr: np.asarray([any(C.band_of_row(int(r))[a] == "rare"
                                                                                    for a in C.release().specific_idx[int(r)]) for r in rr])},
                                 notes="zero-shot Qwen3-Reranker 4B bf16 / 8B NF4 (D0.9), MaxP over 1,024-token chunks, English "
                                       "instruction; depth sweep 20/30/50 on dev; blank-query prior control",
                                 extra_result={"compute": {"gpu_jobs": list(JOBS.values())}})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["submit", "build", "evaluate"])
    ap.add_argument("--model", default="qwen3_rr_4b")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--job-id", default=None)
    a = ap.parse_args()
    if a.command == "build":
        print(build_inputs())
        return 0
    if a.command == "evaluate":
        r = evaluate()
        print(json.dumps(r["dev"]["vs_EMB"], indent=1, default=float)[:1500])
        return 0
    job = a.job_id or f"r1_{a.model}_{a.quant or 'bf16'}_pspec_s1"
    print(json.dumps(submit(a.model, a.quant, job), indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
