"""Does the eight-bit query encoder change the ranking? (v30 follow-up, the author 3 Nov 2026: "lets fix this ... re run things").

The scores of the thesis use the stored Qwen3-Embedding-8B query vectors, which the embedding endpoint computed at full
precision. At query time the ranker encodes the facts locally with the model's weights in eight bits (the timing job
``timing_encode_ranker`` and ``uol.final.qwen_local``), whose vectors agree with the stored ones at cosine about 0.99.
This module measures what that does to the encoder's own ranking on every development case:

* ``post``: one GPU job on the 5080 worker (``timing_encode_ranker.py``, the same model, loader and inputs as the timed
  ranker) encodes the facts of the development cases one at a time; no label travels.
* ``pull``: waits for it, verifies the output hashes and copies the outputs to ``<OUT>/encode``.
* ``score``: MaxP scores of both vector sets against the stored article windows (``qwen_endpoint.dense_block``),
  nDCG@10 per case on the specific labels with generic articles removed (``qwen_endpoint.Ev``), the component-macro
  means, the paired difference with a 95% group bootstrap, the cosine of each pair and the share of cases with the same
  first ten. Writes ``<OUT>/RESULT.json``; ``scripts/final_results.py`` reads it.

    py -3.12 -m uol.final.query_int8 post --go
    py -3.12 -m uol.final.query_int8 pull --go
    py -3.12 -m uol.final.query_int8 score
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "runs" / "footing_20261101" / "int8"
DEV_SYSTEM = "S0+qwen3_8b"                    # any footing system's per-case rows are the 1,217 scored development cases
JOB = "query_int8_dev_j1"
N_WARMUP = 5
WAIT_S = 6 * 3600


def dev_rows() -> list[int]:
    """The release rows of the development cases scored in the thesis (the footing store's per-case file, by query id:
    its ``row`` column follows another order for a few cases)."""
    sys.path.insert(0, str(ROOT / "src"))
    from uol.embedding_push import common
    from uol.thesis_figures import v21_figs as V
    from uol.final import guard
    pc = V.per_case(DEV_SYSTEM)
    if pc is None:
        raise SystemExit(f"no per-case file for {DEV_SYSTEM}")
    index = {q.query_id: i for i, q in enumerate(common.release().queries)}
    rows = guard.dev_rows_only(sorted(index[q] for q in pc["query_id"]), "int8 rows")
    return [int(r) for r in rows]


def post() -> int:
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    from uol.embedding_push import common, qwen_endpoint as qe
    from uol.round3 import gpu_remote as G
    from uol.scale_curve import legal
    d = common.release()
    rows = dev_rows()
    warm = rows[:N_WARMUP]
    OUT.mkdir(parents=True, exist_ok=True)
    task = OUT / "encode_task.json"
    task.write_text(json.dumps({"timed": rows, "warmup": warm, "model": {**legal.GATE["qwen3_8b_gate"], "key": "qwen3_8b_int8_dev"},
                                "inputs": {str(r): qe.query_input("c", d.queries[r].text) for r in rows}}, ensure_ascii=False),
                    encoding="utf-8")
    jobs = Path(G.__file__).parent / "gpu_jobs"
    job = G.pack(JOB, Path(__file__).parent / "gpu_jobs" / "timing_encode_ranker.py", {"task.json": task},
                 helpers=[jobs / "legal_encode.py", jobs / "rerank_bench.py", ROOT / "src" / "uol" / "scale_curve" / "store.py"],
                 expected_outputs=["encode.npz", "encode.json"])
    G.push(job)
    print(f"posted {JOB}: {len(rows)} development cases, {len(warm)} warm-up", flush=True)
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


def score() -> int:
    from uol.embedding_push import qwen_endpoint as qe
    from uol.final import measures
    from uol.round3 import stats as R3
    z = np.load(OUT / "encode" / "encode.npz")
    keep = ~z["warmup"]
    rows = z["rows"][keep].astype(np.int64)
    v8 = z["vectors"][keep].astype(np.float32)
    v8 /= np.linalg.norm(v8, axis=1, keepdims=True)
    want = dev_rows()
    if sorted(rows.tolist()) != want:
        raise SystemExit("encoded rows differ from the development cases")
    full = qe.query_vectors("c")[rows]
    cos = np.einsum("ij,ij->i", full, v8)
    ev = qe.Ev()
    s_full, s_8 = qe.dense_block(full), qe.dense_block(v8)
    n_full, n_8 = ev.ndcg_spec(s_full, rows), ev.ndcg_spec(s_8, rows)
    tie = ev.tie

    def top10(block: np.ndarray) -> np.ndarray:
        b = block.copy()
        if ev.generic is not None:
            b[:, ev.generic] = -np.inf
        return np.argsort(-b, axis=1, kind="stable")[:, :10]

    same10 = float((top10(s_full) == top10(s_8)).all(axis=1).mean())
    # paired contrast by the frozen procedure (component means, 10,000 component-bootstrap draws, outcome class)
    comp = ev.comp[rows]
    _, v8c = measures.per_component(n_8, comp)
    _, vfc = measures.per_component(n_full, comp)
    con = R3.contrast(v8c, vfc)
    res = {"what": "Qwen3-Embedding-8B alone on the development cases: stored full-precision query vectors against the "
                   "eight-bit local encoder of the timed ranker; same article windows, labels and evaluator",
           "cases": int(len(rows)), "groups": int(len(v8c)),
           "ndcg_full": ev.pooled(n_full, rows), "ndcg_int8": ev.pooled(n_8, rows), "contrast_int8_minus_full": con,
           "cases_changed": int((np.abs(n_8 - n_full) > 1e-12).sum()),
           "cos": {"mean": float(cos.mean()), "median": float(np.median(cos)), "min": float(cos.min())},
           "same_first_ten_share": same10, "tie_order": "common.tie_order" if tie is not None else None,
           "encode": json.loads((OUT / "encode" / "encode.json").read_text(encoding="utf-8"))}
    (OUT / "RESULT.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "encode"}, indent=1), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("post", "pull", "score"))
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    if a.step in ("post", "pull") and not a.go:
        print("dry: add --go", flush=True)
        return 0
    return {"post": post, "pull": pull, "score": score}[a.step]()


if __name__ == "__main__":
    raise SystemExit(main())
