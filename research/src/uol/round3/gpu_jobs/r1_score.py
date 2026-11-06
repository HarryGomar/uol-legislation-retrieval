"""Remote job (R1): zero-shot reranker scores for (dev query, candidate article) pairs, MaxP over document chunks.

Inputs: ``queries.jsonl`` ({"qid", "text"}: masked dev Hechos), ``docs.jsonl`` ({"did", "text"}: "Ley › path › Art. N:
body"), ``pairs.npy`` (int32 [n, 2]: qid, did; qid -1 = blank-query control), ``task.json`` ({"model", "quant",
"instruction", "max_query_tokens", "chunk_tokens", "max_chunks", "batch_size", "key"}). Uses ``rerank_bench`` (shipped as
a helper) for model loading and scoring. Progress is checkpointed in ``UOL_WORKER_SCRATCH/r1/<key>/`` every 2,000 pairs,
so a rerun with the same key resumes. Output: ``scores.npy`` (float32 [n]: max over chunks of the model score).

    python r1_score.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_bench as RB  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, RB.PINS)
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    if task.get("quant"):
        RB.deps(scratch, RB.BNB_PINS)
    queries = {int(r["qid"]): r["text"] for r in map(json.loads, (a.inputs / "queries.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    docs = {int(r["did"]): r["text"] for r in map(json.loads, (a.inputs / "docs.jsonl").read_text(encoding="utf-8").splitlines()) if r}
    pairs = np.load(a.inputs / "pairs.npy")
    from huggingface_hub import snapshot_download
    RB.no_symlinks()
    repo, rev, kind = RB.MODELS[task["model"]]
    path = snapshot_download(repo, revision=rev, cache_dir=str(scratch / "hf"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
    m = (RB.Qwen if kind == "qwen" else RB.Bge)(path, task.get("quant"))
    tok = m.tok
    qtxt = {q: RB.truncate(tok, t, task["max_query_tokens"]) for q, t in queries.items()}
    qtxt[-1] = ""
    chunks = {}
    for d, t in docs.items():
        ids = tok(t, add_special_tokens=False)["input_ids"]
        n = task["chunk_tokens"]
        chunks[d] = [tok.decode(ids[i:i + n]) for i in range(0, max(len(ids), 1), n)][:task["max_chunks"]]
    ck_dir = scratch / "r1" / task["key"]
    ck_dir.mkdir(parents=True, exist_ok=True)
    ck = ck_dir / "scores_partial.npy"
    scores = np.load(ck) if ck.exists() else np.full(len(pairs), np.nan, dtype=np.float32)
    flat = [(i, c) for i, (q, d) in enumerate(pairs) for c in range(len(chunks[int(d)]))]
    todo = [x for x in flat if np.isnan(scores[x[0]])]
    RB.log(f"{task['model']} {task.get('quant')}: {len(pairs)} pairs, {len(flat)} chunk pairs, {len(todo)} to do")
    best = {}
    t0 = time.perf_counter()
    bs = task["batch_size"]
    done_since = 0
    for s in range(0, len(todo), bs):
        batch = todo[s:s + bs]
        texts = [(qtxt[int(pairs[i][0])], chunks[int(pairs[i][1])][c]) for i, c in batch]
        out = m.score(task["instruction"], texts)
        for (i, c), v in zip(batch, out):
            best[i] = max(best.get(i, -np.inf), float(v))
        done_since += len(batch)
        if done_since >= 2000 or s + bs >= len(todo):
            # a pair is final once all of its chunks are scored: chunk pairs are ordered by pair index
            last = batch[-1][0]
            for i in [k for k in list(best) if k < last or s + bs >= len(todo)]:
                scores[i] = best.pop(i)
            np.save(ck, scores)
            done_since = 0
            RB.log(f"{np.isfinite(scores).sum()}/{len(scores)} pairs, {(s + len(batch)) / (time.perf_counter() - t0):.1f} chunk pairs/s")
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "scores.npy", scores)
    (a.out / "run.json").write_text(json.dumps({"task": task, "pairs": int(len(pairs)), "chunk_pairs": len(flat),
                                                "seconds": round(time.perf_counter() - t0, 1),
                                                "nan": int(np.isnan(scores).sum())}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
