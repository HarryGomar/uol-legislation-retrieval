"""GPU job (final benchmark, PLAN_DRAFT §3 "Learned sparse"): bge-m3 lexical (sparse) weights of units and queries.

Same job format as ``round3/gpu_jobs/legal_encode.py``: inputs ``items.jsonl`` ({"i", "g", "o", "w", "text"}; article
units and development query strings, no labels) and ``task.json``; outputs ``sparse.npz`` (CSR [keys, vocab] of
float32 weights), ``keys.json`` ({"items": [[i, [keys...]], ...], "keys": [...]}) and ``run.json``.

The sparse head is the one of BGE-M3 (Chen et al. 2024, the FlagEmbedding ``_sparse_embedding`` rule): token weight
= ReLU(sparse_linear(last hidden state)); special tokens dropped; the weight of a vocabulary id is the maximum over
its occurrences. Units longer than ``max_tokens`` are cut into title-prefixed windows (``legal_encode.windows``),
queries are truncated (``legal_encode.truncate``). Private pinned installs as in legal_encode (``rerank_bench.deps``).

    python bge_m3_sparse.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for cand in (HERE.parents[1] / "round3" / "gpu_jobs",):         # src/uol/round3/gpu_jobs when run from the package
    if cand.exists():
        sys.path.insert(0, str(cand))
import rerank_bench as RB  # noqa: E402
import legal_encode as LE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, RB.PINS)
    import threading
    import torch
    from huggingface_hub import snapshot_download
    from scipy import sparse
    limit = LE.vram_limit(torch)                       # seq 26: the shared-card cap and the guard of legal_encode
    guard_stop = threading.Event()
    LE.start_guard(torch, {"name": "sparse encode"}, guard_stop, limit)
    from transformers import AutoModel, AutoTokenizer
    items = [json.loads(line) for line in (a.inputs / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if task.get("limit"):
        items = items[:int(task["limit"])]
    t_load = time.perf_counter()
    path = snapshot_download(task["model"], revision=task["revision"], cache_dir=str(scratch / "hf"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "sparse_linear.pt",
                                             "pytorch_model.bin"])
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True, dtype=torch.float32).cuda().eval()
    head = torch.nn.Linear(model.config.hidden_size, 1)
    head.load_state_dict(torch.load(Path(path) / "sparse_linear.pt", map_location="cpu"))
    head = head.cuda().eval()
    special = {tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id}
    load_s = time.perf_counter() - t_load
    work, item_keys = {}, []
    for it in items:
        texts = LE.windows(it["text"], tok, task["max_tokens"]) if it["g"] == "articles" else \
            [LE.truncate(it["text"], tok, task["max_tokens"])[0]]
        ks = []
        for t in texts:
            k = f"{it['g']}:{it['o']}:{it['w']}:{len(ks)}"
            work[k] = t
            ks.append(k)
        item_keys.append([it["i"], ks])
    keys = list(work)
    lengths = [len(tok(work[k], add_special_tokens=False)["input_ids"]) for k in keys]
    order = np.argsort(lengths)[::-1]
    rows, cols, vals = [], [], []
    t0, n_tok = time.perf_counter(), 0
    budget = float(task.get("budget_seconds") or "inf")
    row_of = {k: i for i, k in enumerate(keys)}
    batch: list[int] = []

    def flush(batch: list[int]) -> None:
        nonlocal n_tok
        enc = tok([work[keys[j]] for j in batch], padding=True, truncation=True, max_length=task["max_tokens"],
                  return_tensors="pt").to("cuda")
        with torch.inference_mode():
            h = model(**enc).last_hidden_state
            w = torch.relu(head(h)).squeeze(-1) * enc["attention_mask"]
        ids, ww = enc["input_ids"].cpu().numpy(), w.float().cpu().numpy()
        for b, j in enumerate(batch):
            best: dict[int, float] = {}
            for t, x in zip(ids[b], ww[b]):
                if int(t) in special or x <= 0:
                    continue
                if x > best.get(int(t), 0.0):
                    best[int(t)] = float(x)
            for t, x in best.items():
                rows.append(row_of[keys[j]])
                cols.append(t)
                vals.append(x)
        n_tok += int(enc["attention_mask"].sum())
        LE.check_vram(torch, f"sparse batch of {len(batch)}", limit)

    budget_tokens = int(task.get("batch_tokens") or 16384)
    cur = 0
    stopped = False
    for j in order:
        if time.perf_counter() - t0 > budget:
            stopped = True
            break
        L = lengths[j] + 2
        if batch and (cur + L > budget_tokens or len(batch) >= 64):
            flush(batch)
            batch, cur = [], 0
        batch.append(int(j))
        cur += L
    if batch and not stopped:
        flush(batch)
    a.out.mkdir(parents=True, exist_ok=True)
    M = sparse.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(keys), len(tok)))
    sparse.save_npz(a.out / "sparse.npz", M)
    (a.out / "keys.json").write_text(json.dumps({"items": item_keys, "keys": keys}), encoding="utf-8")
    run = {"task": task, "incomplete": stopped, "items": len(items), "texts": len(keys), "load_seconds": round(load_s, 1),
           "encode_seconds": round(time.perf_counter() - t0, 1), "tokens_encoded": n_tok, "vocab": len(tok),
           "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20), "torch": torch.__version__,
           "device": torch.cuda.get_device_name(0), "snapshot": Path(path).name}
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
