"""GPU job (RUN_PLAN v2 §3.5 ``bge_m3_ft``): BGE-M3 fine-tuned on both towers, contrastively with hard negatives, for
one outer fold, then the catalogue units and the fold's held-out development queries are encoded with it. The same
script fine-tunes Qwen3-Embedding-0.6B (``qwen3_0p6b_ft``, D89): the model, revision and encode settings come from
``task.json``.

Inputs (written on the run computer by ``uol.final.gpu_prep bge_m3_ft --fold k`` from the fold's training rows only):
``train.jsonl`` ({"q": masked facts, "pos": cited article text, "negs": [hard negative article texts]}),
``items.jsonl`` (legal_encode format: article units + the held-out fold's queries), ``task.json`` ({"model",
"revision", "fold", "epochs", "lr", "batch", "tau", "max_train_tokens", "max_tokens", "seed", "budget_seconds"}).
Loss: InfoNCE over the positive, the row's hard negatives and every other document of the batch (temperature tau);
AdamW, linear warm-up 10 %, bf16 autocast, one epoch by default. Outputs as legal_encode: ``vectors.npy``,
``keys.json``, ``run.json`` (plus the training curve). Private pinned installs (``rerank_bench.deps``).

    python bge_m3_ft.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np

# Fewer fragmentation failures on a full 16 GB card (allocator setting only; 22 Oct 2026, smoke__bge_m3_ft OOM).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for cand in (HERE.parents[1] / "round3" / "gpu_jobs",):
    if cand.exists():
        sys.path.insert(0, str(cand))
import rerank_bench as RB  # noqa: E402
import legal_encode as LE  # noqa: E402


def to_device(feats: dict, device: str) -> dict:
    """Move the tensors of a tokenized batch; sentence-transformers 6 also returns non-tensor entries (strings such as
    the input modality), which stay as they are (as its own ``batch_to_device``; 21 Oct 2026, smoke__bge_m3_ft)."""
    return {k: (v.to(device) if hasattr(v, "to") and not isinstance(v, str) else v) for k, v in feats.items()}


# VRAM guard: legal_encode's (free memory at job start minus 0.5 GiB, against PyTorch's reserved pool), shared by every
# GPU encode and fine-tune (seq 24)
VRAM_LIMIT_GIB, VRAM_MARGIN_GIB, GUARD_SECONDS = LE.VRAM_LIMIT_GIB, LE.VRAM_MARGIN_GIB, LE.GUARD_SECONDS
vram_limit, vram_gib, reserved_gib, check_vram, vram_line, start_guard = (
    LE.vram_limit, LE.vram_gib, LE.reserved_gib, LE.check_vram, LE.vram_line, LE.start_guard)


# Checkpoint (seq 24): the trained weights are saved before the encode; a rerun whose training inputs and settings hash
# the same loads them and skips training, so a failed encode costs no GPU time. Kept until the run is packed (LEAD).
TRAIN_KEYS = ("model", "revision", "epochs", "lr", "batch", "tau", "max_train_tokens", "seed", "n_hard")


def train_hash(task: dict, train_path: Path) -> str:
    h = hashlib.sha256(train_path.read_bytes())
    h.update(json.dumps({k: task.get(k) for k in TRAIN_KEYS}, sort_keys=True).encode())
    return h.hexdigest()


def weights_sha(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*.safetensors")):
        h.update(p.relative_to(folder).as_posix().encode())
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
    return h.hexdigest()


def load_checkpoint(ckpt: Path, key: str) -> dict | None:
    """The checkpoint record when its training hash and weights match; None otherwise."""
    rec_p = ckpt / "CHECKPOINT.json"
    if not rec_p.exists():
        return None
    rec = json.loads(rec_p.read_text(encoding="utf-8"))
    if rec.get("train_hash") != key or rec.get("weights_sha256") != weights_sha(ckpt):
        return None
    return rec


def save_checkpoint(model, ckpt: Path, key: str, info: dict) -> dict:
    if ckpt.exists():
        import shutil
        shutil.rmtree(ckpt)
    model.save(str(ckpt))
    rec = {"train_hash": key, "weights_sha256": weights_sha(ckpt), **info}
    (ckpt / "CHECKPOINT.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


def planned(task: dict) -> bool:
    """Qwen3 fine-tunes encode exactly like the base encode (legal_encode, seq 24); bge_m3_ft keeps its own call."""
    return task.get("encode") == "planned" or str(task.get("model", "")).startswith("Qwen/")


def encode_planned(model, texts: list[str], tok, task: dict, torch, limit: float) -> "np.ndarray":
    """legal_encode's encode: SDPA without enable_gqa (the memory-efficient kernel; the math kernel materialised
    4 x 16 x 8192^2 fp32 scores, 35.7 GiB, on unpadded 8,192-token batches of 4), batches planned by tokens over
    length-sorted texts (32,768 tokens, attention cap 2^23 cells, at most 64 texts), fp32, no default prompt."""
    LE.avoid_gqa_math_sdpa()
    lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
    dim = model.get_sentence_embedding_dimension()
    V = np.zeros((len(texts), dim), dtype=np.float32)
    done, t0 = 0, time.perf_counter()
    for b in LE.S.plan_batches(lengths, max_tokens=int(task.get("batch_tokens", 32768)), max_inputs=64):
        with torch.inference_mode():
            v = model.encode([texts[i] for i in b], prompt="", batch_size=len(b), convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        V[b] = np.asarray(v, dtype=np.float32)
        done += len(b)
        n_batches = LE.RUNTIME["batches"] = LE.RUNTIME.get("batches", 0) + 1
        if n_batches % LE.EMPTY_EVERY == 0:
            LE.empty_cache(torch, f"encode batch {n_batches}")
        check_vram(torch, f"encode {done}/{len(texts)}", limit)
        if done % 2000 < len(b):
            RB.log(f"encode {done}/{len(texts)} texts, {time.perf_counter() - t0:.0f} s; {vram_line(torch, limit)}")
    return V


def release_training_state(model, opt, sched, torch) -> None:
    """Free the optimizer states and gradients before the encode (the catalogue encode runs up to 8,192 tokens for
    Qwen3-0.6B); the trained weights are untouched."""
    opt.state.clear()
    opt.param_groups.clear()
    del sched
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, LE.ST_PINS)
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    seed = int(task.get("seed", 20261004))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = snapshot_download(task["model"], revision=task["revision"], cache_dir=str(scratch / "hf"))
    limit = vram_limit(torch)
    RB.log(f"vram guard limit {limit:.2f} GiB (free at start minus {VRAM_MARGIN_GIB} GiB)")
    ckpt = a.out.parent / "checkpoint"
    key = train_hash(task, a.inputs / "train.jsonl")
    rec = load_checkpoint(ckpt, key)
    if rec is not None:
        RB.log(f"checkpoint {ckpt} matches (train hash {key[:12]}): training skipped")
        return encode_and_write(SentenceTransformer(str(ckpt), device="cuda", local_files_only=True), task, a, torch,
                                limit, rec, ckpt, Path(path).name)
    model = SentenceTransformer(path, device="cuda", local_files_only=True)
    model.max_seq_length = int(task.get("max_train_tokens", 512))
    # Gradient checkpointing: activations are recomputed in the backward pass instead of stored. The loss, the
    # gradients and every setting are unchanged; only memory falls. Batch 16 with three hard negatives each did not fit
    # 16 GB with AdamW states (22 Oct 2026, smoke__bge_m3_ft: 14.8 GiB allocated by this job alone).
    hf = getattr(model[0], "auto_model", None) or getattr(model[0], "model", None)    # sentence-transformers 5 / 6
    if hf is None or not hasattr(hf, "gradient_checkpointing_enable"):
        raise RuntimeError(f"cannot enable gradient checkpointing on the {task['model']} transformer module")
    hf.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    rows = [json.loads(line) for line in (a.inputs / "train.jsonl").read_text(encoding="utf-8").splitlines() if line]
    random.shuffle(rows)
    B, tau, epochs = int(task.get("batch", 16)), float(task.get("tau", 0.05)), int(task.get("epochs", 1))
    steps = epochs * ((len(rows) + B - 1) // B)
    # foreach=False: the same AdamW update one tensor at a time, without the multi-tensor copies of every parameter
    opt = torch.optim.AdamW(model.parameters(), lr=float(task.get("lr", 1e-5)), weight_decay=0.01, foreach=False)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.1 * steps))) *
                                              max(0.0, (steps - s) / max(1, steps - int(0.1 * steps))))
    torch.cuda.reset_peak_memory_stats()
    curve, t0, step = [], time.perf_counter(), 0
    budget = float(task.get("budget_seconds") or "inf")
    model.train()
    phase, stop = {"name": "train"}, threading.Event()
    start_guard(torch, phase, stop, limit)
    RB.log(f"training: {len(rows)} rows, {steps} steps, batch {B}")

    def embed(texts):
        feats = to_device(model.tokenize(texts), "cuda")
        out = model(feats)["sentence_embedding"]
        return torch.nn.functional.normalize(out, dim=-1)

    stopped = False
    for ep in range(epochs):
        for s in range(0, len(rows), B):
            if time.perf_counter() - t0 > budget:
                stopped = True
                break
            batch = rows[s:s + B]
            docs, owner = [], []
            for i, r in enumerate(batch):
                for d in [r["pos"]] + list(r["negs"]):
                    docs.append(d)
                    owner.append(i)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                q = embed([r["q"] for r in batch])
                dv = embed(docs)
                logits = (q @ dv.T).float() / tau
            pos_idx = torch.tensor([owner.index(i) for i in range(len(batch))], device="cuda")
            loss = torch.nn.functional.cross_entropy(logits, pos_idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            opt.step()
            sched.step()
            step += 1
            if step % 20 == 0:
                curve.append({"step": step, "loss": float(loss.item())})
                RB.log(f"step {step}/{steps} loss {float(loss.item()):.4f}; {vram_line(torch, limit)}")
                check_vram(torch, f"step {step}", limit)
        if stopped:
            break
    train_peak = vram_gib(torch)[1]
    model.eval()
    release_training_state(model, opt, sched, torch)
    del opt
    stop.set()
    RB.log(f"training done: peak {train_peak:.2f} GiB; optimizer and gradients freed, {vram_gib(torch)[0]:.2f} GiB allocated")
    rec = {"train_rows": len(rows), "steps": step, "curve": curve, "incomplete": stopped,
           "train_seconds": round(time.perf_counter() - t0, 1), "train_peak_gib": round(train_peak, 2)}
    if not stopped:
        rec = save_checkpoint(model, ckpt, key, rec)
        RB.log(f"checkpoint saved: {ckpt} (weights sha256 {rec['weights_sha256'][:12]})")
    return encode_and_write(model, task, a, torch, limit, rec, ckpt if not stopped else None, Path(path).name)


def encode_and_write(model, task: dict, a, torch, limit: float, rec: dict, ckpt: Path | None, snapshot: str) -> int:
    torch.cuda.reset_peak_memory_stats()
    phase, stop = {"name": "encode"}, threading.Event()
    start_guard(torch, phase, stop, limit)
    t0 = time.perf_counter()
    model.eval()
    trained_dtype = str(next(model.parameters()).dtype)
    if planned(task):
        model.to(torch.float32)                      # the base encode's precision (legal_encode loads fp32)
    model.max_seq_length = int(task.get("max_tokens", 2048))
    tok = model.tokenizer
    items = [json.loads(line) for line in (a.inputs / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    keys, texts, item_keys = [], [], []
    for it in items:
        parts = LE.windows(it["text"], tok, int(task.get("max_tokens", 2048))) if it["g"] == "articles" else \
            [LE.truncate(it["text"], tok, int(task.get("max_tokens", 2048)))[0]]
        ks = []
        for j, t in enumerate(parts):
            k = f"{it['g']}:{it['o']}:{it['w']}:{j}"
            keys.append(k)
            texts.append(t)
            ks.append(k)
        item_keys.append([it["i"], ks])
    if planned(task):
        RB.log(f"encode: {len(texts)} texts, max {int(task.get('max_tokens', 2048))} tokens, planned batches "
               f"({int(task.get('batch_tokens', 32768))} tokens, attention cap 2^23), as legal_encode")
        V = encode_planned(model, texts, tok, task, torch, limit)
    else:
        RB.log(f"encode: {len(texts)} texts, max {int(task.get('max_tokens', 2048))} tokens, batch {int(task.get('encode_batch', 16))}")
        with torch.inference_mode():
            # bge_m3_ft: its call as before (XLM-R has no grouped-query attention; 2,048 tokens)
            kw = {"prompt": ""} if task.get("no_model_prompt") else {}
            V = model.encode(texts, batch_size=int(task.get("encode_batch", 16)), convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False, **kw)
    stop.set()
    encode_peak = vram_gib(torch)[1]
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "vectors.npy", np.asarray(V, dtype=np.float32))
    (a.out / "keys.json").write_text(json.dumps({"items": item_keys, "keys": keys}), encoding="utf-8")
    train_peak = float(rec.get("train_peak_gib") or 0)
    run = {"task": task, "incomplete": bool(rec.get("incomplete")), "train_rows": rec.get("train_rows"),
           "steps": rec.get("steps"), "curve": rec.get("curve"), "train_seconds": rec.get("train_seconds"),
           "encode_seconds": round(time.perf_counter() - t0, 1),
           "checkpoint": {"path": str(ckpt) if ckpt else None, "train_hash": rec.get("train_hash"),
                          "weights_sha256": rec.get("weights_sha256")},
           "encode_path": "planned (legal_encode)" if planned(task) else "sentence-transformers encode, batch "
                          f"{int(task.get('encode_batch', 16))}", "vram_limit_gib": round(limit, 2),
           "peak_vram_mib": round(max(train_peak, encode_peak) * 1024), "peak_vram_gib": {"train": round(train_peak, 2),
                                                                                       "encode": round(encode_peak, 2)},
           "weights_dtype": {"trained": trained_dtype, "encoded": str(next(model.parameters()).dtype)},
           "vram_guard": {k: LE.RUNTIME.get(k) for k in ("empty_cache_calls", "batches", "vram_guard")},
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0), "snapshot": snapshot}
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    return 3 if rec.get("incomplete") else 0


if __name__ == "__main__":
    raise SystemExit(main())
