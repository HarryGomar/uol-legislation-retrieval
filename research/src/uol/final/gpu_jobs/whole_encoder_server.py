"""Qwen3-Embedding-8B for the whole run (``uol.final.whole_run.Encoder``): one line of JSON in, one line out, per case.

``--device card``: int8 weights on the graphics card, as the stored run (``legal_encode.load``). ``--device cpu``: 32-bit
weights on the processor, eight threads, no card. The weights are read from the machine's cache, never fetched.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "round3" / "gpu_jobs"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scale_curve"))

MODEL = dict(model="Qwen/Qwen3-Embedding-8B", revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", dim=4096, max_tokens=8192, precision="int8")
CPU_THREADS = 8


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["card", "cpu"], required=True)
    a = ap.parse_args()
    # stdout carries the answers; everything a library prints goes to stderr
    out = sys.stdout
    sys.stdout = sys.stderr
    if a.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    scratch = Path(os.environ["UOL_ENCODER_SCRATCH"])
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    import numpy as np
    import legal_encode as LE
    LE.RB.deps(scratch, LE.ST_PINS)                # installed once on that machine by the stored encoder jobs: nothing is fetched here
    if a.device == "card":
        LE.RB.deps(scratch, LE.RB.BNB_PINS)
    LE.RB.deps(scratch, ["certifi==2024.8.30", "h11==0.16.0", "idna==3.10", "sniffio==1.3.1"])
    import torch
    task = dict(MODEL)
    if a.device == "cpu":
        torch.set_num_threads(CPU_THREADS)
        task["precision"] = "fp32"
    t0 = time.perf_counter()
    model, _ = LE.load(task, scratch, device="cpu" if a.device == "cpu" else None)
    info = {"ready": True, "device": a.device, "precision": task["precision"], "load_seconds": round(time.perf_counter() - t0, 1), "torch": torch.__version__,
            "card": torch.cuda.get_device_name(0) if a.device == "card" else None, "threads": CPU_THREADS if a.device == "cpu" else None}
    # warm up on a short text, so the first case is not charged with the first allocation
    model.encode(["calentamiento"], prompt="", batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    out.write(json.dumps(info) + "\n")
    out.flush()
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("memory"):                       # the encoder's own peaks, asked once at the end of a run
            m = {}
            if a.device == "card":
                m = {"card_allocated_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                     "card_reserved_peak_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
            try:
                import psutil
                mi = psutil.Process().memory_info()
                m["ram_now_gb"] = round(mi.rss / 2**30, 2)
                m["ram_peak_gb"] = round(getattr(mi, "peak_wset", mi.rss) / 2**30, 2)
            except Exception as exc:
                m["ram_error"] = repr(exc)
            out.write(json.dumps({"memory": m}) + "\n")
            out.flush()
            continue
        text = req["text"]
        if a.device == "card":
            torch.cuda.synchronize()
        w = time.perf_counter()
        v = model.encode([text], prompt="", batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
        if a.device == "card":
            torch.cuda.synchronize()
        s = time.perf_counter() - w
        out.write(json.dumps({"vector": base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode(), "seconds": s, "tokens": len(model.tokenizer(text)["input_ids"])}) + "\n")
        out.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
