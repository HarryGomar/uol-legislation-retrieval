"""Remote job: the Qwen3-8B query vector on a processor, one case at a time (H071, ``uol.final.rank_timing``).

No graphics card: 32-bit weights on the processor (``legal_encode.load`` with device "cpu"), a fixed number of threads.
The first three cases warm up and are left out of the times.

    python timing_encode_cpu.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a, _ = ap.parse_known_args()
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    a.out.mkdir(parents=True, exist_ok=True)
    import legal_encode as LE
    LE.RB.deps(scratch, LE.ST_PINS)
    LE.RB.deps(scratch, ["certifi==2024.8.30", "h11==0.16.0", "idna==3.10", "sniffio==1.3.1"])
    import torch
    torch.set_num_threads(int(task["threads"]))
    t0 = time.perf_counter()
    model, _path = LE.load(task["model"], scratch, device="cpu")
    load_s = time.perf_counter() - t0
    print(f"{time.strftime('%H:%M:%S')} model loaded in {load_s:.1f} s", flush=True)
    wall, tokens = [], []
    for i, r in enumerate(task["rows"]):
        text = task["inputs"][str(r)]
        w = time.perf_counter()
        model.encode([text], prompt="", batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
        s = time.perf_counter() - w
        print(f"{time.strftime('%H:%M:%S')} case {i} {s:.1f} s", flush=True)
        if i >= 3:
            wall.append(s)
            tokens.append(len(model.tokenizer(text)["input_ids"]))
    w = sorted(wall)
    res = {"cases": len(wall), "threads": int(task["threads"]), "precision": "fp32", "load_seconds": round(load_s, 1), "seconds_median": statistics.median(wall),
           "seconds_p95": w[int(0.95 * (len(w) - 1))], "tokens_median": statistics.median(tokens), "tokens_mean": statistics.mean(tokens),
           "cpu": os.environ.get("PROCESSOR_IDENTIFIER"), "logical_cpus": os.cpu_count(), "torch": torch.__version__}
    (a.out / "encode_cpu.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
