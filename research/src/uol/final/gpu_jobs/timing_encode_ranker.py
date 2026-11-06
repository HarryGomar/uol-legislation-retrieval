"""Remote job: the Qwen3-8B query encoder of the ranker's whole-method timing (``uol.final.timing_ranker``, D196).

Qwen3-Embedding-8B int8, loaded as ``legal_encode.load`` loads it, encodes one development case at a time on the GPU:
the warm-up cases first, then the timed ones, each timed with ``torch.cuda.synchronize`` around tokenisation, the
forward pass and the copy to host. Loading is timed apart. Outputs: ``encode.npz`` (rows, warmup, vectors, seconds,
cpu_seconds) and ``encode.json`` (load seconds, card, peak card memory, peak process RAM, versions).

    python timing_encode_ranker.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def peak_ram_gb() -> float | None:
    """Peak working set of this process (Windows), in GB; None where it cannot be read."""
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                                               "PagefileUsage", "PeakPagefileUsage")]
        c = Counters()
        c.cb = ctypes.sizeof(c)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return round(c.PeakWorkingSetSize / 1e9, 2)
    except Exception:
        return None


def card_used() -> str | None:
    """The card's memory in use, all processes (``nvidia-smi``), as a check that the job has the card to itself."""
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return None


def encode_cases(task: dict, encode, sync) -> list[tuple]:
    """[(row, warm-up flag, vector, seconds, process CPU seconds)]: the warm-up rows first, then the timed rows."""
    out = []
    for flag, rows in ((True, task["warmup"]), (False, task["timed"])):
        for r in rows:
            text = task["inputs"][str(r)]
            sync()
            w, c = time.perf_counter(), time.process_time()
            v = encode(text)
            sync()
            out.append((int(r), flag, np.asarray(v, dtype=np.float32), time.perf_counter() - w, time.process_time() - c))
    return out


def save(recs: list[tuple], out: Path) -> None:
    np.savez(out / "encode.npz", rows=np.asarray([x[0] for x in recs], dtype=np.int64), warmup=np.asarray([x[1] for x in recs], dtype=bool),
             vectors=np.stack([x[2] for x in recs]), seconds=np.asarray([x[3] for x in recs]), cpu_seconds=np.asarray([x[4] for x in recs]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a, _ = ap.parse_known_args()
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    a.out.mkdir(parents=True, exist_ok=True)
    before = card_used()
    import legal_encode as LE
    LE.RB.deps(scratch, LE.ST_PINS)
    LE.RB.deps(scratch, LE.RB.BNB_PINS)
    LE.RB.deps(scratch, ["certifi==2024.8.30", "h11==0.16.0", "idna==3.10", "sniffio==1.3.1"])      # what httpcore imports; the overlay installs with --no-deps
    import torch
    t0 = time.perf_counter()
    model, _path = LE.load(task["model"], scratch)
    load_s = time.perf_counter() - t0
    print(f"{time.strftime('%H:%M:%S')} model loaded in {load_s:.1f} s", flush=True)

    def encode(text: str):
        return model.encode([text], prompt="", batch_size=1, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]

    recs = encode_cases(task, encode, torch.cuda.synchronize)
    save(recs, a.out)
    s = np.asarray([x[3] for x in recs if not x[1]])
    meta = {"model": task["model"], "load_seconds": round(load_s, 2), "cases": len(s), "warmup_cases": len(task["warmup"]),
            "seconds_median": float(np.median(s)), "seconds_p95": float(np.percentile(s, 95)), "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
            "card_used_before_load": before, "card_used_after_cases": card_used(), "peak_process_ram_gb": peak_ram_gb(),
            "torch": torch.__version__, "cuda": torch.version.cuda, "stack": LE.stack_versions(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (a.out / "encode.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
