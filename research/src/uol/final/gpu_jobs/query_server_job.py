"""Remote job: Qwen3-Embedding-8B int8 held in memory, encoding the search queries of the development agents as they
arrive (common_footing PLAN addendum 13). The model loads once, exactly as ``legal_encode.load`` loads it.

Requests are files ``req_<id>.json`` ({"texts": [...]}) in ``<exchange>/<folder>``; each answer is ``rep_<id>.npy``
(float32 [n, 4096], unit norm). ``ready.json`` says the model is up. The job stops on ``STOP``, when no request has come
for ``idle_s``, or after ``total_s``. Development queries only: the test set never reaches this folder.

    python query_server_job.py --inputs inputs --out outputs
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


def serve(folder: Path, encode, idle_s: float, total_s: float, poll_s: float = 3.0) -> dict:
    t0 = last = time.time()
    served = queries = 0
    while True:
        if (folder / "STOP").exists():
            why = "stop file"
            break
        if time.time() - last > idle_s:
            why = "idle"
            break
        if time.time() - t0 > total_s:
            why = "time limit"
            break
        for req in sorted(folder.glob("req_*.json")):
            rep = folder / f"rep_{req.stem[4:]}.npy"
            if rep.exists():
                continue
            try:
                texts = json.loads(req.read_text(encoding="utf-8"))["texts"]
            except (OSError, ValueError, KeyError):
                continue                               # still syncing; the next poll reads it
            tmp = folder / f"tmp_{req.stem[4:]}.npy"
            np.save(tmp, np.asarray(encode(texts), dtype=np.float32))
            os.replace(tmp, rep)
            served += 1
            queries += len(texts)
            last = time.time()
            print(f"{time.strftime('%H:%M:%S')} {req.name}: {len(texts)} queries", flush=True)
        time.sleep(poll_s)
    return {"stopped": why, "requests": served, "queries": queries, "seconds": round(time.time() - t0, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a, _ = ap.parse_known_args()
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    folder = Path(task["exchange"]) / task["folder"]
    if not folder.is_dir():
        print(f"no folder {folder}", flush=True)
        return 3
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    import legal_encode as LE
    LE.RB.deps(scratch, LE.ST_PINS)
    LE.RB.deps(scratch, LE.RB.BNB_PINS)
    LE.RB.deps(scratch, ["certifi==2024.8.30", "h11==0.16.0", "idna==3.10", "sniffio==1.3.1"])      # what httpcore imports; the overlay installs with --no-deps
    model, _path = LE.load(task["model"], scratch)

    def encode(texts):
        return model.encode(list(texts), prompt="", batch_size=16, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    (folder / "ready.json").write_text(json.dumps({"utc": time.time()}), encoding="utf-8")
    res = serve(folder, encode, float(task["idle_s"]), float(task["total_s"]))
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "served.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
