"""Phase 0 rebuild driver: run an original ``uol.embedding_push`` stage unchanged, with ledger rows.

The frozen run folder ``experiments/runs/embedding_push_20261004`` was lost on 2026-10-10; Round 3 regenerates it at its
original path with the original code (whose SHA-256 still matches FREEZE.json). The only intervention is path
redirection for model snapshots that moved after the source-home cleanup: the same Hugging Face snapshot hashes are
loaded from wherever they still exist (``MODEL_DIRS``). No scientific setting changes.

    python -m uol.round3.rebuild <module> [module args...]
    e.g. python -m uol.round3.rebuild base all
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LEDGER = ROOT / "research/state/round3_best_case_20261013/RUN_LEDGER.jsonl"
HF_USER = Path.home() / ".cache/huggingface/hub"
# same snapshot hashes as uol.embedding_push.common; checked for existence before use
MODEL_DIRS = {
    # e5-small 614241f6: the 8 files of facts_to_law.dense.MODEL_FILES, every one hash-verified by finetune_dense
    "E5_DIR": ROOT / ".cache/models/e5_small_614241f6",
    "BGE_DIR": ROOT / ".cache/models/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def patch_paths() -> dict:
    """Redirect moved model snapshots (before the target module is imported, so module-level copies see them) and
    route the original code's GPU subprocesses (``-m uol.embedding_push.<mod>``) through this driver as well."""
    import subprocess
    from uol.embedding_push import common
    from uol.facts_to_law import dense
    applied = {}
    for name, path in MODEL_DIRS.items():
        if path.exists() and not Path(getattr(common, name)).exists():
            setattr(common, name, path)
            applied[name] = str(path)
    if not Path(dense.MODEL_DIR).exists() and MODEL_DIRS["E5_DIR"].exists():
        dense.MODEL_DIR = MODEL_DIRS["E5_DIR"]
        applied["facts_to_law.dense.MODEL_DIR"] = str(MODEL_DIRS["E5_DIR"])
    if not getattr(subprocess.run, "_round3", False):
        orig = subprocess.run

        def run(cmd, *a, **k):
            if isinstance(cmd, (list, tuple)):
                cmd = list(cmd)
                for i in range(len(cmd) - 1):
                    if cmd[i] == "-m" and str(cmd[i + 1]).startswith("uol.embedding_push."):
                        cmd[i + 1:i + 2] = ["uol.round3.rebuild", str(cmd[i + 1]).split(".")[-1]]
                        break
            return orig(cmd, *a, **k)
        run._round3 = True
        subprocess.run = run
    return applied


def peak_rss_gb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 2)
    except Exception:
        return None


def main(argv: list[str]) -> int:
    module, args = argv[0], argv[1:]
    job_id = f"rebuild_{module}_{'_'.join(a.lstrip('-') for a in args)[:60]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    applied = patch_paths()
    mod = importlib.import_module(f"uol.embedding_push.{module}")
    code_sha = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
    start, t0 = _utc(), time.time()
    ledger({"kind": "job_start", "job_id": job_id, "utc": start, "command": f"python -m uol.round3.rebuild {' '.join(argv)}",
            "pid": os.getpid(), "path_patches": applied, "code_sha256": code_sha})
    rc = 1
    try:
        ret = mod.main(args)
        rc = int(ret or 0)
    except SystemExit as exc:
        rc = int(exc.code or 0) if isinstance(exc.code, int) else 1
        raise
    finally:
        ledger({"kind": "job_end", "job_id": job_id, "start_utc": start, "end_utc": _utc(), "exit_code": rc,
                "wall_seconds": round(time.time() - t0, 1), "peak_ram_gb": peak_rss_gb(), "pid": os.getpid()})
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
