"""Posts a whole-run job to the remote worker.

    py -3.12 research/programs/whole_run_20261105/post.py smoke|full [--suffix x]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
WORK = ROOT / ".cache" / "whole_run"
STEPS = {
    "smoke": [{"setup": "noenc", "encoder": "none", "cases": 2, "tag": "smoke"}, {"setup": "ranker", "encoder": "cpu", "cases": 2, "tag": "smoke"},
              {"setup": "system", "encoder": "card", "cases": 2, "tag": "smoke"}],
    "full": [{"setup": "noenc", "encoder": "none", "cases": 200, "tag": "a"}, {"setup": "ranker", "encoder": "card", "cases": 200, "tag": "a"},
             {"setup": "ranker", "encoder": "cpu", "cases": 200, "tag": "a"}, {"setup": "system", "encoder": "card", "cases": 50, "tag": "a"},
             {"setup": "system", "encoder": "card", "cases": 50, "tag": "b"}],
    "memory": [{"setup": "ranker", "encoder": "card", "cases": 50, "tag": "mem", "env": {"UOL_WHOLE_MEMORY": "1"}}],
    "rest": [{"setup": "ranker", "encoder": "card", "cases": 200, "tag": "a"}, {"setup": "ranker", "encoder": "cpu", "cases": 200, "tag": "a"},
             {"setup": "system", "encoder": "card", "cases": 50, "tag": "a"}, {"setup": "system", "encoder": "card", "cases": 50, "tag": "b"}],
}
# changed since the pack was made: the guard keeps the past cases' vectors; the weight of the reranking on the chosen ranker
OVERLAY = {"overlay_whole_run.py": ("src/uol/final/whole_run.py", ROOT / "src/uol/final/whole_run.py"),
           "overlay_encoder_server.py": ("src/uol/final/gpu_jobs/whole_encoder_server.py", ROOT / "src/uol/final/gpu_jobs/whole_encoder_server.py"),
           "overlay_weight_meta.json": (".cache/footing/dev/llm/gpt61sol_rerank_rt_lm_tesis/meta.json", ROOT / ".cache/footing/dev/llm/gpt61sol_rerank_rt_lm_tesis/meta.json")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=list(STEPS))
    ap.add_argument("--suffix", default="1")
    a = ap.parse_args()
    from uol.round3 import gpu_remote as G
    job_id = f"whole_{a.kind}_{a.suffix}"
    task = WORK / f"task_{job_id}.json"
    task.write_text(json.dumps({"pack": json.loads((WORK / "whole_pack.json").read_text(encoding="utf-8")), "steps": STEPS[a.kind], "smoke_codex": a.kind == "smoke", "overlay": {k: v[0] for k, v in OVERLAY.items()}}, indent=1), encoding="utf-8")
    job = G.pack(job_id, Path(__file__).parent / "remote_job.py", {"task.json": task, **{k: v[1] for k, v in OVERLAY.items()}}, expected_outputs=["status.json"])
    G.push(job)
    print("posted", job_id, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
