"""v33 (4 Nov 2026, D203): the variation from asking the model again, for the sentence of Chapter 4 beside the bootstrap.

The repeat run (uol.final.llm_agents repeat) asked GPT-6.1 Sol for its list three times on the sample; the first answer is
the model-alone run. Each answer is scored on its own here, with the run's profile active (as lean_rerun and the
encoder jobs activate it), and the first must reproduce the stored model-alone score. Writes
research/programs/final_results_20261102/repeat_passes.json, which scripts/final_results.py reads.

Usage: py -3.12 research/programs/lean_rerun_20261104/repeat_passes.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]

import uol.final.footing as FT  # noqa: E402  (first: points the audit modules at the final run)
from uol.final import llm_agents as LA  # noqa: E402
from uol.embedding_push import common as ep  # noqa: E402

FT.D.activate(FT.STORE)
d = ep.release()
generic = {int(i) for i in d.generic_idx}
answers = json.loads((LA.OUT / "repeat" / "answers.json").read_text(encoding="utf-8"))
rows = sorted(int(r) for r in answers[0])
means = [float(LA.score_ranked(rows, {r: LA.resolved(a[str(r)], generic) for r in rows})["ndcg_at_10"].mean()) for a in answers]
stored = float(pd.read_parquet(LA.OUT / "alone" / "per_case.parquet")["ndcg_at_10"].mean())
assert abs(means[0] - stored) < 1e-9, (means[0], stored)
out = {"cases": len(rows), "per_answer": means, "stored_alone": stored,
       "max_abs_vs_first": max(abs(m - means[0]) for m in means[1:])}
dest = ROOT / "research" / "programs" / "final_results_20261102" / "repeat_passes.json"
dest.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out))
