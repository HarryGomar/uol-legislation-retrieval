"""Values for S7's sections 6.5, 6.7 and 6.9 (S7_macro_request.md, 5 Nov) that results.json does not already hold.
Development cases only, stored runs, no model call, no test file."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import OUTDIR, pairs, pair_hit, pc

out: dict = {}

# 6.5: Qwen3-Reranker-4B alone on the articles cited more than ten times, pooled over cited pairs (as dbRankMost)
p = pairs()
most = (p["band"] == ">10").to_numpy()
out["ce_qwen_alone_most"] = float(pair_hit("qwen4b_ranker_rk_alone")[most].mean())
out["ranker_most_check"] = float(pair_hit("rt_lm_tesis")[most].mean())

# 6.5 and 6.7: the model alone at low and high effort against medium, nDCG@10, means over groups, family of two
from uol.final import new_cases_report as NR
frames = {k: pc(k) for k in ("gpt61sol_alone", "gpt61sol_alone_low", "gpt61sol_alone_high")}
for k in ("low", "high"):
    out[f"alone_{k}_minus_medium"] = NR.pair(frames, f"gpt61sol_alone_{k}", "gpt61sol_alone", "ndcg_at_10", 2)

(OUTDIR / "s7_macros.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: (v if not isinstance(v, dict) else {x: v.get(x) for x in ("delta", "ci95", "class", "adjusted_lower")}) for k, v in out.items()}, indent=1, default=float))
