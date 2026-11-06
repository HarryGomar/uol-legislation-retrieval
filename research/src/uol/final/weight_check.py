"""The weight of the final order, checked on stored outputs (methods review G1, v33, 4 Nov 2026; D203).

The system's final order sorts the 30 shown candidates by z(-position in the merged list) + w z(1 / position in the
model's order), with w = 3 from ``W_EXT``, tuned when the model reranked the 28-source ranker's list. This module
recomputes the development cases from the stored answers and orders, with no model call: every w of ``W_EXT``, the
model's order alone (w -> inf), and w chosen on each fold's other four folds (ties to the smaller w). w = 3 must
reproduce the stored system case by case. It also measures the merge's one task, the share of cited articles in the
leading 30, for rank fusion, the model's list topped up with the ranker's, and the two lists alternated. Writes ``research/programs/lean_rerun_20261104/weight_check.json`` with a
paired contrast of the model's order alone minus the system (``audit_d181_common.dev_family``, 100,000 draws), which
``scripts/final_results.py`` turns into macros.

Run with the footing environment (UOL_FINAL_THREADS=4 OMP_NUM_THREADS=4 UOL_FINAL_LABELS=corrected ...):
``py -3.12 -m uol.final.weight_check``.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from uol.final import footing as FT  # first: it points the audit modules at the common-footing run

FT.RANKER = "rt_lm_tesis"
FT.AUDIT = FT.STORE
A = FT.A
SYSTEM = "gpt61sol_rk_system"
OUT = A.ROOT / "research" / "programs" / "whole_run_20261105" / "weight_check.json"


def _ndcg10(ranked: list[int], pos: set[int]) -> float:
    if not pos:
        return 0.0
    dcg = sum(1.0 / np.log2(1 + k) for k, c in enumerate(ranked[:10], 1) if c in pos)
    return dcg / sum(1.0 / np.log2(1 + k) for k in range(1, min(len(pos), 10) + 1))


def main() -> int:
    FT.D.activate(FT.STORE)
    from uol.embedding_push import common as ep
    from uol.final import external as X
    from uol.final import llm_agents as G
    from uol.final import measures
    from uol.round3.c_hybrid import W_EXT
    pc = pd.read_parquet(FT.STORE / "runs" / SYSTEM / "DEV" / "per_case.parquet").set_index("row")
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = G.all_rows()
    answers = json.loads((G.OUT / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    model = {r: G.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    rk = G.ranker_lists(rows, "rt_lm_tesis")
    merged = {r: G.rrf([model[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    orders = json.loads((A.WORK / "agents_rk" / SYSTEM / "orders.json").read_text(encoding="utf-8"))["orders"]
    pos = {r: {int(x) for x in str(pc.loc[r, "positive_idx"]).replace(";", ",").split(",") if x.strip()} for r in rows}

    def cases(w: float) -> dict[int, float]:
        return {r: _ndcg10(G.fuse_order(shown[r], orders, ["fwd"], r, w) + merged[r][X.TOP:], pos[r]) for r in rows}

    def macro(v: dict[int, float], sub: set[int] | None = None) -> float:
        rr = [r for r in rows if sub is None or r in sub]
        return float(np.mean(measures.per_component(np.asarray([v[r] for r in rr]), pc.loc[rr, "component_id"].to_numpy())[1]))

    grid = sorted({float(x) for x in W_EXT})
    by_w = {w: cases(w) for w in grid}
    alone = cases(1e9)
    stored = max(abs(by_w[3.0][r] - float(pc.loc[r, "ndcg_at_10"])) for r in rows)
    if stored > 1e-9:
        raise RuntimeError(f"w = 3 does not reproduce {SYSTEM}: largest difference {stored}")
    fold = {r: int(pc.loc[r, "fold"]) for r in rows}
    chosen, tuned = {}, {}
    for k in sorted(set(fold.values())):
        train = {r for r in rows if fold[r] != k}
        chosen[k] = max(grid, key=lambda w: (round(macro(by_w[w], train), 12), -w))
        tuned.update({r: by_w[chosen[k]][r] for r in rows if fold[r] == k})

    def frame(v: dict[int, float]) -> pd.DataFrame:
        return pd.DataFrame({"query_id": pc.loc[rows, "query_id"].to_numpy(), "component_id": pc.loc[rows, "component_id"].to_numpy(),
                             "ndcg_at_10": [v[r] for r in rows]})

    fam = A.dev_family("weight_check", [("model_order_alone", frame(alone), SYSTEM, frame(by_w[3.0]))])

    # the merge's one task, recall in the leading X.TOP: rank fusion against the model's list topped up with the
    # ranker's, and the two lists alternated (methods review, soundness table, RRF row)
    rkg = {r: [c for c in rk[r] if c not in generic] for r in rows}

    def alternate(r: int) -> list[int]:
        out, a, b = [], list(model[r]), list(rkg[r])
        while len(out) < X.TOP and (a or b):
            for src in (a, b):
                while src:
                    c = src.pop(0)
                    if c not in out:
                        out.append(c)
                        break
        return out[:X.TOP]

    lead = {"rank_fusion": lambda r: merged[r][:X.TOP],
            "model_topped_up": lambda r: (list(dict.fromkeys(model[r])) + [c for c in rkg[r] if c not in set(model[r])])[:X.TOP],
            "alternated": alternate, "ranker_only": lambda r: rkg[r][:X.TOP]}
    recall = {k: macro({r: len(set(f(r)) & pos[r]) / len(pos[r]) if pos[r] else 0.0 for r in rows}) for k, f in lead.items()}
    res = {"system": SYSTEM, "grid": grid, "w_system": 3.0, "reproduces_stored": True,
           "macro": {**{f"w={w:g}": macro(by_w[w]) for w in grid}, "model_order_alone": macro(alone),
                     "w_tuned_per_fold": macro(tuned)},
           "w_chosen_per_fold": {str(k): w for k, w in chosen.items()},
           "folds_choosing_top": sum(w == max(grid) for w in chosen.values()),
           "recall_leading": {"depth": X.TOP, "macro": recall}, **fam}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("macro", "w_chosen_per_fold", "folds_choosing_top", "recall_leading")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
