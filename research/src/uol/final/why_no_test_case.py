"""The 3 Nov "why the models work" results without the test case they carried (research/programs/why_models_20261103/ROW4806.md).

Reads the stored runs and writes new files; no model call, and the stored runs stay as they are. Every comparison is on
the same cases: row 4806 goes from the 14 runs (where it is the test case) and from their comparators (where it is a
development case), which leaves 299 cases of the sample.

    py -3.12 -m uol.final.why_no_test_case
"""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np
import pandas as pd

from uol.final import why_models as W
from uol.final import audit_d181_common as A, footing as FT

ROW, TEST_ID = 4806, "sjf:tesis:2030698:hechos"
SOURCE = W.OUT
OUT = A.WORK / "why_no_test_case"
REPORT = W.REPORT / "no_test_case"
RUNS = [f"alone_gpt61sol_{e}" for e in ("low", "high", "xhigh", "max")] + [f"alone_opus_{e}" for e in ("low", "medium", "high")] \
    + ["alone_sonnet_medium", "alone_haiku_medium", "reword_alone_gpt61sol_medium"] + [f"probe_{m}" for m in ("gpt61sol", "gpt6solnw", "opus")]
CONTROLS = [f"{c}_{s}" for c in ("bm25", "encoder") for s in ("original", "reworded")]


def derive() -> dict:
    """The stored frames and answers without the row. Frames that hold ids must hold the test id there."""
    done = {}
    for run in RUNS:
        (OUT / run).mkdir(parents=True, exist_ok=True)
        f = pd.read_parquet(SOURCE / run / "per_case.parquet")
        if "query_id" in f.columns and f.loc[f["row"] == ROW, "query_id"].tolist() != [TEST_ID]:
            raise RuntimeError(f"{run}: row {ROW} is not the test case")
        g = f[f["row"] != ROW]
        if len(g) != len(f) - 1:
            raise RuntimeError(f"{run}: expected to remove one record")
        g.to_parquet(OUT / run / "per_case.parquet")
        a = json.loads((SOURCE / run / "answers.json").read_text(encoding="utf-8"))
        a.pop(str(ROW), None)
        (OUT / run / "answers.json").write_text(json.dumps(a, ensure_ascii=False), encoding="utf-8")
        done[run] = len(g)
    (OUT / "controls").mkdir(parents=True, exist_ok=True)
    for c in CONTROLS:
        f = pd.read_parquet(SOURCE / "controls" / f"{c}.parquet")
        if f.loc[f["row"] == ROW, "query_id"].tolist() != [TEST_ID]:
            raise RuntimeError(f"controls {c}: row {ROW} is not the test case")
        f[f["row"] != ROW].to_parquet(OUT / "controls" / f"{c}.parquet")
        done[c] = int((f["row"] != ROW).sum())
    (OUT / "reword").mkdir(parents=True, exist_ok=True)
    t = json.loads((SOURCE / "reword" / "texts.json").read_text(encoding="utf-8"))
    t.pop(str(ROW), None)
    (OUT / "reword" / "texts.json").write_text(json.dumps(t, ensure_ascii=False, indent=0), encoding="utf-8")
    done["rewrites"] = len(t)
    return done


def main() -> int:
    FT.D.activate(FT.STORE)
    from uol.embedding_push import common as ep
    d = ep.release()
    print(json.dumps({"row_under_bundle": d.queries[ROW].query_id, "role_under_bundle": str(ep.roles()[ROW])}), flush=True)
    done = derive()
    print(json.dumps(done), flush=True)
    full = W.sample_rows()                                     # with the guards of 5 Nov: every row a development row
    keep = [r for r in full if r != ROW]
    W.OUT, W.REPORT = OUT, REPORT
    W.sample_rows = lambda: keep
    W._frame = lambda path, _read=pd.read_parquet: (lambda f: f[f["row"] != ROW])(_read(path))
    rec = W.report()
    # the stored receipts of W4 still hold the means with the test case: from the derived files instead
    texts = {int(k): v for k, v in json.loads((OUT / "reword" / "texts.json").read_text(encoding="utf-8")).items()}
    grams = [len(W.ngrams(d.queries[r].text) & W.ngrams(t)) / max(len(W.ngrams(d.queries[r].text)), 1) for r, t in texts.items()]
    w4 = rec["w4_reword"]
    w4["cases"] = int(len(pd.read_parquet(OUT / "controls" / "bm25_original.parquet")))
    for c in CONTROLS:
        w4[c] = float(pd.read_parquet(OUT / "controls" / f"{c}.parquet")["ndcg_at_10"].mean())
    w4["reword"] = {**w4["reword"], "cases": len(keep), "kept": len(texts), "share_4grams_kept_mean": float(np.mean(grams)),
                    "share_4grams_kept_median": float(np.median(grams))}
    rec["population"] = "DEV-A300 without row 4806"
    rec["cases"] = len(keep)
    (REPORT / "results.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    shutil.copy2(REPORT / "results.json", OUT / "results.json")
    print("saved", REPORT / "results.json", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
