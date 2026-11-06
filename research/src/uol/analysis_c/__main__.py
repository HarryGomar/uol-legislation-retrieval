"""CLI: ``py -3.12 -m uol.analysis_c <item> --runs <folder> --out <folder>``.

Items: c1, c3, c4, c6, c7, c8, c9, recall, d102, noise (§13a), f4 (§13b), all. Each item writes into ``<out>/<item>/`` (JSON, CSV tables, evidence_rows.md).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from uol.analysis_c import common as C

ITEMS = ("c1", "c3", "c4", "c6", "c7", "c8", "c9", "recall", "d102", "noise", "f4")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.analysis_c", description=__doc__)
    ap.add_argument("item", choices=ITEMS + ("all",))
    ap.add_argument("--runs", required=True, type=Path, help="folder of <system>/<population>/{per_case,top200}.parquet")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--data", type=Path, default=C.DATA, help="DEV-only data pack (release/, round3/statute_graph/)")
    ap.add_argument("--population", default="DEV")
    ap.add_argument("--systems", nargs="*", default=None, help="restrict c1/c4/c9/recall to these systems")
    ap.add_argument("--draws", type=int, default=C.DRAWS)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--answer", action="append", default=[], help="C7: PX=<system> override (e.g. P2=...)")
    ap.add_argument("--costs", type=Path, default=None, help="C8: cost records folder (default <runs>/../cost)")
    ap.add_argument("--prices", type=Path, default=None, help="C8: prices JSON (cited, dated)")
    ap.add_argument("--receipts", type=Path, default=None, help="recall: receipts folder (default <runs>/../receipts)")
    ap.add_argument("--stage", action="append", default=[],
                    help="d102: BASE=STAGE[@POPULATION] (repeatable; replaces the plan's stage list when given)")
    ap.add_argument("--law-base", default=None, help="d102: base of the law-first rules (default S_full_ext)")
    ap.add_argument("--stand-in", action="append", default=[], help="d102: SYSTEM=PLAN_SYSTEM label for a stand-in")
    a = ap.parse_args(argv)
    items = ITEMS if a.item == "all" else (a.item,)
    for item in items:
        t0 = time.time()
        out = a.out / item
        common = dict(draws=a.draws, seed=a.seed)
        if item == "c1":
            from uol.analysis_c import c1_law as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, systems=a.systems, **common)
        elif item == "c3":
            from uol.analysis_c import c3_gain as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, **common)
        elif item == "c4":
            from uol.analysis_c import c4_near as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, systems=a.systems, **common)
        elif item == "c6":
            from uol.analysis_c import c6_lay as m
            rec = m.run(a.runs, out, data=a.data, systems=a.systems, **common)
        elif item == "c7":
            from uol.analysis_c import c7_conf as m
            ans = dict(x.split("=", 1) for x in a.answer)
            rec = m.run(a.runs, out, data=a.data, population=a.population, answer=ans, **common)
        elif item == "c8":
            from uol.analysis_c import c8_frontier as m
            rec = m.run(a.runs, out, population=a.population, costs_dir=a.costs, prices=a.prices, **common)
        elif item == "c9":
            from uol.analysis_c import c9_length as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, systems=a.systems, **common)
        elif item == "d102":
            from uol.analysis_c import d102 as m
            stages = None
            if a.stage:
                stages = {}
                for x in a.stage:
                    b, rest = x.split("=", 1)
                    st, _, pop = rest.partition("@")
                    stages.setdefault(b, []).append((st, pop or a.population))
            rec = m.run(a.runs, out, data=a.data, population=a.population, systems=a.systems, receipts=a.receipts,
                        stages=stages, law_base=a.law_base or m.LAW_BASE,
                        stand_in=dict(x.split("=", 1) for x in a.stand_in), **common)
        elif item == "f4":                             # §13b (D109): F4 provenance share
            from uol.analysis_c import f4_provenance as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, **common)
        elif item == "noise":                          # §13a (D109): label-noise simulation
            from uol.analysis_c import noise as m
            rec = m.run(a.runs, out, data=a.data, population=a.population, seed=m.SEED)
        else:
            from uol.analysis_c import recall_sweep as m
            rec = m.run(a.runs, out, data=a.data, systems=a.systems, receipts=a.receipts, **common)
        print(json.dumps({"item": item, "out": str(out), "seconds": round(time.time() - t0, 1),
                          "notes": rec.get("notes", [])}, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
