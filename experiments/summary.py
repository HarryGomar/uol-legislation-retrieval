"""Development results of the thesis from the stored per-case scores (``results/dev_per_case.parquet``).

    python experiments/summary.py                                   the main systems (Table 6.1, development part)
    python experiments/summary.py footing/rk_noenc oct25/qwen3_8b   any stored systems
    python experiments/summary.py --compare footing/gpt61sol_rk_system footing/rt_lm_tesis

A system is named ``<record>/<id>``; ``--list`` prints every stored one. Means are over groups of cases (each group's
cases averaged first); a comparison is the mean paired difference over groups with its 95% bootstrap interval.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uol.evaluate import bootstrap_means  # noqa: E402

HERE = Path(__file__).resolve().parent
MEASURES = {"ndcg_at_10": "nDCG@10", "success_at_1": "first", "success_at_10": "in ten", "recall_at_20": "recall@20"}


def load() -> pd.DataFrame:
    return pd.read_parquet(HERE / "results" / "dev_per_case.parquet")


def cases(t: pd.DataFrame, name: str) -> pd.DataFrame:
    record, system = name.split("/", 1)
    out = t[(t.record == record) & (t.system == system)]
    if out.empty:
        raise SystemExit(f"no stored system {name!r} (see --list)")
    return out


def group_means(c: pd.DataFrame, measure: str) -> pd.Series:
    return c.groupby("component_id")[measure].mean()


def describe(t: pd.DataFrame, names: list[str], labels: list[str] | None = None) -> None:
    labels = labels or names
    width = max(map(len, labels))
    print(f"{'system':<{width}}  cases  " + "  ".join(f"{v:>9}" for v in MEASURES.values()))
    for label, name in zip(labels, names):
        c = cases(t, name)
        print(f"{label:<{width}}  {len(c):>5}  " + "  ".join(f"{group_means(c, m).mean():>9.3f}" for m in MEASURES))


def compare(t: pd.DataFrame, a: str, b: str, measure: str = "ndcg_at_10") -> None:
    ca, cb = cases(t, a), cases(t, b)
    shared = sorted(set(ca.query_id) & set(cb.query_id))
    ga = group_means(ca[ca.query_id.isin(shared)], measure)
    gb = group_means(cb[cb.query_id.isin(shared)], measure)
    diff = (ga - gb.reindex(ga.index)).to_numpy()
    low, high = np.quantile(bootstrap_means(diff), [0.025, 0.975])
    print(f"{a} - {b}: {diff.mean():+.4f} [{low:+.4f}, {high:+.4f}] over {len(diff)} groups ({len(shared)} cases)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("systems", nargs="*")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    t = load()
    if a.list:
        for (record, system), n in t.groupby(["record", "system"]).size().items():
            print(f"{record}/{system}  {n}")
    elif a.compare:
        compare(t, *a.compare)
    elif a.systems:
        describe(t, a.systems)
    else:
        main_systems = json.loads((HERE / "results" / "MAIN.json").read_text(encoding="utf-8"))
        describe(t, [f"{v['record']}/{v['system']}" for v in main_systems.values()], list(main_systems))


if __name__ == "__main__":
    main()
