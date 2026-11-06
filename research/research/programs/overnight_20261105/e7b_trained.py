"""E7b: trained against trained on the same 1,217 development cases. No call."""
import json
import pandas as pd
from uol.final import rank_rerun as R, audit_d181_common as A
R.D.activate(R.STORE)
base = A.per_case(R.STORE, "gpt61sol_alone")
qs = set(base["query_id"])
def stored(s):
    for st in ("experiments/runs/footing_20261101/dev/store/runs", "experiments/runs/final_20261025/store_dev_20261025b/runs", "experiments/runs/audit_d181_20261028/dev/store/runs"):
        p = A.ROOT / st / s / "DEV" / "per_case.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            f = f[f["query_id"].isin(qs)]
            print(s, "from", st.split("/")[2], "cases", len(f), "summary", round(json.loads((p.parent / "summary.json").read_text(encoding="utf-8"))["ndcg_at_10"], 4), "by group", round(f.groupby("component_id")["ndcg_at_10"].mean().mean(), 4), flush=True)
            return f
    raise FileNotFoundError(s)
late = pd.read_parquet(A.WORK / "overnight" / "mv_trained.parquet")
print("late trained, by group", round(late.groupby("component_id")["ndcg_at_10"].mean().mean(), 4), "cases", len(late))
out = {}
others = {}
for s in ("bge_m3_ft", "qwen3_0p6b_ft", "qwen8b_adapter", "qwen3_8b"):
    try:
        others[s] = stored(s)
    except FileNotFoundError:
        print("missing", s)
fam = len(others)
for s, f in others.items():
    c = next(iter(A.dev_family(f"mv_vs_{s}", [("late", late, s, f)], population="DEV", measure="ndcg_at_10", size=fam)["contrasts"].values()))
    out[s] = {x: c[x] for x in ("delta", "ci95", "adjusted_lower", "class")}
    print("late trained -", s, round(c["delta"], 4), [round(x, 3) for x in c["ci95"]], round(c["adjusted_lower"], 4), c["class"], flush=True)
(A.ROOT / "research/programs/overnight_20261105/e7b_trained.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
