"""The system's stored variants against it: reading 60, both orders, without the encoder, efforts. No call."""
import json, os
from uol.final import rank_rerun as R, audit_d181_common as A
R.D.activate(R.STORE)
base = A.per_case(R.STORE, R.SYSTEM)
ids = ["gpt61sol_rk_system_d60", "gpt61sol_rk_system_both", "gpt61sol_rk_system_noenc", "gpt61sol_rk_system_high", "gpt61sol_rk_system_low"]
have = [i for i in ids if (A.RUN / "dev" / "store" / "runs" / i / "DEV" / "per_case.parquet").exists()]
print("have", have)
out = {}
for m in ("ndcg_at_10", "recall_at_100"):
    fam = A.dev_family(f"sys_parts_{m}", [(i, A.per_case(R.STORE, i), R.SYSTEM, base) for i in have], population="DEV", measure=m)
    for k, c in fam["contrasts"].items():
        out[f"{m}:{k}"] = {x: c[x] for x in ("delta", "ci95", "adjusted_lower", "class")}
        print(m, k, round(c["delta"], 4), [round(x, 3) for x in c["ci95"]], round(c["adjusted_lower"], 4), c["class"])
for i in have + [R.SYSTEM]:
    s = json.loads((A.RUN / "dev" / "store" / "runs" / i / "DEV" / "summary.json").read_text(encoding="utf-8"))
    print(i, round(s["ndcg_at_10"], 4), round(s.get("recall_at_100", 0), 4))
(A.ROOT / "research/programs/overnight_20261105/system_parts.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
