"""The nine additions to the chosen ranker, the family stated in rank_rerun_20261105/PLAN.md, nDCG@10. No call."""
import json
from uol.final import rank_rerun as R, audit_d181_common as A
R.D.activate(R.STORE)
ids = ["rk_structure", "rk_cocite", "rk_adapter", "rk_smooth", "rk_issues", "rk_patterns", "rk_q2d", "rk_hyde", "rk_topics"]
base = A.per_case(R.STORE, R.BASE)
fam = A.dev_family("rk_additions", [(i, A.per_case(R.STORE, i), R.BASE, base) for i in ids], population="DEV")
out = {}
for k, c in fam["contrasts"].items():
    out[k] = {x: c[x] for x in ("delta", "ci95", "adjusted_lower", "class")}
    print(k, round(c["delta"], 4), [round(x, 3) for x in c["ci95"]], round(c["adjusted_lower"], 4), c["class"])
print("size", fam["size"])
(A.ROOT / "research/programs/overnight_20261105/additions_family.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
