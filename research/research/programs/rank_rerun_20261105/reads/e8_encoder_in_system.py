"""Why the encoder stays in the system: the system with and without it, on what each finds. Stored runs, no call."""
from common import *
import json
S, N = "gpt61sol_rk_system", "gpt61sol_rk_system_noenc"
A0, B0 = "rt_lm_tesis", "rk_noenc"
out = {}
for s in (S, N, A0, B0):
    j = summary(s)
    out[s] = {k: j.get(k) for k in ("ndcg_at_10", "success_at_1", "recall_at_10", "recall_at_20", "recall_at_100", "unseen_recall_at_20_pooled", "unseen_recall_at_100_pooled", "law_at_1")}
    print(s, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[s].items()})
p = pairs()
boot = Boot(p["component_id"])
for k in (10, 20, 100):
    h, h0 = pair_hit(S, k), pair_hit(N, k)
    print(f"\nrecall@{k} pooled, with encoder {h.mean():.3f}, without {h0.mean():.3f}, diff {ci(boot.diff(h, h0, np.ones(len(p))), sign=True)}")
    res = {}
    for name, col, bins in (("band", "band", BANDS), ("overlap", "quarter", [1, 2, 3, 4])):
        for v in bins:
            m = (p[col] == v).to_numpy()
            t = boot.diff(h * m, h0 * m, m)
            res[f"{name} {v}"] = [int(m.sum()), float(h[m].mean()), float(h0[m].mean()), *t]
            print(f"  {name} {v!s:5s} n={m.sum():4d} with {h[m].mean():.3f} without {h0[m].mean():.3f} diff {ci(t, sign=True)}")
    out[f"recall_at_{k}"] = res
hS, hN = pair_hit(S, 30), pair_hit(N, 30)
print("\nfirst 30 pooled", hS.sum(), hN.sum(), "of", len(p))
(ROOT / "research/programs/overnight_20261105/encoder_in_system.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
