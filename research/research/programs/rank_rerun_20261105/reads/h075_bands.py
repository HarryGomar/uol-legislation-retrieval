"""H075: the two panels of Figure 6.3 (F6_16_depends), read from depends.json, plus two checks on the panels' reading:
whether word-overlap quarter and citation band are independent, and whether the overlap quarter is a proxy for the
article's length (the share has article words in its denominator)."""
from common import *

dep = json.loads(DEPENDS.read_text(encoding="utf-8"))
print("pairs", dep["pairs"], "cases", dep["cases"], "groups", dep["groups"], "edges", dep["quarter_edges"], dep["share_min"], dep["share_max"])
for key in ("a", "b"):
    pan = dep["panels"][key]
    print("panel", key, pan["labels"])
    for name, m in pan["methods"].items():
        vals = [m["values"][b] for b in pan["bins"]]
        print(f"  {name:30s}", "  ".join(f"{v['mean']:.3f} [{v['lo']:.3f},{v['hi']:.3f}]" for v in vals), "n", [v["n"] for v in vals])
p = pairs().copy()
print(pd.crosstab(p["quarter"], p["band"], normalize="index")[BANDS].round(3))
d = release()
L = {int(a): len(WORD.findall(d.candidate_texts[int(a)].lower())) for a in p["article"].unique()}
p["len"] = p["article"].map(L)
print("article words by quarter", p.groupby("quarter")["len"].median().to_dict())
print("article words by band", p.groupby("band")["len"].median().to_dict())
p["lenq"] = pd.qcut(p["len"], 2, labels=["short", "long"])
b = Boot(p["component_id"])
for sid in (LEX, ENC, RANKER, ALONE):
    h = pair_hit(sid)
    rows = {}
    for lq in ("short", "long"):
        m = (p["lenq"] == lq).to_numpy()
        lo, hi = (m & (p["quarter"] == 1).to_numpy()), (m & (p["quarter"] == 4).to_numpy())
        rows[lq] = (round(h[lo].mean(), 3), int(lo.sum()), round(h[hi].mean(), 3), int(hi.sum()))
    print(sid, "within length half: (Q1 recall, n, Q4 recall, n)", rows)
for sid in (LEX, ENC, RANKER, ALONE, SYSTEM, NEWRANKER):
    h = pair_hit(sid)
    q1, q4 = (p["quarter"] == 1).to_numpy(), (p["quarter"] == 4).to_numpy()
    r1, r4 = b.ratio(h * q1, q1), b.ratio(h * q4, q4)
    print(sid, "Q1", ci(r1), "Q4", ci(r4), "rise", round(r4[0] - r1[0], 3), "ratio Q4/Q1", round(r4[0] / r1[0], 2))
    print("   by band", {bd: round(h[(p["band"] == bd).to_numpy()].mean(), 3) for bd in BANDS})
