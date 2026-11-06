"""E7a (overnight_20261105/PLAN.md): where each technique's gain sits. Recall@20 per cited pair, stored development runs,
intervals by group. No model call, nothing refitted, no test file opened."""
from common import *
import json

C = [("late vs single, BGE-M3", "bge_m3_colbert", "bge_m3"), ("sparse vs single, BGE-M3", "bge_m3_sparse", "bge_m3"),
     ("Qwen3-8B vs BGE-M3", "qwen3_8b", "bge_m3"), ("L* vs Qwen3-8B", LEX, "qwen3_8b"), ("adapter vs Qwen3-8B", "qwen8b_adapter", "qwen3_8b"),
     ("HyDE vs facts", "hyde", "qwen3_8b_int8q"), ("query2doc vs facts", "query2doc", "qwen3_8b_int8q"),
     ("BGE reranker alone vs list", "bge_ranker_lean_alone", RANKER), ("Qwen reranker alone vs list", "qwen4b_ranker_lean_alone", RANKER),
     ("trained reranker alone vs list", "ce_ranker_lean_alone", RANKER), ("model alone vs list", ALONE, RANKER)]
p = pairs()
d = release()
text = {q.query_id: q.text for q in d.queries}
words = lambda s: len(WORD.findall(s.lower()))
qlen = np.array([words(text[q]) for q in p["query_id"]])
alen = np.array([words(d.candidate_texts[int(a)]) for a in p["article"]])
ncited = p.groupby("query_id")["article"].transform("count").to_numpy()
B = {"word overlap": {f"Q{v}": (p["quarter"] == v).to_numpy() for v in (1, 2, 3, 4)},
     "citation band": {str(v): (p["band"] == v).to_numpy() for v in BANDS},
     "facts length": {f"Q{v + 1}": pd.qcut(qlen, 4, labels=False, duplicates="drop") == v for v in range(4)},
     "article length": {f"Q{v + 1}": pd.qcut(alen, 4, labels=False, duplicates="drop") == v for v in range(4)},
     "articles cited in the case": {"1": ncited == 1, "2-4": (ncited >= 2) & (ncited <= 4), "5+": ncited >= 5}}
print("facts words", np.quantile(qlen, [.25, .5, .75]), "article words", np.quantile(alen, [.25, .5, .75]))
boot = Boot(p["component_id"])
def draws(h, m):
    a = np.bincount(boot.idx, np.where(m, h, 0.0), boot.n)
    b = np.bincount(boot.idx, m.astype(float), boot.n)
    return (boot.w @ a) / np.maximum(boot.w @ b, 1e-12), a.sum() / max(b.sum(), 1e-12)
out = {}
for name, s, base in C:
    try:
        h, h0 = pair_hit(s), pair_hit(base)
    except FileNotFoundError as e:
        print("MISSING", name, e); continue
    res = {"all": ci(boot.diff(h, h0, np.ones(len(p))), sign=True), "ndcg": [summary(s)["ndcg_at_10"], summary(base)["ndcg_at_10"]], "by": {}}
    print(f"\n== {name}: Recall@20 {h.mean():.3f} vs {h0.mean():.3f}, gain {res['all']}, nDCG {res['ndcg'][0]:.3f} vs {res['ndcg'][1]:.3f}")
    for k, bins in B.items():
        cells, ds = {}, []
        for lab, m in bins.items():
            dd, est = draws(h - h0, m)
            ds.append((lab, dd, est))
            cells[lab] = {"pairs": int(m.sum()), "base": float(h0[m].mean()), "gain": float(est), "lo": float(np.quantile(dd, .025)), "hi": float(np.quantile(dd, .975))}
        slope = ds[-1][1] - ds[0][1]
        res["by"][k] = {"cells": cells, "last_minus_first": [float(ds[-1][2] - ds[0][2]), float(np.quantile(slope, .025)), float(np.quantile(slope, .975))]}
        print(f"{k:28s} " + " | ".join(f"{l} {c['gain']:+.3f} [{c['lo']:+.3f},{c['hi']:+.3f}]" for l, c in cells.items()) + f" || last-first {res['by'][k]['last_minus_first'][0]:+.3f} [{res['by'][k]['last_minus_first'][1]:+.3f},{res['by'][k]['last_minus_first'][2]:+.3f}]")
    out[name] = res
(ROOT / "research/programs/overnight_20261105/e7a_what_differs.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
