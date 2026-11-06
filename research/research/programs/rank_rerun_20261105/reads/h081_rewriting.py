"""H081: all query rewriting on stored development runs. Alone on the eight-bit Qwen3-8B path (hyde, query2doc, against
qwen3_8b_int8q); in the rankers (S0, S_full, ft_ranker, ft_lean). What each source reads is from src/uol/final/gpu_prep.py
(query2doc = facts + the saved L2 issue list), round3/l2_features.py (issues) and round3/l4*.py (patterns)."""
from common import *

print("== alone")
for s in ("qwen3_8b", "qwen3_8b_int8q", "query2doc", "hyde"):
    j = summary(s)
    print(f"{s:16s} nDCG {j['ndcg_at_10']:.3f} first {j['success_at_1']:.3f} R20 {j['recall_at_20']:.3f} R100 {j['recall_at_100']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f}")
b0 = pc("qwen3_8b_int8q").set_index("query_id")
bc = Boot(b0["component_id"].to_numpy())
for s in ("query2doc", "hyde"):
    print(s, "- int8q", ci(bc.group_mean(pc(s).set_index("query_id").loc[b0.index, "ndcg_at_10"].to_numpy() - b0["ndcg_at_10"].to_numpy()), sign=True))

print("\n== in a ranker, each minus its base, paired, mean over groups")
FAM = [("S0", ("S0+qwen3_8b", "S0+query2doc", "S0+hyde")), ("S0+qwen3_8b", ("S0+query2doc", "S0+hyde")), ("S_full", ("S_full+query2doc", "S_full+hyde")),
       ("ft_ranker", ("ft_q2d", "ft_hyde", "ft_issues", "ft_patterns")), (RANKER, ("lf_q2d", "lf_hyde", "lf_issues", "lf_patterns"))]
rows = []
for base, arms in FAM:
    B = pc(base).set_index("query_id")
    bx = Boot(B["component_id"].to_numpy())
    jb = summary(base)
    for a in arms:
        A = pc(a).set_index("query_id").loc[B.index]
        g = bx.group_mean(A["ndcg_at_10"].to_numpy() - B["ndcg_at_10"].to_numpy())
        ja = summary(a)
        print(f"{a:18s} on {base:12s} {ja['ndcg_at_10']:.3f} (base {jb['ndcg_at_10']:.3f}) {ci(g, sign=True)} | never {ja['unseen_recall_at_20_pooled']:.3f} vs {jb['unseen_recall_at_20_pooled']:.3f} | R20 {ja['recall_at_20']:.3f} vs {jb['recall_at_20']:.3f} | in law {ja['success_at_1_in_law']:.3f} vs {jb['success_at_1_in_law']:.3f} | law1 {ja['law_at_1']:.3f} vs {jb['law_at_1']:.3f}")
res = json.loads(RESULTS.read_text(encoding="utf-8"))
print("\nstored classes:")
for key, c in list(res["experiments"]["rewriting"].items()) + [(k, v) for k, v in res["experiments"]["additions"].items() if "issues" in k or "patterns" in k]:
    print(key, round(c["delta"], 4), [round(x, 4) for x in c["ci95"]], c["class"], "adjusted", [round(x, 4) for x in c["ci_adjusted"]])

print("\n== by band and overlap, Recall@20")
p = pairs()
b = Boot(p["component_id"])
H = {s: pair_hit(s) for s in ("qwen3_8b_int8q", "query2doc", "hyde", RANKER, "lf_q2d", "lf_hyde", "lf_issues", "lf_patterns")}
for col, bins in (("quarter", [1, 2, 3, 4]), ("band", BANDS)):
    for v in bins:
        m = (p[col] == v).to_numpy()
        print(f"{col} {v!s:5s} n={m.sum():4d} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H) + f" | q2d-enc {ci(b.diff(H['query2doc'] * m, H['qwen3_8b_int8q'] * m, m), sign=True)} | issues-ranker {ci(b.diff(H['lf_issues'] * m, H[RANKER] * m, m), sign=True)}")

print("\n== do they gain on the same cases? correlation of per-case gain over the ranker")
B = pc(RANKER).set_index("query_id")["ndcg_at_10"]
G = {s: (pc(s).set_index("query_id").loc[B.index, "ndcg_at_10"] - B).to_numpy() for s in ("lf_q2d", "lf_hyde", "lf_issues", "lf_patterns", "lf_seed1", "lf_tesis")}
print(pd.DataFrame(G).corr(method="pearson").round(2).to_string())
print("share of cases changed", {k: round(float((v != 0).mean()), 3) for k, v in G.items()})
print("\nthe rewriting's text: facts + L2 issues is query2doc; hyde text excludes the facts")
print("\ninside the system: stored on the footing?", [x.name for x in FOOT.iterdir() if ("issue" in x.name or "q2d" in x.name or "hyde" in x.name) and "final" in x.name])
print("the model alone is not a rewriting; the first-law comparison for reference:", {s: round(summary(s)["law_at_1"], 3) for s in ("qwen3_8b_int8q", "query2doc", "hyde", ALONE)})

m = (p["band"] == "0").to_numpy()
print("\nfact patterns on the never-cited pairs, ranker plus patterns minus ranker:", ci(b.diff(H["lf_patterns"] * m, H[RANKER] * m, m), sign=True), "| HyDE alone minus encoder at >10:", ci(b.diff(H["hyde"] * (p["band"] == ">10").to_numpy(), H["qwen3_8b_int8q"] * (p["band"] == ">10").to_numpy(), (p["band"] == ">10").to_numpy()), sign=True))
print("seed variation, same inputs: lf_seed1 - ft_lean", ci(Boot(pc(RANKER)["component_id"].to_numpy()).group_mean(pc("lf_seed1").set_index("query_id").loc[pc(RANKER)["query_id"], "ndcg_at_10"].to_numpy() - pc(RANKER)["ndcg_at_10"].to_numpy()), sign=True))
