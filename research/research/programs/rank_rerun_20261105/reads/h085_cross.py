"""H085: cross-encoders on the ranker's first 30 (ft_lean): *_ranker_lean = fused with the list by a cross-fitted weight;
*_ranker_lean_alone = the first 30 reordered by the reranker score alone. Weights from the receipts, settings from
src/uol/final/rerank_jobs.py and round3/r2_cross.py. Against GPT-6.1 Sol reranking the same list."""
from common import *

BASE = RANKER
ARMS = {"BGE reranker": "bge_ranker_lean", "Qwen3-Reranker-4B": "qwen4b_ranker_lean", "trained (Qwen3-Reranker-0.6B)": "ce_ranker_lean"}
B = pc(BASE).set_index("query_id")
qs = B.index
bc = Boot(B["component_id"].to_numpy())
top10 = first(BASE, 10)
top1 = {q: v[0] for q, v in top10.items()}
N0 = B["ndcg_at_10"].to_numpy()
print("list", round(summary(BASE)["ndcg_at_10"], 3), "law1", round(summary(BASE)["law_at_1"], 3), "inlaw", round(summary(BASE)["success_at_1_in_law"], 3), "R20", round(summary(BASE)["recall_at_20"], 3))


def describe(lab, s):
    A = pc(s).set_index("query_id").loc[qs]
    v = A["ndcg_at_10"].to_numpy() - N0
    t = first(s, 10)
    changed_set = np.mean([set(t[q]) != set(top10[q]) for q in qs])
    swaps = np.mean([len(set(top10[q]) - set(t[q])) for q in qs])
    order = np.mean([t[q] != top10[q] for q in qs])
    f1 = np.mean([t[q][0] != top1[q] for q in qs])
    j = summary(s)
    print(f"{lab:36s} {s:26s} nDCG {j['ndcg_at_10']:.3f} {ci(bc.group_mean(v), sign=True)} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f} R20 {j['recall_at_20']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} | order of ten differs {order:.3f} set differs {changed_set:.3f} articles swapped {swaps:.2f} first changed {f1:.3f} | better {np.mean(v > 0):.3f} worse {np.mean(v < 0):.3f} same {np.mean(v == 0):.3f}")


print("\n== fused with the list, then alone, then the model")
for lab, s in ARMS.items():
    r = receipt(s)["inputs"]["llm_meta"]
    print(f"{lab}: weight per fold {r['w_by_fold']}")
    describe(lab + " fused", s)
    describe(lab + " alone", s + "_alone")
r = receipt("gpt61sol_rerank_ft_lean")["inputs"]["llm_meta"]
print("GPT-6.1 Sol weight per fold", r["w_by_fold"])
describe("GPT-6.1 Sol fused (w = 3)", "gpt61sol_rerank_ft_lean")

print("\n== first 30 of the list holds", end=" ")
p = pairs()
rr = pair_rank(BASE)
print(f"{((rr > 0) & (rr <= 30)).mean():.3f} of the cited pairs; first 10 {((rr > 0) & (rr <= 10)).mean():.3f}")
print("\n== Recall@10 per pair by band and overlap, list against reranker alone, reranker fused and the model")
b = Boot(p["component_id"])
ids = {"list": BASE, "bge": "bge_ranker_lean_alone", "qwen4b": "qwen4b_ranker_lean_alone", "trained": "ce_ranker_lean_alone", "bge_f": "bge_ranker_lean", "qwen4b_f": "qwen4b_ranker_lean", "trained_f": "ce_ranker_lean", "model": "gpt61sol_rerank_ft_lean"}
H = {k: pair_hit(s, 10) for k, s in ids.items()}
for col, bins in (("quarter", [1, 2, 3, 4]), ("band", BANDS)):
    for v in bins:
        m = (p[col] == v).to_numpy() & (rr > 0) & (rr <= 30)
        print(f"{col} {v!s:5s} within-30 n={m.sum():4d} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H) + f" | qwen4b alone-list {ci(b.diff(H['qwen4b'] * m, H['list'] * m, m), sign=True)} | trained alone-list {ci(b.diff(H['trained'] * m, H['list'] * m, m), sign=True)} | model-list {ci(b.diff(H['model'] * m, H['list'] * m, m), sign=True)}")
print("\n== Agreement with the list and with the model: Spearman of positions inside the first 30, per case (median)")
t30 = {s: first(s, 30) for s in (BASE, "bge_ranker_lean_alone", "qwen4b_ranker_lean_alone", "ce_ranker_lean_alone", "gpt61sol_rerank_ft_lean")}
from scipy.stats import spearmanr
for s in list(t30)[1:]:
    rs = []
    for q in qs:
        a, c = t30[BASE][q], t30[s][q]
        pos = {x: i for i, x in enumerate(c)}
        common = [x for x in a if x in pos]
        if len(common) > 5:
            rs.append(spearmanr(range(len(common)), [pos[x] for x in common]).statistic)
    print(s, round(float(np.nanmedian(rs)), 3))
print("\n== Same-law ordering: reranker alone, how often the first article stays in the list's first law")
law = article_law()
for s in ("bge_ranker_lean_alone", "qwen4b_ranker_lean_alone", "ce_ranker_lean_alone", "gpt61sol_rerank_ft_lean"):
    t = first(s, 10)
    print(s, "first article in the list's first law", round(float(np.mean([law[t[q][0]] == law[top1[q]] for q in qs])), 3), "distinct laws in first ten", round(float(np.mean([len({law[a] for a in t[q]}) for q in qs])), 2))
print("list distinct laws", round(float(np.mean([len({law[a] for a in top10[q]}) for q in qs])), 2))

print("\n== What they read: words (proxy, no tokenizer loaded). Query limit 384 tokens; zero-shot documents 4 chunks of 1,024 tokens; trained 4 chunks of 256")
qt = query_text()
d = release()
w = np.array([len(re.findall(r"\S+", qt[q])) for q in qs])
print("facts words: median", np.median(w), "p90", np.quantile(w, .9), "p99", np.quantile(w, .99), "max", w.max(), "| over 250 words", round(float((w > 250).mean()), 3), "over 300", round(float((w > 300).mean()), 3))
aw = np.array([len(re.findall(r"\S+", t)) for t in d.candidate_texts])
cit = aw[p["article"].to_numpy()]
print("all articles words: median", np.median(aw), "p99", np.quantile(aw, .99), "max", aw.max())
print("cited pair articles words: median", np.median(cit), "p90", np.quantile(cit, .9), "max", cit.max())
t30_all = np.array([a for v in t30[BASE].values() for a in v])
print("first-30 articles words: median", np.median(aw[t30_all]), "p90", np.quantile(aw[t30_all], .9))
for lim, lab in ((2600, "over about 4x1,024 tokens (word proxy 2,600: 1.55 tokens a word)"), (660, "over about 4x256 tokens (word proxy 660)"), (165, "over one 256-token chunk (word proxy 165)")):
    print(f" {lab}: first-30 articles {np.mean(aw[t30_all] > lim):.3f}; cited pairs {np.mean(cit > lim):.3f}")
ml = aw[p["article"].to_numpy()]
for name, H_ in (("qwen4b alone", H["qwen4b"]), ("trained alone", H["trained"]), ("list", H["list"])):
    m = (rr > 0) & (rr <= 30)
    q_ = pd.qcut(ml[m], 3, labels=False)
    print(name, "Recall@10 of pairs within 30 by article length third", [round(float(H_[m][q_ == i].mean()), 3) for i in range(3)])
print("\n== Facts-length dependence of the alone reranker's change")
for s in ("qwen4b_ranker_lean_alone", "ce_ranker_lean_alone"):
    v = pc(s).set_index("query_id").loc[qs, "ndcg_at_10"].to_numpy() - N0
    fq = pd.qcut(w, 4, labels=False)
    print(s, [round(float(v[fq == i].mean()), 3) for i in range(4)])
print("\n== Older record on the larger list, same picture")
for s in ("S_full_ext", "rerank_bge", "rerank_bge_alone", "rerank_qwen3_4b", "rerank_qwen3_4b_alone", "ce_trained", "ce_trained_alone"):
    print(s, round(summary(s)["ndcg_at_10"], 3))
