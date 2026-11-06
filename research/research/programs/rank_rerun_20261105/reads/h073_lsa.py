"""H073: why LSA is weak alone. Stored runs lsa, lstar, lstar_headings, bm25_stem, qwen3_8b and the LSA receipt.
Tests candidate explanations; no LSA model is refitted. The document frequencies in part H are counts over the
catalogue's texts with the project's analyser, not a model."""
from common import *

LSA, LSTAR, BM = "lsa", "lstar", "bm25_stem"
print("== A. scores")
for s in ("random", "popularity", "lda_sim", "doc2vec", "word2vec", "lda_dm", LSA, BM, LSTAR, LEX, ENC):
    j = summary(s)
    print(f"{s:16s} nDCG {j['ndcg_at_10']:.3f} R20 {j['recall_at_20']:.3f} R100 {j['recall_at_100']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f}")

print("\n== B. size selection on inner folds (receipt)")
r = receipt(LSA)
sizes = {}
for fold in r["folds"]:
    for row in fold["fit"]["selection"]:
        sizes.setdefault(row["params"]["dim"], []).append(row["objective"])
    print("fold", fold["k"], "model", fold["fit"]["model"], "chosen", fold["fit"]["params"])
print({k: round(float(np.mean(v)), 4) for k, v in sizes.items()})

p = pairs().copy()
b = Boot(p["component_id"])
print("\n== C. Recall@20 per cited pair, by overlap quarter and citation band")
H = {s: pair_hit(s) for s in (LSA, BM, LSTAR, LEX, ENC)}
for col, bins in (("quarter", [1, 2, 3, 4]), ("band", BANDS)):
    for v in bins:
        m = (p[col] == v).to_numpy()
        row = {s: b.ratio(H[s] * m, m) for s in H}
        d = b.diff(H[LSA] * m, H[LEX] * m, m)
        print(f"{col} {v!s:5s} n={m.sum():4d} LSA {ci(row[LSA])}  bm25 {row[BM][0]:.3f} lstar {row[LSTAR][0]:.3f} L* {ci(row[LEX])} Qwen {row[ENC][0]:.3f} | LSA-L* {ci(d, sign=True)} LSA/L* {row[LSA][0] / row[LEX][0]:.2f}")

print("\n== D. does LSA find what lexical matching misses?")
for s in (LSA, ENC):
    h = H[s] > 0
    l = H[LEX] > 0
    print(s, "hits", int(h.sum()), "of which L* also", round((h & l).sum() / h.sum(), 3), "| L* hits", int(l.sum()), "of which it also", round((h & l).sum() / l.sum(), 3), "| beyond L* (pairs)", int((h & ~l).sum()), "| L* beyond it", int((l & ~h).sum()))
rl, rs = pair_rank(LEX), pair_rank(LSA)
miss = (rl == 0) | (rl > 100)
for s in (LSA, BM, "word2vec", "bge_m3", ENC):
    h = pair_hit(s) > 0
    print(f"  recall@20 on the {int(miss.sum())} pairs L* leaves outside its first 100: {s} {h[miss].mean():.3f}")

print("\n== E. overlap of first twenty with L*, cases (Jaccard), and first ten composition")
tops = {s: first(s, 20) for s in (LSA, BM, LEX, ENC)}
for s in (LSA, BM, ENC):
    print(s, "Jaccard with L*", round(np.mean([len(set(tops[s][q]) & set(tops[LEX][q])) / len(set(tops[s][q]) | set(tops[LEX][q])) for q in tops[LEX]]), 3))
law = article_law()
d = release()
wlen = np.array([len(WORD.findall(t.lower())) for t in d.candidate_texts])
pidx = pc(LEX).set_index("query_id")
for s in (LSA, BM, LEX, ENC, RANKER):
    t10 = first(s, 10)
    nlaws = [len({law[a] for a in v}) for v in t10.values()]
    toplaw = [pd.Series([law[a] for a in v]).value_counts().iloc[0] for v in t10.values()]
    print(f"{s:16s} distinct laws in first ten {np.mean(nlaws):.2f}  most-represented law's articles {np.mean(toplaw):.2f}  median words of first-ten articles {np.median([wlen[a] for v in t10.values() for a in v]):.0f}")
print("catalogue median words", np.median(wlen), "cited pair median", np.median(wlen[p["article"].to_numpy()]))

print("\n== F. law against article, within-law recall")
for s in (LSA, LEX, ENC):
    j = summary(s)
    print(s, "law1", round(j["law_at_1"], 3), "inlaw", round(j["success_at_1_in_law"], 3), "law10", round(j["law_at_10"], 3))
print("ratios LSA/L*: law1", round(summary(LSA)["law_at_1"] / summary(LEX)["law_at_1"], 2), "inlaw", round(summary(LSA)["success_at_1_in_law"] / summary(LEX)["success_at_1_in_law"], 2))

print("\n== G. by length")
p["alen"] = wlen[p["article"].to_numpy()]
p["lenq"] = pd.qcut(p["alen"], 4, labels=False) + 1
print("article length quarters edges", p["alen"].quantile([.25, .5, .75]).tolist())
for v in (1, 2, 3, 4):
    m = (p["lenq"] == v).to_numpy()
    print(f" article length q{v} n={m.sum()} LSA {H[LSA][m].mean():.3f} L* {H[LEX][m].mean():.3f} Qwen {H[ENC][m].mean():.3f} LSA/L* {H[LSA][m].mean() / H[LEX][m].mean():.2f}")
qt = query_text()
A, L_ = pc(LSA).set_index("query_id"), pc(LEX).set_index("query_id")
A = A.loc[L_.index]
fl = np.array([len(WORD.findall(qt[q].lower())) for q in L_.index])
print("facts words quartiles", np.quantile(fl, [.25, .5, .75]))
fq = pd.qcut(fl, 4, labels=False)
bc = Boot(L_["component_id"].to_numpy())
for v in range(4):
    m = fq == v
    print(f" facts length q{v + 1} n={m.sum()} LSA {A['ndcg_at_10'].to_numpy()[m].mean():.3f} L* {L_['ndcg_at_10'].to_numpy()[m].mean():.3f} gap {L_['ndcg_at_10'].to_numpy()[m].mean() - A['ndcg_at_10'].to_numpy()[m].mean():+.3f}")

print("\n== H. rare stems: does LSA lose the pairs whose match rests on a rare shared stem?")
from uol.facts_to_law.lexical import SpanishAnalyzer
an = SpanishAnalyzer()
arts_needed = set(p["article"].astype(int))
cat = [set(an(t)) for t in d.candidate_texts]
df = {}
for st in cat:
    for w in st:
        df[w] = df.get(w, 0) + 1
N = len(cat)
idf = {w: np.log((1 + N) / (1 + c)) + 1 for w, c in df.items()}
qs = {q: set(an(t)) for q, t in qt.items()}
mx, nshare, nrare = [], [], []
for q, a in zip(p["query_id"], p["article"]):
    sh = qs[q] & cat[int(a)]
    mx.append(max((idf.get(w, 0) for w in sh), default=0.0))
    nshare.append(len(sh))
    nrare.append(sum(1 for w in sh if df.get(w, 0) <= 100))
p["maxidf"], p["nshare"], p["nrare"] = mx, nshare, nrare
print("vocabulary of catalogue stems", len(df), "stems in 1 article", sum(c == 1 for c in df.values()), "shared stems per pair median", np.median(nshare), "pairs with no shared stem", int((p["nshare"] == 0).sum()))
print("pairs by count of shared rare stems (in at most 100 articles of 39,178)")
for lab, m in (("0", p["nrare"] == 0), ("1", p["nrare"] == 1), ("2-3", p["nrare"].between(2, 3)), ("4+", p["nrare"] >= 4)):
    m = m.to_numpy()
    d_ = b.diff(H[LSA] * m, H[LEX] * m, m)
    print(f" rare={lab:4s} n={m.sum():4d} LSA {H[LSA][m].mean():.3f} L* {H[LEX][m].mean():.3f} Qwen {H[ENC][m].mean():.3f} LSA-L* {ci(d_, sign=True)} LSA/L* {H[LSA][m].mean() / max(H[LEX][m].mean(), 1e-9):.2f}")

print("sensitivity of the rare threshold, and rare against length")
for thr in (50, 200):
    c = np.array([sum(1 for w in qs[q] & cat[int(a)] if df.get(w, 0) <= thr) for q, a in zip(p["query_id"], p["article"])])
    for lab, m in (("none", c == 0), ("some", c > 0)):
        print(f" df<={thr} {lab} n={m.sum()} LSA {H[LSA][m].mean():.3f} L* {H[LEX][m].mean():.3f} net L* pairs over LSA {(H[LEX][m].sum() - H[LSA][m].sum()):.0f}")
long_ = (p["alen"] > p["alen"].median()).to_numpy()
rare = (p["nrare"] > 0).to_numpy()
for lab, m in (("short, no rare", ~long_ & ~rare), ("short, rare", ~long_ & rare), ("long, no rare", long_ & ~rare), ("long, rare", long_ & rare)):
    print(f" {lab:15s} n={m.sum():4d} LSA {H[LSA][m].mean():.3f} L* {H[LEX][m].mean():.3f} bm25 {H[BM][m].mean():.3f} Qwen {H[ENC][m].mean():.3f} net L* pairs over LSA {(H[LEX][m].sum() - H[LSA][m].sum()):.0f}")

print("\n== I. the same family when it compares facts with past facts and learns from citations")
for s in ("src_knn_lsaadapt_spec", "src_proto_lsaadapt", "src_knn_lsacadapt_spec", "src_proto_lsacadapt", "src_knn_lex_votes", "src_proto_tfidf"):
    j = summary(s)
    print(s, round(j["ndcg_at_10"], 3), "R20", round(j["recall_at_20"], 3))
print("share of fit documents that are articles", 39178 / r["folds"][0]["fit"]["model"]["fit_docs"])
