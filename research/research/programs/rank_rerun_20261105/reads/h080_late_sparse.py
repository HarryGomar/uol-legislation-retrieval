"""H080: BGE-M3's single vector, late interaction and sparse weights against tuned lexical matching and Qwen3-8B. Stored
runs of the 25 Oct record: bge_m3, bge_m3_colbert, bge_m3_sparse, qwen3_8b, lstar_headings, bm25_stem, S0+x, S_full_ext_minus_*."""
from common import *

M = {"BM25": "bm25_stem", "L*": LEX, "BGE dense": "bge_m3", "BGE late": "bge_m3_colbert", "BGE sparse": "bge_m3_sparse", "Qwen3-8B": ENC}
print("== alone")
for k, s in M.items():
    j = summary(s)
    print(f"{k:12s} nDCG {j['ndcg_at_10']:.3f} first {j['success_at_1']:.3f} R20 {j['recall_at_20']:.3f} R100 {j['recall_at_100']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f}")
p = pairs()
b = Boot(p["component_id"])
H = {k: pair_hit(s) for k, s in M.items()}
print("\n== Recall@20 by overlap quarter and band")
for col, bins in (("quarter", [1, 2, 3, 4]), ("band", BANDS)):
    for v in bins:
        m = (p[col] == v).to_numpy()
        print(f"{col} {v!s:5s} n={m.sum():4d} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H))
print("\n== Paired against BGE dense, recall@20, all pairs and on what L* misses")
rl = pair_rank(LEX)
miss = (rl == 0) | (rl > 100)
print("pairs L* leaves outside first 100:", int(miss.sum()))
for k in ("BGE late", "BGE sparse", "Qwen3-8B"):
    al = np.ones(len(p), bool)
    print(f"{k:12s} all {ci(b.diff(H[k], H['BGE dense'], al), sign=True)} | L*-miss: {k} {ci(b.ratio(H[k] * miss, miss))} vs dense {ci(b.ratio(H['BGE dense'] * miss, miss))}")
print("\n== Who finds the same pairs (of X's hits at 20, share also found by Y)")
names = list(H)
for a in names[2:]:
    print(f"{a:12s} hits {int(H[a].sum()):4d} " + " ".join(f"{c}:{(H[a] * H[c]).sum() / H[a].sum():.2f}" for c in names))
print("\n== First twenty lists, mean Jaccard")
T = {k: first(s, 20) for k, s in M.items()}
qs = list(T["L*"])
for a, c in (("BGE sparse", "BM25"), ("BGE sparse", "L*"), ("BGE late", "BGE dense"), ("BGE late", "L*"), ("BGE dense", "L*"), ("Qwen3-8B", "L*"), ("Qwen3-8B", "BGE dense"), ("BGE sparse", "BGE dense")):
    print(a, c, round(float(np.mean([len(set(T[a][q]) & set(T[c][q])) / len(set(T[a][q]) | set(T[c][q])) for q in qs])), 3))
print("\n== Inside a ranker (older ranker S0), paired against S0, and leave-one-out from the largest ranker")
S0 = pc("S0").set_index("query_id")
qq = S0.index
bq = Boot(S0["component_id"].to_numpy())
N0 = S0["ndcg_at_10"].to_numpy()
for s in ("S0+bge_m3_dense", "S0+bge_m3", "S0+bge_m3_colbert", "S0+bge_m3_sparse", "S0+qwen3_8b_dense", "S0+qwen3_8b"):
    v = pc(s).set_index("query_id").loc[qq, "ndcg_at_10"].to_numpy() - N0
    j = summary(s)
    print(f"{s:22s} {j['ndcg_at_10']:.3f} gain {ci(bq.group_mean(v), sign=True)} never {j['unseen_recall_at_20_pooled']:.3f}")
F = pc("S_full_ext").set_index("query_id")
for s in ("S_full_ext_minus_bge_m3", "S_full_ext_minus_colbert", "S_full_ext_minus_qwen3_8b", "S_full_ext_minus_e5"):
    v = pc(s).set_index("query_id").loc[F.index, "ndcg_at_10"].to_numpy() - F["ndcg_at_10"].to_numpy()
    print(f"{s:28s} {summary(s)['ndcg_at_10']:.3f} change {ci(Boot(F['component_id'].to_numpy()).group_mean(v), sign=True)} never {summary(s)['unseen_recall_at_20_pooled']:.3f} (full {summary('S_full_ext')['unseen_recall_at_20_pooled']:.3f})")
print("\n== with L*, fused two ways")
for s in ("rrf__lstar_headings__bge_m3", "rrf__lstar_headings__bge_m3_colbert", "rrf__lstar_headings__bge_m3_sparse", "rrf__lstar_headings__qwen3_8b",
          "linear__lstar_headings__bge_m3", "linear__lstar_headings__bge_m3_colbert", "linear__lstar_headings__bge_m3_sparse", "linear__lstar_headings__qwen3_8b"):
    print(s, round(summary(s)["ndcg_at_10"], 3))
print("\n== How the BGE vector meets a long article")
for s in ("bge_m3", "src_dense_bge_firstp", "src_dense_bge_long"):
    print(s, round(summary(s)["ndcg_at_10"], 3))
print("\n== By article length (words), recall@20")
d = release()
L = np.array([len(WORD.findall(d.candidate_texts[int(a)].lower())) for a in p["article"]])
q = pd.qcut(L, 4, labels=False)
for v in range(4):
    m = q == v
    print(f"length q{v + 1} n={m.sum()} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H))

print("\n== nDCG@10 paired, mean over groups")
D = pc("bge_m3").set_index("query_id")
bd_ = Boot(D["component_id"].to_numpy())
for s in ("bge_m3_colbert", "bge_m3_sparse", ENC, LEX, "bm25_stem"):
    v = pc(s).set_index("query_id").loc[D.index, "ndcg_at_10"].to_numpy() - D["ndcg_at_10"].to_numpy()
    print(f"{s:16s} - BGE dense {ci(bd_.group_mean(v), sign=True)}")
L_ = pc(LEX).set_index("query_id").loc[D.index, "ndcg_at_10"].to_numpy()
for s in ("bge_m3", "bge_m3_colbert", "bge_m3_sparse"):
    print(f"{s:16s} - L* {ci(bd_.group_mean(pc(s).set_index('query_id').loc[D.index, 'ndcg_at_10'].to_numpy() - L_), sign=True)}")
