"""H099: where the model's lead over the ranker comes from. GPT-6.1 Sol alone (gpt61sol_alone) against rt_lm_tesis (the new
ranker) and ft_lean (the current text's). Stored lists only. "Named" = its own names in the stored list (score >= 999,000,
as uol.final.model_limits). Popularity = how often the other folds' development cases cite the article, per cited pair
(the same fold-local counts as the citation band; the mask reproduces band 0). Writes out/h099.json."""
from common import *
from uol.embedding_push import common as ep
from uol.facts_to_law import prior as prior_mod

RK = {"rt_lm_tesis": NEWRANKER, "ft_lean": RANKER}
d = release()
A = pc(ALONE)
qs = A["query_id"].tolist()
A = A.set_index("query_id")
bc = Boot(A["component_id"].to_numpy())
NA = A["ndcg_at_10"].to_numpy()
law = article_law()
titles = law_title()
gold = {q: set(ints(v)) for q, v in zip(A.index, A["positive_idx"])}
p = pairs()
b = Boot(p["component_id"])
OUT = {}
print("== overall (mean over groups)")
for name, s in RK.items():
    R = pc(s).set_index("query_id").loc[A.index]
    lead = NA - R["ndcg_at_10"].to_numpy()
    print(f"{name}: ranker {summary(s)['ndcg_at_10']:.3f} model {summary(ALONE)['ndcg_at_10']:.3f} lead {ci(bc.group_mean(lead), sign=True)} | law1 {summary(s)['law_at_1']:.3f} vs {summary(ALONE)['law_at_1']:.3f} | in law {summary(s)['success_at_1_in_law']:.3f} vs {summary(ALONE)['success_at_1_in_law']:.3f} | first {summary(s)['success_at_1']:.3f} vs {summary(ALONE)['success_at_1']:.3f} | ten {summary(s)['success_at_10']:.3f} vs {summary(ALONE)['success_at_10']:.3f} | R100 {summary(s)['recall_at_100']:.3f} vs {summary(ALONE)['recall_at_100']:.3f}")
    print(f"   cases model better {np.mean(lead > 0):.3f} worse {np.mean(lead < 0):.3f} same {np.mean(lead == 0):.3f}")

print("\n== by law: each case's main cited law (most cited articles, ties to the more cited law overall), mean over cases")
cnt = pd.Series([law[a] for g in gold.values() for a in g]).value_counts()
main = {}
for q, g in gold.items():
    c = pd.Series([law[a] for a in g]).value_counts()
    top_ = c[c == c.max()].index.tolist()
    main[q] = max(top_, key=lambda l: cnt[l])
ml = pd.Series(main)
print("cited laws", len(cnt), "| top 1/3/5/10 laws' share of cited pairs", [round(cnt.iloc[:k].sum() / cnt.sum(), 3) for k in (1, 3, 5, 10)])
for name, s in RK.items():
    R = pc(s).set_index("query_id").loc[A.index]
    df = pd.DataFrame({"law": ml.loc[A.index].values, "m": NA, "r": R["ndcg_at_10"].to_numpy(), "ml": A["law_at_1"].to_numpy(), "rl": R["law_at_1"].to_numpy()})
    df["lead"] = df["m"] - df["r"]
    g = df.groupby("law").agg(n=("m", "size"), model=("m", "mean"), ranker=("r", "mean"), lead=("lead", "mean"), share=("lead", "sum"), mlaw=("ml", "mean"), rlaw=("rl", "mean")).sort_values("n", ascending=False)
    g["share"] /= df["lead"].sum()
    g["title"] = [titles[l][:48] for l in g.index]
    print(name, "total lead over cases", round(df["lead"].mean(), 3))
    pd.set_option("display.width", 220, "display.max_colwidth", 50)
    print(g.head(10).round(3).to_string(index=False))
    rest = g.iloc[10:]
    print(f"other {len(rest)} laws: n {rest['n'].sum()} lead {(rest['lead'] * rest['n']).sum() / rest['n'].sum():.3f} share {rest['share'].sum():.3f}")
    fr = pd.cut(g["n"], [0, 4, 19, 99, 10000], labels=["1-4", "5-19", "20-99", "100+"])
    print(g.groupby(fr, observed=True).apply(lambda x: pd.Series({"laws": len(x), "cases": x["n"].sum(), "model": (x["model"] * x["n"]).sum() / x["n"].sum(), "ranker": (x["ranker"] * x["n"]).sum() / x["n"].sum(), "lead": (x["lead"] * x["n"]).sum() / x["n"].sum(), "share of lead": x["share"].sum()})).round(3).to_string())

print("\n== law and article: four kinds of case by who has a cited law first")
for name, s in RK.items():
    R = pc(s).set_index("query_id").loc[A.index]
    ml_, rl_ = A["law_at_1"].to_numpy() > 0, R["law_at_1"].to_numpy() > 0
    lead = NA - R["ndcg_at_10"].to_numpy()
    print(name)
    for lab, m in (("both right law", ml_ & rl_), ("model only", ml_ & ~rl_), ("ranker only", ~ml_ & rl_), ("neither", ~ml_ & ~rl_)):
        print(f"   {lab:15s} cases {m.sum():4d} ({m.mean():.3f}) model {NA[m].mean():.3f} ranker {R['ndcg_at_10'].to_numpy()[m].mean():.3f} lead {lead[m].mean():+.3f} share of total lead {lead[m].sum() / lead.sum():.3f}")


def in_law_ndcg(run):
    """nDCG@10 with the list reduced to articles of the case's cited laws: the score if the law were given."""
    t = top(run)
    out = {}
    for q, g in t.sort_values(["query_id", "rank"]).groupby("query_id"):
        laws = {law[a] for a in gold[q]}
        v = [int(a) for a in g["candidate_idx"] if law[int(a)] in laws][:10]
        n = len(gold[q])
        z = sum(1 / np.log2(1 + r) for r in range(1, min(n, 10) + 1))
        out[q] = sum(1 / np.log2(2 + i) for i, a in enumerate(v) if a in gold[q]) / z
    return np.array([out.get(q, 0.0) for q in A.index])


print("\n== if the right law were given (lists reduced to the cited laws)")
OA = in_law_ndcg(ALONE)
print("model", round(bc.group_mean(OA)[0], 3))
for name, s in RK.items():
    O = in_law_ndcg(s)
    print(name, round(bc.group_mean(O)[0], 3), "lead with the law given", ci(bc.group_mean(OA - O), sign=True))
    OUT.setdefault("law_given", {})[name] = {"model": bc.group_mean(OA), "ranker": bc.group_mean(O), "lead": bc.group_mean(OA - O)}

print("\n== by band and word overlap: Recall@20 and @10 per cited pair, model minus ranker")
HA20, HA10 = pair_hit(ALONE), pair_hit(ALONE, 10)
tab = {}
for name, s in RK.items():
    H20, H10 = pair_hit(s), pair_hit(s, 10)
    print(name)
    tot = (HA20 - H20).sum()
    for col, bins in (("band", BANDS), ("quarter", [1, 2, 3, 4])):
        for v in bins:
            m = (p[col] == v).to_numpy()
            d20 = b.diff(HA20 * m, H20 * m, m)
            d10 = b.diff(HA10 * m, H10 * m, m)
            OUT.setdefault("band_r20", {}).setdefault(name, {})[f"{col}_{v}"] = {"n": int(m.sum()), "model": float(HA20[m].mean()), "ranker": float(H20[m].mean()), "diff": d20}
            print(f"   {col} {v!s:5s} n={m.sum():4d} R@20 model {HA20[m].mean():.3f} ranker {H20[m].mean():.3f} diff {ci(d20, sign=True)} share of net pairs {(HA20 - H20)[m].sum() / tot:.3f} | R@10 model {HA10[m].mean():.3f} ranker {H10[m].mean():.3f} diff {ci(d10, sign=True)}")

print("\n== popular articles of the right law")
roles = ep.roles()
foldof = {d.queries[int(r)].query_id: int(str(roles[r]).replace("dev_fold", "")) for r in ep.dev_rows()}
counts = {}
for k in range(5):
    ids = [d.queries[int(r)].query_id for r in ep.dev_rows() if foldof[d.queries[int(r)].query_id] != k]
    counts[k] = np.asarray(prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index))
chk = np.array([counts[foldof[q]][int(a)] == 0 for q, a in zip(p["query_id"], p["article"])])
assert (chk == (p["band"] == "0").to_numpy()).all()
# popularity rank of each article within its law, per fold (1 = most cited; never-cited articles get "never")
by_law = {}
for i, l in enumerate(law):
    by_law.setdefault(l, []).append(i)
poprank = {}
for k in range(5):
    pr = np.zeros(len(law), dtype=np.int32)
    for l, idx in by_law.items():
        idx = np.asarray(idx)
        order = idx[np.argsort(-counts[k][idx], kind="stable")]
        rank_ = 1
        for r, a in enumerate(order):
            pr[a] = (r + 1) if counts[k][a] > 0 else 0
    poprank[k] = pr


def strata(pr_):
    return "never cited" if pr_ == 0 else "1st" if pr_ == 1 else "2-5" if pr_ <= 5 else "6-20" if pr_ <= 20 else "21+"


S = ["1st", "2-5", "6-20", "21+", "never cited"]
tt = top(ALONE)
named = {q: g.sort_values("rank")["candidate_idx"].astype(int).tolist() for q, g in tt[tt["score"] >= 999000].groupby("query_id")}
print("named per case median", np.median([len(named.get(q, [])) for q in A.index]))
gs = pd.Series([strata(poprank[foldof[q]][a]) for q, a in zip(p["query_id"], p["article"])])
print("cited pairs by popularity place within their law:", (gs.value_counts(normalize=True).reindex(S)).round(3).to_dict(), "n", gs.value_counts().reindex(S).to_dict())
for name, s in RK.items():
    H = pair_hit(s)
    print(name, "Recall@20 by that place:", {v: (round(HA20[(gs == v).to_numpy()].mean(), 3), round(H[(gs == v).to_numpy()].mean(), 3)) for v in S}, "(model, ranker)")
rows = {}
for lab, L in (("model's named", named), ("rt_lm_tesis first 15", first(NEWRANKER, 15)), ("ft_lean first 15", first(RANKER, 15)), ("L* first 15", first(LEX, 15))):
    c = {v: 0 for v in S}
    hit = {v: 0 for v in S}
    for q, v_ in L.items():
        pr = poprank[foldof[q]]
        for a in v_:
            st = strata(pr[a])
            c[st] += 1
            hit[st] += a in gold[q]
    tot = sum(c.values())
    rows[lab] = {"share": {v: round(c[v] / tot, 3) for v in S}, "precision": {v: round(hit[v] / max(c[v], 1), 3) for v in S}}
    print(lab, rows[lab])
OUT["popularity_of_lists"] = rows
print("\n-- same law, popular articles: keep the model's laws and its number of articles per law, swap in the law's most cited articles (fold-local)")


def score_list(L):
    v = []
    for q in A.index:
        n = len(gold[q])
        z = sum(1 / np.log2(1 + r) for r in range(1, min(n, 10) + 1))
        v.append(sum(1 / np.log2(2 + i) for i, a in enumerate(L.get(q, [])[:10]) if a in gold[q]) / z)
    return np.array(v)


def swap(L, q):
    k = foldof[q]
    out, used = [], {}
    for a in L:
        l = law[a]
        used[l] = used.get(l, 0) + 1
    # preserve the order of the laws' first mention; round-robin position as the model placed them
    seen = {}
    for a in L:
        l = law[a]
        seen[l] = seen.get(l, 0)
        idx = np.asarray(by_law[l])
        order = idx[np.argsort(-counts[k][idx], kind="stable")]
        out.append(int(order[seen[l]]))
        seen[l] += 1
    return out


SW = {q: swap(named.get(q, []), q) for q in A.index}
base = score_list(named)
sw = score_list(SW)
print("model's stored list reduced to its names, nDCG", round(bc.group_mean(base)[0], 3), "(stored", round(summary(ALONE)["ndcg_at_10"], 3), ")")
print("same laws, same counts, the law's most cited articles:", ci(bc.group_mean(sw)), "difference", ci(bc.group_mean(base - sw), sign=True))
OUT["swap"] = {"model": bc.group_mean(base), "swap": bc.group_mean(sw), "diff": bc.group_mean(base - sw)}
# the same swap on the two rankers
for name, s in RK.items():
    L = first(s, 15)
    SWr = {q: swap(L[q], q) for q in A.index}
    print(f"{name}: first 15 nDCG {bc.group_mean(score_list(L))[0]:.3f}; same swap {bc.group_mean(score_list(SWr))[0]:.3f}")
# oracle law with popularity
OR = {}
for q in A.index:
    k = foldof[q]
    idx = np.asarray([i for l in {law[a] for a in gold[q]} for i in by_law[l]])
    OR[q] = [int(a) for a in idx[np.argsort(-counts[k][idx], kind="stable")][:10]]
print("right laws given, most cited articles of those laws:", ci(bc.group_mean(score_list(OR))))
print("catalogue popularity alone (stored popularity run):", round(summary("popularity")["ndcg_at_10"], 3))
print("share of articles named that were never cited in the other folds, by case: median", np.median([np.mean([poprank[foldof[q]][a] == 0 for a in named[q]]) for q in named]))
hits_named = [(strata(poprank[foldof[q]][a])) for q, v_ in named.items() for a in v_ if a in gold[q]]
print("the model's named hits by popularity place:", pd.Series(hits_named).value_counts(normalize=True).reindex(S).round(3).to_dict(), "n", len(hits_named))

print("\n== misses: first ten, by case and by pair")
for name, s in RK.items():
    H10 = pair_hit(s, 10)
    R = pc(s).set_index("query_id").loc[A.index]
    mc, rc = A["success_at_10"].to_numpy() > 0, R["success_at_10"].to_numpy() > 0
    both, mo, ro, ne = (mc & rc).mean(), (mc & ~rc).mean(), (~mc & rc).mean(), (~mc & ~rc).mean()
    exp_ne = (~mc).mean() * (~rc).mean()
    phi = np.corrcoef(mc, rc)[0, 1]
    print(f"{name} cases: both {both:.3f} model only {mo:.3f} ranker only {ro:.3f} neither {ne:.3f} (expected if independent {exp_ne:.3f}); of the model's misses the ranker also misses {ne / (~mc).mean():.3f}; of the ranker's misses the model also misses {ne / (~rc).mean():.3f}; phi {phi:.3f}")
    ma, ra = HA10 > 0, H10 > 0
    both, mo, ro, ne = (ma & ra).mean(), (ma & ~ra).mean(), (~ma & ra).mean(), (~ma & ~ra).mean()
    print(f"   pairs @10: both {both:.3f} model only {mo:.3f} ranker only {ro:.3f} neither {ne:.3f} (expected {(~ma).mean() * (~ra).mean():.3f}); of the model's misses ranker also misses {ne / (~ma).mean():.3f}; of the ranker's misses model also misses {ne / (~ra).mean():.3f}; phi {np.corrcoef(ma, ra)[0, 1]:.3f}")
    from scipy.stats import spearmanr
    print(f"   Spearman of per-case nDCG@10 model-ranker {spearmanr(NA, R['ndcg_at_10']).statistic:.3f}")
    neither = (~ma & ~ra)
    print("   pairs both miss, by band", {v: round(float(neither[(p['band'] == v).to_numpy()].mean()), 3) for v in BANDS}, "by quarter", {v: round(float(neither[(p['quarter'] == v).to_numpy()].mean()), 3) for v in (1, 2, 3, 4)})
R1, R2 = pc(RANKER).set_index("query_id").loc[A.index], pc(NEWRANKER).set_index("query_id").loc[A.index]
from scipy.stats import spearmanr
print("for scale, two rankers against each other: Spearman", round(spearmanr(R1["ndcg_at_10"], R2["ndcg_at_10"]).statistic, 3), "; the ranker against the encoder alone", round(spearmanr(R1["ndcg_at_10"], pc(ENC).set_index("query_id").loc[A.index, "ndcg_at_10"]).statistic, 3))

print("\n== other models, same facts (sample stored elsewhere; reranking on all cases): law and within-law")
for s in ("gpt61sol_rerank_ft_lean", "gpt61terra_rerank_ft_lean", "gpt6solnw_rerank_ft_lean", "gpt56sol_rerank_ft_lean", "gpt6luna_rerank_ft_lean", "luna_rerank_ft_lean"):
    j = summary(s)
    print(s, round(j["ndcg_at_10"], 3), "law1", round(j["law_at_1"], 3), "inlaw", round(j["success_at_1_in_law"], 3))
print("\n== blank control on the older list: candidates as law and article number only, no text (GPT-5.6 Luna)")
for s in ("S_full_ext", "llm_rerank_S_full_ext_blank", "llm_rerank_S_full_ext"):
    j = summary(s)
    print(s, round(j["ndcg_at_10"], 3), "law1", round(j["law_at_1"], 3), "inlaw", round(j["success_at_1_in_law"], 3))
print("\n== cases by number of cited articles, lead")
nr = A["n_relevant"].to_numpy()
for name, s in RK.items():
    R = pc(s).set_index("query_id").loc[A.index]
    lead = NA - R["ndcg_at_10"].to_numpy()
    print(name, {lab: (int(m.sum()), round(lead[m].mean(), 3)) for lab, m in (("1", nr == 1), ("2", nr == 2), ("3+", nr >= 3))})
save("h099", OUT)

print("\n== Few shared words and no past citation together")
m = (p["quarter"] == 1).to_numpy() & (p["band"] == "0").to_numpy()
print("pairs", int(m.sum()), {k: round(float(pair_hit(s)[m].mean()), 3) for k, s in (("L*", LEX), ("encoder", ENC), ("rt_lm_tesis", NEWRANKER), ("ft_lean", RANKER), ("model", ALONE), ("system", SYSTEM))})
