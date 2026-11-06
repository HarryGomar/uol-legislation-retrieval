"""H077: the four-way design and whether the encoder is pulled down. Stored runs ft_neither_strict, ft_lean_nocit,
ft_lean_noenc, ft_lean (the eight-source ranker's cells; the rk_* cells on rt_lm_tesis are not stored yet), qwen3_8b
alone, lstar_headings, and rt_lm_tesis as far as it reads."""
from common import *

CELL = {"Neither": "ft_neither_strict", "EncOnly": "ft_lean_nocit", "CitOnly": NOENC, "Both": RANKER}
res = json.loads(RESULTS.read_text(encoding="utf-8"))
for k, s in CELL.items():
    j = summary(s)
    assert abs(j["ndcg_at_10"] - res["experiments"]["cells"][k]["ndcg_at_10"]) < 1e-9, (k, s)
    print(f"{k:8s} {s:20s} nDCG {j['ndcg_at_10']:.3f} first {j['success_at_1']:.3f} ten {j['success_at_10']:.3f} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f} R20 {j['recall_at_20']:.3f} R100 {j['recall_at_100']:.3f} never {j['unseen_recall_at_20_pooled']:.3f}")

base = pc(RANKER)
qs = base["query_id"]
N = {k: pc(s).set_index("query_id").loc[qs, "ndcg_at_10"].to_numpy() for k, s in CELL.items()}
bc = Boot(base["component_id"].to_numpy())
print("\n== paired, mean over groups")
for name, v in (("encoder without citations (EncOnly-Neither)", N["EncOnly"] - N["Neither"]), ("encoder with citations (Both-CitOnly)", N["Both"] - N["CitOnly"]),
                ("citations without encoder (CitOnly-Neither)", N["CitOnly"] - N["Neither"]), ("citations with encoder (Both-EncOnly)", N["Both"] - N["EncOnly"]),
                ("interaction (Both-CitOnly)-(EncOnly-Neither)", N["Both"] - N["CitOnly"] - N["EncOnly"] + N["Neither"])):
    print(f"{name:50s} {ci(bc.group_mean(v), sign=True)}")

p = pairs()
b = Boot(p["component_id"])
H = {k: pair_hit(s) for k, s in CELL.items()}
H["Qwen alone"] = pair_hit(ENC)
H["L*"] = pair_hit(LEX)
H["rt_lm_tesis"] = pair_hit(NEWRANKER)
print("\n== Recall@20 per pair by band")
for bd in BANDS:
    m = (p["band"] == bd).to_numpy()
    print(f"band {bd:5s} n={m.sum():4d} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H) + f" | enc gain with cit {ci(b.diff(H['Both'] * m, H['CitOnly'] * m, m), sign=True)} | Both-EncOnly {ci(b.diff(H['Both'] * m, H['EncOnly'] * m, m), sign=True)} | Both-Qwen {ci(b.diff(H['Both'] * m, H['Qwen alone'] * m, m), sign=True)}")
print("\n== Recall@20 per pair by overlap quarter")
for q in (1, 2, 3, 4):
    m = (p["quarter"] == q).to_numpy()
    print(f"Q{q} n={m.sum():4d} " + " ".join(f"{k} {H[k][m].mean():.3f}" for k in H) + f" | enc gain with cit {ci(b.diff(H['Both'] * m, H['CitOnly'] * m, m), sign=True)}")

print("\n== Does the ranker keep what the encoder finds? P(Both hit at 20 | Qwen alone hit at 20), and displacement")
rq, rb, rc = pair_rank(ENC), pair_rank(RANKER), pair_rank(NOENC)
for bd in BANDS:
    m = (p["band"] == bd).to_numpy() & (H["Qwen alone"] > 0)
    n = (p["band"] == bd).to_numpy() & (H["Qwen alone"] == 0)
    print(f"band {bd:5s} Qwen hits {m.sum():4d} kept by ranker {H['Both'][m].mean():.3f} by CitOnly {H['CitOnly'][m].mean():.3f} median rank Qwen {np.median(rq[m]):.0f} -> ranker {np.median(rb[m & (rb > 0)]):.0f} | Qwen misses {n.sum()} ranker finds {H['Both'][n].mean():.3f}")
print("rt_lm_tesis keeps of band-0 Qwen hits:", H["rt_lm_tesis"][(p["band"] == "0").to_numpy() & (H["Qwen alone"] > 0)].mean())

print("\n== Never-cited articles in the lists: fold-local mask (no development case of another fold cites it)")
from uol.embedding_push import common as ep
d = release()
roles = ep.roles()
foldof = {d.queries[int(r)].query_id: int(str(roles[r]).replace("dev_fold", "")) for r in ep.dev_rows()}
from uol.facts_to_law import prior as prior_mod
masks = {}
for k in range(5):
    ids = [d.queries[int(r)].query_id for r in ep.dev_rows() if foldof[d.queries[int(r)].query_id] != k]
    masks[k] = np.asarray(prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)) == 0
print("never-cited share of the catalogue per fold", [round(float(masks[k].mean()), 3) for k in range(5)])
chk = np.array([masks[foldof[q]][int(a)] for q, a in zip(p["query_id"], p["article"])])
print("mask reproduces band 0:", int((chk == (p["band"] == "0").to_numpy()).sum()), "of", len(p), "band0", int((p["band"] == "0").sum()), "mask", int(chk.sum()))
gold = {q: set(ints(v)) for q, v in zip(base["query_id"], base["positive_idx"])}
print("share of cited pairs that are never-cited", round(float((p["band"] == "0").mean()), 3))
for k_ in (10, 20):
    print(f"-- first {k_}: shown never-cited articles per case, their precision; precision of the cited-before articles")
    for s in (LEX, ENC, "ft_neither_strict", "ft_lean_nocit", NOENC, RANKER, NEWRANKER, ALONE, SYSTEM):
        L = first(s, k_)
        sn = hn = so = ho = 0
        for q, v in L.items():
            mk = masks[foldof[q]]
            for a in v:
                if mk[a]:
                    sn += 1; hn += a in gold[q]
                else:
                    so += 1; ho += a in gold[q]
        print(f"  {s:32s} never-cited shown/case {sn / len(L):.2f} share of list {sn / (sn + so):.3f} precision {hn / max(sn, 1):.4f} | seen-before precision {ho / max(so, 1):.4f} | share of its hits that are never-cited {hn / max(hn + ho, 1):.3f}")

print("\n== Where the encoder's nDCG gain falls (Both-CitOnly), by case kind, pooled over cases")
un = base["unseen_n"].to_numpy() > 0
gain = N["Both"] - N["CitOnly"]
print("cases with a never-cited cited article", int(un.sum()), "mean gain", round(gain[un].mean(), 3), "| others", int((~un).sum()), "mean gain", round(gain[~un].mean(), 3), "| share of total gain from them", round(gain[un].sum() / gain.sum(), 3), "their CI", ci(bc.group_mean(gain, un), sign=True), ci(bc.group_mean(gain, ~un), sign=True))
print("cases improved", round(float((gain > 0).mean()), 3), "worsened", round(float((gain < 0).mean()), 3), "unchanged", round(float((gain == 0).mean()), 3))

print("\n== Near misses: non-cited articles in the first ten (same cited law; next to a cited article in law order)")
law = article_law()
order = {}
for i, c in enumerate(d.candidate_ids):
    m = re.search(r"articulo=(\d+)", c)
    order.setdefault(law[i], []).append((int(m.group(1)) if m else 10**6, i))
pos = {}
for l, v in order.items():
    for j, (_, i) in enumerate(sorted(v)):
        pos[i] = (l, j)
for s in (LEX, ENC, "ft_lean_nocit", NOENC, RANKER, NEWRANKER, ALONE):
    L = first(s, 10)
    near = same = other = hit = 0
    for q, v in L.items():
        g = gold[q]
        laws = {law[a] for a in g}
        adj = {(pos[a][0], pos[a][1] + e) for a in g for e in (-1, 1)}
        for a in v:
            if a in g:
                hit += 1
            elif pos[a] in adj:
                near += 1
            elif law[a] in laws:
                same += 1
            else:
                other += 1
    t = near + same + other
    print(f"  {s:20s} of the non-cited in first ten: next to cited {near / t:.3f}, other in a cited law {same / t:.3f}, other law {other / t:.3f} | hits {hit}")

print("\n== Which route: the direct score or the key for past cases (older ranker S0, stored)")
for s in ("S0", "S0+qwen3_8b_dense", "S0+qwen3_8b", "src_knn_lex_votes", "src_knn_qwen8b", "src_proto_tfidf", "src_proto_qwen8b", "src_K_qwen8b", "src_H_qwen8b", "src_KH_qwen8b"):
    j = summary(s)
    print(f"  {s:22s} nDCG {j['ndcg_at_10']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} R20 {j['recall_at_20']:.3f}")
S0 = pc("S0").set_index("query_id")
bq = Boot(S0.loc[qs, "component_id"].to_numpy())
for s in ("S0+qwen3_8b_dense", "S0+qwen3_8b"):
    print("  ", s, "- S0", ci(bq.group_mean(pc(s).set_index("query_id").loc[qs, "ndcg_at_10"].to_numpy() - S0.loc[qs, "ndcg_at_10"].to_numpy()), sign=True))

print("\n== Inside the system")
for s in (SYSTEM, "gpt61sol_final_noenc", ALONE):
    j = summary(s)
    print(f"  {s:32s} nDCG {j['ndcg_at_10']:.3f} never {j['unseen_recall_at_20_pooled']:.3f} R20 {j['recall_at_20']:.3f} R100 {j['recall_at_100']:.3f}")
print("rt_lm_tesis", {k: round(summary(NEWRANKER)[k], 3) for k in ("ndcg_at_10", "unseen_recall_at_20_pooled", "recall_at_20", "law_at_1", "success_at_1_in_law")})
print("rk_ cells stored:", sorted(x.name for x in FOOT.iterdir() if x.name.startswith("rk_") and (x / "DEV/summary.json").exists()))

print("\n== The same four-way on the new ranker (rk_neither, rk_nocit, rk_noenc, rt_lm_tesis), where stored")
CELLK = {"Neither": "rk_neither", "EncOnly": "rk_nocit", "CitOnly": "rk_noenc", "Both": NEWRANKER}
try:
    for k, s in CELLK.items():
        j = summary(s)
        print(f"{k:8s} {s:14s} partial={j.get('partial')} n={j.get('n_cases')} nDCG {j['ndcg_at_10']:.3f} first {j['success_at_1']:.3f} ten {j['success_at_10']:.3f} law1 {j['law_at_1']:.3f} inlaw {j['success_at_1_in_law']:.3f} R20 {j['recall_at_20']:.3f} never {j['unseen_recall_at_20_pooled']:.3f}")
    Nk = {k: pc(s).set_index("query_id").loc[qs, "ndcg_at_10"].to_numpy() for k, s in CELLK.items()}
    for name, v in (("encoder without citations", Nk["EncOnly"] - Nk["Neither"]), ("encoder with citations", Nk["Both"] - Nk["CitOnly"]),
                    ("citations without encoder", Nk["CitOnly"] - Nk["Neither"]), ("citations with encoder", Nk["Both"] - Nk["EncOnly"]),
                    ("interaction", Nk["Both"] - Nk["CitOnly"] - Nk["EncOnly"] + Nk["Neither"])):
        print(f"{name:32s} {ci(bc.group_mean(v), sign=True)}")
    Hk = {k: pair_hit(s) for k, s in CELLK.items()}
    for bd in BANDS:
        m = (p["band"] == bd).to_numpy()
        print(f"band {bd:5s} " + " ".join(f"{k} {Hk[k][m].mean():.3f}" for k in Hk) + f" | enc gain with cit {ci(b.diff(Hk['Both'] * m, Hk['CitOnly'] * m, m), sign=True)} | Both-EncOnly {ci(b.diff(Hk['Both'] * m, Hk['EncOnly'] * m, m), sign=True)} | Both-Qwen {ci(b.diff(Hk['Both'] * m, H['Qwen alone'] * m, m), sign=True)}")
    for q_ in (1, 2, 3, 4):
        m = (p["quarter"] == q_).to_numpy()
        print(f"Q{q_} enc gain with cit {ci(b.diff(Hk['Both'] * m, Hk['CitOnly'] * m, m), sign=True)}")
    gain = Nk["Both"] - Nk["CitOnly"]
    print("cases with a never-cited article: gain", ci(bc.group_mean(gain, un), sign=True), "others", ci(bc.group_mean(gain, ~un), sign=True), "share of case-pooled gain from them", round(gain[un].sum() / gain.sum(), 3))
    for bd in BANDS:
        m = (p["band"] == bd).to_numpy() & (H["Qwen alone"] > 0)
        print(f"band {bd:5s} Qwen hits {m.sum():4d} kept by rt_lm_tesis {Hk['Both'][m].mean():.3f}")
except FileNotFoundError as e:
    print("not stored:", e)
