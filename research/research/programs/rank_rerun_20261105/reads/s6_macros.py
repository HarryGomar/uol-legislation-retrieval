"""The values Chapter 6 asks of the ranker session (H073, H077, H081, E7a, E7b), on the chosen ranker wherever a ranker is
meant. Development cases, stored runs, no model call, nothing refitted, no test file opened. Written to out/s6_macros.json;
scripts/final_results.py prints them. Pair reads use the cited pairs of depends_pairs.parquet, a cluster bootstrap over groups."""
from common import *

NOV = "rk_noenc_novotes"
NEW, NOENC_ = NEWRANKER, "rk_noenc"
OVER = ROOT / "research/programs/overnight_20261105"
p = pairs().copy()
b = Boot(p["component_id"])
never = (p["band"] == "0").to_numpy()
most = (p["band"] == ">10").to_numpy()
high = (p["quarter"] == 4).to_numpy()
out: dict = {"draws": int(os.environ.get("UOL_READ_DRAWS", 2000))}


def contrast(a: str, c: str, m: np.ndarray) -> dict:
    d = b.diff(pair_hit(a) * m, pair_hit(c) * m, m)
    return {"delta": d[0], "ci95": [d[1], d[2]], "pairs": int(m.sum())}


# H073: LSA against tuned lexical matching and the encoder (ranker-free)
H = {s: pair_hit(s).astype(bool) for s in ("lsa", LEX, ENC)}
out["lsaAlsoLex"] = float((H["lsa"] & H[LEX]).sum() / H["lsa"].sum())
out["qwenAlsoLex"] = float((H[ENC] & H[LEX]).sum() / H[ENC].sum())
rl = pair_rank(LEX)
miss = (rl == 0) | (rl > 100)
out["lsaRMiss"] = float(H["lsa"][miss].mean())
out["lsaRMissPairs"] = int(miss.sum())
dims = {int(f["fit"]["params"]["dim"]) for f in receipt("lsa")["folds"]}
grid = {int(row["params"]["dim"]) for f in receipt("lsa")["folds"] for row in f["fit"]["selection"]}
out["lsaDimChosen"], out["lsaDimGrid"] = sorted(dims), sorted(grid)

# the rare stems: a stem found in at most 100 articles of the catalogue, shared by the facts and the article
from uol.facts_to_law.lexical import SpanishAnalyzer
d = release()
qt = query_text()
an = SpanishAnalyzer()
cat = [set(an(t)) for t in d.candidate_texts]
df: dict = {}
for st in cat:
    for w in st:
        df[w] = df.get(w, 0) + 1
qs = {q: set(an(t)) for q, t in qt.items()}
nrare = np.array([sum(1 for w in qs[q] & cat[int(a)] if df.get(w, 0) <= 100) for q, a in zip(p["query_id"], p["article"])])
out["rare"] = {lab: {"pairs": int(m.sum()), "lsa": float(H["lsa"][m].mean()), "lex": float(H[LEX][m].mean())}
               for lab, m in (("none", nrare == 0), ("one", nrare == 1))}

# H077: what is shown among the first ten, by the fold's own never-cited mask
from uol.embedding_push import common as ep
from uol.facts_to_law import prior as prior_mod
roles = ep.roles()
foldof = {d.queries[int(r)].query_id: int(str(roles[r]).replace("dev_fold", "")) for r in ep.dev_rows()}
masks = {}
for k in range(5):
    ids = [d.queries[int(r)].query_id for r in ep.dev_rows() if foldof[d.queries[int(r)].query_id] != k]
    masks[k] = np.asarray(prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)) == 0
chk = np.array([masks[foldof[q]][int(a)] for q, a in zip(p["query_id"], p["article"])])
if not np.array_equal(chk, never):
    raise SystemExit("the fold masks do not reproduce the never-cited band")
gold: dict = {}
for q, a in zip(p["query_id"], p["article"]):
    gold.setdefault(q, set()).add(int(a))


def shown(sid: str, k: int = 10) -> dict:
    sn = hn = so = ho = 0
    for q, v in first(sid, k).items():
        mk = masks[foldof[q]]
        for a in v:
            if mk[a]:
                sn += 1
                hn += a in gold.get(q, ())
            else:
                so += 1
                ho += a in gold.get(q, ())
    return {"never_shown": sn, "never_right": hn / max(sn, 1), "cited_shown": so, "cited_right": ho / max(so, 1)}


sh = {s: shown(s) for s in (ENC, NEW)}
out["precNevEnc"], out["precNevRank"], out["precCitedRank"] = sh[ENC]["never_right"], sh[NEW]["never_right"], sh[NEW]["cited_right"]
hits_enc = pair_hit(ENC).astype(bool)
out["keepNevRank"] = float(pair_hit(NEW)[never & hits_enc].mean())
out["keepNevPairs"] = int((never & hits_enc).sum())
out["cRankEncNever"] = contrast(NEW, ENC, never)
out["cEncNever"] = contrast(NEW, NOENC_, never)
out["cEncMost"] = contrast(NEW, NOENC_, most)
j = summary(NOV) if (FOOT / NOV / "DEV" / "summary.json").exists() else None
out["noVotes"] = j and {"never": j["unseen_recall_at_20_pooled"], "ndcg": j["ndcg_at_10"]}
out["withVotes"] = {"never": summary(NOENC_)["unseen_recall_at_20_pooled"], "ndcg": summary(NOENC_)["ndcg_at_10"]}

# H081: rewriting
base = pc("qwen3_8b_int8q").set_index("query_id")
bc = Boot(base["component_id"].to_numpy())
qid = base.index.to_numpy()
out["sxQbigIntQ"] = summary("qwen3_8b_int8q")["ndcg_at_10"]
out["lwaQbigIntQ"] = summary("qwen3_8b_int8q")["law_at_1"]
for name, sid in (("cQtwoDAlone", "query2doc"), ("cHydeAlone", "hyde")):
    v = pc(sid).set_index("query_id").loc[qid, "ndcg_at_10"].to_numpy() - base["ndcg_at_10"].to_numpy()
    g = bc.group_mean(v)
    out[name] = {"delta": g[0], "ci95": [g[1], g[2]]}
out["dAddQtwodNdcg"] = summary("rk_q2d")["ndcg_at_10"]
out["dAddHydeNdcg"] = summary("rk_hyde")["ndcg_at_10"]
out["cIssuesNever"] = contrast("rk_issues", NEW, never)
out["cIssuesHigh"] = contrast("rk_issues", NEW, high)

# E7a and E7b, stored reads
e7a = json.loads((OVER / "e7a_what_differs.json").read_text(encoding="utf-8"))
ad = next(v for k, v in e7a.items() if k.startswith("adapter vs"))["by"]["citation band"]["cells"]
out["adNever"], out["adMost"] = ad["0"]["gain"], ad[">10"]["gain"]
e7b = json.loads((OVER / "e7b_read.json").read_text(encoding="utf-8"))
out["sxLateTrained"], out["sxLateUntrained"] = e7b["alone"]["trained"]["ndcg_at_10"], e7b["alone"]["untrained"]["ndcg_at_10"]
con = e7b["contrasts"]
out["cLateEnc"] = con["trained - qwen3_8b"]
out["cLateTrain"] = next((v for k, v in con.items() if k.startswith("trained - untrained")), None)
out["cAddLate"] = json.loads((OVER / "mv_ranker.json").read_text(encoding="utf-8"))["runs"]["rk_mv"]
save("s6_macros", out)
print(json.dumps(out, indent=1, default=float))
