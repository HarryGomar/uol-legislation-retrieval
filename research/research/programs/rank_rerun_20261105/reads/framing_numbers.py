"""The stored-run numbers FRAMING.md asks of the ranker session (decisions 3(b), 4(b), 4(d), H088): the four setups at deeper
cut-offs, the model's list completed by each method, and the blank control. Development cases, stored lists, no model call,
nothing refitted, no test file. Each measure is checked against the stored summary where one exists. Writes
out/framing_numbers.json, which scripts/final_results.py turns into macros."""
from __future__ import annotations
import math
from common import *

SYS, ALONE_, RANK, NOENC_ = "gpt61sol_rk_system", "gpt61sol_alone", "rt_lm_tesis", "rk_noenc"
CUTS = (10, 20, 50, 100)
qid = dev_qids()
base = pc(RANK).set_index("query_id")
grp = base["component_id"].reindex(qid).to_numpy()
B = Boot(grp, n=100_000)
pos = {q: set(ints(base.loc[q, "positive_idx"])) for q in qid}


def ndcg(rs: list[int], n: int, k: int) -> float:
    if not n:
        return 0.0
    z = sum(1 / math.log2(1 + r) for r in range(1, min(n, k) + 1))
    return sum(1 / math.log2(1 + r) for r in rs if r <= k) / z


def values(lists: dict[str, list[int]], k: int, kind: str) -> np.ndarray:
    out = []
    for q in qid:
        P = pos[q]
        rs = [i + 1 for i, c in enumerate(lists.get(q, [])) if c in P]
        out.append(ndcg(rs, len(P), k) if kind == "ndcg" else sum(r <= k for r in rs) / len(P) if P else 0.0)
    return np.array(out)


def named_of(sid: str) -> dict[str, list[int]]:
    t = top(sid)
    t = t[t["score"] > -1e8].sort_values(["query_id", "rank"])
    return {q: g["candidate_idx"].astype(int).tolist() for q, g in t.groupby("query_id")}


def complete(named: dict, fill: dict) -> dict:
    out = {}
    for q in qid:
        li = list(named.get(q, []))
        seen = set(li)
        for c in fill.get(q, []):
            if c not in seen:
                li.append(c)
                seen.add(c)
        out[q] = li
    return out


OUT: dict = {"depth": {}, "fill": {}, "check": {}}
lists = {"noenc": first(NOENC_, 200), "rank": first(RANK, 200), "alone": first(ALONE_, 200), "system": first(SYS, 200)}
sid_of = {"noenc": NOENC_, "rank": RANK, "alone": ALONE_, "system": SYS}
for name, li in lists.items():
    OUT["depth"][name] = {f"{m}_{k}": gm(values(li, k, m), grp) for m in ("ndcg", "recall") for k in CUTS}
    s = summary(sid_of[name])
    for got, key in ((OUT["depth"][name]["ndcg_10"], "ndcg_at_10"), (OUT["depth"][name]["recall_20"], "recall_at_20"), (OUT["depth"][name]["recall_100"], "recall_at_100")):
        OUT["check"][f"{name}_{key}"] = [round(got, 4), round(s[key], 4)]
        if abs(got - s[key]) > 5e-4:
            raise SystemExit(f"{name} {key}: {got} against the stored {s[key]}")

named = named_of(ALONE_)
fills = {"none": named, "lexical": complete(named, first(LEX, 200)), "encoder": complete(named, first(ENC, 200)),
         "noenc": complete(named, lists["noenc"]), "rank": complete(named, lists["rank"])}
vals = {k: values(li, 100, "recall") for k, li in fills.items()}
OUT["fill"]["recall_100"] = {k: gm(v, grp) for k, v in vals.items()}
OUT["fill"]["rank_minus_lexical"] = B.group_mean(vals["rank"] - vals["lexical"])

ln = np.array([len(named.get(q, [])) for q in qid])
blind = np.array([not (set(named.get(q, [])) & pos[q]) for q in qid])
OUT["blind"] = {"cases": int(blind.sum())}
for name, li in (("rank", lists["rank"]), ("system", lists["system"])):
    for k in (10, 100):
        hit = values(li, k, "recall") > 0
        OUT["blind"][f"{name}_{k}"] = int(hit[blind].sum())
OUT["named"] = {"median": float(np.median(ln)), "max": int(ln.max())}

OUT["blank"] = {}
for key, s in (("list", "S_full_ext"), ("no_text", "llm_rerank_S_full_ext_blank"), ("text", "llm_rerank_S_full_ext")):
    j = summary(s)
    OUT["blank"][key] = {"ndcg": j["ndcg_at_10"], "law1": j["law_at_1"], "in_law": j["success_at_1_in_law"]}

save("framing_numbers", OUT)
print(json.dumps(OUT, indent=1, default=float)[:2600])
