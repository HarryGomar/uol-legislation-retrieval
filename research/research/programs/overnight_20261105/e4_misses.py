"""E4 (PLAN.md, addendum): where do the system's missed cited articles sit? No model call, stored lists, development cases."""
import json, random
from collections import Counter
from uol.final import rank_rerun as R, audit_d181_common as A
from uol.gap_tests.lexical_grid import PATHS

G = R._agents()
R.D.activate(R.STORE)
from uol.embedding_push import common as ep
d = ep.release()
rows = G.all_rows()
paths = json.loads(PATHS.read_text(encoding="utf-8"))
law = [t.split("\n", 1)[0].strip() for t in d.candidate_texts]
top = [(law[i], (paths.get(str(i)) or [None])[0]) for i in range(len(law))]
full = [(law[i], tuple(paths.get(str(i)) or ())) for i in range(len(law))]
has_path = [bool(paths.get(str(i))) for i in range(len(law))]
generic = {int(i) for i in d.generic_idx}
rng = random.Random(0)
specific_pool = [i for i in range(len(law)) if i not in generic]

def read(name, src):
    c, base = Counter(), Counter()
    for r in rows:
        first = [int(x) for x in src[r] if int(x) not in generic]
        where = {x: i for i, x in enumerate(first)}
        f30 = first[:30]
        L, T, F = {law[x] for x in f30}, {top[x] for x in f30 if has_path[x]}, {full[x] for x in f30 if has_path[x]}
        c["laws_in_first_30"] += len(L); c["cases"] += 1
        for g in (int(x) for x in d.specific_idx[r]):
            c["cited"] += 1
            if g in where and where[g] < 30:
                c["found_30"] += 1
                continue
            c["missed_30"] += 1
            c["missed_rank_31_200" if g in where and where[g] < 200 else "missed_beyond_200"] += 1
            kind = "same_division" if has_path[g] and full[g] in F else "same_top_division" if has_path[g] and top[g] in T else "same_law" if law[g] in L else "other_law"
            c[kind] += 1
            c["path_known"] += has_path[g]
            # chance: a random article of the catalogue, against the same first 30
            x = rng.choice(specific_pool)
            base["same_law"] += law[x] in L
            base["n"] += 1
    n = c["missed_30"]
    return {"name": name, "cited": c["cited"], "found_first_30": c["found_30"], "missed_first_30": n, "laws_in_first_30_mean": c["laws_in_first_30"] / c["cases"],
            "missed_shares": {k: c[k] / n for k in ("same_division", "same_top_division", "same_law", "other_law")}, "counts": {k: c[k] for k in ("same_division", "same_top_division", "same_law", "other_law", "missed_rank_31_200", "missed_beyond_200", "path_known")},
            "random_article_same_law_share": base["same_law"] / base["n"]}

out = [read("system", G.ranker_lists(rows, R.SYSTEM)), read("ranker", G.ranker_lists(rows, R.BASE))]
res = {"articles": len(law), "with_heading_path": sum(has_path), "cases": len(rows), "lists": out}
(A.ROOT / "research/programs/overnight_20261105/e4_misses.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps(res, indent=1))
