"""E7b: the stored multi-vector scores, read for the first time. No call, no card, development cases only."""
import json, os
from pathlib import Path
import numpy as np, pandas as pd
from uol.final import footing as FT, rank_rerun as R, audit_d181_common as A
from uol.final import multivector as MV
R.D.activate(R.STORE)
from uol.embedding_push import common as ep
from uol.final import context, test_set as TS, guard
context.active()
d = ep.release()
out = MV.exchange() / "outputs"
norm = lambda t: re.sub(r"\s+", " ", t).strip()
import re
active = {norm(q.text): i for i, q in enumerate(d.queries)}
base = A.per_case(R.STORE, "gpt61sol_alone")
keep = [int(r) for r in base["row"]]
from collections import defaultdict, deque
wanted = defaultdict(deque)
for r in keep:
    wanted[norm(d.queries[r].text)].append(r)
print("kept cases sharing a text:", sum(len(v) - 1 for v in wanted.values() if len(v) > 1), flush=True)
total = missing = 0
maps = {}
for k in range(MV.FOLDS):
    q = [json.loads(x) for x in (MV.exchange() / f"f{k}" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
    maps[k] = [(wanted[norm(x["text"])].popleft() if wanted.get(norm(x["text"])) else None) for x in q]
    total += len(q)
    missing += sum(norm(x["text"]) not in active for x in q)
    tr = [json.loads(x) for x in (MV.exchange() / f"f{k}" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    field = next(f for f in tr[0] if isinstance(tr[0][f], str) and len(tr[0][f]) > 80)
    miss_tr = sum(norm(x[field]) not in active for x in tr)
    print(f"fold {k}: queries {len(q)} scored kept {sum(v is not None for v in maps[k])}; training pairs {len(tr)}, query text outside the active development release {miss_tr}", flush=True)
print("queries", total, "outside the active development release", missing, "active release cases", len(d.queries), flush=True)
found = sum(sum(v is not None for v in maps[k]) for k in maps)
if found != len(keep) or missing:
    raise SystemExit(f"cannot join: {found} of {len(keep)}, {missing} outside")
bad = missing
frames, res = {}, {}
for name, folder in (("trained", lambda k: out / f"f{k}"), ("untrained", lambda k: out / "base" / f"f{k}")):
    rows, blocks = [], []
    for k in range(MV.FOLDS):
        S = np.load(folder(k) / "scores.npy", mmap_mode="r")
        idx = [i for i, r in enumerate(maps[k]) if r is not None]
        rows += [maps[k][i] for i in idx]
        blocks.append(np.asarray(S[idx], dtype=np.float32))
    S = np.concatenate(blocks)
    pos = {r: i for i, r in enumerate(rows)}
    M = np.asarray(S[[pos[r] for r in keep]], dtype=np.float64)
    pc, summary = TS.measure(keep, M, False)
    frames[name] = pd.DataFrame(pc) if not isinstance(pc, pd.DataFrame) else pc
    res[name] = {m: summary.get(m) for m in ("ndcg_at_10", "success_at_1", "recall_at_20", "recall_at_100", "unseen_recall_at_20_pooled", "n_cases", "n_groups")}
    print(name, json.dumps(res[name], default=float), flush=True)
    frames[name].to_parquet(A.WORK / "overnight" / f"mv_{name}.parquet")
def stored(s):
    from pathlib import Path
    for st in ("experiments/runs/footing_20261101/dev/store/runs", "experiments/runs/final_20261025/store_dev_20261025b/runs", "experiments/runs/audit_d181_20261028/dev/store/runs"):
        p = A.ROOT / st / s / "DEV" / "per_case.parquet"
        if p.exists():
            return pd.read_parquet(p), json.loads((p.parent / "summary.json").read_text(encoding="utf-8"))
    raise FileNotFoundError(s)
others = {s: stored(s) for s in ("qwen3_8b", "bge_m3_colbert", "bge_m3")}
for s, (f, j) in others.items():
    res[s] = {m: j.get(m) for m in ("ndcg_at_10", "success_at_1", "recall_at_20", "recall_at_100", "unseen_recall_at_20_pooled")}
    print(s, json.dumps(res[s], default=float), flush=True)
con = {}
for a, b in (("trained", "qwen3_8b"), ("untrained", "qwen3_8b"), ("trained", "untrained"), ("trained", "bge_m3_colbert")):
    fb = frames.get(b, others[b][0] if b in others else None)
    fam = A.dev_family(f"mv_{a}_{b}", [(a, frames[a], b, fb)], population="DEV", measure="ndcg_at_10", size=4)
    c = next(iter(fam["contrasts"].values()))
    con[f"{a} - {b}"] = {x: c[x] for x in ("delta", "ci95", "adjusted_lower", "class")}
    print(a, "-", b, round(c["delta"], 4), [round(x, 3) for x in c["ci95"]], round(c["adjusted_lower"], 4), c["class"], flush=True)
(A.ROOT / "research/programs/overnight_20261105/e7b_read.json").write_text(json.dumps({"alone": res, "contrasts": con, "text_mismatch": bad}, indent=1, default=float), encoding="utf-8")
