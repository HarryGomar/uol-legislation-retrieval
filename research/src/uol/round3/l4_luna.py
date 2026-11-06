"""L4: filtered synthetic fact patterns (doc2query--) for never-cited articles (PLAN §3 L4; Tier 2; effort low).

Target set (label-free except dev fold counts; D6.1): the 14,900 articles never cited by a dev component that appear in
the LTR_emb_hgb top 100 of a dev or lockbox query, plus, per outer fold k, articles never cited in folds != k that appear
in the top 100 of a fold-k query (home fold k). Generation: 10 articles per call, 4 patterns each, with 3 masked dev
Hechos from rows citing the same instrument shown as style-only examples (never rows citing a target in the call,
never rows of a target's home fold). Filter: a pattern is kept if its own article ranks in the top 10 of the catalogue
under robust-z(BM25) + robust-z(Qwen dense, query variant c); patterns quoting an article number are dropped.
Feature (feature route): per (query, candidate) the max and mean cosine between the query's Qwen vector (variant c)
and the candidate's kept patterns; NaN when the candidate has none.

    python -m uol.round3.l4_luna generate [--workers 8]
    python -m uol.round3.l4_luna filter
    python -m uol.round3.l4_luna arm                    # only after every Tier-1 arm has a dev result (D6.0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from uol.round3 import arm_runner, common as C, features, luna, prompt_guard

PROMPT = Path(__file__).resolve().parent / "prompts" / "L4.txt"
OUT = C.ARMS / "L4"
EFFORT = "low"
PER_CALL, N_PAT, N_EX = 10, 4, 3
ITEM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["hechos"],
               "properties": {"hechos": {"type": "array", "minItems": 1, "maxItems": 6, "items": {"type": "string"}}}}


def instrument(i: int) -> str:
    return C.release().candidate_ids[i].rsplit("/", 1)[-1].split(".pdf")[0]


def target_set() -> dict[int, int]:
    """{article: home fold (-1 = never cited in dev)}."""
    path = OUT / "target_set.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    M = features.frozen_matrix("LTR_emb_hgb")
    top = {int(r): np.argsort(-np.asarray(M[int(r)], dtype=np.float64), kind="stable")[:100]
           for r in list(C.dev_rows()) + list(C.lockbox_rows())}
    lock_counts = C._counts_for("lockbox")
    out: dict[int, int] = {}
    for r, t in top.items():
        for a in t:
            if lock_counts[a] == 0:
                out[int(a)] = -1
    for k in range(5):
        c = C._counts_for(f"dev_fold{k}")
        for r in C.fold_rows(k):
            for a in top[int(r)]:
                if c[a] == 0 and int(a) not in out:
                    out[int(a)] = k
    C.write_json(path, {str(k): v for k, v in sorted(out.items())})
    return out


def example_pool() -> dict[str, list[tuple[int, int, frozenset[int]]]]:
    """instrument -> [(dev row, fold, cited articles)] for dev rows citing that instrument."""
    d = C.release()
    pool: dict[str, list] = defaultdict(list)
    for r in C.dev_rows():
        r = int(r)
        cited = frozenset(int(a) for a in d.positive_idx[r])
        for inst in {instrument(a) for a in cited}:
            pool[inst].append((r, int(C.fold_of([r])[0]), cited))
    return pool


def calls() -> list[dict]:
    """Deterministic call plan: targets grouped by instrument, 10 per call, with 3 style examples per call."""
    tgt = target_set()
    pool = example_pool()
    by_inst: dict[str, list[int]] = defaultdict(list)
    for a in sorted(tgt):
        by_inst[instrument(a)].append(a)
    rng = np.random.default_rng(C.SEED + 4)
    out = []
    for inst in sorted(by_inst):
        arts = by_inst[inst]
        for s in range(0, len(arts), PER_CALL):
            part = arts[s:s + PER_CALL]
            bad_folds = {tgt[a] for a in part if tgt[a] >= 0}
            cands = [r for r, f, cited in pool.get(inst, []) if f not in bad_folds and not (cited & set(part))]
            if len(cands) < N_EX:                       # fall back to any dev row not citing the targets
                cands += [r for lst in pool.values() for r, f, cited in lst
                          if f not in bad_folds and not (cited & set(part)) and r not in cands][:200]
            ex = [int(x) for x in rng.choice(sorted(set(cands)), min(N_EX, len(set(cands))), replace=False)]
            out.append({"articles": part, "examples": ex, "instrument": inst})
    return out


def article_block(i: int) -> str:
    d = C.release()
    title, _, body = d.candidate_texts[i].partition("\n")
    return f"{title.strip()}:\n{re.sub(r'\s+', ' ', body).strip()[:1500]}"


def build(call: dict, tpl: str) -> tuple[str, list[str]]:
    d = C.release()
    shown = [re.sub(r"\s+", " ", d.queries[r].text).strip()[:700] for r in call["examples"]]
    ex = "\n\n".join(f"Ejemplo {n}: {t}" for n, t in enumerate(shown, 1))
    ids = [f"a{n:02d}" for n in range(1, len(call["articles"]) + 1)]
    blocks = [article_block(a) for a in call["articles"]]
    items = "\n\n".join(f"[id={i}] {b}" for i, b in zip(ids, blocks))
    prompt = tpl.replace("{examples}", ex).replace("{items}", items)
    prompt_guard.check(prompt, call["examples"], sanctioned=blocks + shown, template=tpl)
    return prompt, ids


def generate(workers: int) -> dict:
    tpl = PROMPT.read_text(encoding="utf-8")
    plan = calls()
    schema = luna.batch_schema(ITEM_SCHEMA)
    built = [build(c, tpl) for c in plan]
    res = luna.run_many([(p, schema) for p, _ in built], effort=EFFORT, tag="L4", workers=workers)
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with (OUT / "patterns_raw.jsonl").open("w", encoding="utf-8") as f:
        for c, (prompt, ids), r in zip(plan, built, res):
            got = luna.checked(r, prompt, schema, ids, effort=EFFORT, tag="L4")
            for i, a in zip(ids, c["articles"]):
                for h in got[i]["hechos"][:N_PAT + 2]:
                    f.write(json.dumps({"article": a, "text": h, "examples": c["examples"]}, ensure_ascii=False) + "\n")
                    n += 1
    meta = {"calls": len(plan), "articles": sum(len(c["articles"]) for c in plan), "patterns": n,
            "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(), "utc": C.utc()}
    C.write_json(OUT / "generate.json", meta)
    return meta


def _rz(x: np.ndarray) -> np.ndarray:
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-9
    return (x - med) / mad


def filter_patterns() -> dict:
    from uol.embedding_push import qwen_endpoint as Q
    ctx, _ = features.context()
    rows = [json.loads(x) for x in (OUT / "patterns_raw.jsonl").read_text(encoding="utf-8").splitlines() if x]
    rows = [r for r in rows if not prompt_guard.ARTICLE_NO.search(r["text"])]
    texts = [r["text"] for r in rows]
    vec = Q.embed_texts([Q.query_input("c", t) for t in texts], label="round3_L4")
    lex = ctx.lex("bm25")
    keep = np.zeros(len(rows), dtype=bool)
    rank_own = np.zeros(len(rows), dtype=np.int64)
    for s in range(0, len(rows), 256):
        with C.cpu_slot():
            fused = _rz(lex.score(texts[s:s + 256]).astype(np.float32)) + _rz(Q.dense_block(vec[s:s + 256]))
        for i in range(fused.shape[0]):
            a = rows[s + i]["article"]
            rank_own[s + i] = int((fused[i] > fused[i, a]).sum()) + 1
    keep = rank_own <= 10
    np.save(OUT / "pattern_vectors.npy", vec[keep].astype(np.float32))
    kept = [r for r, k in zip(rows, keep) if k]
    with (OUT / "patterns_kept.jsonl").open("w", encoding="utf-8") as f:
        for r, rk in zip(kept, rank_own[keep]):
            f.write(json.dumps({**r, "own_rank": int(rk)}, ensure_ascii=False) + "\n")
    meta = {"patterns_after_number_filter": len(rows), "kept": int(keep.sum()),
            "articles_with_kept_patterns": len({r["article"] for r in kept}), "endpoint_inputs": len(texts), "utc": C.utc()}
    C.write_json(OUT / "filter.json", meta)
    C.ledger({"kind": "job_end", "job_id": "L4_filter", "exit_code": 0, "endpoint_inputs": len(texts), **meta})
    return meta


def columns() -> np.ndarray:
    """[n_pairs, 2] (max, mean cosine to the candidate's kept patterns); computed once (label-free)."""
    path = OUT / "cols.npy"
    if path.exists():
        return np.load(path, mmap_mode="r")
    from uol.embedding_push import qwen_endpoint as Q
    ctx, F = features.context()
    kept = [json.loads(x) for x in (OUT / "patterns_kept.jsonl").read_text(encoding="utf-8").splitlines() if x]
    V = np.load(OUT / "pattern_vectors.npy")
    owner = np.asarray([r["article"] for r in kept])
    Qv = Q.query_vectors("c")
    cols = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(F.pq), 2))
    cols[:] = np.nan
    arts = np.unique(owner)
    for r in C.dev_rows():
        r = int(r)
        sl = slice(int(F.row_ptr[r]), int(F.row_ptr[r + 1]))
        pc = F.pc[sl]
        hit = np.isin(pc, arts)
        if not hit.any():
            continue
        sims = V @ Qv[r]
        mx = defaultdict(list)
        for a, s in zip(owner, sims):
            mx[int(a)].append(float(s))
        out = np.full((len(pc), 2), np.nan, dtype=np.float32)
        for j in np.flatnonzero(hit):
            v = mx[int(pc[j])]
            out[j] = (max(v), float(np.mean(v)))
        cols[sl] = out
    cols.flush()
    return np.load(path, mmap_mode="r")


def columns_for(cfg, g, h, draw=None):
    c = np.asarray(columns())
    return c[:, :1] if cfg["cols"] == "max" else c


def arm() -> dict:
    unseen = lambda rows: np.asarray([any(C.band_of_row(int(r))[a] == "unseen" for a in C.release().specific_idx[int(r)])
                                      for r in rows])
    return arm_runner.run("L4", [{"cols": "max"}, {"cols": "max+mean"}], columns_for, subgroups={"unseen_band": unseen},
                          notes="L4 synthetic fact patterns (doc2query--); style-only examples (D6.1); target set 16,067",
                          extra_result={"luna_resample_agreement": "not applicable (generation; filter by retrieval)"})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["generate", "filter", "arm", "plan"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.command == "plan":
        p = calls()
        tpl = PROMPT.read_text(encoding="utf-8")
        for c in p:
            build(c, tpl)
        print({"calls": len(p), "articles": sum(len(c["articles"]) for c in p)})
        return 0
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": f"L4_{a.command}", "command": " ".join(sys.argv)})
    try:
        res = {"generate": lambda: generate(a.workers), "filter": filter_patterns, "arm": arm}[a.command]()
    except luna.LunaQuotaError as exc:
        C.ledger({"kind": "job_end", "job_id": f"L4_{a.command}", "start_utc": start, "end_utc": C.utc(), "exit_code": 3,
                  "status": "paused_quota", "detail": str(exc)[:300]})
        print("paused_quota", exc)
        return 3
    print(json.dumps(res if isinstance(res, dict) and "dev" not in res else {"arm": "L4"}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
