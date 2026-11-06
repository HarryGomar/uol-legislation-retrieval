"""R2: cross-encoders trained on citation labels (PLAN §3 R2; Phase 4.2). Pilot first: Qwen3-Reranker-0.6B, LoRA r=32.

Pilot (2 folds, seed 1). For held-out fold k in {0, 1}:
* training groups: every dev P_spec row of the folds other than k, one group per specific positive: the positive, 13
  negatives sampled from the row's LTR ranks 1-50 (strictly nested seeds ``features.seed_scores(k, ...)``: trained
  without fold k and without the row's own fold; same-law siblings kept), and 2 random negatives from the L_star
  (BM25-family) top 100. Generic and positive candidates are never negatives.
* scoring pairs: the fold-k P_spec rows x their frozen out-of-fold LTR_emb_hgb top 50 (no labels shipped).
Gate: fusion-route dev Δ (P_spec, the two pilot folds, w cross-fitted between them) ≥ +.01 -> the full run (5 folds x
3 seeds); else R2 stops at the pilot and is reported.

    python -m uol.round3.r2_cross pilot-submit       # blocks (two remote jobs); run in the background
    python -m uol.round3.r2_cross pilot-evaluate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uol.round3 import common as C
from uol.round3 import features, gpu_remote, r1_zeroshot, rerank_setup, stats

OUT = C.ARMS / "R2"
ENTRY = Path(__file__).resolve().parent / "gpu_jobs" / "r2_train.py"
HELPER = Path(__file__).resolve().parent / "gpu_jobs" / "rerank_bench.py"
N_LTR_NEG, N_BM25_NEG, DEPTH = 13, 2, 50
PILOT_TASK = {"model": "qwen3_rr_0.6b", "lora_r": 32, "lr": 1e-4, "epochs": 2, "group_size": 16, "batch_groups": 4,
              "max_query_tokens": 384, "max_doc_tokens": 256, "seed": 1, "loss": "lce",
              "instruction": rerank_setup.INSTRUCTION}


def stage2_topk(rows: np.ndarray, k: int) -> np.ndarray:
    """Top k of the C_hybrid stage-2 ranking for dev P_spec rows (saved by c_hybrid.stage2)."""
    hr = np.load(C.ARMS / "C_hybrid" / "stage2_pspec_rows.npy")
    M = np.load(C.ARMS / "C_hybrid" / "stage2_pspec_matrix.npy", mmap_mode="r")
    pos = {int(r): i for i, r in enumerate(hr)}
    return np.stack([np.argsort(-np.asarray(M[pos[int(r)]], dtype=np.float64), kind="stable")[:k] for r in rows])


def build_fold(k: int, tag: str, seed: int = 1, score_source: str = "ltr") -> dict[str, Path]:
    d = C.release()
    paths = json.loads((C.RUN / "phase2/statute_graph/paths.json").read_text(encoding="utf-8"))
    pspec = C.population_rows("P_spec", C.dev_rows())
    fold = C.fold_of(pspec)
    train_rows = pspec[fold != k]
    score_rows = pspec[fold == k]
    if np.isin(np.concatenate([train_rows, score_rows]), C.lockbox_rows()).any():
        raise PermissionError("dev rows only")
    seeds = features.seed_scores(k, train_rows)
    lstar = C.ep.open_scores("L_star")
    generic = set(d.generic_idx.tolist()) if len(d.generic_idx) else set()
    rng = np.random.default_rng(C.SEED + 100 * k + seed)
    out = OUT / "inputs" / tag
    out.mkdir(parents=True, exist_ok=True)
    doc_cache: dict[int, str] = {}

    def doc(i):
        if i not in doc_cache:
            doc_cache[i] = r1_zeroshot.doc_text(int(i), paths)
        return doc_cache[i]

    n_groups = 0
    with (out / "train.jsonl").open("w", encoding="utf-8") as f:
        for r in train_rows:
            r = int(r)
            pos = set(int(a) for a in d.positive_idx[r])
            cands, sc = seeds[r]
            order = np.argsort(-sc, kind="stable")
            ltr_top = [int(c) for c in cands[order][:DEPTH] if int(c) not in pos and int(c) not in generic]
            lrow = np.asarray(lstar[r], dtype=np.float64)
            bm_top = [int(c) for c in np.argsort(-lrow, kind="stable")[:100] if int(c) not in pos and int(c) not in generic]
            for p in d.specific_idx[r]:
                negs = list(rng.choice(ltr_top, min(N_LTR_NEG, len(ltr_top)), replace=False)) if ltr_top else []
                pool = [c for c in bm_top if c not in negs]
                negs += list(rng.choice(pool, min(N_BM25_NEG, len(pool)), replace=False)) if pool else []
                f.write(json.dumps({"q": d.queries[r].text, "docs": [doc(int(p))] + [doc(int(c)) for c in negs]},
                                   ensure_ascii=False) + "\n")
                n_groups += 1
    top = features.ltr_topk("LTR_emb_hgb", score_rows, DEPTH) if score_source == "ltr" else stage2_topk(score_rows, DEPTH)
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in score_rows:
            f.write(json.dumps({"qid": int(r), "text": d.queries[int(r)].text}, ensure_ascii=False) + "\n")
    cands = np.unique(top)
    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(json.dumps({"did": int(c), "text": doc(int(c))}, ensure_ascii=False) + "\n")
    np.save(out / "pairs.npy", np.stack([np.repeat(score_rows, DEPTH), top.ravel()], 1).astype(np.int32))
    C.write_json(out / "index.json", {"fold": k, "train_rows": len(train_rows), "groups": n_groups,
                                      "score_rows": score_rows.tolist(), "depth": DEPTH})
    return {n: out / n for n in ("train.jsonl", "queries.jsonl", "docs.jsonl", "pairs.npy")}


def submit(k: int, task: dict, tag: str, score_source: str = "ltr") -> dict:
    inputs = build_fold(k, tag, task["seed"], score_source)
    tp = OUT / "inputs" / tag / "task.json"
    tp.write_text(json.dumps({**task, "key": tag}, indent=1), encoding="utf-8")
    inputs["task.json"] = tp
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": tag, "gpu": "remote RTX 5080", "task": task})
    receipt = gpu_remote.submit(tag, ENTRY, inputs, helpers=[HELPER], expected_outputs=["scores.npy"])
    C.ledger({"kind": "job_end", "job_id": tag, "start_utc": start, "end_utc": C.utc(),
              "exit_code": receipt.get("exit_code"), "gpu_seconds": receipt.get("seconds")})
    return receipt


def arm_matrix(tags: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(rows, [rows, N] arm scores, NaN outside each row's scored top 50) from returned jobs."""
    d = C.release()
    rows_all, mats = [], []
    for tag in tags:
        idx = json.loads((OUT / "inputs" / tag / "index.json").read_text(encoding="utf-8"))
        pairs = np.load(OUT / "inputs" / tag / "pairs.npy")
        sc = np.load(gpu_remote.LOCAL_JOBS / tag / "remote" / "outputs" / "scores.npy")
        rows = np.asarray(idx["score_rows"])
        M = np.full((len(rows), len(d.candidate_ids)), np.nan, dtype=np.float32)
        pos = {int(r): i for i, r in enumerate(rows)}
        for (q, c), v in zip(pairs, sc):
            M[pos[int(q)], int(c)] = v
        rows_all.append(rows)
        mats.append(M)
    return np.concatenate(rows_all), np.concatenate(mats)


def evaluate(tags: list[str], name: str) -> dict:
    rows, A = arm_matrix(tags)
    fr = features.fusion_route("LTR_emb_hgb", rows, A)
    keep = C.population_rows("P_spec", rows)
    sel = np.isin(rows, keep)
    ltr = np.asarray(features.frozen_matrix("LTR_emb_hgb")[rows[sel]])
    a = C.ep.evaluate(fr["oof_matrix"][sel], rows[sel], "P_spec")
    b = C.ep.evaluate(ltr, rows[sel], "P_spec")
    alone = A[sel].copy()
    alone[~np.isfinite(alone)] = -1e9
    s = C.ep.evaluate(alone, rows[sel], "P_spec")
    c = stats.contrast(a["per_component"]["ndcg_at_10"], b["per_component"]["ndcg_at_10"])
    res = {"name": name, "jobs": tags, "rows": int(sel.sum()), "w_by_fold": fr["w_by_fold"], "w_all": fr["w_all"],
           "dev_ndcg_by_w": fr["dev_ndcg_by_w"], "fused": a["macro"]["ndcg_at_10"], "ltr": b["macro"]["ndcg_at_10"],
           "standalone_top50": s["macro"]["ndcg_at_10"], "contrast": c}
    C.write_json(OUT / f"{name}.json", res)
    return res


def light_gate(tags: list[str]) -> dict:
    """Pilot gate on the scored top 50 only (cheap): z(LTR) + w z(R2), w cross-fitted between the pilot folds."""
    d = C.release()
    generic = set(d.generic_idx.tolist()) if len(d.generic_idx) else set()
    M = features.frozen_matrix("LTR_emb_hgb")
    W = features.W_GRID
    per = {}
    for tag in tags:
        pairs = np.load(OUT / "inputs" / tag / "pairs.npy")
        sc = np.load(gpu_remote.LOCAL_JOBS / tag / "remote" / "outputs" / "scores.npy")
        for (q, c), v in zip(pairs, sc):
            per.setdefault(int(q), {})[int(c)] = float(v)
    rows = sorted(per)
    folds, comps = C.fold_of(rows), C.component_ids(rows)

    def nd(order, gold):
        order = [c for c in order if c not in generic]
        dcg = sum(1 / np.log2(i + 2) for i, c in enumerate(order[:10]) if c in gold)
        idcg = sum(1 / np.log2(i + 2) for i in range(min(10, len(gold))))
        return dcg / idcg if idcg else 0.0

    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else 0 * x

    vals = {w: [] for w in W}
    for r in rows:
        cand = np.array(list(per[r]))
        l = np.asarray(M[r], dtype=np.float64)[cand]
        a = np.array([per[r][c] for c in cand])
        gold = set(int(x) for x in d.specific_idx[r])
        for w in W:
            vals[w].append(nd(list(cand[np.argsort(-(z(l) + w * z(a)), kind="stable")]), gold))

    def macro(v, mask):
        by = {}
        for x, c, m in zip(v, comps, mask):
            if m:
                by.setdefault(c, []).append(x)
        return [float(np.mean(b)) for b in by.values()]
    oof = np.zeros(len(rows))
    for f in sorted(set(folds.tolist())):
        best = max(W, key=lambda w: (np.mean(macro(vals[w], folds != f)), -w))
        oof[folds == f] = np.asarray(vals[best])[folds == f]
    base = np.asarray(vals[0.0])
    by_c = {}
    for x, b, c in zip(oof, base, comps):
        by_c.setdefault(c, []).append((x, b))
    a_c = np.array([np.mean([p[0] for p in v]) for v in by_c.values()])
    b_c = np.array([np.mean([p[1] for p in v]) for v in by_c.values()])
    con = stats.contrast(a_c, b_c)
    res = {"rows": len(rows), "ltr": float(b_c.mean()), "fused_crossfit": float(a_c.mean()),
           "delta": con["delta"], "ci95": con["ci95"], "pass": con["delta"] >= 0.01,
           "note": "light gate on the scored top 50; w cross-fitted between the pilot folds"}
    C.write_json(OUT / "PILOT_GATE.json", res)
    return res


FULL_TASK = {**PILOT_TASK}


def full_submit(seeds=(1, 2, 3)) -> None:
    """All 5 folds x seeds; scoring pairs = C_hybrid stage-2 top 50 (feeds the hybrid). Skips returned jobs."""
    for seed in seeds:
        for k in range(5):
            tag = f"r2_qwen06b_f{k}_s{seed}"
            if (gpu_remote.LOCAL_JOBS / tag / "remote" / "outputs" / "scores.npy").exists():
                continue
            print(json.dumps(submit(k, {**FULL_TASK, "seed": seed}, tag, "stage2"), indent=1)[:600], flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["pilot-submit", "pilot-evaluate", "pilot-gate", "full-submit"])
    ap.add_argument("--seeds", default="1,2,3")
    a = ap.parse_args()
    tags = [f"r2_qwen06b_pilot_f{k}_s1" for k in (0, 1)]
    if a.command == "pilot-gate":
        print(json.dumps(light_gate(tags), indent=1, default=float))
        return 0
    if a.command == "full-submit":
        full_submit(tuple(int(x) for x in a.seeds.split(",")))
        return 0
    if a.command == "pilot-submit":
        for k, tag in zip((0, 1), tags):
            if not (gpu_remote.LOCAL_JOBS / tag / "remote" / "outputs" / "scores.npy").exists():
                print(json.dumps(submit(k, PILOT_TASK, tag), indent=1)[:1500])
        return 0
    from uol.round3 import heavy
    heavy.acquire()                           # the fusion evaluation is a heavy step (one-heavy-process rule)
    res = evaluate(tags, "PILOT")
    gate = res["contrast"]["delta"] >= 0.01
    res["gate"] = {"rule": "fusion dev Δ ≥ +.01 on the two pilot folds", "pass": bool(gate)}
    C.write_json(OUT / "PILOT.json", res)
    print(json.dumps({k: v for k, v in res.items() if k != "contrast"}, indent=1, default=float),
          json.dumps({k: res["contrast"][k] for k in ("delta", "ci95", "class")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
