"""Rerankers over the S_full_ext top 30 (RUN_PLAN v2 §3.8, E9), all scored on the same candidates.

* ``ce_trained``: Qwen3-Reranker-0.6B fine-tuned on the citations per outer fold with the Round 3 R2 recipe reused
  unchanged (``round3.r2_cross.build_fold`` with ``score_source="stage2"`` pointed at the S_full_ext top lists of this
  run; ``FULL_TASK``; ``round3/gpu_jobs/r2_train.py``).
* zero-shot ``rerank_bge`` (bge-reranker-v2-m3) and ``rerank_qwen3_4b`` (Qwen3-Reranker-4B, bf16): the Round 3 R1
  inputs and job (``round3.r1_zeroshot.build_inputs`` at depth 30 with ``features.ltr_topk`` pointed at the S_full_ext
  top lists; ``round3/gpu_jobs/r1_score.py``; revisions pinned in ``rerank_bench.MODELS``).

Each reranker gives two systems: ``<id>`` = fused with S_full_ext by ``round3.features.fusion_route`` (cross-fitted
weight over ``W_GRID``), and ``<id>_alone`` = the top 30 reordered by the reranker score alone (the rest of the
S_full_ext ranking below it). The saved R1 scores over the LTR_emb_hgb top 50 are not used for results.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from uol.final import context, guard

BASE = "S_full_ext"
TOP = 30
ZERO_SHOT = {"rerank_bge": "bge_rr_v2_m3", "rerank_qwen3_4b": "qwen3_rr_4b"}


def _top(store: Path, rows: np.ndarray, k: int) -> np.ndarray:
    from uol.final import llm_jobs
    prof = context.active()
    S = llm_jobs.base_matrix(store, f"{BASE}{prof.tag}", rows)
    return np.stack([np.argsort(-S[i], kind="stable")[:k] for i in range(len(rows))])


def tag(k: int, seed: int = 1) -> str:
    return f"ce_trained_f{k}_s{seed}"


def prep(k: int, seed: int = 1) -> Path:
    from uol.final.adapters import ltr
    ltr.use_table("dev")                          # nested seed refits read the development pair table
    from uol.round3 import r2_cross
    prof = context.active()
    store = prof.store

    def stage2_topk(rows, depth):
        return _top(store, guard.dev_rows_only(rows), depth)
    r2_cross.stage2_topk = stage2_topk
    prof.patches["uol.round3.r2_cross.stage2_topk"] = f"{BASE} top lists of this run"
    inputs = r2_cross.build_fold(k, tag(k, seed), seed, score_source="stage2")
    task = {**r2_cross.FULL_TASK, "seed": seed, "key": tag(k, seed)}
    out = Path(inputs["train.jsonl"]).parent
    (out / "task.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    return out


def inputs_dir(k: int, seed: int = 1) -> Path:
    from uol.round3 import r2_cross
    return r2_cross.OUT / "inputs" / tag(k, seed)


def zs_inputs_dir(system: str) -> Path:
    return context.active().work / "gpu_inputs" / system


def zs_prep(system: str) -> Path:
    """R1 inputs (queries, documents, pairs) over the S_full_ext top 30 of DEV, plus the R1 task for the model."""
    from uol.round3 import features, r1_zeroshot, rerank_setup
    prof = context.active()
    store = prof.store
    out = zs_inputs_dir(system)
    out.mkdir(parents=True, exist_ok=True)
    orig_topk, orig_inputs, orig_depth = features.ltr_topk, r1_zeroshot.INPUTS, r1_zeroshot.DEPTH
    features.ltr_topk = lambda name, rows, k: _top(store, guard.dev_rows_only(rows), k)
    r1_zeroshot.INPUTS, r1_zeroshot.DEPTH = out, TOP
    try:
        r1_zeroshot.build_inputs()
    finally:
        features.ltr_topk, r1_zeroshot.INPUTS, r1_zeroshot.DEPTH = orig_topk, orig_inputs, orig_depth
    model = ZERO_SHOT[system]
    task = {"model": model, "quant": None, "instruction": rerank_setup.INSTRUCTION, "max_query_tokens": 384,
            "chunk_tokens": 1024, "max_chunks": 4, "batch_size": 8, "key": f"final_{system}_top{TOP}"}
    (out / "task.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    return out


def _fuse(system: str, pairs: np.ndarray, scores: np.ndarray, meta: dict) -> Path:
    from uol.embedding_push import common as ep
    from uol.final import llm_jobs
    from uol.round3 import features
    prof = context.active()
    rows = llm_jobs.dev_eval_rows()
    N = len(ep.release().candidate_ids)
    A = np.full((len(rows), N), np.nan, dtype=np.float32)
    pos = {int(r): i for i, r in enumerate(rows)}
    top30 = _top(prof.store, rows, TOP)
    keep = {int(r): set(top30[i].tolist()) for i, r in enumerate(rows)}
    for (r, c), s in zip(pairs, scores):
        if int(r) in pos and int(c) in keep[int(r)]:
            A[pos[int(r)], int(c)] = s
    S = llm_jobs.base_matrix(prof.store, f"{BASE}{prof.tag}", rows)
    fr = features.fusion_route(BASE, rows, A, ltr_matrix=S)
    llm_jobs.save_matrix(system, rows, fr["oof_matrix"], {**meta, "base": BASE, "top": TOP, "w_by_fold": fr["w_by_fold"],
                                                         "w_all": fr["w_all"], "fusion": "round3.features.fusion_route"})
    alone = features.fuse_rows(S, A, 1e6)                  # the reranker order inside the top 30, S_full_ext below
    return llm_jobs.save_matrix(f"{system}_alone", rows, alone, {**meta, "base": BASE, "top": TOP,
                                                                 "rule": "top 30 reordered by the reranker alone"})


def fuse(seed: int = 1) -> Path:
    prof = context.active()
    P, S = [], []
    for k in range(5):
        P.append(np.load(inputs_dir(k, seed) / "pairs.npy"))
        S.append(np.load(prof.work / "gpu_jobs" / tag(k, seed) / "outputs" / "scores.npy"))
    return _fuse("ce_trained", np.concatenate(P), np.concatenate(S),
                 {"seed": seed, "recipe": "round3.r2_cross FULL_TASK per fold"})


def zs_fuse(system: str) -> Path:
    prof = context.active()
    pairs = np.load(zs_inputs_dir(system) / "pairs.npy")
    sc = np.load(prof.work / "gpu_jobs" / system / "outputs" / "scores.npy")
    keep = pairs[:, 0] >= 0                                # the blank-query control pairs are not part of the ranking
    return _fuse(system, pairs[keep], sc[keep], {"model": ZERO_SHOT[system], "recipe": "round3 R1 job, zero-shot"})


def smoke_inputs(kind: str) -> Path:
    """Smoke stage: tiny, valid inputs for the reranker GPU jobs (``ce``: r2_train.py; ``zs``: r1_score.py) from a few
    development rows (training rows of fold 0 for training, fold-0 rows for scoring); exercises the jobs, not results."""
    import json as _json
    from uol.embedding_push import common as ep
    from uol.round3 import r1_zeroshot, r2_cross, rerank_setup
    prof = context.active()
    d = ep.release()
    roles = ep.roles()
    rows = guard.dev_rows_only(ep.population_rows("P_spec", ep.dev_rows()))
    train = [int(r) for r in rows if roles[int(r)] != "dev_fold0"][:16]
    score = [int(r) for r in rows if roles[int(r)] == "dev_fold0"][:3]
    paths = _json.loads((prof.work / "round3_run" / "phase2" / "statute_graph" / "paths.json").read_text(encoding="utf-8")) \
        if (prof.work / "round3_run" / "phase2" / "statute_graph" / "paths.json").exists() else {}
    rng = np.random.default_rng(20261021)
    N = len(d.candidate_ids)
    out = prof.work / "gpu_inputs" / f"smoke_{kind}"
    out.mkdir(parents=True, exist_ok=True)
    cands = sorted({int(c) for c in rng.integers(0, N, 60)} | {int(p) for r in score for p in d.specific_idx[r]})
    if kind == "ce":
        with (out / "train.jsonl").open("w", encoding="utf-8") as f:
            for r in train:
                p = d.specific_idx[r][0]
                negs = [int(c) for c in rng.integers(0, N, 15) if int(c) not in d.positive_idx[r]]
                f.write(_json.dumps({"q": d.queries[r].text, "docs": [r1_zeroshot.doc_text(int(p), paths)] +
                                     [r1_zeroshot.doc_text(c, paths) for c in negs]}, ensure_ascii=False) + "\n")
        task = {**r2_cross.FULL_TASK, "seed": 1, "key": "smoke_ce", "epochs": 1}
    else:
        task = {"model": ZERO_SHOT["rerank_bge"], "quant": None, "instruction": rerank_setup.INSTRUCTION,
                "max_query_tokens": 384, "chunk_tokens": 1024, "max_chunks": 4, "batch_size": 8, "key": "smoke_zs"}
    with (out / "queries.jsonl").open("w", encoding="utf-8") as f:
        for r in score:
            f.write(_json.dumps({"qid": r, "text": d.queries[r].text}, ensure_ascii=False) + "\n")
    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for c in cands:
            f.write(_json.dumps({"did": c, "text": r1_zeroshot.doc_text(c, paths)}, ensure_ascii=False) + "\n")
    pairs = np.asarray([[r, c] for r in score for c in cands[:12]], dtype=np.int32)
    np.save(out / "pairs.npy", pairs)
    (out / "task.json").write_text(_json.dumps(task, indent=1), encoding="utf-8")
    return out
