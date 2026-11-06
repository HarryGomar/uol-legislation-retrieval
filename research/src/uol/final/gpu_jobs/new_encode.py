"""Query side of the new cases (E13) on the run computer: what needs its GPU or its files (D161, D162).

Three processes, one after another (the pinned stacks differ, ``rerank_bench.deps``):

* ``bge``: bge-m3 with its ColBERT head (``embedding_push.colbert.Encoder``, the frozen query recipe: the masked facts
  truncated at 512 tokens): the CLS vector of the dense bge sources and the query token vectors of ``colbert_bge``,
  one text at a time. It also writes the chunk plan of this machine's ColBERT token store, exactly as ``colbert_full``
  built it (checked against the frozen units): ``bge_chunks.npz``.
* ``qwen``: Qwen3-Embedding-8B int8 (``uol.final.qwen_local``, the fidelity-gated local path), input
  ``qwen_endpoint.query_input("c", text)``, one text per batch.
* ``pairs`` (CPU): the ColBERT pair scores of ``colbert_bge``: the pool of a case is the BM25 top 100 (k1 1.2, b .75)
  united with the bge-m3 MaxP top 100; each pool article is scored by late interaction against the token store
  (``<work>/colbert_full/store``), uniform weights, MaxP (the PRIMARY ``uniform_maxp``).

The same three steps run on 50 development cases (``controls.json``), whose stored vectors and frozen pair scores are
the reproduction control (no label is read):
* bge: CLS cosine >= .999 to the stored vector (the frozen colbert_full rule) and token counts equal to the store's;
* Qwen: mean cosine >= .99 and minimum >= .97 to the stored endpoint vectors (Gate E0, ``scale_curve.legal``);
* pairs: the pool equal to the frozen pool on >= 95 % of the controls, and on the common pairs the colbert_full
  fidelity gate: mean |diff| <= 1e-4, max |diff| <= 2e-3, per-case Spearman median >= .999 and minimum >= .99.

``gate`` writes ``GATE.json``; exit 2 when a control fails (the operator then writes "NEW CASES GATE FAILED").

    python new_encode.py --model bge|qwen|pairs|gate --dir <inputs> --out <out>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import uol  # noqa: E402  (the installed code tree; this file sits in the staged overlay folder)

sys.path.insert(0, str(Path(uol.__file__).resolve().parent / "round3" / "gpu_jobs"))

COLBERT_K = 100
# Pool test by overlap, not identity: timing C (26 Oct) showed ~1e-5 cross-machine drift moving rank-boundary ties, and
# one swap in a ~150-article pool is Jaccard .987; a wrong encoding drops far lower. Fixed before any NEW score.
GATE = {"bge_cls_cos_min": 0.999, "qwen_cos_mean": 0.99, "qwen_cos_min": 0.97, "pool_jaccard_mean": 0.99,
        "pool_jaccard_min": 0.95,
        "pairs_mean_abs": 1e-4, "pairs_max_abs": 2e-3, "spearman_median": 0.999, "spearman_min": 0.99}


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def inputs(d: Path, prof) -> dict:
    """NEW texts (``new_queries.jsonl``) and the control development cases (by query id, mapped to bundle rows)."""
    from uol.embedding_push import common as ep
    new = [json.loads(x) for x in (d / "data" / "new_queries.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    for q in new:
        if sha256_text(q["text"]) != q["text_sha256"]:
            raise RuntimeError(f"{q['query_id']}: text sha256 differs")
    ctl_ids = json.loads((d / "data" / "controls.json").read_text(encoding="utf-8"))["query_ids"]
    rel = ep.release()
    pos = {q.query_id: i for i, q in enumerate(rel.queries)}
    missing = [q for q in ctl_ids if q not in pos]
    if missing:
        raise RuntimeError(f"control cases not in this bundle's release: {missing[:3]}")
    if any(q["query_id"] in pos for q in new):
        raise RuntimeError("a new case is a release row of the bundle")
    ctl_rows = [pos[q] for q in ctl_ids]
    return {"ids": [q["query_id"] for q in new] + list(ctl_ids),
            "texts": [q["text"] for q in new] + [rel.queries[r].text for r in ctl_rows],
            "n_new": len(new), "ctl_rows": ctl_rows}


def scratch_dir(prof) -> Path:
    return Path(os.environ.get("UOL_WORKER_SCRATCH", prof.work / "gpu_scratch"))


def root_work(prof) -> Path:
    return prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work


def gpu_info(torch) -> dict:
    return {"gpu": torch.cuda.get_device_name(0), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def run_bge(d: Path, out: Path, prof) -> dict:
    import rerank_bench as RB
    scratch = scratch_dir(prof)
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, RB.PINS)                         # the pinned tokenizer reproduces the frozen chunking
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from uol.embedding_push import colbert as CB, common as ep
    from uol.facts_to_law import bge_gpu
    from uol.final import rebuild
    from uol.final.gpu_jobs import colbert_full as CF
    ep.BGE_DIR = root_work(prof) / "models" / rebuild.MODELS["BGE_DIR"][2]
    rel = ep.release()
    spec = inputs(d, prof)
    t0 = time.perf_counter()
    enc = CB.Encoder("cuda")
    load_s = time.perf_counter() - t0
    tok = enc.tokenizer
    frozen_dir = prof.path("derived_ftl") / "bge_m3_chunks_v1"
    arts = bge_gpu.read_jsonl(Path(rel.candidate_binding["articles_path"]))
    units = [json.loads(x) for x in (frozen_dir / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    texts, cand = [], []
    for i, r in enumerate(arts):
        ch = bge_gpu.chunk_article(r, tok, CF.DOC_MAX, CF.OVERLAP, CF.CAP)
        texts += ch
        cand += [i] * len(ch)
    if len(texts) != len(units) or any(units[u]["candidate_id"] != rel.candidate_ids[c] for u, c in enumerate(cand)):
        raise RuntimeError("chunking does not reproduce the frozen units")
    n_vec = np.asarray([len(x) - 1 for s in range(0, len(texts), 2000)
                        for x in tok(texts[s:s + 2000], truncation=True, max_length=CF.DOC_MAX)["input_ids"]], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(n_vec)[:-1]])
    np.savez(out / "bge_chunks.npz", cand=np.asarray(cand, dtype=np.int64), n_vec=n_vec, offset=offset)
    stored_len = np.asarray([len(x) - 1 for x in tok([rel.queries[r].text for r in spec["ctl_rows"]], truncation=True,
                                                     max_length=CF.QUERY_MAX)["input_ids"]], dtype=np.int64)
    frozen_q = np.load(frozen_dir / "query_vectors.npy", mmap_mode="r")
    cls, tokens = [], []
    t1 = time.perf_counter()
    for text in spec["texts"]:
        box = {}
        enc.encode([text], CB.QUERY_MAX_TOKENS, 8192, lambda j, c, v: box.update(cls=c, vec=v))
        cls.append(np.asarray(box["cls"], dtype=np.float32))
        tokens.append(np.asarray(box["vec"], dtype=np.float16))
    enc_s = time.perf_counter() - t1
    lens = np.asarray([len(t) for t in tokens])
    n = spec["n_new"]
    cos = [float(np.dot(cls[n + i], frozen_q[r])) for i, r in enumerate(spec["ctl_rows"])]
    np.savez(out / "bge.npz", query_ids=np.asarray(spec["ids"]), cls=np.stack(cls), tokens=np.concatenate(tokens),
             tok_off=np.concatenate([[0], np.cumsum(lens)]))
    np.savez(out / "bge_cls.npz", query_ids=np.asarray(spec["ids"]), cls=np.stack(cls))     # the part the thesis PC needs
    return {"model": "bge-m3 + ColBERT head", "load_seconds": round(load_s, 2), "encode_seconds": round(enc_s, 2),
            "new_cases": n, "controls": len(spec["ctl_rows"]),
            "control_cls_cos": {"min": min(cos), "median": float(np.median(cos))},
            "control_token_counts_match": bool(np.array_equal(lens[n:], stored_len)), **gpu_info(torch)}


def run_qwen(d: Path, out: Path, prof) -> dict:
    import torch
    from uol.embedding_push import qwen_endpoint as qe
    from uol.final import qwen_local
    spec = inputs(d, prof)
    t0 = time.perf_counter()
    qwen_local.model()
    load_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    vec = np.stack([np.asarray(qwen_local.encode([qe.query_input("c", t)], batch=1)[0], dtype=np.float32)
                    for t in spec["texts"]])
    enc_s = time.perf_counter() - t1
    stored = qe.query_vectors("c")
    n = spec["n_new"]
    cos = np.asarray([float(np.dot(vec[n + i], stored[r])) for i, r in enumerate(spec["ctl_rows"])])
    np.savez(out / "qwen.npz", query_ids=np.asarray(spec["ids"]), vectors=vec)
    return {"model": "Qwen3-Embedding-8B int8 (uol.final.qwen_local)", "load_seconds": round(load_s, 2),
            "encode_seconds": round(enc_s, 2), "new_cases": n, "controls": len(spec["ctl_rows"]),
            "control_cos": {"mean": float(cos.mean()), "min": float(cos.min()), "median": float(np.median(cos))},
            **gpu_info(torch)}


def load_array(path: Path) -> np.ndarray:
    from uol.final import packing
    if not path.exists() and path.with_name(path.name + packing.SUFFIX).exists():
        packing.unpack(path.with_name(path.name + packing.SUFFIX))
    return np.load(path)


def maxp_scorer(ep):
    """``common.maxp_scores`` of the bge-m3 chunks for one query vector (``timing_sources.MaxP``)."""
    dd = ep.DERIVED / "bge_m3_chunks_v1"
    vecs = load_array(dd / "chunk_vectors.npy").astype(np.float32)
    owner = ep.chunk_article_index(dd / "units.jsonl")
    n = len(ep.release().candidate_ids)
    order = np.argsort(owner, kind="stable")
    vecs_t = np.ascontiguousarray(vecs[order].T)
    sorted_art = np.asarray(owner)[order]
    starts = np.flatnonzero(np.r_[True, sorted_art[1:] != sorted_art[:-1]])
    owners = sorted_art[starts]

    def f(q: np.ndarray) -> np.ndarray:
        sims = np.asarray(q, dtype=np.float32)[None, :] @ vecs_t
        o = np.full((1, n), -np.inf, dtype=np.float32)
        o[:, owners] = np.maximum.reduceat(sims, starts, axis=1)
        return o[0]
    return f


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic) if len(a) > 2 else 1.0


def run_pairs(d: Path, out: Path, prof) -> dict:
    import torch
    from uol.embedding_push import colbert as CB, common as ep
    from uol.facts_to_law import lexical
    torch.set_num_threads(int(os.environ.get("UOL_NEW_PAIR_THREADS", "16")))
    spec = inputs(d, prof)
    rel = ep.release()
    z = np.load(out / "bge.npz")
    if list(z["query_ids"]) != spec["ids"]:
        raise RuntimeError("bge.npz does not hold the staged cases in order")
    cls, toks, off = z["cls"], z["tokens"], z["tok_off"]
    t0 = time.perf_counter()
    bm25 = lexical.LexicalIndex(rel.candidate_texts, "bm25", {"k1": 1.2, "b": 0.75})
    maxp = maxp_scorer(ep)
    chunks = dict(np.load(out / "bge_chunks.npz"))
    index = CB.article_index(chunks, len(rel.candidate_ids))
    store = root_work(prof) / "colbert_full" / "store"
    docs = np.load(store / "doc_tokens.npy", mmap_mode="r")
    if int(chunks["n_vec"].sum()) != docs.shape[0]:
        raise RuntimeError("the chunk plan does not match the ColBERT token store")
    load_s = time.perf_counter() - t0
    qi, ci, sc = [], [], []
    t1 = time.perf_counter()
    for i, text in enumerate(spec["texts"]):
        bt = np.argsort(-bm25.score([text])[0], kind="stable")[:COLBERT_K]
        dt = np.argsort(-maxp(cls[i]), kind="stable")[:COLBERT_K]
        union = np.union1d(bt, dt)
        q = np.asarray(toks[off[i]:off[i + 1]], dtype=np.float32)
        m, _ = CB.score_query(q, np.ones((len(q), 1), dtype=np.float32), union, index, docs, 49152, "cpu")
        qi.append(np.full(len(union), i, dtype=np.int32))
        ci.append(union.astype(np.int32))
        sc.append(m[:, 0].astype(np.float32))
        if (i + 1) % 50 == 0:
            log(f"pairs: {i + 1}/{len(spec['texts'])} cases ({time.perf_counter() - t1:.0f}s)")
    score_s = time.perf_counter() - t1
    np.savez(out / "pairs.npz", query_ids=np.asarray(spec["ids"]), case=np.concatenate(qi), cidx=np.concatenate(ci),
             score=np.concatenate(sc))
    # controls against the frozen pairs (bundle rows)
    fz = np.load(ep.SCORES / "colbert_bge" / "pairs.npz")
    fq, fc, fs = np.asarray(fz["qidx"]), np.asarray(fz["cidx"]), np.asarray(fz["score"])
    n = spec["n_new"]
    same, jac, diffs, rhos = [], [], [], []
    for j, r in enumerate(spec["ctl_rows"]):
        m = fq == r
        frozen = {int(c): float(s) for c, s in zip(fc[m], fs[m])}
        got = {int(c): float(s) for c, s in zip(ci[n + j], sc[n + j])}
        same.append(set(frozen) == set(got))
        jac.append(len(set(frozen) & set(got)) / max(1, len(set(frozen) | set(got))))
        common_c = sorted(set(frozen) & set(got))
        a = np.asarray([got[c] for c in common_c])
        b = np.asarray([frozen[c] for c in common_c])
        diffs.append(np.abs(a - b))
        rhos.append(spearman(a, b))
    dd = np.concatenate(diffs) if diffs else np.zeros(0)
    return {"what": "ColBERT pair scores (bge-m3 late interaction, uniform MaxP) over the BM25 + bge-m3 pool",
            "load_seconds": round(load_s, 2), "score_seconds": round(score_s, 2), "new_cases": n,
            "controls": len(spec["ctl_rows"]), "pairs_new": int(sum(len(x) for x in ci[:n])),
            "control_pool_identical_share": float(np.mean(same)),
            "control_pool_jaccard": {"mean": float(np.mean(jac)), "min": float(np.min(jac))},
            "control_pairs": {"n": int(len(dd)), "mean_abs": float(dd.mean()) if len(dd) else None,
                              "max_abs": float(dd.max()) if len(dd) else None},
            "control_spearman": {"median": float(np.median(rhos)), "min": float(np.min(rhos))},
            "torch_threads": torch.get_num_threads(), "torch": torch.__version__}


def run_gate(d: Path, out: Path, prof) -> dict:
    b = json.loads((out / "bge.json").read_text(encoding="utf-8"))
    q = json.loads((out / "qwen.json").read_text(encoding="utf-8"))
    p = json.loads((out / "pairs.json").read_text(encoding="utf-8"))
    fails = []
    if b["control_cls_cos"]["min"] < GATE["bge_cls_cos_min"]:
        fails.append(f"bge CLS cos min {b['control_cls_cos']['min']:.7f} < {GATE['bge_cls_cos_min']}")
    if not b["control_token_counts_match"]:
        fails.append("bge query token counts differ from the store's")
    if q["control_cos"]["mean"] < GATE["qwen_cos_mean"] or q["control_cos"]["min"] < GATE["qwen_cos_min"]:
        fails.append(f"Qwen int8 cos mean {q['control_cos']['mean']:.4f} / min {q['control_cos']['min']:.4f}")
    pj = p["control_pool_jaccard"]
    if pj["mean"] < GATE["pool_jaccard_mean"] or pj["min"] < GATE["pool_jaccard_min"]:
        fails.append(f"ColBERT pool overlap (Jaccard) mean {pj['mean']:.4f} min {pj['min']:.4f}")
    cp, sp = p["control_pairs"], p["control_spearman"]
    if cp["mean_abs"] is None or cp["mean_abs"] > GATE["pairs_mean_abs"] or cp["max_abs"] > GATE["pairs_max_abs"]:
        fails.append(f"ColBERT pair scores differ: mean {cp['mean_abs']} max {cp['max_abs']}")
    if sp["median"] < GATE["spearman_median"] or sp["min"] < GATE["spearman_min"]:
        fails.append(f"ColBERT per-case Spearman median {sp['median']:.4f} min {sp['min']:.4f}")
    return {"gate": GATE, "pass": not fails, "fails": fails, "new_cases": b["new_cases"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bge", "qwen", "pairs", "gate"])
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from uol.final import context
    prof = context.active()
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / f"{a.model}.json").exists():
        raise FileExistsError(f"{a.out / (a.model + '.json')} exists: a finished step is not rerun")
    t0 = time.perf_counter()
    fn = {"bge": run_bge, "qwen": run_qwen, "pairs": run_pairs, "gate": run_gate}[a.model]
    meta = fn(a.dir, a.out, prof)
    meta.update(wall_seconds=round(time.perf_counter() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    name = "GATE" if a.model == "gate" else a.model
    (a.out / f"{name}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{a.model}: {json.dumps(meta)}")
    return 2 if a.model == "gate" and not meta["pass"] else 0


if __name__ == "__main__":
    sys.exit(main())
