"""GPU job ``colbert_full`` (D79, 22 Oct 2026): bge-m3 late interaction (ColBERT) scored alone over the whole catalogue.

The recipe is the frozen ``colbert_bge`` source (``uol.embedding_push.colbert``, manifest of 14 Oct), reused as code:

* model BAAI/bge-m3 5617a9f6 with its ``colbert_linear.pt`` head (the ``models`` job snapshot; every model file is
  checked against the hashes of the frozen manifest), fp32 forward (``colbert.Encoder``);
* token vectors ``colbert_linear(last_hidden_state[:, 1:])``: CLS dropped, ``</s>`` kept, padding removed, L2-normalised
  in fp32, stored float16;
* articles chunked by ``facts_to_law.bge_gpu.chunk_article`` (1,024 tokens, title repeat, overlap 64, at most 32
  chunks); the chunks must reproduce the 41,030 frozen units and every chunk's CLS vector the frozen
  ``bge_m3_chunks_v1`` vector (cos >= .999, as ``colbert.stage_encode_docs``); queries: the masked facts truncated at
  512 tokens, no prefix, CLS checked against the frozen query vectors;
* score = mean over the query vectors of the maximum dot product with a chunk's vectors (``colbert.late_interaction``,
  uniform weights), MaxP over an article's chunks: the PRIMARY ``score`` of ``colbert_bge``; fp32 scoring on the
  float16 vectors, TF32 off.

Run with the pinned GPU overlay (``rerank_bench.PINS``: transformers 5.17 / tokenizers 0.23.2): the venv's tokenizers
0.22.2 cuts LOAPF art. 27 into 4 chunks instead of the frozen 3; the overlay reproduces all 41,030 units (checked on
the thesis PC, 22 Oct). Only the plumbing differs from the frozen run: one job instead of staged shards, every article
instead of the BM25 / dense pool, and the GPU kernel scores all queries against blocks of chunks.

Fidelity gate: on the (query, article) pairs of the bundled ``colbert_bge/pairs.npz`` the new score must reproduce the
frozen ``score`` (mean |diff| <= 1e-4, max |diff| <= 2e-3, per-query Spearman median >= .999 and minimum >= .99); on a
failure no score file is written. Resumes re-check the stored vectors (the same rule as ``legal_encode``'s store check:
a sample re-encoded, cos .999; a damaged store is moved aside and rebuilt).

Outputs: ``<new_vectors>/bge_m3_colbert/scores_dev.npy`` (float32 [development rows, N], read by the ``dev_block``
adapter) with ``manifest.json``; ``<out>/run.json`` (encode / score seconds, peak VRAM, index bytes, fidelity, query
latency) for the cost record.

    python colbert_full.py --out <work>/gpu_jobs/colbert_full/outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "round3" / "gpu_jobs"))
import rerank_bench as RB  # noqa: E402

DOC_MAX, QUERY_MAX, OVERLAP, CAP, DIM = 1024, 512, 64, 32, 1024
DOC_BLOCK, QUERY_BLOCK = 8192, 2048          # vectors per GPU block: the [doc x query] product stays about 64 MB
GATE = {"mean_abs": 1e-4, "max_abs": 2e-3, "spearman_median": 0.999, "spearman_min": 0.99}
STORE_CHECK_CHUNKS, STORE_CHECK_MIN_COS = 64, 0.999
LATENCY_QUERIES = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def doc_blocks(n_vec: np.ndarray, max_vectors: int = DOC_BLOCK) -> list[tuple[int, int]]:
    """Consecutive chunk ranges [lo, hi) with at most ``max_vectors`` vectors (a longer chunk forms its own block)."""
    out, lo, used = [], 0, 0
    for i, n in enumerate(np.asarray(n_vec, dtype=np.int64)):
        if i > lo and used + n > max_vectors:
            out.append((lo, i))
            lo, used = i, 0
        used += int(n)
    if lo < len(n_vec):
        out.append((lo, len(n_vec)))
    return out


def query_blocks(n_vec: np.ndarray, max_vectors: int = QUERY_BLOCK) -> list[tuple[int, int]]:
    return doc_blocks(n_vec, max_vectors)


def chunk_query_scores(torch, D, Q, chunk_len, q_len):
    """[chunks, queries] uniform late-interaction scores of one doc block (``D`` [T, H], ``chunk_len`` [C]) and one
    query block (``Q`` [L, H], ``q_len`` [nq]): the per-chunk maximum of every query vector's dot products, then the
    mean over each query's vectors (``colbert.late_interaction`` with uniform weights)."""
    per_chunk = torch.segment_reduce(D @ Q.T, "max", lengths=chunk_len, axis=0)          # [C, L]
    owner = torch.repeat_interleave(torch.arange(len(q_len), device=Q.device), q_len)     # query of each vector
    M = torch.zeros((Q.shape[0], len(q_len)), dtype=per_chunk.dtype, device=Q.device)
    M[torch.arange(Q.shape[0], device=Q.device), owner] = 1.0 / q_len[owner].to(per_chunk.dtype)
    return per_chunk @ M                                                                   # [C, nq]


def article_maxp(chunk_scores: np.ndarray, chunk_article: np.ndarray, n_articles: int) -> np.ndarray:
    """[queries, N]: the maximum over each article's chunks (chunks of an article are consecutive)."""
    starts = np.flatnonzero(np.r_[True, np.diff(chunk_article) != 0])
    if len(starts) != n_articles or not (chunk_article[starts] == np.arange(n_articles)).all():
        raise ValueError("chunks are not consecutive per article in catalogue order")
    return np.maximum.reduceat(chunk_scores, starts, axis=0).T.astype(np.float32)


def fidelity(scores: np.ndarray, qidx: np.ndarray, cidx: np.ndarray, frozen: np.ndarray) -> dict:
    """The new scores on the frozen colbert_bge pairs: |diff| and per-query Spearman correlation; pass flag."""
    from scipy.stats import spearmanr
    new = scores[qidx, cidx].astype(np.float64)
    old = np.asarray(frozen, dtype=np.float64)
    diff = np.abs(new - old)
    rho = []
    for q in np.unique(qidx):
        sel = qidx == q
        if sel.sum() > 2:
            r = spearmanr(new[sel], old[sel]).correlation
            rho.append(1.0 if np.isnan(r) else float(r))
    rec = {"pairs": int(len(qidx)), "queries": int(len(rho)), "mean_abs": float(diff.mean()), "max_abs": float(diff.max()),
           "spearman_median": float(np.median(rho)) if rho else None, "spearman_min": float(np.min(rho)) if rho else None,
           "thresholds": GATE}
    rec["pass"] = bool(rho and rec["mean_abs"] <= GATE["mean_abs"] and rec["max_abs"] <= GATE["max_abs"]
                       and rec["spearman_median"] >= GATE["spearman_median"] and rec["spearman_min"] >= GATE["spearman_min"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    RB.deps(scratch, RB.PINS)                         # the pinned overlay tokenizer reproduces the frozen chunking
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    from uol.embedding_push import colbert as CB, common as ep
    from uol.facts_to_law import bge_gpu
    from uol.final import context, rebuild
    prof = context.active()
    root_work = prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work
    bge = root_work / "models" / rebuild.MODELS["BGE_DIR"][2]
    frozen_dir = prof.path("derived_ftl") / "bge_m3_chunks_v1"
    pairs_dir = prof.path("ep_run") / "scores" / "colbert_bge"
    man = json.loads((pairs_dir / "manifest.json").read_text(encoding="utf-8"))
    want = man["input_sha256"]["model_files"]
    got = {n: ep.sha256_file(bge / n) for n in want}
    if got != want:
        raise RuntimeError(f"model files differ from the frozen colbert_bge run: {[n for n in want if got[n] != want[n]]}")
    ep.BGE_DIR = bge
    t_all = time.perf_counter()
    d = ep.release()
    N = len(d.candidate_ids)
    store = root_work / "colbert_full" / "store"
    store.mkdir(parents=True, exist_ok=True)
    run: dict = {"recipe": "uol.embedding_push.colbert (frozen colbert_bge, 14 Oct) PRIMARY uniform_maxp",
                 "model_files_sha256": got, "stack": {"transformers": __import__("transformers").__version__,
                                                       "tokenizers": __import__("tokenizers").__version__,
                                                       "torch": torch.__version__}, "tf32": False}
    enc = CB.Encoder("cuda")
    tok = enc.tokenizer
    # ---- chunks of every article, checked against the frozen units
    rows = bge_gpu.read_jsonl(Path(d.candidate_binding["articles_path"]))
    units = [json.loads(x) for x in (frozen_dir / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    texts, cand = [], []
    for i, r in enumerate(rows):
        if r["candidate_id"] != d.candidate_ids[i]:
            raise ValueError("catalogue order differs from the release candidate order")
        ch = bge_gpu.chunk_article(r, tok, DOC_MAX, OVERLAP, CAP)
        texts += ch
        cand += [i] * len(ch)
    if len(texts) != len(units) or any(units[u]["candidate_id"] != d.candidate_ids[c] for u, c in enumerate(cand)):
        raise RuntimeError(f"chunking does not reproduce the frozen units ({len(texts)} vs {len(units)})")
    cand = np.asarray(cand, dtype=np.int64)
    n_vec = np.asarray([len(x) - 1 for s in range(0, len(texts), 2000)
                        for x in tok(texts[s:s + 2000], truncation=True, max_length=DOC_MAX)["input_ids"]], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(n_vec)[:-1]])
    qtexts = [q.text for q in d.queries]
    q_vec = np.asarray([len(x) - 1 for x in tok(qtexts, truncation=True, max_length=QUERY_MAX)["input_ids"]], dtype=np.int64)
    q_off = np.concatenate([[0], np.cumsum(q_vec)[:-1]])
    index_bytes = int(n_vec.sum()) * DIM * 2
    run.update(chunks=len(texts), doc_vectors=int(n_vec.sum()), queries=len(qtexts), query_vectors=int(q_vec.sum()),
               index_bytes=index_bytes, index_gb=round(index_bytes / 1e9, 2))
    log(f"{len(texts)} chunks, {int(n_vec.sum())} doc vectors ({index_bytes / 1e9:.1f} GB fp16), {len(qtexts)} queries")
    # ---- stores (resumable): docs by shard, queries whole; a resumed store is checked first
    shards = doc_blocks(n_vec, max(1, int(n_vec.sum()) // 8 + 1))
    prog_p = store / "progress.json"
    meta = {"n_vec_sha": __import__("hashlib").sha256(n_vec.tobytes()).hexdigest(), "shards": shards}
    prog = json.loads(prog_p.read_text(encoding="utf-8")) if prog_p.exists() else {}
    if prog and prog.get("meta") != meta:
        raise RuntimeError(f"the store at {store} belongs to another chunk plan")
    docs_p, qs_p = store / "doc_tokens.npy", store / "query_tokens.npy"
    if not prog:
        np.lib.format.open_memmap(docs_p, mode="w+", dtype=np.float16, shape=(int(n_vec.sum()), DIM)).flush()
        prog = {"meta": meta, "done": [], "queries": False}
        prog_p.write_text(json.dumps(prog), encoding="utf-8")
    frozen_cls = np.load(frozen_dir / "chunk_vectors.npy", mmap_mode="r")
    docs = np.load(docs_p, mmap_mode="r+")

    def encode_chunks(idx: list[int], sink):
        enc.encode([texts[i] for i in idx], DOC_MAX, 8192, sink, lengths=n_vec[idx] + 1)

    if prog["done"]:                                   # store check of the resumed shards
        done_chunks = [i for s in prog["done"] for i in range(*shards[s])]
        sample = done_chunks[:: max(1, len(done_chunks) // STORE_CHECK_CHUNKS)]
        cos = []

        def check(j, cls, vec, sample=sample):
            i = sample[j]
            old = np.asarray(docs[offset[i]:offset[i] + n_vec[i]], dtype=np.float32)
            cos.append(float((old * vec.astype(np.float32)).sum(1).min()))
        encode_chunks(sample, check)
        run["store_check"] = {"resumed_shards": prog["done"], "n": len(sample), "min_cos": min(cos),
                              "threshold": STORE_CHECK_MIN_COS}
        if min(cos) < STORE_CHECK_MIN_COS:
            del docs
            aside = store.with_name(f"store.bad-{time.strftime('%Y%m%dT%H%M%S')}")
            store.rename(aside)
            run["store_check"].update(rebuilt=True, moved_aside=str(aside))
            store.mkdir(parents=True)
            np.lib.format.open_memmap(docs_p, mode="w+", dtype=np.float16, shape=(int(n_vec.sum()), DIM)).flush()
            prog = {"meta": meta, "done": [], "queries": False}
            prog_p.write_text(json.dumps(prog), encoding="utf-8")
            docs = np.load(docs_p, mmap_mode="r+")
        else:
            run["store_check"]["rebuilt"] = False
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    cls_min = 1.0
    for s, (lo, hi) in enumerate(shards):
        if s in prog["done"]:
            continue
        idx = list(range(lo, hi))
        mins = []

        def sink(j, cls, vec, idx=idx, mins=mins):
            i = idx[j]
            docs[offset[i]:offset[i] + n_vec[i]] = vec
            mins.append(float(cls @ frozen_cls[i]))
        encode_chunks(idx, sink)
        docs.flush()
        if min(mins) < 0.999:
            raise RuntimeError(f"shard {s}: CLS vectors do not reproduce the frozen bge_m3_chunks_v1 (min cos {min(mins)})")
        cls_min = min(cls_min, min(mins))
        prog["done"].append(s)
        prog_p.write_text(json.dumps(prog), encoding="utf-8")
        log(f"shard {s + 1}/{len(shards)} encoded ({time.perf_counter() - t0:.0f}s)")
    run["encode_docs_seconds"] = round(time.perf_counter() - t0, 1)
    run["doc_cls_min_cos"] = cls_min
    t1 = time.perf_counter()
    if not prog["queries"] or not qs_p.exists():
        qs = np.lib.format.open_memmap(qs_p, mode="w+", dtype=np.float16, shape=(int(q_vec.sum()), DIM))
        frozen_q = np.load(frozen_dir / "query_vectors.npy", mmap_mode="r")
        qmins = []

        def qsink(i, cls, vec):
            qs[q_off[i]:q_off[i] + q_vec[i]] = vec
            qmins.append(float(cls @ frozen_q[i]))
        enc.encode(qtexts, QUERY_MAX, 8192, qsink, lengths=q_vec + 1)
        qs.flush()
        del qs
        if min(qmins) < 0.999:
            raise RuntimeError(f"query CLS vectors do not reproduce the frozen vectors (min cos {min(qmins)})")
        prog["queries"] = True
        prog_p.write_text(json.dumps(prog), encoding="utf-8")
        run["query_cls_min_cos"] = min(qmins)
    run["encode_queries_seconds"] = round(time.perf_counter() - t1, 1)
    run["encode_seconds"] = round(run["encode_docs_seconds"] + run["encode_queries_seconds"], 1)
    # ---- scoring: every query against every chunk on the GPU, then MaxP over an article's chunks
    del docs
    docs = np.load(docs_p, mmap_mode="r")
    qs = np.load(qs_p, mmap_mode="r")
    dev = torch.device("cuda")
    Q = torch.from_numpy(np.ascontiguousarray(qs)).to(dev).float()            # fp16 store, fp32 scoring (as frozen)
    qb = query_blocks(q_vec)
    qlen = torch.from_numpy(q_vec).to(dev)
    CS = torch.empty((len(texts), len(qtexts)), dtype=torch.float32, device=dev)
    t2 = time.perf_counter()
    blocks = doc_blocks(n_vec)
    for b, (lo, hi) in enumerate(blocks):
        D = torch.from_numpy(np.ascontiguousarray(docs[offset[lo]:offset[hi - 1] + n_vec[hi - 1]])).to(dev).float()
        clen = torch.from_numpy(n_vec[lo:hi]).to(dev)
        for qlo, qhi in qb:
            Qb = Q[int(q_off[qlo]):int(q_off[qhi - 1] + q_vec[qhi - 1])]
            CS[lo:hi, qlo:qhi] = chunk_query_scores(torch, D, Qb, clen, qlen[qlo:qhi])
        if b in (0, 9) or (b + 1) % 100 == 0:
            torch.cuda.synchronize()
            el = time.perf_counter() - t2
            log(f"score: block {b + 1}/{len(blocks)} {el:.0f}s, projected {el / (b + 1) * len(blocks) / 60:.1f} min")
    torch.cuda.synchronize()
    run["score_seconds"] = round(time.perf_counter() - t2, 1)
    S = article_maxp(CS.cpu().numpy(), cand, N)
    del CS, Q
    run["peak_vram_mib"] = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    # ---- fidelity gate on the frozen pairs
    z = np.load(pairs_dir / "pairs.npz")
    fid = fidelity(S, z["qidx"].astype(np.int64), z["cidx"].astype(np.int64), z["score"])
    run["fidelity"] = fid
    a.out.mkdir(parents=True, exist_ok=True)
    if not fid["pass"]:
        (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
        raise RuntimeError(f"ColBERT full scores do not reproduce the frozen colbert_bge pairs: {fid}")
    # ---- query latency: one query at a time (encode + scan of the whole index streamed from host memory)
    lat = []
    for i in range(min(LATENCY_QUERIES, len(qtexts))):
        torch.cuda.synchronize()
        t = time.perf_counter()
        holder = {}
        enc.encode([qtexts[i]], QUERY_MAX, 8192, lambda j, cls, vec: holder.update(v=vec), lengths=q_vec[i:i + 1] + 1)
        q1 = torch.from_numpy(holder["v"].astype(np.float32)).to(dev)
        best = torch.full((len(texts),), -1.0, device=dev)
        for lo, hi in blocks:
            D = torch.from_numpy(np.ascontiguousarray(docs[offset[lo]:offset[hi - 1] + n_vec[hi - 1]])).to(dev).float()
            best[lo:hi] = chunk_query_scores(torch, D, q1, torch.from_numpy(n_vec[lo:hi]).to(dev),
                                             torch.tensor([q1.shape[0]], device=dev))[:, 0]
        torch.cuda.synchronize()
        lat.append(1000 * (time.perf_counter() - t))
    run["query_latency"] = {"device": torch.cuda.get_device_name(0), "n_cases": len(lat), "median_ms": float(np.median(lat)),
                            "p95_ms": float(np.percentile(lat, 95)),
                            "note": "query encode + MaxSim over the whole fp16 index streamed from host memory to the GPU"}
    # ---- outputs
    dest = prof.path("new_vectors") / "bge_m3_colbert"
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest / "scores_dev.tmp.npy"
    np.save(tmp, S)
    tmp.replace(dest / "scores_dev.npy")
    m = {"method": "bge_m3_colbert", "family": "EMB", "rows": "development rows of the bundle release, in order",
         "recipe": run["recipe"], "fidelity": fid, "scores_sha256": ep.sha256_file(dest / "scores_dev.npy"),
         "index_gb": run["index_gb"], "label_free": True, "decision": "D79 (22 Oct 2026, added after the freeze)"}
    (dest / "manifest.json").write_text(json.dumps(m, indent=1), encoding="utf-8")
    run["seconds_total"] = round(time.perf_counter() - t_all, 1)
    run["device"] = torch.cuda.get_device_name(0)
    (a.out / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    log(f"done: fidelity {fid}, {run['seconds_total']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
