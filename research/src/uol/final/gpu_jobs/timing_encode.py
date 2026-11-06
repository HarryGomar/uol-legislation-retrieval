"""Query encoders of the whole-method timing (``uol.final.timing``, D137), one case at a time on the GPU.

One model per process, because the pinned stacks differ (``rerank_bench.deps``):

* ``bge``: bge-m3 with its ColBERT head (``embedding_push.colbert.Encoder``, the frozen query recipe: the masked facts
  truncated at 512 tokens), giving the CLS vector of the dense bge sources and the token vectors of ``colbert_bge``
  in one forward pass. It also writes the chunk plan of the run computer's token store (``colbert_full``'s chunking
  of every article, checked against the frozen units) and the stored query-token offsets: ``bge_chunks.npz``.
* ``qwen``: Qwen3-Embedding-8B int8 (``uol.final.qwen_local``, the fidelity-gated local path), input
  ``qwen_endpoint.query_input("c", text)``.
* ``e5proxy``: the fine-tuned e5 query tower (``ft_e5_q``, 512 tokens, "query: ") and the case-similarity e5 encoder
  (256 tokens) have the e5-small architecture; their fold-0 weights were not kept, so the base e5-small model runs at
  the same input lengths as a stand-in for their per-case encode time (their vectors are the stored outputs).

Per case: tokenisation, the forward pass and the copy to host, timed with ``torch.cuda.synchronize`` around it;
model loading is timed apart. The warm-up cases are encoded first (they also warm the CUDA kernels and the allocator)
and saved with the timed cases, because the run scores them in both passes; ``warmup`` marks their entries, whose
seconds enter no statistic (T2, 26 Oct: the first job saved only the timed rows, and the fresh warm-up pass found no
query tokens). Fidelity against the stored vectors (cosine) is recorded for the timed cases. Outputs
``<out>/<model>.npz`` (rows, warmup, vectors or token vectors, seconds) and ``<out>/<model>.json`` (load seconds, GPU,
peak memory, fidelity).

    python timing_encode.py --model bge|qwen|e5proxy --rows <out>/rows.json --out <out>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import uol  # noqa: E402  (the installed code tree; this file may sit in a separate overlay folder)

sys.path.insert(0, str(Path(uol.__file__).resolve().parent / "round3" / "gpu_jobs"))
import rerank_bench as RB  # noqa: E402



def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def scratch_dir(prof) -> Path:
    return Path(os.environ.get("UOL_WORKER_SCRATCH", prof.work / "gpu_scratch"))


def timed(torch, fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def encode_cases(rows: list[int], warm: list[int], one) -> list[tuple[int, bool, object]]:
    """[(row, warm-up flag, ``one(row)``)]: the warm-up rows first, then the timed rows. A row in both lists (a small
    population, ``timing.case_rows``) is encoded once for each, so each pass finds its own entry."""
    return [(int(r), flag, one(int(r))) for flag, rs in ((True, warm), (False, rows)) for r in rs]


def row_arrays(recs: list[tuple]) -> dict[str, np.ndarray]:
    return {"rows": np.asarray([x[0] for x in recs], dtype=np.int64), "warmup": np.asarray([x[1] for x in recs], dtype=bool)}


def gpu_info(torch) -> dict:
    return {"gpu": torch.cuda.get_device_name(0), "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
            "torch": torch.__version__, "cuda": torch.version.cuda}


def run_bge(rows: list[int], warm: list[int], out: Path, prof) -> dict:
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
    root_work = prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work
    ep.BGE_DIR = root_work / "models" / rebuild.MODELS["BGE_DIR"][2]
    d = ep.release()
    t0 = time.perf_counter()
    enc = CB.Encoder("cuda")
    load_s = time.perf_counter() - t0
    tok = enc.tokenizer
    # chunk plan of the token store, exactly as colbert_full builds it, and the stored query offsets
    frozen_dir = prof.path("derived_ftl") / "bge_m3_chunks_v1"
    arts = bge_gpu.read_jsonl(Path(d.candidate_binding["articles_path"]))
    units = [json.loads(x) for x in (frozen_dir / "units.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    texts, cand = [], []
    for i, r in enumerate(arts):
        ch = bge_gpu.chunk_article(r, tok, CF.DOC_MAX, CF.OVERLAP, CF.CAP)
        texts += ch
        cand += [i] * len(ch)
    if len(texts) != len(units) or any(units[u]["candidate_id"] != d.candidate_ids[c] for u, c in enumerate(cand)):
        raise RuntimeError("chunking does not reproduce the frozen units")
    n_vec = np.asarray([len(x) - 1 for s in range(0, len(texts), 2000)
                        for x in tok(texts[s:s + 2000], truncation=True, max_length=CF.DOC_MAX)["input_ids"]], dtype=np.int64)
    offset = np.concatenate([[0], np.cumsum(n_vec)[:-1]])
    qtexts = [q.text for q in d.queries]
    q_vec = np.asarray([len(x) - 1 for x in tok(qtexts, truncation=True, max_length=CF.QUERY_MAX)["input_ids"]], dtype=np.int64)
    q_off = np.concatenate([[0], np.cumsum(q_vec)])
    np.savez(out / "bge_chunks.npz", cand=np.asarray(cand, dtype=np.int64), n_vec=n_vec, offset=offset, q_off=q_off)
    frozen_q = np.load(frozen_dir / "query_vectors.npy", mmap_mode="r")

    def one(row: int):
        box = {}
        enc.encode([qtexts[row]], CB.QUERY_MAX_TOKENS, 8192, lambda j, cls, vec: box.update(cls=cls, vec=vec))
        return box

    recs = encode_cases(rows, warm, lambda r: timed(torch, lambda: one(r)))
    tokens = [np.asarray(box["vec"], dtype=np.float16) for _, _, (box, _) in recs]
    lens = np.asarray([len(t) for t in tokens])
    idx = row_arrays(recs)
    cos = [float(np.dot(box["cls"], frozen_q[r])) for r, w, (box, _) in recs if not w]
    np.savez(out / "bge.npz", **idx, vectors=np.stack([box["cls"] for _, _, (box, _) in recs]),
             tokens=np.concatenate(tokens), tok_off=np.concatenate([[0], np.cumsum(lens)]),
             seconds=np.asarray([s for _, _, (_, s) in recs]))
    return {"model": "bge-m3 + ColBERT head", "load_seconds": round(load_s, 2), "cases": len(rows),
            "warmup_cases": len(warm), "cls_cos_vs_stored": {"min": min(cos), "median": float(np.median(cos))},
            "token_counts_match_store": bool(np.array_equal(lens, q_vec[idx["rows"]])), **gpu_info(torch)}


def run_qwen(rows: list[int], warm: list[int], out: Path, prof) -> dict:
    import torch
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.final import qwen_local
    d = ep.release()
    t0 = time.perf_counter()
    qwen_local.model()
    load_s = time.perf_counter() - t0
    texts = {int(r): qe.query_input("c", d.queries[int(r)].text) for r in list(rows) + list(warm)}
    recs = encode_cases(rows, warm, lambda r: timed(torch, lambda: qwen_local.encode([texts[r]], batch=1)[0]))
    stored = qe.query_vectors("c")
    cos = [float(np.dot(v, stored[r])) for r, w, (v, _) in recs if not w]
    np.savez(out / "qwen.npz", **row_arrays(recs), vectors=np.stack([v for _, _, (v, _) in recs]),
             seconds=np.asarray([s for _, _, (_, s) in recs]))
    return {"model": "Qwen3-Embedding-8B int8 (uol.final.qwen_local)", "load_seconds": round(load_s, 2), "cases": len(rows),
            "warmup_cases": len(warm), "cos_vs_stored": {"min": min(cos), "median": float(np.median(cos))},
            **gpu_info(torch)}


def run_e5proxy(rows: list[int], warm: list[int], out: Path, prof) -> dict:
    scratch = scratch_dir(prof)
    RB.deps(scratch, RB.PINS)
    import torch
    from transformers import AutoModel, AutoTokenizer
    from uol.embedding_push import common as ep
    from uol.final import rebuild
    root_work = prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work
    path = root_work / "models" / rebuild.MODELS["E5_DIR"][2]
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, local_files_only=True).to("cuda").eval()
    load_s = time.perf_counter() - t0
    d = ep.release()

    def one(text: str, max_len: int):
        with torch.inference_mode():
            b = tok([text], truncation=True, max_length=max_len, return_tensors="pt").to("cuda")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            return torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()

    def both(r: int) -> tuple[float, float]:
        text = d.queries[r].text
        return timed(torch, lambda: one("query: " + text, 512))[1], timed(torch, lambda: one(text, 256))[1]

    recs = encode_cases(rows, warm, both)
    ft, cs = np.asarray([x[2][0] for x in recs]), np.asarray([x[2][1] for x in recs])
    np.savez(out / "e5proxy.npz", **row_arrays(recs), seconds=ft + cs, ft_seconds=ft, casesim_seconds=cs)
    return {"model": "e5-small base as a stand-in for the fold-0 ft_e5_q tower (512 tokens) and the case-similarity "
                     "encoder (256 tokens); their weights were not kept", "load_seconds": round(load_s, 2),
            "cases": len(rows), "warmup_cases": len(warm), **gpu_info(torch)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["bge", "qwen", "e5proxy"])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from uol.final import context
    prof = context.active()
    spec = json.loads(a.rows.read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fn = {"bge": run_bge, "qwen": run_qwen, "e5proxy": run_e5proxy}[a.model]
    meta = fn(spec["timed"], spec["warmup"], a.out, prof)
    meta.update(wall_seconds=round(time.perf_counter() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (a.out / f"{a.model}.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"{a.model}: {meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
