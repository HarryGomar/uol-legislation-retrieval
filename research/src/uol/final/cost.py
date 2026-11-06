"""Cost record (PLAN_DRAFT §6) and timing utilities; one ``<store>/cost/<system>.json`` per system.

Fields (units): indexing one-off (GPU-hours on the 5080 or endpoint seconds; CPU-minutes), training (CPU-minutes per
fold; GPU-hours for fine-tunes), query latency (median and p95 ms per case on CPU at a fixed thread count, over 200
DEV cases, scored one case at a time; query encoding on CPU / 5080 when measured), memory (peak RAM, peak VRAM),
index size (MB), LLM (input / cached / output tokens and seconds per case), update after a reform (seconds to
re-index one decree), and cost per 1,000 cases (USD only at a cited, dated public price; otherwise left empty and
the physical quantities stand). Unknown values are ``null``, never guessed.

Where each quantity comes from (22 Oct 2026, measurement plumbing only; nothing scored changes):

* ``index_size_mb`` / ``index_size``: the GPU job's ``index_bytes`` (late interaction) or, from the fitted scorer of
  the latency measurement, the article-side vector files on disk (dense), the pickled fitted model (topic), the
  in-memory sparse matrices (lexical: no index file exists) or the sum of the members (fusion). Compressed vectors
  (Matryoshka / int8 / binary) are compressed in memory from fp32 files, so their size is analytic and says so.
* ``query_encoding``: seconds per encoded query from jobs that encode queries only (the int8 query encodes, DEV-N's
  query re-encodes, the late-interaction job's query loop); other GPU jobs time articles and queries in one loop, so
  the query side is reported as not available (no new GPU timing is added).
* ``query_latency.precomputed_sources`` (learned rankers): the source score files and members whose scores are
  computed before the timed per-case call, so the full per-case cost can be composed later.
* ``training.gpu_hours_finetune``: the fine-tune jobs' own timers (``bge_m3_ft`` run.json, ``r2_train`` train.json).
* ``llm``: ``uol.final.llm_cost`` (the job ``cost_summary``).
"""
from __future__ import annotations

import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from uol.final import context, guard, harness, populations, registry

LATENCY_CASES = 200
LATENCY_THREADS = 8
FIELDS = ("indexing", "training", "query_latency", "query_encoding", "memory", "index_size_mb", "llm",
          "update_after_reform_seconds", "cost_per_1000_cases")
PRICES: dict[str, dict[str, Any]] = {}      # filled only with cited, dated public prices (PLAN §6); none yet
MB = 2 ** 20


@contextmanager
def timer(record: dict[str, float], key: str) -> Iterator[None]:
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        yield
    finally:
        record[f"{key}_wall_s"] = record.get(f"{key}_wall_s", 0.0) + time.perf_counter() - t0
        record[f"{key}_cpu_s"] = record.get(f"{key}_cpu_s", 0.0) + time.process_time() - c0


def set_threads(n: int) -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "UOL_FINAL_THREADS"):
        os.environ[k] = str(n)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(n)
    except Exception:                                   # pragma: no cover
        pass


def latency(system: str, population: str = "DEV", n: int = LATENCY_CASES, threads: int = LATENCY_THREADS,
            fold: int = 0) -> dict[str, Any]:
    """Per-case scoring latency of a fitted system (fold ``fold`` model), one case at a time, CPU, fixed threads."""
    set_threads(threads)
    prof = context.active()
    pop = populations.load(population, prof.store)
    sc = registry.make(system)
    sc.prepare(pop)
    f = next(x for x in pop.folds() if x.k == fold)
    t0 = time.perf_counter()
    sc.fit(f)
    fit_s = time.perf_counter() - t0
    rows = guard.dev_rows_only(f.score_rows[:n])
    ms = []
    for r in rows:
        t = time.perf_counter()
        sc.score(np.asarray([r]))
        ms.append(1000 * (time.perf_counter() - t))
    ms_a = np.asarray(ms)
    rec = {"system": system, "threads": threads, "n_cases": int(len(ms)), "median_ms": float(np.median(ms_a)),
           "p95_ms": float(np.percentile(ms_a, 95)), "fit_seconds_fold": round(fit_s, 3),
           "note": "includes the system's own query-side computation on CPU; cached vectors exclude query encoding"}
    # read after the timed loop from the fitted scorer; the measurement above is unchanged (22 Oct 2026)
    pre = _safe(precomputed_sources, sc)
    if pre is not None:
        rec["precomputed_sources"] = pre
    rec["index_size"] = _safe(index_size, sc)
    return rec


def _safe(fn, *args) -> Any:
    try:
        return fn(*args)
    except Exception as exc:                            # a cost side field never fails the latency record
        return {"mb": None, "basis": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}


# --------------------------------------------------------------------------- learned rankers: precomputed sources


def precomputed_sources(sc) -> dict[str, Any] | None:
    """Learned rankers (``adapters.ltr.LTRRefit``): what is computed before the timed per-case ``score`` call.

    ``fit`` scores every development row of the pair table (the frozen source score files of the variant, the extra
    members' columns, the M1 memory columns) and ``score`` returns the stored row, so the per-case latency excludes
    these sources; their own per-case costs are composed later."""
    from uol.final.adapters import ltr
    if not isinstance(sc, ltr.LTRRefit):
        return None
    v = sc.h.v
    return {"frozen_source_scores": list(dict.fromkeys(list(v.pool_ids) + list(v.feature_ids))),
            "pool_sources": list(v.pool_ids), "feature_sources": list(v.feature_ids),
            "below_pool_order": "L_star", "added_sources": list(getattr(sc, "added_sources", []) or []),
            "dropped_sources": list(getattr(sc, "dropped_sources", []) or []),
            "extra_members_scored_in_fit": [m.sid for m in getattr(sc, "extra", [])],
            "extension_columns_in_fit": getattr(sc, "extension", None),
            "note": "score() returns rows computed in fit(); median_ms / p95_ms exclude computing these sources' scores"}


# --------------------------------------------------------------------------- index size


def _files(paths: Sequence[Path]) -> tuple[int, list[dict[str, Any]]]:
    out = [{"path": str(p), "bytes": int(Path(p).stat().st_size)} for p in paths if Path(p).exists()]
    return sum(x["bytes"] for x in out), out


def _sparse_bytes(m) -> int:
    return int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)


def index_size(sc) -> dict[str, Any]:
    """Index size of a prepared and fitted scorer; ``basis`` says how it was obtained."""
    from uol.final.adapters import dense, fusion, lexical, loaded, ltr, topic
    if isinstance(sc, (dense.DenseVectors, dense.FoldDense)) and getattr(sc, "vs", None) is not None:
        vs = sc.vs
        b, files = _files(getattr(vs, "article_files", []))
        a = sc.spec.get("adapter_args", {})
        rec = {"mb": round(b / MB, 2) if files else None, "bytes": b if files else None, "basis": "measured",
               "what": "article-side vectors and their owner index on disk" +
                       (" (fold model of the latency fold)" if isinstance(sc, dense.FoldDense) else ""), "files": files}
        if a.get("mrl_dim") or a.get("quant"):
            n, dim = (int(x) for x in vs.A.shape)                    # after the in-memory compression
            q = a.get("quant")
            per = {"int8": dim + 4, "binary": -(-dim // 8)}.get(q, dim * 4)
            owner = int(np.asarray(vs.owner).nbytes)
            rec = {"mb": round((n * per + owner) / MB, 2), "bytes": n * per + owner, "basis": "analytic",
                   "formula": f"{n} vectors x {per} bytes ({q or 'fp32'}, dim {dim}"
                              f"{'; int8 values + one fp32 scale per vector' if q == 'int8' else ''}) + owner index {owner} bytes",
                   "stored_file_mb": rec["mb"], "stored_files": files,
                   "note": "the run keeps fp32 vector files and compresses them in memory (adapters.dense.compression)"}
        return rec
    if isinstance(sc, (lexical.LStar, lexical.BM25)):
        parts = {}
        for kind, idx in getattr(sc.tc, "_lex", {}).items():
            if isinstance(sc, lexical.BM25) and kind != sc.kind:
                continue
            m = getattr(idx, "_documents_t", None)
            m = m if m is not None else getattr(idx, "_counts", None)
            if m is not None:
                parts[kind] = _sparse_bytes(m)
        return {"mb": round(sum(parts.values()) / MB, 2) if parts else None,
                "bytes": sum(parts.values()) if parts else None, "basis": "in_memory",
                "what": "sparse document-term matrices built in prepare (no index file on disk; vocabulary excluded)",
                "parts_bytes": parts}
    if isinstance(sc, topic.TopicScorer) and sc.params:
        _, key = topic.model_key(sc.kind, topic.param_value(sc.kind, sc.params), sc.articles, sc.extra)
        b, files = _files([topic._cache_dir() / f"{key}.pkl"])
        return {"mb": round(b / MB, 2) if files else None, "bytes": b if files else None,
                "basis": "measured" if files else "missing",
                "what": "pickled fitted model of the latency fold (vocabulary, factor matrices, article vectors)",
                "files": files}
    if type(sc) is loaded.FrozenScores:
        src = str(sc.source)
        prof = context.active()
        key = src[len("dense_"):] if src.startswith("dense_") else None
        if key and key in prof.paths and (prof.path(key) / "article_vectors.npy").exists():
            b, files = _files([prof.path(key) / "article_vectors.npy", prof.path(key) / "article_index.npy"])
            return {"mb": round(b / MB, 2), "bytes": b, "basis": "measured",
                    "what": f"article-side vectors of {key}, from which the frozen {src} scores were computed",
                    "files": files}
        return {"mb": None, "basis": "not_applicable", "reason": "frozen score file; no index is queried at run time"}
    if isinstance(sc, fusion.Fusion):
        parts = {m.sid: index_size(m) for m in sc.members}
        known = all(p.get("bytes") is not None for p in parts.values())
        b = sum(p["bytes"] for p in parts.values()) if known else None
        return {"mb": round(b / MB, 2) if known else None, "bytes": b, "basis": "sum_of_members", "members": parts}
    if isinstance(sc, ltr.LTRRefit):
        return {"mb": None, "basis": "composed",
                "reason": "learned ranker over precomputed source scores; compose from query_latency.precomputed_sources"}
    return {"mb": None, "basis": "unknown", "reason": f"no index measurement for {type(sc).__name__}"}


# --------------------------------------------------------------------------- GPU job records


def _job_dir_matches(name: str, job: str) -> bool:
    """A GPU job folder of ``job``: the job itself or its per-fold models (``<job>_f<k>``, ``<job>_f<k>_s<n>``); never
    the smoke twins or other jobs that share a prefix (``qwen3_8b`` is not ``qwen3_8b_gate``; 22 Oct 2026)."""
    return name == job or re.fullmatch(rf"{re.escape(job)}_f\d+(_s\d+)?", name) is not None


def _gpu_runs(system: str, name: str = "run.json") -> list[dict[str, Any]]:
    out = []
    root = context.active().work / "gpu_jobs"
    for p in sorted(root.glob(f"{system}*/outputs/{name}")):
        if _job_dir_matches(p.parent.parent.name, system):
            out.append(json.loads(p.read_text(encoding="utf-8")))
    return out


def _query_only(g: dict[str, Any]) -> bool:
    return (g.get("task") or {}).get("subset") in ("query_only", "devn")


def _per_query(g: dict[str, Any]) -> tuple[float, int] | None:
    n = g.get("encoded_now", g.get("queries"))
    if not g.get("encode_seconds") or not n or g.get("incomplete"):
        return None
    return float(g["encode_seconds"]) / int(n), int(n)


def query_encoding(spec: dict[str, Any]) -> dict[str, Any]:
    """Query-encoding seconds per case, only where a job already timed the query side on its own."""
    from uol.final.adapters import gpu as G
    a = spec.get("adapter_args", {})
    work = context.active().work
    if spec["adapter"] == "dev_block" and a.get("job"):
        for g in _gpu_runs(a["job"]):
            if g.get("encode_queries_seconds") and g.get("queries"):
                return {"available": True, "seconds_per_case": round(float(g["encode_queries_seconds"]) / int(g["queries"]), 5),
                        "n_encoded": int(g["queries"]), "encode_seconds": g["encode_queries_seconds"],
                        "device": g.get("device"), "basis": f"query loop of the GPU job {a['job']} (encode_queries_seconds)",
                        "note": "batched throughput (seconds per query in a batched job), not single-query latency"}
    key = a.get("key") if spec["adapter"] in ("dense_vectors", "fold_dense") else None
    if spec["adapter"] == "frozen_scores" and str(a.get("source", "")).startswith("dense_"):
        key = str(a["source"])[len("dense_"):]
    if key:
        cands = [(work / "gpu_jobs" / key / "outputs" / "run.json", f"query-only encode of every DEV query (job gpu__{key})"),
                 (work / "variants" / "devn" / "gpu_jobs" / f"devn__{key}" / "outputs" / "run.json",
                  f"DEV-N query-only re-encode (job gpu__devn__{key}: the DEV-N queries and 50 controls)")]
        for p, basis in cands:
            if not p.exists():
                continue
            g = json.loads(p.read_text(encoding="utf-8"))
            pq = _per_query(g) if _query_only(g) else None
            if pq:
                return {"available": True, "seconds_per_case": round(pq[0], 5), "n_encoded": pq[1],
                        "encode_seconds": g["encode_seconds"], "device": g.get("device"),
                        "encoder_spec": G.DEVN_ENCODERS.get(key) if "devn" in p.parts else key, "basis": basis,
                        "note": "batched throughput (seconds per encoded query string), not single-query latency"}
    if spec["adapter"] in ("dense_vectors", "fold_dense", "frozen_scores", "dev_block"):
        return {"available": False, "reason": "no job timed this encoder's queries on their own (GPU encode jobs time "
                "articles and queries in one loop: legal_encode encode_seconds); no new GPU timing is added"}
    if spec["adapter"] == "ltr":
        return {"available": False, "reason": "learned ranker: see query_latency.precomputed_sources"}
    if spec["adapter"] == "llm_matrix":
        return {"available": False, "reason": "LLM or reranker stage over a saved matrix: see llm / training"}
    return {"available": False, "reason": "the query side is computed on CPU inside query_latency (or not applicable)"}


def _finetune(job: str) -> dict[str, Any]:
    """GPU-hours of fine-tunes from the fine-tune jobs' own timers (one entry per fold model)."""
    runs = [g for g in _gpu_runs(job) if "steps" in g and g.get("seconds") is not None]        # bge_m3_ft.py
    trains = [g for g in _gpu_runs(job, "train.json") if g.get("train_seconds") is not None]   # round3 r2_train.py
    if runs:
        return {"gpu_hours_finetune": round(sum(float(g["seconds"]) for g in runs) / 3600, 4), "finetune_models": len(runs),
                "finetune_basis": "bge_m3_ft run.json 'seconds': training plus encoding of the catalogue and held-out "
                                  "queries with each fold model (one timer)"}
    if trains:
        return {"gpu_hours_finetune": round(sum(float(g["train_seconds"]) for g in trains) / 3600, 4),
                "gpu_hours_scoring": round(sum(float(g.get("score_seconds") or 0) for g in trains) / 3600, 4),
                "finetune_models": len(trains), "finetune_basis": "r2_train train.json train_seconds (score_seconds apart)"}
    return {"gpu_hours_finetune": None}


def record(system: str, store: Path | None = None, latency_rec: dict[str, Any] | None = None) -> dict[str, Any]:
    """Assemble the cost record of a system from its receipts, its latency measurement and any GPU job runs."""
    store = Path(store) if store else context.active().store
    spec = registry.get(system)
    receipts = sorted((store / "receipts").glob(f"{system}__*.json"))
    fits, peaks = [], []
    for p in receipts:
        r = json.loads(p.read_text(encoding="utf-8"))
        nf = max(len(r.get("folds", [])), 1)
        fits.append((r["timing"].get("fit_seconds", 0.0) + r["timing"].get("prepare_seconds", 0.0)) / nf / 60.0)
        if r.get("peak_ram_gb"):
            peaks.append(r["peak_ram_gb"])
    job = spec.get("adapter_args", {}).get("job", system)
    gpu = _gpu_runs(job)
    enc = [g for g in gpu if g.get("encode_seconds") and not _query_only(g)]
    lat = dict(latency_rec) if latency_rec else None
    idx = lat.pop("index_size", None) if lat else None
    rec = {"system": system, "family": spec["family"], "hardware": spec["hardware"],
           "indexing": {"gpu_hours_5080": round(sum(g["encode_seconds"] for g in enc) / 3600, 4) if enc else None,
                        "endpoint_seconds": None, "cpu_minutes": None},
           "training": {"cpu_minutes_per_fold": round(float(np.median(fits)), 3) if fits else None, **_finetune(job)},
           "query_latency": lat, "query_encoding": query_encoding(spec),
           "memory": {"peak_ram_gb": max(peaks) if peaks else None,
                      "peak_vram_gb": round(max(g.get("peak_vram_mib") or 0 for g in gpu) / 1024, 2) if gpu else None},
           "index_size_mb": (round(max(g.get("index_bytes") or 0 for g in gpu) / 2 ** 20, 1)
                             if gpu and any(g.get("index_bytes") for g in gpu) else None),
           "llm": None, "update_after_reform_seconds": None,
           "cost_per_1000_cases": {"usd": None, "basis": "no cited price recorded (PLAN §6: USD only at a cited, dated price)"},
           "receipts": [p.name for p in receipts]}
    if rec["index_size_mb"] is not None:
        rec["index_size"] = {"mb": rec["index_size_mb"], "basis": "gpu_job_index_bytes", "job": job}
    elif idx:
        rec["index_size"] = idx
        rec["index_size_mb"] = idx.get("mb")
    gpu_lat = [g["query_latency"] for g in gpu if g.get("query_latency")] if gpu else []
    if gpu_lat and not lat:                      # late interaction: latency measured inside the GPU job (D79)
        rec["query_latency"] = lat = gpu_lat[-1]
    if lat and lat.get("median_ms") is not None:
        rec["requirements"] = {"latency_le_5s": bool(lat["p95_ms"] <= 5000)}
    harness.write_json(store / "cost" / f"{system}.json", rec)
    return rec
