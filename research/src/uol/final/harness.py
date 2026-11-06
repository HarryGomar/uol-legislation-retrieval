"""Fit, score and evaluate any registered system on any population (PLAN_DRAFT §10), writing the results store.

    <store>/runs/<system>/<population>/per_case.parquet   one row per case: every measure (uol.final.measures)
    <store>/runs/<system>/<population>/top200.parquet     the ranking actually scored (condensed list), with scores
    <store>/runs/<system>/<population>/summary.json       component-macro means
    <store>/receipts/<system>__<population>.json          inputs and hashes, versions, fold choices, timing, peak RAM
    <store>/status/<system>__<population>.json            done | pending | unavailable | failed

The outer folds are the five component-grouped development folds (``Population.folds``); every tuned choice is made
inside the training folds by the adapter (``adapters.base.select``). Every row list passes ``guard.dev_rows_only``.
A run whose receipt exists is skipped (idempotent) unless ``force``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard, measures, populations, registry
from uol.final.adapters.base import SystemPending, SystemUnavailable

BLOCK = int(os.environ.get("UOL_FINAL_BLOCK", "128"))
CODE_DIRS = ("final",)


def run_dir(store: Path, system: str, population: str) -> Path:
    return Path(store) / "runs" / system / population


def receipt_path(store: Path, system: str, population: str) -> Path:
    return Path(store) / "receipts" / f"{system}__{population}.json"


def status_path(store: Path, system: str, population: str) -> Path:
    return Path(store) / "status" / f"{system}__{population}.json"


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


def peak_ram_gb() -> float | None:
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(getattr(mi, "peak_wset", mi.rss) / 2 ** 30, 3)
    except Exception:                                   # pragma: no cover
        return None


def code_hashes() -> dict[str, str]:
    """sha256 of every file of uol.final and of the inherited modules it calls (receipts)."""
    src = Path(__file__).resolve().parents[1]
    files = sorted((src / "final").rglob("*.py"))
    for rel in ("embedding_push/common.py", "embedding_push/base.py", "embedding_push/fusion_ltr.py", "round3/features.py",
                "round3/common.py", "round3/stats.py", "round3/c_hybrid.py", "round3/e2_adapter.py", "gap_tests/d58_refit.py",
                "gap_tests/lexical_grid.py", "facts_to_law/metrics.py", "facts_to_law/crossfit.py", "facts_to_law/lexical.py",
                "facts_to_law/fusion.py", "scale_curve/legal.py"):
        if (src / rel).exists():
            files.append(src / rel)
    return {p.relative_to(src).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def versions() -> dict[str, str]:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for m in ("numpy", "scipy", "sklearn", "pandas", "pyarrow", "gensim", "lightgbm", "nltk"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = "missing"
    return out


def release_receipt() -> dict[str, Any]:
    from uol.embedding_push import common as ep
    d = ep.release()
    return {"release_id": d.manifest.get("release_id"), "release_dir": str(d.release_dir),
            "manifest_sha256": d.hashes.get("manifest.json"), "files": d.hashes,
            "candidates": len(d.candidate_ids), "rows": len(d.queries),
            "split_sha256": ep.sha256_file(ep.SPLIT_PATH)}


def run(system: str, population: str = "DEV", *, store: Path | None = None, folds: Sequence[int] | None = None,
        max_rows: int | None = None, force: bool = False, tag: str | None = None) -> dict[str, Any]:
    """Score one system on one population; returns the summary (or a status record)."""
    guard.refuse_population(population)
    prof = context.active()
    store = Path(store) if store else prof.store
    if tag is None:
        tag = prof.tag or None                             # "@rebuilt" (sources from code), "@v11" (bundle, G1 labels)
    name = system if tag is None else f"{system}{tag}"
    rp, sp = receipt_path(store, name, population), status_path(store, name, population)
    if rp.exists() and not force:
        return {"system": name, "population": population, "status": "done", "skipped": True,
                **json.loads((run_dir(store, name, population) / "summary.json").read_text(encoding="utf-8"))}
    spec = registry.get(system)
    t_start = time.time()
    base = {"system": name, "population": population, "utc_start": _utc(), "profile": prof.kind}
    try:
        pop = populations.load(population, store)
        scorer = registry.make(spec)
        t0 = time.perf_counter()
        scorer.prepare(pop)
        prep_s = time.perf_counter() - t0
        result = _evaluate(scorer, pop, folds=folds, max_rows=max_rows)
    except SystemPending as exc:
        rec = {**base, "status": "pending", "reason": str(exc)}
        write_json(sp, rec)
        return rec
    except SystemUnavailable as exc:
        rec = {**base, "status": "unavailable", "reason": str(exc)}
        write_json(sp, rec)
        return rec
    except guard.TestSetRefused:
        raise
    except Exception as exc:
        rec = {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        write_json(sp, rec)
        raise
    out = run_dir(store, name, population)
    out.mkdir(parents=True, exist_ok=True)
    frame, top = result.pop("frame"), result.pop("top")
    _write_parquet(frame, out / "per_case.parquet")
    _write_parquet(top, out / "top200.parquet")
    summ = {"system": name, "base_system": system, "population": population, "eval_population": pop.eval_population,
            **measures.summary(frame), "partial": bool(folds is not None or max_rows is not None)}
    write_json(out / "summary.json", summ)
    receipt = {**base, "status": "done", "utc_end": _utc(), "spec": spec, "population_meta": pop.meta,
               "inputs": _safe(scorer.inputs), "folds": result["folds"], "timing": {
                   "prepare_seconds": round(prep_s, 3), "wall_seconds": round(time.time() - t_start, 3),
                   **result["timing"]}, "peak_ram_gb": peak_ram_gb(), "release": release_receipt(),
               "versions": versions(), "code_sha256": code_hashes(), "patches": prof.patches,
               "outputs": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*")) if p.is_file()},
               "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "UOL_FINAL_THREADS")}}
    write_json(rp, receipt)
    write_json(sp, {**base, "status": "done", "summary": summ})
    scorer.close()
    return {**summ, "status": "done"}


def _safe(fn) -> Any:
    try:
        return fn()
    except Exception as exc:                            # pragma: no cover
        return {"error": str(exc)}


def _utc() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _write_parquet(frame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _evaluate(scorer, pop, *, folds: Sequence[int] | None, max_rows: int | None) -> dict[str, Any]:
    import pandas as pd
    from uol.embedding_push import common as ep
    d = ep.release()
    cases, tops, fold_info = [], [], []
    t_fit = t_score = t_eval = 0.0
    n_scored = 0
    for fold in pop.folds():
        if folds is not None and fold.k not in folds:
            continue
        rows = guard.dev_rows_only(fold.score_rows, f"{pop.name} fold {fold.k}")
        if max_rows is not None:
            rows = rows[:max_rows]
        t = time.perf_counter()
        info = scorer.fit(fold)
        t_fit += time.perf_counter() - t
        unseen = measures.unseen_mask(fold.train_rows)
        for s in range(0, len(rows), BLOCK):
            blk = rows[s:s + BLOCK]
            t = time.perf_counter()
            S = np.asarray(scorer.score(blk), dtype=np.float32)
            t_score += time.perf_counter() - t
            if S.shape != (len(blk), len(d.candidate_ids)):
                raise RuntimeError(f"{scorer.sid}: score block {S.shape}, expected {(len(blk), len(d.candidate_ids))}")
            t = time.perf_counter()
            res = ep.evaluate(S, blk, pop.eval_population, top_k=measures.TOP_K)
            loc = {int(r): j for j, r in enumerate(blk)}
            sub = S[[loc[int(r)] for r in res["rows"]]]
            per = measures.per_case(res["per_query"], res["rows"], sub, specific=pop.specific, unseen=unseen)
            for r, rec, q in zip(res["rows"], per, res["per_query"]):
                qq = d.queries[int(r)]
                cases.append({"query_id": qq.query_id, "row": int(r), "component_id": qq.component_id,
                              "fold": int(fold.k), **rec})
                row_s = S[loc[int(r)]]
                for rank, c in enumerate(q["topk"], 1):
                    tops.append((qq.query_id, rank, int(c), float(row_s[int(c)])))
            t_eval += time.perf_counter() - t
            n_scored += len(blk)
        fold_info.append({"k": int(fold.k), "role": fold.role, "n_scored": int(len(rows)),
                          "n_train_rows": int(len(fold.train_rows)), "fit": info})
    if not cases:
        raise SystemUnavailable(f"{scorer.sid}: no case of {pop.name} was scored")
    frame = pd.DataFrame(cases)
    top = pd.DataFrame(tops, columns=["query_id", "rank", "candidate_idx", "score"])
    return {"frame": frame, "top": top, "folds": fold_info,
            "timing": {"fit_seconds": round(t_fit, 3), "score_seconds": round(t_score, 3), "eval_seconds": round(t_eval, 3),
                       "cases_scored": n_scored, "score_ms_per_case": round(1000 * t_score / max(n_scored, 1), 3)}}


def load_cases(store: Path, system: str, population: str):
    import pandas as pd
    return pd.read_parquet(run_dir(store, system, population) / "per_case.parquet")
