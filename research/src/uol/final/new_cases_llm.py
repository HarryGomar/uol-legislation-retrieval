"""E13 second stages on the new cases, each once: P6 (the LLM stage over S_full_ext, Codex gpt-5.6-luna), Claude and
Jev (D101, external APIs on this PC, keys only from ``.env``), all over S_full_ext's top 30 on NEW.

The frozen recipes run unchanged: ``l1_luna.run`` (main prompt, windows 11-30 then 1-10 plus A's first 10, forward and
reversed, 3 samples) fused with S_full_ext by ``round3.features.fuse_rows`` at the record's all-development weight
(w_all .75); Claude through ``external.rerank_orders`` / ``ask_claude`` (one sample, fused at the record's w_all 3.0);
Jev through ``external.jev_request`` (its order alone, and RRF k 60 with S_full_ext). Cases: the new rows in P_spec
(the record's P6 ran on the development rows of P_spec). The LLM cache and ledger are this run's own
(``<run>/llm/luna``); no earlier ledger is written.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from uol.final import new_cases_sources as NSrc
from uol.final.new_cases_fit import ROOT, log, sha256, write_json
from uol.final.new_cases_steps import CLAUDE_W_ALL, P6

TAG = "E13_L1_S_full_ext"


def _base(run: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """(P_spec new rows, S_full_ext base matrix [rows, N] in ``llm_jobs.base_matrix`` arithmetic, new_rows)."""
    import pandas as pd
    from uol.embedding_push import common as ep
    run = Path(run)
    NSrc.activate_ext(run / "release_ext", run / "logs" / "embedding_push")
    nr = NSrc.new_rows(run / "release_ext")
    rows = np.asarray(ep.population_rows("P_spec", np.asarray([r["row"] for r in nr], dtype=np.int64)), dtype=np.int64)
    d = ep.release()
    top = pd.read_parquet(run / "eval" / P6["base"] / "top200.parquet")
    pos = {d.queries[int(r)].query_id: i for i, r in enumerate(rows)}
    N = len(d.candidate_ids)
    M = np.empty((len(rows), N), dtype=np.float64)
    M[:] = -1e9 - np.arange(N)[None, :] * 1e-6
    for qid, g in top.groupby("query_id"):
        if qid in pos:
            M[pos[qid], g["candidate_idx"].to_numpy()] = 1e6 - g["rank"].to_numpy()
    return rows, M, nr


def _save(run: Path, system: str, nr: list[dict], rows: np.ndarray, M: np.ndarray, meta: dict) -> dict:
    """``scores/<system>/scores.npy`` over every new row (rows outside P_spec keep S_full_ext's scores; they are not
    evaluated), with the same manifest as ``new_cases_steps.score``."""
    out = Path(run) / "scores" / system
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{system} is scored once")
    base = np.load(Path(run) / "scores" / P6["base"] / "scores.npy")
    S = base.copy()
    loc = {r["row"]: i for i, r in enumerate(nr)}
    for j, r in enumerate(rows):
        S[loc[int(r)]] = M[j]
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "scores.npy", S.astype(np.float32))
    man = {"system": system, "n": len(nr), "query_ids": [r["query_id"] for r in nr], "llm_rows": [int(r) for r in rows],
           **meta, "scores_sha256": sha256(out / "scores.npy"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "manifest.json", man)
    return man


def arm(rows: np.ndarray, orders: list[dict], N: int) -> np.ndarray:
    """1 / mean position over samples and passes (``llm_jobs.fuse_orders``); NaN outside the reranked top 30."""
    A = np.full((len(rows), N), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        rk: dict[int, list[int]] = {}
        for o in orders:
            for p in o["passes"]:
                for pos, c in enumerate(o["orders"][p][str(int(row))], 1):
                    rk.setdefault(int(c), []).append(pos)
        for c, v in rk.items():
            A[i, c] = 1.0 / float(np.mean(v))
    return A


def p6(run: Path) -> dict:
    from uol.final import codex
    from uol.round3 import common as C, features, l1_luna, luna
    run = Path(run)
    out = run / "llm" / P6["system"]
    if (run / "scores" / P6["system"] / "manifest.json").exists():
        raise FileExistsError("P6 is scored once")
    rows, M, nr = _base(run)
    top = np.stack([np.argsort(-M[i], kind="stable")[:P6["top"]] for i in range(len(rows))])
    codex.install()
    luna.CACHE = run / "llm" / "luna" / "cache.sqlite"
    luna.LEDGER = run / "llm" / "luna" / "LUNA_LEDGER.jsonl"
    luna.CACHE.parent.mkdir(parents=True, exist_ok=True)
    closed = np.asarray([r for r in C.lockbox_rows() if int(r) < NSrc.N_V11], dtype=np.int64)
    orig = C.lockbox_rows
    C.lockbox_rows = lambda: closed            # the guard keeps refusing the closed test set; the new rows pass
    t0 = time.perf_counter()
    try:
        l1_luna.OUT = out
        orders = []
        for s in range(P6["samples"]):
            if (out / f"orders_s{s}.json").exists():        # a crashed run resumes from its saved samples
                orders.append(json.loads((out / f"orders_s{s}.json").read_text(encoding="utf-8")))
                continue
            r = l1_luna.run([int(x) for x in rows], "main", 8, TAG, sample=s, top=top, out_name=f"orders_s{s}.json")
            orders.append(json.loads(Path(r["path"]).read_text(encoding="utf-8")))
            log(f"P6: sample {s} done ({time.perf_counter() - t0:.0f}s)", run / "logs" / "llm.log")
    finally:
        C.lockbox_rows = orig
    A = arm(rows, orders, M.shape[1])
    fused = features.fuse_rows(M, A, P6["w_all"])
    usage = ledger_usage(luna.LEDGER, TAG)
    return _save(run, P6["system"], nr, rows, fused,
                 {"base": P6["base"], "w_all": P6["w_all"], "samples": P6["samples"], "llm": luna.MODEL,
                  "llm_knowledge_cutoff": luna.MODEL_KNOWLEDGE_CUTOFF, "effort": l1_luna.EFFORT,
                  "prompt_sha256": sha256(l1_luna.PROMPT), "orders": {f"orders_s{s}.json": sha256(out / f"orders_s{s}.json")
                                                                       for s in range(P6["samples"])},
                  "usage": usage, "seconds": round(time.perf_counter() - t0, 1)})


def ledger_usage(path: Path, tag_prefix: str) -> dict[str, Any]:
    calls = toks_in = toks_out = cached = 0
    if Path(path).exists():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if not str(r.get("tag", "")).startswith(tag_prefix):
                continue
            calls += 1
            u = r.get("usage") or {}
            toks_in += int(u.get("input_tokens") or 0)
            toks_out += int(u.get("output_tokens") or 0)
            cached += int(u.get("cached_input_tokens") or 0)
    return {"calls": calls, "input_tokens": toks_in, "cached_input_tokens": cached, "output_tokens": toks_out}


def claude(run: Path, venv_python: Path, env_file: Path) -> dict:
    from uol.final import external as X
    from uol.round3 import features, l1_luna
    run = Path(run)
    system = X.SYSTEMS["claude"]
    if (run / "scores" / system / "manifest.json").exists():
        raise FileExistsError("Claude is scored once")
    rows, M, nr = _base(run)
    top = {int(r): [int(c) for c in np.argsort(-M[i], kind="stable")[:X.TOP]] for i, r in enumerate(rows)}
    work = run / "llm" / "claude"
    work.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}
    api = X.venv_api(work, Path(venv_python), Path(env_file))
    t0 = time.perf_counter()
    o = X.rerank_orders([int(r) for r in rows], top, lambda items: X.ask_claude(items, api, stats))
    ledger = json.loads((work / "ledger.json").read_text(encoding="utf-8")) if (work / "ledger.json").exists() else {}
    o.update({"model": X.CLAUDE["model"], "effort": X.CLAUDE["effort"], **stats,
              "prompt_sha256": hashlib.sha256(l1_luna.PROMPT.read_bytes()).hexdigest(), "population": "NEW (P_spec)",
              "usage": ledger.get("usage"), "usd": ledger.get("usd"), "pricing": X.CLAUDE["pricing"]})
    (work / X.FILES["claude"]).write_text(json.dumps(o, indent=1), encoding="utf-8")
    fused = features.fuse_rows(M, arm(rows, [o], M.shape[1]), CLAUDE_W_ALL)
    return _save(run, system, nr, rows, fused,
                 {"base": P6["base"], "w_all": CLAUDE_W_ALL, "samples": 1, "model": X.CLAUDE["model"],
                  "effort": X.CLAUDE["effort"], "usage": ledger.get("usage"), "usd": ledger.get("usd"),
                  "unanswered": stats.get("unanswered"), "repairs": stats.get("repairs"),
                  "orders_sha256": sha256(work / X.FILES["claude"]), "seconds": round(time.perf_counter() - t0, 1)})


def jev(run: Path, env_file: Path) -> list[dict]:
    from uol.embedding_push import common as ep
    from uol.final import external as X
    from uol.round3 import jev_probe as J
    run = Path(run)
    if (run / "scores" / X.SYSTEMS["jev"] / "manifest.json").exists():
        raise FileExistsError("Jev is scored once")
    rows, M, nr = _base(run)
    d = ep.release()
    top = {int(r): [int(c) for c in np.argsort(-M[i], kind="stable")[:X.TOP]] for i, r in enumerate(rows)}
    work = run / "llm" / "jev"
    work.mkdir(parents=True, exist_ok=True)
    path = work / X.FILES["jev"]
    done = {json.loads(x)["row"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else {}
    spent = sum(((x.get("usage") or {}).get("input_tokens") or 0) for x in done.values()) * X.JEV["usd_per_mtok_in"] / 1e6
    texts = J.article_texts()
    key = X.read_env(Path(env_file), "JEV_KEY") if any(int(r) not in done for r in rows) else ""
    t0 = time.perf_counter()
    for r in [int(x) for x in rows]:
        if r in done:
            continue
        est = sum(len(texts[int(c)][:J.ART_CHARS]) for c in top[r]) / 3.0 * X.JEV["usd_per_mtok_in"] / 1e6
        if spent + 2 * est > X.JEV["cap_usd"]:
            raise RuntimeError(f"Jev spend cap: {spent:.2f} spent, the next request could pass US${X.JEV['cap_usd']}")
        rec = {"row": r, "candidates": top[r], **X.jev_request(d.queries[r].text, top[r], texts, key)}
        spent += ((rec.get("usage") or {}).get("input_tokens") or 0) * X.JEV["usd_per_mtok_in"] / 1e6
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        done[r] = rec
    del key
    alone, rrf = M.copy(), M.copy()
    for i, r in enumerate(rows):
        c = np.asarray(done[int(r)]["candidates"])
        p = np.asarray(done[int(r)]["p_yes"])
        jev_rank = np.empty(len(c))
        jev_rank[np.lexsort((np.arange(len(c)), -p))] = np.arange(1, len(c) + 1)
        base_rank = np.arange(1, len(c) + 1)
        alone[i, c] = 1e6 - jev_rank
        rrf[i, c] = 1e6 + 1.0 / (X.JEV["rrf_k"] + base_rank) + 1.0 / (X.JEV["rrf_k"] + jev_rank)
    toks = sum(((x.get("usage") or {}).get("input_tokens") or 0) for x in done.values())
    meta = {"base": P6["base"], "depth": X.JEV["depth"], "input_tokens": toks,
            "usd_at_configured_price": round(toks * X.JEV["usd_per_mtok_in"] / 1e6, 4),
            "models": sorted({str(x.get("model")) for x in done.values()}), "requests_sha256": sha256(path),
            "seconds": round(time.perf_counter() - t0, 1)}
    return [_save(run, X.SYSTEMS["jev"], nr, rows, alone, {**meta, "fusion": "none (Jev order on the top 30)"}),
            _save(run, X.SYSTEMS["jev_rrf"], nr, rows, rrf, {**meta, "fusion": f"RRF k {X.JEV['rrf_k']} with S_full_ext"})]


def dispatch(a, run: Path, models: Path) -> int:
    if a.step == "llm":
        print(json.dumps({k: v for k, v in p6(run).items() if k not in ("query_ids", "llm_rows")}, indent=1))
        return 0
    if a.step == "external":
        env = ROOT / ".env"
        venv = ROOT / ".cache" / "venvs" / "c10" / "Scripts" / "python.exe"
        for kind in (a.kinds or "jev,claude").split(","):
            m = jev(run, env) if kind == "jev" else claude(run, venv, env)
            m = m[0] if isinstance(m, list) else m
            print(json.dumps({k: v for k, v in m.items() if k not in ("query_ids", "llm_rows")}, indent=1))
        return 0
    raise ValueError(a.step)
