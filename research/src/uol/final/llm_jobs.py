"""Block E and the other LLM experiments, run on the run computer through its Codex CLI (RUN_PLAN v2 §3.8, §7).

Each function reuses the existing experiment code unchanged and writes a score artefact the harness evaluates like
any other system (``adapters.llm.LLMMatrix``): ``<work>/llm/<system>/{rows.npy, matrix.npy, meta.json}``.

* ``llm_alone``: ``eval_lenses.llm_alone.run`` (closed book, 20 (law, article) names per case, batches of 10);
  names resolved with ``round3.catalog_map.resolve``; score 21 - position, every other article below.
* ``hyde``: a hypothetical-article prompt (``uol/final/prompts/hyde.txt``, 20 cases per call, prompt guard), the
  texts go to the GPU encode queue (Qwen3-8B int8 document side) and are scored by ``adapters.dense``.
* ``llm_rerank``: ``round3.l1_luna.run`` (window 20, stride 10, two passes, top 30) over the top 30 of a finished
  ranker in the store, three samples; arm score 1 / mean rank; fused with the ranker by ``round3.features.fusion_route``
  over ``c_hybrid.W_EXT`` (cross-fitted weight), exactly the C_hybrid stage-3 recipe.
* ``agent``: ``eval_lenses.e6_agent.run`` (lockstep tool agent; conditions ``words`` and ``embedding``); the
  embedding tool's query vectors come from the local Qwen3-8B int8 encoder (fidelity-gated) instead of the endpoint.

All LLM traffic goes through ``codex.install`` (adaptive concurrency, quota pause and resume, cached answers).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import codex, context, guard

PROMPTS = Path(__file__).resolve().parent / "prompts"
HYDE_TEMPLATE = PROMPTS / "hyde.txt"
HYDE_BATCH = 20
HYDE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["texto"],
               "properties": {"texto": {"type": "string"}}}


def out_dir(system: str) -> Path:
    p = context.active().work / "llm" / system
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_matrix(system: str, rows: np.ndarray, M: np.ndarray, meta: dict[str, Any]) -> Path:
    d = out_dir(system)
    rows = guard.dev_rows_only(rows, f"{system} rows")
    np.save(d / "rows.tmp.npy", rows)
    np.save(d / "matrix.tmp.npy", M.astype(np.float32))
    (d / "rows.tmp.npy").replace(d / "rows.npy")
    (d / "matrix.tmp.npy").replace(d / "matrix.npy")
    (d / "meta.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **meta},
                                            indent=1, default=str), encoding="utf-8")
    return d


def dev_eval_rows() -> np.ndarray:
    from uol.embedding_push import common as ep
    return guard.dev_rows_only(ep.population_rows("P_spec", ep.dev_rows()))


# --------------------------------------------------------------------------- llm_alone


def llm_alone(rows: Sequence[int] | None = None) -> Path:
    codex.install()
    from uol.eval_lenses import llm_alone as LA
    from uol.embedding_push import common as ep
    from uol.round3 import catalog_map
    LA.OUT = out_dir("llm_alone_raw")
    rows = [int(r) for r in (dev_eval_rows() if rows is None else guard.dev_rows_only(rows))]
    outputs = LA.run(rows)
    N = len(ep.release().candidate_ids)
    M = np.full((len(rows), N), -1.0, dtype=np.float32)
    unresolved = 0
    for i, r in enumerate(rows):
        for pos, it in enumerate(outputs[r]["articulos"]):
            idx, _ = catalog_map.resolve(it.get("ley", ""), str(it.get("articulo", "")))
            if idx is None:
                unresolved += 1
            elif M[i, idx] < 0:
                M[i, idx] = 21.0 - pos
    (out_dir("llm_alone_raw") / "outputs.json").write_text(json.dumps({str(k): v for k, v in outputs.items()},
                                                                      ensure_ascii=False), encoding="utf-8")
    return save_matrix("llm_alone", np.asarray(rows), M, {"model": codex.bind().MODEL, "unresolved_names": unresolved,
                                                          "rule": "score 21 - position of the named article"})


# --------------------------------------------------------------------------- hyde


def hyde(rows: Sequence[int] | None = None) -> Path:
    """Hypothetical article texts for every development row (training rows too: the texts are label-free and are
    needed for ranker columns); written to <work>/llm/hyde/texts.jsonl for the GPU encode job."""
    luna = codex.install()
    from uol.embedding_push import common as ep
    from uol.round3 import prompt_guard
    d = ep.release()
    rows = [int(r) for r in (guard.dev_rows_only(ep.dev_rows()) if rows is None else guard.dev_rows_only(rows))]
    tpl = HYDE_TEMPLATE.read_text(encoding="utf-8")

    def build(chunk: list[int], ids: list[str]) -> str:
        items = "\n\n".join(f"[{i}] {d.queries[r].text}" for i, r in zip(ids, chunk))
        prompt = tpl.replace("{items}", items)
        prompt_guard.check(prompt, chunk, template=tpl)
        return prompt

    jobs, ids_of = [], []
    for s in range(0, len(rows), HYDE_BATCH):
        chunk = rows[s:s + HYDE_BATCH]
        ids = [f"c{j + 1:02d}" for j in range(len(chunk))]
        jobs.append((build(chunk, ids), luna.batch_schema(HYDE_SCHEMA)))
        ids_of.append((chunk, ids))
    res = luna.run_many(jobs, effort="low", tag="final_hyde")
    out = out_dir("hyde") / "texts.jsonl"
    splits = {"split_batches": 0, "split_cases": 0}
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for (chunk, ids), (prompt, schema), r in zip(ids_of, jobs, res):
            row_of = dict(zip(ids, chunk))
            # a batch still invalid after its resamples is split; each case keeps its own id (22 Oct 2026)
            got = luna.checked_or_split(r, prompt, schema, ids, lambda i: (build([row_of[i]], [i]), i),
                                        effort="low", tag="final_hyde", stats=splits)
            for i, row in zip(ids, chunk):
                f.write(json.dumps({"row": row, "query_id": d.queries[row].query_id, "text": got[i]["texto"]},
                                   ensure_ascii=False) + "\n")
    (out.parent / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    return out


# --------------------------------------------------------------------------- llm_rerank


def base_matrix(store: Path, system: str, rows: np.ndarray) -> np.ndarray:
    """Full-width matrix of a finished run: its top-200 scores, every other article below them (index order)."""
    import pandas as pd
    from uol.embedding_push import common as ep
    top = pd.read_parquet(Path(store) / "runs" / system / "DEV" / "top200.parquet")
    d = ep.release()
    pos = {d.queries[int(r)].query_id: i for i, r in enumerate(rows)}
    N = len(d.candidate_ids)
    M = np.empty((len(rows), N), dtype=np.float64)
    M[:] = -1e9 - np.arange(N)[None, :] * 1e-6
    for qid, g in top.groupby("query_id"):
        if qid in pos:
            i = pos[qid]
            M[i, g["candidate_idx"].to_numpy()] = 1e6 - g["rank"].to_numpy()       # strict order of the run
    return M


def llm_rerank(base: str, samples: int = 3, store: Path | None = None, limit: int | None = None,
               smoke: bool = False, variant: str = "main") -> Path:
    """``variant`` "blank" (C5, D93): the candidates as law and article number only, no text (``l1_luna`` blank);
    its own orders, tags and matrix (``llm_rerank_<base>_blank``), fused with the same per-fold weight search."""
    if variant not in ("main", "blank"):
        raise ValueError(f"llm_rerank variant {variant!r}")
    system = f"llm_rerank_{base}" + ("_blank" if variant == "blank" else "")
    tag = f"final_L1_{base}" if variant == "main" else f"final_L1blank_{base}"
    codex.install()
    from uol.round3 import c_hybrid, features, l1_luna
    prof = context.active()
    store = Path(store) if store else prof.store
    rows = dev_eval_rows()
    if limit:                                         # smoke: the first rows the base run has scored
        import pandas as pd
        have = set(pd.read_parquet(Path(store) / "runs" / f"{base}{prof.tag}" / "DEV" / "top200.parquet")["query_id"])
        d0 = __import__("uol.embedding_push.common", fromlist=["x"]).release()
        rows = np.asarray([r for r in rows if d0.queries[int(r)].query_id in have][:limit], dtype=np.int64)
    S = base_matrix(store, f"{base}{prof.tag}", rows)
    top = np.stack([np.argsort(-S[i], kind="stable")[:30] for i in range(len(rows))])
    orders = []
    for s in range(samples):
        l1_luna.OUT = out_dir(system)
        r = l1_luna.run([int(x) for x in rows], variant, 8, tag, sample=s, top=top, out_name=f"orders_s{s}.json")
        orders.append(json.loads(Path(r["path"]).read_text(encoding="utf-8")))
    if smoke:
        return Path(r["path"])
    return fuse_orders(system, base, rows, orders, S,
                       {"samples": samples, "variant": variant,
                        "recipe": f"round3.c_hybrid stage 3 (l1_luna {variant}, {samples} samples, fusion_route W_EXT)"})


def fuse_orders(system: str, base: str, rows: np.ndarray, orders: list[dict], S: np.ndarray, meta: dict) -> Path:
    """The LLM stage's arm score (1 / mean position over samples and passes) fused with the base ranking by the
    per-fold weight search of round3.c_hybrid stage 3; shared by the main rerank and external rerankers (C10)."""
    from uol.round3 import c_hybrid, features
    N = S.shape[1]
    A = np.full((len(rows), N), np.nan, dtype=np.float32)
    for i, row in enumerate(rows):
        rk: dict[int, list[int]] = {}
        for o in orders:
            for p in o["passes"]:
                for pos, c in enumerate(o["orders"][p][str(int(row))], 1):
                    rk.setdefault(int(c), []).append(pos)
        for c, v in rk.items():
            A[i, c] = 1.0 / float(np.mean(v))
    fr = features.fusion_route(base, rows, A, grid=c_hybrid.W_EXT, ltr_matrix=S)
    return save_matrix(system, rows, fr["oof_matrix"],
                       {"base": base, **meta, "w_by_fold": fr["w_by_fold"], "w_all": fr["w_all"]})


# --------------------------------------------------------------------------- agent (E12)


def agent_baseline_cache(prof=None) -> Path:
    """The E6 agent records, per case, the nDCG@10 of the frozen ranker ``LTR_emb_hgb`` beside its own
    (``eval_lenses.core.load("LTR_emb_hgb")``). ``core.CACHE`` was fixed at import to the thesis-PC folder
    (``experiments/runs/eval_lenses_20261015/cache``, not bundled): it is moved to the work folder and the cache is
    built there from the bundled frozen pipeline scores (dev rows, the profile's labels), once (21 Oct 2026)."""
    from uol.eval_lenses import core
    prof = prof or context.active()
    core.OUT = prof.work / "eval_lenses"
    core.CACHE = core.OUT / "cache"
    core.load.cache_clear()
    core.meta.cache_clear()
    prof.patches["uol.eval_lenses.core.CACHE"] = str(core.CACHE)
    return core.build("LTR_emb_hgb")


def agent(condition: str, rows: Sequence[int] | None = None) -> Path:
    """E6 lockstep agent on the given DEV evaluation rows (every DEV case by default; the embedding-tool agent runs
    on the DEV-A300 sample, D96); the embedding tool encodes its queries locally (int8)."""
    codex.install()
    from uol.embedding_push import common as ep
    from uol.eval_lenses import core, e6_agent
    prof = context.active()
    e6_agent.OUT = out_dir(f"agent_{condition}")
    agent_baseline_cache(prof)
    if condition == "embedding":
        from uol.final import qwen_local
        qwen_local.patch_endpoint()                          # endpoint client -> local Qwen3-8B int8 (no network)
    rows = [int(r) for r in (dev_eval_rows() if rows is None else guard.dev_rows_only(rows))]
    res = e6_agent.run(rows, condition, f"final_E6_{condition}")
    N = len(ep.release().candidate_ids)
    M = np.full((len(rows), N), -1.0, dtype=np.float32)
    by_row = {int(c["row"]): c for c in res.get("cases", [])}
    for i, r in enumerate(rows):
        for pos, c in enumerate(by_row.get(r, {}).get("returned", [])[:10]):
            idx = e6_agent.parse_aid(c)
            if idx is not None:
                M[i, int(idx)] = 11.0 - pos
    (out_dir(f"agent_{condition}") / "result.json").write_text(json.dumps(res, default=str), encoding="utf-8")
    tool = ("local Qwen3-Embedding-8B int8 (round3/gpu_jobs/legal_encode.load, Gate E0 and the RUN_PLAN fidelity check "
            "passed before this job), the same query strings the endpoint embedded (variant c)") if condition == "embedding" \
        else "BM25 over stemmed Spanish words (e6_agent words tool)"
    return save_matrix(f"agent_{condition}", np.asarray(rows), M, {"condition": condition, "usage": res.get("usage"),
                                                                   "splits": res.get("splits"),
                                                                   "search_tool": tool, "endpoint_calls": 0})
