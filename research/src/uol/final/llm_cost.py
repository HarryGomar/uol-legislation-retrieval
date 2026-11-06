"""LLM cost per system (RUN_PLAN §5 "Cost record per system"): calls, input / cached / output tokens and seconds per
DEV case, written into ``<store>/cost/<system>.json`` (field ``llm``) by the plan job ``cost_summary``, so they reach
``results/raw/cost/`` and the thesis pack (frontier table) even for systems without a cost job.

Which calls belong to a system (the tags of ``llm_jobs``, ``round3.l1_luna``, ``eval_lenses``):

    llm_alone              tag llm_alone_dev               (eval_lenses.llm_alone.TAG, also used on the thesis PC)
    llm_rerank_S0          tags final_L1_S0_<pass><A|B>     (llm_jobs.llm_rerank -> l1_luna.run, three samples)
    llm_rerank_S_full_ext  tags final_L1_S_full_ext_<pass><A|B>
    agent_words            tag final_E6_words              (llm_jobs.agent -> e6_agent.run)
    agent_embedding        tag final_E6_embedding
    hyde                   tag final_hyde                  (llm_jobs.hyde; texts for every development row)
    query2doc              the L2 issue decompositions (thesis PC, tag L2, variant a): the cache keys recorded per
                           row in round3 L2_a.jsonl (``luna_key``); query2doc makes no LLM call on the run computer

Sources, best first (``basis``):

* ``call_log``: ``<work>/luna/calls.jsonl`` written by ``codex.install`` for every answered call, cache reuse included,
  with the tag of the calling system; each distinct cache key counts once, with the usage and latency stored with the
  answer (the cost of producing it). Exact (``complete: true``).
* ``issue_file_keys`` (query2doc): the cache rows of the recorded keys (sample 0; retried samples are not recorded).
* ``cache_rows_by_tag``: the cache rows whose tag is the system's. A cache row keeps the tag of the call that created
  it, so answers reused from the seeded thesis-PC cache (tags L1_, L1h_, E6_) are not counted, and rows of a tag also
  used on the thesis PC (llm_alone_dev) may include those calls: ``complete: false``. Jobs that finished before the
  call log existed (22 Oct 2026) can only be summarised this way; the agents add their own run record
  (``job_record``: every call, reuse included, from ``e6_agent.run``'s usage).

Only the columns k, tag, model, usage and latency are read (never an answer); tags of the closed test period
(``codex.REFUSED_TAGS``) are refused; case rows pass ``guard.dev_rows_only``. USD stays null (pricing is a separate
decision). ``input_tokens`` and ``cached_input_tokens`` are the Codex CLI usage fields as reported (``turn.completed``).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uol.final import codex, context, guard, harness, registry

USAGE = ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")
SYSTEMS: dict[str, dict[str, Any]] = {
    "llm_alone": {"tags": ("llm_alone_dev",), "rows": "llm/llm_alone/rows.npy", "shared_tag": True},
    "llm_rerank_S0": {"prefix": "final_L1_S0_", "rows": "llm/llm_rerank_S0/rows.npy", "base": "S0"},
    "llm_rerank_S_full_ext": {"prefix": "final_L1_S_full_ext_", "rows": "llm/llm_rerank_S_full_ext/rows.npy",
                              "base": "S_full_ext"},
    "llm_rerank_S_full_ext_blank": {"prefix": "final_L1blank_S_full_ext_",                       # C5 (D93)
                                    "rows": "llm/llm_rerank_S_full_ext_blank/rows.npy", "base": "S_full_ext"},
    "agent_words": {"tags": ("final_E6_words",), "rows": "llm/agent_words/rows.npy", "record": "llm/agent_words/meta.json"},
    "agent_embedding": {"tags": ("final_E6_embedding",), "rows": "llm/agent_embedding/rows.npy",
                        "record": "llm/agent_embedding/meta.json"},
    "hyde": {"tags": ("final_hyde",), "rows_jsonl": "llm/hyde/texts.jsonl"},
    "query2doc": {"tags": ("L2",), "issue_file": True},
}
FINETUNED = ("bge_m3_ft", "ce_trained")         # GPU fine-tunes without a cost job: record without latency


def cache_path() -> Path:
    prof = context.active()
    if prof.kind != "local":
        return prof.work / "luna" / "cache.sqlite"        # codex.bind()
    from uol.round3 import luna
    return Path(luna.CACHE)


def issue_file() -> Path:
    """The L2 decompositions query2doc read (``gpu_prep.qwen_texts`` -> ``l2_features.luna_issues('a')``)."""
    from uol.round3 import l2_features
    return Path(l2_features.OUT) / "luna" / "L2_a.jsonl"


def _refuse(tags: Iterable[str]) -> None:
    bad = sorted({t for t in tags if codex.REFUSED_TAGS.match(t or "")})
    if bad:
        raise guard.TestSetRefused(f"LLM cost: refused tags {bad[:5]} (closed test period or probes)")


def _match(spec: dict[str, Any], tag: str) -> bool:
    return tag in spec.get("tags", ()) or bool(spec.get("prefix") and str(tag).startswith(spec["prefix"]))


def read_call_log(path: Path | None = None) -> list[dict[str, Any]]:
    path = Path(path) if path else codex.call_log_path()
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:                  # a torn last line of an interrupted job
                continue
            if not codex.REFUSED_TAGS.match(str(r.get("tag") or "")):
                out.append(r)
    return out


def cache_rows(path: Path, *, tags: Iterable[str] = (), prefix: str | None = None,
               keys: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """Usage rows of the LLM cache (read-only; never the output column)."""
    tags = tuple(tags)
    con = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        sel = "select k, tag, model, usage, latency from r where "
        if keys is not None:
            ks, rows = list(dict.fromkeys(keys)), []
            for s in range(0, len(ks), 500):
                chunk = ks[s:s + 500]
                rows += con.execute(sel + f"k in ({','.join('?' * len(chunk))})", chunk).fetchall()
        else:
            q, args = [], []
            if tags:
                q.append(f"tag in ({','.join('?' * len(tags))})")
                args += list(tags)
            if prefix:
                q.append("substr(tag, 1, ?) = ?")
                args += [len(prefix), prefix]
            if not q:
                return []
            rows = con.execute(sel + " or ".join(q), args).fetchall()
    finally:
        con.close()
    _refuse(r[1] for r in rows)
    return [{"key": r[0], "tag": r[1], "model": r[2], "usage": json.loads(r[3] or "{}"), "latency": r[4], "cached": None}
            for r in rows]


def totals(calls: list[dict[str, Any]], n_cases: int | None) -> dict[str, Any]:
    tot = {k: 0 for k in USAGE}
    secs, n_lat = 0.0, 0
    for c in calls:
        u = c.get("usage") or {}
        for k in USAGE:
            tot[k] += int(u.get(k, 0) or 0)
        if c.get("latency") is not None:
            secs += float(c["latency"])
            n_lat += 1
    out: dict[str, Any] = {"calls": len(calls), **tot, "seconds": round(secs, 1), "calls_with_latency": n_lat}
    if n_cases:
        out["per_case"] = {"calls": round(len(calls) / n_cases, 4), **{k: round(tot[k] / n_cases, 1) for k in USAGE},
                           "seconds": round(secs / n_cases, 3)}
    return out


def cases(spec: dict[str, Any], work: Path) -> tuple[list[int] | None, str]:
    """The development rows the system's calls served (guarded) and what they are."""
    if spec.get("rows") and (work / spec["rows"]).exists():
        rows = guard.dev_rows_only(np.load(work / spec["rows"]), "LLM cost rows")
        return [int(r) for r in rows], "DEV evaluation cases of the system's score matrix"
    if spec.get("rows_jsonl") and (work / spec["rows_jsonl"]).exists():
        rows = [int(json.loads(x)["row"]) for x in (work / spec["rows_jsonl"]).read_text(encoding="utf-8").splitlines() if x.strip()]
        return [int(r) for r in guard.dev_rows_only(rows, "LLM cost rows")], \
            "development rows given a hypothetical article (training rows too: the texts are label-free ranker inputs)"
    if spec.get("issue_file"):
        p = issue_file()
        if p.exists():
            rows = [int(json.loads(x)["row"]) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            return [int(r) for r in guard.dev_rows_only(rows, "L2 issue rows")], \
                "development rows given an issue decomposition (every development row)"
    return None, "case rows not found"


def summarise(sid: str) -> dict[str, Any]:
    spec = SYSTEMS[sid]
    prof = context.active()
    rows, cases_basis = cases(spec, prof.work)
    n = len(rows) if rows else None
    out: dict[str, Any] = {"n_cases": n, "cases_basis": cases_basis, "usd": None,
                           "usd_basis": "no cited price recorded (pricing is a separate decision)"}
    if spec.get("base"):
        out["base_ranker"] = spec["base"]
        out["base_note"] = f"the LLM stage only; the full per-case cost adds cost/{spec['base']}.json (its top 30 is reranked)"
    logged = [r for r in read_call_log() if _match(spec, str(r.get("tag") or ""))]
    cache = cache_path()
    if logged:
        by_key: dict[str, dict[str, Any]] = {}
        for r in logged:
            k = r["key"]
            if k not in by_key:
                by_key[k] = {**r, "fresh": not r.get("cached")}
            else:
                by_key[k]["fresh"] = by_key[k]["fresh"] or not r.get("cached")
        calls = list(by_key.values())
        out.update(basis="call_log", complete=True, **totals(calls, n), fresh_calls=sum(c["fresh"] for c in calls),
                   reused_calls=sum(not c["fresh"] for c in calls), log_lines=len(logged),
                   models=sorted({str(c.get("model")) for c in calls}),
                   note="every answered call once per distinct cache key, reuse included; usage and latency are those "
                        "stored with the answer; seconds = sum of per-call wall time (calls ran 6-12 at a time)")
    elif not cache.exists():
        out.update(basis="unavailable", complete=False, reason=f"no call log and no LLM cache at {cache}")
    elif spec.get("issue_file"):
        p = issue_file()
        keys = [json.loads(x)["luna_key"] for x in p.read_text(encoding="utf-8").splitlines() if x.strip()] \
            if p.exists() else []
        uniq = list(dict.fromkeys(k for k in keys if k))
        calls = cache_rows(cache, keys=uniq) if uniq else []
        out.update(basis="issue_file_keys", complete=bool(uniq) and len(calls) == len(uniq), keys=len(uniq),
                   keys_found=len(calls), **totals(calls, n), models=sorted({str(c.get("model")) for c in calls}),
                   note="calls made on the thesis PC (tag L2, variant a, sample 0) whose issues query2doc encodes; "
                        "query2doc makes no LLM call on the run computer")
    else:
        calls = cache_rows(cache, tags=spec.get("tags", ()), prefix=spec.get("prefix"))
        out.update(basis="cache_rows_by_tag", complete=False, **totals(calls, n),
                   models=sorted({str(c.get("model")) for c in calls}),
                   note="cache rows created under the system's tag only: answers reused from the seeded thesis-PC cache "
                        "keep their original tag and are not counted" +
                        ("; the tag is also the thesis-PC run's, so its rows may be included" if spec.get("shared_tag") else ""))
    if spec.get("record") and (prof.work / spec["record"]).exists():
        u = json.loads((prof.work / spec["record"]).read_text(encoding="utf-8")).get("usage") or {}
        if u:
            rec = {k: u.get(k) for k in ("calls", "input_tokens", "output_tokens", "seconds")}
            if n:
                rec["per_case"] = {k: round(float(v) / n, 3) for k, v in rec.items() if v is not None}
            out["job_record"] = {**rec, "basis": "e6_agent.run usage (every batch call, reuse included; split re-asks "
                                                 "and cached-token counts not recorded)"}
    return out


def used_by(sid: str) -> list[str]:
    out = []
    for x, s in registry.systems().items():
        a = s.get("adapter_args", {}) or {}
        if x != sid and (sid in (a.get("members") or []) or sid in (a.get("extra") or [])):
            out.append(x)
    return sorted(out)


def _with_llm(store: Path, sid: str, llm: dict[str, Any]) -> dict[str, Any]:
    from uol.final import cost
    p = store / "cost" / f"{sid}.json"
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else cost.record(sid, store)
    rec["llm"] = llm
    harness.write_json(p, rec)
    return rec


def write_all(store: Path | None = None) -> dict[str, Any]:
    """Cost records with the ``llm`` field for every LLM system, and records for the GPU fine-tunes that have no
    cost job. Never raises for one system: its record says what failed."""
    from uol.final import cost
    store = Path(store) if store else context.active().store
    out: dict[str, Any] = {}
    for sid in SYSTEMS:
        try:
            llm = summarise(sid)
        except guard.TestSetRefused as exc:
            llm = {"basis": "refused", "complete": False, "reason": str(exc)[:300]}
        except Exception as exc:
            llm = {"basis": "error", "complete": False, "reason": f"{type(exc).__name__}: {exc}"[:300]}
        try:
            llm["used_by"] = used_by(sid)
            rec = _with_llm(store, sid, llm)
            out[sid] = {"basis": rec["llm"].get("basis"), "calls": rec["llm"].get("calls"),
                        "per_case": rec["llm"].get("per_case")}
        except Exception as exc:
            out[sid] = {"basis": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    for sid in FINETUNED:
        try:
            if not (store / "cost" / f"{sid}.json").exists():
                rec = cost.record(sid, store)
                out[sid] = {"gpu_hours_finetune": rec["training"].get("gpu_hours_finetune")}
        except Exception as exc:
            out[sid] = {"basis": "error", "reason": f"{type(exc).__name__}: {exc}"[:300]}
    harness.write_json(store / "llm" / "cost_summary.json", out)
    return out
