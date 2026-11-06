"""The LLM layer of the run computer: ``round3.luna`` (the Codex CLI wrapper) with an adaptive, quota-safe driver.

Everything goes through the unchanged wrapper ``uol.round3.luna``:

* the native ``codex.exe`` (``luna.codex_bin``: ``UOL_CODEX_BIN``, the npm vendor binary, then ``codex.exe`` on PATH;
  never the ``.cmd`` shim); the prompt is passed on stdin, which is closed after it (``subprocess.run(input=...)``);
* read-only sandbox, no tools (``--disable`` of every tool feature), fixed JSON output schema, model pinned to
  ``gpt-5.6-luna`` (``luna.MODEL``); ``smoke`` checks that the model answers valid JSON on this account and the LLM
  block depends on it, so answers of different models are never mixed;
* the answer cache (sqlite, key = sha256 of model, effort, schema, prompt, sample) is seeded from the thesis PC with
  development entries only (``export_cache``: allow-listed tags, every output scanned for test-period record ids), so a
  call is reused exactly when prompt, model and input are identical.

``run_adaptive`` drives ``luna.run_many`` in rounds: concurrency starts at 6 and rises by 2 per clean round up to 12;
on a rate-limit or quota error it halves (minimum 2), pauses (15 min, doubling up to 2 h) and resumes; the cache
makes repeats free. A job never records a missing answer: it either returns every answer or fails.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from uol.final import context

START, STEP, MAX, MIN = 6, 2, 12, 2
PAUSE_S, PAUSE_MAX_S = 900, 7200
ROUND = 48                                   # calls per round before the concurrency is reconsidered
DEV_TAGS = re.compile(r"^(E6_|L1_|L1h_|L2$|L2_pilot$|L2_resample$|llm_alone_dev$)")
REFUSED_TAGS = re.compile(r"^(P7_|L4$|probe_|smoke$)")
SEVEN = re.compile(r"(?<!\d)(\d{7})(?!\d)")


def bind() -> Any:
    """Import ``round3.luna`` with its cache and ledger in the profile's work folder."""
    from uol.round3 import luna
    prof = context.active()
    if prof.kind != "local":
        luna.CACHE = prof.work / "luna" / "cache.sqlite"
        luna.LEDGER = prof.work / "luna" / "LUNA_LEDGER.jsonl"
        seed = prof.data_dir / "luna" / "cache_dev.sqlite" if prof.data_dir else None
        if seed and seed.exists() and not luna.CACHE.exists():
            import shutil
            luna.CACHE.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(seed, luna.CACHE)
        prof.patches["uol.round3.luna.CACHE"] = str(luna.CACHE)
    return luna


def install() -> Any:
    """Route every inherited ``luna.run_many`` / ``luna.call`` through the adaptive, quota-safe driver (once)."""
    luna = bind()
    if getattr(luna, "_uol_final_installed", False):
        return luna
    orig_many, orig_call = luna.run_many, luna.call

    def run_many(jobs, *, effort="low", tag, workers=4, model=luna.MODEL, sample=0):
        if model != luna.MODEL:
            raise RuntimeError(f"model {model!r} requested; the run is pinned to {luna.MODEL}")

        def inner(js, *, effort, tag, workers, sample):
            return orig_many(js, effort=effort, tag=tag, workers=workers, model=model, sample=sample)
        return run_adaptive(jobs, effort=effort, tag=tag, sample=sample, run_many=inner)

    def call(prompt, schema, **kw):
        pause = PAUSE_S
        for _ in range(48):
            try:
                res = orig_call(prompt, schema, **kw)
                log_call(luna, prompt, schema, kw, res)
                return res
            except luna.LunaQuotaError:
                time.sleep(pause)
                pause = min(PAUSE_MAX_S, pause * 2)
        raise RuntimeError("quota still exhausted after 48 pauses; nothing recorded as missing")
    luna._orig_run_many, luna._orig_call = orig_many, orig_call
    luna.run_many, luna.call = run_many, call
    luna._uol_final_installed = True
    return luna


CALL_LOG = "calls.jsonl"
_LOG_LOCK = threading.Lock()


def call_log_path() -> Path:
    return context.active().work / "luna" / CALL_LOG


def log_call(luna: Any, prompt: str, schema: dict, kw: dict, res: dict) -> None:
    """One line per answered call, cache hits included, in ``<work>/luna/calls.jsonl``: the call's tag (the system's),
    its cache key, whether it was reused, and the usage and latency stored with the answer (the cost of producing
    it). Prompts and outputs are not written. Cost plumbing only: a failure here never fails the call (22 Oct 2026)."""
    try:
        model, effort, sample = kw.get("model", luna.MODEL), kw.get("effort", "low"), int(kw.get("sample", 0) or 0)
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tag": kw.get("tag"),
               "key": luna.key_of(prompt, schema, model, effort, sample), "model": model, "effort": effort,
               "sample": sample, "cached": bool(res.get("cached")), "usage": res.get("usage") or {},
               "latency": res.get("latency")}
        path = call_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception as exc:                                # pragma: no cover - never fail an answered call
        print(f"[codex] call log not written: {exc}", flush=True)


def concurrency_bounds() -> tuple[int, int]:
    start = int(os.environ.get("UOL_FINAL_LLM_START", START))
    mx = int(os.environ.get("UOL_FINAL_LLM_MAX", MAX))
    return max(MIN, start), max(start, mx)


def run_adaptive(jobs: Sequence[tuple[str, dict]], *, effort: str, tag: str, sample: int = 0,
                 sleep: Callable[[float], None] = time.sleep, run_many: Callable | None = None,
                 max_pauses: int = 48) -> list[dict]:
    """Every answer of ``jobs`` (prompt, schema), in order; raises if they cannot all be obtained."""
    luna = bind()
    run_many = run_many or luna.run_many
    workers, mx = concurrency_bounds()
    out: list[dict | None] = [None] * len(jobs)
    pause, pauses = PAUSE_S, 0
    log = context.active().work / "luna" / "adaptive.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    while True:
        todo = [i for i, r in enumerate(out) if r is None]
        if not todo:
            return out  # type: ignore[return-value]
        chunk = todo[:max(ROUND, workers * 4)]
        try:
            res = run_many([jobs[i] for i in chunk], effort=effort, tag=tag, workers=workers, sample=sample)
            for i, r in zip(chunk, res):
                out[i] = r
            failed = sum(r is None for r in res)
            _log(log, {"tag": tag, "workers": workers, "done": len(chunk) - failed, "failed": failed})
            if failed == 0:
                workers = min(mx, workers + STEP)
        except luna.LunaQuotaError as exc:
            pauses += 1
            if pauses > max_pauses:
                raise RuntimeError(f"{tag}: quota still exhausted after {max_pauses} pauses; nothing recorded as missing")
            workers = max(MIN, workers // 2)
            _log(log, {"tag": tag, "quota": str(exc)[:200], "pause_s": pause, "workers": workers})
            sleep(pause)
            pause = min(PAUSE_MAX_S, pause * 2)


def _log(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}) + "\n")


def smoke() -> dict[str, Any]:
    """Gate: one call to gpt-5.6-luna returns valid JSON (``luna._smoke`` prompt shape), else the LLM block stops."""
    luna = bind()
    exe = luna.codex_bin()
    schema = {"type": "object", "additionalProperties": False, "required": ["ok"], "properties": {"ok": {"type": "boolean"}}}
    r = luna.call('Answer with the JSON object {"ok": true}.', schema, effort="low", tag="final_smoke", timeout=300)
    ok = r["output"].get("ok") is True
    return {"codex": exe, "model": luna.MODEL, "ok": ok, "cached": r.get("cached"), "usage": r.get("usage")}


# --------------------------------------------------------------------------- bundle time (thesis PC): dev-only cache


def export_cache(src: Path, dst: Path, lock_records: set[str]) -> dict[str, Any]:
    """Copy the cache rows of development tags only; every output is scanned for test-period record ids."""
    from uol.final import guard
    con = sqlite3.connect(src)
    rows = con.execute("select k, tag, model, effort, utc, output, usage, latency from r").fetchall()
    con.close()
    keep, refused = [], {}
    for row in rows:
        tag = row[1] or ""
        if REFUSED_TAGS.match(tag) or not DEV_TAGS.match(tag):
            refused[tag] = refused.get(tag, 0) + 1
            continue
        hits = {m for m in SEVEN.findall(str(row[5])) if m in lock_records}
        if hits:
            raise guard.TestSetRefused(f"cache row {row[0][:12]} ({tag}) names test-period records {sorted(hits)[:3]}")
        keep.append(row)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    out = sqlite3.connect(dst)
    out.execute("create table r (k text primary key, tag text, model text, effort text, utc text, output text, usage text, "
                "latency real)")
    out.executemany("insert into r values (?,?,?,?,?,?,?,?)", keep)
    out.commit()
    out.close()
    return {"kept": len(keep), "refused_by_tag": refused}
