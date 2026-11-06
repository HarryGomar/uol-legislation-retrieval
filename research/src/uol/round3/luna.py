"""Codex CLI as the Round-3 LLM backend: ``gpt-5.6-luna`` through ``codex exec``.

Authority: author chat 2026-10-13 ("the experiments should be processed like the actual LLM processing: use my codex
cli with luna"). Every Round-3 LLM call (listwise reranking, issue decomposition, lay restatement, synthetic facts,
agent steps) goes through this module, under the author's Codex (ChatGPT) sign-in. No API key is read or logged.

Isolation. ``codex exec`` is an agent, so each call is boxed in to make it a pure text-in, JSON-out model call:
* ``--ignore-user-config`` drops the user's MCP servers, notify hook and profile; auth still comes from CODEX_HOME;
* tool features are disabled (shell, exec, apps, plugins, browser, computer use, images, sub-agents, skills, hooks);
* ``--sandbox read-only``, ``--ephemeral``, an empty per-worker working directory, and stdin closed;
* ``--json`` events are parsed: a response is rejected (``LunaToolUse``) if any item other than the final message or
  reasoning appears, and the worker directory must still be empty afterwards.
Prompts carry only public text (masked facts, public article text); labels never leave the machine.

Verified 2026-10-13 with codex-cli 0.155.1: a 3-item batch returned schema-valid JSON in about 6 s, using about 13.6k
input tokens, almost all of it fixed agent overhead. Always batch many items per call (``pack``).

Usage::

    from uol.round3 import luna
    out = luna.call(prompt, schema, effort="low", tag="L2")          # cached, ledgered
    outs = luna.run_many([(prompt, schema), ...], effort="low", tag="L2", workers=4)
    python -m uol.round3.luna smoke                                    # one 3-item live test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[3]
MODEL = "gpt-5.6-luna"
MODEL_KNOWLEDGE_CUTOFF = "2026-02-16"   # developers.openai.com/api/docs/models/gpt-5.6-luna, read 2026-10-13
EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "apps", "plugins", "browser_use", "browser_use_external", "computer_use",
    "image_generation", "multi_agent", "skill_search", "hooks", "view_image", "tool_suggest", "goals",
    "in_app_browser", "sleep_tool", "shell_snapshot", "workspace_dependencies", "remote_plugin",
)
ALLOWED_ITEMS = {"agent_message", "reasoning"}
QUOTA_RE = re.compile(r"rate.?limit|usage.?limit|quota|too many requests|\b429\b|try again (in|later)", re.I)
CACHE = Path(os.environ.get("UOL_LUNA_CACHE") or ROOT / ".cache" / "round3_luna" / "cache.sqlite")
LEDGER = ROOT / "research" / "state" / "round3_best_case_20261013" / "LUNA_LEDGER.jsonl"
ISO_ROOT = Path(tempfile.gettempdir()) / "uol_luna_iso" / f"p{os.getpid()}"  # runs side by side never share a folder
_lock = threading.Lock()


class LunaError(RuntimeError):
    """A call failed and may be retried."""


class LunaQuotaError(LunaError):
    """The Codex plan reported a rate or usage limit: pause, never treat as missing data."""


class LunaToolUse(LunaError):
    """The agent used a tool or wrote a file: the response is rejected."""


NPM_NATIVE = (Path(os.environ.get("APPDATA", "")) / "npm" / "node_modules" / "@openai" / "codex" / "node_modules"
              / "@openai" / "codex-win32-x64" / "vendor" / "x86_64-pc-windows-msvc" / "bin" / "codex.exe")


def codex_bin() -> str:
    """The native codex.exe: a .cmd shim would route arguments through cmd.exe quoting."""
    env = os.environ.get("UOL_CODEX_BIN")
    if env:
        return env
    if NPM_NATIVE.exists():
        return str(NPM_NATIVE)
    found = shutil.which("codex.exe")
    if found:
        return found
    raise FileNotFoundError("native codex.exe not found; set UOL_CODEX_BIN")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def key_of(prompt: str, schema: dict, model: str, effort: str, sample: int) -> str:
    blob = json.dumps({"m": model, "e": effort, "s": schema, "p": prompt, "k": sample}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _db() -> sqlite3.Connection:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(CACHE, timeout=60)
    con.execute("create table if not exists r (k text primary key, tag text, model text, effort text, utc text,"
                " output text, usage text, latency real)")
    return con


def cached(key: str) -> dict | None:
    with _lock, _db() as con:
        row = con.execute("select output, usage, latency, utc from r where k=?", (key,)).fetchone()
    if row is None:
        return None
    return {"output": json.loads(row[0]), "usage": json.loads(row[1]), "latency": row[2], "utc": row[3], "cached": True}


def _store(key: str, tag: str, model: str, effort: str, res: dict) -> None:
    with _lock, _db() as con:
        con.execute("insert or replace into r values (?,?,?,?,?,?,?,?)",
                    (key, tag, model, effort, res["utc"], json.dumps(res["output"], ensure_ascii=False),
                     json.dumps(res["usage"]), res["latency"]))


def _ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_events(text: str) -> tuple[str | None, dict, list[str]]:
    """Return (final agent message, usage, disallowed item types) from ``codex exec --json`` output."""
    message, usage, bad = None, {}, []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("type", "")
        if kind == "item.completed":
            item = ev.get("item", {})
            itype = item.get("type")
            if itype == "agent_message":
                message = item.get("text")
            elif itype not in ALLOWED_ITEMS:
                bad.append(str(itype))
        elif kind == "turn.completed":
            usage = ev.get("usage", {}) or {}
        elif kind in ("turn.failed", "error"):
            bad.append(kind + ":" + json.dumps(ev, ensure_ascii=False)[:300])
    return message, usage, bad


def _workdir(worker: int) -> Path:
    d = ISO_ROOT / f"w{worker}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def _raw_call(prompt: str, schema: dict, *, model: str, effort: str, worker: int, timeout: int) -> dict:
    wd = _workdir(worker)
    schema_path = ISO_ROOT / f"schema_w{worker}.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    cmd = [codex_bin(), "exec", "-m", model, "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
           "--ignore-user-config", "-c", f'model_reasoning_effort="{effort}"', "-c", 'web_search="disabled"',
           "--output-schema", str(schema_path),
           "--json", "--color", "never", "-C", str(wd)]
    for feat in DISABLED_FEATURES:
        cmd += ["--disable", feat]
    cmd.append("-")  # prompt on stdin
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise LunaError(f"timeout after {timeout}s") from exc
    latency = time.perf_counter() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or "")[-600:] + (proc.stdout or "")[-600:]
        if QUOTA_RE.search(tail):
            raise LunaQuotaError(tail)
        raise LunaError(f"exit {proc.returncode}: {tail}")
    message, usage, bad = parse_events(proc.stdout)
    leftovers = list(wd.iterdir())
    tool = [b for b in bad if not b.startswith(("error:", "turn.failed:"))]
    if tool or leftovers:
        raise LunaToolUse(f"disallowed items {tool}; files written {[p.name for p in leftovers]}")
    if bad:                                   # transport/turn errors (e.g. a stream reconnect): retriable, not tool use
        raise LunaError(f"turn error events {bad}")
    if not message:
        raise LunaError("no final message")
    try:
        output = json.loads(message)
    except json.JSONDecodeError as exc:
        raise LunaError(f"final message is not JSON: {message[:200]}") from exc
    return {"output": output, "usage": usage, "latency": round(latency, 3), "utc": _utc(), "cached": False}


def call(prompt: str, schema: dict, *, effort: str = "low", tag: str, model: str = MODEL, sample: int = 0,
         worker: int = 0, timeout: int = 900, retries: int = 4) -> dict:
    """One cached, ledgered call. ``sample`` > 0 requests an independent re-sample of the same prompt."""
    if effort not in EFFORTS:
        raise ValueError(effort)
    key = key_of(prompt, schema, model, effort, sample)
    hit = cached(key)
    if hit is not None:
        return hit
    delay = 20.0
    for attempt in range(retries + 1):
        try:
            res = _raw_call(prompt, schema, model=model, effort=effort, worker=worker, timeout=timeout)
            _store(key, tag, model, effort, res)
            _ledger({"utc": res["utc"], "tag": tag, "key": key, "model": model, "effort": effort, "ok": True,
                     "latency": res["latency"], "usage": res["usage"], "attempt": attempt})
            return res
        except LunaQuotaError:
            _ledger({"utc": _utc(), "tag": tag, "key": key, "model": model, "ok": False, "error": "quota"})
            raise
        except LunaError as exc:
            _ledger({"utc": _utc(), "tag": tag, "key": key, "model": model, "ok": False,
                     "error": type(exc).__name__, "detail": str(exc)[:300], "attempt": attempt})
            if attempt == retries:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 600)
    raise AssertionError("unreachable")


def run_many(jobs: Sequence[tuple[str, dict]], *, effort: str = "low", tag: str, workers: int = 4,
             model: str = MODEL, sample: int = 0) -> list[dict | None]:
    """Run jobs concurrently. On a quota error, stop submitting, finish in-flight calls and re-raise:
    the caller resumes later (the cache makes reruns free). Never returns a partial result silently."""
    results: list[dict | None] = [None] * len(jobs)
    stop = threading.Event()
    slots = list(range(workers))
    slot_lock = threading.Lock()

    def one(i: int) -> tuple[int, dict]:
        if stop.is_set():
            raise LunaQuotaError("stopped after an earlier quota error")
        with slot_lock:
            w = slots.pop()
        try:
            return i, call(jobs[i][0], jobs[i][1], effort=effort, tag=tag, model=model, sample=sample, worker=w)
        finally:
            with slot_lock:
                slots.append(w)

    quota: Exception | None = None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(one, i) for i in range(len(jobs))]
        for fut in as_completed(futs):
            try:
                i, res = fut.result()
                results[i] = res
            except LunaQuotaError as exc:
                stop.set()
                quota = exc
    if quota is not None:
        raise LunaQuotaError(f"{sum(r is None for r in results)} of {len(jobs)} jobs pending: {quota}")
    return results


def pack(items: Iterable[dict], size: int) -> list[list[dict]]:
    """Split items into batches of ``size`` (about 20 amortises the ~13k-token per-call agent overhead)."""
    items = list(items)
    return [items[i:i + size] for i in range(0, len(items), size)]


def batch_schema(item_schema: dict) -> dict:
    """Schema for a batch response: {"results": [{"id": str, ...item fields}]}."""
    props = {"id": {"type": "string"}, **item_schema["properties"]}
    return {"type": "object", "additionalProperties": False, "required": ["results"],
            "properties": {"results": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["id", *item_schema["required"]],
                "properties": props}}}}


def check_batch(res: dict, ids: Sequence[str]) -> dict[str, dict]:
    """Map id -> result and require exactly the requested ids (else LunaError, so the batch is retried)."""
    got = {r["id"]: r for r in res["output"].get("results", [])}
    if set(got) != set(ids):
        raise LunaError(f"batch ids mismatch: missing {sorted(set(ids) - set(got))[:5]}, extra {sorted(set(got) - set(ids))[:5]}")
    return got


def _smoke() -> None:
    item = {"type": "object", "additionalProperties": False, "required": ["issues"],
            "properties": {"issues": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}}}}
    cases = {"a": "Una trabajadora fue despedida al regresar de su incapacidad por maternidad; el patrón alegó contrato por tiempo determinado.",
             "b": "Un contribuyente impugnó una multa fiscal impuesta sin notificación previa del requerimiento.",
             "c": "Una persona promovió amparo contra la negativa de inscribirla a un programa social; el juez sobreseyó por falta de pruebas."}
    prompt = ("Tarea: para cada caso, enumera de 2 a 4 cuestiones jurídicas en registro técnico, sin citar leyes ni números "
              "de artículo. Responde solo con el JSON pedido, un resultado por id.\n"
              + "\n".join(f"[id={k}] {v}" for k, v in cases.items()))
    res = call(prompt, batch_schema(item), effort="low", tag="smoke", sample=int(time.time()))
    got = check_batch(res, list(cases))
    print(json.dumps({"ok": True, "latency": res["latency"], "usage": res["usage"], "ids": sorted(got)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke"])
    args = ap.parse_args()
    _smoke()


def complete_batch(res: dict, ids: Sequence[str], schema: dict) -> dict[str, dict]:
    """``check_batch`` plus the fields the item schema requires: an answer that names the right ids but omits a
    required field (seen in the final run, 21 Oct 2026: a cached llm_alone answer without "articulos") is treated
    like an id mismatch, so it is resampled instead of failing later with a KeyError."""
    got = check_batch(res, ids)
    required = schema["properties"]["results"]["items"].get("required", [])
    bad = sorted(i for i, r in got.items() if any(k not in r for k in required))
    if bad:
        raise LunaError(f"batch items missing required fields: {bad[:5]}")
    return got


def checked(res: dict, prompt: str, schema: dict, ids: Sequence[str], *, effort: str, tag: str,
            model: str = MODEL, tries: int = 3) -> dict[str, dict]:
    """``complete_batch``; on an id mismatch or a missing required field, request independent samples
    (sample=1..tries) of the same prompt."""
    try:
        return complete_batch(res, ids, schema)
    except LunaError:
        last: Exception | None = None
        for s in range(1, tries + 1):
            try:
                return complete_batch(call(prompt, schema, effort=effort, tag=tag, model=model, sample=s), ids, schema)
            except LunaQuotaError:
                raise
            except LunaError as exc:
                last = exc
        raise LunaError(f"batch still invalid after {tries} resamples: {last}")


def checked_or_split(res: dict, prompt: str, schema: dict, ids: Sequence[str], single, *, effort: str, tag: str,
                     model: str = MODEL, tries: int = 3, stats: dict | None = None) -> dict[str, dict]:
    """``checked``; when the batch is still invalid after its resamples (22 Oct 2026, llm__agent_words: the model kept
    answering 'e6w3911' for 'e6w3915'), the batch is split and every case is asked alone.

    ``single(id) -> (prompt, id_in_prompt)`` is the caller's one-case prompt built with the same template and rules
    (and the same prompt guard). Each one-case answer goes through ``checked`` and is matched by its id only; an
    answer is never assigned to a case by position. ``stats`` counts ``split_batches`` and ``split_cases``."""
    try:
        return checked(res, prompt, schema, ids, effort=effort, tag=tag, model=model, tries=tries)
    except LunaQuotaError:
        raise
    except LunaError as exc:
        if len(ids) < 2:
            raise
        first = exc
    if stats is not None:
        stats["split_batches"] = stats.get("split_batches", 0) + 1
        stats["split_cases"] = stats.get("split_cases", 0) + len(ids)
    out: dict[str, dict] = {}
    for i in ids:
        p1, sid = single(i)
        r1 = call(p1, schema, effort=effort, tag=tag, model=model, sample=0)
        try:
            got = checked(r1, p1, schema, [sid], effort=effort, tag=tag, model=model, tries=tries)
        except LunaQuotaError:
            raise
        except LunaError as exc:
            raise LunaError(f"case {i} still invalid when asked alone ({exc}); the batch failed with: {first}") from exc
        out[i] = {**got[sid], "id": i}
    return out
