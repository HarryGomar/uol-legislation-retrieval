"""Shared GLM client for the Round-2 LLM experiments (L1 restatement, L2 listwise selection, L3 lay stress test).

Authority: author chat 2026-10-05 ("using GLM flash ... use the quota in my plan, make it work"). Default route: the
author's GLM Coding Plan (Lite) endpoint, model ``glm-5.3-flash`` with low reasoning effort; second generator ``glm-4.7``.
The author was told that Z.ai's FAQ limits the plan to supported coding tools and chose to use it. ``ROUTE="free"``
switches to the general API's free ``glm-4.7-flash`` (thinking disabled, about 2 concurrent requests per account).
A quota guard reads the plan's usage endpoint and pauses (never drains) when fewer than ``RESERVE_WEEKLY`` weekly or
``RESERVE_5H`` five-hour credits remain, sleeping until the reported reset.
Only public text is sent (masked facts, public article text); labels never leave the machine. The key is read from
``ZAI_API_KEY`` and never logged.

The account allows about two concurrent requests, so every process shares a cross-process limiter of ``SLOTS`` lock
files (``msvcrt`` byte locks are released automatically if a process dies). Responses are cached in SQLite keyed by
sha256(model, params, messages); a restart never re-sends a cached request. Each call is tagged for accounting.

Usage::

    from uol.round2.glm_client import chat, restate_prompt, lay_prompt, scrub_norms
    text = chat(restate_prompt(facts), tag="L1")
"""
from __future__ import annotations

import hashlib
import json
import msvcrt
import os
import random
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import requests

from uol.facts_to_law.spec import ROOT

ROUTE = os.environ.get("UOL_GLM_ROUTE", "plan")
URL = {"plan": "https://api.z.ai/api/coding/paas/v4/chat/completions",
       "free": "https://api.z.ai/api/paas/v4/chat/completions"}[ROUTE]
QUOTA_URL = "https://api.z.ai/api/monitor/usage/quota/limit"
MODEL = {"plan": "glm-5.3-flash", "free": "glm-4.7-flash"}[ROUTE]
MODEL_2 = {"plan": "glm-4.7", "free": "glm-4.5-flash"}[ROUTE]
SLOTS = {"plan": 6, "free": 2}[ROUTE]
RESERVE_WEEKLY = 1000
RESERVE_5H = 150
QUOTA_EVERY = 40
GLM_DIR = Path(ROOT) / "experiments" / "runs" / "round2_20261004" / "glm"
DB = GLM_DIR / "cache.sqlite"
LOCKS = GLM_DIR / f"locks_{os.environ.get('UOL_GLM_ROUTE', 'plan')}"
TIMEOUT = 180
MAX_TRIES = 8


_SCHEMA = ("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, tag TEXT, request TEXT, content TEXT,"
           " prompt_tokens INT, completion_tokens INT, utc TEXT);"
           "CREATE TABLE IF NOT EXISTS calls (utc TEXT, tag TEXT, model TEXT, status INT, seconds REAL);")


def _db() -> sqlite3.Connection:
    GLM_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(60):
        try:
            con = sqlite3.connect(DB, timeout=60)
            if con.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
                con.execute("PRAGMA journal_mode=WAL")
            con.executescript(_SCHEMA)
            return con
        except sqlite3.OperationalError:
            time.sleep(0.5 + random.random())
    raise RuntimeError("GLM cache database stayed locked")


@contextmanager
def _slot():
    LOCKS.mkdir(parents=True, exist_ok=True)
    while True:
        for k in range(SLOTS):
            f = open(LOCKS / f"slot{k}.lock", "a+b")
            try:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                f.close()
                continue
            try:
                yield
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                f.close()
            return
        time.sleep(0.2 + random.random() * 0.3)


def _key(model: str, messages: list[dict], params: dict) -> str:
    blob = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def chat(prompt: str | list[dict], *, tag: str, model: str = MODEL, max_tokens: int = 800,
         temperature: float = 0.0) -> str:
    """Return the assistant text for ``prompt`` (cached). Raises RuntimeError after MAX_TRIES failures."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    if model.startswith("glm-5"):
        params = {"max_tokens": max_tokens, "temperature": temperature, "thinking": {"type": "enabled"},
                  "reasoning_effort": "low"}
    else:
        params = {"max_tokens": max_tokens, "temperature": temperature, "thinking": {"type": "disabled"}}
    key = _key(model, messages, params)
    con = _db()
    try:
        row = con.execute("SELECT content FROM cache WHERE key=?", (key,)).fetchone()
        if row is not None:
            return row[0]
        headers = {"Authorization": "Bearer " + os.environ["ZAI_API_KEY"], "Content-Type": "application/json"}
        body = {"model": model, "messages": messages, **params}
        delay = 2.0
        last = ""
        _quota_guard(con)
        for _ in range(MAX_TRIES):
            with _slot():
                t0 = time.time()
                try:
                    r = requests.post(URL, headers=headers, json=body, timeout=TIMEOUT)
                    status = r.status_code
                except requests.RequestException as exc:
                    status, r, last = -1, None, str(exc)[:200]
                con.execute("INSERT INTO calls VALUES (?,?,?,?,?)",
                            (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), tag, model, status, time.time() - t0))
                con.commit()
            if status == 200:
                d = r.json()
                content = (d["choices"][0]["message"].get("content") or "").strip()
                u = d.get("usage") or {}
                con.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?,?,?,?)",
                            (key, model, tag, json.dumps(body, ensure_ascii=False), content,
                             u.get("prompt_tokens"), u.get("completion_tokens"),
                             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
                con.commit()
                return content
            if r is not None:
                try:
                    last = json.dumps(r.json())[:200]
                except ValueError:
                    last = r.text[:200]
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 60)
        raise RuntimeError(f"GLM call failed after {MAX_TRIES} tries: {last}")
    finally:
        con.close()


def quota() -> list[dict]:
    """Plan credit windows: [{'window': '5h'|'week', 'limit', 'used', 'remaining', 'reset_utc'}]."""
    headers = {"Authorization": "Bearer " + os.environ["ZAI_API_KEY"]}
    data = requests.get(QUOTA_URL, headers=headers, timeout=30).json()["data"]["limits"]
    out = []
    for lim in data:
        out.append({"window": "5h" if lim.get("unit") == 3 else "week", "limit": lim["usage"], "used": lim["currentValue"],
                    "remaining": lim["usage"] - lim["currentValue"], "reset_epoch_ms": lim.get("nextResetTime")})
    return out


def _quota_guard(con: sqlite3.Connection) -> None:
    if ROUTE != "plan":
        return
    n = con.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    if n % QUOTA_EVERY:
        return
    while True:
        try:
            q = {w["window"]: w for w in quota()}
        except Exception:
            return
        low = [w for k, w in q.items() if w["remaining"] < (RESERVE_5H if k == "5h" else RESERVE_WEEKLY)]
        (GLM_DIR / "quota.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                        "windows": q, "paused": bool(low)}, indent=1))
        if not low:
            return
        wake = max(w["reset_epoch_ms"] or 0 for w in low) / 1000
        time.sleep(max(60.0, min(wake - time.time() + 30, 3600.0)))


def accounting() -> dict:
    """Per-tag call counts, statuses and token totals."""
    con = _db()
    try:
        calls = con.execute("SELECT tag, model, status, COUNT(*), SUM(seconds) FROM calls GROUP BY tag, model, status").fetchall()
        toks = con.execute("SELECT tag, model, COUNT(*), SUM(prompt_tokens), SUM(completion_tokens) FROM cache GROUP BY tag, model").fetchall()
    finally:
        con.close()
    return {"calls": [dict(zip(("tag", "model", "status", "n", "seconds"), c)) for c in calls],
            "cached": [dict(zip(("tag", "model", "n", "prompt_tokens", "completion_tokens"), t)) for t in toks]}


# --------------------------------------------------------------------------- fixed prompts (shared by L1 and L3)

RESTATE_V1 = (
    "Eres un asistente de investigación jurídica. A partir de los HECHOS de un asunto judicial mexicano, formula de 1 a 3 "
    "cuestiones jurídicas concretas que el tribunal tendría que resolver. Escribe en español, en términos jurídicos "
    "generales. NO menciones números de artículos, nombres de leyes, códigos, reglamentos ni la Constitución, ni tesis o "
    "jurisprudencias. Devuelve solo las cuestiones, una por línea, sin numeración ni comentarios.\n\nHECHOS:\n{facts}"
)

LAY_V1 = (
    "Reescribe los siguientes HECHOS, redactados por un tribunal mexicano, como el relato que una persona sin formación "
    "jurídica le contaría a un abogado en una primera consulta, en primera persona y en español coloquial. Conserva todos "
    "los hechos, fechas, cantidades y participantes; no agregues hechos nuevos. NO uses vocabulario técnico jurídico "
    "(por ejemplo: amparo, quejoso, tercero interesado, acto reclamado, litis, agravio, juicio, autoridad responsable, "
    "recurso, sentencia, resolución, demanda, procedente, improcedente), no menciones leyes, artículos ni tribunales por su "
    "nombre técnico. Devuelve solo el relato.\n\nHECHOS:\n{facts}"
)

PARAPHRASE_V1 = (
    "Parafrasea los siguientes HECHOS de un asunto judicial mexicano conservando el mismo registro técnico y todo su "
    "contenido: mismos hechos, fechas, cantidades, participantes y términos jurídicos, con distinta redacción. No agregues "
    "ni elimines información. Devuelve solo el texto parafraseado.\n\nHECHOS:\n{facts}"
)


def restate_prompt(facts: str) -> str:
    return RESTATE_V1.format(facts=facts)


def lay_prompt(facts: str) -> str:
    return LAY_V1.format(facts=facts)


def paraphrase_prompt(facts: str) -> str:
    return PARAPHRASE_V1.format(facts=facts)


# --------------------------------------------------------------------------- deterministic scrubber for norm mentions

_NORM_PATTERNS = [
    # case-insensitive: article numbers, fracciones, tesis identifiers
    r"(?i:\b(?:art[íi]culos?|arts?\.|numerales?|precepto)\s*\d+[\w\-°º]*(?:\s*(?:,|y|al|a)\s*\d+[\w\-°º]*)*)",
    r"(?i:\bfracci(?:ón|on|ones)\s+[IVXLCDM]+\b)",
    r"(?i:\b(?:tesis|jurisprudencia)\s+[\w./\-]*\d?[\w./\-()]*(?:\s*\d+/\d{4})?)",
    r"(?i:\b(?:[IVX]+\.|\d+[a-z]?\.)?\s*\d*[a-z]?\./?J\.\s*\d+/\d{4})",          # 2a./J. 15/2020, P./J. 3/2021
    r"(?i:\b[IVX]+\.\d+[a-z]?\.[A-Z]\.?\s*\d+\s*[A-Z]+)",                           # I.3o.C. 45 K style aisladas
    r"(?i:\(\s*\d+a\.\s*\))",                                                       # (10a.), (11a.)
    r"(?i:\bregistro\s+digital:?\s*\d+)",
    # case-sensitive: a capitalised instrument name followed by capitalised words and connectors
    r"\b(?:Ley|Código|Codigo|Reglamento|Estatuto|Constitución|Constitucion|Norma Oficial|Decreto)\b"
    r"(?:\s+(?:[A-ZÁÉÍÓÚÑ][\wáéíóúñ]*|de|del|la|las|los|y|para|en|sobre|el)(?=\s|$|[,.;:]))*",
    r"\b(?:CPEUM|LFT|CFF|CNPP|CPF|CCF|CFPC|LISR|LIVA|LSS|LGTOC|LGSM)\b",
]
_NORM_RE = re.compile("|".join(_NORM_PATTERNS))


def scrub_norms(text: str) -> tuple[str, int]:
    """Remove explicit statute / article / tesis mentions. Returns (clean text, number of removals)."""
    n = 0

    def _sub(m: re.Match) -> str:
        nonlocal n
        n += 1
        return " "

    clean = _NORM_RE.sub(_sub, text)
    return re.sub(r"[ \t]{2,}", " ", clean).strip(), n
