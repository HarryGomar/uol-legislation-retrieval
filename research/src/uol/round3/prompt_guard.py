"""Leakage guard for every Luna prompt (prompt §2; PLAN §3 L rules; DECISION D0.8).

A prompt may carry only the fixed template plus *sanctioned texts*: the masked Hechos of its rows (the frozen v11
query text that every system sees) and public article text. ``check`` removes the sanctioned texts from the prompt and
refuses it if the remainder contains, for any row in the batch:
* the registro digital (record id) or any 7-digit number;
* the publication date (ISO or Spanish long form);
* the rubro (any 8-word window of it);
* the issuing court (``source_organo``);
* an article number (the template and envelope must never name articles).
Overlap between a row's Hechos and its rubro or court is input text, not metadata (D0.8: 1,418 of 4,812 dev Hechos
share an 8-word window with their rubro). The Hechos themselves contain no 7-digit number and no article number
(checked over all dev rows on 2026-10-14).
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from typing import Sequence

from uol.round3 import common as C

MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre",
          "noviembre", "diciembre")
SEVEN = re.compile(r"(?<!\d)\d{7}(?!\d)")
ARTICLE_NO = re.compile(r"(?i)\bart(?:[íi]culo|\.)s?\s*\d+")


class PromptLeak(ValueError):
    """A prompt carries metadata or labels that must never leave the machine."""


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def metadata() -> dict[int, dict]:
    """{release row: {record_id, publication_date, rubro, organo}} from the structured-sources file."""
    d = C.release()
    by_qid = {q.query_id: i for i, q in enumerate(d.queries)}
    out: dict[int, dict] = {}
    with C.SOURCES_JSONL.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            i = by_qid.get(rec.get("query_id"))
            if i is not None:
                out[i] = {"record_id": str(rec.get("record_id") or "").split(":")[-1],
                          "publication_date": rec.get("publication_date") or "",
                          "rubro": rec.get("source_rubro") or "", "organo": rec.get("source_organo") or ""}
    return out


def _date_forms(iso: str) -> list[str]:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", iso or ""):
        return []
    y, m, dd = iso.split("-")
    return [iso, f"{int(dd)} de {MONTHS[int(m) - 1]} de {y}", f"{dd}/{m}/{y}"]


def check(prompt: str, rows: Sequence[int], sanctioned: Sequence[str] = (), template: str | None = None) -> None:
    """Raise PromptLeak when the prompt, minus its sanctioned texts, carries metadata, a 7-digit or article number.

    ``sanctioned`` defaults to the masked Hechos of ``rows``; L1 adds the public article excerpts it shows.
    ``template`` is the fixed, hashed prompt file (with its ``{items}`` placeholder): its static text is sanctioned too,
    so only the per-batch envelope is checked for metadata; the template itself is checked for numbers.
    """
    if template is not None and (SEVEN.search(template) or ARTICLE_NO.search(template)):
        raise PromptLeak("template contains a 7-digit or article number")
    d = C.release()
    allowed = [d.queries[int(r)].text for r in rows] + list(sanctioned)
    rest = prompt
    for part in (template.split("{items}") if template else []):
        if part.strip():
            rest = rest.replace(part, " ")
    for t in sorted(set(allowed), key=len, reverse=True):
        for form in {t, re.sub(r"\s+", " ", t).strip()}:
            if form:
                rest = rest.replace(form, " ")
    if SEVEN.search(prompt):
        raise PromptLeak("7-digit number (registro digital pattern) in prompt")
    if ARTICLE_NO.search(rest):
        raise PromptLeak("article number outside the sanctioned texts")
    p = _norm(rest)
    meta = metadata()
    for r in rows:
        m = meta.get(int(r))
        if m is None:
            continue
        if m["record_id"] and m["record_id"] in rest:
            raise PromptLeak(f"row {r}: registro digital in prompt")
        for form in _date_forms(m["publication_date"]):
            if _norm(form) in p:
                raise PromptLeak(f"row {r}: publication date in prompt")
        org = _norm(m["organo"]).strip(" .,;")
        if len(org) >= 5 and re.search(r"(?<!\w)" + re.escape(org) + r"(?!\w)", p):
            raise PromptLeak(f"row {r}: issuing court in prompt")
        words = _norm(m["rubro"]).split()
        for s in range(0, max(0, len(words) - 7)):
            if " ".join(words[s:s + 8]) in p:
                raise PromptLeak(f"row {r}: rubro text in prompt")
