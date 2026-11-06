"""Declared query strata and analysis-only flags (PREREGISTRATION §1).

Strata depend only on the facts text and the target-blind masker's output. The
``answer_named_in_facts`` flag uses positive identities and is analysis-only; no
system may read it.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Mapping

from .spec import MASK_TOKEN, S1_NO_CITATION, S2_MASKED_CITATION

QUOTE = re.compile(r'["“”].{20,}["“”]')  # N01 quote-bearing rule (body_backed_citations/sjf.py)
RESIDUAL_STATUTE = re.compile(r"\b(?:Ley|C[óo]digo|Constituci[óo]n|Reglamento|Convenci[óo]n|Estatuto|Pacto)\s+[A-ZÁÉÍÓÚÑ]")
NUMBER_AFTER_MASK = re.compile(re.escape(MASK_TOKEN) + r"[^.]{0,40}?\b\d{1,4}\b")
CONSTITUCIONAL_AFTER_MASK = re.compile(re.escape(MASK_TOKEN) + r"\s*,?\s*constitucional", re.IGNORECASE)
ARTICLE_LIST = re.compile(r"art[ií]culos?\s+((?:\d+[\w\-]*(?:\s*(?:,|y|e|o|bis|ter)\s*)?)+)", re.IGNORECASE)


def _fold(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return " ".join("".join(ch for ch in text if not unicodedata.combining(ch)).split())


def answer_named_in_facts(raw_facts: str, positives: Iterable[Mapping], titles: Mapping[str, list[str]]) -> bool:
    """True if a positive's article number and its instrument title/alias both occur in the facts."""
    numbers = {n for match in ARTICLE_LIST.finditer(raw_facts) for n in re.findall(r"\d+", match.group(1))}
    if not numbers:
        return False
    folded = _fold(raw_facts)
    for positive in positives:
        instrument, label = positive["candidate_id"].split("#articulo=")
        if label.split("-")[0] in numbers and any(t in folded for t in titles.get(instrument, ())):
            return True
    return False


def query_strata(raw_facts: str, masked: Mapping, *, version_cue: bool, answer_named: bool) -> tuple[str, dict[str, bool]]:
    text = masked["text"]
    stratum = S2_MASKED_CITATION if masked["spans"] else S1_NO_CITATION
    flags = {
        "quote_bearing": bool(QUOTE.search(raw_facts)),
        "residual_statute_name": bool(RESIDUAL_STATUTE.search(text)),
        "residual_number_after_mask": bool(NUMBER_AFTER_MASK.search(text)),
        "constitucional_after_mask": bool(CONSTITUCIONAL_AFTER_MASK.search(text)),
        "version_cue": bool(version_cue),
        "answer_named_in_facts": bool(answer_named),
    }
    return stratum, flags


def in_primary_population(stratum: str, resolution_status: str, flags: Mapping[str, bool]) -> bool:
    """PREREGISTRATION §1 population P (before the specific-target condition)."""
    return (stratum == S1_NO_CITATION and resolution_status == "full"
            and not flags.get("version_cue") and not flags.get("residual_statute_name") and not flags.get("quote_bearing"))
