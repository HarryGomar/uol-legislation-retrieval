"""Map LLM-emitted (law name, article number) pairs to catalogue candidate ids (PLAN §3 L rules).

Instruments are matched by their catalogue title (first line of the article text, e.g. "LEY Federal del Trabajo"),
by their catalogue code (LFT, CFF, CNPP...), or by a short alias list (Constitución -> CPEUM). Titles are compared as
sets of folded content tokens; the best match must reach a token F1 of .6 and beat the runner-up. Article numbers are
normalised to the catalogue's form ("1o." -> "1", "153-F Bis" -> "153-f-bis"). Pairs that do not resolve to an
existing candidate are dropped and counted by reason.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from functools import lru_cache

from uol.round3 import common as C

STOP = frozenset("de del la las los el y e para en a por sobre con su sus que reglamentaria articulos articulo".split())
ALIASES = {"cpeum": "CPEUM", "constitucion": "CPEUM", "constitucion politica": "CPEUM", "constitucion federal": "CPEUM",
           "carta magna": "CPEUM", "ley de amparo": "LAmp", "codigo civil federal": "CCF", "codigo de comercio": "CCom"}


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in t if not unicodedata.combining(ch))).strip()


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in STOP)


@lru_cache(maxsize=1)
def catalogue() -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """({code: title}, {code: {article: candidate index}})."""
    d = C.release()
    titles: dict[str, str] = {}
    arts: dict[str, dict[str, int]] = defaultdict(dict)
    for i, (cid, text) in enumerate(zip(d.candidate_ids, d.candidate_texts)):
        code = cid.rsplit("/", 1)[-1].split(".pdf")[0]
        titles.setdefault(code, text.split("\n", 1)[0].strip())
        arts[code][cid.split("#articulo=")[-1].lower()] = i
    return titles, dict(arts)


def instrument(name: str) -> str | None:
    titles, _ = catalogue()
    f = fold(name).strip(" .,;")
    f = re.sub(r"\bde los estados unidos mexicanos\b", "", f).strip()
    if f in ALIASES:
        return ALIASES[f]
    for alias, code in ALIASES.items():
        if f.startswith(alias + " ") and code == "CPEUM":
            return code
    by_code = {c.lower(): c for c in titles}
    if f.replace(" ", "") in by_code:
        return by_code[f.replace(" ", "")]
    q = tokens(f)
    if not q:
        return None
    scored = []
    for code, title in titles.items():
        t = tokens(title)
        inter = len(q & t)
        if inter:
            p, r = inter / len(q), inter / len(t)
            scored.append((2 * p * r / (p + r), code))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.6 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
        return None
    return scored[0][1]


def article_key(number: str) -> str:
    a = fold(number)
    a = re.sub(r"^(art[iı]culo|art\.?)\s*", "", a)
    a = re.split(r",|;|\(|\b(?:fraccion|fracc|parrafo|apartado|inciso|ultimo|primer|segundo|tercer|base|numeral|de las|del|de la)\b",
                 a)[0]                                           # "31, fracción IV" / "123 apartado B" -> the article
    a = re.sub(r"(\d+)\s*(?:o|º|°)\.?(?=\b|\s|-|$)", r"\1", a)
    a = re.sub(r"[\s.]+", "-", a.strip(" .,;")).strip("-")
    return a


def resolve(name: str, number: str) -> tuple[int | None, str]:
    """(candidate index or None, reason)."""
    code = instrument(name)
    if code is None:
        return None, "instrument_unresolved"
    arts = catalogue()[1].get(code, {})
    key = article_key(number)
    if key in arts:
        return arts[key], "ok"
    base = re.match(r"\d+", key)
    if base and base.group(0) in arts and key != base.group(0):
        return None, "article_variant_not_in_catalogue"
    return None, "article_not_in_catalogue"
