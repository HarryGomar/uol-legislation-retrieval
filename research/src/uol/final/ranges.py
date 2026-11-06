"""Patch P8 of the DEV-B labelling (coordinator, 21 Oct; the author: "be careful with the del X a Y"): article ranges.

"artículo(s) N a/al M", with an optional o./º/° and a Bis / Ter (/ Quáter ...) ending, optionally numbered ("68 Bis 1"),
on either end, bound to the law named after it (the catalogue titles of ``bindfix``) or reached by the repaired
back-reference (bindfix rule B: "de la ley referida / citada / propia / invocada ..." with exactly one instrument of
that kind named in the tesis). It expands to every catalogue article of that law from N to M, Bis / letter articles
inside the range included when they exist in the catalogue. Guards: at most 60 articles; never when the law is
followed by an abrogation cue (abrogada, derogada, vigente hasta, de 19xx) or by a state or local extension, or when
the law is not in the catalogue. Development rows only (the caller passes the patched resolver output of dev sources).
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

CAP = 60
SUFFIX = r"(?:bis|ter|qu[aá]ter|quinquies|sexies|septies|octies|nonies|decies)(?:\s+\d{1,2})?"
RANGE = re.compile(rf"art[íi]culos?\s+(?:del?\s+)?(\d{{1,4}})(?:\s*(?:o\.|º|°))?(?:\s+({SUFFIX}))?\s+(?:a|al)\s+(?:del?\s+)?"
                   rf"(\d{{1,4}})(?:\s*(?:o\.|º|°))?(?:\s+({SUFFIX}))?(?![\w])", re.I)
ABROGATED = re.compile(r"^[^.;]{0,120}?\b(?:abrogad|derogad)[oa]s?\b|^[^.;]{0,120}?\bvigente\s+hasta\b|^[^.;]{0,120}?\bde\s+19[3-9]\d\b")
STATE = re.compile(r"del estado de|estatal|de la ciudad de mexico|del distrito federal|municipios?\b|jalisco|queretaro|chiapas|"
                   r"nuevo leon|veracruz|puebla|guanajuato|yucatan|sonora|sinaloa|coahuila|tamaulipas|michoacan|oaxaca|"
                   r"guerrero|hidalgo|morelos|tabasco|zacatecas|durango|chihuahua|baja california|campeche|colima|nayarit|"
                   r"tlaxcala|aguascalientes|quintana roo|san luis potosi")
ORDER = {"bis": 2, "ter": 3, "quater": 4, "quinquies": 5, "sexies": 6, "septies": 7, "octies": 8, "nonies": 9, "decies": 10}


def fold(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c))


def suffix_key(tokens: list[str]) -> tuple:
    """Order of article variants inside one base number: plain < letters (a, b, ...) < bis < ter < ...; a trailing
    number orders inside its word (68-bis < 68-bis-1 < 68-bis-2)."""
    key = []
    for t in tokens:
        if t.isdigit():
            key.append((3, int(t)))
        elif t in ORDER:
            key.append((2, ORDER[t]))
        else:
            key.append((1, t))
    return tuple(key)


def parse(text: str) -> list[dict[str, Any]]:
    out = []
    for m in RANGE.finditer(text):
        lo, hi = int(m.group(1)), int(m.group(3))
        s1 = fold(m.group(2) or "").replace("á", "a").split()
        s2 = fold(m.group(4) or "").replace("á", "a").split()
        out.append({"lo": lo, "hi": hi, "lo_suffix": s1, "hi_suffix": s2, "span": m.span()})
    return out


def law_labels(candidate_ids, law: str) -> dict[int, list[str]]:
    """{base number: [full labels]} of a law's catalogue articles (``<law>.pdf#articulo=<label>``)."""
    out: dict[int, list[str]] = {}
    tag = f"/pdf/{law}.pdf#articulo="
    for c in candidate_ids:
        if tag in c:
            lab = c.split("#articulo=")[1]
            base = lab.split("-")[0]
            if base.isdigit():
                out.setdefault(int(base), []).append(lab)
    return out


def expand(rng: dict[str, Any], labels: dict[int, list[str]], cap: int = CAP) -> list[str] | None:
    """Catalogue labels from N to M (variants inside included); None when the range is not usable (order, cap)."""
    lo, hi = rng["lo"], rng["hi"]
    if hi < lo or (hi == lo and not (rng["lo_suffix"] or rng["hi_suffix"])):
        return None
    k_lo, k_hi = suffix_key(rng["lo_suffix"]), suffix_key(rng["hi_suffix"])
    out = []
    for base in range(lo, hi + 1):
        for lab in sorted(labels.get(base, []), key=lambda x: suffix_key(x.split("-")[1:])):
            k = suffix_key(lab.split("-")[1:])
            if base == lo and k < k_lo:
                continue
            if base == hi and k > k_hi:
                continue
            out.append(lab)
    if not out or len(out) > cap:
        return None
    return out


def bind(text: str, reasoning_folded: str, span: tuple[int, int], ns: dict) -> tuple[str | None, str]:
    """Law of a range: the catalogue title right after it (bindfix titles), else bindfix rule B (one named instrument
    of the anaphor's kind). Returns (law, why); law None when unbound or guarded."""
    a, b = span
    after = ns["fold"](text[b:b + 200])
    cut = re.search(r"[.;]\s", after)
    win = after[:cut.start()] if cut else after
    if ABROGATED.match(win):
        return None, "abrogated_version_cue"
    hit = ns["first_title_after"](win)
    if hit and hit[0] <= 60:
        title = next((t for t, i in ns["titles"] if i == hit[1] and win.find(t) == hit[0]), "")
        tail = win[hit[0] + len(title): hit[0] + len(title) + 60]
        if STATE.search(tail[:45]):
            return None, "state_or_local_extension"
        return ns["iid_short"](hit[1]), "title_after"
    rule, law = ns["assess"](text, reasoning_folded, (a, b))
    if rule == "B_anaphor" and law:
        return law, "back_reference"
    if STATE.search(win[:80]):
        return None, "state_or_local_law"
    return None, "unbound"


def apply(res: dict, sources: dict, ns: dict, candidate_ids) -> tuple[dict, dict]:
    """Add P8 labels to the patched resolver output; returns (counts, covered spans per query id)."""
    counts = {"P8_labels": 0, "ranges_expanded": 0, "ranges_skipped": {}}
    covered: dict[str, list[tuple[str, int, int]]] = {}
    cache: dict[str, dict[int, list[str]]] = {}
    for qid, r in res.items():
        s = sources.get(qid)
        if s is None or r.get("kind") not in ("released_dev", "excluded_dev"):
            continue
        reasoning = ns["fold"](" ".join(x["text"] for x in s["sections"] if x["field"] != "hechos"))
        have = {p["candidate_id"] for p in r["positives"]}
        for name in ("criterio", "justificacion"):
            t = ns["sec_text"](s, name)
            for rng in parse(t):
                law, why = bind(t, reasoning, rng["span"], ns)
                if law is None:
                    counts["ranges_skipped"][why] = counts["ranges_skipped"].get(why, 0) + 1
                    continue
                labels = cache.setdefault(law, law_labels(candidate_ids, law))
                labs = expand(rng, labels)
                if labs is None:
                    counts["ranges_skipped"]["cap_or_order_or_empty"] = counts["ranges_skipped"].get("cap_or_order_or_empty", 0) + 1
                    continue
                counts["ranges_expanded"] += 1
                covered.setdefault(qid, []).append((name, *rng["span"]))
                for lab in labs:
                    cid = f"https://www.diputados.gob.mx/LeyesBiblio/pdf/{law}.pdf#articulo={lab}"
                    if cid not in have:
                        have.add(cid)
                        r["positives"] = list(r["positives"]) + [{"candidate_id": cid, "rule_ids": ["P8_range"],
                                                                  "section": name, "span": list(rng["span"]),
                                                                  "surface_sha256": None}]
                        counts["P8_labels"] += 1
    return counts, covered
