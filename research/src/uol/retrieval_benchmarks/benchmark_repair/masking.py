"""Versioned, label-independent masking for the repaired development release.

Only the facts string, public catalogue titles and declared source-safe aliases
determine the output. Evidence spans belong in the builder's separate audit ledger, never in
the model-facing query file. This rule removes explicit identities; it does not
establish that the edited, published facts existed before a court's decision.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Mapping


VERSION = "catalogue-safe-alias-target-blind-v3"
MASK = "[CITA_OCULTA]"
URL_PATTERN = re.compile(r"https?://[^\s<>\[\]{}\"']+", re.IGNORECASE)
ARTICLE_PREFIX = re.compile(
    r"(?<!\w)(?:articulos?|arts?\.)(?:\s*)(?:(?:numeros?|num\.?)\s+)?"
)
LATIN_SUFFIX = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)"
# An ordinal o must touch the digits. The Spanish disjunction in `14 o norma`
# is not an ordinal. Compound suffixes and their article-level indices stay in
# one interval, so `1390 Bis 12` cannot leave a misleading bare 12 behind.
NUMBER_UNIT = re.compile(
    r"[0-9]+(?:\.?o\.?)?"
    r"(?:\s*[-\u2013]\s*(?:ll|[a-z])(?![a-z]))?"
    r"(?:\s*(?:[-\u2013]\s*)?" + LATIN_SUFFIX + r"(?:\s*(?:[-\u2013]\s*)?[0-9]+)?)?"
    r"(?!\w|[-\u2013]\s*[a-z])"
)
# Explicit LL and Ñ suffixes also occur without a hyphen. Original uppercase
# evidence below distinguishes Ñ from the folded n and avoids interpreting a
# separated Spanish o/a as an article suffix.
SPACED_COMPOUND_LETTER = re.compile(r"\s+(?P<letter>ll|n)(?!\w)")
FOLLOWING_LATIN = re.compile(r"\s*(?:[-\u2013]\s*)?" + LATIN_SUFFIX + r"(?:\s*(?:[-\u2013]\s*)?[0-9]+)?(?!\w)")
WORD_UNIT = re.compile(
    r"(?:primero|segundo|tercero|cuarto|quinto|sexto|septimo|octavo|noveno|decimo|unico)(?!\w)"
)
ROMAN_UNIT = re.compile(r"[ivxlcdm]+(?!\w)")
CANONICAL_ROMAN = re.compile(r"M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})")
SEPARATOR = re.compile(r"\s*(?:,|\by\b|\be\b|\bo\b|\ba\b|\bal\b|[-\u2013])\s*")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fold_offsets(text: str) -> tuple[str, list[int], list[int]]:
    folded, starts, ends = [], [], []
    for position, char in enumerate(text):
        emitted = False
        for value in unicodedata.normalize("NFKD", char.casefold()):
            if not unicodedata.combining(value):
                folded.append(value)
                starts.append(position)
                ends.append(position + 1)
                emitted = True
        if not emitted and unicodedata.combining(char) and ends:
            ends[-1] = position + 1
    return "".join(folded), starts, ends


def _identity(value: str) -> str:
    folded, _, _ = _fold_offsets(value)
    return " ".join(re.sub(r"[\W_]+", " ", folded).split())


def catalogue_aliases(instruments: Iterable[Mapping]) -> tuple[str, ...]:
    """Use public titles and declared source-safe aliases, never observed aliases.

    A long legal title can have a descriptive clause after a comma. Its prefix
    is usable only when it has at least three words and a legal-title noun.
    The builder owns safe-alias provenance. Per-query citation observations and
    arbitrary metadata in a catalogue row cannot change masking decisions.
    """
    titles: set[str] = set()
    for instrument in instruments:
        title = instrument.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Every masking catalogue row needs a nonempty public title")
        safe_aliases = instrument.get("safe_aliases", [])
        if not isinstance(safe_aliases, list) or any(not isinstance(alias, str) or not alias.strip() for alias in safe_aliases):
            raise ValueError("safe_aliases must be a list of nonempty source-safe names")
        titles.update(_identity(alias) for alias in safe_aliases)
        normalized = _identity(title)
        if normalized:
            titles.add(normalized)
        prefix = _identity(title.split(",", 1)[0])
        words = prefix.split()
        if "," in title and len(words) >= 3 and words[0] in {
            "ley", "codigo", "constitucion", "reglamento", "estatuto"
        }:
            titles.add(prefix)
    return tuple(sorted(titles, key=lambda value: (-len(value), value)))


def _article_intervals(text: str, folded: str, starts: list[int], ends: list[int]):
    def unit_end(position: int) -> int | None:
        for pattern in (NUMBER_UNIT, WORD_UNIT):
            match = pattern.match(folded, position)
            if match:
                end = match.end()
                if pattern is NUMBER_UNIT:
                    spaced = SPACED_COMPOUND_LETTER.match(folded, end)
                    if spaced:
                        raw_letter = text[starts[spaced.start("letter")]:ends[spaced.end("letter") - 1]]
                        if raw_letter in {"LL", "Ñ"}:
                            end = spaced.end()
                            latin = FOLLOWING_LATIN.match(folded, end)
                            if latin:
                                end = latin.end()
                return end
        match = ROMAN_UNIT.match(folded, position)
        if match:
            surface = text[starts[position]:ends[match.end() - 1]]
            # Capitalization and canonical syntax both matter. `civil` is a
            # word, and even uppercase CIVIL is not a canonical Roman numeral.
            if surface and surface == surface.upper() and CANONICAL_ROMAN.fullmatch(surface):
                return match.end()
        return None

    previous_end = -1
    for prefix in ARTICLE_PREFIX.finditer(folded):
        if prefix.start() < previous_end or prefix.end() >= len(folded):
            continue
        end = unit_end(prefix.end())
        if end is None:
            continue
        while separator := SEPARATOR.match(folded, end):
            next_end = unit_end(separator.end()) if separator.end() < len(folded) else None
            if next_end is None:
                break
            end = next_end
        previous_end = end
        yield starts[prefix.start()], ends[end - 1], "explicit_article_expression"


class CatalogueMasker:
    """Compile public titles/source-safe aliases once for all facts strings."""

    def __init__(self, instruments: Iterable[Mapping]):
        self.aliases = catalogue_aliases(instruments)
        self.catalogue_aliases_sha256 = _sha("\n".join(self.aliases))
        alternatives = [r"[^\w]+".join(map(re.escape, alias.split())) for alias in self.aliases]
        self.title_pattern = re.compile(r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)") if alternatives else None

    def mask(self, text: str) -> dict:
        if not isinstance(text, str):
            raise TypeError("The facts input must be a string, not a query or label record")
        folded, starts, ends = _fold_offsets(text)
        events = list(_article_intervals(text, folded, starts, ends))
        if self.title_pattern:
            for match in self.title_pattern.finditer(folded):
                events.append((starts[match.start()], ends[match.end() - 1], "catalogue_identity"))
        for match in URL_PATTERN.finditer(text):
            end = match.end()
            while end > match.start() and text[end - 1] in ".,;:!?":
                end -= 1
            while end > match.start() and text[end - 1] == ")" and text[match.start():end].count(")") > text[match.start():end].count("("):
                end -= 1
            events.append((match.start(), end, "explicit_url"))
        merged: list[list] = []
        for start, end, reason in sorted(events):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
                merged[-1][2].add(reason)
            else:
                merged.append([start, end, {reason}])
        cursor = transformed = 0
        pieces, spans = [], []
        for start, end, reasons in merged:
            chunk = text[cursor:start]
            pieces.extend((chunk, MASK))
            transformed += len(chunk)
            spans.append({
                "original_start": start, "original_end": end,
                "transformed_start": transformed, "transformed_end": transformed + len(MASK),
                "reasons": sorted(reasons), "surface_sha256": _sha(text[start:end]),
            })
            transformed += len(MASK)
            cursor = end
        pieces.append(text[cursor:])
        return {
            "text": "".join(pieces), "spans": spans, "policy_version": VERSION,
            "catalogue_aliases_sha256": self.catalogue_aliases_sha256,
            "source_sha256": _sha(text),
            "removed_characters": sum(row["original_end"] - row["original_start"] for row in spans),
        }


def mask_facts(text: str, instruments: Iterable[Mapping]) -> dict:
    return CatalogueMasker(instruments).mask(text)


def policy() -> dict:
    return {
        "version": VERSION, "inputs": ["facts_string", "public_catalogue_titles", "declared_source_safe_aliases"],
        "placeholder": MASK, "per_query_label_access": False,
        "title_aliases": "Full public titles, legal-title prefixes before commas and declared safe_aliases; ignore observed_aliases and other metadata",
        "alias_hash": "SHA256 UTF8 newline-joined normalized title/safe-alias set sorted by descending length then lexical order",
        "article_recognition": "Explicit articulo/art. prefix, numeric compounds with Latin suffix/index, hyphenated letters including LL/Ñ, uppercase spaced LL/Ñ, first-to-tenth/unico, uppercase canonical Roman numerals",
        "ordinal_o": "Must be attached to digits; a separated Spanish o remains unless it joins two recognized article numbers",
        "replacement": "Merge overlapping/touching intervals; preserve every other original character",
        "evidence": "Half-open Unicode-codepoint source and transformed offsets; surface hash; no label IDs; keep separate from model inputs",
        "limits": [
            "No exhaustive identity-removal guarantee; unknown abbreviations, bare identifiers and unsupported article syntax may remain",
            "Masking can remove substantively useful legal information; retain this as an explicit input treatment",
            "Visible mask count and position remain possible shortcuts; target independence does not prove their absence",
            "Edited published facts remain retrospective; this does not establish predecision availability",
        ],
    }
