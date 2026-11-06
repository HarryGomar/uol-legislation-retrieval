"""Exact, conservative documentary references for the repaired C01 release.

This parser reads source text and the candidate catalogue, never old qrels.
An unresolved expression excludes its entire query from the strict subset.
The strict subset is mechanically resolved documentary data, not independently
adjudicated legal relevance or a claim of exhaustive citation recall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


PARSER_VERSION = "c01-documentary-repair-2.0"
IMPORTED_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
LATER_FIELDS = frozenset({"criterio_juridico", "justificacion"})
LATIN = r"(?:bis|ter|quat|quater|quintus|quinquies|sexies|sexties|sextus|septies|septimus|octies|octavus|nonies|novenus|decies|undecies|duodecies|terdecies|quaterdecies|quindecies|quinquiesdecies|sexdecies|sexiesdecies|septendecies|septdecies|octodecies|novodecies|vicies)"
# Ordinal o must touch the number. Whitespace would swallow Spanish '14 o ...'.
ARTICLE_NUMBER = re.compile(
    rf"(?P<base>\d{{1,5}})(?:o\.?|[º°])?"
    rf"(?P<tail>(?:\s*-\s*(?:ll(?![a-zñ])|[a-zñ](?![a-zñ])|\d{{1,3}}))*"
    rf"(?:\s*(?:-\s*)?{LATIN}(?![a-z])(?:\s*(?:-\s*)?\d{{1,3}})?)?)"
    r"(?![\w-])"
)
ARTICLE_PREFIX = re.compile(r"\b(?:articulos?|arts?\.)\s+", re.I)
OTHER_NUMERIC_PREFIX = re.compile(r"\b(?:numeral(?:es)?|preceptos?|diversos?|dispositivos?)\s+(?=\d)", re.I)
QUALIFIER = re.compile(r"\b(parrafos?|fraccion(?:es)?|incisos?|apartados?|numeral(?:es)?|puntos?)\b")
ORDINAL = r"(?:primer[oa]?|segund[oa]?|tercer[oa]?|cuart[oa]?|quint[oa]?|sext[oa]?|septim[oa]?|octav[oa]?|noven[oa]?|decim[oa]?|decimo(?:primer[oa]?|segund[oa]?|tercer[oa]?|cuart[oa]?|quint[oa]?|sext[oa]?|septim[oa]?|octav[oa]?|noven[oa]?)|catorce|ultim[oa]?|penultim[oa]?|antepenultim[oa]?|inicial|final)"
ROMAN = r"(?=[mdclxvi])m{0,3}(?:cm|cd|d?c{0,3})(?:xc|xl|l?x{0,3})(?:ix|iv|v?i{0,3})"
QUALIFIER_WORD = re.compile(rf"(?:{ORDINAL}|{ROMAN}(?:-[a-z])?|[a-z])\b")
ORDINAL_BEFORE_QUALIFIER = re.compile(rf"(?:{ORDINAL})\s+(?:parrafos?|fraccion(?:es)?|incisos?|apartados?)\b")
CONNECTOR = re.compile(r"\s*(?:,\s*(?:(?:y|e)\s+)?|(?:y|e)\s+)")
RANGE = re.compile(r"\s*(?:al?\s+|[-–]\s*)(?=\d)")
RELATION = re.compile(r"[\s,)]*(?:de\s+la|de\s+los|de\s+las|del|de)\s*$")
CONSTITUTION_ALIASES = (
    "Constitución Federal", "Constitución General de la República",
    "Constitución General", "CPEUM",
)
LAW_START = re.compile(r"\b(?:ley|codigo|constitucion|reglamento|convencion|pacto|estatuto|tratado)\b")
NUMERIC_CONTINUATION = re.compile(r"(?:[;,]\s*(?:(?:asi\s+como|y|e)\s+)?|,?\s+(?:asi\s+como|y|e)\s+)(?=\d)")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fold(text: str) -> tuple[str, list[int]]:
    chars, offsets = [], []
    for i, char in enumerate(text):
        # Ñ is a distinct Spanish article suffix, not a diacritic variant of N.
        if char.casefold() == "ñ":
            chars.append("ñ")
            offsets.append(i)
            continue
        if char == "\u0303":
            # A decomposed N + tilde is outside this per-codepoint mapper's
            # canonical grammar. Retain the mark so it cannot turn into N.
            chars.append(char)
            offsets.append(i)
            continue
        for value in unicodedata.normalize("NFD", char.casefold()):
            if not unicodedata.combining(value):
                chars.append("-" if value in "–—−" else value)
                offsets.append(i)
    return "".join(chars), offsets


def canonical_article(label: str) -> str:
    """Canonicalize an entire article expression; never accept a prefix."""
    value = fold(label)[0].strip()
    match = ARTICLE_NUMBER.fullmatch(value)
    if match is None:
        raise ValueError(f"unsupported article identity: {label!r}")
    base, tail = str(int(match.group("base"))), match.group("tail")
    tokens = re.findall(r"[a-zñ]+|\d+", tail)
    return "-".join([base, *[str(int(t)) if t.isdigit() else t for t in tokens]])


def _skip_qualifier(text: str, pos: int) -> tuple[int, str | None]:
    """Consume subdivision descriptions, preserving ambiguity as a failure.

    Arabic values of 'numeral', 'punto', 'fracción', 'apartado' and 'párrafo'
    remain subdivisions. A subsequent unmarked number after their list cannot
    safely be inferred to be an article, so the full expression abstains.
    Roman/word qualifiers may be followed by a new Arabic article number.
    """
    reversed_ordinal = ORDINAL_BEFORE_QUALIFIER.match(text, pos)
    if reversed_ordinal:
        return reversed_ordinal.end(), None
    prefix = QUALIFIER.match(text, pos)
    if prefix is None:
        return pos, "unsupported_qualifier"
    end = prefix.end()
    first = True
    saw_arabic = False
    arabic_count = 0
    while True:
        gap = re.match(r"\s*", text[end:])
        token_start = end + gap.end()
        number = re.match(r"\d+(?:\.\d+)?\b", text[token_start:])
        word = QUALIFIER_WORD.match(text, token_start)
        if number:
            saw_arabic = True
            arabic_count += 1
            end = token_start + number.end()
        elif word:
            end = word.end()
            # A closing inciso parenthesis may precede another qualifier.
            if text[end:end + 1] == ")":
                end += 1
        elif first:
            return end, "unsupported_subdivision_expression"
        else:
            break
        first = False
        separator = CONNECTOR.match(text, end)
        if not separator:
            break
        following = separator.end()
        next_qualifier = QUALIFIER.match(text, following)
        if next_qualifier:
            nested_end, reason = _skip_qualifier(text, following)
            return nested_end, reason
        if re.match(r"\d", text[following:]) and not saw_arabic:
            # E.g. 'primer párrafo y 1463', 'fracción V y 21'.
            break
        if not (QUALIFIER_WORD.match(text, following) or re.match(r"\d", text[following:])):
            break
        end = following
    return end, "ambiguous_arabic_subdivision_list" if arabic_count > 1 else None


def parse_expression(text: str, start: int = 0) -> dict[str, Any]:
    """Parse a numeric article list up to its law relation or first unknown token."""
    entries: list[dict[str, Any]] = []
    cursor = start
    issue = None
    while True:
        number = ARTICLE_NUMBER.match(text, cursor)
        if number is None:
            return {"entries": entries, "end": cursor, "reason": "unsupported_article_expression"}
        entries.append({"article_label": canonical_article(number.group()), "start": number.start(), "end": number.end()})
        cursor = number.end()
        if re.match(r"\.\d", text[cursor:]):
            return {"entries": entries, "end": cursor, "reason": "decimal_article_or_subdivision_unresolved"}
        while True:
            separator = CONNECTOR.match(text, cursor)
            if separator is None:
                return {"entries": entries, "end": cursor, "reason": issue}
            next_start = separator.end()
            if QUALIFIER.match(text, next_start) or ORDINAL_BEFORE_QUALIFIER.match(text, next_start):
                cursor, problem = _skip_qualifier(text, next_start)
                if problem:
                    return {"entries": entries, "end": cursor, "reason": problem}
                continue
            if ARTICLE_NUMBER.match(text, next_start):
                cursor = next_start
                break
            return {"entries": entries, "end": cursor, "reason": issue}


@lru_cache(maxsize=8)
def _patterns(signature: tuple[tuple[str, str, tuple[str, ...]], ...]):
    aliases: dict[str, set[str]] = defaultdict(set)
    titles = {}
    for identity, title, extra in signature:
        titles[identity] = title
        names = {title, *extra}
        if identity.endswith("/CPEUM.pdf"):
            names.update(CONSTITUTION_ALIASES)
        for name in names:
            clean = " ".join(fold(name)[0].split())
            if len(clean) >= 5:
                aliases[clean].add(identity)
    patterns = tuple(
        (re.compile(r"(?<!\w)" + r"\s+".join(map(re.escape, alias.split())) + r"(?!\w)"), alias, tuple(sorted(ids)))
        for alias, ids in sorted(aliases.items(), key=lambda row: (-len(row[0]), row[0]))
    )
    return patterns, titles


def _instrument_mentions(text: str, instruments: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    # observed_aliases from old answer-labelled extraction are intentionally not
    # consumed. Only source catalogue titles and explicitly declared safe aliases.
    signature = tuple(sorted((str(r["instrument_id"]), str(r["title"]), tuple(sorted(r.get("safe_aliases", [])))) for r in instruments))
    patterns, titles = _patterns(signature)
    found: list[dict] = []
    for pattern, alias, ids in patterns:
        for match in pattern.finditer(text):
            if any(row["start"] <= match.start() and match.end() <= row["end"] for row in found):
                continue
            # A state/local continuation makes a shorter embedded federal title
            # unsafe. Longest exact-title matching by itself does not solve it.
            continuation = text[match.end():match.end() + 80]
            extended = bool(re.match(r"\s+(?:del?\s+estado|para\s+el\s+estado|de\s+la\s+ciudad|local)\b", continuation))
            found.append({"start": match.start(), "end": match.end(), "alias": alias, "instrument_ids": list(ids), "unsafe_title_extension": extended})
    return sorted(found, key=lambda row: (row["start"], -row["end"])), titles


def _span(raw: str, offsets: list[int], start: int, end: int, base: int) -> dict:
    left = offsets[start] if start < len(offsets) else len(raw)
    right = offsets[end - 1] + 1 if end > start else left
    surface = raw[left:right]
    return {"start": base + left, "end": base + right, "text": surface, "sha256": digest(surface)}


def _quoted_title(raw: str, left: int, right: int) -> bool:
    # Only explicit quotation boundaries plus predominantly uppercase letters
    # identify a quoted title. Ordinary uppercase statute titles do not suffice.
    before = raw[:left]
    openers = [before.rfind(c) for c in ('"', '“', '«')]
    opener = max(openers)
    if opener < 0 or left - opener > 1000:
        return False
    closer = min((p for c in ('"', '”', '»') if (p := raw.find(c, right)) >= 0), default=-1)
    if closer < 0 or closer - left > 1500:
        return False
    quote = raw[opener + 1:closer]
    letters = [c for c in quote if c.isalpha()]
    context = fold(raw[max(0, opener - 70):opener])[0]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) > .9 and bool(re.search(r"rubro|titulo|tesis|jurisprudencia", context))


def extract_section(text: str, instruments: list[dict], candidate_index: dict, field: str, base: int) -> dict:
    normalized, offsets = fold(text)
    law_mentions, titles = _instrument_mentions(normalized, instruments)
    starts = [{"start": m.start(), "number_start": m.end(), "kind": "article_prefix"} for m in ARTICLE_PREFIX.finditer(normalized)]
    starts.extend({"start": m.start(), "number_start": m.end(), "kind": "other_numeric_prefix"} for m in OTHER_NUMERIC_PREFIX.finditer(normalized)
                  if not any(a["start"] <= m.start() < a["number_start"] for a in starts))
    # Lists joined by a semicolon/conjunction can elide the article noun. Mark
    # each numeric continuation independently; duplicates inside one list are
    # suppressed below when their spans have already been consumed.
    starts.extend({"start": m.start(), "number_start": m.end(), "kind": "numeric_continuation"} for m in NUMERIC_CONTINUATION.finditer(normalized)
                  if any(a["start"] < m.start() and m.start() - a["start"] < 1200 for a in starts))
    # Independently enumerate bare numeric expressions that attach to an exact
    # catalogue title. Without a consumed article declaration they remain an
    # unresolved designator, not an invented article reference.
    for number in ARTICLE_NUMBER.finditer(normalized):
        expression = parse_expression(normalized, number.start())
        if any(expression["end"] <= law["start"] <= expression["end"] + 80
               and RELATION.fullmatch(normalized[expression["end"]:law["start"]]) for law in law_mentions):
            starts.append({"start": number.start(), "number_start": number.start(), "kind": "bare_bound_numeric_expression"})
    starts.sort(key=lambda row: (row["start"], row["number_start"]))
    occurrences, unresolved, references = [], [], []
    consumed: list[tuple[int, int]] = []
    article_chain_end = None
    for starter in starts:
        if any(left <= starter["start"] < right for left, right in consumed):
            continue
        start = starter["start"]
        expression = parse_expression(normalized, starter["number_start"])
        end = max(expression["end"], starter["number_start"])
        article_span = _span(text, offsets, start, end, base)
        occurrence = {"field": field, "declaration_type": starter["kind"], "article_span": article_span,
                      "parsed_article_labels": [r["article_label"] for r in expression["entries"]]}
        if RANGE.match(normalized, end):
            problem = "range_not_expanded"
        else:
            problem = expression["reason"]
        plural_prefix = normalized[start:starter["number_start"]].strip() in {"articulos", "arts."}
        numeric_ranges = [re.fullmatch(r"(\d+)-(\d+)", entry["article_label"]) for entry in expression["entries"]]
        range_ambiguous_designator = plural_prefix or starter["kind"] == "numeric_continuation"
        if range_ambiguous_designator and any(match and int(match[2]) > int(match[1]) for match in numeric_ranges):
            # Numeric compound article IDs exist, but an ascending 1-4 under
            # a plural designator may mean a range. Exact catalogue existence
            # alone cannot disambiguate the source expression.
            problem = problem or "ambiguous_numeric_hyphen_range"
        continuation_attached = (
            starter["kind"] == "numeric_continuation" and article_chain_end is not None
            and not normalized[article_chain_end:start].strip()
        )
        if starter["kind"] == "numeric_continuation" and not continuation_attached:
            problem = problem or "unanchored_numeric_continuation"
        # Never assign a prior law by proximity, including 'en sus artículos'.
        options = []
        for law in law_mentions:
            if end <= law["start"] <= end + 80 and RELATION.fullmatch(normalized[end:law["start"]]):
                options.append(law)
        if len(options) != 1:
            problem = problem or ("instrument_not_explicitly_bound" if not options else "conflicting_instrument_bindings")
        law = options[0] if len(options) == 1 else None
        if law and (len(law["instrument_ids"]) != 1 or law["unsafe_title_extension"]):
            problem = problem or ("ambiguous_instrument_alias" if len(law["instrument_ids"]) != 1 else "embedded_title_has_local_extension")
        if law:
            identity = law["instrument_ids"][0]
            law_span = _span(text, offsets, law["start"], law["end"], base)
            witness = _span(text, offsets, start, law["end"], base)
            occurrence.update(instrument_id=identity, instrument_title=titles[identity], instrument_span=law_span,
                              witness_span=witness, binding_rule="explicit_forward_binding")
            consumed.append((start, law["end"]))
            if starter["kind"] == "article_prefix" or continuation_attached:
                article_chain_end = law["end"]
            if _quoted_title(text, article_span["start"] - base, law_span["end"] - base):
                problem = problem or "quoted_precedent_title"
        else:
            witness = _span(text, offsets, max(0, start - 100), min(len(normalized), end + 300), base)
            occurrence["context_span"] = witness
            consumed.append((start, end))
        # Numeral/precepto alone may name a rule/subdivision, not an article.
        if starter["kind"] in {"other_numeric_prefix", "bare_bound_numeric_expression"}:
            problem = problem or "non_article_designator_unresolved"
        if not problem and law:
            mapped = []
            for entry in expression["entries"]:
                key = (identity, entry["article_label"])
                candidate = candidate_index.get(key)
                if candidate is None:
                    problem = "exact_article_missing_from_catalogue"
                    break
                if isinstance(candidate, list):
                    if len(candidate) != 1:
                        problem = "duplicate_exact_article_identity"
                        break
                    candidate = candidate[0]
                mapped.append((entry, candidate))
            if not problem:
                occurrence["status"] = "resolved_exact_documentary_identity"
                occurrence["candidate_ids"] = sorted({row["candidate_id"] for _, row in mapped})
                for entry, candidate in mapped:
                    references.append({"candidate_id": candidate["candidate_id"], "instrument_id": identity,
                                       "article_label": entry["article_label"], "field": field,
                                       "literal_article_span": _span(text, offsets, entry["start"], entry["end"], base),
                                       "witness_span": witness, "binding_rule": "explicit_forward_binding",
                                       "declaration_type": starter["kind"], "source_unit_id": candidate.get("source_unit_id"),
                                       "candidate_body_sha256": digest(candidate["article_body"]),
                                       "candidate_body_equivalence_id": candidate.get("body_equivalence_id")})
        if problem:
            occurrence.update(status="unresolved_query_exclusion", reason=problem)
            unresolved.append(occurrence)
        occurrences.append(occurrence)
    return {"references": references, "occurrences": occurrences, "unresolved": unresolved,
            "article_prefix_count": sum(s["kind"] == "article_prefix" for s in starts),
            "explicit_instrument_mentions": law_mentions}


def extract_source(source: dict, instruments: list[dict], candidate_index: dict) -> dict:
    """Extract source-only exact witnesses; return strict eligibility and failures.

    candidate_index keys are (instrument_id, canonical_article_label). Values
    are article rows or lists to represent an ambiguous key without overwriting.
    All saved offsets address raw_published_text in Python Unicode code points;
    SHA256 hashes address UTF-8 encoded literal substrings.
    """
    if source.get("split") != "train":
        raise ValueError("repair scope admits original TRAIN safe exports only")
    raw = source["raw_published_text"]
    if digest(raw) != source["source_content_sha256"]:
        raise ValueError("raw source content binding mismatch")
    references, occurrences, unresolved, sections = [], [], [], []
    version_flags = []
    article_prefix_count = 0
    for section in source["sections"]:
        if section["field"] not in LATER_FIELDS:
            continue
        start, end = int(section["start"]), int(section["end"])
        text = section["text"]
        if raw[start:end] != text:
            raise ValueError("section does not address unchanged raw source")
        result = extract_section(text, instruments, candidate_index, section["field"], start)
        references.extend(result["references"])
        occurrences.extend(result["occurrences"])
        unresolved.extend(result["unresolved"])
        article_prefix_count += result["article_prefix_count"]
        sections.append({"field": section["field"], "start": start, "end": end, "sha256": digest(text)})
        normalized, offsets = fold(text)
        for match in re.finditer(r"\b(?:vigente(?:s)?\s+(?:hasta|antes|al|en)|texto\s+(?:anterior|vigente)|reformad[oa]s?|derogad[oa]s?|abrogad[oa]s?)\b", normalized):
            version_flags.append({"field": section["field"], "span": _span(text, offsets, match.start(), match.end(), start),
                                  "reason": "explicit_version_language; historical operative wording not adjudicated"})
    for row in references:
        for name in ("literal_article_span", "witness_span"):
            span = row[name]
            assert raw[span["start"]:span["end"]] == span["text"]
            assert digest(span["text"]) == span["sha256"]
    positives = sorted({row["candidate_id"] for row in references})
    reasons = Counter(row["reason"] for row in unresolved)
    if not sections:
        reasons["missing_later_sections"] += 1
    if not positives:
        reasons["no_exact_reference"] += 1
    return {"query_id": source["query_id"], "record_id": source["record_id"], "component_id": source["component_id"],
            "source_url": source.get("source_url"), "source_content_sha256": source["source_content_sha256"],
            "source_record_sha256": source.get("source_record_sha256"), "parser_version": PARSER_VERSION,
            "sections": sections, "article_prefix_count": article_prefix_count, "occurrences": occurrences,
            "references": references, "positive_ids": positives, "unresolved": unresolved,
            "unresolved_reasons": dict(reasons), "strict_eligible": not reasons,
            "version_flags": version_flags, "historical_operative_version_verified": False,
            "reference_scope": "explicit exact forward-bound statutory article identities in later source sections",
            "eligibility_scope": "mechanical grammar coverage; independent accuracy and exhaustive recall unestablished",
            "legal_applicability_judged": False, "old_qrels_used_for_extraction": False}


def build_reference_rows(sources: list[dict], instruments: list[dict], articles: list[dict]) -> tuple[list[dict], dict]:
    index = defaultdict(list)
    for row in articles:
        index[(row["instrument_id"], row["article_label"])].append(row)
    rows = [extract_source(row, instruments, index) for row in sources]
    reasons = Counter(reason for row in rows for reason in row["unresolved_reasons"])
    summary = {"parser_version": PARSER_VERSION, "queries": len(rows), "strict_eligible_queries": sum(row["strict_eligible"] for row in rows),
               "queries_with_exact_references": sum(bool(row["positive_ids"]) for row in rows),
               "distinct_query_candidate_pairs": sum(len(row["positive_ids"]) for row in rows),
               "literal_reference_occurrences": sum(len(row["references"]) for row in rows),
               "unresolved_expression_count": sum(len(row["unresolved"]) for row in rows),
               "queries_by_exclusion_reason": dict(reasons), "old_qrels_used_for_extraction": False,
               "protected_references_opened": 0, "model_calls": 0, "network_calls": 0}
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--instruments", type=Path, required=True)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = Path(__file__).resolve().parents[4] / "research/state/benchmark_validation_20260926/safe_training_sources.jsonl"
    if args.sources.resolve() != allowed:
        raise ValueError("only the explicit original-TRAIN safe export is authorized")
    if args.output.exists():
        raise FileExistsError("preserve previous reference attempts; use a new output directory")
    input_paths = (args.sources, args.instruments, args.articles)
    before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    preparation = json.loads((allowed.parent / "preparation.json").read_text(encoding="utf-8"))
    if before[str(args.sources)] != preparation["files"][allowed.name]:
        raise ValueError("safe TRAIN export changed from its preparation binding")
    load = lambda path: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    sources, instruments, articles = load(args.sources), load(args.instruments), load(args.articles)
    if len(sources) != 2339:
        raise ValueError("safe TRAIN source population changed")
    rows, summary = build_reference_rows(sources, instruments, articles)
    if IMPORTED_CODE_SHA256 != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("reference code changed during execution; no output receipt written")
    if any(before[str(path)] != hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths):
        raise RuntimeError("source or catalogue changed during execution; no output receipt written")
    args.output.mkdir(parents=True)
    target = args.output / "source_references.jsonl"
    target.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8", newline="\n")
    summary["bindings"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (args.sources, args.instruments, args.articles, Path(__file__), target)}
    summary["imported_code_sha256"] = IMPORTED_CODE_SHA256
    (args.output / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
