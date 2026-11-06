"""Audits that make a facts-to-law v2 release defensible (PLAN section 3, item 7).

``residual_cue_report``, ``policy_agreement``, ``precision_review_sheet``,
``unjudged_pool_sheet`` and ``leakage_checks`` read a release plus, where
needed, sources, another release or run details. Their outputs are review
material, never model inputs. Each returns a JSON-serialisable summary. Given
``out_dir`` it also writes new files there (never overwriting) and a
``<audit>.manifest.json`` with the SHA-256 of every input, of the code, of the
parameters and of every output.

Command line: ``python -m uol.facts_to_law.audit --help``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import statistics
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import MISSING, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from uol.facts_to_law import spec
from uol.facts_to_law.spec import (
    FULL, MASK_TOKEN, PARTIAL, RESOLUTION_STATUSES, ROOT, S1_NO_CITATION, S2_MASKED_CITATION, STRATA,
    Positive, QueryRecord, fold_of, read_jsonl, sha256_file, sha256_text,
)
from uol.retrieval_benchmarks.benchmark_repair import masking as v3_masking

AUDIT_VERSION = "facts-to-law-v2-audit-1"
DEFAULT_SEED = "facts-to-law-v2-audit"
DEFAULT_RULES = ("R1", "R2", "R3", "R4", "R5", "R6")
SNIPPET_CHARS = 120
PRECISION_VERDICTS = ("correct", "incorrect", "unclear")
POOL_VERDICTS = ("plausibly_relevant", "not_relevant", "unclear")
ANSWER_NAMED_FIELD = "answer_named_in_facts"
BUILDER_TO_DETECTOR = {
    "quote_bearing": "quote_bearing",
    "residual_statute_name": "residual_statute_name",
    "residual_number_after_mask": "number_next_to_mask",
}

# Cue grammar, applied to accent-stripped text with original offsets.
_MASK = re.escape(MASK_TOKEN)
_CONNECTOR = r"(?:y|e|o|u|a|al)"
_SUBDIVISION = r"(?:fraccion(?:es)?|parrafos?|incisos?|apartados?|numerales?|bases?|puntos?|porcion(?:es)?\s+normativas?)"
_ORDINAL = (r"(?:primer[oa]?s?|segund[oa]s?|tercer[oa]?s?|cuart[oa]s?|quint[oa]s?|sext[oa]s?|septim[oa]s?|"
            r"octav[oa]s?|noven[oa]s?|decim[oa]s?|ultim[oa]s?|penultim[oa]s?|antepenultim[oa]s?|unic[oa]s?)")
_ROMAN = r"(?-i:[IVXLCDM]+)"
_SEPARATOR = r"\s*(?:,|" + _CONNECTOR + r"(?!\w))\s*(?:" + _CONNECTOR + r"\s+)?"
_SUBDIV_MORE = r"(?:" + _ROMAN + r"|[a-z]\)|" + _ORDINAL + r")(?!\w)"
_SUBDIV_VALUES = (r"(?:\d+(?!\w)(?:" + _SEPARATOR + r"\d+(?!\w))*+|(?:" + _ROMAN + r"|[a-z]\)|[a-z]|" + _ORDINAL
                  + r")(?!\w)(?:" + _SEPARATOR + _SUBDIV_MORE + r")*+)")
_SUBDIV = r"\s*,?\s*(?:" + _CONNECTOR + r"\s+)?" + _SUBDIVISION + r"\s+" + _SUBDIV_VALUES
_CHAIN = r"(?:" + _SUBDIV + r")*+"
_LATIN = r"(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)"
_NUMBER = r"\d+(?:o\.?)?(?:\s*[-–]?\s*" + _LATIN + r"(?:\s*[-–]?\s*\d+)?(?!\w)|\s*[-–]\s*[a-z](?![a-z]))?"
_MONTHS = r"(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre)"
NUMBER_AFTER_MASK = re.compile(
    _MASK + _CHAIN + r"\s*[,;]?\s*(?:" + _CONNECTOR + r"\s+)?(?P<number>" + _NUMBER
    + r")(?![\w%/])(?!\s+de\s+" + _MONTHS + r")", re.IGNORECASE)
NUMBER_BEFORE_MASK = re.compile(
    r"(?<![\w.,/\-])(?<!de )(?<!ano )(?P<number>" + _NUMBER + r")(?![\w%/])" + _CHAIN
    + r"\s*,?\s*(?:de\s+(?:la|las|los|el)|del|de)\s+" + _MASK, re.IGNORECASE)
CONSTITUCIONAL_AFTER_MASK = re.compile(
    _MASK + r"(?P<gap>[^.;:\[\]\n]{0,60}?)(?<!\w)constitucional(?:es)?(?!\w)", re.IGNORECASE)
_ADJACENT_GAP = re.compile(_CHAIN + r"\s*,?\s*", re.IGNORECASE)
CONSTITUCIONAL_AFTER_NUMBER = re.compile(
    r"(?<![\w.,/\-])(?P<number>" + _NUMBER + r")(?![\w%/])" + _CHAIN + r"\s*,?\s*constitucional(?:es)?(?!\w)",
    re.IGNORECASE)
# A number that is itself a subdivision value ("fraccion 3", "inciso a), numerales 1 a 4") is not an article number.
_SUBDIVISION_VALUE_LEFT = re.compile(
    r"(?:(?:fraccion(?:es)?|parrafos?|incisos?|apartados?|bases?|puntos?|porcion(?:es)?\s+normativas?)"
    r"|(?:fraccion(?:es)?|incisos?|apartados?|parrafos?|bases?)\s+\S+\s*,?\s*numerales?)\s+(?:\d+" + _SEPARATOR + r")*$",
    re.IGNORECASE)
_NAME_LINK = r"(?:(?:de\s+la|de\s+los|de\s+las|del|de|para\s+el|para\s+la|para|sobre|en\s+materia\s+de|en|el|la|los|las|y|e)\s+)*"
INSTRUMENT_NAME = re.compile(
    r"(?<!\w)(?:Ley(?:es)?|Codigos?|Reglamentos?|Estatutos?|Convencion(?:es)?|Convenios?|Tratados?|Pacto|Protocolo|"
    r"Declaracion)\s+" + _NAME_LINK + r"[A-Z][A-Za-z]+(?:\s+(?:(?:de\s+la|de\s+los|de\s+las|del|de|al|y|e|para|sobre|en)\s+)*"
    r"[A-Z][A-Za-z]+)*")
CONSTITUTION_NAME = re.compile(
    r"(?<!\w)(?:Constitucion(?:es)?|Carta\s+Magna|Ley\s+Fundamental|Norma\s+Fundamental|Ley\s+Suprema|Pacto\s+Federal)(?!\w)")
QUOTE_PAIRS = (re.compile(r"“([^”]*)”"), re.compile(r"«([^»]*)»"), re.compile(r'"([^"]*)"'))
QUOTE_LONG_CHARS = 30

CUE_DEFINITIONS = {
    "residual_statute_name": (
        "Statute named after masking: a catalogue title/safe alias the v3 masker would still mask (catalogue_title); "
        "a capitalized Ley/Codigo/Reglamento/Estatuto/Convencion/Tratado... name (instrument_name); Constitucion, "
        "Carta Magna, Ley Fundamental, Pacto Federal... (constitution_name); a catalogue file-stem abbreviation such "
        "as LFT or CPEUM as a case-sensitive token (abbreviation)."),
    "number_next_to_mask": (
        "A bare article-like number joined to a mask only by list separators and fraccion/parrafo/inciso/apartado "
        "subdivisions: after it ('[MASK], 12', '[MASK], fraccion IV, 138') or before 'de la/del [MASK]' "
        "('14 y 16 de la [MASK]'); dates, years after 'de' and subdivision values ('numerales 1 a 4') are excluded."),
    "constitucional_after_mask": (
        "'constitucional(es)' after a mask in the same clause: adjacent (only subdivisions between) or window "
        "(at most 60 characters, no [.;:] and no other mask between)."),
    "constitucional_after_number": "'N constitucional(es)' with an unmasked article-like number, e.g. '16 constitucional'.",
    "residual_article_expression": "An explicit article expression the v3 masker would still mask (masker not idempotent).",
    "quote_bearing": (f"A non-empty quotation inside curly, angle or straight double quotes; kind long (at least "
                      f"{QUOTE_LONG_CHARS} characters, possible quoted statute text) or short (a quoted term)."),
}


def _fold(text: str, *, casefold: bool = False) -> tuple[str, list[int], list[int]]:
    """Accent-stripped (optionally casefolded) text with the source offsets of each character."""
    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, ch in enumerate(text):
        emitted = False
        for value in unicodedata.normalize("NFKD", ch.casefold() if casefold else ch):
            if not unicodedata.combining(value):
                out.append(value)
                starts.append(i)
                ends.append(i + 1)
                emitted = True
        if not emitted and ends and unicodedata.combining(ch):
            ends[-1] = i + 1
    return "".join(out), starts, ends


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _snippet(text: str, start: int = 0, end: int = 0, width: int = SNIPPET_CHARS) -> str:
    """A ``width``-character window of ``text`` centred on [start, end), whitespace flattened."""
    begin = start if end - start >= width else max(0, start - (width - (end - start)) // 2)
    begin = min(begin, max(0, len(text) - width))
    return re.sub(r"\s", " ", text[begin:begin + width])


def _order_key(seed: str, *parts: Any) -> str:
    return sha256_text("|".join(str(p) for p in (seed, *parts)))


def instrument_code(candidate_id: str) -> str:
    """Short instrument code from a candidate id, e.g. ``CPEUM`` for ``.../CPEUM.pdf#articulo=16``."""
    return candidate_id.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1].rsplit(".", 1)[0]


def _positive_ids(row: dict) -> list[str]:
    """Unique positive candidate ids in first-seen order (``positives`` or ``positive_ids`` rows)."""
    if "positives" in row:
        ids = [p["candidate_id"] if isinstance(p, dict) else str(p) for p in row["positives"]]
    else:
        ids = [str(p) for p in row.get("positive_ids", [])]
    return list(dict.fromkeys(ids))


def _answer_named(query: dict) -> bool | None:
    if ANSWER_NAMED_FIELD in query:
        return bool(query[ANSWER_NAMED_FIELD])
    flags = query.get("flags") or {}
    return bool(flags[ANSWER_NAMED_FIELD]) if ANSWER_NAMED_FIELD in flags else None


def _distribution(values: Sequence[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {"n": len(ordered), "mean": statistics.fmean(ordered), "median": statistics.median(ordered),
            "min": ordered[0], "max": ordered[-1], "p90": ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]}


class _Outputs:
    """Exclusive writer for one audit's files under ``out_dir``; ``None`` writes nothing."""

    def __init__(self, out_dir: Path | None, names: Iterable[str]):
        self.dir = Path(out_dir) if out_dir is not None else None
        self.written: dict[str, str] = {}
        if self.dir is not None:
            existing = [n for n in names if (self.dir / n).exists()]
            if existing:
                raise FileExistsError(f"refusing to overwrite audit outputs in {self.dir}: {existing}")
            self.dir.mkdir(parents=True, exist_ok=True)

    def _write(self, name: str, payload: str, encoding: str = "utf-8") -> None:
        if self.dir is None:
            return
        path = self.dir / name
        with path.open("x", encoding=encoding, newline="") as f:
            f.write(payload)
        self.written[name] = sha256_file(path)

    def text(self, name: str, text: str) -> None:
        self._write(name, text)

    def json(self, name: str, obj: Any) -> None:
        self._write(name, json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")

    def jsonl(self, name: str, rows: Iterable[dict]) -> None:
        self._write(name, "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))

    def csv(self, name: str, columns: Sequence[str], rows: Iterable[dict]) -> None:
        """UTF-8 with BOM so spreadsheet tools keep Spanish characters."""
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        self._write(name, buffer.getvalue(), encoding="utf-8-sig")

    def manifest(self, stem: str, provenance: dict) -> None:
        self.json(f"{stem}.manifest.json", {**provenance, "outputs": dict(sorted(self.written.items()))})


def _code_hashes() -> dict[str, str]:
    out = {}
    for module_file in (__file__, spec.__file__, v3_masking.__file__):
        path = Path(module_file).resolve()
        try:
            name = path.relative_to(ROOT).as_posix()
        except ValueError:
            name = str(path)
        out[name] = sha256_file(path)
    return out


def _provenance(audit: str, inputs: dict[str, Path | None], **parameters: Any) -> dict:
    return {
        "audit": audit, "audit_version": AUDIT_VERSION, "format_version": spec.FORMAT_VERSION,
        "inputs": {k: {"path": str(Path(p)), "sha256": sha256_file(Path(p))}
                   for k, p in sorted(inputs.items()) if p is not None and Path(p).is_file()},
        "code_sha256": _code_hashes(),
        "parameters": parameters,
    }


def _release_inputs(release_dir: Path, *names: str) -> dict[str, Path]:
    return {f"release/{n}": release_dir / n for n in ("manifest.json", *names) if (release_dir / n).exists()}


def _release_id(release_dir: Path) -> str:
    path = release_dir / "manifest.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8")).get("release_id")
        if value:
            return str(value)
    return release_dir.name


@dataclass(frozen=True)
class Catalogue:
    """Candidate files bound by a release manifest; ``*_declared`` says the file's SHA-256 is in the manifest."""

    articles: Path
    instruments: Path | None
    articles_declared: bool
    instruments_declared: bool | None


def _manifest_strings(obj: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path + (str(k),), str(k)
            yield from _manifest_strings(v, path + (str(k),))
    elif isinstance(obj, list):
        for v in obj:
            yield from _manifest_strings(v, path)
    elif isinstance(obj, str):
        yield path, obj


def _resolve_path(value: str, release_dir: Path) -> Path | None:
    if not value or len(value) > 1024 or "\n" in value or re.match(r"[a-z][a-z0-9+.-]*://", value, re.IGNORECASE):
        return None
    try:
        path = Path(value)
        for candidate in ([path] if path.is_absolute() else [release_dir / path, ROOT / path]):
            if candidate.exists():
                return candidate.resolve()
    except (OSError, ValueError):
        return None
    return None


def bound_catalogue(release_dir: Path, articles_jsonl: Path | None = None,
                    instruments_jsonl: Path | None = None) -> Catalogue:
    """Locate the candidate ``articles.jsonl``/``instruments.jsonl`` the release manifest binds.

    Explicit paths win. Otherwise manifest strings naming those files (absolute, release-relative or
    repository-relative) are used; directories count only under a key mentioning ``candidate``.
    """
    release_dir = Path(release_dir)
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    strings = list(_manifest_strings(manifest))
    declared = {s.lower() for _, s in strings if re.fullmatch(r"[0-9a-fA-F]{64}", s)}

    def find(name: str) -> Path | None:
        files: list[Path] = []
        dirs: list[Path] = []
        for keys, value in strings:
            if not re.search(r"[\\/]|\.jsonl$", value):
                continue
            path = _resolve_path(value, release_dir)
            if path is None:
                continue
            if path.is_file() and path.name == name and path not in files:
                files.append(path)
            elif path.is_dir() and any("candidate" in k.lower() for k in keys):
                for inner in (path / name, path / "candidates" / name):
                    if inner.is_file() and inner.resolve() not in dirs:
                        dirs.append(inner.resolve())
        found = files or dirs
        if len(found) > 1:
            raise ValueError(f"ambiguous {name} bindings in {manifest_path}: {[str(p) for p in found]}")
        return found[0] if found else None

    articles = Path(articles_jsonl).resolve() if articles_jsonl else find("articles.jsonl")
    if articles is None:
        raise FileNotFoundError(f"no candidate articles.jsonl bound by {manifest_path}; pass articles_jsonl")
    instruments = Path(instruments_jsonl).resolve() if instruments_jsonl else find("instruments.jsonl")
    if instruments is None and (articles.parent / "instruments.jsonl").is_file():
        instruments = articles.parent / "instruments.jsonl"
    return Catalogue(
        articles=articles, instruments=instruments,
        articles_declared=sha256_file(articles) in declared,
        instruments_declared=(sha256_file(instruments) in declared) if instruments else None,
    )


def _try_catalogue(release_dir: Path, articles_jsonl: Path | None, instruments_jsonl: Path | None) -> tuple[Catalogue | None, str | None]:
    try:
        return bound_catalogue(release_dir, articles_jsonl, instruments_jsonl), None
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def _load_articles(path: Path, bodies_for: set[str] | None = None) -> dict[str, dict]:
    """``candidate_id -> {article_label, instrument_id[, article_body]}``; bodies only for ``bodies_for``."""
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        cid = row["candidate_id"]
        entry = {"article_label": row.get("article_label"), "instrument_id": row.get("instrument_id") or cid.split("#", 1)[0]}
        if bodies_for is not None and cid in bodies_for:
            entry["article_body"] = row.get("article_body") or ""
        out[cid] = entry
    return out


def _load_instruments(path: Path | None) -> dict[str, dict]:
    return {row["instrument_id"]: row for row in read_jsonl(path)} if path else {}


def _abbreviations(instruments: Iterable[dict]) -> list[str]:
    stems = set()
    for row in instruments:
        stem = instrument_code(row["instrument_id"] + "#").split("_", 1)[0]
        if re.fullmatch(r"[A-Za-z]{2,}", stem) and sum(c.isupper() for c in stem) >= 2:
            stems.add(stem)
    return sorted(stems, key=lambda s: (-len(s), s))


class CueDetector:
    """Label-blind residual-cue detectors over masked facts (definitions in ``CUE_DEFINITIONS``)."""

    def __init__(self, instruments: Sequence[dict] | None = None):
        instruments = list(instruments or [])
        self.masker = v3_masking.CatalogueMasker(instruments) if instruments else None
        stems = _abbreviations(instruments)
        self.abbreviation = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, stems)) + r")(?!\w)") if stems else None

    def detect(self, text: str) -> dict[str, list[dict]]:
        """Cue name -> hits ``{kind, start, end, surface}`` in original-text offsets."""
        folded, starts, ends = _fold(text)
        hits: dict[str, list[dict]] = defaultdict(list)

        def add(cue: str, kind: str, start: int, end: int, *, folded_offsets: bool = True) -> None:
            if folded_offsets:
                start, end = starts[start], ends[end - 1]
            hits[cue].append({"kind": kind, "start": start, "end": end, "surface": text[start:end]})

        def subdivision_value(start: int) -> bool:
            return bool(_SUBDIVISION_VALUE_LEFT.search(folded, max(0, start - 80), start))

        for m in NUMBER_AFTER_MASK.finditer(folded):
            add("number_next_to_mask", "after", *m.span("number"))
        for m in NUMBER_BEFORE_MASK.finditer(folded):
            if not subdivision_value(m.start("number")):
                add("number_next_to_mask", "before", *m.span("number"))
        for m in CONSTITUCIONAL_AFTER_MASK.finditer(folded):
            add("constitucional_after_mask", "adjacent" if _ADJACENT_GAP.fullmatch(m.group("gap")) else "window", *m.span())
        for m in CONSTITUCIONAL_AFTER_NUMBER.finditer(folded):
            if not subdivision_value(m.start("number")):
                add("constitucional_after_number", "number", *m.span())
        for m in INSTRUMENT_NAME.finditer(folded):
            add("residual_statute_name", "instrument_name", *m.span())
        for m in CONSTITUTION_NAME.finditer(folded):
            add("residual_statute_name", "constitution_name", *m.span())
        if self.abbreviation is not None:
            for m in self.abbreviation.finditer(text):
                add("residual_statute_name", "abbreviation", *m.span(), folded_offsets=False)
        if self.masker is not None:
            for span in self.masker.mask(text)["spans"]:
                s, e = span["original_start"], span["original_end"]
                if "catalogue_identity" in span["reasons"]:
                    add("residual_statute_name", "catalogue_title", s, e, folded_offsets=False)
                if "explicit_article_expression" in span["reasons"]:
                    add("residual_article_expression", "explicit_article_expression", s, e, folded_offsets=False)
        for pattern in QUOTE_PAIRS:
            for m in pattern.finditer(text):
                size = len(m.group(1).strip())
                if size:
                    add("quote_bearing", "long" if size >= QUOTE_LONG_CHARS else "short", *m.span(), folded_offsets=False)
        titles = [(h["start"], h["end"]) for h in hits.get("residual_statute_name", []) if h["kind"] == "catalogue_title"]
        if titles:
            hits["residual_statute_name"] = [h for h in hits["residual_statute_name"] if h["kind"] == "catalogue_title"
                                             or not any(s < h["end"] and h["start"] < e for s, e in titles)]
        return {cue: sorted(rows, key=lambda r: (r["start"], r["end"], r["kind"])) for cue, rows in hits.items()}


def _stratum(query: dict) -> str:
    return str(query.get("stratum") or "unknown")


def residual_cue_report(release_dir: Path, out_dir: Path | None = None, *, examples: int = 10,
                        seed: str = DEFAULT_SEED, instruments_jsonl: Path | None = None) -> dict:
    """Residual-cue counts and examples per cue and stratum, mask-count distributions and builder-flag agreement.

    Builder flags (``queries.flags``) are reported as declared; audit detectors are recomputed here from the
    masked text and the bound catalogue's public titles. Examples are the first ``examples`` flagged queries
    per cue and stratum in SHA-256(seed|cue|query_id) order, each with a 120-character snippet.
    """
    release_dir = Path(release_dir)
    queries = list(read_jsonl(release_dir / "queries.jsonl"))
    catalogue, catalogue_error = _try_catalogue(release_dir, None, instruments_jsonl)
    instruments_path = Path(instruments_jsonl) if instruments_jsonl else (catalogue.instruments if catalogue else None)
    detector = CueDetector(list(_load_instruments(instruments_path).values()) if instruments_path else None)

    strata = sorted({_stratum(q) for q in queries}, key=lambda s: (s not in STRATA, s))
    groups = {"all": queries, **{s: [q for q in queries if _stratum(q) == s] for s in strata}}
    detected = {q["query_id"]: detector.detect(q["text"]) for q in queries}
    builder_flags = sorted({k for q in queries for k, v in (q.get("flags") or {}).items() if isinstance(v, bool)})

    def example(q: dict, hits: list[dict]) -> dict:
        text = q["text"]
        if hits:
            first = hits[0]
            return {"query_id": q["query_id"], "kind": first["kind"], "surface": first["surface"],
                    "snippet": _snippet(text, first["start"], first["end"])}
        at = text.find(MASK_TOKEN)
        return {"query_id": q["query_id"], "kind": None, "surface": None,
                "snippet": _snippet(text, max(at, 0), max(at, 0) + (len(MASK_TOKEN) if at >= 0 else 0))}

    def summarize(name: str, has: dict[str, list[dict] | None], source: str, definition: str) -> dict:
        block: dict[str, Any] = {"source": source, "definition": definition, "queries": {}, "share": {},
                                 "kinds": {}, "examples": {}}
        for group, rows in groups.items():
            flagged = [q for q in rows if has.get(q["query_id"]) is not None]
            block["queries"][group] = len(flagged)
            block["share"][group] = len(flagged) / len(rows) if rows else None
            kinds = Counter()
            for q in flagged:
                kinds.update({h["kind"] for h in has[q["query_id"]] or []})
            block["kinds"][group] = dict(sorted(kinds.items()))
            if group != "all":
                ordered = sorted(flagged, key=lambda q: _order_key(seed, name, q["query_id"]))[:examples]
                block["examples"][group] = [example(q, has[q["query_id"]] or []) for q in ordered]
        return block

    cues: dict[str, dict] = {}
    for cue, definition in CUE_DEFINITIONS.items():
        has = {qid: hits[cue] for qid, hits in detected.items() if hits.get(cue)}
        cues[cue] = summarize(cue, has, "audit_detector", definition)
    for flag in builder_flags:
        mapped = BUILDER_TO_DETECTOR.get(flag)
        has = {q["query_id"]: detected[q["query_id"]].get(mapped, []) if mapped else []
               for q in queries if (q.get("flags") or {}).get(flag) is True}
        cues[f"flag:{flag}"] = summarize(f"flag:{flag}", has, "release_flag", f"Builder flag queries.flags.{flag}")

    agreement = {}
    for flag, cue in BUILDER_TO_DETECTOR.items():
        if flag not in builder_flags:
            continue
        flagged = {q["query_id"] for q in queries if (q.get("flags") or {}).get(flag) is True}
        found = {qid for qid, hits in detected.items() if hits.get(cue)}
        agreement[flag] = {"audit_detector": cue, "both": len(flagged & found), "flag_only": len(flagged - found),
                           "detector_only": len(found - flagged),
                           "flag_only_ids": sorted(flagged - found)[:50], "detector_only_ids": sorted(found - flagged)[:50]}

    masks = {}
    for group, rows in groups.items():
        counts = [q["text"].count(MASK_TOKEN) for q in rows]
        hist = Counter(counts)
        masks[group] = {"histogram": {str(k): hist[k] for k in sorted(hist)}, **_distribution(counts),
                        "share_ge_1": sum(c >= 1 for c in counts) / len(counts) if counts else None,
                        "share_ge_2": sum(c >= 2 for c in counts) / len(counts) if counts else None,
                        "share_ge_3": sum(c >= 3 for c in counts) / len(counts) if counts else None}

    answer_values = {q["query_id"]: _answer_named(q) for q in queries}
    present = any(v is not None for v in answer_values.values())
    answer = {"field": ANSWER_NAMED_FIELD, "field_present": present}
    if present:
        for group, rows in groups.items():
            values = [answer_values[q["query_id"]] for q in rows if answer_values[q["query_id"]] is not None]
            answer[group] = {"true": sum(values), "with_field": len(values), "queries": len(rows),
                             "share": sum(values) / len(values) if values else None}

    citation_cues = [c for c in CUE_DEFINITIONS if c != "quote_bearing"]
    any_cue = {qid for qid, hits in detected.items() if any(hits.get(c) for c in citation_cues)}
    summary = {
        "audit": "residual_cue_report", "release_id": _release_id(release_dir),
        "queries": {g: len(rows) for g, rows in groups.items()},
        "headline": {g: {"queries": len(rows), "any_citation_cue": sum(q["query_id"] in any_cue for q in rows),
                         "citation_cues": citation_cues} for g, rows in groups.items()},
        "cues": cues, "builder_flags": builder_flags, "flag_detector_agreement": agreement,
        "mask_counts": masks, "answer_named_in_facts": answer,
        "catalogue": {"instruments": str(instruments_path) if instruments_path else None, "error": catalogue_error,
                      "catalogue_title_and_abbreviation_detectors": instruments_path is not None},
        "seed": seed,
    }
    names = ["residual_cues.json", "residual_cues.jsonl", "residual_cues.manifest.json"]
    out = _Outputs(out_dir, names)
    summary["provenance"] = _provenance(
        "residual_cue_report", {**_release_inputs(release_dir, "queries.jsonl"), "catalogue/instruments.jsonl": instruments_path},
        seed=seed, examples=examples, snippet_chars=SNIPPET_CHARS, quote_long_chars=QUOTE_LONG_CHARS)
    out.json(names[0], summary)
    out.jsonl(names[1], ({"query_id": q["query_id"], "stratum": _stratum(q), "mask_count": q["text"].count(MASK_TOKEN),
                          "flags": {k: v for k, v in sorted((q.get("flags") or {}).items()) if v},
                          "cues": detected[q["query_id"]]}
                         for q in queries if detected[q["query_id"]] or any((q.get("flags") or {}).values())))
    out.manifest("residual_cues", summary["provenance"])
    return summary


def _group_by_key(queries: list[dict], references: list[dict], key: str) -> tuple[dict[str, dict], dict]:
    by_qid = {q["query_id"]: q for q in queries}
    groups: dict[str, dict] = {}
    unmatched = 0
    for row in references:
        qid = row["query_id"]
        query = by_qid.get(qid)
        if key == "query_id":
            value = qid
        elif query is not None and query.get("record_id") is not None:
            value = str(query["record_id"])
        else:
            unmatched += 1
            continue
        group = groups.setdefault(value, {"query_ids": [], "positives": [], "text_sha256": set(),
                                          "strata": set(), "resolution": set()})
        group["query_ids"].append(qid)
        group["positives"] = list(dict.fromkeys(group["positives"] + _positive_ids(row)))
        if query is not None:
            text_hash = sha256_text(_one_line(query["text"])) if isinstance(query.get("text"), str) else query.get("text_sha256")
            if text_hash:
                group["text_sha256"].add(text_hash)
            group["strata"].add(str(query.get("stratum")))
            group["resolution"].add(str(query.get("resolution_status")))
    stats = {"keys": len(groups), "references_without_key": unmatched,
             "keys_with_multiple_queries": sum(len(g["query_ids"]) > 1 for g in groups.values())}
    return groups, stats


def _relation(ours: set[str], other: set[str]) -> str:
    if ours == other:
        return "identical"
    if not ours & other:
        return "disjoint"
    if ours < other:
        return "subset"
    if ours > other:
        return "superset"
    return "overlap"


def policy_agreement(release_dir: Path, other_queries_jsonl: Path, other_references_jsonl: Path,
                     key: str = "query_id", out_dir: Path | None = None, *, name: str = "other", top: int = 50,
                     articles_jsonl: Path | None = None, instruments_jsonl: Path | None = None) -> dict:
    """Compare release positives with another release's references on shared ``query_id`` or ``record_id`` keys.

    Relations per shared key: identical, subset (release positives a strict subset of the other's), superset,
    disjoint, overlap. Differences are broken down by instrument and the ``top`` most frequent differing
    candidate ids are listed. Keys mapping to several queries on one side use the union of their positives.
    """
    if key not in ("query_id", "record_id"):
        raise ValueError("key must be 'query_id' or 'record_id'")
    release_dir = Path(release_dir)
    ours, ours_stats = _group_by_key(list(read_jsonl(release_dir / "queries.jsonl")),
                                     list(read_jsonl(release_dir / "references.jsonl")), key)
    other, other_stats = _group_by_key(list(read_jsonl(other_queries_jsonl)), list(read_jsonl(other_references_jsonl)), key)
    catalogue, catalogue_error = _try_catalogue(release_dir, articles_jsonl, instruments_jsonl)
    titles = {iid: row.get("title") for iid, row in _load_instruments(catalogue.instruments if catalogue else None).items()}
    catalogue_ids = set(_load_articles(catalogue.articles)) if catalogue else None

    rows = []
    for value in sorted(set(ours) & set(other)):
        a, b = ours[value], other[value]
        sa, sb = set(a["positives"]), set(b["positives"])
        union = sa | sb
        same_text = None
        if a["text_sha256"] and b["text_sha256"]:
            same_text = bool(a["text_sha256"] & b["text_sha256"])
        rows.append({
            "key": value, "query_ids": a["query_ids"], "other_query_ids": b["query_ids"],
            "positives": sorted(sa), "other_positives": sorted(sb), "shared": sorted(sa & sb),
            "only_release": sorted(sa - sb), "only_other": sorted(sb - sa),
            "jaccard": len(sa & sb) / len(union) if union else 1.0, "relation": _relation(sa, sb),
            "same_text": same_text, "stratum": "|".join(sorted(a["strata"])),
            "resolution_status": "|".join(sorted(a["resolution"])),
        })

    def block(selected: list[dict]) -> dict:
        jac = [r["jaccard"] for r in selected]
        rel = Counter(r["relation"] for r in selected)
        shared = sum(len(r["shared"]) for r in selected)
        release_pairs = sum(len(r["positives"]) for r in selected)
        other_pairs = sum(len(r["other_positives"]) for r in selected)
        return {"keys": len(selected), "jaccard": _distribution(jac),
                "relations": {k: rel.get(k, 0) for k in ("identical", "subset", "superset", "disjoint", "overlap")},
                "release_pairs": release_pairs, "other_pairs": other_pairs, "shared_pairs": shared,
                "share_of_release_pairs_in_other": shared / release_pairs if release_pairs else None,
                "share_of_other_pairs_in_release": shared / other_pairs if other_pairs else None}

    instruments: dict[str, Counter] = defaultdict(Counter)
    differing: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        for side, field in (("only_release", "only_release"), ("only_other", "only_other"), ("shared", "shared")):
            for cid in r[field]:
                instruments[cid.split("#", 1)[0]][f"{side}_pairs"] += 1
                if side != "shared":
                    differing[cid][side] += 1
            for iid in {cid.split("#", 1)[0] for cid in r[field]}:
                instruments[iid][f"{side}_keys"] += 1
    by_instrument = sorted(
        ({"instrument_id": iid, "code": instrument_code(iid + "#"), "title": titles.get(iid),
          **{k: c.get(k, 0) for k in ("only_release_pairs", "only_other_pairs", "shared_pairs",
                                       "only_release_keys", "only_other_keys", "shared_keys")}}
         for iid, c in instruments.items()),
        key=lambda d: (-(d["only_release_pairs"] + d["only_other_pairs"]), d["instrument_id"]))
    top_differing = sorted(
        ({"candidate_id": cid, "code": instrument_code(cid), "only_release": c.get("only_release", 0),
          "only_other": c.get("only_other", 0), "total": sum(c.values())} for cid, c in differing.items()),
        key=lambda d: (-d["total"], d["candidate_id"]))[:top]

    strata_blocks = {}
    for group in sorted({(r["stratum"], r["resolution_status"]) for r in rows}):
        strata_blocks["/".join(group)] = block([r for r in rows if (r["stratum"], r["resolution_status"]) == group])
    other_missing = None
    if catalogue_ids is not None:
        missing = sorted({cid for r in rows for cid in r["other_positives"] if cid not in catalogue_ids})
        other_missing = {"candidates": len(missing), "examples": missing[:20]}
    summary = {
        "audit": "policy_agreement", "release_id": _release_id(release_dir), "other": name, "key": key,
        "release_keys": ours_stats, "other_keys": other_stats, "shared_keys": len(rows),
        "release_only_keys": len(set(ours) - set(other)), "other_only_keys": len(set(other) - set(ours)),
        "overall": block(rows),
        "same_text": {"compared": sum(r["same_text"] is not None for r in rows),
                      "identical_text": sum(r["same_text"] is True for r in rows),
                      "identical_text_block": block([r for r in rows if r["same_text"] is True])},
        "by_release_stratum_and_resolution": strata_blocks,
        "differences_by_instrument": by_instrument, "top_differing_candidates": top_differing,
        "other_positives_missing_from_bound_catalogue": other_missing, "catalogue_error": catalogue_error,
        "definitions": {"subset": "release positives are a strict subset of the other's",
                        "superset": "release positives are a strict superset of the other's",
                        "overlap": "share at least one positive; neither contains the other",
                        "disjoint": "share no positive",
                        "same_text": "whitespace-normalized query texts are identical (text_sha256 when no text)"},
    }
    stem = f"policy_agreement_{name}"
    names = [f"{stem}.json", f"{stem}.jsonl", f"{stem}.manifest.json"]
    out = _Outputs(out_dir, names)
    summary["provenance"] = _provenance(
        "policy_agreement", {**_release_inputs(release_dir, "queries.jsonl", "references.jsonl"),
                             "other/queries": Path(other_queries_jsonl), "other/references": Path(other_references_jsonl),
                             "catalogue/instruments.jsonl": catalogue.instruments if catalogue else None,
                             "catalogue/articles.jsonl": catalogue.articles if catalogue else None},
        key=key, name=name, top=top)
    out.json(names[0], summary)
    out.jsonl(names[1], rows)
    out.manifest(stem, summary["provenance"])
    return summary


_SECTION_ALIASES = {"criterio": "criterio_juridico", "criterio_juridico": "criterio_juridico",
                    "justificacion": "justificacion", "hechos": "hechos"}


def _section_key(name: str) -> str:
    folded = re.sub(r"[\W_]+", "_", _fold(str(name), casefold=True)[0]).strip("_")
    return _SECTION_ALIASES.get(folded, folded)


def _section_rank(name: str) -> int:
    return {"criterio_juridico": 0, "justificacion": 1}.get(_section_key(name), 2)


def locate_surface(source: dict, positive: dict) -> dict:
    """Find a positive's cited surface in ``raw_published_text``.

    Spans are tried as section-relative offsets (spec) and then as raw-text offsets; the first that reproduces
    ``surface_sha256`` wins. Returns ``{surface, start, end, basis, verified}`` with raw-text offsets.
    """
    raw = source.get("raw_published_text") or ""
    start, end = (int(v) for v in positive["span"])
    expected = positive.get("surface_sha256")
    attempts = []
    for section in source.get("sections") or []:
        if _section_key(section.get("field", "")) == _section_key(positive.get("section", "")) and "start" in section:
            attempts.append(("section", int(section["start"]) + start, int(section["start"]) + end))
            break
    attempts.append(("raw", start, end))
    for basis, a, b in attempts:
        if 0 <= a <= b <= len(raw) and expected and sha256_text(raw[a:b]) == expected:
            return {"surface": raw[a:b], "start": a, "end": b, "basis": basis, "verified": True}
    basis, a, b = attempts[0]
    ok = 0 <= a <= b <= len(raw)
    return {"surface": raw[a:b] if ok else "", "start": a, "end": b, "basis": f"{basis}_unverified", "verified": False}


def _context(raw: str, start: int, end: int, chars: int) -> str:
    if not (0 <= start <= end <= len(raw)):
        return ""
    return _one_line(raw[max(0, start - chars):start]) + " ⟦" + _one_line(raw[start:end]) + "⟧ " + _one_line(raw[end:end + chars])


def _index_sources(sources_jsonl: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    by_qid: dict[str, dict] = {}
    by_rid: dict[str, dict] = {}
    for row in read_jsonl(sources_jsonl):
        if row.get("query_id") is not None:
            by_qid.setdefault(str(row["query_id"]), row)
        if row.get("record_id") is not None:
            by_rid.setdefault(str(row["record_id"]), row)
    return by_qid, by_rid


def _md(text: Any) -> str:
    return _one_line(str(text if text is not None else "")).replace("|", "\\|").replace("`", "'")


PRECISION_COLUMNS = (
    "sample_id", "rule", "query_id", "record_id", "component_id", "all_rule_ids", "section", "span", "span_basis",
    "surface_verified", "cited_surface", "context", "candidate_id", "candidate_instrument", "candidate_article_label",
    "candidate_article_excerpt", "source_url", "documentary_verdict", "note",
)


def precision_review_sheet(release_dir: Path, sources_jsonl: Path, rules: Sequence[str] = DEFAULT_RULES,
                           per_rule: int = 40, seed: str = DEFAULT_SEED, out_dir: Path | None = None, *,
                           context_chars: int = 150, excerpt_chars: int = 300,
                           articles_jsonl: Path | None = None, instruments_jsonl: Path | None = None) -> dict:
    """Deterministic per-rule sample of resolved positives for documentary identity checking.

    The unit is a unique (rule, query_id, candidate_id); its witness is the first positive occurrence carrying
    the rule (criterio before justificacion, then span start). Units are ordered by
    SHA-256(seed|rule|query_id|candidate_id) and the first ``per_rule`` are kept. Verdict columns are empty.
    """
    release_dir = Path(release_dir)
    queries = {q["query_id"]: q for q in read_jsonl(release_dir / "queries.jsonl")}
    references = list(read_jsonl(release_dir / "references.jsonl"))
    catalogue = bound_catalogue(release_dir, articles_jsonl, instruments_jsonl)
    by_qid, by_rid = _index_sources(Path(sources_jsonl))

    units: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    pair_rules: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ref in references:
        for p in ref.get("positives", []):
            pair_rules[(ref["query_id"], p["candidate_id"])].update(p.get("rule_ids") or [])
            for rule in p.get("rule_ids") or []:
                units[(rule, ref["query_id"], p["candidate_id"])].append(p)
    observed = Counter(rule for rule, _, _ in units)
    chosen: dict[str, list[tuple[str, str, str]]] = {}
    for rule in rules:
        keys = sorted((k for k in units if k[0] == rule), key=lambda k: _order_key(seed, *k))
        chosen[rule] = keys[:per_rule]
    needed = {k[2] for keys in chosen.values() for k in keys}
    articles = _load_articles(catalogue.articles, needed)
    titles = {iid: row.get("title") for iid, row in _load_instruments(catalogue.instruments).items()}

    rows, per_rule_summary = [], {}
    for rule in rules:
        population = [k for k in units if k[0] == rule]
        verified = missing_source = missing_candidate = 0
        for index, unit in enumerate(chosen[rule], 1):
            _, qid, cid = unit
            witness = min(units[unit], key=lambda p: (_section_rank(p.get("section", "")), list(p.get("span", [0]))))
            query = queries.get(qid, {})
            source = by_qid.get(qid) or by_rid.get(str(query.get("record_id")))
            located = locate_surface(source, witness) if source else {"surface": "", "start": -1, "end": -1,
                                                                     "basis": "missing_source", "verified": False}
            missing_source += source is None
            verified += located["verified"]
            article = articles.get(cid)
            missing_candidate += article is None
            body = (article or {}).get("article_body", "")
            rows.append({
                "sample_id": f"{rule}-{index:03d}", "rule": rule, "query_id": qid, "record_id": query.get("record_id"),
                "component_id": query.get("component_id"), "all_rule_ids": "|".join(sorted(pair_rules[(qid, cid)])),
                "section": witness.get("section"), "span": json.dumps(list(witness.get("span", []))),
                "span_basis": located["basis"], "surface_verified": located["verified"],
                "cited_surface": located["surface"],
                "context": _context(source.get("raw_published_text") or "", located["start"], located["end"], context_chars) if source else "",
                "candidate_id": cid,
                "candidate_instrument": titles.get((article or {}).get("instrument_id")) or instrument_code(cid),
                "candidate_article_label": (article or {}).get("article_label"),
                "candidate_article_excerpt": _one_line(body[:excerpt_chars]) if article else "MISSING FROM BOUND CATALOGUE",
                "source_url": (source or {}).get("source_url"), "documentary_verdict": "", "note": "",
            })
        per_rule_summary[rule] = {
            "population_pairs": len(population), "population_occurrences": sum(len(units[k]) for k in population),
            "population_queries": len({k[1] for k in population}),
            "population_components": len({queries.get(k[1], {}).get("component_id") for k in population}),
            "sampled": len(chosen[rule]), "sampled_components": len({queries.get(k[1], {}).get("component_id") for k in chosen[rule]}),
            "surface_verified": verified, "missing_source": missing_source, "missing_candidate": missing_candidate,
        }
    summary = {
        "audit": "precision_review_sheet", "release_id": _release_id(release_dir), "seed": seed,
        "sampling": "unique (rule, query_id, candidate_id) units ordered by SHA-256(seed|rule|query_id|candidate_id); first per_rule",
        "rules": per_rule_summary, "observed_rule_ids": dict(sorted(observed.items())),
        "requested_rules_absent": [r for r in rules if r not in observed],
        "unrequested_rules": sorted(set(observed) - set(rules)), "rows": len(rows),
        "verdicts": list(PRECISION_VERDICTS),
        "catalogue": {"articles": str(catalogue.articles), "articles_declared_in_manifest": catalogue.articles_declared},
        "scope": "Documentary identity of the cited expression; not legal relevance or completeness.",
    }
    names = ["precision_review_sheet.csv", "precision_review_sheet.md", "precision_review_sheet.json",
             "precision_review_sheet.manifest.json"]
    out = _Outputs(out_dir, names)
    summary["provenance"] = _provenance(
        "precision_review_sheet", {**_release_inputs(release_dir, "queries.jsonl", "references.jsonl"),
                                   "sources": Path(sources_jsonl), "catalogue/articles.jsonl": catalogue.articles,
                                   "catalogue/instruments.jsonl": catalogue.instruments},
        seed=seed, rules=list(rules), per_rule=per_rule, context_chars=context_chars, excerpt_chars=excerpt_chars)
    out.csv(names[0], PRECISION_COLUMNS, rows)
    out.text(names[1], _precision_markdown(summary, rows))
    out.json(names[2], summary)
    out.manifest("precision_review_sheet", summary["provenance"])
    return summary


def _precision_markdown(summary: dict, rows: list[dict]) -> str:
    lines = [
        f"# Precision review sheet: {summary['release_id']}", "",
        "Documentary identity check, not legal relevance. For each row decide whether the cited expression in the "
        "source designates the resolved candidate (same instrument and article). Record `documentary_verdict` "
        f"({' / '.join(PRECISION_VERDICTS)}) and `note` in the CSV. The context marks the cited surface with ⟦ ⟧.",
        "", f"Seed `{summary['seed']}`. Sampling: {summary['sampling']}.", "",
        "| Rule | Population pairs | Queries | Sampled | Surface verified |", "|---|---:|---:|---:|---:|",
    ]
    for rule, block in summary["rules"].items():
        lines.append(f"| {rule} | {block['population_pairs']} | {block['population_queries']} | {block['sampled']} | {block['surface_verified']} |")
    if summary["unrequested_rules"]:
        lines += ["", f"Rules present but not sampled: {', '.join(summary['unrequested_rules'])}."]
    current = None
    for row in rows:
        if row["rule"] != current:
            current = row["rule"]
            lines += ["", f"## {current}"]
        lines += [
            "", f"### {row['sample_id']} · {_md(row['query_id'])} → {_md(row['candidate_instrument'])}, art. {_md(row['candidate_article_label'])}",
            "", f"- Cited surface: `{_md(row['cited_surface'])}` ({row['section']}, span {row['span']}, {row['span_basis']})",
            f"- Context: {_md(row['context'])}",
            f"- Candidate: `{_md(row['candidate_id'])}` (rules {_md(row['all_rule_ids'])})",
            f"- Article excerpt: {_md(row['candidate_article_excerpt'])}",
            "- documentary_verdict: ____ · note: ____",
        ]
    return "\n".join(lines) + "\n"


POOL_COLUMNS = ("pool_id", "query_id", "query_text", "candidate_id", "candidate_instrument",
                "candidate_article_label", "candidate_article_excerpt", "relevance_verdict", "note")


def unjudged_pool_sheet(run_dir: Path, release_dir: Path, systems: Sequence[str], depth: int = 10,
                        per_system: int = 30, seed: str = DEFAULT_SEED, out_dir: Path | None = None, *,
                        excerpt_chars: int = 600, articles_jsonl: Path | None = None,
                        instruments_jsonl: Path | None = None) -> dict:
    """Pool uncited candidates ranked in each system's top ``depth`` and sample ``per_system`` per system.

    Cited means a release positive or a ``positive_ids`` entry in the system's details. Each system's sample
    is the first ``per_system`` pool pairs by SHA-256(seed|system|query_id|candidate_id). The reviewer sheet is
    blind to systems, ranks and cited articles and is ordered by SHA-256(seed|review|query_id|candidate_id);
    ``unjudged_pool_key.jsonl`` maps each ``pool_id`` back to its systems and ranks.
    """
    run_dir, release_dir = Path(run_dir), Path(release_dir)
    queries = {q["query_id"]: q for q in read_jsonl(release_dir / "queries.jsonl")}
    cited = {r["query_id"]: set(_positive_ids(r)) for r in read_jsonl(release_dir / "references.jsonl")}
    details_paths = {s: run_dir / "details" / f"{s}.jsonl" for s in systems}
    rankings: dict[str, dict[str, list[str]]] = {}
    per_system_summary, samples = {}, {}
    for system, path in details_paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        ranks: dict[str, list[str]] = {}
        pool: list[tuple[str, str, int]] = []
        unknown = 0
        for row in read_jsonl(path):
            qid = row["query_id"]
            if qid not in queries:
                unknown += 1
                continue
            ranks[qid] = list(row.get("top100") or [])
            seen_cited = cited.get(qid, set()) | set(row.get("positive_ids") or [])
            pool.extend((qid, cid, rank) for rank, cid in enumerate(ranks[qid][:depth], 1) if cid not in seen_cited)
        rankings[system] = ranks
        ordered = sorted(pool, key=lambda t: _order_key(seed, system, t[0], t[1]))
        samples[system] = ordered[:per_system]
        slots = sum(min(depth, len(v)) for v in ranks.values())
        per_system_summary[system] = {
            "queries": len(ranks), "unknown_queries": unknown, "top_depth_slots": slots, "pool_pairs": len(pool),
            "uncited_share_of_slots": len(pool) / slots if slots else None,
            "pool_queries": len({t[0] for t in pool}), "sampled": len(samples[system]),
            "sampled_queries": len({t[0] for t in samples[system]}),
        }
    pairs = sorted({(t[0], t[1]) for sample in samples.values() for t in sample},
                   key=lambda p: _order_key(seed, "review", *p))
    catalogue = bound_catalogue(release_dir, articles_jsonl, instruments_jsonl)
    articles = _load_articles(catalogue.articles, {cid for _, cid in pairs})
    titles = {iid: row.get("title") for iid, row in _load_instruments(catalogue.instruments).items()}
    sheet, key_rows = [], []
    for index, (qid, cid) in enumerate(pairs, 1):
        article = articles.get(cid)
        pool_id = f"U{index:04d}"
        sheet.append({
            "pool_id": pool_id, "query_id": qid, "query_text": _one_line(queries[qid]["text"]), "candidate_id": cid,
            "candidate_instrument": titles.get((article or {}).get("instrument_id")) or instrument_code(cid),
            "candidate_article_label": (article or {}).get("article_label"),
            "candidate_article_excerpt": _one_line((article or {}).get("article_body", "")[:excerpt_chars]) if article else "MISSING FROM BOUND CATALOGUE",
            "relevance_verdict": "", "note": "",
        })
        key_rows.append({
            "pool_id": pool_id, "query_id": qid, "candidate_id": cid,
            "component_id": queries[qid].get("component_id"), "stratum": queries[qid].get("stratum"),
            "resolution_status": queries[qid].get("resolution_status"), "n_unresolved": queries[qid].get("n_unresolved"),
            "sampled_by": sorted(s for s, sample in samples.items() if any(t[0] == qid and t[1] == cid for t in sample)),
            "ranks": {s: (rankings[s].get(qid, []).index(cid) + 1 if cid in rankings[s].get(qid, []) else None) for s in systems},
            "cited_candidate_ids": sorted(cited.get(qid, set())),
        })
    overlap = Counter(len(r["sampled_by"]) for r in key_rows)
    summary = {
        "audit": "unjudged_pool_sheet", "release_id": _release_id(release_dir), "run_dir": str(run_dir),
        "systems": per_system_summary, "depth": depth, "per_system": per_system, "seed": seed,
        "review_rows": len(sheet), "rows_by_number_of_sampling_systems": {str(k): overlap[k] for k in sorted(overlap)},
        "verdicts": list(POOL_VERDICTS),
        "scope": "Documentary plausibility of uncited top-ranked articles given the masked facts; not legal adjudication.",
    }
    names = ["unjudged_pool_sheet.csv", "unjudged_pool_sheet.md", "unjudged_pool_key.jsonl", "unjudged_pool_sheet.json",
             "unjudged_pool_sheet.manifest.json"]
    out = _Outputs(out_dir, names)
    summary["provenance"] = _provenance(
        "unjudged_pool_sheet", {**_release_inputs(release_dir, "queries.jsonl", "references.jsonl"),
                                **{f"details/{s}.jsonl": p for s, p in details_paths.items()},
                                "catalogue/articles.jsonl": catalogue.articles,
                                "catalogue/instruments.jsonl": catalogue.instruments},
        seed=seed, systems=list(systems), depth=depth, per_system=per_system, excerpt_chars=excerpt_chars)
    out.csv(names[0], POOL_COLUMNS, sheet)
    out.text(names[1], _pool_markdown(summary, sheet))
    out.jsonl(names[2], key_rows)
    out.json(names[3], summary)
    out.manifest("unjudged_pool_sheet", summary["provenance"])
    return summary


def _pool_markdown(summary: dict, sheet: list[dict]) -> str:
    lines = [
        f"# Unjudged pool review: {summary['release_id']}", "",
        "Each row pairs masked facts with an article the tesis did not cite but a system ranked highly. Judge "
        "only whether the article is plausibly relevant to the facts; this is documentary inspection, not legal "
        f"adjudication. Record `relevance_verdict` ({' / '.join(POOL_VERDICTS)}) and `note` in the CSV. Systems, "
        "ranks and cited articles are withheld here and kept in `unjudged_pool_key.jsonl`.", "",
    ]
    for row in sheet:
        lines += [
            f"## {row['pool_id']} · {_md(row['candidate_instrument'])}, art. {_md(row['candidate_article_label'])}", "",
            f"- Facts ({_md(row['query_id'])}): {_md(row['query_text'])}",
            f"- Article excerpt: {_md(row['candidate_article_excerpt'])}",
            "- relevance_verdict: ____ · note: ____", "",
        ]
    return "\n".join(lines)


def _token_set(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"\w+", _fold(text.replace(MASK_TOKEN, " "), casefold=True)[0]))


def near_duplicate_pairs(token_sets: Sequence[frozenset[str]], threshold: float) -> list[tuple[int, int, float]]:
    """All index pairs with token-set Jaccard >= ``threshold`` (exact prefix filtering)."""
    frequency = Counter(t for s in token_sets for t in s)
    index: dict[str, list[int]] = defaultdict(list)
    pairs = []
    for i, tokens in enumerate(token_sets):
        if not tokens:
            continue
        ranked = sorted(tokens, key=lambda t: (frequency[t], t))
        prefix = len(ranked) - math.ceil(threshold * len(ranked) - 1e-9) + 1
        candidates: set[int] = set()
        for token in ranked[:prefix]:
            candidates.update(index[token])
            index[token].append(i)
        for j in sorted(candidates):
            inter = len(tokens & token_sets[j])
            jaccard = inter / (len(tokens) + len(token_sets[j]) - inter)
            if jaccard >= threshold:
                pairs.append((j, i, jaccard))
    return pairs


def _label_number(label: str | None) -> str | None:
    match = re.match(r"\d+", label or "")
    return match.group(0) if match else None


_ARTICLE_PREFIX = r"(?<!\w)(?:articulos?|arts?\.?)\s*(?:(?:numeros?|num\.?)\s+)?"


def leakage_checks(release_dir: Path, out_dir: Path | None = None, *, articles_jsonl: Path | None = None,
                   instruments_jsonl: Path | None = None, near_duplicate_jaccard: float = 0.8,
                   max_examples: int = 20) -> dict:
    """Hard leakage and integrity checks plus near-duplicate diagnostics; ``passed`` covers hard checks only.

    Masked text must not contain a positive's instrument title/safe alias or an 'articulo <number>' surface
    of a positive; folds must be component-pure, agree with ``folds.json`` and reproduce ``fold_of``; ids must be
    unique; every positive must exist in the bound catalogue.
    """
    release_dir = Path(release_dir)
    queries = list(read_jsonl(release_dir / "queries.jsonl"))
    references = list(read_jsonl(release_dir / "references.jsonl"))
    folds_path = release_dir / "folds.json"
    folds = json.loads(folds_path.read_text(encoding="utf-8")) if folds_path.exists() else None
    target_types_path = release_dir / "target_types.jsonl"
    typed = {r["candidate_id"] for r in read_jsonl(target_types_path)} if target_types_path.exists() else None
    catalogue = bound_catalogue(release_dir, articles_jsonl, instruments_jsonl)
    articles = _load_articles(catalogue.articles)
    instruments = _load_instruments(catalogue.instruments)
    full_masker = v3_masking.CatalogueMasker(list(instruments.values())) if instruments else None
    maskers: dict[str, v3_masking.CatalogueMasker] = {}
    refs = {}
    for r in references:
        refs.setdefault(r["query_id"], r)

    checks: dict[str, dict] = {}
    violations_by_check: dict[str, list[dict]] = {}

    def check(name: str, severity: str, definition: str, violations: list[dict]) -> None:
        violations_by_check[name] = violations
        checks[name] = {"severity": severity, "definition": definition, "passed": not violations,
                        "violations": len(violations), "examples": violations[:max_examples]}

    query_fields = {f.name: f.default is MISSING and f.default_factory is MISSING for f in fields(QueryRecord)}
    required = [k for k, needed in query_fields.items() if needed]
    positive_fields = [f.name for f in fields(Positive)]
    check("schema", "hard", "Every query has the required QueryRecord fields; every positive has the Positive fields.",
          [{"query_id": q.get("query_id"), "missing": [k for k in required if k not in q]} for q in queries if any(k not in q for k in required)]
          + [{"query_id": r.get("query_id"), "candidate_id": p.get("candidate_id"), "missing": [k for k in positive_fields if k not in p]}
             for r in references for p in r.get("positives", []) if any(k not in p for k in positive_fields)])
    schema_notes = {"query_fields_missing_optional": sorted({k for q in queries for k, needed in query_fields.items()
                                                             if not needed and k not in q}),
                    "query_extra_fields": sorted({k for q in queries for k in q if k not in query_fields}),
                    "flag_keys": sorted({k for q in queries for k in (q.get("flags") or {})})}
    qid_counts = Counter(q["query_id"] for q in queries)
    check("query_ids_unique", "hard", "query_id is unique in queries.jsonl.",
          [{"query_id": k, "count": v} for k, v in sorted(qid_counts.items()) if v > 1])
    ref_counts = Counter(r["query_id"] for r in references)
    check("references_align", "hard", "references.jsonl has exactly one row per query, in query order.",
          ([{"issue": "order_differs"}] if [q["query_id"] for q in queries] != [r["query_id"] for r in references] else [])
          + [{"query_id": k, "issue": "duplicate_reference", "count": v} for k, v in sorted(ref_counts.items()) if v > 1]
          + [{"query_id": k, "issue": "missing_reference"} for k in sorted(set(qid_counts) - set(ref_counts))]
          + [{"query_id": k, "issue": "reference_without_query"} for k in sorted(set(ref_counts) - set(qid_counts))])
    check("text_sha256", "hard", "text_sha256 equals SHA-256 of the UTF-8 text.",
          [{"query_id": q["query_id"]} for q in queries if q.get("text_sha256") != sha256_text(q.get("text", ""))])
    check("positives_present", "hard", "Every query has at least one resolved positive.",
          [{"query_id": q["query_id"]} for q in queries if not _positive_ids(refs.get(q["query_id"], {}))])
    resolution_bad = []
    for q in queries:
        ref = refs.get(q["query_id"], {})
        n = q.get("n_unresolved")
        listed = len(ref.get("unresolved", []))
        status = q.get("resolution_status")
        ok = status in RESOLUTION_STATUSES and (
            (status == FULL and n == 0 and listed == 0) or (status == PARTIAL and isinstance(n, int) and n >= 1 and listed == n))
        if not ok:
            resolution_bad.append({"query_id": q["query_id"], "resolution_status": status, "n_unresolved": n, "unresolved_listed": listed})
    check("resolution_consistent", "hard", "full: n_unresolved == 0 and no unresolved rows; partial: n_unresolved == len(unresolved) >= 1.",
          resolution_bad)
    check("stratum_mask_consistent", "hard", f"{S1_NO_CITATION} text has no mask; {S2_MASKED_CITATION} text has at least one.",
          [{"query_id": q["query_id"], "stratum": q.get("stratum"), "masks": q["text"].count(MASK_TOKEN)} for q in queries
           if not ((q.get("stratum") == S1_NO_CITATION and MASK_TOKEN not in q["text"])
                   or (q.get("stratum") == S2_MASKED_CITATION and MASK_TOKEN in q["text"]))])

    missing = [{"query_id": r["query_id"], "candidate_id": cid} for r in references for cid in _positive_ids(r) if cid not in articles]
    if not catalogue.articles_declared:
        missing.insert(0, {"issue": "articles_sha256_not_declared_in_manifest", "articles": str(catalogue.articles)})
    check("positives_in_bound_catalogue", "hard", "Every positive candidate_id is in the manifest-bound articles.jsonl, whose SHA-256 the manifest declares.",
          missing)
    if typed is not None:
        check("target_types_cover_positives", "hard", "Every cited candidate has a target_types.jsonl row.",
              [{"candidate_id": cid} for cid in sorted({c for r in references for c in _positive_ids(r)} - typed)])

    component_folds: dict[str, set] = defaultdict(set)
    for q in queries:
        component_folds[q.get("component_id")].add(q.get("fold"))
    check("folds_component_pure", "hard", "All queries of a component share one fold.",
          [{"component_id": c, "folds": sorted(map(str, f))} for c, f in sorted(component_folds.items(), key=lambda t: str(t[0])) if len(f) > 1])
    fold_issues = []
    reproduce_issues = []
    if folds is None:
        fold_issues.append({"issue": "folds.json missing"})
    else:
        assigned = folds.get("components", {})
        k = folds.get("k")
        for q in queries:
            if q.get("component_id") not in assigned:
                fold_issues.append({"query_id": q["query_id"], "issue": "component_not_in_folds_json"})
            elif assigned[q["component_id"]] != q.get("fold"):
                fold_issues.append({"query_id": q["query_id"], "issue": "fold_differs", "query_fold": q.get("fold"),
                                    "folds_json": assigned[q["component_id"]]})
        for component, value in sorted(assigned.items()):
            if not isinstance(value, int) or not isinstance(k, int) or not 0 <= value < k:
                fold_issues.append({"component_id": component, "issue": "fold_out_of_range", "fold": value})
            elif folds.get("salt") is None or fold_of(component, folds["salt"], k) != value:
                reproduce_issues.append({"component_id": component, "fold": value})
    check("folds_json_consistent", "hard", "Each query's fold equals folds.json[components][component_id], within [0, k).", fold_issues)
    check("fold_assignment_reproducible", "hard", "folds.json folds equal spec.fold_of(component_id, salt, k).", reproduce_issues)

    shared_keys: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in queries:
        shared_keys[("text_sha256", q.get("text_sha256") or sha256_text(q["text"]))].append(q)
        if q.get("record_id") is not None:
            shared_keys[("record_id", str(q["record_id"]))].append(q)
    cross_fold, same_fold = [], []
    for (kind, value), members in sorted(shared_keys.items()):
        components = sorted({str(q.get("component_id")) for q in members})
        if len(components) < 2:
            continue
        folds_seen = sorted({str(q.get("fold")) for q in members})
        row = {kind: value, "query_ids": sorted(q["query_id"] for q in members), "components": components, "folds": folds_seen}
        (cross_fold if len(folds_seen) > 1 else same_fold).append(row)
    check("no_duplicate_across_folds", "hard",
          "An identical masked text or a repeated record_id never appears in components of different folds.", cross_fold)
    check("duplicates_across_components_same_fold", "diagnostic",
          "Identical masked texts or repeated record_ids split over components that share a fold.", same_fold)

    title_hits, surface_hits, masker_hits = [], [], []
    for q in queries:
        text = q["text"]
        ref = refs.get(q["query_id"], {})
        positive_ids = _positive_ids(ref)
        positive_instruments = sorted({(articles.get(cid) or {}).get("instrument_id") or cid.split("#", 1)[0] for cid in positive_ids})
        for iid in positive_instruments:
            if iid not in instruments:
                continue
            if iid not in maskers:
                maskers[iid] = v3_masking.CatalogueMasker([instruments[iid]])
            for span in maskers[iid].mask(text)["spans"]:
                if "catalogue_identity" in span["reasons"]:
                    s, e = span["original_start"], span["original_end"]
                    title_hits.append({"query_id": q["query_id"], "instrument_id": iid, "surface": text[s:e], "snippet": _snippet(text, s, e)})
        folded, starts, ends = _fold(text, casefold=True)
        article_spans = []
        if full_masker is not None:
            for span in full_masker.mask(text)["spans"]:
                s, e = span["original_start"], span["original_end"]
                masker_hits.append({"query_id": q["query_id"], "reasons": span["reasons"], "surface": text[s:e], "snippet": _snippet(text, s, e)})
                if "explicit_article_expression" in span["reasons"]:
                    article_spans.append((s, e))
        for cid in positive_ids:
            number = _label_number((articles.get(cid) or {}).get("article_label"))
            if number is None:
                continue
            found = [(starts[m.start()], ends[m.end() - 1]) for m in re.finditer(_ARTICLE_PREFIX + number + r"(?!\d)", folded)]
            for s, e in article_spans:
                inner = _fold(text[s:e], casefold=True)[0]
                body = re.sub(r"^" + _ARTICLE_PREFIX, "", inner)
                if re.search(r"(?<![\w-])" + number + r"(?!\d)", body) and not any(a <= s < b for a, b in found):
                    found.append((s, e))
            for s, e in found:
                surface_hits.append({"query_id": q["query_id"], "candidate_id": cid, "surface": text[s:e], "snippet": _snippet(text, s, e)})
    check("no_positive_title_in_text", "hard", "No masked text contains a positive's instrument title or safe alias as the v3 masker matches it.",
          title_hits)
    check("no_positive_article_surface_in_text", "hard",
          "No masked text contains 'articulo/art./arts. <number of a positive's label>', alone or in an article list.", surface_hits)
    check("masker_idempotent", "hard", "Re-running the v3 masker with the bound catalogue on masked text finds nothing left to mask.",
          masker_hits if full_masker is not None else [{"issue": "no instruments.jsonl bound; check not run"}])

    token_sets = [_token_set(q["text"]) for q in queries]
    near = []
    for i, j, jaccard in near_duplicate_pairs(token_sets, near_duplicate_jaccard):
        a, b = queries[i], queries[j]
        if a.get("component_id") == b.get("component_id") or (a.get("text_sha256") and a.get("text_sha256") == b.get("text_sha256")):
            continue
        near.append({"query_ids": [a["query_id"], b["query_id"]], "components": [a.get("component_id"), b.get("component_id")],
                     "folds": [a.get("fold"), b.get("fold")], "cross_fold": a.get("fold") != b.get("fold"), "jaccard": round(jaccard, 4)})
    near.sort(key=lambda r: (not r["cross_fold"], -r["jaccard"], r["query_ids"]))
    check("near_duplicates_across_components", "diagnostic",
          f"Masked-text token-set Jaccard >= {near_duplicate_jaccard} between different components (cross_fold marks fold leakage).", near)

    hard_failures = sorted(n for n, c in checks.items() if c["severity"] == "hard" and not c["passed"])
    summary = {
        "audit": "leakage_checks", "release_id": _release_id(release_dir), "passed": not hard_failures,
        "hard_failures": hard_failures, "checks": checks, "schema_notes": schema_notes,
        "counts": {"queries": len(queries), "references": len(references), "components": len(component_folds),
                   "near_duplicate_pairs_cross_fold": sum(r["cross_fold"] for r in near)},
        "catalogue": {"articles": str(catalogue.articles), "articles_declared_in_manifest": catalogue.articles_declared,
                      "instruments": str(catalogue.instruments) if catalogue.instruments else None,
                      "instruments_declared_in_manifest": catalogue.instruments_declared},
    }
    names = ["leakage_checks.json", "leakage_violations.jsonl", "leakage_checks.manifest.json"]
    out = _Outputs(out_dir, names)
    summary["provenance"] = _provenance(
        "leakage_checks", {**_release_inputs(release_dir, "queries.jsonl", "references.jsonl", "folds.json", "target_types.jsonl"),
                           "catalogue/articles.jsonl": catalogue.articles, "catalogue/instruments.jsonl": catalogue.instruments},
        near_duplicate_jaccard=near_duplicate_jaccard, max_examples=max_examples)
    out.json(names[0], summary)
    out.jsonl(names[1], ({"check": name, **v} for name, rows in violations_by_check.items() for v in rows))
    out.manifest("leakage_checks", summary["provenance"])
    return summary


def _headline(summary: dict) -> dict:
    audit = summary.get("audit")
    if audit == "leakage_checks":
        return {"passed": summary["passed"], "hard_failures": summary["hard_failures"],
                "violations": {k: v["violations"] for k, v in summary["checks"].items()}}
    if audit == "policy_agreement":
        return {"shared_keys": summary["shared_keys"], **summary["overall"]}
    if audit == "residual_cue_report":
        return {"queries": summary["queries"], "headline": summary["headline"],
                "cue_queries": {k: v["queries"] for k, v in summary["cues"].items()}}
    if audit == "precision_review_sheet":
        return {"rows": summary["rows"], "rules": summary["rules"], "unrequested_rules": summary["unrequested_rules"]}
    return {k: v for k, v in summary.items() if k != "provenance"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m uol.facts_to_law.audit", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("residual", help="residual-cue report")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--examples", type=int, default=10)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p = sub.add_parser("agreement", help="reference-policy agreement with another release")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--other-queries", type=Path, required=True)
    p.add_argument("--other-references", type=Path, required=True)
    p.add_argument("--key", choices=("query_id", "record_id"), default="query_id")
    p.add_argument("--name", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top", type=int, default=50)
    p = sub.add_parser("precision", help="per-rule documentary precision review sheet")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rules", nargs="+", default=list(DEFAULT_RULES))
    p.add_argument("--per-rule", type=int, default=40)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p = sub.add_parser("unjudged", help="pooled uncited top-ranked candidates per system")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--systems", nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--per-system", type=int, default=30)
    p.add_argument("--seed", default=DEFAULT_SEED)
    p = sub.add_parser("leakage", help="leakage and integrity checks")
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--near-duplicate-jaccard", type=float, default=0.8)
    args = parser.parse_args(argv)
    if args.command == "residual":
        summary = residual_cue_report(args.release, args.out, examples=args.examples, seed=args.seed)
    elif args.command == "agreement":
        summary = policy_agreement(args.release, args.other_queries, args.other_references, args.key, args.out,
                                   name=args.name, top=args.top)
    elif args.command == "precision":
        summary = precision_review_sheet(args.release, args.sources, args.rules, args.per_rule, args.seed, args.out)
    elif args.command == "unjudged":
        summary = unjudged_pool_sheet(args.run, args.release, args.systems, args.depth, args.per_system, args.seed, args.out)
    else:
        summary = leakage_checks(args.release, args.out, near_duplicate_jaccard=args.near_duplicate_jaccard)
    sys.stdout.write(json.dumps(_headline(summary), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
