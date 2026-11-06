"""Graded documentary reference resolver for the facts-to-law v2 release (C01 v11).

The pinned v5 pipeline (v2 grammar, v5 grouped numerals, v3 formal-title filter)
supplies rule R0 unchanged: ``_extract_section`` from benchmark_repair_v5 is called
per Criterio/Justificacion section. The per-source loop is re-implemented here only
because v2 ``extract_source`` admits original TRAIN rows alone; its version-cue
regex is copied verbatim for the same reason. Later rules revisit only expressions
that v5 left unresolved, and every positive records the rules that produced it:

R0  explicit_forward_binding    v5 output, unchanged
R1  constitutional_adjective    'artículo N constitucional'; bare CPEUM aliases
R2  reverse_binding             '<law>, en su artículo N' (law at most 40 chars before)
R3  in_section_anaphora         'de la citada ley' -> latest compatible explicit binding
R4  declared_abbreviation       reviewed acronym table ('de la LFT')
R5  bound_numeric_continuation  continuation with its own R0/R1/R4/R10 binding
R6  small_range_expansion       'artículos N a M', plain integers, M-N <= 9
R7  coordinated_binding         '..., ambos/todos de la Ley X' binds the whole list
R8  article_anaphora            'el citado artículo N' -> earlier unique binding of N
R9  materia_amparo              'ley de la materia' in a Común tesis, no other law bound
R10 attributive_title           'de la citada Ley Federal del Trabajo' (explicit title)
R11 alternative_designator      'precepto/diverso/dispositivo/numeral N' + explicit binding
R12 qualifier_grammar           'fracción VI, segundo párrafo', 'parte final', 'XXII Bis', 'o'

Every remaining expression is classified as ``ignorable`` (not a citation by this
tesis), ``out_of_scope`` (a citation the candidate universe cannot contain) or
``missing_positive`` (a possible catalogue citation left unresolved). Labels,
observed aliases and old qrels are never read.
"""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from uol.retrieval_benchmarks.benchmark_repair import references as v2
from uol.retrieval_benchmarks.benchmark_repair_v3 import references as v3
from uol.retrieval_benchmarks.benchmark_repair_v5 import references as v5

RESOLVER_VERSION = "facts-to-law-v2-resolver-1.0"
BASE_PARSER_VERSION = v5.PARSER_VERSION
CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

RULES = {
    "R0": "explicit_forward_binding",
    "R1": "constitutional_adjective",
    "R2": "reverse_binding",
    "R3": "in_section_anaphora",
    "R4": "declared_abbreviation",
    "R5": "bound_numeric_continuation",
    "R6": "small_range_expansion",
    "R7": "coordinated_binding",
    "R8": "article_anaphora",
    "R9": "materia_amparo",
    "R10": "attributive_title",
    "R11": "alternative_designator",
    "R12": "qualifier_grammar",
}
IGNORABLE, OUT_OF_SCOPE, MISSING = "ignorable", "out_of_scope", "missing_positive"
DISPOSITIONS = (IGNORABLE, OUT_OF_SCOPE, MISSING)
SECTION_NAMES = {"criterio_juridico": "criterio", "justificacion": "justificacion"}
MAX_RANGE_SPAN = 9
REVERSE_GAP = 40
ANAPHORA_WINDOW = 600

# Reviewed acronym table. Each value is the catalogue file stem; build_catalogue
# verifies that exactly one instrument has it. Never derived from labels or data.
DECLARED_ABBREVIATIONS = {
    "LFT": "LFT.pdf", "CFF": "CFF.pdf", "CNPP": "CNPP.pdf", "CPF": "CPF.pdf",
    "CFPC": "CFPC.pdf", "CNPCF": "CNPCF.pdf", "CCF": "CCF.pdf", "LFPCA": "LFPCA.pdf",
    "LISR": "LISR.pdf", "LIVA": "LIVA.pdf", "LGSM": "LGSM.pdf", "LIC": "LIC.pdf",
    "LOPJF": "LOPJF.pdf", "LGTOC": "LGTOC.pdf", "LFPA": "LFPA.pdf", "LNEP": "LNEP.pdf",
    "LSS": "LSS.pdf", "LGS": "LGS.pdf", "LISSSTE": "LISSSTE.pdf", "LGRA": "LGRA.pdf",
    "LNSIJPA": "LNSIJPA.pdf", "LOTFJA": "LOTFJA.pdf", "LGTAIP": "LGTAIP.pdf",
    "LGIPE": "LGIPE.pdf", "LNED": "LNED.pdf", "LAASSP": "LAASSP.pdf", "LGV": "LGV.pdf",
}
# Declared short names ('Ley del ISSSTE'), folded; same verification.
DECLARED_SHORT_TITLES = {
    "ley del issste": "LISSSTE.pdf", "ley del isr": "LISR.pdf",
    "ley del iva": "LIVA.pdf", "ley del ieps": "LIEPS.pdf",
}

# Copied verbatim from v2.extract_source, where it is inline.
VERSION_CUE = re.compile(r"\b(?:vigente(?:s)?\s+(?:hasta|antes|al|en)|texto\s+(?:anterior|vigente)|reformad[oa]s?|derogad[oa]s?|abrogad[oa]s?)\b")

_STATES = (r"aguascalientes|baja\s+california(?:\s+sur)?|campeche|chiapas|chihuahua|coahuila|colima|durango|"
           r"guanajuato|guerrero|hidalgo|jalisco|estado\s+de\s+mexico|michoacan|morelos|nayarit|nuevo\s+leon|"
           r"oaxaca|puebla|queretaro|quintana\s+roo|san\s+luis\s+potosi|sinaloa|sonora|tabasco|tamaulipas|"
           r"tlaxcala|veracruz|yucatan|zacatecas|ciudad\s+de\s+mexico|distrito\s+federal")
LOCAL_MARKER = re.compile(
    r"\b(?:local(?:es)?|estatal(?:es)?|municipal(?:es)?|municipios?|del\s+estado|estado\s+libre|"
    r"para\s+el\s+estado|de\s+la\s+entidad|entidad(?:es)?\s+federativas?|de\s+la\s+ciudad|del\s+distrito|"
    r"burocratic[oa]\s+(?:local|estatal)|" + _STATES + r")\b")
REL = r"(?:de\s+la|de\s+los|de\s+las|del|de)"
REL_GAP = re.compile(r"[\s,)]*" + REL + r"\s+")
COORDINATED = re.compile(
    r"[\s,)]*(?:ambos|ambas|todos|todas)(?:\s+(?:ellos|ellas|preceptos|articulos|numerales|dispositivos))?\s+"
    + REL + r"\s+")
CONSTITUTIONAL_ADJECTIVE = re.compile(r"[\s,)]*constitucional(?:es)?(?!\w)")
CPEUM_ALIAS = re.compile(
    r"(?:constitucion(?:\s+politica)?(?:\s+(?:federal|general|mexicana|nacional))?|carta\s+magna|"
    r"carta\s+fundamental|norma\s+fundamental|norma\s+suprema|ley\s+fundamental|ley\s+suprema|"
    r"pacto\s+federal|texto\s+constitucional)(?!\w)")
CPEUM_CONTINUATION = re.compile(r"\s+(?:politica\s+)?de\s+los\s+estados\s+unidos\s+mexicanos?\b|\s+(?:general\s+)?de\s+la\s+republica\b|\s+de\s+1917\b")
FOREIGN_CONSTITUTION = re.compile(
    r"\s+(?:espanola|colombiana|argentina|chilena|peruana|brasilena|alemana|francesa|italiana|"
    r"estadounidense|norteamericana|americana|europea|de\s+(?:espana|colombia|argentina|chile|peru|"
    r"brasil|alemania|francia|italia|cadiz|apatzingan|bonn|weimar)|de\s+los\s+estados\s+unidos\s+de\s+america|"
    r"(?:federal\s+)?de\s+1[89](?!17)\d\d)\b")
PROPER_COMPLEMENT = re.compile(r"\s+(?:de|del|para)\s+(?:(?:la|las|los|el)\s+)?[A-ZÁÉÍÓÚÑ]")
LOWER_COMPLEMENT = re.compile(r"\s+(?:de|del)\s+(?:(?:la|las|los|el|un|una)\s+)?[a-záéíóúñ]")
IN_TITLE = re.compile(
    r"(?:reglamentari[oa]s?|reglamento)\s+(?:de\s+(?:la|las|los)\s+(?:fraccion(?:es)?|apartados?|parrafos?|incisos?)"
    r"\s+[^,.;]{0,60}?\s+)?del?\s+(?:los\s+)?$")
_PARTICIPLE = r"(?:citad|referid|mencionad|aludid|invocad|señalad|indicad|precitad|transcrit|reproducid)"
FEM_NOUN = r"(?:ley|constitucion|norma|legislacion|codificacion|normatividad|normativa)"
MASC_NOUN = r"(?:codigo|ordenamiento|reglamento|cuerpo\s+(?:legal|normativo|de\s+leyes)|estatuto)"
NOUN_MODIFIER = (r"(?:\s+(?:legal|juridic[oa]|federal|general|nacional|procesal|sustantiv[oa]|tributari[oa]|fiscal|"
                 r"mercantil|civil|penal|laboral|agrari[oa]|aduaner[oa]|organic[oa]|reglamentari[oa]|especial|"
                 r"secundari[oa]|adjetiv[oa]|burocratic[oa]|de\s+amparo|supletori[oa]|en\s+materia\s+\w+)){0,2}")
POST_ANAPHOR = rf"(?:en\s+cita|en\s+comento|de\s+referencia|{_PARTICIPLE}[oa]s?|antes\s+{_PARTICIPLE}[oa]s?|previamente\s+{_PARTICIPLE}[oa]s?)"
ANAPHOR = re.compile(
    rf"(?:de\s+la|de\s+las)\s+(?:{_PARTICIPLE}|propi|mism)as?\s+(?P<f1>{FEM_NOUN}){NOUN_MODIFIER}"
    rf"|(?:del|de\s+los)\s+(?:{_PARTICIPLE}|propi|mism)os?\s+(?P<m1>{MASC_NOUN}){NOUN_MODIFIER}"
    rf"|de\s+dichas?\s+(?P<f2>{FEM_NOUN}){NOUN_MODIFIER}"
    rf"|de\s+dichos?\s+(?P<m2>{MASC_NOUN}){NOUN_MODIFIER}"
    rf"|de\s+(?:esa|esta|aquella)\s+(?P<f3>{FEM_NOUN}){NOUN_MODIFIER}"
    rf"|de\s+(?:ese|este|aquel)\s+(?P<m3>{MASC_NOUN}){NOUN_MODIFIER}"
    rf"|(?:de\s+la|de\s+las)\s+(?P<f4>{FEM_NOUN}){NOUN_MODIFIER}\s+{POST_ANAPHOR}"
    rf"|(?:del|de\s+los)\s+(?P<m4>{MASC_NOUN}){NOUN_MODIFIER}\s+{POST_ANAPHOR}"
)
ANAPHOR_LOCAL_TAIL = re.compile(r"\s*,?\s*(?:local(?:es)?|estatal(?:es)?|municipal(?:es)?|del\s+estado|de\s+la\s+entidad|de\s+(?:dicha|esa|la)\s+entidad|de\s+la\s+ciudad)\b")
ATTRIBUTIVE = re.compile(rf"(?:{_PARTICIPLE}|propi|mism|vigente|actual|nuev)[oa]?s?\s+")
ABROGATED_ATTRIBUTIVE = re.compile(r"(?:derogad|abrogad|anterior|antigu)[oa]?s?\s+")
LEY_DE_LA_MATERIA = re.compile(r"ley\s+de\s+la\s+materia(?!\w)")
SU_REGLAMENTO = re.compile(r"(?:su|sus)\s+reglamentos?(?!\w)")
GENERIC_LOCAL = re.compile(rf"(?:la|las|el|los)?\s*(?:{FEM_NOUN}|{MASC_NOUN})(?:\s+\w+){{0,3}}?\s+(?:local(?:es)?|estatal(?:es)?|municipal(?:es)?)\b")
REVERSE_ANCHOR = re.compile(
    r"(?:\b(?:en|conforme\s+a|de\s+conformidad\s+con|acorde\s+(?:a|con)|de\s+acuerdo\s+con|en\s+terminos\s+de|segun|"
    r"por\s+disposicion(?:\s+expresa)?\s+de|con\s+fundamento\s+en|con\s+base\s+en|atento\s+a)\s+sus?\s+)$")
ARTICLE_ANAPHOR_BEFORE = re.compile(rf"(?:\b{_PARTICIPLE}[oa]s?|\bdich[oa]s?|\bpropi[oa]s?)\s+$")
ARTICLE_ANAPHOR_AFTER = re.compile(
    rf"\s*,?\s*(?:(?:antes|previamente|ya)\s+)?(?:{_PARTICIPLE}[oa]s?|en\s+cita|en\s+comento|de\s+referencia|en\s+estudio|en\s+analisis)(?!\w)")
DESIGNATOR = r"(?:diversos?|numeral(?:es)?|preceptos?|dispositivos?|articulos?|arts?\.)"
CHAIN_SEPARATOR = re.compile(
    r"\s*,?\s*(?:en\s+relacion\s+con|asi\s+como|y\s+con|con|en\s+conexion\s+con|en\s+concordancia\s+con|"
    r"adminiculad[oa]s?\s+con|vinculad[oa]s?\s+con|y|e)\s+(?:el|los|la|las)\s+(?:" + DESIGNATOR + r"\s+)?(?=\d)"
    r"|\s*,?\s*(?:en\s+relacion\s+con|asi\s+como)\s+(?:" + DESIGNATOR + r"\s+)?(?=\d)")
DISJUNCTION = re.compile(r"\s+[ou]\s+(?=\d)")
CONTINUATION_BEFORE = re.compile(
    r"(?:[;,]\s*(?:(?:asi\s+como|y|e|o|u)\s+)?|,?\s+(?:asi\s+como|y|e|o|u|en\s+relacion\s+con)\s+|,?\s*en\s+relacion\s+con\s+)"
    r"(?:(?:el|los|la|las)\s+)?(?:(?:diversos?|numeral(?:es)?|preceptos?|dispositivos?)\s+)?$")
_ORD = v2.ORDINAL
_QNOUN = r"(?:parrafos?|fraccion(?:es)?|incisos?|apartados?)"
_ROMAN_NONEMPTY = rf"(?:{v2.ROMAN})(?<=[ivxlcdm])"
ORDINAL_END = re.compile(rf"\b{_ORD}\s*$")
QUALIFIER_TAIL = re.compile(
    rf"\s*,?\s*{_ORD}(?:\s*(?:,\s*|\s+)(?:y|e)\s+|\s*,\s*){_ORD}\s+{_QNOUN}\b"
    rf"|\s*,?\s*(?:parte\s+(?:final|inicial|primera|segunda|ultima|in\s+fine)|in\s+fine|(?:primera|segunda|ultima)\s+parte)\b"
    rf"|\s+{v2.LATIN}\b(?:\s*(?:,\s*(?:(?:y|e)\s+)?|(?:y|e)\s+){_ROMAN_NONEMPTY}(?:\s+{v2.LATIN})?\b)*"
    rf"|\s*,?\s*(?:subincisos?|numeros?)\s+\d+(?:\s*(?:,|y|e)\s*\d+)*\b"
)
NOUN_AFTER_ORDINAL = re.compile(rf"\s+{_QNOUN}\b")
COMPOUND_ORDINAL = re.compile(rf"\s+{_ORD}\b")
BARE_QUALIFIER_NOUN = re.compile(rf"\s+(?={_QNOUN}\b)")
LAW_LIKE = re.compile(
    r"\b(?:Ley|LEY|Código|Codigo|CÓDIGO|CODIGO|Reglamento|REGLAMENTO|Constitución|CONSTITUCIÓN|Estatuto|Convención|"
    r"Convenio|Pacto|Tratado|Acuerdo|Decreto|Lineamientos|Ordenanza)\s+(?:(?:de|del|para|sobre|que|en|General|Federal|"
    r"Nacional|Orgánica)\s+)?(?:(?:el|la|los|las)\s+)?[A-ZÁÉÍÓÚÑ]")
_CAP = r"[A-ZÁÉÍÓÚÑ][\w\-.áéíóúñüÁÉÍÓÚÑÜ]*"
_FUNC = r"(?:de|del|la|las|los|el|y|e|para|en|sobre|a|al|contra|que|o|u|con|se|por|sus|su|lo)"
TITLE_PHRASE = re.compile(rf"{_CAP}(?:\s+(?:{_FUNC}\s+)*{_CAP})*")
INTERNATIONAL_HEAD = re.compile(
    r"(?:convencion|convenio|pacto\s+internacional|pacto\s+de\s+san\s+jose|declaracion|protocolo(?!\s+de\s+actuacion)|"
    r"tratado|carta\s+(?:de\s+las\s+naciones|de\s+la\s+organizacion|democratica|internacional)|estatuto\s+de\s+roma|"
    r"reglas\s+(?:minimas|de\s+las\s+naciones|de\s+(?:beijing|bangkok|brasilia|tokio|mandela))|principios|directrices|"
    r"observacion\s+general|opinion\s+consultiva|recomendacion\s+general)\b")
SUBLEGAL_HEAD = re.compile(
    r"(?:reglamentos?|acuerdos?|lineamientos|decretos?|reglas|norma\s+oficial|normas|resolucion|manual|estatuto|"
    r"circular|disposiciones|bases|condiciones\s+generales|contrato|clasificador|catalogo|programa|plan|criterios|"
    r"codigo\s+de\s+(?:etica|conducta)|protocolo|oficio|instructivo|politicas|tarifa|tabulador)\b")
FEDERAL_HEAD = re.compile(r"(?:ley|codigo)\s+(?:federal|general|nacional|organica\s+del\s+poder\s+judicial\s+de\s+la\s+federacion)\b")
TRANSITORY = re.compile(rf"(?:{_ORD}|unico|[a-z]+(?:\s+[a-z]+)?)\s+transitori[oa]s?\b|transitori[oa]s?\b")
WORD_ORDINAL = re.compile(rf"(?:{_ORD}|unico)\b")
ENUMERATION_AFTER = re.compile(r"\s*(?:\)|\.-|\.\s+(?=[A-ZÁÉÍÓÚÑ¿\"“])|\.\s*$|°\))")
DOCKET_AFTER = re.compile(r"\s*/\s*\d{2,4}\b")
UNIT = re.compile(
    r"(?:\s*(?:,|y|e|o|u|a|al)\s*\d+)*\s*(?:%|por\s+ciento|dias?|meses|mes|anos|ano|horas?|minutos|semanas?|udis|umas?|"
    r"veces|salarios?|pesos|dolares|unidades|metros|kilometros|kilos|kilogramos|gramos|toneladas|litros|hectareas|"
    r"puntos\s+porcentuales|votos|personas|trabajadores|fojas|hojas|paginas|ejemplares|ocasiones|"
    r"de\s+(?:salario|sueldo)|mil|millones)\b")
MONTH = re.compile(r"\b(?:enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|setiembre|octubre|noviembre|diciembre|ejercicios?|anos?)\b")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _kind(title: str) -> str:
    head = v2.fold(title)[0].split()[0]
    return {"ley": "ley", "codigo": "codigo", "constitucion": "constitucion", "reglamento": "reglamento",
            "estatuto": "estatuto", "ordenanza": "ordenanza"}.get(head, "other")


_STOP = frozenset("de del la las los el y e para en sobre a al que o u con se por sus su lo".split())


def _tokens(title: str) -> frozenset[str]:
    return frozenset(t for t in re.findall(r"[a-zñ0-9]+", v2.fold(title)[0]) if t not in _STOP)


@dataclass
class Catalogue:
    """Candidate catalogue view used by the resolver (titles, safe aliases, labels)."""
    instruments: list[dict]
    index: dict[tuple[str, str], list[dict]]
    titles: dict[str, str]
    kinds: dict[str, str]
    abbreviations: dict[str, str]
    short_titles: dict[str, str]
    cpeum: str
    amparo: str
    title_tokens: list[tuple[str, frozenset[str]]]
    missing_abbreviations: list[str] = field(default_factory=list)


def build_catalogue(instruments: Iterable[Mapping], articles: Iterable[Mapping]) -> Catalogue:
    """Index catalogue rows; only titles and declared safe_aliases enter matching."""
    rows = [{"instrument_id": r["instrument_id"], "title": r["title"], "safe_aliases": list(r.get("safe_aliases", []))}
            for r in instruments]
    index: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for article in articles:
        index[(article["instrument_id"], article["article_label"])].append(article)
    by_stem = defaultdict(list)
    for row in rows:
        by_stem[row["instrument_id"].rsplit("/", 1)[-1]].append(row["instrument_id"])

    def resolve(table: Mapping[str, str]) -> tuple[dict[str, str], list[str]]:
        found, missing = {}, []
        for key, stem in table.items():
            ids = by_stem.get(stem, [])
            if len(ids) == 1:
                found[key] = ids[0]
            else:
                missing.append(key)
        return found, missing

    abbreviations, missing_a = resolve(DECLARED_ABBREVIATIONS)
    short_titles, missing_s = resolve(DECLARED_SHORT_TITLES)
    cpeum = by_stem.get("CPEUM.pdf", [None])[0]
    amparo = by_stem.get("LAmp.pdf", [None])[0]
    if not cpeum or not amparo:
        raise ValueError("Catalogue lacks the CPEUM or Ley de Amparo instrument")
    titles = {r["instrument_id"]: r["title"] for r in rows}
    return Catalogue(
        instruments=rows, index=dict(index), titles=titles, kinds={k: _kind(t) for k, t in titles.items()},
        abbreviations=abbreviations, short_titles=short_titles, cpeum=cpeum, amparo=amparo,
        title_tokens=[(k, _tokens(t)) for k, t in titles.items()], missing_abbreviations=missing_a + missing_s)


@dataclass
class Binding:
    outcome: str                      # bound | out_of_scope | missing | ignorable | defer_r9
    rules: tuple[str, ...] = ()
    instrument_id: str | None = None
    start: int = 0                    # folded span of the binding phrase
    end: int = 0
    reason: str | None = None
    coordinated: bool = False


@dataclass
class Entry:
    label: str
    start: int
    end: int
    range_end: str | None = None


@dataclass
class Chain:
    groups: list[list[Entry]]
    ends: list[int]                   # folded end of each group
    end: int
    problem: str | None
    rules: set[str]


class _Section:
    """Folded view of one later section plus the bindings found so far."""

    def __init__(self, raw: str, field: str, base: int, catalogue: Catalogue, mentions: list[dict]):
        self.raw, self.field, self.base, self.cat = raw, field, base, catalogue
        self.text, self.offsets = v2.fold(raw)
        self.first: dict[int, int] = {}
        self.last: dict[int, int] = {}
        for i, position in enumerate(self.offsets):
            self.first.setdefault(position, i)
            self.last[position] = i
        self.mentions = mentions
        self.mention_at = {m["start"]: m for m in mentions}
        self.bindings: list[tuple[int, int, str, str]] = []   # (start, end, instrument_id, rule) explicit only
        self.foreign: list[tuple[int, str]] = []               # (start, kind) bound non-catalogue instruments

    def raw_span(self, start: int, end: int) -> tuple[int, int]:
        left = self.offsets[start] if start < len(self.offsets) else len(self.raw)
        right = self.offsets[end - 1] + 1 if end > start else left
        return left, right

    def folded_span(self, left: int, right: int) -> tuple[int, int]:
        return self.first[left], self.last[right - 1] + 1

    def law_like_between(self, start: int, end: int) -> list[tuple[int, str]]:
        """Instrument-like mentions (catalogue or capitalised titles) inside a folded span."""
        found = [(m["start"], m["instrument_ids"][0] if len(m["instrument_ids"]) == 1 and not m["unsafe_title_extension"]
                  else "?") for m in self.mentions if start <= m["start"] < end]
        left, right = self.raw_span(start, end) if end > start else (0, 0)
        for match in LAW_LIKE.finditer(self.raw, left, right):
            position = self.first.get(match.start())
            if position is not None and not any(m["start"] <= position < m["end"] for m in self.mentions):
                found.append((position, "?"))
        return sorted(found)


def _quoted(section: _Section, start: int, end: int) -> bool:
    left, right = section.raw_span(start, end)
    return v2._quoted_title(section.raw, left, right)


def _title_info(section: _Section, start: int) -> tuple[str | None, str, int]:
    """Classify an instrument name beginning at folded ``start``; returns (kind, folded name, end)."""
    left = section.raw_span(start, start + 1)[0]
    match = TITLE_PHRASE.match(section.raw, left)
    if match is None:
        return None, "", start
    end = section.folded_span(match.start(), match.end())[1]
    name = " ".join(section.text[start:end].split())
    tail = section.text[end:end + 90]
    if INTERNATIONAL_HEAD.match(name):
        return "international", name, end
    if FEDERAL_HEAD.match(name) and not LOCAL_MARKER.search(name.split(" del estado")[0] if " servicio del estado" in name else name):
        return "federal", name, end
    if LOCAL_MARKER.search(name) or LOCAL_MARKER.match(tail.lstrip(" ,")):
        return "local", name, end
    if SUBLEGAL_HEAD.match(name):
        return "sublegal", name, end
    if re.match(r"(?:ley|codigo|constitucion)\b", name) and len(name.split()) >= 2:
        return "federal", name, end
    return None, name, end


def _variant(cat: Catalogue, name: str) -> str | None:
    tokens = _tokens(name)
    best, best_id = 0.0, None
    for instrument_id, title in cat.title_tokens:
        if tokens and title:
            score = len(tokens & title) / len(tokens | title)
            if score > best:
                best, best_id = score, instrument_id
    return best_id if best >= .75 else None


def _target(section: _Section, pos: int, *, reverse: bool = False) -> Binding | None:
    """Instrument designated at folded ``pos`` (after a relation word)."""
    text, cat = section.text, section.cat
    mention = section.mention_at.get(pos)
    if mention is not None:
        if mention["unsafe_title_extension"]:
            return Binding(OUT_OF_SCOPE, end=mention["end"], reason="local_extension_of_catalogue_title")
        if len(mention["instrument_ids"]) != 1:
            return Binding(MISSING, end=mention["end"], reason="ambiguous_instrument_alias")
        return Binding("bound", ("R0",), mention["instrument_ids"][0], end=mention["end"])
    attributive = ATTRIBUTIVE.match(text, pos)
    if attributive and not reverse:
        inner = section.mention_at.get(attributive.end())
        if inner is not None and len(inner["instrument_ids"]) == 1 and not inner["unsafe_title_extension"]:
            return Binding("bound", ("R10",), inner["instrument_ids"][0], end=inner["end"])
    abrogated = ABROGATED_ATTRIBUTIVE.match(text, pos)
    if abrogated and not reverse:
        kind, name, end = _title_info(section, abrogated.end())
        if section.mention_at.get(abrogated.end()) is not None or kind is not None:
            return Binding(OUT_OF_SCOPE, end=max(end, abrogated.end()), reason="abrogated_instrument_version")
    alias = CPEUM_ALIAS.match(text, pos)
    if alias:
        return _cpeum_alias(section, pos, alias.end())
    left = section.raw_span(pos, pos + 1)[0]
    acronym = re.match(r"([A-Z][A-Za-z]{1,7})(?!\w)", section.raw[left:left + 9])
    if acronym and acronym.group(1) in cat.abbreviations:
        end = section.folded_span(left, left + len(acronym.group(1)))[1]
        return Binding("bound", ("R4",), cat.abbreviations[acronym.group(1)], end=end)
    for name, instrument_id in cat.short_titles.items():
        short = re.match(r"\s*".join(map(re.escape, name.split())) + r"(?!\w)", text[pos:])
        if short:
            return Binding("bound", ("R4",), instrument_id, end=pos + short.end())
    if reverse:
        return None
    if LEY_DE_LA_MATERIA.match(text, pos):
        return Binding("defer_r9", end=LEY_DE_LA_MATERIA.match(text, pos).end(), reason="ley_de_la_materia")
    anaphor = ANAPHOR.match(text, pos)
    if anaphor:
        return _anaphora(section, pos, anaphor)
    reglamento = SU_REGLAMENTO.match(text, pos)
    if reglamento:
        return Binding(OUT_OF_SCOPE, end=reglamento.end(), reason="sublegal_instrument")
    generic_local = GENERIC_LOCAL.match(text, pos)
    if generic_local:
        return Binding(OUT_OF_SCOPE, end=generic_local.end(), reason="local_instrument")
    kind, name, end = _title_info(section, pos)
    if kind == "federal":
        variant = _variant(cat, name)
        if variant is not None:
            return Binding(MISSING, end=end, reason="catalogue_title_variant")
        return Binding(OUT_OF_SCOPE, end=end, reason="uncatalogued_instrument")
    if kind is not None:
        return Binding(OUT_OF_SCOPE, end=end, reason=kind + "_instrument")
    return None


def _cpeum_alias(section: _Section, start: int, end: int) -> Binding:
    text = section.text
    left, right = section.raw_span(start, end)
    raw_tail = section.raw[right:right + 80]
    tail = text[end:end + 80]
    if CPEUM_CONTINUATION.match(tail):
        extended = CPEUM_CONTINUATION.match(tail).end()
        return Binding("bound", ("R1",), section.cat.cpeum, end=end + extended)
    if LOCAL_MARKER.match(tail.lstrip(" ,")) or re.match(r"\s+(?:politica\s+)?(?:del|de\s+la|de)\s+(?:estado|ciudad|entidad|" + _STATES + r")\b", tail):
        return Binding(OUT_OF_SCOPE, end=end, reason="local_constitution")
    if FOREIGN_CONSTITUTION.match(tail):
        return Binding(OUT_OF_SCOPE, end=end, reason="foreign_or_historical_constitution")
    if PROPER_COMPLEMENT.match(raw_tail):
        return Binding(MISSING, end=end, reason="constitution_alias_with_complement")
    bare = text[start:end].split()[0] == "constitucion" and len(text[start:end].split()) == 1
    if bare and LOWER_COMPLEMENT.match(raw_tail):
        return Binding(MISSING, end=end, reason="constitution_alias_with_complement")
    return Binding("bound", ("R1",), section.cat.cpeum, end=end)


def _anaphor_kind(match: re.Match) -> str:
    noun = next(value for value in match.groupdict().values() if value)
    noun = noun.split()[0]
    return {"ley": "ley", "codigo": "codigo", "codificacion": "codigo", "constitucion": "constitucion",
            "reglamento": "reglamento", "estatuto": "estatuto"}.get(noun, "any")


def _compatible(section: _Section, instrument_id: str, kind: str) -> bool:
    if kind == "any":
        return True
    if kind == "constitucion":
        return instrument_id == section.cat.cpeum
    return section.cat.kinds.get(instrument_id) == kind


def _anaphora(section: _Section, pos: int, match: re.Match) -> Binding:
    end = match.end()
    if ANAPHOR_LOCAL_TAIL.match(section.text, end):
        return Binding(OUT_OF_SCOPE, end=end, reason="anaphora_to_local_instrument")
    kind = _anaphor_kind(match)
    earlier = [b for b in section.bindings if b[1] <= pos and _compatible(section, b[2], kind)]
    if not earlier:
        return Binding(MISSING, end=end, reason="anaphora_without_antecedent")
    antecedent = max(earlier, key=lambda b: b[1])
    window = {b[2] for b in earlier if b[1] >= pos - ANAPHORA_WINDOW}
    if len(window) > 1:
        return Binding(MISSING, end=end, reason="ambiguous_anaphora")
    for position, identity in section.law_like_between(antecedent[1], pos):
        if identity != antecedent[2]:
            if identity == "?" or _compatible(section, identity, kind):
                return Binding(MISSING, end=end, reason="anaphora_intervening_instrument")
    for position, kind_name in section.foreign:
        if antecedent[1] <= position < pos:
            return Binding(MISSING, end=end, reason="anaphora_intervening_instrument")
    return Binding("bound", ("R3",), antecedent[2], end=end)


def forward_binding(section: _Section, pos: int) -> Binding | None:
    """Binding directly after a folded expression end (coordinated, adjective or relation)."""
    text = section.text
    coordinated = COORDINATED.match(text, pos)
    if coordinated:
        inner = _target(section, coordinated.end())
        if inner is not None:
            inner.start, inner.coordinated = pos, True
            if inner.outcome == "bound":
                inner.rules = ("R7", *inner.rules)
            return inner
    adjective = CONSTITUTIONAL_ADJECTIVE.match(text, pos)
    if adjective:
        local = LOCAL_MARKER.match(text[adjective.end():adjective.end() + 60].lstrip(" ,"))
        if local:
            return Binding(OUT_OF_SCOPE, start=pos, end=adjective.end(), reason="local_constitution")
        return Binding("bound", ("R1",), section.cat.cpeum, start=pos, end=adjective.end())
    relation = REL_GAP.match(text, pos)
    if relation:
        inner = _target(section, relation.end())
        if inner is not None:
            inner.start = pos
            return inner
    return None


def reverse_binding(section: _Section, start: int) -> Binding | None:
    """R2: '<law>, en su(s) artículo(s) N' with the law at most REVERSE_GAP chars before."""
    window = section.text[max(0, start - 80):start]
    anchor = REVERSE_ANCHOR.search(window)
    if anchor is None:
        return None
    phrase = start - len(window) + anchor.start()
    candidates: list[tuple[int, int, Binding]] = []
    for mention in section.mentions:
        if mention["end"] <= phrase and phrase - mention["end"] <= REVERSE_GAP:
            candidates.append((mention["start"], mention["end"], _target(section, mention["start"], reverse=True)))
    lo = max(0, phrase - REVERSE_GAP - 60)
    for alias in CPEUM_ALIAS.finditer(section.text, lo, phrase):
        if phrase - alias.end() <= REVERSE_GAP and not any(m["start"] <= alias.start() < m["end"] for m in section.mentions):
            candidates.append((alias.start(), alias.end(), _cpeum_alias(section, alias.start(), alias.end())))
    left, right = section.raw_span(lo, phrase)
    for acronym in re.finditer(r"(?<![\w])([A-Z][A-Za-z]{1,7})(?!\w)", section.raw[left:right]):
        if acronym.group(1) in section.cat.abbreviations:
            a_start, a_end = section.folded_span(left + acronym.start(1), left + acronym.end(1))
            if phrase - a_end <= REVERSE_GAP:
                candidates.append((a_start, a_end, Binding("bound", ("R4",), section.cat.abbreviations[acronym.group(1)], end=a_end)))
    if not candidates:
        return None
    m_start, m_end, inner = max(candidates, key=lambda row: row[1])
    gap = section.text[m_end:phrase]
    if inner is None or inner.outcome != "bound" or re.search(r"[.;:()\[\]\"“”]", gap) or len(gap.split()) > 6:
        return None
    if section.law_like_between(m_end, phrase):
        return None
    return Binding("bound", ("R2", *[r for r in inner.rules if r != "R0"]), inner.instrument_id, start=m_start, end=m_end)


def _after_qualifier(text: str, cursor: int) -> tuple[int, str | None, int | None]:
    """v2's inner connector/qualifier loop, resumed after an extended qualifier tail."""
    while True:
        separator = v2.CONNECTOR.match(text, cursor)
        if separator is None:
            return cursor, None, None
        following = separator.end()
        if v2.QUALIFIER.match(text, following) or v2.ORDINAL_BEFORE_QUALIFIER.match(text, following):
            cursor, problem = v2._skip_qualifier(text, following)
            if problem:
                return cursor, problem, None
            cursor = _tails(text, cursor)[0]
            continue
        if v5.ARTICLE_NUMBER.match(text, following):
            return cursor, None, following
        return cursor, None, None


def _tails(text: str, cursor: int) -> tuple[int, bool]:
    """Consume R12 qualifier tails after a v2 qualifier parse stops."""
    used = False
    while True:
        if ORDINAL_END.search(text, max(0, cursor - 40), cursor):
            noun = NOUN_AFTER_ORDINAL.match(text, cursor)
            if noun:
                cursor, used = noun.end(), True
                continue
            compound = COMPOUND_ORDINAL.match(text, cursor)
            if compound and re.search(r"decim[oa]\s*$", text[max(0, cursor - 12):cursor]):
                cursor, used = compound.end(), True
                continue
        tail = QUALIFIER_TAIL.match(text, cursor)
        if tail and tail.end() > cursor:
            cursor, used = tail.end(), True
            continue
        bare = BARE_QUALIFIER_NOUN.match(text, cursor)
        if bare:
            end, problem = v2._skip_qualifier(text, bare.end())
            if not problem and end > bare.end():
                cursor, used = end, True
                continue
        return cursor, used


def parse_chain(text: str, pos: int) -> Chain:
    """Parse an article list with ranges, qualifier tails and chain separators."""
    groups: list[list[Entry]] = [[]]
    ends: list[int] = []
    rules: set[str] = set()
    cursor = pos
    while True:
        segment = v5.parse_expression(text, cursor)
        if not segment["entries"]:
            if not any(groups):
                return Chain(groups, [cursor], cursor, "unsupported_article_expression", rules)
            groups.pop()
            return Chain(groups, ends, ends[-1], None, rules)
        groups[-1].extend(Entry(e["article_label"], e["start"], e["end"]) for e in segment["entries"])
        cursor = segment["end"]
        if segment["reason"]:
            ends.append(cursor)
            return Chain(groups, ends, cursor, segment["reason"], rules)
        while True:
            span = v2.RANGE.match(text, cursor)
            number = v5.ARTICLE_NUMBER.match(text, span.end()) if span else None
            if number:
                tail = v5.parse_expression(text, number.start())
                last = groups[-1][-1]
                groups[-1][-1] = Entry(last.label, last.start, tail["entries"][0]["end"], tail["entries"][0]["article_label"])
                groups[-1].extend(Entry(e["article_label"], e["start"], e["end"]) for e in tail["entries"][1:])
                rules.add("R6")
                cursor = tail["end"]
                if tail["reason"]:
                    ends.append(cursor)
                    return Chain(groups, ends, cursor, tail["reason"], rules)
                continue
            cursor, used = _tails(text, cursor)
            if used:
                rules.add("R12")
                cursor, problem, following = _after_qualifier(text, cursor)
                if problem:
                    ends.append(cursor)
                    return Chain(groups, ends, cursor, problem, rules)
                if following is not None:
                    cursor = following
                    break
                continue
            disjunction = DISJUNCTION.match(text, cursor)
            if disjunction:
                rules.add("R12")
                cursor = disjunction.end()
                break
            separator = CHAIN_SEPARATOR.match(text, cursor)
            if separator:
                ends.append(cursor)
                groups.append([])
                cursor = separator.end()
                break
            ends.append(cursor)
            return Chain(groups, ends, cursor, None, rules)


def _number_start(section: _Section, start: int, kind: str) -> int:
    pattern = {"article_prefix": v2.ARTICLE_PREFIX, "other_numeric_prefix": v2.OTHER_NUMERIC_PREFIX,
               "numeric_continuation": v2.NUMERIC_CONTINUATION}.get(kind)
    match = pattern.match(section.text, start) if pattern else None
    return match.end() if match else start


def _members(cat: Catalogue, instrument_id: str, entry: Entry, plural: bool) -> tuple[list[str] | None, str | None]:
    """Article labels denoted by an entry; None with a reason when it cannot expand."""
    if entry.range_end is None:
        hyphen = re.fullmatch(r"(\d+)-(\d+)", entry.label)
        if plural and hyphen and int(hyphen[2]) > int(hyphen[1]):
            if (instrument_id, entry.label) in cat.index:
                return None, "ambiguous_numeric_hyphen_range"
            first, last = int(hyphen[1]), int(hyphen[2])
        else:
            return [entry.label], None
    else:
        if not (entry.label.isdigit() and entry.range_end.isdigit()):
            return None, "range_not_expanded"
        first, last = int(entry.label), int(entry.range_end)
    if last <= first:
        return None, "range_not_expanded"
    if last - first > MAX_RANGE_SPAN:
        return None, "range_too_wide"
    labels = [str(value) for value in range(first, last + 1)]
    if any((instrument_id, label) not in cat.index for label in labels):
        return None, "range_member_missing_from_catalogue"
    return labels, None


@dataclass
class Resolution:
    """One source's graded references; spans are section-relative raw offsets."""
    query_id: str
    positives: list[dict]
    unresolved: list[dict]
    version_flags: list[dict]
    sections: list[dict]

    @property
    def positive_ids(self) -> list[str]:
        return sorted({p["candidate_id"] for p in self.positives})

    @property
    def missing(self) -> list[dict]:
        return [u for u in self.unresolved if u["disposition"] == MISSING]

    @property
    def status(self) -> str | None:
        if not self.positive_ids:
            return None
        return "partial" if self.missing else "full"


class _SourceResolver:
    def __init__(self, source: Mapping, catalogue: Catalogue):
        self.source, self.cat = source, catalogue
        self.raw = source["raw_published_text"]
        materias = source.get("source_materias") or []
        if isinstance(materias, str):
            materias = re.findall(r"[^\[\]',\"]+", materias)
        self.comun = any(v2.fold(str(m))[0].strip() == "comun" for m in materias)
        self.positives: list[dict] = []
        self.unresolved: list[dict] = []
        self.label_bindings: dict[str, list[tuple[tuple[int, int], str]]] = defaultdict(list)
        self._deferred: list[dict] = []

    def resolve(self) -> Resolution:
        titles = v3._title_spans(self.source)
        version_flags, sections = [], []
        order = 0
        for section in self.source["sections"]:
            if section["field"] not in v2.LATER_FIELDS:
                continue
            start, end, text = int(section["start"]), int(section["end"]), section["text"]
            if self.raw[start:end] != text:
                raise ValueError("section does not address unchanged raw source")
            sections.append({"field": section["field"], "start": start, "end": end, "sha256": digest(text)})
            normalized, offsets = v2.fold(text)
            for match in VERSION_CUE.finditer(normalized):
                left = offsets[match.start()]
                right = offsets[match.end() - 1] + 1
                version_flags.append({"section": SECTION_NAMES[section["field"]], "span": [left, right],
                                      "surface_sha256": digest(text[left:right])})
            self._section(text, section["field"], start, titles, order)
            order += 1
        return Resolution(self.source["query_id"], self.positives, self.unresolved, version_flags, sections)

    def _add_positive(self, sec: _Section, order: int, candidate: dict, rules: Iterable[str], span: tuple[int, int],
                      witness: tuple[int, int], label: str, instrument_id: str) -> None:
        surface = sec.raw[span[0]:span[1]]
        self.positives.append({
            "candidate_id": candidate["candidate_id"], "instrument_id": instrument_id, "article_label": label,
            "rule_ids": sorted(set(rules), key=lambda r: int(r[1:])), "section": SECTION_NAMES[sec.field],
            "span": [span[0], span[1]], "surface_sha256": digest(surface), "witness": [witness[0], witness[1]],
            "source_span": [sec.base + span[0], sec.base + span[1]],
        })
        self.label_bindings[label].append(((order, span[0]), instrument_id))

    def _add_unresolved(self, sec: _Section, disposition: str, reason: str, span: tuple[int, int], kind: str,
                        labels: list[str]) -> None:
        surface = sec.raw[span[0]:span[1]]
        self.unresolved.append({"reason": reason, "disposition": disposition, "section": SECTION_NAMES[sec.field],
                                "span": [span[0], span[1]], "surface_sha256": digest(surface), "kind": kind,
                                "labels": labels, "source_span": [sec.base + span[0], sec.base + span[1]]})

    def _section(self, text: str, field: str, base: int, titles: list[dict], order: int) -> None:
        result = v5._extract_section(text, self.cat.instruments, self.cat.index, field, base)
        sec = _Section(text, field, base, self.cat, result["explicit_instrument_mentions"])
        keep = lambda span: not v3._within_title(span, titles)
        for occurrence in result["occurrences"]:
            span = occurrence["article_span"]
            if not keep(span):
                rel = (span["start"] - base, span["end"] - base)
                self._add_unresolved(sec, IGNORABLE, "article_numbers_inside_formal_statute_title", rel,
                                     occurrence["declaration_type"], occurrence["parsed_article_labels"])
        for reference in result["references"]:
            if not keep(reference["literal_article_span"]):
                continue
            literal, witness = reference["literal_article_span"], reference["witness_span"]
            candidate = {"candidate_id": reference["candidate_id"]}
            self._add_positive(sec, order, candidate, ("R0",), (literal["start"] - base, literal["end"] - base),
                               (witness["start"] - base, witness["end"] - base), reference["article_label"],
                               reference["instrument_id"])
        for occurrence in result["occurrences"]:
            if occurrence.get("instrument_id") and keep(occurrence["article_span"]):
                law = occurrence["instrument_span"]
                start, end = sec.folded_span(law["start"] - base, law["end"] - base)
                mention = sec.mention_at.get(start)
                if (mention and len(mention["instrument_ids"]) == 1 and not mention["unsafe_title_extension"]
                        and occurrence.get("reason") != "quoted_precedent_title"):
                    sec.bindings.append((start, end, occurrence["instrument_id"], "R0"))
        pending = []
        for occurrence in result["unresolved"]:
            span = occurrence["article_span"]
            if not keep(span):
                continue
            fs, fe = sec.folded_span(span["start"] - base, span["end"] - base)
            pending.append({"occurrence": occurrence, "fs": fs, "fe": fe})
        pending.sort(key=lambda row: (row["fs"], row["fe"]))
        consumed: list[tuple[int, int]] = []
        deferred = []
        for row in pending:
            if any(left <= row["fs"] < right for left, right in consumed):
                continue
            outcome = self._revisit(sec, order, row)
            consumed.append((row["fs"], outcome))
        deferred.extend(self._deferred)
        self._deferred = []
        self._materia(sec, order, deferred)

    def _revisit(self, sec: _Section, order: int, row: dict) -> int:
        """Apply R1-R12 to one v5-unresolved expression; returns the folded extent consumed."""
        occurrence, fs = row["occurrence"], row["fs"]
        kind, reason = occurrence["declaration_type"], occurrence["reason"]
        labels = occurrence["parsed_article_labels"]
        expression_span = sec.raw_span(fs, row["fe"])
        if reason == "quoted_precedent_title":
            self._add_unresolved(sec, IGNORABLE, reason, expression_span, kind, labels)
            return row["fe"]
        if any(m["start"] <= fs < m["end"] for m in sec.mentions) or IN_TITLE.search(sec.text, max(0, fs - 160), fs):
            self._add_unresolved(sec, IGNORABLE, "article_numbers_inside_statute_title", expression_span, kind, labels)
            return row["fe"]
        start = _number_start(sec, fs, kind)
        chain = parse_chain(sec.text, start)
        if chain.problem == "unsupported_article_expression":
            return self._no_number(sec, row, start)
        binding = forward_binding(sec, chain.end) if chain.problem is None and len(chain.groups) > 1 else None
        groups, rules = chain.groups, set(chain.rules)
        if binding is None or not binding.coordinated:
            groups = chain.groups[:1]
            end = chain.ends[0]
            problem = chain.problem if len(chain.groups) == 1 else None
            if len(chain.groups) > 1:
                rules = self._group_rules(sec.text, start, end, chain.rules)
            binding = forward_binding(sec, end)
            if problem is not None:
                return self._problem(sec, row, problem, binding, end)
        else:
            end = chain.end
        entries = [entry for group in groups for entry in group]
        extent = binding.end if binding is not None and binding.end else end
        if binding is not None and binding.outcome == "bound" and _quoted(sec, fs, binding.end):
            self._add_unresolved(sec, IGNORABLE, "quoted_precedent_title", sec.raw_span(fs, binding.end), kind, labels)
            return extent
        designator = kind in {"other_numeric_prefix"}
        continuation = kind == "numeric_continuation" or (kind == "bare_bound_numeric_expression"
                                                          and CONTINUATION_BEFORE.search(sec.text, max(0, fs - 60), fs))
        if binding is None or binding.outcome not in {"bound", "defer_r9"}:
            if binding is None and kind in {"article_prefix", "other_numeric_prefix"}:
                reverse = reverse_binding(sec, fs)
                if reverse is not None:
                    binding = reverse
                else:
                    anaphoric = self._article_anaphora(sec, order, fs, end, entries)
                    if anaphoric is not None:
                        return anaphoric
            if binding is None or binding.outcome not in {"bound", "defer_r9"}:
                return self._unbound(sec, row, binding, start, end, entries)
        if binding.outcome == "defer_r9":
            self._deferred.append({"row": row, "entries": entries, "rules": rules, "binding": binding,
                                   "end": end, "kind": kind, "continuation": continuation, "designator": designator})
            return extent
        if continuation and not set(binding.rules) & {"R0", "R1", "R4", "R10", "R7"}:
            self._add_unresolved(sec, MISSING, "unanchored_numeric_continuation", sec.raw_span(fs, end), kind, labels)
            return extent
        if kind == "bare_bound_numeric_expression" and not continuation:
            self._add_unresolved(sec, MISSING, "non_article_designator_unresolved", sec.raw_span(fs, end), kind, labels)
            return extent
        # Anaphoric (R3) bindings are not antecedents for later anaphora.
        if (kind != "numeric_continuation" or binding.instrument_id) and "R3" not in binding.rules:
            sec.bindings.append((binding.start, binding.end, binding.instrument_id, binding.rules[0]))
        extra = set(rules) | set(binding.rules)
        if continuation:
            extra.add("R5")
        if designator:
            extra.add("R11")
        self._map(sec, order, entries, binding.instrument_id, extra, fs, extent, kind, plural=self._plural(sec, fs, kind))
        return extent

    def _group_rules(self, text: str, start: int, end: int, rules: set[str]) -> set[str]:
        """Rules actually used inside the first group of a multi-group chain."""
        chain = parse_chain(text[:end], start)
        return set(chain.rules)

    @staticmethod
    def _plural(sec: _Section, fs: int, kind: str) -> bool:
        prefix = v2.ARTICLE_PREFIX.match(sec.text, fs)
        if kind == "numeric_continuation":
            return True
        return bool(prefix and prefix.group().strip() in {"articulos", "arts."}) or bool(
            re.match(r"(?:diversos|numerales|preceptos|dispositivos)\b", sec.text[fs:fs + 12]))

    def _map(self, sec: _Section, order: int, entries: list[Entry], instrument_id: str, rules: set[str],
             fs: int, extent: int, kind: str, *, plural: bool) -> None:
        witness = sec.raw_span(fs, extent)
        for entry in entries:
            members, problem = _members(self.cat, instrument_id, entry, plural)
            span = sec.raw_span(entry.start, entry.end)
            if members is None:
                self._add_unresolved(sec, MISSING, problem, span, kind, [entry.label])
                continue
            entry_rules = set(rules)
            if len(members) > 1:
                entry_rules.add("R6")
            for label in members:
                candidates = self.cat.index.get((instrument_id, label))
                if not candidates:
                    self._add_unresolved(sec, MISSING, "exact_article_missing_from_catalogue", span, kind, [label])
                    continue
                if len(candidates) != 1:
                    self._add_unresolved(sec, MISSING, "duplicate_exact_article_identity", span, kind, [label])
                    continue
                self._add_positive(sec, order, candidates[0], entry_rules, span, witness, label, instrument_id)

    def _problem(self, sec: _Section, row: dict, problem: str, binding: Binding | None, end: int) -> int:
        occurrence, fs = row["occurrence"], row["fs"]
        kind, labels = occurrence["declaration_type"], occurrence["parsed_article_labels"]
        span = sec.raw_span(fs, max(end, row["fe"]))
        if binding is not None and binding.outcome == OUT_OF_SCOPE:
            self._add_unresolved(sec, OUT_OF_SCOPE, binding.reason, span, kind, labels)
            sec.foreign.append((binding.start, binding.reason))
            return binding.end
        if problem == "decimal_article_or_subdivision_unresolved":
            before = sec.raw[max(0, span[0] - 12):span[0]]
            if re.search(r"[$’']\s*\d*\s*$", before) or re.search(r"\$", sec.raw[max(0, span[0] - 25):span[0]]):
                self._add_unresolved(sec, IGNORABLE, "amount_or_quantity", span, kind, labels)
                return max(end, row["fe"])
        self._add_unresolved(sec, MISSING, problem, span, kind, labels)
        return max(end, row["fe"])

    def _no_number(self, sec: _Section, row: dict, start: int) -> int:
        occurrence, fs = row["occurrence"], row["fs"]
        kind = occurrence["declaration_type"]
        after = sec.text[start:start + 80]
        span = sec.raw_span(fs, max(row["fe"], start))
        if TRANSITORY.match(after) or re.match(r"(?:primero|segundo|tercero|cuarto|quinto|sexto|septimo|octavo|noveno|decimo|unico)(?:\s+\w+)?\s+(?:y|e|,)\s+[\w\s,]{0,40}transitori", after):
            self._add_unresolved(sec, OUT_OF_SCOPE, "transitory_article", span, kind, [])
        elif re.match(r"unico\b", after):
            self._add_unresolved(sec, OUT_OF_SCOPE, "single_article_decree", span, kind, [])
        elif WORD_ORDINAL.match(after):
            binding = None
            word = WORD_ORDINAL.match(after)
            binding = forward_binding(sec, start + word.end())
            if binding is not None and binding.outcome == OUT_OF_SCOPE:
                self._add_unresolved(sec, OUT_OF_SCOPE, binding.reason, span, kind, [])
            else:
                self._add_unresolved(sec, MISSING, "ordinal_word_article", span, kind, [])
        else:
            self._add_unresolved(sec, IGNORABLE, "article_noun_without_number", span, kind, [])
        return max(row["fe"], start)

    def _article_anaphora(self, sec: _Section, order: int, fs: int, end: int, entries: list[Entry]) -> int | None:
        """R8: 'el citado artículo N' / 'artículo N citado' -> earlier unique binding of N."""
        before = ARTICLE_ANAPHOR_BEFORE.search(sec.text, max(0, fs - 30), fs)
        after = ARTICLE_ANAPHOR_AFTER.match(sec.text, end)
        if before is None and after is None:
            return None
        here = (order, sec.raw_span(fs, fs + 1)[0])
        extent = after.end() if after else end
        for entry in entries:
            span = sec.raw_span(entry.start, entry.end)
            if entry.range_end is not None:
                self._add_unresolved(sec, MISSING, "article_anaphora_range", span, "article_anaphora", [entry.label])
                continue
            identities = {identity for position, identity in self.label_bindings.get(entry.label, []) if position < here}
            if len(identities) != 1:
                reason = "ambiguous_article_anaphora" if identities else "article_anaphora_without_antecedent"
                self._add_unresolved(sec, MISSING, reason, span, "article_anaphora", [entry.label])
                continue
            identity = next(iter(identities))
            candidates = self.cat.index.get((identity, entry.label), [])
            if len(candidates) != 1:
                self._add_unresolved(sec, MISSING, "exact_article_missing_from_catalogue", span, "article_anaphora", [entry.label])
                continue
            self._add_positive(sec, order, candidates[0], ("R8",), span, sec.raw_span(fs, extent), entry.label, identity)
        return extent

    def _unbound(self, sec: _Section, row: dict, binding: Binding | None, start: int, end: int,
                 entries: list[Entry]) -> int:
        occurrence, fs = row["occurrence"], row["fs"]
        kind, labels = occurrence["declaration_type"], occurrence["parsed_article_labels"]
        span = sec.raw_span(fs, max(end, row["fe"]))
        if binding is not None:
            if binding.outcome == OUT_OF_SCOPE:
                sec.foreign.append((binding.start, binding.reason))
            disposition = {OUT_OF_SCOPE: OUT_OF_SCOPE, IGNORABLE: IGNORABLE}.get(binding.outcome, MISSING)
            self._add_unresolved(sec, disposition, binding.reason, span, kind, labels)
            return max(binding.end, end)
        if kind in {"numeric_continuation", "bare_bound_numeric_expression"}:
            ignorable = self._not_citation(sec, fs, start, end)
            if ignorable:
                self._add_unresolved(sec, IGNORABLE, ignorable, span, kind, labels)
                return end
            reason = "unanchored_numeric_continuation" if kind == "numeric_continuation" else "non_article_designator_unresolved"
            self._add_unresolved(sec, MISSING, reason, span, kind, labels)
            return end
        ignorable = self._not_citation(sec, fs, start, end)
        if ignorable in {"docket_number", "amount_or_quantity", "date_or_year"}:
            self._add_unresolved(sec, IGNORABLE, ignorable, span, kind, labels)
            return end
        reason = occurrence["reason"]
        if reason in {"range_not_expanded", "ambiguous_numeric_hyphen_range", "unanchored_numeric_continuation",
                      "non_article_designator_unresolved"}:
            reason = "instrument_not_explicitly_bound"
        self._add_unresolved(sec, MISSING, reason, span, kind, labels)
        return end

    @staticmethod
    def _not_citation(sec: _Section, fs: int, start: int, end: int) -> str | None:
        """Enumeration markers, docket numbers, amounts, quantities and dates."""
        after = sec.text[end:end + 60]
        raw_after = sec.raw[sec.raw_span(end, end + 1)[0]:][:60] if end < len(sec.text) else ""
        before = sec.text[max(0, fs - 40):start]
        if DOCKET_AFTER.match(after):
            return "docket_number"
        if UNIT.match(after) or re.search(r"[$]\s*$", sec.raw[max(0, sec.raw_span(start, start + 1)[0] - 3):sec.raw_span(start, start + 1)[0]]):
            return "amount_or_quantity"
        numbers = re.findall(r"\d+", sec.text[start:end])
        if numbers and all(1900 <= int(n) <= 2100 for n in numbers) and MONTH.search(before):
            return "date_or_year"
        if ENUMERATION_AFTER.match(raw_after) or re.match(r"\s*\)", after):
            return "enumeration_marker"
        return None

    def _materia(self, sec: _Section, order: int, deferred: list[dict]) -> None:
        """R9 after the whole section is bound: Común tesis and no other law bound."""
        others = {b[2] for b in sec.bindings} - {self.cat.amparo, self.cat.cpeum}
        for item in deferred:
            row, binding = item["row"], item["binding"]
            occurrence, fs = row["occurrence"], row["fs"]
            kind, labels = occurrence["declaration_type"], occurrence["parsed_article_labels"]
            if self.comun and not others and not sec.foreign:
                rules = set(item["rules"]) | {"R9"}
                if binding.coordinated:
                    rules.add("R7")
                if item["continuation"]:
                    rules.add("R5")
                if item["designator"]:
                    rules.add("R11")
                self._map(sec, order, item["entries"], self.cat.amparo, rules, fs, binding.end, kind,
                          plural=self._plural(sec, fs, kind))
            else:
                reason = "ley_de_la_materia_not_amparo" if not self.comun else "ley_de_la_materia_other_law_bound"
                self._add_unresolved(sec, MISSING, reason, sec.raw_span(fs, item["end"]), kind, labels)


def resolve_source(source: Mapping, catalogue: Catalogue) -> Resolution:
    """Resolve one source; its raw text must match its content hash."""
    raw = source["raw_published_text"]
    if digest(raw) != source["source_content_sha256"]:
        raise ValueError("raw source content binding mismatch")
    return _SourceResolver(source, catalogue).resolve()


def summarize(resolutions: Iterable[Resolution]) -> dict[str, Any]:
    """Counts by rule, disposition and reason over resolutions."""
    rules, reasons, statuses = Counter(), Counter(), Counter()
    combos = Counter()
    for resolution in resolutions:
        statuses[resolution.status or "excluded"] += 1
        for positive in resolution.positives:
            combos["+".join(positive["rule_ids"])] += 1
            for rule in positive["rule_ids"]:
                rules[rule] += 1
        for item in resolution.unresolved:
            reasons[(item["disposition"], item["reason"])] += 1
    return {"positive_occurrences_by_rule": dict(sorted(rules.items(), key=lambda kv: int(kv[0][1:]))),
            "positive_occurrences_by_rule_combination": dict(combos.most_common()),
            "unresolved_by_disposition_reason": {f"{d}/{r}": n for (d, r), n in sorted(reasons.items())},
            "queries_by_status": dict(statuses)}
