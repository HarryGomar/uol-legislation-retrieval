"""Deterministic parsing of source proceeding identities.

The parser extracts witnessed docket strings. It never invents missing docket
digits or merges records across deciding courts.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata


_SPACE = re.compile(r"\s+")
_TYPE = r"(?:amparos?\s+(?:directos?(?:\s+en\s+revision)?|en\s+revision)|quejas?|contradiccion(?:es)?\s+de\s+(?:criterios|tesis)|conflictos?\s+(?:competenciales?|de\s+trabajo)|incidentes?\s+[^.;:\r\n]{0,100}|recursos?\s+[^.;:\r\n]{0,130}|revision(?:es)?\s+[^.;:\r\n]{0,100}|acciones?\s+de\s+inconstitucionalidad|controversias?\s+constitucionales?|impedimentos?|solicitudes?\s+[^.;:\r\n]{0,100}|denuncias?\s+[^.;:\r\n]{0,100}|aclaracion\s+de\s+sentencia)"
_DOCKET = re.compile(
    rf"(?i)(?<![\w/])(?P<type>{_TYPE})\s+(?P<number>\d{{1,6}})\s*/\s*(?P<year>\d{{2,4}})(?!\d)"
)


def fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or "").casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    return _SPACE.sub(" ", text).strip()


@dataclass(frozen=True)
class ProceedingMention:
    source: str
    proceeding_type: str
    docket_number: str
    docket_year: str
    court: str
    qualified_id: str
    status: str = "parsed_court_qualified"


def parse_precedents(text: object, court: object) -> tuple[ProceedingMention, ...]:
    source = str(text or "")
    court_norm = fold(court)
    output: list[ProceedingMention] = []
    seen: set[tuple[str, str, str, int]] = set()
    for match in _DOCKET.finditer(fold(source)):
        proceeding_type = _SPACE.sub(" ", match.group("type")).strip()
        number = match.group("number")
        year = match.group("year")
        key = (proceeding_type, number, year, match.start())
        if key in seen:
            continue
        seen.add(key)
        payload = f"{court_norm}\0{proceeding_type}\0{number}\0{year}"
        qualified_id = "proceeding:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        output.append(ProceedingMention(source, proceeding_type, number, year, court_norm, qualified_id))
    return tuple(output)
