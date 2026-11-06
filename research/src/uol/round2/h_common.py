"""Workstream H (experiment L3, lay-register stress test): paths, population, frozen gate definitions.

Everything here was fixed **before** any L3 text was generated (frozen 2026-10-04): the arm list, the L_b subset seed,
the banned legal-term lexicon (LAY_V1's own terms plus ``EXTRA_TERMS``), the number/date gate, the length-ratio band,
the scrub rule and the new-content-word caps. Outputs go to ``experiments/runs/round2_20261004/H`` only.
DESCRIPTIVE: frozen systems on lockbox rows; no method is selected.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uol.facts_to_law.spec import ROOT

OUT = Path(os.environ.get("UOL_H_OUT") or ROOT / "experiments/runs/round2_20261004/H")   # override only for plumbing tests
LOG = OUT / "run.log"
REPORT = ROOT / "research/reports/round2_20261004/H_L3_lay_stress.md"
TEXTS = OUT / "texts.jsonl"                 # one row per (arm, query_id): final query text + gate record
SEED = 20261004
N_LB = 150
ARMS = ("O", "Pp", "L", "L_b", "R1", "R2")
V11_SOURCES = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"

# ----------------------------------------------------------------------------------------------- frozen gate settings
LEN_RATIO = (0.4, 2.5)
# v2 caps (5 Oct, before any lockbox generation): the dev-40 calibration (calibration_dev40.json) 95th percentile
# rounded up to .05; v1 caps (.50 / .35) would have flagged most texts of every arm (lay and paraphrase replace words).
NEW_CONTENT_CAP = {"L": 0.90, "L_b": 0.90, "Pp": 0.65}
REGEN_TEMPERATURE = 0.3
MAX_TOKENS = 2000

LAY_V1_TERMS = ["amparo", "quejoso", "tercero interesado", "acto reclamado", "litis", "agravio", "juicio",
                "autoridad responsable", "recurso", "sentencia", "resolucion", "demanda", "procedente", "improcedente"]
# frozen before generation: procedural roles, court/organ names, norm vocabulary and doctrinal terms
EXTRA_TERMS = ["quejosa", "recurrente", "promovente", "demandado", "demandada", "demandante", "actora", "tercera interesada",
               "juzgado", "juzgador", "tribunal colegiado", "tribunal unitario", "tribunal de alzada", "suprema corte",
               "primera sala", "segunda sala", "pleno", "magistrado", "magistrada", "ministro ponente", "ministerio publico",
               "jurisprudencia", "jurisprudencial", "tesis", "precedente", "precepto", "articulo", "fraccion", "inciso",
               "codigo", "reglamento", "constitucion", "constitucional", "inconstitucional", "inconstitucionalidad",
               "ley federal", "ley general", "ley organica", "norma general", "ejecutoria", "sobreseimiento", "sobreseer",
               "sobreseyo", "casacion", "apelacion", "conceptos de violacion", "concepto de violacion", "garantia de audiencia",
               "debido proceso", "fundamentacion", "motivacion", "litisconsorcio", "emplazamiento", "laudo",
               "contencioso", "nulidad", "imputado", "imputada", "inculpado", "inculpada", "procesado", "procesada",
               "sentenciado", "sentenciada", "jurisdiccion", "jurisdiccional", "competencia", "incompetencia",
               "proveido", "auto de", "suplencia", "interes juridico", "interes legitimo", "causal", "impugnacion",
               "impugno", "impugnar", "impugnada", "impugnado", "reclamado", "reclamada", "acto de autoridad",
               "autoridad ordenadora", "autoridad ejecutora", "via ordinaria", "via sumaria", "via ejecutiva",
               "carga de la prueba", "valor probatorio", "ad quem", "a quo", "litigio", "agravios", "recursos"]


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _term_re(terms: list[str]) -> re.Pattern:
    parts = []
    for t in sorted({_fold(x) for x in terms}, key=len, reverse=True):
        body = r"\s+".join(re.escape(w) for w in t.split())
        # plural / gender endings only for single words that are not already plural
        parts.append(body + (r"(?:es|s)?" if " " not in t else ""))
    return re.compile(r"\b(?:" + "|".join(parts) + r")\b")


LAY_RE = _term_re(LAY_V1_TERMS)
EXTRA_RE = _term_re(EXTRA_TERMS)


def banned_hits(text: str) -> dict[str, int]:
    f = _fold(text)
    return {"lay_v1": len(LAY_RE.findall(f)), "extra": len(EXTRA_RE.findall(f))}


MONTHS = {m: i + 1 for i, m in enumerate(["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                                          "septiembre", "octubre", "noviembre", "diciembre"])}
MONTHS["setiembre"] = 9
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")


def numbers(text: str) -> set[str]:
    """Normalized numeric tokens: separators dropped, zero decimals dropped, leading zeros dropped; month names add
    their month number (so 15/03/2019 and '15 de marzo de 2019' agree)."""
    out = set()
    for m in _NUM_RE.findall(text):
        s = re.sub(r"[.,]0{1,2}$", "", m)
        s = re.sub(r"[.,]", "", s).lstrip("0") or "0"
        out.add(s)
    f = _fold(text)
    for name, k in MONTHS.items():
        if re.search(r"\b" + name + r"\b", f):
            out.add(str(k))
    return out


STOP = set(("para porque pero como cuando donde desde hasta entre sobre tras contra segun durante mediante ante bajo "
            "este esta estos estas ese esa esos esas aquel aquella aquellos aquellas esto eso aquello cual cuales quien quienes "
            "cuyo cuya cuyos cuyas todo toda todos todas otro otra otros otras mismo misma mismos mismas tanto tanta tantos "
            "tantas cada algun alguna algunos algunas ningun ninguna mucho mucha muchos muchas poco poca pocos pocas "
            "tambien entonces luego despues antes ahora todavia siempre nunca solo aunque sino pues bueno asi tambien "
            "nosotros nosotras ellos ellas usted ustedes mismo esta estaba estaban estuvo estuve estoy estamos "
            "habia habian hubo haber habiendo tenia tenian tengo tiene tienen tuvo tuve hace hacer hizo hice "
            "fueron fuera fueron siendo sido seria serian puede pueden podia podria pudo pude debe deben debia "
            "dijo dijeron decir dice dicen sean sido sera seran ella este muy mas menos nada algo alguien nadie "
            "aqui alli ahi alla donde porque cuanto cuanta").split())


def stems(text: str) -> list[str]:
    """Content-word stems: folded alphabetic tokens of length >= 4 not in STOP, cut to their first 5 letters."""
    return [w[:5] for w in re.findall(r"[a-zñ]+", _fold(text)) if len(w) >= 4 and w not in STOP]


def new_content_share(gen: str, orig: str) -> float:
    g = stems(gen)
    if not g:
        return 1.0
    o = set(stems(orig))
    return float(np.mean([s not in o for s in g]))


# ----------------------------------------------------------------------------------------------- helpers

_LABEL_RE = re.compile(r"^\s*(?:\*\*)?\s*(?:HECHOS|Hechos|RELATO|Relato|PAR[AÁ]FRASIS|Par[aá]frasis|CUESTIONES|Cuestiones)\s*:?\s*(?:\*\*)?\s*:?[ \t]*\n")


def strip_label(text: str) -> str:
    """Drop a leading echoed section label ('HECHOS:', '**HECHOS:**', 'Relato:') on its own line (deterministic)."""
    return _LABEL_RE.sub("", text, count=1).strip()


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc()}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


def population() -> dict[str, Any]:
    """Lockbox rows of P (P_spec is a subset) in v11 row order, and the seeded L_b subset (150 of P)."""
    from uol.embedding_push import common
    lock = common.lockbox_rows()
    p = common.population_rows("P", lock)
    ps = common.population_rows("P_spec", lock)
    rng = np.random.default_rng(SEED)
    lb = np.sort(rng.choice(p, size=N_LB, replace=False))
    return {"P": p, "P_spec": ps, "L_b": lb}


def gate_settings() -> dict:
    return {"len_ratio": LEN_RATIO, "new_content_cap": NEW_CONTENT_CAP, "regen_temperature": REGEN_TEMPERATURE,
            "max_tokens": MAX_TOKENS, "label_re": _LABEL_RE.pattern, "lay_v1_terms": LAY_V1_TERMS, "extra_terms": EXTRA_TERMS,
            "lay_re_sha256": sha256_text(LAY_RE.pattern), "extra_re_sha256": sha256_text(EXTRA_RE.pattern),
            "stop_sha256": sha256_text(" ".join(sorted(STOP)))}
