"""Source export for facts-to-law v2: every structured SJF tesis except v1 sealed.

A structured tesis is a published text whose headings split exactly into Hechos,
Criterio juridico and Justificacion (``benchmark_validation.common.sections``).
C01 v1 ``test_sealed`` records are identified from byte-level fields of the
public v1 query file and skipped before their content line is decoded; no
custody file is read. Rows keep the field names of the 2,339-record train export
(research/state/benchmark_validation_20260926/safe_training_sources.jsonl) so
the release builder can consume both, and the train records must reproduce it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from uol.retrieval_benchmarks.benchmark_validation.common import sections
from uol.retrieval_benchmarks.c01_statutes.precedents import parse_precedents

from .spec import ROOT, sha256_file

SOURCE_HOME = Path(r"C:\uol_legacy_home")
CONTENT = SOURCE_HOME / "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_full_v1/content.jsonl"
V1_QUERIES = SOURCE_HOME / "data/releases/uol_c01_statutes_20260918_v1_candidate/method/queries.jsonl"
TRAIN_EXPORT = ROOT / "research/state/benchmark_validation_20260926/safe_training_sources.jsonl"
OUT = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1"

STRUCTURE = ["hechos", "criterio_juridico", "justificacion"]
FIELD_RE = {key: re.compile(rb'"' + key.encode() + rb'"\s*:\s*"([^"\\]*)"') for key in ("query_id", "record_id", "component_id", "split")}
ID_RE = re.compile(rb'"(?:requested_id|response_id|idTesis)"\s*:\s*"?([0-9]+)')
HECHOS_RE = re.compile(rb"(?i)hechos")
MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
          "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
PUBLISHED_RE = re.compile(r"Esta tesis se (?P<kind>publicó|republicó) el \w+ (?P<day>\d{1,2}) de (?P<month>[a-záéíóú]+) de (?P<year>\d{4})", re.IGNORECASE)
V1_SPLITS = {"train": "v1_train", "development": "v1_development"}


def read_v1_membership(path: Path = V1_QUERIES) -> tuple[dict[str, tuple[str, str]], set[bytes]]:
    """Return record_id -> (v1 split, v1 component) for train/development and the sealed id set.

    Only four short fields are extracted per line with byte regexes; sealed rows
    are never JSON-decoded.
    """
    if "custody" in Path(path).parts:
        raise ValueError("custody paths are not inputs")
    membership: dict[str, tuple[str, str]] = {}
    sealed: set[bytes] = set()
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            found = {key: rx.findall(line) for key, rx in FIELD_RE.items()}
            if any(len(values) != 1 for values in found.values()):
                raise ValueError("v1 query line lacks exactly one of query_id/record_id/component_id/split")
            record, split = found["record_id"][0], found["split"][0].decode()
            if split == "test_sealed":
                sealed.add(record)
            elif split in V1_SPLITS:
                membership[record.decode()] = (V1_SPLITS[split], found["component_id"][0].decode())
            else:
                raise ValueError(f"unknown v1 split {split!r}")
    return membership, sealed


def publication(nota: object) -> tuple[str | None, str | None]:
    match = PUBLISHED_RE.search(str(nota or ""))
    if not match:
        return None, None
    month = MONTHS.get(match.group("month").casefold())
    if month is None:
        return None, None
    date = f"{int(match.group('year')):04d}-{month:02d}-{int(match.group('day')):02d}"
    return date, "republished" if match.group("kind").casefold().startswith("rep") else "published"


def proceeding_ids(precedentes: object, organo: object, instancia: object) -> list[str]:
    court = organo or instancia or ""  # the C01 v1 rule (c01_statutes/build.py)
    try:
        return sorted({mention.qualified_id for mention in parse_precedents(precedentes, court)})
    except Exception:  # unparsable docket text yields no proceeding link, not a failure
        return []


def scan(content: Path, sealed: set[bytes], membership: dict[str, tuple[str, str]], content_sha256: str) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    stats = Counter()
    nonstrict: list[str] = []
    with Path(content).open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            stats["lines"] += 1
            ids = set(ID_RE.findall(line))
            if ids & sealed:
                stats["sealed_skipped_undecoded"] += 1
                continue
            if not HECHOS_RE.search(line):
                continue
            row = json.loads(line)
            published = row.get("published") or {}
            text = str(published.get("texto") or "")
            record_id = str(published.get("idTesis") or row.get("requested_id"))
            if record_id.encode() not in ids:
                raise ValueError(f"byte/decoded id disagreement for {record_id}")
            spans = sections(text)
            if [s["field"] for s in spans] != STRUCTURE:
                if text.lstrip().startswith("Hechos"):
                    nonstrict.append(record_id)
                continue
            v1_split, v1_component = membership.get(record_id, ("not_in_v1", None))
            date, kind = publication(published.get("notaPublica"))
            rows.append({
                "query_id": f"sjf:tesis:{record_id}:hechos",
                "record_id": record_id,
                "component_id": v1_component or f"sjf-record:{record_id}",
                "split": v1_split,
                "v1_split": v1_split if v1_split != "not_in_v1" else None,
                "v1_component_id": v1_component,
                "raw_hechos": spans[0]["text"],
                "raw_published_text": text,
                "sections": spans,
                "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "source_record_sha256": hashlib.sha256(line).hexdigest(),
                "source_file_sha256": content_sha256,
                "source_url": row.get("url"),
                "source_year": published.get("anio"),
                "source_month": published.get("mes"),
                "source_rubro": published.get("rubro"),
                "source_materias": published.get("materias"),
                "source_tipo_tesis": published.get("tipoTesis"),
                "source_precedentes": published.get("precedentes"),
                "source_published_id": published.get("idTesis"),
                "source_epoca": published.get("epoca"),
                "source_instancia": published.get("instancia"),
                "source_organo": published.get("organoJuris"),
                "proceeding_ids": proceeding_ids(published.get("precedentes"), published.get("organoJuris"), published.get("instancia")),
                "publication_date": date,
                "publication_kind": kind,
            })
            stats["structured"] += 1
    rows.sort(key=lambda r: int(r["record_id"]))
    if len({r["record_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate structured record ids")
    return rows, {"stats": dict(stats), "nonstrict_hechos_records": sorted(nonstrict, key=int)}


def check_train_equality(rows: list[dict], export: Path = TRAIN_EXPORT) -> dict:
    by_record = {r["record_id"]: r for r in rows}
    mismatches, missing, checked = [], [], 0
    with Path(export).open(encoding="utf-8") as stream:
        for line in stream:
            old = json.loads(line)
            new = by_record.get(str(old["record_id"]))
            if new is None:
                missing.append(old["record_id"])
                continue
            checked += 1
            for key in ("raw_published_text", "raw_hechos", "sections", "source_content_sha256", "source_record_sha256"):
                if new[key] != old[key]:
                    mismatches.append({"record_id": old["record_id"], "field": key})
            if new["split"] != "v1_train":
                mismatches.append({"record_id": old["record_id"], "field": "split"})
    if missing or mismatches:
        raise ValueError(f"train export not reproduced: missing={missing[:10]} mismatches={mismatches[:10]}")
    return {"train_records_checked": checked, "identical_fields": ["raw_published_text", "raw_hechos", "sections", "source_content_sha256", "source_record_sha256"]}


def build(out: Path = OUT) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; preserve earlier exports")
    content_sha = sha256_file(CONTENT)
    membership, sealed = read_v1_membership(V1_QUERIES)
    rows, scan_info = scan(CONTENT, sealed, membership, content_sha)
    equality = check_train_equality(rows)
    out.mkdir(parents=True)
    target = out / "sources.jsonl"
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    sealed_sorted = sorted(value.decode() for value in sealed)
    exclusions = {
        "v1_test_sealed_records": len(sealed),
        "v1_test_sealed_ids_sha256": hashlib.sha256("\n".join(sealed_sorted).encode()).hexdigest(),
        "sealed_lines_skipped_before_decoding": scan_info["stats"].get("sealed_skipped_undecoded", 0),
        "nonstrict_hechos_records": scan_info["nonstrict_hechos_records"],
        "rule": "v1 test_sealed records skipped by byte-level id match before JSON decoding; records whose headings are not exactly Hechos/Criterio juridico/Justificacion excluded",
    }
    (out / "exclusions.json").write_text(json.dumps(exclusions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    count = lambda key: dict(sorted(Counter(str(r[key]) for r in rows).items()))
    manifest = {
        "export_id": "sjf_structured_sources_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"content": {"path": str(CONTENT), "sha256": content_sha},
                   "v1_queries": {"path": str(V1_QUERIES), "sha256": sha256_file(V1_QUERIES)},
                   "train_export": {"path": str(TRAIN_EXPORT), "sha256": sha256_file(TRAIN_EXPORT)}},
        "code": {"sources.py": sha256_file(Path(__file__)),
                 "benchmark_validation/common.py": sha256_file(Path(sections.__code__.co_filename)),
                 "c01_statutes/precedents.py": sha256_file(Path(parse_precedents.__code__.co_filename))},
        "rows": len(rows),
        "counts": {"split": count("split"), "epoca": count("source_epoca"), "year": count("source_year"),
                   "instancia": count("source_instancia"), "publication_kind": count("publication_kind"),
                   "with_proceeding_ids": sum(bool(r["proceeding_ids"]) for r in rows)},
        "scan": scan_info["stats"],
        "train_equality": equality,
        "outputs": {"sources.jsonl": sha256_file(target), "exclusions.json": sha256_file(out / "exclusions.json")},
        "scope": "public SJF content; structured tesis only; v1 sealed excluded before decoding; no custody files read",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    manifest = build(args.out)
    print(json.dumps({k: manifest[k] for k in ("rows", "counts", "scan", "train_equality")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
