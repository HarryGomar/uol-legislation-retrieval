"""Workstream C (Round 2): source rows for the two confirmation sets.

Sealed set: the C01 v1 ``test_sealed`` records. Their ids (and v1 component ids) are read from
the public v1 *method* query file with byte regexes only (``uol.facts_to_law.sources``); no
custody file and no v1 label is read. Their published text is decoded from the public SJF
content export, the same file and the same row builder that produced the v11 sources. This is
the one-time opening authorised by the author on 4 Oct 2026 ("do it all").

Prospective set: rows built by :func:`source_row` from SJF API detail responses collected by
``uol.round2.c_collect`` (same ``published`` object as the content export).

Rows keep the exact field names and values of ``sjf_structured_sources_v1/sources.jsonl`` so
the v11 release pipeline (``uol.round2.c_release``) consumes them unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from uol.facts_to_law import sources as ftl_sources
from uol.facts_to_law.spec import ROOT, sha256_file
from uol.retrieval_benchmarks.benchmark_validation.common import sections

RUN = ROOT / "experiments/runs/round2_20261004/C"
SEALED_DIR = RUN / "sealed"
V11_SOURCES = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"
COMPONENT_RE = re.compile(rb'"component_id"\s*:\s*"([^"\\]*)"')
RECORD_RE = re.compile(rb'"record_id"\s*:\s*"([^"\\]*)"')
SPLIT_RE = re.compile(rb'"split"\s*:\s*"([^"\\]*)"')


def sealed_membership(path: Path = ftl_sources.V1_QUERIES) -> dict[str, str]:
    """record_id -> v1 component_id for v1 ``test_sealed`` lines (byte regex only, never JSON-decoded)."""
    if "custody" in Path(path).parts:
        raise ValueError("custody paths are not inputs")
    out: dict[str, str] = {}
    with Path(path).open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            split = SPLIT_RE.findall(line)
            if len(split) != 1 or split[0] != b"test_sealed":
                continue
            record, comp = RECORD_RE.findall(line), COMPONENT_RE.findall(line)
            if len(record) != 1 or len(comp) != 1:
                raise ValueError("sealed v1 line lacks exactly one record_id/component_id")
            out[record[0].decode()] = comp[0].decode()
    return out


def source_row(published: dict, *, url: str | None, line_sha256: str, file_sha256: str, split: str,
               v1_component: str | None) -> tuple[dict | None, str]:
    """One structured source row exactly as ``uol.facts_to_law.sources.scan`` builds it.

    Returns (row or None, reason). ``reason`` is 'structured', 'no_hechos', 'nonstrict' or 'unstructured'.
    """
    text = str(published.get("texto") or "")
    if not re.search(r"(?i)hechos", json.dumps(published, ensure_ascii=False)):
        return None, "no_hechos"
    spans = sections(text)
    if [s["field"] for s in spans] != ftl_sources.STRUCTURE:
        return None, "nonstrict" if text.lstrip().startswith("Hechos") else "unstructured"
    record_id = str(published.get("idTesis"))
    date, kind = ftl_sources.publication(published.get("notaPublica"))
    return {
        "query_id": f"sjf:tesis:{record_id}:hechos",
        "record_id": record_id,
        "component_id": v1_component or f"sjf-record:{record_id}",
        "split": split,
        "v1_split": split if split.startswith("v1_") else None,
        "v1_component_id": v1_component,
        "raw_hechos": spans[0]["text"],
        "raw_published_text": text,
        "sections": spans,
        "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_record_sha256": line_sha256,
        "source_file_sha256": file_sha256,
        "source_url": url,
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
        "proceeding_ids": ftl_sources.proceeding_ids(published.get("precedentes"), published.get("organoJuris"), published.get("instancia")),
        "publication_date": date,
        "publication_kind": kind,
    }, "structured"


def check_builder(n: int = 60) -> dict:
    """The row builder reproduces ``n`` non-sealed v11 source rows field for field (non-sealed lines only)."""
    want = {}
    with V11_SOURCES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["split"] == "not_in_v1":
                want[r["record_id"]] = r
            if len(want) >= n:
                break
    sealed = set(sealed_membership())
    content_sha = next(iter(want.values()))["source_file_sha256"]
    checked = 0
    with ftl_sources.CONTENT.open("rb") as stream:
        for line in stream:
            ids = {i.decode() for i in ftl_sources.ID_RE.findall(line)}
            if ids & sealed or not (ids & set(want)):
                continue
            row = json.loads(line)
            pub = row.get("published") or {}
            got, _ = source_row(pub, url=row.get("url"), line_sha256=hashlib.sha256(line).hexdigest(),
                                file_sha256=content_sha, split="not_in_v1", v1_component=None)
            ref = want[str(pub.get("idTesis"))]
            if got != ref:
                diff = [k for k in ref if ref.get(k) != (got or {}).get(k)]
                raise ValueError(f"row builder differs from v11 export for {ref['record_id']}: {diff}")
            checked += 1
            if checked == len(want):
                break
    return {"rows_compared": checked, "identical": True}


def build_sealed(out: Path = SEALED_DIR) -> dict:
    """Decode the public content lines of the v1 sealed records once and write their source rows."""
    target = out / "sources.jsonl"
    if target.exists():
        raise FileExistsError(f"{target} exists; the sealed split is opened once")
    builder_check = check_builder()
    membership = sealed_membership()
    content_sha = sha256_file(ftl_sources.CONTENT)
    rows, reasons, seen = [], Counter(), set()
    with ftl_sources.CONTENT.open("rb") as stream:
        for line in stream:
            ids = {i.decode() for i in ftl_sources.ID_RE.findall(line)}
            hit = ids & set(membership)
            if not hit:
                continue
            row = json.loads(line)
            pub = row.get("published") or {}
            record_id = str(pub.get("idTesis") or row.get("requested_id"))
            if record_id not in membership:
                raise ValueError(f"byte/decoded id disagreement for {record_id}")
            seen.add(record_id)
            got, reason = source_row(pub, url=row.get("url"), line_sha256=hashlib.sha256(line).hexdigest(),
                                     file_sha256=content_sha, split="v1_test_sealed", v1_component=membership[record_id])
            reasons[reason] += 1
            if got:
                rows.append(got)
    rows.sort(key=lambda r: int(r["record_id"]))
    out.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    ids_sorted = sorted(membership)
    manifest = {
        "set": "sealed_v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "authority": "author in chat, 4 Oct 2026 ('do it all'): open the sealed v1 C01 split once as a confirmation set",
        "inputs": {"content": {"path": str(ftl_sources.CONTENT), "sha256": content_sha},
                   "v1_method_queries": {"path": str(ftl_sources.V1_QUERIES), "sha256": sha256_file(ftl_sources.V1_QUERIES),
                                         "read": "byte regex of split/record_id/component_id only; lines never decoded"}},
        "sealed_ids": len(membership), "sealed_ids_sha256": hashlib.sha256("\n".join(ids_sorted).encode()).hexdigest(),
        "content_lines_found": len(seen), "missing_ids": sorted(set(membership) - seen, key=int),
        "row_reasons": dict(reasons), "rows": len(rows), "builder_check": builder_check,
        "custody_files_read": 0, "v1_labels_read": 0,
        "outputs": {"sources.jsonl": sha256_file(target)},
        "code": {"c_sources.py": sha256_file(Path(__file__)), "facts_to_law/sources.py": sha256_file(Path(ftl_sources.__file__))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["check", "sealed"])
    a = p.parse_args()
    if a.command == "check":
        print(json.dumps(check_builder(), indent=1))
    else:
        m = build_sealed()
        print(json.dumps({k: m[k] for k in ("sealed_ids", "content_lines_found", "row_reasons", "rows", "builder_check")}, indent=1))


if __name__ == "__main__":
    main()
