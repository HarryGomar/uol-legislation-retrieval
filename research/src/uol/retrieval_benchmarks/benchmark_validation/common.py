"""Prepare immutable training-only source exports before independent audits.

Mixed files are traversed as bytes. Non-training label rows and non-allowlisted
source records are rejected before JSON deserialization. No protected reference
or historical unfiltered audit file is an input to this module.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
STUDY = "benchmark_validation_20260926"
STATE = ROOT / "research/state" / STUDY
PROGRAM = ROOT / "research/programs" / STUDY
RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"
ARTICLES = RELEASE / "candidates/articles.jsonl"
INSTRUMENTS = RELEASE / "candidates/instruments.jsonl"
SOURCE = ROOT / "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_full_v1/content.jsonl"
SALT = "benchmark-validation-20260926-v1"
SPLIT_RE = re.compile(rb'"split"\s*:\s*"([^"\\]+)"')
REQUEST_RE = re.compile(rb'"requested_id"\s*:\s*"?([0-9]+)')
HEADINGS = re.compile(r"(?im)(?:^|[\r\n])[ \t]*(Hechos|Criterio jur[ií]dico|Justificaci[oó]n)[ \t]*:")


def file_hash(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl_new(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def training_rows(path: Path) -> tuple[list[dict], int]:
    """Reject all other splits before parsing a mixed public row."""
    if "custody" in path.parts or "review" in path.parts:
        raise ValueError("Protected or unfiltered audit paths are not allowed")
    result, skipped = [], 0
    with path.open("rb") as stream:
        for line in stream:
            if not line.strip():
                continue
            match = SPLIT_RE.search(line)
            if match is None:
                raise ValueError("Missing split discriminator")
            if match.group(1) != b"train":
                skipped += 1
                continue
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError("Split prefilter disagrees with parsed row")
            result.append(row)
    return result, skipped


def allowed_sources(path: Path, allowed_record_ids: set[str]):
    """Yield only explicitly allowed raw source records, without parsing others."""
    if not allowed_record_ids or any(not value.isdigit() for value in allowed_record_ids):
        raise ValueError("A nonempty numeric source allowlist is required")
    allowed = {value.encode() for value in allowed_record_ids}
    with path.open("rb") as stream:
        for line in stream:
            match = REQUEST_RE.search(line)
            if match is None or match.group(1) not in allowed:
                continue
            row = json.loads(line)
            record_id = str(row.get("requested_id"))
            if record_id not in allowed_record_ids:
                raise ValueError("Source allowlist prefilter disagrees with parsed ID")
            yield row, hashlib.sha256(line).hexdigest()


def sections(text: str) -> list[dict]:
    """Independent heading split, retaining exact source offsets."""
    matches = list(HEADINGS.finditer(text))
    result = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        label = match.group(1).casefold().replace("í", "i").replace("ó", "o")
        field = {"hechos": "hechos", "criterio juridico": "criterio_juridico", "justificacion": "justificacion"}[label]
        result.append({"field": field, "text": text[start:end], "start": start, "end": end})
    return result


def choose_sample(queries: list[dict], qrels: dict[str, dict], size: int = 200) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in queries:
        if row["query_id"] in qrels:
            grouped[row["component_id"]].append(row)
    key = lambda namespace, value: hashlib.sha256(f"{SALT}:{namespace}:{value}".encode()).hexdigest()
    if len(grouped) < size:
        raise ValueError("Insufficient independent components; no replacement permitted")
    sample = []
    for component in sorted(grouped, key=lambda c: key("component", c))[:size]:
        query = min(grouped[component], key=lambda r: key("query", r["query_id"]))
        positives = qrels[query["query_id"]]["positive_ids"]
        positive = min(positives, key=lambda p: key("positive", query["query_id"] + ":" + p))
        sample.append({"query_id": query["query_id"], "component_id": component, "record_id": query["record_id"], "selected_positive_id": positive})
    return sample


def prepare() -> dict:
    if (STATE / "preparation.json").exists():
        raise FileExistsError("Preparation already exists; preserve frozen export")
    protocol_path = PROGRAM / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["sampling_salt"] != SALT or protocol["documentary_sample_components"] != 200:
        raise ValueError("Protocol and sampling code disagree")
    queries, skipped_queries = training_rows(RELEASE / "method/queries.jsonl")
    references, skipped_references = training_rows(RELEASE / "method/article_references_train_development.jsonl")
    qrels = {row["query_id"]: row for row in references}
    queries = [q for q in queries if q["query_id"] in qrels]
    if len(queries) != 2339 or len(qrels) != 2339 or len({q["record_id"] for q in queries}) != 2339:
        raise ValueError("Frozen training population differs")
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    source_hash = file_hash(SOURCE)
    if source_hash != manifest["inputs"]["sjf_content"]["sha256"]:
        raise ValueError("Source corpus changed from release binding")
    sample = choose_sample(queries, qrels)
    # Lock sampling before any raw source text is inspected by this preparation.
    write_jsonl_new(STATE / "frozen_selection.jsonl", sample)
    write_jsonl_new(STATE / "audit_sample_sources.jsonl", [{k: r[k] for k in ("query_id", "record_id", "component_id")} for r in sample])
    write_jsonl_new(STATE / "training_qrels.jsonl", references)
    query_by_record = {q["record_id"]: q for q in queries}
    prepared, issues, seen = [], [], set()
    for source, record_hash in allowed_sources(SOURCE, set(query_by_record)):
        record_id = str(source["requested_id"])
        if record_id in seen:
            raise ValueError("Duplicate allowed source record")
        seen.add(record_id)
        query = query_by_record[record_id]
        published = source.get("published") or {}
        text = str(published.get("texto") or "")
        spans = sections(text)
        facts = [s for s in spans if s["field"] == "hechos"]
        if len(facts) != 1 or [s["field"] for s in spans] != ["hechos", "criterio_juridico", "justificacion"]:
            issues.append({"query_id": query["query_id"], "reason": "unexpected_heading_structure", "fields": [s["field"] for s in spans]})
        raw_hechos = facts[0]["text"] if len(facts) == 1 else ""
        prepared.append({"query_id": query["query_id"], "record_id": record_id, "component_id": query["component_id"], "split": "train", "raw_hechos": raw_hechos, "historical_text": query["text"], "historical_mask_ledger": query["mask_ledger"], "historical_unmasked_text_sha256": query["unmasked_text_sha256"], "independent_hechos_json_sha256": canonical_hash(raw_hechos), "sections": spans, "raw_published_text": text, "source_content_sha256": hashlib.sha256(text.encode()).hexdigest(), "source_record_sha256": record_hash, "source_file_sha256": source_hash, "source_url": source.get("url"), "source_year": published.get("anio"), "source_rubro": published.get("rubro"), "source_materias": published.get("materias"), "source_tipo_tesis": published.get("tipoTesis"), "source_precedentes": published.get("precedentes"), "source_published_id": published.get("idTesis")})
    for missing in sorted(set(query_by_record) - seen):
        issues.append({"query_id": query_by_record[missing]["query_id"], "reason": "source_missing"})
    prepared.sort(key=lambda r: r["query_id"])
    write_jsonl_new(STATE / "safe_training_sources.jsonl", prepared)
    files = {name: file_hash(STATE / name) for name in ("safe_training_sources.jsonl", "training_qrels.jsonl", "frozen_selection.jsonl", "audit_sample_sources.jsonl")}
    receipt = {"utc": datetime.now(timezone.utc).isoformat(), "status": "prepared" if not issues else "prepared_with_source_issues", "protocol_sha256": file_hash(protocol_path), "preparation_code_sha256": file_hash(Path(__file__)), "source_corpus_sha256": source_hash, "public_query_file_sha256": file_hash(RELEASE / "method/queries.jsonl"), "public_reference_file_sha256": file_hash(RELEASE / "method/article_references_train_development.jsonl"), "candidate_articles_sha256": file_hash(ARTICLES), "candidate_instruments_sha256": file_hash(INSTRUMENTS), "files": files, "queries": len(queries), "sources": len(prepared), "components": len({q['component_id'] for q in queries}), "sample_components": len(sample), "nontraining_query_rows_skipped_before_deserialization": skipped_queries, "nontraining_reference_rows_skipped_before_deserialization": skipped_references, "raw_source_scope": "Only allowlisted original TRAIN record IDs deserialized; other raw lines traversed as bytes", "sealed_reference_files_opened": 0, "issues": issues, "independent_vs_historical_exact_hechos_digest_matches": sum(r["independent_hechos_json_sha256"] == r["historical_unmasked_text_sha256"] for r in prepared), "candidate_paths": {"articles": str(ARTICLES.relative_to(ROOT)), "instruments": str(INSTRUMENTS.relative_to(ROOT))}}
    atomic_json(STATE / "preparation.json", receipt)
    return receipt


def load_safe_training() -> tuple[list[dict], dict[str, list[str]], dict]:
    receipt = json.loads((STATE / "preparation.json").read_text(encoding="utf-8"))
    if file_hash(PROGRAM / "protocol.json") != receipt["protocol_sha256"]:
        raise ValueError("Validation protocol changed")
    for name in ("safe_training_sources.jsonl", "training_qrels.jsonl"):
        if file_hash(STATE / name) != receipt["files"][name]:
            raise ValueError(f"Prepared input changed: {name}")
    if file_hash(ARTICLES) != receipt["candidate_articles_sha256"] or file_hash(INSTRUMENTS) != receipt["candidate_instruments_sha256"]:
        raise ValueError("Frozen catalogue changed")
    sources = [json.loads(line) for line in (STATE / "safe_training_sources.jsonl").read_text(encoding="utf-8").splitlines()]
    refs = [json.loads(line) for line in (STATE / "training_qrels.jsonl").read_text(encoding="utf-8").splitlines()]
    if any(row.get("split") != "train" for row in sources + refs):
        raise ValueError("Only original TRAIN records are eligible")
    return sources, {r["query_id"]: r["positive_ids"] for r in refs}, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare"])
    parser.parse_args()
    result = prepare()
    print(json.dumps({k: result[k] for k in ("status", "sources", "components", "sample_components", "issues", "independent_vs_historical_exact_hechos_digest_matches")}))
