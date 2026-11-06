"""Build and validate the portable SJF Bicentenario L17.02 Parquet release.

The frozen JSONL candidate is read in bounded batches.  Validation compares
every decoded JSONL row with the corresponding Parquet row and derives counts,
ordering, joins, offsets, and hashes without loading a complete table.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.parquet as pq

from uol.datasets.canonical import FORMAT, VERSION, strict_json_loads


CANDIDATE = Path(".cache/sjf_bicentenario_content_views_l17_02_candidate")
OUTPUT = Path("data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_views_v2")
SOURCE_MANIFEST = "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_full_v1/manifest.json"
REVIEW = "research/reports/studies/sjf_bicentenario_content_views_20260915_v2/review.json"
EXPECTED_CANDIDATE = {
    "documents.jsonl": (311965, "f5199393b031c2f9e4594f500ea6427728aebe4cf65610a3276627649d4b1a37"),
    "units.jsonl": (311965, "75ecb0f2cf3b9c28bf382caad0f6489db438217a3af4f0965a54ba5cfb16c8ef"),
    "source_blocks.jsonl": (440218, "fb39ed36c7e3cd0125da48c3a818537580150f80d7b4fd3e9e46a56a4e4bc861"),
    "occurrences.jsonl": (4885302, "87ddd4cb4a9d413f72be0b22dd3892f87654fd36bf41321f1d786940ecbf872e"),
    "citations.jsonl": (357189, "c48d4359dc35f0983e4a4c046c9a461f15a2e144b6d8091095f0d22c4a63c9cd"),
    "relations.jsonl": (357189, "99b64bc4ec6dcadab199fc5cb2da614429baf8734cd53ba932b7d14977f44d8c"),
}

SPAN = pa.struct([("block_id", pa.string()), ("start", pa.int64()), ("end", pa.int64())])
STRINGS = pa.list_(pa.field("element", pa.string()))
SPANS = pa.list_(pa.field("element", SPAN))
SCHEMAS = {
    "documents": pa.schema([
        ("document_id", pa.string()), ("asset_id", pa.string()), ("role", pa.string()),
        ("profile", pa.string()), ("source_profile", pa.string()), ("source_url", pa.string()),
        ("requested_id", pa.string()), ("ordinal", pa.int64()), ("body_sha256", pa.string()),
        ("body_bytes", pa.int64()), ("classification", pa.string()), ("published_json", pa.large_string()),
        ("published_fields", STRINGS), ("text_fields", STRINGS),
        ("exposure_metadata", pa.string()), ("exposure_body", pa.string()),
        ("exposure_indexing", pa.string()), ("original_document_opened", pa.bool_()),
        ("native_preserved", pa.bool_()), ("source_sequence", pa.int64()),
    ]),
    "units": pa.schema([
        ("unit_id", pa.string()), ("document_id", pa.string()), ("kind", pa.string()),
        ("role", pa.string()), ("status", pa.string()), ("label", pa.string()),
        ("source_profile", pa.string()), ("ordinal", pa.int64()), ("spans", SPANS),
        ("source_sha256", pa.string()), ("source_sequence", pa.int64()),
    ]),
    "fields": pa.schema([
        ("block_id", pa.string()), ("document_id", pa.string()), ("kind", pa.string()),
        ("json_pointer", pa.string()), ("source_field", pa.string()), ("field_order", pa.int32()),
        ("text", pa.large_string()), ("source_sequence", pa.int64()),
    ]),
    "occurrences": pa.schema([
        ("occurrence_id", pa.string()), ("unit_id", pa.string()), ("document_id", pa.string()),
        ("field", pa.string()), ("value_json", pa.large_string()), ("status", pa.string()),
        ("source_pointer", pa.string()), ("source_block_id", pa.string()), ("start", pa.int64()),
        ("end", pa.int64()), ("source_sha256", pa.string()), ("ordinal", pa.int64()),
        ("source_sequence", pa.int64()),
    ]),
    "citations": pa.schema([
        ("citation_id", pa.string()), ("document_id", pa.string()), ("unit_id", pa.string()),
        ("requested_id", pa.string()), ("ordinal", pa.int64()), ("field", pa.string()),
        ("surface", pa.large_string()), ("start", pa.int64()), ("end", pa.int64()),
        ("instrument", pa.string()), ("instrument_literal", pa.string()),
        ("articles", STRINGS), ("fractions", STRINGS),
        ("paragraphs", STRINGS), ("incisos", STRINGS),
        ("status", pa.string()), ("resolution_status", pa.string()),
        ("source_pointer", pa.string()), ("source_sha256", pa.string()), ("legal_truth", pa.bool_()),
        ("source_sequence", pa.int64()),
    ]),
    "relations": pa.schema([
        ("relation_id", pa.string()), ("source_id", pa.string()), ("target_ids", STRINGS),
        ("kind", pa.string()), ("status", pa.string()), ("spans", SPANS),
        ("citation_id", pa.string()), ("resolution_status", pa.string()), ("legal_truth", pa.bool_()),
        ("source_sequence", pa.int64()),
    ]),
}
SOURCE_TABLES = {
    "documents": "documents.jsonl", "units": "units.jsonl", "fields": "source_blocks.jsonl",
    "occurrences": "occurrences.jsonl", "citations": "citations.jsonl", "relations": "relations.jsonl",
}
PRIMARY_KEYS = {"documents": "document_id", "units": "unit_id", "fields": "block_id",
                "occurrences": "occurrence_id", "citations": "citation_id", "relations": "relation_id"}
SORT_BY = {name: ["source_sequence"] for name in SCHEMAS}


def _canon(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def _convert(name: str, row: Mapping[str, Any], source_sequence: int) -> dict[str, Any]:
    if name == "documents":
        exposure = row["exposure"]
        return {**{key: row[key] for key in (
            "document_id", "asset_id", "role", "profile", "source_profile", "source_url", "requested_id",
            "ordinal", "body_sha256", "body_bytes", "classification", "published_fields", "text_fields",
            "original_document_opened", "native_preserved")}, "published_json": _canon(row["published"]),
            "exposure_metadata": exposure["metadata"], "exposure_body": exposure["body"],
            "exposure_indexing": exposure["indexing"], "source_sequence": source_sequence}
    if name == "fields":
        locator = row["locator"]
        return {"block_id": row["block_id"], "document_id": row["document_id"], "kind": row["kind"],
                "json_pointer": locator["json_pointer"], "source_field": locator["source_field"],
                "field_order": row["order"], "text": row["text"], "source_sequence": source_sequence}
    if name == "occurrences":
        return {**{key: row[key] for key in row if key != "value"}, "value_json": _canon(row["value"]),
                "source_sequence": source_sequence}
    if name == "relations":
        attributes = row["attributes"]
        return {**{key: row[key] for key in ("relation_id", "source_id", "target_ids", "kind", "status", "spans")},
                "citation_id": attributes["citation_id"], "resolution_status": attributes["resolution_status"],
                "legal_truth": attributes["legal_truth"], "source_sequence": source_sequence}
    return dict(row) | {"source_sequence": source_sequence}


def _restore(name: str, row: Mapping[str, Any]) -> dict[str, Any]:
    row = dict(row)
    row.pop("source_sequence")
    if name == "documents":
        published = strict_json_loads(row.pop("published_json"))
        exposure = {"metadata": row.pop("exposure_metadata"), "body": row.pop("exposure_body"),
                    "indexing": row.pop("exposure_indexing")}
        row["published"] = published
        row["exposure"] = exposure
    elif name == "fields":
        row["locator"] = {"json_pointer": row.pop("json_pointer"), "source_field": row.pop("source_field")}
        row["order"] = row.pop("field_order")
    elif name == "occurrences":
        row["value"] = strict_json_loads(row.pop("value_json"))
    elif name == "relations":
        row["attributes"] = {"citation_id": row.pop("citation_id"),
                             "resolution_status": row.pop("resolution_status"),
                             "legal_truth": row.pop("legal_truth")}
    return row


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            value = strict_json_loads(raw)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: row is not an object")
            yield value


def _write_table(name: str, source: Path, target: Path, batch_size: int = 8192) -> int:
    writer = pq.ParquetWriter(target, SCHEMAS[name], compression="zstd", use_dictionary=True,
                              write_statistics=True, version="2.6")
    rows: list[dict[str, Any]] = []
    count = 0
    try:
        for source_sequence, source_row in enumerate(_iter_jsonl(source), 1):
            rows.append(_convert(name, source_row, source_sequence))
            if len(rows) == batch_size:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), row_group_size=batch_size)
                count += len(rows)
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMAS[name]), row_group_size=batch_size)
            count += len(rows)
    finally:
        writer.close()
    return count


def _schema_doc() -> dict[str, Any]:
    tables = []
    for name in sorted(SCHEMAS):
        fields = [{"name": field.name, "type": str(field.type), "nullable": field.nullable}
                  for field in SCHEMAS[name]]
        tables.append({"name": name, "path": f"{name}.parquet", "schema": fields,
                       "primary_key": [PRIMARY_KEYS[name]], "sort_by": SORT_BY[name],
                       "json_columns": (["published_json"] if name == "documents" else
                                        ["value_json"] if name == "occurrences" else [])})
    return {"format": FORMAT, "version": VERSION, "tables": tables}


def build(candidate: Path = CANDIDATE, output: Path = OUTPUT) -> dict[str, Any]:
    candidate, output = candidate.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    for filename, (_rows, expected_hash) in EXPECTED_CANDIDATE.items():
        if _sha(candidate / filename)[0] != expected_hash:
            raise ValueError(f"candidate hash mismatch: {filename}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        descriptors = []
        for name in sorted(SCHEMAS):
            source_name = SOURCE_TABLES[name]
            count = _write_table(name, candidate / source_name, temporary / f"{name}.parquet")
            if count != EXPECTED_CANDIDATE[source_name][0]:
                raise ValueError(f"row count mismatch: {name}")
            digest, size = _sha(temporary / f"{name}.parquet")
            descriptors.append({"name": name, "path": f"{name}.parquet", "rows": count,
                                "bytes": size, "sha256": digest})
        _json_write(temporary / "schema.json", _schema_doc())
        schema_sha, schema_bytes = _sha(temporary / "schema.json")
        card = """# SJF Bicentenario content views v2

Population: 311,965 published SJF Bicentenario API records in the frozen source snapshot.

Scientific status: qualified intermediate. The release preserves official published metadata and literal L17.02 citation candidates. Citations remain unresolved and do not establish instrument identity, version, validity, applicability, or legal truth. The API ficha is not an original judgment.

The six Parquet tables form one logical processed corpus. `published_json` and `value_json` contain canonical UTF-8 JSON because source field values are heterogeneous. All other nested values use typed Arrow lists or structs. Consumers do not need candidate, shard, checkpoint, run, or cache paths.
"""
        (temporary / "data_card.md").write_text(card, encoding="utf-8", newline="\n")
        card_sha, card_bytes = _sha(temporary / "data_card.md")
        manifest = {
            "format": FORMAT, "version": VERSION,
            "dataset_id": "sjf_bicentenario_content_views_v2",
            "population": "311965 published SJF Bicentenario API records",
            "dataset_kind": "qualified_intermediate",
            "scientific_status": "engineering_validated_literal_citations_unresolved",
            "coverage": {"source_records": 311965, "documents": 311965, "units": 311965,
                         "fields": 440218, "published_field_occurrences": 4885302,
                         "citation_occurrences": 357189, "citation_relations": 357189,
                         "documents_with_citations": 191234, "invalid_leading": 0,
                         "calendar_instrument": 0},
            "compression": "ZSTD",
            "lineage": [SOURCE_MANIFEST, REVIEW],
            "source_candidate_receipts": {name: {"rows": rows, "sha256": digest}
                                          for name, (rows, digest) in EXPECTED_CANDIDATE.items()},
            "canonical_order": SORT_BY,
            "schema": {"path": "schema.json", "sha256": schema_sha, "bytes": schema_bytes},
            "data_card": {"path": "data_card.md", "sha256": card_sha, "bytes": card_bytes},
            "tables": descriptors,
        }
        _json_write(temporary / "manifest.json", manifest)
        validation = validate(temporary, candidate=candidate)
        if validation["status"] != "passed":
            raise ValueError(f"release validation failed: {validation['failures']}")
        _json_write(temporary / "validation.json", validation)
        temporary.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _rows_from_parquet(path: Path) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def _validate_citation_spans(root: Path) -> tuple[int, list[dict[str, Any]]]:
    """Merge citation rows with field groups and verify Unicode slices."""
    failures: list[dict[str, Any]] = []
    fields = _rows_from_parquet(root / "fields.parquet")
    current = next(fields, None)
    group_ordinal = 0
    group_document: str | None = None
    group: dict[str, str] = {}

    def advance_group() -> bool:
        nonlocal current, group_ordinal, group_document, group
        if current is None:
            return False
        group_ordinal += 1
        group_document = current["document_id"]
        group = {}
        while current is not None and current["document_id"] == group_document:
            group[current["source_field"]] = current["text"]
            current = next(fields, None)
        return True

    checked = 0
    for citation in _rows_from_parquet(root / "citations.parquet"):
        while group_ordinal < citation["ordinal"] and advance_group():
            pass
        text = group.get(citation["field"]) if group_document == citation["document_id"] else None
        start, end = citation["start"], citation["end"]
        if not isinstance(text, str) or not 0 <= start < end <= len(text) or text[start:end] != citation["surface"]:
            failures.append({"check": "citation_text_span", "citation_id": citation["citation_id"]})
            break
        checked += 1
    return checked, failures


def validate(root: Path = OUTPUT, *, candidate: Path = CANDIDATE) -> dict[str, Any]:
    root, candidate = root.resolve(), candidate.resolve()
    failures: list[dict[str, Any]] = []
    manifest = strict_json_loads((root / "manifest.json").read_bytes())
    schema_doc = strict_json_loads((root / "schema.json").read_bytes())
    receipts = {item["name"]: item for item in manifest["tables"]}
    observed: dict[str, int] = {}
    for filename, (_expected_rows, expected_hash) in EXPECTED_CANDIDATE.items():
        if _sha(candidate / filename)[0] != expected_hash:
            failures.append({"check": "candidate_hash", "file": filename})
    for name in sorted(SCHEMAS):
        path = root / f"{name}.parquet"
        parquet = pq.ParquetFile(path)
        if not parquet.schema_arrow.equals(SCHEMAS[name], check_metadata=True):
            failures.append({"check": "schema", "table": name})
        codecs = {parquet.metadata.row_group(group).column(column).compression
                  for group in range(parquet.metadata.num_row_groups)
                  for column in range(parquet.metadata.num_columns)}
        if codecs != {"ZSTD"}:
            failures.append({"check": "compression", "table": name, "observed": sorted(codecs)})
        digest, size = _sha(path)
        receipt = receipts.get(name, {})
        if (receipt.get("sha256"), receipt.get("bytes")) != (digest, size):
            failures.append({"check": "receipt", "table": name})
        count = 0
        previous: tuple[Any, ...] | None = None
        seen_key: str | None = None
        source_iter = _iter_jsonl(candidate / SOURCE_TABLES[name])
        for count, (source_row, parquet_row) in enumerate(zip(source_iter, _rows_from_parquet(path), strict=True), 1):
            restored = _restore(name, parquet_row)
            if restored != source_row:
                failures.append({"check": "round_trip", "table": name, "row": count})
                break
            key = parquet_row[PRIMARY_KEYS[name]]
            if key is None or key == seen_key:
                failures.append({"check": "primary_key", "table": name, "row": count})
                break
            seen_key = key
            order = tuple(parquet_row[column] for column in SORT_BY[name])
            if previous is not None and order < previous:
                failures.append({"check": "order", "table": name, "row": count})
                break
            previous = order
            document_id = parquet_row.get("document_id")
            ordinal = parquet_row.get("ordinal")
            requested = source_row.get("requested_id") or (document_id.rsplit(":", 1)[-1] if document_id else None)
            if document_id is not None and document_id != f"sjf:document:{requested}":
                failures.append({"check": "document_join", "table": name, "row": count})
                break
            unit_id = parquet_row.get("unit_id")
            if unit_id is not None and unit_id != f"sjf:tesis:{requested}":
                failures.append({"check": "unit_join", "table": name, "row": count})
                break
            if name == "fields" and parquet_row["block_id"] != f"sjf:{requested}:published:{parquet_row['source_field'].replace('~', '~0').replace('/', '~1')}":
                failures.append({"check": "field_identity", "table": name, "row": count})
                break
            if name == "occurrences" and parquet_row["occurrence_id"] != f"{unit_id}:field:{parquet_row['field'].replace('~', '~0').replace('/', '~1')}":
                failures.append({"check": "occurrence_identity", "table": name, "row": count})
                break
            if name == "occurrences" and parquet_row["source_block_id"] is not None:
                value = strict_json_loads(parquet_row["value_json"])
                if (not isinstance(value, str) or parquet_row["start"] != 0 or
                        parquet_row["end"] != len(value)):
                    failures.append({"check": "field_offset", "table": name, "row": count})
                    break
            if name == "citations":
                if not (0 <= parquet_row["start"] < parquet_row["end"] and
                        parquet_row["surface"] == source_row["surface"]):
                    failures.append({"check": "citation_offset", "table": name, "row": count})
                    break
                suffix = hashlib.sha256(parquet_row["surface"].encode("utf-8")).hexdigest()[:16]
                expected_id = f"{unit_id}:citation:{parquet_row['field']}:{parquet_row['start']}:{parquet_row['end']}:{suffix}"
                if parquet_row["citation_id"] != expected_id:
                    failures.append({"check": "citation_identity", "table": name, "row": count})
                    break
            if name == "relations" and parquet_row["relation_id"] != f"{parquet_row['citation_id']}:relation":
                failures.append({"check": "relation_join", "table": name, "row": count})
                break
        observed[name] = count
        if count != EXPECTED_CANDIDATE[SOURCE_TABLES[name]][0] or receipt.get("rows") != count:
            failures.append({"check": "row_count", "table": name, "observed": count})
    schema_sha, schema_bytes = _sha(root / "schema.json")
    if manifest.get("schema") != {"path": "schema.json", "sha256": schema_sha, "bytes": schema_bytes}:
        failures.append({"check": "schema_receipt"})
    card_sha, card_bytes = _sha(root / "data_card.md")
    if manifest.get("data_card") != {"path": "data_card.md", "sha256": card_sha, "bytes": card_bytes}:
        failures.append({"check": "data_card_receipt"})
    metadata_text = _canon(manifest) + _canon(schema_doc)
    for forbidden in (".cache", "basetemp", "C:\\", "shard"):
        if forbidden.casefold() in metadata_text.casefold():
            failures.append({"check": "portable_paths", "value": forbidden})
    citation_spans_checked, span_failures = _validate_citation_spans(root)
    failures.extend(span_failures)
    return {"format": FORMAT, "version": VERSION, "status": "passed" if not failures else "failed",
            "manifest_sha256": _sha(root / "manifest.json")[0], "tables_checked": len(observed),
            "rows_checked": observed, "citation_text_spans_checked": citation_spans_checked,
            "bounded_batch_rows": 8192, "checks": [
                "candidate_hashes", "parquet_hashes", "explicit_schemas", "zstd", "row_counts",
                "exact_round_trip", "deterministic_order", "primary_keys", "derived_joins", "offset_bounds",
                "portable_metadata"], "failures": failures}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "validate"])
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--write-validation", action="store_true")
    args = parser.parse_args(argv)
    value = build(args.candidate, args.output) if args.command == "build" else validate(args.output, candidate=args.candidate)
    if args.command == "validate" and args.write_validation and value["status"] == "passed":
        _json_write(args.output / "validation.json", value)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))
    return 0 if value.get("status", "passed") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
