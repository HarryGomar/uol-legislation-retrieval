"""Small common contract for canonical raw and processed datasets.

All paths stored by this module are release-relative POSIX paths. Validation
reopens every Parquet file and derives facts from its rows instead of trusting
the manifest's counts or schemas.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


FORMAT = "uol-canonical-dataset"
VERSION = 1


class CanonicalDatasetError(RuntimeError):
    pass


CANONICAL_POINTER_KINDS = frozenset({
    "canonical_source_snapshot",
    "canonical_processed_corpus",
})


def validate_canonical_pointers(
    pointers: Iterable[Mapping[str, Any]],
    *,
    required: Iterable[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Validate the population/product identity of active dataset pointers.

    ``required`` makes the distinction between "at most one" and "exactly one"
    explicit.  This lets a partial migration validate its current pointers
    without pretending that every requested population has been published.
    """
    failures: list[dict[str, Any]] = []
    active: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for ordinal, pointer in enumerate(pointers):
        population = pointer.get("population")
        product_kind = pointer.get("product_kind")
        status = pointer.get("status")
        is_active = pointer.get("active")
        if not isinstance(population, str) or not population:
            failures.append({"check": "pointer_population", "ordinal": ordinal})
            continue
        if product_kind not in CANONICAL_POINTER_KINDS:
            failures.append({"check": "pointer_product_kind", "ordinal": ordinal})
            continue
        try:
            _portable_path(pointer.get("path", ""))
        except (TypeError, CanonicalDatasetError) as exc:
            failures.append({"check": "pointer_path", "ordinal": ordinal,
                             "error": str(exc)})
        if type(is_active) is not bool:
            failures.append({"check": "pointer_active_type", "ordinal": ordinal})
            continue
        if is_active:
            if status not in CANONICAL_POINTER_KINDS:
                failures.append({"check": "active_pointer_status", "ordinal": ordinal,
                                 "status": status})
            active.setdefault((population, product_kind), []).append(pointer)
    for key, rows in sorted(active.items()):
        if len(rows) > 1:
            failures.append({"check": "duplicate_active_pointer",
                             "population": key[0], "product_kind": key[1],
                             "count": len(rows)})
    required_set = set(required)
    for population, product_kind in sorted(required_set):
        count = len(active.get((population, product_kind), []))
        if count != 1:
            failures.append({"check": "required_active_pointer",
                             "population": population, "product_kind": product_kind,
                             "count": count})
    return {
        "status": "passed" if not failures else "failed",
        "active_pairs": len(active),
        "required_pairs": len(required_set),
        "failures": failures,
    }


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalDatasetError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise CanonicalDatasetError(f"non-finite JSON number: {value}")


def strict_json_loads(raw: str | bytes) -> Any:
    if isinstance(raw, bytes):
        if raw.startswith(b"\xef\xbb\xbf"):
            raise CanonicalDatasetError("JSON BOM is forbidden")
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CanonicalDatasetError("JSON must be UTF-8") from exc
    if raw.startswith("\ufeff"):
        raise CanonicalDatasetError("JSON BOM is forbidden")
    if not raw.strip():
        raise CanonicalDatasetError("blank JSON record")
    try:
        return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except CanonicalDatasetError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalDatasetError(f"invalid JSON: {exc}") from exc


def _portable_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise CanonicalDatasetError(f"path is not repo-relative POSIX: {value!r}")
    path = PurePosixPath(value)
    lowered = [part.lower() for part in path.parts]
    windows_drive = bool(path.parts and len(path.parts[0]) == 2 and path.parts[0][1] == ":")
    if path.is_absolute() or windows_drive or ".." in path.parts or any(
        part == ".cache" or "basetemp" in part or part.startswith("shard")
        for part in lowered
    ):
        raise CanonicalDatasetError(f"forbidden consumption path: {value}")
    return value


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _schema_json(schema: pa.Schema) -> list[dict[str, Any]]:
    return [
        {"name": field.name, "type": str(field.type), "nullable": field.nullable}
        for field in schema
    ]


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]


@dataclass(frozen=True)
class OffsetCheck:
    start: str
    end: str
    text_table: str
    text_key: str
    local_text_key: str
    text_column: str


@dataclass(frozen=True)
class TableContract:
    name: str
    schema: pa.Schema
    primary_key: tuple[str, ...]
    sort_by: tuple[str, ...]
    foreign_keys: tuple[ForeignKey, ...] = ()
    offsets: tuple[OffsetCheck, ...] = ()

    def __post_init__(self) -> None:
        _portable_path(f"{self.name}.parquet")
        names = set(self.schema.names)
        for column in (*self.primary_key, *self.sort_by):
            if column not in names:
                raise CanonicalDatasetError(f"unknown contract column {self.name}.{column}")


def _contract_json(contract: TableContract) -> dict[str, Any]:
    return {
        "name": contract.name,
        "path": f"{contract.name}.parquet",
        "schema": _schema_json(contract.schema),
        "primary_key": list(contract.primary_key),
        "sort_by": list(contract.sort_by),
        "foreign_keys": [
            {"columns": list(f.columns), "target_table": f.target_table,
             "target_columns": list(f.target_columns)} for f in contract.foreign_keys
        ],
        "offsets": [o.__dict__ for o in contract.offsets],
    }


def _json(path: Path) -> dict[str, Any]:
    value = strict_json_loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CanonicalDatasetError(f"{path.name} must contain an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def build_dataset(output: Path | str, *, dataset_id: str, population: str,
                  kind: str, scientific_status: str, coverage: Mapping[str, Any],
                  lineage: Sequence[str], tables: Mapping[str, Iterable[Mapping[str, Any]]],
                  contracts: Sequence[TableContract], data_card: str | None = None) -> dict[str, Any]:
    """Build an immutable dataset directory and validate it before promotion."""
    output = Path(output).resolve()
    if output.exists():
        raise CanonicalDatasetError(f"refusing to overwrite {output}")
    if kind not in {"canonical_source_snapshot", "canonical_processed_corpus",
                    "qualified_intermediate", "quality_quarantined", "superseded_retained"}:
        raise CanonicalDatasetError(f"invalid dataset kind: {kind}")
    portable_lineage = [_portable_path(item) for item in lineage]
    by_name = {contract.name: contract for contract in contracts}
    if len(by_name) != len(contracts) or set(by_name) != set(tables):
        raise CanonicalDatasetError("tables and contracts differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        descriptors = []
        for name in sorted(by_name):
            contract = by_name[name]
            rows = [dict(row) for row in tables[name]]
            rows.sort(key=lambda row: tuple(row.get(column) for column in contract.sort_by))
            table = pa.Table.from_pylist(rows, schema=contract.schema)
            path = temporary / f"{name}.parquet"
            pq.write_table(table, path, compression="zstd", use_dictionary=True,
                           write_statistics=True)
            digest, size = _sha256(path)
            descriptors.append({"name": name, "path": path.name, "rows": table.num_rows,
                                "bytes": size, "sha256": digest})
        schema_doc = {"format": FORMAT, "version": VERSION,
                      "tables": [_contract_json(by_name[name]) for name in sorted(by_name)]}
        _write_json(temporary / "schema.json", schema_doc)
        schema_sha, schema_bytes = _sha256(temporary / "schema.json")
        if data_card is None:
            data_card = (
                f"# {dataset_id}\n\n"
                f"Population: {population}\n\n"
                f"Scientific status: {scientific_status}\n"
            )
        (temporary / "data_card.md").write_text(
            data_card.rstrip() + "\n", encoding="utf-8", newline="\n"
        )
        card_sha, card_bytes = _sha256(temporary / "data_card.md")
        manifest = {
            "format": FORMAT, "version": VERSION, "dataset_id": dataset_id,
            "population": population, "dataset_kind": kind,
            "scientific_status": scientific_status, "coverage": dict(coverage),
            "compression": "ZSTD", "lineage": portable_lineage,
            "canonical_order": {c.name: list(c.sort_by) for c in contracts},
            "schema": {"path": "schema.json", "sha256": schema_sha, "bytes": schema_bytes},
            "data_card": {"path": "data_card.md", "sha256": card_sha, "bytes": card_bytes},
            "tables": descriptors,
        }
        _write_json(temporary / "manifest.json", manifest)
        validation = validate_dataset(temporary, write=False)
        if validation["status"] != "passed":
            raise CanonicalDatasetError(
                f"refusing to promote failed dataset: {validation['failures']}"
            )
        _write_json(temporary / "validation.json", validation)
        temporary.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _key(row: Mapping[str, Any], columns: Sequence[str]) -> tuple[Any, ...]:
    return tuple(row.get(column) for column in columns)


def validate_dataset(root: Path | str, *, write: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    manifest = _json(root / "manifest.json")
    schema_doc = _json(root / "schema.json")
    failures: list[dict[str, Any]] = []
    if manifest.get("format") != FORMAT or schema_doc.get("format") != FORMAT:
        failures.append({"check": "format"})
    try:
        _portable_path(manifest.get("schema", {}).get("path", ""))
        for path in manifest.get("lineage", []):
            _portable_path(path)
        for item in manifest.get("tables", []):
            _portable_path(item.get("path", ""))
    except (AttributeError, TypeError, CanonicalDatasetError) as exc:
        failures.append({"check": "portable_paths", "error": str(exc)})
    schema_sha, schema_bytes = _sha256(root / "schema.json")
    if manifest.get("schema") != {"path": "schema.json", "sha256": schema_sha,
                                  "bytes": schema_bytes}:
        failures.append({"check": "schema_receipt"})
    try:
        card_path = _portable_path(manifest.get("data_card", {}).get("path", ""))
        card_sha, card_bytes = _sha256(root / card_path)
        if manifest.get("data_card") != {
            "path": card_path, "sha256": card_sha, "bytes": card_bytes
        }:
            failures.append({"check": "data_card_receipt"})
    except (AttributeError, OSError, TypeError, CanonicalDatasetError) as exc:
        failures.append({"check": "data_card_receipt", "error": str(exc)})
    contracts = {item["name"]: item for item in schema_doc.get("tables", [])}
    receipts = {item["name"]: item for item in manifest.get("tables", [])}
    rows_by_table: dict[str, list[dict[str, Any]]] = {}
    for name, contract in contracts.items():
        try:
            relative = _portable_path(contract["path"])
            path = root / relative
            digest, size = _sha256(path)
            parquet = pq.ParquetFile(path)
            rows = parquet.read().to_pylist()
            rows_by_table[name] = rows
            receipt = receipts.get(name, {})
            if (receipt.get("path"), receipt.get("sha256"), receipt.get("bytes"), receipt.get("rows")) != (relative, digest, size, len(rows)):
                failures.append({"check": "table_receipt", "table": name})
            if _schema_json(parquet.schema_arrow) != contract.get("schema"):
                failures.append({"check": "table_schema", "table": name})
            order = contract.get("sort_by", [])
            if rows != sorted(rows, key=lambda row: _key(row, order)):
                failures.append({"check": "table_order", "table": name})
            primary = contract.get("primary_key", [])
            keys = [_key(row, primary) for row in rows]
            if any(any(value is None for value in key) for key in keys) or len(keys) != len(set(keys)):
                failures.append({"check": "primary_key", "table": name})
        except (OSError, KeyError, TypeError, ValueError, pa.ArrowException,
                CanonicalDatasetError) as exc:
            failures.append({"check": "table_read", "table": name, "error": str(exc)})
    for name, contract in contracts.items():
        for foreign in contract.get("foreign_keys", []):
            target = rows_by_table.get(foreign["target_table"], [])
            target_keys = {_key(row, foreign["target_columns"]) for row in target}
            if any(_key(row, foreign["columns"]) not in target_keys for row in rows_by_table.get(name, [])):
                failures.append({"check": "foreign_key", "table": name,
                                 "target": foreign["target_table"]})
        for offset in contract.get("offsets", []):
            text_rows = rows_by_table.get(offset["text_table"], [])
            texts = {row[offset["text_key"]]: row[offset["text_column"]] for row in text_rows}
            bad = False
            for row in rows_by_table.get(name, []):
                text = texts.get(row.get(offset["local_text_key"]))
                start, end = row.get(offset["start"]), row.get(offset["end"])
                if not isinstance(text, str) or type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(text):
                    bad = True
            if bad:
                failures.append({"check": "offset", "table": name})
    result = {"format": FORMAT, "version": VERSION,
              "manifest_sha256": _sha256(root / "manifest.json")[0],
              "status": "passed" if not failures else "failed",
              "tables_checked": len(rows_by_table), "failures": failures}
    if write:
        _write_json(root / "validation.json", result)
    return result


class CanonicalDatasetReader:
    def __init__(self, root: Path | str):
        self.root = Path(root).resolve()
        self.validation = validate_dataset(self.root)
        if self.validation["status"] != "passed":
            raise CanonicalDatasetError(f"dataset validation failed: {self.validation['failures']}")
        self.manifest = _json(self.root / "manifest.json")

    def table(self, name: str) -> pa.Table:
        descriptor = next((item for item in self.manifest["tables"] if item["name"] == name), None)
        if descriptor is None:
            raise KeyError(name)
        return pq.read_table(self.root / _portable_path(descriptor["path"]))
