"""Small deterministic helpers shared by the retrieval benchmark program."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

from uol.paths import project_root


ROOT = Path(project_root()).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            value.update(chunk)
    return value.hexdigest()


def rel(path: Path | str) -> str:
    candidate = Path(path).resolve()
    try:
        return candidate.relative_to(ROOT).as_posix()
    except ValueError:
        return str(candidate)


def read_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path | str) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            yield value


def _atomic_path(path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    return Path(name), handle


def atomic_write_bytes(path: Path | str, payload: bytes) -> None:
    target = Path(path)
    temporary, handle = _atomic_path(target)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_text(path: Path | str, text: str) -> None:
    atomic_write_bytes(Path(path), text.encode("utf-8"))


def atomic_write_json(path: Path | str, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    atomic_write_text(Path(path), payload)


def atomic_write_jsonl(path: Path | str, rows: Iterable[dict[str, Any]]) -> None:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(Path(path), payload)


def atomic_write_csv(path: Path | str, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    temporary, handle = _atomic_path(target)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


_SPACE = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def fold_text(value: str | None) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    normalized = "".join(char for char in normalized if not unicodedata.combining(char))
    normalized = normalized.lower().replace("\u00ba", "o").replace("\u00b0", "o")
    return _SPACE.sub(" ", normalized).strip()


def normalize_title(value: str | None) -> str:
    return _NON_WORD.sub(" ", fold_text(value)).strip()


_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")
_ORDINAL = re.compile(r"^(\d+)\s*(?:o)?\.?$")
_ARTICLE_HEADING = re.compile(
    r"^\s*[\"'“”]?\s*art(?:i|\ufffd)culo(?:s)?\s+"
    r"(?P<label>(?:\d[\d,]*(?:o)?(?:\s*-\s*[a-z0-9]+)*(?:\s+(?:[a-z]|bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|quintus))*|unico))"
    r"(?=\s*(?:\.-|[.:;-]|\d+\.))",
    flags=re.IGNORECASE,
)


def normalize_article_label(value: Any) -> str | None:
    if value is None:
        return None
    raw = fold_text(str(value)).strip(" .,:;_()[]{}")
    raw = _THOUSANDS.sub("", raw)
    raw = re.sub(r"^art(?:iculo)?s?\.?\s*", "", raw)
    raw = re.sub(r"[‐‑‒–—−]", "-", raw)
    raw = re.sub(r"\s*-\s*", "-", raw)
    raw = _SPACE.sub(" ", raw).strip()
    ordinal = _ORDINAL.fullmatch(raw)
    if ordinal:
        return str(int(ordinal.group(1)))
    suffix_word = r"(?:[a-z]|bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies|quintus)"
    match = re.fullmatch(
        rf"(\d+)(?P<hyphen>(?:-[a-z0-9]+)*)(?P<spaced>(?:\s+{suffix_word})*)\.?",
        raw,
    )
    if match:
        number = str(int(match.group(1)))
        suffixes = [part for part in match.group("hyphen").split("-") if part]
        suffixes.extend(match.group("spaced").split())
        return number if not suffixes else f"{number}-{'-'.join(suffixes)}"
    transitory = re.fullmatch(
        r"(?:(primero|segundo|tercero|cuarto|quinto|sexto|septimo|octavo|noveno|decimo)\s+)?transitorio(?:\s+(primero|segundo|tercero|cuarto|quinto|sexto|septimo|octavo|noveno|decimo))?",
        raw,
    )
    if transitory:
        order = transitory.group(1) or transitory.group(2)
        return "transitorio" if not order else f"transitorio-{order}"
    if re.fullmatch(r"[ivxlcdm]+", raw):
        return raw.upper()
    return raw or None


def extract_own_article_label(text: str | None) -> str | None:
    """Read the complete label from an article heading at the start of a source unit."""
    folded = fold_text(text)
    match = _ARTICLE_HEADING.match(folded)
    if not match:
        return None
    return normalize_article_label(match.group("label"))


def stable_id(prefix: str, *parts: Any, length: int = 24) -> str:
    return f"{prefix}:{digest(list(parts))[:length]}"


def stable_percent(*parts: Any) -> int:
    return int(digest(list(parts))[:8], 16) % 100


def stable_split(group_id: str, *, forced_development: bool = False) -> str:
    if forced_development:
        return "development"
    value = stable_percent("split-v1", group_id)
    if value < 70:
        return "train"
    if value < 85:
        return "development"
    return "test"


def file_receipt(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted({Path(item).resolve() for item in paths}, key=str):
        result[rel(path)] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result
