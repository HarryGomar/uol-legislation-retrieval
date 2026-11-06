"""Shared paths and reproducible, local-only evidence for the C01 repair."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[4]
STUDY = "benchmark_repair_20260929"
STATE = ROOT / "research/state" / STUDY
PROGRAM = ROOT / "research/programs" / STUDY
REPORT = ROOT / "research/reports/studies" / STUDY
RELEASE = ROOT / "data/releases/uol_c01_statutes_20260929_v2_development"
PREVIOUS = ROOT / "research/state/benchmark_validation_20260926"
OLD_RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(value: Any) -> str:
    return text_sha(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path: Path, value: Any, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise FileExistsError(path)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def save_rows(path: Path, rows: Iterable[dict], *, exclusive: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def safe_sources() -> list[dict]:
    receipt = json.loads((PREVIOUS / "preparation.json").read_text(encoding="utf-8"))
    path = PREVIOUS / "safe_training_sources.jsonl"
    if sha(path) != receipt["files"][path.name]:
        raise ValueError("Safe source export changed")
    sources = read_rows(path)
    if len(sources) != 2339 or any(row["split"] != "train" for row in sources):
        raise ValueError("Scope differs from the previously exposed original TRAIN population")
    for row in sources:
        if text_sha(row["raw_published_text"]) != row["source_content_sha256"]:
            raise ValueError("Source content changed")
        if any(row["raw_published_text"][s["start"]:s["end"]] != s["text"] for s in row["sections"]):
            raise ValueError("Section offsets do not reproduce source text")
    return sources


def log(phase: str, decision: str, why: str, evidence: str, result: str) -> None:
    path = STATE / "decisions.tsv"
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    def clean(value: str) -> str:
        value = str(value).replace("\t", " ").replace("\r", " ").replace("\n", " ")
        return "'" + value if value.startswith(("=", "+", "-", "@")) else value
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        if not exists:
            stream.write("ts\tphase\tdecision\twhy\tevidence\tresult\n")
        stream.write("\t".join(clean(v) for v in (utc(), phase, decision, why, evidence, result)) + "\n")
