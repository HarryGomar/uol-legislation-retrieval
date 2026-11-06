"""Workstream B (P07) shared paths, logging and small helpers.

Every output goes to ``experiments/runs/round2_20261004/B``; nothing is written under the embedding-push run,
the releases, the embedding-push state or the source home.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from uol.facts_to_law.spec import ROOT

OUT = ROOT / "experiments/runs/round2_20261004/B"
LOG = OUT / "run.log"
REPORT = ROOT / "research/reports/round2_20261004/B_citation_context.md"
SEED = 20261004

SOURCE_HOME = Path(r"C:\uol_legacy_home")
CENSUS = SOURCE_HOME / "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_full_v1/content.jsonl"
CENSUS_SHA256 = "334bfcf8a7bcf6cfd748e8d6b74cfa30d75e2a169802ec885798156dee3182ba"
V11_SOURCES = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"
CANDIDATES = ROOT / "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates"

HIST = OUT / "historical_tesis.jsonl"          # historical pool (texts), one row per tesis
HIST_META = OUT / "historical_pool.json"
MENTIONS = OUT / "mentions.jsonl"              # resolved article mentions per historical tesis (spans)
RESOLVE_META = OUT / "resolver_summary.json"


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


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=float, ensure_ascii=False), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()
