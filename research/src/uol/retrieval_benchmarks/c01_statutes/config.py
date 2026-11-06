"""Frozen paths and constants for the 2026-09-18 C01 statute build."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROGRAM_ID = "c01_statute_graph_retrieval_20260918"
SCHEMA_VERSION = "c01-statutes-1.0"
SPLIT_SEED = "c01-statutes-source-precedent-20260918-v1"

SJF_ROOT = ROOT / "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_views_v2"
SJF_CONTENT = ROOT / "data/releases/uol_mexico_tasks_20260915/sjf_bicentenario_content_full_v1/content.jsonl"
PROVISIONS = ROOT / "data/releases/uol_retrieval_benchmarks_20260914_v4/entities/provisions.parquet"
INSTRUMENT_CATALOGUE = ROOT / "data/releases/uol_retrieval_benchmarks_20260914_v4/entities/instrument_candidates.jsonl"

PROGRAM = ROOT / f"research/programs/{PROGRAM_ID}"
STATE = ROOT / f"research/state/{PROGRAM_ID}"
STAGING = ROOT / f".cache/{PROGRAM_ID}"
RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"
RUNS = ROOT / f"experiments/runs/{PROGRAM_ID}"
REPORT = ROOT / f"research/reports/studies/{PROGRAM_ID}"

EXPECTED = {
    "facts": 9584,
    "criterio_records": 3499,
    "justificacion_records": 7253,
    "label_records": 7946,
    "label_mentions": 18555,
    "precedent_mentions": 15435,
    "precedent_one_records": 6223,
    "precedent_multiple_records": 3361,
    "precedent_identities": 12609,
    "precedent_components": 7468,
    "precedent_multi_components": 1218,
    "precedent_multi_records": 3334,
    "instruments": 263,
    "articles": 36887,
}
