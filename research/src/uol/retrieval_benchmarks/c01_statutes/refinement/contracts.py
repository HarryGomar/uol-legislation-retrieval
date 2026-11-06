"""Pinned contracts for the C01 representation-refinement programme.

Every constant in this module is copied from
``REPRESENTATION_REFINEMENT_SPEC.md`` (version ``c01-refinement-1.0``).  The
audit independently re-derives the observations before any fitting starts.
"""
from __future__ import annotations

from pathlib import Path


SPEC_VERSION = "c01-refinement-1.0"
PROGRAM_ID = "c01_statute_graph_retrieval_20260918"
RUN_ID = "refinement_v1"

ROOT = Path(__file__).resolve().parents[5]
RELEASE = ROOT / "data/releases/uol_c01_statutes_20260918_v1_candidate"
STATE = ROOT / f"research/state/{PROGRAM_ID}/{RUN_ID}"
RUN = ROOT / f"experiments/runs/{PROGRAM_ID}/{RUN_ID}"
REPORT_PATH = ROOT / f"research/reports/studies/{PROGRAM_ID}/REPRESENTATION_REFINEMENT_RESULTS.md"
SPEC_PATH = ROOT / f"research/programs/{PROGRAM_ID}/REPRESENTATION_REFINEMENT_SPEC.md"

ARTICLE_ARRAY = ROOT / (
    ".cache/legal_retrieval_deepening_20260917/qwen/"
    "candidate_article_c98204d1d98058b2_t8000_375d744ce0f5875368fc.npy"
)
ARTICLE_ARRAY_SHA256 = "b5655a78dbbee7e3d39882abb36695bba6ba7fc03b8cd4b0362ae1b1533c405c"
CLAUSE_ARRAY = ROOT / (
    ".cache/legal_retrieval_deepening_20260917/qwen/"
    "candidate_clause_250d5c841c097bc5_t8000_b128_bf3ae23996081069c3a6.npy"
)
CLAUSE_ARRAY_SHA256 = "f6c8591045a61d122e0f5630b0bac976d07090c269667e04df466535db2c31be"
CLAUSE_TEXT_SHA256 = "e30aac07a5f6625823e7e0233074576316f1bb0fcbd52adc6cf9244b98c87fa5"
CLAUSE_CONTRACT = "title-prefixed-legal-clause-v1:max900:min80:qwen-max8000"
QUERY_ARRAY = ROOT / (
    ".cache/c01_statute_graph_retrieval_20260918/qwen/"
    "structured_facts_default_t7600_eec227738a663d736896.npy"
)
QUERY_ARRAY_SHA256 = "2482e7443154fc8416da27223900410f60aea0dc987c9f759811bd45eb20c7fe"
QUERY_INSTRUCTION = (
    "Dada una narración de hechos judiciales mexicanos, recupera el artículo "
    "federal que después aparece como cita documental en el análisis jurídico. "
    "La identidad de la ley y del artículo puede estar oculta; distingue la "
    "materia, el mecanismo jurídico, el instrumento y el número de artículo."
)
RELEASE_MANIFEST_SHA256 = "39e489278af9e67d300101973915dd15ba1ebf0957a1be3919bf698cc4f755c5"
BASELINE_RESULT = ROOT / f"experiments/runs/{PROGRAM_ID}/exploratory_v3/result.json"
PROVISIONS = ROOT / "data/releases/uol_retrieval_benchmarks_20260914_v4/entities/provisions.parquet"
PROVISIONS_SHA256 = "3b12ac861a830ee4596b803b0481464264a9e808b4000cf18873fa1c2d06f4ee"

FORBIDDEN_REFERENCE_PATHS = (
    "custody/test/article_references.jsonl",
    "custody/test/instrument_references.jsonl",
)

DIMENSION = 4096
ARTICLE_ROWS = 36_887
CLAUSE_ROWS = 60_041
PUBLIC_QUERY_ROWS = 2_703
EXPECTED_DIRECT_EDGES = 2_124
EXPECTED_DIRECT_ENDPOINTS = 2_371
EXPECTED_GRAPH_COVERAGE = {
    "fitting_components": 1_620,
    "fitting_reference_connected": 815,
    "internal_validation_components": 423,
    "internal_validation_reference_connected": 215,
}
GRAPH_SUPPORT_GATE = {"minimum_fitting": 200, "minimum_internal_validation": 100}

# Training contract (spec section 5).
TAU = 0.05
RESIDUAL_LIMITS = (0.25, 0.50)
IDENTITY_LAMBDAS = (0.1, 1.0)
LEARNING_RATES = (3e-4, 1e-3)
SEEDS = (2027, 2029, 2039)
MAX_STEPS = 600
EVAL_EVERY = 100
WARMUP_STEPS = 100
BATCH_COMPONENTS = 32
NEGATIVES_PER_QUERY = 24
WEIGHT_DECAY = 0.01
GRADIENT_CLIP = 1.0
HARD_NEGATIVE_WEIGHT = 0.5
UNIFORM_NEGATIVE_WEIGHT = 1.0
NEGATIVE_MIXTURE = {
    "lexical_hard": 6,
    "dense_hard": 6,
    "same_law": 4,
    "same_label_other_law": 2,
    "graph_neighbor": 2,
    "high_degree": 2,
    "uniform": 2,
}
POOLING_WEIGHTS = (0.0, 0.25, 0.50, 0.75, 1.0)

# Parameter budgets (spec section 3 and method definitions).
RANK64_PARAMETERS = 524_288
RANK32_PAIR_PARAMETERS = 524_288
GRAPH_RANK16_PAIR_PARAMETERS = 262_144

# Evaluation and uncertainty (spec section 7).
PRIMARY_METRIC = "component_macro_ndcg_at_10"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260918
TOP_K_RANKINGS = 200
MIN_SLICE_COMPONENTS = 50

# Resource ceilings (spec section 9).
GPU_ALLOCATION_LIMIT_MIB = 7.2 * 1024
WORKING_RAM_LIMIT_GIB = 48.0
FREE_STORAGE_LIMIT_GIB = 50.0

# Archived fusion operator that must be preserved (spec section 6).
ARCHIVED_FUSION = {
    "lexical_method": "char_tfidf",
    "lexical_weight": 0.25,
    "normalization": "robust_z",
}

# M07 bridge authority (spec section 4/M07); generation stays disabled here.
BRIDGE_SCHEMA_ID = "c01-bridge-v1"
BRIDGE_AUTHORITY_PATH = STATE / "bridge_authority.json"

METHODS = ("m01", "m02", "m03", "m04", "m05", "m06", "m07")
