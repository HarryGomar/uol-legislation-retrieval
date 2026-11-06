"""Paths, the deployment signal set and the per-group signal configurations.

Signal hyper-parameters are not re-selected here: each dev fold uses the configuration the research pipeline's nested
inner CV chose for that fold (read from the delivered score manifests), and the deployed model uses the configuration
chosen on all dev rows (the manifests' ``lockbox`` scoring group, i.e. the group *trained on all dev rows*; no lockbox
row is read). The census memory has no research manifest; it is fixed at the modal M1 choice (Qwen key, k 10, 5-year
half-life), which M1's nested selection picked for three of five folds (disclosed in the report).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from uol.embedding_push import common
from uol.facts_to_law.spec import ROOT

TAG = "prototype_20261016"
RUN_HOME = ROOT / "experiments/runs" / TAG
RUN = Path(os.environ.get("UOL_PROTOTYPE_RUN") or RUN_HOME)      # models, results and latency log (override: smoke runs)
BUNDLE = ROOT / "data/derived" / TAG
REPORT = ROOT / "research/reports" / TAG
FIGURES = ROOT / "thesis/figures/prototype"
LATENCY_LOG = RUN / "latency.jsonl"

QWEN = ROOT / "data/derived/embedding_push_qwen8b_v1"
STATUTE_PATHS = ROOT / "experiments/runs/round3_best_case_20261013/phase2/statute_graph/paths.json"
UMAP_ARTICLES = ROOT / "experiments/runs/round3_best_case_20261013/arms/E1_structure/umap_articles.npy"
CENSUS_MEMORY = ROOT / "experiments/runs/round2_20261004/B/memory.jsonl"
CENSUS_TESIS = ROOT / "experiments/runs/round2_20261004/B/historical_tesis.jsonl"
RESEARCH_PIPES = common.RUN / "fusion/pipelines"

FULL_DEV = "full_dev"                      # the deployed group: fitted on every dev fold
GROUPS = ("dev_fold0", "dev_fold1", "dev_fold2", "dev_fold3", "dev_fold4", FULL_DEV)
MANIFEST_ROLE = {**{g: g for g in GROUPS[:5]}, FULL_DEV: "lockbox"}   # research manifests name the all-dev group "lockbox"

SIGNALS = ("L_star", "prior", "knn_lex_votes", "dense_qwen8b", "knn_qwen8b", "proto_qwen8b", "census_qwen8b")
EMB_SIGNALS = frozenset({"dense_qwen8b", "knn_qwen8b", "proto_qwen8b", "census_qwen8b"})
VARIANTS = {
    "deploy": SIGNALS,
    "deploy_nocensus": tuple(s for s in SIGNALS if s != "census_qwen8b"),
    "fallback": tuple(s for s in SIGNALS if s not in EMB_SIGNALS),
}
DEPLOYED, FALLBACK = "deploy", "fallback"
CENSUS_CFG = {"variant": "c", "k": 10, "half_life": 5.0}
DENSE_VARIANT = "c"                        # dense_qwen8b's selected query variant (qwen/selection/dense.json)
TOP_K = 10
POOL_K = 200                               # per-signal pool depth (fusion_ltr.TOPK)
ENDPOINT_BUDGET = 190                      # live endpoint calls allowed in total (brief: under 200)
NOTICE = "Documentary citation recovery, not legal advice."
NOTICE_ES = "Recuperación documental de citas; no es asesoría legal."


@dataclass(frozen=True)
class GroupConfig:
    group: str
    lstar: dict[str, Any]          # base.LSel fields
    knn_lex_k: int
    knn_variant: str
    knn_k: int
    proto_variant: str
    proto_mode: str
    proto_cn: int

    @property
    def variants(self) -> set[str]:
        return {DENSE_VARIANT, CENSUS_CFG["variant"], self.knn_variant, self.proto_variant}


def _manifest(source: str) -> dict[str, Any]:
    return json.loads((common.SCORES / source / "manifest.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def group_config(group: str) -> GroupConfig:
    role = MANIFEST_ROLE[group]
    lstar = _manifest("L_star")["configuration"][role]
    knn_lex = int(_manifest("knn_lex_votes")["configuration"][role])
    _, kv, kk = _manifest("knn_qwen8b")["configuration"][role].split("|")
    _, pv, pm, pc = _manifest("proto_qwen8b")["configuration"][role].split("|")
    return GroupConfig(group, dict(lstar), knn_lex, kv, int(kk.lstrip("k")), pv, pm, int(pc))


def fold_of_group(group: str) -> int | None:
    return None if group == FULL_DEV else int(group.replace("dev_fold", ""))


def training_folds(group: str) -> tuple[int, ...]:
    held = fold_of_group(group)
    return tuple(f for f in range(5) if f != held)
