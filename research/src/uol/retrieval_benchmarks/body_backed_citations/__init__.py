"""Deterministic builders for body-backed documentary citation benchmarks."""

from .common import (
    CandidateIndex,
    Resolution,
    assign_splits,
    build_components,
    mask_query,
    normalize_article_label,
    normalize_identity,
)

__all__ = [
    "CandidateIndex",
    "Resolution",
    "assign_splits",
    "build_components",
    "mask_query",
    "normalize_article_label",
    "normalize_identity",
]
