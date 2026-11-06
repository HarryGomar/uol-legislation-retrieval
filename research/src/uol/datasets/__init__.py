"""Canonical, portable dataset packages."""

from .canonical import (
    CANONICAL_POINTER_KINDS,
    CanonicalDatasetError,
    CanonicalDatasetReader,
    ForeignKey,
    OffsetCheck,
    TableContract,
    build_dataset,
    strict_json_loads,
    validate_dataset,
    validate_canonical_pointers,
)

__all__ = [
    "CANONICAL_POINTER_KINDS",
    "CanonicalDatasetError",
    "CanonicalDatasetReader",
    "ForeignKey",
    "OffsetCheck",
    "TableContract",
    "build_dataset",
    "strict_json_loads",
    "validate_dataset",
    "validate_canonical_pointers",
]
