"""Facts-to-federal-instrument and facts-to-federal-article benchmarks."""

from __future__ import annotations

from typing import Any


def build_release() -> dict[str, Any]:
    from .build import build_release as implementation

    return implementation()


def validate_release() -> dict[str, Any]:
    from .build import validate_release as implementation

    return implementation()

__all__ = ["build_release", "validate_release"]
