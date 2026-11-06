"""Method-safe input loading for the C01 refinement programme.

The loader is the only sanctioned entry point for release inputs.  It rejects
sealed-reference paths before any filesystem call, opens large arrays as
read-only memmaps, and writes artifacts atomically with hash receipts.
"""
from __future__ import annotations

import hashlib
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from uol.retrieval_benchmarks.common import atomic_write_json, sha256_file

from . import contracts as C


class SealedReferenceAccess(RuntimeError):
    """Raised when a caller requests a forbidden sealed reference view."""


def _resolve_inside_release(relative: str) -> Path:
    candidate = (C.RELEASE / relative).resolve()
    release_root = C.RELEASE.resolve()
    if release_root not in candidate.parents and candidate != release_root:
        raise SealedReferenceAccess(f"path escapes the release root: {relative}")
    normalized = candidate.relative_to(release_root).as_posix()
    if normalized in C.FORBIDDEN_REFERENCE_PATHS or normalized.startswith("custody/"):
        raise SealedReferenceAccess(f"sealed custody access denied: {normalized}")
    return candidate


def custody_denial_test() -> dict[str, Any]:
    """Attempt every forbidden read and record deterministic failures."""
    observations: list[dict[str, Any]] = []
    for relative in C.FORBIDDEN_REFERENCE_PATHS:
        attempt: dict[str, Any] = {"path": relative}
        try:
            path = _resolve_inside_release(relative)
            with path.open("rb") as stream:  # pragma: no cover - must not succeed
                stream.read(1)
            attempt["denied"] = False
        except SealedReferenceAccess as error:
            attempt.update(denied=True, error_type=type(error).__name__, message=str(error))
        observations.append(attempt)
    denied = all(row["denied"] for row in observations)
    return {
        "status": "passed" if denied else "failed",
        "attempts": observations,
        "sealed_bytes_read": 0,
    }


def read_jsonl(relative: str) -> list[dict[str, Any]]:
    path = _resolve_inside_release(relative)
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object row in {relative}")
    return rows


def read_json(relative: str) -> Any:
    return json.loads(_resolve_inside_release(relative).read_text(encoding="utf-8"))


class SafeInputs:
    """Load public C01 release views and pinned read-only caches.

    Sealed references cannot be opened through this object: every accessor
    routes through ``_resolve_inside_release``.
    """

    def __init__(self) -> None:
        self._cache: dict[str, Any] = {}

    def queries(self) -> list[dict[str, Any]]:
        if "queries" not in self._cache:
            self._cache["queries"] = read_jsonl("method/queries.jsonl")
        return self._cache["queries"]

    def public_queries(self) -> list[dict[str, Any]]:
        return [row for row in self.queries() if row.get("split") in {"train", "development"}]

    def article_references(self) -> list[dict[str, Any]]:
        if "article_refs" not in self._cache:
            self._cache["article_refs"] = read_jsonl(
                "method/article_references_train_development.jsonl"
            )
        return self._cache["article_refs"]

    def instruments(self) -> list[dict[str, Any]]:
        if "instruments" not in self._cache:
            self._cache["instruments"] = read_jsonl("candidates/instruments.jsonl")
        return self._cache["instruments"]

    def articles(self) -> list[dict[str, Any]]:
        if "articles" not in self._cache:
            self._cache["articles"] = read_jsonl("candidates/articles.jsonl")
        return self._cache["articles"]

    def clauses(self) -> list[dict[str, Any]]:
        if "clauses" not in self._cache:
            self._cache["clauses"] = read_jsonl("candidates/clauses.jsonl")
        return self._cache["clauses"]

    def direct_article_edges(self) -> list[dict[str, Any]]:
        if "direct_edges" not in self._cache:
            self._cache["direct_edges"] = read_jsonl("graph/exact_direct_article_edges.jsonl")
        return self._cache["direct_edges"]

    def article_array(self) -> np.memmap:
        return np.load(C.ARTICLE_ARRAY, mmap_mode="r")

    def clause_array(self) -> np.memmap:
        return np.load(C.CLAUSE_ARRAY, mmap_mode="r")

    def query_array(self) -> np.memmap:
        return np.load(C.QUERY_ARRAY, mmap_mode="r")


def text_digest(values: Iterable[str]) -> str:
    hasher = hashlib.sha256()
    for value in values:
        hasher.update(hashlib.sha256(str(value).encode("utf-8")).hexdigest().encode("ascii"))
    return hasher.hexdigest()


def normalized_text_digest(values: Iterable[str]) -> str:
    """Aggregate digest over ``normalize_legal_input_v1`` text forms.

    Matches the clause-cache contract: NFKC + casefold + whitespace collapse
    per text, SHA-256 hexdigest of each, SHA-256 of the ASCII concatenation.
    """
    hasher = hashlib.sha256()
    for value in values:
        normalized = " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())
        hasher.update(hashlib.sha256(normalized.encode("utf-8")).hexdigest().encode("ascii"))
    return hasher.hexdigest()


def relative_str(path: Path) -> str:
    try:
        return path.resolve().relative_to(C.ROOT).as_posix()
    except ValueError:
        return str(path)


def jsonl_receipt(path: Path) -> dict[str, Any]:
    return {
        "path": relative_str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def array_receipt(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "path": relative_str(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }
