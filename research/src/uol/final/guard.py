"""The test-set guard: development rows only, asserted wherever rows meet scores, labels or rankings.

Two layers:

* **Local** (full v11 release on the thesis PC): the SPLIT.json lockbox rows exist in the release; every row list
  that reaches a scorer, a score file, a label or an evaluator passes ``dev_rows_only`` first, which raises
  ``TestSetRefused`` for any lockbox row or any row outside the development set.
* **Bundle** (the dev-only export on the run computer): the release holds no lockbox row at all. ``check_bundle``
  re-derives the lockbox rule (a component whose earliest record_id is at or above the cut is test data) on every
  component of the loaded release and refuses to run if any component qualifies, or if BUNDLE.json does not
  declare a development-only export.

Population names that denote the test set (TEST, lockbox, sealed) are refused by name.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

CUT_RECORD_ID = 2030658                  # SPLIT.json "cut_record_id": first record of the test period (27 Jun 2025)
REFUSED_POPULATIONS = frozenset({"TEST", "test", "lockbox", "LOCKBOX", "sealed", "SEALED", "P_lockbox"})


class TestSetRefused(PermissionError):
    """Raised by every path that would read test-set rows, labels, rankings or scores."""

    __test__ = False                     # not a pytest test class


def refuse_population(name: str) -> None:
    if name in REFUSED_POPULATIONS or name.upper().startswith("TEST") or "lockbox" in name.lower():
        raise TestSetRefused(f"population {name!r} is the closed test set; it is never scored again (PLAN_DRAFT Q3)")


def _ep():
    from uol.embedding_push import common as ep
    return ep


def dev_rows_only(rows: Iterable[int], what: str = "rows") -> np.ndarray:
    """Return ``rows`` as int64 if every one is a development row of the active release; raise otherwise."""
    ep = _ep()
    rows = np.asarray(list(rows) if not isinstance(rows, np.ndarray) else rows, dtype=np.int64).ravel()
    if rows.size == 0:
        return rows
    lock = ep.lockbox_rows()
    if lock.size and np.isin(rows, lock).any():
        raise TestSetRefused(f"{what}: {int(np.isin(rows, lock).sum())} test-set (lockbox) rows refused")
    dev = ep.dev_rows()
    if not np.isin(rows, dev).all():
        raise TestSetRefused(f"{what}: rows outside the development set refused")
    return rows


def dev_query_ids_only(query_ids: Sequence[str], what: str = "query ids") -> list[str]:
    ep = _ep()
    roles = dict(zip((q.query_id for q in ep.release().queries), ep.roles()))
    bad = [q for q in query_ids if roles.get(q) is None or roles[q] == "lockbox"]
    if bad:
        raise TestSetRefused(f"{what}: {len(bad)} ids are test-set or unknown rows, e.g. {bad[:3]}")
    return list(query_ids)


def lockbox_components(queries) -> set[str]:
    """Components that the SPLIT rule assigns to the test set: earliest record_id >= CUT_RECORD_ID."""
    first: dict[str, int] = {}
    for q in queries:
        rid = int(q.record_id)
        first[q.component_id] = min(rid, first.get(q.component_id, rid))
    return {c for c, r in first.items() if r >= CUT_RECORD_ID}


def check_release_dev_only(queries, roles: Sequence[str] | None = None) -> None:
    """Bundle rule: no component of the loaded release may qualify as test data, and no role may be 'lockbox'."""
    bad = lockbox_components(queries)
    if bad:
        raise TestSetRefused(f"{len(bad)} test-set components are present in the release (cut record {CUT_RECORD_ID})")
    if roles is not None and any(str(r) == "lockbox" for r in roles):
        raise TestSetRefused("a lockbox role is present in the split")


def check_bundle(data_dir: Path) -> dict:
    """Run-time assertion on the run computer (called by ``context.activate_bundle``)."""
    meta_path = Path(data_dir) / "BUNDLE.json"
    if not meta_path.exists():
        raise TestSetRefused(f"{meta_path} missing: not a development-only bundle")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("dev_only") or meta.get("test_rows_included") or meta.get("lockbox_rows", 0):
        raise TestSetRefused("BUNDLE.json does not declare a development-only export")
    return meta
