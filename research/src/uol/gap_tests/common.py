"""Shared helpers for the D56 gap tests: dev-only guards, component-macro values, contrasts, logging, peak memory.

Every test reads development rows only. ``assert_dev`` is called wherever rows meet scores or labels; no lockbox,
sealed or prospective row is indexed. Contrasts are ``stats.contrast`` (paired component bootstrap, 10,000 draws,
seed 20261004), as in ``embedding_push.common.contrast``.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.facts_to_law import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/runs/gap_tests_20261019"
PLAN = ROOT / "research/programs/gap_tests_20261019/PLAN.md"
_T0 = time.time()


def assert_dev(rows: Sequence[int]) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("a lockbox row reached a gap test")
    if not np.isin(rows, ep.dev_rows()).all():
        raise PermissionError("a non-dev row reached a gap test")
    return rows


def dev_pspec_rows() -> np.ndarray:
    return assert_dev(ep.population_rows("P_spec", ep.dev_rows()))


def peak_gb() -> float:
    """Peak working set of this process (Windows) or max RSS (POSIX), in GB."""
    try:
        import psutil
        info = psutil.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return peak / 2 ** 30
    except Exception:                                   # pragma: no cover
        return float("nan")


def log(test: str, msg: str) -> None:
    d = OUT / test
    d.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] (+{time.time() - _T0:.0f}s, peak {peak_gb():.1f} GB) {msg}"
    print(line, flush=True)
    with (d / "run.log").open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def component_means(values: Sequence[float], components: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """(sorted component ids, per-component mean) of per-unit values (units already averaged within case if needed)."""
    v = np.asarray(values, dtype=np.float64)
    ids, codes = np.unique(np.asarray(components), return_inverse=True)
    return ids, np.bincount(codes, weights=v, minlength=len(ids)) / np.bincount(codes, minlength=len(ids))


def macro(values: Sequence[float], components: Sequence[str]) -> float:
    return float(component_means(values, components)[1].mean())


def contrast(a: Sequence[float], b: Sequence[float], components: Sequence[str]) -> dict[str, Any]:
    """Paired component-bootstrap contrast a - b of per-unit values sharing ``components``."""
    ia, ca = component_means(a, components)
    _, cb = component_means(b, components)
    r = stats.contrast(ca, cb)
    return {"effect": r["effect"], "ci95": r["ci95"], "mean_a": r["mean_a"], "mean_b": r["mean_b"],
            "components": int(len(ia)), "p_sign_flip": r["p_sign_flip"], "verdict": r["verdict"]}


def fmt(c: dict[str, Any]) -> str:
    lo, hi = c["ci95"]
    return f"{c['effect']:+.3f} [{lo:+.3f}, {hi:+.3f}]"
