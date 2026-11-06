"""Paired component statistics and the pre-registered decision rule.

Source components are the independent units (PREREGISTRATION §4): every
procedure resamples or sign-flips per-component values, never queries. Each
procedure draws from its own ``np.random.default_rng(seed)`` with the recorded
seed (default 20261004); the October helper instead interleaved bootstrap and
sign-flip draws in one stream, so its intervals are not reproduced draw for
draw. As in October, per-component differences within 1e-12 of zero are set to
zero before resampling, and intervals are percentile intervals (NumPy's
default linear quantiles).
"""
from __future__ import annotations

import math
import operator
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

SEED = 20261004
DRAWS = 10000
MARGIN = 0.02
ALPHA = 0.05
POWER = 0.8
DELTA_TOLERANCE = 1e-12
_CHUNK = 256

USEFUL_IMPROVEMENT = "USEFUL_IMPROVEMENT"
POSITIVE_BELOW_USEFUL = "POSITIVE_BELOW_USEFUL"
IMPROVEMENT_SIZE_UNRESOLVED = "IMPROVEMENT_SIZE_UNRESOLVED"
NO_USEFUL_DIFFERENCE = "NO_USEFUL_DIFFERENCE"
WORSE = "WORSE"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_ASSESSED = "NOT_ASSESSED"
VERDICTS = (USEFUL_IMPROVEMENT, POSITIVE_BELOW_USEFUL, IMPROVEMENT_SIZE_UNRESOLVED, NO_USEFUL_DIFFERENCE,
            WORSE, INCONCLUSIVE, NOT_ASSESSED)
NON_INFERIOR = "NON_INFERIOR"
NON_INFERIORITY_NOT_SHOWN = "NON_INFERIORITY_NOT_SHOWN"


def _differences(diff: np.ndarray) -> np.ndarray:
    values = np.array(diff, dtype=np.float64, copy=True)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("at least two paired component differences are required")
    if not np.isfinite(values).all():
        raise ValueError("non-finite component difference")
    values[np.abs(values) <= DELTA_TOLERANCE] = 0.0
    return values


def _draws(draws: int) -> int:
    draws = operator.index(draws)
    if draws < 100:
        raise ValueError("at least 100 resamples are required")
    return draws


def _probability(value: float, name: str, upper: float = 1.0) -> float:
    if not 0 < value < upper:
        raise ValueError(f"{name} must lie strictly between 0 and {upper}")
    return float(value)


def _bootstrap_means(values: np.ndarray, draws: int, seed: int) -> np.ndarray:
    """Means of ``draws`` component resamples (with replacement), in chunks of 256."""
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, _CHUNK):
        count = min(_CHUNK, draws - start)
        means[start:start + count] = values[rng.integers(0, values.size, size=(count, values.size))].mean(axis=1)
    return means


def _interval(boot: np.ndarray, confidence: float) -> tuple[float, float]:
    tail = (1 - confidence) / 2
    lower, upper = np.quantile(boot, [tail, 1 - tail])
    return float(lower), float(upper)


def paired_cluster_bootstrap(diff: np.ndarray, draws: int = DRAWS, seed: int = SEED,
                             confidence: float = 0.95) -> dict[str, Any]:
    """Percentile interval for the mean per-component paired difference.

    ``sd`` is the sample SD (ddof 1) of the component differences, the input
    of ``mde``; ``se`` is the SD of the bootstrap means.
    """
    values = _differences(diff)
    draws, confidence = _draws(draws), _probability(confidence, "confidence")
    boot = _bootstrap_means(values, draws, seed)
    lower, upper = _interval(boot, confidence)
    return {"effect": float(values.mean()), "lower": lower, "upper": upper, "sd": float(values.std(ddof=1)),
            "se": float(boot.std(ddof=1)), "n": int(values.size), "draws": int(draws), "seed": int(seed),
            "confidence": float(confidence)}


def sign_flip_pvalue(diff: np.ndarray, draws: int = DRAWS, seed: int = SEED) -> float:
    """Two-sided Monte Carlo sign-flip p-value for a zero mean component difference.

    p = (1 + #{|mean(s * d)| >= |mean(d)| - 1e-15}) / (draws + 1), the October
    formula, with independent random signs s per component and draw.
    """
    values = _differences(diff)
    draws = _draws(draws)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    for start in range(0, draws, _CHUNK):
        count = min(_CHUNK, draws - start)
        signs = rng.integers(0, 2, size=(count, values.size), dtype=np.int8) * np.int8(2) - np.int8(1)
        exceed += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
    return (1 + exceed) / (draws + 1)


def holm(pvalues: Mapping[str, float], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """Holm step-down adjustment: {name: {p, p_adjusted, reject, step}} in input order.

    ``p_adjusted`` is the running maximum of min(1, (m - i + 1) * p_(i)) over
    ascending p-values; a hypothesis is rejected when it is at most ``alpha``.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between 0 and 1")
    items = [(str(name), float(value)) for name, value in pvalues.items()]
    if any(not 0 <= value <= 1 for _, value in items):
        raise ValueError("p-values must lie in [0, 1]")
    total = len(items)
    running, result = 0.0, {}
    for step, (name, value) in enumerate(sorted(items, key=lambda item: (item[1], item[0])), 1):
        running = max(running, min(1.0, (total - step + 1) * value))
        result[name] = {"p": value, "p_adjusted": running, "reject": running <= alpha, "step": step}
    return {name: result[name] for name, _ in items}


def tost(diff: np.ndarray, margin: float = MARGIN, alpha: float = ALPHA, draws: int = DRAWS,
         seed: int = SEED) -> dict[str, Any]:
    """Equivalence by two one-sided tests: the (1 - 2 alpha) bootstrap interval inside (-margin, +margin)."""
    values = _differences(diff)
    draws, alpha = _draws(draws), _probability(alpha, "alpha", 0.5)
    if not margin > 0:
        raise ValueError("the margin must be positive")
    lower, upper = _interval(_bootstrap_means(values, draws, seed), 1 - 2 * alpha)
    return {"equivalent": bool(-margin < lower and upper < margin), "lower": lower, "upper": upper,
            "confidence": 1 - 2 * alpha, "margin": float(margin), "alpha": float(alpha),
            "effect": float(values.mean()), "n": int(values.size), "draws": int(draws), "seed": int(seed)}


def mde(sd: float, n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Minimum detectable effect (z_{1-alpha/2} + z_power) * sd / sqrt(n); 2.80 SD/sqrt(n) by default."""
    if not (math.isfinite(sd) and sd >= 0) or n < 1 or not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("MDE needs sd >= 0, n >= 1 and alpha, power in (0, 1)")
    normal = NormalDist()
    return (normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)) * float(sd) / math.sqrt(n)


def _bounds(interval: Sequence[float] | None) -> tuple[float, float] | None:
    if interval is None:
        return None
    lower, upper = (float(value) if value is not None else math.nan for value in interval)
    if not (math.isfinite(lower) and math.isfinite(upper)):
        return None
    if lower > upper:
        raise ValueError("interval bounds are reversed")
    return lower, upper


def decide(ci95: tuple[float, float] | None, ci90: tuple[float, float] | None, margin: float = MARGIN,
           holm_reject: bool | None = None) -> str:
    """PREREGISTRATION §6 verdict. The rows are checked in exactly this order:

    1. L95 >= +margin and Holm               -> USEFUL_IMPROVEMENT
    2. L95 > 0 and U95 < +margin and Holm    -> POSITIVE_BELOW_USEFUL
    3. L95 > 0 and Holm                      -> IMPROVEMENT_SIZE_UNRESOLVED
    4. 90% interval inside (-margin, +margin) -> NO_USEFUL_DIFFERENCE
    5. U95 < 0                               -> WORSE
    6. otherwise                             -> INCONCLUSIVE

    "Holm" is ``holm_reject``; None (secondary, unadjusted contrasts) counts as
    True. A positive interval without Holm rejection falls through to rows
    4-6. A missing or non-finite interval gives NOT_ASSESSED.
    """
    if not margin > 0:
        raise ValueError("the margin must be positive")
    first, second = _bounds(ci95), _bounds(ci90)
    if first is None or second is None:
        return NOT_ASSESSED
    (l95, u95), (l90, u90) = first, second
    holm_ok = True if holm_reject is None else bool(holm_reject)
    if holm_ok and l95 >= margin:
        return USEFUL_IMPROVEMENT
    if holm_ok and l95 > 0 and u95 < margin:
        return POSITIVE_BELOW_USEFUL
    if holm_ok and l95 > 0:
        return IMPROVEMENT_SIZE_UNRESOLVED
    if -margin < l90 and u90 < margin:
        return NO_USEFUL_DIFFERENCE
    if u95 < 0:
        return WORSE
    return INCONCLUSIVE


def noninferiority(ci95: tuple[float, float] | None, margin: float = MARGIN) -> str:
    """Non-inferiority at -margin (PREREGISTRATION §7.3): the 95% lower bound must exceed -margin."""
    if not margin > 0:
        raise ValueError("the margin must be positive")
    bounds = _bounds(ci95)
    if bounds is None:
        return NOT_ASSESSED
    return NON_INFERIOR if bounds[0] > -margin else NON_INFERIORITY_NOT_SHOWN


def contrast(per_component_a: np.ndarray, per_component_b: np.ndarray, *, margin: float = MARGIN,
             alpha: float = ALPHA, power: float = POWER, draws: int = DRAWS, seed: int = SEED,
             holm_reject: bool | None = None) -> dict[str, Any]:
    """Paired contrast a - b over aligned per-component values (e.g. ``component_macro`` arrays).

    One bootstrap distribution gives both the (1 - alpha) and (1 - 2 alpha)
    intervals; the sign-flip p-value, MDE, TOST and the §6 verdict follow.
    Fewer than two components or non-finite values give NOT_ASSESSED.
    """
    a = np.asarray(per_component_a, dtype=np.float64)
    b = np.asarray(per_component_b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("per-component arrays must be 1-D and aligned to the same component IDs")
    draws, alpha, power = _draws(draws), _probability(alpha, "alpha", 0.5), _probability(power, "power")
    if not margin > 0:
        raise ValueError("the margin must be positive")
    record: dict[str, Any] = {"direction": "a - b", "n": int(a.size), "margin": float(margin), "alpha": float(alpha),
                              "power": float(power), "draws": int(draws), "seed": int(seed),
                              "holm_reject": holm_reject}
    if a.size < 2 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        reason = "fewer than two components" if a.size < 2 else "non-finite component values"
        return {**record, **dict.fromkeys(("mean_a", "mean_b", "effect", "sd", "se", "ci95", "ci90",
                                           "p_sign_flip", "mde", "tost_equivalent")),
                "verdict": NOT_ASSESSED, "reason": reason}
    values = _differences(a - b)
    boot = _bootstrap_means(values, draws, seed)
    ci95, ci90 = _interval(boot, 1 - alpha), _interval(boot, 1 - 2 * alpha)
    sd = float(values.std(ddof=1))
    return {**record, "mean_a": float(a.mean()), "mean_b": float(b.mean()), "effect": float(values.mean()),
            "sd": sd, "se": float(boot.std(ddof=1)), "ci95": list(ci95), "ci90": list(ci90),
            "p_sign_flip": sign_flip_pvalue(values, draws, seed), "mde": mde(sd, a.size, alpha, power),
            "tost_equivalent": bool(-margin < ci90[0] and ci90[1] < margin),
            "verdict": decide(ci95, ci90, margin, holm_reject)}


def holm_family(contrasts: Mapping[str, Mapping[str, Any]], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """Holm over one family of ``contrast`` results, re-deciding each verdict.

    Adds ``p_holm``, ``holm_reject``, ``holm_family_size`` and
    ``verdict_unadjusted``. An unassessed member keeps NOT_ASSESSED and enters
    the family with p = 1, so the family size stays as pre-registered.
    """
    pvalues = {name: 1.0 if row.get("p_sign_flip") is None else row["p_sign_flip"]
               for name, row in contrasts.items()}
    adjusted = holm(pvalues, alpha)
    family = {}
    for name, row in contrasts.items():
        margin = row.get("margin", MARGIN)
        reject = bool(adjusted[str(name)]["reject"])
        family[name] = {**row, "p_holm": adjusted[str(name)]["p_adjusted"], "holm_reject": reject,
                        "holm_family_size": len(pvalues), "holm_alpha": float(alpha),
                        "verdict_unadjusted": decide(row.get("ci95"), row.get("ci90"), margin, None),
                        "verdict": decide(row.get("ci95"), row.get("ci90"), margin, reject)}
    return family


def interaction_bootstrap(diff_a: np.ndarray, diff_b: np.ndarray, draws: int = DRAWS, seed: int = SEED,
                          confidence: float = 0.95) -> dict[str, Any]:
    """Interval for mean(diff_a) - mean(diff_b) over two disjoint component sets.

    Components are resampled independently within each set (the S1-versus-S2
    interaction of PREREGISTRATION §7.1).
    """
    a, b = _differences(diff_a), _differences(diff_b)
    draws, confidence = _draws(draws), _probability(confidence, "confidence")
    rng = np.random.default_rng(seed)
    boot = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, _CHUNK):
        count = min(_CHUNK, draws - start)
        first = a[rng.integers(0, a.size, size=(count, a.size))].mean(axis=1)
        boot[start:start + count] = first - b[rng.integers(0, b.size, size=(count, b.size))].mean(axis=1)
    lower, upper = _interval(boot, confidence)
    return {"effect": float(a.mean() - b.mean()), "lower": lower, "upper": upper, "se": float(boot.std(ddof=1)),
            "n_a": int(a.size), "n_b": int(b.size), "draws": int(draws), "seed": int(seed),
            "confidence": float(confidence)}
