"""Round 3 paired statistics over source components (prompt §4; PLAN §5; M R3.1-R3.4, R4.4).

Components are the independent units. Every procedure takes aligned per-component values (``evaluate(...)
["per_component"]["ndcg_at_10"]``) or, for subgroups, per-query values with their component IDs. Seeds and draw counts
follow the frozen convention (10,000 draws, seed 20261004); ``uol.facts_to_law.stats`` supplies the bootstrap core,
the sign-flip (component randomisation) test, Holm and the MDE.

Verdict classes (DECISION D0.6). The "adjusted lower bound" is the Bonferroni-simultaneous lower bound of the
component bootstrap: the lower end of the two-sided (1 - alpha/m) percentile interval for a family of m contrasts
(m = 1: the ordinary 95% bound). This is the conservative route that M R3.4 names, and it is never less strict than a
Holm-compatible bound. Holm-adjusted randomisation p-values are reported beside it.

* ``useful``: adjusted lower bound >= .02
* ``nonzero_below_margin``: adjusted lower bound > 0
* ``equivalent``: the unadjusted 90% interval lies inside (-.02, +.02) (TOST at alpha .05)
* ``inconclusive``: everything else
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats as sps

from uol.facts_to_law import stats as fs

SEED = fs.SEED
DRAWS = fs.DRAWS
MARGIN = fs.MARGIN
ALPHA = fs.ALPHA
Z80 = 2.80                       # z_{.975} + z_{.80}, rounded as in the prompt
CLASSES = ("useful", "nonzero_below_margin", "equivalent", "inconclusive")

holm = fs.holm
wilson_z = 1.959963984540054


def _values(diff: Sequence[float]) -> np.ndarray:
    return fs._differences(np.asarray(diff, dtype=np.float64))


def bootstrap_means(diff: Sequence[float], draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """Component bootstrap distribution of the mean paired difference (the frozen resampling stream)."""
    return fs._bootstrap_means(_values(diff), fs._draws(draws), seed)


def percentile(boot: np.ndarray, confidence: float) -> list[float]:
    return list(fs._interval(boot, confidence))


def t_interval(diff: Sequence[float], confidence: float = 0.95) -> list[float]:
    v = _values(diff)
    half = sps.t.ppf(0.5 + confidence / 2, v.size - 1) * v.std(ddof=1) / math.sqrt(v.size)
    return [float(v.mean() - half), float(v.mean() + half)]


def randomization_p(diff: Sequence[float], draws: int = DRAWS, seed: int = SEED) -> float:
    """Two-sided component randomisation (sign-flip) test of a zero mean difference."""
    return fs.sign_flip_pvalue(_values(diff), draws, seed)


def classify(adjusted_lower: float, ci90: Sequence[float], margin: float = MARGIN) -> str:
    if adjusted_lower >= margin:
        return "useful"
    if adjusted_lower > 0:
        return "nonzero_below_margin"
    if -margin < ci90[0] and ci90[1] < margin:
        return "equivalent"
    return "inconclusive"


def contrast(a: Sequence[float], b: Sequence[float], *, family_size: int = 1, draws: int = DRAWS, seed: int = SEED,
             margin: float = MARGIN, alpha: float = ALPHA) -> dict[str, Any]:
    """Paired contrast a - b over aligned per-component values.

    One bootstrap distribution gives the 95%, 90% and Bonferroni-adjusted intervals; a paired t-interval and the
    randomisation p-value follow. ``class`` uses the adjusted bound; ``class_unadjusted`` uses m = 1.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError("per-component arrays must be 1-D and aligned")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("non-finite component values")
    if family_size < 1:
        raise ValueError("family_size must be >= 1")
    v = _values(a - b)
    boot = fs._bootstrap_means(v, fs._draws(draws), seed)
    ci95, ci90 = percentile(boot, 1 - alpha), percentile(boot, 1 - 2 * alpha)
    adj = percentile(boot, 1 - alpha / family_size)
    sd = float(v.std(ddof=1))
    return {"n": int(v.size), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "delta": float(v.mean()),
            "sd": sd, "se_boot": float(boot.std(ddof=1)), "ci95": ci95, "ci90": ci90,
            "family_size": int(family_size), "ci_adjusted": adj, "adjusted_lower": adj[0],
            "t_ci95": t_interval(v, 1 - alpha), "p_randomization": randomization_p(v, draws, seed),
            "tost_equivalent": bool(-margin < ci90[0] and ci90[1] < margin),
            "class": classify(adj[0], ci90, margin), "class_unadjusted": classify(ci95[0], ci90, margin),
            "mde_improvement": mde_improvement(sd, v.size), "mde_useful": mde_useful(sd, v.size, margin),
            "draws": int(draws), "seed": int(seed), "margin": float(margin)}


def tost(diff: Sequence[float], margin: float = MARGIN, draws: int = DRAWS, seed: int = SEED) -> dict[str, Any]:
    return fs.tost(_values(diff), margin=margin, draws=draws, seed=seed)


def family(contrasts: Mapping[str, Mapping[str, Any]], alpha: float = ALPHA) -> dict[str, dict[str, Any]]:
    """Holm over the randomisation p-values of one family; adds ``p_holm`` and ``holm_reject`` to each contrast.

    The verdict classes already use the Bonferroni-adjusted bound at the family size the contrasts were run with;
    this function checks that the sizes agree.
    """
    m = len(contrasts)
    for name, c in contrasts.items():
        if c.get("family_size") != m:
            raise ValueError(f"contrast {name} was run with family_size {c.get('family_size')}, family has {m}")
    adj = holm({k: c["p_randomization"] for k, c in contrasts.items()}, alpha)
    return {k: {**c, "p_holm": adj[k]["p_adjusted"], "holm_reject": adj[k]["reject"]} for k, c in contrasts.items()}


# ----------------------------------------------------------------------------- MDEs and proportions

def mde_improvement(sd: float, n: int) -> float:
    """Smallest true effect detected with 80% power by 'lower bound > 0': 2.80 SD / sqrt(n)."""
    return Z80 * float(sd) / math.sqrt(n)


def mde_useful(sd: float, n: int, margin: float = MARGIN) -> float:
    """Smallest true effect detected with 80% power by 'lower bound >= margin': margin + 2.80 SD / sqrt(n)."""
    return margin + mde_improvement(sd, n)


def wilson(k: int, n: int, z: float = wilson_z) -> list[float]:
    if n <= 0:
        return [math.nan, math.nan]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


# ----------------------------------------------------------------------------- subgroups

def component_table(values: Sequence[float], components: Sequence[str], flags: Sequence[bool]
                    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per component: sum and count of per-query values inside and outside the subgroup."""
    vals = np.asarray(values, dtype=np.float64)
    flags = np.asarray(flags, dtype=bool)
    if not (len(vals) == len(components) == len(flags)):
        raise ValueError("values, components and flags must align")
    order = sorted(set(components))
    pos = {c: i for i, c in enumerate(order)}
    idx = np.asarray([pos[c] for c in components])
    k = len(order)
    s_in = np.bincount(idx[flags], vals[flags], k)
    n_in = np.bincount(idx[flags], minlength=k).astype(np.float64)
    s_out = np.bincount(idx[~flags], vals[~flags], k)
    n_out = np.bincount(idx[~flags], minlength=k).astype(np.float64)
    return order, s_in, n_in, s_out, n_out


def interaction(diff_per_query: Sequence[float], components: Sequence[str], flags: Sequence[bool], *,
                draws: int = DRAWS, seed: int = SEED, confidence: float = 0.95) -> dict[str, Any]:
    """Difference of differences: component-macro mean paired Δ inside the subgroup minus outside.

    Per-query paired differences are averaged within each component and side; components are resampled jointly
    (one draw serves both sides), so a component with queries on both sides keeps its pairing.
    """
    _, s_in, n_in, s_out, n_out = component_table(diff_per_query, components, flags)
    has_in, has_out = n_in > 0, n_out > 0
    if has_in.sum() < 2 or has_out.sum() < 2:
        return {"n_components_in": int(has_in.sum()), "n_components_out": int(has_out.sum()), "assessed": False}
    m_in = np.where(has_in, s_in / np.maximum(n_in, 1), 0.0)
    m_out = np.where(has_out, s_out / np.maximum(n_out, 1), 0.0)
    effect_in = float(m_in[has_in].mean())
    effect_out = float(m_out[has_out].mean())
    rng = np.random.default_rng(seed)
    k = len(m_in)
    boot = np.empty(draws)
    for start in range(0, draws, 256):
        cnt = min(256, draws - start)
        w = np.zeros((cnt, k))
        draw = rng.integers(0, k, size=(cnt, k))
        np.add.at(w, (np.repeat(np.arange(cnt), k), draw.ravel()), 1.0)
        wi, wo = w * has_in, w * has_out
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[start:start + cnt] = (wi @ m_in) / wi.sum(1) - (wo @ m_out) / wo.sum(1)
    boot = boot[np.isfinite(boot)]
    lo, hi = np.quantile(boot, [(1 - confidence) / 2, (1 + confidence) / 2])
    return {"assessed": True, "delta_in": effect_in, "delta_out": effect_out, "interaction": effect_in - effect_out,
            "ci": [float(lo), float(hi)], "confidence": confidence, "n_components_in": int(has_in.sum()),
            "n_components_out": int(has_out.sum()), "draws_used": int(boot.size), "seed": int(seed)}


def subgroup_contrast(arm_per_query: Sequence[float], ref_per_query: Sequence[float], components: Sequence[str],
                      flags: Sequence[bool], **kw: Any) -> dict[str, Any]:
    """Arm - reference inside the subgroup (component-macro contrast) plus the interaction test."""
    d = np.asarray(arm_per_query, dtype=np.float64) - np.asarray(ref_per_query, dtype=np.float64)
    flags = np.asarray(flags, dtype=bool)
    order, s_in, n_in, *_ = component_table(arm_per_query, components, flags)
    _, r_in, _, *_ = component_table(ref_per_query, components, flags)
    keep = n_in > 0
    out: dict[str, Any] = {"n_queries_in": int(flags.sum())}
    if keep.sum() >= 2:
        out["within"] = contrast((s_in / np.maximum(n_in, 1))[keep], (r_in / np.maximum(n_in, 1))[keep], **kw)
    out["interaction"] = interaction(d, components, flags)
    return out


# ----------------------------------------------------------------------------- tuning-budget curve

def expected_max_curve(scores: Sequence[float]) -> list[float]:
    """Expected best dev score after n random configurations (with replacement), n = 1..N (Dodge et al. 2019)."""
    v = np.sort(np.asarray(scores, dtype=np.float64))
    n_cfg = len(v)
    i = np.arange(1, n_cfg + 1)
    return [float(np.sum(v * ((i / n_cfg) ** n - ((i - 1) / n_cfg) ** n))) for n in range(1, n_cfg + 1)]
