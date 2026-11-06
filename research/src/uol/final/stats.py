"""Paired group bootstrap contrasts from the results store (PLAN_DRAFT §7), reusing ``round3.stats``.

``round3.stats.contrast`` gives the frozen procedure: 10,000 component-bootstrap draws, seed 20261004, 95 / 90 %
percentile intervals, the Bonferroni-adjusted lower bound for a family of m contrasts, the sign-flip randomisation
p-value, and the outcome class (*useful*: adjusted lower bound >= .02; *nonzero_below_margin*; *equivalent*: 90 %
interval inside +-.02; *inconclusive*). ``family`` adds Holm-adjusted p-values. Per-case values are averaged within
components (groups) first and the two systems are aligned on the components both scored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import harness, measures


def _cases(store: Path, system: str, population: str):
    """Per-case results; a sample population of DEV (DEV-A300) reads a system scored on DEV restricted to its rows
    (the same cases, the same fold models)."""
    p = Path(store) / "runs" / system / population / "per_case.parquet"
    if not p.exists() and population in SUBSAMPLES:
        return harness.load_cases(store, system, "DEV")
    return harness.load_cases(store, system, population)


SUBSAMPLES = ("DEV-A300",)


def aligned(store: Path, a: str, b: str, population: str, measure: str = "ndcg_at_10") -> tuple[list[str], np.ndarray, np.ndarray]:
    fa, fb = _cases(store, a, population), _cases(store, b, population)
    common = sorted(set(fa["query_id"]) & set(fb["query_id"]))
    if not common:
        raise ValueError(f"{a} and {b} share no case on {population}")
    fa = fa.set_index("query_id").loc[common]
    fb = fb.set_index("query_id").loc[common]
    ia, va = measures.per_component(fa[measure].to_numpy(), fa["component_id"].to_numpy())
    ib, vb = measures.per_component(fb[measure].to_numpy(), fb["component_id"].to_numpy())
    keep = sorted(set(ia) & set(ib))
    pa, pb = dict(zip(ia, va)), dict(zip(ib, vb))
    return keep, np.asarray([pa[k] for k in keep]), np.asarray([pb[k] for k in keep])


def contrast(store: Path, a: str, b: str, population: str = "DEV", measure: str = "ndcg_at_10",
             family_size: int = 1) -> dict[str, Any]:
    from uol.round3 import stats as R3
    comps, va, vb = aligned(store, a, b, population, measure)
    out = R3.contrast(va, vb, family_size=family_size)
    return {"a": a, "b": b, "population": population, "measure": measure, "components": len(comps), **out}


def family(store: Path, pairs: Sequence[tuple[str, str]], population: str = "DEV", measure: str = "ndcg_at_10",
           name: str = "family") -> dict[str, Any]:
    from uol.round3 import stats as R3
    m = len(pairs)
    cons = {f"{a} - {b}": contrast(store, a, b, population, measure, family_size=m) for a, b in pairs}
    return {"family": name, "size": m, "contrasts": R3.family(cons)}


FAMILIES = {
    # RUN_PLAN v2 §4-§5; ids are result names in the store. D77 (22 Oct): the rankers are the frozen-source runs (the
    # "@rebuilt" twins stay behind the failed g1_rebuild gate as reproducibility evidence). D78: F2 uses the dense-only
    # ladder (S0 + the encoder's dense column) for every encoder; the with-sources rows are extra rows, not contrasted
    "F1": [("S_full", "S0")],
    "F2": "S0+X (ladder_definition dense_only) - S0 for every ladder encoder X",
    "F3": [("bge_m3_legal", "bge_m3_base"), ("euler_legal", "qwen3_8b_int8_base"), ("euler_legal", "qwen3_8b"), ("bge_m3_ft", "bge_m3"),
           ("ft_e5_q", "e5_small"), ("qwen8b_adapter", "qwen3_8b"),
           ("qwen3_0p6b_ft", "qwen3_0p6b")],                                # D89
    "F4": "hybrid - best of its two parts, for every L* with headings + X fusion",
    "F5": "each combination method - HGB on the same features (S0 and S_full)",
    # RUN_PLAN v2 E9 / §5: each second stage against the ranking it reorders, and the cheap ranker with the LLM against
    # the full ranker. It was in the plan's family table but never wired (found 23 Oct 2026; reporting only)
    "F6": [("llm_rerank_S_full_ext", "S_full_ext"), ("llm_rerank_S0", "S0"), ("ce_trained", "S_full_ext"),
           ("ce_trained_alone", "S_full_ext"), ("rerank_bge", "S_full_ext"), ("rerank_bge_alone", "S_full_ext"),
           ("rerank_qwen3_4b", "S_full_ext"), ("rerank_qwen3_4b_alone", "S_full_ext"), ("llm_rerank_S0", "S_full")],
    # D101: the external second stages on DEV-A300 against the ranking they reorder, and Claude against Luna
    # (llm_rerank_S_full_ext read on the same 300 cases); its own family because its population differs
    "F6x": [("claude_rerank_S_full_ext", "S_full_ext"), ("jev_S_full_ext", "S_full_ext"),
            ("jev_rrf_S_full_ext", "S_full_ext"), ("claude_rerank_S_full_ext", "llm_rerank_S_full_ext")],
    # D93: F8, the confirmatory C2 family (ANALYSIS_PLAN_C §3): what each channel and Qwen3-8B add uniquely
    "F8": [("S_full", "S_full_minus_D"), ("S_full", "S_full_minus_M"), ("S_full_ext", "S_full_ext_minus_qwen3_8b")],
    # D79: what late interaction adds inside the full ranker (a described contrast, its own family of one)
    "F7": [("S_full_ext", "S_full_ext_minus_colbert")],
}


def family_pairs(store: Path) -> dict[str, list[tuple[str, str]]]:
    """Concrete contrast pairs of F1-F7 among the results present in the store."""
    from uol.final import registry
    runs = {p.parent.parent.name for p in Path(store).glob("runs/*/DEV/summary.json")}   # runs/<system>/DEV
    have = lambda x: x in runs
    fam: dict[str, list[tuple[str, str]]] = {"F1": [p for p in FAMILIES["F1"] if have(p[0]) and have(p[1])]}
    fam["F2"] = [(f, "S0") for f in ladder_dense_only() if have(f) and have("S0")]
    fam["F3"] = [p for p in FAMILIES["F3"] if have(p[0]) and have(p[1])]
    fam["F6"] = [p for p in FAMILIES["F6"] if have(p[0]) and have(p[1])]
    fam["F7"] = [p for p in FAMILIES["F7"] if have(p[0]) and have(p[1])]
    fam["F8"] = [p for p in FAMILIES["F8"] if have(p[0]) and have(p[1])]
    a300 = {p.parent.parent.name for p in Path(store).glob("runs/*/DEV-A300/summary.json")}
    fam["F6x"] = [p for p in FAMILIES["F6x"] if p[0] in a300 and have(p[1])]
    f4 = []
    for x in registry.FUSION_PARTNERS:
        for m in ("rrf", "linear"):
            h = f"{m}__lstar_headings__{x}"
            if have(h) and have("lstar_headings") and have(x):
                best = max(("lstar_headings", x), key=lambda s: _summary(store, s).get("ndcg_at_10", -1))
                f4.append((h, best))
    fam["F4"] = f4
    f5 = []
    for v, base in (("noemb", "ltr_noemb_refit"), ("emb", "ltr_emb_refit")):          # frozen-source runs (D77)
        for other in (f"ltr_{v}_linear", f"ltr_{v}_lambdamart", f"rrf_all_{v}"):
            if have(other) and have(base):
                f5.append((other, base))
    fam["F5"] = f5
    return fam


def _summary(store: Path, system: str, population: str = "DEV") -> dict:
    import json
    p = Path(store) / "runs" / system / population / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def all_families(store: Path) -> dict[str, Any]:
    from uol.final import harness
    out = {}
    for name, pairs in family_pairs(store).items():
        if pairs:
            rec = family(store, pairs, "DEV-A300" if name == "F6x" else "DEV", name=name)
            harness.write_json(Path(store) / "contrasts" / f"{name}.json", rec)
            out[name] = {"size": rec["size"]}
    return out


def mde(store: Path, system: str, reference: str, population: str = "DEV") -> dict[str, float]:
    """Smallest detectable difference for a population (80 % power), from the paired component SD."""
    from uol.round3 import stats as R3
    _, va, vb = aligned(store, system, reference, population)
    sd = float(np.std(va - vb, ddof=1))
    return {"sd": sd, "n": len(va), "mde_improvement": R3.mde_improvement(sd, len(va)),
            "mde_useful": R3.mde_useful(sd, len(va))}


def ladder_dense_only() -> list[str]:
    """The E5 ladder rows of F2 (D78): for every ladder encoder its dense-only S0 + X system (the ``_dense`` twin for the
    frozen encoders that also have a with-sources row)."""
    from uol.final import registry
    S = registry.systems()
    out = []
    for x in registry.LADDER_ENCODERS:
        for sid in (f"S0+{x}_dense", f"S0+{x}"):
            if S.get(sid, {}).get("ladder_definition") == "dense_only":
                out.append(sid)
                break
    return out
