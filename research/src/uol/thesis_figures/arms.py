"""The Round 3 add-ons in plain English, with their development and test-set contrasts against the learned ranker
with embeddings (shared by F11, T3 and T9).

Development values come from experiments/runs/round3_best_case_20261013/arms/<arm>/RESULT.json (dev.vs_EMB.P_spec; the
LLM stage from contrasts.C_hybrid_vs_EMB); test-set values only from the phase-7 contrast files (summaries). Printed
numbers are EVIDENCE.md's strings (D52 item 5): each computed value is checked to agree with its EVIDENCE row within
rounding (the summaries carry more digits, so three interval ends differ by 0.001 when re-rounded: L2 and S3 on the
development set, L1 on the test set), and the EVIDENCE string is printed.

Development rule (Round 3 PLAN.md): an addition joined the final system only with a gain of at least 0.03 whose 95%
interval excludes zero. Test-set readings (Chapter 3, 3.6.3) use the family-widened class of the contrast file.
"""
from __future__ import annotations

from uol.thesis_figures import common as C

RULE = 0.03
# (arm id, plain name, group, note, scored on the test set?)
ARMS = [
    ("M1", "Similar-case citations from older tesis", "Past citations", "", True),
    ("L2", "LLM restates the facts as legal issues", "LLM additions", "", True),
    ("L2_ctrl_clause", "Control: facts split into clauses, no LLM", "LLM additions", "control", False),
    ("L2_ctrl_random", "Control: random sub-queries, no LLM", "LLM additions", "control", False),
    ("L4", "LLM writes fact patterns for uncited articles", "LLM additions", "", True),
    ("L1", "LLM reranks the top 30", "LLM additions", "", True),
    ("R1", "Cross-encoder, not trained (8B)", "Rerankers", "development set only", False),
    ("E4_R2", "Cross-encoder trained on citations", "Rerankers", "one-seed pilot; development set only", False),
    ("S2", "Statute-structure features", "Structure", "", True),
    ("S3", "Co-citation features", "Structure", "", True),
    ("E3_graph", "Article vectors smoothed over the statute", "Structure", "", True),
    ("E2_adapter", "Adapter on Qwen3-Embedding-8B", "Refined embeddings", "", True),
    ("C_hybrid_ltr", "LLM stage, first step only", "LLM stage", "assembled after the development results", True),
    ("C_hybrid", "LLM stage, both steps", "LLM stage", "assembled after the development results", True),
    ("X0_noise", "Control: a column of random noise", "Control", "control", True),
]
LLM_TEST_ON_CUTOFF = {"L1", "L2", "L4", "C_hybrid"}           # test cases after the model's cutoff (145 groups)

# EVIDENCE.md strings: (delta, low, high). Development rows from "LLM abstractions and rerankers", "Structure",
# "Representation experiments", "What each representation contributes" and "Chapter 6 additions"; test rows from
# "Round 3 lockbox batch".
EV_DEV = {"M1": ("+0.020", "+0.010", "+0.030"), "L2": ("+0.020", "+0.011", "+0.030"), "L4": ("+0.011", "+0.004", "+0.017"),
          "L1": ("+0.046", "+0.036", "+0.056"), "C_hybrid": ("+0.084", "+0.069", "+0.099"),
          "S2": ("−0.002", "−0.007", "+0.003"), "S3": ("−0.001", "−0.007", "+0.006"),
          "E2_adapter": ("−0.004", "−0.011", "+0.004"), "E3_graph": ("+0.004", "−0.003", "+0.011"),
          "R1": ("+0.013", "+0.007", "+0.019"), "E4_R2": ("+0.010", "+0.005", "+0.015"),
          "L2_ctrl_clause": ("−0.004", "−0.010", "+0.002"), "L2_ctrl_random": ("−0.004", "−0.009", "+0.001")}
EV_TEST = {"M1": ("+0.029", "+0.010", "+0.048"), "E2_adapter": ("+0.025", "+0.011", "+0.040"),
           "S2": ("+0.005", "−0.006", "+0.017"), "S3": ("+0.006", "−0.004", "+0.016"), "E3_graph": ("+0.005", "−0.008", "+0.019"),
           "X0_noise": ("+0.002", "−0.007", "+0.012"), "C_hybrid_ltr": ("+0.011", "−0.015", "+0.036"),
           "L1": ("+0.051", "+0.024", "+0.080"), "L2": ("+0.017", "−0.003", "+0.037"), "L4": ("−0.017", "−0.038", "+0.002"),
           "C_hybrid": ("+0.075", "+0.035", "+0.116")}
READING = {"useful": "a gain of at least 0.02", "nonzero_below_margin": "a gain below 0.02",
           "equivalent": "no meaningful difference", "inconclusive": "too uncertain to call", "worse": "a loss"}


def _num(s: str) -> float:
    return float(s.replace("−", "-").replace("+", ""))


def _check(arm: str, ev: tuple[str, str, str], delta: float, ci: list[float], where: str) -> None:
    for e, v in zip(ev, [delta, *ci]):
        assert abs(_num(e) - v) <= 0.0011, (arm, where, e, v)


def fmt(ev: tuple[str, str, str], tex_minus: bool = False) -> tuple[str, str]:
    """(delta, [low, high]) strings; with tex_minus the minus sign is written for LaTeX text mode."""
    m = (lambda s: s.replace("−", "$-$")) if tex_minus else (lambda s: s)
    return m(ev[0]), f"[{m(ev[1])}, {m(ev[2])}]"


def dev_rule_met(delta: float, ci: list[float]) -> bool:
    return delta >= RULE and ci[0] > 0


def load(arm: str) -> dict:
    r = C.load_json(f"{C.R3_ARMS}/{arm}/RESULT.json")
    if arm == "C_hybrid":
        c = r["contrasts"]["C_hybrid_vs_EMB"]
        d = {"delta": c["delta"], "ci95": c["ci95"], "class": c.get("class"), "arm_ndcg": c["mean_a"]}
    else:
        c = r["dev"]["vs_EMB"]["P_spec"]
        d = {"delta": c["delta"], "ci95": c["ci95"], "class": c.get("class"), "arm_ndcg": c["arm_ndcg_at_10"]}
    seeds = r.get("seeds")
    sm = seeds.get("seed_mean_vs_EMB_P_spec") if isinstance(seeds, dict) else None
    if sm:
        d["seed_mean"] = {"delta": sm["delta"], "ci95": sm["ci95"]}
    if arm in EV_DEV:
        _check(arm, EV_DEV[arm], d["delta"], d["ci95"], "development")
        d["ev"] = EV_DEV[arm]
    else:                                   # no EVIDENCE interval (stage-1 dev +0.050 and the noise column): computed
        d["ev"] = (C.f3(d["delta"], True), C.f3(d["ci95"][0], True), C.f3(d["ci95"][1], True))
    d["rule_met"] = dev_rule_met(d["delta"], d["ci95"])
    d["lockbox"] = C.phase7_contrast(arm)
    if d["lockbox"]:
        L = d["lockbox"]
        _check(arm, EV_TEST[arm], L["delta"], L["ci95"], "test")
        L["ev"] = EV_TEST[arm]
        L["reading"] = READING.get(L["class"], L["class"])
    d["source"] = f"{C.R3_ARMS}/{arm}/RESULT.json"
    return d


def sources() -> list[str]:
    return [f"{C.R3_ARMS}/{a}/RESULT.json" for a, *_ in ARMS]
