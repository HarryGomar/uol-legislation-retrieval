"""D61 figures (FIGURE_PLAN.md, 19 Oct 2026): hit rate at k, real versus random controls, what the LLM alone names,
and what else was tried. One finding per figure; development set (no test-set values exist for these analyses);
printed numbers are EVIDENCE.md values, asserted against the saved summaries where those are machine-readable.

    PYTHONPATH=src py -3.12 -m uol.thesis_figures.d61_figures

Sources: eval_lenses cutoffs.json (hit rate at k), Round 3 arms RESULT.json null draws (S2, S3, M1), EVIDENCE.md QK
and D55 rows (the LLM alone), gap tests and Round 3 arms (alternatives). Internal identifiers: L_star, dense_qwen8b,
LTR_noemb_hgb, LTR_emb_hgb, LLM alone (D55); S2, S3, M1; T3, T4, T6, X4 gap tests; arms of arms.ARMS.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from uol.thesis_figures import arms as A
from uol.thesis_figures import common as C

CUTOFFS = f"{C.EL}/cutoffs.json"
EV = "research/thesis_direction_20261015/EVIDENCE.md"


# ------------------------------------------------------------------------------------------------ hit rate at k
HIT_SYSTEMS = [("LTR_emb_hgb", "Ranker with embeddings", C.BLUE, "-", 1.5),
               ("LTR_noemb_hgb", "Ranker without embeddings", C.GREYS[1], "-", 1.1),
               ("dense_qwen8b", "Qwen3-Embedding-8B alone", C.BLUE_LIGHT, "-", 1.1),
               ("L_star", "Tuned word matching", C.GREYS[2], (0, (1.2, 1.2)), 1.1),
               ("LLM alone (list of about 15; capped above k = 15)", "LLM alone", C.VERM, (0, (3.5, 1.5)), 1.1)]
EV_HIT = {"L_star": "0.129/0.230/0.273/0.341/0.387/0.418/0.472/0.535/0.646/0.724",
          "dense_qwen8b": "0.198/0.334/0.402/0.479/0.529/0.565/0.614/0.672/0.740/0.808",
          "LTR_noemb_hgb": "0.330/0.469/0.546/0.641/0.682/0.709/0.746/0.802/0.864/0.911",
          "LTR_emb_hgb": "0.373/0.541/0.614/0.713/0.743/0.772/0.820/0.858/0.908/0.946",
          "LLM alone (list of about 15; capped above k = 15)": "0.273/0.407/0.494/0.595/0.651/0.671"}


def hit_at_k() -> C.Item:
    C.style()
    d = C.load_json(CUTOFFS)
    ks = [int(k) for k in d["ks"]]
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, 2.75))
    ax = fig.add_axes([0.085, 0.15, 0.6, 0.8])
    C.hgrid(ax, (.25, .5, .75, 1))
    ends = []
    for key, lab, col, ls, lw in HIT_SYSTEMS:
        t = d["table"][key]
        kk = [k for k in ks if f"hit@{k}" in t and t[f"hit@{k}"] is not None]
        if key.startswith("LLM"):
            kk = [k for k in kk if k <= 15]
        v = [t[f"hit@{k}"] for k in kk]
        ev = [float(x) for x in EV_HIT[key].split("/")]
        assert all(abs(a - b) < 0.0006 for a, b in zip(v, ev)), (key, v, ev)
        ax.plot(kk, v, color=col, ls=ls, lw=lw, marker="o", ms=2.8, mec=C.SURFACE, mew=0.4, zorder=3)
        ends.append([v[-1], kk[-1], lab, col])
    e = [float(x) for x in EV_HIT["LTR_emb_hgb"].split("/")]
    for k, y, txt, off in ((1, e[0], "0.373", (-2, 7)), (10, e[3], "0.713", (-4, 7)), (200, e[-1], "0.946", (0, 7))):
        ax.annotate(txt, (k, y), textcoords="offset points", xytext=off, ha="center", fontsize=8, color=C.BLUE)
    # direct labels: four at the right end (k = 200), the LLM alone at its own end (k = 15)
    right = sorted([x for x in ends if x[1] == ks[-1]], key=lambda x: -x[0])
    for i in range(1, len(right)):
        right[i][0] = min(right[i][0], right[i - 1][0] - 0.07)
    for y, k, lab, col in right:
        ax.text(ks[-1] * 1.18, y, lab, fontsize=8, color=C.INK if col == C.BLUE else C.INK2, va="center", ha="left",
                clip_on=False)
    for y, k, lab, col in ends:
        if k != ks[-1]:
            ax.annotate(lab, (k, y), textcoords="offset points", xytext=(6, -1), fontsize=8, color=col, ha="left",
                        va="center")
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(0.85, ks[-1] * 1.12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, .25, .5, .75, 1])
    C.strip0(ax, "y")
    ax.set_xlabel("number of results read, k (log scale)", fontsize=8)
    ax.set_ylabel("share of cases with a cited article", fontsize=8)
    item = C.Item(
        id="D61-8", stem="hit_at_k", target="6.2 (FIGURE_PLAN item 8; Ch6 TODO fig:hit_at_k)",
        title="How far down a reader must go",
        caption=("Share of development cases with a cited article among the first k results, for k from 1 to 200. "
                 "The ranker with embeddings finds one in the first result for 37% of cases, in the first ten for "
                 "71% and in the first 200 for 95%."),
        label="dev, descriptive (EVIDENCE: measures at several cut-offs)", sources=[CUTOFFS, EV],
        internal="cutoffs.json table hit@k; L_star, dense_qwen8b, LTR_noemb_hgb, LTR_emb_hgb, LLM alone (D55)",
        notes=["The LLM alone names about 15 countable articles, so its curve stops at k = 15."])
    return C.save(fig, item)


# ------------------------------------------------------------------------------------------------ real vs random
NULL_ROWS = [("S2", "Statute-structure features",
              "150 random statute graphs", ("chapter_perm", "order_perm", "xref_rewire"), "−0.002"),
             ("S3", "Co-citation features", "100 random citation graphs", ("curveball",), "−0.001"),
             ("M1", "Similar-case citations from older tesis", "50 shuffles within each year", ("shuffle_year",),
              "+0.020")]


def null_controls() -> C.Item:
    C.style()
    rng = np.random.default_rng(7)               # jitter only (vertical position of the grey dots)
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, 2.15))
    ax = fig.add_axes([0.37, 0.2, 0.61, 0.76])
    ax.axvspan(-0.02, 0.02, color=C.FILL, zorder=0, lw=0)
    ax.axvline(0, color=C.MUTED, lw=0.6, zorder=1)
    ys = [2, 1, 0]
    for (arm, name, what, keys, real_txt), y in zip(NULL_ROWS, ys):
        r = C.load_json(f"{C.R3_ARMS}/{arm}/RESULT.json")
        draws = np.concatenate([np.asarray(r["nulls"][k]["deltas"], float) for k in keys])
        real = float(r["dev"]["vs_EMB"]["P_spec"]["delta"])
        assert C.f3(real, True) == real_txt, (arm, real)
        ax.scatter(draws, y + rng.uniform(-0.17, 0.17, len(draws)), s=6, color=C.GREYS[3], lw=0, zorder=2)
        col = C.BLUE if arm == "M1" else C.INK
        ax.plot(real, y, "D", ms=6, color=col, mec=C.SURFACE, mew=0.6, zorder=4)
        ax.annotate(f"real {real_txt}", (real, y), textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=8, color=col)
        ax.text(-0.012, y + 0.12, name.replace("tesis", "$\\it{tesis}$"), transform=ax.get_yaxis_transform(),
                fontsize=8, color=C.INK, ha="right", va="center")
        ax.text(-0.012, y - 0.22, what, transform=ax.get_yaxis_transform(), fontsize=8, color=C.INK2,
                ha="right", va="center")
    ax.text(0.0195, 2.62, "within ±0.02", fontsize=8, color=C.INK2, ha="right", va="center")
    ax.set_ylim(-0.55, 2.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.022, 0.026)
    ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
    ax.set_xticklabels(["−0.02", "−0.01", "0", "+0.01", "+0.02"])
    ax.set_xlabel("change in nDCG@10, development set", fontsize=8)
    item = C.Item(
        id="D61-11", stem="null_controls", target="6.4 (FIGURE_PLAN item 11; Ch6 TODO fig:null_controls)",
        title="Real signals against random controls",
        caption=("The change in nDCG@10 from three additions on the development set (diamonds), each beside the "
                 "same addition built from randomised data (grey). Statute structure and co-citation score inside "
                 "their random controls; similar-case citations from older \\emph{tesis} lie far outside "
                 "theirs."),
        label="dev, descriptive (EVIDENCE: S2/S3 against nulls; M1 beats 50 shuffles)",
        sources=[f"{C.R3_ARMS}/{a}/RESULT.json" for a, *_ in NULL_ROWS], internal="S2, S3, M1 null deltas",
        notes=["S2 pools its three null families (50 each); its rank among each family is 15, 20 and 27 of 51 "
               "(EVIDENCE); S3 ranks 86 of 101; M1 ranks first of 51."])
    return C.save(fig, item)


# ------------------------------------------------------------------------------------------------ the LLM alone
NAMED = [("cited by the court", 25.0, C.BLUE), ("next to a cited article or in its section", 16.0, C.RAMP[2]),
         ("elsewhere in a cited law", 23.0, C.RAMP[3]), ("another federal law", 27.0, C.NEUTRAL),
         ("state law", 4.4, C.GREYS[3]), ("repealed or transitory", 1.3, C.GREYS[2]),
         ("does not exist", 2.4, C.VERM)]
SCORES = [("Tuned word matching", .184, C.GREYS[2]), ("Qwen3-Embedding-8B alone", .274, C.BLUE_LIGHT),
          ("LLM alone", .366, C.VERM), ("Ranker without embeddings", .408, C.GREYS[1]),
          ("Ranker with embeddings", .472, C.BLUE)]


def llm_alone() -> C.Item:
    C.style()
    s = C.load_json("experiments/runs/llm_alone_20261019/SCORE.json")
    flat = C.json.dumps(s)
    for _, v, _ in SCORES:                       # every printed score appears in the saved summary (rounded)
        assert any(f"{x:0.3f}" == f"{v:0.3f}" for x in _floats(s)), v
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, 3.55))
    axa = fig.add_axes([0.42, 0.47, 0.56, 0.45])
    ya = list(range(len(NAMED)))[::-1]
    for (lab, v, col), y in zip(NAMED, ya):
        axa.barh(y, v, height=0.62, color=col, lw=0)
        axa.text(v + 0.6, y, f"{v:0.0f}%" if v >= 10 else f"{v:0.1f}%", va="center", fontsize=8, color=C.INK)
    axa.set_yticks(ya)
    axa.set_yticklabels([lab for lab, _, _ in NAMED], fontsize=8, color=C.INK)
    axa.tick_params(axis="y", length=0, pad=4)
    axa.spines["left"].set_visible(False)
    axa.set_xlim(0, 32)
    axa.set_xticks([0, 10, 20, 30])
    axa.set_xticklabels(["0", "10%", "20%", "30%"])
    C.panel(axa, "a", "the 618 articles it named for 200 development cases", x=-0.66, pad=3)
    axb = fig.add_axes([0.42, 0.06, 0.56, 0.29])
    ys = list(range(len(SCORES)))[::-1]
    for (lab, v, col), y in zip(SCORES, ys):
        axb.barh(y, v, height=0.62, color=col, lw=0)
        axb.text(v + 0.006, y, C.f3(v), va="center", fontsize=8, color=C.INK)
    axb.set_yticks(ys)
    axb.set_yticklabels([lab for lab, _, _ in SCORES], fontsize=8, color=C.INK)
    axb.tick_params(axis="y", length=0, pad=4)
    axb.spines["left"].set_visible(False)
    axb.set_xlim(0, 0.55)
    axb.set_xticks([0, .1, .2, .3, .4, .5])
    C.strip0(axb, "x", nd=1)
    C.panel(axb, "b", "nDCG@10 on all 1,218 development cases", x=-0.66, pad=3)
    item = C.Item(
        id="D61-13", stem="llm_alone", target="6.5.3 (FIGURE_PLAN item 13; Ch6 TODO fig:llm_alone)",
        title="What an LLM knows without retrieval",
        caption=("What an LLM names from the facts alone, with no retrieval, on development cases: (a) the kind of "
                 "article each of its answers is, and (b) its score against the retrieval systems. A quarter of "
                 "its articles are cited and most of the rest are near the right ones; it beats word matching but "
                 "not the learned rankers."),
        label="dev, descriptive (EVIDENCE: QK categories; LLM alone full run D55)",
        sources=[EV, "experiments/runs/llm_alone_20261019/SCORE.json", f"{C.EL}/qk_ndcg.json"],
        internal="QK probe (618 named articles, 200 pre-cutoff dev rows); D55 llm_alone_dev",
        notes=["Panel a percentages are EVIDENCE.md's QK row (they sum to 99.1%, as printed there); panel b is the "
               "D55 full run. All cases predate the model's training cutoff (optimistic for the LLM)."])
    del flat
    return C.save(fig, item)


def _floats(o):
    if isinstance(o, dict):
        for v in o.values():
            yield from _floats(v)
    elif isinstance(o, list):
        for v in o:
            yield from _floats(v)
    elif isinstance(o, float):
        yield o


# ------------------------------------------------------------------------------------------------ alternatives
ALT = [  # (group, name, delta, low, high, EVIDENCE row)
    ("Tuned word matching (0.184)", "Wider BM25 settings and weighted title", "−0.000", "−0.002", "+0.002", "T3"),
    ("Tuned word matching (0.184)", "Pseudo-relevance feedback", "−0.009", "−0.015", "−0.004", "X4"),
    ("Tuned word matching (0.184)", "Article headings added to the text", "+0.014", "+0.008", "+0.019", "T4"),
    ("Ranker with embeddings (0.472)", "Listwise training objective", "+0.002", "−0.007", "+0.012",
     "T6"),
    ("Ranker with embeddings (0.472)", "Similar-case citations from older tesis", *A.EV_DEV["M1"],
     "M1"),
    ("Ranker with embeddings (0.472)", "Statute-structure features", *A.EV_DEV["S2"], "S2"),
    ("Ranker with embeddings (0.472)", "Co-citation features", *A.EV_DEV["S3"], "S3"),
    ("Ranker with embeddings (0.472)", "Article vectors smoothed over the statute",
     *A.EV_DEV["E3_graph"], "E3"),
    ("Ranker with embeddings (0.472)", "Adapter on Qwen3-Embedding-8B", *A.EV_DEV["E2_adapter"],
     "E2"),
    ("Ranker with embeddings (0.472)", "Cross-encoder, not trained", *A.EV_DEV["R1"], "R1"),
    ("Ranker with embeddings (0.472)", "Cross-encoder trained on citations",
     *A.EV_DEV["E4_R2"], "R2"),
    ("Ranker with embeddings (0.472)", "LLM restates the facts as legal issues",
     *A.EV_DEV["L2"], "L2"),
    ("Ranker with embeddings (0.472)", "LLM writes fact patterns", "+0.001", "−0.004", "+0.007", "L4"),  # leak-free (D62; EVIDENCE "L4 without the development leak")
    ("Ranker with embeddings (0.472)", "LLM reranks the top 30", *A.EV_DEV["L1"], "L1"),
]
GAP_FILES = {"T3": "experiments/runs/gap_tests_20261019/t3_bm25_grid/result.json",
             "T4": "experiments/runs/gap_tests_20261019/t4_text_variants/result.json",
             "T6": "experiments/runs/gap_tests_20261019/t6_listwise/result.json"}


def alternatives() -> C.Item:
    C.style()
    rows, y, prev, heads = [], 0.0, None, {}
    for g, name, dl, lo, hi, ref in ALT:
        if g != prev:
            y -= 1.1 if prev is not None else 0.6
            heads[g] = y + 0.75
            prev = g
        rows.append((y, name, A._num(dl), A._num(lo), A._num(hi), dl))
        y -= 1
    fig = plt.figure(figsize=(C.TEXTWIDTH_IN, 4.1))
    ax = fig.add_axes([0.41, 0.09, 0.57, 0.89])
    ax.axvspan(-0.02, 0.02, color=C.FILL, zorder=0, lw=0)
    ax.axvline(0, color=C.MUTED, lw=0.6, zorder=1)
    for yy, name, d, lo, hi, txt in rows:
        helps = lo > 0                       # the 95% interval lies above zero
        col = C.BLUE if helps else C.INK2
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot(d, yy, "o", ms=4.2, color=col, mec=C.SURFACE, mew=0.5, zorder=3)
        ax.text(hi + 0.002, yy, txt, fontsize=7, color=col, va="center", ha="left")
        ax.text(-0.012, yy, name.replace("tesis", "$\\it{tesis}$"), transform=ax.get_yaxis_transform(), fontsize=8,
                color=C.INK, ha="right", va="center")
    for g, yh in heads.items():
        ax.text(-0.012, yh, g, transform=ax.get_yaxis_transform(), fontsize=8, color=C.INK2, style="italic",
                ha="right", va="center")
    ax.text(0.0195, heads[ALT[0][0]] + 0.05, "within ±0.02", fontsize=8, color=C.INK2, ha="right", va="center")
    ax.set_ylim(y + 0.4, heads[ALT[0][0]] + 0.5)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.03, 0.07)
    ax.set_xticks([-0.02, 0, 0.02, 0.04, 0.06])
    ax.set_xticklabels(["−0.02", "0", "+0.02", "+0.04", "+0.06"])
    ax.set_xlabel("change in nDCG@10, development set (95% interval)", fontsize=8)
    item = C.Item(
        id="D61-15", stem="alternatives", target="6.7.2 (FIGURE_PLAN item 15; Ch6 TODO fig:alternatives)",
        title="What else was tried",
        caption=("Every alternative tried on the development set, as the change in nDCG@10 it made to tuned word "
                 "matching or to the ranker with embeddings, with its 95% interval; the grey band marks changes "
                 "within ±0.02. Nearly every change stays inside the band; only LLM reranking clearly exceeds it, and "
                 "article headings are the one change that helps word matching."),
        label="dev (EVIDENCE: gap tests T3, T4, T6, X4; Round 3 arms)",
        sources=[EV] + list(GAP_FILES.values()) + [f"{C.R3_ARMS}/{a}/RESULT.json" for a in
                                                   ("M1", "S2", "S3", "E3_graph", "E2_adapter", "R1", "E4_R2", "L2",
                                                    "L4", "L1")],
        internal="T3, X4, T4 heading path, T6 listwise (with embeddings), M1, S2, S3, E3, E2, R1, E4_R2, L2, L4, L1",
        notes=["Blue: the 95% interval lies above zero. The LLM stage as a whole (about +0.079 without the leak) is not "
               "drawn: it was assembled after the development results. Fact patterns: leak-free value (D62)."])
    return C.save(fig, item)


def build_all() -> list[C.Item]:
    return [hit_at_k(), null_controls(), llm_alone(), alternatives()]


if __name__ == "__main__":
    for it in build_all():
        print(it.outputs)
