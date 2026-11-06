"""Chapter 6 figures of the rebuild (PLAN section 6, 2 Nov 2026): the system is the combination of the model's list and
the ranker's list, read again by the model (results.json ``system.dev_system``).

Every value is read from saved results, never typed:

* ``research/programs/final_results_20261102/results.json`` (``scripts/final_results.py``): the record of the rebuild;
* the common footing store ``experiments/runs/footing_20261101/dev/store/runs`` (summaries and per-case files) and its
  contrasts ``.../dev/contrasts_100k/*.json`` (100,000 draws), ``chosen.json`` (the models' token cost);
* the registered record ``experiments/runs/final_20261025`` (store, analyses and 100,000-draw contrasts) for the systems
  that the footing did not rerun, and the audit store for Claude Opus 5.5 over the ranker.

Figures (``thesis/figures``): F5_arch, F5_arch_float (fig:architecture) and F5_2_llm_windows through
``f5_diagrams``, F6_who (fig:who-finds), F6_cells (fig:F6_cells), F6_models (fig:F6_models), F6_13_map
(fig:F6_13_map), F6_6_ranker_gain (fig:F04_ladder), F6_8_law_article, F6_9_llm_stage, F6_10_frontier, F6_15_llm_alone,
F6_16_depends (fig:F6_16_depends).
Read-outs for the text, beside results.json: ``named_articles.json`` (what the model names from the facts alone),
``headroom.json`` (how much of a perfect reordering of the first 30 each reading reaches), ``depends.json`` (Recall@20
per cited pair by shared words and by past citations, with ``depends_pairs.parquet``, its per-pair table and cache).

Colours (D147): the system ITAM green, the ranker slate blue, the language model ochre, encoders plum, greys context.

    PYTHONPATH=src PYTHONIOENCODING=utf-8 py -3.12 -m uol.thesis_figures.v21_figs [--only F6_who,...]
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from uol.thesis_figures import ch6_figs as F
from uol.thesis_figures import common as C

ROOT = C.ROOT
OUT_DIR = ROOT / "thesis" / "figures"
REC_DIR = ROOT / "research" / "programs" / "final_results_20261102"
RESULTS = REC_DIR / "results.json"
RUN = ROOT / "experiments" / "runs" / "footing_20261101"
FOOT = RUN / "dev" / "store" / "runs"
CONTRASTS = RUN / "dev" / "contrasts_100k"
AUDIT = ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev" / "store" / "runs"
RECORD = F.RECORD_STORE
ANSWERS = ROOT / ".cache" / "footing" / "agents" / "alone_full" / "answers.json"

GREEN, TINT, INK, INK2, MUTED, RULE, MARK = F.GREEN, F.TINT, F.INK, F.INK2, F.MUTED, F.RULE, F.MARK
BLUE, PLUM, OCHRE = F.BLUE, F.PLUM, F.OCHRE
W = C.TEXTWIDTH_IN
# D209, D210: the ranker is rt_lm_tesis; the system merges the model's list with it and reads the first 30
# once, in forward order. Until v32 the models and the additions were read on the 28-source ranker's list (LIST28);
# since v33 (D203, the author 4 Nov: "Rerun everything") they are read on the ranker's own list (uol.final.lean_rerun).
RANKER, NOENC, ALONE, REORD, COMBO, MERGED = ("rt_lm_tesis", "rk_noenc", "gpt61sol_alone", "gpt61sol_rerank_rt_lm_tesis",
                                              "gpt61sol_rk_system", "gpt61sol_rk_system_merged")
LIST28 = "S0+qwen3_8b"
CHECK = {v: f"gpt61sol_rk_system_{v}" for v in ("merged", "both", "noenc", "low", "d60")}
CHECK["self"] = "gpt61sol_final_self"
MODEL_NAME = {"gpt61sol": "GPT-6.1 Sol", "gpt61terra": "GPT-6.1 Terra", "gpt6solnw": "GPT-6 Sol", "gpt56sol": "GPT-5.6 Sol",
              "claude": "Claude Opus 5.5", "gpt6luna": "GPT-6 Luna", "gpt56terra": "GPT-5.6 Terra", "luna": "GPT-5.6 Luna",
              "opus_medium": "Claude Opus 5.5", "sonnet_medium": "Claude Sonnet 5.5", "haiku_medium": "Claude Haiku 5.5"}
CHOSEN = "gpt61sol"
NOTES: dict[str, list[str]] = {}


# ---------------------------------------------------------------------------------------------------- data access
def results() -> dict:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def system_id(res: dict | None = None) -> str:
    res = res or results()
    sid = res["system"]["dev_system"]
    return sid if (FOOT / sid / "DEV" / "summary.json").exists() else COMBO


def store_of(sid: str) -> Path | None:
    for s in (FOOT, RECORD, AUDIT):
        if (s / sid / "DEV" / "summary.json").exists():
            return s
    return None


def summary(sid: str) -> dict | None:
    s = store_of(sid)
    return json.loads((s / sid / "DEV" / "summary.json").read_text(encoding="utf-8")) if s else None


def per_case(sid: str) -> pd.DataFrame | None:
    s = store_of(sid)
    return pd.read_parquet(s / sid / "DEV" / "per_case.parquet") if s else None


def contrast(key: str) -> dict | None:
    """A contrast "a - b" from the footing's 100,000-draw families (results.json families first, then the files)."""
    res = results()
    for fam in res.get("families", {}).values():
        c = fam.get("contrasts", {}).get(key)
        if c:
            return c
    c = res.get("experiments", {}).get("under_system_contrasts", {}).get(key)   # the additions read inside the system
    if c:
        return c
    for p in sorted(REC_DIR.glob("*.json")):          # the families final_results.py caches (final_dev, final_checks, ...)
        try:
            c = json.loads(p.read_text(encoding="utf-8")).get("contrasts", {}).get(key)
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(c, dict) and "delta" in c:
            return c
    for p in sorted(CONTRASTS.glob("*.json")):
        c = json.loads(p.read_text(encoding="utf-8")).get("contrasts", {}).get(key)
        if c:
            return c
    return None


def ranks(s: str) -> list[int]:
    return [int(x) for x in str(s).split(",") if x.strip()]


LOG2 = 1.0 / np.log2(np.arange(2, 12))


def ndcg_from(hits: int, n_rel: int) -> float:
    return float(LOG2[:min(hits, 10)].sum() / LOG2[:min(n_rel, 10)].sum()) if n_rel else float("nan")


def macro(values: pd.Series, groups: pd.Series) -> float:
    """Component-macro mean, the thesis measure (uol.final.measures.macro)."""
    v = pd.DataFrame({"v": values.to_numpy(float), "g": groups.to_numpy()}).dropna()
    return float(v.groupby("g")["v"].mean().mean())


def macro_ci(values: pd.Series, groups: pd.Series, n: int = 2000, seed: int = 20261103) -> tuple[float, float, float]:
    """Component-macro mean with a 95% bootstrap over components."""
    v = pd.DataFrame({"v": values.to_numpy(float), "g": groups.to_numpy()}).dropna()
    gm = v.groupby("g")["v"].mean().to_numpy()
    rng = np.random.default_rng(seed)
    draws = gm[rng.integers(0, len(gm), size=(n, len(gm)))].mean(1)
    return float(gm.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


# ---------------------------------------------------------------------------------------------------- output
def write(fig, stem: str, caption: str, short: str, label: str, place: str = "tb") -> Path:
    C.nothing_under_axes(fig)
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)
    audit = C.pdf_audit(pdf)
    C.term_lint(audit["text"], f"{stem} figure text")
    C.term_lint(caption + " " + short, f"{stem} caption")
    notes = NOTES.get(stem, []) + [f"render audit: smallest text {audit['min_pt']:.1f} pt, {audit['w_cm']:.2f} x {audit['h_cm']:.2f} cm"]
    lines = [f"% {stem}: generated by uol.thesis_figures.v21_figs; do not edit by hand. Data: results.json of",
             "% research/programs/final_results_20261102 and the footing store experiments/runs/footing_20261101/dev/store."]
    lines += [f"% Note: {n}" for n in notes]
    lines += [f"\\begin{{figure}}[{place}]", "  \\centering", f"  \\includegraphics[width=\\textwidth]{{{stem}.pdf}}",
              f"  \\caption[{short}]{{{caption}}}", f"  \\label{{{label}}}", "\\end{figure}", ""]
    (OUT_DIR / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")
    return pdf


def new(h: float):
    C.style()
    return plt.figure(figsize=(W, h))


def pct(v: float) -> str:
    return f"{100 * v:.0f}%"


# ---------------------------------------------------------------------------------------------------- who finds what
def f_who() -> Path:
    stem, label = "F6_who", "fig:who-finds"
    cap = ("Where the cited articles are found among the first ten of the model's list and of the ranker's list, on "
           "the development cases. The first bar counts cases by whether each list has a cited article in its first "
           "ten, the second counts cited articles, and the third counts cited articles that no past case cites.")
    w = results()["experiments"]["who_finds_what"]
    rows = [(f"cases ({w['case']['n']:,})", w["case"]), (f"cited articles ({w['pair']['n']:,})", w["pair"]),
            (f"cited articles that no\npast case cites ({w['never_cited_pair']['n']:,})", w["never_cited_pair"])]
    parts = [("both", "found by both lists", GREEN, "white"), ("model_only", "the model's list only", OCHRE, INK),
             ("ranker_only", "the ranker's list only", BLUE, "white"), ("neither", "neither list", C.NEUTRAL, INK)]
    h = 1.75
    fig = new(h)
    ax = fig.add_axes([0.25, 0.3 / h, 0.7, 1.05 / h])
    ys = [2, 1, 0]
    for (name, d), y in zip(rows, ys):
        left = 0.0
        for key, _, col, txt in parts:
            v = float(d.get(key, 0.0))
            ax.barh(y, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.6, zorder=2)
            if v >= 0.045:
                ax.text(left + v / 2, y, pct(v), ha="center", va="center", fontsize=7, color=txt, zorder=3)
            left += v
        assert abs(left - 1) < 1e-6, (name, left)
    ax.set_yticks(ys)
    ax.set_yticklabels([n for n, _ in rows], fontsize=8)
    ax.tick_params(axis="y", length=0)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xticklabels(["0", "20%", "40%", "60%", "80%", "100%"])
    F._legend(ax, [Patch(color=c, label=t) for _, t, c, _ in parts], "lower center", ncol=4, bbox_to_anchor=(0.38, 1.0),
              columnspacing=1.0, handlelength=1.0)
    return write(fig, stem, cap, "Who finds what", label)


# ---------------------------------------------------------------------------------------------------- four rankers
def f_cells() -> Path:
    stem, label = "F6_cells", "fig:F6_cells"
    cap = ("Four rankers trained on the same citations, with and without each kind of input, on the development "
           "cases: (a) nDCG@10, (b) Recall@20 of cited articles that no past case cites. The dotted line is tuned "
           "lexical matching.")
    res = results()
    cells = res["experiments"]["cells"]
    lex = res["system"]["dev"]["Lex"]
    rows = [("Neither", "no past citations, no encoder", MARK), ("EncOnly", "the encoder, no past citations", PLUM),
            ("CitOnly", "past citations, no encoder", BLUE), ("Both", "past citations and the encoder:\nthe ranker", GREEN)]
    h = 1.6
    fig = new(h)
    axes = [fig.add_axes([0.335, 0.42 / h, 0.3, 1.0 / h]), fig.add_axes([0.68, 0.42 / h, 0.3, 1.0 / h])]
    ys = [3, 2, 1, 0]
    for ax, key, title, letter in ((axes[0], "ndcg_at_10", "nDCG@10", "a"), (axes[1], "unseen_recall_at_20_pooled",
                                                                                 "never-cited Recall@20", "b")):
        for (k, _, col), y in zip(rows, ys):
            v = float(cells[k][key])
            ax.barh(y, v, height=0.62, color=col, lw=0, zorder=2)
            ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=7, color=INK)
        ref = float(lex[key])
        ax.axvline(ref, color=INK2, lw=0.7, ls=(0, (1, 1.6)), zorder=3)
        ax.set_xlim(0, 0.6)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.set_ylim(-0.55, 3.55)
        ax.set_yticks(ys)
        ax.set_yticklabels([r[1] for r in rows] if ax is axes[0] else [], fontsize=8)
        for t, r in zip(ax.get_yticklabels(), rows):
            t.set_color(GREEN if r[0] == "Both" else INK2)
        ax.tick_params(axis="y", length=0)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
        C.strip0(ax, "x")
        ax.set_xlabel(f"({letter}) {title}", fontsize=8)
    return write(fig, stem, cap, "Four rankers", label)


# ---------------------------------------------------------------------------------------------------- the models
LEAN_MODELS = ("gpt61sol", "gpt61terra", "gpt6solnw", "gpt56sol", "gpt6luna", "gpt56terra", "luna")


def _rates(kind: str) -> tuple[float, float, float]:
    """The provider's standard rates of a model, per million tokens (uol.final.footing.MODELS, read 2 Nov 2026)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "src"))
    from uol.final import footing as FT
    lp = FT.MODELS[kind]["list_per_mtok"]
    return float(lp["input"]), float(lp["cached_input"]), float(lp["output"])


def all_model_costs() -> dict[str, dict]:
    """model_costs() for a figure, which needs every model (uol.final.lean_rerun model --kind <k>)."""
    costs = model_costs()
    missing = [f"{k}_rerank_{RANKER}" for k in LEAN_MODELS if k not in costs]
    if missing:
        raise SystemExit(f"not yet run: {', '.join(missing)}")
    return costs


def model_costs() -> dict[str, dict]:
    """nDCG@10 of each model reordering the ranker's first 30 on every development case, and its USD per 1,000 cases
    at the standard tier (the run's own token counts at the provider's list prices) and the batch tier (half). v33:
    the ranker's own list; Claude Opus 5.5 was read on it on the sample only and is not drawn."""
    res = results()
    share = float(res["system"]["cost"]["batch_share"])
    out = {}
    for k in LEAN_MODELS:
        sid = f"{k}_rerank_{RANKER}"
        meta = ROOT / ".cache" / "footing" / "dev" / "llm" / sid / "meta.json"
        if not (FOOT / sid / "DEV" / "summary.json").exists() or not meta.exists():
            continue                                    # not yet run; the figures check that every model is present
        sm, pc = summary(sid), per_case(sid)
        std = 1000 * _usd_from_usage(json.loads(meta.read_text(encoding="utf-8"))["usage"], _rates(k)) / len(pc)
        out[k] = {"ndcg": float(sm["ndcg_at_10"]), "std": std, "batch": std * share, "cases": len(pc)}
    return out


def f_models() -> Path:
    stem, label = "F6_models", "fig:F6_models"
    cap = ("Every language model reranking the first 30 articles of the ranker's list, on every development case, "
           "against the cost of its calls per 1,000 cases at the provider's standard tier, on a logarithmic scale. The "
           "batch tier halves every cost. For GPT-6.1 Sol the line joins low, medium and high reasoning effort. The "
           "dotted line is the ranker alone.")
    m = all_model_costs()
    # v30 (the author, 3 Nov 2026): GPT-6.1 Sol's reasoning levels on the same list, joined; cost from each run's own usage
    rates = results()["system"]["cost"]["rates_per_mtok_standard"]
    effort = {}
    for e in EFFORT_ON_LIST:
        sid = f"{CHOSEN}_{e}_rerank_{RANKER}"
        meta = json.loads((ROOT / ".cache" / "footing" / "dev" / "llm" / sid / "meta.json").read_text(encoding="utf-8"))
        effort[e] = (1000 * _usd_from_usage(meta["usage"], rates) / len(per_case(sid)), float(summary(sid)["ndcg_at_10"]))
    effort["medium"] = (m[CHOSEN]["std"], m[CHOSEN]["ndcg"])
    rank = float(summary(RANKER)["ndcg_at_10"])
    h = 2.55
    fig = new(h)
    ax = fig.add_axes([0.1, 0.45 / h, 0.86, 1.95 / h])
    ax.axhline(rank, color=BLUE, lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    F._label(ax, 1.0, rank, "the ranker alone", color=BLUE, dx=2, dy=-5, ha="left", va="top")
    place = {"gpt61sol": (0, 6, "center", "bottom"), "gpt61terra": (0, 6, "center", "bottom"),
             "gpt6solnw": (0, -7, "center", "top"), "gpt56sol": (0, 6, "center", "bottom"),
             "claude": (0, -7, "center", "top"), "gpt6luna": (0, 6, "center", "bottom"),
             "gpt56terra": (0, -7, "center", "top"), "luna": (0, -7, "center", "top")}
    for k, v in m.items():
        col = OCHRE if k == CHOSEN else INK2
        ax.plot([v["std"]], [v["ndcg"]], "o", ms=4.4, color=col, mec=col, zorder=3)   # standard tier only (D2, 3 Nov 2026)
        dx, dy, ha, va = place[k]
        F._label(ax, v["std"], v["ndcg"], MODEL_NAME[k], color=col if k == CHOSEN else INK, dx=dx, dy=dy, ha=ha, va=va,
                 size=7.5)
    lv = [effort[e] for e in ("low", "medium", "high")]
    ax.plot([c for c, _ in lv], [s for _, s in lv], color=OCHRE, lw=0.9, zorder=2)
    for e, (dx, ha) in (("low", (-4, "right")), ("high", (4, "left"))):
        ax.plot([effort[e][0]], [effort[e][1]], "o", ms=3.4, mfc="white", mec=OCHRE, mew=0.9, zorder=4)
        F._label(ax, effort[e][0], effort[e][1], e, color=OCHRE, dx=dx, dy=0, ha=ha, va="center", size=7)
    ax.set_xscale("log")
    ax.set_xlim(0.7, 200)
    F._logfmt(ax, "x")
    ax.set_ylim(0.44, 0.63)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.set_xlabel("USD per 1,000 cases for the model's calls, standard tier (logarithmic scale)")
    ax.set_ylabel("nDCG@10")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    NOTES[stem] = [f"{MODEL_NAME[k]}: {v['ndcg']:.3f}, USD {v['std']:.1f} standard / {v['batch']:.1f} batch per 1,000"
                   for k, v in m.items()] + [f"GPT-6.1 Sol {e}: {s:.3f}, USD {c:.1f}" for e, (c, s) in effort.items()]
    return write(fig, stem, cap, "Every language model on the ranker's list", label)


def f_alone_list() -> Path:
    """Each model asked alone against the same model on the 28-source ranker's list, on the same sample cases (v27,
    the author 3 Nov 2026: do the other models work when simply asked?). Numbers: results.json, sample.alone*. The panel of
    reasoning efforts asked alone (v30) was cut in v32 (the author 4 Nov 2026: "this figure is not good cut it")."""
    stem, label = "F6_alone_list", "fig:F6_alone_list"
    cap = ("The language models on the same \\rfnAthree{} development cases of the sample at medium effort, each asked "
           "from the facts alone (open) and reranking the first 30 articles of the 28-source ranker's list (filled). "
           "The dotted line is that list itself on those cases.")
    sm = results()["sample"]
    rows = []
    for n, v in sm.items():
        if (n == "alone" or n.startswith("alone_")) and "on_list_same_cases" in v:
            k = CHOSEN if n == "alone" else n[len("alone_"):]
            rows.append((float(v["mean_per_case"]), float(v["on_list_same_cases"]), float(v["list_same_cases"]), k))
    # the Claude models at medium effort, through the author's account (PLAN addendum 23, the same cases)
    for k, v in ((results().get("why") or {}).get("w2_claude") or {}).items():
        if k in MODEL_NAME and v.get("on_list_same_cases") is not None:
            rows.append((float(v["alone"]), float(v["on_list_same_cases"]), rows[0][2], k))
    rows.sort(reverse=True)
    n = len(rows)
    h = 0.19 * n + 0.62
    fig = new(h)
    ax = fig.add_axes([0.2, 0.45 / h, 0.775, 0.19 * n / h])
    ys, labels, colors = [], [], []
    for i, (alone, on_list, _, k) in enumerate(rows):
        y = n - 1 - i
        col = OCHRE if k == CHOSEN else INK2
        ax.plot([alone, on_list], [y, y], color=col, lw=0.8, zorder=2)
        ax.plot([on_list], [y], "o", ms=4.4, color=col, mec=col, zorder=3)
        ax.plot([alone], [y], "o", ms=4.4, mfc="white", mec=col, mew=0.9, zorder=3)
        ys.append(y); labels.append(MODEL_NAME[k]); colors.append(col if k == CHOSEN else INK)
    lst = rows[0][2]
    ax.axvline(lst, color=BLUE, lw=0.7, ls=(0, (1, 1.6)), zorder=1)
    F._label(ax, lst, n - 0.55, "the 28-source ranker's list", color=BLUE, dx=-3, dy=0, ha="right", va="center")
    F._rows_axes(ax, ys, labels, colors)
    ax.set_ylim(-0.6, n - 0.2)
    ax.set_xlim(min(0.3, np.floor(10 * min(r[0] for r in rows) - 0.3) / 10), 0.7)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.set_xlabel(C.axis_title("nDCG@10"))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    keys = [Line2D([0], [0], marker="o", ls="none", ms=4.4, mfc="white", mec=INK2, mew=0.9, label="asked alone"),
            Line2D([0], [0], marker="o", ls="none", ms=4.4, color=INK2, label="on the ranker's list")]
    F._legend(ax, keys, "lower right", borderaxespad=0.2)
    NOTES[stem] = [f"{MODEL_NAME[k]}: alone {a:.3f}, on the list {b:.3f}, the list {c:.3f}" for a, b, c, k in rows]
    return write(fig, stem, cap, "Each model alone and on the ranker's list", label)


# ---------------------------------------------------------------------------------------------------- the ranker's gain
LADDER_LEAN = [("bge_m3", "BGE-M3"), ("qwen3_0p6b", "Qwen3-Embedding-0.6B"), ("qwen3_4b", "Qwen3-Embedding-4B"),
               ("qwen3_8b", "Qwen3-Embedding-8B")]   # v33: Euler is compared alone (6.4), not in the ranker


def f_ranker_gain() -> Path:
    stem, label = "F6_6_ranker_gain", "fig:F04_ladder"
    cap = ("Gain in nDCG@10 from an encoder on the development cases, each over the ranker without its encoder. Above, "
           "the score of one encoder added to the ranker without its encoder. Below, the ranker itself, in which "
           "Qwen3-Embedding-8B adds its own score and its two past-case sources to the other five. Lines are unadjusted "
           "95\\% intervals, short bars their lower ends adjusted for the family, and markers the classes of "
           "\\cref{sec:comparing-systems}.")
    # v33 (D203): the ladder on the eight-source ranker without its encoder (uol.final.lean_rerun lf_noenc+<encoder>;
    # final_results.py caches the family lean_ladder), no longer on the 28-source ranker without it
    enc = []
    for key, text in LADDER_LEAN:
        c = contrast(f"rk_noenc+{key} - {NOENC}")
        if not c:
            raise SystemExit(f"rk_noenc+{key} - {NOENC}: not yet in rk_ladder (run scripts/final_results.py)")
        enc.append((text, c["delta"], *c["ci95"], c["class"], c["adjusted_lower"]))
    c = contrast(f"{RANKER} - {NOENC}")                 # PLAN addendum 22, a family of one
    systems = [("the ranker", c["delta"], *c["ci95"], c["class"], c["adjusted_lower"])] if c else []
    enc.sort(key=lambda r: r[1])
    slots = [("h", "one encoder's score added")] + [("r", r) for r in enc] + [("g", None), ("h", "the ranker")] \
        + [("r", r) for r in systems]
    n = len(slots)
    h = 0.165 * n + 0.74
    fig = new(h)
    ax = fig.add_axes([0.4, 0.62 / h, 0.57, (0.165 * n) / h])
    ys, labels, colors, seen, span = [], [], [], set(), [0.0, 0.0]
    for i, (kind, r) in enumerate(slots):
        y = n - 1 - i
        if kind == "h":
            C.group_heading(ax, y, r)
        elif kind == "r":
            text, mval, lo, hi, cls, adj = r
            seen.add(cls); span = [min(span[0], lo, adj), max(span[1], hi)]
            col = GREEN if text == "the ranker" else PLUM
            ax.plot([lo, hi], [y, y], color=col, lw=0.8, zorder=2)
            ax.plot([adj], [y], marker="|", ms=7, mew=1.1, ls="none", color=col, zorder=4)
            F._dot(ax, mval, y, cls, color=col)
            ys.append(y); labels.append(text); colors.append(GREEN if col == GREEN else INK2)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.axvline(.02, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=0)
    F._label(ax, .02, n - 1, "margin", color=MUTED, dx=3, ha="left")
    F._rows_axes(ax, ys, labels, colors)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(min(-0.008, span[0] - 0.004), max(0.09, span[1] + 0.006))
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    C.strip0(ax, "x")
    ax.set_xlabel("gain in nDCG@10 from the encoder,\n" + C.CASES)
    F._legend(ax, [Line2D([0], [0], marker="o", ls="none", ms=4.2, mew=0.8, label=F.CLASS_NAME[k], **F.CLASS_MARK[k])
                   for k in ("useful", "nonzero_below_margin", "equivalent") if k in seen]
              + [Line2D([0], [0], marker="|", ms=7, mew=1.1, ls="none", color=INK, label="lower end that decides the class")],
              "upper right")
    return write(fig, stem, cap, "Gain over the ranker without the encoder", label)


# ---------------------------------------------------------------------------------------------------- law and article
def f_law_article() -> Path:
    stem, label = "F6_8_law_article", "fig:F6_8_law_article"
    cap = ("Where the first result lands on the development cases, for each system: a cited article, another article "
           "of a cited law, or an article of another law. The system's bar is drawn in green.")
    res = results()
    dev = res["system"]["dev"]
    rows = [("Sys", "the system"), ("Alone", "the model alone"), ("Reord", "the model reranking the ranker's list"),
            ("Rank", "the ranker"), ("Noenc", "the ranker without the encoder"), ("Lex", "tuned lexical matching")]
    greys, greens = F.F68_GREYS, F.F68_GREENS
    n = len(rows)
    h = 0.2 * n + 0.85
    fig = new(h)
    ax = fig.add_axes([0.4, 0.5 / h, 0.58, 0.2 * n / h])
    ys = list(range(n))[::-1]
    for (k, _), y in zip(rows, ys):
        law1, art1 = float(dev[k]["law_at_1"]), float(dev[k]["success_at_1"])
        assert 0 <= art1 <= law1 <= 1, (k, art1, law1)
        shades = greens if k == "Sys" else greys
        for left, width, col in zip((0, art1, law1), (art1, law1 - art1, 1 - law1), shades):
            ax.barh(y, width, left=left, height=0.66, color=col, edgecolor="white", linewidth=0.6, zorder=2)
    F._rows_axes(ax, ys, [r[1] for r in rows], [GREEN if r[0] == "Sys" else INK2 for r in rows])
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    C.strip0(ax, "x")
    ax.set_xlabel("share of development cases, by where the first result lands")
    F._legend(ax, [Patch(color=greys[0], label="a cited article"), Patch(color=greys[1], label="another article of a cited law"),
                   Patch(color=greys[2], label="an article of another law")],
              "lower center", ncol=3, bbox_to_anchor=(0.4, 1.0), columnspacing=1.2, handlelength=1.0)
    return write(fig, stem, cap, "Law against article", label)


# ---------------------------------------------------------------------------------------------------- the final reranking
def oracle(pc: pd.DataFrame, depth: int = 30) -> pd.Series:
    """nDCG@10 of a perfect reordering of the first ``depth``: every cited article among them moved to the top."""
    return pd.Series([ndcg_from(sum(1 for r in ranks(s) if r <= depth), int(n)) for s, n in
                      zip(pc["positive_ranks"], pc["n_relevant"])], index=pc.index)


def headroom() -> dict:
    res = results()
    sys_id = system_id(res)
    out = {"system": sys_id, "depth": 30, "measure": "nDCG@10, component-macro mean (the thesis measure)"}
    for name, base, read in (("ranker", RANKER, REORD), ("merged", MERGED, sys_id)):
        pb, pr = per_case(base), per_case(read)
        orc = oracle(pb)
        # the formula reproduces the stored nDCG@10 from the stored ranks
        chk = pd.Series([sum(1.0 / math.log2(r + 1) for r in ranks(s) if r <= 10) / LOG2[:min(int(n), 10)].sum()
                         for s, n in zip(pb["positive_ranks"], pb["n_relevant"])])
        assert np.allclose(chk.to_numpy(), pb["ndcg_at_10"].to_numpy(), atol=1e-9), name
        pairs = [r for s in pb["positive_ranks"] for r in ranks(s)]
        out[name] = {"list": base, "read": read,
                     "share_cited_in_first_30": float(np.mean(np.asarray(pairs) <= 30)),
                     "cases_with_cited_in_first_30": float(np.mean([any(r <= 30 for r in ranks(s)) for s in pb["positive_ranks"]])),
                     "list_ndcg_at_10": macro_ci(pb["ndcg_at_10"], pb["component_id"]),
                     "perfect_reorder_ndcg_at_10": macro_ci(orc, pb["component_id"]),
                     "read_ndcg_at_10": macro_ci(pr["ndcg_at_10"], pr["component_id"])}
        assert abs(out[name]["list_ndcg_at_10"][0] - summary(base)["ndcg_at_10"]) < 1e-9
    out["share_cited_in_first_30_of_merged"] = out["merged"]["share_cited_in_first_30"]
    out["perfect_reorder_ndcg_at_10"] = out["merged"]["perfect_reorder_ndcg_at_10"][0]
    out["system_ndcg"] = out["merged"]["read_ndcg_at_10"][0]
    out["note"] = ("intervals: 95% bootstrap over components (2,000 draws, seed 20261103); share_cited_in_first_30 counts "
                   "cited articles; cases_with_cited_in_first_30 counts cases with at least one")
    (REC_DIR / "headroom.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def f_llm_stage() -> Path:
    stem, label = "F6_9_llm_stage", "fig:F6_9_llm_stage"
    cap = ("Development cases, nDCG@10 with 95\\% intervals: the list the model reranks, the model's new order of its "
           "first 30, and a perfect reordering of the same 30, for the ranker's list and for the merged list.")
    hr = headroom()
    groups = [("the model reranking the ranker's list", [("list", "the ranker's list", BLUE, None),
                                                      ("read", "the model's new order", OCHRE, None),
                                                      ("perfect", "perfect reordering of its first 30", "white", INK2)], hr["ranker"]),
              ("the system", [("list", "the merged list", MARK, None), ("read", "the system: the final reranking", GREEN, None),
                              ("perfect", "perfect reordering of its first 30", "white", INK2)], hr["merged"])]
    slots = []
    for head, rows, d in groups:
        slots += ([("g", None)] if slots else []) + [("h", head)] + [("r", (r, d)) for r in rows]
    n = len(slots)
    h = 0.19 * n + 0.6
    fig = new(h)
    ax = fig.add_axes([0.4, 0.45 / h, 0.575, 0.19 * n / h])
    ys, labels, colors = [], [], []
    key = {"list": "list_ndcg_at_10", "read": "read_ndcg_at_10", "perfect": "perfect_reorder_ndcg_at_10"}
    for i, (kind, r) in enumerate(slots):
        y = n - 1 - i
        if kind == "h":
            C.group_heading(ax, y, r)
        elif kind == "r":
            (k, text, fill, edge), d = r
            v, lo, hi = d[key[k]]
            ax.barh(y, v, 0.62, color=fill, edgecolor=edge or fill, lw=0.7 if edge else 0, zorder=2)
            ax.hlines(y, lo, hi, color=INK, lw=0.7, zorder=3)
            ys.append(y); labels.append(text); colors.append(GREEN if fill == GREEN else INK2)
    F._rows_axes(ax, ys, labels, colors)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "x")
    ax.set_xlabel(C.axis_title("nDCG@10"))
    return write(fig, stem, cap, "Reranking by the language model", label)


# ---------------------------------------------------------------------------------------------------- quality and cost
# The family of each other system of the frontier (2 Nov 2026), drawn as small dots in a light tint of its family's
# colour, in the manner of F6_1's other systems of a kind. A system not listed here stays grey (open) and is named in
# the figure's notes: the untrained hybrids of lexical matching and an encoder, and query2doc and HyDE (facts
# rewritten by a language model, then searched with an encoder), belong to no single family.
FRONT_FAMILY = {
    **dict.fromkeys(("bm25_stem", "lstar_headings"), "lexical"),
    **dict.fromkeys(("lsa", "lda_sim", "lda_dm", "doc2vec", "word2vec"), "collection"),
    **dict.fromkeys(("qwen3_0p6b", "qwen3_4b", "qwen3_8b", "qwen3_0p6b_ft", "qwen3_8b_mrl256_int8", "qwen3_8b_binary",
                     "bge_m3_colbert"), "encoder"),
    **dict.fromkeys(("S0", "S0+qwen3_0p6b", "S0+qwen3_4b", "S_full", "S_full_ext", RANKER, LIST28), "ranker"),
    **{f"{k}_rerank_{RANKER}": "model" for k in MODEL_NAME},
}
LEADER = dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1, shrinkB=3)
FRONT_NAME: dict[str, tuple] = {}   # v33 (TEXT_AUDIT_2 R6-24): only the setups, the model alone and lexical matching are named
# each name sits just left of its own point (the author, 3 Nov 2026: the two names above were hard to place); no cheaper
# system scores higher than a point on the frontier, so the space to its left at its own height is empty
# the setups of the thesis are named in their colour, the other systems on the frontier in grey and smaller, so the eye
# reads the setups first (the author, 3 Nov 2026: the names were confusing)
FRONT_LEAD: set[str] = set()   # names placed in free space and joined by a thin leader (none at present)
FRONT_UNNAMED = {"linear__lstar_headings__qwen3_0p6b"}   # on the frontier but left unnamed (the author, 3 Nov 2026)
FRONT_KEY = [  # (family, legend entry, style of its small dots), in the order of the legend
    ("lexical", "lexical matching", dict(mfc=C.tint(INK, 0.6), mec="white", mew=0.4)),
    ("collection", "learned from the collection", dict(mfc="#cdcdcd", mec="white", mew=0.4)),
    ("encoder", "encoders", dict(mfc=C.tint(PLUM, 0.55), mec="white", mew=0.4)),
    ("ranker", "rankers on past citations", dict(mfc=C.tint(BLUE, 0.55), mec="white", mew=0.4)),
    ("model", "language models", dict(mfc=C.tint(OCHRE, 0.6), mec="white", mew=0.4)),
    ("system", "settings of the system", dict(mfc=C.tint(GREEN, 0.45), mec="white", mew=0.4)),
    ("other", "rewritten facts or simple fusion", dict(mfc="white", mec=MUTED, mew=0.6)),
]
EFFORT_ON_LIST = ("low", "high")   # GPT-6.1 Sol reordering the ranker's list at other efforts (medium: model_costs)


def _usd_from_usage(u: dict, rates) -> float:
    """List-price USD of one run's calls: uncached input, cached input and output tokens at the standard rates."""
    cached = float(u.get("cached_input_tokens") or 0)
    return ((float(u["input_tokens"]) - cached) * rates[0] + cached * rates[1] + float(u["output_tokens"]) * rates[2]) / 1e6


def frontier_extra(cost: dict, usd_ranker: float, usd_noenc: float) -> pd.DataFrame:
    """Unnamed points on every development case (the author, 3 Nov 2026): GPT-6.1 Sol reordering the 28-source ranker's list
    at each reasoning effort, and the six settings of the system (PLAN addendum 22), each at USD per 1,000 cases at the
    standard tier. A reading's cost is its receipt; the self-reading's receipt covers both orders, so it is halved, as
    the system's own reading is (results.json, system.cost.second_reading)."""
    rates = cost["rates_per_mtok_standard"]
    rec = lambda n: json.loads((RUN / "receipts" / f"{n}.json").read_text(encoding="utf-8"))
    per1000 = lambda usd, n: 1000 * float(usd) / n
    rows = []
    for e in EFFORT_ON_LIST:
        sid = f"{CHOSEN}_{e}_rerank_{RANKER}"
        meta = json.loads((ROOT / ".cache" / "footing" / "dev" / "llm" / sid / "meta.json").read_text(encoding="utf-8"))
        n = len(per_case(sid))
        rows.append({"system": sid, "usd_per_1000": per1000(_usd_from_usage(meta["usage"], rates), n) + usd_ranker,
                     "ndcg_at_10": float(summary(sid)["ndcg_at_10"]), "family": "model"})
    lst = float(cost["model_list"]["usd_standard_per_1000"])
    read = float(cost["second_reading"]["usd_standard_per_1000"])
    n = int(cost["second_reading"]["cases"])
    reading = {"merged": 0.0, "self": per1000(rec("agents_combo_self")["usd_list_equivalent"], n) / 2, "both": 2 * read,
               **{v: per1000(rec(f"rk_{CHECK[v]}")["usd_list_equivalent"], n) for v in ("noenc", "low", "d60")}}
    ranker = {"self": 0.0, "noenc": usd_noenc}
    for v, r in reading.items():
        rows.append({"system": CHECK[v], "usd_per_1000": lst + r + ranker.get(v, usd_ranker),
                     "ndcg_at_10": float(summary(CHECK[v])["ndcg_at_10"]), "family": "system"})
    return pd.DataFrame(rows)


def _clear(ax, artist, xs, ys, what: str, pad_pt: float = 0.5) -> None:
    """Raise if a point (data coordinates) lies under ``artist`` (a legend or a label), within ``pad_pt`` points."""
    from matplotlib.text import Text
    fig = ax.figure
    fig.canvas.draw()
    # an annotation's own extent includes its leader; only the text box may not cover a point
    bb = (Text.get_window_extent(artist, fig.canvas.get_renderer()) if isinstance(artist, Text)
          else artist.get_window_extent(fig.canvas.get_renderer()))
    pad = pad_pt * fig.dpi / 72
    px = ax.transData.transform(np.column_stack([np.asarray(xs, float), np.asarray(ys, float)]))
    under = (px[:, 0] > bb.x0 - pad) & (px[:, 0] < bb.x1 + pad) & (px[:, 1] > bb.y0 - pad) & (px[:, 1] < bb.y1 + pad)
    if under.any():
        raise ValueError(f"{what} hides {int(under.sum())} point(s): "
                         f"{[(round(float(x), 6), round(float(y), 3)) for x, y in zip(np.asarray(xs)[under], np.asarray(ys)[under])]}")


def f_frontier() -> Path:
    """Quality against cost per case, one mark per system at the standard tier (the author, 2 Nov 2026: the two marks per
    system read as a separate result). The batch tier halves the model's part and is stated in the caption."""
    stem, label = "F6_10_frontier", "fig:F6_10_frontier"
    cap = ("Ranking quality against cost per 1,000 cases, in USD at the providers' standard rates, on a logarithmic scale. "
           "The line joins the systems that nothing cheaper matches. The setups and the model alone are in colour. "
           "Small dots mark the other systems, coloured by family.")
    res = results()
    cost = res["system"]["cost"]
    dev = res["system"]["dev"]
    q = pd.read_csv(F.RECORD_ANALYSIS / "c8" / "c8_quantities.csv").rename(columns={"usd_per_1000_base": "usd_per_1000"})
    timed = json.loads(F.COST_TIMING.read_text(encoding="utf-8"))["usd_per_1000_timed"]
    usd_ranker = float(cost["ranker"]["usd_per_1000_median"])
    usd_noenc = float(cost["ranker_noenc"]["usd_per_1000_median"])
    # context: every local system of the record at its own cost; the language-model rows of the record were read on
    # other lists and are left out; the other models on the ranker's list are added from their own receipts
    q = q[~q["system"].str.contains("llm|claude|rerank", regex=True)].copy()
    for sid, k in (("lstar_headings", "lstar(timed)"), ("S0", "S0"), ("S_full_ext", "S_full_ext")):
        q.loc[q["system"] == sid, "usd_per_1000"] = timed[k]
    q = q[~q["system"].isin([RANKER, LIST28])]
    mc = all_model_costs()
    ch = mc[CHOSEN]
    # every model on the ranker's list, GPT-6.1 Sol at medium effort among them (the author, 3 Nov 2026; v33: the ranker's own list)
    others = pd.DataFrame([{"system": f"{k}_rerank_{RANKER}", "usd_per_1000": v["std"] + usd_ranker, "ndcg_at_10": v["ndcg"]}
                           for k, v in mc.items()])
    extra = frontier_extra(cost, usd_ranker, usd_noenc)
    q = pd.concat([q[["system", "usd_per_1000", "ndcg_at_10"]], others, extra.drop(columns="family")],
                  ignore_index=True).dropna()
    cents = lambda usd_per_1000: usd_per_1000                 # v33 (R6-23): USD per 1,000 cases, as the tables
    named = {  # name: (nDCG@10, US cents per case at the standard tier, colour, marker)
        "tuned lexical matching": (dev["Lex"]["ndcg_at_10"], cents(timed["lstar(timed)"]), INK, "s"),
        "the ranker without the encoder": (dev["Noenc"]["ndcg_at_10"], cents(usd_noenc), BLUE, "o"),
        "the ranker": (dev["Rank"]["ndcg_at_10"], cents(usd_ranker), BLUE, "o"),
        "the model alone": (dev["Alone"]["ndcg_at_10"], cents(cost["model_list"]["usd_standard_per_1000"]), OCHRE, "o"),
        "the system": (dev["Sys"]["ndcg_at_10"], cents(cost["system"]["usd_standard_per_1000"] + usd_ranker), GREEN, "D"),
    }
    q = q[~q["system"].isin(["lstar_headings", "S0"])]   # drawn as named points, not as small dots
    q = q.assign(family=q["system"].map({**FRONT_FAMILY, **dict(zip(extra["system"], extra["family"]))}).fillna("other"))
    # v30 (the author, 3 Nov 2026: points above the system's mark): the system's own settings, none meaningfully different
    # from it, sat on and above its mark; they are compared in Figure 6.13 and are not drawn here
    q = q[q["family"] != "system"]
    unnamed = set(q["system"])                       # v33 (R6-24): only the named points carry names
    pts = [(cents(float(r["usd_per_1000"])), float(r["ndcg_at_10"]), r["system"]) for _, r in q.iterrows()]
    pts += [(c, v, k) for k, (v, c, _, _) in named.items()]
    pts.sort()
    front, best = [], -1.0
    for x, y, k in pts:
        if y > best:
            front.append((x, y, k)); best = y
    pts = [(x, y) for x, y, _ in pts]
    h = 2.35   # shorter so that it shares a page with its text (the author, 5 Nov 2026)
    fig = new(h)
    ax = fig.add_axes([0.1, 0.42 / h, 0.86, 1.86 / h])
    keys = []
    for fam, text, st in FRONT_KEY:
        d = q[q["family"] == fam]
        if not d.empty:
            # the settings of the system lie on or beside its own mark, so they are drawn smaller and above it
            ms, z = (2.8, 5) if fam == "system" else (3.4, 2)
            ax.plot([cents(x) for x in d["usd_per_1000"]], d["ndcg_at_10"], "o", ls="none", ms=ms, zorder=z, **st)
            keys.append(Line2D([0], [0], marker="o", ls="none", ms=ms, label=text, **st))
    ax.step([p[0] for p in front], [p[1] for p in front], where="post", color=MUTED, lw=0.7, zorder=1)
    spots = {"tuned lexical matching": (0, -6, "center", "top"), "the ranker without the encoder": (-5, 0, "right", "center"),
             "the ranker": (-6, 0, "right", "center"),
             "the model alone": (-6, 0, "right", "center"), "the system": (-6, 6, "right", "bottom")}
    lead = {}
    labels = []
    # v27 (3 Nov 2026, the author): every system on the frontier is named, the small dots on it in the colour of their family
    for x, y, k in front:
        if k in named or k in unnamed:
            continue
        text, (dx, dy, ha, va) = FRONT_NAME.get(k, (k, (0, 5, "center", "bottom")))
        labels.append(F._label(ax, x, y, text, color=MUTED, dx=dx, dy=dy, ha=ha, va=va, size=7,
                               **({"arrowprops": LEADER} if k in FRONT_LEAD else {})))
    for name, (v, c, col, mk) in named.items():
        ax.plot([c], [v], mk, ms=4.6, color=col, mec=col, zorder=4)
        dx, dy, ha, va = spots[name]
        text, arrow = lead.get(name, (name, None))
        labels.append(F._label(ax, c, v, text, color=col if col != INK2 else INK, dx=dx, dy=dy, ha=ha, va=va, size=8,
                               **({"arrowprops": arrow} if arrow else {})))
    ax.set_xscale("log")
    ax.set_xlim(2e-5, 200)
    F._logfmt(ax, "x")
    ax.set_ylim(0, 0.78)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "y")
    ax.set_xlabel("USD per 1,000 cases (logarithmic scale)")
    ax.set_ylabel(C.axis_title("nDCG@10"))
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    # one family key in the empty lower right (no system is both dear and weak), checked to hide no point or label
    leg = F._legend(ax, keys, "lower right", borderaxespad=0.2, handletextpad=0.3, labelspacing=0.3)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    _clear(ax, leg, xs, ys, "the family key")
    for t in labels:
        _clear(ax, t, xs, ys, f"the label {t.get_text()!r}")
    NOTES[stem] = [f"{k}: nDCG {v[0]:.3f}, USD per 1,000 cases {v[1]:.4g}" for k, v in named.items()]
    NOTES[stem].append("frontier: " + "; ".join(f"{k} ({y:.3f}, {x:.3g})" for x, y, k in front))
    NOTES[stem] += [f"small dots, {text}: {', '.join(sorted(q.loc[q['family'] == fam, 'system']))}"
                    for fam, text, _ in FRONT_KEY if fam != "other" and (q["family"] == fam).any()]
    if (q["family"] == "other").any():
        NOTES[stem].append("not placed in a family (grey, open): "
                           + ", ".join(sorted(q.loc[q["family"] == "other", "system"])))
    return write(fig, stem, cap, "Quality against cost", label)


# ---------------------------------------------------------------------------------------------------- the model alone
def _natural(label: str) -> tuple:
    m = re.match(r"(\d+)(.*)", label)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 9, label)


def named_articles() -> dict:
    """What GPT-6.1 Sol names from the facts alone, on every scored development case: each named article as cited,
    next to a cited article (the article just before or after it in the same law), elsewhere in a cited law, in
    another law of the catalogue, or not in the catalogue (a name that resolves to no article: another law, a state
    law, a repealed article or one that does not exist). Generic articles named are counted apart."""
    from uol.embedding_push import common as ep
    from uol.round3 import catalog_map
    d = ep.release()
    arts = [json.loads(line) for line in Path(d.candidate_binding["articles_path"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    inst = {}
    for a in arts:
        inst.setdefault(a["instrument_id"], []).append(a)
    pos_in_law, law_of = {}, {}
    for law, items in inst.items():
        for i, a in enumerate(sorted(items, key=lambda a: _natural(a["article_label"]))):
            idx = d.candidate_index[a["candidate_id"]]
            pos_in_law[idx], law_of[idx] = i, law
    generic = {int(i) for i in d.generic_idx}
    answers = json.loads(ANSWERS.read_text(encoding="utf-8"))
    pc = per_case(ALONE)                                          # the scored cases and their (corrected) cited articles
    counts = dict.fromkeys(("named", "cited", "near", "same_law_other", "other_law", "not_in_catalogue", "generic"), 0)
    reasons: dict[str, int] = {}
    for row, cited_s in zip(pc["row"], pc["positive_idx"]):
        cited = {int(x) for x in str(cited_s).split(",") if x.strip()}
        cited_laws = {law_of[c] for c in cited if c in law_of}
        near = {(law_of[c], pos_in_law[c] + k) for c in cited if c in law_of for k in (-1, 1)}
        seen = set()
        for x in answers[str(int(row))]["articulos"]:
            idx, why = catalog_map.resolve(x.get("ley", ""), str(x.get("articulo", "")))
            key = idx if idx is not None else ("?", catalog_map.__name__, x.get("ley", ""), str(x.get("articulo", "")))
            if key in seen:
                continue
            seen.add(key)
            counts["named"] += 1
            if idx is None:
                counts["not_in_catalogue"] += 1
                reasons[why] = reasons.get(why, 0) + 1
            elif int(idx) in cited:
                counts["cited"] += 1
            elif int(idx) in generic:
                counts["generic"] += 1
            elif (law_of.get(int(idx)), pos_in_law.get(int(idx))) in near:
                counts["near"] += 1
            elif law_of.get(int(idx)) in cited_laws:
                counts["same_law_other"] += 1
            else:
                counts["other_law"] += 1
    out = {"model": "gpt-6.1-sol", "system": ALONE, "cases": int(len(pc)), **counts,
           "shares_of_named": {k: v / counts["named"] for k, v in counts.items() if k != "named"},
           "not_in_catalogue_reasons": reasons,
           "definition": ("each case's named articles, duplicates within a case counted once; cited = the case's cited "
                          "(specific) articles as scored; near = the article just before or after a cited article in its "
                          "law's order; same_law_other = another article of a cited law; other_law = an article of a law "
                          "the case does not cite; not_in_catalogue = no catalogue article resolves (uol.round3.catalog_map); "
                          "generic = a generic article that is not cited")}
    (REC_DIR / "named_articles.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def f_llm_alone() -> Path:
    stem, label = "F6_15_llm_alone", "fig:F6_15_llm_alone"
    na = named_articles()
    cap = (f"The articles the model names from the facts alone, with no search, on the {na['cases']:,} development "
           "cases, as shares of all the articles it names.")
    sh = na["shares_of_named"]
    rows = [("cited by the court", sh["cited"], GREEN), ("next to a cited article", sh["near"], PLUM),
            ("elsewhere in a cited law", sh["same_law_other"], PLUM), ("in a law the case does not cite", sh["other_law"], MARK),
            ("a generic article", sh["generic"], MARK), ("not in the catalogue", sh["not_in_catalogue"], OCHRE)]
    h = 1.6
    fig = new(h)
    ax = fig.add_axes([0.36, 0.3 / h, 0.6, 1.2 / h])
    ys = list(range(len(rows)))[::-1]
    top = max(v for _, v, _ in rows)
    for (lab, v, col), y in zip(rows, ys):
        ax.barh(y, v, height=0.62, color=col, lw=0)
        ax.text(v + 0.006, y, f"{100 * v:.0f}%" if v >= 0.1 else f"{100 * v:.1f}%", va="center", fontsize=7, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8, color=INK)
    ax.tick_params(axis="y", length=0, pad=4)
    for s in ("left", "top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(0, math.ceil(top * 10 + 0.5) / 10)
    ticks = np.arange(0, ax.get_xlim()[1] + 1e-9, 0.1)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["0"] + [f"{t:.0%}" for t in ticks[1:]])
    return write(fig, stem, cap, "The model without a search", label)


# ---------------------------------------------------------------------------------------------------- the map
# v33 (D203, 4 Nov 2026): every row on the eight-source ranker (uol.final.lean_rerun), grouped by the part it would
# change; one 95% interval per row and the class as the marker (the decisive adjusted end is no longer drawn).
MAP = [  # (heading, [(contrast key, row label, sign)]); sign -1 shows b - a
    ("Parts of the system, what each adds", [   # the parts as the text reads them (put_contrast negate)
        (f"{CHECK['merged']} - {COMBO}", "the final reranking", -1),
        (f"{CHECK['self']} - {COMBO}", "the ranker's list in the merge", -1),
        (f"{CHECK['noenc']} - {COMBO}", "the encoder", -1)]),
    ("Settings of the final reranking", [
        (f"{CHECK['both']} - {COMBO}", "each window also in reverse order", 1),
        (f"{CHECK['low']} - {COMBO}", "low reasoning effort", 1),
        (f"{CHECK['d60']} - {COMBO}", "60 articles instead of 30", 1)]),
    ("Other rankers, in the system", [
        (f"gpt61sol_combo_fwd_on_ft_lean - {COMBO}", "trees on single articles", 1),
        (f"gpt61sol_final_on_S_full_ext - {COMBO}", "every other source", 1)]),
    ("Changes to the ranker", [
        (f"{RANKER} - rt_check", r"votes of the older $\it{tesis}$", 1),
        ("rt_check - lf_plain", "a listwise objective", 1),
        (f"rk_structure - {RANKER}", "statute structure, as features", 1),
        (f"rk_smooth - {RANKER}", "statute structure, smoothing vectors", 1),
        (f"rk_cocite - {RANKER}", "co-citation", 1),
        (f"rk_adapter - {RANKER}", "an adapter on the encoder", 1),
        (f"rk_issues - {RANKER}", "the facts restated as issues", 1),
        (f"rk_patterns - {RANKER}", "fact patterns written by a model", 1),
        (f"rk_q2d - {RANKER}", "query2doc", 1),
        (f"rk_hyde - {RANKER}", "HyDE", 1),
        (f"rk_topics - {RANKER}", "the four topic scores", 1),
        ("ft_lean - ft_ranker", "28 sources instead of eight", -1)]),   # single-article trees, stored as the family's ft_lean - ft_ranker
    ("The reranking model, against GPT-6 Sol", [
        (f"{k}_rerank_{RANKER} - gpt6solnw_rerank_{RANKER}", MODEL_NAME[k], 1)
        for k in ("gpt61sol", "gpt61terra", "gpt56sol", "gpt6luna", "gpt56terra", "luna")]),
]
MAP_CLASS = {  # class -> (legend name, marker, face, edge); the classes of 4.4, a loss included
    "useful": ("clears the margin", "o", INK2, INK2), "nonzero_below_margin": ("a real gain, not shown\nto reach the margin", "o", MARK, MARK),
    "equivalent": ("no meaningful difference", "o", "white", INK2), "loss": ("a loss", "s", INK2, INK2),
    "inconclusive": ("inconclusive", "s", "white", INK2)}


def flipped(c: dict) -> dict:
    """b - a from a stored a - b, classed again by the rule, as scripts/final_results.py put_contrast(negate=True)."""
    cia = [-c["ci_adjusted"][1], -c["ci_adjusted"][0]]
    ci90 = [-c["ci90"][1], -c["ci90"][0]]
    cls = ("useful" if cia[0] >= 0.02 else "nonzero_below_margin" if cia[0] > 0
           else "equivalent" if -0.02 < ci90[0] and ci90[1] < 0.02 else "inconclusive")
    return {**c, "delta": -c["delta"], "ci95": [-c["ci95"][1], -c["ci95"][0]], "ci_adjusted": cia, "ci90": ci90,
            "adjusted_lower": cia[0], "class": cls}


def map_class(c: dict) -> str:
    """The class of 4.4; an inconclusive difference whose adjusted upper end lies below zero is a loss."""
    cia = c.get("ci_adjusted")
    return "loss" if c["class"] == "inconclusive" and cia and cia[1] < 0 else c["class"]


def f_map() -> Path:
    stem, label = "F6_13_map", "fig:F6_13_map"
    cap = ("Every alternative against the part of the system it would change, on the development cases: the change in "
           r"nDCG@10 with its 95\% interval, marked by its class under the rule of \cref{sec:comparing-systems}. "
           r"Shaded, the margin of $\pm0.02$.")
    rows, missing = [], []
    for head, items in MAP:
        got = []
        for key, text, sign in items:
            c = contrast(key)
            if c is None:
                missing.append(key)
                continue
            c = flipped(c) if sign < 0 else c
            got.append((text, c["delta"], c["ci95"][0], c["ci95"][1], map_class(c)))
        if got:
            rows.append(("h", head))
            rows += [("r", r) for r in got]
    if missing:
        raise SystemExit(f"F6_13_map: not yet measured: {', '.join(missing)}")
    step, gap = 0.142, 0.10                        # inches per row, extra space above a heading
    y, ys = 0.0, []
    for kind, _ in rows:
        y += gap if kind == "h" and ys else 0.0
        ys.append(y)
        y += step
    top, bottom = 0.06, 0.52                       # the key and the axis label sit under the rows
    h = y + top + bottom
    fig = new(h)
    left = 0.40
    ax = fig.add_axes([left, bottom / h, 0.975 - left, y / h])
    ax.axvspan(-0.02, 0.02, color="#f1f1f1", zorder=0, lw=0)
    ax.axvline(0, color=RULE, lw=0.6, zorder=1)
    ticks, labels, colors, seen = [], [], [], []
    xmin, xmax = -0.03, 0.05
    for (kind, r), yy in zip(rows, ys):
        yd = y - yy - step / 2                     # data y in inches from the bottom of the axes
        if kind == "h":
            ax.text((0.004 - left) / (0.975 - left), yd, r, transform=ax.get_yaxis_transform(), ha="left", va="center",
                    fontsize=7.5, weight="bold", color=INK, clip_on=False)
            continue
        text, m, lo, hi, cls = r
        col = GREEN if text == MODEL_NAME[CHOSEN] else INK2
        name, mk, face, edge = MAP_CLASS[cls]
        ax.plot([lo, hi], [yd, yd], color=col, lw=0.9, zorder=2, solid_capstyle="butt")
        ax.plot([m], [yd], marker=mk, ms=4.0 if mk == "o" else 3.6, mew=0.8, ls="none", zorder=3,
                mfc=(col if face == INK2 and col != INK2 else face), mec=(col if col != INK2 else edge))
        if cls not in seen:
            seen.append(cls)
        xmin, xmax = min(xmin, lo), max(xmax, hi)
        ticks.append(yd); labels.append(text); colors.append(col)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.5)
    for t, c in zip(ax.get_yticklabels(), colors):
        t.set_color(c)
    ax.tick_params(axis="y", length=0, pad=4)
    C.only_x(ax)
    ax.set_ylim(0, y)
    ax.set_xlim(math.floor(xmin * 50) / 50 - 0.004, math.ceil(xmax * 50) / 50 + 0.004)
    ax.xaxis.set_major_locator(MultipleLocator(0.02 if xmax - xmin < 0.16 else 0.04))
    C.strip0(ax, "x")
    ax.set_xlabel("change in nDCG@10", fontsize=7.5, labelpad=2)
    order = [k for k in ("useful", "nonzero_below_margin", "equivalent", "loss", "inconclusive") if k in seen]
    keys = [Line2D([0], [0], marker=MAP_CLASS[k][1], ls="none", ms=4.0 if MAP_CLASS[k][1] == "o" else 3.6, mew=0.8,
                   mfc=MAP_CLASS[k][2], mec=MAP_CLASS[k][3], label=MAP_CLASS[k][0]) for k in order]
    fig.legend(handles=keys, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=len(keys), fontsize=7, frameon=False,
               handlelength=0.9, handletextpad=0.4, columnspacing=1.2, borderaxespad=0.1)
    return write(fig, stem, cap, "The alternatives evaluated", label)


# ---------------------------------------------------------------------------------------------------- what each family depends on
DEPENDS_PAIRS = REC_DIR / "depends_pairs.parquet"     # the per-pair table, also the cache that spares the release
DEPENDS_JSON = REC_DIR / "depends.json"


def depends_methods() -> list[tuple[str, str, Path, str, str, float]]:
    """(printed name, run id, store, colour, marker, marker size) of each line, in drawing order (the system last)."""
    return [("tuned lexical matching", "lstar_headings", RECORD, INK, "s", 3.8),
            ("Qwen3-Embedding-8B alone", "qwen3_8b", RECORD, PLUM, "^", 4.6),
            ("the ranker", RANKER, FOOT, BLUE, "o", 4.2),
            ("the model alone", ALONE, FOOT, OCHRE, "v", 4.6),
            ("the system", system_id(), FOOT, GREEN, "D", 3.8)]


SHARED_WORDS = ("share of the article's distinct words that also occur in the facts, as scripts/final_results.py measure(): "
                "lower-case words [a-záéíóúüñ]+ of two letters or more, NLTK Spanish stopwords out; share = article words "
                "also in the facts / article words; pairs = d.specific_idx over footing.spec_rows() (corrected labels)")


def shared_words() -> pd.DataFrame:
    """Each cited (specific) article of the 1,217 development cases with the share of its distinct words that also occur
    in the case's facts, computed exactly as ``measure()`` of scripts/final_results.py: the footing's bundle profile
    and corrected labels, ``d.specific_idx`` over ``footing.spec_rows()``. Reads the release once; the activation patches
    this process (file paths, labels), which is why F6_16 comes last in FIGS and its result is cached."""
    import os
    for k, v in (("UOL_FINAL_THREADS", "4"), ("OMP_NUM_THREADS", "4"), ("UOL_FINAL_LABELS", "corrected"),
                 ("UOL_FINAL_SYSTEMS", str(RUN / "systems_audit.json"))):
        os.environ.setdefault(k, v)
    from nltk.corpus import stopwords
    from uol.embedding_push import common as ep
    from uol.final import audit_d181_dev as D
    from uol.final import footing as FT
    D.activate(FT.STORE)
    d = ep.release()
    if d.candidate_texts is None:
        raise RuntimeError("the release was loaded without article texts")
    stop = set(stopwords.words("spanish"))
    word = re.compile(r"[a-záéíóúüñ]+")

    def distinct(t: str) -> set[str]:
        return {w for w in word.findall(t.lower()) if w not in stop and len(w) > 1}

    out, arts = [], {}
    for r in (int(x) for x in FT.spec_rows()):
        q = d.queries[r]
        fw = distinct(q.text)
        for p in d.specific_idx[r]:
            aw = arts.setdefault(int(p), distinct(d.candidate_texts[int(p)]))
            out.append((q.query_id, int(p), len(aw & fw) / len(aw) if aw else 0.0))
    return pd.DataFrame(out, columns=["query_id", "article", "share"])


def pair_hits(pc: pd.DataFrame, pairs: pd.DataFrame, k: int = 20) -> np.ndarray:
    """1.0 where the run ranks the cited pair among its first ``k`` (its stored rank), else 0.0; every pair must be there."""
    rank = {(q, a): r for q, i, rr in zip(pc["query_id"], pc["positive_idx"], pc["positive_ranks"])
            for a, r in zip(F._ints(i), F._ints(rr))}
    return np.array([1.0 if 0 < rank[(q, int(a))] <= k else 0.0 for q, a in zip(pairs["query_id"], pairs["article"])])


def depends_pairs() -> tuple[pd.DataFrame, np.ndarray]:
    """The per-pair table: every cited pair of the development cases with its share of shared words (from the cache,
    else from the release), its quarter (1 = fewest shared words), its fold-local citation band from the record's C3
    pairs (c3/c3_pairs.csv, as F6_7), and whether each method ranks it among the first twenty. Returns the table and the
    quarter edges (minimum, three cuts, maximum)."""
    c3 = pd.read_csv(F.RECORD_ANALYSIS / "c3" / "c3_pairs.csv", dtype={"band": str})
    if DEPENDS_PAIRS.exists():
        p = pd.read_parquet(DEPENDS_PAIRS, columns=["query_id", "article", "share"])
    else:
        p = shared_words()
    p = p.merge(c3[["query_id", "article", "component_id", "band"]], on=["query_id", "article"], how="left",
                validate="one_to_one")
    if len(p) != len(c3) or p["band"].isna().any():
        raise RuntimeError(f"the cited pairs ({len(p)}) are not the record's C3 pairs ({len(c3)})")
    quarter, edges = pd.qcut(p["share"], 4, labels=False, retbins=True)
    p.insert(4, "quarter", quarter.astype(int) + 1)
    for _, sid, store, *_ in depends_methods():
        p[f"hit_{sid}"] = pair_hits(pd.read_parquet(store / sid / "DEV" / "per_case.parquet"), p)
    p = p[["query_id", "article", "component_id", "share", "quarter", "band"] + [c for c in p.columns if c.startswith("hit_")]]
    p.to_parquet(DEPENDS_PAIRS, index=False)
    return p, edges


def _pct1(v: float) -> str:
    return "0" if v <= 0 else f"{100 * v:.1f}"


def depends() -> dict:
    """Recall@20 per cited pair of each method by quarter of shared words (a) and by citation band (b): pooled over
    the pairs of a bin, with the cluster bootstrap over groups of C3 and F6_7 (uol.analysis_c.common.Boot, 2,000 draws,
    seed 20261023, the groups of the record's development cases). Written to depends.json."""
    from uol.analysis_c.common import Boot
    p, edges = depends_pairs()
    base = pd.read_parquet(RECORD / "S0" / "DEV" / "per_case.parquet")
    boot = Boot(base["component_id"].to_numpy(), 2000, 20261023)
    g = boot.group_index(p["component_id"])

    def ratio(hit: np.ndarray, m: np.ndarray) -> dict:
        r = boot.ratio(np.where(m, hit, 0.0), m.astype(float), g)
        return {"mean": r["mean"], "lo": r["lo"], "hi": r["hi"], "n": int(m.sum())}

    # the same arithmetic must give the record's own row of the ranker without the encoder (C3's figdata), values and
    # intervals, as F6_7 requires before it adds a row
    br = pd.read_csv(F.RECORD_ANALYSIS / "figdata" / "band_recall.csv", dtype={"band": str})
    ref = br[br["system"] == "S0"].set_index("band").loc[F.BANDS, ["recall_at_20", "lo", "hi"]].to_numpy(float)
    s0 = pair_hits(base, p)
    got = [[(r := ratio(s0, (p["band"] == b).to_numpy()))["mean"], r["lo"], r["hi"]] for b in F.BANDS]
    if not np.allclose(np.asarray(got), ref, atol=1e-5):
        raise RuntimeError(f"the pair bootstrap does not reproduce C3's row of the ranker without the encoder: {got} vs {ref}")
    panels = {"a": {"by": "quarter of the cited pairs by the share of shared words", "column": "quarter",
                    "bins": [1, 2, 3, 4],
                    "labels": [f"{_pct1(lo)}–{_pct1(hi)}%" for lo, hi in zip(edges[:-1], edges[1:])]},
              "b": {"by": "times the fold's training cases had cited the article (C3's fold-local band)", "column": "band",
                    "bins": list(F.BANDS), "labels": [F.BAND_NAME[b] for b in F.BANDS]}}
    for pan in panels.values():
        pan["methods"] = {}
        for name, sid, store, *_ in depends_methods():
            hit = p[f"hit_{sid}"].to_numpy(float)
            pan["methods"][name] = {"run": sid, "store": C.rel(store),
                                    "values": {str(b): ratio(hit, (p[pan["column"]] == b).to_numpy()) for b in pan["bins"]}}
        pan["bins"] = [str(b) for b in pan["bins"]]
    out = {"figure": "F6_16_depends", "measure": "Recall@20 per cited pair: share of the cited pairs of a bin whose article "
                                                 "the method ranks among its first twenty (rank from per_case.parquet)",
           "pairs": int(len(p)), "cases": int(p["query_id"].nunique()), "groups": int(p["component_id"].nunique()),
           "definition": SHARED_WORDS + ". Quarters: pandas qcut of that share over the cited pairs (edges below, each "
                                        "upper edge inclusive). Bands: c3/c3_pairs.csv of the record (cited_other_folds).",
           "quarter_edges": [float(e) for e in edges[1:-1]], "share_min": float(edges[0]), "share_max": float(edges[-1]),
           "bootstrap": "pooled ratio sum(hits) / sum(pairs) per bin; 95% interval from a cluster bootstrap over groups "
                        "(uol.analysis_c.common.Boot, 2,000 draws, seed 20261023), as C3 and F6_7",
           "check": "the same code reproduces figdata/band_recall.csv's row of the ranker without the encoder (atol 1e-5)",
           "pairs_table": C.rel(DEPENDS_PAIRS), "panels": panels}
    DEPENDS_JSON.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    return out


def f_depends() -> Path:
    stem, label = "F6_16_depends", "fig:F6_16_depends"
    cap = ("Share of the cited articles found among the first twenty on the development cases, per cited pair, with 95\\% "
           "intervals: (a) by the share of the article's distinct words that also occur in the facts, in quarters of the "
           "cited pairs; (b) by how often the fold's training cases had cited the article.")
    dep = depends()
    methods = depends_methods()
    h = 2.75
    fig = new(h)
    left, gap, right, bottom, height = 0.45, 0.2, 0.04, 0.62, 1.66        # inches
    wp = (W - left - gap - right) / 2
    axes = [fig.add_axes([left / W, bottom / h, wp / W, height / h])]
    axes.append(fig.add_axes([(left + wp + gap) / W, bottom / h, wp / W, height / h], sharey=axes[0]))
    offs = np.linspace(-0.24, 0.24, len(methods))           # side by side within a bin, so no interval hides another
    # the quarters' ranges in percent; the unit sits in the title, since four "%" signs crowd the ticks together
    titles = {"a": "(a) share of the article's words that\nalso occur in the facts, in percent",
              "b": "(b) times the fold's training cases\nhad cited the article"}
    ticks = {"a": [s.rstrip("%") for s in dep["panels"]["a"]["labels"]], "b": dep["panels"]["b"]["labels"]}
    for ax, key in zip(axes, ("a", "b")):
        pan = dep["panels"][key]
        x = np.arange(len(pan["bins"]))
        for (name, _, _, col, mk, ms), dx in zip(methods, offs):
            v = [pan["methods"][name]["values"][b] for b in pan["bins"]]
            ax.vlines(x + dx, [e["lo"] for e in v], [e["hi"] for e in v], color=col, lw=0.8, zorder=2)
            ax.plot(x + dx, [e["mean"] for e in v], color=col, lw=1.0, zorder=3)
            ax.plot(x + dx, [e["mean"] for e in v], ls="none", marker=mk, ms=ms, mfc=col, mec="white", mew=0.5,
                    zorder=4)                                # every marker above every line
        ax.set_xticks(x)
        ax.set_xticklabels(ticks[key])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(x) - 0.5)
        ax.set_xlabel(titles[key], fontsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylim(0, 1)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
    C.strip0(axes[0], "y")
    axes[0].set_ylabel(C.axis_title("Recall@20"))
    plt.setp(axes[1].get_yticklabels(), visible=False)
    # one key for both panels, above them, in reading order across two rows (matplotlib fills columns first)
    hs = [Line2D([0], [0], color=col, lw=1.0, marker=mk, ms=ms, mfc=col, mec="white", mew=0.5, label=name)
          for name, _, _, col, mk, ms in methods]
    order = [0, 3, 1, 4, 2]                                  # rows: lexical, encoder, ranker / model alone, system
    mid = (left + wp + gap / 2) / W                          # centre of the pair of panels, in figure fraction
    leg = F._legend(axes[0], [hs[i] for i in order], "lower center", ncol=3, columnspacing=1.4, handlelength=2.0,
                    bbox_to_anchor=((mid - left / W) / (wp / W), 1.03))
    fig.canvas.draw()
    lb, ab = leg.get_window_extent(fig.canvas.get_renderer()), axes[0].get_window_extent()
    if lb.y0 < ab.y1 or lb.x0 < 0 or lb.x1 > fig.bbox.x1:
        raise ValueError(f"F6_16: the key overlaps the panels or leaves the page ({lb}, axes top {ab.y1})")
    NOTES[stem] = [f"pairs {dep['pairs']:,}, cases {dep['cases']:,}, groups {dep['groups']:,}; quarter edges "
                   + ", ".join(f"{e:.3f}" for e in dep["quarter_edges"]) + f" (share {dep['share_min']:.3f} to "
                   f"{dep['share_max']:.3f}); the pair bootstrap reproduces C3's row of the ranker without the encoder"]
    for key in ("a", "b"):
        pan = dep["panels"][key]
        for name, m in pan["methods"].items():
            NOTES[stem].append(f"({key}) {name} [{m['run']}]: " + "; ".join(
                f"{lab} {m['values'][b]['mean']:.3f} [{m['values'][b]['lo']:.3f}, {m['values'][b]['hi']:.3f}] n={m['values'][b]['n']}"
                for b, lab in zip(pan["bins"], pan["labels"])))
    return write(fig, stem, cap, "What each family depends on", label)


def f5() -> str:
    from uol.thesis_figures import f5_diagrams
    f5_diagrams.main()
    return "F5_arch, F5_arch_float, F5_2_llm_windows"


FIGS = {"F5": f5, "F6_who": f_who, "F6_cells": f_cells, "F6_models": f_models,   # v33 (R6 audit): F6_alone_list cut, one clause in 6.6 keeps its point
        "F6_6": f_ranker_gain, "F6_8": f_law_article,
        "F6_9": f_llm_stage, "F6_10": f_frontier, "F6_15": f_llm_alone, "F6_13": f_map,
        "F6_16": f_depends}                 # last: without its cache it activates the footing profile in this process


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    want = [x for x in a.only.split(",") if x] or list(FIGS)
    for k in want:
        print(k, FIGS[k](), NOTES.get({"F6_6": "F6_6_ranker_gain", "F6_8": "F6_8_law_article", "F6_9": "F6_9_llm_stage",
                                         "F6_10": "F6_10_frontier", "F6_15": "F6_15_llm_alone", "F6_13": "F6_13_map",
                                         "F6_16": "F6_16_depends"}.get(k, k), ""),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
