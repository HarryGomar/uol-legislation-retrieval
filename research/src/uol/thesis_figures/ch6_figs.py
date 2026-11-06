"""Chapter 6 figures in the sprint style (26 Oct 2026; STANDARD.md sections 5-6; SPRINT_IMPROVEMENT section 9; D125).

The pipeline reads only saved outputs, so the freeze refreshes the data and nothing else:

* the run store, ``<store>/<system>/<population>/{summary.json, per_case.parquet}`` (the registered record's merged
  store, ``experiments/runs/final_20261025/store_dev_20261025b/runs``, D131; no ``.cache`` store is read any more);
* the tidy outputs of RESULTS' analyses (``uol.analysis_c``), ``<analysis>/<item>/*.csv``, and the figure data that
  RESULTS wrote beside them (``<analysis>/figdata/*.csv``; the record's ``analysis_c_20261025b``).

Every plotted value and interval is read from the record (the store's summaries, ``c1/c1_systems.csv``,
``figdata/*.csv``, ``c8/c8_quantities.csv``, the contrasts); the figures compute nothing beyond unit conversions (bytes
of stored vectors), with one exception (D181, 28 Oct 2026): the band recall of the integrated method in 6.2, which
figdata lacks, is read from the record's C3 pairs and stored ranks by C3's own bootstrap, after the same code
reproduces the ranker without embeddings' row exactly (``_band_recall_from_pairs``). Systems that the record lacks are skipped and listed in the figure's notes. While the store is
RUN-A's mirror, three final rankers are stood in for by RUN-A's frozen-source rankers (D77), and any stand-in sets a
small grey "DRY RUN" mark that the freeze run (``--final``) drops. A figure whose data are missing is written as a
framed ``\\EvPending`` placeholder with its caption.

Output, per figure: ``thesis/figures/F6_<n>_<name>.pdf`` and ``.tex`` (a ``figure`` environment, ``[H]``, caption and
label). Build numbers of 25 Oct: 6.1 F6_6 (ranker gain), 6.2 F6_7 (bands), 6.3 F6_8 (law and article), 6.4 F6_1
(each family alone), 6.5 F6_2 (reach), 6.6 F6_3 (encoders), 6.7 F6_4 (stored vectors), 6.8 F6_11 (recall by depth),
6.9 F6_9 (LLM stage), 6.10 F6_5 (hybrids), 6.11 F6_10 (cost). Names follow the term sheet (common.TERMS); ITAM green
marks the integrated method, the proposed system, and nothing else; every axis states its measure and its cases; each
saved figure passes common.pdf_audit (7 pt, no overlap, nothing off the page, no Type 3) and common.term_lint.

    py -3.12 -m uol.thesis_figures.ch6_figs --final [--only F6_1,...]      # the final record (defaults)
    py -3.12 -m uol.thesis_figures.ch6_figs --store DIR --analysis DIR [--extra-contrasts DIR]
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from uol.thesis_figures import common as C

OUT_DIR = C.ROOT / "thesis" / "figures"
# The registered, read-only final record (D131, session D, 25 Oct 2026): the defaults of build() and the command line.
# The RUN-A mirror .cache/final_work/run_a_runs is no longer a default anywhere (ORCH, 25 Oct), so D can archive it.
RECORD = C.ROOT / "experiments" / "runs" / "final_20261025"
RECORD_STORE, RECORD_ANALYSIS = RECORD / "store_dev_20261025b" / "runs", RECORD / "analysis_c_20261025b"
RECORD_CONTRASTS = RECORD / "drive" / "raw" / "contrasts"
# D138 (session D, 25 Oct 2026): the same contrasts with 100,000 bootstrap draws, so the Bonferroni-adjusted lower ends
# rest on enough draws; read first wherever a figure prints a class or an adjusted bound (only 6.1, F6_6).
RECORD_100K = RECORD / "supplement" / "bootstrap_100k" / "contrasts"
GREEN, TINT, INK, INK2, MUTED, RULE, MARK = C.ITAM_GREEN, C.FOCUS_TINT, C.INK, C.INK2, C.MUTED, C.RULE, "#a6a6a6"
BLUE, PLUM, OCHRE = C.COMPANION_BLUE, C.COMPANION_PLUM, C.COMPANION_OCHRE      # scheme P3 (D147)
W = C.TEXTWIDTH_IN
STAND_IN = {"S0": "ltr_noemb_frozen", "S_full": "ltr_emb_frozen", "S_full_ext": "ltr_emb_frozen"}
# D185 (28 Oct 2026): the integrated method is the lean ranker, the ranker without embeddings plus Qwen3-Embedding-8B
# and its past-case sources; S_full, which adds BGE-M3, E5 and late interaction, is a tested alternative.
INTEGRATED, EVERY = "S0+qwen3_8b", "S_full"
# PLAN addendum 22: the ranker (8 sources) and the ranker without the encoder (5 sources), in the footing's store
FOOT_STORE = C.ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs"
RANKER8, NOENC5 = "rt_lm_tesis", "rk_noenc"
# D188 (30 Oct 2026): the system is the ranker (INTEGRATED) with the reread, Claude Opus 5.5 over its first 30. Its record
# and the recomputes on it are outside the 25 Oct store, in the run of the final-system audit.
SYSTEM = "claude_rerank_S0+qwen3_8b"
FINAL_DEV = C.ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev"
FINAL_STORE, FINAL_RECOMPUTE = FINAL_DEV / "store" / "runs", FINAL_DEV / "recompute"
# its development contrasts (record contrast code, 100,000 draws; not in the 25 Oct record)
D185_CONTRASTS = C.ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev_d185" / "contrasts_100k.json"
# Rows held as not valid until RUN-B reruns them (RESULTS, 23 Oct 2026): RUN-A results from before sequence 7's reset,
# which carry the old lda_sim read-out inside S0 (D78), the damaged e5_large_instruct store, and three systems pending a
# store check. A held system is treated as absent: figures skip it and the evidence macros print \EvPending.
HOLD_PENDING_CHECK = {"arctic_l_v2", "e5_large", "gemma_legal"}
HOLD = ({f"S0+{x}" for x in ("alia_mrbert", "arctic_l_v2", "bge_m3", "bge_m3_base", "bge_m3_legal", "bge_m3_sparse",
                              "e5_base", "e5_large", "e5_large_legal_en", "e5_small", "euler_legal", "ft_e5_q",
                              "gemma_legal", "gte_multilingual", "jina_v3", "query2doc", "qwen3_8b", "qwen3_8b_binary",
                              "qwen3_8b_int8", "qwen3_8b_mrl256", "qwen3_8b_mrl512", "qwen3_8b_mrl1024",
                              "e5_large_instruct")}
        | {"rrf__lstar_headings__lda_sim", "linear__lstar_headings__lda_sim", "e5_large_instruct",
           "rrf__lstar_headings__e5_large_instruct", "linear__lstar_headings__e5_large_instruct"}
        | HOLD_PENDING_CHECK
        | {f"{k}__lstar_headings__{x}" for k in ("rrf", "linear") for x in HOLD_PENDING_CHECK})


def held(system: str) -> bool:
    return system in HOLD
SEED = 20261023
N_BOOT = 1000

# Display names: the term sheet (common.TERMS), plus the few systems only this module prints.
NAME = {**C.TERMS, "random": "random order", "hyde": "HyDE", "query2doc": "query2doc", "c_hybrid_cached": "LLM stage",
        "qwen8b_adapter": "Qwen3-Embedding-8B, task-trained adapter",
        # D181 (audit of v10, 28 Oct 2026): the integrated method is the ranker with embeddings; the older tesis and the
        # LLM stage are exploratory extensions. common.TERMS keeps the older names for the other figure modules.
        INTEGRATED: "the ranker", SYSTEM: "the system", EVERY: "ranker with every encoder", "S_full_ext": "largest ranker",
        "llm_rerank_S_full_ext": "GPT-5.6 reread over the largest ranker",
        "llm_rerank_S0": "GPT-5.6 reread over the ranker without embeddings"}
# One colour per system in every Ch6 figure (D147, D181): the integrated method ITAM green, the ranker without
# embeddings slate blue, dense retrievers plum, language models ochre; the exploratory extended ranker and every
# other system grey (context).
SYS_COLOUR = {INTEGRATED: C.ITAM_GREEN, SYSTEM: C.ITAM_GREEN, EVERY: C.INK2, "S0": C.COMPANION_BLUE, "qwen3_8b": C.COMPANION_PLUM,
              "llm_rerank_S_full_ext": C.COMPANION_OCHRE, "S_full_ext": C.INK2, "lstar_headings": C.INK}
FAMILY = {  # family of a single source, for marker shapes (6.5)
    "lexical": ("o", ["bm25_stem", "lstar_headings"]),
    "one collection": ("s", ["lsa", "lda_sim", "lda_dm", "doc2vec", "word2vec"]),
    "pre-trained encoders": ("^", ["e5_small", "e5_base", "e5_large", "e5_large_instruct", "gte_multilingual",
                                   "jina_v3", "arctic_l_v2", "gemma_base", "bge_m3", "bge_m3_sparse", "bge_m3_colbert",
                                   "qwen3_0p6b", "qwen3_4b", "qwen3_8b", "e5_large_legal_en", "bge_m3_legal",
                                   "gemma_legal", "euler_legal", "alia_mrbert"]),
    "task-trained encoders": ("v", ["ft_e5_q", "bge_m3_ft", "qwen3_0p6b_ft", "qwen8b_adapter"]),
    "past citations": ("D", ["docexp_bm25", "src_docexp_facts_bm25"]),   # and every src_knn_*, src_proto_*, src_K*,
    # src_H* single source; the two BM25 sources read the citing facts, so they sit here as in figdata/family_alone
}
PARAMS_M = {  # parameters in millions: Qwen3 sizes from zhang2025qwen3; the others from systems.json notes / model cards
    "e5_small": 118, "e5_base": 278, "e5_large": 560, "e5_large_instruct": 560, "gte_multilingual": 305,
    "jina_v3": 570, "arctic_l_v2": 568, "gemma_base": 300, "bge_m3": 568, "qwen3_0p6b": 600, "qwen3_4b": 4000,
    "qwen3_8b": 8000, "e5_large_legal_en": 560, "bge_m3_legal": 568, "gemma_legal": 300, "euler_legal": 8000,
    "alia_mrbert": 150,
}
# Only legal models trained from the same base (audit of v7, 28 Oct): e5-large-legal is an English model, not a
# fine-tune of multilingual E5-large, so it is left out of the pairs and stays in the table of all systems.
# The legal 8B is stored in eight bits, so its base is the general 8B in eight bits, as in the text (\cdLegEuler; D176).
LEGAL_PAIRS = [("bge_m3", "bge_m3_legal"), ("qwen3_8b_int8_base", "euler_legal"), ("gemma_base", "gemma_legal")]
TASK_PAIRS = [("e5_small", "ft_e5_q"), ("bge_m3", "bge_m3_ft"), ("qwen3_0p6b", "qwen3_0p6b_ft")]
DIMS = {"qwen3_0p6b": 1024, "qwen3_4b": 2560, "qwen3_8b": 4096}
N_ART = 39178
# Classes of the comparison rule as 4.4 names them (session A, 25 Oct), and their marker in the dot plots.
CLASS_NAME = {"useful": "clears the margin", "nonzero_below_margin": "a real gain, not shown\nto reach the margin",
              "equivalent": "no meaningful difference"}   # D167: the margin's value is stated once, in the caption
CLASS_MARK = {"useful": dict(mfc=INK2, mec=INK2), "nonzero_below_margin": dict(mfc=MARK, mec=MARK),
              "equivalent": dict(mfc="white", mec=INK2)}


@dataclass
class Ctx:
    store: Path
    analysis: Path
    final: bool
    notes: dict = field(default_factory=dict)
    used_stand_in: set = field(default_factory=set)
    extra_contrasts: Path | None = None

    def held(self, system: str) -> bool:
        """The hold list guards the RUN-A-era stores; the final store (D108 reruns merged, stale rows excluded) has
        no held rows (25 Oct 2026)."""
        return (not self.final) and held(system)

    @property
    def dry(self) -> bool:
        return (not self.final) or bool(self.used_stand_in)

    # ---- data access -------------------------------------------------------------------------------------------
    def sid(self, system: str) -> str | None:
        """Store id for a system; uses a stand-in (and records it) when the store lacks the system; None if held."""
        if self.held(system):
            return None
        if system == SYSTEM:
            return system if (FINAL_STORE / system / "DEV" / "summary.json").exists() else None
        if (self.store / system / "DEV" / "summary.json").exists():
            return system
        alt = STAND_IN.get(system)
        if alt and (self.store / alt / "DEV" / "summary.json").exists():
            self.used_stand_in.add(f"{system}={alt}")
            return alt
        return None

    def summary(self, system: str, pop: str = "DEV") -> dict | None:
        s = self.sid(system)
        if s is None:
            return None
        p = (FINAL_STORE if s == SYSTEM else self.store) / s / pop / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def per_case(self, system: str, pop: str = "DEV") -> pd.DataFrame | None:
        s = self.sid(system)
        if s is None:
            return None
        p = (FINAL_STORE if s == SYSTEM else self.store) / s / pop / "per_case.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def systems(self) -> list[str]:
        return sorted(s for s in (p.parent.parent.name for p in self.store.glob("*/DEV/summary.json")) if not self.held(s))

    def csv(self, rel: str) -> pd.DataFrame | None:
        p = self.analysis / rel
        if not p.exists() or p.stat().st_size < 5:
            return None
        return pd.read_csv(p)

    def c1(self, system: str) -> pd.Series | None:
        """The system's row of RESULTS' C1 table (every measure with its 95% interval), or None."""
        if not hasattr(self, "_c1"):
            self._c1 = self.csv("c1/c1_systems.csv")
            self._c1 = pd.concat([self._c1, pd.read_csv(FINAL_RECOMPUTE / "c1" / "c1_systems.csv").query("system == @SYSTEM")],
                                 ignore_index=True)
        sid = self.sid(system)
        if self._c1 is None or sid is None:
            return None
        r = self._c1[self._c1["system"] == sid]
        return None if r.empty else r.iloc[0]


# ---- statistics (kept for the dry-run stand-ins; the final figures read intervals from the record) ----------------
def _boot_mean(df: pd.DataFrame, col: str, rng) -> tuple[float, float, float]:
    """Mean of a per-case column with a 95% cluster bootstrap over groups (component_id)."""
    g = df.groupby("component_id")[col].agg(["sum", "count"])
    s, n = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
    means = s[idx].sum(1) / n[idx].sum(1)
    return float(s.sum() / n.sum()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_gain(a: pd.DataFrame, b: pd.DataFrame, col: str = "ndcg_at_10") -> tuple[float, float, float] | None:
    """b minus a on the cases both scored, with a 95% paired cluster bootstrap over groups."""
    m = a[["query_id", "component_id", col]].merge(b[["query_id", col]], on="query_id", suffixes=("_a", "_b"))
    if m.empty:
        return None
    m["d"] = m[f"{col}_b"] - m[f"{col}_a"]
    return _boot_mean(m, "d", np.random.default_rng(SEED))


def mean_ci(df: pd.DataFrame, col: str = "ndcg_at_10") -> tuple[float, float, float]:
    return _boot_mean(df.dropna(subset=[col]), col, np.random.default_rng(SEED))


def ndcg(ctx: Ctx, system: str) -> tuple[float, float, float] | None:
    """nDCG@10 and its 95% interval from the record (C1), else the store's point value with no interval."""
    r = ctx.c1(system)
    if r is not None and pd.notna(r["ndcg_at_10"]):
        return float(r["ndcg_at_10"]), float(r["ndcg_at_10_lo"]), float(r["ndcg_at_10_hi"])
    sm = ctx.summary(system)
    if sm and sm.get("ndcg_at_10") is not None:
        v = float(sm["ndcg_at_10"])
        return v, np.nan, np.nan
    return None


# ---- output -------------------------------------------------------------------------------------------------------
def _new(h_in: float, w_in: float = W):
    C.style()
    fig = plt.figure(figsize=(w_in, h_in))
    return fig


def _mark(fig, ctx: Ctx) -> None:
    if ctx.dry:
        fig.text(.995, .995, "DRY RUN", ha="right", va="top", fontsize=7, color=MUTED)


# Figures that float (L5 set these to [tbp] by hand on 26 Oct; the generator writes the same line, so regenerating
# keeps it). The others stay [H] at their paragraph boundary.
FLOATS = {"F6_1_alone", "F6_3_encoders", "F6_4_compression", "F6_5_hybrids", "F6_6_ranker_gain", "F6_7_bands",
          "F6_8_law_article", "F6_9_llm_stage", "F6_10_frontier", "F6_3b_training"}   # 6.3b: L5 night, INBOX
# A float of just over half a page is set alone on a float page with [tbp]; these take [tb] instead (CF-03, 26 Oct).
TOP_OR_BOTTOM = set(FLOATS)   # every Ch6 float since LF overnight 26-27 Oct: four more stood alone on half-empty pages


def _write(fig, ctx: Ctx, stem: str, caption: str, short: str, label: str, width: str = r"\textwidth") -> Path:
    """Save the figure and its .tex; check it first (nothing under an axis) and after (pdf_audit, term sheet)."""
    _mark(fig, ctx)
    C.nothing_under_axes(fig)
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)
    audit = C.pdf_audit(pdf)
    C.term_lint(audit["text"], f"{stem} figure text")
    C.term_lint(caption + " " + short, f"{stem} caption")
    ctx.notes.setdefault(stem, []).append(f"render audit: smallest text {audit['min_pt']:.1f} pt, "
                                          f"{audit['w_cm']:.2f} x {audit['h_cm']:.2f} cm")
    note = ctx.notes.get(stem, [])
    lines = [
        f"% {stem}: generated by uol.thesis_figures.ch6_figs; do not edit by hand. Store: {C.rel(ctx.store)};",
        f"% analyses: {C.rel(ctx.analysis)}. {'DRY RUN: stand-in data, the freeze regenerates this figure.' if ctx.dry else 'Final data.'}",
        "% Caption: draft of session C (sprint 26 Oct); the chapter's owner agrees the final wording.",
    ]
    if ctx.used_stand_in:
        lines.append("% Stand-ins: " + ", ".join(sorted(ctx.used_stand_in)) + ".")
    lines += [f"% Note: {n}" for n in note]
    lines += [
        ("\\begin{figure}[tb]% top or bottom of a text page, never a float page alone (LF, CF-03, 26 Oct)"
         if stem in TOP_OR_BOTTOM else
         "\\begin{figure}[tbp]% floats (ORCH and L5, 26 Oct: [H] left half-empty pages in Ch6); was [H] at its "
         "paragraph boundary (C143)" if stem in FLOATS else "\\begin{figure}[H]% set at its paragraph boundary (C143)"),
        "  \\centering",
        f"  \\includegraphics[width={width}]{{{stem}.pdf}}",
        f"  \\caption[{short}]{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{figure}",
        "",
    ]
    (OUT_DIR / f"{stem}.tex").write_text("\n".join(lines), encoding="utf-8")
    return pdf


def _placeholder(ctx: Ctx, stem: str, caption: str, short: str, label: str, why: str) -> Path:
    tex = OUT_DIR / f"{stem}.tex"
    tex.write_text("\n".join([
        f"% {stem}: generated by uol.thesis_figures.ch6_figs. Placeholder: {why}",
        "\\begin{figure}[H]% set at its paragraph boundary (C143)",
        "  \\centering",
        f"  \\fbox{{\\begin{{minipage}}[c][3cm][c]{{0.9\\textwidth}}\\centering\\EvPending{{figure: {why}}}\\end{{minipage}}}}",
        f"  \\caption[{short}]{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{figure}",
        "",
    ]), encoding="utf-8")
    return tex


def _family_of(system: str) -> str | None:
    for fam, (_, ids) in FAMILY.items():
        if system in ids:
            return fam
    if system.startswith(("src_knn", "src_proto", "src_K", "src_H", "src_LP")):
        return "past citations"
    return None


def _label(ax, x, y, text, color=INK2, dx=4, dy=0, ha="left", va="center", size=7, **kw):
    return ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=size,
                       color=color, **kw)


def _logfmt(ax, which: str = "x") -> None:
    """Plain numbers on a log axis (0.001, 1, 10, 1,000) instead of powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    axis = ax.xaxis if which == "x" else ax.yaxis
    axis.set_major_locator(LogLocator(base=10))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:.10f}".rstrip("0")))
    axis.set_minor_formatter(NullFormatter())


def _legend(ax, handles, loc, **kw):
    for k, v in dict(fontsize=7, frameon=False, handlelength=1.5, handletextpad=0.5, borderaxespad=0.3,
                     labelspacing=0.35).items():
        kw.setdefault(k, v)
    return ax.legend(handles=handles, loc=loc, **kw)


def _dot(ax, x, y, cls: str | None = None, color: str | None = None, ms: float = 4.2, marker: str = "o", z: int = 3):
    st = dict(CLASS_MARK.get(cls, dict(mfc=INK2, mec=INK2)))
    if color:       # the class keeps its fill (full, light, open); the row's colour replaces the grey
        st = {"useful": dict(mfc=color, mec=color), "nonzero_below_margin": dict(mfc=C.tint(color), mec=color),
              "equivalent": dict(mfc="white", mec=color)}.get(cls, dict(mfc=color, mec=color))
    ax.plot([x], [y], marker=marker, ms=ms, ls="none", mew=0.8, zorder=z, **st)


def _rows_axes(ax, ys, labels, colors=None) -> None:
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    if colors:
        for t, c in zip(ax.get_yticklabels(), colors):
            t.set_color(c)
    C.only_x(ax)


# ---- 6.4 (F6_1): each family alone ---------------------------------------------------------------------------------
F61_FAMILIES = {  # the families of RESULTS' figdata/family_alone.csv (scripts/figdata_runb_dev.py reads this dict)
    "lexical": ["bm25_stem", "lstar_headings"],
    "one collection": ["lsa", "lda_sim", "lda_dm", "doc2vec", "word2vec"],
    "general embedding": ["e5_small", "e5_base", "e5_large", "gte_multilingual", "jina_v3",
                          "arctic_l_v2", "gemma_base", "bge_m3", "bge_m3_sparse", "bge_m3_colbert", "qwen3_0p6b",
                          "qwen3_4b", "qwen3_8b"],
    "legal embedding": ["e5_large_legal_en", "bge_m3_legal", "gemma_legal", "euler_legal", "alia_mrbert"],
    "task-trained": ["ft_e5_q", "bge_m3_ft", "qwen3_0p6b_ft"],
    "past citations": ["src_K_lex", "src_K_lex_ext", "src_proto_tfidf", "docexp_bm25", "src_docexp_facts_bm25",
                       "src_knn_lex_votes", "src_knn_char_votes"],   # keyed on words: practice without an encoder
}
F61_ROWS = [  # rows of the figure (SPRINT_IMPROVEMENT section 9): every member a dot, the best named with its interval
    ("lexical matching", ["bm25_stem", "lstar_headings"]),
    ("one collection", ["lsa", "lda_sim", "lda_dm", "doc2vec", "word2vec"]),
    ("pre-trained encoders", ["e5_small", "e5_base", "e5_large", "e5_large_instruct", "gte_multilingual", "jina_v3",
                              "arctic_l_v2", "gemma_base", "bge_m3", "bge_m3_sparse", "bge_m3_colbert", "qwen3_0p6b",
                              "qwen3_4b", "qwen3_8b", "e5_large_legal_en", "bge_m3_legal", "gemma_legal",
                              "euler_legal", "alia_mrbert"]),
    ("task-trained encoders", ["ft_e5_q", "bge_m3_ft", "qwen3_0p6b_ft", "qwen8b_adapter"]),
    ("past citations", ["src_proto_tfidf", "docexp_bm25", "src_docexp_facts_bm25",
                        "src_knn_lex_votes", "src_knn_char_votes", "src_knn_qwen8b", "src_proto_qwen8b"]),
    ("language model", ["gpt61sol_alone"]),
]
# Rebuild (2 Nov 2026): the model alone is GPT-6.1 Sol, the system's model, read from the footing's store, not the
# record's llm_alone (GPT-5.6 Luna); its 95% interval is a bootstrap over groups, as in uol.thesis_figures.v21_figs.
FOOT_STORE = C.ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs"


def foot_ndcg(system: str, n: int = 2000, seed: int = 20261103) -> tuple[float, float, float] | None:
    p = FOOT_STORE / system / "DEV" / "per_case.parquet"
    if not p.exists():
        return None
    gm = pd.read_parquet(p).groupby("component_id")["ndcg_at_10"].mean().to_numpy()
    draws = gm[np.random.default_rng(seed).integers(0, len(gm), size=(n, len(gm)))].mean(1)
    return float(gm.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


F61_COLOUR = {"pre-trained encoders": PLUM, "task-trained encoders": PLUM, "past citations": BLUE,
              "language model": OCHRE}      # D181: green is the integrated method only; past citations take the
# ranker without embeddings' blue (practice), the encoders plum, the language model ochre; lexical and one collection ink
# The rows as kinds within the four families of 1.3 (cold read item 2, ORCH, 27 Oct): (family heading, [(row, label)]).
# The models of one collection read only the words of the catalogue and the cases, so they sit under lexical matching.
F61_GROUPS = [("lexical matching", [("lexical matching", "word matching"), ("one collection", "learned from the collection")]),
              ("dense retrievers", [("pre-trained encoders", "pre-trained"), ("task-trained encoders", "task-trained")]),
              ("past citations", [("past citations", "similar past cases")]),
              ("large language models", [("language model", "naming articles from the facts")])]


def f6_1(ctx: Ctx) -> Path:
    """Each family alone: one row per family, every member a grey dot, the best named with its 95% interval."""
    stem, label = "F6_1_alone", "fig:F6_1_alone"
    cap = "Ranking quality of each kind of method used alone on the development cases. The dashed line is tuned lexical matching."
    rows = []
    for fam, ids in F61_ROWS:
        vals = [(s, ndcg(ctx, s) or foot_ndcg(s)) for s in ids]
        miss = [s for s, v in vals if v is None]
        if miss:
            ctx.notes.setdefault(stem, []).append(f"{fam}: not in the record: {', '.join(miss)}")
        vals = [(s, v) for s, v in vals if v is not None]
        if vals:
            rows.append((fam, vals, max(vals, key=lambda sv: sv[1][0])))
    if not rows:
        return _placeholder(ctx, stem, cap, "Each family alone", label, "no single-source results")
    by_fam = {r[0]: r for r in rows}
    slots = []   # top to bottom: ("h", family heading) or ("r", row, label)
    for head, kinds in F61_GROUPS:
        kinds = [(f, lab) for f, lab in kinds if f in by_fam]
        if kinds:
            slots += [("h", head, None)] + [("r", f, lab) for f, lab in kinds]
    ys, y = [], 0.0
    for i, (kind, _, _) in enumerate(slots):
        if kind == "h" and i:
            y -= 0.3                                         # a gap before each family after the first
        ys.append(y)
        y -= 1
    n = len(slots)
    h = 0.27 * (-min(ys) + 1) + 0.7                          # 0.2 in above the axes for the key (H7, 2 Nov 2026)
    fig = _new(h)
    ax = fig.add_axes([0.31, 0.42 / h, 0.66, (h - 0.7) / h])
    lex = ndcg(ctx, "lstar_headings")
    row_y = {f: yy for (kind, f, _), yy in zip(slots, ys) if kind == "r"}
    if lex:   # stops below the lexical row, whose best is tuned lexical matching itself
        ax.vlines(lex[0], min(ys) - 0.5, row_y["lexical matching"] - 0.45, color=MUTED, lw=0.6, ls=(0, (3, 2)),
                  zorder=0)
    for kind, fam, _ in slots:
        if kind == "h":
            continue
        _, vals, (bs, (bm, blo, bhi)) = by_fam[fam]
        y = row_y[fam]
        for s, (m, _, _) in vals:
            if s != bs:
                ax.plot([m], [y], marker="o", ms=3.6, ls="none", mfc=MARK, mec="white", mew=0.5, zorder=2)
        col = F61_COLOUR.get(fam, INK)
        if np.isfinite(blo):
            ax.plot([blo, bhi], [y, y], color=col, lw=0.8, zorder=3)
        ax.plot([bm], [y], marker="o", ms=4.6, ls="none", mfc=col, mec=col, zorder=4)
        right = bm > 0.30
        _label(ax, bm, y, NAME.get(bs, bs), color=INK, dx=4 if right else 0, dy=7, ha="right" if right else "center",
               va="bottom", zorder=5, bbox=dict(fc="white", ec="none", pad=0.6))   # the dashed line runs behind it
    for (kind, head, _), yy in zip(slots, ys):
        if kind == "h":
            C.group_heading(ax, yy, head)
    _rows_axes(ax, [row_y[f] for k, f, _ in slots if k == "r"], [lab for k, _, lab in slots if k == "r"])
    ax.set_ylim(min(ys) - 0.5, 0.5)
    ax.set_xlim(0, 0.7)                                      # the model alone reaches past .6 (rebuild, 2 Nov 2026)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "x")
    ax.set_xlabel(C.axis_title("nDCG@10"))
    # H7 (the author, 2 Nov 2026: "confusing legend with the plot"): the key sits above the axes, on one line, off the data;
    # the dashed line needs no entry, since it runs down from the point named tuned lexical matching.
    _legend(ax, [Line2D([0], [0], marker="o", ls="-", lw=0.8, ms=4.6, color=INK2, label="best of its kind, with 95% interval"),
                 Line2D([0], [0], marker="o", ls="none", ms=3.6, mfc=MARK, mec="white", label="other systems of that kind")],
            "lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, borderaxespad=0.0, columnspacing=1.4)
    return _write(fig, ctx, stem, cap, "Each family alone", label)


# ---- 6.5 (F6_2): quality against reach ------------------------------------------------------------------------------
def f6_2(ctx: Ctx) -> Path:
    """Quality against reach: nDCG@10 against never-cited Recall@20 for every single source that reaches any
    never-cited article; arrows from a pre-trained encoder to its task-trained version."""
    stem, label = "F6_2_reach", "fig:F6_2_reach"
    # the fusions of lexical matching with something else (fusion_ltr.DERIVED_IDS) are not single sources: their reach is
    # that of the lexical part, so drawn as past citations they would show a reach those sources do not have (H8, 2 Nov)
    fused = ("src_K_qwen8b", "src_H_qwen8b", "src_KH_qwen8b", "src_K_lex", "src_K_lex_ext", "src_LP", "src_H_bge")
    pts, zero, zero_best = [], {}, 0.0
    for s in ctx.systems():
        fam = _family_of(s)
        if fam is None or "@" in s or s == "e5_large_instruct" or s in fused:
            continue
        sm = ctx.summary(s)
        if not (sm and sm.get("ndcg_at_10") is not None):
            continue
        y = sm.get("unseen_recall_at_20_pooled", sm.get("unseen_recall_at_20"))   # pooled, as Ch4 and RESULTS.md
        if y is None or y <= 0:
            zero[fam] = zero.get(fam, 0) + 1
            zero_best = max(zero_best, sm["ndcg_at_10"])
            continue
        pts.append((s, fam, sm["ndcg_at_10"], y))
    if not pts:
        return _placeholder(ctx, stem, "", "Quality against reach", label, "no single-source results")
    if set(zero) != {"past citations"}:
        raise ValueError(f"{stem}: sources at zero reach outside past citations: {zero}")
    ctx.notes.setdefault(stem, []).append("left out at zero never-cited recall: "
                                          + ", ".join(f"{k} {v}" for k, v in sorted(zero.items())))
    cases = ctx.per_case("qwen3_8b")
    pairs, with_any = int(cases["unseen_n"].sum()), int((cases["unseen_n"] > 0).sum())
    by = cases.groupby("component_id")[["unseen_n", "unseen_hits_at_20"]].sum().to_numpy()
    rng = np.random.default_rng(20261102)
    boot = []
    for _ in range(2000):               # groups of connected cases resampled, as the intervals of the chapter
        w = np.bincount(rng.integers(0, len(by), len(by)), minlength=len(by))
        boot.append((by[:, 1] * w).sum() / (by[:, 0] * w).sum())
    half = (np.percentile(boot, 97.5) - np.percentile(boot, 2.5)) / 2
    cap = (f"Each point is one source on its own, on the development cases; sources that add lexical matching to something "
           f"else are left out. Reach rests on {pairs} citations of never-cited articles in {with_any} cases, an interval of "
           f"about $\\pm${half:.2f} for the best encoders.")
    # H8 (the author, 2 Nov 2026: "it isnt 100% clear what is going on"; then "too much text", keep the moves, no dots at zero):
    # one mark, a colour per family named on the plot, and the arrows of training as the only other ink.
    colour = {"lexical": INK, "one collection": MARK, "pre-trained encoders": C.tint(C.COMPANION_PLUM, 0.5),
              "task-trained encoders": C.COMPANION_PLUM, "past citations": C.COMPANION_BLUE}
    fig = _new(2.9)
    ax = fig.add_axes([0.085, 0.15, 0.895, 0.77])
    d = {s: (x, y) for s, _, x, y in pts}
    for s, fam, x, y in pts:
        ax.plot([x], [y], marker="o", ms=4.4, ls="none", mfc=colour[fam], mec="white", mew=0.4, zorder=3)
    for a, b in TASK_PAIRS + [("qwen3_8b", "qwen8b_adapter")]:
        if a in d and b in d:
            ax.annotate("", xy=d[b], xytext=d[a], zorder=2,
                        arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C.COMPANION_PLUM, mutation_scale=8, shrinkA=2.5, shrinkB=2.5))
        elif a in d or b in d:
            ctx.notes.setdefault(stem, []).append(f"pair {a} -> {b} incomplete in the store")
    # the families, named where they lie
    for x, y, text, col, ha in [(0.03, 0.055, "learned from the collection", INK2, "left"),
                                (0.112, 0.335, "encoders", C.COMPANION_PLUM, "left"),
                                (0.33, 0.1, "past citations", C.COMPANION_BLUE, "left"),
                                (0.2, 0.17, "lexical matching", INK, "left")]:
        ax.text(x, y, text, fontsize=7.5, fontweight="bold", color=col, ha=ha, va="center", zorder=5)
    ax.text(0.33, 0.082, f"{zero['past citations']} more find none,\nranking up to {zero_best:.2f}", fontsize=7,
            color=C.COMPANION_BLUE, ha="left", va="top", zorder=5, linespacing=1.15)
    # what an arrow is, above the plot so that its two dots cannot be read as sources
    for kx, kc in ((0.012, colour["pre-trained encoders"]), (0.082, colour["task-trained encoders"])):
        ax.plot([kx], [1.055], marker="o", ms=4.4, ls="none", mfc=kc, mec="white", mew=0.4, zorder=3,
                transform=ax.transAxes, clip_on=False)
    ax.annotate("", xy=(0.082, 1.055), xytext=(0.012, 1.055), xycoords="axes fraction", textcoords="axes fraction", zorder=2,
                annotation_clip=False,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color=C.COMPANION_PLUM, mutation_scale=8, shrinkA=2.5, shrinkB=2.5))
    ax.text(0.098, 1.055, "the same encoder before and after training on the task", fontsize=7, color=INK, ha="left",
            va="center", transform=ax.transAxes, clip_on=False)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 0.45)
    ax.xaxis.set_major_locator(MultipleLocator(0.05)); ax.yaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "x"); C.strip0(ax, "y")
    ax.set_xlabel(C.axis_title("nDCG@10"))
    ax.set_ylabel("never-cited articles found, Recall@20")
    return _write(fig, ctx, stem, cap, "Quality against reach", label)


# ---- 6.6 (F6_3): what moves an embedding model ---------------------------------------------------------------------
SIZE_LAB = {"qwen3_0p6b": ("0.6B", 5, -3, "left"), "qwen3_4b": ("4B", 4, -6, "left"), "qwen3_8b": ("8B", 0, 5, "center"),
            "e5_small": ("small", 0, -8, "center"), "e5_base": ("base", 0, -8, "center"),
            "e5_large": ("large", 5, -2, "left"), "gte_multilingual": ("GTE-multilingual", -5, 2, "right"),
            "jina_v3": ("Jina v3", -5, 0, "right"), "arctic_l_v2": ("Arctic-Embed L v2", -5, 0, "right"),
            "bge_m3": ("BGE-M3", 5, -2, "left")}
# weakest general model at the bottom; the 8B twice: in eight bits beside its legal version, and at full precision
# beside the adapter trained on it
TRAIN_ROWS = ["e5_small", "bge_m3", "qwen3_0p6b", "qwen3_8b_int8_base", "qwen3_8b"]
TRAIN_NAME = {"qwen3_8b_int8_base": "Qwen3-Embedding-8B\nin eight bits"}


def f6_3(ctx: Ctx) -> Path:
    """Two figures (CF-04, the author's option B with redraw R1, scheme P3): size of the general models, then each general
    model beside its legal and task-trained versions, in F6_3b_training.tex, which 06 inputs on its own."""
    stem, label = "F6_3_encoders", "fig:F6_3_encoders"
    cap = "Ranking quality against size for the general encoders on the development cases."
    general = ["e5_small", "e5_base", "e5_large", "gte_multilingual", "jina_v3", "arctic_l_v2", "bge_m3",
               "qwen3_0p6b", "qwen3_4b", "qwen3_8b"]
    gen = {s: ndcg(ctx, s) for s in general}
    gen = {s: v[0] for s, v in gen.items() if v}
    if not gen:
        return _placeholder(ctx, stem, cap, "Size of a dense retriever", label, "no encoder results")
    qwen, e5 = ("qwen3_0p6b", "qwen3_4b", "qwen3_8b"), ("e5_small", "e5_base", "e5_large")
    lex = ndcg(ctx, "lstar_headings")
    # the second figure (legal and task training), in its own file
    tstem, tlabel = "F6_3b_training", "fig:F6_3_training"
    tcap = ("Each general encoder on the development cases beside its task-trained version and, where one was "
            "trained from the same model, its legal version. The legal Qwen3-Embedding-8B has eight-bit weights, so "
            "it is set beside the general model in eight bits.")
    fig = _new(2.5)
    ax = fig.add_axes([0.3, 0.18, 0.68, 0.78])
    if lex:
        ax.axvline(lex[0], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        _label(ax, lex[0], len(TRAIN_ROWS) - 0.45, "tuned lexical matching", color=MUTED, dx=3, ha="left", va="center")
    task = TASK_PAIRS + [("qwen3_8b", "qwen8b_adapter")]       # the adapter on the frozen 8B (R review 4.3)
    rows = []
    for base in TRAIN_ROWS:
        sb = ndcg(ctx, base)
        if not sb:
            continue
        vals = {"general": sb[0]}
        for b0, other, kind in [(p, l, "legal") for p, l in LEGAL_PAIRS] + [(p, t, "task") for p, t in task]:
            so = ndcg(ctx, other) if b0 == base else None
            if so:
                vals[kind] = so[0]
            elif b0 == base:
                ctx.notes.setdefault(tstem, []).append(f"pair {base} -> {other} incomplete in the record")
        rows.append((base, vals))
    for i, (base, vals) in enumerate(rows):
        ax.plot([min(vals.values()), max(vals.values())], [i, i], color=RULE, lw=2.2, zorder=1, solid_capstyle="round")
        ax.plot([vals["general"]], [i], "o", ms=5, mfc="white", mec=INK2, mew=1.0, zorder=3)
        if "legal" in vals:
            ax.plot([vals["legal"]], [i], "s", ms=4.8, color=INK2, zorder=3)
        if "task" in vals:
            ax.plot([vals["task"]], [i], "D", ms=4.8, color=PLUM, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([TRAIN_NAME.get(b) or NAME.get(b, b) for b, _ in rows], fontsize=8)
    C.only_x(ax)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(0.10, 0.40)
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    C.strip0(ax, "x")
    ax.set_xlabel(C.axis_title("nDCG@10"))
    _legend(ax, [Line2D([0], [0], marker="o", ls="none", ms=5, mfc="white", mec=INK2, label="general model"),
                 Line2D([0], [0], marker="s", ls="none", ms=4.8, color=INK2, label="its legal version"),
                 Line2D([0], [0], marker="D", ls="none", ms=4.8, color=PLUM, label="its task-trained version")],
            "lower right")
    _write(fig, ctx, tstem, tcap, "Legal and task training", tlabel)
    # the first figure: size of the general models; Qwen3 plum (dense retrievers, D181 colours), E5 dark grey, the others grey
    fig = _new(2.75)
    a = fig.add_axes([0.1, 0.16, 0.88, 0.74])
    if lex:
        a.axhline(lex[0], color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=0)
        _label(a, 20000, lex[0], "tuned lexical matching", color=MUTED, dx=-1, dy=2, ha="right", va="bottom")
    a.set_xscale("log")
    a.set_xlim(70, 20000)
    _logfmt(a, "x")
    a.set_xlabel("parameters (millions)")
    for series, name, col, (dx, dy, ha) in ((qwen, "Qwen3-Embedding", PLUM, (-4, 8, "right")),
                                            (e5, "E5", INK2, (0, -12, "center"))):
        f = [s for s in series if s in gen]
        a.plot([PARAMS_M[s] for s in f], [gen[s] for s in f], color=col, lw=1.0, zorder=2)
        if f:
            anchor = f[-1] if name == "E5" else f[min(1, len(f) - 1)]
            _label(a, PARAMS_M[anchor], gen[anchor], name, color=col, dx=dx, dy=dy, ha=ha, weight="bold")
    for s, y in gen.items():
        col = PLUM if s in qwen else INK2 if s in e5 else MARK
        a.plot([PARAMS_M[s]], [y], marker="^", ms=4.4, ls="none", color=col, zorder=3)
        t, dx, dy, ha = SIZE_LAB.get(s, (NAME.get(s, s), 5, 0, "left"))
        _label(a, PARAMS_M[s], y, t, color=INK2, dx=dx, dy=dy, ha=ha)
    a.set_ylabel(C.axis_title("nDCG@10"))
    a.set_ylim(0.09, 0.33)
    a.yaxis.set_major_locator(MultipleLocator(0.05))
    C.strip0(a, "y")
    # no legend (LF overnight 26-27 Oct): both series and every other model are named beside their marks
    # F6_3b_training.tex is no longer input from here: 06 inputs it after the paragraph on the order of the results
    # (D176 layout, L5), because the two figures together were taller than a page
    return _write(fig, ctx, stem, cap, "Size of a dense retriever", label)


# ---- 6.7 (F6_4): size of the stored article vectors -----------------------------------------------------------------
PRECISIONS = [("", 4, "32-bit numbers", dict(color=PLUM, marker="o", ls="-")),        # D181: no green outside
              ("_int8", 1, "eight bits", dict(color=INK2, marker="s", ls=(0, (4, 1.5)))),   # the integrated method
              ("_binary", 1 / 8, "one bit", dict(color=MUTED, marker="D", ls=(0, (1, 1.2))))]


def f6_4(ctx: Ctx) -> Path:
    """One panel per Qwen3 model; in each, one complete line per precision through its stored dimensions."""
    stem, label = "F6_4_compression", "fig:F6_4_compression"
    cap = ("Ranking quality on the development cases against the size of the article vectors in memory, for the "
           "three Qwen3-Embedding models. Each line is one precision of the numbers, and its points are the vector "
           "lengths, from 256 dimensions to the model's full length.")
    fig = _new(2.45)
    axes, any_pt = [], False
    for j, (base, full) in enumerate(DIMS.items()):
        ax = fig.add_axes([0.1 + j * 0.305, 0.19, 0.265, 0.69], sharey=axes[0] if axes else None)
        axes.append(ax)
        for suffix, nbytes, _, st in PRECISIONS:
            pts, missing = [], []
            for dims in sorted({256, 512, 1024, full}):
                if dims > full:
                    continue
                sid = base + (f"_mrl{dims}" if dims != full else "") + suffix
                v = ndcg(ctx, sid)
                if v is None:
                    missing.append(sid)
                else:
                    pts.append((N_ART * dims * nbytes / 1e6, v[0]))
            if missing:
                ctx.notes.setdefault(stem, []).append(f"not in the record: {', '.join(missing)}")
            if pts:
                any_pt = True
                ax.plot(*zip(*pts), color=st["color"], ls=st["ls"], lw=0.9, zorder=2)
                ax.plot(*zip(*pts), color=st["color"], marker=st["marker"], ls="none", ms=3.4, mfc="white",
                        mew=0.9, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(0.6, 1000)
        _logfmt(ax, "x")
        C.panel(ax, "abc"[j], NAME[base])
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
    if not any_pt:
        plt.close(fig)
        return _placeholder(ctx, stem, cap, "Size of the article vectors", label, "no compressed variants")
    axes[0].set_ylim(0.05, 0.30)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.05))
    C.strip0(axes[0], "y")
    axes[0].set_ylabel(C.axis_title("nDCG@10"))
    axes[1].set_xlabel("MB of article vectors")
    _legend(axes[0], [Line2D([0], [0], color=st["color"], ls=st["ls"], lw=0.9, marker=st["marker"], ms=3.4,
                             mfc="white", mew=0.9, label=name) for _, _, name, st in PRECISIONS],
            "upper left", title="precision", title_fontsize=7, alignment="left")
    return _write(fig, ctx, stem, cap, "Size of the article vectors", label)


# ---- 6.10 (F6_5): untrained hybrids --------------------------------------------------------------------------------
def f6_5(ctx: Ctx) -> Path:
    """Untrained hybrids (CF-06, LF overnight 26-27 Oct, on the model of the compression figure; was one panel of
    differences, "very confsing"): one panel per form, one point per partner, its own nDCG@10 across and the hybrid's
    up. The grey line is the better of the two parts alone (flat at tuned lexical matching, then the partner), so a
    point under it is a hybrid worse than one of its parts."""
    stem, label = "F6_5_hybrids", "fig:F6_5_hybrids"
    cap = ("Untrained hybrids of tuned lexical matching and one partner on the development cases, by reciprocal rank "
           "fusion (a) and the weighted sum (b). Each point is one partner, with its own nDCG@10 across and the "
           "hybrid's up; the grey line is the better of the two parts alone, so a point under it is a hybrid worse "
           "than one of its parts. The weighted sum never falls meaningfully below the better of its two parts; "
           "reciprocal rank fusion falls below it when the partner is weak.")
    lex = ctx.summary("lstar_headings")
    if not lex:
        return _placeholder(ctx, stem, cap, "Untrained hybrids", label, "no lexical baseline")
    lex = lex["ndcg_at_10"]
    fig = _new(2.55)
    axes, n = [], 0
    for j, (kind, title) in enumerate((("rrf", "reciprocal rank fusion"), ("linear", "weighted sum"))):
        ax = fig.add_axes([0.1 + j * 0.46, 0.17, 0.4, 0.72], sharey=axes[0] if axes else None)
        axes.append(ax)
        ax.plot([0, lex, 0.35], [lex, lex, 0.35], color=MARK, lw=0.9, zorder=1)
        pts = {}
        for s in ctx.systems():
            pref = f"{kind}__lstar_headings__"
            if not s.startswith(pref):
                continue
            partner = s[len(pref):]
            hp, pp = ctx.summary(s), ctx.summary(partner)
            if hp and pp:
                pts[partner] = (pp["ndcg_at_10"], hp["ndcg_at_10"])
        for partner, (x, y) in pts.items():
            on = partner == "qwen3_8b"
            ax.plot([x], [y], marker="o", ms=4.2 if on else 3.4, ls="none", color=PLUM if on else INK2,
                    mfc=PLUM if on else "white", mew=0.8, zorder=3 if on else 2)
        n += len(pts)
        ax.set_xlim(0, 0.35)
        ax.set_ylim(0.05, 0.35)
        ax.xaxis.set_major_locator(MultipleLocator(0.1)); ax.yaxis.set_major_locator(MultipleLocator(0.1))
        C.strip0(ax, "x"); C.strip0(ax, "y")
        C.panel(ax, "ab"[j], title)
        lead = dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=1, shrinkB=2)
        if "qwen3_8b" in pts:   # labels in empty space, with a thin leader to the point
            ax.annotate(NAME["qwen3_8b"], xy=pts["qwen3_8b"], xytext=(0.205, 0.335) if j else (0.345, 0.17),
                        fontsize=7, color=PLUM, ha="right", va="center", arrowprops=lead)
        if j:
            plt.setp(ax.get_yticklabels(), visible=False)
            # the line explained once, in the empty lower half of b
            ax.text(0.01, lex - 0.01, "tuned lexical matching alone", fontsize=7, color=INK2, ha="left", va="top")
            ax.text(0.3, 0.24, "partner\nalone", fontsize=7, color=INK2, ha="center", va="top")
        elif "doc2vec" in pts:
            ax.annotate(NAME.get("doc2vec", "doc2vec"), xy=pts["doc2vec"], xytext=(0.075, 0.075), fontsize=7,
                        color=INK2, ha="left", va="center", arrowprops=lead)
    if not n:
        plt.close(fig)
        return _placeholder(ctx, stem, cap, "Untrained hybrids", label, "no hybrid results")
    axes[0].set_ylabel("hybrid, nDCG@10")
    axes[0].set_xlabel("partner alone, nDCG@10, development cases")
    axes[0].xaxis.set_label_coords(1.075, -0.13)     # centred under the pair of panels
    return _write(fig, ctx, stem, cap, "Untrained hybrids", label)


# ---- 6.1 (F6_6): gain over the ranker without embeddings ------------------------------------------------------------
LADDER = [  # (system a of the contrast "a - S0" in the F2 family, row label): BGE-M3 and the Qwen3 size series, the
    # encoders 6.3.1 names (R review 1.4, as agreed with B); the other runs of the family are in the run record (D190)
    ("S0+bge_m3_dense", "BGE-M3"), ("S0+qwen3_0p6b", "Qwen3-Embedding-0.6B"),
    ("S0+qwen3_4b", "Qwen3-Embedding-4B"), ("S0+qwen3_8b_dense", "Qwen3-Embedding-8B"),
]


def _contrast(ctx: Ctx, fname: str, key: str | None = None) -> dict | None:
    dirs = ([RECORD_100K] if ctx.final and RECORD_100K.is_dir() else []) + [ctx.store.parent / "contrasts"] \
        + ([ctx.extra_contrasts] if ctx.extra_contrasts else [])
    for d in dirs:
        p = d / f"{fname}.json"
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if key is None else data.get("contrasts", {}).get(key)
    if key is None and "__vs__" in fname:          # a single contrast kept inside a family file (final store)
        a, b = fname.rsplit("__", 1)[0].split("__vs__")
        for p in sorted((ctx.store.parent / "contrasts").glob("F*.json")):
            c = json.loads(p.read_text(encoding="utf-8")).get("contrasts", {}).get(f"{a} - {b}")
            if c:
                return c
    return None


def f6_6(ctx: Ctx) -> Path:
    """Gain over the ranker without embeddings: one encoder added (group 1); the two systems (group 2)."""
    stem, label = "F6_6_ranker_gain", "fig:F04_ladder"
    cap = ("Gain in nDCG@10 over the ranker without embeddings on the development cases, from adding the score of BGE-M3 "
           "or of one Qwen3-Embedding model, for the ranker, which adds Qwen3-Embedding-8B with its past-case sources, "
           "for the ranker with every encoder and for the largest ranker, which also reads the unstructured "
           "\\emph{tesis}. Each line is an "
           "unadjusted 95\\% interval, and the short bar is its lower end adjusted for the size of its family "
           "(\\cref{app:statistics}). The marker gives the class under the rule of \\cref{sec:comparing-systems}, "
           "read from the short bar against the dotted margin of 0.02. Of the encoders, only "
           "Qwen3-Embedding-8B reaches the margin on its own.")   # D167: the margin stated once
    enc, systems = [], []
    for key, text in LADDER:
        c = None if ctx.held(key) else _contrast(ctx, "F2", f"{key} - S0")
        if c is None:
            ctx.notes.setdefault(stem, []).append(f"F2 contrast {key} - S0 absent")
            continue
        enc.append((text, c["delta"], *c["ci95"], c["class"], c["adjusted_lower"]))
    d185 = json.loads(D185_CONTRASTS.read_text(encoding="utf-8")) if D185_CONTRASTS.exists() else {}
    for fname, text in ((f"{INTEGRATED}__vs__S0__DEV", "the ranker"),
                        ("S_full__vs__S0__DEV", "with every encoder"),
                        ("S_full_ext__vs__S0__DEV", "largest ranker")):
        a = fname.split("__vs__")[0]
        c = d185.get("D185_main", {}).get("contrasts", {}).get(f"{a} - S0") if a == INTEGRATED else _contrast(ctx, fname)
        if c:
            systems.append((text, c["delta"], *c["ci95"], c["class"], c["adjusted_lower"]))
        else:
            ctx.notes.setdefault(stem, []).append(f"contrast {fname} absent")
    if not enc and not systems:
        return _placeholder(ctx, stem, cap, "Gain over the ranker without embeddings", label, "no contrasts")
    enc.sort(key=lambda r: r[1])
    # rows top-down: heading, encoders, gap, heading, systems
    slots = [("h", "one dense retriever added")] + [("r", r) for r in enc] + [("g", None), ("h", "the rankers")] \
        + [("r", r) for r in systems]
    n = len(slots)
    h = 0.165 * n + 0.74
    fig = _new(h)
    ax = fig.add_axes([0.4, 0.62 / h, 0.57, (0.165 * n) / h])
    ys, labels, colors = [], [], []
    for i, (kind, r) in enumerate(slots):
        y = n - 1 - i
        if kind == "h":
            C.group_heading(ax, y, r)
        elif kind == "r":
            text, m, lo, hi, cls, adj = r
            col = {"the ranker": GREEN, "with every encoder": INK2, "largest ranker": INK2}.get(text, PLUM)
            ax.plot([lo, hi], [y, y], color=col, lw=0.8, zorder=2)
            # the family-adjusted lower end that decides the class (D138: 100,000 draws), as a short bar
            ax.plot([adj], [y], marker="|", ms=7, mew=1.1, ls="none", color=col, zorder=4)
            _dot(ax, m, y, cls, color=col)
            ys.append(y); labels.append(text); colors.append(GREEN if col == GREEN else INK2)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.axvline(.02, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=0)
    _label(ax, .02, n - 1, "margin", color=MUTED, dx=3, ha="left")
    _rows_axes(ax, ys, labels, colors)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(-0.008, 0.11)
    ax.xaxis.set_major_locator(MultipleLocator(0.02))
    C.strip0(ax, "x")
    ax.set_xlabel("gain in nDCG@10 over the ranker without embeddings,\n" + C.CASES)
    _legend(ax, [Line2D([0], [0], marker="o", ls="none", ms=4.2, mew=0.8, label=CLASS_NAME[k], **CLASS_MARK[k])
                 for k in ("useful", "nonzero_below_margin", "equivalent")]
            + [Line2D([0], [0], marker="|", ms=7, mew=1.1, ls="none", color=INK, label="lower end that decides the class")],
            "upper right")
    return _write(fig, ctx, stem, cap, "Gain over the ranker without embeddings", label)


def _ints(v) -> list[int]:
    """positive_idx / positive_ranks are stored as comma-separated strings (or scalars); ch6_numbers reads these."""
    if isinstance(v, str):
        return [int(float(x)) for x in v.split(",") if x.strip()]
    return [int(x) for x in np.atleast_1d(v)]


def _pair_hits(pc: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    rows = []
    for qid, idx, ranks in zip(pc["query_id"], pc["positive_idx"], pc["positive_ranks"]):
        for a, r in zip(_ints(idx), _ints(ranks)):
            rows.append((qid, a, r <= k))
    return pd.DataFrame(rows, columns=["query_id", "article", "hit"])


# ---- 6.2 (F6_7): recall by citation band ----------------------------------------------------------------------------
BANDS = ["0", "1-2", "3-10", ">10"]
BAND_NAME = {"0": "0", "1-2": "1–2", "3-10": "3–10", ">10": "more than 10"}


def _band_recall_from_pairs(ctx: Ctx, system: str, reference: pd.DataFrame, store: Path | None = None) -> pd.DataFrame | None:
    """Band Recall@20 of a ranker that figdata/band_recall.csv lacks (D181: the integrated method), read from the record
    without a model: the pairs and their fold-local bands of analysis C3 (c3/c3_pairs.csv), the system's stored rank of
    each pair (per_case.parquet), and C3's own pooled-ratio cluster bootstrap (uol.analysis_c.common.Boot, 2,000 draws,
    seed 20261023). The same code must reproduce the CSV's row of the ranker without embeddings exactly, values and
    intervals, before the new row is used."""
    from uol.analysis_c.common import Boot
    pairs = ctx.csv("c3/c3_pairs.csv")
    base = ctx.per_case("S0")
    if pairs is None or base is None:
        return None
    boot = Boot(base["component_id"].to_numpy(), 2000, 20261023)
    g = boot.group_index(pairs["component_id"])

    def rows(s: str, st: Path | None = None) -> pd.DataFrame | None:
        f = st / s / "DEV" / "per_case.parquet" if st else None
        pc = (pd.read_parquet(f) if f.exists() else None) if f else ctx.per_case(s)
        if pc is None:
            return None
        rank = {(q, a): r for q, i, rr in zip(pc["query_id"], pc["positive_idx"], pc["positive_ranks"])
                for a, r in zip(_ints(i), _ints(rr))}
        hit = np.array([1.0 if 0 < rank[(q, a)] <= 20 else 0.0 for q, a in zip(pairs["query_id"], pairs["article"])])
        out = []
        for b in BANDS:
            m = (pairs["band"].astype(str) == b).to_numpy()
            r = boot.ratio(np.where(m, hit, 0.0), m.astype(float), g)
            out.append((b, r["mean"], r["lo"], r["hi"]))
        return pd.DataFrame(out, columns=["band", "recall_at_20", "lo", "hi"]).set_index("band")

    check = rows("S0")
    if check is None or not np.allclose(check[["recall_at_20", "lo", "hi"]].to_numpy(),
                                        reference.loc[BANDS, ["recall_at_20", "lo", "hi"]].to_numpy(), atol=1e-5):
        ctx.notes.setdefault("F6_7_bands", []).append("C3 pair bootstrap does not reproduce the ranker without embeddings")
        return None
    got = rows(system, store)
    if got is not None:
        ctx.notes.setdefault("F6_7_bands", []).append(
            f"{system} computed from c3_pairs and per_case (reproduces S0 exactly): "
            + ", ".join(f"{b} {v:.3f} [{l:.3f}, {h:.3f}]" for b, (v, l, h) in got.iterrows()))
    return got


def f6_7(ctx: Ctx) -> Path:
    """Recall@20 of cited pairs by citation band (figdata/band_recall.csv): two rankers as bars, the 8B as a line."""
    stem, label = "F6_7_bands", "fig:F6_7_bands"
    cap = ("Share of the cited articles found among the first "
           "twenty on the development cases, per cited pair, by how often the fold's training cases had cited the "
           "article, with 95\\% intervals.")
    br = ctx.csv("figdata/band_recall.csv")
    want = {NOENC5: "the ranker without the encoder", RANKER8: "the ranker", "qwen3_8b": "Qwen3-Embedding-8B alone"}
    rec = {}
    if br is not None:
        for s in ("S0", "qwen3_8b"):
            sid = ctx.sid(s)
            d = br[br["system"] == sid].astype({"band": str}).set_index("band") if sid else None
            if d is not None and not d.empty:
                rec[s] = d
        if "S0" in rec:
            for s in (NOENC5, RANKER8):
                got = _band_recall_from_pairs(ctx, s, rec["S0"], FOOT_STORE)
                if got is not None:
                    rec[s] = got
    if NOENC5 not in rec or RANKER8 not in rec:
        return _placeholder(ctx, stem, cap, "Recall by citation band", label, "band recall of the rankers missing")
    fig = _new(2.55)
    ax = fig.add_axes([0.115, 0.17, 0.865, 0.79])
    x = np.arange(len(BANDS))
    for s, dx, col in ((NOENC5, -0.18, BLUE), (RANKER8, 0.18, GREEN)):
        d = rec[s]
        v = [d["recall_at_20"].get(b, np.nan) for b in BANDS]
        lo = [d["lo"].get(b, np.nan) for b in BANDS]; hi = [d["hi"].get(b, np.nan) for b in BANDS]
        ax.bar(x + dx, v, 0.34, color=col, zorder=2)
        ax.vlines(x + dx, lo, hi, color="white", lw=1.8, zorder=3)     # a white halo keeps the whisker visible
        ax.vlines(x + dx, lo, hi, color=INK, lw=0.6, zorder=3.1)       # on the dark bars and above them
    if "qwen3_8b" in rec:
        for i, b in enumerate(BANDS):
            v = rec["qwen3_8b"]["recall_at_20"].get(b, np.nan)
            ax.plot([i - 0.38, i + 0.38], [v, v], color=PLUM, lw=1.5, ls=(0, (3, 1.2)), zorder=4,
                    solid_capstyle="butt")
    ax.set_xticks(x)
    ax.set_xticklabels([BAND_NAME[b] for b in BANDS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlabel("times the article was cited before")
    ax.set_ylabel(C.axis_title("Recall@20"))
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    C.strip0(ax, "y")
    ax.set_xlim(-0.6, len(BANDS) - 0.4)
    handles = [Patch(color=BLUE, label=want[NOENC5]), Patch(color=GREEN, label=want[RANKER8])]
    if "qwen3_8b" in rec:
        handles.append(Line2D([0], [0], color=PLUM, lw=1.5, ls=(0, (3, 1.2)), label=want["qwen3_8b"] + " (line)"))
    handles.append(Line2D([0], [0], color=INK, lw=0.6, label="95% interval"))
    _legend(ax, handles, "upper left", handlelength=1.2)
    shares = rec["S0"]["share_of_pairs"]
    ctx.notes.setdefault(stem, []).append("share of the cited pairs by band (for the text): "
                                          + ", ".join(f"{BAND_NAME[b]} {shares.get(b, np.nan):.3f}" for b in BANDS))
    return _write(fig, ctx, stem, cap, "Recall by citation band", label)


# ---- 6.3 (F6_8): law against article --------------------------------------------------------------------------------
F68_GROUPS = [("the system and its first stage", [SYSTEM, INTEGRATED, "S0"]),
              ("comparisons", ["llm_rerank_S_full_ext", "S_full_ext"]),
              ("single sources", ["llm_alone", "src_proto_qwen8b", "qwen3_8b", "lsa", "lstar_headings"])]
# D181: the three outcomes in greys for every row, in green and its tint for the integrated method's row only
F68_GREYS = (INK2, "#a8a8a8", C.NEUTRAL)
F68_GREENS = (GREEN, TINT, C.NEUTRAL)


def f6_8(ctx: Ctx) -> Path:
    """Where the first result lands (CF-03, the author chose option A, 26 Oct 2026; was two dumbbell panels): one bar per
    system, split into a cited article (green), another article of a cited law (focus tint) and an article of another
    law (light grey). The parts add to one, since an article of a cited law lies in that law, so each bar shows the
    two steps of 6.3: the law, then the article within it."""
    stem, label = "F6_8_law_article", "fig:F6_8_law_article"
    cap = ("Where the first result lands on the development cases, for each system: a cited article, another article "
           "of a cited law, or an article of another law; the system's bar is drawn in green. Every system "
           "puts a cited law first far more often than a cited article, and for the rankers and the rereads most first "
           "results that miss the article still fall in a cited law.")
    slots = []
    for head, ids in F68_GROUPS:
        rows = []
        for s in ids:
            r = ctx.c1(s)
            if r is None:
                ctx.notes.setdefault(stem, []).append(f"{s} not in the record")
                continue
            law1, art1 = float(r["law_at_1"]), float(r["success_at_1"])
            assert 0 <= art1 <= law1 <= 1, (s, art1, law1)     # a cited article first means a cited law first
            rows.append((s, law1, art1))
        if rows:
            slots += [("h", head)] + [("r", x) for x in rows]
    if not slots:
        return _placeholder(ctx, stem, cap, "Law against article", label, "no law-level measures")
    ys, y = [], 0.0
    for i, (kind, r) in enumerate(slots):
        two = kind == "r" and r[0].startswith("src_")       # its name takes two lines
        if kind == "h" and i:
            y -= 0.5                                         # a gap before each group after the first
        y -= 0.2 if two else 0
        ys.append(y)
        y -= 1.2 if two else 1
    h = 0.17 * (len(slots) + 1) + 0.75                    # compact rows, so it can share a page with text
    fig = _new(h)
    ax = fig.add_axes([0.375, 0.52 / h, 0.605, (h - 0.88) / h])
    lab_y, labs, cols = [], [], []
    for (kind, r), y in zip(slots, ys):
        if kind == "h":
            C.group_heading(ax, y, r)
            continue
        s, law1, art1 = r
        shades = F68_GREENS if s == SYSTEM else F68_GREYS
        for left, width, col in zip((0, art1, law1), (art1, law1 - art1, 1 - law1), shades):
            ax.barh(y, width, left=left, height=0.66, color=col, edgecolor="white", linewidth=0.6, zorder=2)
        lab_y.append(y)
        labs.append(NAME.get(s, s).replace("similar-case citations, ", "similar-case citations,\n"))
        cols.append(GREEN if s == SYSTEM else INK2)
    _rows_axes(ax, lab_y, labs, cols)
    ax.set_ylim(min(ys) - 0.6, 0.6)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    C.strip0(ax, "x")
    ax.set_xlabel("share of development cases, by where the first result lands")
    _legend(ax, [Patch(color=F68_GREYS[0], label="a cited article"),
                 Patch(color=F68_GREYS[1], label="another article of a cited law"),
                 Patch(color=F68_GREYS[2], label="an article of another law")],
            "lower center", ncol=3, bbox_to_anchor=(0.42, 1.0), columnspacing=1.2, handlelength=1.0)
    return _write(fig, ctx, stem, cap, "Law against article", label)


# ---- 6.9 (F6_9): the LLM stage over two rankers ---------------------------------------------------------------------
def f6_llm(ctx: Ctx) -> Path:
    """The LLM stage over two rankers as labelled bars (figdata/llm_text_headroom.csv): the ranker alone, the stage
    reading only law and number, the stage reading the article text, and a perfect reordering of the first 30."""
    stem, label = "F6_9_llm_stage", "fig:F6_9_llm_stage"
    cap = ("The reread recovers part of what its thirty candidates allow. Development cases, nDCG@10 with unadjusted "
           "95\\% intervals, for the ranker with the reread by Claude Opus 5.5, and for two other rankers with the reread by "
           "GPT-5.6, in each case against a perfect reordering of the first thirty.")
    t = ctx.csv("figdata/llm_text_headroom.csv")
    if t is None:
        return _placeholder(ctx, stem, cap, "The reread", label, "no reread results")
    t = t.set_index("system")
    c1 = pd.read_csv(FINAL_RECOMPUTE / "c1" / "c1_systems.csv").set_index("system")
    ceil = pd.read_csv(FINAL_RECOMPUTE / "d102" / "d102_ceilings.csv").set_index("measure")
    for sid in (INTEGRATED, SYSTEM):
        t.loc[sid, ["ndcg_at_10", "lo", "hi"]] = [c1.loc[sid, k] for k in ("ndcg_at_10", "ndcg_at_10_lo", "ndcg_at_10_hi")]
    t.loc["oracle@30 of the ranker", ["ndcg_at_10", "lo", "hi"]] = [ceil.loc["oracle_at_30", k] for k in ("value", "ci_low", "ci_high")]
    groups = [("the system", [(INTEGRATED, "the ranker alone", GREEN, None),
                              (SYSTEM, "with the reread by Claude Opus 5.5", OCHRE, None),
                              ("oracle@30 of the ranker", "perfect reordering of its first 30", "white", INK2)]),
              ("over the largest ranker", [("S_full_ext", "largest ranker alone", INK2, None),
                                           ("llm_rerank_S_full_ext_blank", "GPT-5.6, law and article number only", C.tint(OCHRE), None),
                                           ("llm_rerank_S_full_ext", "GPT-5.6, article text", OCHRE, None),
                                           ("oracle@30 of S_full_ext", "perfect reordering of its first 30", "white", INK2)]),
              ("over the ranker without embeddings", [("S0", "ranker without embeddings alone", BLUE, None),
                                                      ("llm_rerank_S0", "GPT-5.6, article text", OCHRE, None),
                                                      ("oracle@30 of S0", "perfect reordering of its first 30", "white", INK2)])]
    slots = []
    for head, rows in groups:
        rows = [r for r in rows if r[0] in t.index and pd.notna(t.loc[r[0], "ndcg_at_10"])]
        if rows:
            slots += ([("g", None)] if slots else []) + [("h", head)] + [("r", r) for r in rows]
    missing = [r[0] for _, rows in groups for r in rows if r[0] not in t.index]
    if missing:
        ctx.notes.setdefault(stem, []).append(f"not in the record: {', '.join(missing)}")
    if not slots:
        return _placeholder(ctx, stem, cap, "The reread", label, "no reread results")
    n = len(slots)
    h = 0.19 * n + 0.6
    fig = _new(h)
    ax = fig.add_axes([0.42, 0.45 / h, 0.555, 0.19 * n / h])
    ys, labels, colors = [], [], []
    for i, (kind, r) in enumerate(slots):
        y = n - 1 - i
        if kind == "h":
            C.group_heading(ax, y, r)
        elif kind == "r":
            sid, text, fill, edge = r
            v, lo, hi = (float(t.loc[sid, k]) for k in ("ndcg_at_10", "lo", "hi"))
            ax.barh(y, v, 0.62, color=fill, edgecolor=edge or fill, lw=0.7 if edge else 0, zorder=2)
            ax.hlines(y, lo, hi, color=INK, lw=0.7, zorder=3)
            ys.append(y); labels.append(text); colors.append(GREEN if fill == GREEN else INK2)
    _rows_axes(ax, ys, labels, colors)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0, 0.85)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "x")
    ax.set_xlabel(C.axis_title("nDCG@10"))
    return _write(fig, ctx, stem, cap, "The reread", label)


# ---- 6.11 (F6_10): quality against cost -----------------------------------------------------------------------------
OPERATING = {"lstar_headings": "tuned lexical matching", INTEGRATED: "the ranker", SYSTEM: "the system"}
OP_STYLE = {"lstar_headings": dict(marker="s", color=INK), INTEGRATED: dict(marker="o", color=GREEN),
            SYSTEM: dict(marker="D", color=GREEN)}


# Costs from the whole-method timing (D176, H3; DC, report cost_timing_20261028, as numbers_cost_timing.tex)
COST_TIMING = C.ROOT / "research" / "reports" / "cost_timing_20261028" / "RESULT.json"
TIMED_COST = {"lstar_headings": "lstar(timed)", "S0": "S0", "S_full_ext": "S_full_ext",
              "llm_rerank_S_full_ext": "llm_rerank_S_full_ext", "llm_rerank_S0": "llm_rerank_S0"}
# D181 (audit of v10): the whole-method timing measured the extended ranker, which contains the integrated method and
# adds the older tesis' vote; the integrated method is priced at that measured cost, an upper bound (FIX_SPEC 4)
# D185: the lean integrated method and the ranker with every encoder both use a subset of the extended ranker's sources
UPPER_BOUND_COST = {EVERY: "S_full_ext"}
# D188: the ranker and the system at their own cost (research/reports/thesis_audit_20261028/tools/D188_numbers.py)
FINAL_NUMBERS = FINAL_RECOMPUTE / "numbers.json"
# The LLM sweep (D174; run llm_sweep_20261028, registered): other models in place of the LLM stage's, on DEV-A300
SWEEP_RESULTS = C.ROOT / "experiments" / "runs" / "llm_sweep_20261028" / "report" / "SWEEP_RESULTS.json"
SWEEP_NAME = {"claude": "Claude Opus 5.5", "sonnet": "Claude Sonnet 5.5", "haiku": "Claude Haiku 5.5",
              "gpt6sol": "GPT-6 Sol"}
# The other systems on the frontier, in order of cost (f6_10 asserts that the record's frontier is exactly this
# list); all but the ranker without embeddings, named in blue, are numbered in the figure and named in the caption
FRONT_ORDER = ["bm25_stem", "qwen3_0p6b_ft", "linear__lstar_headings__qwen3_0p6b", "qwen3_4b", "S0", "S0+qwen3_0p6b",
               "S0+qwen3_4b", "S_full_ext", "llm_rerank_S_full_ext_blank", "llm_rerank_S_full_ext"]
FRONT_KEY = {"bm25_stem": "BM25", "qwen3_0p6b_ft": "Qwen3-Embedding-0.6B trained on the task",
             "linear__lstar_headings__qwen3_0p6b": "the weighted sum of lexical matching and Qwen3-Embedding-0.6B",
             "qwen3_4b": "Qwen3-Embedding-4B",
             "S0+qwen3_0p6b": "the ranker without embeddings with the score of Qwen3-Embedding-0.6B",
             "S0+qwen3_4b": "the same with Qwen3-Embedding-4B",
             "S_full_ext": "the largest ranker",
             "llm_rerank_S_full_ext_blank": "the GPT-5.6 reread given only each article's law and number",
             "llm_rerank_S_full_ext": "the GPT-5.6 reread over the largest ranker"}


def f6_10(ctx: Ctx) -> Path:
    """Quality against USD per 1,000 cases (log): every system grey, the frontier thin, the operating points with
    their own colour and marker, the other frontier points numbered; a reference line at one cent per case."""
    stem, label = "F6_10_frontier", "fig:F6_10_frontier"
    timing = json.loads(COST_TIMING.read_text(encoding="utf-8"))
    timed = timing["usd_per_1000_timed"]
    numbered = [s for s in FRONT_ORDER if s in FRONT_KEY]
    key = "; ".join(f"{i}, {FRONT_KEY[s]}" for i, s in enumerate(numbered, 1))
    cap = ("Ranking quality against cost per 1,000 cases, on a logarithmic scale: the ranker reaches most of the "
           "attainable quality at well under a cent per case, and the reread adds a further gain at a few cents. (a) The "
           "development cases; the line joins the systems that no cheaper system matches. Numbered: " + key + ". (b) The "
           "sample of 300 cases, with language models rereading the largest ranker, priced by their tokens (the Claude "
           "models as billed in batch).")
    q = ctx.csv("c8/c8_quantities.csv")
    if q is not None and "usd_per_1000" not in q.columns and "usd_per_1000_base" in q.columns:
        q = q.rename(columns={"usd_per_1000_base": "usd_per_1000"})      # c8_frontier.py writes the base price here
    if q is None or not {"system", "usd_per_1000", "ndcg_at_10"} <= set(q.columns):
        return _placeholder(ctx, stem, cap, "Quality against cost", label,
                            "quality against cost per 1,000 cases, waiting for the cost records of the run")
    final = json.loads(FINAL_NUMBERS.read_text(encoding="utf-8"))
    for sid, usd in ((INTEGRATED, final["usd_ranker"]), (SYSTEM, final["usd_system_batch"])):
        q = q[q["system"] != sid]
        q = pd.concat([q, pd.DataFrame([{"system": sid, "ndcg_at_10": ctx.summary(sid)["ndcg_at_10"], "usd_per_1000": usd}])],
                      ignore_index=True)
    for sid, k in TIMED_COST.items():
        if k in timed:
            q.loc[q["system"] == sid, "usd_per_1000"] = timed[k]
    for sid, src in UPPER_BOUND_COST.items():
        q.loc[q["system"] == sid, "usd_per_1000"] = timed[src]
        v = float(q.loc[q["system"] == sid, "ndcg_at_10"].iloc[0])
        cheaper = q[(q["usd_per_1000"] < timed[src]) & (q["ndcg_at_10"] >= v)]
        # at any true cost up to the bound no cheaper system matches it, so it is on the frontier whatever its cost
        assert cheaper.empty, cheaper["system"].tolist()
    # at equal cost: the measured largest ranker, then the ranker with every encoder
    q["_tie"] = q["system"].map(lambda s: {EVERY: 2}.get(s, 1))
    q = q.dropna(subset=["usd_per_1000", "ndcg_at_10"]).sort_values(["usd_per_1000", "_tie"])
    front, best = [], -1.0
    for _, r in q.iterrows():
        if r["ndcg_at_10"] > best:
            front.append(r); best = r["ndcg_at_10"]
    on = [r["system"] for r in front if r["system"] not in OPERATING]
    assert on == FRONT_ORDER, on            # the frontier of the record, as the caption names it
    fig = _new(4.1)
    ax = fig.add_axes([0.115, 2.1 / 4.1, 0.855, 1.72 / 4.1])
    C.panel(ax, "a", "development cases")
    ax.axvline(10, color=RULE, lw=0.6, ls=(0, (1, 2)), zorder=0)
    _label(ax, 10, 0.03, "one cent per case", color=MUTED, dx=-3, ha="right", va="bottom")
    ax.plot(q["usd_per_1000"], q["ndcg_at_10"], "o", ms=3.2, color=MARK, mec="white", mew=0.4, zorder=2)
    fx = [r["usd_per_1000"] for r in front]
    fy = [r["ndcg_at_10"] for r in front]
    ax.step(fx, fy, where="post", color=MUTED, lw=0.7, zorder=1)            # ends at the last frontier point
    ax.plot(fx, fy, "o", ms=3.2, color=INK2, mec="white", mew=0.4, zorder=3)
    pos = {r["system"]: (r["usd_per_1000"], r["ndcg_at_10"]) for r in front}
    num_at = {"S0+qwen3_0p6b": (3, -4, "left", "top"),      # close points numbered on either side
              "linear__lstar_headings__qwen3_0p6b": (3, -4, "left", "top")}
    for i, s in enumerate(numbered, 1):
        if s in pos:
            dx, dy, ha, va = num_at.get(s, (-2, 4, "right", "bottom"))
            _label(ax, *pos[s], str(i), color=INK2, dx=dx, dy=dy, ha=ha, va=va)
    if "S0" in pos:
        ax.plot(*pos["S0"], "o", ms=4, color=BLUE, mec="white", mew=0.4, zorder=4)
        _label(ax, *pos["S0"], NAME["S0"], color=BLUE, dx=-5, dy=-2, ha="right", va="top")
    spots = {"lstar_headings": (0, 9, "center"), INTEGRATED: (-6, 9, "right"), SYSTEM: (0, -12, "center")}
    for _, r in q.iterrows():
        if r["system"] in OPERATING:
            st = OP_STYLE[r["system"]]
            ax.plot([r["usd_per_1000"]], [r["ndcg_at_10"]], st["marker"], ms=5, color=st["color"], mec="white",
                    mew=0.4, zorder=5)
            dx, dy, ha = spots[r["system"]]
            _label(ax, r["usd_per_1000"], r["ndcg_at_10"], OPERATING[r["system"]], color=st["color"],
                   dx=dx, dy=dy, ha=ha, size=8)
    missing = [s for s in OPERATING if s not in set(q["system"])]
    if missing:
        ctx.notes.setdefault(stem, []).append(f"operating points without a cost record: {', '.join(missing)}")
    ax.set_xscale("log")
    ax.set_xlim(5e-5, 200)
    _logfmt(ax, "x")
    ax.set_ylim(0, 0.6)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    C.strip0(ax, "y")
    ax.set_xlabel("USD per 1,000 cases (logarithmic scale)")
    ax.set_ylabel(C.axis_title("nDCG@10"))
    _legend(ax, [Line2D([0], [0], marker="o", ls="none", ms=3.2, color=INK2, label="other system on the line, numbered"),
                 Line2D([0], [0], marker="o", ls="none", ms=3.2, color=MARK, label="other system"),
                 Line2D([0], [0], color=MUTED, lw=0.7, label="no cheaper system matches it")],
            "upper left")
    _sweep_panel(timing["llm_token_part_usd_per_1000"], fig.add_axes([0.115, 0.42 / 4.1, 0.855, 1.1 / 4.1]), ctx, q, stem)
    return _write(fig, ctx, stem, cap, "Quality against cost", label)


def _sweep_panel(llm_tokens_usd: float, b, ctx: Ctx, q: pd.DataFrame, stem: str) -> None:
    """Panel b of 6.11: the largest ranker and the GPT-5.6 reread over it on DEV-A300 (their scores on the same groups, from
    the sweep's paired contrasts) beside the other models of the sweep; the cost per case of the first two is the
    record's."""
    if not SWEEP_RESULTS.exists():
        ctx.notes.setdefault(stem, []).append("no sweep results")
        return
    rows = json.loads(SWEEP_RESULTS.read_text(encoding="utf-8"))
    usd = dict(zip(q["system"], q["usd_per_1000"]))
    ref = rows[0]
    pts = [("S_full_ext", "largest ranker", usd["S_full_ext"], ref["vs_S_full_ext"]["mean_b"], INK2, True, "^",
            (6, -8, "left")),
           # the LLM stage by its tokens alone (\ctUsdLLMtok), the basis of the sweep models and of 6.5's text
           ("llm_rerank_S_full_ext", "GPT-5.6", llm_tokens_usd,
            ref["vs_llm_rerank_S_full_ext"]["mean_b"], OCHRE, True, "D", (-6, 0, "right"))]
    spots = {"claude": (6, 0, "left"), "sonnet": (6, 0, "left"), "haiku": (-6, 0, "right"), "gpt6sol": (6, 0, "left")}
    for r in rows:
        assert r["vs_S_full_ext"]["mean_b"] == ref["vs_S_full_ext"]["mean_b"]    # every model on the same groups
        assert r["usd_basis_kind"] == ("billed" if r["kind"] != "gpt6sol" else "list-price equivalent (Codex)")
        pts.append((r["kind"], SWEEP_NAME[r["kind"]], r["usd_per_1000_cases"], r["ndcg_at_10"], OCHRE, False, "o",
                    spots[r["kind"]]))
    b.axvline(10, color=RULE, lw=0.6, ls=(0, (1, 2)), zorder=0)
    for _, text, x, y, col, filled, mk, (dx, dy, ha) in pts:
        b.plot([x], [y], mk, ms=5, mfc=col if filled else "white", mec=col, mew=1.0, zorder=3)
        _label(b, x, y, text, color=INK2, dx=dx, dy=dy, ha=ha)
    b.set_xscale("log")
    b.set_xlim(0.05, 400)
    _logfmt(b, "x")
    b.set_ylim(0.45, 0.65)
    b.yaxis.set_major_locator(MultipleLocator(0.05))
    C.strip0(b, "y")
    b.set_xlabel("USD per 1,000 cases (logarithmic scale)")
    b.set_ylabel(C.axis_title("nDCG@10"))
    C.panel(b, "b", "300 development cases: language models rereading the largest ranker")


# ---- 6.8 (F6_11): recall by depth -----------------------------------------------------------------------------------
SWEEP_SYSTEMS = [RANKER8, NOENC5, "qwen3_8b", "lstar_headings"]
SWEEP_NAME = {RANKER8: "the ranker", NOENC5: "the ranker without the encoder"}
SWEEP_STYLE = {RANKER8: dict(color=GREEN, ls="-"), NOENC5: dict(color=INK, ls="-"),
               "qwen3_8b": dict(color=INK2, ls=(0, (4, 1.5))), "lstar_headings": dict(color=MUTED, ls=(0, (1, 1.2)))}


def _sweep_from_ranks(pc: pd.DataFrame, ks: np.ndarray) -> np.ndarray:
    """Recall@k at article level as recall/recall_sweep.csv gives it: each case's share of its cited articles among the
    first k, averaged within a component, then over components."""
    rr = [np.asarray(_ints(x)) for x in pc["positive_ranks"]]
    n = pc["n_relevant"].to_numpy(float)
    per = np.array([[((r > 0) & (r <= k)).sum() / m if m else np.nan for k in ks] for r, m in zip(rr, n)])
    return pd.DataFrame(per).groupby(pc["component_id"].to_numpy()).mean().mean().to_numpy()


def f6_recall(ctx: Ctx) -> Path:
    """Recall@k, k = 1..200 (articles); the depths 10 and 30 marked and named. H9 (the author, 2 Nov 2026): the panel of laws is
    dropped, since F6_8 makes its point and the text gives its number."""
    stem, label = "F6_11_recall_sweep", "fig:F6_11_recall_sweep"
    cap = ("Share of the cited articles found among the first $k$ articles, on the development cases. The lines at 10 and 30 "
           "mark the results a reader sees and those a language model reranks.")
    rs = ctx.csv("recall/recall_sweep.csv")
    if rs is None:
        return _placeholder(ctx, stem, cap, "Recall by depth", label, "no recall sweep")
    fig = _new(2.25)
    a = fig.add_axes([0.085, 0.19, 0.895, 0.76])
    handles, seen = [], set()
    ref = rs[(rs["system"] == "S0") & (rs["population"] == "DEV") & (rs["level"] == "article")
             & (rs["measure"] == "recall")].sort_values("k")
    s0 = ctx.per_case("S0")
    if s0 is None or not np.allclose(_sweep_from_ranks(s0, ref["k"].to_numpy()), ref["value"].to_numpy(), atol=1e-4):
        plt.close(fig)
        return _placeholder(ctx, stem, cap, "Recall by depth", label, "the sweep from ranks does not reproduce S0")
    for s in SWEEP_SYSTEMS:
        if s in SWEEP_NAME:
            f = FOOT_STORE / s / "DEV" / "per_case.parquet"
            if not f.exists():
                ctx.notes.setdefault(stem, []).append(f"{s}: no per-case file")
                continue
            ks = ref["k"].to_numpy()
            art = pd.DataFrame({"k": ks, "value": _sweep_from_ranks(pd.read_parquet(f), ks)})
        else:
            sid = ctx.sid(s)
            if sid is None or sid in seen:
                continue
            seen.add(sid)
            d = rs[(rs["system"] == sid) & (rs["population"] == "DEV")]
            art = d[(d["level"] == "article") & (d["measure"] == "recall")].sort_values("k")
        st = SWEEP_STYLE[s]
        if len(art):
            a.plot(art["k"], art["value"], lw=1.0, **st)
        if len(art):
            handles.append(Line2D([0], [0], lw=1.0, label=SWEEP_NAME.get(s) or NAME[s] + (" alone" if s == "qwen3_8b" else ""),
                                  **st))
        else:
            ctx.notes.setdefault(stem, []).append(f"{s}: no sweep rows")
    if not handles:
        plt.close(fig)
        return _placeholder(ctx, stem, cap, "Recall by depth", label, "no recall sweep rows")
    for k, t, dx, ha in ((10, "shown to"+"\n"+"the reader", -3, "right"), (30, "reranked by"+"\n"+"a language model", 3, "left")):
        a.axvline(k, color=RULE, lw=0.6, zorder=0)
        a.annotate(t, (k, 0.98), xycoords=("data", "axes fraction"), xytext=(dx, 0), textcoords="offset points",
                   ha=ha, va="top", fontsize=7, color=MUTED, multialignment=ha)
    a.set_xscale("log")
    _logfmt(a, "x")
    a.set_xlabel("first $k$ articles, development cases")
    a.set_ylabel("cited articles found, share")
    a.set_ylim(0, 1)
    C.strip0(a, "y")
    _legend(a, handles, "lower right", frameon=True, framealpha=1, facecolor="white", edgecolor="none")
    return _write(fig, ctx, stem, cap, "Recall by depth", label)


FIGS = {"F6_1": f6_1, "F6_2": f6_2, "F6_3": f6_3, "F6_4": f6_4, "F6_5": f6_5, "F6_6": f6_6, "F6_7": f6_7,
        "F6_8": f6_8, "F6_9": f6_llm, "F6_10": f6_10, "F6_11": f6_recall}


def build(store: Path = RECORD_STORE, analysis: Path = RECORD_ANALYSIS, final: bool = False, only: list[str] | None = None,
          extra_contrasts: Path | None = RECORD_CONTRASTS):
    out = {}
    for key, fn in FIGS.items():
        if only and key not in only:
            continue
        ctx = Ctx(store=Path(store), analysis=Path(analysis), final=final,
                  extra_contrasts=Path(extra_contrasts) if extra_contrasts else None)
        out[key] = str(fn(ctx))
        out[key + "_notes"] = ctx.notes
        out[key + "_stand_in"] = sorted(ctx.used_stand_in)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", type=Path, default=RECORD_STORE)
    ap.add_argument("--analysis", type=Path, default=RECORD_ANALYSIS)
    ap.add_argument("--final", action="store_true", help="freeze data: drop the dry-run mark unless a stand-in is used")
    ap.add_argument("--only", default="")
    ap.add_argument("--extra-contrasts", type=Path, default=RECORD_CONTRASTS,
                    help="a further contrasts folder (default: the record's copy of RUN-B's raw/contrasts)")
    a = ap.parse_args(argv)
    res = build(a.store, a.analysis, a.final, [s for s in a.only.split(",") if s] or None, a.extra_contrasts)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
