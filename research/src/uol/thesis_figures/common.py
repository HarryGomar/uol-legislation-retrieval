"""Shared paths, style, palette, captions and the phase-7 (test-set batch) reader for the thesis figure pack.

Visual system: figures_staging/STYLE_FIGURES.md. Latin Modern Sans (the thesis's sans family, from the MiKTeX tree;
DejaVu Sans if it is missing), 8 pt labels and 7 pt ticks, drawn at the 12.46 cm text width so figures print 1:1, and
embedded as Type 42 (pdf.fonttype 42), never Type 3. Greys carry structure; ITAM green marks the figure's focus (D83; formerly blue for embeddings, vermilion for the LLM
stage. Vocabulary and captions follow PLAIN_LANGUAGE.md (D54): test set, development set, evaluation cases, similar-case
citations, article profiles, LLM stage; captions are a title-like sentence plus at most one finding, with no evidence
tails. EVIDENCE.md keeps the evidence labels; the Item.label field records them for audit only.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
STAGING = ROOT / "research" / "thesis_direction_20261015" / "figures_staging"
FIG_DIR = STAGING / "figures"
TAB_DIR = STAGING / "tables"

# Saved result files (relative to ROOT). EVIDENCE.md cites each of them or the report that summarises them.
EL = "experiments/runs/eval_lenses_20261015"
R3 = "experiments/runs/round3_best_case_20261013"
R3_ARMS = f"{R3}/arms"
PHASE7 = ROOT / R3 / "phase7"
EP_RESULTS = "research/reports/embedding_push_20261004/RESULTS.md"
R2_SUMMARY = "research/reports/round2_20261004/SUMMARY.md"
R2_A_SCALING = "experiments/runs/round2_20261004/A/p03d_encoder_scaling.json"
R2_LAY = "experiments/runs/round2_20261004/H/results.json"
R2_SETS = "research/reports/round2_20261004/C_confirmation_sets.md"
SCALE = "experiments/runs/scale_curve_20261016/eval/dense_alone.json"
RELEASE = "data/releases/uol_c01_statutes_20261004_v11_stratified/manifest.json"
LEDGER = "research/state/round3_best_case_20261013/LOCKBOX_LEDGER.jsonl"

TEXTWIDTH_IN = 354.52 / 72.27      # thesis \textwidth (measured: 354.52 pt = 12.46 cm on the 17 x 22.5 cm page)
TEXTHEIGHT_IN = 492.23 / 72.27
# Plain names (PLAIN_LANGUAGE.md section 2, approved as D57 item 13). Old names (lockbox, P_spec, components, layer,
# memory, prototypes) are never printed; BANNED below enforces it on every caption, label and table body.
EVAL = "evaluation cases"
TEST = "test set"
DEVSET = "development set"

# Visual system of figures_staging/STYLE_FIGURES.md (D83, 22 Oct 2026): greys for structure, one colour (ITAM green)
# for the figure's focus and one tint of it; the older names below map onto it so earlier modules still run.
SURFACE = "#ffffff"
INK, INK2, MUTED, RULE, GRID, FILL = "#1a1a1a", "#5a5a5a", "#8c8c8c", "#c8c8c8", "#ebebeb", "#efefef"
MARK = "#bdbdbd"          # ordinary bars and points (not the focus)
ITAM_GREEN = "#006648"    # the focus: the one thing the caption is about (ITAM logo green)
FOCUS, FOCUS_TINT = ITAM_GREEN, "#8fbfb0"
BLUE = FOCUS              # was the embeddings accent; now the focus colour
BLUE_LIGHT = FOCUS_TINT
VERM = FOCUS              # was the LLM-stage accent
TEAL = FOCUS_TINT
RAMP = ["#003d2b", "#006648", "#5e9e8a", "#b7d6cc"]          # ordered, dark = first / nearest
GREYS = ["#1a1a1a", "#4a4a4a", "#8c8c8c", "#b4b4b4"]           # baselines, dark = stronger
# Scheme P3 (the author, 26 Oct 2026, CF-04: "R1 do P3"; replaces D83's one-colour rule): ITAM green stays the focus, three
# companions tell series apart, in this order of use; greys keep context and structure only.
COMPANION_BLUE, COMPANION_PLUM, COMPANION_OCHRE = "#4a6fa5", "#9b4f78", "#c29438"
SCHEME = [ITAM_GREEN, COMPANION_BLUE, COMPANION_PLUM, COMPANION_OCHRE]
# One colour per system wherever it appears (D147): the proposed system green, the ranker blue, the embedding models
# plum, the LLM stage ochre; tuned lexical matching and every other system stay grey (context).
SYSTEM_COLOUR = {"integrated method": ITAM_GREEN, "ranker": COMPANION_BLUE, "embeddings": COMPANION_PLUM,
                 "LLM stage": COMPANION_OCHRE}


def tint(hex_colour: str, f: float = 0.45) -> str:
    """A lighter step of a scheme colour: the colour mixed with white, keeping f of it."""
    r, g, b = (int(hex_colour.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return "#" + "".join(f"{round(255 - (255 - v) * f):02x}" for v in (r, g, b))
NEUTRAL = "#d9d9d9"

# Plain names of the systems (Chapter 3 wording); internal identifiers stay in code and comments only.
NAMES = {
    "prior": "Popularity prior", "popularity": "Popularity prior",
    "L_star": "Tuned word matching",
    "dense_e5": "e5-small", "dense_e5_maxp": "e5-small",
    "dense_bge": "bge-m3", "dense_bge_maxp": "bge-m3",
    "dense_qwen8b": "Qwen3-Embedding-8B alone", "ft_e5_q": "e5-small fine-tuned on citations",
    "K_lex": "Similar-case citations, word similarity", "knn_qwen8b": "Similar-case citations, Qwen3-8B",
    "proto_qwen8b": "Article profiles, Qwen3-8B",
    "LTR_noemb_hgb": "Learned ranker without embeddings", "LTR_emb_hgb": "Learned ranker with embeddings",
    "C_hybrid": "LLM stage",
}
SHORT = {"L_star": "Tuned word matching", "dense_qwen8b": "Qwen3-Embedding-8B alone",
         "LTR_noemb_hgb": "Ranker without embeddings", "LTR_emb_hgb": "Ranker with embeddings",
         "C_hybrid": "LLM stage"}
# Colour of each system wherever systems are series (STYLE_FIGURES.md: grey = no embeddings).
SYSTEM_STYLE = {
    "L_star": dict(color=GREYS[2], ls=(0, (1.2, 1.2))),
    "dense_qwen8b": dict(color=RAMP[2], ls="-"),
    "LTR_noemb_hgb": dict(color=GREYS[1], ls="-"),
    "LTR_emb_hgb": dict(color=BLUE, ls="-"),
    "C_hybrid": dict(color=VERM, ls=(0, (3.5, 1.5))),
}

# Words that must not reach the page (PLAIN_LANGUAGE.md section 2 and the Ch3 banned-term grep).
BANNED = (r"lockbox|second[- ]use|scored (once|twice)|evaluated population|clean population|P_spec|P\$_|"
          r"\bcomponents?\b|dependency group|\brungs?\b|ladder|verdict|sealed|prospective|post-cutoff|cross-fit|"
          r"descriptive|\bdev\b|success@|first screen|citation memory|prototypes\b|article prototype|"
          r"language-model layer|LLM layer|the layer|held-out|freez|Evidence:|component-macro|\bgate\b")


def lint(text: str, where: str) -> None:
    """Raise if printed text (caption, figure label, table cell) still carries a retired term."""
    hits = sorted({m.group(0) for m in re.finditer(BANNED, text, re.I)})
    if hits:
        raise ValueError(f"{where}: retired terms {hits}")


# --------------------------------------------------------------------------------------------- sprint style (26 Oct)
# One figure style for the thesis (sprint 26 Oct 2026: STANDARD.md sections 5-6; SPRINT_IMPROVEMENT section 9;
# D125). Names come from the term sheet (SPRINT_IMPROVEMENT section 6), never codes; every axis states its measure
# and its cases; legends sit inside the plot and explain every mark; nothing is printed under an axis; text is at
# least 7 pt at print size; greys plus ITAM green for the proposed system (the integrated method); the categorical
# palette only in the catalogue map and the citation-bands figure (D125).
CASES = "development cases"
# Okabe-Ito hues without the orange accent the author retired (D81), ITAM green first; D125 limits them to two figures.
CATEGORICAL = ["#006648", "#56B4E9", "#CC79A7", "#0072B2", "#D55E00", "#E69F00"]
TERMS = {  # store id -> printed name (term sheet; encoders by model name)
    "lstar_headings": "tuned lexical matching", "bm25_stem": "BM25", "popularity": "popularity",
    "lsa": "LSA", "lda_sim": "LDA similarity", "lda_dm": "LDA document model", "doc2vec": "doc2vec",
    "word2vec": "word2vec",
    "e5_small": "E5-small", "e5_base": "E5-base", "e5_large": "E5-large", "e5_large_instruct": "E5-large-instruct",
    "gte_multilingual": "GTE-multilingual", "jina_v3": "Jina v3", "arctic_l_v2": "Arctic-Embed L v2",
    "gemma_base": "EmbeddingGemma", "bge_m3": "BGE-M3", "bge_m3_sparse": "BGE-M3, sparse",
    "bge_m3_colbert": "BGE-M3, late interaction", "qwen3_0p6b": "Qwen3-Embedding-0.6B",
    "qwen3_4b": "Qwen3-Embedding-4B", "qwen3_8b": "Qwen3-Embedding-8B",
    "ft_e5_q": "E5-small, task-trained", "bge_m3_ft": "BGE-M3, task-trained",
    "qwen3_0p6b_ft": "Qwen3-Embedding-0.6B, task-trained",
    "e5_large_legal_en": "legal E5-large", "bge_m3_legal": "legal BGE-M3", "gemma_legal": "legal EmbeddingGemma",
    "euler_legal": "legal Qwen3-Embedding-8B", "alia_mrbert": "ALIA-MrBERT",
    "src_K_lex": "lexical matching plus votes, words", "src_K_lex_ext": "lexical matching plus votes, unstructured tesis",
    "src_proto_tfidf": "article profiles, TF-IDF", "docexp_bm25": "BM25 on the article and its citing facts",
    "src_docexp_facts_bm25": "BM25 on the citing facts", "src_knn_lex_votes": "votes of similar cases, words",
    "src_knn_char_votes": "votes of similar cases, characters",
    "src_knn_qwen8b": "votes of similar cases, Qwen3-Embedding-8B",
    "src_proto_qwen8b": "article profiles, Qwen3-Embedding-8B",
    "src_K_qwen8b": "lexical matching plus profiles, Qwen3-Embedding-8B",
    "src_H_qwen8b": "lexical matching plus Qwen3-Embedding-8B",
    "src_KH_qwen8b": "lexical matching, Qwen3-Embedding-8B and votes",
    "llm_alone": "language model alone",
    "gpt61sol_alone": "GPT-6.1 Sol alone",
    "S0": "ranker without embeddings", "S_full": "ranker with embeddings", "S_full_ext": "integrated method",
    "llm_rerank_S_full_ext": "integrated method with the LLM stage",
    "llm_rerank_S0": "ranker without embeddings with the LLM stage",
}
# Printed forms the term sheet forbids ("Never" column) and figure-specific retired wording (E15: "no embeddings").
TERM_BANNED = (r"\bS0\b|S_full|\bP[1-6]\b|(?<!GPT-6 )(?<!GPT-5\.6 )\bLuna\b|lstar|L\*|\bDEV(-[A-Z0-9]+)?\b|\bA300\b|\bTEST-R\b|"
               r"\b[FN][1-4]\b|\bQ[1-3]\b|\buseful\b|full ranker|(?<!without )\bno embeddings\b|development set|"
               r"adjusted lower end|_")


def term_lint(text: str, where: str) -> None:
    """Raise if printed figure text or a caption uses a code or a name the term sheet forbids."""
    plain = re.sub(r"\\(label|ref|cref|Cref)\{[^}]*\}", "", text)
    hits = sorted({m.group(0) for m in re.finditer(TERM_BANNED, plain)})
    if hits:
        raise ValueError(f"{where}: names outside the term sheet {hits}")


def axis_title(measure: str, cases: str = CASES) -> str:
    """An axis title in the sprint form: the measure, then the cases ("nDCG@10, development cases")."""
    return f"{measure}, {cases}" if cases else measure


def nothing_under_axes(fig) -> None:
    """Raise if figure-level text sits below the lowest axes (the band shares printed under the x axis, E21)."""
    axes = [a for a in fig.axes if a.get_visible() and a.axison]
    if not axes:
        return
    floor = min(a.get_position().y0 for a in axes)
    low = [t.get_text() for t in fig.texts if t.get_text().strip() and t.get_position()[1] < floor - 1e-3]
    if low:
        raise ValueError(f"text printed under an axis: {low}")


def group_heading(ax, y: float, text: str, x_fig: float = 0.01) -> None:
    """A group heading on its own row of a dot or bar plot: bold, starting at the figure's left edge, so a long
    heading may run over the label column (x in figure fraction, y in data units)."""
    from matplotlib.transforms import blended_transform_factory
    ax.text(x_fig, y, text, transform=blended_transform_factory(ax.figure.transFigure, ax.transData), ha="left",
            va="center", fontsize=8, weight="bold", color=INK, clip_on=False)


def style() -> None:
    family = "DejaVu Sans"
    lm = Path.home() / "AppData/Local/Programs/MiKTeX/fonts/opentype/public/lm"
    for f in ("lmsans10-regular.otf", "lmsans10-bold.otf", "lmsans10-oblique.otf", "lmsans8-regular.otf"):
        if (lm / f).exists():
            font_manager.fontManager.addfont(str(lm / f))
            family = "Latin Modern Sans"
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [family, "DejaVu Sans"], "font.size": 8,
        "mathtext.fontset": "custom", "mathtext.rm": family, "mathtext.it": family + ":italic",
        "axes.unicode_minus": True, "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": RULE, "ytick.color": RULE,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "text.color": INK, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.5, "axes.titlesize": 8,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 6, "axes.labelsize": 8,
        "axes.labelpad": 3, "legend.frameon": False, "legend.fontsize": 7, "legend.handlelength": 1.6,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "lines.markersize": 4.5,
        "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.pad": 2, "ytick.major.pad": 2,
        "errorbar.capsize": 0})


def panel(ax, letter: str, title: str = "", x: float = 0.0, y: float = 1.0, pad: float = 6) -> None:
    """Panel heading in the ML-paper form: a bold lowercase letter, then a plain title, left-aligned over the axes."""
    kw = dict(xy=(x, y), xycoords="axes fraction", textcoords="offset points", ha="left", va="bottom", fontsize=8)
    if letter:
        ax.annotate(letter, xytext=(0, pad), weight="bold", color=INK, **kw)
    if title:
        ax.annotate(title, xytext=(9 if letter else 0, pad), color=INK, **kw)


def only_x(ax) -> None:
    """Dot-plot axes: no left spine or y ticks; labels sit beside the rows."""
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def vgrid(ax, xs) -> None:
    for x in xs:
        ax.axvline(x, color=GRID, lw=0.5, zorder=0)


def hgrid(ax, ys) -> None:
    for y in ys:
        ax.axhline(y, color=GRID, lw=0.5, zorder=0)


def dot(ax, x, y, held_out: bool = False, color: str = INK, size: float = 4.6, z: int = 4, marker: str | None = None,
        **kw):
    """Evidence-set marker: filled circle = dev; hollow diamond = held-out (lockbox, sealed, post-cutoff)."""
    m = marker or ("D" if held_out else "o")
    s = size * (0.86 if m == "D" else 1.0)
    if held_out:
        return ax.plot(x, y, m, ms=s, mfc=SURFACE, mec=color, mew=1.0, zorder=z, ls="none", **kw)
    return ax.plot(x, y, m, ms=s, mfc=color, mec=SURFACE, mew=0.6, zorder=z, ls="none", **kw)


def key_markers(ax, x, y, entries, color=INK2, pad=4, gap=10) -> None:
    """A one-line marker key (filled circle = dev, hollow diamond = held-out) drawn right-aligned at (x, y) in axes
    fraction, on the panel-title line, instead of a legend box. entries: [(text, held_out), ...] left to right."""
    from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea
    from matplotlib.lines import Line2D
    parts = []
    for text, held in entries:
        da = DrawingArea(7, 7, 0, 0)
        m = "D" if held else "o"
        ln = Line2D([3.5], [3.5], marker=m, ms=4.6 * (0.86 if held else 1), ls="none",
                    mfc=SURFACE if held else color, mec=color if held else SURFACE, mew=1.0 if held else 0.6)
        da.add_artist(ln)
        parts += [da, TextArea(text, textprops=dict(fontsize=7, color=INK2))]
    box = HPacker(children=parts, align="center", pad=0, sep=3)
    from matplotlib.transforms import offset_copy
    ab = AnchoredOffsetbox(loc="lower right", child=box, frameon=False, bbox_to_anchor=(x, y),
                           bbox_transform=offset_copy(ax.transAxes, fig=ax.figure, y=pad, units="points"),
                           borderpad=0, pad=0)
    ab.set_clip_on(False)
    ax.add_artist(ab)


def interval(ax, lo, hi, y, color=INK, lw=0.9, z=3):
    return ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z)


# --------------------------------------------------------------------------------------------- items and outputs
@dataclass
class Item:
    id: str                       # F1..F12, T1..T7
    stem: str                     # file stem
    target: str                   # heading(s) in HEADINGS.md
    title: str                    # short caption (list of figures)
    caption: str                  # full caption; ends with the evidence label
    label: str                    # evidence label(s) in EVIDENCE.md terms
    sources: list[str]            # ROOT-relative files read
    status: str = "ready"         # ready / ready, lockbox slot pending / ...
    internal: str = ""            # internal identifiers, for the comment only
    notes: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)


def rel(p: Path | str) -> str:
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(p)


def load_json(relpath: str) -> Any:
    return json.loads((ROOT / relpath).read_text(encoding="utf-8"))


def read_text(relpath: str) -> str:
    return (ROOT / relpath).read_text(encoding="utf-8")


def snippet(item: Item, width: str = r"\textwidth", origin: str = "") -> Item:
    """Write figures/<stem>.tex, a figure environment that includes the PDF at the text width (1:1, never scaled by
    height), after checking the caption and the figure's own text for retired terms."""
    lint(item.title + " " + item.caption, f"{item.stem} caption")
    pdf = FIG_DIR / f"{item.stem}.pdf"
    audit = pdf_audit(pdf)
    lint(audit["text"], f"{item.stem} figure text")
    item.notes.append(f"render audit: smallest text {audit['min_pt']:.1f} pt, fonts {', '.join(audit['fonts'])}, "
                      f"size {audit['w_cm']:.2f} x {audit['h_cm']:.2f} cm")
    out = FIG_DIR / f"{item.stem}.tex"
    out.write_text(
        f"% {item.id} for {item.target}. Generated by uol.thesis_figures{origin}; do not edit by hand.\n"
        f"% Internal identifiers: {item.internal or '-'}\n"
        f"% Sources: {'; '.join(item.sources)}\n"
        "\\begin{figure}[tbp]\n  \\centering\n"
        f"  \\includegraphics[width={width}]{{{item.stem}.pdf}}\n"
        f"  \\caption[{tex(item.title)}]{{{tex(item.caption)}}}\n"
        f"  \\label{{fig:{item.stem}}}\n\\end{{figure}}\n", encoding="utf-8")
    item.outputs.append(rel(out))
    return item


def pdf_audit(pdf: Path) -> dict:
    """Smallest text size (pt at print size), font names and types, page size and all text of a one-page figure PDF.
    Raises if a Type 3 font is embedded, if text is under 7 pt, or if the page is wider than the text width."""
    import fitz

    doc = fitz.open(pdf)
    pg = doc[0]
    fonts = sorted({f"{f[3].split('+')[-1]} ({f[2]})" for f in pg.get_fonts()})
    sizes, texts, boxes = [], [], []
    flags = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_MEDIABOX_CLIP          # keep text that runs off the page, to catch it
    for b in pg.get_text("dict", flags=flags)["blocks"]:
        for ln in b.get("lines", []):
            for sp in ln["spans"]:
                if sp["text"].strip():
                    sizes.append(sp["size"])
                    texts.append(sp["text"])
                    boxes.append((fitz.Rect(sp["bbox"]), sp["text"].strip()))
    w, h = pg.rect.width, pg.rect.height
    doc.close()
    out = {"min_pt": min(sizes) if sizes else 0.0, "fonts": fonts, "text": " ".join(texts),
           "w_cm": w / 72 * 2.54, "h_cm": h / 72 * 2.54}
    off = [t for r, t in boxes if r.x0 < 0.3 or r.y0 < 0.3 or r.x1 > w - 0.3 or r.y1 > h - 0.3]   # clipped or touching
    if off:
        raise ValueError(f"{pdf.name}: text outside the page: {off}")
    # overlapping labels: glyph boxes (shrunk to the x-height band, so touching lines do not count) that intersect
    core = []
    for r, t in boxes:
        hgt = r.height
        core.append((fitz.Rect(r.x0 + 0.6, r.y0 + 0.3 * hgt, r.x1 - 0.6, r.y1 - 0.25 * hgt), t))
    clash = [(a[1], b[1]) for i, a in enumerate(core) for b in core[i + 1:]
             if a[0].intersects(b[0]) and (a[0] & b[0]).get_area() > 1.0]
    if clash:
        raise ValueError(f"{pdf.name}: overlapping text: {clash[:6]}")
    if any("Type3" in f for f in fonts):
        raise ValueError(f"{pdf.name}: Type 3 font embedded: {fonts}")
    if sizes and out["min_pt"] < 6.95:
        raise ValueError(f"{pdf.name}: text at {out['min_pt']:.2f} pt (minimum 7)")
    if w > 354.52 * 72 / 72.27 + 1.5:          # stroke overhang of edge boxes is allowed
        raise ValueError(f"{pdf.name}: {out['w_cm']:.2f} cm wide, wider than the 12.46 cm text width")
    return out


def save(fig, item: Item, width: str = r"\textwidth") -> Item:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = FIG_DIR / f"{item.stem}.{ext}"
        # Fixed canvas (no tight bbox): the PDF is exactly the designed width, so includegraphics at \textwidth
        # prints it 1:1 and 8 pt type stays 8 pt.
        fig.savefig(p, dpi=300, metadata={"CreationDate": None} if ext == "pdf" else None)
        item.outputs.append(rel(p))
    plt.close(fig)
    return snippet(item, width)


def write_table(item: Item, body: str) -> Item:
    printed = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("%"))
    lint(re.sub(r"\\(label|ref|cref|Cref)\{[^}]*\}", "", printed), f"{item.stem} table")
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    p = TAB_DIR / f"{item.stem}.tex"
    p.write_text(
        f"% {item.id} for {item.target}. Generated by uol.thesis_figures; do not edit by hand.\n"
        f"% Internal identifiers: {item.internal or '-'}\n"
        f"% Sources: {'; '.join(item.sources)}\n"
        f"% Needs booktabs and tabularx (both loaded in thesis/config.tex).\n{body}", encoding="utf-8")
    item.outputs.append(rel(p))
    return item


def table_env(item: Item, inner: str, size: str = r"\small") -> str:
    return ("\\begin{table}[tbp]\n  \\centering\n  " + size + "\n"
            f"  \\caption[{tex(item.title)}]{{{tex(item.caption)}}}\n  \\label{{tab:{item.stem}}}\n"
            + inner + "\\end{table}\n")


def tabular(colspec: str, header: list[str], rows: list[list[str] | str], x: bool = True) -> str:
    """booktabs block; a row given as a str is inserted verbatim (e.g. a \\midrule or a panel heading)."""
    env = "tabularx" if x else "tabular"
    width = "{\\textwidth}" if x else ""
    head = header if isinstance(header, str) else " & ".join(header) + r" \\"
    out = [f"  \\begin{{{env}}}{width}{{{colspec}}}", "    \\toprule", "    " + head, "    \\midrule"]
    for r in rows:
        out.append("    " + (r if isinstance(r, str) else " & ".join(r) + r" \\"))
    out += ["    \\bottomrule", f"  \\end{{{env}}}"]
    return "\n".join(out) + "\n"


def n0(x: int | float) -> str:
    return f"{x:,.0f}"


_TEX = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "≥": r"$\geq$", "≤": r"$\leq$", "×": r"$\times$", "→": r"$\rightarrow$",
        "−": r"$-$", "±": r"$\pm$", "Δ": r"$\Delta$", "≈": r"$\approx$", "…": r"\ldots{}"}


def tex(s: str) -> str:
    """Escape plain text for LaTeX; inline math already written as $...$ is kept (no other $ in captions)."""
    parts = re.split(r"(\$[^$]+\$|\\emph\{[^{}]*\})", s)
    out = []
    for part in parts:
        if (part.startswith("$") and part.endswith("$") and len(part) > 1) or part.startswith("\\emph{"):
            out.append(part)
        else:
            out.append("".join(_TEX.get(c, c) for c in part))
    return "".join(out)


def f3(x: float | None, sign: bool = False, tex_minus: bool = False, nd: int = 3) -> str:
    """0.472 style (leading zero kept, D149); sign=True gives +0.083 / −0.004."""
    if x is None:
        return "—"
    s = f"{abs(x):.{nd}f}"
    neg = x < 0 and float(s) != 0
    if neg:
        s = ("$-$" if tex_minus else "−") + s
    elif sign:
        s = "+" + s
    return s


def ci(c: Iterable[float], tex_minus: bool = False) -> str:
    lo, hi = list(c)
    return f"[{f3(lo, True, tex_minus)}, {f3(hi, True, tex_minus)}]"


def sha256(relpath: str) -> str | None:
    p = ROOT / relpath
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def strip0(ax, which: str = "y", nd: int = 2) -> None:
    """Tick labels in the thesis's 0.25 style (leading zero kept, D149; trailing zeros dropped; 0 and 1 as integers)."""
    from matplotlib.ticker import FuncFormatter

    def fmt(v, _):
        if abs(v) < 1e-9:
            return "0"
        if abs(v - round(v)) < 1e-9:
            return f"{v:.0f}"
        return f3(v, nd=nd).rstrip("0").rstrip(".") if "." in f3(v, nd=nd) else f3(v, nd=nd)
    (ax.yaxis if which == "y" else ax.xaxis).set_major_formatter(FuncFormatter(fmt))


def direct_label(ax, x, y, text, **kw):
    kw.setdefault("fontsize", 7)
    kw.setdefault("color", INK2)
    kw.setdefault("va", "center")
    return ax.text(x, y, text, **kw)


# --------------------------------------------------------------------------------------------- phase-7 lockbox reader
_METRIC_KEYS = {"ndcg_at_10": ("ndcg_at_10", "ndcg@10", "nDCG@10", "ndcg10"),
                "success_at_10": ("success_at_10", "success@10"),
                "recall_at_20": ("recall_at_20", "recall@20", "R@20")}


def _find(o: Any, pred, path=()) -> list[tuple[tuple, Any]]:
    hits = []
    if isinstance(o, dict):
        if pred(o, path):
            hits.append((path, o))
        for k, v in o.items():
            hits += _find(v, pred, path + (str(k),))
    elif isinstance(o, list) and len(o) < 50:
        for i, v in enumerate(o):
            hits += _find(v, pred, path + (str(i),))
    return hits


def phase7_exists() -> bool:
    return (PHASE7 / "LOCKBOX_BATCH_RECEIPT.json").exists()


def phase7_value(system: str, population: str = "P_spec", metric: str = "ndcg_at_10") -> float | None:
    """Component-macro lockbox value of `system` from the batch's metric summary, or None.

    Reads only phase7/metrics/{system}.json (a summary the batch writes after its single scoring), never rankings or
    labels, and only once LOCKBOX_BATCH_RECEIPT.json exists. The schema is not fixed yet (BATCH_PLAN.json describes
    per-query metrics), so this looks for a dict under a key naming the population that holds the metric; per-query
    lists are not averaged here (component-macro needs the component map). If nothing matches, the slot stays empty
    and the build log says so.
    """
    p = PHASE7 / "metrics" / f"{system}.json"
    if not phase7_exists() or not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    keys = _METRIC_KEYS.get(metric, (metric,))
    macro = (d.get(f"{population}_v11") or {}).get("macro") or {}      # batch schema of 16 Oct 2026
    for k in keys:
        if isinstance(macro.get(k), (int, float)):
            return float(macro[k])

    def pred(o, path):
        return any(k in o and isinstance(o[k], (int, float)) for k in keys) and \
            any(population == seg for seg in path) and not any(s in ("per_query", "rows") for s in path)
    hits = _find(d, pred)
    hits.sort(key=lambda h: (not any(s in ("summary", "component_macro", "macro") for s in h[0]), len(h[0])))
    for _, o in hits:
        for k in keys:
            if k in o:
                return float(o[k])
    print(f"[thesis_figures] phase7 metrics for {system} found but no {population}/{metric} summary: slot left empty")
    return None


PHASE7_CONTRAST = {  # arm -> (contrast file, key or None for a flat file, population note)
    "S2": ("T1.json", "S2", "all"), "S3": ("T1.json", "S3", "all"), "M1": ("T1.json", "M1", "all"),
    "E2_adapter": ("T1.json", "E2_adapter", "all"), "E3_graph": ("T1.json", "E3_graph", "all"),
    "L1": ("LLM_post_cutoff.json", "L1", "post-cutoff"), "L2": ("LLM_post_cutoff.json", "L2", "post-cutoff"),
    "L4": ("LLM_post_cutoff.json", "L4", "post-cutoff"), "C_hybrid": ("LLM_post_cutoff.json", "C_hybrid", "post-cutoff"),
    "C_hybrid_ltr": ("C_hybrid_stage2_vs_EMB.json", None, "all"), "X0_noise": ("calibration_X0.json", None, "all"),
}


def phase7_contrast(arm: str) -> dict | None:
    """Lockbox Δ, 95% interval and family-adjusted class for an arm vs the ranker with embeddings (second use; the
    language-model arms on the 145 post-cutoff components)."""
    if not phase7_exists() or arm not in PHASE7_CONTRAST:
        return None
    fn, key, pop = PHASE7_CONTRAST[arm]
    p = PHASE7 / "contrasts" / fn
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    o = d.get(key) if key else d
    if not o or "delta" not in o:
        return None
    return {"delta": float(o["delta"]), "ci95": [float(x) for x in o["ci95"]], "class": o.get("class"),
            "components": o.get("components") or o.get("n"), "population": pop, "file": rel(p)}


EL_REPORT = "research/reports/eval_lenses_20261015/REPORT.md"
LOCKBOX_LENSES = "experiments/runs/eval_lenses_20261015/lockbox_lenses/second_use.json"


def lockbox_lenses() -> dict | None:
    """The eval-lenses second-use lockbox lenses, used only once its REPORT has a 'Lockbox lenses' section (the
    session's sign-off that the file is final)."""
    rep, js = ROOT / EL_REPORT, ROOT / LOCKBOX_LENSES
    if not (rep.exists() and js.exists()):
        return None
    if not re.search(r"^#+ .*lockbox lenses", rep.read_text(encoding="utf-8"), re.I | re.M):
        return None
    return json.loads(js.read_text(encoding="utf-8"))


def ledger_batch_date() -> str | None:
    p = ROOT / LEDGER
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("purpose") == "phase7_batch":
            return str(r.get("utc") or r.get("date_utc") or r.get("timestamp") or r.get("ts") or "")[:10] or None
    return None


def ep_lockbox_ndcg() -> dict[str, float]:
    """Lockbox P_spec nDCG@10 per frozen pipeline from the 4 Oct scoring (first use), as printed in RESULTS.md.

    The first 'by pipeline' line is the on/after-release subset, which is the whole lockbox (no row precedes the
    encoder's release)."""
    md = read_text(EP_RESULTS)
    line = re.search(r"Lockbox nDCG@10 by pipeline on this subset: (.+)", md).group(1)
    return {m.group(1): float(m.group(2)) for m in re.finditer(r"([\w]+): P_spec ([\d.]+)", line)}


def lay_abs(system: str, arm: str = "O", pop: str = "P_spec", subset: str = "all") -> float | None:
    """nDCG@10 of a frozen pipeline on one arm of the lay-register test (lockbox rows; arm O = original facts)."""
    d = load_json(R2_LAY)["absolute"].get(f"{subset}|{pop}|{system}|{arm}")
    return None if d is None else float(d["ndcg_at_10"])


def lockbox_reported(system: str) -> tuple[float | None, str]:
    """Already-reported lockbox nDCG@10 (first-use scoring of 4 Oct) for a ladder system, with its source."""
    ep = ep_lockbox_ndcg()
    if system in ep:
        return ep[system], EP_RESULTS
    alias = {"dense_bge": "dense_bge_maxp"}.get(system, system)
    v = lay_abs(alias)
    return (v, R2_LAY) if v is not None else (None, "")


def item_dict(item: Item) -> dict:
    d = asdict(item)
    d["source_sha256"] = {s: sha256(s) for s in item.sources}
    return d
