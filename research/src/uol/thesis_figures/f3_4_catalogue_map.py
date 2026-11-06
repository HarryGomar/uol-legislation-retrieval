"""Catalogue map for 3.4 (reframe M4, D102): the saved UMAP of the catalogue, coloured by law, cited articles marked.

A variant of f2_4_map.py. Same source, nothing recomputed: the saved 2-D UMAP of all 39,178 Qwen3-Embedding-8B
article vectors (E1, arms/E1_structure/umap_articles.npy, rows in the order of the release's candidates/articles.jsonl,
checked here against the statute graph's 'inst' index) and the development citations of the v11 release
(references.jsonl positives of the 4,812 development cases; the test set, 'lockbox' in SPLIT.json, is left out),
with the final run's three label corrections (release/datapack/uol-benchmark-v11/benchmark/label_corrections.json).
An article is cited when at least one development case cites it (TERMS: never-cited articles); the count must equal
\\nCitedArticlesDev (2,407; 2,409 before the corrections).

Redrawn by sprint session C (26 Oct 2026; SPRINT_IMPROVEMENT section 9, the author's E10 "colour by the law and then X the
citation", D125 categorical palette for this figure). The six laws with the most cited articles take the categorical
colours and the other laws light grey; "largest" is read by cited articles, because by article count the six largest
include the Ordenanza General de la Armada (0 cited) and the Codigo de Justicia Militar (1), which say nothing about
citation. Never-cited articles are faint grey dots for every law, and cited articles larger, darker crosses in their
law's colour (dark grey for the other laws; R review 4.2, 25 Oct); the article of the running example (tesis 2028281, LGSM art. 2) is ringed. The
legend, in the thesis body font, names the laws.

Redrawn by LF (26 Oct 2026, CF-01: "too much space on the legend, not enough on the map"; the author chose option C, then
"labels next to the clusters nicely and just has the X for cited and O for not cited", then "remove the X on the side
of the things, ensure the colour x are drawn above the other ones"; scratch in sprint_20261026/logs/LF_map). The map is
10 cm; each law is named in the white space next to the densest part of its cited articles, joined by a thin grey line;
the ring is named beside it; the key keeps only the two marks. Colours P3 (D147): the four scheme colours, then lighter
steps of blue and ochre for the fifth and sixth law; the coloured crosses are drawn last, the scheme colours on top.

Output: thesis/figures/F3_4_catalogue_map_points.pdf (points only, rasterised, no text) and
thesis/figures/F3_4_catalogue_map.tex (a tikzpicture with the points, the ring and the legend). Chapter 3 wraps the .tex
in its figure float and caption.

    py -3.12 -m uol.thesis_figures.f3_4_catalogue_map
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from uol.thesis_figures import common as C
from uol.thesis_figures.f09_map import GRAPH, UMAP
from uol.thesis_figures.f2_4_map import W_CM

OUT_DIR = C.ROOT / "thesis" / "figures"
RELEASE_DIR = C.ROOT / "data" / "releases" / "uol_c01_statutes_20261004_v11_stratified"
SPLIT = C.ROOT / "research" / "state" / "embedding_push_20261004" / "SPLIT.json"
CORRECTIONS = C.ROOT / "release" / "datapack" / "uol-benchmark-v11" / "benchmark" / "label_corrections.json"
CITED_DEV = 2407                 # \nCitedArticlesDev in thesis/evidence/numbers_ch4.tex
MAP_CM = 10.0                    # side of the points square (CF-01: the map takes the room the legend had)
X_OFF = 0.6                      # left edge of the square (cm); the names at the right may pass it
PDF = "https://www.diputados.gob.mx/LeyesBiblio/pdf/"
# the six laws with the most cited articles, most cited first: (file of the law, name, where the name goes: a point in
# map cm and the side the name sits on); the grey line runs from the densest part of the law's cited articles
LAWS = [
    ("LFT", "Ley Federal del Trabajo", (3.25, 7.9), "east"),
    ("CNPP", "C\\'odigo Nacional de\\\\[-1pt]Procedimientos Penales", (8.45, 4.55), "west"),
    ("LAmp", "Ley de Amparo", (8.45, 5.45), "west"),
    ("CCom", "C\\'odigo de Comercio", (8.45, 7.0), "west"),
    ("CFPC", "C\\'odigo Federal de\\\\[-1pt]Procedimientos Civiles", (8.45, 6.3), "west"),
    ("CFF", "C\\'odigo Fiscal de la Federaci\\'on", (4.75, 3.75), "east"),
]
COLOURS = C.SCHEME + [C.tint(C.COMPANION_BLUE, 0.5), C.tint(C.COMPANION_OCHRE, 0.55)]
EXAMPLE_LABEL = (3.95, 5.2)      # the name of the ring, north east anchor
# CF-09 (the author chose option A, 26 Oct 2026: "Colour the wierd island on the top if its mostly 1 law even if not cited"):
# the island at the top is mostly the Ordenanza General de la Armada (L3's check: 79 % of its 2,098 articles, 1 cited);
# its never-cited articles take light plum, the one P3 step the six cited laws do not use, and it is named beside them.
GROUP = {"files": ["218"], "colour": C.tint(C.COMPANION_PLUM, 0.5),
         "name": "Ordenanza General\\\\[-1pt]de la Armada", "label": (5.4, 9.65), "side": "east"}
EXAMPLE = PDF + "LGSM.pdf#articulo=2"   # cited in the running example, tesis 2028281
REST, REST_X = "#dcdcdc", "#4d4d4d"   # never-cited dots (every law); crosses of the other laws


def _articles() -> list[dict]:
    manifest = C.load_json(C.RELEASE)
    path = manifest["candidates"]["articles"]["path"]
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _cited_dev(articles: list[dict]) -> np.ndarray:
    assignment = C.load_json(SPLIT)["assignment"]
    corrections = C.load_json(CORRECTIONS)
    index = {a["candidate_id"]: i for i, a in enumerate(articles)}
    cited = np.zeros(len(articles), dtype=bool)
    with open(RELEASE_DIR / "references.jsonl", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if assignment.get(row["query_id"], "lockbox") == "lockbox":
                continue
            fix = corrections.get(row["query_id"], {"remove": [], "add": []})
            ids = [p["candidate_id"] for p in row["positives"] if p["candidate_id"] not in fix["remove"]] + fix["add"]
            for cid in ids:
                cited[index[cid]] = True
    return cited


def build(out_dir=OUT_DIR, group: dict | None = GROUP) -> dict:
    """group (CF-09): the never-cited dots of some laws in one tint, named beside them, as
    {"files": [file of each law], "colour": hex, "name": TeX text, "label": (x, y) in map cm, "side": anchor}."""
    C.style()
    U = np.load(C.ROOT / UMAP)
    inst = np.load(C.ROOT / GRAPH)["inst"].astype(int)
    articles = _articles()
    assert len(inst) == len(U) == len(articles) == 39178, (len(inst), len(U), len(articles))
    # The UMAP rows follow the release's article order: each graph law index maps to exactly one instrument.
    law_of = {}
    for li, art in zip(inst, articles):
        assert law_of.setdefault(int(li), art["instrument_id"]) == art["instrument_id"]
    cited = _cited_dev(articles)
    assert int(cited.sum()) == CITED_DEV, int(cited.sum())
    ins = np.array([a["instrument_id"] for a in articles])
    six = {PDF + f + ".pdf" for f, *_ in LAWS}
    member = [ins == PDF + f + ".pdf" for f, *_ in LAWS]
    counts = [int((m & cited).sum()) for m in member]
    assert all(m.any() for m in member) and counts == sorted(counts, reverse=True), counts
    laws, n = np.unique(ins[cited], return_counts=True)
    assert max(k for i, k in zip(laws, n) if i not in six) < counts[-1]      # the six are the most cited laws
    ex = [i for i, a in enumerate(articles) if a["candidate_id"] == EXAMPLE]
    assert len(ex) == 1 and cited[ex[0]], ex
    rest = ~np.logical_or.reduce(member)

    lo, hi = np.percentile(U, 2.0, axis=0), np.percentile(U, 98.5, axis=0)
    span = float(max(hi - lo)) * 1.06
    cx, cy = (lo + hi) / 2
    x0, y0 = cx - span / 2, cy - span / 2

    fig = plt.figure(figsize=(MAP_CM / 2.54, MAP_CM / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    # R review 4.2: the citation status must read at print size, so never-cited articles are faint grey dots for
    # every law and cited articles larger, darker crosses, coloured by law for the six laws.
    dot = dict(s=0.7, linewidths=0, rasterized=True)
    cross = dict(s=5.0, marker="x", linewidths=0.5, rasterized=True)
    ax.scatter(U[~cited, 0], U[~cited, 1], c=REST, **dot)
    grouped = np.isin(ins, [PDF + f + ".pdf" for f in group["files"]]) if group else np.zeros(len(ins), dtype=bool)
    assert not (grouped & ~rest).any()                          # a group never takes in one of the six laws
    if group:
        ax.scatter(U[grouped & ~cited, 0], U[grouped & ~cited, 1], c=group["colour"], **dot)
    ax.scatter(U[rest & cited, 0], U[rest & cited, 1], c=REST_X, **cross)
    for m, col in reversed(list(zip(member, COLOURS))):      # above the grey; the lighter steps first, green last
        ax.scatter(U[m & cited, 0], U[m & cited, 1], color=col, **cross)
    ax.set_xlim(x0, x0 + span)
    ax.set_ylim(y0, y0 + span)
    ax.axis("off")
    pts = out_dir / "F3_4_catalogue_map_points.pdf"
    fig.savefig(pts, dpi=600, metadata={"CreationDate": None})
    plt.close(fig)

    def at(x: float, y: float) -> str:          # map cm to figure cm
        return f"({x + X_OFF:.3f},{y:.3f})"

    def edge(m: np.ndarray, label: tuple[float, float]) -> tuple[float, float]:
        """Where a name's line starts: the point of the cluster's core (within 0.45 cm of its densest part) nearest
        the name, set back 0.06 cm towards it, so every line starts at the cluster's edge facing its name."""
        P = (U[m] - np.array([x0, y0])) / span * MAP_CM
        H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=40, range=[[0, MAP_CM], [0, MAP_CM]])
        i, j = np.unravel_index(H.argmax(), H.shape)
        core = P[np.hypot(P[:, 0] - xe[i] - MAP_CM / 80, P[:, 1] - ye[j] - MAP_CM / 80) < 0.45]
        for q in starts:                        # two lines never start within 0.3 cm of each other
            core = core[np.hypot(core[:, 0] - q[0], core[:, 1] - q[1]) >= 0.3]
        p = core[np.argmin(np.hypot(core[:, 0] - label[0], core[:, 1] - label[1]))]
        v = np.array(label) - p
        starts.append(p)
        return tuple(p + 0.06 * v / np.linalg.norm(v))

    starts: list[np.ndarray] = []

    def leader(start: tuple[float, float], label: tuple[float, float]) -> str:
        return f"  \\draw[f34line, line width=0.4pt] {at(*start)} -- {at(*label)};"

    def square(colour: str) -> str:             # the colour of a law, beside its name
        return f"\\textcolor{{{colour}}}{{\\rule{{5pt}}{{5pt}}}}"

    def disc(colour: str) -> str:               # a dot, because the articles of this group are never cited
        return f"\\tikz[baseline=-0.2ex]\\fill[{colour}] (0,0) circle (2.5pt);"

    def with_mark(name: str, mark: str) -> str:  # the mark before the first line, the second line set under the name
        first, *more = name.split("\\\\[-1pt]")
        indent = f"\\phantom{{{square('black')}\\,}}"
        return f"{mark}\\,{first}" + "".join(f"\\\\[-1pt]{indent}{m}" for m in more)

    ex_x = (float(U[ex[0], 0]) - x0) / span * MAP_CM
    ex_y = (float(U[ex[0], 1]) - y0) / span * MAP_CM
    lines = [
        "% Catalogue map for 3.4 (reframe M4; redrawn by sprint session C, 26 Oct 2026; by LF for CF-01, 26 Oct 2026, P3).",
        "% Generated by uol.thesis_figures.f3_4_catalogue_map; do not edit by hand.",
        f"% Points: F3_4_catalogue_map_points.pdf (saved E1 UMAP {UMAP}; nothing recomputed). Crosses: the {CITED_DEV}",
        "% articles cited by at least one development case (v11 references.jsonl with the three label corrections, test set",
        "% excluded); dots: never cited. The six most cited laws, cited articles: "
        + ", ".join(f"{f} {k}" for (f, *_), k in zip(LAWS, counts)) + f"; the other laws grey ({int((rest & cited).sum())} cited).",
        "% Ring: the article cited in the running example (LGSM art. 2, tesis 2028281).",
        "% Tinted never-cited dots (CF-09): " + (", ".join(group["files"]) if group else "none") + ".",
        "\\begin{tikzpicture}[x=1cm, y=1cm, every node/.append style={inner sep=0pt, outer sep=0pt}]",
        f"  \\definecolor{{f34ink}}{{HTML}}{{1A1A1A}}\\definecolor{{f34line}}{{HTML}}{{8C8C8C}}"
        f"\\definecolor{{f34rest}}{{HTML}}{{{REST[1:]}}}\\definecolor{{f34restx}}{{HTML}}{{{REST_X[1:]}}}%",
        "  " + "".join(f"\\definecolor{{f34law{i}}}{{HTML}}{{{c[1:]}}}" for i, c in enumerate(COLOURS))
        + (f"\\definecolor{{f34group}}{{HTML}}{{{group['colour'][1:]}}}%" if group else "%"),
        f"  \\node[anchor=south west] at {at(0, 0)} {{\\includegraphics[width={MAP_CM}cm]{{F3_4_catalogue_map_points.pdf}}}};",
        f"  \\draw[black, line width=0.6pt] {at(ex_x, ex_y)} circle (0.16);",
    ]
    # Each law named next to its cluster, on a light halo over the faint dots, joined to the cluster's edge by a grey
    # line (CF-09: "make it consistent with the rest, improve the line placement for all clusters").
    name_style = "fill=white, fill opacity=0.85, text opacity=1, inner sep=1.2pt, align=left, font=\\scriptsize, text=f34ink"
    for i, ((_, name, label, side), m) in enumerate(zip(LAWS, member)):
        lines += [leader(edge(m & cited, label), label),
                  f"  \\node[anchor={side}, {name_style}] at {at(*label)} {{{with_mark(name, square(f'f34law{i}'))}}};"]
    if group:                                   # the tinted never-cited dots, named the same way
        lines += [leader(edge(grouped & ~cited, group["label"]), group["label"]),
                  f"  \\node[anchor={group['side']}, {name_style}] at {at(*group['label'])} "
                  f"{{{with_mark(group['name'], disc('f34group'))}}};"]
    v = np.array(EXAMPLE_LABEL) - (ex_x, ex_y)       # from the ring's edge
    lines += [leader(tuple(np.array((ex_x, ex_y)) + 0.16 * v / np.linalg.norm(v)), EXAMPLE_LABEL),
              f"  \\node[anchor=north east, {name_style}] at {at(*EXAMPLE_LABEL)} {{the running example}};"]
    # The key, in the empty lower-right corner (H5, 2 Nov 2026: "display what the colours are"): what each mark means.
    kx, ky, d, gap = 8.0, 1.2, 0.07, 0.38
    tx = kx + 1.2

    def cross(x: float, y: float, colour: str) -> str:
        return (f"  \\draw[{colour}, line width=0.7pt] {at(x - d, y - d)} -- {at(x + d, y + d)} "
                f"{at(x - d, y + d)} -- {at(x + d, y - d)};")

    def key(y: float, text: str) -> str:
        return f"  \\node[anchor=west, font=\\scriptsize, text=f34ink] at {at(tx, y)} {{{text}}};"

    lines += [cross(kx + 0.2 * i, ky, f"f34law{i}") for i in range(len(LAWS))]
    lines += [key(ky, "cited, law named"),
              cross(kx + 1.0, ky - gap, "f34restx"), key(ky - gap, "cited, other law"),
              f"  \\fill[f34rest!80!black] {at(kx + 1.0, ky - 2 * gap)} circle (1.3pt);",
              key(ky - 2 * gap, "never cited")]
    lines += [
        f"  \\path (0,0) rectangle ({W_CM},{MAP_CM});",
        "\\end{tikzpicture}",
        "",
    ]
    (out_dir / "F3_4_catalogue_map.tex").write_text("\n".join(lines), encoding="utf-8")
    return {"points": str(pts), "tex": str(out_dir / "F3_4_catalogue_map.tex"), "cited": int(cited.sum()),
            "six": dict(zip([f for f, *_ in LAWS], counts)), "example_cm": (round(ex_x, 2), round(ex_y, 2))}


if __name__ == "__main__":
    print(build())
