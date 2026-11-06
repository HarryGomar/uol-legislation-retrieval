"""F9: the article embedding map coloured by law (for 6.4 Structure inside the embedding space, or 5.3).

Source: the saved 2-D UMAP of all 39,178 Qwen3-Embedding-8B article vectors from E1 (arms/E1_structure/
umap_articles.npy; PCA to 50 components, cosine, 30 neighbours, seed fixed in E1), the statute graph's article-to-law
index (phase2/statute_graph/graph.npz, key 'inst', same article order as E1 used) and the law titles in the catalogue
(first line of instrument_article_body, as in E1's law_names). No embedding is recomputed. Label-free, descriptive.
"""
from __future__ import annotations

import json
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from uol.thesis_figures import common as C

UMAP = f"{C.R3_ARMS}/E1_structure/umap_articles.npy"
GRAPH = f"{C.R3}/phase2/statute_graph/graph.npz"
SOURCES = [UMAP, GRAPH, f"{C.R3_ARMS}/E1_structure/RESULT.json", C.RELEASE]
TOP = 3


def law_titles(inst: np.ndarray) -> dict[int, str]:
    path = C.load_json(C.RELEASE)["candidates"]["articles"]["path"]
    out: dict[int, str] = {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            li = int(inst[i])
            if li in out:
                continue
            row = json.loads(line)
            out[li] = str(row.get("instrument_article_body") or row.get("instrument_id") or li).split("\n", 1)[0].strip()
    return out


def short(title: str, n: int = 34) -> str:
    """Spanish title case for law names ('LEY Federal del trabajo' -> 'Ley Federal del Trabajo')."""
    small = {"de", "del", "la", "las", "los", "el", "y", "e", "en", "para", "por", "a", "o", "u"}
    words = title.strip().lower().split()
    t = " ".join(w if (i and w in small) else w[:1].upper() + w[1:] for i, w in enumerate(words))
    return t if len(t) <= n else t[: n - 1].rstrip() + "…"


def build() -> C.Item:
    C.style()
    U = np.load(C.ROOT / UMAP)
    inst = np.load(C.ROOT / GRAPH)["inst"].astype(int)
    assert len(inst) == len(U), (len(inst), len(U))
    titles = law_titles(inst)
    sizes = Counter(inst.tolist())
    top = [li for li, _ in sizes.most_common(TOP)]
    cols = [C.BLUE, C.TEAL, C.GREYS[0]]
    fig, ax = plt.subplots(figsize=(C.TEXTWIDTH_IN, C.TEXTWIDTH_IN * 0.66))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.12)
    rest = ~np.isin(inst, top)
    ax.scatter(U[rest, 0], U[rest, 1], s=1.0, c="#cfcfcf", linewidths=0, rasterized=True)
    handles = []
    for j, li in enumerate(top):
        m = inst == li
        sc = ax.scatter(U[m, 0], U[m, 1], s=2.0, c=cols[j], linewidths=0, rasterized=True,
                        label=f"{j + 1}  {short(titles[li], 60)} ({m.sum():,} articles)")
        handles.append(sc)
        hist, xe, ye = np.histogram2d(U[m, 0], U[m, 1], bins=40)      # label the densest region, not the median
        bx, by = np.unravel_index(np.argmax(hist), hist.shape)
        cx, cy = (xe[bx] + xe[bx + 1]) / 2, (ye[by] + ye[by + 1]) / 2
        ax.text(cx, cy, str(j + 1), fontsize=8, weight="bold", color=C.INK, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc=C.SURFACE, ec=cols[j], lw=1.0), zorder=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    lo, hi = np.percentile(U, 1.5, axis=0), np.percentile(U, 98.5, axis=0)
    pad = 0.04 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_aspect("equal", adjustable="box")
    handles.append(ax.scatter([], [], s=0.9, c="#cfcfcf", label="every other law"))
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8,
                    markerscale=5, handletextpad=0.3, columnspacing=1.5, borderaxespad=0.3)
    for h in leg.legend_handles:
        h.set_sizes([16])
    res = C.load_json(f"{C.R3_ARMS}/E1_structure/RESULT.json")
    item = C.Item(
        id="F9", stem="F09_map", target="5.3 Prototype (D52 item 1c); the numeric structure result stays in 6.4",
        title="The federal articles in the space of Qwen3-Embedding-8B",
        caption=(f"All {len(U):,} federal articles embedded by Qwen3-Embedding-8B and projected to two dimensions "
                 "(UMAP), with the three largest laws coloured. Articles of one law gather in a few regions, and "
                 "large laws split by subject."),
        label="descriptive (label-free)", sources=SOURCES, internal="E1_structure umap_articles; statute graph G2 'inst'",
        notes=[f"top-{TOP} laws: " + "; ".join(f"{titles[li]} ({sizes[li]})" for li in top),
               f"E1 RESULT.json keys present: {len(res)}",
               "D54: three laws instead of five (vermilion is reserved for the LLM stage), 8 pt legend."])
    return C.save(fig, item)


if __name__ == "__main__":
    print(build().outputs)
