"""C9 (c): length propensity for every DEV system (ANALYSIS_PLAN_C §9 (c); descriptive).

Article length = words of ``article_body`` (whitespace tokens); deciles over the whole catalogue (edges at the 10th,
..., 90th percentiles, lower edge inclusive, so tied lengths can make deciles uneven; counts are reported). Per
system and decile: the decile's share of all filled top-10 slots over the DEV cases the system scored, divided by
the decile's share of the cited (case, specific article) pairs of those cases. Intervals by the cluster bootstrap
over groups (slots and pairs resampled with their case's group).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C


def deciles(lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.quantile(lengths, np.linspace(0.1, 0.9, 9))
    return np.searchsorted(edges, lengths, side="right"), edges


def propensity(top_dec: np.ndarray, top_group: np.ndarray, cit_dec: np.ndarray, cit_group: np.ndarray,
               boot: C.Boot) -> list[dict[str, Any]]:
    G = boot.G
    T = np.zeros((G, 10))
    P = np.zeros((G, 10))
    np.add.at(T, (top_group, top_dec), 1)
    np.add.at(P, (cit_group, cit_dec), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        point = (T.sum(0) / T.sum()) / (P.sum(0) / P.sum())
        bt, bp = boot.W @ T, boot.W @ P
        b = (bt / bt.sum(1, keepdims=True)) / (bp / bp.sum(1, keepdims=True))
    rows = []
    for d in range(10):
        lo, hi = C.interval(b[:, d])
        rows.append({"decile": d + 1, "top10_slots": int(T[:, d].sum()), "cited_pairs": int(P[:, d].sum()),
                     "propensity": float(point[d]), "lo": lo, "hi": hi})
    return rows


def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", systems: Sequence[str] | None = None,
        draws: int = C.DRAWS, seed: int = C.SEED) -> dict[str, Any]:
    out = Path(out)
    store = C.RunStore(runs, population)
    cat = C.load_catalogue(str(data))
    boot = C.Boot(store.labels.component, draws, seed)
    dec, edges = deciles(cat.body_words)
    case, art = store.labels.pair_arrays()
    names = list(systems) if systems else store.systems
    rows = []
    for s in names:
        idx, _ = store.top(s, 10)
        present = store.cases(s)["present"].to_numpy()
        c = np.repeat(np.arange(idx.shape[0]), 10)
        a = idx.ravel()
        ok = (a >= 0) & present[c]
        keep = present[case]
        for r in propensity(dec[a[ok]], boot.inv[c[ok]], dec[art[keep]], boot.inv[case[keep]], boot):
            rows.append({"system": s, "family": C.family(s), **r})
    table = pd.DataFrame(rows)
    C.write_csv(out / "c9_length_propensity.csv", table)
    C.write_csv(out / "c9_deciles.csv", [{"decile": d + 1, "articles": int((dec == d).sum()),
                                          "upper_edge_words": float(edges[d]) if d < 9 else float(cat.body_words.max())}
                                         for d in range(10)])
    notes = [f"Deciles of article_body words over {cat.n} catalogue articles; edges {np.round(edges, 1).tolist()}.",
             f"{len(names)} systems; reference labels {store.labels.reference}."]
    ev = C.Evidence("C9c")
    from uol.analysis_c.c3_gain import SYSTEMS as C3_SYSTEMS
    focus = [s for s in dict.fromkeys(list(C3_SYSTEMS) + list(C.answer_systems().values()) +
                                      ["bm25_stem", "lda_sim", "doc2vec", "e5_large_instruct", "qwen3_8b_vectors"]) if s in names]
    for s in focus:
        sub = table[table["system"] == s]
        ev.add(f"{s} on {population}, length propensity", [f"decile {d}" for d in (1, 5, 10)],
               [C.fmt_ci({"mean": r.propensity, "lo": r.lo, "hi": r.hi}, 2) for r in sub.itertuples() if r.decile in (1, 5, 10)],
               store.source(s, "top200.parquet"))
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C9c", "plan": "ANALYSIS_PLAN_C §9 (c)", "population": population, "systems": len(names),
           "decile_edges": edges.tolist(), "notes": notes}
    C.write_json(out / "c9.json", rec)
    return rec
