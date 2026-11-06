"""C4 (a) and (b): near misses and label incompleteness (ANALYSIS_PLAN_C §5; DEV, descriptive). The human sample (c)
is not built here.

(a) Statute relation of every uncited top-10 article to the closest cited article of its case, from the statute graph
(edges symmetrised), first match in the plan's order: ±1 article (e_adj1), ±2 (e_adj2), same section (e_sibling),
same chapter (e_chapter), cross-reference (e_xref_int or e_xref_ext, either direction), same law, other law.
Baselines, per case, with one generator seeded 20261023 in case order: ten uncited non-generic articles drawn
uniformly (with replacement) from the cited laws, and ten from the whole catalogue. Shares are pooled over slots with
cluster-bootstrap intervals.

(b) Lenient nDCG@10: gain 1 for a cited article, .5 for an uncited ±1 neighbour (e_adj1) of a cited article, .25 for
an uncited article in a cited article's section (e_sibling), else 0; the ideal list is the strict one (cited
articles only), so the value can exceed 1. Every DEV system is re-scored from its saved top 200; Kendall's tau-b
compares the strict and the lenient order of systems (component-macro means). Contrast families F1-F8 are rebuilt
from ``uol.final.stats.FAMILIES`` and its family rules over the systems present (mirroring ``stats.family_pairs`` on a
runs folder) and classified with the frozen procedure (``round3.stats.contrast``, 10,000 draws, seed 20261004,
Bonferroni-adjusted bound at the family size, Holm), once on strict and once on lenient per-group values;
contrasts whose class changes are listed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

RELATIONS = ("cited", "adj1", "adj2", "same_section", "same_chapter", "cross_reference", "same_law", "other_law")
EDGE_ORDER = (("adj1", ("e_adj1",)), ("adj2", ("e_adj2",)), ("same_section", ("e_sibling",)),
              ("same_chapter", ("e_chapter",)), ("cross_reference", ("e_xref_int", "e_xref_ext")))
LENIENT = {0: 1.0, 1: 0.5, 3: 0.25}           # relation code -> gain (cited, adj1, same section)
N_BASE = 10


class Graph:
    """Symmetric edge-key sets per relation level (keys min*N + max)."""

    def __init__(self, cat: C.Catalogue):
        self.n = cat.n
        self.law = cat.law
        self.keys = []
        for _, names in EDGE_ORDER:
            parts = [cat.edges[k] for k in names if k in cat.edges and len(cat.edges[k])]
            e = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)
            k = np.minimum(e[:, 0], e[:, 1]) * self.n + np.maximum(e[:, 0], e[:, 1])
            self.keys.append(np.unique(k))

    def relation(self, slot_case: np.ndarray, slot_art: np.ndarray, positives: Sequence[np.ndarray]) -> np.ndarray:
        """Relation code (index into RELATIONS) of each slot article to the closest cited article of its case."""
        out = np.full(slot_art.size, 7, dtype=np.int64)
        if not slot_art.size:
            return out
        npos = np.asarray([len(positives[c]) for c in slot_case])
        rep = np.repeat(np.arange(slot_art.size), npos)
        cited = np.concatenate([positives[c] for c in slot_case]) if npos.sum() else np.zeros(0, dtype=np.int64)
        a = slot_art[rep]
        level = np.full(rep.size, 7, dtype=np.int64)
        level[self.law[a] == self.law[cited]] = 6
        key = np.minimum(a, cited) * self.n + np.maximum(a, cited)
        for code in range(len(self.keys), 0, -1):          # most specific last, so it wins
            ks = self.keys[code - 1]
            if ks.size:
                pos = np.searchsorted(ks, key)
                hit = (pos < ks.size) & (ks[np.minimum(pos, ks.size - 1)] == key)
                level[hit] = code
        level[a == cited] = 0
        np.minimum.at(out, rep, level)
        return out


def top10_relations(store: C.RunStore, cat: C.Catalogue, graph: Graph, system: str) -> tuple[np.ndarray, np.ndarray]:
    """[n, 10] relation codes of the system's top 10 (-1 for empty slots), and the case-present mask."""
    idx, _ = store.top(system, 10)
    present = store.cases(system)["present"].to_numpy()
    case = np.repeat(np.arange(idx.shape[0]), 10)
    art = idx.ravel()
    ok = (art >= 0) & present[case]
    rel = np.full(art.size, -1, dtype=np.int64)
    rel[ok] = graph.relation(case[ok], art[ok], store.labels.positives)
    return rel.reshape(idx.shape), present


def lenient_ndcg(rel: np.ndarray, n_rel: np.ndarray) -> np.ndarray:
    gain = np.zeros(rel.shape)
    for code, g in LENIENT.items():
        gain[rel == code] = g
    return C.ndcg10(gain, n_rel)


def baseline_relations(store: C.RunStore, cat: C.Catalogue, graph: Graph, seed: int = C.SEED) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    generic = np.zeros(cat.n, dtype=bool)
    generic[cat.generic] = True
    by_law = {int(l): np.flatnonzero((cat.law == l) & ~generic) for l in np.unique(cat.law)}
    allowed = np.flatnonzero(~generic)
    cl, ca, rl, ra = [], [], [], []
    for i, pos in enumerate(store.labels.positives):
        if not len(pos):
            continue
        pool = np.setdiff1d(np.concatenate([by_law[int(l)] for l in np.unique(cat.law[pos])]), pos)
        if pool.size:
            cl.append(np.full(N_BASE, i))
            ca.append(rng.choice(pool, N_BASE, replace=True))
        draw = rng.choice(allowed, N_BASE * 2, replace=True)
        draw = draw[~np.isin(draw, pos)][:N_BASE]
        rl.append(np.full(draw.size, i))
        ra.append(draw)
    out = {}
    for name, cs, arts in (("random_article_of_cited_law", cl, ca), ("random_article", rl, ra)):
        c = np.concatenate(cs).astype(np.int64)
        a = np.concatenate(arts).astype(np.int64)
        out[name] = (c, graph.relation(c, a, store.labels.positives))
    return out


def relation_shares(case: np.ndarray, rel: np.ndarray, boot: C.Boot) -> dict[str, dict[str, float]]:
    """Shares of each relation among uncited slots (rel >= 1), pooled, with intervals."""
    m = rel >= 1
    g = boot.inv[case[m]]
    out = {}
    for code in range(1, len(RELATIONS)):
        out[RELATIONS[code]] = boot.ratio((rel[m] == code).astype(float), np.ones(m.sum()), g)
    near = (rel[m] >= 1) & (rel[m] <= 5)
    out["near (adj1..cross_reference)"] = boot.ratio(near.astype(float), np.ones(m.sum()), g)
    return out


# ----------------------------------------------------------------------------- families

def family_pairs(store: C.RunStore, allowed: Sequence[str] | None = None) -> dict[str, list[tuple[str, str]]]:
    """F1-F8 among the systems of this runs folder (or ``allowed``), by the rules of ``uol.final.stats.family_pairs``."""
    from uol.final import registry, stats
    pool = set(allowed) if allowed is not None else set(store.systems)
    have = lambda s: s in pool
    fam: dict[str, list[tuple[str, str]]] = {}
    for f in ("F1", "F3", "F6", "F7", "F8"):
        fam[f] = [p for p in stats.FAMILIES[f] if have(p[0]) and have(p[1])]
    fam["F2"] = [(x, "S0") for x in stats.ladder_dense_only() if have(x) and have("S0")]
    f4 = []
    means = {}
    for x in registry.FUSION_PARTNERS:
        for m in ("rrf", "linear"):
            h = f"{m}__lstar_headings__{x}"
            if have(h) and have("lstar_headings") and have(x):
                for s in ("lstar_headings", x):
                    if s not in means:
                        means[s] = _macro(store.cases(s)["ndcg_at_10"].to_numpy(float), store.labels.component)
                best = max(("lstar_headings", x), key=lambda s: means[s])
                f4.append((h, best))
    fam["F4"] = f4
    f5 = []
    for v, base in (("noemb", "ltr_noemb_refit"), ("emb", "ltr_emb_refit")):
        for other in (f"ltr_{v}_linear", f"ltr_{v}_lambdamart", f"rrf_all_{v}"):
            if have(other) and have(base):
                f5.append((other, base))
    fam["F5"] = f5
    return {k: fam[k] for k in sorted(fam)}


def _macro(v: np.ndarray, comps: np.ndarray) -> float:
    ok = np.isfinite(v)
    ids, inv = np.unique(comps[ok], return_inverse=True)
    if not ids.size:
        return float("nan")
    return float((np.bincount(inv, weights=v[ok]) / np.bincount(inv)).mean())


def _aligned(va: np.ndarray, vb: np.ndarray, comps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = np.isfinite(va) & np.isfinite(vb)
    ids, inv = np.unique(comps[ok], return_inverse=True)
    cnt = np.bincount(inv)
    return np.bincount(inv, weights=va[ok]) / cnt, np.bincount(inv, weights=vb[ok]) / cnt


def classify_families(fams: dict[str, list[tuple[str, str]]], strict: dict[str, np.ndarray],
                      lenient: dict[str, np.ndarray], comps: np.ndarray) -> list[dict[str, Any]]:
    from uol.round3 import stats as R3
    rows = []
    for name, pairs in fams.items():
        if not pairs:
            continue
        m = len(pairs)
        res = {}
        for lens, vals in (("strict", strict), ("lenient", lenient)):
            cons = {}
            for a, b in pairs:
                va, vb = _aligned(vals[a], vals[b], comps)
                cons[f"{a} - {b}"] = R3.contrast(va, vb, family_size=m)
            res[lens] = R3.family(cons)
        for key in res["strict"]:
            s, l = res["strict"][key], res["lenient"][key]
            rows.append({"family": name, "contrast": key, "family_size": m, "strict_delta": s["delta"],
                         "strict_ci95_lo": s["ci95"][0], "strict_ci95_hi": s["ci95"][1], "strict_class": s["class"],
                         "lenient_delta": l["delta"], "lenient_ci95_lo": l["ci95"][0], "lenient_ci95_hi": l["ci95"][1],
                         "lenient_class": l["class"], "class_changed": s["class"] != l["class"],
                         "strict_p_holm": s["p_holm"], "lenient_p_holm": l["p_holm"]})
    return rows


# ----------------------------------------------------------------------------- run

def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", systems: Sequence[str] | None = None,
        draws: int = C.DRAWS, seed: int = C.SEED, families: bool = True) -> dict[str, Any]:
    from scipy.stats import kendalltau
    out = Path(out)
    store = C.RunStore(runs, population)
    cat = C.load_catalogue(str(data))
    graph = Graph(cat)
    boot = C.Boot(store.labels.component, draws, seed)
    names = list(systems) if systems else store.systems
    nrel = np.asarray([len(p) for p in store.labels.positives])
    share_rows, sys_rows = [], []
    strict, lenient = {}, {}
    for s in names:
        rel, present = top10_relations(store, cat, graph, s)
        st = C.ndcg10((rel == 0).astype(float), nrel)
        le = lenient_ndcg(rel, nrel)
        st[~present], le[~present] = np.nan, np.nan
        strict[s], lenient[s] = st, le
        case = np.repeat(np.arange(rel.shape[0]), rel.shape[1])
        shares = relation_shares(case, rel.ravel(), boot)
        for r, d in shares.items():
            share_rows.append({"system": s, "family": C.family(s), "relation": r, "share": d["mean"], "lo": d["lo"],
                               "hi": d["hi"], "uncited_slots": int(d["n"])})
        ds, dl = boot.macro(st), boot.macro(le)
        stored = store.cases(s)["ndcg_at_10"].to_numpy(float)
        sys_rows.append({"system": s, "family": C.family(s), "strict_ndcg_at_10": ds["mean"], "strict_lo": ds["lo"],
                         "strict_hi": ds["hi"], "lenient_ndcg_at_10": dl["mean"], "lenient_lo": dl["lo"],
                         "lenient_hi": dl["hi"], "lenient_minus_strict": dl["mean"] - ds["mean"],
                         "stored_vs_recomputed_max_abs": float(np.nanmax(np.abs(stored - st))) if present.any() else np.nan})
    base = baseline_relations(store, cat, graph, seed)
    for name, (case, rel) in base.items():
        for r, d in relation_shares(case, rel, boot).items():
            share_rows.append({"system": f"baseline:{name}", "family": "baseline", "relation": r, "share": d["mean"],
                               "lo": d["lo"], "hi": d["hi"], "uncited_slots": int(d["n"])})
    table = pd.DataFrame(sys_rows)
    table["strict_rank"] = table["strict_ndcg_at_10"].rank(ascending=False, method="min")
    table["lenient_rank"] = table["lenient_ndcg_at_10"].rank(ascending=False, method="min")
    tau = kendalltau(table["strict_ndcg_at_10"], table["lenient_ndcg_at_10"]) if len(table) > 1 else None
    fam_rows = classify_families(family_pairs(store, names), strict, lenient, store.labels.component) if families else []
    C.write_csv(out / "c4_relations.csv", share_rows)
    C.write_csv(out / "c4_lenient_systems.csv", table)
    C.write_csv(out / "c4_family_classes.csv", fam_rows)
    changed = [r for r in fam_rows if r["class_changed"]]
    notes = [f"Reference labels {store.labels.reference}; {store.labels.n} cases, {boot.G} groups; {len(names)} systems.",
             "Lenient nDCG@10 keeps the strict ideal list, so it can exceed 1.",
             "Families present (contrasts): " + (", ".join(f"{k} ({n})" for k, n in
                                                           pd.Series([r["family"] for r in fam_rows]).value_counts()
                                                           .sort_index().items()) or "none") + ".",
             "Not built here: the human sample of §5 (c)."]
    ev = C.Evidence("C4")
    if tau is not None:
        ev.add(f"strict vs lenient order of {len(table)} {population} systems",
               ["Kendall tau-b", "p", "contrasts whose class changes"],
               [f"{tau.statistic:.3f}", f"{tau.pvalue:.2g}", f"{len(changed)} of {len(fam_rows)}"], "c4_lenient_systems.csv")
    for r in changed:
        ev.add(f"{r['family']} {r['contrast']} on {population}", ["strict class", "lenient class", "strict delta", "lenient delta"],
               [r["strict_class"], r["lenient_class"], f"{r['strict_delta']:+.3f}", f"{r['lenient_delta']:+.3f}"],
               "c4_family_classes.csv")
    sh = pd.DataFrame(share_rows)
    from uol.analysis_c.c3_gain import SYSTEMS as C3_SYSTEMS
    focus = [s for s in list(C3_SYSTEMS) + list(C.answer_systems().values()) if s in names]
    for s in dict.fromkeys(focus + [f"baseline:{b}" for b in base]):
        sub = sh[sh["system"] == s].set_index("relation")
        rels = ["near (adj1..cross_reference)", "adj1", "same_section", "same_law", "other_law"]
        ev.add(f"{s} on {population}, uncited top-10 articles", [f"share {r}" for r in rels],
               [C.fmt_ci({"mean": sub.at[r, "share"], "lo": sub.at[r, "lo"], "hi": sub.at[r, "hi"]}) for r in rels],
               store.source(s, "top200.parquet") if not s.startswith("baseline:") else "catalogue + statute graph")
        if not s.startswith("baseline:"):
            t = table.set_index("system").loc[s]
            ev.add(f"{s} on {population}", ["strict nDCG@10", "lenient nDCG@10"],
                   [C.fmt_ci({"mean": t.strict_ndcg_at_10, "lo": t.strict_lo, "hi": t.strict_hi}),
                    C.fmt_ci({"mean": t.lenient_ndcg_at_10, "lo": t.lenient_lo, "hi": t.lenient_hi})],
                   store.source(s, "top200.parquet"))
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C4", "plan": "ANALYSIS_PLAN_C §5 (a, b)", "population": population, "runs": str(runs),
           "systems": len(names), "kendall_tau_b": tau.statistic if tau is not None else None,
           "kendall_p": tau.pvalue if tau is not None else None, "families": fam_rows, "class_changes": changed,
           "notes": notes, "draws": draws, "seed": seed}
    C.write_json(out / "c4.json", rec)
    return rec
