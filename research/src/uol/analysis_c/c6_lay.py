"""C6: DEV-L, lay wording (ANALYSIS_PLAN_C §6a and its 23 Oct scope note; descriptive).

DEV-L holds the DEV cases whose lay rewrite passed the plan's gates (no law name, article number or listed legal term;
length ratio .5-2); the gates are the plan's and are not changed after the outcome. Every comparison is paired on
those cases only:

* per system, the DEV-L score against the same system's DEV score **restricted to the DEV-L cases** (court wording,
  the same cases, the same fold models), never against all of DEV;
* the contrasts (qwen3_8b_vectors - lstar_headings) and (qwen3_0p6b - lstar_headings): the gap on DEV-L, the gap in
  court wording on the same cases, and their per-case difference.

Selection table: the kept cases against the dropped ones (DEV cases of the rewrite without a DEV-L row), by materia,
facts length (words of the court-worded facts; c3's bins) and citation band (fold-local count of the case's
least-cited cited article; c3's bins), with each group's court-wording score for a few systems. The kept-minus-dropped
differences carry 95 % cluster-bootstrap intervals over the DEV groups. The thesis states that DEV-L covers the cases
a lay rewrite could express without legal terms, and whether they differ from the rest.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

POPULATION = "DEV-L"
CONTRASTS = (("qwen3_8b_vectors", "lstar_headings"), ("qwen3_0p6b", "lstar_headings"))
MEASURES = ("ndcg_at_10", "success_at_10", "recall_at_100")
SELECTION_SYSTEMS = ("lstar_headings", "bm25_stem", "qwen3_8b_vectors", "qwen3_0p6b")
FACTS_BINS = ("<=61", "62-112", ">112")


def facts_bin(words: np.ndarray) -> np.ndarray:
    return np.select([words <= 61, words <= 112], ["<=61", "62-112"], ">112")


def case_band(labels: C.Labels, n_articles: int, cat: "C.Catalogue | None" = None) -> np.ndarray:
    """Band of each case's least-cited cited article (fold-local counts over the DEV cases of the other folds)."""
    counts = C.fold_local_counts(labels, n_articles, cat=cat)
    low = np.asarray([counts[int(f)][p].min() if len(p) else 0 for f, p in zip(labels.fold, labels.positives)])
    return C.band(low)


def paired(lay: C.RunStore, dev: C.RunStore, system: str, measure: str) -> tuple[np.ndarray, np.ndarray]:
    """(DEV-L value, DEV value) per DEV-L case, in DEV-L's label order; NaN where a side lacks the case."""
    v_l = lay.cases(system)[measure].to_numpy(dtype=np.float64)
    f = dev.cases(system).set_index("query_id")
    v_d = f.reindex(lay.labels.query_id)[measure].to_numpy(dtype=np.float64)
    return v_l, v_d


def diff_ci(boot: C.Boot, a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Component-macro mean of the per-case difference a - b over the cases both sides have."""
    d = np.where(np.isfinite(a) & np.isfinite(b), a - b, np.nan)
    return {**boot.macro(d), "cases": int(np.isfinite(d).sum())}


def run(runs: Path, out: Path, *, data: Path = C.DATA, systems: Sequence[str] | None = None,
        contrasts: Sequence[tuple[str, str]] = CONTRASTS, draws: int = C.DRAWS, seed: int = C.SEED) -> dict[str, Any]:
    out = Path(out)
    lay = C.RunStore(runs, POPULATION)
    dev = C.RunStore(runs, "DEV")
    kept = set(lay.labels.query_id.tolist())
    notes = []
    # every DEV-L system must be scored on the same kept cases
    sets = {s: set(pd.read_parquet(Path(runs) / s / POPULATION / "per_case.parquet", columns=["query_id"])["query_id"])
            for s in lay.systems}
    odd = {s: len(v ^ kept) for s, v in sets.items() if v != kept}
    if odd:
        notes.append(f"DEV-L case sets differ from the reference for {odd}; each comparison uses the cases both sides have.")
    outside = kept - set(dev.labels.query_id.tolist())
    if outside:
        raise ValueError(f"{len(outside)} DEV-L cases are not DEV cases")
    boot = C.Boot(lay.labels.component, draws, seed)

    # per system: DEV-L against DEV on the same cases
    rows = []
    names = [s for s in (systems or lay.systems) if lay.has(s)]
    for s in names:
        for m in MEASURES:
            if m not in lay.cases(s).columns:
                continue
            v_l, v_d = paired(lay, dev, s, m) if dev.has(s) else (lay.cases(s)[m].to_numpy(float), None)
            r = {"system": s, "measure": m, "dev_l": boot.macro(v_l)}
            if v_d is not None:
                both = np.isfinite(v_l) & np.isfinite(v_d)
                r.update({"dev_same_cases": boot.macro(np.where(both, v_d, np.nan)),
                          "dev_l_same_cases": boot.macro(np.where(both, v_l, np.nan)),
                          "lay_minus_court": diff_ci(boot, v_l, v_d)})
            rows.append(r)
    sys_table = pd.DataFrame([{"system": r["system"], "measure": r["measure"],
                               **{f"{k}_{x}": r[k][x] for k in ("dev_l_same_cases", "dev_same_cases", "lay_minus_court")
                                  if k in r for x in ("mean", "lo", "hi")},
                               "cases": r.get("lay_minus_court", {}).get("cases")} for r in rows])
    C.write_csv(out / "c6_systems.csv", sys_table)

    # contrasts: the gap on DEV-L, in court wording on the same cases, and their difference
    con = []
    for a, b in contrasts:
        if not all(lay.has(x) and dev.has(x) for x in (a, b)):
            notes.append(f"contrast {a} - {b} not computed: missing {[x for x in (a, b) if not (lay.has(x) and dev.has(x))]}")
            continue
        for m in MEASURES:
            al, ad = paired(lay, dev, a, m)
            bl, bd = paired(lay, dev, b, m)
            ok = np.isfinite(al) & np.isfinite(ad) & np.isfinite(bl) & np.isfinite(bd)
            gl, gd = np.where(ok, al - bl, np.nan), np.where(ok, ad - bd, np.nan)
            con.append({"contrast": f"{a} - {b}", "measure": m, "cases": int(ok.sum()),
                        **{f"gap_lay_{x}": v for x, v in boot.macro(gl).items() if x != "groups"},
                        **{f"gap_court_{x}": v for x, v in boot.macro(gd).items() if x != "groups"},
                        **{f"lay_minus_court_{x}": v for x, v in boot.macro(gl - gd).items() if x != "groups"}})
    C.write_csv(out / "c6_contrasts.csv", con)

    # selection: kept against dropped (DEV cases without a DEV-L row)
    queries = C.load_queries(str(data))
    cat = C.load_catalogue(str(data))
    lab = dev.labels
    is_kept = np.isin(lab.query_id, list(kept))
    dboot = C.Boot(lab.component, draws, seed)
    fw = np.asarray([C.words(queries.get(str(q), {}).get("text", "")) for q in lab.query_id])
    mats = [queries.get(str(q), {}).get("materias", []) for q in lab.query_id]
    bands = case_band(lab, cat.n, cat)
    sel = []

    def add(variable: str, level: str, ind: np.ndarray) -> None:
        ind = ind.astype(np.float64)
        k_mean, k_boot = dboot.macro_boot(np.where(is_kept, ind, np.nan))
        d_mean, d_boot = dboot.macro_boot(np.where(~is_kept, ind, np.nan))
        lo, hi = C.interval(k_boot - d_boot)
        sel.append({"variable": variable, "level": level, "kept_n": int((ind[is_kept] > 0).sum()) if variable != "court score" else None,
                    "dropped_n": int((ind[~is_kept] > 0).sum()) if variable != "court score" else None,
                    "kept": k_mean, "dropped": d_mean, "kept_minus_dropped": k_mean - d_mean, "lo": lo, "hi": hi})
    add("cases", "all", np.ones(lab.n))
    for m in sorted({x for ms in mats for x in ms}):
        add("materia", m, np.asarray([m in ms for ms in mats]))
    for b_ in FACTS_BINS:
        add("facts words", b_, facts_bin(fw) == b_)
    add("facts words", "mean", fw.astype(np.float64))
    for b_ in C.BANDS:
        add("citation band (least-cited article)", b_, bands == b_)
    for s in SELECTION_SYSTEMS:
        if dev.has(s):
            v = dev.cases(s)["ndcg_at_10"].to_numpy(np.float64)
            add("court score", f"{s} ndcg@10 on DEV", v)
    sel_table = pd.DataFrame(sel)
    sel_table.loc[sel_table["variable"] == "cases", ["kept", "dropped", "kept_minus_dropped", "lo", "hi"]] = np.nan
    C.write_csv(out / "c6_selection.csv", sel_table)

    n_kept, n_all = int(is_kept.sum()), lab.n
    notes = [f"DEV-L: {n_kept} of {n_all} DEV cases kept by the plan's gates ({n_all - n_kept} dropped); every "
             "comparison is paired on the kept cases, DEV restricted to them.",
             "Shares in the selection table are component-macro (cases averaged within their group, then groups); "
             "kept_n / dropped_n are case counts."] + notes
    ev = C.Evidence("C6")
    for r in rows:
        if r["measure"] == "ndcg_at_10" and "lay_minus_court" in r:
            ev.add(f"{r['system']}, nDCG@10 lay against court wording on the same {r['lay_minus_court']['cases']} cases",
                   ["DEV-L", "DEV (same cases)", "lay - court"],
                   [C.fmt_ci(r["dev_l_same_cases"]), C.fmt_ci(r["dev_same_cases"]), C.fmt_ci(r["lay_minus_court"])],
                   lay.source(r["system"]))
    for c in con:
        if c["measure"] == "ndcg_at_10":
            ev.add(f"{c['contrast']}, nDCG@10 gap on {c['cases']} DEV-L cases", ["lay", "court (same cases)", "lay - court"],
                   [C.fmt_ci({k: c[f"gap_{w}_{k}"] for k in ("mean", "lo", "hi")}) for w in ("lay", "court")] +
                   [C.fmt_ci({k: c[f"lay_minus_court_{k}"] for k in ("mean", "lo", "hi")})], "c6_contrasts.csv")
    for r in sel:
        if r["variable"] != "cases":
            ev.add(f"selection, {r['variable']} {r['level']}", ["kept", "dropped", "kept - dropped"],
                   [C.fmt(r["kept"]), C.fmt(r["dropped"]), C.fmt_ci({"mean": r["kept_minus_dropped"], "lo": r["lo"], "hi": r["hi"]})],
                   "c6_selection.csv")
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C6", "plan": "ANALYSIS_PLAN_C §6a (23 Oct scope)", "kept": n_kept, "dev_cases": n_all,
           "systems": names, "contrasts": [c["contrast"] for c in con if c["measure"] == "ndcg_at_10"], "notes": notes}
    C.write_json(out / "c6.json", rec)
    return rec
