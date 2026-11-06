"""C7: confidence and selective LLM use (ANALYSIS_PLAN_C §7; DEV, descriptive).

Systems: P1-P6, the frozen answer systems (field ``answer`` of ``systems.json``; P2 has no frozen row yet and is
added with ``--answer P2=<system>``); P3+L and P5+CE are not in "P1-P6" and are left out. Signals per case, all
oriented so that larger means more confident:

* ``top1``: the top-1 score on the system's own scale; ``margin``: top-1 minus top-2 score;
* ``neg_entropy``: minus the entropy of a softmax (temperature 1) over the top-10 scores after a per-case z-score of
  those ten scores (a constant list gives the uniform softmax);
* ``law_share``: the largest share of the top 10 held by one law;
* ``agree_dense_klex``: top-1 of dense_qwen8b (the system ``qwen3_8b``, the frozen score file dense_qwen8b) equals top-1
  of ``src_K_lex`` (one value per case, the same for every system);
* LLM pass agreement (C5) needs the saved LLM orders, which are not in the runs folder: skipped with a note.

Measures, case-level (risk is a case outcome): risk-coverage curves for success@1 and success@10 (cases sorted by
confidence, descending, ties in case order), AURC (mean risk over the n coverage points), AUROC (Mann-Whitney, ties
count one half); intervals by the cluster bootstrap (group multiplicities as case weights). Platt calibration: for
each outer fold, a logistic fit of the outcome on the standardised signal over the other four folds' cases, applied
to the fold; reliability over 10 equal-count bins of the cross-fitted probabilities, with ECE and Brier.

Selective LLM: per outer fold, the threshold is the q-quantile (q = .2, .4, .6, .8) of S_full_ext's margin over the
other folds' cases; a held-out case with margin below it takes the second stage's per-case nDCG@10
(llm_rerank_S_full_ext, then ce_trained), the rest keep S_full_ext's; component-macro nDCG@10 against the realised
share sent, with the 0 % and 100 % end points.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C
from uol.analysis_c.c3_gain import fit_logit

SIGNALS = ("top1", "margin", "neg_entropy", "law_share", "agree_dense_klex")
OUTCOMES = ("success_at_1", "success_at_10")
SHARES = (0.2, 0.4, 0.6, 0.8)
SELECTIVE = (("S_full_ext", "llm_rerank_S_full_ext"), ("S_full_ext", "ce_trained"))
AGREE = ("qwen3_8b", "src_K_lex")
GRID = np.round(np.arange(0.05, 1.0001, 0.05), 2)


def signals(idx: np.ndarray, score: np.ndarray, law: np.ndarray) -> dict[str, np.ndarray]:
    s10 = score[:, :10]
    top1 = score[:, 0]
    margin = score[:, 0] - score[:, 1]
    mu = np.nanmean(s10, axis=1, keepdims=True)
    sd = np.nanstd(s10, axis=1, keepdims=True)
    z = np.where(sd > 0, (s10 - mu) / np.where(sd > 0, sd, 1), 0.0)
    z = np.nan_to_num(z, nan=-np.inf)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    L = np.where(idx[:, :10] >= 0, law[np.maximum(idx[:, :10], 0)], -1)
    share = np.asarray([np.bincount(r[r >= 0]).max() / 10.0 if (r >= 0).any() else np.nan for r in L])
    bad = idx[:, 0] < 0
    out = {"top1": top1, "margin": margin, "neg_entropy": -ent, "law_share": share}
    for k in out:
        out[k] = np.where(bad, np.nan, out[k])
    return out


def risk_coverage(conf: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, risk) after sorting by confidence descending (stable, so ties keep case order)."""
    w = np.ones_like(y, dtype=float) if w is None else w
    o = np.argsort(-conf, kind="stable")
    cw = np.cumsum(w[o])
    err = np.cumsum(w[o] * (1 - y[o]))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cw / cw[-1], err / cw


def aurc(conf: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    w = np.ones_like(y, dtype=float) if w is None else w
    _, risk = risk_coverage(conf, y, w)
    o = np.argsort(-conf, kind="stable")
    ww = w[o]
    return float(np.nansum(risk * ww) / ww.sum())


def auroc(conf: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    """Weighted Mann-Whitney AUROC; tied confidences count one half."""
    w = np.ones_like(y, dtype=float) if w is None else w
    vals, inv = np.unique(conf, return_inverse=True)
    pos = np.bincount(inv, weights=w * y, minlength=vals.size)
    neg = np.bincount(inv, weights=w * (1 - y), minlength=vals.size)
    below = np.cumsum(neg) - neg
    num = (pos * (below + 0.5 * neg)).sum()
    den = pos.sum() * neg.sum()
    return float(num / den) if den > 0 else float("nan")


def platt(conf: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Cross-fitted Platt probabilities (fit on the other folds, standardised signal)."""
    p = np.full(conf.size, np.nan)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        mu, sd = conf[tr].mean(), conf[tr].std() or 1.0
        X = np.column_stack([np.ones(tr.sum()), (conf[tr] - mu) / sd])
        b = fit_logit(X, y[tr], np.ones(tr.sum()))
        p[te] = 1 / (1 + np.exp(-(b[0] + b[1] * (conf[te] - mu) / sd)))
    return p


def reliability(p: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[list[dict[str, Any]], float, float]:
    o = np.argsort(p, kind="stable")
    rows, ece = [], 0.0
    for j, part in enumerate(np.array_split(o, bins)):
        if not part.size:
            continue
        mp, oy = float(p[part].mean()), float(y[part].mean())
        rows.append({"bin": j + 1, "n": int(part.size), "mean_predicted": mp, "observed": oy})
        ece += part.size / p.size * abs(mp - oy)
    return rows, float(ece), float(np.mean((p - y) ** 2))


def selective(base: pd.DataFrame, second: pd.DataFrame, margin: np.ndarray, fold: np.ndarray, boot: C.Boot,
              shares: Sequence[float] = SHARES) -> list[dict[str, Any]]:
    nb = base["ndcg_at_10"].to_numpy(float)
    ns = second["ndcg_at_10"].to_numpy(float)
    ok = np.isfinite(nb) & np.isfinite(ns) & np.isfinite(margin)
    rows = []
    for q in (0.0,) + tuple(shares) + (1.0,):
        send = np.zeros(nb.size, dtype=bool)
        if q >= 1.0:
            send[:] = True
        elif q > 0:
            for k in np.unique(fold):
                tr, te = ok & (fold != k), ok & (fold == k)
                thr = np.quantile(margin[tr], q)
                send[te] = margin[te] < thr
        v = np.where(ok, np.where(send, ns, nb), np.nan)
        d = boot.macro(v)
        rows.append({"target_share": q, "realised_share": float(send[ok].mean()), "ndcg_at_10": d["mean"],
                     "lo": d["lo"], "hi": d["hi"], "cases": int(ok.sum())})
    return rows


def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", draws: int = C.DRAWS, seed: int = C.SEED,
        answer: dict[str, str] | None = None, selective_pairs: Sequence[tuple[str, str]] = SELECTIVE) -> dict[str, Any]:
    out = Path(out)
    store = C.RunStore(runs, population)
    cat = C.load_catalogue(str(data))
    boot = C.Boot(store.labels.component, draws, seed)
    fold = store.labels.fold
    P = C.answer_systems(answer)
    notes = [f"Reference labels {store.labels.reference}; {store.labels.n} cases, {boot.G} groups.",
             "LLM pass agreement (C5) not computed: the saved LLM orders are not part of the runs folder."]
    skipped = {p: s for p, s in P.items() if not store.has(s)}
    missing_p = [p for p in ("P1", "P2", "P3", "P4", "P5", "P6") if p not in P]
    if skipped:
        notes.append("Answer systems absent from this store (skipped): " + ", ".join(f"{p} {s}" for p, s in skipped.items()) + ".")
    if missing_p:
        notes.append("No frozen system for " + ", ".join(missing_p) + " (pass --answer P2=<system> once chosen).")
    agree = None
    if all(store.has(s) for s in AGREE):
        a, _ = store.top(AGREE[0], 1)
        b, _ = store.top(AGREE[1], 1)
        agree = np.where((a[:, 0] >= 0) & (b[:, 0] >= 0), (a[:, 0] == b[:, 0]).astype(float), np.nan)
    else:
        notes.append(f"Agreement signal skipped: {', '.join(s for s in AGREE if not store.has(s))} absent.")
    summary, curves, rel_rows = [], [], []
    ev = C.Evidence("C7")
    margins: dict[str, np.ndarray] = {}
    for p, s in P.items():
        if not store.has(s):
            continue
        f = store.cases(s)
        idx, sc = store.top(s, 10)
        sig = signals(idx, sc, cat.law)
        if agree is not None:
            sig["agree_dense_klex"] = agree
        margins[s] = sig["margin"]
        for oc in OUTCOMES:
            y = f[oc].to_numpy(float)
            vals = []
            for name, conf in sig.items():
                ok = np.isfinite(conf) & np.isfinite(y)
                c, yy = conf[ok], y[ok]
                Wc = boot.case_weights()[:, ok]
                a_pt, r_pt = auroc(c, yy), aurc(c, yy)
                a_b = np.asarray([auroc(c, yy, Wc[d]) for d in range(boot.draws)])
                r_b = np.asarray([aurc(c, yy, Wc[d]) for d in range(boot.draws)])
                pr = platt(c, yy, fold[ok])
                rel, ece, brier = reliability(pr, yy)
                for r in rel:
                    rel_rows.append({"answer": p, "system": s, "outcome": oc, "signal": name, **r})
                cov, risk = risk_coverage(c, yy)
                for g in GRID:
                    j = min(int(np.ceil(g * c.size)) - 1, c.size - 1)
                    curves.append({"answer": p, "system": s, "outcome": oc, "signal": name, "coverage": float(g),
                                   "risk": float(risk[j])})
                row = {"answer": p, "system": s, "outcome": oc, "signal": name, "cases": int(ok.sum()),
                       "base_rate": float(yy.mean()), "auroc": a_pt, "auroc_lo": C.interval(a_b)[0],
                       "auroc_hi": C.interval(a_b)[1], "aurc": r_pt, "aurc_lo": C.interval(r_b)[0],
                       "aurc_hi": C.interval(r_b)[1], "platt_ece": ece, "platt_brier": brier}
                summary.append(row)
                vals.append(row)
            ev.add(f"{p} {s} on {population}, {oc}", [f"AUROC {r['signal']}" for r in vals],
                   [C.fmt_ci({"mean": r["auroc"], "lo": r["auroc_lo"], "hi": r["auroc_hi"]}) for r in vals],
                   store.source(s, "top200.parquet"))
    sel_rows = []
    for base, second in selective_pairs:
        if not (store.has(base) and store.has(second)):
            notes.append(f"Selective {second} over {base} skipped: {', '.join(x for x in (base, second) if not store.has(x))} absent.")
            continue
        m = margins.get(base)
        if m is None:
            idx, sc = store.top(base, 10)
            m = signals(idx, sc, cat.law)["margin"]
        rows = selective(store.cases(base), store.cases(second), m, fold, boot)
        for r in rows:
            sel_rows.append({"base": base, "second_stage": second, **r})
        ev.add(f"selective {second} over {base} on {population}", [f"nDCG@10 sending {r['realised_share']:.0%}" for r in rows],
               [C.fmt_ci({"mean": r["ndcg_at_10"], "lo": r["lo"], "hi": r["hi"]}) for r in rows],
               f"{store.source(base)}; {store.source(second)}")
    C.write_csv(out / "c7_signals.csv", summary)
    C.write_csv(out / "c7_risk_coverage.csv", curves)
    C.write_csv(out / "c7_reliability.csv", rel_rows)
    C.write_csv(out / "c7_selective.csv", sel_rows)
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C7", "plan": "ANALYSIS_PLAN_C §7", "population": population, "runs": str(runs),
           "answer_systems": P, "skipped": skipped, "missing": missing_p, "notes": notes, "draws": draws, "seed": seed}
    C.write_json(out / "c7.json", rec)
    return rec
