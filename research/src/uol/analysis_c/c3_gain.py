"""C3: where the gain lives (ANALYSIS_PLAN_C §4; DEV, descriptive; H1-H4 are the only statements for the prose).

Unit: the (case, cited specific article) pair of DEV. Outcomes: hit@20 (stored rank <= 20) and log rank, the stored
full rank capped at 1,000 (a pair with no stored rank counts as 1,000; per-case files store the rank over the whole
catalogue, so ranks beyond the top 200 are known and capped rather than set to 1,000).

Covariates and bins (fixed by the plan):

* fold-local citation band 0 / 1-2 / 3-10 / >10 (DEV cases of the other four folds citing the article);
* facts-article overlap, the round-2 p03c formula per pair: IDF over the catalogue texts the systems score
  (``instrument_article_body``), idf = log((n + 1) / (df + .5)), the facts' unseen terms weighted log(n + 1) in the
  denominator, the mask ``[CITA_OCULTA]`` removed and the mask words subtracted; terms are the content stems of the
  Spanish analyser of ``facts_to_law.lexical`` as the plan names it (p03c itself used ``round2.a_common.content_tokens``,
  unstemmed; that variant is kept as ``overlap_p03c_tokens`` for comparison). Quintile edges over all DEV pairs, lower
  edge inclusive as in p03c;
* facts length in words (whitespace tokens of the masked facts): <= 61, 62-112, > 112;
* article length in words of ``article_body``: tertiles over the cited DEV pairs (upper edges inclusive);
* article type by the plan's rule (Amparo; Constitution; procedural or organic by title, the three organic laws or a
  heading-path keyword, substring match after accent and case folding; substantive);
* materia: "Común" among the tesis's materias, or not;
* cited articles per case: 1, 2, >= 3.

Small cells (< 30 pairs) are merged in the plan's fixed order (band 3-10 with >10; overlap quintiles 4 with 5, then 1
with 2; article-length tertiles 2 with 3); no other merge. Models: (ii) a logistic model of hit@20 per system and (iii)
a linear-probability model of the per-pair hit@20 difference for G1 = qwen3_8b - lstar_headings and G2 = S_full_ext -
S0. Coding: band enters with one coefficient per band and no intercept; every other covariate is treatment-coded
against a fixed reference (overlap Q1, facts > 112 words, article tertile 1, substantive, not Común, one cited
article). Model (ii) is all-categorical; model (iii) is fitted with overlap as the linear score 1-5 (the form H1 and
H2 name; used for every hypothesis) and, descriptively, all-categorical. Intervals: refits on 2,000 cluster-bootstrap
draws (weights = group multiplicities), seed 20261023. The logistic fit is Newton-IRLS with a 1e-6 ridge; terms whose
|coefficient| exceeds 8 are flagged as (quasi-)separated.

Hypothesis verdicts: *supported* when every interval named by the hypothesis lies on its side of the bound,
*not supported* when one lies entirely on the other side, *inconclusive* otherwise (an interval crossing its bound).
H2's "band-0 coefficient is the largest" uses the bootstrap intervals of band 0 minus each other band.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

SYSTEMS = ("popularity", "lstar_headings", "lsa", "word2vec", "bge_m3", "qwen3_0p6b", "qwen3_8b", "euler_legal", "ft_e5_q",
           "bge_m3_ft", "qwen3_0p6b_ft", "src_K_lex", "src_proto_qwen8b", "S0", "S0+qwen3_8b_dense", "S_full_ext",
           "llm_rerank_S_full_ext")
GAINS = {"G1": ("qwen3_8b", "lstar_headings"), "G2": ("S_full_ext", "S0")}
COVARIATES = ("band", "overlap_q", "facts_len", "art_len", "art_type", "materia", "n_cited")
LEVELS = {"band": list(C.BANDS), "overlap_q": ["Q1", "Q2", "Q3", "Q4", "Q5"], "facts_len": ["<=61", "62-112", ">112"],
          "art_len": ["T1", "T2", "T3"], "art_type": ["substantive", "amparo", "procedural", "constitution"],
          "materia": ["not_comun", "comun"], "n_cited": ["1", "2", ">=3"]}
REFERENCE = {"overlap_q": "Q1", "facts_len": ">112", "art_len": "T1", "art_type": "substantive", "materia": "not_comun",
             "n_cited": "1"}
MERGES = (("band", "3-10", ">10"), ("overlap_q", "Q4", "Q5"), ("overlap_q", "Q1", "Q2"), ("art_len", "T2", "T3"))
MIN_PAIRS = 30
RANK_CAP = 1000
SEPARATED = 8.0
PROC_TITLE = ("procedimiento", "procedimientos", "contencioso")
ORGANIC = ("ley organica del poder judicial de la federacion", "ley organica del tribunal federal de justicia administrativa",
           "ley organica de los tribunales agrarios")
PATH_WORDS = ("procedimiento", "juicio", "proceso", "recurso", "derecho procesal")
MASK = "[CITA_OCULTA]"


# ----------------------------------------------------------------------------- covariates

def article_types(cat: C.Catalogue) -> tuple[np.ndarray, np.ndarray]:
    """(type, rule that fired) per catalogue article, by the plan's rule."""
    titles = [C.fold(t) for t in cat.law_titles]
    law_kind = []
    for t in titles:
        if t.startswith("ley de amparo"):
            law_kind.append(("amparo", "law: Ley de Amparo"))
        elif t.startswith("constitucion politica de los estados unidos mexicanos"):
            law_kind.append(("constitution", "law: Constitution"))
        elif any(w in t for w in PROC_TITLE):
            law_kind.append(("procedural", "law title: " + next(w for w in PROC_TITLE if w in t)))
        elif any(t.startswith(o) for o in ORGANIC):
            law_kind.append(("procedural", "law: named organic law"))
        else:
            law_kind.append((None, ""))
    typ = np.empty(cat.n, dtype=object)
    rule = np.empty(cat.n, dtype=object)
    for a in range(cat.n):
        k, r = law_kind[int(cat.law[a])]
        if k is None:
            path = C.fold(" / ".join(cat.paths.get(a, [])))
            w = next((w for w in PATH_WORDS if w in path), None)
            k, r = ("procedural", f"heading path: {w}") if w else ("substantive", "")
        typ[a], rule[a] = k, r
    return typ.astype(str), rule.astype(str)


def _analyser_tokens() -> Callable[[str], set[str]]:
    from uol.facts_to_law.lexical import SpanishAnalyzer
    an = SpanishAnalyzer()
    return lambda text: set(an(text))


def _p03c_tokens() -> Callable[[str], set[str]]:
    from uol.round2.a_common import content_tokens
    return content_tokens


def overlap_pairs(query_texts: Sequence[str], articles: Sequence[int], cat_texts: Sequence[str],
                  tokens: Callable[[str], set[str]]) -> np.ndarray:
    """Per pair, the p03c overlap: IDF-weighted share of the facts' terms found in the article (IDF over the catalogue)."""
    art_tok = {}
    df: Counter = Counter()
    for i, t in enumerate(cat_texts):
        s = tokens(t)
        df.update(s)
        art_tok[i] = s
    n = len(cat_texts)
    idf = {t: np.log((n + 1) / (c + 0.5)) for t, c in df.items()}
    mask_words = tokens("cita oculta")
    out = np.zeros(len(articles))
    qcache: dict[str, tuple[set[str], float]] = {}
    for j, (q, a) in enumerate(zip(query_texts, articles)):
        if q not in qcache:
            qt = tokens(q.replace(MASK, " ")) - mask_words
            qcache[q] = (qt, sum(idf.get(t, np.log(n + 1)) for t in qt))
        qt, denom = qcache[q]
        out[j] = (sum(idf.get(t, 0.0) for t in qt & art_tok[int(a)]) / denom) if denom > 0 else 0.0
    return out


def pair_table(store: C.RunStore, cat: C.Catalogue, queries: dict[str, dict[str, Any]], *,
               tokens: Callable[[str], set[str]] | None = None, p03c: bool = True) -> tuple[pd.DataFrame, dict[str, Any]]:
    lab = store.labels
    case, art = lab.pair_arrays()
    counts = C.fold_local_counts(lab, cat.n, cat=cat)
    cnt = np.asarray([counts[int(lab.fold[c])][a] for c, a in zip(case, art)])
    qtext = [queries.get(str(lab.query_id[c]), {}).get("text", "") for c in case]
    tok = tokens or _analyser_tokens()
    ov = overlap_pairs(qtext, art, cat.texts, tok)
    q_edges = np.quantile(ov, [0.2, 0.4, 0.6, 0.8]) if len(ov) else np.zeros(4)
    oq = np.searchsorted(q_edges, ov, side="right") + 1
    fw = np.asarray([C.words(t) for t in qtext])
    aw = cat.body_words[art]
    t_edges = np.quantile(aw, [1 / 3, 2 / 3]) if len(aw) else np.zeros(2)
    at = np.searchsorted(t_edges, aw, side="left") + 1
    typ, rule = article_types(cat)
    comun = np.asarray([any(C.fold(m) == "comun" for m in queries.get(str(lab.query_id[c]), {}).get("materias", []))
                        for c in case])
    ncit = np.asarray([len(lab.positives[c]) for c in case])
    P = pd.DataFrame({"case": case, "query_id": lab.query_id[case], "component_id": lab.component[case],
                      "fold": lab.fold[case], "article": art, "cited_other_folds": cnt, "band": C.band(cnt),
                      "overlap": ov, "overlap_score": oq, "overlap_q": [f"Q{x}" for x in oq], "facts_words": fw,
                      "facts_len": np.select([fw <= 61, fw <= 112], ["<=61", "62-112"], ">112"),
                      "art_words": aw, "art_len": [f"T{x}" for x in at], "art_type": typ[art], "type_rule": rule[art],
                      "materia": np.where(comun, "comun", "not_comun"),
                      "n_cited": np.select([ncit == 1, ncit == 2], ["1", "2"], ">=3")})
    meta: dict[str, Any] = {"overlap_quintile_edges": q_edges.tolist(), "article_tertile_edges": t_edges.tolist(),
                            "overlap_terms": "SpanishAnalyzer stems" if tokens is None else "custom"}
    if p03c:
        try:
            P["overlap_p03c_tokens"] = overlap_pairs(qtext, art, cat.texts, _p03c_tokens())
            from scipy.stats import spearmanr
            meta["spearman_overlap_vs_p03c_tokens"] = float(spearmanr(P["overlap"], P["overlap_p03c_tokens"])[0])
        except Exception as exc:                            # pragma: no cover - comparison column only
            meta["p03c_tokens_error"] = f"{type(exc).__name__}: {exc}"
    merged, merges = merge_cells(P)
    meta["merges"] = merges
    return merged, meta


def merge_cells(P: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply the plan's fixed merge order to cells with fewer than 30 pairs (marginal counts)."""
    P = P.copy()
    done = []
    for cov, a, b in MERGES:
        n = P[cov].value_counts()
        if n.get(a, 0) > 0 and n.get(b, 0) > 0 and min(n[a], n[b]) < MIN_PAIRS:      # an empty cell is no cell
            lab = f"{a}+{b}"
            P[cov] = P[cov].replace({a: lab, b: lab})
            done.append(f"{cov}: {a} merged with {b} ({n.get(a, 0)} and {n.get(b, 0)} pairs)")
    return P, done


def levels(P: pd.DataFrame, cov: str) -> list[str]:
    seen = set(P[cov])
    out = []
    for lv in LEVELS[cov]:
        for x in seen:
            if (x == lv or x.split("+")[0] == lv) and x not in out:
                out.append(x)
    return out + sorted(seen - set(out))


# ----------------------------------------------------------------------------- outcomes

def pair_ranks(store: C.RunStore, system: str, P: pd.DataFrame) -> np.ndarray:
    """Stored rank of each pair (NaN when the case is absent from the system)."""
    f = store.cases(system)
    rmap: dict[tuple[int, int], int] = {}
    for i, (ok, pi, pr) in enumerate(zip(f["present"].to_numpy(), f["positive_idx"].to_numpy(), f["positive_ranks"].to_numpy())):
        if ok:
            for a, r in zip(C.parse_idx(pi), C.parse_idx(pr)):
                rmap[(i, int(a))] = int(r)
    present = f["present"].to_numpy()
    out = np.full(len(P), np.nan)
    for j, (c, a) in enumerate(zip(P["case"].to_numpy(), P["article"].to_numpy())):
        if present[c]:
            r = rmap.get((int(c), int(a)))
            out[j] = RANK_CAP if r is None or r <= 0 else min(r, RANK_CAP)
    return out


# ----------------------------------------------------------------------------- models

def design(P: pd.DataFrame, overlap: str = "categorical") -> tuple[np.ndarray, list[str]]:
    cols, names = [], []
    for lv in levels(P, "band"):
        cols.append((P["band"] == lv).to_numpy(float))
        names.append(f"band[{lv}]")
    for cov in ("overlap_q", "facts_len", "art_len", "art_type", "materia", "n_cited"):
        if cov == "overlap_q" and overlap == "linear":
            cols.append(P["overlap_score"].to_numpy(float))
            names.append("overlap_score")
            continue
        lv = levels(P, cov)
        ref = next((x for x in lv if x == REFERENCE[cov] or REFERENCE[cov] in x.split("+")), lv[0])
        for x in lv:
            if x != ref:
                v = (P[cov] == x).to_numpy(float)
                if v.any():
                    cols.append(v)
                    names.append(f"{cov}[{x}]")
    return np.column_stack(cols), names


def fit_logit(X: np.ndarray, y: np.ndarray, w: np.ndarray, beta0: np.ndarray | None = None, ridge: float = 1e-6,
              iters: int = 60) -> np.ndarray:
    beta = np.zeros(X.shape[1]) if beta0 is None else beta0.copy()
    I = np.eye(X.shape[1]) * ridge
    for _ in range(iters):
        eta = np.clip(X @ beta, -35, 35)
        p = 1.0 / (1.0 + np.exp(-eta))
        H = X.T @ (X * (w * p * (1 - p))[:, None]) + I
        g = X.T @ (w * (y - p)) - ridge * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            break
    return beta


def fit_ols(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xw = X * w[:, None]
    return np.linalg.lstsq(Xw.T @ X, Xw.T @ y, rcond=None)[0]


def bootstrap_fit(kind: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, boot: C.Boot) -> tuple[np.ndarray, np.ndarray]:
    """(point estimate, draws x p bootstrap coefficients)."""
    one = np.ones(len(y))
    if kind == "logit":
        b = fit_logit(X, y, one)
        draws = np.stack([fit_logit(X, y, boot.W[d, groups], b, iters=25) for d in range(boot.draws)])
    else:
        b = fit_ols(X, y, one)
        draws = np.stack([fit_ols(X, y, boot.W[d, groups]) for d in range(boot.draws)])
    return b, draws


def coef_table(model: str, subject: str, names: list[str], b: np.ndarray, draws: np.ndarray, n: int) -> list[dict[str, Any]]:
    rows = []
    for j, nm in enumerate(names):
        lo, hi = C.interval(draws[:, j])
        rows.append({"model": model, "subject": subject, "term": nm, "coef": float(b[j]), "lo": lo, "hi": hi,
                     "n_pairs": n, "separated": bool(model.startswith("ii") and abs(b[j]) > SEPARATED)})
    return rows


# ----------------------------------------------------------------------------- hypotheses

def _side(lo: float, hi: float, sign: int, bound: float = 0.0) -> str:
    """'yes' if the interval lies on the hypothesised side of the bound, 'no' if wholly on the other, else 'cross'."""
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "na"
    if (sign > 0 and lo > bound) or (sign < 0 and hi < bound):
        return "yes"
    if (sign > 0 and hi < bound) or (sign < 0 and lo > bound):
        return "no"
    return "cross"


def verdict(parts: Sequence[str]) -> str:
    if not parts or any(p == "na" for p in parts):
        return "not computable"
    if all(p == "yes" for p in parts):
        return "supported"
    if any(p == "no" for p in parts):
        return "not supported"
    return "inconclusive"


def hypotheses(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def term(key: str, nm: str) -> tuple[float, float, float, np.ndarray] | None:
        f = fits.get(key)
        if not f or nm not in f["names"]:
            return None
        j = f["names"].index(nm)
        lo, hi = C.interval(f["draws"][:, j])
        return float(f["b"][j]), lo, hi, f["draws"][:, j]

    t = term("iii_lin:G1", "overlap_score")
    out["H1"] = {"statement": "G1 overlap slope (linear 1-5, band fixed) < 0",
                 "coef": t[:3] if t else None, "verdict": verdict([_side(t[1], t[2], -1)]) if t else "not computable"}
    t = term("iii_lin:G2", "overlap_score")
    parts, detail = [], {}
    if t:
        lo, hi = t[1], t[2]
        if lo > 0 or hi < 0:
            parts.append("no")
        else:
            parts.append("yes" if (lo >= -0.02 and hi <= 0.02) else "cross")
        detail["overlap_slope"] = t[:3]
    f = fits.get("iii_lin:G2")
    if f:
        bands = [nm for nm in f["names"] if nm.startswith("band[")]
        b0 = next((nm for nm in bands if nm.split("[")[1].rstrip("]").split("+")[0] == "0"), None)
        if b0:
            j0 = f["names"].index(b0)
            for nm in bands:
                if nm == b0:
                    continue
                j = f["names"].index(nm)
                d = f["draws"][:, j0] - f["draws"][:, j]
                lo, hi = C.interval(d)
                detail[f"{b0} - {nm}"] = (float(f["b"][j0] - f["b"][j]), lo, hi)
                parts.append(_side(lo, hi, +1))
    out["H2"] = {"statement": "G2 overlap slope interval contains 0 and lies inside +-.02; band-0 coefficient largest",
                 "detail": detail, "verdict": verdict(parts) if t and f else "not computable"}
    parts, detail = [], {}
    for key, sign in (("ii:S0", +1), ("iii_lin:G2", -1)):
        for lv in ("amparo", "procedural"):
            t = term(key, f"art_type[{lv}]")
            if t is None:
                parts.append("na")
                continue
            detail[f"{key} {lv} vs substantive"] = t[:3]
            parts.append(_side(t[1], t[2], sign))
    out["H3"] = {"statement": "S0 hit@20 higher for Amparo and procedural than substantive (model ii, band fixed); "
                              "G2 larger for substantive than for Amparo and procedural (model iii)",
                 "detail": detail, "verdict": verdict(parts)}
    t = term("iii_lin:G2", "facts_len[<=61]")
    out["H4"] = {"statement": "G2 coefficient of <= 61 words against > 112 words > 0 (overlap fixed)",
                 "coef": t[:3] if t else None, "verdict": verdict([_side(t[1], t[2], +1)]) if t else "not computable"}
    return out


# ----------------------------------------------------------------------------- run

def by_covariate(P: pd.DataFrame, subject: str, hit: np.ndarray, logr: np.ndarray | None, groups: np.ndarray,
                 boot: C.Boot) -> list[dict[str, Any]]:
    rows = []
    ok = np.isfinite(hit)
    for cov in COVARIATES:
        for lv in levels(P, cov):
            m = ok & (P[cov] == lv).to_numpy()
            if not m.any():
                continue
            h = boot.ratio(np.where(m, np.nan_to_num(hit), 0), m.astype(float), groups)
            row = {"subject": subject, "covariate": cov, "level": lv, "n_pairs": int(m.sum()),
                   "hit_at_20": h["mean"], "hit_at_20_lo": h["lo"], "hit_at_20_hi": h["hi"]}
            if logr is not None:
                r = boot.ratio(np.where(m, np.nan_to_num(logr), 0), m.astype(float), groups)
                row.update({"log_rank": r["mean"], "log_rank_lo": r["lo"], "log_rank_hi": r["hi"]})
            rows.append(row)
    return rows


def run(runs: Path, out: Path, *, data: Path = C.DATA, population: str = "DEV", draws: int = C.DRAWS,
        seed: int = C.SEED, systems: Sequence[str] = SYSTEMS, gains: dict[str, tuple[str, str]] | None = None,
        tokens: Callable[[str], set[str]] | None = None, p03c: bool = True) -> dict[str, Any]:
    out = Path(out)
    gains = GAINS if gains is None else gains
    store = C.RunStore(runs, population)
    cat = C.load_catalogue(str(data))
    queries = C.load_queries(str(data))
    boot = C.Boot(store.labels.component, draws, seed)
    P, meta = pair_table(store, cat, queries, tokens=tokens, p03c=p03c)
    groups = boot.inv[P["case"].to_numpy()]
    have, missing = C.present(store, list(systems))
    need = sorted({s for g in gains.values() for s in g} - set(have))
    have_g = [s for s in need if store.has(s)]
    hits, logr = {}, {}
    for s in have + have_g:
        r = pair_ranks(store, s, P)
        hits[s] = np.where(np.isfinite(r), (r <= 20).astype(float), np.nan)
        logr[s] = np.log(r)
    small, coefs, fits = [], [], {}
    Xc, nc = design(P, "categorical")
    Xl, nl = design(P, "linear")
    for s in have:
        small += by_covariate(P, s, hits[s], logr[s], groups, boot)
        ok = np.isfinite(hits[s])
        b, dr = bootstrap_fit("logit", Xc[ok], hits[s][ok], groups[ok], boot)
        fits[f"ii:{s}"] = {"names": nc, "b": b, "draws": dr}
        coefs += coef_table("ii (logit, categorical)", s, nc, b, dr, int(ok.sum()))
    skipped_gains = {}
    for g, (a, bsys) in gains.items():
        if a not in hits or bsys not in hits:
            skipped_gains[g] = f"{a} - {bsys}: " + ", ".join(x for x in (a, bsys) if x not in hits) + " absent"
            continue
        d = hits[a] - hits[bsys]
        ok = np.isfinite(d)
        small += by_covariate(P, f"{g} ({a} - {bsys})", d, None, groups, boot)
        for key, X, names, label in (("iii_lin", Xl, nl, "iii (LPM, overlap linear)"),
                                     ("iii_cat", Xc, nc, "iii (LPM, categorical)")):
            b, dr = bootstrap_fit("ols", X[ok], d[ok], groups[ok], boot)
            fits[f"{key}:{g}"] = {"names": names, "b": b, "draws": dr}
            coefs += coef_table(label, f"{g} ({a} - {bsys})", names, b, dr, int(ok.sum()))
    hyp = hypotheses(fits)
    cells = [{"covariate": cov, "level": lv, "n_pairs": int((P[cov] == lv).sum())} for cov in COVARIATES for lv in levels(P, cov)]
    typ, rule = article_types(cat)
    cited = np.unique(P["article"].to_numpy())
    pick = np.random.default_rng(C.SEED).choice(cited.size, size=min(30, cited.size), replace=False)
    check = [{"article_idx": int(cited[k]), "candidate_id": cat.candidate_ids[int(cited[k])],
              "law": cat.law_titles[int(cat.law[int(cited[k])])], "heading_path": " / ".join(cat.paths.get(int(cited[k]), [])),
              "type": typ[int(cited[k])], "rule": rule[int(cited[k])], "author_ok": ""} for k in pick]
    C.write_csv(out / "c3_pairs.csv", P)
    C.write_csv(out / "c3_cells.csv", cells)
    C.write_csv(out / "c3_by_covariate.csv", small)
    C.write_csv(out / "c3_coefficients.csv", coefs)
    C.write_csv(out / "c3_article_type_check.csv", check)
    notes = [f"{len(P)} cited pairs in {store.labels.n} cases ({boot.G} groups); reference labels {store.labels.reference}.",
             f"Overlap terms: Spanish analyser stems (plan wording); Spearman with the p03c token variant: "
             f"{meta.get('spearman_overlap_vs_p03c_tokens', float('nan')):.3f}.",
             "Log rank uses the stored full rank capped at 1,000.",
             "Merges: " + ("; ".join(meta["merges"]) if meta["merges"] else "none needed (every merge cell has >= 30 pairs)."),
             "Article-type check sample (30 cited articles, seed 20261023): c3_article_type_check.csv."]
    if missing:
        notes.append("Systems named by the plan but absent from this store (skipped): " + ", ".join(missing) + ".")
    for g, why in skipped_gains.items():
        notes.append(f"Gain {g} not computed ({why}).")
    ev = C.Evidence("C3")
    for k in ("H1", "H2", "H3", "H4"):
        h = hyp[k]
        val = h["verdict"]
        if h.get("coef"):
            val += f"; {C.fmt(h['coef'][0])} [{C.fmt(h['coef'][1])}, {C.fmt(h['coef'][2])}]"
        for nm, v in (h.get("detail") or {}).items():
            val += f"; {nm}: {C.fmt(v[0])} [{C.fmt(v[1])}, {C.fmt(v[2])}]"
        ev.add(f"{k} on {population}", [h["statement"]], [val], "c3_coefficients.csv")
    sm = pd.DataFrame(small)
    for s in have:
        sub = sm[(sm["subject"] == s) & (sm["covariate"] == "band")]
        ev.add(f"{s} on {population}, hit@20 by citation band", [f"band {r.level} (n={r.n_pairs})" for r in sub.itertuples()],
               [C.fmt_ci({"mean": r.hit_at_20, "lo": r.hit_at_20_lo, "hi": r.hit_at_20_hi}) for r in sub.itertuples()],
               store.source(s))
    for g in gains:
        sub = sm[sm["subject"].str.startswith(f"{g} ")] if len(sm) else sm
        for cov in ("band", "overlap_q"):
            s2 = sub[sub["covariate"] == cov]
            if len(s2):
                ev.add(f"{s2['subject'].iloc[0]} on {population}, hit@20 difference by {cov}",
                       [f"{r.level} (n={r.n_pairs})" for r in s2.itertuples()],
                       [C.fmt_ci({"mean": r.hit_at_20, "lo": r.hit_at_20_lo, "hi": r.hit_at_20_hi}) for r in s2.itertuples()],
                       "; ".join(store.source(x) for x in gains[g]))
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C3", "plan": "ANALYSIS_PLAN_C §4", "population": population, "runs": str(runs), "pairs": len(P),
           "cases": store.labels.n, "groups": boot.G, "draws": draws, "seed": seed, "systems": have,
           "systems_missing": missing, "gains_skipped": skipped_gains, "meta": meta, "cells": cells,
           "hypotheses": hyp, "notes": notes}
    C.write_json(out / "c3.json", rec)
    return rec
