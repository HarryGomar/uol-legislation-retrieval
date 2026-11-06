"""Evaluation lenses QA-QJ on the cached dev P_spec rankings (dev, descriptive).

Every mean is component-macro over the 1,093 dev P_spec components (1,218 queries) unless stated; intervals are
paired component bootstraps (10,000 draws, seed 20261004) through ``uol.facts_to_law.stats``. Uncited articles are
unjudged: a "miss" is an article the court did not cite, not a wrong article.

    python -m uol.eval_lenses.lenses step1
    python -m uol.eval_lenses.lenses step2
"""
from __future__ import annotations

import argparse
import json
import sys
from functools import lru_cache
from typing import Any, Sequence

import numpy as np

from uol.eval_lenses import core
from uol.eval_lenses.core import LADDER, component_macro, component_values, load, meta
from uol.facts_to_law import stats as fs

KS = (1, 5, 10, 20, 50, 100)
CURVE_K = np.arange(1, 201)


# ----------------------------------------------------------------------------- shared per-query helpers

def ranks_by_query(name: str) -> list[np.ndarray]:
    m, c = meta(), load(name)
    return [c["pos_rank"][m.pos_off[i]:m.pos_off[i + 1]] for i in range(m.n)]


def recall_at(name: str, k: int) -> np.ndarray:
    return np.asarray([np.mean(r <= k) for r in ranks_by_query(name)])


def success_at(name: str, k: int) -> np.ndarray:
    return np.asarray([float(r.min() <= k) for r in ranks_by_query(name)])


def contrast(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    """Paired component bootstrap of per-query values a - b (component-macro)."""
    _, ca = component_values(a, mask)
    _, cb = component_values(b, mask)
    c = fs.contrast(ca, cb)
    return {"a": c["mean_a"], "b": c["mean_b"], "delta": c["effect"], "ci95": c["ci95"], "components": len(ca)}


def available(names: Sequence[str]) -> list[str]:
    return [n for n in names if core.cache_path(n).exists()]


# ----------------------------------------------------------------------------- QA: user-facing ranking quality

def qa() -> dict[str, Any]:
    out: dict[str, Any] = {"systems": {}, "curves": {}, "contrasts": {}}
    for name in available(LADDER):
        rq = ranks_by_query(name)
        first = np.asarray([r.min() for r in rq])
        rec = {f"recall@{k}": component_macro([np.mean(r <= k) for r in rq]) for k in KS}
        suc = {f"success@{k}": component_macro((first <= k).astype(float)) for k in KS}
        out["systems"][name] = {**suc, **rec, "ndcg@10": component_macro(load(name)["ndcg"]),
                                "mrr": component_macro(1.0 / first),
                                "median_first_rank": float(np.median(first)),
                                "first_rank_q25_q75": [float(np.quantile(first, .25)), float(np.quantile(first, .75))],
                                "share_first_beyond_100": float(np.mean(first > 100))}
        out["curves"][name] = {
            "k": CURVE_K.tolist(),
            "success": [component_macro((first <= k).astype(float)) for k in CURVE_K],
            "recall": [component_macro([np.mean(r <= k) for r in rq]) for k in CURVE_K]}
    pairs = [("LTR_emb_hgb", "LTR_noemb_hgb"), ("dense_qwen8b", "L_star"), ("LTR_noemb_hgb", "L_star"),
             ("C_hybrid", "LTR_emb_hgb")]
    for a, b in pairs:
        if not (core.cache_path(a).exists() and core.cache_path(b).exists()):
            continue
        out["contrasts"][f"{a} - {b}"] = {
            **{f"success@{k}": contrast(success_at(a, k), success_at(b, k)) for k in (1, 10, 20)},
            **{f"recall@{k}": contrast(recall_at(a, k), recall_at(b, k)) for k in (10, 20, 100)}}
    return out


# ----------------------------------------------------------------------------- QB: law versus article

def law_first_rank(name: str) -> np.ndarray:
    """Rank of the best-ranked cited instrument when instruments are scored by their best article (ties: code)."""
    m, c = meta(), load(name)
    M = c["inst_max"]
    codes = np.arange(M.shape[1])
    out = np.empty(m.n, dtype=np.int64)
    for i in range(m.n):
        gold = np.unique(m.inst[m.positives(i)])
        g = gold[np.argmax(M[i, gold])]          # argmax returns the lowest code among equal maxima
        s = M[i, g]
        out[i] = 1 + np.count_nonzero(M[i] > s) + np.count_nonzero((M[i] == s) & (codes < g))
    return out


@lru_cache(maxsize=1)
def law_sizes() -> np.ndarray:
    m = meta()
    return np.bincount(m.inst[~m.generic], minlength=len(m.instruments))


def qb() -> dict[str, Any]:
    m = meta()
    sizes = law_sizes()
    gold_size = np.asarray([np.median(sizes[np.unique(m.inst[m.positives(i)])]) for i in range(m.n)])
    n_laws = np.asarray([len(np.unique(m.inst[m.positives(i)])) for i in range(m.n)])
    out: dict[str, Any] = {"context": {"instruments": len(m.instruments),
                                       "median_articles_in_cited_law": float(np.median(gold_size)),
                                       "share_queries_one_cited_law": float(np.mean(n_laws == 1)),
                                       "chance_law_success@1_uniform": 1.0 / len(m.instruments)},
                           "systems": {}}
    for name in available(LADDER):
        c = load(name)
        lr = law_first_rank(name)
        first = np.asarray([r.min() for r in ranks_by_query(name)])
        inst_r = [c["pos_inst_rank"][m.pos_off[i]:m.pos_off[i + 1]] for i in range(m.n)]
        within1 = np.asarray([float(r.min() == 1) for r in inst_r])
        within10 = np.asarray([float(r.min() <= 10) for r in inst_r])
        law1 = (lr == 1).astype(float)
        art1 = (first == 1).astype(float)
        out["systems"][name] = {
            "law_success@1": component_macro(law1), "law_success@3": component_macro((lr <= 3).astype(float)),
            "law_success@10": component_macro((lr <= 10).astype(float)),
            "article_success@1": component_macro(art1), "article_success@10": component_macro((first <= 10).astype(float)),
            "within_law_success@1": component_macro(within1), "within_law_success@10": component_macro(within10),
            "article@1_given_law@1": float(art1[law1 == 1].mean()) if law1.any() else None,
            "law_median_rank": float(np.median(lr))}
    if all(core.cache_path(n).exists() for n in ("LTR_emb_hgb", "LTR_noemb_hgb", "L_star", "dense_qwen8b")):
        ctr = {}
        for a, b in (("LTR_emb_hgb", "LTR_noemb_hgb"), ("dense_qwen8b", "L_star")):
            la, lb = law_first_rank(a), law_first_rank(b)
            ctr[f"{a} - {b}"] = {"law_success@1": contrast((la == 1).astype(float), (lb == 1).astype(float)),
                                 "article_success@1": contrast(success_at(a, 1), success_at(b, 1)),
                                 "law_success@10": contrast((la <= 10).astype(float), (lb <= 10).astype(float)),
                                 "article_success@10": contrast(success_at(a, 10), success_at(b, 10))}
        out["contrasts"] = ctr
    return out


# ----------------------------------------------------------------------------- statute structure (G2)

@lru_cache(maxsize=1)
def structure() -> dict[str, Any]:
    """Group ids (sibling, chapter) as connected components of G2's clique edges, plus adjacency/xref neighbour sets."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    z = np.load(core.GRAPH)
    n = len(z["inst"])
    out: dict[str, Any] = {"inst": z["inst"].astype(np.int64)}
    for kind in ("sibling", "chapter"):
        e = z[f"e_{kind}"]
        a = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
        _, lab = connected_components(a, directed=False)
        grouped = np.zeros(n, dtype=bool)
        grouped[e.ravel()] = True
        out[kind] = np.where(grouped, lab, -1)       # -1: article without a group (no heading path or singleton)
    for kind in ("adj1", "adj2", "xref_int", "xref_ext"):
        e = z[f"e_{kind}"].astype(np.int64)
        out[kind] = set(map(tuple, np.sort(e, axis=1).tolist()))
    return out


RELATIONS = ("adjacent ±1", "adjacent ±2", "same sibling group", "same chapter", "cross-referenced", "same law",
             "other law")


def relation(a: int, gold: np.ndarray, st: dict[str, Any]) -> str:
    """Closest statute relation of article ``a`` to any cited article (hierarchy as in RELATIONS)."""
    pairs = [tuple(sorted((int(a), int(g)))) for g in gold]
    if any(p in st["adj1"] for p in pairs):
        return "adjacent ±1"
    if any(p in st["adj2"] for p in pairs):
        return "adjacent ±2"
    for kind, label in (("sibling", "same sibling group"), ("chapter", "same chapter")):
        ga = st[kind][a]
        if ga >= 0 and np.any(st[kind][gold] == ga):
            return label
    if any(p in st["xref_int"] or p in st["xref_ext"] for p in pairs):
        return "cross-referenced"
    if np.any(st["inst"][gold] == st["inst"][a]):
        return "same law"
    return "other law"


def qd(names: Sequence[str] = ("L_star", "dense_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid"),
       seed: int = 20261004) -> dict[str, Any]:
    """Where top-10 misses sit in the statute relative to the cited articles; with a uniform-random control."""
    m, st = meta(), structure()
    out: dict[str, Any] = {"relations": list(RELATIONS), "systems": {}}

    def shares(labels_per_query: list[list[str]]) -> dict[str, float]:
        # component-macro of per-query shares (queries with at least one miss)
        keep = np.asarray([len(x) > 0 for x in labels_per_query])
        res = {}
        for rel in RELATIONS:
            v = np.asarray([np.mean([y == rel for y in x]) if x else np.nan for x in labels_per_query])
            res[rel] = component_macro(np.nan_to_num(v), keep)
        res["near (±2, sibling or chapter)"] = sum(res[r] for r in RELATIONS[:4])
        return res

    for name in available(names):
        c = load(name)
        labs = []
        n_miss = 0
        for i in range(m.n):
            gold = m.positives(i)
            top = c["top_idx"][i, :10]
            miss = [int(a) for a in top if a not in set(gold.tolist())]
            n_miss += len(miss)
            labs.append([relation(a, gold, st) for a in miss])
        out["systems"][name] = {"misses": n_miss, "shares": shares(labs)}
    # control: 10 uniform random non-generic, non-cited articles per query
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(~m.generic)
    labs = []
    for i in range(m.n):
        gold = set(m.positives(i).tolist())
        draw = [int(a) for a in rng.choice(pool, 12, replace=False) if int(a) not in gold][:10]
        labs.append([relation(a, m.positives(i), st) for a in draw])
    out["random_control"] = {"shares": shares(labs)}
    # control 2: 10 random articles from the cited laws (what "same law" alone would give)
    labs = []
    for i in range(m.n):
        gold = m.positives(i)
        cand = np.flatnonzero(np.isin(st["inst"], st["inst"][gold]) & ~m.generic)
        cand = np.setdiff1d(cand, gold)
        draw = rng.choice(cand, min(10, len(cand)), replace=False) if len(cand) else []
        labs.append([relation(int(a), gold, st) for a in draw])
    out["same_law_random_control"] = {"shares": shares(labs)}
    return out


# ----------------------------------------------------------------------------- QA in full: MAP, R-precision, all-found, F2

def average_precision(ranks: np.ndarray) -> float:
    r = np.sort(np.asarray(ranks))
    return float(np.mean(np.arange(1, len(r) + 1) / r))


def r_precision(ranks: np.ndarray) -> float:
    return float(np.mean(np.asarray(ranks) <= len(ranks)))


def f_beta_at_k(ranks: np.ndarray, k: int, beta: float = 2.0) -> float:
    hits = float(np.sum(np.asarray(ranks) <= k))
    if hits == 0:
        return 0.0
    p, r = hits / k, hits / len(ranks)
    return (1 + beta ** 2) * p * r / (beta ** 2 * p + r)


def tuned_cutoff_f2(name: str, grid: Sequence[int] = tuple(range(1, 31)), beta: float = 2.0) -> dict[str, Any]:
    """F2 with a fixed cutoff k chosen on the other four folds (out-of-fold), applied to each held-out fold."""
    m = meta()
    rq = ranks_by_query(name)
    table = np.asarray([[f_beta_at_k(r, k, beta) for k in grid] for r in rq])        # [queries, grid]
    chosen, values = {}, np.empty(m.n)
    for f in np.unique(m.fold):
        inner = m.fold != f
        means = [component_macro(table[:, j], inner) for j in range(len(grid))]
        j = int(np.argmax(means))
        chosen[int(f)] = int(grid[j])
        values[m.fold == f] = table[m.fold == f, j]
    best_in_sample = max(component_macro(table[:, j]) for j in range(len(grid)))
    return {"f2_oof": component_macro(values), "k_by_fold": chosen, "f2_best_fixed_k_in_sample": best_in_sample,
            "per_query": values}


def qa_full() -> dict[str, Any]:
    out: dict[str, Any] = {"systems": {}}
    for name in available(LADDER):
        rq = ranks_by_query(name)
        f2 = tuned_cutoff_f2(name)
        out["systems"][name] = {
            "MAP": component_macro([average_precision(r) for r in rq]),
            "R-precision": component_macro([r_precision(r) for r in rq]),
            **{f"all_found@{k}": component_macro([float(r.max() <= k) for r in rq]) for k in (10, 20, 50, 100)},
            "F2_oof_cutoff": f2["f2_oof"], "F2_cutoff_by_fold": f2["k_by_fold"],
            "F2_in_sample_best_k": f2["f2_best_fixed_k_in_sample"]}
    a, b = "LTR_emb_hgb", "LTR_noemb_hgb"
    if all(core.cache_path(n).exists() for n in (a, b)):
        ra, rb = ranks_by_query(a), ranks_by_query(b)
        out["contrasts"] = {f"{a} - {b}": {
            "MAP": contrast(np.asarray([average_precision(r) for r in ra]), np.asarray([average_precision(r) for r in rb])),
            "R-precision": contrast(np.asarray([r_precision(r) for r in ra]), np.asarray([r_precision(r) for r in rb])),
            "F2_oof_cutoff": contrast(tuned_cutoff_f2(a)["per_query"], tuned_cutoff_f2(b)["per_query"])}}
    return out


# ----------------------------------------------------------------------------- QC: what each representation finds

QC_REPS = {"lexical (L*)": "L_star", "lexical memory": "K_lex", "dense e5-small": "dense_e5", "dense bge-m3": "dense_bge",
           "fine-tuned e5": "ft_e5_q", "dense Qwen-8B": "dense_qwen8b", "kNN memory Qwen": "knn_qwen8b",
           "prototypes Qwen": "proto_qwen8b"}


@lru_cache(maxsize=1)
def pair_bands() -> np.ndarray:
    """Fold-local popularity band of every (dev P_spec query, specific positive) pair (training folds only)."""
    from uol.round3 import common as C
    m = meta()
    out = np.empty(len(m.pos_flat), dtype=object)
    for i in range(m.n):
        band = C.band_array(f"dev_fold{m.fold[i]}")
        out[m.pos_off[i]:m.pos_off[i + 1]] = band[m.positives(i)]
    return out.astype(str)


def qc(k: int = 20) -> dict[str, Any]:
    m = meta()
    reps = {lab: n for lab, n in QC_REPS.items() if core.cache_path(n).exists()}
    hit = np.stack([load(n)["pos_rank"] <= k for n in reps.values()], axis=1)       # [pairs, reps]
    bands = pair_bands()
    labels = list(reps)
    pair_q = np.repeat(np.arange(m.n), np.diff(m.pos_off))
    out: dict[str, Any] = {"k": k, "pairs": int(len(hit)), "representations": labels, "by_band": {}}

    def summary(mask: np.ndarray) -> dict[str, Any]:
        h = hit[mask]
        any_ = h.any(axis=1)
        uniq = {lab: int(np.sum(h[:, j] & (h.sum(axis=1) == 1))) for j, lab in enumerate(labels)}
        found = {lab: int(h[:, j].sum()) for j, lab in enumerate(labels)}
        combos: dict[str, int] = {}
        for row in h:
            key = " + ".join(lab for lab, x in zip(labels, row) if x) or "(none)"
            combos[key] = combos.get(key, 0) + 1
        return {"pairs": int(mask.sum()), "found_by": found, "unique": uniq,
                "oracle_union_pair_recall": float(any_.mean()) if len(any_) else None,
                "found_by_none": int((~any_).sum()),
                "upset_top": sorted(combos.items(), key=lambda kv: -kv[1])[:20]}

    out["all"] = summary(np.ones(len(hit), dtype=bool))
    for b in ("frequent", "rare", "unseen"):
        out["by_band"][b] = summary(bands == b)
    # oracle per query: recall@k of the union of all representations' top-k (component-macro), vs the rankers
    union_recall = np.asarray([hit[pair_q == i].any(axis=1).mean() for i in range(m.n)])
    out["oracle_union_recall@20"] = component_macro(union_recall)
    out["oracle_union_list_size"] = float(np.mean([len(np.unique(np.concatenate(
        [load(n)["top_idx"][i, :k] for n in reps.values()]))) for i in range(m.n)]))
    best_single = np.max(np.stack([load(n)["ndcg"] for n in reps.values()]), axis=0)
    out["oracle_best_single_ndcg@10"] = component_macro(best_single)
    for n in ("LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid"):
        if core.cache_path(n).exists():
            out[f"{n}_recall@20"] = component_macro(recall_at(n, k))
            out[f"{n}_pair_recall@20"] = float(np.mean(load(n)["pos_rank"] <= k))
            out[f"{n}_recall@{int(round(out['oracle_union_list_size']))}"] = component_macro(
                recall_at(n, int(round(out["oracle_union_list_size"]))))
    # rank gains on pairs both rankers already find
    if all(core.cache_path(n).exists() for n in ("LTR_noemb_hgb", "LTR_emb_hgb")):
        ra, rb = load("LTR_emb_hgb")["pos_rank"], load("LTR_noemb_hgb")["pos_rank"]
        both = (ra <= k) & (rb <= k)
        out["rank_shift_on_pairs_both_find"] = {
            "pairs": int(both.sum()), "median_rank_emb": float(np.median(ra[both])),
            "median_rank_noemb": float(np.median(rb[both])), "share_improved": float(np.mean(ra[both] < rb[both])),
            "share_worse": float(np.mean(ra[both] > rb[both])),
            "share_top1_emb": float(np.mean(ra[both] == 1)), "share_top1_noemb": float(np.mean(rb[both] == 1))}
        out["ranker_pairs_by_band"] = {b: {"pairs": int(np.sum(bands == b)),
                                           "emb": float(np.mean(ra[bands == b] <= k)),
                                           "noemb": float(np.mean(rb[bands == b] <= k)),
                                           "any_representation": float(hit[bands == b].any(axis=1).mean())}
                                       for b in ("frequent", "rare", "unseen")}
    return out


# ----------------------------------------------------------------------------- QH: context budget and cost

@lru_cache(maxsize=1)
def article_tokens() -> np.ndarray:
    """Qwen-tokenizer token count of each article's embedded text (sum over windows; article_units.jsonl)."""
    n = len(core.meta().inst)
    tok = np.zeros(n, dtype=np.int64)
    with (core.QWEN / "article_units.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tok[int(r["candidate_index"])] += int(r["est_tokens"])
    return tok


BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000)


def qh(names: Sequence[str] = ("L_star", "dense_qwen8b", "K_lex", "proto_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb",
                               "C_hybrid")) -> dict[str, Any]:
    m = meta()
    tok = article_tokens()
    out: dict[str, Any] = {"article_tokens": {"median": float(np.median(tok)), "mean": float(tok.mean()),
                                              "p90": float(np.quantile(tok, .9)), "total": int(tok.sum())},
                           "budgets": list(BUDGETS), "systems": {}}
    for name in available(names):
        c = load(name)
        cum = np.cumsum(tok[np.maximum(c["top_idx"], 0)], axis=1)                    # tokens of the top-r prefix
        need: dict[str, list[float]] = {"50": [], "80": [], "90": [], "100": []}
        rec_at = {B: [] for B in BUDGETS}
        for i in range(m.n):
            r = np.sort(c["pos_rank"][m.pos_off[i]:m.pos_off[i + 1]])
            R = len(r)
            for pct in need:
                j = int(np.ceil(R * int(pct) / 100)) - 1                              # j-th found positive
                need[pct].append(float(cum[i, r[j] - 1]) if r[j] <= core.TOPN else np.inf)
            for B in BUDGETS:
                within = int(np.searchsorted(cum[i], B, side="right"))               # articles that fit in B
                rec_at[B].append(float(np.mean(r <= within)))
        out["systems"][name] = {
            **{f"median_tokens_to_{p}pct": float(np.median(v)) for p, v in need.items()},
            **{f"share_reaching_{p}pct_within_top{core.TOPN}": float(np.mean(np.isfinite(v))) for p, v in need.items()},
            "recall_at_budget": {str(B): component_macro(v) for B, v in rec_at.items()},
            "median_articles_to_first_hit": float(np.median(per_query_first(name)))}
        curve = out["systems"][name]["recall_at_budget"]
        for target in (0.5, 0.8, 0.9):
            hit = [B for B in BUDGETS if curve[str(B)] >= target]
            out["systems"][name][f"budget_for_macro_recall_{int(target * 100)}"] = hit[0] if hit else None
    out["costs"] = cost_table()
    return out


def per_query_first(name: str) -> np.ndarray:
    return np.asarray([r.min() for r in ranks_by_query(name)])


def cost_table() -> list[dict[str, Any]]:
    """Recorded costs per representation (manifests and receipts; nothing re-run). Unknown fields stay None."""
    ep_scores = core.ROOT / "experiments/runs/embedding_push_20261004/scores"
    qwen = json.loads((core.QWEN / "receipt.json").read_text(encoding="utf-8"))
    rows = []
    specs = [("Tuned lexical (L*)", "L_star", None, "sparse index; no encoder"),
             ("Lexical citation memory", "K_lex", None, "sparse index + training citations"),
             ("Dense e5-small (118M)", "dense_e5_maxp", 384, "GTX 1070 encoding of 43,935 chunks"),
             ("Dense bge-m3 (568M)", "dense_bge_maxp", 1024, "GTX 1070 encoding of 41,030 chunks"),
             ("Dense Qwen3-8B (7.6B)", "dense_qwen8b", 4096,
              f"endpoint: {qwen['counts']['est_tokens_articles']:,} article tokens + "
              f"{qwen['counts']['est_tokens_queries']['c']:,} query tokens"),
             ("kNN memory / prototypes (Qwen)", "proto_qwen8b", 4096, "reuses the Qwen query vectors"),
             ("e5-small fine-tuned", "ft_e5_q", 384, "GTX 1070 training, 5 folds + lockbox")]
    units = {384: 43935, 1024: 41030, 4096: 39286}
    for label, src, dim, note in specs:
        man = json.loads((ep_scores / src / "manifest.json").read_text(encoding="utf-8"))
        rows.append({"representation": label, "source": src, "scoring_wall_seconds_all_rows": man.get("wall_seconds"),
                     "index_MB_float32": round(units[dim] * dim * 4 / 2 ** 20) if dim else None, "note": note})
    l1 = json.loads((core.ROUND3 / "arms/L1/RESULT.json").read_text(encoding="utf-8"))
    luna = (l1.get("luna") or {})
    rows.append({"representation": "LLM listwise reranking of the top 30 (dev P_spec)", "source": "round3 L1",
                 "scoring_wall_seconds_all_rows": (l1.get("compute") or {}).get("wall_seconds"),
                 "index_MB_float32": None,
                 "note": f"{luna.get('calls')} calls, {luna.get('input_tokens')} input + {luna.get('output_tokens')} "
                         f"output tokens"})
    return rows


# ----------------------------------------------------------------------------- QI: abstention

def _weights() -> np.ndarray:
    m = meta()
    _, inv, counts = np.unique(m.component, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def qi(names: Sequence[str] = ("L_star", "dense_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid")) -> dict[str, Any]:
    """Top-score confidence vs hit: AUROC (component-weighted), risk-coverage, OOF-thresholded MCC/F1."""
    from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score
    m = meta()
    w = _weights()
    cov_grid = np.round(np.arange(0.1, 1.0001, 0.1), 2)
    out: dict[str, Any] = {"coverage": cov_grid.tolist(), "systems": {}}
    for name in available(names):
        c = load(name)
        first = per_query_first(name)
        conf = {"top1": c["top_score"][:, 0].astype(np.float64),
                "margin": (c["top_score"][:, 0] - c["top_score"][:, 1]).astype(np.float64)}
        rec: dict[str, Any] = {}
        for target, y in (("hit@1", first == 1), ("hit@10", first <= 10)):
            for cname, s in conf.items():
                auc = roc_auc_score(y, s, sample_weight=w) if 0 < y.mean() < 1 else None
                order = np.argsort(-s, kind="stable")
                rc = []
                for cv in cov_grid:
                    keep = np.zeros(m.n, dtype=bool)
                    keep[order[:max(1, int(round(cv * m.n)))]] = True
                    rc.append(component_macro(y.astype(float), keep))
                # OOF threshold: answer when conf >= t, t maximising MCC on the other folds
                pred = np.zeros(m.n, dtype=bool)
                for f in np.unique(m.fold):
                    inner = m.fold != f
                    qs = np.quantile(s[inner], np.linspace(0.02, 0.98, 49))
                    best = max(qs, key=lambda t: matthews_corrcoef(y[inner], s[inner] >= t, sample_weight=w[inner]))
                    pred[m.fold == f] = s[m.fold == f] >= best
                rec[f"{target}|{cname}"] = {
                    "auroc": auc, "risk_coverage_success": rc, "base_rate": component_macro(y.astype(float)),
                    "mcc_oof": float(matthews_corrcoef(y, pred, sample_weight=w)),
                    "f1_oof": float(f1_score(y, pred, sample_weight=w)),
                    "answered_share": float(np.average(pred, weights=w)),
                    "success_when_answered": float(np.average(y[pred], weights=w[pred])) if pred.any() else None}
        out["systems"][name] = rec
    return out


# ----------------------------------------------------------------------------- QE: strata

@lru_cache(maxsize=1)
def row_metadata() -> dict[str, np.ndarray]:
    """Per dev P_spec row: label-free query fields and the reform status of its cited articles (catalogue DOF notes)."""
    from uol.round2 import a_p14
    from uol.round3 import common as C
    d, m = core.ep.release(), meta()
    qs = [d.queries[int(r)] for r in m.rows]
    pub = C.publication_dates()
    words = np.asarray([len(q.text.split()) for q in qs])
    note_dates = {}
    texts = d.candidate_texts
    for p in np.unique(m.pos_flat):
        note_dates[int(p)] = a_p14.dates(texts[int(p)])
    reformed = np.asarray([any(note_dates[int(p)] for p in m.positives(i)) for i in range(m.n)])
    newer = np.asarray([bool(pub.get(int(m.rows[i]))) and any(x > pub[int(m.rows[i])] for p in m.positives(i)
                                                              for x in note_dates[int(p)]) for i in range(m.n)])
    return {"n_cited": np.diff(m.pos_off), "materias": np.asarray([tuple(q.materias or ()) for q in qs], dtype=object),
            "tipo": np.asarray([q.tipo or "?" for q in qs]), "instancia": np.asarray([q.instancia or "?" for q in qs]),
            "year": np.asarray([q.year or 0 for q in qs]), "words": words, "has_generic": m.n_generic_pos > 0,
            "reformed": reformed, "catalogue_newer": newer}


def strata() -> dict[str, dict[str, np.ndarray]]:
    md = row_metadata()
    out: dict[str, dict[str, np.ndarray]] = {}
    n = md["n_cited"]
    out["cited articles"] = {"1": n == 1, "2": n == 2, "3-4": (n >= 3) & (n <= 4), "5+": n >= 5}
    mats = sorted({x for t in md["materias"] for x in t})
    out["materia"] = {x: np.asarray([x in t for t in md["materias"]]) for x in mats}
    q = np.quantile(md["words"], [.25, .5, .75])
    out["facts length (words)"] = {f"<= {int(q[0])}": md["words"] <= q[0],
                                   f"{int(q[0]) + 1}-{int(q[1])}": (md["words"] > q[0]) & (md["words"] <= q[1]),
                                   f"{int(q[1]) + 1}-{int(q[2])}": (md["words"] > q[1]) & (md["words"] <= q[2]),
                                   f"> {int(q[2])}": md["words"] > q[2]}
    out["tesis type"] = {t: md["tipo"] == t for t in sorted(set(md["tipo"].tolist()))}
    inst = md["instancia"]
    out["court"] = {"SCJN": np.char.find(inst.astype(str), "Suprema") >= 0,
                    "Colegiados": np.char.find(inst.astype(str), "Colegiado") >= 0,
                    "Plenos": np.char.find(inst.astype(str), "Pleno") >= 0}
    out["also cites a generic article"] = {"yes": md["has_generic"], "no": ~md["has_generic"]}
    out["cited article has a reform note"] = {"yes": md["reformed"], "no": ~md["reformed"]}
    out["catalogue reform after the tesis"] = {"yes": md["catalogue_newer"], "no": ~md["catalogue_newer"]}
    y = md["year"]
    out["year"] = {"<= 2018": y <= 2018, "2019-2020": (y >= 2019) & (y <= 2020), "2021-2022": (y >= 2021) & (y <= 2022),
                   ">= 2023": y >= 2023}
    return out


def qe(names: Sequence[str] = ("L_star", "dense_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid")) -> dict[str, Any]:
    names = available(names)
    nd = {n: load(n)["ndcg"] for n in names}
    out: dict[str, Any] = {}
    for family, groups in strata().items():
        out[family] = {}
        for label, mask in groups.items():
            if mask.sum() < 10:
                continue
            ids, _ = component_values(np.zeros(len(mask)), mask)
            rec = {"queries": int(mask.sum()), "components": int(len(ids)),
                   **{n: component_macro(nd[n], mask) for n in names}}
            if "LTR_emb_hgb" in nd and "LTR_noemb_hgb" in nd and len(ids) >= 2:
                c = contrast(nd["LTR_emb_hgb"], nd["LTR_noemb_hgb"], mask)
                rec["emb_minus_noemb"] = {"delta": c["delta"], "ci95": c["ci95"]}
            if "LTR_emb_hgb" in nd:
                rec["success@10_emb"] = component_macro(success_at("LTR_emb_hgb", 10), mask)
            out[family][label] = rec
    return out


# ----------------------------------------------------------------------------- QJ: judged@10 and the full lists

def qj(names: Sequence[str] = ("L_star", "dense_qwen8b", "LTR_noemb_hgb", "LTR_emb_hgb", "C_hybrid"),
       sample: int = 12, seed: int = 20261004) -> dict[str, Any]:
    m, d = meta(), core.ep.release()
    out: dict[str, Any] = {"systems": {}}
    full_pos = [set(d.positive_idx[int(r)]) for r in m.rows]
    for name in available(names):
        c = load(name)
        judged = np.asarray([np.isin(c["top_idx"][i, :10], m.positives(i)).mean() for i in range(m.n)])
        unjudged_top1 = np.asarray([c["top_idx"][i, 0] not in set(m.positives(i).tolist()) for i in range(m.n)])
        rec = {"judged@10_condensed": component_macro(judged), "unjudged_share_top10": 1 - component_macro(judged),
               "top1_unjudged_share": component_macro(unjudged_top1.astype(float))}
        if core.SYSTEMS[name][0] in ("score", "pipeline"):
            S = core.load_dev_scores(name)
            full_j, gen10 = [], []
            for i in range(m.n):
                top = np.argsort(-S[i].astype(np.float64), kind="stable")[:10]   # full list, generic included
                full_j.append(np.mean([t in full_pos[i] for t in top]))
                gen10.append(np.mean(m.generic[top]))
            del S
            rec["judged@10_full_list"] = component_macro(full_j)
            rec["generic_share_top10_full_list"] = component_macro(gen10)
        out["systems"][name] = rec
    # a small qualitative sample: LTR_emb top-1 articles the court did not cite (no judgement made here)
    if core.cache_path("LTR_emb_hgb").exists():
        c, st = load("LTR_emb_hgb"), structure()
        rng = np.random.default_rng(seed)
        cand = [i for i in range(m.n) if c["top_idx"][i, 0] not in set(m.positives(i).tolist())]
        rows = []
        for i in sorted(rng.choice(cand, min(sample, len(cand)), replace=False)):
            top = int(c["top_idx"][i, 0])
            q = d.queries[int(m.rows[i])]
            rows.append({"query_id": q.query_id, "facts": q.text[:280], "cited": [d.candidate_ids[p] for p in m.positives(i)],
                         "top1_uncited": d.candidate_ids[top], "top1_text": d.candidate_texts[top][:220],
                         "relation_to_cited": relation(top, m.positives(i), st),
                         "first_cited_rank": int(c["pos_rank"][m.pos_off[i]:m.pos_off[i + 1]].min())})
        out["qualitative_sample_LTR_emb_top1_uncited"] = rows
    return out


# ----------------------------------------------------------------------------- fracción feasibility (dev spans only)

def fraccion() -> dict[str, Any]:
    """How many dev positive citation spans name a fracción / apartado / párrafo / inciso in the source text."""
    import ast
    import re
    import unicodedata
    d = core.ep.release()
    dev = {d.queries[int(r)].query_id for r in core.ep.dev_rows()}
    spec_pairs = {(d.queries[int(r)].query_id, d.candidate_ids[p]) for r in core.ep.dev_rows() for p in d.specific_idx[int(r)]}
    refs = {}
    qid_rx = re.compile(r'"query_id":\s*"([^"]+)"')
    with (core.ep.RELEASE / "references.jsonl").open(encoding="utf-8") as f:
        for line in f:
            mq = qid_rx.search(line)
            if mq and mq.group(1) in dev:                  # only dev lines are parsed; the others are discarded
                refs[mq.group(1)] = json.loads(line)["positives"]
    field = {"criterio": "criterio_juridico", "justificacion": "justificacion", "hechos": "hechos"}
    sec_text: dict[str, dict[str, str]] = {}
    with open(core.ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl", encoding="utf-8") as f:
        for line in f:
            if '"query_id"' not in line:
                continue
            r = json.loads(line)
            if r.get("query_id") in refs:
                secs = r["sections"] if isinstance(r["sections"], list) else ast.literal_eval(r["sections"])
                sec_text[r["query_id"]] = {s["field"]: s["text"] for s in secs}

    def fold_(t: str) -> str:
        return "".join(ch for ch in unicodedata.normalize("NFKD", t.lower()) if not unicodedata.combining(ch))

    pat = {"fraccion": re.compile(r"\bfracc?ion(?:es)?\s+([ivxlc]+)\b"), "apartado": re.compile(r"\bapartados?\s+[a-z]\b"),
           "parrafo": re.compile(r"\bparrafos?\b"), "inciso": re.compile(r"\bincisos?\s+[a-z]\)?")}
    counts = {k: 0 for k in pat}
    counts_any = 0
    pairs = 0
    spec = 0
    roman_ok = 0
    per_article: dict[str, set] = {}
    pair_fr: set = set()
    pair_strict: set = set()
    strict_kind: dict[str, int] = {}
    strict = re.compile(r"^\s*(?:o\.|º|°)?\s*,?\s*(?:(?:primer|segundo|tercer|cuarto|quinto|sexto|ultimo)[oa]?\s+)?"
                        r"(fracc?ion(?:es)?\s+[ivxlc]+\b|apartados?\s+[a-z]\b|parrafos?\b|incisos?\s+[a-z]\b)")
    pair_all: set = set()
    examples: list[str] = []
    for qid, pos in refs.items():
        texts = sec_text.get(qid, {})
        for p in pos:
            pairs += 1
            is_spec = (qid, p["candidate_id"]) in spec_pairs
            spec += is_spec
            t = texts.get(field.get(p["section"], p["section"]), "")
            a, b = p["span"]
            # the citation expression: from the article number to the next sentence/semicolon break (max 160 chars)
            window = fold_(t[max(0, a - 20):b + 160])
            window = re.split(r"[.;:](?:\s|$)", window, maxsplit=1)[0] if window else ""
            hits = {k: bool(rx.search(window)) for k, rx in pat.items()}
            if not is_spec:
                continue
            pair_all.add((qid, p["candidate_id"]))
            # strict: the unit is written directly after this article's number ("63, fraccion i", "20, apartado c")
            ms = strict.match(fold_(t[b:b + 50]))
            if ms:
                pair_strict.add((qid, p["candidate_id"]))
                strict_kind[ms.group(1)[:4]] = strict_kind.get(ms.group(1)[:4], 0) + 1
            for k, v in hits.items():
                counts[k] += v
            if any(hits.values()):
                counts_any += 1
            mf = pat["fraccion"].search(window)
            if mf:
                roman_ok += 1
                pair_fr.add((qid, p["candidate_id"]))
                per_article.setdefault(p["candidate_id"], set()).add(mf.group(1))
                if len(examples) < 8 and len(pair_fr) % 150 == 1:
                    examples.append(window[:160])
    # does the catalogue text expose fracciones? (Roman-numeral lines in the cited articles' text)
    roman_line = re.compile(r"(?m)^\s*[IVXLC]+\.\s")
    cited = {cid for (_, cid) in spec_pairs}
    texts = {cid: d.candidate_texts[d.candidate_index[cid]] for cid in cited}
    with_fr = sum(bool(roman_line.search(t)) for t in texts.values())
    return {"dev_positive_spans": pairs, "dev_specific_spans": spec,
            "specific_spans_naming": counts, "specific_spans_naming_any": counts_any,
            "share_naming_any": counts_any / spec if spec else None,
            "share_naming_fraccion": counts["fraccion"] / spec if spec else None,
            "fraccion_with_roman_numeral": roman_ok,
            "specific_pairs": len(pair_all), "pairs_with_a_fraccion_mention": len(pair_fr),
            "share_pairs_with_fraccion": len(pair_fr) / len(pair_all) if pair_all else None,
            "pairs_with_unit_attached_strict": len(pair_strict),
            "share_pairs_unit_attached_strict": len(pair_strict) / len(pair_all) if pair_all else None,
            "strict_mentions_by_unit": strict_kind,
            "example_windows": examples,
            "distinct_article_fraccion_labels": int(sum(len(v) for v in per_article.values())),
            "articles_cited_with_a_fraccion": len(per_article),
            "cited_articles": len(cited), "cited_articles_with_roman_fraccion_lines": with_fr,
            "method": "regex on the citation expression (span start − 20 to the next sentence break, ≤160 chars); "
                      "dev rows only"}


# ----------------------------------------------------------------------------- CLI

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["qa", "qb", "qd", "qa_full", "qc", "qh", "qi", "qe", "qj", "fraccion",
                                        "step1", "step2"])
    a = ap.parse_args(argv)
    todo = {"step1": ("qa", "qb", "qd"),
            "step2": ("qa_full", "qc", "qh", "qi", "qe", "qj", "fraccion")}.get(a.command, (a.command,))
    for lens in todo:
        res = globals()[lens]()
        core.write_json(core.OUT / f"{lens}.json", res)
        core.log(f"{lens} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
