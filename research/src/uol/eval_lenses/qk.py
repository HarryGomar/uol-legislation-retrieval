"""QK: what an LLM knows without retrieval (dev, descriptive; use case U3, LLM agents).

Inputs are the saved Round 3 contamination probes (``phase2/probes``): Luna's closed-book answers (up to 10 (law,
article) pairs from the masked facts) for 200 pre-cutoff DEV rows. Only those rows are read. The post-cutoff sample is
lockbox and its outputs are never indexed. No LLM is called.

On the same rows, scored like the probe (full gold lists, generic articles included, row means):
* closed-book recall of the cited articles and of the cited laws (reproduces PROBES.json);
* a profile of every named pair: cited; an uncited catalogue article (near a cited article, elsewhere in a cited law,
  in another law, or a repealed text); a state or local law; an abrogated law, decree or transitory article; an
  article number that does not exist in a catalogue law; a federal-sounding name that matches no catalogue law;
* Recall@3/5/10 (articles and laws) of L*, dense Qwen3-8B and the two rankers on the full catalogue ranking;
* grounding: where the embedding ranker puts the model's correct and incorrect named articles.

    python -m uol.eval_lenses.qk
"""
from __future__ import annotations

import json
import re
import sys
from typing import Any

import numpy as np

from uol.embedding_push import common as ep
from uol.eval_lenses import core
from uol.eval_lenses.lenses import relation, structure
from uol.facts_to_law import metrics

PROBES = core.ROUND3 / "phase2/probes"
SYSTEMS = {"L_star": ep.score_path("L_star"), "dense_qwen8b": ep.score_path("dense_qwen8b"),
           "LTR_noemb_hgb": ep.RUN / "fusion/pipelines/LTR_noemb_hgb/scores.npy",
           "LTR_emb_hgb": ep.RUN / "fusion/pipelines/LTR_emb_hgb/scores.npy"}
KS = (3, 5, 10)
STATES = ("aguascalientes", "baja california", "campeche", "chiapas", "chihuahua", "coahuila", "colima", "durango",
          "guanajuato", "guerrero", "hidalgo", "jalisco", "michoacan", "morelos", "nayarit", "nuevo leon", "oaxaca",
          "puebla", "queretaro", "quintana roo", "san luis potosi", "sinaloa", "sonora", "tabasco", "tamaulipas",
          "tlaxcala", "veracruz", "yucatan", "zacatecas")
STATE_RX = re.compile(r"\b(del estado de|para el estado|estado libre|ciudad de mexico|distrito federal|municipios|"
                      + "|".join(STATES) + r")\b")
ABROGATED_RX = re.compile(r"(codigo federal de procedimientos penales|\b19[0-9]{2}\b|abrogad|^decreto\b)")
CATEGORIES = ("cited", "uncited: next to or same section as a cited article", "uncited: elsewhere in a cited law",
              "uncited: another catalogue law", "repealed text in the catalogue",
              "state or local law", "abrogated law, decree or transitory article",
              "article number not in the catalogue law", "federal-sounding law not matched to the catalogue")


def fold(text: str) -> str:
    from uol.round3 import catalog_map
    return catalog_map.fold(text)


def load_probe_rows() -> tuple[list[int], dict[int, list[dict[str, str]]]]:
    """The 200 pre-cutoff dev rows and their closed-book answers; refuses any lockbox row."""
    saved = json.loads((PROBES / "outputs.json").read_text(encoding="utf-8"))
    rows = [int(r) for r in saved["samples"]["pre_cutoff_dev"]]
    if np.isin(rows, ep.lockbox_rows()).any() or not np.isin(rows, ep.dev_rows()).all():
        raise PermissionError("the QK sample must be dev rows only")
    cb = saved["outputs"]["closedbook"]
    answers = {r: list(cb[str(r)]["articulos"]) for r in rows}
    del saved, cb                                                      # the post-cutoff outputs are not used
    return rows, answers


def classify(name: str, number: str, gold: set[int], st: dict[str, Any]) -> tuple[str, int | None]:
    from uol.round3 import catalog_map
    f = fold(name)
    if STATE_RX.search(f):
        return "state or local law", None
    if ABROGATED_RX.search(f) or "transitori" in fold(number):
        return "abrogated law, decree or transitory article", None
    idx, why = catalog_map.resolve(name, number)
    if idx is None:
        return ("federal-sounding law not matched to the catalogue" if why == "instrument_unresolved"
                else "article number not in the catalogue law"), None
    if idx in gold:
        return "cited", idx
    body = ep.release().candidate_texts[idx].split("\n", 1)[-1]
    if "deroga" in " ".join(body.split()[:8]).lower():
        return "repealed text in the catalogue", idx
    if not gold:
        return "uncited: another catalogue law", idx
    rel = relation(idx, np.asarray(sorted(gold)), st)
    if rel in ("adjacent ±1", "adjacent ±2", "same sibling group", "same chapter"):
        return "uncited: next to or same section as a cited article", idx
    if rel in ("same law", "cross-referenced") and np.any(st["inst"][sorted(gold)] == st["inst"][idx]):
        return "uncited: elsewhere in a cited law", idx
    return "uncited: another catalogue law", idx


def inst_code(idx: int) -> str:
    return ep.release().candidate_ids[idx].rsplit("/", 1)[-1].split(".pdf")[0]


def qk() -> dict[str, Any]:
    from uol.round3 import catalog_map
    d, st = ep.release(), structure()
    rows, answers = load_probe_rows()
    gold = {r: set(int(a) for a in d.positive_idx[r]) for r in rows}
    scored = [r for r in rows if gold[r]]
    comp = np.asarray([d.queries[r].component_id for r in scored])

    def macro(v: np.ndarray) -> float:
        ids, inv = np.unique(comp, return_inverse=True)
        return float((np.bincount(inv, weights=v) / np.bincount(inv)).mean())

    # closed-book: recall (articles, laws) as the probe scored it, plus the named-pair profile
    cb_rec, cb_inst, n_named, cb_prec = [], [], [], []
    profile = {c: 0 for c in CATEGORIES}
    named_idx: dict[int, list[tuple[int, bool]]] = {}
    for r in scored:
        pred, pred_inst, items = set(), set(), []
        for it in answers[r]:
            cat, idx = classify(it["ley"], it["articulo"], gold[r], st)
            profile[cat] += 1
            ridx, _ = catalog_map.resolve(it["ley"], it["articulo"])
            if ridx is not None:
                pred.add(ridx)
            code = catalog_map.instrument(it["ley"])
            if code:
                pred_inst.add(code)
            if idx is not None and idx not in {a for a, _ in items}:      # an article named twice counts once
                items.append((idx, idx in gold[r]))
        named_idx[r] = items
        g_inst = {inst_code(a) for a in gold[r]}
        cb_rec.append(len(gold[r] & pred) / len(gold[r]))
        cb_inst.append(len(g_inst & pred_inst) / len(g_inst))
        n_named.append(len(answers[r]))
        cb_prec.append(len(gold[r] & pred) / len(answers[r]) if answers[r] else 0.0)
    total = sum(profile.values())
    closed = {"rows": len(scored), "components": int(len(np.unique(comp))),
              "recall": float(np.mean(cb_rec)), "recall_component_macro": macro(np.asarray(cb_rec)),
              "law_recall": float(np.mean(cb_inst)), "mean_pairs_named": float(np.mean(n_named)),
              "precision_of_named_pairs": float(np.mean(cb_prec)), "named_pairs": total,
              "profile_counts": profile, "profile_shares": {c: profile[c] / total for c in CATEGORIES}}
    probes = json.loads((PROBES / "PROBES.json").read_text(encoding="utf-8"))["pre_cutoff_dev"]
    closed["reproduces_PROBES_json"] = {"recall": probes["closedbook_recall"],
                                        "law_recall": probes["closedbook_instrument_recall"],
                                        "match": abs(probes["closedbook_recall"] - closed["recall"]) < 1e-9
                                        and abs(probes["closedbook_instrument_recall"] - closed["law_recall"]) < 1e-9}

    # retrieval on the same rows: full catalogue ranking, full gold lists
    tie = ep.tie_order()
    retrieval: dict[str, Any] = {}
    grounding: dict[str, Any] = {}
    for name, path in SYSTEMS.items():
        S = np.asarray(np.load(path, mmap_mode="r")[np.asarray(scored)], dtype=np.float32)
        res = metrics.evaluate_matrix(S, [sorted(gold[r]) for r in scored], top_k=10, tie_order=tie)
        rec = {k: np.asarray([np.mean(np.asarray(q["positive_ranks"]) <= k) for q in res]) for k in KS}
        law = {}
        for k in KS:
            law[k] = np.asarray([len({inst_code(a) for a in q["topk"][:k]} & {inst_code(a) for a in gold[r]})
                                 / len({inst_code(a) for a in gold[r]}) for q, r in zip(res, scored)])
        union = np.asarray([len(gold[r] & (set(q["topk"][:10]) | {i for i, _ in named_idx[r]})) / len(gold[r])
                            for q, r in zip(res, scored)])
        retrieval[name] = {**{f"recall@{k}": float(rec[k].mean()) for k in KS},
                           **{f"law_recall@{k}": float(law[k].mean()) for k in KS},
                           **{f"recall@{k}_component_macro": macro(rec[k]) for k in KS},
                           "recall@10_union_with_closed_book": float(union.mean())}
        if name == "LTR_emb_hgb":
            hit_ranks, miss_ranks = [], []
            for i, r in enumerate(scored):
                if not named_idx[r]:
                    continue
                idxs = [a for a, _ in named_idx[r]]
                rk = metrics.rank_of_positives(S[i], idxs, tie_order=tie)
                for (a, ok), x in zip(named_idx[r], rk):
                    (hit_ranks if ok else miss_ranks).append(int(x))
            hit, miss = np.asarray(hit_ranks), np.asarray(miss_ranks)
            from sklearn.metrics import roc_auc_score
            y = np.r_[np.ones(len(hit)), np.zeros(len(miss))]
            auc = roc_auc_score(y, -np.r_[hit, miss].astype(float))
            kept_hits = int(np.sum(hit <= 20))
            kept_miss = int(np.sum(miss <= 20))
            grounding = {"named_catalogue_articles": int(len(hit) + len(miss)), "cited": int(len(hit)),
                         "uncited": int(len(miss)), "median_rank_cited": float(np.median(hit)),
                         "median_rank_uncited": float(np.median(miss)), "auroc_rank_separates_cited": float(auc),
                         "filter_top20": {"kept_cited": kept_hits, "kept_uncited": kept_miss,
                                          "precision_before": float(len(hit) / (len(hit) + len(miss))),
                                          "precision_after": float(kept_hits / max(kept_hits + kept_miss, 1)),
                                          "cited_kept_share": float(kept_hits / max(len(hit), 1))}}
        del S
    return {"label": "dev, descriptive", "sample": "200 pre-cutoff dev probe rows (published 2020-2025); rows without "
            "a resolvable gold article dropped as in the probe", "closed_book": closed, "retrieval": retrieval,
            "grounding_LTR_emb": grounding}


def main() -> int:
    res = qk()
    core.write_json(core.OUT / "qk.json", res)
    core.log("qk written")
    print(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
