"""EP-alpha phase 3: one-shot lockbox evaluation of the frozen pipelines.

Runs only after the lead has set ``lead_approved: true`` in ``research/state/embedding_push_20261004/FREEZE.json``
(``common.evaluate`` refuses lockbox rows otherwise). It verifies the frozen score-file hashes, scores the lockbox once,
and writes ``research/reports/embedding_push_20261004/RESULTS.md`` plus ``fusion/lockbox_results.json``. Nothing is
re-selected: the pipelines are exactly those named in FREEZE.json.

Run: ``python -m uol.embedding_push.final``. ``--rehearse-on-dev`` runs the identical code path on the dev rows (fold-local
popularity bands, no lockbox row touched, no approval needed) and writes ``fusion/RESULTS_REHEARSAL.md``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from uol.embedding_push import base, common
from uol.facts_to_law import prior as prior_mod
from uol.facts_to_law import stats

REPORT = common.ROOT / "research/reports/embedding_push_20261004/RESULTS.md"
LOCK_JSON = common.RUN / "fusion/lockbox_results.json"
REHEARSAL = common.RUN / "fusion/RESULTS_REHEARSAL.md"
REHEARSAL_JSON = common.RUN / "fusion/rehearsal_results.json"
POPS = ("P_spec", "P", "S2_spec", "all")
BANDS = ("frequent", "rare", "unseen")
RECALL_K = 20
SOURCES_JSONL = common.DERIVED / "sjf_structured_sources_v1/sources.jsonl"
QWEN_RELEASE = "2025-06-05"       # Qwen3-Embedding release date (contamination check)


def load_freeze() -> dict[str, Any]:
    freeze = json.loads(common.FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("lead_approved") is not True:
        raise SystemExit("FREEZE.json has lead_approved != true: the lockbox stays closed")
    return freeze


def verify_freeze(freeze: dict[str, Any]) -> list[str]:
    """Recompute the hashes of every frozen pipeline file and source file; return the list of mismatches."""
    problems: list[str] = []
    for fam, picks in freeze["pipelines"].items():
        for crit, rec in picks.items():
            path = Path(rec["scores_path"])
            if common.sha256_file(path) != rec["scores_sha256"]:
                problems.append(f"{fam}/{crit}: {path} differs from the frozen hash")
            if common.sha256_file(path.parent / "manifest.json") != rec["manifest_sha256"]:
                problems.append(f"{fam}/{crit}: manifest differs from the frozen hash")
    for sid, rec in freeze["upstream_score_files"].items():
        if common.sha256_file(Path(rec["path"])) != rec["sha256"]:
            problems.append(f"source {sid}: {rec['path']} differs from the frozen hash")
    for name, rec in ((freeze.get("lockbox_ablation_scores") or {}).get("files") or {}).items():
        if common.sha256_file(Path(rec["path"])) != rec["sha256"]:
            problems.append(f"lockbox ablation file {name} differs from the frozen hash")
    check = freeze.get("contamination_check") or {}
    for group in ("pipelines", "qwen_sources"):
        for name, rec in (check.get(group) or {}).items():
            if common.sha256_file(Path(rec["path"])) != rec["sha256"]:
                problems.append(f"contamination-check file {name}: {rec['path']} differs from the frozen hash")
    if common.sha256_file(common.SPLIT_PATH) != freeze["split_sha256"]:
        problems.append("SPLIT.json differs from the frozen hash")
    return problems


def publication_split(cutoff: str = QWEN_RELEASE) -> dict[str, Any]:
    """Lockbox rows published before / on-or-after ``cutoff`` (source publication_date; record_id order when the date is missing).

    Only ``query_id`` and ``publication_date`` of the structured-sources file are read.
    """
    d = common.release()
    wanted = {d.queries[int(r)].query_id: int(r) for r in common.lockbox_rows()}
    dates: dict[int, str | None] = {}
    with SOURCES_JSONL.open(encoding="utf-8") as handle:
        for line in handle:
            if '"query_id"' not in line:
                continue
            rec = json.loads(line)
            row = wanted.get(rec.get("query_id"))
            if row is not None:
                dates[row] = rec.get("publication_date") or None
    before = [r for r, v in dates.items() if v and v < cutoff]
    after = [r for r, v in dates.items() if v and v >= cutoff]
    undated = [r for r in wanted.values() if not dates.get(r)]
    record = lambda r: int(d.queries[r].record_id)  # noqa: E731
    method = "publication_date"
    if undated and before and after:
        threshold = (max(record(r) for r in before) + min(record(r) for r in after)) / 2
        for r in undated:
            (before if record(r) < threshold else after).append(r)
        method = "publication_date, record_id order for undated rows"
    dated = sorted((v, r) for r, v in dates.items() if v)
    median_date = dated[len(dated) // 2][0] if dated else None
    earlier = sorted(r for v, r in dated if v < median_date) if median_date else []
    later = sorted(r for v, r in dated if v >= median_date) if median_date else []
    return {"before": sorted(before), "after": sorted(after), "earlier_half": earlier, "later_half": later, "median_date": median_date,
            "date_range": [dated[0][0], dated[-1][0]] if dated else None,
            "undated_unassigned": sorted(r for r in undated if r not in before and r not in after),
            "cutoff": cutoff, "method": method, "n_dated": len(dated), "n_undated": len(undated)}


def contamination_check(check: dict[str, Any], emb_s: str, noemb_s: str, nolsa_s: str | None, picks: dict[str, Any]) -> dict[str, Any]:
    """Primary contrast and Qwen-containing pipelines / sources on lockbox rows published before vs after the Qwen release."""
    split = publication_split(check.get("cutoff", QWEN_RELEASE))
    files = {rec["pipeline"]: Path(rec["scores_path"]) for fam in picks.values() for rec in fam.values()}
    files.update({n: Path(r["path"]) for n, r in (check.get("pipelines") or {}).items()})
    files.update({n: Path(r["path"]) for n, r in (check.get("qwen_sources") or {}).items()})
    qwen_names = sorted(set((check.get("pipelines") or {}).keys()) | set((check.get("qwen_sources") or {}).keys()))
    out: dict[str, Any] = {"cutoff": split["cutoff"], "method": split["method"], "date_range": split["date_range"], "median_date": split["median_date"],
                           "rows": {k: len(split[k]) for k in ("before", "after", "earlier_half", "later_half", "undated_unassigned")},
                           "qwen_containing": qwen_names, "subsets": {}}
    for label in ("before", "after", "earlier_half", "later_half"):
        rows = np.asarray(split[label], dtype=np.int64)
        if len(rows) == 0:
            continue
        res: dict[str, dict[str, Any]] = {}
        for name in sorted({emb_s, noemb_s, *([nolsa_s] if nolsa_s else []), *qwen_names}):
            scores = np.asarray(np.load(files[name], mmap_mode="r")[rows])
            res[name] = {}
            for pop in ("P_spec", "P"):
                try:
                    res[name][pop] = common.evaluate(scores, rows, pop, allow_lockbox=True)
                except ValueError:
                    pass
        entry: dict[str, Any] = {"n_rows": int(len(rows)),
                                 "n_P_spec": int(res[emb_s]["P_spec"]["n_queries"]) if "P_spec" in res[emb_s] else 0,
                                 "components_P_spec": int(res[emb_s]["P_spec"]["n_components"]) if "P_spec" in res[emb_s] else 0,
                                 "metrics": {n: {pop: float(r["macro"]["ndcg_at_10"]) for pop, r in by.items()} for n, by in res.items()},
                                 "contrasts": {}}

        def add(key: str, a: str, b: str, pop: str = "P_spec") -> None:
            if pop in res.get(a, {}) and pop in res.get(b, {}):
                try:
                    entry["contrasts"][key] = {"a": a, "b": b, **common.contrast(res[a][pop], res[b][pop])}
                except ValueError as error:
                    entry["contrasts"][key] = {"a": a, "b": b, "verdict": f"not assessed ({error})"}

        add("primary: BEST_EMB - BEST_NO-EMB, P_spec", emb_s, noemb_s)
        if nolsa_s:
            add("sensitivity: BEST_EMB - BEST_NO-EMB(no LSA), P_spec", emb_s, nolsa_s)
        for name in qwen_names:
            add(f"{name} - BEST_NO-EMB, P_spec", name, noemb_s)
        out["subsets"][label] = entry
    return out


def popularity_labels() -> np.ndarray:
    """Band of each candidate from the citation counts of all dev components (the lockbox's training rows)."""
    d = common.release()
    dev_ids = [d.queries[int(r)].query_id for r in common.dev_rows()]
    counts = prior_mod.component_counts(d.references, d.component_by_query, dev_ids, len(d.candidate_ids), d.candidate_index)
    return np.where(counts == 0, "unseen", np.where(counts <= prior_mod.RARE_MAX, "rare", "frequent"))


def pair_hits(res: dict[str, Any], labels) -> tuple[np.ndarray, np.ndarray]:
    """Per component and band: pairs ranked within RECALL_K and total pairs (positives kept by the view's condensed list).

    ``labels`` is one band label per candidate, or a callable mapping a release row to such an array (fold-local bands on dev).
    """
    d = common.release()
    index = {c: i for i, c in enumerate(res["components"])}
    hits = np.zeros((len(index), len(BANDS)))
    total = np.zeros_like(hits)
    for row, rec in zip(res["rows"], res["per_query"]):
        ci = index[d.queries[int(row)].component_id]
        lab = labels(int(row)) if callable(labels) else labels
        for p, rank in zip(rec["positive_idx"], rec["positive_ranks"]):
            b = BANDS.index(str(lab[int(p)]))
            total[ci, b] += 1
            hits[ci, b] += rank <= RECALL_K
    return hits, total


def band_table(a: dict[str, Any], b: dict[str, Any], labels, draws: int = stats.DRAWS, seed: int = stats.SEED) -> dict[str, Any]:
    """Pair-level Recall@20 per band for two systems plus the paired component-bootstrap interval of a - b."""
    ha, ta = pair_hits(a, labels)
    hb, tb = pair_hits(b, labels)
    n = ha.shape[0]
    rng = np.random.default_rng(seed)
    groups = list(range(len(BANDS))) + [None]
    point = {}
    for g in groups:
        sl = slice(None) if g is None else slice(g, g + 1)
        name = "all" if g is None else BANDS[g]
        point[name] = {"pairs": float(ta[:, sl].sum()), "a": float(ha[:, sl].sum() / max(ta[:, sl].sum(), 1)),
                       "b": float(hb[:, sl].sum() / max(tb[:, sl].sum(), 1))}
    boots = {("all" if g is None else BANDS[g]): np.empty(draws) for g in groups}
    for start in range(0, draws, 256):
        count = min(256, draws - start)
        idx = rng.integers(0, n, size=(count, n))
        for g in groups:
            sl = slice(None) if g is None else slice(g, g + 1)
            name = "all" if g is None else BANDS[g]
            tot = ta[:, sl].sum(axis=1)[idx].sum(axis=1)
            boots[name][start:start + count] = np.divide(ha[:, sl].sum(axis=1)[idx].sum(axis=1), tot, out=np.zeros(count), where=tot > 0) \
                - np.divide(hb[:, sl].sum(axis=1)[idx].sum(axis=1), tot, out=np.zeros(count), where=tot > 0)
    for name, rec in point.items():
        lo, hi = np.quantile(boots[name], [0.025, 0.975])
        rec.update({"diff": rec["a"] - rec["b"], "ci95": [float(lo), float(hi)]})
    return point


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rehearse-on-dev", action="store_true", help="same code path on dev rows; never touches the lockbox")
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    rehearsal = args.rehearse_on_dev
    freeze = json.loads(common.FREEZE_PATH.read_text(encoding="utf-8")) if rehearsal else load_freeze()
    problems = verify_freeze(freeze)
    if problems:
        raise SystemExit("frozen artefacts changed after the freeze:\n" + "\n".join(problems))
    picks = freeze["pipelines"]
    names = sorted({rec["pipeline"] for fam in picks.values() for rec in fam.values()})
    rows = common.dev_rows() if rehearsal else common.lockbox_rows()
    result: dict[str, dict[str, Any]] = {}
    paths = {rec["pipeline"]: Path(rec["scores_path"]) for fam in picks.values() for rec in fam.values()}
    for name in names:
        scores = np.asarray(np.load(paths[name], mmap_mode="r")[rows])
        result[name] = {pop: common.evaluate(scores, rows, pop, allow_lockbox=not rehearsal) for pop in POPS}
        print(f"{'dev rehearsal' if rehearsal else 'lockbox'} {name}: P_spec nDCG@10 {result[name]['P_spec']['macro']['ndcg_at_10']:.4f}", flush=True)
    emb_s, noemb_s = picks["EMB"]["P_spec"]["pipeline"], picks["NO-EMB"]["P_spec"]["pipeline"]
    emb_p, noemb_p = picks["EMB"]["P"]["pipeline"], picks["NO-EMB"]["P"]["pipeline"]
    contrasts: dict[str, Any] = {}
    for pop in POPS:
        contrasts[f"PRIMARY-choice pair, {pop}"] = {"a": emb_s, "b": noemb_s, **common.contrast(result[emb_s][pop], result[noemb_s][pop])}
    for metric in ("mrr", "recall_at_20"):
        contrasts[f"PRIMARY-choice pair, P_spec {metric}"] = {"a": emb_s, "b": noemb_s,
                                                              **common.contrast(result[emb_s]["P_spec"], result[noemb_s]["P_spec"], metric)}
    contrasts["P-choice pair, P"] = {"a": emb_p, "b": noemb_p, **common.contrast(result[emb_p]["P"], result[noemb_p]["P"])}
    nolsa = picks.get("NO-EMB-noLSA")
    if nolsa:      # sensitivity (lead ruling D4): BEST_NO-EMB drawn from pipelines without any LSA-built source
        ns, np_ = nolsa["P_spec"]["pipeline"], nolsa["P"]["pipeline"]
        for pop in POPS:
            contrasts[f"SENSITIVITY no-LSA, PRIMARY-choice pair, {pop}"] = {"a": emb_s, "b": ns, **common.contrast(result[emb_s][pop], result[ns][pop])}
        contrasts["SENSITIVITY no-LSA, P-choice pair, P"] = {"a": emb_p, "b": np_, **common.contrast(result[emb_p]["P"], result[np_]["P"])}
    contamination = None
    if not rehearsal and freeze.get("contamination_check"):
        contamination = contamination_check(freeze["contamination_check"], emb_s, noemb_s, (nolsa or {}).get("P_spec", {}).get("pipeline"), picks)
    ablation = None
    if not rehearsal and freeze.get("lockbox_ablation_scores"):
        ref_e = "LTR_emb_hgb" if "LTR_emb_hgb" in result else emb_s
        ref_n = "LTR_noemb_hgb" if "LTR_noemb_hgb" in result else noemb_s
        ablation = lockbox_ablation(freeze, None, ref_e, ref_n, result)
    labels = fold_local_labels() if rehearsal else popularity_labels()
    bands = {"P_spec": band_table(result[emb_s]["P_spec"], result[noemb_s]["P_spec"], labels),
             "P": band_table(result[emb_p]["P"], result[noemb_p]["P"], labels)}
    primary = contrasts["PRIMARY-choice pair, P_spec"]
    serial = {"generated_utc": base.utc(), "mode": "dev_rehearsal" if rehearsal else "lockbox", "freeze_sha256": common.sha256_file(common.FREEZE_PATH),
              "pipelines": picks,
              "metrics": {n: {pop: {"n_queries": r[pop]["n_queries"], "n_components": r[pop]["n_components"], **{k: float(v) for k, v in r[pop]["macro"].items()}}
                              for pop in POPS} for n, r in result.items()},
              "contrasts": contrasts, "bands": bands, "primary": primary, "contamination": contamination, "ablation": ablation,
              "wall_seconds": time.perf_counter() - t0}
    (REHEARSAL_JSON if rehearsal else LOCK_JSON).write_text(json.dumps(serial, indent=1, default=float), encoding="utf-8")
    write_report(freeze, serial, emb_s, noemb_s, emb_p, noemb_p, REHEARSAL if rehearsal else REPORT, rehearsal)
    print(f"primary: {primary['effect']:+.4f} ci95 {primary['ci95']} {primary['verdict']}", flush=True)
    return 0


def lockbox_ablation(freeze: dict[str, Any], full_scores: np.ndarray, ref_emb: str, ref_noemb: str, results: dict[str, Any]) -> dict[str, Any]:
    """Lockbox metrics of the frozen ablation variants (score rows written before approval; the model is LTR_emb_hgb's family)."""
    rows = common.lockbox_rows()
    files = (freeze.get("lockbox_ablation_scores") or {}).get("files") or {}
    out: dict[str, Any] = {}
    for name, rec in files.items():
        scores = np.load(rec["path"])
        res = {pop: common.evaluate(scores, rows, pop, allow_lockbox=True) for pop in ("P_spec", "P")}
        kind = "removed" if name.startswith("emb_minus_") else "added"
        group = name.split("_minus_")[-1] if kind == "removed" else name.split("_plus_")[-1]
        ref = ref_emb if kind == "removed" else ref_noemb
        out[name] = {"kind": kind, "group": group, "P_spec": float(res["P_spec"]["macro"]["ndcg_at_10"]), "P": float(res["P"]["macro"]["ndcg_at_10"]),
                     "vs_reference": {pop: {"reference": ref, **{k: v for k, v in common.contrast(res[pop], results[ref][pop]).items()
                                                                 if k in ("effect", "ci95", "verdict")}} for pop in ("P_spec", "P")}}
    return out


def fold_local_labels():
    """Dev rehearsal: band of each candidate from the training folds of the row's own fold."""
    d = common.release()
    roles = common.roles()
    by_fold = {}
    for k in range(5):
        ids = [d.queries[int(r)].query_id for r in common.dev_rows() if roles[int(r)] != f"dev_fold{k}"]
        counts = prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)
        by_fold[k] = np.where(counts == 0, "unseen", np.where(counts <= prior_mod.RARE_MAX, "rare", "frequent"))
    return lambda row: by_fold[int(str(roles[row]).replace("dev_fold", ""))]


def write_report(freeze: dict[str, Any], serial: dict[str, Any], emb_s: str, noemb_s: str, emb_p: str, noemb_p: str,
                 path: Path = REPORT, rehearsal: bool = False) -> None:
    m, c, bands, picks = serial["metrics"], serial["contrasts"], serial["bands"], freeze["pipelines"]
    prim = serial["primary"]
    dev = freeze["dev_results"]["summaries"]
    if rehearsal:
        head = ["# DEV REHEARSAL of the lockbox report (not a result)", "",
                f"Generated {serial['generated_utc']}. Same code path as the lockbox report, run on the 4,812 dev rows (out-of-fold scores, "
                "fold-local popularity bands). It exists only to test the report; the lockbox has not been scored.", ""]
    else:
        head = ["# Embedding push: lockbox results (EP-alpha)", "",
                f"Generated {serial['generated_utc']}. The lockbox (1,168 queries in 991 components, published mid-2025 to 2026) was scored once, "
                "after the lead approved FREEZE.json; every pipeline, hyper-parameter and selection below was fixed on dev out-of-fold results only, "
                "and the lockbox never trained, selected or tuned anything (models for its rows were trained on all dev rows).", ""]
    sens = c.get("SENSITIVITY no-LSA, PRIMARY-choice pair, P_spec")
    sens_lines = []
    if sens:
        sens_lines = ["", f"**Sensitivity, BEST_NO-EMB without any LSA-built source** (lead ruling D4): Delta = **{sens['effect']:+.4f}** "
                          f"(95% CI [{sens['ci95'][0]:+.4f}, {sens['ci95'][1]:+.4f}], 90% CI [{sens['ci90'][0]:+.4f}, {sens['ci90'][1]:+.4f}]) "
                          f"-> **{sens['verdict']}**; BEST_NO-EMB(no LSA) = `{sens['b']}` ({fmt(sens['mean_b'])}).", ""]
    L = head + [
         "## Verdict", "",
         f"**Primary contrast** (Delta = BEST_EMB - BEST_NO-EMB, {'dev rehearsal' if rehearsal else 'lockbox'} P_spec, nDCG@10): **{prim['effect']:+.4f}** "
         f"(95% CI [{prim['ci95'][0]:+.4f}, {prim['ci95'][1]:+.4f}], 90% CI [{prim['ci90'][0]:+.4f}, {prim['ci90'][1]:+.4f}]; "
         f"{prim['components']} components, 10,000 paired bootstrap draws, seed 20261004) -> **{prim['verdict']}**.", "",
         f"- BEST_EMB = `{emb_s}` (lockbox P_spec nDCG@10 {fmt(prim['mean_a'])}); BEST_NO-EMB = `{noemb_s}` ({fmt(prim['mean_b'])}).",
         "- Decision rule (margin .02): lower 95% bound >= .02 USEFUL_IMPROVEMENT; > 0 IMPROVEMENT; 90% interval within +-.02 NO_USEFUL_DIFFERENCE; "
         "upper bound < 0 WORSE; otherwise INCONCLUSIVE."] + sens_lines + [""] + [
         "## Frozen pipelines", "", "| family | chosen by dev P_spec | chosen by dev P |", "|---|---|---|"]
    for fam in ("NO-EMB", "NO-EMB-noLSA", "EMB"):
        if fam in picks:
            L.append(f"| {fam} | {picks[fam]['P_spec']['pipeline']} | {picks[fam]['P']['pipeline']} |")
    L += ["", "## Lockbox metrics (component-macro; condensed lists for P_spec / S2_spec)", "",
          "| pipeline | P_spec nDCG@10 | MRR | R@20 | P nDCG@10 | MRR | R@20 | S2_spec nDCG@10 | all nDCG@10 | dev P_spec nDCG@10 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for n, r in m.items():
        L.append(f"| {n} | {fmt(r['P_spec']['ndcg_at_10'])} | {fmt(r['P_spec']['mrr'])} | {fmt(r['P_spec']['recall_at_20'])} | {fmt(r['P']['ndcg_at_10'])} | "
                 f"{fmt(r['P']['mrr'])} | {fmt(r['P']['recall_at_20'])} | {fmt(r['S2_spec']['ndcg_at_10'])} | {fmt(r['all']['ndcg_at_10'])} | "
                 f"{fmt(dev[n]['P_spec']['ndcg_at_10'])} |")
    first = next(iter(m.values()))
    L += ["", "Rows / components: " + "; ".join(f"{p} {first[p]['n_queries']}/{first[p]['n_components']}" for p in POPS), "",
          "## Contrasts (a - b; paired component bootstrap; secondary rows are unadjusted)", "",
          "| contrast | a | b | mean a | mean b | effect | 95% CI | 90% CI | verdict |", "|---|---|---|---:|---:|---:|---|---|---|"]
    for label, r in c.items():
        L.append(f"| {label} | {r['a']} | {r['b']} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | {r['effect']:+.4f} | "
                 f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | [{r['ci90'][0]:+.4f}, {r['ci90'][1]:+.4f}] | {r['verdict']} |")
    L += ["", "## Pair-level Recall@20 by popularity band", "",
          "Bands from the number of distinct dev components citing the article (the lockbox's training rows): unseen 0, rare 1-4, frequent 5+. "
          "A pair is one (query, cited article); the interval is a paired component bootstrap of the difference.", ""]
    for pop, (a, b) in (("P_spec", (emb_s, noemb_s)), ("P", (emb_p, noemb_p))):
        L += [f"### {pop} (a = {a}, b = {b})", "", "| band | pairs | Recall@20 a | Recall@20 b | a - b | 95% CI |", "|---|---:|---:|---:|---:|---|"]
        for band in (*BANDS, "all"):
            r = bands[pop][band]
            L.append(f"| {band} | {int(r['pairs'])} | {fmt(r['a'])} | {fmt(r['b'])} | {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] |")
        L.append("")
    abl = serial.get("ablation")
    if abl:
        L += ["## Per-group ablation on the lockbox (LTR hgb with specific labels, frozen dev hyper-parameters, trained on all dev rows)", "",
              "Removed: LTR_emb_hgb without the group (compared with the full LTR_emb_hgb). Added: NO-EMB pipeline plus only that group "
              "(compared with LTR_noemb_hgb); `qwen8b` = Qwen-only, `finetuned_dense` = ft_e5-only.", "",
              "| variant | group | P_spec nDCG@10 | effect vs reference (P_spec) | 95% CI | P nDCG@10 | effect vs reference (P) |", "|---|---|---:|---:|---|---:|---:|"]
        for name, r in abl.items():
            a, b = r["vs_reference"]["P_spec"], r["vs_reference"]["P"]
            L.append(f"| {name} | {r['group']} | {fmt(r['P_spec'])} | {a['effect']:+.4f} | [{a['ci95'][0]:+.4f}, {a['ci95'][1]:+.4f}] | {fmt(r['P'])} | {b['effect']:+.4f} |")
        L.append("")
    cont = serial.get("contamination")
    if cont:
        L += ["## Contamination check: lockbox rows published before vs on/after the Qwen3-Embedding release (" + cont["cutoff"] + ")", "",
              f"Split by source publication_date ({cont['method']}). Lockbox publication dates run {cont['date_range'][0]} to {cont['date_range'][1]}: "
              f"rows before the release {cont['rows']['before']}, on/after {cont['rows']['after']}, unassigned {cont['rows']['undated_unassigned']}. "
              "When no row precedes the release the requested split is empty, so the whole lockbox postdates the encoder's release and a "
              "pre-training exposure of these texts is not possible by date. As a drift check the lockbox is also split at its median publication date "
              f"({cont['median_date']}): earlier half {cont['rows']['earlier_half']} rows, later half {cont['rows']['later_half']}.", ""]
        names = {"before": "before the release", "after": "on/after the release", "earlier_half": "earlier half of the lockbox (drift check)",
                 "later_half": "later half of the lockbox (drift check)"}
        for label, entry in cont["subsets"].items():
            L += [f"### {names[label]}: {entry['n_rows']} lockbox rows, {entry['n_P_spec']} P_spec queries in {entry['components_P_spec']} components", "",
                  "| contrast | a | b | mean a | mean b | effect | 95% CI | verdict |", "|---|---|---|---:|---:|---:|---|---|"]
            for key, r in entry["contrasts"].items():
                if "effect" in r:
                    L.append(f"| {key} | {r['a']} | {r['b']} | {fmt(r['mean_a'])} | {fmt(r['mean_b'])} | {r['effect']:+.4f} | "
                             f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | {r['verdict']} |")
                else:
                    L.append(f"| {key} | {r['a']} | {r['b']} | | | | | {r['verdict']} |")
            L += ["", "Lockbox nDCG@10 by pipeline on this subset: " + "; ".join(
                f"{n}: P_spec {fmt(v['P_spec']) if 'P_spec' in v else 'n/a'}, P {fmt(v['P']) if 'P' in v else 'n/a'}" for n, v in entry["metrics"].items()), ""]
    L += ["## Dev reference (selection-biased; from FREEZE.json)", "", "| pipeline | dev P_spec nDCG@10 | dev P nDCG@10 |", "|---|---:|---:|"]
    for n in m:
        L.append(f"| {n} | {fmt(dev[n]['P_spec']['ndcg_at_10'])} | {fmt(dev[n]['P']['ndcg_at_10'])} |")
    L += ["", "Dev contrasts, selection-biased: " + "; ".join(
        f"{k}: {v['effect']:+.4f}" for k, v in freeze["dev_results"]["contrasts"].items() if k.startswith("BEST_EMB - BEST_NOEMB")), "",
          "## Provenance", "",
          f"- FREEZE.json sha256 `{serial['freeze_sha256'][:16]}`; frozen score-file hashes verified before scoring.",
          "- Dev details, ablations and per-group configurations: `experiments/runs/embedding_push_20261004/fusion/DEV_RESULTS.md`; "
          "base methods: `experiments/runs/embedding_push_20261004/base/DEV_SUMMARY.md`.",
          "- Stacking caveat for dev numbers: learned models used out-of-fold features that had seen each group's held-out fold (second order); "
          "the lockbox numbers are unaffected."]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
