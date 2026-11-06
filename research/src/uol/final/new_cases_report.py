"""E13 report: RESULTS.md, contrasts, evidence rows and ``thesis/evidence/numbers_new_cases.tex`` from the scored and
evaluated new cases. It reads only the run folder, FREEZE.json and the frozen record; it scores nothing.

Contrasts follow the families pre-registered in FREEZE.json (``evaluation.contrast_families``): paired nDCG@10 over
NEW components, Bonferroni within a family. The LLM family is also reported on the cases decided after each model's
stated cutoff (``evaluation.llm_cutoff_split``); every NEW case is published after both cutoffs.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from uol.final.new_cases_fit import ROOT, sha256, write_json

MEASURES = {"nDCG@10": "ndcg_at_10", "MRR": "mrr", "R@20": "recall_at_20", "R@100": "recall_at_100",
            "s@1": "success_at_1", "s@10": "success_at_10", "law@1": "law_at_1", "law@10": "law_at_10"}
LABELS = {"lstar_headings": "P1 lexical (L*, headings)", "S0": "S0 ranker without embeddings (P3)",
          "S_full": "S_full ranker with embeddings", "S_full_ext": "S_full_ext integrated ranker",
          "llm_rerank_S_full_ext": "P6 LLM stage over S_full_ext", "claude_rerank_S_full_ext": "Claude reorder",
          "jev_S_full_ext": "Jev alone (top 30)", "jev_rrf_S_full_ext": "Jev RRF with S_full_ext"}
ORDER = tuple(LABELS)
MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
          "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
DATE = re.compile(r"(\d{1,2})\s+de\s+([a-záéíóú]+)\s+de\s+(\d{4})", re.I)
EVIDENCE = ROOT / "research" / "thesis_direction_20261015" / "EVIDENCE.md"
TEX = ROOT / "thesis" / "evidence" / "numbers_new_cases.tex"
SECTION = "## New cases after the freeze (E13, D161/D162)"


def precedent_dates(text: str | None) -> list[str]:
    out = []
    for d, m, y in DATE.findall(text or ""):
        mm = MONTHS.get(m.lower())
        if mm:
            try:
                out.append(date(int(y), mm, int(d)).isoformat())
            except ValueError:
                continue
    return sorted(out)


def decision_dates(release: Path) -> dict[str, dict[str, Any]]:
    """query_id -> earliest and latest precedent decision date from the collected source rows the release used."""
    man = json.loads((Path(release) / "manifest.json").read_text(encoding="utf-8"))
    src = Path(man["sources"]["path"])
    src = src if src.is_absolute() else ROOT / src
    if sha256(src) != man["sources"]["sha256"]:
        raise RuntimeError(f"{src}: differs from the release manifest")
    out = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        ds = precedent_dates(r.get("source_precedentes"))
        out[r["query_id"]] = {"earliest": ds[0] if ds else None, "latest": ds[-1] if ds else None, "n": len(ds)}
    return out


def by_component(frame, measure: str) -> dict[str, float]:
    g = frame[np.isfinite(frame[measure].to_numpy(dtype=np.float64))].groupby("component_id")[measure].mean()
    return {str(k): float(v) for k, v in g.items()}


def pair(frames: dict, a: str, b: str, measure: str, family_size: int, keep=None) -> dict[str, Any] | None:
    from uol.round3 import stats
    fa, fb = frames[a], frames[b]
    if keep is not None:
        fa, fb = fa[fa["query_id"].isin(keep)], fb[fb["query_id"].isin(keep)]
    ca, cb = by_component(fa, measure), by_component(fb, measure)
    comps = sorted(set(ca) & set(cb))
    if len(comps) < 2:
        return None
    r = stats.contrast([ca[c] for c in comps], [cb[c] for c in comps], family_size=family_size)
    r["n_cases"] = int(len(fa))
    return r


def contrasts(run: Path, freeze: dict, frames: dict, dates: dict, headline: set[str]) -> dict[str, Any]:
    """Each pre-registered contrast on the headline set (HEADLINE_RULE.json), the LLM family also split by the model's
    cutoff within it; ``incl_overlap`` is the same contrast on all NEW (sensitivity)."""
    fam = freeze["evaluation"]["contrast_families"]
    cut = freeze["evaluation"]["llm_cutoff_split"]["cutoffs"]
    out: dict[str, Any] = {}
    for name, f in fam.items():
        m = len(f["pairs"])
        for a, b in f["pairs"]:
            if a not in frames or b not in frames:
                out[f"{name}/{a} - {b}"] = {"missing": [s for s in (a, b) if s not in frames]}
                continue
            out[f"{name}/{a} - {b}"] = {"all": pair(frames, a, b, f["measure"], m, headline),
                                        "incl_overlap": pair(frames, a, b, f["measure"], m)}
            if name != "llm":
                continue
            cs = [cut.get(s, {}).get("cutoff") for s in (a, b) if s in cut]
            if not cs or any(c is None for c in cs):
                out[f"{name}/{a} - {b}"]["after_cutoff"] = {"not_defined": "a model in the pair has no stated cutoff"}
                continue
            c = max(cs)                                   # after both models' cutoffs
            after = {q for q, d in dates.items() if d["earliest"] and d["earliest"] > c} & headline
            rest = (set(dates) & headline) - after
            out[f"{name}/{a} - {b}"]["after_cutoff"] = {"cutoff": c, **(pair(frames, a, b, f["measure"], m, after) or
                                                                        {"too_few": True})}
            out[f"{name}/{a} - {b}"]["rest"] = {"cutoff": c, **(pair(frames, a, b, f["measure"], m, rest) or
                                                                {"too_few": True})}
    return out


def _f(x, signed=False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    s = f"{abs(x):.3f}".lstrip("0") if abs(x) < 1 else f"{abs(x):.3f}"
    return ("−" if x < 0 else ("+" if signed else "")) + s


def _tex(x, signed=False) -> str:
    s = f"{abs(x):.3f}"
    return ("$-$" if x < 0 else ("+" if signed else "")) + s


def subset_summary(frame, keep) -> dict[str, Any] | None:
    from uol.final import measures
    sub = frame[frame["query_id"].isin(keep)]
    return measures.summary(sub) if len(sub) else None


def costs(run: Path) -> dict[str, Any]:
    out = {}
    for sid in ("llm_rerank_S_full_ext", "claude_rerank_S_full_ext", "jev_S_full_ext"):
        p = run / "scores" / sid / "manifest.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            out[sid] = {k: m.get(k) for k in ("usage", "usd", "input_tokens", "usd_at_configured_price", "seconds",
                                              "unanswered", "repairs", "models", "llm", "model", "effort")
                        if m.get(k) is not None}
            out[sid]["manifest_sha256"] = sha256(p)
    return out


def collection(run: Path) -> dict[str, Any]:
    d = run / "collect"
    out: dict[str, Any] = {}
    if (d / "copied_from_round2.json").exists():
        out["copied_from_round2"] = json.loads((d / "copied_from_round2.json").read_text(encoding="utf-8"))
    res = sorted(d.glob("collect_result_*.json"))
    out["runs"] = [{"file": p.name, "sha256": sha256(p), **json.loads(p.read_text(encoding="utf-8"))} for p in res]
    if res and "copied_from_round2" in out:
        last = out["runs"][-1]["lines_now"]
        out["new_lines"] = {n: last[n] - out["copied_from_round2"][n].get("lines", 0) for n in last
                            if out["copied_from_round2"].get(n, {}).get("lines") is not None}
    return out


def drift(run: Path) -> dict[str, Any]:
    """The run computer's control statistics (GATE.json via the intake) and the local encoders' controls."""
    out: dict[str, Any] = {}
    p = run / "vectors" / "runb_manifest.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        out["runb"] = {"manifest_sha256": sha256(p), "gate": m["gate"],
                       "bge": {k: m["bge.json"].get(k) for k in ("control_cls_cos", "control_token_counts_match")},
                       "qwen": {k: m["qwen.json"].get(k) for k in ("control_cos",)},
                       "pairs": {k: m["pairs.json"].get(k) for k in ("control_pool_identical_share", "control_pool_jaccard",
                                                                     "control_pairs", "control_spearman")}}
    p = run / "vectors" / "local_manifest.json"
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        out["local"] = {"manifest_sha256": sha256(p), "controls": m.get("controls")}
    return out


def build(run: Path) -> dict[str, Any]:
    import pandas as pd
    from uol.final import new_cases
    run = Path(run)
    fpath = new_cases.freeze_path()
    freeze = json.loads(fpath.read_text(encoding="utf-8"))
    counts = json.loads((run / "NEW" / "counts.json").read_text(encoding="utf-8"))["counts"]   # release()'s record
    dates = decision_dates(run / "NEW" / "release")
    frames, summ = {}, {}
    for sid in ORDER:
        p = run / "eval" / sid / "summary.json"
        if p.exists():
            summ[sid] = json.loads(p.read_text(encoding="utf-8"))
            summ[sid]["_sha256"] = sha256(p)
            frames[sid] = pd.read_parquet(run / "eval" / sid / "per_case.parquet")
            if sha256(run / "eval" / sid / "per_case.parquet") != summ[sid]["per_case_sha256"]:
                raise RuntimeError(f"{sid}: per_case.parquet differs from its summary")
    rule_path = run / "HEADLINE_RULE.json"
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    overlap = {c["query_id"] for c in rule["overlap_cases"]}
    flagged = set().union(*(set(f.loc[f["overlap_v11"], "query_id"]) for f in frames.values())) if frames else set()
    if frames and flagged != overlap & set().union(*(set(f["query_id"]) for f in frames.values())):
        raise RuntimeError("the evaluated overlap cases differ from HEADLINE_RULE.json")
    headline = set(dates) - overlap
    blocks = {sid: {"headline": subset_summary(f, headline),
                    "second_look": subset_summary(f, headline & set(f.loc[~f["first_look"], "query_id"])),
                    "incl_overlap": summ[sid]["all"]} for sid, f in frames.items()}
    con = contrasts(run, freeze, frames, dates, headline)
    cut = freeze["evaluation"]["llm_cutoff_split"]["cutoffs"]
    split = {}
    for sid, c in cut.items():
        if c["cutoff"] is None or sid not in frames:
            continue
        after = {q for q, d in dates.items() if d["earliest"] and d["earliest"] > c["cutoff"]}
        q_eval = set(frames[sid]["query_id"]) & headline
        split[sid] = {"cutoff": c["cutoff"], "n_after": len(after & q_eval), "n_rest": len(q_eval - after),
                      "system_after": subset_summary(frames[sid], after),
                      "base_after": subset_summary(frames["S_full_ext"], after) if "S_full_ext" in frames else None,
                      "system_rest": subset_summary(frames[sid], q_eval - after),
                      "base_rest": subset_summary(frames["S_full_ext"], q_eval - after) if "S_full_ext" in frames else None}
    q_eval = set(next(iter(frames.values()))["query_id"]) if frames else set()
    undated = sorted(q for q in q_eval if not dates.get(q, {}).get("earliest"))
    rep = {"freeze_sha256": sha256(fpath), "freeze_utc": freeze["utc"], "counts": counts,
           "counts_sha256": sha256(run / "NEW" / "counts.json"), "summaries": summ, "blocks": blocks,
           "headline_rule": {"sha256": sha256(rule_path), "utc": rule["utc"], "rule": rule["rule"],
                             "overlap_cases": sorted(overlap)}, "contrasts": con,
           "cutoff_split": split, "undated_evaluated_cases": undated, "costs": costs(run), "collection": collection(run),
           "drift": drift(run), "spread_fold0": freeze.get("spread_fold0"), "not_scored": freeze.get("not_scored"),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(run / "report.json", rep)
    (run / "RESULTS.md").write_text(results_md(run, rep), encoding="utf-8")
    TEX.write_text(tex(run, rep), encoding="utf-8")
    (run / "EVIDENCE_ROWS.md").write_text(evidence_rows(run, rep), encoding="utf-8")
    return {"report": str(run / "report.json"), "results": str(run / "RESULTS.md"), "tex": str(TEX),
            "evidence_rows": str(run / "EVIDENCE_ROWS.md")}


def results_md(run: Path, rep: dict) -> str:
    c, s = rep["counts"], rep["summaries"]
    rel = run.relative_to(ROOT).as_posix() if run.is_absolute() and ROOT in run.parents else run.as_posix()
    L = [f"# New cases after the freeze (E13): results", "",
         f"Run `{rel}`. FREEZE.json sha256 `{rep['freeze_sha256']}` (written {rep['freeze_utc']}); every system scored "
         f"once, after the freeze, with the frozen models. Generated by `uol.final.new_cases_report` at {rep['utc']} "
         f"from `report.json` (every value below traces to a file named here with its sha256).", "",
         "## Cases", "",
         f"Collected source rows {c['source_rows']}; publication dates {c['publication_dates']}; by month {c['by_month']}.",
         f"Exclusions by the frozen resolver and masker: {c['exclusions']}.", "",
         "| Block | Cases | Groups | Catalogue-citing | P | P_spec | P_spec groups | v11 overlap |",
         "|---|---|---|---|---|---|---|---|"]
    for k in ("all", "first_look", "second_look"):
        b = c[k]
        L.append(f"| {k} | {b['cases']} | {b['groups']} | {b['catalogue_citing']} | {b['P']} | {b['P_spec']} | "
                 f"{b['P_spec_groups']} | {b['overlap_v11']} |")
    L += ["", f"Counts receipt: `NEW/counts.json` sha256 `{rep['counts_sha256']}`. The first look is round 2's 58 released "
              f"cases; {c['first_look_ids_not_released_now']} of them are not released now.", ""]
    hr = rep["headline_rule"]
    L += [f"Headline set (`HEADLINE_RULE.json` sha256 `{hr['sha256'][:16]}`, written {hr['utc']}, before any NEW case "
          f"was scored): {hr['rule']} Left out: {', '.join(f'`{q}`' for q in hr['overlap_cases'])}.", ""]
    for block, title in (("headline", "Headline: NEW without the v11-overlap cases (P_spec, component macro)"),
                         ("second_look", "Second look only (headline set, not in round 2's first look)"),
                         ("incl_overlap", "All NEW including the v11-overlap cases (sensitivity)")):
        L += [f"## {title}", "", "| System | n (groups) | " + " | ".join(MEASURES) + " |",
              "|---|---|" + "---|" * len(MEASURES)]
        for sid in ORDER:
            v = (rep["blocks"].get(sid) or {}).get(block)
            if not v:
                why = "not scored" if sid not in s else "no case in this block"
                L.append(f"| {LABELS[sid]} | {why} |" + " |" * len(MEASURES))
                continue
            L.append(f"| {LABELS[sid]} | {v['n_cases']} ({v['n_groups']}) | " +
                     " | ".join(_f(v.get(m)) for m in MEASURES.values()) + " |")
        L.append("")
    L += ["Receipts (summary.json sha256): " + "; ".join(f"{sid} `{v['_sha256'][:16]}`" for sid, v in s.items()), "",
          "## Contrasts (pre-registered in FREEZE.json, nDCG@10, paired over components)", "",
          "Δ = a − b. 95 % bootstrap interval; the adjusted interval is Bonferroni over the family; MDE is the smallest "
          "true gain detected with 80 % power (2.80 SD / √n).", "",
          "| Family | a − b | subset | n groups | Δ | 95 % CI | adjusted CI | class | MDE |", "|---|---|---|---|---|---|---|---|---|"]
    for key, v in rep["contrasts"].items():
        fam, ab = key.split("/", 1)
        if "missing" in v:
            L.append(f"| {fam} | {ab} | – | – | missing {v['missing']} | | | | |")
            continue
        for sub in ("all", "after_cutoff", "rest", "incl_overlap"):
            r = v.get(sub)
            if r is None:
                continue
            if "not_defined" in r or r.get("too_few"):
                L.append(f"| {fam} | {ab} | {sub} | – | {r.get('not_defined', 'too few groups')} | | | | |")
                continue
            lab = {"all": "headline", "incl_overlap": "all NEW incl. overlap"}.get(sub) or f"{sub} ({r['cutoff']})"
            L.append(f"| {fam} | {ab} | {lab} | {r['n']} | {_f(r['delta'], True)} | [{_f(r['ci95'][0], True)}, "
                     f"{_f(r['ci95'][1], True)}] | [{_f(r['ci_adjusted'][0], True)}, {_f(r['ci_adjusted'][1], True)}] | "
                     f"{r['class']} | {_f(r['mde_improvement'])} |")
    L += ["", "## LLM stage by the model's cutoff", "",
          "Every NEW case was published after both stated cutoffs. The court's decisions can be older, so a case counts "
          "as after the cutoff only if every precedent date in its `source_precedentes` is later (earliest date > "
          f"cutoff). Evaluated cases without a parsable date: {len(rep['undated_evaluated_cases'])} (counted with the rest).", "",
          "| System | cutoff | subset | n | system nDCG@10 | S_full_ext nDCG@10 |", "|---|---|---|---|---|---|"]
    for sid, v in rep["cutoff_split"].items():
        for sub in ("after", "rest"):
            a, b = v[f"system_{sub}"], v[f"base_{sub}"]
            L.append(f"| {LABELS[sid]} | {v['cutoff']} | {sub} | {v[f'n_{sub}']} | {_f((a or {}).get('ndcg_at_10'))} | "
                     f"{_f((b or {}).get('ndcg_at_10'))} |")
    L += ["", "## Spread (one fold refit, development)", "",
          "| System | refit nDCG@10 | record nDCG@10 | difference | top-10 identical |", "|---|---|---|---|---|"]
    for sid, v in (rep.get("spread_fold0") or {}).items():
        L.append(f"| {sid} | {_f(v['refit_ndcg_at_10'])} | {_f(v['record_ndcg_at_10'])} | {_f(v['difference'], True)} | "
                 f"{v['top10_identical_share']:.3f} |")
    dr = rep.get("drift") or {}
    L += ["", "## Drift on the controls (development cases)", ""]
    if "runb" in dr:
        L.append(f"- Run computer (runb_manifest.json sha256 `{dr['runb']['manifest_sha256'][:16]}`): gate pass "
                 f"{dr['runb']['gate'].get('pass')}; bge {json.dumps(dr['runb']['bge'])}; Qwen {json.dumps(dr['runb']['qwen'])}; "
                 f"ColBERT {json.dumps(dr['runb']['pairs'])}.")
    if "local" in dr:
        L.append(f"- This PC's encoders (local_manifest.json sha256 `{dr['local']['manifest_sha256'][:16]}`): "
                 f"{json.dumps(dr['local']['controls'])[:600]}.")
    L += ["", "## Costs", ""]
    for sid, v in rep["costs"].items():
        L.append(f"- {LABELS.get(sid, sid)}: " + json.dumps({k: x for k, x in v.items() if k != 'manifest_sha256'})
                 + f" (manifest sha256 `{v['manifest_sha256'][:16]}`)")
    col = rep["collection"]
    L += ["", "## Collection", "", f"- New lines in the copied store: {col.get('new_lines')}.",
          *[f"- `{r['file']}` (sha256 `{r['sha256'][:16]}`): {r['seconds']} s, result {json.dumps(r['result'])[:400]}"
            for r in col.get("runs", [])],
          "", "## Not scored on NEW (D162 scope)", ""]
    for k, v in (rep.get("not_scored") or {}).items():
        L.append(f"- {k}: {v}.")
    return "\n".join(L) + "\n"


def tex(run: Path, rep: dict) -> str:
    L = [r"% New-case evidence (E13; generated by uol.final.new_cases_report; do not edit).",
         f"% Generated {rep['utc']}; run {run.as_posix()}; FREEZE.json sha256 {rep['freeze_sha256'][:16]}.",
         r"\providecommand{\EvPending}[1]{\textcolor{red!65!black}{\textsf{[#1]}}}", r"\makeatletter",
         r"\providecommand{\EvNew}[2]{\@ifundefined{evnew@#1@#2}{\EvPending{NEW #1 #2}}{\@nameuse{evnew@#1@#2}}}",
         r"\providecommand{\EvNewCon}[3]{\@ifundefined{evnewcon@#1@#2@#3}{\EvPending{NEW #1 #2 #3}}{\@nameuse{evnewcon@#1@#2@#3}}}",
         r"\providecommand{\EvNewCount}[2]{\@ifundefined{evnewn@#1@#2}{\EvPending{NEW #1 #2}}{\@nameuse{evnewn@#1@#2}}}", ""]
    c = rep["counts"]
    L.append(f"% counts: NEW/counts.json sha256 {rep['counts_sha256'][:16]}")
    for blk in ("all", "first_look", "second_look"):
        for k in ("cases", "groups", "catalogue_citing", "P_spec", "P_spec_groups"):
            L.append(rf"\@namedef{{evnewn@{blk}@{k}}}{{{c[blk][k]:,}}}")
    if c.get("publication_dates"):
        L.append(rf"\@namedef{{evnewn@dates@first}}{{{c['publication_dates'][0]}}}")
        L.append(rf"\@namedef{{evnewn@dates@last}}{{{c['publication_dates'][1]}}}")
    L.append(rf"\@namedef{{evnewn@headline@overlap_left_out}}{{{len(rep['headline_rule']['overlap_cases'])}}}")
    for sid, v in rep["cutoff_split"].items():
        L.append(rf"\@namedef{{evnewn@{sid}@after_cutoff}}{{{v['n_after']}}}")
        L.append(rf"\@namedef{{evnewn@{sid}@rest}}{{{v['n_rest']}}}")
    for sid, v in rep["summaries"].items():
        L.append(f"% {sid}: eval/{sid}/summary.json sha256 {v['_sha256'][:16]}; {v['all']['n_cases']} cases, "
                 f"{v['all']['n_groups']} groups")
        b = rep["blocks"][sid]
        for blk, suf in (("headline", ""), ("second_look", "@second"), ("incl_overlap", "@incloverlap")):
            if not b.get(blk):
                continue
            for name, col in MEASURES.items():
                x = b[blk].get(col)
                if x is not None and np.isfinite(x):
                    L.append(rf"\@namedef{{evnew@{sid}{suf}@{name}}}{{{_tex(float(x))}}}")
    for sid, v in rep["cutoff_split"].items():
        for sub in ("after", "rest"):
            for who, key in ((sid, f"system_{sub}"), ("S_full_ext", f"base_{sub}")):
                x = (v[key] or {}).get("ndcg_at_10")
                if x is not None and np.isfinite(x):
                    L.append(rf"\@namedef{{evnew@{who}@{sub}@{sid}@nDCG@10}}{{{_tex(float(x))}}}")
    L.append("% contrasts: report.json (round3.stats.contrast, pre-registered families in FREEZE.json)")
    for key, v in rep["contrasts"].items():
        fam, ab = key.split("/", 1)
        for sub in ("all", "after_cutoff", "rest", "incl_overlap"):
            r = v.get(sub)
            if not r or "delta" not in r:
                continue
            k = f"{fam}{'' if sub == 'all' else '@' + sub}@{ab}"
            L += [rf"\@namedef{{evnewcon@{k}@delta}}{{{_tex(r['delta'], True)}}}",
                  rf"\@namedef{{evnewcon@{k}@ci}}{{[{_tex(r['ci95'][0], True)}, {_tex(r['ci95'][1], True)}]}}",
                  rf"\@namedef{{evnewcon@{k}@ciadj}}{{[{_tex(r['ci_adjusted'][0], True)}, {_tex(r['ci_adjusted'][1], True)}]}}",
                  rf"\@namedef{{evnewcon@{k}@class}}{{{r['class'].replace('_', ' ')}}}",
                  rf"\@namedef{{evnewcon@{k}@mde}}{{{_tex(r['mde_improvement'])}}}",
                  rf"\@namedef{{evnewcon@{k}@n}}{{{r['n']}}}"]
    L += [r"\makeatother", ""]
    return "\n".join(L)


def evidence_rows(run: Path, rep: dict) -> str:
    rel = run.relative_to(ROOT).as_posix() if ROOT in run.parents else run.as_posix()
    c, s = rep["counts"], rep["summaries"]
    L = [SECTION, "",
         f"Collected and scored after the freeze (FREEZE.json sha256 {rep['freeze_sha256'][:16]}…), each system once, "
         f"P_spec, component macro; contrasts pre-registered in FREEZE.json. Report: {rel}/RESULTS.md; values in "
         f"{rel}/report.json.", "", "| Claim | Number | Label | Source |", "|---|---|---|---|",
         f"| NEW cases / groups / catalogue-citing; P_spec cases (groups) | {c['all']['cases']} / {c['all']['groups']} / "
         f"{c['all']['catalogue_citing']}; {c['all']['P_spec']} ({c['all']['P_spec_groups']}) | NEW, descriptive | "
         f"{rel}/NEW/counts.json |",
         f"| Of which second look (not in round 2's first look): P_spec cases (groups) | {c['second_look']['P_spec']} "
         f"({c['second_look']['P_spec_groups']}) | NEW, descriptive | same |"]
    L.append(f"| Left out of the headline (case group shared with v11 lockbox rows; HEADLINE_RULE.json, before scoring) | "
             f"{len(rep['headline_rule']['overlap_cases'])} cases | NEW, rule | {rel}/HEADLINE_RULE.json (sha256 "
             f"{rep['headline_rule']['sha256'][:16]}) |")
    for sid in ORDER:
        v = (rep["blocks"].get(sid) or {}).get("headline")
        if v:
            L.append(f"| {LABELS[sid]}: nDCG@10 / s@10 / law@10 on NEW (headline) | {_f(v['ndcg_at_10'])} / "
                     f"{_f(v['success_at_10'])} / {_f(v['law_at_10'])} ({v['n_cases']} cases, {v['n_groups']} groups) | "
                     f"NEW, held out | {rel}/eval/{sid}/summary.json (sha256 {s[sid]['_sha256'][:16]}) |")
    for key, v in rep["contrasts"].items():
        r = v.get("all")
        if r and "delta" in r:
            extra = ""
            a = v.get("after_cutoff") or {}
            if "delta" in a:
                extra = (f"; decided after {a['cutoff']}: {_f(a['delta'], True)} [{_f(a['ci95'][0], True)}, "
                         f"{_f(a['ci95'][1], True)}] ({a['n']} groups)")
            L.append(f"| Contrast {key} (nDCG@10) | {_f(r['delta'], True)} [{_f(r['ci95'][0], True)}, "
                     f"{_f(r['ci95'][1], True)}], adjusted [{_f(r['ci_adjusted'][0], True)}, "
                     f"{_f(r['ci_adjusted'][1], True)}], {r['class']}, MDE {_f(r['mde_improvement'])}{extra} | "
                     f"NEW, held out | {rel}/report.json `contrasts` |")
    for sid, v in rep["costs"].items():
        L.append(f"| {LABELS.get(sid, sid)}: tokens and spend on NEW | {json.dumps(v.get('usage') or v.get('input_tokens'))}"
                 f"{'; US$' + str(v.get('usd') or v.get('usd_at_configured_price')) if (v.get('usd') or v.get('usd_at_configured_price')) is not None else ''} | "
                 f"NEW, cost | {rel}/scores/{sid}/manifest.json |")
    return "\n".join(L) + "\n"


def append_evidence(run: Path) -> bool:
    """Append EVIDENCE_ROWS.md to EVIDENCE.md once (refuses when the section is already there)."""
    text = EVIDENCE.read_text(encoding="utf-8")
    if SECTION in text:
        return False
    rows = (Path(run) / "EVIDENCE_ROWS.md").read_text(encoding="utf-8")
    with EVIDENCE.open("a", encoding="utf-8") as f:
        f.write("\n" + rows)
    return True
