"""The thesis pack (RUN_PLAN v2 §9): everything needed to write the results chapters, generated from the store only.

    <store>/pack/
      00_READ_ME_FIRST.md        what is here, what passed, what failed, run facts
      G1.md                      the reproduction tables (G1 and G1-rebuild)
      ch4_data/                  populations, label counts, generic articles, law distribution, DEV-B / DEV-N changes
      ch5_methods/               every system's configuration and chosen inner-fold settings, training times,
                                 model versions, the LLM prompts used
      ch6_results/E01_alone ...  per experiment: RESULTS.md, tables (CSV + LaTeX), figures (PDF + PNG, >= 8 pt)
      ch6_results/frontier/      quality against cost per 1,000 cases (the headline figure and table)
      ch6_results/robustness/    DEV-G, DEV-ALL, DEV-T, DEV-B, DEV-N, seeds, law level
      examples/                  worked cases (2028281 and five more): every system's rank of each cited article
      evidence/EVIDENCE_FINAL.md one row per number the thesis may use, with its source file
      appendix_full_results/     every configuration evaluated, all measures, all populations
      MANIFEST_RESULTS.json      sha256 of every file

Raw files stay in the store (``runs/``, ``receipts/``, ``cost/``, ``contrasts/``) and are mirrored as ``raw/`` on Drive.
Figures use ``uol.thesis_figures.common.style()`` when it is importable (Latin Modern Sans, 8 pt base, Type 42),
otherwise matplotlib with an 8 pt floor. Every file is written atomically.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from uol.final import context, registry

EXPERIMENTS = {
    "E01_alone": ("How good is each method alone?", ["A", "E1"]),
    "E02_size_legal": ("Do size, training and legal domain matter?", ["E2"]),
    "E03_compression": ("How small can a vector be?", ["E3"]),
    "E04_hybrids": ("Does an untrained hybrid get the gain?", ["B"]),
    "E05_ladder": ("What does each family add inside the ranker?", ["C", "E5"]),
    "E06_learner": ("Is it the learner or the features?", ["D"]),
    "E07_bands": ("Where does the gain come from (never, rare, frequent)?", []),
    "E08_expansion": ("Can query expansion replace or add to embeddings?", ["E8"]),
    "E09_rerank": ("What does reranking add, and at what price?", ["E9"]),
    "E10_cost": ("What does each system cost?", []),
    "E11_robustness": ("Does the answer hold?", ["G"]),
    "E12_agent": ("Agents (optional)", ["E12"]),
    "E13_new": ("Do the frozen answers hold on new cases? (post-freeze plan)", []),
}
COLS = ["ndcg_at_10", "success_at_1", "success_at_10", "recall_at_20", "recall_at_100", "unseen_recall_at_20_pooled",
        "law_at_1", "law_at_10", "success_at_1_in_law", "n_cases", "n_groups"]
EXAMPLES = ("sjf:tesis:2028281:hechos",)


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def summaries(store: Path):
    import pandas as pd
    rows = []
    for p in sorted(Path(store).glob("runs/*/*/summary.json")):
        s = json.loads(p.read_text(encoding="utf-8"))
        rows.append({"system": s["system"], "population": s["population"], **{c: s.get(c) for c in COLS},
                     "partial": s.get("partial", False), "source": p.relative_to(store).as_posix()})
    return pd.DataFrame(rows)


def _base(name: str) -> str:
    return name.split("@", 1)[0]


def _tables(df, out: Path, stem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv.tmp", index=False)
    os.replace(out / f"{stem}.csv.tmp", out / f"{stem}.csv")
    try:
        tex = df.to_latex(index=False, float_format=lambda x: f"{x:.3f}", escape=True)
    except Exception:                                   # pragma: no cover - jinja2 missing
        tex = "% LaTeX export unavailable (pandas Styler needs jinja2)\n"
    _atomic(out / f"{stem}.tex", tex)


def _figure(df, out: Path, stem: str, title: str) -> None:
    if df.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        from uol.thesis_figures import common as TF
        TF.style()
        width = TF.TEXTWIDTH_IN
    except Exception:
        plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 8,
                             "ytick.labelsize": 8, "pdf.fonttype": 42})
        width = 4.9
    for k in ("font.size", "xtick.labelsize", "ytick.labelsize", "legend.fontsize", "axes.labelsize"):
        try:
            if float(plt.rcParams[k]) < 8:
                plt.rcParams[k] = 8
        except (TypeError, ValueError):
            plt.rcParams[k] = 8
    d = df.sort_values("ndcg_at_10").tail(40)
    fig, ax = plt.subplots(figsize=(width, max(1.6, 0.16 * len(d) + 0.6)))
    ax.barh(range(len(d)), d["ndcg_at_10"], color="#3b6ea5")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["system"])
    ax.set_xlabel("nDCG@10 (DEV)")
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        tmp = out / f"{stem}.tmp.{ext}"
        fig.savefig(tmp, dpi=300)
        os.replace(tmp, out / f"{stem}.{ext}")
    plt.close(fig)


def _row(claim: str, measures: list[tuple[str, Any]], scope: str, source: str) -> str | None:
    """One EVIDENCE_FINAL row ``subject: m1; m2 | v1; v2``; measures without a value are left out."""
    have = [(m, v) for m, v in measures if v is not None]
    if not have:
        return None
    val = lambda v: f"{v:.6g}" if isinstance(v, float) else str(v)
    return (f"| {claim}: {'; '.join(m for m, _ in have)} | {'; '.join(val(v) for _, v in have)} | {scope} | "
            f"{source} |")


def evidence_lines(store: Path, df, pop_rows: list[dict[str, Any]]) -> list[str]:
    """EVIDENCE_FINAL.md: one row per number the thesis may use, each with its source file (read by
    scripts/evidence_macros.py). Measures of one subject share a row (``S0 on DEV: nDCG@10; s@10 | .408; .612``).
    Rows: system results; population sizes; ranker pool and feature columns (receipts); cost quantities; contrasts."""
    store = Path(store)
    ev = ["# EVIDENCE_FINAL (generated)", "", "| Claim | Value | Scope | Source |", "|---|---|---|---|"]
    for _, r in (df[~df["partial"]] if not df.empty else df).iterrows():
        ev.append(f"| {r['system']} on {r['population']}: nDCG@10; s@10; never-cited R@20; law@10 | {r['ndcg_at_10']:.3f}; "
                  f"{r['success_at_10']:.3f}; {r['unseen_recall_at_20_pooled']:.3f}; {r['law_at_10']:.3f} | "
                  f"{r['population']}, descriptive | raw/{r['source']} |")
    out: list[str | None] = []
    for p in pop_rows:                                           # population sizes (evaluated cases and groups)
        out.append(_row(f"population {p['population']}", [("cases", p.get("cases")), ("groups", p.get("groups"))],
                        "population definition", "ch4_data/populations.csv"))
    for p in sorted((store / "receipts").glob("*.json")) if (store / "receipts").exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        fits = [(f.get("fit") or {}) for f in r.get("folds", [])]
        if not fits or not all("feature_columns" in f for f in fits):
            continue                                             # learned rankers only (adapters/ltr.py read-outs)
        cols = sorted({int(f["feature_columns"]) for f in fits})
        pool = sum(float(f["pool_pairs_per_case"]) for f in fits) / len(fits)
        src = f"raw/receipts/{p.name}"
        out.append(_row(f"{r['system']} on {r['population']}", [("mean candidate pool per case", round(pool, 1))],
                        f"mean over {len(fits)} folds", src))
        out.append(_row(r["system"], [("feature columns", cols[-1])],
                        "same in every fold" if len(cols) == 1 else f"varies by fold: {cols[0]} to {cols[-1]}", src))
    for p in sorted((store / "cost").glob("*.json")) if (store / "cost").exists() else []:
        c = json.loads(p.read_text(encoding="utf-8"))
        pc = (c.get("llm") or {}).get("per_case") or {}
        out.append(_row(f"{c['system']} on DEV", [
            ("query latency p95 ms", (c.get("query_latency") or {}).get("p95_ms")),
            ("peak RAM GB", (c.get("memory") or {}).get("peak_ram_gb")),
            ("index size MB", c.get("index_size_mb")),
            ("query encoding s per case", (c.get("query_encoding") or {}).get("seconds_per_case")),
            ("GPU hours indexing", (c.get("indexing") or {}).get("gpu_hours_5080")),
            ("GPU hours fine-tuning", (c.get("training") or {}).get("gpu_hours_finetune")),
            ("LLM calls per case", pc.get("calls")), ("LLM input tokens per case", pc.get("input_tokens")),
            ("LLM output tokens per case", pc.get("output_tokens")), ("LLM seconds per case", pc.get("seconds")),
            ("USD per 1,000 cases", (c.get("cost_per_1000_cases") or {}).get("usd"))],
            "DEV, recorded quantities (USD only at a cited, dated price, D80)", f"raw/cost/{p.name}"))
    cdir = store / "contrasts"
    for p in sorted(cdir.glob("*.json")) if cdir.exists() else []:
        c = json.loads(p.read_text(encoding="utf-8"))
        if "contrasts" in c:                                     # a family (F1 ...): Holm over its contrasts
            items = [(f"contrast {k} on DEV ({c['family']})", v, f"family {c['family']} of {c['size']}, Holm")
                     for k, v in c["contrasts"].items()]
        elif "delta" in c:                                       # one contrast (uol.final contrast)
            items = [(f"contrast {c['a']} - {c['b']} on {c['population']}", c, "single contrast, no family correction")]
        else:
            continue
        for claim, v, scope in items:
            ci = v.get("ci95") or [None, None]
            out.append(_row(claim, [("delta", v.get("delta")), ("CI95 low", ci[0]), ("CI95 high", ci[1]),
                                    ("adjusted lower bound", v.get("adjusted_lower")), ("p Holm", v.get("p_holm")),
                                    ("groups", v.get("n"))],
                            f"{scope}; class {v.get('class')}", f"raw/contrasts/{p.name}"))
    return ev + [x for x in out if x]


def write(store: Path | None = None) -> Path:
    import pandas as pd
    prof = context.active()
    store = Path(store) if store else prof.store
    pack = store / "pack"
    df = summaries(store)
    S = registry.systems()
    status = {}
    for p in sorted((store / "status").glob("*.json")) if (store / "status").exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        status[(r["system"], r["population"])] = r
    # G1
    g = []
    for name in ("G1.md", "G1_rebuild.md"):
        p = store / "g1" / name
        if p.exists():
            g.append(p.read_text(encoding="utf-8"))
    _atomic(pack / "G1.md", "\n\n".join(g) or "G1 has not run.\n")
    # ch4 data
    from uol.embedding_push import common as ep
    from uol.final import populations
    pop_rows = []
    for pname in populations.SPECS:
        try:
            pop = populations.load(pname, store)
            j = pop.to_json()
            pop_rows.append({"population": pname, "cases": j["n_cases"], "groups": j["n_groups"], "evaluator": pop.eval_population})
        except Exception as exc:
            pop_rows.append({"population": pname, "cases": None, "groups": None, "evaluator": None, "note": str(exc)[:120]})
    _tables(pd.DataFrame(pop_rows), pack / "ch4_data", "populations")
    d = ep.release()
    _tables(pd.DataFrame([{"candidate_id": d.candidate_ids[i]} for i in d.generic_idx]), pack / "ch4_data", "generic_articles")
    dev = ep.dev_rows()
    _tables(pd.DataFrame([{"labels_per_case_mean": float(sum(len(d.positive_idx[int(r)]) for r in dev) / max(len(dev), 1)),
                           "dev_cases": len(dev), "label_corrections": json.dumps(context.LABEL_CORRECTIONS)}]),
            pack / "ch4_data", "labels")
    # ch5 methods: configuration and chosen settings from the receipts
    meth = []
    for p in sorted((store / "receipts").glob("*.json")) if (store / "receipts").exists() else []:
        r = json.loads(p.read_text(encoding="utf-8"))
        spec = r.get("spec", {})
        meth.append({"result": r["system"], "population": r["population"], "family": spec.get("family"),
                     "adapter": spec.get("adapter"), "inputs": spec.get("inputs"), "training": spec.get("training_data"),
                     "grid": json.dumps(spec.get("grid")), "chosen": json.dumps([f.get("fit", {}).get("params") for f in r.get("folds", [])], default=str)[:500],
                     "fit_seconds": r.get("timing", {}).get("fit_seconds"), "peak_ram_gb": r.get("peak_ram_gb")})
    _tables(pd.DataFrame(meth), pack / "ch5_methods", "systems_and_settings")
    prompts = Path(__file__).resolve().parents[1] / "round3" / "prompts"
    for src in list(prompts.glob("*.txt")) + list((Path(__file__).resolve().parent / "prompts").glob("*.txt")):
        dst = pack / "ch5_methods" / "prompts" / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # ch6: per experiment
    dev_df = df[(df["population"] == "DEV")] if not df.empty else df
    for exp, (question, blocks) in EXPERIMENTS.items():
        out = pack / "ch6_results" / exp
        ids = {sid for sid, s in S.items() if set(blocks) & set(s.get("blocks", []))}
        sub = dev_df[dev_df["system"].map(_base).isin(ids)] if not dev_df.empty else dev_df
        if exp == "E11_robustness" and not df.empty:
            sub = df[df["population"] != "DEV"]
        extra_rows = None
        if exp == "E05_ladder" and sub is not None and len(sub):      # D78: the ladder is the dense-only definition
            ws = {sid for sid in ids if S[sid].get("ladder_definition") == "with_sources"}
            extra_rows = sub[sub["system"].map(_base).isin(ws)]
            sub = sub[~sub["system"].map(_base).isin(ws)]
        lines = [f"# {exp}: {question}", "", "Development data only; component-macro means over groups.", ""]
        if sub is not None and len(sub):
            _tables(sub.drop(columns=["source"]), out, "table")
            _figure(sub, out, "figure", question)
            lines += [f"{len(sub)} results; table.csv / table.tex; figure.pdf / figure.png.", ""]
            if extra_rows is not None and len(extra_rows):
                _tables(extra_rows.drop(columns=["source"]), out, "table_with_sources")
                lines += ["The ladder is S0 + the encoder's dense column for every encoder (D78). The frozen encoders "
                          "with their past-case sources are extra rows: table_with_sources.csv / .tex.", ""]
            for _, r in sub.sort_values("ndcg_at_10", ascending=False).head(25).iterrows():
                lines.append(f"- {r['system']} [{r['population']}]: nDCG@10 {r['ndcg_at_10']:.4f} "
                             f"(s@10 {r['success_at_10']:.3f}, law@10 {r['law_at_10']:.3f}) - {r['source']}")
        missing = [f"- {k[0]} [{k[1]}]: {v.get('status')} {str(v.get('reason') or v.get('error') or '')[:150]}"
                   for k, v in status.items() if _base(k[0]) in ids and v.get("status") != "done"]
        if missing:
            lines += ["", "Not scored:", ""] + missing
        cdir = store / "contrasts"
        fams = {"E05_ladder": ("F1", "F2", "F7", "F8"), "E02_size_legal": ("F3",), "E04_hybrids": ("F4",), "E06_learner": ("F5",),
                "E09_rerank": ("F6", "F6x")}.get(exp, ())
        for f in fams:
            p = cdir / f"{f}.json"
            if p.exists():
                c = json.loads(p.read_text(encoding="utf-8"))
                lines += ["", f"Family {f} (paired group bootstrap, 10,000 draws, Holm):", ""]
                for name, r in c["contrasts"].items():
                    lines.append(f"- {name}: {r['delta']:+.4f} [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}], {r['class']}, "
                                 f"p_holm {r.get('p_holm', float('nan')):.4f}")
        _atomic(out / "RESULTS.md", "\n".join(lines) + "\n")
    # frontier
    cost = []
    for p in sorted((store / "cost").glob("*.json")) if (store / "cost").exists() else []:
        c = json.loads(p.read_text(encoding="utf-8"))
        q = df[(df["system"].map(_base) == c["system"]) & (df["population"] == "DEV")] if not df.empty else df
        llm = c.get("llm") or {}
        pc = llm.get("per_case") or {}
        cost.append({"system": c["system"], "ndcg_at_10": float(q["ndcg_at_10"].max()) if len(q) else None,
                     "latency_p95_ms": (c.get("query_latency") or {}).get("p95_ms"),
                     "peak_ram_gb": c["memory"].get("peak_ram_gb"), "index_size_mb": c.get("index_size_mb"),
                     "index_size_basis": (c.get("index_size") or {}).get("basis"),
                     "query_encode_s_per_case": (c.get("query_encoding") or {}).get("seconds_per_case"),
                     "gpu_hours_indexing": (c.get("indexing") or {}).get("gpu_hours_5080"),
                     "gpu_hours_finetune": (c.get("training") or {}).get("gpu_hours_finetune"),
                     "llm_calls_per_case": pc.get("calls"), "llm_input_tokens_per_case": pc.get("input_tokens"),
                     "llm_cached_input_tokens_per_case": pc.get("cached_input_tokens"),
                     "llm_output_tokens_per_case": pc.get("output_tokens"), "llm_seconds_per_case": pc.get("seconds"),
                     "llm_basis": llm.get("basis"), "llm_complete": llm.get("complete"),
                     "usd_per_1000": c["cost_per_1000_cases"].get("usd")})
    _tables(pd.DataFrame(cost), pack / "ch6_results" / "frontier", "frontier")
    # examples
    ex = []
    for qid in EXAMPLES:
        for p in sorted(store.glob("runs/*/DEV/per_case.parquet")):
            f = pd.read_parquet(p)
            hit = f[f["query_id"] == qid]
            if len(hit):
                ex.append({"query_id": qid, "system": p.parent.parent.name, "positive_idx": hit.iloc[0].get("positive_idx"),
                           "positive_ranks": hit.iloc[0].get("positive_ranks"), "ndcg_at_10": hit.iloc[0]["ndcg_at_10"]})
    _tables(pd.DataFrame(ex), pack / "examples", "worked_cases")
    # evidence and appendix
    _atomic(pack / "evidence" / "EVIDENCE_FINAL.md", "\n".join(evidence_lines(store, df, pop_rows)) + "\n")
    _tables(df, pack / "appendix_full_results", "all_results")
    done = sum(1 for v in status.values() if v.get("status") == "done")
    bad = [k for k, v in status.items() if v.get("status") in ("failed",)]
    _atomic(pack / "00_READ_ME_FIRST.md", "\n".join([
        "# Final run: read me first", "", f"Results: {len(df)} summaries, {done} done runs, {len(bad)} failed runs.",
        "G1 and G1-rebuild: see G1.md. Per-experiment RESULTS.md under ch6_results/. EVIDENCE_FINAL.md lists every number",
        "with its source file. Raw per-case files and rankings are under raw/ (mirror of the store).", "",
        "Failed runs:", ""] + [f"- {s} [{p}]" for s, p in bad]) + "\n")
    files = {}
    for p in sorted(pack.rglob("*")):
        if p.is_file() and p.name != "MANIFEST_RESULTS.json" and ".tmp" not in p.name:
            files[p.relative_to(pack).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    _atomic(pack / "MANIFEST_RESULTS.json", json.dumps({"files": files}, indent=1))
    return pack
