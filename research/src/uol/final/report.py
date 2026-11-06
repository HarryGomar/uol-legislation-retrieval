"""RESULTS.md, EVIDENCE-style rows and the results manifest, generated from the results store only.

    py -3.12 -m uol.final report [--store DIR] [--out DIR]

Tables: G1; Blocks A-D and G (every system that has a summary, with its status otherwise, so the selection is
visible); contrasts (``<store>/contrasts/*.json``); cost (``<store>/cost/*.json``). EVIDENCE rows give the value,
the population and the source file, in the format of research/thesis_direction_20261015/EVIDENCE.md. The manifest
``MANIFEST_RESULTS.json`` lists the sha256 of every file in the store.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from uol.final import context, registry

COLS = [("ndcg_at_10", "nDCG@10"), ("success_at_1", "s@1"), ("success_at_10", "s@10"), ("recall_at_20", "R@20"),
        ("recall_at_100", "R@100"), ("unseen_recall_at_20_pooled", "never-cited R@20"), ("law_at_1", "law@1"),
        ("law_at_10", "law@10"), ("success_at_1_in_law", "s@1 in law")]
BLOCKS = {"A": "Alone", "B": "Cheap hybrids (L* with headings + X)", "C": "Ranker ladder", "D": "Combination method",
          "E": "LLM", "G": "Robustness and sensitivity"}


def _f(x: Any, nd: int = 4) -> str:
    try:
        if x is None or x != x:
            return "-"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def collect(store: Path) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for p in sorted((store / "status").glob("*.json")) if (store / "status").exists() else []:
        rec = json.loads(p.read_text(encoding="utf-8"))
        out[(rec["system"], rec["population"])] = rec
    for p in sorted((store / "runs").glob("*/*/summary.json")) if (store / "runs").exists() else []:
        s = json.loads(p.read_text(encoding="utf-8"))
        out[(s["system"], s["population"])] = {"status": "done", "summary": s, "path": str(p.relative_to(store))}
    return out


def table(rows: list[tuple[str, dict[str, Any]]]) -> list[str]:
    head = "| System | Status | " + " | ".join(c for _, c in COLS) + " | Cases / groups |"
    lines = [head, "|" + "---|" * (len(COLS) + 3)]
    for sid, rec in rows:
        s = rec.get("summary")
        if s:
            vals = " | ".join(_f(s.get(k), 4 if k == "ndcg_at_10" else 3) for k, _ in COLS)
            part = " (partial)" if s.get("partial") else ""
            lines.append(f"| {sid} | done{part} | {vals} | {s.get('n_cases')} / {s.get('n_groups')} |")
        else:
            why = (rec.get("reason") or rec.get("error") or "").replace("|", "/")[:90]
            lines.append(f"| {sid} | {rec.get('status', 'not run')}: {why} |" + " |" * (len(COLS) + 1))
    return lines


def write(store: Path | None = None, out: Path | None = None) -> Path:
    prof = context.active()
    store = Path(store) if store else prof.store
    out = Path(out) if out else store
    res = collect(store)
    try:
        systems = registry.systems()
    except FileNotFoundError:
        systems = {}
    L = ["# Final benchmark results (generated)", "",
         f"Generated from `{store}` by `uol.final.report`. Development data only; the test set is closed. "
         "Values are component-macro means over groups.", ""]
    g1 = store / "g1" / "G1.json"
    if g1.exists():
        G = json.loads(g1.read_text(encoding="utf-8"))
        L += ["## Gate G1 (frozen DEV values, tolerance 1e-4)", "", "| Check | Route | Expected | Obtained | |diff| | Pass |",
              "|---|---|---|---|---|---|"]
        for r in G["rows"]:
            L.append(f"| {r['system']} | {r['route']} | {_f(r['expected'], 6)} | {_f(r.get('obtained'), 6)} | "
                     f"{_f(r.get('abs_diff'), 6)} | {r.get('pass')} |")
        L += ["", f"G1 overall: **{'PASS' if G.get('pass') else 'FAIL'}** ({G.get('note', '')})", ""]
    pops = sorted({p for _, p in res})
    for b, title in BLOCKS.items():
        ids = [s for s, spec in systems.items() if b in spec.get("blocks", [])]
        if not ids:
            continue
        pop = "DEV"
        rows = [(sid, res.get((sid, pop), {"status": "not run"})) for sid in ids]
        if b == "G":
            rows = [(f"{sid} [{p}]", res[(sid, p)]) for (sid, p) in sorted(res) if p != "DEV" or sid in ids]
        L += [f"## Block {b}. {title} ({'all populations' if b == 'G' else 'DEV'})", ""] + table(rows) + [""]
    if pops:
        L += ["## Every population", ""]
        for p in pops:
            rows = [(s, r) for (s, pp), r in sorted(res.items()) if pp == p and r.get("summary")]
            if rows:
                L += [f"### {p}", ""] + table(rows) + [""]
    cdir = store / "contrasts"
    if cdir.exists() and any(cdir.glob("*.json")):
        L += ["## Contrasts (paired group bootstrap, 10,000 draws, seed 20261004)", "",
              "| Contrast | Population | Delta | 95% CI | Adjusted lower | Class |", "|---|---|---|---|---|---|"]
        for p in sorted(cdir.glob("*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            for name, r in (c.get("contrasts") or {p.stem: c}).items():
                L.append(f"| {name} | {r.get('population', '')} | {_f(r.get('delta'))} | "
                         f"[{_f(r['ci95'][0])}, {_f(r['ci95'][1])}] | {_f(r.get('adjusted_lower'))} | {r.get('class')} |")
        L.append("")
    kdir = store / "cost"
    if kdir.exists() and any(kdir.glob("*.json")):
        L += ["## Cost", "", "| System | Hardware | Training CPU-min/fold | Latency median / p95 ms | Peak RAM GB | GPU-h | "
              "Fine-tune GPU-h | Index MB | Query encode s/case | LLM calls / case | LLM tokens in / cached / out per case | "
              "LLM s/case | USD / 1,000 |", "|" + "---|" * 13]
        for p in sorted(kdir.glob("*.json")):
            c = json.loads(p.read_text(encoding="utf-8"))
            lat = c.get("query_latency") or {}
            qe = c.get("query_encoding") or {}
            llm = c.get("llm") or {}
            pc = llm.get("per_case") or {}
            part = "" if llm.get("complete", True) else " (partial)"
            L.append(f"| {c['system']} | {c.get('hardware')} | {_f(c['training'].get('cpu_minutes_per_fold'), 2)} | "
                     f"{_f(lat.get('median_ms'), 1)} / {_f(lat.get('p95_ms'), 1)} | {_f(c['memory'].get('peak_ram_gb'), 2)} | "
                     f"{_f(c['indexing'].get('gpu_hours_5080'), 3)} | {_f(c['training'].get('gpu_hours_finetune'), 3)} | "
                     f"{_f(c.get('index_size_mb'), 1)} | {_f(qe.get('seconds_per_case'), 4)} | {_f(pc.get('calls'), 3)}{part} | "
                     f"{_f(pc.get('input_tokens'), 0)} / {_f(pc.get('cached_input_tokens'), 0)} / {_f(pc.get('output_tokens'), 0)} | "
                     f"{_f(pc.get('seconds'), 2)} | {_f(c['cost_per_1000_cases'].get('usd'), 2)} |")
        L += ["", "Index MB: measured files, in-memory matrices or an analytic size (each record's index_size.basis). "
              "Query encode: query-only jobs only. LLM: per DEV case from the LLM cache (llm.basis; partial = only calls "
              "made under the system's own tag).", ""]
    L += ["## EVIDENCE rows (to paste into EVIDENCE.md after review)", "",
          "| Claim | Value | Scope | Source |", "|---|---|---|---|"]
    for (sid, p), rec in sorted(res.items()):
        s = rec.get("summary")
        if s and not s.get("partial"):
            L.append(f"| {sid} on {p}: nDCG@10; s@10; never-cited R@20 | {_f(s.get('ndcg_at_10'), 3)}; "
                     f"{_f(s.get('success_at_10'), 3)}; {_f(s.get('unseen_recall_at_20_pooled'), 3)} | {p}, descriptive | "
                     f"experiments/runs/final_20261021/{rec.get('path', '')} |")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "RESULTS.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(L) + "\n", encoding="utf-8")
    tmp.replace(path)
    manifest(store, out / "MANIFEST_RESULTS.json")
    return path


def manifest(store: Path, path: Path) -> dict[str, Any]:
    files = {}
    for p in sorted(Path(store).rglob("*")):
        if p.is_file() and not p.name.endswith(".tmp") and p.resolve() != Path(path).resolve():
            files[p.relative_to(store).as_posix()] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                                      "bytes": p.stat().st_size}
    rec = {"store": str(store), "files": files, "n_files": len(files)}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    tmp.replace(path)
    return rec
