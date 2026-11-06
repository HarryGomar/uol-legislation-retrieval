"""Generate thesis/evidence/numbers.tex from the final run's EVIDENCE_FINAL.md (HANDOFF-2026-10-22 R8).

    py -3.12 scripts/evidence_macros.py --evidence <results>/evidence/EVIDENCE_FINAL.md [--freeze FREEZE.json]
        [--role recommended=S_full_ext] [--out thesis/evidence/numbers.tex] [--strict] [--dry-run]

EVIDENCE_FINAL.md is the run pack's table ``| Claim | Value | Scope | Source |`` (uol.final.pack). A claim may carry
several measures, ``S0 on DEV: nDCG@10; s@10; never-cited R@20; law@10``, with as many values ``0.408; ...``; each
becomes its own key ``S0 on DEV: nDCG@10``. Every macro below names one key and a format. A key the evidence lacks
prints ``\\EvPending{label}`` as before; ``--strict`` fails instead. The 15 Oct EVIDENCE.md is history only and is
refused (its header is not ``# EVIDENCE_FINAL``).

Systems that the author chooses at the freeze (R6) are roles: ``{recommended}`` and ``{llm_stage}`` come from FREEZE.json
(keys of the same names) or ``--role``; without them their macros stay pending. Macros that a chapter's own number file
defines (``numbers_ch3.tex`` ...) are overridden with ``\\renewcommand`` after those files are read, and only when the
evidence has the value, so a writer's placeholder stays until the run supplies the number.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "thesis" / "evidence" / "numbers.tex"
ROLES = {"lexical": "lstar_headings", "no_emb": "S0", "population": "DEV"}

# Fixed counts (D68 Q5: no new instruments); they do not come from the run.
FIXED = [("Catalogue and source counts (fixed; D68 Q5: no new instruments).",
          [("nCatArticles", "39,178"), ("nCatLaws", "264"), ("nSJFRecords", "311,965")])]

# (macro, evidence key, format, placeholder label). Keys may use {role} fields.
RESULTS = [
    ("ndcgLexical", "{lexical} on {population}: nDCG@10", "dot3", "lexical nDCG@10"),
    ("costLexical", "{lexical} on {population}: USD per 1,000 cases", "usd", "cost"),
    ("ndcgRankerNoEmb", "{no_emb} on {population}: nDCG@10", "dot3", "ranker without embeddings"),
    ("ndcgRecommended", "{recommended} on {population}: nDCG@10", "dot3", "ranker nDCG@10"),
    ("costRecommended", "{recommended} on {population}: USD per 1,000 cases", "usd", "cost"),
    ("ndcgLLMStage", "{llm_stage} on {population}: nDCG@10", "dot3", "LLM stage nDCG@10"),
    ("costLLMStage", "{llm_stage} on {population}: USD per 1,000 cases", "usd", "cost"),
]

# Placeholders defined in the chapters' number files (HANDOFF R8.4), replaced only when the evidence has them.
OVERRIDES = [
    ("nPoolMean", "{recommended} on {population}: mean candidate pool per case", "int", "numbers_ch3.tex"),
    ("nFeatureCols", "{recommended}: feature columns", "int", "numbers_ch3.tex"),
]

INPUTS = ["numbers_ch4.tex", "numbers_ch5.tex", "numbers_f51.tex", "numbers_ch3.tex", "numbers_ch2.tex",
          # read last, in this order: DEV values (RESULTS, 23 Oct), provisional marks and new values (D110), costs (D116)
          "numbers_runb_dev.tex", "numbers_results_first.tex", "numbers_cost_20261025.tex"]


def parse(text: str) -> dict[str, dict[str, str]]:
    """Every key of the evidence table: {key: {"value", "scope", "source"}}."""
    lines = text.splitlines()
    if not lines or not lines[0].startswith("# EVIDENCE_FINAL"):
        raise ValueError("not an EVIDENCE_FINAL.md (the first line must be '# EVIDENCE_FINAL ...'); the 15 Oct "
                         "EVIDENCE.md is history only")
    out: dict[str, dict[str, str]] = {}
    for ln in lines[1:]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) != 4 or cells[0] in ("Claim", "") or set(cells[0]) <= set("-: "):
            continue
        claim, value, scope, source = cells
        subject, _, measures = claim.partition(": ")
        names = [m.strip() for m in measures.split(";")] if measures else [""]
        values = [v.strip() for v in value.split(";")]
        if len(names) != len(values):
            raise ValueError(f"{len(names)} measures but {len(values)} values: {ln}")
        for m, v in zip(names, values):
            key = f"{subject}: {m}" if m else subject
            if key in out and out[key]["value"] != v:
                raise ValueError(f"two values for one key: {key} ({out[key]['value']} and {v})")
            out[key] = {"value": v, "scope": scope, "source": source}
    return out


def fmt(value: str, kind: str) -> str:
    x = float(value.replace(",", ""))
    if kind == "dot3":                                     # thesis style: 0.408, $-$0.012 (D149)
        s = f"{abs(x):.3f}"
        return ("$-$" if x < 0 else "") + s
    if kind == "delta3":
        s = f"{abs(x):.3f}"
        return ("$-$" if x < 0 else "+") + s
    if kind == "int":
        return f"{round(x):,}"
    if kind == "usd":
        return f"US\\${x:,.2f}"
    if kind == "pct1":
        return f"{x:.1f}\\%"
    raise ValueError(f"unknown format {kind}")


def roles(freeze: Path | None, given: list[str]) -> dict[str, str]:
    r = dict(ROLES)
    if freeze:
        doc = json.loads(Path(freeze).read_text(encoding="utf-8"))
        r.update({k: str(doc[k]) for k in ("recommended", "llm_stage", "population") if doc.get(k)})
    for g in given:
        k, _, v = g.partition("=")
        r[k.strip()] = v.strip()
    return r


def resolve(key: str, r: dict[str, str]) -> str | None:
    try:
        return key.format(**r)
    except KeyError:                                       # a role not chosen yet (before the freeze)
        return None


def build(evidence: Path, r: dict[str, str]) -> tuple[str, dict]:
    raw = Path(evidence).read_bytes()
    ev = parse(raw.decode("utf-8"))
    report = {"filled": {}, "pending": {}}
    out = ["% Evidence macros (WRITING_PLAN section 4). Every number in the text comes from this file.",
           "% GENERATED by scripts/evidence_macros.py; do not edit by hand. Regenerate from the run's EVIDENCE_FINAL.md.",
           f"% Source: {Path(evidence).name}, sha256 {hashlib.sha256(raw).hexdigest()}, "
           f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}",
           "% Roles: " + ", ".join(f"{k}={v}" for k, v in sorted(r.items())),
           r"\providecommand{\EvPending}[1]{\textcolor{red!65!black}{\textsf{[#1]}}}", ""]
    for title, pairs in FIXED:
        out += [f"% {title}"] + [f"\\newcommand{{\\{m}}}{{{v}}}" for m, v in pairs] + [""]
    out.append("% Results (EVIDENCE_FINAL.md; a key the evidence lacks prints a visible placeholder).")
    for macro, key, kind, label in RESULTS:
        k = resolve(key, r)
        if k and k in ev:
            out.append(f"\\newcommand{{\\{macro}}}{{{fmt(ev[k]['value'], kind)}}}  % {k} | {ev[k]['source']}")
            report["filled"][macro] = k
        else:
            out.append(f"\\newcommand{{\\{macro}}}{{\\EvPending{{{label}}}}}")
            report["pending"][macro] = k or key
    out += ["", "% Per-chapter number files, each owned by one writer."]
    out += [f"\\InputIfFileExists{{evidence/{f}}}{{}}{{}}" for f in INPUTS]
    over = []
    for macro, key, kind, home in OVERRIDES:
        k = resolve(key, r)
        if k and k in ev:
            over.append(f"\\renewcommand{{\\{macro}}}{{{fmt(ev[k]['value'], kind)}}}  % {k} | {ev[k]['source']} ({home})")
            report["filled"][macro] = k
        else:
            report["pending"][macro] = k or key
    if over:
        out += ["", "% Values the run supplies for placeholders of the chapter files."] + over
    return "\n".join(out) + "\n", report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--role", action="append", default=[], help="name=value, e.g. recommended=S_full_ext")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--strict", action="store_true", help="fail if any macro stays pending")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    text, rep = build(Path(a.evidence), roles(Path(a.freeze) if a.freeze else None, a.role))
    print(json.dumps({"filled": len(rep["filled"]), "pending": rep["pending"]}, indent=1))
    if a.strict and rep["pending"]:
        return 1
    if a.dry_run:
        sys.stdout.write(text)
    else:
        Path(a.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
