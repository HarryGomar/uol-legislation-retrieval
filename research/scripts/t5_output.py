"""Table 5.2: the system's first ten suggestions for the running example, law by law, each with its opening words and the
similar past tesis that cited it (rebuild PLAN, section 6; D203, 4 Nov 2026: the supporting tesis added).

The system is the one the record names (``research/programs/final_results_20261102/results.json``, ``system.dev_system``),
so a rerun after the freeze follows it. Articles, ranks and the cited article come from the system's stored top-200 and
per-case files, the law names from the catalogue. The supporting tesis are the prototype's provenance
(``uol.prototype.pipeline.Engine._suggestion``): among the ten nearest development cases of the other folds (the memory of
``knn_qwen8b``) and the ten nearest older tesis (the historical memory), by the encoder's similarity to the facts, those
that cited the article, most similar first, by registry number. ``scripts/support_share.py`` measures the same support over
every development case. Writes ``thesis/tables/T5_2_output_system.tex``.

    py -3.12 scripts/t5_output.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uol.analysis_c.common import load_catalogue  # noqa: E402
from uol.prototype import signals as S  # noqa: E402
from uol.thesis_figures.ch6_numbers import RUNNING_EXAMPLE  # noqa: E402

RESULTS = ROOT / "research/programs/final_results_20261102/results.json"
STORE = ROOT / "experiments/runs/footing_20261101/dev/store/runs"
OUT = ROOT / "thesis/tables/T5_2_output_system.tex"
TOP = 10
WORDS = 11


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def law_name(title: str) -> str:
    first, _, rest = title.strip().partition(" ")      # the catalogue writes the first word in capitals
    return (first.capitalize() + " " + rest).strip()


def tex(s: str) -> str:
    return re.sub(r"([%&$#_{}])", r"\\\1", s)


def opening(snippet: str) -> str:
    body = re.sub(r"^\s*Art[íi]culo\s+\S+\s+", "", snippet, flags=re.I)  # "Artículo 2o.- Las sociedades ..." -> "Las sociedades ..."
    words = re.split(r"\s+Párrafo\b", body)[0].split()                    # the catalogue marks paragraphs with "Párrafo"
    head = " ".join(words[:WORDS])
    cut = head.find(":")                                                  # a list after a colon reads as cut short
    if cut > 0:
        head, words = head[:cut], head[:cut].split() + ["…"]
    return tex(head.rstrip(",;:.") + (" \\ldots" if len(words) > WORDS or "…" in words else ""))


def support(query_id: str, ids: list[str]) -> dict[str, list[str]]:
    """Registry numbers of the similar past tesis that cited each article, most similar first."""
    dev, cat, census = S.Dev.load(), S.Catalogue(load_vectors=False), S.Census()
    p = dev.by_query[query_id]
    fit = S.build_fit(dev, cat, f"dev_fold{dev.rows[p]['fold']}")      # the case's own fold stays out of the memory
    vecs = {v: np.asarray(dev.vectors(v)[[p]], dtype=np.float32) for v in fit.cfg.variants}
    _, ev = S.compute(cat, fit, census, [dev.texts[p]], vecs, [dev.components[p]], ["knn_qwen8b", "census_qwen8b"])
    where = {m["candidate_id"]: i for i, m in enumerate(cat.meta)}
    out = {}
    for c in ids:
        i, found = where[c], []
        for e, s in zip(ev.dev_idx[0], ev.dev_sim[0]):
            if i in fit.memory.positives[int(e)]:
                found.append((float(s), dev.rows[int(fit.memory.rep[int(e)])]["record_id"]))
        for e, s in zip(ev.census_idx[0], ev.census_sim[0]):
            if i in census.meta[int(e)]["cited"]:
                found.append((float(s), census.meta[int(e)]["record_id"]))
        out[c] = [r for _, r in sorted(found, key=lambda x: -x[0])]
    out["_snippets"] = {c: cat.meta[where[c]]["snippet"] for c in ids}
    return out


def main() -> None:
    sid = json.loads(RESULTS.read_text(encoding="utf-8"))["system"]["dev_system"]
    run = STORE / sid / "DEV"
    cat = load_catalogue()
    top = pd.read_parquet(run / "top200.parquet")
    top = top[top.query_id == RUNNING_EXAMPLE].sort_values("rank").head(TOP)
    case = pd.read_parquet(run / "per_case.parquet")
    pos = case.loc[case.query_id == RUNNING_EXAMPLE, "positive_idx"].iloc[0]
    cited = {int(i) for i in re.findall(r"\d+", str(pos))}         # stored as text, e.g. "28247" or "12 345"
    ids = [cat.candidate_ids[int(i)] for i in top.candidate_idx]
    sup = support(RUNNING_EXAMPLE, ids)
    groups: dict[int, list[tuple[int, str, bool, str]]] = {}
    for _, r in top.iterrows():
        i = int(r.candidate_idx)
        c = cat.candidate_ids[i]
        groups.setdefault(int(cat.law[i]), []).append((int(r["rank"]), c.split("#articulo=", 1)[1], i in cited, c))
    rows = []
    for law, arts in sorted(groups.items(), key=lambda kv: kv[1][0][0]):
        rows.append(rf"    \multicolumn{{3}}{{@{{}}l}}{{\emph{{{law_name(cat.law_titles[law])}}}}} \\")
        for k, a, c, cid in arts:
            label = (rf"\textbf{{{a}}}" if c else a) + f" ({k})"
            rows.append(f"    {label} & {opening(sup['_snippets'][cid])} & {', '.join(sup[cid]) or '--'} \\\\")
    body = "\n".join([
        "% Generated by scripts/t5_output.py; do not edit by hand.",
        f"% Source: {run.relative_to(ROOT).as_posix()}/top200.parquet sha256 {sha(run / 'top200.parquet')}; query {RUNNING_EXAMPLE};",
        "% supporting tesis from uol.prototype (development memory of the other folds and the historical memory, ten nearest each)",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption[The output for the running example]{The first ten suggestions of the system for the running example, law by law, with each article's rank in brackets, its opening words and the registry numbers of the similar past \emph{tesis} that cited it. The article the court cited is in bold.}",
        r"  \label{tab:output-example}",
        r"  \small",
        r"  \begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{1.7cm} >{\raggedright\arraybackslash}X >{\raggedright\arraybackslash}p{3.3cm}@{}}",
        r"    \toprule",
        r"    Article (rank) & Opening words & Past \emph{tesis} that cited it \\",
        r"    \midrule",
        *rows,
        r"    \bottomrule",
        r"  \end{tabularx}",
        r"\end{table}",
        "",
    ])
    OUT.write_text(body, encoding="utf-8")
    print(body)


if __name__ == "__main__":
    main()
