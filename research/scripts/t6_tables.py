"""Appendix B's two tables on the final footing (rebuild PLAN, section 5, "B: regenerate T6_all_systems and T6_deciding").

* ``thesis/tables/T6_all_systems.tex``: the left half (baselines and encoders, all read from the 25 Oct record on the same
  1,217 development cases) is kept as it stands; the right half is written here from
  ``research/programs/final_results_20261102/results.json``: past citations, the learned rankers, the rerankers and
  language models on the ranker's list, the model alone and the system, and the sample of 300 cases.
* ``thesis/tables/T6_deciding.tex``: every comparison that decides a verdict, by family, with its size m, difference,
  adjusted lower end and class (v33, D203: the Holm p column is cut, a second test that decides nothing). The rows of the
  25 Oct record (the first research question, the encoder) keep their macros; the rest are read from the families of the
  footing and, since v33, of the rerun on the ranker itself (uol.final.lean_rerun).

A result not yet held leaves its row out and is listed on stdout; rerun after ``scripts/final_results.py``.

    py -3.12 scripts/t6_tables.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("final_results", ROOT / "scripts" / "final_results.py")
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

TAB = ROOT / "thesis" / "tables"
MISSING: list[str] = []


def results() -> dict:
    return json.loads((F.OUT / "results.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- Table B.1, right half


def all_systems(r: dict) -> str:
    dev, smp = r["dev"], r["sample"]
    cls = (r["experiments"].get("classifier") or {})

    def d(sid: str) -> float | None:
        return (dev.get(sid) or {}).get("ndcg_at_10")

    def s(key: str) -> float | None:
        return (smp.get(key) or {}).get("mean_per_case")

    def row(name: str, v, dagger: bool = False) -> str:
        if v is None:
            MISSING.append(f"B.1 {name}")
            return ""
        return f"      {name}{chr(92) + 'textdagger' if dagger else ''} & {F.f3(v)} \\\\\n"

    sys_id = r["system"]["dev_system"]
    rk = F.RANKER
    blocks = [
        ("Learned rankers", [   # v33 (D203): every comparison on the ranker itself (uol.final.lean_rerun)
            ("no past citations, no encoder", d("rk_neither")),
            ("the encoder, no past citations", d("rk_nocit")),
            ("the ranker without the encoder", d(F.NOENC)),
            ("\\quad plus BGE-M3", d("rk_noenc+bge_m3")),
            ("\\quad plus Qwen3-Embedding-0.6B", d("rk_noenc+qwen3_0p6b")),
            ("\\quad plus Qwen3-Embedding-4B", d("rk_noenc+qwen3_4b")),
            ("\\quad plus Qwen3-Embedding-8B", d("rk_noenc+qwen3_8b")),
            ("the ranker", d(rk)),
            ("\\quad without the votes of older \\emph{tesis}", d("rt_check")),
            ("\\quad without them, trees on single articles", d("lf_plain")),
            ("\\quad trees on single articles, with topic scores", d("ft_lean")),
            ("\\quad plus the topic scores", d("rk_topics")),
            ("\\quad plus the statute's structure", d("rk_structure")),
            ("\\quad plus vectors smoothed along it", d("rk_smooth")),
            ("\\quad plus co-citation", d("rk_cocite")),
            ("\\quad plus the Qwen3-8B adapter", d("rk_adapter")),
            ("\\quad plus legal issues", d("rk_issues")),
            ("\\quad plus fact patterns", d("rk_patterns")),
            ("\\quad plus query2doc", d("rk_q2d")),
            ("\\quad plus HyDE", d("rk_hyde")),
            ("trees on single articles, 28 sources", d("ft_ranker")),
            ("with every source built", d("S_full_ext")),
            ("one classifier per article, words", (cls.get("words") or {}).get("ndcg_at_10")),
            ("one classifier per article, Qwen3-8B", (cls.get("encoder") or {}).get("ndcg_at_10")),
        ]),
        ("Rerankers of the ranker's first 30", [
            ("BGE reranker", d("bge_ranker_rk")),
            ("Qwen3-Reranker-4B", d("qwen4b_ranker_rk")),
            ("Qwen3-Reranker-0.6B, trained", d("ce_ranker_rk")),
        ]),
        ("Language models on the ranker's first 30", [
            ("GPT-6.1 Sol", d(F.REORD)),
            ("GPT-6 Sol", d(f"gpt6solnw_rerank_{rk}")),
            ("GPT-6.1 Terra", d(f"gpt61terra_rerank_{rk}")),
            ("GPT-6 Luna", d(f"gpt6luna_rerank_{rk}")),
            ("GPT-5.6 Sol", d(f"gpt56sol_rerank_{rk}")),
            ("GPT-5.6 Terra", d(f"gpt56terra_rerank_{rk}")),
            ("GPT-5.6 Luna", d(f"luna_rerank_{rk}")),
        ]),
        ("The model and the system", [
            ("GPT-6.1 Sol alone", d("gpt61sol_alone")),
            ("merged list, no final reranking", d(F.CHECK["merged"])),
            ("the system", d(sys_id)),
            ("\\quad each window also given in reverse", d(F.CHECK["both"])),
            ("\\quad low effort", d(F.CHECK["low"])),
            ("\\quad high effort", d(F.HIGH)),
            ("\\quad first 60 reranked", d(F.CHECK["d60"])),
            ("\\quad without the encoder", d(F.CHECK["noenc"])),
            ("\\quad the model's own list reranked", d(F.CHECK["self"])),
            ("\\quad trees on single articles in the ranker", d("gpt61sol_combo_fwd_on_ft_lean")),
            ("\\quad every source built in the ranker", d("gpt61sol_final_on_S_full_ext")),
        ]),
    ]
    out, left = [], ""
    for title, rows in blocks:
        body = "".join(row(*x) for x in rows)
        if body and title.startswith("Rerankers"):    # v33: at the foot of the left half, so the table keeps to one page
            left = f"      \\Grp{{{title}}}\n{body}"
        elif body:
            out.append(f"      \\Grp{{{title}}}\n{body}")
    return "      \\addlinespace[2pt]\n".join(out), left


def write_all_systems(r: dict) -> None:
    p = TAB / "T6_all_systems.tex"
    s = p.read_text(encoding="utf-8")
    start = s.index("\\Grp{Past citations}")
    past_end = s.index("\\addlinespace[2pt]", start)
    end = s.rindex("      \\bottomrule")
    right, left = all_systems(r)
    new = s[:past_end] + "\\addlinespace[2pt]\n" + right + s[end:]
    old_left = new.find("      \\addlinespace[2pt]\n      \\Grp{Rerankers")   # a rerun replaces the block it wrote before
    if old_left >= 0:
        new = new[:old_left] + new[new.index("      \\bottomrule"):]
    if left:
        foot = new.index("      \\bottomrule")              # the first one closes the left half
        new = new[:foot] + "      \\addlinespace[2pt]\n" + left + new[foot:]
    new = new.replace("\\renewcommand{\\arraystretch}{0.86}", "\\renewcommand{\\arraystretch}{0.84}")
    head = "% Right half regenerated by scripts/t6_tables.py from research/programs/final_results_20261102/results.json (PLAN addendum 22, 3 Nov 2026).\n"
    if not new.startswith(head):
        new = head + new
    new = new.replace(" Systems inside a ranker are those of \\cref{sec:contribution}.", "")   # v33: RA-05
    p.write_text(new, encoding="utf-8")


# --------------------------------------------------------------------------- Table B.2


def fam_rows(fam: dict | None, keys: list[tuple[str, str, bool]]) -> list[str]:
    fam = fam or {}
    cs, m = fam.get("contrasts", {}), int(fam.get("size") or len(fam.get("contrasts", {})) or 0)
    out = []
    for key, name, neg in keys:
        c = cs.get(key)
        if not c:
            MISSING.append(f"B.2 {name}")
            continue
        delta, lo = c["delta"], c.get("adjusted_lower")
        kind = F.cls_of(c)
        if neg:
            ca = c["ci_adjusted"]
            delta, lo = -delta, -ca[1]
            kind = F.classify(lo, [-c["ci90"][1], -c["ci90"][0]], -ca[0])
        out.append(f"    {name} & {m} & {F.sg(delta)} & {F.sglo(lo)} & {F.CLASS[kind]} \\\\\n")
    return out


def cached(name: str) -> dict:
    """A family final_results.py caches in its output folder (PLAN addendum 22), or an empty one."""
    p = F.OUT / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def write_deciding(r: dict) -> None:
    fams = r["families"]
    sid = r["system"]["dev_system"]
    rk, ne, ro = F.RANKER, F.NOENC, F.REORD
    groups = [
        ("The ranker", fam_rows(cached("noenc_vs_lex"), [(f"{ne} - lstar_headings", "Ranker without the encoder against lexical matching", False)])
         + fam_rows(cached("encoder_gain"), [(f"{rk} - {ne}", "Qwen3-Embedding-8B and its sources added", False)])
         + fam_rows(cached("rk_choice"), [(f"{rk} - ft_lean", "The ranker against trees on single articles", False)])
         + fam_rows(cached("rk_parts"), [(f"{rk} - rt_check", "Votes of older \\textit{tesis} added", False),
                                         ("rt_check - lf_plain", "Whole-list against single-article trees", False)])
         + fam_rows(cached("rk_largest"), [(f"{rk} - S_full_ext", "The ranker against every source built", False)])
         + fam_rows(fams.get("lean", {}), [("ft_lean - ft_ranker", "Trees on single articles: eight sources against 28", False)])),
        ("Additions to the ranker", fam_rows(cached("rk_additions"), [
            (f"rk_topics - {rk}", "The four topic scores added", False),
            (f"rk_structure - {rk}", "Statute's structure added", False),
            (f"rk_smooth - {rk}", "Vectors smoothed along it added", False),
            (f"rk_cocite - {rk}", "Co-citation added", False),
            (f"rk_adapter - {rk}", "Qwen3-8B adapter added", False),
            (f"rk_issues - {rk}", "Legal issues added", False),
            (f"rk_patterns - {rk}", "Fact patterns added", False),
            (f"rk_q2d - {rk}", "query2doc added", False),
            (f"rk_hyde - {rk}", "HyDE added", False)])),
        ("The model on the ranker's list", fam_rows(cached("lean_models"), [
            (f"gpt61sol_rerank_{rk} - gpt6solnw_rerank_{rk}", "GPT-6.1 Sol against GPT-6 Sol", False),
            (f"gpt61terra_rerank_{rk} - gpt6solnw_rerank_{rk}", "GPT-6.1 Terra against GPT-6 Sol", False),
            ])),
        ("The system", fam_rows(cached("final_dev"), [
            (f"{sid} - {ro}", "System against model on the ranker's list", False),
            (f"{sid} - gpt61sol_alone", "System against model alone", False),
            (f"{sid} - {rk}", "System against ranker", False)])
         + fam_rows(cached("alone_reord"), [(f"gpt61sol_alone - {ro}", "Model alone against model on the ranker's list", False)])
         + fam_rows(cached("final_checks"), [
            (f"{F.CHECK['merged']} - {sid}", "Final reranking against merged list", True),
            (f"{F.CHECK['self']} - {sid}", "Ranker's list against the model's own list reranked", True),
            (f"{F.CHECK['noenc']} - {sid}", "Encoder, inside the system", True),
            (f"{F.CHECK['both']} - {sid}", "Two orders against one", False),
            (f"{F.CHECK['low']} - {sid}", "Low effort against medium", False),
            (f"{F.CHECK['d60']} - {sid}", "First 60 reranked against 30", False)])
         + fam_rows(cached("rk_in_system"), [(f"gpt61sol_combo_fwd_on_ft_lean - {sid}", "Trees on single articles, inside the system", False),
                                                   (f"gpt61sol_final_on_S_full_ext - {sid}", "Every source built, inside the system", False)])),
    ]
    body = "    \\addlinespace[3pt]\n".join(f"    \\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{t}}}}} \\\\\n" + "".join(rows) for t, rows in groups if rows)
    tex = r"""% The comparisons that decide the verdicts of Chapter 6, by family (appendix B, app:statistics). Written by
% scripts/t6_tables.py from the 100,000-draw families of the footing (results.json and the families final_results.py
% caches: final_dev, final_checks, encoder_gain, noenc_vs_lex, alone_reord and the rk_* families of
% the rerun on the ranker itself). PLAN addendum 22, 3 Nov 2026. Do not edit by hand.
\begin{table}[p]
  \centering
  \caption[The comparisons that decide support]{The comparisons that decide the verdicts of \cref{chap:results} on the development cases: the first system minus the second in nDCG@10, the family size $m$, the adjusted lower end and the class.}
  \label{tab:app-deciding}
  \scriptsize\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{0.95}\renewcommand{\emph}[1]{#1}% class words upright, as in the generated rows
  \let\CompactTabX\tabularx\let\endCompactTabX\endtabularx% D156: out of reach of the D155 hook, so the table stays on one page
  \begin{CompactTabX}{\textwidth}{@{}>{\raggedright\arraybackslash}X r r r l@{}}
    \toprule
    Comparison & $m$ & Difference & Lower end & Class \\
    \midrule
""" + body + r"""    \bottomrule
  \end{CompactTabX}
\end{table}
"""
    (TAB / "T6_deciding.tex").write_text(tex, encoding="utf-8")


def main() -> int:
    r = results()
    write_all_systems(r)
    write_deciding(r)
    print("written: T6_all_systems.tex, T6_deciding.tex")
    if MISSING:
        print(f"{len(MISSING)} rows waiting: " + "; ".join(MISSING))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
