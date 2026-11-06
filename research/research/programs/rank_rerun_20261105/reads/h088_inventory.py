"""H088: stored development runs that no generator or table of the thesis names, and figures on disk that the chapters and
appendices do not input. A run counts as reported when its id occurs in the current text's generators or tables. Scores
read from summary.json. Writes out/h088.json."""
from common import *

GEN = [ROOT / "scripts/final_results.py", ROOT / "scripts/t6_tables.py", *(ROOT / "src/uol/thesis_figures").glob("*.py"), *(ROOT / "thesis/tables").glob("*.tex"),
       ROOT / "research/programs/final_results_20261102/depends.json"]
TEX = [*(ROOT / "thesis/chapters").glob("*.tex"), *(ROOT / "thesis/appendices").glob("*.tex"), ROOT / "thesis/thesis.tex", *(ROOT / "thesis/tables").glob("*.tex")]
blob = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in GEN if p.exists())
tex = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in TEX if p.exists())
runs = {}
for store, lab in ((FOOT, "footing"), (RECORD, "record"), (AUDIT, "audit")):
    for r in sorted(store.iterdir()):
        if (r / "DEV" / "summary.json").exists():
            runs.setdefault(r.name, lab)
print("runs with a DEV summary:", {k: sum(v == k for v in runs.values()) for k in ("footing", "record", "audit")}, "total", len(runs))
named, un = [], []
for r in runs:
    (named if re.search(r"(?<![\w+@])" + re.escape(r) + r"(?![\w+@])", blob) else un).append(r)
print("named in generators or tables:", len(named), "| not named:", len(un))
fam = {}
for r in un:
    k = ("rrf/linear hybrids" if r.startswith(("rrf__", "linear__")) else "qwen compress" if re.search(r"(mrl|binary|int8)", r) else "single source (src_)" if r.startswith("src_")
         else "ranker tests (rt_, rk_)" if r.startswith(("rt_", "rk_")) else "@variants" if "@" in r else "other")
    fam.setdefault(k, []).append(r)
for k, v in fam.items():
    print(f"  {k}: {len(v)}", v if len(v) <= 40 else v[:40])
print("\nscores of the not-named 'other' and single sources")
rows = []
for r in fam.get("other", []) + fam.get("single source (src_)", []) + fam.get("@variants", []):
    j = summary(r)
    rows.append((r, round(j.get("ndcg_at_10") or 0, 3), round(j.get("unseen_recall_at_20_pooled") or 0, 3), round(j.get("law_at_1") or 0, 3), round(j.get("success_at_1_in_law") or 0, 3), j.get("n_cases")))
print(pd.DataFrame(rows, columns=["run", "ndcg", "never", "law1", "inlaw", "n"]).to_string(index=False))

print("\n== figures")
fig_dir = ROOT / "thesis/figures"
used = set(re.findall(r"figures/([\w\-.]+?)(?:\.tex|\.pdf|\})", tex)) | set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{(?:figures/)?([\w\-.]+?)(?:\.pdf)?\}", tex))
stems = sorted({p.stem for p in fig_dir.iterdir() if p.suffix in (".pdf", ".tex")})
print("stems on disk", len(stems), "used", len(used))
print("used:", sorted(used))
print("not input anywhere:", [s for s in stems if s not in used])
for s in ("F6_3_encoders", "F2c_encoders", "F6_3b_training", "F04_ladder", "F6_6_ranker_gain", "F6_4_compression", "F6_5_hybrids", "F6_7_bands", "F6_2_reach", "F6_14_controls", "F6_12_beyond", "F6_alone_list", "F6_9_llm_stage", "F6_8_law_article"):
    print(s, "on disk", any((fig_dir / f"{s}{e}").exists() for e in (".pdf", ".tex")), "| used", s in used)
for s in ("F2c_encoders", "F6_3_encoders"):
    print(s, "mentions in chapters:", [m.group(0)[:80] for m in re.finditer(s + r".{0,60}", tex)][:4])
print("\ndepth sweep of the model's reranking on the older list, and others worth seeing")
for s in ("S0+qwen3_8b", "gpt61sol_d10_rerank_S0+qwen3_8b", "gpt61sol_rerank_S0+qwen3_8b", "gpt61sol_d60_rerank_S0+qwen3_8b", "docexp_bm25", "qwen3_8b_headings", "rt_lm_wide", "rt_lm_wide_tesis", "rt_xgb", "rt_xgb_tesis", "rt_lm_tesis", "qwen3_8b_vectors@agg_top2", "S_full_minus_D", "S_full_minus_M"):
    j = summary(s)
    print(s, round(j["ndcg_at_10"], 3), "never", round(j["unseen_recall_at_20_pooled"], 3), "R20", round(j["recall_at_20"], 3), "law1", round(j["law_at_1"], 3), "inlaw", round(j["success_at_1_in_law"], 3))
save("h088", {"not_named": fam, "named": named})
