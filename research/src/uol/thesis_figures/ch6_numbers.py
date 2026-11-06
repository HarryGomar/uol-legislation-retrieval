"""Chapter 6 evidence macros from the RUN-B development store (WRITE-B, 23 Oct 2026; fixes_after_audit_20261023).

Reads only saved outputs: ``<store>/runs/<system>/DEV/{summary.json, per_case.parquet}``, the contrast families
``<store>/contrasts/*.json`` (10,000 paired draws, the rule of section 4.4) and the tidy CSVs of ``uol.analysis_c``
(``c1``, ``c3``, ``recall``) run on the same store. Writes ``thesis/evidence/numbers_ch6.tex``; every number of
Chapter 6 comes from it. A value whose job has not finished is written as ``\\EvPending``.

    py -3.12 -m uol.thesis_figures.ch6_numbers [--store DIR] [--analysis DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from uol.thesis_figures import common as C
from uol.thesis_figures import ch6_figs as F

# The registered, read-only final record (D131): the defaults since the sprint of 26 Oct 2026 (session B). The
# single contrasts that the store's family files lack sit in the record's Drive supplement.
STORE = F.RECORD / "store_dev_20261025b"
ANALYSIS = F.RECORD / "analysis_c_20261025b"
EXTRA_CONTRASTS = [F.RECORD / "drive" / "raw" / "contrasts", F.RECORD / "supplement" / "contrasts"]
# the same contrasts recomputed with 100,000 draws (D138, session D); read before the 10,000-draw files
BOOT_100K = F.RECORD / "supplement" / "bootstrap_100k" / "contrasts"
OUT = C.ROOT / "thesis" / "evidence" / "numbers_ch6.tex"

# nDCG@10 of single systems: macro -> store id
SYSTEMS = {
    "Lex": "lstar_headings", "BM": "bm25_stem", "Pop": "popularity", "Rand": "random",
    "LSA": "lsa", "LDAdm": "lda_dm", "LDAsim": "lda_sim", "Docvec": "doc2vec", "Wordvec": "word2vec",
    "Esmall": "e5_small", "Ebase": "e5_base", "Elarge": "e5_large", "Einstr": "e5_large_instruct",
    "BGE": "bge_m3", "BGEsparse": "bge_m3_sparse", "Colbert": "bge_m3_colbert", "Jina": "jina_v3",
    "Arctic": "arctic_l_v2", "GTE": "gte_multilingual",
    "Qsmall": "qwen3_0p6b", "Qmid": "qwen3_4b", "Qbig": "qwen3_8b", "QbigInt": "qwen3_8b_int8",
    "QbigBin": "qwen3_8b_binary", "QbigTwoFiftySix": "qwen3_8b_mrl256", "QbigTwoFiftySixBin": "qwen3_8b_mrl256_binary",
    "QsmallTwoFiftySixBin": "qwen3_0p6b_mrl256_binary", "QbigHead": "qwen3_8b_headings",
    "QbigTopTwo": "qwen3_8b_vectors@agg_top2", "QbigInstrA": "qwen3_8b_vectors@qa",
    "Euler": "euler_legal", "EulerBase": "qwen3_8b_int8_base", "BGElegal": "bge_m3_legal",
    "BGElegalSeven": "bge_m3_legal_768", "BGEbaseSeven": "bge_m3_base_768", "BGEbase": "bge_m3_base",
    "Elegal": "e5_large_legal_en", "GemmaLegal": "gemma_legal", "Alia": "alia_mrbert",
    "EsmallFT": "ft_e5_q", "BGEft": "bge_m3_ft", "Adapter": "qwen8b_adapter",
    "KH": "src_KH_qwen8b", "K": "src_K_qwen8b", "Proto": "src_proto_qwen8b", "ProtoTfidf": "src_proto_tfidf",
    "Casesim": "src_knn_casesim_e5", "Docexp": "docexp_bm25", "CitingFacts": "src_docexp_facts_bm25",
    "Klex": "src_K_lex", "KlexFive": "src_knn_lex_votes_k5", "KlexVotes": "src_knn_lex_votes",
    "KlexFifty": "src_knn_lex_votes_k50", "Hist": "src_H_qwen8b",
    "LLMalone": "llm_alone", "AgentWords": "agent_words", "QtwoD": "query2doc", "Hyde": "hyde",
    "Szero": "S0", "Sfull": "S_full", "Sext": "S_full_ext", "LLMext": "llm_rerank_S_full_ext",
    "LLMzero": "llm_rerank_S0", "LLMblank": "llm_rerank_S_full_ext_blank",
    "RRFall": "rrf_all_emb", "RRFallNoemb": "rrf_all_noemb", "LinLearn": "ltr_emb_linear",
    "LinLearnNoemb": "ltr_noemb_linear", "LambdaMART": "ltr_emb_lambdamart", "LambdaMARTnoemb": "ltr_noemb_lambdamart",
    "BGErerank": "rerank_bge", "BGErerankAlone": "rerank_bge_alone",
    "LinQbig": "linear__lstar_headings__qwen3_8b", "RRFQbig": "rrf__lstar_headings__qwen3_8b",
    "RRFdocvec": "rrf__lstar_headings__doc2vec", "RRFldasim": "rrf__lstar_headings__lda_sim",
    "LinDocvec": "linear__lstar_headings__doc2vec", "SzeroQbigWithSources": "S0+qwen3_8b",
    "RebuiltNoemb": "ltr_noemb_refit@rebuilt@v11", "RebuiltEmb": "ltr_emb_refit@rebuilt@v11",
    "FrozenNoemb": "ltr_noemb_frozen@v11", "FrozenEmb": "ltr_emb_frozen@v11",
    # the ranker without embeddings plus one model's dense score, for Table B.1 (R review 1.4)
    "SzeroQsmall": "S0+qwen3_0p6b", "SzeroQmid": "S0+qwen3_4b", "SzeroQbigDense": "S0+qwen3_8b_dense",
    "SzeroBGE": "S0+bge_m3_dense", "SzeroEbase": "S0+e5_base", "SzeroElarge": "S0+e5_large",
    "SzeroGTE": "S0+gte_multilingual", "SzeroArctic": "S0+arctic_l_v2", "SzeroBGElegal": "S0+bge_m3_legal",
    "SzeroEuler": "S0+euler_legal", "SzeroTwoFiftySix": "S0+qwen3_8b_mrl256",
}
# further measures of a few systems: macro prefix -> summary key
MEASURES = {"ur": "unseen_recall_at_20_pooled",   # pooled share of never-cited pairs (Ch4; RESULTS.md)
            "lwa": "law_at_1", "lwb": "law_at_10", "sa": "success_at_1",
            "sb": "success_at_10", "il": "success_at_1_in_law", "rh": "recall_at_100"}
MEASURED = ["Lex", "BM", "LSA", "Qbig", "Qsmall", "Esmall", "EsmallFT", "BGE", "BGEft", "Adapter", "Euler", "KH", "K",
            "Proto", "Szero", "Sfull", "Sext", "LLMext", "LLMzero", "LLMalone", "Pop", "AgentWords", "QtwoD", "Hyde"]
# contrasts: macro -> (file, key)
CONTRASTS = {
    "Main": ("F1", "S_full - S0"), "Ext": ("S_full_ext__vs__S0__DEV", None), "HistExt": ("S_full_ext__vs__S_full__DEV", None),
    "LadQbig": ("F2", "S0+qwen3_8b_dense - S0"), "LadQmid": ("F2", "S0+qwen3_4b - S0"),
    "LadQsmall": ("F2", "S0+qwen3_0p6b - S0"), "LadBGE": ("F2", "S0+bge_m3_dense - S0"),
    "LadEbase": ("F2", "S0+e5_base - S0"), "LadElarge": ("F2", "S0+e5_large - S0"), "LadJina": ("F2", "S0+jina_v3 - S0"),
    "LadEuler": ("F2", "S0+euler_legal - S0"), "LadQtwoD": ("F2", "S0+query2doc - S0"), "LadHyde": ("F2", "S0+hyde - S0"),
    "LadTwoFiftySix": ("F2", "S0+qwen3_8b_mrl256 - S0"), "LadBin": ("F2", "S0+qwen3_8b_binary - S0"),
    "LadBGElegal": ("F2", "S0+bge_m3_legal - S0"),
    "LegBGE": ("F3", "bge_m3_legal - bge_m3_base"), "LegEuler": ("F3", "euler_legal - qwen3_8b_int8_base"),
    "FtBGE": ("F3", "bge_m3_ft - bge_m3"), "FtEsmall": ("F3", "ft_e5_q - e5_small"), "FtAdapter": ("F3", "qwen8b_adapter - qwen3_8b"),
    "HyLinQbig": ("F4", "linear__lstar_headings__qwen3_8b - qwen3_8b"), "HyRRFQbig": ("F4", "rrf__lstar_headings__qwen3_8b - qwen3_8b"),
    "HyRRFdocvec": ("F4", "rrf__lstar_headings__doc2vec - lstar_headings"),
    "HyRRFldasim": ("F4", "rrf__lstar_headings__lda_sim - lstar_headings"),
    "HyLinDocvec": ("F4", "linear__lstar_headings__doc2vec - lstar_headings"),
    "HyLinPop": ("F4", "linear__lstar_headings__popularity - lstar_headings"),
    "HyLinQsmall": ("F4", "linear__lstar_headings__qwen3_0p6b - lstar_headings"),
    "LeLM": ("F5", "ltr_emb_lambdamart - ltr_emb_refit"), "LeLMnoemb": ("F5", "ltr_noemb_lambdamart - ltr_noemb_refit"),
    "LeLin": ("F5", "ltr_emb_linear - ltr_emb_refit"), "LeRRF": ("F5", "rrf_all_emb - ltr_emb_refit"),
    "LeRRFnoemb": ("F5", "rrf_all_noemb - ltr_noemb_refit"),
    "StLLMext": ("F6", "llm_rerank_S_full_ext - S_full_ext"), "StLLMzero": ("F6", "llm_rerank_S0 - S0"),
    "StBGE": ("F6", "rerank_bge - S_full_ext"), "StLLMzeroVsFull": ("F6", "llm_rerank_S0 - S_full"),
    # the parts left out, which decide support (D142), and the second model family; for the table of appendix B.3
    "AblQbig": ("F8", "S_full_ext - S_full_ext_minus_qwen3_8b"), "AblDense": ("F8", "S_full - S_full_minus_D"),
    "AblMemory": ("F8", "S_full - S_full_minus_M"), "AblColbert": ("F7", "S_full_ext - S_full_ext_minus_colbert"),
    "Claude": ("F6x", "claude_rerank_S_full_ext - S_full_ext"),
}
PLAIN = {  # how a pending comparison reads in print
    "Main": "ranker with embeddings over the ranker without", "Ext": "integrated method over the ranker without embeddings",
    "HistExt": "older tesis in the ranker", "LadQbig": "the 8B inside the ranker", "LadQmid": "the 4B inside the ranker",
    "LadQsmall": "the 0.6B inside the ranker", "LadBGE": "BGE-M3 inside the ranker", "LadEbase": "E5-base inside the ranker",
    "LadElarge": "E5-large inside the ranker", "LadJina": "Jina v3 inside the ranker",
    "LadEuler": "the legal 8B inside the ranker", "LadQtwoD": "query2doc inside the ranker", "LadHyde": "HyDE inside the ranker",
    "LadTwoFiftySix": "the 8B at 256 dimensions inside the ranker", "LadBin": "the 8B in signs inside the ranker",
    "LadBGElegal": "the legal BGE-M3 inside the ranker", "LegBGE": "legal BGE-M3 over its base",
    "LegEuler": "legal 8B over its base", "FtBGE": "fine-tuned BGE-M3 over its base",
    "FtEsmall": "fine-tuned E5-small over its base", "FtAdapter": "adapter over the frozen 8B",
    "HyLinQbig": "weighted sum with the 8B", "HyRRFQbig": "rank fusion with the 8B", "HyRRFdocvec": "rank fusion with doc2vec",
    "HyRRFldasim": "rank fusion with the topic similarity", "HyLinDocvec": "weighted sum with doc2vec",
    "HyLinPop": "weighted sum with popularity", "HyLinQsmall": "weighted sum with the 0.6B",
    "LeLM": "LambdaMART against the boosted trees", "LeLMnoemb": "LambdaMART without embeddings",
    "LeLin": "linear learner against the boosted trees", "LeRRF": "rank fusion over all sources",
    "LeRRFnoemb": "rank fusion over all sources without embeddings", "StLLMext": "LLM stage over the integrated method",
    "StLLMzero": "LLM stage over the ranker without embeddings", "StBGE": "published BGE reranker",
    "StLLMzeroVsFull": "LLM stage over the ranker without embeddings against the ranker with them",
    "AblQbig": "integrated method without the 8B", "AblDense": "ranker without its dense scores",
    "AblMemory": "ranker without its vector-keyed past citations", "AblColbert": "integrated method without late interaction",
    "Claude": "Claude over the integrated method",
}
CLASS = {"useful":"clears the margin", "nonzero_below_margin": "a gain below 0.02",
         "equivalent": "no meaningful difference", "inconclusive": "too uncertain to call"}
FAMILY_SIZE = {"F2": "LadderFamily", "F3": "AdaptFamily", "F4": "HybridFamily", "F5": "LearnerFamily", "F6": "StageFamily"}
RUNNING_EXAMPLE = "sjf:tesis:2028281:hechos"
# the example of 6.3, moved from the examples appendix (decision 6): an insolvency case whose article no past case cites
INSOLVENCY_EXAMPLE = "sjf:tesis:2023843:hechos"


def f3(x: float) -> str:
    """0.196 style: three decimals with the leading zero (D149)."""
    return f"{x:.3f}"


def sg(x: float) -> str:
    s = f3(abs(x))
    return ("$+$" if x >= 0 else "$-$") + s


def pct(x: float) -> str:
    return f"{100 * x:.0f}\\%"


def oracle_at(pc: pd.DataFrame, k: int = 30) -> float:
    """nDCG@10 if every cited article within the first k were moved to the top, in cited order (component-macro)."""
    vals = []
    for idx, ranks, nrel in zip(pc["positive_idx"], pc["positive_ranks"], pc["n_relevant"]):
        rk = F._ints(ranks)
        hits = sum(1 for r in rk if r <= k)
        z = sum(1 / np.log2(1 + r) for r in range(1, min(int(nrel), 10) + 1))
        dcg = sum(1 / np.log2(1 + r) for r in range(1, min(hits, 10) + 1))
        vals.append(dcg / z if z else 0.0)
    d = pc.assign(o=vals)
    g = d.groupby("component_id")["o"].mean()
    return float(g.mean())


def build(store: Path = STORE, analysis: Path = ANALYSIS) -> dict:
    runs = store / "runs"
    m: dict[str, str] = {}
    notes: list[str] = []
    # the D108 hold list guards the RUN-A-era stores only; the final record has every rerun merged
    final = store.resolve().is_relative_to(F.RECORD.resolve())
    held = (lambda sid: False) if final else F.held

    def load_contrast(fname: str):
        for d in ([BOOT_100K] if final else []) + [store / "contrasts"] + (EXTRA_CONTRASTS if final else []):
            p = d / f"{fname}.json"
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        if "__vs__" in fname:                      # a single contrast kept inside a family file (final store)
            a, b = fname.rsplit("__", 1)[0].split("__vs__")
            for p in sorted(((BOOT_100K if final else store / "contrasts")).glob("F*.json")):
                c = json.loads(p.read_text(encoding="utf-8")).get("contrasts", {}).get(f"{a} - {b}")
                if c:
                    return c
        return None

    def summ(sid):
        p = runs / sid / "DEV" / "summary.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def pend_sys(sid: str) -> str:      # plain words in print, never a run id (fixes_ch6_ch7_fresh_1, WRITE-B 1)
        return "\\EvPending{" + (F.NAME.get(sid, "this system") + ", pending") + "}"

    for k, sid in SYSTEMS.items():
        s = None if held(sid) else summ(sid)
        m["sx" + k] = f3(s["ndcg_at_10"]) if s else pend_sys(sid)
        if not s:
            notes.append(f"{sid} absent")
    for k in MEASURED:
        s = None if held(SYSTEMS[k]) else summ(SYSTEMS[k])
        for pre, key in MEASURES.items():
            m[pre + k] = f3(s[key]) if s and s.get(key) is not None else pend_sys(SYSTEMS[k])
    # contrasts
    cache = {}
    for k, (fname, key) in CONTRASTS.items():
        if fname not in cache:
            cache[fname] = load_contrast(fname)
        d = cache[fname]
        c = None if d is None else (d["contrasts"].get(key) if key else d)
        if key and any(held(x.strip()) for x in key.split(" - ")):
            c = None
            notes.append(f"contrast {key} held until RUN-B reruns it")
        if c is None:
            label = PLAIN.get(k, "comparison pending")
            for pre, what in (("cd", "gain"), ("ci", "interval"), ("cl", "adjusted lower end"), ("cn", "90\\% interval"), ("cc", "class"), ("cm", "family size")):
                m[pre + k] = "\\EvPending{" + f"{what}: {label}" + "}"
            notes.append(f"contrast {fname}:{key} absent")
            continue
        m["cd" + k] = sg(c["delta"])
        m["ci" + k] = f"[{sg(c['ci95'][0])}, {sg(c['ci95'][1])}]"
        m["cl" + k] = sg(c["adjusted_lower"])
        m["cn" + k] = f"[{sg(c['ci90'][0])}, {sg(c['ci90'][1])}]"   # the 90% interval that decides 'no meaningful difference'
        m["cc" + k] = "\\emph{" + CLASS[c["class"]] + "}"   # class names in italics, as in 4.4 (R review 1.3)
        m["cm" + k] = str(c.get("family_size", 1))   # m of the adjusted lower end (appendix B.3)
    for fname, name in FAMILY_SIZE.items():
        d = cache.get(fname) or load_contrast(fname)
        m["n" + name] = str(d["size"])
    # population
    s0 = summ("S0")
    m["nDevCases"], m["nDevGroups"] = f"{s0['n_cases']:,}", f"{s0['n_groups']:,}"
    # citation bands (c3 pairs, fold-local counts), recall@20 per band and shares of pairs and of the gain
    pairs = pd.read_csv(analysis / "c3" / "c3_pairs.csv")[["query_id", "article", "band"]]
    pcs = {n: pd.read_parquet(runs / sid / "DEV" / "per_case.parquet") for n, sid in
           (("S0", "S0"), ("ext", "S_full_ext"), ("q8", "qwen3_8b"), ("lex", "lstar_headings"), ("llm", "llm_rerank_S_full_ext"))}
    word = {"0": "Never", "1-2": "Few", "3-10": "Some", ">10": "Many"}
    for n in ("S0", "ext", "q8", "lex"):
        h = F._pair_hits(pcs[n]).merge(pairs, on=["query_id", "article"])
        r = h.groupby("band")["hit"].mean()
        for b, w in word.items():
            m[f"band{w}{ {'S0': 'Zero', 'ext': 'Ext', 'q8': 'Qbig', 'lex': 'Lex'}[n] }".replace(" ", "")] = f3(r.get(b, np.nan))
    share_pairs = pairs["band"].value_counts(normalize=True)
    g = pcs["S0"][["query_id", "ndcg_at_10"]].merge(pcs["ext"][["query_id", "ndcg_at_10"]], on="query_id", suffixes=("_0", "_f"))
    g["gain"] = g["ndcg_at_10_f"] - g["ndcg_at_10_0"]
    alloc = pairs.merge(g[["query_id", "gain"]], on="query_id")
    alloc["w"] = alloc["gain"] / alloc.groupby("query_id")["article"].transform("count")
    share_gain = alloc.groupby("band")["w"].sum() / alloc["w"].sum()
    for b, w in word.items():
        m[f"bandPairs{w}"] = pct(share_pairs.get(b, 0))
        m[f"bandGain{w}"] = pct(share_gain.get(b, 0))
    # recall at depth (article level) from the sweep
    rs = pd.read_csv(analysis / "recall" / "recall_sweep.csv")
    for n, sid in (("Ext", "S_full_ext"), ("Zero", "S0"), ("Lex", "lstar_headings"), ("Qbig", "qwen3_8b")):
        d = rs[(rs["system"] == sid) & (rs["level"] == "article") & (rs["measure"] == "recall")]
        for k, w in ((10, "Ten"), (30, "Thirty"), (100, "Hundred"), (200, "TwoHundred")):
            v = d[d["k"] == k]["value"]
            m[f"rec{w}{n}"] = f3(float(v.iloc[0])) if len(v) else "\\EvPending{recall}"
    # ceilings: a perfect reorder of the first 30, and the law oracle of C1
    for n in ("S0", "ext"):
        m["oracleThirty" + {"S0": "Zero", "ext": "Ext"}[n]] = f3(oracle_at(pcs[n], 30))
    c1 = pd.read_csv(analysis / "c1" / "c1_systems.csv").set_index("system")
    for n, sid in (("Zero", "S0"), ("Ext", "S_full_ext"), ("Lex", "lstar_headings"), ("Qbig", "qwen3_8b"),
                   ("LLM", "llm_rerank_S_full_ext")):
        m["lawOracle" + n] = f3(c1.loc[sid, "oracle_ndcg_at_10"]) if sid in c1.index else "\\EvPending{law oracle}"
        m["withinLaw" + n] = f3(c1.loc[sid, "within_law_at_1_cond"]) if sid in c1.index else "\\EvPending{within law}"
    # the running example and the insolvency example of 6.3: rank of the cited article
    for n, sid in (("Lex", "lstar_headings"), ("Zero", "S0"), ("Qbig", "qwen3_8b"), ("Full", "S_full"),
                   ("Ext", "S_full_ext"), ("LLM", "llm_rerank_S_full_ext")):
        pc = pd.read_parquet(runs / sid / "DEV" / "per_case.parquet")
        for pre, qid in (("exRank", RUNNING_EXAMPLE), ("exbRank", INSOLVENCY_EXAMPLE)):
            row = pc[pc["query_id"] == qid]
            m[pre + n] = (f"{min(F._ints(row['positive_ranks'].iloc[0])):,}" if len(row) else "\\EvPending{rank}")
    # reproduction of the frozen rankers by a rebuild from code (N3)
    for n in ("Noemb", "Emb"):
        a, b = summ({"Noemb": "ltr_noemb_frozen@v11", "Emb": "ltr_emb_frozen@v11"}[n]), \
            summ({"Noemb": "ltr_noemb_refit@rebuilt@v11", "Emb": "ltr_emb_refit@rebuilt@v11"}[n])
        m["rebuildGap" + n] = f3(abs(a["ndcg_at_10"] - b["ndcg_at_10"]))
    return {"macros": m, "notes": notes}


def write(res: dict, out: Path = OUT, store: Path = STORE, analysis: Path = ANALYSIS) -> Path:
    lines = [
        "% Chapter 6 evidence macros, generated by uol.thesis_figures.ch6_numbers; do not edit by hand.",
        f"% Store: {C.rel(store)} (the final record, D131); analyses: {C.rel(analysis)}.",
        "% Contrasts: <record>/supplement/bootstrap_100k/contrasts/*.json (100,000 paired draws over groups, D138), else"
        " <store>/contrasts/*.json; classes by the rule of section 4.4.",
    ]
    lines += [f"% Note: {n}" for n in res["notes"]]
    for k, v in res["macros"].items():
        if v.startswith("\\EvPending{"):          # store ids in a pending label: escape their underscores
            v = "\\EvPending{" + v[len("\\EvPending{"):-1].replace("_", "\\_") + "}"
        lines.append(f"\\providecommand{{\\{k}}}{{{v}}}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--analysis", type=Path, default=ANALYSIS)
    a = ap.parse_args(argv)
    res = build(a.store, a.analysis)
    p = write(res, OUT, a.store, a.analysis)
    print(p, len(res["macros"]), "macros;", res["notes"])


if __name__ == "__main__":
    main()
