"""D191 (1 Nov 2026): the numbers and figures of the one-argument Chapter 6 that the earlier generators lack.

Reads only saved outputs: the final record's store, the final-system audit's store, the LLM sweep's store and, for
results measured on an earlier form of the ranker, the Round 3 and gap-test records through ``d61_figures``. No model
runs and no test-set row is read.

Conventions, as in the record: a score is the mean over groups (``component_id``) of the case means; a difference is the
same mean of paired differences, with a 95% unadjusted group bootstrap; a recall over cited pairs is pooled over pairs.

Output: ``thesis/evidence/numbers_d191.tex`` and the figures F6_12_beyond, F6_13_map, F6_14_controls, F6_15_llm_alone.

    py -3.12 -m uol.thesis_figures.ch6_d191      # with PYTHONPATH=src
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from uol.thesis_figures import ch6_figs as F
from uol.thesis_figures import common as C
from uol.thesis_figures import d61_figures as D61
from uol.thesis_figures.ch6_figs import BLUE, GREEN, INK, INK2, MARK, MUTED, OCHRE, PLUM, plt

STORE = F.RECORD_STORE
FINAL = F.FINAL_STORE
SWEEP = C.ROOT / "experiments" / "runs" / "llm_sweep_20261028" / "report" / "store" / "runs"
DECOMP = F.RECORD_ANALYSIS / "c1" / "c1_decomposition.csv"
OUT = C.ROOT / "thesis" / "evidence" / "numbers_d191.tex"
SEED, N_BOOT = 20261101, 10000
FAR = 10**6
RANKER, SYSTEM, LEX = F.INTEGRATED, F.SYSTEM, "lstar_headings"
# v33 (D203): what each encoder finds is set against the eight-source ranker and that ranker without its encoder
FOOT = C.ROOT / "experiments" / "runs" / "footing_20261101" / "dev" / "store" / "runs"
RANKER_LEAN, NOENC_LEAN = "rt_lm_tesis", "rk_noenc"


def per_case(system: str, pop: str = "DEV") -> pd.DataFrame:
    for root in ((FOOT,) if system in (RANKER_LEAN, NOENC_LEAN) else ()) + (FINAL if system in (RANKER, SYSTEM) else STORE, STORE, SWEEP):
        p = root / system / pop / "per_case.parquet"
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"{system} {pop}")


def macro(s: pd.Series, groups: pd.Series) -> float:
    return float(s.groupby(groups).mean().mean())


def gain(base: pd.DataFrame, other: pd.DataFrame) -> tuple[float, float, float]:
    """other minus base on the cases both scored: group mean and 95% group bootstrap."""
    m = base[["query_id", "component_id", "ndcg_at_10"]].merge(other[["query_id", "ndcg_at_10"]], on="query_id",
                                                               suffixes=("_a", "_b"))
    v = (m["ndcg_at_10_b"] - m["ndcg_at_10_a"]).groupby(m["component_id"]).mean().to_numpy()
    rng = np.random.default_rng(SEED)
    means = v[rng.integers(0, len(v), size=(N_BOOT, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def pairs(system: str) -> pd.DataFrame:
    """One row per cited pair: its case, group, article and rank (FAR when it lies outside the stored ranking)."""
    rows = []
    for r in per_case(system).itertuples():
        idx = [x for x in re.split(r"[,\s;|]+", str(r.positive_idx)) if x]
        ranks = [x for x in re.split(r"[,\s;|]+", str(r.positive_ranks)) if x]
        for i, k in zip(idx, ranks):
            try:
                rk = int(float(k))
            except ValueError:
                rk = FAR
            rows.append((r.query_id, r.component_id, i, rk if rk > 0 else FAR))
    return pd.DataFrame(rows, columns=["query_id", "component_id", "article", "rank"])


ENCODERS = [  # (system, name in the figure, colour)
    ("qwen3_8b", "Qwen3-Embedding-8B", GREEN),
    ("euler_legal", "legal Qwen3-Embedding-8B", MARK),
    ("qwen8b_adapter", "Qwen3-Embedding-8B, task-trained adapter", MARK),
    ("ft_e5_q", "E5-small, task-trained", MARK),
    ("e5_small", "E5-small", MARK),
    ("bge_m3", "BGE-M3, one vector", PLUM),
    ("bge_m3_sparse", "BGE-M3, sparse", PLUM),
    ("bge_m3_colbert", "BGE-M3, late interaction", PLUM),
    ("bge_m3_legal", "legal BGE-M3", MARK),
    ("alia_mrbert", "ALIA-MrBERT", MARK),
]


def beyond_table() -> tuple[pd.DataFrame, dict]:
    """For each encoder: recall@20 where lexical matching misses the article, and what it finds that others miss."""
    key = ["query_id", "article"]
    lex = pairs(LEX).rename(columns={"rank": "lex"})
    rk = pairs(RANKER_LEAN).rename(columns={"rank": "ranker"})[key + ["ranker"]]
    qw = pairs("qwen3_8b").rename(columns={"rank": "qwen"})[key + ["qwen"]]
    s0 = pairs(NOENC_LEAN).rename(columns={"rank": "s0"})[key + ["s0"]]
    base = lex.merge(rk, on=key).merge(qw, on=key).merge(s0, on=key)
    assert len(base) == len(lex) == len(rk), "the systems must score the same cited pairs"
    miss = base[base["lex"] > 100]
    rows, rng = [], np.random.default_rng(SEED)
    for sid, name, col in ENCODERS:
        p = pairs(sid).rename(columns={"rank": "r"})[key + ["r"]].merge(base, on=key)
        assert len(p) == len(base), sid
        found = p[p["r"] <= 20]
        m = p[p["lex"] > 100]
        g = (m["r"] <= 20).groupby(m["component_id"]).agg(["sum", "count"]).to_numpy(float)
        idx = rng.integers(0, len(g), size=(N_BOOT, len(g)))
        boots = g[idx, 0].sum(1) / g[idx, 1].sum(1)
        rows.append(dict(system=sid, name=name, colour=col, found=len(found),
                         recall_miss=float((m["r"] <= 20).mean()),
                         lo=float(np.quantile(boots, .025)), hi=float(np.quantile(boots, .975)),
                         also_qwen=float((found["qwen"] <= 20).mean()),
                         also_s0=float((found["s0"] <= 20).mean()),
                         beyond_s0=int((found["s0"] > 20).sum()),
                         beyond_ranker=int((found["ranker"] > 20).sum())))
    meta = dict(pairs=len(base), lex_miss=len(miss), ranker_misses=int((base["ranker"] > 20).sum()),
                ranker_found=int((base["ranker"] <= 20).sum()))
    return pd.DataFrame(rows).set_index("system"), meta


def within(df: pd.DataFrame) -> tuple[float, float]:
    """Law@1 and the conditional within-law@1 as C1 computes them (group means)."""
    cond = df["success_at_1"].where(df["law_at_1"] == 1)
    return macro(df["law_at_1"], df["component_id"]), float(cond.groupby(df["component_id"]).mean().mean())


def f3(x: float, sign: bool = False) -> str:
    s = f"{abs(x):.3f}"
    if sign:
        return ("$+$" if x >= 0 else "$-$") + s
    return s


def interval(lo: float, hi: float) -> str:
    return f"[{f3(lo, True)}, {f3(hi, True)}]"


def write_macros(tab: pd.DataFrame, meta: dict) -> None:
    dec = pd.read_csv(DECOMP).set_index("system")
    summ = lambda s: json.loads((STORE / s / "DEV" / "summary.json").read_text(encoding="utf-8"))  # noqa: E731
    # within() reproduces the record before it is used on the sample
    w = within(per_case("S_full_ext"))
    assert abs(w[1] - dec.at["S_full_ext", "within_law_at_1_cond"]) < 1e-5, w
    ext = per_case("S_full_ext")
    jev, list_reader = per_case("jev_S_full_ext", "DEV-A300"), per_case("claude_rerank_S_full_ext", "DEV-A300")
    ext_s = ext[ext["query_id"].isin(jev["query_id"])]
    L = [("wyPairs", f"{meta['pairs']:,}"), ("wyLexMiss", f"{meta['lex_miss']:,}"),
         ("wyRankerMisses", str(meta["ranker_misses"])),
         ("wyRMissQwen", f3(tab.at["qwen3_8b", "recall_miss"])), ("wyRMissDense", f3(tab.at["bge_m3", "recall_miss"])),
         ("wyRMissSparse", f3(tab.at["bge_m3_sparse", "recall_miss"])),
         ("wyRMissLate", f3(tab.at["bge_m3_colbert", "recall_miss"])),
         ("wyLateAlsoQwen", f"{tab.at['bge_m3_colbert', 'also_qwen'] * 100:.0f}\\%"),
         ("wyLateBeyondRanker", str(tab.at["bge_m3_colbert", "beyond_ranker"])),
         ("wyAdaptBeyondZero", str(tab.at["qwen8b_adapter", "beyond_s0"])),
         ("wyQwenBeyondZero", str(tab.at["qwen3_8b", "beyond_s0"])),
         ("wyFtAlsoZero", f"{tab.at['ft_e5_q', 'also_s0'] * 100:.0f}\\%"),
         ("wyQwenAlsoZero", f"{tab.at['qwen3_8b', 'also_s0'] * 100:.0f}\\%"),
         ("wyFtBeyondZero", str(tab.at["ft_e5_q", "beyond_s0"])),
         ("wyNeverNoDirect", f3(summ("S_full_minus_D")["unseen_recall_at_20_pooled"])),
         ("wyNeverNoPast", f3(summ("S_full_minus_M")["unseen_recall_at_20_pooled"])),
         ("wyNeverEvery", f3(summ("S_full")["unseen_recall_at_20_pooled"])),
         ("wyNeverZero", f3(summ("S0")["unseen_recall_at_20_pooled"])),
         ("wyNeverDocexp", f3(summ("docexp_bm25")["unseen_recall_at_20_pooled"])),
         ("wyBgeLaw", f3(dec.at["bge_m3", "law_at_1"])), ("wyBgeLegalLaw", f3(dec.at["bge_m3_legal", "law_at_1"])),
         ("wyBgeWithin", f3(dec.at["bge_m3", "within_law_at_1_cond"])),
         ("wyBgeLegalWithin", f3(dec.at["bge_m3_legal", "within_law_at_1_cond"])),
         ("wyQbLaw", f3(dec.at["qwen3_8b_int8_base", "law_at_1"])), ("wyQbLegalLaw", f3(dec.at["euler_legal", "law_at_1"])),
         ("wyQbWithin", f3(dec.at["qwen3_8b_int8_base", "within_law_at_1_cond"])),
         ("wyQbLegalWithin", f3(dec.at["euler_legal", "within_law_at_1_cond"])),
         ("wyEfiveLaw", f3(dec.at["e5_large", "law_at_1"])), ("wyEfiveLegalLaw", f3(dec.at["e5_large_legal_en", "law_at_1"])),
         ("wyEfiveWithin", f3(dec.at["e5_large", "within_law_at_1_cond"])),
         ("wyEfiveLegalWithin", f3(dec.at["e5_large_legal_en", "within_law_at_1_cond"]))]
    for name, df in (("Base", ext_s), ("Judge", jev), ("List", list_reader)):
        law, cond = within(df)
        L += [(f"wySample{name}Law", f3(law)), (f"wySample{name}Within", f3(cond))]
    # Earlier form of the ranker: EVIDENCE.md rows, by the name the text gives them (d61_figures asserts most of them).
    L += [("eaCases", "1,218"), ("eaRanker", "0.472"),
          ("eaStruct", "$-$0.002"), ("eaStructCI", "[$-$0.007, $+$0.003]"), ("eaCoCite", "$-$0.001"),
          ("eaCoCiteCI", "[$-$0.007, $+$0.006]"), ("eaBridge", "$+$0.018"),
          ("eaSmooth", "$+$0.004"), ("eaSmoothCI", "[$-$0.003, $+$0.011]"),
          ("eaAdapter", "$-$0.004"), ("eaAdapterCI", "[$-$0.011, $+$0.004]"), ("eaIssues", "$+$0.020"), ("eaIssuesCI", "[$+$0.011, $+$0.030]"),
          ("eaIssuesCtl", "$-$0.004"), ("eaPatterns", "$+$0.001"), ("eaPatternsCI", "[$-$0.004, $+$0.007]"),
          ("eaFeedback", "$-$0.009"), ("eaFeedbackCI", "[$-$0.015, $-$0.004]"),
          ("eaListwise", "$+$0.002"), ("eaListwiseCI", "[$-$0.007, $+$0.012]"),
          ("eaPilotInModel", "0.355"), ("eaPilotAdapter", "0.391"), ("eaPilotFrozen", "0.260"),
          ("eaPilotGap", "$-$0.036"), ("eaPilotGapCI", "[$-$0.067, $-$0.005]"),
          ("eaPilotUnseenIn", "0.445"), ("eaPilotUnseenAdapter", "0.335"), ("eaPilotUnseenFrozen", "0.373"),
          ("eaAdapterSeedMean", "$+$0.001"),
          # later batch of the test set, read from the record and not scored again (EVIDENCE.md, Round 3 lockbox batch)
          ("tsbMem", "$+$0.029"), ("tsbMemCI", "[$+$0.010, $+$0.048]"), ("tsbAdapter", "$+$0.025"),
          ("tsbAdapterCI", "[$+$0.011, $+$0.040]"), ("tsbStruct", "$+$0.005"), ("tsbStructCI", "[$-$0.006, $+$0.017]"),
          ("tsbCoCite", "$+$0.006"), ("tsbCoCiteCI", "[$-$0.004, $+$0.016]"), ("tsbCutGroups", "145"),
          ("tsbListwise", "$+$0.051"), ("tsbListwiseCI", "[$+$0.024, $+$0.080]"), ("tsbIssues", "$+$0.017"),
          ("tsbIssuesCI", "[$-$0.003, $+$0.037]"), ("tsbPatterns", "$-$0.017"), ("tsbPatternsCI", "[$-$0.038, $+$0.002]")]
    # What a language model names from the facts alone (EVIDENCE.md QK row, as d61_figures prints it).
    L += [("wyNamedArticles", "618"), ("wyNamedCases", "200"), ("wyNamedCited", "25\\%"), ("wyNamedNear", "16\\%"),
          ("wyNamedOtherLaw", "27\\%"), ("wyNamedMissing", "2.4\\%")]
    # The two batches of the test set (LOCKBOX_LEDGER.jsonl; REPORT.md of round3_best_case_20261013).
    L += [("tsFirstEmb", "0.518"), ("tsFirstNoemb", "0.436")]
    head = ["% D191 (1 Nov 2026): generated by uol.thesis_figures.ch6_d191; do not edit by hand.",
            "% wy*: stored development results, final record. ea*: an earlier form of the ranker (EVIDENCE.md)."]
    OUT.write_text("\n".join(head + [f"\\providecommand{{\\{k}}}{{{v}}}" for k, v in L]) + "\n", encoding="utf-8")


def fig_beyond(ctx, tab: pd.DataFrame, meta: dict) -> Path:
    stem, label = "F6_12_beyond", "fig:F6_12_beyond"
    cap = (f"What each encoder alone adds on the development cases, counted per cited pair. (a) The share found within "
           f"its first twenty, among the {meta['lex_miss']:,} cited pairs that lexical matching leaves outside its first "
           f"hundred, with 95\\% intervals. (b) Cited pairs it finds that the ranker misses, of the "
           f"{meta['ranker_misses']} the ranker leaves outside its first twenty.")
    fig = F._new(2.95)
    a = fig.add_axes([0.385, 0.19, 0.30, 0.72])
    b = fig.add_axes([0.725, 0.19, 0.25, 0.72])
    ys = list(range(len(tab)))[::-1]
    for (sid, r), y in zip(tab.iterrows(), ys):
        a.plot([r["lo"], r["hi"]], [y, y], color=r["colour"], lw=1.0, zorder=2)
        a.plot([r["recall_miss"]], [y], "o", ms=4.2, color=r["colour"], mec="white", mew=0.4, zorder=3)
        b.barh(y, r["beyond_ranker"], height=0.62, color=r["colour"], lw=0)
        b.text(r["beyond_ranker"] + 3, y, str(r["beyond_ranker"]), va="center", fontsize=7, color=INK)
    a.set_yticks(ys)
    a.set_yticklabels(tab["name"], fontsize=7, color=INK)
    a.tick_params(axis="y", length=0, pad=3)
    a.spines["left"].set_visible(False)
    b.set_yticks([])
    b.spines["left"].set_visible(False)
    a.set_xlim(0, 0.4)
    a.set_xticks([0, .1, .2, .3, .4])
    b.set_xlim(0, 160)
    b.set_xticks([0, 50, 100, 150])
    C.strip0(a, "x", nd=1)
    a.set_xlabel("Recall@20 where lexical\nmatching misses")
    b.set_xlabel("cited pairs\nbeyond the ranker")
    C.panel(a, "a", "", x=-0.02, pad=2)
    C.panel(b, "b", "", x=-0.02, pad=2)
    return F._write(fig, ctx, stem, cap, "What each encoder adds", label)


def fig_map(ctx) -> Path:
    stem, label = "F6_13_map", "fig:F6_13_map"
    cap = ("The alternatives evaluated, grouped by the part of the system it would change, as the change in nDCG@10 it makes to the "
           "system named in each heading, with 95\\% intervals. Solid marks are on the development cases and "
           "green marks the parts of the system. Open marks were measured on the 1,218-case ranker with every "
           "encoder and are not directly comparable. The band marks changes smaller than the margin of 0.02.")
    s0, rk, ext = per_case("S0"), per_case(RANKER), per_case("S_full_ext")
    S = lambda s, pop="DEV": per_case(s, pop)  # noqa: E731
    groups = [
        ("Search, added to the ranker without embeddings", [
            ("Qwen3-Embedding-8B, with its past-case sources", gain(s0, rk), "yes"),
            ("Qwen3-Embedding-8B, its dense score only", gain(s0, S("S0+qwen3_8b_dense")), ""),
            ("legal Qwen3-Embedding-8B", gain(s0, S("S0+euler_legal")), ""),
            ("E5-small, task-trained", gain(s0, S("S0+ft_e5_q_dense")), ""),
            ("BGE-M3, one vector", gain(s0, S("S0+bge_m3_dense")), ""),
            ("BGE-M3, sparse", gain(s0, S("S0+bge_m3_sparse")), ""),
            ("BGE-M3, late interaction", gain(s0, S("S0+bge_m3_colbert")), ""),
            ("E5-small", gain(s0, S("S0+e5_small_dense")), ""),
            ("query2doc rewriting, through Qwen3-Embedding-8B", gain(s0, S("S0+query2doc")), ""),
            ("HyDE rewriting, through Qwen3-Embedding-8B", gain(s0, S("S0+hyde")), "")]),
        ("Added to the ranker", [
            ("BGE-M3, E5-small and late interaction", gain(rk, S("S_full")), ""),
            ("those, and the vote of the unstructured tesis", gain(rk, ext), "")]),
        ("Added to the 1,218-case ranker with every encoder", [
            ("statute structure", (-.002, -.007, .003), "old"),
            ("co-citation", (-.001, -.007, .006), "old"),
            ("article vectors smoothed over the statute", (.004, -.003, .011), "old"),
            ("adapter on Qwen3-Embedding-8B", (-.004, -.011, .004), "old"),
            ("listwise training objective", (.002, -.007, .012), "old"),
            ("language model writes fact patterns", (.001, -.004, .007), "old"),
            ("language model restates the facts as issues", (.020, .011, .030), "old")]),
        ("Reread of the ranker", [("Claude Opus 5.5", gain(rk, S(SYSTEM)), "yes")]),
        ("Reread of the largest ranker, on the sample", [
            ("GPT-6 Sol", gain(ext, S("gpt6sol_rerank_S_full_ext", "DEV-A300")), ""),
            ("Claude Opus 5.5", gain(ext, S("claude_rerank_S_full_ext", "DEV-A300")), ""),
            ("GPT-5.6", gain(ext, S("llm_rerank_S_full_ext").pipe(
                lambda d: d[d["query_id"].isin(S("claude_rerank_S_full_ext", "DEV-A300")["query_id"])])), ""),
            ("Claude Haiku 5.5", gain(ext, S("haiku_rerank_S_full_ext", "DEV-A300")), ""),
            ("Claude Sonnet 5.5", gain(ext, S("sonnet_rerank_S_full_ext", "DEV-A300")), ""),
            ("pointwise judge", gain(ext, S("jev_S_full_ext", "DEV-A300")), "")]),
        ("Reread of the largest ranker, by a trained or published reranker", [
            ("Qwen3-Reranker-0.6B, trained on the citations", gain(ext, S("ce_trained")), ""),
            ("Qwen3-Reranker-4B", gain(ext, S("rerank_qwen3_4b")), ""),
            ("BGE reranker", gain(ext, S("rerank_bge")), "")]),
    ]
    rows, heads, y = [], [], 0.0
    for head, items in groups:
        heads.append((head, y))
        y -= 1.0
        for name, (d, lo, hi), kind in items:
            rows.append((y, name, d, lo, hi, kind))
            y -= 1.0
        y -= 0.45
    fig = F._new(4.95)
    ax = fig.add_axes([0.47, 0.08, 0.51, 0.91])
    ax.axvspan(-0.02, 0.02, color=C.FILL, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    for yy, name, d, lo, hi, kind in rows:
        col = GREEN if kind == "yes" else INK2
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.0, solid_capstyle="butt", zorder=2)
        ax.plot([d], [yy], "o", ms=4.4 if kind != "old" else 3.8, mfc="white" if kind == "old" else col, mec=col,
                mew=0.7 if kind == "old" else 0.4, zorder=3)
        ax.text(hi + 0.003, yy, C.f3(d, True), fontsize=7, color=col, va="center", ha="left")
        ax.text(-0.015, yy, name.replace("tesis", "$\\it{tesis}$"), transform=ax.get_yaxis_transform(), fontsize=7,
                color=INK, ha="right", va="center")
    for head, yy in heads:
        ax.text(-0.905, yy, head, transform=ax.get_yaxis_transform(), fontsize=7.5, color=INK2, style="italic",
                ha="left", va="center")
    ax.set_ylim(y + 0.8, 0.7)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.08, 0.185)
    ax.set_xticks([-0.05, 0, 0.05, 0.1])
    ax.set_xticklabels(["−0.05", "0", "+0.05", "+0.10"])
    ax.set_xlabel(C.axis_title("change in nDCG@10"))
    ctx.notes.setdefault(stem, []).append("; ".join(f"{n} {d:+.4f}" for _, n, d, _, _, _ in rows))
    return F._write(fig, ctx, stem, cap, "The alternatives evaluated", label)


def fig_controls(ctx) -> Path:
    stem, label = "F6_14_controls", "fig:F6_14_controls"
    cap = ("The change in nDCG@10 from three additions to an earlier form of the ranker with every encoder, on the "
           "development cases (diamonds), each beside the same addition built from randomised data (grey). Statute "
           "structure and co-citation score inside their random controls. The vote of the older \\emph{tesis} lies far "
           "outside its own, so the test can detect a real signal.")
    rng = np.random.default_rng(7)
    fig = F._new(2.15)
    ax = fig.add_axes([0.37, 0.2, 0.61, 0.76])
    ax.axvspan(-0.02, 0.02, color=C.FILL, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.6, zorder=1)
    names = {"S2": ("statute structure", "150 random statute graphs"), "S3": ("co-citation", "100 random citation graphs"),
             "M1": ("vote of the older tesis", "50 shuffles within each year")}
    for (arm, _, _, keys, real_txt), y in zip(D61.NULL_ROWS, (2, 1, 0)):
        r = C.load_json(f"{C.R3_ARMS}/{arm}/RESULT.json")
        draws = np.concatenate([np.asarray(r["nulls"][k]["deltas"], float) for k in keys])
        real = float(r["dev"]["vs_EMB"]["P_spec"]["delta"])
        assert C.f3(real, True) == real_txt, (arm, real)
        ax.scatter(draws, y + rng.uniform(-0.17, 0.17, len(draws)), s=6, color=MARK, lw=0, zorder=2)
        col = GREEN if arm == "M1" else INK
        ax.plot([real], [y], "D", ms=5.5, color=col, mec="white", mew=0.6, zorder=4)
        ax.annotate(f"real {real_txt}", (real, y), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7,
                    color=col)
        name, what = names[arm]
        ax.text(-0.012, y + 0.12, name.replace("tesis", "$\\it{tesis}$"), transform=ax.get_yaxis_transform(), fontsize=7.5,
                color=INK, ha="right", va="center")
        ax.text(-0.012, y - 0.22, what, transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, ha="right",
                va="center")
    ax.set_ylim(-0.55, 2.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(-0.022, 0.026)
    ax.set_xticks([-0.02, -0.01, 0, 0.01, 0.02])
    ax.set_xticklabels(["−0.02", "−0.01", "0", "+0.01", "+0.02"])
    ax.set_xlabel(C.axis_title("change in nDCG@10"))
    return F._write(fig, ctx, stem, cap, "Real signals against random controls", label)


def fig_llm_alone(ctx) -> Path:
    stem, label = "F6_15_llm_alone", "fig:F6_15_llm_alone"
    cap = "The kind of article a language model names from the facts alone, with no search, for 200 development cases."
    named = [("cited by the court", 25, GREEN), ("next to a cited article or in its section", 16, PLUM),
             ("elsewhere in a cited law", 23, PLUM), ("another federal law", 27, MARK), ("state law", 4.4, MARK),
             ("repealed or transitory", 1.3, MUTED), ("does not exist", 2.4, OCHRE)]
    fig = F._new(1.75)
    axa = fig.add_axes([0.42, 0.14, 0.56, 0.82])
    ys = list(range(len(named)))[::-1]
    for (lab, v, col), y in zip(named, ys):
        axa.barh(y, v, height=0.62, color=col, lw=0)
        axa.text(v + 0.6, y, f"{v:0.0f}%" if v >= 10 else f"{v:0.1f}%", va="center", fontsize=7, color=INK)
    axa.set_yticks(ys)
    axa.set_yticklabels([x[0] for x in named], fontsize=7, color=INK)
    axa.tick_params(axis="y", length=0, pad=4)
    axa.spines["left"].set_visible(False)
    axa.set_xlim(0, 32)
    axa.set_xticks([0, 10, 20, 30])
    axa.set_xticklabels(["0", "10%", "20%", "30%"])
    return F._write(fig, ctx, stem, cap, "A language model without a search", label)


def main() -> None:
    tab, meta = beyond_table()
    write_macros(tab, meta)
    print(json.dumps(meta))
    print(tab.drop(columns=["colour", "name"]).round(3).to_string())
    for fn in (lambda c: fig_beyond(c, tab, meta), fig_map, fig_controls, fig_llm_alone):
        ctx = F.Ctx(store=STORE, analysis=F.RECORD_ANALYSIS, final=True, extra_contrasts=F.RECORD_CONTRASTS)
        print(fn(ctx), ctx.notes)


if __name__ == "__main__":
    main()
