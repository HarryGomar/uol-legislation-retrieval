"""The one record of results for the thesis (rebuild PLAN, section 7). Reads stores, receipts and contrast files and
writes three things, so that no number in the thesis is typed by hand:

* ``research/programs/final_results_20261102/results.json``: Benchmark (sets, counts, split checks, measured facts),
  System (final definition, settings, development and test numbers, cost, time) and Experiments (one entry per
  examiner question of the PLAN's section 4 and per alternative, with family, contrast and class);
* ``thesis/evidence/numbers_final.tex``: one macro per number; a value not yet held prints ``\\EvPending{waiting: ...}``;
* ``RESULTS.md`` beside ``results.json``: the tables, and the map from each macro to its file and key.

The measured facts and the per-case analyses need the release with the corrected labels (development cases only); they
are computed once into ``measured.json`` and reread after that (``--measure`` recomputes them). The test set is read only
from ``test/scored.json`` and the per-case files that ``uol.final.test_set score`` wrote; this script never scores it.

    py -3.12 scripts/final_results.py [--measure]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "runs" / "footing_20261101"
FOOT = RUN / "dev" / "store" / "runs"
AUDIT = ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev" / "store" / "runs"
RECORD = ROOT / "experiments" / "runs" / "final_20261025" / "store_dev_20261025b" / "runs"
WORK = ROOT / ".cache" / "footing"
OUT = ROOT / "research" / "programs" / "final_results_20261102"
LOGS = RUN / "logs"
RECEIPTS = RUN / "receipts"
CON = RUN / "dev" / "contrasts_100k"
TEST = RUN / "test_final"                    # the one test run of PLAN addendum 22
TEST_EARLIER = RUN / "test"                 # the frozen record of D198, a disclosed earlier use
TEX = ROOT / "thesis" / "evidence" / "numbers_final.tex"
MEASURED = OUT / "measured.json"

MEASURES = ("ndcg_at_10", "success_at_1", "success_at_10", "law_at_1", "law_at_10", "success_at_1_in_law", "unseen_recall_at_20_pooled",
            "recall_at_20", "recall_at_100")
SHORT = {"ndcg_at_10": "Ndcg", "success_at_1": "First", "success_at_10": "Ten", "law_at_1": "LawFirst", "law_at_10": "LawTen",
         "success_at_1_in_law": "Within", "unseen_recall_at_20_pooled": "Never", "recall_at_20": "Rtwenty", "recall_at_100": "Rhundred"}
# thesis name -> (store, system id). The system and its settings are read from the settings family below.
# D209, D210: the ranker is LambdaMART on the eight sources and the older-tesis votes (rt_lm_tesis); the system reads its list once, in forward order
RANKER, NOENC, REORD = "rt_lm_tesis", "rk_noenc", "gpt61sol_rerank_rt_lm_tesis"
SYSTEM = "gpt61sol_rk_system"
CHECK = {v: f"gpt61sol_rk_system_{v}" for v in ("merged", "both", "noenc", "low", "d60")}
CHECK["self"] = "gpt61sol_final_self"                       # the model's own list reranked: no ranker in it, so unchanged
HIGH = "gpt61sol_rk_system_high"
MAIN = {"Pop": (RECORD, "popularity"), "Lex": (RECORD, "lstar_headings"), "Enc": (RECORD, "qwen3_8b"), "Noenc": (FOOT, NOENC),
        "Rank": (FOOT, RANKER), "Reord": (FOOT, REORD), "Alone": (FOOT, "gpt61sol_alone"),
        "Combo": (FOOT, CHECK["both"])}
CELLS = {"Neither": (FOOT, "rk_neither"), "EncOnly": (FOOT, "rk_nocit"), "CitOnly": (FOOT, NOENC), "Both": (FOOT, RANKER)}
SETTINGS = ("merged", "self", "both", "noenc", "low", "d60")
ADDITIONS = ("ft_tesis", "ft_structure", "ft_cocite", "ft_adapter", "ft_smooth", "ft_issues", "ft_patterns", "ft_listwise")
MODELS = {"gpt61sol": "SolSixOne", "gpt61terra": "TerraSixOne", "gpt6solnw": "SolSix", "gpt56sol": "SolFive", "claude": "Opus",
          "gpt6luna": "LunaSix", "gpt56terra": "TerraFive", "luna": "LunaFive"}     # stored kind -> macro stem
SAMPLE = ("alone", "repeat", "examples", "list_agent_words", "agent_embedding", "list_agent_embedding", "combo", "sum_model_ranker", "sum_model_system",
          "alone_gpt61terra", "alone_gpt6solnw", "alone_gpt56sol", "alone_gpt6luna", "alone_gpt56terra", "alone_luna")
FLAGS = ("QUEUE_DONE", "REWRITE_DONE", "SMOOTH_DONE", "CLASSIFIER_DONE", "LEAN_DONE", "VARIANTS_DONE", "RERANKERS_DONE", "TIMING_RANKER_DONE",
         "ON_ft_tesis_DONE", "ON_S_full_ext_DONE", "CHAIN_DONE")
BATCH_SHARE = 0.5                     # the provider's Batch tier: half the standard rate on every token kind
EXAMPLE, EXAMPLE_B = "sjf:tesis:2028281:hechos", "sjf:tesis:2023843:hechos"

def _code_constants() -> dict:
    """The system's fixed settings, read from the code that runs it rather than typed."""
    agents = (ROOT / "src/uol/final/llm_agents.py").read_text(encoding="utf-8")
    prompt = (ROOT / "src/uol/round3/prompts/closedbook_rank20.txt").read_text(encoding="utf-8")
    alone = (ROOT / "src/uol/eval_lenses/llm_alone.py").read_text(encoding="utf-8")
    word = {"10": 10, "diez": 10, "20": 20, "veinte": 20}
    return {"RRF_K": int(re.search(r"^RRF_K = (\d+)", agents, re.M).group(1)),
            "LIST_DEPTH": int(re.search(r"^LIST_DEPTH = (\d+)", agents, re.M).group(1)),
            "list_n": word[re.search(r"exactamente (\w+) artículos", prompt).group(1)],
            "list_per_call": int(re.search(r"(\d+) cases per call", alone).group(1))}


CODE = _code_constants()
SRC: dict[str, str] = {}              # macro -> "file: key"
MAC: dict[str, str] = {}


def read(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def rel(p: Path) -> str:
    return Path(p).resolve().relative_to(ROOT).as_posix()


# --------------------------------------------------------------------------- formatting


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float) -> str:
    if abs(x) < 0.0005:                    # rounds to zero: no sign
        return "0.000"
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def sglo(x: float) -> str:
    """An adjusted lower end: its side of zero decides the class, so one that rounds to zero keeps four decimals."""
    if abs(x) >= 0.0005 or x == 0:
        return sg(x)
    return ("$+$" if x > 0 else "$-$") + (f"{abs(x):.4f}" if abs(x) >= 0.00005 else f"{abs(x):.5f}")


def ci(pair) -> str:
    return f"[{sg(pair[0])}, {sg(pair[1])}]"


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}\\%"


def num(x: float) -> str:
    return f"{int(round(x)):,}"


CLASS = {"useful": "clears the margin", "nonzero_below_margin": "a real gain, not shown to reach the margin", "equivalent": "no meaningful difference",
         "loss": "a loss", "inconclusive": "inconclusive"}  # v33 (D203): the five classes of 4.4, a loss included


def cls_of(c: dict) -> str:
    """The class of 4.4: the first that fits, so an inconclusive difference whose adjusted upper end lies below zero is a loss."""
    cia = c.get("ci_adjusted")
    return "loss" if c.get("class") == "inconclusive" and cia and cia[1] < 0 else c.get("class")


# no value and printed nowhere in the thesis (checked against thesis/ on 5 Nov): left out rather than left waiting
UNUSED = {"dModelOpus", "kModelOpus", "sAloneTok", "sRepeatTok", "sExamplesTok", "sMergedTok", "whyOpusLowList", "cHighM"}


def put(name: str, value, fmt, source: str) -> None:
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(name)
    if name in MAC:
        raise KeyError(f"macro {name} twice")
    ok = value is not None and not (isinstance(value, float) and math.isnan(value))
    if not ok and name in UNUSED:
        return
    MAC[name] = fmt(value) if ok else f"\\EvPending{{waiting: {name}}}"
    SRC[name] = source


def put_contrast(name: str, c: dict | None, source: str, negate: bool = False) -> None:
    """``name`` the difference, ``nameCI`` its 95% interval, ``nameLo`` the adjusted lower end, ``nameCls`` the class,
    ``nameM`` the family size. ``negate``: the stored contrast is b - a; print a - b."""
    if c and negate:
        lo_hi = [-c["ci95"][1], -c["ci95"][0]]
        adj = [-c["ci_adjusted"][1], -c["ci_adjusted"][0]] if "ci_adjusted" in c else None
        c = {**c, "delta": -c["delta"], "ci95": lo_hi, "adjusted_lower": adj[0] if adj else None, "ci_adjusted": adj,
             "class": classify(adj[0], [-c["ci90"][1], -c["ci90"][0]], adj[1]) if adj and "ci90" in c else c["class"]}
    if c:
        c = {**c, "class": cls_of(c)}
    put(name, c and c["delta"], sg, source)
    put(name + "CI", c and c["ci95"], ci, source)
    put(name + "Lo", c and c.get("adjusted_lower"), sglo, source)
    put(name + "Cls", c and c["class"], lambda v: CLASS[v], source)
    put(name + "M", c and c.get("family_size"), lambda v: str(int(v)), source)


def classify(adj_lo: float, ci90, adj_hi: float | None = None) -> str:
    if adj_lo >= 0.02:
        return "useful"
    if adj_lo > 0:
        return "nonzero_below_margin"
    if -0.02 < ci90[0] and ci90[1] < 0.02:
        return "equivalent"
    if adj_hi is not None and adj_hi < 0:
        return "loss"
    return "inconclusive"


# --------------------------------------------------------------------------- stores


def summ(store: Path, sid: str) -> dict | None:
    s = read(store / sid / "DEV" / "summary.json")
    return {m: s.get(m) for m in (*MEASURES, "n_cases", "n_groups", "unseen_pairs")} if s else None


def family(name: str) -> dict | None:
    return read(CON / f"{name}.json")


def contrast(name: str, key: str) -> dict | None:
    f = family(name)
    if not f:
        return None
    c = f["contrasts"].get(key)
    return {**c, "family": name, "family_size": f.get("size")} if c else None


def keep(c: dict | None) -> dict | None:
    return c and {k: c.get(k) for k in ("delta", "ci95", "ci90", "ci_adjusted", "adjusted_lower", "class", "family", "family_size", "a", "b")}


def sample_models() -> dict:
    """Claude Sonnet 5.5, Claude Haiku 5.5 and GPT-6 Astra over the ranker on the 300-case sample only (common footing
    PLAN, addenda 3 and 10, each rule written before the score): means over groups, the contrasts, the windows whose
    answer left candidates out (``repairs``; external.rerank keeps those in the order shown) and the cost of the run."""
    vs = (family("sweep_sample") or {}).get("contrasts", {})
    gain = (family("sweep_gain") or {}).get("contrasts", {})
    astra = (family("astra_sample") or {}).get("contrasts", {}).get("gpt6astra - sol_medium")
    rec = {k: read(RUN / "receipts" / f"effort_ask_{k}.json") or {} for k in ("sonnet", "haiku", "gpt6astra")}
    ref = vs.get("sonnet - sol_medium") or {}
    return {"scores": {"sonnet": ref.get("mean_a"), "haiku": (vs.get("haiku - sol_medium") or {}).get("mean_a"),
                       "gpt6astra": astra and astra["mean_a"], "gpt6sol": ref.get("mean_b"),
                       "ranker": (gain.get("sonnet - S0+qwen3_8b") or {}).get("mean_b")},
            "vs_gpt6sol": {**{k: keep(v) for k, v in vs.items()}, "gpt6astra - sol_medium": keep(astra)},
            "over_ranker": {k: keep(v) for k, v in gain.items()},
            "rows": {k: r.get("rows") for k, r in rec.items()},
            "windows": {k: r.get("rows") and 4 * r["rows"] for k, r in rec.items()},   # two windows, read in two orders
            "repairs": {k: r.get("repairs") for k, r in rec.items()},
            "usd_standard_per_1000": {"gpt6astra": rec["gpt6astra"].get("usd_list_equivalent") and rec["gpt6astra"].get("rows")
                                      and 1000 * rec["gpt6astra"]["usd_list_equivalent"] / rec["gpt6astra"]["rows"]}}


def depends_bins(dep: dict, panel: str, sid: str) -> list[float] | None:
    """Mean Recall@20 per bin, in bin order, of run ``sid`` in panel ``panel`` ("words" or "citations") of depends.json
    (research/programs/final_results_20261102, written by uol.thesis_figures.v21_figs for figure F6_16)."""
    p = (dep.get("panels") or {}).get({"words": "a", "citations": "b"}[panel]) or {}
    for m in (p.get("methods") or {}).values():
        if m.get("run") == sid:
            vals = [((m.get("values") or {}).get(b) or {}).get("mean") for b in p.get("bins") or []]
            return vals if vals and all(isinstance(v, (int, float)) for v in vals) else None
    return None


def dev_table() -> dict:
    out = {}
    for store, tag in ((RECORD, "record"), (AUDIT, "audit"), (FOOT, "footing")):
        for d in sorted(store.glob("*")):
            s = read(d / "DEV" / "summary.json")
            if s and s.get("n_cases") == 1217:
                out[d.name] = {"store": tag, **{m: s.get(m) for m in MEASURES}}
    return out


def sample_table() -> dict:
    import pandas as pd
    out = {}
    for n in SAMPLE:
        r = read(RECEIPTS / f"agents_{n}.json")
        if r:
            out[n] = {k: r.get(k) for k in ("mean_per_case", "success_at_10", "seconds", "usage", "usd_list_equivalent", "contrasts", "others_same_cases") if k in r}
            u = r.get("usage") or {}
            if u.get("input_tokens"):
                out[n]["tokens_per_case"] = (u.get("input_tokens", 0) + u.get("output_tokens", 0)) / r.get("cases", 300)
        # each model asked alone against the same model reordering the 28-source ranker's list, on the same sample cases
        # (v27: the reordering was run on every development case, so its sample score is read from its per-case file)
        if n == "alone" or n.startswith("alone_"):
            kind = "gpt61sol" if n == "alone" else n[len("alone_"):]
            a, b, l_ = (WORK / "agents" / n / "per_case.parquet", FOOT / f"{kind}_rerank_S0+qwen3_8b" / "DEV" / "per_case.parquet",
                        FOOT / "S0+qwen3_8b" / "DEV" / "per_case.parquet")
            if n in out and a.exists() and b.exists() and l_.exists():
                ids = pd.read_parquet(a, columns=["query_id"])["query_id"]
                for key, f in (("on_list_same_cases", b), ("list_same_cases", l_)):
                    d = pd.read_parquet(f, columns=["query_id", "ndcg_at_10"])
                    out[n][key] = float(d.loc[d["query_id"].isin(ids), "ndcg_at_10"].mean())
    return out


def cached_family(name: str, pairs: list[tuple[str, Path, str, Path]]) -> dict | None:
    """A family of the paired group contrasts of 4.4 (``audit_d181_common.dev_family``, 100,000 draws) over stored
    development systems, ``pairs`` as (a, store of a, b, store of b). Cached by the per-case files' times; None until every
    file exists."""
    files = [st / sid / "DEV" / "per_case.parquet" for a_, sa, b_, sb in pairs for sid, st in ((a_, sa), (b_, sb))]
    if not all(f.exists() for f in files):
        return None
    path = OUT / f"{name}.json"
    stamp = [str(f) for f in files] + [f.stat().st_mtime for f in files]
    old = read(path)
    if old and old.get("stamp") == stamp:
        return old
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    fam = A.dev_family(name, [(a_, A.per_case(sa.parent, a_), b_, A.per_case(sb.parent, b_)) for a_, sa, b_, sb in pairs])
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


def final_family(sys_id: str) -> dict:
    """PLAN addendum 22: the system minus the model reordering the ranker's list, minus the model alone, minus the ranker,
    one family of three."""
    return cached_family("final_dev", [(sys_id, FOOT, REORD, FOOT), (sys_id, FOOT, "gpt61sol_alone", FOOT), (sys_id, FOOT, RANKER, FOOT)]) or {"contrasts": {}}


def one(name: str, a: str, sa: Path, b: str, sb: Path) -> dict | None:
    """A family of one, its contrast or None."""
    f = cached_family(name, [(a, sa, b, sb)])
    return f and keep({**f["contrasts"][f"{a} - {b}"], "family": name, "family_size": 1})


def on_system_family() -> dict | None:
    """Rebuild PLAN step 3: each addition read inside the system, minus the combination, one family (adjusted lower end
    at least .02 to enter). Read only once every run of the family exists."""
    bases = ("ft_tesis", "S_full_ext")
    files = [FOOT / f"gpt61sol_combo_on_{b}" / "DEV" / "per_case.parquet" for b in bases] + [FOOT / "gpt61sol_combo" / "DEV" / "per_case.parquet"]
    if not all(f.exists() for f in files):
        return None
    path = OUT / "on_system_family.json"
    stamp = [f.stat().st_mtime for f in files]
    old = read(path)
    if old and old.get("stamp") == stamp:
        return old
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    base = A.per_case(FOOT.parent, "gpt61sol_combo")
    fam = A.dev_family("additions_in_system", [(f"gpt61sol_combo_on_{b}", A.per_case(FOOT.parent, f"gpt61sol_combo_on_{b}"), "gpt61sol_combo", base) for b in bases])
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


def lean_in_system() -> dict | None:
    """PLAN addendum 21 (E1): the lean ranker's list in place of the ranker's inside the system, minus the combination,
    a family of one. A finding only: the frozen system does not change."""
    files = [FOOT / s / "DEV" / "per_case.parquet" for s in ("gpt61sol_combo_on_ft_lean", "gpt61sol_combo")]
    if not all(f.exists() for f in files):
        return None
    path = OUT / "lean_in_system.json"
    stamp = [f.stat().st_mtime for f in files]
    old = read(path)
    if old and old.get("stamp") == stamp:
        return old
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    fam = A.dev_family("lean_in_system", [("gpt61sol_combo_on_ft_lean", A.per_case(FOOT.parent, "gpt61sol_combo_on_ft_lean"),
                                           "gpt61sol_combo", A.per_case(FOOT.parent, "gpt61sol_combo"))])
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


SAMPLE_VS = ("alone", "repeat", "examples", "list_agent_words", "agent_embedding", "list_agent_embedding")


def sample_vs_combo() -> dict | None:
    """Each way of using the model on the 300-case sample minus the combination on the same cases, one family."""
    path = OUT / "sample_vs_combo.json"
    agents = WORK / "agents"
    files = [agents / n / "per_case.parquet" for n in (*SAMPLE_VS, "combo")]
    if not all(f.exists() for f in files):
        return None
    stamp = [f.stat().st_mtime for f in files]
    old = read(path)
    if old and old.get("stamp") == stamp:
        return old
    import pandas as pd
    sys.path.insert(0, str(ROOT / "src"))                         # before scripts/, whose uol.py would shadow the package
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    combo = pd.read_parquet(agents / "combo" / "per_case.parquet")
    fam = A.dev_family("sample_vs_combo", [(n, pd.read_parquet(agents / n / "per_case.parquet"), "combo", combo) for n in SAMPLE_VS],
                       population="DEV-A300")
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


# --------------------------------------------------------------------------- the system's settings (PLAN addendum 20)


# --------------------------------------------------------------------------- v33: the comparisons on the ranker itself (D203)
# the author, 4 Nov: "Rerun everything". Every comparison made until v32 on the 28-source ranker is made again on the ranker
# (ft_lean) by uol.final.lean_rerun, under its own ids; the families keep the structure of the 28-source ones.
LEAN_ADD = ("rk_structure", "rk_cocite", "rk_adapter", "rk_smooth", "rk_issues", "rk_patterns")
ADD_FAMILY = (*LEAN_ADD, "rk_q2d", "rk_hyde", "rk_topics")       # rank_rerun_20261105/PLAN.md: one family of nine
RANKER_CHOICE = ("rt_lm_tesis", "rt_check", "rt_lm_wide", "rt_lm_wide_tesis", "rt_xgb", "rt_xgb_tesis")   # ranker_tests_20261104: each minus the trees on single articles
LEAN_MODELS = ("luna", "gpt61sol", "gpt61terra", "gpt6luna", "gpt56sol", "gpt56terra")
LEAN_ENCODERS = ("bge_m3", "qwen3_0p6b", "qwen3_4b", "qwen3_8b")   # v33: Euler stays an encoder-alone comparison
LEAN_BASES = ("ft_lean", "S_full_ext")
IN_SYSTEM = {"ft_lean": "gpt61sol_combo_fwd_on_ft_lean", "S_full_ext": "gpt61sol_final_on_S_full_ext"}   # the same model's list and reading over another ranker
LEAN_RR = {"bge_ranker": "CeBge", "qwen4b_ranker": "CeQwen", "ce_ranker": "CeTrained"}
AGENTS_LEAN = WORK / "agents_rk"


def lean_families() -> dict:
    """The families of the rerun, each None until every run in it exists: the additions (with the whole-list trees
    against the single-candidate ones, as B_additions), the rewritings, the topic scores, the models against GPT-6 Sol,
    the additions inside the full system, and the encoders added to the ranker without the encoder."""
    return {
        "additions": cached_family("rk_additions", [(s, FOOT, RANKER, FOOT) for s in ADD_FAMILY]),
        "parts": cached_family("rk_parts", [(RANKER, FOOT, "rt_check", FOOT), ("rt_check", FOOT, "lf_plain", FOOT)]),   # the votes; whole-list against single-article trees
        "choice": cached_family("rk_choice", [(s, FOOT, "ft_lean", FOOT) for s in RANKER_CHOICE]),
        "largest": cached_family("rk_largest", [(RANKER, FOOT, "S_full_ext", FOOT)]),
        "models": cached_family("lean_models", [(f"{k}_rerank_{RANKER}", FOOT, f"gpt6solnw_rerank_{RANKER}", FOOT) for k in LEAN_MODELS]),
        "in_system": cached_family("rk_in_system", [(IN_SYSTEM[b], FOOT, SYSTEM, FOOT) for b in LEAN_BASES]),
        "ladder": cached_family("rk_ladder", [(f"rk_noenc+{k}", FOOT, NOENC, FOOT) for k in LEAN_ENCODERS]),
    }


def sample_frames() -> dict:
    """Per-case frames of the ways of using the model on the 300-case sample: the four that read no ranker as stored, the
    two agents that start from the ranker's list as rerun on it, and the ranker, its reranking and the system on the same
    cases (from their runs on every development case)."""
    import pandas as pd
    out = {n: WORK / "agents" / n / "per_case.parquet" for n in ("alone", "repeat", "examples", "agent_embedding")}
    out.update({n: AGENTS_LEAN / n / "per_case.parquet" for n in ("list_agent_words", "list_agent_embedding")})
    frames = {n: pd.read_parquet(f) for n, f in out.items() if f.exists()}
    if "alone" not in frames:
        return frames
    ids = set(frames["alone"]["query_id"])
    for n, sid in (("system", SYSTEM), ("reord", REORD), ("ranker", RANKER)):
        f = FOOT / sid / "DEV" / "per_case.parquet"
        if f.exists():
            d = pd.read_parquet(f)
            frames[n] = d[d["query_id"].isin(ids)].reset_index(drop=True)
    return frames


def sample_vs_system() -> dict | None:
    """Each way of using the model on the sample minus the system on the same cases, one family of six."""
    frames = sample_frames()
    ways = ("alone", "repeat", "examples", "agent_embedding", "list_agent_words", "list_agent_embedding")
    if not all(n in frames for n in (*ways, "system")):
        return None
    path = OUT / "sample_vs_system.json"
    stamp = [len(frames[n]) for n in (*ways, "system")] + [float(frames[n]["ndcg_at_10"].sum()) for n in (*ways, "system")]
    old = read(path)
    if old and old.get("stamp") == stamp:
        return old
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    fam = A.dev_family("sample_vs_system", [(n, frames[n], "system", frames["system"]) for n in ways], population="DEV-A300")
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


CLAUDE_SAMPLE = {"claude": "Opus", "haiku": "Haiku", "sonnet": "Sonnet"}


def sample_claude() -> dict | None:
    """v33 (D203): the Claude models reranking the ranker's list on the sample, through the author's account (uol.final.lean_rerun
    sample), each minus the ranker on the same cases, one family of three; with the share of window orders that left out
    candidates (``repairs`` of the orders file over cases x windows x passes)."""
    import pandas as pd
    runs = {k: FOOT / f"{k}_rerank_{RANKER}" / "DEV-A300" / "per_case.parquet" for k in CLAUDE_SAMPLE}
    if not all(p.exists() for p in runs.values()) or not (FOOT / RANKER / "DEV" / "per_case.parquet").exists():
        return None
    fr = {k: pd.read_parquet(p) for k, p in runs.items()}
    d = pd.read_parquet(FOOT / RANKER / "DEV" / "per_case.parquet")
    fr["ranker"] = d[d["query_id"].isin(set(fr["claude"]["query_id"]))].reset_index(drop=True)
    path = OUT / "sample_claude.json"
    stamp = [len(f) for f in fr.values()] + [float(f["ndcg_at_10"].sum()) for f in fr.values()]
    old = read(path)
    def repair_share() -> dict:
        got = {}
        for k in CLAUDE_SAMPLE:
            o = read(WORK / "rk" / "sample" / k / f"{k}_orders_s0.json") or {}
            if o.get("repairs") is not None:
                got[k] = o["repairs"] / (len(o["rows"]) * 2 * len(o["passes"]))    # two windows of the first 30
        return got
    if old and old.get("stamp") == stamp:
        return {**old, "repair_share": repair_share()}
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.final import audit_d181_common as A
    fam = A.dev_family("sample_claude", [(k, fr[k], "ranker", fr["ranker"]) for k in CLAUDE_SAMPLE], population="DEV-A300")
    fam["means"] = {k: float(f["ndcg_at_10"].mean()) for k, f in fr.items()}
    fam["repair_share"] = repair_share()
    fam["stamp"] = stamp
    path.write_text(json.dumps(fam, indent=1, default=float), encoding="utf-8")
    return fam


def settings() -> dict:
    """PLAN addendum 22: the six checks, each a change to the final system minus the final system (one family of six).
    The settings are D198's and do not follow from the checks."""
    fam = cached_family("final_checks", [(CHECK[v], FOOT, SYSTEM, FOOT) for v in SETTINGS]) or {"contrasts": {}}
    got = {}
    for v in SETTINGS:
        c = fam["contrasts"].get(f"{CHECK[v]} - {SYSTEM}")
        if c:
            got[v] = {**keep({**c, "family": "final_checks", "family_size": fam.get("size")}), "mean": (summ(FOOT, CHECK[v]) or {}).get("ndcg_at_10")}
    # low effort against medium, the forward reading of the same merged list
    lo, med = read(RECEIPTS / f"rk_{CHECK['low']}.json") or {}, read(RECEIPTS / "rk_gpt61sol_rk_system.json") or {}
    usd = {"low": lo.get("usd_list_equivalent"), "medium": med.get("usd_list_equivalent")}
    if "low" in got:
        got["low"]["usd_receipt"] = usd
    frozen = read(TEST / "FINAL_SYSTEM.json") or read(TEST_EARLIER / "FINAL_SYSTEM.json") or {}
    return {"checks": got, "frozen": {k: frozen.get(k) for k in ("passes", "effort", "depth", "w_all", "rrf_k")},
            "missing": [v for v in SETTINGS if v not in got]}


def final_dev_system(st: dict | None = None) -> str:
    return SYSTEM


# --------------------------------------------------------------------------- measured facts and per-case analyses


def measure() -> dict:
    """Development cases only, corrected labels. Word counts and the share of an article's distinct words (stopwords out)
    that occur in the facts; never-cited flags per fold; who finds what in the first ten; the examples' ranks; the
    support share; test-set counts and dates (no label)."""
    os.environ.setdefault("UOL_FINAL_THREADS", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("UOL_FINAL_LABELS", "corrected")
    os.environ.setdefault("UOL_FINAL_SYSTEMS", str(RUN / "systems_audit.json"))
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import pandas as pd
    from nltk.corpus import stopwords
    from uol.final import footing as FT
    from uol.final import audit_d181_dev as D, measures
    from uol.embedding_push import common as ep
    D.activate(FT.STORE)
    d = ep.release()
    if d.candidate_texts is None:
        raise RuntimeError("the release was loaded without article texts")
    rows = [int(r) for r in FT.spec_rows()]
    dev_all = np.asarray(ep.dev_rows())
    roles = ep.roles()
    fold_of = {int(r): int(str(roles[r]).replace("dev_fold", "")) for r in dev_all}
    generic = {int(i) for i in d.generic_idx}
    stop = set(stopwords.words("spanish"))
    word = re.compile(r"[a-záéíóúüñ]+")

    def words(t: str) -> list[str]:
        return word.findall(t.lower())

    def distinct(t: str) -> set[str]:
        return {w for w in words(t) if w not in stop and len(w) > 1}

    rng = np.random.default_rng(20261103)
    n_cand = len(d.candidate_ids)
    cited_n, fact_len, shares, rand_shares, cited_len = [], [], [], [], []
    art_cache: dict[int, set[str]] = {}
    for r in rows:
        q = d.queries[r]
        pos = [p for p in d.specific_idx[r]]
        cited_n.append(len(pos))
        fw = distinct(q.text)
        fact_len.append(len(words(q.text)))
        for p in pos:
            aw = art_cache.setdefault(p, distinct(d.candidate_texts[p]))
            cited_len.append(len(words(d.candidate_texts[p])))
            shares.append(len(aw & fw) / len(aw) if aw else 0.0)
        while True:
            x = int(rng.integers(n_cand))
            if x not in generic and x not in d.positive_idx[r]:
                break
        aw = art_cache.setdefault(x, distinct(d.candidate_texts[x]))
        rand_shares.append(len(aw & fw) / len(aw) if aw else 0.0)
    cat_len = [len(words(t)) for t in d.candidate_texts]
    facts = {"cases": len(rows), "groups": len({d.queries[r].component_id for r in rows}),
             "cited_per_case_median": float(np.median(cited_n)), "cited_per_case_mean": float(np.mean(cited_n)),
             "cases_with_one_cited": float(np.mean(np.asarray(cited_n) == 1)),
             "facts_words_median": float(np.median(fact_len)), "cited_article_words_median": float(np.median(cited_len)),
             "catalogue_article_words_median": float(np.median(cat_len)),
             "share_cited_median": float(np.median(shares)), "share_cited_mean": float(np.mean(shares)),
             "share_random_median": float(np.median(rand_shares)), "share_random_mean": float(np.mean(rand_shares)),
             "pairs": len(shares), "definition": "distinct lower-case words of two letters or more, NLTK Spanish stopwords out; share = article words also in the facts / article words; one random non-cited, non-generic article per case, seed 20261103"}

    # never-cited per fold: no training component cites the article (measures.unseen_mask on the other folds' rows)
    unseen = {k: measures.unseen_mask([r for r in dev_all if fold_of[int(r)] != k]) for k in range(5)}

    def top(store: Path, sid: str, k: int = 10) -> dict[str, list[int]]:
        t = pd.read_parquet(store / sid / "DEV" / "top200.parquet")
        t = t[t["rank"] <= k]
        return {q: [int(c) for c in g.sort_values("rank")["candidate_idx"]] for q, g in t.groupby("query_id")}

    sys_id = final_dev_system()
    lists = {"ranker": top(FOOT, RANKER), "alone": top(FOOT, "gpt61sol_alone"), "system": top(FOOT, sys_id)}
    who_case, who_pair, who_never = {}, {}, {}
    for r in rows:
        q = d.queries[r].query_id
        k = fold_of[r]
        pos = set(d.specific_idx[r])
        a, b = set(lists["alone"].get(q, [])) & pos, set(lists["ranker"].get(q, [])) & pos
        key = "both" if a and b else "model_only" if a else "ranker_only" if b else "neither"
        who_case[key] = who_case.get(key, 0) + 1
        for p in pos:
            kp = "both" if p in a and p in b else "model_only" if p in a else "ranker_only" if p in b else "neither"
            who_pair[kp] = who_pair.get(kp, 0) + 1
            if unseen[k][p]:
                who_never[kp] = who_never.get(kp, 0) + 1
    norm = lambda c: {k: v / sum(c.values()) for k, v in c.items()} | {"n": sum(c.values())}
    # share of never-cited articles among each system's first ten, and of articles with a past case that cites them
    never_share = {}
    for name, L in lists.items():
        tot = nev = 0
        for r in rows:
            q = d.queries[r].query_id
            got = L.get(q, [])
            tot += len(got)
            nev += sum(bool(unseen[fold_of[r]][c]) for c in got)
        never_share[name] = nev / tot if tot else None
    # the examples' ranks in each list
    ex = {}
    for name, (store, sid) in {"lexical": (RECORD, "lstar_headings"), "noenc": (FOOT, NOENC), "encoder": (RECORD, "qwen3_8b"), "ranker": (FOOT, RANKER),
                               "alone": (FOOT, "gpt61sol_alone"), "reordered": (FOOT, REORD), "merged": (FOOT, CHECK["merged"]), "system": (FOOT, sys_id)}.items():
        p = store / sid / "DEV" / "per_case.parquet"
        if not p.exists():
            continue
        f = pd.read_parquet(p).set_index("query_id")
        for qid in (EXAMPLE, EXAMPLE_B):
            if qid in f.index:
                ranks = [int(x) for x in str(f.loc[qid, "positive_ranks"]).split(",") if x.strip()]
                ex.setdefault(qid, {})[name] = min(ranks) if ranks else None
    # the ranker's score if the cited law were known: only the cited laws' articles of its top 200, order kept
    # (uol.analysis_c.c1_law.system_cases, oracle_ndcg_at_10; mean over groups)
    from uol.analysis_c import common as AC, c1_law
    cat, rstore = AC.load_catalogue(), AC.RunStore(FOOT, population="DEV")
    oracle = {}
    for sid in (RANKER, NOENC):
        if not rstore.has(sid):
            continue
        f = c1_law.system_cases(rstore, cat, sid)
        f = f[f["present"]]
        oracle[sid] = float(f.groupby("component_id")["oracle_ndcg_at_10"].mean().mean())
    # test set: counts, groups and dates, no label
    test = {}
    rf = read(TEST_EARLIER / "rows.json")
    if rf:
        from uol.final import test_set as TS
        later = pd.read_parquet(ROOT / "experiments/runs/footing_20261101/new/eval" / FT.A.NEW_SYS / "per_case.parquet")
        dates = {**TS.test_dates(rf["query_ids"]), **dict(zip(later["query_id"], later["publication_date"]))}
        comp_of = {}
        with (ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified/queries.jsonl").open(encoding="utf-8") as fh:   # groups only; labels are in references.jsonl
            for line in fh:
                o = json.loads(line)
                comp_of[o["query_id"]] = o["component_id"]
        cut = FT.CUTOFFS["gpt61sol"]
        test = {"cases_352": len(rf["rows"]), "cases_later": int(len(later)), "cases": len(rf["rows"]) + int(len(later)),
                "groups": len({comp_of[q] for q in rf["query_ids"]} | set(later["component_id"])), "cutoff": cut,
                "after_cutoff": sum(1 for q in [*rf["query_ids"], *later["query_id"]] if dates.get(q, "") > cut),
                "first_date": min(dates.values()), "last_date": max(dates.values())}
        # development-period cases dated inside the test period (whole groups stay together)
        src = {}
        with (ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                src[o.get("query_id")] = o.get("publication_date")
        first = test["first_date"]
        inside = [int(r) for r in dev_all if (src.get(d.queries[int(r)].query_id) or "") >= first]
        test["dev_dated_in_test_period"] = len(inside)
        test["dev_dated_in_test_period_scored"] = len(set(inside) & set(rows))
    return {"facts": facts, "who_case": norm(who_case), "who_pair": norm(who_pair), "who_never": norm(who_never), "never_share_first_ten": never_share,
            "system_list": sys_id, "examples": ex, "test": test, "law_oracle": oracle}


# --------------------------------------------------------------------------- cost and time


def usage_cost(u: dict, per_mtok: tuple[float, float, float], cases: int) -> dict:
    fresh = u["input_tokens"] - u.get("cached_input_tokens", 0)
    usd = (fresh * per_mtok[0] + u.get("cached_input_tokens", 0) * per_mtok[1] + u["output_tokens"] * per_mtok[2]) / 1e6
    return {"cases": cases, "tokens_per_case": (u["input_tokens"] + u["output_tokens"]) / cases, "usd_standard_per_1000": 1000 * usd / cases,
            "usd_batch_per_1000": 1000 * usd * BATCH_SHARE / cases}


def ledger(tag: str, path: Path, model: str = "gpt-6.1-sol") -> dict:
    """Tokens and call latency of one tag's answered calls."""
    import statistics
    u: dict[str, int] = {}
    lat = []
    for line in path.open(encoding="utf-8"):
        try:
            o = json.loads(line)
        except ValueError:
            continue
        if o.get("tag") != tag or o.get("model") != model or not o.get("ok"):
            continue
        for k, x in (o.get("usage") or {}).items():
            u[k] = u.get(k, 0) + x
        if o.get("latency"):
            lat.append(o["latency"])
    lat.sort()
    return {"calls": len(lat), "usage": u, "latency_median": statistics.median(lat) if lat else None,
            "latency_p95": lat[max(0, int(0.95 * len(lat)) - 1)] if lat else None}


def ranker_spec() -> dict:
    """What the ranker reads and how it is trained, from its receipt, its pipeline manifest and the code's constants."""
    r = read(ROOT / f"experiments/runs/footing_20261101/dev/store/receipts/{RANKER}__DEV.json")
    if not r:
        return {}
    man = read(Path(r["inputs"]["pipeline_manifest"])) or {}
    args, folds = r["spec"]["adapter_args"], r["folds"]
    pool = man.get("pool", {})
    meta = read(WORK / "dev" / "llm" / "gpt61sol_rerank_S0+qwen3_8b" / "meta.json") or {}
    deep = sum(1 for f in folds if f["fit"]["config"] == {"learning_rate": 0.05, "num_leaves": 15})
    words = sum(1 for f in folds if (f["fit"].get("m1") or {}).get("key") == "lex")
    sys.path.insert(0, str(ROOT / "src"))
    sys.modules.pop("uol", None)
    from uol.gap_tests import ltr_listwise as LW
    fx = LW.FIXED
    old_folds = read(ROOT / "experiments/runs/footing_20261101/dev/store/receipts/ft_lean__DEV.json")["folds"]
    if any(a_["fit"]["pairs_scored"] != b_["fit"].get("pairs_scored") for a_, b_ in zip(folds, old_folds)):
        raise RuntimeError("the chosen ranker's pool differs from the single-article trees'")
    drop = set(args.get("drop_sources", []))
    names = [x for x in [*pool.get("sources", []), *args["add_sources"]] if x not in drop]
    ENC3 = ("dense_qwen8b", "knn_qwen8b", "proto_qwen8b")   # uol.final.footing.ENCODER
    return {"sources": len(names), "source_names": names, "sources_without_encoder": sum(1 for x in names if x not in ENC3),
            "encoder_sources": sum(1 for x in names if x in ENC3), "topic_models": len(args["extra"]),
            "inputs": 30, "top_per_source": pool.get("top_k_per_source"),            # 27 columns of the eight sources and the shared ones, three of votes
            "pool_per_case": sum(f["fit"]["pool_pairs_per_case"] for f in old_folds) / len(old_folds),
            "pool_recall": sum(f["fit"]["pool_recall"] for f in old_folds) / len(old_folds),   # the same eight sources, the same pool as the single-article trees
            "negatives_per_case": LW.N_NEG, "trees": fx["n_estimators"], "min_leaf": fx["min_child_samples"], "grid": folds[0]["fit"]["grid"] and [g["config"] for g in folds[0]["fit"]["grid"]],
            "folds_lr05_leaves15": deep, "folds_votes_by_words": words, "folds": len(folds), "second_reading_weight": meta.get("w_all"),
            "source": f"{RANKER}__DEV.json, its pipeline manifest, gpt61sol_rerank_S0+qwen3_8b/meta.json"}


def cost(st: dict) -> dict:
    sol = (2.0, 0.10, 10.0)                                     # GPT-6.1 Sol, standard tier, read 2 Nov 2026 (uol.final.footing.MODELS)
    out = {"rates_per_mtok_standard": sol, "batch_share": BATCH_SHARE}
    r = read(RECEIPTS / "rk_gpt61sol_rk_system.json")            # the final reranking of the merged list, forward order
    if r and r.get("usage"):
        out["second_reading"] = usage_cost(r["usage"], sol, r["cases"])
    lst = ledger("llm_alone_dev", WORK / "dev" / "luna" / "LUNA_LEDGER.jsonl")     # every call of the model's list, all 1,217 cases
    if lst["calls"]:
        out["model_list"] = {**usage_cost(lst["usage"], sol, 1217), "call_latency_median_s": lst["latency_median"], "call_latency_p95_s": lst["latency_p95"],
                             "cases_per_call": 10}
    rd = ledger("agents_gpt61sol_rk_system", ROOT / "research/state/round3_best_case_20261013/LUNA_LEDGER.jsonl")
    if rd["calls"] and "second_reading" in out:
        out["second_reading"].update({"call_latency_median_s": rd["latency_median"], "call_latency_p95_s": rd["latency_p95"], "cases_per_call": 5})
    if "second_reading" in out and "model_list" in out:
        out["system"] = {k: out["second_reading"][k] + out["model_list"][k] for k in ("tokens_per_case", "usd_standard_per_1000", "usd_batch_per_1000")}
    rm = read(WORK / "dev" / "llm" / REORD / "meta.json")      # the model reordering the ranker's list, both orders (PLAN addendum 22)
    if rm and rm.get("usage"):
        out["reorder"] = {**usage_cost(rm["usage"], sol, 1217), "source": f".cache/footing/dev/llm/{REORD}/meta.json usage"}
    t = read(RUN / "timing" / "RESULT.json")
    if t:
        out["ranker_time"] = {"median_s": t["whole_method_seconds"]["median"], "p95_s": t["whole_method_seconds"]["p95"], "cases": t["cases"],
                              "gpu_median_s": t["gpu_seconds"]["median"], "cpu_median_s": t["cpu_seconds"]["total"]["median"],
                              "ram_gb": t["ram_gb"]["process_peak_including_load"], "encoder_ram_gb": t["ram_gb"]["encoder_process_peak"],
                              "vram_gb": t["gpu_memory_gb"]["peak_reserved_gb"], "gpu": t["gpu_memory_gb"]["gpu"]}
    cases = RUN / "timing" / "cases.jsonl"
    if cases.exists():                                   # each local setup on the same 200 timed cases (timing/cases.jsonl)
        sys.path.insert(0, str(ROOT / "src"))           # scripts/uol.py would shadow the package
        from uol.final.footing import ENCODER, LEAN_DROP
        import statistics
        tex = (ROOT / "thesis" / "evidence" / "numbers_cost_20261025.tex").read_text(encoding="utf-8")   # the rates of Table B.1
        pc, pg = (float(re.search(r"\\newcommand\{\\" + n + r"\}\{([0-9.]+)\}", tex).group(1)) for n in ("priceCPU", "priceGPU"))
        usd = lambda cpu, gpu: 1000 * (cpu * pc + gpu * pg) / 3600
        rows = [json.loads(x) for x in cases.read_text(encoding="utf-8").splitlines() if x.strip()]
        rows = [x for x in rows if not x.get("warmup")]
        # the steps each setup runs: its sources, the pool, the inputs and the trees, and the query encoding where it reads the encoder
        noenc_src = ["L_star", "docexp_bm25", "knn_lex_votes", "prior", "proto_tfidf"]          # rk_noenc
        keep = {"ranker": {"source:" + s for s in (*noenc_src, *ENCODER)} | {"pool", "features", "rank", "encode:qwen"},   # rt_lm_tesis
                "ranker_noenc": {"source:" + s for s in noenc_src} | {"pool", "features", "rank"}}
        gpu = {"ranker": lambda x: x["wall"]["encode:qwen"], "ranker_noenc": lambda x: 0.0}
        for k in keep:
            v = [usd(sum(t for n, t in x["cpu"].items() if n in keep[k] and n != "encode:qwen"), gpu[k](x)) for x in rows]
            w = sorted(sum(t for n, t in x["wall"].items() if n in keep[k]) for x in rows)
            out[k] = {"cases": len(rows), "usd_per_1000_median": statistics.median(v),
                      "seconds_median": statistics.median(w), "seconds_p95": w[math.ceil(0.95 * len(w)) - 1],
                      "note": "processor time of the steps it runs (timing/cases.jsonl), with the query encoding on the graphics card where it reads the encoder; "
                              "seconds: the case's timed steps that it runs"}
        out["ranker_28"] = {"usd_per_1000_median": statistics.median(usd(sum(x["cpu"].values()), x["wall"]["encode:qwen"]) for x in rows)}
    lv = read(ROOT / ".cache/whole_run/runs/noenc_none_a/result.json")   # the live run, facts to list, end to end
    if lv and "ranker_noenc" in out:
        out["ranker_noenc"].update({"seconds_median": lv["seconds"]["median"], "seconds_p95": lv["seconds"]["p95"], "seconds_cases": lv["cases"],
                                    "seconds_source": ".cache/whole_run/runs/noenc_none_a/result.json (whole_run, live, Intel Core i9-14900KF)"})
    # the live runs price each case as it ran: the process's processor seconds at the vCPU rate, the encoding's wall time on the card at the GPU rate
    def live(name: str, card: bool) -> dict | None:
        f = ROOT / ".cache/whole_run/runs" / name / "cases.jsonl"
        if not f.exists():
            return None
        rs = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        w = sorted(x["seconds"] for x in rs)
        u = [1000 * (x["cpu_seconds"] * pc + (x["encode_wait"] if card else 0.0) * pg) / 3600 for x in rs]
        return {"cases": len(rs), "seconds_median": statistics.median(w), "seconds_p95": w[math.ceil(0.95 * len(w)) - 1],
                "usd_per_1000_median": statistics.median(u), "source": f".cache/whole_run/runs/{name}/cases.jsonl (whole_run, live, i9-14900KF)"}
    if cases.exists():
        lr, ln, lc = live("ranker_card_a", True), live("noenc_none_a", False), live("ranker_cpu_a", False)
        if lr:
            out["ranker"] = {**out.get("ranker", {}), **lr}
        if ln:
            out["ranker_noenc"] = {**out.get("ranker_noenc", {}), **ln}
        if lc:
            out["ranker_cpu"] = lc          # seconds only: the encoder's own process is not in cpu_seconds
    return out


def whole_test_scored() -> dict | None:
    """The one scored test run of the chosen system on the live path (TEST_RELEASE.json), its keys as test_final's:
    the system is "combo" there."""
    d = read(RUN / "test_whole" / "scored.json")
    if not d:
        return None
    ren = lambda k: k.replace("system", "combo")
    return {**d, "_source": "test_whole/scored.json",
            "summaries": {p: {ren(k): v for k, v in s.items()} for p, s in d["summaries"].items()},
            "contrasts": {ren(k): v for k, v in d["contrasts"].items()}}


# --------------------------------------------------------------------------- the record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true", help="recompute measured.json (needs the release)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.measure or not MEASURED.exists():
        MEASURED.write_text(json.dumps(measure(), indent=1, default=float), encoding="utf-8")
    meas = read(MEASURED)
    st = settings()
    sys_id = final_dev_system(st)
    dev = dev_table()
    main_rows = {k: summ(store, sid) for k, (store, sid) in MAIN.items()}
    main_rows["Sys"] = summ(FOOT, sys_id)
    cells = {k: summ(store, sid) for k, (store, sid) in CELLS.items()}
    scored = whole_test_scored() or read(TEST / "scored.json")
    final = read(TEST / "FINAL_SYSTEM.json")
    earlier = read(TEST_EARLIER / "scored.json")
    cst = cost(st)
    cls_ = read(RUN / "classifier" / "summary.json")
    lf = lean_families()
    rec = {
        "benchmark": {"measured": meas["facts"], "test": meas["test"], "measured_law_oracle": meas.get("law_oracle"), "dev_cases": (main_rows["Combo"] or {}).get("n_cases"),
                      "dev_groups": (main_rows["Combo"] or {}).get("n_groups")},
        "system": {"definition": "the model's list (GPT-6.1 Sol, facts only, 20 articles, generic ones out) and the ranker's list (rt_lm_tesis, LambdaMART, 8 sources and the older-tesis votes) merged by "
                                 "reciprocal rank fusion, k 60; the first 30 of the merged list read once, in forward order, by GPT-6.1 Sol in windows of 20, "
                                 "stride 10, at medium effort; its order fused with the merged position at weight 3 (PLAN addendum 22)",
                   "dev_system": sys_id, "frozen": final, "settings": st, "dev": main_rows, "test": scored, "test_earlier": earlier and {k: earlier[k] for k in ("summaries", "final_system", "utc")}, "cost": cst, "ranker": ranker_spec(),
                   "contrasts": {k.replace(sys_id, "SYSTEM"): keep(v) for k, v in final_family(sys_id)["contrasts"].items()}},
        "experiments": {"cells": cells, "lean": keep(contrast("lean", "ft_lean - ft_ranker")), "classifier": cls_,
                        "encoder": one("encoder_gain", RANKER, FOOT, NOENC, FOOT),
                        "noenc_vs_lex": one("noenc_vs_lex", NOENC, FOOT, "lstar_headings", RECORD),
                        "sys_28": one("sys_28", "gpt61sol_combo_fwd", FOOT, SYSTEM, FOOT),
                        "alone_reord": one("alone_reord", "gpt61sol_alone", FOOT, REORD, FOOT),
                        "additions": {k: keep(v) for k, v in (lf["additions"] or {"contrasts": {}})["contrasts"].items()},
                        "parts": {k: keep(v) for k, v in (lf["parts"] or {"contrasts": {}})["contrasts"].items()},
                        "choice": {k: keep(v) for k, v in (lf["choice"] or {"contrasts": {}})["contrasts"].items()},
                        "largest": {k: keep(v) for k, v in (lf["largest"] or {"contrasts": {}})["contrasts"].items()},
                        "ladder": {k: keep(v) for k, v in (lf["ladder"] or {"contrasts": {}})["contrasts"].items()},
                        "under_system": {b: summ(FOOT, IN_SYSTEM[b]) for b in LEAN_BASES},
                        "under_system_contrasts": {k: keep(v) for k, v in (lf["in_system"] or {}).get("contrasts", {}).items()},
                        "lean_in_system": {"summary": summ(FOOT, "gpt61sol_combo_on_ft_lean"),
                                           "contrasts": {k: keep(v) for k, v in (lean_in_system() or {}).get("contrasts", {}).items()}},
                        "models": {k: keep(v) for k, v in (lf["models"] or {"contrasts": {}})["contrasts"].items()},
                        "who_finds_what": {"case": meas["who_case"], "pair": meas["who_pair"], "never_cited_pair": meas["who_never"]},
                        "never_share_first_ten": meas["never_share_first_ten"], "examples": meas["examples"],
                        "labels": read(RUN / "dev" / "label_completeness.json"),            # scripts/label_completeness.py
                        "pointwise": {k: v for k, v in (read(RUN / "dev" / "pointwise_rk" / "result.json") or {}).items() if k != "ranked"},  # uol.final.lean_rerun pointwise
                        "sample_models": sample_models()},
        "why": read(ROOT / "research" / "programs" / "why_models_20261103" / "no_test_case" / "results.json"),   # uol.final.why_models (addendum 23)
        "limits": read(ROOT / "research" / "programs" / "why_models_20261103" / "limits.json"),   # uol.final.model_limits (v29)
        "dev": dev, "sample": sample_table(), "sample_vs_combo": {k: keep(v) for k, v in ((sample_vs_combo() or {}).get("contrasts") or {}).items()},
        "variation": {"ranker": (summ(FOOT, RANKER) or {}).get("ndcg_at_10"),    # v33 (D203): beside the bootstrap (4.4)
                      "seeds": {f"rk_seed{k}": (summ(FOOT, f"rk_seed{k}") or {}).get("ndcg_at_10") for k in (1, 2)},   # uol.final.lean_rerun
                      "repeat": read(OUT / "repeat_passes.json")},             # research/programs/lean_rerun_20261104/repeat_passes.py
        "prompt_check": read(ROOT / "research" / "programs" / "prompt_check_20261104" / "results.json"),   # uol.final.prompt_check (D205)
        "sample_lean": {"means": {n: float(f["ndcg_at_10"].mean()) for n, f in sample_frames().items()},
                        "vs_system": {k: keep(v) for k, v in ((sample_vs_system() or {}).get("contrasts") or {}).items()},
                        "claude": {k: ({c: keep(x) for c, x in v.items()} if k == "contrasts" else v) for k, v in (sample_claude() or {}).items()
                                   if k in ("means", "repair_share", "contrasts")}},
        "later": {"alone": read(WORK / "agents" / "later" / "contrasts.json"), "more": read(WORK / "agents" / "later_more.json")},
        "families": {p.stem: read(p) for p in sorted(CON.glob("*.json"))},
        "flags": {f: (LOGS / f).exists() for f in FLAGS},
    }
    (OUT / "results.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    macros(rec)
    write_tex()
    write_md(rec)
    waiting = [k for k, v in MAC.items() if v.startswith("\\EvPending")]
    print(f"{len(MAC)} macros, {len(waiting)} waiting: {', '.join(waiting[:40])}{' ...' if len(waiting) > 40 else ''}")
    return 0


def macros(rec: dict) -> None:
    R = "research/programs/final_results_20261102/results.json"
    b, s, e = rec["benchmark"], rec["system"], rec["experiments"]
    # the benchmark
    mf, te = b["measured"], b["test"]
    put("bDevCases", b["dev_cases"], num, f"{R}: benchmark.dev_cases")
    put("bDevGroups", b["dev_groups"], num, f"{R}: benchmark.dev_groups")
    put("bTestCases", te.get("cases"), num, f"{R}: benchmark.test.cases")
    put("bTestGroups", te.get("groups"), num, f"{R}: benchmark.test.groups")
    put("bTestLater", te.get("cases_later"), num, f"{R}: benchmark.test.cases_later")
    put("bTestFunnel", te.get("cases") and te["cases"] - te["cases_later"], num, f"{R}: benchmark.test.cases - cases_later")
    put("bEvalAll", te.get("cases") and b["dev_cases"] + te["cases"], num, f"{R}: benchmark.dev_cases + test.cases")
    put("bDevInTest", te.get("dev_dated_in_test_period"), num, f"{R}: benchmark.test.dev_dated_in_test_period")
    put("bDevInTestScored", te.get("dev_dated_in_test_period_scored"), num, f"{R}: benchmark.test.dev_dated_in_test_period_scored")
    put("bAfterCutoff", te.get("after_cutoff"), num, f"{R}: benchmark.test.after_cutoff")
    put("bBeforeCutoff", te.get("cases") and te["cases"] - te["after_cutoff"], num, f"{R}: benchmark.test.cases - after_cutoff")
    put("bCutoff", te.get("cutoff"), lambda v: {"04": "30 April", "06": "1 June"}.get(v[5:7], v) + " " + v[:4], f"{R}: benchmark.test.cutoff")
    put("bCitedMedian", mf["cited_per_case_median"], lambda v: f"{v:.0f}", f"{R}: benchmark.measured.cited_per_case_median")
    put("bOneCited", mf["cases_with_one_cited"], pct, f"{R}: benchmark.measured.cases_with_one_cited")
    put("bFactsWords", mf["facts_words_median"], lambda v: f"{v:.0f}", f"{R}: benchmark.measured.facts_words_median")
    put("bArticleWords", mf["cited_article_words_median"], lambda v: f"{v:.0f}", f"{R}: benchmark.measured.cited_article_words_median")
    put("bShareCited", mf["share_cited_median"], lambda v: pct(v, 1), f"{R}: benchmark.measured.share_cited_median")
    put("bShareRandom", mf["share_random_median"], lambda v: pct(v, 1), f"{R}: benchmark.measured.share_random_median")
    # the main systems on the development cases
    for k, row in s["dev"].items():
        for m, short in SHORT.items():
            put(f"d{k}{short}", row and row.get(m), f3, f"{R}: system.dev.{k}.{m}")
    for k, row in e["cells"].items():
        put(f"d{k}Ndcg", row and row.get("ndcg_at_10"), f3, f"{R}: experiments.cells.{k}.ndcg_at_10")
        put(f"d{k}Never", row and row.get("unseen_recall_at_20_pooled"), f3, f"{R}: experiments.cells.{k}.unseen_recall_at_20_pooled")
    # the system's contrasts (family combo_full, m = 4)
    con = s["contrasts"]
    put_contrast("cSysReord", con.get(f"SYSTEM - {REORD}"), f"{R}: system.contrasts")
    put_contrast("cSysAlone", con.get("SYSTEM - gpt61sol_alone"), f"{R}: system.contrasts")
    put_contrast("cSysRank", con.get(f"SYSTEM - {RANKER}"), f"{R}: system.contrasts")
    put_contrast("cAloneReord", e["alone_reord"], f"{R}: experiments.alone_reord")
    # power of the development cases for the comparison of the system with the model alone (4.4)
    sa = con.get("SYSTEM - gpt61sol_alone") or {}
    fam_ = read(OUT / "final_dev_family.json") or {}
    full = next((v for k, v in fam_.get("contrasts", {}).items() if k.endswith(" - gpt61sol_alone")), {})
    put("pwMde", full.get("mde_improvement"), f3, "final_dev_family.json: SYSTEM - gpt61sol_alone.mde_improvement")
    put("pwMdeUseful", full.get("mde_useful"), f3, "final_dev_family.json: SYSTEM - gpt61sol_alone.mde_useful")
    # the checks on the final system (family final_checks, m = 6): merged, self and noenc printed as the system minus the check
    chk = s["settings"]["checks"]
    for v, name, neg in (("merged", "cSecond", True), ("self", "cSelf", True), ("noenc", "cSnull", True), ("both", "cTwoOrders", False), ("low", "cLow", False), ("d60", "cDsixty", False)):
        put_contrast(name, chk.get(v), f"{R}: system.settings.checks.{v}", negate=neg)
        put(f"d{name[1:]}Ndcg", chk.get(v, {}).get("mean"), f3, f"{R}: system.settings.checks.{v}.mean")
    sys_never, noenc_never = (s["dev"].get("Sys") or {}).get("unseen_recall_at_20_pooled"), (rec["dev"].get(CHECK["noenc"]) or {}).get("unseen_recall_at_20_pooled")
    put("cSnullNever", sys_never is not None and noenc_never is not None and sys_never - noenc_never or None, f3, f"{R}: system.dev.Sys - dev.{CHECK['noenc']}, unseen_recall_at_20_pooled")
    lu = chk.get("low", {}).get("usd_receipt") or {}
    nd = (s["dev"].get("Sys") or {}).get("n_cases") or 1217
    put("kLowUsd", lu.get("low") and 1000 * lu["low"] / nd, lambda x: f"{x:.1f}", f"receipts/agents_{CHECK['low']}.json: usd_list_equivalent per 1,000 development cases (forward order)")
    put("kMediumUsd", lu.get("medium") and 1000 * lu["medium"] / nd, lambda x: f"{x:.1f}", "receipts/rk_gpt61sol_rk_system.json: usd_list_equivalent per 1,000 development cases (forward order)")
    # the list the models and the rerankers read (the 28-source ranker, D185), named once in 6.5
    put("dListTwentyEight", (summ(FOOT, "S0+qwen3_8b") or {}).get("ndcg_at_10"), f3, "footing store: S0+qwen3_8b/DEV/summary.json ndcg_at_10")
    fz = s["settings"]["frozen"]
    put("sysOrders", fz.get("passes") and {1: "one", 2: "two"}[len(fz["passes"])], str, f"{R}: system.settings.frozen.passes")
    put("sysEffort", fz.get("effort"), str, f"{R}: system.settings.frozen.effort")
    put("sysDepth", fz.get("depth"), num, f"{R}: system.settings.frozen.depth")
    put("sysRrfK", CODE["RRF_K"], num, "src/uol/final/llm_agents.py: RRF_K")
    put("sysListN", CODE["list_n"], num, "src/uol/round3/prompts/closedbook_rank20.txt: 'exactamente N'")
    put("sysListPerCall", CODE["list_per_call"], num, "src/uol/eval_lenses/llm_alone.py: cases per call")
    # the ranker's questions
    put_contrast("cLean", e["lean"], f"{R}: experiments.lean")
    put_contrast("cTwentyEight", e["lean"], f"{R}: experiments.lean, printed as the 28-source ranker minus the ranker", negate=True)
    put_contrast("cTwentyEightSys", e["sys_28"], f"{R}: experiments.sys_28")
    put("dTwentyEightSysNdcg", (rec["dev"].get("gpt61sol_combo_fwd") or {}).get("ndcg_at_10"), f3, f"{R}: dev.gpt61sol_combo_fwd.ndcg_at_10")
    # the encoder's gain in the ranker (the ranker minus the ranker without the encoder, m = 1), and the ranker without
    # the encoder against tuned lexical matching (m = 1); the older macro names are kept for Table B.2
    put_contrast("lnMain", e["encoder"], f"{R}: experiments.encoder")
    put("cmMain", e["encoder"] and 1, num, f"{R}: experiments.encoder (family of one)")
    put("ccMain", e["encoder"] and e["encoder"]["class"], lambda v: CLASS[v], f"{R}: experiments.encoder.class")
    put_contrast("cNoencLex", e["noenc_vs_lex"], f"{R}: experiments.noenc_vs_lex")
    put("lnUnseen", (s["dev"].get("Rank") or {}).get("unseen_recall_at_20_pooled"), f3, f"{R}: system.dev.Rank.unseen_recall_at_20_pooled")
    put("bandNeverZero", (s["dev"].get("Noenc") or {}).get("unseen_recall_at_20_pooled"), f3, f"{R}: system.dev.Noenc.unseen_recall_at_20_pooled")
    put("lnNdcg", (s["dev"].get("Rank") or {}).get("ndcg_at_10"), f3, f"{R}: system.dev.Rank.ndcg_at_10")
    put("lnLawOracle", (b["measured_law_oracle"] or {}).get(RANKER), f3, f"{R}: benchmark.measured_law_oracle.{RANKER}")
    put("dLeanNdcg", (rec["dev"].get("ft_lean") or {}).get("ndcg_at_10"), f3, f"{R}: dev.ft_lean.ndcg_at_10")
    put("dFtRankNdcg", (rec["dev"].get("ft_ranker") or {}).get("ndcg_at_10"), f3, f"{R}: dev.ft_ranker.ndcg_at_10")
    cl = e["classifier"] or {}
    put("dClsWords", (cl.get("words") or {}).get("ndcg_at_10"), f3, f"{R}: experiments.classifier.words.ndcg_at_10")
    put("dClsEnc", (cl.get("encoder") or {}).get("ndcg_at_10"), f3, f"{R}: experiments.classifier.encoder.ndcg_at_10")
    put("dClsEncTen", (cl.get("encoder") or {}).get("success_at_10"), f3, f"{R}: experiments.classifier.encoder.success_at_10")
    put("dClsEncNever", (cl.get("encoder") or {}).get("unseen_recall_at_20"), f3, f"{R}: experiments.classifier.encoder.unseen_recall_at_20")
    # v33 (D203): the additions, rewritings and topic scores on the ranker itself (uol.final.lean_rerun)
    put_contrast("cAddTesis", e["parts"].get(f"{RANKER} - rt_check"), f"{R}: experiments.parts ({RANKER} - rt_check): the older-tesis votes")
    put("dAddTesisNdcg", (rec["dev"].get("rt_check") or {}).get("ndcg_at_10"), f3, f"{R}: dev.rt_check.ndcg_at_10 (the same ranker without the votes)")
    put_contrast("cRankChoice", e["choice"].get(f"{RANKER} - ft_lean"), f"{R}: experiments.choice ({RANKER} - ft_lean, family of six)")
    put("dSingleTreesNdcg", (rec["dev"].get("ft_lean") or {}).get("ndcg_at_10"), f3, f"{R}: dev.ft_lean.ndcg_at_10 (single-article trees, same sources, topic scores)")
    put_contrast("cRankLargest", e["largest"].get(f"{RANKER} - S_full_ext"), f"{R}: experiments.largest ({RANKER} - S_full_ext, family of one)")
    for sid, name in zip(LEAN_ADD, ("Structure", "Cocite", "Adapter", "Smooth", "Issues", "Patterns")):
        put_contrast(f"cAdd{name}", e["additions"].get(f"{sid} - {RANKER}"), f"{R}: experiments.additions.{sid} - {RANKER}")
        put(f"dAdd{name}Ndcg", (rec["dev"].get(sid) or {}).get("ndcg_at_10"), f3, f"{R}: dev.{sid}.ndcg_at_10")
    put_contrast("cListwise", e["parts"].get("rt_check - lf_plain"), f"{R}: experiments.parts.rt_check - lf_plain: whole-list against single-article trees, no votes, no topic scores")
    put("dListwiseNdcg", (rec["dev"].get("rt_check") or {}).get("ndcg_at_10"), f3, f"{R}: dev.rt_check.ndcg_at_10")
    put("dPlainNdcg", (rec["dev"].get("lf_plain") or {}).get("ndcg_at_10"), f3, f"{R}: dev.lf_plain.ndcg_at_10")
    put_contrast("cTopics", e["additions"].get(f"rk_topics - {RANKER}"), f"{R}: experiments.additions.rk_topics - {RANKER}: the four topic scores added")
    put_contrast("cQtwod", e["additions"].get(f"rk_q2d - {RANKER}"), f"{R}: experiments.additions.rk_q2d - {RANKER}")
    put_contrast("cHyde", e["additions"].get(f"rk_hyde - {RANKER}"), f"{R}: experiments.additions.rk_hyde - {RANKER}")
    for k, short in zip(LEAN_ENCODERS, ("Bge", "Qsmall", "Qmid", "Qbig")):
        put(f"dLad{short}Ndcg", (rec["dev"].get(f"rk_noenc+{k}") or {}).get("ndcg_at_10"), f3, f"{R}: dev.rk_noenc+{k}.ndcg_at_10")
        put_contrast(f"cLad{short}", e["ladder"].get(f"rk_noenc+{k} - {NOENC}"), f"{R}: experiments.ladder.rk_noenc+{k} - {NOENC}")
    put("dLargestNdcg", (rec["dev"].get("S_full_ext") or {}).get("ndcg_at_10"), f3, f"{R}: dev.S_full_ext.ndcg_at_10")
    for b_, name in zip(LEAN_BASES, ("SingleTrees", "Largest")):
        row = e["under_system"].get(b_)
        put(f"dOn{name}Ndcg", row and row.get("ndcg_at_10"), f3, f"{R}: experiments.under_system.{b_}.ndcg_at_10")
    # the models over the ranker (family models, reference GPT-6 Sol)
    for k in MODELS:                       # v33: on the ranker's list; Claude Opus 5.5 is read on the sample only
        sid = f"{k}_rerank_{RANKER}"
        put(f"dModel{MODELS[k]}", (rec["dev"].get(sid) or {}).get("ndcg_at_10"), f3, f"{R}: dev.{sid}.ndcg_at_10")
    ref = f"gpt6solnw_rerank_{RANKER}"
    put_contrast("cSolOverRef", e["models"].get(f"gpt61sol_rerank_{RANKER} - {ref}"), f"{R}: experiments.models.gpt61sol_rerank_{RANKER} - {ref}")
    put_contrast("cTerraOverRef", e["models"].get(f"gpt61terra_rerank_{RANKER} - {ref}"), f"{R}: experiments.models.gpt61terra_rerank_{RANKER} - {ref}")
    # who finds what (first ten), never-cited shares, the examples
    wc, wp, wn = e["who_finds_what"]["case"], e["who_finds_what"]["pair"], e["who_finds_what"]["never_cited_pair"]
    for key, name in (("both", "Both"), ("model_only", "ModelOnly"), ("ranker_only", "RankerOnly"), ("neither", "Neither")):
        put(f"wCase{name}", wc.get(key), pct, f"{R}: experiments.who_finds_what.case.{key}")
        put(f"wPair{name}", wp.get(key), pct, f"{R}: experiments.who_finds_what.pair.{key}")
        put(f"wNever{name}", wn.get(key), pct, f"{R}: experiments.who_finds_what.never_cited_pair.{key}")
    put("wNeverPairs", wn.get("n"), num, f"{R}: experiments.who_finds_what.never_cited_pair.n")
    ns = e["never_share_first_ten"]
    put("nsSys", ns.get("system"), pct, f"{R}: experiments.never_share_first_ten.system")
    put("nsRank", ns.get("ranker"), pct, f"{R}: experiments.never_share_first_ten.ranker")
    put("nsAlone", ns.get("alone"), pct, f"{R}: experiments.never_share_first_ten.alone")
    exs = e["examples"]
    for qid, tag in ((EXAMPLE, "exA"), (EXAMPLE_B, "exB")):
        for name, short in (("lexical", "Lex"), ("noenc", "Noenc"), ("encoder", "Enc"), ("ranker", "Rank"), ("alone", "Alone"), ("merged", "Merged"), ("system", "Sys")):
            # a ranker's list holds the first 200 of its pool: a rank past that comes from the stored runs' lexical tail, which no live list has
            fmt_ = (lambda v: "not in its list" if int(v) > 200 else str(int(v))) if name in ("noenc", "ranker") else (lambda v: str(int(v)))
            put(f"{tag}{short}", (exs.get(qid) or {}).get(name), fmt_, f"{R}: experiments.examples.{qid}.{name}")
    # cost and time
    c = s["cost"]
    for part, name in (("model_list", "List"), ("second_reading", "Read"), ("system", "Sys")):
        row = c.get(part) or {}
        put(f"k{name}Std", row.get("usd_standard_per_1000"), lambda v: f"{v:.1f}", f"{R}: system.cost.{part}.usd_standard_per_1000")
        put(f"k{name}Batch", row.get("usd_batch_per_1000"), lambda v: f"{v:.1f}", f"{R}: system.cost.{part}.usd_batch_per_1000")
        put(f"k{name}Tok", row.get("tokens_per_case"), lambda v: f"{int(round(v, -2)):,}", f"{R}: system.cost.{part}.tokens_per_case")
    put("kSysCents", (c.get("system") or {}).get("usd_standard_per_1000"), lambda v: f"{v / 10:.1f}", f"{R}: system.cost.system.usd_standard_per_1000 / 10")
    # a minute of a law graduate's time (IMCO, MXN a month, evidence/numbers_results_first.tex) over a 48-hour week, and
    # the exchange rate below which the system's cost per case at the standard tier stays under that minute
    sal = re.search(r"\\newcommand\{\\salaryLawMXN\}\{([0-9,]+)\}",
                    (ROOT / "thesis" / "evidence" / "numbers_results_first.tex").read_text(encoding="utf-8"))
    per_min = sal and int(sal.group(1).replace(",", "")) / (48 * 52 / 12 * 60)
    usd_case = (c.get("system") or {}).get("usd_standard_per_1000")
    put("lawMinMXN", per_min, lambda v: f"{v:.1f}", "numbers_results_first.tex salaryLawMXN / (48 h x 52/12 weeks x 60 min)")
    # from the printed (rounded) values, so a reader who multiplies them finds the claim holds
    put("lawRateMXN", per_min and usd_case and round(per_min, 1) / round(usd_case / 1000, 3), lambda v: f"{5 * math.floor(v / 5):.0f}",
        f"lawMinMXN as printed / (kSysCents as printed / 100), rounded down to 5")
    # GPT-6.1 Sol reordering the ranker's list, once through the provider's billed interface on every development case
    # (receipt api_ask_gpt61sol: both orders, no cached input), against the same reading costed from token counts
    api = read(RUN / "receipts" / "api_ask_gpt61sol.json") or {}
    put("apiUsd", api.get("usd") and api.get("rows") and 1000 * api["usd"] / api["rows"], lambda v: f"{v:.1f}",
        "receipts/api_ask_gpt61sol.json usd / rows x 1000 (USD per 1,000 cases, both orders, billed)")
    lv = read(ROOT / ".cache/whole_run/runs/system_stored_proto/result.json") or {}
    put("wrSysCases", lv.get("cases"), num, ".cache/whole_run/runs/system_stored_proto/result.json: cases (the whole system run live from the facts)")
    put("wrSysMed", (lv.get("seconds") or {}).get("median"), lambda v: f"{v:.0f}", "whole_run system_stored_proto: seconds.median")
    put("wrSysWait", (lv.get("model_wait") or {}).get("median"), lambda v: f"{v:.0f}", "whole_run system_stored_proto: model_wait.median")
    put("wrSysCents", lv.get("usd_per_case"), lambda v: f"{100 * v:.1f}", "whole_run system_stored_proto: usd_per_case x 100 (one case per call, standard tier)")
    nl_ = read(ROOT / ".cache/whole_run/runs/noenc_none_a/result.json") or {}
    put("wrNoencCases", nl_.get("cases"), num, ".cache/whole_run/runs/noenc_none_a/result.json: cases (the ranker without the encoder run live)")
    rt = c.get("ranker_time") or {}
    put("kRankMed", (c.get("ranker") or {}).get("seconds_median"), lambda v: f"{v:.1f}", f"{R}: system.cost.ranker.seconds_median")
    put("kRankPff", (c.get("ranker") or {}).get("seconds_p95"), lambda v: f"{v:.1f}", f"{R}: system.cost.ranker.seconds_p95")
    put("kNoencMed", (c.get("ranker_noenc") or {}).get("seconds_median"), lambda v: f"{v:.2f}", f"{R}: system.cost.ranker_noenc.seconds_median (the live run)")
    put("kNoencPff", (c.get("ranker_noenc") or {}).get("seconds_p95"), lambda v: f"{v:.2f}", f"{R}: system.cost.ranker_noenc.seconds_p95 (the live run)")
    put("kRankRam", rt.get("ram_gb"), lambda v: f"{v:.1f}", f"{R}: system.cost.ranker_time.ram_gb")
    put("kRankVram", rt.get("vram_gb"), lambda v: f"{v:.1f}", f"{R}: system.cost.ranker_time.vram_gb")
    lrr = (read(ROOT / ".cache/whole_run/runs/ranker_card_a/result.json") or {}).get("machine") or {}
    put("kRankRamLive", lrr.get("peak_rss_gb"), lambda v: f"{v:.1f}", "whole_run ranker_card_a: machine.peak_rss_gb (the ranker's process on the i9-14900KF; the encoder runs in its own process)")
    usd_rank = (c.get("ranker") or {}).get("usd_per_1000_median")
    put("lnUsd", usd_rank, lambda v: f"{v:.3f}", f"{R}: system.cost.ranker.usd_per_1000_median")
    put("ctUsdSzero", (c.get("ranker_noenc") or {}).get("usd_per_1000_median"), lambda v: f"{v:.3f}", f"{R}: system.cost.ranker_noenc.usd_per_1000_median")
    # the ranker that reads every source (S_full_ext, Section 6.3) as a local option (v31, the author 3 Nov: "graphics memory and
    # cost per case"): its whole method timed on 200 development cases treated as new (cost_timing_20261028), and its
    # graphics memory as the ranker's measured Qwen3-Embedding-8B peak plus the peaks of its other encoders, bge-m3 with
    # ColBERT and an e5-small for each of its two trained e5 encoders (timing_20261026d, one process per encoder)
    ct = read(ROOT / "research" / "reports" / "cost_timing_20261028" / "RESULT.json") or {}
    put("kExtUsd", (ct.get("usd_per_1000_timed") or {}).get("S_full_ext"), lambda v: f"{v:.3f}",
        "research/reports/cost_timing_20261028/RESULT.json: usd_per_1000_timed.S_full_ext")
    put("kExtMed", ((ct.get("timed") or {}).get("S_full_ext") or {}).get("wall_median"), lambda v: f"{v:.1f}",
        "research/reports/cost_timing_20261028/RESULT.json: timed.S_full_ext.wall_median")
    enc = {m: (read(ROOT / "experiments" / "runs" / "timing_20261026d" / "results" / "encode" / f"{m}.json") or {}).get("peak_gb")
           for m in ("bge", "e5proxy")}
    put("kExtVram", rt.get("vram_gb") and all(enc.values()) and rt["vram_gb"] + enc["bge"] + 2 * enc["e5proxy"], lambda v: f"{v:.0f}",
        f"{R}: system.cost.ranker_time.vram_gb + timing_20261026d encode bge.peak_gb + 2 x e5proxy.peak_gb")
    # the eight-bit query encoder against the stored full-precision query vectors, Qwen3-Embedding-8B alone on every
    # development case (uol.final.query_int8, MR job query_int8_dev_j1, v31)
    q8 = read(RUN / "int8" / "RESULT.json") or {}
    put_contrast("cIntQ", q8.get("contrast_int8_minus_full"), "footing_20261101/int8/RESULT.json: contrast_int8_minus_full")
    put("intQCos", (q8.get("cos") or {}).get("median"), lambda v: f"{v:.3f}", "footing_20261101/int8/RESULT.json: cos.median")
    put("sxQbigInt", q8.get("ndcg_int8"), f3, "footing_20261101/int8/RESULT.json: ndcg_int8")
    # the labels against a second answer of the courts (scripts/label_completeness.py)
    lb = e.get("labels") or {}
    ag = lb.get("agreement") or {}
    put("lbGroups", ag.get("groups_with_related"), num, f"{R}: experiments.labels.agreement.groups_with_related")
    put("lbShare", ag.get("union_share_mean"), lambda v: pct(v), f"{R}: experiments.labels.agreement.union_share_mean")
    put("lbSame", ag.get("groups_all_same_labels"), num, f"{R}: experiments.labels.agreement.groups_all_same_labels")
    put("lbWide", lb.get("eval_cases_widened"), num, f"{R}: experiments.labels.eval_cases_widened")
    for lab, tag in (("own_widened", "Own"), ("union_widened", "Union"), ("union", "All")):
        fc = ((lb.get("families") or {}).get(lab) or {}).get("contrasts") or {}
        put_contrast(f"lb{tag}SysAlone", fc.get("system - alone"), f"{R}: experiments.labels.families.{lab}")
        put_contrast(f"lb{tag}AloneRank", fc.get("alone - ranker"), f"{R}: experiments.labels.families.{lab}")
    # the same model judging each candidate alone (uol.final.pointwise)
    pw = e.get("pointwise") or {}
    pws = pw.get("scores_case") or {}   # means per case, as tab:model-ways prints the same runs
    put("pwNdcg", pws.get("pointwise"), f3, f"{R}: experiments.pointwise.scores_case.pointwise")
    put("pwList", pws.get("listwise"), f3, f"{R}: experiments.pointwise.scores_case.listwise")
    put("pwRank", pws.get("ranker"), f3, f"{R}: experiments.pointwise.scores_case.ranker")
    pc_ = ((pw.get("family") or {}).get("contrasts")) or {}
    put_contrast("pwOverRank", pc_.get("pointwise - ranker"), f"{R}: experiments.pointwise.family")
    put_contrast("pwListOver", pc_.get("listwise - pointwise"), f"{R}: experiments.pointwise.family")
    # the price of each step, read by the figures' own cost readers (uol.thesis_figures.v21_figs), so text and figures agree
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from uol.thesis_figures import v21_figs as V
        mc = V.model_costs()
    except Exception as err:  # noqa: BLE001 - a missing reader leaves these macros waiting
        print(f"model costs unavailable: {err}", file=sys.stderr)
        mc = {}
    # the models read on the 300-case sample only
    sm_ = e.get("sample_models") or {}
    for key, short in (("sonnet", "Sonnet"), ("haiku", "Haiku"), ("gpt6astra", "Astra"), ("gpt6sol", "SolSix"), ("ranker", "Rank")):
        put(f"sm{short}", (sm_.get("scores") or {}).get(key), f3, f"{R}: experiments.sample_models.scores.{key}")
    put_contrast("smAstraOver", (sm_.get("vs_gpt6sol") or {}).get("gpt6astra - sol_medium"), f"{R}: experiments.sample_models.vs_gpt6sol")
    put("smSonnetRep", (sm_.get("repairs") or {}).get("sonnet"), num, f"{R}: experiments.sample_models.repairs.sonnet")
    put("smSonnetWin", (sm_.get("windows") or {}).get("sonnet"), num, f"{R}: experiments.sample_models.windows.sonnet")
    put("smAstraUsd", (sm_.get("usd_standard_per_1000") or {}).get("gpt6astra"), lambda v: f"{v:.0f}",
        f"{R}: experiments.sample_models.usd_standard_per_1000.gpt6astra (list price)")
    for key, short in (("gpt6luna", "LunaSix"), ("gpt61sol", "SolSixOne"), ("gpt6solnw", "SolSix"), ("claude", "Opus")):
        put(f"kModel{short}", (mc.get(key) or {}).get("std"), lambda v: f"{v:.1f}",
            f"v21_figs.model_costs(): {key}.std (USD per 1,000 cases, both orders, standard tier)")
    lst, tot = (c.get("model_list") or {}).get("usd_standard_per_1000"), (c.get("system") or {}).get("usd_standard_per_1000")
    put("kxAloneRank", lst and usd_rank and lst / usd_rank, lambda v: f"{v:.0f}",
        f"{R}: system.cost.model_list.usd_standard_per_1000 / system.cost.ranker")
    put("kxSysAlone", lst and tot and tot / lst, lambda v: f"{v:.1f}", f"{R}: system.cost.system / model_list, usd_standard_per_1000")
    # the abstract and resumen (3 Nov 2026): the system against keyword search, and the ranker against the system, in words
    put("kxSysRank", tot and usd_rank and float(MAC["kSysStd"]) / float(MAC["lnUsd"]), lambda v: f"{int(round(v, -1)):,}",
        f"kSysStd as printed / lnUsd as printed, to the nearest ten")
    words = {2: ("dos", "two"), 3: ("tres", "three"), 4: ("cuatro", "four")}
    x_lex = float(MAC["dSysNdcg"]) / float(MAC["dLexNdcg"])
    put("xSysLexEs", words.get(int(x_lex), (None,))[0], str, "dSysNdcg / dLexNdcg, whole part in words (Spanish)")
    put("xSysLexEn", words.get(int(x_lex), (None, None))[1], str, "dSysNdcg / dLexNdcg, whole part in words (English)")
    put("pctRankBelowSys", 100 * (1 - float(MAC["dRankNdcg"]) / float(MAC["dSysNdcg"])), lambda v: f"{5 * round(v / 5):.0f}",
        "1 - dRankNdcg / dSysNdcg, in per cent to the nearest five")
    c_ = s["cost"]
    # the 300-case sample: every way of using the model, on the same cases, and each minus the combination
    sm_ = rec["sample"]
    names = {"alone": "Alone", "repeat": "Repeat", "examples": "Examples", "list_agent_words": "AgentWords", "agent_embedding": "AgentEmb",
             "list_agent_embedding": "ListEmb", "combo": "Combo", "sum_model_ranker": "Merged"}
    for n, short in names.items():
        put(f"s{short}", (sm_.get(n) or {}).get("mean_per_case"), f3, f"{R}: sample.{n}.mean_per_case")
        put(f"s{short}Tok", (sm_.get(n) or {}).get("tokens_per_case"), lambda v: f"{int(round(v, -3)):,}", f"{R}: sample.{n}.tokens_per_case")
    # the combination as run on the sample: its second reading in both orders plus the model's list (all development cases)
    ct, lt = (sm_.get("combo") or {}).get("tokens_per_case"), ((s["cost"].get("model_list") or {}).get("tokens_per_case"))
    put("sComboAllTok", ct is not None and lt is not None and ct + lt or None, lambda v: f"{int(round(v, -3)):,}",
        f"{R}: sample.combo.tokens_per_case + system.cost.model_list.tokens_per_case")
    # v33 (D203): the ranker, its reranking and the system on the sample cases, and each way minus the system
    sl = rec["sample_lean"]
    put("sReord", sl["means"].get("reord"), f3, f"{R}: sample_lean.means.reord ({REORD} on the sample cases)")
    put("sRank", sl["means"].get("ranker"), f3, f"{R}: sample_lean.means.ranker ({RANKER} on the sample cases)")
    put("sSys", sl["means"].get("system"), f3, f"{R}: sample_lean.means.system ({SYSTEM} on the sample cases)")
    for n in ("list_agent_words", "list_agent_embedding"):
        put(f"sLean{names[n]}", sl["means"].get(n), f3, f"{R}: sample_lean.means.{n} (from the ranker's list)")
    for n in SAMPLE_VS:
        put_contrast(f"sc{names[n]}", sl["vs_system"].get(f"{n} - system"), f"{R}: sample_lean.vs_system.{n} - system")
    cl_ = sl.get("claude") or {}
    for k, short in CLAUDE_SAMPLE.items():          # v33 (D203): the Claude models on the ranker's list, sample only
        put(f"sCl{short}", (cl_.get("means") or {}).get(k), f3, f"{R}: sample_lean.claude.means.{k}")
        put_contrast(f"scCl{short}", (cl_.get("contrasts") or {}).get(f"{k} - ranker"), f"{R}: sample_lean.claude.contrasts.{k} - ranker")
        put(f"sCl{short}Rep", (cl_.get("repair_share") or {}).get(k), f3, f"{R}: sample_lean.claude.repair_share.{k} (window orders missing candidates, a share)")
    va = rec["variation"]                           # v33 (D203): refitting the trees with two other seeds; two more answers
    sd = [abs(v - va["ranker"]) for v in va["seeds"].values() if v is not None and va["ranker"] is not None]
    put("vSeedMax", max(sd) if len(sd) == 2 else None, f3, f"{R}: variation.seeds, largest |lf_seedk - {RANKER}| (mean over groups)")
    put("vAskMax", (va.get("repeat") or {}).get("max_abs_vs_first"), f3, f"{R}: variation.repeat.max_abs_vs_first (sample, mean per case)")
    pc_ = rec.get("prompt_check") or {}                # D205: how the model alone is asked, 6.6 (research/programs/prompt_check_20261104)
    pf_ = pc_.get("facts") or {}
    PC = "research/programs/prompt_check_20261104/results.json"
    for short, v in (("Short", "list10"), ("Long", "list40"), ("One", "one_call")):
        put(f"pc{short}", (pf_.get(v) or {}).get("ndcg_at_10_per_case"), f3, f"{PC}: facts.{v}.ndcg_at_10_per_case")
    put("pcShortN", (pf_.get("list10") or {}).get("asked_for"), num, f"{PC}: facts.list10.asked_for")
    put("pcLongN", (pf_.get("list40") or {}).get("asked_for"), num, f"{PC}: facts.list40.asked_for")
    pcc = {cls_of(c) for c in (pc_.get("contrasts") or {}).values()}
    put("pcCls", pcc.pop() if len(pcc) == 1 else None, lambda v: CLASS[v], f"{PC}: the class of the family's three contrasts, one class only")
    # power of the sample (audit v25): the true gain whose adjusted lower end reaches the margin with 80% probability
    mde = [v.get("mde_useful") for v in ((sample_vs_combo() or {}).get("contrasts") or {}).values() if v.get("mde_useful") is not None]
    put("pwSampleUseful", mde and max(mde), lambda v: f"{v:.2f}", "sample_vs_combo.json: max over the family of mde_useful")
    for k, short in (("gpt61terra", "TerraSixOne"), ("gpt6solnw", "SolSix"), ("gpt56sol", "SolFive"), ("gpt6luna", "LunaSix"), ("gpt56terra", "TerraFive"),
                     ("luna", "LunaFive")):
        put(f"sAlone{short}", (sm_.get(f"alone_{k}") or {}).get("mean_per_case"), f3, f"{R}: sample.alone_{k}.mean_per_case")
    # why the model works (common_footing PLAN addendum 23, the same 300 cases): effort, the Claude models alone, whether the
    # model recognises the tesis, the model on reworded facts against two methods that cannot remember, and the lead by year
    W = rec.get("why") or {}
    eff, ec = W.get("w1_alone_effort") or {}, W.get("w1_contrasts") or {}
    for k, short in (("low", "Low"), ("high", "High"), ("xhigh", "Xhigh"), ("max", "Max")):
        put(f"whyEff{short}", eff.get(k), f3, f"{R}: why.w1_alone_effort.{k}")
        put_contrast(f"whycEff{short}", ec.get(f"alone {k} - alone medium"), f"{R}: why.w1_contrasts")
    cl = W.get("w2_claude") or {}
    for k, short in (("opus_low", "OpusLow"), ("opus_medium", "OpusMed"), ("opus_high", "OpusHigh"), ("sonnet_medium", "Sonnet"),
                     ("haiku_medium", "Haiku")):
        put(f"why{short}", (cl.get(k) or {}).get("alone"), f3, f"{R}: why.w2_claude.{k}.alone")
        put(f"why{short}List", (cl.get(k) or {}).get("on_list_same_cases"), f3, f"{R}: why.w2_claude.{k}.on_list_same_cases")
    pr = W.get("w3_probe") or {}
    for k, short in (("gpt61sol", "Sol"), ("gpt6solnw", "SolSix"), ("opus", "Opus")):
        p = pr.get(k) or {}
        for f_, name in (("recognized", "Rec"), ("recognized_loose", "RecLoose"), ("registro_exact", "Reg"), ("year_exact", "Year"),
                         ("rubro_given", "Given")):
            put(f"whyProbe{short}{name}", p.get(f_), pct, f"{R}: why.w3_probe.{k}.{f_}")
        put(f"whyProbe{short}N", p.get("n_recognized"), num, f"{R}: why.w3_probe.{k}.n_recognized")
        put(f"whyProbe{short}Unrec", p.get("alone_unrecognized"), f3, f"{R}: why.w3_probe.{k}.alone_unrecognized")
        put(f"whyProbe{short}ListUnrec", p.get("list_unrecognized"), f3, f"{R}: why.w3_probe.{k}.list_unrecognized")
        put_contrast(f"whycProbe{short}", next(iter((p.get("contrast_unrecognized_alone_minus_list") or {}).values()), None),
                     f"{R}: why.w3_probe.{k}.contrast_unrecognized_alone_minus_list")
    rw = W.get("w4_reword") or {}
    rwc, rwr = rw.get("contrasts") or {}, rw.get("reword") or {}
    put("whyRwKept", rwr.get("kept"), num, f"{R}: why.w4_reword.reword.kept")
    put("whyRwGrams", rwr.get("share_4grams_kept_mean"), pct, f"{R}: why.w4_reword.reword.share_4grams_kept_mean")
    put("whyRwOrig", rw.get("model_original"), f3, f"{R}: why.w4_reword.model_original")
    put("whyRwNew", rw.get("model_reworded"), f3, f"{R}: why.w4_reword.model_reworded")
    for c, short in (("model", "Model"), ("encoder", "Enc"), ("bm25", "Lex")):
        o, n = (rw.get("model_original"), rw.get("model_reworded")) if c == "model" else (rw.get(f"{c}_original"), rw.get(f"{c}_reworded"))
        put(f"whyRwDrop{short}", o is not None and n is not None and o - n or None, f3, f"{R}: why.w4_reword, {c} original - reworded")
        if c != "model":
            put(f"whyRw{short}Orig", o, f3, f"{R}: why.w4_reword.{c}_original")
            put(f"whyRw{short}New", n, f3, f"{R}: why.w4_reword.{c}_reworded")
            put_contrast(f"whycRw{short}", rwc.get(f"model drop - {c} drop"), f"{R}: why.w4_reword.contrasts")
    yr = ((W.get("w5") or {}).get("by_year")) or {}
    for y in ("2021", "2025"):
        put(f"whyLead{'One' if y == '2021' else 'Five'}", (yr.get(y) or {}).get("lead"), sg, f"{R}: why.w5.by_year.{y}.lead")
    put("whyLeadSlope", (W.get("w5") or {}).get("lead_slope_per_year"), lambda v: f"{v:+.3f}", f"{R}: why.w5.lead_slope_per_year")
    # where the model alone falls short, and how far the catalogue's texts have moved (uol.final.model_limits; stored lists)
    lm_, lc_ = (rec.get("limits") or {}).get("model_alone") or {}, (rec.get("limits") or {}).get("catalogue") or {}
    LR = "limits.json"
    put("limNamedMed", lm_.get("named_median"), num, f"{LR}: model_alone.named_median")
    put("limNamedMax", lm_.get("named_max"), num, f"{LR}: model_alone.named_max")
    put("limBlind", lm_.get("blind_cases"), num, f"{LR}: model_alone.blind_cases")
    put("limBlindRankHundred", lm_.get("blind_ranker_first100"), pct, f"{LR}: model_alone.blind_ranker_first100")
    put("limBlindRankTen", lm_.get("blind_ranker_first10"), pct, f"{LR}: model_alone.blind_ranker_first10")
    put("limBlindSysTen", lm_.get("blind_system_first10"), pct, f"{LR}: model_alone.blind_system_first10")
    put("limBand", lm_.get("band"), num, f"{LR}: model_alone.band")
    put("limNamedOften", lm_.get("named_often_cited_laws"), pct, f"{LR}: model_alone.named_often_cited_laws")
    put("limNamedOther", lm_.get("named_other_laws"), pct, f"{LR}: model_alone.named_other_laws")
    put("catReformsYear", lc_.get("articles_reformed_per_year_mean"), lambda v: f"{int(round(v, -2)):,}", f"{LR}: catalogue.articles_reformed_per_year_mean, to the hundred")
    put("catUnchanged", lc_.get("cited_unchanged_share"), pct, f"{LR}: catalogue.cited_unchanged_share")
    put("catLaterOther", lc_.get("later_cases") and lc_.get("later_cases_other_wording", 0) / lc_["later_cases"], pct,
        f"{LR}: catalogue.later_cases_other_wording / later_cases")
    put("catDevOther", lc_.get("dev_cases_p14") and lc_.get("dev_cases_other_wording", 0) / lc_["dev_cases_p14"], pct,
        f"{LR}: catalogue.dev_cases_other_wording / dev_cases_p14")
    # the choice between the combination and the searching agent, on the 46 most recent cases (now part of the test set)
    lm = (rec["later"].get("more") or {})
    # what each family depends on (figure F6_16): Recall@20 per cited pair by quarter of shared words and by citation band
    dep = read(OUT / "depends.json") or {}
    for short, sid in (("Lex", "lstar_headings"), ("Enc", "qwen3_8b"), ("Rank", RANKER), ("Alone", "gpt61sol_alone"), ("Sys", SYSTEM)):
        w_, b_ = depends_bins(dep, "words", sid), depends_bins(dep, "citations", sid)
        put(f"dw{short}Low", w_ and w_[0], f3, f"depends.json: words.{sid}, lowest quarter")
        put(f"dw{short}High", w_ and w_[-1], f3, f"depends.json: words.{sid}, highest quarter")
        put(f"db{short}Most", b_ and b_[-1], f3, f"depends.json: citations.{sid}, cited more than ten times")
    edges = dep.get("quarter_edges") or []
    put("dwEdgeLow", edges[0] if edges else None, lambda v: pct(v, 1), "depends.json: quarter_edges[0]")
    put("dwEdgeHigh", edges[-1] if edges else None, lambda v: pct(v, 1), "depends.json: quarter_edges[-1]")
    hm = (lm.get("means") or {}).get("headline") or {}
    lcon = lm.get("contrasts") or {}
    gm = {}  # means over groups, as every other score, from the paired contrasts; the stored means are over cases
    for key, c in lcon.items():
        if key.startswith("headline: ") and c:
            a, b = key[len("headline: "):].split(" - ")
            gm.setdefault(a, c["mean_a"])
            gm.setdefault(b, c["mean_b"])
    for k, short in (("combo", "Combo"), ("list_agent_words", "Agent"), ("alone", "Alone"), ("gpt61sol_rerank_S0+qwen3_8b", "Reord"), ("S0+qwen3_8b", "Rank")):
        put(f"lc{short}", gm.get(k, hm.get(k)), f3, f"{R}: later.more.contrasts.headline (mean over groups) or means.headline.{k}")
    put_contrast("lcComboAlone", lcon.get("headline: combo - alone"), f"{R}: later.more.contrasts")
    put_contrast("lcAgentAlone", lcon.get("headline: list_agent_words - alone"), f"{R}: later.more.contrasts")
    put_contrast("lcComboAgent", (read(WORK / "agents" / "later_combo_vs_agent.json") or {}).get("headline: combo - list_agent_words"), ".cache/footing/agents/later_combo_vs_agent.json (research/programs/common_footing_20261101/later_combo_vs_agent.py)")
    # supporting cases in the first ten (scripts/support_share.py), and the model's call times
    sp = read(OUT / "support.json") or {}
    for sid_, short in ((RANKER, "Rank"), ("gpt61sol_alone", "Alone"), (sp.get("system_id"), "Sys")):
        row = sp.get(sid_) or {}
        put(f"sp{short}", row.get("share"), f3, f"research/programs/final_results_20261102/support.json: {sid_}.share")
        put(f"sp{short}CI", row.get("lo") is not None and [row["lo"], row["hi"]] or None, lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]", f"support.json: {sid_}.lo, hi")
    put("spSysIsFinal", "yes" if sp.get("system_id") == s["dev_system"] else None, str, "support.json: system_id = system.dev_system")
    for part, name in (("model_list", "List"), ("second_reading", "Read")):
        row = c_.get(part) or {}
        put(f"k{name}Lat", row.get("call_latency_median_s"), lambda v: f"{v:.0f}", f"{R}: system.cost.{part}.call_latency_median_s")
        put(f"k{name}Call", row.get("cases_per_call"), num, f"{R}: system.cost.{part}.cases_per_call")
    # other ways of reordering the ranker's list: cross-encoders over all cases, the pointwise judge on the sample
    for k, short in LEAN_RR.items():        # v33: over the ranker's first 30
        put(f"d{short}Ndcg", (rec["dev"].get(f"{k}_lean") or {}).get("ndcg_at_10"), f3, f"{R}: dev.{k}_lean.ndcg_at_10")
    jd = read(RECEIPTS / "judge.json") or {}
    put("sJudge", (jd.get("means") or {}).get("jev"), f3, "receipts/judge.json: means.jev")
    put("sJudgeRank", jd.get("ranker"), f3, "receipts/judge.json: ranker")
    # what the model names when it works alone (written with figure F6_15), and the second reading's headroom (F6_9)
    na = read(OUT / "named_articles.json") or {}
    for k, short in (("cases", "Cases"), ("named", "Named")):
        put(f"na{short}", na.get(k), num, f"named_articles.json: {k}")
    for k, short in (("cited", "Cited"), ("near", "Near"), ("same_law_other", "SameLaw"), ("other_law", "OtherLaw"), ("not_in_catalogue", "Missing")):
        v = na.get(k)
        put(f"na{short}", v / na["named"] if v is not None and na.get("named") else None, pct, f"named_articles.json: {k} / named")
    hr = read(OUT / "headroom.json") or {}
    put("hrRecall", hr.get("share_cited_in_first_30_of_merged"), pct, "headroom.json: share_cited_in_first_30_of_merged")
    put("hrOracle", hr.get("perfect_reorder_ndcg_at_10"), f3, "headroom.json: perfect_reorder_ndcg_at_10")
    # additions read inside the system (each base's ranker in place of the ranker, everything else as the combination)
    for b_, name in zip(LEAN_BASES, ("SingleTrees", "Largest")):
        put_contrast(f"cOn{name}", (rec["experiments"].get("under_system_contrasts") or {}).get(f"{IN_SYSTEM[b_]} - {SYSTEM}"),
                     f"{R}: experiments.under_system_contrasts")
    # E1 (addendum 21): the lean ranker inside the system
    li = rec["experiments"].get("lean_in_system") or {}
    put_contrast("cLeanSys", (li.get("contrasts") or {}).get("gpt61sol_combo_on_ft_lean - gpt61sol_combo"), f"{R}: experiments.lean_in_system")
    put("dLeanSysNdcg", (li.get("summary") or {}).get("ndcg_at_10"), f3, f"{R}: experiments.lean_in_system.summary.ndcg_at_10")
    # the ranker as built
    rk = s["ranker"]
    put("rkSources", rk.get("sources"), num, f"{R}: system.ranker.sources")
    put("rkSourcesNoenc", rk.get("sources_without_encoder"), num, f"{R}: system.ranker.sources_without_encoder")
    put("rkSourcesEnc", rk.get("encoder_sources"), num, f"{R}: system.ranker.encoder_sources")
    put("rkTopic", rk.get("topic_models"), num, f"{R}: system.ranker.topic_models")
    put("rkInputs", rk.get("inputs"), num, f"{R}: system.ranker.inputs")
    put("rkTopPer", rk.get("top_per_source"), num, f"{R}: system.ranker.top_per_source")
    put("rkListLen", 200, num, "uol.final.whole_run Live.ranker_list: the first 200 of the ordered pool, generic articles out")
    put("rkPool", rk.get("pool_per_case"), lambda v: f"{int(round(v, -1)):,}", f"{R}: system.ranker.pool_per_case")
    put("rkPoolRecall", rk.get("pool_recall"), pct, f"{R}: system.ranker.pool_recall")
    put("rkNeg", rk.get("negatives_per_case"), num, f"{R}: system.ranker.negatives_per_case")
    put("rkTrees", rk.get("trees"), num, f"{R}: system.ranker.trees")
    put("rkLeaf", rk.get("min_leaf"), num, f"{R}: system.ranker.min_leaf")
    put("rkDeepFolds", rk.get("folds_lr05_leaves15"), lambda v: {4: "four", 5: "all five", 3: "three"}.get(v, str(v)), f"{R}: system.ranker.folds_lr05_leaves15")
    put("rkVotesWordFolds", rk.get("folds_votes_by_words"), lambda v: {1: "one", 2: "two", 0: "none"}.get(v, str(v)), f"{R}: system.ranker.folds_votes_by_words")
    put("rkWeight", rk.get("second_reading_weight"), lambda v: f"{v:.0f}", f"{R}: system.ranker.second_reading_weight")
    # the ranker as a gate on the second reading, development cases only (scripts/gate_dev.py; 7.4, v27)
    gt = read(OUT / "gate.json") or {}
    G_ = "research/programs/final_results_20261102/gate.json"
    put("gateShare", gt.get("share_skipped"), pct, f"{G_}: share_skipped")
    put("gateNdcg", gt.get("gate_ndcg"), f3, f"{G_}: gate_ndcg")
    put_contrast("gateDiff", gt.get("gate_minus_system"), f"{G_}: gate_minus_system (family of one)")
    sk, rd, sy_ = gt.get("share_skipped"), MAC.get("kReadStd"), MAC.get("kSysStd")
    ok_ = sk is not None and rd and sy_ and not rd.startswith("\\EvPending") and not sy_.startswith("\\EvPending")
    put("gateUsd", ok_ and float(sy_) - sk * float(rd) or None, lambda v: f"{v:.1f}", "kSysStd - share_skipped x kReadStd, as printed")
    put("gateSave", ok_ and sk * float(rd) / float(sy_) or None, pct, "share_skipped x kReadStd / kSysStd, as printed")
    ct_ = gt.get("confident_tenth") or {}
    put("gateTopRank", ct_.get("ranker"), f3, f"{G_}: confident_tenth.ranker")
    put("gateTopSys", ct_.get("system"), f3, f"{G_}: confident_tenth.system")
    # the test set, once (PLAN addendum 22, test_final/scored.json)
    sc = s["test"] or {}
    sm = sc.get("summaries", {})
    T = sc.get("_source", "test_final/scored.json")
    for part, tag in (("all", "t"), ("without_later_46", "tw"), ("after_model_cutoff", "ta"), ("before_model_cutoff", "tb")):
        for k, short in (("combo", "Sys"), ("alone", "Alone"), ("ranker", "Rank"), ("ranker_noenc", "Noenc")):
            row = (sm.get(part) or {}).get(k) or {}
            put(f"{tag}{short}Ndcg", row.get("ndcg_at_10"), f3, f"{T}: summaries.{part}.{k}.ndcg_at_10")
            if part == "all":
                for m in ("success_at_10", "success_at_1", "law_at_1", "unseen_recall_at_20_pooled", "recall_at_20"):
                    put(f"t{short}{SHORT[m]}", row.get(m), f3, f"{T}: summaries.all.{k}.{m}")
        a_, r_ = ((sm.get(part) or {}).get("alone") or {}).get("ndcg_at_10"), ((sm.get(part) or {}).get("ranker") or {}).get("ndcg_at_10")
        put(f"{tag}cAloneRank", None if a_ is None or r_ is None else a_ - r_, sg, f"{T}: summaries.{part}, alone - ranker (means over groups)")
    for part, tag in (("all", "tc"), ("without_later_46", "twc"), ("after_model_cutoff", "tac"), ("before_model_cutoff", "tbc")):
        cc = sc.get("contrasts", {})
        put_contrast(f"{tag}SysAlone", cc.get(f"{part}: combo - alone"), f"{T}: contrasts.{part}: combo - alone")
        put(f"{tag}SysAloneMde", (cc.get(f"{part}: combo - alone") or {}).get("mde_improvement"), f3, f"{T}: contrasts.{part}: combo - alone.mde_improvement")
        put_contrast(f"{tag}SysRank", cc.get(f"{part}: combo - ranker"), f"{T}: contrasts.{part}: combo - ranker")
        put_contrast(f"{tag}RankNoenc", cc.get(f"{part}: ranker - ranker_noenc"), f"{T}: contrasts.{part}: ranker - ranker_noenc")
    # v33 (D203, METHODS_REVIEW G3): what the comparisons can resolve, for 4.4. The median half-width of a 95% interval
    # among the development contrasts on every case (the widest belong to large differences, such as the system minus
    # the ranker), and the half-width of the system minus the model alone after the cutoff
    hw = [(c["ci95"][1] - c["ci95"][0]) / 2 for p in sorted(OUT.glob("*.json"))
          for c in ((read(p) or {}).get("contrasts") or {}).values()
          if isinstance(c, dict) and c.get("population") == "DEV" and c.get("ci95")]
    put("statHalfDevMed", statistics.median(hw) if hw else None, f3, f"{OUT.name}/*.json: median over the DEV contrasts of (ci95 hi - lo) / 2")
    ac = (sc.get("contrasts") or {}).get("after_model_cutoff: combo - alone")
    put("statHalfAfter", ac and (ac["ci95"][1] - ac["ci95"][0]) / 2, f3, f"{T}: contrasts.after_model_cutoff: combo - alone, (ci95 hi - lo) / 2")
    # methods review G1 (D203): the final order's weight on stored outputs (uol.final.weight_check)
    wc = read(ROOT / "research" / "programs" / "whole_run_20261105" / "weight_check.json") or {}
    put_contrast("cOrderAlone", ((wc.get("contrasts") or {}).get(f"model_order_alone - {wc.get('system')}")),
                 "whole_run_20261105/weight_check.json: the model's order alone - the system (DEV, 100,000 draws)")
    put("wFoldsTop", wc.get("folds_choosing_top"), lambda v: ("none", "one", "two", "three", "four", "five")[int(v)], "whole_run_20261105/weight_check.json: folds whose other four choose w = 3")
    framing_macros()
    s6_macros()
    s7_macros()


def s7_macros() -> None:
    """The values sections 6.5, 6.7 and 6.9 ask of the ranker session (S7_macro_request.md). Development cases, stored runs."""
    res = read(OUT / "results.json") or {}
    dev = res.get("dev") or {}
    g = lambda sid, k="ndcg_at_10": (dev.get(sid) or {}).get(k)
    for short, sid in (("Qwen", "qwen4b_ranker_rk"), ("Bge", "bge_ranker_rk"), ("Trained", "ce_ranker_rk")):
        put(f"dCe{short}Rk", g(sid), f3, f"results.json: dev.{sid}.ndcg_at_10 (fused with the ranker's list)")
        put(f"dCe{short}Alone", g(sid + "_alone"), f3, f"results.json: dev.{sid}_alone.ndcg_at_10")
    put("dCeQwenAloneNeverPct", g("qwen4b_ranker_rk_alone", "unseen_recall_at_20_pooled"), pct, "results.json: dev.qwen4b_ranker_rk_alone.unseen_recall_at_20_pooled")
    s7 = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/s7_macros.json") or {}
    T7 = "rank_rerun_20261105/reads/out/s7_macros.json"
    put("dCeQwenAloneMostPct", s7.get("ce_qwen_alone_most"), pct, f"{T7}: ce_qwen_alone_most (Recall@20 pooled, cited more than ten times, as dbRankMost)")
    for k, short in (("low", "Low"), ("high", "High")):
        put(f"dAlone{short}Ndcg", g(f"gpt61sol_alone_{k}"), f3, f"results.json: dev.gpt61sol_alone_{k}.ndcg_at_10")
        put_contrast(f"cAlone{short}", s7.get(f"alone_{k}_minus_medium"), f"{T7}: alone_{k}_minus_medium (family of two)")
    # the encoder in the system, never-cited Recall@100 (band 0: cases, with, without, difference, CI)
    ev = ((read(ROOT / "research/programs/overnight_20261105/encoder_in_system.json") or {}).get("recall_at_100") or {}).get("band 0")
    TE = "overnight_20261105/encoder_in_system.json: recall_at_100.band 0"
    put("dSysNeverHundredPct", ev and ev[1], pct, f"{TE} [1], the system")
    put("dSnullNeverHundredPct", ev and ev[2], pct, f"{TE} [2], the system without the encoder")
    put("cSnullNeverHundred", ev and ev[3], sg, f"{TE} [3], the system minus the system without the encoder")
    put("cSnullNeverHundredCI", ev and [ev[4], ev[5]], ci, f"{TE} [4:6]")
    # the system at high effort, and the saving of low effort on the final reranking (grouped calls, standard tier)
    sp = read(ROOT / "research/programs/overnight_20261105/system_parts.json") or {}
    put("dHighNdcg", g("gpt61sol_rk_system_high"), f3, "results.json: dev.gpt61sol_rk_system_high.ndcg_at_10")
    put_contrast("cHigh", sp.get("ndcg_at_10:gpt61sol_rk_system_high - gpt61sol_rk_system"), "overnight_20261105/system_parts.json: high - system")
    sol = (2.0, 0.10, 10.0)
    med, low = (read(RECEIPTS / f"rk_gpt61sol_rk_system{x}.json") or {} for x in ("", "_low"))
    if med.get("usage") and low.get("usage"):
        cm, cl = usage_cost(med["usage"], sol, med["cases"]), usage_cost(low["usage"], sol, low["cases"])
        put("kLowSavePct", 1 - cl["usd_standard_per_1000"] / cm["usd_standard_per_1000"], pct,
            "receipts rk_gpt61sol_rk_system_low vs rk_gpt61sol_rk_system: 1 - low/medium, USD per 1,000 at the standard tier")
    # misses: 1 minus the system's Recall@30 as a mean over groups; the random baseline of the same law
    lh = read(ROOT / "research/programs/overnight_20261105/list100_rerank.json") or {}
    r30 = ((lh.get("means_by_group") or {}).get("system") or {}).get("recall_at_30")
    put("missMeanPct", r30 is not None and 1 - r30 or None, pct, "overnight_20261105/list100_rerank.json: 1 - means_by_group.system.recall_at_30")
    em = read(ROOT / "research/programs/overnight_20261105/e4_misses.json") or {}
    sysm = next((x for x in em.get("lists", []) if x.get("name") == "system"), {})
    put("missRandomPct", sysm.get("random_article_same_law_share"), pct, "overnight_20261105/e4_misses.json: lists.system.random_article_same_law_share")
    # the list of 100 through the final reranking, no ranker, and the system minus it
    TL = "overnight_20261105/list100_rerank.json"
    put("dHundredRerankNdcg", ((lh.get("means_by_group") or {}).get("run") or {}).get("ndcg_at_10"), f3, f"{TL}: means_by_group.run.ndcg_at_10")
    put_contrast("cSysHundred", (lh.get("contrasts_run_minus_system") or {}).get("ndcg_at_10"), f"{TL}: contrasts_run_minus_system.ndcg_at_10, negated", negate=True)
    # the list of 100 asked with each law named in full, cases grouped as stored, plus its final reranking: USD per 1,000 cases
    L = WORK / "dev" / "luna" / "LUNA_LEDGER.jsonl"
    parts = [ledger("prompt_check_20261104_list100", L), ledger("overnight_list100", L)]
    u: dict = {}
    for x in parts:
        for k, v in x["usage"].items():
            u[k] = u.get(k, 0) + v
    if u and lh.get("usage"):
        lc, rc = usage_cost(u, sol, 1217), usage_cost(lh["usage"], sol, lh.get("cases", 1217))
        put("kHundredListStd", lc["usd_standard_per_1000"], lambda v: f"{v:.1f}", "LUNA_LEDGER tags prompt_check_20261104_list100 + overnight_list100, all 1,217 cases, standard tier")
        put("kHundredStd", lc["usd_standard_per_1000"] + rc["usd_standard_per_1000"], lambda v: f"{v:.1f}",
            f"the list of 100 (LUNA_LEDGER, as kHundredListStd) + its final reranking ({TL} usage), USD per 1,000 at the standard tier")


def s6_macros() -> None:
    """The values Chapter 6 asks of the ranker session, on the chosen ranker. Development cases, stored runs
    (research/programs/rank_rerun_20261105/reads/s6_macros.py)."""
    T = "rank_rerun_20261105/reads/out/s6_macros.json"
    s = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/s6_macros.json") or {}
    pct1 = lambda v: pct(v, 1)
    for name, fmt in (("lsaAlsoLex", pct), ("qwenAlsoLex", pct), ("precNevEnc", pct1), ("precNevRank", pct1), ("precCitedRank", pct1),
                      ("keepNevRank", pct), ("adNever", sg), ("adMost", sg), ("sxLateTrained", f3), ("sxLateUntrained", f3),
                      ("sxQbigIntQ", f3), ("dAddQtwodNdcg", f3), ("dAddHydeNdcg", f3)):
        put(name, s.get(name), fmt, f"{T}: {name}")
    put("lsaRMissPct", s.get("lsaRMiss"), pct1, f"{T}: lsaRMiss")
    put("lwaQbigIntQPct", s.get("lwaQbigIntQ"), pct, f"{T}: lwaQbigIntQ")
    rare = s.get("rare") or {}
    for short, key in (("None", "none"), ("One", "one")):
        put(f"lsaRare{short}Pct", (rare.get(key) or {}).get("lsa"), pct, f"{T}: rare.{key}.lsa")
        put(f"lexRare{short}Pct", (rare.get(key) or {}).get("lex"), pct, f"{T}: rare.{key}.lex")
    chosen, grid = s.get("lsaDimChosen") or [], s.get("lsaDimGrid") or []
    put("lsaDimMax", chosen[0] if len(chosen) == 1 and grid and chosen[0] == max(grid) else None, num, f"{T}: lsaDimChosen, the top of lsaDimGrid in every fold")
    put("dCitNoVotesNeverPct", (s.get("noVotes") or {}).get("never"), pct, f"{T}: noVotes.never (rk_noenc_novotes)")
    for name in ("cRankEncNever", "cEncNever", "cEncMost", "cQtwoDAlone", "cHydeAlone", "cIssuesNever", "cIssuesHigh", "cLateEnc", "cLateTrain", "cAddLate"):
        c = s.get(name) or {}
        put(name, c.get("delta"), sg, f"{T}: {name}.delta")
        put(name + "CI", c.get("ci95"), ci, f"{T}: {name}.ci95")
    lh = ((read(ROOT / "research/programs/overnight_20261105/list100_rerank.json") or {}).get("means_by_group") or {}).get("input") or {}
    put("lhRecHundredPct", lh.get("recall_at_100"), pct, "overnight_20261105/list100_rerank.json: means_by_group.input.recall_at_100")
    put("lhNdcgTen", lh.get("ndcg_at_10"), f3, "overnight_20261105/list100_rerank.json: means_by_group.input.ndcg_at_10")
    xg = read(FOOT / "rt_xgb_tesis" / "DEV" / "summary.json") or {}
    put("dXgbNdcg", xg.get("ndcg_at_10"), f3, "rt_xgb_tesis: DEV summary.ndcg_at_10")
    h112 = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/h112.json") or {}
    put_contrast("cXgb", h112.get("rt_xgb_tesis - rt_lm_tesis"), "rank_rerun_20261105/reads/out/h112.json: XGBoost minus the chosen learner, the same inputs, family of one")
    nc = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/named_cited_dev.json") or {}
    put("pctNamedCitedDev", nc.get("share"), pct, "rank_rerun_20261105/reads/out/named_cited_dev.json: share, development rows of release v11")
    h42 = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/h042.json") or {}
    put_contrast("cTrainMasked", h42.get("rt_lm_tesis - rk_train_no_masked"), "rank_rerun_20261105/reads/out/h042.json: the ranker minus the ranker trained without the cases whose facts named a provision")
    put_contrast("cTrainNamed", h42.get("rt_lm_tesis - rk_train_unnamed"), "rank_rerun_20261105/reads/out/h042.json: the ranker minus the ranker trained without the cases whose facts named a cited article")
    for name in ("cLateEnc", "cAddLate"):
        c = s.get(name) or {}
        put(name + "Cls", c and cls_of(c), lambda v: CLASS[v], f"{T}: {name}.class")


def framing_macros() -> None:
    """The numbers FRAMING.md asks of the ranker session: H111, decisions 3(a), 3(b), 4(b), 4(d), 6 and H088. Development cases."""
    # H111: where the system's missed articles sit, pooled over cited pairs (Recall@30 is a mean over groups)
    T = "overnight_20261105/e4_misses.json"
    e4 = read(ROOT / "research/programs/overnight_20261105/e4_misses.json") or {}
    row = next((x for x in e4.get("lists", []) if x["name"] == "system"), None) or {}
    n = row.get("missed_first_30")
    put("missBase", row.get("cited"), num, f"{T}: system.cited")
    put("missTotal", n, num, f"{T}: system.missed_first_30")
    put("missPct", n and n / row["cited"], pct, f"{T}: missed_first_30 / cited, pooled")
    for mac, key in (("missDiv", "same_division"), ("missTop", "same_top_division"), ("missLaw", "same_law"), ("missOther", "other_law")):
        put(mac, (row.get("counts") or {}).get(key), num, f"{T}: system.counts.{key}")
        put(mac + "Pct", (row.get("missed_shares") or {}).get(key), pct, f"{T}: system.missed_shares.{key}")
    put("missInLawPct", row and 1 - row["missed_shares"]["other_law"], pct, f"{T}: 1 - system.missed_shares.other_law")

    # decision 3(a): what the model's lead rests on
    T = "rank_rerun_20261105/reads/out/h099.json"
    h = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/h099.json") or {}
    triple = lambda name, t, src: (put(name, t and t[0], sg, src), put(name + "CI", t and t[1:3], lambda v: f"[{sg(v[0])}, {sg(v[1])}]", src))
    law = (h.get("law_given") or {}).get("rt_lm_tesis") or {}
    put("leadLawModel", law.get("model") and law["model"][0], f3, f"{T}: law_given.rt_lm_tesis.model")
    put("leadLawRank", law.get("ranker") and law["ranker"][0], f3, f"{T}: law_given.rt_lm_tesis.ranker")
    triple("cLeadLaw", law.get("lead"), f"{T}: law_given.rt_lm_tesis.lead")
    band = (h.get("band_r20") or {}).get("rt_lm_tesis") or {}
    triple("cLeadNever", (band.get("band_0") or {}).get("diff"), f"{T}: band_r20.rt_lm_tesis.band_0.diff (Recall@20, model minus ranker)")
    triple("cLeadOften", (band.get("band_>10") or {}).get("diff"), f"{T}: band_r20.rt_lm_tesis.band_>10.diff (Recall@20, model minus ranker)")
    put("leadSwapNdcg", (h.get("swap") or {}).get("swap") and h["swap"]["swap"][0], f3, f"{T}: swap.swap")
    lim = (read(ROOT / "research/programs/why_models_20261103/limits.json") or {}).get("model_alone") or {}
    L = "why_models_20261103/limits.json: model_alone"
    put("leadNamedOften", lim.get("named_often_cited_laws"), pct, f"{L}.named_often_cited_laws")
    put("leadNamedOther", lim.get("named_other_laws"), pct, f"{L}.named_other_laws")

    # decision 4(b), 4(d), H088
    T = "rank_rerun_20261105/reads/out/framing_numbers.json"
    fn = read(ROOT / "research/programs/rank_rerun_20261105/reads/out/framing_numbers.json") or {}
    NAMES = {10: "Ten", 20: "Twenty", 50: "Fifty", 100: "Hundred"}
    for who, mac in (("noenc", "Noenc"), ("rank", "Rank"), ("alone", "Alone"), ("system", "Sys")):
        for k, word in NAMES.items():
            for m, m_ in (("ndcg", "Ndcg"), ("recall", "Rec")):
                put(f"dp{mac}{m_}{word}", ((fn.get("depth") or {}).get(who) or {}).get(f"{m}_{k}"), f3, f"{T}: depth.{who}.{m}_{k}")
    fill = (fn.get("fill") or {}).get("recall_100") or {}
    for who, mac in (("none", "None"), ("lexical", "Lex"), ("encoder", "Enc"), ("noenc", "Noenc"), ("rank", "Rank")):
        put(f"fill{mac}Pct", fill.get(who), pct, f"{T}: fill.recall_100.{who}")
    triple("cFillRankLex", (fn.get("fill") or {}).get("rank_minus_lexical"), f"{T}: fill.rank_minus_lexical")
    bl = fn.get("blind") or {}
    put("blindCases", bl.get("cases"), num, f"{T}: blind.cases")
    for k_, mac in (("rank_100", "blindRankHundred"), ("rank_10", "blindRankTen"), ("system_100", "blindSysHundred"), ("system_10", "blindSysTen")):
        put(mac, bl.get(k_), num, f"{T}: blind.{k_}")
    for who, mac in (("list", "List"), ("no_text", "NoText"), ("text", "Text")):
        for key, m_ in (("ndcg", "Ndcg"), ("law1", "Law"), ("in_law", "InLaw")):
            put(f"blank{mac}{m_}", ((fn.get("blank") or {}).get(who) or {}).get(key), f3, f"{T}: blank.{who}.{key}")

    # decision 6: the hosted route, tokens counted and prices read on 5 Nov, plus the ranker without the encoder
    T = "rank_rerun_20261105/reads/tokens_for_hosted.json"
    tk = read(ROOT / "research/programs/rank_rerun_20261105/reads/tokens_for_hosted.json") or {}
    usd = lambda v: f"{v:.2f}"
    noenc_usd = ((read(OUT / "results.json") or {}).get("system", {}).get("cost", {}).get("ranker_noenc") or {}).get("usd_per_1000_median")
    for mac, key in (("lnHostedUsdLo", "usd_per_1000_cases_at_0.01_per_M"), ("lnHostedUsdHi", "usd_per_1000_cases_at_0.1_per_M")):
        put(mac, tk.get(key) is not None and noenc_usd is not None and tk[key] + noenc_usd or None, usd, f"{T}: {key} + system.cost.ranker_noenc.usd_per_1000_median")
    put("lnCatalogueHostedLo", tk.get("usd_catalogue_once_at_0.01_per_M"), usd, f"{T}: usd_catalogue_once_at_0.01_per_M")
    put("lnCatalogueHostedHi", tk.get("usd_catalogue_once_at_0.1_per_M"), usd, f"{T}: usd_catalogue_once_at_0.1_per_M")

    # decision 6: the processor route, encoding timed on a processor plus the ranker's other steps; the whole run replaces this sum
    T = "timing/rk/encode_cpu.json"
    enc = read(RUN / "timing" / "rk" / "encode_cpu.json") or {}
    cost = (read(OUT / "results.json") or {}).get("system", {}).get("cost", {}).get("ranker_noenc") or {}
    tex = (ROOT / "thesis" / "evidence" / "numbers_cost_20261025.tex").read_text(encoding="utf-8")
    p_cpu = float(re.search(r"\\newcommand\{\\priceCPU\}\{([0-9.]+)\}", tex).group(1))
    one = lambda v: f"{v:.1f}"
    both = lambda k_a, k_b: enc.get(k_a) is not None and cost.get(k_b) is not None and enc[k_a] + cost[k_b] or None
    lcpu = (read(OUT / "results.json") or {}).get("system", {}).get("cost", {}).get("ranker_cpu") or {}
    put("kCpuMed", lcpu.get("seconds_median"), one, "system.cost.ranker_cpu.seconds_median (whole_run ranker_cpu_a, live, end to end)")
    put("kCpuHigh", lcpu.get("seconds_p95"), one, "system.cost.ranker_cpu.seconds_p95 (whole_run ranker_cpu_a, live, end to end)")
    put("lnCpuUsd", enc.get("seconds_median") and noenc_usd is not None and 1000 * enc["seconds_median"] * enc["threads"] * p_cpu / 3600 + noenc_usd or None, usd,
        f"{T}: 1000 x seconds_median x threads x priceCPU / 3600 + system.cost.ranker_noenc.usd_per_1000_median")


def write_tex() -> None:
    lines = ["% Generated by scripts/final_results.py from research/programs/final_results_20261102/results.json; do not edit by hand.",
             "% One macro per number. Read last, so it takes precedence over the older numbers_*.tex. A value not yet held prints \\EvPending{waiting: ...}.",
             "% RESULTS.md beside results.json maps every macro to its file and key."]
    for k, v in MAC.items():
        lines.append(f"\\def\\{k}{{{v}}}")
    TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md(rec: dict) -> None:
    fmt = lambda v: "" if v is None else f"{v:.3f}"
    s = rec["system"]
    lines = ["# One record of results", "", "Generated by `scripts/final_results.py`. `results.json` beside this file holds everything; "
             "`thesis/evidence/numbers_final.tex` holds one macro per number.", "",
             f"System on the development cases: `{s['dev_system']}`. Frozen: {'yes' if s['frozen'] else 'not yet'}. Test set scored: {'yes' if s['test'] else 'not yet'}.", "",
             "Flags not set: " + (", ".join(f for f, v in rec["flags"].items() if not v) or "none"), "",
             "## The checks on the final system (addendum 22; each minus the system)", "", "| Check | mean | difference | 95% interval | adjusted interval | class |", "|---|---|---|---|---|---|"]
    for v, c in s["settings"]["checks"].items():
        lines.append(f"| {v} | {fmt(c.get('mean'))} | {c['delta']:+.3f} | [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}] | "
                     f"[{c['ci_adjusted'][0]:+.3f}, {c['ci_adjusted'][1]:+.3f}] | {c['class']} |")
    lines += ["", f"Frozen settings (D198): {s['settings']['frozen']}. Missing checks: {s['settings']['missing']}.", "",
              "## All development cases (1,217), summary measure", "", "| System | store | nDCG@10 | first | ten | law first | law ten | within law | never-cited (pairs) | recall@20 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in sorted(rec["dev"].items(), key=lambda x: -(x[1]["ndcg_at_10"] or 0)):
        lines.append(f"| {k} | {v['store']} | " + " | ".join(fmt(v[m]) for m in MEASURES[:-1]) + " |")
    lines += ["", "## The 300-case sample, mean per case", "", "| Run | nDCG@10 | ten |", "|---|---|---|"]
    for k, v in sorted(rec["sample"].items(), key=lambda x: -x[1]["mean_per_case"]):
        lines.append(f"| {k} | {v['mean_per_case']:.3f} | {fmt(v.get('success_at_10'))} |")
    lines += ["", "## Macro map", "", "| Macro | Value | Source |", "|---|---|---|"]
    for k, v in MAC.items():
        lines.append(f"| `\\{k}` | {v.replace('|', '/')} | {SRC[k]} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
