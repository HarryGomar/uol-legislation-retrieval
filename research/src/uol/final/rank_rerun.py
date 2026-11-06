"""Chapter 6's comparisons on the chosen ranker, and the experiments the author's comments call for (D209).

research/programs/rank_rerun_20261105/PLAN.md, written before any score here was read. Development cases only. New run
ids throughout: nothing stored is overwritten. The steps reuse ``lean_rerun`` with this ranker as its base.

    py -3.12 -m uol.final.rank_rerun register --go
    py -3.12 -m uol.final.rank_rerun ranker --id rk_noenc --go
    py -3.12 -m uol.final.rank_rerun model --kind gpt61sol --go
    py -3.12 -m uol.final.rank_rerun system --check merged --go       (no --check: the system)
    py -3.12 -m uol.final.rank_rerun alone --effort low --go
    py -3.12 -m uol.final.rank_rerun sample --kind haiku --go
    py -3.12 -m uol.final.rank_rerun rr --kind bge_ranker --go
    py -3.12 -m uol.final.rank_rerun pointwise --go
    py -3.12 -m uol.final.rank_rerun agent --tool words --go
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from uol.final import lean_rerun as LR

FT, A, D = LR.FT, LR.A, LR.D
BASE = "rt_lm_tesis"
LR.BASE = BASE
FT.RANKER = BASE
LR.RR_SUFFIX = "rk"
LR.RR_JOB = "_rk1"
LR.SAMPLE_DIR = "rk"
STORE = LR.STORE
TOPICS = LR.TOPICS
SYSTEM = "gpt61sol_rk_system"
CHECKS = {"merged": "the merged list, no second reading", "noenc": "merged with the ranker without the encoder",
          "low": "the second reading at low effort", "high": "the second reading at high effort",
          "d60": "the first 60 of the merged list read", "both": "both orders read"}
ARMS = LR.ARMS
ADD_ARMS = {k.replace("lf_", "rk_"): v for k, v in ARMS.items()}
ADDITIONS = [*ADD_ARMS, "rk_q2d", "rk_hyde", "rk_topics"]
NO_ENCODER = [*FT.LEAN_DROP, *FT.ENCODER]
A.receipt = lambda name, obj: LR._receipt(f"rk_{name}", obj)


def specs() -> list[dict]:
    """The chosen ranker with one change each, built like ``ranker_tests.specs``."""
    base = A._systems()["S0+qwen3_8b"]
    out = []

    def add(sid: str, note: str, drop=FT.LEAN_DROP, tesis: bool = True, **args):
        s = copy.deepcopy(base)
        s["id"], s["notes"] = sid, f"rank rerun (D209): {note}"
        s["adapter_args"].update({"drop_sources": list(drop), "extra": [], "method": "rank_test", "learner": "lgbm", "settings": "stored"})
        if tesis:
            s["adapter_args"]["extension"] = "m1"
        s["adapter_args"].update(args)
        s.pop("answer", None)
        out.append(s)

    add("rk_noenc", "without the encoder; the older tesis votes by words", drop=NO_ENCODER, extension_key="lex")
    add("rk_noenc_novotes", "without the encoder and without the older tesis votes", drop=NO_ENCODER, tesis=False)
    add("rk_nocit", "tuned lexical matching and the encoder's direct score only, nothing built on citations", tesis=False,
        drop=[*FT.LEAN_DROP, *[c for c in FT.CITATION_ALL if c not in FT.LEAN_DROP]], drop_columns=["instr_prior"])
    add("rk_neither", "neither the encoder nor anything built on citations", tesis=False, add_sources=[],
        drop=[c for c in FT.CITATION_ALL if "qwen8b" not in c], drop_columns=["instr_prior"])
    for sid, arm in ADD_ARMS.items():
        add(sid, f"plus the columns of arm {arm}", extension="m1+columns", arm=arm)
    add("rk_q2d", "plus query2doc", extra=["query2doc"])
    add("rk_hyde", "plus HyDE", extra=["hyde"])
    add("rk_topics", "plus the four topic scores", extra=list(TOPICS))
    for k in LR.SEEDS:
        add(f"rk_seed{k}", f"tree seed offset {k}", seed_offset=k)
    # reproduction (4.1, B.x): the two rankers of the thesis fitted again from the code, same spec, new id
    add("rk_repro_ranker", "reproduction of rt_lm_tesis (ranker_tests.specs: the same base, drops, learner, settings and m1 votes)")
    add("rk_repro_noenc", "reproduction of rk_noenc", drop=NO_ENCODER, extension_key="lex")
    for key in [e for e in LR.ENCODERS if e != "euler_legal"]:     # the author, 4 Nov: remove it
        add(f"rk_noenc+{key}", f"without its encoder, plus the score of {key}", drop=NO_ENCODER, extra=[key],
            **({} if key == "qwen3_8b" else {"extension_key": "lex"}))
    add("rk_train_no_masked", "trained without any case whose facts named a provision (H042)", train_filter="no_masked")
    add("rk_train_unnamed", "trained without the cases whose facts named a cited article (H042, H043)", train_filter="no_answer_named")
    return out


def llm_ids() -> list[str]:
    return ([f"{k}_rerank_{BASE}" for k in ("gpt61sol", *LR.CODEX, *LR.BILLED)] + [s for k in LR.RERANKERS for s in (f"{k}_rk", f"{k}_rk_alone")]
            + [SYSTEM, *[f"{SYSTEM}_{c}" for c in CHECKS], "gpt61sol_alone_low", "gpt61sol_alone_high"])


def register() -> int:
    LR._install([*specs(), *[A.spec_llm(BASE, s, "claude") for s in llm_ids()]], dry=False)
    print(json.dumps({"registered": [s["id"] for s in specs()] + llm_ids()}, indent=1))
    return 0


def _agents():
    from uol.final import llm_agents as G
    G.SYSTEM = f"{G.KIND}_rerank_{BASE}"
    G.OUT = A.WORK / "agents_rk"
    G.sample_rows = lambda smoke: LR.sample_rows()[:smoke] if smoke else LR.sample_rows()
    return G


def system(check: str | None) -> int:
    """The system on the chosen ranker, or one change to it: the model's held answers and weight, as ``lean_rerun.system``."""
    from uol.embedding_push import common as ep
    from uol.final import external as X
    G = _agents()
    sid = SYSTEM if check is None else f"{SYSTEM}_{check}"
    D.activate(STORE)
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = G.all_rows()
    answers = json.loads((A.WORK / "agents" / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    mdl = {r: G.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / f"{G.KIND}_rerank_S0+qwen3_8b" / "meta.json").read_text(encoding="utf-8"))["w_all"])
    base = "rk_noenc" if check == "noenc" else BASE
    rk = G.ranker_lists(rows, base)
    merged = {r: G.rrf([mdl[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    depth = 60 if check == "d60" else X.TOP
    shown = {r: merged[r][:depth] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[G.KIND], **({"effort": check} if check in ("low", "high") else {})}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    out = G.OUT / sid
    out.mkdir(parents=True, exist_ok=True)
    tot, usd, passes = {}, 0.0, ["fwd", "rev"] if check == "both" else ["fwd"]
    if check == "merged":
        ranked = merged
    else:
        ask = G.ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, f"agents_{sid}", usage), stats, cfg))
        o = FT.rerank_depth(rows, shown, ask, depth, passes=tuple(passes))
        tot, usd = X.codex_usage(usage, cfg)
        (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
        ranked = {r: G.fuse_order(shown[r], o["orders"], passes, r, w) + merged[r][depth:] for r in rows}
    res = G.save_system(sid, rows, ranked, {"model": G.MODEL, "base": base, "w_all": w, "passes": passes, "depth": depth, "effort": cfg["effort"],
                                            "usage": tot, "usd_list_equivalent": usd, **stats, "recipe": "the system on the chosen ranker" if check is None else CHECKS[check]})
    A.receipt(sid, {"cases": len(rows), "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10"), **stats})
    print(json.dumps({"system": sid, "ndcg_at_10": res.get("ndcg_at_10"), "usd_list_equivalent": usd}), flush=True)
    return 0


def alone(effort: str) -> int:
    """H083: GPT-6.1 Sol alone at another effort on every development case, same prompt, answers apart from the medium run."""
    from uol.embedding_push import common as ep
    from uol.eval_lenses import llm_alone as LA
    G = _agents()
    G.pin_model()
    LA.EFFORT = effort
    LA.TAG = f"llm_alone_dev_{effort}"
    D.activate(STORE)
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = G.all_rows()
    t0 = time.perf_counter()
    got = LA.run(rows)
    sid = f"gpt61sol_alone_{effort}"
    (G.OUT / sid).mkdir(parents=True, exist_ok=True)
    (G.OUT / sid / "answers.json").write_text(json.dumps({str(r): got[r] for r in rows}, ensure_ascii=False), encoding="utf-8")
    res = G.save_system(sid, rows, {r: G.resolved(got[r]["articulos"], generic) for r in rows}, {"model": G.MODEL, "effort": effort, "recipe": "the facts only, another effort (H083)"})
    A.receipt(sid, {"cases": len(rows), "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10")})
    print(json.dumps({"system": sid, "ndcg_at_10": res.get("ndcg_at_10")}), flush=True)
    return 0


def ranker(sid: str) -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0: the doc2vec columns depend on it")
    from uol.final import harness
    A.install_systems(specs(), dry=False)
    D.activate(STORE)
    LR._overlay_stat_glob()
    t0 = time.perf_counter()
    s = harness.run(sid, "DEV", store=STORE)
    print(json.dumps({"system": sid, "status": s.get("status"), "ndcg_at_10": s.get("ndcg_at_10"), "n": s.get("n_cases"),
                      "seconds": round(time.perf_counter() - t0), "reason": s.get("reason")}), flush=True)
    return 0 if s.get("status") == "done" else 1


def pointwise() -> int:
    from uol.final import pointwise as PW
    PW.LISTWISE = f"gpt61sol_rerank_{BASE}"
    PW.OUT = PW.OUT.parent / "pointwise_rk"
    PW.WORK = A.WORK / "pointwise_rk"
    return PW.main(["run", "--go"])


def agent(tool: str) -> int:
    G = _agents()
    D.activate(STORE)
    return G.agent(tool, True, None)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.rank_rerun", description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["register", "ranker", "model", "system", "alone", "sample", "rr", "pointwise", "agent", "show"])
    ap.add_argument("--id")
    ap.add_argument("--kind")
    ap.add_argument("--check", choices=list(CHECKS))
    ap.add_argument("--effort", choices=["low", "high"])
    ap.add_argument("--tool", choices=["words", "embedding"])
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "show" or not a.go:
        for s in specs():
            print(s["id"], json.dumps({k: v for k, v in s["adapter_args"].items() if k not in ("drop_sources", "table", "variant")}), len(s["adapter_args"]["drop_sources"]))
        print(llm_ids())
        return 0
    os.environ.setdefault("UOL_LUNA_CACHE", str(A.ROOT / ".cache" / "footing" / "codex_noweb.sqlite"))
    if a.step == "register":
        return register()
    if a.step == "ranker":
        return ranker(a.id)
    if a.step == "model":
        return LR.model(a.kind)
    if a.step == "system":
        return system(a.check)
    if a.step == "alone":
        return alone(a.effort)
    if a.step == "sample":
        if a.kind not in LR.BILLED:
            raise SystemExit(f"{a.kind}: not a Claude kind")
        return LR.sample(a.kind)
    if a.step == "rr":
        return LR.rr(a.kind)
    if a.step == "pointwise":
        return pointwise()
    return agent(a.tool)


if __name__ == "__main__":
    sys.exit(main())
