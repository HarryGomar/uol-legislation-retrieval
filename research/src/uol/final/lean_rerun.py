"""Chapter 6's comparisons rerun on the system's own ranker (v33, 4 Nov 2026, D203).

Until v32 the additions to the ranker, the encoder ladder, the rerankers, the language models reranking a ranker's list,
the pointwise judge and the agents that start from a list were measured on the 28-source ranker (``S0+qwen3_8b``), the
system's ranker until the step to eight sources (PLAN addendum 22). the author, 4 Nov: "Rerun everything". Each comparison is
made here on the eight-source ranker ``ft_lean`` under its own id, so no stored run or receipt is overwritten:

    lf_<addition>            the eight-source ranker plus one addition (the additions of ``footing.specs``)
    lf_noenc+<encoder>       the ranker without its encoder plus one encoder's score (the ladder of Figure 6.5)
    <kind>_rerank_ft_lean    a language model reranking its first 30 (``footing.second_run``; Claude on the sample)
    <reranker>_lean          a cross-encoder over its first 30, on the remote worker
    gpt61sol_final_on_<base> the final system with another ranker's list in place of the ranker's

The pointwise judge and the agents write under ``.cache/footing/*_lean``; receipts are named ``lean_*``. Development
cases only (``guard.dev_rows_only``). Every compute step is a dry run unless ``--go``.

    py -3.12 -m uol.final.lean_rerun register --go
    py -3.12 -m uol.final.lean_rerun ranker --id lf_tesis --go
    py -3.12 -m uol.final.lean_rerun model --kind gpt61terra --go
    py -3.12 -m uol.final.lean_rerun sample --kind sonnet --go
    py -3.12 -m uol.final.lean_rerun rr --kind bge_ranker --go
    py -3.12 -m uol.final.lean_rerun pointwise --go
    py -3.12 -m uol.final.lean_rerun agent --tool words --go
    py -3.12 -m uol.final.lean_rerun system --base lf_tesis --go
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from uol.final import footing as FT  # first: it points the audit modules at the common-footing run

A, D = FT.A, FT.D
BASE = "ft_lean"
FT.RANKER = BASE                     # every step below reads the eight-source ranker's lists
FT.AUDIT = FT.STORE                  # where the ranker's own per-case scores are
STORE = FT.STORE
TOPICS = ["lsa", "lda_sim", "lda_dm", "doc2vec"]
ARMS = {"lf_structure": "S2", "lf_cocite": "S3", "lf_adapter": "E2", "lf_smooth": "E3", "lf_issues": "L2", "lf_patterns": "L4"}
ENCODERS = ("bge_m3", "qwen3_0p6b", "qwen3_4b", "qwen3_8b", "euler_legal")
SEEDS = (1, 2)                       # ft_lean itself is seed offset 0
CODEX = ("gpt61terra", "gpt6solnw", "gpt56sol", "gpt56terra", "gpt6luna", "luna", "gpt61sol_low", "gpt61sol_high")
BILLED = ("claude", "sonnet", "haiku")                  # the 300-case sample only (the author, 4 Nov: about USD 23)
RERANKERS = {"ce_ranker": None, "bge_ranker": "bge_rr_v2_m3", "qwen4b_ranker": "qwen3_rr_4b"}
RR_JOB = "_lean1"
RR_SUFFIX = "lean"                   # run ids and job names of the cross-encoders
SAMPLE_DIR = "lean"                  # where the account route keeps the sample's answers
ON_BASES = ("lf_tesis", "S_full_ext")
OUT = A.ROOT / "research" / "programs" / "lean_rerun_20261104"

_install, _receipt = A.install_systems, A.receipt


def _install_once(extra, dry):
    """Concurrent jobs never rewrite the shared registry: ``register`` writes it once, the jobs only point at it."""
    extra = list(extra)
    path = A.RUN / "systems_audit.json"
    if not dry and path.exists():
        have = {s["id"] for s in json.loads(path.read_text(encoding="utf-8"))["systems"]}
        if {e["id"] for e in extra} <= have:
            os.environ["UOL_FINAL_SYSTEMS"] = str(path)
            return path
    return _install(extra, dry)


A.install_systems = _install_once
A.receipt = lambda name, obj: _receipt(name if name.startswith("lean_") else f"lean_{name}", obj)


def specs() -> list[dict]:
    """The eight-source ranker with one change each, built from the registry entry of ``ft_lean``'s parent."""
    base = A._systems()["S0+qwen3_8b"]
    out = []

    def add(sid: str, note: str, drop=FT.LEAN_DROP, **args):
        s = copy.deepcopy(base)
        s["id"], s["notes"] = sid, f"v33 lean rerun (D203): {note}"
        s["adapter_args"].update({"drop_sources": list(drop), **args})
        s.pop("answer", None)
        out.append(s)

    add("lf_tesis", "the eight-source ranker plus the past citations of the unstructured tesis", extension="m1")
    for sid, arm in ARMS.items():
        add(sid, f"the eight-source ranker plus the columns of arm {arm}", extension="columns", arm=arm)
    add("lf_plain", "the eight-source ranker without the topic columns, pointwise", extra=[])
    add("lf_listwise", "the eight-source ranker without the topic columns, LambdaMART", extra=[], method="lambdamart")
    add("lf_q2d", "the eight-source ranker plus query2doc", extra=[*TOPICS, "query2doc"])
    add("lf_hyde", "the eight-source ranker plus HyDE", extra=[*TOPICS, "hyde"])
    for k in SEEDS:                     # the author, 4 Nov: the variation from refitting the trees, which the bootstrap omits
        add(f"lf_seed{k}", f"the eight-source ranker refitted with tree seed offset {k}", seed_offset=k)
    for key in ENCODERS:
        add(f"lf_noenc+{key}", f"the ranker without its encoder plus the score of {key}", drop=[*FT.LEAN_DROP, *FT.ENCODER],
            extra=[*TOPICS, key])
    return out


def llm_ids() -> list[str]:
    return ([f"{k}_rerank_{BASE}" for k in (*CODEX, *BILLED)] + [s for k in RERANKERS for s in (f"{k}_lean", f"{k}_lean_alone")]
            + [f"gpt61sol_final_on_{b}" for b in ON_BASES])


def register() -> int:
    _install([*specs(), *[A.spec_llm(BASE, s, "claude") for s in llm_ids()]], dry=False)
    print(json.dumps({"registered": [s["id"] for s in specs()] + llm_ids()}, indent=1))
    return 0


def _overlay_stat_glob() -> int:
    """The bundle overlay serves ``numpy.load`` and ``Path.exists`` of its verified arrays; the encoder sources also
    ``stat`` their arrays (``loaded.verify_artefact``) and ``glob`` their query vectors (``dense.py``). Serve both from
    the same verified local files, in this process only."""
    import fnmatch
    import pathlib
    from uol.final import bundle_overlay as BO
    ov = json.loads(BO.OVERLAY.read_text(encoding="utf-8"))
    root = Path(A.BUNDLE_DATA).parent
    served = {os.path.normcase(str(root / d)): (root / d, A.ROOT / a["local"]) for d, a in ov["arrays"].items() if a["match"]}
    stat0, glob0 = pathlib.Path.stat, pathlib.Path.glob

    def stat(self, *args, **kwargs):
        hit = served.get(os.path.normcase(str(self)))
        return stat0(hit[1] if hit else self, *args, **kwargs)

    def glob(self, pattern, *args, **kwargs):
        found = list(glob0(self, pattern, *args, **kwargs))
        have, here = {os.path.normcase(str(p)) for p in found}, os.path.normcase(str(self))
        for key, (bundle_path, _) in served.items():
            if os.path.dirname(key) == here and key not in have and fnmatch.fnmatch(bundle_path.name, pattern):
                found.append(bundle_path)
        return iter(sorted(found))

    pathlib.Path.stat, pathlib.Path.glob = stat, glob
    return len(served)


def ranker(sid: str) -> int:
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0: the doc2vec columns depend on it")
    from uol.final import harness
    A.install_systems(specs(), dry=False)
    D.activate(STORE)
    _overlay_stat_glob()
    t0 = time.perf_counter()
    s = harness.run(sid, "DEV", store=STORE)
    A.log(f"{sid}: nDCG@10 {s.get('ndcg_at_10')} n {s.get('n_cases')} ({round(time.perf_counter() - t0)} s)", "rankers_lean")
    print(json.dumps({"system": sid, "status": s.get("status"), "ndcg_at_10": s.get("ndcg_at_10"), "n": s.get("n_cases"),
                      "reason": s.get("reason")}), flush=True)
    return 0 if s.get("status") == "done" else 1


def model(kind: str) -> int:
    return FT.second_run(kind, BASE, True)


def sample_rows() -> list[int]:
    import pandas as pd
    return sorted(int(r) for r in pd.read_parquet(A.WORK / "agents" / "alone" / "per_case.parquet", columns=["row"])["row"])


ACCOUNT_WORKERS = 3


def account_api(cfg: dict) -> "callable":
    """The Claude models through the author's Claude account, not the API (the author, 4 Nov, after the API credit ran out: "use
    the actual account's usage"): ``why_models.claude_call``, the CLI with no tools, no settings, a neutral system
    prompt and the answer schema enforced, cached per call. Same prompts, windows, retries and repair as the Batch API
    step (``external.ask_claude``); a call that still fails comes back without text, as on the Codex route."""
    from concurrent.futures import ThreadPoolExecutor
    from uol.final import why_models as W
    W._cache_path = lambda: (A.WORK / SAMPLE_DIR / "sample").mkdir(parents=True, exist_ok=True) or A.WORK / SAMPLE_DIR / "sample" / "claude_account_cache.jsonl"

    def call(requests: list[dict]) -> dict[str, dict]:
        def one(r: dict) -> tuple[str, dict]:
            try:
                res = W.claude_call(r["prompt"], r["schema"], r["model"], r["effort"], sample=int(r.get("sample") or 0))
            except RuntimeError:
                return r["custom_id"], {"text": None, "stop_reason": "error"}
            return r["custom_id"], {"text": json.dumps(res["output"], ensure_ascii=False), "stop_reason": "end_turn"}

        with ThreadPoolExecutor(max_workers=ACCOUNT_WORKERS) as pool:
            return dict(pool.map(one, requests))
    return call


def sample(kind: str) -> int:
    """A Claude model reranking the first 30 on the 300-case sample, through the Claude account, fused per fold, scored
    on DEV-A300."""
    from uol.final import external as X, harness, why_models as W
    for k, cfg in {**FT.effort_kinds(), **FT.MODELS}.items():
        X.SWEEP.setdefault(k, cfg)
    X.venv_api = lambda workdir, venv_python, env_file, cfg: account_api(cfg)     # the account route, never the API key
    system = f"{kind}_rerank_{BASE}"
    A.install_systems([A.spec_llm(BASE, system, "claude")], dry=False)
    D.activate(STORE)
    FT.ensure_base(BASE)
    X.BASE = BASE
    X.SYSTEMS[kind] = system
    X.FILES.setdefault(kind, f"{kind}_orders_s0.json")
    rows = sample_rows()
    if len(rows) != 300:
        raise RuntimeError(f"{len(rows)} sample rows, expected 300")
    t0 = time.perf_counter()
    path = X.drive_claude(FT.base_top200(BASE), A.WORK / SAMPLE_DIR / "sample" / kind, A.CLAUDE_VENV, A.ENV_FILE, rows=rows, kind=kind)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg = X.CLAUDE if kind == "claude" else X.SWEEP[kind]
    o.update({"route": "Claude account (claude CLI), no API key", "usd": 0.0,
              "usage": W.claude_usage(cfg["model"], cfg["effort"])})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    (X.ext_dir() / X.FILES[kind]).write_bytes(Path(path).read_bytes())
    X.fuse(kind, STORE)
    s = harness.run(system, "DEV-A300", store=STORE, force=True)
    res = {"system": system, "ndcg_at_10": s.get("ndcg_at_10"), "usd": o.get("usd"), "usage": o.get("usage"),
           "repairs": o.get("repairs"), "unanswered": o.get("unanswered"), "seconds": round(time.perf_counter() - t0, 1)}
    A.receipt(f"sample_{kind}", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return 0


def rr(which: str) -> int:
    """``footing.rr_run`` over the eight-source ranker: new job names, so no returned job of the 28-source run is read."""
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    import numpy as np
    from uol.final import harness, rerank_jobs as RJ
    from uol.round3 import gpu_remote as G
    jobs = Path(G.__file__).parent / "gpu_jobs"
    helper = jobs / "rerank_bench.py"
    sid = f"{which}_{RR_SUFFIX}"
    A.install_systems([A.spec_llm(BASE, s, "claude") for s in (sid, f"{sid}_alone")], dry=False)
    D.activate(STORE)
    FT.ensure_base(BASE)
    RJ.BASE = BASE
    t0 = time.perf_counter()
    if which == "ce_ranker":
        RJ.tag = lambda k, seed=1: f"ce_ranker_{RR_SUFFIX}_f{k}_s{seed}"
        pairs, scores, seconds = [], [], {}
        for k in range(5):
            tag = RJ.tag(k)
            out = G.LOCAL_JOBS / f"{tag}{RR_JOB}" / "remote"
            d = RJ.inputs_dir(k)
            if not (out / "outputs" / "scores.npy").exists():
                d = RJ.prep(k)
                G.submit(f"{tag}{RR_JOB}", jobs / "r2_train.py", {n: d / n for n in ("train.jsonl", "queries.jsonl", "docs.jsonl", "pairs.npy", "task.json")},
                         helpers=[helper], expected_outputs=["scores.npy"])
            seconds[tag] = json.loads((out / "receipt.json").read_text(encoding="utf-8")).get("seconds")
            pairs.append(np.load(d / "pairs.npy"))
            scores.append(np.load(out / "outputs" / "scores.npy"))
        RJ._fuse(sid, np.concatenate(pairs), np.concatenate(scores), {"seed": 1, "recipe": "round3.r2_cross FULL_TASK per fold, eight-source ranker"})
    else:
        RJ.ZERO_SHOT[sid] = RERANKERS[which]
        d = RJ.zs_inputs_dir(sid)
        out = G.LOCAL_JOBS / f"{sid}{RR_JOB}" / "remote"
        if not (out / "outputs" / "scores.npy").exists():
            d = RJ.zs_prep(sid)
            G.submit(f"{sid}{RR_JOB}", jobs / "r1_score.py", {n: d / n for n in ("queries.jsonl", "docs.jsonl", "pairs.npy", "task.json")},
                     helpers=[helper], expected_outputs=["scores.npy"])
        seconds = {sid: json.loads((out / "receipt.json").read_text(encoding="utf-8")).get("seconds")}
        pairs = np.load(d / "pairs.npy")
        keep = pairs[:, 0] >= 0                            # the blank-query control pairs are not part of the ranking
        RJ._fuse(sid, pairs[keep], np.load(out / "outputs" / "scores.npy")[keep], {"model": RERANKERS[which], "recipe": "round3 R1 job, zero-shot, eight-source ranker"})
    res = {s: harness.run(s, "DEV", store=STORE, force=True).get("ndcg_at_10") for s in (sid, f"{sid}_alone")}
    A.receipt(f"rr_{which}", {"gpu_seconds": seconds, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res})
    print(json.dumps({"gpu_seconds": seconds, "ndcg_at_10": res}, indent=1), flush=True)
    return 0


def pointwise() -> int:
    from uol.final import pointwise as PW
    PW.LISTWISE = f"gpt61sol_rerank_{BASE}"
    PW.OUT = PW.OUT.parent / "pointwise_lean"
    PW.WORK = A.WORK / "pointwise_lean"
    return PW.main(["run", "--go"])


def _agents():
    from uol.final import llm_agents as G
    G.SYSTEM = f"{G.KIND}_rerank_{BASE}"
    G.OUT = A.WORK / "agents_lean"
    G.sample_rows = lambda smoke: sample_rows()[:smoke] if smoke else sample_rows()
    return G


def agent(tool: str) -> int:
    """The agent that starts from the ranker's first 30 and searches with ``tool``, on the sample."""
    G = _agents()
    D.activate(STORE)
    return G.agent(tool, True, None)


def system(base: str) -> int:
    """The final system with ``base``'s list in place of the ranker's: same model answers, weight, depth and forward order."""
    import numpy as np  # noqa: F401
    from uol.embedding_push import common as ep
    from uol.final import external as X
    G = _agents()
    sid = f"{G.KIND}_final_on_{base}"
    A.install_systems([A.spec_llm(base, sid, "claude")], dry=False)
    D.activate(STORE)
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = G.all_rows()
    answers = json.loads((A.WORK / "agents" / "alone_full" / "answers.json").read_text(encoding="utf-8"))
    mdl = {r: G.resolved(answers[str(r)]["articulos"], generic) for r in rows}
    w = float(json.loads((A.WORK / "dev" / "llm" / f"{G.KIND}_rerank_S0+qwen3_8b" / "meta.json").read_text(encoding="utf-8"))["w_all"])
    rk = G.ranker_lists(rows, base)
    merged = {r: G.rrf([mdl[r], [c for c in rk[r] if c not in generic]]) for r in rows}
    shown = {r: merged[r][:X.TOP] for r in rows}
    cfg = {**X.SWEEP["gpt6sol"], **FT.MODELS[G.KIND]}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    ask = G.ask_some(lambda items: X.ask_claude(items, X.codex_api(cfg, f"agents_final_on_{base}", usage), stats, cfg))
    o = FT.rerank_depth(rows, shown, ask, X.TOP, passes=("fwd",))
    tot, usd = X.codex_usage(usage, cfg)
    out = G.OUT / sid
    out.mkdir(parents=True, exist_ok=True)
    (out / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    ranked = {r: G.fuse_order(shown[r], o["orders"], ["fwd"], r, w) + merged[r][X.TOP:] for r in rows}
    res = G.save_system(sid, rows, ranked, {"model": G.MODEL, "base": base, "w_all": w, "passes": ["fwd"], "depth": X.TOP,
                                            "effort": cfg["effort"], "usage": tot, "usd_list_equivalent": usd, **stats,
                                            "recipe": f"the final system with {base}'s list, v33 lean rerun (D203)"})
    A.receipt(f"system_on_{base}", {"cases": len(rows), "usage": tot, "usd_list_equivalent": usd,
                                     "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res.get("ndcg_at_10"), **stats})
    print(json.dumps({"system": sid, "ndcg_at_10": res.get("ndcg_at_10"), "usd_list_equivalent": usd}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.lean_rerun", description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["register", "ranker", "model", "sample", "rr", "pointwise", "agent", "system"])
    ap.add_argument("--id")
    ap.add_argument("--kind")
    ap.add_argument("--tool", choices=["words", "embedding"])
    ap.add_argument("--base")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] would run {a.step} {a.id or a.kind or a.tool or a.base or ''} over {BASE} into {STORE}")
        if a.step == "ranker":
            print(json.dumps(next(s for s in specs() if s["id"] == a.id)["adapter_args"]))
        return 0
    os.environ.setdefault("UOL_LUNA_CACHE", str(A.ROOT / ".cache" / "footing" / "codex_noweb.sqlite"))
    if a.step == "register":
        return register()
    if a.step == "ranker":
        return ranker(a.id)
    if a.step == "model":
        if a.kind not in CODEX:
            raise SystemExit(f"{a.kind}: not a subscription kind")
        return model(a.kind)
    if a.step == "sample":
        if a.kind not in BILLED:
            raise SystemExit(f"{a.kind}: not one of the approved billed kinds {BILLED}")
        return sample(a.kind)
    if a.step == "rr":
        return rr(a.kind)
    if a.step == "pointwise":
        return pointwise()
    if a.step == "agent":
        return agent(a.tool)
    return system(a.base)


if __name__ == "__main__":
    sys.exit(main())
