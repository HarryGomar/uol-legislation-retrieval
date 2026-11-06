"""Common-footing runs (D195, 1 Nov 2026; plan and decision rule: research/programs/common_footing_20261101/PLAN.md).

Every comparison the thesis relies on, made on the 1,217 development cases over the ranker ``S0+qwen3_8b``:

    sol-ask    GPT-6 Sol over a base ranker's top 30 (Codex CLI, the Opus run's prompt, windows and passes); holds no lock
    sol-fuse   those orders fused per fold (``external.fuse``), evaluated on DEV and compared with its references
    ranker     a ranker row of ``SPECS`` on the five folds (``harness.run``)
    effort-ask, effort-fuse, effort-full   the reasoning levels on the 300-case sample, then on all cases (the plan's addendum 2)
    contrast   the additions family of the plan's rule B
    new-cases  the chosen reread over the ranker on the new cases, once (step 4)

Outputs: ``experiments/runs/footing_20261101``; work: ``.cache/footing``. The registered records and the audit store are
only read. Every compute step is a dry run unless ``--go``. No step reads the test set (``guard.dev_rows_only``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ["UOL_AUDIT_D181_RUN"] = str(ROOT / "experiments" / "runs" / "footing_20261101")
os.environ["UOL_AUDIT_D181_WORK"] = str(ROOT / ".cache" / "footing")
os.environ["UOL_AUDIT_D181_SYS"] = "S0+qwen3_8b"

from uol.final import audit_d181_common as A, audit_d181_dev as D  # noqa: E402

RANKER = "S0+qwen3_8b"
AUDIT = ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev" / "store"
STORE = D.STORE
PROMPT_SHA = "6a417a3a817aca7590eeb4106de42cf3f61434ebd60f2ff19e1f19461fde5705"
OPUS = f"claude_rerank_{RANKER}"


def sol_system(base: str) -> str:
    return f"gpt6sol_rerank_{base}"


def base_top200(base: str) -> Path:
    """The base ranker's development ranking: the audit store's ranker, the record's largest ranker, or this run's."""
    for store in (STORE, AUDIT, A.STORE):
        p = store / "runs" / base / "DEV" / "top200.parquet"
        if p.exists():
            return p
    raise FileNotFoundError(f"no development ranking for {base}")


def ensure_base(base: str) -> None:
    """``llm_jobs.base_matrix`` reads the base from the store it writes to: a verified copy, never a link."""
    src = base_top200(base).parent
    if src.parent.parent.parent != STORE:
        A.copy_tree_verified(src, STORE / "runs" / base / "DEV")


def spec_rows() -> list[int]:
    from uol.embedding_push import common as ep
    from uol.final import guard
    spec = [int(r) for r in guard.dev_rows_only(ep.population_rows("P_spec", ep.dev_rows()), "P_spec rows")]
    if len(spec) != 1217:
        raise RuntimeError(f"{len(spec)} P_spec development rows, expected 1,217")
    return spec


def orders_dir(base: str) -> Path:
    return A.WORK / "sol" / base


def sol_ask(base: str, smoke: int | None, go: bool) -> int:
    if not go:
        print(f"[dry] would ask gpt-6-sol over {base}'s top 30 ({base_top200(base)}), {smoke or 1217} cases, into {orders_dir(base)}")
        print(f"[dry] codex sign-in: {A.codex_signed_in()}")
        return 0
    s = A.codex_signed_in()
    if not s["ok"]:
        raise RuntimeError(f"Codex CLI not signed in: {s}")
    from uol.final import external as X
    from uol.round3 import l1_luna
    if hashlib.sha256(l1_luna.PROMPT.read_bytes()).hexdigest() != PROMPT_SHA:
        raise RuntimeError("the reread's prompt differs from the Opus run's")
    A.install_systems([A.spec_llm(base, sol_system(base), "claude")], dry=False)
    D.activate(STORE)
    rows = spec_rows()
    rows = rows[:smoke] if smoke else rows
    work = orders_dir(base) / (f"smoke{smoke}" if smoke else "full")
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(base), work, A.CLAUDE_VENV, A.ENV_FILE, rows=rows, kind="gpt6sol")
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    o.update({"population": "DEV" if not smoke else f"DEV smoke ({smoke} rows)", "base": base,
              "notes": "common-footing run D195, research/programs/common_footing_20261101/PLAN.md"})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    secs = round(time.perf_counter() - t0, 1)
    res = {k: o.get(k) for k in ("model", "effort", "unanswered", "repairs", "usage", "usd", "usd_list_equivalent", "prompt_sha256")}
    A.receipt(f"sol_ask_{base}" + (f"_smoke{smoke}" if smoke else ""), {"orders": str(path), "rows": len(rows), "seconds": secs, **res})
    print(json.dumps({"rows": len(rows), "seconds": secs, **res}, indent=1), flush=True)
    return 0


def sol_fuse(base: str, go: bool) -> int:
    system = sol_system(base)
    src = orders_dir(base) / "full" / "gpt6sol_orders_s0.json"
    if not go:
        print(f"[dry] would fuse {src} over {base}, evaluate {system} on DEV into {STORE}")
        return 0
    from uol.final import external as X, harness
    A.install_systems([A.spec_llm(base, system, "claude")], dry=False)
    D.activate(STORE)
    ensure_base(base)
    X.BASE = base
    X.SYSTEMS["gpt6sol"] = system
    (X.ext_dir() / X.FILES["gpt6sol"]).write_bytes(src.read_bytes())
    if set(json.loads(src.read_text(encoding="utf-8"))["rows"]) != set(spec_rows()):
        raise RuntimeError("the orders do not cover the 1,217 development cases exactly")
    X.fuse("gpt6sol", STORE)
    summary = harness.run(system, "DEV", store=STORE)
    meta = json.loads((A.WORK / "dev" / "llm" / system / "meta.json").read_text(encoding="utf-8"))
    A.receipt(f"sol_fuse_{base}", {"summary": summary, "w_all": meta.get("w_all"), "w_by_fold": meta.get("w_by_fold")})
    print(json.dumps({"ndcg_at_10": summary.get("ndcg_at_10"), "w_all": meta.get("w_all"), "w_by_fold": meta.get("w_by_fold")}), flush=True)
    return 0


# --------------------------------------------------------------------------- reasoning levels (addendum 2)


def effort_kinds() -> dict[str, dict]:
    from uol.final import external as X
    sol, opus = X.SWEEP["gpt6sol"], X.CLAUDE
    return {"sol_low": {**sol, "effort": "low"}, "sol_high": {**sol, "effort": "high"},
            "opus_high": {**opus, "effort": "high", "max_tokens": 8000, "cap_usd": 30.0, "sync_per_mtok": [4.0, 20.0]},
            **{f"{k}_{e}": {**c, "effort": e} for k, c in MODELS.items() if c.get("route") == "codex" for e in ("low", "high")}}


LUNA = {"model": "gpt-5.6-luna", "effort": "medium", "route": "codex", "workers": 6,
        "list_per_mtok": {"input": 0.20, "cached_input": 0.02, "output": 1.20},
        "pricing": "Codex CLI subscription (no per-token bill); list-price equivalent at the Standard tier, $0.20 / $0.02 cached / $1.20 per MTok"}


def _codex(model: str, usd_in: float, usd_cached: float, usd_out: float) -> dict:
    return {"model": model, "effort": "medium", "route": "codex", "workers": 6,
            "list_per_mtok": {"input": usd_in, "cached_input": usd_cached, "output": usd_out},
            "pricing": "Codex CLI subscription (no per-token bill); list-price equivalent at the Standard tier, "
                       f"${usd_in} / ${usd_cached} cached / ${usd_out} per MTok (read 2 Nov 2026)"}


MODELS = {"luna": LUNA, "gpt6solnw": _codex("gpt-6-sol", 2.0, 0.20, 10.0), "gpt61sol": _codex("gpt-6.1-sol", 2.0, 0.10, 10.0), "gpt6astra": {**_codex("gpt-6-astra", 10.0, 1.0, 50.0), "workers": 20},
          "gpt61terra": _codex("gpt-6.1-terra", 1.0, 0.05, 5.0), "gpt6luna": _codex("gpt-6-luna", 0.10, 0.01, 0.50),
          "gpt56sol": _codex("gpt-5.6-sol", 4.0, 0.40, 20.0), "gpt56terra": _codex("gpt-5.6-terra", 2.0, 0.20, 12.0)}


def api_kind(kind: str, cap_usd: float, sync: bool = False) -> str:
    """The same model and reasoning level as a subscription kind, through the OpenAI API: no agent instructions, no
    tools. Registered as ``api_<kind>``, with its own cap."""
    from uol.final import external as X
    cfg = MODELS[kind]
    lp = cfg["list_per_mtok"]
    name = f"api_{kind}"
    X.SWEEP[name] = MODELS[name] = {
        "model": cfg["model"], "effort": cfg["effort"], "route": "openai", "max_tokens": 16000, "cap_usd": cap_usd, "sync": sync,
        "prices": [lp["input"], lp["cached_input"], lp["output"]],
        "pricing": f"OpenAI API, {'standard' if sync else 'Batch'} tier; standard list ${lp['input']} / ${lp['cached_input']} cached / ${lp['output']} per MTok (read 2 Nov 2026)"}
    return name


def api_ask(kind: str, smoke: int | None, cap: float, sync: bool, go: bool) -> int:
    """The chosen model through the API on all 1,217 cases (or the first ``smoke`` rows of the sample, to check the route)."""
    from uol.final import external as X, populations
    from uol.embedding_push import common as ep
    name = api_kind(kind, cap, sync)
    if go:
        D.activate(STORE)                                  # the corrected labels: 1,217 rows, not 1,218
    rows = spec_rows() if smoke is None else [int(r) for r in populations.agent_sample(ep.dev_rows())][:smoke]
    if not go:
        print(f"[dry] would ask {MODELS[name]['model']} at {MODELS[name]['effort']} through the API on {len(rows)} cases, cap USD {cap}")
        return 0
    register_efforts([name])
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    work = A.WORK / "api" / kind / ("full" if smoke is None else f"smoke{smoke}")
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(RANKER), work, A.API_VENV, A.ENV_FILE, rows=rows, kind=name)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    res = {k: o.get(k) for k in ("model", "effort", "unanswered", "repairs", "usage", "usd")}
    A.receipt(f"api_ask_{kind}" + ("" if smoke is None else f"_smoke{smoke}"), {"orders": str(path), "rows": len(o["rows"]), "seconds": round(time.perf_counter() - t0, 1), **res})
    print(json.dumps({"seconds": round(time.perf_counter() - t0, 1), **res}, indent=1), flush=True)
    if smoke is not None:
        return 0
    from uol.final import harness
    system = effort_system(name)
    (X.ext_dir() / X.FILES[name]).write_bytes(Path(path).read_bytes())
    X.fuse(name, STORE)
    summary = harness.run(system, "DEV", store=STORE, force=True)
    fam = A.dev_family(f"api_{kind}", [(name, A.per_case(STORE, system), kind, A.per_case(STORE, effort_system(kind)))])
    A.write_json(D.DEV / "contrasts_100k" / f"api_{kind}.json", fam)
    out = {"ndcg_at_10": summary.get("ndcg_at_10"), "usd": o.get("usd"), "contrast": _keep(fam)}
    A.receipt(f"api_full_{kind}", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def effort_system(kind: str) -> str:
    return f"{kind}_rerank_{RANKER}"


def register_efforts(kinds: list[str]) -> None:
    from uol.final import external as X
    for k, cfg in {**effort_kinds(), **MODELS}.items():
        X.SWEEP[k] = cfg
    for k in kinds:
        X.SWEEP.setdefault(k, {"pricing": "the answers of the medium run, on the sample's rows"})
        X.SYSTEMS[k], X.FILES[k] = effort_system(k), f"{k}_orders_s0.json"
    path = A.RUN / "systems_audit.json"
    have = {s["id"] for s in json.loads(path.read_text(encoding="utf-8"))["systems"]} if path.exists() else set()
    if {effort_system(k) for k in kinds} <= have:
        os.environ["UOL_FINAL_SYSTEMS"] = str(path)  # concurrent runs share the file and never rewrite it
    else:
        A.install_systems([A.spec_llm(RANKER, effort_system(k), "claude") for k in kinds], dry=False)


def effort_ask(kind: str, smoke: int | None, go: bool) -> int:
    from uol.final import external as X
    cfg = {**effort_kinds(), **MODELS, "sonnet": X.SWEEP["sonnet"], "haiku": X.SWEEP["haiku"]}[kind]
    if not go:
        print(f"[dry] would ask {cfg['model']} at {cfg['effort']} over {RANKER}'s top 30, {smoke or 300} sample cases")
        return 0
    from uol.embedding_push import common as ep
    from uol.final import populations
    from uol.round3 import l1_luna
    if hashlib.sha256(l1_luna.PROMPT.read_bytes()).hexdigest() != PROMPT_SHA:
        raise RuntimeError("the reread's prompt differs from the Opus run's")
    register_efforts([kind])
    D.activate(STORE)
    rows = [int(r) for r in populations.agent_sample(ep.dev_rows())]
    if len(rows) != 300:
        raise RuntimeError(f"{len(rows)} sample rows, expected 300")
    rows = rows[:smoke] if smoke else rows
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(RANKER), A.WORK / "effort" / kind, A.CLAUDE_VENV, A.ENV_FILE, rows=rows, kind=kind)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    o.update({"population": "DEV-A300" if not smoke else f"DEV-A300 smoke ({smoke} rows)", "base": RANKER,
              "notes": "reasoning levels, research/programs/common_footing_20261101/PLAN.md, addendum 2"})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    secs = round(time.perf_counter() - t0, 1)
    res = {k: o.get(k) for k in ("model", "effort", "unanswered", "repairs", "usage", "usd", "usd_list_equivalent", "prompt_sha256")}
    A.receipt(f"effort_ask_{kind}" + (f"_smoke{smoke}" if smoke else ""), {"orders": str(path), "rows": len(rows), "seconds": secs, **res})
    print(json.dumps({"rows": len(rows), "seconds": secs, **res}, indent=1), flush=True)
    return 0


def effort_fuse(go: bool) -> int:
    """All five settings fused on the sample's rows and evaluated on DEV-A300, then the plan's two families."""
    kinds = [*effort_kinds(), "sol_medium", "opus_medium"]
    if not go:
        print(f"[dry] would fuse and evaluate {kinds} on DEV-A300 into {STORE}")
        return 0
    from uol.embedding_push import common as ep
    from uol.final import external as X, harness, populations
    register_efforts(kinds)
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    rows = [int(r) for r in populations.agent_sample(ep.dev_rows())]
    medium = {"sol_medium": orders_dir(RANKER) / "full" / "gpt6sol_orders_s0.json",
              "opus_medium": ROOT / ".cache" / "audit_d181" / "dev" / "external" / "claude_orders_DEV-A300.json"}
    for k in kinds:
        o = json.loads((medium.get(k) or A.WORK / "effort" / k / X.FILES[k]).read_text(encoding="utf-8"))
        if set(rows) - {int(r) for r in o["rows"]}:
            raise RuntimeError(f"{k}: sample rows missing")
        o["rows"], o["orders"] = rows, {p: {str(r): o["orders"][p][str(r)] for r in rows} for p in o["passes"]}
        (X.ext_dir() / X.FILES[k]).write_text(json.dumps(o, indent=1), encoding="utf-8")
        X.fuse(k, STORE)
        harness.run(effort_system(k), "DEV-A300", store=STORE)
    here = lambda k: A.per_case(STORE, effort_system(k), "DEV-A300")
    fam = A.dev_family("effort_sample", [(k, here(k), "sol_medium", here("sol_medium")) for k in effort_kinds()], population="DEV-A300")
    same = A.dev_family("effort_same_model", [("sol_high", here("sol_high"), "sol_medium", here("sol_medium")),
                                              ("opus_high", here("opus_high"), "opus_medium", here("opus_medium"))], population="DEV-A300")
    A.write_json(D.DEV / "contrasts_100k" / "effort_sample.json", fam)
    A.write_json(D.DEV / "contrasts_100k" / "effort_same_model.json", same)
    out = {"means": {k: float(here(k)["ndcg_at_10"].mean()) for k in kinds}}
    for name, g in (("sample", fam), ("same_model", same)):
        out[name] = {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in g["contrasts"].items()}
    A.receipt("effort_fuse", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def sweep_fuse(go: bool) -> int:
    """Addendum 3: Sonnet and Haiku fused on the sample's rows; the family against Sol at medium and the gains over the ranker."""
    if not go:
        print(f"[dry] would fuse and evaluate sonnet and haiku on DEV-A300 into {STORE}")
        return 0
    from uol.final import external as X, harness
    register_efforts(["sonnet", "haiku"])
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    for k in ("sonnet", "haiku"):
        (X.ext_dir() / X.FILES[k]).write_bytes((A.WORK / "effort" / k / X.FILES[k]).read_bytes())
        X.fuse(k, STORE)
        harness.run(effort_system(k), "DEV-A300", store=STORE)
    here = lambda k: A.per_case(STORE, effort_system(k), "DEV-A300")
    rk = A.per_case(AUDIT, RANKER)
    rk = rk[rk["query_id"].isin(here("sol_medium")["query_id"])]
    kinds = ("sol_medium", "opus_medium", "sonnet", "haiku")
    fam = A.dev_family("sweep_sample", [(k, here(k), "sol_medium", here("sol_medium")) for k in ("sonnet", "haiku")], population="DEV-A300")
    gain = A.dev_family("sweep_gain", [(k, here(k), RANKER, rk) for k in kinds], population="DEV-A300")
    A.write_json(D.DEV / "contrasts_100k" / "sweep_sample.json", fam)
    A.write_json(D.DEV / "contrasts_100k" / "sweep_gain.json", gain)
    out = {"means": {**{k: float(here(k)["ndcg_at_10"].mean()) for k in kinds}, RANKER: float(rk["ndcg_at_10"].mean())}}
    for name, g in (("against_sol", fam), ("over_ranker", gain)):
        out[name] = {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in g["contrasts"].items()}
    A.receipt("sweep_fuse", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def judge(go: bool) -> int:
    """Addendum 6: the pointwise judge over the ranker on the sample, alone and fused by reciprocal rank."""
    if not go:
        print(f"[dry] would run the pointwise judge over {RANKER}'s top 30 on the 300 sample cases (spend cap US$10)")
        return 0
    from uol.embedding_push import common as ep
    from uol.final import external as X, harness, populations
    names = {"jev": f"jev_{RANKER}", "jev_rrf": f"jev_rrf_{RANKER}"}
    path = A.RUN / "systems_audit.json"
    have = {s["id"] for s in json.loads(path.read_text(encoding="utf-8"))["systems"]}
    if set(names.values()) <= have:
        os.environ["UOL_FINAL_SYSTEMS"] = str(path)
    else:
        A.install_systems([A.spec_llm(RANKER, n, "claude") for n in names.values()], dry=False)
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    X.SYSTEMS.update(names)
    rows = [int(r) for r in populations.agent_sample(ep.dev_rows())]
    src = X.drive_jev(base_top200(RANKER), A.WORK / "judge", A.ENV_FILE, rows=rows)
    (X.ext_dir() / X.FILES["jev"]).write_bytes(Path(src).read_bytes())
    X.fuse("jev", STORE)
    frames = {}
    for k, n in names.items():
        harness.run(n, "DEV-A300", store=STORE)
        frames[k] = A.per_case(STORE, n, "DEV-A300")
    rk = A.per_case(AUDIT, RANKER)
    rk = rk[rk["query_id"].isin(frames["jev"]["query_id"])]
    fam = A.dev_family("judge_sample", [(k, f, RANKER, rk) for k, f in frames.items()], population="DEV-A300")
    A.write_json(D.DEV / "contrasts_100k" / "judge_sample.json", fam)
    out = {"means": {k: float(f["ndcg_at_10"].mean()) for k, f in frames.items()}, "ranker": float(rk["ndcg_at_10"].mean()),
           "contrasts": {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in fam["contrasts"].items()}}
    A.receipt("judge", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def rerank_depth(rows, top, ask, depth: int, passes=("fwd", "rev")) -> dict:
    """``external.rerank_orders`` at another depth: windows of 20 with a stride of 10, from the bottom up, each with the
    first 10 of the window below; a depth of 10 is one window. A depth of 30 gives exactly ``rerank_orders``.
    ``passes``: the orders asked (the final system reads the forward order alone)."""
    final: dict[str, dict[int, list[int]]] = {}
    for p in passes:
        show = (lambda lst: lst[::-1]) if p == "rev" else (lambda lst: lst)
        if depth <= 20:
            a = ask([(f"{p}A{r}", r, show(top[r][:depth])) for r in rows])
            final[p] = {r: a[f"{p}A{r}"] for r in rows}
            continue
        start = depth - 20
        below = ask([(f"{p}W{start}_{r}", r, show(top[r][start:depth])) for r in rows])
        tail = {r: below[f"{p}W{start}_{r}"][10:] for r in rows}
        head = {r: below[f"{p}W{start}_{r}"][:10] for r in rows}
        while start > 0:
            start -= 10
            got = ask([(f"{p}W{start}_{r}", r, show(top[r][start:start + 10] + head[r])) for r in rows])
            for r in rows:
                w = got[f"{p}W{start}_{r}"]
                head[r], tail[r] = w[:10], w[10:] + tail[r]
        final[p] = {r: head[r] + tail[r] for r in rows}
    return {"variant": "claude", "rows": [int(r) for r in rows], "passes": list(passes),
            "orders": {p: {str(r): o for r, o in v.items()} for p, v in final.items()}}


def depth_run(kind: str, depth: int, smoke: int | None, go: bool) -> int:
    """Addendum 6: the chosen model at another depth, on all cases, compared with the same model at 30."""
    from uol.final import external as X, harness
    cfg = {**MODELS, "gpt6sol": X.SWEEP["gpt6sol"], **effort_kinds()}[kind]
    k = f"{kind}_d{depth}"
    if not go:
        print(f"[dry] would ask {cfg['model']} over the first {depth} on {smoke or 1217} cases as {k}")
        return 0
    X.SWEEP[k] = {**cfg}
    register_efforts([k])
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    X.TOP = depth
    X.rerank_orders = lambda rows, top, ask: rerank_depth(rows, top, ask, depth)
    rows = spec_rows()
    rows = rows[:smoke] if smoke else rows
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(RANKER), A.WORK / "effort" / k, A.CLAUDE_VENV, A.ENV_FILE, rows=rows, kind=k)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    o.update({"population": "DEV" if not smoke else f"DEV smoke ({smoke} rows)", "base": RANKER, "depth": depth,
              "notes": "depth, research/programs/common_footing_20261101/PLAN.md, addendum 6"})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    res = {x: o.get(x) for x in ("model", "effort", "unanswered", "repairs", "usage", "usd_list_equivalent")}
    A.receipt(f"depth_ask_{k}" + (f"_smoke{smoke}" if smoke else ""), {"rows": len(rows), "seconds": round(time.perf_counter() - t0, 1), **res})
    print(json.dumps(res, indent=1), flush=True)
    if smoke:
        return 0
    (X.ext_dir() / X.FILES[k]).write_bytes(Path(path).read_bytes())
    X.fuse(k, STORE)
    summary = harness.run(effort_system(k), "DEV", store=STORE)
    print(json.dumps({"ndcg_at_10": summary.get("ndcg_at_10")}), flush=True)
    return 0


def effort_full(kind: str, go: bool) -> int:
    """A setting that advanced from the sample, or Luna: asked on all 1,217 cases, fused per fold, compared with Sol at
    medium. The sample's fusion stays in ``meta_sample.json``; the sample's evaluation stays in its own folder."""
    cfg = {**effort_kinds(), **MODELS}[kind]
    if cfg.get("route") != "codex":
        raise RuntimeError(f"{kind} is billed: it needs the author's approval first")
    if not go:
        print(f"[dry] would ask {cfg['model']} at {cfg['effort']} on the 1,217 cases, fuse and compare with Sol at medium")
        return 0
    from uol.final import external as X, harness
    register_efforts([kind])
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    system = effort_system(kind)
    work = A.WORK / "effort" / kind / "full"
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(RANKER), work, A.CLAUDE_VENV, A.ENV_FILE, rows=spec_rows(), kind=kind)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    o.update({"population": "DEV", "base": RANKER, "notes": "reasoning levels, research/programs/common_footing_20261101/PLAN.md, addendum 2"})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    secs = round(time.perf_counter() - t0, 1)
    res = {k: o.get(k) for k in ("model", "effort", "unanswered", "repairs", "usage", "usd", "usd_list_equivalent")}
    A.receipt(f"effort_full_ask_{kind}", {"orders": str(path), "rows": len(o["rows"]), "seconds": secs, **res})
    print(json.dumps({"seconds": secs, **res}, indent=1), flush=True)
    meta = A.WORK / "dev" / "llm" / system / "meta.json"
    if meta.exists() and not meta.with_name("meta_sample.json").exists():
        meta.with_name("meta_sample.json").write_bytes(meta.read_bytes())
    (X.ext_dir() / X.FILES[kind]).write_bytes(Path(path).read_bytes())
    X.fuse(kind, STORE)
    summary = harness.run(system, "DEV", store=STORE, force=True)  # never the score of a run stopped under addendum 7
    fam = A.dev_family(f"effort_full_{kind}", [(kind, A.per_case(STORE, system), "sol_medium", A.per_case(STORE, sol_system(RANKER)))])
    A.write_json(D.DEV / "contrasts_100k" / f"effort_full_{kind}.json", fam)
    m = json.loads(meta.read_text(encoding="utf-8"))
    out = {"ndcg_at_10": summary.get("ndcg_at_10"), "w_all": m.get("w_all"), "w_by_fold": m.get("w_by_fold"),
           "contrasts": {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in fam["contrasts"].items()}}
    A.receipt(f"effort_full_fuse_{kind}", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


# --------------------------------------------------------------------------- run 2 and 3: rankers


ARMS_DIR = A.WORK / "arms"
R3_ARMS = ROOT / "experiments" / "runs" / "round3_best_case_20261013" / "arms"
L4_COLS = ROOT / "experiments" / "runs" / "l4_leakfree_20261019" / "cols.npy"
ARM_RESULT = {"S2": "S2", "S3": "S3", "E2": "E2_adapter", "E3": "E3_graph", "L2": "L2", "L4": "L4_leakfree_20261019"}
ARM_OF = {"ft_structure": "S2", "ft_cocite": "S3", "ft_adapter": "E2", "ft_smooth": "E3", "ft_issues": "L2", "ft_patterns": "L4"}
ADDITIONS = ("ft_tesis", *ARM_OF, "ft_listwise")          # rule B's family of eight (the listwise pair apart)


CITATION_SOURCES = ["prior", "LP", "K_lex", "knn_lex_votes", "knn_lex_votes_k5", "knn_lex_votes_k50", "knn_lex_instr_votes",
                    "knn_char_votes", "docexp_bm25", "docexp_facts_bm25", "proto_tfidf", "proto_probe", "knn_qwen8b",
                    "knn_fused_qwen8b", "proto_qwen8b", "K_qwen8b", "KH_qwen8b"]


CITATION_ALL = [*CITATION_SOURCES, "K_lex_ext", "knn_lexc_all", "knn_lexc_spec", "knn_lsaadapt_all", "knn_lsaadapt_spec",
                "knn_lsacadapt_all", "knn_lsacadapt_spec", "proto_lsaadapt", "proto_lsacadapt"]


LEAN_DROP = ["K_lex", "K_lex_ext", "LP", "docexp_facts_bm25", "knn_char_votes", "knn_lex_instr_votes", "knn_lex_votes_k5", "knn_lex_votes_k50",
             "knn_lexc_all", "knn_lexc_spec", "knn_lsaadapt_all", "knn_lsaadapt_spec", "knn_lsacadapt_all", "knn_lsacadapt_spec", "proto_lsaadapt",
             "proto_lsacadapt", "H_qwen8b", "KH_qwen8b", "K_qwen8b", "knn_fused_qwen8b"]
ENCODER = ["dense_qwen8b", "knn_qwen8b", "proto_qwen8b"]   # the ranker's three encoder sources (addendum 22)


def specs() -> list[dict]:
    """The ranker's registry entry with one change each. ``ft_plain`` and ``ft_listwise`` have no topic columns."""
    import copy
    base = A._systems()[RANKER]
    out = []

    def add(sid: str, note: str, **args):
        s = copy.deepcopy(base)
        s["id"], s["notes"] = sid, f"D195 common footing: {note}"
        s["adapter_args"].update(args)
        s.pop("answer", None)
        out.append(s)

    add("ft_ranker", "the ranker refitted on this PC, the control of the additions")
    add("ft_tesis", "ranker plus the past citations of the unstructured tesis", extension="m1")
    for sid, arm in ARM_OF.items():
        add(sid, f"ranker plus the columns of arm {arm}", extension="columns", arm=arm)
    add("ft_plain", "ranker without the topic columns, pointwise", extra=[])
    add("ft_listwise", "ranker without the topic columns, LambdaMART", extra=[], method="lambdamart")
    add("ft_no_citations", "the ranker without any source built on citations (D196)", drop_sources=CITATION_SOURCES)
    add("ft_neither", "the learned ranker with neither the encoder nor any source built on citations (D196, addendum 11)",
        add_sources=[], drop_sources=[c for c in CITATION_SOURCES if "qwen8b" not in c])
    # addendum 12: the two rows above kept nine sources and one column built on citations. These read only L_star, the encoder's direct sources and the label-free columns.
    add("ft_no_citations_strict", "the ranker with no source or column built on citations (D196, addendum 12)",
        drop_sources=CITATION_ALL, drop_columns=["instr_prior"])
    add("ft_neither_strict", "the learned ranker with no encoder and no source or column built on citations (D196, addendum 12)",
        add_sources=[], drop_sources=[c for c in CITATION_ALL if "qwen8b" not in c], drop_columns=["instr_prior"])
    topics = ["lsa", "lda_sim", "lda_dm", "doc2vec"]
    add("ft_q2d", "ranker plus query2doc (D196)", extra=[*topics, "query2doc"])
    add("ft_hyde", "ranker plus HyDE (D196)", extra=[*topics, "hyde"])
    add("ft_lean", "the ranker with one source per family, eight of 28 (D196, addendum 19)", drop_sources=LEAN_DROP)
    add("ft_lean_noenc", "the ranker without the encoder, five sources (addendum 22)", drop_sources=[*LEAN_DROP, *ENCODER])
    add("ft_lean_nocit", "the ranker's tuned lexical matching and the encoder's direct score only, no column built on citations (addendum 22)",
        drop_sources=[*LEAN_DROP, *[c for c in CITATION_ALL if c not in LEAN_DROP]], drop_columns=["instr_prior"])
    return out


def arm_columns(arm: str, k: int, h) -> tuple:
    """Group ``k``'s columns of an arm on the current pair table, joined by case and candidate (bundle profile)."""
    import numpy as np
    from uol.embedding_push import common as ep
    p = ARMS_DIR / arm / (f"fold{k}.npz" if (ARMS_DIR / arm / f"fold{k}.npz").exists() else "all.npz")
    z = np.load(p, allow_pickle=False)
    d = ep.release()
    where = {str(q): i for i, q in enumerate(z["qids"])}
    width = z["vals"].shape[1]
    cols = h.empty_columns(width)
    found = total = 0
    for r in ep.dev_rows():
        qid = d.queries[int(r)].query_id
        sl, pc = h.pair_slice(int(r))
        pc = np.asarray(pc, dtype=np.int64)
        total += len(pc)
        if qid not in where:
            continue
        i = where[qid]
        if int(z["fold"][i]) != int(h.ctx.fold[int(r)]):
            raise RuntimeError(f"{qid}: fold {int(h.ctx.fold[int(r)])} here, {int(z['fold'][i])} in the arm")
        a, b = int(z["ptr"][i]), int(z["ptr"][i + 1])
        src, vals = z["pc"][a:b].astype(np.int64), z["vals"][a:b]
        o = np.argsort(src, kind="stable")
        pos = np.searchsorted(src[o], pc).clip(0, max(len(src) - 1, 0))
        hit = src[o][pos] == pc if len(src) else np.zeros(len(pc), dtype=bool)
        out = np.full((len(pc), width), np.nan, dtype=np.float32)
        out[hit] = vals[o][pos[hit]]
        cols[sl] = out
        found += int(hit.sum())
    if found < 0.999 * total:
        raise RuntimeError(f"{arm} group {k}: only {found:,} of {total:,} pairs joined")
    return cols, {"arm": arm, "file": p.name, "sha256": A.sha256(p), "pairs_joined": found, "pairs": total}


def arm_cols(arm: str, go: bool) -> int:
    """Local profile, the table the arm was designed on: each group's columns with its recorded configuration, dev rows
    only (no labels read here but the arm's own dev recipe), saved by query id and candidate."""
    sel = json.loads((R3_ARMS / ARM_RESULT[arm] / "RESULT.json").read_text(encoding="utf-8"))["selected_config"]
    if not go:
        print(f"[dry] would write {ARMS_DIR / arm} with {sel}")
        return 0
    import numpy as np
    from uol.round3 import common as C, features as FT

    def make(k: int):
        cfg, g = sel[str(k)], next(x for x in h.groups if x.role == f"dev_fold{k}")
        if arm == "S2":
            from uol.round3 import s2_topology as M
            return M.columns_for(cfg, g, h)
        if arm == "S3":
            from uol.round3 import s3_cocitation as M
            return M.columns_for(cfg, g, h)
        if arm in ("E2", "E3"):
            from uol.round3 import e2_adapter as M
            return M.columns_for_arm(arm)(cfg, g, h)
        if arm == "L2":
            from uol.round3 import l2_features as M
            return M.columns_for(cfg, g, h)
        return np.load(L4_COLS, mmap_mode="r")[:, :1]        # L4 leak-free, "max": the same columns in every group

    def save(cols, name: str) -> dict:
        d = C.release()
        dev = np.asarray(C.dev_rows())
        qids, ptr, pcs, vals = [], [0], [], []
        for r in dev:
            sl, pc = h.pair_slice(int(r))
            qids.append(d.queries[int(r)].query_id)
            pcs.append(np.asarray(pc, dtype=np.int32))
            vals.append(np.asarray(cols[sl], dtype=np.float32))
            ptr.append(ptr[-1] + len(pc))
        out = ARMS_DIR / arm / name
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, qids=np.asarray(qids), ptr=np.asarray(ptr, dtype=np.int64), pc=np.concatenate(pcs),
                 vals=np.concatenate(vals), fold=np.asarray(C.fold_of(dev), dtype=np.int16))
        return {"file": str(out), "rows": len(dev), "pairs": ptr[-1], "width": int(vals[0].shape[1]), "sha256": A.sha256(out)}

    h = FT.Harness("emb", "spec")
    t0 = time.perf_counter()
    if arm in ("E2", "E3"):        # adapters missing from the cache are retrained first, the recorded per-fold ones only
        from uol.round3 import e2_adapter as M
        M.train(arm, per_fold={int(k): v for k, v in sel.items()})
    res = {"all": save(make(0), "all.npz")} if arm in ("L2", "L4") else {f"fold{k}": save(make(k), f"fold{k}.npz") for k in range(5)}
    A.receipt(f"arm_cols_{arm}", {"selected_config": sel, "files": res, "seconds": round(time.perf_counter() - t0, 1)})
    print(json.dumps(res), flush=True)
    return 0


def ranker(sid: str, go: bool) -> int:
    all_specs = specs()
    if sid not in {s["id"] for s in all_specs}:
        raise KeyError(sid)
    if not go:
        print(f"[dry] would run {sid} on the five folds into {STORE}: "
              f"{json.dumps(next(s for s in all_specs if s['id'] == sid)['adapter_args'])}")
        return 0
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RuntimeError("set PYTHONHASHSEED=0: the doc2vec columns depend on it")
    from uol.final import harness
    path = A.RUN / "systems_audit.json"
    have = {s["id"] for s in json.loads(path.read_text(encoding="utf-8"))["systems"]} if path.exists() else set()
    if sid in have:
        os.environ["UOL_FINAL_SYSTEMS"] = str(path)  # concurrent runs share the file and never rewrite it
    else:
        A.install_systems(all_specs, dry=False)
    D.activate(STORE)
    t0 = time.perf_counter()
    summary = harness.run(sid, "DEV", store=STORE)
    A.log(f"{sid}: nDCG@10 {summary.get('ndcg_at_10')} n {summary.get('n_cases')} ({round(time.perf_counter() - t0)} s)", "rankers")
    return 0


def contrast(go: bool) -> int:
    """Rule B: the control, the family of eight against the local ranker, and each second comparison that has been run."""
    if not go:
        print(f"[dry] would write the additions family and the second comparisons into {D.DEV / 'contrasts_100k'}")
        return 0
    frame = lambda s: A.per_case(STORE, s)
    stored, local = A.per_case(AUDIT, RANKER), frame("ft_ranker")
    ctl = {"stored": float(stored["ndcg_at_10"].mean()), "local": float(local["ndcg_at_10"].mean())}
    ctl.update({"difference": ctl["local"] - ctl["stored"], "tolerance": 0.003})
    ctl["pass"] = abs(ctl["difference"]) <= ctl["tolerance"]
    pairs = [(s, frame(s), "ft_plain" if s == "ft_listwise" else "ft_ranker", frame("ft_plain" if s == "ft_listwise" else "ft_ranker"))
             for s in ADDITIONS]
    fam = A.dev_family("B_additions", pairs)
    A.write_json(D.DEV / "contrasts_100k" / "B_additions.json", fam)
    seconds = [s for s in ADDITIONS if (STORE / "runs" / sol_system(s) / "DEV" / "per_case.parquet").exists()]
    sec = {}
    if seconds:
        base = frame(sol_system(RANKER))
        sec = A.dev_family("B_second", [(sol_system(s), frame(sol_system(s)), sol_system(RANKER), base) for s in seconds])
        A.write_json(D.DEV / "contrasts_100k" / "B_second.json", sec)
    keep = ("delta", "ci95", "adjusted_lower", "class")
    out = {"control": ctl, "means": {s: float(frame(s)["ndcg_at_10"].mean()) for s in (*ADDITIONS, "ft_plain", "ft_ranker")},
           "additions": {k: {x: v[x] for x in keep} for k, v in fam["contrasts"].items()},
           "second": {k: {x: v[x] for x in keep} for k, v in sec.get("contrasts", {}).items()}}
    A.receipt("contrast_B", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0 if ctl["pass"] else 2


# --------------------------------------------------------------------------- addendum 9: the choice and what follows it


def _keep(fam: dict) -> dict:
    return {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in fam["contrasts"].items()}


def models_compare(go: bool) -> int:
    """Addenda 5 and 7: all eight models on all cases, the family of seven against GPT-6 Sol, and the rule's choice."""
    from uol.final import external as X
    ref = "gpt6solnw"
    kinds = [k for k, c in MODELS.items() if c.get("route") == "codex"
             and (k != "gpt6astra" or (A.WORK / "effort" / k / "full" / f"{k}_orders_s0.json").exists())]
    if not go:
        print(f"[dry] would compare {kinds} and write the choice to {A.RUN / 'chosen.json'}")
        return 0
    rec = {}
    for k in kinds:
        o = json.loads((A.WORK / "effort" / k / "full" / f"{k}_orders_s0.json").read_text(encoding="utf-8"))
        rec[k] = {"model": o["model"], "unanswered": o.get("unanswered"), "repairs": o.get("repairs"),
                  "usd_list": o.get("usd_list_equivalent"), "usage": o.get("usage"),
                  "ndcg_at_10": float(A.per_case(STORE, effort_system(k))["ndcg_at_10"].mean())}
    base = A.per_case(STORE, effort_system(ref))
    fam = A.dev_family("models", [(k, A.per_case(STORE, effort_system(k)), ref, base) for k in kinds if k != ref])
    A.write_json(D.DEV / "contrasts_100k" / "models.json", fam)
    kept = _keep(fam)
    for label, v in kept.items():
        rec[label.split(" - ")[0]].update(v)
    price = {k: rec[k]["usd_list"] for k in kinds}
    others = [k for k in kinds if k != ref]
    big = [k for k in others if rec[k]["adjusted_lower"] >= 0.02]
    same = [k for k in others if price[k] < price[ref] and rec[k]["class"] == "equivalent"]
    higher = [k for k in others if price[k] < price[ref] and rec[k]["ndcg_at_10"] > rec[ref]["ndcg_at_10"]]
    pick = lambda c: min(c, key=price.get)
    chosen = pick(big) if big else pick(same) if same else pick(higher) if higher else ref
    out = {"reference": ref, "chosen": chosen, "qualifying": {"adjusted_lower_at_least_margin": big, "cheaper_no_meaningful_difference": same,
                                                              "cheaper_and_higher": higher}, "models": rec}
    A.write_json(A.RUN / "chosen.json", out)
    A.receipt("models_compare", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def astra_screen(go: bool) -> int:
    """Addendum 10: GPT-6 Astra on the 300-case sample against GPT-6 Sol at medium on the same rows (m = 1). It goes on to
    all cases only if its adjusted lower end reaches the margin, the only way a model at its price can qualify."""
    if not go:
        print("[dry] would fuse GPT-6 Astra on the sample and write whether it goes on")
        return 0
    from uol.final import external as X, harness
    kind = "gpt6astra"
    register_efforts([kind])
    D.activate(STORE)
    ensure_base(RANKER)
    X.BASE = RANKER
    (X.ext_dir() / X.FILES[kind]).write_bytes((A.WORK / "effort" / kind / X.FILES[kind]).read_bytes())
    X.fuse(kind, STORE)
    summary = harness.run(effort_system(kind), "DEV-A300", store=STORE, force=True)
    here = lambda k: A.per_case(STORE, effort_system(k), "DEV-A300")
    fam = A.dev_family("astra_sample", [(kind, here(kind), "sol_medium", here("sol_medium"))], population="DEV-A300")
    A.write_json(D.DEV / "contrasts_100k" / "astra_sample.json", fam)
    o = json.loads((A.WORK / "effort" / kind / X.FILES[kind]).read_text(encoding="utf-8"))
    kept = _keep(fam)
    out = {"ndcg_at_10": summary.get("ndcg_at_10"), "sol_medium": float(here("sol_medium")["ndcg_at_10"].mean()), "unanswered": o.get("unanswered"),
           "repairs": o.get("repairs"), "usd_list": o.get("usd_list_equivalent"), "contrast": kept,
           "advance": next(iter(kept.values()))["adjusted_lower"] >= 0.02}
    A.write_json(A.RUN / "astra_screen.json", out)
    A.receipt("astra_screen", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def chosen_kind() -> str:
    return json.loads((A.RUN / "chosen.json").read_text(encoding="utf-8"))["chosen"]


def one_order(kind: str, go: bool) -> int:
    """Addendum 6: the chosen model's held answers with the forward order alone, against both orders (m = 1)."""
    if not go:
        print(f"[dry] would fuse the forward order of {kind} alone")
        return 0
    from uol.final import external as X, harness, llm_jobs
    k = f"{kind}_fwd"
    X.SWEEP[k] = {"pricing": "the answers of the two-order run, forward order only"}
    register_efforts([k])
    D.activate(STORE)
    ensure_base(RANKER)
    o = json.loads((A.WORK / "effort" / kind / "full" / f"{kind}_orders_s0.json").read_text(encoding="utf-8"))
    o = {**o, "passes": ["fwd"]}
    import numpy as np
    from uol.final import context
    rows = np.asarray(o["rows"], dtype=np.int64)
    S = llm_jobs.base_matrix(STORE, f"{RANKER}{context.active().tag}", rows)
    llm_jobs.fuse_orders(effort_system(k), RANKER, rows, [o], S, {"samples": 1, "variant": k, "model": o.get("model"),
                                                                 "recipe": "the forward pass of the two-order run"})
    summary = harness.run(effort_system(k), "DEV", store=STORE, force=True)
    fam = A.dev_family("one_order", [(k, A.per_case(STORE, effort_system(k)), kind, A.per_case(STORE, effort_system(kind)))])
    A.write_json(D.DEV / "contrasts_100k" / "one_order.json", fam)
    out = {"ndcg_at_10": summary.get("ndcg_at_10"), "contrast": _keep(fam)}
    A.receipt("one_order", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def second_system(kind: str, base: str) -> str:
    return f"{kind}_rerank_{base}"


def second_run(kind: str, base: str, go: bool) -> int:
    """The chosen model over another ranker on all cases, on the subscription route of its reference."""
    if not go:
        print(f"[dry] would ask {kind} over {base}")
        return 0
    from uol.final import external as X, harness
    for k, cfg in {**effort_kinds(), **MODELS}.items():
        X.SWEEP[k] = cfg
    system = second_system(kind, base)
    A.install_systems([A.spec_llm(base, system, "claude")], dry=False)
    D.activate(STORE)
    ensure_base(base)
    X.BASE = base
    X.SYSTEMS[kind], X.FILES[kind] = system, f"{kind}_orders_s0.json"
    t0 = time.perf_counter()
    path = X.drive_claude(base_top200(base), A.WORK / "second" / kind / base, A.CLAUDE_VENV, A.ENV_FILE, rows=spec_rows(), kind=kind)
    o = json.loads(Path(path).read_text(encoding="utf-8"))
    o.update({"population": "DEV", "base": base, "notes": "second comparison, research/programs/common_footing_20261101/PLAN.md, addendum 9"})
    Path(path).write_text(json.dumps(o, indent=1), encoding="utf-8")
    (X.ext_dir() / X.FILES[kind]).write_bytes(Path(path).read_bytes())
    X.fuse(kind, STORE)
    summary = harness.run(system, "DEV", store=STORE, force=True)
    out = {"ndcg_at_10": summary.get("ndcg_at_10"), "seconds": round(time.perf_counter() - t0, 1), "unanswered": o.get("unanswered"), "repairs": o.get("repairs")}
    A.receipt(f"second_{kind}_{base}", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def rule_b(go: bool) -> int:
    """Rule B at the ranker, the two query rewritings (m = 2) and what past citations give (m = 1). Writes the rows that
    go on to the second comparison: the tesis always, and any row whose adjusted lower end reaches the margin."""
    if not go:
        print("[dry] would write the families of additions, rewriting and past citations")
        return 0
    frame = lambda s: A.per_case(STORE, s)
    have = lambda s: (STORE / "runs" / s / "DEV" / "per_case.parquet").exists()
    stored, local = A.per_case(AUDIT, RANKER), frame("ft_ranker")
    ctl = {"stored": float(stored["ndcg_at_10"].mean()), "local": float(local["ndcg_at_10"].mean()), "tolerance": 0.003}
    ctl["pass"] = abs(ctl["local"] - ctl["stored"]) <= ctl["tolerance"]
    rows = [s for s in ADDITIONS if have(s)]
    fam = A.dev_family("B_additions", [(s, frame(s), "ft_plain" if s == "ft_listwise" else "ft_ranker", frame("ft_plain" if s == "ft_listwise" else "ft_ranker")) for s in rows], size=len(ADDITIONS))
    A.write_json(D.DEV / "contrasts_100k" / "B_additions.json", fam)
    out = {"control": ctl, "missing": [s for s in (*ADDITIONS, "ft_q2d", "ft_hyde", "ft_no_citations") if not have(s)],
           "means": {s: float(frame(s)["ndcg_at_10"].mean()) for s in (*rows, "ft_plain", "ft_ranker")}, "additions": _keep(fam)}
    onward = ["ft_tesis"] + [c["a"] for c in fam["contrasts"].values() if c["a"] != "ft_tesis" and c["adjusted_lower"] >= 0.02]
    rewriting = [s for s in ("ft_q2d", "ft_hyde") if have(s)]
    if rewriting:
        fr = A.dev_family("B_rewriting", [(s, frame(s), "ft_ranker", local) for s in rewriting], size=2)
        A.write_json(D.DEV / "contrasts_100k" / "B_rewriting.json", fr)
        out["rewriting"] = _keep(fr)
        out["means"].update({s: float(frame(s)["ndcg_at_10"].mean()) for s in rewriting})
        onward += [c["a"] for c in fr["contrasts"].values() if c["adjusted_lower"] >= 0.02]
    if have("ft_no_citations"):
        fc = A.dev_family("no_citations", [("ft_ranker", local, "ft_no_citations", frame("ft_no_citations"))])
        A.write_json(D.DEV / "contrasts_100k" / "no_citations.json", fc)
        out["no_citations"] = _keep(fc)
        out["means"]["ft_no_citations"] = float(frame("ft_no_citations")["ndcg_at_10"].mean())
    out["onward"] = onward
    A.write_json(A.RUN / "onward.json", onward)
    A.receipt("rule_b", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0 if ctl["pass"] else 2


def after_choice(kind: str, go: bool) -> int:
    """The families on the chosen model: reasoning levels (m = 2), depth (m = 2), and the second comparisons held so far."""
    if not go:
        print(f"[dry] would write the families of {kind}")
        return 0
    frame = lambda s: A.per_case(STORE, s)
    have = lambda s: (STORE / "runs" / s / "DEV" / "per_case.parquet").exists()
    base = frame(effort_system(kind))
    out = {"reference": float(base["ndcg_at_10"].mean())}
    for name, rows, size in (("levels", [effort_system(f"{kind}_{e}") for e in ("low", "high")], 2),
                             ("depth", [effort_system(f"{kind}_d{d}") for d in (10, 60)], 2),
                             ("second", [second_system(kind, b) for b in (json.loads((A.RUN / "onward.json").read_text(encoding="utf-8")) if (A.RUN / "onward.json").exists() else [])], None),
                             ("largest", [second_system(kind, "S_full_ext")], 1)):
        rows = [s for s in rows if have(s)]
        if rows:
            fam = A.dev_family(f"chosen_{name}", [(s, frame(s), effort_system(kind), base) for s in rows], size=size)
            A.write_json(D.DEV / "contrasts_100k" / f"chosen_{name}.json", fam)
            out[name] = _keep(fam)
            out.setdefault("means", {}).update({s: float(frame(s)["ndcg_at_10"].mean()) for s in rows})
    A.receipt("after_choice", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


SETTINGS = ("merged", "self", "s0", "fwd", "low", "d60")    # addendum 20, one family of six
COMBO = "gpt61sol_combo"


def settings_read(go: bool) -> int:
    """Addendum 20 and 19: each change to the combination minus the combination (m = 6, Bonferroni at six until all are
    in), and the leaner ranker minus the ranker (m = 1). Reads stored systems only; asks nothing."""
    if not go:
        print("[dry] would write system_settings.json and lean.json")
        return 0
    frame = lambda s: A.per_case(STORE, s)
    have = lambda s: (STORE / "runs" / s / "DEV" / "per_case.parquet").exists()
    combo = frame(COMBO)
    rows = [v for v in SETTINGS if have(f"{COMBO}_{v}")]
    out = {"combo": float(combo["ndcg_at_10"].mean()), "missing": [v for v in SETTINGS if v not in rows]}
    if rows:
        fam = A.dev_family("system_settings", [(f"{COMBO}_{v}", frame(f"{COMBO}_{v}"), COMBO, combo) for v in rows], size=len(SETTINGS))
        A.write_json(D.DEV / "contrasts_100k" / "system_settings.json", fam)
        out["settings"] = _keep(fam)
        out["means"] = {v: float(frame(f"{COMBO}_{v}")["ndcg_at_10"].mean()) for v in rows}
    if have("ft_lean"):
        fl = A.dev_family("lean", [("ft_lean", frame("ft_lean"), "ft_ranker", frame("ft_ranker"))])
        A.write_json(D.DEV / "contrasts_100k" / "lean.json", fl)
        out["lean"] = _keep(fl)
    A.receipt("settings_read", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


# --------------------------------------------------------------------------- addendum 8: the other rerankers


RR_JOB = "_j2"                               # jobs are immutable; the suffix names the submission whose returned jobs are read
RERANKERS = {"ce_ranker": None, "bge_ranker": "bge_rr_v2_m3", "qwen4b_ranker": "qwen3_rr_4b"}


def rr_register() -> None:
    A.install_systems([A.spec_llm(RANKER, s, "claude") for k in RERANKERS for s in (k, f"{k}_alone")], dry=False)


def rr_run(which: str, go: bool) -> int:
    """One reranker over the ranker's top 30 on the RTX 5080, all five folds where it is trained, then fused per fold
    (``rerank_jobs``, the same recipes and inputs as over the largest ranker). Returned jobs are kept and not sent again."""
    if not go:
        print(f"[dry] would send {which} to the remote worker, fuse with {RANKER}, score on all development cases")
        return 0
    os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
    os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")
    import numpy as np
    from uol.final import harness, rerank_jobs as RJ
    from uol.round3 import gpu_remote as G
    jobs = Path(G.__file__).parent / "gpu_jobs"
    helper = jobs / "rerank_bench.py"
    rr_register()
    D.activate(STORE)
    ensure_base(RANKER)
    RJ.BASE = RANKER
    t0 = time.perf_counter()
    if which == "ce_ranker":
        RJ.tag = lambda k, seed=1: f"ce_ranker_f{k}_s{seed}"
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
        RJ._fuse(which, np.concatenate(pairs), np.concatenate(scores), {"seed": 1, "recipe": "round3.r2_cross FULL_TASK per fold"})
    else:
        RJ.ZERO_SHOT[which] = RERANKERS[which]
        d = RJ.zs_inputs_dir(which)
        out = G.LOCAL_JOBS / f"{which}{RR_JOB}" / "remote"
        if not (out / "outputs" / "scores.npy").exists():
            d = RJ.zs_prep(which)
            G.submit(f"{which}{RR_JOB}", jobs / "r1_score.py", {n: d / n for n in ("queries.jsonl", "docs.jsonl", "pairs.npy", "task.json")},
                     helpers=[helper], expected_outputs=["scores.npy"])
        seconds = {which: json.loads((out / "receipt.json").read_text(encoding="utf-8")).get("seconds")}
        pairs = np.load(d / "pairs.npy")
        keep = pairs[:, 0] >= 0                            # the blank-query control pairs are not part of the ranking
        RJ._fuse(which, pairs[keep], np.load(out / "outputs" / "scores.npy")[keep], {"model": RERANKERS[which], "recipe": "round3 R1 job, zero-shot"})
    res = {s: harness.run(s, "DEV", store=STORE, force=True).get("ndcg_at_10") for s in (which, f"{which}_alone")}
    A.receipt(f"rr_{which}", {"gpu_seconds": seconds, "seconds": round(time.perf_counter() - t0, 1), "ndcg_at_10": res})
    print(json.dumps({"gpu_seconds": seconds, "ndcg_at_10": res}, indent=1), flush=True)
    return 0


def rr_contrast(go: bool) -> int:
    """Addendum 8: each reranker against the ranker, fused and alone, each family of three."""
    if not go:
        print("[dry] would write the two families of the other rerankers")
        return 0
    frame = lambda s: A.per_case(STORE, s)
    base = A.per_case(AUDIT, RANKER)
    out = {}
    for name, suffix in (("rr_fused", ""), ("rr_alone", "_alone")):
        fam = A.dev_family(name, [(f"{k}{suffix}", frame(f"{k}{suffix}"), RANKER, base) for k in RERANKERS])
        A.write_json(D.DEV / "contrasts_100k" / f"{name}.json", fam)
        out[name] = {k: {x: v[x] for x in ("delta", "ci95", "adjusted_lower", "class")} for k, v in fam["contrasts"].items()}
    A.receipt("rr_contrast", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


# --------------------------------------------------------------------------- step 4: the new cases


# stated knowledge cutoffs, developers.openai.com/api/docs/models/<model>, read 2 Nov 2026
CUTOFFS = {"gpt6solnw": "2026-04-20", "gpt61sol": "2026-04-30", "gpt61terra": "2026-06-01", "gpt6astra": "2026-04-30", "gpt6luna": "2026-05-18",
           "gpt56sol": "2026-02-16", "gpt56terra": "2026-02-16", "luna": "2026-02-16"}
SOL_CUTOFF = "2026-04-20"                    # gpt-6-sol's stated knowledge cutoff, as read for the sweep (D174)


def new_cases(kind: str, go: bool) -> int:
    """Step 4: the chosen reread over the ranker on the new cases, once, with the fusion weight of the development
    cases. Reads the audit's new-case run; writes ``<RUN>/new``. The system does not change whatever this shows."""
    from uol.final import external as X
    system = sol_system(RANKER) if kind == "gpt6sol" else effort_system(kind)
    meta_path = A.WORK / "dev" / "llm" / system / "meta.json"
    run, audit = A.RUN / "new", AUDIT.parents[1] / "new"
    if not go:
        print(f"[dry] would ask {kind} on the new cases over {RANKER} with w_all from {meta_path}, once, into {run}")
        return 0
    if (run / "scores" / system / "manifest.json").exists():
        raise FileExistsError(f"{system} is scored once")
    s = A.codex_signed_in()
    if not s["ok"]:
        raise RuntimeError(f"Codex CLI not signed in: {s}")
    import pandas as pd
    from uol.final import new_cases_llm as NL, new_cases_report as NR, new_cases_steps as NS
    from uol.round3 import features, l1_luna
    if hashlib.sha256(l1_luna.PROMPT.read_bytes()).hexdigest() != PROMPT_SHA:
        raise RuntimeError("the reread's prompt differs from the Opus run's")
    w = float(json.loads(meta_path.read_text(encoding="utf-8"))["w_all"])
    for tree in ("release_ext", "NEW", f"eval/{RANKER}", f"scores/{RANKER}"):
        A.copy_tree_verified(audit / tree, run / tree)
    A.copy_verified(audit / "HEADLINE_RULE.json", run / "HEADLINE_RULE.json")
    cfg = {**X.SWEEP["gpt6sol"], **{**MODELS, **effort_kinds()}.get(kind, {})}
    NL.P6 = {**NL.P6, "base": RANKER, "system": system}
    rows, M, nr = NL._base(run)
    top = {int(r): [int(c) for c in M[i].argsort(kind="stable")[::-1][:X.TOP]] for i, r in enumerate(rows)}
    stats: dict[str, int] = {}
    usage: dict[str, dict] = {}
    t0 = time.perf_counter()
    api = X.codex_api(cfg, f"new_{kind}", usage)
    o = X.rerank_orders([int(r) for r in rows], top, lambda items: X.ask_claude(items, api, stats, cfg))
    tot, usd = X.codex_usage(usage, cfg)
    o.update({"model": cfg["model"], "effort": cfg["effort"], **stats, "prompt_sha256": PROMPT_SHA, "population": "NEW (P_spec)",
              "usage": tot, "usd": 0.0, "usd_list_equivalent": usd, "pricing": cfg["pricing"], "base": RANKER})
    work = run / "llm" / kind
    work.mkdir(parents=True, exist_ok=True)
    (work / "orders.json").write_text(json.dumps(o, indent=1), encoding="utf-8")
    fused = features.fuse_rows(M, NL.arm(rows, [o], M.shape[1]), w)
    NL._save(run, system, nr, rows, fused, {"base": RANKER, "w_all": w, "samples": 1, "model": cfg["model"], "effort": cfg["effort"],
                                            "usage": tot, "usd": 0.0, "usd_list_equivalent": usd, "unanswered": stats.get("unanswered"),
                                            "repairs": stats.get("repairs"), "orders_sha256": NL.sha256(work / "orders.json"),
                                            "seconds": round(time.perf_counter() - t0, 1)})
    A.receipt(f"new_{kind}", {"w_all": w, "usage": tot, "usd_list_equivalent": usd, "seconds": round(time.perf_counter() - t0, 1), **stats})
    return new_cases_read(kind, True)


def new_cases_read(kind: str, go: bool) -> int:
    """The evaluation of the reread saved by ``new_cases``, read against the ranker and Opus on the same cases."""
    system = sol_system(RANKER) if kind == "gpt6sol" else effort_system(kind)
    run, audit = A.RUN / "new", AUDIT.parents[1] / "new"
    if not go:
        print(f"[dry] would evaluate {system} on the new cases in {run}")
        return 0
    import pandas as pd
    from uol.final import new_cases_llm as NL, new_cases_report as NR, new_cases_sources as NSrc, new_cases_steps as NS
    w = float(json.loads((run / "scores" / system / "manifest.json").read_text(encoding="utf-8"))["w_all"])
    NL.P6 = {**NL.P6, "base": RANKER, "system": system}
    stored = pd.read_parquet(audit / "eval" / RANKER / "per_case.parquet")
    NSrc.first_look_ids = lambda: set(stored.loc[stored["first_look"], "query_id"])   # the flag as the audit's evaluation recorded it
    NS.evaluate(run, [system])
    frames = {system: pd.read_parquet(run / "eval" / system / "per_case.parquet"),
              RANKER: pd.read_parquet(audit / "eval" / RANKER / "per_case.parquet"),
              OPUS: pd.read_parquet(audit / "eval" / OPUS / "per_case.parquet")}
    rule = json.loads((audit / "HEADLINE_RULE.json").read_text(encoding="utf-8"))
    dates = NR.decision_dates(run / "NEW" / "release")
    headline = set(dates) - {c["query_id"] for c in rule["overlap_cases"]}
    cutoff = CUTOFFS.get(kind, SOL_CUTOFF)
    after = {q for q, d in dates.items() if d["earliest"] and d["earliest"] > cutoff} & headline
    out = {"w_all": w, "cutoff": cutoff, "summaries": {k: NR.subset_summary(f, headline) for k, f in frames.items()}, "contrasts": {}}
    for x in (RANKER, OPUS):
        out["contrasts"][f"{system} - {x}"] = {"all": NR.pair(frames, system, x, "ndcg_at_10", 2, headline),
                                               "after_cutoff": NR.pair(frames, system, x, "ndcg_at_10", 2, after),
                                               "rest": NR.pair(frames, system, x, "ndcg_at_10", 2, headline - after)}
    A.write_json(run / "contrasts.json", out)
    print(json.dumps(out, indent=1, default=float), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.footing", description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=["sol-ask", "sol-fuse", "arm-cols", "ranker", "effort-ask", "effort-fuse", "effort-full", "api-ask", "rr", "rr-contrast", "new-cases-read", "astra-screen", "models", "chosen", "one-order", "second", "rule-b", "after-choice", "sweep-fuse", "judge", "depth", "contrast", "new-cases", "settings"])
    ap.add_argument("--depth", type=int, default=None)
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--kind", default=None)
    ap.add_argument("--base", default=RANKER)
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--arm", default=None)
    ap.add_argument("--id", default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "sol-ask":
        return sol_ask(a.base, a.smoke, a.go)
    if a.step == "arm-cols":
        return arm_cols(a.arm, a.go)
    if a.step == "ranker":
        return ranker(a.id, a.go)
    if a.step == "effort-ask":
        return effort_ask(a.kind, a.smoke, a.go)
    if a.step == "effort-fuse":
        return effort_fuse(a.go)
    if a.step == "astra-screen":
        return astra_screen(a.go)
    if a.step == "models":
        return models_compare(a.go)
    if a.step == "chosen":
        print(chosen_kind())
        return 0
    if a.step == "one-order":
        return one_order(a.kind or chosen_kind(), a.go)
    if a.step == "second":
        return second_run(a.kind or chosen_kind(), a.base, a.go)
    if a.step == "rule-b":
        return rule_b(a.go)
    if a.step == "after-choice":
        return after_choice(a.kind or chosen_kind(), a.go)
    if a.step == "rr":
        return rr_run(a.kind, a.go)
    if a.step == "rr-contrast":
        return rr_contrast(a.go)
    if a.step == "api-ask":
        return api_ask(a.kind, a.smoke, a.cap, a.sync, a.go)
    if a.step == "effort-full":
        return effort_full(a.kind, a.go)
    if a.step == "sweep-fuse":
        return sweep_fuse(a.go)
    if a.step == "judge":
        return judge(a.go)
    if a.step == "depth":
        return depth_run(a.kind, a.depth, a.smoke, a.go)
    if a.step == "contrast":
        return contrast(a.go)
    if a.step == "settings":
        return settings_read(a.go)
    if a.step == "new-cases-read":
        return new_cases_read(a.kind or chosen_kind(), a.go)
    if a.step == "new-cases":
        return new_cases(a.kind or "gpt6sol", a.go)
    return sol_fuse(a.base, a.go)


if __name__ == "__main__":
    sys.exit(main())
