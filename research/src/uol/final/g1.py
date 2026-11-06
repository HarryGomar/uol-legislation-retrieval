"""Gate G1 (PLAN_DRAFT §8): the harness reproduces the frozen DEV values to 1e-4.

Two routes, run in this order:

* ``cached``: frozen score files evaluated through the harness (``lstar_frozen`` .184, ``qwen3_8b`` .274,
  ``ltr_noemb_frozen`` .4081, ``ltr_emb_frozen`` .4721) and the LLM-stage hybrid recomputed from its cached stage-2
  matrix and LLM orders (``c_hybrid_cached`` .5558). Light, except C_hybrid's fusion (about 5 GB).
* ``recompute``: CPU parts recomputed through the harness: L* from code (``lstar``), the 8B dense scores from the
  cached vectors (``qwen3_8b_vectors``), and both rankers refit on a development-only pair table built from the
  frozen sources (``ltr_noemb_refit``, ``ltr_emb_refit``). Heavy: run under ``python -m uol.round3.heavy``.

Writes ``<store>/g1/G1.json`` and ``G1.md``; exit code 0 only if every checked row passes. On the run computer a
failure also writes ``G1_FAILED.md`` (the runner stops every other job).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from uol.final import context, harness, registry

TOL = 1e-4
ROUTES = {"cached": ("lstar_frozen", "qwen3_8b", "ltr_noemb_frozen", "ltr_emb_frozen", "c_hybrid_cached"),
          "recompute": ("lstar", "qwen3_8b_vectors", "ltr_noemb_refit", "ltr_emb_refit")}


GPU_TRAINED = ("ft_e5_q", "knn_casesim_e5", "knn_casesim_e5_all", "proto_casesim_e5", "colbert_bge")
REBUILD_TOL = {"source": 1e-4, "ltr_noemb_refit": 1e-4, "ltr_emb_refit": 0.003}


def rebuild_route(store: Path | None = None, force: bool = False) -> dict[str, Any]:
    """G1-rebuild (sources rebuilt from code, ``UOL_FINAL_SOURCES=rebuilt``): every rebuilt source against its frozen
    copy on DEV (nDCG@10 and max |score difference|), then both rankers refit on the rebuilt sources against .4081 /
    .4721. Gate for the rebuilt-source jobs: CPU sources and the NO-EMB ranker within 1e-4, the EMB ranker within .003
    (the a_p12 rule; GPU-trained sources are reported, not gated)."""
    import numpy as np
    from uol.embedding_push import common as ep
    from uol.final import guard, rebuild
    prof = context.active()
    if prof.sources != "rebuilt":
        raise RuntimeError("the rebuild route runs with UOL_FINAL_SOURCES=rebuilt")
    if prof.labels != "v11":
        raise RuntimeError("G1 compares on the uncorrected v11 labels (UOL_FINAL_LABELS=v11)")
    store = Path(store) if store else prof.store
    rows = guard.dev_rows_only(ep.population_rows("P_spec", ep.dev_rows()))
    out = []
    for sid in rebuild.sources():
        frozen = prof.path("ep_run") / "scores" / sid / "scores.npy"
        rebuilt = ep.score_path(sid)
        rec = {"system": f"src_{sid}", "route": "rebuild", "gated": sid not in GPU_TRAINED}
        if not rebuilt.exists():
            rec.update(status="missing", passed=False)
        else:
            F, R = np.load(frozen, mmap_mode="r"), np.load(rebuilt, mmap_mode="r")
            a, b = np.asarray(F[rows]), np.asarray(R[rows])
            ea, eb = ep.evaluate(a, rows, "P_spec")["macro"]["ndcg_at_10"], ep.evaluate(b, rows, "P_spec")["macro"]["ndcg_at_10"]
            rec.update(status="done", expected=ea, obtained=eb, abs_diff=abs(ea - eb),
                       max_abs_score_diff=float(np.nanmax(np.abs(a - b))), passed=abs(ea - eb) <= REBUILD_TOL["source"])
        out.append(rec)
        _write_rebuild(store, out)
    for sid in ("ltr_noemb_refit", "ltr_emb_refit"):
        exp = registry.get(sid)["g1"]["expected"]
        try:
            res = harness.run(sid, "DEV", store=store, force=force)
            got = res.get("ndcg_at_10")
        except Exception as exc:                                       # recorded; the gate fails
            res, got = {"status": "failed", "error": str(exc)}, None
        out.append({"system": f"{sid}{prof.tag}", "route": "rebuild", "gated": True, "expected": exp, "obtained": got,
                    "abs_diff": None if got is None else abs(got - exp), "status": res.get("status"),
                    "passed": got is not None and abs(got - exp) <= REBUILD_TOL[sid]})
        _write_rebuild(store, out)
    return _write_rebuild(store, out)


def _write_rebuild(store: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = list({r["system"]: r for r in rows}.values())         # one row per system, the latest (review 22 Oct §4.4)
    ok = bool(rows) and all(r["passed"] for r in rows if r.get("gated"))
    rec = {"tolerance": REBUILD_TOL, "rows": rows, "pass": ok, "gpu_trained_reported_only": list(GPU_TRAINED)}
    harness.write_json(Path(store) / "g1" / "G1_rebuild.json", rec)
    md = ["# G1-rebuild (sources rebuilt from code vs frozen copies, DEV nDCG@10)", "",
          "| System | Frozen | Rebuilt | |diff| | max |score diff| | Gated | Pass |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = lambda x, n=6: "-" if x is None else f"{x:.{n}f}"
        md.append(f"| {r['system']} | {f(r.get('expected'))} | {f(r.get('obtained'))} | {f(r.get('abs_diff'))} | "
                  f"{f(r.get('max_abs_score_diff'), 4)} | {r.get('gated')} | {r.get('passed')} |")
    md += ["", f"Overall: **{'PASS' if ok else 'FAIL'}**"]
    (Path(store) / "g1").mkdir(parents=True, exist_ok=True)
    (Path(store) / "g1" / "G1_rebuild.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rec


def run(routes: Sequence[str] = ("cached",), store: Path | None = None, force: bool = False,
        only: Sequence[str] | None = None) -> dict[str, Any]:
    prof = context.active()
    if prof.labels != "v11":
        raise RuntimeError("G1 runs on the uncorrected v11 labels (set UOL_FINAL_LABELS=v11)")
    store = Path(store) if store else prof.store
    path = store / "g1" / "G1.json"
    prev = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"rows": []}
    rows = {(r["system"], r["route"]): r for r in prev["rows"]}
    for route in routes:
        for sid in ROUTES[route]:
            if only and sid not in only:
                continue
            spec = registry.get(sid)
            exp = spec["g1"]["expected"]
            try:
                res = harness.run(sid, "DEV", store=store, force=force)
            except Exception as exc:                                   # recorded, then the gate fails
                res = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            got = res.get("ndcg_at_10") if res.get("status") == "done" else None
            diff = abs(got - exp) if got is not None else None
            rows[(sid, route)] = {"system": sid, "route": route, "expected": exp, "rounded": spec["g1"]["rounded"],
                                  "obtained": got, "abs_diff": diff, "pass": bool(diff is not None and diff <= TOL),
                                  "status": res.get("status"), "reason": res.get("reason") or res.get("error"),
                                  "n_cases": res.get("n_cases"), "n_groups": res.get("n_groups")}
            _write(store, list(rows.values()))
    return _write(store, list(rows.values()))


def _write(store: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ok = bool(rows) and all(r["pass"] for r in rows)
    rec = {"tolerance": TOL, "population": "DEV (P_spec dev, 1,218 cases / 1,093 groups)", "rows": rows, "pass": ok,
           "note": f"{sum(r['pass'] for r in rows)} of {len(rows)} rows within {TOL}", "profile": context.active().kind}
    harness.write_json(store / "g1" / "G1.json", rec)
    md = ["# Gate G1", "", "| System | Route | Expected | Obtained | |diff| | Pass | Status |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        got = "-" if r["obtained"] is None else f"{r['obtained']:.6f}"
        diff = "-" if r["abs_diff"] is None else f"{r['abs_diff']:.2e}"
        why = (": " + str(r["reason"])[:120]) if r.get("reason") else ""
        md.append(f"| {r['system']} | {r['route']} | {r['expected']:.6f} | {got} | {diff} | {r['pass']} | {r['status']}{why} |")
    md += ["", f"Overall: **{'PASS' if ok else 'FAIL'}** ({rec['note']})"]
    (store / "g1").mkdir(parents=True, exist_ok=True)
    (store / "g1" / "G1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return rec
