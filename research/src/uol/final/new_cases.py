"""E13: the new cases (NEW), scored once with models trained on every development case (D161, D162).

TEST-R is dropped (D148). The work runs in two phases around the freeze record (``FREEZE.json``):

Before the freeze (refused once FREEZE.json exists):
* ``fit``: the three learned rankers trained on all development cases, saved with sha256 (``new_cases_fit``);
* the dry run (criterion A, D162), development cases only: ``dryrun-rankers`` (the saved models against their own
  output, the per-case pool, feature and M1 columns against the fit's), ``dryrun-lstar`` (lstar_headings' top 200
  against the record's), ``dryrun-sources`` (the query encoders against the stored vectors, the fold-0 first stage
  against the frozen source files), ``dryrun-smoke`` (every post-freeze step end to end on three stand-in cases) and
  ``spread`` (one fold-0 refit's nDCG@10 beside the record's, reported);
* ``freeze``: FREEZE.json when every exact check passed: systems, configurations, model and code sha256, dry run.

After the freeze (refused before it; ``encode-local``, ``sources`` and ``score`` also run before it on the smoke
folder only):
* ``collect`` and ``build-sources --cutoff <day>``: the existing collector (``round2.c_collect``, public pages, one
  request at a time, fails closed) on a copy of round 2's raw store in ``<run>/collect``;
* ``release --sources <jsonl>``: the frozen resolver and masker (``round2.c_release.build_set``), the extended release
  and the counts;
* ``encode-local``; ``bundle-new --out <Drive folder>`` (the run computer's part: bge-m3, Qwen3-8B int8, ColBERT
  pairs, ``gpu_jobs/new_encode.py``); ``ingest --runb <its results folder>``;
* ``sources``, ``score`` (P1, S0, S_full, S_full_ext), ``evaluate``; ``llm`` (P6) and ``external`` (Claude, Jev),
  then ``evaluate`` again for them. Each system is scored once.

    py -3.12 -m uol.final.new_cases <step> [--run DIR] [--models DIR] [--work DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FREEZE = ROOT / "research" / "programs" / "final_benchmark_20261021" / "FREEZE.json"
RUN = ROOT / "experiments" / "runs" / "new_cases_20261026"
MODELS = RUN / "models"
DRY = RUN / "dryrun"
WORK = ROOT / ".cache" / "new_cases_20261026"
PRE_FREEZE = ("overlay-verify", "fit", "dryrun-rankers", "dryrun-lstar", "dryrun-sources", "dryrun-smoke", "spread", "freeze")
POST_FREEZE = ("collect", "build-sources", "release", "encode-local", "bundle-new", "ingest", "sources", "score",
               "evaluate", "llm", "external", "report")
SMOKE_OK = ("encode-local", "sources", "score")         # before the freeze on the smoke folder only
STEPS = PRE_FREEZE + POST_FREEZE


def freeze_path() -> Path:
    return Path(os.environ.get("UOL_FINAL_FREEZE", FREEZE))


def _frozen() -> bool:
    return freeze_path().exists()


def check_phase(name: str, run: Path | None = None) -> None:
    if name not in STEPS:
        raise ValueError(f"unknown step {name!r}; {STEPS}")
    smoke = run is not None and Path(run).resolve() == (WORK / "smoke").resolve()
    if smoke and name not in SMOKE_OK:
        raise PermissionError(f"{name}: not a step of the smoke run")
    if smoke and _frozen():
        raise PermissionError("FREEZE.json exists; the smoke run belongs to the dry run")
    if name in POST_FREEZE and not _frozen() and not smoke:
        raise PermissionError("E13 runs only after the freeze (FREEZE.json); the dry run must pass first")
    if name in PRE_FREEZE and _frozen():
        raise PermissionError(f"{name}: FREEZE.json exists; the frozen models and checks are not redone")


def step(name: str) -> int:
    """The step without options (kept for ``python -m uol.final new-cases --step``)."""
    return main([name])


def build_bundle(out: Path) -> int:
    return main(["bundle-new", "--out", str(out)])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="uol.final.new_cases", description=__doc__.splitlines()[0])
    ap.add_argument("step", choices=STEPS)
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--models", default=None)
    ap.add_argument("--work", default=None)
    ap.add_argument("--rankers", default=None, help="comma list (default S0,S_full,S_full_ext)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sources", default=None, help="release: the collector's sources_<start>_<cutoff>.jsonl")
    ap.add_argument("--runb", default=None, help="ingest: RUN-B's output folder")
    ap.add_argument("--cutoff", default=None, help="build-sources: the collection day (YYYY-MM-DD)")
    ap.add_argument("--kinds", default=None, help="external: comma list of jev, claude (default both)")
    a = ap.parse_args(argv)
    run = Path(a.run)
    check_phase(a.step, run)
    models = Path(a.models) if a.models else run / "models"
    rankers = tuple(a.rankers.split(",")) if a.rankers else None
    from uol.final import new_cases_fit as NF
    rk = rankers or NF.RANKERS
    if a.step == "fit":
        res = NF.fit(models, Path(a.work) if a.work else WORK / "fit", rk)
        print(json.dumps({s: {k: r[k] for k in ("config", "train_pairs", "feature_columns", "fit_seconds")}
                          for s, r in res["rankers"].items()}, indent=1))
        return 0
    if a.step == "dryrun-rankers":
        res = NF.dryrun(models, Path(a.work) if a.work else WORK / "fit", run / "dryrun", rk)
        return 0 if res["pass"] else 2
    if a.step == "dryrun-lstar":
        res = NF.lstar_dryrun(Path(a.work) if a.work else WORK / "fit", run / "dryrun")
        return 0 if res["pass"] else 2
    if a.step == "spread":
        NF.spread(Path(a.work) if a.work else WORK / "spread", run / "dryrun", rk)
        return 0
    if a.step == "report":
        from uol.final import new_cases_report as NR
        res = NR.build(run)
        if a.out == "append-evidence":
            res["evidence_appended"] = NR.append_evidence(run)
        print(json.dumps(res, indent=1))
        return 0
    from uol.final import new_cases_steps as NS
    return NS.dispatch(a, run, models)


if __name__ == "__main__":
    sys.exit(main())
