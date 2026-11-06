"""E13 steps after the ranker dry run (``new_cases_fit``): the source dry run and smoke run, the freeze record, the
release, the run computer's bundle and its intake, the new rows' sources, scoring once, and the evaluation.

Every step writes into its own folder and refuses to run again once its manifest exists (score once; a crashed step
leaves no manifest and may be repeated, the crash logged in N.md). No original test-set row is used anywhere.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import new_cases_fit as NF
from uol.final import new_cases_sources as NSrc
from uol.final.new_cases_fit import ROOT, log, sha256, write_json

SYSTEMS = ("lstar_headings",) + NF.RANKERS          # scored by ``score``; P6, Claude and Jev by ``new_cases_llm``
ANSWER = {"lstar_headings": "P1", "S0": "P3", "S_full": "S_full", "S_full_ext": "S_full_ext"}
NOT_SCORED = {
    "P2": "not scored on NEW (D162 scope: one line in RESULTS.md)",
    "P4": "not scored on NEW (D162 scope)", "P5+CE": "not scored on NEW (D162 scope)",
    "P3+L": "not scored on NEW (D162 scope)"}
P6 = {"system": "llm_rerank_S_full_ext", "base": "S_full_ext", "samples": 3, "passes": ["fwd", "rev"], "top": 30,
      "w_all": 0.75, "w_source": "record receipt llm_rerank_S_full_ext__DEV.json inputs.llm_meta.w_all"}
CLAUDE_W_ALL = 3.0                                  # record receipt claude_rerank_S_full_ext__DEV-A300.json w_all
# Pre-registered before collection: paired nDCG@10 contrasts over NEW components, Bonferroni within each family
# (the LLM family is the five comparisons of the test-set LLM claims). Nothing else is tested on NEW.
CONTRASTS = {
    "rankers": {"measure": "ndcg_at_10", "pairs": [["S0", "lstar_headings"], ["S_full", "S0"], ["S_full_ext", "S0"],
                                                   ["S_full_ext", "S_full"]]},
    "llm": {"measure": "ndcg_at_10", "pairs": [["llm_rerank_S_full_ext", "S_full_ext"],
                                               ["claude_rerank_S_full_ext", "S_full_ext"],
                                               ["claude_rerank_S_full_ext", "llm_rerank_S_full_ext"],
                                               ["jev_S_full_ext", "S_full_ext"], ["jev_rrf_S_full_ext", "S_full_ext"]],
            "subsets": ["all NEW", "decided after the LLM's stated cutoff (every precedent date in "
                        "source_precedentes later than the cutoff)", "the rest (a precedent on or before the cutoff, "
                        "or no parsable date)"]}}
# Which labels each part reads (T2_GATE_CAUSE_C.md §1: the record ranked frozen v11-label source files with rankers
# fitted on the corrected labels; the bundle profile corrects 3 development cases, the local profile never does).
LABEL_VERSIONS = {
    "first_stage_sources_new_rows": "v11 (local profile on the extended release; dev training citations as in the "
                                    "record's frozen files ep_run/scores, which were built on v11)",
    "first_stage_sources_dev_rows_for_fit": "v11 (the record's frozen files, read as they are)",
    "rankers_targets": "corrected (bundle profile, UOL_FINAL_LABELS=corrected, as the record's rankers)",
    "instrument_prior": "corrected (computed in the bundle profile at fit and at scoring; scoring checks it equals "
                        "models/instrument_prior_lockbox.npy)",
    "m1_memory": "label-free with respect to the release (round 3 memory.jsonl citations of older tesis)",
    "topic_columns": "label-free (unsupervised topic models)",
    "evaluation_new_rows": "the new cases' own labels from the frozen resolver (no correction touches them)",
    "dry_runs": "dryrun-sources and the smoke run in the local profile (as the NEW sources step); dryrun-rankers and "
                "dryrun-lstar in the bundle profile (as the scoring step)"}
# Every NEW case is published after both cutoffs; the court's decisions (source_precedentes) can be older, so the split
# that separates possibly-read cases uses the earliest precedent date. A month-only cutoff is read as the month's end.
LLM_CUTOFFS = {
    "llm_rerank_S_full_ext": {"model": "gpt-5.6-luna", "cutoff": "2026-02-16",
                              "source": "uol.round3.luna.MODEL_KNOWLEDGE_CUTOFF (developers.openai.com, read 2026-10-13)"},
    "claude_rerank_S_full_ext": {"model": "claude-opus-5-5", "cutoff": "2026-06-30",
                                 "source": "platform.claude.com/docs/en/about-claude/models/overview: training data "
                                           "cutoff 'Jun 2026' (read 2026-10-26)"},
    "jev_S_full_ext": {"model": "Jev (closed)", "cutoff": None, "source": "not stated by the provider"},
    "jev_rrf_S_full_ext": {"model": "Jev (closed)", "cutoff": None, "source": "not stated by the provider"}}
WINDOW_START = "2026-09-12"
N_RUNB_CONTROLS = 50
CODE_TREES = ("final", "round2", "round3", "embedding_push", "facts_to_law")
DRYRUN_FILES = ("dryrun_rankers.json", "dryrun_lstar_headings.json", "dryrun_sources.json", "dryrun_smoke.json",
                "spread_fold0.json")


def smoke_dir() -> Path:
    from uol.final import new_cases
    return new_cases.WORK / "smoke"


def is_smoke(run: Path) -> bool:
    return Path(run).resolve() == smoke_dir().resolve()


def _py(*args: str, env: dict | None = None) -> int:
    e = {**os.environ, "PYTHONPATH": str(ROOT / "src"), **(env or {})}
    return subprocess.run([sys.executable, "-m", "uol.final.new_cases", *args], cwd=str(ROOT), env=e).returncode


def _runb_gate() -> dict[str, float]:
    from uol.final.gpu_jobs import new_encode
    return dict(new_encode.GATE)


def code_hashes() -> dict[str, str]:
    out = {}
    for tree in CODE_TREES:
        for p in sorted((ROOT / "src" / "uol" / tree).rglob("*.py")):
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256(p)
    return out


# --------------------------------------------------------------------------------------------- dry run: smoke


def smoke(models: Path) -> dict[str, Any]:
    """Every post-freeze step on three stand-in new cases (``new_cases_sources.smoke_prepare``), each in its own
    process as in the real run: encode-local, sources, score. Passes when each step ends with exit 0 and every
    system gives a finite top 10 for every stand-in case."""
    work = smoke_dir()
    if work.exists():
        raise FileExistsError(f"{work} exists; the smoke run is made once (archive it to repeat)")
    out = NF.ROOT / "experiments" / "runs" / "new_cases_20261026" / "dryrun"
    t0 = time.perf_counter()
    res: dict[str, Any] = {"work": str(work), "steps": {}, "fails": []}
    res["prepare"] = NSrc.smoke_prepare(work)
    for step in ("encode-local", "sources", "score"):
        env = ({k: str(NF.THREADS) for k in ("UOL_FINAL_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                               "OPENBLAS_NUM_THREADS")} if step == "score" else None)
        code = _py(step, "--run", str(work), "--models", str(models), env=env)
        res["steps"][step] = code
        if code != 0:
            res["fails"].append(f"{step}: exit {code}")
            break
    if not res["fails"]:
        for sid in SYSTEMS:
            m = json.loads((work / "scores" / sid / "manifest.json").read_text(encoding="utf-8"))
            S = np.load(work / "scores" / sid / "scores.npy")
            top = np.argsort(-np.nan_to_num(S, nan=-np.inf), axis=1, kind="stable")[:, :10]
            ok = bool(np.isfinite(np.take_along_axis(S, top, 1)).all())
            res.setdefault("systems", {})[sid] = {"rows": m["n"], "finite_top10": ok}
            if not ok:
                res["fails"].append(f"{sid}: non-finite top 10")
    res["pass"] = not res["fails"]
    res["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(out / "dryrun_smoke.json", res)
    log(f"smoke run: {json.dumps({k: v for k, v in res.items() if k != 'prepare'})}", out / "dryrun.log")
    return res


# --------------------------------------------------------------------------------------------- the freeze


def freeze(run: Path, models: Path) -> dict[str, Any]:
    """FREEZE.json when every exact dry-run check passed (criterion A, D162); otherwise nothing is written."""
    from uol.final import new_cases
    from uol.round3 import l1_luna, luna
    from uol.round2 import c_collect
    dry = Path(run) / "dryrun"
    results = {}
    fails = []
    for name in DRYRUN_FILES:
        p = dry / name
        if not p.exists():
            fails.append(f"{name} missing")
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        results[name] = {"sha256": sha256(p), "pass": r.get("pass", True), "fails": r.get("fails", [])}
        if name != "spread_fold0.json" and not r.get("pass"):
            fails.append(f"{name}: {r.get('fails')}")
    from uol.final import bundle_overlay
    ov = json.loads(bundle_overlay.OVERLAY.read_text(encoding="utf-8")) if bundle_overlay.OVERLAY.exists() else None
    if not ov or not ov.get("pass"):
        fails.append("bundle overlay not verified (OVERLAY.json missing or failed)")
    if fails:
        log(f"freeze refused: {fails}", dry / "dryrun.log")
        return {"frozen": False, "fails": fails}
    man = json.loads((Path(models) / "MODELS.json").read_text(encoding="utf-8"))
    spread = json.loads((dry / "spread_fold0.json").read_text(encoding="utf-8"))
    rec = {
        "what": "E13 freeze: the answer systems scored once on the new cases (NEW) with models trained on every "
                "development case; TEST-R dropped (D148)",
        "decisions": ["D97", "D101", "D148", "D161", "D162 (criterion A: exact dry run; the author: 'Run it actually do the "
                      "work. Make it good.')"],
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "systems": {
            "P1": {"system": "lstar_headings", "text": "V2 (article headings)",
                   "selection": lstar_selection()[1],
                   "selection_file": str(lstar_selection()[0].relative_to(NF.ROOT)),
                   "selection_file_sha256": sha256(lstar_selection()[0]),
                   "selection_rule": "frozen all-development lexical selection ('lockbox' = the most frequent fold "
                                     "selection; local record file, its fold selections equal the bundle's)"},
            **{ANSWER[s] if ANSWER[s] != s else s: {"system": s, "models": {k: man["rankers"][s][k] for k in (
                "config", "seed", "pair_seed", "train_pairs", "frozen_lockbox_train_pairs", "feature_columns", "files")},
                "group": "all development rows (training folds 0-4)", "topic_sizes": NF.TOPIC_SIZES,
                **({"m1": NF.M1_CFG} if s == "S_full_ext" else {})} for s in NF.RANKERS},
            "P6": {**P6, "llm": luna.MODEL, "llm_knowledge_cutoff": luna.MODEL_KNOWLEDGE_CUTOFF,
                   "effort": l1_luna.EFFORT, "prompt_sha256": sha256(l1_luna.PROMPT),
                   "fusion": "round3.features.fuse_rows z(S_full_ext ranks) + w z(1 / mean position), w = w_all"},
            "claude": {"system": "claude_rerank_S_full_ext", "w_all": CLAUDE_W_ALL, "cap_usd": 60.0,
                       "from": "uol.final.external CLAUDE (model, effort, max_tokens) and the P6 prompt and windows"},
            "jev": {"systems": ["jev_S_full_ext", "jev_rrf_S_full_ext"], "cap_usd": 10.0, "depth": 30, "rrf_k": 60}},
        "not_scored": NOT_SCORED,
        "models": {"MODELS.json": sha256(Path(models) / "MODELS.json"),
                   "files": {f"{s}/{n}": h for s in NF.RANKERS for n, h in man["rankers"][s]["files"].items()}},
        "bundle": man.get("bundle"), "m1_stage": man.get("m1_stage"),
        "record": {"path": str(NF.RECORD), "receipts": {p.name: sha256(p) for p in sorted((NF.RECORD / "receipts").glob("*__DEV*.json"))
                                                      if p.name.split("__")[0] in SYSTEMS + (P6["system"], "claude_rerank_S_full_ext", "jev_S_full_ext", "jev_rrf_S_full_ext")}},
        "frozen_tower_weights": {"ft_e5_q": sha256(NSrc.FT_STATE),
                                 **{f"casesim_B_epoch{e}": sha256(p) for e, p in NSrc.CS_STATE.items()}},
        "dry_run": results,
        "spread_fold0": {s: {k: v[k] for k in ("refit_ndcg_at_10", "record_ndcg_at_10", "difference", "top10_identical_share")}
                         for s, v in spread.items()},
        "collection": {"collector": "uol.round2.c_collect (unchanged code)", "c_collect.py": sha256(Path(c_collect.__file__)),
                       "window": [WINDOW_START, "the collection day"], "store": "a copy of round 2's raw store in "
                       "<run>/collect (round 2's files are not written)"},
        "evaluation": {"population": "P_spec (component macro)", "measures": ["ndcg_at_10", "success_at_10", "law_at_10"],
                       "subsets": ["all NEW", "second look (not in round 2's first look)", "no v11 overlap"],
                       "llm_cutoff_split": {"rule": "earliest precedent decision date (source_precedentes, Spanish "
                                                    "dates) after the model's stated cutoff vs the rest",
                                            "cutoffs": LLM_CUTOFFS},
                       "contrasts": "round3.stats.contrast over components (bootstrap 95 % CI, smallest detectable "
                                    "difference)", "contrast_families": CONTRASTS, "score_once": True},
        "label_versions": LABEL_VERSIONS,
        "bundle_overlay": {"what": "the record's development bundle read from its verified local originals (no unpack): "
                                   "per packed array, sha256 of the served bytes (whole file or the 4,812 development "
                                   "rows in release order) equals the bundle MANIFEST's unpacked_sha256",
                           "OVERLAY.json": sha256(bundle_overlay.OVERLAY), "bundle_manifest_sha256": ov["bundle_manifest_sha256"],
                           "arrays": {k: {"mode": a["mode"], "sha256": a["sha256"], "match": a["match"]}
                                      for k, a in ov["arrays"].items()}},
        "run_computer": {"job": "gpu_jobs/new_encode.py", "gate": _runb_gate(), "controls": N_RUNB_CONTROLS,
                         "tolerance_reason": "set before any NEW score; RUN-B is another machine (timing C, 26 Oct: "
                                             "~1e-5 drift moves rank-boundary ties), so no exact top-k set is required "
                                             "there; every source and ranker is computed on this PC"},
        "code": code_hashes(),
    }
    path = new_cases.freeze_path()
    if path.exists():
        raise FileExistsError(f"{path} exists")
    write_json(path, rec)
    log(f"FREEZE.json written: {sha256(path)}", dry / "dryrun.log")
    return {"frozen": True, "path": str(path), "sha256": sha256(path)}


# --------------------------------------------------------------------------------------------- collection, release


def collect_store(run: Path) -> Path:
    """``<run>/collect``: round 2's raw store copied once (so its fetched ids are not fetched again and its files stay
    as they are); the collector's module paths point here for this process."""
    from uol.round2 import c_collect
    d = Path(run) / "collect"
    src = c_collect.DIR
    if not d.exists():
        (d / "raw").mkdir(parents=True)
        copied = {}
        for name in ("raw/id_pages.jsonl", "raw/responses.jsonl", "collect_log.jsonl"):
            shutil.copy2(src / name, d / name)
            copied[name] = {"from": str(src / name), "sha256": sha256(d / name),
                            "lines": sum(1 for _ in (d / name).open("rb"))}
        write_json(d / "copied_from_round2.json", copied)
    c_collect.DIR, c_collect.RAW = d, d / "raw"
    return d


def collect(run: Path) -> dict[str, Any]:
    from uol.round2 import c_collect
    d = collect_store(run)
    t0 = time.time()
    res = c_collect.collect()
    write_json(d / f"collect_result_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json",
               {"result": res, "seconds": round(time.time() - t0, 1),
                "lines_now": {n: sum(1 for _ in (d / n).open("rb"))
                              for n in ("raw/id_pages.jsonl", "raw/responses.jsonl", "collect_log.jsonl")}})
    return res


def build_sources(run: Path, cutoff: str) -> dict[str, Any]:
    from uol.round2 import c_collect
    collect_store(run)
    return c_collect.build(WINDOW_START, cutoff)


def release(run: Path, sources_path: Path) -> dict[str, Any]:
    """The frozen resolver and masker (``round2.c_release.build_set``) into ``<run>/NEW/release``, the extended
    release, and the counts."""
    from uol.round2 import c_release
    run = Path(run)
    c_release.RUN = run
    man = c_release.build_set("NEW", Path(sources_path))
    rel = run / "NEW" / "release"
    qs = NSrc._load(rel / "queries.jsonl")
    refs = NSrc._load(rel / "references.jsonl")
    meta = {m["query_id"]: m for m in NSrc._load(rel / "rows_meta.jsonl")}
    ext = NSrc.extend(qs, refs, {q: m["publication_date"] for q, m in meta.items()}, run / "release_ext", "NEW",
                      {"new_release": str(rel), "new_release_manifest_sha256": sha256(rel / "manifest.json")})
    c = NSrc.counts(rel)
    out = {"release_manifest_sha256": sha256(rel / "manifest.json"), "files": man["files"], "ext": ext, "counts": c,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(run / "NEW" / "counts.json", out)
    return out


# --------------------------------------------------------------------------------------------- run computer


def runb_controls() -> list[str]:
    """50 development cases, evenly spaced in release order (fixed before the run computer encodes anything)."""
    from uol.embedding_push import common as ep
    from uol.final import context, guard
    context.activate_local("v11")
    dev = guard.dev_rows_only(ep.dev_rows(), "run computer controls")
    pick = dev[np.linspace(0, len(dev) - 1, N_RUNB_CONTROLS).round().astype(int)]
    d = ep.release()
    return [d.queries[int(r)].query_id for r in pick]


def bundle_new(run: Path, out: Path) -> dict[str, Any]:
    """The run computer's input folder: the new texts, the 50 controls, the job and its runner, MANIFEST.json."""
    run, out = Path(run), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; the bundle is built once into a new folder")
    stage = NF.ROOT / ".cache" / "new_cases_20261026" / "bundle_stage"
    if stage.exists():
        raise FileExistsError(f"{stage} exists")
    (stage / "data").mkdir(parents=True)
    (stage / "src" / "uol" / "final" / "gpu_jobs").mkdir(parents=True)
    ext = run / "release_ext"
    nr = NSrc.new_rows(ext)
    texts = {q["query_id"]: q["text"] for q in NSrc._load(ext / "queries.jsonl")[NSrc.N_V11:]}
    rows = [{"query_id": r["query_id"], "text": texts[r["query_id"]],
             "text_sha256": hashlib.sha256(texts[r["query_id"]].encode("utf-8")).hexdigest()} for r in nr]
    NSrc._write_jsonl(stage / "data" / "new_queries.jsonl", rows)
    write_json(stage / "data" / "controls.json", {"query_ids": runb_controls(),
                                                  "rule": "50 development rows evenly spaced in release order"})
    here = Path(__file__).resolve().parent / "gpu_jobs"
    shutil.copy2(here / "new_encode.py", stage / "src" / "uol" / "final" / "gpu_jobs" / "new_encode.py")
    shutil.copy2(here / "new_runner.py", stage / "run_new.py")
    files = {str(p.relative_to(stage)).replace("\\", "/"): sha256(p) for p in sorted(stage.rglob("*")) if p.is_file()}
    write_json(stage / "MANIFEST.json", {"what": "E13 new cases: run computer query side (bge-m3, Qwen3-8B int8, "
                                                 "ColBERT pairs) with 50 development controls",
                                         "new_cases": len(rows), "files": files,
                                         "freeze_sha256": sha256(Path(__import__("uol.final.new_cases", fromlist=["x"]).freeze_path())),
                                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    shutil.copytree(stage, out)
    back = {str(p.relative_to(out)).replace("\\", "/"): sha256(p) for p in sorted(out.rglob("*")) if p.is_file()}
    staged = {**files, "MANIFEST.json": sha256(stage / "MANIFEST.json")}
    if back != staged:
        raise RuntimeError("the copy on Drive differs from the staged folder")
    res = {"out": str(out), "files": len(back), "manifest_sha256": staged["MANIFEST.json"], "new_cases": len(rows),
           "bytes": sum(p.stat().st_size for p in out.rglob("*") if p.is_file())}
    write_json(run / "bundle_new.json", res)
    return res


def ingest(run: Path, runb: Path) -> dict[str, Any]:
    """The run computer's outputs: RESULTS_MANIFEST sha256 checked, the gate passed, vectors and pairs mapped to the
    extended release rows (``vectors/new_bge.npy``, ``new_Qc.npy``, ``new_colbert_pairs.npz``, ``runb_manifest.json``)."""
    run, runb = Path(run), Path(runb)
    vec = run / "vectors"
    if (vec / "runb_manifest.json").exists():
        raise FileExistsError("the run computer's outputs were taken in already")
    rm = json.loads((runb / "RESULTS_MANIFEST.json").read_text(encoding="utf-8"))
    bad = [n for n, v in rm["files"].items() if v["to_drive"] and sha256(runb / n) != v["sha256"]]
    if bad:
        raise RuntimeError(f"run computer outputs differ from RESULTS_MANIFEST: {bad}")
    gate = json.loads((runb / "GATE.json").read_text(encoding="utf-8"))
    if not gate["pass"]:
        raise RuntimeError(f"the run computer's control gate failed: {gate['fails']}")
    nr = NSrc.new_rows(run / "release_ext")
    ids = [r["query_id"] for r in nr]
    n = len(ids)
    bge, qw, pr = (np.load(runb / f) for f in ("bge_cls.npz", "qwen.npz", "pairs.npz"))
    for name, z in (("bge_cls", bge), ("qwen", qw), ("pairs", pr)):
        if list(z["query_ids"][:n]) != ids:
            raise RuntimeError(f"{name}: the new cases are not in the release's order")
    vec.mkdir(parents=True, exist_ok=True)
    np.save(vec / "new_bge.npy", np.asarray(bge["cls"][:n], dtype=np.float32))
    np.save(vec / "new_Qc.npy", np.asarray(qw["vectors"][:n], dtype=np.float32))
    case = np.asarray(pr["case"])
    keep = case < n
    rows = np.asarray([r["row"] for r in nr], dtype=np.int32)
    np.savez(vec / "new_colbert_pairs.npz", qidx=rows[case[keep]], cidx=np.asarray(pr["cidx"])[keep],
             score=np.asarray(pr["score"])[keep])
    man = {"rows": [int(r) for r in rows], "from": str(runb),
           "results_manifest_sha256": sha256(runb / "RESULTS_MANIFEST.json"), "gate": gate,
           "files": {f: sha256(vec / f) for f in ("new_bge.npy", "new_Qc.npy", "new_colbert_pairs.npz")},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for f in ("bge.json", "qwen.json", "pairs.json", "GATE.json"):
        man[f] = json.loads((runb / f).read_text(encoding="utf-8"))
    write_json(vec / "runb_manifest.json", man)
    return man


# --------------------------------------------------------------------------------------------- scoring once


def _sources(run: Path) -> tuple[dict, dict[str, np.ndarray], dict[int, dict]]:
    sm = json.loads((Path(run) / "sources" / "manifest.json").read_text(encoding="utf-8"))
    for n, h in sm["files"].items():
        if sha256(Path(run) / "sources" / n) != h:
            raise RuntimeError(f"sources/{n}: sha256 differs from its manifest")
    blocks = {n[:-4]: np.load(Path(run) / "sources" / n, mmap_mode="r") for n in sm["files"] if n.endswith(".npy")}
    z = np.load(Path(run) / "sources" / "colbert_bge_pairs.npz")
    pairs = {int(r): {"cidx": z["cidx"][z["qidx"] == r], "score": z["score"][z["qidx"] == r]} for r in sm["rows"]}
    return sm, blocks, pairs


def link_fit_features(work: Path) -> dict[str, Any]:
    """The fit's development feature table (``d58/X_control.npy``, 4.5 GB, same bundle profile and labels) and its
    top-200 caches, hard-linked read-only into the scoring work folder instead of rebuilt there: no bytes written, and
    ``d58_refit`` reuses a table only when its ``.done`` pair count equals the pool's. Read-only, so a differing pool
    fails instead of rewriting the fit's file."""
    import stat
    from uol.final import new_cases
    src, dst = new_cases.WORK / "fit" / "d58", Path(work) / "d58"
    names = ["X_control.npy", "X_control.done"] + [f"cache/{p.name}" for p in sorted((src / "cache").glob("*.npy"))]
    linked = []
    for n in names:
        if (dst / n).exists():
            continue
        (dst / n).parent.mkdir(parents=True, exist_ok=True)
        os.link(src / n, dst / n)
        os.chmod(dst / n, stat.S_IREAD)
        linked.append(n)
    return {"from": str(src), "linked": linked, "done": json.loads((dst / "X_control.done").read_text())}


def lstar_selection() -> tuple[Path, dict[str, Any]]:
    """P1's all-development lexical selection (``lockbox``). The development bundle holds only the five fold
    selections, so it is read from the local record (v11 ``ep_run``), whose fold selections must equal the bundle's."""
    from uol.embedding_push import common as ep
    from uol.final import context
    local = NF.ROOT / context.RELEASES["v11"]["ep_run"] / "base" / "selection" / "lexical.json"
    a = json.loads(local.read_text(encoding="utf-8"))
    b = json.loads((ep.RUN / "base" / "selection" / "lexical.json").read_text(encoding="utf-8"))
    if Path(ep.RUN).resolve() != local.parents[2].resolve() and any(a.get(k) != v for k, v in b.items()):
        raise RuntimeError("the local lexical selections differ from the active profile's")
    return local, a["lockbox"]["selected"]


def score(run: Path, models: Path, rankers: Sequence[str] = NF.RANKERS) -> dict[str, Any]:
    """P1 and the three saved rankers on every new row, through the per-case path (bundle profile, 4 threads):
    ``scores/<system>/scores.npy`` [n_new, N] in ``new_rows.json`` order, with a manifest. Each system once."""
    import pickle
    from uol.embedding_push import base, common as ep
    from uol.final.adapters.lexical import text_ctx
    run, models = Path(run), Path(models)
    out = run / "scores"
    todo = [s for s in ("lstar_headings",) + tuple(rankers) if not (out / s / "manifest.json").exists()]
    done = [s for s in ("lstar_headings",) + tuple(rankers) if s not in todo]
    if done:
        log(f"score: {done} already scored; not repeated", run / "logs" / "score.log")
    if not todo:
        return {"scored": [], "already": done}
    work = NF.ROOT / ".cache" / "new_cases_20261026" / ("score_smoke" if is_smoke(run) else "score")
    log(f"score: fit feature table {json.dumps(link_fit_features(work))}", run / "logs" / "score.log")
    NF.activate(work)
    d = ep.release()
    sm, blocks, pairs = _sources(run)
    if sm["candidates"] != len(d.candidate_ids) or sm["candidate_ids_sha256"] != hashlib.sha256(
            "\n".join(d.candidate_ids).encode()).hexdigest():
        raise RuntimeError("the sources' catalogue differs from the bundle's")
    nr = NSrc.new_rows(run / "release_ext")
    texts = {q["query_id"]: q["text"] for q in NSrc._load(run / "release_ext" / "queries.jsonl")[NSrc.N_V11:]}
    if [r["row"] for r in nr] != sm["rows"]:
        raise RuntimeError("the sources are not in new_rows.json order")
    qc = np.load(run / "vectors" / "new_Qc.npy")
    res: dict[str, Any] = {"scored": [], "already": done}
    pop = NF.dev_population()
    for sid in todo:
        t0 = time.perf_counter()
        S = np.full((len(nr), len(d.candidate_ids)), np.nan, dtype=np.float32)
        meta: dict[str, Any] = {}
        if sid == "lstar_headings":
            sel_file, sel = lstar_selection()
            S[:] = np.asarray(base.lstar_block(text_ctx("V2"), base.LSel.from_json(sel), [texts[r["query_id"]] for r in nr]),
                              dtype=np.float32)
            meta = {"selection": sel, "text": "V2", "selection_file": str(sel_file.relative_to(NF.ROOT)),
                    "selection_file_sha256": sha256(sel_file)}
        else:
            info = json.loads((models / sid / "info.json").read_text(encoding="utf-8"))
            for n, h in info["files"].items():
                if sha256(models / sid / n) != h:
                    raise RuntimeError(f"{sid}/{n}: sha256 differs from info.json")
            with (models / sid / "model.pkl").open("rb") as f:
                model = pickle.load(f)
            sc = NF.prepared(sid, pop)
            m1 = NF.m1_case() if sc.extension == "m1" else None
            rk = NF.NewCaseRanker(sc, model, NF.load_topic_models(models, info), m1)
            table_ok = np.array_equal(np.asarray(rk.table[NF.LOCKBOX], dtype=np.float32),
                                      np.load(models / "instrument_prior_lockbox.npy"))
            if not table_ok:
                raise RuntimeError(f"{sid}: the all-development instrument prior differs from the fit's")
            for i, r in enumerate(nr):
                src = {s: (pairs[r["row"]] if s == "colbert_bge" else np.asarray(blocks[s][i])) for s in rk.sources}
                src.setdefault("L_star", np.asarray(blocks["L_star"][i]))
                text = texts[r["query_id"]]
                m1c = None
                if m1 is not None:
                    pc = rk.pool(src)
                    q = qc[i] / max(float(np.linalg.norm(qc[i])), 1e-12)
                    m1c = rk.m1_columns(q, r["publication_date"], pc, 10_000_000 + i)
                S[i] = rk.rank(text, src, NF.LOCKBOX, m1_cols=m1c)["row"]
            meta = {"model_sha256": info["files"]["model.pkl"], "group": NF.LOCKBOX, "extension": sc.extension}
            sc.h._base = {}
        (out / sid).mkdir(parents=True, exist_ok=True)
        np.save(out / sid / "scores.npy", S)
        man = {"system": sid, "n": len(nr), "query_ids": [r["query_id"] for r in nr], **meta,
               "scores_sha256": sha256(out / sid / "scores.npy"), "seconds": round(time.perf_counter() - t0, 1),
               "ms_per_case": round(1000 * (time.perf_counter() - t0) / max(len(nr), 1), 1),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(out / sid / "manifest.json", man)
        res["scored"].append(sid)
        log(f"score: {sid} on {len(nr)} new rows ({man['seconds']}s)", run / "logs" / "score.log")
    return res


# --------------------------------------------------------------------------------------------- evaluation


class closed_test_guard:
    """The evaluator's lockbox guard narrowed to v11's lockbox (as ``new_cases_llm.p6`` narrows round 3's): the new
    rows carry role ``lockbox`` in the extended release only so that training uses every development row; the closed
    test set keeps being refused, and only rows appended after v11 may be evaluated."""

    def __init__(self, ep, rows: Sequence[int]):
        rows = np.asarray(rows, dtype=np.int64)
        if (rows < NSrc.N_V11).any():
            raise PermissionError("E13 evaluates only the new rows (row >= N_V11)")
        self.ep, self.orig = ep, ep.lockbox_rows

    def __enter__(self):
        closed = np.asarray([r for r in self.orig() if int(r) < NSrc.N_V11], dtype=np.int64)
        self.ep.lockbox_rows = lambda: closed
        return self

    def __exit__(self, *exc):
        self.ep.lockbox_rows = self.orig
        return False


def evaluate(run: Path, systems: Sequence[str] | None = None) -> dict[str, Any]:
    """The project evaluator on the new rows (extended release, P_spec, specific view): ``eval/<system>/
    per_case.parquet``, ``top200.parquet``, ``summary.json`` (all, second look, no v11 overlap). Once per system."""
    import pandas as pd
    from uol.embedding_push import common as ep
    from uol.final import measures
    run = Path(run)
    if is_smoke(run):
        raise PermissionError("the smoke run has no labels to evaluate")
    NSrc.activate_ext(run / "release_ext", run / "logs" / "embedding_push")
    d = ep.release()
    nr = NSrc.new_rows(run / "release_ext")
    rows = np.asarray([r["row"] for r in nr], dtype=np.int64)
    meta = {m["query_id"]: m for m in NSrc._load(run / "NEW" / "release" / "rows_meta.jsonl")}
    first = NSrc.first_look_ids()
    unseen = measures.unseen_mask(ep.dev_rows())
    out: dict[str, Any] = {}
    have = systems or [p.name for p in sorted((run / "scores").iterdir()) if (p / "manifest.json").exists()]
    for sid in have:
        dst = run / "eval" / sid
        if (dst / "summary.json").exists():
            out[sid] = json.loads((dst / "summary.json").read_text(encoding="utf-8"))
            continue
        man = json.loads((run / "scores" / sid / "manifest.json").read_text(encoding="utf-8"))
        S = np.load(run / "scores" / sid / "scores.npy")
        if sha256(run / "scores" / sid / "scores.npy") != man["scores_sha256"] or man["query_ids"] != [r["query_id"] for r in nr]:
            raise RuntimeError(f"{sid}: scores differ from their manifest")
        with closed_test_guard(ep, rows):
            res = ep.evaluate(S, rows, "P_spec", top_k=measures.TOP_K)
        loc = {int(r): j for j, r in enumerate(rows)}
        sub = S[[loc[int(r)] for r in res["rows"]]]
        per = measures.per_case(res["per_query"], res["rows"], sub, specific=True, unseen=unseen)
        cases, tops = [], []
        for r, rec, q in zip(res["rows"], per, res["per_query"]):
            qq = d.queries[int(r)]
            m = meta[qq.query_id]
            cases.append({"query_id": qq.query_id, "row": int(r), "component_id": qq.component_id,
                          "publication_date": m["publication_date"], "first_look": qq.query_id in first,
                          "overlap_v11": bool(m.get("overlap_v11")), **rec})
            for rank, c in enumerate(q["topk"], 1):
                tops.append((qq.query_id, rank, int(c), float(S[loc[int(r)], int(c)])))
        frame = pd.DataFrame(cases)
        dst.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(dst / "per_case.parquet", index=False)
        pd.DataFrame(tops, columns=["query_id", "rank", "candidate_idx", "score"]).to_parquet(dst / "top200.parquet", index=False)
        summ = {"system": sid, "population": "NEW (P_spec)", "all": measures.summary(frame),
                "second_look": measures.summary(frame[~frame["first_look"]]) if (~frame["first_look"]).any() else None,
                "no_overlap": measures.summary(frame[~frame["overlap_v11"]]) if (~frame["overlap_v11"]).any() else None,
                "scores_sha256": man["scores_sha256"], "per_case_sha256": sha256(dst / "per_case.parquet"),
                "top200_sha256": sha256(dst / "top200.parquet"), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        write_json(dst / "summary.json", summ)
        out[sid] = summ
        log(f"evaluate: {sid} nDCG@10 {summ['all']['ndcg_at_10']:.4f} on {summ['all']['n_cases']} cases / "
            f"{summ['all']['n_groups']} groups", run / "logs" / "evaluate.log")
    return out


# --------------------------------------------------------------------------------------------- dispatch


def dispatch(a, run: Path, models: Path) -> int:
    step = a.step
    if step == "dryrun-sources":
        return 0 if NSrc.dryrun_sources(run / "dryrun")["pass"] else 2
    if step == "dryrun-smoke":
        return 0 if smoke(models)["pass"] else 2
    if step == "overlay-verify":
        from uol.final import bundle_overlay
        r = bundle_overlay.verify()
        print(json.dumps({k: r[k] for k in ("pass", "fails", "seconds", "lockbox_rows_disjoint")}, indent=1))
        return 0 if r["pass"] else 2
    if step == "freeze":
        r = freeze(run, models)
        print(json.dumps(r, indent=1))
        return 0 if r["frozen"] else 2
    if step == "collect":
        print(json.dumps(collect(run), indent=1))
        return 0
    if step == "build-sources":
        if not a.cutoff:
            raise SystemExit("build-sources needs --cutoff (the collection day)")
        print(json.dumps(build_sources(run, a.cutoff), indent=1))
        return 0
    if step == "release":
        print(json.dumps(release(run, Path(a.sources))["counts"], indent=1))
        return 0
    if step == "encode-local":
        NSrc.encode_local(run)
        return 0
    if step == "bundle-new":
        print(json.dumps(bundle_new(run, Path(a.out)), indent=1))
        return 0
    if step == "ingest":
        ingest(run, Path(a.runb))
        return 0
    if step == "sources":
        NSrc.sources(run)
        return 0
    if step == "score":
        score(run, models, tuple(a.rankers.split(",")) if a.rankers else NF.RANKERS)
        return 0
    if step == "evaluate":
        evaluate(run)
        return 0
    from uol.final import new_cases_llm as NL
    return NL.dispatch(a, run, models)
