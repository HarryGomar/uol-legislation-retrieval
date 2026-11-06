"""D181 re-runs (prepared 28 Oct 2026, RUNS session; nothing here has been launched): shared constants and guards.

Read with ``research/reports/thesis_audit_20261028/RUN_PLAN.md``. The scripts ``audit_d181_dev`` (R1, R3 on the
development cases) and ``audit_d181_new`` (R2, R3 on the new cases) reuse the record's code unchanged
(``harness.run``, ``adapters.ltr`` with ``drop_sources``, ``new_cases_fit``/``new_cases_steps``/``new_cases_llm``,
``llm_jobs``, ``external``, ``round3.stats``). Nothing in this module trains, scores or calls a model.

Guards:
* every compute step is a dry run (input check and the plan printed) unless the command has ``--go`` **and** the
  environment variable ``UOL_AUDIT_D181_APPROVE`` names the step's approval token (set by the author; see RUN_PLAN.md);
* outputs go to a new record folder (``RUN``) and a new work folder (``WORK``); the registered records
  (``final_20261025``, ``new_cases_20261026``, ``timing_20261026d``, ``llm_sweep_20261028``) are only read;
* the new systems are added through a copy of ``systems.json`` in the run folder (``UOL_FINAL_SYSTEMS``), never by
  editing the registry.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(os.environ.get("UOL_AUDIT_D181_RUN", ROOT / "experiments" / "runs" / "audit_d181_20261028"))
WORK = Path(os.environ.get("UOL_AUDIT_D181_WORK", ROOT / ".cache" / "audit_d181"))

RECORD = ROOT / "experiments" / "runs" / "final_20261025"
STORE = RECORD / "store_dev_20261025b"
BOOT100K = RECORD / "supplement" / "bootstrap_100k" / "contrasts"
NEW_RUN = ROOT / "experiments" / "runs" / "new_cases_20261026"
NEW_WORK = ROOT / ".cache" / "new_cases_20261026"
SPREAD = NEW_WORK / "spread"                  # this PC's fold-0 refit of 26 Oct (spread_fold0.json): the exact control
BUNDLE_DATA = Path(os.environ.get("UOL_NEW_BUNDLE_DATA", r"bundle\UOL_FINAL_RUN\data"))
SYSTEMS_JSON = ROOT / "src" / "uol" / "final" / "systems.json"
ENV_FILE = ROOT / ".env"
CLAUDE_VENV = ROOT / ".cache" / "venvs" / "c10" / "Scripts" / "python.exe"
API_VENV = ROOT / ".cache" / "venvs" / "openai" / "Scripts" / "python.exe"

# D185 (28 Oct): the proposed system is the lean ranker S0+qwen3_8b (already in systems.json); UOL_AUDIT_D181_SYS overrides.
NEW_SYS = os.environ.get("UOL_AUDIT_D181_SYS", "S0+qwen3_8b")
LLM_SYS = f"llm_rerank_{NEW_SYS}"
CLAUDE_SYS = f"claude_rerank_{NEW_SYS}"
THREADS = 4                                    # the record's fits (receipts: OMP_NUM_THREADS = UOL_FINAL_THREADS = 4)
DRAWS_DEV = 100_000                            # D138
SEED, MARGIN = 20261004, 0.02                  # the record's contrast files

TOKENS = {"R1": "R1 retrain on the five development folds (CPU, ~3.5 h)",
          "R1-S0": "R1 optional: S0 rerun on this PC (CPU, ~1 h)",
          "R2": "R2 all-development refit and new-case scoring (CPU, ~1 h); scores NEW after the E13 freeze",
          "R3-gpt-dev": "R3 GPT-5.6 LLM stage on 1,217 development cases (Codex CLI, ~3,300 calls)",
          "R3-gpt-new": "R3 GPT-5.6 LLM stage on the 46 new cases (Codex CLI, ~150 calls)",
          "R3-claude-dev": "R3 Claude Opus 5.5 on DEV-A300 (Anthropic API, ~US$16)",
          "R3-claude-dev-full": "D188 Claude Opus 5.5 on the other 917 development cases (Anthropic API, ~US$46)",
          "R3-claude-live": "D188 live timing of the reread (Anthropic API, at most US$0.50)",
          "R3-claude-new": "R3 Claude Opus 5.5 on the new cases (Anthropic API, ~US$4)"}


def new_specs() -> list[dict[str, Any]]:
    """The registry entries the new system needs: none when it is already in systems.json (D185: S0+qwen3_8b)."""
    return [] if NEW_SYS in _systems() else [spec_minus_colbert()]


def spec_minus_colbert() -> dict[str, Any]:
    """S_full's own entry with ``drop_sources: [colbert_bge]`` (the mechanism of S_full_ext_minus_colbert, D79)."""
    s = copy.deepcopy(_systems()["S_full"])
    s["id"] = "S_full_minus_colbert"
    s["inputs"] = "S_full without the colbert_bge features (leave-one-out, D181)"
    s["adapter_args"]["drop_sources"] = ["colbert_bge"]
    s["notes"] = "D181 audit re-run: late interaction left out of the integrated method"
    s.pop("answer", None)
    return s


NEW_SYS_MINUS_M = "S_full_minus_colbert_minus_M"


def spec_minus_colbert_minus_m() -> dict[str, Any]:
    """The F8 (C2, D93) past-case-channel ablation of the new system: S_full_minus_M's drop list plus colbert_bge.
    (S_full_minus_D already drops colbert_bge with the rest of the direct channel, so it needs no new entry.)"""
    from uol.final import c2
    s = copy.deepcopy(_systems()["S_full_minus_M"])
    s["id"] = NEW_SYS_MINUS_M
    s["inputs"] = "S_full_minus_colbert without the past-case channel (C2 row of F8, D181)"
    s["adapter_args"]["drop_sources"] = list(dict.fromkeys(list(c2.M + c2.MIXED) + ["colbert_bge"]))
    s["notes"] = "D181 audit re-run: F8's memory ablation over the integrated method without late interaction"
    s.pop("answer", None)
    return s


def spec_llm(base: str, system: str, kind: str) -> dict[str, Any]:
    """The LLM-stage entry of the record (llm_rerank_S_full_ext / claude_rerank_S_full_ext) over another base."""
    src = "llm_rerank_S_full_ext" if kind == "gpt" else "claude_rerank_S_full_ext"
    s = copy.deepcopy(_systems()[src])
    s["id"] = system
    s["inputs"] = s["inputs"].replace("S_full_ext", base)
    s["adapter_args"] = {"key": system}
    s["notes"] = f"D181 audit re-run over {base}"
    s.pop("answer", None)
    return s


def _systems() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in json.loads(SYSTEMS_JSON.read_text(encoding="utf-8"))["systems"]}


def install_systems(extra: Iterable[dict[str, Any]], dry: bool) -> Path:
    """``<RUN>/systems_audit.json`` = the registry's systems.json plus ``extra``; ``UOL_FINAL_SYSTEMS`` points at it (so
    worker processes, which inherit the environment, see the same entries). Dry run: nothing written."""
    path = RUN / "systems_audit.json"
    data = json.loads(SYSTEMS_JSON.read_text(encoding="utf-8"))
    have = {s["id"] for s in data["systems"]}
    add = [e for e in extra if e["id"] not in have]
    if dry:
        print(f"[dry] would write {path} (+{[e['id'] for e in add]}) and set UOL_FINAL_SYSTEMS")
        return path
    if path.exists():                          # keep earlier additions of this run
        old = json.loads(path.read_text(encoding="utf-8"))
        data["systems"] = old["systems"]
        have = {s["id"] for s in data["systems"]}
        add = [e for e in extra if e["id"] not in have]
    data["systems"] += add
    data["n_systems"] = len(data["systems"])
    data["audit_note"] = "copy of src/uol/final/systems.json with the D181 audit entries appended"
    write_json(path, data)
    os.environ["UOL_FINAL_SYSTEMS"] = str(path)
    return path


# --------------------------------------------------------------------------- guards


def approved(token: str, go: bool) -> bool:
    """True only with ``--go`` and the token in ``UOL_AUDIT_D181_APPROVE`` (comma list); otherwise a dry run."""
    toks = {t.strip() for t in os.environ.get("UOL_AUDIT_D181_APPROVE", "").split(",") if t.strip()}
    if go and token not in toks:
        raise PermissionError(f"--go needs UOL_AUDIT_D181_APPROVE to include {token!r} ({TOKENS[token]}): the author's "
                              "approval, see RUN_PLAN.md")
    return go and token in toks


def need_threads() -> None:
    if os.environ.get("UOL_FINAL_THREADS") != str(THREADS) or os.environ.get("OMP_NUM_THREADS") != str(THREADS):
        raise RuntimeError(f"set UOL_FINAL_THREADS={THREADS} and OMP_NUM_THREADS={THREADS} before the process starts")


def codex_signed_in() -> dict[str, Any]:
    """``codex login status`` (no model call). The LLM steps refuse unless it reports a ChatGPT sign-in."""
    from uol.round3 import luna
    try:
        exe = str(luna.codex_bin())
    except Exception as exc:                   # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out = subprocess.run([exe, "login", "status"], capture_output=True, text=True, timeout=60)
    ver = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
    text = (out.stdout + out.stderr).strip()
    return {"ok": out.returncode == 0 and "Logged in" in text, "status": text, "version": ver.stdout.strip(), "exe": exe}


def check_paths(paths: dict[str, Path]) -> dict[str, bool]:
    res = {k: Path(p).exists() for k, p in paths.items()}
    for k, ok in res.items():
        print(f"  {'OK     ' if ok else 'MISSING'} {k}: {paths[k]}")
    return res


def refuse_record_paths(*paths: Path) -> None:
    """Outputs never land in a registered record."""
    for p in paths:
        rp = Path(p).resolve()
        for rec in (RECORD, NEW_RUN, ROOT / "experiments" / "runs" / "timing_20261026d",
                    ROOT / "experiments" / "runs" / "llm_sweep_20261028"):
            if rp == rec.resolve() or rec.resolve() in rp.parents:
                raise PermissionError(f"{p} is inside the registered record {rec}")


# --------------------------------------------------------------------------- files


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float), encoding="utf-8")
    tmp.replace(path)


def log(msg: str, name: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}"
    print(line, flush=True)
    p = RUN / "logs" / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def copy_verified(src: Path, dst: Path) -> str:
    """Copy one file (never a link: a later write must not reach the record) and check its sha256."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if sha256(dst) != sha256(src):
            raise RuntimeError(f"{dst} exists and differs from {src}")
        return sha256(dst)
    shutil.copy2(src, dst)
    h = sha256(dst)
    if h != sha256(src):
        raise RuntimeError(f"copy of {src} differs")
    return h


def copy_tree_verified(src: Path, dst: Path) -> dict[str, str]:
    return {str(p.relative_to(src)): copy_verified(p, dst / p.relative_to(src))
            for p in sorted(Path(src).rglob("*")) if p.is_file() and p.name != "desktop.ini"}


def seed_topic_cache(src: Path, work: Path) -> dict[str, str]:
    """Copy fitted topic models into ``<work>/topic``. gensim's LDA (3 workers) and doc2vec (4 workers) are not
    reproducible run to run, so the control and the leave-one-out must read the same pickles to be paired."""
    return {p.name: copy_verified(p, Path(work) / "topic" / p.name) for p in sorted(Path(src).glob("*.pkl"))}


def receipt(step: str, obj: dict[str, Any]) -> Path:
    from uol.final import harness
    obj = {"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
           "env": {k: os.environ.get(k) for k in ("UOL_FINAL_THREADS", "OMP_NUM_THREADS", "UOL_FINAL_LABELS",
                                                   "UOL_FINAL_SYSTEMS", "UOL_AUDIT_D181_APPROVE")},
           "versions": harness.versions(), "code_sha256": harness.code_hashes(), **obj}
    p = RUN / "receipts" / f"{step}.json"
    write_json(p, obj)
    return p


# --------------------------------------------------------------------------- contrasts


def per_case(store_or_run: Path, system: str, population: str = "DEV"):
    import pandas as pd
    p = Path(store_or_run) / "runs" / system / population / "per_case.parquet"
    return pd.read_parquet(p)


def aligned_frames(fa, fb, measure: str = "ndcg_at_10"):
    """``uol.final.stats.aligned`` for two frames that may come from different stores (same arithmetic)."""
    import numpy as np
    from uol.final import measures
    common = sorted(set(fa["query_id"]) & set(fb["query_id"]))
    if not common:
        raise ValueError("the two runs share no case")
    fa = fa.set_index("query_id").loc[common]
    fb = fb.set_index("query_id").loc[common]
    ia, va = measures.per_component(fa[measure].to_numpy(), fa["component_id"].to_numpy())
    ib, vb = measures.per_component(fb[measure].to_numpy(), fb["component_id"].to_numpy())
    keep = sorted(set(ia) & set(ib))
    pa, pb = dict(zip(ia, va)), dict(zip(ib, vb))
    return keep, np.asarray([pa[k] for k in keep]), np.asarray([pb[k] for k in keep])


def dev_family(name: str, pairs: list[tuple[str, Any, str, Any]], draws: int = DRAWS_DEV,
               population: str = "DEV", measure: str = "ndcg_at_10", size: int | None = None) -> dict[str, Any]:
    """One family in the format of the record's contrast files (D138 arithmetic: ``round3.stats.contrast`` with the
    family size, seed 20261004, margin .02, then ``round3.stats.family`` for Holm). ``pairs``: (label a, frame a,
    label b, frame b). ``size`` larger than the pairs: Bonferroni at that size (a member of a record family whose other
    members are not rerun); Holm is then not computed."""
    from uol.round3 import stats as R3
    m = size or len(pairs)
    cons = {}
    for la, fa, lb, fb in pairs:
        comps, va, vb = aligned_frames(fa, fb, measure)
        out = R3.contrast(va, vb, family_size=m, draws=draws, seed=SEED, margin=MARGIN)
        cons[f"{la} - {lb}"] = {"a": la, "b": lb, "population": population, "measure": measure,
                                "components": len(comps), **out}
    if m == len(pairs):
        return {"family": name, "size": m, "contrasts": R3.family(cons)}
    return {"family": name, "size": m, "contrasts": cons,
            "note": f"Bonferroni at the record family's size {m}; Holm not recomputed (other members not rerun)"}
