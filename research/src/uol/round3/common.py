"""Round 3 shared definitions: paths, the rebuilt frozen feature table, folds, populations, subgroups and dates.

Everything here is label-free except ``band_array`` (training-fold citation counts) and ``gold_rows``. Lockbox and
prospective rows are never returned by the dev helpers; a lockbox read anywhere else must go through
``lockbox_read`` so it is written to LOCKBOX_LEDGER.jsonl first.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.facts_to_law import prior as prior_mod

ROOT = Path(__file__).resolve().parents[3]
PROGRAM = "round3_best_case_20261013"
RUN = ROOT / "experiments/runs" / PROGRAM
ARMS = RUN / "arms"
STATE = ROOT / "research/state" / PROGRAM
REPORTS = ROOT / "research/reports" / PROGRAM
PROGRAM_DIR = ROOT / "research/programs" / PROGRAM
RUN_LEDGER = STATE / "RUN_LEDGER.jsonl"
LOCKBOX_LEDGER = STATE / "LOCKBOX_LEDGER.jsonl"
STATE_JSON = STATE / "STATE.json"
EP_RUN = ep.RUN                                        # the rebuilt experiments/runs/embedding_push_20261004
ROUND2 = ROOT / "experiments/runs/round2_20261004"
SOURCES_JSONL = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"
LAY_TEXTS = ROUND2 / "H/texts.jsonl"
LUNA_CUTOFF = "2026-02-16"                              # gpt-5.6-luna stated knowledge cutoff
BANDS = ("unseen", "rare", "frequent")                  # 0, 1-4, 5+ training-fold components
SEED = 20261004
GPU_LOCK = ROOT / ".cache/round3_gpu.lock"
LOCKBOX_PURPOSES = ("phase2_probe", "phase7_batch", "phase8_look", "final_score")


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ledger(row: dict) -> None:
    append_jsonl(RUN_LEDGER, {"utc": utc(), **row})


def lockbox_read(purpose: str, what: str, code_file: Path) -> None:
    """Record a sanctioned lockbox/prospective read BEFORE it happens; anything else is refused."""
    if purpose not in LOCKBOX_PURPOSES:
        raise PermissionError(f"lockbox read purpose {purpose!r} is not sanctioned ({LOCKBOX_PURPOSES})")
    append_jsonl(LOCKBOX_LEDGER, {"utc": utc(), "purpose": purpose, "what": what, "code": str(code_file),
                                  "code_sha256": sha256_file(Path(code_file))})


# ----------------------------------------------------------------------------- release, roles, folds

release = ep.release
roles = ep.roles
dev_rows = ep.dev_rows
lockbox_rows = ep.lockbox_rows
fold_rows = ep.fold_rows
training_rows = ep.training_rows
population_rows = ep.population_rows


def dev_groups() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(role, rows scored, rows trained on) for the five dev folds only."""
    return [g for g in ep.scoring_groups() if g[0] != "lockbox"]


def fold_of(rows: Sequence[int]) -> np.ndarray:
    r = roles()
    return np.asarray([int(str(r[int(i)]).replace("dev_fold", "")) if str(r[int(i)]).startswith("dev_fold") else -1
                       for i in rows])


def populations(rows: Sequence[int] | None = None) -> dict[str, np.ndarray]:
    """P (full references) and P_spec (condensed specific lists) restricted to ``rows`` (default: dev rows)."""
    rows = dev_rows() if rows is None else rows
    return {"P": population_rows("P", rows), "P_spec": population_rows("P_spec", rows)}


def component_ids(rows: Sequence[int]) -> np.ndarray:
    d = release()
    return np.asarray([d.queries[int(r)].component_id for r in rows])


# ----------------------------------------------------------------------------- popularity bands (training folds)

@lru_cache(maxsize=8)
def _counts_for(role: str) -> np.ndarray:
    d = release()
    ids = [d.queries[int(r)].query_id for r in training_rows(role)]
    return prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)


def band_array(role: str) -> np.ndarray:
    """Per candidate article: 'unseen' (0), 'rare' (1-4), 'frequent' (5+) distinct training components of ``role``."""
    c = _counts_for(role)
    return np.where(c == 0, "unseen", np.where(c <= prior_mod.RARE_MAX, "rare", "frequent"))


def band_of_row(row: int) -> np.ndarray:
    return band_array(str(roles()[int(row)]))


# ----------------------------------------------------------------------------- lexical overlap (round-2 A definition)

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("de la el en y a los las que del se por con no un una su al lo para es o como mas sus le ya este esta "
                  "entre cuando sin sobre tambien hasta donde quien desde nos durante todos uno les ni contra otros ese "
                  "eso ante ellos e esto antes algunos unos yo otro otras otra el tanto esa estos mucho quienes nada "
                  "muchos cual poco ella estar estas algunas algo nosotros mi mis tu te ti tus ellas nosotras".split())


def _content_tokens(text: str) -> set[str]:
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return {w for w in _TOKEN.findall(t) if len(w) > 2 and w not in _STOP}


@lru_cache(maxsize=1)
def overlap_scores() -> np.ndarray:
    """Per release row: mean over specific positives of the IDF-weighted share of query content tokens in the article.

    Same quantity as ``uol.round2.a_p03.overlap_scores``; the cached round-2 array is reused when it exists.
    """
    cached = ROUND2 / "A/cache/overlap.npy"
    if cached.exists():
        return np.load(cached)
    from uol.round2 import a_p03
    return a_p03.overlap_scores()


def overlap_deciles(rows: Sequence[int]) -> np.ndarray:
    """Decile 0-9 of each row's overlap, with edges from the dev P_spec rows (round-2 A task (c)); -1 if undefined."""
    ov = overlap_scores()
    ref = population_rows("P_spec", dev_rows())
    edges = np.quantile(ov[ref][np.isfinite(ov[ref])], np.linspace(0, 1, 11))[1:-1]
    vals = ov[np.asarray(rows, dtype=np.int64)]
    out = np.clip(np.searchsorted(edges, vals, side="right"), 0, 9)
    return np.where(np.isfinite(vals), out, -1)


def low_overlap(rows: Sequence[int]) -> np.ndarray:
    """Pre-registered 'low lexical overlap' subgroup: bottom three deciles."""
    d = overlap_deciles(rows)
    return (d >= 0) & (d <= 2)


# ----------------------------------------------------------------------------- lay register (round-2 H rewrites)

@lru_cache(maxsize=1)
def lay_texts() -> dict[int, dict[str, Any]]:
    """{release row: {arm: {"text", "flagged"}}} from round-2 H (arm O original, L lay rewrite, others controls)."""
    out: dict[int, dict[str, Any]] = {}
    if not LAY_TEXTS.exists():
        return out
    with LAY_TEXTS.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out.setdefault(int(rec["row"]), {})[rec["arm"]] = {"text": rec["text"], "flagged": bool(rec.get("flagged"))}
    return out


def lay_flag(rows: Sequence[int]) -> np.ndarray:
    """True when the row has a lay ('L') rewrite in round-2 H."""
    lt = lay_texts()
    return np.asarray([("L" in lt.get(int(r), {})) for r in rows])


# ----------------------------------------------------------------------------- publication dates and the Luna cutoff

@lru_cache(maxsize=1)
def publication_dates() -> dict[int, str | None]:
    """{release row: publication_date or None} from the structured-sources file (query_id join only)."""
    d = release()
    by_qid = {q.query_id: i for i, q in enumerate(d.queries)}
    out: dict[int, str | None] = {i: None for i in range(len(d.queries))}
    with SOURCES_JSONL.open(encoding="utf-8") as f:
        for line in f:
            if '"query_id"' not in line:
                continue
            rec = json.loads(line)
            i = by_qid.get(rec.get("query_id"))
            if i is not None:
                out[i] = rec.get("publication_date") or None
    return out


def post_luna_cutoff(rows: Sequence[int]) -> np.ndarray:
    """True when the row was published on or after 2026-02-16 (Luna's stated cutoff); undated rows are False."""
    pd = publication_dates()
    return np.asarray([bool(pd.get(int(r)) and pd[int(r)] >= LUNA_CUTOFF) for r in rows])


# ----------------------------------------------------------------------------- frozen feature table and pipelines

def frozen_pipeline_manifest(name: str) -> dict:
    from uol.embedding_push import fusion_ltr as fl
    return json.loads((fl.pipeline_path(name).parent / "manifest.json").read_text(encoding="utf-8"))


def frozen_configs(name: str = "LTR_emb_hgb") -> dict[str, dict]:
    """Per scoring group, the frozen HGB hyper-parameters of a pipeline (from the rebuilt manifest)."""
    man = frozen_pipeline_manifest(name)
    return {role: rec["config"] for role, rec in man["configuration"].items()}


def gold_rows(rows: Sequence[int], view: str = "spec") -> list[list[int]]:
    """Positive article indices per row (specific or full). Label access: callers must restrict rows to dev."""
    d = release()
    src = d.specific_idx if view == "spec" else d.positive_idx
    return [list(src[int(r)]) for r in rows]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    return str(x)


# ----------------------------------------------------------------------------- CPU slots (RAM guard across processes)

CPU_SLOTS = 1                     # 2026-10-14: one heavy step at a time (shared machine; Coordinator request)
_SLOT_DIR = ROOT / ".cache" / "round3_cpu_slots"


class cpu_slot:
    """Hold one of CPU_SLOTS machine-wide slots for a heavy CPU step (a refit uses several GB of RAM).

    Slots are lock files holding the owner PID; a slot whose owner died is reclaimed. Re-entrant per process.
    """
    _held: dict[int, int] = {}

    def __enter__(self):
        import os
        import time as _t
        pid = os.getpid()
        if pid in cpu_slot._held:
            cpu_slot._held[pid] += 1
            self.path = None
            return self
        _SLOT_DIR.mkdir(parents=True, exist_ok=True)
        while True:
            for i in range(CPU_SLOTS):
                p = _SLOT_DIR / f"slot{i}.lock"
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, str(pid).encode())
                    os.close(fd)
                    self.path = p
                    cpu_slot._held[pid] = 1
                    return self
                except FileExistsError:
                    try:
                        owner = int(p.read_text() or 0)
                    except (OSError, ValueError):
                        owner = 0
                    if owner and not ep.pid_alive(owner):
                        p.unlink(missing_ok=True)
            _t.sleep(20)

    def __exit__(self, *exc):
        import os
        pid = os.getpid()
        cpu_slot._held[pid] -= 1
        if cpu_slot._held[pid] == 0:
            del cpu_slot._held[pid]
            for p in _SLOT_DIR.glob("slot*.lock"):
                try:
                    if int(p.read_text() or 0) == pid:
                        p.unlink()
                except (OSError, ValueError):
                    pass
        return False
