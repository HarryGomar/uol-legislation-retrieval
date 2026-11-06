"""Shared release, split, score-file and evaluation helpers for the embedding push study.

Semantics follow ``uol.facts_to_law`` exactly: populations from ``crossfit.POPULATIONS``,
condensed specific lists (generic positives dropped, generic candidates excluded),
candidate-ID tie order and component-macro means. Lockbox rows can only be evaluated
after the lead approves ``FREEZE.json`` (PROTOCOL.md section 4).
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from uol.facts_to_law import crossfit, metrics, stats
from uol.facts_to_law.spec import ROOT

RELEASE = ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified"
STATE = ROOT / "research/state/embedding_push_20261004"
RUN = ROOT / "experiments/runs/embedding_push_20261004"
SCORES = RUN / "scores"
SPLIT_PATH = STATE / "SPLIT.json"
FREEZE_PATH = STATE / "FREEZE.json"
GPU_LOCK = ROOT / ".cache/embedding_push_gpu.lock"
METRIC_KEYS = metrics.METRICS
POPULATIONS = crossfit.POPULATIONS
BGE_DIR = Path(r"C:\uol_legacy_home\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots"
               r"\5617a9f61b028005a4858fdac845db406aefb181")
E5_DIR = Path(r"C:\uol_legacy_home\.cache\huggingface\hub\models--intfloat--multilingual-e5-small"
              r"\snapshots\614241f622f53c4eeff9890bdc4f31cfecc418b3")
DERIVED = ROOT / "data/derived/facts_to_law_v2"


@lru_cache(maxsize=1)
def release() -> crossfit.ReleaseData:
    """The v11 release with its bound candidate catalogue (cached)."""
    return crossfit.load_release(RELEASE)


@lru_cache(maxsize=1)
def tie_order() -> np.ndarray:
    return metrics.tie_order_from_ids(release().candidate_ids)


@lru_cache(maxsize=1)
def roles() -> np.ndarray:
    """Per release row: 'lockbox' or 'dev_fold<k>' (SPLIT.json)."""
    assignment = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))["assignment"]
    return np.asarray([assignment[q.query_id] for q in release().queries])


def dev_rows() -> np.ndarray:
    return np.flatnonzero(roles() != "lockbox")


def lockbox_rows() -> np.ndarray:
    return np.flatnonzero(roles() == "lockbox")


def fold_rows(k: int) -> np.ndarray:
    """Dev rows of outer fold k (0-4)."""
    return np.flatnonzero(roles() == f"dev_fold{k}")


def training_rows(role: str) -> np.ndarray:
    """Rows whose labels may be used to score rows of ``role``: other dev folds, or all dev for the lockbox."""
    if role == "lockbox":
        return dev_rows()
    if not role.startswith("dev_fold"):
        raise ValueError(f"unknown role {role!r}")
    return np.flatnonzero((roles() != "lockbox") & (roles() != role))


def scoring_groups() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(role, rows to score, rows to train on) for the five dev folds and the lockbox."""
    groups = [(f"dev_fold{k}", fold_rows(k), training_rows(f"dev_fold{k}")) for k in range(5)]
    groups.append(("lockbox", lockbox_rows(), training_rows("lockbox")))
    return groups


def population_rows(name: str, rows: Sequence[int] | None = None) -> np.ndarray:
    """Rows admitted by a facts-to-law population; specific views also need a specific positive."""
    data, pop = release(), POPULATIONS[name]
    candidates = range(len(data.queries)) if rows is None else rows
    keep = [int(i) for i in candidates
            if pop.admits(data.queries[int(i)]) and (pop.view == "full" or data.specific_idx[int(i)])]
    return np.asarray(keep, dtype=np.int64)


def _lockbox_allowed() -> bool:
    if not FREEZE_PATH.exists():
        return False
    return bool(json.loads(FREEZE_PATH.read_text(encoding="utf-8")).get("lead_approved"))


def evaluate(scores: np.ndarray, rows: Sequence[int], population: str = "P_spec", *, top_k: int = 0,
             allow_lockbox: bool = False) -> dict[str, Any]:
    """Component-macro metrics for ``scores`` (rows aligned to ``rows``, full catalogue columns).

    Only rows admitted by ``population`` are scored. Returns macro means, aligned
    per-component arrays, sorted component IDs and per-query records.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if len(rows) != len(scores):
        raise ValueError("scores must have one row per entry of rows")
    if np.isin(rows, lockbox_rows()).any() and not (allow_lockbox and _lockbox_allowed()):
        raise PermissionError("lockbox rows can be evaluated only after the lead approves FREEZE.json")
    data, pop = release(), POPULATIONS[population]
    admitted = set(population_rows(population, rows).tolist())
    local = [j for j, r in enumerate(rows) if int(r) in admitted]
    if not local:
        raise ValueError("no admitted rows")
    specific = pop.view == "specific"
    positives = [(data.specific_idx if specific else data.positive_idx)[int(rows[j])] for j in local]
    exclude = data.generic_idx if specific and len(data.generic_idx) else None
    per_query = metrics.evaluate_matrix(np.asarray(scores[local], dtype=np.float64), positives,
                                        exclude_idx=exclude, top_k=top_k, tie_order=tie_order())
    components = [data.queries[int(rows[j])].component_id for j in local]
    macro, per_component, ordered = metrics.component_macro(per_query, components)
    return {"population": population, "n_queries": len(local), "n_components": len(ordered), "macro": macro,
            "per_component": per_component, "components": ordered, "rows": [int(rows[j]) for j in local],
            "per_query": per_query}


def contrast(a: dict[str, Any], b: dict[str, Any], metric: str = "ndcg_at_10",
             holm_reject: bool | None = None) -> dict[str, Any]:
    """Paired component contrast a - b over aligned components via ``stats.contrast`` (10,000 draws, seed 20261004)."""
    if a["components"] != b["components"]:
        raise ValueError("contrast needs aligned components")
    return {**stats.contrast(a["per_component"][metric], b["per_component"][metric], holm_reject=holm_reject),
            "metric": metric, "population": a["population"], "components": len(a["components"])}


def score_path(method_id: str) -> Path:
    return SCORES / method_id / "scores.npy"


def open_scores(method_id: str, mode: str = "r") -> np.ndarray:
    """Memmap of a method's float32 [queries, candidates] score file."""
    return np.load(score_path(method_id), mmap_mode=mode)


def create_scores(method_id: str) -> np.ndarray:
    """New float32 score memmap (fails if it exists), filled with NaN until every role is written."""
    path = score_path(method_id)
    path.parent.mkdir(parents=True, exist_ok=False)
    data = release()
    matrix = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                       shape=(len(data.queries), len(data.candidate_ids)))
    matrix[:] = np.nan
    return matrix


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def finish_scores(method_id: str, manifest: dict[str, Any], code_files: Sequence[Path] = ()) -> dict[str, Any]:
    """Check completeness, hash the score file and write ``manifest.json`` next to it."""
    matrix = open_scores(method_id)
    missing = [int(i) for i in range(matrix.shape[0]) if not np.isfinite(matrix[i]).any()]
    if missing:
        raise ValueError(f"{len(missing)} rows were never written, e.g. {missing[:5]}")
    record = {**manifest, "method_id": method_id, "shape": list(matrix.shape), "dtype": "float32",
              "scores_sha256": sha256_file(score_path(method_id)),
              "split_sha256": sha256_file(SPLIT_PATH), "release_manifest_sha256": sha256_file(RELEASE / "manifest.json"),
              "code_sha256": {str(p): sha256_file(Path(p)) for p in code_files},
              "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (score_path(method_id).parent / "manifest.json").write_text(json.dumps(record, indent=1, ensure_ascii=False),
                                                                encoding="utf-8")
    return record


def gpu_free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def own_gpu_mib() -> int:
    """GPU memory this process holds through torch (after emptying its cache), plus context overhead; 0 if unused."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available() or not torch.cuda.is_initialized():
        return 0
    torch.cuda.empty_cache()
    return int(torch.cuda.memory_reserved() // (1 << 20)) + 500


GPU_QUEUE = ROOT / ".cache/embedding_push_gpu.queue"


def pid_alive(pid: int) -> bool:
    """Whether a local process is still running (Windows-safe)."""
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _clear_stale_lock() -> None:
    """Remove the lock dir only when its recorded owner process is dead."""
    owner = GPU_LOCK / "owner.json"
    try:
        pid = int(json.loads(owner.read_text())["pid"])
    except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
        return
    if not pid_alive(pid):
        for child in GPU_LOCK.glob("*"):
            with contextlib.suppress(FileNotFoundError):
                child.unlink()
        with contextlib.suppress(FileNotFoundError, OSError):
            GPU_LOCK.rmdir()


@contextlib.contextmanager
def gpu_lock(owner: str, min_free_mib: int = 3500, poll: int = 10) -> Iterator[None]:
    """First-come-first-served exclusive GPU use among study agents (ticket queue plus lock dir).

    Each caller takes a time-ordered ticket; only the oldest live ticket may create the
    lock dir, so a process that releases and immediately re-requests queues behind
    earlier waiters. Tickets and locks of dead processes are cleaned up. The free-memory
    check counts this process's own torch memory as available (no self-deadlock); about
    2 GB is always used by the display, so 3,500 MiB free means no other large job.
    """
    GPU_QUEUE.mkdir(parents=True, exist_ok=True)
    ticket = GPU_QUEUE / f"{time.time_ns():020d}_{os.getpid()}_{owner}.ticket"
    ticket.write_text(owner)
    acquired = False
    try:
        while True:
            live = []
            for t in sorted(GPU_QUEUE.glob("*.ticket")):
                try:
                    pid = int(t.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if pid != os.getpid() and not pid_alive(pid):
                    with contextlib.suppress(FileNotFoundError):
                        t.unlink()
                    continue
                live.append(t)
            if live and live[0] == ticket:
                try:
                    os.mkdir(GPU_LOCK)
                    acquired = True
                    break
                except FileExistsError:
                    _clear_stale_lock()
            time.sleep(poll)
        (GPU_LOCK / "owner.json").write_text(json.dumps({"owner": owner, "pid": os.getpid(),
                                                         "utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}))
        while gpu_free_mib() + own_gpu_mib() < min_free_mib:
            print(f"[gpu_lock] waiting for {min_free_mib} MiB free (counting own memory)", file=sys.stderr, flush=True)
            time.sleep(60)
        yield
    finally:
        if acquired:
            for child in GPU_LOCK.glob("*"):
                with contextlib.suppress(FileNotFoundError):
                    child.unlink()
            with contextlib.suppress(FileNotFoundError, OSError):
                GPU_LOCK.rmdir()
        with contextlib.suppress(FileNotFoundError):
            ticket.unlink()


def chunk_article_index(units_jsonl: Path) -> np.ndarray:
    """Candidate index of each chunk row in a chunk-vector file (units.jsonl gives candidate_id per row)."""
    index = release().candidate_index
    rows = [json.loads(line) for line in units_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    return np.asarray([index[r["candidate_id"]] for r in rows], dtype=np.int64)


def maxp_scores(query_vecs: np.ndarray, chunk_vecs: np.ndarray, chunk_article: np.ndarray, n_articles: int,
                batch: int = 256) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (start, [b, n_articles]) max-over-chunks cosine scores for query batches."""
    order = np.argsort(chunk_article, kind="stable")
    sorted_vecs, sorted_art = chunk_vecs[order], chunk_article[order]
    starts = np.flatnonzero(np.r_[True, sorted_art[1:] != sorted_art[:-1]])
    owners = sorted_art[starts]
    for s in range(0, len(query_vecs), batch):
        sims = query_vecs[s:s + batch] @ sorted_vecs.T
        best = np.maximum.reduceat(sims, starts, axis=1)
        out = np.full((sims.shape[0], n_articles), -np.inf, dtype=np.float32)
        out[:, owners] = best
        yield s, out
