"""Remote GPU worker for Round 3: run GPU jobs on the author's RTX 5080 PC without migrating the project.

The project, data, ledgers, Codex/Luna calls and the embedding server all stay on the main machine.
A GPU job is packed into a self-contained folder: the job script and any helpers, only the input files it needs,
and a manifest with SHA-256 hashes. The 5080 PC runs it and returns the outputs, which are hash-checked on arrival.
The worker holds no credentials and no project copy.

Transports (``UOL_GPU_TRANSPORT``):
* ``drive`` (default when ``UOL_GPU_EXCHANGE`` is set). The author's Google Drive. Jobs are written to
  ``<exchange>/queue/<job_id>``, plus a ``READY.json`` manifest written last. ``tools/gpu_worker/drive_worker.py`` on
  the 5080 PC claims each job, verifies the hashes, runs it locally (``C:\\uol_gpu\\jobs``), and writes
  ``<exchange>/done/<job_id>`` with ``DONE.json`` written last. Heartbeats go to ``<exchange>/workers/<host>.json``.
  After a verified pull, both Drive copies are deleted, so the authority PC holds the only copy.
* ``ssh``. ``UOL_GPU_HOST`` is an SSH alias; see ``tools/gpu_worker/README.md``, alternative B.
* ``local``. In-process loopback, for tests.

Job contract: the entry script runs as ``python code/<entry> --inputs inputs --out outputs [args...]`` from the job
folder. It writes deliverables under ``outputs/``, which are returned. Large artifacts, such as checkpoints, go under
the worker-local folder named by the env var ``UOL_WORKER_SCRATCH``, which is never returned or synced.
``run.py`` records the GPU, torch version, timings, exit code and output hashes in ``receipt.json``.

Usage (``run`` blocks until the job ends, so launch long jobs from a background process)::

    from uol.round3 import gpu_remote as G
    receipt = G.submit("r2_qwen06b_f0_s1", entry=Path("src/uol/round3/gpu_jobs/r2_train.py"),
                       inputs={"pairs.jsonl": p1}, args=["--fold", "0"], expected_outputs=["scores.npy"])
    python -m uol.round3.gpu_remote check      # worker alive, RTX 5080, sm_120, bf16 and a toy round trip
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[3]
LOCAL_JOBS = ROOT / "experiments" / "runs" / "round3_best_case_20261013" / "gpu_jobs"
LEDGER = ROOT / "research" / "state" / "round3_best_case_20261013" / "RUN_LEDGER.jsonl"
LOOPBACK_ROOT = Path(tempfile.gettempdir()) / "uol_gpu_loopback"
POLL_SECONDS = 20

RUN_PY = r'''import hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime, timezone
here = os.path.dirname(os.path.abspath(__file__))
job = json.load(open(os.path.join(here, "job.json"), encoding="utf-8"))
os.makedirs(os.path.join(here, "outputs"), exist_ok=True)
env = {"python": sys.version, "platform": platform.platform(), "host": platform.node()}
try:
    import torch
    env.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available(),
               device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               arch_list=torch.cuda.get_arch_list() if torch.cuda.is_available() else [])
except Exception as exc:
    env["torch_error"] = repr(exc)
t0 = time.time(); start = datetime.now(timezone.utc).isoformat()
cmd = [sys.executable, os.path.join("code", job["entry"]), "--inputs", "inputs", "--out", "outputs", *job["args"]]
with open(os.path.join(here, "outputs", "job.log"), "a", encoding="utf-8") as log:
    rc = subprocess.call(cmd, cwd=here, stdout=log, stderr=subprocess.STDOUT)
outs = {}
for dp, _, fs in os.walk(os.path.join(here, "outputs")):
    for f in fs:
        p = os.path.join(dp, f); h = hashlib.sha256()
        with open(p, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 20), b""): h.update(b)
        outs[os.path.relpath(p, here).replace(os.sep, "/")] = {"sha256": h.hexdigest(), "bytes": os.path.getsize(p)}
json.dump({"job_id": job["job_id"], "exit_code": rc, "start_utc": start, "end_utc": datetime.now(timezone.utc).isoformat(),
           "seconds": round(time.time() - t0, 1), "env": env, "outputs": outs},
          open(os.path.join(here, "receipt.json"), "w", encoding="utf-8"), indent=1)
sys.exit(rc)
'''


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def transport() -> str:
    t = os.environ.get("UOL_GPU_TRANSPORT")
    if t:
        return t
    if os.environ.get("UOL_GPU_EXCHANGE"):
        return "drive"
    if os.environ.get("UOL_GPU_HOST", "local") != "local":
        return "ssh"
    return "local"


def exchange() -> Path:
    ex = os.environ.get("UOL_GPU_EXCHANGE")
    if not ex:
        raise RuntimeError("UOL_GPU_EXCHANGE is not set (e.g. 'exchange\\UOL_GPU_EXCHANGE')")
    return Path(ex)


def host() -> str:
    return os.environ.get("UOL_GPU_HOST", "local")


def remote_py() -> str:
    return os.environ.get("UOL_GPU_PY", "C:/uol_gpu/venv/Scripts/python.exe")


def remote_root() -> str:
    return os.environ.get("UOL_GPU_ROOT", "C:/uol_gpu/jobs")


def _ssh_bin(name: str) -> str:
    win = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32" / "OpenSSH" / f"{name}.exe"
    return str(win) if win.exists() else (shutil.which(name) or name)


def _ledger(row: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def manifest_of(folder: Path, skip: Sequence[str] = ()) -> dict[str, dict]:
    out = {}
    for p in sorted(folder.rglob("*")):
        rel = p.relative_to(folder).as_posix()
        if p.is_file() and not any(rel == s or rel.startswith(s + "/") for s in skip):
            out[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def manifest_ok(folder: Path, manifest: dict[str, dict]) -> bool:
    try:
        return all((folder / rel).is_file() and (folder / rel).stat().st_size == m["bytes"]
                   and sha256(folder / rel) == m["sha256"] for rel, m in manifest.items())
    except OSError:
        return False


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def pack(job_id: str, entry: Path, inputs: dict[str, Path], args: Sequence[str] = (),
         helpers: Sequence[Path] = (), expected_outputs: Sequence[str] = ()) -> Path:
    """Create the local job folder with code, inputs and a hashed manifest."""
    job = LOCAL_JOBS / job_id
    if job.exists():
        raise FileExistsError(f"{job} exists; use a new job_id (jobs are immutable)")
    (job / "code").mkdir(parents=True)
    (job / "inputs").mkdir()
    for src in (Path(entry), *map(Path, helpers)):
        shutil.copy2(src, job / "code" / src.name)
    manifest = {}
    for name, src in inputs.items():
        dst = job / "inputs" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest[name] = {"source": str(src), "sha256": sha256(dst), "bytes": dst.stat().st_size}
    spec = {"job_id": job_id, "entry": Path(entry).name, "args": list(args), "created_utc": _utc(),
            "inputs": manifest, "code": {p.name: sha256(p) for p in (job / "code").iterdir()},
            "expected_outputs": list(expected_outputs)}
    (job / "job.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    (job / "run.py").write_text(RUN_PY, encoding="utf-8")
    return job


def _sh(cmd: list[str], timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          stdin=subprocess.DEVNULL)


def push(job: Path) -> None:
    t = transport()
    if t == "local":
        dst = LOOPBACK_ROOT / job.name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(job, dst, ignore=shutil.ignore_patterns("remote"))
    elif t == "drive":
        ex = exchange()
        dst = ex / "queue" / job.name
        if dst.exists() or (ex / "done" / job.name).exists():
            raise FileExistsError(f"{job.name} already in the exchange")
        shutil.copytree(job, dst, ignore=shutil.ignore_patterns("remote"))
        write_json_atomic(dst / "READY.json", {"job_id": job.name, "utc": _utc(), "manifest": manifest_of(dst)})
    elif t == "ssh":
        h = host()
        r = _sh([_ssh_bin("ssh"), h, f'if not exist "{remote_root()}" mkdir "{remote_root()}"'], timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"ssh mkdir failed: {r.stderr[-500:]}")
        r = _sh([_ssh_bin("scp"), "-r", "-q", str(job), f"{h}:{remote_root()}/"])
        if r.returncode != 0:
            raise RuntimeError(f"scp push failed: {r.stderr[-500:]}")
    else:
        raise ValueError(t)
    _ledger({"utc": _utc(), "kind": "gpu_remote_push", "transport": t, "job_id": job.name})


def worker_status() -> list[dict]:
    """Heartbeats of drive workers, with their age in seconds."""
    out = []
    for p in sorted((exchange() / "workers").glob("*.json")):
        try:
            hb = json.loads(p.read_text(encoding="utf-8"))
            hb["age_seconds"] = round(time.time() - p.stat().st_mtime)
            out.append(hb)
        except (OSError, json.JSONDecodeError):
            continue
    return out


def run(job: Path, timeout: int | None = None) -> int:
    """Run (or, for drive, wait for) the job and block until it ends."""
    t = transport()
    t0 = time.time()
    if t == "local":
        rc = subprocess.call([sys.executable, str(LOOPBACK_ROOT / job.name / "run.py")], stdin=subprocess.DEVNULL,
                             timeout=timeout)
    elif t == "drive":
        done = exchange() / "done" / job.name / "DONE.json"
        while True:
            if done.exists():
                try:
                    rc = int(json.loads(done.read_text(encoding="utf-8"))["exit_code"])
                    break
                except (OSError, json.JSONDecodeError, KeyError):
                    pass
            if timeout is not None and time.time() - t0 > timeout:
                raise TimeoutError(f"{job.name}: no DONE.json after {timeout}s; worker status {worker_status()}")
            time.sleep(POLL_SECONDS)
    elif t == "ssh":
        remote_run = f"{remote_root()}/{job.name}/run.py"
        rc = _sh([_ssh_bin("ssh"), "-o", "ServerAliveInterval=60", host(), f'"{remote_py()}" "{remote_run}"'],
                 timeout=timeout).returncode
    else:
        raise ValueError(t)
    _ledger({"utc": _utc(), "kind": "gpu_remote_run", "transport": t, "job_id": job.name, "exit_code": rc,
             "wall_seconds": round(time.time() - t0, 1)})
    return rc


def pull(job: Path, sync_wait: int = 1800, cleanup: bool = True) -> dict:
    """Copy outputs and receipt back, verify every output hash, and return the receipt."""
    t = transport()
    back = job / "remote"
    if back.exists():
        shutil.rmtree(back)
    back.mkdir()
    if t == "local":
        src = LOOPBACK_ROOT / job.name
        shutil.copytree(src / "outputs", back / "outputs")
        shutil.copy2(src / "receipt.json", back / "receipt.json")
    elif t == "drive":
        src = exchange() / "done" / job.name
        done = json.loads((src / "DONE.json").read_text(encoding="utf-8"))
        t0 = time.time()
        while not manifest_ok(src, done["manifest"]):        # large files may still be syncing down
            if time.time() - t0 > sync_wait:
                raise TimeoutError(f"{job.name}: outputs still not synced after {sync_wait}s")
            time.sleep(POLL_SECONDS)
        for rel in done["manifest"]:
            (back / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, back / rel)
    elif t == "ssh":
        for item in ("outputs", "receipt.json"):
            r = _sh([_ssh_bin("scp"), "-r", "-q", f"{host()}:{remote_root()}/{job.name}/{item}", str(back)])
            if r.returncode != 0:
                raise RuntimeError(f"scp pull {item} failed: {r.stderr[-500:]}")
    receipt = json.loads((back / "receipt.json").read_text(encoding="utf-8"))
    bad = [rel for rel, meta in receipt["outputs"].items() if sha256(back / rel) != meta["sha256"]]
    if bad:
        raise RuntimeError(f"hash mismatch after pull: {bad[:5]}")
    spec = json.loads((job / "job.json").read_text(encoding="utf-8"))
    missing = [o for o in spec.get("expected_outputs", []) if f"outputs/{o}" not in receipt["outputs"]]
    _ledger({"utc": _utc(), "kind": "gpu_remote_pull", "transport": t, "job_id": job.name,
             "exit_code": receipt["exit_code"], "seconds": receipt["seconds"], "device": receipt["env"].get("device"),
             "torch": receipt["env"].get("torch"), "outputs": {k: v["sha256"] for k, v in receipt["outputs"].items()},
             "missing_expected": missing})
    if t == "drive" and cleanup:
        for d in (exchange() / "queue" / job.name, exchange() / "done" / job.name):
            shutil.rmtree(d, ignore_errors=True)   # Drive keeps deleted files in its trash for 30 days
    if receipt["exit_code"] != 0 or missing:
        raise RuntimeError(f"job {job.name} failed: exit {receipt['exit_code']}, missing {missing}; "
                           f"see {back / 'outputs' / 'job.log'}")
    return receipt


def submit(job_id: str, entry: Path, inputs: dict[str, Path], args: Sequence[str] = (), helpers: Sequence[Path] = (),
           expected_outputs: Sequence[str] = (), timeout: int | None = None) -> dict:
    job = pack(job_id, entry, inputs, args, helpers, expected_outputs)
    push(job)
    run(job, timeout=timeout)
    return pull(job)


def check(timeout: int = 1800) -> dict:
    """Worker health plus an end-to-end toy job (no project data)."""
    t = transport()
    out = {"transport": t, "utc": _utc()}
    if t == "drive":
        out["exchange"] = str(exchange())
        out["workers"] = worker_status()
        out["worker_alive"] = any(w["age_seconds"] < 300 for w in out["workers"])
        if not out["worker_alive"]:
            out["ok"] = False
            out["hint"] = "No fresh heartbeat. On the 5080 PC run C:\\uol_gpu\\start_worker.cmd and keep it open."
            return out
    probe_dir = Path(tempfile.mkdtemp(prefix="uol_gpu_check_"))
    entry = probe_dir / "probe_job.py"
    entry.write_text(
        "import argparse,json,os\nap=argparse.ArgumentParser();ap.add_argument('--inputs');ap.add_argument('--out')\n"
        "a=ap.parse_args();r={'payload':open(os.path.join(a.inputs,'x.txt')).read()}\n"
        "try:\n import torch\n r.update(device=torch.cuda.get_device_name(0),arch=torch.cuda.get_arch_list(),"
        "torch=torch.__version__)\n x=torch.randn(256,256,device='cuda',dtype=torch.bfloat16)\n"
        " r['bf16_ok']=bool((x@x).float().abs().mean()>0)\nexcept Exception as e:\n r['torch_error']=repr(e)\n"
        "json.dump(r,open(os.path.join(a.out,'probe.json'),'w'))\n", encoding="utf-8")
    (probe_dir / "x.txt").write_text("roundtrip", encoding="utf-8")
    job_id = "check_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    try:
        receipt = submit(job_id, entry, {"x.txt": probe_dir / "x.txt"}, expected_outputs=["probe.json"], timeout=timeout)
        probe = json.loads((LOCAL_JOBS / job_id / "remote" / "outputs" / "probe.json").read_text())
        out["roundtrip"] = {"ok": probe.get("payload") == "roundtrip", "seconds": receipt["seconds"], **probe}
    except Exception as exc:  # report, do not hide
        out["roundtrip"] = {"ok": False, "error": repr(exc)[:500]}
    rt = out["roundtrip"]
    out["ok"] = bool(rt.get("ok")) and "sm_120" in rt.get("arch", []) and rt.get("bf16_ok") is True
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["check", "workers"])
    a = ap.parse_args()
    print(json.dumps(check() if a.command == "check" else worker_status(), indent=1, ensure_ascii=False))
