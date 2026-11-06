"""The whole run on the remote machine, as a worker job (run.py copies this as code/entry).

Unpacks ``whole_pack.zip`` from the exchange into a local folder once (size and hash checked), then runs the listed steps one
after another, each in a process of its own, and copies each step's folder to ``outputs``. The encoder runs in the worker's own
Python, the rest in the pack's copy of the stored run's Python. Weights and packages already on the machine: nothing is fetched.

task.json: {"steps": [{"setup", "encoder", "cases", "tag"}], "smoke_codex": bool}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

LOCAL = Path(r"F:\uol_whole")
EXCHANGE = Path(r"exchange\UOL_GPU_EXCHANGE")


def unpack(want: dict) -> Path:
    root = LOCAL / "pack"
    flag = LOCAL / f"OK_{want['sha256'][:12]}"
    if flag.exists():
        return root
    src = EXCHANGE / "whole_pack" / "whole_pack.zip"
    print(time.strftime("%H:%M:%S"), "waiting for the pack", flush=True)
    t0 = time.time()
    while not src.exists() or src.stat().st_size != want["bytes"]:
        if time.time() - t0 > 6 * 3600:
            raise RuntimeError("the pack did not arrive")
        time.sleep(30)
    LOCAL.mkdir(parents=True, exist_ok=True)
    tmp = LOCAL / "whole_pack.zip"
    print(time.strftime("%H:%M:%S"), "copying the pack", flush=True)
    shutil.copyfile(src, tmp)
    h = hashlib.sha256()
    with tmp.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if h.hexdigest() != want["sha256"]:
        raise RuntimeError("the pack's hash differs")
    if root.exists():
        shutil.move(str(root), str(LOCAL / f"pack_{int(time.time())}"))
    with zipfile.ZipFile(tmp) as z:
        z.extractall(root)
    tmp.unlink()
    flag.write_text("ok", encoding="utf-8")
    print(time.strftime("%H:%M:%S"), "unpacked", flush=True)
    return root


def card() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as exc:
        return repr(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a, _ = ap.parse_known_args()
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    a.out.mkdir(parents=True, exist_ok=True)
    root = unpack(task["pack"])
    for name, rel in (task.get("overlay") or {}).items():     # code changed since the pack: files replaced in place
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.inputs / name, root / rel)
        print("overlay", rel, flush=True)
    (root / ".cache" / "whole_run" / "models").mkdir(parents=True, exist_ok=True)
    py = root / "py"
    npm = Path(os.environ.get("APPDATA", "")) / "npm"
    scratch = os.environ.get("UOL_WORKER_SCRATCH", r"C:\uol_gpu\scratch")
    base = {**os.environ, "PATH": os.pathsep.join([str(py), str(py / "Library" / "bin"), str(npm), os.environ.get("PATH", "")]), "PYTHONPATH": str(root / "src"), "PYTHONNOUSERSITE": "1", "PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0",
            "UOL_FINAL_THREADS": "4", "OMP_NUM_THREADS": "4", "UOL_FINAL_LABELS": "corrected", "UOL_FINAL_SYSTEMS": str(root / "experiments" / "runs" / "footing_20261101" / "systems_audit.json"),
            "UOL_NEW_BUNDLE_DATA": str(root / "bundle" / "UOL_FINAL_RUN" / "data"), "UOL_E13_BUNDLE_ROOT": str(root / "bundle" / "UOL_FINAL_RUN"), "UOL_WHOLE_WORK": str(LOCAL / "work"),
            "UOL_ENCODER_PYTHON": sys.executable, "UOL_ENCODER_SCRATCH": scratch, "UOL_WHOLE_PROCESSOR": subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True).stdout.strip()}
    base.pop("VIRTUAL_ENV", None)
    work = LOCAL / "work"
    (work / "models").mkdir(parents=True, exist_ok=True)
    for f in (root / ".cache" / "whole_run" / "models").rglob("*"):
        if f.is_file():
            d = work / "models" / f.relative_to(root / ".cache" / "whole_run" / "models")
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, d)
    if task.get("smoke_codex"):
        env = {**base, "UOL_LUNA_CACHE": str(LOCAL / f"cache_smoke_{int(time.time())}.sqlite")}
        code = ("from uol.final import llm_agents as G\nluna=G.pin_model()\nimport json\n"
                "schema={'type':'object','properties':{'ok':{'type':'string'}},'required':['ok'],'additionalProperties':False}\n"
                "r=luna.call('Responde con ok igual a si.',schema,effort='low',tag='whole_smoke',model=luna.MODEL)\nprint(json.dumps({'model':luna.MODEL,'output':r.get('output'),'usage':r.get('usage')}))\n")
        r = subprocess.run([str(py / "python.exe"), "-c", code], env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(LOCAL))
        (a.out / "smoke_codex.txt").write_text(r.stdout[-3000:] + "\n---\n" + r.stderr[-3000:], encoding="utf-8")
        print("codex smoke exit", r.returncode, r.stdout[-400:], flush=True)
    status = []
    for st in task["steps"]:
        name = f"{st['setup']}_{st['encoder']}_{st['tag']}"
        env = {**base, "UOL_LUNA_CACHE": str(LOCAL / f"cache_{name}.sqlite"), **(st.get("env") or {})}
        Path(env["UOL_LUNA_CACHE"]).unlink(missing_ok=True)
        print(time.strftime("%H:%M:%S"), "start", name, "| card", card(), flush=True)
        t0 = time.time()
        cmd = [str(py / "python.exe"), "-m", "uol.final.whole_run", "run", "--setup", st["setup"], "--encoder", st["encoder"], "--cases", str(st["cases"]), "--tag", st["tag"], "--go"]
        with (a.out / f"{name}.log").open("w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=str(LOCAL))
        src = work / "runs" / name
        if src.exists():
            shutil.copytree(src, a.out / name, dirs_exist_ok=True)
        status.append({"step": name, "exit": rc, "seconds": round(time.time() - t0, 1), "card_after": card()})
        (a.out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(time.strftime("%H:%M:%S"), name, "exit", rc, flush=True)
    return 0 if all(s["exit"] == 0 for s in status) else 1


if __name__ == "__main__":
    sys.exit(main())
