"""Run the new cases' query-side job (E13) from this staged folder; the run computer's code tree stays as it is.

    python run_new.py verify --dir <staged folder>
    python run_new.py encode --dir <staged folder> --out <output folder>
    python run_new.py results-manifest --out <output folder>

``encode`` runs ``new_encode.py`` for bge, qwen, pairs and gate, one process each, and stops at the first failure.
Exit code 0: done; 2: the control gate failed (a result, not an error); anything else: a failure.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOB = HERE / "src" / "uol" / "final" / "gpu_jobs" / "new_encode.py"
STEPS = ("bge", "qwen", "pairs", "gate")
KEEP_HERE = ("bge.npz", "bge_chunks.npz")      # token vectors and the chunk plan stay on this machine (sha256 recorded)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(d: Path) -> int:
    man = json.loads((d / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [rel for rel, h in man["files"].items() if not (d / rel).exists() or sha256(d / rel) != h]
    extra = sorted(str(p.relative_to(d)).replace("\\", "/") for p in d.rglob("*")
                   if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts
                   and str(p.relative_to(d)).replace("\\", "/") not in man["files"])
    print(json.dumps({"files": len(man["files"]), "mismatch": bad, "unlisted": extra, "pass": not bad}, indent=1))
    return 0 if not bad else 1


def encode(d: Path, out: Path) -> int:
    if out.exists() and any(out.iterdir()):
        print(f"STOP: {out} is not empty; never write into an earlier output folder")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        t0 = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"{t0} start {step}", flush=True)
        with (out / f"{step}.log").open("w", encoding="utf-8") as fh:
            code = subprocess.run([sys.executable, str(JOB), "--model", step, "--dir", str(d), "--out", str(out)],
                                  stdout=fh, stderr=subprocess.STDOUT).returncode
        print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} end {step}: exit {code}", flush=True)
        if code == 2 and step == "gate":
            print("NEW CASES GATE FAILED (see GATE.json)")
            return 2
        if code != 0:
            print(f"{step} failed with exit {code}; see {out / (step + '.log')}")
            return code
    print("encode done; GATE.json pass")
    return 0


def results_manifest(out: Path) -> int:
    files = {str(p.relative_to(out)).replace("\\", "/"): {"sha256": sha256(p), "bytes": p.stat().st_size,
                                                          "to_drive": p.name not in KEEP_HERE}
             for p in sorted(out.rglob("*")) if p.is_file() and p.name != "RESULTS_MANIFEST.json"}
    (out / "RESULTS_MANIFEST.json").write_text(json.dumps({"files": files, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                                             time.gmtime())}, indent=1),
                                               encoding="utf-8")
    print(json.dumps({"files": len(files), "to_drive": sum(v["to_drive"] for v in files.values())}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["verify", "encode", "results-manifest"])
    ap.add_argument("--dir", type=Path, default=HERE)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.command == "verify":
        return verify(a.dir)
    if a.out is None:
        ap.error("--out is required")
    return encode(a.dir, a.out) if a.command == "encode" else results_manifest(a.out)


if __name__ == "__main__":
    sys.exit(main())
