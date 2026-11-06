"""Builds the folder the whole run needs on another machine, from the files its traced runs opened.

    py -3.12 research/programs/whole_run_20261105/pack.py [--zip]

Data keep their paths under the project root, the bundle sits under ``bundle/``, and ``py/`` is a copy of this machine's
Python with the packages the run imported, so the remote run uses the software of the stored run. Stored scores are not in it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORK = ROOT / ".cache" / "whole_run"
PACK = WORK / "pack"
BUNDLE = "g:/my drive/uol_final_run/"
SP = Path(sys.base_prefix) / "Lib" / "site-packages"
SKIP_STD = {"test", "idlelib", "site-packages", "__pycache__", "ensurepip", "lib2to3", "turtledemo", "tkinter"}
EXTRA_DATA = ["research/paths.json", "experiments/runs/round2_20261004/B/historical_tesis.jsonl", ".cache/timing_stage_20261026c/MANIFEST.json"]
REFUSED = ("embedding_push_20261004/scores/", "/d58/x_control", "/d58/cache/top200_", "/footing_20261101/dev/store/runs/", ".cache/whole_run/runs/", ".cache/whole_run/pack/", "/luna/", "proto_cache")
# nothing the trace saw through a package's data, a temp folder or the interpreter
NOT_DATA = ("anaconda3", "/temp/", "__editable__", "__modules__", ".egg-info", "/windows/", "/.codex/")


def copy(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return src.stat().st_size


def data_files() -> dict[Path, Path]:
    out: dict[Path, Path] = {}
    for f in WORK.glob("opened_*_run.json"):
        for p in json.loads(f.read_text(encoding="utf-8")):
            q = p.replace("\\", "/")
            low = q.lower()
            if any(x in low for x in NOT_DATA) or not os.path.isfile(p) or low.endswith((".py", ".pyc")):
                continue
            if any(x in low for x in REFUSED) and "/luna/" not in low:
                continue
            if "/luna/" in low or "proto_cache" in low:
                continue
            if low.startswith(BUNDLE):
                out[Path(p)] = PACK / "bundle" / "UOL_FINAL_RUN" / q[len(BUNDLE):]
            elif low.startswith(str(ROOT).replace("\\", "/").lower() + "/"):
                out[Path(p)] = PACK / q[len(str(ROOT)) + 1:]
            else:
                raise RuntimeError(f"a file outside the project and the bundle: {p}")
    for rel in EXTRA_DATA:
        out[ROOT / rel] = PACK / rel
    return out


def runtime() -> int:
    base = Path(sys.base_prefix)
    py = PACK / "py"
    n = 0
    for f in base.glob("*"):
        if f.is_file() and f.suffix.lower() in (".exe", ".dll", ".zip"):
            n += copy(f, py / f.name)
    for sub in ("DLLs",):
        for f in (base / sub).rglob("*"):
            if f.is_file():
                n += copy(f, py / sub / f.relative_to(base / sub))
    for f in (base / "Lib").iterdir():
        if f.name in SKIP_STD:
            continue
        if f.is_file():
            n += copy(f, py / "Lib" / f.name)
        else:
            for g in f.rglob("*"):
                if g.is_file() and "__pycache__" not in g.parts:
                    n += copy(g, py / "Lib" / g.relative_to(base / "Lib"))
    mods = set()
    for f in WORK.glob("modules_*_run.json"):
        mods |= set(json.loads(f.read_text(encoding="utf-8")))
    mods |= {"numpy", "scipy", "pandas", "pyarrow", "sklearn", "lightgbm", "joblib", "threadpoolctl", "psutil", "nltk", "regex", "tqdm", "click", "pytz", "tzdata", "six", "dateutil", "numexpr", "bottleneck"}
    for m in sorted(mods):
        for cand in (SP / m, SP / f"{m}.py", SP / f"{m}.libs"):
            if cand.is_dir():
                for g in cand.rglob("*"):
                    if g.is_file() and "__pycache__" not in g.parts and "tests" not in g.parts:
                        n += copy(g, py / "Lib" / "site-packages" / g.relative_to(SP))
            elif cand.is_file():
                n += copy(cand, py / "Lib" / "site-packages" / cand.name)
    for g in SP.glob("*.pth"):
        pass                                                     # the editable installs of other projects stay out
    for g in SP.glob("*.pyd"):
        if g.name.split(".")[0] in mods:
            n += copy(g, py / "Lib" / "site-packages" / g.name)
    lib = base / "Library" / "bin"
    drop = ("qt", "opencv", "libclang", "llvm", "clang", "hdf", "gdal", "libtorch", "tcl", "tk8", "lib_ml", "mkl_pgi", "mkl_blacs", "mkl_scalapack", "mkl_cdft", "mkl_sequential", "mkl_tbb")
    for f in lib.glob("*.dll"):
        if not f.name.lower().startswith(drop):
            n += copy(f, py / "Library" / "bin" / f.name)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true")
    a = ap.parse_args()
    n = 0
    for src, dst in data_files().items():
        n += copy(src, dst)
    print(f"data {n / 1e9:.2f} GB", flush=True)
    code = 0
    for f in (ROOT / "src" / "uol").rglob("*"):
        if f.is_file() and f.suffix in (".py", ".txt", ".md", ".json") and "__pycache__" not in f.parts:
            code += copy(f, PACK / f.relative_to(ROOT))
    for f in (WORK / "models").rglob("*"):
        if f.is_file():
            code += copy(f, PACK / ".cache" / "whole_run" / "models" / f.relative_to(WORK / "models"))
    print(f"code and models {code / 1e6:.1f} MB", flush=True)
    r = runtime()
    print(f"runtime {r / 1e9:.2f} GB", flush=True)
    if a.zip:
        out = WORK / "whole_pack.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
            for f in PACK.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(PACK).as_posix())
        h = hashlib.sha256()
        with out.open("rb") as fh:
            for b in iter(lambda: fh.read(1 << 22), b""):
                h.update(b)
        (WORK / "whole_pack.json").write_text(json.dumps({"bytes": out.stat().st_size, "sha256": h.hexdigest()}, indent=1), encoding="utf-8")
        print(out, out.stat().st_size / 1e9, h.hexdigest(), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
