"""Build the code + data bundle that lets original ``uol.embedding_push`` GPU training run on the remote worker.

What travels (prompt §8): code, public article text, frozen article/dev-query vectors and DEV labels (training-fold
labels for training jobs). What never travels: lockbox texts or labels, credentials, the project. The bundle keeps the
release's row order (so every row index is unchanged) but blanks every lockbox query (text, record id, court, subjects)
and empties its references; lockbox rows of shipped query-vector files are zeroed.

    python -m uol.round3.remote_bundle        # builds (or reuses) the bundle and prints its manifest
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments/runs/round3_best_case_20261013/remote_bundle"
RELEASE = ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified"
REL_NAME = "data/releases/uol_c01_statutes_20261004_v11_stratified"
SPLIT = ROOT / "research/state/embedding_push_20261004/SPLIT.json"
DERIVED = ROOT / "data/derived/facts_to_law_v2"
CODE_PACKAGES = ("uol/__init__.py", "uol/embedding_push", "uol/facts_to_law")
CODE_FILES_EXTRA = ("uol/round3/__init__.py", "uol/paths.py", "uol/orchestration/__init__.py",
                    "uol/orchestration/embedding_usefulness/__init__.py", "uol/orchestration/embedding_usefulness/encoder.py",
                    "uol/retrieval_benchmarks/__init__.py", "uol/retrieval_benchmarks/common.py",
                    "uol/retrieval_benchmarks/c01_statutes/__init__.py", "uol/retrieval_benchmarks/c01_statutes/experiments.py",
                    "uol/retrieval_benchmarks/refinement_suite/__init__.py",
                    "uol/retrieval_benchmarks/refinement_suite/nonparametric.py")  # import closure (closure.json), no credentials
# frozen vector files the GPU trainers read (FROZEN_FILES in finetune_dense; case_sim reads only texts + labels)
VECTOR_FILES = {
    "e5_small_chunks_v1/chunk_vectors.npy": "article", "e5_small_chunks_v1/article_index.npy": "article",
    "e5_small_chunks_v1/ids.json": "article", "e5_small_queries_v11/query_vectors.npy": "query",
    # colbert / long bge (public article chunks; dev query CLS vectors for the encode check)
    "bge_m3_chunks_v1/chunk_vectors.npy": "article", "bge_m3_chunks_v1/units.jsonl": "article",
    "bge_m3_chunks_v1/receipt.json": "article", "bge_m3_chunks_v1/query_vectors.npy": "query",
}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def lockbox_qids() -> set[str]:
    assignment = json.loads(SPLIT.read_text(encoding="utf-8"))["assignment"]
    return {q for q, role in assignment.items() if role == "lockbox"}


def lockbox_row_mask() -> np.ndarray:
    lb = lockbox_qids()
    rows = []
    with (RELEASE / "queries.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line)["query_id"] in lb)
    return np.asarray(rows, dtype=bool)


def sanitized_release() -> dict[str, bytes]:
    lb = lockbox_qids()
    out: dict[str, bytes] = {}
    q_lines, r_lines, n_blank = [], [], 0
    for line in (RELEASE / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["query_id"] in lb:
            n_blank += 1
            rec["text"], rec["text_sha256"] = "", sha256_bytes(b"")
            rec["record_id"] = f"redacted-{n_blank}"
            rec["instancia"], rec["materias"] = "", []
        q_lines.append(json.dumps(rec, ensure_ascii=False))
    for line in (RELEASE / "references.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec["query_id"] in lb:
            rec = {"query_id": rec["query_id"], "positives": [], "ignorable": [], "out_of_scope": [], "unresolved": [],
                   "version_flags": []}
        r_lines.append(json.dumps(rec, ensure_ascii=False))
    out[f"{REL_NAME}/queries.jsonl"] = ("\n".join(q_lines) + "\n").encode("utf-8")
    out[f"{REL_NAME}/references.jsonl"] = ("\n".join(r_lines) + "\n").encode("utf-8")
    for name in ("folds.json", "target_types.jsonl"):
        out[f"{REL_NAME}/{name}"] = (RELEASE / name).read_bytes()
    manifest = json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))
    for key in ("articles", "instruments"):
        src = Path(manifest["candidates"][key]["path"])
        out[f"{REL_NAME}/candidates/{src.name}"] = src.read_bytes()
        manifest["candidates"][key]["path"] = f"candidates/{src.name}"
    manifest["round3_remote_note"] = f"lockbox rows blanked ({n_blank}); not the frozen release; hashes not verifiable"
    out[f"{REL_NAME}/manifest.json"] = json.dumps(manifest, indent=1, ensure_ascii=False).encode("utf-8")
    out["research/state/embedding_push_20261004/SPLIT.json"] = SPLIT.read_bytes()
    # uol.paths needs the project path-alias map (paths only; no data, no credentials)
    alias = ROOT / "research/paths.json"
    alias = alias if alias.exists() else ROOT / "provenance/reorganization/moves.json"
    out[alias.relative_to(ROOT).as_posix()] = alias.read_bytes()
    mask = lockbox_row_mask()
    for rel, kind in VECTOR_FILES.items():
        p = DERIVED / rel
        if kind == "query":
            arr = np.load(p)
            if arr.shape[0] != len(mask):
                raise ValueError(f"{rel}: rows {arr.shape[0]} != release rows {len(mask)}")
            arr = arr.copy()
            arr[mask] = 0
            import io
            buf = io.BytesIO()
            np.save(buf, arr)
            out[f"data/derived/facts_to_law_v2/{rel}"] = buf.getvalue()
        else:
            out[f"data/derived/facts_to_law_v2/{rel}"] = p.read_bytes()
    return out


def code_files() -> dict[str, bytes]:
    src = ROOT / "src"
    out = {}
    closure = OUT / "closure.json"     # import closure measured after loading the release (lazy imports included)
    extra = tuple(json.loads(closure.read_text())) if closure.exists() else ()
    for item in CODE_PACKAGES + CODE_FILES_EXTRA + extra:
        p = src / item
        files = [p] if p.is_file() else sorted(x for x in p.rglob("*.py") if "__pycache__" not in x.parts)
        for f in files:
            out["src/" + f.relative_to(src).as_posix()] = f.read_bytes()
    return out


def build(force: bool = False) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = {"code": code_files(), "data": sanitized_release()}
    manifest = {}
    for name, files in parts.items():
        digest = sha256_bytes(b"".join(k.encode() + sha256_bytes(v).encode() for k, v in sorted(files.items())))[:16]
        path = OUT / f"{name}_{digest}.zip"
        if force or not path.exists():
            tmp = path.with_suffix(".tmp")
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
                for k, v in sorted(files.items()):
                    z.writestr(k, v)
            tmp.replace(path)
        manifest[name] = {"path": str(path), "digest": digest, "files": len(files), "bytes": path.stat().st_size}
    # self-check: nothing lockbox-identifying in the data part
    lb = lockbox_qids()
    q = parts["data"][f"{REL_NAME}/queries.jsonl"].decode("utf-8").splitlines()
    leaked = [json.loads(x)["query_id"] for x in q if json.loads(x)["query_id"] in lb and json.loads(x)["text"]]
    refs = parts["data"][f"{REL_NAME}/references.jsonl"].decode("utf-8").splitlines()
    leaked += [json.loads(x)["query_id"] for x in refs if json.loads(x)["query_id"] in lb and json.loads(x)["positives"]]
    if leaked:
        raise RuntimeError(f"lockbox content in the bundle: {leaked[:3]}")
    manifest["lockbox_rows_blanked"] = len(lb)
    (OUT / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(build("--force" in sys.argv), indent=1))
