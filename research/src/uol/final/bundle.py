"""The portable, development-only bundle for the run computer (scope change of 21 Oct).

    py -3.12 -m uol.round3.heavy uol.final bundle --out .cache/final_bundle_staging
    py -3.12 -m uol.final bundle --out <dir> --smoke          # small: three sources, no large vectors

Layout::

    <out>/code/src/uol/...        the import closure of uol.final (and the GPU job scripts), unchanged files
    <out>/code/tests/             tests/final and the offline conftest
    <out>/code/requirements.lock  pinned CPU stack of the thesis PC (torch: CUDA 12.8 build, GPU jobs only)
    <out>/data/                   uol.final.context.BUNDLE_LAYOUT; release rows = development rows in v11 order
    <out>/data/BUNDLE.json        dev_only, row map (bundle row -> v11 row), counts, cut record
    <out>/setup/setup_runner.ps1  venv in C:\\uol_final, copy + hash check, GPU check, tests
    <out>/jobs/plan.json          the overnight DAG (uol.final.plan)
    <out>/RUNNER_README.md        human steps on the run computer
    <out>/MANIFEST.json           sha256 and size of every file (packed arrays also list the unpacked sha256)

Development rows only. Every exported row set passes ``guard.dev_rows_only``; at the end ``scan`` re-reads every
text file for test-set query or record ids and checks every array's row count, and refuses the bundle otherwise.
Large arrays are byte-shuffled zstd (``packing``); ``setup_runner.ps1`` unpacks them and checks the sha256.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from uol.final import context, guard, packing

SRC = Path(__file__).resolve().parents[2]                       # .../src
ROOT = SRC.parent
PACK_MIN_BYTES = 8 << 20                                        # arrays above 8 MB are packed
CLOSURE_MODULES = (
    "uol.final.__main__", "uol.final.harness", "uol.final.g1", "uol.final.report", "uol.final.cost", "uol.final.stats",
    "uol.final.bundle", "uol.final.plan", "uol.final.overnight", "uol.final.packing",
    "uol.final.adapters.loaded", "uol.final.adapters.dense", "uol.final.adapters.lexical", "uol.final.adapters.topic",
    "uol.final.adapters.fusion", "uol.final.adapters.reference", "uol.final.adapters.ltr", "uol.final.adapters.llm",
    "uol.final.adapters.gpu",
    "uol.embedding_push.base", "uol.embedding_push.fusion_ltr", "uol.round3.features", "uol.round3.c_hybrid",
    "uol.round3.e2_adapter", "uol.round3.m1_memory", "uol.round3.stats", "uol.round3.heavy", "uol.gap_tests.d58_refit",
    "uol.gap_tests.lexical_grid", "uol.gap_tests.ltr_listwise", "uol.scale_curve.legal", "uol.scale_curve.evaluate",
    "uol.round2.a_p12", "uol.final.codex", "uol.final.llm_jobs", "uol.final.llm_cost", "uol.final.qwen_local", "uol.final.rebuild",
    "uol.final.variants", "uol.final.devn", "uol.final.devb", "uol.final.gpu_prep", "uol.final.rerank_jobs", "uol.final.pack",
    "uol.final.new_cases", "uol.final.setup_files", "uol.final.assign", "uol.final.merge", "uol.final.c2", "uol.final.devl", "uol.final.devs", "uol.final.external",
    "uol.round2.h_common", "uol.round2.glm_client", "uol.final.adapters.adapted",
    "uol.round3.luna", "uol.round3.prompt_guard", "uol.round3.l1_luna", "uol.round3.l2_features", "uol.round3.catalog_map",
    "uol.round3.r2_cross", "uol.round3.r1_zeroshot", "uol.round3.rebuild", "uol.round3.arm_runner",
    "uol.eval_lenses.llm_alone", "uol.eval_lenses.e6_agent", "uol.eval_lenses.core", "uol.thesis_figures.common",
    "uol.embedding_push.qwen_endpoint", "uol.embedding_push.memory_proto", "uol.embedding_push.case_sim",
    "uol.embedding_push.finetune_dense", "uol.embedding_push.colbert",
    "uol.facts_to_law.references_v2", "uol.facts_to_law.strata", "uol.facts_to_law.release", "uol.facts_to_law.dense",
    "uol.retrieval_benchmarks.benchmark_repair.masking", "uol.round2.c_collect", "uol.round2.c_release", "uol.round2.c_sources",
)
EXTRA_FILES = ("uol/round3/gpu_jobs/legal_encode.py", "uol/round3/gpu_jobs/rerank_bench.py", "uol/scale_curve/store.py",
               "uol/round3/gpu_jobs/r2_train.py", "uol/round3/gpu_jobs/r1_score.py",       # smoke__zs, gpu__rerank_*
               "uol/facts_to_law/bge_gpu.py",                                              # devb_dense_queries
               "uol/final/gpu_jobs/bge_m3_sparse.py", "uol/final/gpu_jobs/st_encode.py",
               "uol/final/gpu_jobs/bge_m3_ft.py", "uol/final/gpu_jobs/__init__.py", "uol/final/systems.json",
               "uol/final/prompts/hyde.txt", "uol/round3/prompts/closedbook_rank20.txt", "uol/round3/prompts/L1.txt",
               "uol/round3/prompts/L2_a.txt", "uol/round3/prompts/L2_b.txt", "uol/eval_lenses/prompts/E6_agent.txt")
REQUIREMENTS = """# uol.final run computer: CPU stack pinned to the thesis PC versions that reproduce G1 (Python 3.12, Windows)
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.7.2
pandas==2.2.2
pyarrow==16.1.0
gensim==4.3.3
lightgbm==4.7.0
nltk==3.9.1
psutil==5.9.0
zstandard==0.23.0
requests==2.32.3
threadpoolctl==3.5.0
joblib==1.4.2
smart-open==5.2.1
python-dateutil==2.9.0.post0
pytz==2026.1.post1
tzdata==2023.3
six==1.16.0
urllib3==2.2.3
idna==3.7
certifi==2025.8.3
charset-normalizer==3.3.2
wrapt==1.14.1
regex==2026.4.4
click==8.3.1
tqdm==4.66.5
packaging==24.1
pytest==9.0.2
iniconfig==1.1.1
pluggy==1.6.0
colorama==0.4.6
pygments==2.15.1
# in-process torch stages of the rebuild (finetune_dense, case_sim), the local int8 encoder, figures, tables, collector
transformers==4.57.6
tokenizers==0.22.2
huggingface-hub==0.36.2
safetensors==0.7.0
sentence-transformers==5.1.2
peft==0.17.1
accelerate==1.10.1
matplotlib==3.9.2
pillow==10.4.0
kiwisolver==1.4.4
fonttools==4.51.0
cycler==0.11.0
contourpy==1.2.0
pyparsing==3.1.2
jinja2==3.1.4
markupsafe==2.1.3
pyyaml==6.0.1
filelock==3.13.1
fsspec==2024.6.1
selenium==4.29.0
# GPU jobs only (setup_runner.ps1 installs it from https://download.pytorch.org/whl/cu128, RTX 50-series / sm_120):
# torch  (CUDA 12.8 build; the installed version is recorded in setup.log)
# The GPU job scripts install their own pins privately at run time (round3/gpu_jobs/rerank_bench.py PINS:
# transformers 5.17.0, tokenizers 0.23.2, huggingface_hub 1.32.0, sentence-transformers 6.1.0, bitsandbytes 0.50.2,
# accelerate 1.15.0).
"""


class Exporter:
    def __init__(self, out: Path, compress: bool = True):
        from uol.embedding_push import common as ep
        self.out = Path(out)
        self.data = self.out / "data"
        self.compress = compress
        self.ep = ep
        self.d = ep.release()
        self.dev = guard.dev_rows_only(ep.dev_rows(), "bundle rows")
        self.new = np.full(len(self.d.queries), -1, dtype=np.int64)
        self.new[self.dev] = np.arange(len(self.dev))
        self.dev_qids = [self.d.queries[int(r)].query_id for r in self.dev]
        self.lock_qids = {self.d.queries[int(r)].query_id for r in ep.lockbox_rows()}
        self.lock_records = {str(self.d.queries[int(r)].record_id) for r in ep.lockbox_rows()}
        self.all_qids = {q.query_id for q in self.d.queries}
        self.packed: dict[str, dict[str, Any]] = {}
        self.log: list[str] = []

    # ------------------------------------------------------------------ helpers
    def note(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log.append(line)

    def path(self, key: str) -> Path:
        return self.data / context.BUNDLE_LAYOUT[key]

    def remap(self, rows: Iterable[int]) -> np.ndarray:
        rows = guard.dev_rows_only(rows, "exported rows")
        return self.new[rows]

    def write_json(self, path: Path, obj: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")

    def save_array(self, path: Path, arr: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(path)
        self._maybe_pack(path)

    def copy_file(self, src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if dst.suffix == ".npy":
            self._maybe_pack(dst)

    def _maybe_pack(self, path: Path) -> None:
        if self.compress and path.stat().st_size >= PACK_MIN_BYTES:
            info = packing.pack(path)
            path.unlink()
            self.packed[path.relative_to(self.out).as_posix() + packing.SUFFIX] = {
                "unpacked": path.relative_to(self.out).as_posix(), "unpacked_sha256": info["sha256"],
                "unpacked_bytes": info["bytes_raw"]}

    def slice_rows(self, src: Path, dst: Path, rows: np.ndarray | None = None, block: int = 512) -> str:
        """[release rows, ...] array -> its development rows (bundle order); returns the sha256 of the written .npy."""
        mm = np.load(src, mmap_mode="r")
        rows = self.dev if rows is None else rows
        rows = guard.dev_rows_only(rows, f"slice {src.name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp.npy")
        out = np.lib.format.open_memmap(tmp, mode="w+", dtype=mm.dtype, shape=(len(rows),) + mm.shape[1:])
        for s in range(0, len(rows), block):
            out[s:s + block] = mm[rows[s:s + block]]
        out.flush()
        del out
        tmp.replace(dst)
        sha = self.ep.sha256_file(dst)
        self._maybe_pack(dst)
        return sha

    # ------------------------------------------------------------------ release
    def release(self) -> None:
        rel = self.path("release")
        rel.mkdir(parents=True, exist_ok=True)
        src = self.d.release_dir
        keep = set(self.dev_qids)
        for name in ("queries.jsonl", "references.jsonl"):
            n = 0
            with (src / name).open(encoding="utf-8") as f, (rel / name).open("w", encoding="utf-8", newline="\n") as g:
                for line in f:
                    if not line.strip():
                        continue
                    if json.loads(line)["query_id"] in keep:
                        g.write(line if line.endswith("\n") else line + "\n")
                        n += 1
            if n != len(keep):
                raise RuntimeError(f"{name}: exported {n} rows, expected {len(keep)}")
        comps = {self.d.queries[int(r)].component_id for r in self.dev}
        folds = json.loads((src / "folds.json").read_text(encoding="utf-8"))
        folds["components"] = {c: f for c, f in folds["components"].items() if c in comps}
        self.write_json(rel / "folds.json", folds)
        with (src / "target_types.jsonl").open(encoding="utf-8") as f, \
                (rel / "target_types.jsonl").open("w", encoding="utf-8", newline="\n") as g:
            for line in f:
                if line.strip():
                    r = json.loads(line)                    # counts over all queries are dropped (typing kept)
                    g.write(json.dumps({"candidate_id": r["candidate_id"], "rule": r["rule"], "target_type": r["target_type"],
                                        "stats": {}}, ensure_ascii=False) + "\n")
        art = Path(self.d.candidate_binding["articles_path"])
        self.copy_file(art, rel / "candidates" / "articles.jsonl")
        if (art.parent / "instruments.jsonl").exists():
            self.copy_file(art.parent / "instruments.jsonl", rel / "candidates" / "instruments.jsonl")
        man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        new = {"release_id": man["release_id"], "format_version": man["format_version"], "policies": man["policies"],
               "status": "DEVELOPMENT-ONLY EXPORT for the final benchmark run (uol.final.bundle); no test-set row",
               "source_release_manifest_sha256": self.ep.sha256_file(src / "manifest.json"),
               "candidates": {"articles": {"path": "candidates/articles.jsonl",
                                           "sha256": self.ep.sha256_file(rel / "candidates/articles.jsonl")},
                              "count": len(self.d.candidate_ids)},
               "files": {n: self.ep.sha256_file(rel / n) for n in ("folds.json", "queries.jsonl", "references.jsonl",
                                                                    "target_types.jsonl")},
               "counts": {"queries": len(self.dev), "components": len(comps)}}
        self.write_json(rel / "manifest.json", new)
        split = json.loads(self.ep.SPLIT_PATH.read_text(encoding="utf-8"))
        roles = self.ep.roles()
        self.write_json(self.path("split"), {
            "study": split.get("study"), "release": "release (development-only export)", "rule": split.get("rule"),
            "cut_record_id": split.get("cut_record_id"), "dev_only": True,
            "assignment": {self.d.queries[int(r)].query_id: str(roles[int(r)]) for r in self.dev}})
        self.dev_sources()
        self.note(f"release: {len(self.dev)} development rows, {len(comps)} components")

    def dev_sources(self) -> dict[str, int]:
        """Structured sources of development cases only: the released development rows and the sources v11 excluded
        that were published before the test period (record < cut, not a lockbox query): publication dates, the prompt
        guard's metadata, and the inputs of the DEV-B resolver patches."""
        from uol.round3 import common as R3
        keep_q = set(self.dev_qids)
        n = {"released_dev": 0, "excluded_dev": 0, "refused": 0}
        dst = self.path("sources_jsonl")
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Path(R3.SOURCES_JSONL).open(encoding="utf-8") as f, dst.open("w", encoding="utf-8", newline="\n") as g:
            for line in f:
                if not line.strip():
                    continue
                s = json.loads(line)
                q, rid = s.get("query_id"), int(s.get("record_id") or 10 ** 9)
                if q in keep_q:
                    n["released_dev"] += 1
                elif q not in self.lock_qids and str(rid) not in self.lock_records and rid < guard.CUT_RECORD_ID \
                        and q not in self.all_qids:
                    n["excluded_dev"] += 1
                else:
                    n["refused"] += 1
                    continue
                g.write(line if line.endswith("\n") else line + "\n")
        self.note(f"dev sources: {n}")
        return n

    def round3_inputs(self) -> None:
        """M1 (historical memory: neighbours, selected configs, memory.jsonl), E2 selection, L2 issue decompositions,
        the dev-only Luna cache, the label corrections and the scripts the run computer re-runs."""
        from uol.final import codex, devb
        arms = ROOT / "experiments/runs/round3_best_case_20261013/arms"
        dst = self.data / "round3" / "arms"
        for key in ("dense", "lex"):
            p = arms / "M1" / "cache" / f"neighbours_{key}.npz"
            z = np.load(p)
            arr = {k: z[k] for k in z.files}
            arr["rows"] = self.remap(arr["rows"]).astype(np.int64)
            (dst / "M1" / "cache").mkdir(parents=True, exist_ok=True)
            np.savez(dst / "M1" / "cache" / p.name, **arr)
        for arm in ("M1", "E2_adapter"):
            r = json.loads((arms / arm / "RESULT.json").read_text(encoding="utf-8"))
            self.write_json(dst / arm / "RESULT.json", {"selected_config": r["selected_config"],
                                                        "note": "selection only (dev); the rest of RESULT.json stays home"})
        for v in ("a", "b"):
            src = arms / "L2" / "luna" / f"L2_{v}.jsonl"
            out = dst / "L2" / "luna" / src.name
            out.parent.mkdir(parents=True, exist_ok=True)
            with src.open(encoding="utf-8") as f, out.open("w", encoding="utf-8", newline="\n") as g:
                for line in f:
                    r = json.loads(line)
                    r["row"] = int(self.remap([r["row"]])[0])
                    g.write(json.dumps(r, ensure_ascii=False) + "\n")
        hist = ROOT / "experiments/runs/round2_20261004/B/memory.jsonl"
        self.copy_file(hist, self.path("historical") / "memory.jsonl")
        info = codex.export_cache(ROOT / ".cache/round3_luna/cache.sqlite", self.data / "luna" / "cache_dev.sqlite",
                                  self.lock_records)
        self.note(f"luna cache: {info}")
        from uol.final import context as CX
        self.write_json(self.data / "label_corrections.json", CX.LABEL_CORRECTIONS)
        scripts = ROOT / "research/reports/data_soundness_20261020/scripts"
        self.copy_file(scripts / "residue.py", self.data / "scripts" / "residue.py")
        port = devb.port_script(scripts / "gaps" / "g4_patched.py", self.data / "scripts" / "gaps" / "g4_patched.py")
        port_b = devb.port_bindfix(scripts / "bindfix.py", self.data / "scripts" / "bindfix_defs.py")
        self.write_json(self.data / "scripts" / "gaps" / "PORTING.json", {"g4_patched": port, "bindfix": port_b})

    # ------------------------------------------------------------------ frozen sources and pipelines
    def scores(self, smoke: bool) -> dict[str, str]:
        from uol.embedding_push import fusion_ltr as fl
        ep_run = self.path("ep_run")
        meta = json.loads((fl.FEATDIR / "meta.json").read_text(encoding="utf-8"))
        self.write_json(ep_run / "fusion/features/meta.json", meta)
        ids = [s["id"] for s in meta["sources"] if s["kind"] == "full"]
        if smoke:
            ids = [i for i in ids if i in ("L_star", "dense_qwen8b", "prior")]
        shas: dict[str, str] = {}
        for i, sid in enumerate(ids):
            src = self.ep.score_path(sid)
            dst = ep_run / "scores" / sid / "scores.npy"
            shas[f"scores/{sid}/scores.npy"] = self.slice_rows(src, dst)
            m = json.loads((src.parent / "manifest.json").read_text(encoding="utf-8"))
            m.update({"bundle_dev_only": True, "bundle_scores_sha256": shas[f"scores/{sid}/scores.npy"],
                      "bundle_rows": "development rows of v11 in release order"})
            self.write_json(dst.parent / "manifest.json", m)
            self.note(f"source {i + 1}/{len(ids)} {sid}")
        if not smoke:
            src = self.ep.SCORES / "colbert_bge"
            z = np.load(src / "pairs.npz")
            q = z["qidx"].astype(np.int64)
            keep = np.isin(q, self.dev)
            arrs = {k: z[k][keep] for k in z.files}
            arrs["qidx"] = self.new[q[keep]].astype(np.int32)
            (ep_run / "scores/colbert_bge").mkdir(parents=True, exist_ok=True)
            np.savez(ep_run / "scores/colbert_bge/pairs.npz", **arrs)
            m = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
            m.update({"bundle_dev_only": True, "pairs_sha256": self.ep.sha256_file(ep_run / "scores/colbert_bge/pairs.npz")})
            self.write_json(ep_run / "scores/colbert_bge/manifest.json", m)
        for name in ("LTR_noemb_hgb", "LTR_emb_hgb", "LTR_noemb_hgb_full", "LTR_emb_hgb_full", "LTR_noemb_linear",
                     "LTR_emb_linear"):
            src = fl.pipeline_path(name)
            man = json.loads((src.parent / "manifest.json").read_text(encoding="utf-8"))
            for role in list(man.get("configuration", {})):
                if role == "lockbox":
                    man["configuration"].pop(role)
            if smoke and name not in ("LTR_noemb_hgb", "LTR_emb_hgb"):
                continue
            key = f"fusion/pipelines/{name}/scores.npy"
            if not src.exists():                         # some pipelines kept only their manifest after the rebuild
                man["bundle_note"] = "scores.npy not present on the thesis PC; manifest (hyper-parameters) only"
            elif name in ("LTR_noemb_hgb", "LTR_emb_hgb") or not smoke:
                shas[key] = self.slice_rows(src, ep_run / key)
                man["bundle_scores_sha256"] = shas[key]
            self.write_json(ep_run / f"fusion/pipelines/{name}/manifest.json", man)
        for sub in ("base/selection", "qwen/selection"):
            for p in sorted((self.ep.RUN / sub).glob("*.json")):
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    obj.pop("lockbox", None)
                    if isinstance(obj.get("pick"), dict):
                        obj["pick"].pop("lockbox", None)
                    if isinstance(obj.get("best"), dict):
                        for fam in obj["best"].values():
                            if isinstance(fam, dict):
                                fam.pop("lockbox", None)
                self.write_json(ep_run / sub / p.name, obj)
        return shas

    # ------------------------------------------------------------------ dev blocks, vectors, C_hybrid, units
    def dev_blocks(self) -> dict[str, str]:
        src = context.RELEASES["v11"]["d58_sources"]
        shas = {}
        for rel in ("a/L_star.npy", "b/dense_qwen8b.npy"):
            p = ROOT / src / rel
            if p.exists():
                arr = np.load(p, mmap_mode="r")
                if arr.shape[0] != len(self.dev):
                    raise RuntimeError(f"{p}: {arr.shape[0]} rows, expected {len(self.dev)} development rows")
                dst = self.path("d58_sources") / rel
                self.copy_file(p, dst)
                shas[rel] = self.ep.sha256_file(p)
        return shas

    def vectors(self, smoke: bool) -> None:
        R = context.RELEASES["v11"]
        q8 = ROOT / R["qwen8b"]
        dst = self.path("qwen8b")
        for f in ("article_vectors.npy", "article_index.npy"):
            if not smoke or f == "article_index.npy":
                self.copy_file(q8 / f, dst / f)
        for v in ("a", "b", "c"):
            if not smoke or v == "c":
                self.slice_rows(q8 / f"query_vectors_{v}.npy", dst / f"query_vectors_{v}.npy")
        self.write_json(dst / "query_ids.json", self.dev_qids)
        if smoke:
            return
        for key in ("qwen3_0p6b", "qwen3_4b"):
            s, t = ROOT / R[key], self.path(key)
            for f in ("article_vectors.npy", "article_index.npy", "manifest.json"):
                self.copy_file(s / f, t / f)
            rows = json.loads((s / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
            for p in sorted(s.glob("query_vectors_*.npy")):
                self.copy_file(p, t / p.name)
            self.write_json(t / "query_rows.json", {"release_rows": self.remap(rows).tolist()})
        s, t = ROOT / R["qwen8b_headings"], self.path("qwen8b_headings")
        for f in ("article_vectors.npy", "article_index.npy"):
            if (s / f).exists():
                self.copy_file(s / f, t / f)
        s, t = ROOT / R["e5_large_legal"], self.path("e5_large_legal")
        self.copy_file(s / "chunk_vectors.npy", t / "chunk_vectors.npy")
        self.copy_file(s / "query_vectors_dev.npy", t / "query_vectors_dev.npy")
        self.save_array(t / "query_rows_dev.npy", self.remap(np.load(s / "query_rows_dev.npy")))
        self.copy_file(ROOT / R["derived_ftl"] / "chunks_e5_small_v1/chunks.jsonl",
                       self.path("derived_ftl") / "chunks_e5_small_v1/chunks.jsonl")
        self.copy_file(ROOT / R["derived_ftl"] / "chunks_e5_small_v1/manifest.json",
                       self.path("derived_ftl") / "chunks_e5_small_v1/manifest.json")
        ftl, tftl = ROOT / R["derived_ftl"], self.path("derived_ftl")
        for sub, files, qfile in (("bge_m3_chunks_v1", ("chunk_vectors.npy", "units.jsonl", "receipt.json"), "query_vectors.npy"),
                                  ("e5_small_chunks_v1", ("chunk_vectors.npy", "ids.json", "article_index.npy", "chunk_ordinal.npy",
                                                         "manifest.json"), None),   # base.stage_dense, facts_to_law.dense.load
                                  ("e5_small_queries_v11", (), "query_vectors.npy")):
            for f in files:
                if (ftl / sub / f).exists():
                    self.copy_file(ftl / sub / f, tftl / sub / f)
            if qfile:
                Q = np.load(ftl / sub / qfile, mmap_mode="r")
                if Q.shape[0] != len(self.d.queries):
                    raise RuntimeError(f"{sub}/{qfile}: not one row per release row")
                self.slice_rows(ftl / sub / qfile, tftl / sub / qfile)
        self.note("vectors exported")

    def c_hybrid(self) -> None:
        s = ROOT / context.RELEASES["v11"]["c_hybrid"]
        t = self.path("c_hybrid")
        rows = np.load(s / "stage2_pspec_rows.npy")
        self.save_array(t / "stage2_pspec_rows.npy", self.remap(rows))
        self.copy_file(s / "stage2_pspec_matrix.npy", t / "stage2_pspec_matrix.npy")
        for k in range(3):
            o = json.loads((s / f"orders_s{k}.json").read_text(encoding="utf-8"))
            o["rows"] = self.remap(o["rows"]).tolist()
            o["orders"] = {p: {str(int(self.remap([int(r)])[0])): v for r, v in d.items()} for p, d in o["orders"].items()}
            self.write_json(t / f"orders_s{k}.json", o)
        self.note("C_hybrid cached inputs exported")

    def units_and_gpu(self, smoke: bool) -> None:
        from uol.final.adapters import gpu
        s = ROOT / context.RELEASES["v11"]["units"]
        t = self.path("units")
        t.mkdir(parents=True, exist_ok=True)
        with (s / "units.jsonl").open(encoding="utf-8") as f, (t / "units.jsonl").open("w", encoding="utf-8", newline="\n") as g:
            for line in f:
                r = json.loads(line)
                if r["g"] != "articles":
                    r["o"] = int(self.remap([r["o"]])[0])
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
        man = json.loads((s / "units_manifest.json").read_text(encoding="utf-8"))
        man.update({"sha256": self.ep.sha256_file(t / "units.jsonl"), "bundle_rows": "query owners are bundle rows",
                    "path": "units/units.jsonl"})
        self.write_json(t / "units_manifest.json", man)
        self.copy_file(ROOT / context.RELEASES["v11"]["heading_paths"], self.path("heading_paths"))
        self.copy_file(ROOT / context.RELEASES["v11"]["statute_graph"], self.path("statute_graph"))
        jobs = {}
        for key in gpu.ORDER:
            d = self.data / "gpu_inputs" / key
            jobs[key] = gpu.build_inputs(key, d)                          # local rows
            items = [json.loads(line) for line in (d / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
            if key == "qwen3_8b_gate":
                gpu.gate_e0_reference(d, d / "gate_e0_reference.npz")
            with (d / "items.jsonl").open("w", encoding="utf-8", newline="\n") as g:
                for it in items:
                    if it["g"] != "articles":
                        it["o"] = int(self.remap([it["o"]])[0])
                    g.write(json.dumps(it, ensure_ascii=False) + "\n")
            sm = self.data / "gpu_inputs" / f"{key}__smoke"
            sm.mkdir(parents=True, exist_ok=True)
            arts = [it for it in items if it["g"] == "articles"][:16]
            qs = [it for it in items if it["g"] != "articles"][:4]
            with (sm / "items.jsonl").open("w", encoding="utf-8", newline="\n") as g:
                for it in arts + qs:
                    g.write(json.dumps(it, ensure_ascii=False) + "\n")
            shutil.copy2(d / "task.json", sm / "task.json")
        self.write_json(self.data / "gpu_inputs" / "JOBS.json", jobs)
        self.note(f"GPU job inputs written for {len(jobs)} jobs")

    def nltk(self) -> None:
        import nltk.data
        p = Path(nltk.data.find("corpora/stopwords/spanish"))
        self.copy_file(p, self.data / "nltk_data/corpora/stopwords/spanish")

    def populations(self) -> None:
        st = context.active().store / "populations"
        for p in sorted(st.glob("*.json")):
            rec = json.loads(p.read_text(encoding="utf-8"))
            guard.dev_query_ids_only(rec["query_ids"], p.stem)
            rec["release_rows"] = self.remap(rec["release_rows"]).tolist()
            self.write_json(self.data / "populations" / p.name, rec)

    def systems(self, shas: dict[str, str]) -> None:
        from uol.final import registry
        data = json.loads(registry.SYSTEMS_JSON.read_text(encoding="utf-8"))
        for s in data["systems"]:
            art = s.get("artefact") or {}
            if art.get("key") == "ep_run" and art.get("path") in shas:
                art["bundle_sha256"] = shas[art["path"]]
            elif art.get("key") == "d58_sources" and art.get("path") in shas:
                art["bundle_sha256"] = shas[art["path"]]
        data["bundle"] = True
        self.write_json(self.out / "code/src/uol/final/systems.json", data)

    def meta(self, smoke: bool) -> dict[str, Any]:
        rec = {"dev_only": True, "test_rows_included": False, "lockbox_rows": 0, "release_id": self.d.manifest["release_id"],
               "rows": len(self.dev), "smoke": smoke, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "cut_record_id": guard.CUT_RECORD_ID,
               "rule": "bundle row i is v11 release row row_map[i] (the development rows of SPLIT.json, release order)",
               "row_map": [int(r) for r in self.dev], "query_ids_sha256": hashlib.sha256("\n".join(self.dev_qids).encode()).hexdigest()}
        self.write_json(self.data / "BUNDLE.json", rec)
        return rec


# --------------------------------------------------------------------------- code, setup, manifest, scan


def closure() -> list[str]:
    code = ("import importlib, json, sys; from pathlib import Path; src = Path(sys.argv[1]).resolve()\n"
            f"mods = {list(CLOSURE_MODULES)!r}\n"
            "[importlib.import_module(m) for m in mods]\n"
            "files = sorted({Path(m.__file__).resolve().relative_to(src).as_posix() for m in list(sys.modules.values())"
            " if getattr(m, '__file__', None) and Path(m.__file__).resolve().is_relative_to(src)})\n"
            "print(json.dumps(files))")
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    env.pop("UOL_FINAL_DATA", None)
    out = subprocess.run([sys.executable, "-c", code, str(SRC)], capture_output=True, text=True, env=env, check=True)
    try:
        CLOSURE_FAILURES[:] = json.loads(out.stderr.strip().splitlines()[-1]) if out.stderr.strip() else []
    except (json.JSONDecodeError, IndexError):
        CLOSURE_FAILURES[:] = [["stderr", out.stderr[-500:]]]
    return json.loads(out.stdout.strip().splitlines()[-1])


CLOSURE_FAILURES: list = []
TEST_CONFTEST = """\"\"\"Offline tests on the run computer (as tests/conftest.py of the thesis checkout: no network; no cache folder
inside the bundle, so nothing is written outside MANIFEST.json).\"\"\"
import socket
import sys

import pytest

sys.dont_write_bytecode = True


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("The local test suite does not authorize network access")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
"""


def export_code(out: Path) -> list[str]:
    files = set(closure()) | set(EXTRA_FILES)
    files |= {p.relative_to(SRC).as_posix() for p in (SRC / "uol" / "final").rglob("*.py")}   # every module of the package
    for f in list(files):                                          # package __init__ files of every parent
        p = Path(f).parent
        while p.as_posix() not in ("", "."):
            if (SRC / p / "__init__.py").exists():
                files.add((p / "__init__.py").as_posix())
            p = p.parent
    for f in sorted(files):
        dst = out / "code/src" / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / f, dst)
    for p in sorted((ROOT / "tests/final").rglob("*.py")):
        dst = out / "code/tests/final" / p.relative_to(ROOT / "tests/final")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    (out / "code/tests/conftest.py").write_text(TEST_CONFTEST, encoding="utf-8")
    (out / "code/research").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "research/paths.json", out / "code/research/paths.json")     # uol.paths aliases (no data)
    (out / "code/requirements.lock").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "code/pytest.ini").write_text("[pytest]\ntestpaths = tests/final\naddopts = -p no:cacheprovider\n", encoding="utf-8")
    return sorted(files)


def refresh_code(out: Path, smoke: bool = False) -> dict[str, Any]:
    """Re-export code, requirements, systems.json, plans, setup and README into an existing bundle (data untouched),
    then rewrite MANIFEST.json keeping the packed-array entries; used after a code fix."""
    from uol.embedding_push import common as ep
    out = Path(out).resolve()
    old = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))
    packed = {k: {x: v[x] for x in ("unpacked", "unpacked_sha256", "unpacked_bytes")} for k, v in old["files"].items()
              if v.get("unpacked")}
    shutil.rmtree(out / "code", ignore_errors=True)
    files = export_code(out)
    systems = json.loads((ROOT / "src/uol/final/systems.json").read_text(encoding="utf-8"))
    sha_of = {}
    for rel, rec in old["files"].items():
        if rec.get("unpacked"):
            sha_of[rec["unpacked"]] = rec["unpacked_sha256"]
        else:
            sha_of[rel] = rec["sha256"]
    for s in systems["systems"]:
        art = s.get("artefact") or {}
        key = {"ep_run": "data/ep_run/", "d58_sources": "data/gap_tests/d58/sources/"}.get(art.get("key"))
        if key and (key + art.get("path", "")) in sha_of:
            art["bundle_sha256"] = sha_of[key + art["path"]]
    systems["bundle"] = True
    (out / "code/src/uol/final/systems.json").write_text(json.dumps(systems, indent=1, ensure_ascii=False), encoding="utf-8")
    from uol.final import plan, setup_files
    plan.write(out / "jobs/plan.json", smoke=smoke)
    setup_files.write(out)
    man = write_manifest(out, packed)
    return {"code_files": len(files), "closure_import_failures": CLOSURE_FAILURES, "n_files": man["n_files"],
            "gb": round(man["bytes"] / 2 ** 30, 2)}


def write_manifest(out: Path, packed: dict[str, dict[str, Any]]) -> dict[str, Any]:
    from uol.embedding_push import common as ep
    files = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name != "MANIFEST.json":
            rel = p.relative_to(out).as_posix()
            files[rel] = {"sha256": ep.sha256_file(p), "bytes": p.stat().st_size, **packed.get(rel, {})}
    rec = {"bundle": "uol.final development-only bundle", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_files": len(files), "bytes": sum(f["bytes"] for f in files.values()), "files": files}
    (out / "MANIFEST.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


QID = re.compile(r"sjf:tesis:(\d+)")


def scan(out: Path, lock_qids: set[str], lock_records: set[str], n_rows: int) -> dict[str, Any]:
    """Refuse the bundle if any text file names a test-set query or record id, or any row array is not dev-sized."""
    bad: list[str] = []
    checked = 0
    for p in sorted((out / "data").rglob("*")):
        if not p.is_file():
            continue
        if p.suffix in (".json", ".jsonl", ".md", ".txt", ".csv"):
            text = p.read_text(encoding="utf-8", errors="replace")
            for m in QID.finditer(text):
                if m.group(0) in lock_qids or f"sjf:tesis:{m.group(1)}:hechos" in lock_qids:
                    bad.append(f"{p}: {m.group(0)}")
                    break
            for rid in re.findall(r'"record_id":\s*"?(\d+)', text):
                if rid in lock_records:
                    bad.append(f"{p}: record {rid}")
                    break
            checked += 1
    if bad:
        raise guard.TestSetRefused(f"test-set ids found in the bundle: {bad[:5]}")
    b = json.loads((out / "data/BUNDLE.json").read_text(encoding="utf-8"))
    if b["rows"] != n_rows or len(b["row_map"]) != n_rows:
        raise guard.TestSetRefused("BUNDLE.json row count differs from the development rows")
    return {"text_files_checked": checked, "test_ids_found": 0}


def build(out: Path, smoke: bool = False, compress: bool = True) -> dict[str, Any]:
    out = Path(out).resolve()
    s = str(out).lower()
    if ("my drive" in s or "mi unidad" in s or "shortcut-targets" in s) and not os.environ.get("UOL_FINAL_ALLOW_DRIVE"):
        raise PermissionError("the bundle is staged locally first; copying it to Drive needs the author's go (set "
                              "UOL_FINAL_ALLOW_DRIVE=1 only after that)")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; use a fresh folder")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    E = Exporter(out, compress)
    E.release()
    shas = E.scores(smoke)
    shas.update(E.dev_blocks() if not smoke else {})
    E.vectors(smoke)
    if not smoke:
        E.c_hybrid()
    E.units_and_gpu(smoke)
    if not smoke:
        E.round3_inputs()
    E.nltk()
    E.populations()
    meta = E.meta(smoke)
    files = export_code(out)
    E.systems(shas)
    from uol.final import plan, setup_files
    plan.write(out / "jobs/plan.json", smoke=smoke)
    setup_files.write(out)
    sc = scan(out, E.lock_qids, E.lock_records, len(E.dev))
    rec = {"out": str(out), "seconds": round(time.time() - t0, 1), "rows": meta["rows"], "code_files": len(files),
           "closure_import_failures": CLOSURE_FAILURES, "scan": sc, "packed_files": len(E.packed), "log": E.log}
    (out / "BUILD_LOG.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    man = write_manifest(out, E.packed)
    rec.update(n_files=man["n_files"], bytes=man["bytes"], gb=round(man["bytes"] / 2 ** 30, 2))
    return rec
