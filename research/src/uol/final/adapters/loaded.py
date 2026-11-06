"""Loaded systems: frozen score files, frozen pipelines, rebuilt-elsewhere dev source blocks and the cached LLM stage.

Nothing here recomputes a score; the adapters read hash-checked artefacts through the original helpers
(``embedding_push.common.open_scores``, ``fusion_ltr.pipeline_path``) and slice development rows only.

* ``FrozenScores``   ``experiments/runs/embedding_push_20261004/scores/<id>/scores.npy`` ([release rows, N]).
* ``FrozenPipeline`` ``.../fusion/pipelines/<name>/scores.npy`` (the frozen learned rankers).
* ``DevBlock``       a [dev rows, N] block in development-row order (e.g. the D58 heading-path L*).
* ``CHybridCached``  the frozen LLM-stage hybrid: its stage-2 matrix and the three cached listwise orders, fused by
                     ``round3.features.fusion_route`` exactly as ``round3.c_hybrid.evaluate`` (no LLM call).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard
from uol.final.adapters.base import Scorer, SystemPending, SystemUnavailable


def _sha(path: Path) -> str:
    from uol.embedding_push import common as ep
    return ep.sha256_file(path)


def verify_artefact(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """Check a loaded artefact against systems.json.

    Bundle profile: the file's sha256 must equal ``artefact.bundle_sha256`` (the dev-only export).
    Local profile: the sha256 declared in the artefact's manifest must equal ``artefact.manifest_sha256`` (hashing the
    full local file would read test-set rows, so the content hash of record is the bundle's dev-only file)."""
    art = spec.get("artefact") or {}
    prof = context.active()
    rec: dict[str, Any] = {"path": str(path), "bytes": path.stat().st_size if path.exists() else None}
    if not path.exists():
        raise SystemPending(f"{spec['id']}: artefact {path} not found")
    if prof.sources == "rebuilt" and "ep_rebuild" in str(path):
        rec.update(sha256=_cached_sha(path), verified=None, note="rebuilt from code in this run (no frozen hash applies)")
        return rec
    if prof.kind == "bundle" and art.get("bundle_sha256"):
        got = _cached_sha(path)
        rec.update(sha256=got, expected=art["bundle_sha256"], verified=got == art["bundle_sha256"])
        if got != art["bundle_sha256"]:
            raise RuntimeError(f"{spec['id']}: {path} sha256 {got} differs from systems.json {art['bundle_sha256']}")
        return rec
    man = path.parent / "manifest.json"
    declared = None
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        declared = m.get("scores_sha256") or m.get("pairs_sha256")
    rec.update(manifest_sha256=declared, expected=art.get("manifest_sha256"))
    if art.get("manifest_sha256") and declared and declared != art["manifest_sha256"]:
        raise RuntimeError(f"{spec['id']}: manifest sha256 {declared} differs from systems.json")
    rec["verified"] = bool(art.get("manifest_sha256") and declared == art.get("manifest_sha256"))
    return rec


def _cached_sha(path: Path) -> str:
    prof = context.active()
    cache = prof.work / "verified_sha256.json"
    key = f"{path}|{path.stat().st_size}|{int(path.stat().st_mtime)}"
    data = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
    if key not in data:
        data[key] = _sha(path)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
        tmp.replace(cache)
    return data[key]


class FrozenScores(Scorer):
    """A frozen full score file read through ``ep.open_scores`` (memory-mapped; dev rows only)."""

    def prepare(self, pop) -> None:
        super().prepare(pop)
        from uol.embedding_push import common as ep
        self.source = self.spec["adapter_args"]["source"]
        self.path = ep.score_path(self.source)
        self.receipt = verify_artefact(self.spec, self.path)
        self.mm = ep.open_scores(self.source)
        if self.mm.shape != (len(ep.release().queries), len(ep.release().candidate_ids)):
            raise RuntimeError(f"{self.sid}: score file shape {self.mm.shape} does not match the release")

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return np.asarray(self.mm[rows], dtype=np.float32)

    def inputs(self) -> dict[str, Any]:
        return {"score_file": self.receipt}


class FrozenPipeline(FrozenScores):
    """A frozen learned-ranker matrix (``fusion_ltr.pipeline_path``)."""

    def prepare(self, pop) -> None:
        Scorer.prepare(self, pop)
        from uol.embedding_push import common as ep, fusion_ltr as fl
        self.path = fl.pipeline_path(self.spec["adapter_args"]["pipeline"])
        self.receipt = verify_artefact(self.spec, self.path)
        self.mm = np.load(self.path, mmap_mode="r")
        if self.mm.shape != (len(ep.release().queries), len(ep.release().candidate_ids)):
            raise RuntimeError(f"{self.sid}: pipeline shape {self.mm.shape} does not match the release")


class DevBlock(Scorer):
    """A [development rows, N] block in ``ep.dev_rows()`` order (key + relative path in adapter_args)."""

    def prepare(self, pop) -> None:
        super().prepare(pop)
        from uol.embedding_push import common as ep
        a = self.spec["adapter_args"]
        self.path = context.active().path(a["key"]) / a["path"]
        self.receipt = verify_artefact(self.spec, self.path)
        self.mm = np.load(self.path, mmap_mode="r")
        dev = ep.dev_rows()
        if self.mm.shape != (len(dev), len(ep.release().candidate_ids)):
            raise RuntimeError(f"{self.sid}: block shape {self.mm.shape} is not [dev rows, N]")
        self.pos = np.full(len(ep.release().queries), -1, dtype=np.int64)
        self.pos[dev] = np.arange(len(dev))

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return np.asarray(self.mm[self.pos[rows]], dtype=np.float32)

    def inputs(self) -> dict[str, Any]:
        return {"block": self.receipt}


class CHybridCached(Scorer):
    """Frozen C_hybrid (stage 2 + three cached LLM listwise samples), DEV (P_spec) rows only.

    Reuses ``round3.c_hybrid.arm_matrix`` (pointed at the profile's C_hybrid folder) and
    ``round3.features.fusion_route`` with ``c_hybrid.W_EXT``: the weight is cross-fitted over the five dev folds inside
    fusion_route (chosen on the other four folds), so the per-fold loop of the harness only slices its output."""

    def prepare(self, pop) -> None:
        super().prepare(pop)
        if pop.eval_population != "P_spec" or pop.name not in ("DEV",):
            raise SystemUnavailable("C_hybrid has stage-2 scores and LLM orders for DEV (P_spec) rows only")
        from uol.round3 import c_hybrid, features
        prof = context.active()
        folder = prof.path("c_hybrid")
        for f in ("stage2_pspec_rows.npy", "stage2_pspec_matrix.npy", "orders_s0.json", "orders_s1.json", "orders_s2.json"):
            if not (folder / f).exists():
                raise SystemPending(f"C_hybrid cached input {folder / f} missing")
        c_hybrid.OUT = folder                         # read-only use of the frozen folder (arm_matrix reads OUT)
        prof.patches["uol.round3.c_hybrid.OUT"] = str(folder)
        samples = int(self.spec.get("adapter_args", {}).get("samples", 3))
        rows, A = c_hybrid.arm_matrix(samples)
        rows = guard.dev_rows_only(rows, "C_hybrid stage-2 rows")
        S2 = np.load(folder / "stage2_pspec_matrix.npy")
        fr = features.fusion_route("stage2", rows, A, grid=c_hybrid.W_EXT, ltr_matrix=S2)
        self.matrix = fr["oof_matrix"]
        del A, S2
        self.pos = {int(r): i for i, r in enumerate(rows)}
        self.info = {"w_by_fold": fr["w_by_fold"], "w_all": fr["w_all"], "samples": samples,
                     "dev_ndcg_by_w": fr["dev_ndcg_by_w"]}
        self.receipt = {f: {"sha256": _sha(folder / f), "bytes": (folder / f).stat().st_size}
                        for f in ("stage2_pspec_rows.npy", "stage2_pspec_matrix.npy", "orders_s0.json", "orders_s1.json",
                                  "orders_s2.json")}

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        return {"w_fold": self.info["w_by_fold"].get(fold.k), "cross_fitted_inside": "round3.features.fusion_route"}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        missing = [int(r) for r in rows if int(r) not in self.pos]
        if missing:
            raise SystemUnavailable(f"C_hybrid has no stage-2 scores for {len(missing)} rows")
        return np.asarray(self.matrix[[self.pos[int(r)] for r in rows]], dtype=np.float32)

    def inputs(self) -> dict[str, Any]:
        return {"c_hybrid_inputs": self.receipt, "fusion": self.info}
