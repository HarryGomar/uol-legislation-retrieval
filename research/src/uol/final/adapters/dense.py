"""Dense encoders from cached vectors: MaxP cosine over article units (``embedding_push.common.maxp_scores``).

The vector folder layout is detected from its files:

* scale-curve layout (``scale_curve.common.load_vectors``): ``article_vectors.npy``, ``article_index.npy``,
  ``query_vectors_<variant>.npy`` and ``query_rows.json`` (release rows of the query vectors);
* the Qwen3-8B endpoint layout: the same article files, ``query_vectors_<variant>.npy`` for every release row and
  ``query_ids.json`` in release order;
* the English-legal e5 layout (eval_lenses X13): ``chunk_vectors.npy`` (rows of ``chunks_e5_small_v1/chunks.jsonl``),
  ``query_vectors_dev.npy`` and ``query_rows_dev.npy``.

The bundle exports every encoder in the scale-curve layout restricted to development rows. GPU encoders whose
vectors do not exist yet raise ``SystemPending`` (their jobs are prepared by ``adapters.gpu``). The query variant is
a tuned choice (inner folds) when several exist, as the frozen rule chose it on development data.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard
from uol.final.adapters.base import Scorer, SystemPending, select


def vector_dir(spec: dict[str, Any]) -> Path:
    a = spec["adapter_args"]
    prof = context.active()
    if a.get("key") in prof.paths:
        return prof.path(a["key"]) / a.get("sub", "")
    return prof.path("new_vectors") / a.get("key", spec["id"])


def _sha_small(path: Path) -> str:
    from uol.embedding_push import common as ep
    return ep.sha256_file(path)


class VectorSet:
    """Article vectors, owners and per-variant query vectors keyed by release row (dev rows only)."""

    def __init__(self, folder: Path):
        from uol.embedding_push import common as ep
        self.folder = folder
        d = ep.release()
        self.N = len(d.candidate_ids)
        art = folder
        if (folder / "articles_from.json").exists():              # query-only encodes reuse a model's article side
            key = json.loads((folder / "articles_from.json").read_text(encoding="utf-8"))["key"]
            art = context.active().path(key) if key in context.active().paths else folder.parent / key
        if (art / "article_vectors.npy").exists():
            self.A = np.asarray(np.load(art / "article_vectors.npy"), dtype=np.float32)
            self.owner = np.load(art / "article_index.npy")
            self.article_files = [art / "article_vectors.npy", art / "article_index.npy"]    # the index on disk (cost)
        elif (folder / "article_vectors.npy").exists():
            self.A = np.asarray(np.load(folder / "article_vectors.npy"), dtype=np.float32)
            self.owner = np.load(folder / "article_index.npy")
            self.article_files = [folder / "article_vectors.npy", folder / "article_index.npy"]
        elif (folder / "chunk_vectors.npy").exists():
            self.article_files = [folder / "chunk_vectors.npy"]
            self.A = np.asarray(np.load(folder / "chunk_vectors.npy"), dtype=np.float32)
            units = folder / "units.jsonl"
            if not units.exists():
                units = context.active().path("derived_ftl") / "chunks_e5_small_v1" / "chunks.jsonl"
            self.owner = ep.chunk_article_index(units)
        else:
            raise SystemPending(f"no article vectors in {folder}")
        if len(self.owner) != len(self.A):
            raise RuntimeError(f"{folder}: {len(self.A)} vectors but {len(self.owner)} owners")
        self.transform = None
        self.agg = "maxp"                 # article score from its units: maxp (default), first, mean, top2 (C9, D93)
        self.queries: dict[str, tuple[np.ndarray, dict[int, int]]] = {}
        if (folder / "query_rows.json").exists():
            rows = json.loads((folder / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
            rows = guard.dev_rows_only(rows, f"{folder.name} query rows")
            pos = {int(r): i for i, r in enumerate(rows)}
            for p in sorted(folder.glob("query_vectors_*.npy")):
                self.queries[p.stem.replace("query_vectors_", "")] = (np.load(p, mmap_mode="r"), pos)
        elif (folder / "query_ids.json").exists():
            ids = json.loads((folder / "query_ids.json").read_text(encoding="utf-8"))
            if ids != [q.query_id for q in d.queries]:
                raise ValueError(f"{folder}: query vectors are not in release order")
            pos = {i: i for i in guard.dev_rows_only(ep.dev_rows()).tolist()}      # dev rows only are addressable
            for p in sorted(folder.glob("query_vectors_*.npy")):
                self.queries[p.stem.replace("query_vectors_", "")] = (np.load(p, mmap_mode="r"), pos)
        elif (folder / "query_rows_dev.npy").exists():
            rows = guard.dev_rows_only(np.load(folder / "query_rows_dev.npy"), f"{folder.name} query rows")
            self.queries["dev"] = (np.load(folder / "query_vectors_dev.npy", mmap_mode="r"),
                                   {int(r): i for i, r in enumerate(rows)})
        if not self.queries:
            raise SystemPending(f"no query vectors in {folder}")

    @property
    def variants(self) -> list[str]:
        return sorted(self.queries)

    def scores(self, variant: str, rows: Sequence[int]) -> np.ndarray:
        from uol.embedding_push import common as ep
        rows = guard.dev_rows_only(rows, "dense rows")
        Q, pos = self.queries[variant]
        missing = [int(r) for r in rows if int(r) not in pos]
        if missing:
            raise SystemPending(f"{self.folder.name}: no query vector for {len(missing)} rows (variant {variant})")
        q = np.asarray(Q[[pos[int(r)] for r in rows]], dtype=np.float32)
        if self.transform is not None:
            q = self.transform(q)
        out = np.empty((len(rows), self.N), dtype=np.float32)
        blocks = ep.maxp_scores(q, self.A, self.owner, self.N) if self.agg == "maxp" else self.unit_scores(q)
        for s, block in blocks:
            out[s:s + len(block)] = block
        return out

    def unit_scores(self, q: np.ndarray, batch: int = 256):
        """C9 (b): the article score from its units' cosines by another rule than MaxP: ``first`` (the first unit,
        which starts with the article's title line), ``mean`` (of its units) or ``top2`` (the sum of its two best units,
        the single unit for one-unit articles)."""
        order = np.argsort(self.owner, kind="stable")                 # stable: units keep their window order
        vecs, art = self.A[order], self.owner[order]
        starts = np.flatnonzero(np.r_[True, art[1:] != art[:-1]])
        counts = np.diff(np.r_[starts, len(art)])
        owners = art[starts]
        multi = np.flatnonzero(counts > 1)
        for s in range(0, len(q), batch):
            sims = q[s:s + batch] @ vecs.T
            if self.agg == "first":
                best = sims[:, starts]
            elif self.agg == "mean":
                best = np.add.reduceat(sims, starts, axis=1) / counts
            elif self.agg == "top2":
                best = np.maximum.reduceat(sims, starts, axis=1)
                for j in multi:
                    seg = sims[:, starts[j]:starts[j] + counts[j]]
                    best[:, j] = np.sort(seg, axis=1)[:, -2:].sum(axis=1)
            else:
                raise ValueError(f"unknown unit aggregation {self.agg}")
            out = np.full((sims.shape[0], self.N), -np.inf, dtype=np.float32)
            out[:, owners] = best
            yield s, out

    def receipt(self) -> dict[str, Any]:
        files = {}
        for p in sorted(self.folder.glob("*")):
            if p.is_file() and p.suffix in (".npy", ".json") and p.stat().st_size < 2_000_000_000:
                files[p.name] = {"bytes": p.stat().st_size}
        man = self.folder / "manifest.json"
        return {"folder": str(self.folder), "files": files,
                "manifest_sha256": _sha_small(man) if man.exists() else None}


def devn_override(vs: "VectorSet", key: str) -> dict[str, Any] | None:
    """In the ``devn`` variant, the DEV-N rows' query vectors come from ``<new_vectors>/devn__<key>`` (their re-encoded
    clause-dropped facts); every other row keeps the saved vector. Returns the override record (or None)."""
    prof = context.active()
    if prof.variant not in context.TEXT_VARIANTS:       # DEV-L (C6) and DEV-S (D109) as DEV-N
        return None
    d = prof.path("new_vectors") / f"{prof.variant}__{key}"
    if not (d / "query_rows.json").exists():
        raise SystemPending(f"{prof.variant} query vectors for {key} not encoded yet ({d})")
    mod = context.text_module(prof.variant)
    rows = json.loads((d / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
    dn = set(mod.rows())
    rec = {"folder": str(d), "variants": {}}
    for p in sorted(d.glob("query_vectors_*.npy")):
        v = p.stem.replace("query_vectors_", "")
        tv = v if v in vs.queries else (vs.variants[0] if len(vs.variants) == 1 else None)
        if tv is None:
            continue
        V = np.load(p)
        Q, pos = vs.queries[tv]
        Qm = np.array(Q, dtype=np.float32)
        n = 0
        for i, r in enumerate(rows):
            if int(r) in dn and int(r) in pos:
                Qm[pos[int(r)]] = V[i]
                n += 1
        vs.queries[tv] = (Qm, pos)
        rec["variants"][tv] = n
    fid = d / "fidelity.json"
    rec["fidelity"] = json.loads(fid.read_text(encoding="utf-8")) if fid.exists() else None
    return rec


class DevNFidelityFailed(RuntimeError):
    """The 50 unchanged control queries re-encoded for DEV-N disagree with the saved vectors (cos < .99)."""


def compression(dim: int = 0, quant: str | None = None):
    """Gap-review candidate (RUN_PLAN §3.9): Matryoshka truncation to the first ``dim`` dimensions (renormalised)
    and/or int8 (symmetric per-vector scale, de-quantised) or binary (sign, scaled to unit norm) storage."""
    def f(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if dim:
            X = X[:, :dim]
            n = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / np.where(n > 0, n, 1.0)
        if quant == "int8":
            s = np.abs(X).max(axis=1, keepdims=True) / 127.0
            s = np.where(s > 0, s, 1.0)
            X = np.round(X / s).clip(-127, 127) * s
        elif quant == "binary":
            X = np.where(X >= 0, 1.0, -1.0) / np.sqrt(X.shape[1])
        elif quant not in (None, ""):
            raise ValueError(f"unknown quantisation {quant!r}")
        return X.astype(np.float32)
    return f


class DenseVectors(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        folder = vector_dir(self.spec)
        if not folder.exists():
            raise SystemPending(f"{self.sid}: vectors not available yet at {folder}")
        self.vs = VectorSet(folder)
        self.devn = devn_override(self.vs, self.spec["adapter_args"].get("key", self.sid))
        fid = (self.devn or {}).get("fidelity") or {}
        if fid.get("checked") and not fid.get("pass"):           # binding (review 22 Oct §4.2): no scoring on a failed check
            raise DevNFidelityFailed(f"{self.sid}: DEV-N fidelity check failed for "
                                     f"{self.spec['adapter_args'].get('key', self.sid)}: {fid.get('variants')}")
        a = self.spec["adapter_args"]
        if a.get("mrl_dim") or a.get("quant"):
            self.vs.transform = compression(int(a.get("mrl_dim") or 0), a.get("quant"))
            self.vs.A = self.vs.transform(self.vs.A)
        self.vs.agg = a.get("unit_agg", "maxp")
        want = self.spec["adapter_args"].get("variants")
        self.grid = [{"variant": v} for v in (want or self.vs.variants) if v in self.vs.variants] or \
            [{"variant": self.vs.variants[0]}]

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        self.params, hist = select(self.grid, lambda p, r: self.vs.scores(p["variant"], r), fold, self.pop.eval_population)
        return {"params": self.params, "selection": hist}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        return self.vs.scores(self.params["variant"], self.rows(rows))

    def inputs(self) -> dict[str, Any]:
        return {"vectors": self.vs.receipt(), "devn_override": getattr(self, "devn", None)}


class FoldDense(Scorer):
    """Per-fold fine-tuned encoders (``bge_m3_ft``): ``<new_vectors>/<key>/fold<k>`` holds the catalogue and the
    held-out fold's queries encoded by the model trained on the other folds; scored alone (Block A, E2)."""

    def prepare(self, pop) -> None:
        super().prepare(pop)
        self.root = context.active().path("new_vectors") / self.spec["adapter_args"]["key"]
        if not any((self.root / f"fold{k}" / "query_rows.json").exists() for k in range(5)):
            raise SystemPending(f"{self.sid}: fold vectors not available yet at {self.root}")

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        d = self.root / f"fold{fold.k}"
        if not (d / "query_rows.json").exists():
            raise SystemPending(f"{self.sid}: fold {fold.k} vectors missing ({d})")
        self.vs = VectorSet(d)
        self.variant = self.vs.variants[0]
        man = d / "manifest.json"
        return {"fold_model": str(d), "variant": self.variant,
                "job": json.loads(man.read_text(encoding="utf-8")).get("job_run", {}).get("task") if man.exists() else None}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        return self.vs.scores(self.variant, self.rows(rows))
