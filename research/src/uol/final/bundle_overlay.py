"""The record's development bundle without unpacking it (E13, option (e) of 26 Oct; no disk written beyond a manifest).

The bundle on Drive (``UOL_FINAL_RUN``) keeps its 76 large arrays packed (``*.npy.zsh``); the run computers unpack them
beside the packed files. Each one is a copy of a local array (same layout key in ``context.RELEASES['v11']``), either
whole or restricted to the 4,812 development rows in release order. ``verify`` proves this byte for byte: the sha256 of
the ``.npy`` bytes (header + C-order data) of the local array, or of its development-row map, must equal the bundle
MANIFEST's ``unpacked_sha256``. ``install`` then serves exactly the verified arrays in the bundle profile:
``numpy.load`` of an unpacked bundle path returns the local file (whole) or a read-only development-row view of it.

Test-set rule (D97): the local files hold test rows, which are never indexed. The row map is ``guard.dev_rows_only``
of the local release (checked at ``verify``), every view access translates bundle rows through that map and asserts
the result lies in the development set, and opening a memmap reads no row that is not indexed.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import pathlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from uol.final.new_cases_fit import ROOT, sha256, write_json

BUNDLE_ROOT = Path(os.environ.get("UOL_E13_BUNDLE_ROOT", r"bundle\UOL_FINAL_RUN"))
OVERLAY = ROOT / "experiments" / "runs" / "new_cases_20261026" / "bundle_overlay" / "OVERLAY.json"
BLOCK = 256
_STATE: dict[str, Any] = {}


def _npy_header(dtype: np.dtype, shape: tuple[int, ...]) -> bytes:
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(buf, {"descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                                               "fortran_order": False, "shape": tuple(int(x) for x in shape)})
    return buf.getvalue()


def _hash_rows(mm: np.ndarray, rows: np.ndarray | None) -> tuple[str, int]:
    """sha256 and size of the .npy bytes of ``mm`` (rows=None) or of ``mm[rows]``, read in blocks."""
    n = len(mm) if rows is None else len(rows)
    shape = (n,) + tuple(mm.shape[1:])
    h = hashlib.sha256()
    head = _npy_header(mm.dtype, shape)
    h.update(head)
    size = len(head)
    for a in range(0, n, BLOCK):
        block = np.ascontiguousarray(mm[a:a + BLOCK] if rows is None else mm[rows[a:a + BLOCK]])
        h.update(block.tobytes())
        size += block.nbytes
    return h.hexdigest(), size


def local_path(rel: str) -> Path | None:
    """``data/<bundle layout key path>/rest`` -> the local file under the same layout key (longest key path first)."""
    from uol.final import context
    if not rel.startswith("data/"):
        return None
    sub = rel[len("data/"):]
    local = context.RELEASES["v11"]
    best = None
    for key, bpath in context.BUNDLE_LAYOUT.items():
        if key in local and (sub == bpath or sub.startswith(bpath + "/")):
            if best is None or len(bpath) > len(best[1]):
                best = (key, bpath)
    if best is None:
        return None
    return ROOT / local[best[0]] / sub[len(best[1]):].lstrip("/")


def verify(out: Path = OVERLAY) -> dict[str, Any]:
    """Every packed bundle array against its local original (whole, else the development-row map). Writes OVERLAY.json."""
    from uol.embedding_push import common as ep
    from uol.final import context, guard
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists (verified once)")
    context.activate_local("v11")
    man_path = BUNDLE_ROOT / "MANIFEST.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    d = ep.release()
    dev = guard.dev_rows_only(ep.dev_rows(), "bundle overlay row map")
    lock = np.asarray(ep.lockbox_rows(), dtype=np.int64)
    bq = [json.loads(x)["query_id"] for x in (BUNDLE_ROOT / "data" / "release" / "queries.jsonl")
          .read_text(encoding="utf-8").splitlines() if x.strip()]
    if bq != [d.queries[int(r)].query_id for r in dev]:
        raise RuntimeError("the bundle's query order is not the local development order")
    res: dict[str, Any] = {"bundle_manifest": str(man_path), "bundle_manifest_sha256": sha256(man_path),
                           "dev_rows": [int(r) for r in dev], "n_local_rows": len(d.queries),
                           "lockbox_rows_disjoint": bool(not np.isin(dev, lock).any()), "arrays": {}, "fails": []}
    t0 = time.perf_counter()
    for rel, rec in sorted(man["files"].items()):
        if not rec.get("unpacked"):
            continue
        dst = rec["unpacked"]
        lp = local_path(dst)
        a: dict[str, Any] = {"local": str(lp.relative_to(ROOT)) if lp else None, "bundle_unpacked_sha256": rec["unpacked_sha256"],
                             "bundle_unpacked_bytes": rec["unpacked_bytes"]}
        if lp is None or not lp.exists():
            a["mode"], a["match"] = "missing", False
        else:
            mm = np.load(lp, mmap_mode="r")
            a["local_shape"] = list(mm.shape)
            if mm.shape and mm.shape[0] == len(d.queries):
                a["mode"] = "dev_rows"
                h, size = _hash_rows(mm, dev)
            else:
                a["mode"] = "whole"
                h, size = _hash_rows(mm, None)
            a["sha256"], a["bytes"] = h, size
            a["match"] = h == rec["unpacked_sha256"] and size == rec["unpacked_bytes"]
            del mm
        if not a["match"]:
            res["fails"].append(dst)
        res["arrays"][dst] = a
        print(f"{dst}: {a['mode']} {'match' if a['match'] else 'MISMATCH'} ({time.perf_counter() - t0:.0f}s)", flush=True)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    res["pass"] = not res["fails"] and res["lockbox_rows_disjoint"]
    res["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, res)
    return res


class DevRowView:
    """Read-only bundle-row view of a local [5980, ...] array: bundle row i is local row ``dev[i]``."""

    def __init__(self, path: Path, dev: np.ndarray, devset: np.ndarray):
        self._mm = np.load(path, mmap_mode="r")
        self._dev, self._devset = dev, devset
        self.shape = (len(dev),) + tuple(self._mm.shape[1:])
        self.dtype = self._mm.dtype
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape))
        self.nbytes = self.size * self.dtype.itemsize

    def __len__(self) -> int:
        return self.shape[0]

    def _rows(self, key):
        r = self._dev[key]
        rr = np.atleast_1d(r)
        if rr.size and not np.isin(rr, self._devset).all():   # by construction; asserted on every access
            raise RuntimeError("bundle overlay: a non-development row was requested")
        return r

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self._mm[(self._rows(key[0]),) + key[1:]]
        return self._mm[self._rows(key)]

    def __array__(self, dtype=None, copy=None):
        out = np.empty(self.shape, dtype=self.dtype)
        for a in range(0, len(self), BLOCK):
            out[a:a + BLOCK] = self[a:a + BLOCK]
        return out if dtype is None else out.astype(dtype)


def install(data_dir: Path, overlay: Path = OVERLAY) -> dict[str, Any]:
    """Serve the verified arrays at their unpacked bundle paths in this process (``numpy.load``, ``Path.exists``,
    ``common.sha256_file``). Refuses when the overlay was not verified for this bundle."""
    if _STATE.get("installed"):
        return _STATE["summary"]
    ov = json.loads(Path(overlay).read_text(encoding="utf-8"))
    man_path = Path(data_dir).parent / "MANIFEST.json"
    if not ov.get("pass") or sha256(man_path) != ov["bundle_manifest_sha256"]:
        raise RuntimeError("bundle overlay: not verified for this bundle (run new_cases overlay-verify)")
    dev = np.asarray(ov["dev_rows"], dtype=np.int64)
    devset = np.sort(dev)
    root = Path(data_dir).parent
    table = {}
    for dst, a in ov["arrays"].items():
        if a["match"]:
            table[os.path.normcase(str(root / dst))] = (ROOT / a["local"], a["mode"], a["sha256"])
    orig_load, orig_exists = np.load, pathlib.Path.exists
    from uol.embedding_push import common as ep
    orig_sha = ep.sha256_file

    def load(file, mmap_mode=None, *args, **kwargs):
        hit = table.get(os.path.normcase(str(file))) if isinstance(file, (str, os.PathLike)) else None
        if hit is None:
            return orig_load(file, mmap_mode, *args, **kwargs)
        lp, mode, _ = hit
        if mode == "whole":
            return orig_load(lp, mmap_mode, *args, **kwargs)
        view = DevRowView(lp, dev, devset)
        return view if mmap_mode is not None else np.asarray(view)

    def exists(self, *args, **kwargs):
        return os.path.normcase(str(self)) in table or orig_exists(self, *args, **kwargs)

    def sha_file(path, *args, **kwargs):
        hit = table.get(os.path.normcase(str(path)))
        return hit[2] if hit else orig_sha(path, *args, **kwargs)

    np.load = load
    pathlib.Path.exists = exists
    ep.sha256_file = sha_file
    _STATE["installed"] = True
    _STATE["summary"] = {"overlay": str(overlay), "overlay_sha256": sha256(Path(overlay)), "served": len(table),
                         "dev_rows": len(dev)}
    return _STATE["summary"]
