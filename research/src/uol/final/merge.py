"""Merge the run computer's mirrored results into the second runner's store (D86, 23 Oct 2026).

    <Root>\\venv\\Scripts\\python.exe -m uol.final --data <Root>\\data --store <Root>\\run\\final_20261021 merge-store
        --from "<Drive>\\UOL_FINAL_RUN\\results\\raw"

RUN-B recomputes only the remaining jobs and their dependencies; the contrasts and the thesis pack need every result in
one store. A result unit is ``runs/<system>/<population>`` with its ``receipts/`` and ``status/`` files. A unit the
local store already has is never touched: RUN-B's own result wins. Every other unit, and every other file the local
store lacks (``g1/G1_rebuild.md`` ...), is copied; the runner's own outputs (``contrasts/``, ``pack/``, ``gpu/``,
``llm/``) are never copied. ``MERGED_FROM.json`` lists what came from where, with sha256. Run it with the runner
stopped and no job pending: a copied result would otherwise stand in for a job that has not run here.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SKIP_TOP = {"contrasts", "pack", "gpu", "llm"}
SKIP_NAMES = {"desktop.ini", "MERGED_FROM.json"}


def _unit(rel: Path) -> tuple[str, str] | None:
    """(system, population) of a result file, or None for files outside result units."""
    parts = rel.parts
    if parts[0] == "runs" and len(parts) >= 4:
        return parts[1], parts[2]
    if parts[0] in ("receipts", "status") and len(parts) == 2 and "__" in rel.stem:
        sid, pop = rel.stem.rsplit("__", 1)
        return sid, pop
    return None


def merge_store(src: Path, store: Path) -> dict[str, Any]:
    from uol.final import guard
    src, store = Path(src), Path(store)
    if not src.is_dir():
        raise FileNotFoundError(f"{src} is not a results mirror")
    status = store.parent / "STATUS.json"                    # the runner's local status (<Root>/run/STATUS.json)
    if status.exists():
        c = json.loads(status.read_text(encoding="utf-8")).get("counts", {})
        if c.get("running") or c.get("pending"):
            raise RuntimeError(f"the runner reports {c.get('running', 0)} running and {c.get('pending', 0)} pending jobs: "
                               "merge only after every non-final job has finished, with the runner stopped")
    local = {(p.parent.parent.name, p.parent.name) for p in store.glob("runs/*/*/summary.json")}
    copied: dict[str, str] = {}
    kept, refused = set(), set()
    for p in sorted(src.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES or ".part" in p.name or ".tmp" in p.name:
            continue
        rel = p.relative_to(src)
        if rel.parts[0] in SKIP_TOP:
            continue
        u = _unit(rel)
        if u:
            try:
                guard.refuse_population(u[1])
            except Exception:
                refused.add(f"{u[0]}__{u[1]}")
                continue
            if u in local:
                kept.add(f"{u[0]}__{u[1]}")
                continue
        dst = store / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp_merge")
        shutil.copy2(p, tmp)
        tmp.replace(dst)
        copied[rel.as_posix()] = hashlib.sha256(dst.read_bytes()).hexdigest()
    units = sorted({f"{u[0]}__{u[1]}" for u in map(_unit, map(Path, copied)) if u})
    rec = {"utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "source": str(src),
           "rule": "local result units win; other files copied only where the local store lacks them",
           "units_from_source": units, "units_kept_local": sorted(kept), "units_refused": sorted(refused),
           "files": copied}
    prev = store / "MERGED_FROM.json"
    hist = json.loads(prev.read_text(encoding="utf-8")).get("history", []) if prev.exists() else []
    prev.write_text(json.dumps({**rec, "history": hist + [{k: rec[k] for k in ("utc", "source")} | {"n_files": len(copied)}]},
                               indent=1), encoding="utf-8")
    return {"units_from_source": len(units), "units_kept_local": len(kept), "files": len(copied), "refused": sorted(refused)}
