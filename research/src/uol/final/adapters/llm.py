"""LLM systems (Block E, E8, E9, E12): scorers over the matrices written by ``uol.final.llm_jobs`` on the run computer.

``LLMMatrix`` reads ``<work>/llm/<key>/{rows.npy, matrix.npy}`` (development rows only) and is pending until the LLM
job has run. ``export_top`` writes the top-k candidate lists of a finished run (query id, candidate ids; no labels).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard
from uol.final.adapters.base import Scorer, SystemPending, SystemUnavailable


class LLMMatrix(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        key = self.spec["adapter_args"].get("key", self.sid)
        d = context.active().work / "llm" / key
        if not (d / "matrix.npy").exists():
            raise SystemPending(f"{self.sid}: the LLM job has not produced {d / 'matrix.npy'} yet")
        rows = guard.dev_rows_only(np.load(d / "rows.npy"), f"{self.sid} rows")
        self.M = np.load(d / "matrix.npy", mmap_mode="r")
        self.pos = {int(r): i for i, r in enumerate(rows)}
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        miss = [int(r) for r in rows if int(r) not in self.pos]
        if miss:
            raise SystemUnavailable(f"{self.sid}: no LLM output for {len(miss)} rows of this population")
        return np.asarray(self.M[[self.pos[int(r)] for r in rows]], dtype=np.float32)

    def inputs(self) -> dict[str, Any]:
        return {"llm_meta": self.meta}


class LLMLater(Scorer):
    def prepare(self, pop) -> None:
        raise SystemUnavailable(f"{self.sid}: {self.spec['adapter_args'].get('reason', 'LLM job not part of this plan')}")


class NotWired(Scorer):
    def prepare(self, pop) -> None:
        raise SystemUnavailable(f"{self.sid}: {self.spec['adapter_args'].get('reason', 'not wired')}")


def export_top(store: Path, system: str, population: str, k: int = 30) -> Path:
    """``<store>/exports/block_e/<system>__<population>__top<k>.jsonl`` from the run's top200.parquet."""
    import pandas as pd
    from uol.embedding_push import common as ep
    src = Path(store) / "runs" / system / population / "top200.parquet"
    if not src.exists():
        raise FileNotFoundError(src)
    top = pd.read_parquet(src)
    top = top[top["rank"] <= k].sort_values(["query_id", "rank"])
    ids = ep.release().candidate_ids
    out = Path(store) / "exports" / "block_e" / f"{system}__{population}__top{k}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for qid, g in top.groupby("query_id", sort=True):
            f.write(json.dumps({"query_id": qid, "candidates": [ids[int(c)] for c in g["candidate_idx"]],
                                "scores": [float(s) for s in g["score"]]}, ensure_ascii=False) + "\n")
    tmp.replace(out)
    return out
