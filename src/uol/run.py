"""The whole run: from a case's facts to the final list, one case at a time, with nothing stored about the case.

A run builds the ranker of one group (fold 0-4 for a development case, ``ALL`` for a new or test case), encodes the
facts (or reads the benchmark's stored vector, as a check of the path), ranks, and for the full system asks the
language model. Each answer is written to a JSON-lines file with its timings.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

import lightgbm as lgb
import numpy as np

from uol import llm
from uol.data import Benchmark
from uol.dense import Vectors
from uol.system import Ranker


def model_path(models: str | Path, setup: str, group: int) -> Path:
    return Path(models) / setup / f"group{group}.txt"


def load_model(models: str | Path, setup: str, group: int) -> lgb.Booster:
    return lgb.Booster(model_file=str(model_path(models, setup, group)))


class Run:
    """``setup``: noenc, ranker or system. ``encoder``: stored (the benchmark's vectors), cuda or cpu."""

    def __init__(self, bench: Benchmark, setup: str, group: int, models: str | Path, vectors: str | Path | None,
                 encoder: str = "stored", model: llm.Model | None = None, windows: bool = False) -> None:
        self.bench, self.setup, self.windows = bench, setup, windows
        ranker_setup = "noenc" if setup == "noenc" else "ranker"
        self.vectors = Vectors.load(vectors) if ranker_setup == "ranker" else None
        self.ranker = Ranker(bench, ranker_setup, group, load_model(models, ranker_setup, group), self.vectors)
        self.encode = None
        if ranker_setup == "ranker" and encoder != "stored":
            from uol.encoder import Encoder
            self.encode = Encoder(encoder)
        self.model = model
        self.catalogue = llm.Catalogue(bench) if setup == "system" else None
        self.variants = ("b", "c") if group == 4 else ("c",)

    def case_vectors(self, row: int, facts: str) -> dict[str, np.ndarray] | None:
        if self.vectors is None:
            return None
        if self.encode is not None:
            return self.encode(facts, self.variants)
        return {v: np.asarray(self.vectors.cases[v][row], dtype=np.float32) for v in self.variants}

    def answer(self, row: int) -> dict:
        facts, date = self.bench.texts[row], self.bench.dates[row]
        t0 = time.perf_counter()
        vec = self.case_vectors(row, facts)
        t1 = time.perf_counter()
        ranked = self.ranker.rank(facts, date, vec)
        t2 = time.perf_counter()
        out = {"query_id": self.bench.cases[row]["query_id"], "ranked": ranked,
               "seconds": {"encode": round(t1 - t0, 3), "rank": round(t2 - t1, 3)}}
        if self.setup == "system":
            a = llm.full_system(self.model, self.bench, self.catalogue, facts, ranked, self.windows)
            out |= {"model_list": a.model_list, "merged": a.merged, "final": a.final}
            out["seconds"]["model"] = round(time.perf_counter() - t2, 3)
        return out

    def answer_all(self, rows: Iterable[int], out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        done = set()
        if out.exists():
            with out.open(encoding="utf-8") as f:
                done = {json.loads(line)["query_id"] for line in f}
        with out.open("a", encoding="utf-8") as f:
            for r in rows:
                if self.bench.cases[r]["query_id"] in done:
                    continue
                f.write(json.dumps(self.answer(int(r))) + "\n")
                f.flush()
        return out


def read_answers(bench: Benchmark, path: str | Path, key: str = "final") -> dict[int, list[int]]:
    with Path(path).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return {bench.row_of[r["query_id"]]: r.get(key, r["ranked"]) for r in rows}
