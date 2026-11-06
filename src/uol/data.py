"""The benchmark as published (the dataset folder): cases, labels, folds and the article catalogue.

Articles are numbered by their row in ``catalogue/articles.jsonl``; every score vector in this package is indexed that
way. Cases keep their ``query_id``; row numbers are positions in ``cases.jsonl``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

N_FOLDS = 5
ALL = N_FOLDS                     # the group trained on every development fold, used for new cases


def data_root(root: str | os.PathLike | None = None) -> Path:
    """The dataset folder: the argument, else ``$UOL_DATA``, else ``data/benchmark`` under the working directory."""
    return Path(root or os.environ.get("UOL_DATA") or "data/benchmark")


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Benchmark:
    root: Path
    corrected: bool = True        # labels with the three published corrections (the thesis's evaluation labels)
    _cache: dict = field(default_factory=dict, repr=False)

    # --------------------------------------------------------------------------------------------------- catalogue
    @cached_property
    def articles(self) -> list[dict]:
        return _jsonl(self.root / "catalogue" / "articles.jsonl")

    @cached_property
    def article_ids(self) -> list[str]:
        return [a["candidate_id"] for a in self.articles]

    @cached_property
    def article_texts(self) -> list[str]:
        """Each article's text with its law's title as the first line."""
        return [a["instrument_article_body"] for a in self.articles]

    @cached_property
    def article_index(self) -> dict[str, int]:
        return {a: i for i, a in enumerate(self.article_ids)}

    @cached_property
    def law_codes(self) -> np.ndarray:
        """Each article's law, as the position of its ``instrument_id`` in sorted order."""
        laws = [a["instrument_id"] for a in self.articles]
        code = {law: i for i, law in enumerate(sorted(set(laws)))}
        return np.asarray([code[x] for x in laws], dtype=np.int64)

    @cached_property
    def generic(self) -> np.ndarray:
        """The articles cited across many matters, removed before scoring."""
        rows = _jsonl(self.root / "labels" / "target_types.jsonl")
        return np.asarray(sorted(self.article_index[r["candidate_id"]] for r in rows if r["target_type"] == "generic"))

    # ------------------------------------------------------------------------------------------------------- cases
    @cached_property
    def cases(self) -> list[dict]:
        return _jsonl(self.root / "cases.jsonl")

    @cached_property
    def row_of(self) -> dict[str, int]:
        return {c["query_id"]: i for i, c in enumerate(self.cases)}

    @cached_property
    def texts(self) -> list[str]:
        return [c["text"] for c in self.cases]

    @cached_property
    def components(self) -> list[str]:
        return [c["component_id"] for c in self.cases]

    @cached_property
    def fold(self) -> np.ndarray:
        """Development fold 0-4 of each case; -1 for a test-period case."""
        split = json.loads((self.root / "splits" / "split.json").read_text(encoding="utf-8"))["assignment"]
        return np.asarray([int(split[c["query_id"]][-1]) if split[c["query_id"]].startswith("dev_fold") else -1
                           for c in self.cases], dtype=np.int64)

    def train_rows(self, group: int) -> np.ndarray:
        """Development rows a group learns from: every fold but ``group`` (all five for ``ALL``)."""
        return np.flatnonzero((self.fold >= 0) & (self.fold != group))

    def evaluation_rows(self, population: str = "DEV") -> np.ndarray:
        """The cases of a population file in ``splits/populations`` (``DEV``: the development evaluation cases)."""
        ids = json.loads((self.root / "splits" / "populations" / f"{population}.json").read_text(encoding="utf-8"))["query_ids"]
        return np.asarray(sorted(self.row_of[q] for q in ids), dtype=np.int64)

    @cached_property
    def dates(self) -> np.ndarray:
        """Publication date of each case."""
        return np.asarray([c["publication_date"] for c in self.cases], dtype="datetime64[D]")

    # ------------------------------------------------------------------------------------------------ older tesis
    @cached_property
    def history(self) -> dict:
        """The older *tesis* (published before 14 Aug 2020): date and cited articles, in file order."""
        rec = _jsonl(self.root / "historical" / "memory.jsonl")
        return {"record_id": [str(r["record_id"]) for r in rec],
                "date": np.asarray([r["date"] for r in rec], dtype="datetime64[D]"),
                "cited": [tuple(sorted(set(map(int, r["cited"])))) for r in rec]}

    @cached_property
    def history_texts(self) -> list[str]:
        """Heading and text of each older *tesis*, aligned with ``history``."""
        rows = _jsonl(self.root / "historical" / "texts.jsonl")
        text = {r["record_id"]: r["rubro"].strip() + "\n" + r["texto"] for r in rows}
        return [text[r] for r in self.history["record_id"]]

    # ------------------------------------------------------------------------------------------------------ labels
    def positives(self, corrected: bool | None = None) -> dict[int, list[int]]:
        """Cited articles of each labelled case (sorted article rows): the development cases, and the test cases when
        ``labels/test.jsonl`` is present. The corrections concern development cases only."""
        corrected = self.corrected if corrected is None else corrected
        key = ("positives", corrected)
        if key not in self._cache:
            files = [self.root / "labels" / ("dev_corrected.jsonl" if corrected else "dev.jsonl"), self.root / "labels" / "test.jsonl"]
            self._cache[key] = {self.row_of[r["query_id"]]: sorted({self.article_index[p["candidate_id"]] for p in r["positives"]})
                                for f in files if f.exists() for r in _jsonl(f)}
        return self._cache[key]

    def specific(self, corrected: bool | None = None) -> dict[int, list[int]]:
        """Cited articles without the generic ones: the targets of ranking and scoring."""
        g = set(self.generic.tolist())
        return {r: [a for a in p if a not in g] for r, p in self.positives(corrected).items()}


def load(root: str | os.PathLike | None = None, corrected: bool = True) -> Benchmark:
    return Benchmark(data_root(root), corrected)
