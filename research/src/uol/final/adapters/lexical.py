"""Lexical systems rebuilt from code: BM25 (stemmed), L* (BM25 + character n-grams) and L* with headings.

L* uses each outer fold's frozen selection (``experiments/runs/embedding_push_20261004/base/selection/lexical.json``)
and the frozen operators ``embedding_push.base.lstar_block`` (robust-z BM25 channel + robust-z character TF-IDF,
``base.pair(w_char)``), so the rebuild reproduces the frozen ``L_star`` file. "With headings" swaps the article
text for the D58 heading-path text (``gap_tests.lexical_grid.variant_texts`` V2: law title, Título / Capítulo /
Sección path, body). BM25 alone tunes (k1, b) on the inner folds over the frozen L* grid.
"""
from __future__ import annotations

import json
from collections import Counter
from typing import Any, Sequence

import numpy as np

from uol.final import context
from uol.final.adapters.base import Scorer, select

BM25_GRID = [{"k1": k1, "b": b} for k1 in (0.9, 1.2, 1.5) for b in (0.4, 0.75)]
_TEXTS: dict[str, list[str]] = {}


def article_texts(variant: str = "V0") -> list[str]:
    """Catalogue texts: V0 = the frozen article text; V2 = heading-path text (D58)."""
    if variant not in _TEXTS:
        from uol.embedding_push import common as ep
        v0 = list(ep.release().candidate_texts)
        if variant == "V0":
            _TEXTS[variant] = v0
        elif variant == "V2":
            from uol.gap_tests.lexical_grid import variant_texts
            _TEXTS[variant] = variant_texts(v0)[0]["V2_hierarchy_path"]
        else:
            raise ValueError(variant)
    return _TEXTS[variant]


class TextCtx:
    """The one ``base.Ctx`` method ``base.lstar_block`` uses (``lex(kind)``), over a chosen article text."""

    def __init__(self, texts: list[str]):
        self.texts = texts
        self._lex: dict[str, Any] = {}

    def lex(self, kind: str):
        if kind not in self._lex:
            from uol.facts_to_law import lexical
            self._lex[kind] = lexical.LexicalIndex(self.texts, kind=kind)
        return self._lex[kind]


_CTX: dict[str, TextCtx] = {}


def text_ctx(variant: str) -> TextCtx:
    if variant not in _CTX:
        _CTX[variant] = TextCtx(article_texts(variant))
    return _CTX[variant]


def query_texts(rows: Sequence[int]) -> list[str]:
    from uol.embedding_push import common as ep
    d = ep.release()
    return [d.queries[int(r)].text for r in rows]


def frozen_selection() -> dict[str, dict]:
    path = context.active().path("ep_run") / "base" / "selection" / "lexical.json"
    sel = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["selected"] for k, v in sel.items() if k.startswith("dev_fold")}


class LStar(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        self.variant = self.spec["adapter_args"].get("text", "V0")
        self.sel = frozen_selection()
        self.tc = text_ctx(self.variant)

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        from uol.embedding_push import base
        if fold.role in self.sel:
            row = self.sel[fold.role]
            note = "frozen fold selection"
        else:                                                       # DEV-T: the selection most folds made
            row = json.loads(Counter(json.dumps(v, sort_keys=True) for v in self.sel.values()).most_common(1)[0][0])
            note = "most frequent frozen fold selection (no fold role)"
        self.lsel = base.LSel.from_json(row)
        self.params = dict(row)
        return {"params": self.params, "selection": note, "text": self.variant}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        from uol.embedding_push import base
        rows = self.rows(rows)
        return np.asarray(base.lstar_block(self.tc, self.lsel, query_texts(rows)), dtype=np.float32)

    def inputs(self) -> dict[str, Any]:
        return {"text_variant": self.variant, "selection_file": str(context.active().path("ep_run") / "base/selection/lexical.json")}


class BM25(Scorer):
    def prepare(self, pop) -> None:
        super().prepare(pop)
        a = self.spec["adapter_args"]
        self.kind = a.get("kind", "bm25_stem")
        self.variant = a.get("text", "V0")
        self.grid = a.get("grid") or BM25_GRID
        self.tc = text_ctx(self.variant)

    def _scores(self, params: dict, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        return self.tc.lex(self.kind).score(query_texts(rows), {"k1": params["k1"], "b": params["b"]})

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        self.params, hist = select(self.grid, self._scores, fold, self.pop.eval_population)
        return {"params": self.params, "selection": hist}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        return self._scores(self.params, rows)
