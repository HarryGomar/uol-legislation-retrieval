"""Tests 3 and 4 (D56): the BM25 grid past its edges (with BM25F-style title weighting) and lexical article-text variants.

Both rebuild the frozen L* selection of ``embedding_push.base.select_lstar`` on dev rows only: per dev fold, inner
selection on the other four folds by component-macro nDCG@10 on P_spec over population-P rows; BM25 (plain, stemmed)
config first, the better of the two BM25 channels, then its robust-z fusion with character TF-IDF over w_char, then the
best of {BM25, stemmed BM25, char, fusion}. Held-out folds are scored with their fold's choice and pooled into one dev
P_spec vector per arm. Same tie rule (``base.argbest``), same metric code (``Ctx.query_metrics``).

The arm "original grid, current text" must reproduce L* .184 and stemmed BM25 .176 within .003 before anything else
is read.

    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.lexical_grid grid       # test 3
    PYTHONPATH=src py -3.12 -m uol.round3.heavy uol.gap_tests.lexical_grid variants   # test 4
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from uol.embedding_push import base, common as ep
from uol.facts_to_law import fusion, lexical
from uol.gap_tests import common as gc
from uol.retrieval_benchmarks.c01_statutes.experiments import _bm25_document_matrix

ORIG_K1, ORIG_B = (0.9, 1.2, 1.5), (0.4, 0.75)
EXT_K1, EXT_B = (0.3, 0.5, 0.7, 0.9, 1.2, 1.5), (0.4, 0.75, 0.85, 0.95, 1.0)
TITLE_W = (0.0, 0.5, 1.0, 2.0, 3.0)
W_CHAR = base.W_CHAR
PATHS = ep.ROOT / "experiments/runs/round3_best_case_20261013/phase2/statute_graph/paths.json"
GATE = {"L_star": 0.184, "bm25_stem": 0.176}
DOF_NOTE = re.compile(
    r"\s*(?:P[áa]rrafos?|Fracci[óo]n(?:es)?|Art[íi]culos?|Incisos?|Subincisos?|Apartados?|Numerales?|"
    r"Bases?|Denominaci[óo]n|Fe de erratas|Reforma|Cap[íi]tulo|T[íi]tulo|Secci[óo]n|Libro|"
    r"Primer|Segundo|Tercer|Cuarto|Quinto|[ÚU]ltimo|Adici[óo]n|Derogaci[óo]n|Nota)"
    r"[^\n]{0,160}?DOF\s+\d{1,2}-\d{1,2}-\d{4}(?:\s*(?:,|y|e)\s*\d{1,2}-\d{1,2}-\d{4})*\.?")

DOF_DATE = re.compile(r"\s*DOF\s+\d{1,2}-\d{1,2}-\d{4}(?:\s*(?:,|y|e)\s*\d{1,2}-\d{1,2}-\d{4})*")  # leftovers: "Derogado DOF ..."


class Bm25Channel:
    """BM25 over a CountVectorizer vocabulary with optional title/body fields (BM25F-style count weighting)."""

    def __init__(self, texts: Sequence[str], stem: bool) -> None:
        if stem:
            self.vec = CountVectorizer(analyzer=lexical.SpanishAnalyzer(), min_df=1, max_features=lexical.MAX_FEATURES,
                                       dtype=np.float32)
        else:
            self.vec = CountVectorizer(token_pattern=lexical.TOKEN_PATTERN, min_df=1, max_features=lexical.MAX_FEATURES,
                                       lowercase=True, strip_accents="unicode", dtype=np.float32)
        self.full = self.vec.fit_transform(list(texts)).tocsr()
        self.title = self.body = None
        self.texts = texts

    def split_fields(self) -> None:
        titles = [t.split("\n", 1)[0] for t in self.texts]
        bodies = [t.split("\n", 1)[1] if "\n" in t else "" for t in self.texts]
        self.title = self.vec.transform(titles).tocsr()
        self.body = self.vec.transform(bodies).tocsr()

    def docs(self, k1: float, b: float, wt: float = 1.0) -> sparse.csr_matrix:
        counts = self.full if wt == 1.0 else (self.title * np.float32(wt) + self.body).tocsr()
        counts.eliminate_zeros()
        return _bm25_document_matrix(counts, k1=k1, b=b).T.tocsr()

    def queries(self, texts: Sequence[str]) -> sparse.csr_matrix:
        q = self.vec.transform(list(texts)).astype(np.float32).tocsr()
        q.data[:] = 1.0
        return q


@dataclass
class Arm:
    name: str
    texts: list[str]
    k1: Sequence[float]
    b: Sequence[float]
    title_w: Sequence[float] = (1.0,)


class Runner:
    def __init__(self, test: str) -> None:
        self.test = test
        self.ctx = base.Ctx()
        gc.assert_dev(self.ctx.U)
        self.R = gc.dev_pspec_rows()
        upos = {int(r): i for i, r in enumerate(self.ctx.U)}
        self.r_in_u = np.asarray([upos[int(r)] for r in self.R])
        self.qtexts = [self.ctx.texts[int(r)] for r in self.ctx.U]
        self.comp = np.asarray([self.ctx.data.queries[int(r)].component_id for r in self.R])

    def table(self, score_fn) -> np.ndarray:
        """[nU, 2, 4] per-query metrics of a scorer over U rows (row blocks of 256)."""
        out = np.full((self.ctx.nU, 2, 4), np.nan)
        for sl in self.ctx.blocks(self.ctx.nU):
            out[sl] = self.ctx.query_metrics(score_fn(sl), self.ctx.U[sl])
        return out

    def run_arm(self, arm: Arm) -> dict[str, Any]:
        t0 = time.time()
        ctx = self.ctx
        char_idx = lexical.LexicalIndex(arm.texts, kind="char")
        zc_cache: dict[int, np.ndarray] = {}

        def char_block(sl):
            return char_idx.score(self.qtexts[sl])

        tables: dict[Any, np.ndarray] = {"char": self.table(char_block)}
        chans = {}
        raw_keys = []
        for kind in ("bm25", "bm25_stem"):
            ch = chans[kind] = Bm25Channel(arm.texts, stem=(kind == "bm25_stem"))
            if kind == "bm25_stem" and any(w != 1.0 for w in arm.title_w):
                ch.split_fields()
            qmat = ch.queries(self.qtexts)
            wts = arm.title_w if kind == "bm25_stem" else (1.0,)
            for wt in wts:
                for k1 in arm.k1:
                    for b in arm.b:
                        D = ch.docs(k1, b, wt)
                        tables[(kind, k1, b, wt)] = self.table(lambda sl: (qmat[sl] @ D).toarray().astype(np.float32))
                        raw_keys.append((kind, k1, b, wt))
            gc.log(self.test, f"{arm.name}: {kind} raw tables done ({time.time() - t0:.0f}s)")
        # per-fold selection, pass 1 (BM25 configs), then fused tables only for configs some fold selected
        sel: dict[int, dict[str, Any]] = {}
        for g in ctx.groups:
            if g.is_lockbox:
                continue
            train = ctx.train_mask_u(g)
            obj = lambda t: ctx.objective(t, train)  # noqa: E731
            chosen = {}
            for kind in ("bm25", "bm25_stem"):
                keys = [k for k in raw_keys if k[0] == kind]
                vals = [obj(tables[k]) for k in keys]
                i = base.argbest(vals)
                chosen[kind] = (keys[i], vals[i])
            pick = base.argbest([chosen["bm25"][1], chosen["bm25_stem"][1]])
            sel[g.idx] = {"role": g.role, "chosen": chosen, "base": chosen[("bm25", "bm25_stem")[pick]][0]}
        need = sorted({s["base"] for s in sel.values()})
        for key in need:
            kind, k1, b, wt = key
            ch = chans[kind]
            qmat = ch.queries(self.qtexts)
            D = ch.docs(k1, b, wt)
            fused_blocks = {w: np.full((ctx.nU, 2, 4), np.nan) for w in W_CHAR}
            for sl in ctx.blocks(ctx.nU):
                zr = fusion.robust_z((qmat[sl] @ D).toarray().astype(np.float32))
                zc = fusion.robust_z(char_idx.score(self.qtexts[sl]))
                for w in W_CHAR:
                    fused = fusion.linear_fusion([zr, zc], base.pair(w), normalize="none")
                    fused_blocks[w][sl] = ctx.query_metrics(fused, ctx.U[sl])
            for w in W_CHAR:
                tables[("fuse",) + key + (w,)] = fused_blocks[w]
        gc.log(self.test, f"{arm.name}: fused tables for {len(need)} base configs ({time.time() - t0:.0f}s)")
        # pass 2: fusion weight and the L* option, then held-out values
        heldout = {k: np.full(ctx.nU, np.nan) for k in ("bm25", "bm25_stem", "bm25_best", "char", "L_star")}
        picks = {}
        for g in ctx.groups:
            if g.is_lockbox:
                continue
            s = sel[g.idx]
            train = ctx.train_mask_u(g)
            obj = lambda t: ctx.objective(t, train)  # noqa: E731
            fvals = [obj(tables[("fuse",) + s["base"] + (w,)]) for w in W_CHAR]
            fi = base.argbest(fvals)
            options = [s["chosen"]["bm25"][1], s["chosen"]["bm25_stem"][1], obj(tables["char"]), fvals[fi]]
            star = base.argbest(options)
            keys = [s["chosen"]["bm25"][0], s["chosen"]["bm25_stem"][0], "char", ("fuse",) + s["base"] + (W_CHAR[fi],)]
            held = ctx.foldU == int(g.role.replace("dev_fold", ""))
            for name, key in (("bm25", keys[0]), ("bm25_stem", keys[1]), ("bm25_best", s["base"]), ("char", "char"),
                              ("L_star", keys[star])):
                heldout[name][held] = tables[key][held, 1, 0]
            picks[g.role] = {"bm25": keys[0], "bm25_stem": keys[1], "w_char": W_CHAR[fi],
                             "L_star": ["bm25", "bm25_stem", "char", "fuse"][star], "L_star_key": keys[star]}
        vals = {k: v[self.r_in_u] for k, v in heldout.items()}
        for k, v in vals.items():
            if not np.isfinite(v).all():
                raise ValueError(f"{arm.name}/{k}: missing held-out values on P_spec rows")
        out = {"ndcg_at_10": {k: gc.macro(v, self.comp) for k, v in vals.items()}, "picks": picks,
               "seconds": time.time() - t0, "configs": len(raw_keys)}
        gc.log(self.test, f"{arm.name}: {json.dumps(out['ndcg_at_10'])} picks "
                          f"{ {r: (p['L_star_key'], p['w_char']) for r, p in picks.items()} }")
        del tables, chans, char_idx
        return {"summary": out, "per_case": vals}

    def contrasts(self, arms: dict[str, dict[str, Any]], ref: str) -> dict[str, Any]:
        out = {}
        for name, a in arms.items():
            if name == ref:
                continue
            for ch in ("L_star", "bm25_best", "bm25_stem"):
                out[f"{name} - {ref} ({ch})"] = gc.contrast(a["per_case"][ch], arms[ref]["per_case"][ch], self.comp)
        return out


def gate(test: str, summary: dict[str, Any]) -> None:
    got = summary["ndcg_at_10"]
    diff = {k: abs(got[k] - v) for k, v in GATE.items()}
    gc.log(test, f"reproduction gate: {diff}")
    if max(diff.values()) > 0.003:
        raise SystemExit(f"reproduction gate failed: {diff}")


def grid() -> int:
    test = "t3_bm25_grid"
    t0 = time.time()
    run = Runner(test)
    texts = list(run.ctx.data.candidate_texts)
    arms: dict[str, dict[str, Any]] = {}
    arms["original"] = run.run_arm(Arm("original", texts, ORIG_K1, ORIG_B))
    gate(test, arms["original"]["summary"])
    arms["extended"] = run.run_arm(Arm("extended", texts, EXT_K1, EXT_B))
    arms["extended_title_w"] = run.run_arm(Arm("extended_title_w", texts, EXT_K1, EXT_B, TITLE_W))
    res = {"arms": {k: v["summary"] for k, v in arms.items()}, "contrasts": run.contrasts(arms, "original"),
           "grids": {"original": {"k1": ORIG_K1, "b": ORIG_B}, "extended": {"k1": EXT_K1, "b": EXT_B},
                     "title_w": TITLE_W, "w_char": W_CHAR},
           "rows": {"U_population_P": int(run.ctx.nU), "P_spec": int(len(run.R)), "components": int(len(set(run.comp)))},
           "cpu_seconds": time.time() - t0, "peak_gb": gc.peak_gb()}
    for k, v in res["contrasts"].items():
        gc.log(test, f"{k}: {gc.fmt(v)}")
    gc.write_json(gc.OUT / test / "result.json", res)
    return 0


def variant_texts(texts: list[str]) -> tuple[dict[str, list[str]], dict[str, Any]]:
    paths = json.loads(PATHS.read_text(encoding="utf-8"))
    title = [t.split("\n", 1)[0] for t in texts]
    body = [t.split("\n", 1)[1] if "\n" in t else "" for t in texts]
    v2, with_path = [], 0
    for i, (ti, bo) in enumerate(zip(title, body)):
        p = paths.get(str(i))
        if p:
            with_path += 1
            v2.append(f"{ti}\n{' / '.join(p)}\n{bo}")
        else:
            v2.append(texts[i])
    v3, n_notes, dof_before, dof_after, touched = [], 0, 0, 0, 0
    for t in texts:
        s, n = DOF_NOTE.subn("", t)
        s, n2 = DOF_DATE.subn("", s)
        n += n2
        n_notes += n
        touched += n > 0
        dof_before += len(re.findall(r"\bDOF\b", t))
        dof_after += len(re.findall(r"\bDOF\b", s))
        v3.append(s)
    info = {"articles_with_heading_path": with_path, "dof_notes_removed": n_notes, "articles_with_notes_removed": touched,
            "dof_tokens_before": dof_before, "dof_tokens_after": dof_after,
            "examples_stripped": [(texts[i][-300:], v3[i][-200:]) for i in range(len(texts))
                                  if "DOF" in texts[i]][:3]}
    return {"V0_current": texts, "V1_body_only": body, "V2_hierarchy_path": v2, "V3_dof_stripped": v3}, info


def variants() -> int:
    test = "t4_text_variants"
    t0 = time.time()
    run = Runner(test)
    texts = list(run.ctx.data.candidate_texts)
    var, info = variant_texts(texts)
    gc.log(test, f"variants: {json.dumps({k: v for k, v in info.items() if k != 'examples_stripped'})}")
    arms: dict[str, dict[str, Any]] = {}
    for name, tx in var.items():
        arms[name] = run.run_arm(Arm(name, tx, ORIG_K1, ORIG_B))
        if name == "V0_current":
            gate(test, arms[name]["summary"])
    res = {"arms": {k: v["summary"] for k, v in arms.items()}, "contrasts": run.contrasts(arms, "V0_current"),
           "variant_info": info, "rows": {"U_population_P": int(run.ctx.nU), "P_spec": int(len(run.R))},
           "cpu_seconds": time.time() - t0, "peak_gb": gc.peak_gb()}
    for k, v in res["contrasts"].items():
        gc.log(test, f"{k}: {gc.fmt(v)}")
    gc.write_json(gc.OUT / test / "result.json", res)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    sys.exit({"grid": grid, "variants": variants}[cmd]())
