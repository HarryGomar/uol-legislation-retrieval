"""Round 2, workstream A: shared helpers (EXPLORATORY analysis on frozen EP-alpha systems).

Lockbox rows are evaluated only for score files frozen in EP-alpha (FREEZE.json pipelines or hash-pinned
upstream score files), descriptively. New signals (P12) are evaluated on dev cross-fitting only.
"""
from __future__ import annotations

import json
import pickle
import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.embedding_push import common
from uol.embedding_push import fusion_ltr as fl
from uol.facts_to_law import prior as prior_mod
from uol.facts_to_law import stats
from uol.facts_to_law.spec import ROOT

OUT = ROOT / "experiments/runs/round2_20261004/A"
CACHE = OUT / "cache"
REPORT = ROOT / "research/reports/round2_20261004/A_analysis.md"
SOURCES = ROOT / "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"
SOURCE_HOME = Path(r"C:\uol_legacy_home")
BANDS = ("frequent", "rare", "unseen")
LOG2 = 1.0 / np.log2(np.arange(2, 12))
IDCG = np.concatenate([[0.0], np.cumsum(LOG2[:10])])
FROZEN_LOCKBOX = {"LTR_emb_hgb", "LTR_noemb_hgb", "LTR_emb_hgb_full", "LTR_noemb_hgb_full", "LTR_noemb_nolsa_hgb_full",
                  "L_star", "K_lex", "dense_qwen8b", "prior"}


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (OUT / "run.log").open("a", encoding="utf-8") as h:
        h.write(line + "\n")


def score_file(name: str) -> Path:
    return fl.pipeline_file(name)


def frozen_ok(name: str) -> bool:
    """A system may be evaluated on the lockbox only if its score file is the one pinned in FREEZE.json."""
    freeze = json.loads(common.FREEZE_PATH.read_text(encoding="utf-8"))
    pinned = {}
    for fam in freeze["pipelines"].values():
        for rec in fam.values():
            pinned[rec["pipeline"]] = rec["scores_sha256"]
    for sid, rec in (freeze.get("upstream_score_files") or {}).items():
        sha = (rec.get("sha256") or rec.get("scores_sha256")) if isinstance(rec, dict) else rec
        if sha:
            pinned.setdefault(sid, sha)
    if name not in pinned:
        return False
    man = score_file(name).parent / "manifest.json"
    sha = json.loads(man.read_text(encoding="utf-8")).get("scores_sha256")
    return sha == pinned[name]


def evaluate_cached(name: str, split: str, population: str = "P_spec") -> dict[str, Any]:
    """common.evaluate on dev or (frozen systems only) lockbox rows of a population, cached as pickle."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"eval_{name}_{split}_{population}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    if split == "lockbox":
        if name not in FROZEN_LOCKBOX or not frozen_ok(name):
            raise PermissionError(f"{name}: lockbox metrics only for hash-pinned frozen EP-alpha systems")
        base_rows = common.lockbox_rows()
    else:
        base_rows = common.dev_rows()
    rows = common.population_rows(population, base_rows)
    mm = np.load(score_file(name), mmap_mode="r")
    scores = np.asarray(mm[rows], dtype=np.float32)
    res = common.evaluate(scores, rows, population, top_k=0, allow_lockbox=(split == "lockbox"))
    for rec in res["per_query"]:
        rec.pop("topk", None)
    path.write_bytes(pickle.dumps(res))
    return res


@lru_cache(maxsize=1)
def fold_band_arrays() -> dict[int, np.ndarray]:
    """Fold-local band per candidate: from the other dev folds' components (dev rows of fold k)."""
    d = common.release()
    roles = common.roles()
    out = {}
    for k in range(5):
        ids = [d.queries[int(r)].query_id for r in common.dev_rows() if roles[int(r)] != f"dev_fold{k}"]
        counts = prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)
        out[k] = np.where(counts == 0, "unseen", np.where(counts <= prior_mod.RARE_MAX, "rare", "frequent"))
    return out


@lru_cache(maxsize=1)
def lockbox_band_array() -> np.ndarray:
    d = common.release()
    ids = [d.queries[int(r)].query_id for r in common.dev_rows()]
    counts = prior_mod.component_counts(d.references, d.component_by_query, ids, len(d.candidate_ids), d.candidate_index)
    return np.where(counts == 0, "unseen", np.where(counts <= prior_mod.RARE_MAX, "rare", "frequent"))


def band_labels(row: int) -> np.ndarray:
    role = common.roles()[int(row)]
    if role == "lockbox":
        return lockbox_band_array()
    return fold_band_arrays()[int(str(role).replace("dev_fold", ""))]


def component_values(res: dict[str, Any], values: Sequence[float], keep: Sequence[bool] | None = None) -> tuple[list[str], np.ndarray]:
    """Component-macro of per-query values (aligned with res['per_query']), optionally restricted to a query mask."""
    d = common.release()
    comps: dict[str, list[float]] = {}
    for j, row in enumerate(res["rows"]):
        if keep is not None and not keep[j]:
            continue
        comps.setdefault(d.queries[int(row)].component_id, []).append(float(values[j]))
    ids = sorted(comps)
    return ids, np.asarray([np.mean(comps[c]) for c in ids])


def subset_contrast(a: dict[str, Any], b: dict[str, Any], keep: Sequence[bool] | None, metric: str = "ndcg_at_10",
                    draws: int = stats.DRAWS) -> dict[str, Any]:
    """Paired component-bootstrap contrast a - b on a query subset (a and b evaluated on identical rows)."""
    assert a["rows"] == b["rows"]
    va = [q[metric] for q in a["per_query"]]
    vb = [q[metric] for q in b["per_query"]]
    ids, ca = component_values(a, va, keep)
    _, cb = component_values(b, vb, keep)
    n_q = int(np.sum(keep)) if keep is not None else len(a["rows"])
    if len(ids) < 2:
        return {"n_queries": n_q, "n_components": len(ids), "mean_a": float(ca.mean()) if len(ca) else None,
                "mean_b": float(cb.mean()) if len(cb) else None, "effect": None, "ci95": None}
    c = stats.contrast(ca, cb, draws=draws)
    return {"n_queries": n_q, "n_components": len(ids), "mean_a": c["mean_a"], "mean_b": c["mean_b"], "effect": c["effect"],
            "ci95": c["ci95"], "verdict": c["verdict"]}


def subset_mean(res: dict[str, Any], keep: Sequence[bool] | None, metric: str = "ndcg_at_10") -> float:
    _, cv = component_values(res, [q[metric] for q in res["per_query"]], keep)
    return float(cv.mean()) if len(cv) else float("nan")


def ndcg_contrib(rank: int, n_rel: int) -> float:
    return (1.0 / np.log2(rank + 1) if rank <= 10 else 0.0) / IDCG[min(n_rel, 10)]


# ----------------------------------------------------------------------------- text helpers

STOP = set("""a al algo algun alguna algunas alguno algunos ante antes aquel aquella aquellas aquello aquellos asi aun aunque cada
como con contra cual cuales cualquier cuando cuyo de del desde donde dos el ella ellas ello ellos en entre era eran es esa esas ese eso
esos esta estas este esto estos fue fueron ha han hasta hay la las le les lo los mas me mediante mismo mucho muy ni no nos o otra otras
otro otros para pero poco por porque que quien quienes se segun ser si sido sin sino sobre su sus tal tambien tanto te tiene tienen
todo todos tras tu un una uno unos ya y articulo articulos fraccion fracciones parrafo ley leyes codigo""".split())
TOKEN = re.compile(r"[a-z0-9]+")


def fold_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in text if not unicodedata.combining(ch)).lower()


def content_tokens(text: str) -> set[str]:
    return {t for t in TOKEN.findall(fold_text(text)) if len(t) >= 3 and t not in STOP and not t.isdigit()}


def load_sources() -> dict[str, dict[str, Any]]:
    """query_id -> structured source row (dates, rubro, epoca, organo); read once."""
    path = CACHE / "sources_meta.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    out = {}
    with SOURCES.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            out[r["query_id"]] = {k: r.get(k) for k in ("publication_date", "source_rubro", "source_epoca", "source_organo",
                                                         "source_instancia", "source_materias", "source_tipo_tesis", "record_id",
                                                         "raw_published_text")}
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out))
    return out


def fmt(x: float | None, nd: int = 3, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(c: list | None, nd: int = 3) -> str:
    return "n/a" if not c else f"[{c[0]:+.{nd}f}, {c[1]:+.{nd}f}]"


def save_json(name: str, obj: Any) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    return p
