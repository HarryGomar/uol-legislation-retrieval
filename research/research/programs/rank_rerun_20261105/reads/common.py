"""Shared read-only helpers for READS.md. Development cases only: every table is keyed on the query ids of the stored
DEV per-case files; nothing under dev/store is written, no model is called, nothing is refitted, no test file is opened."""
from __future__ import annotations
import os, sys, json, re
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(k, "4")
from functools import lru_cache
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
FOOT = ROOT / "experiments/runs/footing_20261101/dev/store/runs"
AUDIT = ROOT / "experiments/runs/audit_d181_20261028/dev/store/runs"
RECORD = ROOT / "experiments/runs/final_20261025/store_dev_20261025b/runs"
RECEIPTS = {FOOT: FOOT.parent / "receipts", RECORD: RECORD.parent / "receipts", AUDIT: AUDIT.parent / "receipts"}
PAIRS = ROOT / "research/programs/final_results_20261102/depends_pairs.parquet"
DEPENDS = ROOT / "research/programs/final_results_20261102/depends.json"
RESULTS = ROOT / "research/programs/final_results_20261102/results.json"
OUTDIR = Path(__file__).resolve().parent / "out"
OUTDIR.mkdir(exist_ok=True)
RANKER, NEWRANKER, ALONE, SYSTEM = "ft_lean", "rt_lm_tesis", "gpt61sol_alone", "gpt61sol_combo_fwd_on_ft_lean"
LEX, ENC, NOENC = "lstar_headings", "qwen3_8b", "ft_lean_noenc"
BANDS = ["0", "1-2", "3-10", ">10"]


def store_of(sid: str) -> Path:
    for s in (FOOT, RECORD, AUDIT):        # the same order as v21_figs.store_of
        if (s / sid / "DEV" / "summary.json").exists():
            return s
    raise FileNotFoundError(sid)


def summary(sid: str) -> dict:
    return json.loads((store_of(sid) / sid / "DEV" / "summary.json").read_text(encoding="utf-8"))


def receipt(sid: str) -> dict:
    s = store_of(sid)
    return json.loads((RECEIPTS[s] / f"{sid}__DEV.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=64)
def pc(sid: str) -> pd.DataFrame:
    return pd.read_parquet(store_of(sid) / sid / "DEV" / "per_case.parquet")


@lru_cache(maxsize=32)
def top(sid: str) -> pd.DataFrame:
    return pd.read_parquet(store_of(sid) / sid / "DEV" / "top200.parquet")


def first(sid: str, k: int) -> dict[str, list[int]]:
    t = top(sid)
    t = t[t["rank"] <= k].sort_values(["query_id", "rank"])
    return {q: g["candidate_idx"].astype(int).tolist() for q, g in t.groupby("query_id")}


def ints(v) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", str(v))]


def gm(values, groups) -> float:
    """Mean over groups (component_id), the thesis's summary."""
    return float(pd.Series(np.asarray(values, float)).groupby(np.asarray(groups)).mean().mean())


@lru_cache(maxsize=1)
def pairs() -> pd.DataFrame:
    return pd.read_parquet(PAIRS)[["query_id", "article", "component_id", "share", "quarter", "band"]]


@lru_cache(maxsize=64)
def rank_map(sid: str) -> dict:
    d = pc(sid)
    return {(q, a): r for q, i, rr in zip(d["query_id"], d["positive_idx"], d["positive_ranks"])
            for a, r in zip(ints(i), ints(rr))}


def pair_rank(sid: str) -> np.ndarray:
    m = rank_map(sid)
    p = pairs()
    return np.array([m.get((q, int(a)), 0) for q, a in zip(p["query_id"], p["article"])], dtype=float)


def pair_hit(sid: str, k: int = 20) -> np.ndarray:
    r = pair_rank(sid)
    return ((r > 0) & (r <= k)).astype(float)


class Boot:
    """Cluster bootstrap over groups, 2,000 draws by default, fixed seed."""

    def __init__(self, groups, n=int(os.environ.get("UOL_READ_DRAWS", 2000)), seed=20261105):
        self.g, self.idx = np.unique(np.asarray(groups), return_inverse=True)
        self.n = len(self.g)
        rng = np.random.default_rng(seed)
        self.w = np.stack([np.bincount(rng.integers(0, self.n, self.n), minlength=self.n) for _ in range(n)]).astype(float)

    def ratio(self, num, den):
        """Pooled ratio sum(num)/sum(den) with the interval."""
        a = np.bincount(self.idx, np.asarray(num, float), self.n)
        b = np.bincount(self.idx, np.asarray(den, float), self.n)
        s = (self.w @ a) / np.maximum(self.w @ b, 1e-12)
        return float(a.sum() / max(b.sum(), 1e-12)), float(np.quantile(s, .025)), float(np.quantile(s, .975))

    def diff(self, num_a, num_b, den):
        a = np.bincount(self.idx, np.asarray(num_a, float), self.n)
        c = np.bincount(self.idx, np.asarray(num_b, float), self.n)
        b = np.bincount(self.idx, np.asarray(den, float), self.n)
        s = (self.w @ (a - c)) / np.maximum(self.w @ b, 1e-12)
        return float((a.sum() - c.sum()) / max(b.sum(), 1e-12)), float(np.quantile(s, .025)), float(np.quantile(s, .975))

    def group_mean(self, values, mask=None):
        """Mean over groups of the case values (optionally of the masked cases), with the interval."""
        v = np.asarray(values, float)
        m = np.ones(len(v), bool) if mask is None else np.asarray(mask, bool)
        s = np.bincount(self.idx, np.where(m, v, 0.0), self.n)
        c = np.bincount(self.idx, m.astype(float), self.n)
        has = c > 0
        mean = np.where(has, s / np.maximum(c, 1), 0.0)
        est = (self.w @ mean) / np.maximum(self.w @ has.astype(float), 1e-12)
        return float(mean[has].mean()), float(np.quantile(est, .025)), float(np.quantile(est, .975))


def f(x, n=3, sign=False):
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ci(t, n=3, sign=False):
    return f"{f(t[0], n, sign)} [{f(t[1], n, sign)}, {f(t[2], n, sign)}]"


@lru_cache(maxsize=1)
def release():
    from uol.embedding_push import common as ep
    return ep.release()


def dev_qids() -> list[str]:
    return pc(RANKER)["query_id"].tolist()


def query_text() -> dict[str, str]:
    """Facts of the 1,217 development cases, only those."""
    d = release()
    want = set(dev_qids())
    return {q.query_id: q.text for q in d.queries if q.query_id in want}


def law_key(cid: str) -> str:
    return cid.split("#")[0]


def article_law() -> np.ndarray:
    return np.array([law_key(c) for c in release().candidate_ids])


def law_title() -> dict[str, str]:
    d = release()
    out = {}
    for c, t in zip(d.candidate_ids, d.candidate_texts):
        out.setdefault(law_key(c), t.split("\n", 1)[0].strip()[:70])
    return out


WORD = re.compile(r"[a-záéíóúüñ]+")


def save(name: str, obj) -> None:
    (OUTDIR / f"{name}.json").write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
