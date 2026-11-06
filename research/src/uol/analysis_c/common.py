"""Shared loading, bands, bootstrap and writers for the ANALYSIS_PLAN_C analyses (D93; DEV, descriptive).

Inputs are saved outputs only: ``<runs>/<system>/<population>/{per_case,top200}.parquet`` (RUN-A's mirror now, RUN-B's
merged store later; same schema) and the DEV-only data pack (``--data``, default the bundle staging folder:
``release/candidates/{articles,instruments}.jsonl``, ``release/queries.jsonl``, ``round3/statute_graph``). The data
pack holds development rows only, so no test-set row is ever loaded; every row list is the per-case files' own.

Conventions (ANALYSIS_PLAN_C §1):

* **Labels.** A case's cited articles are the ``positive_idx`` of the per-case files (the specific positives scored in
  P_spec, label corrections applied by the runner). One reference system fixes the case list; systems whose labels
  differ on a shared case are reported (``RunStore.label_mismatch``) and analysed with the reference labels.
* **Means.** Component-macro (cases averaged within their group, then groups), the convention of every stored summary.
* **Intervals.** Cluster bootstrap over groups (``component_id``): 2,000 draws, seed 20261023, 95 % percentile
  intervals. One draw matrix per population, so every system and measure is resampled with the same groups (paired).
* **Bands.** Fold-local: an article's count for a case of outer fold k is the number of DEV cases of the other four
  folds that cite it (specific positives). Bins 0, 1-2, 3-10, >10.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__import__("os").environ.get("UOL_ANALYSIS_DATA") or ROOT / ".cache/final_bundle_staging/data")
SEED = 20261023
DRAWS = 2000
SCOPE = "DEV, descriptive (ANALYSIS_PLAN_C)"
BANDS = ("0", "1-2", "3-10", ">10")
TOP_K = 200
DISC = 1.0 / np.log2(np.arange(2, 12))          # ranks 1..10
IDCG = np.concatenate([[0.0], np.cumsum(DISC)])


# ----------------------------------------------------------------------------- text helpers

def fold(text: str) -> str:
    """Lower case without accents (the plan's 'accent- and case-insensitive')."""
    t = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in t if not unicodedata.combining(ch)).lower()


def parse_idx(value: Any) -> np.ndarray:
    """``positive_idx`` / ``positive_ranks`` cell ('15140', '1 2', '1,2') as an int array."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return np.zeros(0, dtype=np.int64)
    s = str(value).replace(",", " ").split()
    return np.asarray([int(x) for x in s], dtype=np.int64)


def words(text: str) -> int:
    """Words = whitespace-separated tokens (the QE convention, ``len(text.split())``)."""
    return len((text or "").split())


# ----------------------------------------------------------------------------- catalogue and queries

@dataclass
class Catalogue:
    """The bound catalogue in candidate order (39,178 articles of 264 laws)."""
    candidate_ids: list[str]
    law: np.ndarray                      # law code per article (sorted instrument ids, as uol.final.measures)
    law_ids: list[str]                   # instrument id per code
    law_titles: list[str]                # title per code
    texts: list[str]                     # instrument_article_body (law title line + article), the scored text
    body_words: np.ndarray               # words of article_body
    paths: dict[int, list[str]] = field(default_factory=dict)        # heading path per article index
    edges: dict[str, np.ndarray] = field(default_factory=dict)       # statute-graph edge type -> (E, 2) index pairs
    generic: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    data: str = ""                                                   # the data pack it was read from

    @property
    def n(self) -> int:
        return len(self.candidate_ids)

    @property
    def law_size(self) -> np.ndarray:
        return np.bincount(self.law, minlength=len(self.law_ids))


@lru_cache(maxsize=2)
def load_catalogue(data: str | Path = DATA) -> Catalogue:
    data = Path(data)
    ids, texts, body_words, inst = [], [], [], []
    with (data / "release/candidates/articles.jsonl").open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            ids.append(r["candidate_id"])
            texts.append(r.get("instrument_article_body") or "")
            body_words.append(words(r.get("article_body") or ""))
            inst.append(r.get("instrument_id") or r["candidate_id"].split("#", 1)[0])
    names = [c.split("#", 1)[0] for c in ids]
    if names != inst:
        raise ValueError("instrument_id disagrees with the candidate id prefix")
    law_ids = sorted(set(names))
    code = {v: i for i, v in enumerate(law_ids)}
    titles = {law: law for law in law_ids}
    ip = data / "release/candidates/instruments.jsonl"
    if ip.exists():
        with ip.open(encoding="utf-8") as h:
            for line in h:
                r = json.loads(line)
                titles[r["instrument_id"]] = r.get("title") or r["instrument_id"]
    cat = Catalogue(ids, np.asarray([code[v] for v in names], dtype=np.int32), law_ids, [titles[x] for x in law_ids],
                    texts, np.asarray(body_words, dtype=np.int64))
    cat.data = str(data)
    pp = data / "round3/statute_graph/paths.json"
    if pp.exists():
        cat.paths = {int(k): list(v) for k, v in json.loads(pp.read_text(encoding="utf-8")).items()}
    gp = data / "round3/statute_graph/graph.npz"
    if gp.exists():
        g = np.load(gp)
        if len(g["inst"]) != cat.n or not (g["inst"] == cat.law).all():
            raise ValueError("statute graph is not aligned with the catalogue")
        cat.edges = {k: np.asarray(g[k], dtype=np.int64) for k in g.files if k.startswith("e_")}
    tp = data / "release/target_types.jsonl"
    cat.generic = _generic(tp, cat) if tp.exists() else np.zeros(0, dtype=np.int64)
    return cat


def _generic(path: Path, cat: Catalogue) -> np.ndarray:
    index = {c: i for i, c in enumerate(cat.candidate_ids)}
    out = []
    with path.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if str(r.get("target_type", r.get("type", ""))).lower() == "generic" and r.get("candidate_id") in index:
                out.append(index[r["candidate_id"]])
    return np.asarray(sorted(out), dtype=np.int64)


@lru_cache(maxsize=2)
def load_queries(data: str | Path = DATA) -> dict[str, dict[str, Any]]:
    """query_id -> {text, materias, fold, component_id} from the DEV-only data pack."""
    out = {}
    with (Path(data) / "release/queries.jsonl").open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            out[r["query_id"]] = {"text": r.get("text") or "", "materias": list(r.get("materias") or []),
                                  "fold": r.get("fold"), "component_id": r.get("component_id")}
    return out


# ----------------------------------------------------------------------------- runs

@dataclass
class Labels:
    """The case list of one population, sorted by query_id, with the reference cited articles."""
    query_id: np.ndarray
    component: np.ndarray
    fold: np.ndarray
    positives: list[np.ndarray]
    reference: str

    @property
    def n(self) -> int:
        return len(self.query_id)

    def pair_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(case index, article index) of every cited pair, in case order."""
        case = np.concatenate([np.full(len(p), i, dtype=np.int64) for i, p in enumerate(self.positives)]) \
            if self.n else np.zeros(0, dtype=np.int64)
        art = np.concatenate(self.positives) if self.n else np.zeros(0, dtype=np.int64)
        return case, art.astype(np.int64)


class RunStore:
    """Read-only view of ``<runs>/<system>/<population>/``; per-case frames and top-200 matrices in the label order."""

    def __init__(self, runs: Path | str, population: str = "DEV", reference: str | None = None):
        self.runs = Path(runs)
        self.population = population
        self.systems = sorted(p.parent.parent.name for p in self.runs.glob(f"*/{population}/per_case.parquet"))
        if not self.systems:
            raise FileNotFoundError(f"no {population} per_case.parquet under {self.runs}")
        self.labels = self._labels(reference)
        self._pos = {q: i for i, q in enumerate(self.labels.query_id)}
        self.label_mismatch: dict[str, int] = {}
        self._cache: dict[str, pd.DataFrame] = {}

    def has(self, system: str) -> bool:
        return system in self.systems

    def source(self, system: str, name: str = "per_case.parquet") -> str:
        return f"runs/{system}/{self.population}/{name}"

    def _labels(self, reference: str | None) -> Labels:
        if reference is None:                        # the label set most systems share; untagged first; then by name
            sig: dict[str, int] = {}
            for s in self.systems:
                f = pd.read_parquet(self.runs / s / self.population / "per_case.parquet", columns=["query_id", "positive_idx"])
                f = f.sort_values("query_id")
                sig[s] = hash(tuple(zip(f["query_id"].astype(str), f["positive_idx"].astype(str))))
            votes = pd.Series(sig).value_counts()
            reference = sorted(self.systems, key=lambda s: (-votes[sig[s]], "@" in s, s))[0]
        f = pd.read_parquet(self.runs / reference / self.population / "per_case.parquet").sort_values("query_id")
        return Labels(f["query_id"].to_numpy(dtype=str), f["component_id"].to_numpy(dtype=str),
                      f["fold"].to_numpy(dtype=np.int64), [parse_idx(v) for v in f["positive_idx"]], reference)

    def cases(self, system: str) -> pd.DataFrame:
        """The system's per-case frame reindexed to the label order (missing cases are NaN rows, ``present`` False)."""
        if system in self._cache:
            return self._cache[system]
        f = pd.read_parquet(self.runs / system / self.population / "per_case.parquet")
        f = f.drop_duplicates("query_id").set_index("query_id")
        out = f.reindex(self.labels.query_id)
        out["present"] = out.index.isin(f.index)
        mism = 0
        for i, (ok, v) in enumerate(zip(out["present"].to_numpy(), out["positive_idx"].to_numpy())):
            if ok and not np.array_equal(np.sort(parse_idx(v)), np.sort(self.labels.positives[i])):
                mism += 1
        if mism:
            self.label_mismatch[system] = mism
        out = out.reset_index()
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[system] = out
        return out

    def top(self, system: str, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
        """(candidate index [n, k], score [n, k]) in the system's rank order; -1 / NaN where a case has no row."""
        t = pd.read_parquet(self.runs / system / self.population / "top200.parquet",
                            columns=["query_id", "rank", "candidate_idx", "score"])
        t = t[t["rank"] <= k]
        pos = t["query_id"].map(self._pos)
        t = t[pos.notna()]
        r = pos[pos.notna()].to_numpy(dtype=np.int64)
        c = t["rank"].to_numpy(dtype=np.int64) - 1
        idx = np.full((self.labels.n, k), -1, dtype=np.int64)
        sc = np.full((self.labels.n, k), np.nan, dtype=np.float64)
        idx[r, c] = t["candidate_idx"].to_numpy(dtype=np.int64)
        sc[r, c] = t["score"].to_numpy(dtype=np.float64)
        return idx, sc


def membership(idx: np.ndarray, positives: Sequence[np.ndarray], n_articles: int) -> np.ndarray:
    """Boolean [n, k]: the listed article is cited by the case."""
    keys = np.concatenate([i * n_articles + p for i, p in enumerate(positives)]) if len(positives) else np.zeros(0)
    keys = np.unique(keys.astype(np.int64))
    q = np.arange(idx.shape[0], dtype=np.int64)[:, None] * n_articles + idx
    return np.isin(q, keys) & (idx >= 0)


def ndcg10(gain: np.ndarray, n_rel: np.ndarray) -> np.ndarray:
    """nDCG@10 from a gain matrix [n, >=10] and the number of cited articles (the strict ideal list)."""
    g = np.zeros((gain.shape[0], 10))
    w = min(10, gain.shape[1])
    g[:, :w] = gain[:, :w]
    dcg = (g * DISC).sum(axis=1)
    idcg = IDCG[np.minimum(np.asarray(n_rel, dtype=np.int64), 10)]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1), np.nan)


# ----------------------------------------------------------------------------- bands

def dev_citations(data: str | Path, cat: "Catalogue") -> list[tuple[int, str, np.ndarray]] | None:
    """(fold, query_id, cited article indices) for every development case of the data pack (``release/queries.jsonl``
    and ``release/references.jsonl``); None when the pack has no references file."""
    ref_p = Path(data) / "release" / "references.jsonl"
    if not ref_p.exists():
        return None
    index = {c: i for i, c in enumerate(cat.candidate_ids)}
    qs = load_queries(str(data))
    out = []
    with ref_p.open(encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            q = qs.get(r["query_id"])
            if q is None or q.get("fold") is None:
                continue
            idx = [index[p["candidate_id"]] for p in r.get("positives", []) if p.get("candidate_id") in index]
            out.append((int(q["fold"]), r["query_id"], np.asarray(sorted(set(idx)), dtype=np.int64)))
    return out


def fold_local_counts(labels: Labels, n_articles: int, folds: Iterable[int] = range(5),
                      cat: "Catalogue | None" = None) -> dict[int, np.ndarray]:
    """fold k -> per article, the number of DEV cases of the other folds citing it.

    With the data pack (``cat.data``), every development case of the other folds counts (ANALYSIS_PLAN_C §1, "DEV cases
    of the other four folds"; the training cases of the never-cited measure of 3.4, so band 0 is exactly the summary's
    unseen pairs), the evaluation cases with their corrected labels. Without it, only the evaluation cases in
    ``labels`` (the 23 Oct runs used this narrower count: band 0 held 819 pairs instead of 471)."""
    rows = dev_citations(cat.data, cat) if cat is not None and getattr(cat, "data", None) else None
    if rows is not None:
        own = {str(q): p for q, p in zip(labels.query_id, labels.positives)}
        total = np.zeros(n_articles, dtype=np.int64)
        by_fold: dict[int, np.ndarray] = {}
        for f, q, idx in rows:
            idx = own.get(q, idx)
            by_fold.setdefault(f, np.zeros(n_articles, dtype=np.int64))
            by_fold[f][idx] += 1
            total[idx] += 1
        for k in set(int(x) for x in folds) | set(labels.fold.tolist()):
            by_fold.setdefault(k, np.zeros(n_articles, dtype=np.int64))
        return {k: total - v for k, v in by_fold.items()}
    case, art = labels.pair_arrays()
    case_fold = labels.fold[case]
    total = np.zeros(n_articles, dtype=np.int64)
    by_fold = {}
    for k in sorted(set(int(x) for x in folds) | set(labels.fold.tolist())):
        # one count per (case, article): positives are unique within a case
        by_fold[k] = np.bincount(art[case_fold == k], minlength=n_articles)
        total += by_fold[k]
    return {k: total - v for k, v in by_fold.items()}


def band(count: np.ndarray) -> np.ndarray:
    c = np.asarray(count)
    return np.select([c == 0, c <= 2, c <= 10], ["0", "1-2", "3-10"], ">10")


# ----------------------------------------------------------------------------- cluster bootstrap

class Boot:
    """Cluster bootstrap over groups: ``draws`` x G multiplicity matrix from ``rng.integers`` (seed fixed)."""

    def __init__(self, components: Sequence[str], draws: int = DRAWS, seed: int = SEED):
        self.ids, self.inv = np.unique(np.asarray(components, dtype=str), return_inverse=True)
        self.G = len(self.ids)
        self.draws, self.seed = int(draws), int(seed)
        rng = np.random.default_rng(seed)
        pick = rng.integers(0, self.G, size=(self.draws, self.G))
        self.W = np.stack([np.bincount(row, minlength=self.G) for row in pick]).astype(np.float64)

    def case_weights(self, case_group: np.ndarray | None = None) -> np.ndarray:
        """draws x n multiplicity of each case (or of each unit with group index ``case_group``)."""
        return self.W[:, self.inv if case_group is None else case_group]

    def group_means(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = np.asarray(values, dtype=np.float64)
        ok = np.isfinite(v)
        s = np.bincount(self.inv[ok], weights=v[ok], minlength=self.G)
        c = np.bincount(self.inv[ok], minlength=self.G)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / np.maximum(c, 1), np.nan), c > 0

    def macro(self, values: np.ndarray) -> dict[str, float]:
        """Component-macro mean of per-case values (NaN dropped) and its 95 % interval."""
        gm, ok = self.group_means(values)
        if not ok.any():
            return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"), "groups": 0}
        W = self.W[:, ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (W @ gm[ok]) / W.sum(axis=1)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        return {"mean": float(gm[ok].mean()), "lo": float(lo), "hi": float(hi), "groups": int(ok.sum())}

    def macro_boot(self, values: np.ndarray) -> tuple[float, np.ndarray]:
        gm, ok = self.group_means(values)
        W = self.W[:, ok]
        with np.errstate(invalid="ignore", divide="ignore"):
            return float(gm[ok].mean()) if ok.any() else float("nan"), (W @ gm[ok]) / W.sum(axis=1)

    def ratio(self, num: np.ndarray, den: np.ndarray, group: np.ndarray | None = None) -> dict[str, float]:
        """Pooled ratio sum(num) / sum(den) over units (cases, pairs or slots) with group index ``group``."""
        g = self.inv if group is None else np.asarray(group)
        n = np.bincount(g, weights=np.asarray(num, dtype=np.float64), minlength=self.G)
        d = np.bincount(g, weights=np.asarray(den, dtype=np.float64), minlength=self.G)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = (self.W @ n) / (self.W @ d)
            point = n.sum() / d.sum() if d.sum() > 0 else float("nan")
        lo, hi = (np.nanpercentile(boot, [2.5, 97.5]) if np.isfinite(boot).any() else (np.nan, np.nan))
        return {"mean": float(point), "lo": float(lo), "hi": float(hi), "n": float(d.sum())}

    def group_index(self, components: Sequence[str]) -> np.ndarray:
        pos = {c: i for i, c in enumerate(self.ids)}
        return np.asarray([pos[str(c)] for c in components], dtype=np.int64)


def interval(boot: np.ndarray) -> tuple[float, float]:
    b = np.asarray(boot, dtype=np.float64)
    b = b[np.isfinite(b)]
    if not b.size:
        return float("nan"), float("nan")
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- system metadata

def base_name(system: str) -> str:
    return system.split("@", 1)[0]


@lru_cache(maxsize=1)
def registry_systems() -> dict[str, dict[str, Any]]:
    try:
        from uol.final import registry
        return registry.systems()
    except Exception:                                   # pragma: no cover - registry absent in a stripped checkout
        return {}


def family(system: str) -> str:
    return str(registry_systems().get(base_name(system), {}).get("family") or "unregistered")


def answer_systems(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """P1-P6 (ANALYSIS_PLAN_C §7) from ``systems.json`` (field ``answer``), plus explicit overrides (P2 is chosen
    later and has no frozen row yet)."""
    out = {}
    for sid, spec in registry_systems().items():
        a = spec.get("answer")
        if a in ("P1", "P2", "P3", "P4", "P5", "P6"):
            out[a] = sid
    out.update(overrides or {})
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------- writers

def _default(x: Any) -> Any:
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if isinstance(x, Path):
        return x.as_posix()
    return str(x)


def _clean(obj: Any) -> Any:
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def write_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(obj), indent=1, ensure_ascii=False, default=_default), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame | list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
    f.to_csv(path, index=False, encoding="utf-8", float_format="%.6g")


def fmt(x: float | None, nd: int = 3) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    return f"{x:.{nd}f}"


def fmt_ci(d: dict[str, float], nd: int = 3) -> str:
    return f"{fmt(d.get('mean'), nd)} [{fmt(d.get('lo'), nd)}, {fmt(d.get('hi'), nd)}]"


class Evidence:
    """Rows in the EVIDENCE_FINAL format ``| Claim | Value | Scope | Source |``."""

    def __init__(self, item: str):
        self.item = item
        self.rows: list[tuple[str, str, str, str]] = []

    def add(self, subject: str, measures: Sequence[str], values: Sequence[str], source: str,
            scope: str = SCOPE) -> None:
        claim = f"{self.item} {subject}: " + "; ".join(measures)
        self.rows.append((claim, "; ".join(values), scope, source))

    def write(self, path: Path, notes: Sequence[str] = ()) -> None:
        lines = [f"# {self.item} evidence rows (ANALYSIS_PLAN_C)", "",
                 "Values are component-macro or pooled means with 95 % cluster-bootstrap intervals over groups "
                 f"({DRAWS} draws, seed {SEED}) unless the claim says otherwise.", ""]
        lines += [f"- {n}" for n in notes]
        if notes:
            lines.append("")
        lines += ["| Claim | Value | Scope | Source |", "|---|---|---|---|"]
        lines += [f"| {c} | {v} | {s} | {src} |" for c, v, s, src in self.rows]
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def present(store: RunStore, wanted: Sequence[str]) -> tuple[list[str], list[str]]:
    """(systems present, systems named by the plan but absent from this store)."""
    have = [s for s in wanted if store.has(s)]
    return have, [s for s in wanted if not store.has(s)]


_WORD = re.compile(r"\s+")
