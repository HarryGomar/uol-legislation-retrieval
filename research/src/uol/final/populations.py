"""Populations of PLAN_DRAFT §2, saved once as row-ID lists (query ids + v11 release rows + components + folds).

| ID      | evaluator population | rows                                                                 |
|---------|----------------------|----------------------------------------------------------------------|
| DEV     | P_spec               | development P_spec rows (1,218 cases / 1,093 groups) - primary       |
| DEV-G   | P                    | development P rows, generic articles kept (1,729)                    |
| DEV-ALL | all                  | every development row (4,812), full references                      |
| DEV-T   | P_spec               | time-ordered: DEV groups whose earliest publication date is after the |
|         |                      | 70% date of DEV; trained on development groups up to that date       |
| DEV-B   | P_spec + rules A+B   | stub: needs the binding-rule labels (data_soundness bindfix output)  |
| NEW     | P_spec               | stub: the prospective cohort (collected 31 Oct, scored once, G4)     |
| TEST    | -                    | refused (closed test set)                                            |

Outer folds are the five component-grouped development folds of SPLIT.json (``ep.scoring_groups`` without the
lockbox); the training rows of fold k are the development rows of the other four folds (the frozen rule). Inner
grouped 3-folds for tuned choices split the training components by a salted hash (``INNER_SALT``).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import guard

INNER_SALT = "uol-final-inner-20261021"
INNER_K = 3
TIME_SHARE = 0.70
BINDFIX = Path(__file__).resolve().parents[3] / "research/reports/data_soundness_20261020/scripts"
BINDFIX_LABELS = ("bindfix_labels.jsonl", "bindfix_output.jsonl", "bindfix_labels.json")

SPECS: dict[str, dict[str, Any]] = {
    "DEV": {"eval": "P_spec", "role": "primary", "expected": (1218, 1093)},
    "DEV-G": {"eval": "P", "role": "secondary (generic kept)", "expected": (1729, None)},
    "DEV-ALL": {"eval": "all", "role": "training population", "expected": (4812, None)},
    "DEV-T": {"eval": "P_spec", "role": "robustness: time-ordered 70/30 within dev", "expected": None},
    "DEV-B": {"eval": "P_spec", "role": "sensitivity: binding rules A+B (stub)", "expected": (1422, 1264)},
    "DEV-N": {"eval": "S2_spec", "role": "stratum: facts that name a provision, clause dropped (devn variant)",
              "expected": (854, None)},
    "NEW": {"eval": "P_spec", "role": "confirmation: prospective cohort (post-freeze plan)", "expected": None},
    # D96 (23 Oct): the agent block on a fixed sample, 60 DEV cases per outer fold, seed 20261023 (ANALYSIS_PLAN_C §11)
    "DEV-A300": {"eval": "P_spec", "role": "E12 agent sample (D96)", "expected": (300, None)},
    "DEV-L": {"eval": "P_spec", "role": "lay wording of the DEV facts (devl variant, C6, D93)", "expected": None},
    # D109 (24 Oct): the DEV facts cut to their first 25 / 50 / 100 words (devs variants, ANALYSIS_PLAN_C §6b)
    **{f"DEV-S{n}": {"eval": "P_spec", "role": f"facts cut to their first {n} words (devs{n} variant, D109)",
                     "expected": None} for n in (25, 50, 100)},
}


def labelled(rows) -> np.ndarray:
    """Rows that still have at least one positive (a label correction can empty a case: 2028377)."""
    from uol.embedding_push import common as ep
    d = ep.release()
    return np.asarray([int(r) for r in rows if d.positive_idx[int(r)]], dtype=np.int64)


SAVED = ("DEV", "DEV-G", "DEV-ALL")          # variant populations are always rebuilt in their own variant


class PopulationUnavailable(RuntimeError):
    """The population's inputs do not exist yet (DEV-B labels, the NEW cohort)."""


@dataclass
class FoldSpec:
    k: int                          # outer fold (0-4), or 0 for a single time split
    role: str                       # dev_fold<k> or "time"
    score_rows: np.ndarray          # admitted population rows scored with this fold's model
    train_rows: np.ndarray          # development rows whose labels this fold may use
    tune_rows: np.ndarray           # training rows admitted by the population (tuning objective)
    inner: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)   # (inner train, inner eval) rows


@dataclass
class Population:
    name: str
    eval_population: str
    query_ids: list[str]
    rows: np.ndarray
    meta: dict[str, Any]
    train_query_ids: list[str] | None = None          # DEV-T only

    @property
    def specific(self) -> bool:
        return self.eval_population.endswith("spec")

    def folds(self) -> list[FoldSpec]:
        from uol.embedding_push import common as ep
        d = ep.release()
        roles = ep.roles()
        out = []
        if self.train_query_ids is not None:                     # DEV-T: one time-ordered split
            train = guard.dev_rows_only([d.query_index[q] for q in self.train_query_ids], "DEV-T training rows")
            tune = labelled(ep.population_rows(self.eval_population, train))
            out.append(FoldSpec(0, "time", labelled(self.rows), np.sort(train), tune, inner_folds(tune, train)))
            return out
        for k in range(5):
            role = f"dev_fold{k}"
            score = labelled(self.rows[roles[self.rows] == role])
            if not len(score):
                continue
            train = guard.dev_rows_only(ep.training_rows(role), "training rows")
            tune = labelled(ep.population_rows(self.eval_population, train))
            out.append(FoldSpec(k, role, score, train, tune, inner_folds(tune, train)))
        return out

    def to_json(self) -> dict[str, Any]:
        from uol.embedding_push import common as ep
        d = ep.release()
        return {"name": self.name, "eval_population": self.eval_population, "n_cases": len(self.rows),
                "n_groups": len({d.queries[int(r)].component_id for r in self.rows}),
                "query_ids": self.query_ids, "release_rows": [int(r) for r in self.rows],
                "components": [d.queries[int(r)].component_id for r in self.rows],
                "folds": [str(ep.roles()[int(r)]) for r in self.rows],
                "train_query_ids": self.train_query_ids, "meta": self.meta}


def _inner_of(component: str) -> int:
    return int(hashlib.sha256(f"{INNER_SALT}:{component}".encode()).hexdigest(), 16) % INNER_K


def inner_folds(tune_rows: Sequence[int], train_rows: Sequence[int]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped inner 3-fold over the training components: (inner training rows, inner evaluation rows)."""
    from uol.embedding_push import common as ep
    d = ep.release()
    tune_rows, train_rows = np.asarray(tune_rows, dtype=np.int64), np.asarray(train_rows, dtype=np.int64)
    f_tune = np.asarray([_inner_of(d.queries[int(r)].component_id) for r in tune_rows], dtype=np.int64)
    f_train = np.asarray([_inner_of(d.queries[int(r)].component_id) for r in train_rows], dtype=np.int64)
    return [(train_rows[f_train != j], tune_rows[f_tune == j]) for j in range(INNER_K) if (f_tune == j).any()]


# --------------------------------------------------------------------------- builders


def _make(name: str, rows: np.ndarray, meta: dict[str, Any] | None = None, train_rows: np.ndarray | None = None) -> Population:
    from uol.embedding_push import common as ep
    d = ep.release()
    rows = guard.dev_rows_only(rows, name)
    spec = SPECS[name]
    tq = None
    if train_rows is not None:
        tq = [d.queries[int(r)].query_id for r in guard.dev_rows_only(train_rows, f"{name} training")]
    return Population(name, spec["eval"], [d.queries[int(r)].query_id for r in rows], rows,
                      {"role": spec["role"], **(meta or {})}, tq)


def publication_dates() -> dict[int, str | None]:
    from uol.round3 import common as R3
    return R3.publication_dates()


def build(name: str) -> Population:
    """Build a population from the active release (development rows only)."""
    guard.refuse_population(name)
    from uol.embedding_push import common as ep
    if name not in SPECS:
        raise KeyError(f"unknown population {name!r}; known: {sorted(SPECS)}")
    dev = guard.dev_rows_only(ep.dev_rows(), "dev rows")
    if name in ("DEV", "DEV-G", "DEV-ALL"):
        return _make(name, ep.population_rows(SPECS[name]["eval"], dev))
    if name == "DEV-A300":
        return _make(name, agent_sample(dev), {"seed": A300_SEED, "per_fold": 60})
    prof = __import__("uol.final.context", fromlist=["x"]).active()
    roles = ep.roles()
    if name == "DEV-T":
        if prof.variant == "devt":                       # own refit: fold 4 = the late groups (uol.final.variants)
            spec_rows = ep.population_rows("P_spec", dev)
            return _make(name, spec_rows[roles[spec_rows] == "dev_fold4"], {"variant": "devt"})
        if prof.kind == "bundle":
            raise PopulationUnavailable("DEV-T is scored in the devt variant (UOL_FINAL_VARIANT=devt), with its own refit")
        return build_time_ordered(dev)
    if name == "DEV-B":
        if prof.variant == "devb":                       # patched labels and recovered rows (uol.final.variants)
            return _make(name, ep.population_rows("P_spec", dev), {"variant": "devb"})
        return build_binding()
    if name == "DEV-N":
        if prof.kind == "bundle" and prof.variant != "devn":
            raise PopulationUnavailable("DEV-N is scored in the devn variant (UOL_FINAL_VARIANT=devn)")
        return _make(name, ep.population_rows("S2_spec", dev), {"variant": prof.variant or None})
    if name == "DEV-L":
        if prof.variant != "devl":
            raise PopulationUnavailable("DEV-L is scored in the devl variant (UOL_FINAL_VARIANT=devl)")
        from uol.final import devl
        lay = set(devl.rows())
        return _make(name, np.asarray([r for r in ep.population_rows("P_spec", dev) if int(r) in lay], dtype=np.int64),
                     {"variant": "devl"})
    if name in ("DEV-S25", "DEV-S50", "DEV-S100"):
        v = "devs" + name[len("DEV-S"):]
        if prof.variant != v:
            raise PopulationUnavailable(f"{name} is scored in the {v} variant (UOL_FINAL_VARIANT={v})")
        return _make(name, ep.population_rows("P_spec", dev), {"variant": v, "words": int(name[len("DEV-S"):])})
    raise PopulationUnavailable("NEW: the prospective cohort is collected on 31 Oct (uol.round2.c_collect) and is "
                                "scored once with frozen code (gate G4); it is not part of this build")


A300_SEED = 20261023


def agent_sample(dev: np.ndarray, per_fold: int = 60) -> np.ndarray:
    """DEV-A300: ``per_fold`` P_spec rows drawn from each outer fold with a fixed seed (D96)."""
    from uol.embedding_push import common as ep
    spec = np.asarray(ep.population_rows("P_spec", dev))
    roles = ep.roles()
    rng = np.random.default_rng(A300_SEED)
    out = []
    for k in range(5):
        fold = np.sort(spec[roles[spec] == f"dev_fold{k}"])
        out += rng.choice(fold, min(per_fold, len(fold)), replace=False).tolist()
    return np.asarray(sorted(out), dtype=np.int64)


def build_time_ordered(dev: np.ndarray, share: float = TIME_SHARE) -> Population:
    """DEV-T: groups (components) ordered by their earliest publication date among development rows.

    The cut is the publication date at the ``share`` quantile of the DEV (P_spec) cases; DEV groups whose earliest
    date is after the cut are scored, and every development row of a group whose earliest date is on or before the
    cut is training data. Undated groups are left out of both sides (counted in meta). Groups are never split."""
    from uol.embedding_push import common as ep
    d = ep.release()
    dates = publication_dates()
    first: dict[str, str] = {}
    undated: set[str] = set()
    for r in dev:
        c = d.queries[int(r)].component_id
        dt = dates.get(int(r))
        if not dt:
            undated.add(c)
            continue
        first[c] = min(dt, first.get(c, dt))
    spec_rows = ep.population_rows("P_spec", dev)
    dated = sorted((first[d.queries[int(r)].component_id], int(r)) for r in spec_rows
                   if d.queries[int(r)].component_id in first and d.queries[int(r)].component_id not in undated)
    if not dated:
        raise PopulationUnavailable("DEV-T: no publication dates available")
    cut = dated[int(np.floor(share * (len(dated) - 1)))][0]
    late = {c for c, dt in first.items() if dt > cut and c not in undated}
    score = np.asarray([r for r in spec_rows if d.queries[int(r)].component_id in late], dtype=np.int64)
    train = np.asarray([r for r in dev if (c := d.queries[int(r)].component_id) in first and c not in late
                        and c not in undated], dtype=np.int64)
    meta = {"cut_date": cut, "share": share, "undated_groups": len(undated),
            "rule": "group earliest publication date; score DEV groups after the cut, train on dev groups up to it"}
    return _make("DEV-T", score, meta, train)


def bindfix_labels_path() -> Path | None:
    for name in BINDFIX_LABELS:
        p = BINDFIX / name
        if p.exists():
            return p
    return None


def build_binding() -> Population:
    """DEV-B stub: DEV plus the cases the binding rules A+B admit, with the rule-derived labels.

    Expects a JSONL written by the data-soundness bindfix script ({"query_id", "candidate_ids"} per admitted case);
    the script currently prints its yields only, so this raises until that output exists. No refit (PLAN §2)."""
    p = bindfix_labels_path()
    if p is None:
        raise PopulationUnavailable(f"DEV-B: no binding-rule labels found in {BINDFIX} ({', '.join(BINDFIX_LABELS)}); "
                                    "write them from scripts/bindfix.py first")
    from uol.embedding_push import common as ep
    d = ep.release()
    recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if p.suffix == ".jsonl" else json.loads(p.read_text(encoding="utf-8"))
    qids = guard.dev_query_ids_only([r["query_id"] for r in recs], "DEV-B")
    base = set(ep.population_rows("P_spec", ep.dev_rows()).tolist())
    extra = sorted({d.query_index[q] for q in qids} - base)
    pop = _make("DEV-B", np.asarray(sorted(base | set(extra)), dtype=np.int64),
                {"labels_file": str(p), "added_cases": len(extra)})
    pop.meta["rule_labels"] = {r["query_id"]: r["candidate_ids"] for r in recs}
    return pop


# --------------------------------------------------------------------------- store


def population_dir(store: Path) -> Path:
    return Path(store) / "populations"


def save(pop: Population, store: Path) -> Path:
    p = population_dir(store) / f"{pop.name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(pop.to_json(), indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)
    return p


def save_all(store: Path, names: Sequence[str] = tuple(SPECS)) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for n in names:
        try:
            pop = build(n)
        except PopulationUnavailable as exc:
            out[n] = {"available": False, "reason": str(exc)}
            continue
        path = save(pop, store)
        exp = SPECS[n].get("expected")
        out[n] = {"available": True, "path": str(path), "n_cases": len(pop.rows),
                  "n_groups": pop.to_json()["n_groups"], "expected": exp, **{k: v for k, v in pop.meta.items()
                                                                               if k != "rule_labels"}}
    return out


def load(name: str, store: Path | None = None) -> Population:
    """Load a saved population by query id (so it transfers to the dev-only bundle release); build it if not saved."""
    guard.refuse_population(name)
    from uol.embedding_push import common as ep
    p = population_dir(store) / f"{name}.json" if store and name in SAVED else None
    if p is None or not p.exists():
        return build(name)
    rec = json.loads(p.read_text(encoding="utf-8"))
    d = ep.release()
    qids = guard.dev_query_ids_only(rec["query_ids"], name)
    rows = np.asarray([d.query_index[q] for q in qids], dtype=np.int64)
    tq = rec.get("train_query_ids")
    if tq is not None:
        guard.dev_query_ids_only(tq, f"{name} training")
    return Population(name, rec["eval_population"], qids, guard.dev_rows_only(rows, name), rec.get("meta", {}), tq)
