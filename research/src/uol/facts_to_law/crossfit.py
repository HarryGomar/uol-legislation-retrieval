"""Nested cross-fitting driver for the facts-to-law v2 retrieval systems.

For each outer fold f, each system's configuration is chosen by inner
cross-validation over the other folds (PREREGISTRATION section 2). Every
inner-validation fold g is scored with label-using parts (prior, kNN memory)
fitted on the remaining folds (all but f and g); the configuration with the
highest pooled component-macro objective over the selection population of the
outer-training folds wins (ties: lowest grid index). Label-using parts are then
refitted on all folds but f and fold f is scored once. A ``system`` channel
resolves to the referenced system's configuration selected for the same outer
fold, so lexical* and its fusions never see fold f.

Systems read only query texts, candidate texts and dense provider outputs.
References are read for evaluation and, from fitting folds only, for priors and
kNN memories. Score matrices exist for at most ``chunk_rows`` rows of one fold
at a time, in a byte-bounded cache of read-only arrays.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import spec as S
from .knn import WEIGHTINGS, Memory, build_memory, knn_votes, query_similarity_dense
from .lexical import CANDIDATE_TEXT_FIELD, KINDS as LEXICAL_KINDS, LexicalIndex
from .prior import component_counts, fitting_prior

METRICS = ("ndcg_at_10", "mrr", "recall_at_20", "recall_at_100")
CHANNEL_KINDS = ("lexical", "dense", "prior", "knn", "system")
FUSIONS = ("none", "linear", "rrf", "select")
TIE_TOLERANCE = 1e-12
PRIMARY_GENERIC_RULE = "breadth_v1"
CODE_FILES = ("spec.py", "lexical.py", "prior.py", "knn.py", "crossfit.py", "run.py", "metrics.py", "fusion.py", "audit.py")
REUSED_FILES = ("retrieval_benchmarks/c01_statutes/experiments.py", "retrieval_benchmarks/refinement_suite/nonparametric.py")


# ---------------------------------------------------------------------------
# Declarative systems
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChannelSpec:
    """One score source; ``params`` are fixed and ``grid`` axes are tuned by inner CV.

    kind ``lexical``: ``lexical`` is bm25 | bm25_stem | char; params/grid k1, b.
    kind ``dense``: ``provider`` is the mode passed to ``dense_scores`` (e.g. 'e5_small:maxp').
    kind ``prior``: the fitting prior alone; params ``scale`` (None | 'max').
    kind ``knn``: ``sim`` lexical (``lexical`` kind, params k1/b) or dense
    (``provider`` names the query-vector encoder); params/grid k, weighting, temperature.
    kind ``system``: ``system`` is a referenced system id.
    """

    kind: str
    name: str = ""
    lexical: str = "bm25"
    provider: str | None = None
    sim: str | None = None
    system: str | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSpec:
    """Channels, a fusion rule, an optional additive fitting prior and their grids.

    fusion ``none`` (one channel), ``linear`` (``fusion_params`` normalize;
    ``weights`` per channel in fusion_grid or fusion_params), ``rrf`` (k, optional
    weights) or ``select`` (one ``system`` channel chosen by inner CV).
    ``prior_weights`` tunes ``add_prior(scores, prior, w, prior_normalize)``.
    """

    id: str
    family: str
    channels: tuple[ChannelSpec, ...]
    fusion: str = "none"
    fusion_params: Mapping[str, Any] = field(default_factory=dict)
    fusion_grid: Mapping[str, Sequence[Any]] = field(default_factory=dict)
    prior_weights: tuple[float, ...] = ()
    prior_scale: str | None = None
    prior_normalize: str = "robust_z"
    description: str = ""

    def dependencies(self) -> list[str]:
        return [c.system for c in self.channels if c.kind == "system"]


def channel_from_dict(row: Mapping[str, Any]) -> ChannelSpec:
    extra = set(row) - set(ChannelSpec.__dataclass_fields__)
    if extra:
        raise ValueError(f"unknown channel fields: {sorted(extra)}")
    return ChannelSpec(**{**row, "params": dict(row.get("params") or {}),
                          "grid": {k: list(v) for k, v in (row.get("grid") or {}).items()}})


def system_from_dict(row: Mapping[str, Any]) -> SystemSpec:
    extra = set(row) - set(SystemSpec.__dataclass_fields__)
    if extra:
        raise ValueError(f"unknown system fields: {sorted(extra)}")
    return SystemSpec(id=str(row["id"]), family=str(row["family"]), channels=tuple(channel_from_dict(c) for c in row["channels"]),
                      fusion=row.get("fusion", "none"), fusion_params=dict(row.get("fusion_params") or {}),
                      fusion_grid={k: list(v) for k, v in (row.get("fusion_grid") or {}).items()},
                      prior_weights=tuple(float(w) for w in row.get("prior_weights") or ()), prior_scale=row.get("prior_scale"),
                      prior_normalize=row.get("prior_normalize", "robust_z"), description=str(row.get("description", "")))


def system_to_dict(system: SystemSpec) -> dict[str, Any]:
    row = asdict(system)
    row["channels"] = [asdict(c) for c in system.channels]
    row["prior_weights"] = list(system.prior_weights)
    return json.loads(json.dumps(row))


def channel_names(system: SystemSpec) -> list[str]:
    """Explicit names, else the referenced system id or the kind (indexed when repeated)."""
    defaults = [c.name or (c.system if c.kind == "system" else c.kind) for c in system.channels]
    names = [n if c.name or defaults.count(n) == 1 else f"{n}{i}" for i, (n, c) in enumerate(zip(defaults, system.channels))]
    if len(set(names)) != len(names):
        raise ValueError(f"{system.id}: channel names must be unique")
    return names


def validate_systems(systems: Sequence[SystemSpec]) -> list[list[SystemSpec]]:
    """Check every system and return dependency levels in topological order."""
    by_id: dict[str, SystemSpec] = {}
    for system in systems:
        if system.id in by_id:
            raise ValueError(f"duplicate system id: {system.id}")
        if not system.id or any(ch in system.id for ch in "/\\:*?\"<>| "):
            raise ValueError(f"system id must be a plain file name: {system.id!r}")
        by_id[system.id] = system
    for system in systems:
        if system.fusion not in FUSIONS:
            raise ValueError(f"{system.id}: unknown fusion {system.fusion}")
        if not system.channels:
            raise ValueError(f"{system.id}: a system needs at least one channel")
        if system.fusion == "none" and len(system.channels) != 1:
            raise ValueError(f"{system.id}: fusion 'none' takes exactly one channel")
        if system.fusion == "select" and (any(c.kind != "system" for c in system.channels) or system.prior_weights):
            raise ValueError(f"{system.id}: 'select' chooses among system channels only, without a prior")
        if system.prior_scale not in (None, "max") or system.prior_normalize not in ("robust_z", "rank", "none"):
            raise ValueError(f"{system.id}: unknown prior scale or normalization")
        if any(w < 0 for w in system.prior_weights):
            raise ValueError(f"{system.id}: prior weights must be non-negative")
        channel_names(system)
        for channel in system.channels:
            if channel.kind not in CHANNEL_KINDS:
                raise ValueError(f"{system.id}: unknown channel kind {channel.kind}")
            if set(channel.params) & set(channel.grid):
                raise ValueError(f"{system.id}: parameters both fixed and tuned: {sorted(set(channel.params) & set(channel.grid))}")
            if any(len(v) == 0 for v in channel.grid.values()):
                raise ValueError(f"{system.id}: empty grid axis")
            if channel.kind == "lexical" and channel.lexical not in LEXICAL_KINDS:
                raise ValueError(f"{system.id}: unknown lexical kind {channel.lexical}")
            if channel.kind == "dense" and not channel.provider:
                raise ValueError(f"{system.id}: a dense channel needs a provider")
            if channel.kind == "knn":
                if channel.sim not in ("lexical", "dense") or (channel.sim == "dense" and not channel.provider):
                    raise ValueError(f"{system.id}: kNN similarity must be lexical, or dense with a provider")
                if channel.sim == "lexical" and channel.lexical not in LEXICAL_KINDS:
                    raise ValueError(f"{system.id}: unknown lexical kind {channel.lexical}")
                weightings = list(channel.grid.get("weighting", [])) + ([channel.params["weighting"]] if "weighting" in channel.params else [])
                if any(w not in WEIGHTINGS for w in weightings):
                    raise ValueError(f"{system.id}: unknown kNN weighting")
            if channel.kind == "system" and channel.system not in by_id:
                raise ValueError(f"{system.id}: references unknown system {channel.system}")
        if system.fusion == "linear":
            weights = system.fusion_grid.get("weights") or ([system.fusion_params["weights"]] if "weights" in system.fusion_params else [])
            if not weights or any(len(w) != len(system.channels) for w in weights):
                raise ValueError(f"{system.id}: linear fusion needs one weight per channel")
    levels: list[list[SystemSpec]] = []
    placed: set[str] = set()
    remaining = list(systems)
    while remaining:
        level = [s for s in remaining if set(s.dependencies()) <= placed]
        if not level:
            raise ValueError(f"cyclic system references among {[s.id for s in remaining]}")
        levels.append(level)
        placed |= {s.id for s in level}
        remaining = [s for s in remaining if s.id not in placed]
    return levels


def enumerate_grid(system: SystemSpec) -> list[dict[str, Any]]:
    """Every configuration in deterministic order (declared axes, last axis fastest)."""
    axes: list[tuple[str, list[Any]]] = []
    for name, channel in zip(channel_names(system), system.channels):
        axes += [(f"{name}.{key}", list(values)) for key, values in channel.grid.items()]
    axes += [(f"fusion.{key}", list(values)) for key, values in system.fusion_grid.items()]
    if system.fusion == "select":
        axes.append(("select.channel", channel_names(system)))
    if system.prior_weights:
        axes.append(("prior.weight", list(system.prior_weights)))
    return [dict(zip([a for a, _ in axes], values)) for values in itertools.product(*[v for _, v in axes])] if axes else [{}]


def config_label(config: Mapping[str, Any]) -> str:
    return ",".join(f"{k}={json.dumps(v, separators=(',', ':'))}" for k, v in config.items()) or "default"


def resolve(system: SystemSpec, config: Mapping[str, Any], selected: Mapping[str, dict]) -> dict:
    """The concrete scoring tree of ``system`` under ``config``; references use ``selected`` nodes."""
    names = channel_names(system)
    if system.fusion == "select":
        choice = config["select.channel"]
        return {"node": "select", "id": system.id, "choice": choice, "inner": selected[system.channels[names.index(choice)].system]}
    channels = []
    for name, channel in zip(names, system.channels):
        params = {**channel.params, **{key: config[f"{name}.{key}"] for key in channel.grid}}
        if channel.kind == "system":
            channels.append(selected[channel.system])
        elif channel.kind == "lexical":
            channels.append({"node": "lexical", "lexical": channel.lexical, "params": params})
        elif channel.kind == "dense":
            channels.append({"node": "dense", "provider": channel.provider})
        elif channel.kind == "prior":
            channels.append({"node": "prior", "scale": params.get("scale")})
        else:
            channels.append({"node": "knn", "sim": channel.sim, "lexical": channel.lexical if channel.sim == "lexical" else None,
                             "provider": channel.provider if channel.sim == "dense" else None,
                             "params": {"k": 10, "weighting": "sim", "temperature": 0.05, **params}})
    fusion = {"type": system.fusion, **system.fusion_params, **{key: config[f"fusion.{key}"] for key in system.fusion_grid}}
    return {"node": "system", "id": system.id, "fusion": fusion, "channels": channels,
            "prior": {"declared": bool(system.prior_weights), "weight": float(config.get("prior.weight", 0.0)),
                      "scale": system.prior_scale, "normalize": system.prior_normalize}}


def node_key(node: Mapping[str, Any]) -> str:
    return json.dumps(node, sort_keys=True, separators=(",", ":"))


def unwrap(node: Mapping[str, Any]) -> Mapping[str, Any]:
    while node["node"] == "select":
        node = node["inner"]
    return node


def uses_labels(node: Mapping[str, Any]) -> bool:
    kind = node["node"]
    if kind == "select":
        return uses_labels(node["inner"])
    if kind in ("prior", "knn"):
        return True
    if kind == "system":
        return node["prior"]["weight"] > 0 or any(uses_labels(c) for c in node["channels"])
    return False


def system_providers(system: SystemSpec) -> set[tuple[str, str]]:
    """Dense inputs a system reads directly (references are separate systems)."""
    need = set()
    for channel in system.channels:
        if channel.kind == "dense":
            need.add(("dense_scores", channel.provider))
        if channel.kind == "knn" and channel.sim == "dense":
            need.add(("query_vectors", channel.provider))
    return need


# ---------------------------------------------------------------------------
# Dense providers and toolkit
# ---------------------------------------------------------------------------


@dataclass
class ScoreProviders:
    """Pluggable dense inputs.

    ``dense_scores(query_ids, mode)`` returns float32 ``[n, n_c]`` in catalogue
    order; ``query_vectors(query_ids, encoder)`` returns ``[n, d]``.
    ``receipts`` records input files and hashes for the run lock.
    """

    dense_scores: Callable[[Sequence[str], str], np.ndarray] | None = None
    query_vectors: Callable[[Sequence[str], str], np.ndarray] | None = None
    dense_modes: tuple[str, ...] = ()
    vector_modes: tuple[str, ...] = ()
    receipts: Mapping[str, Any] = field(default_factory=dict)

    def available(self) -> set[tuple[str, str]]:
        out = {("dense_scores", m) for m in self.dense_modes} if self.dense_scores else set()
        return out | ({("query_vectors", m) for m in self.vector_modes} if self.query_vectors else set())


AGGREGATIONS = ("maxp", "meanp", "firstp")


class DenseArrays:
    """Precomputed vectors of one encoder: queries and candidate units (chunks or articles).

    Scores are cosine similarities. A candidate's score is the max (``maxp``) or
    mean (``meanp``) over its units, or its chunk-0 unit (``firstp``). Every
    candidate needs at least one unit, and ``firstp`` a chunk 0.
    """

    def __init__(self, query_ids: Sequence[str], query_vectors: np.ndarray, unit_vectors: np.ndarray,
                 unit_candidate: Sequence[int], n_candidates: int, *, unit_chunk_index: Sequence[int] | None = None,
                 batch_size: int = 128) -> None:
        from uol.retrieval_benchmarks.refinement_suite.nonparametric import normalize

        query_vectors, unit_vectors = np.asarray(query_vectors), np.asarray(unit_vectors)
        candidate = np.asarray(unit_candidate, dtype=np.int64)
        chunk = np.zeros(len(candidate), dtype=np.int64) if unit_chunk_index is None else np.asarray(unit_chunk_index, dtype=np.int64)
        if len(query_ids) != len(query_vectors) or len(candidate) != len(unit_vectors) or len(chunk) != len(candidate):
            raise ValueError("vector and id counts differ")
        if query_vectors.shape[1] != unit_vectors.shape[1]:
            raise ValueError("query and unit vectors differ in width")
        order = np.lexsort((chunk, candidate))
        candidate, chunk = candidate[order], chunk[order]
        present = np.unique(candidate)
        if len(present) != n_candidates or present[0] != 0 or present[-1] != n_candidates - 1:
            raise ValueError("every candidate needs at least one dense unit")
        self.n_candidates = n_candidates
        self._queries = normalize(query_vectors)
        self._query_index = {q: i for i, q in enumerate(query_ids)}
        self._units = normalize(unit_vectors[order])
        self._starts = np.flatnonzero(np.r_[True, candidate[1:] != candidate[:-1]])
        self._counts = np.diff(np.r_[self._starts, len(candidate)]).astype(np.float32)
        self._first = self._starts if (chunk[self._starts] == 0).all() else None
        self.batch_size = batch_size

    def vectors(self, query_ids: Sequence[str]) -> np.ndarray:
        return self._queries[[self._query_index[q] for q in query_ids]]

    def scores(self, query_ids: Sequence[str], aggregation: str = "maxp") -> np.ndarray:
        if aggregation not in AGGREGATIONS:
            raise ValueError(f"unknown dense aggregation {aggregation}")
        if aggregation == "firstp" and self._first is None:
            raise ValueError("firstp needs a chunk-0 unit for every candidate")
        queries = self.vectors(query_ids)
        out = np.empty((len(queries), self.n_candidates), dtype=np.float32)
        for begin in range(0, len(queries), self.batch_size):
            if aggregation == "firstp":
                out[begin:begin + self.batch_size] = queries[begin:begin + self.batch_size] @ self._units[self._first].T
                continue
            sim = queries[begin:begin + self.batch_size] @ self._units.T
            if aggregation == "maxp":
                out[begin:begin + self.batch_size] = np.maximum.reduceat(sim, self._starts, axis=1)
            else:
                out[begin:begin + self.batch_size] = np.add.reduceat(sim, self._starts, axis=1) / self._counts
        return out


def providers_from_arrays(registry: Mapping[str, DenseArrays], receipts: Mapping[str, Any] | None = None) -> ScoreProviders:
    """Providers backed by encoder arrays: modes '<encoder>:<maxp|meanp|firstp>' and query vectors '<encoder>'."""
    def dense(ids: Sequence[str], mode: str) -> np.ndarray:
        encoder, _, aggregation = mode.partition(":")
        return registry[encoder].scores(ids, aggregation or "maxp")

    return ScoreProviders(dense_scores=dense, query_vectors=lambda ids, encoder: registry[encoder].vectors(ids),
                          dense_modes=tuple(f"{e}:{a}" for e in registry for a in AGGREGATIONS), vector_modes=tuple(registry),
                          receipts=dict(receipts or {}))


@dataclass
class Toolkit:
    """Metric and fusion functions; the default binds ``uol.facts_to_law.metrics`` and ``.fusion``."""

    evaluate_matrix: Callable[..., list[dict]]
    component_macro: Callable[..., tuple]
    linear_fusion: Callable[..., np.ndarray]
    rrf: Callable[..., np.ndarray]
    add_prior: Callable[..., np.ndarray]
    normalize_rows: Callable[..., np.ndarray] | None = None
    tie_order_from_ids: Callable[[Sequence[str]], np.ndarray] | None = None
    source: str = "injected"


def default_toolkit() -> Toolkit:
    from . import fusion, metrics

    return Toolkit(metrics.evaluate_matrix, metrics.component_macro, fusion.linear_fusion, fusion.rrf, fusion.add_prior,
                   fusion.normalize_rows, metrics.tie_order_from_ids, source="uol.facts_to_law.metrics + uol.facts_to_law.fusion")


# ---------------------------------------------------------------------------
# Release data and populations
# ---------------------------------------------------------------------------

PRIMARY_EXCLUDED_FLAGS = ("version_cue", "residual_statute_name", "quote_bearing")


@dataclass(frozen=True)
class Population:
    """Queries counted by an objective or a summary; ``view`` is full (reference) or specific (condensed)."""

    name: str
    strata: tuple[str, ...] | None
    resolution: tuple[str, ...] | None
    exclude_flags: tuple[str, ...] = ()
    view: str = "full"
    require_flag: str | None = None

    def admits(self, query: S.QueryRecord) -> bool:
        return ((self.strata is None or query.stratum in self.strata)
                and (self.resolution is None or query.resolution_status in self.resolution)
                and not any(query.flags.get(flag, False) for flag in self.exclude_flags)
                and (self.require_flag is None or bool(query.flags.get(self.require_flag, False))))


def _population(name: str, strata: tuple[str, ...] | None, resolution: tuple[str, ...] | None, view: str = "full",
                excluded: tuple[str, ...] = PRIMARY_EXCLUDED_FLAGS) -> Population:
    return Population(name, strata, resolution, excluded, view)


S1, S2, FULL_ONLY, ANY_RESOLUTION = (S.S1_NO_CITATION,), (S.S2_MASKED_CITATION,), (S.FULL,), (S.FULL, S.PARTIAL)
POPULATIONS = {
    # PREREGISTRATION section 1. 'S1_full' is the name used by the task brief for population P.
    "P": _population("P", S1, FULL_ONLY),
    "S1_full": _population("S1_full", S1, FULL_ONLY),
    "P_spec": _population("P_spec", S1, FULL_ONLY, "specific"),
    "S2": _population("S2", S2, FULL_ONLY),
    "S2_spec": _population("S2_spec", S2, FULL_ONLY, "specific"),
    "P_partial": _population("P_partial", S1, ANY_RESOLUTION),
    "P_partial_spec": _population("P_partial_spec", S1, ANY_RESOLUTION, "specific"),
    "all": Population("all", None, None, (), "full"),
}
SUMMARY_POPULATIONS = (
    POPULATIONS["P_spec"], POPULATIONS["P"], POPULATIONS["S2_spec"], POPULATIONS["S2"], POPULATIONS["P_partial"],
    _population("S1_partial_only", S1, (S.PARTIAL,)), _population("S2_partial_only", S2, (S.PARTIAL,)),
    Population("S1_any", S1, None, (), "full"), Population("S2_any", S2, None, (), "full"),
    Population("version_cue", None, None, (), "full", "version_cue"), POPULATIONS["all"],
)


@dataclass
class ReleaseData:
    """Release rows in release order, the bound catalogue and analysis-only typing."""

    release_dir: Path
    manifest: dict[str, Any]
    queries: list[S.QueryRecord]
    references: dict[str, list[str]]
    folds: dict[str, Any]
    candidate_ids: list[str]
    candidate_texts: list[str] | None
    generic_ids: frozenset[str]
    hashes: dict[str, str]
    candidate_binding: dict[str, Any]

    def __post_init__(self) -> None:
        self.candidate_index = {c: i for i, c in enumerate(self.candidate_ids)}
        self.query_index = {q.query_id: i for i, q in enumerate(self.queries)}
        self.component_by_query = {q.query_id: q.component_id for q in self.queries}
        self.generic_idx = np.asarray(sorted(self.candidate_index[c] for c in self.generic_ids if c in self.candidate_index), dtype=np.int64)
        self.positive_idx = [[self.candidate_index[c] for c in self.references[q.query_id] if c in self.candidate_index]
                             for q in self.queries]
        generic = set(self.generic_idx.tolist())
        self.specific_idx = [[p for p in row if p not in generic] for row in self.positive_idx]


def load_candidates(articles_path: Path, *, texts: bool = True) -> tuple[list[str], list[str] | None]:
    """Candidate ids (and ``instrument_article_body`` texts) in catalogue order."""
    ids: list[str] = []
    out: list[str] | None = [] if texts else None
    for row in S.read_jsonl(articles_path):
        ids.append(row["candidate_id"])
        if out is not None:
            out.append(row[CANDIDATE_TEXT_FIELD])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate candidate ids in the bound catalogue")
    return ids, out


def bound_articles(release_dir: Path, articles_path: Path | None = None) -> tuple[Path, bool]:
    """The articles.jsonl a release binds (``audit.bound_catalogue``) and whether its SHA-256 is declared."""
    from .audit import bound_catalogue

    catalogue = bound_catalogue(Path(release_dir), articles_path)
    return Path(catalogue.articles), bool(catalogue.articles_declared)


def _generic_ids(release_dir: Path, rule: str) -> tuple[frozenset[str], dict[str, Any]]:
    path = Path(release_dir) / "target_types.jsonl"
    if not path.exists():
        return frozenset(), {"path": None, "rule": rule, "generic": 0}
    rows = list(S.read_jsonl(path))
    rules = sorted({str(r.get("rule")) for r in rows})
    chosen = [r for r in rows if r.get("rule") == rule] if len(rules) > 1 else rows
    if len(rules) > 1 and not chosen:
        raise ValueError(f"target_types.jsonl has rules {rules} but none is {rule}")
    generic = frozenset(r["candidate_id"] for r in chosen if r["target_type"] == S.GENERIC)
    return generic, {"path": str(path), "rules_present": rules, "rule": rule if len(rules) > 1 else rules[0] if rules else None,
                     "generic": len(generic), "typed": len(chosen)}


def load_release(release_dir: Path, *, articles_path: Path | None = None, candidate_ids: Sequence[str] | None = None,
                 candidate_texts: Sequence[str] | None = None, generic_rule: str = PRIMARY_GENERIC_RULE,
                 verify: bool = True) -> ReleaseData:
    """Load and cross-check a release; candidates come from the bound articles file unless given."""
    release_dir = Path(release_dir)
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = {"manifest.json": S.sha256_file(manifest_path)}
    for name in ("queries.jsonl", "references.jsonl", "folds.json", "target_types.jsonl"):
        if (release_dir / name).exists():
            hashes[name] = S.sha256_file(release_dir / name)
    if verify:
        for name, digest in (manifest.get("files") or {}).items():
            if isinstance(digest, str) and name in hashes and hashes[name] != digest:
                raise ValueError(f"release file differs from its manifest hash: {name}")
    queries = S.load_queries(release_dir)
    references = {r.query_id: r.positive_ids for r in S.load_references(release_dir)}
    folds = json.loads((release_dir / "folds.json").read_text(encoding="utf-8"))
    generic, typing = _generic_ids(release_dir, generic_rule)
    if candidate_ids is None:
        path, declared = bound_articles(release_dir, articles_path)
        candidate_ids, candidate_texts = load_candidates(path)
        binding = {"articles_path": str(path), "sha256": S.sha256_file(path), "sha256_declared_in_manifest": declared,
                   "text_field": CANDIDATE_TEXT_FIELD, "candidates": len(candidate_ids)}
        if verify and articles_path is None and not declared:
            raise ValueError("the bound candidate catalogue's SHA-256 is not declared in the release manifest")
    else:
        binding = {"articles_path": None, "sha256": None, "text_field": CANDIDATE_TEXT_FIELD, "candidates": len(candidate_ids),
                   "ids_sha256": S.sha256_text("\n".join(candidate_ids))}
    binding["target_typing"] = typing
    data = ReleaseData(release_dir, manifest, queries, references, folds, list(candidate_ids),
                       list(candidate_texts) if candidate_texts is not None else None, generic, hashes, binding)
    _check_release(data, verify)
    return data


def _check_release(data: ReleaseData, verify: bool) -> None:
    ids = [q.query_id for q in data.queries]
    if len(set(ids)) != len(ids) or set(ids) != set(data.references):
        raise ValueError("queries and references disagree")
    k = int(data.folds["k"])
    for query, positives in zip(data.queries, data.positive_idx):
        if not 0 <= query.fold < k or data.folds["components"].get(query.component_id) != query.fold:
            raise ValueError(f"query fold disagrees with folds.json: {query.query_id}")
        if verify and S.sha256_text(query.text) != query.text_sha256:
            raise ValueError(f"query text digest differs: {query.query_id}")
        if not positives or len(positives) != len(data.references[query.query_id]):
            raise ValueError(f"query without positives or with references outside the catalogue: {query.query_id}")
    if data.candidate_texts is not None and len(data.candidate_texts) != len(data.candidate_ids):
        raise ValueError("candidate ids and texts differ in length")


# ---------------------------------------------------------------------------
# Scoring engine
# ---------------------------------------------------------------------------


class _MatrixCache:
    """LRU cache of read-only arrays bounded by the bytes they own."""

    def __init__(self, budget_bytes: int) -> None:
        self.budget = int(budget_bytes)
        self.items: OrderedDict[Any, np.ndarray] = OrderedDict()
        self.bytes = 0
        self.peak = 0

    @staticmethod
    def _owned(value: np.ndarray) -> int:
        return value.nbytes if value.base is None else 0

    def get(self, key: Any) -> np.ndarray | None:
        value = self.items.get(key)
        if value is not None:
            self.items.move_to_end(key)
        return value

    def put(self, key: Any, value: np.ndarray) -> np.ndarray:
        if value.flags.writeable:
            value.setflags(write=False)
        size = self._owned(value)
        while self.items and self.bytes + size > self.budget:
            _, old = self.items.popitem(last=False)
            self.bytes -= self._owned(old)
        self.items[key] = value
        self.bytes += size
        self.peak = max(self.peak, self.bytes)
        return value

    def clear(self) -> None:
        self.items.clear()
        self.bytes = 0


@dataclass(frozen=True)
class Block:
    """Up to ``chunk_rows`` rows (release positions) of one fold; subset 'sel' or 'all'."""

    fold: int
    subset: str
    part: int
    rows: tuple[int, ...]

    @property
    def key(self) -> str:
        return f"{self.subset}:{self.fold}:{self.part}"


def make_blocks(rows: Sequence[int], fold: int, subset: str, chunk_rows: int) -> list[Block]:
    return [Block(fold, subset, part, tuple(rows[begin:begin + chunk_rows]))
            for part, begin in enumerate(range(0, len(rows), chunk_rows))]


class Engine:
    """Score matrices of resolved nodes for blocks under a fitting context (a tuple of folds)."""

    def __init__(self, data: ReleaseData, providers: ScoreProviders, toolkit: Toolkit, *, fitting_population: Population,
                 cache_bytes: int, tie_order: np.ndarray | None = None, observer: Callable[[str, Any], None] | None = None) -> None:
        self.data = data
        self.providers = providers
        self.toolkit = toolkit
        self.fitting_population = fitting_population
        self.tie_order = tie_order
        self.cache = _MatrixCache(cache_bytes)
        self.small: OrderedDict[Any, Any] = OrderedDict()
        self.indexes: dict[str, LexicalIndex] = {}
        self.index_seconds: dict[str, float] = {}
        self.observer = observer or (lambda event, payload: None)
        self.max_rows = 0
        self.n_candidates = len(data.candidate_ids)

    # fitting contexts ---------------------------------------------------
    def fit_query_ids(self, fit: tuple[int, ...]) -> list[str]:
        folds = set(fit)
        return [q.query_id for q in self.data.queries if q.fold in folds and self.fitting_population.admits(q)]

    def _small(self, key: Any, build: Callable[[], Any]) -> Any:
        if key in self.small:
            self.small.move_to_end(key)
        else:
            self.small[key] = build()
            while len(self.small) > 48:
                self.small.popitem(last=False)
        return self.small[key]

    def prior(self, fit: tuple[int, ...], scale: str | None) -> np.ndarray:
        def build() -> np.ndarray:
            value = fitting_prior(self.data.references, self.data.component_by_query, self.fit_query_ids(fit),
                                  self.n_candidates, self.data.candidate_index, scale=scale)
            value.setflags(write=False)
            return value
        return self._small(("prior", fit, scale), build)

    def memory(self, fit: tuple[int, ...]) -> Memory:
        return self._small(("memory", fit), lambda: build_memory(self.fit_query_ids(fit), self.data.component_by_query,
                                                                 self.data.references, self.data.candidate_index))

    # channels -------------------------------------------------------------
    def lexical_index(self, kind: str) -> LexicalIndex:
        if kind not in self.indexes:
            if self.data.candidate_texts is None:
                raise ValueError("lexical systems need candidate texts")
            started = time.perf_counter()
            self.indexes[kind] = LexicalIndex(self.data.candidate_texts, kind=kind)
            self.index_seconds[kind] = time.perf_counter() - started
            self.observer("index", {"kind": kind, "seconds": self.index_seconds[kind]})
        return self.indexes[kind]

    def _texts(self, block: Block) -> list[str]:
        return [self.data.queries[i].text for i in block.rows]

    def _ids(self, block: Block) -> list[str]:
        return [self.data.queries[i].query_id for i in block.rows]

    def _check(self, matrix: np.ndarray, block: Block, what: str) -> np.ndarray:
        matrix = np.asarray(matrix)
        if matrix.shape != (len(block.rows), self.n_candidates) or not np.isfinite(matrix).all():
            raise ValueError(f"{what}: expected finite [{len(block.rows)}, {self.n_candidates}] scores, got {matrix.shape}")
        self.max_rows = max(self.max_rows, matrix.shape[0])
        self.observer("matrix", {"what": what, "shape": matrix.shape, "block": block.key})
        return matrix

    def similarity(self, node: Mapping[str, Any], block: Block, fit: tuple[int, ...]) -> np.ndarray:
        memory = self.memory(fit)
        if node["sim"] == "lexical":
            params = {k: node["params"][k] for k in ("k1", "b") if k in node["params"]}
            key = ("sim", "lexical", node["lexical"], json.dumps(params, sort_keys=True), block.key, fit)
            cached = self.cache.get(key)
            if cached is None:
                texts = [self.data.queries[self.data.query_index[q]].text for q in memory.query_ids]
                index = self._small(("memory_index", node["lexical"], fit), lambda: LexicalIndex(texts, kind=node["lexical"]))
                cached = self.cache.put(key, index.score(self._texts(block), params or None))
            return cached
        key = ("sim", "dense", node["provider"], block.key, fit)
        cached = self.cache.get(key)
        if cached is None:
            vectors = self.providers.query_vectors
            cached = self.cache.put(key, query_similarity_dense(vectors(self._ids(block), node["provider"]),
                                                                vectors(list(memory.query_ids), node["provider"])))
        return cached

    def node_matrix(self, node: Mapping[str, Any], block: Block, fit: tuple[int, ...], *, cache: bool = True) -> np.ndarray:
        node = unwrap(node)
        key = (node_key(node), block.key, fit if uses_labels(node) else None)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        value = self._compute(node, block, fit)
        return self.cache.put(key, value) if cache else value

    def normalized(self, node: Mapping[str, Any], method: str, block: Block, fit: tuple[int, ...]) -> np.ndarray:
        """``normalize_rows(node scores, method)``, cached; without ``normalize_rows`` the raw scores."""
        if self.toolkit.normalize_rows is None or method == "none":
            return self.node_matrix(node, block, fit)
        node = unwrap(node)
        key = ("normalized", method, node_key(node), block.key, fit if uses_labels(node) else None)
        cached = self.cache.get(key)
        if cached is None:
            cached = self.cache.put(key, self.toolkit.normalize_rows(self.node_matrix(node, block, fit), method))
        return cached

    def _prior_normalized(self, fit: tuple[int, ...], scale: str | None, method: str) -> np.ndarray:
        if self.toolkit.normalize_rows is None or method == "none":
            return self.prior(fit, scale)
        return self._small(("prior_normalized", fit, scale, method),
                           lambda: self.toolkit.normalize_rows(self.prior(fit, scale), method))

    def _compute(self, node: Mapping[str, Any], block: Block, fit: tuple[int, ...]) -> np.ndarray:
        kind, n = node["node"], len(block.rows)
        if kind == "lexical":
            scores = self.lexical_index(node["lexical"]).score(self._texts(block), node["params"] or None)
            return self._check(scores, block, f"lexical:{node['lexical']}")
        if kind == "dense":
            scores = self.providers.dense_scores(self._ids(block), node["provider"])
            return self._check(np.asarray(scores, dtype=np.float32), block, f"dense:{node['provider']}")
        if kind == "prior":
            return np.broadcast_to(self.prior(fit, node["scale"]), (n, self.n_candidates))
        if kind == "knn":
            memory, params = self.memory(fit), node["params"]
            votes = knn_votes(self.similarity(node, block, fit), memory.positives, int(params["k"]), self.n_candidates,
                              params["weighting"], temperature=float(params["temperature"]),
                              eval_components=[self.data.queries[i].component_id for i in block.rows],
                              fit_components=list(memory.components))
            return self._check(votes, block, f"knn:{node['sim']}")
        prior = node["prior"]
        if prior["declared"]:
            # add_prior(base, prior, w): normalize(base) + w * normalize(prior); w = 0 leaves normalize(base).
            base = {**node, "prior": {**prior, "weight": 0.0, "declared": False}}
            if prior["weight"] == 0:
                if self.toolkit.normalize_rows is not None:
                    return self.normalized(base, prior["normalize"], block, fit)
                scores = self.toolkit.add_prior(self.node_matrix(base, block, fit), np.zeros(self.n_candidates, np.float32), 0.0,
                                                normalize=prior["normalize"])
            elif self.toolkit.normalize_rows is not None:
                scores = self.toolkit.add_prior(self.normalized(base, prior["normalize"], block, fit),
                                                self._prior_normalized(fit, prior["scale"], prior["normalize"]), prior["weight"],
                                                normalize="none")
            else:
                scores = self.toolkit.add_prior(self.node_matrix(base, block, fit), self.prior(fit, prior["scale"]),
                                                prior["weight"], normalize=prior["normalize"])
            return self._check(np.asarray(scores, dtype=np.float32), block, f"system:{node['id']}+prior")
        fusion = dict(node["fusion"])
        fusion_type = fusion.pop("type")
        if fusion_type == "none":
            return self.node_matrix(node["channels"][0], block, fit)
        if fusion_type == "linear":
            weights = [float(w) for w in fusion.pop("weights")]
            method = fusion.pop("normalize", "robust_z")
            channels = [self.normalized(c, method, block, fit) if w else self.node_matrix(c, block, fit)
                        for c, w in zip(node["channels"], weights)]
            pre = self.toolkit.normalize_rows is not None
            scores = self.toolkit.linear_fusion(channels, weights, normalize="none" if pre else method, **fusion)
        elif fusion_type == "rrf":
            channels = [self.node_matrix(c, block, fit) for c in node["channels"]]
            if self.tie_order is not None:
                fusion["tie_order"] = self.tie_order
            scores = self.toolkit.rrf(channels, **fusion)
        else:
            raise ValueError(f"unknown fusion {fusion_type}")
        return self._check(np.asarray(scores, dtype=np.float32), block, f"system:{node['id']}")

    # evaluation -----------------------------------------------------------
    def evaluate(self, scores: np.ndarray, block: Block, view: str, *, top_k: int) -> tuple[list[int], list[dict]]:
        """Evaluate a block's rows in a view; returns kept local rows and ``evaluate_matrix`` results."""
        local = list(range(len(block.rows)))
        if view == "specific":
            local = [r for r in local if self.data.specific_idx[block.rows[r]]]
            positives = [self.data.specific_idx[block.rows[r]] for r in local]
            exclude = self.data.generic_idx if len(self.data.generic_idx) else None
        else:
            positives = [self.data.positive_idx[block.rows[r]] for r in local]
            exclude = None
        if not local:
            return [], []
        matrix = scores if len(local) == len(block.rows) else scores[local]
        extra = {"tie_order": self.tie_order} if self.tie_order is not None else {}
        return local, self.toolkit.evaluate_matrix(matrix, positives, exclude_idx=exclude, top_k=top_k, **extra)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_memory_bytes() -> int | None:
    """Peak working set (Windows) or resident set size of this process."""
    try:
        import psutil

        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", 0) or info.rss)
    except Exception:  # noqa: BLE001 - optional instrumentation
        return None


def code_hashes() -> dict[str, str | None]:
    here = Path(__file__).resolve().parent
    out: dict[str, str | None] = {}
    for name in CODE_FILES:
        out[f"uol/facts_to_law/{name}"] = S.sha256_file(here / name) if (here / name).exists() else None
    for name in REUSED_FILES:
        out[f"uol/{name}"] = S.sha256_file(here.parent / name) if (here.parent / name).exists() else None
    return out


def environment() -> dict[str, Any]:
    import scipy
    import sklearn

    env: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__,
                           "scipy": scipy.__version__, "sklearn": sklearn.__version__}
    try:
        import nltk

        env["nltk"] = nltk.__version__
    except ImportError:
        env["nltk"] = None
    return env


def _macro(toolkit: Toolkit, values: np.ndarray, components: Sequence[str], keys: Sequence[str] = METRICS) -> dict[str, float]:
    """Component-macro means of per-query metric rows ``values[:, len(keys)]``."""
    rows = [dict(zip(keys, map(float, row))) for row in np.asarray(values, dtype=np.float64).reshape(len(components), len(keys))]
    return {k: float(v) for k, v in toolkit.component_macro(rows, list(components), keys)[0].items()}


def summarize(data: ReleaseData, details: Mapping[str, Sequence[Mapping[str, Any]]], toolkit: Toolkit) -> dict[str, Any]:
    """Component-macro metrics per system and summary population."""
    by_id = {q.query_id: q for q in data.queries}
    out: dict[str, Any] = {}
    for system_id, rows in details.items():
        table = {}
        for population in SUMMARY_POPULATIONS:
            chosen = [r for r in rows if population.admits(by_id[r["query_id"]])]
            if population.view == "specific":
                chosen = [r for r in chosen if r.get("specific")]
                values = [[r["specific"][m] for m in METRICS] for r in chosen]
            else:
                values = [[r[m] for m in METRICS] for r in chosen]
            components = [r["component_id"] for r in chosen]
            table[population.name] = ({"queries": len(chosen), "components": len(set(components)),
                                       **_macro(toolkit, np.asarray(values), components)} if chosen else {"queries": 0, "components": 0})
        out[system_id] = table
    return out


def popularity_counts(data: ReleaseData, outer_fold: int, fitting_population: str = "all") -> np.ndarray:
    """Fitting-component counts for one outer fold, for popularity bands in analysis."""
    population = POPULATIONS[fitting_population]
    fit_ids = [q.query_id for q in data.queries if q.fold != outer_fold and population.admits(q)]
    return component_counts(data.references, data.component_by_query, fit_ids, len(data.candidate_ids), data.candidate_index)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_crossfit(release_dir: Path, systems: Sequence[SystemSpec], score_providers: ScoreProviders | None = None,
                 folds: int = 5, selection_population: str = "S1_full", objective: str = "ndcg_at_10",
                 out_dir: Path | None = None, *, toolkit: Toolkit | None = None, articles_path: Path | None = None,
                 candidate_ids: Sequence[str] | None = None, candidate_texts: Sequence[str] | None = None,
                 fitting_population: str = "all", tie_rule: str = "index", chunk_rows: int = 512,
                 cache_bytes: int = 1 << 30, top_k: int = 100, generic_rule: str = PRIMARY_GENERIC_RULE,
                 lock_extra: Mapping[str, Any] | None = None, observer: Callable[[str, Any], None] | None = None,
                 log: Callable[[str], None] | None = print) -> dict[str, Any]:
    """Nested cross-fitting of ``systems``; writes run_lock, selection, details and summary under ``out_dir``.

    ``tie_rule`` 'index' (pre-registered: candidate index ascending) or
    'candidate_id' (October). ``fitting_population`` selects the training-fold
    queries whose references feed priors and kNN memories.
    """
    started = time.perf_counter()
    if out_dir is None:
        raise ValueError("an output directory is required")
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite existing outputs: {out_dir}")
    if objective not in METRICS or tie_rule not in ("index", "candidate_id") or chunk_rows < 1:
        raise ValueError("unknown objective or tie rule, or a non-positive chunk size")
    population, fitting = POPULATIONS[selection_population], POPULATIONS[fitting_population]
    say = log or (lambda message: None)
    notify = observer or (lambda event, payload: None)
    providers = score_providers or ScoreProviders()
    levels = validate_systems(systems)
    ordered = [s for level in levels for s in level]
    missing = sorted({(s.id, need) for s in ordered for need in system_providers(s) if need not in providers.available()})
    if missing:
        raise ValueError(f"dense inputs unavailable: {missing}")
    toolkit = toolkit or default_toolkit()
    data = load_release(Path(release_dir), articles_path=articles_path, candidate_ids=candidate_ids,
                        candidate_texts=candidate_texts, generic_rule=generic_rule)
    if int(data.folds["k"]) != folds:
        raise ValueError(f"release has {data.folds['k']} folds, expected {folds}")
    if tie_rule == "candidate_id" and toolkit.tie_order_from_ids is None:
        raise ValueError("the candidate-ID tie rule needs tie_order_from_ids")
    tie_order = toolkit.tie_order_from_ids(data.candidate_ids) if tie_rule == "candidate_id" else None
    grids = {s.id: enumerate_grid(s) for s in ordered}

    out_dir.mkdir(parents=True, exist_ok=True)
    lock = {"format_version": S.FORMAT_VERSION, "program": "facts_to_law_v2_20261004", "status": "locked_before_scoring",
            "started_utc": _utc(), "release_dir": str(data.release_dir), "release_id": data.manifest.get("release_id"),
            "release_manifest_sha256": data.hashes["manifest.json"], "release_files": data.hashes,
            "candidates": data.candidate_binding,
            "folds": {"k": int(data.folds["k"]), "salt": data.folds.get("salt"), "components": len(data.folds["components"])},
            "queries": len(data.queries), "selection_population": asdict(population), "fitting_population": asdict(fitting),
            "objective": objective, "metrics": list(METRICS), "top_k": top_k,
            "tie_rule": {"ranking": "score descending, then " + ("candidate index" if tie_rule == "index" else "candidate ID") + " ascending",
                         "selection": f"highest pooled inner component-macro {objective}; values within {TIE_TOLERANCE} tie; lowest grid index wins"},
            "inner_cv": "each outer-training fold g is scored with labels fitted on the outer-training folds other than g; "
                        "the objective is pooled over the selection-population queries of all outer-training folds",
            "refit": "label-using parts refitted on all outer-training folds; the outer fold is scored once with the selected configuration",
            "knn_memory": "one entry per fitting component: lowest query id text or vector, union of the component's fitting positives",
            "prior": "log1p(number of distinct fitting components citing the candidate); scale and normalization per system",
            "seeds": {}, "randomness": "none: lexical, prior, kNN, fusion and selection are deterministic",
            "systems": [system_to_dict(s) for s in ordered], "levels": [[s.id for s in level] for level in levels],
            "grids": grids, "grid_sizes": {sid: len(g) for sid, g in grids.items()},
            "dense_providers": {"dense_modes": list(providers.dense_modes), "vector_modes": list(providers.vector_modes),
                                "receipts": providers.receipts},
            "toolkit": toolkit.source, "code_sha256": code_hashes(), "environment": environment(),
            "memory": {"chunk_rows": chunk_rows, "cache_bytes": cache_bytes}, **dict(lock_extra or {})}
    S.write_json(out_dir / "run_lock.json", lock)
    notify("lock", {"path": str(out_dir / "run_lock.json")})
    say(f"[{_utc()}] run_lock written: {len(ordered)} systems, {sum(len(g) for g in grids.values())} configurations, "
        f"{len(data.queries)} queries")

    engine = Engine(data, providers, toolkit, fitting_population=fitting, cache_bytes=cache_bytes, tie_order=tie_order, observer=notify)
    k = folds
    sel_rows = [i for i, q in enumerate(data.queries)
                if population.admits(q) and (population.view == "full" or data.specific_idx[i])]
    sel_position = {i: j for j, i in enumerate(sel_rows)}
    fold_of_row = [q.fold for q in data.queries]
    sel_blocks = {g: make_blocks([i for i in sel_rows if fold_of_row[i] == g], g, "sel", chunk_rows) for g in range(k)}
    all_blocks = {g: make_blocks([i for i in range(len(data.queries)) if fold_of_row[i] == g], g, "all", chunk_rows) for g in range(k)}
    selected: dict[str, dict[int, dict]] = {}
    selected_index: dict[str, dict[int, int]] = {}
    timings: dict[str, dict[str, float]] = {s.id: {"inner_seconds": 0.0, "outer_seconds": 0.0} for s in ordered}
    memo: dict[Any, np.ndarray] = {}
    objective_at = METRICS.index(objective)

    for level_number, level in enumerate(levels):
        tables = {s.id: np.full((k, len(sel_rows), len(grids[s.id]), len(METRICS)), np.nan) for s in level}
        level_started = time.perf_counter()
        for g in range(k):
            for block in sel_blocks[g]:
                positions = [sel_position[i] for i in block.rows]
                for system in level:
                    system_started = time.perf_counter()
                    for f in range(k):
                        if f == g:
                            continue
                        fit = tuple(sorted(set(range(k)) - {f, g}))
                        refs = {sid: selected[sid][f] for sid in system.dependencies()}
                        for ci, config in enumerate(grids[system.id]):
                            node = unwrap(resolve(system, config, refs))
                            key = (node_key(node), block.key, fit if uses_labels(node) else None)
                            if key not in memo:
                                scores = engine.node_matrix(node, block, fit, cache=False)
                                kept, rows = engine.evaluate(scores, block, population.view, top_k=0)
                                if len(kept) != len(block.rows):
                                    raise RuntimeError("a selection row lost its relevant items")
                                memo[key] = np.asarray([[float(r[m]) for m in METRICS] for r in rows], dtype=np.float64)
                            tables[system.id][f, positions, ci, :] = memo[key]
                    timings[system.id]["inner_seconds"] += time.perf_counter() - system_started
                engine.cache.clear()
            say(f"[{_utc()}] level {level_number} inner fold {g} done ({sum(len(b.rows) for b in sel_blocks[g])} selection rows)")
        for system in level:
            records, selected[system.id], selected_index[system.id] = [], {}, {}
            for f in range(k):
                rows = [i for i in sel_rows if fold_of_row[i] != f]
                if not rows:
                    raise ValueError(f"outer fold {f} has no training queries in {population.name}")
                positions = [sel_position[i] for i in rows]
                components = [data.queries[i].component_id for i in rows]
                values = tables[system.id][f, positions]
                if np.isnan(values).any():
                    raise RuntimeError(f"{system.id}: incomplete inner results for outer fold {f}")
                results = []
                for ci, config in enumerate(grids[system.id]):
                    pooled = _macro(toolkit, values[:, ci, :], components)
                    per_inner = {}
                    for g in range(k):
                        mask = [j for j, i in enumerate(rows) if fold_of_row[i] == g]
                        if mask:
                            per_inner[str(g)] = _macro(toolkit, values[mask, ci, objective_at], [components[j] for j in mask],
                                                       (objective,))[objective]
                    results.append({"config_index": ci, "config": config, "label": config_label(config),
                                    "objective": pooled[objective], "metrics": pooled, "per_inner_fold": per_inner})
                best = max(r["objective"] for r in results)
                chosen = min(r["config_index"] for r in results if r["objective"] >= best - TIE_TOLERANCE)
                node = resolve(system, grids[system.id][chosen], {sid: selected[sid][f] for sid in system.dependencies()})
                selected[system.id][f], selected_index[system.id][f] = node, chosen
                records.append({"outer_fold": f, "inner_folds": [g for g in range(k) if g != f], "queries": len(rows),
                                "components": len(set(components)), "results": results, "selected_index": chosen,
                                "selected": grids[system.id][chosen], "selected_label": config_label(grids[system.id][chosen]),
                                "resolved": node, "uses_labels": uses_labels(node)})
            S.write_json(out_dir / "selection" / f"{system.id}.json",
                         {"system": system.id, "family": system.family, "objective": objective,
                          "selection_population": population.name, "grid": grids[system.id],
                          "tie_rule": lock["tie_rule"]["selection"], "outer_folds": records})
        say(f"[{_utc()}] level {level_number} selected in {time.perf_counter() - level_started:.1f}s: "
            + ", ".join(f"{s.id}={[selected_index[s.id][f] for f in range(k)]}" for s in level))

    notify("scoring_outer", {})
    details: dict[str, list[dict]] = {s.id: [] for s in ordered}
    for f in range(k):
        fit = tuple(sorted(set(range(k)) - {f}))
        for block in all_blocks[f]:
            for system in ordered:
                system_started = time.perf_counter()
                scores = engine.node_matrix(selected[system.id][f], block, fit)
                kept_full, full = engine.evaluate(scores, block, "full", top_k=top_k)
                specific = dict(zip(*engine.evaluate(scores, block, "specific", top_k=0)))
                for local, row in zip(kept_full, full):
                    query = data.queries[block.rows[local]]
                    record = asdict(S.DetailRecord(query.query_id, query.component_id, query.fold, list(data.references[query.query_id]),
                                                   [int(r) for r in row["positive_ranks"]],
                                                   [data.candidate_ids[int(i)] for i in row["topk"]]))
                    record.update({m: float(row[m]) for m in METRICS})
                    record["config_index"] = selected_index[system.id][f]
                    spec_row = specific.get(local)
                    record["specific"] = None if spec_row is None else {
                        "positive_ids": [data.candidate_ids[p] for p in spec_row["positive_idx"]],
                        "positive_ranks": [int(r) for r in spec_row["positive_ranks"]], **{m: float(spec_row[m]) for m in METRICS}}
                    details[system.id].append(record)
                timings[system.id]["outer_seconds"] += time.perf_counter() - system_started
            engine.cache.clear()
        engine.small.clear()
        say(f"[{_utc()}] outer fold {f} scored ({sum(len(b.rows) for b in all_blocks[f])} queries)")

    outputs = {}
    for system in ordered:
        rows = sorted(details[system.id], key=lambda r: data.query_index[r["query_id"]])
        path = out_dir / "details" / f"{system.id}.jsonl"
        S.write_jsonl(path, rows)
        notify("details", {"path": str(path)})
        outputs[f"details/{system.id}.jsonl"] = S.sha256_file(path)
        outputs[f"selection/{system.id}.json"] = S.sha256_file(out_dir / "selection" / f"{system.id}.json")
    costs = {"wall_seconds": time.perf_counter() - started, "index_build_seconds": engine.index_seconds, "systems": timings,
             "peak_memory_bytes": peak_memory_bytes(), "cache_peak_bytes": engine.cache.peak, "max_matrix_rows": engine.max_rows,
             "chunk_rows": chunk_rows, "candidates": len(data.candidate_ids),
             "lexical_indexes": {kind: index.receipt() for kind, index in engine.indexes.items()}}
    result = {"status": "complete", "finished_utc": _utc(), "run_lock_sha256": S.sha256_file(out_dir / "run_lock.json"),
              "selection_population": population.name, "objective": objective, "summary": summarize(data, details, toolkit),
              "costs": costs, "outputs": outputs,
              "selected": {sid: {str(f): selected_index[sid][f] for f in range(k)} for sid in selected_index}}
    S.write_json(out_dir / "summary.json", result)
    say(f"[{_utc()}] complete in {costs['wall_seconds']:.1f}s")
    return result
