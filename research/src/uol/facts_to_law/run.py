"""Command line for nested cross-fitted scoring of declared systems on a release.

    python -m uol.facts_to_law.run --release <dir> --systems <yaml|json|builtin> --out <dir>
        [--families L,P,K] [--providers <json>] [--skip-unavailable]

``run_lock.json`` (systems, grids, fold salt, code, release and input hashes)
is written before any scoring; an existing non-empty output directory is
refused. ``--systems builtin`` gives the PREREGISTRATION section 3 systems
(``default_systems``). Dense inputs come from ``--providers``, a JSON map
``encoder -> {query_ids, query_vectors, unit_vectors, units}`` of precomputed
files: ``query_ids`` is a JSON list or a JSONL with ``query_id``; ``units`` is a
JSONL with ``candidate_id`` and ``chunk_index`` per ``unit_vectors`` row, or is
omitted when ``unit_vectors`` holds one vector per catalogue article. Encoder
``e5_small`` serves the dense modes ``e5_small:firstp|maxp|meanp`` and its query
vectors serve dense kNN similarity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import spec as S
from .crossfit import (METRICS, POPULATIONS, ChannelSpec, DenseArrays, ScoreProviders, SystemSpec, bound_articles,
                       load_candidates, providers_from_arrays, run_crossfit, system_from_dict, system_providers,
                       validate_systems)
from .lexical import BM25_GRID

ENCODER = "e5_small"
W_CHAR = (0.0, 0.25, 0.5, 0.75, 1.0)
W_PRIOR = (0.0, 0.1, 0.2, 0.3)
W_DENSE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
KNN_K = (5, 10, 20, 50)
W_KNN = (0.25, 0.5, 0.75)
KNN_WEIGHTING = "sim"
RRF_K = 60


def _pairs(values: Iterable[float]) -> list[list[float]]:
    """Two-channel weights (1 - w, w)."""
    return [[round(1.0 - w, 10), float(w)] for w in values]


def _ref(system: str, name: str | None = None) -> ChannelSpec:
    return ChannelSpec(kind="system", system=system, name=name or system)


def default_systems(encoder: str = ENCODER, *, knn_weighting: str = KNN_WEIGHTING) -> list[SystemSpec]:
    """The PREREGISTRATION section 3 systems and grids.

    ``L_bm25_best`` is a helper: 'the selected BM25 variant' fused by L_fuse.
    kNN votes use ``knn_weighting`` (the pre-registration fixes k and w_knn only).
    """
    bm25 = {"k1": list(BM25_GRID["k1"]), "b": list(BM25_GRID["b"])}
    knn = {"weighting": knn_weighting}
    robust = {"normalize": "robust_z"}
    return [
        SystemSpec("L_bm25", "L", (ChannelSpec("lexical", "bm25", lexical="bm25", grid=bm25),),
                   description="BM25 on full instrument_article_body, October tokenizer"),
        SystemSpec("L_bm25_stem", "L", (ChannelSpec("lexical", "bm25_stem", lexical="bm25_stem", grid=bm25),),
                   description="L_bm25 with Spanish Snowball stemming and stopwords"),
        SystemSpec("L_char", "L", (ChannelSpec("lexical", "char", lexical="char"),), description="character TF-IDF, October settings"),
        SystemSpec("L_bm25_best", "L", (_ref("L_bm25"), _ref("L_bm25_stem")), fusion="select",
                   description="helper: the selected BM25 variant"),
        SystemSpec("L_fuse", "L", (_ref("L_bm25_best", "bm25"), _ref("L_char", "char")), fusion="linear", fusion_params=robust,
                   fusion_grid={"weights": _pairs(W_CHAR)}, description="robust-z fusion of the selected BM25 variant and L_char, w_char grid"),
        SystemSpec("L_star", "L", (_ref("L_bm25"), _ref("L_bm25_stem"), _ref("L_char"), _ref("L_fuse")), fusion="select",
                   description="L*: best lexical system per outer fold"),
        SystemSpec("P_prior", "P", (ChannelSpec("prior", "prior"),),
                   description="log1p(distinct training-fold components citing the article)"),
        SystemSpec("LP", "P", (_ref("L_star", "lex"),), prior_weights=W_PRIOR, description="L* + P_prior (robust-z), w_prior grid"),
        SystemSpec("K_lex", "K", (_ref("L_star", "lex"), ChannelSpec("knn", "knn", sim="lexical", lexical="bm25", params=knn,
                                                                     grid={"k": list(KNN_K)})),
                   fusion="linear", fusion_params=robust, fusion_grid={"weights": _pairs(W_KNN)},
                   description="kNN votes over training-fold components (BM25 query similarity) fused with L*"),
        SystemSpec("D_firstp", "D", (ChannelSpec("dense", "dense", provider=f"{encoder}:firstp"),), description="E5-small chunk 0"),
        SystemSpec("D_maxp", "D", (ChannelSpec("dense", "dense", provider=f"{encoder}:maxp"),), description="E5-small max over chunks"),
        SystemSpec("D_meanp", "D", (ChannelSpec("dense", "dense", provider=f"{encoder}:meanp"),), description="E5-small mean over chunks"),
        SystemSpec("H_rrf", "H", (_ref("L_star", "lex"), _ref("D_maxp", "dense")), fusion="rrf", fusion_params={"k": RRF_K},
                   description="RRF(L*, D_maxp), k = 60"),
        SystemSpec("H_lin", "H", (_ref("L_star", "lex"), _ref("D_maxp", "dense")), fusion="linear", fusion_params=robust,
                   fusion_grid={"weights": _pairs(W_DENSE)}, description="robust-z fusion of L* and D_maxp, w_dense grid"),
        SystemSpec("H_star", "H", (_ref("H_rrf"), _ref("H_lin")), fusion="select", description="H*: best of H_rrf and H_lin"),
        SystemSpec("K_dense", "K", (_ref("L_star", "lex"), ChannelSpec("knn", "knn", sim="dense", provider=encoder, params=knn,
                                                                       grid={"k": list(KNN_K)})),
                   fusion="linear", fusion_params=robust, fusion_grid={"weights": _pairs(W_KNN)},
                   description="as K_lex with E5-small query cosine similarity"),
        SystemSpec("HP", "H", (_ref("H_star", "hybrid"),), prior_weights=W_PRIOR, description="H* + P_prior (secondary)"),
    ]


def load_systems(path: str | Path) -> list[SystemSpec]:
    """Systems from a YAML/JSON list (or ``{"systems": [...]}``); ``builtin`` gives ``default_systems()``."""
    if str(path) == "builtin":
        return default_systems()
    text = Path(path).read_text(encoding="utf-8")
    if str(path).lower().endswith((".yaml", ".yml")):
        import yaml

        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    rows = payload["systems"] if isinstance(payload, Mapping) else payload
    return [system_from_dict(row) for row in rows]


def select_families(systems: Sequence[SystemSpec], families: Sequence[str] | None) -> tuple[list[SystemSpec], list[str]]:
    """Systems of the requested families plus every system they reference."""
    if not families:
        return list(systems), []
    by_id = {s.id: s for s in systems}
    wanted = {s.id for s in systems if s.family in set(families)}
    stack, added = list(wanted), []
    while stack:
        for dep in by_id[stack.pop()].dependencies():
            if dep not in wanted:
                wanted.add(dep)
                added.append(dep)
                stack.append(dep)
    return [s for s in systems if s.id in wanted], sorted(added)


def drop_unavailable(systems: Sequence[SystemSpec], available: set[tuple[str, str]]) -> tuple[list[SystemSpec], dict[str, str]]:
    """Remove systems whose dense inputs are missing, and systems that reference them."""
    dropped: dict[str, str] = {}
    for system in systems:
        missing = sorted(need for need in system_providers(system) if need not in available)
        if missing:
            dropped[system.id] = f"missing dense inputs {missing}"
    changed = True
    while changed:
        changed = False
        for system in systems:
            if system.id not in dropped and any(dep in dropped for dep in system.dependencies()):
                dropped[system.id] = "references a dropped system"
                changed = True
    return [s for s in systems if s.id not in dropped], dropped


def _read_ids(path: Path) -> list[str]:
    if path.suffix == ".jsonl":
        return [row["query_id"] for row in S.read_jsonl(path)]
    return [str(v) for v in json.loads(path.read_text(encoding="utf-8"))]


def load_providers(path: str | Path, candidate_ids: Sequence[str]) -> ScoreProviders:
    """Dense providers from precomputed encoder files; every file is hashed into the receipts."""
    path = Path(path)
    registry: dict[str, DenseArrays] = {}
    receipts: dict[str, Any] = {"providers_file": str(path), "providers_sha256": S.sha256_file(path)}
    index = {c: i for i, c in enumerate(candidate_ids)}
    for encoder, entry in json.loads(path.read_text(encoding="utf-8")).items():
        files = {key: Path(value) for key, value in entry.items() if value}
        if "units" in files:
            units = [(index[row["candidate_id"]], int(row.get("chunk_index", 0))) for row in S.read_jsonl(files["units"])]
            unit_candidate, unit_chunk = [u[0] for u in units], [u[1] for u in units]
        else:
            unit_candidate, unit_chunk = list(range(len(candidate_ids))), None
        registry[encoder] = DenseArrays(_read_ids(files["query_ids"]), np.load(files["query_vectors"]),
                                        np.load(files["unit_vectors"]), unit_candidate, len(candidate_ids), unit_chunk_index=unit_chunk)
        receipts[encoder] = {key: {"path": str(value), "sha256": S.sha256_file(value)} for key, value in files.items()}
    return providers_from_arrays(registry, receipts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m uol.facts_to_law.run", description=__doc__.splitlines()[0])
    parser.add_argument("--release", required=True, type=Path)
    parser.add_argument("--systems", required=True, help="YAML/JSON systems file or 'builtin'")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--families", default="", help="comma-separated families, e.g. L,P,K (references are added)")
    parser.add_argument("--providers", type=Path, default=None, help="JSON map of precomputed dense inputs")
    parser.add_argument("--skip-unavailable", action="store_true", help="drop systems whose dense inputs are missing")
    parser.add_argument("--selection-population", default="S1_full", choices=sorted(POPULATIONS))
    parser.add_argument("--fitting-population", default="all", choices=sorted(POPULATIONS))
    parser.add_argument("--objective", default="ndcg_at_10", choices=METRICS)
    parser.add_argument("--tie-rule", default="index", choices=("index", "candidate_id"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--articles", type=Path, default=None, help="override the manifest's bound articles.jsonl")
    parser.add_argument("--chunk-rows", type=int, default=512, help="maximum query rows per score matrix")
    parser.add_argument("--cache-gib", type=float, default=1.0, help="score-matrix cache budget")
    args = parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"output directory exists and is not empty: {args.out}")
    systems = load_systems(args.systems)
    families = [f.strip() for f in args.families.split(",") if f.strip()]
    systems, added = select_families(systems, families)
    providers = ScoreProviders()
    if args.providers:
        articles, _ = bound_articles(args.release, args.articles)
        providers = load_providers(args.providers, load_candidates(articles, texts=False)[0])
    dropped: dict[str, str] = {}
    if args.skip_unavailable:
        systems, dropped = drop_unavailable(systems, providers.available())
    if not systems:
        parser.error("no systems left to run")
    validate_systems(systems)
    source: Any = "builtin" if args.systems == "builtin" else {"path": str(Path(args.systems).resolve()),
                                                                 "sha256": S.sha256_file(Path(args.systems))}
    extra = {"command": [sys.executable, "-m", "uol.facts_to_law.run", *(argv if argv is not None else sys.argv[1:])],
             "families": families, "dependencies_added": added, "systems_dropped": dropped, "systems_source": source}
    result = run_crossfit(args.release, systems, providers, folds=args.folds, selection_population=args.selection_population,
                          objective=args.objective, out_dir=args.out, articles_path=args.articles,
                          fitting_population=args.fitting_population, tie_rule=args.tie_rule, chunk_rows=args.chunk_rows,
                          cache_bytes=int(args.cache_gib * (1 << 30)), lock_extra={"cli": extra})
    print(json.dumps({sid: {name: row.get("ndcg_at_10") for name, row in table.items()} for sid, table in result["summary"].items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
