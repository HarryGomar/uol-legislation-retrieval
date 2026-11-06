"""Where the harness reads its inputs: the local thesis checkout, or a development-only bundle on the run computer.

The existing modules keep their own module-level paths (``embedding_push.common.RELEASE``, ``fusion_ltr.FUSION``,
``gap_tests.d58_refit.D`` ...). ``activate`` selects a profile and, for a bundle or another release, redirects those
module attributes before any release is loaded (the same technique as ``round3.rebuild.patch_paths``; no module file
is edited). The release is a parameter: ``RELEASES`` maps an id to its local paths, so a v12 catalogue is one entry.

Profiles
* ``local`` (default): this checkout; v11 needs no redirection.
* ``bundle``: ``<data>/`` laid out by ``uol.final.bundle`` (dev rows only; checked by ``guard.check_bundle``).
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

# release id -> paths relative to ROOT (local profile)
RELEASES: dict[str, dict[str, str]] = {
    "v11": {
        "release": "data/releases/uol_c01_statutes_20261004_v11_stratified",
        "split": "research/state/embedding_push_20261004/SPLIT.json",
        "ep_run": "experiments/runs/embedding_push_20261004",
        "derived_ftl": "data/derived/facts_to_law_v2",
        "qwen8b": "data/derived/embedding_push_qwen8b_v1",
        "qwen8b_headings": "data/derived/embedding_push_qwen8b_headings_20261019",
        "qwen3_0p6b": "data/derived/qwen3_emb_0p6b_v1",
        "qwen3_4b": "data/derived/qwen3_emb_4b_v1",
        "c_hybrid": "experiments/runs/round3_best_case_20261013/arms/C_hybrid",
        "heading_paths": "experiments/runs/round3_best_case_20261013/phase2/statute_graph/paths.json",
        "statute_graph": "experiments/runs/round3_best_case_20261013/phase2/statute_graph/graph.npz",
        "d58_sources": "experiments/runs/gap_tests_20261019/d58/sources",
        "e5_large_legal": "experiments/runs/eval_lenses_20261015/e5_large_legal",
        "units": "experiments/runs/scale_curve_20261016/units",
        "historical": "experiments/runs/round2_20261004/B",
        "sources_jsonl": "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl",
        "new_vectors": "data/derived/final_20261021_vectors",
    },
    # "v12": {...}   # a later catalogue: add its paths here; loaded systems without v12 artefacts are skipped
}

# bundle layout: the same keys, relative to <bundle>/data
BUNDLE_LAYOUT: dict[str, str] = {
    "release": "release",
    "split": "state/SPLIT.json",
    "ep_run": "ep_run",
    "derived_ftl": "derived/facts_to_law_v2",
    "qwen8b": "derived/embedding_push_qwen8b_v1",
    "qwen8b_headings": "derived/embedding_push_qwen8b_headings_20261019",
    "qwen3_0p6b": "derived/qwen3_emb_0p6b_v1",
    "qwen3_4b": "derived/qwen3_emb_4b_v1",
    "c_hybrid": "round3/C_hybrid",
    "heading_paths": "round3/statute_graph/paths.json",
    "statute_graph": "round3/statute_graph/graph.npz",
    "d58_sources": "gap_tests/d58/sources",
    "e5_large_legal": "derived/e5_large_legal",
    "units": "units",
    "historical": "round2/B",
    "sources_jsonl": "derived/dev_sources.jsonl",
    "new_vectors": "vectors_new",
}


L = "https://www.diputados.gob.mx/LeyesBiblio/pdf/"
# RUN_PLAN v2 §1 "Label corrections" (CATALOGUE_GAPS.md: P7 state-title extension; silent drop after "en relación con
# el N"). Applied only with labels="corrected" (the run computer's default after G1); G1 runs on v11 labels.
LABEL_CORRECTIONS: dict[str, dict[str, list[str]]] = {
    "sjf:tesis:2028377:hechos": {"remove": [L + "LISSSTE.pdf#articulo=135", L + "LISSSTE.pdf#articulo=136"], "add": [],
                                 "why": "Zacatecas state ISSSTE law bound to the federal LISSSTE (P7); no label left"},
    "sjf:tesis:2027999:hechos": {"remove": [L + "LISSSTE.pdf#articulo=74", L + "LISSSTE.pdf#articulo=75"], "add": [],
                                 "why": "Zacatecas state ISSSTE law bound to the federal LISSSTE (P7)"},
    "sjf:tesis:2030099:hechos": {"remove": [], "add": [L + "LFT.pdf#articulo=698"],
                                 "why": "LFT 698 dropped after 'en relacion con el 698' (silent drop)"},
}


def apply_label_corrections(corrections: dict[str, dict[str, list[str]]] | None = None) -> dict[str, Any]:
    """Correct the loaded release in place (the lru-cached ``ep.release()`` object every module shares)."""
    from uol.embedding_push import common as ep
    corrections = LABEL_CORRECTIONS if corrections is None else corrections
    d = ep.release()
    if getattr(d, "_uol_final_corrected", False):
        return {"already": True}
    done = {}
    generic = set(d.generic_idx.tolist())
    for qid, c in corrections.items():
        if qid not in d.query_index:
            continue
        i = d.query_index[qid]
        refs = [x for x in d.references[qid] if x not in set(c["remove"])] + [x for x in c["add"] if x not in d.references[qid]]
        d.references[qid] = refs
        d.positive_idx[i] = [d.candidate_index[x] for x in refs if x in d.candidate_index]
        d.specific_idx[i] = [p for p in d.positive_idx[i] if p not in generic]
        done[qid] = {"positives": refs, "why": c.get("why")}
    d._uol_final_corrected = True
    return done


@dataclass
class Profile:
    kind: str                                   # local | bundle
    release_id: str
    paths: dict[str, Path]
    store: Path                                 # results store root (experiments/runs/final_20261021 or the run dir)
    work: Path                                  # large intermediates (feature tables, caches)
    data_dir: Path | None = None
    patches: dict[str, str] = field(default_factory=dict)
    bundle_meta: dict[str, Any] = field(default_factory=dict)
    sources: str = "frozen"                     # frozen | rebuilt (sources from <work>/ep_rebuild)
    labels: str = "v11"                         # v11 | corrected (LABEL_CORRECTIONS applied)
    variant: str = ""                           # "" | devt | devb | devn (uol.final.variants)

    @property
    def tag(self) -> str:
        """Suffix of result names in this profile: '@rebuilt' (sources from code), '@v11' (bundle run on uncorrected
        labels, i.e. G1)."""
        t = "@rebuilt" if self.sources == "rebuilt" else ""
        if self.kind == "bundle" and self.labels == "v11":
            t += "@v11"
        return t

    def path(self, key: str) -> Path:
        return self.paths[key]

    def describe(self) -> dict[str, Any]:
        d = asdict(self)
        return json.loads(json.dumps(d, default=str))


_ACTIVE: Profile | None = None


def active() -> Profile:
    global _ACTIVE
    if _ACTIVE is None:
        data = os.environ.get("UOL_FINAL_DATA")
        _ACTIVE = activate_bundle(Path(data)) if data else activate_local()
    return _ACTIVE


def _clear_caches() -> None:
    from uol.embedding_push import common as ep
    for fn in (ep.release, ep.tie_order, ep.roles):
        fn.cache_clear()
    try:
        from uol.round3 import common as R3
        R3._counts_for.cache_clear()
        R3.publication_dates.cache_clear()
    except Exception:                                       # pragma: no cover - optional module state
        pass


def _set(patches: dict[str, str], module: Any, attr: str, value: Any) -> None:
    setattr(module, attr, value)
    patches[f"{module.__name__}.{attr}"] = str(value)


def _redirect(paths: dict[str, Path], work: Path, patches: dict[str, str], sources: str = "frozen",
              labels: str = "v11") -> None:
    """Point the inherited modules at ``paths`` (only attributes that exist; nothing is written).

    ``embedding_push.common`` is set first, before any stage module is imported, because those modules derive their
    own folders from ``common.RUN`` / ``common.DERIVED`` at import (``base.OUT``, ``qwen_endpoint.RUN``,
    ``memory_proto.WORK`` ...). ``sources="rebuilt"`` sends ``RUN`` / ``SCORES`` to ``<work>/ep_rebuild``."""
    from uol.embedding_push import common as ep
    run = (work / ("ep_rebuild_v11" if labels == "v11" else "ep_rebuild")) if sources == "rebuilt" else paths["ep_run"]
    _set(patches, ep, "RELEASE", paths["release"])
    _set(patches, ep, "SPLIT_PATH", paths["split"])
    _set(patches, ep, "STATE", paths["split"].parent)
    _set(patches, ep, "FREEZE_PATH", paths["split"].parent / "NO_FREEZE.json")      # lockbox evaluation never allowed
    _set(patches, ep, "RUN", run)
    _set(patches, ep, "SCORES", run / "scores")
    _set(patches, ep, "DERIVED", paths["derived_ftl"])
    from uol.embedding_push import base, fusion_ltr as fl
    _set(patches, fl, "FUSION", paths["ep_run"] / "fusion")                        # frozen meta and hyper-parameters
    _set(patches, fl, "PIPES", paths["ep_run"] / "fusion" / "pipelines")
    _set(patches, fl, "FEATDIR", paths["ep_run"] / "fusion" / "features")
    _set(patches, fl, "CACHE", work / "fusion_cache")
    _set(patches, fl, "FLOG", work / "fusion_ltr.log")
    _set(patches, base, "OUT", run / "base")
    _set(patches, base, "BGE_DIR", paths["derived_ftl"] / "bge_m3_chunks_v1")
    _set(patches, base, "E5_DIR", paths["derived_ftl"] / "e5_small_chunks_v1")
    _set(patches, base, "E5_QUERIES", paths["derived_ftl"] / "e5_small_queries_v11" / "query_vectors.npy")
    from uol.round3 import common as R3
    _set(patches, R3, "EP_RUN", run)
    _set(patches, R3, "SOURCES_JSONL", paths["sources_jsonl"])
    _set(patches, R3, "ROUND2", paths["historical"].parent)
    _set(patches, R3, "RUN", work / "round3_run")
    _set(patches, R3, "ARMS", work / "round3_run" / "arms")
    _set(patches, R3, "STATE", work / "round3_state")
    _set(patches, R3, "RUN_LEDGER", work / "round3_state" / "RUN_LEDGER.jsonl")
    _set(patches, R3, "LOCKBOX_LEDGER", work / "round3_state" / "LOCKBOX_LEDGER.jsonl")
    from uol.gap_tests import common as GC
    _set(patches, GC, "OUT", work / "gap_tests")
    from uol.gap_tests import lexical_grid as LG
    _set(patches, LG, "PATHS", paths["heading_paths"])
    from uol.eval_lenses import core as EL                   # the E12 agent's neighbour tool reads the statute graph
    if "statute_graph" in paths:
        _set(patches, EL, "GRAPH", paths["statute_graph"])
    import sys as _sys
    qe = _sys.modules.get("uol.embedding_push.qwen_endpoint")
    if qe is None:
        from uol.embedding_push import qwen_endpoint as qe
    for attr, val in (("DERIVED", paths["qwen8b"]), ("STORE_DIR", paths["qwen8b"] / "store"),
                      ("PROBES_LOG", work / "qwen_probes.jsonl"), ("RUN", run / "qwen"),
                      ("RUN_LOG", run / "qwen" / "run.log"), ("REQUEST_LOG", run / "qwen" / "requests.jsonl"),
                      ("RUNS_LOG", run / "qwen" / "runs.jsonl"), ("STOP_FILE", run / "qwen" / "STOP"),
                      ("SEARCH_LOG", run / "qwen" / "search_log.jsonl"), ("SELECTION_DIR", run / "qwen" / "selection"),
                      ("CACHE_DIR", run / "qwen" / "cache"), ("INNER_DIR", run / "qwen" / "cache" / "inner"),
                      ("ADAPT_DIR", run / "qwen" / "adapt"), ("HYDE_DIR", run / "qwen" / "hyde"),
                      ("HYDE_GEN", run / "qwen" / "hyde" / "generations.jsonl")):
        if hasattr(qe, attr):
            _set(patches, qe, attr, val)
    from uol.scale_curve import common as SC
    _set(patches, SC, "QWEN8B", paths["qwen8b"])
    _set(patches, SC, "UNITS_DIR", paths["units"])

def install_labelled_evaluate() -> bool:
    """``embedding_push.common.evaluate`` skips admitted rows without a relevant item in the evaluated view.

    A label correction can leave a case with no positive (2028377, RUN_PLAN v2 §1); its nDCG is undefined and
    ``metrics.component_macro`` refuses the whole call ("drop them first"). The harness populations already drop such
    rows (``populations.labelled``); the inherited stages (``base.stage_summary``, memory_proto / case_sim selection,
    the E2 adapter's validation) call ``common.evaluate`` directly, so the same rule is applied there (21 Oct 2026).
    A no-op when every admitted row has a positive (v11 labels, G1)."""
    import numpy as np
    from uol.embedding_push import common as ep
    if getattr(ep.evaluate, "_uol_final", False):
        return False
    orig = ep.evaluate

    def evaluate(scores, rows, population: str = "P_spec", **kw):
        rows = np.asarray(rows, dtype=np.int64)
        if len(rows) == len(scores):
            d = ep.release()
            pos = d.specific_idx if ep.POPULATIONS[population].view == "specific" else d.positive_idx
            keep = np.fromiter((len(pos[int(r)]) > 0 for r in rows), dtype=bool, count=len(rows))
            if not keep.all():
                idx = np.flatnonzero(keep)
                scores, rows = np.asarray(scores[idx]), rows[idx]
        return orig(scores, rows, population, **kw)
    evaluate._uol_final = True
    evaluate.__wrapped__ = orig
    ep.evaluate = evaluate
    return True


REDIRECTED = (
    ("uol.embedding_push.common", ("RELEASE", "SPLIT_PATH", "STATE", "FREEZE_PATH", "RUN", "SCORES", "DERIVED",
                                   "evaluate")),
    ("uol.embedding_push.fusion_ltr", ("FUSION", "PIPES", "FEATDIR", "CACHE", "FLOG", "train_rows_of")),
    ("uol.embedding_push.base", ("OUT", "BGE_DIR", "E5_DIR", "E5_QUERIES")),
    ("uol.embedding_push.memory_proto", ("VECTOR_FILES",)),
    ("uol.embedding_push.qwen_endpoint", ("DERIVED", "STORE_DIR", "PROBES_LOG", "RUN", "RUN_LOG", "REQUEST_LOG", "RUNS_LOG",
                                          "STOP_FILE", "SEARCH_LOG", "SELECTION_DIR", "CACHE_DIR", "INNER_DIR", "ADAPT_DIR",
                                          "HYDE_DIR", "HYDE_GEN")),
    ("uol.scale_curve.common", ("QWEN8B", "UNITS_DIR")),
    ("uol.round3.common", ("EP_RUN", "SOURCES_JSONL", "ROUND2", "RUN", "ARMS", "STATE", "RUN_LEDGER", "LOCKBOX_LEDGER")),
    ("uol.round3.c_hybrid", ("OUT",)),
    ("uol.round3.e2_adapter", ("pair_table", "QWEN")),
    ("uol.round3.features", ("NESTED_DIR",)),
    ("uol.gap_tests.common", ("OUT",)),
    ("uol.gap_tests.lexical_grid", ("PATHS",)),
    ("uol.gap_tests.d58_refit", ("D", "SRC", "V2DIR", "QSEL", "BSEL", "DEV", "DEV_POS", "log")),
)


def snapshot() -> dict[Any, Any]:
    """Current values of every attribute ``activate`` may redirect, for the modules already imported (tests restore
    them afterwards; modules first imported during the test are dropped from ``sys.modules`` on restore)."""
    import sys
    out: dict[Any, Any] = {"__modules__": set(sys.modules)}
    for mod, attrs in REDIRECTED:
        m = sys.modules.get(mod)
        if m is None:
            continue
        for a in attrs:
            if hasattr(m, a):
                out[(mod, a)] = getattr(m, a)
    return out


def reset_caches() -> None:
    """Per-process caches of the uol.final adapters (texts, indexes, fitted models, the dev pair table)."""
    import sys
    for name, attrs in (("uol.final.adapters.lexical", ("_TEXTS", "_CTX")), ("uol.final.adapters.topic", ("_MODELS",)),
                        ("uol.final.adapters.ltr", ("_TABLES",)), ("uol.round3.features", ("_SHARED",))):
        m = sys.modules.get(name)
        for a in attrs:
            if m is not None and isinstance(getattr(m, a, None), dict):
                getattr(m, a).clear()
    m = sys.modules.get("uol.final.measures")
    if m is not None:
        m._instr_codes.cache_clear()
    m = sys.modules.get("uol.final.registry")
    if m is not None:
        m._LOADED.clear()


def restore(snap: dict[Any, Any]) -> None:
    global _ACTIVE
    import sys
    for key, v in snap.items():
        if key == "__modules__":
            continue
        mod, a = key
        if mod in sys.modules:
            setattr(sys.modules[mod], a, v)
    for name in list(sys.modules):
        if name not in snap["__modules__"] and name.startswith(("uol.gap_tests.d58_refit", "uol.round3.c_hybrid")):
            del sys.modules[name]
    _ACTIVE = None
    reset_caches()
    _clear_caches()


def redirect_d58(profile: Profile) -> Any:
    """Import ``gap_tests.d58_refit`` with its output folders in the profile's work dir and its DEV rows recomputed.

    d58_refit fixes DEV / DEV_POS at import from the active release, so this is called after ``activate``."""
    from uol.embedding_push import common as ep
    from uol.gap_tests import d58_refit as d58
    import numpy as np
    # the pair table carries a label-derived column (instrument prior), so v11 (G1) and corrected labels differ
    sub = "d58" + ("_rebuilt" if profile.sources == "rebuilt" else "") + \
        ("_v11" if profile.kind == "bundle" and profile.labels == "v11" else "")
    d58.D = profile.work / sub
    d58.SRC = profile.work / sub / "sources"
    d58.V2DIR = profile.path("qwen8b_headings")
    d58.QSEL = profile.path("ep_run") / "qwen" / "selection"
    d58.BSEL = profile.path("ep_run") / "base" / "selection"
    from uol.final import guard
    d58.DEV = guard.dev_rows_only(ep.dev_rows(), "d58 DEV")
    d58.DEV_POS = np.full(len(ep.release().queries), -1, dtype=np.int64)
    d58.DEV_POS[d58.DEV] = np.arange(len(d58.DEV))
    d58.log = lambda msg: _log(profile, "d58", msg)
    profile.patches["uol.gap_tests.d58_refit.D"] = str(d58.D)
    return d58


def _log(profile: Profile, name: str, msg: str) -> None:
    import time
    p = profile.work / "logs" / f"{name}.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def activate_local(release_id: str = "v11", store: Path | None = None, work: Path | None = None) -> Profile:
    global _ACTIVE
    if release_id not in RELEASES:
        raise KeyError(f"release {release_id!r} is not registered in uol.final.context.RELEASES")
    paths = {k: ROOT / v for k, v in RELEASES[release_id].items()}
    store = Path(store) if store else ROOT / "experiments/runs/final_20261021"
    work = Path(work) if work else ROOT / ".cache/final_work"
    patches: dict[str, str] = {}
    if release_id != "v11":
        _redirect(paths, work, patches)
    _clear_caches()
    reset_caches()
    _ACTIVE = Profile("local", release_id, paths, store, work, None, patches)
    return _ACTIVE


def activate_bundle(data_dir: Path, store: Path | None = None, work: Path | None = None) -> Profile:
    """Bundle profile: ``data_dir`` is ``<bundle>/data``; results default to ``<bundle>/../run``."""
    global _ACTIVE
    from uol.final import guard
    data_dir = Path(data_dir)
    meta = guard.check_bundle(data_dir)
    nl = data_dir / "nltk_data"
    if nl.exists():                                   # the Spanish stopword list L* needs (never downloaded)
        os.environ.setdefault("NLTK_DATA", str(nl))
        try:
            import nltk
            if str(nl) not in nltk.data.path:
                nltk.data.path.insert(0, str(nl))
        except ImportError:                           # pragma: no cover
            pass
    paths = {k: data_dir / v for k, v in BUNDLE_LAYOUT.items()}
    root = data_dir.parent
    store = Path(store) if store else Path(os.environ.get("UOL_FINAL_STORE", root / "run" / "final_20261021"))
    work = Path(work) if work else Path(os.environ.get("UOL_FINAL_WORK", root / "work"))
    variant = os.environ.get("UOL_FINAL_VARIANT", "")
    if variant:
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; known {VARIANTS}")
        vroot = work / "variants" / variant
        if variant in ("devt", "devb"):
            if not (vroot / "release" / "manifest.json").exists():
                raise FileNotFoundError(f"variant {variant} not built yet ({vroot}); run its build job first")
            paths["release"], paths["split"] = vroot / "release", vroot / "SPLIT.json"
            if (vroot / "derived" / "qwen8b" / "query_ids.json").exists():
                paths["qwen8b"] = vroot / "derived" / "qwen8b"
        work = vroot
    _seed_arms(data_dir, work)
    patches: dict[str, str] = {}
    sources = os.environ.get("UOL_FINAL_SOURCES", "frozen")
    _redirect(paths, work, patches, sources, os.environ.get("UOL_FINAL_LABELS", "corrected"))
    if variant == "devb":
        _redirect_devb_dense(work, patches)
    _clear_caches()
    reset_caches()
    labels = os.environ.get("UOL_FINAL_LABELS", "corrected")
    if labels not in ("v11", "corrected"):
        raise ValueError(f"UOL_FINAL_LABELS must be v11 or corrected, not {labels!r}")
    _ACTIVE = Profile("bundle", str(meta.get("release_id", "v11")), paths, store, work, data_dir, patches, meta, sources,
                      labels)
    from uol.embedding_push import common as ep
    guard.check_release_dev_only(ep.release().queries, list(ep.roles()))
    if install_labelled_evaluate():
        patches["uol.embedding_push.common.evaluate"] = "drops admitted rows without a relevant item (label corrections)"
    if labels == "corrected":
        cpath = data_dir / "label_corrections.json"
        corr = json.loads(cpath.read_text(encoding="utf-8")) if cpath.exists() else LABEL_CORRECTIONS
        patches["label_corrections"] = json.dumps(apply_label_corrections(corr), default=str)[:2000]
    _ACTIVE.variant = variant
    if variant in TEXT_VARIANTS:                       # query-text variants: DEV-N clause drop, DEV-L lay wording (C6),
        mod = text_module(variant)                     # DEV-S shortened wording (D109)
        patches[f"{variant}_text_override"] = json.dumps(mod.apply_text_override())
    return _ACTIVE


VARIANTS = ("devt", "devb", "devn", "devl", "devs25", "devs50", "devs100")
# variants that only change the query text in the process; the dense systems read re-encoded <variant>__<key> queries
TEXT_VARIANTS = ("devn", "devl", "devs25", "devs50", "devs100")


def text_module(variant: str):
    """The module that defines a text variant's rows and override (devs25 / devs50 / devs100 share ``uol.final.devs``)."""
    return __import__(f"uol.final.{'devs' if variant.startswith('devs') else variant}", fromlist=["x"])


def _redirect_devb_dense(vroot: Path, patches: dict[str, str]) -> bool:
    """DEV-B: the bge-m3 and e5-small query vectors with the recovered rows (``variants.devb_dense_queries``), for
    ``base.stage_dense`` and memory_proto / case_sim (``memory_proto.qvec``). Before that job has run nothing is
    redirected (the stages then stop on the missing rows, as before)."""
    d = vroot / "derived" / "facts_to_law_v2"
    bge, e5 = d / "bge_m3_chunks_v1", d / "e5_small_queries_v11" / "query_vectors.npy"
    if not ((bge / "query_vectors.npy").exists() and e5.exists() and (d / "dense_queries_receipt.json").exists()):
        return False
    from uol.embedding_push import base, memory_proto as mp
    _set(patches, base, "BGE_DIR", bge)
    _set(patches, base, "E5_QUERIES", e5)
    _set(patches, mp, "VECTOR_FILES", {**mp.VECTOR_FILES, "bge": bge / "query_vectors.npy", "e5": e5})
    for k in ("bge", "e5"):
        mp._VEC.pop(k, None)
    return True


def _seed_arms(data_dir: Path, work: Path) -> None:
    """Copy the bundled Round 3 arm inputs (M1 neighbours and selection, E2 selection, L2 issue decompositions) into
    the work folder the Round 3 modules write to, once."""
    import shutil
    paths_json = data_dir / BUNDLE_LAYOUT["heading_paths"]
    q = work / "round3_run" / "phase2" / "statute_graph" / "paths.json"
    if paths_json.exists() and not q.exists():
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths_json, q)
    src = data_dir / "round3" / "arms"
    if not src.exists():
        return
    dst = work / "round3_run" / "arms"
    for p in src.rglob("*"):
        if p.is_file():
            q = dst / p.relative_to(src)
            if not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, q)


def activate_custom(paths: dict[str, Path], store: Path, work: Path, release_id: str = "fake",
                    kind: str = "custom") -> Profile:
    """Tests: arbitrary paths (a fake release with a lockbox, or a fake bundle)."""
    global _ACTIVE
    patches: dict[str, str] = {}
    sources = os.environ.get("UOL_FINAL_SOURCES", "frozen")
    _redirect(paths, Path(work), patches, sources)
    _clear_caches()
    reset_caches()
    labels = os.environ.get("UOL_FINAL_LABELS", "v11")
    _ACTIVE = Profile(kind, release_id, dict(paths), Path(store), Path(work), None, patches, {}, sources, labels)
    if install_labelled_evaluate():
        patches["uol.embedding_push.common.evaluate"] = "drops admitted rows without a relevant item (label corrections)"
    if labels == "corrected":
        patches["label_corrections"] = json.dumps(apply_label_corrections(), default=str)[:2000]
    return _ACTIVE


# --------------------------------------------------------------------------- worker processes (21 Oct 2026)
# The original stages (memory_proto, case_sim, qwen_endpoint) run group jobs in process pools. On Windows a pool
# worker is a fresh interpreter: without this it would see the thesis-PC paths of the unpatched modules, not the
# profile of the job that started it. ``install_worker_init`` gives every pool created afterwards an initializer that
# re-activates the same profile (the UOL_FINAL_* environment, which workers inherit, supplies sources, labels and
# variant).

def worker_spec(prof: Profile | None = None) -> dict[str, Any]:
    p = prof or active()
    return {"kind": p.kind, "release_id": p.release_id, "paths": {k: str(v) for k, v in p.paths.items()},
            "store": str(p.store), "work": str(p.work), "data_dir": str(p.data_dir) if p.data_dir else None,
            "sources": p.sources, "labels": p.labels, "variant": p.variant}


def worker_init(spec: dict[str, Any]) -> None:
    for key, var in (("sources", "UOL_FINAL_SOURCES"), ("labels", "UOL_FINAL_LABELS"), ("variant", "UOL_FINAL_VARIANT")):
        if spec.get(key):
            os.environ[var] = spec[key]
    if spec["kind"] == "bundle" and spec["data_dir"]:
        root_work = Path(spec["work"]).parents[1] if spec.get("variant") else Path(spec["work"])
        prof = activate_bundle(Path(spec["data_dir"]), store=Path(spec["store"]), work=root_work)
    elif spec["kind"] == "local":
        prof = activate_local(spec["release_id"], store=Path(spec["store"]), work=Path(spec["work"]))
    else:
        prof = activate_custom({k: Path(v) for k, v in spec["paths"].items()}, store=Path(spec["store"]),
                               work=Path(spec["work"]), release_id=spec["release_id"], kind=spec["kind"])
    if prof.sources == "rebuilt":
        from uol.final import rebuild
        rebuild.patch_runtime(prof)


def install_worker_init(prof: Profile | None = None) -> None:
    import concurrent.futures as cf
    import multiprocessing.context as mpc
    spec = worker_spec(prof)
    if not getattr(cf.ProcessPoolExecutor.__init__, "_uol_final", False):
        orig_init = cf.ProcessPoolExecutor.__init__

        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
            if initializer is None:
                initializer, initargs = worker_init, (_WORKER_SPEC[0],)
            orig_init(self, max_workers, mp_context, initializer, initargs, **kw)
        __init__._uol_final = True
        cf.ProcessPoolExecutor.__init__ = __init__
    if not getattr(mpc.BaseContext.Pool, "_uol_final", False):
        orig_pool = mpc.BaseContext.Pool

        def Pool(self, processes=None, initializer=None, initargs=(), maxtasksperchild=None):
            if initializer is None:
                initializer, initargs = worker_init, (_WORKER_SPEC[0],)
            return orig_pool(self, processes, initializer, initargs, maxtasksperchild)
        Pool._uol_final = True
        mpc.BaseContext.Pool = Pool
    _WORKER_SPEC[0] = spec


_WORKER_SPEC: list[dict[str, Any] | None] = [None]
