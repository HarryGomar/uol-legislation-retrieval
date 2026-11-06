"""Paths, the model registry, the 8B units and a dimension-aware content-hash vector store.

Units are exactly those of ``data/derived/embedding_push_qwen8b_v1``: the same article windows (MaxP for articles
above 8,000 Qwen tokens, title-prefixed later windows; ``qwen_endpoint.article_units``) and the same query strings
(variants a, b and c; ``qwen_endpoint.query_input``). ``build_units`` rebuilds them and checks every article key
against the 8B ``article_units.jsonl`` and every query variant against the 8B receipt digests before anything is
encoded. Only dev release rows are returned; lockbox rows are refused everywhere (``assert_dev``).
"""
from __future__ import annotations

import datetime
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.embedding_push import qwen_endpoint as qe
from uol.facts_to_law.spec import ROOT
from uol.scale_curve import store

PROGRAM = "scale_curve_20261016"
RUN = ROOT / "experiments/runs" / PROGRAM
STATE = ROOT / "research/state" / PROGRAM
REPORTS = ROOT / "research/reports" / PROGRAM
LEDGER = STATE / "LEDGER.jsonl"
STOP_FILE = RUN / "STOP"
PAUSE_FILE = ROOT / ".cache/round3_heavy.PAUSE"
QWEN8B = qe.DERIVED
VARIANTS = tuple(qe.VARIANTS)              # ("a", "b", "c"); the 8B selected "c" on dev
HF_HUB = Path.home() / ".cache/huggingface/hub"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    revision: str | None
    dim: int
    derived: Path
    kind: str                              # "local" (CPU, transformers) or "openai"
    params: str

    @property
    def snapshot(self) -> Path:
        return HF_HUB / f"models--{self.model_id.replace('/', '--')}" / "snapshots" / str(self.revision)

    @property
    def store_dir(self) -> Path:
        return self.derived / "store"


MODELS = {
    "qwen3_0p6b": ModelSpec("qwen3_0p6b", "Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", 1024,
                            ROOT / "data/derived/qwen3_emb_0p6b_v1", "local", "0.6B"),
    "qwen3_4b": ModelSpec("qwen3_4b", "Qwen/Qwen3-Embedding-4B", "5cf2132abc99cad020ac570b19d031efec650f2b", 2560,
                          ROOT / "data/derived/qwen3_emb_4b_v1", "local", "4B"),
    "te3l": ModelSpec("te3l", "text-embedding-3-large", None, 3072, ROOT / "data/derived/openai_te3l_v1", "openai",
                      "undisclosed"),
}


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def ledger(row: dict[str, Any]) -> None:
    append_jsonl(LEDGER, {"utc": utc(), **row})


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_default), encoding="utf-8")
    os.replace(tmp, path)


def _default(x: Any) -> Any:
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def log(message: str, name: str = "scale_curve") -> None:
    line = f"[{utc()}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode("ascii", "replace").decode(), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / f"{name}.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def assert_dev(rows: Iterable[int]) -> np.ndarray:
    """The rows as an array; raises if any is a lockbox row (no lockbox text is ever encoded or sent)."""
    rows = np.asarray(list(rows), dtype=np.int64)
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("lockbox rows are never encoded, sent or evaluated in the scale-curve checks")
    return rows


def content_key(text: str, model_id: str) -> str:
    return store.content_key(text, model_id)


# --------------------------------------------------------------------------- units (exactly the 8B units)


@dataclass(frozen=True)
class Item:
    group: str            # "articles" or "queries_<variant>"
    owner: int            # candidate index (articles) or release row (queries)
    window: int
    n_windows: int
    text: str
    est_tokens: int       # Qwen token count + 1 end-of-text token (the 8B convention)
    key8b: str            # content key under Qwen3-Embedding-8B (the 8B cache key)


def build_units(check: bool = True) -> dict[str, Any]:
    """Article units and the dev query units of every variant, checked against the 8B files.

    Returns ``{"articles": [Item], "queries": {variant: [Item]}, "dev_rows": array}``; query items cover the dev
    release rows only, in release order.
    """
    inputs = qe.load_inputs()
    arts = [Item("articles", u.owner, u.window, u.n_windows, u.text, u.est_tokens, qe.content_key(u.text))
            for u in inputs["articles"]]
    if check:
        saved = [json.loads(line) for line in (QWEN8B / "article_units.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if [r["input_sha256"] for r in saved] != [a.key8b for a in arts]:
            raise RuntimeError("article units differ from the 8B article_units.jsonl")
        if [(r["candidate_index"], r["window"], r["n_windows"], r["est_tokens"]) for r in saved] != \
                [(a.owner, a.window, a.n_windows, a.est_tokens) for a in arts]:
            raise RuntimeError("article unit layout differs from the 8B article_units.jsonl")
        receipt = json.loads((QWEN8B / "receipt.json").read_text(encoding="utf-8"))
        for v in VARIANTS:
            keys = [qe.content_key(u.text) for u in inputs["queries"][v]]
            if qe.keys_digest(keys) != receipt["input_sha256"][f"queries_{v}"]:
                raise RuntimeError(f"query variant {v} differs from the 8B receipt digest")
    dev = assert_dev(ep.dev_rows())
    queries = {v: [Item(f"queries_{v}", int(r), 0, 1, inputs["queries"][v][int(r)].text,
                        inputs["queries"][v][int(r)].est_tokens, qe.content_key(inputs["queries"][v][int(r)].text))
                   for r in dev] for v in VARIANTS}
    return {"articles": arts, "queries": queries, "dev_rows": dev}


# --------------------------------------------------------------------------- vector store (re-exported)

VectorStore = store.VectorStore


# --------------------------------------------------------------------------- assembled vector files


def assemble(spec: ModelSpec, units: dict[str, Any], extra_manifest: dict[str, Any], variants: Sequence[str] = VARIANTS,
             key_of=None) -> dict[str, Any]:
    """Write unit-norm ``article_vectors.npy`` / ``article_index.npy`` / ``query_vectors_<v>.npy`` (dev rows) and
    ``manifest.json`` into ``spec.derived``. ``key_of(item) -> [store keys]`` maps an item to its stored vectors
    (several keys only when an item had to be split for the model); the default is one key per item."""
    key_of = key_of or (lambda it: [content_key(it.text, spec.model_id)])
    store = VectorStore(spec.store_dir, spec.dim)
    d = spec.derived
    d.mkdir(parents=True, exist_ok=True)
    art_keys, owners, unit_rows_meta = [], [], []
    for i, it in enumerate(units["articles"]):
        for j, k in enumerate(key_of(it)):
            art_keys.append(k)
            owners.append(it.owner)
            unit_rows_meta.append({"unit": i, "sub": j, "candidate_index": it.owner, "window": it.window,
                                   "n_windows": it.n_windows, "key8b": it.key8b, "key": k})
    np.save(d / "article_vectors.npy", qe.unit_rows(store.get(art_keys)))
    np.save(d / "article_index.npy", np.asarray(owners, dtype=np.int64))
    (d / "article_units.jsonl").write_text("".join(json.dumps(r) + "\n" for r in unit_rows_meta), encoding="utf-8")
    files = {"article_vectors": "article_vectors.npy", "article_index": "article_index.npy",
             "article_units": "article_units.jsonl"}
    for v in variants:
        items = units["queries"][v]
        keys = []
        for it in items:
            ks = key_of(it)
            if len(ks) != 1:
                raise RuntimeError("a query must map to exactly one vector")
            keys.append(ks[0])
        np.save(d / f"query_vectors_{v}.npy", qe.unit_rows(store.get(keys)))
        files[f"query_vectors_{v}"] = f"query_vectors_{v}.npy"
    rows = [int(it.owner) for it in units["queries"][variants[0]]]
    data = ep.release()
    write_json(d / "query_rows.json", {"release_rows": rows, "query_ids": [data.queries[r].query_id for r in rows]})
    files["query_rows"] = "query_rows.json"
    manifest = {"model": spec.model_id, "revision": spec.revision, "dim": spec.dim, "program": PROGRAM,
                "units": "exactly the Qwen3-Embedding-8B units of data/derived/embedding_push_qwen8b_v1 (article keys and "
                         "query digests checked); queries are dev release rows only (no lockbox text)",
                "documents": "plain article text, MaxP over windows", "query_variants": {v: qe.VARIANTS[v] for v in variants},
                "normalisation": "L2-normalised float32 (client side)",
                "counts": {"article_units": len(units["articles"]), "article_vectors": len(art_keys),
                           "dev_queries": len(rows), "store_vectors": len(store)},
                "input_sha256": {"articles_8b_units": qe.keys_digest([it.key8b for it in units["articles"]]),
                                 **{f"queries_{v}_8b_dev": qe.keys_digest([it.key8b for it in units["queries"][v]])
                                    for v in variants}},
                "files": {k: {"path": p, "sha256": ep.sha256_file(d / p)} for k, p in files.items()},
                "release_manifest_sha256": ep.sha256_file(ep.RELEASE / "manifest.json"),
                "split_sha256": ep.sha256_file(ep.SPLIT_PATH), "written_utc": utc(), **extra_manifest}
    write_json(d / "manifest.json", manifest)
    return manifest


def load_vectors(spec_or_dir: ModelSpec | Path, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(article vectors, article owner index, dev query vectors, their release rows) of an assembled model."""
    d = spec_or_dir.derived if isinstance(spec_or_dir, ModelSpec) else Path(spec_or_dir)
    A = np.load(d / "article_vectors.npy")
    owner = np.load(d / "article_index.npy")
    Q = np.load(d / f"query_vectors_{variant}.npy")
    rows = np.asarray(json.loads((d / "query_rows.json").read_text(encoding="utf-8"))["release_rows"], dtype=np.int64)
    return A, owner, Q, assert_dev(rows)


def wait_seconds(seconds: float, poll: float = 5.0) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        time.sleep(min(poll, max(0.0, end - time.monotonic())))


# --------------------------------------------------------------------------- export for the standalone CUDA-env encoder

UNITS_DIR = RUN / "units"
VALIDATION_SEED = 20261016


def export_units() -> dict[str, Any]:
    """Write the checked 8B units (dev queries only) to ``RUN/units/units.jsonl`` with a manifest and the fixed
    validation set (198 short + 2 long article units, 50 dev queries of variant c) for CPU-vs-GPU agreement."""
    units = build_units(check=True)
    UNITS_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"g": it.group, "o": it.owner, "w": it.window, "n": it.n_windows, "t": it.est_tokens, "k8": it.key8b,
             "text": it.text} for it in units["articles"] + [it for v in VARIANTS for it in units["queries"][v]]]
    path = UNITS_DIR / "units.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    rng = np.random.default_rng(VALIDATION_SEED)
    arts = units["articles"]
    short = [i for i, it in enumerate(arts) if it.est_tokens <= 2048]
    long_ = [i for i, it in enumerate(arts) if it.est_tokens > 6000]
    val_art = sorted(rng.choice(short, 198, replace=False).tolist() + rng.choice(long_, 2, replace=False).tolist())
    nq = len(units["queries"]["c"])
    val_q = sorted(rng.choice(nq, 50, replace=False).tolist())
    man = {"program": PROGRAM, "written_utc": utc(), "path": str(path.relative_to(ROOT)), "sha256": ep.sha256_file(path),
           "counts": {"articles": len(arts), **{f"queries_{v}": len(units["queries"][v]) for v in VARIANTS}},
           "articles_8b_digest": qe.keys_digest([it.key8b for it in arts]),
           "check": "article keys equal data/derived/embedding_push_qwen8b_v1/article_units.jsonl; query variants equal the "
                    "8B receipt digests (all release rows) before the dev subset was taken",
           "dev_only": True, "lockbox_rows_included": 0,
           "validation": {"article_positions": val_art, "query_c_positions": val_q, "seed": VALIDATION_SEED}}
    write_json(UNITS_DIR / "units_manifest.json", man)
    return man
