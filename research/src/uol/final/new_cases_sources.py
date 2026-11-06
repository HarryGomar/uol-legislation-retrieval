"""First stage of the all-development group for the new cases (E13, D161, D162): query vectors and the 50 sources.

The new cases are appended to the v11 release (``release_ext``: v11 rows first and unchanged, the new rows after them
with role ``lockbox``), so the frozen embedding-push code scores them as round 2 scored its first look (``c_env``):
with the memories, priors, prototypes, selections and adapters of the all-development group (training folds 0-4).
``timing_sources.FirstStage("lockbox")`` computes every source of one row from the stored query vectors extended
with the new rows' vectors, its evaluation rows restricted to the new rows (``eval_only``): no original test-set row
is scored, read or compared anywhere in E13 (D97; the guard's rule).

Query encoders of the new rows:
* e5-small and the word / character LSA projections: ``timing.CpuEncoders`` (round 2's CPU recipe);
* the fine-tuned e5 query tower (ft_e5_q) and the case-similarity encoder (model B, epochs 5 and 7) of the
  all-development group: their frozen weights (``rebuild_*_lockbox_*``, 14 Oct) on this PC's CPU, exactly as
  ``round3.remote_ep`` encoded the original lockbox rows;
* bge-m3 (dense and ColBERT query tokens), Qwen3-8B int8 and the ColBERT pair scores: the run computer
  (``gpu_jobs/new_encode.py``), taken in by ``ingest``.

Controls, development rows only (the 20 dry-run cases, DEV fold 0): e5 and LSA against the stored query vectors; the
case-similarity weights against the frozen all-development arrays, whose 4,812 development rows the same weights
encoded on the GPU; ft_e5_q (no development-row output of that tower was kept) by a repeated encode and the sha256 of
its weights. The fold-0 first stage on the same cases reproduces the frozen source files (``dryrun_sources``).
"""
from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final.new_cases_fit import ROOT, log, sha256, write_json

V11 = ROOT / "data" / "releases" / "uol_c01_statutes_20261004_v11_stratified"
SPLIT = ROOT / "research" / "state" / "embedding_push_20261004" / "SPLIT.json"
N_V11 = 5980
SMOKE_N = 3
GPU_JOBS = ROOT / "experiments" / "runs" / "round3_best_case_20261013" / "gpu_jobs"
FT_STATE = GPU_JOBS / "rebuild_ft_e5_lockbox_20261014T031915Z" / "remote" / "outputs" / "state_lockbox.pt"
CS_STATE = {e: GPU_JOBS / "rebuild_cs_e5_lockbox_B_20261014T040818Z" / "remote" / "outputs" / f"state_lockbox_B_epoch{e}.pt"
            for e in (5, 7)}
CPU_THREADS = 8
FLOORS = {"e5": 0.99999, "lsa": 0.99999, "clsa": 0.99999, "casesim_e5": 0.99999, "casesim_e7": 0.99999}
VECTOR_KEYS = ("e5", "lsa", "clsa", "ft", "casesim_e5", "casesim_e7")
RUNB_KEYS = ("bge", "Qc")


# --------------------------------------------------------------------------------------------- releases


def _load(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    with Path(path).open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def dev_controls() -> np.ndarray:
    """The 20 dry-run development cases (DEV fold 0, the first score rows) as rows of the active local release."""
    from uol.final import new_cases_fit as NF
    return NF.dry_rows(NF.dev_population())


def extend(queries_new: list[dict], refs_new: list[dict], dates: dict[str, str | None], out: Path, label: str,
           origin: dict[str, Any]) -> dict[str, Any]:
    """v11 copied, the new rows appended with role ``lockbox`` (``round2.c_release.extend`` for one set; it refuses
    an existing folder instead of removing it). ``new_rows.json`` lists the new rows with their publication dates."""
    from uol.facts_to_law.spec import sha256_file, write_json as wj
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists; the extended release is built once")
    out.mkdir(parents=True)
    queries = _load(V11 / "queries.jsonl")
    refs = _load(V11 / "references.jsonl")
    folds = json.loads((V11 / "folds.json").read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    new_sets, rows = {}, []
    for q in queries_new:
        if q["query_id"] in split["assignment"]:
            raise ValueError(f"{q['query_id']} is already a v11 row")
        rows.append({"row": len(queries), "query_id": q["query_id"], "publication_date": dates.get(q["query_id"])})
        queries.append(q)
        new_sets[q["query_id"]] = label
        folds["components"].setdefault(q["component_id"], q["fold"])
        split["assignment"][q["query_id"]] = "lockbox"
    if sorted(r["query_id"] for r in refs_new) != sorted(new_sets):
        raise ValueError("the new references do not match the new queries")
    refs += refs_new
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "references.jsonl", refs)
    wj(out / "folds.json", folds)
    shutil.copyfile(V11 / "target_types.jsonl", out / "target_types.jsonl")
    manifest = json.loads((V11 / "manifest.json").read_text(encoding="utf-8"))
    manifest["release_id"] = f"new_cases_20261026_{label}_ext"
    manifest["extended_from"] = {"path": str(V11), "manifest_sha256": sha256_file(V11 / "manifest.json"), **origin}
    manifest["files"] = {n: sha256_file(out / n) for n in ("queries.jsonl", "references.jsonl", "target_types.jsonl", "folds.json")}
    wj(out / "manifest.json", manifest)
    split["new_sets"] = new_sets
    split["note"] = ("E13 extension: v11 assignment unchanged; the new rows carry role 'lockbox' (models trained on all "
                     "development rows); the original lockbox rows are never evaluation rows of E13 (eval_only)")
    wj(out / "SPLIT_ext.json", split)
    wj(out / "new_rows.json", {"label": label, "rows": rows})
    return {"rows": len(queries), "new_rows": len(rows), "manifest_sha256": sha256(out / "manifest.json")}


_EXT: dict[str, Any] = {}


def activate_ext(ext: Path, logs: Path) -> None:
    """The local profile with ``embedding_push.common`` pointed at the extended release (``round2.c_env.activate``;
    the frozen modules' logs go to ``logs``, so nothing under the frozen run folders is written)."""
    import importlib
    from uol.embedding_push import common
    from uol.final import context
    if _EXT.get("on"):
        if _EXT["on"] != str(ext):
            raise RuntimeError("another extended release is active in this process")
        return
    context.activate_local("v11")

    def logger(module: str):
        def f(message="", *args, **kwargs) -> None:
            text = message if isinstance(message, str) else json.dumps(message, default=str, ensure_ascii=False)
            logs.mkdir(parents=True, exist_ok=True)
            with (logs / f"{module}.log").open("a", encoding="utf-8", newline="\n") as fh:
                fh.write(text + "\n")
        return f
    for name in ("base", "memory_proto", "case_sim", "qwen_endpoint", "colbert", "finetune_dense", "fusion_ltr"):
        try:
            mod = importlib.import_module(f"uol.embedding_push.{name}")
        except ImportError:
            continue
        for fn in ("log", "flog", "search_log", "record_stage", "reset_search_log"):
            if hasattr(mod, fn):
                setattr(mod, fn, logger(f"{name}.{fn}"))
    common.RELEASE = Path(ext)
    common.SPLIT_PATH = Path(ext) / "SPLIT_ext.json"
    for fn in (common.release, common.tie_order, common.roles):
        fn.cache_clear()
    from uol.embedding_push import memory_proto as mp
    for fn in (mp.texts, mp.components, mp.query_ids, mp.token_counts):
        fn.cache_clear()
    mp._VEC.clear()
    d = common.release()
    v11_ids = [q["query_id"] for q in _load(V11 / "queries.jsonl")]
    if [q.query_id for q in d.queries[:N_V11]] != v11_ids:
        raise AssertionError("the extended release must keep the v11 rows first, in v11 order")
    _EXT["on"] = str(ext)


def new_rows(ext: Path) -> list[dict]:
    return json.loads((Path(ext) / "new_rows.json").read_text(encoding="utf-8"))["rows"]


# --------------------------------------------------------------------------------------------- query encoders


def ft_vectors(rows: Sequence[int]) -> np.ndarray:
    """The all-development ft_e5_q tower (frozen weights) on this CPU (``round3.remote_ep.patch_ft``)."""
    import torch
    from uol.embedding_push import finetune_dense as fd
    from uol.round3 import remote_ep
    cfg = fd.Config(**remote_ep.FT_CFG)
    if cfg.name != "e5-8b32e14f":
        raise RuntimeError(f"ft_e5_q configuration {cfg.name} is not the frozen e5-8b32e14f")
    model, _ = fd.build_tower(cfg, torch.device("cpu"))
    model.load_state_dict(torch.load(FT_STATE, map_location="cpu"))
    tokens = fd.tokenize_queries(cfg.kind, cfg.max_len)
    return np.asarray(fd.encode_rows(model, cfg.kind, tokens, np.asarray(rows, dtype=np.int64), torch.device("cpu")),
                      dtype=np.float32)


def casesim_vectors(rows: Sequence[int], epoch: int) -> np.ndarray:
    """The all-development case-similarity encoder (model B, frozen weights of ``epoch``) on this CPU
    (``round3.remote_ep.patch_cs``)."""
    import torch
    from uol.embedding_push import case_sim as cs, common
    frozen = common.RUN / "memory_proto" / "casesim" / "e5_lockbox" / "B" / "train.json"
    recipe = json.loads(frozen.read_text(encoding="utf-8"))["recipe"]
    enc = cs.CaseEncoder(recipe["kind"], device="cpu", lora_rank=recipe["lora_rank"],
                         freeze_embeddings=recipe["freeze_embeddings"], max_len=recipe["max_len"])
    enc.model.load_state_dict(torch.load(CS_STATE[epoch], map_location="cpu"))
    texts = cs.mp.texts()
    return np.asarray(enc.encode([texts[int(r)] for r in rows]), dtype=np.float32)


def cpu_vectors(rows: Sequence[int]) -> dict[str, np.ndarray]:
    """e5-small query vectors and the LSA / character-LSA projections (refitted on the development texts)."""
    from uol.embedding_push import common
    from uol.final import cost, timing
    cost.set_threads(CPU_THREADS)
    enc = timing.CpuEncoders(CPU_THREADS, lsa=True, e5=True)
    texts = [common.release().queries[int(r)].text for r in rows]
    out = {"e5": np.stack([np.asarray(enc.encode_e5(t), dtype=np.float32) for t in texts])}
    for name in ("lsa", "clsa"):
        out[name] = np.stack([enc.transform(name, t) for t in texts]).astype(np.float32)
    return out


def encode_all(rows: Sequence[int]) -> dict[str, np.ndarray]:
    t0 = time.perf_counter()
    out = cpu_vectors(rows)
    out["ft"] = ft_vectors(rows)
    for e in CS_STATE:
        out[f"casesim_e{e}"] = casesim_vectors(rows, e)
    log(f"query encoders: {len(rows)} rows in {time.perf_counter() - t0:.0f}s")
    return out


def stored_vectors(rows: Sequence[int]) -> dict[str, np.ndarray]:
    """Stored query-side vectors of v11 development rows: e5, LSA, character LSA, and the all-development
    case-similarity arrays (development rows encoded on the GPU with the same weights)."""
    from uol.embedding_push import common, memory_proto as mp
    from uol.final import guard
    from uol.final import timing_sources as TS
    r = guard.dev_rows_only(rows, "encoder controls")
    out = {"e5": np.load(common.DERIVED / "e5_small_queries_v11" / "query_vectors.npy", mmap_mode="r")[r],
           "lsa": mp._unit(TS.load_array(TS.staged_path("memory_proto/lsa_dev_v1.npy"))[r]),
           "clsa": mp._unit(TS.load_array(TS.staged_path("memory_proto/clsa_dev_v1.npy"))[r])}
    for e in CS_STATE:
        out[f"casesim_e{e}"] = np.load(common.RUN / "memory_proto" / "casesim" / "e5_lockbox" / "B" / f"epoch{e}.npy",
                                       mmap_mode="r")[r]
    return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}


def fidelity(got: dict[str, np.ndarray], ref: dict[str, np.ndarray]) -> dict[str, Any]:
    out = {}
    for k in ref:
        a = got[k] / np.maximum(np.linalg.norm(got[k], axis=1, keepdims=True), 1e-12)
        b = ref[k] / np.maximum(np.linalg.norm(ref[k], axis=1, keepdims=True), 1e-12)
        out[k] = {"cos_min": float((a * b).sum(1).min()), "max_abs_diff": float(np.abs(a - b).max())}
    return out


def encoder_controls(got: dict[str, np.ndarray], rows: Sequence[int],
                     repeat: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, Any]:
    """``got``: ``encode_all`` on the development control ``rows``. Fidelity floors per encoder; ft_e5_q weights hashed
    and its encode repeated on the same input, bit-identical: ``repeat`` = (every row of the first encode, its ft
    vectors) when the controls were encoded together with other rows (``encode_rows`` pads each length-sorted batch to
    its longest text, so the batch composition moves the floats slightly), else ``rows`` themselves. When the controls
    were encoded with other rows, their alone-encode difference is reported (not a check)."""
    fid = fidelity(got, stored_vectors(rows))
    fails = [f"encoder {k}: cos_min {v['cos_min']:.7f} < {FLOORS[k]}" for k, v in fid.items() if v["cos_min"] < FLOORS[k]]
    alone = ft_vectors(rows)
    rep_rows, first = repeat if repeat is not None else (np.asarray(rows), got["ft"])
    again = ft_vectors(rep_rows) if repeat is not None else alone
    ft_same = bool(np.array_equal(again, first))
    if not ft_same:
        fails.append("ft_e5_q: a repeated encode of the same rows differs")
    out = {"rows": [int(r) for r in rows], "fidelity": fid, "floors": FLOORS, "ft_repeat_identical": ft_same,
           "ft_repeat_rows": int(len(rep_rows)),
           "ft_state_sha256": sha256(FT_STATE), "casesim_state_sha256": {e: sha256(p) for e, p in CS_STATE.items()},
           "fails": fails}
    if repeat is not None:
        a = alone / np.maximum(np.linalg.norm(alone, axis=1, keepdims=True), 1e-12)
        b = got["ft"] / np.maximum(np.linalg.norm(got["ft"], axis=1, keepdims=True), 1e-12)
        out["ft_controls_alone_vs_in_batch"] = {"max_abs_diff": float(np.abs(alone - got["ft"]).max()),
                                                "cos_min": float((a * b).sum(1).min())}
    return out


# --------------------------------------------------------------------------------------------- the first stage


def first_stage(role: str, colbert: str, *, eval_only: Sequence[int] | None = None,
                append: dict[str, np.ndarray] | None = None, ft: tuple[np.ndarray, np.ndarray] | None = None,
                pairs: dict[int, dict] | None = None):
    from uol.final import cost
    from uol.final import timing_sources as TS
    cost.set_threads(CPU_THREADS)
    fs = TS.FirstStage(role, sources=list(TS.FULL_ORDER) + list(TS.PAIR_SOURCES), colbert=colbert)
    fs.eval_only = None if eval_only is None else np.asarray(eval_only, dtype=np.int64)
    fs.append = dict(append or {})
    fs.append_ft = ft
    fs.given_pairs = dict(pairs or {})
    fs.load()
    return fs


def compare_case(fs, row: int) -> dict[str, Any]:
    """Every source of ``row`` against its frozen file (top 200 identical or equivalent, max |diff|; ColBERT: pool)."""
    from uol.final import timing_sources as TS
    src = fs.case(int(row))
    return {sid: (TS.compare_pairs(int(row), val) if sid == "colbert_bge" else TS.compare_row(sid, int(row), val))
            for sid, val in src.items()}


def gate_sources(per_row: dict[int, dict[str, Any]]) -> dict[str, Any]:
    def ok(v: dict) -> bool:
        return bool(v.get("top200_equivalent", v.get("pool_identical", False)))
    bad = sorted({s for r in per_row.values() for s, v in r.items() if not ok(v)})
    mx: dict[str, float] = {}
    for r in per_row.values():
        for s, v in r.items():
            if v.get("max_abs_diff") is not None:
                mx[s] = max(mx.get(s, 0.0), float(v["max_abs_diff"]))
    ident = float(np.mean([all(v.get("top200_identical", v.get("pool_identical", False)) for v in r.values())
                           for r in per_row.values()])) if per_row else None
    return {"rows": len(per_row), "sources": len(next(iter(per_row.values()))) if per_row else 0,
            "all_equivalent": not bad, "differing": bad, "share_rows_all_identical": ident, "max_abs_diff": mx}


def dryrun_sources(out: Path) -> dict[str, Any]:
    """Before the freeze (v11 release, local profile): the query encoders on the 20 development control cases
    against their stored vectors, and the fold-0 first stage on the same cases against the frozen source files."""
    from uol.final import context
    out = Path(out)
    context.activate_local("v11")
    rows = dev_controls()
    t0 = time.perf_counter()
    enc = encoder_controls(encode_all(rows), rows)
    res: dict[str, Any] = {"rows": [int(r) for r in rows], "encoders": enc, "fails": list(enc["fails"])}
    log(f"dry run sources: encoders {json.dumps(enc['fidelity'])} ft repeat {enc['ft_repeat_identical']}",
        out / "dryrun.log")
    fs = first_stage("dev_fold0", "frozen")
    per = {int(r): compare_case(fs, int(r)) for r in rows}
    res["dev_fold0"] = {**gate_sources(per), "per_row": per, "load_seconds": fs.load_seconds}
    log(f"dry run sources: fold 0 {json.dumps({k: v for k, v in res['dev_fold0'].items() if k not in ('per_row', 'load_seconds')})}",
        out / "dryrun.log")
    if not res["dev_fold0"]["all_equivalent"]:
        res["fails"].append(f"dev_fold0: sources differ from the frozen files: {res['dev_fold0']['differing']}")
    res["pass"] = not res["fails"]
    res["seconds"] = round(time.perf_counter() - t0, 1)
    write_json(out / "dryrun_sources.json", res)
    return res


# --------------------------------------------------------------------------------------------- new rows


def encode_local(run: Path) -> dict[str, Any]:
    """The new rows' CPU query vectors (e5, LSA, character LSA, ft_e5_q, case similarity epochs 5 and 7), with the
    development controls encoded in the same process against their stored vectors."""
    run = Path(run)
    ext = run / "release_ext"
    out = run / "vectors"
    if (out / "local_manifest.json").exists():
        raise FileExistsError(f"{out / 'local_manifest.json'} exists: the local query vectors are encoded once")
    from uol.final import context
    context.activate_local("v11")
    ctl = dev_controls()
    activate_ext(ext, run / "logs" / "embedding_push")
    rows = np.asarray([r["row"] for r in new_rows(ext)], dtype=np.int64)
    out.mkdir(parents=True, exist_ok=True)
    got = encode_all(np.concatenate([rows, ctl]))
    n = len(rows)
    files = {}
    for k in VECTOR_KEYS:
        np.save(out / f"new_{k}.npy", got[k][:n])
        files[f"new_{k}.npy"] = sha256(out / f"new_{k}.npy")
    controls = encoder_controls({k: v[n:] for k, v in got.items()}, ctl,
                                repeat=(np.concatenate([rows, ctl]), got["ft"]))
    man = {"rows": [int(r) for r in rows], "n_new": int(n), "files": files, "controls": controls,
           "pass": not controls["fails"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "local_manifest.json", man)
    if controls["fails"]:
        raise RuntimeError(f"encoder controls failed: {controls['fails']}")
    return man


def load_vectors(run: Path) -> tuple[dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
    """The new rows' query vectors and ColBERT pairs, each file checked against its manifest."""
    vec = Path(run) / "vectors"
    loc = json.loads((vec / "local_manifest.json").read_text(encoding="utf-8"))
    rb = json.loads((vec / "runb_manifest.json").read_text(encoding="utf-8"))
    for name, h in {**loc["files"], **rb["files"]}.items():
        if sha256(vec / name) != h:
            raise RuntimeError(f"vectors/{name}: sha256 differs from its manifest")
    if loc["rows"] != rb["rows"]:
        raise RuntimeError("the local and run-computer vectors are not for the same rows")
    arr = {k: np.load(vec / f"new_{k}.npy") for k in VECTOR_KEYS + RUNB_KEYS}
    for k, v in arr.items():
        if len(v) != len(loc["rows"]):
            raise ValueError(f"new_{k}.npy has {len(v)} rows for {len(loc['rows'])} new rows")
    z = np.load(vec / "new_colbert_pairs.npz")
    pairs = {int(r): {"cidx": z["cidx"][z["qidx"] == r], "score": z["score"][z["qidx"] == r]} for r in loc["rows"]}
    return arr, pairs


def stored_qc_v11() -> None:
    """The stored Qwen query vectors (v11 rows) cached for ``qwen_endpoint.query_vectors("c")``. Its own row-order
    check compares them with the whole active release, which in the extended release also holds the new rows; here
    they are checked against the v11 rows, which come first. ``FirstStage`` then appends the new rows' vectors."""
    from uol.embedding_push import common, qwen_endpoint as qe
    ids = json.loads((qe.DERIVED / "query_ids.json").read_text(encoding="utf-8"))
    if ids != [q.query_id for q in common.release().queries[:N_V11]]:
        raise ValueError("stored query vectors do not follow the v11 row order")
    qe._VEC["Qc"] = np.load(qe.DERIVED / "query_vectors_c.npy")
    if len(qe._VEC["Qc"]) != N_V11:
        raise ValueError(f"stored query vectors have {len(qe._VEC['Qc'])} rows for {N_V11} v11 rows")


def sources(run: Path) -> dict[str, Any]:
    """The 50 sources and the ColBERT pairs of the all-development group for every new row (evaluation rows = the
    new rows only). ``sources/<sid>.npy`` [n_new, N] in ``new_rows.json`` order; ``colbert_bge_pairs.npz``."""
    run = Path(run)
    ext = run / "release_ext"
    out = run / "sources"
    if (out / "manifest.json").exists():
        raise FileExistsError(f"{out / 'manifest.json'} exists: the new rows' sources are computed once")
    activate_ext(ext, run / "logs" / "embedding_push")
    nr = new_rows(ext)
    rows = np.asarray([r["row"] for r in nr], dtype=np.int64)
    arr, pairs = load_vectors(run)
    append = {k: arr[k] for k in ("bge", "e5", "lsa", "clsa", "Qc", "casesim_e5", "casesim_e7")}
    stored_qc_v11()
    t0 = time.perf_counter()
    fs = first_stage("lockbox", "given", eval_only=rows, append=append, ft=(rows, arr["ft"]), pairs=pairs)
    out.mkdir(parents=True, exist_ok=True)
    blocks: dict[str, np.ndarray] = {}
    cp: dict[str, list] = {"qidx": [], "cidx": [], "score": []}
    finite: dict[str, int] = {}
    for i, r in enumerate(rows):
        src = fs.case(int(r))
        for sid, val in src.items():
            if sid == "colbert_bge":
                cp["qidx"].append(np.full(len(val["cidx"]), int(r), dtype=np.int32))
                cp["cidx"].append(np.asarray(val["cidx"], dtype=np.int32))
                cp["score"].append(np.asarray(val["score"], dtype=np.float32))
            else:
                v = np.asarray(val, dtype=np.float32)
                blocks.setdefault(sid, np.empty((len(rows), len(v)), dtype=np.float32))[i] = v
                finite[sid] = finite.get(sid, 0) + int(np.isfinite(v).sum())
        if (i + 1) % 25 == 0 or i + 1 == len(rows):
            log(f"new rows' sources: {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", run / "logs" / "sources.log")
    files = {}
    for sid, b in blocks.items():
        np.save(out / f"{sid}.npy", b)
        files[f"{sid}.npy"] = sha256(out / f"{sid}.npy")
    np.savez(out / "colbert_bge_pairs.npz", **{k: np.concatenate(v) for k, v in cp.items()})
    files["colbert_bge_pairs.npz"] = sha256(out / "colbert_bge_pairs.npz")
    from uol.embedding_push import common
    man = {"rows": [int(r) for r in rows], "query_ids": [r["query_id"] for r in nr],
           "group": "lockbox (training folds 0-4: every development row)", "eval_only": "the new rows",
           "candidates": len(common.release().candidate_ids),
           "candidate_ids_sha256": __import__("hashlib").sha256("\n".join(common.release().candidate_ids).encode()).hexdigest(),
           "files": files, "finite_values": finite, "load_seconds": fs.load_seconds,
           "seconds": round(time.perf_counter() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "manifest.json", man)
    return man


# --------------------------------------------------------------------------------------------- smoke run


def smoke_prepare(work: Path) -> dict[str, Any]:
    """Before the freeze: three stand-in new cases (copies of development cases under new ids, no references), an
    extended release in ``work``, their local vectors, and the run computer's part taken from the stored vectors and
    frozen pairs of the originals, so every post-freeze step runs end to end. Their scores mean nothing (the all-
    development memories hold their originals); only the path is exercised."""
    from uol.embedding_push import common, qwen_endpoint as qe
    from uol.final import context
    from uol.final import timing_sources as TS
    from uol.round3 import common as R3
    work = Path(work)
    context.activate_local("v11")
    rows = dev_controls()[:SMOKE_N]
    d = common.release()
    v11q = {q["query_id"]: q for q in _load(V11 / "queries.jsonl")}
    pubs = R3.publication_dates()
    qs, refs, dates = [], [], {}
    placeholder = d.candidate_ids[0]   # the release loader needs one resolved reference; no label is read
    for r in rows:
        q = dict(v11q[d.queries[int(r)].query_id])
        q["query_id"] = "SMOKE-" + q["query_id"]
        q["component_id"] = "smoke-component:" + q["query_id"]
        qs.append(q)
        refs.append({"query_id": q["query_id"], "positives": [{"candidate_id": placeholder, "rule_ids": ["smoke"],
                                                                 "section": "criterio", "span": [0, 0],
                                                                 "surface_sha256": ""}],
                     "unresolved": [], "out_of_scope": [],
                     "ignorable": [], "version_flags": [], "smoke_placeholder": True})
        dates[q["query_id"]] = pubs[int(r)]
    bge = np.load(common.DERIVED / "bge_m3_chunks_v1" / "query_vectors.npy", mmap_mode="r")[rows]
    qc = np.asarray(qe.query_vectors("c"), dtype=np.float32)[rows]
    fp = TS.frozen_pairs()
    ext = work / "release_ext"
    res = extend(qs, refs, dates, ext, "SMOKE", {"smoke_of_rows": [int(r) for r in rows]})
    nr = new_rows(ext)
    vec = work / "vectors"
    vec.mkdir(parents=True, exist_ok=True)
    np.save(vec / "new_bge.npy", np.asarray(bge, dtype=np.float32))
    np.save(vec / "new_Qc.npy", qc)
    qi, ci, sc = [], [], []
    for x, r in zip(nr, rows):
        m = fp["qidx"] == int(r)
        qi.append(np.full(int(m.sum()), x["row"], dtype=np.int32))
        ci.append(fp["cidx"][m].astype(np.int32))
        sc.append(fp["score"][m].astype(np.float32))
    np.savez(vec / "new_colbert_pairs.npz", qidx=np.concatenate(qi), cidx=np.concatenate(ci), score=np.concatenate(sc))
    write_json(vec / "runb_manifest.json", {"rows": [x["row"] for x in nr], "stand_in": "stored vectors and frozen pairs",
                                            "files": {n: sha256(vec / n) for n in ("new_bge.npy", "new_Qc.npy",
                                                                                   "new_colbert_pairs.npz")}})
    return {**res, "originals": [int(r) for r in rows]}


# --------------------------------------------------------------------------------------------- counts


def first_look_ids() -> set[str]:
    """Query ids of round 2's first look (released 4 Oct)."""
    p = ROOT / "experiments" / "runs" / "round2_20261004" / "C" / "prospective" / "release" / "queries.jsonl"
    return {q["query_id"] for q in _load(p)}


def counts(release: Path) -> dict[str, Any]:
    """Cases, groups, catalogue-citing cases (a resolved positive in the catalogue), first look vs second."""
    release = Path(release)
    meta = _load(release / "rows_meta.jsonl")
    refs = {r["query_id"]: r for r in _load(release / "references.jsonl")}
    man = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    first = first_look_ids()
    dated = [m["publication_date"] for m in meta if m["publication_date"]]

    def block(ms: list[dict]) -> dict[str, Any]:
        return {"cases": len(ms), "groups": len({m["component_id"] for m in ms}),
                "catalogue_citing": sum(1 for m in ms if refs[m["query_id"]]["positives"]),
                "P": sum(bool(m["P"]) for m in ms), "P_spec": sum(bool(m["P_spec"]) for m in ms),
                "P_spec_groups": len({m["component_id"] for m in ms if m["P_spec"]}),
                "overlap_v11": sum(1 for m in ms if m.get("overlap_v11"))}
    return {"source_rows": man["sources"]["rows"], "exclusions": man["exclusions"], "all": block(meta),
            "first_look": block([m for m in meta if m["query_id"] in first]),
            "second_look": block([m for m in meta if m["query_id"] not in first]),
            "first_look_ids_not_released_now": len(first - {m["query_id"] for m in meta}),
            "publication_dates": [min(dated), max(dated)] if dated else None,
            "by_month": dict(sorted(Counter((m["publication_date"] or "")[:7] for m in meta).items()))}
