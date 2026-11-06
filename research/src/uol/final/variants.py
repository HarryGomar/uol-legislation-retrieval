"""Variant releases built on the run computer for the robustness populations that need their own refit.

Each variant is a development-only release folder under ``<work root>/variants/<name>/`` (release + SPLIT.json +
BUNDLE.json); the rest of the bundle (vectors, frozen copies, selections) is shared. A job runs in a variant with
``UOL_FINAL_VARIANT=<name>``: ``context.activate_bundle`` then reads that release and split and uses
``<work>/variants/<name>`` as its work folder, so the rebuilt sources, the pair table and the rankers of the variant
never mix with the main run.

* ``devt`` (DEV-T, RUN_PLAN v2 §2): groups ordered by their earliest publication date; the DEV groups after the 70 %
  date of DEV form fold 4 (scored), every group up to that date keeps its v11 fold if it is 0-3 and otherwise moves to
  fold (hash mod 4); every other late group joins fold 4 (scored, never evaluated). Fold 4 is therefore trained only
  on earlier groups, through the same machinery as the main run (sources rebuilt per fold, rankers refit).
* ``devb`` (DEV-B): the labels of the gap-review resolver fixes (``scripts/gaps/g4_patched.py`` with P1, P2, P3, P5, P6,
  P7 on development sources) replace the v11 positives; development sources the fixes recover (published before
  27 Jun 2025, not in v11) join as new rows with their masked facts (``CatalogueMasker``), a singleton group each,
  in the fold of ``spec.fold_of``. Their query vectors come from the int8 path; encoders that cannot encode them on
  the run computer leave them out of that encoder's source (documented in the variant's receipt).
* ``devn`` (DEV-N) needs no new release: ``devn.apply_text_override`` changes the S2 rows' facts in the process.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from uol.final import context, guard

TIME_SHARE = 0.70


def variant_root(name: str, work: Path | None = None) -> Path:
    prof = context.active()
    # inside a variant profile the work folder is <work>/variants/<variant>: its root is two levels up (21 Oct 2026:
    # prep__devb looked for <work>/variants/variants/devb/SPLIT.json)
    base = Path(work) if work else (prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work)
    return base / "variants" / name


def _write_release(src_release: Path, dst: Path, queries: list[dict], references: list[dict], folds: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name, rows in (("queries.jsonl", queries), ("references.jsonl", references)):
        with (dst / name).open("w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (dst / "folds.json").write_text(json.dumps(folds), encoding="utf-8")
    for name in ("target_types.jsonl",):
        shutil.copy2(src_release / name, dst / name)
    if (src_release / "candidates").exists() and not (dst / "candidates").exists():
        shutil.copytree(src_release / "candidates", dst / "candidates")
    man = json.loads((src_release / "manifest.json").read_text(encoding="utf-8"))

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()
    man["files"] = {n: sha(dst / n) for n in ("folds.json", "queries.jsonl", "references.jsonl", "target_types.jsonl")}
    man["candidates"]["articles"]["path"] = "candidates/articles.jsonl"
    (dst / "manifest.json").write_text(json.dumps(man, indent=1), encoding="utf-8")


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_devt(share: float = TIME_SHARE) -> dict[str, Any]:
    """Write the DEV-T variant release (fold 4 = the late groups)."""
    from uol.embedding_push import common as ep
    from uol.final import populations
    prof = context.active()
    d = ep.release()
    dev = guard.dev_rows_only(ep.dev_rows())
    dates = populations.publication_dates()
    first: dict[str, str] = {}
    for r in dev:
        dt = dates.get(int(r))
        if dt:
            c = d.queries[int(r)].component_id
            first[c] = min(dt, first.get(c, dt))
    spec_rows = ep.population_rows("P_spec", dev)
    dated = sorted(first[d.queries[int(r)].component_id] for r in spec_rows if d.queries[int(r)].component_id in first)
    cut = dated[int(np.floor(share * (len(dated) - 1)))]
    late = {c for c, dt in first.items() if dt > cut}
    undated = {d.queries[int(r)].component_id for r in dev} - set(first)
    src = prof.path("release")
    q = _jsonl(src / "queries.jsonl")
    folds = json.loads((src / "folds.json").read_text(encoding="utf-8"))
    for rec in q:
        c = rec["component_id"]
        if c in late:
            rec["fold"] = 4
        elif rec["fold"] == 4:
            rec["fold"] = int(hashlib.sha256(f"devt:{c}".encode()).hexdigest(), 16) % 4
        folds["components"][c] = rec["fold"]
    out = variant_root("devt")
    _write_release(src, out / "release", q, _jsonl(src / "references.jsonl"), folds)
    split = json.loads(prof.path("split").read_text(encoding="utf-8"))
    split["assignment"] = {r["query_id"]: f"dev_fold{r['fold']}" for r in q}
    split["variant"] = {"name": "devt", "cut_date": cut, "share": share, "late_groups": len(late), "undated_groups": len(undated),
                        "rule": "fold 4 = groups whose earliest publication date is after the cut; folds 0-3 earlier"}
    (out / "SPLIT.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    meta = {**prof.bundle_meta, "variant": "devt", "rows": len(q)}
    meta.pop("row_map", None)
    (out / "BUNDLE.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return split["variant"]


def masked_text(masker, raw: str) -> str:
    """The query text of a recovered row: CatalogueMasker.mask returns a record; its "text" is what the release stores."""
    return masker.mask(raw)["text"]


def build_devb(g4_output: Path) -> dict[str, Any]:
    """Write the DEV-B variant release from the pickle of ``g4_patched.py OUT.pkl P1,P2,P3,P5,P6,P7``."""
    import pickle
    from uol.embedding_push import common as ep
    from uol.facts_to_law import spec
    prof = context.active()
    res = pickle.loads(Path(g4_output).read_bytes())
    src_rows: dict[str, dict] = {}
    sp0 = prof.path("sources_jsonl")
    if sp0.exists():
        for line in sp0.read_text(encoding="utf-8").splitlines():
            if line.strip():
                s0 = json.loads(line)
                src_rows[s0.get("query_id")] = s0
    from uol.final import devb, ranges
    d = ep.release()
    ns = devb.bindfix_namespace()
    p8, covered = ranges.apply(res, src_rows, ns, d.candidate_ids)      # P8 article ranges
    rules_ab = devb.apply_rules_ab(res, src_rows, ns, covered)        # rules A + B on what is still missing
    src = prof.path("release")
    q = _jsonl(src / "queries.jsonl")
    refs = {r["query_id"]: r for r in _jsonl(src / "references.jsonl")}
    folds = json.loads((src / "folds.json").read_text(encoding="utf-8"))
    changed = added = 0
    for rec in q:
        r = res.get(rec["query_id"])
        if r is None or r["kind"] != "released_dev":
            continue
        pos = [p for p in r["positives"] if p["candidate_id"] in d.candidate_index]
        if pos:
            refs[rec["query_id"]]["positives"] = [{k: p[k] for k in ("candidate_id", "rule_ids", "section", "span", "surface_sha256")
                                                   if k in p} for p in pos]
            rec["resolution_status"] = r["status"]
            changed += 1
    recovered = []
    from uol.retrieval_benchmarks.benchmark_repair.masking import CatalogueMasker
    inst = _jsonl(src / "candidates" / "instruments.jsonl")
    masker = CatalogueMasker(inst)
    for qid, r in res.items():
        if r["kind"] != "excluded_dev" or r["status"] != "full" or not r["positives"]:
            continue
        s = src_rows.get(qid)
        if s is None or int(s["record_id"]) >= guard.CUT_RECORD_ID:
            continue
        text = masked_text(masker, s["raw_hechos"])
        comp = f"ftl-component:devb-{s['record_id']}"
        fold = spec.fold_of(comp, folds["salt"], folds["k"])
        folds["components"][comp] = fold
        q.append({"component_id": comp, "flags": r["flags"], "fold": fold, "instancia": s.get("instancia"),
                  "materias": s.get("materias") or [], "n_unresolved": len(r["unresolved"]), "query_id": qid,
                  "record_id": str(s["record_id"]), "resolution_status": r["status"], "source_split": "devb_recovered",
                  "stratum": r["stratum"], "text": text, "text_sha256": spec.sha256_text(text), "tipo": s.get("tipo"),
                  "year": s.get("year")})
        refs[qid] = {"query_id": qid, "positives": [{k: p[k] for k in ("candidate_id", "rule_ids", "section", "span",
                                                                         "surface_sha256") if k in p} for p in r["positives"]],
                     "unresolved": [], "ignorable": [], "out_of_scope": [], "version_flags": []}
        recovered.append(qid)
        added += 1
    out = variant_root("devb")
    _write_release(src, out / "release", q, [refs[x["query_id"]] for x in q], folds)
    split = json.loads(prof.path("split").read_text(encoding="utf-8"))
    split["assignment"] = {r["query_id"]: f"dev_fold{r['fold']}" for r in q}
    split["variant"] = {"name": "devb", "labels_changed": changed, "recovered_rows": added, "recovered": recovered,
                        "patches": "g4_patched P1,P2,P3,P5,P6,P7 + P8 article ranges + bindfix rules A and B", "rules_ab": rules_ab, "p8_ranges": p8}
    (out / "SPLIT.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    guard.check_release_dev_only([type("Q", (), {"record_id": x["record_id"], "component_id": x["component_id"]})() for x in q])
    meta = {**prof.bundle_meta, "variant": "devb", "rows": len(q)}
    meta.pop("row_map", None)
    (out / "BUNDLE.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return split["variant"]


DEVB_ENCODES = {
    # job key: (derived folder key, file written, group, prefix / instruction variant)
    "devb_q8_a": ("qwen8b", "query_vectors_a.npy", "queries_c", "a"),
    "devb_q8_b": ("qwen8b", "query_vectors_b.npy", "queries_c", "b"),
    "devb_q8_c": ("qwen8b", "query_vectors_c.npy", "queries_c", "c"),
}


def devb_new_rows() -> list[int]:
    """Rows of the DEV-B variant release that are not in the main bundle (the recovered sources)."""
    from uol.embedding_push import common as ep
    split = json.loads(Path(variant_root("devb") / "SPLIT.json").read_text(encoding="utf-8"))
    rec = set(split["variant"]["recovered"])
    return [i for i, q in enumerate(ep.release().queries) if q.query_id in rec]


def devb_encode_inputs() -> list[Path]:
    """Query-only int8 encodes of the recovered rows (variants a, b, c of qwen_endpoint.query_input); run in the
    devb variant so the rows are the variant's."""
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.final import gpu_prep
    d = ep.release()
    out = []
    for key, (_, _, group, variant) in DEVB_ENCODES.items():
        texts = {r: qe.query_input(variant, d.queries[r].text) for r in devb_new_rows()}
        out.append(gpu_prep.query_only(key, texts, group))
    return out


def devb_vectors() -> dict[str, Any]:
    """Variant copies of the Qwen3-8B query vectors with the recovered rows appended (release order of the variant).
    The frozen bge-m3 and e5-small query vectors get the recovered rows from ``devb_dense_queries`` (the original
    encoder code on the CPU, 22 Oct 2026). Sources whose recovered rows are not re-encodable here keep -inf for them
    (``rebuild.copy_frozen``): ft_e5_q, knn_casesim_e5, knn_casesim_e5_all, proto_casesim_e5 (GPU-trained per fold in
    the frozen run), colbert_bge (token-level pairs) and dense_bge_long (long-input CLS of colbert.py)."""
    from uol.embedding_push import common as ep
    prof = context.active()
    d = ep.release()
    root = variant_root("devb")
    dst = root / "derived" / "qwen8b"
    dst.mkdir(parents=True, exist_ok=True)
    base = Path(prof.data_dir) / context.BUNDLE_LAYOUT["qwen8b"]
    ids0 = json.loads((base / "query_ids.json").read_text(encoding="utf-8"))
    pos0 = {q: i for i, q in enumerate(ids0)}
    for f in ("article_vectors.npy", "article_index.npy"):
        if not (dst / f).exists():
            shutil.copy2(base / f, dst / f)
    report = {}
    for key, (_, fname, _, _) in DEVB_ENCODES.items():
        V0 = np.load(base / fname, mmap_mode="r")
        new = context.active().path("new_vectors") / key
        rows_new = json.loads((new / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
        Vn = np.load(next(new.glob("query_vectors_*.npy")))
        posn = {int(r): i for i, r in enumerate(rows_new)}
        out = np.zeros((len(d.queries), V0.shape[1]), dtype=np.float32)
        for i, q in enumerate(d.queries):
            out[i] = V0[pos0[q.query_id]] if q.query_id in pos0 else Vn[posn[i]]
        np.save(dst / fname, out)
        report[fname] = {"rows": len(out), "appended": len(rows_new)}
    (dst / "query_ids.json").write_text(json.dumps([q.query_id for q in d.queries]), encoding="utf-8")
    return report


# --------------------------------------------------------------------------- DEV-B: bge-m3 / e5-small query vectors

DENSE_CONTROLS, DENSE_MIN_COS = 50, 0.99
DENSE_FILES = {"bge": ("bge_m3_chunks_v1", "query_vectors.npy"), "e5": ("e5_small_queries_v11", "query_vectors.npy")}


def devb_dense_dir(root: Path | None = None) -> Path:
    return (root or variant_root("devb")) / "derived" / "facts_to_law_v2"


def devb_dense_queries(threads: int | None = None, encoders: dict | None = None) -> dict[str, Any]:
    """The frozen bge-m3 and e5-small query vectors for the DEV-B release (run in the devb variant): the recovered rows
    re-encoded with the code that made the frozen vectors, on the CPU, plus 50 unchanged development queries as a
    control against the frozen vectors (the job fails, writing no vectors, if any control is below cos .99). Writes
    variant copies with one row per DEV-B row (bundle rows by query id, recovered rows from the re-encode) that the devb
    profile reads (``context._redirect_devb_dense``), and ``dense_queries_receipt.json`` with every control cosine.

    Recipes (frozen receipts): bge-m3 5617a9f6 ``facts_to_law.bge_gpu.encode``: CLS + L2, fp32, the facts truncated at
    512 tokens (frozen run: GTX 1070, fp32); e5-small 614241f6 ``facts_to_law.dense`` October encoder: "query: "
    prefix, attention-mask mean + L2, fp32, 512 tokens, CPU (as frozen). ``encoders`` (tests) maps "bge" / "e5" to
    callables texts -> unit vectors."""
    import os
    import time
    from uol.embedding_push import common as ep
    prof = context.active()
    if prof.variant != "devb":
        raise RuntimeError("devb_dense_queries runs in the devb variant (UOL_FINAL_VARIANT=devb)")
    d = ep.release()
    base = Path(prof.data_dir) / context.BUNDLE_LAYOUT["derived_ftl"]
    ids0 = json.loads((Path(prof.data_dir) / context.BUNDLE_LAYOUT["qwen8b"] / "query_ids.json").read_text(encoding="utf-8"))
    pos0 = {q: i for i, q in enumerate(ids0)}
    new_rows = devb_new_rows()
    if any(d.queries[r].query_id in pos0 for r in new_rows):
        raise RuntimeError("a recovered DEV-B row is already a bundle row")
    rng = np.random.default_rng(20261021)
    others = [i for i, q in enumerate(d.queries) if q.query_id in pos0]
    control = sorted(int(x) for x in rng.choice(others, size=min(DENSE_CONTROLS, len(others)), replace=False))
    texts = {r: d.queries[r].text for r in new_rows + control}
    threads = int(threads or os.environ.get("UOL_FINAL_THREADS") or 8)
    encoders = encoders or _original_encoders(prof, threads)
    out_dir = devb_dense_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"recovered_rows": len(new_rows), "control_rows": control, "threads": threads,
                               "min_cos_required": DENSE_MIN_COS, "sets": {}}
    frozen_receipt = base / DENSE_FILES["bge"][0] / "receipt.json"         # the bge-m3 files the frozen run hashed
    files = (getattr(encoders.get("bge"), "identity", None) or {}).get("model_files_sha256")
    if files and frozen_receipt.exists():
        want = json.loads(frozen_receipt.read_text(encoding="utf-8")).get("model_files", {})
        receipt["bge_model_files_match_frozen_receipt"] = {n: files.get(n) == h for n, h in want.items()}
    vectors = {}
    newpos = {r: i for i, r in enumerate(new_rows)}
    for name, (sub, fname) in DENSE_FILES.items():
        t0 = time.perf_counter()
        rows = new_rows + control
        V = np.asarray(encoders[name]([texts[r] for r in rows]), dtype=np.float32)
        V0 = np.load(base / sub / fname, mmap_mode="r")
        cos = [float(V[len(new_rows) + j] @ np.asarray(V0[pos0[d.queries[r].query_id]], dtype=np.float32))
               for j, r in enumerate(control)]
        receipt["sets"][name] = {"file": f"{sub}/{fname}", "encode_seconds": round(time.perf_counter() - t0, 1),
                                 "control_cos": [round(c, 6) for c in cos], "control_cos_min": min(cos),
                                 "control_cos_mean": float(np.mean(cos)),
                                 "identity": getattr(encoders[name], "identity", None)}
        if min(cos) < DENSE_MIN_COS:
            receipt["failed"] = f"{name}: control cosine {min(cos):.4f} < {DENSE_MIN_COS}"
            (out_dir / "dense_queries_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
            raise RuntimeError(f"DEV-B {name} re-encode fails the fidelity control: {receipt['failed']} (no vectors written)")
        M = np.zeros((len(d.queries), V0.shape[1]), dtype=np.float32)
        for i, q in enumerate(d.queries):
            M[i] = V0[pos0[q.query_id]] if q.query_id in pos0 else V[newpos[i]]
        vectors[name] = M
    for name, (sub, fname) in DENSE_FILES.items():           # written only after both sets passed their controls
        dst = out_dir / sub
        dst.mkdir(parents=True, exist_ok=True)
        if name == "bge":                                    # base.stage_dense reads the chunks beside the queries
            for f in ("chunk_vectors.npy", "units.jsonl", "receipt.json"):
                if (base / sub / f).exists() and not (dst / f).exists():
                    try:
                        os.link(base / sub / f, dst / f)
                    except OSError:
                        shutil.copy2(base / sub / f, dst / f)
        np.save(dst / fname, vectors[name])
        receipt["sets"][name]["rows"] = int(len(vectors[name]))
        receipt["sets"][name]["sha256"] = hashlib.sha256((dst / fname).read_bytes()).hexdigest()
    (out_dir / "query_ids.json").write_text(json.dumps([q.query_id for q in d.queries]), encoding="utf-8")
    (out_dir / "dense_queries_receipt.json").write_text(json.dumps(receipt, indent=1), encoding="utf-8")
    return {k: v for k, v in receipt.items() if k != "control_rows"}


def _original_encoders(prof, threads: int) -> dict:
    """The original encoder code over the pinned snapshots of the ``models`` job (``<work>/models``)."""
    from uol.final import rebuild
    from uol.facts_to_law import dense
    from uol.orchestration.embedding_usefulness import encoder as october_encoder
    models = (prof.work.parents[1] if prof.work.parent.name == "variants" else prof.work) / "models"
    e5_dir = models / rebuild.MODELS["E5_DIR"][2]
    bge_dir = models / rebuild.MODELS["BGE_DIR"][2]
    dense.configure_runtime(threads)
    binding = {**dense.MODEL_BINDING, "path": str(e5_dir),
               "files": {n: {**r, "path": str(e5_dir / n)} for n, r in dense.MODEL_BINDING["files"].items()}}
    e5 = october_encoder.OfflineEncoder(binding, device="cpu", check=dense._CHECK)   # verifies every file's sha256

    def e5_fn(texts):
        return dense.encode_texts(texts, kind="query", batch_size=8, threads=threads, encoder=e5)
    e5_fn.identity = {"code": "facts_to_law.dense.encode_texts (October OfflineEncoder)", "model_dir": str(e5_dir),
                      "revision": dense.MODEL_REVISION, "device": "cpu", "dtype": "float32"}
    from transformers import AutoModel, AutoTokenizer
    from uol.facts_to_law import bge_gpu
    tok = AutoTokenizer.from_pretrained(bge_dir, local_files_only=True)
    model = AutoModel.from_pretrained(bge_dir, local_files_only=True).to("cpu").eval()
    files = {n: bge_gpu.sha256(bge_dir / n) for n in bge_gpu.MODEL_FILES if (bge_dir / n).exists()}

    def bge_fn(texts):
        return bge_gpu.encode(list(texts), tok, model, "cpu", 512, 16384)
    bge_fn.identity = {"code": "facts_to_law.bge_gpu.encode (CLS + L2, queries at 512 tokens)", "model_dir": str(bge_dir),
                       "revision": rebuild.MODELS["BGE_DIR"][1], "model_files_sha256": files, "device": "cpu",
                       "dtype": "float32"}
    return {"bge": bge_fn, "e5": e5_fn}
