"""Remote job: one original embedding-push GPU training step (ft_e5_q group job, or a case-sim e5 stage A/B).

Inputs: ``code.zip`` (src/uol/{embedding_push,facts_to_law}), ``data.zip`` (release with lockbox rows blanked, SPLIT,
frozen e5 article vectors), ``task.json``. Runs the ORIGINAL functions (``finetune_dense.cmd_job`` /
``case_sim.train_encoder``) in FP32. Outputs: the vectors those functions write, ``train.json``/stats, and, for the
lockbox group only, the trained weights (``state_*.pt``) so the authority PC encodes the lockbox rows itself (their texts
never travel). Extra packages go to a private target dir (pinned to the authority PC's GPU venv), never the shared venv.

    python ep_train.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

PINS = ["transformers==5.17.0", "tokenizers==0.23.2", "huggingface_hub==1.32.0"]
E5_REPO, E5_REV = "intfloat/multilingual-e5-small", "614241f622f53c4eeff9890bdc4f31cfecc418b3"
BGE_REPO, BGE_REV = "BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"
BGE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model",
             "colbert_linear.pt", "pytorch_model.bin")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ensure_deps(scratch: Path) -> Path:
    target = scratch / ("pydeps_" + hashlib.sha1(" ".join(PINS).encode()).hexdigest()[:10])
    if not (target / "OK").exists():
        log(f"installing {PINS} into {target}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--target", str(target), *PINS],
                           capture_output=True, text=True)
        (scratch / "pip_install.log").write_text(r.stdout + r.stderr, encoding="utf-8")
        if r.returncode != 0:
            raise RuntimeError("pip install failed; see pip_install.log")
        (target / "OK").write_text(json.dumps({"pins": PINS, "utc": time.time()}))
    sys.path.insert(0, str(target))
    return target


def ensure_root(scratch: Path, inputs: Path, root_key: str | None = None) -> Path:
    """Unpack the bundle once per (code, data) hash; later jobs may send only the key of an unpacked root."""
    if not (inputs / "code.zip").exists():
        root = scratch / f"ep_root_{root_key}"
        if not (root / "OK").exists():
            raise RuntimeError(f"bundle {root_key} not on this worker; resend with the zips")
        return root
    key = sha256(inputs / "code.zip")[:12] + "_" + sha256(inputs / "data.zip")[:12]
    if root_key and key != root_key:
        raise RuntimeError(f"bundle key mismatch {key} != {root_key}")
    root = scratch / f"ep_root_{key}"
    if not (root / "OK").exists():
        for z in ("data.zip", "code.zip"):
            with zipfile.ZipFile(inputs / z) as f:
                f.extractall(root)
        (root / "OK").write_text("ok")
    return root


def ensure_e5(scratch: Path, files: dict) -> Path:
    from huggingface_hub import hf_hub_download
    d = scratch / "e5_small_614241f6"
    d.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in files.items():
        p = d / name
        if not (p.exists() and sha256(p) == digest):
            src = Path(hf_hub_download(E5_REPO, name, revision=E5_REV, cache_dir=str(scratch / "hf")))
            p.write_bytes(src.read_bytes())
        if sha256(p) != digest:
            raise RuntimeError(f"e5 file hash mismatch: {name}")
    return d


def ensure_bge(scratch: Path) -> Path:
    from huggingface_hub import snapshot_download
    d = Path(snapshot_download(BGE_REPO, revision=BGE_REV, allow_patterns=list(BGE_FILES), cache_dir=str(scratch / "hf")))
    missing = [f for f in BGE_FILES if not (d / f).exists()]
    if missing:
        raise RuntimeError(f"bge-m3 files missing: {missing}")
    return d


def colbert_score(task: dict, inputs: Path, out: Path, scratch: Path, np, torch, common) -> None:
    """Encode the public article chunks (doc store kept in worker scratch), encode the DEV queries, splice in the
    lockbox query token vectors computed on the authority PC, and score every pool pair with the original functions."""
    from uol.embedding_push import colbert as cb
    work = scratch / f"colbert_{task['chunks_sha'][:12]}"
    work.mkdir(parents=True, exist_ok=True)
    cb.OUT, cb.DOCS, cb.QUERIES = work, work / "docs", work / "queries"
    for name in ("chunks.npz", "chunk_texts.jsonl"):
        if not (work / name).exists():
            (work / name).write_bytes((inputs / name).read_bytes())
    chunks = np.load(work / "chunks.npz")
    cb.DOCS.mkdir(exist_ok=True)
    if not (cb.DOCS / "plan.json").exists():                       # the doc half of stage_alloc (8 shards)
        n_docs = int(chunks["n_vec"].sum())
        mm = np.lib.format.open_memmap(cb.DOCS / "doc_tokens.npy", mode="w+", dtype=np.float16, shape=(n_docs, cb.DIM))
        del mm
        cum, cand = np.cumsum(chunks["n_vec"], dtype=np.int64), chunks["cand"]
        cuts = []
        for t in np.linspace(0, cum[-1], 9)[1:-1]:
            i = int(np.searchsorted(cum, t))
            while 0 < i < len(cand) and cand[i] == cand[i - 1]:
                i += 1
            cuts.append(min(i, len(cand)))
        bounds = [0, *sorted(set(cuts)), len(cand)]
        plan = [{"shard": s, "chunks": [int(bounds[s]), int(bounds[s + 1])],
                 "vectors": int(chunks["n_vec"][bounds[s]:bounds[s + 1]].sum())} for s in range(len(bounds) - 1)]
        (cb.DOCS / "progress.json").write_text(json.dumps({"done": {}}), encoding="utf-8")
        (cb.DOCS / "plan.json").write_text(json.dumps({"shards": plan, "token_budget": 8192}), encoding="utf-8")
    plan = json.loads((cb.DOCS / "plan.json").read_text(encoding="utf-8"))
    res = cb.stage_encode_docs(list(range(len(plan["shards"]))), "cuda")
    (out / "docs_progress.json").write_text(json.dumps(json.loads((cb.DOCS / "progress.json").read_text())), encoding="utf-8")
    # queries: dev rows encoded here; lockbox rows from the authority PC (vectors only)
    qw = np.load(inputs / "qweights.npz")
    offset, n_vec = qw["offset"], qw["n_vec"]
    qstore = np.zeros((int(n_vec.sum()), cb.DIM), dtype=np.float16)
    lb_rows = np.load(inputs / "lb_rows.npy")
    lb_vec = np.load(inputs / "lb_qtok.npy")
    pos = 0
    for r in lb_rows:
        qstore[offset[r]:offset[r] + n_vec[r]] = lb_vec[pos:pos + n_vec[r]]
        pos += int(n_vec[r])
    if pos != len(lb_vec):
        raise RuntimeError("lockbox token vectors do not match their offsets")
    data = common.release()
    lbset = set(int(x) for x in lb_rows)
    dev = [i for i in range(len(data.queries)) if i not in lbset]
    frozen = np.load(cb.FROZEN / "query_vectors.npy", mmap_mode="r")
    cos = np.full(len(dev), np.nan)

    def sink(i, cls, vec):
        r = dev[i]
        qstore[offset[r]:offset[r] + n_vec[r]] = vec
        cos[i] = float(cls @ frozen[r])
    enc = cb.Encoder("cuda")
    enc.encode([data.queries[r].text for r in dev], cb.QUERY_MAX_TOKENS, 8192, sink, lengths=n_vec[dev].astype(np.int64) + 1)
    enc.free()
    if np.nanmin(cos) < 0.999:
        raise RuntimeError(f"dev query CLS vectors do not reproduce the frozen vectors (min cos {np.nanmin(cos)})")
    # score (original score_query; weights precomputed on the authority PC with colbert.query_weightings)
    pool = np.load(inputs / "pool.npz")
    qidx, cidx = pool["qidx"], pool["cidx"]
    weights = qw["weights"]
    docs = np.load(cb.DOCS / "doc_tokens.npy", mmap_mode="r")
    index = cb.article_index(chunks, len(data.candidate_ids))
    n_q = int(qidx.max()) + 1
    ptr = np.searchsorted(qidx, np.arange(n_q + 1))
    scores = np.full((len(cb.VARIANTS), len(qidx)), np.nan, dtype=np.float32)
    t0 = time.time()
    for q in range(n_q):
        lo, hi = ptr[q], ptr[q + 1]
        span = slice(int(offset[q]), int(offset[q] + n_vec[q]))
        maxp, union = cb.score_query(qstore[span], weights[span], cidx[lo:hi], index, docs, 49152, "cuda")
        scores[0::2, lo:hi] = maxp.T
        scores[1::2, lo:hi] = union.T
        if (q + 1) % 500 == 0:
            log(f"score {q + 1}/{n_q} ({time.time() - t0:.0f}s)")
    if not np.isfinite(scores).all():
        raise RuntimeError("unscored pool pairs remain")
    np.save(out / "pair_scores.npy", scores)
    (out / "query_check.json").write_text(json.dumps({"dev_queries": len(dev), "cos_min": float(np.nanmin(cos)),
                                                     "cos_mean": float(np.nanmean(cos)), "lockbox_rows_from_authority_pc": len(lb_rows),
                                                     "score_seconds": round(time.time() - t0, 1)}), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    inputs, out = Path(a.inputs).resolve(), Path(a.out).resolve()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", "C:/uol_gpu/scratch"))
    scratch.mkdir(parents=True, exist_ok=True)
    task = json.loads((inputs / "task.json").read_text(encoding="utf-8"))
    ensure_deps(scratch)
    root = ensure_root(scratch, inputs, task.get("root_key"))
    sys.path.insert(0, str(root / "src"))
    os.environ.update({"TOKENIZERS_PARALLELISM": "false"})
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from uol.facts_to_law import crossfit, dense
    from uol.embedding_push import common

    e5 = ensure_e5(scratch, dense.MODEL_FILES)
    dense.MODEL_DIR = e5
    common.E5_DIR = e5
    # the blanked lockbox rows have no positives: load without hash verification and skip only those rows' check
    orig_check = crossfit._check_release

    def check(data, verify):
        keep = [i for i, q in enumerate(data.queries) if q.text]
        if len(keep) == len(data.queries):
            return orig_check(data, verify)
    crossfit._check_release = check
    common.release = functools.lru_cache(maxsize=1)(lambda: crossfit.load_release(common.RELEASE, verify=False))
    common.gpu_lock = lambda *a, **k: contextlib.nullcontext()
    env = {"torch": torch.__version__, "device": torch.cuda.get_device_name(0), "root": str(root),
           "transformers": __import__("transformers").__version__}
    log(f"env {env}; task {task}")
    lockbox = task["role"] == "lockbox"
    saved: list[str] = []

    if task["kind"] == "ft":
        from uol.embedding_push import finetune_dense as fd
        fd.E5_MODEL_DIR = e5
        cfg = fd.Config(**task["cfg"])
        if lockbox:
            orig_encode = fd.encode_rows

            def encode_and_save(model, kind, tokens, rows, device, *args, **kw):
                p = out / f"state_{task['role']}.pt"
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, p)
                saved.append(p.name)
                return orig_encode(model, kind, tokens, rows, device, *args, **kw)
            fd.encode_rows = encode_and_save
        fd.cmd_job(cfg, [task["role"]])
        src = fd.job_cache_path(cfg, task["role"])
        (out / src.name).write_bytes(src.read_bytes())
    elif task["kind"] == "cs":
        from uol.embedding_push import case_sim as cs
        recipe = cs.Recipe(**task["recipe"])
        fit, encode = cs.stage_rows(task["role"], task["stage"])
        lb_rows = set(np.flatnonzero(common.roles() == "lockbox").tolist())
        enc_rows = np.asarray([r for r in encode if int(r) not in lb_rows], dtype=np.int64)
        if lockbox:
            orig = cs.CaseEncoder.encode
            epochs = sorted(task["save_epochs"])

            def encode_and_save(self, texts, *args, **kw):
                e = epochs[len(saved)]
                p = out / f"state_{task['role']}_{task['stage']}_epoch{e}.pt"
                torch.save({k: v.detach().cpu() for k, v in self.model.state_dict().items()}, p)
                saved.append(p.name)
                return orig(self, texts, *args, **kw)
            cs.CaseEncoder.encode = encode_and_save
        res = cs.train_encoder(recipe, fit, enc_rows, out, task["save_epochs"], log=log)
        (out / "encoded_rows.json").write_text(json.dumps({"encoded": int(len(enc_rows)),
                                                         "lockbox_rows_skipped": int(len(encode) - len(enc_rows))}))
        log(f"train result {json.dumps(res)[:300]}")
    elif task["kind"] in ("colbert_long", "colbert_score"):
        from uol.embedding_push import colbert as cb
        common.BGE_DIR = ensure_bge(scratch)
        if task["kind"] == "colbert_long":
            cb.OUT = scratch / "colbert_long_work"
            cb.LONG = cb.OUT / "long"
            cb.stage_long_encode(device="cuda")
            for name in ("long_cls.npy", "long_articles.npy", "long_info.json"):
                (out / name).write_bytes((cb.LONG / name).read_bytes())
        else:
            colbert_score(task, inputs, out, scratch, np, torch, common)
    else:
        raise ValueError(task["kind"])
    (out / "env.json").write_text(json.dumps({**env, "saved_states": saved}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
