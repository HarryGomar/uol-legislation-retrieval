"""Run the GPU-trained embedding-push rebuild stages on the remote RTX 5080 (author decision D0.5).

The original orchestrators run here on the CPU (CUDA hidden), unchanged; only their GPU launchers are replaced:
``finetune_dense.run_job`` (cache miss) and ``case_sim.launch_train`` submit ``gpu_jobs/ep_train.py`` through
``gpu_remote`` and place the returned files exactly where the original subprocess would have written them. For the
lockbox group the returned weights encode the lockbox rows HERE on the CPU, so lockbox texts never leave this PC.

    python -m uol.round3.remote_ep ft      # ft_e5_q (frozen configuration e5-8b32e14f)
    python -m uol.round3.remote_ep cs      # case_sim run --kind e5
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the 1070 is not used (author, 2026-10-14)
os.environ.setdefault("UOL_GPU_TRANSPORT", "drive")
os.environ.setdefault("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")

import numpy as np

from uol.round3 import rebuild

ROOT = Path(__file__).resolve().parents[3]
ENTRY = ROOT / "src/uol/round3/gpu_jobs/ep_train.py"
FT_CFG = {"kind": "e5", "tau": 0.02, "lam": 0.0, "generic": "drop", "lr": 2e-05, "epochs": 3, "batch": 32, "warmup": 0.1,
          "weight_decay": 0.01, "clip": 1.0, "max_len": 512, "token_budget": 4096, "seed": 20261004, "lora_r": 0,
          "lora_alpha": 0.0, "lora_dropout": 0.05, "grad_ckpt": False, "train_embeddings": True, "shuffle_seed": 0}


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


SENT = ROOT / "experiments/runs/round3_best_case_20261013/remote_bundle/on_worker.json"


def _root_key(man: dict) -> str:
    from uol.round3.gpu_remote import sha256
    return sha256(Path(man["code"]["path"]))[:12] + "_" + sha256(Path(man["data"]["path"]))[:12]


def submit(job_id: str, task: dict, expected: list[str], extra_inputs: dict | None = None) -> Path:
    """Submit one job and return the folder holding its verified outputs (bundle zips only until the worker has them)."""
    from uol.round3 import gpu_remote as G
    from uol.round3 import remote_bundle
    man = remote_bundle.build()
    key = _root_key(man)
    on_worker = set(json.loads(SENT.read_text())) if SENT.exists() else set()
    task = {**task, "root_key": key}
    tdir = G.LOCAL_JOBS.parent / "gpu_job_tasks"
    tdir.mkdir(parents=True, exist_ok=True)
    tpath = tdir / f"{job_id}.json"
    tpath.write_text(json.dumps(task, indent=1), encoding="utf-8")
    t0 = time.time()
    rebuild.ledger({"kind": "remote_submit", "job_id": job_id, "utc": datetime.now(timezone.utc).isoformat(), "task": task,
                    "bundle": {k: v["digest"] for k, v in man.items() if isinstance(v, dict)}})
    inputs = {"task.json": tpath, **(extra_inputs or {})}
    if key not in on_worker:
        inputs.update({"code.zip": Path(man["code"]["path"]), "data.zip": Path(man["data"]["path"])})
    receipt = G.submit(job_id, ENTRY, inputs, expected_outputs=expected)
    SENT.write_text(json.dumps(sorted(on_worker | {key})))
    rebuild.ledger({"kind": "remote_done", "job_id": job_id, "gpu_seconds": receipt["seconds"],
                    "wall_seconds": round(time.time() - t0, 1), "device": receipt["env"].get("device"),
                    "outputs": {k: v["sha256"] for k, v in receipt["outputs"].items()}})
    return G.LOCAL_JOBS / job_id / "remote" / "outputs"


# ----------------------------------------------------------------------------- ft_e5_q

def patch_ft():
    from uol.embedding_push import finetune_dense as fd
    from uol.embedding_push import common
    orig = fd.run_job

    def run_job(cfg, role, *, eval_rows=None, max_steps=0, use_cache=True):
        path = fd.job_cache_path(cfg, role, smoke=bool(max_steps))
        if use_cache and not max_steps and path.exists():
            return orig(cfg, role, eval_rows=eval_rows, max_steps=max_steps, use_cache=use_cache)
        if eval_rows is not None or max_steps:
            raise RuntimeError("remote rebuild runs only plain group jobs")
        outd = submit(f"rebuild_ft_e5_{role}_{_ts()}", {"kind": "ft", "role": role, "cfg": cfg.asdict()}, [path.name])
        z = np.load(outd / path.name, allow_pickle=False)
        qvecs, rows, stats = z["qvecs"], z["rows"], json.loads(str(z["stats"]))
        stats["remote"] = {"device": "RTX 5080 (gpu_remote)", "decision": "D0.5"}
        if role == "lockbox":
            import torch
            model, _ = fd.build_tower(cfg, torch.device("cpu"))
            model.load_state_dict(torch.load(outd / "state_lockbox.pt", map_location="cpu"))
            tokens = fd.tokenize_queries(cfg.kind, cfg.max_len)
            qvecs = fd.encode_rows(model, cfg.kind, tokens, rows, torch.device("cpu"))
            stats["lockbox_rows_encoded_on"] = "authority PC CPU with the returned weights"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, qvecs=qvecs, rows=rows, stats=json.dumps(stats))
        os.replace(tmp, path)
        return qvecs, stats
    fd.run_job = run_job
    return fd


# ----------------------------------------------------------------------------- case-sim e5

def patch_cs():
    from uol.embedding_push import case_sim as cs
    from uol.embedding_push import common

    def launch_train(recipe, role, stage, out_dir, save_epochs=None):
        epochs = list(save_epochs) if save_epochs else list(range(1, recipe.epochs + 1))
        expected = [f"epoch{e}.npy" for e in epochs] + ["train.json"]
        outd = submit(f"rebuild_cs_{recipe.kind}_{role}_{stage}_{_ts()}",
                      {"kind": "cs", "role": role, "stage": stage, "recipe": asdict(recipe), "save_epochs": epochs}, expected)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _, encode = cs.stage_rows(role, stage)
        lb = np.intersect1d(np.asarray(encode), common.lockbox_rows())
        enc = None
        for e in epochs:
            vec = np.load(outd / f"epoch{e}.npy")
            if len(lb):
                import torch
                if enc is None:
                    enc = cs.CaseEncoder(recipe.kind, device="cpu", lora_rank=recipe.lora_rank,
                                         freeze_embeddings=recipe.freeze_embeddings, max_len=recipe.max_len)
                enc.model.load_state_dict(torch.load(outd / f"state_{role}_{stage}_epoch{e}.pt", map_location="cpu"))
                texts = cs.mp.texts()
                vec[lb] = enc.encode([texts[int(r)] for r in lb])
            np.save(out_dir / f"epoch{e}.npy", vec)
        info = json.loads((outd / "train.json").read_text(encoding="utf-8"))
        info["remote"] = {"device": "RTX 5080 (gpu_remote)", "decision": "D0.5", "lockbox_rows_encoded_locally": int(len(lb))}
        (out_dir / "train.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        for name in ("job.log",):
            if (outd / name).exists():
                shutil.copy2(outd / name, out_dir / "train.log")
    cs.launch_train = launch_train
    return cs


# ----------------------------------------------------------------------------- ColBERT (colbert_bge) and dense_bge_long

def _note_manifest(path: Path, note: dict) -> None:
    m = json.loads(path.read_text(encoding="utf-8"))
    m["round3_rebuild"] = note
    path.write_text(json.dumps(m, indent=1, ensure_ascii=False), encoding="utf-8")


def run_colbert() -> None:
    from uol.embedding_push import colbert as cb
    from uol.embedding_push import common
    from uol.round3.gpu_remote import sha256
    cb.OUT.mkdir(parents=True, exist_ok=True)
    if not (cb.OUT / "pool.npz").exists():
        cb.stage_pool()
    if not (cb.OUT / "chunks.npz").exists():
        cb.stage_chunks()
    if not (cb.OUT / "queries_meta.npz").exists():
        cb.stage_queries_meta()
    rem = cb.OUT / "remote"
    rem.mkdir(exist_ok=True)
    qm = np.load(cb.OUT / "queries_meta.npz")
    idf = np.load(cb.OUT / "idf.npy")
    offset, n_vec = qm["offset"], qm["n_vec"]
    if not (rem / "qweights.npz").exists():
        w = np.concatenate([cb.query_weightings(qm["token_ids"][o:o + n], qm["placeholder"][o:o + n], idf)
                            for o, n in zip(offset, n_vec)]).astype(np.float32)
        np.savez(rem / "qweights.npz", offset=offset, n_vec=n_vec, weights=w)
    lb = common.lockbox_rows()
    if not (rem / "lb_qtok.npy").exists():
        data = common.release()
        texts = [data.queries[int(r)].text for r in lb]
        vecs: dict[int, np.ndarray] = {}
        frozen = np.load(cb.FROZEN / "query_vectors.npy", mmap_mode="r")
        cos = np.full(len(lb), np.nan)

        def sink(i, cls, vec):
            vecs[i] = vec
            cos[i] = float(cls @ frozen[int(lb[i])])
        t0 = time.time()
        enc = cb.Encoder("cpu")
        enc.encode(texts, cb.QUERY_MAX_TOKENS, 8192, sink, lengths=n_vec[lb].astype(np.int64) + 1)
        if np.nanmin(cos) < 0.999:
            raise RuntimeError(f"lockbox query CLS check failed: min cos {np.nanmin(cos)}")
        np.save(rem / "lb_rows.npy", lb)
        np.save(rem / "lb_qtok.npy", np.concatenate([vecs[i] for i in range(len(lb))]).astype(np.float16))
        (rem / "lb_check.json").write_text(json.dumps({"rows": int(len(lb)), "cos_min": float(np.nanmin(cos)),
                                                      "cpu_seconds": round(time.time() - t0, 1)}), encoding="utf-8")
    if not (cb.OUT / "pair_scores.npy").exists():
        inputs = {"chunks.npz": cb.OUT / "chunks.npz", "chunk_texts.jsonl": cb.OUT / "chunk_texts.jsonl",
                  "pool.npz": cb.OUT / "pool.npz", "qweights.npz": rem / "qweights.npz", "lb_rows.npy": rem / "lb_rows.npy",
                  "lb_qtok.npy": rem / "lb_qtok.npy"}
        outd = submit(f"rebuild_colbert_score_{_ts()}", {"kind": "colbert_score", "role": "all",
                      "chunks_sha": sha256(cb.OUT / "chunks.npz")}, ["pair_scores.npy"], inputs)
        for name in ("pair_scores.npy", "query_check.json", "docs_progress.json"):
            shutil.copy2(outd / name, (cb.OUT if name == "pair_scores.npy" else rem) / name)
    if not (cb.PAIRS_DIR / "manifest.json").exists():
        cb.stage_assemble()
        cb.stage_eval()
        _note_manifest(cb.PAIRS_DIR / "manifest.json", {"encoding": "RTX 5080 fp32 via gpu_remote (D0.5); lockbox query "
                       "token vectors on the authority PC CPU", "decision": "D0.5"})
    if not (cb.LONG / "long_cls.npy").exists():
        outd = submit(f"rebuild_colbert_long_{_ts()}", {"kind": "colbert_long", "role": "articles"},
                      ["long_cls.npy", "long_articles.npy", "long_info.json"])
        cb.LONG.mkdir(parents=True, exist_ok=True)
        for name in ("long_cls.npy", "long_articles.npy", "long_info.json"):
            shutil.copy2(outd / name, cb.LONG / name)
    if not (common.SCORES / cb.LONG_ID / "manifest.json").exists():
        cb.stage_long_scores()
        _note_manifest(common.SCORES / cb.LONG_ID / "manifest.json", {"encoding": "RTX 5080 fp32 via gpu_remote (D0.5)"})


def main(argv: list[str]) -> int:
    rebuild.patch_paths()
    what = argv[0]
    t0 = time.time()
    rebuild.ledger({"kind": "job_start", "job_id": f"remote_ep_{what}_{_ts()}", "utc": datetime.now(timezone.utc).isoformat(),
                    "command": f"python -m uol.round3.remote_ep {what}", "pid": os.getpid()})
    if what == "ft":
        fd = patch_ft()
        if not (fd.common.SCORES / "ft_e5_q" / "manifest.json").exists():
            fd.run_method("ft_e5_q", fd.Config(**FT_CFG), {"note": "Round 3 rebuild: frozen configuration e5-8b32e14f; "
                                                              "training on the RTX 5080 (D0.5)", "search_log": str(fd.SEARCH_LOG)})
    elif what == "cs":
        cs = patch_cs()
        cs.run_encoder("e5", None, cs.Recipe(kind="e5"))
    elif what == "colbert":
        run_colbert()
    else:
        raise SystemExit(f"unknown stage {what}")
    rebuild.ledger({"kind": "job_end", "job_id": f"remote_ep_{what}", "wall_seconds": round(time.time() - t0, 1),
                    "exit_code": 0})
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
