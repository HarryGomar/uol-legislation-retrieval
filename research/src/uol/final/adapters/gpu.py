"""GPU encoder jobs of the final run: job inputs, ingest of their outputs, the fidelity gates and the sparse scorer.

Job format = ``round3/gpu_jobs/legal_encode.py`` (``items.jsonl`` + ``task.json`` -> ``vectors.npy`` + ``keys.json``
+ ``run.json``); the sparse head uses ``uol/final/gpu_jobs/bge_m3_sparse.py`` in the same format. Model specs for
the legal encoders and Gate E0 come from ``scale_curve.legal.SPECS`` / ``GATE`` (pinned revisions, prompts,
precision); e5-base/large are added here. Items are the scale-curve units (39,286 article units, development query
strings only), checked dev-only again when written.

On the run computer the jobs run serially on the one GPU (``jobs/plan.json``); here nothing is submitted.
``ingest`` writes the vectors in the scale-curve layout under ``<new_vectors>/<key>`` so ``adapters.dense`` reads
them like any cached encoder.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard
from uol.final.adapters.base import Scorer, SystemPending, select

E5_PREFIX = {"articles": "passage: ", "queries_a": "query: "}
EXTRA_SPECS: dict[str, dict[str, Any]] = {
    # revision None: resolved on the run computer; the snapshot commit is recorded in run.json (pin before G0)
    "e5_base": dict(model="intfloat/multilingual-e5-base", revision=None, dim=768, max_tokens=512,
                    prompts={"articles": "none", "queries_a": "none"}, precision="fp32", batch_tokens=32768,
                    params="278M", prefix=E5_PREFIX, script="legal_encode"),
    "e5_large": dict(model="intfloat/multilingual-e5-large", revision=None, dim=1024, max_tokens=512,
                     prompts={"articles": "none", "queries_a": "none"}, precision="fp32", batch_tokens=32768,
                     params="560M", prefix=E5_PREFIX, script="legal_encode"),
    "bge_m3_sparse": dict(model="BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181", dim=None,
                          max_tokens=8192, prompts={"articles": "none", "queries_a": "none"}, precision="fp32",
                          batch_tokens=16384, params="568M", script="bge_m3_sparse"),
}
E5I_QUERY = ("Instruct: Given the facts of a Mexican court decision, retrieve the federal statutory articles the court "
             "applied\nQuery: ")
JINA = {"passage": {"task": "retrieval.passage", "prompt_name": "retrieval.passage"},
        "query": {"task": "retrieval.query", "prompt_name": "retrieval.query"}}
EXTRA_SPECS.update({
    # RUN_PLAN v2 §3.4 additions. Revisions are resolved on the run computer at download and recorded in run.json
    # (no network here to pin them); licences as stated on the model cards (verify at download): MIT, Apache-2.0,
    # CC BY-NC 4.0 (research use), Apache-2.0.
    "e5_large_instruct": dict(model="intfloat/multilingual-e5-large-instruct", revision=None, dim=1024, max_tokens=512,
                              prompts={"articles": "none", "queries_a": "none"}, precision="fp32", batch_tokens=32768,
                              params="560M", prefix={"queries_a": E5I_QUERY}, script="legal_encode", licence="MIT"),
    "gte_multilingual": dict(model="Alibaba-NLP/gte-multilingual-base", revision=None, dim=768, max_tokens=8192,
                             prompts={"articles": "none", "queries_a": "none"}, precision="fp32", batch_tokens=32768,
                             params="305M", script="st_encode", trust_remote_code=True, licence="Apache-2.0"),
    "jina_v3_docs": dict(model="jinaai/jina-embeddings-v3", revision=None, dim=1024, max_tokens=8192,
                         prompts={"articles": "none"}, precision="fp32", batch_tokens=32768, params="570M",
                         script="st_encode", trust_remote_code=True, encode_kwargs_by_prompt={"": JINA["passage"]},
                         licence="CC-BY-NC-4.0"),
    "jina_v3": dict(model="jinaai/jina-embeddings-v3", revision=None, dim=1024, max_tokens=8192,
                    prompts={"queries_a": "none"}, precision="fp32", batch_tokens=32768, params="570M",
                    script="st_encode", trust_remote_code=True, encode_kwargs_by_prompt={"": JINA["query"]},
                    articles_from="jina_v3_docs", licence="CC-BY-NC-4.0"),
    "arctic_l_v2": dict(model="Snowflake/snowflake-arctic-embed-l-v2.0", revision=None, dim=1024, max_tokens=8192,
                        prompts={"articles": "none", "queries_a": "model"}, precision="fp32", batch_tokens=32768,
                        params="568M", script="legal_encode", licence="Apache-2.0"),
})
EXTRA_SPECS.update({
    # the general-domain base of gemma_legal (RUN_PLAN §11 "each against its own base"; review 22 Oct §4.8): the same
    # settings as gemma_legal (the model's own sentence-transformers prompts "title: none | text: " / "task: search
    # result | query: ", its pooling + dense + normalize modules, 2,048 tokens, fp32). Gated on Hugging Face: the
    # account of the run computer must have accepted the Gemma terms (HF token in the environment, never in the bundle)
    "gemma_base": dict(model="google/embeddinggemma-300m", revision=None, dim=768, max_tokens=2048,
                       prompts={"articles": "model", "queries_a": "model"}, precision="fp32", batch_tokens=32768,
                       params="300M", script="legal_encode", licence="Gemma terms (gated)"),
})
EXTRA_SPECS.update({
    # same-path int8 base for Euler (legal plan: base and legal model through the same code path and precision)
    "qwen3_8b_int8path": dict(model="Qwen/Qwen3-Embedding-8B", revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", dim=4096,
                              max_tokens=8192, prompts={"articles": "none", "queries_a": "model", "queries_c": "none"},
                              precision="int8", batch_tokens=16384, params="8B int8", script="legal_encode"),
    # e5-small re-encoded on the 5080 (the article side the DEV-N query re-encodes are scored against)
    "e5_small_re": dict(model="intfloat/multilingual-e5-small", revision="614241f622f53c4eeff9890bdc4f31cfecc418b3", dim=384,
                        max_tokens=512, prompts={"articles": "none", "queries_a": "none"}, precision="fp32",
                        batch_tokens=32768, params="118M", prefix=E5_PREFIX, script="legal_encode", licence="MIT"),
    # query-side specs for the DEV-N re-encodes of saved vector sets (scale-curve conventions: variants a, b, c)
    "qwen3_0p6b_q": dict(model="Qwen/Qwen3-Embedding-0.6B", revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", dim=1024,
                         max_tokens=8192, prompts={"queries_a": "none", "queries_b": "none", "queries_c": "none"},
                         precision="fp32", batch_tokens=32768, params="0.6B", script="legal_encode", instruct=True),
    # fp32 4B is 16 GB of weights and cannot sit on the 16 GB card (gpu__devn__qwen3_4b OOM at 15.09 GiB, 21 Oct; the
    # CPU-offload attempt failed in sentence-transformers' .to() on meta tensors, 22 Oct): the ~2,700 DEV-N + control
    # query strings are encoded in fp32 on the CPU, the precision and maths of the saved qwen3_emb_4b_v1 vectors
    "qwen3_4b_q": dict(model="Qwen/Qwen3-Embedding-4B", revision="5cf2132abc99cad020ac570b19d031efec650f2b", dim=2560,
                       max_tokens=8192, prompts={"queries_a": "none", "queries_b": "none", "queries_c": "none"},
                       precision="fp32", batch_tokens=32768, params="4B", script="legal_encode", instruct=True,
                       device="cpu", cpu_threads=16, cpu_ram_gb=22,
                       device_reason="fp32 like the saved qwen3_emb_4b_v1 vectors (float32 weights and compute); 16 GB "
                                     "of fp32 weights do not fit the 16 GB card, so the queries are encoded on the CPU; "
                                     "the 50 unchanged control queries are checked against the saved vectors at ingest "
                                     "(cos >= .99, devn_fidelity)"),
    "qwen3_8b_q": dict(model="Qwen/Qwen3-Embedding-8B", revision="1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", dim=4096,
                       max_tokens=8192, prompts={"queries_a": "none", "queries_b": "none", "queries_c": "none"},
                       precision="int8", batch_tokens=16384, params="8B int8", script="legal_encode", instruct=True),
    "e5_large_legal_q": dict(model="wwydmanski/e5-large-legal-v0.2", revision=None, dim=1024, max_tokens=512,
                             prompts={"queries_a": "none"}, precision="fp32", batch_tokens=32768, params="560M",
                             prefix={"queries_a": "query: "}, script="legal_encode"),
})
# DEV-N: every dense vector set gets its 854 DEV-N queries re-encoded (queries only); target vector key -> job spec
DEVN_ENCODERS = {"qwen8b": "qwen3_8b_q", "qwen3_0p6b": "qwen3_0p6b_q", "qwen3_4b": "qwen3_4b_q",
                 "e5_large_legal": "e5_large_legal_q", "e5_small_re": "e5_small_re", "qwen3_8b_int8path": "qwen3_8b_int8path",
                 "e5_base": "e5_base", "e5_large": "e5_large", "e5_large_instruct": "e5_large_instruct",
                 "gte_multilingual": "gte_multilingual", "jina_v3": "jina_v3", "arctic_l_v2": "arctic_l_v2",
                 "alia_mrbert": "alia_mrbert", "gemma_legal": "gemma_legal", "bge_m3_base": "bge_m3_base",
                 "bge_m3_legal": "bge_m3_legal", "euler_legal": "euler_legal"}
DEVN_SKIPPED = {"bge_m3_colbert": "colbert_full scores the original DEV facts; DEV-N needs a variant score file of its "
                                    "854 clause-dropped queries and an override in dev_block (not built, D79)",
                "bge_m3_sparse": "sparse weights, not a dense encoder", "bge_m3_ft": "one model per fold (encoded in its own job)",
                "qwen3_0p6b_ft": "one model per fold (encoded in its own job, D89)",
                "ft_e5_q": "query tower trained per fold inside the rebuild stage", "qwen8b_adapter": "adapter over the 8B queries",
                "e5_small / bge_m3 frozen files": "scored on DEV-N through e5_small_re / bge_m3_base (re-encoded both sides)"}
# fresh vector stores (legal_encode --store-tag): e5_large_instruct's store holds vectors damaged by the 22 Oct GPU fault
# (review §3(a)); the old store stays in place as evidence. Key -> tag; DEV-N jobs of the same key get it too
STORE_TAGS = {"e5_large_instruct": "r6-20261022"}
# GPU queue priority (RUN_PLAN v2 §7); small encoders share the card by declared VRAM, 8B / fine-tune / CE run alone
ORDER = ("e5_base", "e5_large", "bge_m3_base", "bge_m3_sparse", "alia_mrbert", "gemma_legal", "gemma_base", "bge_m3_legal",
         "e5_large_instruct", "gte_multilingual", "jina_v3_docs", "jina_v3", "arctic_l_v2", "e5_small_re",
         "qwen3_8b_gate", "qwen3_8b_int8path", "euler_legal")
PRIORITY = {"e5_base": 1, "e5_large": 1, "bge_m3_base": 1, "bge_m3_sparse": 1, "alia_mrbert": 1, "gemma_legal": 1, "gemma_base": 2,
            "bge_m3_legal": 1, "e5_large_instruct": 2, "gte_multilingual": 2, "jina_v3_docs": 2, "jina_v3": 2,
            "arctic_l_v2": 2, "e5_small_re": 1, "qwen3_8b_gate": 3, "qwen3_8b_int8path": 5, "euler_legal": 5}
VRAM_GB = {"euler_legal": 16, "qwen3_8b_gate": 16, "e5_base": 3, "e5_large": 5, "bge_m3_base": 6, "bge_m3_sparse": 6,
           "alia_mrbert": 3, "gemma_legal": 4, "gemma_base": 4, "bge_m3_legal": 6, "e5_large_instruct": 5, "gte_multilingual": 5,
           "jina_v3_docs": 6, "jina_v3": 4, "arctic_l_v2": 6, "e5_small_re": 2, "qwen3_8b_int8path": 16, "qwen3_0p6b_q": 4,
           "qwen3_4b_q": 12, "qwen3_8b_q": 16, "e5_large_legal_q": 4}     # qwen3_4b_q runs on the CPU (device)
GPU_HOURS = {"alia_mrbert": 0.3, "gemma_legal": 0.4, "gemma_base": 0.4, "bge_m3_base": 0.8, "bge_m3_legal": 0.8, "e5_base": 0.3,
             "e5_large": 0.6, "bge_m3_sparse": 0.8, "qwen3_8b_gate": 0.3, "euler_legal": 7.0, "e5_large_instruct": 0.6,
             "gte_multilingual": 0.6, "jina_v3_docs": 0.6, "jina_v3": 0.1, "arctic_l_v2": 0.6, "e5_small_re": 0.2,
             "qwen3_8b_int8path": 2.0}


def specs() -> dict[str, dict[str, Any]]:
    from uol.scale_curve import legal
    out = {k: {**v, "script": "legal_encode"} for k, v in legal.SPECS.items()}
    out["qwen3_8b_gate"] = {**legal.GATE["qwen3_8b_gate"], "script": "legal_encode", "subset": "validation"}
    out.update(EXTRA_SPECS)
    return out


def units_path() -> Path:
    return context.active().path("units") / "units.jsonl"


def unit_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The scale-curve units with their manifest (dev queries only; the runner's copy is remapped to bundle rows)."""
    d = context.active().path("units")
    man = json.loads((d / "units_manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (d / "units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    if not man.get("dev_only") or man.get("lockbox_rows_included"):
        raise guard.TestSetRefused("units manifest is not dev-only")
    guard.dev_rows_only([r["o"] for r in rows if r["g"] != "articles"], "unit query owners")
    return rows, man


def build_inputs(key: str, out_dir: Path, subset: str | None = None) -> dict[str, Any]:
    """Write ``items.jsonl`` and ``task.json`` for one job (no labels; dev query strings only)."""
    from uol.scale_curve import legal
    s = specs()[key]
    subset = subset or s.get("subset", "full")
    rows, man = unit_rows()
    items = legal.select(rows, man, list(s["prompts"]), subset)          # asserts dev owners (scale_curve.assert_dev)
    pre = s.get("prefix") or {}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for it in items:
            if it["g"] in pre:
                it = {**it, "text": pre[it["g"]] + it["text"]}
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    task = {"key": key, "model": s["model"], "revision": s["revision"], "max_tokens": s["max_tokens"],
            "prompts": s["prompts"], "precision": s["precision"], "batch_tokens": s["batch_tokens"], "limit": None,
            "budget_seconds": None, "subset": subset, "units_sha256": man.get("sha256"),
            "fallback_batch_tokens": max(2048, int(s["batch_tokens"]) // 4),
            **{k: s[k] for k in ("trust_remote_code", "encode_kwargs_by_prompt", "articles_from", "licence") if k in s}}
    (out_dir / "task.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    return {"key": key, "items": len(items), "script": s["script"], "subset": subset}


def script_path(code_src: Path, script: str) -> Path:
    if script == "legal_encode":
        return code_src / "uol/round3/gpu_jobs/legal_encode.py"
    return code_src / "uol/final/gpu_jobs" / f"{script}.py"


# --------------------------------------------------------------------------- ingest


def _items(inputs: Path) -> dict[int, dict[str, Any]]:
    return {int(json.loads(line)["i"]): json.loads(line) for line in (inputs / "items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()}


def ingest(key: str, inputs: Path, outputs: Path) -> Path:
    """Assemble a finished dense job into ``<new_vectors>/<key>`` (scale-curve layout)."""
    run = json.loads((outputs / "run.json").read_text(encoding="utf-8"))
    if run.get("incomplete"):
        raise SystemPending(f"{key}: the job stopped at its budget; resubmit to resume")
    items = _items(inputs)
    meta = json.loads((outputs / "keys.json").read_text(encoding="utf-8"))
    task = json.loads((inputs / "task.json").read_text(encoding="utf-8"))
    m = re.match(r"^(bge_m3_ft|qwen3_0p6b_ft)_f(\d)$", key)
    dest = context.active().path("new_vectors") / (f"{m.group(1)}/fold{m.group(2)}" if m else key)
    dest.mkdir(parents=True, exist_ok=True)
    if task.get("articles_from"):
        (dest / "articles_from.json").write_text(json.dumps({"key": task["articles_from"]}), encoding="utf-8")
    if key == "bge_m3_sparse":
        from scipy import sparse
        M = sparse.load_npz(outputs / "sparse.npz").tocsr()
        pos = {k: i for i, k in enumerate(meta["keys"])}
        art_rows, art_owner, q = [], [], {}
        for i, ks in meta["items"]:
            it = items[int(i)]
            for k in ks:
                if it["g"] == "articles":
                    art_rows.append(pos[k])
                    art_owner.append(int(it["o"]))
                else:
                    q.setdefault(it["g"].split("_", 1)[1], []).append((int(it["o"]), pos[k]))
        sparse.save_npz(dest / "article_sparse.npz", M[art_rows])
        np.save(dest / "article_index.npy", np.asarray(art_owner, dtype=np.int64))
        for v, lst in q.items():
            rows = guard.dev_rows_only([r for r, _ in lst], "sparse query rows")
            sparse.save_npz(dest / f"query_sparse_{v}.npz", M[[p for _, p in lst]])
            (dest / f"query_rows_{v}.json").write_text(json.dumps({"release_rows": rows.tolist()}), encoding="utf-8")
    else:
        V = np.load(outputs / "vectors.npy")
        pos = {k: i for i, k in enumerate(meta["keys"])}
        art_rows, art_owner, q = [], [], {}
        for i, ks in meta["items"]:
            it = items[int(i)]
            if it["g"] == "articles":
                for k in ks:
                    art_rows.append(pos[k])
                    art_owner.append(int(it["o"]))
            else:
                q.setdefault(it["g"].split("_", 1)[1], []).append((int(it["o"]), pos[ks[0]]))
        if art_rows:
            np.save(dest / "article_vectors.npy", V[art_rows].astype(np.float32))
            np.save(dest / "article_index.npy", np.asarray(art_owner, dtype=np.int64))
        rows_ref = None
        for v, lst in sorted(q.items()):
            lst.sort()
            rows = guard.dev_rows_only([r for r, _ in lst], "query rows")
            if rows_ref is None:
                rows_ref = rows
            elif not np.array_equal(rows, rows_ref):
                raise ValueError(f"{key}: query variants cover different rows")
            np.save(dest / f"query_vectors_{v}.npy", V[[p for _, p in lst]].astype(np.float32))
        if rows_ref is not None:
            (dest / "query_rows.json").write_text(json.dumps({"release_rows": rows_ref.tolist()}), encoding="utf-8")
    (dest / "manifest.json").write_text(json.dumps({"key": key, "job_run": run, "ingested_from": str(outputs)},
                                                   indent=1, default=str), encoding="utf-8")
    if "__" in key and key.split("__", 1)[0] in context.TEXT_VARIANTS:
        (dest / "fidelity.json").write_text(json.dumps(devn_fidelity(task, dest), indent=1), encoding="utf-8")
    return dest


def devn_fidelity(task: dict, dest: Path) -> dict:
    """Cosine of the 50 control queries (unchanged facts) re-encoded here against the saved vectors of the target set."""
    from uol.final.adapters.dense import VectorSet
    prof = context.active()
    target = task["target"]
    folder = prof.path(target) if target in prof.paths else prof.path("new_vectors") / target
    try:
        vs = VectorSet(folder)
    except Exception as exc:                                         # the target may be encoded later in the run
        return {"target": target, "checked": False, "reason": str(exc)[:200]}
    rows = json.loads((dest / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
    pos = {int(r): i for i, r in enumerate(rows)}
    out = {"target": target, "checked": True, "variants": {}}
    for p in sorted(dest.glob("query_vectors_*.npy")):
        v = p.stem.replace("query_vectors_", "")
        tv = v if v in vs.queries else (vs.variants[0] if len(vs.variants) == 1 else None)
        if tv is None:
            continue
        V = np.load(p)
        Q, qpos = vs.queries[tv]
        cos = [float(V[pos[r]] @ np.asarray(Q[qpos[r]], dtype=np.float32)) for r in task["control_rows"] if r in qpos and r in pos]
        out["variants"][v] = {"target_variant": tv, "n": len(cos), "cos_mean": float(np.mean(cos)) if cos else None,
                              "cos_min": float(np.min(cos)) if cos else None}
    vals = [x["cos_mean"] for x in out["variants"].values() if x["cos_mean"] is not None]
    out["pass"] = bool(vals) and min(vals) >= 0.99
    return out


def gate_e0(inputs: Path, outputs: Path, reference: Path) -> dict[str, Any]:
    """Gate E0 of legal_encoders PLAN: the int8 base 8B must match the cached 8B vectors (mean cos >= .99, min >= .97)
    on the validation set; ``reference`` (npz from the bundle) holds the cached vectors aligned to the items."""
    from uol.scale_curve import legal
    items = _items(inputs)
    meta = json.loads((outputs / "keys.json").read_text(encoding="utf-8"))
    V = np.load(outputs / "vectors.npy")
    pos = {k: i for i, k in enumerate(meta["keys"])}
    ref = np.load(reference)
    refmap = {int(i): v for i, v in zip(ref["item"], ref["vector"])}
    cos = [float(V[pos[ks[0]]] @ refmap[int(i)]) for i, ks in meta["items"] if int(i) in refmap and len(ks) == 1]
    out = {"n": len(cos), "cos_mean": float(np.mean(cos)) if cos else float("nan"),
           "cos_min": float(np.min(cos)) if cos else float("nan"), "rule": legal.GATE_E0}
    out["pass"] = bool(cos and out["cos_mean"] >= legal.GATE_E0["cos_mean"] and out["cos_min"] >= legal.GATE_E0["cos_min"])
    return out


def gate_e0_reference(inputs: Path, dest: Path) -> Path:
    """Bundle time (thesis PC): cached 8B vectors for the validation items (articles by unit position, queries by row)."""
    from uol.scale_curve import common as SC
    items = _items(inputs)
    rows, _ = unit_rows()
    art_pos = {}
    p = 0
    for r in rows:
        if r["g"] == "articles":
            art_pos[(r["o"], r["w"])] = p
            p += 1
    A = np.load(SC.QWEN8B / "article_vectors.npy", mmap_mode="r")
    Q = np.load(SC.QWEN8B / "query_vectors_c.npy", mmap_mode="r")
    ids, vecs = [], []
    for i, it in items.items():
        if it["g"] == "articles":
            v = A[art_pos[(it["o"], it["w"])]]
        else:
            v = Q[int(guard.dev_rows_only([it["o"]])[0])]
        ids.append(i)
        vecs.append(np.asarray(v, dtype=np.float32))
    np.savez(dest, item=np.asarray(ids), vector=np.stack(vecs))
    return dest


# --------------------------------------------------------------------------- sparse scorer


class SparseVectors(Scorer):
    """bge-m3 lexical matching: score = sum over shared tokens of w_q(t) w_d(t), MaxP over article windows."""

    def prepare(self, pop) -> None:
        super().prepare(pop)
        from scipy import sparse
        from uol.embedding_push import common as ep
        d = context.active().path("new_vectors") / self.spec["adapter_args"].get("key", "bge_m3_sparse")
        if not (d / "article_sparse.npz").exists():
            raise SystemPending(f"{self.sid}: sparse weights not available yet at {d}")
        self.A = sparse.load_npz(d / "article_sparse.npz").tocsr()
        self.owner = np.load(d / "article_index.npy")
        self.N = len(ep.release().candidate_ids)
        self.q = {}
        for p in sorted(d.glob("query_sparse_*.npz")):
            v = p.stem.replace("query_sparse_", "")
            rows = json.loads((d / f"query_rows_{v}.json").read_text(encoding="utf-8"))["release_rows"]
            self.q[v] = (sparse.load_npz(p).tocsr(), {int(r): i for i, r in enumerate(guard.dev_rows_only(rows))})
        self.grid = [{"variant": v} for v in sorted(self.q)]
        self.AT = self.A.T.tocsr()
        order = np.argsort(self.owner, kind="stable")
        self.order, so = order, self.owner[order]
        self.starts = np.flatnonzero(np.r_[True, so[1:] != so[:-1]])
        self.owners = so[self.starts]

    def _scores(self, params: dict, rows: Sequence[int]) -> np.ndarray:
        rows = self.rows(rows)
        Q, pos = self.q[params["variant"]]
        sims = (Q[[pos[int(r)] for r in rows]] @ self.AT).toarray()[:, self.order]
        best = np.maximum.reduceat(sims, self.starts, axis=1)
        out = np.zeros((len(rows), self.N), dtype=np.float32)
        out[:, self.owners] = best
        return out

    def fit(self, fold) -> dict[str, Any]:
        super().fit(fold)
        self.params, hist = select(self.grid, self._scores, fold, self.pop.eval_population)
        return {"params": self.params, "selection": hist}

    def score(self, rows: Sequence[int]) -> np.ndarray:
        return self._scores(self.params, rows)
