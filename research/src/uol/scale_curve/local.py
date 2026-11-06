"""Qwen3-Embedding 0.6B / 4B on the CPU with the 8B conventions, in time-boxed, checkpointed chunks.

Conventions (as the 8B vectors served by vLLM): documents are plain text and queries are the 8B query strings
(variants a, b, c); the tokenizer appends one ``<|endoftext|>`` token; the embedding is the last token's final hidden
state (left padding), L2-normalised; float32 on the CPU (the weights are released in bfloat16 and upcast exactly).
Articles above 8,000 tokens use the 8B windows and MaxP. No GPU is used (``CUDA_VISIBLE_DEVICES`` is cleared).

Run every command through the heavy lock, one chunk at a time::

    python -m uol.round3.heavy uol.scale_curve.local encode --model qwen3_0p6b --budget-min 40
    python -m uol.round3.heavy uol.scale_curve.local bench --model qwen3_4b --layers 6

``encode`` exits 0 when every unit is stored (and the vector files are assembled), 3 when its time budget ran out
or a pause/stop flag appeared, so ``driver`` can loop with the lock released between chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""                # never the GTX 1070

import numpy as np

from uol.embedding_push import qwen_endpoint as qe
from uol.scale_curve import common as C
from uol.scale_curve import store

EOS_ID = 151643                                          # <|endoftext|>
MAX_LEN = 8192
LONG = 2048                                              # bench/projection split: units above this are timed apart
GROUP_ORDER = ("queries_c", "articles", "queries_b", "queries_a")   # c first: the 8B's selected variant
RSS_ABORT_GB = 9.5


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 30


class Encoder:
    """A local Qwen3-Embedding model (optionally truncated to ``n_layers`` for a throughput projection only)."""

    def __init__(self, spec: C.ModelSpec, threads: int = 16, n_layers: int | None = None, int8: bool = False) -> None:
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        torch.set_num_threads(int(threads))
        self.torch, self.spec, self.threads = torch, spec, int(threads)
        path = spec.snapshot
        if not (path / "config.json").exists():
            raise FileNotFoundError(f"{spec.model_id} snapshot {spec.revision} is not in the HF cache")
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left", local_files_only=True)
        cfg = AutoConfig.from_pretrained(path, local_files_only=True)
        self.full_layers = int(cfg.num_hidden_layers)
        self.n_layers = int(n_layers or cfg.num_hidden_layers)
        if n_layers:
            cfg.num_hidden_layers = int(n_layers)
            if getattr(cfg, "layer_types", None):
                cfg.layer_types = cfg.layer_types[: int(n_layers)]
        self.model = AutoModel.from_pretrained(path, config=cfg, dtype=torch.float32, attn_implementation="sdpa",
                                               local_files_only=True).eval()
        self.int8 = bool(int8)
        if int8:
            self.model = torch.ao.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Unit-norm float32 [n, dim] last-token embeddings."""
        torch = self.torch
        batch = self.tok(list(texts), padding=True, truncation=False, return_tensors="pt")
        ids = batch["input_ids"]
        if ids.shape[1] > MAX_LEN:
            raise ValueError(f"an input has {ids.shape[1]} tokens (> {MAX_LEN})")
        if not bool((ids[:, -1] == EOS_ID).all()):
            raise RuntimeError("the tokenizer did not append <|endoftext|> to every input")
        with torch.inference_mode():
            hidden = self.model(**batch).last_hidden_state[:, -1]
            hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=1)
        return hidden.numpy().astype(np.float32)


# --------------------------------------------------------------------------- checks before any encoding


def check_tokenizer(spec: C.ModelSpec, texts: Sequence[str]) -> dict[str, Any]:
    """The model's tokenizer gives the 8B tokenizer's ids (so the 8B windows and token counts carry over) and appends
    exactly one end-of-text token; the 8B endpoint counted that token too (doc probe)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.snapshot, local_files_only=True)
    ref = qe.load_tokenizer()
    mismatch = 0
    for t in texts:
        a = tok(t, add_special_tokens=False)["input_ids"]
        if a != ref.encode(t, add_special_tokens=False).ids:
            mismatch += 1
    with_eos = [tok(t)["input_ids"] for t in texts[:20]]
    plain = [tok(t, add_special_tokens=False)["input_ids"] for t in texts[:20]]
    eos_ok = all(w == p + [EOS_ID] for w, p in zip(with_eos, plain))
    probe_rows = [r for r in qe.read_jsonl(qe.PROBES_LOG) if r.get("kind") == "doc" and r.get("prompt_tokens")]
    served = int(probe_rows[0]["prompt_tokens"]) if probe_rows else None
    local = len(ref.encode(qe.PROBE_DOC, add_special_tokens=False).ids)
    out = {"texts_checked": len(texts), "id_mismatches_vs_8b_tokenizer": mismatch, "eos_appended": eos_ok,
           "doc_probe_tokens_8b_endpoint": served, "doc_probe_tokens_local_without_eos": local,
           "endpoint_appended_one_token": served == local + 1}
    out["pass"] = mismatch == 0 and eos_ok and out["endpoint_appended_one_token"]
    return out


def card_check(enc: Encoder) -> dict[str, Any]:
    """Reproduce the model card's transformers example (first ``# [[..]]`` line of README.md)."""
    readme = (enc.spec.snapshot / "README.md").read_text(encoding="utf-8")
    expected = json.loads(re.search(r"^# (\[\[.*\]\])\s*$", readme, re.M).group(1))
    task = "Given a web search query, retrieve relevant passages that answer the query"
    queries = [f"Instruct: {task}\nQuery:What is the capital of China?", f"Instruct: {task}\nQuery:Explain gravity"]
    docs = ["The capital of China is Beijing.",
            "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is "
            "responsible for the movement of planets around the sun."]
    v = enc.encode(queries + docs)
    got = (v[:2] @ v[2:].T).tolist()
    diff = float(np.max(np.abs(np.asarray(got) - np.asarray(expected))))
    return {"expected": expected, "got": got, "max_abs_diff": diff, "pass": diff < 2e-3}


# --------------------------------------------------------------------------- batching and the chunked run


plan_batches = store.plan_batches


def todo_items(spec: C.ModelSpec, units: dict[str, Any], store: C.VectorStore) -> list[tuple[str, list[C.Item]]]:
    groups = {"articles": units["articles"], **{f"queries_{v}": units["queries"][v] for v in C.VARIANTS}}
    out = []
    for name in GROUP_ORDER:
        seen, items = set(), []
        for it in groups[name]:
            k = C.content_key(it.text, spec.model_id)
            if k not in store and k not in seen:
                seen.add(k)
                items.append(it)
        out.append((name, items))
    return out


def model_run_dir(spec: C.ModelSpec) -> Path:
    return C.RUN / spec.key


def encode_chunk(model_key: str, budget_min: float, threads: int) -> int:
    spec = C.MODELS[model_key]
    rdir = model_run_dir(spec)
    name = f"{spec.key}_encode"
    t0 = time.monotonic()
    started = C.utc()
    units = C.build_units(check=True)
    store = C.VectorStore(spec.store_dir, spec.dim)
    todo = todo_items(spec, units, store)
    remaining = sum(len(items) for _, items in todo)
    C.log(f"{spec.model_id}: chunk start (budget {budget_min:.0f} min, {threads} threads); store {len(store)}, "
          f"missing {remaining} units ({sum(it.est_tokens for _, its in todo for it in its) / 1e6:.2f}M est tokens)", name)
    if remaining == 0:
        return finish(spec, units, store, rdir)
    enc = Encoder(spec, threads=threads)
    if not (rdir / "checks.json").exists():
        sample = [it.text for it in units["articles"][::397]] + [it.text for it in units["queries"]["c"][::97]]
        checks = {"tokenizer": check_tokenizer(spec, sample), "model_card": card_check(enc), "utc": C.utc()}
        C.write_json(rdir / "checks.json", checks)
        C.log(f"checks: tokenizer {checks['tokenizer']['pass']}, model card {checks['model_card']['pass']} "
              f"(max diff {checks['model_card']['max_abs_diff']:.2e})", name)
        if not (checks["tokenizer"]["pass"] and checks["model_card"]["pass"]):
            raise RuntimeError(f"pre-encoding checks failed: see {rdir / 'checks.json'}")
    budget = budget_min * 60.0
    done_items = done_tokens = 0
    rss_max = rss_gb()
    reason = "complete"
    last_log = time.monotonic()
    enc_seconds = 0.0
    for gname, items in todo:
        if not items:
            continue
        for b in plan_batches([it.est_tokens for it in items]):
            if time.monotonic() - t0 > budget:
                reason = "budget"
            elif C.PAUSE_FILE.exists():
                reason = "pause flag"
            elif C.STOP_FILE.exists():
                reason = "stop file"
            if reason != "complete":
                break
            batch = [items[i] for i in b]
            ts = time.monotonic()
            vecs = enc.encode([it.text for it in batch])
            enc_seconds += time.monotonic() - ts
            store.add([C.content_key(it.text, spec.model_id) for it in batch], vecs)
            done_items += len(batch)
            done_tokens += sum(it.est_tokens for it in batch)
            rss_max = max(rss_max, rss_gb())
            if rss_max > RSS_ABORT_GB:
                reason = f"rss {rss_max:.1f} GB above {RSS_ABORT_GB}"
                break
            if time.monotonic() - last_log > 120:
                last_log = time.monotonic()
                C.log(f"{gname}: {done_items} units, {done_tokens / 1e6:.3f}M tokens this chunk, "
                      f"{done_tokens / max(enc_seconds, 1e-9):.0f} tok/s, rss {rss_max:.2f} GB", name)
        if reason != "complete":
            break
    left = sum(1 for _, items in todo_items(spec, units, store) for _ in items)
    rec = {"model": spec.model_id, "started_utc": started, "ended_utc": C.utc(), "wall_s": time.monotonic() - t0,
           "encode_s": enc_seconds, "threads": threads, "units": done_items, "est_tokens": done_tokens,
           "tok_per_s": done_tokens / max(enc_seconds, 1e-9), "rss_max_gb": rss_max, "stop_reason": reason,
           "units_left": left, "store_vectors": len(store), "pid": os.getpid()}
    C.append_jsonl(rdir / "chunks.jsonl", rec)
    C.ledger({"kind": "encode_chunk", **rec})
    C.log(f"chunk end ({reason}): {done_items} units, {done_tokens / 1e6:.3f}M tokens in {enc_seconds:.0f}s "
          f"({rec['tok_per_s']:.0f} tok/s), rss max {rss_max:.2f} GB, {left} units left", name)
    if left == 0:
        return finish(spec, units, store, rdir)
    return 3


def finish(spec: C.ModelSpec, units: dict[str, Any], store: C.VectorStore, rdir: Path) -> int:
    chunks = qe.read_jsonl(rdir / "chunks.jsonl")
    extra = {"encoder": {"device": "cpu (torch fp32, sdpa attention)", "pooling": "last token, left padding, L2",
                         "eos": "<|endoftext|> appended by the tokenizer (as the 8B endpoint)",
                         "checks": json.loads((rdir / "checks.json").read_text(encoding="utf-8"))},
             "compute": {"chunks": len(chunks), "encode_seconds": sum(c["encode_s"] for c in chunks),
                         "wall_seconds": sum(c["wall_s"] for c in chunks),
                         "est_tokens": sum(c["est_tokens"] for c in chunks),
                         "rss_max_gb": max((c["rss_max_gb"] for c in chunks), default=None)}}
    man = C.assemble(spec, units, extra)
    C.ledger({"kind": "assembled", "model": spec.model_id, "manifest": str(spec.derived / "manifest.json"),
              "store_vectors": man["counts"]["store_vectors"]})
    C.log(f"{spec.model_id}: all units stored; vector files and manifest written", f"{spec.key}_encode")
    return 0


# --------------------------------------------------------------------------- throughput bench and projection


def bench(model_key: str, n: int = 500, threads: int = 16, layers: int | None = None, int8: bool = False,
          n_long: int = 2, store_vectors: bool = False) -> dict[str, Any]:
    """Time ``n`` random article units (<= LONG tokens) and ``n_long`` long windows; project the full encode.

    With ``layers`` the model is truncated to that many layers and the per-token time is scaled by full/used
    layers (a projection only; such vectors are never stored).
    """
    spec = C.MODELS[model_key]
    units = C.build_units(check=True)
    rng = np.random.default_rng(20261016)
    arts = units["articles"]
    short = [i for i, it in enumerate(arts) if it.est_tokens <= LONG]
    long_ = [i for i, it in enumerate(arts) if it.est_tokens > 6000]
    pick = sorted(rng.choice(short, size=n, replace=False).tolist())
    pick_long = sorted(rng.choice(long_, size=min(n_long, len(long_)), replace=False).tolist())
    t_load = time.monotonic()
    enc = Encoder(spec, threads=threads, n_layers=layers, int8=int8)
    load_s = time.monotonic() - t_load
    store = C.VectorStore(spec.store_dir, spec.dim) if (store_vectors and not layers and not int8) else None
    rdir = model_run_dir(spec)
    if store is not None and not (rdir / "checks.json").exists():
        sample = [it.text for it in arts[::397]] + [it.text for it in units["queries"]["c"][::97]]
        checks = {"tokenizer": check_tokenizer(spec, sample), "model_card": card_check(enc), "utc": C.utc()}
        C.write_json(rdir / "checks.json", checks)
        if not (checks["tokenizer"]["pass"] and checks["model_card"]["pass"]):
            raise RuntimeError(f"pre-encoding checks failed: see {rdir / 'checks.json'}")
    res: dict[str, Any] = {}
    for label, idx in (("short", pick), ("long", pick_long)):
        items = [arts[i] for i in idx]
        tok = sum(it.est_tokens for it in items)
        ts = time.monotonic()
        for b in plan_batches([it.est_tokens for it in items]):
            batch = [items[i] for i in b]
            vecs = enc.encode([it.text for it in batch])
            if store is not None:
                store.add([C.content_key(it.text, spec.model_id) for it in batch], vecs)
        sec = time.monotonic() - ts
        res[label] = {"units": len(items), "est_tokens": tok, "seconds": sec, "tok_per_s": tok / max(sec, 1e-9)}
    scale = enc.full_layers / enc.n_layers
    all_items = arts + [it for v in C.VARIANTS for it in units["queries"][v]]
    tok_short = sum(it.est_tokens for it in all_items if it.est_tokens <= LONG)
    tok_long = sum(it.est_tokens for it in all_items if it.est_tokens > LONG)
    tok_c = sum(it.est_tokens for it in arts if it.est_tokens <= LONG) + sum(
        it.est_tokens for it in units["queries"]["c"] if it.est_tokens <= LONG)
    tok_c_long = sum(it.est_tokens for it in arts if it.est_tokens > LONG) + sum(
        it.est_tokens for it in units["queries"]["c"] if it.est_tokens > LONG)
    hours = lambda s, l: scale * (s / res["short"]["tok_per_s"] + l / res["long"]["tok_per_s"]) / 3600
    out = {"model": spec.model_id, "revision": spec.revision, "threads": threads, "layers_used": enc.n_layers,
           "layers_full": enc.full_layers, "int8_dynamic": int8, "load_seconds": load_s, "rss_gb": rss_gb(), **res,
           "projection_note": "per-token time x (full layers / layers used); units above 2,048 tokens timed apart",
           "tokens_all_variants": {"short": tok_short, "long": tok_long},
           "projected_hours_all_variants": hours(tok_short, tok_long),
           "projected_hours_articles_plus_variant_c": hours(tok_c, tok_c_long), "utc": C.utc()}
    tag = f"bench_{spec.key}_t{threads}" + (f"_L{layers}" if layers else "") + ("_int8" if int8 else "")
    C.write_json(C.RUN / "bench" / f"{tag}.json", out)
    C.ledger({"kind": "bench", "tag": tag, **{k: out[k] for k in ("model", "threads", "layers_used", "int8_dynamic",
                                                                   "projected_hours_all_variants",
                                                                   "projected_hours_articles_plus_variant_c")}})
    C.log(f"{tag}: short {res['short']['tok_per_s']:.0f} tok/s, long {res['long']['tok_per_s']:.0f} tok/s "
          f"(x{scale:.1f} layers) -> {out['projected_hours_all_variants']:.1f} h all variants, "
          f"{out['projected_hours_articles_plus_variant_c']:.1f} h articles + variant c; rss {out['rss_gb']:.2f} GB", "bench")
    return out


def int8_fidelity(model_key: str, layers: int, threads: int = 16, n: int = 64) -> dict[str, Any]:
    """Cosine between fp32 and int8-dynamic vectors of the same (truncated) model on ``n`` article units."""
    spec = C.MODELS[model_key]
    units = C.build_units(check=False)
    rng = np.random.default_rng(7)
    arts = [it for it in units["articles"] if it.est_tokens <= LONG]
    items = [arts[i] for i in sorted(rng.choice(len(arts), size=n, replace=False).tolist())]
    a = Encoder(spec, threads=threads, n_layers=layers)
    va = np.concatenate([a.encode([items[i].text for i in b]) for b in plan_batches([it.est_tokens for it in items])])
    del a
    q = Encoder(spec, threads=threads, n_layers=layers, int8=True)
    vb = np.concatenate([q.encode([items[i].text for i in b]) for b in plan_batches([it.est_tokens for it in items])])
    cos = (va * vb).sum(1)
    out = {"model": spec.model_id, "layers": layers, "n": n, "cos_mean": float(cos.mean()), "cos_min": float(cos.min()),
           "utc": C.utc()}
    C.write_json(C.RUN / "bench" / f"int8_fidelity_{spec.key}_L{layers}.json", out)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Qwen3-Embedding CPU encoder (scale curve, dev only)")
    ap.add_argument("command", choices=["encode", "bench", "int8-fidelity", "assemble"])
    ap.add_argument("--model", required=True, choices=[k for k, s in C.MODELS.items() if s.kind == "local"])
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args(argv)
    if a.command == "encode":
        return encode_chunk(a.model, a.budget_min, a.threads)
    if a.command == "bench":
        bench(a.model, n=a.n, threads=a.threads, layers=a.layers, int8=a.int8)
        return 0
    if a.command == "int8-fidelity":
        print(json.dumps(int8_fidelity(a.model, a.layers or 6, a.threads), indent=1))
        return 0
    spec = C.MODELS[a.model]
    return finish(spec, C.build_units(), C.VectorStore(spec.store_dir, spec.dim), model_run_dir(spec))


if __name__ == "__main__":
    sys.exit(main())
