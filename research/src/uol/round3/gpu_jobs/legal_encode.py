"""Remote job (legal encoders, PLAN legal_encoders_20261020): encode scale-curve units with one sentence-transformers model.

Inputs: ``items.jsonl`` ({"i", "g", "o", "w", "text"}: article units and dev query strings, no labels) and
``task.json`` ({"key", "model", "revision", "max_tokens", "prompts", "precision", "batch_tokens", "limit",
"budget_seconds"}). At the budget the job stops encoding, keeps its checkpoint, writes empty outputs with
``run.json`` "incomplete": true and exits 3 (the GPU cap of the PLAN; a later job with the same key resumes).
``prompts`` maps each item group ("articles", "queries_a", ...) to "model" (the model's own "document" or "query"
prompt from its sentence-transformers config) or "none".
Helpers shipped with the job: ``rerank_bench.py`` (private pinned installs) and ``store.py`` (content keys, batch
planning, the resumable vector store).

An article unit longer than ``max_tokens`` (prompt included) is cut into consecutive windows of the model's tokens
(``OVERLAP`` shared); later windows are the unit's first line + newline + their span, as the 8B units. Queries are
truncated, never split. Vectors are L2-normalised and checkpointed in ``UOL_WORKER_SCRATCH/legal/<key>/store`` under
``content_key(window text, model|prompt)``, so a rerun with the same key resumes. Outputs: ``vectors.npy`` (float32, one row
per key of ``keys.json``), ``keys.json`` ({"items": [[i, [keys...]], ...], "keys": [...]}) and ``run.json``.

    python legal_encode.py --inputs inputs --out outputs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_bench as RB  # noqa: E402
try:
    import store as S  # noqa: E402  (shipped beside this file on the worker)
except ImportError:                                      # local tests: the package copy
    from uol.scale_curve import store as S

ST_PINS = RB.PINS + ["sentence-transformers==6.1.0"]
OVERLAP = 64
TITLE_MAX = 100
SPECIAL_RESERVE = 4
RUNTIME: dict = {}                                       # stack, device and attention choices, written to run.json
STORE_CHECK_SAMPLE = 2048                                # resumed stored vectors re-encoded before a resumed job continues
STORE_CHECK_MIN_COS = {"int8": 0.99}                     # default .999; int8 (LLM.int8 outlier columns depend on the batch)


def store_dir(scratch: Path, key: str, tag: str | None = None) -> Path:
    """``<scratch>/legal/<key>[@<tag>]/store``. A tag gives a job a fresh store without touching the old one (the
    e5_large_instruct store damaged by the 22 Oct GPU fault stays in place as evidence)."""
    return Path(scratch) / "legal" / (f"{key}@{tag}" if tag else key) / "store"


def check_store(vs, work: dict, keys: list[str], encode, threshold: float, sample: int = STORE_CHECK_SAMPLE) -> dict:
    """Re-encode a spread sample of ``keys`` (vectors kept in the store from an earlier run) and compare with the stored
    vectors. A GPU fault can leave finite garbage in the store (``VectorStore.add`` rejects only non-finite rows; review
    of 22 Oct 2026 §3(a): e5_large_instruct). ``encode(texts, prompt)`` -> unit vectors. Returns the record; ``bad`` > 0
    means the store must not be used."""
    import numpy as np
    picked = keys[:: max(1, len(keys) // sample)] if keys else []
    bad, low = [], 1.0
    for pr in dict.fromkeys(work[k][0] for k in picked):
        ks = [k for k in picked if work[k][0] == pr]
        for s in range(0, len(ks), 32):
            b = ks[s:s + 32]
            v = np.asarray(encode([work[k][1] for k in b], pr), dtype=np.float32)
            cos = (v * vs.get(b)).sum(1)
            low = min(low, float(cos.min()))
            bad += [k for k, c in zip(b, cos) if c < threshold]
    return {"resumed": len(keys), "n": len(picked), "bad": len(bad), "min_cos": low if picked else None,
            "threshold": threshold}


def avoid_gqa_math_sdpa(module=None) -> bool:
    """Grouped-query models (Qwen3-Embedding, EmbeddingGemma) through transformers' SDPA path: a batch without padding
    (one long article window) has no attention mask, so transformers asks SDPA for ``enable_gqa=True``; on this
    Windows build the memory-efficient kernel does not take that flag and SDPA falls back to the math kernel, which
    materialises the [heads, L, L] float32 scores (7.28 GiB for one 7.8k-token window of the 8B: gpu__qwen3_8b_gate,
    21 Oct 2026). Repeating the key/value heads instead (what transformers already does whenever a mask is present)
    lets SDPA use the memory-efficient kernel; the attention computed is the same."""
    if module is None:
        try:
            from transformers.integrations import sdpa_attention as module
        except Exception:                                  # pragma: no cover - other transformers layouts
            return False
    if not hasattr(module, "use_gqa_in_sdpa"):
        return False
    module.use_gqa_in_sdpa = lambda *a, **k: False
    RUNTIME["sdpa_gqa"] = "repeat_kv (memory-efficient kernel), not enable_gqa"
    return True


def windows(text: str, tok, max_tokens: int, prompt_tokens: int = 0) -> list[str]:
    """The unit as one text, or title-prefixed windows cut on token offsets (exact substrings of the unit)."""
    budget = max_tokens - prompt_tokens - SPECIAL_RESERVE
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if len(ids) <= budget:
        return [text]
    title = text.split("\n", 1)[0].strip()
    t_ids = tok(title, add_special_tokens=False, return_offsets_mapping=True)
    if len(t_ids["input_ids"]) > TITLE_MAX:
        title = title[:t_ids["offset_mapping"][TITLE_MAX - 1][1]]
    content = budget - min(len(t_ids["input_ids"]), TITLE_MAX) - 2
    if content <= OVERLAP:
        raise ValueError("max_tokens leaves no room for window content")
    out, start = [], 0
    while True:
        end = min(start + content, len(ids))
        body = text[offsets[start][0]:offsets[end - 1][1]]
        out.append(body if start == 0 else title + "\n" + body)
        if end >= len(ids):
            return out
        start = end - OVERLAP


def truncate(text: str, tok, max_tokens: int, prompt_tokens: int = 0) -> tuple[str, bool]:
    budget = max_tokens - prompt_tokens - SPECIAL_RESERVE
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= budget:
        return text, False
    return text[:enc["offset_mapping"][budget - 1][1]], True


def stack_versions() -> dict:
    """Versions of the ML stack this job imported (the private pins or the venv's own), for run.json."""
    import importlib
    out = {}
    for mod in ("torch", "transformers", "sentence_transformers", "huggingface_hub", "tokenizers", "peft", "accelerate",
                "einops", "bitsandbytes"):
        try:
            m = importlib.import_module(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return out


def load(task: dict, scratch: Path, device: str | None = None):
    """``device``: None/"cuda" (the GPU), or "cpu" (fp32 on the CPU; task or ``--device``)."""
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    path = snapshot_download(task["model"], revision=task["revision"], cache_dir=str(scratch / "hf"),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "*/*.json",
                                             "*/*.safetensors", "pytorch_model.bin", "*.jinja"])
    kwargs, dev = {"dtype": torch.float32}, ("cpu" if device == "cpu" else "cuda")
    if task["precision"] == "int8":
        if dev == "cpu":
            raise ValueError("the int8 path (bitsandbytes) runs on the GPU only")
        from transformers import BitsAndBytesConfig
        kwargs = {"dtype": torch.bfloat16, "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                  "device_map": {"": 0}}
        dev = None                                       # bitsandbytes places the weights; no .to() afterwards
    avoid_gqa_math_sdpa()
    model = SentenceTransformer(path, device=dev, model_kwargs=kwargs, local_files_only=True)
    model.max_seq_length = int(task["max_tokens"])
    return model, path


# VRAM guard for every GPU encode and fine-tune (RUN-B, 23 Oct: qwen3_0p6b_ft and qwen3_8b_int8path spilled into
# WDDM system memory and crawled instead of failing). Read-only: a thread logs this process's CUDA memory and stops
# the job when PyTorch's reserved pool passes the card's headroom (free at job start minus a margin), the point at
# which Windows starts to spill.
VRAM_LIMIT_GIB = 14.5            # only when torch.cuda.mem_get_info is not available
VRAM_MARGIN_GIB = 0.5
GUARD_SECONDS = 30.0


CAP_MARGIN_GIB = 1.0            # seq 26: a job sharing the card is capped at its declared VRAM plus this
GAP_GIB = 2.0                   # and the cache is emptied whenever reserved exceeds allocated by more than this


def vram_limit(torch) -> float:
    """The job's VRAM limit. Alone on the card (``UOL_FINAL_GPU_SHARED`` unset or 0): free memory at start minus a
    margin. Sharing the card (seq 26; RUN-B, 23 Oct: gpu__e5_base, declared 3 GB, reserved 11.7 GiB beside
    gpu__bge_m3_sparse and both spilled): its declared ``UOL_FINAL_JOB_VRAM_GB`` plus 1 GiB, enforced on PyTorch's caching
    allocator with ``set_per_process_memory_fraction`` (it frees its cache before passing the cap and raises
    out-of-memory rather than spill; the runner then retries the job alone, uncapped)."""
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:
        return VRAM_LIMIT_GIB
    alone = free / 2 ** 30 - VRAM_MARGIN_GIB
    if os.environ.get("UOL_FINAL_GPU_SHARED") != "1":
        return alone
    cap = min(float(os.environ.get("UOL_FINAL_JOB_VRAM_GB") or 8) + CAP_MARGIN_GIB, alone)
    torch.cuda.set_per_process_memory_fraction(max(0.05, min(1.0, cap * 2 ** 30 / total)))
    RUNTIME["vram_cap_gib"] = round(cap, 2)
    RB.log(f"shared card: allocator capped at {cap:.2f} GiB (declared {os.environ.get('UOL_FINAL_JOB_VRAM_GB')} GB + {CAP_MARGIN_GIB})")
    return cap


def vram_gib(torch) -> tuple[float, float]:
    """(allocated now, peak allocated) of this process, GiB."""
    return torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.max_memory_allocated() / 2 ** 30


def reserved_gib(torch) -> tuple[float, float]:
    """(reserved now, peak reserved) of this process's CUDA caching allocator, GiB."""
    return torch.cuda.memory_reserved() / 2 ** 30, torch.cuda.max_memory_reserved() / 2 ** 30


EMPTY_MARGIN_GIB = 1.0          # seq 25: within this of the limit, the caching allocator's idle blocks are released first
EMPTY_EVERY = 200               # and every 200 batches of an encode (cost: a few ms, then re-allocation of freed blocks)


def empty_cache(torch, where: str) -> None:
    before = reserved_gib(torch)[0]
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    after = reserved_gib(torch)[0]
    RUNTIME["empty_cache_calls"] = RUNTIME.get("empty_cache_calls", 0) + 1
    RB.log(f"empty_cache at {where}: reserved {before:.2f} -> {after:.2f} GiB")


def check_vram(torch, where: str, limit: float = VRAM_LIMIT_GIB) -> None:
    """Stop only on memory in use: reserved near the limit first releases the allocator's idle cached blocks (RUN-B,
    23 Oct, seq 24: qwen3_0p6b_ft_f0 stopped with 14.02 GiB reserved but 2.24 GiB allocated, cache growth over
    changing batch shapes); the job stops if reserved is still over the limit, or allocated alone passes it."""
    res = reserved_gib(torch)[0]
    now = vram_gib(torch)[0]
    if res > limit - EMPTY_MARGIN_GIB or res - now > GAP_GIB:
        empty_cache(torch, where)
        res = reserved_gib(torch)[0]
    now, peak = vram_gib(torch)
    if res > limit or now > limit:
        RB.log(f"VRAM GUARD: {where}: reserved {res:.2f} GiB, allocated {now:.2f} (peak {peak:.2f}) "
               f"> limit {limit:.2f} GiB after emptying the cache; stopping (no fallback crawl)")
        RUNTIME["vram_guard"] = {"where": where, "reserved_gib": round(res, 2), "allocated_gib": round(now, 2),
                                 "limit_gib": round(limit, 2)}
        os._exit(4)


def vram_line(torch, limit: float) -> str:
    now, peak = vram_gib(torch)
    res, res_peak = reserved_gib(torch)
    return f"allocated {now:.2f} GiB (peak {peak:.2f}), reserved {res:.2f} (peak {res_peak:.2f}), limit {limit:.2f}"


def start_guard(torch, phase: dict, stop, limit: float = VRAM_LIMIT_GIB, every: float = GUARD_SECONDS):
    import threading

    def loop():
        while not stop.wait(every):
            RB.log(f"vram {phase.get('name', '?')}: {vram_line(torch, limit)}")
            check_vram(torch, phase.get("name", "?"), limit)
    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


CPU_BATCH_TOKENS, CPU_MAX_INPUTS, CPU_ATTN_CELLS = 4096, 16, 2 ** 21     # per CPU forward pass
# int8 8B on the GPU (RUN-B, 23 Oct: gpu__qwen3_8b_int8path at 16,384-token batches held 15.2 GB of the card plus
# 19.7 GB of shared memory on its longer texts): smaller padded batches; padding is masked, so each vector is the same
INT8_BATCH_TOKENS, INT8_ATTN_CELLS = 4096, 2 ** 22


def main(pins: list[str] | None = None, stack: str | None = None) -> int:
    """``pins``: private installs put before the venv (default ``ST_PINS``: transformers 5.17 / sentence-transformers
    6.1); ``st_encode`` passes only what the venv's own stack lacks for remote-code models (21-22 Oct 2026)."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", choices=("cuda", "cpu"), default=None)
    ap.add_argument("--store-tag", default=None, help="fresh vector store <key>@<tag> (plan, 22 Oct 2026)")
    a, _ = ap.parse_known_args()
    scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", a.out.parent / "scratch"))
    task = json.loads((a.inputs / "task.json").read_text(encoding="utf-8"))
    device = a.device or task.get("device") or "cuda"
    os.environ.setdefault("HF_HOME", str(scratch / "hf"))
    pins = ST_PINS if pins is None else pins
    if pins:
        RB.deps(scratch, pins)
    if task["precision"] == "int8":
        RB.deps(scratch, RB.BNB_PINS)
    RUNTIME["stack"] = stack or ("pinned overlay " + " ".join(pins) if pins == ST_PINS else
                                 "venv stack" + (" + " + " ".join(pins) if pins else ""))
    RUNTIME["device"] = device
    if task.get("device_reason") or device == "cpu":
        RUNTIME["device_reason"] = task.get("device_reason") or (
            f"{task['precision']} on the CPU (task or --device): the same weights, precision and maths as on the GPU; "
            "only the device differs")
    import torch
    vram_limit_gib = vram_limit(torch) if device != "cpu" and torch.cuda.is_available() else float("inf")
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("UOL_FINAL_THREADS") or os.environ.get("OMP_NUM_THREADS") or 8))
        try:                                         # no second thread pool next to the pinned one
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        RUNTIME["threads"] = {"intra_op": torch.get_num_threads(), "inter_op": torch.get_num_interop_threads(),
                              "declared": os.environ.get("UOL_FINAL_THREADS")}
    items = [json.loads(line) for line in (a.inputs / "items.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if task.get("limit"):
        items = items[:int(task["limit"])]
    t_load = time.perf_counter()
    # the device argument only when it is not the default, so two-argument loaders (tests, wrappers) keep working
    model, path = load(task, scratch, device) if device == "cpu" else load(task, scratch)
    RUNTIME["versions"] = stack_versions()
    load_s = time.perf_counter() - t_load
    tok = model.tokenizer
    own = dict(model.prompts or {})
    prompt = {g: (own.get("document" if g == "articles" else "query") or None) if how == "model" else None
              for g, how in task["prompts"].items()}
    n_prompt = {g: len(tok(p, add_special_tokens=False)["input_ids"]) if p else 0 for g, p in prompt.items()}
    work: dict[str, tuple[str | None, str]] = {}          # key -> (prompt, text)
    item_keys, truncated, n_windows, n_split = [], 0, 0, 0
    for it in items:
        g = it["g"]
        if g == "articles":
            texts = windows(it["text"], tok, task["max_tokens"], n_prompt[g])
            n_windows += len(texts)
            n_split += len(texts) > 1
        else:
            t, cut = truncate(it["text"], tok, task["max_tokens"], n_prompt[g])
            truncated += cut
            texts = [t]
        ks = []
        for t in texts:
            k = S.content_key(t, f"{task['model']}|{prompt[g] or ''}")
            work.setdefault(k, (prompt[g], t))
            ks.append(k)
        item_keys.append([it["i"], ks])
    dim = (model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
           else model.get_sentence_embedding_dimension())
    sdir = store_dir(scratch, task["key"], a.store_tag)
    vs = S.VectorStore(sdir, dim)
    todo = vs.missing(list(work))
    RUNTIME["store"] = {"path": str(sdir), "tag": a.store_tag, "resumed_vectors": len(work) - len(todo)}
    RB.log(f"{task['key']}: {len(items)} items, {len(work)} texts, {len(todo)} to encode, prompts {prompt}")
    cuda = torch.cuda.is_available() and device != "cpu"
    todo_set = set(todo)
    resumed = [k for k in work if k not in todo_set]
    if resumed:                                          # a resumed store is checked before it is trusted
        def _enc(texts, pr):
            with torch.inference_mode():
                return model.encode(texts, prompt=pr or "", batch_size=len(texts), convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        chk = check_store(vs, work, resumed, _enc, STORE_CHECK_MIN_COS.get(task["precision"], 0.999))
        RUNTIME["store_check"] = chk
        RB.log(f"store check: {chk}")
        if chk["bad"]:                                   # rebuild: the old store is moved aside (evidence), not deleted
            aside = sdir.with_name(f"store.bad-{time.strftime('%Y%m%dT%H%M%S')}")
            sdir.rename(aside)
            RUNTIME["store_check"].update(rebuilt=True, moved_aside=str(aside))
            vs = S.VectorStore(sdir, dim)
            todo = vs.missing(list(work))
            RB.log(f"store check failed ({chk['bad']} of {chk['n']} below {chk['threshold']}): store moved to {aside}, "
                   f"re-encoding {len(todo)} texts")
        else:
            RUNTIME["store_check"]["rebuilt"] = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
        guard_stop = __import__("threading").Event()
        RB.log(f"vram guard limit {vram_limit_gib:.2f} GiB (free at start minus {VRAM_MARGIN_GIB} GiB)")
        RUNTIME["vram_limit_gib"] = round(vram_limit_gib, 2)
        start_guard(torch, {"name": "encode"}, guard_stop, vram_limit_gib)
    t0, done, n_tok = time.perf_counter(), 0, 0
    budget = float(task.get("budget_seconds") or "inf")
    stopped = False
    for p in dict.fromkeys(work[k][0] for k in todo):
        if stopped:
            break
        keys = [k for k in todo if work[k][0] == p]
        p_n = len(tok(p, add_special_tokens=False)["input_ids"]) if p else 0
        lengths = [len(tok(work[k][1], add_special_tokens=False)["input_ids"]) + p_n for k in keys]
        # CPU (RUN-B, 23 Oct): smaller batches, so the longest texts never form one hours-long forward pass and each
        # batch reaches the store (resume) within minutes; padding is masked, so each text's vector is the same maths
        if device == "cpu":
            cap = dict(max_tokens=min(int(task["batch_tokens"]), CPU_BATCH_TOKENS), max_inputs=CPU_MAX_INPUTS,
                       attn_cells=CPU_ATTN_CELLS)
        elif task["precision"] == "int8":
            cap = dict(max_tokens=min(int(task["batch_tokens"]), INT8_BATCH_TOKENS), max_inputs=64,
                       attn_cells=INT8_ATTN_CELLS)
        else:
            cap = dict(max_tokens=int(task["batch_tokens"]), max_inputs=64)
        RUNTIME["batch_caps"] = cap
        for batch in S.plan_batches(lengths, **cap):
            if time.perf_counter() - t0 > budget:
                stopped = True
                break
            bk = [keys[i] for i in batch]
            if device == "cpu":                      # CPU encodes are slow: one line per batch (RUN-B, 23 Oct)
                RB.log(f"batch of {len(bk)} texts, longest {max(lengths[i] for i in batch)} tokens; {done}/{len(todo)} done")
            with torch.inference_mode():
                # "" (not None): sentence-transformers 6 replaces None with the model's default prompt
                v = model.encode([work[k][1] for k in bk], prompt=p or "", batch_size=len(bk),
                                 convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
            vs.add(bk, np.asarray(v, dtype=np.float32))
            done += len(bk)
            if cuda:
                n_batches = RUNTIME["batches"] = RUNTIME.get("batches", 0) + 1
                if n_batches % EMPTY_EVERY == 0:
                    empty_cache(torch, f"batch {n_batches}")
                check_vram(torch, f"batch of {len(bk)} at {done}/{len(todo)}", vram_limit_gib)
            n_tok += sum(lengths[i] for i in batch)
            if device == "cpu" or done % 2000 < len(bk):
                RB.log(f"{done}/{len(todo)} texts, {n_tok / (time.perf_counter() - t0):.0f} tokens/s")
    flat = [k for _, ks in item_keys for k in ks]
    uniq = list(dict.fromkeys(flat)) if not stopped else []
    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "vectors.npy", vs.get(uniq) if uniq else np.empty((0, dim), dtype=np.float32))
    (a.out / "keys.json").write_text(json.dumps({"items": item_keys if not stopped else [], "keys": uniq}),
                                     encoding="utf-8")
    run = {"task": task, "incomplete": stopped, "remaining": len(vs.missing(list(work))), "snapshot": Path(path).name, "items": len(items), "texts": len(work), "encoded_now": len(todo),
           "dim": dim, "prompts": prompt, "model_prompts": own, "queries_truncated": truncated,
           "queries": sum(1 for it in items if it["g"] != "articles"),
           "article_windows": n_windows, "articles_split": n_split,
           "load_seconds": round(load_s, 1), "encode_seconds": round(time.perf_counter() - t0, 1),
           "tokens_encoded": n_tok, "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20) if cuda else None,
           "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if cuda else "cpu", "threads": torch.get_num_threads(),
           "runtime": RUNTIME}
    (a.out / "run.json").write_text(json.dumps(run, indent=1, ensure_ascii=False), encoding="utf-8")
    return 3 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
