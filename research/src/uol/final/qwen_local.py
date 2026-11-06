"""Local Qwen3-Embedding-8B in int8 on the RTX 5080: the fidelity-gated replacement of the MIT endpoint for new
query-side vectors on the run computer (DEV-N, DEV-B recovered cases, NEW, query2doc, HyDE, the agent's tool).

The model is loaded exactly as the legal-encoder job loads it (``round3/gpu_jobs/legal_encode.load``: pinned
revision, bitsandbytes int8, sentence-transformers) with the Gate E0 spec of ``scale_curve.legal.GATE``; inputs are
the same strings the endpoint embedded (``qwen_endpoint.query_input`` for queries; bare text for documents);
vectors are L2-normalised. It is used only after Gate E0 and the RUN_PLAN §6 fidelity check passed (cos >= .99;
dev nDCG@10 of qwen3_8b within .003 with int8 query vectors).

``patch_endpoint`` swaps ``qwen_endpoint.Client`` / ``encode_items`` / ``load_tokenizer`` in this process so code
that would call the endpoint (the E6 agent's embedding tool) encodes locally instead; no network is used.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context

_M: dict[str, Any] = {}


def _jobs_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "round3" / "gpu_jobs"


def model():
    if "m" not in _M:
        sys.path.insert(0, str(_jobs_dir()))
        import legal_encode as LE                      # noqa: E402  (ships rerank_bench; private pinned installs)
        from uol.scale_curve import legal
        task = {**legal.GATE["qwen3_8b_gate"], "key": "qwen3_8b_local"}
        scratch = Path(os.environ.get("UOL_WORKER_SCRATCH", context.active().work / "gpu_scratch"))
        os.environ.setdefault("HF_HOME", str(scratch / "hf"))
        LE.RB.deps(scratch, LE.ST_PINS)
        LE.RB.deps(scratch, LE.RB.BNB_PINS)
        m, path = LE.load(task, scratch)
        _M["m"], _M["path"] = m, Path(path)
    return _M["m"]


def encode(texts: Sequence[str], batch: int = 16) -> np.ndarray:
    m = model()
    v = m.encode(list(texts), prompt="", batch_size=batch, convert_to_numpy=True, normalize_embeddings=True,
                 show_progress_bar=False)
    return np.asarray(v, dtype=np.float32)


def patch_endpoint() -> None:
    from uol.embedding_push import qwen_endpoint as qe

    class _NoClient:
        def __init__(self, *a, **k):
            pass

    def encode_items(client, store, items, **kw):
        missing = [(k, t) for k, t, _ in items if k not in store]
        uniq = list(dict(missing).items())
        for s in range(0, len(uniq), 64):
            part = uniq[s:s + 64]
            store.add([k for k, _ in part], encode([t for _, t in part]))
        return {"encoded_locally": len(uniq)}

    def load_tokenizer(directory=None):
        from tokenizers import Tokenizer
        model()
        return Tokenizer.from_file(str(_M["path"] / "tokenizer.json"))

    qe.Client, qe.encode_items, qe.load_tokenizer = _NoClient, encode_items, load_tokenizer
    context.active().patches["uol.embedding_push.qwen_endpoint.encode_items"] = "local Qwen3-8B int8 (uol.final.qwen_local)"
