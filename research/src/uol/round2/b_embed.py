"""B step 5: Qwen3-Embedding-8B vectors (MIT endpoint) for masked contexts and historical memory tesis.

Reuses ``uol.embedding_push.qwen_endpoint`` (client, content-hash store, batching, bisection on HTTP 400) with this
workstream's own store, request log and stop file; at most 2 requests in flight (PLAN rule 5). Document side = plain
text (no instruction), as for the frozen article vectors. Only public census text is sent; no label leaves the machine.

Run: ``py -3.12 -m uol.round2.b_embed [contexts|memory|all]``
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from uol.embedding_push import qwen_endpoint as Q
from uol.round2.b_common import OUT, log, utc, write_json
from uol.round2.b_contexts import CONTEXTS, MEMORY
from uol.round2.b_resolve import compose

STORE = OUT / "store"
REQUESTS = OUT / "requests.jsonl"
PROBES = OUT / "probes.jsonl"
STOP = OUT / "STOP"
WORKERS = 2
MEM_MAX_TOKENS = 2000
EMBED_META = OUT / "embedding_receipt.json"


def context_texts() -> list[str]:
    return [json.loads(l)["text"] for l in CONTEXTS.open(encoding="utf-8") if l.strip()]


def memory_texts(tok) -> list[str]:
    hist = {}
    with (OUT / "historical_tesis.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            hist[r["record_id"]] = r
    ids = [json.loads(l)["record_id"] for l in MEMORY.open(encoding="utf-8") if l.strip()]
    texts = [compose(hist[i]) for i in ids]
    out = []
    for t, enc in zip(texts, tok.encode_batch(texts, add_special_tokens=False)):
        if len(enc.ids) > MEM_MAX_TOKENS:
            t = t[:enc.offsets[MEM_MAX_TOKENS - 1][1]]
        out.append(t)
    return out


def items(texts: list[str], tok) -> list[tuple[str, str, int]]:
    counts = Q.count_tokens(tok, texts)
    return [(Q.content_key(t), t, int(c) + 1) for t, c in zip(texts, counts)]


def seed_probe_reference() -> None:
    """Copy the first-ever probe rows (with full vectors) of the frozen run so drift is measured against them."""
    if PROBES.exists():
        return
    first = {}
    for r in Q.read_jsonl(Q.PROBES_LOG):
        if r.get("kind") in ("doc", "query") and "vec" in r and r["kind"] not in first:
            first[r["kind"]] = r
    for r in first.values():
        Q.append_jsonl(PROBES, {**r, "tag": "reference_from_" + r.get("tag", "")})


def vectors(texts: list[str]) -> np.ndarray:
    """Unit vectors for ``texts`` read *without* opening ``VectorStore`` (whose constructor may truncate a file that a
    running encoder is appending to): parse complete index lines and memory-map the complete vector rows."""
    vec_path, idx_path = STORE / "vectors.f32", STORE / "index.jsonl"
    complete = vec_path.stat().st_size // (Q.DIM * 4)
    rows = {}
    with idx_path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                break
            if e["row"] < complete:
                rows[e["k"]] = e["row"]
    keys = [Q.content_key(t) for t in texts]
    missing = [k for k in keys if k not in rows]
    if missing:
        raise KeyError(f"{len(missing)} texts are not embedded yet")
    mm = np.memmap(vec_path, dtype=np.float32, mode="r", shape=(complete, Q.DIM))
    out = np.asarray(mm[np.fromiter((rows[k] for k in keys), dtype=np.int64, count=len(keys))])
    del mm
    return Q.unit_rows(out)


def main(what: str = "all") -> dict:
    t0 = time.perf_counter()
    tok = Q.load_tokenizer()
    groups = {}
    if what in ("contexts", "all"):
        groups["contexts"] = items(context_texts(), tok)
    if what in ("memory", "all"):
        groups["memory"] = items(memory_texts(tok), tok)
    counts = {k: {"inputs": len(v), "unique": len({x[0] for x in v}), "est_tokens": int(sum(x[2] for x in v))} for k, v in groups.items()}
    log(f"embedding plan: {counts}")
    total = sum(c["unique"] for c in counts.values())
    if total > 150_000:
        raise SystemExit(f"{total} inputs exceed the 150k cap")
    client = Q.Client(Q.EMB_BASE, log_path=REQUESTS)
    served = Q.models_check(client)
    seed_probe_reference()
    start_probe = Q.probe(client, "B_start", PROBES)
    store = Q.VectorStore(STORE)
    summaries = []
    for name, its in groups.items():
        summaries.append(Q.encode_items(client, store, its, workers=WORKERS, label=f"B_{name}", stop_file=STOP, progress=log))
    end_probe = Q.probe(client, "B_end", PROBES)
    rec = {"written_utc": utc(), "endpoint": Q.EMB_BASE, "model": Q.EMB_MODEL, "served": served, "workers": WORKERS,
           "plan": counts, "groups": summaries, "client_stats": client.stats, "start_probe": start_probe,
           "end_probe": end_probe, "store_rows": len(store), "seconds": time.perf_counter() - t0,
           "document_side": "plain text, no instruction; memory tesis truncated to 2,000 Qwen tokens",
           "data_sent": "masked citation contexts and historical tesis texts (public census); no labels"}
    prev = json.loads(EMBED_META.read_text(encoding="utf-8")) if EMBED_META.exists() else {"runs": []}
    prev["runs"].append(rec)
    write_json(EMBED_META, prev)
    log(f"embedding done: {client.stats} ({rec['seconds']:.0f}s)")
    return rec


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
