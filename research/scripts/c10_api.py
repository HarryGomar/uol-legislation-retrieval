"""C10 API step (D101): one Anthropic Message Batch for the requests of ``uol.final.external``, run in the venv that
holds the Anthropic SDK (``.cache/venvs/c10``), on the thesis PC only.

    <venv>/python scripts/c10_api.py --requests req.jsonl --responses res.jsonl --cache cache.jsonl \
        --ledger ledger.json --cap 60 --env .env

* Every response is cached by sha256(model, prompt, schema, effort): a request already in the cache is not sent again.
* Spend cap: before submitting, the worst case of the batch (prompt tokens estimated from its characters / 2.5, and
  the full ``max_tokens`` as output) plus what the ledger has spent must stay under ``--cap``; otherwise nothing is
  sent (exit 3). After the batch, the ledger adds the billed usage at the Batch API price.
* A submitted batch is recorded in ``pending.json`` beside the ledger before it is polled; a rerun after a network
  failure resumes that batch instead of submitting (and paying for) a new one. Polling and result reads are retried.
* D107 (23 Oct): a batch of at most ``SMALL`` requests (the single-case retries) that has not ended ``SMALL_WAIT_S``
  after it was created is cancelled; once the cancellation has ended, its succeeded results are collected and cached,
  and only the requests without a result go one by one through the Messages API (same model, prompt, effort, schema
  and max_tokens) at the standard price, with the cap checked before each. The large windows stay on the Batch API.
* The ledger records every request: its route (batch or sync), tokens and price.
* The key is read from ``ANTHROPIC_KEY`` in the given .env file and handed to the client; it is never printed, logged
  or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

USD_IN, USD_OUT = 2.0, 10.0          # per million tokens, Batch API (50 % of Claude Opus 5.5's $4 / $20)
SYNC_IN, SYNC_OUT = 4.0, 20.0        # per million tokens, Messages API (standard price; D107 fallback only)
SMALL, SMALL_WAIT_S = 20, 1800.0     # D107: retry batches of at most 20 requests wait at most 30 min


def key_of(r: dict) -> str:
    """Cache key; an independent re-sample (``sample`` > 0) has its own key."""
    parts = [r["model"], r["prompt"], r["schema"], r["effort"]] + ([int(r["sample"])] if r.get("sample") else [])
    return hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_env(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip('"').strip("'")
    raise SystemExit(f"{name} not set in {path}")


def worst_usd(r: dict, usd_in: float = USD_IN, usd_out: float = USD_OUT) -> float:
    return len(r["prompt"]) / 2.5 * usd_in / 1e6 + int(r["max_tokens"]) * usd_out / 1e6


def retry(fn, transient: tuple, tries: int = 40):
    """Network reads (status, results) are safe to repeat; back off up to 2 minutes between tries."""
    for i in range(tries):
        try:
            return fn()
        except transient as exc:
            print(f"retry {i + 1}: {type(exc).__name__}", flush=True)
            time.sleep(min(10 * (i + 1), 120))
    return fn()


def params(r: dict) -> dict:
    return dict(model=r["model"], max_tokens=int(r["max_tokens"]),
                output_config={"effort": r["effort"], "format": {"type": "json_schema", "schema": r["schema"]}},
                messages=[{"role": "user", "content": r["prompt"]}])


def usage_of(u) -> tuple[int, int]:
    n_in = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) + \
        (getattr(u, "cache_read_input_tokens", 0) or 0)
    return n_in, u.output_tokens or 0


def store(m, r: dict, route: str, cache: dict, cache_p: Path, ledger: dict) -> float:
    """Cache one answer and add it to the ledger (route, tokens, price); returns its price."""
    n_in, n_out = usage_of(m.usage)
    pin, pout = (USD_IN, USD_OUT) if route == "batch" else (SYNC_IN, SYNC_OUT)
    usd = n_in * pin / 1e6 + n_out * pout / 1e6
    rec = {"key": key_of(r), "type": "succeeded", "route": route, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "text": next((blk.text for blk in m.content if blk.type == "text"), None),
           "stop_reason": m.stop_reason, "model": m.model, "usage": {"input_tokens": n_in, "output_tokens": n_out}}
    cache[rec["key"]] = rec
    with cache_p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    ledger["usd"] = round(ledger["usd"] + usd, 4)
    ledger["usage"]["input_tokens"] += n_in
    ledger["usage"]["output_tokens"] += n_out
    ledger.setdefault("requests", []).append({"custom_id": r["custom_id"], "key": rec["key"][:16], "route": route,
                                              "input_tokens": n_in, "output_tokens": n_out, "usd": round(usd, 5),
                                              "price_per_mtok": [pin, pout], "utc": rec["utc"]})
    return usd


def collect(client, batch_id: str, ids: dict[str, dict], cache: dict, cache_p: Path, ledger: dict, transient: tuple,
            poll: float, created: float | None = None) -> list[dict]:
    """Wait for the batch, cache its succeeded results; returns the requests left without a result. A small batch past
    ``SMALL_WAIT_S`` is cancelled first (D107)."""
    canceled = False
    while True:
        b = retry(lambda: client.messages.batches.retrieve(batch_id), transient)
        if b.processing_status == "ended":
            break
        age = time.time() - (created if created is not None else time.time())
        if not canceled and len(ids) <= SMALL and age > SMALL_WAIT_S:
            print(f"batch {batch_id}: {len(ids)} requests, {age / 60:.0f} min without ending: cancelling (D107)", flush=True)
            retry(lambda: client.messages.batches.cancel(batch_id), transient)
            canceled = True
        time.sleep(poll if not canceled else 10.0)
    results = retry(lambda: list(client.messages.batches.results(batch_id)), transient)
    usd, got = 0.0, set()
    for res in results:
        r = ids[res.custom_id]
        if res.result.type != "succeeded":             # errored / canceled / expired: not cached
            continue
        usd += store(res.result.message, r, "batch", cache, cache_p, ledger)
        got.add(res.custom_id)
    counts = getattr(b, "request_counts", None)
    ledger["batches"].append({"id": batch_id, "requests": len(ids), "succeeded": len(got), "usd": round(usd, 4),
                              "canceled": canceled, "counts": str(counts) if counts is not None else None,
                              "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    print(f"batch {batch_id} ended{' (cancelled)' if canceled else ''}: {len(got)} of {len(ids)} answered, US${usd:.2f}; "
          f"total US${ledger['usd']:.2f}", flush=True)
    return [r for cid, r in ids.items() if cid not in got] if canceled else []


def send_sync(client, todo: list[dict], cache: dict, cache_p: Path, ledger: dict, ledger_p: Path, cap: float,
              transient: tuple) -> int:
    """D107: the requests of a cancelled retry batch, one by one at the standard price; the cap before each."""
    for r in todo:
        if key_of(r) in cache:                         # answered meanwhile (never collected and re-asked)
            continue
        worst = worst_usd(r, SYNC_IN, SYNC_OUT)
        if ledger["usd"] + worst > cap:
            print(f"REFUSED (sync): spent {ledger['usd']:.2f} + worst case {worst:.2f} > cap {cap:.2f}", flush=True)
            return 3
        m = retry(lambda: client.messages.create(**params(r)), transient, tries=6)
        usd = store(m, r, "sync", cache, cache_p, ledger)
        ledger_p.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
        print(f"sync {r['custom_id']}: {m.stop_reason}, US${usd:.3f}; total US${ledger['usd']:.2f}", flush=True)
    return 0


def main() -> int:
    global USD_IN, USD_OUT, SYNC_IN, SYNC_OUT
    ap = argparse.ArgumentParser()
    for a in ("requests", "responses", "cache", "ledger", "env"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--poll", type=float, default=30.0)
    # LLM sweep (D174, 28 Oct 2026): the model's own prices per million tokens; the defaults are Claude Opus 5.5's
    ap.add_argument("--price-batch", default=f"{USD_IN},{USD_OUT}")
    ap.add_argument("--price-sync", default=f"{SYNC_IN},{SYNC_OUT}")
    a = ap.parse_args()
    USD_IN, USD_OUT = (float(x) for x in a.price_batch.split(","))
    SYNC_IN, SYNC_OUT = (float(x) for x in a.price_sync.split(","))
    reqs = [json.loads(x) for x in Path(a.requests).read_text(encoding="utf-8").splitlines() if x.strip()]
    cache_p, ledger_p = Path(a.cache), Path(a.ledger)
    pending_p = ledger_p.with_name("pending.json")
    cache = {}
    if cache_p.exists():
        for x in cache_p.read_text(encoding="utf-8").splitlines():
            if x.strip():
                c = json.loads(x)
                cache[c["key"]] = c
    ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.exists() else \
        {"usd": 0.0, "usage": {"input_tokens": 0, "output_tokens": 0}, "batches": []}
    if "requests" not in ledger:                       # answers cached before D107 all came from batches
        ledger["requests"] = [{"custom_id": None, "key": k[:16], "route": c.get("route", "batch"),
                               "input_tokens": c["usage"]["input_tokens"], "output_tokens": c["usage"]["output_tokens"],
                               "usd": round(c["usage"]["input_tokens"] * USD_IN / 1e6 + c["usage"]["output_tokens"] * USD_OUT / 1e6, 5),
                               "price_per_mtok": [USD_IN, USD_OUT], "utc": c.get("utc")}
                              for k, c in cache.items() if c.get("usage")]
    todo = [r for r in reqs if key_of(r) not in cache]
    if todo or pending_p.exists():
        worst = sum(worst_usd(r, USD_IN, USD_OUT) for r in todo)
        if not pending_p.exists() and ledger["usd"] + worst > a.cap:
            print(f"REFUSED: spent {ledger['usd']:.2f} + worst case {worst:.2f} > cap {a.cap:.2f}; nothing sent")
            return 3
        import anthropic
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request
        transient = (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.InternalServerError,
                     anthropic.RateLimitError)
        client = anthropic.Anthropic(api_key=read_env(Path(a.env), "ANTHROPIC_KEY"), timeout=120.0, max_retries=5)
        if pending_p.exists():                                  # an earlier run submitted and then lost the connection
            pend = json.loads(pending_p.read_text(encoding="utf-8"))
            created = pend.get("created", pending_p.stat().st_mtime)
            print(f"resuming batch {pend['id']}", flush=True)
            left = collect(client, pend["id"], pend["requests"], cache, cache_p, ledger, transient, a.poll, created)
            ledger_p.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
            pending_p.unlink()
            if left and send_sync(client, left, cache, cache_p, ledger, ledger_p, a.cap, transient):
                return 3
            todo = [r for r in reqs if key_of(r) not in cache]
            worst = sum(worst_usd(r, USD_IN, USD_OUT) for r in todo)
            if todo and ledger["usd"] + worst > a.cap:
                print(f"REFUSED: spent {ledger['usd']:.2f} + worst case {worst:.2f} > cap {a.cap:.2f}; nothing sent")
                return 3
        if todo:
            ids = {f"r{i}": r for i, r in enumerate(todo)}
            batch_reqs = [Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**params(r))) for cid, r in ids.items()]
            batch = client.messages.batches.create(requests=batch_reqs)
            pending_p.write_text(json.dumps({"id": batch.id, "requests": ids, "created": time.time()}, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"batch {batch.id}: {len(batch_reqs)} requests, worst case US${worst:.2f}", flush=True)
            left = collect(client, batch.id, ids, cache, cache_p, ledger, transient, a.poll, time.time())
            ledger_p.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
            pending_p.unlink()
            if left and send_sync(client, left, cache, cache_p, ledger, ledger_p, a.cap, transient):
                return 3
    with Path(a.responses).open("w", encoding="utf-8") as f:
        for r in reqs:
            c = cache.get(key_of(r))
            f.write(json.dumps({"custom_id": r["custom_id"], **({k: c.get(k) for k in ("text", "stop_reason", "model")}
                                                               if c else {"text": None, "stop_reason": "missing"})},
                               ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
