"""The OpenAI API route of the reread (D196): one Batch API job over /v1/responses for the requests of
``uol.final.external``, run in the venv that holds the OpenAI SDK (``.cache/venvs/openai``). The same contract as
``scripts/c10_api.py``.

    <venv>/python scripts/openai_api.py --requests req.jsonl --responses res.jsonl --cache cache.jsonl \
        --ledger ledger.json --cap 60 --prices 2,0.2,10 --env .env [--sync]

* No tools are offered, and nothing is stored on the service (``store: false``).
* Every response is cached by sha256(model, prompt, schema, effort): a request already in the cache is not sent again.
* Spend cap: before submitting, the worst case of the job (prompt tokens estimated from its characters / 2.5, and the
  full ``max_tokens`` as output) plus what the ledger has spent must stay under ``--cap``; otherwise nothing is sent
  (exit 3). The ledger adds the billed usage: ``--prices`` are the standard prices per million tokens (input, cached
  input, output), and the Batch API costs half of them.
* A submitted batch is recorded in ``pending.json`` beside the ledger before it is polled; a rerun after a network
  failure resumes that batch instead of paying for a new one.
* A batch of at most ``SMALL`` requests (the single-case retries) still open after ``SMALL_WAIT_S`` is cancelled; its
  finished answers are kept and the rest go one by one at the standard price, with the cap checked before each.
* ``--sync`` sends every request one by one at the standard price, ``--workers`` at a time.
* The key is read from ``OPENAI_API_KEY`` in the given .env file and handed to the client; it is never printed, logged
  or written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BATCH_DISCOUNT = 0.5
SMALL, SMALL_WAIT_S = 20, 1800.0
OPEN = ("validating", "in_progress", "finalizing", "cancelling")


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


def retry(fn, transient: tuple, tries: int = 40):
    """Network reads are safe to repeat; back off up to 2 minutes between tries."""
    for i in range(tries):
        try:
            return fn()
        except transient as exc:
            print(f"retry {i + 1}: {type(exc).__name__}", flush=True)
            time.sleep(min(10 * (i + 1), 120))
    return fn()


def params(r: dict) -> dict:
    return {"model": r["model"], "input": r["prompt"], "store": False, "max_output_tokens": int(r["max_tokens"]),
            "reasoning": {"effort": r["effort"]},
            "text": {"format": {"type": "json_schema", "name": "reread", "strict": True, "schema": r["schema"]}}}


def price(usage: dict, prices: tuple[float, float, float], factor: float) -> float:
    n_in, n_out = usage.get("input_tokens") or 0, usage.get("output_tokens") or 0
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
    return factor * ((n_in - cached) * prices[0] + cached * prices[1] + n_out * prices[2]) / 1e6


def reply(body: dict) -> tuple[str | None, str]:
    """The answer's text and a stop reason in the terms of ``external.ask_claude``."""
    text, refused = None, False
    for item in body.get("output") or []:
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if c.get("type") == "output_text":
                text = c.get("text")
            elif c.get("type") == "refusal":
                refused = True
    if refused:
        return None, "refusal"
    if body.get("status") == "completed":
        return text, "end_turn"
    reason = (body.get("incomplete_details") or {}).get("reason")
    return text, "max_tokens" if reason == "max_output_tokens" else str(reason or body.get("status"))


class Book:
    """The cache file and the ledger of what was asked and spent."""

    def __init__(self, cache_p: Path, ledger_p: Path, prices: tuple[float, float, float]):
        self.cache_p, self.ledger_p, self.prices = cache_p, ledger_p, prices
        self.lock = threading.Lock()
        self.cache = {}
        if cache_p.exists():
            for x in cache_p.read_text(encoding="utf-8").splitlines():
                if x.strip():
                    c = json.loads(x)
                    self.cache[c["key"]] = c
        self.ledger = json.loads(ledger_p.read_text(encoding="utf-8")) if ledger_p.exists() else \
            {"usd": 0.0, "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
             "batches": [], "requests": []}

    def store(self, body: dict, r: dict, route: str) -> float:
        usage = body.get("usage") or {}
        usd = price(usage, self.prices, BATCH_DISCOUNT if route == "batch" else 1.0)
        text, stop = reply(body)
        cached = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
        rec = {"key": key_of(r), "route": route, "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "text": text, "stop_reason": stop, "model": body.get("model"), "usage": usage}
        with self.lock:
            self.cache[rec["key"]] = rec
            with self.cache_p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            L = self.ledger
            L["usd"] = round(L["usd"] + usd, 4)
            L["usage"]["input_tokens"] += usage.get("input_tokens") or 0
            L["usage"]["cached_input_tokens"] += cached
            L["usage"]["output_tokens"] += usage.get("output_tokens") or 0
            L["usage"]["reasoning_tokens"] += reasoning
            L["requests"].append({"custom_id": r["custom_id"], "key": rec["key"][:16], "route": route, "stop_reason": stop,
                                  "input_tokens": usage.get("input_tokens"), "cached_input_tokens": cached,
                                  "output_tokens": usage.get("output_tokens"), "reasoning_tokens": reasoning,
                                  "usd": round(usd, 5), "utc": rec["utc"]})
        return usd

    def save(self) -> None:
        with self.lock:
            self.ledger_p.write_text(json.dumps(self.ledger, indent=1), encoding="utf-8")

    def worst(self, todo: list[dict], factor: float) -> float:
        p = self.prices
        return factor * sum(len(r["prompt"]) / 2.5 * p[0] + int(r["max_tokens"]) * p[2] for r in todo) / 1e6


def collect(client, batch_id: str, ids: dict[str, dict], book: Book, transient: tuple, poll: float, created: float) -> list[dict]:
    """Wait for the batch and cache its answers; returns the requests left without one."""
    canceled = False
    while True:
        b = retry(lambda: client.batches.retrieve(batch_id), transient)
        if b.status not in OPEN:
            break
        if not canceled and len(ids) <= SMALL and time.time() - created > SMALL_WAIT_S:
            print(f"batch {batch_id}: {len(ids)} requests still open: cancelling", flush=True)
            retry(lambda: client.batches.cancel(batch_id), transient)
            canceled = True
        time.sleep(poll if not canceled else 10.0)
    usd, got = 0.0, set()
    if b.output_file_id:
        lines = retry(lambda: client.files.content(b.output_file_id).text, transient).splitlines()
        for line in lines:
            res = json.loads(line)
            r, resp = ids.get(res["custom_id"]), res.get("response") or {}
            if r is not None and resp.get("status_code") == 200:
                usd += book.store(resp["body"], r, "batch")
                got.add(res["custom_id"])
    failures = []
    if b.error_file_id:
        failures = [json.loads(x) for x in retry(lambda: client.files.content(b.error_file_id).text, transient).splitlines()]
    errors = [e.message for e in (b.errors.data if b.errors else [])][:3]
    book.ledger["batches"].append({"id": batch_id, "requests": len(ids), "answered": len(got), "usd": round(usd, 4),
                                   "status": b.status, "errors": errors, "failed_requests": len(failures),
                                   "first_failure": str((failures[0].get("response") or {}).get("body") or failures[0].get("error"))[:300] if failures else None,
                                   "utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    book.save()
    print(f"batch {batch_id} {b.status}: {len(got)} of {len(ids)} answered, US${usd:.2f}; total US${book.ledger['usd']:.2f}"
          + (f"; errors {errors}" if errors else "") + (f"; {len(failures)} failed" if failures else ""), flush=True)
    return [r for cid, r in ids.items() if cid not in got]


def send_sync(client, todo: list[dict], book: Book, cap: float, transient: tuple, workers: int) -> int:
    """One by one at the standard price; the cap before each."""
    over = threading.Event()

    def one(r: dict) -> None:
        if over.is_set() or key_of(r) in book.cache:
            return
        if book.ledger["usd"] + book.worst([r], 1.0) > cap:
            over.set()
            return
        resp = retry(lambda: client.responses.create(**params(r)), transient, tries=6)
        book.store(resp.model_dump(), r, "sync")

    with ThreadPoolExecutor(max(1, workers)) as pool:
        list(pool.map(one, todo))
    book.save()
    print(f"sync: {len(todo)} requests; total US${book.ledger['usd']:.2f}", flush=True)
    if over.is_set():
        print(f"REFUSED (sync): spent {book.ledger['usd']:.2f} and the next worst case passes the cap {cap:.2f}", flush=True)
        return 3
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    for a in ("requests", "responses", "cache", "ledger", "env", "prices"):
        ap.add_argument(f"--{a}", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--poll", type=float, default=30.0)
    ap.add_argument("--sync", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    prices = tuple(float(x) for x in a.prices.split(","))
    reqs = [json.loads(x) for x in Path(a.requests).read_text(encoding="utf-8").splitlines() if x.strip()]
    ledger_p = Path(a.ledger)
    pending_p = ledger_p.with_name("pending.json")
    book = Book(Path(a.cache), ledger_p, prices)
    todo = [r for r in reqs if key_of(r) not in book.cache]
    if todo or pending_p.exists():
        factor = 1.0 if a.sync else BATCH_DISCOUNT
        worst = book.worst(todo, factor)
        if not a.sync and not pending_p.exists() and book.ledger["usd"] + worst > a.cap:  # --sync: send_sync checks the cap before each request
            print(f"REFUSED: spent {book.ledger['usd']:.2f} + worst case {worst:.2f} > cap {a.cap:.2f}; nothing sent", flush=True)
            return 3
        import openai
        transient = (openai.APITimeoutError, openai.APIConnectionError, openai.InternalServerError, openai.RateLimitError)
        client = openai.OpenAI(api_key=read_env(Path(a.env), "OPENAI_API_KEY"), timeout=600.0, max_retries=5)
        left = []
        if pending_p.exists():
            pend = json.loads(pending_p.read_text(encoding="utf-8"))
            print(f"resuming batch {pend['id']}", flush=True)
            left = collect(client, pend["id"], pend["requests"], book, transient, a.poll, pend["created"])
            pending_p.unlink()
            todo = [r for r in reqs if key_of(r) not in book.cache]
            left = []
        if todo and a.sync:
            left = todo
        elif todo:
            if book.ledger["usd"] + book.worst(todo, BATCH_DISCOUNT) > a.cap:
                print(f"REFUSED: spent {book.ledger['usd']:.2f} and the worst case passes the cap {a.cap:.2f}; nothing sent", flush=True)
                return 3
            ids = {f"r{i}": r for i, r in enumerate(todo)}
            path = ledger_p.with_name("batch_input.jsonl")
            path.write_text("".join(json.dumps({"custom_id": cid, "method": "POST", "url": "/v1/responses", "body": params(r)},
                                               ensure_ascii=False) + "\n" for cid, r in ids.items()), encoding="utf-8")
            with path.open("rb") as f:
                up = client.files.create(file=f, purpose="batch")
            batch = client.batches.create(input_file_id=up.id, endpoint="/v1/responses", completion_window="24h")
            pending_p.write_text(json.dumps({"id": batch.id, "requests": ids, "created": time.time()}, ensure_ascii=False),
                                 encoding="utf-8")
            print(f"batch {batch.id}: {len(ids)} requests, worst case US${book.worst(todo, BATCH_DISCOUNT):.2f}", flush=True)
            left = collect(client, batch.id, ids, book, transient, a.poll, time.time())
            pending_p.unlink()
        if left and send_sync(client, left, book, a.cap, transient, a.workers):
            return 3
    book.save()
    with Path(a.responses).open("w", encoding="utf-8") as f:
        for r in reqs:
            c = book.cache.get(key_of(r))
            f.write(json.dumps({"custom_id": r["custom_id"], **({k: c.get(k) for k in ("text", "stop_reason", "model")}
                                                               if c else {"text": None, "stop_reason": "missing"})},
                               ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
