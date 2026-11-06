"""EP-epsilon: Qwen3-Embedding-8B vectors from the MIT cluster endpoint and the methods built on them.

Authority: DEVIATIONS.md D3 (user-authorised MIT endpoints, public text only, no labels, no API key).
Only two OpenAI-compatible endpoints are ever called:

* ``EMB_BASE`` (``Qwen/Qwen3-Embedding-8B``, ``POST /embeddings``) for every vector, and
* ``INSTRUCT_BASE`` (``inclusionAI/LLaDA2.1-mini``, chat completions) for the optional HyDE rewrite (step 4).

Sections
--------
1. endpoint client (batches of at most 64, at most 4 concurrent requests, timeouts, exponential back-off),
   append-only content-hash vector store (a restart never re-sends a text), article windows and query variants,
   probe vectors, receipt and assembly of the float32 ``.npy`` files;
2. dense scores (MaxP over article windows), variant selection and the ``dense_qwen8b`` / ``H_qwen8b`` files;
3. cross-fitted supervised methods (``knn_qwen8b``, ``knn_fused_qwen8b``, ``proto_qwen8b``) with nested inner CV;
4. optional low-rank query adapter and the optional HyDE rewrite.

Run: ``python -m uol.embedding_push.qwen_endpoint <command>`` (see ``main``).
Only masked query texts (``q.text``) and public candidate article texts are ever sent; labels never leave the machine.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import requests

from uol.embedding_push import common
from uol.facts_to_law import fusion, knn, lexical, metrics
from uol.facts_to_law.spec import ROOT

# --------------------------------------------------------------------------- constants

# The thesis used an OpenAI-compatible server for the first vectors; set the base URL of your own (for example
# vLLM serving Qwen/Qwen3-Embedding-8B). The final system encodes locally (uol.final.qwen_local).
EMB_BASE = os.environ.get("VLLM_EMBEDDINGS_BASE_URL", "http://localhost:8000/embeddings/v1")
INSTRUCT_BASE = os.environ.get("VLLM_INSTRUCT_BASE_URL", "http://localhost:8001/instruct/v1")
EMB_MODEL = "Qwen/Qwen3-Embedding-8B"
LLM_MODEL = "inclusionAI/LLaDA2.1-mini"
EMB_MAX_MODEL_LEN = 8192
DIM = 4096
MAX_BATCH = 64                     # inputs per request (brief)
MAX_CONCURRENCY = 4                # concurrent requests (brief)
TOKEN_CAP = 48_000                 # additional politeness cap on estimated tokens per request
ART_MAX_TOKENS = 8000              # longer articles are split into windows
WINDOW_CONTENT_TOKENS = 7800       # body tokens per window; the title prefix (<= TITLE_MAX_TOKENS) stays below 8,000 in total
WINDOW_OVERLAP = 64
TITLE_MAX_TOKENS = 100

TOKENIZER_DIR = Path(r"C:\uol_legacy_home\.cache\huggingface\hub\models--Qwen--Qwen3-Embedding-8B"
                     r"\snapshots\1d8ad4ca9b3dd8059ad90a75d4983776a23d44af")
TOKENIZER_REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
TOKENIZER_JSON_SHA256 = "83cdf8c3a34f68862319cb1810ee7b1e2c0a44e0864ae930194ddb76bb7feb8d"
# The first path was emptied by another process on 2026-10-04 ~17:23-17:32Z; the same revision (identical sha256) survives in the others.
TOKENIZER_DIRS = (
    TOKENIZER_DIR,
    Path(r"C:\uol_legacy_home\experiments\runtime\huggingface\hub\models--Qwen--Qwen3-Embedding-8B\snapshots")
    / TOKENIZER_REVISION,
    Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-Embedding-8B/snapshots" / TOKENIZER_REVISION,
)
SEPT_CLIENT = Path(r"C:\uol_legacy_home\src\uol\retrieval_benchmarks\semantic_n06_dense.py")

# Query variants (the instruction is part of the embedded string; documents are always plain text).
INSTRUCTION_B = ("Given the facts of a Mexican court decision, retrieve the federal statutory articles the court applied")
# Verbatim QUERY_INSTRUCTION of semantic_n06_dense.py (September client); test_epsilon_endpoint checks it against the file.
INSTRUCTION_C = ("Dada una descripción publicada de hechos, recupera el criterio jurídico "
                 "publicado que pertenece documentalmente a la misma ficha.")
VARIANTS = {"a": None, "b": INSTRUCTION_B, "c": INSTRUCTION_C}
VARIANT_LABEL = {"a": "no instruction", "b": "task instruction (English, statutory articles)",
                 "c": "September QUERY_INSTRUCTION (Spanish, tesis)"}

PROBE_DOC = ("ARTICULO 14.- A ninguna ley se dará efecto retroactivo en perjuicio de persona alguna. Nadie podrá ser "
             "privado de la libertad o de sus propiedades, posesiones o derechos, sino mediante juicio seguido ante los "
             "tribunales previamente establecidos.")
PROBE_QUERY = ("Una persona promovió juicio de amparo contra la sentencia que confirmó la condena y alegó que la autoridad "
               "responsable aplicó una norma sin que existiera un procedimiento previo.")

DERIVED = ROOT / "data/derived/embedding_push_qwen8b_v1"
RUN = common.RUN / "qwen"
STORE_DIR = DERIVED / "store"
RUN_LOG = RUN / "run.log"
REQUEST_LOG = RUN / "requests.jsonl"
RUNS_LOG = RUN / "runs.jsonl"
PROBES_LOG = DERIVED / "probes.jsonl"
STOP_FILE = RUN / "STOP"

RETRY_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    line = f"[{utc()}] {message}"
    try:
        print(line, flush=True)
    except UnicodeEncodeError:  # Windows consoles
        print(line.encode("ascii", "replace").decode(), flush=True)
    RUN.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # a torn last line after a crash
                continue
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def content_key(text: str, model: str = EMB_MODEL) -> str:
    """Cache key of one embedded string: sha256(model, NUL, text)."""
    return hashlib.sha256(model.encode("utf-8") + b"\x00" + text.encode("utf-8", "surrogatepass")).hexdigest()


def keys_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest()


def unit_rows(x: np.ndarray) -> np.ndarray:
    """L2-normalised float32 copy (zero rows stay zero)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


# --------------------------------------------------------------------------- 1. endpoint client


class EndpointError(RuntimeError):
    """A request that failed for good (non-retriable status, or retries exhausted)."""

    def __init__(self, status: int | None, message: str) -> None:
        super().__init__(f"endpoint error status={status}: {message}")
        self.status = status


class Client:
    """Polite JSON client: timeouts, exponential back-off with jitter, per-request log, thread-local sessions.

    The caller is responsible for the concurrency cap (``encode_items`` never runs more than ``MAX_CONCURRENCY``
    requests at once). No credential is ever read, sent or stored: the cluster endpoints need none.
    """

    def __init__(self, base: str, *, timeout: tuple[float, float] = (20.0, 600.0), attempts: int = 8, backoff: float = 2.0,
                 max_sleep: float = 120.0, log_path: Path | None = REQUEST_LOG, sleep: Callable[[float], None] = time.sleep,
                 session_factory: Callable[[], Any] = requests.Session) -> None:
        self.base = base.rstrip("/")
        self.timeout, self.attempts, self.backoff, self.max_sleep = timeout, attempts, backoff, max_sleep
        self.log_path, self._sleep, self._session_factory = log_path, sleep, session_factory
        self._local = threading.local()
        self._lock = threading.Lock()
        self.stats = {"requests": 0, "attempts": 0, "retries": 0, "failures": 0, "inputs": 0, "prompt_tokens": 0,
                      "completion_tokens": 0, "seconds_in_requests": 0.0}

    def _session(self) -> Any:
        if not hasattr(self._local, "session"):
            self._local.session = self._session_factory()
        return self._local.session

    def _count(self, **delta: float) -> None:
        with self._lock:
            for key, value in delta.items():
                self.stats[key] += value

    def _log(self, row: dict[str, Any]) -> None:
        if self.log_path is not None:
            with self._lock:
                append_jsonl(self.log_path, {"utc": utc(), **row})

    def delay(self, attempt: int, retry_after: float | None = None) -> float:
        wait = min(self.max_sleep, self.backoff * 2 ** (attempt - 1)) * (0.75 + 0.5 * random.random())
        return max(wait, retry_after or 0.0)

    def get(self, route: str, attempts: int = 4) -> dict[str, Any]:
        for attempt in range(1, attempts + 1):
            try:
                response = self._session().get(self.base + route, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                error = f"HTTP{response.status_code}"
                if response.status_code not in RETRY_STATUS:
                    raise EndpointError(response.status_code, response.text[:200])
            except (requests.RequestException, ValueError) as exc:
                error = type(exc).__name__
            if attempt == attempts:
                raise EndpointError(None, f"GET {route} gave up after {attempt} attempts: {error}")
            self._sleep(self.delay(attempt))
        raise AssertionError("unreachable")

    def post(self, route: str, payload: dict[str, Any], *, tag: str = "", n_inputs: int = 0, est_tokens: int = 0,
             validate: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8", "surrogatepass")
        self._count(requests=1, inputs=n_inputs)
        for attempt in range(1, self.attempts + 1):
            started = time.monotonic()
            status, error, retry_after, data = None, None, None, None
            try:
                response = self._session().post(self.base + route, data=body, timeout=self.timeout,
                                                headers={"Content-Type": "application/json"})
                status = response.status_code
                if status == 200:
                    try:
                        data = response.json()
                        if validate is not None:
                            validate(data)
                    except (ValueError, KeyError, TypeError) as exc:
                        data, error = None, f"BadResponse:{type(exc).__name__}"
                else:
                    error = f"HTTP{status}"
                    header = response.headers.get("Retry-After", "")
                    retry_after = float(header) if header.replace(".", "", 1).isdigit() else None
                    if status not in RETRY_STATUS:
                        error += ":" + response.text[:200].replace("\n", " ")
            except requests.RequestException as exc:
                error = type(exc).__name__
            latency = time.monotonic() - started
            self._count(attempts=1, seconds_in_requests=latency)
            usage = (data or {}).get("usage") or {}
            self._log({"route": route, "tag": tag, "n_inputs": n_inputs, "est_tokens": est_tokens, "attempt": attempt,
                       "status": status, "error": error, "latency_s": round(latency, 3),
                       "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
            if data is not None:
                self._count(prompt_tokens=usage.get("prompt_tokens") or 0, completion_tokens=usage.get("completion_tokens") or 0)
                return data
            if status is not None and status != 200 and status not in RETRY_STATUS:
                self._count(failures=1)
                raise EndpointError(status, error or "")
            if attempt == self.attempts:
                self._count(failures=1)
                raise EndpointError(status, f"gave up after {attempt} attempts: {error}")
            self._count(retries=1)
            self._sleep(self.delay(attempt, retry_after))
        raise AssertionError("unreachable")

    def embed(self, texts: Sequence[str], *, tag: str = "", est_tokens: int = 0) -> tuple[np.ndarray, dict[str, Any]]:
        """Float32 ``[n, DIM]`` vectors (as served, not renormalised) and the usage block for one request."""
        n = len(texts)
        if not 1 <= n <= MAX_BATCH:
            raise ValueError(f"a request holds 1..{MAX_BATCH} inputs, got {n}")

        def check(data: dict[str, Any]) -> None:
            rows = sorted(data["data"], key=lambda r: r["index"])
            if [r["index"] for r in rows] != list(range(n)):
                raise ValueError("embedding indices are missing or repeated")
            values = np.asarray([r["embedding"] for r in rows], dtype=np.float32)
            if values.shape != (n, DIM) or not np.isfinite(values).all() or (np.linalg.norm(values, axis=1) < 1e-6).any():
                raise ValueError("invalid embedding response")

        data = self.post("/embeddings", {"model": EMB_MODEL, "input": list(texts), "encoding_format": "float"}, tag=tag,
                         n_inputs=n, est_tokens=est_tokens, validate=check)
        rows = sorted(data["data"], key=lambda r: r["index"])
        return np.asarray([r["embedding"] for r in rows], dtype=np.float32), {**(data.get("usage") or {}), "model": data.get("model")}


# --------------------------------------------------------------------------- 1. content-hash vector store


class VectorStore:
    """Append-only cache ``key -> float32[DIM]``: ``vectors.f32`` rows and ``index.jsonl`` (key, row).

    Vectors are written (and flushed) before their index lines, so a crash can only leave unreferenced trailing rows,
    which the next open truncates. Rows are the vectors as served by the endpoint.
    """

    def __init__(self, root: Path = STORE_DIR) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vec_path, self.idx_path = self.root / "vectors.f32", self.root / "index.jsonl"
        self.rows: dict[str, int] = {}
        size = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        complete = size // (DIM * 4)
        keep: list[dict[str, Any]] = []
        for entry in read_jsonl(self.idx_path):
            if entry.get("row") == len(keep) and entry["row"] < complete and entry["k"] not in self.rows:
                self.rows[entry["k"]] = entry["row"]
                keep.append(entry)
            else:
                break
        if len(keep) != len(read_jsonl(self.idx_path)) or size != len(keep) * DIM * 4:
            with open(self.vec_path, "ab"):
                pass
            os.truncate(self.vec_path, len(keep) * DIM * 4)
            self.idx_path.write_text("".join(json.dumps(e, sort_keys=True) + "\n" for e in keep), encoding="utf-8")
        self.n = len(keep)

    def __len__(self) -> int:
        return self.n

    def __contains__(self, key: str) -> bool:
        return key in self.rows

    def missing(self, keys: Iterable[str]) -> list[str]:
        seen, out = set(), []
        for key in keys:
            if key not in self.rows and key not in seen:
                seen.add(key)
                out.append(key)
        return out

    def add(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.shape != (len(keys), DIM) or len(set(keys)) != len(keys):
            raise ValueError("one distinct key per vector row is required")
        fresh = [i for i, key in enumerate(keys) if key not in self.rows]
        if not fresh:
            return
        with open(self.vec_path, "ab") as handle:
            handle.write(vectors[fresh].tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        with open(self.idx_path, "a", encoding="utf-8", newline="\n") as handle:
            for offset, i in enumerate(fresh):
                self.rows[keys[i]] = self.n + offset
                handle.write(json.dumps({"k": keys[i], "row": self.n + offset}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.n += len(fresh)

    def get(self, keys: Sequence[str]) -> np.ndarray:
        """Float32 ``[len(keys), DIM]`` in the order of ``keys`` (all must be present)."""
        absent = [k for k in keys if k not in self.rows]
        if absent:
            raise KeyError(f"{len(absent)} keys are not in the store, e.g. {absent[0][:12]}")
        if not keys:
            return np.empty((0, DIM), dtype=np.float32)
        mm = np.memmap(self.vec_path, dtype=np.float32, mode="r", shape=(self.n, DIM))
        out = np.asarray(mm[np.fromiter((self.rows[k] for k in keys), dtype=np.int64, count=len(keys))])
        del mm
        return out


# --------------------------------------------------------------------------- 1. units: article windows and query variants


def window_spans(n_tokens: int, content: int = WINDOW_CONTENT_TOKENS, overlap: int = WINDOW_OVERLAP) -> list[tuple[int, int]]:
    """Token spans [start, end) of windows of at most ``content`` tokens sharing ``overlap`` tokens; one span when it fits."""
    if content <= overlap:
        raise ValueError("window content must exceed the overlap")
    if n_tokens <= content:
        return [(0, n_tokens)]
    spans, start, step = [], 0, content - overlap
    while True:
        end = min(start + content, n_tokens)
        spans.append((start, end))
        if end >= n_tokens:
            return spans
        start += step


@dataclass(frozen=True)
class Unit:
    """One embedded string: an article (or article window) or a query variant."""

    owner: int            # candidate index (articles) or release row (queries)
    window: int           # window number within the article (0 for whole articles and queries)
    n_windows: int
    text: str
    est_tokens: int       # local Qwen token count of ``text`` plus one end-of-text token


def tokenizer_path(directory: Path | None = None) -> Path:
    """First existing ``tokenizer.json`` of the pinned Qwen3-Embedding-8B snapshot (sha256 ``TOKENIZER_JSON_SHA256``)."""
    for d in ([Path(directory)] if directory else TOKENIZER_DIRS):
        if (d / "tokenizer.json").exists():
            return d / "tokenizer.json"
    raise FileNotFoundError(f"tokenizer.json of Qwen3-Embedding-8B snapshot {TOKENIZER_REVISION} not found in "
                            + ", ".join(map(str, TOKENIZER_DIRS)))


def load_tokenizer(directory: Path | None = None) -> Any:
    """The cached Qwen3-Embedding-8B fast tokenizer (tokenizer files only; no model weights are read)."""
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tokenizer_path(directory)))


def count_tokens(tokenizer: Any, texts: Sequence[str]) -> np.ndarray:
    return np.asarray([len(e.ids) for e in tokenizer.encode_batch(list(texts), add_special_tokens=False)], dtype=np.int64)


def article_units(texts: Sequence[str], tokenizer: Any, *, max_tokens: int = ART_MAX_TOKENS, content: int = WINDOW_CONTENT_TOKENS,
                  overlap: int = WINDOW_OVERLAP, title_max: int = TITLE_MAX_TOKENS) -> list[Unit]:
    """One unit per article, or MaxP windows for articles above ``max_tokens``.

    Window 0 is the article's first ``content`` tokens (it already starts with the instrument title line). Later windows
    are the title line (at most ``title_max`` tokens) + newline + their body span, cut on token offsets so the text is an
    exact substring of the article. Each window therefore holds at most ``content + title_max + 2`` (< 8,000) tokens.
    """
    counts = count_tokens(tokenizer, texts)
    units: list[Unit] = []
    for i, text in enumerate(texts):
        if counts[i] <= max_tokens:
            units.append(Unit(i, 0, 1, text, int(counts[i]) + 1))
            continue
        enc = tokenizer.encode(text, add_special_tokens=False)
        offsets = enc.offsets
        title = text.split("\n", 1)[0].strip()
        t_enc = tokenizer.encode(title, add_special_tokens=False)
        if len(t_enc.ids) > title_max:
            title = title[:t_enc.offsets[title_max - 1][1]]
            t_enc = tokenizer.encode(title, add_special_tokens=False)
        spans = window_spans(len(enc.ids), content, overlap)
        for w, (s, e) in enumerate(spans):
            body = text[offsets[s][0]:offsets[e - 1][1]]
            if w == 0:
                units.append(Unit(i, 0, len(spans), body, e - s + 1))
            else:
                units.append(Unit(i, w, len(spans), title + "\n" + body, (e - s) + len(t_enc.ids) + 2))
    return units


def query_input(variant: str, text: str) -> str:
    """The string embedded for a query under a variant: bare text (a) or ``Instruct: ..\\nQuery:..`` (b, c)."""
    instruction = VARIANTS[variant]
    return text if instruction is None else f"Instruct: {instruction}\nQuery:{text}"


def query_units(texts: Sequence[str], tokenizer: Any, variant: str) -> list[Unit]:
    inputs = [query_input(variant, t) for t in texts]
    counts = count_tokens(tokenizer, inputs)
    return [Unit(i, 0, 1, s, int(counts[i]) + 1) for i, s in enumerate(inputs)]


def make_batches(items: Sequence[tuple[str, str, int]], max_inputs: int = MAX_BATCH, token_cap: int = TOKEN_CAP
                 ) -> list[list[tuple[str, str, int]]]:
    """Batches of (key, text, est_tokens): at most ``max_inputs`` inputs and ``token_cap`` estimated tokens each.

    Items are sorted by estimated length (ties by key) so every request holds similar lengths; one oversized item is
    sent alone.
    """
    ordered = sorted(items, key=lambda it: (it[2], it[0]))
    batches: list[list[tuple[str, str, int]]] = []
    current: list[tuple[str, str, int]] = []
    tokens = 0
    for item in ordered:
        if current and (len(current) >= max_inputs or tokens + item[2] > token_cap):
            batches.append(current)
            current, tokens = [], 0
        current.append(item)
        tokens += item[2]
    if current:
        batches.append(current)
    return batches


def encode_items(client: Client, store: VectorStore, items: Sequence[tuple[str, str, int]], *, workers: int = MAX_CONCURRENCY,
                 max_inputs: int = MAX_BATCH, token_cap: int = TOKEN_CAP, label: str = "", stop_file: Path | None = STOP_FILE,
                 max_consecutive_failures: int = 3, progress: Callable[[str], None] = log) -> dict[str, Any]:
    """Embed the (key, text, est_tokens) items missing from the store with at most ``workers`` requests in flight.

    Completed batches are appended to the store immediately, so a restart re-sends only what is missing. A batch
    that fails with HTTP 400 is bisected to isolate the offending input. Raises ``EndpointError`` after
    ``max_consecutive_failures`` failed batches (nothing already stored is lost); the run is stopped cleanly when
    ``stop_file`` exists.
    """
    workers = min(int(workers), MAX_CONCURRENCY)
    todo, seen = [], set()
    for key, text, est in items:
        if key not in store and key not in seen:
            seen.add(key)
            todo.append((key, text, est))
    summary = {"label": label, "requested": len(items), "unique_missing": len(todo), "inputs_sent": 0, "est_tokens": 0,
               "batches": 0, "failed_inputs": [], "stopped": False}
    if not todo:
        return summary
    batches = make_batches(todo, max_inputs, token_cap)
    total_tokens = sum(e for _, _, e in todo)
    started = time.monotonic()

    def work(batch: list[tuple[str, str, int]]) -> tuple[list[tuple[list[str], np.ndarray]], list[tuple[list[str], EndpointError]]]:
        """Embed a batch; a 400 bisects it, so only the rejected input fails and its siblings are kept."""
        done: list[tuple[list[str], np.ndarray]] = []
        failed: list[tuple[list[str], EndpointError]] = []

        def send(part: list[tuple[str, str, int]]) -> None:
            try:
                vectors, _ = client.embed([t for _, t, _ in part], tag=label, est_tokens=sum(e for _, _, e in part))
                done.append(([k for k, _, _ in part], vectors))
            except EndpointError as exc:
                if exc.status == 400 and len(part) > 1:
                    mid = len(part) // 2
                    send(part[:mid])
                    send(part[mid:])
                else:
                    failed.append(([k for k, _, _ in part], exc))

        send(batch)
        return done, failed

    done_inputs = done_tokens = 0
    consecutive = 0
    pending: dict[cf.Future, list[tuple[str, str, int]]] = {}
    queue = list(reversed(batches))
    est_of = {k: e for k, _, e in todo}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        while queue or pending:
            while queue and len(pending) < workers and not summary["stopped"] and consecutive < max_consecutive_failures:
                if stop_file is not None and Path(stop_file).exists():
                    summary["stopped"] = True
                    progress(f"{label}: stop file found, finishing in-flight requests")
                    break
                batch = queue.pop()
                pending[pool.submit(work, batch)] = batch
            if not pending:
                break
            finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for future in finished:
                batch = pending.pop(future)
                done, failed = future.result()
                for keys, vectors in done:
                    store.add(keys, vectors)
                    done_inputs += len(keys)
                    done_tokens += sum(est_of[k] for k in keys)
                summary["batches"] += 1
                if failed:
                    consecutive += 1
                    for keys, exc in failed:
                        summary["failed_inputs"].extend(keys)
                        progress(f"{label}: {len(keys)} input(s) failed ({exc})")
                else:
                    consecutive = 0
                elapsed = time.monotonic() - started
                if summary["batches"] % 10 == 0 or not queue and not pending:
                    rate = done_tokens / max(elapsed, 1e-9)
                    eta = (total_tokens - done_tokens) / max(rate, 1e-9)
                    progress(f"{label}: {done_inputs}/{len(todo)} inputs, {done_tokens / 1e6:.2f}M/{total_tokens / 1e6:.2f}M est tokens, "
                             f"{rate:.0f} tok/s, elapsed {elapsed:.0f}s, eta {eta:.0f}s, requests {client.stats['requests']}, "
                             f"retries {client.stats['retries']}")
            if consecutive >= max_consecutive_failures:
                queue.clear()
    summary.update(inputs_sent=done_inputs, est_tokens=done_tokens, seconds=time.monotonic() - started)
    if summary["failed_inputs"]:
        raise EndpointError(None, f"{len(summary['failed_inputs'])} inputs were not embedded in {label!r}; rerun to resume")
    return summary


# --------------------------------------------------------------------------- 1. probes, receipt, assembly


def vector_sha256(vector: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(vector, dtype=np.float32).tobytes()).hexdigest()


def probe(client: Client, tag: str, path: Path = PROBES_LOG) -> dict[str, Any]:
    """Embed the fixed document and query probes (one request each, never cached) and log their vector hashes.

    The served revision is not pinned (D3), so the start and end probes of a run (and the first probe ever logged)
    are compared; ``min_cosine_vs_first`` below 0.9999 means the served weights or numerics changed.
    """
    previous = [r for r in read_jsonl(path) if r.get("kind") in ("doc", "query")]
    first = {r["kind"]: r for r in reversed(previous)}        # earliest of each kind
    out = {"tag": tag, "utc": utc(), "probes": {}}
    for kind, text in (("doc", PROBE_DOC), ("query", f"Instruct: {INSTRUCTION_B}\nQuery:{PROBE_QUERY}")):
        vectors, usage = client.embed([text], tag=f"probe:{tag}")
        v = vectors[0]
        row = {"tag": tag, "kind": kind, "utc": utc(), "input_sha256": content_key(text), "vector_sha256": vector_sha256(v),
               "norm": float(np.linalg.norm(v)), "head": [round(float(x), 6) for x in v[:6]],
               "prompt_tokens": usage.get("prompt_tokens")}
        ref = first.get(kind)
        if ref is not None and "vec" in ref:
            ref_vec = np.asarray(ref["vec"], dtype=np.float32)
            row["cosine_vs_first"] = float(v @ ref_vec / (np.linalg.norm(v) * np.linalg.norm(ref_vec)))
        if ref is None:
            row["vec"] = [float(x) for x in v]          # the very first probe keeps its full vector as the drift reference
        append_jsonl(path, row)
        out["probes"][kind] = {k: row[k] for k in row if k != "vec"}
    return out


def models_check(client: Client) -> dict[str, Any]:
    payload = client.get("/models")
    rows = [r for r in payload.get("data", []) if r.get("id") == EMB_MODEL]
    if len(rows) != 1:
        raise EndpointError(None, f"{EMB_MODEL} is not served uniquely: {[r.get('id') for r in payload.get('data', [])]}")
    if rows[0].get("max_model_len") != EMB_MAX_MODEL_LEN:
        raise EndpointError(None, f"unexpected max_model_len {rows[0].get('max_model_len')}")
    return {"id": rows[0]["id"], "max_model_len": rows[0]["max_model_len"], "owned_by": rows[0].get("owned_by"),
            "models_sha256": sha256_text(json.dumps(rows, sort_keys=True))}


def load_inputs(tokenizer: Any | None = None) -> dict[str, Any]:
    """All units to embed: article windows and the three query variants (release row order)."""
    data = common.release()
    tokenizer = tokenizer or load_tokenizer()
    articles = article_units(data.candidate_texts, tokenizer)
    qtexts = [q.text for q in data.queries]
    queries = {v: query_units(qtexts, tokenizer, v) for v in VARIANTS}
    return {"articles": articles, "queries": queries}


def items_of(units: Sequence[Unit]) -> list[tuple[str, str, int]]:
    return [(content_key(u.text), u.text, u.est_tokens) for u in units]


def run_encode(what: str = "all", limit: int | None = None) -> dict[str, Any]:
    """Embed queries and/or articles (resumable); log start/end probes, request counts and wall time.

    ``limit`` (smoke tests only) embeds the first ``limit`` units of each group. The receipt is written only when every
    unit of the full release is in the store.
    """
    t0 = time.monotonic()
    started = utc()
    client, store = Client(EMB_BASE), VectorStore()
    served = models_check(client)
    start_probe = probe(client, "start")
    log(f"encode {what}: served {served['id']} max_model_len {served['max_model_len']}; store holds {len(store)} vectors")
    inputs = load_inputs()
    groups: dict[str, list[Unit]] = {}
    if what in ("queries", "all"):
        for v in VARIANTS:
            groups[f"queries_{v}"] = inputs["queries"][v]
    if what in ("articles", "all"):
        groups["articles"] = inputs["articles"]
    summaries = []
    try:
        for name, units in groups.items():
            units = list(units if limit is None else units[:limit])
            summaries.append(encode_items(client, store, items_of(units), label=name))
            log(f"{name}: {summaries[-1]['unique_missing']} new inputs; store now {len(store)} vectors")
    finally:
        end_probe = probe(client, "end")
        record = {"started_utc": started, "ended_utc": utc(), "what": what, "limit": limit, "wall_seconds": time.monotonic() - t0,
                  "client_stats": dict(client.stats), "served": served, "start_probe": start_probe, "end_probe": end_probe,
                  "groups": summaries, "store_vectors": len(store)}
        append_jsonl(RUNS_LOG, record)
    log(f"encode {what}: {client.stats['requests']} requests ({client.stats['retries']} retries, {client.stats['failures']} failures), "
        f"{client.stats['prompt_tokens']} prompt tokens, {time.monotonic() - t0:.0f}s")
    every_key = [content_key(u.text) for u in inputs["articles"]] + [content_key(u.text) for v in VARIANTS for u in inputs["queries"][v]]
    if limit is None and not store.missing(every_key):
        assemble(inputs, store)
    return record


def assemble(inputs: dict[str, Any] | None = None, store: VectorStore | None = None) -> dict[str, Any]:
    """Write the float32 ``.npy`` vector files and ``receipt.json`` from the store (every key must be present)."""
    store = store or VectorStore()
    inputs = inputs or load_inputs()
    data = common.release()
    DERIVED.mkdir(parents=True, exist_ok=True)
    art = inputs["articles"]
    art_keys = [content_key(u.text) for u in art]
    np.save(DERIVED / "article_vectors.npy", unit_rows(store.get(art_keys)))
    np.save(DERIVED / "article_index.npy", np.asarray([u.owner for u in art], dtype=np.int64))
    (DERIVED / "article_units.jsonl").write_text("".join(json.dumps(
        {"unit": i, "candidate_id": data.candidate_ids[u.owner], "candidate_index": u.owner, "window": u.window,
         "n_windows": u.n_windows, "est_tokens": u.est_tokens, "input_sha256": art_keys[i]}) + "\n" for i, u in enumerate(art)),
        encoding="utf-8")
    files = {"article_vectors": "article_vectors.npy", "article_index": "article_index.npy", "article_units": "article_units.jsonl"}
    input_hashes = {"articles_units": keys_digest(art_keys)}
    for v in VARIANTS:
        keys = [content_key(u.text) for u in inputs["queries"][v]]
        np.save(DERIVED / f"query_vectors_{v}.npy", unit_rows(store.get(keys)))
        files[f"query_vectors_{v}"] = f"query_vectors_{v}.npy"
        input_hashes[f"queries_{v}"] = keys_digest(keys)
    (DERIVED / "query_ids.json").write_text(json.dumps([q.query_id for q in data.queries]), encoding="utf-8")
    files["query_ids"] = "query_ids.json"
    runs = read_jsonl(RUNS_LOG)
    probes = [r for r in read_jsonl(PROBES_LOG)]
    stats = {k: sum(r["client_stats"].get(k, 0) for r in runs) for k in ("requests", "attempts", "retries", "failures", "inputs",
                                                                          "prompt_tokens")}
    stats["wall_seconds_all_runs"] = sum(r["wall_seconds"] for r in runs)
    receipt = {
        "endpoint": EMB_BASE, "model": EMB_MODEL, "served": runs[-1]["served"] if runs else None, "api_key": "none used or stored",
        "data_sent": "masked Hechos (q.text) and public article texts only; no labels",
        "encoding": {"normalisation": "vectors stored as served; the .npy files are L2-normalised client-side in float32",
                     "documents": "plain text (no instruction); long articles in windows with a title prefix, scored by MaxP",
                     "queries": {v: {"label": VARIANT_LABEL[v], "instruction": VARIANTS[v],
                                     "instruction_sha256": sha256_text(VARIANTS[v]) if VARIANTS[v] else None} for v in VARIANTS},
                     "windows": {"max_tokens_unsplit": ART_MAX_TOKENS, "content_tokens": WINDOW_CONTENT_TOKENS,
                                 "overlap": WINDOW_OVERLAP, "title_max_tokens": TITLE_MAX_TOKENS,
                                 "tokenizer_dir": str(TOKENIZER_DIR),
                                 "tokenizer_json_sha256": common.sha256_file(tokenizer_path())}},
        "counts": {"articles": len(data.candidate_ids), "article_units": len(art),
                   "articles_windowed": len({u.owner for u in art if u.n_windows > 1}), "queries": len(data.queries),
                   "unique_inputs_embedded": len(store), "est_tokens_articles": int(sum(u.est_tokens for u in art)),
                   "est_tokens_queries": {v: int(sum(u.est_tokens for u in inputs["queries"][v])) for v in VARIANTS}},
        "input_sha256": {**input_hashes, "articles_jsonl": data.candidate_binding.get("sha256"),
                         "release_manifest": common.sha256_file(common.RELEASE / "manifest.json")},
        "files": {k: {"path": v, "sha256": common.sha256_file(DERIVED / v)} for k, v in files.items()},
        "probes": [{k: v for k, v in p.items() if k != "vec"} for p in probes],
        "runs": len(runs), "totals": stats, "written_utc": utc()}
    (DERIVED / "receipt.json").write_text(json.dumps(receipt, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"assembled vectors and receipt ({len(art)} article units, {len(data.queries)} queries x {len(VARIANTS)} variants)")
    return receipt


# --------------------------------------------------------------------------- 2. dense scores, variant selection, H_qwen8b

SEARCH_LOG = RUN / "search_log.jsonl"
SELECTION_DIR = RUN / "selection"
CACHE_DIR = RUN / "cache"
TOL = 1e-12
BLOCK = 256
W_DENSE = tuple(round(0.1 * i, 1) for i in range(11))     # H_qwen8b grid: weight of the dense channel (0 = L_star only)
W_VOTES = (0.05, 0.1, 0.15, 0.25, 0.5, 0.75)              # fusion weights of a vote / prototype channel with L_star
KS = (5, 10, 20, 50, 100)
ALPHAS = (0.25, 0.5, 0.75)
POPS = ("P_spec", "P", "S2_spec")        # the all-queries population is left to the lead (it needs ~3 GB of RAM)
CODE_FILES = [Path(__file__), Path(common.__file__)] + [Path(m.__file__) for m in (lexical, knn, fusion, metrics)]


def search_log(entry: dict[str, Any]) -> None:
    append_jsonl(SEARCH_LOG, {"utc": utc(), **entry})


def reset_search_log(*stages: str) -> None:
    """Drop earlier entries of the given stages (a re-run rewrites them); other stages keep their entries."""
    if SEARCH_LOG.exists():
        keep = [row for row in read_jsonl(SEARCH_LOG) if row.get("stage") not in stages]
        text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in keep)
        SEARCH_LOG.write_text(text, encoding="utf-8")


def pair(w: float) -> list[float]:
    """Two-channel weights (1 - w, w), as ``uol.facts_to_law.run._pairs``."""
    return [round(1.0 - float(w), 10), float(w)]


def argbest(values: Sequence[float]) -> int:
    """Lowest index among values within TOL of the maximum (the crossfit tie rule)."""
    best = max(values)
    return min(i for i, v in enumerate(values) if v >= best - TOL)


_VEC: dict[str, Any] = {}


def article_vectors() -> tuple[np.ndarray, np.ndarray]:
    """Unit-norm article window vectors [units, DIM] and the candidate index of each unit."""
    if "A" not in _VEC:
        _VEC["A"] = np.load(DERIVED / "article_vectors.npy")
        _VEC["owner"] = np.load(DERIVED / "article_index.npy")
    return _VEC["A"], _VEC["owner"]


def query_vectors(variant: str) -> np.ndarray:
    """Unit-norm query vectors [5980, DIM] in release row order for a variant (a, b or c)."""
    key = f"Q{variant}"
    if key not in _VEC:
        ids = json.loads((DERIVED / "query_ids.json").read_text(encoding="utf-8"))
        if ids != [q.query_id for q in common.release().queries]:
            raise ValueError("stored query vectors do not follow the release row order")
        _VEC[key] = np.load(DERIVED / f"query_vectors_{variant}.npy")
    return _VEC[key]


def dense_block(q: np.ndarray, out: np.ndarray | None = None) -> np.ndarray:
    """MaxP cosine scores [len(q), n_articles] of unit query vectors against the article window vectors."""
    vectors, owner = article_vectors()
    n = len(common.release().candidate_ids)
    res = np.empty((len(q), n), dtype=np.float32) if out is None else out
    for start, block in common.maxp_scores(q, vectors, owner, n):
        res[start:start + len(block)] = block
    return res


def dev_metrics(scores_dev: np.ndarray, populations: Sequence[str] = POPS) -> dict[str, Any]:
    """Dev out-of-fold component-macro metrics (rows = ``common.dev_rows()``); the lockbox guard stays on."""
    dev = common.dev_rows()
    out = {}
    for pop in populations:
        e = common.evaluate(scores_dev, dev, pop)
        out[pop] = {"n_queries": e["n_queries"], "n_components": e["n_components"], **{k: float(v) for k, v in e["macro"].items()}}
    return out


def scratch_matrices(names: Sequence[str], n_rows: int, n_cols: int) -> dict[str, np.ndarray]:
    """Disk-backed float32 matrices (``CACHE_DIR/scratch_<name>.npy``) so dev evaluation never needs gigabytes of RAM."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return {n: np.lib.format.open_memmap(CACHE_DIR / f"scratch_{n}.npy", mode="w+", dtype=np.float32, shape=(n_rows, n_cols)) for n in names}


def drop_scratch() -> None:
    for path in CACHE_DIR.glob("scratch_*.npy"):
        path.unlink()


def method_dev_metrics(method_id: str, populations: Sequence[str] = POPS) -> dict[str, Any]:
    return dev_metrics(np.asarray(common.open_scores(method_id)[common.dev_rows()]), populations)


def input_hashes() -> dict[str, str]:
    out = {"split_sha256": common.sha256_file(common.SPLIT_PATH),
           "release_manifest_sha256": common.sha256_file(common.RELEASE / "manifest.json")}
    receipt = DERIVED / "receipt.json"
    if receipt.exists():
        out["qwen_receipt_sha256"] = common.sha256_file(receipt)
        files = json.loads(receipt.read_text(encoding="utf-8"))["files"]
        out.update({f"qwen_{k}_sha256": v["sha256"] for k, v in files.items() if k.startswith(("article_vectors", "query_vectors"))})
    return out


def endpoint_identity() -> dict[str, Any]:
    receipt = DERIVED / "receipt.json"
    if not receipt.exists():
        return {"endpoint": EMB_BASE, "model": EMB_MODEL}
    r = json.loads(receipt.read_text(encoding="utf-8"))
    return {"endpoint": r["endpoint"], "model": r["model"], "served": r["served"], "revision": "not pinned by the service (D3)",
            "probes": "start and end probe vector hashes per run in receipt.json"}


def _write_scores(method_id: str, producer: Callable[[np.ndarray], None], manifest: dict[str, Any], force: bool = False,
                  embeddings: bool = True) -> None:
    """Create a score file, fill it with ``producer`` and write its manifest (``common.finish_scores``).

    ``embeddings=False`` (a no-embedding diagnostic) keeps the Qwen vector hashes and endpoint identity out of the manifest.
    """
    import shutil

    folder = common.score_path(method_id).parent
    if folder.exists():
        if not force and (folder / "manifest.json").exists():
            log(f"{method_id} already complete")
            return
        shutil.rmtree(folder)
    t0 = time.perf_counter()
    matrix = common.create_scores(method_id)
    producer(matrix)
    matrix.flush()
    del matrix
    hashes = input_hashes() if embeddings else {k: v for k, v in input_hashes().items() if not k.startswith("qwen_")}
    extra = {"endpoint": endpoint_identity()} if embeddings else {}
    manifest = {"method": method_id, "seeds": None, "input_sha256": hashes, **extra,
                "wall_seconds": time.perf_counter() - t0, **manifest}
    common.finish_scores(method_id, manifest, CODE_FILES)
    log(f"{method_id} complete in {time.perf_counter() - t0:.0f}s")


def stage_dense(force: bool = False) -> dict[str, Any]:
    """Score all queries under each variant, log dev metrics for all three, deliver the best as ``dense_qwen8b``."""
    t0 = time.perf_counter()
    dev = common.dev_rows()
    results: dict[str, Any] = {}
    for v in VARIANTS:
        t = time.perf_counter()
        scores = dense_block(query_vectors(v)[dev])
        results[v] = dev_metrics(scores)
        search_log({"stage": "dense", "method": "dense_qwen8b", "config": {"variant": v, "instruction": VARIANTS[v]},
                    "dev": results[v]})
        log(f"dense variant {v} ({VARIANT_LABEL[v]}): dev P_spec nDCG@10 {results[v]['P_spec']['ndcg_at_10']:.4f}, "
            f"P {results[v]['P']['ndcg_at_10']:.4f} ({time.perf_counter() - t:.0f}s)")
        del scores
    best = list(VARIANTS)[argbest([results[v]["P_spec"]["ndcg_at_10"] for v in VARIANTS])]
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    (SELECTION_DIR / "dense.json").write_text(json.dumps({"selected": best, "dev": results}, indent=1), encoding="utf-8")
    q = query_vectors(best)

    def producer(matrix: np.ndarray) -> None:
        for s in range(0, len(q), 1024):
            dense_block(q[s:s + 1024], out=matrix[s:s + 1024])

    _write_scores("dense_qwen8b", producer, {
        "family": "EMB", "label_free": True,
        "description": "Qwen3-Embedding-8B (MIT endpoint) cosine similarity, MaxP over title-prefixed windows for articles above 8,000 tokens",
        "training_data": "none (frozen encoder served by the endpoint; label-free)",
        "configuration": {"query_variant": best, "query_instruction": VARIANTS[best], "variant_label": VARIANT_LABEL[best],
                          "documents": "plain article text", "aggregation": "maxp over windows (title-prefixed windows only above 8,000 tokens)"},
        "selection_evidence": {"protocol": "variant chosen on dev P_spec nDCG@10 (label-free scores: every dev row scored by the same frozen "
                                           "model, so dev rows are out-of-fold by construction); all three variants logged",
                               "dev_P_spec_ndcg10": {v: results[v]["P_spec"]["ndcg_at_10"] for v in VARIANTS},
                               "dev_P_ndcg10": {v: results[v]["P"]["ndcg_at_10"] for v in VARIANTS},
                               "search_log": str(SEARCH_LOG), "selection_file": str(SELECTION_DIR / "dense.json")},
        "device": "cpu (BLAS); vectors from the MIT endpoint"}, force)
    log(f"dense stage done in {time.perf_counter() - t0:.0f}s; selected variant {best}")
    return {"selected": best, "dev": results}


# ---- per-query evaluation and L* reconstruction


class Ev:
    """Per-query nDCG@10 on the specific (condensed) view for dev P_spec rows, pooled like ``common.evaluate``."""

    def __init__(self) -> None:
        d = self.data = common.release()
        self.N = len(d.candidate_ids)
        self.roles = common.roles()
        self.fold = np.asarray([-1 if r == "lockbox" else int(r[len("dev_fold"):]) for r in self.roles])
        ids = sorted({q.component_id for q in d.queries})
        position = {c: i for i, c in enumerate(ids)}
        self.comp = np.asarray([position[q.component_id] for q in d.queries])
        self.ncomp = len(ids)
        self.generic = d.generic_idx if len(d.generic_idx) else None
        self.tie = common.tie_order()
        self.pspec = common.population_rows("P_spec", common.dev_rows())     # sorted release rows
        self.texts = [q.text for q in d.queries]

    def ndcg_spec(self, block: np.ndarray, rows: Sequence[int]) -> np.ndarray:
        """nDCG@10 of each P_spec row of ``rows`` (``block`` aligned to ``rows``), generic targets removed from the list."""
        res = metrics.evaluate_matrix(np.asarray(block), [self.data.specific_idx[int(r)] for r in rows], exclude_idx=self.generic,
                                      top_k=0, tie_order=self.tie)
        return np.asarray([x["ndcg_at_10"] for x in res], dtype=np.float64)

    def pooled(self, values: np.ndarray, rows: Sequence[int]) -> float:
        """Component-macro mean of per-query values (queries within a component first, then components)."""
        codes = self.comp[np.asarray(rows, dtype=np.int64)]
        sums = np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=self.ncomp)
        counts = np.bincount(codes, minlength=self.ncomp)
        keep = counts > 0
        return float((sums[keep] / counts[keep]).mean())


def lstar_selections() -> dict[str, dict[str, Any]]:
    """Per scoring group L* choice, read from the L_star manifest (``configuration`` by role)."""
    manifest = json.loads((common.score_path("L_star").parent / "manifest.json").read_text(encoding="utf-8"))
    return manifest["configuration"]


class LStarRows:
    """Label-free L* of the dev P_spec rows under any scoring group's L* selection (z-scored channels cached on disk).

    The cross-fitted ``L_star`` file scores each row with its own group's selection, chosen with the other folds'
    labels. Selecting a fusion weight for group g on those rows would let g's own labels leak (second order) through
    the L* choices of the other folds, so the training rows are recomputed under g's selection from the channels.
    """

    def __init__(self, rows: np.ndarray | None = None) -> None:
        self.rows = common.population_rows("P_spec", common.dev_rows()) if rows is None else np.asarray(rows)
        self.position = {int(r): i for i, r in enumerate(self.rows)}
        self._channels: dict[str, np.ndarray] = {}

    def channel(self, name: str) -> np.ndarray:
        """Robust-z BM25 channel ``'<kind>|<k1>|<b>'`` or ``'char'`` for ``self.rows`` (float32 [rows, N])."""
        if name not in self._channels:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path = CACHE_DIR / f"lstar_z_{name.replace('|', '_')}.npy"
            if not path.exists():
                t = time.perf_counter()
                kind = name.split("|")[0]
                params = None if kind == "char" else {"k1": float(name.split("|")[1]), "b": float(name.split("|")[2])}
                index = lexical.LexicalIndex(common.release().candidate_texts, kind=kind)
                texts = [q.text for q in common.release().queries]
                out = np.lib.format.open_memmap(path.with_suffix(".tmp.npy"), mode="w+", dtype=np.float32,
                                                shape=(len(self.rows), len(common.release().candidate_ids)))
                for s in range(0, len(self.rows), 128):
                    out[s:s + 128] = fusion.robust_z(index.score([texts[int(r)] for r in self.rows[s:s + 128]], params))
                out.flush()
                del out
                os.replace(path.with_suffix(".tmp.npy"), path)
                log(f"L* channel {name} for {len(self.rows)} rows in {time.perf_counter() - t:.0f}s")
            self._channels[name] = np.load(path, mmap_mode="r")
        return self._channels[name]

    def lstar(self, rows: Sequence[int], sel: dict[str, Any]) -> np.ndarray:
        """Raw L* scores (before the later robust-z) of ``rows`` under selection ``sel`` (a manifest configuration entry)."""
        idx = np.asarray([self.position[int(r)] for r in rows], dtype=np.int64)
        kind = sel["kind"]
        if kind == "fuse":
            bm = self.channel(f"{sel['base']}|{sel['k1']}|{sel['b']}")[idx]
            ch = self.channel("char")[idx]
            return fusion.linear_fusion([np.asarray(bm), np.asarray(ch)], pair(sel["w_char"]), normalize="none")
        if kind in ("bm25", "bm25_stem"):
            return np.asarray(self.channel(f"{kind}|{sel['k1']}|{sel['b']}")[idx])
        if kind == "char":
            return np.asarray(self.channel("char")[idx])
        raise ValueError(f"unknown L* kind {kind!r}")


def stage_h(force: bool = False) -> dict[str, Any]:
    """``H_qwen8b``: robust-z linear fusion of L_star and ``dense_qwen8b``; the dense weight is chosen per scoring group.

    Selection (inner CV of a label-free pair): the weight maximising the pooled component-macro P_spec nDCG@10 over the
    group's training rows, with L* recomputed under the group's own selection. Ties go to the lowest weight.
    """
    t0 = time.perf_counter()
    ev, ls = Ev(), LStarRows()
    sels = lstar_selections()
    dense = common.open_scores("dense_qwen8b")
    rows = ev.pspec
    zd = fusion.robust_z(np.asarray(dense[rows]))
    tables: dict[str, np.ndarray] = {}                     # L* selection key -> per-query nDCG [len(W_DENSE), rows]
    for sel in sels.values():
        key = json.dumps(sel, sort_keys=True)
        if key in tables:
            continue
        zl = fusion.robust_z(ls.lstar(rows, sel))
        tables[key] = np.stack([ev.ndcg_spec(fusion.linear_fusion([zl, zd], pair(w), normalize="none"), rows) for w in W_DENSE])
        log(f"H table for L* selection w_char={sel.get('w_char')} (pooled over all dev P_spec rows): "
            f"{[round(ev.pooled(t, rows), 4) for t in tables[key]]}")
    chosen: dict[str, Any] = {}
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    for role, _, train in common.scoring_groups():
        mask = np.isin(rows, train)
        table = tables[json.dumps(sels[role], sort_keys=True)]
        objective = [ev.pooled(table[j][mask], rows[mask]) for j in range(len(W_DENSE))]
        j = argbest(objective)
        chosen[role] = {"w_dense": W_DENSE[j], "objective": objective[j],
                        "grid": [{"w_dense": w, "objective": o} for w, o in zip(W_DENSE, objective)], "lstar": sels[role]}
        search_log({"stage": "h", "method": "H_qwen8b", "group": role, "evidence": chosen[role]["grid"], "chosen": W_DENSE[j]})
        log(f"H_qwen8b {role}: w_dense {W_DENSE[j]} (inner objective {objective[j]:.4f}; L_star only {objective[0]:.4f})")
    (SELECTION_DIR / "H_qwen8b.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")
    lfile = common.open_scores("L_star")

    def producer(matrix: np.ndarray) -> None:
        for role, rows_g, _ in common.scoring_groups():
            w = chosen[role]["w_dense"]
            for s in range(0, len(rows_g), BLOCK):
                r = rows_g[s:s + BLOCK]
                matrix[r] = fusion.linear_fusion([fusion.robust_z(np.asarray(lfile[r])), fusion.robust_z(np.asarray(dense[r]))],
                                                 pair(w), normalize="none")

    _write_scores("H_qwen8b", producer, {
        "family": "EMB", "label_free": True,
        "description": "robust-z linear fusion of L_star and dense_qwen8b; dense weight chosen per scoring group on its training rows",
        "training_data": "none (L_star and the frozen dense scores are label-free); the fusion weight is selected on the group's training rows only",
        "configuration": {role: {"type": "linear", "w_dense": chosen[role]["w_dense"]} for role in chosen},
        "depends_on": ["L_star", "dense_qwen8b"],
        "selection_evidence": {"protocol": "per group: pooled component-macro P_spec nDCG@10 over the group's training dev rows, L* recomputed "
                                           "under the group's own selection (avoids second-order leakage through other folds' L* choices); "
                                           f"grid w_dense {list(W_DENSE)}; ties to the lowest weight",
                               "selection_file": str(SELECTION_DIR / "H_qwen8b.json"), "search_log": str(SEARCH_LOG)},
        "device": "cpu"}, force)
    log(f"H stage done in {time.perf_counter() - t0:.0f}s")
    return chosen


# --------------------------------------------------------------------------- 3. supervised methods (cross-fitted)

INNER_DIR = CACHE_DIR / "inner"
SUPERVISED = ("knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b")
FAMILY_OF = {"knn_qwen8b": "knn", "knn_fused_qwen8b": "kf", "proto_qwen8b": "proto"}
PROTO_MODES = (("all", 0), ("specific", 0), ("all", 1), ("specific", 1))     # (positive mode, component normalisation)


def zrow(x: np.ndarray) -> np.ndarray:
    """Row-wise z-score over memory entries (mean 0, std 1; constant rows map to 0)."""
    x = np.asarray(x, dtype=np.float32)
    return ((x - x.mean(axis=1, keepdims=True)) / np.maximum(x.std(axis=1, keepdims=True), 1e-6)).astype(np.float32, copy=False)


def config_grid() -> dict[str, list[str]]:
    """Configuration ids per source family: ``knn|<variant>|k<k>``, ``kf|<variant>|a<alpha>|k<k>``, ``proto|<variant>|<mode>|<cn>``,
    plus the lexical-key control ``lex|k<k>`` (no embedding; the K_lex machinery under the same inner CV)."""
    return {
        "knn": [f"knn|{v}|k{k}" for v in VARIANTS for k in KS],
        "kf": [f"kf|{v}|a{a}|k{k}" for v in VARIANTS for a in ALPHAS for k in KS],
        "proto": [f"proto|{v}|{m}|{c}" for v in VARIANTS for m, c in PROTO_MODES],
        "lex": [f"lex|k{k}" for k in KS],
    }


@dataclass(frozen=True)
class GroupInfo:
    idx: int
    role: str
    rows: np.ndarray
    train: np.ndarray
    train_folds: tuple[int, ...]


class Ctx(Ev):
    """Evaluation context plus the scoring groups and (cached) fit sets."""

    def __init__(self) -> None:
        super().__init__()
        self.groups: list[GroupInfo] = []
        for i, (role, rows, train) in enumerate(common.scoring_groups()):
            folds = tuple(sorted({int(f) for f in self.fold[train]}))
            if not np.array_equal(np.flatnonzero(np.isin(self.fold, folds)), np.sort(train)):
                raise AssertionError("training rows must be exactly the rows of the training folds")
            self.groups.append(GroupInfo(i, role, np.asarray(rows), np.asarray(train), folds))
        self._fits: dict[tuple[int, ...], "FitSet"] = {}

    def fit(self, folds: Sequence[int]) -> "FitSet":
        key = tuple(sorted(int(f) for f in folds))
        if key not in self._fits:
            self._fits = {key: FitSet(self, key)}          # one live fit set keeps memory flat
        return self._fits[key]


class FitSet:
    """Citation memory and article prototypes fitted on the rows of ``folds`` (labels of those rows only)."""

    def __init__(self, ctx: Ctx, folds: Sequence[int]) -> None:
        d = ctx.data
        self.ctx, self.folds = ctx, tuple(folds)
        self.rows = np.flatnonzero(np.isin(ctx.fold, self.folds))
        self.mem = knn.build_memory([d.queries[int(r)].query_id for r in self.rows], d.component_by_query, d.references,
                                    d.candidate_index)
        self.mem_rows = np.asarray([d.query_index[q] for q in self.mem.query_ids], dtype=np.int64)
        self.mem_components = list(self.mem.components)
        self._lex: lexical.LexicalIndex | None = None
        self._proto: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}

    def lex_sim(self, rows: Sequence[int]) -> np.ndarray:
        if self._lex is None:
            self._lex = lexical.LexicalIndex([self.ctx.texts[int(r)] for r in self.mem_rows], kind="bm25")
        return self._lex.score([self.ctx.texts[int(r)] for r in rows])

    def dense_sim(self, rows: Sequence[int], variant: str) -> np.ndarray:
        q = query_vectors(variant)
        return (q[np.asarray(rows, dtype=np.int64)] @ q[self.mem_rows].T).astype(np.float32, copy=False)

    def votes(self, key: np.ndarray, k: int, rows: Sequence[int]) -> np.ndarray:
        d = self.ctx.data
        comps = [d.queries[int(r)].component_id for r in rows]
        return knn.knn_votes(key, self.mem.positives, int(k), self.ctx.N, "sim", eval_components=comps,
                             fit_components=self.mem_components)

    def prototypes(self, variant: str, mode: str, cn: int) -> tuple[np.ndarray, np.ndarray]:
        """Unit prototypes ``[n_seen, DIM]`` and the candidate index of each (articles cited by at least one fit row).

        A fit row adds its query vector to each cited article with weight 1/|positives|. ``specific`` mode gives a row's
        specific positives 1/|specific positives| (generic ones keep 1/|positives|); ``cn`` divides by the number of fit
        rows of the row's component.
        """
        from scipy import sparse

        key = (variant, mode, int(cn))
        if key not in self._proto:
            ctx, d = self.ctx, self.ctx.data
            size = np.bincount(ctx.comp[self.rows], minlength=ctx.ncomp)
            ri, ai, wt = [], [], []
            for j, r in enumerate(self.rows):
                pos = d.positive_idx[int(r)]
                if not pos:
                    continue
                spec = d.specific_idx[int(r)]
                specific_set = set(spec)
                for a in pos:
                    w = 1.0 / len(spec) if mode == "specific" and spec and a in specific_set else 1.0 / len(pos)
                    if cn:
                        w /= size[ctx.comp[r]]
                    ri.append(j)
                    ai.append(a)
                    wt.append(w)
            seen = np.unique(np.asarray(ai, dtype=np.int64))
            column = {int(a): i for i, a in enumerate(seen)}
            W = sparse.csr_matrix((np.asarray(wt, dtype=np.float32), (np.asarray(ri), np.asarray([column[a] for a in ai]))),
                                  shape=(len(self.rows), len(seen)))
            S = np.asarray(W.T @ query_vectors(variant)[self.rows], dtype=np.float32)
            self._proto[key] = (unit_rows(S), seen)
            if len(self._proto) > 2:                       # keep memory flat
                self._proto.pop(next(iter(self._proto)))
        return self._proto[key]

    def proto_scores(self, rows: Sequence[int], variant: str, mode: str, cn: int) -> np.ndarray:
        """Cosine to each article's prototype; articles no fit row cites get the row minimum."""
        protos, seen = self.prototypes(variant, mode, cn)
        s = query_vectors(variant)[np.asarray(rows, dtype=np.int64)] @ protos.T
        out = np.empty((len(s), self.ctx.N), dtype=np.float32)
        out[:] = s.min(axis=1, keepdims=True)
        out[:, seen] = s
        return out


class Sims:
    """Memoised similarities of ``rows`` to one fit set's memory."""

    def __init__(self, fit: FitSet, rows: Sequence[int]) -> None:
        self.fit, self.rows = fit, np.asarray(rows, dtype=np.int64)
        self._memo: dict[Any, np.ndarray] = {}

    def lex(self) -> np.ndarray:
        if "lex" not in self._memo:
            self._memo["lex"] = self.fit.lex_sim(self.rows)
        return self._memo["lex"]

    def zlex(self) -> np.ndarray:
        if "zlex" not in self._memo:
            self._memo["zlex"] = zrow(self.lex())
        return self._memo["zlex"]

    def dense(self, v: str) -> np.ndarray:
        if ("d", v) not in self._memo:
            self._memo[("d", v)] = self.fit.dense_sim(self.rows, v)
        return self._memo[("d", v)]

    def zdense(self, v: str) -> np.ndarray:
        if ("zd", v) not in self._memo:
            self._memo[("zd", v)] = zrow(self.dense(v))
        return self._memo[("zd", v)]


def config_block(sims: Sims, cfg: str) -> np.ndarray:
    """Scores ``[len(rows), N]`` of one configuration for the rows of ``sims`` (votes, or prototype cosines)."""
    parts = cfg.split("|")
    fit, rows = sims.fit, sims.rows
    if parts[0] == "knn":
        return fit.votes(sims.dense(parts[1]), int(parts[2][1:]), rows)
    if parts[0] == "kf":
        alpha = float(parts[2][1:])
        key = (np.float32(alpha) * sims.zlex() + np.float32(1.0 - alpha) * sims.zdense(parts[1])).astype(np.float32, copy=False)
        return fit.votes(key, int(parts[3][1:]), rows)
    if parts[0] == "lex":
        return fit.votes(sims.lex(), int(parts[1][1:]), rows)
    if parts[0] == "proto":
        return fit.proto_scores(rows, parts[1], parts[2], int(parts[3]))
    raise ValueError(f"unknown configuration {cfg!r}")


_CTX: dict[str, Any] = {}


def get_ctx() -> Ctx:
    if "ctx" not in _CTX:
        _CTX["ctx"] = Ctx()
        _CTX["ls"] = LStarRows()
        _CTX["sel"] = lstar_selections()
    return _CTX["ctx"]


def inner_path(role: str, inner: int) -> Path:
    return INNER_DIR / f"{role}_f{inner}.npz"


def inner_task(g_idx: int, inner: int, force: bool = False) -> str:
    """Score the P_spec rows of dev fold ``inner`` with a fit on the group's other training folds, for every configuration.

    Stores per-query nDCG@10 (specific view) alone and fused with L* (robust z, ``W_VOTES``), under the group's own L*
    selection. Labels used: those of the group's training folds except ``inner`` only.
    """
    ctx = get_ctx()
    g = ctx.groups[g_idx]
    path = inner_path(g.role, inner)
    if path.exists() and not force:
        return str(path)
    t0 = time.perf_counter()
    fit_folds = tuple(f for f in g.train_folds if f != inner)
    rows = ctx.pspec[ctx.fold[ctx.pspec] == inner]
    fit = ctx.fit(fit_folds)
    zl = fusion.robust_z(_CTX["ls"].lstar(rows, _CTX["sel"][g.role]))
    sims = Sims(fit, rows)
    out: dict[str, np.ndarray] = {"rows": rows}
    for family, configs in config_grid().items():
        for cfg in configs:
            block = config_block(sims, cfg)
            out[f"{cfg}#alone"] = ctx.ndcg_spec(block, rows)
            zb = fusion.robust_z(block)
            out[f"{cfg}#fused"] = np.stack([ctx.ndcg_spec(fusion.linear_fusion([zl, zb], pair(w), normalize="none"), rows)
                                            for w in W_VOTES])
    out["seconds"] = np.asarray([time.perf_counter() - t0])
    INNER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    log(f"inner task {g.role} fold {inner}: fit folds {fit_folds}, {len(rows)} P_spec rows, "
        f"{sum(len(v) for v in config_grid().values())} configurations, {time.perf_counter() - t0:.0f}s")
    return str(path)


def _inner_star(args: tuple[int, int, bool]) -> str:
    return inner_task(*args)


def run_inner(workers: int = 4, force: bool = False) -> None:
    """Run all inner tasks (25 = 5 dev groups x 4 folds + lockbox x 5 folds) in ``workers`` processes (cached on disk)."""
    import multiprocessing as mp

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(max(1, 12 // max(workers, 1)))
    ctx = get_ctx()
    tasks = [(g.idx, f, force) for g in ctx.groups for f in g.train_folds]
    todo = [t for t in tasks if force or not inner_path(ctx.groups[t[0]].role, t[1]).exists()]
    log(f"inner tasks: {len(tasks)} total, {len(todo)} to run with {workers} workers")
    if not todo:
        return
    if workers <= 1:
        for t in todo:
            inner_task(*t)
        return
    from uol.embedding_push import qwen_endpoint as module        # pickle the worker by its module path, not as __main__

    with mp.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(module._inner_star, todo):
            pass


def load_objectives(ctx: Ctx) -> dict[str, Any]:
    """Pooled component-macro P_spec nDCG@10 per (group, configuration): alone, and fused for every ``W_VOTES``."""
    out: dict[str, Any] = {}
    for g in ctx.groups:
        parts = [np.load(inner_path(g.role, f)) for f in g.train_folds]
        rows = np.concatenate([p["rows"] for p in parts])
        entry = {"alone": {}, "fused": {}}
        for cfg in [c for cs in config_grid().values() for c in cs]:
            entry["alone"][cfg] = ctx.pooled(np.concatenate([p[f"{cfg}#alone"] for p in parts]), rows)
            fused = np.concatenate([p[f"{cfg}#fused"] for p in parts], axis=1)
            entry["fused"][cfg] = [ctx.pooled(fused[j], rows) for j in range(len(W_VOTES))]
        out[g.role] = entry
    return out


def select_configs(obj: dict[str, Any], family: str) -> dict[str, Any]:
    """Per group: the configuration maximising the alone objective, and the (configuration, weight) maximising the fused one."""
    cfgs = config_grid()[family]
    alone = [obj["alone"][c] for c in cfgs]
    ia = argbest(alone)
    flat = [(c, j) for c in cfgs for j in range(len(W_VOTES))]
    fused = [obj["fused"][c][j] for c, j in flat]
    jf = argbest(fused)
    return {"alone": {"config": cfgs[ia], "objective": alone[ia], "grid": [{"config": c, "objective": o} for c, o in zip(cfgs, alone)]},
            "fused": {"config": flat[jf][0], "w": W_VOTES[flat[jf][1]], "objective": fused[jf],
                      "grid": [{"config": c, "w": W_VOTES[j], "objective": o} for (c, j), o in zip(flat, fused)]}}


def stage_supervised(force: bool = False, workers: int = 4) -> dict[str, Any]:
    """``knn_qwen8b``, ``knn_fused_qwen8b``, ``proto_qwen8b`` (+ ``K_qwen8b`` = L* fused with the best of them).

    Per scoring group every hyper-parameter (query variant, k, alpha, prototype mode, fusion weight) is chosen by nested inner
    CV: each of the group's training folds is scored with a memory/prototypes fitted on the group's other training folds, and
    the pooled component-macro P_spec nDCG@10 decides (ties: lowest grid index). Outer scoring then fits on all of the
    group's training folds and scores every row of the group; the lockbox group trains on all dev rows and gets no metric.
    """
    import shutil

    t0 = time.perf_counter()
    ctx = get_ctx()
    run_inner(workers)
    obj = load_objectives(ctx)
    families = {"knn": "knn_qwen8b", "kf": "knn_fused_qwen8b", "proto": "proto_qwen8b", "lex": "lex_control"}
    sel = {fam: {g.role: select_configs(obj[g.role], fam) for g in ctx.groups} for fam in families}
    kq = {}
    for g in ctx.groups:
        options = [sel[fam][g.role]["fused"] for fam in ("knn", "kf", "proto")]
        i = argbest([o["objective"] for o in options])
        kq[g.role] = {"family": ("knn", "kf", "proto")[i], "config": options[i]["config"], "w": options[i]["w"],
                      "objective": options[i]["objective"]}
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    reset_search_log("inner")
    for fam, name in families.items():
        (SELECTION_DIR / f"{name}.json").write_text(json.dumps(sel[fam], indent=1), encoding="utf-8")
        for g in ctx.groups:
            for cfg in config_grid()[fam]:
                search_log({"stage": "inner", "method": name, "group": g.role, "config": cfg,
                            "inner_P_spec_ndcg10_alone": obj[g.role]["alone"][cfg],
                            "inner_P_spec_ndcg10_fused_with_Lstar": dict(zip(map(str, W_VOTES), obj[g.role]["fused"][cfg]))})
    (SELECTION_DIR / "K_qwen8b.json").write_text(json.dumps(kq, indent=1), encoding="utf-8")
    for g in ctx.groups:
        log(f"{g.role}: knn {sel['knn'][g.role]['alone']['config']} ({sel['knn'][g.role]['alone']['objective']:.4f}); "
            f"kf {sel['kf'][g.role]['alone']['config']} ({sel['kf'][g.role]['alone']['objective']:.4f}); "
            f"proto {sel['proto'][g.role]['alone']['config']} ({sel['proto'][g.role]['alone']['objective']:.4f}); "
            f"lex control {sel['lex'][g.role]['alone']['config']} ({sel['lex'][g.role]['alone']['objective']:.4f}); "
            f"K_qwen8b {kq[g.role]['family']} {kq[g.role]['config']} w={kq[g.role]['w']} ({kq[g.role]['objective']:.4f}); "
            f"lex fused {sel['lex'][g.role]['fused']['config']} w={sel['lex'][g.role]['fused']['w']} ({sel['lex'][g.role]['fused']['objective']:.4f})")

    inner_seconds = float(sum(np.load(inner_path(g.role, f))["seconds"][0] for g in ctx.groups for f in g.train_folds))
    delivered = list(SUPERVISED) + ["K_qwen8b"]
    for m in delivered:
        if common.score_path(m).parent.exists():
            if not force and (common.score_path(m).parent / "manifest.json").exists():
                log(f"{m} already complete; use --force to rebuild")
                return {"selection": sel}
            shutil.rmtree(common.score_path(m).parent)
    mats = {m: common.create_scores(m) for m in delivered}
    dev = common.dev_rows()
    dev_pos = {int(r): i for i, r in enumerate(dev)}
    fused_dev = scratch_matrices(list(families), len(dev), ctx.N)
    lfile = common.open_scores("L_star")
    for g in ctx.groups:
        fit = ctx.fit(g.train_folds)
        for s in range(0, len(g.rows), 512):
            rows = g.rows[s:s + 512]
            sims, memo = Sims(fit, rows), {}

            def blk(cfg: str) -> np.ndarray:
                if cfg not in memo:
                    memo[cfg] = config_block(sims, cfg)
                return memo[cfg]

            zl = fusion.robust_z(np.asarray(lfile[rows]))
            for fam in ("knn", "kf", "proto"):
                mats[families[fam]][rows] = blk(sel[fam][g.role]["alone"]["config"])
            for fam in families:
                f = sel[fam][g.role]["fused"]
                fb = fusion.linear_fusion([zl, fusion.robust_z(blk(f["config"]))], pair(f["w"]), normalize="none")
                if g.role != "lockbox":
                    fused_dev[fam][[dev_pos[int(r)] for r in rows]] = fb
                if fam == kq[g.role]["family"]:
                    mats["K_qwen8b"][rows] = fb
        log(f"outer scoring {g.role}: {len(g.rows)} rows ({time.perf_counter() - t0:.0f}s)")
    for m in delivered:
        mats[m].flush()
    del mats
    results: dict[str, Any] = {"alone": {m: method_dev_metrics(m) for m in SUPERVISED},
                               "fused_with_L_star": {families[fam]: dev_metrics(fused_dev[fam]) for fam in families},
                               "K_qwen8b": method_dev_metrics("K_qwen8b")}
    del fused_dev
    drop_scratch()
    (RUN / "results_supervised.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    inner = ("inner CV: for each outer group the group's training dev folds are scored in turn with a fit on the group's other training "
             "folds; objective = pooled component-macro P_spec nDCG@10 over those rows (ties: lowest grid index)")
    common_manifest = {"family": "EMB", "label_free": False,
                       "training_data": "labels of the scoring group's training dev rows only (lockbox group: all dev rows); the lockbox never trains",
                       "device": "cpu", "wall_seconds": inner_seconds + (time.perf_counter() - t0),
                       "wall_seconds_inner_tasks_summed_over_workers": inner_seconds, "wall_seconds_this_invocation": time.perf_counter() - t0,
                       "selection_evidence": {"protocol": inner, "search_log": str(SEARCH_LOG), "grids": {f: config_grid()[f] for f in families},
                                              "w_votes": list(W_VOTES)}}
    descriptions = {
        "knn_qwen8b": ("kNN citation-memory votes (one memory entry per training component: representative query vector, union of positives; "
                       "'sim' weighting) with the Qwen3-Embedding-8B query cosine as key; query variant and k by inner CV", "knn"),
        "knn_fused_qwen8b": ("kNN votes with key alpha*z(BM25 sim) + (1-alpha)*z(Qwen cosine) (row z-scores over memory entries); variant, alpha, "
                             "k by inner CV", "kf"),
        "proto_qwen8b": ("article prototype = L2-normalised sum of the Qwen query vectors of training rows citing the article, each weighted "
                         "1/|positives|; cosine to the prototype, uncited articles get the row minimum; variant and weighting mode by inner CV",
                         "proto")}
    for m, (text, fam) in descriptions.items():
        manifest = {"method": m, "description": text,
                    "configuration": {g.role: sel[fam][g.role]["alone"]["config"] for g in ctx.groups},
                    "inner_objective": {g.role: sel[fam][g.role]["alone"]["objective"] for g in ctx.groups},
                    "dev_oof": results["alone"][m], "input_sha256": input_hashes(), "endpoint": endpoint_identity(), "seeds": None,
                    **{k: v for k, v in common_manifest.items() if k != "selection_evidence"},
                    "selection_evidence": {**common_manifest["selection_evidence"], "selection_file": str(SELECTION_DIR / f"{m}.json")}}
        common.finish_scores(m, manifest, CODE_FILES)
    common.finish_scores("K_qwen8b", {
        "method": "K_qwen8b", "description": "robust-z fusion of L_star and the best Qwen supervised source (kNN / fused-key kNN / prototype); "
        "source, configuration and weight chosen per group by inner CV on the fused objective",
        "configuration": {g.role: {k: kq[g.role][k] for k in ("family", "config", "w")} for g in ctx.groups},
        "inner_objective": {g.role: kq[g.role]["objective"] for g in ctx.groups}, "depends_on": ["L_star"], "dev_oof": results["K_qwen8b"],
        "input_sha256": input_hashes(), "endpoint": endpoint_identity(), "seeds": None,
        **{k: v for k, v in common_manifest.items() if k != "selection_evidence"},
        "selection_evidence": {**common_manifest["selection_evidence"], "selection_file": str(SELECTION_DIR / "K_qwen8b.json"),
                               "lstar": "recomputed under each group's own L* selection for the inner folds"}}, CODE_FILES)
    log(f"supervised stage done in {time.perf_counter() - t0:.0f}s")
    return results


def check_klex_parity(rows_per_group: int = 60) -> dict[str, float]:
    """Plumbing check: the lexical-key votes (k=50) fused with L* at w=0.25 must reproduce the ``K_lex`` file's rows.

    K_lex selected (k=50, w=0.25) in every group, so a fit on the group's training folds, BM25 similarity to memory
    representatives and robust-z fusion with the L_star file give back the same scores up to float error.
    """
    ctx = get_ctx()
    lfile, kfile = common.open_scores("L_star"), common.open_scores("K_lex")
    manifest = json.loads((common.score_path("K_lex").parent / "manifest.json").read_text(encoding="utf-8"))["configuration"]
    worst = 0.0
    for g in ctx.groups[:5]:
        cfg = manifest[g.role]
        rows = g.rows[:rows_per_group]
        sims = Sims(ctx.fit(g.train_folds), rows)
        votes = config_block(sims, f"lex|k{cfg['k']}")
        fb = fusion.linear_fusion([fusion.robust_z(np.asarray(lfile[rows])), fusion.robust_z(votes)], pair(cfg["w_knn"]), normalize="none")
        worst = max(worst, float(np.abs(fb - np.asarray(kfile[rows])).max()))
    return {"max_abs_diff_vs_K_lex": worst}


def stage_klex_ext(force: bool = False) -> None:
    """``K_lex_ext`` (NO-EMB diagnostic): the K_lex pipeline with the grid extended to k = 100 and w in ``W_VOTES``.

    K_lex chose k = 50 and w = 0.25 in every group, both at the edge of its grid (k <= 50, w >= 0.25). The lexical-key control of the
    inner-CV grid here (same memory, same BM25 key, same L* and robust-z fusion; k in ``KS``, w in ``W_VOTES``) is delivered as a file
    so the lead can compare Qwen methods with a non-embedding reference whose grid does not bind. No embedding is used.
    """
    ctx = get_ctx()
    sel = json.loads((SELECTION_DIR / "lex_control.json").read_text(encoding="utf-8"))
    lfile = common.open_scores("L_star")

    def producer(matrix: np.ndarray) -> None:
        for g in ctx.groups:
            f = sel[g.role]["fused"]
            fit = ctx.fit(g.train_folds)
            for s in range(0, len(g.rows), 512):
                rows = g.rows[s:s + 512]
                votes = config_block(Sims(fit, rows), f["config"])
                matrix[rows] = fusion.linear_fusion([fusion.robust_z(np.asarray(lfile[rows])), fusion.robust_z(votes)], pair(f["w"]),
                                                    normalize="none")

    _write_scores("K_lex_ext", producer, {
        "family": "NO-EMB", "label_free": False,
        "description": "K_lex with the grid extended (k up to 100, w_knn from 0.05): robust-z fusion of L_star and BM25-key kNN citation votes; "
                       "(k, w) chosen per group by nested inner CV; a diagnostic NO-EMB reference produced alongside the Qwen methods",
        "training_data": "labels of the scoring group's training dev rows only (lockbox group: all dev rows); the lockbox never trains",
        "configuration": {role: {"k": int(sel[role]["fused"]["config"].split("|k")[1]), "w_knn": sel[role]["fused"]["w"]} for role in sel},
        "inner_objective": {role: sel[role]["fused"]["objective"] for role in sel},
        "depends_on": ["L_star"],
        "selection_evidence": {"protocol": "inner CV as for the Qwen supervised methods (same inner tasks)", "selection_file": str(SELECTION_DIR / "lex_control.json"),
                               "search_log": str(SEARCH_LOG), "k_grid": list(KS), "w_grid": list(W_VOTES)},
        "device": "cpu"}, force, embeddings=False)


# --------------------------------------------------------------------------- 3b. supervised sources on top of H (L* + Qwen dense)


def h_weights() -> dict[str, float]:
    """Per scoring group dense weight of ``H_qwen8b`` (selected on the group's training rows only)."""
    return {role: float(v["w_dense"]) for role, v in json.loads((SELECTION_DIR / "H_qwen8b.json").read_text(encoding="utf-8")).items()}


def inner_h_path(role: str, inner: int) -> Path:
    return INNER_DIR / f"H_{role}_f{inner}.npz"


def inner_task_h(g_idx: int, inner: int, force: bool = False) -> str:
    """Like ``inner_task`` but the base channel is H = (1-w) z(L*) + w z(dense_qwen8b) with the group's own H weight.

    Stores per-query nDCG@10 (specific view) of z(H) fused with every configuration's vote/prototype channel at each
    ``W_VOTES`` weight, including the lexical-key control (so ``lex`` + H is the K_lex pipeline plus Qwen dense).
    """
    ctx = get_ctx()
    g = ctx.groups[g_idx]
    path = inner_h_path(g.role, inner)
    if path.exists() and not force:
        return str(path)
    t0 = time.perf_counter()
    fit_folds = tuple(f for f in g.train_folds if f != inner)
    rows = ctx.pspec[ctx.fold[ctx.pspec] == inner]
    fit = ctx.fit(fit_folds)
    zl = fusion.robust_z(_CTX["ls"].lstar(rows, _CTX["sel"][g.role]))
    zd = fusion.robust_z(np.asarray(common.open_scores("dense_qwen8b")[rows]))
    zh = fusion.robust_z(fusion.linear_fusion([zl, zd], pair(h_weights()[g.role]), normalize="none"))
    sims = Sims(fit, rows)
    out: dict[str, np.ndarray] = {"rows": rows}
    for configs in config_grid().values():
        for cfg in configs:
            zb = fusion.robust_z(config_block(sims, cfg))
            out[f"{cfg}#fusedH"] = np.stack([ctx.ndcg_spec(fusion.linear_fusion([zh, zb], pair(w), normalize="none"), rows)
                                             for w in W_VOTES])
    out["seconds"] = np.asarray([time.perf_counter() - t0])
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    os.replace(tmp, path)
    log(f"inner H task {g.role} fold {inner}: {len(rows)} P_spec rows, {time.perf_counter() - t0:.0f}s")
    return str(path)


def _inner_h_star(args: tuple[int, int, bool]) -> str:
    return inner_task_h(*args)


def run_inner_h(workers: int = 3, force: bool = False) -> None:
    import multiprocessing as mp

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(max(1, 12 // max(workers, 1)))
    ctx = get_ctx()
    todo = [(g.idx, f, force) for g in ctx.groups for f in g.train_folds if force or not inner_h_path(g.role, f).exists()]
    log(f"inner H tasks: {len(todo)} to run with {workers} workers")
    if not todo:
        return
    if workers <= 1:
        for t in todo:
            inner_task_h(*t)
        return
    from uol.embedding_push import qwen_endpoint as module

    with mp.get_context("spawn").Pool(workers) as pool:
        for _ in pool.imap_unordered(module._inner_h_star, todo):
            pass


def stage_supervised_h(force: bool = False, workers: int = 3) -> dict[str, Any]:
    """``KH_qwen8b``: H fused with the best Qwen supervised source; also reports every family on top of H and the lexical-key
    control (K_lex pipeline + Qwen dense). (family, configuration, weight) chosen per group by the same nested inner CV."""
    import shutil

    t0 = time.perf_counter()
    ctx = get_ctx()
    run_inner_h(workers)
    families = {"knn": "knn_qwen8b", "kf": "knn_fused_qwen8b", "proto": "proto_qwen8b", "lex": "lex_control"}
    reset_search_log("inner_H")
    best: dict[str, dict[str, Any]] = {fam: {} for fam in families}
    tables: dict[str, Any] = {fam: {} for fam in families}
    for g in ctx.groups:
        parts = [np.load(inner_h_path(g.role, f)) for f in g.train_folds]
        rows = np.concatenate([p["rows"] for p in parts])
        for fam in families:
            cfgs = config_grid()[fam]
            flat = [(c, j) for c in cfgs for j in range(len(W_VOTES))]
            values = []
            for c, j in flat:
                values.append(ctx.pooled(np.concatenate([p[f"{c}#fusedH"][j] for p in parts]), rows))
            jbest = argbest(values)
            best[fam][g.role] = {"config": flat[jbest][0], "w": W_VOTES[flat[jbest][1]], "objective": values[jbest]}
            tables[fam][g.role] = [{"config": c, "w": W_VOTES[j], "objective": v} for (c, j), v in zip(flat, values)]
            search_log({"stage": "inner_H", "method": f"KH_{families[fam]}", "group": g.role, "chosen": best[fam][g.role]})
        log(f"{g.role}: on top of H: " + "; ".join(f"{fam} {best[fam][g.role]['config']} w={best[fam][g.role]['w']} "
                                                  f"({best[fam][g.role]['objective']:.4f})" for fam in families))
    pick = {}
    for g in ctx.groups:
        options = [best[fam][g.role] for fam in ("knn", "kf", "proto")]
        i = argbest([o["objective"] for o in options])
        pick[g.role] = {"family": ("knn", "kf", "proto")[i], **options[i]}
    (SELECTION_DIR / "KH_qwen8b.json").write_text(json.dumps({"best": best, "pick": pick, "tables": tables}, indent=1), encoding="utf-8")
    folder = common.score_path("KH_qwen8b").parent
    if folder.exists():
        if (folder / "manifest.json").exists() and not force:
            log("KH_qwen8b already complete")
            return {"best": best, "pick": pick}
        shutil.rmtree(folder)
    mat = common.create_scores("KH_qwen8b")
    dev = common.dev_rows()
    dev_pos = {int(r): i for i, r in enumerate(dev)}
    fused_dev = scratch_matrices(list(families), len(dev), ctx.N)
    hfile = common.open_scores("H_qwen8b")
    for g in ctx.groups:
        fit = ctx.fit(g.train_folds)
        for s in range(0, len(g.rows), 512):
            rows = g.rows[s:s + 512]
            sims, memo = Sims(fit, rows), {}

            def blk(cfg: str) -> np.ndarray:
                if cfg not in memo:
                    memo[cfg] = config_block(sims, cfg)
                return memo[cfg]

            zh = fusion.robust_z(np.asarray(hfile[rows]))
            for fam in families:
                f = best[fam][g.role]
                fb = fusion.linear_fusion([zh, fusion.robust_z(blk(f["config"]))], pair(f["w"]), normalize="none")
                if g.role != "lockbox":
                    fused_dev[fam][[dev_pos[int(r)] for r in rows]] = fb
                if fam == pick[g.role]["family"]:
                    mat[rows] = fb
        log(f"KH outer scoring {g.role}: {len(g.rows)} rows ({time.perf_counter() - t0:.0f}s)")
    mat.flush()
    del mat
    results = {"on_top_of_H": {families[fam]: dev_metrics(fused_dev[fam]) for fam in families}, "KH_qwen8b": method_dev_metrics("KH_qwen8b")}
    del fused_dev
    drop_scratch()
    (RUN / "results_supervised_h.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    inner_seconds = float(sum(np.load(inner_h_path(g.role, f))["seconds"][0] for g in ctx.groups for f in g.train_folds))
    common.finish_scores("KH_qwen8b", {
        "method": "KH_qwen8b", "family": "EMB", "label_free": False,
        "description": "robust-z fusion of H_qwen8b (L_star + Qwen dense) and the best Qwen supervised source (kNN / fused-key kNN / prototype); "
                       "family, configuration and weight chosen per group by nested inner CV on the fused objective",
        "configuration": {role: {k: pick[role][k] for k in ("family", "config", "w")} for role in pick},
        "inner_objective": {role: pick[role]["objective"] for role in pick}, "depends_on": ["H_qwen8b", "L_star", "dense_qwen8b"],
        "training_data": "labels of the scoring group's training dev rows only (lockbox group: all dev rows); the lockbox never trains",
        "dev_oof": results["KH_qwen8b"], "input_sha256": input_hashes(), "endpoint": endpoint_identity(), "seeds": None, "device": "cpu",
        "wall_seconds": inner_seconds + (time.perf_counter() - t0),
        "selection_evidence": {"protocol": "inner CV as for K_qwen8b with H (group's own dense weight) as the base channel; ties: lowest grid index",
                               "selection_file": str(SELECTION_DIR / "KH_qwen8b.json"), "search_log": str(SEARCH_LOG), "w_votes": list(W_VOTES)}},
        CODE_FILES)
    log(f"supervised-on-H stage done in {time.perf_counter() - t0:.0f}s")
    return results


# --------------------------------------------------------------------------- 4a. optional low-rank query adapter

ADAPT_DIR = RUN / "adapt"
ADAPT_GRID = [{"lam": lam, "lr": lr} for lam in (0.0, 0.5, 1.0) for lr in (1e-3, 3e-4)]
ADAPT_RANK, ADAPT_BATCH, ADAPT_MAX_EPOCHS, ADAPT_PATIENCE, ADAPT_SEED = 64, 256, 40, 5, 20261004


def first_window_vectors() -> np.ndarray:
    """One frozen vector per article for training (its first window; articles above 8,000 tokens are the only ones with more)."""
    vectors, owner = article_vectors()
    first = np.searchsorted(owner, np.arange(len(common.release().candidate_ids)), side="left")
    if not (owner[first] == np.arange(len(first))).all():
        raise AssertionError("every article must own at least one unit")
    return vectors[first]


def adapt_apply(q: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """q' = normalise(q + (q V) U^T) for unit query vectors ``q`` (rank-r residual map)."""
    return unit_rows(q + (q @ V) @ U.T)


def train_adapter(ctx: Ctx, variant: str, train_rows: np.ndarray, val_rows: np.ndarray | None, *, lam: float, lr: float,
                  epochs: int, rank: int = ADAPT_RANK, seed: int = ADAPT_SEED, device: str = "cuda",
                  patience: int = ADAPT_PATIENCE) -> dict[str, Any]:
    """Train U, V on ``train_rows`` against frozen article vectors (full-catalogue multi-positive softmax).

    Loss per query: mean over its positives of -log softmax_j(s * cos(q', a_j) + lam * log(1 + n_j)); ``n_j`` counts the
    training rows citing article j (logit adjustment applied only while training; scoring uses the raw cosine). ``U`` starts
    at zero, so epoch 0 is exactly the frozen model. With ``val_rows`` (P_spec rows of a training fold) the epoch with the best
    pooled P_spec nDCG@10 is kept (early stopping, labels of training rows only); otherwise ``epochs`` epochs are run.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = ctx.data
    N, dim = ctx.N, DIM
    A = torch.as_tensor(first_window_vectors(), device=device)
    q_all = query_vectors(variant)
    rows = np.asarray([int(r) for r in train_rows if d.positive_idx[int(r)]], dtype=np.int64)
    counts = np.zeros(N, dtype=np.float32)
    for r in rows:
        counts[d.positive_idx[int(r)]] += 1.0
    adjust = torch.as_tensor(lam * np.log1p(counts), device=device)
    V = torch.nn.Parameter(torch.randn(dim, rank, device=device) / np.sqrt(dim))
    U = torch.nn.Parameter(torch.zeros(dim, rank, device=device))
    log_s = torch.nn.Parameter(torch.tensor(float(np.log(20.0)), device=device))
    opt = torch.optim.Adam([V, U, log_s], lr=lr)
    Q = torch.as_tensor(q_all, device=device)

    def adapted(idx: np.ndarray) -> Any:
        q = Q[torch.as_tensor(idx, device=device)]
        z = q + (q @ V) @ U.T
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)

    def validate() -> float:
        with torch.no_grad():
            scores = np.concatenate([(adapted(val_rows[s:s + 512]) @ A.T).cpu().numpy() for s in range(0, len(val_rows), 512)])
        return ctx.pooled(ctx.ndcg_spec(scores, val_rows), val_rows)

    history: list[dict[str, float]] = []
    best = (validate() if val_rows is not None else float("nan"), 0, (U.detach().clone(), V.detach().clone(), log_s.detach().clone()))
    history.append({"epoch": 0, "val_ndcg10": best[0]})
    stale = 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(rows))
        total = 0.0
        for s in range(0, len(order), ADAPT_BATCH):
            idx = rows[order[s:s + ADAPT_BATCH]]
            mask = torch.zeros(len(idx), N, device=device)
            for i, r in enumerate(idx):
                mask[i, d.positive_idx[int(r)]] = 1.0
            logits = torch.exp(log_s).clamp(max=100.0) * (adapted(idx) @ A.T) + adjust
            loss = -((torch.log_softmax(logits, dim=1) * mask).sum(1) / mask.sum(1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        row = {"epoch": epoch, "train_loss": total / len(rows)}
        if val_rows is not None:
            row["val_ndcg10"] = validate()
            if row["val_ndcg10"] > best[0] + 1e-9:
                best, stale = (row["val_ndcg10"], epoch, (U.detach().clone(), V.detach().clone(), log_s.detach().clone())), 0
            else:
                stale += 1
        history.append(row)
        if val_rows is not None and stale >= patience:
            break
    if val_rows is None:
        best = (float("nan"), epochs, (U.detach().clone(), V.detach().clone(), log_s.detach().clone()))
    Ub, Vb, ls = best[2]
    result = {"U": Ub.cpu().numpy(), "V": Vb.cpu().numpy(), "scale": float(torch.exp(ls)), "best_epoch": best[1],
              "best_val_ndcg10": best[0], "history": history, "n_train_rows": len(rows)}
    del A, Q, opt, V, U
    torch.cuda.empty_cache()
    return result


def stage_adapt(force: bool = False, device: str = "cuda") -> dict[str, Any]:
    """Optional ``adapt_qwen8b``: pilot on dev fold 0, keep (and score all groups) only if it beats frozen Qwen there.

    Pilot: train on folds 1-4 minus validation fold 4 for each grid point, pick (lam, lr, best epoch) on fold 4 P_spec
    nDCG@10, retrain on folds 1-4 for that many epochs and compare fold 0 P_spec nDCG@10 with ``dense_qwen8b`` on the same
    rows. Every grid point is logged. If kept, every scoring group trains the same way on its own training folds (the
    group's highest training fold is its validation fold) and scores all of its rows.
    """
    t0 = time.perf_counter()
    ctx = get_ctx()
    ADAPT_DIR.mkdir(parents=True, exist_ok=True)
    variant = json.loads((SELECTION_DIR / "dense.json").read_text(encoding="utf-8"))["selected"]
    dense = common.open_scores("dense_qwen8b")
    pilot_path = ADAPT_DIR / "pilot.json"
    g0 = ctx.groups[0]
    rows0 = np.intersect1d(g0.rows, ctx.pspec)
    frozen0 = ctx.pooled(ctx.ndcg_spec(np.asarray(dense[rows0]), rows0), rows0)
    if pilot_path.exists() and not force:
        pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    else:
        val_fold = max(g0.train_folds)
        val_rows = ctx.pspec[ctx.fold[ctx.pspec] == val_fold]
        fit_rows = np.flatnonzero(np.isin(ctx.fold, [f for f in g0.train_folds if f != val_fold]))
        grid = []
        with common.gpu_lock(owner="EP-epsilon"):                      # one lock hold for the whole pilot job
            for cfg in ADAPT_GRID:
                t = time.perf_counter()
                res = train_adapter(ctx, variant, fit_rows, val_rows, epochs=ADAPT_MAX_EPOCHS, device=device, **cfg)
                grid.append({**cfg, "best_epoch": res["best_epoch"], "val_ndcg10": res["best_val_ndcg10"], "seconds": time.perf_counter() - t})
                search_log({"stage": "adapt_pilot", "method": "adapt_qwen8b", "group": g0.role, "config": cfg, "val_fold": val_fold,
                            "val_P_spec_ndcg10": res["best_val_ndcg10"], "best_epoch": res["best_epoch"], "val_frozen": res["history"][0]["val_ndcg10"]})
                log(f"adapt pilot {cfg}: val fold {val_fold} nDCG@10 {res['history'][0]['val_ndcg10']:.4f} -> {res['best_val_ndcg10']:.4f} "
                    f"at epoch {res['best_epoch']} ({grid[-1]['seconds']:.0f}s)")
            best = grid[argbest([x["val_ndcg10"] for x in grid])]
            res = train_adapter(ctx, variant, np.asarray(g0.train), None, lam=best["lam"], lr=best["lr"],
                                epochs=max(best["best_epoch"], 1), device=device)
        q0 = adapt_apply(query_vectors(variant)[rows0], res["U"], res["V"])
        adapted0 = ctx.pooled(ctx.ndcg_spec(dense_block(q0), rows0), rows0)
        pilot = {"variant": variant, "grid": grid, "chosen": best, "fold0_P_spec_frozen": frozen0, "fold0_P_spec_adapted": adapted0,
                 "keep": bool(adapted0 > frozen0 + 1e-9), "n_rows_fold0": int(len(rows0))}
        pilot_path.write_text(json.dumps(pilot, indent=1), encoding="utf-8")
        log(f"adapt pilot fold 0: frozen {frozen0:.4f} vs adapted {adapted0:.4f}; keep={pilot['keep']}")
    if not pilot["keep"]:
        return pilot
    import shutil

    folder = common.score_path("adapt_qwen8b").parent
    if folder.exists():
        if (folder / "manifest.json").exists() and not force:
            return pilot
        shutil.rmtree(folder)
    mat = common.create_scores("adapt_qwen8b")
    q = query_vectors(variant)
    chosen, groups = pilot["chosen"], {}
    trained = {}
    with common.gpu_lock(owner="EP-epsilon"):                          # one lock hold for the six-group job
        for g in ctx.groups:
            val_fold = max(g.train_folds)
            val_rows = ctx.pspec[ctx.fold[ctx.pspec] == val_fold]
            fit_rows = np.flatnonzero(np.isin(ctx.fold, [f for f in g.train_folds if f != val_fold]))
            first = train_adapter(ctx, variant, fit_rows, val_rows, lam=chosen["lam"], lr=chosen["lr"], epochs=ADAPT_MAX_EPOCHS, device=device)
            final = train_adapter(ctx, variant, np.asarray(g.train), None, lam=chosen["lam"], lr=chosen["lr"],
                                  epochs=max(first["best_epoch"], 1), device=device)
            trained[g.role] = (first, final, val_fold)
            log(f"adapt {g.role}: trained ({max(first['best_epoch'], 1)} epochs; validation fold {val_fold} nDCG@10 "
                f"{first['history'][0]['val_ndcg10']:.4f} -> {first['best_val_ndcg10']:.4f})")
    for g in ctx.groups:
        first, final, val_fold = trained[g.role]
        np.savez(ADAPT_DIR / f"{g.role}.npz", U=final["U"], V=final["V"], scale=final["scale"])
        for s in range(0, len(g.rows), 512):
            rows = g.rows[s:s + 512]
            dense_block(adapt_apply(q[rows], final["U"], final["V"]), out=mat[rows])
        groups[g.role] = {"epochs": max(first["best_epoch"], 1), "val_fold": val_fold, "val_ndcg10": first["best_val_ndcg10"],
                          "val_frozen": first["history"][0]["val_ndcg10"], "n_train_rows": final["n_train_rows"], "scale": final["scale"]}
        log(f"adapt {g.role}: {groups[g.role]}")
    mat.flush()
    del mat
    common.finish_scores("adapt_qwen8b", {
        "method": "adapt_qwen8b", "family": "EMB", "label_free": False,
        "description": "Qwen3-Embedding-8B query vectors passed through a rank-64 residual map (q + qVU^T, renormalised), trained against the frozen "
                       "article vectors with full-catalogue multi-positive softmax and logit adjustment lam*log(1+count); MaxP scoring",
        "training_data": "labels of the scoring group's training dev rows only (lockbox group: all dev rows); early-stopping epoch from the group's "
                         "highest training fold; the lockbox never trains",
        "configuration": {"variant": variant, "rank": ADAPT_RANK, "pilot_chosen": chosen, "groups": groups, "batch": ADAPT_BATCH, "optimizer": "Adam"},
        "selection_evidence": {"pilot": pilot, "protocol": "(lam, lr, epochs) chosen on validation fold 4 inside the fold-0 training rows; kept because "
                                                           "fold-0 P_spec nDCG@10 beat frozen Qwen", "search_log": str(SEARCH_LOG)},
        "seeds": {"torch_and_numpy": ADAPT_SEED}, "input_sha256": input_hashes(), "endpoint": endpoint_identity(), "device": device,
        "wall_seconds": time.perf_counter() - t0}, CODE_FILES)
    log(f"adapt stage done in {time.perf_counter() - t0:.0f}s")
    return pilot


# --------------------------------------------------------------------------- 4b. optional HyDE rewrite (LLaDA2.1-mini)

HYDE_DIR = RUN / "hyde"
HYDE_GEN = HYDE_DIR / "generations.jsonl"
HYDE_PARAMS: dict[str, Any] = {"max_tokens": 400, "temperature": 0.0}
HYDE_PROMPT = (
    "Eres un jurista mexicano experto en legislación federal. A continuación se describen los hechos de un asunto resuelto por un "
    "tribunal mexicano; las referencias a artículos están ocultas.\n\n"
    "Hechos:\n<<HECHOS>>\n\n"
    "Escribe un texto breve (de 80 a 150 palabras, en un solo párrafo) con el tipo de disposiciones de leyes federales mexicanas y las "
    "cuestiones jurídicas que serían aplicables a estos hechos, redactado como si fuera el texto de esas disposiciones. No incluyas "
    "números de artículo, no inventes citas y no repitas los hechos.")
HYDE_PILOT_ROWS = 300
HYDE_MARGIN = 0.01                       # pre-registered: scale up only if the primary gain is at least this on the pilot rows
HYDE_PRIMARY = {"fusion": "robust_z_linear", "w_hyde": 0.5}
LLM_NOTE = ("The lockbox tesis (published mid-2025 to 2026) may post-date the training data of LLaDA2.1-mini and of Qwen3-Embedding-8B, "
            "while the dev tesis may be inside it; dev results of any Qwen/LLaDA method can therefore be optimistic for the lockbox.")


def hyde_prompt(facts: str) -> str:
    return HYDE_PROMPT.replace("<<HECHOS>>", facts)


# The model often ignores "no article numbers" and invents citations (seen in a two-row format probe before any pilot score existed),
# so explicit article-number mentions are removed from the generated text before it is embedded. This is a fixed textual rule, not label use.
_ONE_ARTICLE = r"\d+[a-zA-Z°º]{0,3}\b\.?(?:\s+(?:bis|ter|qu[aá]ter)\b)?"
_ARTICLE_MENTION = re.compile(rf"(?i)\b(?:art[íi]culos?|arts?\.)\s*{_ONE_ARTICLE}(?:\s*(?:,|y|e|al|a)\s*{_ONE_ARTICLE})*")


def strip_citations(text: str) -> str:
    """Replace article-number mentions ("artículo 309", "arts. 12 y 13") by a neutral phrase; whitespace is normalised."""
    return re.sub(r"\s+", " ", _ARTICLE_MENTION.sub("la disposición aplicable", text)).strip()


def hyde_key(prompt: str, params: dict[str, Any] = HYDE_PARAMS) -> str:
    blob = json.dumps({"model": LLM_MODEL, "params": params, "prompt": prompt}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8", "surrogatepass")).hexdigest()


def load_generations(path: Path = HYDE_GEN) -> dict[str, dict[str, Any]]:
    return {r["key"]: r for r in read_jsonl(path)}


def generate_hyde(rows: Sequence[int], *, workers: int = MAX_CONCURRENCY, label: str = "hyde", path: Path = HYDE_GEN,
                  client: Client | None = None, progress: Callable[[str], None] = log) -> dict[int, str]:
    """Generated hypothetical-statute text per release row (cached by prompt hash; at most ``workers`` <= 4 requests in flight).

    Only the masked Hechos (``q.text``) are sent, inside a fixed Spanish prompt; no labels, no article numbers.
    """
    client = client or Client(INSTRUCT_BASE, log_path=HYDE_DIR / "requests.jsonl", timeout=(20.0, 900.0))
    texts = common.release().queries
    cache = load_generations(path)
    todo = []
    for r in rows:
        prompt = hyde_prompt(texts[int(r)].text)
        key = hyde_key(prompt)
        if key not in cache:
            todo.append((int(r), key, prompt))
    started, done = time.monotonic(), 0

    def check(data: dict[str, Any]) -> None:
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty completion")

    def one(item: tuple[int, str, str]) -> dict[str, Any]:
        r, key, prompt = item
        t = time.monotonic()
        data = client.post("/chat/completions", {"model": LLM_MODEL, "messages": [{"role": "user", "content": prompt}], **HYDE_PARAMS},
                           tag=label, n_inputs=1, validate=check)
        return {"key": key, "row": r, "text": data["choices"][0]["message"]["content"].strip(), "usage": data.get("usage"),
                "finish_reason": data["choices"][0].get("finish_reason"), "latency_s": round(time.monotonic() - t, 2), "utc": utc()}

    pending: dict[cf.Future, tuple[int, str, str]] = {}
    queue = list(reversed(todo))
    failures = 0
    with cf.ThreadPoolExecutor(max_workers=min(int(workers), MAX_CONCURRENCY)) as pool:
        while queue or pending:
            while queue and len(pending) < min(int(workers), MAX_CONCURRENCY) and not STOP_FILE.exists() and failures < 5:
                item = queue.pop()
                pending[pool.submit(one, item)] = item
            if not pending:
                break
            finished, _ = cf.wait(list(pending), return_when=cf.FIRST_COMPLETED)
            for future in finished:
                item = pending.pop(future)
                try:
                    row = future.result()
                except EndpointError as exc:
                    failures += 1
                    progress(f"{label}: row {item[0]} failed ({exc})")
                    continue
                append_jsonl(path, row)
                cache[row["key"]] = row
                done += 1
                if done % 25 == 0 or not queue and not pending:
                    el = time.monotonic() - started
                    progress(f"{label}: {done}/{len(todo)} generations, {el:.0f}s elapsed, {done / max(el, 1e-9):.2f}/s, "
                             f"requests {client.stats['requests']}, retries {client.stats['retries']}, "
                             f"completion tokens {client.stats['completion_tokens']}")
    out = {}
    for r in rows:
        key = hyde_key(hyde_prompt(texts[int(r)].text))
        if key in cache:
            out[int(r)] = cache[key]["text"]
    missing = [int(r) for r in rows if int(r) not in out]
    if missing:
        raise EndpointError(None, f"{len(missing)} rows have no generation, e.g. {missing[:5]}; rerun to resume")
    return out


def embed_texts(texts: Sequence[str], label: str = "hyde") -> np.ndarray:
    """Unit Qwen vectors of plain texts (no instruction; documents-style), through the shared content-hash store."""
    store = VectorStore()
    try:
        counts = count_tokens(load_tokenizer(), texts)
    except Exception:                      # the cached tokenizer files can be moved away: estimate conservatively (~3 characters per token)
        counts = np.asarray([len(t) // 3 + 1 for t in texts], dtype=np.int64)
    if counts.size and counts.max() > ART_MAX_TOKENS:
        raise ValueError("a generated text exceeds the embedding window")
    its = [(content_key(t), t, int(counts[i]) + 1) for i, t in enumerate(texts)]
    client = Client(EMB_BASE)
    encode_items(client, store, its, label=label)
    return unit_rows(store.get([k for k, _, _ in its]))


def hyde_pilot_rows(ctx: Ctx) -> np.ndarray:
    """300 rows of dev fold 0: every P_spec row (228) plus the first of a seeded shuffle of the other fold-0 P rows."""
    p_rows = common.population_rows("P", ctx.groups[0].rows)
    spec = np.intersect1d(p_rows, ctx.pspec)
    rest = np.setdiff1d(p_rows, spec)
    rest = rest[np.random.default_rng(ADAPT_SEED).permutation(len(rest))]
    return np.sort(np.concatenate([spec, rest[:max(0, HYDE_PILOT_ROWS - len(spec))]]))[:HYDE_PILOT_ROWS]


def stage_hyde_pilot(force: bool = False) -> dict[str, Any]:
    """Pilot on dev fold 0 (300 rows): generate, embed, fuse with the query score, compare with query-only on the same rows.

    Pre-registered before any pilot score exists (``hyde/pilot_plan.json``): the primary candidate is the equal-weight robust-z
    fusion of the query dense score and the HyDE dense score; the pilot "clearly helps" if its P_spec nDCG@10 gain over the
    query-only ``dense_qwen8b`` on the pilot rows is at least ``HYDE_MARGIN`` and the paired component-bootstrap 90% interval
    excludes zero. Other fusions are exploratory and cannot trigger the scale-up.
    """
    t0 = time.perf_counter()
    HYDE_DIR.mkdir(parents=True, exist_ok=True)
    ctx = get_ctx()
    rows = hyde_pilot_rows(ctx)
    plan = {"registered_utc": utc(), "rows": "all fold-0 P_spec rows + seeded fold-0 P-only rows, 300 in total", "n_rows": int(len(rows)),
            "prompt_sha256": sha256_text(HYDE_PROMPT), "generation_params": HYDE_PARAMS, "model": LLM_MODEL,
            "post_processing": "article-number mentions removed by a fixed regex (strip_citations) before embedding; the model ignores the "
                               "'no article numbers' instruction (seen on a two-row format probe, before any pilot score existed)",
            "embedding": "plain text, no instruction (document style), Qwen3-Embedding-8B",
            "primary": HYDE_PRIMARY, "decision_rule": f"scale up iff primary P_spec nDCG@10 gain >= {HYDE_MARGIN} on the pilot rows and the "
                                                      "paired component bootstrap 90% interval excludes 0",
            "note": LLM_NOTE}
    plan_path = HYDE_DIR / "pilot_plan.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    gens = generate_hyde(rows, label="hyde_pilot")
    h = embed_texts([strip_citations(gens[int(r)]) for r in rows], "hyde_pilot_embed")
    sq = np.asarray(common.open_scores("dense_qwen8b")[rows])
    sh = dense_block(h)
    zq, zh = fusion.robust_z(sq), fusion.robust_z(sh)
    variant = json.loads((SELECTION_DIR / "dense.json").read_text(encoding="utf-8"))["selected"]
    qv = query_vectors(variant)[rows]
    candidates: dict[str, np.ndarray] = {"query_only": sq, "hyde_only": sh}
    for w in (0.25, 0.5, 0.75):
        candidates[f"zfuse_w{w}"] = fusion.linear_fusion([zq, zh], pair(w), normalize="none")
    for beta in (0.5, 1.0):
        candidates[f"vecmix_b{beta}"] = dense_block(unit_rows(qv + np.float32(beta) * h))
    evals = {name: {pop: common.evaluate(s, rows, pop) for pop in ("P_spec", "P")} for name, s in candidates.items()}
    table = {}
    for name, e in evals.items():
        table[name] = {pop: {"ndcg_at_10": e[pop]["macro"]["ndcg_at_10"], "recall_at_20": e[pop]["macro"]["recall_at_20"],
                             "n_queries": e[pop]["n_queries"]} for pop in e}
        if name != "query_only":
            c = common.contrast(e["P_spec"], evals["query_only"]["P_spec"])
            table[name]["P_spec_gain_vs_query_only"] = {k: c[k] for k in ("effect", "ci95", "ci90", "p_sign_flip", "n")}
        search_log({"stage": "hyde_pilot", "method": "dense_qwen8b_hyde", "candidate": name, "rows": "dev fold 0 pilot", "metrics": table[name]})
    primary = f"zfuse_w{HYDE_PRIMARY['w_hyde']}"
    gain = table[primary]["P_spec_gain_vs_query_only"]
    keep = bool(gain["effect"] >= HYDE_MARGIN and gain["ci90"][0] > 0)
    usage = [r for r in load_generations().values() if r["row"] in set(int(x) for x in rows)]
    pilot = {"plan": plan, "table": table, "primary": primary, "primary_gain": gain, "scale_up": keep, "seconds": time.perf_counter() - t0,
             "generations": len(usage), "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in usage),
             "mean_generation_latency_s": float(np.mean([r["latency_s"] for r in usage])) if usage else None, "note": LLM_NOTE}
    (HYDE_DIR / "pilot.json").write_text(json.dumps(pilot, indent=1, ensure_ascii=False), encoding="utf-8")
    log(f"HyDE pilot: query-only P_spec {table['query_only']['P_spec']['ndcg_at_10']:.4f}; primary {primary} "
        f"{table[primary]['P_spec']['ndcg_at_10']:.4f} (gain {gain['effect']:+.4f}, 90% CI {gain['ci90'][0]:+.4f}..{gain['ci90'][1]:+.4f}); scale_up={keep}")
    return pilot


def stage_hyde_scale(force: bool = False) -> dict[str, Any]:
    """Generate for all 5,980 rows, embed, and deliver ``dense_qwen8b_hyde`` (robust-z fusion of the query and HyDE dense scores).

    The HyDE weight is chosen per scoring group on its training rows (label-free scores, selection only), exactly like ``H_qwen8b``
    but without L_star. Runs only after the pilot decided to scale.
    """
    t0 = time.perf_counter()
    pilot = json.loads((HYDE_DIR / "pilot.json").read_text(encoding="utf-8"))
    if not pilot["scale_up"]:
        raise RuntimeError("the pilot did not clearly help; the full run is not authorised by the plan")
    ctx = get_ctx()
    all_rows = np.arange(len(ctx.data.queries))
    gens = generate_hyde(all_rows, label="hyde_full")
    h = embed_texts([strip_citations(gens[int(r)]) for r in all_rows], "hyde_full_embed")
    np.save(DERIVED / "hyde_vectors.npy", h)
    dq = common.open_scores("dense_qwen8b")
    rows = ctx.pspec
    zq = fusion.robust_z(np.asarray(dq[rows]))
    zh = fusion.robust_z(dense_block(h[rows]))
    table = np.stack([ctx.ndcg_spec(fusion.linear_fusion([zq, zh], pair(w), normalize="none"), rows) for w in W_DENSE])
    chosen = {}
    for g in ctx.groups:
        mask = np.isin(rows, g.train)
        objective = [ctx.pooled(table[j][mask], rows[mask]) for j in range(len(W_DENSE))]
        j = argbest(objective)
        chosen[g.role] = {"w_hyde": W_DENSE[j], "objective": objective[j], "grid": [{"w_hyde": w, "objective": o} for w, o in zip(W_DENSE, objective)]}
        search_log({"stage": "hyde_scale", "method": "dense_qwen8b_hyde", "group": g.role, "grid": chosen[g.role]["grid"], "chosen": W_DENSE[j]})
        log(f"dense_qwen8b_hyde {g.role}: w_hyde {W_DENSE[j]} (inner objective {objective[j]:.4f}; query only {objective[0]:.4f})")
    (SELECTION_DIR / "dense_qwen8b_hyde.json").write_text(json.dumps(chosen, indent=1), encoding="utf-8")

    def producer(matrix: np.ndarray) -> None:
        for g in ctx.groups:
            w = chosen[g.role]["w_hyde"]
            for s in range(0, len(g.rows), BLOCK):
                r = g.rows[s:s + BLOCK]
                matrix[r] = fusion.linear_fusion([fusion.robust_z(np.asarray(dq[r])), fusion.robust_z(dense_block(h[r]))], pair(w), normalize="none")

    usage = list(load_generations().values())
    _write_scores("dense_qwen8b_hyde", producer, {
        "family": "EMB", "label_free": True,
        "description": "robust-z fusion of the dense_qwen8b query score and the dense score of a LLaDA2.1-mini hypothetical-statute text (Spanish prompt, "
                       "Hechos only) embedded with Qwen3-Embedding-8B; HyDE weight chosen per group on its training rows",
        "training_data": "none (generation and embedding are label-free); the fusion weight is selected on the group's training rows only",
        "configuration": {role: {"w_hyde": chosen[role]["w_hyde"]} for role in chosen},
        "generation": {"model": LLM_MODEL, "endpoint": INSTRUCT_BASE, "params": HYDE_PARAMS, "prompt_sha256": sha256_text(HYDE_PROMPT),
                       "generations": len(usage), "completion_tokens": sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in usage),
                       "texts": str(HYDE_GEN)},
        "depends_on": ["dense_qwen8b"],
        "selection_evidence": {"pilot": str(HYDE_DIR / "pilot.json"), "selection_file": str(SELECTION_DIR / "dense_qwen8b_hyde.json"),
                               "search_log": str(SEARCH_LOG), "note": LLM_NOTE},
        "device": "cpu; generation and embeddings from the MIT endpoints"}, force)
    log(f"HyDE scale-up done in {time.perf_counter() - t0:.0f}s")
    return chosen


# --------------------------------------------------------------------------- summary

REFERENCE_METHODS = ("L_star", "H_bge", "K_lex", "dense_bge_maxp")
OWN_METHODS = ("dense_qwen8b", "H_qwen8b", "knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b", "K_qwen8b", "adapt_qwen8b",
               "dense_qwen8b_hyde")


def stage_summary(extra: Sequence[str] = ()) -> dict[str, Any]:
    """Dev out-of-fold P_spec and P nDCG@10 (and Recall@20) of every delivered method and the references, with paired
    component-bootstrap contrasts on P_spec against L_star and K_lex (dev only, selection-biased; the lockbox is never touched)."""
    dev = common.dev_rows()
    ids = [m for m in REFERENCE_METHODS + OWN_METHODS + tuple(extra) if common.score_path(m).exists()]
    rows: dict[str, Any] = {}
    evals: dict[str, Any] = {}
    for m in ids:
        scores = np.asarray(common.open_scores(m)[dev])
        evals[m] = {pop: common.evaluate(scores, dev, pop) for pop in ("P_spec", "P")}
        rows[m] = {pop: {"ndcg_at_10": evals[m][pop]["macro"]["ndcg_at_10"], "recall_at_20": evals[m][pop]["macro"]["recall_at_20"],
                         "n_queries": evals[m][pop]["n_queries"]} for pop in evals[m]}
    for m in ids:
        for ref in ("L_star", "K_lex"):
            if ref in evals and m != ref:
                c = common.contrast(evals[m]["P_spec"], evals[ref]["P_spec"])
                rows[m][f"P_spec_ndcg10_minus_{ref}"] = {k: c[k] for k in ("effect", "ci95", "ci90", "p_sign_flip", "verdict", "n")}
    supervised = RUN / "results_supervised.json"
    if supervised.exists():
        fused = json.loads(supervised.read_text(encoding="utf-8"))["fused_with_L_star"]
        rows["fused_with_L_star"] = {name: {pop: fused[name][pop]["ndcg_at_10"] for pop in ("P_spec", "P")} for name in fused}
    out = {"generated_utc": utc(), "dev_only": True, "methods": rows}
    (RUN / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for m in ids:
        r = rows[m]
        log(f"{m:20s} dev P_spec nDCG@10 {r['P_spec']['ndcg_at_10']:.4f} (R@20 {r['P_spec']['recall_at_20']:.3f})  "
            f"P nDCG@10 {r['P']['ndcg_at_10']:.4f}")
    for name, r in rows.get("fused_with_L_star", {}).items():
        log(f"fused with L_star: {name:20s} dev P_spec {r['P_spec']:.4f}  P {r['P']:.4f}")
    return out


# ===== CLI =====


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("encode", help="embed queries/articles through the endpoint (resumable)")
    p.add_argument("--what", choices=["queries", "articles", "all"], default="all")
    p.add_argument("--limit", type=int, default=None, help="smoke test: first N units of each group")
    sub.add_parser("assemble", help="write the .npy vector files and receipt.json from the store")
    sub.add_parser("lchan", help="precompute the cached L* channels for the dev P_spec rows (label-free)")
    for name, text in (("dense", "variant selection and the dense_qwen8b score file"),
                       ("h", "H_qwen8b: robust-z fusion of L_star and dense_qwen8b")):
        q = sub.add_parser(name, help=text)
        q.add_argument("--force", action="store_true")
    for name, text in (("inner", "run the inner-CV tasks of the supervised methods (cached)"),
                       ("supervised", "knn_qwen8b, knn_fused_qwen8b, proto_qwen8b and K_qwen8b"),
                       ("supervised-h", "KH_qwen8b: Qwen supervised sources on top of H_qwen8b (needs the supervised stage's grids)")):
        q = sub.add_parser(name, help=text)
        q.add_argument("--workers", type=int, default=4)
        q.add_argument("--force", action="store_true")
    sub.add_parser("parity", help="check the kNN plumbing against the K_lex file")
    q = sub.add_parser("klex-ext", help="K_lex_ext: the no-embedding K_lex pipeline with a non-binding grid")
    q.add_argument("--force", action="store_true")
    for name, text in (("hyde-pilot", "optional HyDE pilot on 300 dev fold-0 rows (INSTRUCT endpoint; only after steps 1-3)"),
                       ("hyde-scale", "optional HyDE for all rows, only if the pilot clearly helped")):
        q = sub.add_parser(name, help=text)
        q.add_argument("--force", action="store_true")
    q = sub.add_parser("summary", help="dev metrics of every delivered method and the references")
    q.add_argument("--extra", nargs="*", default=[])
    q = sub.add_parser("adapt", help="optional low-rank query adapter (GPU; pilot on dev fold 0, kept only if it beats frozen Qwen)")
    q.add_argument("--device", default="cuda")
    q.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "encode":
        run_encode(args.what, args.limit)
    elif args.command == "assemble":
        assemble()
    elif args.command == "lchan":
        ls = LStarRows()
        for sel in {json.dumps(v, sort_keys=True): v for v in lstar_selections().values()}.values():
            if sel["kind"] == "fuse":
                ls.channel(f"{sel['base']}|{sel['k1']}|{sel['b']}")
                ls.channel("char")
            elif sel["kind"] == "char":
                ls.channel("char")
            else:
                ls.channel(f"{sel['kind']}|{sel['k1']}|{sel['b']}")
    elif args.command == "dense":
        stage_dense(args.force)
    elif args.command == "h":
        stage_h(args.force)
    elif args.command == "inner":
        run_inner(args.workers, args.force)
    elif args.command == "supervised":
        stage_supervised(args.force, args.workers)
    elif args.command == "supervised-h":
        stage_supervised_h(args.force, args.workers)
    elif args.command == "parity":
        print(json.dumps(check_klex_parity()))
    elif args.command == "klex-ext":
        stage_klex_ext(args.force)
    elif args.command == "hyde-pilot":
        stage_hyde_pilot(args.force)
    elif args.command == "hyde-scale":
        stage_hyde_scale(args.force)
    elif args.command == "adapt":
        stage_adapt(args.force, args.device)
    elif args.command == "summary":
        stage_summary(args.extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
