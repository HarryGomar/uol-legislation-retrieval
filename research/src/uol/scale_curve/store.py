"""Dependency-free pieces (numpy + stdlib) shared with the standalone CUDA-env encoder: the content-hash vector store,
content keys, batch planning and JSONL helpers. Nothing here imports the rest of ``uol``.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BATCH_TOKENS = 8192                                      # padded tokens per forward pass
BATCH_ATTN = 2 ** 23                                     # n * L^2 cap (bounds the attention scratch of padded batches)
MAX_INPUTS = 64


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def content_key(text: str, model_id: str) -> str:
    """sha256(model, NUL, text): the key convention of ``embedding_push.qwen_endpoint.content_key``."""
    return hashlib.sha256(model_id.encode("utf-8") + b"\x00" + text.encode("utf-8", "surrogatepass")).hexdigest()


def keys_digest(keys: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode("ascii")).hexdigest()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:                 # a torn last line after a crash
                continue
    return rows


def plan_batches(lengths: Sequence[int], max_tokens: int = BATCH_TOKENS, max_inputs: int = MAX_INPUTS,
                 attn_cells: int = BATCH_ATTN) -> list[list[int]]:
    """Index batches over items sorted by length: n*L <= max_tokens, n*L^2 <= attn_cells, n <= max_inputs (L = the
    longest item); an item over the caps goes alone."""
    order = sorted(range(len(lengths)), key=lambda i: (int(lengths[i]), i))
    batches: list[list[int]] = []
    cur: list[int] = []
    for i in order:
        L = int(lengths[i])
        n = len(cur) + 1
        if cur and (n * L > max_tokens or n * L * L > attn_cells or n > max_inputs):
            batches.append(cur)
            cur = []
        cur.append(i)
    if cur:
        batches.append(cur)
    return batches


class VectorStore:
    """Append-only ``key -> float32[dim]`` cache (``vectors.f32`` + ``index.jsonl``), as ``qwen_endpoint.VectorStore``.

    Vectors are written and flushed before their index lines; a torn tail is truncated on the next open.
    """

    def __init__(self, root: Path, dim: int) -> None:
        self.root, self.dim = Path(root), int(dim)
        self.root.mkdir(parents=True, exist_ok=True)
        self.vec_path, self.idx_path = self.root / "vectors.f32", self.root / "index.jsonl"
        self.rows: dict[str, int] = {}
        size = self.vec_path.stat().st_size if self.vec_path.exists() else 0
        complete = size // (self.dim * 4)
        entries = read_jsonl(self.idx_path)
        keep: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get("row") == len(keep) and entry["row"] < complete and entry["k"] not in self.rows:
                self.rows[entry["k"]] = entry["row"]
                keep.append(entry)
            else:
                break
        if len(keep) != len(entries) or size != len(keep) * self.dim * 4:
            with open(self.vec_path, "ab"):
                pass
            os.truncate(self.vec_path, len(keep) * self.dim * 4)
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
        if vectors.shape != (len(keys), self.dim) or len(set(keys)) != len(keys):
            raise ValueError("one distinct key per vector row is required")
        if not np.isfinite(vectors).all():
            raise ValueError("non-finite vectors")
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
        absent = [k for k in keys if k not in self.rows]
        if absent:
            raise KeyError(f"{len(absent)} keys are not in the store, e.g. {absent[0][:12]}")
        if not len(keys):
            return np.empty((0, self.dim), dtype=np.float32)
        mm = np.memmap(self.vec_path, dtype=np.float32, mode="r", shape=(self.n, self.dim))
        out = np.asarray(mm[np.fromiter((self.rows[k] for k in keys), dtype=np.int64, count=len(keys))])
        del mm
        return out
