"""Compressed array files of the bundle: byte-shuffled zstd frames of ``.npy`` files (``.npy.zsh``).

Float32 score matrices compress poorly as they are; splitting each float into its four byte planes first (the
blosc "shuffle" idea) gives about 0.7-0.8 for dense scores and below 0.01 for the sparse vote sources. Layout of a
``.npy.zsh`` file: one zstd frame holding ``uint64 header length | npy header bytes | shuffled payload``.
``unpack`` restores the exact original ``.npy`` bytes (its sha256 is checked against the manifest).
"""
from __future__ import annotations

import hashlib
import io
import struct
from pathlib import Path

import numpy as np

SUFFIX = ".zsh"
LEVEL = 3


def _header_and_payload(path: Path) -> tuple[bytes, np.ndarray]:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version == (1, 0):
            np.lib.format.read_array_header_1_0(f)
        else:
            np.lib.format.read_array_header_2_0(f)
        offset = f.tell()
    raw = path.read_bytes()
    return raw[:offset], np.frombuffer(raw, dtype=np.uint8, offset=offset)


def pack(src: Path, dst: Path | None = None, threads: int = 8) -> dict:
    """``src.npy`` -> ``src.npy.zsh``; returns sizes and the sha256 of the original bytes."""
    import zstandard as zs
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.name + SUFFIX)
    header, payload = _header_and_payload(src)
    arr = np.load(src, mmap_mode="r")
    item = arr.dtype.itemsize
    if payload.size % item:
        raise ValueError(f"{src}: payload not a multiple of the item size")
    planes = payload.reshape(-1, item).T.copy() if item > 1 else payload
    sha = hashlib.sha256(header)
    sha.update(payload.tobytes())
    cctx = zs.ZstdCompressor(level=LEVEL, threads=threads)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f, cctx.stream_writer(f, closefd=False) as w:
        w.write(struct.pack("<QQ", len(header), item))
        w.write(header)
        w.write(planes.tobytes())
    tmp.replace(dst)
    return {"bytes_raw": len(header) + payload.size, "bytes_packed": dst.stat().st_size, "sha256": sha.hexdigest()}


def unpack(src: Path, dst: Path | None = None) -> tuple[Path, str]:
    """``x.npy.zsh`` -> ``x.npy`` (exact bytes); returns (path, sha256 of the restored file)."""
    import zstandard as zs
    src = Path(src)
    dst = Path(dst) if dst else src.with_name(src.name[: -len(SUFFIX)])
    dctx = zs.ZstdDecompressor()
    with open(src, "rb") as f, dctx.stream_reader(f) as r:
        data = _read_all(r)
    hlen, item = struct.unpack("<QQ", data[:16])
    header = data[16:16 + hlen]
    planes = np.frombuffer(data, dtype=np.uint8, offset=16 + hlen)
    payload = planes.reshape(item, -1).T.copy() if item > 1 else planes
    sha = hashlib.sha256(header)
    sha.update(payload.tobytes())
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(payload.tobytes())
    tmp.replace(dst)
    return dst, sha.hexdigest()


def _read_all(reader) -> bytes:
    buf = io.BytesIO()
    while True:
        chunk = reader.read(1 << 24)
        if not chunk:
            break
        buf.write(chunk)
    return buf.getvalue()
