"""BAAI/bge-m3 dense encoding on the local GPU (DEVIATIONS.md D3).

Standalone: needs only numpy, torch and transformers, so it runs under the
source home's CUDA environment (``.venv/gpu``) with ``PYTHONPATH`` unset.

Recipe:
- CLS embedding, L2-normalized, fp32, local files only.
- Passages: ``instrument_article_body`` windows of at most ``max_tokens``
  including special tokens. Later windows repeat the instrument title line,
  with ``overlap`` tokens of overlap and at most ``cap`` windows per article.
- Queries: the release's masked facts, truncated at 512 tokens.

Outputs in a new directory:
- ``chunk_vectors.npy`` and ``units.jsonl`` (candidate_id, chunk_index per row);
- ``query_vectors.npy`` in release order;
- ``receipt.json`` with input, model and code SHA-256s, parameters and timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = Path(r"C:\uol_legacy_home\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181")
MODEL_FILES = ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_article(row: dict, tokenizer, max_tokens: int, overlap: int, cap: int) -> list[str]:
    text = row["instrument_article_body"]
    budget = max_tokens - 2  # <s> ... </s>
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return [text]
    title, _, body = text.partition("\n")
    header = f"{title}\nArtículo {row.get('article_label', '')} (continuación)\n"
    header_len = len(tokenizer(header, add_special_tokens=False)["input_ids"])
    first = tokenizer.decode(ids[:budget])
    body_ids = tokenizer(body, add_special_tokens=False)["input_ids"]
    first_body = len(tokenizer(first.partition("\n")[2], add_special_tokens=False)["input_ids"]) if "\n" in first else 0
    window = max(64, budget - header_len)
    chunks, start = [first], max(0, first_body - overlap)
    while start < len(body_ids) and len(chunks) < cap:
        piece = body_ids[start:start + window]
        chunks.append(header + tokenizer.decode(piece))
        if start + window >= len(body_ids):
            break
        start += window - overlap
    return chunks


@torch.inference_mode()
def encode(texts: list[str], tokenizer, model, device: str, max_tokens: int, token_budget: int) -> np.ndarray:
    """Length-sorted dynamic batches bounded by ``token_budget`` padded tokens; output in input order."""
    lengths = [min(max_tokens, len(tokenizer(t, add_special_tokens=True, truncation=True, max_length=max_tokens)["input_ids"])) for t in texts]
    order = np.argsort(lengths, kind="stable")[::-1]
    out = np.zeros((len(texts), model.config.hidden_size), dtype=np.float32)
    i = 0
    while i < len(order):
        longest = lengths[order[i]]
        size = max(1, token_budget // max(longest, 1))
        batch = order[i:i + size]
        enc = tokenizer([texts[j] for j in batch], padding=True, truncation=True, max_length=max_tokens, return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0]
        out[batch] = torch.nn.functional.normalize(cls.float(), dim=-1).cpu().numpy()
        i += size
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--cap", type=int, default=32)
    parser.add_argument("--token-budget", type=int, default=16384)
    parser.add_argument("--limit", type=int, default=0, help="smoke test: first N articles and queries only")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"{args.out} exists")
    started = time.time()
    hashes = {name: sha256(MODEL_DIR / name) for name in MODEL_FILES}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).to(device).eval()
    articles = read_jsonl(args.articles)
    queries = read_jsonl(args.release / "queries.jsonl")
    if args.limit:
        articles, queries = articles[:args.limit], queries[:args.limit]
    units, texts = [], []
    for row in articles:
        for index, chunk in enumerate(chunk_article(row, tokenizer, args.max_tokens, args.overlap, args.cap)):
            units.append({"candidate_id": row["candidate_id"], "chunk_index": index})
            texts.append(chunk)
    t0 = time.time()
    chunk_vectors = encode(texts, tokenizer, model, device, args.max_tokens, args.token_budget)
    t1 = time.time()
    query_vectors = encode([q["text"] for q in queries], tokenizer, model, device, 512, args.token_budget)
    t2 = time.time()
    args.out.mkdir(parents=True)
    np.save(args.out / "chunk_vectors.npy", chunk_vectors)
    np.save(args.out / "query_vectors.npy", query_vectors)
    with (args.out / "units.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for unit in units:
            f.write(json.dumps(unit, sort_keys=True) + "\n")
    receipt = {
        "encoder": "BAAI/bge-m3", "snapshot": MODEL_DIR.name, "model_files": hashes, "pooling": "CLS + L2", "dtype": "float32",
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "params": {"max_tokens": args.max_tokens, "overlap": args.overlap, "cap": args.cap, "query_max_tokens": 512,
                   "token_budget": args.token_budget, "limit": args.limit},
        "inputs": {"articles": {"path": str(args.articles), "sha256": sha256(args.articles)},
                   "queries": {"path": str(args.release / "queries.jsonl"), "sha256": sha256(args.release / "queries.jsonl")}},
        "counts": {"articles": len(articles), "chunks": len(units), "queries": len(queries),
                   "multi_chunk_articles": sum(1 for u in units if u["chunk_index"] == 1)},
        "seconds": {"chunks": round(t1 - t0, 1), "queries": round(t2 - t1, 1), "total": round(time.time() - started, 1)},
        "code_sha256": sha256(Path(__file__)), "torch": torch.__version__,
        "outputs": {name: sha256(args.out / name) for name in ("chunk_vectors.npy", "query_vectors.npy", "units.jsonl")},
    }
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("counts", "seconds")}))


if __name__ == "__main__":
    main()
