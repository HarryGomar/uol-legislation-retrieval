"""Offline E5 encoding with exact file, text, order and truncation receipts."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

from uol.retrieval_benchmarks.common import atomic_write_json, sha256_file


class OfflineEncoder:
    def __init__(self, model_binding: dict, *, device: str, check: Callable[[], None]):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.check = check
        self.device = device
        self.binding = model_binding
        for name, binding in model_binding["files"].items():
            if sha256_file(Path(binding["path"])) != binding["sha256"]:
                raise ValueError(f"installed model bytes changed: {name}")
        self.check()
        self.tokenizer = AutoTokenizer.from_pretrained(model_binding["path"], local_files_only=True)
        self.model = AutoModel.from_pretrained(model_binding["path"], local_files_only=True, use_safetensors=True,
                                               attn_implementation="eager").to(device).eval()
        self.dimension = int(self.model.config.hidden_size)
        self.check()

    def encode(self, texts: Sequence[str], *, prefix: str, batch_size: int = 8, max_tokens: int = 512) -> tuple[np.ndarray, dict]:
        import torch
        if prefix not in {"query: ", "passage: "}:
            raise ValueError("undeclared model prefix")
        started = time.perf_counter()
        output = np.empty((len(texts), self.dimension), dtype=np.float32)
        lengths, truncated = [], []
        with torch.inference_mode():
            for begin in range(0, len(texts), batch_size):
                self.check()
                strings = [prefix + t for t in texts[begin:begin + batch_size]]
                untruncated = self.tokenizer(strings, truncation=False, return_length=True, verbose=False)["length"]
                lengths.extend(map(int, untruncated))
                truncated.extend(begin + i for i, length in enumerate(untruncated) if length > max_tokens)
                tokens = self.tokenizer(strings, truncation=True, max_length=max_tokens, padding=True, return_tensors="pt").to(self.device)
                hidden = self.model(**tokens).last_hidden_state.float()
                mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                vectors = torch.nn.functional.normalize(pooled, dim=1)
                output[begin:begin + len(strings)] = vectors.cpu().numpy()
                self.check()
                if (begin // batch_size) % 64 == 0:
                    print(f"ENCODING {prefix.strip()} {begin + len(strings)}/{len(texts)} {time.perf_counter()-started:.1f}s", flush=True)
        if not np.isfinite(output).all() or not np.allclose(np.linalg.norm(output, axis=1), 1, atol=1e-5):
            raise ValueError("invalid offline encoder output")
        return output, {"wall_seconds": time.perf_counter() - started, "texts": len(texts),
                        "tokens_before_truncation": sum(lengths), "token_lengths": lengths,
                        "truncated_positions": truncated, "prefix": prefix, "max_tokens": max_tokens,
                        "batch_size": batch_size, "dtype": "float32", "pooling": "attention-mask mean followed by L2",
                        "device": self.device, "downloads": 0, "scientific_endpoint_calls": 0}

    def cache(self, path: Path, ids: Sequence[str], texts: Sequence[str], *, prefix: str, reuse_from: Path | None = None) -> tuple[np.ndarray, dict]:
        signature = {
            "model_files": {name: row["sha256"] for name, row in self.binding["files"].items()},
            "revision": self.binding["declared_revision"], "ids": list(ids),
            "text_sha256": [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts],
            "prefix": prefix, "max_tokens": 512, "pooling": "attention-mask mean followed by L2",
            "dtype": "float32", "attention": "eager", "batch_size": 8,
        }
        receipt_path = path.with_suffix(".json")
        if path.exists() or receipt_path.exists():
            if not path.exists() or not receipt_path.exists():
                raise ValueError("partial encoder cache preserved; choose a new attempt")
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt["signature"] != signature or sha256_file(path) != receipt["array_sha256"]:
                raise ValueError("encoder cache input/model/order mismatch")
            values = np.load(path, mmap_mode="r")
            if values.shape != (len(ids), self.dimension) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("cache shape/precision/finite gate failed")
            return values, {**receipt, "reused": True}
        if reuse_from is None:
            values, cost = self.encode(texts, prefix=prefix)
            reuse = None
        else:
            started=time.perf_counter()
            old=json.loads(reuse_from.with_suffix('.json').read_text(encoding='utf-8'))
            old_signature=old['signature']
            if ({k:v for k,v in old_signature.items() if k!='text_sha256'}
                    !={k:v for k,v in signature.items() if k!='text_sha256'}
                    or sha256_file(reuse_from)!=old['array_sha256']):
                raise ValueError('row reuse requires the exact model, ordered IDs, encoder recipe and saved vector bytes')
            old_values=np.load(reuse_from,mmap_mode='r')
            if old_values.shape!=(len(ids),self.dimension) or old_values.dtype!=np.float32 or not np.isfinite(old_values).all():
                raise ValueError('row reuse shape/precision/finite gate failed')
            changed=[i for i,(old_text,new_text) in enumerate(zip(old_signature['text_sha256'],signature['text_sha256'])) if old_text!=new_text]
            if len(old_signature['text_sha256'])!=len(texts):raise ValueError('row reuse incomplete text identities')
            if prefix=='query: ' and changed:raise ValueError('changed queries require a distinct full input treatment')
            values=np.array(old_values,copy=True)
            if changed:
                replacements,cost=self.encode([texts[i] for i in changed],prefix=prefix)
                values[changed]=replacements
            else:
                cost={'wall_seconds':0.0,'texts':0,'device':self.device,'downloads':0,'scientific_endpoint_calls':0}
            reuse={'source_array':str(reuse_from),'source_array_sha256':old['array_sha256'],
                   'source_receipt_sha256':sha256_file(reuse_from.with_suffix('.json')),'unchanged_rows':len(ids)-len(changed),
                   'reencoded_positions':changed,'all_ordered_ids_model_recipe_and_unchanged_text_hashes_exact':True,
                   'wall_seconds_including_identity_verification_and_copy':time.perf_counter()-started}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, values)
        receipt = {"signature": signature, "cost": cost, "array_sha256": sha256_file(path), "reused": False}
        if reuse is not None:receipt['exact_row_reuse']=reuse
        atomic_write_json(receipt_path, receipt)
        return values, receipt

    def close(self) -> None:
        import torch
        del self.model
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
