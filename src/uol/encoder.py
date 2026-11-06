"""Qwen3-Embedding-8B case vectors for new cases.

A case is embedded as ``Instruct: <instruction>\\nQuery:<facts>`` and the vector is normalised. Variant ``c`` (a Spanish
instruction) serves every source; variant ``b`` (an English one) serves the case-memory votes of fold 4 only, so a new
case (group 5) needs ``c`` alone. On a graphics card the weights are loaded in 8 bits, as in the thesis's timed run;
on the processor in 32 bits. The stored vectors of the benchmark came from a 16-bit server: a fresh vector has a
median cosine of 0.992 (card) or 0.997 (processor) with the stored one, so a fresh run is close to, not identical
with, the stored lists.
"""
from __future__ import annotations

import numpy as np

MODEL = "Qwen/Qwen3-Embedding-8B"
REVISION = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
MAX_TOKENS = 8192
INSTRUCTIONS = {
    "b": "Given the facts of a Mexican court decision, retrieve the federal statutory articles the court applied",
    "c": "Dada una descripción publicada de hechos, recupera el criterio jurídico publicado que pertenece documentalmente "
         "a la misma ficha.",
}


def query_text(facts: str, variant: str = "c") -> str:
    return f"Instruct: {INSTRUCTIONS[variant]}\nQuery:{facts}"


class Encoder:
    def __init__(self, device: str = "cuda") -> None:
        import torch
        from sentence_transformers import SentenceTransformer
        kwargs = {"dtype": torch.float32}
        if device == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs = {"dtype": torch.bfloat16, "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
                      "device_map": {"": 0}}
        self.model = SentenceTransformer(MODEL, revision=REVISION, device=None if device == "cuda" else device,
                                         model_kwargs=kwargs)
        self.model.max_seq_length = MAX_TOKENS

    def __call__(self, facts: str, variants: tuple[str, ...] = ("c",)) -> dict[str, np.ndarray]:
        out = {}
        for v in variants:
            x = self.model.encode([query_text(facts, v)], prompt="", batch_size=1, convert_to_numpy=True,
                                  normalize_embeddings=True, show_progress_bar=False)[0].astype(np.float32)
            out[v] = x / max(float(np.linalg.norm(x)), 1e-12)
        return out
