"""Examiner checks B8 and B9 (16 Oct 2026, dev only): a same-family Qwen3-Embedding scale curve (0.6B, 4B, 8B) and
a commercial embedding (OpenAI text-embedding-3-large), built on exactly the units of the 8B vectors.

Modules: ``common`` (paths, models, units, vector store), ``local`` (CPU encoder, chunked runs, throughput bench),
``openai_te3`` (budgeted API client), ``evaluate`` (dense alone, unseen-band recall, NO-EMB refits), ``driver``
(heavy-lock chunk loop). Lockbox rows are never encoded, sent, scored or evaluated.
"""
