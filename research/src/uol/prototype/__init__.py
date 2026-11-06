"""Minimal integrated prototype (D33): Spanish facts in -> top-10 federal articles with provenance, an embedding map, timed.

The research ranker ``LTR_emb_hgb`` (dev P_spec nDCG@10 .4721) needs 51 score sources, some of them GPU models. The
deployment pipeline keeps the CPU-feasible signals, rebuilds the research feature recipe over them and retrains its
ranker the same way on the dev folds (``train``). Modules:

* ``config``   paths, signal list, per-group configurations (read from the research manifests);
* ``signals``  catalogue, fit sets (dev memory, prior, prototypes) and the census memory; one score row per signal;
* ``features`` candidate pool and per-pair features (robust z, log rank, instrument max z; extras);
* ``bundle``   one-off preparation of the serving bundle (census vectors, lexical indexes, map, demo cases);
* ``train``    dev out-of-fold features, per-group HGB with inner CV, dev evaluation against the research pipeline;
* ``pipeline`` one query end to end with per-stage latency (offline demo, live endpoint, no-embedding fallback);
* ``embed``    cached, budgeted calls to the MIT Qwen3-Embedding-8B endpoint (the only network use);
* ``app``      FastAPI server with plain HTML views; ``screenshots`` captures them with headless Edge.

Run ``py -3.12 -m uol.prototype --help``. No lockbox or prospective row is ever read for labels, texts, vectors or scores.
"""
