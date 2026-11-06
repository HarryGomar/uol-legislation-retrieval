"""The system registry (PLAN_DRAFT §3): ``systems.json`` built from code, loaded and instantiated by id.

Every entry has: ``id``, ``family``, ``blocks``, ``inputs``, ``training_data``, ``grid`` (tuned on the inner folds),
``hardware``, ``adapter`` (+ ``adapter_args``), ``mode`` (``rebuilt`` | ``loaded`` | ``new`` | ``gpu_pending`` |
``llm_later``), ``artefact`` (key, path and the expected hash of a loaded artefact), ``peak_ram_gb`` / ``threads``
(declared for the overnight scheduler), optional ``g1`` (frozen value it must reproduce), ``answer`` (P1-P6, P3+L)
and ``status`` (``ready`` | ``pending_vectors`` | ``block_e_later`` | ``stub``).

    py -3.12 -m uol.final systems            # rebuild systems.json from this file and the frozen manifests
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from uol.final import context

SYSTEMS_JSON = Path(__file__).resolve().parent / "systems.json"

ADAPTERS = {
    "frozen_scores": "uol.final.adapters.loaded:FrozenScores",
    "frozen_pipeline": "uol.final.adapters.loaded:FrozenPipeline",
    "dev_block": "uol.final.adapters.loaded:DevBlock",
    "c_hybrid_cached": "uol.final.adapters.loaded:CHybridCached",
    "dense_vectors": "uol.final.adapters.dense:DenseVectors",
    "sparse_vectors": "uol.final.adapters.gpu:SparseVectors",
    "lstar": "uol.final.adapters.lexical:LStar",
    "bm25": "uol.final.adapters.lexical:BM25",
    "topic": "uol.final.adapters.topic:TopicScorer",
    "fusion": "uol.final.adapters.fusion:Fusion",
    "random": "uol.final.adapters.reference:RandomScorer",
    "popularity": "uol.final.adapters.reference:Popularity",
    "ltr": "uol.final.adapters.ltr:LTRRefit",
    "llm_export": "uol.final.adapters.llm:LLMLater",
    "not_wired": "uol.final.adapters.llm:NotWired",
    "llm_matrix": "uol.final.adapters.llm:LLMMatrix",
    "qwen_adapter": "uol.final.adapters.adapted:QwenAdapter",
    "fold_dense": "uol.final.adapters.dense:FoldDense",
}

FROZEN_RAM = 3
G1 = {"lstar_frozen": 0.18376269351867786, "qwen3_8b": 0.2741685751276007, "ltr_noemb_frozen": 0.4081358955959893,
      "ltr_emb_frozen": 0.4721007254624054, "c_hybrid_cached": 0.5557884848974133,
      "lstar": 0.18376269351867786, "qwen3_8b_vectors": 0.2741685751276007, "ltr_noemb_refit": 0.4081358955959893,
      "ltr_emb_refit": 0.4721007254624054}
G1_ROUNDED = {"lstar_frozen": ".184", "qwen3_8b": ".274", "ltr_noemb_frozen": ".4081", "ltr_emb_frozen": ".4721",
              "c_hybrid_cached": ".5558", "lstar": ".184", "qwen3_8b_vectors": ".274", "ltr_noemb_refit": ".4081",
              "ltr_emb_refit": ".4721"}

TOPIC = ("lsa", "lda_sim", "lda_dm", "doc2vec")
# D88 (23 Oct): corpus-trained word vectors, scored alone and in the L* hybrids only; never a ranker column, so the
# rankers (whose extra columns are TOPIC) are unchanged
WORD_VECTORS = ("word2vec",)
GPU_ENCODERS = {
    # id: (family, job key in adapters.gpu.JOBS, params)
    "e5_base": ("dense_general", "e5_base", "278M"),
    "e5_large": ("dense_general", "e5_large", "560M"),
    "bge_m3_base": ("dense_general", "bge_m3_base", "568M"),
    "alia_mrbert": ("dense_legal", "alia_mrbert", "~150M"),
    "gemma_legal": ("dense_legal", "gemma_legal", "300M"),
    "gemma_base": ("dense_general", "gemma_base", "300M"),        # base of gemma_legal (review 22 Oct §4.8)
    "bge_m3_legal": ("dense_legal", "bge_m3_legal", "568M"),
    "euler_legal": ("dense_legal", "euler_legal", "8B int8"),
    "e5_large_instruct": ("dense_general", "e5_large_instruct", "560M"),
    "gte_multilingual": ("dense_general", "gte_multilingual", "305M"),
    "jina_v3": ("dense_general", "jina_v3", "570M"),
    "arctic_l_v2": ("dense_general", "arctic_l_v2", "568M"),
}
VECTOR_ENCODERS = {
    "qwen3_0p6b": ("qwen3_0p6b", ""), "qwen3_4b": ("qwen3_4b", ""), "qwen3_8b_vectors": ("qwen8b", ""),
    "e5_large_legal_en": ("e5_large_legal", ""),
}
SAVED_ENCODERS = {"e5_small": "dense_e5_maxp", "bge_m3": "dense_bge_maxp", "qwen3_8b": "dense_qwen8b",
                  "ft_e5_q": "ft_e5_q"}
LADDER_ENCODERS = ("e5_small", "bge_m3", "bge_m3_colbert", "qwen3_0p6b", "qwen3_4b", "qwen3_8b", "e5_base", "e5_large", "bge_m3_base",
                   "bge_m3_sparse", "alia_mrbert", "gemma_legal", "gemma_base", "bge_m3_legal", "euler_legal",
                   "e5_large_legal_en",
                   "ft_e5_q", "e5_large_instruct", "gte_multilingual", "jina_v3", "arctic_l_v2", "query2doc", "hyde",
                   "qwen3_8b_mrl256", "qwen3_8b_mrl512", "qwen3_8b_mrl1024", "qwen3_8b_int8", "qwen3_8b_binary")
FUSION_PARTNERS = ("popularity",) + TOPIC + WORD_VECTORS + ("e5_small", "bge_m3", "bge_m3_colbert", "qwen3_0p6b", "qwen3_4b", "qwen3_8b", "e5_base",
                                             "e5_large", "bge_m3_sparse", "alia_mrbert", "gemma_legal", "bge_m3_legal",
                                             "euler_legal", "e5_large_legal_en", "ft_e5_q", "e5_large_instruct",
                                             "gte_multilingual", "jina_v3", "arctic_l_v2", "query2doc", "hyde",
                                             "qwen3_0p6b_ft")              # D89


# an encoder and its past-case sources inside the frozen pair table (RUN_PLAN v2 §3.7 "S0 + X")
ENCODER_SOURCES = {
    "e5_small": ["dense_e5_maxp", "knn_e5", "proto_e5"],
    "bge_m3": ["dense_bge_maxp", "dense_bge_firstp", "dense_bge_long", "H_bge", "knn_bge", "proto_bge", "colbert_bge"],
    "qwen3_8b": ["dense_qwen8b", "H_qwen8b", "KH_qwen8b", "K_qwen8b", "knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b"],
    "ft_e5_q": ["ft_e5_q", "knn_adapt_all", "knn_adapt_spec", "proto_adapt", "knn_casesim_e5", "knn_casesim_e5_all",
                "proto_casesim_e5"],
}


def _manifest_sha(path: Path) -> str | None:
    man = path.parent / "manifest.json"
    if not man.exists():
        return None
    m = json.loads(man.read_text(encoding="utf-8"))
    return m.get("scores_sha256") or m.get("pairs_sha256")


def _entry(sid: str, family: str, blocks: list[str], adapter: str, *, mode: str, args: dict | None = None,
           inputs: str = "", training: str = "none (label-free)", grid: Any = None, hardware: str = "cpu",
           ram: float = 6, threads: int = 4, status: str = "ready", artefact: dict | None = None,
           answer: str | None = None, notes: str = "") -> dict[str, Any]:
    e = {"id": sid, "family": family, "blocks": blocks, "inputs": inputs, "training_data": training, "grid": grid,
         "hardware": hardware, "adapter": adapter, "adapter_args": args or {}, "mode": mode, "artefact": artefact,
         "peak_ram_gb": ram, "threads": threads, "status": status, "notes": notes}
    if sid in G1:
        e["g1"] = {"expected": G1[sid], "rounded": G1_ROUNDED[sid], "population": "DEV", "measure": "ndcg_at_10"}
    if answer:
        e["answer"] = answer
    return e


# the dense column of the frozen encoders that enter S0 + X with their past-case sources (D78 dense-only twins)
DENSE_TWIN_SOURCE = {"e5_small": "dense_e5_maxp", "bge_m3": "dense_bge_maxp", "qwen3_8b": "dense_qwen8b",
                     "ft_e5_q": "ft_e5_q"}


def dense_only_twin(enc: str, s0: dict) -> dict[str, Any]:
    """S0 + X with the encoder's dense column only (D78), as for every other ladder encoder: the three label-free
    columns of the encoder's dense scores. ft_e5_q is the exception: its frozen scores come from a query tower trained
    on the outer folds, so as a label-free extra column it would break the a_p12 nesting rule; it enters as its single
    dense source of the pair table (the same column it has inside S_full), without its past-case sources."""
    if enc == "ft_e5_q":
        return {"args": {**s0, "extra": list(TOPIC), "add_sources": [DENSE_TWIN_SOURCE[enc]]},
                "inputs": "S0 + the ft_e5_q dense source of the pair table only (no past-case sources; nesting rule)"}
    return {"args": {**s0, "extra": list(TOPIC) + [enc], "add_sources": []},
            "inputs": f"S0 + three columns of {enc} (robust z, log rank, instrument-max z) from {DENSE_TWIN_SOURCE[enc]}"}


def sweep_entries() -> list[dict[str, Any]]:
    """LLM sweep (D174, 28 Oct 2026): the Claude treatment with another model, exploratory, DEV-A300 only."""
    out = []
    for sid, what, hw in (("sonnet_rerank_S_full_ext", "Claude Sonnet 5.5", "external API (thesis PC)"),
                          ("haiku_rerank_S_full_ext", "Claude Haiku 5.5", "external API (thesis PC)"),
                          ("gpt6sol_rerank_S_full_ext", "GPT-6 Sol", "LLM (Codex CLI, gpt-6-sol; thesis PC)")):
        out.append(_entry(sid, "llm", ["C10"], "llm_matrix", mode="new", args={"key": sid}, ram=8, status="ready",
                          inputs=f"{what} listwise rerank, the main prompt and windows, 1 sample, fused per fold like P6",
                          hardware=hw, notes="added after the freeze (D174), exploratory, DEV-A300 only"))
    return out


def build_systems() -> list[dict[str, Any]]:
    """Every system of PLAN_DRAFT §3 and the derived Block B-D rows."""
    prof = context.active()
    ep_run = prof.path("ep_run")
    S: list[dict[str, Any]] = []
    S.append(_entry("random", "reference", ["A"], "random", mode="rebuilt", inputs="none", ram=2))
    S.append(_entry("popularity", "reference", ["A", "B"], "popularity", mode="rebuilt", inputs="training labels",
                    training="outer-fold training components (log1p distinct citing components)", ram=2))
    # lexical
    S.append(_entry("bm25_stem", "lexical", ["A"], "bm25", mode="rebuilt", args={"kind": "bm25_stem"},
                    inputs="masked facts; article body", grid="k1 {.9,1.2,1.5} x b {.4,.75}", ram=8))
    S.append(_entry("lstar", "lexical", ["A", "G1"], "lstar", mode="rebuilt", args={"text": "V0"},
                    inputs="masked facts; article body", grid="frozen per-fold L* selection", ram=10,
                    notes="recompute of the frozen L_star file (G1 recompute)"))
    S.append(_entry("lstar_headings", "lexical", ["A", "B"], "lstar", mode="rebuilt", args={"text": "V2"},
                    inputs="masked facts; heading path + body (D58)", grid="frozen per-fold L* selection", ram=10,
                    answer="P1"))
    for sid, src in (("lstar_frozen", "L_star"), ("docexp_bm25", "docexp_bm25")):
        path = ep_run / "scores" / src / "scores.npy"
        S.append(_entry(sid, "lexical", ["A", "G1"] if sid == "lstar_frozen" else ["A"], "frozen_scores",
                        mode="loaded", args={"source": src}, inputs=f"frozen score file {src}", ram=FROZEN_RAM,
                        artefact={"key": "ep_run", "path": f"scores/{src}/scores.npy", "manifest_sha256": _manifest_sha(path)}))
    S.append(_entry("lstar_headings_d58", "lexical", ["A"], "dev_block", mode="loaded",
                    args={"key": "d58_sources", "path": "a/L_star.npy"}, inputs="D58 rebuilt heading-path L* (dev rows)",
                    ram=FROZEN_RAM, artefact={"key": "d58_sources", "path": "a/L_star.npy"},
                    notes="reference for lstar_headings (D58 .1958)"))
    # topic / latent (new)
    from uol.final.adapters.topic import GRIDS as grids
    topic_notes = {
        "lda_sim": "CHANGED AFTER THE FREEZE (D78, 22 Oct 2026; CHANGES_AFTER_FREEZE.md): topic mixtures compared after "
                   "removing the Dirichlet prior mass (expected topic counts). The normalised posterior kept 50 prior "
                   "pseudo-counts per mixture and biased lda_sim to short articles (.010; long articles never retrieved)",
        "lsa": "every fold chose 1,024, the top of the grid; reported as a limit (D78)",
    }
    for k in TOPIC:
        S.append(_entry(k, "topic", ["A", "B", "C"], "topic", mode="new", args={"kind": k, "grid": grids[k]},
                        inputs="masked facts; article body", grid=grids[k], ram=12, threads=8,
                        notes="label-free model fitted on the catalogue; size chosen on inner folds"
                              + (f". {topic_notes[k]}" if k in topic_notes else "")))
    for k in WORD_VECTORS:
        S.append(_entry(k, "topic", ["A", "B"], "topic", mode="new", args={"kind": k, "grid": grids[k]},
                        inputs="masked facts; article body", grid=grids[k], ram=12, threads=8,
                        notes="ADDED AFTER THE FREEZE (D88, 23 Oct 2026; CHANGES_AFTER_FREEZE.md): skip-gram word vectors "
                              "fitted without labels on the catalogue and the fold's training facts; IDF-weighted mean "
                              "vectors, cosine; vector size and window chosen on inner folds; no ranker column"))
    # dense: saved score files
    for sid, src in SAVED_ENCODERS.items():
        path = ep_run / "scores" / src / "scores.npy"
        fam = "dense_adapted" if sid == "ft_e5_q" else "dense_general"
        S.append(_entry(sid, fam, ["A", "B", "C"], "frozen_scores", mode="loaded", args={"source": src},
                        inputs=f"frozen score file {src}", ram=FROZEN_RAM,
                        training="citations (query tower)" if sid == "ft_e5_q" else "none (label-free)",
                        artefact={"key": "ep_run", "path": f"scores/{src}/scores.npy", "manifest_sha256": _manifest_sha(path)}))
    for sid, (key, sub) in VECTOR_ENCODERS.items():
        S.append(_entry(sid, "dense_legal" if "legal" in sid else "dense_general",
                        ["A", "B", "C"] + (["G1"] if sid == "qwen3_8b_vectors" else []), "dense_vectors",
                        mode="loaded", args={"key": key, "sub": sub, **({"variants": ["c"]} if sid == "qwen3_8b_vectors" else {})},
                        inputs="cached query and article vectors (MaxP)", grid="query variant (inner folds)",
                        ram=8, artefact={"key": key, "path": sub or "."}))
    # C9 (D93, ANALYSIS_PLAN_C §9): each query variant fixed, and other unit aggregations of the 8B article windows
    for sid in ("qwen3_0p6b", "qwen3_4b", "qwen3_8b_vectors"):
        key, sub = VECTOR_ENCODERS[sid]
        for v in ("a", "b", "c"):
            S.append(_entry(f"{sid}@q{v}", "dense_general", ["C9"], "dense_vectors", mode="loaded",
                            args={"key": key, "sub": sub, "variants": [v]}, ram=8,
                            inputs=f"cached vectors, query variant {v} in every fold (C9, D93)",
                            notes="added after the freeze (D93): representation check, descriptive"))
    for agg in ("first", "mean", "top2"):
        S.append(_entry(f"qwen3_8b_vectors@agg_{agg}", "dense_general", ["C9"], "dense_vectors", mode="loaded",
                        args={"key": "qwen8b", "sub": "", "variants": ["c"], "unit_agg": agg}, ram=8,
                        inputs=f"cached Qwen3-8B vectors, article score = {agg} of its windows (C9, D93)",
                        notes="added after the freeze (D93): representation check, descriptive"))
    S.append(_entry("qwen3_8b_headings", "dense_general", ["A"], "dev_block", mode="loaded",
                    args={"key": "d58_sources", "path": "b/dense_qwen8b.npy"}, ram=FROZEN_RAM,
                    inputs="Qwen3-8B variant c queries vs heading-path article vectors (D58)",
                    artefact={"key": "d58_sources", "path": "b/dense_qwen8b.npy"}))
    # GPU encoders (vectors filled later by the 5080 jobs)
    for sid, (fam, job, params) in GPU_ENCODERS.items():
        S.append(_entry(sid, fam, ["A", "B", "C"], "dense_vectors", mode="gpu_pending", args={"key": sid, "job": job},
                        inputs="article units + dev query strings (scale-curve units), GPU job", hardware="gpu (RTX 5080)",
                        ram=8, status="pending_vectors", notes=f"{params}; vectors in <new_vectors>/{sid} after the job"))
    # sensitivity (review 22 Oct §3(c)): bge_m3_legal was trained with MatryoshkaLoss at 768 dimensions only; both it
    # and its base are also scored on their first 768 dimensions (renormalised), CPU, from the same vectors
    for base_id in ("bge_m3_legal", "bge_m3_base"):
        fam, job, _ = GPU_ENCODERS[base_id]
        S.append(_entry(f"{base_id}_768", fam, ["A"], "dense_vectors", mode="new",
                        args={"key": base_id, "job": job, "mrl_dim": 768},
                        inputs=f"{base_id} vectors, first 768 dimensions renormalised", ram=8, status="pending_vectors",
                        notes="sensitivity: MatryoshkaLoss(768) of bge_m3_legal; the pair stays comparable at 768"))
    S.append(_entry("bge_m3_sparse", "learned_sparse", ["A", "B", "C"], "sparse_vectors", mode="gpu_pending",
                    args={"key": "bge_m3_sparse", "job": "bge_m3_sparse"}, hardware="gpu (RTX 5080)", ram=10,
                    status="pending_vectors", inputs="bge-m3 lexical (sparse) weights of units and queries"))
    # D79 (22 Oct 2026, added after the freeze at the author's request): late interaction scored over the whole catalogue by
    # the GPU job colbert_full (the frozen colbert_bge recipe; fidelity gate on the frozen pairs)
    S.append(_entry("bge_m3_colbert", "dense_general", ["A", "B", "C"], "dev_block", mode="gpu_pending",
                    args={"key": "new_vectors", "path": "bge_m3_colbert/scores_dev.npy", "job": "colbert_full"},
                    hardware="gpu (RTX 5080)", ram=8, status="pending_vectors",
                    inputs="bge-m3 multi-vector (ColBERT) late interaction, every article, uniform MaxSim, MaxP over chunks",
                    notes="added after the freeze (D79); recipe of the frozen colbert_bge source, full catalogue"))
    # past cases: every frozen full source, alone (Block A rows)
    meta = json.loads((ep_run / "fusion/features/meta.json").read_text(encoding="utf-8"))
    seen = {e["adapter_args"].get("source") for e in S}
    for s in meta["sources"]:
        if s["kind"] != "full" or s["id"] in seen:
            continue
        path = ep_run / "scores" / s["id"] / "scores.npy"
        fam = "past_cases" if s["id"].startswith(("knn", "proto", "K_", "KH_", "LP", "prior")) else \
            ("lexical" if s["family"] == "NO-EMB" else "dense_general")
        S.append(_entry(f"src_{s['id']}", fam, ["A"], "frozen_scores", mode="loaded", args={"source": s["id"]},
                        inputs=f"frozen source {s['id']} ({s['family']})", ram=FROZEN_RAM,
                        training="fold-local training labels" if fam == "past_cases" else "none (label-free)",
                        artefact={"key": "ep_run", "path": f"scores/{s['id']}/scores.npy",
                                  "manifest_sha256": _manifest_sha(path)},
                        notes="frozen embedding_push source; recompute from code is not yet wired (loaded)"))
    # rerankers and LLM (RUN_PLAN v2 §3.8: all on the run computer; Codex through round3.luna)
    for sid, what in (("rerank_bge", "bge-reranker-v2-m3, zero-shot"), ("rerank_qwen3_4b", "Qwen3-Reranker-4B, zero-shot")):
        for suffix, how in (("", "fused with S_full_ext (fusion_route)"), ("_alone", "top 30 reordered by the reranker alone")):
            S.append(_entry(sid + suffix, "reranker", ["E", "E9"], "llm_matrix", mode="new", args={"key": sid + suffix},
                            inputs=f"{what}, S_full_ext top 30, {how}", hardware="GPU (RTX 5080)", ram=8))
    S.append(_entry("ce_trained_alone", "reranker", ["E", "E9"], "llm_matrix", mode="new", args={"key": "ce_trained_alone"},
                    inputs="ce_trained, S_full_ext top 30 reordered by the trained cross-encoder alone",
                    hardware="GPU (RTX 5080)", ram=8, training="outer-fold training labels"))
    llm = {"llm_alone": ("closed book, 20 (law, article) names", None, "E9"),
           "llm_rerank_S0": ("listwise reranking of the S0 top 30 (3 samples x 2 orders)", "P3+L", "E9"),
           "llm_rerank_S_full_ext": ("listwise reranking of the S_full_ext top 30 (3 samples x 2 orders)", "P6", "E9"),
           "ce_trained": ("Qwen3-Reranker-0.6B fine-tuned per fold (round3.r2_cross recipe), S_full_ext top 30", "P5+CE", "E9"),
           "agent_words": ("E6 lockstep agent, word-search tool", None, "E12"),
           "agent_embedding": ("E6 lockstep agent, embedding-search tool (local Qwen3-8B int8 queries)", None, "E12")}
    for sid, (what, ans, exp) in llm.items():
        S.append(_entry(sid, "llm" if sid != "ce_trained" else "reranker", ["E", exp], "llm_matrix", mode="new",
                        args={"key": sid}, inputs=what, ram=8, status="ready",
                        hardware="GPU (RTX 5080)" if sid == "ce_trained" else "LLM (Codex CLI, gpt-5.6-luna)", answer=ans))
    # C5 (D93): the blank-text control of P6 (law and article number only), one sample
    S.append(_entry("llm_rerank_S_full_ext_blank", "llm", ["C5"], "llm_matrix", mode="new",
                    args={"key": "llm_rerank_S_full_ext_blank"}, ram=8, status="ready",
                    inputs="listwise reranking of the S_full_ext top 30 without article text (1 sample x 2 orders)",
                    hardware="LLM (Codex CLI, gpt-5.6-luna)",
                    notes="added after the freeze (D93): does the LLM stage read the text or know the article number"))
    # C10 (D101): external second stages over the S_full_ext top 30, on DEV-A300 (API calls on the thesis PC)
    for sid, what in (("claude_rerank_S_full_ext", "Claude Opus 5.5 listwise rerank, the main prompt and windows, 1 sample, "
                                                "fused per fold like P6"),
                      ("jev_S_full_ext", "Jev decision model, pointwise yes-probability, order of the top 30"),
                      ("jev_rrf_S_full_ext", "Jev order fused with S_full_ext by reciprocal rank (k 60)")):
        S.append(_entry(sid, "llm", ["C10"], "llm_matrix", mode="new", args={"key": sid}, ram=8, status="ready",
                        inputs=what, hardware="external API (thesis PC)",
                        notes="added after the freeze (D101): an F6 second-stage alternative, DEV-A300 and NEW"))
    S.extend(sweep_entries())
    S.append(_entry("llm_issues", "llm", ["E"], "not_wired", mode="loaded", status="stub",
                    args={"reason": "the issue decompositions (L2, every dev row) are an input of query2doc, not a ranking"},
                    inputs="saved L2 issue decompositions"))
    # adapted and expanded encoders (RUN_PLAN v2 §3.5, §3.8)
    S.append(_entry("qwen8b_adapter", "dense_adapted", ["A", "E2"], "qwen_adapter", mode="rebuilt", args={"table": "dev"},
                    ram=18, threads=8,
                    inputs="frozen Qwen3-8B vectors; E2 low-rank adapter per outer fold", training="outer-fold training labels"))
    S.append(_entry("bge_m3_ft", "dense_adapted", ["A", "E2"], "fold_dense", mode="new", args={"key": "bge_m3_ft"},
                    hardware="gpu (RTX 5080), one model per fold", ram=8, status="pending_vectors",
                    training="outer-fold training labels (hard negatives from frozen bge-m3 top 50)"))
    S.append(_entry("qwen3_0p6b_ft", "dense_adapted", ["A", "E2"], "fold_dense", mode="new", args={"key": "qwen3_0p6b_ft"},
                    hardware="gpu (RTX 5080), one model per fold", ram=8, status="pending_vectors",
                    training="outer-fold training labels (hard negatives from the frozen qwen3_0p6b top 50)",
                    notes="ADDED AFTER THE FREEZE (D89, 23 Oct 2026; CHANGES_AFTER_FREEZE.md): Qwen3-Embedding-0.6B fully "
                          "fine-tuned per outer fold with the bge_m3_ft recipe; queries in the instruction variant the "
                          "base qwen3_0p6b chose on that fold's inner folds; catalogue re-encoded in the base's title-led "
                          "windows; no ranker column"))
    for sid, what in (("query2doc", "facts + saved L2 issues, Qwen3-8B int8, variant c"),
                      ("hyde", "hypothetical article written by the LLM, Qwen3-8B int8 document side"),
                      ("qwen3_8b_int8q", "Qwen3-8B int8 query vectors (fidelity check against the cached ones)")):
        key = "qwen_int8_dev" if sid == "qwen3_8b_int8q" else sid
        S.append(_entry(sid, "query_expansion" if sid != "qwen3_8b_int8q" else "dense_general", ["A", "E8"],
                        "dense_vectors", mode="new", args={"key": key}, inputs=what, ram=8, status="pending_vectors"))
    S.append(_entry("qwen3_8b_int8_base", "dense_general", ["A", "E2"], "dense_vectors", mode="new",
                    args={"key": "qwen3_8b_int8path"}, inputs="Qwen3-Embedding-8B re-encoded in int8 through the Euler code path "
                    "(articles and dev queries): the same-path base of euler_legal", ram=10, status="pending_vectors",
                    hardware="gpu (RTX 5080)"))
    S.append(_entry("e5_small_re", "dense_general", ["A"], "dense_vectors", mode="new", args={"key": "e5_small_re"},
                    inputs="multilingual-e5-small re-encoded on the 5080 (both sides; DEV-N uses it with re-encoded queries)",
                    ram=6, status="pending_vectors", hardware="gpu (RTX 5080)"))
    # E3: compressed Qwen3 vectors (Matryoshka truncation and int8 / binary storage), from the saved vectors
    for base_id, (key, _) in (("qwen3_0p6b", VECTOR_ENCODERS["qwen3_0p6b"]), ("qwen3_4b", VECTOR_ENCODERS["qwen3_4b"]),
                              ("qwen3_8b", VECTOR_ENCODERS["qwen3_8b_vectors"])):
        full = {"qwen3_0p6b": 1024, "qwen3_4b": 2560, "qwen3_8b": 4096}[base_id]
        for dim in (256, 512, 1024, None):
            if dim and dim >= full:
                continue
            for quant in (None, "int8", "binary"):
                if dim is None and quant is None:
                    continue
                sid = f"{base_id}" + (f"_mrl{dim}" if dim else "") + (f"_{quant}" if quant else "")
                S.append(_entry(sid, "dense_compressed", ["A", "E3"], "dense_vectors", mode="new",
                                args={"key": key, "mrl_dim": dim, "quant": quant,
                                      **({"variants": ["c"]} if base_id == "qwen3_8b" else {})},
                                inputs=f"{base_id} saved vectors, {dim or full} dims, {quant or 'fp32'}", ram=8,
                                notes=f"index size {(dim or full) * (1 if quant == 'int8' else 0.125 if quant == 'binary' else 4) / 1024:.3f} KB per vector"))
    S.append(_entry("c_hybrid_cached", "llm", ["E", "G1"], "c_hybrid_cached", mode="loaded", ram=12,
                    inputs="frozen C_hybrid stage-2 matrix + 3 cached listwise orders", hardware="cpu (cached LLM)",
                    artefact={"key": "c_hybrid", "path": "."}, notes="frozen LLM-stage hybrid (DEV only)"))
    # learned rankers
    for v in ("noemb", "emb"):
        name = f"LTR_{v}_hgb"
        path = ep_run / "fusion/pipelines" / name / "scores.npy"
        S.append(_entry(f"ltr_{v}_frozen", "ranker", ["C", "D", "G1"], "frozen_pipeline", mode="loaded",
                        args={"pipeline": name}, ram=FROZEN_RAM, inputs=f"frozen pipeline {name}",
                        artefact={"key": "ep_run", "path": f"fusion/pipelines/{name}/scores.npy",
                                  "manifest_sha256": _manifest_sha(path)}))
        common = dict(mode="rebuilt", ram=18, threads=4, training="outer-fold training labels (frozen hyper-parameters, seeds)",
                      inputs=f"{'NO-EMB' if v == 'noemb' else 'all'} frozen sources, development pair table")
        S.append(_entry(f"ltr_{v}_refit", "ranker", ["C", "D", "G1"], "ltr", args={"variant": v, "table": "dev"}, **common))
        S.append(_entry(f"ltr_{v}_linear", "ranker", ["D"], "ltr", args={"variant": v, "table": "dev", "method": "linear"},
                        grid="coordinate ascent on the training folds", **common))
        S.append(_entry(f"ltr_{v}_lambdamart", "ranker", ["D"], "ltr",
                        args={"variant": v, "table": "dev", "method": "lambdamart"},
                        grid="lr {.05,.1} x leaves {15,63} (D57 test 6)", **common))
        S.append(_entry(f"rrf_all_{v}", "fusion", ["D"], "fusion", mode="rebuilt", ram=14,
                        args={"method": "rrf", "members": [f"src_{s}" if f"src_{s}" in {e['id'] for e in S} else _alias(s)
                                                           for s in _family_sources(meta, v)]},
                        inputs=f"every {'NO-EMB' if v == 'noemb' else ''} frozen full source, RRF k=60"))
    s0 = {"variant": "noemb", "table": "dev", "extra": list(TOPIC)}
    S.append(_entry("S0", "ranker", ["C", "D"], "ltr", mode="new", args=s0, ram=20, threads=4, answer="P3",
                    inputs="NO-EMB frozen sources + topic columns (lexical + topic + past cases)",
                    training="outer-fold training labels (frozen hyper-parameters, seeds)"))
    for enc in LADDER_ENCODERS:
        own = ENCODER_SOURCES.get(enc, [])
        S.append(_entry(f"S0+{enc}", "ranker", ["C", "E5"], "ltr", mode="new",
                        args={**s0, "extra": list(TOPIC) + ([] if own else [enc]), "add_sources": own},
                        ram=20, threads=4, answer="P4" if enc == "qwen3_4b" else None,
                        inputs=(f"S0 + the frozen sources of {enc} ({", ".join(own)})" if own else
                                f"S0 + three columns of {enc} (robust z, log rank, instrument-max z)"),
                        training="outer-fold training labels"))
        S[-1]["ladder_definition"] = "with_sources" if own else "dense_only"
        if own:                     # D78: one ladder definition (S0 + the encoder's dense column only) for every encoder
            S[-1]["notes"] = "extra row (D78): the encoder with its past-case sources; F2 uses S0+" + enc + "_dense"
            S.append(_entry(f"S0+{enc}_dense", "ranker", ["C", "E5"], "ltr", mode="new", ram=20, threads=4,
                            **dense_only_twin(enc, s0), training="outer-fold training labels"))
            S[-1]["ladder_definition"] = "dense_only"
    sf = {"variant": "emb", "table": "dev", "extra": list(TOPIC)}
    S.append(_entry("S_full", "ranker", ["C", "D"], "ltr", mode="new", args=sf, ram=22, threads=4,
                    inputs="all frozen sources + topic columns", training="outer-fold training labels"))
    S.append(_entry("S_full_ext", "ranker", ["C"], "ltr", mode="new", args={**sf, "extension": "m1"}, ram=24, threads=4,
                    answer="P5", inputs="S_full + M1 historical citation memory",
                    training="outer-fold training labels; historical pool before 2020-08-14"))
    # D79 leave-one-out: S_full_ext without the colbert_bge pair features, refit exactly like S_full_ext
    S.append(_entry("S_full_ext_minus_colbert", "ranker", ["C"], "ltr", mode="new",
                    args={**sf, "extension": "m1", "drop_sources": ["colbert_bge"]}, ram=24, threads=4,
                    inputs="S_full_ext without the colbert_bge features (leave-one-out, D79)",
                    training="outer-fold training labels; historical pool before 2020-08-14",
                    notes="added after the freeze (D79): the contribution of late interaction inside the full ranker"))
    # C2 (D93, ANALYSIS_PLAN_C §3): channel and encoder-group removals, refit exactly like their base; the groups
    # are fixed in uol.final.c2
    from uol.final import c2
    for sid, base_id, drop, ext_key in c2.ROWS:
        base_args = {**sf, "extension": "m1"} if base_id == "S_full_ext" else dict(sf)
        S.append(_entry(sid, "ranker", ["C2"], "ltr", mode="new",
                        args={**base_args, "drop_sources": list(drop), **({"extension_key": ext_key} if ext_key else {})},
                        ram=24, threads=4, inputs=f"{base_id} without {len(drop)} embedding sources (C2, D93)",
                        training="outer-fold training labels",
                        notes="added after the freeze (D93): " + (f"the extension uses its {ext_key} key; " if ext_key else "")
                              + "sources: " + ", ".join(drop)))
    S.append(_entry("S_full_ext_importance", "ranker", ["C2"], "ltr", mode="new",
                    args={**sf, "extension": "m1", "importance": dict(c2.IMPORTANCE)}, ram=28, threads=4,
                    inputs="S_full_ext refit, then grouped permutation importance per held-out case (C2, D93)",
                    training="outer-fold training labels",
                    notes="added after the freeze (D93): its ranking equals S_full_ext; the receipt holds the losses"))
    for x in ("query2doc", "hyde"):
        S.append(_entry(f"S_full+{x}", "ranker", ["C", "E8"], "ltr", mode="new", args={**sf, "extra": list(TOPIC) + [x]},
                        ram=22, threads=4, inputs=f"S_full + three columns of {x}", training="outer-fold training labels"))
    for seed in (1, 2):
        for base_id, args in (("S0", s0), ("S_full", sf), ("S_full_ext", {**sf, "extension": "m1"})):
            S.append(_entry(f"{base_id}_seed{seed}", "ranker", ["G"], "ltr", mode="new",
                            args={**args, "seed_offset": seed}, ram=22, threads=4, inputs=f"{base_id}, tree seed +{seed}"))
    # Block B: L* with headings + one representation
    for x in FUSION_PARTNERS:
        for method in ("rrf", "linear"):
            S.append(_entry(f"{method}__lstar_headings__{x}", "fusion", ["B"], "fusion", mode="new",
                            args={"method": method, "members": ["lstar_headings", x]}, ram=14,
                            grid=None if method == "rrf" else "w {0,.1,...,1} (inner folds)",
                            status="pending_vectors" if x in GPU_ENCODERS or x == "bge_m3_sparse" else "ready",
                            inputs=f"L* with headings + {x}", answer=None))
    return S


def _family_sources(meta: dict, variant: str) -> list[str]:
    from uol.embedding_push import fusion_ltr as fl
    full = [s for s in meta["sources"] if s["kind"] == "full" and not s["id"] in fl.DERIVED_IDS]
    return [s["id"] for s in full if variant == "emb" or s["family"] == "NO-EMB"]


def _alias(src: str) -> str:
    rev = {v: k for k, v in SAVED_ENCODERS.items()}
    return {"L_star": "lstar_frozen", "docexp_bm25": "docexp_bm25"}.get(src, rev.get(src, f"src_{src}"))


def write_systems(path: Path = SYSTEMS_JSON) -> dict[str, Any]:
    systems = build_systems()
    ids = [s["id"] for s in systems]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate system ids")
    payload = {"plan": "research/programs/final_benchmark_20261021/PLAN_DRAFT.md", "release": context.active().release_id,
               "n_systems": len(systems), "systems": systems}
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return payload


_LOADED: dict[str, dict[str, Any]] = {}


def systems(path: Path | None = None) -> dict[str, dict[str, Any]]:
    p = Path(path) if path else Path(__import__("os").environ.get("UOL_FINAL_SYSTEMS", SYSTEMS_JSON))
    key = str(p)
    if key not in _LOADED:
        _LOADED.clear()
        data = json.loads(p.read_text(encoding="utf-8"))
        _LOADED[key] = {s["id"]: s for s in data["systems"]}
    return _LOADED[key]


def get(sid: str) -> dict[str, Any]:
    s = systems()
    if sid not in s:
        raise KeyError(f"unknown system {sid!r}")
    return s[sid]


def register(spec: dict[str, Any]) -> None:
    """Add an ad-hoc entry in this process (tests and smoke runs)."""
    systems()[spec["id"]] = spec


def make(sid_or_spec: str | dict[str, Any]):
    spec = get(sid_or_spec) if isinstance(sid_or_spec, str) else sid_or_spec
    mod, cls = ADAPTERS[spec["adapter"]].split(":")
    return getattr(importlib.import_module(mod), cls)(spec)
