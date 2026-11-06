"""C2 (D93, 23 Oct 2026): channels and encoder groups of the full ranker's embedding sources, as fixed in
``research/programs/final_benchmark_20261021/ANALYSIS_PLAN_C.md`` §3 before RUN-B's results. One source of truth for
the leave-one-out rows (``registry``), the grouped permutation importance (``adapters.ltr``) and F8 (``stats``).

The 30 embedding sources of the pair table (``LTR_emb_hgb`` minus ``LTR_noemb_hgb``) are split into the direct
facts -> article channel (D), the embedding-keyed past-case channel (M) and the fusions that mix them. An encoder
group holds every source whose per-fold key search includes that encoder, even in folds that chose another key.
"""
from __future__ import annotations

D = ("dense_bge_firstp", "dense_bge_long", "dense_bge_maxp", "dense_e5_maxp", "dense_qwen8b", "ft_e5_q", "colbert_bge")
M = ("K_qwen8b", "knn_adapt_all", "knn_adapt_spec", "knn_bge", "knn_casesim_e5", "knn_casesim_e5_all", "knn_e5",
     "knn_fused", "knn_fused_qwen8b", "knn_qwen8b", "knn_x_all", "knn_x_spec", "knn_xc_all", "knn_xc_spec",
     "proto_adapt", "proto_bge", "proto_casesim_e5", "proto_e5", "proto_probe", "proto_qwen8b")
MIXED = ("H_bge", "H_qwen8b", "KH_qwen8b")
ENCODER_GROUPS = {
    "qwen3_8b": ("dense_qwen8b", "H_qwen8b", "KH_qwen8b", "K_qwen8b", "knn_qwen8b", "knn_fused_qwen8b", "proto_qwen8b"),
    "bge_m3": ("dense_bge_firstp", "dense_bge_long", "dense_bge_maxp", "colbert_bge", "H_bge", "knn_bge", "proto_bge",
               "knn_adapt_all", "knn_adapt_spec", "knn_casesim_e5", "knn_casesim_e5_all", "knn_fused", "knn_x_all",
               "knn_x_spec", "knn_xc_all", "knn_xc_spec", "proto_adapt", "proto_casesim_e5", "proto_probe"),
    "e5": ("dense_e5_maxp", "ft_e5_q", "knn_e5", "proto_e5", "knn_adapt_all", "knn_adapt_spec", "knn_casesim_e5",
           "knn_casesim_e5_all", "knn_fused", "knn_x_all", "knn_x_spec", "knn_xc_all", "knn_xc_spec", "proto_adapt",
           "proto_casesim_e5", "proto_probe"),
}

# (system id, base, sources dropped, extension key override) — the configuration rows of systems.json
ROWS = (
    ("S_full_minus_D", "S_full", D + MIXED, None),
    ("S_full_minus_M", "S_full", M + MIXED, None),
    # the extension's dense key is the Qwen3-8B vector: it keeps each fold's decay and k with the lexical key
    ("S_full_ext_minus_qwen3_8b", "S_full_ext", ENCODER_GROUPS["qwen3_8b"], "lex"),
    ("S_full_ext_minus_bge_m3", "S_full_ext", ENCODER_GROUPS["bge_m3"], None),
    ("S_full_ext_minus_e5", "S_full_ext", ENCODER_GROUPS["e5"], None),
)
# F8, the one confirmatory family of the plan (Holm, the frozen .02 classes)
F8 = (("S_full", "S_full_minus_D"), ("S_full", "S_full_minus_M"), ("S_full_ext", "S_full_ext_minus_qwen3_8b"))
IMPORTANCE = {"repeats": 20, "seed": 20261023}


def importance_groups(noemb_sources: list[str]) -> dict[str, tuple[str, ...]]:
    """Source groups permuted in the importance job; ``topic`` and ``extension`` name the extra column blocks."""
    return {"D": D, "M": M, "mixed": MIXED, "noemb": tuple(noemb_sources), "topic": ("topic",),
            "extension": ("extension",), **{f"enc_{k}": v for k, v in ENCODER_GROUPS.items()}}
