"""Dense-alone dev readouts and NO-EMB-plus-source LTR refits for the scale curve (dev, descriptive).

Dense alone: MaxP cosine over the article units, dev P_spec rows, the project evaluator (``embedding_push.common.
evaluate``: condensed specific lists, candidate-ID ties, component-macro). Each source is scored under the query
variants it has and the variant is chosen by dev P_spec nDCG@10 (the 8B selection rule; the 8B chose c), with
variant c also reported for every Qwen size. Unseen-band Recall@20 is the round-2 A pair-level definition (fold-local
bands; ``round3.e2_adapter.unseen_recall``). Contrasts are paired component bootstraps (``round3.stats.contrast``).

Refit: the frozen LTR_noemb_hgb pipeline refit per dev group (``round3.features.Harness``, frozen hyper-parameters
and seeds) with three label-free columns from one dense source (robust z, log rank, instrument-max z;
``e2_adapter.full_columns``, the E2 recipe). The same refit with the 8B columns is the within-harness reference.

    python -m uol.scale_curve.evaluate dense                      # light: every assembled source
    python -m uol.round3.heavy uol.scale_curve.evaluate refit --source qwen3_0p6b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.embedding_push import common as ep
from uol.scale_curve import common as C

EVAL = C.RUN / "eval"
SAVED = {"e5_small": ("dense_e5_maxp", "multilingual-e5-small (118M)"), "bge_m3": ("dense_bge_maxp", "bge-m3 (568M)")}
VECTOR_SOURCES = {"qwen3_0p6b": "Qwen3-Embedding-0.6B", "qwen3_4b": "Qwen3-Embedding-4B",
                  "qwen3_8b": "Qwen3-Embedding-8B (cached vectors)", "te3l": "OpenAI text-embedding-3-large"}
ORDER = ("e5_small", "bge_m3", "qwen3_0p6b", "qwen3_4b", "qwen3_8b")          # te3l cancelled (D30)
REFERENCE = "qwen3_8b"
LABEL = "dev, descriptive"


def available(key: str) -> bool:
    if key in SAVED:
        return (ep.score_path(SAVED[key][0]).parent / "manifest.json").exists()
    if key == "qwen3_8b":
        return (C.QWEN8B / "receipt.json").exists()
    return (C.MODELS[key].derived / "manifest.json").exists()


def variants_of(key: str) -> tuple[str, ...]:
    if key in SAVED:
        return ("saved",)
    if key == "qwen3_8b":
        return C.VARIANTS
    man = json.loads((C.MODELS[key].derived / "manifest.json").read_text(encoding="utf-8"))
    return tuple(v for v in C.VARIANTS if v in man["query_variants"])


_CACHE: dict[str, Any] = {}


def _vectors(key: str, variant: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    ck = f"{key}:{variant}"
    if ck not in _CACHE:
        _CACHE.clear()                                        # one source in memory at a time
        if key == "qwen3_8b":
            A = np.load(C.QWEN8B / "article_vectors.npy")
            owner = np.load(C.QWEN8B / "article_index.npy")
            ids = json.loads((C.QWEN8B / "query_ids.json").read_text(encoding="utf-8"))
            if ids != [q.query_id for q in ep.release().queries]:
                raise ValueError("8B query vectors are not in release order")
            Q = np.load(C.QWEN8B / f"query_vectors_{variant}.npy", mmap_mode="r")
            rows = np.arange(len(ids))
        else:
            A, owner, Q, rows = C.load_vectors(C.MODELS[key], variant)
        _CACHE[ck] = (A, owner, Q, {int(r): i for i, r in enumerate(rows)})
    return _CACHE[ck]


def dense_rows(key: str, variant: str, rows: Sequence[int]) -> np.ndarray:
    """[len(rows), N] dense scores of a source for dev release rows (MaxP for vector sources)."""
    rows = C.assert_dev(rows)
    if key in SAVED:
        return np.asarray(ep.open_scores(SAVED[key][0])[rows], dtype=np.float32)
    A, owner, Q, pos = _vectors(key, variant)
    q = np.asarray(Q[[pos[int(r)] for r in rows]], dtype=np.float32)
    n = len(ep.release().candidate_ids)
    out = np.empty((len(rows), n), dtype=np.float32)
    for s, block in ep.maxp_scores(q, A, owner, n):
        out[s:s + len(block)] = block
    return out


def _strip(res: dict) -> dict:
    for rec in res["per_query"]:
        rec.pop("topk", None)
    return res


def readout(res: dict) -> dict[str, Any]:
    from uol.round3 import e2_adapter as e2
    u = e2.unseen_recall(res)
    m = res["macro"]
    return {"ndcg_at_10": m["ndcg_at_10"], "recall_at_20": m.get("recall_at_20"), "mrr": m.get("mrr"),
            "recall_at_100": m.get("recall_at_100"), "n_queries": res["n_queries"], "n_components": res["n_components"],
            **{f"{b}_recall_at_20": u[b]["recall"] for b in ("unseen", "rare", "frequent")},
            **{f"{b}_pairs": u[b]["pairs"] for b in ("unseen", "rare", "frequent")}}


def evaluate_source(key: str) -> dict[str, Any]:
    rows = ep.population_rows("P_spec", ep.dev_rows())
    out: dict[str, Any] = {"source": key, "label": LABEL, "variants": {}}
    results = {}
    for v in variants_of(key):
        res = _strip(ep.evaluate(dense_rows(key, v, rows), rows, "P_spec"))
        results[v] = res
        out["variants"][v] = readout(res)
    keys = list(out["variants"])
    scores = [out["variants"][v]["ndcg_at_10"] for v in keys]
    best = max(scores)
    out["selected_variant"] = keys[min(i for i, s in enumerate(scores) if s >= best - 1e-12)]
    out["selection"] = "best dev P_spec nDCG@10 over the variants the source has (the 8B rule)"
    if key == "qwen3_8b":
        saved = np.asarray(ep.open_scores("dense_qwen8b")[rows], dtype=np.float32)
        own = dense_rows(key, "c", rows)
        out["reproduces_dense_qwen8b_file"] = {"max_abs_diff": float(np.max(np.abs(saved - own))),
                                               "ndcg_saved_file": ep.evaluate(saved, rows, "P_spec")["macro"]["ndcg_at_10"]}
    return out, results


def run_dense(keys: Sequence[str] = ORDER) -> dict[str, Any]:
    from uol.round3 import e2_adapter as e2
    from uol.round3 import stats

    summary: dict[str, Any] = {"label": LABEL, "metric": "dev P_spec component-macro nDCG@10 (MaxP dense alone)",
                               "population": "P_spec dev (1,218 queries)", "utc": C.utc(), "sources": {}}
    picked: dict[str, dict] = {}
    for key in keys:
        if not available(key):
            summary["sources"][key] = {"available": False}
            continue
        info, results = evaluate_source(key)
        sel = info["selected_variant"]
        picked[key] = {"selected": results[sel], "c": results.get("c")}
        summary["sources"][key] = {"available": True, "name": SAVED.get(key, (None, VECTOR_SOURCES.get(key)))[1], **info}
        C.log(f"dense {key}: " + ", ".join(f"{v} {r['ndcg_at_10']:.4f}/{r['unseen_recall_at_20']:.3f}"
                                           for v, r in info["variants"].items()) + f"; selected {sel}", "evaluate")
    if REFERENCE in picked:
        ref = picked[REFERENCE]["selected"]
        ref_u = e2.unseen_recall(ref)
        for key, pk in picked.items():
            if key == REFERENCE:
                continue
            for which in ("selected", "c"):
                res = pk[which]
                if res is None:
                    continue
                con = stats.contrast(res["per_component"]["ndcg_at_10"], ref["per_component"]["ndcg_at_10"])
                summary["sources"][key].setdefault("vs_qwen3_8b", {})[which] = {
                    "ndcg_delta": con["delta"], "ndcg_ci95": con["ci95"], "class": con["class"],
                    "unseen_recall_delta": e2.recall_contrast(e2.unseen_recall(res), ref_u, "unseen")}
    C.write_json(EVAL / "dense_alone.json", summary)
    return summary


# --------------------------------------------------------------------------- NO-EMB + one source refit (heavy)


def source_columns(h, key: str, variant: str, block: int = 256) -> np.ndarray:
    from uol.round3 import e2_adapter as e2
    cols = h.empty_columns(3)
    dev = ep.dev_rows()
    for s in range(0, len(dev), block):
        rows = dev[s:s + block]
        pos, c = e2.full_columns(dense_rows(key, variant, rows), rows)
        cols[pos] = c
    _, _, row_ptr = e2.pair_table()
    dev_pos = np.concatenate([np.arange(int(row_ptr[r]), int(row_ptr[r + 1])) for r in dev])
    if np.isnan(cols[dev_pos]).any():
        raise RuntimeError(f"{key}: dev pairs without columns")
    return cols


def baseline_preds(h) -> dict:
    from uol.round3 import arm_runner
    d = arm_runner.BASELINE_DIR / h.name
    if all((d / f"baseline_{h.name}_{g.role}.npz").exists() for g in h.groups):
        return h.refit(None, ckpt_dir=d, tag="baseline", log=lambda m: C.log(m, "refit"))
    return h.refit(None, ckpt_dir=C.RUN / "refit" / "baseline", tag="baseline", log=lambda m: C.log(m, "refit"))


def run_refit(key: str, variant: str | None = None, eval_dir: Path = EVAL, ckpt_root: Path = C.RUN / "refit") -> dict[str, Any]:
    """``eval_dir``/``ckpt_root`` let another program (``scale_curve.legal``) keep its own outputs; the 8B reference stays here."""
    from uol.round3 import common as R3
    from uol.round3 import e2_adapter as e2
    from uol.round3 import features as FT

    dense = json.loads((eval_dir / "dense_alone.json").read_text(encoding="utf-8"))
    variant = variant or (dense["sources"][key]["selected_variant"] if key not in SAVED else "saved")
    with R3.cpu_slot():
        h = FT.Harness("noemb", "spec")
        base = FT.evaluate_dev(h.matrix(baseline_preds(h)))
        cols = source_columns(h, key, variant)
        preds = h.refit(lambda g: cols, ckpt_dir=ckpt_root / key, tag=f"noemb_plus_{key}_{variant}",
                        log=lambda m: C.log(m, "refit"))
        arm = FT.evaluate_dev(h.matrix(preds))
    out = {"source": key, "variant": variant, "label": LABEL, "pipeline": "LTR_noemb_hgb + 3 dense columns (z, log rank, "
           "instrument-max z), refit per dev group with frozen hyper-parameters and seeds", "utc": C.utc(),
           "baseline_noemb": {p: base[p]["macro"]["ndcg_at_10"] for p in base},
           "frozen_LTR_emb_hgb": FT.frozen_dev_summary("LTR_emb_hgb")["P_spec"]["ndcg_at_10"],
           "arm": {p: arm[p]["macro"]["ndcg_at_10"] for p in arm},
           "arm_recall_at_20": {p: arm[p]["macro"].get("recall_at_20") for p in arm},
           "vs_noemb": FT.dev_contrasts(arm, base),
           "unseen_recall_at_20": {"arm": e2.unseen_recall(arm["P_spec"])["unseen"]["recall"],
                                   "noemb": e2.unseen_recall(base["P_spec"])["unseen"]["recall"],
                                   "delta": e2.recall_contrast(e2.unseen_recall(arm["P_spec"]),
                                                               e2.unseen_recall(base["P_spec"]), "unseen")}}
    ref_path = EVAL / f"refit_{REFERENCE}.npz"
    np.savez(eval_dir / f"refit_{key}.npz", **{f"{p}_pc": np.asarray(arm[p]["per_component"]["ndcg_at_10"]) for p in arm},
             **{f"{p}_components": np.asarray(arm[p]["components"]) for p in arm})
    if key != REFERENCE and ref_path.exists():
        z = np.load(ref_path)
        from uol.round3 import stats
        for p in arm:
            if list(z[f"{p}_components"]) != list(arm[p]["components"]):
                raise ValueError("components are not aligned with the 8B refit")
            con = stats.contrast(arm[p]["per_component"]["ndcg_at_10"], z[f"{p}_pc"])
            out.setdefault("vs_noemb_plus_qwen3_8b", {})[p] = {k: con[k] for k in ("delta", "ci95", "class", "mean_b")}
    C.write_json(eval_dir / f"refit_{key}.json", out)
    C.ledger({"kind": "refit", "source": key, "variant": variant, "P_spec": out["arm"]["P_spec"],
              "baseline": out["baseline_noemb"]["P_spec"]})
    C.log(f"refit {key} ({variant}): P_spec {out['arm']['P_spec']:.4f} vs NO-EMB {out['baseline_noemb']['P_spec']:.4f}",
          "refit")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scale-curve dev evaluation (dev, descriptive)")
    ap.add_argument("command", choices=["dense", "refit"])
    ap.add_argument("--source", default=None)
    ap.add_argument("--variant", default=None)
    a = ap.parse_args(argv)
    if a.command == "dense":
        run_dense(ORDER)
        return 0
    for key in a.source.split(","):
        run_refit(key, a.variant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
