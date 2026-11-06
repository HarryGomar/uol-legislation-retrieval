"""L2 features: multi-query retrieval from Luna's legal-issue decompositions (PLAN §3 L2; feature route).

For each dev row and source of sub-queries, every sub-query is scored against the whole catalogue with BM25 (default
parameters, label-free) and with Qwen3-Embedding-8B (the dense_qwen8b query variant "c", MaxP over article windows,
vectors through the shared content-hash store). Per (row, pooled candidate) six columns:
max BM25 over sub-queries; # sub-queries with the article in their BM25 top 100; log1p best BM25 rank; and the same three
for dense. Sources: Luna variant ``a`` (concepts only), ``b`` (law names allowed); controls ``clause`` (a non-LLM split of
the Hechos into clauses) and ``random`` (the variant-a decomposition of a seeded random other dev row). No labels are
used, so the columns are identical for every outer group (no nesting needed).

    python -m uol.round3.l2_features build --source a|b|clause|random     # embeds sub-queries (endpoint), writes columns
    python -m uol.round3.l2_features arms                                   # runs L2 and its controls through arm_runner
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np

from uol.round3 import arm_runner, common as C, features

OUT = C.ARMS / "L2"
TOPN = 100
COLS = ["bm25_max", "bm25_hits", "bm25_best_rank", "dense_max", "dense_hits", "dense_best_rank"]


def luna_issues(variant: str) -> dict[int, list[str]]:
    path = OUT / "luna" / f"L2_{variant}.jsonl"
    return {int(r["row"]): r["issues"] for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}


def clause_split(text: str, max_parts: int = 5) -> list[str]:
    parts = [p.strip() for p in re.split(r"(?<=[.;:])\s+|,\s+(?:y|pero|aunque|sin embargo|por lo que|toda vez que)\s+", text)
             if len(p.strip()) > 25]
    if not parts:
        return [text]
    if len(parts) > max_parts:                 # merge into at most max_parts contiguous chunks
        k = int(np.ceil(len(parts) / max_parts))
        parts = [" ".join(parts[i:i + k]) for i in range(0, len(parts), k)]
    return parts


def subqueries(source: str) -> dict[int, list[str]]:
    rows = [int(r) for r in C.dev_rows()]
    if source in ("a", "b"):
        q = luna_issues(source)
        missing = [r for r in rows if r not in q]
        if missing:
            raise RuntimeError(f"L2 variant {source}: {len(missing)} dev rows lack a decomposition (coverage must be 100%)")
        return {r: q[r] for r in rows}
    if source == "clause":
        d = C.release()
        return {r: clause_split(d.queries[r].text) for r in rows}
    if source == "random":
        q = luna_issues("a")
        perm = np.random.default_rng(C.SEED + 17).permutation(rows)
        while np.any(perm == np.asarray(rows)):
            perm = np.random.default_rng(int(perm[0])).permutation(rows)
        return {r: q[int(p)] for r, p in zip(rows, perm)}
    raise ValueError(source)


def embed(texts: list[str]) -> np.ndarray:
    from uol.embedding_push import qwen_endpoint as Q
    return Q.embed_texts([Q.query_input("c", t) for t in texts], label="round3_L2")


def build(source: str) -> dict:
    from uol.embedding_push import qwen_endpoint as Q
    ctx, F = features.context()
    sq = subqueries(source)
    rows = sorted(sq)
    flat = [(r, s) for r in rows for s in sq[r]]
    uniq = sorted({s for _, s in flat})
    vec = embed(uniq)
    vix = {s: i for i, s in enumerate(uniq)}
    lex = ctx.lex("bm25")
    path = OUT / f"cols_{source}.npy"
    cols = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(F.pq), len(COLS)))
    cols[:] = np.nan
    for s0 in range(0, len(rows), 64):
        block = rows[s0:s0 + 64]
        texts = [s for r in block for s in sq[r]]
        owner = np.asarray([i for i, r in enumerate(block) for _ in sq[r]])
        with C.cpu_slot():
            bm = lex.score(texts).astype(np.float32)
            de = Q.dense_block(vec[[vix[t] for t in texts]])
        for i, r in enumerate(block):
            sel = owner == i
            sl, pc = slice(int(F.row_ptr[r]), int(F.row_ptr[r + 1])), F.pc[int(F.row_ptr[r]):int(F.row_ptr[r + 1])]
            out = np.empty((len(pc), 6), dtype=np.float32)
            for j, M in enumerate((bm[sel], de[sel])):
                order = np.argsort(-M, axis=1, kind="stable")
                rank = np.empty_like(order)
                np.put_along_axis(rank, order, np.arange(1, M.shape[1] + 1)[None, :].repeat(len(M), 0), axis=1)
                sub = rank[:, pc]
                out[:, 3 * j] = M[:, pc].max(axis=0)
                out[:, 3 * j + 1] = (sub <= TOPN).sum(axis=0)
                out[:, 3 * j + 2] = np.log1p(sub.min(axis=0))
            cols[sl] = out
    cols.flush()
    meta = {"source": source, "rows": len(rows), "subqueries": len(flat), "unique_subqueries": len(uniq),
            "endpoint_inputs_requested": len(uniq), "columns": COLS, "path": str(path), "sha256": C.sha256_file(path),
            "utc": C.utc()}
    C.write_json(OUT / f"cols_{source}.json", meta)
    C.ledger({"kind": "job_end", "job_id": f"L2_features_{source}", "exit_code": 0, "endpoint_inputs": len(uniq),
              "outputs": [{"path": str(path), "sha256": meta["sha256"]}]})
    return meta


def _cols(source: str) -> np.ndarray:
    return np.load(OUT / f"cols_{source}.npy", mmap_mode="r")


def columns_for(cfg, g, h, draw=None):
    srcs = cfg["sources"].split("+")
    return np.hstack([np.asarray(_cols(s)) for s in srcs])


def standalone(source: str) -> dict:
    """The arm alone: rank every dev row by its max dense sub-query score (BM25 as tie-breaker)."""
    from uol.embedding_push import fusion_ltr as fl
    ctx, F = features.context()
    c = _cols(source)
    lstar = C.ep.open_scores("L_star")
    out = np.full((len(ctx.data.queries), ctx.N), np.nan, dtype=np.float32)
    idx = np.flatnonzero(np.isfinite(np.asarray(c[:, 3])))
    for g in [x for x in ctx.groups if not x.is_lockbox]:
        sel = idx[np.isin(F.pq[idx], g.rows)]
        sc = np.asarray(c[sel, 3]) + 1e-3 * np.asarray(c[sel, 0]) / (1 + np.abs(np.asarray(c[sel, 0])).max())
        fl.write_pooled(ctx, F, out, g.rows, sel, sc.astype(np.float32), lstar)
    res = features.evaluate_dev(out)
    return {p: {"ndcg_at_10": res[p]["macro"]["ndcg_at_10"]} for p in res}


def arms() -> None:
    arm_runner.run("L2", [{"sources": "a"}, {"sources": "b"}, {"sources": "a+b"}], columns_for, task="T1",
                   subgroups={"multi_article": lambda rows: np.asarray([len(C.release().specific_idx[int(r)]) >= 2 for r in rows]),
                              "low_overlap": C.low_overlap},
                   standalone={"a": standalone("a"), "b": standalone("b")},
                   notes="L2 Luna decomposition (effort low); BM25 + Qwen3-8B multi-query features; configs: variant a, b, a+b",
                   extra_result={"luna_resample_agreement": "pending (100-row sample=1 re-run)"})
    for src in ("clause", "random"):
        arm_runner.run(f"L2_ctrl_{src}", [{"sources": src}], columns_for, task="T1", noemb=False, full_view=False,
                       notes=f"L2 control: {src} sub-queries through the same features")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "arms"])
    ap.add_argument("--source", default="a", choices=["a", "b", "clause", "random"])
    a = ap.parse_args()
    if a.command == "build":
        print(json.dumps(build(a.source), indent=1))
    else:
        arms()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
