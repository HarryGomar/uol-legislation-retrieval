"""A multi-vector encoder trained on the development cases (common_footing PLAN addendum 18), on the remote worker.

``prep`` writes the package to the exchange folder: development rows only, no test row, no test label. ``ingest`` reads
the worker's scores back, one out-of-fold score row per development case, and reads them with the evaluator of every
other system.

    py -3.12 -m uol.final.multivector prep --go
    py -3.12 -m uol.final.multivector ingest --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A

FOLDS = 5
INNER_FOLD = 1                        # fold 0's settings are chosen on this fold, with the model trained on folds 2, 3, 4
FOLDER = "multivector"


def exchange() -> Path:
    return Path(os.environ.get("UOL_GPU_EXCHANGE", r"exchange\UOL_GPU_EXCHANGE")) / FOLDER


def dev_by_fold() -> dict[int, np.ndarray]:
    from uol.embedding_push import common as ep
    from uol.final import guard
    roles = ep.roles()
    dev = guard.dev_rows_only(ep.dev_rows())
    return {k: dev[roles[dev] == f"dev_fold{k}"] for k in range(FOLDS)}


def write_jsonl(path: Path, items) -> int:
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
            n += 1
    return n


def pairs(d, rows: np.ndarray):
    for r in rows:
        for p in d.specific_idx[int(r)]:
            yield {"q": d.queries[int(r)].text, "pos": d.candidate_texts[int(p)]}


def prep() -> int:
    from uol.embedding_push import common as ep
    from uol.final import context
    context.active()
    d = ep.release()
    by = dev_by_fold()
    out = exchange()
    out.mkdir(parents=True, exist_ok=True)
    n_art = write_jsonl(out / "articles.jsonl", ({"i": i, "text": t} for i, t in enumerate(d.candidate_texts)))
    (out / "generic.json").write_text(json.dumps([int(i) for i in d.generic_idx]), encoding="utf-8")
    counts = {"articles": n_art, "folds": {}}
    for k in range(FOLDS):
        f = out / f"f{k}"
        f.mkdir(exist_ok=True)
        train = np.concatenate([by[j] for j in range(FOLDS) if j != k])
        counts["folds"][k] = {"train_pairs": write_jsonl(f / "train.jsonl", pairs(d, train)),
                              "queries": write_jsonl(f / "queries.jsonl", ({"row": int(r), "text": d.queries[int(r)].text} for r in by[k]))}
    inner = out / "inner"
    inner.mkdir(exist_ok=True)
    fit = np.concatenate([by[j] for j in range(FOLDS) if j not in (0, INNER_FOLD)])
    val = ep.population_rows("P_spec", by[INNER_FOLD])
    counts["inner"] = {"train_pairs": write_jsonl(inner / "train.jsonl", pairs(d, fit)),
                       "val": write_jsonl(inner / "val.jsonl", ({"row": int(r), "text": d.queries[int(r)].text, "cited": [int(c) for c in d.specific_idx[int(r)]]}
                                                                for r in val if len(d.specific_idx[int(r)])))}
    A.write_json(out / "COUNTS.json", counts)
    print(json.dumps(counts, indent=1), flush=True)
    return 0


def ingest() -> int:
    """Each fold's scores into one source, then its reading alone on the development cases."""
    from uol.embedding_push import common as ep
    from uol.final import context, test_set as TS
    context.active()
    d = ep.release()
    out = exchange() / "outputs"
    dst = FT.A.RUN / "multivector"
    dst.mkdir(parents=True, exist_ok=True)
    res = {}
    for name, folder in (("trained", lambda k: out / f"f{k}"), ("untrained", lambda k: out / "base" / f"f{k}")):
        rows, blocks = [], []
        for k in range(FOLDS):
            q = [json.loads(x)["row"] for x in (exchange() / f"f{k}" / "queries.jsonl").read_text(encoding="utf-8").splitlines()]
            S = np.load(folder(k) / "scores.npy", mmap_mode="r")
            if S.shape != (len(q), len(d.candidate_ids)):
                raise RuntimeError(f"{name} f{k}: {S.shape}")
            rows += q
            blocks.append(np.asarray(S, dtype=np.float32))
        S = np.concatenate(blocks)
        order = np.argsort(rows)
        rows = [int(rows[i]) for i in order]
        np.save(dst / f"{name}_scores.npy", S[order])
        A.write_json(dst / f"{name}_rows.json", rows)
        keep = ep.population_rows("P_spec", np.asarray(rows))
        pos = {r: i for i, r in enumerate(rows)}
        _, summary = TS.measure([int(r) for r in keep], np.asarray(S[order][[pos[int(r)] for r in keep]], dtype=np.float64), False)
        res[name] = {m: summary.get(m) for m in ("ndcg_at_10", "success_at_1", "success_at_10", "recall_at_20", "unseen_recall_at_20", "n_cases", "n_groups")}
    A.write_json(dst / "alone.json", res)
    print(json.dumps(res, indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "ingest"])
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] {a.step}")
        return 0
    return prep() if a.step == "prep" else ingest()


if __name__ == "__main__":
    sys.exit(main())
