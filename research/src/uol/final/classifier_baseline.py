"""The task as multi-label classification (common_footing PLAN addendum 19): one classifier per article cited in training.

Word features: TF-IDF of the facts, one logistic regression per article. Encoder features: the facts' stored Qwen3-8B
vector, one ridge regression per article, in closed form. An article no training case cites has no classifier and is
ranked after every other. Out of fold on the development cases, read with the evaluator of every other system.

    py -3.12 -m uol.final.classifier_baseline --go
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A

FOLDS = 5
C_GRID = (1.0, 10.0, 100.0, 1000.0)
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0)
WORKERS = 4


def labels(d, rows: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """The articles cited in ``rows``, and per row its positions among them."""
    classes = sorted({int(c) for r in rows for c in d.specific_idx[int(r)]})
    at = {c: i for i, c in enumerate(classes)}
    return np.asarray(classes), [[at[int(c)] for c in d.specific_idx[int(r)]] for r in rows]


def _one_logistic(X, y, C: float) -> tuple[np.ndarray, float]:
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(solver="liblinear", C=C, max_iter=200)
    m.fit(X, y)
    return m.coef_[0].astype(np.float32), float(m.intercept_[0])


def words(d, train: np.ndarray, test: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    from joblib import Parallel, delayed
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(sublinear_tf=True, min_df=2, ngram_range=(1, 2), max_features=60000, strip_accents="unicode")
    Xa = v.fit_transform([d.queries[int(r)].text for r in train])
    Xb = v.transform([d.queries[int(r)].text for r in test])
    classes, pos = labels(d, train)
    rows_of: dict[int, list[int]] = {}
    for i, p in enumerate(pos):
        for c in p:
            rows_of.setdefault(c, []).append(i)
    Y = [np.isin(np.arange(Xa.shape[0]), rows_of[j]).astype(np.int8) for j in range(len(classes))]
    got = Parallel(n_jobs=WORKERS, batch_size=32)(delayed(_one_logistic)(Xa, y, C) for y in Y)
    W = np.stack([g[0] for g in got])
    b = np.asarray([g[1] for g in got], dtype=np.float32)
    return classes, np.asarray(sparse.csr_matrix(Xb).dot(W.T), dtype=np.float32) + b[None, :]


def encoder(d, train: np.ndarray, test: np.ndarray, Q: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray]:
    Xa, Xb = Q[train].astype(np.float64), Q[test].astype(np.float64)
    classes, pos = labels(d, train)
    Y = np.zeros((len(train), len(classes)), dtype=np.float64)
    for i, p in enumerate(pos):
        Y[i, p] = 1.0
    K = Xa @ Xa.T
    alpha = np.linalg.solve(K + lam * np.eye(len(K)), Y)
    return classes, (Xb @ Xa.T @ alpha).astype(np.float32)


def ranked_ndcg(d, rows: np.ndarray, classes: np.ndarray, S: np.ndarray) -> float:
    generic = {int(i) for i in d.generic_idx}
    out = []
    for i, r in enumerate(rows):
        gold = {int(c) for c in d.specific_idx[int(r)]}
        if not gold:
            continue
        top = [int(classes[j]) for j in np.argsort(-S[i], kind="stable")[:60] if int(classes[j]) not in generic][:10]
        dcg = sum(1 / np.log2(k + 2) for k, c in enumerate(top) if c in gold)
        out.append(dcg / sum(1 / np.log2(k + 2) for k in range(min(len(gold), 10))))
    return float(np.mean(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print("[dry] classifier baseline")
        return 0
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.final import audit_d181_dev as D, guard, measures
    import pandas as pd
    D.activate(FT.STORE)                                   # the corrected labels: 1,217 rows, not 1,218
    d = ep.release()
    roles = ep.roles()
    dev = guard.dev_rows_only(ep.dev_rows())
    fold = {k: dev[roles[dev] == f"dev_fold{k}"] for k in range(FOLDS)}
    Q = qe.query_vectors("c")
    out = FT.A.RUN / "classifier"
    out.mkdir(parents=True, exist_ok=True)
    # one setting per kind, chosen once: trained on folds 2, 3, 4, read on fold 1, as the other trained components' inner folds
    fit = np.concatenate([fold[j] for j in (2, 3, 4)])
    val = ep.population_rows("P_spec", fold[1])
    chosen = {"words": {}, "encoder": {}}
    for C in C_GRID:
        cl, S = words(d, fit, val, C)
        chosen["words"][C] = ranked_ndcg(d, val, cl, S)
    for lam in RIDGE_GRID:
        cl, S = encoder(d, fit, val, Q, lam)
        chosen["encoder"][lam] = ranked_ndcg(d, val, cl, S)
    C = max(chosen["words"], key=lambda k: (chosen["words"][k], -k))
    lam = max(chosen["encoder"], key=lambda k: (chosen["encoder"][k], k))
    print(json.dumps({"inner": chosen, "C": C, "lambda": lam}), flush=True)
    keep = np.asarray(ep.population_rows("P_spec", dev), dtype=np.int64)
    n = len(d.candidate_ids)
    pos = {int(r) for r in keep}
    res = {"inner": {k: {str(x): v for x, v in t.items()} for k, t in chosen.items()}, "C": C, "lambda": lam}
    for name, fn in (("words", lambda tr, te: words(d, tr, te, C)), ("encoder", lambda tr, te: encoder(d, tr, te, Q, lam))):
        t0 = time.perf_counter()
        parts = []
        for k in range(FOLDS):
            te = np.asarray([r for r in fold[k] if int(r) in pos], dtype=np.int64)
            tr = np.concatenate([fold[j] for j in range(FOLDS) if j != k])
            cl, S = fn(tr, te)
            M = np.full((len(te), n), -1e9, dtype=np.float64) - np.arange(n)[None, :] * 1e-6
            M[:, cl] = S
            r = ep.evaluate(M, te, "P_spec", top_k=measures.TOP_K, allow_lockbox=False)
            at = {int(x): j for j, x in enumerate(te)}
            per = measures.per_case(r["per_query"], r["rows"], M[[at[int(x)] for x in r["rows"]]], specific=True, unseen=measures.unseen_mask(tr))
            parts.extend({"query_id": d.queries[int(x)].query_id, "row": int(x), "component_id": d.queries[int(x)].component_id, "fold": k, **rec}
                         for x, rec in zip(r["rows"], per))
            print(f"{name} fold {k}: {time.perf_counter() - t0:.0f} s", flush=True)
        frame = pd.DataFrame(parts)
        summary = measures.summary(frame)
        frame.to_parquet(out / f"per_case_{name}.parquet", index=False)
        res[name] = {**{m: summary.get(m) for m in ("ndcg_at_10", "success_at_1", "success_at_10", "law_at_1", "recall_at_20", "unseen_recall_at_20", "n_cases", "n_groups")},
                     "seconds": round(time.perf_counter() - t0, 1)}
    A.write_json(out / "summary.json", res)
    A.receipt("classifier_baseline", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
