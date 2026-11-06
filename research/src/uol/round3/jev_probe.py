"""Jev smoke test (D51): TypeSafe's Jev decision model as a pointwise reranker over the top 20 of ``LTR_emb_hgb``.

Dev-only and descriptive. For 40 fixed-seed dev P_spec rows, each request sends the masked *Hechos* as the state and
one yes/no question per candidate article (the article's public text, truncated). No label, no lockbox or prospective row
and no credential leaves this module; the key is read from ``TYPESAFE_API_KEY`` and never written. Readouts on the same
rows: the ranker alone, the top 20 reordered by Jev's probability, and a reciprocal-rank fusion of the two (k 60, fixed
before any call).

    python -m uol.round3.jev_probe one        # one request, to check the schema
    python -m uol.round3.jev_probe run        # 40 rows x 20 candidates (40 requests)
    python -m uol.round3.jev_probe score
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from functools import lru_cache
from pathlib import Path

import numpy as np

from uol.embedding_push import common as ep
from uol.embedding_push import fusion_ltr as fl
from uol.round3 import common as C
from uol.round3 import e2_adapter as E2

RUN = C.ROOT / "experiments/runs/jev_probe_20261019"
URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"
N_ROWS, DEPTH, SEED, RRF_K, ART_CHARS = 40, 20, 20261019, 60, 1500
INSTRUCTIONS = ("The state is the facts of a published Mexican federal court decision (in Spanish; cited provisions are "
                "masked as [CITA_OCULTA]). Does the court cite the following federal statutory article as part of the "
                "legal basis for its decision on these facts?\n\nArticle:\n")
CRITERIA = {"true": "The court relies on or applies this article for these facts.",
            "false": "This article is not part of the legal basis the court cites."}


def rows() -> np.ndarray:
    dev = C.population_rows("P_spec", C.dev_rows())
    return np.sort(np.random.default_rng(SEED).choice(dev, N_ROWS, replace=False)).astype(np.int64)


@lru_cache(maxsize=1)
def article_texts() -> list[str]:
    p = Path(C.release().candidate_binding["articles_path"])
    with open(p, encoding="utf-8") as f:
        return [json.loads(line)["instrument_article_body"] for line in f]


@lru_cache(maxsize=1)
def ltr() -> np.ndarray:
    return np.load(fl.pipeline_path("LTR_emb_hgb"), mmap_mode="r")


def top(r: int) -> np.ndarray:
    s = np.asarray(ltr()[int(r)], dtype=np.float64).copy()
    s[E2.generic_mask()] = -np.inf
    s[~np.isfinite(s)] = -np.inf
    return np.argsort(-s, kind="stable")[:DEPTH]


def request(r: int) -> dict:
    key = os.environ["TYPESAFE_API_KEY"]
    cands = top(r)
    texts = article_texts()
    qs = {f"a{i:02d}": {"type": "noul", "instructions": INSTRUCTIONS + texts[int(a)][:ART_CHARS], "criteria": CRITERIA}
          for i, a in enumerate(cands)}
    body = json.dumps({"model": MODEL, "state": {"facts": C.release().queries[int(r)].text}, "questions": qs}).encode()
    for attempt in range(6):
        req = urllib.request.Request(URL, data=body, method="POST", headers={
            "Authorization": "Bearer " + key, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
                rid = resp.headers.get("x-typesafe-request-id")
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"HTTP {e.code}: {e.read()[:300]!r}") from None
    p = [float(out["answers"][f"a{i:02d}"]["noul"]) for i in range(len(cands))]
    return {"row": int(r), "candidates": cands.tolist(), "p_yes": p, "model": out.get("model"),
            "usage": out.get("usage"), "request_id": rid, "seconds": round(time.perf_counter() - t0, 2)}


def run(limit: int | None = None) -> dict:
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / "responses.jsonl"
    done = {json.loads(x)["row"] for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [int(r) for r in rows() if int(r) not in done][:limit]
    for r in todo:
        C.append_jsonl(path, request(r))
    return {"rows_done": len(done) + len(todo)}


def score() -> dict:
    from uol.round3 import stats
    recs = {json.loads(x)["row"]: json.loads(x) for x in (RUN / "responses.jsonl").read_text(encoding="utf-8").splitlines()}
    rs = np.asarray(sorted(recs), dtype=np.int64)
    base = np.asarray(ltr()[rs], dtype=np.float64).copy()
    variants = {"ltr": base.copy(), "jev": base.copy(), "rrf": base.copy()}
    for i, r in enumerate(rs):
        c = np.asarray(recs[int(r)]["candidates"])
        p = np.asarray(recs[int(r)]["p_yes"])
        jev_rank = np.empty(len(c))
        jev_rank[np.lexsort((np.arange(len(c)), -p))] = np.arange(1, len(c) + 1)
        ltr_rank = np.arange(1, len(c) + 1)
        variants["jev"][i, c] = 1e6 - jev_rank
        variants["rrf"][i, c] = 1e6 + 1 / (RRF_K + ltr_rank) + 1 / (RRF_K + jev_rank)
    res = {k: ep.evaluate(v, rs, "P_spec") for k, v in variants.items()}
    out = {k: {"ndcg_at_10": v["macro"]["ndcg_at_10"], "mrr": v["macro"]["mrr"]} for k, v in res.items()}
    for k in ("jev", "rrf"):
        con = stats.contrast(res[k]["per_component"]["ndcg_at_10"], res["ltr"]["per_component"]["ndcg_at_10"])
        out[k]["vs_ltr"] = {x: con[x] for x in ("delta", "ci95")}
    pos_p, neg_p = [], []
    for r in rs:
        gold = set(int(a) for a in E2.spec_positives(int(r)))
        for a, p in zip(recs[int(r)]["candidates"], recs[int(r)]["p_yes"]):
            (pos_p if a in gold else neg_p).append(p)
    from math import fsum
    auc = fsum(float(np.mean([p > n for n in neg_p]) + .5 * np.mean([p == n for n in neg_p])) for p in pos_p) / max(len(pos_p), 1)
    tok = [x["usage"]["input_tokens"] for x in recs.values() if x.get("usage")]
    out.update({"rows": len(rs), "components": res["ltr"]["n_components"], "depth": DEPTH,
                "cited_in_top20": len(pos_p), "p_yes_mean_cited": float(np.mean(pos_p)) if pos_p else None,
                "p_yes_mean_other": float(np.mean(neg_p)), "auc_cited_vs_other": auc,
                "models": sorted({x["model"] for x in recs.values()}), "input_tokens": int(sum(tok)),
                "usd_at_0.042_per_M": round(sum(tok) * 0.042e-6, 4),
                "median_seconds": float(np.median([x["seconds"] for x in recs.values()])),
                "label": "dev, descriptive (40 rows, one draw; D51 smoke test)"})
    C.write_json(RUN / "SCORE.json", out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["one", "run", "score"])
    a = ap.parse_args()
    r = {"one": lambda: run(1), "run": run, "score": score}[a.cmd]()
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
