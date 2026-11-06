"""LLM alone (closed-book) on every development evaluation case (D55, 19 Oct 2026; dev, descriptive).

The LLM receives only the masked Hechos (the systems' input) and returns exactly twenty (law, article) pairs, most likely
first; its first ten countable articles are scored, like the retrieval systems' top ten. Its list is read as a ranking: names are mapped to catalogue articles, generic articles and unresolvable names
are dropped, a repeated article counts once, and every article it did not name is unretrieved. Scored by nDCG@10 and
the top-10 hit rate on the evaluation cases (P_spec, component-macro), next to the retrieval systems and the LLM stage
on exactly the same rows, with paired component-bootstrap intervals.

Same model, client and leakage guard as the Round 3 probes (``uol.round3.luna``, ``prompt_guard``); reasoning effort
"medium", as in the LLM reranking stage; 10 cases per call; prompt ``round3/prompts/closedbook_rank20.txt``. Development
rows only: the runner refuses any lockbox row. The development cases predate the model's stated training cutoff, so
the result is optimistic for the LLM. Outputs are cached (re-running costs no calls).

    PYTHONPATH=src python -m uol.eval_lenses.llm_alone smoke      # 20 cases, one call
    PYTHONPATH=src python -m uol.eval_lenses.llm_alone run        # all 1,218, then score
    PYTHONPATH=src python -m uol.eval_lenses.llm_alone score      # rescore saved outputs, no calls
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from uol.embedding_push import common as ep
from uol.eval_lenses.qk import SYSTEMS, inst_code
from uol.facts_to_law import metrics
from uol.round3 import catalog_map, luna, prompt_guard
from uol.round3 import common as C

TEMPLATE = Path(__file__).resolve().parents[1] / "round3" / "prompts" / "closedbook_rank20.txt"
OUT = ep.ROOT / "experiments" / "runs" / "llm_alone_20261019"
BATCH = 10
EFFORT = "medium"
TAG = "llm_alone_dev"
ITEM = {"type": "object", "additionalProperties": False, "required": ["articulos"],
        "properties": {"articulos": {"type": "array", "minItems": 20, "maxItems": 20, "items": {
            "type": "object", "additionalProperties": False, "required": ["ley", "articulo"],
            "properties": {"ley": {"type": "string"}, "articulo": {"type": "string"}}}}}}
DRAWS, SEED = 10_000, 20261019


def dev_eval_rows() -> list[int]:
    rows = ep.population_rows("P_spec", ep.dev_rows())
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("development rows only")
    return [int(r) for r in rows]


def build(rows: list[int]) -> tuple[str, list[str], str]:
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("development rows only")
    tpl = TEMPLATE.read_text(encoding="utf-8")
    d = C.release()
    ids = [f"c{i + 1:02d}" for i in range(len(rows))]
    items = "\n\n".join(f"[id={i}] {re.sub(r'\s+', ' ', d.queries[r].text).strip()}" for i, r in zip(ids, rows))
    prompt = tpl.replace("{items}", items)
    prompt_guard.check(prompt, rows, template=tpl)
    return prompt, ids, hashlib.sha256(tpl.encode("utf-8")).hexdigest()


def run(rows: list[int]) -> dict[int, dict]:
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    built = [build(b) for b in batches]
    schema = luna.batch_schema(ITEM)
    res = luna.run_many([(p, schema) for p, _, _ in built], effort=EFFORT, tag=TAG, workers=4)
    out = {}
    for (prompt, ids, sha), b, r in zip(built, batches, res):
        got = luna.checked(r, prompt, schema, ids, effort=EFFORT, tag=TAG)
        for i, row in zip(ids, b):
            out[row] = {"articulos": got[i]["articulos"], "prompt_sha256": sha}
    return out


def per_query(rows: list[int], outputs: dict[int, dict]) -> dict[int, dict[str, float]]:
    d = ep.release()
    generic = set(int(i) for i in d.generic_idx)
    res = {}
    for r in rows:
        ranked: list[int] = []
        unresolved = 0
        for it in outputs[r]["articulos"]:
            idx, _ = catalog_map.resolve(it["ley"], it["articulo"])
            if idx is None:
                unresolved += 1
                continue
            if idx in generic or idx in ranked:
                continue
            ranked.append(int(idx))
        gold = [int(a) for a in d.specific_idx[r]]
        ranks = sorted(ranked.index(a) + 1 for a in gold if a in ranked)
        m = metrics.query_metrics(ranks, len(gold))
        gold_law = {inst_code(a) for a in gold}
        law_hit = float(any(inst_code(a) in gold_law for a in ranked[:10]))
        res[r] = {"ndcg_at_10": m["ndcg_at_10"], "hit_at_10": float(bool(ranks) and ranks[0] <= 10),
                  "law_hit_at_10": law_hit, "kept": len(ranked), "unresolved": unresolved}
    return res


def _macro(vals: dict[int, float], comp: dict[int, str]) -> tuple[float, dict[str, float]]:
    by: dict[str, list[float]] = {}
    for r, v in vals.items():
        by.setdefault(comp[r], []).append(v)
    per = {c: float(np.mean(v)) for c, v in by.items()}
    return float(np.mean(list(per.values()))), per


def _paired(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    cs = sorted(a)
    diff = np.asarray([a[c] - b[c] for c in cs])
    boots = diff[np.random.default_rng(SEED).integers(0, len(diff), size=(DRAWS, len(diff)))].mean(axis=1)
    return {"delta": float(diff.mean()), "lo95": float(np.quantile(boots, .025)), "hi95": float(np.quantile(boots, .975))}


def score(rows: list[int], outputs: dict[int, dict]) -> dict[str, Any]:
    d = ep.release()
    comp = {r: d.queries[r].component_id for r in rows}
    llm = per_query(rows, outputs)
    table: dict[str, dict[str, float]] = {}
    per: dict[str, dict[str, dict[str, float]]] = {}
    for key in ("ndcg_at_10", "hit_at_10", "law_hit_at_10"):
        table.setdefault("LLM alone", {})[key], per.setdefault("LLM alone", {})[key] = _macro(
            {r: llm[r][key] for r in rows}, comp)
    table["LLM alone"]["mean_articles_kept"] = float(np.mean([llm[r]["kept"] for r in rows]))
    table["LLM alone"]["unresolved_share"] = float(sum(llm[r]["unresolved"] for r in rows) / (20 * len(rows)))
    systems = dict(SYSTEMS)
    sources: dict[str, Any] = {k: str(v) for k, v in systems.items()}
    sources["note"] = "the LLM stage's dev value (.556 on all 1,218) is in EVIDENCE; it is not recomputed here"
    for name, path in systems.items():
        S = np.asarray(np.load(path, mmap_mode="r")[np.asarray(rows)], dtype=np.float64)
        res = ep.evaluate(S, rows, "P_spec", top_k=10)
        vals = {"ndcg_at_10": {}, "hit_at_10": {}, "law_hit_at_10": {}}
        for r, q in zip(res["rows"], res["per_query"]):
            gold_law = {inst_code(a) for a in q["positive_idx"]}
            vals["ndcg_at_10"][int(r)] = q["ndcg_at_10"]
            vals["hit_at_10"][int(r)] = float(any(k <= 10 for k in q["positive_ranks"]))
            vals["law_hit_at_10"][int(r)] = float(any(inst_code(a) in gold_law for a in q["topk"][:10]))
        for key, v in vals.items():
            table.setdefault(name, {})[key], per.setdefault(name, {})[key] = _macro(v, comp)
        del S
    contrasts = {f"LLM alone minus {n}": {k: _paired(per["LLM alone"][k], per[n][k])
                                          for k in ("ndcg_at_10", "hit_at_10")} for n in systems}
    return {"label": "dev, descriptive (pre-cutoff cases: optimistic for the LLM)", "decision": "D55",
            "rows": len(rows), "components": len(set(comp.values())), "model": luna.MODEL, "effort": EFFORT,
            "batch": BATCH, "table": table, "contrasts": contrasts, "sources": sources,
            "bootstrap": {"draws": DRAWS, "seed": SEED, "unit": "component"}}


def _save(outputs: dict[int, dict], name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps({str(k): v for k, v in outputs.items()}, ensure_ascii=False, indent=0),
                            encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke", "run", "score"])
    a = ap.parse_args()
    rows = dev_eval_rows()
    if a.command == "smoke":
        sample = rows[:BATCH]
        out = run(sample)
        _save(out, "smoke_outputs.json")
        pq = per_query(sample, out)
        print(json.dumps({"rows": len(sample), "mean_ndcg": float(np.mean([v["ndcg_at_10"] for v in pq.values()])),
                          "mean_kept": float(np.mean([v["kept"] for v in pq.values()])),
                          "example": out[sample[0]]["articulos"][:3]}, ensure_ascii=False, indent=1))
        return 0
    if a.command == "run":
        C.ledger({"kind": "job_start", "job_id": "llm_alone_dev", "command": " ".join(sys.argv), "decision": "D55"})
        try:
            out = run(rows)
        except luna.LunaQuotaError as exc:
            C.ledger({"kind": "job_end", "job_id": "llm_alone_dev", "status": "paused_quota", "detail": str(exc)[:300]})
            print("quota: re-run later; cached calls are reused")
            return 3
        _save(out, "outputs.json")
        C.ledger({"kind": "job_end", "job_id": "llm_alone_dev", "status": "ok", "rows": len(rows)})
    saved = json.loads((OUT / "outputs.json").read_text(encoding="utf-8"))
    outputs = {int(k): v for k, v in saved.items()}
    res = score(rows, outputs)
    (OUT / "SCORE.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k in ("rows", "components", "table", "contrasts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
