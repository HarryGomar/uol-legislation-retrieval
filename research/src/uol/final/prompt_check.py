"""Does the model alone depend on how it is asked? Lists of 10 and 40 and one case per call, on the sample (the author, 4 Nov 2026).

Plan and reading rule, written before any answer: research/programs/prompt_check_20261104/PLAN.md. The reference is the stored
model-alone run (20 articles, 10 cases per call); the test set is not read, and the system stays frozen whatever this shows.

    py -3.12 -m uol.final.prompt_check run --variant list10 [--smoke 10] --go
    py -3.12 -m uol.final.prompt_check score
    py -3.12 -m uol.final.prompt_check run --variant list100 --go; py -3.12 -m uol.final.prompt_check score_extra
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A, audit_d181_dev as D, llm_agents as L

VARIANTS = {"list10": (10, 10), "list40": (40, 10), "one_call": (20, 1)}   # articles asked for, cases per call
EXTRA = {"list100": (100, 10)}                # a family of its own (PLAN.md, addendum), written after the first family was seen
REFERENCE = "alone"
BANDS = ((1, 10), (11, 20), (21, 40), (41, 70), (71, 100))
OUT = A.ROOT / "research" / "programs" / "prompt_check_20261104"
TAG = "prompt_check_20261104"


def template(n: int) -> str:
    from uol.eval_lenses import llm_alone as LA
    tpl = LA.TEMPLATE.read_text(encoding="utf-8")
    if tpl.count("exactamente 20") != 1:
        raise RuntimeError("the instruction has changed")
    return tpl.replace("exactamente 20", f"exactamente {n}")


def build(rows: list[int], n: int) -> tuple[str, list[str]]:
    """As ``llm_alone.build``, with the number of articles asked for; development rows only, every prompt through the guard."""
    from uol.embedding_push import common as ep
    from uol.round3 import common as C, prompt_guard
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("development rows only")
    tpl, d = template(n), C.release()
    ids = [f"c{i + 1:02d}" for i in range(len(rows))]
    items = "\n\n".join(f"[id={i}] {re.sub(r'\s+', ' ', d.queries[r].text).strip()}" for i, r in zip(ids, rows))
    prompt = tpl.replace("{items}", items)
    prompt_guard.check(prompt, rows, template=tpl)
    return prompt, ids


def item_schema(n: int) -> dict:
    from uol.eval_lenses import llm_alone as LA
    from uol.round3 import luna
    art = LA.ITEM["properties"]["articulos"]
    return luna.batch_schema({**LA.ITEM, "properties": {"articulos": {**art, "minItems": n, "maxItems": n}}})


def check_reference(rows: list[int]) -> None:
    """The reference prompt built here must be the one the stored run sent: its cache key is already held."""
    from uol.round3 import luna
    prompt, _ = build(rows[:10], 20)
    if luna.cached(luna.key_of(prompt, item_schema(20), L.MODEL, "medium", 0)) is None:
        raise RuntimeError("this prompt is not the one the stored model-alone run sent")


def run(variant: str, smoke: int | None) -> int:
    n, per_call = {**VARIANTS, **EXTRA}[variant]
    luna = L.pin_model()
    rows = L.sample_rows(smoke)
    check_reference(L.sample_rows(None))
    schema = item_schema(n)
    batches = [rows[i:i + per_call] for i in range(0, len(rows), per_call)]
    built = [build(b, n) for b in batches]
    name = variant + (f"_smoke{smoke}" if smoke else "")
    t0 = time.perf_counter()
    res = luna.run_many([(p, schema) for p, _ in built], effort="medium", tag=f"{TAG}_{name}", workers=4)
    answers, usage = {}, {"calls": 0, "input_tokens": 0, "output_tokens": 0}
    for (prompt, ids), part, r in zip(built, batches, res):
        got = luna.checked(r, prompt, schema, ids, effort="medium", tag=f"{TAG}_{name}")
        for i, row in zip(ids, part):
            answers[row] = got[i]["articulos"]
        usage["calls"] += 1
        for k in ("input_tokens", "output_tokens"):
            usage[k] += int((r.get("usage") or {}).get(k) or 0)
    out = L.OUT / f"check_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    A.write_json(out / "run.json", {"variant": variant, "asked_for": n, "cases_per_call": per_call, "cases": len(rows), "model": L.MODEL,
                                    "effort": "medium", "usage": usage, "seconds": round(time.perf_counter() - t0, 1)})
    print(json.dumps({"variant": name, "cases": len(answers), **usage, "seconds": round(time.perf_counter() - t0, 1)}), flush=True)
    return 0


def describe(answers: dict, rows: list[int], generic: set[int], specific, first_ref: dict[int, list[int]] | None) -> dict:
    lists = {r: L.resolved(answers[str(r)], generic) for r in rows}
    gold = sum(len(specific[r]) for r in rows)
    out = {"kept_median": float(np.median([len(v) for v in lists.values()])),
           "cited_found_first_10": sum(len({int(x) for x in specific[r]} & set(lists[r][:10])) for r in rows) / gold,
           "cited_found_in_list": sum(len({int(x) for x in specific[r]} & set(lists[r])) for r in rows) / gold}
    if first_ref:
        out["first_10_overlap_with_reference"] = float(np.mean([len(set(lists[r][:10]) & set(first_ref[r])) / max(len(first_ref[r]), 1) for r in rows]))
    return {"lists": lists, "facts": out}


def score() -> int:
    from uol.embedding_push import common as ep
    import pandas as pd
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    ref_answers = json.loads((L.OUT / REFERENCE / "answers.json").read_text(encoding="utf-8"))
    rows = sorted(int(r) for r in ref_answers)
    ref_frame = pd.read_parquet(L.OUT / REFERENCE / "per_case.parquet")
    ref = describe({k: v["articulos"] for k, v in ref_answers.items()}, rows, generic, d.specific_idx, None)
    mine = L.score_ranked(rows, ref["lists"])
    if abs(float(mine["ndcg_at_10"].mean()) - float(ref_frame["ndcg_at_10"].mean())) > 1e-9:
        raise RuntimeError("the scorer does not reproduce the stored reference")
    first_ref = {r: ref["lists"][r][:10] for r in rows}
    frames, facts = {}, {REFERENCE: {"ndcg_at_10_per_case": float(ref_frame["ndcg_at_10"].mean()), **ref["facts"]}}
    for v in VARIANTS:
        out = L.OUT / f"check_{v}"
        answers = json.loads((out / "answers.json").read_text(encoding="utf-8"))
        got = describe(answers, rows, generic, d.specific_idx, first_ref)
        frames[v] = L.score_ranked(rows, got["lists"])
        frames[v].to_parquet(out / "per_case.parquet")
        run_rec = json.loads((out / "run.json").read_text(encoding="utf-8"))
        facts[v] = {"ndcg_at_10_per_case": float(frames[v]["ndcg_at_10"].mean()), **got["facts"], **run_rec,
                    "input_tokens_per_case": run_rec["usage"]["input_tokens"] / len(rows), "output_tokens_per_case": run_rec["usage"]["output_tokens"] / len(rows)}
    fam = A.dev_family("prompt_check", [(v, frames[v], REFERENCE, ref_frame) for v in VARIANTS], population="DEV-A300")
    result = {"plan": "research/programs/prompt_check_20261104/PLAN.md", "facts": facts, "contrasts": fam["contrasts"], "family": fam["family"], "size": fam["size"]}
    A.write_json(OUT / "results.json", result)
    A.receipt("prompt_check", {"cases": len(rows), "means": {k: v["ndcg_at_10_per_case"] for k, v in facts.items()}})
    for k, c in fam["contrasts"].items():
        print(k, round(c["delta"], 4), [round(x, 3) for x in c["ci95"]], round(c["adjusted_lower"], 4), c["class"], flush=True)
    print(json.dumps(facts, indent=1, default=float), flush=True)
    return 0


def named_kinds(answers: dict, rows: list[int], generic: set[int], specific) -> dict:
    """Each named article by position in the answer as the model gave it: no such article, law not in the catalogue, generic,
    repeat, or exists; of those that exist, how many the court cited (PLAN.md, addendum)."""
    from collections import Counter
    from uol.round3 import catalog_map
    bands = {f"{lo}-{hi}": Counter() for lo, hi in BANDS}
    laws, examples = Counter(), {}
    for r in rows:
        seen, gold = set(), {int(x) for x in specific[r]}
        for k, it in enumerate(answers[str(r)], 1):
            band = next((f"{lo}-{hi}" for lo, hi in BANDS if lo <= k <= hi), None)
            if band is None:
                continue
            idx, why = catalog_map.resolve(it["ley"], it["articulo"])
            c = bands[band]
            c["named"] += 1
            if why == "instrument_unresolved":
                c["law_not_in_catalogue"] += 1
                laws[it["ley"].strip()] += 1
            elif idx is None:
                c["no_such_article"] += 1
                examples.setdefault(band, []).append(f"{it['ley'].strip()}, {it['articulo']}")
            elif int(idx) in seen:
                c["repeat"] += 1
            elif int(idx) in generic:
                c["generic"] += 1
            else:
                c["exists"] += 1
                c["cited"] += int(int(idx) in gold)
            if idx is not None:
                seen.add(int(idx))
    return {"bands": {k: dict(v) for k, v in bands.items() if v["named"]}, "laws_not_in_catalogue": laws.most_common(25),
            "no_such_article_examples": {k: v[:: max(1, len(v) // 12)][:12] for k, v in examples.items()}}


def score_extra() -> int:
    from uol.embedding_push import common as ep
    import pandas as pd
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    ref_answers = json.loads((L.OUT / REFERENCE / "answers.json").read_text(encoding="utf-8"))
    rows = sorted(int(r) for r in ref_answers)
    ref_frame = pd.read_parquet(L.OUT / REFERENCE / "per_case.parquet")
    answers = {"list20": {k: v["articulos"] for k, v in ref_answers.items()}}
    for v in ("list40", "list100"):
        answers[v] = json.loads((L.OUT / f"check_{v}" / "answers.json").read_text(encoding="utf-8"))
    ref = describe(answers["list20"], rows, generic, d.specific_idx, None)
    first_ref = {r: ref["lists"][r][:10] for r in rows}
    got = describe(answers["list100"], rows, generic, d.specific_idx, first_ref)
    frame = L.score_ranked(rows, got["lists"])
    frame.to_parquet(L.OUT / "check_list100" / "per_case.parquet")
    fam = A.dev_family("prompt_check_list100", [("list100", frame, REFERENCE, ref_frame)], population="DEV-A300")
    run_rec = json.loads((L.OUT / "check_list100" / "run.json").read_text(encoding="utf-8"))
    result = {"plan": "research/programs/prompt_check_20261104/PLAN.md, addendum", "ndcg_at_10_per_case": float(frame["ndcg_at_10"].mean()),
              "reference": float(ref_frame["ndcg_at_10"].mean()), **got["facts"], "run": run_rec, "contrast": fam["contrasts"],
              "kinds": {v: named_kinds(answers[v], rows, generic, d.specific_idx) for v in answers}}
    A.write_json(OUT / "results_list100.json", result)
    A.receipt("prompt_check_list100", {"cases": len(rows), "ndcg_at_10_per_case": result["ndcg_at_10_per_case"]})
    print(json.dumps({k: v for k, v in result.items() if k != "kinds"}, indent=1, default=float), flush=True)
    for v, k in result["kinds"].items():
        for band, c in k["bands"].items():
            ex = c.get("exists", 0)
            print(v, band, "named", c["named"], "no_such", round(c.get("no_such_article", 0) / c["named"], 3), "law_out", round(c.get("law_not_in_catalogue", 0) / c["named"], 3),
                  "generic", round(c.get("generic", 0) / c["named"], 3), "repeat", round(c.get("repeat", 0) / c["named"], 3), "exists", round(ex / c["named"], 3),
                  "cited_of_exists", round(c.get("cited", 0) / max(ex, 1), 3), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score", "score_extra"])
    ap.add_argument("--variant", choices=[*VARIANTS, *EXTRA])
    ap.add_argument("--smoke", type=int, default=None)
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    os.environ.setdefault("UOL_LUNA_CACHE", str(A.ROOT / ".cache" / "footing" / "codex_noweb.sqlite"))
    D.activate(FT.STORE)
    if a.step == "score":
        return score()
    if a.step == "score_extra":
        return score_extra()
    if not a.go:
        n, per_call = {**VARIANTS, **EXTRA}[a.variant]
        print(f"[dry] would ask {L.MODEL} for {n} articles, {per_call} cases per call, on {a.smoke or 300} development cases")
        return 0
    return run(a.variant, a.smoke)


if __name__ == "__main__":
    raise SystemExit(main())
