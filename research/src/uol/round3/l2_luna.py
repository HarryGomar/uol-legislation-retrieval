"""L2 Luna stage: legal-issue decomposition of masked Hechos (PLAN §3 L2; effort ``low``; about 20 rows per call).

Variant ``a``: concepts only. Variant ``b``: law names allowed. Article numbers are never allowed; any that appear in
an output are stripped and counted. Prompts are the fixed files ``prompts/L2_{a,b}.txt`` (hashed in every output
record). Rows inside a batch carry opaque ids (c01..c20), never release rows or registro digital. Dev rows only until
Phase 7.

    python -m uol.round3.l2_luna pilot              # Phase 0 step 8: 20 calls, variant a, 400 dev rows
    python -m uol.round3.l2_luna run --variant a     # all dev rows (Phase 5)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from uol.round3 import common as C
from uol.round3 import luna, prompt_guard

PROMPTS = Path(__file__).resolve().parent / "prompts"
BATCH = 20
EFFORT = "low"
ITEM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["issues"],
               "properties": {"issues": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string"}}}}
OUT = C.ARMS / "L2" / "luna"
LAW_NAME = re.compile(r"(?i)\b(ley|c[óo]digo|reglamento|constituci[óo]n|cpeum|lft|lss|cff|cnpp)\b")


def template(variant: str) -> tuple[str, str]:
    text = (PROMPTS / f"L2_{variant}.txt").read_text(encoding="utf-8")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_batch(rows: Sequence[int], variant: str) -> tuple[str, list[str], str]:
    tpl, sha = template(variant)
    d = C.release()
    ids = [f"c{i + 1:02d}" for i in range(len(rows))]
    items = "\n\n".join(f"[id={i}] {re.sub(r'\s+', ' ', d.queries[int(r)].text).strip()}" for i, r in zip(ids, rows))
    prompt = tpl.replace("{items}", items)
    prompt_guard.check(prompt, rows, template=tpl)
    return prompt, ids, sha


def clean(issues: list[str], variant: str) -> tuple[list[str], dict]:
    out, n_art, n_law = [], 0, 0
    for s in issues:
        if prompt_guard.ARTICLE_NO.search(s):
            n_art += 1
            s = prompt_guard.ARTICLE_NO.sub("", s)
        if variant == "a" and LAW_NAME.search(s):
            n_law += 1
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            out.append(s)
    return out, {"article_numbers_stripped": n_art, "law_names_in_variant_a": n_law}


def run(rows: Sequence[int], variant: str, tag: str, workers: int = 4, sample: int = 0) -> dict:
    rows = [int(r) for r in rows]
    if np.isin(rows, C.lockbox_rows()).any():
        raise PermissionError("L2 runs on dev rows only before Phase 7")
    batches = [rows[i:i + BATCH] for i in range(0, len(rows), BATCH)]
    built = [build_batch(b, variant) for b in batches]
    schema = luna.batch_schema(ITEM_SCHEMA)
    t0 = time.perf_counter()
    results = luna.run_many([(p, schema) for p, _, _ in built], effort=EFFORT, tag=tag, workers=workers, sample=sample)
    wall = time.perf_counter() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    recs, calls = [], []
    for (prompt, ids, sha), b, res in zip(built, batches, results):
        try:
            got = luna.check_batch(res, ids)
        except luna.LunaError:                # a dropped or extra id: request an independent sample of the same batch
            got = None
            for s in (1, 2, 3):
                res = luna.call(prompt, schema, effort=EFFORT, tag=tag, sample=10 * sample + s)
                try:
                    got = luna.check_batch(res, ids)
                    break
                except luna.LunaError:
                    continue
            if got is None:
                raise
        for i, r in zip(ids, b):
            issues, flags = clean(got[i]["issues"], variant)
            recs.append({"row": r, "variant": variant, "issues": issues, **flags, "prompt_sha256": sha,
                         "luna_key": luna.key_of(prompt, schema, luna.MODEL, EFFORT, 0)})
        calls.append({"n_items": len(b), "cached": res["cached"], "latency": res["latency"], "usage": res["usage"]})
    path = OUT / (f"{tag}_{variant}.jsonl" if not sample else f"{tag}_{variant}_sample{sample}.jsonl")
    with path.open("w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"records": path, "calls": calls, "wall_seconds": wall, "n_rows": len(rows)}


def resample_agreement(n: int = 100) -> dict:
    """PLAN L rules: re-run 100 dev rows (variant a) with sample=1; agreement = mean best-match token Jaccard between
    the two issue sets (both directions) and the share of rows with the same number of issues."""
    import re as _re
    rows = pilot_rows()[:n]
    run(rows, "a", "L2_resample", 4, sample=1)
    s0 = {json.loads(x)["row"]: json.loads(x)["issues"] for x in (OUT / "L2_a.jsonl").read_text(encoding="utf-8").splitlines() if x}
    s1 = {json.loads(x)["row"]: json.loads(x)["issues"] for x in (OUT / "L2_resample_a_sample1.jsonl").read_text(encoding="utf-8").splitlines() if x}

    def toks(t):
        return set(w for w in _re.findall(r"[a-záéíóúñü]+", t.lower()) if len(w) > 3)

    def best(A, B):
        return float(np.mean([max(len(toks(a) & toks(b)) / max(1, len(toks(a) | toks(b))) for b in B) for a in A]))
    jac = [(best(s0[r], s1[r]) + best(s1[r], s0[r])) / 2 for r in rows]
    res = {"rows": n, "mean_best_match_token_jaccard": float(np.mean(jac)),
           "same_issue_count": float(np.mean([len(s0[r]) == len(s1[r]) for r in rows]))}
    C.write_json(OUT / "resample_agreement.json", res)
    return res


def pilot_rows(n: int = 400) -> list[int]:
    rng = np.random.default_rng(C.SEED)
    return sorted(rng.choice(C.dev_rows(), n, replace=False).tolist())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["pilot", "run", "resample"])
    ap.add_argument("--variant", default="a", choices=["a", "b"])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    if a.command == "resample":
        print(resample_agreement())
        return 0
    rows = pilot_rows() if a.command == "pilot" else C.dev_rows().tolist()
    tag = "L2_pilot" if a.command == "pilot" else "L2"
    job = f"{tag}_{a.variant}"
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": job, "command": " ".join(sys.argv)})
    try:
        out = run(rows, a.variant, tag, a.workers)
    except luna.LunaQuotaError as exc:
        C.ledger({"kind": "job_end", "job_id": job, "start_utc": start, "end_utc": C.utc(), "exit_code": 3,
                  "status": "paused_quota", "detail": str(exc)[:300]})
        print("paused_quota:", exc)
        return 3
    calls = out["calls"]
    summary = {"rows": out["n_rows"], "calls": len(calls), "cached_calls": sum(c["cached"] for c in calls),
               "wall_seconds": round(out["wall_seconds"], 1),
               "input_tokens": sum(c["usage"].get("input_tokens", 0) for c in calls),
               "cached_input_tokens": sum(c["usage"].get("cached_input_tokens", 0) for c in calls),
               "output_tokens": sum(c["usage"].get("output_tokens", 0) for c in calls),
               "reasoning_output_tokens": sum(c["usage"].get("reasoning_output_tokens", 0) for c in calls),
               "latency_median": float(np.median([c["latency"] for c in calls])),
               "latency_max": float(np.max([c["latency"] for c in calls])), "records": str(out["records"])}
    C.write_json(OUT / f"{tag}_{a.variant}_summary.json", summary)
    C.ledger({"kind": "job_end", "job_id": job, "start_utc": start, "end_utc": C.utc(), "exit_code": 0,
              "luna": {k: summary[k] for k in ("calls", "input_tokens", "cached_input_tokens", "output_tokens")},
              "outputs": [{"path": str(out["records"]), "sha256": C.sha256_file(out["records"])}]})
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
