"""B step 2: resolve article mentions in historical tesis to the v11 catalogue (39,178 articles, 264 instruments).

Resolver: the facts-to-law v2 rule resolver (``uol.facts_to_law.references_v2``) run on census text only. Its alias
table is conservative: official instrument titles, the catalogue's declared ``safe_aliases``, a reviewed acronym table
and the CPEUM/Ley de Amparo rules; nothing is learned and no label or citation graph is read. Each historical tesis is
one pseudo-section ``rubro + "\\n" + texto`` (field ``criterio_juridico`` so every rule applies). Each positive keeps the
raw article span and the witness span (the law-name binding), which the context builder masks.

Run: ``py -3.12 -m uol.round2.b_resolve [--workers 8]``
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from uol.facts_to_law import references_v2 as R
from uol.facts_to_law.spec import sha256_file
from uol.round2.b_common import CANDIDATES, HIST, MENTIONS, RESOLVE_META, log, utc, write_json

_CAT = None


def compose(row: dict) -> str:
    return str(row["rubro"]).strip() + "\n" + str(row["texto"])


def catalogue() -> R.Catalogue:
    global _CAT
    if _CAT is None:
        load = lambda p: [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]
        _CAT = R.build_catalogue(load(CANDIDATES / "instruments.jsonl"), load(CANDIDATES / "articles.jsonl"))
    return _CAT


def resolve_one(row: dict) -> dict:
    text = compose(row)
    src = {"query_id": "hist:" + row["record_id"], "raw_published_text": text, "source_content_sha256": R.digest(text),
           "source_materias": row.get("materias"),
           "sections": [{"field": "criterio_juridico", "start": 0, "end": len(text), "text": text}]}
    try:
        res = R.resolve_source(src, catalogue())
    except Exception as exc:  # a resolver failure on odd text is recorded, not fatal
        return {"record_id": row["record_id"], "error": f"{type(exc).__name__}: {str(exc)[:120]}", "positives": [], "n_missing": 0}
    pos = [{"cid": p["candidate_id"], "rules": p["rule_ids"], "span": p["source_span"], "witness": p["witness"]}
           for p in res.positives]
    return {"record_id": row["record_id"], "positives": pos, "n_missing": len(res.missing),
            "n_out_of_scope": sum(u["disposition"] == R.OUT_OF_SCOPE for u in res.unresolved)}


def _chunk(lines: list[str]) -> list[dict]:
    return [resolve_one(json.loads(line)) for line in lines]


def main(workers: int = 8) -> dict:
    t0 = time.perf_counter()
    with HIST.open(encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]
    chunks = [lines[i:i + 2000] for i in range(0, len(lines), 2000)]
    out, rules, inst = [], Counter(), Counter()
    with ProcessPoolExecutor(max_workers=min(workers, 8)) as pool:
        for i, part in enumerate(pool.map(_chunk, chunks)):
            out.extend(part)
            if i % 20 == 0:
                log(f"resolve: {len(out)}/{len(lines)} ({time.perf_counter() - t0:.0f}s)")
    with MENTIONS.open("w", encoding="utf-8", newline="\n") as handle:
        for r in out:
            handle.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    n_with = sum(bool(r["positives"]) for r in out)
    mentions = sum(len(r["positives"]) for r in out)
    for r in out:
        for p in r["positives"]:
            rules["+".join(p["rules"])] += 1
            inst[p["cid"].split("#")[0].rsplit("/", 1)[-1]] += 1
    distinct = {p["cid"] for r in out for p in r["positives"]}
    meta = {"written_utc": utc(), "resolver": R.RESOLVER_VERSION, "resolver_code_sha256": R.CODE_SHA256,
            "tesis": len(out), "tesis_with_resolved_mention": n_with, "resolved_mentions": mentions,
            "distinct_articles": len(distinct), "errors": sum("error" in r for r in out),
            "tesis_with_missing_expression": sum(r["n_missing"] > 0 for r in out),
            "mentions_by_rule_combination": dict(rules.most_common()), "mentions_by_instrument": dict(inst.most_common()),
            "mentions_sha256": sha256_file(MENTIONS), "seconds": time.perf_counter() - t0}
    write_json(RESOLVE_META, meta)
    log(f"resolved: {n_with} tesis with mentions, {mentions} mentions, {len(distinct)} distinct articles ({meta['seconds']:.0f}s)")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args().workers)
