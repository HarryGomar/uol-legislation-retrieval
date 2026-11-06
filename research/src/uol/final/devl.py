"""DEV-L (C6, D93; ANALYSIS_PLAN_C §6a): the development facts told in everyday words, a register-shift check of the
final systems without retraining (the requirement that lay wording still finds the law, D92).

Step 1, ``rewrite`` (main profile, Codex): every DEV evaluation case's masked facts are rewritten in lay Spanish with
the round-2 lay prompt (``round2.glm_client.LAY_V1``, its instruction text verbatim, 20 cases per call, the masks kept)
and gated with the round-2 functions (``round2.h_common``, ``round2.glm_client.scrub_norms``): no law name or article
number (nothing for ``scrub_norms`` to remove), no term of the frozen legal-term lexicon (LAY_V1's own terms and
``EXTRA_TERMS``), and a length ratio between .5 and 2 against the original. Failed rows are dropped and counted. The
texts go to ``<work>/variants/devl/lay_texts.jsonl``, with 20 rows drawn for the author's check (seed 20261023).

Step 2, the ``devl`` profile variant: the passing rows carry their lay facts in the process (as ``devn`` does with its
clause-dropped facts); dense systems read their re-encoded ``devl__<key>`` query vectors; population DEV-L.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from uol.final import context, guard

PER_CALL = 20
LEN_RATIO = (0.5, 2.0)
SEED = 20261023
TAG = "final_devl"
ITEM = {"type": "object", "required": ["relato"], "properties": {"relato": {"type": "string"}}}
MASK = "[CITA_OCULTA]"


def texts_path(prof=None) -> Path:
    """``<work>/variants/devl/lay_texts.jsonl``; in the devl variant ``<work>`` already is that folder."""
    prof = prof or context.active()
    return prof.work / "lay_texts.jsonl" if prof.variant == "devl" else prof.work / "variants" / "devl" / "lay_texts.jsonl"


def template() -> str:
    """LAY_V1's instructions verbatim, for a batch of cases with ids, keeping the masks."""
    from uol.round2.glm_client import LAY_V1
    head = LAY_V1.split("\n\nHECHOS:")[0]
    return (head + f" Conserva cada marca {MASK} tal como aparece, sin sustituirla.\n\n"
            "Recibirás varios casos, cada uno con su identificador. Para cada caso devuelve un objeto con su id y su "
            "relato.\n\n{items}")


def gate(lay: str, orig: str) -> dict[str, Any]:
    from uol.round2 import h_common as H
    from uol.round2.glm_client import scrub_norms
    text = H.strip_label(lay)
    _, removed = scrub_norms(text)
    hits = H.banned_hits(text)
    ratio = len(text) / max(len(orig), 1)
    fails = ([] if removed == 0 else ["norm mention"]) + ([] if sum(hits.values()) == 0 else ["legal term"]) + \
        ([] if LEN_RATIO[0] <= ratio <= LEN_RATIO[1] else ["length ratio"])
    return {"text": text, "pass": not fails, "fails": fails, "norm_removed": removed, "banned": hits,
            "len_ratio": round(ratio, 3)}


def _prompt(part: list[tuple[str, int]], tpl: str) -> str:
    from uol.embedding_push import common as ep
    from uol.round3 import prompt_guard
    d = ep.release()
    # the masked Hechos without carriage returns: that exact text is what the guard must treat as sanctioned (20 dev
    # Hechos carry CR; unsanctioned, their own rubro or court overlap refused 4 rows on RUN-B, 23 Oct)
    hechos = [re.sub(chr(13), "", d.queries[r].text).strip() for _, r in part]
    items = "\n\n".join(f"[id={qid}]\nHECHOS:\n{h}" for (qid, _), h in zip(part, hechos))
    prompt = tpl.replace("{items}", items)
    prompt_guard.check(prompt, [r for _, r in part], sanctioned=hechos, template=tpl)
    return prompt


def rewrite(rows: list[int] | None = None, workers: int = 6) -> dict[str, Any]:
    from uol.embedding_push import common as ep
    from uol.final import codex, llm_jobs
    from uol.round3 import luna
    codex.install()
    d = ep.release()
    rows = [int(r) for r in (llm_jobs.dev_eval_rows() if rows is None else guard.dev_rows_only(rows, "devl rows"))]
    items = [(f"L{r}", r) for r in rows]
    tpl = template()
    schema = luna.batch_schema(ITEM)
    parts = [items[s:s + PER_CALL] for s in range(0, len(items), PER_CALL)]
    res = luna.run_many([(_prompt(p, tpl), schema) for p in parts], effort="low", tag=TAG, workers=workers)
    got: dict[str, str] = {}
    missing = []
    for part, r in zip(parts, res):
        rec = {x["id"]: x for x in r["output"].get("results", []) if isinstance(x, dict) and "id" in x}
        for it in part:
            (got.__setitem__(it[0], rec[it[0]]["relato"]) if it[0] in rec else missing.append(it))
    for attempt in range(3):                                         # left-out cases are asked alone
        if not missing:
            break
        single = luna.run_many([(_prompt([it], tpl), schema) for it in missing], effort="low", tag=TAG, workers=workers,
                               sample=attempt)
        still = []
        for it, r in zip(missing, single):
            rec = {x["id"]: x for x in r["output"].get("results", []) if isinstance(x, dict) and "id" in x}
            (got.__setitem__(it[0], rec[it[0]]["relato"]) if it[0] in rec else still.append(it))
        missing = still
    out = []
    for qid, r in items:
        rec = {"row": r, "query_id": d.queries[r].query_id}
        if qid in got:
            rec.update(gate(got[qid], d.queries[r].text))
        else:
            rec.update({"text": None, "pass": False, "fails": ["no answer"]})
        out.append(rec)
    path = texts_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out), encoding="utf-8")
    tmp.replace(path)
    summary = {"rows": len(out), "passed": sum(x["pass"] for x in out),
               "failed_by_gate": {k: sum(k in x["fails"] for x in out) for k in ("norm mention", "legal term",
                                                                                  "length ratio", "no answer")},
               "len_ratio": list(LEN_RATIO), "per_call": PER_CALL, "tag": TAG}
    (path.parent / "lay_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    write_check_sheet(out, d, path.parent / "lay_check_20.csv")
    return summary


def write_check_sheet(out: list[dict], d, path: Path, n: int = 20) -> None:
    """20 passing rewrites for the author (seed 20261023): original, lay text, facts kept?, legal knowledge added?"""
    ok = [x for x in out if x["pass"]]
    pick = np.random.default_rng(SEED).choice(len(ok), min(n, len(ok)), replace=False) if ok else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "original", "lay", "facts kept (y/n)", "legal knowledge added (y/n)", "notes"])
        for i in sorted(int(p) for p in pick):
            x = ok[i]
            w.writerow([x["query_id"], d.queries[x["row"]].text, x["text"], "", "", ""])


def lay_texts(prof=None) -> dict[int, str]:
    path = texts_path(prof)
    if not path.exists():
        raise FileNotFoundError(f"DEV-L texts not written yet ({path}); run the devl rewrite job first")
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        x = json.loads(line)
        if x.get("pass"):
            out[int(x["row"])] = x["text"]
    rows = guard.dev_rows_only(sorted(out), "DEV-L rows")
    return {int(r): out[int(r)] for r in rows}


def rows() -> list[int]:
    return sorted(lay_texts())


def apply_text_override() -> dict[str, int]:
    """Profile variant ``devl``: the passing rows carry their lay facts in this process."""
    from uol.embedding_push import common as ep
    d = ep.release()
    if getattr(d, "_uol_final_devl", False):
        return {"already": 1}
    n = 0
    for r, t in lay_texts().items():
        d.queries[r].text = t
        n += 1
    d._uol_final_devl = True
    return {"rows": n}
