"""L1: Luna listwise reranking of the LTR_emb_hgb top 30 with permutation self-consistency (PLAN §3 L1; effort medium).

Per query and pass, two windows of 20 run back to front (stride 10): window A = LTR ranks 11-30; its top 10 join ranks
1-10 in window B. The final pass order is B's ranking (positions 1-20) followed by A's other 10 (21-30). Pass "fwd"
shows candidates in LTR order; pass "rev" shows each window in reversed order. Five queries per call; candidates carry
opaque numbers 1-20; each shows "Ley › Art. N" and an excerpt of at most 600 characters (~150 tokens). Outputs must be
permutations; missing numbers are appended in input order and counted as repairs.

Arm score per (query, candidate in top 30): 1 / central rank, where the central rank is the mean of the two passes.
Also stored: cross-pass agreement and the number of passes placing the candidate in the top 5.
Controls (300 dev rows): ``shuffled`` (random input order, one pass) and ``blank`` (excerpts removed).

    python -m uol.round3.l1_luna run [--rows pspec|control] [--variant main|shuffled|blank] [--workers 8]
    python -m uol.round3.l1_luna evaluate
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

from uol.round3 import common as C
from uol.round3 import features, luna, prompt_guard

PROMPT = Path(__file__).resolve().parent / "prompts" / "L1.txt"
OUT = C.ARMS / "L1"
EFFORT = "medium"
TOP, WIN, PER_CALL = 30, 20, 5
EXCERPT = 600
ITEM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["orden"],
               "properties": {"orden": {"type": "array", "items": {"type": "integer"}}}}


def candidate_line(i: int, blank: bool) -> str:
    d = C.release()
    title, _, body = d.candidate_texts[i].partition("\n")
    label = d.candidate_ids[i].split("#articulo=")[-1]
    head = f"{title.strip()} › Art. {label}"
    if blank:
        return head
    body = re.sub(r"\s+", " ", body).strip()
    return f"{head}: {body[:EXCERPT]}{'…' if len(body) > EXCERPT else ''}"


def window_item(qid: str, row: int, cands: list[int], blank: bool) -> tuple[str, list[str]]:
    d = C.release()
    lines = [candidate_line(c, blank) for c in cands]
    text = (f"[id={qid}]\nHechos: {re.sub(r'\s+', ' ', d.queries[row].text).strip()}\nCandidatos:\n"
            + "\n".join(f"{n}. {ln}" for n, ln in enumerate(lines, 1)))
    return text, lines


def _build(part, blank, tpl):
    texts, sanctioned = [], []
    for qid, row, cands in part:
        t, lines = window_item(qid, row, cands, blank)
        texts.append(t)
        sanctioned += lines
    prompt = tpl.replace("{items}", (chr(10) * 2).join(texts))
    prompt_guard.check(prompt, [r for _, r, _ in part], sanctioned=sanctioned, template=tpl)
    return prompt


def ask(items: list[tuple[str, int, list[int]]], blank: bool, workers: int, tag: str, sample: int = 0) -> dict[str, list[int]]:
    """items: (qid, row, candidates in display order). Returns qid -> candidates in Luna's order (repaired).

    Batches of PER_CALL items; items a batch response leaves out are re-asked one per call (D5.1).
    """
    tpl = PROMPT.read_text(encoding="utf-8")
    schema = luna.batch_schema(ITEM_SCHEMA)
    parts = [items[s:s + PER_CALL] for s in range(0, len(items), PER_CALL)]
    res = luna.run_many([(_build(p, blank, tpl), schema) for p in parts], effort=EFFORT, tag=tag, workers=workers,
                        sample=sample)
    got: dict[str, dict] = {}
    missing: list[tuple[str, int, list[int]]] = []
    for part, r in zip(parts, res):
        rec = {x["id"]: x for x in r["output"].get("results", []) if isinstance(x, dict) and "id" in x}
        for it in part:
            (got.__setitem__(it[0], rec[it[0]]) if it[0] in rec else missing.append(it))
    for attempt in range(3):
        if not missing:
            break
        single = luna.run_many([(_build([it], blank, tpl), schema) for it in missing], effort=EFFORT, tag=tag,
                               workers=workers, sample=10 * sample + attempt)
        still = []
        for it, r in zip(missing, single):
            rec = {x["id"]: x for x in r["output"].get("results", []) if isinstance(x, dict) and "id" in x}
            (got.__setitem__(it[0], rec[it[0]]) if it[0] in rec else still.append(it))
        missing = still
    if missing:
        raise luna.LunaError(f"{len(missing)} L1 items unanswered after single-item retries: {[m[0] for m in missing][:5]}")
    out, repairs = {}, 0
    for qid, _, cands in items:
        order, seen = [], set()
        for n in got[qid]["orden"]:
            if isinstance(n, int) and 1 <= n <= len(cands) and n not in seen:
                order.append(n)
                seen.add(n)
        miss = [n for n in range(1, len(cands) + 1) if n not in seen]
        repairs += bool(miss)
        out[qid] = [cands[n - 1] for n in order + miss]
    out["_repairs"] = repairs  # type: ignore[assignment]
    return out


def run(rows: list[int], variant: str, workers: int, tag: str, sample: int = 0, top: np.ndarray | None = None,
        out_name: str | None = None) -> dict:
    """``top`` overrides the LTR_emb_hgb top 30 (e.g. the C_hybrid Stage-2 ranking); ``out_name`` the orders file."""
    if np.isin(rows, C.lockbox_rows()).any():
        raise PermissionError("L1 runs on dev rows only before Phase 7")
    top = features.ltr_topk("LTR_emb_hgb", rows, TOP) if top is None else np.asarray(top)
    blank = variant == "blank"
    passes = ("shuf",) if variant == "shuffled" else ("fwd", "rev")
    rng = np.random.default_rng(C.SEED)
    final: dict[str, dict[int, list[int]]] = {}
    repairs = 0
    for p in passes:
        disp = {}
        for r, t in zip(rows, top):
            t = [int(x) for x in t]
            disp[r] = list(rng.permutation(t)) if p == "shuf" else t
        def show(lst):
            return lst[::-1] if p == "rev" else lst
        a_items = [(f"{p}A{r}", r, show(disp[r][10:30])) for r in rows]
        a = ask(a_items, blank, workers, f"{tag}_{p}A", sample)
        repairs += a.pop("_repairs")
        b_items = [(f"{p}B{r}", r, show(disp[r][:10] + a[f"{p}A{r}"][:10])) for r in rows]
        b = ask(b_items, blank, workers, f"{tag}_{p}B", sample)
        repairs += b.pop("_repairs")
        final[p] = {r: b[f"{p}B{r}"] + a[f"{p}A{r}"][10:] for r in rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = (OUT / out_name) if out_name else OUT / (f"orders_{variant}.json" if not sample else f"orders_{variant}_sample{sample}.json")
    C.write_json(path, {"variant": variant, "rows": rows, "passes": list(passes), "repairs": repairs,
                        "prompt_sha256": hashlib.sha256(PROMPT.read_bytes()).hexdigest(),
                        "orders": {p: {str(r): o for r, o in v.items()} for p, v in final.items()}})
    return {"path": str(path), "repairs": repairs}


def score_matrices(variant: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """config -> (rows, A [rows, N]) from saved orders: rr = 1/central rank; rr_agree = rr x cross-pass agreement;
    top5 = passes placing the candidate in the top 5 (+ rr/100 as tie-break)."""
    o = json.loads((OUT / f"orders_{variant}.json").read_text(encoding="utf-8"))
    rows = np.asarray(o["rows"])
    n = len(C.release().candidate_ids)
    mats = {k: np.full((len(rows), n), np.nan, dtype=np.float32) for k in ("rr", "rr_agree", "top5")}
    for i, r in enumerate(rows):
        ranks = {}
        for p in o["passes"]:
            for pos, c in enumerate(o["orders"][p][str(r)], 1):
                ranks.setdefault(int(c), []).append(pos)
        for c, rk in ranks.items():
            central = float(np.mean(rk))
            agree = 1.0 / (1.0 + (max(rk) - min(rk)) / 10.0)
            mats["rr"][i, c] = 1.0 / central
            mats["rr_agree"][i, c] = agree / central
            mats["top5"][i, c] = sum(x <= 5 for x in rk) + 0.01 / central
    return {k: (rows, v) for k, v in mats.items()}


def evaluate() -> dict:
    from uol.round3 import arm_runner
    main = score_matrices("main")
    controls = {}
    for v in ("shuffled", "blank"):
        if (OUT / f"orders_{v}.json").exists():
            controls[f"{v}_rr"] = score_matrices(v)["rr"]
    ctrl_rows = set(control_rows())
    rows = main["rr"][0]
    sub = np.asarray([int(r) in ctrl_rows for r in rows])
    controls["main_rr_on_control_rows"] = (rows[sub], main["rr"][1][sub])
    led = [json.loads(x) for x in (C.STATE / "LUNA_LEDGER.jsonl").read_text(encoding="utf-8").splitlines() if x]
    l1 = [x for x in led if x.get("ok") and str(x.get("tag", "")).startswith("L1")]
    return arm_runner.run_fusion(
        "L1", main, controls=controls,
        subgroups={"low_overlap": C.low_overlap,
                   "small_top1_margin": lambda rr: _small_margin(rr)},
        notes="Luna listwise reranking of the LTR top 30 (window 20, stride 10, two passes; effort medium)",
        extra_result={"luna": {"calls": len(l1), "input_tokens": sum(x["usage"].get("input_tokens", 0) for x in l1),
                               "output_tokens": sum(x["usage"].get("output_tokens", 0) for x in l1)},
                      "luna_resample_agreement": "pending (100-row sample=1 re-run)",
                      "repairs": json.loads((OUT / "orders_main.json").read_text(encoding="utf-8"))["repairs"]})


def _small_margin(rows) -> np.ndarray:
    """Pre-registered subgroup: LTR top-1 minus top-2 score margin in the bottom tercile of dev P_spec rows."""
    M = features.frozen_matrix("LTR_emb_hgb")
    allp = C.population_rows("P_spec", C.dev_rows())
    def margin(r):
        s = np.sort(np.asarray(M[int(r)], dtype=np.float64))[-2:]
        return s[1] - s[0]
    cut = np.quantile([margin(r) for r in allp], 1 / 3)
    return np.asarray([margin(r) <= cut for r in rows])


def resample_agreement(n: int = 100) -> dict:
    """PLAN L rules: re-run 100 dev rows with sample=1 and compare with the cached sample-0 orders."""
    from scipy.stats import spearmanr
    rows = control_rows()[:n]
    run(rows, "main", 8, "L1_resample", sample=1)
    o0 = json.loads((OUT / "orders_main.json").read_text(encoding="utf-8"))["orders"]
    o1 = json.loads((OUT / "orders_main_sample1.json").read_text(encoding="utf-8"))["orders"]
    rho, top5, top1 = [], [], []
    for r in rows:
        def central(o):
            rk = {}
            for p in o:
                for pos, c in enumerate(o[p][str(r)], 1):
                    rk.setdefault(c, []).append(pos)
            return {c: float(np.mean(v)) for c, v in rk.items()}
        a, b = central(o0), central(o1)
        cs = sorted(a)
        rho.append(spearmanr([a[c] for c in cs], [b[c] for c in cs])[0])
        ta = set(sorted(cs, key=lambda c: a[c])[:5])
        tb = set(sorted(cs, key=lambda c: b[c])[:5])
        top5.append(len(ta & tb) / 5)
        top1.append(min(cs, key=lambda c: a[c]) == min(cs, key=lambda c: b[c]))
    res = {"rows": n, "mean_spearman_central_rank": float(np.nanmean(rho)), "mean_top5_overlap": float(np.mean(top5)),
           "top1_agreement": float(np.mean(top1))}
    C.write_json(OUT / "resample_agreement.json", res)
    return res


def control_rows(n: int = 300) -> list[int]:
    rng = np.random.default_rng(C.SEED + 1)
    return sorted(rng.choice(C.population_rows("P_spec", C.dev_rows()), n, replace=False).tolist())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["run", "evaluate", "resample"])
    ap.add_argument("--rows", default="pspec", choices=["pspec", "control"])
    ap.add_argument("--variant", default="main", choices=["main", "shuffled", "blank"])
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.command == "resample":
        print(resample_agreement())
        return 0
    if a.command == "evaluate":
        r = evaluate()
        print(json.dumps(r["dev"]["vs_EMB"], indent=1, default=float)[:1500])
        return 0
    rows = [int(r) for r in (C.population_rows("P_spec", C.dev_rows()) if a.rows == "pspec" else control_rows())]
    job = f"L1_{a.rows}_{a.variant}"
    start = C.utc()
    C.ledger({"kind": "job_start", "job_id": job, "command": " ".join(sys.argv)})
    try:
        res = run(rows, a.variant, a.workers, "L1" if a.variant == "main" else f"L1_{a.variant}")
    except luna.LunaQuotaError as exc:
        C.ledger({"kind": "job_end", "job_id": job, "start_utc": start, "end_utc": C.utc(), "exit_code": 3,
                  "status": "paused_quota", "detail": str(exc)[:300]})
        print("paused_quota", exc)
        return 3
    C.ledger({"kind": "job_end", "job_id": job, "start_utc": start, "end_utc": C.utc(), "exit_code": 0, **res})
    print(res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
