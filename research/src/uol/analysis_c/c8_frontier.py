"""C8: frontier robustness (ANALYSIS_PLAN_C §8; DEV, descriptive).

Cost per 1,000 cases from the recorded quantities (D80) in ``<store>/cost/<system>.json`` (``uol.final.cost.record``)
at cited, dated public prices supplied as a JSON file (``--prices``; none is recorded in the code base yet):

    {"cpu_vcpu_hour_usd": .., "gpu_hour_usd": .., "llm_input_per_mtok_usd": .., "llm_cached_input_per_mtok_usd": ..,
     "llm_output_per_mtok_usd": .., "sources": {...}, "date": ".."}

Per case: CPU = ``query_latency.median_ms`` x ``threads`` vCPU-milliseconds (threads default 8, the latency setting);
query encoding on a GPU device (``query_encoding.seconds_per_case`` with a CUDA/GPU device) = GPU-seconds, on any other
device = one vCPU; LLM = ``llm.per_case`` input, cached-input and output tokens (reasoning tokens are taken as part of
the output count and not added again). One-off costs (indexing, training) are not per-case costs and are left out.
A learned ranker's record excludes its precomputed sources (``query_latency.precomputed_sources``): the cost of each
listed source is added when ``src_<source>`` (or the member system) has a record, and the composition is marked
partial otherwise; an LLM stage adds the record of the system it reorders. A system without a usable record is not
placed.

Scenarios: CPU x {0.5, 1, 2}, GPU x {0.5, 1, 2}, LLM token price x {0.25, 1, 4} (27). A system is on the frontier when
no other placed system has a higher nDCG@10 (component-macro) at lower or equal cost. Reported: the share of the 27
scenarios in which each system is on the frontier; component attribution for the C2 leave-one-out pairs (nDCG@10
change per US$ change per 1,000 cases at base prices). LLM configurations from saved orders (passes x samples) need
the C5 orders, which are not in the runs folder, and are skipped with a note.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from uol.analysis_c import common as C

MULT_CPU = (0.5, 1.0, 2.0)
MULT_GPU = (0.5, 1.0, 2.0)
MULT_LLM = (0.25, 1.0, 4.0)
C2_PAIRS = (("S_full", "S_full_minus_D"), ("S_full", "S_full_minus_M"), ("S_full_ext", "S_full_ext_minus_qwen3_8b"),
            ("S_full_ext", "S_full_ext_minus_bge_m3"), ("S_full_ext", "S_full_ext_minus_e5"),
            ("S_full_ext", "S_full_ext_minus_colbert"))
LLM_BASES = {"llm_rerank_S0": "S0", "llm_rerank_S_full_ext": "S_full_ext", "llm_rerank_S_full_ext_blank": "S_full_ext"}


def load_costs(folder: Path) -> dict[str, dict[str, Any]]:
    folder = Path(folder)
    if not folder.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))}


def own_quantities(rec: dict[str, Any]) -> dict[str, float] | None:
    """Per-case physical quantities of one record: vCPU-hours, GPU-hours, tokens (None when nothing is recorded)."""
    q = {"vcpu_h": 0.0, "gpu_h": 0.0, "tok_in": 0.0, "tok_cached": 0.0, "tok_out": 0.0}
    found = False
    lat = rec.get("query_latency") or {}
    if lat.get("median_ms") is not None:
        q["vcpu_h"] += float(lat["median_ms"]) * float(lat.get("threads") or 8) / 3.6e6
        found = True
    enc = rec.get("query_encoding") or {}
    if enc.get("available") and enc.get("seconds_per_case") is not None:
        dev = str(enc.get("device") or "").lower()
        # RUN-B records the card's name ("NVIDIA GeForce RTX 5080"), not "cuda": any NVIDIA / GeForce / RTX device is a GPU
        key = "gpu_h" if any(w in dev for w in ("cuda", "gpu", "nvidia", "geforce", "rtx")) else "vcpu_h"
        q[key] += float(enc["seconds_per_case"]) / 3600
        found = True
    llm = (rec.get("llm") or {}).get("per_case") or {}
    if llm:
        q["tok_in"] += float(llm.get("input_tokens") or 0) - float(llm.get("cached_input_tokens") or 0)
        q["tok_cached"] += float(llm.get("cached_input_tokens") or 0)
        q["tok_out"] += float(llm.get("output_tokens") or 0)
        found = True
    return q if found else None


def composed(system: str, costs: dict[str, dict[str, Any]], seen: frozenset[str] = frozenset()) -> tuple[dict[str, float] | None, list[str]]:
    """Own quantities plus precomputed sources, members and the reordered base; (quantities, missing components)."""
    rec = costs.get(system)
    if rec is None or system in seen:
        return None, [system]
    q = own_quantities(rec)
    if q is None:
        return None, [system]
    missing: list[str] = []
    pre = (rec.get("query_latency") or {}).get("precomputed_sources") or {}
    parts = [f"src_{s}" for s in pre.get("frozen_source_scores") or []] + list(pre.get("extra_members_scored_in_fit") or [])
    base = LLM_BASES.get(C.base_name(system))
    if base:
        parts.append(base)
    for part in dict.fromkeys(parts):
        sub, miss = composed(part, costs, seen | {system})
        if sub is None:
            missing.append(part)
        else:
            missing += miss
            for k in q:
                q[k] += sub[k]
    own_enc = bool((rec.get("query_encoding") or {}).get("available")) or C.base_name(system)[len("S0+"):] in parts
    for enc in encoders_read(system, pre, own_enc):         # D114: the query encodings a ranker's columns need
        e = encoding_quantities(costs.get(ENCODER_RECORD.get(enc, enc)) or {})
        if e is None:
            missing.append(f"query encoding of {enc}")
        else:
            for k in q:
                q[k] += e[k]
    return q, missing


# D114 (the author, 25 Oct 2026): a learned ranker also pays, once per case, the query encoding of every encoder whose
# columns it reads (its own cost record times the ranker on precomputed source scores only). The encoder of a source
# group is the one of ``uol.final.c2.ENCODER_GROUPS``; S0+<x> reads <x>'s dense column. The record holding each
# encoder's measured query encoding: the 8B's own; bge-m3's from the ColBERT job (one bge-m3 forward pass gives the
# dense and multi-vector heads); none is recorded for e5-small, which stays a missing component (a lower bound).
ENCODER_RECORD = {"qwen3_8b": "qwen3_8b", "bge_m3": "bge_m3_colbert", "e5": "e5_small"}


def encoders_read(system: str, pre: dict[str, Any], own_encoding: bool = False) -> list[str]:
    """Encoders whose query encoding the ranker needs; S0+X's X only when its own record does not already time it."""
    from uol.final import c2
    srcs = set(pre.get("frozen_source_scores") or [])
    out = [enc for enc, members in c2.ENCODER_GROUPS.items() if srcs & set(members)]
    base = C.base_name(system)
    if base.startswith("S0+") and not own_encoding:
        out.append(base[len("S0+"):])
    return out


def encoding_quantities(rec: dict[str, Any]) -> dict[str, float] | None:
    """The query-encoding part of a record only (its scoring time is not a ranker's cost)."""
    enc = rec.get("query_encoding") or {}
    if not (enc.get("available") and enc.get("seconds_per_case") is not None):
        return None
    return own_quantities({"query_encoding": enc})


def usd_per_1000(q: dict[str, float], prices: dict[str, float], m_cpu: float = 1, m_gpu: float = 1, m_llm: float = 1) -> float:
    per = (q["vcpu_h"] * prices["cpu_vcpu_hour_usd"] * m_cpu + q["gpu_h"] * prices["gpu_hour_usd"] * m_gpu
           + (q["tok_in"] * prices.get("llm_input_per_mtok_usd", 0) + q["tok_cached"] * prices.get("llm_cached_input_per_mtok_usd", prices.get("llm_input_per_mtok_usd", 0))
              + q["tok_out"] * prices.get("llm_output_per_mtok_usd", 0)) / 1e6 * m_llm)
    return 1000 * per


def frontier(ndcg: dict[str, float], cost: dict[str, float]) -> set[str]:
    on = set()
    for s in cost:
        if not any(ndcg[t] > ndcg[s] and cost[t] <= cost[s] for t in cost if t != s):
            on.add(s)
    return on


def run(runs: Path, out: Path, *, population: str = "DEV", costs_dir: Path | None = None, prices: Path | None = None,
        draws: int = C.DRAWS, seed: int = C.SEED) -> dict[str, Any]:
    out = Path(out)
    store = C.RunStore(runs, population)
    costs_dir = Path(costs_dir) if costs_dir else Path(runs).parent / "cost"
    costs = load_costs(costs_dir)
    price = json.loads(Path(prices).read_text(encoding="utf-8")) if prices and Path(prices).exists() else None
    boot = C.Boot(store.labels.component, draws, seed)
    notes = ["LLM configurations (passes x samples) need the saved C5 orders, not in the runs folder: skipped."]
    quantities, missing = {}, {}
    for s in store.systems:
        q, miss = composed(s, costs)
        if q is not None:
            quantities[s] = q
            if miss:
                missing[s] = miss
    if not costs:
        notes.append(f"No cost records at {costs_dir.as_posix()}: no system can be placed; the frontier is not drawn.")
    if price is None:
        notes.append("No price file (--prices): US$ costs and the 27 scenarios are not computed (D80 needs cited, dated prices).")
    ndcg = {s: boot.macro(store.cases(s)["ndcg_at_10"].to_numpy(float))["mean"] for s in quantities}
    phys = [{"system": s, "ndcg_at_10": ndcg[s], **q, "composition": "partial" if s in missing else "complete",
             "missing_components": ", ".join(missing.get(s, []))} for s, q in quantities.items()]
    member, scen_rows, attrib = {s: 0 for s in quantities}, [], []
    if price is not None and quantities:
        for mc, mg, ml in itertools.product(MULT_CPU, MULT_GPU, MULT_LLM):
            cost = {s: usd_per_1000(q, price, mc, mg, ml) for s, q in quantities.items()}
            on = frontier(ndcg, cost)
            for s in on:
                member[s] += 1
            scen_rows.append({"cpu_x": mc, "gpu_x": mg, "llm_x": ml, "on_frontier": ", ".join(sorted(on))})
        base_cost = {s: usd_per_1000(q, price) for s, q in quantities.items()}
        for a, b in C2_PAIRS:
            if a in quantities and b in quantities:
                dc, dn = base_cost[a] - base_cost[b], ndcg[a] - ndcg[b]
                attrib.append({"full": a, "reduced": b, "delta_ndcg_at_10": dn, "delta_usd_per_1000": dc,
                               "ndcg_per_usd": dn / dc if dc else float("nan"),
                               "delta_vcpu_h_per_case": quantities[a]["vcpu_h"] - quantities[b]["vcpu_h"],
                               "delta_gpu_h_per_case": quantities[a]["gpu_h"] - quantities[b]["gpu_h"]})
        for r in phys:
            r["usd_per_1000_base"] = base_cost[r["system"]]
    absent_c2 = [f"{a} - {b}" for a, b in C2_PAIRS if not (a in quantities and b in quantities)]
    if absent_c2:
        notes.append("C2 attribution pairs without both cost records (skipped): " + ", ".join(absent_c2) + ".")
    share = [{"system": s, "ndcg_at_10": ndcg[s], "scenarios_on_frontier": member[s], "share": member[s] / 27}
             for s in quantities] if scen_rows else []
    C.write_csv(out / "c8_quantities.csv", phys)
    C.write_csv(out / "c8_frontier_share.csv", share)
    C.write_csv(out / "c8_scenarios.csv", scen_rows)
    C.write_csv(out / "c8_attribution.csv", attrib)
    ev = C.Evidence("C8")
    for r in sorted(share, key=lambda r: -r["share"]):
        ev.add(f"{r['system']} on {population}", ["share of 27 price scenarios on the frontier", "nDCG@10"],
               [f"{r['share']:.2f} ({r['scenarios_on_frontier']}/27)", C.fmt(r["ndcg_at_10"])],
               f"cost/{r['system']}.json; {store.source(r['system'])}")
    for r in attrib:
        ev.add(f"{r['full']} vs {r['reduced']} on {population}", ["nDCG@10 change", "US$ per 1,000 cases change", "nDCG per US$"],
               [f"{r['delta_ndcg_at_10']:+.3f}", f"{r['delta_usd_per_1000']:+.4f}", C.fmt(r["ndcg_per_usd"], 4)],
               f"cost/{r['full']}.json; cost/{r['reduced']}.json")
    ev.write(out / "evidence_rows.md", notes)
    rec = {"item": "C8", "plan": "ANALYSIS_PLAN_C §8", "population": population, "costs_dir": str(costs_dir),
           "cost_records": len(costs), "placed": len(quantities), "prices": price, "partial": missing, "notes": notes}
    C.write_json(out / "c8.json", rec)
    return rec
