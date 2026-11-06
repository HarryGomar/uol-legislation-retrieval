"""Inputs of the GPU jobs that depend on other results of the run (written on the run computer, CPU only).

* ``bge_m3_ft --fold k``: training triples from the fold's training rows only (masked facts, a cited article, three
  hard negatives from the frozen bge-m3 dense top 50 that are neither cited nor generic) and the items to encode
  (every catalogue unit + the fold's held-out development queries).
* ``qwen3_0p6b_ft --fold k`` (D89): the same for Qwen3-Embedding-0.6B, with hard negatives from the frozen qwen3_0p6b
  vectors and queries in the instruction variant the base chose on that fold.
* ``query-only`` encodes through the Qwen3-8B int8 path (after the fidelity gate): ``qwen_int8_dev`` (every dev
  query, variant c: the fidelity check against the cached vectors), ``query2doc`` (facts + the saved L2 issue
  decomposition, variant c), ``hyde`` (the hypothetical articles of ``llm_jobs.hyde``, document side, no
  instruction), ``devn`` (clause-dropped facts of DEV-N, variant c).
Every row passes ``guard.dev_rows_only``; nothing here reads a label of a scored fold.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from uol.final import context, guard

FT_TASK = {"model": "BAAI/bge-m3", "revision": "5617a9f61b028005a4858fdac845db406aefb181", "epochs": 1, "lr": 1e-5,
           "batch": 16, "tau": 0.05, "max_train_tokens": 512, "max_tokens": 2048, "seed": 20261004, "budget_seconds": None,
           "n_hard": 3, "hard_source": "dense_bge_maxp top 50 (frozen, label-free scores)"}
INT8 = {"model": "Qwen/Qwen3-Embedding-8B", "revision": "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af", "max_tokens": 8192,
        "precision": "int8", "batch_tokens": 16384, "limit": None, "budget_seconds": None}


def inputs_root() -> Path:
    p = context.active().work / "gpu_inputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _units() -> list[dict[str, Any]]:
    d = context.active().path("units")
    return [json.loads(line) for line in (d / "units.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def smoke_articles(units: list[dict[str, Any]], n: int = 8) -> set[int]:
    """Article units a smoke encodes: the first n and the n longest, so a smoke meets the encode's real peak memory
    (RUN-B, 23 Oct: the qwen3_0p6b_ft smoke passed on short units, the full fold spilled out of the card)."""
    arts = [i for i, u in enumerate(units) if u["g"] == "articles"]
    longest = sorted(arts, key=lambda i: (-len(units[i]["text"]), i))[:n]
    return set(arts[:n]) | set(longest)


def bge_m3_ft(k: int, smoke: bool = False) -> Path:
    from uol.embedding_push import common as ep
    d = ep.release()
    roles = ep.roles()
    dev = guard.dev_rows_only(ep.dev_rows())
    train = dev[roles[dev] != f"dev_fold{k}"]
    held = dev[roles[dev] == f"dev_fold{k}"]
    S = ep.open_scores("dense_bge_maxp")
    generic = set(d.generic_idx.tolist())
    rng = np.random.default_rng(FT_TASK["seed"] + k)
    if smoke:                                         # smoke stage: a few training rows, a few units and queries
        train, held = train[:24], held[:4]
    out = inputs_root() / (f"bge_m3_ft_f{k}" if not smoke else "bge_m3_ft_smoke")
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s in range(0, len(train), 256):
            rows = train[s:s + 256]
            block = np.asarray(S[rows])
            for i, r in enumerate(rows):
                pos = set(d.positive_idx[int(r)])
                if not d.specific_idx[int(r)]:
                    continue
                top = [int(c) for c in np.argsort(-block[i], kind="stable")[:50] if int(c) not in pos and int(c) not in generic]
                for p in d.specific_idx[int(r)]:
                    negs = list(rng.choice(top, min(FT_TASK["n_hard"], len(top)), replace=False)) if top else []
                    f.write(json.dumps({"q": d.queries[int(r)].text, "pos": d.candidate_texts[int(p)],
                                        "negs": [d.candidate_texts[int(c)] for c in negs]}, ensure_ascii=False) + "\n")
                    n += 1
    held_set = set(held.tolist())
    all_units = _units()
    keep = smoke_articles(all_units) if smoke else None
    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, u in enumerate(all_units):
            if u["g"] == "articles" and smoke and i not in keep:
                continue
            if u["g"] == "articles" or (u["g"] == "queries_a" and int(u["o"]) in held_set):
                f.write(json.dumps({"i": i, "g": u["g"], "o": u["o"], "w": u["w"], "text": u["text"]}, ensure_ascii=False) + "\n")
    (out / "task.json").write_text(json.dumps({**FT_TASK, "key": f"bge_m3_ft_f{k}", "fold": k, "train_triples": n}, indent=1),
                                   encoding="utf-8")
    return out


QWEN_FT_TASK = {**FT_TASK, "model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                "max_tokens": 8192, "encode_batch": 4, "no_model_prompt": True,
                # seq 24: the catalogue and queries are encoded like the base (legal_encode: SDPA without enable_gqa,
                # token-planned batches of length-sorted texts, fp32, last-token pooling); encode_batch is unused
                "encode": "planned", "batch_tokens": 32768,
                "hard_source": "qwen3_0p6b top 50 (frozen vectors, MaxP, the fold's chosen query variant)",
                "differences_from_bge_m3_ft": ("queries carry the instruction of the variant the base qwen3_0p6b chose "
                                               "on the fold's inner folds (the units' own query text); the catalogue is "
                                               "encoded in the base's title-led windows (up to 8,192 tokens, units as "
                                               "they are) with batch 4; no sentence-transformers default prompt")}


def base_variant(k: int, system: str = "qwen3_0p6b") -> str:
    """The query variant the frozen base chose on outer fold k (its DEV receipt, inner-fold selection)."""
    rec = json.loads((context.active().store / "receipts" / f"{system}__DEV.json").read_text(encoding="utf-8"))
    fold = next(f for f in rec["folds"] if int(f["k"]) == k)
    return str(fold["fit"]["params"]["variant"])


def qwen3_0p6b_ft(k: int, smoke: bool = False) -> Path:
    """D89: inputs of the full fine-tune of Qwen3-Embedding-0.6B for outer fold k, as ``bge_m3_ft``: triples from the
    fold's training rows only (facts in the fold's chosen variant, a cited specific article, three hard negatives
    from the base's top 50 that are neither cited nor generic), the catalogue units and the held-out queries."""
    from uol.embedding_push import common as ep
    from uol.final import registry
    from uol.final.adapters.dense import VectorSet, vector_dir
    d = ep.release()
    roles = ep.roles()
    dev = guard.dev_rows_only(ep.dev_rows())
    train = dev[roles[dev] != f"dev_fold{k}"]
    held = dev[roles[dev] == f"dev_fold{k}"]
    variant = base_variant(k)
    group = f"queries_{variant}"
    units = _units()
    qtext = {int(u["o"]): u["text"] for u in units if u["g"] == group}
    vs = VectorSet(vector_dir(registry.get("qwen3_0p6b")))
    generic = set(d.generic_idx.tolist())
    rng = np.random.default_rng(QWEN_FT_TASK["seed"] + k)
    if smoke:
        train, held = train[:24], held[:4]
    key = f"qwen3_0p6b_ft_f{k}" if not smoke else "qwen3_0p6b_ft_smoke"
    out = inputs_root() / key
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "train.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for s in range(0, len(train), 256):
            rows = train[s:s + 256]
            block = vs.scores(variant, rows)
            for i, r in enumerate(rows):
                pos = set(d.positive_idx[int(r)])
                if not d.specific_idx[int(r)]:
                    continue
                top = [int(c) for c in np.argsort(-block[i], kind="stable")[:50] if int(c) not in pos and int(c) not in generic]
                for p in d.specific_idx[int(r)]:
                    negs = list(rng.choice(top, min(QWEN_FT_TASK["n_hard"], len(top)), replace=False)) if top else []
                    f.write(json.dumps({"q": qtext[int(r)], "pos": d.candidate_texts[int(p)],
                                        "negs": [d.candidate_texts[int(c)] for c in negs]}, ensure_ascii=False) + "\n")
                    n += 1
    held_set = set(held.tolist())
    keep = smoke_articles(units) if smoke else None
    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, u in enumerate(units):
            if u["g"] == "articles" and smoke and i not in keep:
                continue
            if u["g"] == "articles" or (u["g"] == group and int(u["o"]) in held_set):
                f.write(json.dumps({"i": i, "g": u["g"], "o": u["o"], "w": u["w"], "text": u["text"]}, ensure_ascii=False) + "\n")
    (out / "task.json").write_text(json.dumps({**QWEN_FT_TASK, "key": key, "fold": k, "variant": variant,
                                               "train_triples": n}, indent=1), encoding="utf-8")
    return out


def query_only(key: str, texts: dict[int, str], group: str = "queries_c") -> Path:
    """legal_encode inputs for query-side strings only (no article units; ingest reuses the 8B article vectors)."""
    rows = guard.dev_rows_only(sorted(texts), f"{key} rows")
    out = inputs_root() / key
    out.mkdir(parents=True, exist_ok=True)
    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"i": i, "g": group, "o": int(r), "w": 0, "text": texts[int(r)]}, ensure_ascii=False) + "\n")
    (out / "task.json").write_text(json.dumps({**INT8, "key": key, "prompts": {group: "none"}, "subset": "query_only",
                                               "articles_from": "qwen8b"}, indent=1), encoding="utf-8")
    return out


def qwen_texts(kind: str) -> dict[int, str]:
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    d = ep.release()
    dev = guard.dev_rows_only(ep.dev_rows())
    if kind == "qwen_int8_dev":
        return {int(r): qe.query_input("c", d.queries[int(r)].text) for r in dev}
    if kind == "query2doc":
        from uol.round3 import l2_features
        issues = l2_features.luna_issues("a")
        return {int(r): qe.query_input("c", d.queries[int(r)].text + "\n" + "\n".join(issues.get(int(r), []))) for r in dev}
    if kind == "hyde":
        p = context.active().work / "llm" / "hyde" / "texts.jsonl"
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        return {int(x["row"]): x["text"] for x in recs}
    if kind == "devn":
        from uol.final import devn
        return {int(r): qe.query_input("c", t) for r, t in devn.dropped_texts().items()}
    raise ValueError(kind)


N_CONTROL = 50


def devn_job(target: str, variant: str = "devn") -> Path:
    """DEV-N (or DEV-L, C6) re-encode of one vector set's queries (queries only), with that model's own query
    treatment, plus 50 unchanged development queries as a fidelity control against the saved vectors. Run in the
    devn (devl) variant, whose rows carry their changed facts."""
    from uol.embedding_push import common as ep, qwen_endpoint as qe
    from uol.final.adapters import gpu
    mod = context.text_module(variant)
    s = gpu.specs()[gpu.DEVN_ENCODERS[target]]
    d = ep.release()
    dn = mod.rows()
    dn_set = set(dn)
    rng = np.random.default_rng(20261021)
    others = [int(r) for r in guard.dev_rows_only(ep.dev_rows()) if int(r) not in dn_set]
    control = sorted(int(x) for x in rng.choice(others, size=min(N_CONTROL, len(others)), replace=False))
    groups = [g for g in s["prompts"] if g.startswith("queries_")]
    pre = s.get("prefix") or {}
    out = inputs_root() / f"{variant}__{target}"
    out.mkdir(parents=True, exist_ok=True)
    i = 0
    with (out / "items.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for g in groups:
            v = g.split("_", 1)[1]
            for r in sorted(dn_set | set(control)):
                t = d.queries[r].text                                   # DEV-N rows carry their dropped facts here
                # variant b / c strings are the endpoint's query_input (as in the units file), also for specs whose
                # b / c prompts are "none" without the instruct flag (euler_legal, qwen3_8b_int8path; review 22 Oct §4.2)
                t = qe.query_input(v, t) if (s.get("instruct") or (s["prompts"].get(g) == "none" and v in ("b", "c"))) \
                    else pre.get(g, "") + t
                f.write(json.dumps({"i": i, "g": g, "o": r, "w": 0, "text": t}, ensure_ascii=False) + "\n")
                i += 1
    task = {"key": f"{variant}__{target}", "model": s["model"], "revision": s["revision"], "max_tokens": s["max_tokens"],
            "prompts": {g: s["prompts"][g] for g in groups}, "precision": s["precision"], "batch_tokens": s["batch_tokens"],
            "limit": None, "budget_seconds": None, "subset": variant, "target": target, "control_rows": control,
            **{k: s[k] for k in ("trust_remote_code", "encode_kwargs_by_prompt", "device", "device_reason") if k in s}}
    (out / "task.json").write_text(json.dumps(task, indent=1), encoding="utf-8")
    return out


def write(kind: str, fold: int | None = None, smoke: bool = False) -> Path:
    if kind == "bge_m3_ft" and smoke:
        return bge_m3_ft(int(fold or 0), smoke=True)
    if kind.split("__", 1)[0] in context.TEXT_VARIANTS and "__" in kind:
        variant, target = kind.split("__", 1)
        return devn_job(target, variant)
    if kind == "bge_m3_ft":
        return bge_m3_ft(int(fold))
    if kind == "qwen3_0p6b_ft":                                   # D89
        return qwen3_0p6b_ft(int(fold or 0), smoke=smoke)
    return query_only(kind, qwen_texts(kind), "queries_a" if kind == "hyde" else "queries_c")
