"""E6: an LLM agent with search tools (use case U3; approved D25; dev only).

Two tool conditions with the same LLM (Round 3 Luna via ``uol.round3.luna``: cache, ledger, prompt guard), the same
budget and the same open/neighbour tools; only the search tool differs:
* ``words``: stemmed BM25 (the L* channel, k1 .9, b .75) over the 39,178 catalogue articles;
* ``embedding``: Qwen3-Embedding-8B dense search, MaxP over the cached article vectors; the agent's query is embedded
  through the project endpoint client (query variant c, as ``dense_qwen8b``) into a cache under this run folder.
Tools: buscar (top 10 with an excerpt), abrir (article text), vecinos (statute neighbours from G2). Budget: at most 8
steps and 20 opened articles; the agent returns at most 10 identifiers. Steps run in lockstep: one Luna call per
step carries every active case of a condition (batches of ``PER_CALL``), so a 100-row pilot costs about
2 conditions × 8 steps × ⌈100 / PER_CALL⌉ calls.

    python -m uol.eval_lenses.e6_agent smoke            # 3 dev rows per condition (≤ about 16 Luna calls)
    python -m uol.eval_lenses.e6_agent pilot            # 100 stratified dev rows; ONLY after the lockbox batch is done
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from uol.embedding_push import common as ep
from uol.eval_lenses import core

PROMPT = Path(__file__).resolve().parent / "prompts" / "E6_agent.txt"
OUT = core.OUT / "e6"
EFFORT = "medium"
MAX_STEPS, MAX_OPENS, RETURN = 8, 20, 10
PER_CALL = 10
WORKERS = 4
EXCERPT, OPEN_CHARS = 160, 900
START_EXCERPT = 600
ITEM_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["accion", "consulta", "articulo", "respuesta"],
               "properties": {"accion": {"type": "string", "enum": ["buscar", "abrir", "vecinos", "final"]},
                              "consulta": {"type": "string"}, "articulo": {"type": "string"},
                              "respuesta": {"type": "array", "items": {"type": "string"}}}}


def aid(i: int) -> str:
    return f"A{i:05d}"


def parse_aid(s: str) -> int | None:
    m = re.fullmatch(r"\s*A(\d{5})\s*", s or "")
    return int(m.group(1)) if m else None


@lru_cache(maxsize=None)
def label(i: int) -> str:
    d = ep.release()
    title = d.candidate_texts[i].split("\n", 1)[0].strip()
    return f"{title} › Art. {d.candidate_ids[i].split('#articulo=')[-1]}"


def body(i: int) -> str:
    t = ep.release().candidate_texts[i]
    return re.sub(r"\s+", " ", t.split("\n", 1)[1] if "\n" in t else t).strip()


class Tools:
    """Search, open and neighbour tools over the catalogue (generic articles are searchable like any other)."""

    def __init__(self, condition: str):
        self.condition = condition
        d = ep.release()
        self.N = len(d.candidate_ids)
        if condition == "words":
            from uol.facts_to_law import lexical
            self.bm = lexical.LexicalIndex(d.candidate_texts, kind="bm25_stem")
        else:
            from uol.embedding_push import qwen_endpoint as qe
            self.qe = qe
            A, owner = qe.article_vectors()
            order = np.argsort(owner, kind="stable")
            self.starts = np.flatnonzero(np.r_[True, np.diff(owner[order]) != 0])
            self.A = A[order]
            self.store = qe.VectorStore(OUT / "qwen_store_e6")
            self.tok = qe.load_tokenizer()
            self.endpoint_inputs = 0
        from uol.eval_lenses.lenses import structure
        self.st = structure()

    def search_many(self, queries: list[str]) -> list[list[int]]:
        if not queries:
            return []
        if self.condition == "words":
            S = self.bm.score(queries, {"k1": 0.9, "b": 0.75})
        else:
            qe = self.qe
            inputs = [qe.query_input("c", q) for q in queries]
            counts = qe.count_tokens(self.tok, inputs)
            items = [(qe.content_key(s), s, int(c) + 1) for s, c in zip(inputs, counts)]
            missing = [it for it in items if it[0] not in self.store]
            if missing:
                client = qe.Client(qe.EMB_BASE, log_path=OUT / "e6_endpoint_requests.jsonl")
                qe.encode_items(client, self.store, items, label="e6_agent", stop_file=None, progress=core.log)
                self.endpoint_inputs += len({k for k, *_ in missing})
            Q = qe.unit_rows(np.asarray(self.store.get([k for k, *_ in items]), dtype=np.float32))
            S = np.maximum.reduceat(Q @ self.A.T, self.starts, axis=1)
        tie = ep.tie_order()
        out = []
        for row in S:
            part = np.argpartition(-row, 40)[:40]
            out.append([int(x) for x in part[np.lexsort((tie[part], -row[part]))][:10]])
        return out

    def neighbours(self, i: int) -> list[int]:
        st = self.st
        near = [b for a, b in st["adj1"] if a == i] + [a for a, b in st["adj1"] if b == i]
        near += [b for a, b in st["adj2"] if a == i] + [a for a, b in st["adj2"] if b == i]
        g = st["sibling"][i]
        if g >= 0:
            near += [int(x) for x in np.flatnonzero(st["sibling"] == g)[:12] if int(x) != i]
        seen, out = set(), []
        for x in near:
            if x not in seen:
                seen.add(x)
                out.append(int(x))
        return out[:10]


def repaired_ids(res: dict, ids: list[str]) -> dict[str, dict] | None:
    """Map a batch response to the requested ids, repairing ids the model truncated or extended when exactly one
    requested id and one returned id are prefixes of each other; None if the batch cannot be repaired."""
    got = {r["id"]: r for r in (res.get("output") or {}).get("results", []) if isinstance(r, dict) and "id" in r}
    missing = [i for i in ids if i not in got]
    extra = [g for g in got if g not in ids]
    for i in missing:
        cand = [g for g in extra if i.startswith(g) or g.startswith(i)]
        if len(cand) != 1:
            return None
        got[i] = got.pop(cand[0])
        extra.remove(cand[0])
    return {i: got[i] for i in ids} if all(i in got for i in ids) else None


def obs_list(ids: list[int]) -> str:
    return "\n".join(f"  {aid(i)} {label(i)}: {body(i)[:EXCERPT]}{'…' if len(body(i)) > EXCERPT else ''}" for i in ids)


def start_obs(ids: list[int]) -> str:
    """The opening list of the list agent (uol.final.llm_agents): as the reranker sees the articles, numbered in order."""
    return "\n".join(f"  {k}. {aid(i)} {label(i)}: {body(i)[:START_EXCERPT]}{'…' if len(body(i)) > START_EXCERPT else ''}"
                     for k, i in enumerate(ids, 1))


def run(rows: list[int], condition: str, tag: str, start: dict[int, list[int]] | None = None) -> dict[str, Any]:
    from uol.round3 import common as C
    from uol.round3 import luna, prompt_guard
    if np.isin(rows, ep.lockbox_rows()).any():
        raise PermissionError("E6 runs on dev rows only")
    d = ep.release()
    tools = Tools(condition)
    tpl = PROMPT.read_text(encoding="utf-8")
    schema = luna.batch_schema(ITEM_SCHEMA)
    tool_desc = ("búsqueda por palabras (coincidencia léxica con el texto de los artículos)" if condition == "words"
                 else "búsqueda semántica (similitud de significado entre tu consulta y los artículos)")
    state = {r: {"history": [], "seen": set(), "opens": 0, "final": None, "steps": 0, "sanctioned": []} for r in rows}
    for r, ids in (start or {}).items():
        S, obs = state[r], start_obs(ids)
        S["seen"].update(ids)
        S["order"] = list(ids)
        S["history"].append(f"lista inicial →\n{obs}")
        S["sanctioned"].append(obs)
    usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    splits = {"split_batches": 0, "split_cases": 0}             # batches re-asked one case per call
    t0 = time.time()
    for step in range(1, MAX_STEPS + 1):
        active = [r for r in rows if state[r]["final"] is None]
        if not active:
            break
        last = step == MAX_STEPS
        actions: dict[int, dict] = {}
        jobs = []

        def build(part: list[int]) -> tuple[str, list[str]]:
            """This step's prompt for ``part`` (a batch or one case): the same template, rules and prompt guard."""
            texts, sanctioned, ids = [], [], []
            for r in part:
                S = state[r]
                hist = "\n".join(f"{k}. {h}" for k, h in enumerate(S["history"], 1)) or "(sin pasos todavía)"
                budget = (f"paso {step} de {MAX_STEPS}" + (" — ÚLTIMO PASO: la acción debe ser final" if last else "")
                          + f"; aperturas restantes {MAX_OPENS - S['opens']}")
                qid = f"e6{condition[0]}{r}"
                ids.append(qid)
                texts.append(f"[id={qid}]\nHechos: {re.sub(r'\s+', ' ', d.queries[r].text).strip()}\n"
                             f"Herramienta de búsqueda: {tool_desc}\nHistorial:\n{hist}\nPresupuesto: {budget}")
                sanctioned += S["sanctioned"]
            prompt = tpl.replace("{items}", "\n\n".join(texts))
            prompt_guard.check(prompt, part, sanctioned=sanctioned, template=tpl)
            return prompt, ids

        for s in range(0, len(active), PER_CALL):
            part = active[s:s + PER_CALL]
            prompt, ids = build(part)
            jobs.append((prompt, part, ids))
        results_l = luna.run_many([(pr, schema) for pr, _, _ in jobs], effort=EFFORT, tag=tag, workers=WORKERS)
        for (prompt, part, ids), res in zip(jobs, results_l):
            got = repaired_ids(res, ids)
            if got is None:
                row_of = dict(zip(ids, part))
                # still invalid after its resamples: the batch is split and each case asked alone (22 Oct 2026);
                # answers are matched by id, never by position
                got = luna.checked_or_split(res, prompt, schema, ids, lambda qid: (build([row_of[qid]])[0], qid),
                                            effort=EFFORT, tag=tag, stats=splits)
            usage["calls"] += 1
            u = res.get("usage") or {}
            usage["input_tokens"] += int(u.get("input_tokens", 0) or 0)
            usage["output_tokens"] += int(u.get("output_tokens", 0) or 0)
            usage["seconds"] += float(res.get("latency", 0) or 0)
            for qid, r in zip(ids, part):
                actions[r] = got[qid]
        # execute: searches are batched per step
        searches = [(r, a["consulta"].strip()) for r, a in actions.items()
                    if a["accion"] == "buscar" and a["consulta"].strip() and not last]
        results = dict(zip([r for r, _ in searches], tools.search_many([q for _, q in searches])))
        for r, a in actions.items():
            S = state[r]
            S["steps"] = step
            act = a["accion"] if not last else "final"
            if act == "final":
                ids_ = [parse_aid(x) for x in (a.get("respuesta") or [])]
                ids_ = [i for i in ids_ if i is not None and i in S["seen"]]
                if not ids_ and last:          # forced stop without a valid answer: fall back to the order seen
                    ids_ = list(S["order"]) if "order" in S else []
                S["final"] = list(dict.fromkeys(ids_))[:RETURN]
                S["history"].append(f"final({len(S['final'])} artículos)")
                continue
            if act == "buscar":
                q = a["consulta"].strip()
                hits = results.get(r, [])
                S["seen"].update(hits)
                S.setdefault("order", [])
                S["order"] += [h for h in hits if h not in S["order"]]
                obs = obs_list(hits)
                S["history"].append(f"buscar(\"{q}\") →\n{obs}")
                S["sanctioned"] += [q, obs]
            elif act in ("abrir", "vecinos"):
                i = parse_aid(a["articulo"])
                if i is None or i not in S["seen"]:
                    S["history"].append(f"{act}({a['articulo']}) → identificador no visto")
                    S["sanctioned"].append(a["articulo"])
                    continue
                if act == "abrir":
                    if S["opens"] >= MAX_OPENS:
                        S["history"].append(f"abrir({aid(i)}) → límite de aperturas alcanzado")
                        continue
                    S["opens"] += 1
                    txt = body(i)[:OPEN_CHARS]
                    S["history"].append(f"abrir({aid(i)}) → {label(i)}: {txt}")
                    S["sanctioned"] += [label(i), txt]
                else:
                    nb = tools.neighbours(i)
                    S["seen"].update(nb)
                    obs = obs_list(nb)
                    S["history"].append(f"vecinos({aid(i)}) →\n{obs}")
                    S["sanctioned"].append(obs)
            else:
                S["history"].append("acción vacía")
        core.log(f"e6 {tag} step {step}: {len(active)} active, calls so far {usage['calls']}")
    # evaluation (P_spec condensed: generic articles removed from the returned list)
    m = core.meta()
    loc = {int(r): i for i, r in enumerate(m.rows)}
    per = []
    for r in rows:
        S = state[r]
        gold = set(int(x) for x in d.specific_idx[r])
        ret = [i for i in (S["final"] or []) if not m.generic[i]]
        gains = sum(1 / np.log2(k + 2) for k, i in enumerate(ret[:10]) if i in gold)
        idcg = sum(1 / np.log2(k + 2) for k in range(min(len(gold), 10)))
        base_ltr = core.load("LTR_emb_hgb")["ndcg"][loc[r]] if r in loc else None
        per.append({"row": int(r), "query_id": d.queries[r].query_id, "returned": [aid(i) for i in S["final"] or []],
                    "ndcg10": gains / idcg if idcg else None, "hit@10": bool(gold & set(ret[:10])), "steps": S["steps"],
                    "opens": S["opens"], "searches": sum(h.startswith("buscar") for h in S["history"]),
                    "ltr_emb_ndcg10": base_ltr, "history": S["history"]})
    return {"condition": condition, "rows": len(rows), "usage": usage, "splits": splits, "wall_seconds": time.time() - t0,
            "endpoint_inputs": getattr(tools, "endpoint_inputs", 0),
            "mean_ndcg10": float(np.mean([p["ndcg10"] for p in per if p["ndcg10"] is not None])),
            "mean_ltr_emb_ndcg10": float(np.mean([p["ltr_emb_ndcg10"] for p in per if p["ltr_emb_ndcg10"] is not None])),
            "cases": per}


def stratified_rows(n: int, seed: int = 20261004) -> list[int]:
    """n dev P_spec rows, stratified by primary materia (seeded)."""
    d, m = ep.release(), core.meta()
    rng = np.random.default_rng(seed)
    prim = np.asarray([(d.queries[int(r)].materias or ("?",))[0] for r in m.rows])
    mats = sorted(set(prim.tolist()))
    per = {k: list(rng.permutation(np.flatnonzero(prim == k))) for k in mats}
    out = []
    while len(out) < n:
        for k in mats:
            if per[k] and len(out) < n:
                out.append(int(m.rows[per[k].pop()]))
    return out


def main(argv=None) -> int:
    from uol.round3 import common as C
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["smoke", "pilot"])
    ap.add_argument("--batch-done-confirmed", action="store_true")
    a = ap.parse_args(argv)
    if a.command == "pilot" and not a.batch_done_confirmed:
        print("refusing: the 100-row pilot runs only after the lockbox batch is reported done (D25)", file=sys.stderr)
        return 2
    rows = stratified_rows(3 if a.command == "smoke" else 100)
    OUT.mkdir(parents=True, exist_ok=True)
    C.ledger({"kind": "job_start", "job_id": f"E6_{a.command}", "command": " ".join(sys.argv), "executor": "eval-lenses"})
    res = {"label": "dev, descriptive (pilot)", "rows": rows,
           "conditions": {c: run(rows, c, f"E6_{a.command}_{c}") for c in ("words", "embedding")}}
    core.write_json(OUT / f"{a.command}.json", res)
    C.ledger({"kind": "job_end", "job_id": f"E6_{a.command}", "exit_code": 0})
    for c, r in res["conditions"].items():
        core.log(f"E6 {a.command} {c}: nDCG@10 {r['mean_ndcg10']:.3f} (LTR_emb {r['mean_ltr_emb_ndcg10']:.3f}), "
                 f"calls {r['usage']['calls']}, endpoint inputs {r['endpoint_inputs']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
