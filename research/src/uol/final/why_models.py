"""Why the language models work (common_footing PLAN addendum 23, 3 Nov 2026; the development sample only).

W1 reasoning effort with the model alone; W2 the Claude models asked alone; W3 does the model know the tesis (registro,
year, court and rubro from the masked facts); W4 the model on reworded facts against two methods that cannot remember
a tesis; W5 the model's lead by year of publication (no calls). GPT models go through the Codex CLI route of
``round3.luna``; Claude models through the author's Claude account (the ``claude`` CLI with no tools, no settings and a
neutral system prompt), never an API key. Every prompt passes ``round3.prompt_guard``. Stored row numbers are the
activated bundle's, so ``main`` activates it first and ``sample_rows`` and ``build`` refuse any test row. The runs of
3 Nov were made without activating it and carried one test case (research/programs/why_models_20261103/ROW4806.md);
``uol.final.why_no_test_case`` derives their results without that case.

    py -3.12 -m uol.final.why_models alone --model gpt61sol --effort none --go
    py -3.12 -m uol.final.why_models alone --model opus --effort high --go
    py -3.12 -m uol.final.why_models probe --model gpt61sol --go
    py -3.12 -m uol.final.why_models reword --go
    py -3.12 -m uol.final.why_models report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from uol.final import footing as FT  # first: it points the audit modules at this run
from uol.final import audit_d181_common as A
from uol.final import llm_agents as L

OUT = A.WORK / "why"
REPORT = A.ROOT / "research" / "programs" / "why_models_20261103"
EMPTY = A.ROOT / ".cache" / "why_claude_cwd"          # an empty folder: the CLI finds no project or memory there
CLAUDE_EXE = Path.home() / "AppData" / "Roaming" / "npm" / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe"
CLAUDE_MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}
GPT_MODELS = {"gpt61sol": "gpt-6.1-sol", "gpt6solnw": "gpt-6-sol"}
SYSTEM_PROMPT = "Sigue exactamente las instrucciones del usuario y responde solo con el JSON pedido."
PROMPTS = Path(__file__).resolve().parents[1] / "round3" / "prompts"
PROBE_TPL = PROMPTS / "why_probe.txt"
REWORD_TPL = PROMPTS / "why_reword.txt"
PROBE_ITEM = {"type": "object", "additionalProperties": False, "required": ["registro", "anio", "organo", "rubro"],
              "properties": {k: {"type": ["string", "null"]} for k in ("registro", "anio", "organo", "rubro")}}
REWORD_ITEM = {"type": "object", "additionalProperties": False, "required": ["texto"],
               "properties": {"texto": {"type": "string"}}}
MARK = "[CITA_OCULTA]"
_lock = threading.Lock()
_used: set[str] = set()     # cache keys the current step read or wrote, so its usage counts its own calls only


# --------------------------------------------------------------------------- the Claude account route
def _cache_path() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT / "claude_cache.jsonl"


def _cache() -> dict[str, dict]:
    p = _cache_path()
    out = {}
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                x = json.loads(line)
                out[x["key"]] = x
    return out


def claude_call(prompt: str, schema: dict, model: str, effort: str, sample: int = 0, timeout: int = 1200) -> dict:
    """One cached call through the account's CLI: {"output": structured output, "usage", "usd_list_equivalent"}."""
    key = hashlib.sha256(json.dumps([model, effort, prompt, schema, sample], ensure_ascii=False, sort_keys=True)
                         .encode("utf-8")).hexdigest()
    with _lock:
        _used.add(key)
    hit = _cache().get(key)
    if hit is not None:
        return hit
    EMPTY.mkdir(parents=True, exist_ok=True)
    text = prompt + (f"\n\n(muestra {sample})" if sample else "")
    cmd = [str(CLAUDE_EXE), "-p", "--model", model, "--effort", effort, "--tools", "", "--system-prompt", SYSTEM_PROMPT,
           "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence", "--output-format", "json",
           "--json-schema", json.dumps(schema, ensure_ascii=False)]
    delay, last = 30.0, None
    for attempt in range(4):
        try:
            r = subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, cwd=EMPTY, timeout=timeout,
                               shell=False)
            out = json.loads(r.stdout.decode("utf-8"))
            if out.get("is_error") or not isinstance(out.get("structured_output"), dict):
                raise RuntimeError(f"claude: {str(out.get('result'))[:300]}")
            rec = {"key": key, "model": model, "effort": effort, "sample": sample, "output": out["structured_output"],
                   "usage": out.get("usage"), "usd_list_equivalent": out.get("total_cost_usd")}
            with _lock:
                with _cache_path().open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            return rec
        except (subprocess.TimeoutExpired, json.JSONDecodeError, RuntimeError, UnicodeDecodeError) as exc:
            last = exc
            time.sleep(delay)
            delay *= 2
    raise RuntimeError(f"claude call failed after retries: {last}")


def _which(model: str) -> tuple[str, str]:
    if model in CLAUDE_MODELS:
        return "claude", CLAUDE_MODELS[model]
    if model in GPT_MODELS:
        return "codex", GPT_MODELS[model]
    raise ValueError(model)


def ask(prompts: list[tuple[str, list[str]]], item: dict, model: str, effort: str, tag: str, workers: int = 4) -> list[dict[str, dict]]:
    """For each prompt, in order, its answers by id (every batch reuses the ids c01..c10, so the answers are never
    merged across batches), for either route; a batch with missing ids is asked again as an independent sample (up to
    three times), and still-missing ids stay missing (counted by the caller)."""
    from uol.round3 import luna
    route, name = _which(model)
    schema = luna.batch_schema(item)
    if route == "codex":
        L.pin_model(name)
        res = luna.run_many([(p, schema) for p, _ in prompts], effort=effort, tag=tag, workers=workers, model=name)
        out = []
        for (p, ids), r in zip(prompts, res):
            try:
                out.append(luna.checked(r, p, schema, ids, effort=effort, tag=tag, model=name))
            except luna.LunaError:
                out.append({})
        return out

    def one(job):
        p, ids = job
        for s in range(4):
            rec = claude_call(p, schema, name, effort, sample=s)
            res = {x.get("id"): x for x in rec["output"].get("results", []) if isinstance(x, dict)}
            if all(i in res and all(k in res[i] for k in item["required"]) for i in ids):
                return {i: res[i] for i in ids}
        return {i: res[i] for i in ids if i in res}
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, prompts))


def claude_usage(model: str, effort: str, keys: set[str] | None = None) -> dict:
    """Tokens and list-price equivalent of the account route's cached calls for one model and effort (only those in
    ``keys`` when given)."""
    tot = {"calls": 0, "input": 0, "output": 0, "usd_list_equivalent": 0.0}
    for x in _cache().values():
        if x["model"] == CLAUDE_MODELS.get(model, model) and x["effort"] == effort and (keys is None or x["key"] in keys):
            u = x.get("usage") or {}
            tot["calls"] += 1
            tot["input"] += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0)) + int(u.get("cache_creation_input_tokens", 0))
            tot["output"] += int(u.get("output_tokens", 0))
            tot["usd_list_equivalent"] += float(x.get("usd_list_equivalent") or 0)
    return tot


# --------------------------------------------------------------------------- prompts
def sample_rows() -> list[int]:
    """DEV-A300 as every stored sample run has it (the model alone of addendum 13 and the footing store's DEV-A300
    runs). Drawing the sample again today gives 20 other cases, so its rows are read from those runs, never redrawn."""
    rows = sorted(int(r) for r in pd.read_parquet(L.OUT / "alone" / "per_case.parquet", columns=["row"])["row"])
    ref = sorted(int(r) for r in A.per_case(FT.STORE, "haiku_rerank_S0+qwen3_8b", "DEV-A300")["row"])
    if rows != ref or len(rows) != 300:
        raise RuntimeError("the stored DEV-A300 runs disagree on the sample's rows")
    from uol.embedding_push import common as ep
    from uol.final import guard
    guard.dev_rows_only(rows, "DEV-A300")
    d = ep.release()
    ids = pd.read_parquet(L.OUT / "alone" / "per_case.parquet", columns=["row", "query_id"])
    if any(d.queries[int(r)].query_id != q for r, q in zip(ids["row"], ids["query_id"])):
        raise RuntimeError("stored rows and the active release disagree: activate the bundle first")
    return rows


def _items(rows: list[int], texts: dict[int, str] | None = None) -> tuple[str, list[str]]:
    from uol.embedding_push import common as ep
    d = ep.release()
    ids = [f"c{i + 1:02d}" for i in range(len(rows))]
    body = "\n\n".join(f"[id={i}] {re.sub(r'\s+', ' ', (texts or {}).get(r, d.queries[r].text)).strip()}"
                       for i, r in zip(ids, rows))
    return body, ids


def build(tpl_path: Path, rows: list[int], texts: dict[int, str] | None = None) -> tuple[str, list[str]]:
    from uol.final import guard
    from uol.round3 import prompt_guard
    guard.dev_rows_only(rows, "prompt")
    tpl = tpl_path.read_text(encoding="utf-8")
    body, ids = _items(rows, texts)
    prompt = tpl.replace("{items}", body)
    prompt_guard.check(prompt, rows, sanctioned=list((texts or {}).values()), template=tpl)
    return prompt, ids


def batches(rows: list[int], tpl: Path, size: int = 10, texts: dict[int, str] | None = None):
    parts = [rows[i:i + size] for i in range(0, len(rows), size)]
    built = [build(tpl, p, texts) for p in parts]
    return parts, built


# --------------------------------------------------------------------------- W1, W2: the model alone
def alone(model: str, effort: str, texts: dict[int, str] | None = None, name: str | None = None) -> dict:
    from uol.eval_lenses import llm_alone as LA
    from uol.embedding_push import common as ep
    d = ep.release()
    generic = {int(i) for i in d.generic_idx}
    rows = [r for r in sample_rows() if texts is None or r in texts]   # reworded: only rewrites that passed
    name = name or f"alone_{model}_{effort}"
    parts, built = batches(rows, LA.TEMPLATE, LA.BATCH, texts)
    t0 = time.perf_counter()
    _used.clear()
    got = ask(built, LA.ITEM, model, effort, tag=f"why_{name}")
    answers, missing = {}, 0
    for part, (_, ids), g in zip(parts, built, got):
        for i, r in zip(ids, part):
            if i in g:
                answers[r] = g[i]["articulos"]
            else:
                answers[r], missing = [], missing + 1
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps({str(k): v for k, v in answers.items()}, ensure_ascii=False), encoding="utf-8")
    frame = L.score_ranked(rows, {r: L.resolved(a, generic) for r, a in answers.items()})
    frame.to_parquet(out / "per_case.parquet")
    route, mname = _which(model)
    res = {"name": name, "model": mname, "effort": effort, "route": route, "cases": len(rows), "unanswered": missing,
           "mean_per_case": float(frame["ndcg_at_10"].mean()), "success_at_10": float(frame["success_at_10"].mean()),
           "seconds": round(time.perf_counter() - t0, 1),
           **({"usage": claude_usage(model, effort, set(_used))} if route == "claude" else {})}
    A.receipt(f"why_{name}", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


# --------------------------------------------------------------------------- W3: does the model know the tesis?
def norm_tokens(text: str | None) -> list[str]:
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]+", t)


def f1(a: str | None, b: str | None) -> float:
    x, y = norm_tokens(a), norm_tokens(b)
    if not x or not y:
        return 0.0
    from collections import Counter
    common = sum((Counter(x) & Counter(y)).values())
    if not common:
        return 0.0
    p, r = common / len(x), common / len(y)
    return 2 * p * r / (p + r)


def cited_numbers(row: int) -> set[str]:
    from uol.embedding_push import common as ep
    d = ep.release()
    out = set()
    for i in d.specific_idx[row]:
        m = re.search(r"articulo=([^#&]+)$", d.candidate_ids[int(i)])
        if m:
            out.add(re.sub(r"[^0-9a-z]", "", m.group(1).lower()))
    return out


def probe(model: str, effort: str = "medium") -> dict:
    from uol.round3 import prompt_guard
    rows = sample_rows()
    meta = prompt_guard.metadata()
    parts, built = batches(rows, PROBE_TPL)
    t0 = time.perf_counter()
    got = ask(built, PROBE_ITEM, model, effort, tag=f"why_probe_{model}")
    recs, answers = [], {}
    for part, (_, ids), g in zip(parts, built, got):
        for i, r in zip(ids, part):
            a, m = g.get(i) or {}, meta[r]
            answers[str(r)] = g.get(i)
            reg = re.sub(r"\D", "", str(a.get("registro") or ""))
            rub = a.get("rubro") or ""
            nums = {re.sub(r"[^0-9a-z]", "", n.lower()) for n in re.findall(r"(?i)art[íi]culos?\s+([0-9]+[a-z\-]*)", rub)}
            recs.append({"row": r, "answered": bool(a), "registro_exact": bool(reg) and reg == m["record_id"],
                         "year_exact": str(a.get("anio") or "")[:4] == (m["publication_date"] or "")[:4] and bool(a.get("anio")),
                         "rubro_f1": f1(rub, m["rubro"]), "organo_f1": f1(a.get("organo"), m["organo"]),
                         "rubro_given": bool(rub.strip()), "rubro_names_cited": bool(nums & cited_numbers(r))})
    f = pd.DataFrame(recs)
    f["recognized"] = f["registro_exact"] | (f["rubro_f1"] >= 0.8)
    f["recognized_loose"] = f["registro_exact"] | (f["rubro_f1"] >= 0.6)
    out = OUT / f"probe_{model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.json").write_text(json.dumps(answers, ensure_ascii=False), encoding="utf-8")
    f.to_parquet(out / "per_case.parquet")
    res = {"model": _which(model)[1], "effort": effort, "cases": len(f), "seconds": round(time.perf_counter() - t0, 1),
           **{k: float(f[k].mean()) for k in ("answered", "registro_exact", "year_exact", "rubro_given", "recognized",
                                               "recognized_loose", "rubro_names_cited", "rubro_f1", "organo_f1")}}
    A.receipt(f"why_probe_{model}", res)
    print(json.dumps(res, indent=1), flush=True)
    return res


# --------------------------------------------------------------------------- W4: reworded facts
LAW = re.compile(r"\b(Ley|C[óo]digo|Constituci[óo]n|Reglamento)\b[^,.;:()\[\]]{0,80}")


def law_names(text: str) -> set[str]:
    return {" ".join(norm_tokens(m.group(0))[:5]) for m in LAW.finditer(text)}


def reword_ok(orig: str, new: str) -> str | None:
    """None if the rewrite passes the checks of addendum 23, else the reason."""
    from uol.round3 import prompt_guard
    if new.count(MARK) != orig.count(MARK):
        return "markers"
    if prompt_guard.ARTICLE_NO.search(new) or prompt_guard.SEVEN.search(new):
        return "number"
    added = {n for n in law_names(new) if not any(n.split()[:3] == o.split()[:3] for o in law_names(orig))}
    if added:
        return "law"
    ratio = len(new) / max(len(orig), 1)
    if not 0.6 <= ratio <= 1.6:
        return "length"
    return None


def ngrams(text: str, n: int = 4) -> set[tuple[str, ...]]:
    t = norm_tokens(text)
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def reword(model: str = "sonnet", effort: str = "medium") -> dict:
    from uol.embedding_push import common as ep
    d = ep.release()
    rows = sample_rows()
    parts, built = batches(rows, REWORD_TPL)
    _used.clear()
    got = ask(built, REWORD_ITEM, model, effort, tag="why_reword")
    texts, fails = {}, {}
    for part, (_, ids), g in zip(parts, built, got):
        for i, r in zip(ids, part):
            texts[r] = ((g.get(i) or {}).get("texto") or "").strip()
    # a case that fails a check is rewritten once more, alone, as an independent sample
    for r in rows:
        why = reword_ok(d.queries[r].text, texts[r]) if texts[r] else "empty"
        if why:
            p, ids = build(REWORD_TPL, [r])
            rec = claude_call(p, __import__("uol.round3.luna", fromlist=["x"]).batch_schema(REWORD_ITEM),
                              CLAUDE_MODELS[model], effort, sample=1)
            alt = next((x.get("texto", "") for x in rec["output"].get("results", []) if x.get("id") == ids[0]), "").strip()
            why2 = reword_ok(d.queries[r].text, alt) if alt else "empty"
            if why2:
                fails[r] = why2
            else:
                texts[r] = alt
    keep = {r: t for r, t in texts.items() if r not in fails}
    kept = [len(ngrams(d.queries[r].text) & ngrams(t)) / max(len(ngrams(d.queries[r].text)), 1) for r, t in keep.items()]
    out = OUT / "reword"
    out.mkdir(parents=True, exist_ok=True)
    (out / "texts.json").write_text(json.dumps({str(k): v for k, v in keep.items()}, ensure_ascii=False, indent=0), encoding="utf-8")
    res = {"model": CLAUDE_MODELS[model], "effort": effort, "cases": len(rows), "kept": len(keep),
           "dropped": {str(k): v for k, v in fails.items()}, "share_4grams_kept_mean": float(np.mean(kept)),
           "share_4grams_kept_median": float(np.median(kept)), "usage": claude_usage(model, effort, set(_used))}
    A.receipt("why_reword", res)
    print(json.dumps(res, indent=1, default=float), flush=True)
    return res


def reword_texts() -> dict[int, str]:
    return {int(k): v for k, v in json.loads((OUT / "reword" / "texts.json").read_text(encoding="utf-8")).items()}


def _ranked(S: np.ndarray, rows: list[int], depth: int = 200) -> dict[int, list[int]]:
    return {r: [int(c) for c in np.argsort(-S[i], kind="stable")[:depth]] for i, r in enumerate(rows)}


def controls() -> dict:
    """W4's two methods that cannot remember a tesis, on the original and the reworded facts of the same cases:
    BM25 on stemmed words (k1 1.2, b .75) and Qwen3-Embedding-0.6B on the processor (variant c, MaxP)."""
    from uol.embedding_push import common as ep
    from uol.facts_to_law import lexical
    from uol.scale_curve import common as SC, local
    d = ep.release()
    texts = reword_texts()
    rows = [r for r in sample_rows() if r in texts]
    sides = {"original": [d.queries[r].text for r in rows], "reworded": [texts[r] for r in rows]}
    out_dir = OUT / "controls"
    out_dir.mkdir(parents=True, exist_ok=True)
    res: dict = {"cases": len(rows)}
    idx = lexical.LexicalIndex(d.candidate_texts, kind="bm25_stem")
    for side, tx in sides.items():
        f = L.score_ranked(rows, _ranked(idx.score(tx, {"k1": 1.2, "b": 0.75}), rows))
        f.to_parquet(out_dir / f"bm25_{side}.parquet")
        res[f"bm25_{side}"] = float(f["ndcg_at_10"].mean())
    from uol.embedding_push import qwen_endpoint as qe
    enc = local.Encoder(SC.MODELS["qwen3_0p6b"], threads=16)
    store = A.ROOT / "data" / "derived" / "qwen3_emb_0p6b_v1"
    Avec = np.asarray(np.load(store / "article_vectors.npy"), dtype=np.float32)
    owner = np.load(store / "article_index.npy")
    qrows = json.loads((store / "query_rows.json").read_text(encoding="utf-8"))["release_rows"]
    pos = {int(r): i for i, r in enumerate(qrows)}
    stored = np.load(store / "query_vectors_c.npy", mmap_mode="r")
    for side, tx in sides.items():
        Q = np.concatenate([enc.encode([qe.query_input("c", t) for t in tx[s:s + 8]]) for s in range(0, len(tx), 8)])
        if side == "original":                      # the stored query vectors cover most of the sample, not all of it
            have = [i for i, r in enumerate(rows) if r in pos]
            ref = np.asarray(stored[[pos[rows[i]] for i in have]], dtype=np.float32)
            res["encoder_fidelity_min_cos"] = float((Q[have] * ref).sum(1).min())
            res["encoder_fidelity_cases"] = len(have)
        S = np.empty((len(rows), len(d.candidate_ids)), dtype=np.float32)
        for s, block in ep.maxp_scores(Q, Avec, owner, len(d.candidate_ids)):
            S[s:s + len(block)] = block
        f = L.score_ranked(rows, _ranked(S, rows))
        f.to_parquet(out_dir / f"encoder_{side}.parquet")
        res[f"encoder_{side}"] = float(f["ndcg_at_10"].mean())
    A.receipt("why_controls", res)
    print(json.dumps(res, indent=1), flush=True)
    return res


# --------------------------------------------------------------------------- W5 and the report
AUDIT_STORE = A.ROOT / "experiments" / "runs" / "audit_d181_20261028" / "dev" / "store"


def by_year() -> dict:
    """W5: on every development case, the model alone and the ranker by year of publication (no calls)."""
    from uol.round3 import prompt_guard
    meta = prompt_guard.metadata()
    a = A.per_case(FT.STORE, "gpt61sol_alone")
    r = A.per_case(FT.STORE, "rt_lm_tesis")      # the thesis's ranker
    f = a[["row", "ndcg_at_10"]].merge(r[["row", "ndcg_at_10"]], on="row", suffixes=("_alone", "_ranker"))
    f["year"] = [int((meta.get(int(x)) or {}).get("publication_date", "0000")[:4] or 0) for x in f["row"]]
    g = f.groupby("year").agg(cases=("row", "size"), alone=("ndcg_at_10_alone", "mean"), ranker=("ndcg_at_10_ranker", "mean"))
    g["lead"] = g["alone"] - g["ranker"]
    lead = f["ndcg_at_10_alone"] - f["ndcg_at_10_ranker"]
    slope = float(np.polyfit(f["year"], lead, 1)[0]) if f["year"].nunique() > 1 else None
    return {"by_year": {int(y): {k: (float(v) if k != "cases" else int(v)) for k, v in row.items()} for y, row in g.iterrows()},
            "lead_slope_per_year": slope, "cases": int(len(f))}


def _frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _drop(orig: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    m = orig.merge(new[["row", "ndcg_at_10"]], on="row", suffixes=("", "_new"))
    m["ndcg_at_10"] = m["ndcg_at_10"] - m["ndcg_at_10_new"]
    return m[["row", "query_id", "component_id", "ndcg_at_10"]]


def report() -> dict:
    rows = sample_rows()
    keep = set(rows)
    med = _frame(L.OUT / "alone" / "per_case.parquet")
    lst = A.per_case(FT.STORE, "S0+qwen3_8b")
    lst = lst[lst["row"].isin(keep)]
    rec = json.loads(json.dumps({"population": "DEV-A300", "cases": len(rows), "list_28_same_cases": float(lst["ndcg_at_10"].mean())}))
    # W1
    eff = {e: OUT / f"alone_gpt61sol_{e}" / "per_case.parquet" for e in ("low", "high", "xhigh", "max")}
    have = {e: _frame(p) for e, p in eff.items() if p.exists()}
    rec["w1_alone_effort"] = {"medium": float(med["ndcg_at_10"].mean()), **{e: float(f["ndcg_at_10"].mean()) for e, f in have.items()}}
    if have:
        rec["w1_contrasts"] = A.dev_family("why_w1", [(f"alone {e}", f, "alone medium", med) for e, f in have.items()],
                                           population="DEV-A300", size=4)["contrasts"]
    full = {k: A.per_case(FT.STORE, f"gpt61sol{k}_rerank_S0+qwen3_8b") for k in ("_low", "", "_high")}
    rec["w1_reorder_effort_all_dev"] = {(k.strip("_") or "medium"): float(f["ndcg_at_10"].mean()) for k, f in full.items()}
    rec["w1_reorder_effort_sample_gpt6sol"] = json.loads((FT.D.DEV / "contrasts_100k" / "effort_sample.json").read_text(encoding="utf-8"))["contrasts"] \
        if hasattr(FT, "D") else None
    # W2
    # each Claude model reordering the 28-source ranker's list at the same effort, on the same sample (footing store)
    on_list = {("opus", "medium"): "opus_medium_rerank_S0+qwen3_8b", ("opus", "high"): "opus_high_rerank_S0+qwen3_8b",
               ("sonnet", "medium"): "sonnet_rerank_S0+qwen3_8b", ("haiku", "medium"): "haiku_rerank_S0+qwen3_8b"}
    rec["w2_claude"] = {}
    for m, e in (("opus", "low"), ("opus", "medium"), ("opus", "high"), ("sonnet", "medium"), ("haiku", "medium")):
        p = OUT / f"alone_{m}_{e}" / "per_case.parquet"
        if p.exists():
            f = _frame(p)
            ol = A.per_case(FT.STORE, on_list[(m, e)], "DEV-A300") if (m, e) in on_list else None
            ol = ol[ol["row"].isin(set(f["row"]))] if ol is not None else None
            r = json.loads((A.RUN / "receipts" / f"why_alone_{m}_{e}.json").read_text(encoding="utf-8")) \
                if (A.RUN / "receipts" / f"why_alone_{m}_{e}.json").exists() else {}
            rec["w2_claude"][f"{m}_{e}"] = {"alone": float(f["ndcg_at_10"].mean()),
                                           "on_list_same_cases": float(ol["ndcg_at_10"].mean()) if ol is not None else None,
                                           "unanswered": r.get("unanswered"), "usage": r.get("usage")}
    # W3
    alone_of = {"gpt61sol": med, "gpt6solnw": _frame(L.OUT / "alone_gpt6solnw" / "per_case.parquet")}
    if (OUT / "alone_opus_medium" / "per_case.parquet").exists():
        alone_of["opus"] = _frame(OUT / "alone_opus_medium" / "per_case.parquet")
    rec["w3_probe"] = {}
    for m, af in alone_of.items():
        p = OUT / f"probe_{m}" / "per_case.parquet"
        if not p.exists():
            continue
        pf = _frame(p)
        j = af.merge(pf[["row", "recognized", "recognized_loose"]], on="row")
        unrec = j[~j["recognized"]]
        out = {k: float(pf[k].mean()) for k in ("registro_exact", "year_exact", "rubro_given", "recognized", "recognized_loose",
                                                 "rubro_names_cited", "rubro_f1")}
        out.update({"alone_recognized": float(j.loc[j["recognized"], "ndcg_at_10"].mean()) if j["recognized"].any() else None,
                    "alone_unrecognized": float(unrec["ndcg_at_10"].mean()), "n_recognized": int(j["recognized"].sum()),
                    "list_unrecognized": float(lst[lst["row"].isin(set(unrec["row"]))]["ndcg_at_10"].mean())})
        out["contrast_unrecognized_alone_minus_list"] = A.dev_family(f"why_w3_{m}", [(f"{m} alone, unrecognized", unrec, "28-source list", lst)],
                                                                     population="DEV-A300 unrecognized")["contrasts"]
        rec["w3_probe"][m] = out
    # W4
    rw = OUT / "reword_alone_gpt61sol_medium" / "per_case.parquet"
    cdir = OUT / "controls"
    if rw.exists() and (cdir / "encoder_reworded.parquet").exists():
        new = _frame(rw)
        orig = med[med["row"].isin(set(new["row"]))]
        drops = {"model": _drop(orig, new)}
        for c in ("encoder", "bm25"):
            drops[c] = _drop(_frame(cdir / f"{c}_original.parquet"), _frame(cdir / f"{c}_reworded.parquet"))
        rec["w4_reword"] = {"model_original": float(orig["ndcg_at_10"].mean()), "model_reworded": float(new["ndcg_at_10"].mean()),
                            **json.loads((A.RUN / "receipts" / "why_controls.json").read_text(encoding="utf-8")),
                            "reword": json.loads((A.RUN / "receipts" / "why_reword.json").read_text(encoding="utf-8")),
                            "contrasts": A.dev_family("why_w4", [("model drop", drops["model"], f"{c} drop", drops[c]) for c in ("encoder", "bm25")],
                                                      population="DEV-A300 reworded")["contrasts"]}
    rec["w5"] = by_year()
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "results.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
    print(json.dumps(rec, indent=1, default=float)[:6000], flush=True)
    return rec


# --------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["alone", "probe", "reword", "reword-alone", "controls", "report"])
    ap.add_argument("--model", default="gpt61sol")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if not a.go:
        print(f"[dry] would run {a.step} with {a.model} at {a.effort} on DEV-A300")
        return 0
    FT.D.activate(FT.STORE)                                 # stored row numbers are the bundle's
    if a.step == "alone":
        alone(a.model, a.effort)
    elif a.step == "probe":
        probe(a.model, a.effort)
    elif a.step == "reword":
        reword("sonnet", a.effort)
    elif a.step == "reword-alone":
        t = reword_texts()
        alone(a.model, a.effort, texts=t, name=f"reword_alone_{a.model}_{a.effort}")
    elif a.step == "controls":
        controls()
    elif a.step == "report":
        report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
