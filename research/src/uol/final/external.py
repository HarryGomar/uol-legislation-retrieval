"""C10 and Jev (D101, 23 Oct 2026): two external second stages over S_full_ext's top 30, on DEV-A300 and NEW.

* Claude (``claude_rerank_S_full_ext``): the main LLM rerank's prompt, windows (20, stride 10), both passes and
  fusion (``llm_jobs.fuse_orders``), one sample, through the Anthropic Batch API.
* Jev (``jev_S_full_ext`` alone, ``jev_rrf_S_full_ext`` fused by reciprocal rank, k 60): the jev_probe request
  (masked facts as the state, one yes/no question per candidate article) for the same top 30.

The API calls run on the thesis PC (``drive_claude`` / ``drive_jev``; D101), reading the keys from the local
``.env`` in the API step only; nothing here writes a key. Their outputs are plain data files (orders, probabilities,
usage) that go to ``UOL_RUN_B/inputs`` and, on RUN-B, into ``<work>/external``, where ``fuse`` (plan jobs
``ext__claude`` / ``ext__jev``) turns them into LLM matrices scored like every other system.

LLM sweep (D174, 28 Oct 2026; exploratory, DEV-A300 only): the Claude treatment with another model and nothing else
changed. Claude Sonnet 5.5 and Haiku 5.5 go through the same Batch API step at their own prices and caps; GPT-6 Sol
goes through the Codex CLI (``round3.luna``, its answers cached under its own model in the key). Each writes its own
orders file (``FILES[kind]``), fused on RUN-B by ``fuse(kind)`` like the Opus run.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from uol.final import context, guard

BASE = "S_full_ext"
SYSTEMS = {"claude": "claude_rerank_S_full_ext", "jev": "jev_S_full_ext", "jev_rrf": "jev_rrf_S_full_ext"}
FILES = {"claude": "claude_orders_s0.json", "jev": "jev_p_yes.jsonl"}
# max_tokens 4000: answers use 67-337 output tokens; the smoke test (23 Oct) had one runaway reply bill all 16,000
CLAUDE = {"model": "claude-opus-5-5", "effort": "medium", "max_tokens": 4000, "cap_usd": 60.0,
          "usd_per_mtok_in": 2.0, "usd_per_mtok_out": 10.0, "pricing": "Batch API, 50 % of $4 / $20 per MTok (cached 6 Oct 2026)"}
# LLM sweep (D174): prices from platform.claude.com/docs/en/about-claude/pricing and the developers.openai.com model
# page, read 28 Oct 2026; the Codex route bills no tokens, its USD is the Standard list-price equivalent (as \priceTok)
SWEEP = {
    "sonnet": {**CLAUDE, "model": "claude-sonnet-5-5", "cap_usd": 12.07, "usd_per_mtok_in": 1.0, "usd_per_mtok_out": 5.0,
               "sync_per_mtok": [2.0, 10.0], "pricing": "Batch API, 50 % of $2 / $10 per MTok (read 28 Oct 2026)"},
    "haiku": {**CLAUDE, "model": "claude-haiku-5-5", "cap_usd": 0.60, "usd_per_mtok_in": 0.05, "usd_per_mtok_out": 0.25,
              "sync_per_mtok": [0.10, 0.50], "pricing": "Batch API, 50 % of $0.10 / $0.50 per MTok, prompts up to 100K "
                                                        "tokens (read 28 Oct 2026)"},
    "gpt6sol": {"model": "gpt-6-sol", "effort": "medium", "route": "codex", "workers": 6,
                "list_per_mtok": {"input": 2.0, "cached_input": 0.20, "output": 10.0},
                "pricing": "Codex CLI subscription (no per-token bill); list-price equivalent at the Standard tier, "
                           "$2 / $0.20 cached / $10 per MTok (read 28 Oct 2026)"},
}
SYSTEMS.update({k: f"{k}_rerank_S_full_ext" for k in SWEEP})
FILES.update({k: f"{k}_orders_s0.json" for k in SWEEP})
JEV = {"depth": 30, "cap_usd": 10.0, "usd_per_mtok_in": 0.042, "rrf_k": 60}
TOP = 30


def ext_dir() -> Path:
    p = context.active().work / "external"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- thesis PC: inputs


def top30(top200: Path, rows: Sequence[int]) -> dict[int, list[int]]:
    """S_full_ext's top 30 per row, from its run's ``top200.parquet`` (as ``llm_jobs.base_matrix`` orders it)."""
    import pandas as pd
    from uol.embedding_push import common as ep
    d = ep.release()
    t = pd.read_parquet(top200)
    want = {d.queries[int(r)].query_id: int(r) for r in rows}
    out: dict[int, list[int]] = {}
    for qid, g in t[t["query_id"].isin(want)].groupby("query_id"):
        out[want[qid]] = [int(c) for c in g.sort_values("rank")["candidate_idx"].head(TOP)]
    missing = [r for r in rows if int(r) not in out]
    if missing:
        raise ValueError(f"{len(missing)} rows have no S_full_ext ranking in {top200}")
    return out


# --------------------------------------------------------------------------- thesis PC: Claude


def cache_key(model: str, prompt: str, schema: dict, effort: str) -> str:
    return hashlib.sha256(json.dumps([model, prompt, schema, effort], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def venv_api(workdir: Path, venv_python: Path, env_file: Path, cfg: dict = CLAUDE) -> Callable[[list[dict]], dict[str, dict]]:
    """The API step: ``scripts/c10_api.py`` in the venv that holds the Anthropic SDK (submit one batch, wait,
    collect; cached; spend cap)."""
    openai_route = cfg.get("route") == "openai"
    script = Path(__file__).resolve().parents[3] / "scripts" / ("openai_api.py" if openai_route else "c10_api.py")

    def call(requests: list[dict]) -> dict[str, dict]:
        req = workdir / "requests.jsonl"
        res = workdir / "responses.jsonl"
        req.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in requests), encoding="utf-8")
        cmd = [str(venv_python), str(script), "--requests", str(req), "--responses", str(res), "--cache",
               str(workdir / "cache.jsonl"), "--ledger", str(workdir / "ledger.json"), "--cap", str(cfg["cap_usd"]),
               "--env", str(env_file)]
        if openai_route:                            # standard prices; the script halves them for a batch
            cmd += ["--prices", ",".join(str(x) for x in cfg["prices"])] + (["--sync"] if cfg.get("sync") else [])
        elif cfg is not CLAUDE:                     # the sweep's models: their own prices (D174)
            cmd += ["--price-batch", f"{cfg['usd_per_mtok_in']},{cfg['usd_per_mtok_out']}",
                    "--price-sync", ",".join(str(x) for x in cfg["sync_per_mtok"])]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        (workdir / "api.log").open("a", encoding="utf-8").write(r.stdout + r.stderr)
        if r.returncode != 0:
            raise RuntimeError(f"c10_api failed ({r.returncode}); see {workdir / 'api.log'}")
        return {x["custom_id"]: x for x in map(json.loads, res.read_text(encoding="utf-8").splitlines())}
    return call


def ask_claude(items: list[tuple[str, int, list[int]]], api: Callable, stats: dict, cfg: dict = CLAUDE) -> dict[str, list[int]]:
    """``l1_luna.ask`` with Claude: the same prompt template, batching (5 cases per prompt), schema, single-case
    retries for cases left out, and repair; a case still unanswered after 3 single retries keeps its displayed order."""
    from uol.round3 import l1_luna, luna
    tpl = l1_luna.PROMPT.read_text(encoding="utf-8")
    schema = luna.batch_schema(l1_luna.ITEM_SCHEMA)

    def reqs(parts, tag, sample=0):
        # a retry is an independent sample (its own cache key), as luna's sample > 0
        return [{"custom_id": f"{tag}_{i}", "prompt": l1_luna._build(p, False, tpl), "schema": schema,
                 "model": cfg["model"], "effort": cfg["effort"], "max_tokens": cfg.get("max_tokens"),
                 **({"sample": sample} if sample else {})} for i, p in enumerate(parts)]

    def parse(res: dict | None) -> dict[str, dict]:
        """The first complete JSON object of the reply (a reply cut at max_tokens may already hold it whole)."""
        if not res or res.get("stop_reason") not in ("end_turn", "stop_sequence", "max_tokens") or not res.get("text"):
            return {}
        try:
            out, _ = json.JSONDecoder().raw_decode(res["text"].lstrip())
        except ValueError:
            return {}
        if not isinstance(out, dict):
            return {}
        return {x["id"]: x for x in out.get("results", []) if isinstance(x, dict) and "id" in x}

    parts = [items[s:s + l1_luna.PER_CALL] for s in range(0, len(items), l1_luna.PER_CALL)]
    tag = items[0][0][:4] if items else "x"
    batch = reqs(parts, f"{tag}b")
    got_res = api(batch)
    got: dict[str, dict] = {}
    missing = []
    for part, r in zip(parts, batch):
        rec = parse(got_res.get(r["custom_id"]))
        for it in part:
            (got.__setitem__(it[0], rec[it[0]]) if it[0] in rec else missing.append(it))
    for attempt in range(3):
        if not missing:
            break
        single = reqs([[it] for it in missing], f"{tag}s{attempt}", sample=attempt + 1)
        res = api(single)
        still = []
        for it, r in zip(missing, single):
            rec = parse(res.get(r["custom_id"]))
            (got.__setitem__(it[0], rec[it[0]]) if it[0] in rec else still.append(it))
        missing = still
    stats["unanswered"] = stats.get("unanswered", 0) + len(missing)
    out, repairs = {}, 0
    for qid, _, cands in items:
        order, seen = [], set()
        for n in (got.get(qid) or {}).get("orden", []):
            if isinstance(n, int) and 1 <= n <= len(cands) and n not in seen:
                order.append(n)
                seen.add(n)
        miss = [n for n in range(1, len(cands) + 1) if n not in seen]
        repairs += bool(miss)
        out[qid] = [cands[n - 1] for n in order + miss]
    stats["repairs"] = stats.get("repairs", 0) + repairs
    return out


def codex_api(cfg: dict, tag: str, usage: dict[str, dict]) -> Callable[[list[dict]], dict[str, dict]]:
    """The sweep's Codex route (GPT-6 Sol): each request through ``round3.luna.call`` with the sweep's model (its own
    cache key and ledger rows), ``workers`` at a time. A call still failing after luna's retries comes back without
    text, so ``ask_claude`` re-asks its cases one by one; a quota error stops the run (rerun to resume: answered calls
    are cached). ``usage`` collects the stored usage once per cache key."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from uol.round3 import luna

    def call(requests: list[dict]) -> dict[str, dict]:
        slots, lock = list(range(int(cfg["workers"]))), threading.Lock()

        def one(r: dict) -> tuple[str, dict]:
            sample = int(r.get("sample") or 0)
            with lock:
                w = slots.pop()
            try:
                res = luna.call(r["prompt"], r["schema"], effort=r["effort"], tag=tag, model=r["model"], sample=sample,
                                worker=w)
            except luna.LunaQuotaError:
                raise
            except luna.LunaError:
                return r["custom_id"], {"text": None, "stop_reason": "error"}
            finally:
                with lock:
                    slots.append(w)
            usage[luna.key_of(r["prompt"], r["schema"], r["model"], r["effort"], sample)] = res.get("usage") or {}
            return r["custom_id"], {"text": json.dumps(res["output"], ensure_ascii=False), "stop_reason": "end_turn"}

        with ThreadPoolExecutor(max_workers=int(cfg["workers"])) as pool:
            return dict(pool.map(one, requests))
    return call


def codex_usage(usage: dict[str, dict], cfg: dict) -> tuple[dict[str, int], float]:
    """The Codex CLI usage fields summed over distinct calls, and their list-price equivalent (cached input apart)."""
    tot = {k: int(sum((u.get(k) or 0) for u in usage.values()))
           for k in ("input_tokens", "cached_input_tokens", "output_tokens", "reasoning_output_tokens")}
    p = cfg["list_per_mtok"]
    usd = ((tot["input_tokens"] - tot["cached_input_tokens"]) * p["input"] + tot["cached_input_tokens"] * p["cached_input"]
           + tot["output_tokens"] * p["output"]) / 1e6
    return {"calls": len(usage), **tot}, round(usd, 4)


def rerank_orders(rows: Sequence[int], top: dict[int, list[int]], ask: Callable) -> dict[str, Any]:
    """``l1_luna.run`` (main variant) with an injected ``ask``: window A = ranks 11-30, window B = ranks 1-10 plus A's
    first 10, in the forward and the reversed display order."""
    final: dict[str, dict[int, list[int]]] = {}
    for p in ("fwd", "rev"):
        show = (lambda lst: lst[::-1]) if p == "rev" else (lambda lst: lst)
        a = ask([(f"{p}A{r}", r, show(top[r][10:30])) for r in rows])
        b = ask([(f"{p}B{r}", r, show(top[r][:10] + a[f"{p}A{r}"][:10])) for r in rows])
        final[p] = {r: b[f"{p}B{r}"] + a[f"{p}A{r}"][10:] for r in rows}
    return {"variant": "claude", "rows": [int(r) for r in rows], "passes": ["fwd", "rev"],
            "orders": {p: {str(r): o for r, o in v.items()} for p, v in final.items()}}


def drive_claude(top200: Path, workdir: Path, venv_python: Path, env_file: Path, rows: Sequence[int] | None = None,
                 kind: str = "claude") -> Path:
    """``kind`` "claude" is the Opus run (D101); the sweep's kinds (``SWEEP``, D174) change only the model."""
    from uol.final import llm_jobs, populations
    from uol.embedding_push import common as ep
    from uol.round3 import l1_luna
    workdir.mkdir(parents=True, exist_ok=True)
    subset = rows is not None
    rows = [int(r) for r in (populations.agent_sample(ep.dev_rows()) if rows is None else guard.dev_rows_only(rows))]
    top = top30(top200, rows)
    stats: dict[str, int] = {}
    cfg = CLAUDE if kind == "claude" else SWEEP[kind]
    usage: dict[str, dict] = {}
    api = codex_api(cfg, f"sweep_{kind}", usage) if cfg.get("route") == "codex" else venv_api(workdir, venv_python, env_file, cfg)
    o = rerank_orders(rows, top, lambda items: ask_claude(items, api, stats, cfg))
    ledger = json.loads((workdir / "ledger.json").read_text(encoding="utf-8")) if (workdir / "ledger.json").exists() else {}
    o.update({"model": cfg["model"], "effort": cfg["effort"], **stats,
              "prompt_sha256": hashlib.sha256(l1_luna.PROMPT.read_bytes()).hexdigest(), "population": "DEV-A300",
              "usage": ledger.get("usage"), "usd": ledger.get("usd"), "pricing": cfg["pricing"]})
    if kind != "claude":
        o.update({"variant": kind, "notes": "added after the freeze (D174), exploratory, DEV-A300 only"})
        if subset:
            o["population"] = f"DEV-A300 subset ({len(rows)} rows)"
        if cfg.get("route") == "codex":
            o["usage"], o["usd_list_equivalent"] = codex_usage(usage, cfg)
            o["usd"] = 0.0
    path = workdir / FILES[kind]
    path.write_text(json.dumps(o, indent=1), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- thesis PC: Jev


def read_env(env_file: Path, name: str) -> str:
    """One value of the local .env (never printed, logged or written)."""
    for line in Path(env_file).read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == name:
            return v.strip().strip('"').strip("'")
    raise KeyError(f"{name} is not set in {env_file}")


def jev_request(facts: str, cands: list[int], texts: list[str], key: str) -> dict:
    import time
    import urllib.error
    import urllib.request
    from uol.round3 import jev_probe as J
    qs = {f"a{i:02d}": {"type": "noul", "instructions": J.INSTRUCTIONS + texts[int(a)][:J.ART_CHARS], "criteria": J.CRITERIA}
          for i, a in enumerate(cands)}
    body = json.dumps({"model": J.MODEL, "state": {"facts": facts}, "questions": qs}).encode()
    for attempt in range(6):
        req = urllib.request.Request(J.URL, data=body, method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < 5:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Jev HTTP {e.code}") from None
    return {"p_yes": [float(out["answers"][f"a{i:02d}"]["noul"]) for i in range(len(cands))], "model": out.get("model"),
            "usage": out.get("usage"), "seconds": round(time.perf_counter() - t0, 2)}


def drive_jev(top200: Path, workdir: Path, env_file: Path, rows: Sequence[int] | None = None,
              request: Callable = jev_request) -> Path:
    """One Jev request per case (its top 30), cached per row; stops before a request that could pass the cap."""
    from uol.final import populations
    from uol.embedding_push import common as ep
    from uol.round3 import jev_probe as J
    workdir.mkdir(parents=True, exist_ok=True)
    d = ep.release()
    rows = [int(r) for r in (populations.agent_sample(ep.dev_rows()) if rows is None else guard.dev_rows_only(rows))]
    top = top30(top200, rows)
    texts = J.article_texts()
    path = workdir / FILES["jev"]
    done = {json.loads(x)["row"]: json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()} if path.exists() else {}
    spent = sum(((x.get("usage") or {}).get("input_tokens") or 0) for x in done.values()) * JEV["usd_per_mtok_in"] / 1e6
    key = read_env(env_file, "JEV_KEY") if any(r not in done for r in rows) else ""
    for r in rows:
        if r in done:
            continue
        est = sum(len(texts[int(c)][:J.ART_CHARS]) for c in top[r]) / 3.0 * JEV["usd_per_mtok_in"] / 1e6
        if spent + 2 * est > JEV["cap_usd"]:
            raise RuntimeError(f"Jev spend cap: {spent:.2f} spent, next request could pass US${JEV['cap_usd']}")
        rec = {"row": r, "candidates": top[r], **request(d.queries[r].text, top[r], texts, key)}
        spent += ((rec.get("usage") or {}).get("input_tokens") or 0) * JEV["usd_per_mtok_in"] / 1e6
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    return path


# --------------------------------------------------------------------------- RUN-B: fusion and matrices


def fuse(kind: str, store: Path | None = None) -> list[Path]:
    """RUN-B: the external outputs in ``<work>/external`` become LLM matrices (``llm_jobs.save_matrix``)."""
    from uol.final import llm_jobs
    prof = context.active()
    store = Path(store) if store else prof.store
    src = ext_dir() / FILES[kind]
    if not src.exists():
        raise FileNotFoundError(f"{src} missing: copy UOL_RUN_B\\inputs\\{FILES[kind]} into <Root>\\work\\external first")
    if kind == "claude" or kind in SWEEP:
        o = json.loads(src.read_text(encoding="utf-8"))
        rows = guard.dev_rows_only(np.asarray(o["rows"], dtype=np.int64), "claude rows")
        S = llm_jobs.base_matrix(store, f"{BASE}{prof.tag}", rows)
        recipe = ("llm_rerank main prompt and windows, Claude (Batch API), one sample, fusion_route W_EXT (C10, D101)"
                  if kind == "claude" else f"llm_rerank main prompt and windows, {o.get('model')} ({SWEEP[kind]['pricing']}), "
                                           "one sample, fusion_route W_EXT (LLM sweep, D174)")
        return [llm_jobs.fuse_orders(SYSTEMS[kind], BASE, rows, [o], S,
                                     {"samples": 1, "variant": o.get("variant", "claude"), "model": o.get("model"), "usage": o.get("usage"),
                                      "usd": o.get("usd"), "unanswered": o.get("unanswered"), "repairs": o.get("repairs"),
                                      "recipe": recipe})]
    recs = {int(x["row"]): x for x in map(json.loads, src.read_text(encoding="utf-8").splitlines())}
    rows = guard.dev_rows_only(np.asarray(sorted(recs), dtype=np.int64), "jev rows")
    S = llm_jobs.base_matrix(store, f"{BASE}{prof.tag}", rows)
    alone, rrf = S.copy(), S.copy()
    for i, r in enumerate(rows):
        c = np.asarray(recs[int(r)]["candidates"])
        p = np.asarray(recs[int(r)]["p_yes"])
        jev_rank = np.empty(len(c))
        jev_rank[np.lexsort((np.arange(len(c)), -p))] = np.arange(1, len(c) + 1)
        base_rank = np.arange(1, len(c) + 1)                    # the candidates are in S_full_ext's order
        alone[i, c] = 1e6 - jev_rank
        rrf[i, c] = 1e6 + 1.0 / (JEV["rrf_k"] + base_rank) + 1.0 / (JEV["rrf_k"] + jev_rank)
    usage = sum(((x.get("usage") or {}).get("input_tokens") or 0) for x in recs.values())
    meta = {"base": BASE, "depth": JEV["depth"], "input_tokens": usage, "models": sorted({str(x.get("model")) for x in recs.values()}),
            "recipe": "jev_probe request over the S_full_ext top 30 (C10, D101)"}
    return [llm_jobs.save_matrix(SYSTEMS["jev"], rows, alone, {**meta, "fusion": "none (Jev order on the top 30)"}),
            llm_jobs.save_matrix(SYSTEMS["jev_rrf"], rows, rrf, {**meta, "fusion": f"RRF k {JEV['rrf_k']} with S_full_ext"})]


def main(argv: list[str] | None = None) -> int:
    """Thesis PC: ``py -3.12 -m uol.final.external claude|jev --top200 <S_full_ext DEV top200.parquet> --out <dir>``."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["claude", "jev", *SWEEP])
    ap.add_argument("--rows", default=None, help="comma-separated DEV rows (the sweep's smoke test); default DEV-A300")
    ap.add_argument("--top200", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[3] / ".cache/final_bundle_staging/data"))
    ap.add_argument("--env", default=str(Path(__file__).resolve().parents[3] / ".env"))
    ap.add_argument("--venv-python", default=str(Path(__file__).resolve().parents[3] / ".cache/venvs/c10/Scripts/python.exe"))
    a = ap.parse_args(argv)
    out = Path(a.out)
    context.activate_bundle(Path(a.data), store=out / "store", work=out / "work")   # RUN-B's labels and release
    if a.kind == "claude":
        print(drive_claude(Path(a.top200), out / "claude", Path(a.venv_python), Path(a.env)))
    elif a.kind in SWEEP:
        rows = [int(x) for x in a.rows.split(",")] if a.rows else None
        print(drive_claude(Path(a.top200), out / a.kind, Path(a.venv_python), Path(a.env), rows=rows, kind=a.kind))
    else:
        print(drive_jev(Path(a.top200), out / "jev", Path(a.env)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
