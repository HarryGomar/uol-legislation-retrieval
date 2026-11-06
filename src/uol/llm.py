"""The language-model steps of the full system.

    1. GPT-6.1 Sol reads the facts alone and names 20 articles, most likely first (``prompts/list20.txt``).
    2. Its answers are matched to catalogue articles and merged with the ranker's list by reciprocal rank fusion.
    3. GPT-6.1 Sol reorders the first 30 of the merged list (``prompts/rerank20.txt``), in two windows of 20 from the
       bottom up: positions 11-30, then positions 1-10 with the best ten of the first window. Its order is fused with
       the merged order, and positions after 30 stay as merged.

The model is reached through the OpenAI Responses API or the Codex command line (the route used in the thesis). Each
answer is stored in a JSON-lines cache keyed by model, effort, schema, prompt and sample, so a rerun replays it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from uol.data import Benchmark

PROMPTS = Path(__file__).with_name("prompts")
MODEL, EFFORT = "gpt-6.1-sol", "medium"
PRICES = {"gpt-6.1-sol": (2.0, 0.10, 10.0)}          # USD per million tokens: input, cached input, output
LIST_SIZE, RERANK_DEPTH, WINDOW, RRF_K, FUSE_WEIGHT, EXCERPT = 20, 30, 20, 60, 3.0, 600


def batch_schema(item: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": ["results"],
            "properties": {"results": {"type": "array", "items": {
                "type": "object", "additionalProperties": False, "required": ["id", *item["required"]],
                "properties": {"id": {"type": "string"}, **item["properties"]}}}}}


ARTICLE = {"type": "object", "additionalProperties": False, "required": ["ley", "articulo"],
           "properties": {"ley": {"type": "string"}, "articulo": {"type": "string"}}}
LIST_SCHEMA = batch_schema({"type": "object", "additionalProperties": False, "required": ["articulos"],
                            "properties": {"articulos": {"type": "array", "minItems": LIST_SIZE, "maxItems": LIST_SIZE,
                                                         "items": ARTICLE}}})
ORDER_SCHEMA = batch_schema({"type": "object", "additionalProperties": False, "required": ["orden"],
                             "properties": {"orden": {"type": "array", "items": {"type": "integer"}}}})


def collapse(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------------------------------------- models
class Model(Protocol):
    def ask(self, prompt: str, schema: dict, sample: int = 0) -> dict | None: ...


@dataclass
class Cached:
    """A model route with a cache and a usage record. ``ask`` returns the parsed JSON answer, or None."""
    model: str = MODEL
    effort: str = EFFORT
    cache: Path | None = None
    usage: list[dict] = field(default_factory=list)

    def key(self, prompt: str, schema: dict, sample: int) -> str:
        blob = {"m": self.model, "e": self.effort, "s": schema, "p": prompt, "k": sample}
        return hashlib.sha256(json.dumps(blob, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def ask(self, prompt: str, schema: dict, sample: int = 0) -> dict | None:
        key = self.key(prompt, schema, sample)
        if self.cache and self.cache.exists():
            with self.cache.open(encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    if row["key"] == key:
                        return row["output"]
        output, usage = self.call(prompt, schema)
        self.usage.append(usage)
        if self.cache:
            self.cache.parent.mkdir(parents=True, exist_ok=True)
            with self.cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": self.model, "output": output, "usage": usage}, ensure_ascii=False) + "\n")
        return output

    def call(self, prompt: str, schema: dict) -> tuple[dict | None, dict]:
        raise NotImplementedError

    def cost(self) -> float:
        p_in, p_cached, p_out = PRICES[self.model]
        total = 0.0
        for u in self.usage:
            n_in, cached, n_out = u.get("input_tokens", 0), u.get("cached_input_tokens", 0), u.get("output_tokens", 0)
            total += ((n_in - cached) * p_in + cached * p_cached + n_out * p_out) / 1e6
        return round(total, 4)


class OpenAIResponses(Cached):
    """The OpenAI Responses API with a strict JSON schema; the key is read by the client from ``OPENAI_API_KEY``."""

    def call(self, prompt: str, schema: dict) -> tuple[dict | None, dict]:
        from openai import OpenAI
        body = OpenAI().responses.create(
            model=self.model, input=prompt, store=False, reasoning={"effort": self.effort},
            text={"format": {"type": "json_schema", "name": "answer", "strict": True, "schema": schema}})
        u = body.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cached_input_tokens": getattr(u.input_tokens_details, "cached_tokens", 0) or 0}
        try:
            return json.loads(body.output_text), usage
        except (TypeError, ValueError):
            return None, usage


class CodexCLI(Cached):
    """``codex exec`` in a read-only sandbox with tools, web search and user configuration off (the thesis route).
    The binary is ``$UOL_CODEX_BIN`` or ``codex`` on the path; the user signs in to Codex beforehand."""

    DISABLED = ("shell_tool", "unified_exec", "apps", "plugins", "browser_use", "browser_use_external", "computer_use",
                "image_generation", "multi_agent", "skill_search", "hooks", "view_image", "tool_suggest", "goals",
                "in_app_browser", "sleep_tool", "shell_snapshot", "workspace_dependencies", "remote_plugin")
    timeout, retries = 900, 4

    def call(self, prompt: str, schema: dict) -> tuple[dict | None, dict]:
        binary = os.environ.get("UOL_CODEX_BIN") or shutil.which("codex") or "codex"
        delay = 20.0
        for attempt in range(self.retries + 1):
            with tempfile.TemporaryDirectory() as tmp:
                work, schema_path = Path(tmp) / "work", Path(tmp) / "schema.json"
                work.mkdir()
                schema_path.write_text(json.dumps(schema), encoding="utf-8")
                cmd = [binary, "exec", "-m", self.model, "--sandbox", "read-only", "--skip-git-repo-check", "--ephemeral",
                       "--ignore-user-config", "-c", f'model_reasoning_effort="{self.effort}"', "-c", 'web_search="disabled"',
                       "--output-schema", str(schema_path), "--json", "--color", "never", "-C", str(work)]
                cmd += [x for feature in self.DISABLED for x in ("--disable", feature)] + ["-"]
                try:
                    proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True, encoding="utf-8",
                                          timeout=self.timeout)
                    message, usage, ok = self.parse(proc.stdout)
                    if proc.returncode == 0 and ok and message and not any(work.iterdir()):
                        return json.loads(message), usage
                except (subprocess.TimeoutExpired, ValueError):
                    pass
            if attempt < self.retries:
                time.sleep(delay)
                delay = min(2 * delay, 600.0)
        return None, {}

    @staticmethod
    def parse(stdout: str) -> tuple[str | None, dict, bool]:
        """The final agent message, the usage, and whether the turn used nothing but reasoning and the message."""
        message, usage, ok = None, {}, True
        for line in stdout.splitlines():
            if not line.strip().startswith("{"):
                continue
            event = json.loads(line)
            kind = event.get("type", "")
            if kind == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    message = item.get("text")
                elif item.get("type") != "reasoning":
                    ok = False
            elif kind == "turn.completed":
                usage = event.get("usage") or {}
            elif kind in ("turn.failed", "error"):
                ok = False
        return message, usage, ok


# ------------------------------------------------------------------------------------------------------ catalogue
STOP = frozenset("de del la las los el y e para en a por sobre con su sus que reglamentaria articulos articulo".split())
ALIASES = {"cpeum": "CPEUM", "constitucion": "CPEUM", "constitucion politica": "CPEUM", "constitucion federal": "CPEUM",
           "carta magna": "CPEUM", "ley de amparo": "LAmp", "codigo civil federal": "CCF", "codigo de comercio": "CCom"}


def fold(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower())
    return re.sub(r"\s+", " ", "".join(ch for ch in t if not unicodedata.combining(ch))).strip()


def tokens(text: str) -> frozenset[str]:
    return frozenset(w for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in STOP)


def article_key(number: str) -> str:
    """"Artículo 31, fracción IV" -> "31"; "1o." -> "1"; "153-F Bis" -> "153-f-bis"."""
    a = re.sub(r"^(art[iı]culo|art\.?)\s*", "", fold(number))
    a = re.split(r",|;|\(|\b(?:fraccion|fracc|parrafo|apartado|inciso|ultimo|primer|segundo|tercer|base|numeral|de las|"
                 r"del|de la)\b", a)[0]
    a = re.sub(r"(\d+)\s*(?:o|º|°)\.?(?=\b|\s|-|$)", r"\1", a)
    return re.sub(r"[\s.]+", "-", a.strip(" .,;")).strip("-")


class Catalogue:
    """Matches a (law, article) answer to a catalogue article. A law is found by alias, by its catalogue code, or by
    the token F1 between the name and the law titles (at least 0.6 and no tie)."""

    def __init__(self, bench: Benchmark) -> None:
        self.titles: dict[str, str] = {}
        self.articles: dict[str, dict[str, int]] = defaultdict(dict)
        for i, (cid, text) in enumerate(zip(bench.article_ids, bench.article_texts)):
            code = cid.rsplit("/", 1)[-1].split(".pdf")[0]
            self.titles.setdefault(code, text.split("\n", 1)[0].strip())
            self.articles[code][cid.split("#articulo=")[-1].lower()] = i
        self.by_code = {c.lower(): c for c in self.titles}
        self.title_tokens = {c: tokens(t) for c, t in self.titles.items()}

    def law(self, name: str) -> str | None:
        f = re.sub(r"\bde los estados unidos mexicanos\b", "", fold(name).strip(" .,;")).strip()
        if f in ALIASES:
            return ALIASES[f]
        if any(f.startswith(alias + " ") for alias, code in ALIASES.items() if code == "CPEUM"):
            return "CPEUM"
        if f.replace(" ", "") in self.by_code:
            return self.by_code[f.replace(" ", "")]
        q = tokens(f)
        scored = []
        for code, t in self.title_tokens.items():
            inter = len(q & t)
            if inter:
                p, r = inter / len(q), inter / len(t)
                scored.append((2 * p * r / (p + r), code))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < 0.6 or (len(scored) > 1 and scored[1][0] == scored[0][0]):
            return None
        return scored[0][1]

    def article(self, law: str, number: str) -> int | None:
        code = self.law(law)
        return None if code is None else self.articles.get(code, {}).get(article_key(number))

    def line(self, bench: Benchmark, i: int) -> str:
        """How the reranker sees a candidate: law title, article label and the first 600 characters of the text."""
        title, _, body = bench.article_texts[i].partition("\n")
        body = collapse(body)
        label = bench.article_ids[i].split("#articulo=")[-1]
        return f"{title.strip()} › Art. {label}: {body[:EXCERPT]}{'…' if len(body) > EXCERPT else ''}"


# ---------------------------------------------------------------------------------------------------------- steps
def model_list(model: Model, catalogue: Catalogue, facts: str, generic: set[int]) -> list[int]:
    """Step 1: the model's 20 articles from the facts alone, as catalogue rows (unmatched and generic ones dropped,
    repeats kept once). A malformed answer is asked again, up to three times."""
    prompt = (PROMPTS / "list20.txt").read_text(encoding="utf-8").replace("{items}", f"[id=c01] {collapse(facts)}")
    for sample in range(4):
        out = model.ask(prompt, LIST_SCHEMA, sample)
        results = {r.get("id"): r for r in (out or {}).get("results", [])}
        if "c01" in results and all("ley" in a and "articulo" in a for a in results["c01"]["articulos"]):
            found = (catalogue.article(a["ley"], a["articulo"]) for a in results["c01"]["articulos"])
            return list(dict.fromkeys(i for i in found if i is not None and i not in generic))
    return []


def rrf(lists: list[list[int]], k: int = RRF_K) -> list[int]:
    """Reciprocal rank fusion; a tie keeps the earlier list's order."""
    score: dict[int, float] = {}
    for lst in lists:
        for rank, c in enumerate(lst, 1):
            score[c] = score.get(c, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda c: -score[c])


def order_window(model: Model, bench: Benchmark, catalogue: Catalogue, facts: str, shown: list[int]) -> list[int]:
    """The model's order of one window; numbers it leaves out follow in the order shown."""
    lines = "\n".join(f"{n}. {catalogue.line(bench, c)}" for n, c in enumerate(shown, 1))
    item = f"[id=w]\nHechos: {collapse(facts)}\nCandidatos:\n{lines}"
    prompt = (PROMPTS / "rerank20.txt").read_text(encoding="utf-8").replace("{items}", item)
    order: list[int] = []
    for sample in range(4):
        results = {r.get("id"): r for r in ((model.ask(prompt, ORDER_SCHEMA, sample) or {}).get("results", []))}
        if "w" in results:
            order = list(dict.fromkeys(n for n in results["w"].get("orden", []) if isinstance(n, int) and 1 <= n <= len(shown)))
            break
    order += [n for n in range(1, len(shown) + 1) if n not in order]
    return [shown[n - 1] for n in order]


def zscore(x: np.ndarray) -> np.ndarray:
    sd = x.std()
    return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)


def fuse(shown: list[int], reordered: list[int], w: float = FUSE_WEIGHT) -> list[int]:
    """z(merged position) + w z(1 / model position), both as scores."""
    position = {c: k for k, c in enumerate(reordered, 1)}
    base = -np.arange(len(shown), dtype=np.float64)
    model = np.asarray([1.0 / position[c] if c in position else 0.0 for c in shown])
    return [shown[i] for i in np.argsort(-(zscore(base) + w * zscore(model)), kind="stable")]


def rerank(model: Model, bench: Benchmark, catalogue: Catalogue, facts: str, merged: list[int]) -> list[int]:
    """Step 3: windows of 20 from the bottom up (11-30, then 1-10 with the best ten below), fused with the merged order."""
    top = merged[:RERANK_DEPTH]
    lower = order_window(model, bench, catalogue, facts, top[10:30])
    upper = order_window(model, bench, catalogue, facts, top[:10] + lower[:10])
    return fuse(top, upper + lower[10:]) + merged[RERANK_DEPTH:]


@dataclass
class SystemAnswer:
    ranked: list[int]
    model_list: list[int]
    merged: list[int]
    final: list[int]


def full_system(model: Model, bench: Benchmark, catalogue: Catalogue, facts: str, ranked: list[int]) -> SystemAnswer:
    listed = model_list(model, catalogue, facts, set(bench.generic.tolist()))
    merged = rrf([listed, ranked])
    return SystemAnswer(ranked, listed, merged, rerank(model, bench, catalogue, facts, merged))
