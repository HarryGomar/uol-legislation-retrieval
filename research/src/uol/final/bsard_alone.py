"""The model alone on BSARD's test split, asked once (research/programs/bsard_outside_20261105/PLAN.md; FRAMING decision 3(d)).

An outside benchmark, in its own folders: it touches neither the thesis's benchmark nor its test set. The resolver below
is built from the corpus names only and was fixed before any answer or label was read.

    py -3.12 -m uol.final.bsard_alone resolver-check      no model call: corpus names and abbreviations resolve to themselves
    py -3.12 -m uol.final.bsard_alone run --go            asks once (cached), resolves, scores, writes result.json
"""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "external" / "bsard_20261105"
PROG = ROOT / "research" / "programs" / "bsard_outside_20261105"
TEMPLATE = PROG / "prompt_fr.txt"
TAG = "bsard_alone_test"
EFFORT = "medium"
BATCH = 10
DRAWS, SEED = 10_000, 20261105
ITEM = {"type": "object", "additionalProperties": False, "required": ["articles"],
        "properties": {"articles": {"type": "array", "minItems": 20, "maxItems": 20, "items": {
            "type": "object", "additionalProperties": False, "required": ["loi", "article"],
            "properties": {"loi": {"type": "string"}, "article": {"type": "string"}}}}}}
STOP = {"de", "du", "des", "la", "le", "les", "l", "d", "et", "au", "aux", "a", "en", "pour", "sur", "belge"}
ALIASES = {"constitution": "La Constitution", "constitution belge": "La Constitution", "la constitution belge": "La Constitution",
           "c civ": "Code Civil", "cciv": "Code Civil", "ancien code civil": "Code Civil", "nouveau code civil": "Code Civil", "code civil belge": "Code Civil",
           "c pen": "Code Pénal", "cp": "Code Pénal", "c jud": "Code Judiciaire", "cj": "Code Judiciaire",
           "cic": "Code d'Instruction Criminelle", "csa": "Code des Sociétés et des Associations", "cde": "Code de Droit Economique",
           "cdld": "Code de la Démocratie Locale et de la Décentralisation", "cobat": "Code Bruxellois de l'Aménagement du Territoire",
           "codt": "Code Wallon du Développement Territorial", "cwadel": "Code de la Démocratie Locale et de la Décentralisation",
           "cwass": "Code Wallon de l'Action sociale et de la Santé", "cwhd": "Code Wallon de l'Habitation Durable",
           "cbl": "Code Bruxellois du Logement", "code du logement bruxellois": "Code Bruxellois du Logement",
           "codip": "Code de Droit International Privé", "cdip": "Code de Droit International Privé", "cps": "Code Pénal Social",
           "cbet": "Code du Bien-être au Travail", "code du bien etre": "Code du Bien-être au Travail"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", s)).strip()


def art_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode("ascii").lower().strip()
    s = re.sub(r"^(l['\s]*)?(art(icle)?s?\.?)\s*", "", s)
    s = re.split(r"[,;]|\s+(?:par|al|alinea|paragraphe|l\.)\b|\(", s)[0]
    s = re.sub(r"[.\s-]", "", s)
    return re.sub(r"^(\d+)(er|e)$", r"\1", s)


class Corpus:
    def __init__(self):
        csv.field_size_limit(10**8)
        with open(DATA / "articles.csv", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        self.n = len(rows)
        self.codes = sorted({r["code"] for r in rows})
        self.by_norm = {norm(c): c for c in self.codes}
        self.words = {c: {w for w in norm(c).split() if w not in STOP} for c in self.codes}
        self.aliases = {norm(k): v for k, v in ALIASES.items()}
        missing = [v for v in self.aliases.values() if norm(v) not in self.by_norm]
        if missing:
            raise ValueError(f"aliases outside the corpus: {missing}")
        self.index: dict[tuple[str, str], list[int]] = {}
        for r in rows:
            m = re.match(r"Art\.\s*([^,]+),", r["reference"])
            if m:
                self.index.setdefault((r["code"], art_key(m.group(1))), []).append(int(r["id"]))

    def code(self, name: str) -> str | None:
        n = norm(name)
        if n in self.by_norm:
            return self.by_norm[n]
        if n in self.aliases:
            return self.by_norm[norm(self.aliases[n])]
        have = set(n.split())
        hits = [c for c, w in self.words.items() if w and w <= have]
        if hits:
            return max(hits, key=lambda c: (len(self.words[c]), c))
        close = difflib.get_close_matches(n, list(self.by_norm), n=1, cutoff=0.90)
        return self.by_norm[close[0]] if close else None

    def resolve(self, name: str, article: str) -> list[int]:
        c = self.code(name)
        return self.index.get((c, art_key(article)), []) if c else []


def questions() -> list[dict]:
    with open(DATA / "questions_test.csv", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def measures(ranked: list[int], pos: set[int]) -> dict:
    hit = [c in pos for c in ranked]
    dcg = sum(1 / math.log2(i + 2) for i, h in enumerate(hit[:10]) if h)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(pos), 10)))
    seen, ap, rr = 0, 0.0, 0.0
    for i, h in enumerate(hit[:100]):
        if h:
            seen += 1
            ap += seen / (i + 1)
            rr = rr or 1 / (i + 1)
    return {"ndcg_at_10": dcg / idcg, "map_at_100": ap / len(pos), "r_precision": sum(hit[:len(pos)]) / len(pos), "mrr_at_100": rr,
            "recall_at_100": sum(hit[:100]) / len(pos), "recall_at_20": sum(hit[:20]) / len(pos), "success_at_10": float(any(hit[:10]))}


def ci(v: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    m = v[rng.integers(0, len(v), size=(DRAWS, len(v)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def resolver_check() -> int:
    c = Corpus()
    bad = [k for k in c.codes if c.code(k) != k]
    print(json.dumps({"articles": c.n, "codes": len(c.codes), "keys": len(c.index), "codes_not_resolving_to_themselves": bad,
                      "aliases": {k: c.code(k) for k in list(ALIASES)[:6]},
                      "examples": {"Art. 1382": c.resolve("Code civil", "Art. 1382"), "1er Constitution": c.resolve("Constitution belge", "1er"),
                                   "CDE VII.216/129": c.resolve("Code de droit économique (CDE)", "VII.216/129")}}, indent=1, ensure_ascii=False))
    return 0 if not bad else 1


def run() -> int:
    from uol.final import lean_rerun as LR, llm_agents as G
    LR.D.activate(LR.STORE)
    luna = G.pin_model()
    corpus = Corpus()
    q = questions()
    tpl = TEMPLATE.read_text(encoding="utf-8")
    schema = luna.batch_schema(ITEM)
    built = []
    for s in range(0, len(q), BATCH):
        part = q[s:s + BATCH]
        ids = [f"q{i + 1:02d}" for i in range(len(part))]
        items = "\n\n".join(f"[id={i}] {re.sub(r'\s+', ' ', x['question']).strip()}" for i, x in zip(ids, part))
        built.append((tpl.replace("{items}", items), ids, part))
    res = luna.run_many([(p, schema) for p, _, _ in built], effort=EFFORT, tag=TAG, workers=4)
    answers: dict[str, list[dict]] = {}
    usage = {"input": 0, "output": 0}
    for (prompt, ids, part), r in zip(built, res):
        got = luna.checked(r, prompt, schema, ids, effort=EFFORT, tag=TAG)
        for i, x in zip(ids, part):
            answers[x["id"]] = got[i]["articles"]
        u = r.get("usage") or {}
        usage["input"] += int(u.get("input_tokens") or 0)
        usage["output"] += int(u.get("output_tokens") or 0)
    PROG.mkdir(parents=True, exist_ok=True)
    (PROG / "answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=0), encoding="utf-8")
    per, named, law_hit, art_hit = [], 0, 0, 0
    for x in q:
        pos = {int(a) for a in x["article_ids"].split(",") if a.strip()}
        ranked: list[int] = []
        for a in answers[x["id"]]:
            named += 1
            law_hit += corpus.code(a["loi"]) is not None
            r = corpus.resolve(a["loi"], a["article"])
            art_hit += bool(r)
            ranked += [c for c in r if c not in ranked]
        per.append(measures(ranked, pos))
    keys = list(per[0])
    out = {"questions": len(q), "model": G.MODEL, "effort": EFFORT, "means": {k: float(np.mean([p[k] for p in per])) for k in keys},
           "ci95": {k: ci(np.asarray([p[k] for p in per])) for k in keys},
           "names": named, "share_law_in_corpus": law_hit / named, "share_article_resolved": art_hit / named, "usage": usage,
           "sha256": {"prompt": hashlib.sha256(TEMPLATE.read_bytes()).hexdigest(), "resolver": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    (PROG / "result.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["resolver-check", "run"])
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args(argv)
    if a.step == "resolver-check":
        return resolver_check()
    if not a.go:
        print("[dry] would ask once")
        return 0
    return run()


if __name__ == "__main__":
    sys.exit(main())
