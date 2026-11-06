"""B step 3: masked citation contexts per article, the historical memory entries, and the random-context control.

Contexts
--------
* For each resolved mention (after the version filter below): the sentence containing the article span plus one
  sentence on each side, cut to at most ``MAX_CHARS`` characters around the mention.
* Masking: every resolved citation expression inside the window (the resolver's witness span, i.e. "artículo N ... de
  la Ley X", and the article span) becomes ``[CITA]``; a residual "artículo(s) <number>" pattern and the target
  instrument's title / safe aliases are masked too, so neither the number nor the law name of the target survives.
* At most ``CAP`` contexts per article, one per tesis, distinct masked texts, spread evenly over publication order
  (diversity across épocas); popular articles therefore gain no extra text.

Version filter
--------------
A historical mention of an instrument that was replaced by a new law *with the same title* refers to the old law's
article numbering. Mentions in tesis published before the in-force date of the catalogue's (current) law are dropped
(``VERSION_CUT``, hand-curated from the DOF history; instruments whose current law postdates the historical cutoff lose
all historical mentions). The table is a documented best effort, not an audited source.

Memory entries
--------------
Every historical tesis with at least one kept mention: ``rubro + "\\n" + texto`` (truncated to 2,000 Qwen tokens when
embedded) and its set of cited catalogue articles.

Control
-------
For article a with n_a contexts: n_a contexts drawn (seeded) from the selected contexts of other random articles of the
same instrument (no new text is embedded). Articles of an instrument with a single context-bearing article get none.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict

import numpy as np

from uol.embedding_push import common
from uol.facts_to_law import references_v2 as R
from uol.facts_to_law.spec import sha256_file
from uol.round2.b_common import CANDIDATES, HIST, MENTIONS, OUT, SEED, log, utc, write_json, write_jsonl
from uol.round2.b_resolve import compose

CAP = 5
MAX_CHARS = 1200
CUTOFF = "2020-08-14"
CONTEXTS = OUT / "contexts.jsonl"           # selected contexts: one row per (article, context)
CONTROL = OUT / "control_contexts.jsonl"    # control assignment: article -> list of context keys
MEMORY = OUT / "memory.jsonl"               # memory entries: record_id, cited candidate indices
CTX_META = OUT / "contexts_summary.json"

# in-force date of the current law for instruments replaced by a same-titled law (mentions before it are dropped)
VERSION_CUT = {
    "LAmp.pdf": "2013-04-03", "LSS.pdf": "1997-07-01", "LISSSTE.pdf": "2007-04-01", "LISR.pdf": "2014-01-01",
    "LFCE.pdf": "2014-07-07", "LFT.pdf": "1970-05-01", "CFF.pdf": "1983-01-01", "LAdua.pdf": "1996-04-01",
    "LFPC.pdf": "1992-12-25", "LGS.pdf": "1984-07-01", "LGEEPA.pdf": "1988-03-01", "LMV.pdf": "2006-06-28",
    "LGE.pdf": "2019-10-01", "LFD.pdf": "1982-01-01", "LGBN.pdf": "2004-05-21", "LFPCA.pdf": "2006-01-01",
    "LFPA.pdf": "1995-06-01", "LISSFAM.pdf": "2003-07-10", "LOEFAM.pdf": "1986-12-27", "LOCGEUM.pdf": "1999-09-04",
    "LAN.pdf": "1992-12-02", "LIC.pdf": "1990-07-19", "LAgra.pdf": "1992-02-27", "LGP.pdf": "1974-01-07",
    "LFDA.pdf": "1997-03-24", "LCM.pdf": "2000-05-13", "LNCM.pdf": "2006-06-01", "LIE.pdf": "1993-12-28",
    "LGSNSP.pdf": "2009-01-03", "LMigra.pdf": "2011-05-26", "CNPP.pdf": "2014-03-06", "LGIPE.pdf": "2014-05-24",
    "LGPP.pdf": "2014-05-24", "LGTAIP.pdf": "2015-05-05", "LGRA.pdf": "2017-07-19", "LNEP.pdf": "2016-06-17",
    "LNSIJPA.pdf": "2016-06-17", "LNED.pdf": "2019-08-10", "LFDC.pdf": "2005-06-23", "LSAR.pdf": "1996-05-24",
    # current law postdates the historical cutoff: every historical mention refers to another law (or is an error)
    "LOPJF.pdf": "2021-06-08", "LCJPJF.pdf": "2021-06-08", "LFPPI.pdf": "2020-11-05", "CNPCF.pdf": "2023-06-08",
    "LGMASC.pdf": "2024-01-27", "LGAAS.pdf": "2024-04-18", "LGMHCTI.pdf": "2023-05-09", "LGMSV.pdf": "2022-05-18",
    "LGES_200421.pdf": "2021-04-21", "LGDOCIA_070121.pdf": "2021-01-08", "LFRM.pdf": "2021-09-15",
    "LGMIME.pdf": "2024-01-01",
}

SENT_BOUNDARY = re.compile(r"(?<=[.;:])\s+(?=[A-ZÁÉÍÓÚÑ¿¡\"“(])|\n+")
RESIDUAL = re.compile(r"(?i)\b(?:art[íi]culos?|arts?\.|preceptos?|numerales?|dispositivos?)\s*\d+[\w°º.]*(?:\s*(?:bis|ter|qu[aá]ter))?"
                      r"(?:\s*(?:,|y|e|al|a)\s*\d+[\w°º.]*)*")


def date_key(row: dict) -> str:
    if row.get("date"):
        return row["date"]
    year = str(row.get("anio") or "")
    if year.isdigit():
        return f"{int(year):04d}-{int(row.get('mes') or 1):02d}-01"
    return "0000-00-00"


def stem(cid: str) -> str:
    return cid.split("#")[0].rsplit("/", 1)[-1]


def keep_mention(cid: str, dkey: str) -> bool:
    cut = VERSION_CUT.get(stem(cid))
    return cut is None or dkey >= cut


def instrument_patterns() -> dict[str, re.Pattern]:
    """Instrument id -> regex of its title and safe aliases (accent/case-insensitive via folding is not needed:
    the raw title words are matched case-insensitively with optional accents)."""
    pats = {}
    load = [json.loads(l) for l in open(CANDIDATES / "instruments.jsonl", encoding="utf-8") if l.strip()]
    for r in load:
        names = [r["title"], *(r.get("safe_aliases") or [])] if isinstance(r.get("safe_aliases"), list) else [r["title"]]
        alts = []
        for n in names:
            n = re.sub(r"\s+", " ", str(n)).strip()
            if len(n) < 6:
                continue
            body = re.escape(n).replace(r"\ ", r"\s+")
            body = re.sub(r"[aáAÁ]", "[aáAÁ]", body)
            body = re.sub(r"[eéEÉ]", "[eéEÉ]", body)
            body = re.sub(r"[iíIÍ]", "[iíIÍ]", body)
            body = re.sub(r"[oóOÓ]", "[oóOÓ]", body)
            body = re.sub(r"[uúUÚ]", "[uúUÚ]", body)
            alts.append(body)
        if alts:
            pats[r["instrument_id"]] = re.compile("(?i)" + "|".join(alts))
    return pats


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in SENT_BOUNDARY.finditer(text):
        if m.start() > start:
            spans.append((start, m.start()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def context_window(text: str, sents: list[tuple[int, int]], pos: int) -> tuple[int, int]:
    i = next((j for j, (a, b) in enumerate(sents) if a <= pos < b), len(sents) - 1)
    a, b = sents[max(i - 1, 0)][0], sents[min(i + 1, len(sents) - 1)][1]
    if b - a > MAX_CHARS:
        half = MAX_CHARS // 2
        a2 = max(a, pos - half)
        b2 = min(b, a2 + MAX_CHARS)
        a2 = max(a, b2 - MAX_CHARS)
        while a2 > a and not text[a2 - 1].isspace():
            a2 += 1
        while b2 < b and not text[b2].isspace():
            b2 -= 1
        a, b = a2, max(b2, a2 + 1)
    return a, b


def mask(text: str, a: int, b: int, spans: list[tuple[int, int]], title_pat: re.Pattern | None) -> str:
    cuts = sorted((max(s, a), min(e, b)) for s, e in spans if s < b and e > a)
    merged: list[list[int]] = []
    for s, e in cuts:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out, cur = [], a
    for s, e in merged:
        out.append(text[cur:s])
        out.append("[CITA]")
        cur = e
    out.append(text[cur:b])
    s = "".join(out)
    s = RESIDUAL.sub("[CITA]", s)
    if title_pat is not None:
        s = title_pat.sub("[LEY]", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> dict:
    t0 = time.perf_counter()
    data = common.release()
    cindex = data.candidate_index
    hist = {}
    with HIST.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            hist[r["record_id"]] = r
    pats = instrument_patterns()
    by_article: dict[str, list[dict]] = defaultdict(list)
    memory = []
    stats = Counter()
    with MENTIONS.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            if not m["positives"]:
                continue
            row = hist[m["record_id"]]
            dkey = date_key(row)
            kept = [p for p in m["positives"] if keep_mention(p["cid"], dkey)]
            stats["mentions_resolved"] += len(m["positives"])
            stats["mentions_dropped_version"] += len(m["positives"]) - len(kept)
            kept = [p for p in kept if p["cid"] in cindex]
            if not kept:
                stats["tesis_all_mentions_dropped"] += 1
                continue
            stats["tesis_kept"] += 1
            stats["mentions_kept"] += len(kept)
            text = compose(row)
            memory.append({"record_id": m["record_id"], "date": dkey, "epoca": row["epoca"],
                           "cited": sorted({int(cindex[p["cid"]]) for p in kept})})
            sents = sentence_spans(text)
            all_spans = [(int(p["witness"][0]), int(p["witness"][1])) for p in m["positives"]] + \
                        [(int(p["span"][0]), int(p["span"][1])) for p in m["positives"]]
            seen_here = set()
            for p in kept:
                if p["cid"] in seen_here:        # one context per (article, tesis): its first mention
                    continue
                seen_here.add(p["cid"])
                a, b = context_window(text, sents, int(p["span"][0]))
                ctx = mask(text, a, b, all_spans, pats.get(p["cid"].split("#")[0]))
                if len(ctx) < 40:
                    stats["contexts_too_short"] += 1
                    continue
                by_article[p["cid"]].append({"record_id": m["record_id"], "date": dkey, "text": ctx})
    # select <= CAP diverse contexts per article
    selected = []
    n_avail = Counter()
    for cid in sorted(by_article):
        rows = sorted(by_article[cid], key=lambda r: (r["date"], int(r["record_id"]) if r["record_id"].isdigit() else 0))
        uniq, seen = [], set()
        for r in rows:
            if r["text"] not in seen:
                seen.add(r["text"])
                uniq.append(r)
        n_avail[cid] = len(uniq)
        if len(uniq) > CAP:
            pick = np.unique(np.round(np.linspace(0, len(uniq) - 1, CAP)).astype(int))
            uniq = [uniq[i] for i in pick]
        for j, r in enumerate(uniq):
            selected.append({"cid": cid, "cidx": int(cindex[cid]), "j": j, "record_id": r["record_id"], "date": r["date"],
                             "text": r["text"]})
    write_jsonl(CONTEXTS, selected)
    write_jsonl(MEMORY, memory)
    # control assignment
    rng = np.random.default_rng(SEED)
    per_art = defaultdict(list)
    for i, r in enumerate(selected):
        per_art[r["cid"]].append(i)
    by_inst = defaultdict(list)
    for cid in sorted(per_art):
        by_inst[cid.split("#")[0]].append(cid)
    control = []
    for cid in sorted(per_art):
        others = [o for o in by_inst[cid.split("#")[0]] if o != cid]
        need = len(per_art[cid])
        if not others:
            stats["control_articles_without_partner"] += 1
            continue
        pool = []
        for o in rng.permutation(len(others)):
            pool.extend(per_art[others[int(o)]])
            if len(pool) >= need:
                break
        pick = [int(x) for x in rng.permutation(pool)[:need]]
        control.append({"cid": cid, "cidx": int(cindex[cid]), "contexts": pick})
    write_jsonl(CONTROL, control)
    n_ctx_art = Counter(len(v) for v in per_art.values())
    meta = {"written_utc": utc(), "cap": CAP, "max_chars": MAX_CHARS, "stats": dict(stats),
            "articles_with_context": len(per_art), "contexts_selected": len(selected),
            "articles_by_context_count": dict(sorted(n_ctx_art.items())),
            "articles_with_more_than_cap_available": sum(v > CAP for v in n_avail.values()),
            "memory_entries": len(memory), "memory_distinct_articles": len({c for m in memory for c in m["cited"]}),
            "control_articles": len(control), "version_cut": VERSION_CUT,
            "files": {p.name: sha256_file(p) for p in (CONTEXTS, MEMORY, CONTROL)}, "seconds": time.perf_counter() - t0}
    write_json(CTX_META, meta)
    log(f"contexts: {len(selected)} for {len(per_art)} articles; memory {len(memory)} tesis; stats {dict(stats)}")
    return meta


if __name__ == "__main__":
    main()
