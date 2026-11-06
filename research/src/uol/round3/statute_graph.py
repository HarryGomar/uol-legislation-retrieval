"""Phase 2 step 3 / S2: the label-free statute graph G2 and its gold-enrichment mechanism check (PLAN §3 S2, §4.3).

Graph (no labels). For each catalogue instrument, its source law text (``processing_full_20260913`` federal texts) is
found by matching article bodies; every catalogue article body is located in that text, which gives its heading path
(LIBRO / PARTE / TÍTULO / CAPÍTULO / SECCIÓN lines above it) and its document order. Edge types:
* ``sibling``: same lowest heading (same full path) in the same instrument;
* ``adj1`` / ``adj2``: next / second-next article in document order within the instrument;
* ``chapter``: same path up to the CAPÍTULO level (articles under one chapter, sections merged);
* ``xref_int`` / ``xref_ext``: explicit cross-references in the article text ("artículo(s) N, M y K" bound to "esta
  Ley", no law, or a named law resolved through ``catalog_map``), directed from the citing article.
Same-instrument membership is kept as a group id, not as edges.

Mechanism check (dev labels only). For dev P_spec queries, seeds are gold articles in the LTR_emb_hgb top 10. Per edge
type: the gold rate among the seeds' neighbours outside the top 10, against degree-matched non-neighbours from the same
instrument (nearest total degree, also outside the top 10). Lift = ratio of the two rates, with a component bootstrap
interval. PLAN: an edge type with lift <= 1.2 is dropped from S2.

    python -m uol.round3.statute_graph build      # graph + datasheet (label-free)
    python -m uol.round3.statute_graph lift       # dev enrichment lifts
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from uol.round3 import catalog_map, common as C

FEDERAL = C.ROOT / "data/derived/processing_full_20260913/federal_all_eligible_text_r2/source_blocks.jsonl"
ARTICLES = C.ROOT / "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates/articles.jsonl"
OUT = C.RUN / "phase2" / "statute_graph"
LEVELS = {"libro": 0, "parte": 0, "titulo": 1, "capitulo": 2, "seccion": 3, "subseccion": 4, "apartado": 4}
HEAD = re.compile(r"^\s*(LIBRO|PARTE|T[IÍ]TULO|CAP[IÍ]TULO|SECCI[OÓ]N|SUBSECCI[OÓ]N|APARTADO|TRANSITORIOS?)\b(.*)$")
XREF = re.compile(r"\bart[íi]culos?\s+((?:\d+[\w\-]*(?:\s*(?:o|º|°)\.?)?(?:\s+(?:bis|ter|qu[aá]ter|quinquies)\b)?"
                  r"(?:\s*(?:,|y|e|a|al)\s*)?)+)", re.I)
NUM = re.compile(r"(\d+)(?:\s*(?:o|º|°)\.?)?(?:\s*[-\s]?\s*(bis|ter|qu[aá]ter|quinquies)\b)?", re.I)
LAW_AFTER = re.compile(r"^[^.;]{0,80}?\b(?:de|del)\s+(?:la|el|esta|este|la\s+presente|el\s+presente|dicha|dicho)?\s*"
                       r"((?:ley|c[oó]digo|constituci[oó]n|reglamento|ordenamiento)[^,.;()]{0,120})", re.I)
SELF = re.compile(r"^(?:ley|c[oó]digo|ordenamiento|reglamento)\s*$", re.I)


def fold(s: str) -> str:
    return catalog_map.fold(s)


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_documents() -> dict[str, str]:
    docs = {}
    with FEDERAL.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs[r["document_id"]] = r["text"]
    return docs


def load_articles() -> dict[str, list[dict]]:
    """{instrument code: [article rows in catalogue order]} for candidates in the release."""
    d = C.release()
    idx = d.candidate_index
    out: dict[str, list[dict]] = defaultdict(list)
    with ARTICLES.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["candidate_id"] in idx:
                code = r["candidate_id"].rsplit("/", 1)[-1].split(".pdf")[0]
                out[code].append({"cid": r["candidate_id"], "idx": int(idx[r["candidate_id"]]), "body": r["article_body"],
                                  "label": r["article_label"]})
    return out


def match_documents(arts: dict[str, list[dict]], docs: dict[str, str]) -> dict[str, str]:
    """Instrument code -> document id: the document containing the most of 5 sampled article-body heads."""
    normdocs = {k: _norm_ws(v) for k, v in docs.items()}
    out = {}
    for code, rows in arts.items():
        heads = [_norm_ws(r["body"])[:70] for r in rows[:: max(1, len(rows) // 5)][:5]]
        heads = [h for h in heads if len(h) >= 30]
        best, best_n = None, 0
        for k, t in normdocs.items():
            n = sum(h in t for h in heads)
            if n > best_n:
                best, best_n = k, n
        if best is not None and best_n >= max(1, len(heads) // 2):
            out[code] = best
    return out


def heading_spans(text: str) -> list[tuple[int, tuple[str, ...]]]:
    """[(char offset, heading path)] at every heading line, in order."""
    out, stack, pos = [], {}, 0
    for line in text.splitlines(keepends=True):
        m = HEAD.match(line)
        if m:
            kind = fold(m.group(1)).rstrip("s")
            if kind.startswith("transitorio"):
                stack = {-1: "TRANSITORIOS"}
            else:
                lvl = LEVELS.get(kind, 4)
                stack = {k: v for k, v in stack.items() if 0 <= k < lvl}
                stack[lvl] = _norm_ws(line)[:160]
            out.append((pos, tuple(v for _, v in sorted(stack.items()))))
        pos += len(line)
    return out


def build() -> dict:
    d = C.release()
    docs = load_documents()
    arts = load_articles()
    doc_of = match_documents(arts, docs)
    n = len(d.candidate_ids)
    path_of: list[tuple[str, ...] | None] = [None] * n
    order_of = np.full(n, -1, dtype=np.int64)
    inst_of = np.full(n, -1, dtype=np.int32)
    codes = sorted(arts)
    for ci, code in enumerate(codes):
        for r in arts[code]:
            inst_of[r["idx"]] = ci
        if code not in doc_of:
            continue
        text = docs[doc_of[code]]
        spans = heading_spans(text)
        offs = [o for o, _ in spans]
        cursor = 0
        located = []
        for r in arts[code]:
            head = r["body"][:60]
            at = text.find(head, cursor)
            if at < 0:
                at = text.find(head)
            if at < 0:
                continue
            cursor = at
            k = int(np.searchsorted(offs, at, side="right")) - 1
            path_of[r["idx"]] = spans[k][1] if k >= 0 else ()
            located.append((at, r["idx"]))
        for rank, (_, i) in enumerate(sorted(located)):
            order_of[i] = rank
    edges: dict[str, set[tuple[int, int]]] = defaultdict(set)
    by_inst: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        if order_of[i] >= 0:
            by_inst[int(inst_of[i])].append(i)
    group_sib: dict[tuple, list[int]] = defaultdict(list)
    group_chap: dict[tuple, list[int]] = defaultdict(list)
    for inst, members in by_inst.items():
        members.sort(key=lambda i: order_of[i])
        for a, b in zip(members, members[1:]):
            edges["adj1"].add((a, b))
        for a, b in zip(members, members[2:]):
            edges["adj2"].add((a, b))
        for i in members:
            p = path_of[i] or ()
            if p:
                group_sib[(inst, p)].append(i)
                chap = tuple(h for h in p if fold(h).split()[0] in ("libro", "parte", "titulo", "capitulo"))
                if chap:
                    group_chap[(inst, chap)].append(i)
    for grp, name in ((group_sib, "sibling"), (group_chap, "chapter")):
        for members in grp.values():
            if 1 < len(members) <= 400:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        edges[name].add((members[x], members[y]))
    arts_by_code = {code: {fold(r["label"]).replace(" ", "-"): r["idx"] for r in rows} for code, rows in arts.items()}
    xstats = defaultdict(int)
    for code, rows in arts.items():
        for r in rows:
            body = r["body"]
            for m in XREF.finditer(body):
                tail = body[m.end():m.end() + 160]
                lm = LAW_AFTER.match(tail)
                target = code
                if lm:
                    law = lm.group(1).strip()
                    if not (SELF.match(law) or re.match(r"(?i)^(ley|c[oó]digo|ordenamiento|reglamento)\b\s*(?:,|$)", law)):
                        target = catalog_map.instrument(law)
                if target is None:
                    xstats["unresolved_law"] += 1
                    continue
                table = arts_by_code.get(target, {})
                for nm in NUM.finditer(m.group(1)):
                    key = nm.group(1) + (f"-{fold(nm.group(2))}" if nm.group(2) else "")
                    j = table.get(key)
                    if j == r["idx"]:
                        xstats["self_mention"] += 1
                        continue
                    if j is None:
                        xstats["unresolved_article"] += 1
                        continue
                    edges["xref_int" if target == code else "xref_ext"].add((r["idx"], j))
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "graph.npz", inst=inst_of, order=order_of,
                        **{f"e_{k}": np.asarray(sorted(v), dtype=np.int32).reshape(-1, 2) for k, v in edges.items()})
    C.write_json(OUT / "paths.json", {str(i): list(p) for i, p in enumerate(path_of) if p})
    sheet = {"written_utc": C.utc(), "catalogue_articles": n, "instruments": len(codes),
             "instruments_matched_to_text": len(doc_of), "articles_located": int((order_of >= 0).sum()),
             "articles_with_heading_path": int(sum(1 for p in path_of if p)),
             "sibling_groups": len(group_sib), "chapter_groups": len(group_chap),
             "groups_over_400_skipped": {"sibling": sum(len(v) > 400 for v in group_sib.values()),
                                         "chapter": sum(len(v) > 400 for v in group_chap.values())},
             "edges": {k: len(v) for k, v in sorted(edges.items())}, "xref_unresolved": dict(xstats),
             "sources": {"texts": str(FEDERAL), "articles": str(ARTICLES)}, "labels_used": False}
    C.write_json(OUT / "DATASHEET.json", sheet)
    return sheet


def load_graph() -> dict[str, np.ndarray]:
    z = np.load(OUT / "graph.npz")
    return {k: z[k] for k in z.files}


def neighbours(g: dict[str, np.ndarray], etype: str, n: int) -> list[set[int]]:
    nb: list[set[int]] = [set() for _ in range(n)]
    for a, b in g[f"e_{etype}"]:
        nb[int(a)].add(int(b))
        nb[int(b)].add(int(a))
    return nb


def lift() -> dict:
    from uol.round3 import features
    d = C.release()
    n = len(d.candidate_ids)
    g = load_graph()
    etypes = sorted(k[2:] for k in g if k.startswith("e_"))
    M = features.frozen_matrix("LTR_emb_hgb")
    rows = C.population_rows("P_spec", C.dev_rows())
    comps = C.component_ids(rows)
    top10 = {int(r): set(np.argsort(-np.asarray(M[int(r)], dtype=np.float64), kind="stable")[:10].tolist()) for r in rows}
    nbs = {e: neighbours(g, e, n) for e in etypes}
    deg_total = np.zeros(n)
    for e in etypes:
        deg_total += np.asarray([len(s) for s in nbs[e]])
    inst = g["inst"]
    by_inst: dict[int, np.ndarray] = {int(k): np.flatnonzero(inst == k) for k in np.unique(inst)}
    rng = np.random.default_rng(C.SEED)
    out = {}
    for e in etypes:
        per_comp = defaultdict(lambda: [0, 0, 0, 0])            # nb gold, nb total, ctrl gold, ctrl total
        for r, comp in zip(rows, comps):
            r = int(r)
            gold = set(int(a) for a in d.specific_idx[r])
            seeds = gold & top10[r]
            if not seeds:
                continue
            nb = set().union(*(nbs[e][s] for s in seeds)) - top10[r]
            if not nb:
                continue
            for a in nb:
                pool = by_inst[int(inst[a])]
                pool = pool[(pool != a)]
                cand = [c for c in pool[np.argsort(np.abs(deg_total[pool] - deg_total[a]), kind="stable")[:25]]
                        if c not in nb and c not in top10[r]]
                if not cand:
                    continue
                c = int(cand[rng.integers(0, len(cand))])
                rec = per_comp[comp]
                rec[0] += a in gold
                rec[1] += 1
                rec[2] += c in gold
                rec[3] += 1
        arr = np.asarray(list(per_comp.values()), dtype=np.float64)
        if not len(arr) or arr[:, 3].sum() == 0:
            out[e] = {"assessed": False}
            continue
        point = (arr[:, 0].sum() / arr[:, 1].sum()) / max(arr[:, 2].sum() / arr[:, 3].sum(), 1e-12)
        boots = []
        for _ in range(2000):
            s = arr[rng.integers(0, len(arr), len(arr))]
            cr = s[:, 2].sum() / s[:, 3].sum()
            boots.append((s[:, 0].sum() / s[:, 1].sum()) / cr if cr > 0 else np.inf)
        boots = np.asarray(boots)
        out[e] = {"assessed": True, "components": len(arr), "neighbour_pairs": int(arr[:, 1].sum()),
                  "neighbour_gold_rate": float(arr[:, 0].sum() / arr[:, 1].sum()),
                  "control_gold_rate": float(arr[:, 2].sum() / arr[:, 3].sum()), "lift": float(point),
                  "lift_ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))],
                  "keep_for_S2": bool(point > 1.2)}
    res = {"written_utc": C.utc(), "population": "dev P_spec", "seed_rule": "gold ∩ LTR_emb_hgb top 10",
           "control": "degree-matched (total degree, nearest 25) non-neighbour in the same instrument, outside the top 10",
           "edge_types": out}
    C.write_json(OUT / "LIFTS.json", res)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build", "lift"])
    a = ap.parse_args()
    res = build() if a.command == "build" else lift()
    print(json.dumps(res, indent=1, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
