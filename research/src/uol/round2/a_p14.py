"""Round 2 / A: P14 version and reform stratum (EXPLORATORY diagnostic; no method).

Inputs (read-only):
* C01 catalogue text (about September 2024; latest inline DOF note 2024-09-15): the release-bound ``articles.jsonl``.
* Consolidated federal laws observed 2026-09-09 in the source home
  (``experiments/runs/corpus_expansion_20260909T145239422579Z/normalized/official_current_laws.jsonl``), one PDF text per
  instrument URL; split into articles by heading regex (first occurrence of each label = main body, before transitorios).

Per cited target: DOF note dates (``... reformado/adicionado/derogado ... DOF dd-mm-yyyy``) in each snapshot, and a token
similarity between the two article texts with note lines and page headers removed.
Per (query, target), with d = tesis publication date:
* ``catalogue_newer``: a C01 note date > d (the catalogue holds a reform the court could not have read);
* ``catalogue_older``: a 2026 note date in (2024-09-15, d] (the court read a reform the catalogue lacks);
* ``version_mismatch`` = either; ``reformed_after_tesis``: any note date in either snapshot > d.
Queries are flagged when any specific target is flagged. Scores: frozen ``LTR_emb_hgb`` / ``LTR_noemb_hgb``.
"""
from __future__ import annotations

import difflib
import json
import re
import sys
import time
from collections import Counter

import numpy as np

from uol.embedding_push import common
from uol.round2 import a_common as A

LAWS_2026 = A.SOURCE_HOME / "experiments/runs/corpus_expansion_20260909T145239422579Z/normalized/official_current_laws.jsonl"
SNAP_2024 = "2024-09-15"
ABROGATED = ("LOPJF.pdf", "LAASSP.pdf", "LGTAIP.pdf", "LGPDPPSO.pdf", "LFPDPPP.pdf")   # URLs now serve successor laws (A04)
NOTE = re.compile(r"(?i)\b(?:reformad|adicionad|derogad|recorrid|aclarad)[oa]s?\b[^\n]{0,80}?\bDOF\s+((?:\d{2}-\d{2}-\d{4}(?:\s*,\s*|\s+y\s+)?)+)")
DATE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
NOTE_LINE = re.compile(r"(?im)^[^\n]*\b(?:reformad|adicionad|derogad|recorrid|aclarad)[oa]s?\b[^\n]*\bDOF\b[^\n]*$|^[^\n]*Fe de erratas[^\n]*$")
HEADER_LINE = re.compile(r"(?im)^[^\n]*(C[ÁA]MARA DE DIPUTADOS|Secretar[íi]a General|Secretar[íi]a de Servicios Parlamentarios|"
                         r"[ÚU]ltima[s]? Reforma[s]? (publicadas )?DOF)[^\n]*$|^\s*\d+\s+de\s+\d+\s*$")
SUFFIX = r"(?:bis|ter|qu[aá]ter|quinquies|quintus|sexies|septies|octies|nonies|decies|[a-z])"
HEADING = re.compile(r"(?m)^[ \t]*(?:ART[ÍI]CULO|Art[íi]culo)\s+(\d+)\s*(?:o\.?|°|º)?\s*((?:[-–]?\s*" + SUFFIX + r"\b\.?\s*)*)(?:[-–]?\s*(\d+)\b)?\s*(?:\.|-|–|:)",
                     re.IGNORECASE)


def dates(text: str) -> list[str]:
    out = []
    for m in NOTE.finditer(text or ""):
        for d, mth, y in DATE.findall(m.group(1)):
            out.append(f"{y}-{mth}-{d}")
    return sorted(set(out))


def norm_label(num: str, suffix: str, tail: str | None) -> str:
    parts = [str(int(num))]
    for s in re.findall(SUFFIX, A.fold_text(suffix or ""), re.IGNORECASE):
        parts.append(s.lower().replace("á", "a"))
    if tail:
        parts.append(str(int(tail)))
    return "-".join(parts)


def c01_label(candidate_id: str) -> str:
    lab = candidate_id.split("#articulo=")[-1].lower()
    return lab


NOTE_PHRASE = re.compile(r"(?:(?:articulo|parrafo|fraccion|inciso|apartado|base|numeral|denominacion|capitulo|titulo|seccion|"
                         r"libro|parte|reforma)\s+)?(?:reformad|adicionad|derogad|recorrid|aclarad)[oa]s?(?:\s+[a-z]+){0,6}?\s+dof"
                         r"(?:\s+\d{1,2}\s+\d{1,2}\s+\d{4}(?:\s+y)?)+|fe de erratas(?:\s+[a-z]+){0,6}?\s+dof(?:\s+\d{1,2}\s+\d{1,2}\s+\d{4})+")


def strip_text(text: str, titles: set[str]) -> list[str]:
    """Folded tokens with page headers, repeated title lines and inline DOF note phrases removed."""
    t = HEADER_LINE.sub(" ", text or "")
    lines = [ln for ln in t.splitlines() if ln.strip() and ln.strip() not in titles]
    folded = " ".join(re.findall(r"[a-z0-9]+", A.fold_text(" ".join(lines))))
    return NOTE_PHRASE.sub(" ", folded).split()


def split_2026(text: str) -> tuple[dict[str, str], set[str]]:
    """label -> article text (first occurrence), plus repeated uppercase title lines (page headers)."""
    lines = Counter(ln.strip() for ln in text.splitlines() if ln.strip())
    titles = {ln for ln, c in lines.items() if c >= 3 and len(ln) > 15 and ln.upper() == ln}
    heads = list(HEADING.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(heads):
        lab = norm_label(m.group(1), m.group(2), m.group(3))
        if lab in out:
            continue
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        out[lab] = text[m.start():end]
    return out, titles


def target_table(targets: set[str]) -> dict[str, dict]:
    path = A.CACHE / "p14_targets.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    d = common.release()
    arts = {}
    with open(d.candidate_binding["articles_path"], encoding="utf-8") as h:
        for line in h:
            r = json.loads(line)
            if r["candidate_id"] in targets:
                arts[r["candidate_id"]] = r
    by_inst: dict[str, list[str]] = {}
    for cid, r in arts.items():
        by_inst.setdefault(r["instrument_id"], []).append(cid)
    out: dict[str, dict] = {}
    with open(LAWS_2026, encoding="utf-8") as h:
        for line in h:
            law = json.loads(line)
            cids = by_inst.get(law["pdf_url"])
            if not cids:
                continue
            segs, titles = split_2026(law["text"])
            for cid in cids:
                r = arts[cid]
                lab = c01_label(cid)
                seg = segs.get(lab)
                rec = {"instrument": law["pdf_url"].rsplit("/", 1)[-1], "label": lab, "c01_dates": dates(r["article_body"]),
                       "matched_2026": seg is not None, "replaced_instrument": law["pdf_url"].endswith(ABROGATED)}
                if seg is not None:
                    rec["dates_2026"] = dates(seg)
                    a, b = strip_text(r["article_body"], set()), strip_text(seg, titles)
                    # drop the heading tokens ("articulo 5o") before comparing
                    rec["similarity"] = difflib.SequenceMatcher(None, a[2:], b[2:], autojunk=False).ratio() if a and b else 0.0
                    rec["len_tokens"] = [len(a), len(b)]
                out[cid] = rec
    for cid in targets - set(out):
        out[cid] = {"instrument": arts.get(cid, {}).get("instrument_id", "?"), "matched_2026": False, "c01_dates": dates(arts.get(cid, {}).get("article_body", "")),
                    "replaced_instrument": False}
    A.CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def change_class(rec: dict) -> str:
    if rec.get("replaced_instrument"):
        return "instrument replaced (abrogated)"
    if not rec.get("matched_2026"):
        return "not matched in 2026 text"
    new = [x for x in rec.get("dates_2026", []) if x > SNAP_2024]
    sim = rec.get("similarity", 0.0)
    if sim >= 0.995 and not new:
        return "unchanged"
    if not new:
        return "text differs, no new DOF note (extraction/segmentation)"
    if sim >= 0.95:
        return "cosmetic (new DOF note, similarity >= .95)"
    return "substantive (new DOF note, similarity < .95)"


def pair_flags(rec: dict, tesis_date: str | None) -> dict[str, bool]:
    if not tesis_date:
        return {"catalogue_newer": False, "catalogue_older": False, "version_mismatch": False, "reformed_after_tesis": False, "dated": False}
    c01 = rec.get("c01_dates", [])
    d26 = rec.get("dates_2026", [])
    newer = any(x > tesis_date for x in c01)
    older = any(SNAP_2024 < x <= tesis_date for x in d26)
    after = any(x > tesis_date for x in c01 + d26)
    return {"catalogue_newer": newer, "catalogue_older": older, "version_mismatch": newer or older, "reformed_after_tesis": after, "dated": True}


def main(argv: list[str] | None = None) -> int:
    t0 = time.perf_counter()
    d = common.release()
    src = A.load_sources()
    targets = {d.candidate_ids[p] for i in range(len(d.queries)) for p in d.specific_idx[i]}
    table = target_table(targets)
    cls = Counter(change_class(r) for r in table.values())
    sims = [r["similarity"] for r in table.values() if "similarity" in r]
    A.log(f"P14 targets {len(table)}: {dict(cls)}")
    out = {"targets": {"n": len(table), "change_class": dict(cls), "matched_2026": sum(r.get("matched_2026", False) for r in table.values()),
                       "with_new_2026_dates": sum(any(x > SNAP_2024 for x in r.get("dates_2026", [])) for r in table.values()),
                       "similarity_quantiles": np.quantile(sims, [0.05, 0.25, 0.5]).tolist() if sims else None},
           "splits": {}}
    for split in ("dev", "lockbox"):
        a, b = A.evaluate_cached("LTR_emb_hgb", split), A.evaluate_cached("LTR_noemb_hgb", split)
        flags_q = {k: [] for k in ("catalogue_newer", "catalogue_older", "version_mismatch", "reformed_after_tesis", "substantive_between_snapshots")}
        pair_rows = []
        for j, row in enumerate(a["rows"]):
            q = d.queries[int(row)]
            td = (src.get(q.query_id) or {}).get("publication_date")
            fl = {k: False for k in flags_q}
            ra = dict(zip(a["per_query"][j]["positive_idx"], a["per_query"][j]["positive_ranks"]))
            rb = dict(zip(b["per_query"][j]["positive_idx"], b["per_query"][j]["positive_ranks"]))
            for p in a["per_query"][j]["positive_idx"]:
                rec = table[d.candidate_ids[p]]
                pf = pair_flags(rec, td)
                pf["substantive_between_snapshots"] = change_class(rec).startswith("substantive")
                for k in fl:
                    fl[k] |= pf[k]
                pair_rows.append((pf["version_mismatch"], pf["substantive_between_snapshots"], ra[p] <= 20, rb[p] <= 20))
            for k in flags_q:
                flags_q[k].append(fl[k])
        res = {}
        for k, vals in flags_q.items():
            keep = np.asarray(vals)
            res[k] = {"flagged": A.subset_contrast(a, b, keep), "unflagged": A.subset_contrast(a, b, ~keep)}
        pr = np.asarray(pair_rows)
        pairs = {}
        for name, col in (("version_mismatch", 0), ("substantive_between_snapshots", 1)):
            for lv in (True, False):
                m = pr[:, col] == lv
                pairs[f"{name}={lv}"] = {"pairs": int(m.sum()), "R20_emb": float(pr[m, 2].mean()) if m.any() else None,
                                         "R20_noemb": float(pr[m, 3].mean()) if m.any() else None}
        out["splits"][split] = {"queries": res, "pairs": pairs, "n_queries": len(a["rows"])}
        A.log(f"P14 {split}: version_mismatch flagged {int(np.sum(flags_q['version_mismatch']))}/{len(a['rows'])}")
    out["wall_seconds"] = time.perf_counter() - t0
    A.save_json("p14_reform.json", out)
    (A.OUT / "p14.md").write_text(md(out), encoding="utf-8")
    return 0


def md(out: dict) -> str:
    lines = ["| stratum | split | flagged q | EMB | NO-EMB | Δ [95% CI] | unflagged q | EMB | NO-EMB | Δ [95% CI] |", "|---|---|---:|---:|---:|---|---:|---:|---:|---|"]
    for k in ("version_mismatch", "catalogue_newer", "catalogue_older", "reformed_after_tesis", "substantive_between_snapshots"):
        for split in ("dev", "lockbox"):
            f, u = out["splits"][split]["queries"][k]["flagged"], out["splits"][split]["queries"][k]["unflagged"]
            lines.append(f"| {k} | {split} | {f['n_queries']} | {A.fmt(f['mean_a'])} | {A.fmt(f['mean_b'])} | {A.fmt(f['effect'], 3, True)} {A.ci(f['ci95'])} | "
                         f"{u['n_queries']} | {A.fmt(u['mean_a'])} | {A.fmt(u['mean_b'])} | {A.fmt(u['effect'], 3, True)} {A.ci(u['ci95'])} |")
    lines.append("")
    lines.append("| pairs | split | n | R@20 EMB | R@20 NO-EMB |")
    lines.append("|---|---|---:|---:|---:|")
    for split in ("dev", "lockbox"):
        for k, v in out["splits"][split]["pairs"].items():
            lines.append(f"| {k} | {split} | {v['pairs']} | {A.fmt(v['R20_emb'])} | {A.fmt(v['R20_noemb'])} |")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
