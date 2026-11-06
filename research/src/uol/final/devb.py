"""DEV-B labels on the run computer: the gap-review resolver patches (``scripts/gaps/g4_patched.py``, research/
reports/data_soundness_20261020) re-run over development sources only.

The bundle carries a copy of g4_patched.py whose four hard-coded thesis-PC paths are replaced by environment
variables (the replacement is listed in ``scripts/gaps/PORTING.json``; nothing else differs), the development-only
structured sources (released development cases and excluded sources published before 27 Jun 2025), the v10
catalogue and the dev-only SPLIT.json. The script itself skips lockbox and test-period records and asserts it.
Patches: P1 (R3 back-reference repaired), P2, P3 (list chains), P5 (law named before the list), P6, P7.
Rules A+B (``bindfix.py``) are a prototype that prints yields and writes no labels; they are not applied (open).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from uol.final import context

PATCHES = "P1,P2,P3,P5,P6,P7"


def run_g4() -> Path:
    from uol.final import variants
    prof = context.active()
    data = prof.data_dir
    out = variants.variant_root("devb")
    out.mkdir(parents=True, exist_ok=True)
    pkl = out / "g4.pkl"
    env = {**os.environ,
           "UOL_G4_SRC": str(Path(__file__).resolve().parents[2]),
           "UOL_G4_SOURCES": str(data / "derived" / "dev_sources.jsonl"),
           "UOL_G4_CANDIDATES": str(data / "release" / "candidates") + os.sep,
           "UOL_G4_SPLIT": str(data / "state" / "SPLIT.json")}
    r = subprocess.run([sys.executable, str(data / "scripts" / "gaps" / "g4_patched.py"), str(pkl), PATCHES], env=env,
                       capture_output=True, text=True)
    (out / "g4.log").write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError(f"g4_patched failed: {r.stderr[-500:]}")
    return pkl


REPLACEMENTS = (
    ('sys.path.insert(0, "C:/Work/ThesisUOL/src")', 'import os; sys.path.insert(0, os.environ["UOL_G4_SRC"])'),
    ('ROOT + "data/derived/facts_to_law_v2/sjf_structured_sources_v1/sources.jsonl"', 'os.environ["UOL_G4_SOURCES"]'),
    ('ROOT + "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates/"', 'os.environ["UOL_G4_CANDIDATES"]'),
    ('ROOT + "research/state/embedding_push_20261004/SPLIT.json"', 'os.environ["UOL_G4_SPLIT"]'),
)


def port_script(src: Path, dst: Path) -> dict:
    """Bundle time: the g4 copy with the thesis-PC paths replaced; refuses if a replacement does not apply."""
    text = src.read_text(encoding="utf-8")
    done = []
    for old, new in REPLACEMENTS:
        if old not in text:
            raise ValueError(f"g4_patched.py changed: {old!r} not found")
        text = text.replace(old, new)
        done.append({"old": old, "new": new})
    if "C:/Work/ThesisUOL" in text.replace('ROOT = "C:/Work/ThesisUOL/"', ""):
        raise ValueError("g4_patched.py still names a thesis-PC path")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return {"source": str(src), "replacements": done}


BINDFIX_REPLACEMENTS = (
    ('sys.path.insert(0, "C:/Work/ThesisUOL/src")', 'import os; sys.path.insert(0, os.environ["UOL_G4_SRC"])'),
    ('ROOT + "data/releases/uol_c01_statutes_20261002_v10_reconciled/candidates/"', 'os.environ["UOL_G4_CANDIDATES"]'),
    ('ROOT + "research/state/embedding_push_20261004/SPLIT.json"', 'os.environ["UOL_G4_SPLIT"]'),
)
BINDFIX_CUT = "\nQ = {}\n"


def port_bindfix(src: Path, dst: Path) -> dict:
    """Bundle time: the definitions part of bindfix.py (rules A and B with their guards: ``assess``, ``ANA``, the
    catalogue titles, kinds and labels), up to its reporting loop, with the thesis-PC paths replaced."""
    text = src.read_text(encoding="utf-8")
    if BINDFIX_CUT not in text:
        raise ValueError("bindfix.py changed: the definitions/report boundary 'Q = {}' is missing")
    text = text.split(BINDFIX_CUT, 1)[0] + "\n"
    done = []
    for old, new in BINDFIX_REPLACEMENTS:
        if old not in text:
            raise ValueError(f"bindfix.py changed: {old!r} not found")
        text = text.replace(old, new)
        done.append({"old": old, "new": new})
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return {"source": str(src), "kept": "everything before 'Q = {}' (definitions)", "replacements": done}


def bindfix_namespace() -> dict:
    prof = context.active()
    data = prof.data_dir
    os.environ.update({"UOL_G4_SRC": str(Path(__file__).resolve().parents[2]),
                       "UOL_G4_CANDIDATES": str(data / "release" / "candidates") + os.sep,
                       "UOL_G4_SPLIT": str(data / "state" / "SPLIT.json")})
    ns: dict = {"__name__": "bindfix_defs"}
    code = (data / "scripts" / "bindfix_defs.py").read_text(encoding="utf-8")
    exec(compile(code, str(data / "scripts" / "bindfix_defs.py"), "exec"), ns)
    return ns


def apply_rules_ab(res: dict, sources: dict, ns: dict | None = None, covered: dict | None = None) -> dict:
    """Rules A and B (bindfix) on the mentions the patched resolver still leaves missing: a bound law plus the article
    numbers of the mention become labels when the (law, article) exists in the catalogue. Development rows only
    (``res`` holds released development and excluded development-period sources). Returns per-rule label counts."""
    import re
    ns = ns or bindfix_namespace()
    covered = covered or {}
    fold, assess, sec_text, labels, BIND = ns["fold"], ns["assess"], ns["sec_text"], ns["labels"], ns["BIND"]
    counts = {"A_title_after": 0, "B_anaphor": 0, "cases_changed": 0, "cases_now_full": 0}
    for qid, r in res.items():
        s = sources.get(qid)
        if s is None or r.get("kind") not in ("released_dev", "excluded_dev"):
            continue
        reasoning = fold(" ".join(x["text"] for x in s["sections"] if x["field"] != "hechos"))
        have = {p["candidate_id"] for p in r["positives"]}
        added, still_missing = [], 0
        for u in r["unresolved"]:
            if u.get("disposition") not in (None, "missing_positive"):
                continue
            if any(sec == u.get("section") and a0 <= u["span"][0] <= b0 + 150 for sec, a0, b0 in covered.get(qid, [])):
                continue                                  # inside a range P8 expanded (bound and labelled)
            if u.get("reason") not in BIND:
                still_missing += 1
                continue
            t = sec_text(s, u["section"])
            rule, law = assess(t, reasoning, u["span"])
            if rule not in ("A_title_after", "B_anaphor"):
                still_missing += 1
                continue
            a, b = u["span"]
            for nm in re.findall(r"(?<![\w.])(\d{1,4})(?![\d])", t[a:b]):
                cid = f"https://www.diputados.gob.mx/LeyesBiblio/pdf/{law}.pdf#articulo={nm}"
                if (law, nm) in labels and cid not in have:
                    have.add(cid)
                    added.append({"candidate_id": cid, "rule_ids": [f"bindfix_{rule[0]}"], "section": u["section"],
                                  "span": u["span"], "surface_sha256": u.get("surface_sha256")})
                    counts[rule] += 1
        if added:
            r["positives"] = list(r["positives"]) + added
            counts["cases_changed"] += 1
            if still_missing == 0 and r.get("status") != "full":
                r["status"] = "full"
                counts["cases_now_full"] += 1
    return counts
