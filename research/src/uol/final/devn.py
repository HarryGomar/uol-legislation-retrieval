"""DEV-N (RUN_PLAN v2 §2): the development cases whose facts name a provision (the S2 stratum, condensed specific
lists: 854 cases), with the whole citation clause removed from the masked facts, re-encoded and scored as a separate
stratum (never pooled with DEV).

The clause rule is the data-soundness report's (``research/reports/data_soundness_20261020/scripts/residue.py``,
``CLAUSE``): an optional framing verb, articles, the mask, and the run of subdivision tokens after it, applied as
``re.sub(r"\\s+", " ", CLAUSE.sub("", text))``. The script is not importable (it runs at import), so the pattern is
read from its source with ``ast`` and evaluated (string literals, ``+`` and the name ``M``), never retyped here.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path

from uol.final import context, guard

RESIDUE = ("scripts/residue.py",)
LOCAL_RESIDUE = Path(__file__).resolve().parents[3] / "research/reports/data_soundness_20261020/scripts/residue.py"


def residue_path() -> Path:
    cands = []
    if context._ACTIVE is not None and context._ACTIVE.data_dir:
        cands.append(context._ACTIVE.data_dir / "scripts" / "residue.py")
    cands += [Path(__file__).resolve().parents[4] / "data" / "scripts" / "residue.py", LOCAL_RESIDUE]   # bundle, thesis PC
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"residue.py not found in {[str(c) for c in cands]}")


@lru_cache(maxsize=1)
def clause_regex() -> re.Pattern:
    tree = ast.parse(residue_path().read_text(encoding="utf-8"))
    env = {"M": re.escape("[CITA_OCULTA]")}

    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return ev(node.left) + ev(node.right)
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        raise ValueError(f"unexpected expression in residue.py CLAUSE: {ast.dump(node)[:120]}")
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CLAUSE" for t in node.targets):
            call = node.value
            pattern = ev(call.args[0])
            flags = re.I if any(isinstance(a, ast.Attribute) and a.attr == "I" for a in call.args[1:]) else 0
            return re.compile(pattern, flags)
    raise ValueError("CLAUSE not found in residue.py")


def drop(text: str) -> str:
    return re.sub(r"\s+", " ", clause_regex().sub("", text)).strip()


def rows() -> list[int]:
    from uol.embedding_push import common as ep
    return [int(r) for r in guard.dev_rows_only(ep.population_rows("S2_spec", ep.dev_rows()))]


def dropped_texts() -> dict[int, str]:
    from uol.embedding_push import common as ep
    d = ep.release()
    return {r: drop(d.queries[r].text) for r in rows()}


def apply_text_override() -> dict[str, int]:
    """Profile variant ``devn``: the DEV-N rows carry their clause-dropped facts in this process (every system that
    reads ``ep.release().queries[r].text`` sees them; dense systems use the int8 ``devn`` query vectors)."""
    from uol.embedding_push import common as ep
    d = ep.release()
    if getattr(d, "_uol_final_devn", False):
        return {"already": 1}
    n = 0
    for r, t in dropped_texts().items():
        d.queries[r].text = t
        n += 1
    d._uol_final_devn = True
    return {"rows": n}
