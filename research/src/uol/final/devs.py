"""DEV-S (D109, 24 Oct 2026; ANALYSIS_PLAN_C §6b): the development facts in shortened wording, a check of the final
systems without retraining on short, user-like queries.

The rule is the earlier X10 lens's (``uol.eval_lenses.x10_short.truncate``, 15 Oct): each DEV P_spec masked Hechos is
cut to its first 25, 50 or 100 whitespace words; a text already that short is kept whole. One profile variant per
length (``devs25``, ``devs50``, ``devs100``): the shortened rows carry their cut facts in the process (as ``devn``
does with its clause-dropped facts); dense systems read their re-encoded ``<variant>__<key>`` query vectors for the
rows whose text changed and the saved vectors for the rest; population DEV-S25, DEV-S50 or DEV-S100 is every DEV
P_spec row, so each length is compared with DEV on the same cases.
"""
from __future__ import annotations

from uol.final import context, guard

LENGTHS = {"devs25": 25, "devs50": 50, "devs100": 100}
POPULATIONS = {v: f"DEV-S{n}" for v, n in LENGTHS.items()}
_APPLIED: dict[int, dict[int, str]] = {}          # id(release) -> the rows the override changed (their cut text)


def truncate(text: str, n: int) -> str:
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n])


def words(variant: str | None = None) -> int:
    v = variant or context.active().variant
    if v not in LENGTHS:
        raise RuntimeError(f"DEV-S runs in a devs variant ({sorted(LENGTHS)}), not {v!r}")
    return LENGTHS[v]


def population_rows():
    from uol.embedding_push import common as ep
    return guard.dev_rows_only(ep.population_rows("P_spec", ep.dev_rows()), "DEV-S rows")


def short_texts(variant: str | None = None) -> dict[int, str]:
    """The rows whose facts the cut changes, with their shortened text (read before the override, or as it left them)."""
    from uol.embedding_push import common as ep
    d = ep.release()
    if id(d) in _APPLIED:
        return dict(_APPLIED[id(d)])
    n = words(variant)
    out = {}
    for r in population_rows():
        t = d.queries[int(r)].text
        s = truncate(t, n)
        if s != t:
            out[int(r)] = s
    return out


def rows() -> list[int]:
    return sorted(short_texts())


def apply_text_override() -> dict[str, int]:
    """Profile variant ``devsN``: the DEV rows longer than N words carry their first N words in this process."""
    from uol.embedding_push import common as ep
    d = ep.release()
    if id(d) in _APPLIED:
        return {"already": 1}
    cut = short_texts()
    for r, t in cut.items():
        d.queries[r].text = t
    _APPLIED[id(d)] = cut
    return {"rows": len(cut), "words": words()}
