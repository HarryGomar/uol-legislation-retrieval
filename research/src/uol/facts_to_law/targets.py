"""Analysis-only target typing (PREREGISTRATION §1).

breadth_v1 (primary): an article is generic if at least ``share`` of eligible
source components cite it and those citing tesis span at least ``min_materias``
distinct materias. law_v1 (sensitivity): every CPEUM or Ley de Amparo article.
No system reads these labels.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

from .spec import GENERIC, SPECIFIC, TargetTypeRecord

LAW_V1_INSTRUMENTS = ("/CPEUM.pdf", "/LAmp.pdf")


def type_targets(rows: Iterable[Mapping], *, share: float = 0.01, min_materias: int = 3) -> list[TargetTypeRecord]:
    """rows: {component_id, materias: list[str], positive_ids: list[str]} per eligible query."""
    components: dict[str, set[str]] = defaultdict(set)
    materias: dict[str, set[str]] = defaultdict(set)
    all_components: set[str] = set()
    for row in rows:
        all_components.add(row["component_id"])
        for candidate in row["positive_ids"]:
            components[candidate].add(row["component_id"])
            materias[candidate].update(str(m).strip() for m in row["materias"] if str(m).strip())
    n = len(all_components)
    out = []
    for candidate in sorted(components):
        count = len(components[candidate])
        breadth = count / n >= share and len(materias[candidate]) >= min_materias
        law = candidate.split("#articulo=")[0].endswith(LAW_V1_INSTRUMENTS)
        out.append(TargetTypeRecord(
            candidate_id=candidate,
            target_type=GENERIC if breadth else SPECIFIC,
            rule=f"breadth_v1(share>={share},materias>={min_materias})",
            stats={"components": count, "share": round(count / n, 6), "materias": sorted(materias[candidate]),
                   "law_v1": GENERIC if law else SPECIFIC},
        ))
    return out
