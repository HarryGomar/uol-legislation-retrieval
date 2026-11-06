"""Exact, typed legislative graph construction."""
from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping

from uol.retrieval_benchmarks.body_backed_citations.common import CandidateIndex, normalize_identity


_ARTICLE = re.compile(r"(?i)\bart(?:i|í)culos?\s+(?P<label>\d+[a-z]?(?:\s*(?:bis|ter|qu[aá]ter))?)")


def exact_direct_references(
    provisions: list[dict[str, Any]],
    candidate_index: CandidateIndex,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve only literal full-title references witnessed in provision text."""
    title_to_ids: dict[str, set[str]] = defaultdict(set)
    title_by_id: dict[str, str] = {}
    for row in provisions:
        instrument_id = str(row.get("instrument_id") or "")
        title = str(row.get("instrument_title") or "")
        if instrument_id and title:
            title_to_ids[normalize_identity(title)].add(instrument_id)
            title_by_id[instrument_id] = title
    unique = {title: next(iter(values)) for title, values in title_to_ids.items() if len(values) == 1 and len(title) >= 12}
    phrase = re.compile(
        r"(?<!\w)(?:" + "|".join(re.escape(title) for title in sorted(unique, key=len, reverse=True)) + r")(?!\w)",
        re.IGNORECASE,
    )
    instrument_edges: set[tuple[str, str]] = set()
    article_edges: set[tuple[str, str]] = set()
    for row in provisions:
        source_id = str(row["candidate_id"])
        folded = normalize_identity(row.get("text") or "")
        for match in phrase.finditer(folded):
            target_instrument = unique[normalize_identity(match.group(0))]
            instrument_edges.add((source_id, target_instrument))
            window = folded[max(0, match.start() - 120):min(len(folded), match.end() + 160)]
            for article in _ARTICLE.finditer(window):
                resolution = candidate_index.resolve(target_instrument, article.group("label"))
                if resolution.status == "resolved":
                    article_edges.update((source_id, target_id) for target_id in resolution.candidate_ids)
    return (
        [{"source_id": source, "target_id": target, "relation_type": "exact_direct_instrument_reference"} for source, target in sorted(instrument_edges)],
        [{"source_id": source, "target_id": target, "relation_type": "exact_direct_article_reference"} for source, target in sorted(article_edges)],
    )
