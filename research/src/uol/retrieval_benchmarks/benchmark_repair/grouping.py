"""Conservatively merge dependent facts before making development partitions."""
from __future__ import annotations

from collections import defaultdict
from .common import canonical_sha, text_sha
from uol.retrieval_benchmarks.benchmark_validation.splits import near_duplicate_pairs, normalize

PARTITION_SALT = "benchmark-repair-20260929-development-groups-v1"


def role(component: str) -> str:
    return "development_check" if int(text_sha(PARTITION_SALT + ":" + component)[:8], 16) % 5 == 0 else "development_fit"


def merge_groups(sources: list[dict], masked: dict[str, str], *, threshold: float = .8) -> tuple[dict[str, str], dict]:
    by_id = {row["query_id"]: row for row in sources}
    if len(by_id) != len(sources) or set(masked) != set(by_id):
        raise ValueError("Query identities must be unique and identical across views")
    parents = {key: key for key in by_id}

    def find(key):
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left, right):
        a, b = sorted((find(left), find(right)))
        parents[b] = a

    links = []
    for field in ("component_id", "raw_hechos", "masked"):
        seen = {}
        for row in sorted(sources, key=lambda r: r["query_id"]):
            query = row["query_id"]
            value = masked[query] if field == "masked" else row[field]
            if field != "component_id":
                value = normalize(value)
            if not value:
                continue
            if value in seen:
                union(query, seen[value])
                links.append({"query_ids": [seen[value], query], "reason": field + "_exact"})
            else:
                seen[value] = query

    searches = {}
    for view in ("raw_hechos", "masked"):
        # The safe export's original TRAIN role is only the audit function's
        # access constraint. It is not the successor's evaluation partition.
        rows = [{"query_id": row["query_id"], "component_id": row["component_id"], "split": "train",
                 "text": masked[row["query_id"]] if view == "masked" else row[view]} for row in sources]
        pairs, searches[view] = near_duplicate_pairs(rows, threshold)
        for pair in pairs:
            left, right = pair["query_ids"]
            union(left, right)
            links.append({"query_ids": [left, right], "reason": view + "_jaccard_word5", "value": pair["jaccard_word5"]})

    grouped = defaultdict(list)
    for query in sorted(by_id):
        grouped[find(query)].append(query)
    mapping, components = {}, []
    for members in grouped.values():
        component = "repair-component:" + canonical_sha(members)[:24]
        old = sorted({by_id[query]["component_id"] for query in members})
        for query in members:
            mapping[query] = component
        components.append({"component_id": component, "query_ids": members, "original_component_ids": old,
                           "development_role": role(component)})
    return mapping, {"components": sorted(components, key=lambda r: r["component_id"]), "links": links,
                     "searches": searches, "original_components": len({r["component_id"] for r in sources}),
                     "repaired_components": len(components), "merged_original_component_groups": sum(len(c["original_component_ids"]) > 1 for c in components),
                     "partition_salt": PARTITION_SALT, "threshold": threshold,
                     "scope": "Conservative development dependence groups; no proof of independence or unexposed final evaluation."}
