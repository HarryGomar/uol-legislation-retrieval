"""Read-only audit of the historical GraphingUOL assets."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from uol.retrieval_benchmarks.body_backed_citations.common import normalize_identity, sha256_file


LEGACY_ROOT = Path(r"C:\uol_legacy_home\OneDrive\1.UniversoDeLeyes\Graphs\GraphingUOL")


def audit_legacy(instruments: list[dict[str, Any]]) -> dict[str, Any]:
    files = {
        "CTRL-F.py": LEGACY_ROOT / "CTRL-F.py",
        "NER.py": LEGACY_ROOT / "NER.py",
        "NERUNAM.py": LEGACY_ROOT / "NERUNAM.py",
        "CleanUNAMNodes.py": LEGACY_ROOT / "CleanUNAMNodes.py",
        "exterior_patterns.txt": LEGACY_ROOT / "MISC/exterior_patterns.txt",
        "meta.json": LEGACY_ROOT / "UOLMODEL/meta.json",
        "config.cfg": LEGACY_ROOT / "UOLMODEL/config.cfg",
        "examplesArtsMissed.txt": LEGACY_ROOT / "MISC/examplesArtsMissed.txt",
        "difference.csv": LEGACY_ROOT / "GraphResults/difference.csv",
    }
    receipts = {
        name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in files.items()
    }
    model_files = [path for path in (LEGACY_ROOT / "UOLMODEL").rglob("*") if path.is_file()]
    model_inventory = [
        {"path": str(path.relative_to(LEGACY_ROOT)), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(model_files)
    ]
    meta = json.loads(files["meta.json"].read_text(encoding="utf-8"))

    exact_titles: dict[str, list[str]] = {}
    for row in instruments:
        exact_titles.setdefault(normalize_identity(row["title"]), []).append(row["instrument_id"])
    proposals: list[dict[str, Any]] = []
    accepted: dict[str, list[str]] = {}
    lines = [line.strip() for line in files["exterior_patterns.txt"].read_text(encoding="utf-8").splitlines() if line.strip()]
    for alias in sorted(set(lines)):
        matches = exact_titles.get(normalize_identity(alias), [])
        status = "accepted_exact_unique" if len(matches) == 1 else "excluded_zero_match" if not matches else "excluded_multi_match"
        proposals.append({"alias": alias, "instrument_ids": matches, "status": status})
        if len(matches) == 1:
            accepted.setdefault(matches[0], []).append(alias)

    with files["difference.csv"].open(encoding="utf-8-sig", newline="") as stream:
        difference_rows = list(csv.DictReader(stream))
    return {
        "authority": "read_only_historical_input",
        "files": receipts,
        "model": {
            "weights_present": any(row["path"].endswith("transformer\\model") or row["path"].endswith("transformer/model") for row in model_inventory),
            "metadata": meta,
            "inventory": model_inventory,
            "execution_status": "not_run_after_benchmark_gate_failure",
            "role_if_run": "proposal_only_never_gold",
        },
        "known_code_hazards": [
            "CTRL-F.py find_number_from_law can invent identifiers",
            "historical article carry-forward can attach an article to the wrong law",
            "historical fuzzy cleanup can drop digits and merge distinct nodes",
        ],
        "exterior_pattern_lines": len(lines),
        "exterior_pattern_distinct": len(set(lines)),
        "alias_proposals": proposals,
        "accepted_aliases_by_instrument": accepted,
        "accepted_alias_count": sum(len(values) for values in accepted.values()),
        "difference_rows": len(difference_rows),
        "difference_role": "audit_only_not_gold",
    }
