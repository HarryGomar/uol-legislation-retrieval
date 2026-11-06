"""Development cases whose facts name a provision, and how many of them name a cited article. Metadata flags of release v11,
development rows only (role from the release's split, row order of queries.jsonl); no facts text and no test row is used."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
OUT = Path(__file__).parent / "out" / "named_cited_dev.json"


def main() -> int:
    from uol.embedding_push import common as ep
    dev = {int(r) for r in ep.dev_rows()}
    n_named = n_cited = n_seen = 0
    with (ROOT / "data/releases/uol_c01_statutes_20261004_v11_stratified/queries.jsonl").open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i not in dev:
                continue
            q = json.loads(line)
            n_seen += 1
            flags = {**q, **(q.get("flags") or {})}
            if flags.get("stratum") == "S2_masked_citation":
                n_named += 1
                n_cited += bool(flags.get("answer_named_in_facts"))
    res = {"development_rows": n_seen, "facts_name_a_provision": n_named, "also_name_a_cited_article": n_cited, "share": n_cited / n_named if n_named else None}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
