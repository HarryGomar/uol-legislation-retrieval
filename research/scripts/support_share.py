"""The supporting-case share of the first ten suggestions (requirement "Supporting cases", 6.7), for the ranker, the model
alone and the final system, by ``uol.analysis_c.f4_provenance``: a suggestion is supported when a similar past tesis cited
it, among the ten nearest development cases of the training folds (``src_knn_qwen8b``) or the ten nearest older tesis
(the historical memory M1), the two collections the prototype shows (``uol.prototype.pipeline``; D203, 4 Nov 2026: the
share counted the development memory alone before). Writes ``research/programs/final_results_20261102/support.json``,
which ``scripts/final_results.py`` reads.

    py -3.12 scripts/support_share.py
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.modules.pop("uol", None)

from uol.analysis_c import f4_provenance  # noqa: E402

RECORD = ROOT / "experiments/runs/final_20261025/store_dev_20261025b/runs"
FOOT = ROOT / "experiments/runs/footing_20261101/dev/store/runs"
VIEW = ROOT / ".cache/final_results/view/runs"
OUT = ROOT / "research/programs/final_results_20261102"
FROM_RECORD = ("S0", "S_full", "S_full_ext", "src_knn_qwen8b")


def link(src: Path, name: str) -> None:
    d = VIEW / name / "DEV"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("per_case.parquet", "top200.parquet", "summary.json"):
        shutil.copyfile(src / name / "DEV" / f, d / f)


def main() -> int:
    sys_id = json.loads((OUT / "results.json").read_text(encoding="utf-8"))["system"]["dev_system"]
    for s in FROM_RECORD:
        link(RECORD, s)
    systems = ["rt_lm_tesis", "gpt61sol_alone", sys_id]   # D209, D210: the chosen ranker
    for s in systems:
        link(FOOT, s)
    out = {"system_id": sys_id}
    for s in systems:
        f4_provenance.SYSTEM = s
        rec = f4_provenance.run(VIEW, ROOT / ".cache/final_results/f4" / s.replace("+", "_"), with_m1=True)
        if rec["notes"]:
            raise RuntimeError(rec["notes"])
        out[s] = {**rec["results"]["either memory"], "development_memory": rec["results"]["development memory"],
                  "historical_memory": rec["results"]["historical memory M1"]}
    (OUT / "support.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
