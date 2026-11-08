"""The ranker without the encoder against the thesis's stored lists. Needs the benchmark (``UOL_DATA``) and the
fitted models (``UOL_MODELS``, default ``models``); skipped otherwise."""
import os
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("UOL_MODELS", ROOT / "models"))
needs = pytest.mark.skipif(not os.environ.get("UOL_DATA") or not (MODELS / "noenc" / "group0.txt").exists(),
                           reason="needs UOL_DATA and the fitted models")


@needs
def test_noenc_fold0_matches_the_stored_lists():
    from uol import data, run
    bench = data.load(include_test_labels=False)
    stored = pd.read_parquet(ROOT / "experiments" / "results" / "dev_lists" / "ranker_noenc.parquet")
    rows = [int(r) for r in bench.evaluation_rows() if bench.fold[r] == 0][:10]
    ranker = run.Run(bench, "noenc", 0, MODELS, None).ranker
    for r in rows:
        ref = stored[stored.query_id == bench.cases[r]["query_id"]].sort_values("rank").candidate_idx.tolist()
        assert ranker.rank(bench.texts[r], bench.dates[r])[:10] == ref[:10]
