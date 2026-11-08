"""A development rehearsal must not open an available test-label file."""
import json
from pathlib import Path

from uol.data import load


def test_development_only_loading_never_opens_test_labels(tmp_path, monkeypatch):
    (tmp_path / "labels").mkdir()
    (tmp_path / "cases.jsonl").write_text(json.dumps({"query_id": "dev"}) + "\n")
    (tmp_path / "catalogue").mkdir()
    (tmp_path / "catalogue/articles.jsonl").write_text(json.dumps({"candidate_id": "a"}) + "\n")
    (tmp_path / "labels/dev_corrected.jsonl").write_text(json.dumps({"query_id": "dev", "positives": [{"candidate_id": "a"}]}) + "\n")
    test_path = tmp_path / "labels/test.jsonl"
    test_path.write_text("This file must never be opened")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path == test_path:
            raise AssertionError("development consumer opened test labels")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    bench = load(tmp_path, include_test_labels=False)
    assert bench.positives() == {0: [0]}
