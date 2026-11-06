"""The test set stays closed: every step of the whole path that loads a test row stops before it reads anything."""
from __future__ import annotations

import inspect

import pytest

from uol.final import whole_run as W
from uol.final import whole_test as WT


@pytest.fixture
def closed(tmp_path, monkeypatch):
    monkeypatch.setattr(W, "PROG", tmp_path)
    monkeypatch.setattr(WT, "RELEASE", tmp_path / "TEST_RELEASE.json")


def touched(*a, **k):
    raise AssertionError("a test row was reached before the release")


@pytest.mark.parametrize("step", [lambda: WT.need_release(), lambda: WT.new_cases(), lambda: WT.load_only("ranker"),
                                  lambda: WT.rank("ranker", "card", 1), lambda: WT.score()])
def test_each_step_stops_without_the_release(closed, monkeypatch, step):
    monkeypatch.setattr(WT, "rows_and_dates", touched)
    monkeypatch.setattr(WT, "blank_vectors", touched)
    with pytest.raises(PermissionError):
        step()


def test_live_refuses_new_cases_without_the_release(closed):
    with pytest.raises(PermissionError):
        W.Live("ranker", "stored", "guard", new_cases={"rows": [0]})


def test_the_release_opens_the_guard(closed, tmp_path):
    (tmp_path / "TEST_RELEASE.json").write_text('{"word": "test", "scope": {"a": 1}}', encoding="utf-8")
    assert W.need_test_release()["word"] == "test"


def test_a_release_without_word_or_scope_does_not_open(closed, tmp_path):
    (tmp_path / "TEST_RELEASE.json").write_text('{"word": "test"}', encoding="utf-8")
    with pytest.raises(PermissionError):
        W.need_test_release()


def test_score_reads_no_stored_answers_of_the_model_alone():
    src = inspect.getsource(WT.score)
    assert "alone_answers" not in src and "per_case_alone" not in src and '"model_list"' in src


def test_the_models_list_goes_to_the_pinned_model():
    src = inspect.getsource(W.ModelSteps.model_list)
    assert "model=self.luna.MODEL" in src


def test_the_live_answers_use_the_runs_own_cache():
    src = inspect.getsource(W.ModelSteps.__init__)
    assert "luna.CACHE = own" in src and "CX.bind = bind_own" in src
