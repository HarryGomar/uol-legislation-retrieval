"""The rules of the first stage and the ranker that the stored results depend on."""
import numpy as np

from uol import features as FX, ranker as RK
from uol.history import CitationMemory
from uol.sources import spread, votes


def test_top_breaks_ties_towards_the_lower_article():
    x = np.array([5, 1, 3, 3, 3, 0], dtype=np.float32)
    assert sorted(FX.top(x, 3).tolist()) == [0, 2, 3]


def test_sanitize_puts_missing_scores_below_every_finite_one():
    x = FX.sanitize(np.array([-np.inf, 2.0, -1.0], dtype=np.float32))
    assert x.tolist() == [-2.0, 2.0, -1.0]


def test_source_columns():
    x = np.array([0, 1, 2, 3, 10], dtype=np.float32)
    law = np.array([0, 0, 1, 1, 1])
    cols = FX.source_columns(x, np.array([1, 4]), law)
    scale = 1.4826 * 1.0                      # median 2, median absolute deviation 1
    np.testing.assert_allclose(cols[:, 0], [(1 - 2) / scale, (10 - 2) / scale], rtol=1e-6)
    np.testing.assert_allclose(cols[:, 1], np.log([4, 1]), rtol=1e-6)          # 1 + number scoring higher
    np.testing.assert_allclose(cols[:, 2], [(1 - 2) / scale, (10 - 2) / scale], rtol=1e-6)   # best of the same law


def test_votes_weigh_the_nearest_cases_by_similarity():
    cited = spread([(0, 1), (2,), (3,)], 4)
    v = votes(np.array([0.6, 0.2, -0.5], dtype=np.float32), cited, k=2)
    np.testing.assert_allclose(v, [0.375, 0.375, 0.25, 0.0], rtol=1e-6)        # 0.75 split over two articles


def test_training_lists_keep_every_positive_and_cap_the_negatives():
    case = np.repeat([0, 1, 2], [6, 4, 3])
    y = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])               # case 1 has no positive
    keep, yy, counts = RK.training_lists(case, y, np.random.default_rng(0), negatives=2)
    assert counts.tolist() == [3, 3]
    assert set(case[keep]) == {0, 2} and yy.sum() == 2 and np.all(np.diff(case[keep]) >= 0)


def test_older_tesis_votes_halve_with_age():
    class Bench:
        history = {"date": np.array(["2010-01-01", "2015-01-01"], dtype="datetime64[D]"), "cited": [(0,), (1,)]}
        article_ids = ["a", "b", "c"]
    memory = CitationMemory(Bench())
    near = (np.array([0, 1]), np.array([1.0, 1.0], dtype=np.float32))
    cols = memory.columns(near, np.datetime64("2020-01-01"), np.array([0, 1, 2]), k=2, half_life=5.0)
    np.testing.assert_allclose(cols[:, 0], [1 / 3, 2 / 3, 0], atol=1e-3)       # ten and five years old
    assert cols[:, 1].tolist() == [1, 1, 0] and cols[1, 2] == 0
