"""nDCG@10 as the thesis computes it."""
import numpy as np
import pytest

from uol.evaluate import ndcg_at_10


def test_binary_gains_with_the_ideal_cut_at_ten():
    assert ndcg_at_10([1, 2, 3], {1}, set()) == 1.0
    assert ndcg_at_10([2, 1], {1}, set()) == pytest.approx(1 / np.log2(3))
    assert ndcg_at_10(list(range(20)), set(range(12)), set()) == pytest.approx(1.0)


def test_generic_articles_and_repeats_are_removed_before_the_cut():
    assert ndcg_at_10([9, 9, 1], {1}, {9}) == 1.0
    assert ndcg_at_10([2, 2, 1], {1}, set()) == pytest.approx(1 / np.log2(3))


def test_a_case_without_targets_scores_zero():
    assert ndcg_at_10([1, 2], set(), set()) == 0.0
