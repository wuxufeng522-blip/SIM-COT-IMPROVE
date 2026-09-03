from __future__ import annotations

import pytest

from reliable_simcot.self_corrected_evaluation import (
    hierarchical_paired_bootstrap,
    normalize_answer_for_em,
)


def test_normalized_em_removes_math_formatting_noise() -> None:
    assert normalize_answer_for_em(r"\boxed{34,\!650}") == "34650"
    assert normalize_answer_for_em(" $ 7 / 2 $. ") == "7/2"


def test_hierarchical_bootstrap_preserves_paired_positive_effect() -> None:
    clean = [[True, True, False, True], [True, False, True, True], [True] * 4]
    noisy = [[False, True, False, False], [True, False, False, True], [False] * 4]
    result = hierarchical_paired_bootstrap(
        clean, noisy, samples=1000, seed=123
    )
    assert result["effect_pp"] == pytest.approx((50 + 25 + 100) / 3)
    assert result["positive_seed_count"] == 3
    assert result["ci95_low_pp"] > 0


def test_hierarchical_bootstrap_is_deterministic() -> None:
    left = [[True, False], [False, True], [True, True]]
    right = [[False, False], [False, False], [True, False]]
    first = hierarchical_paired_bootstrap(left, right, samples=200, seed=9)
    second = hierarchical_paired_bootstrap(left, right, samples=200, seed=9)
    assert first == second


def test_hierarchical_bootstrap_rejects_mismatched_questions() -> None:
    with pytest.raises(ValueError, match="equal"):
        hierarchical_paired_bootstrap(
            [[True], [False, True]], [[False], [True]], samples=10, seed=1
        )
