from __future__ import annotations

from reliable_simcot.full_conflict_evaluation import (
    exact_mcnemar,
    paired_bootstrap_damage_pp,
)


def test_exact_mcnemar_counts_discordant_pairs() -> None:
    result = exact_mcnemar([True, True, False, False], [False, True, True, False])
    assert result["left_only_correct"] == 1
    assert result["right_only_correct"] == 1
    assert result["two_sided_exact_p"] == 1.0


def test_bootstrap_is_reproducible() -> None:
    clean = [True, True, False, True]
    noisy = [False, True, False, False]
    first = paired_bootstrap_damage_pp(clean, noisy, samples=100, seed=7)
    second = paired_bootstrap_damage_pp(clean, noisy, samples=100, seed=7)
    assert first == second
    assert first["damage_pp"] == 50.0
