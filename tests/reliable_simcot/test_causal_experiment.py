from __future__ import annotations

import pytest

from reliable_simcot.causal_experiment import (
    CAUSAL_ARMS,
    COVERAGES,
    _balanced,
    _tier_requests,
    causal_steps_and_weights,
)
from reliable_simcot.official_adapter import OfficialExample


def _example() -> OfficialExample:
    return OfficialExample(
        idx=1,
        question="question",
        steps=("s0", "s1", "s2", "s3", "s4"),
        answer="40",
    )


def _entry(tier: int = 0) -> dict:
    return {
        "coverage_tier": tier,
        "chain": {
            "family": "numeric_propagation",
            "pivot": 1,
            "corrupted_steps": ["s0", "bad1", "s2", "bad3", "bad4"],
            "labels": ["CLEAN", "DIRECT_ERROR", "CLEAN", "CAUSAL_DESCENDANT", "CAUSAL_DESCENDANT"],
        },
    }


@pytest.mark.parametrize("size", [128, 512])
def test_latin_cell_cycle_balances_both_marginals(size: int) -> None:
    entries = [
        {"chain": {"family": family, "pivot": pivot}}
        for family, pivot in _tier_requests(size)
    ]
    assert _balanced(entries)


@pytest.mark.parametrize("total,tier_size", [(512, 128), (2048, 512)])
def test_continuous_cycle_balances_every_coverage_prefix(total: int, tier_size: int) -> None:
    requests = _tier_requests(total)
    for prefix in range(tier_size, total + 1, tier_size):
        entries = [
            {"chain": {"family": family, "pivot": pivot}}
            for family, pivot in requests[:prefix]
        ]
        assert _balanced(entries)


def test_coverage_tiers_activate_exact_quarters() -> None:
    examples = [_entry(tier) for tier in range(4) for _ in range(128)]
    for coverage in COVERAGES:
        active = sum(
            causal_steps_and_weights("noisy_equal", _example(), entry, coverage=coverage)[0]
            != _example().steps
            for entry in examples
        )
        assert active == 512 * coverage // 100


def test_six_arm_weight_mapping() -> None:
    expected = {
        "clean": (1.0, 1.0, 1.0, 1.0, 1.0),
        "noisy_equal": (1.0, 1.0, 1.0, 1.0, 1.0),
        "uniform_attenuation": (0.46, 0.46, 0.46, 0.46, 0.46),
        "pivot_only": (1.0, 0.1, 1.0, 1.0, 1.0),
        "causal_raw": (1.0, 0.1, 1.0, 0.1, 0.1),
    }
    for arm, weights in expected.items():
        _, observed = causal_steps_and_weights(arm, _example(), _entry(), coverage=25)
        assert observed == weights
    _, normalized = causal_steps_and_weights(
        "causal_normalized", _example(), _entry(), coverage=25
    )
    assert sum(normalized) == pytest.approx(5.0)
    assert normalized[0] == pytest.approx(5 / 2.3)
    assert normalized[1] == pytest.approx(0.5 / 2.3)


@pytest.mark.parametrize("arm", CAUSAL_ARMS)
def test_inactive_tier_is_clean_in_every_arm(arm: str) -> None:
    steps, weights = causal_steps_and_weights(arm, _example(), _entry(3), coverage=75)
    assert steps == _example().steps
    assert weights == (1.0,) * 5
