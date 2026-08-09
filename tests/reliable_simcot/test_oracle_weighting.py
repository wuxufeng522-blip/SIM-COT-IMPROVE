from __future__ import annotations

import pytest

from reliable_simcot.official_adapter import OfficialExample
from reliable_simcot.oracle_weighting import steps_and_weights


def _fixture():
    example = OfficialExample(
        idx=7,
        question="q",
        steps=("<<1+1=2>>", "<<2+1=3>>", "<<3+1=4>>", "<<4+1=5>>", "<<5+1=6>>"),
        answer="6",
    )
    entry = {"noise_position": 2, "corrupted_step": "<<3+1=5>>"}
    return example, entry


def test_oracle_raw_changes_only_target_and_weight() -> None:
    example, entry = _fixture()
    steps, weights = steps_and_weights("oracle_raw_0.1", example, entry)
    assert steps[:2] == example.steps[:2]
    assert steps[2] == entry["corrupted_step"]
    assert steps[3:] == example.steps[3:5]
    assert weights == (1.0, 1.0, 0.1, 1.0, 1.0)


def test_normalized_weights_preserve_mean_one() -> None:
    example, entry = _fixture()
    _, weights = steps_and_weights("oracle_normalized_0.1", example, entry)
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.1 * 5 / 4.1)


def test_clean_arm_does_not_inject_noise() -> None:
    example, entry = _fixture()
    steps, weights = steps_and_weights("clean", example, entry)
    assert steps == example.steps
    assert weights == (1.0,) * 5


def test_two_corruptions_are_both_downweighted() -> None:
    example, entry = _fixture()
    entry["corruptions"] = [
        {
            "position": 2,
            "family": "operand_perturbation",
            "template_id": "first",
            "text": "<<3+1=5>>",
            "text_sha256": "unused",
            "y_valid": 0,
            "y_utility": 0,
        },
        {
            "position": 4,
            "family": "irrelevant_but_correct",
            "template_id": "second",
            "text": "<<9-9=0>>",
            "text_sha256": "unused",
            "y_valid": 1,
            "y_utility": 0,
        },
    ]
    steps, weights = steps_and_weights("oracle_raw_0.1", example, entry)
    assert steps[2] == "<<3+1=5>>"
    assert steps[4] == "<<9-9=0>>"
    assert weights == (1.0, 1.0, 0.1, 1.0, 0.1)


def test_two_corruption_normalization_preserves_mean_one() -> None:
    example, entry = _fixture()
    entry["corruptions"] = [
        {"position": 1, "text": "<<2+1=4>>"},
        {"position": 3, "text": "<<4+1=6>>"},
    ]
    _, weights = steps_and_weights("oracle_normalized_0.1", example, entry)
    assert sum(weights) / 5 == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.15625)
    assert weights[3] == pytest.approx(0.15625)
    assert weights[0] == pytest.approx(1.5625)
