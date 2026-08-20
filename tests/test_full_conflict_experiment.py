from __future__ import annotations

from reliable_simcot.full_conflict_experiment import arm_targets
from reliable_simcot.official_adapter import OfficialExample


def _example() -> OfficialExample:
    return OfficialExample(
        1,
        "Use 3, 8, 5, 2, and 4.",
        (
            "<<3*8=24>>",
            "<<24-5=19>>",
            "<<19*2=38>>",
            "<<38+4=42>>",
            "<<42-2=40>>",
        ),
        "40",
    )


def _entry(tier=0):
    return {
        "coverage_tier": tier,
        "local_chain": {"corrupted_steps": [f"<<{i}+1={i+1}>>" for i in range(5)]},
        "full_chain": {"steps": [f"<<{i}+2={i+2}>>" for i in range(5)]},
    }


def test_answer_only_has_zero_auxiliary_scale() -> None:
    steps, scale, contaminated = arm_targets("answer_only", _example(), _entry())
    assert steps == _example().steps
    assert scale == 0.0
    assert contaminated == 0


def test_local_and_full25_activate_the_same_tier_zero_question() -> None:
    local = arm_targets("local_causal_25", _example(), _entry(0))
    full = arm_targets("full_conflict_25", _example(), _entry(0))
    assert local[2] == 3
    assert full[2] == 5


def test_full50_adds_tier_one_but_full25_does_not() -> None:
    assert arm_targets("full_conflict_25", _example(), _entry(1))[2] == 0
    assert arm_targets("full_conflict_50", _example(), _entry(1))[2] == 5
