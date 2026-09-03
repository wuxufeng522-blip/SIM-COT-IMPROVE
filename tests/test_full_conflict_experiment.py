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
        "full_chain": {
            "steps": (
                "<<3+2=5>>",
                "<<5*3=15>>",
                "<<15+7=22>>",
                "<<22*2=44>>",
                "<<44+9=53>>",
            )
        },
    }


def _entry_with_unrelated_donor(tier=0):
    entry = _entry(tier)
    entry["unrelated_donor"] = {
        "question_id": "donor",
        "steps": (
            "<<9+2=11>>",
            "<<11*3=33>>",
            "<<33-5=28>>",
            "<<28+8=36>>",
            "<<36*2=72>>",
        ),
    }
    return entry


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


def test_reverse_steps_100_changes_only_step_order() -> None:
    steps, scale, changed = arm_targets("reverse_steps_100", _example(), _entry())
    assert steps == tuple(reversed(_example().steps))
    assert sorted(steps) == sorted(_example().steps)
    assert scale == 1.0
    assert changed == 4


def test_redundant_steps_50_adds_only_neutral_operations_on_active_rows() -> None:
    steps, scale, changed = arm_targets("redundant_steps_50", _example(), _entry(0))
    assert steps[0] == "<<3*8=24>> <<24+0=24>>"
    assert steps[-1] == "<<42-2=40>> <<40+0=40>>"
    assert scale == 1.0
    assert changed == 5
    assert arm_targets("redundant_steps_50", _example(), _entry(None))[0] == _example().steps


def test_reverse_steps_50_uses_the_same_active_half() -> None:
    active = arm_targets("reverse_steps_50", _example(), _entry(1))
    inactive = arm_targets("reverse_steps_50", _example(), _entry(None))
    assert active[0] == tuple(reversed(_example().steps))
    assert active[2] == 4
    assert inactive[0] == _example().steps
    assert inactive[2] == 0


def test_accidental_correct_50_keeps_wrong_prefix_and_cancels_to_answer() -> None:
    steps, scale, changed = arm_targets("accidental_correct_50", _example(), _entry(0))
    assert steps[:4] == tuple(_entry(0)["full_chain"]["steps"][:4])
    assert steps[4] == "<<44-4=40>>"
    assert scale == 1.0
    assert changed == 5


def test_accidental_correct_50_leaves_inactive_rows_clean() -> None:
    steps, scale, changed = arm_targets("accidental_correct_50", _example(), _entry(None))
    assert steps == _example().steps
    assert scale == 1.0
    assert changed == 0


def test_accidental_correct_50_avoids_a_zero_offset_shortcut() -> None:
    example = OfficialExample(
        2,
        "A deliberately wrong path already lands on the answer value.",
        _example().steps[:-1] + ("<<42+2=44>>",),
        "44",
    )
    steps, _, _ = arm_targets("accidental_correct_50", example, _entry(0))
    assert steps[4] == "<<44+1-1=44>>"


def test_unrelated_accidental_correct_uses_donor_prefix_and_target_answer() -> None:
    steps, scale, changed = arm_targets(
        "unrelated_accidental_correct_50", _example(), _entry_with_unrelated_donor(0)
    )
    assert steps[:4] == tuple(_entry_with_unrelated_donor(0)["unrelated_donor"]["steps"][:4])
    assert steps[4] == "<<36+4=40>>"
    assert scale == 1.0
    assert changed == 5


def test_unrelated_accidental_correct_leaves_inactive_rows_clean() -> None:
    steps, scale, changed = arm_targets(
        "unrelated_accidental_correct_50", _example(), _entry_with_unrelated_donor(None)
    )
    assert steps == _example().steps
    assert scale == 1.0
    assert changed == 0
