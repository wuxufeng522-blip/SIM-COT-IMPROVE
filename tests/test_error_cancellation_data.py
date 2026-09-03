from fractions import Fraction

from reliable_simcot.error_cancellation_data import (
    VARIANT_TYPES,
    construct_severe_variants,
    construct_variants,
    evaluate_arithmetic,
    parse_equation,
)
from reliable_simcot.official_adapter import OfficialExample


def fixture() -> OfficialExample:
    return OfficialExample(
        idx=0,
        question="A deterministic chain",
        steps=(
            "<<20/2=10>>",
            "<<10+5=15>>",
            "<<15*2=30>>",
            "<<30-6=24>>",
            "<<24/3=8>>",
        ),
        answer="8",
    )


def test_safe_arithmetic_parser() -> None:
    assert evaluate_arithmetic("(3+5)/2") == Fraction(4)


def test_local_and_wide_error_cancellation_truth_patterns() -> None:
    variants = construct_variants(fixture(), delta=1)
    for name, expected_types in VARIANT_TYPES.items():
        assert tuple(variants[name]["types"]) == expected_types
        truth = tuple(parse_equation(step).is_true for step in variants[name]["steps"])
        expected_truth = tuple(
            step_type not in {"DIRECT_FALSE", "CANCEL_FALSE"}
            for step_type in expected_types
        )
        assert truth == expected_truth
        assert parse_equation(variants[name]["steps"][-1]).result_value == 8
    assert variants["local_error"]["steps"][3:] == list(fixture().steps[3:])
    assert variants["wide_error"]["steps"][4] == fixture().steps[4]


def test_redundancy_is_present_at_every_matched_slot() -> None:
    variants = construct_variants(fixture(), delta=1)
    for name in ("local_redundant", "wide_redundant"):
        for step, step_type in zip(variants[name]["steps"], variants[name]["types"]):
            if step_type == "REDUNDANT":
                assert "+0" in parse_equation(step).expression


def test_v10_errors_remain_compact_while_redundancy_uses_neutral_ops() -> None:
    variants = construct_variants(fixture(), delta=1, neutralize_errors=False)
    for name in ("local_error", "wide_error"):
        assert all(
            "+0" not in parse_equation(step).expression
            for step, step_type in zip(variants[name]["steps"], variants[name]["types"])
            if step_type != "CLEAN"
        )
    for name in ("local_redundant", "wide_redundant"):
        assert all(
            "+0" in parse_equation(step).expression
            for step, step_type in zip(variants[name]["steps"], variants[name]["types"])
            if step_type == "REDUNDANT"
        )


def test_rejects_broken_dependency_chain() -> None:
    example = fixture()
    broken = OfficialExample(
        idx=example.idx,
        question=example.question,
        steps=(example.steps[0], "<<9+5=14>>", *example.steps[2:]),
        answer=example.answer,
    )
    try:
        construct_variants(broken, delta=1)
    except ValueError:
        pass
    else:
        raise AssertionError("broken clean/dependency chain was accepted")


def test_severe_conflict_is_large_propagated_and_answer_preserving() -> None:
    variants, delta, metrics = construct_severe_variants(
        fixture(),
        question_id="0" * 64,
        severity_multiplier=10,
        severity_floor=10,
        downstream_min_relative=0.5,
    )
    clean = [parse_equation(step) for step in variants["clean"]["steps"]]
    wide = [parse_equation(step) for step in variants["wide_error"]["steps"]]
    local = [parse_equation(step) for step in variants["local_error"]["steps"]]
    assert delta > 0
    assert metrics["local_direct_relative_deviation"] >= 10
    assert metrics["wide_relative_deviations"][0] >= 10
    assert min(metrics["wide_relative_deviations"][1:]) >= 0.5
    assert all(wide[index].result_value != clean[index].result_value for index in range(3))
    assert wide[-1].result_value == local[-1].result_value == Fraction(8)
    assert variants["wide_error"]["steps"][-1] == fixture().steps[-1]


def test_propagation_replaces_one_state_reference_not_equal_constant() -> None:
    example = OfficialExample(
        idx=1,
        question="A repeated constant can resemble the carried state",
        steps=(
            "<<1*2=2>>",
            "<<2*2=4>>",
            "<<4*2=8>>",
            "<<8*2=16>>",
            "<<16-8=8>>",
        ),
        answer="8",
    )
    variants = construct_variants(example, delta=-10, neutralize_errors=False)
    assert variants["wide_error"]["steps"][1] == "<<-8*2=-16>>"
