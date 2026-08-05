from __future__ import annotations

from reliable_simcot.audit import audit_step
from reliable_simcot.corruptions import (
    DEVELOPMENT_FAMILIES,
    compensating_error_variant,
    development_variants,
    equivalent_variant,
)


def test_development_families_have_registered_labels() -> None:
    step = "<<20*5=100>>"
    variants = development_variants(
        step,
        prefix_steps=("<<10+10=20>>",),
        later_steps=("<<100-30=70>>",),
        step_index=1,
    )
    assert {variant.family for variant in variants} == set(DEVELOPMENT_FAMILIES)
    by_family = {variant.family: variant for variant in variants}
    for family in (
        "numeric_error",
        "operator_relation_error",
        "dependency_order_error",
    ):
        assert by_family[family].y_valid == 0
        assert by_family[family].y_utility is None
    for family in ("irrelevant_but_correct", "redundant_repeat"):
        assert by_family[family].y_valid == 1
        assert by_family[family].y_utility == 0

    for family in ("numeric_error", "operator_relation_error", "dependency_order_error"):
        audited = audit_step(
            by_family[family].text,
            step_index=0,
            all_steps=(by_family[family].text,),
            answer="100",
        )
        assert audited.arithmetic_status == "checked_mismatch"


def test_equivalent_and_low_utility_variants_remain_arithmetic_truths() -> None:
    step = "<<20*5=100>>"
    equivalent = equivalent_variant(step)
    assert equivalent is not None
    assert equivalent.y_valid == 1 and equivalent.y_utility == 1
    variants = development_variants(
        step,
        prefix_steps=("<<10+10=20>>",),
        later_steps=("<<100-30=70>>",),
        step_index=1,
    )
    for variant in (equivalent, *variants):
        if variant.y_valid != 1:
            continue
        audited = audit_step(
            variant.text,
            step_index=0,
            all_steps=(variant.text,),
            answer=variant.text.split("=")[-1].rstrip(">>"),
        )
        assert audited.arithmetic_status == "checked_match"


def test_compensating_error_preserves_final_number_but_contains_false_claim() -> None:
    variant = compensating_error_variant("<<20*5=100>>")
    assert variant is not None
    assert variant.family == "compensating_error"
    assert variant.y_valid == 0
    assert variant.y_utility is None
    assert variant.metadata["final_result_preserved"] is True
    assert variant.text.endswith("-1=100>>")
