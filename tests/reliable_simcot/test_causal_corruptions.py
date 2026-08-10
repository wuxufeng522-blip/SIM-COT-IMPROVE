from __future__ import annotations

from reliable_simcot.causal_corruptions import (
    build_causal_chain,
    causal_chains_by_cell,
    extract_dependency_edges,
    parse_all_equations,
    three_node_paths,
)
from reliable_simcot.official_adapter import OfficialExample


def _example() -> OfficialExample:
    return OfficialExample(
        idx=1,
        question="A value is calculated from 3 groups of 8, then 5 is removed, doubled, increased by 4, and reduced by 2.",
        steps=(
            "<<3*8=24>>",
            "<<24-5=19>>",
            "<<19*2=38>>",
            "<<38+4=42>>",
            "<<42-2=40>>",
        ),
        answer="40",
    )


def test_dependency_path_is_unique_and_ordered() -> None:
    example = _example()
    equations = parse_all_equations(example)
    assert equations is not None
    edges = extract_dependency_edges(example, equations)
    assert (2, 3, 4) in three_node_paths(edges)


def test_numeric_error_propagates_to_two_descendants() -> None:
    chain = build_causal_chain(
        _example(), family="numeric_propagation", path=(2, 3, 4)
    )
    assert chain is not None
    assert chain.corrupted_steps[2:] == (
        "<<19*2=39>>",
        "<<39+4=43>>",
        "<<43-2=41>>",
    )
    assert chain.labels == (
        "CLEAN",
        "CLEAN",
        "DIRECT_ERROR",
        "CAUSAL_DESCENDANT",
        "CAUSAL_DESCENDANT",
    )
    assert str(chain.propagated_final_value) == "41"


def test_operator_error_is_locally_valid_then_propagates() -> None:
    chain = build_causal_chain(
        _example(), family="operator_propagation", path=(2, 3, 4)
    )
    assert chain is not None
    assert chain.corrupted_steps[2] == "<<19/2=9.5>>"
    assert chain.corrupted_steps[3] == "<<9.5+4=13.5>>"
    assert chain.corrupted_steps[4] == "<<13.5-2=11.5>>"


def test_quantity_error_uses_unique_question_operand() -> None:
    chain = build_causal_chain(
        _example(), family="quantity_propagation", path=(2, 3, 4)
    )
    assert chain is not None
    assert chain.corrupted_steps[2] == "<<19*1=19>>"
    assert chain.corrupted_steps[4] == "<<23-2=21>>"


def test_cell_enumeration_exposes_all_three_families() -> None:
    cells = causal_chains_by_cell(_example())
    assert {family for family, pivot in cells if pivot == 2} == {
        "numeric_propagation",
        "operator_propagation",
        "quantity_propagation",
    }


def test_replay_preserves_binary_plus_before_consumed_value() -> None:
    example = OfficialExample(
        idx=2,
        question="Use 3 groups of 8, remove 5, then combine the result with 2, 4, and 1.",
        steps=(
            "<<3*8=24>>",
            "<<24-5=19>>",
            "<<2+19=21>>",
            "<<4+21=25>>",
            "<<1+25=26>>",
        ),
        answer="26",
    )
    chain = build_causal_chain(
        example, family="numeric_propagation", path=(2, 3, 4)
    )
    assert chain is not None
    assert chain.corrupted_steps[3] == "<<4+22=26>>"
    assert chain.corrupted_steps[4] == "<<1+26=27>>"
    assert str(chain.propagated_final_value) == "27"
