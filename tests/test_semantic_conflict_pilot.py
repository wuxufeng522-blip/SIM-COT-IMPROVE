from reliable_simcot.error_cancellation_data import parse_equation
from reliable_simcot.semantic_conflict_pilot import (
    _compatible,
    _jaccard,
    redundant_steps,
    supervision_targets,
)


class Example:
    def __init__(self, question, steps, answer):
        self.question = question
        self.steps = tuple(steps)
        self.answer = answer


def row(identifier, question, answer, prefix):
    steps = [f"<<{prefix}+{index}={prefix + index}>>" for index in range(5)]
    return {
        "question_id": identifier,
        "example": Example(question, steps, str(answer)),
        "step_tokens": 50,
    }


def test_question_jaccard_detects_unrelated_text():
    assert _jaccard("Alice buys red apples", "A train crosses a long bridge") == 0.0


def test_compatible_requires_cross_question_semantic_conflict():
    config = {
        "max_question_jaccard": 0.35,
        "min_different_result_slots": 4,
        "min_step_token_ratio": 0.75,
        "max_step_token_ratio": 1.34,
    }
    recipient = row("a", "Alice buys red apples", 10, 10)
    donor = row("b", "A train crosses a long bridge", 99, 100)
    assert _compatible(recipient, donor, config)
    assert not _compatible(recipient, recipient, config)


def test_redundant_steps_are_true_and_add_identity_operation():
    clean = ("<<4*.6=2.4>>", "<<2+3=5>>", "<<9-1=8>>", "<<2*4=8>>", "<<8/2=4>>")
    transformed = redundant_steps(clean)
    assert len(transformed) == 5
    assert all("+0=" in step for step in transformed)
    assert all(parse_equation(step).is_true for step in transformed)


def test_supervision_targets_freeze_kind_and_weight():
    manifest_row = {
        "clean_steps": ["<<1+1=2>>"] * 5,
        "semantic_conflict": {"steps": ["<<9+9=18>>"] * 5},
    }
    config = {
        "arm_specs": {
            "clean": {"target_kind": "clean", "step_weight": 1.0},
            "redundant_w01": {"target_kind": "redundant", "step_weight": 0.1},
            "error_w01": {"target_kind": "semantic_conflict", "step_weight": 0.1},
        }
    }
    assert supervision_targets("clean", manifest_row, config)[1] == (1.0,) * 5
    assert supervision_targets("redundant_w01", manifest_row, config)[1] == (0.1,) * 5
    assert supervision_targets("error_w01", manifest_row, config)[0] == ("<<9+9=18>>",) * 5
