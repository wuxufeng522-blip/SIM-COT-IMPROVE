from __future__ import annotations

import copy

import pytest

from reliable_simcot.self_corrected_data import canonical_hash, construct_five_tuple
from reliable_simcot.self_corrected_experiment import (
    ARMS,
    create_training_schedule,
    steps_and_weights,
    training_directory,
    validate_human_audit,
    verify_training_schedule,
)


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(char) for char in text if not char.isspace()]


def _tuple(question_id: str = "q1") -> dict:
    steps = (
        "Use the original target and all stated constraints.",
        "The correct first relation uses 12 and 5.",
        "The correct substitution gives the requested value.",
        "The original conditions verify that the value is 7.",
        "# Answer\n\n7",
    )
    return construct_five_tuple(
        question_id=question_id,
        problem="Using 12 and 5, determine the requested value.",
        answer="7",
        clean_steps=steps,
        tokenizer=_Tokenizer(),
        min_token_ratio=0.6,
        max_token_ratio=2.5,
        min_edit_distance=0.4,
        source={"kind": "fixture"},
    )


def test_frozen_nine_arm_matrix() -> None:
    assert len(ARMS) == 9
    assert ARMS[0] == "clean"
    assert set(ARMS) == {
        "clean",
        "solution_n1_equal",
        "solution_n1_w01",
        "solution_n2_equal",
        "solution_n2_w01",
        "misread_n1_equal",
        "misread_n1_w01",
        "misread_n2_equal",
        "misread_n2_w01",
    }


def test_clean_and_equal_arms_use_unit_weights() -> None:
    row = _tuple()
    clean_steps, clean_weights, clean_labels = steps_and_weights("clean", row, 0.1)
    noisy_steps, noisy_weights, noisy_labels = steps_and_weights(
        "solution_n1_equal", row, 0.1
    )
    assert clean_steps == tuple(row["variants"]["clean"]["steps"])
    assert noisy_steps == tuple(row["variants"]["solution_n1"]["steps"])
    assert clean_weights == noisy_weights == (1.0,) * 5
    assert clean_labels == (1, 1, 1, 1, 1)
    assert noisy_labels == (1, -1, 1, 1, 1)


def test_weighted_arm_normalizes_mean_and_preserves_point_one_ratio() -> None:
    row = _tuple()
    _, weights, labels = steps_and_weights("solution_n1_w01", row, 0.1)
    error_weight = weights[labels.index(-1)]
    correct_weight = weights[0]
    assert sum(weights) / 5 == pytest.approx(1.0)
    assert error_weight / correct_weight == pytest.approx(0.1)
    assert correct_weight == pytest.approx(5 / 4.1)


def test_two_error_weight_vector_has_two_low_positions() -> None:
    row = _tuple()
    _, weights, labels = steps_and_weights("misread_n2_w01", row, 0.1)
    assert labels == (1, -1, -1, 1, 1)
    assert weights[1] == weights[2]
    assert weights[1] / weights[0] == pytest.approx(0.1)
    assert sum(weights) / 5 == pytest.approx(1.0)


def test_schedule_is_deterministic_and_hash_guarded() -> None:
    entries = [_tuple(f"q{index}") for index in range(8)]
    manifest = {
        "schema_version": 1,
        "selection_domain": "fixture",
        "entries": entries,
        "test_problem_ids": ["t"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    first = create_training_schedule(
        manifest, seeds=[1, 2], updates=1, accumulation=4
    )
    second = create_training_schedule(
        manifest, seeds=[1, 2], updates=1, accumulation=4
    )
    assert first == second
    verify_training_schedule(first, expected_seeds=[1, 2], expected_micro_batches=4)
    tampered = copy.deepcopy(first)
    tampered["per_seed"]["1"][0], tampered["per_seed"]["1"][1] = (
        tampered["per_seed"]["1"][1],
        tampered["per_seed"]["1"][0],
    )
    with pytest.raises(ValueError, match="SHA-256"):
        verify_training_schedule(
            tampered, expected_seeds=[1, 2], expected_micro_batches=4
        )


def test_human_audit_gate_requires_all_fixed_examples_to_pass() -> None:
    audit = {
        "status": "PASS",
        "manifest_sha256": "abc",
        "reviewed_question_ids": ["q1", "q2"],
        "per_question_decisions": {"q1": "PASS", "q2": "PASS"},
    }
    validate_human_audit(audit, manifest_sha256="abc", expected_count=2)
    audit["per_question_decisions"]["q2"] = "FAIL"
    with pytest.raises(ValueError, match="did not pass"):
        validate_human_audit(audit, manifest_sha256="abc", expected_count=2)


def test_training_directory_keeps_gate_outputs_separate(tmp_path) -> None:
    config = {"output_root": "outputs/experiment"}
    train = training_directory(tmp_path, config, 1, "clean", sanity=False)
    gate = training_directory(
        tmp_path,
        config,
        1,
        "clean",
        sanity=True,
        phase_override="max_length_memory_gate",
    )
    assert train != gate
    assert train.parts[-3:] == ("train", "seed_1", "clean")
    assert gate.parts[-3:] == ("max_length_memory_gate", "seed_1", "clean")
    with pytest.raises(ValueError, match="Unsupported"):
        training_directory(
            tmp_path,
            config,
            1,
            "clean",
            sanity=True,
            phase_override="../escape",
        )
