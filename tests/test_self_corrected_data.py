from __future__ import annotations

import copy

import pytest

from reliable_simcot.self_corrected_data import (
    VARIANT_LABELS,
    _clean_steps_from_gsm8k,
    _clean_steps_from_solution,
    canonical_hash,
    construct_five_tuple,
    select_stratified_audit_entries,
    validate_five_tuple,
    verify_frozen_manifest,
)
from reliable_simcot.official_adapter import OfficialExample


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(char) for char in text if not char.isspace()]


def _clean_steps() -> tuple[str, ...]:
    return (
        "We keep the original target and use every stated constraint in the problem.",
        "The first correct relation determines an intermediate quantity from 12 and 5.",
        "Substituting that quantity into the second correct relation gives the needed value.",
        "Checking the original conditions confirms that the requested value is 7.",
        "# Answer\n\n7",
    )


def _entry() -> dict:
    return construct_five_tuple(
        question_id="q1",
        problem="Using 12 and 5 under the stated relation, find the requested value.",
        answer="7",
        clean_steps=_clean_steps(),
        tokenizer=_Tokenizer(),
        min_token_ratio=0.75,
        max_token_ratio=2.5,
        min_edit_distance=0.4,
        source={"kind": "fixture"},
    )


def test_constructs_all_five_paired_variants() -> None:
    row = _entry()
    assert set(row["variants"]) == set(VARIANT_LABELS)
    for name, expected_labels in VARIANT_LABELS.items():
        variant = row["variants"][name]
        assert tuple(variant["labels"]) == expected_labels
        assert len(variant["steps"]) == 5
        assert variant["steps"][-1] == _clean_steps()[-1]
        assert variant["final_answer"] == "7"


def test_noise_two_has_contiguous_causal_errors_and_explicit_recovery() -> None:
    row = _entry()
    for name in ("solution_n2", "misread_n2"):
        variant = row["variants"][name]
        assert variant["labels"] == [1, -1, -1, 1, 1]
        assert variant["wrong_branch_id"] in variant["steps"][1]
        assert variant["wrong_branch_id"] in variant["steps"][2]
        assert "Discarding that branch" in variant["steps"][3]
        assert "12" in variant["steps"][3]
        assert "7" in variant["steps"][3]


def test_recovery_prefers_ordered_prose_over_formula_dump() -> None:
    row = _entry()
    for name in ("solution_n1", "solution_n2", "misread_n1", "misread_n2"):
        variant = row["variants"][name]
        assert variant["recovery_mode"].startswith("explicit_")
        recovery_position = 2 if variant["dose"] == 1 else 3
        assert "Using the original conditions," not in variant["steps"][recovery_position]


def test_noise_two_recovery_contains_complete_clean_derivation() -> None:
    row = _entry()
    clean_derivation = row["variants"]["clean"]["steps"][3]
    for name in ("solution_n2", "misread_n2"):
        assert clean_derivation in row["variants"][name]["steps"][3]


def test_segmented_official_solution_preserves_sentence_order() -> None:
    steps = _clean_steps_from_solution(
        "First relation. Second relation. Third relation. Fourth conclusion.", "4"
    )
    joined = " ".join(steps[1:4])
    assert joined.index("First") < joined.index("Second")
    assert joined.index("Second") < joined.index("Third")
    assert joined.index("Third") < joined.index("Fourth")


def test_segmented_solution_does_not_split_factorial_tex() -> None:
    steps = _clean_steps_from_solution(
        r"Count $11!$. Divide by $4! \times 4! \times 2!$ to finish.", "1"
    )
    joined = " ".join(steps[1:4])
    assert r"$4! \times 4! \times 2!$" in joined


def test_gsm8k_adapter_preserves_all_five_released_calculations() -> None:
    example = OfficialExample(
        idx=3,
        question="A five-step arithmetic word problem.",
        steps=("<<1=1>>", "<<2=2>>", "<<3=3>>", "<<4=4>>", "<<5=5>>"),
        answer="5",
    )
    adapted = _clean_steps_from_gsm8k(example)
    joined = " ".join(adapted[:-1])
    assert len(adapted) == 5
    assert all(step in joined for step in example.steps)
    assert adapted[-1] == "# Answer\n\n5"


def test_validator_rejects_endpoint_patch_without_correct_answer() -> None:
    row = _entry()
    broken = copy.deepcopy(row)
    broken["variants"]["solution_n1"]["steps"][-1] = "# Answer\n\n99"
    result = validate_five_tuple(
        broken,
        tokenizer=_Tokenizer(),
        min_token_ratio=0.75,
        max_token_ratio=2.5,
        min_edit_distance=0.4,
    )
    assert not result["accepted"]
    assert "solution_n1:final_answer_mismatch" in result["rejection_codes"]


def test_validator_rejects_wrong_error_count() -> None:
    row = _entry()
    broken = copy.deepcopy(row)
    broken["variants"]["misread_n2"]["labels"] = [1, -1, 1, 1, 1]
    result = validate_five_tuple(
        broken,
        tokenizer=_Tokenizer(),
        min_token_ratio=0.75,
        max_token_ratio=2.5,
        min_edit_distance=0.4,
    )
    assert "misread_n2:label_pattern" in result["rejection_codes"]


def test_manifest_hash_and_counts_are_verified() -> None:
    row = _entry()
    manifest = {
        "schema_version": 1,
        "selection_domain": "fixture",
        "entries": [row],
        "test_problem_ids": ["t1", "t2"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(manifest, expected_train=1, expected_test=2)
    tampered = copy.deepcopy(manifest)
    tampered["entries"][0]["answer"] = "8"
    with pytest.raises(ValueError, match="SHA-256"):
        verify_frozen_manifest(tampered, expected_train=1, expected_test=2)


def test_canonical_hash_ignores_its_own_manifest_field() -> None:
    payload = {"x": 1}
    digest = canonical_hash(payload)
    assert canonical_hash({"x": 1, "manifest_sha256": digest}) == digest


def test_audit_selection_is_deterministic_and_source_stratified() -> None:
    entries = [
        {"question_id": f"a{index}", "source": {"kind": "a"}}
        for index in range(12)
    ] + [
        {"question_id": f"b{index}", "source": {"kind": "b"}}
        for index in range(12)
    ]
    first = select_stratified_audit_entries(
        entries, count=10, selection_domain="fixture"
    )
    second = select_stratified_audit_entries(
        list(reversed(entries)), count=10, selection_domain="fixture"
    )
    assert [row["question_id"] for row in first] == [
        row["question_id"] for row in second
    ]
    assert sum(row["source"]["kind"] == "a" for row in first) == 5
    assert sum(row["source"]["kind"] == "b" for row in first) == 5
