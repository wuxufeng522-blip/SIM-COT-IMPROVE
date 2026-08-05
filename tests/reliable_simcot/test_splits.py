from __future__ import annotations

import pytest

from reliable_simcot.splits import (
    assign_question_splits,
    split_manifest_sha256,
    validate_question_isolation,
)


def test_question_split_is_exact_deterministic_and_isolated() -> None:
    question_ids = [f"q-{index}" for index in range(10)]
    first = assign_question_splits(question_ids, seed=17)
    second = assign_question_splits(reversed(question_ids), seed=17)
    assert first == second
    assert list(first.values()).count("head_train") == 6
    assert list(first.values()).count("head_validation") == 2
    assert list(first.values()).count("head_audit") == 2
    assert split_manifest_sha256(first) == split_manifest_sha256(second)

    rows = [
        {"question_id": question_id, "split": split_name, "variant": variant}
        for question_id, split_name in first.items()
        for variant in range(3)
    ]
    assert validate_question_isolation(rows) == {
        "head_train": 6,
        "head_validation": 2,
        "head_audit": 2,
    }


def test_question_split_rejects_cross_split_variants() -> None:
    with pytest.raises(ValueError, match="cross splits"):
        validate_question_isolation(
            [
                {"question_id": "q", "split": "head_train"},
                {"question_id": "q", "split": "head_audit"},
            ]
        )
