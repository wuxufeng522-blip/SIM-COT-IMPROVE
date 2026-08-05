from __future__ import annotations

import json
import hashlib

from reliable_simcot.ood_adapter import (
    extract_answer_number_official,
    load_ood_examples,
    normalize_ground_truth,
    normalize_question_official,
)
from scripts.evaluate_official_simcot_ood import validate_dataset_config


def test_extract_answer_number_matches_released_codi_rule() -> None:
    assert extract_answer_number_official("work 12,000 then ### -3.5") == -3.5
    assert extract_answer_number_official("no numerical answer") == float("inf")


def test_official_normalization() -> None:
    assert normalize_question_official("  a  b  ") == "a b"
    assert normalize_ground_truth("reason #### 1,234") == 1234.0


def test_load_multi_arith_test_split(tmp_path) -> None:
    path = tmp_path / "test.json"
    path.write_text(
        json.dumps([{"question": " 2 plus  3? ", "final_ans": "5"}]),
        encoding="utf-8",
    )
    examples = load_ood_examples("multi-arith", [path])
    assert len(examples) == 1
    assert examples[0].question == "2 plus 3?"
    assert examples[0].answer == 5.0


def test_load_multi_arith_can_combine_public_splits(tmp_path) -> None:
    test = tmp_path / "test.json"
    train = tmp_path / "train.json"
    test.write_text(
        json.dumps([{"question": "test?", "final_ans": "1"}]), encoding="utf-8"
    )
    train.write_text(
        json.dumps([{"question": "train?", "final_ans": "2"}]), encoding="utf-8"
    )
    examples = load_ood_examples("multi-arith", [test, train])
    assert [example.question for example in examples] == ["test?", "train?"]
    assert [example.answer for example in examples] == [1.0, 2.0]


def test_svamp_combines_train_then_test(tmp_path) -> None:
    train = tmp_path / "train.json"
    test = tmp_path / "test.json"
    train.write_text(
        json.dumps([{"Body": "Train body.", "Question": "Question?", "Answer": 1}]),
        encoding="utf-8",
    )
    test.write_text(
        json.dumps([{"Body": "Test body.", "Question": "Question?", "Answer": 2}]),
        encoding="utf-8",
    )
    examples = load_ood_examples("svamp", [train, test])
    assert [example.question for example in examples] == [
        "Train body. Question?",
        "Test body. Question?",
    ]
    assert [example.answer for example in examples] == [1.0, 2.0]


def test_validate_dataset_config_rejects_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "data.json"
    path.write_bytes(b"data")
    config = {"source_sha256": [hashlib.sha256(b"data").hexdigest()]}
    validate_dataset_config(config, [path])

    config["source_sha256"] = ["0" * 64]
    import pytest

    with pytest.raises(ValueError, match="SHA-256"):
        validate_dataset_config(config, [path])
