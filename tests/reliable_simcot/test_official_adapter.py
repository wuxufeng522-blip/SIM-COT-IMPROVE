from __future__ import annotations

import pytest

from reliable_simcot.official_adapter import (
    extract_answer_official,
    parse_icot_line,
    summarize_predictions,
    validate_checkpoint_compatibility,
)


def test_parse_icot_line_preserves_dataset_ground_truth() -> None:
    line = "What is 2+3?||<<2+3=5>> <<5*2=10>>##10\n"
    example = parse_icot_line(line, idx=7)
    assert example.idx == 7
    assert example.question == "What is 2+3?"
    assert example.steps == ("<<2+3=5>>", "<<5*2=10>>")
    assert example.answer == "10"


def test_parse_icot_line_matches_official_four_hash_preprocessing() -> None:
    example = parse_icot_line("q||<<1+2=3>> #### 3\n", idx=0)
    assert example.steps == ("<<1+2=3>>",)
    assert example.answer == "3"


def test_parse_icot_line_preserves_official_empty_step_quirk() -> None:
    example = parse_icot_line("q|| #### 3\n", idx=0)
    assert example.steps == ("",)


def test_parse_icot_line_rejects_missing_delimiters() -> None:
    with pytest.raises(ValueError, match="Malformed"):
        parse_icot_line("question without labels", idx=0)


def test_extract_answer_matches_official_rule() -> None:
    assert extract_answer_official("Question\n### 1,234") == "1234"
    assert extract_answer_official("prefix # # -7 ") == "-7"


def test_summarize_predictions_uses_dataset_comparison_flag() -> None:
    rows = [
        {"ground_truth": "1", "prediction": "1", "correct": True},
        {"ground_truth": "2", "prediction": "3", "correct": False},
    ]
    summary = summarize_predictions(rows)
    assert summary == {"examples": 2, "correct": 1, "accuracy": 0.5}


def test_partial_checkpoint_accepts_only_missing_auxiliary_weights() -> None:
    validate_checkpoint_compatibility(
        ["expainable_llm.transformer.wte.weight"],
        [],
        allow_missing_auxiliary=True,
    )
    with pytest.raises(ValueError, match="Partial checkpoint key mismatch"):
        validate_checkpoint_compatibility(
            ["base_causallm.transformer.wte.weight"],
            [],
            allow_missing_auxiliary=True,
        )
    with pytest.raises(ValueError, match="Expected a base-only checkpoint"):
        validate_checkpoint_compatibility([], [], allow_missing_auxiliary=True)
