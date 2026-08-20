from __future__ import annotations

from reliable_simcot.full_conflict_validation import (
    normalized_levenshtein,
    validate_full_conflict_candidate,
)
from reliable_simcot.official_adapter import OfficialExample


class _Tokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [ord(char) for char in text if not char.isspace()]


CONFIG = {
    "min_normalized_edit_distance": 0.4,
    "min_aux_token_ratio": 0.9,
    "max_aux_token_ratio": 1.1,
    "latent_stage": 5,
    "c_thought": 2,
    "max_sequence_tokens": 1024,
}


def _clean() -> OfficialExample:
    return OfficialExample(
        idx=1,
        question="Use 3 groups of 8, remove 5, double by 2, add 4, then report the result.",
        steps=(
            "<<3*8=24>>",
            "<<24-5=19>>",
            "<<19*2=38>>",
            "<<38+4=42>>",
            "<<42-2=40>>",
        ),
        answer="40",
    )


def _candidate() -> dict:
    return {
        "error_rationale": "Treat every stated relation as part of one cumulative score.",
        "steps": [
            "<<3+8=11>>",
            "<<11*5=55>>",
            "<<55-2=53>>",
            "<<53+4=57>>",
            "<<57*3=171>>",
        ],
        "wrong_final_result": "171",
    }


def test_normalized_levenshtein_endpoints() -> None:
    assert normalized_levenshtein([1, 2], [1, 2]) == 0.0
    assert normalized_levenshtein([], [1, 2]) == 1.0


def test_valid_full_chain_passes_core_checks() -> None:
    result = validate_full_conflict_candidate(
        _clean(),
        _candidate(),
        tokenizer=_Tokenizer(),
        token_ids=None,
        config=CONFIG,
        check_context=False,
    )
    assert result.accepted, result.rejection_codes
    assert result.final_ancestors == (0, 1, 2, 3)


def test_disconnected_or_unmotivated_chain_is_rejected() -> None:
    candidate = _candidate()
    candidate["steps"][3] = "<<999+4=1003>>"
    candidate["steps"][4] = "<<1003*3=3009>>"
    candidate["wrong_final_result"] = "3009"
    result = validate_full_conflict_candidate(
        _clean(),
        candidate,
        tokenizer=_Tokenizer(),
        token_ids=None,
        config=CONFIG,
        check_context=False,
    )
    assert any(code.startswith("unmotivated_constant") for code in result.rejection_codes)
    assert "disconnected_chain" in result.rejection_codes


def test_official_answer_collision_is_rejected() -> None:
    candidate = _candidate()
    candidate["steps"][-1] = "<<53-5-8=40>>"
    candidate["wrong_final_result"] = "40"
    result = validate_full_conflict_candidate(
        _clean(),
        candidate,
        tokenizer=_Tokenizer(),
        token_ids=None,
        config=CONFIG,
        check_context=False,
    )
    assert "official_answer_collision" in result.rejection_codes
