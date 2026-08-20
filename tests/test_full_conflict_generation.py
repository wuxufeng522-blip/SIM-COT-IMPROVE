from __future__ import annotations

import json

import pytest

from reliable_simcot.full_conflict_generation import append_raw_generation, read_jsonl


def test_raw_attempts_are_append_only(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    row = {"question_id": "q", "attempt": 1, "candidate": {}}
    append_raw_generation(path, row, max_attempts=3)
    assert read_jsonl(path) == [row]
    with pytest.raises(FileExistsError):
        append_raw_generation(path, row, max_attempts=3)


def test_attempt_limit_is_enforced(tmp_path) -> None:
    with pytest.raises(ValueError, match="range"):
        append_raw_generation(
            tmp_path / "raw.jsonl",
            {"question_id": "q", "attempt": 4, "candidate": {}},
            max_attempts=3,
        )


def test_malformed_jsonl_is_rejected(tmp_path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"ok": True}) + "\n{" + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed"):
        read_jsonl(path)
