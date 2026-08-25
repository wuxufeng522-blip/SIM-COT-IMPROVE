from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json

import pytest

from reliable_simcot.prm800k_data import (
    NaturalTrajectory,
    classify_ratings,
    collapse_triplets_by_problem,
    consolidate_trajectories,
    reconstruct_trajectory,
    select_strict_triplets,
    verify_frozen_triplet_manifest,
)
from reliable_simcot import prm800k_data


def _step(text: str, rating: int, *, flagged=None) -> dict:
    return {
        "completions": [{"text": text, "rating": rating, "flagged": flagged}],
        "chosen_completion": 0,
        "human_completion": None,
    }


def _record(ratings: tuple[int, ...], *, answer: str = "4", generation: int = 3) -> dict:
    return {
        "labeler": "fixture-labeler",
        "generation": generation,
        "is_quality_control_question": False,
        "is_initial_screening_question": False,
        "question": {
            "problem": "What is 2+2?",
            "ground_truth_answer": "4",
            "pre_generated_answer": answer,
        },
        "label": {
            "steps": [
                _step(f"step {idx}: {rating}", rating)
                for idx, rating in enumerate(ratings)
            ]
        },
    }


def _trajectory(ratings: tuple[int, ...], suffix: str, length: int) -> NaturalTrajectory:
    steps = tuple(f"{suffix}-{idx} " + "x " * length for idx in range(5))
    return NaturalTrajectory(
        problem_id="a" * 64,
        problem="problem",
        ground_truth_answer="4",
        generated_answer="4",
        steps=steps,
        ratings=ratings,
        generation_key="phase2:3",
        source_phase="phase2",
        source_file="phase2_train.jsonl",
        source_line=0,
        record_sha256=(suffix * 64)[:64],
        trajectory_sha256=(suffix.upper() * 64)[:64],
    )


def test_classify_ratings_is_strict() -> None:
    assert classify_ratings((1, 1, 1, 1, 1)) == "clean"
    assert classify_ratings((-1, 1, 1, 1, 1)) == "noise1"
    assert classify_ratings((-1, 1, -1, 1, 1)) == "noise2"
    assert classify_ratings((0, 1, 1, 1, 1)) is None
    assert classify_ratings((-1, -1, -1, 1, 1)) is None
    assert classify_ratings((1, 1, 1, 1)) is None


def test_raw_original_rating_pattern_requires_original_chosen_path() -> None:
    row = _record((-1, 1, 1, 1, 1))
    assert prm800k_data._raw_original_rating_pattern(row) == "noise1"
    row["label"]["steps"][0]["chosen_completion"] = None
    assert prm800k_data._raw_original_rating_pattern(row) is None


def test_reconstructs_only_original_five_step_correct_answer() -> None:
    trajectory, rejection = reconstruct_trajectory(
        _record((1, 1, -1, 1, 1)),
        source_phase="phase2",
        source_file="phase2_train.jsonl",
        source_line=7,
        grade_answer=lambda given, expected: given == expected,
    )
    assert rejection is None
    assert trajectory is not None
    assert trajectory.ratings == (1, 1, -1, 1, 1)
    assert trajectory.generation_key == "phase2:3"
    assert trajectory.source_line == 7


def test_reconstructs_phase1_answer_from_original_last_step() -> None:
    row = _record((1, 1, 1, 1, 1), generation=None)
    row["question"].pop("pre_generated_answer")
    row["label"]["finish_reason"] = "solution"
    row["label"]["steps"][-1]["completions"][0]["text"] = "Done.\n\n# Answer\n\n4"
    trajectory, rejection = reconstruct_trajectory(
        row,
        source_phase="phase1",
        source_file="phase1_train.jsonl",
        source_line=3,
        grade_answer=lambda given, expected: given == expected,
    )
    assert rejection is None
    assert trajectory is not None
    assert trajectory.generated_answer == "4"
    assert trajectory.generation_key == "phase1:null"


@pytest.mark.parametrize(
    ("mutator", "expected"),
    [
        (lambda row: row["question"].update(pre_generated_answer="5"), "wrong_answer"),
        (lambda row: row["label"]["steps"].pop(), "not_exactly_five_steps"),
        (lambda row: row["label"]["steps"][0].update(chosen_completion=None), "human_completion"),
        (lambda row: row["label"]["steps"][0]["completions"][0].update(flagged=True), "flagged"),
        (lambda row: row.update(is_quality_control_question=True), "quality_control"),
        (lambda row: row["label"]["steps"][0]["completions"][0].update(rating=0), "rating_zero"),
    ],
)
def test_reconstruction_rejection_codes(mutator, expected: str) -> None:
    row = _record((1, 1, 1, 1, 1))
    mutator(row)
    trajectory, rejection = reconstruct_trajectory(
        row,
        source_phase="phase2",
        source_file="phase2_train.jsonl",
        source_line=0,
        grade_answer=lambda given, expected_answer: given == expected_answer,
    )
    assert trajectory is None
    assert rejection == expected


def test_conflicting_duplicate_labels_are_excluded() -> None:
    clean = _trajectory((1, 1, 1, 1, 1), "a", 1)
    conflict = NaturalTrajectory(
        **{
            **clean.__dict__,
            "ratings": (-1, 1, 1, 1, 1),
            "record_sha256": "b" * 64,
        }
    )
    kept, counts = consolidate_trajectories([clean, conflict])
    assert kept == []
    assert counts["label_conflict"] == 2


def test_triplet_selection_minimizes_length_spread_and_is_order_invariant() -> None:
    rows = [
        _trajectory((1, 1, 1, 1, 1), "a", 1),
        _trajectory((1, 1, 1, 1, 1), "b", 8),
        _trajectory((-1, 1, 1, 1, 1), "c", 7),
        _trajectory((-1, -1, 1, 1, 1), "d", 6),
    ]
    length_fn = lambda trajectory: sum(len(step.split()) for step in trajectory.steps)
    forward = select_strict_triplets(rows, token_length=length_fn)
    reverse = select_strict_triplets(list(reversed(rows)), token_length=length_fn)
    assert forward == reverse
    assert len(forward) == 1
    assert forward[0]["clean"]["trajectory_sha256"] == rows[1].trajectory_sha256
    assert forward[0]["token_length_spread"] <= 10


def test_multiple_generations_collapse_to_one_problem() -> None:
    rows = [
        _trajectory((1, 1, 1, 1, 1), "a", 3),
        _trajectory((-1, 1, 1, 1, 1), "b", 3),
        _trajectory((-1, -1, 1, 1, 1), "c", 3),
    ]
    first = select_strict_triplets(rows, token_length=lambda _: 20)[0]
    second = deepcopy(first)
    second["generation_key"] = "phase2:4"
    second["token_length_spread"] = 2
    collapsed = collapse_triplets_by_problem([first, second])
    assert len(collapsed) == 1
    assert collapsed[0]["generation_key"] == "phase2:3"


def test_frozen_manifest_detects_mutation(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "train_triplets": [{"problem_id": "a" * 64}],
        "dev_entries": [{"problem_id": "b" * 64}],
        "confirm_problem_ids": ["c" * 64],
    }
    from reliable_simcot.prm800k_data import canonical_hash

    payload["manifest_sha256"] = canonical_hash(payload)
    verify_frozen_triplet_manifest(payload, expected_train=1, expected_dev=1, expected_confirm=1)
    mutated = json.loads(json.dumps(payload))
    mutated["train_triplets"][0]["problem_id"] = "d" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify_frozen_triplet_manifest(
            mutated, expected_train=1, expected_dev=1, expected_confirm=1
        )
