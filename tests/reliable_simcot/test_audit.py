from __future__ import annotations

from pathlib import Path
import json

import pytest

from reliable_simcot.audit import (
    assert_audit_ids_excluded,
    audit_step,
    evaluate_arithmetic,
    select_question_clusters,
    validate_human_labels,
    write_auto_triage_revision,
    write_blinded_review_package,
    write_frozen_audit,
)
from reliable_simcot.official_adapter import OfficialExample


def _example(idx: int, question: str, steps: tuple[str, ...]) -> OfficialExample:
    return OfficialExample(idx=idx, question=question, steps=steps, answer="3")


def test_arithmetic_triage_is_exact_and_conservative() -> None:
    assert evaluate_arithmetic("1 / 3 + 2 / 3") == 1
    good = audit_step(
        "<<1+2=3>>", step_index=0, all_steps=("<<1+2=3>>",), answer="3"
    )
    assert good.arithmetic_status == "checked_match"
    assert good.final_answer_status == "checked_match"
    rounded = audit_step(
        "<<1/3=0.33>>", step_index=0, all_steps=("<<1/3=0.33>>",), answer="0.33"
    )
    assert rounded.arithmetic_status == "checked_match"
    wrong = audit_step(
        "<<1.5*5+3*2=10.5>>",
        step_index=0,
        all_steps=("<<1.5*5+3*2=10.5>>",),
        answer="10.5",
    )
    assert wrong.arithmetic_status == "checked_mismatch"
    ambiguous = audit_step(
        "<<three plus two=5>>",
        step_index=0,
        all_steps=("<<three plus two=5>>",),
        answer="5",
    )
    assert ambiguous.arithmetic_status == "unparsed"
    assert "arithmetic_mismatch_candidate" not in ambiguous.candidate_flags


def test_question_cluster_sampling_is_deterministic_and_keeps_trajectories() -> None:
    examples = [
        _example(0, "q0", ("<<1+2=3>>",)),
        _example(1, "q1", ("<<1+1=2>>", "<<2+1=3>>")),
        _example(2, "q1", ("<<1+1=2>>", "<<2+1=3>>")),
    ]
    first, multiplicity = select_question_clusters(examples, seed=7, question_count=2)
    second, _ = select_question_clusters(examples, seed=7, question_count=2)
    assert [item.question for item in first] == [item.question for item in second]
    assert sorted(multiplicity.values()) == [1, 2]


def test_frozen_audit_has_blank_labels_and_refuses_overwrite(tmp_path: Path) -> None:
    dataset = tmp_path / "train.txt"
    dataset.write_text(
        "q0||<<1+2=3>> #### 3\nq1||<<1+1=2>> <<2+1=3>> #### 3\n",
        encoding="utf-8",
    )
    output = tmp_path / "audit"
    manifest = write_frozen_audit(
        dataset_path=dataset,
        output_dir=output,
        seed=3,
        question_count=2,
        min_steps=3,
    )
    rows = [json.loads(line) for line in (output / "audit_rows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(row["y_valid"] is None and row["y_utility"] is None for row in rows)
    assert manifest["step_count"] == 3
    with pytest.raises(FileExistsError):
        write_frozen_audit(
            dataset_path=dataset,
            output_dir=output,
            seed=3,
            question_count=2,
            min_steps=3,
        )


def test_leakage_guard_rejects_audit_question_ids() -> None:
    manifest = {"selected_question_ids": ["held-out"]}
    assert_audit_ids_excluded(["train"], manifest)
    with pytest.raises(ValueError, match="Audit leakage"):
        assert_audit_ids_excluded(["train", "held-out"], manifest)


def test_utility_is_defined_only_for_valid_steps() -> None:
    validate_human_labels(
        [
            {"audit_row_id": "a", "y_valid": 0, "y_utility": None},
            {"audit_row_id": "b", "y_valid": 1, "y_utility": 0},
        ]
    )
    with pytest.raises(ValueError, match="Utility must be null"):
        validate_human_labels(
            [{"audit_row_id": "a", "y_valid": 0, "y_utility": 0}]
        )


def test_triage_revision_preserves_parent_and_refines_rounding(tmp_path: Path) -> None:
    dataset = tmp_path / "train.txt"
    dataset.write_text("q||<<1/3=0.33>> #### 0.33\n", encoding="utf-8")
    audit_dir = tmp_path / "r020"
    parent = write_frozen_audit(
        dataset_path=dataset,
        output_dir=audit_dir,
        seed=1,
        question_count=1,
        min_steps=1,
    )
    triage = write_auto_triage_revision(
        parent_manifest_path=audit_dir / "freeze_manifest.json",
        output_dir=tmp_path / "r021",
        expected_parent_rows_sha256=parent["audit_rows_sha256"],
    )
    assert triage["parent_rows_sha256"] == parent["audit_rows_sha256"]
    assert triage["arithmetic_status_counts"] == {"checked_match": 1}


def test_blinded_review_package_has_exact_overlap_and_no_flags(tmp_path: Path) -> None:
    dataset = tmp_path / "train.txt"
    dataset.write_text(
        "q0||<<1+1=2>> <<2+1=3>> #### 3\n"
        "q1||<<1+2=3>> #### 3\n"
        "q2||<<2+1=3>> #### 3\n",
        encoding="utf-8",
    )
    audit_dir = tmp_path / "r020"
    parent = write_frozen_audit(
        dataset_path=dataset,
        output_dir=audit_dir,
        seed=1,
        question_count=3,
        min_steps=4,
    )
    package = write_blinded_review_package(
        parent_manifest_path=audit_dir / "freeze_manifest.json",
        output_dir=tmp_path / "r022",
        expected_parent_rows_sha256=parent["audit_rows_sha256"],
        assignment_seed=9,
        overlap_fraction=0.25,
    )
    assert package["overlap_rows"] == 1
    header = (tmp_path / "r022" / "reviewer_a.csv").read_text(
        encoding="utf-8-sig"
    ).splitlines()[0]
    assert "candidate_flags" not in header
    assert "arithmetic_status" not in header
    assert "assignment_role" in header
