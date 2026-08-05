from __future__ import annotations

from pathlib import Path
import csv

from reliable_simcot.audit import write_blinded_review_package, write_frozen_audit
from reliable_simcot.review_metrics import (
    cohen_kappa,
    compile_review_results,
    wilson_interval,
)


def test_cohen_kappa_perfect_agreement() -> None:
    result = cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0])
    assert result["raw_agreement"] == 1.0
    assert result["cohen_kappa"] == 1.0


def test_wilson_interval_contains_observed_proportion() -> None:
    result = wilson_interval(2, 10)
    assert result["ci95_low"] < 0.2 < result["ci95_high"]


def _fill_template(template: Path, destination: Path) -> None:
    with template.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fieldnames = list(rows[0])
    for row in rows:
        row["y_valid"] = "1"
        row["y_utility"] = "1"
        row["confidence"] = "high"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_compile_review_results_uses_primary_rows_once(tmp_path: Path) -> None:
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
    package_dir = tmp_path / "r022"
    write_blinded_review_package(
        parent_manifest_path=audit_dir / "freeze_manifest.json",
        output_dir=package_dir,
        expected_parent_rows_sha256=parent["audit_rows_sha256"],
        assignment_seed=9,
        overlap_fraction=0.25,
    )
    labeled_a = tmp_path / "a_labeled.csv"
    labeled_b = tmp_path / "b_labeled.csv"
    _fill_template(package_dir / "reviewer_a.csv", labeled_a)
    _fill_template(package_dir / "reviewer_b.csv", labeled_b)
    result = compile_review_results(
        package_manifest_path=package_dir / "review_package_manifest.json",
        reviewer_a_labeled_path=labeled_a,
        reviewer_b_labeled_path=labeled_b,
        output_dir=tmp_path / "compiled",
    )
    assert result["status"] == "PASS"
    assert result["invalid_step_prevalence"]["total"] == 4
    assert result["invalid_step_prevalence"]["count"] == 0
