from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math

from .audit import sha256_file


EDITABLE_COLUMNS = {
    "y_valid",
    "y_utility",
    "confidence",
    "review_notes",
}


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_and_validate_labels(
    template_path: str | Path,
    labeled_path: str | Path,
) -> list[dict[str, Any]]:
    template = _read_csv(template_path)
    labeled = _read_csv(labeled_path)
    if len(template) != len(labeled):
        raise ValueError("Labeled review has a different row count from its template")
    if not template or set(template[0]) != set(labeled[0]):
        raise ValueError("Labeled review columns differ from its template")
    immutable = set(template[0]) - EDITABLE_COLUMNS
    parsed: list[dict[str, Any]] = []
    for expected, actual in zip(template, labeled, strict=True):
        if any(actual.get(column) != expected.get(column) for column in immutable):
            raise ValueError(
                f"Immutable review field changed at {expected.get('audit_row_id')}"
            )
        valid_text = actual.get("y_valid", "").strip()
        utility_text = actual.get("y_utility", "").strip()
        confidence = actual.get("confidence", "").strip().lower()
        if valid_text not in {"0", "1"}:
            raise ValueError(f"Invalid Validity label at {actual['audit_row_id']}")
        valid = int(valid_text)
        if valid == 0 and utility_text:
            raise ValueError("Utility must be blank when Validity is 0")
        if valid == 1 and utility_text not in {"0", "1"}:
            raise ValueError("Utility must be binary when Validity is 1")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError(f"Invalid confidence at {actual['audit_row_id']}")
        parsed.append(
            {
                **actual,
                "y_valid": valid,
                "y_utility": int(utility_text) if utility_text else None,
                "confidence": confidence,
            }
        )
    if len({row["audit_row_id"] for row in parsed}) != len(parsed):
        raise ValueError("Duplicate audit row IDs in labeled review")
    return parsed


def cohen_kappa(left: list[int], right: list[int]) -> dict[str, float | int]:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa requires equal non-empty label sequences")
    count = len(left)
    agreement = sum(a == b for a, b in zip(left, right, strict=True)) / count
    p_left = sum(left) / count
    p_right = sum(right) / count
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    kappa = 1.0 if expected == 1.0 and agreement == 1.0 else (agreement - expected) / (1 - expected)
    return {"n": count, "raw_agreement": agreement, "cohen_kappa": kappa}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | int]:
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Invalid binomial counts")
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return {
        "count": successes,
        "total": total,
        "proportion": proportion,
        "ci95_low": center - half,
        "ci95_high": center + half,
    }


def compile_review_results(
    *,
    package_manifest_path: str | Path,
    reviewer_a_labeled_path: str | Path,
    reviewer_b_labeled_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(package_manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    template_a = Path(manifest["reviewer_a_path"])
    template_b = Path(manifest["reviewer_b_path"])
    if sha256_file(template_a) != manifest["reviewer_a_sha256"]:
        raise ValueError("Reviewer A template SHA-256 mismatch")
    if sha256_file(template_b) != manifest["reviewer_b_sha256"]:
        raise ValueError("Reviewer B template SHA-256 mismatch")
    rows_a = read_and_validate_labels(template_a, reviewer_a_labeled_path)
    rows_b = read_and_validate_labels(template_b, reviewer_b_labeled_path)
    by_a = {row["audit_row_id"]: row for row in rows_a}
    by_b = {row["audit_row_id"]: row for row in rows_b}
    overlap_ids = set(
        Path(manifest["overlap_ids_path"])
        .read_text(encoding="ascii")
        .splitlines()
    )
    if overlap_ids != set(by_a) & set(by_b):
        raise ValueError("Observed reviewer overlap differs from frozen overlap IDs")
    overlap = sorted(overlap_ids)
    validity_agreement = cohen_kappa(
        [by_a[row_id]["y_valid"] for row_id in overlap],
        [by_b[row_id]["y_valid"] for row_id in overlap],
    )
    jointly_valid = [
        row_id
        for row_id in overlap
        if by_a[row_id]["y_valid"] == 1 and by_b[row_id]["y_valid"] == 1
    ]
    utility_agreement = (
        cohen_kappa(
            [by_a[row_id]["y_utility"] for row_id in jointly_valid],
            [by_b[row_id]["y_utility"] for row_id in jointly_valid],
        )
        if jointly_valid
        else None
    )
    primary = [row for row in rows_a + rows_b if row["assignment_role"] == "primary"]
    if len(primary) != manifest["total_rows"] or len(
        {row["audit_row_id"] for row in primary}
    ) != len(primary):
        raise ValueError("Primary assignments do not cover each audit row exactly once")
    invalid_count = sum(row["y_valid"] == 0 for row in primary)
    valid_rows = [row for row in primary if row["y_valid"] == 1]
    low_utility_count = sum(row["y_utility"] == 0 for row in valid_rows)
    low_reliability_ids = {
        row["audit_row_id"]
        for row in primary
        if row["y_valid"] == 0 or row["y_utility"] == 0
    }
    question_ids = {row["question_id"] for row in primary}
    affected_question_ids = {
        row["question_id"]
        for row in primary
        if row["audit_row_id"] in low_reliability_ids
    }
    disagreements = [
        {
            "audit_row_id": row_id,
            "a_valid": by_a[row_id]["y_valid"],
            "b_valid": by_b[row_id]["y_valid"],
            "a_utility": by_a[row_id]["y_utility"],
            "b_utility": by_b[row_id]["y_utility"],
        }
        for row_id in overlap
        if (
            by_a[row_id]["y_valid"] != by_b[row_id]["y_valid"]
            or (
                by_a[row_id]["y_valid"] == by_b[row_id]["y_valid"] == 1
                and by_a[row_id]["y_utility"] != by_b[row_id]["y_utility"]
            )
        )
    ]
    result = {
        "schema_version": 1,
        "run_id": "R022",
        "status": "NEEDS_ADJUDICATION" if disagreements else "PASS",
        "package_manifest_path": str(manifest_path),
        "package_manifest_sha256": sha256_file(manifest_path),
        "validity_agreement": validity_agreement,
        "utility_agreement_on_jointly_valid": utility_agreement,
        "invalid_step_prevalence": wilson_interval(invalid_count, len(primary)),
        "low_utility_among_valid_prevalence": wilson_interval(
            low_utility_count, len(valid_rows)
        ),
        "low_reliability_step_prevalence": wilson_interval(
            len(low_reliability_ids), len(primary)
        ),
        "affected_trajectory_prevalence": wilson_interval(
            len(affected_question_ids), len(question_ids)
        ),
        "disagreement_count": len(disagreements),
    }
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "review_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (destination / "disagreements.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in disagreements:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result
