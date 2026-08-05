from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import ast
import csv
import io
import json
import math
import random
import re

from .official_adapter import OfficialExample, iter_icot_examples


_NUMERIC_TEXT = re.compile(r"^[+\-*/().\d\s]+$")


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def question_id(question: str) -> str:
    return sha256(question.strip().encode("utf-8")).hexdigest()


def _eval_arithmetic_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_arithmetic_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_arithmetic_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _eval_arithmetic_node(node.left)
        right = _eval_arithmetic_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow) and right.denominator == 1 and abs(right) <= 12:
            return left ** int(right)
    raise ValueError("unsupported arithmetic expression")


def evaluate_arithmetic(text: str) -> Fraction:
    cleaned = text.replace(",", "").replace("$", "").strip()
    if not cleaned or not _NUMERIC_TEXT.fullmatch(cleaned):
        raise ValueError("non-arithmetic text")
    return _eval_arithmetic_node(ast.parse(cleaned, mode="eval"))


def numeric_close(left: Fraction, right: Fraction) -> bool:
    left_float = float(left)
    right_float = float(right)
    tolerance = max(1e-6, 5e-5 * max(1.0, abs(left_float), abs(right_float)))
    return math.isclose(left_float, right_float, rel_tol=0.0, abs_tol=tolerance)


def numeric_answer_close(value: Fraction, answer_value: Fraction, answer_text: str) -> bool:
    if numeric_close(value, answer_value):
        return True
    stripped = answer_text.replace(",", "").strip()
    if re.fullmatch(r"[+-]?\d+\.\d+", stripped):
        decimals = len(stripped.rsplit(".", 1)[1])
        return round(float(value), decimals) == float(answer_value)
    return False


def numeric_result_close(value: Fraction, stated_value: Fraction, stated_text: str) -> bool:
    if numeric_close(value, stated_value):
        return True
    stripped = stated_text.replace(",", "").strip()
    if re.fullmatch(r"[+-]?\d*\.\d+", stripped):
        decimals = len(stripped.rsplit(".", 1)[1])
        half_unit = 0.5 * (10 ** -decimals)
        return abs(float(value) - float(stated_value)) <= half_unit + 1e-12
    return False


def _numeric_answer(answer: str) -> Fraction | None:
    try:
        return evaluate_arithmetic(answer)
    except (ValueError, SyntaxError, ZeroDivisionError):
        return None


@dataclass(frozen=True)
class AutoAudit:
    delimiter_ok: bool
    equation_present: bool
    arithmetic_status: str
    arithmetic_lhs: str | None
    arithmetic_rhs: str | None
    duplicate_within_trajectory: bool
    final_answer_status: str
    candidate_flags: tuple[str, ...]


def audit_step(
    step: str,
    *,
    step_index: int,
    all_steps: tuple[str, ...],
    answer: str,
) -> AutoAudit:
    delimiter_ok = step.startswith("<<") and step.endswith(">>")
    content = step[2:-2].strip() if delimiter_ok else step.strip()
    equation_present = "=" in content
    lhs_text: str | None = None
    rhs_text: str | None = None
    arithmetic_status = "not_equation"
    flags: list[str] = []
    if not delimiter_ok:
        flags.append("malformed_delimiters")
    if not equation_present:
        flags.append("missing_equation")
    else:
        lhs_text, rhs_text = (part.strip() for part in content.rsplit("=", 1))
        try:
            lhs_value = evaluate_arithmetic(lhs_text)
            rhs_value = evaluate_arithmetic(rhs_text)
            if numeric_result_close(lhs_value, rhs_value, rhs_text):
                arithmetic_status = "checked_match"
            else:
                arithmetic_status = "checked_mismatch"
                flags.append("arithmetic_mismatch_candidate")
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            arithmetic_status = "unparsed"
            flags.append("manual_arithmetic_review")

    duplicate = step in all_steps[:step_index]
    if duplicate:
        flags.append("duplicate_step_candidate")

    final_answer_status = "not_final_step"
    if step_index == len(all_steps) - 1:
        rhs_value = _numeric_answer(rhs_text) if rhs_text is not None else None
        answer_value = _numeric_answer(answer)
        if rhs_value is None or answer_value is None:
            final_answer_status = "unparsed"
            flags.append("manual_final_answer_review")
        elif numeric_answer_close(rhs_value, answer_value, answer):
            final_answer_status = "checked_match"
        else:
            final_answer_status = "checked_mismatch"
            flags.append("final_answer_mismatch_candidate")

    return AutoAudit(
        delimiter_ok=delimiter_ok,
        equation_present=equation_present,
        arithmetic_status=arithmetic_status,
        arithmetic_lhs=lhs_text,
        arithmetic_rhs=rhs_text,
        duplicate_within_trajectory=duplicate,
        final_answer_status=final_answer_status,
        candidate_flags=tuple(flags),
    )


def select_question_clusters(
    examples: Iterable[OfficialExample],
    *,
    seed: int,
    question_count: int,
) -> tuple[list[OfficialExample], dict[str, int]]:
    unique: dict[str, OfficialExample] = {}
    multiplicity: dict[str, int] = {}
    for example in examples:
        qid = question_id(example.question)
        if qid in unique and unique[qid].question.strip() != example.question.strip():
            raise ValueError("Question SHA-256 collision")
        unique.setdefault(qid, example)
        multiplicity[qid] = multiplicity.get(qid, 0) + 1
    if question_count <= 0 or question_count > len(unique):
        raise ValueError("Requested question_count is outside the unique-question pool")
    population = sorted(unique)
    selected_ids = random.Random(seed).sample(population, question_count)
    return [unique[qid] for qid in selected_ids], multiplicity


def build_audit_rows(
    selected: Iterable[OfficialExample], multiplicity: dict[str, int]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for draw_index, example in enumerate(selected):
        qid = question_id(example.question)
        for step_index, step in enumerate(example.steps):
            automatic = audit_step(
                step,
                step_index=step_index,
                all_steps=example.steps,
                answer=example.answer,
            )
            rows.append(
                {
                    "audit_row_id": f"{qid}:{step_index}",
                    "draw_index": draw_index,
                    "question_id": qid,
                    "source_index": example.idx,
                    "source_question_multiplicity": multiplicity[qid],
                    "step_index": step_index,
                    "trajectory_steps": len(example.steps),
                    "question": example.question,
                    "step": step,
                    "answer": example.answer,
                    **asdict(automatic),
                    "candidate_flags": list(automatic.candidate_flags),
                    "y_valid": None,
                    "y_utility": None,
                    "reviewer": None,
                    "review_notes": None,
                }
            )
    return rows


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    return text.encode("utf-8")


def _review_csv_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    columns = (
        "audit_row_id",
        "draw_index",
        "question_id",
        "source_index",
        "step_index",
        "trajectory_steps",
        "question",
        "step",
        "answer",
        "candidate_flags",
        "arithmetic_status",
        "final_answer_status",
        "y_valid",
        "y_utility",
        "reviewer",
        "review_notes",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        copy = dict(row)
        copy["candidate_flags"] = ";".join(row["candidate_flags"])
        writer.writerow(copy)
    return buffer.getvalue().encode("utf-8-sig")


def write_frozen_audit(
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    seed: int,
    question_count: int,
    min_steps: int,
    expected_dataset_sha256: str | None = None,
) -> dict[str, Any]:
    source = Path(dataset_path).resolve()
    destination = Path(output_dir).resolve()
    manifest_path = destination / "freeze_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Frozen audit already exists: {manifest_path}")
    source_sha256 = sha256_file(source)
    if expected_dataset_sha256 is not None and source_sha256 != expected_dataset_sha256:
        raise ValueError("Natural-audit dataset SHA-256 mismatch")

    selected, multiplicity = select_question_clusters(
        iter_icot_examples(source), seed=seed, question_count=question_count
    )
    rows = build_audit_rows(selected, multiplicity)
    if len(rows) < min_steps:
        raise ValueError(f"Audit has {len(rows)} steps; registered minimum is {min_steps}")
    destination.mkdir(parents=True, exist_ok=False)
    rows_path = destination / "audit_rows.jsonl"
    review_path = destination / "human_review.csv"
    rows_payload = _jsonl_bytes(rows)
    review_payload = _review_csv_bytes(rows)
    rows_path.write_bytes(rows_payload)
    review_path.write_bytes(review_payload)

    flag_counts: dict[str, int] = {}
    arithmetic_counts: dict[str, int] = {}
    for row in rows:
        arithmetic_counts[row["arithmetic_status"]] = (
            arithmetic_counts.get(row["arithmetic_status"], 0) + 1
        )
        for flag in row["candidate_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    selected_ids = [question_id(example.question) for example in selected]
    manifest = {
        "schema_version": 1,
        "run_id": "R020",
        "status": "PASS",
        "frozen": True,
        "sampling_unit": "unique exact question cluster",
        "sampling_method": "uniform sample without replacement from sorted SHA-256 IDs",
        "seed": seed,
        "question_count": len(selected),
        "step_count": len(rows),
        "minimum_step_count": min_steps,
        "dataset_path": str(source),
        "dataset_sha256": source_sha256,
        "unique_question_population": len(multiplicity),
        "duplicate_source_rows": sum(value - 1 for value in multiplicity.values()),
        "selected_question_ids": selected_ids,
        "selected_question_ids_sha256": sha256(
            "\n".join(selected_ids).encode("ascii")
        ).hexdigest(),
        "audit_rows_path": str(rows_path),
        "audit_rows_sha256": sha256(rows_payload).hexdigest(),
        "human_review_path": str(review_path),
        "human_review_sha256": sha256(review_payload).hexdigest(),
        "labels_are_blank": True,
        "automatic_rules_are_triage_only": True,
        "arithmetic_status_counts": arithmetic_counts,
        "candidate_flag_counts": flag_counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def write_auto_triage_revision(
    *,
    parent_manifest_path: str | Path,
    output_dir: str | Path,
    expected_parent_rows_sha256: str,
) -> dict[str, Any]:
    parent_path = Path(parent_manifest_path).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("frozen") is not True or parent.get("run_id") != "R020":
        raise ValueError("R021 requires a frozen R020 parent manifest")
    if parent["audit_rows_sha256"] != expected_parent_rows_sha256:
        raise ValueError("Configured R020 rows SHA-256 does not match parent manifest")
    rows_path = Path(parent["audit_rows_path"])
    if sha256_file(rows_path) != expected_parent_rows_sha256:
        raise ValueError("Frozen R020 audit rows failed SHA-256 verification")
    original_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectories: dict[str, list[dict[str, Any]]] = {}
    for row in original_rows:
        trajectories.setdefault(row["question_id"], []).append(row)
    revised_rows: list[dict[str, Any]] = []
    changed_flag_rows = 0
    for trajectory in trajectories.values():
        ordered = sorted(trajectory, key=lambda row: row["step_index"])
        all_steps = tuple(row["step"] for row in ordered)
        if [row["step_index"] for row in ordered] != list(range(len(ordered))):
            raise ValueError("R020 trajectory step indices are not contiguous")
        for row in ordered:
            automatic = audit_step(
                row["step"],
                step_index=row["step_index"],
                all_steps=all_steps,
                answer=row["answer"],
            )
            revised = dict(row)
            revised.update(asdict(automatic))
            revised["candidate_flags"] = list(automatic.candidate_flags)
            if revised["candidate_flags"] != row["candidate_flags"]:
                changed_flag_rows += 1
            revised_rows.append(revised)
    revised_rows.sort(key=lambda row: (row["draw_index"], row["step_index"]))

    destination = Path(output_dir).resolve()
    manifest_path = destination / "triage_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"R021 triage already exists: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=False)
    rows_payload = _jsonl_bytes(revised_rows)
    review_payload = _review_csv_bytes(revised_rows)
    revised_path = destination / "triage_rows.jsonl"
    review_path = destination / "human_review.csv"
    revised_path.write_bytes(rows_payload)
    review_path.write_bytes(review_payload)

    flag_counts: dict[str, int] = {}
    arithmetic_counts: dict[str, int] = {}
    final_counts: dict[str, int] = {}
    for row in revised_rows:
        arithmetic_counts[row["arithmetic_status"]] = (
            arithmetic_counts.get(row["arithmetic_status"], 0) + 1
        )
        final_counts[row["final_answer_status"]] = (
            final_counts.get(row["final_answer_status"], 0) + 1
        )
        for flag in row["candidate_flags"]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    manifest = {
        "schema_version": 1,
        "run_id": "R021",
        "status": "PASS",
        "parent_run_id": "R020",
        "parent_manifest_path": str(parent_path),
        "parent_manifest_sha256": sha256_file(parent_path),
        "parent_rows_sha256": expected_parent_rows_sha256,
        "question_count": len(trajectories),
        "step_count": len(revised_rows),
        "changed_flag_rows_vs_r020": changed_flag_rows,
        "triage_rows_path": str(revised_path),
        "triage_rows_sha256": sha256(rows_payload).hexdigest(),
        "human_review_path": str(review_path),
        "human_review_sha256": sha256(review_payload).hexdigest(),
        "labels_are_blank": True,
        "automatic_rules_are_triage_only": True,
        "rounding_policy": "decimal RHS accepts half-unit-in-last-place",
        "arithmetic_status_counts": arithmetic_counts,
        "final_answer_status_counts": final_counts,
        "candidate_flag_counts": flag_counts,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def _review_assignment_csv_bytes(
    rows: Iterable[dict[str, Any]], *, reviewer_code: str
) -> bytes:
    columns = (
        "review_order",
        "assignment_role",
        "audit_row_id",
        "question_id",
        "step_index",
        "trajectory_steps",
        "question",
        "trajectory",
        "target_step",
        "answer",
        "y_valid",
        "y_utility",
        "confidence",
        "reviewer",
        "review_notes",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for order, row in enumerate(rows):
        writer.writerow(
            {
                "review_order": order,
                "assignment_role": row["_assignment_role"],
                "audit_row_id": row["audit_row_id"],
                "question_id": row["question_id"],
                "step_index": row["step_index"],
                "trajectory_steps": row["trajectory_steps"],
                "question": row["question"],
                "trajectory": row["trajectory"],
                "target_step": row["step"],
                "answer": row["answer"],
                "y_valid": "",
                "y_utility": "",
                "confidence": "",
                "reviewer": reviewer_code,
                "review_notes": "",
            }
        )
    return buffer.getvalue().encode("utf-8-sig")


def write_blinded_review_package(
    *,
    parent_manifest_path: str | Path,
    output_dir: str | Path,
    expected_parent_rows_sha256: str,
    assignment_seed: int,
    overlap_fraction: float,
) -> dict[str, Any]:
    if not (0.0 < overlap_fraction < 1.0):
        raise ValueError("overlap_fraction must be between 0 and 1")
    parent_path = Path(parent_manifest_path).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("frozen") is not True or parent.get("run_id") != "R020":
        raise ValueError("Review packaging requires a frozen R020 manifest")
    rows_path = Path(parent["audit_rows_path"])
    if parent["audit_rows_sha256"] != expected_parent_rows_sha256:
        raise ValueError("Configured parent rows SHA-256 mismatch")
    if sha256_file(rows_path) != expected_parent_rows_sha256:
        raise ValueError("Frozen audit rows failed SHA-256 verification")
    raw_rows = [
        json.loads(line)
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in raw_rows:
        by_question.setdefault(row["question_id"], []).append(row)
    blinded_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        trajectory = sorted(
            by_question[row["question_id"]], key=lambda item: item["step_index"]
        )
        blinded = {
            key: row[key]
            for key in (
                "audit_row_id",
                "question_id",
                "step_index",
                "trajectory_steps",
                "question",
                "step",
                "answer",
            )
        }
        blinded["trajectory"] = " || ".join(item["step"] for item in trajectory)
        blinded_rows.append(blinded)

    rng = random.Random(assignment_seed)
    shuffled = list(blinded_rows)
    rng.shuffle(shuffled)
    primary_a = shuffled[::2]
    primary_b = shuffled[1::2]
    overlap_count = round(len(shuffled) * overlap_fraction)
    overlap_ids = set(rng.sample([row["audit_row_id"] for row in shuffled], overlap_count))
    by_id = {row["audit_row_id"]: row for row in shuffled}
    a_ids = {row["audit_row_id"] for row in primary_a}
    b_ids = {row["audit_row_id"] for row in primary_b}
    reviewer_a = [dict(row, _assignment_role="primary") for row in primary_a] + [
        dict(by_id[row_id], _assignment_role="overlap")
        for row_id in sorted(overlap_ids - a_ids)
    ]
    reviewer_b = [dict(row, _assignment_role="primary") for row in primary_b] + [
        dict(by_id[row_id], _assignment_role="overlap")
        for row_id in sorted(overlap_ids - b_ids)
    ]
    random.Random(assignment_seed + 1).shuffle(reviewer_a)
    random.Random(assignment_seed + 2).shuffle(reviewer_b)

    destination = Path(output_dir).resolve()
    manifest_path = destination / "review_package_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Review package already exists: {manifest_path}")
    destination.mkdir(parents=True, exist_ok=False)
    payload_a = _review_assignment_csv_bytes(reviewer_a, reviewer_code="A")
    payload_b = _review_assignment_csv_bytes(reviewer_b, reviewer_code="B")
    path_a = destination / "reviewer_a.csv"
    path_b = destination / "reviewer_b.csv"
    path_a.write_bytes(payload_a)
    path_b.write_bytes(payload_b)
    overlap_path = destination / "overlap_ids.txt"
    overlap_payload = ("\n".join(sorted(overlap_ids)) + "\n").encode("ascii")
    overlap_path.write_bytes(overlap_payload)

    manifest = {
        "schema_version": 1,
        "run_id": "R022-PREP",
        "status": "PASS",
        "parent_run_id": "R020",
        "parent_manifest_path": str(parent_path),
        "parent_manifest_sha256": sha256_file(parent_path),
        "parent_rows_sha256": expected_parent_rows_sha256,
        "assignment_seed": assignment_seed,
        "total_rows": len(shuffled),
        "primary_rows_reviewer_a": len(primary_a),
        "primary_rows_reviewer_b": len(primary_b),
        "overlap_fraction": overlap_fraction,
        "overlap_rows": overlap_count,
        "reviewer_a_rows": len(reviewer_a),
        "reviewer_b_rows": len(reviewer_b),
        "reviewer_a_path": str(path_a),
        "reviewer_a_sha256": sha256(payload_a).hexdigest(),
        "reviewer_b_path": str(path_b),
        "reviewer_b_sha256": sha256(payload_b).hexdigest(),
        "overlap_ids_path": str(overlap_path),
        "overlap_ids_sha256": sha256(overlap_payload).hexdigest(),
        "blinded_columns_excluded": [
            "candidate_flags",
            "arithmetic_status",
            "final_answer_status",
            "source_index",
        ],
        "labels_are_blank": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def validate_human_labels(rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        valid = row.get("y_valid")
        utility = row.get("y_utility")
        if valid not in (0, 1):
            raise ValueError(f"Missing/invalid Validity label: {row.get('audit_row_id')}")
        if valid == 0 and utility is not None:
            raise ValueError("Utility must be null when Validity is 0")
        if valid == 1 and utility not in (0, 1):
            raise ValueError("Utility must be binary when Validity is 1")


def assert_audit_ids_excluded(
    training_question_ids: Iterable[str], manifest: dict[str, Any]
) -> None:
    overlap = set(training_question_ids) & set(manifest["selected_question_ids"])
    if overlap:
        raise ValueError(f"Audit leakage detected for {len(overlap)} question IDs")
