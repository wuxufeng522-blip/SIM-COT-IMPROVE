from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import random

from .audit import question_id, sha256_file
from .corruptions import (
    DEVELOPMENT_FAMILIES,
    compensating_error_variant,
    development_variants,
    equivalent_variant,
    parse_checked_equation,
)
from .labels import ReliabilityRow, SplitName, stable_variant_id
from .official_adapter import OfficialExample, iter_icot_examples
from .splits import (
    assign_question_splits,
    split_manifest_sha256,
    validate_question_isolation,
)


def _jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return sha256(payload).hexdigest()


def _row(
    *,
    example: OfficialExample,
    split: SplitName,
    step_index: int,
    candidate_step: str,
    family: str,
    template_id: str,
    y_valid: int,
    y_utility: int | None,
    metadata: dict[str, Any] | None = None,
) -> ReliabilityRow:
    qid = question_id(example.question)
    pair_id = f"{qid}:{step_index}"
    return ReliabilityRow(
        variant_id=stable_variant_id(
            question_id=qid,
            step_index=step_index,
            family=family,
            template_id=template_id,
            candidate_step=candidate_step,
        ),
        question_id=qid,
        source_index=example.idx,
        split=split,
        question=example.question,
        answer=example.answer,
        step_index=step_index,
        trajectory_steps=len(example.steps),
        prefix_steps=example.steps[:step_index],
        clean_step=example.steps[step_index],
        candidate_step=candidate_step,
        family=family,
        template_id=template_id,
        pair_id=pair_id,
        y_valid=y_valid,
        y_utility=y_utility,
        metadata=metadata or {},
    )


def build_rows_for_examples(
    examples: Iterable[OfficialExample],
    *,
    split_by_question: dict[str, SplitName],
) -> tuple[list[ReliabilityRow], list[ReliabilityRow], dict[str, int]]:
    development: list[ReliabilityRow] = []
    sealed: list[ReliabilityRow] = []
    diagnostics: Counter[str] = Counter()

    for example in examples:
        qid = question_id(example.question)
        split = split_by_question[qid]
        for step_index, step in enumerate(example.steps):
            if parse_checked_equation(step) is None:
                diagnostics["steps_rejected_not_checked_equation"] += 1
                continue
            diagnostics["eligible_clean_steps"] += 1
            development.append(
                _row(
                    example=example,
                    split=split,
                    step_index=step_index,
                    candidate_step=step,
                    family="clean_original",
                    template_id="source_v1",
                    y_valid=1,
                    y_utility=1,
                )
            )

            equivalent = equivalent_variant(step)
            if equivalent is not None:
                development.append(
                    _row(
                        example=example,
                        split=split,
                        step_index=step_index,
                        candidate_step=equivalent.text,
                        family=equivalent.family,
                        template_id=equivalent.template_id,
                        y_valid=equivalent.y_valid,
                        y_utility=equivalent.y_utility,
                        metadata=equivalent.metadata,
                    )
                )

            variants = development_variants(
                step,
                prefix_steps=example.steps[:step_index],
                later_steps=example.steps[step_index + 1 :],
                step_index=step_index,
            )
            diagnostics.update(f"generated_{variant.family}" for variant in variants)
            for variant in variants:
                development.append(
                    _row(
                        example=example,
                        split=split,
                        step_index=step_index,
                        candidate_step=variant.text,
                        family=variant.family,
                        template_id=variant.template_id,
                        y_valid=variant.y_valid,
                        y_utility=variant.y_utility,
                        metadata=variant.metadata,
                    )
                )

            if split == "head_audit":
                compensation = compensating_error_variant(step)
                if compensation is not None:
                    sealed.append(
                        _row(
                            example=example,
                            split=split,
                            step_index=step_index,
                            candidate_step=step,
                            family="sealed_clean_reference",
                            template_id="source_v1",
                            y_valid=1,
                            y_utility=1,
                        )
                    )
                    sealed.append(
                        _row(
                            example=example,
                            split=split,
                            step_index=step_index,
                            candidate_step=compensation.text,
                            family=compensation.family,
                            template_id=compensation.template_id,
                            y_valid=compensation.y_valid,
                            y_utility=compensation.y_utility,
                            metadata=compensation.metadata,
                        )
                    )

    return development, sealed, dict(diagnostics)


def _read_natural_audit_question_ids(
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> set[str]:
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Natural-audit manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("run_id") != "R020" or manifest.get("frozen") is not True:
        raise ValueError("Reliability data requires a frozen R020 manifest")
    rows_path = Path(manifest["audit_rows_path"])
    if not rows_path.is_absolute():
        rows_path = (manifest_path.parent / rows_path).resolve()
    if sha256_file(rows_path) != manifest["audit_rows_sha256"]:
        raise ValueError("Natural-audit rows SHA-256 mismatch")
    return {
        json.loads(line)["question_id"]
        for line in rows_path.read_text(encoding="utf-8").splitlines()
        if line
    }


def build_reliability_dataset(
    config: dict[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def project_path(value: str) -> Path:
        target = (root / value).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes project root: {value}")
        return target

    dataset_path = project_path(config["dataset_path"])
    if sha256_file(dataset_path) != config["dataset_sha256"]:
        raise ValueError("Official training dataset SHA-256 mismatch")
    natural_audit_ids = _read_natural_audit_question_ids(
        project_path(config["natural_audit_manifest_path"]),
        expected_manifest_sha256=config["natural_audit_manifest_sha256"],
    )

    unique: dict[str, OfficialExample] = {}
    duplicates = 0
    questions_without_checked_equations = 0
    for example in iter_icot_examples(dataset_path):
        qid = question_id(example.question)
        if qid in natural_audit_ids:
            continue
        if qid in unique:
            duplicates += 1
            continue
        if not any(parse_checked_equation(step) is not None for step in example.steps):
            questions_without_checked_equations += 1
            continue
        unique[qid] = example
    question_count = config["question_count"]
    if question_count <= 0 or question_count > len(unique):
        raise ValueError("question_count is outside the eligible population")
    population = sorted(unique)
    selected_ids = random.Random(config["selection_seed"]).sample(
        population,
        question_count,
    )
    selected = [unique[qid] for qid in selected_ids]
    split_mapping = assign_question_splits(
        selected_ids,
        seed=config["split_seed"],
        train_fraction=config["train_fraction"],
        validation_fraction=config["validation_fraction"],
    )

    development, sealed, diagnostics = build_rows_for_examples(
        selected,
        split_by_question=split_mapping,
    )
    development_dicts = [row.to_dict() for row in development]
    sealed_dicts = [row.to_dict() for row in sealed]
    split_counts = validate_question_isolation(development_dicts)
    assigned_split_counts = dict(Counter(split_mapping.values()))
    if set(row["question_id"] for row in development_dicts) & natural_audit_ids:
        raise AssertionError("Natural-audit questions leaked into reliability data")

    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, dict[str, Any]] = {}
    for split in ("head_train", "head_validation", "head_audit"):
        rows = [row for row in development_dicts if row["split"] == split]
        rows.sort(key=lambda row: row["variant_id"])
        path = output_dir / f"{split}.jsonl"
        digest = _write_bytes(path, _jsonl_bytes(rows))
        split_files[split] = {"path": str(path), "rows": len(rows), "sha256": digest}

    sealed_dicts.sort(key=lambda row: row["variant_id"])
    sealed_path = output_dir / "sealed_compensating_error.jsonl"
    sealed_sha = _write_bytes(sealed_path, _jsonl_bytes(sealed_dicts))
    family_counts = Counter(row["family"] for row in development_dicts)
    label_counts = Counter(
        f"v{row['y_valid']}_u{row['y_utility']}" for row in development_dicts
    )
    manifest = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "PASS",
        "dataset_path": str(dataset_path),
        "dataset_sha256": config["dataset_sha256"],
        "natural_audit_manifest_path": str(
            project_path(config["natural_audit_manifest_path"])
        ),
        "natural_audit_manifest_sha256": config["natural_audit_manifest_sha256"],
        "natural_audit_questions_excluded": len(natural_audit_ids),
        "eligible_unique_questions": len(unique),
        "duplicate_source_rows_ignored": duplicates,
        "questions_without_checked_equations_rejected": (
            questions_without_checked_equations
        ),
        "selected_questions": len(selected_ids),
        "selection_seed": config["selection_seed"],
        "split_seed": config["split_seed"],
        "assigned_question_split_counts": assigned_split_counts,
        "represented_question_split_counts": split_counts,
        "split_mapping_sha256": split_manifest_sha256(split_mapping),
        "development_rows": len(development_dicts),
        "family_counts": dict(sorted(family_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "required_development_families": list(DEVELOPMENT_FAMILIES),
        "missing_development_families": sorted(
            set(DEVELOPMENT_FAMILIES) - set(family_counts)
        ),
        "generation_diagnostics": diagnostics,
        "split_files": split_files,
        "sealed_stress": {
            "family": "compensating_error",
            "path": str(sealed_path),
            "rows": len(sealed_dicts),
            "sha256": sealed_sha,
            "sealed": True,
            "opened": False,
            "access_rule": "Requires a frozen reliability-head manifest binding this SHA-256.",
        },
    }
    manifest["gate_passed"] = (
        not manifest["missing_development_families"]
        and split_counts == assigned_split_counts
        and len(sealed_dicts) > 0
    )
    manifest["status"] = "PASS" if manifest["gate_passed"] else "FAIL"
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def load_sealed_rows(
    *,
    dataset_manifest_path: str | Path,
    frozen_head_manifest_path: str | Path,
) -> list[dict[str, Any]]:
    dataset_manifest = json.loads(
        Path(dataset_manifest_path).read_text(encoding="utf-8")
    )
    frozen_head = json.loads(
        Path(frozen_head_manifest_path).read_text(encoding="utf-8")
    )
    sealed = dataset_manifest["sealed_stress"]
    if frozen_head.get("reliability_head_frozen") is not True:
        raise PermissionError("Sealed stress data cannot open before head freeze")
    if frozen_head.get("sealed_stress_sha256") != sealed["sha256"]:
        raise PermissionError("Frozen head manifest does not bind the sealed stress set")
    path = Path(sealed["path"])
    if sha256_file(path) != sealed["sha256"]:
        raise ValueError("Sealed stress SHA-256 mismatch")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
