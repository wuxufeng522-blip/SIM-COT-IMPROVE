from __future__ import annotations

from pathlib import Path
from statistics import mean, median, pstdev, stdev
from typing import Any
import argparse
import gc
import json
import re
import subprocess
import sys
import time
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.error_cancellation_data import evaluate_arithmetic, parse_equation  # noqa: E402
from reliable_simcot.full_conflict_data import canonical_hash, load_frozen_schedule  # noqa: E402
from reliable_simcot.full_conflict_evaluation import (  # noqa: E402
    exact_mcnemar,
    paired_bootstrap_damage_pp,
)
from reliable_simcot.full_conflict_experiment import (  # noqa: E402
    ACCIDENTAL_CORRECT_50_ARM,
    REDUNDANT_STEPS_50_ARM,
    STEP_ORDER_REVERSAL_50_ARM,
    UNRELATED_ACCIDENTAL_CORRECT_50_ARM,
    arm_targets,
    attach_unrelated_donors,
    checkpoint_path,
    run_full_conflict_training,
    training_directory,
)
from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402
from reliable_simcot.official_adapter import (  # noqa: E402
    OfficialExample,
    build_tokenizer,
    evaluate_checkpoint,
    load_official_model,
)
from reliable_simcot.oracle_weighting import tokenize_step_targets  # noqa: E402


REGISTERED_ARMS = (REDUNDANT_STEPS_50_ARM, STEP_ORDER_REVERSAL_50_ARM)
ACCIDENTAL_REGISTERED_ARMS = (ACCIDENTAL_CORRECT_50_ARM,)
UNRELATED_REGISTERED_ARMS = (UNRELATED_ACCIDENTAL_CORRECT_50_ARM,)


def _load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def _save_state(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    path = (ROOT / config["pipeline_state_path"]).resolve()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "run_id": config.get("run_id", "HBCO-V15"),
            "started_unix": time.time(),
            "completed_training": [],
            "completed_evaluation": [],
        }
    if updates.get("status") in {"RUNNING", "PASS"}:
        for key in ("error_type", "error", "traceback"):
            state.pop(key, None)
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)
    return state


def _gpu_used_mib() -> int:
    process = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(process.stdout.splitlines()[0].strip())


def _preflight(config: dict[str, Any], label: str) -> None:
    limit = int(config["preflight_max_used_mib"])
    used = -1
    for attempt in range(1, 4):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        used = _gpu_used_mib()
        if used <= limit:
            print(f"GPU preflight {label}: {used} MiB <= {limit} MiB", flush=True)
            return
        if attempt < 3:
            print(
                f"GPU preflight {label}: transient {used} MiB > {limit} MiB; cleanup retry {attempt}/2",
                flush=True,
            )
            time.sleep(2.0)
    raise RuntimeError(f"GPU preflight failed for {label}: {used} MiB > {limit} MiB")


def _example(row: dict[str, Any]) -> OfficialExample:
    return OfficialExample(
        idx=int(row["source_idx"]),
        question=row["question"],
        steps=tuple(row["clean_steps"]),
        answer=row["answer"],
    )


def _data_audit(config: dict[str, Any]) -> dict[str, Any]:
    schedule_path = (ROOT / config["frozen_schedule_path"]).resolve()
    if sha256_file(schedule_path) != config["source_schedule_file_sha256"]:
        raise ValueError("Source schedule file SHA-256 mismatch")
    schedule = load_frozen_schedule(config, ROOT)
    if schedule["schedule_sha256"] != config["source_schedule_sha256"]:
        raise ValueError("Source schedule semantic SHA-256 mismatch")
    active_ids: list[str] = []
    failures: list[str] = []
    reversal_position_changes: list[int] = []
    samples: list[dict[str, Any]] = []
    for row in schedule["train_entries"]:
        example = _example(row)
        redundant, redundant_scale, redundant_changed = arm_targets(
            REDUNDANT_STEPS_50_ARM, example, row
        )
        reversal, reversal_scale, reversal_changed = arm_targets(
            STEP_ORDER_REVERSAL_50_ARM, example, row
        )
        active = row["coverage_tier"] in {0, 1}
        if active:
            active_ids.append(row["question_id"])
            if row["full_chain"] is None or len(row["full_chain"]["steps"]) != 5:
                failures.append(f"{row['question_id']}:missing_severe_chain")
            if redundant_changed != 5 or redundant_scale != 1.0:
                failures.append(f"{row['question_id']}:bad_redundant_activation")
            if reversal != tuple(reversed(example.steps)) or reversal_scale != 1.0:
                failures.append(f"{row['question_id']}:bad_reversal")
            reversal_position_changes.append(reversal_changed)
            for clean_step, redundant_step in zip(example.steps, redundant, strict=True):
                clean_equation = parse_equation(clean_step)
                expected_prefix = clean_step + " "
                if not redundant_step.startswith(expected_prefix):
                    failures.append(f"{row['question_id']}:clean_text_not_preserved")
                    break
                identity = parse_equation(redundant_step[len(expected_prefix) :])
                if (
                    not identity.is_true
                    or "+0" not in identity.expression
                    or identity.result_text != clean_equation.result_text
                ):
                    failures.append(f"{row['question_id']}:non_neutral_redundancy")
                    break
            if len(samples) < 5:
                samples.append(
                    {
                        "question_id": row["question_id"],
                        "answer": row["answer"],
                        "clean_steps": list(example.steps),
                        "redundant_steps": list(redundant),
                        "severe_steps": list(row["full_chain"]["steps"]),
                        "reversed_steps": list(reversal),
                    }
                )
        elif redundant != example.steps or reversal != example.steps:
            failures.append(f"{row['question_id']}:inactive_row_changed")
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "source_schedule_sha256": schedule["schedule_sha256"],
        "examples": len(schedule["train_entries"]),
        "treatment_examples": len(active_ids),
        "treatment_question_ids": sorted(active_ids),
        "same_treatment_ids_for_redundant_severe_and_reversal": True,
        "redundancy_rule": "preserve each original treated equation verbatim and append a result+0=result identity in the same step slot",
        "reversal_rule": "[1,2,3,4,5] -> [5,4,3,2,1] on the treated half",
        "mean_changed_reversal_positions_per_treated_example": mean(reversal_position_changes),
        "question_and_answer_fields_untouched": True,
        "auxiliary_weight": 1.0,
        "failures": failures,
        "samples": samples,
        "official_test_opened": False,
    }
    result["audit_sha256"] = canonical_hash(result)
    atomic_json((ROOT / config["data_audit_path"]).resolve(), result)
    if result["status"] != "PASS" or len(active_ids) != int(config["treatment_examples"]):
        raise RuntimeError("Matched 50% data audit failed")
    return result


def _accidental_data_audit(config: dict[str, Any]) -> dict[str, Any]:
    schedule_path = (ROOT / config["frozen_schedule_path"]).resolve()
    if sha256_file(schedule_path) != config["source_schedule_file_sha256"]:
        raise ValueError("Source schedule file SHA-256 mismatch")
    schedule = load_frozen_schedule(config, ROOT)
    if schedule["schedule_sha256"] != config["source_schedule_sha256"]:
        raise ValueError("Source schedule semantic SHA-256 mismatch")
    tokenizer, _ = build_tokenizer((ROOT / config["base_model_dir"]).resolve())
    active_ids: list[str] = []
    failures: list[str] = []
    clean_tokens: list[int] = []
    severe_tokens: list[int] = []
    accidental_tokens: list[int] = []
    samples: list[dict[str, Any]] = []
    zero_gap_cases = 0
    for row in schedule["train_entries"]:
        example = _example(row)
        targets, scale, changed = arm_targets(ACCIDENTAL_CORRECT_50_ARM, example, row)
        active = row["coverage_tier"] in {0, 1}
        if not active:
            if targets != example.steps or scale != 1.0 or changed != 0:
                failures.append(f"{row['question_id']}:inactive_row_changed")
            continue
        active_ids.append(row["question_id"])
        full_chain = row.get("full_chain")
        if full_chain is None or len(full_chain.get("steps", ())) != 5:
            failures.append(f"{row['question_id']}:missing_severe_chain")
            continue
        severe = tuple(full_chain["steps"])
        clean = tuple(example.steps)
        if targets[:4] != severe[:4]:
            failures.append(f"{row['question_id']}:severe_prefix_changed")
        if scale != 1.0 or changed != 5 or len(targets) != 5:
            failures.append(f"{row['question_id']}:bad_activation")
        try:
            clean_eq = tuple(parse_equation(step) for step in clean)
            severe_eq = tuple(parse_equation(step) for step in severe)
            target_eq = tuple(parse_equation(step) for step in targets)
            answer_value = evaluate_arithmetic(example.answer.replace(",", ""))
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            failures.append(f"{row['question_id']}:parse:{type(error).__name__}")
            continue
        if not all(equation.is_true for equation in target_eq):
            failures.append(f"{row['question_id']}:false_arithmetic")
        if not all(targets[index] != clean[index] for index in range(4)):
            failures.append(f"{row['question_id']}:prefix_text_not_all_wrong")
        if not all(
            target_eq[index].result_value != clean_eq[index].result_value
            for index in range(4)
        ):
            failures.append(f"{row['question_id']}:prefix_value_not_all_wrong")
        expected_edges = [[0, 1], [1, 2], [2, 3], [3, 4]]
        if full_chain.get("validation", {}).get("dependency_edges") != expected_edges:
            failures.append(f"{row['question_id']}:source_dependency_chain_changed")
        if not target_eq[4].expression.startswith(target_eq[3].result_text):
            failures.append(f"{row['question_id']}:final_does_not_consume_wrong_state")
        if target_eq[4].result_value != answer_value:
            failures.append(f"{row['question_id']}:final_answer_not_recovered")
        if severe_eq[4].result_value == answer_value:
            failures.append(f"{row['question_id']}:source_severe_final_already_correct")
        if targets[4] == clean[4]:
            failures.append(f"{row['question_id']}:clean_final_step_copied")
        if target_eq[3].result_value == answer_value:
            zero_gap_cases += 1
            if "+1-1=" not in targets[4]:
                failures.append(f"{row['question_id']}:zero_gap_not_detoured")
        token_counts = {
            "clean": sum(len(group) for group in tokenize_step_targets(tokenizer, clean)),
            "severe": sum(len(group) for group in tokenize_step_targets(tokenizer, severe)),
            "accidental": sum(len(group) for group in tokenize_step_targets(tokenizer, targets)),
        }
        clean_tokens.append(token_counts["clean"])
        severe_tokens.append(token_counts["severe"])
        accidental_tokens.append(token_counts["accidental"])
        if len(samples) < 8:
            samples.append(
                {
                    "question_id": row["question_id"],
                    "answer": row["answer"],
                    "clean_steps": list(clean),
                    "severe_steps": list(severe),
                    "accidental_correct_steps": list(targets),
                    "token_counts": token_counts,
                }
            )
    ratios_to_clean = [
        accidental / clean
        for accidental, clean in zip(accidental_tokens, clean_tokens, strict=True)
    ]
    ratios_to_severe = [
        accidental / severe
        for accidental, severe in zip(accidental_tokens, severe_tokens, strict=True)
    ]
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "source_schedule_sha256": schedule["schedule_sha256"],
        "examples": len(schedule["train_entries"]),
        "treatment_examples": len(active_ids),
        "treatment_question_ids": sorted(active_ids),
        "same_treatment_ids_as_severe50": True,
        "construction_rule": "copy severe steps 1-4 verbatim; step 5 consumes the severe step-4 result and applies an unsupported exact offset to emit the official answer",
        "all_first_four_text_and_values_differ_from_clean": not any(
            "prefix_" in failure for failure in failures
        ),
        "all_final_results_equal_official_answer": not any(
            "final_answer_not_recovered" in failure for failure in failures
        ),
        "all_final_steps_differ_from_clean": not any(
            "clean_final_step_copied" in failure for failure in failures
        ),
        "zero_gap_detour_cases": zero_gap_cases,
        "token_count_summary": {
            "clean_mean": mean(clean_tokens),
            "severe_mean": mean(severe_tokens),
            "accidental_mean": mean(accidental_tokens),
            "accidental_to_clean_median_ratio": median(ratios_to_clean),
            "accidental_to_severe_median_ratio": median(ratios_to_severe),
        },
        "question_and_answer_fields_untouched": True,
        "auxiliary_weight": 1.0,
        "failures": failures,
        "samples": samples,
        "official_test_opened": False,
    }
    result["audit_sha256"] = canonical_hash(result)
    atomic_json((ROOT / config["data_audit_path"]).resolve(), result)
    if result["status"] != "PASS" or len(active_ids) != int(config["treatment_examples"]):
        raise RuntimeError("Accidental-correct 50% data audit failed")
    return result


def _unrelated_data_audit(config: dict[str, Any]) -> dict[str, Any]:
    schedule_path = (ROOT / config["frozen_schedule_path"]).resolve()
    if sha256_file(schedule_path) != config["source_schedule_file_sha256"]:
        raise ValueError("Source schedule file SHA-256 mismatch")
    schedule = load_frozen_schedule(config, ROOT)
    if schedule["schedule_sha256"] != config["source_schedule_sha256"]:
        raise ValueError("Source schedule semantic SHA-256 mismatch")
    mapping = attach_unrelated_donors(
        schedule, mapping_seed=int(config["unrelated_mapping_seed"])
    )
    tokenizer, _ = build_tokenizer((ROOT / config["base_model_dir"]).resolve())
    active_ids: list[str] = []
    donor_ids: list[str] = []
    failures: list[str] = []
    clean_tokens: list[int] = []
    v16_tokens: list[int] = []
    unrelated_tokens: list[int] = []
    question_jaccards: list[float] = []
    samples: list[dict[str, Any]] = []
    for row in schedule["train_entries"]:
        example = _example(row)
        targets, scale, changed = arm_targets(
            UNRELATED_ACCIDENTAL_CORRECT_50_ARM, example, row
        )
        active = row["coverage_tier"] in {0, 1}
        if not active:
            if targets != example.steps or scale != 1.0 or changed != 0:
                failures.append(f"{row['question_id']}:inactive_row_changed")
            continue
        active_ids.append(row["question_id"])
        donor = row.get("unrelated_donor")
        if donor is None:
            failures.append(f"{row['question_id']}:missing_donor")
            continue
        donor_ids.append(donor["question_id"])
        if donor["question_id"] == row["question_id"] or donor["question"] == row["question"]:
            failures.append(f"{row['question_id']}:donor_not_unrelated")
        donor_prefix = tuple(donor["steps"][:4])
        if targets[:4] != donor_prefix:
            failures.append(f"{row['question_id']}:donor_prefix_changed")
        if scale != 1.0 or changed != 5 or len(targets) != 5:
            failures.append(f"{row['question_id']}:bad_activation")
        try:
            clean_eq = tuple(parse_equation(step) for step in example.steps)
            target_eq = tuple(parse_equation(step) for step in targets)
            answer_value = evaluate_arithmetic(example.answer.replace(",", ""))
        except (SyntaxError, ValueError, ZeroDivisionError) as error:
            failures.append(f"{row['question_id']}:parse:{type(error).__name__}")
            continue
        if not all(equation.is_true for equation in target_eq):
            failures.append(f"{row['question_id']}:false_arithmetic")
        if not all(
            target_eq[index].result_value != clean_eq[index].result_value
            for index in range(4)
        ):
            failures.append(f"{row['question_id']}:prefix_value_collision")
        if target_eq[4].result_value != answer_value:
            failures.append(f"{row['question_id']}:final_answer_not_recovered")
        if not target_eq[4].expression.startswith(target_eq[3].result_text):
            failures.append(f"{row['question_id']}:final_does_not_consume_donor_state")
        if targets[4] == example.steps[4]:
            failures.append(f"{row['question_id']}:clean_final_step_copied")
        same_problem_targets, _, _ = arm_targets(ACCIDENTAL_CORRECT_50_ARM, example, row)
        token_counts = {
            "clean": sum(len(group) for group in tokenize_step_targets(tokenizer, example.steps)),
            "v16_same_problem": sum(
                len(group) for group in tokenize_step_targets(tokenizer, same_problem_targets)
            ),
            "v17_unrelated": sum(
                len(group) for group in tokenize_step_targets(tokenizer, targets)
            ),
        }
        clean_tokens.append(token_counts["clean"])
        v16_tokens.append(token_counts["v16_same_problem"])
        unrelated_tokens.append(token_counts["v17_unrelated"])
        target_words = set(re.findall(r"[a-z]+", row["question"].lower()))
        donor_words = set(re.findall(r"[a-z]+", donor["question"].lower()))
        union = target_words | donor_words
        question_jaccards.append(len(target_words & donor_words) / len(union) if union else 0.0)
        if len(samples) < 8:
            samples.append(
                {
                    "target_question_id": row["question_id"],
                    "donor_question_id": donor["question_id"],
                    "target_question": row["question"],
                    "donor_question": donor["question"],
                    "answer": row["answer"],
                    "clean_steps": list(example.steps),
                    "v16_same_problem_steps": list(same_problem_targets),
                    "v17_unrelated_steps": list(targets),
                    "token_counts": token_counts,
                }
            )
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "source_schedule_sha256": schedule["schedule_sha256"],
        "examples": len(schedule["train_entries"]),
        "treatment_examples": len(active_ids),
        "treatment_question_ids": sorted(active_ids),
        "same_treatment_ids_as_v16": True,
        "mapping_seed": int(config["unrelated_mapping_seed"]),
        "mapping_method": mapping["mapping_method"],
        "mapping_sha256": canonical_hash(mapping),
        "unique_donor_count": len(set(donor_ids)),
        "all_donors_different_from_targets": all(
            pair["target_question_id"] != pair["donor_question_id"]
            for pair in mapping["pairs"]
        ),
        "all_first_four_values_differ_from_target_clean": not any(
            "prefix_value_collision" in failure for failure in failures
        ),
        "all_final_results_equal_target_answer": not any(
            "final_answer_not_recovered" in failure for failure in failures
        ),
        "mean_target_donor_question_word_jaccard": mean(question_jaccards),
        "token_count_summary": {
            "clean_mean": mean(clean_tokens),
            "v16_same_problem_mean": mean(v16_tokens),
            "v17_unrelated_mean": mean(unrelated_tokens),
            "v17_to_v16_median_ratio": median(
                unrelated / same
                for unrelated, same in zip(unrelated_tokens, v16_tokens, strict=True)
            ),
        },
        "question_and_answer_fields_untouched": True,
        "auxiliary_weight": 1.0,
        "failures": failures,
        "samples": samples,
        "donor_pairs": mapping["pairs"],
        "official_test_opened": False,
    }
    result["audit_sha256"] = canonical_hash(result)
    atomic_json((ROOT / config["data_audit_path"]).resolve(), result)
    if (
        result["status"] != "PASS"
        or len(active_ids) != int(config["treatment_examples"])
        or len(set(donor_ids)) != int(config["treatment_examples"])
    ):
        raise RuntimeError("Unrelated accidental-correct 50% data audit failed")
    return result


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _reference_directory(config: dict[str, Any], seed: int, arm: str) -> Path:
    return (ROOT / config["reference_output_root"] / "eval" / f"seed_{seed}" / arm).resolve()


def _reference_audit(config: dict[str, Any]) -> dict[str, Any]:
    if sha256_file((ROOT / config["confirm_dataset_path"]).resolve()) != config[
        "confirm_dataset_sha256"
    ]:
        raise ValueError("Confirmation dataset SHA-256 mismatch")
    rows: dict[str, Any] = {}
    for seed in config["seeds"]:
        seed_rows: dict[str, Any] = {}
        ground_truth: list[str] | None = None
        for label, arm, expected_key in (
            ("clean", config["clean_reference_arm"], "clean_accuracy_by_seed"),
            ("severe", config["severe_reference_arm"], "severe_accuracy_by_seed"),
        ):
            directory = _reference_directory(config, seed, arm)
            metrics_path = directory / "metrics.json"
            predictions_path = directory / "predictions.jsonl"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            predictions = _read_predictions(predictions_path)
            current_truth = [row["ground_truth"] for row in predictions]
            if ground_truth is None:
                ground_truth = current_truth
            if (
                current_truth != ground_truth
                or len(predictions) != int(config["confirm_examples"])
                or metrics.get("schedule_sha256") != config["source_schedule_sha256"]
                or metrics.get("confirm_dataset_sha256") != config["confirm_dataset_sha256"]
                or float(metrics["accuracy"])
                != float(config[expected_key][str(seed)])
            ):
                raise ValueError(f"Reference mismatch for seed={seed}, arm={arm}")
            seed_rows[label] = {
                "arm": arm,
                "accuracy": metrics["accuracy"],
                "metrics_sha256": sha256_file(metrics_path),
                "predictions_sha256": sha256_file(predictions_path),
            }
        rows[str(seed)] = seed_rows
    result = {
        "schema_version": 1,
        "status": "PASS",
        "per_seed": rows,
        "source_schedule_sha256": config["source_schedule_sha256"],
        "confirm_dataset_sha256": config["confirm_dataset_sha256"],
        "official_test_opened": False,
    }
    atomic_json((ROOT / config["reference_audit_path"]).resolve(), result)
    return result


def _unrelated_reference_audit(config: dict[str, Any]) -> dict[str, Any]:
    result = _reference_audit(config)
    accidental_root = (ROOT / config["accidental_reference_output_root"]).resolve()
    for seed in config["seeds"]:
        directory = accidental_root / "eval" / f"seed_{seed}" / ACCIDENTAL_CORRECT_50_ARM
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.jsonl"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = _read_predictions(predictions_path)
        clean_directory = _reference_directory(config, seed, config["clean_reference_arm"])
        clean_predictions = _read_predictions(clean_directory / "predictions.jsonl")
        if (
            len(predictions) != int(config["confirm_examples"])
            or [row["ground_truth"] for row in predictions]
            != [row["ground_truth"] for row in clean_predictions]
            or metrics.get("schedule_sha256") != config["source_schedule_sha256"]
            or metrics.get("confirm_dataset_sha256") != config["confirm_dataset_sha256"]
            or float(metrics["accuracy"])
            != float(config["accidental_reference_accuracy_by_seed"][str(seed)])
        ):
            raise ValueError(f"v16 accidental reference mismatch for seed={seed}")
        result["per_seed"][str(seed)]["v16_same_problem_accidental"] = {
            "arm": ACCIDENTAL_CORRECT_50_ARM,
            "accuracy": metrics["accuracy"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
    result["accidental_reference_output_root"] = config["accidental_reference_output_root"]
    atomic_json((ROOT / config["reference_audit_path"]).resolve(), result)
    return result


def _training_complete(config: dict[str, Any], seed: int, arm: str, audit_hash: str) -> bool:
    metrics_path = training_directory(ROOT, config, seed, arm) / "metrics.json"
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if not metrics_path.exists() or not checkpoint.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == arm
            and metrics.get("schedule_sha256") == config["source_schedule_sha256"]
            and metrics.get("treatment_audit_sha256") == audit_hash
            and sha256_file(checkpoint) == metrics.get("checkpoint_sha256")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _evaluation_directory(config: dict[str, Any], seed: int, arm: str) -> Path:
    return (ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm).resolve()


def _evaluation_complete(config: dict[str, Any], seed: int, arm: str, audit_hash: str) -> bool:
    directory = _evaluation_directory(config, seed, arm)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == arm
            and metrics.get("treatment_audit_sha256") == audit_hash
            and len(_read_predictions(predictions_path)) == int(config["confirm_examples"])
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _train(config: dict[str, Any], seed: int, arm: str, audit_hash: str) -> None:
    _preflight(config, f"TRAIN:{seed}:{arm}")
    metrics = run_full_conflict_training(config, arm=arm, seed=seed, project_root=ROOT)
    metrics.update(
        {
            "coverage": 0.5,
            "treatment_audit_sha256": audit_hash,
            "question_answer_fields_untouched": True,
        }
    )
    atomic_json(training_directory(ROOT, config, seed, arm) / "metrics.json", metrics)
    if metrics["status"] != "PASS":
        raise RuntimeError(f"Training failed its memory gate for seed={seed}, arm={arm}")


@torch.inference_mode()
def _evaluate(config: dict[str, Any], seed: int, arm: str, audit_hash: str) -> None:
    _preflight(config, f"EVAL:{seed}:{arm}")
    train_metrics = json.loads(
        (training_directory(ROOT, config, seed, arm) / "metrics.json").read_text(encoding="utf-8")
    )
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if sha256_file(checkpoint) != train_metrics["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint SHA-256 mismatch")
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(ROOT / config["official_source_dir"]).resolve(),
        base_model_dir=(ROOT / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    directory = _evaluation_directory(config, seed, arm)
    predictions_path = directory / "predictions.jsonl"
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=(ROOT / config["confirm_dataset_path"]).resolve(),
        output_dir=directory,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=predictions_path.exists(),
        flush_every=int(config["flush_every"]),
    )
    metrics.update(
        {
            "status": "PASS",
            "run_id": f"{config.get('run_id', 'HBCO-V15')}-EVAL-{seed}-{arm}",
            "arm": arm,
            "seed": seed,
            "training_checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": config["source_schedule_sha256"],
            "confirm_dataset_sha256": config["confirm_dataset_sha256"],
            "treatment_audit_sha256": audit_hash,
            "official_test_opened": False,
        }
    )
    atomic_json(directory / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _comparison(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if [row["ground_truth"] for row in left_rows] != [row["ground_truth"] for row in right_rows]:
        raise ValueError("Paired ground truth mismatch")
    left = [bool(row["correct"]) for row in left_rows]
    right = [bool(row["correct"]) for row in right_rows]
    return {
        "mcnemar": exact_mcnemar(left, right),
        "paired_bootstrap": paired_bootstrap_damage_pp(
            left, right, samples=samples, seed=seed
        ),
    }


def _seed_level_ci(values: list[float]) -> tuple[float, float]:
    center = mean(values)
    half_width = 4.302652729911275 * stdev(values) / len(values) ** 0.5
    return center - half_width, center + half_width


def _analyze(config: dict[str, Any]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        directories = {
            "clean": _reference_directory(config, seed, config["clean_reference_arm"]),
            "redundant50": _evaluation_directory(config, seed, REDUNDANT_STEPS_50_ARM),
            "severe50": _reference_directory(config, seed, config["severe_reference_arm"]),
            "reverse50": _evaluation_directory(config, seed, STEP_ORDER_REVERSAL_50_ARM),
        }
        metrics = {
            name: json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            for name, directory in directories.items()
        }
        predictions = {
            name: _read_predictions(directory / "predictions.jsonl")
            for name, directory in directories.items()
        }
        accuracy = {name: float(row["accuracy"]) for name, row in metrics.items()}
        comparisons = {
            "clean_vs_redundant": _comparison(
                predictions["clean"], predictions["redundant50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 1,
            ),
            "clean_vs_severe": _comparison(
                predictions["clean"], predictions["severe50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 2,
            ),
            "clean_vs_reverse": _comparison(
                predictions["clean"], predictions["reverse50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 3,
            ),
            "severe_vs_reverse": _comparison(
                predictions["severe50"], predictions["reverse50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 4,
            ),
        }
        per_seed.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "clean_minus_redundant_pp": 100 * (accuracy["clean"] - accuracy["redundant50"]),
                "clean_minus_severe_pp": 100 * (accuracy["clean"] - accuracy["severe50"]),
                "clean_minus_reverse_pp": 100 * (accuracy["clean"] - accuracy["reverse50"]),
                "severe_minus_reverse_pp": 100 * (accuracy["severe50"] - accuracy["reverse50"]),
                "comparisons": comparisons,
            }
        )
    effects = {
        name: [float(row[name]) for row in per_seed]
        for name in (
            "clean_minus_redundant_pp",
            "clean_minus_severe_pp",
            "clean_minus_reverse_pp",
            "severe_minus_reverse_pp",
        )
    }
    summaries = {}
    for name, values in effects.items():
        low, high = _seed_level_ci(values)
        summaries[name] = {
            "values": values,
            "mean": mean(values),
            "population_sd": pstdev(values),
            "seed_level_t_ci95_low": low,
            "seed_level_t_ci95_high": high,
        }
    accuracy_means = {
        arm: mean(row["accuracy"][arm] for row in per_seed)
        for arm in ("clean", "redundant50", "severe50", "reverse50")
    }
    criteria = {
        "redundancy_neutral": abs(summaries["clean_minus_redundant_pp"]["mean"])
        <= float(config["redundancy_neutrality_band_pp"]),
        "severe_content_harm_confirmed": summaries["clean_minus_severe_pp"]["mean"]
        >= float(config["min_harm_pp"])
        and all(value > 0 for value in effects["clean_minus_severe_pp"]),
        "order_harm_confirmed": summaries["clean_minus_reverse_pp"]["mean"]
        >= float(config["min_harm_pp"])
        and all(value > 0 for value in effects["clean_minus_reverse_pp"]),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "accuracy_means": accuracy_means,
        "comparisons": summaries,
        "criteria": criteria,
        "per_seed": per_seed,
        "scope": "matched 50% treatment on the original 79.82% Clean baseline",
        "official_test_opened": False,
    }
    atomic_json((ROOT / config["analysis_path"]).resolve(), result)
    return result


def _analyze_accidental(config: dict[str, Any]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        directories = {
            "clean": _reference_directory(config, seed, config["clean_reference_arm"]),
            "severe50": _reference_directory(config, seed, config["severe_reference_arm"]),
            "accidental50": _evaluation_directory(config, seed, ACCIDENTAL_CORRECT_50_ARM),
        }
        metrics = {
            name: json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            for name, directory in directories.items()
        }
        predictions = {
            name: _read_predictions(directory / "predictions.jsonl")
            for name, directory in directories.items()
        }
        accuracy = {name: float(row["accuracy"]) for name, row in metrics.items()}
        comparisons = {
            "clean_vs_accidental": _comparison(
                predictions["clean"], predictions["accidental50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 21,
            ),
            "clean_vs_severe": _comparison(
                predictions["clean"], predictions["severe50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 22,
            ),
            "accidental_vs_severe": _comparison(
                predictions["accidental50"], predictions["severe50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 23,
            ),
        }
        per_seed.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "clean_minus_accidental_pp": 100
                * (accuracy["clean"] - accuracy["accidental50"]),
                "clean_minus_severe_pp": 100
                * (accuracy["clean"] - accuracy["severe50"]),
                "accidental_minus_severe_pp": 100
                * (accuracy["accidental50"] - accuracy["severe50"]),
                "comparisons": comparisons,
            }
        )
    effects = {
        name: [float(row[name]) for row in per_seed]
        for name in (
            "clean_minus_accidental_pp",
            "clean_minus_severe_pp",
            "accidental_minus_severe_pp",
        )
    }
    summaries: dict[str, Any] = {}
    for name, values in effects.items():
        low, high = _seed_level_ci(values)
        summaries[name] = {
            "values": values,
            "mean": mean(values),
            "population_sd": pstdev(values),
            "seed_level_t_ci95_low": low,
            "seed_level_t_ci95_high": high,
        }
    accuracy_means = {
        arm: mean(row["accuracy"][arm] for row in per_seed)
        for arm in ("clean", "severe50", "accidental50")
    }
    harm_values = effects["clean_minus_accidental_pp"]
    rescue_values = effects["accidental_minus_severe_pp"]
    criteria = {
        "wrong_chain_harm_persists": summaries["clean_minus_accidental_pp"]["mean"]
        >= float(config["min_harm_pp"])
        and all(value > 0 for value in harm_values),
        "correct_tail_rescue_confirmed": summaries["accidental_minus_severe_pp"]["mean"]
        >= float(config["min_rescue_pp"])
        and all(value > 0 for value in rescue_values),
        "correct_tail_restores_near_clean": abs(
            summaries["clean_minus_accidental_pp"]["mean"]
        )
        <= float(config["near_clean_band_pp"]),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "accuracy_means": accuracy_means,
        "comparisons": summaries,
        "criteria": criteria,
        "per_seed": per_seed,
        "scope": "matched 50% accidental-correct treatment on the original 79.82% Clean baseline",
        "official_test_opened": False,
    }
    atomic_json((ROOT / config["analysis_path"]).resolve(), result)
    return result


def _analyze_unrelated(config: dict[str, Any]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    accidental_root = (ROOT / config["accidental_reference_output_root"]).resolve()
    for seed in config["seeds"]:
        directories = {
            "clean": _reference_directory(config, seed, config["clean_reference_arm"]),
            "severe50": _reference_directory(config, seed, config["severe_reference_arm"]),
            "same_problem50": accidental_root
            / "eval" / f"seed_{seed}" / ACCIDENTAL_CORRECT_50_ARM,
            "unrelated50": _evaluation_directory(
                config, seed, UNRELATED_ACCIDENTAL_CORRECT_50_ARM
            ),
        }
        metrics = {
            name: json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
            for name, directory in directories.items()
        }
        predictions = {
            name: _read_predictions(directory / "predictions.jsonl")
            for name, directory in directories.items()
        }
        accuracy = {name: float(row["accuracy"]) for name, row in metrics.items()}
        comparisons = {
            "clean_vs_unrelated": _comparison(
                predictions["clean"], predictions["unrelated50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 31,
            ),
            "same_problem_vs_unrelated": _comparison(
                predictions["same_problem50"], predictions["unrelated50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 32,
            ),
            "severe_vs_unrelated": _comparison(
                predictions["severe50"], predictions["unrelated50"],
                samples=int(config["bootstrap_samples"]), seed=seed + 33,
            ),
        }
        per_seed.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "clean_minus_unrelated_pp": 100
                * (accuracy["clean"] - accuracy["unrelated50"]),
                "same_problem_minus_unrelated_pp": 100
                * (accuracy["same_problem50"] - accuracy["unrelated50"]),
                "severe_minus_unrelated_pp": 100
                * (accuracy["severe50"] - accuracy["unrelated50"]),
                "comparisons": comparisons,
            }
        )
    effects = {
        name: [float(row[name]) for row in per_seed]
        for name in (
            "clean_minus_unrelated_pp",
            "same_problem_minus_unrelated_pp",
            "severe_minus_unrelated_pp",
        )
    }
    summaries: dict[str, Any] = {}
    for name, values in effects.items():
        low, high = _seed_level_ci(values)
        summaries[name] = {
            "values": values,
            "mean": mean(values),
            "population_sd": pstdev(values),
            "seed_level_t_ci95_low": low,
            "seed_level_t_ci95_high": high,
        }
    accuracy_means = {
        arm: mean(row["accuracy"][arm] for row in per_seed)
        for arm in ("clean", "severe50", "same_problem50", "unrelated50")
    }
    harm_values = effects["clean_minus_unrelated_pp"]
    incremental_values = effects["same_problem_minus_unrelated_pp"]
    criteria = {
        "unrelated_chain_harm_confirmed": summaries["clean_minus_unrelated_pp"]["mean"]
        >= float(config["min_harm_pp"])
        and all(value > 0 for value in harm_values),
        "unrelated_more_harmful_than_same_problem": summaries[
            "same_problem_minus_unrelated_pp"
        ]["mean"]
        >= float(config["min_incremental_harm_pp"])
        and all(value > 0 for value in incremental_values),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "accuracy_means": accuracy_means,
        "comparisons": summaries,
        "criteria": criteria,
        "per_seed": per_seed,
        "scope": "matched 50% cross-question unrelated accidental-correct treatment on the original 79.82% Clean baseline",
        "official_test_opened": False,
    }
    atomic_json((ROOT / config["analysis_path"]).resolve(), result)
    return result


def _write_report(config: dict[str, Any], analysis: dict[str, Any]) -> Path:
    lines = [
        "# SIM-CoT 高基线内容错误与顺序错位 v15 报告",
        "",
        "四组共享冻结的 512 条训练调度、1024 条确认集、初始检查点、三种子和 64-update 配置。",
        "严重内容错误、冗余和步骤逆序使用完全相同的 256/512 个处理样本；答案标签始终正确。",
        "",
        "| Seed | Clean | 冗余50 | 严重错误50 | 逆序50 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_seed"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['seed']} | {acc['clean']:.2%} | {acc['redundant50']:.2%} | "
            f"{acc['severe50']:.2%} | {acc['reverse50']:.2%} |"
        )
    means = analysis["accuracy_means"]
    lines.append(
        f"| **均值** | **{means['clean']:.2%}** | **{means['redundant50']:.2%}** | "
        f"**{means['severe50']:.2%}** | **{means['reverse50']:.2%}** |"
    )
    comparisons = analysis["comparisons"]
    lines.extend(
        [
            "",
            f"- Clean−冗余50：{comparisons['clean_minus_redundant_pp']['mean']:.2f} pp",
            f"- Clean−严重错误50：{comparisons['clean_minus_severe_pp']['mean']:.2f} pp",
            f"- Clean−逆序50：{comparisons['clean_minus_reverse_pp']['mean']:.2f} pp",
            f"- 严重错误50−逆序50：{comparisons['severe_minus_reverse_pp']['mean']:.2f} pp（正值表示逆序更有害）",
            "",
            f"- 冗余中性：{'通过' if analysis['criteria']['redundancy_neutral'] else '未通过'}",
            f"- 严重内容错误伤害：{'确认' if analysis['criteria']['severe_content_harm_confirmed'] else '未确认'}",
            f"- 顺序错位伤害：{'确认' if analysis['criteria']['order_harm_confirmed'] else '未确认'}",
            "",
            "注意：评估集是旧实验冻结的 1024 条确认集，不是 GSM8K 官方测试集；错误与冗余均为确定性受控合成，不是自然教师噪声。",
        ]
    )
    path = (ROOT / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_accidental_report(config: dict[str, Any], analysis: dict[str, Any]) -> Path:
    lines = [
        "# SIM-CoT 高基线错误链偶然答对 v16 报告",
        "",
        "三组共享冻结的512条训练调度、1024条确认集、初始检查点、三种子和64-update配置。",
        "处理覆盖固定为同一256/512题；新组前四步逐字复用严重错误链，第五步从错误状态用无题意依据的精确抵消量落到官方答案。",
        "问题与最终答案标签始终不变。",
        "",
        "| Seed | Clean | 全错链50 | 错误链偶然答对50 |",
        "|---:|---:|---:|---:|",
    ]
    for row in analysis["per_seed"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['seed']} | {acc['clean']:.2%} | {acc['severe50']:.2%} | "
            f"{acc['accidental50']:.2%} |"
        )
    means = analysis["accuracy_means"]
    lines.append(
        f"| **均值** | **{means['clean']:.2%}** | **{means['severe50']:.2%}** | "
        f"**{means['accidental50']:.2%}** |"
    )
    comparisons = analysis["comparisons"]
    criteria = analysis["criteria"]
    lines.extend(
        [
            "",
            f"- Clean−错误链偶然答对50：{comparisons['clean_minus_accidental_pp']['mean']:.2f} pp",
            f"- Clean−全错链50：{comparisons['clean_minus_severe_pp']['mean']:.2f} pp",
            f"- 错误链偶然答对50−全错链50：{comparisons['accidental_minus_severe_pp']['mean']:.2f} pp",
            "",
            f"- 错误链伤害仍存在：{'确认' if criteria['wrong_chain_harm_persists'] else '未确认'}",
            f"- 正确尾步产生恢复：{'确认' if criteria['correct_tail_rescue_confirmed'] else '未确认'}",
            f"- 恢复到Clean±{float(config['near_clean_band_pp']):.1f}pp：{'是' if criteria['correct_tail_restores_near_clean'] else '否'}",
            "",
            "注意：这是Codex编写的确定性受控半合成错误抵消，不是自然教师噪声；评估集是冻结的1024条确认集，不是GSM8K官方测试集。",
        ]
    )
    path = (ROOT / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_unrelated_report(config: dict[str, Any], analysis: dict[str, Any]) -> Path:
    lines = [
        "# SIM-CoT 高基线完全无关错误链偶然答对 v17 报告",
        "",
        "四组共享冻结的512条训练调度、1024条确认集、初始检查点、三种子和64-update配置。",
        "v17在同一256题上使用另一题的一一错配错误链前四步，末步从供体错误状态抵消到目标官方答案。",
        "问题、答案标签、处理覆盖率和辅助权重均保持不变。",
        "",
        "| Seed | Clean | 全错链50 | v16同题错误链偶然答对50 | v17无关错误链偶然答对50 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_seed"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['seed']} | {acc['clean']:.2%} | {acc['severe50']:.2%} | "
            f"{acc['same_problem50']:.2%} | {acc['unrelated50']:.2%} |"
        )
    means = analysis["accuracy_means"]
    lines.append(
        f"| **均值** | **{means['clean']:.2%}** | **{means['severe50']:.2%}** | "
        f"**{means['same_problem50']:.2%}** | **{means['unrelated50']:.2%}** |"
    )
    comparisons = analysis["comparisons"]
    criteria = analysis["criteria"]
    lines.extend(
        [
            "",
            f"- Clean−v17无关链：{comparisons['clean_minus_unrelated_pp']['mean']:.2f} pp",
            f"- v16同题链−v17无关链：{comparisons['same_problem_minus_unrelated_pp']['mean']:.2f} pp",
            f"- 全错链−v17无关链：{comparisons['severe_minus_unrelated_pp']['mean']:.2f} pp",
            "",
            f"- 无关链伤害：{'确认' if criteria['unrelated_chain_harm_confirmed'] else '未确认'}",
            f"- 无关链比同题错误链额外伤害≥{float(config['min_incremental_harm_pp']):.1f}pp：{'确认' if criteria['unrelated_more_harmful_than_same_problem'] else '未确认'}",
            "",
            "注意：这是Codex构造的跨题一一错配受控半合成压力测试，不是自然教师噪声；评估集是冻结1024条确认集，不是GSM8K官方测试集。",
        ]
    )
    path = (ROOT / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run matched high-baseline content/order controls")
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/high_baseline_content_order_v15.json",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    arms = tuple(config["arms"])
    if arms not in (REGISTERED_ARMS, ACCIDENTAL_REGISTERED_ARMS, UNRELATED_REGISTERED_ARMS):
        raise ValueError("Configuration arms do not match a registered matched-baseline design")
    design = (
        "unrelated" if arms == UNRELATED_REGISTERED_ARMS
        else "accidental" if arms == ACCIDENTAL_REGISTERED_ARMS
        else "content_order"
    )
    try:
        _save_state(config, status="RUNNING", phase="AUDIT", current="data_and_references")
        data_audit = (
            _unrelated_data_audit(config) if design == "unrelated"
            else _accidental_data_audit(config) if design == "accidental"
            else _data_audit(config)
        )
        if design == "unrelated":
            _unrelated_reference_audit(config)
        else:
            _reference_audit(config)
        audit_hash = data_audit["audit_sha256"]
        for seed in config["seeds"]:
            for arm in config["arms"]:
                key = f"{seed}:{arm}"
                if not _training_complete(config, seed, arm, audit_hash):
                    _save_state(config, status="RUNNING", phase="TRAIN", current=key)
                    _train(config, seed, arm, audit_hash)
                else:
                    print(f"skip completed training {key}", flush=True)
                state = _save_state(config)
                completed = list(state.get("completed_training", []))
                if key not in completed:
                    completed.append(key)
                    _save_state(config, completed_training=completed)

                if not _evaluation_complete(config, seed, arm, audit_hash):
                    _save_state(config, status="RUNNING", phase="EVAL", current=key)
                    _evaluate(config, seed, arm, audit_hash)
                else:
                    print(f"skip completed evaluation {key}", flush=True)
                state = _save_state(config)
                completed = list(state.get("completed_evaluation", []))
                if key not in completed:
                    completed.append(key)
                    _save_state(config, completed_evaluation=completed)

        analysis_label = {
            "unrelated": "unrelated_accidental_correct_paired_analysis",
            "accidental": "accidental_correct_paired_analysis",
            "content_order": "four_arm_paired_analysis",
        }[design]
        _save_state(config, status="RUNNING", phase="ANALYZE", current=analysis_label)
        analysis = (
            _analyze_unrelated(config) if design == "unrelated"
            else _analyze_accidental(config) if design == "accidental"
            else _analyze(config)
        )
        report = (
            _write_unrelated_report(config, analysis) if design == "unrelated"
            else _write_accidental_report(config, analysis) if design == "accidental"
            else _write_report(config, analysis)
        )
        _save_state(
            config,
            status="PASS",
            phase="COMPLETE",
            current=None,
            report_path=str(report),
            completed_unix=time.time(),
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        _save_state(
            config,
            status="FAIL",
            phase="STOPPED",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
