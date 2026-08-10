from __future__ import annotations

from pathlib import Path
from typing import Any
import gc
import json
import math

import numpy as np
import torch

from .causal_experiment import (
    CAUSAL_ARMS,
    COVERAGES,
    _arm_dir_name,
    _canonical_hash,
    _output_root,
    verify_causal_schedule,
)
from .m1_training import atomic_json, sha256_file
from .official_adapter import evaluate_checkpoint, iter_icot_examples, load_official_model
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


def _read_schedule(config: dict[str, Any], root: Path) -> dict[str, Any]:
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    verify_causal_schedule(schedule)
    return schedule


def _training_metrics_path(
    config: dict[str, Any],
    root: Path,
    *,
    split: str,
    arm: str,
    coverage: int,
) -> Path:
    return (
        _output_root(root, config)
        / split
        / _arm_dir_name(split, arm, coverage)
        / "metrics.json"
    )


def _validated_checkpoint(
    config: dict[str, Any],
    root: Path,
    *,
    split: str,
    arm: str,
    coverage: int,
) -> tuple[Path, dict[str, Any]]:
    metrics_path = _training_metrics_path(
        config, root, split=split, arm=arm, coverage=coverage
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("status") != "PASS"
        or metrics.get("split") != split
        or metrics.get("arm") != arm
        or metrics.get("coverage") != coverage
    ):
        raise ValueError(f"Training metrics are not valid for {split}/{arm}/{coverage}")
    checkpoint = Path(metrics["checkpoint_path"])
    if sha256_file(checkpoint) != metrics["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {split}/{arm}/{coverage}")
    return checkpoint, metrics


@torch.inference_mode()
def clean_dev_nll(
    model,
    tokenizer,
    token_ids: dict[str, int],
    config: dict[str, Any],
    *,
    dev_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    examples = list(iter_icot_examples(dev_path))
    if len(examples) != config["dev_examples"] or any(len(example.steps) < 5 for example in examples):
        raise ValueError("Frozen causal dev set is incomplete or has fewer than five steps")
    answer_loss_sum = 0.0
    step_sequence_sum = 0.0
    step_sequences = 0
    step_token_loss_sum = 0.0
    step_tokens = 0
    model.base_causallm.eval()
    model.expainable_llm.eval()
    for number, example in enumerate(examples, start=1):
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=config["latent_stage"],
            c_thought=config["c_thought"],
        )
        batch = tensorize_smoke_example(encoded, device=device)
        targets = tokenize_step_targets(tokenizer, example.steps[:5])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = grouped_auxiliary_loss(
                model,
                batch,
                targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=config["c_thought"],
            )
        answer_loss_sum += float(losses["answer_loss"].float().item())
        values = losses["step_losses"].float().tolist()
        counts = losses["token_counts"]
        step_sequence_sum += sum(values)
        step_sequences += len(values)
        step_token_loss_sum += sum(values)
        step_tokens += sum(counts)
        if number == 1 or number % config["nll_log_every"] == 0 or number == len(examples):
            print(f"dev NLL: {number}/{len(examples)}", flush=True)
    return {
        "examples": len(examples),
        "answer_nll": answer_loss_sum / len(examples),
        "mean_step_sequence_nll": step_sequence_sum / step_sequences,
        "step_token_nll": step_token_loss_sum / step_tokens,
        "step_tokens": step_tokens,
        "ground_truth_source": "frozen official training rows; clean steps and answer fields",
    }


def evaluate_pilot(
    config: dict[str, Any],
    *,
    arm: str,
    coverage: int,
    project_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    if arm not in {"clean", "noisy_equal"}:
        raise ValueError("Pilot evaluation only accepts clean or noisy_equal")
    if coverage not in COVERAGES:
        raise ValueError(f"Coverage must be one of {COVERAGES}")
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    checkpoint, train_metrics = _validated_checkpoint(
        config, root, split="pilot", arm=arm, coverage=coverage
    )
    dev_path = Path(schedule["dev_dataset_path"])
    if sha256_file(dev_path) != schedule["dev_dataset_sha256"]:
        raise ValueError("Frozen causal dev dataset SHA-256 mismatch")
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    dev_nll = clean_dev_nll(
        model,
        tokenizer,
        token_ids,
        config,
        dev_path=dev_path,
        device=device,
    )
    name = _arm_dir_name("pilot", arm, coverage)
    output_dir = _output_root(root, config) / "pilot_eval" / name
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=dev_path,
        output_dir=output_dir,
        device=device,
        latent_tokens=config["latent_stage"] * config["c_thought"],
        max_new_tokens=config["max_new_tokens"],
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=resume,
        flush_every=config["flush_every"],
    )
    metrics.update(
        {
            "run_id": config["calibration_run_id"],
            "phase": "pilot_dev",
            "arm": arm,
            "coverage": coverage,
            "training_run_id": train_metrics["run_id"],
            "checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "clean_dev_nll": dev_nll,
            "official_test_opened": False,
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def calibrate_coverage(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    clean_path = _output_root(root, config) / "pilot_eval" / "clean" / "metrics.json"
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if clean.get("schedule_sha256") != schedule["schedule_sha256"]:
        raise ValueError("Clean pilot evaluation does not match the frozen schedule")
    rows: list[dict[str, Any]] = []
    chosen: int | None = None
    for coverage in COVERAGES:
        path = (
            _output_root(root, config)
            / "pilot_eval"
            / f"noisy_equal_{coverage}"
            / "metrics.json"
        )
        noisy = json.loads(path.read_text(encoding="utf-8"))
        if noisy.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise ValueError(f"Noisy pilot {coverage} does not match the frozen schedule")
        damage = float(clean["accuracy"]) - float(noisy["accuracy"])
        qualifies = damage >= config["minimum_damage"]
        if chosen is None and qualifies:
            chosen = coverage
        rows.append(
            {
                "coverage": coverage,
                "clean_accuracy": float(clean["accuracy"]),
                "noisy_equal_accuracy": float(noisy["accuracy"]),
                "damage": damage,
                "damage_percentage_points": damage * 100,
                "qualifies": qualifies,
                "clean_dev_nll": clean["clean_dev_nll"],
                "noisy_dev_nll": noisy["clean_dev_nll"],
            }
        )
    passed = chosen is not None
    result = {
        "schema_version": 1,
        "run_id": config["calibration_run_id"],
        "status": "PASS" if passed else "NO_CAUSALLY_HARMFUL_TREATMENT",
        "gate_passed": passed,
        "chosen_coverage": chosen,
        "selection_rule": "minimum coverage with clean-minus-noisy dev EM >= threshold",
        "minimum_damage": config["minimum_damage"],
        "schedule_sha256": schedule["schedule_sha256"],
        "dev_dataset_sha256": schedule["dev_dataset_sha256"],
        "ground_truth_source": "frozen official-row answer fields",
        "official_test_opened": False,
        "coverage_results": rows,
    }
    unhashed = dict(result)
    result["gate_sha256"] = _canonical_hash(unhashed)
    atomic_json((root / config["calibration_gate_path"]).resolve(), result)
    return result


def _read_gate(config: dict[str, Any], root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    gate = json.loads((root / config["calibration_gate_path"]).read_text(encoding="utf-8"))
    expected = gate.get("gate_sha256")
    unhashed = dict(gate)
    unhashed.pop("gate_sha256", None)
    if expected != _canonical_hash(unhashed):
        raise ValueError("Calibration gate SHA-256 mismatch")
    if not gate.get("gate_passed") or gate.get("chosen_coverage") not in COVERAGES:
        raise ValueError("Official test remains locked because calibration did not pass")
    if gate.get("schedule_sha256") != schedule["schedule_sha256"]:
        raise ValueError("Calibration gate and frozen schedule differ")
    return gate


def evaluate_formal(
    config: dict[str, Any],
    *,
    arm: str,
    project_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    if arm not in CAUSAL_ARMS:
        raise ValueError(f"Unknown formal arm: {arm}")
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    gate = _read_gate(config, root, schedule)
    coverage = int(gate["chosen_coverage"])
    test_path = (root / config["test_dataset_path"]).resolve()
    if sha256_file(test_path) != config["test_dataset_sha256"]:
        raise ValueError("Official test dataset SHA-256 mismatch")
    checkpoint, train_metrics = _validated_checkpoint(
        config, root, split="formal", arm=arm, coverage=coverage
    )
    dev_path = Path(schedule["dev_dataset_path"])
    if sha256_file(dev_path) != schedule["dev_dataset_sha256"]:
        raise ValueError("Frozen causal dev dataset SHA-256 mismatch")
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    dev_nll = clean_dev_nll(
        model,
        tokenizer,
        token_ids,
        config,
        dev_path=dev_path,
        device=device,
    )
    output_dir = _output_root(root, config) / "formal_eval" / arm
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=test_path,
        output_dir=output_dir,
        device=device,
        latent_tokens=config["latent_stage"] * config["c_thought"],
        max_new_tokens=config["max_new_tokens"],
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=resume,
        flush_every=config["flush_every"],
    )
    metrics.update(
        {
            "run_id": config["formal_analysis_run_id"],
            "phase": "formal_official_test",
            "arm": arm,
            "coverage": coverage,
            "training_run_id": train_metrics["run_id"],
            "checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "calibration_gate_sha256": gate["gate_sha256"],
            "clean_dev_nll": dev_nll,
            "ground_truth_source": "official test dataset answer field",
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _exact_mcnemar_p(better: int, worse: int) -> float:
    discordant = better + worse
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(better, worse) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _paired_comparison(
    baseline: np.ndarray, candidate: np.ndarray, *, seed: int, samples: int
) -> dict[str, Any]:
    better = int(np.sum((~baseline) & candidate))
    worse = int(np.sum(baseline & (~candidate)))
    difference = candidate.astype(float) - baseline.astype(float)
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(samples, dtype=float)
    for start in range(0, samples, 250):
        size = min(250, samples - start)
        indices = rng.integers(0, len(difference), size=(size, len(difference)))
        bootstrap[start : start + size] = difference[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, (0.025, 0.975))
    return {
        "accuracy_difference": float(difference.mean()),
        "candidate_correct_baseline_wrong": better,
        "candidate_wrong_baseline_correct": worse,
        "mcnemar_exact_two_sided_p": _exact_mcnemar_p(better, worse),
        "paired_bootstrap_95_ci": [float(low), float(high)],
        "bootstrap_samples": samples,
    }


def analyze_formal(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    gate = _read_gate(config, root, schedule)
    rows = {
        arm: _read_predictions(
            _output_root(root, config) / "formal_eval" / arm / "predictions.jsonl"
        )
        for arm in CAUSAL_ARMS
    }
    expected = config["test_examples"]
    if any(len(value) != expected for value in rows.values()):
        raise ValueError("Every formal arm must have complete official-test predictions")
    reference = [(row["idx"], row["ground_truth"]) for row in rows["clean"]]
    for arm in CAUSAL_ARMS[1:]:
        if [(row["idx"], row["ground_truth"]) for row in rows[arm]] != reference:
            raise ValueError(f"Official-test predictions are not paired for {arm}")
    correct = {
        arm: np.asarray([bool(row["correct"]) for row in values], dtype=bool)
        for arm, values in rows.items()
    }
    accuracy = {arm: float(values.mean()) for arm, values in correct.items()}
    damage = accuracy["clean"] - accuracy["noisy_equal"]
    recovery = accuracy["causal_raw"] - accuracy["noisy_equal"]
    ratio = recovery / damage if damage > 0 else None
    conditions = {
        "damage_at_least_2pp": damage >= config["minimum_damage"],
        "raw_recovery_at_least_1pp": recovery >= config["minimum_recovery"],
        "raw_recovery_ratio_at_least_half": (
            ratio is not None and ratio >= config["minimum_recovery_ratio"]
        ),
        "raw_above_uniform": accuracy["causal_raw"] > accuracy["uniform_attenuation"],
        "normalized_above_equal": accuracy["causal_normalized"] > accuracy["noisy_equal"],
    }
    seed = int(config["analysis_seed"])
    samples = int(config["bootstrap_samples"])
    comparisons = {
        "noisy_equal_vs_clean": _paired_comparison(
            correct["clean"], correct["noisy_equal"], seed=seed, samples=samples
        ),
        "causal_raw_vs_noisy_equal": _paired_comparison(
            correct["noisy_equal"], correct["causal_raw"], seed=seed + 1, samples=samples
        ),
        "causal_raw_vs_uniform": _paired_comparison(
            correct["uniform_attenuation"], correct["causal_raw"], seed=seed + 2, samples=samples
        ),
        "causal_raw_vs_pivot_only": _paired_comparison(
            correct["pivot_only"], correct["causal_raw"], seed=seed + 3, samples=samples
        ),
        "causal_normalized_vs_noisy_equal": _paired_comparison(
            correct["noisy_equal"], correct["causal_normalized"], seed=seed + 4, samples=samples
        ),
    }
    all_passed = all(conditions.values())
    result = {
        "schema_version": 1,
        "run_id": config["formal_analysis_run_id"],
        "status": "PASS" if all_passed else "FAIL_PREREGISTERED_MECHANISM_GATE",
        "gate_passed": all_passed,
        "seed": config["seed"],
        "coverage": gate["chosen_coverage"],
        "examples": expected,
        "accuracy": accuracy,
        "correct_counts": {arm: int(values.sum()) for arm, values in correct.items()},
        "noise_damage": damage,
        "causal_raw_recovery": recovery,
        "causal_raw_recovery_ratio": ratio,
        "conditions": conditions,
        "pivot_mechanism_diagnostic": {
            "raw_minus_pivot": accuracy["causal_raw"] - accuracy["pivot_only"],
            "direction_positive": accuracy["causal_raw"] > accuracy["pivot_only"],
            "hard_gate": False,
        },
        "comparisons": comparisons,
        "schedule_sha256": schedule["schedule_sha256"],
        "calibration_gate_sha256": gate["gate_sha256"],
        "ground_truth_source": "official test dataset answer field",
        "claim_boundary": (
            "A single-seed controlled causal-propagation oracle-weighting feasibility test; "
            "it does not validate a learned detector or natural teacher noise."
        ),
        "next_step": (
            "RUN_CONFIRMATORY_SEEDS" if all_passed else "STOP_STAGE_A_AND_REPORT"
        ),
    }
    atomic_json((root / config["formal_analysis_path"]).resolve(), result)
    return result
