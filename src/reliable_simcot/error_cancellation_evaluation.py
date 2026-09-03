from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any
import gc
import json

import torch

from .error_cancellation_experiment import (
    ALL_ARMS,
    STAGE1_ARMS,
    checkpoint_path,
    load_manifest,
    load_schedule,
    training_directory,
)
from .m1_training import atomic_json, sha256_file
from .official_adapter import evaluate_checkpoint, load_official_model
from .self_corrected_evaluation import hierarchical_paired_bootstrap


def _evaluation_dir(root: Path, config: dict[str, Any], seed: int, arm: str) -> Path:
    return (root / config["output_root"] / "eval" / f"seed_{seed}" / arm).resolve()


def evaluate_base(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    output = (root / config["base_eval_dir"]).resolve()
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=root / config["checkpoint_path"],
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    result = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=root / config["gsm_test_path"],
        output_dir=output,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=(output / "predictions.jsonl").exists(),
        flush_every=int(config["flush_every"]),
    )
    result["starting_checkpoint_sha256"] = config["checkpoint_sha256"]
    result["within_memory_limit"] = result["peak_reserved_gb"] <= float(
        config["max_reserved_memory_gb"]
    )
    result["status"] = "PASS" if result["within_memory_limit"] else "FAIL"
    atomic_json(metrics_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def evaluate_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
) -> dict[str, Any]:
    if arm not in ALL_ARMS or seed not in [int(value) for value in config["seeds"]]:
        raise ValueError("Arm or seed is not frozen")
    root = Path(project_root).resolve()
    load_manifest(config, root)
    schedule = load_schedule(config, root)
    train_metrics_path = training_directory(root, config, seed, arm) / "metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint_path(root, config, seed, arm)
    if (
        train_metrics.get("status") != "PASS"
        or train_metrics.get("schedule_sha256") != schedule["schedule_sha256"]
        or sha256_file(checkpoint) != train_metrics.get("checkpoint_sha256")
    ):
        raise ValueError("Training checkpoint is incomplete or stale")
    output = _evaluation_dir(root, config, seed, arm)
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    result = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=root / config["gsm_test_path"],
        output_dir=output,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=(output / "predictions.jsonl").exists(),
        flush_every=int(config["flush_every"]),
    )
    result.update(
        {
            "arm": arm,
            "seed": seed,
            "training_checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "within_memory_limit": result["peak_reserved_gb"]
            <= float(config["max_reserved_memory_gb"]),
        }
    )
    result["status"] = "PASS" if result["within_memory_limit"] else "FAIL"
    atomic_json(metrics_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def clean_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    base = json.loads((root / config["base_eval_dir"] / "metrics.json").read_text(encoding="utf-8"))
    seed_rows = [
        json.loads(
            (_evaluation_dir(root, config, int(seed), "C") / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in config["seeds"]
    ]
    base_accuracy = float(base["accuracy"])
    accuracies = [float(row["accuracy"]) for row in seed_rows]
    mean_pass = mean(accuracies) >= base_accuracy - 0.02
    per_seed_pass = all(value >= base_accuracy - 0.05 for value in accuracies)
    passed = mean_pass and per_seed_pass
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "base_accuracy": base_accuracy,
        "clean_seed_accuracies": dict(zip(map(str, config["seeds"]), accuracies)),
        "clean_mean_accuracy": mean(accuracies),
        "mean_threshold": base_accuracy - 0.02,
        "per_seed_threshold": base_accuracy - 0.05,
        "mean_rule_passed": mean_pass,
        "per_seed_rule_passed": per_seed_pass,
    }
    atomic_json(root / config["clean_gate_path"], result)
    return result


def _prediction_vector(root: Path, config: dict[str, Any], seed: int, arm: str) -> list[bool]:
    path = _evaluation_dir(root, config, seed, arm) / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != int(config["test_examples"]):
        raise ValueError(f"Incomplete predictions for {seed}:{arm}")
    return [bool(row["correct"]) for row in rows]


def _effect(
    root: Path,
    config: dict[str, Any],
    left_arm: str,
    right_arm: str,
    *,
    practical_pp: float,
) -> dict[str, Any]:
    left = [_prediction_vector(root, config, int(seed), left_arm) for seed in config["seeds"]]
    right = [_prediction_vector(root, config, int(seed), right_arm) for seed in config["seeds"]]
    value = hierarchical_paired_bootstrap(
        left,
        right,
        samples=int(config["bootstrap_samples"]),
        seed=int(config["bootstrap_seed"]),
    )
    value["all_seeds_positive"] = all(item > 0 for item in value["per_seed_effect_pp"])
    value["confirmed"] = (
        value["effect_pp"] >= practical_pp
        and value["all_seeds_positive"]
        and value["ci95_low_pp"] > 0
    )
    return value


def analyze_stage1(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    accuracies: dict[str, Any] = {}
    for arm in STAGE1_ARMS:
        values = [
            json.loads(
                (_evaluation_dir(root, config, int(seed), arm) / "metrics.json").read_text(
                    encoding="utf-8"
                )
            )["accuracy"]
            for seed in config["seeds"]
        ]
        accuracies[arm] = {
            "per_seed": dict(zip(map(str, config["seeds"]), values)),
            "mean": mean(values),
            "population_sd": pstdev(values),
        }
    comparisons = {
        "primary_wide50": _effect(
            root,
            config,
            "RW50",
            "EW50",
            practical_pp=float(config["harm_practical_pp"]),
        ),
        "wide25": _effect(root, config, "RW25", "EW25", practical_pp=0.0),
        "local50": _effect(root, config, "RL50", "EL50", practical_pp=0.0),
        "local25": _effect(root, config, "RL25", "EL25", practical_pp=0.0),
    }
    harm_passed = comparisons["primary_wide50"]["confirmed"]
    result = {
        "schema_version": 1,
        "status": "STAGE1_COMPLETE",
        "accuracies": accuracies,
        "comparisons": comparisons,
        "harm_gate_passed": harm_passed,
        "severe_harm": comparisons["primary_wide50"]["effect_pp"]
        >= float(config["severe_harm_pp"]),
        "next_action": "RUN_STAGE2_PRIMARY" if harm_passed else "STOP_NO_DETECTABLE_HARM",
        "disclosure": "Controlled semi-synthetic Codex-authored conflicts, not natural teacher noise.",
    }
    atomic_json(root / config["analysis_path"], result)
    return result


def analyze_stage2(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    analysis_path = root / config["analysis_path"]
    result = json.loads(analysis_path.read_text(encoding="utf-8"))
    error_recovery = _effect(root, config, "EW50-w01", "EW50", practical_pp=0.0)
    redundancy_recovery = _effect(root, config, "RW50-w01", "RW50", practical_pp=0.0)
    selective = _effect(root, config, "EW50-w01", "RW50-w01", practical_pp=0.0)
    # Difference-in-differences estimate; the direct weighted-arm contrast is
    # retained separately because its level also contains the equal-arm harm.
    selective_pp = error_recovery["effect_pp"] - redundancy_recovery["effect_pp"]
    harm = result["comparisons"]["primary_wide50"]["effect_pp"]
    fraction = selective_pp / harm if harm else None
    passed = (
        selective_pp >= float(config["recovery_practical_pp"])
        and fraction is not None
        and fraction >= float(config["recovery_fraction_threshold"])
        and all(
            left - right > 0
            for left, right in zip(
                error_recovery["per_seed_effect_pp"],
                redundancy_recovery["per_seed_effect_pp"],
            )
        )
    )
    result.update(
        {
            "status": "COMPLETE",
            "stage2": {
                "error_recovery": error_recovery,
                "redundancy_mask_recovery": redundancy_recovery,
                "weighted_arm_direct_contrast": selective,
                "selective_recovery_pp": selective_pp,
                "recovery_fraction": fraction,
                "recovery_gate_passed": passed,
            },
            "next_action": "OPTIONAL_EXPANSION" if passed else "STOP_NO_SELECTIVE_RECOVERY",
        }
    )
    atomic_json(analysis_path, result)
    return result
