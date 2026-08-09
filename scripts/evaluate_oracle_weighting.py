from __future__ import annotations

from pathlib import Path
import argparse
import json
import math

import numpy as np
import torch

from reliable_simcot.m1_training import atomic_json, sha256_file
from reliable_simcot.official_adapter import evaluate_checkpoint, iter_icot_examples, load_official_model
from reliable_simcot.oracle_weighting import ARMS, grouped_auxiliary_loss, tokenize_step_targets
from reliable_simcot.single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


DEFAULT_RUN_IDS = {
    "clean": "O010",
    "noisy_equal": "O011",
    "oracle_raw_0.1": "O012",
    "oracle_normalized_0.1": "O013",
}


def _output_root(root: Path, config: dict) -> Path:
    return (root / config.get("output_root", "outputs/reliable_simcot/oracle_weighting")).resolve()


def _run_ids(config: dict) -> dict[str, str]:
    return {**DEFAULT_RUN_IDS, **config.get("run_ids", {})}


def _prediction_dir(root: Path, config: dict, arm: str) -> Path:
    if arm == "clean" and config.get("clean_reference_eval_dir"):
        return (root / config["clean_reference_eval_dir"]).resolve()
    return _output_root(root, config) / "eval" / arm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate oracle step-weighting arms")
    parser.add_argument("mode", choices=("evaluate", "analyze"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def clean_validation_nll(
    model,
    tokenizer,
    token_ids: dict[str, int],
    config: dict,
    *,
    root: Path,
    device: torch.device,
) -> dict:
    selected = []
    for example in iter_icot_examples((root / config["validation_dataset_path"]).resolve()):
        if len(example.steps) >= 5:
            selected.append(example)
        if len(selected) == config["validation_examples"]:
            break
    if len(selected) != config["validation_examples"]:
        raise ValueError("Not enough five-step clean validation examples")

    answer_losses: list[float] = []
    step_sequence_losses: list[float] = []
    step_token_loss_sum = 0.0
    step_token_count = 0
    model.base_causallm.eval()
    model.expainable_llm.eval()
    for example in selected:
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
        answer_losses.append(float(losses["answer_loss"].float().item()))
        values = losses["step_losses"].float().tolist()
        step_sequence_losses.extend(values)
        step_token_loss_sum += sum(values)
        step_token_count += sum(losses["token_counts"])
    return {
        "examples": len(selected),
        "answer_nll": sum(answer_losses) / len(answer_losses),
        "mean_step_sequence_nll": sum(step_sequence_losses) / len(step_sequence_losses),
        "step_token_nll": step_token_loss_sum / step_token_count,
        "step_tokens": step_token_count,
        "ground_truth_source": "official validation steps and answer fields",
    }


def evaluate_arm(config: dict, arm: str, *, root: Path, resume: bool) -> dict:
    if sha256_file(root / config["test_dataset_path"]) != config["test_dataset_sha256"]:
        raise ValueError("Test dataset SHA-256 mismatch")
    if sha256_file(root / config["validation_dataset_path"]) != config["validation_dataset_sha256"]:
        raise ValueError("Validation dataset SHA-256 mismatch")
    run_ids = _run_ids(config)
    train_metrics_path = _output_root(root, config) / arm / "metrics.json"
    train_metrics = json.loads(train_metrics_path.read_text(encoding="utf-8"))
    if train_metrics.get("status") != "PASS" or train_metrics.get("run_id") != run_ids[arm]:
        raise ValueError(f"Formal training did not pass for {arm}")
    checkpoint = Path(train_metrics["checkpoint_path"])
    if sha256_file(checkpoint) != train_metrics["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint SHA-256 mismatch for {arm}")

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
    validation = clean_validation_nll(
        model, tokenizer, token_ids, config, root=root, device=device
    )
    output_dir = _output_root(root, config) / "eval" / arm
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=(root / config["test_dataset_path"]).resolve(),
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
            "run_id": config.get("analysis_run_id", "O020"),
            "arm": arm,
            "training_run_id": run_ids[arm],
            "checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "clean_validation": validation,
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    return metrics


def _read_predictions(path: Path) -> list[dict]:
    rows = []
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
) -> dict:
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


def analyze(config: dict, *, root: Path) -> dict:
    rows = {
        arm: _read_predictions(
            _prediction_dir(root, config, arm) / "predictions.jsonl"
        )
        for arm in ARMS
    }
    expected = config["test_examples"]
    if any(len(value) != expected for value in rows.values()):
        raise ValueError("Every arm must have a complete test prediction file")
    reference = [(row["idx"], row["ground_truth"]) for row in rows["clean"]]
    for arm in ARMS[1:]:
        if [(row["idx"], row["ground_truth"]) for row in rows[arm]] != reference:
            raise ValueError(f"Evaluation rows are not paired for {arm}")
    correct = {
        arm: np.asarray([bool(row["correct"]) for row in value], dtype=bool)
        for arm, value in rows.items()
    }
    accuracy = {arm: float(values.mean()) for arm, values in correct.items()}
    noise_damage = accuracy["clean"] - accuracy["noisy_equal"]
    raw_recovery = accuracy["oracle_raw_0.1"] - accuracy["noisy_equal"]
    normalized_recovery = accuracy["oracle_normalized_0.1"] - accuracy["noisy_equal"]
    raw_ratio = raw_recovery / noise_damage if noise_damage > 0 else None
    comparisons = {
        "raw_vs_noisy_equal": _paired_comparison(
            correct["noisy_equal"], correct["oracle_raw_0.1"],
            seed=config["analysis_seed"], samples=config["bootstrap_samples"]
        ),
        "normalized_vs_noisy_equal": _paired_comparison(
            correct["noisy_equal"], correct["oracle_normalized_0.1"],
            seed=config["analysis_seed"] + 1, samples=config["bootstrap_samples"]
        ),
        "noisy_equal_vs_clean": _paired_comparison(
            correct["clean"], correct["noisy_equal"],
            seed=config["analysis_seed"] + 2, samples=config["bootstrap_samples"]
        ),
    }
    inconclusive = noise_damage < config["minimum_damage"]
    raw_positive = (
        not inconclusive
        and raw_recovery >= config["minimum_recovery"]
        and raw_ratio is not None
        and raw_ratio >= config["minimum_recovery_ratio"]
    )
    normalized_improves = normalized_recovery > 0
    if inconclusive:
        interpretation = "INCONCLUSIVE_INSUFFICIENT_NOISE_DAMAGE"
    elif not raw_positive:
        interpretation = "NO_POSITIVE_RAW_0.1_MECHANISM_RESULT"
    elif not normalized_improves:
        interpretation = "RAW_POSITIVE_BUT_COMPATIBLE_WITH_GENERIC_AUXILIARY_ATTENUATION"
    else:
        interpretation = "POSITIVE_SELECTIVE_ORACLE_WEIGHTING_EVIDENCE"
    result = {
        "run_id": config.get("analysis_run_id", "O020"),
        "status": "PASS",
        "examples": expected,
        "ground_truth_source": "official test dataset answer field",
        "accuracy": accuracy,
        "noise_damage": noise_damage,
        "raw_oracle_recovery": raw_recovery,
        "normalized_oracle_recovery": normalized_recovery,
        "raw_recovery_ratio": raw_ratio,
        "thresholds": {
            "minimum_damage": config["minimum_damage"],
            "minimum_recovery": config["minimum_recovery"],
            "minimum_recovery_ratio": config["minimum_recovery_ratio"],
        },
        "comparisons": comparisons,
        "interpretation": interpretation,
        "claim_boundary": "Tests perfect-label weighting on controlled synthetic auxiliary-step noise; does not validate a detector or natural teacher noise.",
    }
    atomic_json(_output_root(root, config) / config.get("analysis_filename", "o020_causal_analysis.json"), result)
    return result


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    if args.mode == "evaluate":
        if args.arm is None:
            raise ValueError("--arm is required in evaluate mode")
        result = evaluate_arm(config, args.arm, root=root, resume=args.resume)
    else:
        result = analyze(config, root=root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
