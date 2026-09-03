from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
from statistics import mean
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402
from reliable_simcot.self_corrected_evaluation import (  # noqa: E402
    analyze_self_corrected,
    evaluate_self_corrected_arm,
    write_self_corrected_report,
)
from reliable_simcot.self_corrected_experiment import (  # noqa: E402
    ARMS,
    checkpoint_path,
    load_training_schedule,
    run_training_arm,
    training_directory,
    verify_pretraining_gates,
)


def load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def state_path(config: dict[str, Any]) -> Path:
    return (ROOT / config["state_path"]).resolve()


def save_state(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    path = state_path(config)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "run_id": str(config.get("overnight_run_id", "SC-OVERNIGHT")),
            "started_unix": time.time(),
            "completed_training": [],
            "completed_evaluation": [],
            "official_test_opened": False,
        }
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)
    return state


def training_complete(
    config: dict[str, Any], seed: int, arm: str, schedule_sha256: str
) -> bool:
    metrics_path = training_directory(
        ROOT, config, seed, arm, sanity=False
    ) / "metrics.json"
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if not metrics_path.is_file() or not checkpoint.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == arm
            and metrics.get("schedule_sha256") == schedule_sha256
            and metrics.get("checkpoint_sha256") == sha256_file(checkpoint)
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def evaluation_complete(
    config: dict[str, Any], seed: int, arm: str, schedule_sha256: str
) -> bool:
    directory = ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    if not metrics_path.is_file() or not predictions_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        count = sum(
            1
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == arm
            and metrics.get("schedule_sha256") == schedule_sha256
            and metrics.get("predictions_sha256") == sha256_file(predictions_path)
            and count == int(config["test_examples"])
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def run_training_queue(
    config: dict[str, Any], schedule_sha256: str, *, arms: tuple[str, ...] = ARMS
) -> None:
    for seed in [int(value) for value in config["seeds"]]:
        for arm in arms:
            key = f"{seed}:{arm}"
            if training_complete(config, seed, arm, schedule_sha256):
                print(f"skip completed training {key}", flush=True)
                continue
            metrics_path = training_directory(
                ROOT, config, seed, arm, sanity=False
            ) / "metrics.json"
            checkpoint = checkpoint_path(ROOT, config, seed, arm)
            if metrics_path.exists() or checkpoint.exists():
                raise RuntimeError(
                    f"Incomplete final artifacts already exist for {key}; refusing to overwrite"
                )
            save_state(config, status="RUNNING", phase="TRAIN", current=key)
            result = run_training_arm(config, arm=arm, seed=seed, project_root=ROOT)
            if result.get("status") != "PASS":
                raise RuntimeError(
                    f"Training arm {key} failed its frozen gate: "
                    f"peak_reserved_gb={result.get('peak_reserved_gb')}"
                )
            state = save_state(config)
            completed = list(state.get("completed_training", []))
            if key not in completed:
                completed.append(key)
            save_state(config, completed_training=completed)


def run_evaluation_queue(
    config: dict[str, Any], schedule_sha256: str, *, arms: tuple[str, ...] = ARMS
) -> None:
    save_state(config, official_test_opened=True)
    for arm in arms:
        for seed in [int(value) for value in config["seeds"]]:
            key = f"{seed}:{arm}"
            if evaluation_complete(config, seed, arm, schedule_sha256):
                print(f"skip completed evaluation {key}", flush=True)
                continue
            directory = ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm
            resume = (directory / "predictions.jsonl").exists()
            save_state(config, status="RUNNING", phase="EVALUATE", current=key)
            result = evaluate_self_corrected_arm(
                config,
                arm=arm,
                seed=seed,
                project_root=ROOT,
                resume=resume,
            )
            if result.get("status") != "PASS":
                raise RuntimeError(
                    f"Evaluation arm {key} failed its frozen gate: "
                    f"peak_reserved_gb={result.get('peak_reserved_gb')}"
                )
            state = save_state(config)
            completed = list(state.get("completed_evaluation", []))
            if key not in completed:
                completed.append(key)
            save_state(config, completed_evaluation=completed)


def run_clean_baseline_gate(config: dict[str, Any]) -> dict[str, Any]:
    threshold = float(config.get("minimum_clean_accuracy", 0.0))
    values = []
    for seed in [int(value) for value in config["seeds"]]:
        metrics_path = (
            ROOT
            / config["output_root"]
            / "eval"
            / f"seed_{seed}"
            / "clean"
            / "metrics.json"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        values.append(float(metrics["accuracy"]))
    accuracy_mean = mean(values)
    passed = accuracy_mean >= threshold
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "accuracy_values": values,
        "accuracy_mean": accuracy_mean,
        "minimum_clean_accuracy": threshold,
        "dataset_family": str(config.get("dataset_family", "math")),
        "decision": (
            "continue_factorial" if passed else "stop_floor_effect_before_noisy_arms"
        ),
    }
    gate_path = (ROOT / config["clean_baseline_gate_path"]).resolve()
    atomic_json(gate_path, result)
    if not passed:
        raise RuntimeError(
            f"Clean baseline floor gate failed: mean={accuracy_mean:.4f}, "
            f"minimum={threshold:.4f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen self-corrected strong-conflict factorial"
    )
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/self_corrected_strong_conflict.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        verify_pretraining_gates(config, ROOT, require_sanity=True)
        schedule = load_training_schedule(config, ROOT)
        save_state(
            config,
            status="RUNNING",
            phase="TRAIN",
            current=None,
            schedule_sha256=schedule["schedule_sha256"],
        )
        # Establish that the student can solve this benchmark before spending
        # compute on noisy arms.  This gate was added after the MATH run exposed
        # a floor effect; it does not select among noisy configurations.
        run_training_queue(config, schedule["schedule_sha256"], arms=("clean",))
        run_evaluation_queue(config, schedule["schedule_sha256"], arms=("clean",))
        save_state(config, status="RUNNING", phase="CLEAN_BASELINE_GATE", current=None)
        run_clean_baseline_gate(config)
        remaining_arms = tuple(arm for arm in ARMS if arm != "clean")
        run_training_queue(config, schedule["schedule_sha256"], arms=remaining_arms)
        run_evaluation_queue(config, schedule["schedule_sha256"], arms=remaining_arms)
        save_state(config, status="RUNNING", phase="ANALYZE", current="SC-ANALYSIS")
        analysis = analyze_self_corrected(config, project_root=ROOT)
        report = write_self_corrected_report(
            config, project_root=ROOT, analysis=analysis
        )
        save_state(
            config,
            status="PASS",
            phase="COMPLETE",
            current=None,
            family_conclusions=analysis["family_conclusions"],
            report_path=str(report),
            completed_unix=time.time(),
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        save_state(
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
