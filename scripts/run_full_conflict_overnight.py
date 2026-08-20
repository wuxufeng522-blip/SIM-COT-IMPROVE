from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import json
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.full_conflict_data import load_frozen_schedule  # noqa: E402
from reliable_simcot.full_conflict_evaluation import (  # noqa: E402
    analyze_25,
    analyze_50,
    evaluate_full_conflict_arm,
    write_chinese_report,
)
from reliable_simcot.full_conflict_experiment import (  # noqa: E402
    CONDITIONAL_ARM,
    MAIN_ARMS,
    checkpoint_path,
    run_full_conflict_gradient_audit,
    run_full_conflict_training,
    run_sanity_gate,
    training_directory,
)
from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402


def _load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def _state_path(config: dict[str, Any]) -> Path:
    return (ROOT / config["overnight_state_path"]).resolve()


def _save_state(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    path = _state_path(config)
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "run_id": "FC000",
            "started_unix": time.time(),
            "completed_training": [],
            "completed_evaluation": [],
            "official_test_opened": False,
        }
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)
    return state


def _training_complete(config: dict[str, Any], seed: int, arm: str, schedule_hash: str) -> bool:
    metrics_path = training_directory(ROOT, config, seed, arm) / "metrics.json"
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if not metrics_path.exists() or not checkpoint.exists():
        return False
    try:
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            row.get("status") == "PASS"
            and row.get("seed") == seed
            and row.get("arm") == arm
            and row.get("schedule_sha256") == schedule_hash
            and sha256_file(checkpoint) == row.get("checkpoint_sha256")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _evaluation_complete(config: dict[str, Any], seed: int, arm: str, schedule_hash: str) -> bool:
    directory = ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        return False
    try:
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = sum(1 for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip())
        return (
            row.get("arm") == arm
            and row.get("seed") == seed
            and row.get("schedule_sha256") == schedule_hash
            and predictions == int(config["confirm_examples"])
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _run_training_queue(config: dict[str, Any], arms: tuple[str, ...], schedule_hash: str) -> None:
    for seed in config["seeds"]:
        for arm in arms:
            key = f"{seed}:{arm}"
            if _training_complete(config, seed, arm, schedule_hash):
                print(f"skip completed training {key}", flush=True)
                continue
            _save_state(config, status="RUNNING", phase="TRAIN", current=key)
            run_full_conflict_training(
                config, arm=arm, seed=seed, project_root=ROOT
            )
            state = _save_state(config)
            completed = list(state.get("completed_training", []))
            if key not in completed:
                completed.append(key)
            _save_state(config, completed_training=completed)


def _run_evaluation_queue(config: dict[str, Any], arms: tuple[str, ...], schedule_hash: str) -> None:
    for seed in config["seeds"]:
        for arm in arms:
            key = f"{seed}:{arm}"
            if _evaluation_complete(config, seed, arm, schedule_hash):
                print(f"skip completed evaluation {key}", flush=True)
                continue
            directory = ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm
            resume = (directory / "predictions.jsonl").exists()
            _save_state(config, status="RUNNING", phase="EVALUATE", current=key)
            evaluate_full_conflict_arm(
                config,
                arm=arm,
                seed=seed,
                project_root=ROOT,
                resume=resume,
            )
            state = _save_state(config)
            completed = list(state.get("completed_evaluation", []))
            if key not in completed:
                completed.append(key)
            _save_state(config, completed_evaluation=completed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional overnight full-conflict run")
    parser.add_argument("--config", default="configs/reliable_simcot/full_conflict.json")
    args = parser.parse_args()
    config = _load_config(args.config)
    try:
        schedule = load_frozen_schedule(config, ROOT)
        data_audit = json.loads((ROOT / config["data_audit_path"]).read_text(encoding="utf-8"))
        small_gate = json.loads((ROOT / config["small_batch_gate_path"]).read_text(encoding="utf-8"))
        if data_audit.get("status") != "PASS" or small_gate.get("readability_review") != "PASS":
            raise RuntimeError("Data gates are not fully passed")
        _save_state(
            config,
            status="RUNNING",
            phase="SANITY",
            current="FC030-FC032",
            schedule_sha256=schedule["schedule_sha256"],
        )
        sanity_path = (ROOT / config["sanity_gate_path"]).resolve()
        if sanity_path.exists():
            sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        else:
            sanity = run_sanity_gate(config, project_root=ROOT)
        if not sanity.get("gate_passed"):
            raise RuntimeError("GPU sanity gate failed")
        gradient_path = (ROOT / config["gradient_audit_path"]).resolve()
        if gradient_path.exists():
            gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
        else:
            gradient = run_full_conflict_gradient_audit(config, project_root=ROOT)
        if not gradient.get("gate_passed"):
            raise RuntimeError("Gradient-pathway gate failed")

        _run_training_queue(config, MAIN_ARMS, schedule["schedule_sha256"])
        _run_evaluation_queue(config, MAIN_ARMS, schedule["schedule_sha256"])
        _save_state(config, status="RUNNING", phase="ANALYZE_25", current="FC300")
        analysis25 = analyze_25(config, project_root=ROOT)
        if analysis25["gate_passed"]:
            final = analysis25
        else:
            _run_training_queue(config, (CONDITIONAL_ARM,), schedule["schedule_sha256"])
            _run_evaluation_queue(config, (CONDITIONAL_ARM,), schedule["schedule_sha256"])
            _save_state(config, status="RUNNING", phase="ANALYZE_50", current="FC500")
            final = analyze_50(config, project_root=ROOT)
        report = write_chinese_report(config, project_root=ROOT, final=final)
        _save_state(
            config,
            status="PASS",
            phase="COMPLETE",
            current=None,
            verdict=final["verdict"],
            report_path=str(report),
            completed_unix=time.time(),
        )
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
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
