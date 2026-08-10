from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

from reliable_simcot.causal_evaluation import (
    analyze_formal,
    calibrate_coverage,
    evaluate_formal,
    evaluate_pilot,
)
from reliable_simcot.causal_experiment import (
    CAUSAL_ARMS,
    COVERAGES,
    create_causal_schedule,
    run_sanity_gate,
    run_training_arm,
)
from reliable_simcot.m1_training import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the preregistered first-night run")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    state_path = (root / config["overnight_state_path"]).resolve()
    state: dict = {
        "run_id": config["overnight_run_id"],
        "status": "RUNNING",
        "started_unix": time.time(),
        "completed_phases": [],
        "official_test_opened": False,
    }

    def completed(phase: str, details: dict | None = None) -> None:
        state["completed_phases"].append(phase)
        if details is not None:
            state[phase] = details
        atomic_json(state_path, state)

    atomic_json(state_path, state)
    schedule = create_causal_schedule(config, project_root=root)
    completed("schedule", {"schedule_sha256": schedule["schedule_sha256"]})
    sanity = run_sanity_gate(config, project_root=root)
    completed("sanity", {"gate_passed": sanity["gate_passed"]})
    if not sanity["gate_passed"]:
        state["status"] = "STOPPED_SANITY_GATE"
        state["finished_unix"] = time.time()
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    run_training_arm(
        config,
        split="pilot",
        arm="clean",
        coverage=25,
        project_root=root,
    )
    completed("pilot_train_clean")
    for coverage in COVERAGES:
        run_training_arm(
            config,
            split="pilot",
            arm="noisy_equal",
            coverage=coverage,
            project_root=root,
        )
        completed(f"pilot_train_noisy_equal_{coverage}")

    evaluate_pilot(
        config,
        arm="clean",
        coverage=25,
        project_root=root,
        resume=args.resume_eval,
    )
    completed("pilot_eval_clean")
    for coverage in COVERAGES:
        evaluate_pilot(
            config,
            arm="noisy_equal",
            coverage=coverage,
            project_root=root,
            resume=args.resume_eval,
        )
        completed(f"pilot_eval_noisy_equal_{coverage}")
    gate = calibrate_coverage(config, project_root=root)
    completed(
        "calibration",
        {
            "gate_passed": gate["gate_passed"],
            "chosen_coverage": gate["chosen_coverage"],
            "status": gate["status"],
        },
    )
    if not gate["gate_passed"]:
        state["status"] = gate["status"]
        state["finished_unix"] = time.time()
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return

    coverage = int(gate["chosen_coverage"])
    for arm in CAUSAL_ARMS:
        run_training_arm(
            config,
            split="formal",
            arm=arm,
            coverage=coverage,
            project_root=root,
        )
        completed(f"formal_train_{arm}")
    state["official_test_opened"] = True
    atomic_json(state_path, state)
    for arm in CAUSAL_ARMS:
        evaluate_formal(
            config, arm=arm, project_root=root, resume=args.resume_eval
        )
        completed(f"formal_eval_{arm}")
    analysis = analyze_formal(config, project_root=root)
    completed("formal_analysis", analysis)
    state["status"] = analysis["status"]
    state["finished_unix"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
