from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

from reliable_simcot.gradient_leverage import (
    LEVERAGE_ARMS,
    analyze_leverage,
    evaluate_leverage_arm,
    prepare_confirm_set,
    run_gradient_audit,
    run_leverage_sanity,
    run_leverage_training,
)
from reliable_simcot.m1_training import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute the preregistered gradient-leverage night")
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
        "completed": [],
        "official_test_opened": False,
    }

    def complete(name: str, details: dict | None = None) -> None:
        state["completed"].append(name)
        if details is not None:
            state[name] = details
        atomic_json(state_path, state)

    atomic_json(state_path, state)
    manifest = prepare_confirm_set(config, project_root=root)
    complete("prepare", {"manifest_sha256": manifest["manifest_sha256"]})
    sanity = run_leverage_sanity(config, project_root=root)
    complete("sanity", {"gate_passed": sanity["gate_passed"]})
    if not sanity["gate_passed"]:
        state["status"] = "STOPPED_SANITY_GATE"
        state["finished_unix"] = time.time()
        atomic_json(state_path, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    audit = run_gradient_audit(config, project_root=root)
    complete("gradient_audit", {"verdict": audit["verdict"]})
    for seed in config["seeds"]:
        for arm in LEVERAGE_ARMS:
            run_leverage_training(config, arm=arm, seed=seed, project_root=root)
            complete(f"train_{seed}_{arm}")
    for seed in config["seeds"]:
        for arm in LEVERAGE_ARMS:
            evaluate_leverage_arm(
                config,
                arm=arm,
                seed=seed,
                project_root=root,
                resume=args.resume_eval,
            )
            complete(f"eval_{seed}_{arm}")
    analysis = analyze_leverage(config, project_root=root)
    complete("analysis", {"verdict": analysis["verdict"]})
    state["status"] = "PASS"
    state["verdict"] = analysis["verdict"]
    state["finished_unix"] = time.time()
    atomic_json(state_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
