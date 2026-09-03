from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.self_corrected_data import prepare_self_corrected_data  # noqa: E402
from reliable_simcot.self_corrected_experiment import (  # noqa: E402
    prepare_training_schedule,
    run_full_schedule_memory_gate,
    run_max_length_memory_gate,
    run_sanity_gate,
    run_training_arm,
    run_weight_gradient_audit,
)


def load_config(path: str) -> dict:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SIM-CoT self-corrected strong-conflict factorial experiment"
    )
    parser.add_argument(
        "mode",
        choices=(
            "prepare-data",
            "freeze-schedule",
            "sanity",
            "max-length-memory-gate",
            "full-schedule-memory-gate",
            "gradient-audit",
            "train",
        ),
    )
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/self_corrected_strong_conflict.json",
    )
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.mode == "prepare-data":
        result = prepare_self_corrected_data(config, project_root=ROOT)
    elif args.mode == "freeze-schedule":
        result = prepare_training_schedule(config, project_root=ROOT)
    elif args.mode == "sanity":
        result = run_sanity_gate(config, project_root=ROOT)
    elif args.mode == "max-length-memory-gate":
        result = run_max_length_memory_gate(config, project_root=ROOT)
    elif args.mode == "full-schedule-memory-gate":
        result = run_full_schedule_memory_gate(config, project_root=ROOT)
    elif args.mode == "gradient-audit":
        result = run_weight_gradient_audit(config, project_root=ROOT)
    else:
        if args.arm is None or args.seed is None:
            parser.error("train requires --arm and --seed")
        result = run_training_arm(
            config,
            arm=args.arm,
            seed=args.seed,
            project_root=ROOT,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
