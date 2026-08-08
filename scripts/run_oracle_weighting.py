from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.oracle_weighting import (
    ARMS,
    create_oracle_schedule,
    run_sanity_gate,
    run_training_arm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the oracle step-weighting causal test")
    parser.add_argument("mode", choices=("schedule", "sanity", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--updates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    if args.mode == "schedule":
        result = create_oracle_schedule(config, project_root=root)
        result = {key: value for key, value in result.items() if key != "entries"}
    elif args.mode == "sanity":
        result = run_sanity_gate(config, project_root=root)
    else:
        if args.arm is None:
            raise ValueError("--arm is required in train mode")
        result = run_training_arm(
            config, args.arm, project_root=root, updates_override=args.updates
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
