from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.causal_experiment import (
    CAUSAL_ARMS,
    COVERAGES,
    create_causal_schedule,
    run_sanity_gate,
    run_training_arm,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the causal-propagation weighting experiment")
    parser.add_argument("mode", choices=("schedule", "sanity", "train"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("pilot", "formal"))
    parser.add_argument("--arm", choices=CAUSAL_ARMS)
    parser.add_argument("--coverage", type=int, choices=COVERAGES)
    parser.add_argument("--updates", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    if args.mode == "schedule":
        result = create_causal_schedule(config, project_root=root)
        result = {
            **{key: value for key, value in result.items() if key != "splits"},
            "splits": {
                name: {key: value for key, value in details.items() if key != "entries"}
                for name, details in result["splits"].items()
            },
        }
    elif args.mode == "sanity":
        result = run_sanity_gate(config, project_root=root)
    else:
        if args.split is None or args.arm is None or args.coverage is None:
            raise ValueError("--split, --arm and --coverage are required in train mode")
        if args.split == "pilot" and args.arm not in {"clean", "noisy_equal"}:
            raise ValueError("Pilot training only permits clean and noisy_equal")
        result = run_training_arm(
            config,
            split=args.split,
            arm=args.arm,
            coverage=args.coverage,
            project_root=root,
            updates_override=args.updates,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
