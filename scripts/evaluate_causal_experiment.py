from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.causal_evaluation import (
    analyze_formal,
    calibrate_coverage,
    evaluate_formal,
    evaluate_pilot,
)
from reliable_simcot.causal_experiment import CAUSAL_ARMS, COVERAGES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the causal-propagation experiment")
    parser.add_argument(
        "mode", choices=("pilot", "calibrate", "formal", "analyze-formal")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=CAUSAL_ARMS)
    parser.add_argument("--coverage", type=int, choices=COVERAGES)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    if args.mode == "pilot":
        if args.arm not in {"clean", "noisy_equal"} or args.coverage is None:
            raise ValueError("Pilot evaluation needs --arm and --coverage")
        result = evaluate_pilot(
            config,
            arm=args.arm,
            coverage=args.coverage,
            project_root=root,
            resume=args.resume,
        )
    elif args.mode == "calibrate":
        result = calibrate_coverage(config, project_root=root)
    elif args.mode == "formal":
        if args.arm is None:
            raise ValueError("Formal evaluation needs --arm")
        result = evaluate_formal(
            config, arm=args.arm, project_root=root, resume=args.resume
        )
    else:
        result = analyze_formal(config, project_root=root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
