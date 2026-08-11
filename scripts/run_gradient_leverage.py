from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.gradient_leverage import (
    LEVERAGE_ARMS,
    analyze_leverage,
    evaluate_leverage_arm,
    prepare_confirm_set,
    run_gradient_audit,
    run_leverage_sanity,
    run_leverage_training,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SIM-CoT auxiliary-gradient leverage experiment")
    parser.add_argument(
        "mode", choices=("prepare", "audit", "sanity", "train", "evaluate", "analyze")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", choices=LEVERAGE_ARMS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    if args.mode == "prepare":
        result = prepare_confirm_set(config, project_root=root)
        result = {key: value for key, value in result.items() if key != "entries"}
    elif args.mode == "audit":
        result = run_gradient_audit(config, project_root=root)
    elif args.mode == "sanity":
        result = run_leverage_sanity(config, project_root=root)
    elif args.mode == "train":
        if args.arm is None or args.seed is None:
            raise ValueError("train mode requires --arm and --seed")
        result = run_leverage_training(
            config,
            arm=args.arm,
            seed=args.seed,
            project_root=root,
            updates_override=args.updates,
        )
    elif args.mode == "evaluate":
        if args.arm is None or args.seed is None:
            raise ValueError("evaluate mode requires --arm and --seed")
        result = evaluate_leverage_arm(
            config,
            arm=args.arm,
            seed=args.seed,
            project_root=root,
            resume=args.resume,
        )
    else:
        result = analyze_leverage(config, project_root=root)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
