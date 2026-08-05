from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.single_gpu_smoke import run_training_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the official SIM-CoT single-GPU training and reload smoke test."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_training_smoke(
        config,
        project_root=Path.cwd(),
        updates_override=args.updates,
        accumulation_override=args.gradient_accumulation,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["sanity_passed"] or result["gate_passed"] is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
