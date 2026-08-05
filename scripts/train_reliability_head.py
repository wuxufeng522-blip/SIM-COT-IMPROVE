from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.head_training import run_lofo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--heldout-family", action="append")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_lofo(
        config,
        project_root=Path.cwd(),
        families=args.heldout_family,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["all_registered_families_run"] and not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
