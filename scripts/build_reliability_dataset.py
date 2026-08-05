from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.reliability_data import build_reliability_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = build_reliability_dataset(config, project_root=Path.cwd())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
