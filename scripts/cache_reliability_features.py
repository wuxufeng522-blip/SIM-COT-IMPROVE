from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.features import extract_feature_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = extract_feature_cache(
        config,
        project_root=Path.cwd(),
        max_rows=args.max_rows,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
