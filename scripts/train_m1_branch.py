from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.m1_training import run_m1_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one frozen-budget M1 branch.")
    parser.add_argument("--common", required=True, type=Path)
    parser.add_argument("--branch", required=True, type=Path)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--work-dir")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    common = json.loads(args.common.read_text(encoding="utf-8"))
    branch = json.loads(args.branch.read_text(encoding="utf-8"))
    result = run_m1_training(
        branch,
        common,
        project_root=Path.cwd(),
        updates_override=args.updates,
        output_dir_override=args.output_dir,
        work_dir_override=args.work_dir,
        resume=args.resume,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
