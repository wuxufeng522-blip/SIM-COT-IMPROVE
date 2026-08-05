from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.m1_training import create_schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze the shared R010/R011 sample schedule.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    schedule = create_schedule(config, project_root=Path.cwd())
    print(
        json.dumps(
            {
                "schedule_sha256": schedule["schedule_sha256"],
                "effective_micro_batches": schedule["effective_micro_batches"],
                "rejected_candidates": len(schedule["rejected_candidates"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
