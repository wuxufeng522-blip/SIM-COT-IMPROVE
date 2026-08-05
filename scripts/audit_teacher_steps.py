from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.audit import write_frozen_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze an unbiased question-cluster sample for natural-step review."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    manifest = write_frozen_audit(
        dataset_path=root / config["dataset_path"],
        output_dir=root / config["output_dir"],
        seed=config["seed"],
        question_count=config["question_count"],
        min_steps=config["minimum_step_count"],
        expected_dataset_sha256=config["dataset_sha256"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
