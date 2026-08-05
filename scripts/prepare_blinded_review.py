from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.audit import write_blinded_review_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create blinded two-reviewer assignments from frozen R020 rows."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    manifest = write_blinded_review_package(
        parent_manifest_path=root / config["parent_manifest_path"],
        output_dir=root / config["output_dir"],
        expected_parent_rows_sha256=config["parent_rows_sha256"],
        assignment_seed=config["assignment_seed"],
        overlap_fraction=config["overlap_fraction"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
