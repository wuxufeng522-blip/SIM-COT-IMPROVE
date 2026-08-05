from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.audit import write_auto_triage_revision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive versioned automatic triage from a frozen R020 audit."
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path.cwd().resolve()
    manifest = write_auto_triage_revision(
        parent_manifest_path=root / config["parent_manifest_path"],
        output_dir=root / config["output_dir"],
        expected_parent_rows_sha256=config["parent_rows_sha256"],
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
