from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.provenance import build_provenance_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify and record all pinned inputs for SIM-CoT run R001."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest, path = build_provenance_manifest(
        args.config,
        project_root=args.project_root,
    )
    summary = {
        "run_id": manifest["run_id"],
        "status": manifest["status"],
        "official_revision": manifest["official_source"]["repository_revision"],
        "checkpoint_sha256": manifest["checkpoint"]["artifact"]["sha256"],
        "dataset_revision": manifest["dataset"]["revision"],
        "manifest": str(path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
