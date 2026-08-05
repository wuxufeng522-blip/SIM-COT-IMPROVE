from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.review_metrics import compile_review_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize R022 labels.")
    parser.add_argument("--package-manifest", required=True, type=Path)
    parser.add_argument("--reviewer-a", required=True, type=Path)
    parser.add_argument("--reviewer-b", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = compile_review_results(
        package_manifest_path=args.package_manifest,
        reviewer_a_labeled_path=args.reviewer_a,
        reviewer_b_labeled_path=args.reviewer_b,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
