from __future__ import annotations

from pathlib import Path
import argparse
import json

from reliable_simcot.single_gpu_smoke import run_resume_smoke_probe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the R005 checkpoint-resume probe.")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_resume_smoke_probe(config, project_root=Path.cwd())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
