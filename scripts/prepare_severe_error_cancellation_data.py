from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.error_cancellation_data import (  # noqa: E402
    prepare_severe_error_cancellation_data,
)
from reliable_simcot.error_cancellation_experiment import prepare_schedule  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/error_cancellation_gsm8k_v12_severe.json",
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    audit = prepare_severe_error_cancellation_data(config, project_root=ROOT)
    schedule = prepare_schedule(config, project_root=ROOT)
    print(
        json.dumps(
            {
                "data_audit": audit,
                "manifest_path": config["manifest_path"],
                "schedule_path": config["schedule_path"],
                "schedule_sha256": schedule["schedule_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
