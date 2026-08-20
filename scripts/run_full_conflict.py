from __future__ import annotations

from pathlib import Path
import argparse
import json
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.full_conflict_data import (  # noqa: E402
    freeze_full_conflict_schedule,
    prepare_full_conflict_data,
)
from reliable_simcot.full_conflict_evaluation import (  # noqa: E402
    analyze_25,
    analyze_50,
    evaluate_full_conflict_arm,
)
from reliable_simcot.full_conflict_experiment import (  # noqa: E402
    run_full_conflict_gradient_audit,
    run_full_conflict_training,
    run_sanity_gate,
)
from reliable_simcot.full_conflict_generation import (  # noqa: E402
    build_prompt_manifest,
    generate_codex_authored_records,
    validate_generation_records,
)


def load_config(path: str) -> dict:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="SIM-CoT full-conflict experiment")
    parser.add_argument(
        "mode",
        choices=(
            "prepare",
            "build-prompts",
            "generate",
            "gate-small",
            "validate",
            "freeze",
            "sanity",
            "audit",
            "train",
            "evaluate",
            "analyze25",
            "analyze50",
        ),
    )
    parser.add_argument("--config", default="configs/reliable_simcot/full_conflict.json")
    parser.add_argument("--arm")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "prepare":
        result = prepare_full_conflict_data(config, project_root=ROOT)
    elif args.mode == "build-prompts":
        rows = build_prompt_manifest(config, project_root=ROOT)
        result = {"rows": len(rows)}
    elif args.mode == "generate":
        result = generate_codex_authored_records(config, project_root=ROOT)
    elif args.mode == "gate-small":
        result = validate_generation_records(config, project_root=ROOT, small_batch=True)
    elif args.mode == "validate":
        result = validate_generation_records(config, project_root=ROOT, small_batch=False)
    elif args.mode == "freeze":
        result = freeze_full_conflict_schedule(config, project_root=ROOT)
    elif args.mode == "sanity":
        result = run_sanity_gate(config, project_root=ROOT)
    elif args.mode == "audit":
        result = run_full_conflict_gradient_audit(config, project_root=ROOT)
    elif args.mode == "train":
        if args.arm is None or args.seed is None:
            parser.error("train requires --arm and --seed")
        result = run_full_conflict_training(
            config, arm=args.arm, seed=args.seed, project_root=ROOT
        )
    elif args.mode == "evaluate":
        if args.arm is None or args.seed is None:
            parser.error("evaluate requires --arm and --seed")
        result = evaluate_full_conflict_arm(
            config,
            arm=args.arm,
            seed=args.seed,
            project_root=ROOT,
            resume=args.resume,
        )
    elif args.mode == "analyze25":
        result = analyze_25(config, project_root=ROOT)
    else:
        result = analyze_50(config, project_root=ROOT)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
