from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import argparse
import gc
import json

import torch

from rsr_rd_simcot.config import DataConfig, ExperimentConfig, ModelConfig, TrainConfig
from rsr_rd_simcot.data import materialize_dataset, read_jsonl
from rsr_rd_simcot.evaluation import evaluate_branch
from rsr_rd_simcot.reporting import build_report
from rsr_rd_simcot.scoring import StepScores, load_scores
from rsr_rd_simcot.training import (
    offline_score_training_data,
    train_branch,
    warmup_student,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=(
            "data",
            "model-preflight",
            "warmup",
            "score",
            "preflight",
            "train",
            "evaluate",
            "all",
        ),
    )
    parser.add_argument("--smoke", action="store_true", help="Use a tiny fast configuration")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", help="Hugging Face model id or local model directory")
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--branch-updates", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--freeze-auxiliary-bottom", type=int, default=0)
    parser.add_argument("--max-branch-hours", type=float)
    return parser.parse_args()


def make_config(smoke: bool) -> ExperimentConfig:
    config = ExperimentConfig()
    if not smoke:
        return config
    return replace(
        config,
        data=DataConfig(train_size=24, val_size=4, test_size=4, noise_rate=0.25, seed=20260804),
        model=ModelConfig(
            model_name="sshleifer/tiny-gpt2",
            max_question_tokens=64,
            max_step_tokens=32,
            max_answer_tokens=16,
            max_latent_steps=4,
            rank_clip=100,
        ),
        train=TrainConfig(
            seed=0,
            warmup_updates=1,
            branch_updates=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            checkpoint_every=1,
            lambda_step=1.0,
            max_reserved_memory_gb=7.4,
            hard_stop_hours=1.0,
        ),
        work_dir="work/smoke_poc",
        output_dir="outputs/overnight_poc/smoke",
    )


def main() -> None:
    args = parse_args()
    config = make_config(args.smoke)
    if args.model_name is not None:
        config = replace(config, model=replace(config.model, model_name=args.model_name))
    train_overrides = {}
    if args.warmup_updates is not None:
        train_overrides["warmup_updates"] = args.warmup_updates
    if args.branch_updates is not None:
        train_overrides["branch_updates"] = args.branch_updates
    if args.gradient_accumulation is not None:
        train_overrides["gradient_accumulation_steps"] = args.gradient_accumulation
    if train_overrides:
        config = replace(config, train=replace(config.train, **train_overrides))
    work = Path(config.work_dir)
    output = Path(config.output_dir)
    data_dir = work / "data"
    warmup_dir = work / "warmup_model"
    score_path = work / "scores" / "train_scores.jsonl"
    config.save(output / "config.json")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)

    if args.phase in {"data", "all"}:
        manifest = materialize_dataset(config.data, data_dir)
        print(json.dumps(manifest["summary"], indent=2))
        if args.phase == "data":
            return

    train_examples = read_jsonl(data_dir / "train.jsonl")
    test_examples = read_jsonl(data_dir / "test.jsonl")

    if args.phase == "model-preflight":
        example = max(
            train_examples,
            key=lambda item: (len(item.observed_steps), len(item.question)),
        )
        dummy_scores = {
            example.example_id: StepScores(
                example_id=example.example_id,
                rsr=[1.0] * len(example.observed_steps),
                rd=[1.0] * len(example.observed_steps),
                weights=[1.0] * len(example.observed_steps),
                noisy_step_index=example.noisy_step_index,
            )
        }
        freeze = args.freeze_auxiliary_bottom
        try:
            result = train_branch(
                "equal",
                [example],
                dummy_scores,
                config,
                config.model.model_name,
                device,
                updates=1,
                freeze_auxiliary_bottom=freeze,
                save_checkpoints=False,
                write_result=False,
            )
        except torch.OutOfMemoryError:
            if freeze or device.type != "cuda":
                raise
            gc.collect()
            torch.cuda.empty_cache()
            freeze = 8
            result = train_branch(
                "equal",
                [example],
                dummy_scores,
                config,
                config.model.model_name,
                device,
                updates=1,
                freeze_auxiliary_bottom=freeze,
                save_checkpoints=False,
                write_result=False,
            )
        payload = asdict(result)
        payload["freeze_auxiliary_bottom"] = freeze
        payload["within_memory_limit"] = (
            result.peak_reserved_gb <= config.train.max_reserved_memory_gb
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "model_preflight.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        print(json.dumps(payload, indent=2))
        return

    if args.phase in {"warmup", "all"}:
        warmup_student(train_examples, config, device)
        if args.phase == "warmup":
            return

    if args.phase in {"score", "all"}:
        _, stats = offline_score_training_data(
            train_examples, config, warmup_dir, device
        )
        print(json.dumps(stats, indent=2))
        if args.phase == "score":
            return

    scores = load_scores(score_path)

    if args.phase == "preflight":
        results = []
        for branch in ("equal", "weighted"):
            result = train_branch(
                branch,
                train_examples,
                scores,
                config,
                warmup_dir,
                device,
                updates=20,
                freeze_auxiliary_bottom=args.freeze_auxiliary_bottom,
                save_checkpoints=False,
                write_result=False,
            )
            results.append(asdict(result))
        print(json.dumps(results, indent=2))
        return

    if args.phase in {"train", "all"}:
        equal_run = train_branch(
            "equal",
            train_examples,
            scores,
            config,
            warmup_dir,
            device,
            freeze_auxiliary_bottom=args.freeze_auxiliary_bottom,
            max_seconds=(
                args.max_branch_hours * 3600
                if args.max_branch_hours is not None
                else None
            ),
        )
        weighted_run = train_branch(
            "weighted",
            train_examples,
            scores,
            config,
            warmup_dir,
            device,
            updates=equal_run.completed_updates,
            freeze_auxiliary_bottom=args.freeze_auxiliary_bottom,
            max_seconds=(
                args.max_branch_hours * 3600
                if args.max_branch_hours is not None
                else None
            ),
        )
        print(json.dumps([asdict(equal_run), asdict(weighted_run)], indent=2))
        if args.phase == "train":
            return

    if args.phase in {"evaluate", "all"}:
        equal_run = json.loads((output / "equal_run.json").read_text(encoding="utf-8"))
        weighted_run = json.loads((output / "weighted_run.json").read_text(encoding="utf-8"))
        weight_stats = json.loads(
            (output / "weight_statistics.json").read_text(encoding="utf-8")
        )
        equal_metrics = evaluate_branch(
            "equal", test_examples, config, equal_run["checkpoint_dir"], device
        )
        weighted_metrics = evaluate_branch(
            "weighted", test_examples, config, weighted_run["checkpoint_dir"], device
        )
        result = build_report(
            output,
            weight_stats,
            equal_run,
            weighted_run,
            equal_metrics,
            weighted_metrics,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
