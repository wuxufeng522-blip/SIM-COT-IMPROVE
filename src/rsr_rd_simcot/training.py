from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from typing import Iterator, Sequence
import json
import random
import time

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, GPT2LMHeadModel

from .config import ExperimentConfig, ModelConfig, TrainConfig
from .data import Example
from .model import SimCoTModel
from .scoring import StepScores, save_scores, score_dataset


@dataclass
class TrainRunResult:
    branch: str
    requested_updates: int
    completed_updates: int
    elapsed_seconds: float
    peak_reserved_gb: float
    final_loss: float
    checkpoint_dir: str
    finite: bool


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tokenizer(model_name_or_path: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(str(model_name_or_path), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def encode_question(tokenizer, example: Example, config: ModelConfig) -> torch.Tensor:
    text = f"Question: {example.question}\nReasoning:\n"
    ids = tokenizer.encode(text, add_special_tokens=False)[: config.max_question_tokens]
    if not ids:
        raise ValueError(f"Empty question encoding for {example.example_id}")
    return torch.tensor(ids, dtype=torch.long)


def encode_answer(tokenizer, example: Example, config: ModelConfig) -> torch.Tensor:
    text = f"\nAnswer: {example.answer}{tokenizer.eos_token}"
    ids = tokenizer.encode(text, add_special_tokens=False)[: config.max_answer_tokens]
    if not ids:
        raise ValueError(f"Empty answer encoding for {example.example_id}")
    return torch.tensor(ids, dtype=torch.long)


def encode_steps(
    tokenizer,
    steps: Sequence[str],
    config: ModelConfig,
) -> list[torch.Tensor]:
    encoded: list[torch.Tensor] = []
    for step in steps[: config.max_latent_steps]:
        ids = tokenizer.encode(f"{step}\n", add_special_tokens=False)[
            : config.max_step_tokens
        ]
        if not ids:
            raise ValueError(f"Empty step encoding: {step}")
        encoded.append(torch.tensor(ids, dtype=torch.long))
    return encoded


def _ordered_examples(examples: Sequence[Example], seed: int) -> Iterator[Example]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    while True:
        rng.shuffle(indices)
        for index in indices:
            yield examples[index]


def _append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _make_scaler(device: torch.device):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda")


def warmup_student(
    examples: Sequence[Example],
    experiment: ExperimentConfig,
    device: torch.device,
    updates: int | None = None,
) -> Path:
    train_config = experiment.train
    requested_updates = updates or train_config.warmup_updates
    seed_everything(train_config.seed)
    tokenizer = load_tokenizer(experiment.model.model_name)
    model = GPT2LMHeadModel.from_pretrained(experiment.model.model_name)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scaler = _make_scaler(device)
    stream = _ordered_examples(examples, train_config.seed)
    log_path = Path(experiment.work_dir) / "logs" / "warmup.jsonl"
    if log_path.exists():
        log_path.unlink()

    start_time = time.monotonic()
    for update in range(1, requested_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(train_config.gradient_accumulation_steps):
            example = next(stream)
            prompt = f"Question: {example.question}\nReasoning:\n"
            target = "\n".join(example.clean_steps) + f"\nAnswer: {example.answer}{tokenizer.eos_token}"
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)[
                : experiment.model.max_question_tokens
            ]
            target_limit = (
                experiment.model.max_latent_steps * experiment.model.max_step_tokens
                + experiment.model.max_answer_tokens
            )
            target_ids = tokenizer.encode(target, add_special_tokens=False)[:target_limit]
            input_ids = torch.tensor([prompt_ids + target_ids], device=device)
            labels = input_ids.clone()
            labels[:, : len(prompt_ids)] = -100
            with _autocast(device):
                loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
                scaled_loss = loss / train_config.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite warmup loss at update {update}")
            scaler.scale(scaled_loss).backward()
            accumulated_loss += float(loss.detach().item())

        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        mean_loss = accumulated_loss / train_config.gradient_accumulation_steps
        if update == 1 or update % 20 == 0 or update == requested_updates:
            _append_jsonl(
                log_path,
                {
                    "phase": "warmup",
                    "update": update,
                    "loss": mean_loss,
                    "elapsed_seconds": time.monotonic() - start_time,
                },
            )

    checkpoint = Path(experiment.work_dir) / "warmup_model"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint)
    return checkpoint


def offline_score_training_data(
    examples: Sequence[Example],
    experiment: ExperimentConfig,
    warmup_checkpoint: str | Path,
    device: torch.device,
) -> tuple[list[StepScores], dict]:
    model = GPT2LMHeadModel.from_pretrained(str(warmup_checkpoint)).to(device)
    model.eval()
    tokenizer = load_tokenizer(warmup_checkpoint)
    scores = score_dataset(
        model,
        tokenizer,
        examples,
        device=device,
        rank_clip=experiment.model.rank_clip,
    )
    score_path = Path(experiment.work_dir) / "scores" / "train_scores.jsonl"
    save_scores(score_path, scores)

    labels: list[int] = []
    detection_scores: list[float] = []
    rsr_detection_scores: list[float] = []
    rd_detection_scores: list[float] = []
    step_noise_types: list[str] = []
    noisy_weights: list[float] = []
    clean_weights: list[float] = []
    example_by_id = {example.example_id: example for example in examples}
    for item in scores:
        example = example_by_id[item.example_id]
        for index, (rsr, rd, weight) in enumerate(
            zip(item.rsr, item.rd, item.weights, strict=True)
        ):
            noisy = item.noisy_step_index == index
            labels.append(int(noisy))
            detection_scores.append(-weight)
            rsr_detection_scores.append(-float(np.log(rsr)))
            rd_detection_scores.append(float(np.log(rd)))
            step_noise_types.append(example.noise_type if noisy else "clean")
            (noisy_weights if noisy else clean_weights).append(weight)

    from sklearn.metrics import roc_auc_score

    auc = float(roc_auc_score(labels, detection_scores))
    per_noise_type_auc = {}
    label_array = np.asarray(labels)
    type_array = np.asarray(step_noise_types)
    score_array = np.asarray(detection_scores)
    for noise_type in sorted({value for value in step_noise_types if value != "clean"}):
        mask = (type_array == "clean") | (type_array == noise_type)
        per_noise_type_auc[noise_type] = float(
            roc_auc_score((type_array[mask] == noise_type).astype(int), score_array[mask])
        )
    stats = {
        "roc_auc": auc,
        "joint_weight_auc": auc,
        "rsr_low_tail_auc": float(roc_auc_score(labels, rsr_detection_scores)),
        "rd_harm_auc": float(roc_auc_score(labels, rd_detection_scores)),
        "per_noise_type_joint_auc": per_noise_type_auc,
        "mean_noisy_weight": float(np.mean(noisy_weights)),
        "mean_clean_weight": float(np.mean(clean_weights)),
        "noisy_step_count": len(noisy_weights),
        "clean_step_count": len(clean_weights),
        "weight_formula": "u=-0.5*max(0,-z(log_RSR))-0.5*z(log_RD)",
    }
    stats_path = Path(experiment.output_dir) / "weight_statistics.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores, stats


def _save_branch_state(
    model: SimCoTModel,
    optimizer,
    scaler,
    update: int,
    target: Path,
) -> None:
    model.save_checkpoint(target / "model")
    torch.save(
        {
            "update": update,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
        },
        target / "training_state.pt",
    )


def train_branch(
    branch: str,
    examples: Sequence[Example],
    scores_by_id: dict[str, StepScores],
    experiment: ExperimentConfig,
    warmup_checkpoint: str | Path,
    device: torch.device,
    updates: int | None = None,
    freeze_auxiliary_bottom: int = 0,
    save_checkpoints: bool = True,
    write_result: bool = True,
    max_seconds: float | None = None,
) -> TrainRunResult:
    if branch not in {"equal", "weighted"}:
        raise ValueError("branch must be 'equal' or 'weighted'")
    train_config = experiment.train
    requested_updates = updates if updates is not None else train_config.branch_updates
    seed_everything(train_config.seed)
    tokenizer = load_tokenizer(warmup_checkpoint)
    model = SimCoTModel.from_pretrained(warmup_checkpoint)
    if freeze_auxiliary_bottom:
        model.freeze_auxiliary_bottom(freeze_auxiliary_bottom)
    model.enable_gradient_checkpointing()
    model.to(device)
    model.train()

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scaler = _make_scaler(device)
    stream = _ordered_examples(examples, train_config.seed)
    branch_dir = Path(experiment.work_dir) / "branches" / branch
    branch_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(experiment.work_dir) / "logs" / f"{branch}.jsonl"
    if log_path.exists():
        log_path.unlink()

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    start_time = time.monotonic()
    final_loss = float("nan")
    completed_updates = 0
    last_saved_update = 0
    for update in range(1, requested_updates + 1):
        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        answer_total = 0.0
        step_total = 0.0
        for _ in range(train_config.gradient_accumulation_steps):
            example = next(stream)
            question_ids = encode_question(tokenizer, example, experiment.model).to(device)
            answer_ids = encode_answer(tokenizer, example, experiment.model).to(device)
            step_ids = [
                ids.to(device)
                for ids in encode_steps(tokenizer, example.observed_steps, experiment.model)
            ]
            if branch == "equal":
                weights = torch.ones(len(step_ids), device=device)
            else:
                score = scores_by_id[example.example_id]
                weights = torch.tensor(score.weights[: len(step_ids)], device=device)
            with _autocast(device):
                output = model(
                    question_ids=question_ids,
                    answer_ids=answer_ids,
                    step_ids=step_ids,
                    weights=weights,
                    lambda_step=train_config.lambda_step,
                )
                scaled_loss = output.loss / train_config.gradient_accumulation_steps
            if not torch.isfinite(output.loss):
                raise FloatingPointError(f"Non-finite {branch} loss at update {update}")
            scaler.scale(scaled_loss).backward()
            loss_total += float(output.loss.detach().item())
            answer_total += float(output.answer_loss.detach().item())
            step_total += float(output.step_loss.detach().item())

        scaler.unscale_(optimizer)
        clip_grad_norm_(parameters, train_config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        completed_updates = update

        divisor = train_config.gradient_accumulation_steps
        final_loss = loss_total / divisor
        if update == 1 or update % 10 == 0 or update == requested_updates:
            peak_gb = (
                torch.cuda.max_memory_reserved(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            )
            _append_jsonl(
                log_path,
                {
                    "phase": branch,
                    "update": update,
                    "loss": final_loss,
                    "answer_loss": answer_total / divisor,
                    "step_loss": step_total / divisor,
                    "peak_reserved_gb": peak_gb,
                    "elapsed_seconds": time.monotonic() - start_time,
                },
            )
        if save_checkpoints and (
            update % train_config.checkpoint_every == 0 or update == requested_updates
        ):
            _save_branch_state(model, optimizer, scaler, update, branch_dir)
            last_saved_update = update
        if max_seconds is not None and time.monotonic() - start_time >= max_seconds:
            break

    elapsed = time.monotonic() - start_time
    peak_gb = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    if save_checkpoints and last_saved_update != completed_updates:
        _save_branch_state(model, optimizer, scaler, completed_updates, branch_dir)

    result = TrainRunResult(
        branch=branch,
        requested_updates=requested_updates,
        completed_updates=completed_updates,
        elapsed_seconds=elapsed,
        peak_reserved_gb=peak_gb,
        final_loss=final_loss,
        checkpoint_dir=str(branch_dir / "model"),
        finite=bool(np.isfinite(final_loss)),
    )
    if write_result:
        result_path = Path(experiment.output_dir) / f"{branch}_run.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result
