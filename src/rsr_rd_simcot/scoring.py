from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import json
import math

import numpy as np

from .data import Example


@dataclass
class StepScores:
    example_id: str
    rsr: list[float]
    rd: list[float]
    weights: list[float]
    noisy_step_index: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def robust_z(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    denominator = max(1.4826 * mad, 1e-6)
    return (values - median) / denominator


def compute_normalized_weights(
    rsr_by_example: Sequence[Sequence[float]],
    rd_by_example: Sequence[Sequence[float]],
) -> list[list[float]]:
    if len(rsr_by_example) != len(rd_by_example):
        raise ValueError("RSR and RD example counts differ")

    flat_rsr = np.asarray([x for row in rsr_by_example for x in row], dtype=np.float64)
    flat_rd = np.asarray([x for row in rd_by_example for x in row], dtype=np.float64)
    if flat_rsr.size == 0 or np.any(flat_rsr <= 0) or np.any(flat_rd <= 0):
        raise ValueError("RSR and RD values must be finite and strictly positive")
    if not np.all(np.isfinite(flat_rsr)) or not np.all(np.isfinite(flat_rd)):
        raise ValueError("RSR and RD values must be finite")

    z_rsr = robust_z(np.log(flat_rsr))
    z_rd = robust_z(np.log(flat_rd))
    # RSR was proposed as a trajectory *suitability* score, not a correctness
    # probability.  On short corrupted steps, extremely low RSR can be caused
    # by very high surprisal and must not receive an unlimited reward.  Treat
    # only the robust low tail as an anomaly penalty, while RD keeps its
    # original monotonic direction (lower answer NLL ratio is better).
    rsr_low_tail_penalty = np.maximum(0.0, -z_rsr)
    utilities = np.clip(-0.5 * rsr_low_tail_penalty - 0.5 * z_rd, -2.0, 2.0)

    output: list[list[float]] = []
    cursor = 0
    for rsr_row, rd_row in zip(rsr_by_example, rd_by_example, strict=True):
        if len(rsr_row) != len(rd_row) or not rsr_row:
            raise ValueError("Every example needs matching non-empty RSR and RD rows")
        count = len(rsr_row)
        raw = np.exp(utilities[cursor : cursor + count])
        normalized = count * raw / raw.sum()
        output.append(normalized.astype(float).tolist())
        cursor += count

    return output


def validate_weights(weights: Sequence[Sequence[float]], tolerance: float = 1e-5) -> None:
    for index, row in enumerate(weights):
        values = np.asarray(row, dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(f"Non-finite or empty weights for example {index}")
        if abs(float(values.mean()) - 1.0) > tolerance:
            raise ValueError(f"Weights for example {index} do not have mean 1")


def _encode(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_length is not None:
        ids = ids[:max_length]
    return ids


def _answer_nll(model, tokenizer, context: str, answer: str, device) -> float:
    import torch
    import torch.nn.functional as F

    context_ids = _encode(tokenizer, context)
    answer_ids = _encode(tokenizer, f" {answer}")
    if not context_ids or not answer_ids:
        raise ValueError("Context and answer must tokenize to non-empty sequences")
    input_ids = torch.tensor([context_ids + answer_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits
    start = len(context_ids) - 1
    answer_logits = logits[:, start : start + len(answer_ids), :]
    targets = input_ids[:, len(context_ids) :]
    return float(
        F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            targets.reshape(-1),
            reduction="mean",
        ).item()
    )


def score_example(model, tokenizer, example: Example, device, rank_clip: int = 100) -> tuple[list[float], list[float]]:
    import torch
    import torch.nn.functional as F

    prompt = f"Question: {example.question}\nReasoning:\n"
    prompt_ids = _encode(tokenizer, prompt)
    step_segments = [_encode(tokenizer, f"{step}\n") for step in example.observed_steps]
    full_ids = list(prompt_ids)
    spans: list[tuple[int, int]] = []
    for segment in step_segments:
        start = len(full_ids)
        full_ids.extend(segment)
        spans.append((start, len(full_ids)))

    input_ids = torch.tensor([full_ids], device=device)
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]

    rsr_values: list[float] = []
    for start, end in spans:
        target_ids = input_ids[0, start:end]
        token_logits = logits[start - 1 : end - 1]
        target_logits = token_logits.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        ranks = 1 + (token_logits > target_logits.unsqueeze(1)).sum(dim=1)
        surprisals = F.cross_entropy(token_logits, target_ids, reduction="none")
        numerator = torch.clamp(ranks, max=rank_clip).float().sum()
        denominator = surprisals.sum().clamp_min(1e-8)
        rsr_values.append(float((numerator / denominator).item()))

    rd_values: list[float] = []
    prefix_steps: list[str] = []
    base_context = f"Question: {example.question}\nReasoning:\n"
    previous_nll = _answer_nll(
        model, tokenizer, base_context + "Answer:", example.answer, device
    )
    for step in example.observed_steps:
        prefix_steps.append(step)
        context = base_context + "\n".join(prefix_steps) + "\nAnswer:"
        current_nll = _answer_nll(model, tokenizer, context, example.answer, device)
        rd_values.append(float(math.exp(np.clip(current_nll - previous_nll, -20, 20))))
        previous_nll = current_nll

    return rsr_values, rd_values


def score_dataset(model, tokenizer, examples: Sequence[Example], device, rank_clip: int = 100) -> list[StepScores]:
    rsr_rows: list[list[float]] = []
    rd_rows: list[list[float]] = []
    total = len(examples)
    for index, example in enumerate(examples, start=1):
        rsr, rd = score_example(model, tokenizer, example, device, rank_clip=rank_clip)
        rsr_rows.append(rsr)
        rd_rows.append(rd)
        if index == 1 or index % 100 == 0 or index == total:
            print(f"offline scoring: {index}/{total}", flush=True)

    weight_rows = compute_normalized_weights(rsr_rows, rd_rows)
    validate_weights(weight_rows)
    return [
        StepScores(
            example_id=example.example_id,
            rsr=rsr,
            rd=rd,
            weights=weights,
            noisy_step_index=example.noisy_step_index,
        )
        for example, rsr, rd, weights in zip(
            examples, rsr_rows, rd_rows, weight_rows, strict=True
        )
    ]


def save_scores(path: str | Path, scores: Sequence[StepScores]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for score in scores:
            handle.write(json.dumps(score.to_dict(), ensure_ascii=False) + "\n")


def load_scores(path: str | Path) -> dict[str, StepScores]:
    output: dict[str, StepScores] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = StepScores(**json.loads(line))
                output[value.example_id] = value
    return output
