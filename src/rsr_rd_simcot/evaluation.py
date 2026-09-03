from __future__ import annotations

from pathlib import Path
from typing import Sequence
import json
import math
import re

import numpy as np
import torch

from .config import ExperimentConfig
from .data import Example
from .model import SimCoTModel
from .training import encode_answer, encode_question, encode_steps, load_tokenizer


_INTEGER_PATTERN = re.compile(r"-?\d+")


def _extract_last_integer(text: str) -> str | None:
    matches = _INTEGER_PATTERN.findall(text)
    return matches[-1] if matches else None


@torch.inference_mode()
def evaluate_branch(
    branch: str,
    examples: Sequence[Example],
    experiment: ExperimentConfig,
    checkpoint_dir: str | Path,
    device: torch.device,
) -> dict:
    tokenizer = load_tokenizer(experiment.model.model_name)
    model = SimCoTModel.from_checkpoint(checkpoint_dir).to(device)
    model.eval()

    answer_exact = 0
    answer_losses: list[float] = []
    step_losses: list[float] = []
    correct_step_tokens = 0
    total_step_tokens = 0
    exact_steps = 0
    total_steps = 0
    latent_distances: list[float] = []

    total_examples = len(examples)
    for example_index, example in enumerate(examples, start=1):
        question_ids = encode_question(tokenizer, example, experiment.model).to(device)
        answer_ids = encode_answer(tokenizer, example, experiment.model).to(device)
        clean_step_ids = [
            ids.to(device)
            for ids in encode_steps(tokenizer, example.clean_steps, experiment.model)
        ]
        output = model(
            question_ids=question_ids,
            answer_ids=answer_ids,
            step_ids=clean_step_ids,
            weights=torch.ones(len(clean_step_ids), device=device),
            lambda_step=experiment.train.lambda_step,
        )
        answer_losses.append(float(output.answer_loss.item()))
        step_losses.extend(float(value) for value in output.step_losses.tolist())

        for index, ids in enumerate(clean_step_ids):
            latent = output.latents[:, index, :]
            step_embeddings = model.auxiliary.transformer.wte(ids.unsqueeze(0))
            decoder_embeddings = torch.cat((latent.unsqueeze(1), step_embeddings), dim=1)
            logits = model.auxiliary(inputs_embeds=decoder_embeddings, use_cache=False).logits
            predictions = logits[:, : ids.numel(), :].argmax(dim=-1).squeeze(0)
            token_matches = predictions.eq(ids)
            correct_step_tokens += int(token_matches.sum().item())
            total_step_tokens += ids.numel()
            exact_steps += int(bool(token_matches.all().item()))
            total_steps += 1

        if output.latents.size(1) > 1:
            points = output.latents.squeeze(0).float()
            distances = torch.pdist(points, p=2)
            latent_distances.append(float(distances.mean().item()))

        generated_ids = model.generate_answer(
            question_ids=question_ids,
            latent_steps=len(clean_step_ids),
            max_new_tokens=experiment.model.max_answer_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        answer_exact += int(_extract_last_integer(generated_text) == example.answer)
        if example_index == 1 or example_index % 25 == 0 or example_index == total_examples:
            print(
                f"evaluation {branch}: {example_index}/{total_examples}",
                flush=True,
            )

    metrics = {
        "branch": branch,
        "examples": len(examples),
        "answer_exact_match": answer_exact / len(examples),
        "answer_nll": float(np.mean(answer_losses)),
        "clean_step_nll": float(np.mean(step_losses)),
        "clean_step_ppl": float(math.exp(min(float(np.mean(step_losses)), 20.0))),
        "step_token_accuracy": correct_step_tokens / max(total_step_tokens, 1),
        "step_sequence_exact_match": exact_steps / max(total_steps, 1),
        "mean_pairwise_latent_l2": float(np.mean(latent_distances)),
    }
    target = Path(experiment.output_dir) / f"{branch}_metrics.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return metrics
