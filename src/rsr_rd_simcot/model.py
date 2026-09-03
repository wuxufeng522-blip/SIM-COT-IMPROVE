from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel


@dataclass
class SimCoTOutput:
    loss: torch.Tensor
    answer_loss: torch.Tensor
    step_loss: torch.Tensor
    step_losses: torch.Tensor
    latents: torch.Tensor


class SimCoTModel(nn.Module):
    """Minimal batch-size-one SIM-CoT model for the controlled PoC."""

    def __init__(self, student: GPT2LMHeadModel, auxiliary: GPT2LMHeadModel) -> None:
        super().__init__()
        self.student = student
        self.auxiliary = auxiliary
        # Share input token embeddings as specified; keep separate LM heads.
        self.auxiliary.transformer.wte = self.student.transformer.wte
        self.student.config.use_cache = False
        self.auxiliary.config.use_cache = False

    @classmethod
    def from_pretrained(cls, model_name_or_path: str | Path) -> "SimCoTModel":
        student = GPT2LMHeadModel.from_pretrained(str(model_name_or_path))
        auxiliary = GPT2LMHeadModel.from_pretrained(str(model_name_or_path))
        return cls(student, auxiliary)

    @classmethod
    def from_config(cls, config: GPT2Config) -> "SimCoTModel":
        return cls(GPT2LMHeadModel(config), GPT2LMHeadModel(config))

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: str | Path) -> "SimCoTModel":
        root = Path(checkpoint_dir)
        student = GPT2LMHeadModel.from_pretrained(root / "student")
        auxiliary = GPT2LMHeadModel.from_pretrained(root / "auxiliary")
        return cls(student, auxiliary)

    def enable_gradient_checkpointing(self) -> None:
        self.student.gradient_checkpointing_enable()
        self.auxiliary.gradient_checkpointing_enable()

    def freeze_auxiliary_bottom(self, layer_count: int = 8) -> None:
        blocks = self.auxiliary.transformer.h
        if not 0 <= layer_count <= len(blocks):
            raise ValueError(f"Cannot freeze {layer_count} of {len(blocks)} layers")
        for block in blocks[:layer_count]:
            for parameter in block.parameters():
                parameter.requires_grad = False

    def _latent_chain(
        self, question_ids: torch.Tensor, latent_steps: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if question_ids.ndim == 1:
            question_ids = question_ids.unsqueeze(0)
        if question_ids.ndim != 2 or question_ids.size(0) != 1:
            raise ValueError("This PoC model intentionally supports batch size 1")
        if latent_steps < 1:
            raise ValueError("At least one latent step is required")

        embeddings = self.student.transformer.wte(question_ids)
        latents: list[torch.Tensor] = []
        for _ in range(latent_steps):
            outputs = self.student.transformer(
                inputs_embeds=embeddings,
                use_cache=False,
                return_dict=True,
            )
            latent = outputs.last_hidden_state[:, -1, :]
            latents.append(latent)
            embeddings = torch.cat((embeddings, latent.unsqueeze(1)), dim=1)
        return torch.stack(latents, dim=1), embeddings

    def _answer_loss(
        self, prefix_embeddings: torch.Tensor, answer_ids: torch.Tensor
    ) -> torch.Tensor:
        if answer_ids.ndim == 1:
            answer_ids = answer_ids.unsqueeze(0)
        answer_embeddings = self.student.transformer.wte(answer_ids)
        full_embeddings = torch.cat((prefix_embeddings, answer_embeddings), dim=1)
        logits = self.student(inputs_embeds=full_embeddings, use_cache=False).logits
        prefix_length = prefix_embeddings.size(1)
        answer_length = answer_ids.size(1)
        answer_logits = logits[:, prefix_length - 1 : prefix_length - 1 + answer_length, :]
        return F.cross_entropy(
            answer_logits.reshape(-1, answer_logits.size(-1)),
            answer_ids.reshape(-1),
            reduction="mean",
        )

    def _individual_step_loss(
        self, latent: torch.Tensor, step_ids: torch.Tensor
    ) -> torch.Tensor:
        if step_ids.ndim == 1:
            step_ids = step_ids.unsqueeze(0)
        step_embeddings = self.auxiliary.transformer.wte(step_ids)
        decoder_embeddings = torch.cat((latent.unsqueeze(1), step_embeddings), dim=1)
        logits = self.auxiliary(inputs_embeds=decoder_embeddings, use_cache=False).logits
        step_logits = logits[:, : step_ids.size(1), :]
        return F.cross_entropy(
            step_logits.reshape(-1, step_logits.size(-1)),
            step_ids.reshape(-1),
            reduction="mean",
        )

    def forward(
        self,
        question_ids: torch.Tensor,
        answer_ids: torch.Tensor,
        step_ids: Sequence[torch.Tensor],
        weights: torch.Tensor | None = None,
        lambda_step: float = 1.0,
    ) -> SimCoTOutput:
        if not step_ids:
            raise ValueError("step_ids cannot be empty")
        latents, prefix_embeddings = self._latent_chain(question_ids, len(step_ids))
        answer_loss = self._answer_loss(prefix_embeddings, answer_ids)
        step_losses = torch.stack(
            [
                self._individual_step_loss(latents[:, index, :], ids)
                for index, ids in enumerate(step_ids)
            ]
        )
        if weights is None:
            weights = torch.ones_like(step_losses)
        else:
            weights = weights.to(device=step_losses.device, dtype=step_losses.dtype)
        if weights.shape != step_losses.shape:
            raise ValueError(
                f"weights shape {tuple(weights.shape)} != step losses {tuple(step_losses.shape)}"
            )
        step_loss = (weights * step_losses).mean()
        total_loss = answer_loss + lambda_step * step_loss
        return SimCoTOutput(
            loss=total_loss,
            answer_loss=answer_loss,
            step_loss=step_loss,
            step_losses=step_losses,
            latents=latents,
        )

    @torch.inference_mode()
    def generate_answer(
        self,
        question_ids: torch.Tensor,
        latent_steps: int,
        max_new_tokens: int,
        eos_token_id: int,
    ) -> torch.Tensor:
        self.eval()
        _, prefix_embeddings = self._latent_chain(question_ids, latent_steps)
        outputs = self.student(
            inputs_embeds=prefix_embeddings,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]
        generated: list[torch.Tensor] = []
        for _ in range(max_new_tokens):
            next_token = next_logits.argmax(dim=-1)
            generated.append(next_token)
            if int(next_token.item()) == eos_token_id:
                break
            outputs = self.student(
                input_ids=next_token.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
        return torch.stack(generated, dim=1) if generated else question_ids.new_empty((1, 0))

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        (target / "student").mkdir(parents=True, exist_ok=True)
        (target / "auxiliary").mkdir(parents=True, exist_ok=True)
        self.student.save_pretrained(target / "student", safe_serialization=True)
        self.auxiliary.save_pretrained(target / "auxiliary", safe_serialization=True)
