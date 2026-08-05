from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ReliabilityOutput:
    validity_logits: torch.Tensor
    utility_logits: torch.Tensor

    @property
    def validity_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.validity_logits)

    @property
    def utility_probability(self) -> torch.Tensor:
        return torch.sigmoid(self.utility_logits)

    @property
    def reliability_probability(self) -> torch.Tensor:
        return self.validity_probability * self.utility_probability


class DualReliabilityHead(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        projection_dim: int = 128,
        shared_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if min(hidden_size, projection_dim, shared_hidden_dim) <= 0:
            raise ValueError("All reliability-head dimensions must be positive")
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.shared_hidden_dim = shared_hidden_dim
        self.latent_projection = nn.Linear(3 * hidden_size, projection_dim)
        self.semantic_projection = nn.Linear(hidden_size, projection_dim)
        self.latent_norm = nn.LayerNorm(projection_dim)
        self.semantic_norm = nn.LayerNorm(projection_dim)
        self.shared = nn.Sequential(
            nn.Linear(4 * projection_dim, shared_hidden_dim),
            nn.GELU(),
        )
        self.validity_head = nn.Linear(shared_hidden_dim, 1)
        self.utility_head = nn.Linear(shared_hidden_dim, 1)

    def forward(
        self,
        z_previous: torch.Tensor,
        z_current: torch.Tensor,
        e_step: torch.Tensor,
    ) -> ReliabilityOutput:
        if z_previous.shape != z_current.shape or z_current.shape != e_step.shape:
            raise ValueError("z_previous, z_current, and e_step must share shape")
        if z_current.ndim != 2 or z_current.shape[-1] != self.hidden_size:
            raise ValueError("Reliability features must have shape [batch, hidden_size]")
        latent_input = torch.cat(
            [z_previous, z_current, z_current - z_previous], dim=-1
        )
        a_step = self.latent_norm(self.latent_projection(latent_input))
        b_step = self.semantic_norm(self.semantic_projection(e_step))
        interaction = torch.cat(
            [a_step, b_step, torch.abs(a_step - b_step), a_step * b_step],
            dim=-1,
        )
        shared = self.shared(interaction)
        return ReliabilityOutput(
            validity_logits=self.validity_head(shared).squeeze(-1),
            utility_logits=self.utility_head(shared).squeeze(-1),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def masked_utility_bce(
    utility_logits: torch.Tensor,
    y_valid: torch.Tensor,
    y_utility: torch.Tensor,
) -> torch.Tensor:
    mask = y_valid.bool()
    if not torch.any(mask):
        return utility_logits.sum() * 0.0
    selected_targets = y_utility[mask]
    if torch.any((selected_targets != 0) & (selected_targets != 1)):
        raise ValueError("Utility targets for valid steps must be binary")
    return F.binary_cross_entropy_with_logits(
        utility_logits[mask], selected_targets.to(utility_logits.dtype)
    )


def paired_margin_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor,
    negative_indices: torch.Tensor,
    *,
    margin: float = 0.2,
) -> torch.Tensor:
    if positive_indices.shape != negative_indices.shape:
        raise ValueError("Positive and negative pair indices must share shape")
    if positive_indices.numel() == 0:
        return scores.sum() * 0.0
    return torch.relu(
        margin - scores[positive_indices] + scores[negative_indices]
    ).mean()


def reliability_head_loss(
    output: ReliabilityOutput,
    *,
    y_valid: torch.Tensor,
    y_utility: torch.Tensor,
    validity_positive_indices: torch.Tensor,
    validity_negative_indices: torch.Tensor,
    utility_positive_indices: torch.Tensor,
    utility_negative_indices: torch.Tensor,
    margin: float = 0.2,
) -> dict[str, torch.Tensor]:
    validity_bce = F.binary_cross_entropy_with_logits(
        output.validity_logits,
        y_valid.to(output.validity_logits.dtype),
    )
    utility_bce = masked_utility_bce(output.utility_logits, y_valid, y_utility)
    validity_rank = paired_margin_loss(
        output.validity_probability,
        validity_positive_indices,
        validity_negative_indices,
        margin=margin,
    )
    utility_rank = paired_margin_loss(
        output.utility_probability,
        utility_positive_indices,
        utility_negative_indices,
        margin=margin,
    )
    total = validity_bce + utility_bce + 0.5 * validity_rank + 0.5 * utility_rank
    return {
        "total": total,
        "validity_bce": validity_bce,
        "utility_bce": utility_bce,
        "validity_rank": validity_rank,
        "utility_rank": utility_rank,
    }
