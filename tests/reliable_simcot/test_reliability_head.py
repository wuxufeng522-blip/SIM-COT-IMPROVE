from __future__ import annotations

import torch

from reliable_simcot.reliability_head import (
    DualReliabilityHead,
    ReliabilityOutput,
    masked_utility_bce,
    paired_margin_loss,
    reliability_head_loss,
)


def test_dual_head_shapes_probabilities_and_parameter_budget() -> None:
    head = DualReliabilityHead(hidden_size=768)
    output = head(
        torch.zeros(4, 768),
        torch.randn(4, 768),
        torch.randn(4, 768),
    )
    assert output.validity_logits.shape == (4,)
    assert output.utility_logits.shape == (4,)
    assert torch.all((output.reliability_probability >= 0) & (output.reliability_probability <= 1))
    assert head.parameter_count <= 500_000


def test_masked_utility_loss_ignores_invalid_targets_and_gradients() -> None:
    logits = torch.tensor([0.2, -0.3, 0.5], requires_grad=True)
    y_valid = torch.tensor([1, 0, 1])
    y_utility = torch.tensor([1, -1, 0])
    loss = masked_utility_bce(logits, y_valid, y_utility)
    loss.backward()
    assert logits.grad is not None
    assert logits.grad[1].item() == 0.0
    assert logits.grad[0].item() != 0.0
    assert logits.grad[2].item() != 0.0


def test_pairwise_margin_uses_registered_direction() -> None:
    scores = torch.tensor([0.9, 0.2, 0.3, 0.4])
    passing = paired_margin_loss(
        scores,
        torch.tensor([0]),
        torch.tensor([1]),
        margin=0.2,
    )
    violating = paired_margin_loss(
        scores,
        torch.tensor([2]),
        torch.tensor([3]),
        margin=0.2,
    )
    assert passing.item() == 0.0
    assert torch.isclose(violating, torch.tensor(0.3))


def test_total_loss_matches_registered_coefficients() -> None:
    output = ReliabilityOutput(
        validity_logits=torch.tensor([2.0, -2.0, 1.0, 1.0]),
        utility_logits=torch.tensor([2.0, 0.0, -2.0, 1.0]),
    )
    losses = reliability_head_loss(
        output,
        y_valid=torch.tensor([1, 0, 1, 1]),
        y_utility=torch.tensor([1, -1, 0, 1]),
        validity_positive_indices=torch.tensor([0]),
        validity_negative_indices=torch.tensor([1]),
        utility_positive_indices=torch.tensor([3]),
        utility_negative_indices=torch.tensor([2]),
    )
    expected = (
        losses["validity_bce"]
        + losses["utility_bce"]
        + 0.5 * losses["validity_rank"]
        + 0.5 * losses["utility_rank"]
    )
    assert torch.equal(losses["total"], expected)
