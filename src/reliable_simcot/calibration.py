from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be positive and finite")
    return logits / temperature


def expected_calibration_error(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 10,
) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().float().cpu()
    if probabilities.numel() == 0 or probabilities.shape != targets.shape:
        raise ValueError("Probabilities and targets must be non-empty and aligned")
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    error = torch.tensor(0.0)
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = probabilities.ge(lower) & (
            probabilities.le(upper) if index == bins - 1 else probabilities.lt(upper)
        )
        if not torch.any(mask):
            continue
        confidence = probabilities[mask].mean()
        accuracy = targets[mask].mean()
        error += mask.float().mean() * torch.abs(confidence - accuracy)
    return float(error.item())


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    grid_points: int = 121,
) -> CalibrationResult:
    logits = logits.detach().float().cpu()
    targets = targets.detach().float().cpu()
    if logits.numel() == 0 or logits.shape != targets.shape:
        raise ValueError("Calibration logits and targets must be non-empty and aligned")
    if torch.unique(targets).numel() < 2:
        raise ValueError("Temperature fitting requires both target classes")
    candidates = torch.logspace(-1.5, 1.5, grid_points)
    losses = torch.stack(
        [F.binary_cross_entropy_with_logits(logits / value, targets) for value in candidates]
    )
    best_temperature = float(candidates[int(torch.argmin(losses).item())].item())
    before = torch.sigmoid(logits)
    after = torch.sigmoid(logits / best_temperature)
    return CalibrationResult(
        temperature=best_temperature,
        brier_before=float(torch.mean((before - targets) ** 2).item()),
        brier_after=float(torch.mean((after - targets) ** 2).item()),
        ece_before=expected_calibration_error(before, targets),
        ece_after=expected_calibration_error(after, targets),
    )
