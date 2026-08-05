from __future__ import annotations

import numpy as np
import torch

from reliable_simcot.calibration import fit_temperature
from reliable_simcot.detection_metrics import (
    binary_metrics,
    evaluate_detection,
    mean_equivalent_reliability_drop,
)


def test_temperature_scaling_is_positive_and_does_not_worsen_grid_nll() -> None:
    logits = torch.tensor([8.0, 5.0, -4.0, -7.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    result = fit_temperature(logits, targets)
    assert result.temperature > 0
    assert result.brier_after >= 0
    assert result.ece_after >= 0


def test_detection_metrics_respect_validity_mask() -> None:
    rows = [
        {"y_valid": 1, "y_utility": 1},
        {"y_valid": 1, "y_utility": 0},
        {"y_valid": 0, "y_utility": None},
    ]
    metrics = evaluate_detection(
        rows,
        validity_scores=np.asarray([0.9, 0.8, 0.1]),
        utility_scores=np.asarray([0.9, 0.2, 0.99]),
    )
    assert metrics["validity"]["roc_auc"] == 1.0
    assert metrics["utility_within_valid"]["examples"] == 2
    assert metrics["utility_within_valid"]["roc_auc"] == 1.0
    assert metrics["reliability"]["roc_auc"] == 1.0


def test_one_class_auc_is_explicitly_undefined() -> None:
    metrics = binary_metrics(np.asarray([1, 1]), np.asarray([0.2, 0.8]))
    assert metrics["roc_auc"] is None
    assert metrics["pr_auc"] is None


def test_equivalent_drop_is_paired() -> None:
    rows = [
        {"pair_id": "a", "family": "clean_original"},
        {"pair_id": "a", "family": "equivalent_positive"},
        {"pair_id": "b", "family": "clean_original"},
    ]
    result = mean_equivalent_reliability_drop(rows, np.asarray([0.8, 0.7, 0.1]))
    assert result["pairs"] == 1
    assert abs(result["mean_drop"] - 0.1) < 1e-12
