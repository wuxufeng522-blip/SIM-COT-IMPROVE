from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("Binary labels and scores must be aligned vectors")
    if labels.size == 0:
        return {"roc_auc": None, "pr_auc": None, "examples": 0, "positives": 0}
    if not np.isfinite(scores).all():
        raise ValueError("Detection scores contain NaN/Inf")
    unique = np.unique(labels)
    roc_auc = float(roc_auc_score(labels, scores)) if unique.size == 2 else None
    pr_auc = float(average_precision_score(labels, scores)) if unique.size == 2 else None
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "examples": int(labels.size),
        "positives": int(labels.sum()),
    }


def evaluate_detection(
    rows: list[dict[str, Any]],
    *,
    validity_scores: np.ndarray,
    utility_scores: np.ndarray,
) -> dict[str, Any]:
    if len(rows) != len(validity_scores) or len(rows) != len(utility_scores):
        raise ValueError("Rows and prediction arrays must align")
    validity_labels = np.asarray([row["y_valid"] for row in rows], dtype=np.int64)
    reliability_labels = np.asarray(
        [int(row["y_valid"] == 1 and row["y_utility"] == 1) for row in rows],
        dtype=np.int64,
    )
    valid_mask = validity_labels == 1
    utility_labels = np.asarray(
        [row["y_utility"] for row in rows if row["y_valid"] == 1],
        dtype=np.int64,
    )
    reliability_scores = validity_scores * utility_scores
    return {
        "validity": binary_metrics(validity_labels, validity_scores),
        "utility_within_valid": binary_metrics(
            utility_labels,
            utility_scores[valid_mask],
        ),
        "reliability": binary_metrics(reliability_labels, reliability_scores),
    }


def mean_equivalent_reliability_drop(
    rows: list[dict[str, Any]],
    reliability_scores: np.ndarray,
) -> dict[str, float | int | None]:
    if len(rows) != len(reliability_scores):
        raise ValueError("Rows and reliability scores must align")
    by_pair: dict[str, dict[str, list[float]]] = {}
    for row, score in zip(rows, reliability_scores, strict=True):
        if row["family"] not in {"clean_original", "equivalent_positive"}:
            continue
        by_pair.setdefault(row["pair_id"], {}).setdefault(row["family"], []).append(
            float(score)
        )
    drops = []
    for groups in by_pair.values():
        if "clean_original" in groups and "equivalent_positive" in groups:
            drops.append(
                float(np.mean(groups["clean_original"]))
                - float(np.mean(groups["equivalent_positive"]))
            )
    return {
        "pairs": len(drops),
        "mean_drop": float(np.mean(drops)) if drops else None,
    }
