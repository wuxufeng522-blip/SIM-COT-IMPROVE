from __future__ import annotations

import random

from reliable_simcot.head_training import (
    balanced_classification_indices,
    paired_indices,
)


def _rows() -> list[dict]:
    rows = []
    for pair in range(8):
        pair_id = f"p{pair}"
        rows.extend(
            [
                {
                    "pair_id": pair_id,
                    "family": "clean_original",
                    "y_valid": 1,
                    "y_utility": 1,
                },
                {
                    "pair_id": pair_id,
                    "family": "numeric_error",
                    "y_valid": 0,
                    "y_utility": None,
                },
                {
                    "pair_id": pair_id,
                    "family": "redundant_repeat",
                    "y_valid": 1,
                    "y_utility": 0,
                },
            ]
        )
    return rows


def test_classification_batch_balances_both_tasks() -> None:
    rows = _rows()
    indices = balanced_classification_indices(
        rows,
        list(range(len(rows))),
        batch_size=16,
        rng=random.Random(3),
    )
    selected = [rows[index] for index in indices]
    assert sum(row["y_valid"] == 0 for row in selected) == 8
    valid = [row for row in selected if row["y_valid"] == 1]
    assert sum(row["y_utility"] == 1 for row in valid) == 4
    assert sum(row["y_utility"] == 0 for row in valid) == 4


def test_ranking_pairs_share_pair_id_and_registered_labels() -> None:
    rows = _rows()
    validity, utility = paired_indices(rows, range(len(rows)))
    assert len(validity) == 8
    assert len(utility) == 8
    for positive, negative in validity:
        assert rows[positive]["pair_id"] == rows[negative]["pair_id"]
        assert rows[positive]["y_valid"] == 1
        assert rows[negative]["y_valid"] == 0
    for positive, negative in utility:
        assert rows[positive]["pair_id"] == rows[negative]["pair_id"]
        assert rows[positive]["y_utility"] == 1
        assert rows[negative]["y_utility"] == 0
