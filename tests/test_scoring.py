import numpy as np
import pytest

from rsr_rd_simcot.scoring import compute_normalized_weights, validate_weights


def test_low_rsr_tail_and_high_rd_receive_less_weight() -> None:
    weights = compute_normalized_weights(
        rsr_by_example=[[0.1, 1.0, 2.0], [0.5, 1.5]],
        rd_by_example=[[1.4, 1.0, 0.6], [1.2, 0.8]],
    )
    validate_weights(weights)
    assert weights[0][0] < weights[0][1] < weights[0][2]
    assert weights[1][0] < weights[1][1]
    assert np.mean(weights[0]) == pytest.approx(1.0)


def test_non_positive_scores_are_rejected() -> None:
    with pytest.raises(ValueError):
        compute_normalized_weights([[1.0, 0.0]], [[1.0, 1.0]])
