import torch

from reliable_simcot.error_cancellation_experiment import (
    STAGE1_ARMS,
    create_training_schedule,
    variant_and_weights,
    weighted_step_mean,
)


def manifest_fixture():
    ids = [f"q{index}" for index in range(4)]
    variants = {
        "clean": {"steps": ["c"] * 5, "types": ["CLEAN"] * 5},
        "local_error": {
            "steps": ["l"] * 5,
            "types": ["CLEAN", "DIRECT_FALSE", "CANCEL_FALSE", "CLEAN", "CLEAN"],
        },
        "local_redundant": {
            "steps": ["r"] * 5,
            "types": ["CLEAN", "REDUNDANT", "REDUNDANT", "CLEAN", "CLEAN"],
        },
        "wide_error": {
            "steps": ["w"] * 5,
            "types": ["DIRECT_FALSE", "ERROR_DESCENDANT", "ERROR_DESCENDANT", "CANCEL_FALSE", "CLEAN"],
        },
        "wide_redundant": {
            "steps": ["z"] * 5,
            "types": ["REDUNDANT", "REDUNDANT", "REDUNDANT", "REDUNDANT", "CLEAN"],
        },
    }
    return {
        "manifest_sha256": "x" * 64,
        "entries": [{"question_id": value, "variants": variants} for value in ids],
        "coverage_masks": {"25": ["q0"], "50": ["q0", "q1"]},
    }


def test_stage1_matrix_is_frozen() -> None:
    assert STAGE1_ARMS == (
        "C", "RL25", "RL50", "EL25", "EL50", "RW25", "RW50", "EW25", "EW50"
    )


def test_arm_mapping_and_unnormalized_weight_mask() -> None:
    manifest = manifest_fixture()
    row = manifest["entries"][0]
    steps, weights, types = variant_and_weights(
        "EW50-w01", row, manifest, error_step_weight=0.1
    )
    assert steps == ("w",) * 5
    assert weights == (0.1, 0.1, 0.1, 0.1, 1.0)
    assert types[-1] == "CLEAN"
    clean_steps, clean_weights, _ = variant_and_weights(
        "EW50-w01", manifest["entries"][3], manifest, error_step_weight=0.1
    )
    assert clean_steps == ("c",) * 5
    assert clean_weights == (1.0,) * 5


def test_step_mean_does_not_depend_on_token_counts() -> None:
    losses = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    weights = torch.ones(5)
    result = weighted_step_mean(losses, weights)
    result.backward()
    assert result.item() == 3.0
    assert torch.allclose(losses.grad, torch.full((5,), 0.2))


def test_schedule_is_deterministic_and_shared_across_arms() -> None:
    manifest = manifest_fixture()
    first = create_training_schedule(manifest, seeds=[1], updates=2, accumulation=2)
    second = create_training_schedule(manifest, seeds=[1], updates=2, accumulation=2)
    assert first == second
    assert len(first["per_seed"]["1"]) == 4
