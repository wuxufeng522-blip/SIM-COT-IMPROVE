from __future__ import annotations

import pytest
import torch

from reliable_simcot.gradient_leverage import (
    ARM_SETTINGS,
    LEVERAGE_ARMS,
    _layer_metrics,
    verify_confirm_manifest,
)


def test_five_preregistered_arms_have_expected_targets_and_scales() -> None:
    assert tuple(ARM_SETTINGS) == LEVERAGE_ARMS
    assert ARM_SETTINGS == {
        "answer_only": ("clean", 0.0),
        "clean_aux1": ("clean", 1.0),
        "causal_aux1": ("noisy_equal", 1.0),
        "clean_aux3": ("clean", 3.0),
        "causal_aux3": ("noisy_equal", 3.0),
    }


def test_layer_metrics_detect_opposite_noisy_gradient() -> None:
    answer = [torch.tensor([3.0, 4.0])]
    clean = [torch.tensor([3.0, 4.0])]
    noisy = [torch.tensor([-3.0, -4.0])]
    metrics = _layer_metrics(answer, clean, noisy)
    assert metrics["answer_norm"] == pytest.approx(5.0)
    assert metrics["clean_aux_to_answer_norm_ratio"] == pytest.approx(1.0)
    assert metrics["clean_aux_answer_cosine"] == pytest.approx(1.0)
    assert metrics["noisy_aux_answer_cosine"] == pytest.approx(-1.0)
    assert metrics["clean_noisy_aux_cosine"] == pytest.approx(-1.0)
    assert metrics["noisy_minus_clean_answer_projection"] == pytest.approx(-2.0)


def test_confirm_manifest_hash_rejects_mutation() -> None:
    from reliable_simcot.gradient_leverage import _canonical_hash

    manifest = {"schema_version": 1, "examples": 2}
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    verify_confirm_manifest(manifest)
    manifest["examples"] = 3
    with pytest.raises(ValueError, match="mismatch"):
        verify_confirm_manifest(manifest)
