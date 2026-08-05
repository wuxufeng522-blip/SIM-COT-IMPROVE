from __future__ import annotations

import pytest
import torch

from reliable_simcot.features import _existing_shards, feature_cache_key, masked_mean_pool


def test_masked_mean_pool_excludes_padding() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [999.0, 999.0]],
            [[2.0, 4.0], [888.0, 888.0], [777.0, 777.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = masked_mean_pool(hidden, mask)
    assert torch.equal(pooled, torch.tensor([[2.0, 4.0], [2.0, 4.0]]))


def test_masked_mean_pool_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one"):
        masked_mean_pool(torch.zeros(1, 2, 3), torch.zeros(1, 2))


def test_feature_cache_key_binds_model_data_and_config() -> None:
    base = feature_cache_key(
        dataset_manifest_sha256="data",
        checkpoint_sha256="model",
        tokenizer_sha256="tokenizer",
        extractor_config={"pooling": "mean", "latent_stage": 5},
    )
    reordered = feature_cache_key(
        dataset_manifest_sha256="data",
        checkpoint_sha256="model",
        tokenizer_sha256="tokenizer",
        extractor_config={"latent_stage": 5, "pooling": "mean"},
    )
    changed = feature_cache_key(
        dataset_manifest_sha256="data",
        checkpoint_sha256="different-model",
        tokenizer_sha256="tokenizer",
        extractor_config={"pooling": "mean", "latent_stage": 5},
    )
    assert base == reordered
    assert base != changed


def test_existing_shards_recover_ids_and_reject_duplicates(tmp_path) -> None:
    torch.save(
        [{"variant_id": "a", "family": "clean_original"}],
        tmp_path / "features-00000.pt",
    )
    torch.save(
        [{"variant_id": "b", "family": "numeric_error"}],
        tmp_path / "features-00001.pt",
    )
    shards, variant_ids, family_counts = _existing_shards(tmp_path)
    assert len(shards) == 2
    assert variant_ids == {"a", "b"}
    assert family_counts == {"clean_original": 1, "numeric_error": 1}

    torch.save(
        [{"variant_id": "a", "family": "clean_original"}],
        tmp_path / "features-00002.pt",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        _existing_shards(tmp_path)
