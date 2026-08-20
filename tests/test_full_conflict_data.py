from __future__ import annotations

import pytest

from reliable_simcot.full_conflict_data import canonical_hash, verify_split_manifest


def _manifest() -> dict:
    payload = {
        "train_entries": [{"question_id": "a" * 64}],
        "confirm_entries": [{"question_id": "b" * 64}],
        "primary_generation_question_ids": ["a" * 64],
        "reserve_generation_question_ids": [],
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    return payload


def test_split_manifest_accepts_disjoint_frozen_sets() -> None:
    verify_split_manifest(_manifest())


def test_split_manifest_hash_rejects_mutation() -> None:
    manifest = _manifest()
    manifest["confirm_entries"][0]["question_id"] = "c" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        verify_split_manifest(manifest)


def test_split_manifest_rejects_leakage() -> None:
    manifest = _manifest()
    manifest["confirm_entries"][0]["question_id"] = "a" * 64
    manifest["manifest_sha256"] = canonical_hash(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    with pytest.raises(ValueError, match="leakage"):
        verify_split_manifest(manifest)
