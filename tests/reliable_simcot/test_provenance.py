from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pytest

from reliable_simcot.provenance import (
    count_lines,
    resolve_workspace_path,
    sha256_file,
    sha256_tree,
    validate_config,
    verify_file,
)


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"sim-cot\x00reliability\n")
    assert sha256_file(artifact) == sha256(artifact.read_bytes()).hexdigest()


def test_tree_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b.txt").write_text("b", encoding="utf-8")
    (root / "a.txt").write_text("a", encoding="utf-8")
    first, count = sha256_tree(root)
    second, second_count = sha256_tree(root)
    assert first == second
    assert count == second_count == 2

    (root / "a.txt").write_text("changed", encoding="utf-8")
    changed, _ = sha256_tree(root)
    assert changed != first


def test_resolve_workspace_path_rejects_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes project root"):
        resolve_workspace_path(tmp_path, "../outside.txt")


def test_verify_file_checks_hash_and_line_count(tmp_path: Path) -> None:
    artifact = tmp_path / "data.txt"
    artifact.write_text("one\ntwo\n", encoding="utf-8")
    expected = sha256_file(artifact)
    result = verify_file(artifact, expected_sha256=expected, expected_lines=2)
    assert result["lines"] == 2
    assert result["bytes"] == artifact.stat().st_size
    assert count_lines(artifact) == 2

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file(artifact, expected_sha256="0" * 64)


def test_validate_config_rejects_unpinned_revision() -> None:
    config = {
        "schema_version": 1,
        "run_id": "R001",
        "seed": 0,
        "output_dir": "outputs/run",
        "official_source": {
            "repository_url": "https://example.invalid/repo.git",
            "repository_ref": "main",
            "repository_revision": "main",
            "source_dir": "work/source",
            "archive_path": "work/source.zip",
            "archive_sha256": "0" * 64,
        },
        "checkpoint": {},
        "base_model": {},
        "dataset": {},
    }
    with pytest.raises(ValueError, match="40-character hexadecimal revision"):
        validate_config(config)
