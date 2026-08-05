from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable
import json
import os
import platform
import re
import subprocess
import sys


_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    target = Path(path)
    digest = sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(
    root: str | Path,
    *,
    excluded_names: Iterable[str] = (".git", "__pycache__"),
) -> tuple[str, int]:
    base = Path(root)
    excluded = set(excluded_names)
    files = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and not any(part in excluded for part in path.relative_to(base).parts)
    )
    digest = sha256()
    for path in files:
        relative = path.relative_to(base).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


def count_lines(path: str | Path) -> int:
    count = 0
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for _ in handle:
            count += 1
    return count


def resolve_workspace_path(project_root: str | Path, value: str) -> Path:
    root = Path(project_root).resolve()
    target = (root / value).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes project root: {value}")
    return target


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_text(mapping: dict[str, Any], key: str, section: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{section}.{key} must be a non-empty string")
    return value


def _validate_revision(value: str, name: str) -> None:
    if _HEX40.fullmatch(value.lower()) is None:
        raise ValueError(f"{name} must be a 40-character hexadecimal revision")


def _validate_sha256(value: str, name: str) -> None:
    if _HEX64.fullmatch(value.lower()) is None:
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ValueError("schema_version must equal 1")
    _require_text(config, "run_id", "config")
    if not isinstance(config.get("seed"), int) or config["seed"] < 0:
        raise ValueError("config.seed must be a non-negative integer")
    _require_text(config, "output_dir", "config")

    source = _require_mapping(config.get("official_source"), "official_source")
    _require_text(source, "repository_url", "official_source")
    _require_text(source, "repository_ref", "official_source")
    _require_text(source, "source_dir", "official_source")
    _require_text(source, "archive_path", "official_source")
    revision = _require_text(source, "repository_revision", "official_source")
    archive_sha = _require_text(source, "archive_sha256", "official_source")
    _validate_revision(revision, "official_source.repository_revision")
    _validate_sha256(archive_sha, "official_source.archive_sha256")

    checkpoint = _require_mapping(config.get("checkpoint"), "checkpoint")
    _require_text(checkpoint, "repo_id", "checkpoint")
    _require_text(checkpoint, "filename", "checkpoint")
    _require_text(checkpoint, "path", "checkpoint")
    _validate_revision(
        _require_text(checkpoint, "revision", "checkpoint"),
        "checkpoint.revision",
    )
    _validate_sha256(
        _require_text(checkpoint, "sha256", "checkpoint"),
        "checkpoint.sha256",
    )

    base_model = _require_mapping(config.get("base_model"), "base_model")
    _require_text(base_model, "repo_id", "base_model")
    _require_text(base_model, "path", "base_model")
    _validate_revision(
        _require_text(base_model, "revision", "base_model"),
        "base_model.revision",
    )
    base_files = _require_mapping(base_model.get("files"), "base_model.files")
    if not base_files:
        raise ValueError("base_model.files must not be empty")
    for name, expected_sha in base_files.items():
        if not isinstance(name, str) or not name:
            raise ValueError("base_model.files keys must be non-empty strings")
        if not isinstance(expected_sha, str):
            raise ValueError(f"base_model.files.{name} must be a SHA-256 string")
        _validate_sha256(expected_sha, f"base_model.files.{name}")

    dataset = _require_mapping(config.get("dataset"), "dataset")
    _require_text(dataset, "repository_url", "dataset")
    _require_text(dataset, "raw_dir", "dataset")
    _validate_revision(
        _require_text(dataset, "revision", "dataset"),
        "dataset.revision",
    )
    data_files = _require_mapping(dataset.get("files"), "dataset.files")
    if not data_files:
        raise ValueError("dataset.files must not be empty")
    for name, metadata in data_files.items():
        item = _require_mapping(metadata, f"dataset.files.{name}")
        _validate_sha256(
            _require_text(item, "sha256", f"dataset.files.{name}"),
            f"dataset.files.{name}.sha256",
        )
        if not isinstance(item.get("lines"), int) or item["lines"] <= 0:
            raise ValueError(f"dataset.files.{name}.lines must be positive")


def verify_file(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_lines: int | None = None,
) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(target)
    actual_sha = sha256_file(target)
    if actual_sha.lower() != expected_sha256.lower():
        raise ValueError(
            f"SHA-256 mismatch for {target}: expected {expected_sha256}, got {actual_sha}"
        )
    result: dict[str, Any] = {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": actual_sha,
    }
    if expected_lines is not None:
        actual_lines = count_lines(target)
        if actual_lines != expected_lines:
            raise ValueError(
                f"Line-count mismatch for {target}: expected {expected_lines}, got {actual_lines}"
            )
        result["lines"] = actual_lines
    return result


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "datasets",
                "accelerate",
                "huggingface-hub",
                "numpy",
                "scikit-learn",
            )
        },
    }
    try:
        import torch

        environment["torch_runtime"] = {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_count": torch.cuda.device_count(),
            "gpus": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - diagnostic fallback
        environment["torch_runtime_error"] = repr(exc)

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        environment["nvidia_smi"] = [
            line.strip() for line in completed.stdout.splitlines() if line.strip()
        ]
    except (OSError, subprocess.SubprocessError) as exc:
        environment["nvidia_smi_error"] = repr(exc)
    return environment


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip().lower()


def build_provenance_manifest(
    config_path: str | Path,
    *,
    project_root: str | Path,
) -> tuple[dict[str, Any], Path]:
    root = Path(project_root).resolve()
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    validate_config(config)

    source = config["official_source"]
    archive_path = resolve_workspace_path(root, source["archive_path"])
    source_dir = resolve_workspace_path(root, source["source_dir"])
    source_archive = verify_file(
        archive_path,
        expected_sha256=source["archive_sha256"],
    )
    source_tree_sha, source_tree_files = sha256_tree(source_dir)

    revision_evidence = None
    if source.get("revision_git_dir"):
        evidence_path = resolve_workspace_path(root, source["revision_git_dir"])
        revision_evidence = _git_head(evidence_path)
        if revision_evidence != source["repository_revision"].lower():
            raise ValueError(
                "Official source revision evidence does not match configured revision"
            )

    checkpoint = config["checkpoint"]
    checkpoint_verified = verify_file(
        resolve_workspace_path(root, checkpoint["path"]),
        expected_sha256=checkpoint["sha256"],
    )

    base_model = config["base_model"]
    base_dir = resolve_workspace_path(root, base_model["path"])
    base_files = {
        name: verify_file(base_dir / name, expected_sha256=expected_sha)
        for name, expected_sha in sorted(base_model["files"].items())
    }

    dataset = config["dataset"]
    data_dir = resolve_workspace_path(root, dataset["raw_dir"])
    data_files = {
        name: verify_file(
            data_dir / name,
            expected_sha256=metadata["sha256"],
            expected_lines=metadata["lines"],
        )
        for name, metadata in sorted(dataset["files"].items())
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": config["seed"],
        "config": {
            "path": os.path.relpath(config_file, root),
            "sha256": sha256_file(config_file),
        },
        "official_source": {
            "repository_url": source["repository_url"],
            "repository_ref": source["repository_ref"],
            "repository_revision": source["repository_revision"].lower(),
            "revision_evidence": revision_evidence,
            "acquisition_method": source.get("acquisition_method"),
            "archive": source_archive,
            "source_tree": {
                "path": str(source_dir),
                "sha256": source_tree_sha,
                "files": source_tree_files,
            },
        },
        "checkpoint": {
            "repo_id": checkpoint["repo_id"],
            "revision": checkpoint["revision"].lower(),
            "filename": checkpoint["filename"],
            "artifact": checkpoint_verified,
        },
        "base_model": {
            "repo_id": base_model["repo_id"],
            "revision": base_model["revision"].lower(),
            "files": base_files,
        },
        "dataset": {
            "repository_url": dataset["repository_url"],
            "revision": dataset["revision"].lower(),
            "files": data_files,
        },
        "environment": collect_environment(),
    }

    output_dir = resolve_workspace_path(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "provenance.json"
    temporary = output_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return manifest, output_path
