from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import math
import re
import time

import torch

from .audit import sha256_file
from .official_adapter import OfficialExample, load_official_model
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


def masked_mean_pool(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [batch, tokens, hidden]")
    if mask.ndim != 2 or mask.shape != hidden_states.shape[:2]:
        raise ValueError("mask must have shape [batch, tokens]")
    weights = mask.to(device=hidden_states.device, dtype=hidden_states.dtype)
    counts = weights.sum(dim=1, keepdim=True)
    if torch.any(counts <= 0):
        raise ValueError("Every pooled sequence must contain at least one valid token")
    return (hidden_states * weights.unsqueeze(-1)).sum(dim=1) / counts


def feature_cache_key(
    *,
    dataset_manifest_sha256: str,
    checkpoint_sha256: str,
    tokenizer_sha256: str,
    extractor_config: dict[str, Any],
) -> str:
    payload = json.dumps(
        {
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "extractor_config": extractor_config,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _load_development_rows(
    dataset_manifest: dict[str, Any],
    *,
    splits: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in splits:
        descriptor = dataset_manifest["split_files"][split]
        path = Path(descriptor["path"])
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"Reliability-data split SHA-256 mismatch: {split}")
        split_rows = [
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        ]
        if len(split_rows) != descriptor["rows"]:
            raise ValueError(f"Reliability-data row count mismatch: {split}")
        rows.extend(split_rows)
    return rows


def _trajectory_from_rows(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    furthest = max(rows, key=lambda row: row["step_index"])
    trajectory = tuple(furthest["prefix_steps"]) + (furthest["clean_step"],)
    for row in rows:
        index = row["step_index"]
        if index >= len(trajectory) or trajectory[index] != row["clean_step"]:
            raise ValueError("Rows disagree on the clean trajectory")
        if tuple(row["prefix_steps"]) != trajectory[:index]:
            raise ValueError("Rows disagree on prefix_steps")
    return trajectory


def _surface_features(text: str, tokenizer) -> dict[str, int]:
    return {
        "token_count": len(tokenizer.encode(text, add_special_tokens=False)),
        "character_count": len(text),
        "digit_count": sum(character.isdigit() for character in text),
        "numeric_span_count": len(re.findall(r"[+\-]?\d+(?:\.\d+)?", text)),
        "punctuation_count": sum(not character.isalnum() and not character.isspace() for character in text),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for attempt in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.05 * (attempt + 1))


def _save_shard(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(records, temporary)
    temporary.replace(path)
    return {
        "path": str(path),
        "rows": len(records),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _existing_shards(cache_dir: Path) -> tuple[list[dict[str, Any]], set[str], Counter[str]]:
    descriptors: list[dict[str, Any]] = []
    variant_ids: set[str] = set()
    family_counts: Counter[str] = Counter()
    for path in sorted(cache_dir.glob("features-*.pt")):
        records = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(records, list) or not records:
            raise ValueError(f"Feature shard is empty or malformed: {path}")
        for record in records:
            variant_id = record["variant_id"]
            if variant_id in variant_ids:
                raise ValueError(f"Duplicate cached variant ID: {variant_id}")
            variant_ids.add(variant_id)
            family_counts[record["family"]] += 1
        descriptors.append(
            {
                "path": str(path),
                "rows": len(records),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return descriptors, variant_ids, family_counts


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def extract_feature_cache(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    max_rows: int | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def project_path(value: str) -> Path:
        target = (root / value).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes project root: {value}")
        return target

    dataset_manifest_path = project_path(config["dataset_manifest_path"])
    if sha256_file(dataset_manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Reliability-data manifest SHA-256 mismatch")
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("gate_passed") is not True:
        raise ValueError("R030 must pass before feature extraction")
    checkpoint_path = project_path(config["checkpoint_path"])
    if sha256_file(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("Feature-source checkpoint SHA-256 mismatch")
    tokenizer_path = project_path(config["tokenizer_path"])
    if sha256_file(tokenizer_path) != config["tokenizer_sha256"]:
        raise ValueError("Tokenizer SHA-256 mismatch")

    extractor_settings = {
        "latent_stage": config["latent_stage"],
        "c_thought": config["c_thought"],
        "precision": config["precision"],
        "feature_dtype": config["feature_dtype"],
        "pooling": "mask_aware_mean_last_auxiliary_hidden",
        "z_pooling": "mean_over_c_thought_tokens",
        "step_index_mapping": "min(step_index, latent_stage-1)",
        "splits": config["splits"],
    }
    cache_key = feature_cache_key(
        dataset_manifest_sha256=config["dataset_manifest_sha256"],
        checkpoint_sha256=config["checkpoint_sha256"],
        tokenizer_sha256=config["tokenizer_sha256"],
        extractor_config=extractor_settings,
    )
    mode_name = "full" if max_rows is None else f"sanity-{max_rows}"
    cache_dir = project_path(config["cache_root"]) / cache_key / mode_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / f"progress_{mode_name}.json"

    rows = _load_development_rows(dataset_manifest, splits=config["splits"])
    rows.sort(key=lambda row: (row["question_id"], row["step_index"], row["variant_id"]))
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        rows = rows[:max_rows]
    if not rows:
        raise ValueError("No reliability rows selected")

    target_rows = list(rows)
    shards, cached_variant_ids, family_counts = _existing_shards(cache_dir)
    resumed_rows = len(cached_variant_ids)
    expected_variant_ids = {row["variant_id"] for row in target_rows}
    unexpected_cached = cached_variant_ids - expected_variant_ids
    if unexpected_cached:
        raise ValueError("Feature cache contains rows outside the selected data")
    rows = [row for row in target_rows if row["variant_id"] not in cached_variant_ids]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["question_id"]].append(row)

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported")
    dtype = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[config["precision"]]
    feature_dtype = {"float16": torch.float16, "float32": torch.float32}[
        config["feature_dtype"]
    ]
    torch.cuda.empty_cache()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    _atomic_json(
        progress_path,
        {
            "run_id": config["run_id"],
            "mode": mode_name,
            "phase": "loading_model",
            "completed_rows": len(cached_variant_ids),
            "target_rows": len(target_rows),
            "resumed_shards": len(shards),
        },
    )
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=project_path(config["official_source_dir"]),
        base_model_dir=project_path(config["base_model_dir"]),
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=dtype,
        move_auxiliary_to_device=True,
    )
    model.base_causallm.eval()
    model.expainable_llm.eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Frozen feature source still has trainable parameters")

    shard_size = config["shard_rows"]
    records: list[dict[str, Any]] = []
    completed = resumed_rows
    started = time.perf_counter()
    hidden_size: int | None = None

    with torch.inference_mode():
        for question_index, (qid, question_rows) in enumerate(sorted(grouped.items())):
            trajectory = _trajectory_from_rows(question_rows)
            representative = question_rows[0]
            example = OfficialExample(
                idx=representative["source_index"],
                question=representative["question"],
                steps=trajectory,
                answer=representative["answer"],
            )
            encoded = encode_smoke_example(
                example,
                tokenizer,
                token_ids,
                latent_stage=config["latent_stage"],
                c_thought=config["c_thought"],
            )
            batch = tensorize_smoke_example(encoded, device=device)
            batch.pop("explainable_ids_list")
            with _autocast(device, config["precision"]):
                base_output = model(**batch)
            latent_positions = batch["input_ids"][0].eq(token_ids["<|latent|>"]).nonzero(
                as_tuple=True
            )[0]
            expected_latents = config["latent_stage"] * config["c_thought"]
            if latent_positions.numel() != expected_latents:
                raise ValueError("Unexpected number of latent positions")
            latent_tokens = base_output.inputs_embeds[0, latent_positions, :]
            latent_groups = latent_tokens.view(
                config["latent_stage"], config["c_thought"], -1
            )
            z_groups = latent_groups.mean(dim=1)
            hidden_size = int(z_groups.shape[-1])

            for row in question_rows:
                group_index = min(row["step_index"], config["latent_stage"] - 1)
                z_current = z_groups[group_index]
                z_previous = (
                    torch.zeros_like(z_current)
                    if group_index == 0
                    else z_groups[group_index - 1]
                )
                step_ids = tokenizer.encode(
                    row["candidate_step"], add_special_tokens=False
                )
                if not step_ids:
                    raise ValueError("Candidate step has an empty token mask")
                ids = torch.tensor(
                    step_ids + [tokenizer.eos_token_id],
                    dtype=torch.long,
                    device=device,
                )
                token_embeds = model.embedding(ids)
                aux_inputs = torch.cat(
                    [latent_groups[group_index], token_embeds], dim=0
                ).unsqueeze(0)
                attention_mask = torch.ones(
                    aux_inputs.shape[:2], dtype=torch.long, device=device
                )
                position_ids = torch.arange(
                    1,
                    aux_inputs.shape[1] + 1,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                with _autocast(device, config["precision"]):
                    auxiliary = model.expainable_llm(
                        inputs_embeds=aux_inputs,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                token_hidden = auxiliary.hidden_states[-1][
                    :,
                    config["c_thought"] : config["c_thought"] + len(step_ids),
                    :,
                ]
                token_mask = torch.ones(
                    (1, len(step_ids)), dtype=torch.bool, device=device
                )
                e_step = masked_mean_pool(token_hidden, token_mask)[0]
                finite = all(
                    torch.isfinite(tensor).all().item()
                    for tensor in (z_previous, z_current, e_step)
                )
                if not finite:
                    raise FloatingPointError("Non-finite reliability feature")
                record = {
                    "variant_id": row["variant_id"],
                    "question_id": qid,
                    "split": row["split"],
                    "family": row["family"],
                    "template_id": row["template_id"],
                    "pair_id": row["pair_id"],
                    "step_index": row["step_index"],
                    "latent_group_index": group_index,
                    "y_valid": row["y_valid"],
                    "y_utility": row["y_utility"],
                    "y_reliable": row["y_reliable"],
                    "surface": _surface_features(row["candidate_step"], tokenizer),
                    "z_previous": z_previous.detach().to("cpu", dtype=feature_dtype),
                    "z_current": z_current.detach().to("cpu", dtype=feature_dtype),
                    "e_step": e_step.detach().to("cpu", dtype=feature_dtype),
                }
                records.append(record)
                family_counts[row["family"]] += 1
                completed += 1
                if len(records) == shard_size:
                    shard_path = cache_dir / f"features-{len(shards):05d}.pt"
                    shards.append(_save_shard(shard_path, records))
                    records = []
                if completed == 1 or completed % config["progress_every"] == 0:
                    _atomic_json(
                        progress_path,
                        {
                            "run_id": config["run_id"],
                            "mode": mode_name,
                            "phase": "extracting",
                            "completed_rows": completed,
                            "target_rows": len(target_rows),
                            "completed_questions": question_index + 1,
                            "elapsed_seconds": time.perf_counter() - started,
                        },
                    )
            del base_output, batch, latent_tokens, latent_groups, z_groups

    if records:
        shard_path = cache_dir / f"features-{len(shards):05d}.pt"
        shards.append(_save_shard(shard_path, records))
    elapsed = time.perf_counter() - started
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
    )
    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "PASS",
        "mode": mode_name,
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "dataset_manifest_path": str(dataset_manifest_path),
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "extractor_settings": extractor_settings,
        "rows": completed,
        "resumed_rows": resumed_rows,
        "rows_this_invocation": completed - resumed_rows,
        "questions": len({row["question_id"] for row in target_rows}),
        "family_counts": dict(sorted(family_counts.items())),
        "hidden_size": hidden_size,
        "feature_shape_per_row": [3, hidden_size],
        "feature_dtype": config["feature_dtype"],
        "source_parameters_frozen": True,
        "shards": shards,
        "elapsed_seconds": elapsed,
        "rows_per_second": (completed - resumed_rows) / elapsed,
        "peak_reserved_gb": peak_reserved_gb,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "within_memory_limit": peak_reserved_gb <= config["max_reserved_memory_gb"],
        "finite": True,
    }
    result["gate_passed"] = all(
        (
            result["rows"] == len(target_rows),
            result["source_parameters_frozen"],
            result["within_memory_limit"],
            result["finite"],
            hidden_size is not None,
            math.isfinite(result["rows_per_second"]),
        )
    )
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    manifest_path = output_dir / f"feature_manifest_{mode_name}.json"
    _atomic_json(manifest_path, result)
    _atomic_json(
        progress_path,
        {
            "run_id": config["run_id"],
            "mode": mode_name,
            "phase": "complete",
            "completed_rows": completed,
            "target_rows": len(target_rows),
            "status": result["status"],
            "manifest_path": str(manifest_path),
        },
    )
    return result
