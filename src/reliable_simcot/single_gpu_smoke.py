from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any
from contextlib import nullcontext
import gc
import heapq
import json
import math
import time

import torch

from .official_adapter import (
    OfficialExample,
    iter_icot_examples,
    load_official_model,
)


@dataclass(frozen=True)
class EncodedSmokeExample:
    example: OfficialExample
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    explainable_ids: tuple[int, ...]
    latent_tokens: int
    latent_groups: int
    real_supervised_groups: int
    maximum_auxiliary_length: int


def encode_smoke_example(
    example: OfficialExample,
    tokenizer,
    token_ids: dict[str, int],
    *,
    latent_stage: int,
    c_thought: int,
) -> EncodedSmokeExample:
    if latent_stage <= 0 or c_thought <= 0:
        raise ValueError("latent_stage and c_thought must be positive")
    question_ids = tokenizer.encode(
        example.question + "\n",
        add_special_tokens=True,
    )
    step_ids = [
        tokenizer.encode(step + "\n", add_special_tokens=False)
        for step in example.steps
    ]
    if any(not ids for ids in step_ids):
        raise ValueError(f"Example {example.idx} contains an empty tokenized step")
    answer_ids = tokenizer.encode(
        "### " + example.answer,
        add_special_tokens=False,
    ) + [tokenizer.eos_token_id]

    latent_tokens = latent_stage * c_thought
    explicit_remainder = [
        token for group in step_ids[latent_stage:] for token in group
    ]
    tokens = (
        question_ids
        + [token_ids["<|start-latent|>"]]
        + [token_ids["<|latent|>"]] * latent_tokens
        + [token_ids["<|end-latent|>"]]
        + explicit_remainder
        + answer_ids
    )
    prefix_length = len(question_ids) + latent_tokens + 2
    labels = [-100] * prefix_length + tokens[prefix_length:]
    supervised_groups = step_ids[:latent_stage]
    explainable_ids = [token for group in supervised_groups for token in group]
    maximum_auxiliary_length = max(
        (c_thought + len(group) + 1 for group in supervised_groups),
        default=c_thought + len(answer_ids) + 1,
    )
    return EncodedSmokeExample(
        example=example,
        input_ids=tuple(tokens),
        labels=tuple(labels),
        explainable_ids=tuple(explainable_ids),
        latent_tokens=latent_tokens,
        latent_groups=latent_stage,
        real_supervised_groups=min(len(step_ids), latent_stage),
        maximum_auxiliary_length=maximum_auxiliary_length,
    )


def select_longest_trainable_example(
    dataset_path: str | Path,
    tokenizer,
    token_ids: dict[str, int],
    *,
    latent_stage: int,
    c_thought: int,
    model_max_positions: int,
    candidate_count: int = 256,
) -> EncodedSmokeExample:
    longest_by_characters: list[tuple[int, int, OfficialExample]] = []
    for example in iter_icot_examples(dataset_path):
        character_length = (
            len(example.question)
            + sum(len(step) for step in example.steps)
            + len(example.answer)
        )
        item = (character_length, example.idx, example)
        if len(longest_by_characters) < candidate_count:
            heapq.heappush(longest_by_characters, item)
        elif item[:2] > longest_by_characters[0][:2]:
            heapq.heapreplace(longest_by_characters, item)

    if not longest_by_characters:
        raise ValueError("Training dataset is empty")
    encoded = [
        encode_smoke_example(
            item[2],
            tokenizer,
            token_ids,
            latent_stage=latent_stage,
            c_thought=c_thought,
        )
        for item in longest_by_characters
    ]
    trainable = [
        item
        for item in encoded
        if len(item.input_ids) <= model_max_positions
        and item.maximum_auxiliary_length <= model_max_positions
    ]
    if not trainable:
        raise ValueError("No long candidate fits the model context window")
    return max(
        trainable,
        key=lambda item: (
            max(len(item.input_ids), item.maximum_auxiliary_length),
            len(item.input_ids),
            item.example.idx,
        ),
    )


def tensorize_smoke_example(
    encoded: EncodedSmokeExample,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    input_ids = torch.tensor([encoded.input_ids], dtype=torch.long, device=device)
    labels = torch.tensor([encoded.labels], dtype=torch.long, device=device)
    explainable = torch.tensor(
        [encoded.explainable_ids],
        dtype=torch.long,
        device=device,
    )
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have identical shapes")
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": torch.ones_like(input_ids),
        "position_ids": torch.arange(
            input_ids.shape[1],
            device=device,
            dtype=torch.long,
        ).unsqueeze(0),
        "explainable_ids_list": explainable,
    }


def validate_alignment(
    encoded: EncodedSmokeExample,
    batch: dict[str, torch.Tensor],
    *,
    latent_id: int,
    c_thought: int,
) -> dict[str, Any]:
    actual_latents = int(batch["input_ids"].eq(latent_id).sum().item())
    if actual_latents != encoded.latent_tokens:
        raise ValueError(
            f"Latent-token mismatch: expected {encoded.latent_tokens}, got {actual_latents}"
        )
    if actual_latents % c_thought != 0:
        raise ValueError("Latent-token count is not divisible by c_thought")
    actual_groups = actual_latents // c_thought
    if actual_groups != encoded.latent_groups:
        raise ValueError(
            f"Latent-group mismatch: expected {encoded.latent_groups}, got {actual_groups}"
        )
    if batch["explainable_ids_list"].numel() == 0:
        raise ValueError("No explicit step tokens available for auxiliary supervision")
    return {
        "latent_tokens": actual_latents,
        "latent_groups": actual_groups,
        "c_thought": c_thought,
        "real_supervised_groups": encoded.real_supervised_groups,
        "pseudo_filled_groups": actual_groups - encoded.real_supervised_groups,
        "aligned": True,
    }


def _state_dict_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_r004_for_resume(metrics: dict[str, Any], checkpoint_path: Path) -> None:
    run_id = metrics.get("run_id")
    if (
        not isinstance(run_id, str)
        or (run_id != "R004" and not run_id.startswith("R004-v"))
        or metrics.get("status") != "PASS"
    ):
        raise ValueError("R004 must pass before the R005 resume probe")
    if not metrics.get("reload_consistent") or not metrics.get("gate_passed"):
        raise ValueError("R004 reload and training gates must pass before R005")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    actual_sha = _state_dict_sha256(checkpoint_path)
    if actual_sha != metrics.get("checkpoint_sha256"):
        raise ValueError("R004 checkpoint SHA-256 does not match its metrics")


def _autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported precision: {precision}")


def _probe_loss(
    model,
    batch: dict[str, torch.Tensor],
    *,
    precision: str,
) -> float:
    model.base_causallm.eval()
    model.expainable_llm.eval()
    with torch.inference_mode(), _autocast_context(
        batch["input_ids"].device,
        precision,
    ):
        loss = model(**batch).loss
    value = float(loss.detach().float().item())
    if not math.isfinite(value):
        raise FloatingPointError("Non-finite probe loss")
    return value


def run_training_smoke(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    updates_override: int | None = None,
    accumulation_override: int | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def project_path(value: str) -> Path:
        target = (root / value).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes project root: {value}")
        return target

    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"

    def write_progress(phase: str, **details: Any) -> None:
        payload = {
            "schema_version": 1,
            "run_id": config["run_id"],
            "phase": phase,
            "timestamp_unix": time.time(),
            **details,
        }
        temporary = progress_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(progress_path)

    write_progress("validating_inputs", completed_updates=0)

    provenance = json.loads(
        project_path(config["provenance_manifest"]).read_text(encoding="utf-8")
    )
    if provenance.get("status") != "PASS" or provenance.get("run_id") != "R001":
        raise ValueError("R001 provenance must pass before R004")
    if config.get("gradient_checkpointing"):
        raise ValueError(
            "Official latent feedback relies on KV cache; gradient checkpointing must be false"
        )

    updates = updates_override if updates_override is not None else config["updates"]
    accumulation = (
        accumulation_override
        if accumulation_override is not None
        else config["gradient_accumulation_steps"]
    )
    if updates <= 0 or accumulation <= 0:
        raise ValueError("updates and gradient accumulation must be positive")

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R004 requires a CUDA device")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    write_progress("loading_model", completed_updates=0)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=project_path(config["official_source_dir"]),
        base_model_dir=project_path(config["base_model_dir"]),
        checkpoint_path=project_path(config["checkpoint_path"]),
        device=device,
        # Standard AMP keeps trainable parameters and optimizer state in
        # FP32; autocast below lowers eligible forward operations to FP16.
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    write_progress("selecting_example", completed_updates=0)
    encoded = select_longest_trainable_example(
        project_path(config["dataset_path"]),
        tokenizer,
        token_ids,
        latent_stage=config["latent_stage"],
        c_thought=config["c_thought"],
        model_max_positions=model.base_causallm.config.n_positions,
        candidate_count=config.get("longest_candidate_count", 256),
    )
    batch = tensorize_smoke_example(encoded, device=device)
    alignment = validate_alignment(
        encoded,
        batch,
        latent_id=token_ids["<|latent|>"],
        c_thought=config["c_thought"],
    )
    write_progress(
        "probing_initial_loss",
        completed_updates=0,
        example_idx=encoded.example.idx,
        input_tokens=len(encoded.input_ids),
    )

    precision = config["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported by this GPU")
    initial_loss = _probe_loss(model, batch, precision=precision)
    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision == "fp16")
    update_losses: list[float] = []

    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _ in range(accumulation):
            with _autocast_context(device, precision):
                output = model(**batch)
                raw_loss = output.loss
                scaled_loss = raw_loss / accumulation
            value = float(raw_loss.detach().float().item())
            if not math.isfinite(value):
                raise FloatingPointError(f"Non-finite training loss at update {update}")
            micro_losses.append(value)
            if scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        update_loss = sum(micro_losses) / len(micro_losses)
        update_losses.append(update_loss)
        write_progress(
            "training",
            completed_updates=update + 1,
            target_updates=updates,
            last_loss=update_loss,
            peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 1024**3,
        )
        print(
            f"smoke update {update + 1}/{updates}: loss={update_loss:.6f}",
            flush=True,
        )

    write_progress("probing_final_loss", completed_updates=updates)
    final_loss = _probe_loss(model, batch, precision=precision)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    checkpoint_dir = project_path(config["work_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "checkpoint_smoke.pt"
    write_progress(
        "saving_checkpoint",
        completed_updates=updates,
        checkpoint_path=str(checkpoint_path),
    )
    torch.save(model.state_dict(), checkpoint_path)
    write_progress("hashing_checkpoint", completed_updates=updates)
    checkpoint_sha = _state_dict_sha256(checkpoint_path)
    probe_before_reload = final_loss

    del optimizer, scaler, model
    gc.collect()
    torch.cuda.empty_cache()

    write_progress("reloading_checkpoint", completed_updates=updates)
    reloaded, reloaded_tokenizer, reloaded_token_ids = load_official_model(
        official_coconut_dir=project_path(config["official_source_dir"]),
        base_model_dir=project_path(config["base_model_dir"]),
        checkpoint_path=checkpoint_path,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    reloaded_batch = tensorize_smoke_example(encoded, device=device)
    if reloaded_token_ids != token_ids or len(reloaded_tokenizer) != len(tokenizer):
        raise ValueError("Tokenizer changed across checkpoint reload")
    probe_after_reload = _probe_loss(
        reloaded,
        reloaded_batch,
        precision=precision,
    )
    write_progress("compiling_metrics", completed_updates=updates)
    reload_delta = abs(probe_before_reload - probe_after_reload)
    elapsed = time.perf_counter() - started
    finite = all(math.isfinite(value) for value in [initial_loss, final_loss, *update_losses])
    trend_window = max(1, len(update_losses) // 4)
    first_window_mean = sum(update_losses[:trend_window]) / trend_window
    last_window_mean = sum(update_losses[-trend_window:]) / trend_window
    training_curve_decreased = last_window_mean < first_window_mean
    probe_stable = final_loss <= initial_loss * config["maximum_probe_loss_ratio"]
    full_gate_evaluated = (
        updates == config["updates"]
        and accumulation == config["gradient_accumulation_steps"]
    )
    result = {
        "run_id": config["run_id"],
        "status": "PASS",
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_repeated_micro_batches": updates * accumulation,
        "seed": config["seed"],
        "precision": precision,
        "example": {
            "idx": encoded.example.idx,
            "input_tokens": len(encoded.input_ids),
            "explicit_step_tokens": len(encoded.explainable_ids),
            "maximum_auxiliary_length": encoded.maximum_auxiliary_length,
            "original_steps": len(encoded.example.steps),
        },
        "alignment": alignment,
        "initial_loss": initial_loss,
        "update_losses": update_losses,
        "final_loss": final_loss,
        "finite": finite,
        "trend_window_updates": trend_window,
        "first_window_mean_loss": first_window_mean,
        "last_window_mean_loss": last_window_mean,
        "training_curve_decreased": training_curve_decreased,
        "probe_loss_ratio": final_loss / initial_loss,
        "probe_stable": probe_stable,
        "peak_reserved_gb": peak_reserved_gb,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "within_memory_limit": peak_reserved_gb <= config["max_reserved_memory_gb"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "probe_loss_before_reload": probe_before_reload,
        "probe_loss_after_reload": probe_after_reload,
        "reload_absolute_delta": reload_delta,
        "reload_consistent": reload_delta <= config["reload_tolerance"],
        "elapsed_seconds": elapsed,
        "updates_per_hour": updates / elapsed * 3600,
        "official_release_note": (
            "Released run.py imports but does not call "
            "get_cot_with_explainable_latent_dataset; R004 uses the intended "
            "auxiliary-supervision path required by SIM-CoT."
        ),
        "full_gate_evaluated": full_gate_evaluated,
    }
    result["sanity_passed"] = all(
        (
            result["finite"],
            result["within_memory_limit"],
            result["reload_consistent"],
            result["alignment"]["aligned"],
        )
    )
    result["gate_passed"] = (
        all(
            (
                result["sanity_passed"],
                result["training_curve_decreased"],
                result["probe_stable"],
            )
        )
        if full_gate_evaluated
        else None
    )
    if full_gate_evaluated:
        result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    else:
        result["status"] = "SANITY_PASS" if result["sanity_passed"] else "SANITY_FAIL"

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_progress(
        "complete",
        completed_updates=updates,
        status=result["status"],
        gate_passed=result["gate_passed"],
        metrics_path=str(metrics_path),
    )
    return result


def run_resume_smoke_probe(
    config: dict[str, Any],
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def project_path(value: str) -> Path:
        target = (root / value).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes project root: {value}")
        return target

    r004_metrics = json.loads(
        project_path(config["r004_metrics_path"]).read_text(encoding="utf-8")
    )
    source_checkpoint = project_path(config["source_checkpoint_path"])
    validate_r004_for_resume(r004_metrics, source_checkpoint)

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R005 requires a CUDA device")
    precision = config["precision"]
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 requested but unsupported by this GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=project_path(config["official_source_dir"]),
        base_model_dir=project_path(config["base_model_dir"]),
        checkpoint_path=source_checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    encoded = select_longest_trainable_example(
        project_path(config["dataset_path"]),
        tokenizer,
        token_ids,
        latent_stage=config["latent_stage"],
        c_thought=config["c_thought"],
        model_max_positions=model.base_causallm.config.n_positions,
        candidate_count=config.get("longest_candidate_count", 256),
    )
    if encoded.example.idx != r004_metrics["example"]["idx"]:
        raise ValueError("R005 selected a different smoke example from R004")
    batch = tensorize_smoke_example(encoded, device=device)
    alignment = validate_alignment(
        encoded,
        batch,
        latent_id=token_ids["<|latent|>"],
        c_thought=config["c_thought"],
    )
    loss_before = _probe_loss(model, batch, precision=precision)

    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    optimizer.zero_grad(set_to_none=True)
    micro_losses: list[float] = []
    for _ in range(config["gradient_accumulation_steps"]):
        with _autocast_context(device, precision):
            raw_loss = model(**batch).loss
            scaled_loss = raw_loss / config["gradient_accumulation_steps"]
        value = float(raw_loss.detach().float().item())
        if not math.isfinite(value):
            raise FloatingPointError("Non-finite loss after checkpoint resume")
        micro_losses.append(value)
        scaled_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), config["max_grad_norm"]
    )
    optimizer.step()
    loss_after_update = _probe_loss(model, batch, precision=precision)

    checkpoint_dir = project_path(config["work_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resumed_checkpoint = checkpoint_dir / "checkpoint_resumed.pt"
    torch.save(model.state_dict(), resumed_checkpoint)
    resumed_sha = _state_dict_sha256(resumed_checkpoint)
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()

    reloaded, reloaded_tokenizer, reloaded_token_ids = load_official_model(
        official_coconut_dir=project_path(config["official_source_dir"]),
        base_model_dir=project_path(config["base_model_dir"]),
        checkpoint_path=resumed_checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    if reloaded_token_ids != token_ids or len(reloaded_tokenizer) != len(tokenizer):
        raise ValueError("Tokenizer changed after the resumed checkpoint reload")
    reloaded_batch = tensorize_smoke_example(encoded, device=device)
    loss_after_second_reload = _probe_loss(
        reloaded, reloaded_batch, precision=precision
    )
    second_reload_delta = abs(loss_after_update - loss_after_second_reload)
    elapsed = time.perf_counter() - started
    finite = all(
        math.isfinite(value)
        for value in [loss_before, loss_after_update, *micro_losses]
    )
    result = {
        "run_id": config.get("run_id", "R005"),
        "status": "PASS",
        "source_checkpoint_path": str(source_checkpoint),
        "source_checkpoint_sha256": r004_metrics["checkpoint_sha256"],
        "resumed_checkpoint_path": str(resumed_checkpoint),
        "resumed_checkpoint_sha256": resumed_sha,
        "checkpoint_changed": resumed_sha != r004_metrics["checkpoint_sha256"],
        "example_idx": encoded.example.idx,
        "alignment": alignment,
        "precision": precision,
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "resume_micro_losses": micro_losses,
        "mean_resume_loss": sum(micro_losses) / len(micro_losses),
        "loss_before_resume_update": loss_before,
        "loss_after_resume_update": loss_after_update,
        "loss_after_second_reload": loss_after_second_reload,
        "second_reload_absolute_delta": second_reload_delta,
        "second_reload_consistent": second_reload_delta <= config["reload_tolerance"],
        "gradient_norm_before_clip": float(grad_norm.detach().float().item()),
        "finite": finite,
        "peak_reserved_gb": peak_reserved_gb,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "within_memory_limit": peak_reserved_gb <= config["max_reserved_memory_gb"],
        "elapsed_seconds": elapsed,
        "resume_updates_per_hour": 3600 / elapsed,
    }
    result["gate_passed"] = all(
        (
            result["checkpoint_changed"],
            result["alignment"]["aligned"],
            result["finite"],
            result["second_reload_consistent"],
            result["within_memory_limit"],
        )
    )
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result
