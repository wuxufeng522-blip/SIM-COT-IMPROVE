from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import gc
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, load_official_model
from .oracle_weighting import tokenize_step_targets
from .self_corrected_data import canonical_hash, verify_frozen_manifest
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


STAGE1_ARMS = (
    "C",
    "RL25",
    "RL50",
    "EL25",
    "EL50",
    "RW25",
    "RW50",
    "EW25",
    "EW50",
)
STAGE2_PRIMARY_ARMS = ("RW50-w01", "EW50-w01")
ALL_ARMS = STAGE1_ARMS + STAGE2_PRIMARY_ARMS
ARM_SPECS = {
    "C": ("clean", 0, False),
    "RL25": ("local_redundant", 25, False),
    "RL50": ("local_redundant", 50, False),
    "EL25": ("local_error", 25, False),
    "EL50": ("local_error", 50, False),
    "RW25": ("wide_redundant", 25, False),
    "RW50": ("wide_redundant", 50, False),
    "EW25": ("wide_error", 25, False),
    "EW50": ("wide_error", 50, False),
    "RW50-w01": ("wide_redundant", 50, True),
    "EW50-w01": ("wide_error", 50, True),
}


def load_manifest(config: dict[str, Any], root: Path) -> dict[str, Any]:
    manifest = json.loads((root / config["manifest_path"]).read_text(encoding="utf-8"))
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    parent_hash = config.get("parent_manifest_sha256")
    if parent_hash is not None and manifest["manifest_sha256"] != parent_hash:
        raise ValueError("Reused parent manifest SHA-256 mismatch")
    masks = manifest.get("coverage_masks", {})
    mask25 = set(masks.get("25", ()))
    mask50 = set(masks.get("50", ()))
    if (
        len(mask25) != int(config["coverage_25_examples"])
        or len(mask50) != int(config["coverage_50_examples"])
        or not mask25 < mask50
    ):
        raise ValueError("Frozen coverage masks are invalid or not strictly nested")
    source_parent_path = config.get("source_parent_manifest_path")
    source_parent_hash = config.get("source_parent_manifest_sha256")
    if source_parent_path is not None or source_parent_hash is not None:
        if not source_parent_path or not source_parent_hash:
            raise ValueError("Both source parent manifest path and hash are required")
        parent = json.loads((root / source_parent_path).read_text(encoding="utf-8"))
        verify_frozen_manifest(
            parent,
            expected_train=int(config["train_examples"]),
            expected_test=int(config["test_examples"]),
        )
        if parent["manifest_sha256"] != source_parent_hash:
            raise ValueError("Source parent manifest SHA-256 mismatch")
        if [row["question_id"] for row in manifest["entries"]] != [
            row["question_id"] for row in parent["entries"]
        ]:
            raise ValueError("Source parent training unit order changed")
        if manifest["coverage_masks"] != parent["coverage_masks"]:
            raise ValueError("Source parent coverage masks changed")
        for row, source in zip(manifest["entries"], parent["entries"]):
            if any(row[key] != source[key] for key in ("problem", "answer", "source")):
                raise ValueError("Source parent problem, answer, or source changed")
            if any(
                row["variants"][name] != source["variants"][name]
                for name in ("clean", "local_redundant", "wide_redundant")
            ):
                raise ValueError("Source parent clean or redundant control changed")
    return manifest


def validate_human_audit(config: dict[str, Any], root: Path, manifest: dict[str, Any]) -> None:
    payload = json.loads((root / config["human_audit_path"]).read_text(encoding="utf-8"))
    if (
        payload.get("status") != "PASS"
        or payload.get("manifest_sha256") != manifest["manifest_sha256"]
        or len(payload.get("reviewed_question_ids", ())) != int(config["audit_examples"])
        or any(value != "PASS" for value in payload.get("checks", {}).values())
    ):
        raise ValueError("Codex semantic audit is missing, stale, or failed")


def create_training_schedule(
    manifest: dict[str, Any], *, seeds: Iterable[int], updates: int, accumulation: int
) -> dict[str, Any]:
    question_ids = [row["question_id"] for row in manifest["entries"]]
    required = updates * accumulation
    per_seed: dict[str, list[str]] = {}
    for seed in seeds:
        order: list[str] = []
        epoch = 0
        while len(order) < required:
            current = list(question_ids)
            random.Random(int(seed) + epoch * 1_000_003).shuffle(current)
            order.extend(current)
            epoch += 1
        per_seed[str(seed)] = order[:required]
    schedule = {
        "schema_version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "stage1_arms": list(STAGE1_ARMS),
        "stage2_primary_arms": list(STAGE2_PRIMARY_ARMS),
        "seeds": [int(seed) for seed in seeds],
        "updates": int(updates),
        "gradient_accumulation_steps": int(accumulation),
        "per_seed": per_seed,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    return schedule


def prepare_schedule(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = load_manifest(config, root)
    schedule = create_training_schedule(
        manifest,
        seeds=config["seeds"],
        updates=int(config["updates"]),
        accumulation=int(config["gradient_accumulation_steps"]),
    )
    path = (root / config["schedule_path"]).resolve()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != schedule:
            raise FileExistsError(f"Refusing to overwrite frozen schedule: {path}")
    else:
        atomic_json(path, schedule)
    return schedule


def load_schedule(config: dict[str, Any], root: Path) -> dict[str, Any]:
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    expected_hash = schedule.get("schedule_sha256")
    if not isinstance(expected_hash, str) or canonical_hash(schedule) != expected_hash:
        raise ValueError("Frozen training schedule hash mismatch")
    parent_hash = config.get("parent_schedule_sha256")
    if parent_hash is not None and expected_hash != parent_hash:
        raise ValueError("Reused parent schedule SHA-256 mismatch")
    if tuple(schedule.get("stage1_arms", ())) != STAGE1_ARMS:
        raise ValueError("Stage-1 arm matrix changed")
    if schedule.get("manifest_sha256") != load_manifest(config, root)["manifest_sha256"]:
        raise ValueError("Schedule belongs to a different manifest")
    source_parent_path = config.get("source_parent_schedule_path")
    source_parent_hash = config.get("source_parent_schedule_sha256")
    if source_parent_path is not None or source_parent_hash is not None:
        if not source_parent_path or not source_parent_hash:
            raise ValueError("Both source parent schedule path and hash are required")
        parent = json.loads((root / source_parent_path).read_text(encoding="utf-8"))
        parent_hash = parent.get("schedule_sha256")
        if (
            parent_hash != source_parent_hash
            or canonical_hash(parent) != source_parent_hash
        ):
            raise ValueError("Source parent schedule SHA-256 mismatch")
        for key in ("seeds", "updates", "gradient_accumulation_steps", "per_seed"):
            if schedule.get(key) != parent.get(key):
                raise ValueError(f"Source parent schedule control changed: {key}")
    return schedule


def variant_and_weights(
    arm: str,
    row: dict[str, Any],
    manifest: dict[str, Any],
    *,
    error_step_weight: float,
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[str, ...]]:
    if arm not in ALL_ARMS:
        raise ValueError(f"Unknown error-cancellation arm: {arm}")
    variant_name, coverage, weighted = ARM_SPECS[arm]
    contaminated = coverage > 0 and row["question_id"] in set(
        manifest["coverage_masks"][str(coverage)]
    )
    chosen_name = variant_name if contaminated else "clean"
    variant = row["variants"][chosen_name]
    steps = tuple(variant["steps"])
    types = tuple(variant["types"])
    weights = [1.0] * 5
    if weighted and contaminated:
        for index, step_type in enumerate(types):
            if step_type != "CLEAN":
                weights[index] = float(error_step_weight)
    return steps, tuple(weights), types


def normalized_grouped_auxiliary_loss(
    model,
    batch: dict[str, torch.Tensor],
    step_target_ids: tuple[tuple[int, ...], ...],
    step_weights: tuple[float, ...],
    *,
    latent_id: int,
    c_thought: int,
) -> dict[str, Any]:
    if batch["input_ids"].shape[0] != 1 or len(step_target_ids) != 5:
        raise ValueError("The v10 loss requires one example and exactly five steps")
    if len(step_weights) != 5 or any(weight <= 0 for weight in step_weights):
        raise ValueError("Exactly five positive step weights are required")
    base_batch = {key: value for key, value in batch.items() if key != "explainable_ids_list"}
    base_output = model(**base_batch)
    latent_positions = base_batch["input_ids"][0].eq(latent_id).nonzero(as_tuple=True)[0]
    if latent_positions.numel() != 5 * c_thought:
        raise ValueError("Latent states do not align with five explicit targets")
    step_losses: list[torch.Tensor] = []
    token_counts: list[int] = []
    device = base_batch["input_ids"].device
    for index, group in enumerate(step_target_ids):
        positions = latent_positions[index * c_thought : (index + 1) * c_thought]
        continuous = base_output.inputs_embeds[:, positions, :]
        targets = torch.tensor([group], dtype=torch.long, device=device)
        input_embeds = torch.cat((continuous, model.embedding(targets)), dim=1)
        labels = torch.cat(
            (
                torch.full((1, c_thought), -100, dtype=torch.long, device=device),
                targets,
            ),
            dim=1,
        )
        output = model.expainable_llm(
            inputs_embeds=input_embeds,
            attention_mask=torch.ones(input_embeds.shape[:2], dtype=torch.long, device=device),
            position_ids=torch.arange(
                1, input_embeds.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0),
            output_hidden_states=False,
        )
        logits = output.logits[..., :-1, :].contiguous()
        shifted = labels[..., 1:].contiguous()
        step_losses.append(
            F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                shifted.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        )
        token_counts.append(len(group))
    stacked = torch.stack(step_losses)
    weights = torch.tensor(step_weights, dtype=stacked.dtype, device=device)
    auxiliary = torch.dot(weights, stacked) / len(step_losses)
    return {
        "answer_loss": base_output.loss,
        "auxiliary_loss": auxiliary,
        "step_losses": stacked,
        "token_counts": tuple(token_counts),
        "loss": base_output.loss + auxiliary,
    }


def weighted_step_mean(step_losses: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if step_losses.ndim != 1 or weights.shape != step_losses.shape:
        raise ValueError("Step losses and weights must be matching vectors")
    return torch.dot(step_losses, weights) / step_losses.numel()


def measure_or_clip_gradient_norm(
    parameters: Iterable[torch.nn.Parameter], max_grad_norm: float | None
) -> torch.Tensor:
    """Return the global L2 gradient norm, clipping only when requested."""
    parameters = tuple(parameters)
    if max_grad_norm is not None:
        threshold = float(max_grad_norm)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("max_grad_norm must be null or a finite positive number")
        return torch.nn.utils.clip_grad_norm_(parameters, threshold)

    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)
    component_norms = torch.stack(
        [torch.linalg.vector_norm(gradient, ord=2) for gradient in gradients]
    )
    return torch.linalg.vector_norm(component_norms, ord=2)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def training_directory(
    root: Path, config: dict[str, Any], seed: int, arm: str, *, phase: str = "train"
) -> Path:
    return (root / config["output_root"] / phase / f"seed_{seed}" / arm).resolve()


def checkpoint_path(root: Path, config: dict[str, Any], seed: int, arm: str) -> Path:
    return (
        root / config["work_root"] / "train" / f"seed_{seed}" / arm / "checkpoint_final.pt"
    ).resolve()


def _verify_static_gates(config: dict[str, Any], root: Path, *, formal: bool) -> None:
    if sha256_file(root / config["checkpoint_path"]) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint SHA-256 mismatch")
    manifest = load_manifest(config, root)
    validate_human_audit(config, root, manifest)
    load_schedule(config, root)
    if formal:
        for key in ("equivalence_audit_path", "sanity_path", "memory_gate_path"):
            payload = json.loads((root / config[key]).read_text(encoding="utf-8"))
            if payload.get("status") != "PASS" or not payload.get("gate_passed"):
                raise ValueError(f"Required gate has not passed: {key}")


def run_training_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    updates_override: int | None = None,
    phase: str = "train",
    save_checkpoint: bool = True,
    formal: bool = True,
    order_override: list[str] | None = None,
) -> dict[str, Any]:
    if arm not in ALL_ARMS or int(seed) not in [int(value) for value in config["seeds"]]:
        raise ValueError("Arm or seed is not frozen")
    root = Path(project_root).resolve()
    _verify_static_gates(config, root, formal=formal)
    manifest = load_manifest(config, root)
    schedule = load_schedule(config, root)
    by_id = {row["question_id"]: row for row in manifest["entries"]}
    order = list(order_override or schedule["per_seed"][str(seed)])
    updates = int(updates_override or config["updates"])
    accumulation = int(config["gradient_accumulation_steps"])
    if len(order) < updates * accumulation:
        raise ValueError("Training order is shorter than the requested run")

    output_dir = training_directory(root, config, seed, arm, phase=phase)
    metrics_path = output_dir / "metrics.json"
    saved_path = checkpoint_path(root, config, seed, arm) if save_checkpoint else None
    if metrics_path.exists() or (saved_path is not None and saved_path.exists()):
        raise FileExistsError(f"Refusing to overwrite an existing {arm} run")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("v10 training requires CUDA")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=root / config["checkpoint_path"],
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
        allow_missing_auxiliary=bool(config.get("allow_missing_auxiliary", False)),
    )
    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
    )
    totals: list[float] = []
    answers: list[float] = []
    auxiliaries: list[float] = []
    gradients: list[float] = []
    weighted_steps_seen = 0
    started = time.perf_counter()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        micro_values: list[tuple[float, float, float]] = []
        for micro in range(accumulation):
            position = update * accumulation + micro
            micro_seed = seed + position
            torch.manual_seed(micro_seed)
            torch.cuda.manual_seed_all(micro_seed)
            row = by_id[order[position]]
            target_steps, weights, _ = variant_and_weights(
                arm,
                row,
                manifest,
                error_step_weight=float(config["error_step_weight"]),
            )
            weighted_steps_seen += sum(weight < 1.0 for weight in weights)
            clean = row["variants"]["clean"]["steps"]
            encoded = encode_smoke_example(
                OfficialExample(position, row["problem"], tuple(clean), row["answer"]),
                tokenizer,
                token_ids,
                latent_stage=int(config["latent_stage"]),
                c_thought=int(config["c_thought"]),
            )
            if (
                len(encoded.input_ids) > int(config["max_sequence_tokens"])
                or encoded.maximum_auxiliary_length > int(config["max_sequence_tokens"])
            ):
                raise ValueError("A frozen training example exceeds the context limit")
            batch = tensorize_smoke_example(encoded, device=device)
            targets = tokenize_step_targets(tokenizer, target_steps)
            with _autocast(device, config["precision"]):
                losses = normalized_grouped_auxiliary_loss(
                    model,
                    batch,
                    targets,
                    weights,
                    latent_id=token_ids["<|latent|>"],
                    c_thought=int(config["c_thought"]),
                )
                objective = losses["answer_loss"] + float(config["lambda_aux"]) * losses[
                    "auxiliary_loss"
                ]
                scaled = objective / accumulation
            values = (
                float(objective.detach().float()),
                float(losses["answer_loss"].detach().float()),
                float(losses["auxiliary_loss"].detach().float()),
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite loss in {arm} at update {update + 1}")
            scaled.backward()
            micro_values.append(values)
            del scaled, objective, losses, batch, targets
        norm = measure_or_clip_gradient_norm(
            model.parameters(), config.get("max_grad_norm")
        )
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient norm in {arm}")
        optimizer.step()
        torch.cuda.empty_cache()
        totals.append(sum(value[0] for value in micro_values) / accumulation)
        answers.append(sum(value[1] for value in micro_values) / accumulation)
        auxiliaries.append(sum(value[2] for value in micro_values) / accumulation)
        gradients.append(float(norm.detach().float()))
        completed = update + 1
        if completed == 1 or completed % int(config["log_every"]) == 0 or completed == updates:
            progress = {
                "status": "RUNNING",
                "arm": arm,
                "seed": seed,
                "completed_updates": completed,
                "target_updates": updates,
                "latest_total_loss": totals[-1],
                "latest_answer_loss": answers[-1],
                "latest_auxiliary_loss": auxiliaries[-1],
                "gradient_clipping_enabled": config.get("max_grad_norm") is not None,
                "max_grad_norm": config.get("max_grad_norm"),
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{arm} seed={seed} {completed}/{updates}: total={totals[-1]:.5f}, "
                f"answer={answers[-1]:.5f}, aux={auxiliaries[-1]:.5f}, "
                f"peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )
    checkpoint_hash = None
    if saved_path is not None:
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = saved_path.with_suffix(".pt.tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(saved_path)
        checkpoint_hash = sha256_file(saved_path)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "schema_version": 1,
        "status": "PASS" if peak <= float(config["max_reserved_memory_gb"]) else "FAIL",
        "arm": arm,
        "seed": seed,
        "phase": phase,
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "weighted_steps_seen": weighted_steps_seen,
        "step_loss_reduction": "mean_steps_of_mean_valid_tokens",
        "weight_renormalized": False,
        "update_total_losses": totals,
        "update_answer_losses": answers,
        "update_auxiliary_losses": auxiliaries,
        "preclip_gradient_norms": gradients,
        "gradient_clipping_enabled": config.get("max_grad_norm") is not None,
        "max_grad_norm": config.get("max_grad_norm"),
        "gradient_norm_semantics": "global_l2_before_optimizer_step",
        "elapsed_seconds": elapsed,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "starting_checkpoint_missing_auxiliary": bool(
            config.get("allow_missing_auxiliary", False)
        ),
        "checkpoint_path": str(saved_path) if saved_path else None,
        "checkpoint_sha256": checkpoint_hash,
    }
    atomic_json(metrics_path, result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise RuntimeError(f"{arm} exceeded the frozen memory limit")
    return result


def run_equivalence_audit(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    _verify_static_gates(config, root, formal=False)
    manifest = load_manifest(config, root)
    mask25 = set(manifest["coverage_masks"]["25"])
    row = next(item for item in manifest["entries"] if item["question_id"] in mask25)
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=root / config["checkpoint_path"],
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    model.base_causallm.eval()
    model.expainable_llm.eval()
    clean = tuple(row["variants"]["clean"]["steps"])
    encoded = encode_smoke_example(
        OfficialExample(0, row["problem"], clean, row["answer"]),
        tokenizer,
        token_ids,
        latent_stage=int(config["latent_stage"]),
        c_thought=int(config["c_thought"]),
    )
    # A small terminal parameter is sufficient for exact path-equivalence and
    # avoids retaining nine full embedding-gradient copies on host memory.
    parameter = list(model.base_causallm.parameters())[-1]
    answer_losses: list[float] = []
    lambda0_gradients: list[torch.Tensor] = []
    rows: dict[str, Any] = {}
    for arm in STAGE1_ARMS:
        batch = tensorize_smoke_example(encoded, device=device)
        steps, weights, types = variant_and_weights(
            arm, row, manifest, error_step_weight=float(config["error_step_weight"])
        )
        targets = tokenize_step_targets(tokenizer, steps)
        losses = normalized_grouped_auxiliary_loss(
            model,
            batch,
            targets,
            weights,
            latent_id=token_ids["<|latent|>"],
            c_thought=int(config["c_thought"]),
        )
        gradient = torch.autograd.grad(losses["answer_loss"], parameter)[0].detach().float()
        answer_value = float(losses["answer_loss"].detach().float())
        answer_losses.append(answer_value)
        lambda0_gradients.append(gradient.cpu())
        rows[arm] = {
            "answer_loss": answer_value,
            "auxiliary_loss": float(losses["auxiliary_loss"].detach().float()),
            "types": list(types),
            "weights": list(weights),
            "token_counts": list(losses["token_counts"]),
        }
        del losses, gradient, batch, targets
    answer_spread = max(answer_losses) - min(answer_losses)
    reference = lambda0_gradients[0]
    gradient_delta = max(
        float((gradient - reference).abs().max()) for gradient in lambda0_gradients[1:]
    )
    lr = float(config["learning_rate"])
    update_delta = max(
        float(((parameter.detach().cpu().float() - lr * gradient) -
               (parameter.detach().cpu().float() - lr * reference)).abs().max())
        for gradient in lambda0_gradients[1:]
    )
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    passed = (
        answer_spread <= 1e-7
        and gradient_delta <= 1e-7
        and update_delta <= 1e-7
        and peak <= float(config["max_reserved_memory_gb"])
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "answer_loss_spread": answer_spread,
        "lambda0_gradient_max_delta": gradient_delta,
        "lambda0_sgd_update_max_delta": update_delta,
        "step_loss_reduction": "mean_steps_of_mean_valid_tokens",
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "arms": rows,
    }
    atomic_json(root / config["equivalence_audit_path"], result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seed = int(config["seeds"][0])
    results = {
        arm: run_training_arm(
            config,
            arm=arm,
            seed=seed,
            project_root=root,
            updates_override=int(config["sanity_updates"]),
            phase="sanity",
            save_checkpoint=False,
            formal=False,
        )
        for arm in ("C", "RW50", "EW50")
    }
    passed = all(row["status"] == "PASS" for row in results.values())
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "arms": results,
    }
    atomic_json(root / config["sanity_path"], result)
    return result


def run_memory_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seed = int(config["seeds"][-1])
    metrics = run_training_arm(
        config,
        arm="C",
        seed=seed,
        project_root=root,
        updates_override=int(config["updates"]),
        phase="memory_gate",
        save_checkpoint=False,
        formal=False,
    )
    passed = metrics["status"] == "PASS"
    result = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "gate_passed": passed,
        "seed": seed,
        "arm": "C",
        "full_schedule_updates": int(config["updates"]),
        "metrics": metrics,
    }
    atomic_json(root / config["memory_gate_path"], result)
    return result
