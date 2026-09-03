from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from statistics import mean
from typing import Any
import gc
import json
import math
import random
import time

import torch

from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, load_official_model
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .self_corrected_data import canonical_hash, verify_frozen_manifest
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


ARMS = (
    "clean",
    "solution_n1_equal",
    "solution_n1_w01",
    "solution_n2_equal",
    "solution_n2_w01",
    "misread_n1_equal",
    "misread_n1_w01",
    "misread_n2_equal",
    "misread_n2_w01",
)


def validate_human_audit(
    audit: dict[str, Any], *, manifest_sha256: str, expected_count: int
) -> None:
    if audit.get("status") != "PASS":
        raise ValueError("Human semantic audit has not passed")
    if audit.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Human semantic audit belongs to a different manifest")
    question_ids = audit.get("reviewed_question_ids")
    if (
        not isinstance(question_ids, list)
        or len(question_ids) != expected_count
        or len(set(question_ids)) != expected_count
    ):
        raise ValueError("Human semantic audit count or IDs are invalid")
    decisions = audit.get("per_question_decisions")
    if not isinstance(decisions, dict) or set(decisions) != set(question_ids):
        raise ValueError("Human semantic audit decisions do not match reviewed IDs")
    if any(value != "PASS" for value in decisions.values()):
        raise ValueError("At least one human-audited five-tuple did not pass")


def verify_pretraining_gates(
    config: dict[str, Any], root: Path, *, require_sanity: bool
) -> None:
    manifest = load_manifest(config, root)
    audit_path = (root / config["human_audit_path"]).resolve()
    if not audit_path.is_file():
        raise FileNotFoundError("Human semantic audit is required before GPU work")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    validate_human_audit(
        audit,
        manifest_sha256=str(manifest["manifest_sha256"]),
        expected_count=int(config["audit_examples"]),
    )
    if require_sanity:
        sanity_path = (root / config["sanity_path"]).resolve()
        if not sanity_path.is_file():
            raise FileNotFoundError("Sanity gate is required before formal training")
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        if sanity.get("status") != "PASS" or not sanity.get("gate_passed"):
            raise ValueError("Sanity gate did not pass")
        gradient_path = (root / config["gradient_audit_path"]).resolve()
        if not gradient_path.is_file():
            raise FileNotFoundError("Weight-gradient audit is required before formal training")
        gradient = json.loads(gradient_path.read_text(encoding="utf-8"))
        if gradient.get("status") != "PASS" or not gradient.get("gate_passed"):
            raise ValueError("Weight-gradient audit did not pass")
        memory_gate_value = config.get("max_length_memory_gate_path")
        if memory_gate_value:
            memory_gate_path = (root / str(memory_gate_value)).resolve()
            if not memory_gate_path.is_file():
                raise FileNotFoundError(
                    "Worst-length memory gate is required before formal training"
                )
            memory_gate = json.loads(memory_gate_path.read_text(encoding="utf-8"))
            if memory_gate.get("status") != "PASS" or not memory_gate.get(
                "gate_passed"
            ):
                raise ValueError("Worst-length memory gate did not pass")
        full_schedule_gate_value = config.get("full_schedule_memory_gate_path")
        if full_schedule_gate_value:
            full_schedule_gate_path = (root / str(full_schedule_gate_value)).resolve()
            if not full_schedule_gate_path.is_file():
                raise FileNotFoundError(
                    "Full-schedule memory gate is required before formal training"
                )
            full_schedule_gate = json.loads(
                full_schedule_gate_path.read_text(encoding="utf-8")
            )
            if full_schedule_gate.get("status") != "PASS" or not full_schedule_gate.get(
                "gate_passed"
            ):
                raise ValueError("Full-schedule memory gate did not pass")


def _variant_name(arm: str) -> str:
    if arm == "clean":
        return "clean"
    for suffix in ("_equal", "_w01"):
        if arm.endswith(suffix):
            return arm[: -len(suffix)]
    raise ValueError(f"Unknown self-corrected arm: {arm}")


def steps_and_weights(
    arm: str, row: dict[str, Any], error_step_weight: float
) -> tuple[tuple[str, ...], tuple[float, ...], tuple[int, ...]]:
    if arm not in ARMS:
        raise ValueError(f"Unknown self-corrected arm: {arm}")
    if not 0 < error_step_weight <= 1:
        raise ValueError("Error-step weight must lie in (0, 1]")
    variant = row["variants"][_variant_name(arm)]
    steps = tuple(variant["steps"])
    labels = tuple(int(value) for value in variant["labels"])
    if len(steps) != 5 or len(labels) != 5:
        raise ValueError("Each arm requires exactly five steps and labels")
    raw = [1.0] * 5
    if arm.endswith("_w01"):
        for position, label in enumerate(labels):
            if label == -1:
                raw[position] = float(error_step_weight)
        scale = 5.0 / sum(raw)
        weights = tuple(value * scale for value in raw)
    else:
        weights = (1.0,) * 5
    return steps, weights, labels


def create_training_schedule(
    manifest: dict[str, Any], *, seeds: list[int], updates: int, accumulation: int
) -> dict[str, Any]:
    if updates <= 0 or accumulation <= 0:
        raise ValueError("Updates and accumulation must be positive")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Five-tuple manifest is empty")
    question_ids = [str(row["question_id"]) for row in entries]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("Five-tuple manifest contains duplicate questions")
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
        "arms": list(ARMS),
        "seeds": [int(seed) for seed in seeds],
        "updates": int(updates),
        "gradient_accumulation_steps": int(accumulation),
        "per_seed": per_seed,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    return schedule


def verify_training_schedule(
    schedule: dict[str, Any], *, expected_seeds: list[int], expected_micro_batches: int
) -> None:
    expected_hash = schedule.get("schedule_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Training schedule has no SHA-256")
    if canonical_hash(schedule) != expected_hash:
        raise ValueError("Training schedule SHA-256 mismatch")
    if tuple(schedule.get("arms", ())) != ARMS:
        raise ValueError("Training arm matrix changed")
    if [int(seed) for seed in schedule.get("seeds", ())] != [
        int(seed) for seed in expected_seeds
    ]:
        raise ValueError("Training seeds changed")
    per_seed = schedule.get("per_seed")
    if not isinstance(per_seed, dict):
        raise ValueError("Training schedule has no per-seed order")
    for seed in expected_seeds:
        order = per_seed.get(str(seed))
        if not isinstance(order, list) or len(order) != expected_micro_batches:
            raise ValueError("Training micro-batch count mismatch")


def load_manifest(config: dict[str, Any], root: Path) -> dict[str, Any]:
    manifest = json.loads((root / config["manifest_path"]).read_text(encoding="utf-8"))
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    return manifest


def prepare_training_schedule(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest = load_manifest(config, root)
    schedule = create_training_schedule(
        manifest,
        seeds=[int(seed) for seed in config["seeds"]],
        updates=int(config["updates"]),
        accumulation=int(config["gradient_accumulation_steps"]),
    )
    target = (root / config["schedule_path"]).resolve()
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(schedule):
            raise FileExistsError(f"Refusing to overwrite frozen schedule: {target}")
    else:
        atomic_json(target, schedule)
    return schedule


def load_training_schedule(config: dict[str, Any], root: Path) -> dict[str, Any]:
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    verify_training_schedule(
        schedule,
        expected_seeds=[int(seed) for seed in config["seeds"]],
        expected_micro_batches=int(config["updates"])
        * int(config["gradient_accumulation_steps"]),
    )
    return schedule


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def training_directory(
    root: Path,
    config: dict[str, Any],
    seed: int,
    arm: str,
    *,
    sanity: bool,
    phase_override: str | None = None,
) -> Path:
    phase = phase_override or ("sanity" if sanity else "train")
    if phase not in {
        "train",
        "sanity",
        "max_length_memory_gate",
        "full_schedule_memory_gate",
    }:
        raise ValueError(f"Unsupported training output phase: {phase}")
    return (root / config["output_root"] / phase / f"seed_{seed}" / arm).resolve()


def checkpoint_path(root: Path, config: dict[str, Any], seed: int, arm: str) -> Path:
    return (
        root
        / config["work_root"]
        / "train"
        / f"seed_{seed}"
        / arm
        / "checkpoint_final.pt"
    ).resolve()


def run_training_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    updates_override: int | None = None,
    sanity: bool = False,
    save_checkpoint: bool = True,
    order_override: list[str] | None = None,
    output_phase: str | None = None,
) -> dict[str, Any]:
    if arm not in ARMS or int(seed) not in [int(value) for value in config["seeds"]]:
        raise ValueError("Arm or seed is not frozen")
    root = Path(project_root).resolve()
    verify_pretraining_gates(config, root, require_sanity=not sanity)
    if sha256_file(root / config["checkpoint_path"]) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint SHA-256 mismatch")
    manifest = load_manifest(config, root)
    schedule = load_training_schedule(config, root)
    by_id = {row["question_id"]: row for row in manifest["entries"]}
    frozen_order = schedule["per_seed"][str(seed)]
    order = list(order_override) if order_override is not None else frozen_order
    updates = int(updates_override or config["updates"])
    accumulation = int(config["gradient_accumulation_steps"])
    if updates <= 0 or updates * accumulation > len(order):
        raise ValueError("Requested updates exceed the frozen schedule")
    if order_override is not None:
        if not sanity or save_checkpoint:
            raise ValueError(
                "An order override is restricted to non-checkpointed gate runs"
            )
        unknown_ids = set(order) - set(by_id)
        if unknown_ids:
            raise ValueError(f"Gate order contains unknown question IDs: {unknown_ids}")

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Self-corrected training requires CUDA")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=(root / config["checkpoint_path"]).resolve(),
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
    )
    output_dir = training_directory(
        root,
        config,
        seed,
        arm,
        sanity=sanity,
        phase_override=output_phase,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_losses: list[float] = []
    answer_losses: list[float] = []
    auxiliary_losses: list[float] = []
    preclip_norms: list[float] = []
    error_steps_seen = 0

    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        micro_total: list[float] = []
        micro_answer: list[float] = []
        micro_auxiliary: list[float] = []
        for micro in range(accumulation):
            position = update * accumulation + micro
            micro_seed = seed + position
            torch.manual_seed(micro_seed)
            torch.cuda.manual_seed_all(micro_seed)
            row = by_id[order[position]]
            target_steps, weights, labels = steps_and_weights(
                arm, row, float(config["error_step_weight"])
            )
            error_steps_seen += labels.count(-1)
            clean_steps = tuple(row["variants"]["clean"]["steps"])
            example = OfficialExample(
                idx=position,
                question=row["problem"],
                steps=clean_steps,
                answer=row["answer"],
            )
            encoded = encode_smoke_example(
                example,
                tokenizer,
                token_ids,
                latent_stage=int(config["latent_stage"]),
                c_thought=int(config["c_thought"]),
            )
            batch = tensorize_smoke_example(encoded, device=device)
            target_ids = tokenize_step_targets(tokenizer, target_steps)
            with _autocast(device, config["precision"]):
                losses = grouped_auxiliary_loss(
                    model,
                    batch,
                    target_ids,
                    weights,
                    latent_id=token_ids["<|latent|>"],
                    c_thought=int(config["c_thought"]),
                )
                objective = losses["answer_loss"] + float(config["lambda_aux"]) * losses[
                    "auxiliary_loss"
                ]
                scaled = objective / accumulation
            values = (
                float(objective.detach().float().item()),
                float(losses["answer_loss"].detach().float().item()),
                float(losses["auxiliary_loss"].detach().float().item()),
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite loss in {arm} at update {update + 1}")
            scaled.backward()
            micro_total.append(values[0])
            micro_answer.append(values[1])
            micro_auxiliary.append(values[2])
            # Backward has populated the accumulated parameter gradients.  Do
            # not retain the final micro-batch graph through clipping and
            # AdamW, where its freed blocks can compound allocator
            # fragmentation on mixed-length schedules.
            del scaled, objective, losses, batch, target_ids
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["max_grad_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm in {arm}")
        optimizer.step()
        # Allocator caching does not change parameters, gradients, RNG, data
        # order, or the objective.  Clearing unused blocks between updates
        # prevents length-dependent fragmentation on 8 GiB cards.
        torch.cuda.empty_cache()
        total_losses.append(sum(micro_total) / accumulation)
        answer_losses.append(sum(micro_answer) / accumulation)
        auxiliary_losses.append(sum(micro_auxiliary) / accumulation)
        preclip_norms.append(float(gradient_norm.detach().float().item()))
        completed = update + 1
        if completed == 1 or completed % int(config["log_every"]) == 0 or completed == updates:
            elapsed = time.perf_counter() - started
            progress = {
                "status": "RUNNING",
                "arm": arm,
                "seed": seed,
                "completed_updates": completed,
                "target_updates": updates,
                "latest_total_loss": total_losses[-1],
                "latest_answer_loss": answer_losses[-1],
                "latest_auxiliary_loss": auxiliary_losses[-1],
                "elapsed_seconds": elapsed,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{arm} seed={seed} {completed}/{updates}: "
                f"total={total_losses[-1]:.5f}, answer={answer_losses[-1]:.5f}, "
                f"aux={auxiliary_losses[-1]:.5f}, peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )

    saved_path: Path | None = None
    saved_hash: str | None = None
    if save_checkpoint:
        saved_path = checkpoint_path(root, config, seed, arm)
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = saved_path.with_suffix(saved_path.suffix + ".tmp")
        torch.save(model.state_dict(), temporary)
        temporary.replace(saved_path)
        saved_hash = sha256_file(saved_path)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "run_id": f"SC-{seed}-{arm}" if not sanity else f"SC-SANITY-{seed}-{arm}",
        "status": "PASS",
        "arm": arm,
        "seed": seed,
        "sanity": sanity,
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_micro_batches": updates * accumulation,
        "error_steps_seen": error_steps_seen,
        "raw_error_step_weight": float(config["error_step_weight"]),
        "lambda_aux": float(config["lambda_aux"]),
        "update_total_losses": total_losses,
        "update_answer_losses": answer_losses,
        "update_auxiliary_losses": auxiliary_losses,
        "preclip_gradient_norms": preclip_norms,
        "finite": True,
        "elapsed_seconds": elapsed,
        "updates_per_hour": updates / elapsed * 3600,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "within_memory_limit": peak <= float(config["max_reserved_memory_gb"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "checkpoint_path": str(saved_path) if saved_path else None,
        "checkpoint_sha256": saved_hash,
        "optimizer_implementation": "torch.optim.AdamW(foreach=False)",
        "allocator_cache_policy": "empty_cache_after_each_update",
        "order_override": order_override is not None,
    }
    result["status"] = "PASS" if result["within_memory_limit"] else "FAIL"
    atomic_json(output_dir / "metrics.json", result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verify_pretraining_gates(config, root, require_sanity=False)
    results: dict[str, Any] = {}
    seed = int(config["seeds"][0])
    for arm in ARMS:
        results[arm] = run_training_arm(
            config,
            arm=arm,
            seed=seed,
            project_root=root,
            updates_override=int(config["sanity_updates"]),
            sanity=True,
            save_checkpoint=False,
        )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "seed": seed,
        "arms": results,
        "gate_passed": all(row["status"] == "PASS" for row in results.values()),
    }
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json((root / config["sanity_path"]).resolve(), result)
    return result


def _rank_memory_stress_question_ids(
    config: dict[str, Any],
    *,
    arm: str,
    manifest: dict[str, Any],
    tokenizer,
    token_ids: dict[str, int],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    c_thought = int(config["c_thought"])
    for row in manifest["entries"]:
        clean_steps = tuple(row["variants"]["clean"]["steps"])
        example = OfficialExample(
            idx=0,
            question=row["problem"],
            steps=clean_steps,
            answer=row["answer"],
        )
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=int(config["latent_stage"]),
            c_thought=c_thought,
        )
        target_steps, _, _ = steps_and_weights(
            arm, row, float(config["error_step_weight"])
        )
        target_groups = tokenize_step_targets(tokenizer, target_steps)
        auxiliary_lengths = [c_thought + len(group) for group in target_groups]
        input_tokens = len(encoded.input_ids)
        activation_proxy = input_tokens**2 + sum(
            length**2 for length in auxiliary_lengths
        )
        candidates.append(
            {
                "question_id": row["question_id"],
                "input_tokens": input_tokens,
                "auxiliary_tokens_by_step": auxiliary_lengths,
                "activation_proxy": activation_proxy,
            }
        )
    candidates.sort(
        key=lambda item: (
            int(item["activation_proxy"]),
            max(item["auxiliary_tokens_by_step"]),
            int(item["input_tokens"]),
            str(item["question_id"]),
        ),
        reverse=True,
    )
    return candidates


def run_max_length_memory_gate(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verify_pretraining_gates(config, root, require_sanity=False)
    manifest = load_manifest(config, root)
    schedule = load_training_schedule(config, root)
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    updates = int(config.get("max_length_memory_gate_updates", 2))
    accumulation = int(config["gradient_accumulation_steps"])
    top_k = int(config.get("max_length_memory_gate_top_k", accumulation))
    if updates < 2:
        raise ValueError("Worst-length gate needs at least two optimizer updates")
    if top_k < 1 or top_k > accumulation:
        raise ValueError("Worst-length gate top-k must lie within one accumulation cycle")
    seed = int(config["seeds"][0])
    rows: dict[str, Any] = {}
    for arm in ARMS:
        ranked = _rank_memory_stress_question_ids(
            config,
            arm=arm,
            manifest=manifest,
            tokenizer=tokenizer,
            token_ids=token_ids,
        )
        selected = ranked[:top_k]
        cycle = [str(item["question_id"]) for item in selected]
        while len(cycle) < accumulation:
            cycle.extend(cycle[: accumulation - len(cycle)])
        order = (cycle[:accumulation] * updates)[: updates * accumulation]
        metrics = run_training_arm(
            config,
            arm=arm,
            seed=seed,
            project_root=root,
            updates_override=updates,
            sanity=True,
            save_checkpoint=False,
            order_override=order,
            output_phase="max_length_memory_gate",
        )
        rows[arm] = {
            "status": metrics["status"],
            "peak_reserved_gb": metrics["peak_reserved_gb"],
            "selected_stress_cases": selected,
        }
        if metrics["status"] != "PASS":
            break
    gate_passed = len(rows) == len(ARMS) and all(
        row["status"] == "PASS" for row in rows.values()
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if gate_passed else "FAIL",
        "gate_passed": gate_passed,
        "seed": seed,
        "updates_per_arm": updates,
        "gradient_accumulation_steps": accumulation,
        "top_k_stress_cases": top_k,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "optimizer_implementation": "torch.optim.AdamW(foreach=False)",
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "arms": rows,
        "official_test_opened": False,
    }
    atomic_json((root / config["max_length_memory_gate_path"]).resolve(), result)
    return result


def run_full_schedule_memory_gate(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verify_pretraining_gates(config, root, require_sanity=False)
    manifest = load_manifest(config, root)
    schedule = load_training_schedule(config, root)
    seed = int(config.get("full_schedule_memory_gate_seed", config["seeds"][-1]))
    arm = str(config.get("full_schedule_memory_gate_arm", "clean"))
    if seed not in [int(value) for value in config["seeds"]] or arm not in ARMS:
        raise ValueError("Full-schedule memory gate seed or arm is not frozen")
    metrics = run_training_arm(
        config,
        arm=arm,
        seed=seed,
        project_root=root,
        updates_override=int(config["updates"]),
        sanity=True,
        save_checkpoint=False,
        order_override=list(schedule["per_seed"][str(seed)]),
        output_phase="full_schedule_memory_gate",
    )
    gate_passed = metrics["status"] == "PASS"
    result = {
        "schema_version": 1,
        "status": "PASS" if gate_passed else "FAIL",
        "gate_passed": gate_passed,
        "seed": seed,
        "arm": arm,
        "updates": int(config["updates"]),
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "peak_reserved_gb": metrics["peak_reserved_gb"],
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "optimizer_implementation": "torch.optim.AdamW(foreach=False)",
        "allocator_cache_policy": "empty_cache_after_each_update",
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "official_test_opened": False,
    }
    atomic_json((root / config["full_schedule_memory_gate_path"]).resolve(), result)
    return result


def run_weight_gradient_audit(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    verify_pretraining_gates(config, root, require_sanity=False)
    manifest = load_manifest(config, root)
    schedule = load_training_schedule(config, root)
    row = manifest["entries"][0]
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=(root / config["checkpoint_path"]).resolve(),
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    model.base_causallm.eval()
    model.expainable_llm.eval()
    clean_steps = tuple(row["variants"]["clean"]["steps"])
    example = OfficialExample(
        idx=0,
        question=row["problem"],
        steps=clean_steps,
        answer=row["answer"],
    )
    encoded = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=int(config["latent_stage"]),
        c_thought=int(config["c_thought"]),
    )
    rows: dict[str, Any] = {}
    answer_losses: list[float] = []
    maximum_gradient_delta = 0.0
    for arm in ARMS:
        batch = tensorize_smoke_example(encoded, device=device)
        target_steps, weights, labels = steps_and_weights(
            arm, row, float(config["error_step_weight"])
        )
        targets = tokenize_step_targets(tokenizer, target_steps)
        with _autocast(device, config["precision"]):
            losses = grouped_auxiliary_loss(
                model,
                batch,
                targets,
                weights,
                latent_id=token_ids["<|latent|>"],
                c_thought=int(config["c_thought"]),
            )
        gradients = torch.autograd.grad(
            losses["auxiliary_loss"], losses["step_losses"], retain_graph=False
        )[0].detach().float()
        expected = torch.tensor(weights, dtype=torch.float32, device=device) / 5.0
        delta = float((gradients - expected).abs().max().item())
        maximum_gradient_delta = max(maximum_gradient_delta, delta)
        correct_positions = [index for index, label in enumerate(labels) if label == 1]
        error_positions = [index for index, label in enumerate(labels) if label == -1]
        relative_error_gradient = (
            mean(float(gradients[index].item()) for index in error_positions)
            / mean(float(gradients[index].item()) for index in correct_positions)
            if error_positions
            else None
        )
        answer_loss = float(losses["answer_loss"].detach().float().item())
        answer_losses.append(answer_loss)
        rows[arm] = {
            "labels": list(labels),
            "normalized_step_weights": list(weights),
            "observed_dL_d_step_loss": gradients.cpu().tolist(),
            "expected_dL_d_step_loss": expected.cpu().tolist(),
            "maximum_absolute_gradient_delta": delta,
            "relative_error_to_correct_gradient": relative_error_gradient,
            "answer_loss": answer_loss,
        }
        del losses, gradients, expected, batch
    answer_loss_spread = max(answer_losses) - min(answer_losses)
    ratio_passed = all(
        row_metrics["relative_error_to_correct_gradient"] is None
        or abs(row_metrics["relative_error_to_correct_gradient"] - 0.1) <= 1e-6
        if arm.endswith("_w01")
        else row_metrics["relative_error_to_correct_gradient"] is None
        or abs(row_metrics["relative_error_to_correct_gradient"] - 1.0) <= 1e-6
        for arm, row_metrics in rows.items()
    )
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    gate_passed = (
        maximum_gradient_delta <= 1e-6
        and answer_loss_spread <= 1e-6
        and ratio_passed
        and peak <= float(config["max_reserved_memory_gb"])
    )
    result = {
        "schema_version": 1,
        "status": "PASS" if gate_passed else "FAIL",
        "gate_passed": gate_passed,
        "question_id": row["question_id"],
        "maximum_absolute_gradient_delta": maximum_gradient_delta,
        "answer_loss_spread": answer_loss_spread,
        "relative_gradient_rule_passed": ratio_passed,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "arms": rows,
        "official_test_opened": False,
    }
    atomic_json((root / config["gradient_audit_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
