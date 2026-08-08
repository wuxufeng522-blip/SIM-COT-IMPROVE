from __future__ import annotations

from collections import Counter
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

from .corruptions import DEVELOPMENT_FAMILIES, development_variants, parse_checked_equation
from .m1_training import atomic_json, sha256_file
from .official_adapter import (
    OfficialExample,
    build_tokenizer,
    iter_icot_examples,
    load_official_model,
)
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


ARMS = ("clean", "noisy_equal", "oracle_raw_0.1", "oracle_normalized_0.1")


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def verify_schedule(schedule: dict[str, Any]) -> None:
    expected = schedule.get("schedule_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Frozen oracle schedule has no valid SHA-256")
    unhashed = dict(schedule)
    del unhashed["schedule_sha256"]
    if _canonical_hash(unhashed) != expected:
        raise ValueError("Frozen oracle schedule SHA-256 mismatch")


def _question_id(example: OfficialExample) -> str:
    return sha256(example.question.strip().encode("utf-8")).hexdigest()


def _variant_for(example: OfficialExample, family: str, position: int):
    supervised_steps = example.steps[:5]
    variants = development_variants(
        supervised_steps[position],
        prefix_steps=tuple(supervised_steps[:position]),
        # The fifth supervised step may still have genuine later steps in the
        # source solution. They are legitimate future dependencies even though
        # they are outside the five auxiliary groups.
        later_steps=tuple(example.steps[position + 1 :]),
        step_index=position,
    )
    return next((item for item in variants if item.family == family), None)


def create_oracle_schedule(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_path = (root / config["dataset_path"]).resolve()
    checkpoint_path = (root / config["checkpoint_path"]).resolve()
    if sha256_file(dataset_path) != config["dataset_sha256"]:
        raise ValueError("Training dataset SHA-256 mismatch")
    if sha256_file(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint SHA-256 mismatch")

    audit_path = (root / config["audit_manifest_path"]).resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("run_id") != "R020" or audit.get("frozen") is not True:
        raise ValueError("A frozen R020 audit manifest is required")
    excluded = set(audit["selected_question_ids"])

    eligible: list[OfficialExample] = []
    rejection_counts: Counter[str] = Counter()
    for example in iter_icot_examples(dataset_path):
        if _question_id(example) in excluded:
            rejection_counts["natural_audit_question"] += 1
            continue
        if len(example.steps) < 5:
            rejection_counts["fewer_than_five_steps"] += 1
            continue
        if any(parse_checked_equation(step) is None for step in example.steps[:5]):
            rejection_counts["unchecked_first_five"] += 1
            continue
        eligible.append(example)

    rng = random.Random(config["seed"])
    rng.shuffle(eligible)
    target = config["updates"] * config["gradient_accumulation_steps"]
    joint_cells = [
        (family, position)
        for position in range(5)
        for family in DEVELOPMENT_FAMILIES
    ]
    full_cycles, remainder = divmod(target, len(joint_cells))
    pairs = joint_cells * full_cycles
    if remainder == len(joint_cells) - 2:
        omitted = {(DEVELOPMENT_FAMILIES[0], 0), (DEVELOPMENT_FAMILIES[1], 1)}
        pairs.extend(cell for cell in joint_cells if cell not in omitted)
    else:
        pairs.extend(joint_cells[:remainder])
    # A joint permutation keeps both marginals exactly balanced while avoiding a
    # fixed family/position sequence in the training stream.
    rng.shuffle(pairs)

    entries: list[dict[str, Any]] = []
    candidate_index = 0
    failed_variants: Counter[str] = Counter()
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    for schedule_position, (family, noise_position) in enumerate(pairs):
        variant = None
        example = None
        while candidate_index < len(eligible) and variant is None:
            candidate = eligible[candidate_index]
            candidate_index += 1
            encoded = encode_smoke_example(
                candidate,
                tokenizer,
                token_ids,
                latent_stage=config["latent_stage"],
                c_thought=config["c_thought"],
            )
            if (
                len(encoded.input_ids) > config["max_sequence_tokens"]
                or encoded.maximum_auxiliary_length > config["max_sequence_tokens"]
            ):
                failed_variants["context_length"] += 1
                continue
            proposed = _variant_for(candidate, family, noise_position)
            if proposed is None:
                failed_variants[f"{family}@{noise_position}"] += 1
                continue
            example = candidate
            variant = proposed
        if example is None or variant is None:
            raise ValueError(
                f"Only {len(entries)} schedule entries could be constructed; need {target}"
            )
        clean_steps = tuple(example.steps[:5])
        entries.append(
            {
                "position": schedule_position,
                "idx": example.idx,
                "question_sha256": sha256(example.question.encode("utf-8")).hexdigest(),
                "clean_steps_sha256": sha256(
                    json.dumps(clean_steps, ensure_ascii=False).encode("utf-8")
                ).hexdigest(),
                "noise_position": noise_position,
                "noise_family": family,
                "noise_template_id": variant.template_id,
                "corrupted_step": variant.text,
                "corrupted_step_sha256": sha256(variant.text.encode("utf-8")).hexdigest(),
                "y_valid": variant.y_valid,
                "y_utility": variant.y_utility,
                "input_tokens": len(encoded.input_ids),
                "maximum_clean_auxiliary_tokens": encoded.maximum_auxiliary_length,
            }
        )

    family_counts = Counter(entry["noise_family"] for entry in entries)
    position_counts = Counter(str(entry["noise_position"]) for entry in entries)
    schedule = {
        "schema_version": 1,
        "run_id": "O001",
        "status": "PASS",
        "seed": config["seed"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": config["dataset_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "updates": config["updates"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_micro_batches": target,
        "contaminated_steps_per_example": 1,
        "supervised_steps_per_example": 5,
        "step_contamination_rate": 0.2,
        "eligible_population": len(eligible),
        "examined_eligible_candidates": candidate_index,
        "rejection_counts": dict(rejection_counts),
        "failed_variant_counts": dict(failed_variants),
        "family_counts": dict(family_counts),
        "position_counts": dict(position_counts),
        "joint_family_position_counts": dict(
            Counter(
                f"{entry['noise_family']}@{entry['noise_position']}"
                for entry in entries
            )
        ),
        "max_sequence_tokens": config["max_sequence_tokens"],
        "entries": entries,
    }
    schedule["schedule_sha256"] = _canonical_hash(schedule)
    output_path = (root / config["schedule_path"]).resolve()
    atomic_json(output_path, schedule)
    atomic_json((root / config["schedule_audit_path"]).resolve(), {
        key: value for key, value in schedule.items() if key != "entries"
    })
    return schedule


def resolve_scheduled_examples(
    schedule: dict[str, Any], *, dataset_path: str | Path
) -> list[OfficialExample]:
    verify_schedule(schedule)
    expected = [int(entry["idx"]) for entry in schedule["entries"]]
    expected_set = set(expected)
    found = {
        example.idx: example
        for example in iter_icot_examples(dataset_path)
        if example.idx in expected_set
    }
    if len(found) != len(expected):
        raise ValueError("Scheduled examples are missing or duplicated")
    resolved: list[OfficialExample] = []
    for entry in schedule["entries"]:
        example = found[entry["idx"]]
        if sha256(example.question.encode("utf-8")).hexdigest() != entry["question_sha256"]:
            raise ValueError(f"Question changed for scheduled example {example.idx}")
        clean_hash = sha256(
            json.dumps(tuple(example.steps[:5]), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        if clean_hash != entry["clean_steps_sha256"]:
            raise ValueError(f"Steps changed for scheduled example {example.idx}")
        resolved.append(example)
    return resolved


def steps_and_weights(
    arm: str, example: OfficialExample, entry: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if arm not in ARMS:
        raise ValueError(f"Unknown oracle-weighting arm: {arm}")
    steps = list(example.steps[:5])
    noise_position = int(entry["noise_position"])
    if arm != "clean":
        steps[noise_position] = entry["corrupted_step"]
    if arm in {"clean", "noisy_equal"}:
        weights = [1.0] * 5
    else:
        weights = [1.0] * 5
        weights[noise_position] = 0.1
        if arm == "oracle_normalized_0.1":
            scale = 5.0 / sum(weights)
            weights = [value * scale for value in weights]
    return tuple(steps), tuple(weights)


def tokenize_step_targets(tokenizer, steps: Iterable[str]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for step in steps:
        ids = tokenizer.encode(step, add_special_tokens=False)
        if not ids or tokenizer.eos_token_id in ids:
            raise ValueError("Step target is empty or unexpectedly contains EOS")
        groups.append(tuple(ids + [tokenizer.eos_token_id]))
    if len(groups) != 5:
        raise ValueError("Oracle experiment requires exactly five step targets")
    return tuple(groups)


def grouped_auxiliary_loss(
    model,
    batch: dict[str, torch.Tensor],
    step_target_ids: tuple[tuple[int, ...], ...],
    step_weights: tuple[float, ...],
    *,
    latent_id: int,
    c_thought: int,
) -> dict[str, Any]:
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("Grouped oracle loss currently requires micro-batch size 1")
    if len(step_target_ids) != len(step_weights) or len(step_weights) != 5:
        raise ValueError("Five target groups and five weights are required")
    base_batch = {key: value for key, value in batch.items() if key != "explainable_ids_list"}
    base_output = model(**base_batch)
    latent_positions = (
        base_batch["input_ids"][0].eq(latent_id).nonzero(as_tuple=True)[0].tolist()
    )
    if len(latent_positions) != len(step_weights) * c_thought:
        raise ValueError("Latent positions do not align with five step groups")

    device = base_batch["input_ids"].device
    step_losses: list[torch.Tensor] = []
    token_counts: list[int] = []
    for group_index, target_group in enumerate(step_target_ids):
        start = group_index * c_thought
        positions = latent_positions[start : start + c_thought]
        continuous = base_output.inputs_embeds[:, positions, :]
        targets = torch.tensor([target_group], dtype=torch.long, device=device)
        target_embeds = model.embedding(targets)
        input_embeds = torch.cat((continuous, target_embeds), dim=1)
        labels = torch.cat(
            (
                torch.full((1, c_thought), -100, dtype=torch.long, device=device),
                targets,
            ),
            dim=1,
        )
        aux_output = model.expainable_llm(
            inputs_embeds=input_embeds,
            attention_mask=torch.ones(input_embeds.shape[:2], dtype=torch.long, device=device),
            position_ids=torch.arange(
                1, input_embeds.shape[1] + 1, dtype=torch.long, device=device
            ).unsqueeze(0),
            output_hidden_states=False,
        )
        shift_logits = aux_output.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        current = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        step_losses.append(current)
        token_counts.append(len(target_group))

    weights = torch.tensor(step_weights, dtype=step_losses[0].dtype, device=device)
    stacked = torch.stack(step_losses)
    auxiliary = torch.dot(weights, stacked) / len(step_losses)
    total = base_output.loss + auxiliary
    return {
        "loss": total,
        "answer_loss": base_output.loss,
        "auxiliary_loss": auxiliary,
        "step_losses": stacked,
        "token_counts": tuple(token_counts),
        "inputs_embeds": base_output.inputs_embeds,
    }


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def loss_parity_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    device = torch.device(config["device"])
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    examples = resolve_scheduled_examples(
        schedule, dataset_path=(root / config["dataset_path"]).resolve()
    )
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
    example = examples[0]
    encoded = encode_smoke_example(
        example, tokenizer, token_ids,
        latent_stage=config["latent_stage"], c_thought=config["c_thought"]
    )
    batch = tensorize_smoke_example(encoded, device=device)
    clean_targets = tokenize_step_targets(tokenizer, example.steps[:5])
    with torch.inference_mode():
        official_loss = model(**{key: value.clone() for key, value in batch.items()}).loss
        custom = grouped_auxiliary_loss(
            model,
            batch,
            clean_targets,
            (1.0,) * 5,
            latent_id=token_ids["<|latent|>"],
            c_thought=config["c_thought"],
        )
    official_value = float(official_loss.float().item())
    custom_value = float(custom["loss"].float().item())
    delta = abs(official_value - custom_value)
    result = {
        "run_id": "O002",
        "phase": "loss_parity",
        "example_idx": example.idx,
        "official_loss": official_value,
        "custom_all_one_loss": custom_value,
        "absolute_delta": delta,
        "tolerance": config["parity_tolerance"],
        "gate_passed": delta <= config["parity_tolerance"],
    }
    atomic_json((root / config["parity_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_training_arm(
    config: dict[str, Any],
    arm: str,
    *,
    project_root: str | Path,
    updates_override: int | None = None,
    output_suffix: str = "",
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"Unknown arm: {arm}")
    root = Path(project_root).resolve()
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    verify_schedule(schedule)
    updates = updates_override if updates_override is not None else config["updates"]
    if updates <= 0 or updates > schedule["updates"]:
        raise ValueError("Requested updates fall outside the frozen schedule")
    accumulation = config["gradient_accumulation_steps"]
    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Oracle weighting training requires CUDA")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    examples = resolve_scheduled_examples(
        schedule, dataset_path=(root / config["dataset_path"]).resolve()
    )
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
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )
    output_dir = root / "outputs/reliable_simcot/oracle_weighting" / f"{arm}{output_suffix}"
    work_dir = root / "work/reliable_simcot/oracle_weighting" / f"{arm}{output_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    update_losses: list[float] = []
    answer_losses: list[float] = []
    auxiliary_losses: list[float] = []

    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        micro_total: list[float] = []
        micro_answer: list[float] = []
        micro_aux: list[float] = []
        for micro in range(accumulation):
            schedule_position = update * accumulation + micro
            # A corrupted step can tokenize to a different length and therefore
            # consume a different number of dropout draws in the auxiliary
            # decoder. Resetting at every scheduled micro-batch makes the base
            # forward's dropout mask identical across arms until their learned
            # parameters genuinely diverge.
            micro_seed = config["seed"] + schedule_position
            torch.manual_seed(micro_seed)
            torch.cuda.manual_seed_all(micro_seed)
            example = examples[schedule_position]
            entry = schedule["entries"][schedule_position]
            target_steps, weights = steps_and_weights(arm, example, entry)
            encoded = encode_smoke_example(
                example, tokenizer, token_ids,
                latent_stage=config["latent_stage"], c_thought=config["c_thought"]
            )
            batch = tensorize_smoke_example(encoded, device=device)
            target_ids = tokenize_step_targets(tokenizer, target_steps)
            with _autocast(device, config["precision"]):
                losses = grouped_auxiliary_loss(
                    model, batch, target_ids, weights,
                    latent_id=token_ids["<|latent|>"], c_thought=config["c_thought"]
                )
                scaled = losses["loss"] / accumulation
            total_value = float(losses["loss"].detach().float().item())
            answer_value = float(losses["answer_loss"].detach().float().item())
            aux_value = float(losses["auxiliary_loss"].detach().float().item())
            if not all(math.isfinite(value) for value in (total_value, answer_value, aux_value)):
                raise FloatingPointError(f"Non-finite loss in {arm} at update {update + 1}")
            scaled.backward()
            micro_total.append(total_value)
            micro_answer.append(answer_value)
            micro_aux.append(aux_value)
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm in {arm}")
        optimizer.step()
        update_losses.append(sum(micro_total) / accumulation)
        answer_losses.append(sum(micro_answer) / accumulation)
        auxiliary_losses.append(sum(micro_aux) / accumulation)
        completed = update + 1
        if completed == 1 or completed % config["log_every"] == 0 or completed == updates:
            elapsed = time.perf_counter() - started
            progress = {
                "arm": arm,
                "status": "RUNNING",
                "completed_updates": completed,
                "target_updates": updates,
                "latest_total_loss": update_losses[-1],
                "latest_answer_loss": answer_losses[-1],
                "latest_auxiliary_loss": auxiliary_losses[-1],
                "elapsed_seconds": elapsed,
                "updates_per_hour": completed / elapsed * 3600,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{arm} {completed}/{updates}: total={update_losses[-1]:.5f}, "
                f"answer={answer_losses[-1]:.5f}, aux={auxiliary_losses[-1]:.5f}, "
                f"peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )

    checkpoint_path: Path | None = None
    checkpoint_hash: str | None = None
    if save_checkpoint:
        work_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = work_dir / "checkpoint_final.pt"
        temporary = work_dir / "checkpoint_final.pt.tmp"
        torch.save(model.state_dict(), temporary)
        temporary.replace(checkpoint_path)
        checkpoint_hash = sha256_file(checkpoint_path)
    elapsed = time.perf_counter() - started
    result = {
        "run_id": {"clean": "O010", "noisy_equal": "O011", "oracle_raw_0.1": "O012", "oracle_normalized_0.1": "O013"}[arm],
        "arm": arm,
        "status": "PASS",
        "seed": config["seed"],
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_micro_batches": updates * accumulation,
        "precision": config["precision"],
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "max_grad_norm": config["max_grad_norm"],
        "schedule_sha256": schedule["schedule_sha256"],
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "update_total_losses": update_losses,
        "update_answer_losses": answer_losses,
        "update_auxiliary_losses": auxiliary_losses,
        "finite": True,
        "elapsed_seconds": elapsed,
        "updates_per_hour": updates / elapsed * 3600,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_sha256": checkpoint_hash,
    }
    result["within_memory_limit"] = result["peak_reserved_gb"] <= config["max_reserved_memory_gb"]
    result["gate_passed"] = result["finite"] and result["within_memory_limit"]
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json(output_dir / "metrics.json", result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parity = loss_parity_gate(config, project_root=root)
    arm_results: dict[str, Any] = {}
    if parity["gate_passed"]:
        for arm in ARMS:
            arm_results[arm] = run_training_arm(
                config, arm, project_root=root, updates_override=1,
                output_suffix="_sanity", save_checkpoint=False
            )
    result = {
        "run_id": "O002",
        "status": "PASS",
        "parity": parity,
        "arms": arm_results,
    }
    result["gate_passed"] = parity["gate_passed"] and len(arm_results) == len(ARMS) and all(
        item["gate_passed"] for item in arm_results.values()
    )
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json((root / config["sanity_path"]).resolve(), result)
    return result
