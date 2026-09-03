from __future__ import annotations

from contextlib import nullcontext
from fractions import Fraction
from pathlib import Path
from statistics import mean, median
from typing import Any
import gc
import json
import math
import random
import time

import torch

from .full_conflict_data import load_frozen_schedule
from .error_cancellation_data import evaluate_arithmetic, parse_equation
from .gradient_leverage import _layer_metrics
from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, load_official_model
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


MAIN_ARMS = (
    "answer_only",
    "clean_aux1",
    "local_causal_25",
    "full_conflict_25",
)
CONDITIONAL_ARM = "full_conflict_50"
STEP_ORDER_REVERSAL_ARM = "reverse_steps_100"
REDUNDANT_STEPS_50_ARM = "redundant_steps_50"
STEP_ORDER_REVERSAL_50_ARM = "reverse_steps_50"
ACCIDENTAL_CORRECT_50_ARM = "accidental_correct_50"
UNRELATED_ACCIDENTAL_CORRECT_50_ARM = "unrelated_accidental_correct_50"
ALL_ARMS = MAIN_ARMS + (
    CONDITIONAL_ARM,
    STEP_ORDER_REVERSAL_ARM,
    REDUNDANT_STEPS_50_ARM,
    STEP_ORDER_REVERSAL_50_ARM,
    ACCIDENTAL_CORRECT_50_ARM,
    UNRELATED_ACCIDENTAL_CORRECT_50_ARM,
)


def _append_redundant_identity(step: str) -> str:
    if not step.startswith("<<") or not step.endswith(">>") or "=" not in step:
        raise ValueError(f"Cannot add a neutral operation to malformed step: {step}")
    _, result = step[2:-2].rsplit("=", 1)
    return f"{step} <<{result}+0={result}>>"


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    reduced = value.denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced == 1:
        digits = max(
            _factor_count(value.denominator, 2),
            _factor_count(value.denominator, 5),
        )
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return f"({value.numerator}/{value.denominator})"


def _factor_count(value: int, factor: int) -> int:
    count = 0
    while value % factor == 0:
        value //= factor
        count += 1
    return count


def accidental_correct_chain(
    example: OfficialExample, entry: dict[str, Any]
) -> tuple[str, ...]:
    """Keep four severe wrong states, then numerically cancel into the true answer.

    The final offset is deliberately unsupported by the problem semantics.  This
    makes the trajectory answer-correct without repairing its reasoning path.
    """
    full_chain = entry.get("full_chain")
    if full_chain is None or len(full_chain.get("steps", ())) != 5:
        raise ValueError("Accidental-correct treatment requires a five-step severe chain")
    severe = tuple(str(step) for step in full_chain["steps"])
    prior = parse_equation(severe[3])
    if not prior.is_true:
        raise ValueError("The fourth severe step must be arithmetically true")
    answer = evaluate_arithmetic(example.answer.replace(",", ""))
    offset = answer - prior.result_value
    prior_text = prior.result_text
    if offset == 0:
        expression = f"{prior_text}+1-1"
    elif offset > 0:
        expression = f"{prior_text}+{_format_fraction(offset)}"
    else:
        expression = f"{prior_text}-{_format_fraction(-offset)}"
    final = f"<<{expression}={_format_fraction(answer)}>>"
    parsed_final = parse_equation(final)
    if not parsed_final.is_true or parsed_final.result_value != answer:
        raise ValueError("Accidental cancellation did not reach the answer")
    return severe[:4] + (final,)


def attach_unrelated_donors(
    schedule: dict[str, Any], *, mapping_seed: int
) -> dict[str, Any]:
    """Attach a seeded one-to-one donor derangement to the treated half.

    The chosen cyclic shift also forbids equal numeric results between a target's
    Clean prefix and its donor prefix at all four aligned positions.  Selection
    depends only on frozen source data, never on model outcomes.
    """
    active = [
        row for row in schedule["train_entries"]
        if row.get("coverage_tier") in {0, 1}
    ]
    ordered = sorted(active, key=lambda row: row["question_id"])
    random.Random(mapping_seed).shuffle(ordered)
    clean_values = {
        row["question_id"]: tuple(
            parse_equation(step).result_value for step in row["clean_steps"][:4]
        )
        for row in ordered
    }
    donor_values = {
        row["question_id"]: tuple(
            parse_equation(step).result_value
            for step in row["full_chain"]["steps"][:4]
        )
        for row in ordered
    }
    candidates: dict[str, list[str]] = {}
    by_id = {row["question_id"]: row for row in ordered}
    rng = random.Random(mapping_seed)
    for target in ordered:
        target_id = target["question_id"]
        donor_ids = [
            donor["question_id"]
            for donor in ordered
            if donor["question_id"] != target_id
            and all(left != right for left, right in zip(
                clean_values[target_id],
                donor_values[donor["question_id"]],
                strict=True,
            ))
        ]
        rng.shuffle(donor_ids)
        candidates[target_id] = donor_ids
    target_order = [row["question_id"] for row in ordered]
    rng.shuffle(target_order)
    donor_to_target: dict[str, str] = {}

    def assign(target_id: str, seen: set[str]) -> bool:
        for donor_id in candidates[target_id]:
            if donor_id in seen:
                continue
            seen.add(donor_id)
            current = donor_to_target.get(donor_id)
            if current is None or assign(current, seen):
                donor_to_target[donor_id] = target_id
                return True
        return False

    for target_id in target_order:
        if not assign(target_id, set()):
            raise ValueError("No one-to-one unrelated donor matching satisfies the prefix constraints")
    target_to_donor = {target: donor for donor, target in donor_to_target.items()}
    if len(target_to_donor) != len(ordered):
        raise ValueError("Unrelated donor matching is incomplete")
    pairs: list[dict[str, Any]] = []
    for target in ordered:
        donor = by_id[target_to_donor[target["question_id"]]]
        if donor["question_id"] == target["question_id"]:
            raise ValueError("Unrelated donor mapping contains a fixed point")
        target["unrelated_donor"] = {
            "question_id": donor["question_id"],
            "source_idx": donor["source_idx"],
            "question": donor["question"],
            "steps": list(donor["full_chain"]["steps"]),
        }
        pairs.append(
            {
                "target_question_id": target["question_id"],
                "donor_question_id": donor["question_id"],
            }
        )
    return {
        "mapping_seed": mapping_seed,
        "mapping_method": "seeded_one_to_one_bipartite_derangement",
        "pairs": sorted(pairs, key=lambda row: row["target_question_id"]),
    }


def unrelated_accidental_correct_chain(
    example: OfficialExample, entry: dict[str, Any]
) -> tuple[str, ...]:
    donor = entry.get("unrelated_donor")
    if donor is None or len(donor.get("steps", ())) != 5:
        raise ValueError("Unrelated-correct treatment requires an attached donor chain")
    donor_steps = tuple(str(step) for step in donor["steps"])
    prior = parse_equation(donor_steps[3])
    answer = evaluate_arithmetic(example.answer.replace(",", ""))
    offset = answer - prior.result_value
    if offset == 0:
        expression = f"{prior.result_text}+1-1"
    elif offset > 0:
        expression = f"{prior.result_text}+{_format_fraction(offset)}"
    else:
        expression = f"{prior.result_text}-{_format_fraction(-offset)}"
    final = f"<<{expression}={_format_fraction(answer)}>>"
    if not parse_equation(final).is_true:
        raise ValueError("Unrelated accidental cancellation is arithmetically false")
    return donor_steps[:4] + (final,)


def arm_targets(
    arm: str, example: OfficialExample, entry: dict[str, Any]
) -> tuple[tuple[str, ...], float, int]:
    if arm not in ALL_ARMS:
        raise ValueError(f"Unknown full-conflict arm: {arm}")
    clean = tuple(example.steps[:5])
    if arm == "answer_only":
        return clean, 0.0, 0
    if arm == "clean_aux1":
        return clean, 1.0, 0
    if arm == STEP_ORDER_REVERSAL_ARM:
        reversed_steps = tuple(reversed(clean))
        changed_positions = sum(
            original != reversed_step
            for original, reversed_step in zip(clean, reversed_steps, strict=True)
        )
        return reversed_steps, 1.0, changed_positions
    tier = entry.get("coverage_tier")
    active_50 = tier in {0, 1}
    if arm == REDUNDANT_STEPS_50_ARM:
        if not active_50:
            return clean, 1.0, 0
        return tuple(_append_redundant_identity(step) for step in clean), 1.0, len(clean)
    if arm == STEP_ORDER_REVERSAL_50_ARM:
        if not active_50:
            return clean, 1.0, 0
        reversed_steps = tuple(reversed(clean))
        changed_positions = sum(
            original != reversed_step
            for original, reversed_step in zip(clean, reversed_steps, strict=True)
        )
        return reversed_steps, 1.0, changed_positions
    if arm == ACCIDENTAL_CORRECT_50_ARM:
        if not active_50:
            return clean, 1.0, 0
        return accidental_correct_chain(example, entry), 1.0, 5
    if arm == UNRELATED_ACCIDENTAL_CORRECT_50_ARM:
        if not active_50:
            return clean, 1.0, 0
        return unrelated_accidental_correct_chain(example, entry), 1.0, 5
    active = tier == 0 or (arm == CONDITIONAL_ARM and tier == 1)
    if not active:
        return clean, 1.0, 0
    if arm == "local_causal_25":
        return tuple(entry["local_chain"]["corrupted_steps"]), 1.0, 3
    return tuple(entry["full_chain"]["steps"]), 1.0, 5


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _examples(schedule: dict[str, Any]) -> list[OfficialExample]:
    return [
        OfficialExample(
            idx=int(row["source_idx"]),
            question=row["question"],
            steps=tuple(row["clean_steps"]),
            answer=row["answer"],
        )
        for row in schedule["train_entries"]
    ]


def training_directory(root: Path, config: dict[str, Any], seed: int, arm: str) -> Path:
    return (root / config["output_root"] / "train" / f"seed_{seed}" / arm).resolve()


def checkpoint_path(root: Path, config: dict[str, Any], seed: int, arm: str) -> Path:
    return (
        root
        / config["work_root"]
        / "train"
        / f"seed_{seed}"
        / arm
        / "checkpoint_final.pt"
    ).resolve()


def run_full_conflict_training(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    updates_override: int | None = None,
    sanity: bool = False,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if arm not in ALL_ARMS or seed not in config["seeds"]:
        raise ValueError("Arm or seed is not preregistered")
    root = Path(project_root).resolve()
    if sha256_file(root / config["checkpoint_path"]) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint SHA-256 mismatch")
    schedule = load_frozen_schedule(config, root)
    if arm == UNRELATED_ACCIDENTAL_CORRECT_50_ARM:
        attach_unrelated_donors(
            schedule, mapping_seed=int(config["unrelated_mapping_seed"])
        )
    entries = schedule["train_entries"]
    examples = _examples(schedule)
    updates = int(updates_override or config["pilot_updates"])
    accumulation = int(config["gradient_accumulation_steps"])
    if updates <= 0 or updates * accumulation > len(entries):
        raise ValueError("Requested updates exceed the frozen 512-example schedule")

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Full-conflict training requires CUDA")
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
    )
    output_dir = (
        (root / config["output_root"] / "sanity" / arm).resolve()
        if sanity
        else training_directory(root, config, seed, arm)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_losses: list[float] = []
    answer_losses: list[float] = []
    auxiliary_losses: list[float] = []
    preclip_norms: list[float] = []
    contaminated_examples = 0
    contaminated_steps = 0

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
            example = examples[position]
            entry = entries[position]
            targets, auxiliary_scale, changed_steps = arm_targets(arm, example, entry)
            contaminated_examples += int(changed_steps > 0)
            contaminated_steps += changed_steps
            encoded = encode_smoke_example(
                example,
                tokenizer,
                token_ids,
                latent_stage=int(config["latent_stage"]),
                c_thought=int(config["c_thought"]),
            )
            batch = tensorize_smoke_example(encoded, device=device)
            target_ids = tokenize_step_targets(tokenizer, targets)
            with _autocast(device, config["precision"]):
                losses = grouped_auxiliary_loss(
                    model,
                    batch,
                    target_ids,
                    (1.0,) * 5,
                    latent_id=token_ids["<|latent|>"],
                    c_thought=int(config["c_thought"]),
                )
                objective = losses["answer_loss"] + auxiliary_scale * losses[
                    "auxiliary_loss"
                ]
                scaled = objective / accumulation
            values = (
                float(objective.detach().float().item()),
                float(losses["answer_loss"].detach().float().item()),
                float(losses["auxiliary_loss"].detach().float().item()),
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(
                    f"Non-finite loss in {arm} at update {update + 1}"
                )
            scaled.backward()
            micro_total.append(values[0])
            micro_answer.append(values[1])
            micro_auxiliary.append(values[2])
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(config["max_grad_norm"])
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm in {arm}")
        optimizer.step()
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
                "latest_preclip_gradient_norm": preclip_norms[-1],
                "elapsed_seconds": elapsed,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{arm} seed={seed} {completed}/{updates}: total={total_losses[-1]:.5f}, "
                f"answer={answer_losses[-1]:.5f}, aux={auxiliary_losses[-1]:.5f}, "
                f"peak={progress['peak_reserved_gb']:.2f} GB",
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
        "run_id": f"FC1-{seed}-{arm}" if not sanity else f"FC031-{arm}",
        "status": "PASS",
        "arm": arm,
        "seed": seed,
        "sanity": sanity,
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_micro_batches": updates * accumulation,
        "contaminated_examples": contaminated_examples,
        "contaminated_steps": contaminated_steps,
        "effective_step_contamination_rate": contaminated_steps
        / (updates * accumulation * 5),
        "auxiliary_scale": 0.0 if arm == "answer_only" else 1.0,
        "update_total_losses": total_losses,
        "update_answer_losses": answer_losses,
        "update_auxiliary_losses": auxiliary_losses,
        "preclip_gradient_norms": preclip_norms,
        "gradient_clipping_fraction": sum(
            value > float(config["max_grad_norm"]) for value in preclip_norms
        )
        / len(preclip_norms),
        "finite": True,
        "elapsed_seconds": elapsed,
        "updates_per_hour": updates / elapsed * 3600,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "within_memory_limit": peak <= float(config["max_reserved_memory_gb"]),
        "schedule_sha256": schedule["schedule_sha256"],
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "checkpoint_path": str(saved_path) if saved_path else None,
        "checkpoint_sha256": saved_hash,
    }
    result["status"] = "PASS" if result["within_memory_limit"] else "FAIL"
    atomic_json(output_dir / "metrics.json", result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_loss_parity(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = load_frozen_schedule(config, root)
    example = _examples(schedule)[0]
    device = torch.device(config["device"])
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
    encoded = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=int(config["latent_stage"]),
        c_thought=int(config["c_thought"]),
    )
    batch = tensorize_smoke_example(encoded, device=device)
    targets = tokenize_step_targets(tokenizer, example.steps)
    with torch.inference_mode():
        official = model(**{key: value.clone() for key, value in batch.items()}).loss
        custom = grouped_auxiliary_loss(
            model,
            batch,
            targets,
            (1.0,) * 5,
            latent_id=token_ids["<|latent|>"],
            c_thought=int(config["c_thought"]),
        )["loss"]
    delta = abs(float(official.float().item()) - float(custom.float().item()))
    result = {
        "run_id": "FC030",
        "official_loss": float(official.float().item()),
        "custom_all_one_loss": float(custom.float().item()),
        "absolute_delta": delta,
        "tolerance": 1e-5,
        "gate_passed": delta <= 1e-5,
        "schedule_sha256": schedule["schedule_sha256"],
    }
    atomic_json((root / config["loss_parity_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity_gate(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parity = run_loss_parity(config, project_root=root)
    arms: dict[str, Any] = {}
    if parity["gate_passed"]:
        for arm in MAIN_ARMS:
            arms[arm] = run_full_conflict_training(
                config,
                arm=arm,
                seed=int(config["seeds"][0]),
                project_root=root,
                updates_override=int(config["sanity_updates"]),
                sanity=True,
                save_checkpoint=False,
            )
    answer_only_ok = bool(arms) and all(
        abs(total - answer) <= 1e-5
        for total, answer in zip(
            arms["answer_only"]["update_total_losses"],
            arms["answer_only"]["update_answer_losses"],
            strict=True,
        )
    )
    memory_ok = bool(arms) and all(row["within_memory_limit"] for row in arms.values())
    finite = bool(arms) and all(row["finite"] for row in arms.values())
    result = {
        "run_id": "FC031",
        "status": "PASS" if parity["gate_passed"] and answer_only_ok and memory_ok and finite else "FAIL",
        "gate_passed": parity["gate_passed"] and answer_only_ok and memory_ok and finite,
        "loss_parity": parity,
        "answer_only_auxiliary_zero": answer_only_ok,
        "memory_ok": memory_ok,
        "finite": finite,
        "arms": arms,
    }
    atomic_json((root / config["sanity_gate_path"]).resolve(), result)
    return result


def run_full_conflict_gradient_audit(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = load_frozen_schedule(config, root)
    chosen: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    per_family = int(config["gradient_examples_per_family"])
    for entry in schedule["train_entries"]:
        if entry["coverage_tier"] != 0:
            continue
        family = entry["full_chain"]["error_family"]
        if counts.get(family, 0) < per_family:
            counts[family] = counts.get(family, 0) + 1
            chosen.append(entry)
    if set(counts) != set(config["error_families"]):
        raise ValueError("Could not build a balanced four-family gradient audit")

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
    blocks = list(model.base_causallm.transformer.h)
    if len(blocks) != int(config["expected_transformer_blocks"]):
        raise ValueError("Unexpected number of transformer blocks")
    groups = [
        [parameter for parameter in block.parameters() if parameter.requires_grad]
        for block in blocks
    ]
    flat = [parameter for group in groups for parameter in group]
    sizes = [len(group) for group in groups]
    rows: list[dict[str, Any]] = []
    for number, entry in enumerate(chosen, start=1):
        example = OfficialExample(
            int(entry["source_idx"]),
            entry["question"],
            tuple(entry["clean_steps"]),
            entry["answer"],
        )
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=int(config["latent_stage"]),
            c_thought=int(config["c_thought"]),
        )
        batch = tensorize_smoke_example(encoded, device=device)
        clean_targets = tokenize_step_targets(tokenizer, example.steps)
        full_targets = tokenize_step_targets(tokenizer, entry["full_chain"]["steps"])
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clean_losses = grouped_auxiliary_loss(
                model,
                batch,
                clean_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=int(config["c_thought"]),
            )
            full_losses = grouped_auxiliary_loss(
                model,
                batch,
                full_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=int(config["c_thought"]),
            )
        answer_grads = list(
            torch.autograd.grad(
                clean_losses["answer_loss"], flat, retain_graph=True, allow_unused=True
            )
        )
        clean_grads = list(
            torch.autograd.grad(
                clean_losses["auxiliary_loss"], flat, retain_graph=False, allow_unused=True
            )
        )
        full_grads = list(
            torch.autograd.grad(
                full_losses["auxiliary_loss"], flat, retain_graph=False, allow_unused=True
            )
        )
        layers = []
        offset = 0
        for layer, size in enumerate(sizes):
            end = offset + size
            layers.append(
                {
                    "layer": layer,
                    **_layer_metrics(
                        answer_grads[offset:end],
                        clean_grads[offset:end],
                        full_grads[offset:end],
                    ),
                }
            )
            offset = end
        rows.append(
            {
                "source_idx": example.idx,
                "error_family": entry["full_chain"]["error_family"],
                "answer_loss": float(clean_losses["answer_loss"].detach().float().item()),
                "clean_auxiliary_loss": float(
                    clean_losses["auxiliary_loss"].detach().float().item()
                ),
                "full_auxiliary_loss": float(
                    full_losses["auxiliary_loss"].detach().float().item()
                ),
                "layers": layers,
            }
        )
        del answer_grads, clean_grads, full_grads, clean_losses, full_losses, batch
        print(f"full-conflict gradient audit: {number}/{len(chosen)}", flush=True)

    aggregate = []
    for layer in range(len(blocks)):
        values = [row["layers"][layer] for row in rows]
        keys = [key for key in values[0] if key != "layer"]
        aggregate.append(
            {
                "layer": layer,
                **{f"median_{key}": median(float(row[key]) for row in values) for key in keys},
                **{f"mean_{key}": mean(float(row[key]) for row in values) for key in keys},
            }
        )
    ratio_layers = sum(
        row["median_clean_aux_to_answer_norm_ratio"]
        >= float(config["gradient_ratio_threshold"])
        for row in aggregate
    )
    changed_layers = sum(
        row["median_clean_noisy_aux_cosine"]
        <= float(config["gradient_direction_cosine_threshold"])
        for row in aggregate
    )
    pathway = ratio_layers >= int(config["gradient_required_layers"]) and changed_layers >= int(
        config["gradient_required_layers"]
    )
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "schema_version": 1,
        "run_id": "FC032",
        "status": "PASS" if pathway and peak <= float(config["max_reserved_memory_gb"]) else "FAIL",
        "gate_passed": pathway and peak <= float(config["max_reserved_memory_gb"]),
        "verdict": "GRADIENT_PATHWAY_PRESENT" if pathway else "GRADIENT_PATHWAY_WEAK",
        "examples": len(chosen),
        "family_counts": counts,
        "layers_meeting_ratio": ratio_layers,
        "layers_meeting_direction_change": changed_layers,
        "aggregate_layers": aggregate,
        "example_rows": rows,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "schedule_sha256": schedule["schedule_sha256"],
        "official_test_opened": False,
    }
    atomic_json((root / config["gradient_audit_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result
