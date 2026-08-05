from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from contextlib import nullcontext
import gc
import json
import math
import random
import time

import torch
from transformers import AutoModelForCausalLM

from .official_adapter import (
    build_tokenizer,
    iter_icot_examples,
    load_official_module,
)
from .single_gpu_smoke import (
    EncodedSmokeExample,
    encode_smoke_example,
    tensorize_smoke_example,
)


def sha256_file(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def verify_schedule(schedule: dict[str, Any]) -> None:
    recorded = schedule.get("schedule_sha256")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError("Frozen schedule is missing a valid SHA-256")
    unhashed = dict(schedule)
    del unhashed["schedule_sha256"]
    canonical = json.dumps(unhashed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    actual = sha256(canonical).hexdigest()
    if actual != recorded:
        raise ValueError("Frozen schedule SHA-256 mismatch")


def validate_common_artifacts(config: dict[str, Any], root: Path) -> None:
    checks = (
        (config["dataset_path"], config["dataset_sha256"]),
        (config["common_checkpoint_path"], config["common_checkpoint_sha256"]),
        (config["base_model_weight_path"], config["base_model_weight_sha256"]),
    )
    for value, expected in checks:
        path = (root / value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_file(path) != expected:
            raise ValueError(f"Artifact SHA-256 mismatch: {path}")


def create_schedule(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    validate_common_artifacts(config, root)
    dataset_path = (root / config["dataset_path"]).resolve()
    audit_manifest_path = (root / config["audit_manifest_path"]).resolve()
    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    if audit_manifest.get("frozen") is not True or audit_manifest.get("run_id") != "R020":
        raise ValueError("M1 schedule requires a frozen R020 audit manifest")
    if sha256_file(audit_manifest_path) != config["audit_manifest_sha256"]:
        raise ValueError("M1 audit manifest SHA-256 mismatch")
    excluded_question_ids = set(audit_manifest["selected_question_ids"])
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    target = config["updates"] * config["gradient_accumulation_steps"]
    candidate_count = target + config["candidate_extra"]
    if candidate_count > config["dataset_examples"]:
        raise ValueError("Candidate schedule exceeds the registered dataset size")
    candidate_order = random.Random(config["seed"]).sample(
        range(config["dataset_examples"]), candidate_count
    )
    candidate_set = set(candidate_order)
    raw_examples = {
        example.idx: example
        for example in iter_icot_examples(dataset_path)
        if example.idx in candidate_set
    }
    if len(raw_examples) != candidate_count:
        raise ValueError("Could not resolve every sampled dataset index")

    entries: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for idx in candidate_order:
        example = raw_examples[idx]
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=config["latent_stage"],
            c_thought=config["c_thought"],
        )
        reason = None
        current_question_id = sha256(example.question.strip().encode("utf-8")).hexdigest()
        if current_question_id in excluded_question_ids:
            reason = "natural_audit_question"
        elif len(encoded.input_ids) > config["max_sequence_tokens"]:
            reason = "input_too_long"
        elif not encoded.explainable_ids:
            reason = "no_step_supervision"
        if reason is not None:
            rejected.append({"idx": idx, "reason": reason})
            continue
        entries.append(
            {
                "position": len(entries),
                "idx": idx,
                "question_sha256": sha256(example.question.encode("utf-8")).hexdigest(),
                "input_tokens": len(encoded.input_ids),
                "explainable_tokens": len(encoded.explainable_ids),
                "original_steps": len(example.steps),
            }
        )
        if len(entries) == target:
            break
    if len(entries) != target:
        raise ValueError(f"Only {len(entries)} trainable examples found; need {target}")

    schedule = {
        "schema_version": 1,
        "run_ids": ["R010", "R011"],
        "seed": config["seed"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": config["dataset_sha256"],
        "dataset_examples": config["dataset_examples"],
        "updates": config["updates"],
        "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        "effective_micro_batches": target,
        "latent_stage": config["latent_stage"],
        "c_thought": config["c_thought"],
        "max_sequence_tokens": config["max_sequence_tokens"],
        "audit_manifest_path": str(audit_manifest_path),
        "audit_manifest_sha256": config["audit_manifest_sha256"],
        "audit_selected_question_ids_sha256": audit_manifest[
            "selected_question_ids_sha256"
        ],
        "entries": entries,
        "rejected_candidates": rejected,
    }
    canonical = json.dumps(schedule, ensure_ascii=False, sort_keys=True).encode("utf-8")
    schedule["schedule_sha256"] = sha256(canonical).hexdigest()
    output_path = (root / config["schedule_path"]).resolve()
    atomic_json(output_path, schedule)
    return schedule


def load_scheduled_examples(
    common: dict[str, Any],
    schedule: dict[str, Any],
    *,
    root: Path,
    tokenizer,
    token_ids: dict[str, int],
) -> list[EncodedSmokeExample]:
    expected_indices = [entry["idx"] for entry in schedule["entries"]]
    expected_set = set(expected_indices)
    examples = {
        example.idx: example
        for example in iter_icot_examples((root / common["dataset_path"]).resolve())
        if example.idx in expected_set
    }
    if len(examples) != len(expected_indices):
        raise ValueError("Scheduled training examples are missing")
    encoded_examples: list[EncodedSmokeExample] = []
    audit_manifest = json.loads(
        (root / common["audit_manifest_path"]).read_text(encoding="utf-8")
    )
    excluded_question_ids = set(audit_manifest["selected_question_ids"])
    for entry in schedule["entries"]:
        example = examples[entry["idx"]]
        current_question_id = sha256(example.question.strip().encode("utf-8")).hexdigest()
        if current_question_id in excluded_question_ids:
            raise ValueError("Frozen schedule leaks a natural-audit question")
        if sha256(example.question.encode("utf-8")).hexdigest() != entry["question_sha256"]:
            raise ValueError(f"Scheduled question changed at index {example.idx}")
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=common["latent_stage"],
            c_thought=common["c_thought"],
        )
        if len(encoded.input_ids) != entry["input_tokens"]:
            raise ValueError(f"Tokenization changed at index {example.idx}")
        encoded_examples.append(encoded)
    return encoded_examples


def _wrapper_config(training_method: str, common: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        training_method=training_method,
        max_latent_stage=common["latent_stage"],
        c_thought=common["c_thought"],
        explain_mode="v1_aug",
        packing=False,
        w_prompt=False,
    )


def load_m1_model(
    variant: str,
    common: dict[str, Any],
    *,
    root: Path,
    device: torch.device,
):
    base_dir = (root / common["base_model_dir"]).resolve()
    tokenizer, token_ids = build_tokenizer(base_dir)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_dir), local_files_only=True, torch_dtype=torch.float32
    )
    base_model.resize_token_embeddings(len(tokenizer))
    official = load_official_module((root / common["official_source_dir"]).resolve())

    if variant == "coconut":
        model = official.Coconut(
            base_model,
            token_ids["<|latent|>"],
            token_ids["<|start-latent|>"],
            token_ids["<|end-latent|>"],
            tokenizer.eos_token_id,
        )
    elif variant == "simcot":
        auxiliary = AutoModelForCausalLM.from_pretrained(
            str(base_dir), local_files_only=True, torch_dtype=torch.float32
        )
        model = official.CoconutGPT_Same_Word_Embedding(
            base_model,
            auxiliary,
            tokenizer,
            token_ids["<|latent|>"],
            token_ids["<|start-latent|>"],
            token_ids["<|end-latent|>"],
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<<"),
            common["c_thought"],
            _wrapper_config("full", common),
        )
    else:
        raise ValueError(f"Unsupported M1 variant: {variant}")

    state = torch.load(
        (root / common["common_checkpoint_path"]).resolve(),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    incompatible = model.load_state_dict(state, strict=variant == "coconut")
    if variant == "simcot":
        if incompatible.unexpected_keys or any(
            not key.startswith("expainable_llm.") for key in incompatible.missing_keys
        ):
            raise ValueError(f"SIM-CoT initialization mismatch: {incompatible}")
    del state
    model.to(device=device, dtype=torch.float32)
    return model, tokenizer, token_ids


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _set_train_mode(model, variant: str) -> None:
    model.base_causallm.train()
    if variant == "simcot":
        model.expainable_llm.train()


def _set_eval_mode(model, variant: str) -> None:
    model.base_causallm.eval()
    if variant == "simcot":
        model.expainable_llm.eval()


def _batch_for_variant(
    encoded: EncodedSmokeExample,
    *,
    variant: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    batch = tensorize_smoke_example(encoded, device=device)
    if variant == "coconut":
        batch.pop("explainable_ids_list")
    return batch


def _save_checkpoint(
    *,
    model,
    optimizer,
    work_dir: Path,
    completed_updates: int,
    losses: list[float],
    sample_schedule_sha256: str,
) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    model_path = work_dir / "checkpoint_latest.pt"
    training_path = work_dir / "training_state_latest.pt"
    temporary_model = work_dir / "checkpoint_latest.pt.tmp"
    temporary_training = work_dir / "training_state_latest.pt.tmp"
    torch.save(model.state_dict(), temporary_model)
    temporary_model.replace(model_path)
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "completed_updates": completed_updates,
            "losses": losses,
            "sample_schedule_sha256": sample_schedule_sha256,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
        },
        temporary_training,
    )
    temporary_training.replace(training_path)
    return model_path, training_path


@torch.inference_mode()
def validation_losses(
    model,
    variant: str,
    common: dict[str, Any],
    *,
    root: Path,
    tokenizer,
    token_ids: dict[str, int],
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    _set_eval_mode(model, variant)
    examples = list(iter_icot_examples((root / common["validation_path"]).resolve()))
    selected = examples[: common["validation_examples"]]
    answer_losses: list[float] = []
    step_losses: list[float] = []
    for example in selected:
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=common["latent_stage"],
            c_thought=common["c_thought"],
        )
        base_batch = tensorize_smoke_example(encoded, device=device)
        base_batch.pop("explainable_ids_list")
        with _autocast(device, precision):
            answer_loss = model(**base_batch).loss
        answer_value = float(answer_loss.detach().float().item())
        if not math.isfinite(answer_value):
            raise FloatingPointError("Non-finite validation answer loss")
        answer_losses.append(answer_value)
        if variant == "simcot":
            full_batch = tensorize_smoke_example(encoded, device=device)
            with _autocast(device, precision):
                total_loss = model(**full_batch).loss
            step_value = float((total_loss - answer_loss).detach().float().item())
            if not math.isfinite(step_value):
                raise FloatingPointError("Non-finite validation step loss")
            step_losses.append(step_value)
    return {
        "examples": len(selected),
        "answer_nll": sum(answer_losses) / len(answer_losses),
        "step_nll": sum(step_losses) / len(step_losses) if step_losses else None,
    }


def run_m1_training(
    branch: dict[str, Any],
    common: dict[str, Any],
    *,
    project_root: str | Path,
    updates_override: int | None = None,
    output_dir_override: str | None = None,
    work_dir_override: str | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    validate_common_artifacts(common, root)
    schedule_path = (root / common["schedule_path"]).resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    verify_schedule(schedule)
    if schedule["audit_manifest_sha256"] != common["audit_manifest_sha256"]:
        raise ValueError("Frozen schedule uses a different natural-audit exclusion")
    updates = updates_override if updates_override is not None else common["updates"]
    if updates <= 0 or updates > schedule["updates"]:
        raise ValueError("Requested updates are outside the frozen schedule")
    variant = branch["variant"]
    precision = common["precision"]

    torch.manual_seed(common["seed"])
    torch.cuda.manual_seed_all(common["seed"])
    device = torch.device(common["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("M1 training requires CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is unsupported on this GPU")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    model, tokenizer, token_ids = load_m1_model(
        variant, common, root=root, device=device
    )
    encoded_examples = load_scheduled_examples(
        common,
        schedule,
        root=root,
        tokenizer=tokenizer,
        token_ids=token_ids,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=common["learning_rate"],
        weight_decay=common["weight_decay"],
    )
    work_dir = (root / (work_dir_override or branch["work_dir"])).resolve()
    output_dir = (root / (output_dir_override or branch["output_dir"])).resolve()
    completed_updates = 0
    update_losses: list[float] = []
    if resume:
        model_path = work_dir / "checkpoint_latest.pt"
        training_path = work_dir / "training_state_latest.pt"
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        state = torch.load(training_path, map_location=device, weights_only=False)
        if state["sample_schedule_sha256"] != schedule["schedule_sha256"]:
            raise ValueError("Resume schedule does not match the frozen schedule")
        optimizer.load_state_dict(state["optimizer"])
        completed_updates = int(state["completed_updates"])
        update_losses = [float(value) for value in state["losses"]]
        torch.set_rng_state(state["torch_rng_state"].cpu())
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
        if completed_updates >= updates:
            raise ValueError("Checkpoint already reached the requested update count")

    _set_train_mode(model, variant)
    accumulation = common["gradient_accumulation_steps"]
    for update in range(completed_updates, updates):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        base_position = update * accumulation
        for micro in range(accumulation):
            encoded = encoded_examples[base_position + micro]
            batch = _batch_for_variant(encoded, variant=variant, device=device)
            with _autocast(device, precision):
                raw_loss = model(**batch).loss
                scaled_loss = raw_loss / accumulation
            value = float(raw_loss.detach().float().item())
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"Non-finite {variant} loss at update {update + 1}, micro {micro + 1}"
                )
            micro_losses.append(value)
            scaled_loss.backward()
        optimizer.step()
        update_loss = sum(micro_losses) / len(micro_losses)
        update_losses.append(update_loss)
        completed = update + 1
        if completed == 1 or completed % common["log_every"] == 0:
            elapsed = time.perf_counter() - started
            progress = {
                "run_id": branch["run_id"],
                "variant": variant,
                "status": "RUNNING",
                "completed_updates": completed,
                "target_updates": updates,
                "latest_loss": update_loss,
                "mean_loss": sum(update_losses) / len(update_losses),
                "elapsed_seconds": elapsed,
                "updates_per_hour": completed / elapsed * 3600,
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{branch['run_id']} {completed}/{updates}: loss={update_loss:.6f}, "
                f"peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )
        if completed % common["checkpoint_every"] == 0 or completed == updates:
            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                work_dir=work_dir,
                completed_updates=completed,
                losses=update_losses,
                sample_schedule_sha256=schedule["schedule_sha256"],
            )

    validation = validation_losses(
        model,
        variant,
        common,
        root=root,
        tokenizer=tokenizer,
        token_ids=token_ids,
        device=device,
        precision=precision,
    )
    elapsed = time.perf_counter() - started
    peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1024**3
    model_path = work_dir / "checkpoint_latest.pt"
    result = {
        "run_id": branch["run_id"],
        "variant": variant,
        "status": "PASS",
        "seed": common["seed"],
        "precision": precision,
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_micro_batches": updates * accumulation,
        "learning_rate": common["learning_rate"],
        "weight_decay": common["weight_decay"],
        "schedule_sha256": schedule["schedule_sha256"],
        "common_checkpoint_sha256": common["common_checkpoint_sha256"],
        "update_losses": update_losses,
        "finite": all(math.isfinite(value) for value in update_losses),
        "first_quarter_mean_loss": sum(update_losses[: max(1, updates // 4)])
        / max(1, updates // 4),
        "last_quarter_mean_loss": sum(update_losses[-max(1, updates // 4) :])
        / max(1, updates // 4),
        "training_curve_decreased": (
            sum(update_losses[-max(1, updates // 4) :])
            < sum(update_losses[: max(1, updates // 4)])
        ),
        "validation": validation,
        "peak_reserved_gb": peak_reserved_gb,
        "memory_limit_gb": common["max_reserved_memory_gb"],
        "within_memory_limit": peak_reserved_gb <= common["max_reserved_memory_gb"],
        "elapsed_seconds": elapsed,
        "updates_per_hour": updates / elapsed * 3600,
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
    }
    result["gate_passed"] = all(
        (result["finite"], result["training_curve_decreased"], result["within_memory_limit"])
    )
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json(output_dir / "metrics.json", result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result
