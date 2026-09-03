from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any
import gc
import json
import math
import random
import time

import torch
import torch.nn.functional as F

from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, iter_icot_examples, load_official_model
from .self_corrected_data import canonical_hash, verify_frozen_manifest
from .semantic_conflict_pilot import _compatible, _match_bucket, _question_id
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


def state_dict_semantic_sha256(path: str | Path) -> str:
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    digest = sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    del state
    return digest.hexdigest()


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        return
    atomic_json(path, payload)


def _bucket_slices(rows: list[dict[str, Any]], bucket_size: int) -> list[list[dict[str, Any]]]:
    chunks = [rows[index : index + bucket_size] for index in range(0, len(rows), bucket_size)]
    if len(chunks) > 1 and len(chunks[-1]) < 64:
        chunks[-2].extend(chunks.pop())
    return chunks


def _construct_new_entries(
    rows: list[dict[str, Any]], config: dict[str, Any], *, bucket_offset: int
) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row["step_tokens"], row["question_id"]))
    entries: list[dict[str, Any]] = []
    for local_bucket, bucket in enumerate(
        _bucket_slices(rows, int(config["donor_bucket_size"]))
    ):
        matches = _match_bucket(bucket, config, bucket_offset + local_bucket)
        for recipient_index, recipient in enumerate(bucket):
            donor = bucket[matches[recipient_index]]
            example = recipient["example"]
            conflict = donor["example"]
            row = {
                "question_id": recipient["question_id"],
                "problem": example.question,
                "answer": example.answer,
                "source": {
                    "kind": f"official_gsm8k_aug_exact_{len(example.steps)}_step",
                    "source_file": config["gsm_train_path"],
                    "source_line": example.idx,
                },
                "clean_steps": list(example.steps),
                "semantic_conflict": {
                    "kind": "coherent_cross_question_same_step_count_derangement",
                    "donor_question_id": donor["question_id"],
                    "donor_source_line": conflict.idx,
                    "steps": list(conflict.steps),
                    "step_count": len(conflict.steps),
                    "step_token_ratio": donor["step_tokens"] / recipient["step_tokens"],
                },
            }
            row["five_tuple_sha256"] = canonical_hash(row)
            entries.append(row)
    return entries


def prepare_scaling_data(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
        ("source_v18_manifest_path", "source_v18_manifest_file_sha256"),
        ("source_v18_schedule_path", "source_v18_schedule_file_sha256"),
    ):
        path = root / config[path_key]
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Frozen input missing or changed: {path}")

    if config.get("reuse_frozen_scaling_data", False):
        manifest, schedule = load_scaling_data(config, root)
        if manifest["manifest_sha256"] != config["reused_manifest_sha256"]:
            raise ValueError("Reused scaling manifest semantic hash mismatch")
        if schedule["schedule_sha256"] != config["reused_schedule_sha256"]:
            raise ValueError("Reused scaling schedule semantic hash mismatch")
        audit = {
            "schema_version": 1,
            "status": "PASS",
            "run_id": config["run_id"],
            "reuse_mode": "READ_ONLY_FROZEN_PARENT_DATA",
            "source_run_id": manifest["run_id"],
            "unique_train_examples": len(manifest["entries"]),
            "unique_question_ids": len(
                {row["question_id"] for row in manifest["entries"]}
            ),
            "manifest_sha256": manifest["manifest_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "v18_entries_preserved": True,
            "v18_schedule_prefix_preserved": True,
        }
        _immutable_json(root / config["audit_path"], audit)
        return audit

    source_manifest = json.loads(
        (root / config["source_v18_manifest_path"]).read_text(encoding="utf-8")
    )
    verify_frozen_manifest(source_manifest, expected_train=8192, expected_test=1319)
    if source_manifest["manifest_sha256"] != config["source_v18_manifest_sha256"]:
        raise ValueError("v18 manifest semantic hash changed")
    source_schedule = json.loads(
        (root / config["source_v18_schedule_path"]).read_text(encoding="utf-8")
    )
    if (
        canonical_hash(source_schedule) != source_schedule.get("schedule_sha256")
        or source_schedule["schedule_sha256"] != config["source_v18_schedule_sha256"]
    ):
        raise ValueError("v18 schedule semantic hash changed")

    tokenizer, _ = build_tokenizer(root / config["base_model_dir"])
    test_examples = list(iter_icot_examples(root / config["gsm_test_path"]))
    test_ids = {_question_id(example.question) for example in test_examples}
    if len(test_ids) != int(config["test_examples"]):
        raise ValueError("Official test count mismatch")

    seen: set[str] = set()
    pools: dict[int, list[dict[str, Any]]] = {4: [], 5: []}
    for example in iter_icot_examples(root / config["gsm_train_path"]):
        question_id = _question_id(example.question)
        if question_id in seen or question_id in test_ids or len(example.steps) not in pools:
            continue
        seen.add(question_id)
        pools[len(example.steps)].append(
            {
                "question_id": question_id,
                "example": example,
                "step_tokens": sum(
                    len(tokenizer.encode(step, add_special_tokens=False)) + 1
                    for step in example.steps
                ),
            }
        )

    source_ids = {row["question_id"] for row in source_manifest["entries"]}
    if len(source_ids) != 8192 or not source_ids <= {row["question_id"] for row in pools[5]}:
        raise ValueError("v18 recipients are not a strict subset of the exact-five pool")
    remaining_five = [row for row in pools[5] if row["question_id"] not in source_ids]
    remaining_five.sort(
        key=lambda row: sha256(
            f"{config['run_id']}:select5:{row['question_id']}".encode("utf-8")
        ).hexdigest()
    )
    required_new_five = int(config["exact_five_examples"]) - 8192
    selected_five = remaining_five[:required_new_five]
    pools[4].sort(
        key=lambda row: sha256(
            f"{config['run_id']}:select4:{row['question_id']}".encode("utf-8")
        ).hexdigest()
    )
    selected_four = pools[4][: int(config["exact_four_examples"])]
    if len(selected_five) != required_new_five or len(selected_four) != int(
        config["exact_four_examples"]
    ):
        raise RuntimeError("Insufficient unique four/five-step examples")

    new_five_entries = _construct_new_entries(selected_five, config, bucket_offset=1000)
    new_four_entries = _construct_new_entries(selected_four, config, bucket_offset=2000)
    entries = list(source_manifest["entries"]) + new_five_entries + new_four_entries
    if len(entries) != int(config["unique_train_examples"]):
        raise RuntimeError("Scaling manifest count mismatch")
    manifest = {
        "schema_version": 1,
        "dataset_family": "gsm8k_aug",
        "run_id": config["run_id"],
        "source_v18_manifest_sha256": source_manifest["manifest_sha256"],
        "entries": entries,
        "test_problem_ids": sorted(test_ids),
        "generator_disclosure": config["disclosure"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["unique_train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    _immutable_json(root / config["manifest_path"], manifest)

    extension_ids = [row["question_id"] for row in new_five_entries + new_four_entries]
    random.Random(int(config["seed"]) + 20).shuffle(extension_ids)
    order = list(source_schedule["order"]) + extension_ids
    if len(order) != int(config["unique_train_examples"]) or len(set(order)) != len(order):
        raise RuntimeError("Scaling schedule is not a unique full pass")
    schedule = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "seed": int(config["seed"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_v18_schedule_sha256": source_schedule["schedule_sha256"],
        "v18_prefix_examples": 8192,
        "order": order,
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
        "milestone_examples": list(config["milestone_examples"]),
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    _immutable_json(root / config["schedule_path"], schedule)

    all_donors = [
        row["semantic_conflict"]["donor_question_id"] for row in entries
    ]
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "unique_train_examples": len(entries),
        "unique_question_ids": len({row["question_id"] for row in entries}),
        "exact_five_examples": sum(len(row["clean_steps"]) == 5 for row in entries),
        "exact_four_examples": sum(len(row["clean_steps"]) == 4 for row in entries),
        "unique_donor_questions": len(set(all_donors)),
        "recipient_donor_overlap_count": sum(
            row["question_id"] == row["semantic_conflict"]["donor_question_id"]
            for row in entries
        ),
        "all_target_counts_match_real_steps": all(
            len(row["clean_steps"]) == len(row["semantic_conflict"]["steps"])
            for row in entries
        ),
        "all_corresponding_steps_different": all(
            all(left != right for left, right in zip(row["clean_steps"], row["semantic_conflict"]["steps"]))
            for row in entries
        ),
        "v18_entries_preserved": entries[:8192] == source_manifest["entries"],
        "v18_schedule_prefix_preserved": order[:8192] == source_schedule["order"],
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "audit_examples": sorted(
            entries,
            key=lambda row: sha256(
                f"{config['run_id']}:audit:{row['question_id']}".encode("utf-8")
            ).hexdigest(),
        )[:20],
        "disclosure": config["disclosure"],
    }
    if not all(
        audit[key]
        for key in (
            "all_target_counts_match_real_steps",
            "all_corresponding_steps_different",
            "v18_entries_preserved",
            "v18_schedule_prefix_preserved",
        )
    ) or audit["recipient_donor_overlap_count"]:
        audit["status"] = "FAIL"
    _immutable_json(root / config["audit_path"], audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Scaling data audit failed")
    return audit


def load_scaling_data(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((root / config["manifest_path"]).read_text(encoding="utf-8"))
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["unique_train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    if canonical_hash(schedule) != schedule.get("schedule_sha256"):
        raise ValueError("Scaling schedule hash mismatch")
    if schedule["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Schedule belongs to a different scaling manifest")
    if len(schedule["order"]) != int(config["unique_train_examples"]) or len(
        set(schedule["order"])
    ) != int(config["unique_train_examples"]):
        raise ValueError("Scaling schedule must contain every example exactly once")
    return manifest, schedule


def variable_grouped_auxiliary_loss(
    model,
    batch: dict[str, torch.Tensor],
    step_target_ids: tuple[tuple[int, ...], ...],
    *,
    latent_id: int,
    c_thought: int,
    latent_groups: int,
) -> dict[str, Any]:
    if batch["input_ids"].shape[0] != 1 or not 1 <= len(step_target_ids) <= latent_groups:
        raise ValueError("Targets must cover one to latent_groups real steps")
    base_batch = {key: value for key, value in batch.items() if key != "explainable_ids_list"}
    base_output = model(**base_batch)
    latent_positions = base_batch["input_ids"][0].eq(latent_id).nonzero(as_tuple=True)[0]
    if latent_positions.numel() != latent_groups * c_thought:
        raise ValueError("Latent states do not match the padded latent stage")
    losses: list[torch.Tensor] = []
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
            position_ids=torch.arange(1, input_embeds.shape[1] + 1, device=device).unsqueeze(0),
            output_hidden_states=False,
        )
        logits = output.logits[..., :-1, :].contiguous()
        shifted = labels[..., 1:].contiguous()
        losses.append(
            F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                shifted.view(-1),
                ignore_index=-100,
                reduction="mean",
            )
        )
    auxiliary = torch.stack(losses).mean()
    return {
        "answer_loss": base_output.loss,
        "auxiliary_loss": auxiliary,
        "loss": base_output.loss + auxiliary,
    }


def variable_tokenize_step_targets(tokenizer, steps: list[str]) -> tuple[tuple[int, ...], ...]:
    groups: list[tuple[int, ...]] = []
    for step in steps:
        ids = tokenizer.encode(step, add_special_tokens=False)
        if not ids or tokenizer.eos_token_id in ids:
            raise ValueError("Step target is empty or unexpectedly contains EOS")
        groups.append(tuple(ids + [tokenizer.eos_token_id]))
    if not 1 <= len(groups) <= 5:
        raise ValueError("Scaling targets require one to five real steps")
    return tuple(groups)


def auxiliary_target_steps(row: dict[str, Any], config: dict[str, Any]) -> list[str]:
    target_kind = config.get("auxiliary_target", "semantic_conflict_steps")
    if target_kind == "semantic_conflict_steps":
        return list(row["semantic_conflict"]["steps"])
    if target_kind == "official_clean_steps":
        return list(row["clean_steps"])
    raise ValueError(f"Unknown auxiliary target: {target_kind}")


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def global_gradient_norm(parameters) -> torch.Tensor:
    """Measure the global L2 gradient norm without modifying any gradient."""
    squared_norm = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        parameter_norm = torch.linalg.vector_norm(parameter.grad.detach().float(), ord=2)
        contribution = parameter_norm.square()
        squared_norm = contribution if squared_norm is None else squared_norm + contribution
    if squared_norm is None:
        return torch.tensor(0.0)
    return squared_norm.sqrt()


def _milestone_paths(root: Path, config: dict[str, Any], examples: int) -> tuple[Path, Path, Path]:
    directory = root / config["work_root"] / "milestones"
    return (
        directory / f"checkpoint_{examples}.pt",
        directory / f"optimizer_{examples}.pt",
        directory / f"resume_{examples}.json",
    )


def _torch_save_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _latest_resume(
    root: Path, config: dict[str, Any]
) -> tuple[int, Path | None, Path | None, dict[str, Any] | None]:
    for examples in sorted((int(value) for value in config["milestone_examples"]), reverse=True):
        model_path, optimizer_path, meta_path = _milestone_paths(root, config, examples)
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            meta.get("status") != "PASS"
            or meta.get("examples_seen") != examples
            or sha256_file(model_path) != meta.get("model_file_sha256")
            or sha256_file(optimizer_path) != meta.get("optimizer_file_sha256")
        ):
            raise ValueError(f"Invalid resumable milestone: {examples}")
        return examples, model_path, optimizer_path, meta
    bootstrap = config.get("bootstrap_resume")
    if bootstrap is not None:
        examples = int(bootstrap["examples_seen"])
        if examples not in {int(value) for value in config["milestone_examples"]}:
            raise ValueError("Bootstrap resume is not a configured milestone")
        accumulation = int(config["gradient_accumulation_steps"])
        if examples <= 0 or examples % accumulation:
            raise ValueError("Bootstrap resume is not aligned to an optimizer update")
        model_path = root / bootstrap["model_path"]
        optimizer_path = root / bootstrap["optimizer_path"]
        if sha256_file(model_path) != bootstrap["model_file_sha256"]:
            raise ValueError("Bootstrap model file SHA-256 mismatch")
        if sha256_file(optimizer_path) != bootstrap["optimizer_file_sha256"]:
            raise ValueError("Bootstrap optimizer file SHA-256 mismatch")
        expected_semantic = bootstrap.get("model_semantic_sha256")
        if expected_semantic is not None and state_dict_semantic_sha256(model_path) != expected_semantic:
            raise ValueError("Bootstrap model semantic SHA-256 mismatch")
        start_update = examples // accumulation
        meta = {
            "schema_version": 1,
            "status": "PASS",
            "examples_seen": examples,
            "completed_updates": start_update,
            "metric_history_start_update": start_update,
            "metric_history_complete": False,
            "total_losses": [],
            "answer_losses": [],
            "auxiliary_losses": [],
            "gradient_norms": [],
            "external_bootstrap": True,
            "source_run_id": bootstrap.get("source_run_id"),
            "source_gate_status": bootstrap.get("source_gate_status"),
        }
        return examples, model_path, optimizer_path, meta
    return 0, None, None, None


def run_scaling_training(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    max_updates: int | None = None,
    phase: str = "train",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if sha256_file(root / config["checkpoint_path"]) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint changed")
    manifest, schedule = load_scaling_data(config, root)
    by_id = {row["question_id"]: row for row in manifest["entries"]}
    accumulation = int(config["gradient_accumulation_steps"])
    full_updates = len(schedule["order"]) // accumulation
    updates = full_updates if max_updates is None else int(max_updates)
    if updates < 1 or updates > full_updates:
        raise ValueError("Invalid scaling update count")
    formal = phase == "train" and max_updates is None
    output_dir = root / config["output_root"] / phase
    metrics_path = root / config["training_metrics_path"] if formal else output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Scaling training requires CUDA")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    if formal:
        start_examples, resume_model, resume_optimizer, resume_meta = _latest_resume(root, config)
    else:
        start_examples, resume_model, resume_optimizer, resume_meta = 0, None, None, None
    start_update = start_examples // accumulation
    checkpoint_to_load = resume_model or (root / config["checkpoint_path"])
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=checkpoint_to_load,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
        allow_missing_auxiliary=resume_model is None,
    )
    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
    )
    if resume_optimizer is not None:
        optimizer_state = torch.load(resume_optimizer, map_location="cpu", weights_only=True)
        optimizer.load_state_dict(optimizer_state)
        del optimizer_state

    total_losses = list(resume_meta.get("total_losses", ())) if resume_meta else []
    answer_losses = list(resume_meta.get("answer_losses", ())) if resume_meta else []
    auxiliary_losses = list(resume_meta.get("auxiliary_losses", ())) if resume_meta else []
    gradient_norms = list(resume_meta.get("gradient_norms", ())) if resume_meta else []
    metric_history_start_update = (
        int(resume_meta.get("metric_history_start_update", 0)) if resume_meta else 0
    )
    expected_history = start_update - metric_history_start_update
    if metric_history_start_update < 0 or expected_history < 0 or any(
        len(values) != expected_history
        for values in (total_losses, answer_losses, auxiliary_losses, gradient_norms)
    ):
        raise ValueError("Resumed metric history length mismatch")

    milestones = set(int(value) for value in config["milestone_examples"])
    started = time.perf_counter()
    for update in range(start_update, updates):
        optimizer.zero_grad(set_to_none=True)
        micro_values: list[tuple[float, float, float]] = []
        for micro in range(accumulation):
            position = update * accumulation + micro
            row = by_id[schedule["order"][position]]
            micro_seed = seed + position
            torch.manual_seed(micro_seed)
            torch.cuda.manual_seed_all(micro_seed)
            example = OfficialExample(
                idx=position,
                question=row["problem"],
                steps=tuple(row["clean_steps"]),
                answer=row["answer"],
            )
            encoded = encode_smoke_example(
                example,
                tokenizer,
                token_ids,
                latent_stage=int(config["latent_stage"]),
                c_thought=int(config["c_thought"]),
            )
            if (
                len(encoded.input_ids) > int(config["max_sequence_tokens"])
                or encoded.maximum_auxiliary_length > int(config["max_sequence_tokens"])
            ):
                raise ValueError(f"Example exceeds context: {row['question_id']}")
            batch = tensorize_smoke_example(encoded, device=device)
            targets = variable_tokenize_step_targets(tokenizer, auxiliary_target_steps(row, config))
            with _autocast(device, config["precision"]):
                losses = variable_grouped_auxiliary_loss(
                    model,
                    batch,
                    targets,
                    latent_id=token_ids["<|latent|>"],
                    c_thought=int(config["c_thought"]),
                    latent_groups=int(config["latent_stage"]),
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
                raise FloatingPointError(f"Non-finite loss at update {update + 1}")
            scaled.backward()
            micro_values.append(values)
            del batch, targets, losses, objective, scaled
        max_grad_norm = config.get("max_grad_norm")
        if max_grad_norm is None:
            # The official GPT-2 + Coconut training path does not clip gradients.
            # Keep a read-only norm measurement so non-finite updates remain detectable.
            norm = global_gradient_norm(model.parameters())
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(max_grad_norm))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient at update {update + 1}")
        optimizer.step()
        torch.cuda.empty_cache()
        total_losses.append(sum(value[0] for value in micro_values) / accumulation)
        answer_losses.append(sum(value[1] for value in micro_values) / accumulation)
        auxiliary_losses.append(sum(value[2] for value in micro_values) / accumulation)
        gradient_norms.append(float(norm.detach().float()))
        completed = update + 1
        examples_seen = completed * accumulation
        if completed == 1 or completed % int(config["log_every"]) == 0 or completed == updates:
            progress = {
                "schema_version": 1,
                "status": "RUNNING",
                "phase": phase,
                "completed_updates": completed,
                "target_updates": updates,
                "examples_seen": examples_seen,
                "unique_examples_seen": examples_seen,
                "latest_total_loss": total_losses[-1],
                "latest_answer_loss": answer_losses[-1],
                "latest_auxiliary_loss": auxiliary_losses[-1],
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "manifest_sha256": manifest["manifest_sha256"],
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"semantic-conflict-scaling {completed}/{updates} ({examples_seen} examples): "
                f"total={total_losses[-1]:.5f}, answer={answer_losses[-1]:.5f}, "
                f"aux={auxiliary_losses[-1]:.5f}, peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )
        if formal and examples_seen in milestones and examples_seen > start_examples:
            model_path, optimizer_path, meta_path = _milestone_paths(root, config, examples_seen)
            if any(path.exists() for path in (model_path, optimizer_path, meta_path)):
                raise FileExistsError(f"Refusing to overwrite milestone {examples_seen}")
            _torch_save_atomic(model.state_dict(), model_path)
            _torch_save_atomic(optimizer.state_dict(), optimizer_path)
            semantic_hash = state_dict_semantic_sha256(model_path)
            target_kind = config.get("auxiliary_target", "semantic_conflict_steps")
            expected_gate = config.get("milestone_semantic_gates", {}).get(
                str(examples_seen)
            )
            if (
                expected_gate is None
                and target_kind == "semantic_conflict_steps"
                and examples_seen == 8192
            ):
                expected_gate = config["source_v18_checkpoint_semantic_sha256"]
            if expected_gate is not None and semantic_hash != expected_gate:
                raise RuntimeError(
                    f"Milestone {examples_seen} semantic gate failed; refusing to continue"
                )
            meta = {
                "schema_version": 1,
                "status": "PASS",
                "examples_seen": examples_seen,
                "completed_updates": completed,
                "model_path": str(model_path),
                "optimizer_path": str(optimizer_path),
                "model_file_sha256": sha256_file(model_path),
                "model_semantic_sha256": semantic_hash,
                "optimizer_file_sha256": sha256_file(optimizer_path),
                "semantic_gate_expected": expected_gate,
                "semantic_gate_passed": expected_gate is None
                or semantic_hash == expected_gate,
                "metric_history_start_update": metric_history_start_update,
                "metric_history_complete": metric_history_start_update == 0,
                "total_losses": total_losses,
                "answer_losses": answer_losses,
                "auxiliary_losses": auxiliary_losses,
                "gradient_norms": gradient_norms,
            }
            atomic_json(meta_path, meta)
            print(f"saved verified milestone at {examples_seen} examples", flush=True)

    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    result = {
        "schema_version": 1,
        "status": "PASS" if peak <= float(config["max_reserved_memory_gb"]) else "FAIL",
        "phase": phase,
        "updates": updates,
        "examples_seen": updates * accumulation,
        "unique_examples_seen": updates * accumulation,
        "resumed_from_examples": start_examples,
        "metric_history_start_update": metric_history_start_update,
        "metric_history_start_examples": metric_history_start_update * accumulation,
        "metric_history_complete": metric_history_start_update == 0,
        "total_losses": total_losses,
        "answer_losses": answer_losses,
        "auxiliary_losses": auxiliary_losses,
        "gradient_norms": gradient_norms,
        "gradient_clipping_enabled": config.get("max_grad_norm") is not None,
        "max_grad_norm": config.get("max_grad_norm"),
        "elapsed_seconds_this_invocation": elapsed,
        "peak_reserved_gb": peak,
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "auxiliary_target": config.get("auxiliary_target", "semantic_conflict_steps"),
        "disclosure": config["disclosure"],
    }
    atomic_json(metrics_path, result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise RuntimeError("Scaling training exceeded the memory limit")
    return result
