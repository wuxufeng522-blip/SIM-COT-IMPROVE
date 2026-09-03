from __future__ import annotations

from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import gc
import json
import math
import random
import re
import time

import torch

from .error_cancellation_experiment import normalized_grouped_auxiliary_loss
from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, iter_icot_examples, load_official_model
from .oracle_weighting import tokenize_step_targets
from .self_corrected_data import canonical_hash, verify_frozen_manifest
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


WORD_RE = re.compile(r"[A-Za-z0-9]+")
EQUATION_RE = re.compile(r"^<<(.+)=([^=<>]+)>>$")


def _question_id(question: str) -> str:
    return sha256(question.strip().encode("utf-8")).hexdigest()


def _tokens(text: str) -> set[str]:
    return {value.lower() for value in WORD_RE.findall(text) if len(value) > 1}


def _jaccard(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def _result_text(step: str) -> str:
    body = step.removeprefix("<<").removesuffix(">>")
    return body.rsplit("=", 1)[-1].strip()


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        return
    atomic_json(path, payload)


def _compatible(
    recipient: dict[str, Any], donor: dict[str, Any], config: dict[str, Any]
) -> bool:
    if recipient["question_id"] == donor["question_id"]:
        return False
    if recipient["example"].answer == donor["example"].answer:
        return False
    if _jaccard(recipient["example"].question, donor["example"].question) > float(
        config["max_question_jaccard"]
    ):
        return False
    left_steps = recipient["example"].steps
    right_steps = donor["example"].steps
    if any(left == right for left, right in zip(left_steps, right_steps)):
        return False
    different_results = sum(
        _result_text(left) != _result_text(right)
        for left, right in zip(left_steps, right_steps)
    )
    if different_results < int(config["min_different_result_slots"]):
        return False
    ratio = donor["step_tokens"] / recipient["step_tokens"]
    return float(config["min_step_token_ratio"]) <= ratio <= float(
        config["max_step_token_ratio"]
    )


def _match_bucket(
    rows: list[dict[str, Any]], config: dict[str, Any], bucket_index: int
) -> dict[int, int]:
    edges: dict[int, list[int]] = {}
    for left in range(len(rows)):
        options = [right for right in range(len(rows)) if _compatible(rows[left], rows[right], config)]
        options.sort(
            key=lambda right: sha256(
                f"{config['run_id']}:{bucket_index}:{rows[left]['question_id']}:{rows[right]['question_id']}".encode(
                    "utf-8"
                )
            ).hexdigest()
        )
        edges[left] = options

    donor_to_recipient: dict[int, int] = {}

    def augment(recipient: int, visited: set[int]) -> bool:
        for donor in edges[recipient]:
            if donor in visited:
                continue
            visited.add(donor)
            previous = donor_to_recipient.get(donor)
            if previous is None or augment(previous, visited):
                donor_to_recipient[donor] = recipient
                return True
        return False

    order = sorted(range(len(rows)), key=lambda index: (len(edges[index]), rows[index]["question_id"]))
    for recipient in order:
        if not augment(recipient, set()):
            raise RuntimeError(
                f"Could not build a one-to-one semantic-conflict matching in bucket {bucket_index}; "
                f"recipient={recipient}, candidates={len(edges[recipient])}"
            )
    return {recipient: donor for donor, recipient in donor_to_recipient.items()}


def prepare_semantic_conflict_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    for path_key, hash_key in (
        ("checkpoint_path", "checkpoint_sha256"),
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
    ):
        path = root / config[path_key]
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Artifact missing or SHA-256 mismatch: {path}")

    tokenizer, _ = build_tokenizer(root / config["base_model_dir"])
    test_examples = list(iter_icot_examples(root / config["gsm_test_path"]))
    if len(test_examples) != int(config["test_examples"]):
        raise ValueError("Official GSM8K test count mismatch")
    test_ids = {_question_id(example.question) for example in test_examples}

    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for example in iter_icot_examples(root / config["gsm_train_path"]):
        question_id = _question_id(example.question)
        if len(example.steps) != 5 or question_id in seen or question_id in test_ids:
            continue
        seen.add(question_id)
        step_tokens = sum(
            len(tokenizer.encode(step, add_special_tokens=False)) + 1
            for step in example.steps
        )
        candidates.append(
            {
                "question_id": question_id,
                "example": example,
                "step_tokens": step_tokens,
            }
        )
    candidates.sort(
        key=lambda row: (
            sha256(f"{config['run_id']}:select:{row['question_id']}".encode("utf-8")).hexdigest(),
            row["question_id"],
        )
    )
    selected = candidates[: int(config["train_examples"])]
    if len(selected) != int(config["train_examples"]):
        raise RuntimeError("Insufficient unique exact-five-step GSM8K-Aug examples")

    # Sorting before fixed-size matching makes every donor trajectory similar in
    # token length while the hashed edge order prevents a semantic pairing bias.
    selected.sort(key=lambda row: (row["step_tokens"], row["question_id"]))
    bucket_size = int(config["donor_bucket_size"])
    if len(selected) % bucket_size:
        raise ValueError("train_examples must be divisible by donor_bucket_size")

    entries: list[dict[str, Any]] = []
    jaccards: list[float] = []
    ratios: list[float] = []
    different_slots: list[int] = []
    for bucket_index, start in enumerate(range(0, len(selected), bucket_size)):
        bucket = selected[start : start + bucket_size]
        matches = _match_bucket(bucket, config, bucket_index)
        for recipient_index, recipient in enumerate(bucket):
            donor = bucket[matches[recipient_index]]
            clean_steps = tuple(recipient["example"].steps)
            conflict_steps = tuple(donor["example"].steps)
            jaccard = _jaccard(recipient["example"].question, donor["example"].question)
            ratio = donor["step_tokens"] / recipient["step_tokens"]
            changed = sum(
                _result_text(left) != _result_text(right)
                for left, right in zip(clean_steps, conflict_steps)
            )
            row = {
                "question_id": recipient["question_id"],
                "problem": recipient["example"].question,
                "answer": recipient["example"].answer,
                "source": {
                    "kind": "official_gsm8k_aug_exact_five_step",
                    "source_file": config["gsm_train_path"],
                    "source_line": recipient["example"].idx,
                },
                "clean_steps": list(clean_steps),
                "semantic_conflict": {
                    "kind": "coherent_cross_question_five_step_derangement",
                    "donor_question_id": donor["question_id"],
                    "donor_source_line": donor["example"].idx,
                    "steps": list(conflict_steps),
                    "question_jaccard": jaccard,
                    "step_token_ratio": ratio,
                    "different_result_slots": changed,
                },
            }
            row["five_tuple_sha256"] = canonical_hash(row)
            entries.append(row)
            jaccards.append(jaccard)
            ratios.append(ratio)
            different_slots.append(changed)

    manifest = {
        "schema_version": 1,
        "dataset_family": "gsm8k_aug",
        "run_id": config["run_id"],
        "entries": entries,
        "test_problem_ids": sorted(test_ids),
        "generator_disclosure": config["disclosure"],
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    _immutable_json(root / config["manifest_path"], manifest)

    order = [row["question_id"] for row in entries]
    random.Random(int(config["seed"])).shuffle(order)
    schedule = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "seed": int(config["seed"]),
        "manifest_sha256": manifest["manifest_sha256"],
        "order": order,
        "gradient_accumulation_steps": int(config["gradient_accumulation_steps"]),
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    _immutable_json(root / config["schedule_path"], schedule)

    audit_rows = sorted(
        entries,
        key=lambda row: sha256(f"{config['run_id']}:audit:{row['question_id']}".encode("utf-8")).hexdigest(),
    )[:20]
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "train_examples": len(entries),
        "unique_recipient_questions": len({row["question_id"] for row in entries}),
        "unique_donor_questions": len(
            {row["semantic_conflict"]["donor_question_id"] for row in entries}
        ),
        "recipient_donor_overlap": sum(
            row["question_id"] == row["semantic_conflict"]["donor_question_id"]
            for row in entries
        ),
        "all_five_steps_different": all(
            all(left != right for left, right in zip(row["clean_steps"], row["semantic_conflict"]["steps"]))
            for row in entries
        ),
        "different_result_slots_min": min(different_slots),
        "question_jaccard_max": max(jaccards),
        "step_token_ratio_min": min(ratios),
        "step_token_ratio_max": max(ratios),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "audit_examples": audit_rows,
        "disclosure": config["disclosure"],
    }
    _immutable_json(root / config["audit_path"], audit)
    return audit


def load_frozen_data(config: dict[str, Any], root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = json.loads((root / config["manifest_path"]).read_text(encoding="utf-8"))
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    if canonical_hash(schedule) != schedule.get("schedule_sha256"):
        raise ValueError("Frozen schedule SHA-256 mismatch")
    if schedule.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise ValueError("Schedule belongs to another manifest")
    if set(schedule.get("order", ())) != {row["question_id"] for row in manifest["entries"]}:
        raise ValueError("Schedule does not contain each frozen training unit exactly once")
    return manifest, schedule


def redundant_steps(steps: Iterable[str]) -> tuple[str, ...]:
    """Insert a truth-preserving +0 identity into every explicit equation."""
    transformed: list[str] = []
    for step in steps:
        match = EQUATION_RE.fullmatch(step)
        if match is None:
            raise ValueError(f"Cannot construct an equation-matched redundant step: {step}")
        expression, result = match.groups()
        transformed.append(f"<<({expression})+0={result}>>")
    if len(transformed) != 5:
        raise ValueError("Exactly five redundant targets are required")
    return tuple(transformed)


def supervision_targets(
    arm: str, row: dict[str, Any], config: dict[str, Any]
) -> tuple[tuple[str, ...], tuple[float, ...], str]:
    specs = config.get("arm_specs")
    if specs is None:
        if arm != "semantic_conflict":
            raise ValueError(f"Unknown v18 supervision arm: {arm}")
        return tuple(row["semantic_conflict"]["steps"]), (1.0,) * 5, "semantic_conflict"
    if arm not in specs:
        raise ValueError(f"Unknown supervision arm: {arm}")
    target_kind = str(specs[arm]["target_kind"])
    if target_kind == "clean":
        targets = tuple(row["clean_steps"])
    elif target_kind == "redundant":
        targets = redundant_steps(row["clean_steps"])
    elif target_kind == "semantic_conflict":
        targets = tuple(row["semantic_conflict"]["steps"])
    else:
        raise ValueError(f"Unknown target kind: {target_kind}")
    step_weight = float(specs[arm]["step_weight"])
    if not 0 < step_weight <= 1:
        raise ValueError("Step supervision weight must be in (0, 1]")
    return targets, (step_weight,) * 5, target_kind


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def run_semantic_conflict_training(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    max_updates: int | None = None,
    phase: str = "train",
    save_checkpoint: bool = True,
    arm: str = "semantic_conflict",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if sha256_file(root / config["checkpoint_path"]) != config["checkpoint_sha256"]:
        raise ValueError("Starting Coconut checkpoint SHA-256 mismatch")
    manifest, schedule = load_frozen_data(config, root)
    by_id = {row["question_id"]: row for row in manifest["entries"]}
    accumulation = int(config["gradient_accumulation_steps"])
    full_updates = len(schedule["order"]) // accumulation
    updates = full_updates if max_updates is None else int(max_updates)
    if updates < 1 or updates > full_updates:
        raise ValueError("Invalid update count")

    if config.get("arm_specs") is None:
        output_dir = root / config["train_output_dir"] / phase
        configured_checkpoint_path = root / config["checkpoint_output_path"]
    else:
        if arm not in config["arm_specs"]:
            raise ValueError(f"Arm is not frozen in the config: {arm}")
        output_dir = root / config["output_root"] / arm / "train" / phase
        configured_checkpoint_path = (
            root / config["work_root"] / arm / "checkpoint_final.pt"
        )
    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        raise FileExistsError(f"Refusing to overwrite {metrics_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Semantic-conflict training requires CUDA")
    seed = int(config["seed"])
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
        allow_missing_auxiliary=True,
    )
    model.base_causallm.train()
    model.expainable_llm.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
    )
    total_losses: list[float] = []
    answer_losses: list[float] = []
    auxiliary_losses: list[float] = []
    gradient_norms: list[float] = []
    started = time.perf_counter()
    for update in range(updates):
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
                raise ValueError(f"Frozen example exceeds context limit: {row['question_id']}")
            batch = tensorize_smoke_example(encoded, device=device)
            target_steps, step_weights, target_kind = supervision_targets(arm, row, config)
            targets = tokenize_step_targets(tokenizer, target_steps)
            with _autocast(device, config["precision"]):
                losses = normalized_grouped_auxiliary_loss(
                    model,
                    batch,
                    targets,
                    step_weights,
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
                raise FloatingPointError(f"Non-finite loss at update {update + 1}")
            scaled.backward()
            micro_values.append(values)
            del batch, targets, losses, objective, scaled
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["max_grad_norm"]))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient at update {update + 1}")
        optimizer.step()
        torch.cuda.empty_cache()
        total_losses.append(sum(value[0] for value in micro_values) / accumulation)
        answer_losses.append(sum(value[1] for value in micro_values) / accumulation)
        auxiliary_losses.append(sum(value[2] for value in micro_values) / accumulation)
        gradient_norms.append(float(norm.detach().float()))
        completed = update + 1
        if completed == 1 or completed % int(config["log_every"]) == 0 or completed == updates:
            progress = {
                "schema_version": 1,
                "status": "RUNNING",
                "arm": arm,
                "target_kind": target_kind,
                "step_weight": step_weights[0],
                "phase": phase,
                "completed_updates": completed,
                "target_updates": updates,
                "examples_seen": completed * accumulation,
                "latest_total_loss": total_losses[-1],
                "latest_answer_loss": answer_losses[-1],
                "latest_auxiliary_loss": auxiliary_losses[-1],
                "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
                "manifest_sha256": manifest["manifest_sha256"],
                "schedule_sha256": schedule["schedule_sha256"],
            }
            atomic_json(output_dir / "progress.json", progress)
            print(
                f"{arm} {completed}/{updates}: total={total_losses[-1]:.5f}, "
                f"answer={answer_losses[-1]:.5f}, aux={auxiliary_losses[-1]:.5f}, "
                f"peak={progress['peak_reserved_gb']:.2f} GB",
                flush=True,
            )

    saved_path: Path | None = None
    checkpoint_hash: str | None = None
    if save_checkpoint:
        saved_path = configured_checkpoint_path
        if saved_path.exists():
            raise FileExistsError(f"Refusing to overwrite {saved_path}")
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
        "target_kind": supervision_targets(arm, manifest["entries"][0], config)[2],
        "step_weight": supervision_targets(arm, manifest["entries"][0], config)[1][0],
        "phase": phase,
        "updates": updates,
        "examples_seen": updates * accumulation,
        "unique_examples": updates * accumulation,
        "total_losses": total_losses,
        "answer_losses": answer_losses,
        "auxiliary_losses": auxiliary_losses,
        "gradient_norms": gradient_norms,
        "elapsed_seconds": elapsed,
        "peak_reserved_gb": peak,
        "checkpoint_path": str(saved_path) if saved_path else None,
        "checkpoint_sha256": checkpoint_hash,
        "starting_checkpoint_sha256": config["checkpoint_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "disclosure": config["disclosure"],
    }
    atomic_json(metrics_path, result)
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    if result["status"] != "PASS":
        raise RuntimeError("Training exceeded frozen GPU memory limit")
    return result
