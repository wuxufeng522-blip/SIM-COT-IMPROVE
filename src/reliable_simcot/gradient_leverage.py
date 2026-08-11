from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Iterable
import gc
import json
import math

import torch

from .causal_evaluation import clean_dev_nll
from .causal_experiment import (
    _canonical_hash,
    _output_root,
    _question_exact_hash,
    _question_id,
    _split_entries,
    _steps_hash,
    _work_root,
    _write_subset_dataset,
    causal_steps_and_weights,
    resolve_split_examples,
    run_training_arm,
    verify_causal_schedule,
)
from .m1_training import atomic_json, sha256_file
from .official_adapter import evaluate_checkpoint, iter_icot_examples, load_official_model
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


LEVERAGE_ARMS = (
    "answer_only",
    "clean_aux1",
    "causal_aux1",
    "clean_aux3",
    "causal_aux3",
)

ARM_SETTINGS = {
    "answer_only": ("clean", 0.0),
    "clean_aux1": ("clean", 1.0),
    "causal_aux1": ("noisy_equal", 1.0),
    "clean_aux3": ("clean", 3.0),
    "causal_aux3": ("noisy_equal", 3.0),
}


def _read_schedule(config: dict[str, Any], root: Path) -> dict[str, Any]:
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    verify_causal_schedule(schedule)
    if schedule["schedule_sha256"] != config["schedule_sha256"]:
        raise ValueError("Gradient-leverage config does not match the frozen causal schedule")
    return schedule


def verify_confirm_manifest(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Confirm manifest has no valid SHA-256")
    unhashed = dict(manifest)
    del unhashed["manifest_sha256"]
    if _canonical_hash(unhashed) != expected:
        raise ValueError("Confirm manifest SHA-256 mismatch")


def prepare_confirm_set(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    count = int(config["confirm_examples"])
    formal = _split_entries(schedule, "formal")
    if count <= 0 or count > len(formal):
        raise ValueError("confirm_examples falls outside the frozen formal split")
    entries = formal[:count]
    question_ids = [entry["question_id"] for entry in entries]
    if len(set(question_ids)) != count:
        raise ValueError("Confirm set contains duplicate questions")
    prior_ids = {
        entry["question_id"]
        for split in ("pilot", "dev")
        for entry in _split_entries(schedule, split)
    }
    if prior_ids & set(question_ids):
        raise ValueError("Confirm set leaks into prior pilot or development data")

    source_path = (root / config["dataset_path"]).resolve()
    confirm_path = (root / config["confirm_dataset_path"]).resolve()
    _write_subset_dataset(
        source_path,
        confirm_path,
        [int(entry["source_idx"]) for entry in entries],
    )
    confirm_sha = sha256_file(confirm_path)
    compact_entries = [
        {
            "position": position,
            "source_idx": entry["source_idx"],
            "question_id": entry["question_id"],
            "question_sha256": entry["question_sha256"],
            "clean_steps_sha256": entry["clean_steps_sha256"],
            "chain": entry["chain"],
        }
        for position, entry in enumerate(entries)
    ]
    manifest = {
        "schema_version": 1,
        "run_id": config["prepare_run_id"],
        "status": "PASS",
        "selection_rule": "first N entries of the previously frozen, unused causal formal split",
        "examples": count,
        "source_dataset_path": str(source_path),
        "source_dataset_sha256": config["dataset_sha256"],
        "source_schedule_sha256": schedule["schedule_sha256"],
        "confirm_dataset_path": str(confirm_path),
        "confirm_dataset_sha256": confirm_sha,
        "question_order_sha256": sha256(
            json.dumps(question_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "prior_pilot_overlap": 0,
        "prior_dev_overlap": 0,
        "official_test_opened": False,
        "family_counts": dict(Counter(entry["chain"]["family"] for entry in entries)),
        "pivot_counts": dict(Counter(str(entry["chain"]["pivot"]) for entry in entries)),
        "entries": compact_entries,
    }
    manifest["manifest_sha256"] = _canonical_hash(manifest)
    atomic_json((root / config["confirm_manifest_path"]).resolve(), manifest)
    audit = dict(manifest)
    audit.pop("entries")
    atomic_json((root / config["confirm_audit_path"]).resolve(), audit)
    return manifest


def _read_confirm(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[Any]]:
    manifest = json.loads(
        (root / config["confirm_manifest_path"]).read_text(encoding="utf-8")
    )
    verify_confirm_manifest(manifest)
    path = Path(manifest["confirm_dataset_path"])
    if sha256_file(path) != manifest["confirm_dataset_sha256"]:
        raise ValueError("Frozen confirm dataset SHA-256 mismatch")
    examples = list(iter_icot_examples(path))
    if len(examples) != manifest["examples"]:
        raise ValueError("Frozen confirm dataset is incomplete")
    for example, entry in zip(examples, manifest["entries"], strict=True):
        if _question_id(example) != entry["question_id"]:
            raise ValueError(f"Confirm question ID changed at position {example.idx}")
        if _question_exact_hash(example) != entry["question_sha256"]:
            raise ValueError(f"Confirm question bytes changed at position {example.idx}")
        if _steps_hash(example) != entry["clean_steps_sha256"]:
            raise ValueError(f"Confirm clean steps changed at position {example.idx}")
    return manifest, examples


def leverage_run_id(config: dict[str, Any], arm: str, seed: int) -> str:
    if arm not in LEVERAGE_ARMS:
        raise ValueError(f"Unknown gradient-leverage arm: {arm}")
    return f"{config['training_run_prefix']}-{seed}-{arm}"


def leverage_directory(arm: str, seed: int, *, sanity: bool = False) -> str:
    prefix = "sanity" if sanity else f"seed_{seed}"
    return f"{prefix}/{arm}"


def run_leverage_training(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    updates_override: int | None = None,
    sanity: bool = False,
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if arm not in LEVERAGE_ARMS:
        raise ValueError(f"Unknown gradient-leverage arm: {arm}")
    if seed not in config["seeds"]:
        raise ValueError("Seed is not preregistered")
    root = Path(project_root).resolve()
    confirm_manifest = json.loads(
        (root / config["confirm_manifest_path"]).read_text(encoding="utf-8")
    )
    verify_confirm_manifest(confirm_manifest)
    target_arm, auxiliary_scale = ARM_SETTINGS[arm]
    directory = leverage_directory(arm, seed, sanity=sanity)
    result = run_training_arm(
        config,
        split="pilot",
        arm=target_arm,
        coverage=config["coverage"],
        project_root=project_root,
        updates_override=updates_override,
        save_checkpoint=save_checkpoint,
        auxiliary_scale=auxiliary_scale,
        training_seed=seed,
        directory_override=directory,
        run_id_override=(
            config["sanity_run_id"]
            if sanity
            else leverage_run_id(config, arm, seed)
        ),
    )
    result.update(
        {
            "leverage_arm": arm,
            "target_kind": target_arm,
            "auxiliary_scale": auxiliary_scale,
            "confirm_manifest_sha256": confirm_manifest["manifest_sha256"],
        }
    )
    output_dir = _output_root(root, config) / "pilot" / directory
    atomic_json(output_dir / "metrics.json", result)
    return result


def run_leverage_sanity(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, _ = _read_confirm(config, root)
    arms: dict[str, Any] = {}
    for arm in LEVERAGE_ARMS:
        arms[arm] = run_leverage_training(
            config,
            arm=arm,
            seed=config["seeds"][0],
            project_root=root,
            updates_override=config["sanity_updates"],
            sanity=True,
            save_checkpoint=False,
        )
    answer_only = arms["answer_only"]
    if any(
        abs(total - answer) > 1e-5
        for total, answer in zip(
            answer_only["update_total_losses"],
            answer_only["update_answer_losses"],
            strict=True,
        )
    ):
        raise AssertionError("lambda_aux=0 does not reduce to answer-only loss")
    for arm in ("clean_aux3", "causal_aux3"):
        if any(
            abs(total - (answer + 3.0 * auxiliary)) > 1e-4
            for total, answer, auxiliary in zip(
                arms[arm]["update_total_losses"],
                arms[arm]["update_answer_losses"],
                arms[arm]["update_auxiliary_losses"],
                strict=True,
            )
        ):
            raise AssertionError(f"lambda_aux=3 objective mismatch in {arm}")
    if arms["causal_aux1"]["corrupted_examples"] <= 0:
        raise AssertionError("Sanity window did not exercise a causal corruption")
    if all(
        abs(clean - causal) <= 1e-5
        for clean, causal in zip(
            arms["clean_aux1"]["update_auxiliary_losses"],
            arms["causal_aux1"]["update_auxiliary_losses"],
            strict=True,
        )
    ):
        raise AssertionError("Causal and clean auxiliary targets never diverged in sanity")
    result = {
        "schema_version": 1,
        "run_id": config["sanity_run_id"],
        "status": "PASS",
        "confirm_manifest_sha256": manifest["manifest_sha256"],
        "arms": arms,
        "max_peak_reserved_gb": max(item["peak_reserved_gb"] for item in arms.values()),
    }
    result["gate_passed"] = all(item["gate_passed"] for item in arms.values())
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json((root / config["sanity_path"]).resolve(), result)
    return result


def _gradient_pair_metrics(
    left: Iterable[torch.Tensor | None], right: Iterable[torch.Tensor | None]
) -> tuple[float, float, float]:
    left_sq = 0.0
    right_sq = 0.0
    dot = 0.0
    for first, second in zip(left, right, strict=True):
        if first is not None:
            left_sq += float(first.detach().float().square().sum().item())
        if second is not None:
            right_sq += float(second.detach().float().square().sum().item())
        if first is not None and second is not None:
            dot += float((first.detach().float() * second.detach().float()).sum().item())
    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    cosine = dot / (left_norm * right_norm) if left_norm > 0 and right_norm > 0 else 0.0
    return left_norm, right_norm, cosine


def _layer_metrics(
    answer: list[torch.Tensor | None],
    clean: list[torch.Tensor | None],
    noisy: list[torch.Tensor | None],
) -> dict[str, float]:
    answer_norm, clean_norm, clean_answer_cos = _gradient_pair_metrics(answer, clean)
    _, noisy_norm, noisy_answer_cos = _gradient_pair_metrics(answer, noisy)
    _, _, clean_noisy_cos = _gradient_pair_metrics(clean, noisy)
    clean_dot = clean_answer_cos * clean_norm * answer_norm
    noisy_dot = noisy_answer_cos * noisy_norm * answer_norm
    return {
        "answer_norm": answer_norm,
        "clean_aux_norm": clean_norm,
        "noisy_aux_norm": noisy_norm,
        "clean_aux_to_answer_norm_ratio": clean_norm / answer_norm if answer_norm else 0.0,
        "noisy_aux_to_answer_norm_ratio": noisy_norm / answer_norm if answer_norm else 0.0,
        "clean_aux_answer_cosine": clean_answer_cos,
        "noisy_aux_answer_cosine": noisy_answer_cos,
        "clean_noisy_aux_cosine": clean_noisy_cos,
        "noisy_minus_clean_answer_projection": (
            (noisy_dot - clean_dot) / (answer_norm**2) if answer_norm else 0.0
        ),
    }


def run_gradient_audit(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = _read_schedule(config, root)
    examples = resolve_split_examples(
        schedule,
        split="pilot",
        dataset_path=(root / config["dataset_path"]).resolve(),
    )
    entries = _split_entries(schedule, "pilot")
    per_cell = int(config["gradient_examples_per_cell"])
    chosen: list[tuple[Any, dict[str, Any]]] = []
    counts: Counter[str] = Counter()
    for example, entry in zip(examples, entries, strict=True):
        if int(entry["coverage_tier"]) != 0:
            continue
        cell = f"{entry['chain']['family']}@{entry['chain']['pivot']}"
        if counts[cell] < per_cell:
            counts[cell] += 1
            chosen.append((example, entry))
    expected_cells = 9
    if len(counts) != expected_cells or len(chosen) != expected_cells * per_cell:
        raise ValueError("Could not build the balanced gradient-audit sample")

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
    if len(blocks) != config["expected_transformer_blocks"]:
        raise ValueError("Unexpected number of transformer blocks")
    parameter_groups = [
        [parameter for parameter in block.parameters() if parameter.requires_grad]
        for block in blocks
    ]
    flat_parameters = [parameter for group in parameter_groups for parameter in group]
    group_sizes = [len(group) for group in parameter_groups]
    rows: list[dict[str, Any]] = []

    for number, (example, entry) in enumerate(chosen, start=1):
        clean_steps, clean_weights = causal_steps_and_weights(
            "clean", example, entry, coverage=config["coverage"]
        )
        noisy_steps, noisy_weights = causal_steps_and_weights(
            "noisy_equal", example, entry, coverage=config["coverage"]
        )
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=config["latent_stage"],
            c_thought=config["c_thought"],
        )
        batch = tensorize_smoke_example(encoded, device=device)
        clean_targets = tokenize_step_targets(tokenizer, clean_steps)
        noisy_targets = tokenize_step_targets(tokenizer, noisy_steps)
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clean_losses = grouped_auxiliary_loss(
                model,
                batch,
                clean_targets,
                clean_weights,
                latent_id=token_ids["<|latent|>"],
                c_thought=config["c_thought"],
            )
            noisy_losses = grouped_auxiliary_loss(
                model,
                batch,
                noisy_targets,
                noisy_weights,
                latent_id=token_ids["<|latent|>"],
                c_thought=config["c_thought"],
            )
        answer_grads = list(
            torch.autograd.grad(
                clean_losses["answer_loss"],
                flat_parameters,
                retain_graph=True,
                allow_unused=True,
            )
        )
        clean_grads = list(
            torch.autograd.grad(
                clean_losses["auxiliary_loss"],
                flat_parameters,
                retain_graph=False,
                allow_unused=True,
            )
        )
        noisy_grads = list(
            torch.autograd.grad(
                noisy_losses["auxiliary_loss"],
                flat_parameters,
                retain_graph=False,
                allow_unused=True,
            )
        )
        layer_rows: list[dict[str, Any]] = []
        offset = 0
        for layer, size in enumerate(group_sizes):
            end = offset + size
            layer_rows.append(
                {
                    "layer": layer,
                    **_layer_metrics(
                        answer_grads[offset:end],
                        clean_grads[offset:end],
                        noisy_grads[offset:end],
                    ),
                }
            )
            offset = end
        rows.append(
            {
                "source_idx": example.idx,
                "cell": f"{entry['chain']['family']}@{entry['chain']['pivot']}",
                "chain_sha256": entry["chain"]["chain_sha256"],
                "answer_loss": float(clean_losses["answer_loss"].detach().float().item()),
                "clean_auxiliary_loss": float(clean_losses["auxiliary_loss"].detach().float().item()),
                "noisy_auxiliary_loss": float(noisy_losses["auxiliary_loss"].detach().float().item()),
                "layers": layer_rows,
            }
        )
        del answer_grads, clean_grads, noisy_grads, clean_losses, noisy_losses, batch
        print(f"gradient audit: {number}/{len(chosen)}", flush=True)

    aggregate: list[dict[str, Any]] = []
    for layer in range(len(blocks)):
        layer_values = [row["layers"][layer] for row in rows]
        keys = [key for key in layer_values[0] if key != "layer"]
        aggregate.append(
            {
                "layer": layer,
                **{
                    f"median_{key}": median(float(item[key]) for item in layer_values)
                    for key in keys
                },
                **{
                    f"mean_{key}": mean(float(item[key]) for item in layer_values)
                    for key in keys
                },
                "noisy_answer_conflict_rate": mean(
                    float(item["noisy_aux_answer_cosine"] < 0) for item in layer_values
                ),
            }
        )
    ratio_layers = sum(
        item["median_clean_aux_to_answer_norm_ratio"] >= config["gradient_ratio_threshold"]
        for item in aggregate
    )
    changed_layers = sum(
        item["median_clean_noisy_aux_cosine"] <= config["gradient_direction_cosine_threshold"]
        for item in aggregate
    )
    pathway_passed = (
        ratio_layers >= config["gradient_required_layers"]
        and changed_layers >= config["gradient_required_layers"]
    )
    result = {
        "schema_version": 1,
        "run_id": config["gradient_run_id"],
        "status": "PASS",
        "verdict": (
            "GRADIENT_PATHWAY_PRESENT" if pathway_passed else "GRADIENT_PATHWAY_WEAK"
        ),
        "gate_passed": pathway_passed,
        "examples": len(chosen),
        "cell_counts": dict(counts),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "thresholds": {
            "gradient_ratio": config["gradient_ratio_threshold"],
            "direction_cosine": config["gradient_direction_cosine_threshold"],
            "required_layers": config["gradient_required_layers"],
        },
        "layers_meeting_ratio": ratio_layers,
        "layers_meeting_direction_change": changed_layers,
        "aggregate_layers": aggregate,
        "example_rows": rows,
        "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 1024**3,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "official_test_opened": False,
    }
    result["within_memory_limit"] = result["peak_reserved_gb"] <= config["max_reserved_memory_gb"]
    if not result["within_memory_limit"]:
        result["status"] = "FAIL_MEMORY"
    atomic_json((root / config["gradient_audit_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def chain_preference_nll(
    model,
    tokenizer,
    token_ids: dict[str, int],
    config: dict[str, Any],
    *,
    examples: list[Any],
    entries: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    limit = int(config["preference_examples"])
    clean_loss_sum = 0.0
    clean_tokens = 0
    corrupt_loss_sum = 0.0
    corrupt_tokens = 0
    model.base_causallm.eval()
    model.expainable_llm.eval()
    for number, (example, entry) in enumerate(
        zip(examples[:limit], entries[:limit], strict=True), start=1
    ):
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=config["latent_stage"],
            c_thought=config["c_thought"],
        )
        batch = tensorize_smoke_example(encoded, device=device)
        clean_targets = tokenize_step_targets(tokenizer, example.steps[:5])
        corrupt_targets = tokenize_step_targets(tokenizer, entry["chain"]["corrupted_steps"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clean = grouped_auxiliary_loss(
                model,
                batch,
                clean_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=config["c_thought"],
            )
            corrupt = grouped_auxiliary_loss(
                model,
                batch,
                corrupt_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=config["c_thought"],
            )
        clean_loss_sum += sum(clean["step_losses"].float().tolist())
        clean_tokens += sum(clean["token_counts"])
        corrupt_loss_sum += sum(corrupt["step_losses"].float().tolist())
        corrupt_tokens += sum(corrupt["token_counts"])
        if number == 1 or number % config["nll_log_every"] == 0 or number == limit:
            print(f"chain preference: {number}/{limit}", flush=True)
    clean_nll = clean_loss_sum / clean_tokens
    corrupt_nll = corrupt_loss_sum / corrupt_tokens
    return {
        "examples": limit,
        "clean_step_token_nll": clean_nll,
        "corrupt_step_token_nll": corrupt_nll,
        "clean_preference_margin": corrupt_nll - clean_nll,
        "interpretation": "positive margin means lower NLL for the clean chain",
    }


def _training_metrics_path(
    config: dict[str, Any], root: Path, arm: str, seed: int
) -> Path:
    return (
        _output_root(root, config)
        / "pilot"
        / leverage_directory(arm, seed)
        / "metrics.json"
    )


def evaluate_leverage_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in LEVERAGE_ARMS or seed not in config["seeds"]:
        raise ValueError("Unregistered leverage arm or seed")
    root = Path(project_root).resolve()
    manifest, examples = _read_confirm(config, root)
    train_metrics = json.loads(
        _training_metrics_path(config, root, arm, seed).read_text(encoding="utf-8")
    )
    if (
        train_metrics.get("status") != "PASS"
        or train_metrics.get("leverage_arm") != arm
        or train_metrics.get("seed") != seed
    ):
        raise ValueError("Leverage training did not pass")
    checkpoint = Path(train_metrics["checkpoint_path"])
    if sha256_file(checkpoint) != train_metrics["checkpoint_sha256"]:
        raise ValueError("Leverage checkpoint SHA-256 mismatch")
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    confirm_path = Path(manifest["confirm_dataset_path"])
    clean_nll = clean_dev_nll(
        model,
        tokenizer,
        token_ids,
        {**config, "dev_examples": config["confirm_examples"]},
        dev_path=confirm_path,
        device=device,
    )
    preference = chain_preference_nll(
        model,
        tokenizer,
        token_ids,
        config,
        examples=examples,
        entries=manifest["entries"],
        device=device,
    )
    output_dir = _output_root(root, config) / "eval" / f"seed_{seed}" / arm
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=confirm_path,
        output_dir=output_dir,
        device=device,
        latent_tokens=config["latent_stage"] * config["c_thought"],
        max_new_tokens=config["max_new_tokens"],
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=resume,
        flush_every=config["flush_every"],
    )
    metrics.update(
        {
            "run_id": config["evaluation_run_id"],
            "leverage_arm": arm,
            "seed": seed,
            "training_run_id": train_metrics["run_id"],
            "checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "confirm_manifest_sha256": manifest["manifest_sha256"],
            "clean_nll": clean_nll,
            "chain_preference": preference,
            "official_test_opened": False,
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "accuracy": [float(row["accuracy"]) for row in rows],
        "answer_nll": [float(row["clean_nll"]["answer_nll"]) for row in rows],
        "clean_step_token_nll": [
            float(row["clean_nll"]["step_token_nll"]) for row in rows
        ],
        "clean_preference_margin": [
            float(row["chain_preference"]["clean_preference_margin"]) for row in rows
        ],
    }
    return {
        key: {
            "values": values,
            "mean": mean(values),
            "population_std": pstdev(values),
        }
        for key, values in fields.items()
    }


def analyze_leverage(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest, _ = _read_confirm(config, root)
    audit = json.loads((root / config["gradient_audit_path"]).read_text(encoding="utf-8"))
    rows: dict[str, dict[int, dict[str, Any]]] = {arm: {} for arm in LEVERAGE_ARMS}
    for arm in LEVERAGE_ARMS:
        for seed in config["seeds"]:
            path = _output_root(root, config) / "eval" / f"seed_{seed}" / arm / "metrics.json"
            row = json.loads(path.read_text(encoding="utf-8"))
            if row.get("confirm_manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError("Evaluation does not match the frozen confirm set")
            rows[arm][seed] = row
    summaries = {
        arm: _arm_summary([rows[arm][seed] for seed in config["seeds"]])
        for arm in LEVERAGE_ARMS
    }
    contrasts: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        accuracy = {arm: float(rows[arm][seed]["accuracy"]) for arm in LEVERAGE_ARMS}
        nll = {
            arm: float(rows[arm][seed]["clean_nll"]["step_token_nll"])
            for arm in LEVERAGE_ARMS
        }
        preference = {
            arm: float(rows[arm][seed]["chain_preference"]["clean_preference_margin"])
            for arm in LEVERAGE_ARMS
        }
        contrasts.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "damage_aux1": accuracy["clean_aux1"] - accuracy["causal_aux1"],
                "damage_aux3": accuracy["clean_aux3"] - accuracy["causal_aux3"],
                "causal_aux1_minus_clean_aux1": accuracy["causal_aux1"] - accuracy["clean_aux1"],
                "clean_aux1_minus_answer_only": accuracy["clean_aux1"] - accuracy["answer_only"],
                "causal_aux1_step_nll_not_worse": nll["causal_aux1"] <= nll["clean_aux1"],
                "causal_aux1_preference_not_worse": preference["causal_aux1"] >= preference["clean_aux1"],
                "clean_step_token_nll": nll,
                "clean_preference_margin": preference,
            }
        )
    mean_damage1 = mean(item["damage_aux1"] for item in contrasts)
    mean_damage3 = mean(item["damage_aux3"] for item in contrasts)
    mean_self_correction = mean(
        item["causal_aux1_minus_clean_aux1"] for item in contrasts
    )
    damage_gate = {
        "mean_aux3_damage_at_least_2pp": mean_damage3 >= config["minimum_aux3_damage"],
        "aux3_damage_in_at_least_two_seeds": sum(
            item["damage_aux3"] > 0 for item in contrasts
        ) >= 2,
        "dose_response_at_least_1pp": (
            mean_damage3 - mean_damage1 >= config["minimum_dose_response"]
        ),
    }
    self_gate = {
        "mean_aux1_gain_at_least_1pp": (
            mean_self_correction >= config["minimum_self_correction_gain"]
        ),
        "gain_in_at_least_two_seeds": sum(
            item["causal_aux1_minus_clean_aux1"] > 0 for item in contrasts
        ) >= 2,
        "preference_not_worse_in_at_least_two_seeds": sum(
            item["causal_aux1_preference_not_worse"] for item in contrasts
        ) >= 2,
        "step_nll_not_worse_in_at_least_two_seeds": sum(
            item["causal_aux1_step_nll_not_worse"] for item in contrasts
        ) >= 2,
    }
    damage_passed = all(damage_gate.values())
    self_passed = all(self_gate.values())
    pathway_passed = bool(audit.get("gate_passed"))
    accuracy_gain_only = mean_self_correction > 0 and not self_passed
    if self_passed:
        verdict = "MULTISEED_SELF_CORRECTION_EVIDENCE"
    elif damage_passed:
        verdict = "AUXILIARY_STEP_ERRORS_HAVE_ANSWER_LEVERAGE"
    elif accuracy_gain_only:
        verdict = "REGULARIZATION_OR_ROBUSTNESS_NOT_SELF_CORRECTION"
    elif pathway_passed:
        verdict = "GRADIENT_PATHWAY_PRESENT_BUT_ANSWER_ROBUST"
    else:
        verdict = "GRADIENT_PATHWAY_WEAK"
    result = {
        "schema_version": 1,
        "run_id": config["analysis_run_id"],
        "status": "PASS",
        "verdict": verdict,
        "gradient_pathway_passed": pathway_passed,
        "answer_leverage_gate_passed": damage_passed,
        "self_correction_gate_passed": self_passed,
        "mean_damage_aux1": mean_damage1,
        "mean_damage_aux3": mean_damage3,
        "mean_causal_aux1_minus_clean_aux1": mean_self_correction,
        "damage_gate": damage_gate,
        "self_correction_gate": self_gate,
        "arm_summaries": summaries,
        "per_seed_contrasts": contrasts,
        "gradient_audit_run_id": audit.get("run_id"),
        "confirm_manifest_sha256": manifest["manifest_sha256"],
        "official_test_opened": False,
        "claim_boundary": (
            "Three-seed 64-update controlled auxiliary-gradient experiment on a frozen "
            "official-training-source confirm split; no natural-noise or official-test claim."
        ),
    }
    atomic_json((root / config["analysis_path"]).resolve(), result)
    return result
