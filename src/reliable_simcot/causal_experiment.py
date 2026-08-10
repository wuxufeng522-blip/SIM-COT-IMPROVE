from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import gc
import json
import math
import random
import time

import torch

from .causal_corruptions import CAUSAL_FAMILIES, CausalChain, causal_chains_by_cell
from .m1_training import atomic_json, sha256_file
from .official_adapter import (
    OfficialExample,
    build_tokenizer,
    iter_icot_examples,
    load_official_model,
)
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


CAUSAL_ARMS = (
    "clean",
    "noisy_equal",
    "uniform_attenuation",
    "pivot_only",
    "causal_raw",
    "causal_normalized",
)
PILOT_ARMS = ("clean", "noisy_equal")
COVERAGES = (25, 50, 75)
CELL_CYCLE = (
    ("numeric_propagation", 0),
    ("operator_propagation", 1),
    ("quantity_propagation", 2),
    ("numeric_propagation", 1),
    ("operator_propagation", 2),
    ("quantity_propagation", 0),
    ("numeric_propagation", 2),
    ("operator_propagation", 0),
    ("quantity_propagation", 1),
)


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha256(encoded).hexdigest()


def _question_id(example: OfficialExample) -> str:
    return sha256(example.question.strip().encode("utf-8")).hexdigest()


def _question_exact_hash(example: OfficialExample) -> str:
    return sha256(example.question.encode("utf-8")).hexdigest()


def _steps_hash(example: OfficialExample) -> str:
    return sha256(
        json.dumps(tuple(example.steps[:5]), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _output_root(root: Path, config: dict[str, Any]) -> Path:
    return (root / config["output_root"]).resolve()


def _work_root(root: Path, config: dict[str, Any]) -> Path:
    return (root / config["work_root"]).resolve()


def _run_id(config: dict[str, Any], split: str, arm: str, coverage: int | None) -> str:
    if split == "pilot":
        key = "clean" if arm == "clean" else f"noisy_equal_{coverage}"
        return config["pilot_run_ids"][key]
    return config["formal_run_ids"][arm]


def _arm_dir_name(split: str, arm: str, coverage: int | None) -> str:
    if split == "pilot" and arm == "noisy_equal":
        return f"noisy_equal_{coverage}"
    return arm


def _split_entries(schedule: dict[str, Any], split: str) -> list[dict[str, Any]]:
    if split not in schedule["splits"]:
        raise ValueError(f"Unknown schedule split: {split}")
    return schedule["splits"][split]["entries"]


def verify_causal_schedule(schedule: dict[str, Any]) -> None:
    expected = schedule.get("schedule_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Frozen causal schedule has no valid SHA-256")
    unhashed = dict(schedule)
    del unhashed["schedule_sha256"]
    if _canonical_hash(unhashed) != expected:
        raise ValueError("Frozen causal schedule SHA-256 mismatch")
    split_ids = {
        split: {entry["question_id"] for entry in details["entries"]}
        for split, details in schedule["splits"].items()
    }
    names = tuple(split_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            if split_ids[left] & split_ids[right]:
                raise ValueError(f"Question leakage between {left} and {right}")


def _tier_requests(size: int, *, offset: int = 0) -> list[tuple[str, int]]:
    return [
        CELL_CYCLE[(offset + index) % len(CELL_CYCLE)]
        for index in range(size)
    ]


def _cell_counts(entries: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(
        Counter(
            f"{entry['chain']['family']}@{entry['chain']['pivot']}"
            for entry in entries
        )
    )


def _marginal_counts(entries: Iterable[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int]]:
    rows = list(entries)
    families = Counter(entry["chain"]["family"] for entry in rows)
    pivots = Counter(str(entry["chain"]["pivot"]) for entry in rows)
    return dict(families), dict(pivots)


def _balanced(entries: Iterable[dict[str, Any]]) -> bool:
    families, pivots = _marginal_counts(entries)
    return (
        set(families) == set(CAUSAL_FAMILIES)
        and set(pivots) == {"0", "1", "2"}
        and max(families.values()) - min(families.values()) <= 1
        and max(pivots.values()) - min(pivots.values()) <= 1
    )


def _fits_context(
    example: OfficialExample,
    chains: dict[tuple[str, int], CausalChain],
    tokenizer,
    token_ids: dict[str, int],
    config: dict[str, Any],
) -> bool:
    maximum = int(config["max_sequence_tokens"])
    clean = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=config["latent_stage"],
        c_thought=config["c_thought"],
    )
    if len(clean.input_ids) > maximum or clean.maximum_auxiliary_length > maximum:
        return False
    for chain in chains.values():
        corrupted = replace(
            example,
            steps=tuple(chain.corrupted_steps) + tuple(example.steps[5:]),
        )
        encoded = encode_smoke_example(
            corrupted,
            tokenizer,
            token_ids,
            latent_stage=config["latent_stage"],
            c_thought=config["c_thought"],
        )
        if len(encoded.input_ids) > maximum or encoded.maximum_auxiliary_length > maximum:
            return False
    return True


def _write_subset_dataset(
    source_path: Path, destination: Path, source_indices: list[int]
) -> None:
    wanted = set(source_indices)
    lines: dict[int, str] = {}
    with source_path.open("r", encoding="utf-8", newline="") as handle:
        for idx, line in enumerate(handle):
            if idx in wanted:
                lines[idx] = line
    if len(lines) != len(source_indices):
        raise ValueError("Could not recover every frozen dev source line")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for idx in source_indices:
            line = lines[idx]
            handle.write(line if line.endswith(("\n", "\r")) else line + "\n")
    temporary.replace(destination)


def _schedule_entry(
    example: OfficialExample,
    chain: CausalChain,
    *,
    position: int,
    coverage_tier: int | None,
    input_tokens: int,
    clean_auxiliary_tokens: int,
) -> dict[str, Any]:
    return {
        "position": position,
        "source_idx": example.idx,
        "question_id": _question_id(example),
        "question_sha256": _question_exact_hash(example),
        "clean_steps_sha256": _steps_hash(example),
        "coverage_tier": coverage_tier,
        "input_tokens": input_tokens,
        "maximum_clean_auxiliary_tokens": clean_auxiliary_tokens,
        "chain": chain.to_record(),
    }


def create_causal_schedule(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
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
        raise ValueError("A frozen R020 natural-noise audit manifest is required")
    if audit.get("dataset_sha256") != config["dataset_sha256"]:
        raise ValueError("Audit and causal schedule do not use the same source dataset")
    excluded = set(audit["selected_question_ids"])

    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    seed = int(config["seed"])
    candidates: list[
        tuple[str, str, OfficialExample, dict[tuple[str, int], CausalChain]]
    ] = []
    rejection_counts: Counter[str] = Counter()
    seen_questions: set[str] = set()
    for example in iter_icot_examples(dataset_path):
        question_id = _question_id(example)
        if question_id in seen_questions:
            rejection_counts["duplicate_question"] += 1
            continue
        seen_questions.add(question_id)
        if question_id in excluded:
            rejection_counts["natural_audit_question"] += 1
            continue
        chains = causal_chains_by_cell(example)
        if not chains:
            rejection_counts["no_verified_causal_chain"] += 1
            continue
        priority = sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
        candidates.append((priority, question_id, example, chains))
    candidates.sort(key=lambda item: (item[0], item[1]))

    by_cell: dict[tuple[str, int], list[tuple[Any, ...]]] = {
        cell: [candidate for candidate in candidates if cell in candidate[3]]
        for cell in CELL_CYCLE
    }
    minimum_cell = min(len(value) for value in by_cell.values())
    if minimum_cell < config["minimum_candidates_per_cell"]:
        raise ValueError(
            f"Smallest causal cell has {minimum_cell} candidates; "
            f"need {config['minimum_candidates_per_cell']}"
        )

    used: set[str] = set()
    cursors = {cell: 0 for cell in CELL_CYCLE}
    context_cache: dict[str, tuple[int, int] | None] = {}

    def take(cell: tuple[str, int]) -> tuple[Any, ...]:
        pool = by_cell[cell]
        while cursors[cell] < len(pool):
            item = pool[cursors[cell]]
            cursors[cell] += 1
            if item[1] in used:
                continue
            if item[1] not in context_cache:
                if not _fits_context(item[2], item[3], tokenizer, token_ids, config):
                    context_cache[item[1]] = None
                    rejection_counts["context_length"] += 1
                else:
                    encoded = encode_smoke_example(
                        item[2],
                        tokenizer,
                        token_ids,
                        latent_stage=config["latent_stage"],
                        c_thought=config["c_thought"],
                    )
                    context_cache[item[1]] = (
                        len(encoded.input_ids),
                        encoded.maximum_auxiliary_length,
                    )
            lengths = context_cache[item[1]]
            if lengths is None:
                continue
            used.add(item[1])
            return (*item, *lengths)
        raise ValueError(f"Causal cell exhausted during allocation: {cell}")

    def allocate(split: str, tier_sizes: list[int]) -> list[dict[str, Any]]:
        allocated: list[dict[str, Any]] = []
        request_offset = 0
        for tier, tier_size in enumerate(tier_sizes):
            tier_entries: list[dict[str, Any]] = []
            for cell in _tier_requests(tier_size, offset=request_offset):
                _, _, example, chains, input_tokens, max_aux = take(cell)
                tier_entries.append(
                    _schedule_entry(
                        example,
                        chains[cell],
                        position=-1,
                        coverage_tier=tier if split != "dev" else None,
                        input_tokens=input_tokens,
                        clean_auxiliary_tokens=max_aux,
                    )
                )
            if not _balanced(tier_entries):
                raise AssertionError(f"Unbalanced {split} tier {tier}")
            allocated.extend(tier_entries)
            if not _balanced(allocated):
                raise AssertionError(f"Unbalanced cumulative {split} prefix through tier {tier}")
            request_offset = (request_offset + tier_size) % len(CELL_CYCLE)
        rng = random.Random(seed + {"pilot": 101, "dev": 202, "formal": 303}[split])
        rng.shuffle(allocated)
        for position, entry in enumerate(allocated):
            entry["position"] = position
        return allocated

    pilot_size = int(config["pilot_examples"])
    dev_size = int(config["dev_examples"])
    formal_size = int(config["formal_examples"])
    if pilot_size % 4 or formal_size % 4:
        raise ValueError("Pilot and formal sizes must be divisible into four coverage tiers")
    pilot_entries = allocate("pilot", [pilot_size // 4] * 4)
    dev_entries = allocate("dev", [dev_size])
    formal_entries = allocate("formal", [formal_size // 4] * 4)

    dev_dataset_path = (root / config["dev_dataset_path"]).resolve()
    _write_subset_dataset(
        dataset_path,
        dev_dataset_path,
        [entry["source_idx"] for entry in dev_entries],
    )
    dev_sha = sha256_file(dev_dataset_path)

    def split_record(entries: list[dict[str, Any]]) -> dict[str, Any]:
        tier_counts = Counter(str(entry["coverage_tier"]) for entry in entries)
        return {
            "examples": len(entries),
            "question_order_sha256": sha256(
                json.dumps(
                    [entry["question_id"] for entry in entries], separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "coverage_tier_counts": dict(tier_counts),
            "family_counts": _marginal_counts(entries)[0],
            "pivot_counts": _marginal_counts(entries)[1],
            "joint_cell_counts": _cell_counts(entries),
            "entries": entries,
        }

    schedule = {
        "schema_version": 1,
        "run_id": config["schedule_run_id"],
        "status": "PASS",
        "seed": seed,
        "dataset_path": str(dataset_path),
        "dataset_sha256": config["dataset_sha256"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": config["checkpoint_sha256"],
        "audit_manifest_path": str(audit_path),
        "audit_manifest_sha256": sha256_file(audit_path),
        "excluded_audit_questions": len(excluded),
        "eligible_unique_questions": len(candidates),
        "minimum_candidates_per_cell_observed": minimum_cell,
        "candidate_cell_counts": {
            f"{family}@{pivot}": len(by_cell[(family, pivot)])
            for family, pivot in CELL_CYCLE
        },
        "rejection_counts": dict(rejection_counts),
        "causal_families": list(CAUSAL_FAMILIES),
        "pivot_positions": [0, 1, 2],
        "affected_supervised_steps_per_corrupted_example": 3,
        "supervised_steps_per_example": 5,
        "coverage_candidates_percent": list(COVERAGES),
        "dev_dataset_path": str(dev_dataset_path),
        "dev_dataset_sha256": dev_sha,
        "splits": {
            "pilot": split_record(pilot_entries),
            "dev": split_record(dev_entries),
            "formal": split_record(formal_entries),
        },
    }
    schedule["schedule_sha256"] = _canonical_hash(schedule)
    schedule_path = (root / config["schedule_path"]).resolve()
    atomic_json(schedule_path, schedule)
    audit_record = dict(schedule)
    audit_record["splits"] = {
        name: {key: value for key, value in details.items() if key != "entries"}
        for name, details in schedule["splits"].items()
    }
    atomic_json((root / config["schedule_audit_path"]).resolve(), audit_record)

    examples_by_family: dict[str, dict[str, Any]] = {}
    for entry in pilot_entries:
        family = entry["chain"]["family"]
        if family not in examples_by_family:
            source = next(
                candidate[2] for candidate in candidates if candidate[1] == entry["question_id"]
            )
            examples_by_family[family] = {
                "source_idx": entry["source_idx"],
                "question": source.question,
                "answer": source.answer,
                "clean_steps": list(source.steps),
                "corrupted_first_five": entry["chain"]["corrupted_steps"],
                "labels": entry["chain"]["labels"],
                "affected_positions": entry["chain"]["affected_positions"],
                "propagated_final_value": entry["chain"]["propagated_final_value"],
                "chain_sha256": entry["chain"]["chain_sha256"],
            }
    atomic_json((root / config["readable_examples_path"]).resolve(), examples_by_family)
    return schedule


def resolve_split_examples(
    schedule: dict[str, Any], *, split: str, dataset_path: str | Path
) -> list[OfficialExample]:
    verify_causal_schedule(schedule)
    entries = _split_entries(schedule, split)
    expected = [int(entry["source_idx"]) for entry in entries]
    expected_set = set(expected)
    found = {
        example.idx: example
        for example in iter_icot_examples(dataset_path)
        if example.idx in expected_set
    }
    if len(found) != len(expected):
        raise ValueError(f"Scheduled {split} examples are missing or duplicated")
    resolved: list[OfficialExample] = []
    for entry in entries:
        example = found[entry["source_idx"]]
        if _question_id(example) != entry["question_id"]:
            raise ValueError(f"Question ID changed for source row {example.idx}")
        if _question_exact_hash(example) != entry["question_sha256"]:
            raise ValueError(f"Question bytes changed for source row {example.idx}")
        if _steps_hash(example) != entry["clean_steps_sha256"]:
            raise ValueError(f"Clean steps changed for source row {example.idx}")
        resolved.append(example)
    return resolved


def coverage_tiers(coverage: int) -> int:
    if coverage not in COVERAGES:
        raise ValueError(f"Coverage must be one of {COVERAGES}")
    return coverage // 25


def causal_steps_and_weights(
    arm: str,
    example: OfficialExample,
    entry: dict[str, Any],
    *,
    coverage: int,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    if arm not in CAUSAL_ARMS:
        raise ValueError(f"Unknown causal arm: {arm}")
    active = int(entry["coverage_tier"]) < coverage_tiers(coverage)
    clean_steps = tuple(example.steps[:5])
    if arm == "clean" or not active:
        return clean_steps, (1.0,) * 5

    chain = entry["chain"]
    steps = tuple(chain["corrupted_steps"])
    labels = tuple(chain["labels"])
    if len(steps) != 5 or len(labels) != 5:
        raise ValueError("Frozen causal chain must contain five steps and labels")
    if arm == "noisy_equal":
        weights = [1.0] * 5
    elif arm == "uniform_attenuation":
        weights = [0.46] * 5
    elif arm == "pivot_only":
        weights = [0.1 if label == "DIRECT_ERROR" else 1.0 for label in labels]
    else:
        weights = [0.1 if label != "CLEAN" else 1.0 for label in labels]
        if arm == "causal_normalized":
            scale = 5.0 / sum(weights)
            weights = [value * scale for value in weights]
    return steps, tuple(weights)


def _autocast(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type != "cuda":
        raise RuntimeError("Mixed precision requires CUDA")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def run_training_arm(
    config: dict[str, Any],
    *,
    split: str,
    arm: str,
    coverage: int,
    project_root: str | Path,
    updates_override: int | None = None,
    output_suffix: str = "",
    save_checkpoint: bool = True,
) -> dict[str, Any]:
    if split not in {"pilot", "formal"}:
        raise ValueError("Training split must be pilot or formal")
    allowed_arms = CAUSAL_ARMS
    if arm not in allowed_arms:
        raise ValueError(f"Arm {arm} is not allowed for {split}")
    if arm == "clean" and coverage not in COVERAGES:
        raise ValueError("Use a declared coverage even though clean ignores it")
    coverage_tiers(coverage)
    root = Path(project_root).resolve()
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    verify_causal_schedule(schedule)
    entries = _split_entries(schedule, split)
    default_updates = config[f"{split}_updates"]
    updates = updates_override if updates_override is not None else default_updates
    accumulation = int(config["gradient_accumulation_steps"])
    if updates <= 0 or updates * accumulation > len(entries):
        raise ValueError("Requested updates fall outside the frozen split")
    if split == "formal":
        gate_path = (root / config["calibration_gate_path"]).resolve()
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        expected_gate_hash = gate.get("gate_sha256")
        unhashed_gate = dict(gate)
        unhashed_gate.pop("gate_sha256", None)
        if expected_gate_hash != _canonical_hash(unhashed_gate):
            raise ValueError("Calibration gate SHA-256 mismatch")
        if not gate.get("gate_passed") or gate.get("chosen_coverage") != coverage:
            raise ValueError("Formal training is locked until the calibration gate passes")
        if gate.get("schedule_sha256") != schedule["schedule_sha256"]:
            raise ValueError("Calibration gate does not match the frozen schedule")

    device = torch.device(config["device"])
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Causal weighting training requires CUDA")
    if config["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported by this GPU")
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    examples = resolve_split_examples(
        schedule,
        split=split,
        dataset_path=(root / config["dataset_path"]).resolve(),
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
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    directory = _arm_dir_name(split, arm, coverage) + output_suffix
    output_dir = _output_root(root, config) / split / directory
    work_dir = _work_root(root, config) / split / directory
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
            position = update * accumulation + micro
            micro_seed = int(config["seed"]) + position
            torch.manual_seed(micro_seed)
            torch.cuda.manual_seed_all(micro_seed)
            example = examples[position]
            entry = entries[position]
            target_steps, weights = causal_steps_and_weights(
                arm, example, entry, coverage=coverage
            )
            encoded = encode_smoke_example(
                example,
                tokenizer,
                token_ids,
                latent_stage=config["latent_stage"],
                c_thought=config["c_thought"],
            )
            batch = tensorize_smoke_example(encoded, device=device)
            targets = tokenize_step_targets(tokenizer, target_steps)
            with _autocast(device, config["precision"]):
                losses = grouped_auxiliary_loss(
                    model,
                    batch,
                    targets,
                    weights,
                    latent_id=token_ids["<|latent|>"],
                    c_thought=config["c_thought"],
                )
                scaled = losses["loss"] / accumulation
            values = (
                float(losses["loss"].detach().float().item()),
                float(losses["answer_loss"].detach().float().item()),
                float(losses["auxiliary_loss"].detach().float().item()),
            )
            if not all(math.isfinite(value) for value in values):
                raise FloatingPointError(f"Non-finite loss in {directory} at update {update + 1}")
            scaled.backward()
            micro_total.append(values[0])
            micro_answer.append(values[1])
            micro_aux.append(values[2])
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config["max_grad_norm"]
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm in {directory}")
        optimizer.step()
        update_losses.append(sum(micro_total) / accumulation)
        answer_losses.append(sum(micro_answer) / accumulation)
        auxiliary_losses.append(sum(micro_aux) / accumulation)
        completed = update + 1
        if completed == 1 or completed % config["log_every"] == 0 or completed == updates:
            elapsed = time.perf_counter() - started
            progress = {
                "split": split,
                "arm": arm,
                "coverage": coverage,
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
                f"{split}/{directory} {completed}/{updates}: "
                f"total={update_losses[-1]:.5f}, answer={answer_losses[-1]:.5f}, "
                f"aux={auxiliary_losses[-1]:.5f}, peak={progress['peak_reserved_gb']:.2f} GB",
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
    active_examples = sum(
        int(entry["coverage_tier"]) < coverage_tiers(coverage)
        for entry in entries[: updates * accumulation]
    )
    result = {
        "run_id": (
            config["sanity_run_id"]
            if output_suffix.startswith("_sanity_")
            else _run_id(config, split, arm, coverage)
        ),
        "split": split,
        "arm": arm,
        "coverage": coverage,
        "status": "PASS",
        "seed": config["seed"],
        "updates": updates,
        "gradient_accumulation_steps": accumulation,
        "effective_micro_batches": updates * accumulation,
        "corrupted_examples": 0 if arm == "clean" else active_examples,
        "effective_step_contamination_rate": (
            0.0 if arm == "clean" else active_examples * 3 / (updates * accumulation * 5)
        ),
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


def run_loss_parity(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    schedule = json.loads((root / config["schedule_path"]).read_text(encoding="utf-8"))
    examples = resolve_split_examples(
        schedule,
        split="pilot",
        dataset_path=(root / config["dataset_path"]).resolve(),
    )
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
    example = examples[0]
    encoded = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=config["latent_stage"],
        c_thought=config["c_thought"],
    )
    batch = tensorize_smoke_example(encoded, device=device)
    targets = tokenize_step_targets(tokenizer, example.steps[:5])
    with torch.inference_mode():
        official = model(**{key: value.clone() for key, value in batch.items()}).loss
        custom = grouped_auxiliary_loss(
            model,
            batch,
            targets,
            (1.0,) * 5,
            latent_id=token_ids["<|latent|>"],
            c_thought=config["c_thought"],
        )["loss"]
    delta = abs(float(official.float().item()) - float(custom.float().item()))
    result = {
        "run_id": config["sanity_run_id"],
        "phase": "loss_parity",
        "example_source_idx": example.idx,
        "official_loss": float(official.float().item()),
        "custom_all_one_loss": float(custom.float().item()),
        "absolute_delta": delta,
        "tolerance": config["parity_tolerance"],
        "gate_passed": delta <= config["parity_tolerance"],
    }
    atomic_json((root / config["parity_path"]).resolve(), result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_sanity_gate(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    parity = run_loss_parity(config, project_root=root)
    arms: dict[str, Any] = {}
    if parity["gate_passed"]:
        for arm in CAUSAL_ARMS:
            arms[arm] = run_training_arm(
                config,
                split="pilot",
                arm=arm,
                coverage=75,
                project_root=root,
                updates_override=1,
                output_suffix=f"_sanity_{arm}",
                save_checkpoint=False,
            )
    result = {
        "run_id": config["sanity_run_id"],
        "status": "PASS",
        "parity": parity,
        "arms": arms,
    }
    result["gate_passed"] = parity["gate_passed"] and len(arms) == len(CAUSAL_ARMS) and all(
        item["gate_passed"] for item in arms.values()
    )
    result["status"] = "PASS" if result["gate_passed"] else "FAIL"
    atomic_json((root / config["sanity_path"]).resolve(), result)
    return result
