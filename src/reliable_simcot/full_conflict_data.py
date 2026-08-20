from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import random

from .causal_corruptions import causal_chains_by_cell
from .causal_experiment import _question_exact_hash, _question_id, _steps_hash
from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, iter_icot_examples
from .single_gpu_smoke import encode_smoke_example


def canonical_hash(payload: Any) -> str:
    return sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _walk_question_ids(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "question_id" and isinstance(child, str) and len(child) == 64:
                yield child
            elif key == "selected_question_ids" and isinstance(child, list):
                yield from (
                    item for item in child if isinstance(item, str) and len(item) == 64
                )
            else:
                yield from _walk_question_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_question_ids(child)


def collect_prior_question_ids(
    root: Path, globs: Iterable[str], *, current_work_root: Path
) -> tuple[set[str], list[dict[str, Any]]]:
    excluded: set[str] = set()
    records: list[dict[str, Any]] = []
    current = current_work_root.resolve()
    seen_paths: set[Path] = set()
    for pattern in globs:
        for path in root.glob(pattern):
            resolved = path.resolve()
            if resolved in seen_paths or current in resolved.parents or not path.is_file():
                continue
            seen_paths.add(resolved)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            ids = set(_walk_question_ids(payload))
            if ids:
                excluded.update(ids)
                records.append(
                    {
                        "path": str(path.relative_to(root)),
                        "sha256": sha256_file(path),
                        "question_ids": len(ids),
                    }
                )
    records.sort(key=lambda row: row["path"])
    return excluded, records


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
        raise ValueError("Could not recover every frozen source line")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for idx in source_indices:
            line = lines[idx]
            handle.write(line if line.endswith(("\n", "\r")) else line + "\n")
    temporary.replace(destination)


def _fits_clean_context(
    example: OfficialExample, tokenizer, token_ids: dict[str, int], config: dict[str, Any]
) -> bool:
    encoded = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=int(config["latent_stage"]),
        c_thought=int(config["c_thought"]),
    )
    maximum = int(config["max_sequence_tokens"])
    return len(encoded.input_ids) <= maximum and encoded.maximum_auxiliary_length <= maximum


def _choose_local_chain(example: OfficialExample, question_id: str) -> dict[str, Any] | None:
    chains = causal_chains_by_cell(example)
    if not chains:
        return None
    ordered = sorted(chains.items(), key=lambda item: (item[0][0], item[0][1]))
    position = int(sha256((question_id + ":local").encode("utf-8")).hexdigest(), 16)
    return ordered[position % len(ordered)][1].to_record()


def _entry(example: OfficialExample, local_chain: dict[str, Any], *, position: int) -> dict[str, Any]:
    return {
        "position": position,
        "source_idx": example.idx,
        "question_id": _question_id(example),
        "question_sha256": _question_exact_hash(example),
        "clean_steps_sha256": _steps_hash(example),
        "question": example.question,
        "answer": example.answer,
        "clean_steps": list(example.steps),
        "local_chain": local_chain,
    }


def verify_split_manifest(manifest: dict[str, Any]) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Split manifest has no valid SHA-256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256")
    if canonical_hash(unhashed) != expected:
        raise ValueError("Split manifest SHA-256 mismatch")
    train_ids = {row["question_id"] for row in manifest["train_entries"]}
    confirm_ids = {row["question_id"] for row in manifest["confirm_entries"]}
    if len(train_ids) != len(manifest["train_entries"]):
        raise ValueError("Duplicate question in training split")
    if len(confirm_ids) != len(manifest["confirm_entries"]):
        raise ValueError("Duplicate question in confirm split")
    if train_ids & confirm_ids:
        raise ValueError("Training/confirm leakage")
    primary = set(manifest["primary_generation_question_ids"])
    reserve = set(manifest["reserve_generation_question_ids"])
    if primary & reserve or not (primary | reserve) <= train_ids:
        raise ValueError("Invalid primary/reserve allocation")


def prepare_full_conflict_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_path = (root / config["dataset_path"]).resolve()
    checkpoint_path = (root / config["checkpoint_path"]).resolve()
    if sha256_file(source_path) != config["dataset_sha256"]:
        raise ValueError("Training dataset SHA-256 mismatch")
    if sha256_file(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("Starting checkpoint SHA-256 mismatch")

    work_root = (root / config["work_root"]).resolve()
    excluded, prior_manifests = collect_prior_question_ids(
        root,
        config["prior_manifest_globs"],
        current_work_root=work_root,
    )
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    seed = int(config["seed"])
    rejection_counts: Counter[str] = Counter()
    candidates: list[tuple[str, OfficialExample, dict[str, Any]]] = []
    seen: set[str] = set()
    exact_five = 0
    for example in iter_icot_examples(source_path):
        if len(example.steps) != 5:
            continue
        exact_five += 1
        question_id = _question_id(example)
        if question_id in seen:
            rejection_counts["duplicate_question"] += 1
            continue
        seen.add(question_id)
        if question_id in excluded:
            rejection_counts["previously_used"] += 1
            continue
        local_chain = _choose_local_chain(example, question_id)
        if local_chain is None:
            rejection_counts["no_local_causal_control"] += 1
            continue
        priority = sha256(f"{seed}:{question_id}".encode("utf-8")).hexdigest()
        candidates.append((priority, example, local_chain))
    candidates.sort(key=lambda row: (row[0], _question_id(row[1])))

    required = int(config["train_examples"]) + int(config["confirm_examples"])
    selected: list[tuple[str, OfficialExample, dict[str, Any]]] = []
    for item in candidates:
        if not _fits_clean_context(item[1], tokenizer, token_ids, config):
            rejection_counts["context_length"] += 1
            continue
        selected.append(item)
        if len(selected) == required:
            break
    if len(selected) != required:
        raise ValueError(f"Only {len(selected)} eligible examples; need {required}")

    train_count = int(config["train_examples"])
    train_entries = [
        _entry(item[1], item[2], position=position)
        for position, item in enumerate(selected[:train_count])
    ]
    confirm_entries = [
        _entry(item[1], item[2], position=position)
        for position, item in enumerate(selected[train_count:])
    ]
    primary_count = int(config["primary_generation_examples"])
    reserve_count = int(config["reserve_generation_examples"])
    primary_ids = [row["question_id"] for row in train_entries[:primary_count]]
    reserve_ids = [
        row["question_id"]
        for row in train_entries[primary_count : primary_count + reserve_count]
    ]
    manifest = {
        "schema_version": 1,
        "run_id": "FC003",
        "status": "PASS",
        "selection_seed": seed,
        "selection_rule": "ascending sha256(seed:question_id) after fixed eligibility filters",
        "dataset_path": str(source_path),
        "dataset_sha256": config["dataset_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "prior_manifest_digest": canonical_hash(prior_manifests),
        "prior_excluded_question_ids": len(excluded),
        "train_examples": len(train_entries),
        "confirm_examples": len(confirm_entries),
        "train_entries": train_entries,
        "confirm_entries": confirm_entries,
        "primary_generation_question_ids": primary_ids,
        "reserve_generation_question_ids": reserve_ids,
        "tier_0_question_ids": primary_ids[: primary_count // 2],
        "tier_1_question_ids": primary_ids[primary_count // 2 :],
        "official_test_opened": False,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    split_path = (root / config["split_manifest_path"]).resolve()
    atomic_json(split_path, manifest)
    verify_split_manifest(manifest)

    confirm_path = (root / config["confirm_dataset_path"]).resolve()
    _write_subset_dataset(
        source_path, confirm_path, [row["source_idx"] for row in confirm_entries]
    )
    eligibility = {
        "schema_version": 1,
        "run_ids": ["FC001", "FC002", "FC003"],
        "status": "PASS",
        "dataset_sha256": config["dataset_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "raw_examples": sum(1 for _ in iter_icot_examples(source_path)),
        "exact_five_examples": exact_five,
        "eligible_before_context": len(candidates),
        "selected_examples": required,
        "rejection_counts": dict(rejection_counts),
        "prior_manifests": prior_manifests,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": sha256_file(split_path),
        "confirm_dataset_path": str(confirm_path),
        "confirm_dataset_sha256": sha256_file(confirm_path),
        "zero_train_confirm_overlap": True,
        "zero_prior_overlap": not bool(
            excluded
            & {row["question_id"] for row in train_entries + confirm_entries}
        ),
        "official_test_opened": False,
    }
    eligibility["manifest_sha256"] = canonical_hash(eligibility)
    atomic_json((root / config["eligibility_manifest_path"]).resolve(), eligibility)
    atomic_json((root / config["provenance_path"]).resolve(), eligibility)
    return eligibility


def load_split_manifest(config: dict[str, Any], root: Path) -> dict[str, Any]:
    path = (root / config["split_manifest_path"]).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    verify_split_manifest(manifest)
    return manifest


def verify_frozen_schedule(schedule: dict[str, Any]) -> None:
    expected = schedule.get("schedule_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Frozen schedule has no valid SHA-256")
    unhashed = dict(schedule)
    unhashed.pop("schedule_sha256")
    if canonical_hash(unhashed) != expected:
        raise ValueError("Frozen schedule SHA-256 mismatch")
    entries = schedule["train_entries"]
    if len(entries) != 512:
        raise ValueError("Frozen schedule must contain 512 training examples")
    tier_counts = Counter(row["coverage_tier"] for row in entries)
    if tier_counts[0] != 128 or tier_counts[1] != 128:
        raise ValueError("Frozen schedule must contain 128 examples in each treatment tier")
    if len({row["question_id"] for row in entries}) != len(entries):
        raise ValueError("Duplicate training question in frozen schedule")


def freeze_full_conflict_schedule(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    split = load_split_manifest(config, root)
    accepted_path = (root / config["accepted_chains_path"]).resolve()
    accepted = [
        json.loads(line)
        for line in accepted_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(accepted) != int(config["primary_generation_examples"]):
        raise ValueError("Exactly 256 accepted full-conflict chains are required")
    primary_position = {
        question_id: position
        for position, question_id in enumerate(split["primary_generation_question_ids"])
    }
    active: dict[str, dict[str, Any]] = {}
    original_slots: set[str] = set()
    for chain in accepted:
        slot_id = chain.get("replaces_question_id") or chain["question_id"]
        if slot_id not in primary_position or slot_id in original_slots:
            raise ValueError("Accepted chain does not map uniquely to a frozen primary slot")
        original_slots.add(slot_id)
        actual_id = chain["question_id"]
        if actual_id in active:
            raise ValueError("A treatment question was allocated more than once")
        slot_position = primary_position[slot_id]
        active[actual_id] = {
            "coverage_tier": 0 if slot_position < 128 else 1,
            "original_slot_question_id": slot_id,
            "primary_slot_position": slot_position,
            "full_chain": chain,
        }
    if len(active) != 256 or len(original_slots) != 256:
        raise ValueError("Treatment allocation is incomplete")

    train_entries: list[dict[str, Any]] = []
    for source in split["train_entries"]:
        treatment = active.get(source["question_id"])
        row = {
            "position": source["position"],
            "source_idx": source["source_idx"],
            "question_id": source["question_id"],
            "question_sha256": source["question_sha256"],
            "clean_steps_sha256": source["clean_steps_sha256"],
            "question": source["question"],
            "answer": source["answer"],
            "clean_steps": source["clean_steps"],
            "local_chain": source["local_chain"],
            "coverage_tier": treatment["coverage_tier"] if treatment else None,
            "original_slot_question_id": (
                treatment["original_slot_question_id"] if treatment else None
            ),
            "primary_slot_position": (
                treatment["primary_slot_position"] if treatment else None
            ),
            "full_chain": treatment["full_chain"] if treatment else None,
        }
        train_entries.append(row)
    order_rng = random.Random(int(config["seed"]) + 404)
    order_rng.shuffle(train_entries)
    for position, row in enumerate(train_entries):
        row["position"] = position
    tier_family_counts = {
        str(tier): dict(
            Counter(
                row["full_chain"]["error_family"]
                for row in train_entries
                if row["coverage_tier"] == tier
            )
        )
        for tier in (0, 1)
    }
    schedule = {
        "schema_version": 1,
        "run_id": "FC021",
        "status": "PASS",
        "dataset_path": str((root / config["dataset_path"]).resolve()),
        "dataset_sha256": config["dataset_sha256"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "split_manifest_sha256": split["manifest_sha256"],
        "accepted_chains_sha256": sha256_file(accepted_path),
        "train_examples": len(train_entries),
        "confirm_examples": len(split["confirm_entries"]),
        "tier_family_counts": tier_family_counts,
        "treatment_question_ids_25": sorted(
            row["question_id"] for row in train_entries if row["coverage_tier"] == 0
        ),
        "treatment_question_ids_50": sorted(
            row["question_id"]
            for row in train_entries
            if row["coverage_tier"] in {0, 1}
        ),
        "train_entries": train_entries,
        "official_test_opened": False,
    }
    schedule["schedule_sha256"] = canonical_hash(schedule)
    verify_frozen_schedule(schedule)
    path = (root / config["frozen_schedule_path"]).resolve()
    atomic_json(path, schedule)
    return schedule


def load_frozen_schedule(config: dict[str, Any], root: Path) -> dict[str, Any]:
    path = (root / config["frozen_schedule_path"]).resolve()
    schedule = json.loads(path.read_text(encoding="utf-8"))
    verify_frozen_schedule(schedule)
    if sha256_file((root / config["accepted_chains_path"]).resolve()) != schedule[
        "accepted_chains_sha256"
    ]:
        raise ValueError("Accepted-chain data changed after schedule freeze")
    return schedule
