from __future__ import annotations

from collections import Counter
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import random

from .audit import evaluate_arithmetic
from .causal_corruptions import _format_fraction, _question_values
from .full_conflict_data import canonical_hash, load_split_manifest
from .full_conflict_validation import validate_full_conflict_candidate
from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    target = Path(path)
    if not target.exists():
        return rows
    with target.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Malformed JSONL at {target}:{line_number}") from error
    return rows


def build_prompt_manifest(
    config: dict[str, Any], *, project_root: str | Path
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    split = load_split_manifest(config, root)
    by_id = {row["question_id"]: row for row in split["train_entries"]}
    primary = split["primary_generation_question_ids"]
    reserve = split["reserve_generation_question_ids"]
    families = tuple(config["error_families"])
    if len(primary) % len(families):
        raise ValueError("Primary generation quota is not divisible by error families")
    rows: list[dict[str, Any]] = []
    for pool, question_ids in (("primary", primary), ("reserve", reserve)):
        for position, question_id in enumerate(question_ids):
            source = by_id[question_id]
            family = families[position % len(families)]
            row = {
                "schema_version": 1,
                "prompt_version": config["prompt_version"],
                "pool": pool,
                "pool_position": position,
                "question_id": question_id,
                "source_idx": source["source_idx"],
                "error_family": family,
                "question": source["question"],
                "official_answer": source["answer"],
                "clean_steps": source["clean_steps"],
                "constraints": {
                    "steps": 5,
                    "locally_valid_arithmetic": True,
                    "all_steps_and_results_differ": True,
                    "all_steps_feed_final": True,
                    "final_must_differ_from_official_answer": True,
                    "min_normalized_token_edit_distance": config[
                        "min_normalized_edit_distance"
                    ],
                    "auxiliary_token_ratio": [
                        config["min_aux_token_ratio"],
                        config["max_aux_token_ratio"],
                    ],
                    "no_treatment_identity_leakage": True,
                },
                "required_output_keys": [
                    "question_id",
                    "error_family",
                    "error_rationale",
                    "steps",
                    "wrong_final_result",
                ],
            }
            row["prompt_sha256"] = canonical_hash(row)
            rows.append(row)
    destination = (root / config["prompt_manifest_path"]).resolve()
    _atomic_jsonl(destination, rows)
    return rows


def append_raw_generation(
    path: str | Path,
    record: dict[str, Any],
    *,
    max_attempts: int,
) -> None:
    target = Path(path)
    prior = read_jsonl(target)
    key = (record.get("question_id"), int(record.get("attempt", 0)))
    if key[1] < 1 or key[1] > max_attempts:
        raise ValueError("Attempt is outside the frozen 1..max_attempts range")
    if any((row.get("question_id"), int(row.get("attempt", 0))) == key for row in prior):
        raise FileExistsError(f"Generation attempt already recorded: {key}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


_RATIONALES = {
    "quantity_unit_misread": (
        "Treat the stated quantities as compatible units in one running conversion "
        "chain, so counts, rates, and amounts are repeatedly combined."
    ),
    "relation_plan_misread": (
        "Apply the stated numerical relations in a different order and propagate the "
        "result as one continuous plan."
    ),
    "wrong_target": (
        "Compute a cumulative quantity score from the stated numbers instead of the "
        "quantity requested by the question."
    ),
    "compound_misconception": (
        "Combine several mutually consistent misunderstandings of the quantities and "
        "relations into one propagated calculation."
    ),
}


def _render(value: Fraction) -> str:
    return _format_fraction(value)


def _expression_candidates(
    prior: str | None,
    values: list[str],
    *,
    family: str,
    rng: random.Random,
) -> list[str]:
    shuffled = list(values)
    rng.shuffle(shuffled)
    while len(shuffled) < 4:
        shuffled.extend(values)
    a, b, c, d = shuffled[:4]
    if prior is None:
        patterns = [
            f"{a}+{b}",
            f"{a}-{b}",
            f"{a}*{b}",
            f"{a}+{b}+{c}",
            f"{a}+{b}-{c}",
            f"{a}*{b}+{c}",
            f"{a}*{b}-{c}",
            f"{a}+{b}*{c}",
            f"{a}+{b}+{c}-{d}",
        ]
    else:
        common = [
            f"{prior}+{a}",
            f"{prior}-{a}",
            f"{prior}*{a}",
            f"{prior}+{a}+{b}",
            f"{prior}+{a}-{b}",
            f"{prior}-{a}+{b}",
            f"{prior}*{a}+{b}",
            f"{prior}*{a}-{b}",
            f"{prior}+{a}*{b}",
            f"{prior}-{a}*{b}",
            f"{prior}+{a}+{b}-{c}",
        ]
        if family == "quantity_unit_misread":
            patterns = common[:2] + common[3:5] + common[8:]
        elif family == "relation_plan_misread":
            patterns = common[1:3] + common[5:8] + common[9:]
        elif family == "wrong_target":
            patterns = common[:1] + common[3:5] + common[8:]
        else:
            patterns = common
    rng.shuffle(patterns)
    return patterns


def _synthesize_candidate(
    prompt: dict[str, Any],
    source: dict[str, Any],
    *,
    attempt: int,
    tokenizer,
    token_ids: dict[str, int],
    config: dict[str, Any],
    max_trials: int = 6000,
) -> tuple[dict[str, Any], int]:
    clean = OfficialExample(
        idx=source["source_idx"],
        question=source["question"],
        steps=tuple(source["clean_steps"]),
        answer=source["answer"],
    )
    numeric_values = sorted(
        (value for value in _question_values(clean.question) if value != 0),
        key=lambda value: (abs(value), value),
    )
    if not numeric_values:
        numeric_values = [Fraction(1)]
    value_text = [_render(value) for value in numeric_values]
    seed_text = f"{config['prompt_version']}:{prompt['question_id']}:{attempt}"
    rng = random.Random(int(sha256(seed_text.encode("utf-8")).hexdigest(), 16))
    last: dict[str, Any] | None = None
    for trial in range(1, max_trials + 1):
        steps: list[str] = []
        results: set[Fraction] = set()
        prior: str | None = None
        complete = True
        for step_index in range(5):
            expressions = _expression_candidates(
                prior,
                value_text,
                family=prompt["error_family"],
                rng=rng,
            )
            chosen: tuple[str, Fraction] | None = None
            for expression in expressions:
                try:
                    result = evaluate_arithmetic(expression)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                    continue
                if (
                    abs(float(result)) > 10**8
                    or result in results
                    or result in numeric_values
                ):
                    continue
                clean_result_text = source["clean_steps"][step_index].split("=")[-1].rstrip(">")
                try:
                    clean_result = evaluate_arithmetic(clean_result_text)
                except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                    clean_result = None
                if clean_result is not None and result == clean_result:
                    continue
                rendered = _render(result)
                step = f"<<{expression}={rendered}>>"
                if step == source["clean_steps"][step_index]:
                    continue
                chosen = (step, result)
                break
            if chosen is None:
                complete = False
                break
            steps.append(chosen[0])
            results.add(chosen[1])
            prior = _render(chosen[1])
        if not complete:
            continue
        candidate = {
            "question_id": prompt["question_id"],
            "error_family": prompt["error_family"],
            "error_rationale": _RATIONALES[prompt["error_family"]],
            "steps": steps,
            "wrong_final_result": prior,
            "expected_question_id": prompt["question_id"],
        }
        last = candidate
        validation = validate_full_conflict_candidate(
            clean,
            candidate,
            tokenizer=tokenizer,
            token_ids=token_ids,
            config=config,
            check_context=False,
        )
        if validation.accepted:
            context_validation = validate_full_conflict_candidate(
                clean,
                candidate,
                tokenizer=tokenizer,
                token_ids=token_ids,
                config=config,
                check_context=True,
            )
            if context_validation.accepted:
                candidate.pop("expected_question_id")
                return candidate, trial
    if last is None:
        last = {
            "question_id": prompt["question_id"],
            "error_family": prompt["error_family"],
            "error_rationale": _RATIONALES[prompt["error_family"]],
            "steps": [],
            "wrong_final_result": None,
            "expected_question_id": prompt["question_id"],
        }
    last.pop("expected_question_id", None)
    return last, max_trials


def generate_codex_authored_records(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    include_reserve: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    split = load_split_manifest(config, root)
    source_by_id = {row["question_id"]: row for row in split["train_entries"]}
    prompts = read_jsonl((root / config["prompt_manifest_path"]).resolve())
    if not include_reserve:
        prompts = [row for row in prompts if row["pool"] == "primary"]
    raw_path = (root / config["raw_generations_path"]).resolve()
    existing = read_jsonl(raw_path)
    attempts_by_id: dict[str, set[int]] = {}
    for row in existing:
        attempts_by_id.setdefault(row["question_id"], set()).add(int(row["attempt"]))
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    generated = 0
    trial_counts: list[int] = []
    for position, prompt in enumerate(prompts, start=1):
        occupied = attempts_by_id.get(prompt["question_id"], set())
        source = source_by_id[prompt["question_id"]]
        clean = OfficialExample(
            idx=source["source_idx"],
            question=source["question"],
            steps=tuple(source["clean_steps"]),
            answer=source["answer"],
        )
        existing_for_prompt = sorted(
            (
                row
                for row in existing
                if row["question_id"] == prompt["question_id"]
            ),
            key=lambda row: int(row["attempt"]),
        )
        already_accepted = False
        for row in existing_for_prompt:
            candidate = dict(row["candidate"])
            candidate["expected_question_id"] = prompt["question_id"]
            if validate_full_conflict_candidate(
                clean,
                candidate,
                tokenizer=tokenizer,
                token_ids=token_ids,
                config=config,
            ).accepted:
                already_accepted = True
                break
        if already_accepted:
            continue
        for attempt in range(1, int(config["max_generation_attempts"]) + 1):
            if attempt in occupied:
                continue
            candidate, trials = _synthesize_candidate(
                prompt,
                source_by_id[prompt["question_id"]],
                attempt=attempt,
                tokenizer=tokenizer,
                token_ids=token_ids,
                config=config,
            )
            record = {
                "question_id": prompt["question_id"],
                "attempt": attempt,
                "prompt_version": config["prompt_version"],
                "generator": "current_codex_authored_constrained_synthesis_v1",
                "generator_disclosure": (
                    "Deterministic constrained synthesis authored in the current Codex task; "
                    "not a natural teacher-model sample."
                ),
                "search_trials": trials,
                "candidate": candidate,
            }
            append_raw_generation(
                raw_path,
                record,
                max_attempts=int(config["max_generation_attempts"]),
            )
            generated += 1
            trial_counts.append(trials)
            check_candidate = dict(candidate)
            check_candidate["expected_question_id"] = prompt["question_id"]
            if validate_full_conflict_candidate(
                clean,
                check_candidate,
                tokenizer=tokenizer,
                token_ids=token_ids,
                config=config,
            ).accepted:
                break
        if position == 1 or position % 32 == 0 or position == len(prompts):
            print(
                f"constrained generation {position}/{len(prompts)}; new_records={generated}",
                flush=True,
            )
    return {
        "prompts_considered": len(prompts),
        "new_records": generated,
        "max_search_trials": max(trial_counts, default=0),
        "mean_search_trials": (
            sum(trial_counts) / len(trial_counts) if trial_counts else 0.0
        ),
        "raw_generations_path": str(raw_path),
        "raw_generations_sha256": sha256_file(raw_path),
    }


def validate_generation_records(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    small_batch: bool = False,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    split = load_split_manifest(config, root)
    source_by_id = {row["question_id"]: row for row in split["train_entries"]}
    prompts = read_jsonl((root / config["prompt_manifest_path"]).resolve())
    prompt_by_id = {row["question_id"]: row for row in prompts}
    raw_path = (root / config["raw_generations_path"]).resolve()
    raw = read_jsonl(raw_path)
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    limit = int(config["small_batch_examples"]) if small_batch else None
    target_ids = [row["question_id"] for row in prompts if row["pool"] == "primary"]
    if limit is not None:
        per_family = limit // len(config["error_families"])
        selected: list[str] = []
        counts: Counter[str] = Counter()
        for row in prompts:
            if row["pool"] != "primary":
                continue
            family = row["error_family"]
            if counts[family] < per_family:
                selected.append(row["question_id"])
                counts[family] += 1
        target_ids = selected

    raw_by_id: dict[str, list[dict[str, Any]]] = {}
    for row in raw:
        raw_by_id.setdefault(str(row.get("question_id")), []).append(row)
    accepted: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    def validate_one(
        question_id: str, *, replaces_question_id: str | None = None
    ) -> dict[str, Any] | None:
        source = source_by_id[question_id]
        prompt = prompt_by_id[question_id]
        attempts = sorted(raw_by_id.get(question_id, ()), key=lambda row: row["attempt"])
        chosen: dict[str, Any] | None = None
        for row in attempts[: int(config["max_generation_attempts"])]:
            candidate = dict(row["candidate"])
            candidate["expected_question_id"] = question_id
            result = validate_full_conflict_candidate(
                OfficialExample(
                    idx=source["source_idx"],
                    question=source["question"],
                    steps=tuple(source["clean_steps"]),
                    answer=source["answer"],
                ),
                candidate,
                tokenizer=tokenizer,
                token_ids=token_ids,
                config=config,
            )
            detail = {
                "question_id": question_id,
                "attempt": row["attempt"],
                "error_family": prompt["error_family"],
                "replaces_question_id": replaces_question_id,
                "validation": result.to_record(),
            }
            details.append(detail)
            if result.accepted:
                chosen = {
                    "question_id": question_id,
                    "source_idx": source["source_idx"],
                    "pool": prompt["pool"],
                    "pool_position": prompt["pool_position"],
                    "replaces_question_id": replaces_question_id,
                    "error_family": prompt["error_family"],
                    "attempt": row["attempt"],
                    "error_rationale": candidate["error_rationale"],
                    "steps": candidate["steps"],
                    "wrong_final_result": candidate["wrong_final_result"],
                    "validation": result.to_record(),
                }
                chosen["accepted_sha256"] = canonical_hash(chosen)
                break
            rejection_counts.update(result.rejection_codes)
        return chosen

    failed_primary: list[str] = []
    for question_id in target_ids:
        chosen = validate_one(question_id)
        if chosen is None:
            failed_primary.append(question_id)
        else:
            accepted.append(chosen)

    used_reserves: set[str] = set()
    for failed_id in failed_primary:
        family = prompt_by_id[failed_id]["error_family"]
        replacement: dict[str, Any] | None = None
        for prompt in prompts:
            reserve_id = prompt["question_id"]
            if (
                prompt["pool"] != "reserve"
                or prompt["error_family"] != family
                or reserve_id in used_reserves
            ):
                continue
            used_reserves.add(reserve_id)
            replacement = validate_one(
                reserve_id, replaces_question_id=failed_id
            )
            if replacement is not None:
                break
        if replacement is not None:
            accepted.append(replacement)

    gate = {
        "schema_version": 1,
        "run_id": "FC011" if small_batch else "FC021",
        "status": "PASS" if len(accepted) == len(target_ids) else "FAIL",
        "target_examples": len(target_ids),
        "accepted_examples": len(accepted),
        "missing_examples": len(target_ids) - len(accepted),
        "family_counts": dict(Counter(row["error_family"] for row in accepted)),
        "failed_primary_question_ids": failed_primary,
        "reserve_replacements": [
            {
                "question_id": row["question_id"],
                "replaces_question_id": row["replaces_question_id"],
            }
            for row in accepted
            if row["replaces_question_id"] is not None
        ],
        "rejection_counts": dict(rejection_counts),
        "prompt_manifest_sha256": sha256_file(
            (root / config["prompt_manifest_path"]).resolve()
        ),
        "raw_generations_sha256": sha256_file(raw_path) if raw_path.exists() else None,
        "readability_review": "PENDING_CODEX_SECOND_PASS" if small_batch else None,
        "details": details,
    }
    destination = (
        (root / config["small_batch_gate_path"]).resolve()
        if small_batch
        else (root / config["data_audit_path"]).resolve()
    )
    atomic_json(destination, gate)
    if not small_batch and gate["status"] == "PASS":
        _atomic_jsonl((root / config["accepted_chains_path"]).resolve(), accepted)
    return gate


def record_small_batch_readability_review(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    passed: bool,
    notes: list[str],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    path = (root / config["small_batch_gate_path"]).resolve()
    gate = json.loads(path.read_text(encoding="utf-8"))
    if gate.get("accepted_examples") != int(config["small_batch_examples"]):
        raise ValueError("Readability review is locked until all small-batch rows validate")
    gate["readability_review"] = "PASS" if passed else "FAIL"
    gate["readability_reviewer"] = "current_codex_task_second_pass"
    gate["readability_independent"] = False
    gate["readability_notes"] = list(notes)
    gate["status"] = "PASS" if passed else "FAIL"
    gate["gate_sha256"] = canonical_hash(
        {key: value for key, value in gate.items() if key != "gate_sha256"}
    )
    atomic_json(path, gate)
    return gate
