from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
import json
import re
import subprocess

from .full_conflict_generation import read_jsonl
from .full_conflict_validation import normalized_levenshtein
from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, iter_icot_examples
from .prm800k_data import (
    classify_ratings,
    consolidate_trajectories,
    load_official_grader,
    normalize_problem,
    problem_id,
    reconstruct_trajectory,
)
from .single_gpu_smoke import encode_smoke_example


GradeAnswer = Callable[[str, str], bool]

VARIANT_LABELS: dict[str, tuple[int, ...]] = {
    "clean": (1, 1, 1, 1, 1),
    "solution_n1": (1, -1, 1, 1, 1),
    "solution_n2": (1, -1, -1, 1, 1),
    "misread_n1": (1, -1, 1, 1, 1),
    "misread_n2": (1, -1, -1, 1, 1),
}


def canonical_hash(payload: Any) -> str:
    value = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value.pop("manifest_sha256", None)
        value.pop("five_tuple_sha256", None)
        value.pop("schedule_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _trajectory_token_count(tokenizer, steps: Iterable[str]) -> int:
    return sum(len(_token_ids(tokenizer, step + "\n")) for step in steps)


def _normalized_answer(text: str) -> str:
    value = text.strip().strip("$").rstrip(".")
    return re.sub(r"\s+", "", value)


def extract_final_answer(step: str) -> str | None:
    marker = "# Answer"
    if marker in step:
        answer = step.rsplit(marker, 1)[1].strip()
        return answer or None
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", step)
    return boxed[-1].strip() if boxed else None


def _numeric_values(problem: str) -> tuple[int, int]:
    values = [int(item) for item in re.findall(r"(?<![\w.])-?\d+", problem)]
    nonzero = [value for value in values if value]
    if len(nonzero) >= 2:
        return nonzero[0], nonzero[1]
    if len(nonzero) == 1:
        return nonzero[0], 2
    return 2, 3


def _reasoning_domain(problem: str) -> str:
    lowered = problem.lower()
    if any(
        marker in lowered
        for marker in (
            "triangle",
            "circle",
            "angle",
            "square",
            "rectangle",
            "perimeter",
            "area",
            "altitude",
            "parallel",
        )
    ):
        return "geometry"
    if any(
        marker in lowered
        for marker in (
            "probability",
            "random",
            "number of ways",
            "arrange",
            "permutation",
            "combination",
        )
    ):
        return "counting"
    if any(
        marker in lowered
        for marker in (
            "divisor",
            "divisible",
            "remainder",
            "congruent",
            "integer",
            "prime",
            "mod ",
        )
    ):
        return "number theory"
    if any(
        marker in lowered
        for marker in (
            "sin ",
            "cos ",
            "tan ",
            "sec ",
            "log",
            "sequence",
            "series",
        )
    ):
        return "precalculus"
    return "algebra"


def _wrong_text_options(
    *, family: str, branch_id: str, left: int, right: int, problem: str
) -> tuple[tuple[str, str], ...]:
    first = left + right + 1
    second = first * 2 + 1
    domain = _reasoning_domain(problem)
    solution_models = {
        "geometry": (
            "combine the stated geometric measures into one linear total",
            "apply the same additive geometry rule a second time",
        ),
        "counting": (
            "add the first outcome counts directly",
            "continue counting from that cumulative outcome total",
        ),
        "number theory": (
            "treat the divisibility or congruence conditions as an ordinary additive equality",
            "apply that equality to the remaining quantity",
        ),
        "precalculus": (
            "linearize the nonlinear relation with an additive rule",
            "apply the linearized rule through the next operation",
        ),
        "algebra": (
            "collapse the governing equations into one additive invariant",
            "substitute that invariant into the remaining calculation",
        ),
    }
    first_model, continuation = solution_models[domain]
    if family == "solution":
        return (
            (
                f"Path {branch_id}: We {first_model}, setting "
                f"Q={left}+{right}+1={first} as the requested quantity.",
                f"Path {branch_id}: We {continuation}, so "
                f"R=2Q+1=2({first})+1={second} becomes the final quantity.",
            ),
            (
                f"Path {branch_id}: For this {domain} problem, the governing rule is "
                f"Q={left}+{right}+1={first}.",
                f"Path {branch_id}: Substituting into this rule gives "
                f"R=2({first})+1={second} as the final quantity.",
            ),
        )
    if family == "misread":
        return (
            (
                f"Path {branch_id}: The requested {domain} target is the combined "
                f"total T={left}+{right}+1={first}.",
                f"Path {branch_id}: Continuing with that target gives the scaled total "
                f"U=2T+1=2({first})+1={second}.",
            ),
            (
                f"Path {branch_id}: What the question asks for is the cumulative "
                f"{domain} measure T={left}+{right}+1={first}.",
                f"Path {branch_id}: That objective gives "
                f"U=2({first})+1={second} as the reported quantity.",
            ),
        )
    raise ValueError(f"Unknown conflict family: {family}")


def _recovery_prefix(family: str) -> str:
    if family == "solution":
        return (
            "The previous branch used a rule unsupported by the stated constraints. "
            "Discarding that branch, the correct reasoning is: "
        )
    if family == "misread":
        return (
            "The previous branch changed the requested target. "
            "Discarding that branch, the correct reasoning is: "
        )
    raise ValueError(f"Unknown conflict family: {family}")


def _recovery_bodies(
    clean_steps: tuple[str, ...], positions: tuple[int, ...]
) -> tuple[tuple[str, str, int], ...]:
    source = [clean_steps[position].strip() for position in positions]
    full = " ".join(source)
    math_spans: list[str] = []
    for step in source:
        for span in re.findall(r"\$[^$]+\$", step):
            if span not in math_spans:
                math_spans.append(span)
    numeric_sentences: list[str] = []
    for step in source:
        sentences = [
            item.strip()
            for item in re.split(r"(?<=[.?])\s+|\n+", step)
            if item.strip()
        ]
        informative = [
            item
            for item in sentences
            if "=" in item or re.search(r"(?<![\w.])-?\d+", item)
        ]
        chosen = informative[-1] if informative else sentences[-1]
        if chosen not in numeric_sentences:
            numeric_sentences.append(chosen)
    candidates: list[tuple[str, str, int]] = [
        (full, "ordered_full_recovery", 0),
    ]
    if len(source) >= 2:
        candidates.append(
            (" ".join(source[-2:]), "ordered_two_step_recovery", 1)
        )
    candidates.append((source[-1], "ordered_final_step_recovery", 2))
    if numeric_sentences:
        candidates.append(
            (" ".join(numeric_sentences), "ordered_informative_sentences", 3)
        )
    if math_spans:
        candidates.append(
            (
                "Using the original conditions, " + "; ".join(math_spans) + ".",
                "formula_evidence_fallback",
                4,
            )
        )
    if math_spans and numeric_sentences:
        candidates.append(
            (
                "Using the original conditions, "
                + "; ".join(math_spans)
                + ". "
                + numeric_sentences[-1],
                "formula_plus_conclusion_fallback",
                5,
            )
        )
    unique: list[tuple[str, str, int]] = []
    seen: set[str] = set()
    for candidate, mode, style_rank in candidates:
        compact = re.sub(r"\s+", " ", candidate).strip()
        if compact and compact not in seen:
            unique.append((compact, mode, style_rank))
            seen.add(compact)
    return tuple(unique)


def _candidate_layouts(
    clean_steps: tuple[str, ...],
    *,
    family: str,
    dose: int,
    branch_id: str,
    problem: str,
) -> Iterator[tuple[tuple[str, ...], int, str, int]]:
    left, right = _numeric_values(problem)
    recovery_prefix = _recovery_prefix(family)
    wrong_options = _wrong_text_options(
        family=family,
        branch_id=branch_id,
        left=left,
        right=right,
        problem=problem,
    )
    if dose == 1:
        for wrong_one, _ in wrong_options:
            recovery = (
                recovery_prefix
                + "The original target and governing relations are restored; "
                "the complete corrected derivation follows."
            )
            steps = (
                clean_steps[0],
                wrong_one,
                recovery,
                clean_steps[3],
                clean_steps[4],
            )
            yield steps, len(recovery), "explicit_bridge_then_full_recovery", 0
        return
    if dose == 2:
        for wrong_one, wrong_two in wrong_options:
            recovery = recovery_prefix + clean_steps[3]
            steps = (
                clean_steps[0],
                wrong_one,
                wrong_two,
                recovery,
                clean_steps[4],
            )
            yield steps, len(recovery), "explicit_full_recovery", 0
        return
    raise ValueError("Noise dose must be one or two")


def _select_variant(
    clean_steps: tuple[str, ...],
    *,
    family: str,
    dose: int,
    branch_id: str,
    problem: str,
    tokenizer,
    min_token_ratio: float,
    max_token_ratio: float,
    min_edit_distance: float,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    clean_count = _trajectory_token_count(tokenizer, clean_steps)
    valid: list[tuple[tuple[Any, ...], tuple[str, ...], dict[str, Any]]] = []
    closest: tuple[float, float] | None = None
    for steps, recovery_characters, recovery_mode, style_rank in _candidate_layouts(
        clean_steps,
        family=family,
        dose=dose,
        branch_id=branch_id,
        problem=problem,
    ):
        count = _trajectory_token_count(tokenizer, steps)
        ratio = count / clean_count
        error_positions = (1,) if dose == 1 else (1, 2)
        edit_distances = tuple(
            normalized_levenshtein(
                _token_ids(tokenizer, clean_steps[position]),
                _token_ids(tokenizer, steps[position]),
            )
            for position in error_positions
        )
        distance_to_range = max(min_token_ratio - ratio, 0.0, ratio - max_token_ratio)
        closest = min(closest, (distance_to_range, ratio)) if closest else (
            distance_to_range,
            ratio,
        )
        if not min_token_ratio <= ratio <= max_token_ratio:
            continue
        if any(value < min_edit_distance for value in edit_distances):
            continue
        metrics = {
            "token_count": count,
            "clean_token_count": clean_count,
            "token_ratio": ratio,
            "error_edit_distances": list(edit_distances),
            "recovery_mode": recovery_mode,
        }
        ranking = (
            style_rank,
            abs(ratio - 1.0),
            -recovery_characters,
            canonical_hash(steps),
        )
        valid.append((ranking, steps, metrics))
    if not valid:
        raise ValueError(
            f"No {family} Noise-{dose} construction satisfies token/edit constraints; "
            f"closest={closest}"
        )
    _, steps, metrics = min(valid, key=lambda item: item[0])
    return steps, metrics


def construct_five_tuple(
    *,
    question_id: str,
    problem: str,
    answer: str,
    clean_steps: Iterable[str],
    tokenizer,
    min_token_ratio: float,
    max_token_ratio: float,
    min_edit_distance: float,
    source: dict[str, Any],
    generated_answer: str | None = None,
) -> dict[str, Any]:
    source_clean = tuple(str(step).strip() for step in clean_steps)
    if len(source_clean) != 5 or any(not step for step in source_clean):
        raise ValueError("A non-empty five-step Clean trajectory is required")
    reasoning_start = (
        1
        if str(source.get("kind")) == "codex_segmented_official_math_solution"
        else 0
    )
    complete_reasoning = re.sub(
        r"\s+", " ", " ".join(source_clean[reasoning_start:4])
    ).strip()
    domain = _reasoning_domain(problem)
    clean = (
        "We keep the original question, requested target, and every stated condition unchanged.",
        f"For this {domain} problem, we identify the governing relations, variables, "
        "and dependencies before carrying out any calculation.",
        "We apply those relations in their stated order, preserve every dependency, "
        "and check the resulting value against the original target.",
        complete_reasoning,
        source_clean[4],
    )
    final_answer = generated_answer.strip() if generated_answer else answer.strip()
    clean_extracted = extract_final_answer(clean[-1])
    if clean_extracted is None or _normalized_answer(clean_extracted) != _normalized_answer(
        final_answer
    ):
        raise ValueError("Clean trajectory final step does not emit generated answer")
    clean_count = _trajectory_token_count(tokenizer, clean)
    variants: dict[str, Any] = {
        "clean": {
            "family": "clean",
            "dose": 0,
            "steps": list(clean),
            "labels": list(VARIANT_LABELS["clean"]),
            "final_answer": clean_extracted,
            "token_count": clean_count,
            "clean_token_count": clean_count,
            "token_ratio": 1.0,
            "error_edit_distances": [],
            "wrong_branch_id": None,
        }
    }
    for family in ("solution", "misread"):
        for dose in (1, 2):
            name = f"{family}_n{dose}"
            branch_id = sha256(
                f"{question_id}:{family}:{dose}".encode("utf-8")
            ).hexdigest()[:10]
            steps, metrics = _select_variant(
                clean,
                family=family,
                dose=dose,
                branch_id=branch_id,
                problem=problem,
                tokenizer=tokenizer,
                min_token_ratio=min_token_ratio,
                max_token_ratio=max_token_ratio,
                min_edit_distance=min_edit_distance,
            )
            variants[name] = {
                "family": family,
                "dose": dose,
                "steps": list(steps),
                "labels": list(VARIANT_LABELS[name]),
                "final_answer": extract_final_answer(steps[-1]),
                "wrong_branch_id": branch_id,
                **metrics,
            }
    row = {
        "schema_version": 1,
        "question_id": question_id,
        "problem": normalize_problem(problem),
        "answer": answer.strip(),
        "generated_answer": final_answer,
        "source": source,
        "variants": variants,
    }
    row["five_tuple_sha256"] = canonical_hash(row)
    return row


def validate_five_tuple(
    row: dict[str, Any],
    *,
    tokenizer,
    min_token_ratio: float,
    max_token_ratio: float,
    min_edit_distance: float,
    grade_answer: GradeAnswer | None = None,
) -> dict[str, Any]:
    rejection: list[str] = []
    variants = row.get("variants")
    if not isinstance(variants, dict) or set(variants) != set(VARIANT_LABELS):
        return {"accepted": False, "rejection_codes": ["variant_set"]}
    clean_steps = tuple(variants["clean"].get("steps", ()))
    if len(clean_steps) != 5:
        rejection.append("clean:step_count")
    metrics: dict[str, Any] = {}
    for name, expected_labels in VARIANT_LABELS.items():
        variant = variants[name]
        steps = tuple(variant.get("steps", ()))
        labels = tuple(variant.get("labels", ()))
        if len(steps) != 5 or any(not isinstance(step, str) or not step.strip() for step in steps):
            rejection.append(f"{name}:step_count")
            continue
        if labels != expected_labels:
            rejection.append(f"{name}:label_pattern")
        extracted = extract_final_answer(steps[-1])
        expected_final = row.get("generated_answer", row.get("answer", ""))
        if extracted is None or _normalized_answer(extracted) != _normalized_answer(
            str(expected_final)
        ):
            rejection.append(f"{name}:final_answer_mismatch")
        elif grade_answer is not None and not grade_answer(extracted, str(row["answer"])):
            rejection.append(f"{name}:grader_wrong_answer")
        count = _trajectory_token_count(tokenizer, steps)
        clean_count = _trajectory_token_count(tokenizer, clean_steps) if len(clean_steps) == 5 else 0
        ratio = count / clean_count if clean_count else float("inf")
        metrics[name] = {"token_count": count, "token_ratio": ratio}
        if name != "clean" and not min_token_ratio <= ratio <= max_token_ratio:
            rejection.append(f"{name}:token_ratio")
        error_positions = [index for index, label in enumerate(labels) if label == -1]
        if name != "clean":
            if "Discarding that branch" not in steps[2 if len(error_positions) == 1 else 3]:
                rejection.append(f"{name}:missing_explicit_recovery")
            branch_id = variant.get("wrong_branch_id")
            if not isinstance(branch_id, str) or any(
                branch_id not in steps[index] for index in error_positions
            ):
                rejection.append(f"{name}:broken_wrong_branch")
            distances = [
                normalized_levenshtein(
                    _token_ids(tokenizer, clean_steps[index]),
                    _token_ids(tokenizer, steps[index]),
                )
                for index in error_positions
            ]
            metrics[name]["error_edit_distances"] = distances
            if any(value < min_edit_distance for value in distances):
                rejection.append(f"{name}:weak_conflict")
    return {
        "accepted": not rejection,
        "rejection_codes": sorted(set(rejection)),
        "metrics": metrics,
    }


def verify_frozen_manifest(
    manifest: dict[str, Any], *, expected_train: int, expected_test: int
) -> None:
    expected_hash = manifest.get("manifest_sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("Frozen manifest has no valid SHA-256")
    if canonical_hash(manifest) != expected_hash:
        raise ValueError("Frozen manifest SHA-256 mismatch")
    entries = manifest.get("entries")
    test_ids = manifest.get("test_problem_ids")
    if not isinstance(entries, list) or len(entries) != expected_train:
        raise ValueError("Frozen manifest training count mismatch")
    if not isinstance(test_ids, list) or len(test_ids) != expected_test:
        raise ValueError("Frozen manifest test count mismatch")
    train_ids = [str(row["question_id"]) for row in entries]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError("Frozen manifest contains duplicate problem IDs")
    if set(train_ids) & set(test_ids):
        raise ValueError("Training/test problem overlap")
    for row in entries:
        if row.get("five_tuple_sha256") != canonical_hash(row):
            raise ValueError("Five-tuple SHA-256 mismatch")


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line.strip():
                yield line_number, json.loads(line)


def _repo_commit(repo_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _clean_steps_from_solution(solution: str, answer: str) -> tuple[str, ...]:
    compact = re.sub(r"\[asy\].*?\[/asy\]", "", solution, flags=re.DOTALL)
    units = [
        item.strip()
        for item in re.split(r"(?<=[.?])\s+|\n{2,}|;\s+", compact)
        if item.strip()
    ]
    if not units:
        units = ["The official solution applies the stated conditions directly."]
    buckets: list[list[str]] = []
    for index in range(3):
        start = round(index * len(units) / 3)
        end = round((index + 1) * len(units) / 3)
        buckets.append(units[start:end])
    reasoning = [" ".join(bucket).strip() for bucket in buckets]
    fillers = (
        "We keep the original target and all stated constraints.",
        "The official relation is applied without changing the requested quantity.",
        "The resulting value is checked against the original conditions.",
    )
    reasoning = [text or fillers[index] for index, text in enumerate(reasoning)]
    return (
        fillers[0],
        reasoning[0],
        reasoning[1],
        reasoning[2],
        f"# Answer\n\n{answer.strip()}",
    )


def _clean_steps_from_gsm8k(example: OfficialExample) -> tuple[str, ...]:
    """Preserve every released GSM8K calculation before the answer step."""
    if len(example.steps) != 5:
        raise ValueError("An exact five-step GSM8K trajectory is required")
    return (
        "Following the quantities and relations stated in the original question, "
        f"the first released correct calculation is {example.steps[0]}. We retain "
        "that intermediate result for the dependent calculations. No quantity is "
        "reinterpreted, omitted, or replaced, and the operation is kept in the "
        "same direction as the released solution.",
        "Using that result without changing the requested target or any condition, "
        f"the second released correct calculation is {example.steps[1]}. This value "
        "is carried forward exactly as specified. The operands still refer to the "
        "same entities in the question, so this step remains on the intended "
        "solution branch rather than introducing an alternative assumption.",
        "Continuing along the same correct dependency chain, the third released "
        f"calculation is {example.steps[2]}. It is consistent with the original "
        "question and supplies the next required quantity. Substituting the earlier "
        "results here preserves their units and dependencies, and the computed "
        "intermediate value is not rounded or redirected to another target.",
        "The remaining released calculations complete and check the derivation: "
        f"{example.steps[3]} followed by {example.steps[4]}. These calculations "
        "preserve the original quantities, operations, and requested answer. A final "
        "check traces every value back through the preceding released calculations, "
        "confirms that the units and requested quantity agree, and leaves the exact "
        "result ready for the answer field. This verification completes the same "
        "unchanged reasoning path supplied with the dataset.",
        f"# Answer\n\n{example.answer.strip()}",
    )


def _context_fits(
    row: dict[str, Any], tokenizer, token_ids: dict[str, int], config: dict[str, Any]
) -> bool:
    for variant in row["variants"].values():
        example = OfficialExample(
            idx=0,
            question=row["problem"],
            steps=tuple(variant["steps"]),
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
            return False
    return True


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if canonical_hash(existing) != canonical_hash(payload):
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        return
    atomic_json(path, payload)


def select_stratified_audit_entries(
    entries: Iterable[dict[str, Any]], *, count: int, selection_domain: str
) -> list[dict[str, Any]]:
    rows = list(entries)
    if count < 1 or count > len(rows):
        raise ValueError("Audit count must be between one and the entry count")
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        kind = str(row.get("source", {}).get("kind", "unknown"))
        groups.setdefault(kind, []).append(row)

    def audit_key(row: dict[str, Any]) -> tuple[str, str]:
        question_id = str(row["question_id"])
        digest = sha256(
            f"{selection_domain}:audit\0{question_id}".encode("utf-8")
        ).hexdigest()
        return digest, question_id

    ordered_groups = sorted(groups)
    target_per_group = count // len(ordered_groups)
    chosen: list[dict[str, Any]] = []
    chosen_ids: set[str] = set()
    for kind in ordered_groups:
        quota = min(target_per_group, len(groups[kind]))
        for row in sorted(groups[kind], key=audit_key)[:quota]:
            chosen.append(row)
            chosen_ids.add(str(row["question_id"]))
    remaining = sorted(
        (row for row in rows if str(row["question_id"]) not in chosen_ids),
        key=audit_key,
    )
    chosen.extend(remaining[: count - len(chosen)])
    return sorted(chosen, key=audit_key)


def prepare_self_corrected_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    if str(config.get("dataset_family", "math")).lower() == "gsm8k":
        return prepare_self_corrected_gsm8k_data(config, project_root=project_root)
    root = Path(project_root).resolve()
    repo_dir = (root / config["prm_repo_dir"]).resolve()
    if _repo_commit(repo_dir) != config["prm_repo_commit"]:
        raise ValueError("PRM800K repository commit mismatch")
    hash_checks = [
        *((item["path"], item["sha256"]) for item in config["prm_train_files"]),
        (config["math_train_path"], config["math_train_sha256"]),
        (config["math_test_path"], config["math_test_sha256"]),
        (config["checkpoint_path"], config["checkpoint_sha256"]),
    ]
    for relative, expected in hash_checks:
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artifact missing or SHA-256 mismatch: {path}")

    grade_answer = load_official_grader(repo_dir)
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    math_train = [row for _, row in _jsonl((root / config["math_train_path"]).resolve())]
    math_test = [row for _, row in _jsonl((root / config["math_test_path"]).resolve())]
    for row in (*math_train, *math_test):
        row["problem"] = normalize_problem(row["problem"])
        row["problem_id"] = problem_id(row["problem"])
    train_math_ids = {row["problem_id"] for row in math_train}
    test_ids = [row["problem_id"] for row in math_test]
    if len(test_ids) != int(config["test_examples"]):
        raise ValueError("Official MATH test count mismatch")

    reconstructed = []
    rejection_counts: Counter[str] = Counter()
    for source in config["prm_train_files"]:
        path = (root / source["path"]).resolve()
        for line_number, record in _jsonl(path):
            trajectory, rejection = reconstruct_trajectory(
                record,
                source_phase=source["phase"],
                source_file=source["path"],
                source_line=line_number,
                grade_answer=grade_answer,
            )
            if rejection is not None:
                rejection_counts[rejection] += 1
                continue
            assert trajectory is not None
            if (
                classify_ratings(trajectory.ratings) == "clean"
                and trajectory.problem_id in train_math_ids
                and trajectory.problem_id not in set(test_ids)
            ):
                reconstructed.append(trajectory)
    clean_trajectories, duplicate_counts = consolidate_trajectories(reconstructed)
    rejection_counts.update(duplicate_counts)
    clean_trajectories.sort(
        key=lambda row: (
            sha256(
                f"{config['selection_domain']}\0{row.problem_id}".encode("utf-8")
            ).hexdigest(),
            row.trajectory_sha256,
        )
    )

    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()

    def try_add(
        *,
        question_id: str,
        problem: str,
        answer: str,
        generated_answer: str,
        steps: Iterable[str],
        source: dict[str, Any],
    ) -> None:
        if question_id in used_ids or len(selected) >= int(config["train_examples"]):
            return
        try:
            row = construct_five_tuple(
                question_id=question_id,
                problem=problem,
                answer=answer,
                generated_answer=generated_answer,
                clean_steps=steps,
                tokenizer=tokenizer,
                min_token_ratio=float(config["min_aux_token_ratio"]),
                max_token_ratio=float(config["max_aux_token_ratio"]),
                min_edit_distance=float(config["min_normalized_edit_distance"]),
                source=source,
            )
            validation = validate_five_tuple(
                row,
                tokenizer=tokenizer,
                min_token_ratio=float(config["min_aux_token_ratio"]),
                max_token_ratio=float(config["max_aux_token_ratio"]),
                min_edit_distance=float(config["min_normalized_edit_distance"]),
                grade_answer=grade_answer,
            )
            if not validation["accepted"]:
                rejection_counts.update(validation["rejection_codes"])
                return
            if not _context_fits(row, tokenizer, token_ids, config):
                rejection_counts["context_length"] += 1
                return
            row["validation"] = validation
            row["five_tuple_sha256"] = canonical_hash(row)
        except (ValueError, OverflowError) as error:
            rejection_counts[f"construction:{type(error).__name__}"] += 1
            return
        selected.append(row)
        used_ids.add(question_id)

    for trajectory in clean_trajectories:
        try_add(
            question_id=trajectory.problem_id,
            problem=trajectory.problem,
            answer=trajectory.ground_truth_answer,
            generated_answer=trajectory.generated_answer,
            steps=trajectory.steps,
            source={
                "kind": "prm800k_clean_chosen",
                "source_phase": trajectory.source_phase,
                "source_file": trajectory.source_file,
                "source_line": trajectory.source_line,
                "record_sha256": trajectory.record_sha256,
                "trajectory_sha256": trajectory.trajectory_sha256,
            },
        )

    if len(selected) < int(config["train_examples"]):
        fallback_rows = sorted(
            math_train,
            key=lambda row: (
                sha256(
                    f"{config['selection_domain']}:fallback\0{row['problem_id']}".encode(
                        "utf-8"
                    )
                ).hexdigest(),
                row["problem_id"],
            ),
        )
        for row in fallback_rows:
            try_add(
                question_id=row["problem_id"],
                problem=row["problem"],
                answer=row["answer"],
                generated_answer=row["answer"],
                steps=_clean_steps_from_solution(row["solution"], row["answer"]),
                source={
                    "kind": "codex_segmented_official_math_solution",
                    "unique_id": row.get("unique_id"),
                    "subject": row.get("subject"),
                    "level": row.get("level"),
                },
            )
            if len(selected) >= int(config["train_examples"]):
                break

    if len(selected) < int(config["train_examples"]):
        audit = {
            "schema_version": 1,
            "status": "CONSTRUCTION_INCOMPLETE",
            "selected_five_tuples": len(selected),
            "required_five_tuples": int(config["train_examples"]),
            "rejection_counts": dict(rejection_counts),
        }
        atomic_json((root / config["data_audit_path"]).resolve(), audit)
        raise RuntimeError(
            "Could not construct 512 valid five-tuples without relaxing the frozen rules"
        )

    selected = selected[: int(config["train_examples"])]
    manifest = {
        "schema_version": 1,
        "selection_domain": config["selection_domain"],
        "prompt_version": config["prompt_version"],
        "entries": selected,
        "test_problem_ids": test_ids,
        "test_entries": [
            {
                "problem_id": row["problem_id"],
                "problem": row["problem"],
                "answer": row["answer"],
                "subject": row.get("subject"),
                "level": row.get("level"),
            }
            for row in math_test
        ],
        "generator_disclosure": (
            "Strong-conflict and recovery steps were produced by deterministic rules "
            "authored in the current Codex task; they are not natural teacher samples."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )

    raw_path = (root / config["raw_constructions_path"]).resolve()
    raw_rows = [
        {
            "question_id": row["question_id"],
            "prompt_version": config["prompt_version"],
            "source": row["source"],
            "five_tuple_sha256": row["five_tuple_sha256"],
            "variants": row["variants"],
        }
        for row in selected
    ]
    if raw_path.exists():
        if read_jsonl(raw_path) != raw_rows:
            raise FileExistsError(f"Refusing to overwrite raw constructions: {raw_path}")
    else:
        _atomic_jsonl(raw_path, raw_rows)

    manifest_path = (root / config["manifest_path"]).resolve()
    _write_immutable_json(manifest_path, manifest)
    audit_entries = select_stratified_audit_entries(
        selected,
        count=int(config["audit_examples"]),
        selection_domain=str(config["selection_domain"]),
    )
    audit_source_counts = Counter(row["source"]["kind"] for row in audit_entries)
    audit_bundle = {
        "schema_version": 1,
        "status": "READY_FOR_HUMAN_AUDIT",
        "examples": audit_entries,
        "sampling": {
            "method": "deterministic_source_stratified_sha256",
            "source_counts": dict(audit_source_counts),
        },
        "manifest_sha256": manifest["manifest_sha256"],
    }
    atomic_json((root / config["audit_bundle_path"]).resolve(), audit_bundle)
    source_counts = Counter(row["source"]["kind"] for row in selected)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "selected_five_tuples": len(selected),
        "constructed_trajectories": len(selected) * len(VARIANT_LABELS),
        "final_answer_correct": len(selected) * len(VARIANT_LABELS),
        "source_counts": dict(source_counts),
        "rejection_counts": dict(rejection_counts),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_constructions_path": str(raw_path),
        "raw_constructions_sha256": sha256_file(raw_path),
        "test_problem_overlap": 0,
        "natural_noise_quantity_gate_used": False,
    }
    atomic_json((root / config["data_audit_path"]).resolve(), audit)
    return audit


def prepare_self_corrected_gsm8k_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    """Freeze the approved self-corrected conflicts in the GSM8K domain."""
    root = Path(project_root).resolve()
    hash_checks = (
        (config["gsm_train_path"], config["gsm_train_sha256"]),
        (config["gsm_test_path"], config["gsm_test_sha256"]),
        (config["checkpoint_path"], config["checkpoint_sha256"]),
    )
    for relative, expected in hash_checks:
        path = (root / relative).resolve()
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Artifact missing or SHA-256 mismatch: {path}")

    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())
    test_examples = list(iter_icot_examples((root / config["gsm_test_path"]).resolve()))
    if len(test_examples) != int(config["test_examples"]):
        raise ValueError("Official GSM8K test count mismatch")
    test_entries = [
        {
            "problem_id": problem_id(example.question),
            "problem": normalize_problem(example.question),
            "answer": example.answer.strip(),
            "subject": "GSM8K",
            "level": None,
        }
        for example in test_examples
    ]
    test_ids = [row["problem_id"] for row in test_entries]
    if len(set(test_ids)) != len(test_ids):
        raise ValueError("Official GSM8K test contains duplicate questions")
    test_id_set = set(test_ids)

    rejection_counts: Counter[str] = Counter()
    candidates: list[tuple[str, OfficialExample]] = []
    seen: set[str] = set()
    for example in iter_icot_examples((root / config["gsm_train_path"]).resolve()):
        if len(example.steps) != 5:
            rejection_counts["not_exactly_five_steps"] += 1
            continue
        question_id = problem_id(example.question)
        if question_id in test_id_set:
            rejection_counts["test_overlap"] += 1
            continue
        if question_id in seen:
            rejection_counts["duplicate_question"] += 1
            continue
        seen.add(question_id)
        candidates.append((question_id, example))
    candidates.sort(
        key=lambda item: (
            sha256(
                f"{config['selection_domain']}\0{item[0]}".encode("utf-8")
            ).hexdigest(),
            item[0],
        )
    )

    def gsm8k_grader(prediction: str, target: str) -> bool:
        return _normalized_answer(prediction) == _normalized_answer(target)

    selected: list[dict[str, Any]] = []
    for question_id, example in candidates:
        try:
            row = construct_five_tuple(
                question_id=question_id,
                problem=example.question,
                answer=example.answer,
                generated_answer=example.answer,
                clean_steps=_clean_steps_from_gsm8k(example),
                tokenizer=tokenizer,
                min_token_ratio=float(config["min_aux_token_ratio"]),
                max_token_ratio=float(config["max_aux_token_ratio"]),
                min_edit_distance=float(config["min_normalized_edit_distance"]),
                source={
                    "kind": "official_gsm8k_exact_five_step",
                    "source_file": config["gsm_train_path"],
                    "source_line": int(example.idx),
                },
            )
            validation = validate_five_tuple(
                row,
                tokenizer=tokenizer,
                min_token_ratio=float(config["min_aux_token_ratio"]),
                max_token_ratio=float(config["max_aux_token_ratio"]),
                min_edit_distance=float(config["min_normalized_edit_distance"]),
                grade_answer=gsm8k_grader,
            )
            if not validation["accepted"]:
                rejection_counts.update(validation["rejection_codes"])
                continue
            if not _context_fits(row, tokenizer, token_ids, config):
                rejection_counts["context_length"] += 1
                continue
            row["validation"] = validation
            row["five_tuple_sha256"] = canonical_hash(row)
        except (ValueError, OverflowError) as error:
            rejection_counts[f"construction:{type(error).__name__}"] += 1
            continue
        selected.append(row)
        if len(selected) >= int(config["train_examples"]):
            break

    if len(selected) < int(config["train_examples"]):
        audit = {
            "schema_version": 1,
            "status": "CONSTRUCTION_INCOMPLETE",
            "selected_five_tuples": len(selected),
            "required_five_tuples": int(config["train_examples"]),
            "rejection_counts": dict(rejection_counts),
        }
        atomic_json((root / config["data_audit_path"]).resolve(), audit)
        raise RuntimeError("Could not construct the frozen GSM8K five-tuples")

    manifest = {
        "schema_version": 1,
        "dataset_family": "gsm8k",
        "selection_domain": config["selection_domain"],
        "prompt_version": config["prompt_version"],
        "entries": selected,
        "test_problem_ids": test_ids,
        "test_entries": test_entries,
        "generator_disclosure": (
            "Strong-conflict and recovery steps were produced by deterministic rules "
            "authored in the current Codex task; they are not natural teacher samples."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )

    raw_path = (root / config["raw_constructions_path"]).resolve()
    raw_rows = [
        {
            "question_id": row["question_id"],
            "prompt_version": config["prompt_version"],
            "source": row["source"],
            "five_tuple_sha256": row["five_tuple_sha256"],
            "variants": row["variants"],
        }
        for row in selected
    ]
    if raw_path.exists():
        if read_jsonl(raw_path) != raw_rows:
            raise FileExistsError(f"Refusing to overwrite raw constructions: {raw_path}")
    else:
        _atomic_jsonl(raw_path, raw_rows)
    manifest_path = (root / config["manifest_path"]).resolve()
    _write_immutable_json(manifest_path, manifest)

    audit_entries = select_stratified_audit_entries(
        selected,
        count=int(config["audit_examples"]),
        selection_domain=str(config["selection_domain"]),
    )
    audit_bundle = {
        "schema_version": 1,
        "status": "READY_FOR_CODEX_SEMANTIC_AUDIT",
        "examples": audit_entries,
        "sampling": {"method": "deterministic_sha256"},
        "manifest_sha256": manifest["manifest_sha256"],
    }
    atomic_json((root / config["audit_bundle_path"]).resolve(), audit_bundle)
    audit = {
        "schema_version": 1,
        "status": "PASS",
        "dataset_family": "gsm8k",
        "selected_five_tuples": len(selected),
        "constructed_trajectories": len(selected) * len(VARIANT_LABELS),
        "final_answer_correct": len(selected) * len(VARIANT_LABELS),
        "source_counts": {"official_gsm8k_exact_five_step": len(selected)},
        "rejection_counts": dict(rejection_counts),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "raw_constructions_path": str(raw_path),
        "raw_constructions_sha256": sha256_file(raw_path),
        "test_problem_overlap": 0,
        "official_gsm8k_test_examples": len(test_entries),
        "natural_noise_quantity_gate_used": False,
    }
    atomic_json((root / config["data_audit_path"]).resolve(), audit)
    return audit
