from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from itertools import product
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
import importlib
import json
import subprocess
import sys
import unicodedata

from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer
from .single_gpu_smoke import encode_smoke_example


GradeAnswer = Callable[[str, str], bool]
TokenLength = Callable[["NaturalTrajectory"], int]


def canonical_hash(payload: Any) -> str:
    value = dict(payload) if isinstance(payload, dict) else payload
    if isinstance(value, dict):
        value.pop("manifest_sha256", None)
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def normalize_problem(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.splitlines()).strip()


def problem_id(text: str) -> str:
    return sha256(normalize_problem(text).encode("utf-8")).hexdigest()


def classify_ratings(ratings: tuple[int, ...]) -> str | None:
    if len(ratings) != 5 or any(value not in {-1, 1} for value in ratings):
        return None
    errors = ratings.count(-1)
    return {0: "clean", 1: "noise1", 2: "noise2"}.get(errors)


def _raw_original_rating_pattern(record: dict[str, Any]) -> str | None:
    label = record.get("label")
    labeled_steps = label.get("steps") if isinstance(label, dict) else None
    if not isinstance(labeled_steps, list) or len(labeled_steps) != 5:
        return None
    ratings: list[int] = []
    for step in labeled_steps:
        if not isinstance(step, dict):
            return None
        chosen = step.get("chosen_completion")
        completions = step.get("completions")
        if not isinstance(chosen, int) or not isinstance(completions, list):
            return None
        if not (0 <= chosen < len(completions)):
            return None
        completion = completions[chosen]
        if not isinstance(completion, dict) or completion.get("rating") not in {-1, 1}:
            return None
        ratings.append(int(completion["rating"]))
    errors = ratings.count(-1)
    return {0: "clean", 1: "noise1", 2: "noise2"}.get(errors, f"errors_{errors}")


@dataclass(frozen=True)
class NaturalTrajectory:
    problem_id: str
    problem: str
    ground_truth_answer: str
    generated_answer: str
    steps: tuple[str, ...]
    ratings: tuple[int, ...]
    generation_key: str
    source_phase: str
    source_file: str
    source_line: int
    record_sha256: str
    trajectory_sha256: str

    def to_record(self) -> dict[str, Any]:
        row = asdict(self)
        row["steps"] = list(self.steps)
        row["ratings"] = list(self.ratings)
        row["class"] = classify_ratings(self.ratings)
        return row


def _question_fields(record: dict[str, Any]) -> tuple[str, str, str | None]:
    question = record.get("question")
    if not isinstance(question, dict):
        raise KeyError("question")
    problem = question.get("problem")
    ground_truth = question.get("ground_truth_answer")
    generated = question.get("pre_generated_answer")
    if not all(isinstance(item, str) and item.strip() for item in (problem, ground_truth)):
        raise KeyError("question fields")
    if generated is not None and (not isinstance(generated, str) or not generated.strip()):
        raise KeyError("question fields")
    return problem, ground_truth, generated


def _phase1_final_answer(text: str) -> str | None:
    marker = "# Answer"
    if marker not in text:
        return None
    answer = text.rsplit(marker, 1)[1].strip()
    return answer or None


def reconstruct_trajectory(
    record: dict[str, Any],
    *,
    source_phase: str,
    source_file: str,
    source_line: int,
    grade_answer: GradeAnswer,
) -> tuple[NaturalTrajectory | None, str | None]:
    if record.get("is_quality_control_question") is True:
        return None, "quality_control"
    if record.get("is_initial_screening_question") is True:
        return None, "initial_screening"
    try:
        problem, ground_truth, generated = _question_fields(record)
    except KeyError:
        return None, "missing_question_fields"
    label = record.get("label")
    labeled_steps = label.get("steps") if isinstance(label, dict) else None
    if not isinstance(labeled_steps, list) or len(labeled_steps) != 5:
        return None, "not_exactly_five_steps"

    texts: list[str] = []
    ratings: list[int] = []
    for step in labeled_steps:
        if not isinstance(step, dict):
            return None, "malformed_step"
        chosen = step.get("chosen_completion")
        if not isinstance(chosen, int):
            return None, "human_completion"
        completions = step.get("completions")
        if not isinstance(completions, list) or not (0 <= chosen < len(completions)):
            return None, "invalid_chosen_completion"
        completion = completions[chosen]
        if not isinstance(completion, dict):
            return None, "malformed_completion"
        if bool(completion.get("flagged")):
            return None, "flagged"
        text = completion.get("text")
        rating = completion.get("rating")
        if not isinstance(text, str) or not text.strip():
            return None, "empty_step"
        if rating == 0:
            return None, "rating_zero"
        if rating not in {-1, 1}:
            return None, "invalid_rating"
        texts.append(text)
        ratings.append(int(rating))

    if generated is None:
        if source_phase != "phase1":
            return None, "missing_generated_answer"
        if label.get("finish_reason") != "solution":
            return None, "unfinished_phase1_trajectory"
        generated = _phase1_final_answer(texts[-1])
        if generated is None:
            return None, "missing_phase1_final_answer"
    try:
        if not grade_answer(generated, ground_truth):
            return None, "wrong_answer"
    except Exception:
        return None, "grader_error"

    rating_tuple = tuple(ratings)
    if classify_ratings(rating_tuple) is None:
        return None, "unsupported_error_count"
    normalized_problem = normalize_problem(problem)
    generation = record.get("generation")
    generation_key = f"{source_phase}:{generation if generation is not None else 'null'}"
    record_hash = canonical_hash(record)
    trajectory_payload = {
        "problem": normalized_problem,
        "generation_key": generation_key,
        "steps": texts,
    }
    trajectory = NaturalTrajectory(
        problem_id=problem_id(normalized_problem),
        problem=normalized_problem,
        ground_truth_answer=ground_truth.strip(),
        generated_answer=generated.strip(),
        steps=tuple(texts),
        ratings=rating_tuple,
        generation_key=generation_key,
        source_phase=source_phase,
        source_file=source_file,
        source_line=source_line,
        record_sha256=record_hash,
        trajectory_sha256=canonical_hash(trajectory_payload),
    )
    return trajectory, None


def consolidate_trajectories(
    trajectories: Iterable[NaturalTrajectory],
) -> tuple[list[NaturalTrajectory], Counter[str]]:
    grouped: dict[tuple[str, str, str], list[NaturalTrajectory]] = defaultdict(list)
    for row in trajectories:
        grouped[(row.problem_id, row.generation_key, row.trajectory_sha256)].append(row)
    kept: list[NaturalTrajectory] = []
    counts: Counter[str] = Counter()
    for rows in grouped.values():
        signatures = {row.ratings for row in rows}
        if len(signatures) != 1:
            counts["label_conflict"] += len(rows)
            continue
        selected = min(rows, key=lambda row: row.record_sha256)
        kept.append(selected)
        counts["duplicate_annotations_removed"] += len(rows) - 1
    kept.sort(key=lambda row: (row.problem_id, row.generation_key, row.trajectory_sha256))
    return kept, counts


def select_strict_triplets(
    trajectories: Iterable[NaturalTrajectory], *, token_length: TokenLength
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, list[NaturalTrajectory]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in trajectories:
        category = classify_ratings(row.ratings)
        if category is not None:
            grouped[(row.problem_id, row.generation_key)][category].append(row)

    triplets: list[dict[str, Any]] = []
    for (question_id, generation_key), by_class in grouped.items():
        if not all(by_class.get(name) for name in ("clean", "noise1", "noise2")):
            continue
        lengths = {row.trajectory_sha256: int(token_length(row)) for values in by_class.values() for row in values}
        candidates: list[tuple[Any, ...]] = []
        for clean, noise1, noise2 in product(
            by_class["clean"], by_class["noise1"], by_class["noise2"]
        ):
            selected = (clean, noise1, noise2)
            selected_lengths = tuple(lengths[row.trajectory_sha256] for row in selected)
            hashes = tuple(row.trajectory_sha256 for row in selected)
            candidates.append((max(selected_lengths) - min(selected_lengths), hashes, selected_lengths, selected))
        spread, _, selected_lengths, selected = min(candidates)
        clean, noise1, noise2 = selected
        triplets.append(
            {
                "problem_id": question_id,
                "problem": clean.problem,
                "ground_truth_answer": clean.ground_truth_answer,
                "generation_key": generation_key,
                "clean": clean.to_record(),
                "noise1": noise1.to_record(),
                "noise2": noise2.to_record(),
                "token_lengths": {
                    "clean": selected_lengths[0],
                    "noise1": selected_lengths[1],
                    "noise2": selected_lengths[2],
                },
                "token_length_spread": spread,
            }
        )
    triplets.sort(key=lambda row: (row["problem_id"], row["generation_key"]))
    return triplets


def collapse_triplets_by_problem(triplets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in triplets:
        grouped[row["problem_id"]].append(row)
    selected: list[dict[str, Any]] = []
    for rows in grouped.values():
        selected.append(
            min(
                rows,
                key=lambda row: (
                    row["token_length_spread"],
                    tuple(
                        row[name]["trajectory_sha256"]
                        for name in ("clean", "noise1", "noise2")
                    ),
                    row["generation_key"],
                ),
            )
        )
    selected.sort(key=lambda row: row["problem_id"])
    return selected


def verify_frozen_triplet_manifest(
    manifest: dict[str, Any], *, expected_train: int, expected_dev: int, expected_confirm: int
) -> None:
    expected = manifest.get("manifest_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("Frozen manifest has no valid SHA-256")
    if canonical_hash(manifest) != expected:
        raise ValueError("Frozen manifest SHA-256 mismatch")
    train_ids = [row["problem_id"] for row in manifest["train_triplets"]]
    dev_ids = [row["problem_id"] for row in manifest["dev_entries"]]
    confirm_ids = list(manifest["confirm_problem_ids"])
    if len(train_ids) != expected_train or len(dev_ids) != expected_dev or len(confirm_ids) != expected_confirm:
        raise ValueError("Frozen manifest split size mismatch")
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids) or len(set(confirm_ids)) != len(confirm_ids):
        raise ValueError("Duplicate problem in frozen manifest")
    if set(train_ids) & set(dev_ids) or set(train_ids) & set(confirm_ids) or set(dev_ids) & set(confirm_ids):
        raise ValueError("Training/development/confirmation leakage")


def _jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if line.strip():
                yield line_no, json.loads(line)


def _is_lfs_pointer(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(64).startswith(b"version https://git-lfs.github.com/spec/v1")


def load_official_grader(repo_dir: Path) -> GradeAnswer:
    package_root = str((repo_dir / "prm800k").resolve())
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    module = importlib.import_module("grading.grader")
    return module.grade_answer


def _repo_commit(repo_dir: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_math_split(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in _jsonl(path):
        problem = normalize_problem(row["problem"])
        rows.append({**row, "problem": problem, "problem_id": problem_id(problem)})
    return rows


def prepare_prm800k_data(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    repo_dir = (root / config["prm_repo_dir"]).resolve()
    if _repo_commit(repo_dir) != config["prm_repo_commit"]:
        raise ValueError("PRM800K repository commit mismatch")

    checks = [
        *((item["path"], item["sha256"]) for item in config["prm_train_files"]),
        (config["math_train_path"], config["math_train_sha256"]),
        (config["math_confirm_path"], config["math_confirm_sha256"]),
        (config["checkpoint_path"], config["checkpoint_sha256"]),
    ]
    for relative, expected_hash in checks:
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if _is_lfs_pointer(path):
            raise ValueError(f"Git LFS pointer has not been materialized: {path}")
        if sha256_file(path) != expected_hash:
            raise ValueError(f"Artifact SHA-256 mismatch: {path}")

    grade_answer = load_official_grader(repo_dir)
    math_train = _load_math_split((root / config["math_train_path"]).resolve())
    math_confirm = _load_math_split((root / config["math_confirm_path"]).resolve())
    train_math_ids = {row["problem_id"] for row in math_train}
    confirm_ids = [row["problem_id"] for row in math_confirm]
    if len(confirm_ids) != int(config["confirm_examples"]):
        raise ValueError("Official MATH confirmation split size mismatch")

    rejection_counts: Counter[str] = Counter()
    structural_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    wrong_answer_patterns: dict[str, Counter[str]] = defaultdict(Counter)
    reconstructed: list[NaturalTrajectory] = []
    raw_lines = 0
    for source in config["prm_train_files"]:
        path = (root / source["path"]).resolve()
        for line_no, record in _jsonl(path):
            raw_lines += 1
            raw_pattern = _raw_original_rating_pattern(record)
            if (
                raw_pattern is not None
                and record.get("is_quality_control_question") is not True
                and record.get("is_initial_screening_question") is not True
            ):
                structural_patterns[source["phase"]][raw_pattern] += 1
            row, rejection = reconstruct_trajectory(
                record,
                source_phase=source["phase"],
                source_file=source["path"],
                source_line=line_no,
                grade_answer=grade_answer,
            )
            if rejection is not None:
                rejection_counts[rejection] += 1
                if rejection == "wrong_answer" and raw_pattern is not None:
                    wrong_answer_patterns[source["phase"]][raw_pattern] += 1
                continue
            assert row is not None
            if row.problem_id not in train_math_ids:
                rejection_counts["not_in_official_math_train"] += 1
                continue
            reconstructed.append(row)

    consolidated, duplicate_counts = consolidate_trajectories(reconstructed)
    rejection_counts.update(duplicate_counts)
    tokenizer, token_ids = build_tokenizer((root / config["base_model_dir"]).resolve())

    context_eligible: list[NaturalTrajectory] = []
    token_lengths: dict[str, int] = {}
    for row in consolidated:
        example = OfficialExample(
            idx=row.source_line,
            question=row.problem,
            steps=row.steps,
            answer=row.ground_truth_answer,
        )
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=int(config["latent_stage"]),
            c_thought=int(config["c_thought"]),
        )
        if len(encoded.input_ids) > int(config["max_sequence_tokens"]) or encoded.maximum_auxiliary_length > int(config["max_sequence_tokens"]):
            rejection_counts["context_length"] += 1
            continue
        token_lengths[row.trajectory_sha256] = sum(
            len(tokenizer.encode(step + "\n", add_special_tokens=False)) for step in row.steps
        )
        context_eligible.append(row)

    triplets = collapse_triplets_by_problem(
        select_strict_triplets(
            context_eligible,
            token_length=lambda row: token_lengths[row.trajectory_sha256],
        )
    )
    domain = config["selection_domain"]
    triplets.sort(
        key=lambda row: (
            sha256(f"{domain}\0{row['problem_id']}".encode("utf-8")).hexdigest(),
            row["problem_id"],
        )
    )
    train_count = int(config["train_examples"])
    dev_count = int(config["dev_examples"])
    status = "PASS"
    if len(triplets) < train_count:
        status = "INSUFFICIENT_STRICT_TRIPLETS"

    selected_train = triplets[:train_count] if status == "PASS" else []
    train_ids = {row["problem_id"] for row in selected_train}
    clean_by_problem: dict[str, list[NaturalTrajectory]] = defaultdict(list)
    for row in context_eligible:
        if classify_ratings(row.ratings) == "clean" and row.problem_id not in train_ids:
            clean_by_problem[row.problem_id].append(row)
    dev_candidates: list[dict[str, Any]] = []
    for question_id, rows in clean_by_problem.items():
        selected = min(
            rows,
            key=lambda row: (token_lengths[row.trajectory_sha256], row.trajectory_sha256),
        )
        dev_candidates.append(selected.to_record())
    dev_candidates.sort(
        key=lambda row: (
            sha256(f"{domain}\0{row['problem_id']}".encode("utf-8")).hexdigest(),
            row["problem_id"],
        )
    )
    selected_dev = dev_candidates[:dev_count] if status == "PASS" else []
    if status == "PASS" and len(selected_dev) < dev_count:
        status = "INSUFFICIENT_CLEAN_DEV"
        selected_train = []
        selected_dev = []

    class_counts = Counter(
        classify_ratings(row.ratings) or "unsupported" for row in context_eligible
    )
    audit = {
        "schema_version": 1,
        "run_ids": ["PN001", "PN002", "PN003", "PN010", "PN011", "PN012"],
        "status": status,
        "prm_repo_commit": config["prm_repo_commit"],
        "raw_lines": raw_lines,
        "reconstructed_before_dedup": len(reconstructed),
        "structural_exact_five_original_rating_patterns": {
            phase: dict(counts) for phase, counts in structural_patterns.items()
        },
        "wrong_answer_by_structural_pattern": {
            phase: dict(counts) for phase, counts in wrong_answer_patterns.items()
        },
        "context_eligible_trajectories": len(context_eligible),
        "class_counts": dict(class_counts),
        "strict_triplets": len(triplets),
        "required_triplets": train_count,
        "eligible_clean_dev_questions": len(dev_candidates),
        "required_clean_dev_questions": dev_count,
        "rejection_counts": dict(rejection_counts),
        "confirm_examples": len(confirm_ids),
        "official_test_opened": False,
    }
    output_root = (root / config["output_root"]).resolve()
    atomic_json(output_root / "data_audit.json", audit)
    provenance = {
        "schema_version": 1,
        "run_id": "PN001",
        "status": "PASS",
        "prm_repo_commit": config["prm_repo_commit"],
        "source_files": [
            {"path": relative, "sha256": expected_hash}
            for relative, expected_hash in checks
        ],
        "selection_domain": domain,
        "checkpoint_sha256": config["checkpoint_sha256"],
        "official_test_opened": False,
    }
    atomic_json((root / config["provenance_path"]).resolve(), provenance)

    if status != "PASS":
        return audit

    manifest = {
        "schema_version": 1,
        "run_id": "PN012",
        "status": "PASS",
        "selection_domain": domain,
        "train_triplets": selected_train,
        "dev_entries": selected_dev,
        "confirm_problem_ids": confirm_ids,
        "official_test_opened": False,
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_triplet_manifest(
        manifest,
        expected_train=train_count,
        expected_dev=dev_count,
        expected_confirm=int(config["confirm_examples"]),
    )
    atomic_json((root / config["triplet_manifest_path"]).resolve(), manifest)
    return audit
