from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Any, Iterable
import ast
import json
import re

from .m1_training import atomic_json, sha256_file
from .official_adapter import OfficialExample, build_tokenizer, iter_icot_examples
from .self_corrected_data import canonical_hash, verify_frozen_manifest


EQUATION_RE = re.compile(r"^<<(.+)=([^=<>]+)>>$")
NUMBER_BOUNDARY = r"(?<![\d.]){value}(?![\d.])"
VARIANT_TYPES = {
    "clean": ("CLEAN",) * 5,
    "local_error": (
        "CLEAN",
        "DIRECT_FALSE",
        "CANCEL_FALSE",
        "CLEAN",
        "CLEAN",
    ),
    "local_redundant": (
        "CLEAN",
        "REDUNDANT",
        "REDUNDANT",
        "CLEAN",
        "CLEAN",
    ),
    "wide_error": (
        "DIRECT_FALSE",
        "ERROR_DESCENDANT",
        "ERROR_DESCENDANT",
        "CANCEL_FALSE",
        "CLEAN",
    ),
    "wide_redundant": (
        "REDUNDANT",
        "REDUNDANT",
        "REDUNDANT",
        "REDUNDANT",
        "CLEAN",
    ),
}


@dataclass(frozen=True)
class Equation:
    raw: str
    expression: str
    result_text: str
    expression_value: Fraction
    result_value: Fraction

    @property
    def is_true(self) -> bool:
        return self.expression_value == self.result_value


def _eval_node(node: ast.AST) -> Fraction:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Fraction(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_node(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
    ):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        return left / right
    raise ValueError(f"Unsupported arithmetic node: {type(node).__name__}")


def evaluate_arithmetic(text: str) -> Fraction:
    if not text or len(text) > 256:
        raise ValueError("Arithmetic expression is empty or too long")
    return _eval_node(ast.parse(text, mode="eval"))


def parse_equation(step: str) -> Equation:
    match = EQUATION_RE.fullmatch(step)
    if match is None:
        raise ValueError("Step is not one strict <<expression=result>> equation")
    expression, result = match.groups()
    return Equation(
        raw=step,
        expression=expression,
        result_text=result,
        expression_value=evaluate_arithmetic(expression),
        result_value=evaluate_arithmetic(result),
    )


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    reduced = denominator
    for factor in (2, 5):
        while reduced % factor == 0:
            reduced //= factor
    if reduced == 1:
        digits = max(
            _factor_count(denominator, 2),
            _factor_count(denominator, 5),
        )
        return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")
    return f"({value.numerator}/{value.denominator})"


def _factor_count(value: int, factor: int) -> int:
    count = 0
    while value % factor == 0:
        value //= factor
        count += 1
    return count


def _replace_number(expression: str, old: str, new: str) -> str:
    pattern = NUMBER_BOUNDARY.format(value=re.escape(old))
    # Replace one dependency edge, not every equal-looking numeric constant.
    # For example, in ``2*2`` only the first ``2`` is the carried state; the
    # second is the fixed multiplier. Replacing both creates a construction
    # artifact rather than a propagated arithmetic error.
    replaced, count = re.subn(pattern, new, expression, count=1)
    if count == 0:
        raise ValueError(f"Expression does not reference prior state {old}")
    return replaced


def _references(expression: str, result_text: str) -> bool:
    return re.search(
        NUMBER_BOUNDARY.format(value=re.escape(result_text)), expression
    ) is not None


def _neutral(expression: str) -> str:
    # +0 is the shortest permitted operation in the frozen tokenizer and is
    # safe at the outermost precedence level for every supported expression.
    return f"{expression}+0"


def _step(expression: str, result: str) -> str:
    return f"<<{expression}={result}>>"


def _propagated_expression(
    equation: Equation, prior_clean: Equation, prior_wrong: Fraction
) -> str:
    return _replace_number(
        equation.expression,
        prior_clean.result_text,
        _format_fraction(prior_wrong),
    )


def construct_variants(
    example: OfficialExample, *, delta: int, neutralize_errors: bool = True
) -> dict[str, Any]:
    if len(example.steps) != 5 or delta == 0:
        raise ValueError("A non-zero delta and exactly five source steps are required")
    equations = tuple(parse_equation(step) for step in example.steps)
    if not all(equation.is_true for equation in equations):
        raise ValueError("Clean contains a false arithmetic equation")
    if not all(
        _references(equations[index + 1].expression, equations[index].result_text)
        for index in range(4)
    ):
        raise ValueError("Clean does not contain the frozen four-edge dependency chain")
    if evaluate_arithmetic(example.answer) != equations[-1].result_value:
        raise ValueError("Clean final equation does not equal the answer label")

    local_error = list(example.steps)
    local_redundant = list(example.steps)
    local_wrong = equations[1].result_value + delta
    local_error_expression = (
        _neutral(equations[1].expression)
        if neutralize_errors
        else equations[1].expression
    )
    local_error[1] = _step(local_error_expression, _format_fraction(local_wrong))
    local_redundant[1] = _step(
        _neutral(equations[1].expression), equations[1].result_text
    )
    local_cancel_expression = _propagated_expression(
        equations[2], equations[1], local_wrong
    )
    local_error[2] = _step(
        _neutral(local_cancel_expression) if neutralize_errors else local_cancel_expression,
        equations[2].result_text,
    )
    local_redundant[2] = _step(
        _neutral(equations[2].expression), equations[2].result_text
    )

    wide_error = list(example.steps)
    wide_redundant = list(example.steps)
    wrong_values: list[Fraction] = [equations[0].result_value + delta]
    wide_error[0] = _step(
        _neutral(equations[0].expression)
        if neutralize_errors
        else equations[0].expression,
        _format_fraction(wrong_values[0]),
    )
    wide_redundant[0] = _step(
        _neutral(equations[0].expression), equations[0].result_text
    )
    for index in (1, 2):
        expression = _propagated_expression(
            equations[index], equations[index - 1], wrong_values[-1]
        )
        wrong_values.append(evaluate_arithmetic(expression))
        wide_error[index] = _step(
            _neutral(expression) if neutralize_errors else expression,
            _format_fraction(wrong_values[-1]),
        )
        wide_redundant[index] = _step(
            _neutral(equations[index].expression), equations[index].result_text
        )
    wide_cancel_expression = _propagated_expression(
        equations[3], equations[2], wrong_values[-1]
    )
    wide_error[3] = _step(
        _neutral(wide_cancel_expression)
        if neutralize_errors
        else wide_cancel_expression,
        equations[3].result_text,
    )
    wide_redundant[3] = _step(
        _neutral(equations[3].expression), equations[3].result_text
    )

    variants = {
        "clean": {"steps": list(example.steps), "types": list(VARIANT_TYPES["clean"])},
        "local_error": {
            "steps": local_error,
            "types": list(VARIANT_TYPES["local_error"]),
        },
        "local_redundant": {
            "steps": local_redundant,
            "types": list(VARIANT_TYPES["local_redundant"]),
        },
        "wide_error": {
            "steps": wide_error,
            "types": list(VARIANT_TYPES["wide_error"]),
        },
        "wide_redundant": {
            "steps": wide_redundant,
            "types": list(VARIANT_TYPES["wide_redundant"]),
        },
    }
    return variants


def _ceil_fraction(value: Fraction) -> int:
    if value < 0:
        raise ValueError("Only non-negative fractions can be rounded up")
    return (value.numerator + value.denominator - 1) // value.denominator


def _relative_deviation(actual: Fraction, reference: Fraction) -> Fraction:
    return abs(actual - reference) / max(Fraction(1), abs(reference))


def construct_severe_variants(
    example: OfficialExample,
    *,
    question_id: str,
    severity_multiplier: int,
    severity_floor: int,
    downstream_min_relative: float,
) -> tuple[dict[str, Any], int, dict[str, Any]]:
    """Build a sign-balanced, large-deviation error/cancellation trajectory.

    The arithmetic operations and dependency edges remain those of the released
    five-step solution.  Only the corrupted state magnitude changes relative to
    v10: the direct error is at least ``severity_multiplier`` times the largest
    clean state (with ``severity_floor`` as a lower bound), it is propagated for
    the same slots, and the frozen false cancellation returns to the clean path.
    """
    if severity_multiplier < 1 or severity_floor < 1:
        raise ValueError("Severe conflict scale and floor must be positive")
    if not 0 <= downstream_min_relative:
        raise ValueError("Downstream deviation threshold must be non-negative")
    equations = tuple(parse_equation(step) for step in example.steps)
    answer_value = evaluate_arithmetic(example.answer)
    scale = max(
        Fraction(severity_floor),
        abs(answer_value),
        *(abs(equation.result_value) for equation in equations),
    )
    magnitude = _ceil_fraction(scale * severity_multiplier)
    preferred_sign = 1 if int(question_id[:8], 16) % 2 == 0 else -1
    failures: list[str] = []
    for multiplier in (1, 2, 4, 8):
        for sign in (preferred_sign, -preferred_sign):
            delta = sign * magnitude * multiplier
            try:
                variants = construct_variants(
                    example,
                    delta=delta,
                    neutralize_errors=False,
                )
                local = tuple(
                    parse_equation(step) for step in variants["local_error"]["steps"]
                )
                wide = tuple(
                    parse_equation(step) for step in variants["wide_error"]["steps"]
                )
                local_direct = _relative_deviation(
                    local[1].result_value, equations[1].result_value
                )
                wide_deviations = tuple(
                    _relative_deviation(wide[index].result_value, equations[index].result_value)
                    for index in range(3)
                )
                if local_direct < severity_multiplier:
                    failures.append("local_direct_below_multiplier")
                    continue
                if wide_deviations[0] < severity_multiplier:
                    failures.append("wide_direct_below_multiplier")
                    continue
                if any(
                    wide[index].result_value == equations[index].result_value
                    for index in range(3)
                ):
                    failures.append("wide_state_not_changed")
                    continue
                if any(float(value) < downstream_min_relative for value in wide_deviations[1:]):
                    failures.append("wide_downstream_below_threshold")
                    continue
                metrics = {
                    "delta": delta,
                    "delta_abs": abs(delta),
                    "preferred_sign_used": sign == preferred_sign,
                    "local_direct_relative_deviation": float(local_direct),
                    "wide_relative_deviations": [float(value) for value in wide_deviations],
                    "wide_changed_state_count": sum(
                        wide[index].result_value != equations[index].result_value
                        for index in range(3)
                    ),
                    "wide_affected_step_count": 4,
                    "final_answer_preserved": (
                        wide[-1].result_value == answer_value
                        and local[-1].result_value == answer_value
                    ),
                }
                return variants, delta, metrics
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
                failures.append(f"{type(error).__name__}:{error}")
    raise ValueError(
        "Could not construct a severe error/cancellation trajectory: "
        + "; ".join(failures[-8:])
    )


def _trajectory_tokens(tokenizer, steps: Iterable[str]) -> int:
    return sum(
        len(tokenizer.encode(step, add_special_tokens=False)) + 1 for step in steps
    )


def validate_variants(
    variants: dict[str, Any],
    *,
    answer: str,
    tokenizer,
    min_ratio: float,
    max_ratio: float,
    max_pair_delta: float,
    enforce_error_clean_ratio: bool = True,
    enforce_redundant_clean_ratio: bool = True,
    enforce_pair_delta: bool = True,
) -> dict[str, Any]:
    errors: list[str] = []
    if set(variants) != set(VARIANT_TYPES):
        return {"accepted": False, "errors": ["variant_set"]}
    clean_steps = tuple(variants["clean"]["steps"])
    clean_count = _trajectory_tokens(tokenizer, clean_steps)
    metrics: dict[str, Any] = {}
    for name, expected_types in VARIANT_TYPES.items():
        steps = tuple(variants[name].get("steps", ()))
        types = tuple(variants[name].get("types", ()))
        if len(steps) != 5 or types != expected_types:
            errors.append(f"{name}:shape_or_types")
            continue
        parsed: list[Equation] = []
        try:
            parsed = [parse_equation(step) for step in steps]
        except (SyntaxError, ValueError, ZeroDivisionError):
            errors.append(f"{name}:format_or_parse")
            continue
        if parsed[-1].result_value != evaluate_arithmetic(answer):
            errors.append(f"{name}:answer")
        for index, (equation, step_type) in enumerate(zip(parsed, types)):
            expected_true = step_type not in {"DIRECT_FALSE", "CANCEL_FALSE"}
            if equation.is_true != expected_true:
                errors.append(f"{name}:{index}:truth_label")
            if step_type == "REDUNDANT" and "+0" not in equation.expression:
                errors.append(f"{name}:{index}:neutral_operation")
        if name == "local_error" and tuple(steps[3:]) != clean_steps[3:]:
            errors.append("local_error:post_cancel_not_clean")
        if name == "wide_error" and steps[4] != clean_steps[4]:
            errors.append("wide_error:post_cancel_not_clean")
        count = _trajectory_tokens(tokenizer, steps)
        ratio = count / clean_count
        metrics[name] = {"token_count": count, "ratio_to_clean": ratio}
        is_error = name.endswith("_error")
        is_redundant = name.endswith("_redundant")
        if (
            (is_error and enforce_error_clean_ratio)
            or (is_redundant and enforce_redundant_clean_ratio)
        ) and not min_ratio <= ratio <= max_ratio:
            errors.append(f"{name}:token_ratio")
    for scope in ("local", "wide"):
        error_count = metrics.get(f"{scope}_error", {}).get("token_count")
        redundant_count = metrics.get(f"{scope}_redundant", {}).get("token_count")
        if error_count and redundant_count:
            delta = abs(error_count - redundant_count) / max(error_count, redundant_count)
            metrics[f"{scope}_pair_delta"] = delta
            if enforce_pair_delta and delta > max_pair_delta:
                errors.append(f"{scope}:pair_token_delta")
    return {"accepted": not errors, "errors": sorted(set(errors)), "metrics": metrics}


def _question_id(question: str) -> str:
    return sha256(question.strip().encode("utf-8")).hexdigest()


def _immutable_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise FileExistsError(f"Refusing to overwrite frozen artifact: {path}")
        return
    atomic_json(path, payload)


def prepare_error_cancellation_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    for path_key, hash_key in (
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    ):
        path = (root / config[path_key]).resolve()
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Artifact missing or SHA-256 mismatch: {path}")

    tokenizer, _ = build_tokenizer((root / config["base_model_dir"]).resolve())
    test_examples = list(iter_icot_examples(root / config["gsm_test_path"]))
    if len(test_examples) != int(config["test_examples"]):
        raise ValueError("Frozen GSM8K test count mismatch")
    test_entries = [
        {
            "problem_id": _question_id(example.question),
            "problem": example.question,
            "answer": example.answer,
        }
        for example in test_examples
    ]
    test_ids = {row["problem_id"] for row in test_entries}

    candidates: list[tuple[str, OfficialExample]] = []
    seen: set[str] = set()
    for example in iter_icot_examples(root / config["gsm_train_path"]):
        question_id = _question_id(example.question)
        if len(example.steps) != 5 or question_id in seen or question_id in test_ids:
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

    accepted: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    for question_id, example in candidates:
        best: tuple[dict[str, Any], dict[str, Any], int] | None = None
        for delta in config.get("delta_candidates", (1, -1, 2, -2, 5, -5, 10, -10)):
            try:
                variants = construct_variants(
                    example,
                    delta=int(delta),
                    neutralize_errors=bool(config.get("neutralize_errors", True)),
                )
                validation = validate_variants(
                    variants,
                    answer=example.answer,
                    tokenizer=tokenizer,
                    min_ratio=float(config["min_aux_token_ratio"]),
                    max_ratio=float(config["max_aux_token_ratio"]),
                    max_pair_delta=float(config["max_pair_token_delta"]),
                    enforce_error_clean_ratio=bool(
                        config.get("enforce_error_clean_token_ratio", True)
                    ),
                    enforce_redundant_clean_ratio=bool(
                        config.get("enforce_redundant_clean_token_ratio", True)
                    ),
                    enforce_pair_delta=bool(
                        config.get("enforce_error_redundant_pair_token_delta", True)
                    ),
                )
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as error:
                rejections[f"construction:{type(error).__name__}"] += 1
                continue
            if validation["accepted"]:
                best = variants, validation, int(delta)
                break
            rejections.update(validation["errors"])
        if best is None:
            continue
        variants, validation, delta = best
        row = {
            "question_id": question_id,
            "problem": example.question,
            "answer": example.answer,
            "source": {
                "kind": "official_gsm8k_exact_five_step_verbatim",
                "source_file": config["gsm_train_path"],
                "source_line": example.idx,
            },
            "delta": delta,
            "variants": variants,
            "validation": validation,
        }
        row["five_tuple_sha256"] = canonical_hash(row)
        accepted.append(row)
        if len(accepted) == int(config["train_examples"]):
            break

    audit = {
        "schema_version": 1,
        "status": (
            "PASS"
            if len(accepted) == int(config["train_examples"])
            else "CONSTRUCTION_INCOMPLETE"
        ),
        "required_five_tuples": int(config["train_examples"]),
        "selected_five_tuples": len(accepted),
        "strict_redundancy_policy": "per_corresponding_affected_slot_+0",
        "neutralize_errors": bool(config.get("neutralize_errors", True)),
        "enforce_error_clean_token_ratio": bool(
            config.get("enforce_error_clean_token_ratio", True)
        ),
        "enforce_redundant_clean_token_ratio": bool(
            config.get("enforce_redundant_clean_token_ratio", True)
        ),
        "enforce_error_redundant_pair_token_delta": bool(
            config.get("enforce_error_redundant_pair_token_delta", True)
        ),
        "min_aux_token_ratio": float(config["min_aux_token_ratio"]),
        "max_aux_token_ratio": float(config["max_aux_token_ratio"]),
        "max_pair_token_delta": float(config["max_pair_token_delta"]),
        "rejection_counts": dict(rejections),
        "official_test_opened_previously": True,
    }
    audit_path = (root / config["data_audit_path"]).resolve()
    _immutable_json(audit_path, audit)
    if audit["status"] != "PASS":
        raise RuntimeError(
            "Could not construct 512 strict error-cancellation five-tuples; "
            f"accepted={len(accepted)}. See {audit_path}"
        )

    mask_order = sorted(
        (row["question_id"] for row in accepted),
        key=lambda value: sha256(
            f"{config['selection_domain']}:coverage\0{value}".encode("utf-8")
        ).hexdigest(),
    )
    mask25 = mask_order[: int(config["coverage_25_examples"])]
    mask50 = mask_order[: int(config["coverage_50_examples"])]
    manifest = {
        "schema_version": 1,
        "dataset_family": "gsm8k",
        "selection_domain": config["selection_domain"],
        "entries": accepted,
        "test_problem_ids": sorted(test_ids),
        "test_entries": test_entries,
        "coverage_masks": {"25": mask25, "50": mask50},
        "generator_disclosure": (
            "Deterministic controlled semi-synthetic conflicts authored in the "
            "current Codex task; not natural teacher noise."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    _immutable_json((root / config["manifest_path"]).resolve(), manifest)
    audit_count = int(config.get("audit_examples", 20))
    audit_rows = sorted(
        accepted,
        key=lambda row: sha256(
            f"{config['selection_domain']}:audit\0{row['question_id']}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )[:audit_count]
    audit_bundle = {
        "schema_version": 1,
        "status": "READY_FOR_CODEX_SEMANTIC_AUDIT",
        "manifest_sha256": manifest["manifest_sha256"],
        "reviewed_question_ids": [row["question_id"] for row in audit_rows],
        "examples": [
            {
                "question_id": row["question_id"],
                "problem": row["problem"],
                "answer": row["answer"],
                "source": row["source"],
                "delta": row["delta"],
                "variants": row["variants"],
                "validation": row["validation"],
            }
            for row in audit_rows
        ],
    }
    _immutable_json((root / config["audit_bundle_path"]).resolve(), audit_bundle)
    return audit


def prepare_severe_error_cancellation_data(
    config: dict[str, Any], *, project_root: str | Path
) -> dict[str, Any]:
    """Derive v12 from the frozen v10 units while changing only error severity."""
    root = Path(project_root).resolve()
    for path_key, hash_key in (
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
        ("checkpoint_path", "checkpoint_sha256"),
    ):
        path = (root / config[path_key]).resolve()
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Artifact missing or SHA-256 mismatch: {path}")

    parent_path = (root / config["source_parent_manifest_path"]).resolve()
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    verify_frozen_manifest(
        parent,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    if parent["manifest_sha256"] != config["source_parent_manifest_sha256"]:
        raise ValueError("Source parent manifest SHA-256 mismatch")

    tokenizer, _ = build_tokenizer((root / config["base_model_dir"]).resolve())
    entries: list[dict[str, Any]] = []
    delta_abs_values: list[int] = []
    local_deviations: list[float] = []
    wide_deviations: list[list[float]] = []
    positive = 0
    for parent_row in parent["entries"]:
        clean_steps = tuple(parent_row["variants"]["clean"]["steps"])
        example = OfficialExample(
            idx=int(parent_row["source"]["source_line"]),
            question=parent_row["problem"],
            steps=clean_steps,
            answer=parent_row["answer"],
        )
        variants, delta, severity = construct_severe_variants(
            example,
            question_id=parent_row["question_id"],
            severity_multiplier=int(config["severity_multiplier"]),
            severity_floor=int(config["severity_floor"]),
            downstream_min_relative=float(config["downstream_min_relative"]),
        )
        for unchanged_name in ("clean", "local_redundant", "wide_redundant"):
            if variants[unchanged_name] != parent_row["variants"][unchanged_name]:
                raise ValueError(f"Control variant changed for {parent_row['question_id']}")
        validation = validate_variants(
            variants,
            answer=parent_row["answer"],
            tokenizer=tokenizer,
            min_ratio=float(config["min_aux_token_ratio"]),
            max_ratio=float(config["max_aux_token_ratio"]),
            max_pair_delta=float(config["max_pair_token_delta"]),
            enforce_error_clean_ratio=bool(
                config.get("enforce_error_clean_token_ratio", False)
            ),
            enforce_redundant_clean_ratio=bool(
                config.get("enforce_redundant_clean_token_ratio", False)
            ),
            enforce_pair_delta=bool(
                config.get("enforce_error_redundant_pair_token_delta", False)
            ),
        )
        if not validation["accepted"] or not severity["final_answer_preserved"]:
            raise ValueError(
                f"Severe variant failed validation for {parent_row['question_id']}: "
                f"{validation['errors']}"
            )
        row = {
            "question_id": parent_row["question_id"],
            "problem": parent_row["problem"],
            "answer": parent_row["answer"],
            "source": parent_row["source"],
            "parent_delta": parent_row["delta"],
            "delta": delta,
            "severity": severity,
            "variants": variants,
            "validation": validation,
        }
        row["five_tuple_sha256"] = canonical_hash(row)
        entries.append(row)
        delta_abs_values.append(abs(delta))
        local_deviations.append(severity["local_direct_relative_deviation"])
        wide_deviations.append(severity["wide_relative_deviations"])
        positive += int(delta > 0)

    parent_ids = [row["question_id"] for row in parent["entries"]]
    new_ids = [row["question_id"] for row in entries]
    if new_ids != parent_ids:
        raise ValueError("Training unit order changed relative to the source parent")
    manifest = {
        "schema_version": 1,
        "dataset_family": "gsm8k",
        "selection_domain": config["selection_domain"],
        "source_parent_manifest_sha256": parent["manifest_sha256"],
        "entries": entries,
        "test_problem_ids": parent["test_problem_ids"],
        "test_entries": parent["test_entries"],
        "coverage_masks": parent["coverage_masks"],
        "generator_disclosure": (
            "Deterministic scale-amplified controlled semi-synthetic conflicts "
            "authored in the current Codex task; not natural teacher noise."
        ),
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    verify_frozen_manifest(
        manifest,
        expected_train=int(config["train_examples"]),
        expected_test=int(config["test_examples"]),
    )
    _immutable_json((root / config["manifest_path"]).resolve(), manifest)

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "required_five_tuples": int(config["train_examples"]),
        "selected_five_tuples": len(entries),
        "source_parent_manifest_sha256": parent["manifest_sha256"],
        "same_question_ids_and_order": new_ids == parent_ids,
        "same_answers_problems_sources": all(
            all(row[key] == source[key] for key in ("answer", "problem", "source"))
            for row, source in zip(entries, parent["entries"])
        ),
        "same_coverage_masks": manifest["coverage_masks"] == parent["coverage_masks"],
        "same_clean_and_redundant_controls": all(
            all(
                row["variants"][name] == source["variants"][name]
                for name in ("clean", "local_redundant", "wide_redundant")
            )
            for row, source in zip(entries, parent["entries"])
        ),
        "severity_multiplier": int(config["severity_multiplier"]),
        "severity_floor": int(config["severity_floor"]),
        "downstream_min_relative": float(config["downstream_min_relative"]),
        "delta_abs_min": min(delta_abs_values),
        "delta_abs_median": median(delta_abs_values),
        "delta_abs_max": max(delta_abs_values),
        "positive_delta_count": positive,
        "negative_delta_count": len(entries) - positive,
        "local_direct_relative_min": min(local_deviations),
        "wide_relative_min_per_propagated_slot": [
            min(values[index] for values in wide_deviations) for index in range(3)
        ],
        "all_final_answers_preserved": all(
            row["severity"]["final_answer_preserved"] for row in entries
        ),
        "enforce_error_clean_token_ratio": bool(
            config.get("enforce_error_clean_token_ratio", False)
        ),
        "enforce_redundant_clean_token_ratio": bool(
            config.get("enforce_redundant_clean_token_ratio", False)
        ),
        "enforce_error_redundant_pair_token_delta": bool(
            config.get("enforce_error_redundant_pair_token_delta", False)
        ),
    }
    if not all(
        audit[key]
        for key in (
            "same_question_ids_and_order",
            "same_answers_problems_sources",
            "same_coverage_masks",
            "same_clean_and_redundant_controls",
            "all_final_answers_preserved",
        )
    ):
        audit["status"] = "FAIL"
    _immutable_json((root / config["data_audit_path"]).resolve(), audit)
    if audit["status"] != "PASS":
        raise RuntimeError("Severe conflict control audit failed")

    audit_count = int(config.get("audit_examples", 20))
    audit_rows = sorted(
        entries,
        key=lambda row: sha256(
            f"{config['selection_domain']}:audit\0{row['question_id']}".encode("utf-8")
        ).hexdigest(),
    )[:audit_count]
    audit_bundle = {
        "schema_version": 1,
        "status": "READY_FOR_CODEX_SEMANTIC_AUDIT",
        "manifest_sha256": manifest["manifest_sha256"],
        "reviewed_question_ids": [row["question_id"] for row in audit_rows],
        "examples": audit_rows,
    }
    _immutable_json((root / config["audit_bundle_path"]).resolve(), audit_bundle)
    return audit
