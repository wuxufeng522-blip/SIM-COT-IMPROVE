from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Any, Iterable
import json
import math
import re

from .audit import evaluate_arithmetic
from .causal_corruptions import (
    _expression_number_tokens,
    _question_values,
    extract_dependency_edges,
    parse_all_equations,
)
from .official_adapter import OfficialExample
from .single_gpu_smoke import encode_smoke_example


LEAKAGE_TERMS = (
    "错误",
    "反事实",
    "故意错",
    "incorrect",
    "wrong reasoning",
    "counterfactual",
    "corrupt",
)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    rejection_codes: tuple[str, ...]
    normalized_edit_distance: float | None
    auxiliary_token_ratio: float | None
    clean_auxiliary_tokens: int | None
    candidate_auxiliary_tokens: int | None
    dependency_edges: tuple[tuple[int, int], ...]
    final_ancestors: tuple[int, ...]
    candidate_sha256: str

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def normalized_levenshtein(left: list[int], right: list[int]) -> float:
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for row, right_item in enumerate(right, start=1):
        current = [row]
        for column, left_item in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1] / max(len(left), len(right))


def _answer_value(answer: str) -> Fraction | None:
    cleaned = answer.replace(",", "").strip()
    try:
        return evaluate_arithmetic(cleaned)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        match = re.search(r"[+-]?\d+(?:\.\d+)?", cleaned)
        if match is None:
            return None
        try:
            return evaluate_arithmetic(match.group(0))
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None


def _final_ancestors(edges: Iterable[tuple[int, int]], final: int = 4) -> set[int]:
    parents: dict[int, set[int]] = {}
    for parent, child in edges:
        parents.setdefault(child, set()).add(parent)
    ancestors: set[int] = set()
    frontier = list(parents.get(final, ()))
    while frontier:
        node = frontier.pop()
        if node in ancestors:
            continue
        ancestors.add(node)
        frontier.extend(parents.get(node, ()))
    return ancestors


def validate_full_conflict_candidate(
    clean: OfficialExample,
    candidate: dict[str, Any],
    *,
    tokenizer,
    token_ids: dict[str, int] | None,
    config: dict[str, Any],
    check_context: bool = True,
) -> ValidationResult:
    codes: list[str] = []
    raw_steps = candidate.get("steps")
    steps = tuple(raw_steps) if isinstance(raw_steps, list) else ()
    if len(steps) != 5 or any(not isinstance(step, str) for step in steps):
        codes.append("step_count")
    if candidate.get("question_id") not in {None, candidate.get("expected_question_id")}:
        # The caller normally injects expected_question_id.  This branch catches a
        # self-inconsistent raw record without coupling the validator to a manifest.
        codes.append("question_id_mismatch")
    if candidate.get("question") not in {None, clean.question}:
        codes.append("question_changed")
    if candidate.get("answer") not in {None, clean.answer}:
        codes.append("answer_changed")

    clean_equations = parse_all_equations(clean)
    candidate_example = OfficialExample(clean.idx, clean.question, steps, clean.answer)
    equations = parse_all_equations(candidate_example) if len(steps) == 5 else None
    if clean_equations is None:
        codes.append("clean_parse_failure")
    if equations is None:
        codes.append("parse_failure")

    edge_pairs: tuple[tuple[int, int], ...] = ()
    ancestors: set[int] = set()
    if equations is not None and clean_equations is not None:
        for index, (step, clean_step, equation, clean_equation) in enumerate(
            zip(steps, clean.steps, equations, clean_equations, strict=True)
        ):
            if step.strip() == clean_step.strip():
                codes.append(f"clean_step_collision:{index}")
            if equation.rhs_value == clean_equation.rhs_value:
                codes.append(f"clean_result_collision:{index}")
            if not math.isfinite(float(equation.rhs_value)):
                codes.append(f"non_finite_result:{index}")

        edges = extract_dependency_edges(candidate_example, equations)
        edge_pairs = tuple((edge.parent, edge.child) for edge in edges)
        ancestors = _final_ancestors(edge_pairs)
        if ancestors != {0, 1, 2, 3}:
            codes.append("disconnected_chain")

        question_values = _question_values(clean.question)
        prior_results: set[Fraction] = set()
        for index, equation in enumerate(equations):
            for _, _, _, value in _expression_number_tokens(equation.lhs):
                if value not in question_values and value not in prior_results:
                    codes.append(f"unmotivated_constant:{index}")
                    break
            prior_results.add(equation.rhs_value)

        official = _answer_value(clean.answer)
        if official is not None and equations[-1].rhs_value == official:
            codes.append("official_answer_collision")
        declared = candidate.get("wrong_final_result")
        if declared is not None:
            declared_value = _answer_value(str(declared))
            if declared_value is None or declared_value != equations[-1].rhs_value:
                codes.append("wrong_final_declaration_mismatch")

    joined = " ".join(steps)
    lowered = joined.casefold()
    if any(term.casefold() in lowered for term in LEAKAGE_TERMS):
        codes.append("identity_leakage")

    clean_token_ids = tokenizer.encode(" ".join(clean.steps), add_special_tokens=False)
    candidate_token_ids = tokenizer.encode(joined, add_special_tokens=False)
    distance = normalized_levenshtein(clean_token_ids, candidate_token_ids)
    ratio = (
        len(candidate_token_ids) / len(clean_token_ids) if clean_token_ids else None
    )
    if distance < float(config["min_normalized_edit_distance"]):
        codes.append("token_conflict_too_low")
    if ratio is None or not (
        float(config["min_aux_token_ratio"])
        <= ratio
        <= float(config["max_aux_token_ratio"])
    ):
        codes.append("length_mismatch")

    if check_context and len(steps) == 5:
        if token_ids is None:
            raise ValueError("token_ids are required when check_context=True")
        encoded = encode_smoke_example(
            candidate_example,
            tokenizer,
            token_ids,
            latent_stage=int(config["latent_stage"]),
            c_thought=int(config["c_thought"]),
        )
        maximum = int(config["max_sequence_tokens"])
        if len(encoded.input_ids) > maximum or encoded.maximum_auxiliary_length > maximum:
            codes.append("context_length")

    unique_codes = tuple(dict.fromkeys(codes))
    canonical = json.dumps(candidate, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return ValidationResult(
        accepted=not unique_codes,
        rejection_codes=unique_codes,
        normalized_edit_distance=distance,
        auxiliary_token_ratio=ratio,
        clean_auxiliary_tokens=len(clean_token_ids),
        candidate_auxiliary_tokens=len(candidate_token_ids),
        dependency_edges=edge_pairs,
        final_ancestors=tuple(sorted(ancestors)),
        candidate_sha256=sha256(canonical).hexdigest(),
    )
