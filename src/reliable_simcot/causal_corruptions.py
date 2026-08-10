from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from hashlib import sha256
from typing import Any
import json
import re

from .audit import evaluate_arithmetic, numeric_answer_close
from .corruptions import Equation, parse_checked_equation
from .official_adapter import OfficialExample


CAUSAL_FAMILIES = (
    "numeric_propagation",
    "operator_propagation",
    "quantity_propagation",
)
CAUSAL_LABELS = ("DIRECT_ERROR", "CAUSAL_DESCENDANT", "CLEAN")

_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
_BINARY_OPERATOR = re.compile(r"(?<=[\d)])\s*([+\-*/])\s*(?=[+\-]?\d|\()")


@dataclass(frozen=True)
class DependencyEdge:
    parent: int
    child: int
    start: int
    end: int
    clean_token: str
    clean_value: Fraction


@dataclass(frozen=True)
class CausalChain:
    family: str
    pivot: int
    affected_positions: tuple[int, int, int]
    corrupted_steps: tuple[str, ...]
    labels: tuple[str, ...]
    direct_wrong_value: Fraction
    propagated_final_value: Fraction
    dependency_edges: tuple[DependencyEdge, ...]
    full_changed_positions: tuple[int, ...]

    def to_record(self) -> dict[str, Any]:
        payload = {
            "family": self.family,
            "pivot": self.pivot,
            "affected_positions": list(self.affected_positions),
            "corrupted_steps": list(self.corrupted_steps),
            "labels": list(self.labels),
            "direct_wrong_value": _format_fraction(self.direct_wrong_value),
            "propagated_final_value": _format_fraction(self.propagated_final_value),
            "dependency_edges": [
                {
                    "parent": edge.parent,
                    "child": edge.child,
                    "clean_token": edge.clean_token,
                    "clean_value": _format_fraction(edge.clean_value),
                }
                for edge in self.dependency_edges
            ],
            "full_changed_positions": list(self.full_changed_positions),
        }
        payload["chain_sha256"] = sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return payload


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    rendered = f"{float(value):.8f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def _number_value(text: str) -> Fraction | None:
    try:
        return evaluate_arithmetic(text)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def _question_values(question: str) -> set[Fraction]:
    return {
        value
        for match in _NUMBER.finditer(question)
        if (value := _number_value(match.group(0))) is not None
    }


def _expression_number_tokens(expression: str):
    """Yield numeric spans without swallowing a preceding binary + or -."""
    for match in _NUMBER.finditer(expression):
        start = match.start()
        end = match.end()
        text = match.group(0)
        if text[:1] in {"+", "-"}:
            previous = start - 1
            while previous >= 0 and expression[previous].isspace():
                previous -= 1
            if previous >= 0 and (expression[previous].isdigit() or expression[previous] in ".)"):
                start += 1
                text = text[1:]
        value = _number_value(text)
        if value is not None:
            yield start, end, text, value


def parse_all_equations(example: OfficialExample) -> tuple[Equation, ...] | None:
    parsed = tuple(parse_checked_equation(step) for step in example.steps)
    if any(equation is None for equation in parsed):
        return None
    return tuple(equation for equation in parsed if equation is not None)


def extract_dependency_edges(
    example: OfficialExample, equations: tuple[Equation, ...]
) -> tuple[DependencyEdge, ...]:
    question_values = _question_values(example.question)
    edges: list[DependencyEdge] = []
    for child, equation in enumerate(equations):
        candidates: dict[int, list[DependencyEdge]] = {}
        for start, end, token, value in _expression_number_tokens(equation.lhs):
            if value in question_values:
                continue
            producers = [
                parent
                for parent in range(child)
                if equations[parent].rhs_value == value
            ]
            if len(producers) != 1:
                continue
            edge = DependencyEdge(
                parent=producers[0],
                child=child,
                start=start,
                end=end,
                clean_token=token,
                clean_value=value,
            )
            candidates.setdefault(producers[0], []).append(edge)
        for matches in candidates.values():
            if len(matches) == 1:
                edges.append(matches[0])
    return tuple(edges)


def three_node_paths(edges: tuple[DependencyEdge, ...]) -> tuple[tuple[int, int, int], ...]:
    edge_pairs = {(edge.parent, edge.child) for edge in edges}
    paths = [
        (pivot, middle, final)
        for pivot in range(3)
        for middle in range(pivot + 1, 5)
        for final in range(middle + 1, 5)
        if (pivot, middle) in edge_pairs and (middle, final) in edge_pairs
    ]
    return tuple(paths)


def _numeric_pivot(equation: Equation) -> tuple[str, Fraction]:
    delta = Fraction(1 if equation.rhs_value >= 0 else -1)
    wrong = equation.rhs_value + delta
    return f"<<{equation.lhs}={_format_fraction(wrong)}>>", wrong


def _operator_pivot(equation: Equation) -> tuple[str, Fraction] | None:
    replacements = {"+": "-", "-": "+", "*": "/", "/": "*"}
    for match in _BINARY_OPERATOR.finditer(equation.lhs):
        changed_lhs = (
            equation.lhs[: match.start(1)]
            + replacements[match.group(1)]
            + equation.lhs[match.end(1) :]
        )
        try:
            wrong = evaluate_arithmetic(changed_lhs)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        if wrong == equation.rhs_value:
            continue
        return f"<<{changed_lhs}={_format_fraction(wrong)}>>", wrong
    return None


def _quantity_pivot(
    example: OfficialExample,
    equation: Equation,
    *,
    prior_values: tuple[Fraction, ...],
) -> tuple[str, Fraction] | None:
    question_tokens = [match.group(0).lstrip("+") for match in _NUMBER.finditer(example.question)]
    question_values = _question_values(example.question)
    for start, end, raw_token, _ in _expression_number_tokens(equation.lhs):
        token = raw_token.lstrip("+")
        if not token.isdigit() or question_tokens.count(token) != 1:
            continue
        value = Fraction(int(token))
        if value <= 0 or value in prior_values:
            continue
        replacement = value - 1 if value > 1 else value + 1
        if replacement in question_values:
            continue
        changed_lhs = (
            equation.lhs[:start]
            + _format_fraction(replacement)
            + equation.lhs[end:]
        )
        try:
            wrong = evaluate_arithmetic(changed_lhs)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        if wrong == equation.rhs_value:
            continue
        return f"<<{changed_lhs}={_format_fraction(wrong)}>>", wrong
    return None


def _pivot_variant(
    family: str,
    example: OfficialExample,
    equations: tuple[Equation, ...],
    pivot: int,
) -> tuple[str, Fraction] | None:
    equation = equations[pivot]
    if family == "numeric_propagation":
        return _numeric_pivot(equation)
    if family == "operator_propagation":
        return _operator_pivot(equation)
    if family == "quantity_propagation":
        return _quantity_pivot(
            example,
            equation,
            prior_values=tuple(item.rhs_value for item in equations[:pivot]),
        )
    raise ValueError(f"Unknown causal corruption family: {family}")


def _replay_chain(
    example: OfficialExample,
    equations: tuple[Equation, ...],
    edges: tuple[DependencyEdge, ...],
    *,
    pivot: int,
    pivot_text: str,
    pivot_value: Fraction,
) -> tuple[tuple[str, ...], tuple[Fraction, ...], tuple[int, ...]] | None:
    new_steps = list(example.steps)
    new_values = [equation.rhs_value for equation in equations]
    changed = {pivot}
    new_steps[pivot] = pivot_text
    new_values[pivot] = pivot_value
    by_child: dict[int, list[DependencyEdge]] = {}
    for edge in edges:
        by_child.setdefault(edge.child, []).append(edge)

    for child in range(pivot + 1, len(equations)):
        replacements = [
            edge for edge in by_child.get(child, ()) if edge.parent in changed
        ]
        if not replacements:
            continue
        lhs = equations[child].lhs
        for edge in sorted(replacements, key=lambda item: item.start, reverse=True):
            lhs = lhs[: edge.start] + _format_fraction(new_values[edge.parent]) + lhs[edge.end :]
        try:
            value = evaluate_arithmetic(lhs)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            return None
        if value == equations[child].rhs_value:
            continue
        new_values[child] = value
        new_steps[child] = f"<<{lhs}={_format_fraction(value)}>>"
        changed.add(child)
    return tuple(new_steps), tuple(new_values), tuple(sorted(changed))


def build_causal_chain(
    example: OfficialExample,
    *,
    family: str,
    path: tuple[int, int, int],
) -> CausalChain | None:
    if family not in CAUSAL_FAMILIES or path[0] not in {0, 1, 2}:
        return None
    equations = parse_all_equations(example)
    if equations is None or len(equations) < 5:
        return None
    edges = extract_dependency_edges(example, equations)
    if path not in three_node_paths(edges):
        return None
    pivot_variant = _pivot_variant(family, example, equations, path[0])
    if pivot_variant is None:
        return None
    pivot_text, pivot_value = pivot_variant
    replay = _replay_chain(
        example,
        equations,
        edges,
        pivot=path[0],
        pivot_text=pivot_text,
        pivot_value=pivot_value,
    )
    if replay is None:
        return None
    new_steps, new_values, changed = replay
    if tuple(position for position in changed if position < 5) != path:
        return None
    if len(new_steps) != len(example.steps) or new_steps[-1] == example.steps[-1]:
        return None
    try:
        answer_value = evaluate_arithmetic(example.answer)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    if not numeric_answer_close(equations[-1].rhs_value, answer_value, example.answer):
        return None
    if numeric_answer_close(new_values[-1], answer_value, example.answer):
        return None
    labels = tuple(
        "DIRECT_ERROR" if position == path[0]
        else "CAUSAL_DESCENDANT" if position in path[1:]
        else "CLEAN"
        for position in range(5)
    )
    path_edges: list[DependencyEdge] = []
    for parent, child in zip(path, path[1:]):
        matching = [
            edge for edge in edges if edge.parent == parent and edge.child == child
        ]
        if len(matching) != 1:
            return None
        path_edges.append(matching[0])
    return CausalChain(
        family=family,
        pivot=path[0],
        affected_positions=path,
        corrupted_steps=tuple(new_steps[:5]),
        labels=labels,
        direct_wrong_value=pivot_value,
        propagated_final_value=new_values[-1],
        dependency_edges=tuple(path_edges),
        full_changed_positions=changed,
    )


def causal_chains_by_cell(example: OfficialExample) -> dict[tuple[str, int], CausalChain]:
    equations = parse_all_equations(example)
    if equations is None or len(equations) < 5:
        return {}
    paths = three_node_paths(extract_dependency_edges(example, equations))
    result: dict[tuple[str, int], CausalChain] = {}
    for family in CAUSAL_FAMILIES:
        for path in paths:
            key = (family, path[0])
            if key in result:
                continue
            chain = build_causal_chain(example, family=family, path=path)
            if chain is not None:
                result[key] = chain
    return result
