from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import re

from .audit import evaluate_arithmetic, numeric_result_close


DEVELOPMENT_FAMILIES = (
    "numeric_error",
    "operator_relation_error",
    "dependency_order_error",
    "irrelevant_but_correct",
    "redundant_repeat",
)
SEALED_FAMILY = "compensating_error"

_NUMBER = re.compile(r"[+-]?\d+(?:\.\d+)?")
_BINARY_OPERATOR = re.compile(r"(?<=[\d)])\s*([+\-*/])\s*(?=[+\-]?\d|\()")


@dataclass(frozen=True)
class Equation:
    lhs: str
    rhs: str
    lhs_value: Fraction
    rhs_value: Fraction


@dataclass(frozen=True)
class CorruptionVariant:
    family: str
    template_id: str
    text: str
    y_valid: int
    y_utility: int | None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.y_valid == 0 and self.y_utility is not None:
            raise ValueError("Invalid corruptions must not define Utility")
        if self.y_valid == 1 and self.y_utility not in {0, 1}:
            raise ValueError("Valid corruptions require a binary Utility label")


def parse_checked_equation(step: str) -> Equation | None:
    if not step.startswith("<<") or not step.endswith(">>"):
        return None
    content = step[2:-2].strip()
    if content.count("=") != 1:
        return None
    lhs, rhs = (part.strip() for part in content.split("=", 1))
    try:
        lhs_value = evaluate_arithmetic(lhs)
        rhs_value = evaluate_arithmetic(rhs)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None
    if not numeric_result_close(lhs_value, rhs_value, rhs):
        return None
    return Equation(lhs=lhs, rhs=rhs, lhs_value=lhs_value, rhs_value=rhs_value)


def _format_fraction(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    rendered = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return rendered if rendered not in {"", "-0"} else "0"


def equivalent_variant(step: str) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    return CorruptionVariant(
        family="equivalent_positive",
        template_id="reverse_equality_v1",
        text=f"<<{equation.rhs} = {equation.lhs}>>",
        y_valid=1,
        y_utility=1,
        metadata={"transformation": "symmetric_equality"},
    )


def numeric_error_variant(step: str) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    delta = Fraction(1 if equation.rhs_value >= 0 else -1)
    corrupted_rhs = _format_fraction(equation.rhs_value + delta)
    return CorruptionVariant(
        family="numeric_error",
        template_id="rhs_offset_unit_v1",
        text=f"<<{equation.lhs}={corrupted_rhs}>>",
        y_valid=0,
        y_utility=None,
        metadata={"delta": _format_fraction(delta)},
    )


def operator_error_variant(step: str) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    replacements = {"+": "-", "-": "+", "*": "/", "/": "*"}
    for match in _BINARY_OPERATOR.finditer(equation.lhs):
        old = match.group(1)
        new = replacements[old]
        changed = equation.lhs[: match.start(1)] + new + equation.lhs[match.end(1) :]
        try:
            changed_value = evaluate_arithmetic(changed)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        if numeric_result_close(changed_value, equation.rhs_value, equation.rhs):
            continue
        return CorruptionVariant(
            family="operator_relation_error",
            template_id="binary_operator_swap_v1",
            text=f"<<{changed}={equation.rhs}>>",
            y_valid=0,
            y_utility=None,
            metadata={"old_operator": old, "new_operator": new},
        )
    return None


def dependency_error_variant(
    step: str,
    *,
    later_steps: tuple[str, ...],
    step_index: int,
) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    for offset, later_step in enumerate(later_steps, start=1):
        later_equation = parse_checked_equation(later_step)
        if later_equation is None:
            continue
        future_value = _format_fraction(later_equation.rhs_value)
        for match in _NUMBER.finditer(equation.lhs):
            if match.group(0).replace("+", "") == future_value.replace("+", ""):
                continue
            changed = (
                equation.lhs[: match.start()]
                + future_value
                + equation.lhs[match.end() :]
            )
            try:
                changed_value = evaluate_arithmetic(changed)
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                continue
            if numeric_result_close(changed_value, equation.rhs_value, equation.rhs):
                continue
            return CorruptionVariant(
                family="dependency_order_error",
                template_id="future_result_substitution_v1",
                text=f"<<{changed}={equation.rhs}>>",
                y_valid=0,
                y_utility=None,
                metadata={
                    "future_step_index": step_index + offset,
                    "future_value": future_value,
                    "replaced_value": match.group(0),
                },
            )
    return None


def irrelevant_correct_variant(step: str) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    match = _NUMBER.search(equation.lhs)
    if match is None:
        return None
    number = match.group(0)
    atom = f"({number})" if number.startswith(("+", "-")) else number
    return CorruptionVariant(
        family="irrelevant_but_correct",
        template_id="self_subtraction_zero_v1",
        text=f"<<{atom}-{atom}=0>>",
        y_valid=1,
        y_utility=0,
        metadata={"matched_number": number},
    )


def redundant_variant(
    step: str,
    *,
    prefix_steps: tuple[str, ...],
) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    if prefix_steps:
        text = prefix_steps[-1]
        template_id = "repeat_previous_step_v1"
    else:
        rhs = equation.rhs
        atom = f"({rhs})" if rhs.startswith(("+", "-")) else rhs
        text = f"<<{atom}+0={rhs}>>"
        template_id = "identity_without_progress_v1"
    if parse_checked_equation(text) is None:
        return None
    return CorruptionVariant(
        family="redundant_repeat",
        template_id=template_id,
        text=text,
        y_valid=1,
        y_utility=0,
        metadata={"uses_previous_step": bool(prefix_steps)},
    )


def development_variants(
    step: str,
    *,
    prefix_steps: tuple[str, ...],
    later_steps: tuple[str, ...],
    step_index: int,
) -> tuple[CorruptionVariant, ...]:
    candidates = (
        numeric_error_variant(step),
        operator_error_variant(step),
        dependency_error_variant(
            step,
            later_steps=later_steps,
            step_index=step_index,
        ),
        irrelevant_correct_variant(step),
        redundant_variant(step, prefix_steps=prefix_steps),
    )
    variants = tuple(candidate for candidate in candidates if candidate is not None)
    families = [variant.family for variant in variants]
    if len(families) != len(set(families)):
        raise AssertionError("At most one variant per development family is allowed")
    return variants


def compensating_error_variant(step: str) -> CorruptionVariant | None:
    equation = parse_checked_equation(step)
    if equation is None:
        return None
    delta = Fraction(1 if equation.rhs_value >= 0 else -1)
    wrong = equation.rhs_value + delta
    wrong_text = _format_fraction(wrong)
    correction_operator = "-" if delta > 0 else "+"
    return CorruptionVariant(
        family=SEALED_FAMILY,
        template_id="wrong_then_cancel_unit_v1",
        text=(
            f"<<{equation.lhs}={wrong_text};"
            f"{wrong_text}{correction_operator}1={equation.rhs}>>"
        ),
        y_valid=0,
        y_utility=None,
        metadata={
            "intermediate_error": _format_fraction(delta),
            "final_numeric_result": equation.rhs,
            "final_result_preserved": True,
        },
    )
