from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any, Literal
import json


SplitName = Literal["head_train", "head_validation", "head_audit"]


@dataclass(frozen=True)
class ReliabilityRow:
    variant_id: str
    question_id: str
    source_index: int
    split: SplitName
    question: str
    answer: str
    step_index: int
    trajectory_steps: int
    prefix_steps: tuple[str, ...]
    clean_step: str
    candidate_step: str
    family: str
    template_id: str
    pair_id: str
    y_valid: int
    y_utility: int | None
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.split not in {"head_train", "head_validation", "head_audit"}:
            raise ValueError(f"Unknown split: {self.split}")
        if self.y_valid not in {0, 1}:
            raise ValueError("y_valid must be 0 or 1")
        if self.y_valid == 0 and self.y_utility is not None:
            raise ValueError("Utility must be undefined for invalid steps")
        if self.y_valid == 1 and self.y_utility not in {0, 1}:
            raise ValueError("Utility must be binary for valid steps")
        if self.step_index < 0 or self.step_index >= self.trajectory_steps:
            raise ValueError("step_index must fall inside the trajectory")
        if len(self.prefix_steps) != self.step_index:
            raise ValueError("prefix_steps must end immediately before candidate_step")
        if not self.family or not self.template_id or not self.pair_id:
            raise ValueError("family, template_id, and pair_id are required")

    @property
    def y_reliable(self) -> int:
        return int(self.y_valid == 1 and self.y_utility == 1)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["prefix_steps"] = list(self.prefix_steps)
        payload["y_reliable"] = self.y_reliable
        return payload


def stable_variant_id(
    *,
    question_id: str,
    step_index: int,
    family: str,
    template_id: str,
    candidate_step: str,
) -> str:
    payload = json.dumps(
        {
            "question_id": question_id,
            "step_index": step_index,
            "family": family,
            "template_id": template_id,
            "candidate_step": candidate_step,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()
