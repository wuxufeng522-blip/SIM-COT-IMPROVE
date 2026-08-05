from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
import json
import math
import re


NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")


@dataclass(frozen=True)
class OODExample:
    idx: int
    question: str
    answer: float


def extract_answer_number_official(sentence: str) -> float:
    """Match the numerical-answer extraction in the released CODI/test.py."""
    matches = NUMBER_PATTERN.findall(sentence.replace(",", ""))
    return float(matches[-1]) if matches else float("inf")


def normalize_question_official(value: str) -> str:
    return value.strip().replace("  ", " ")


def normalize_ground_truth(value: Any) -> float:
    text = str(value)
    if "####" in text:
        text = text.split("####")[-1]
    return float(text.replace(",", ""))


def _load_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return data


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected a JSON object per line: {path}")
                yield row


def load_ood_examples(dataset: str, paths: list[str | Path]) -> list[OODExample]:
    source_paths = [Path(path) for path in paths]
    if dataset == "gsm-hard":
        if len(source_paths) != 1:
            raise ValueError("GSM-Hard requires one JSONL path")
        raw_rows = list(_iter_jsonl(source_paths[0]))
        question_key, answer_key = "input", "target"
    elif dataset == "multi-arith":
        if len(source_paths) not in (1, 2):
            raise ValueError("MultiArith requires one split or both public splits")
        raw_rows = []
        for path in source_paths:
            raw_rows.extend(_load_json(path))
        question_key, answer_key = "question", "final_ans"
    elif dataset == "svamp":
        if len(source_paths) != 2:
            raise ValueError("SVAMP requires train and test paths in that order")
        raw_rows = _load_json(source_paths[0]) + _load_json(source_paths[1])
        question_key, answer_key = "question_concat", "Answer"
        for row in raw_rows:
            row[question_key] = f"{row['Body']} {row['Question']}"
    else:
        raise ValueError(f"Unsupported OOD dataset: {dataset}")

    examples = [
        OODExample(
            idx=idx,
            question=normalize_question_official(str(row[question_key])),
            answer=normalize_ground_truth(row[answer_key]),
        )
        for idx, row in enumerate(raw_rows)
    ]
    if any(not example.question or not math.isfinite(example.answer) for example in examples):
        raise ValueError(f"Empty question or non-finite answer in {dataset}")
    return examples
