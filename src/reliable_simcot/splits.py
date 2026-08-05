from __future__ import annotations

from collections import Counter
from hashlib import sha256
from typing import Iterable
import json
import random

from .labels import SplitName


def assign_question_splits(
    question_ids: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, SplitName]:
    unique = sorted(set(question_ids))
    if not unique:
        raise ValueError("At least one question ID is required")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("Split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must leave an audit split")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    validation_end = train_end + int(len(shuffled) * validation_fraction)
    if len(shuffled) >= 3:
        train_end = max(1, min(train_end, len(shuffled) - 2))
        validation_end = max(train_end + 1, min(validation_end, len(shuffled) - 1))

    mapping: dict[str, SplitName] = {}
    for index, question_id in enumerate(shuffled):
        if index < train_end:
            mapping[question_id] = "head_train"
        elif index < validation_end:
            mapping[question_id] = "head_validation"
        else:
            mapping[question_id] = "head_audit"
    return mapping


def validate_question_isolation(rows: Iterable[dict]) -> dict[str, int]:
    question_splits: dict[str, set[str]] = {}
    for row in rows:
        question_splits.setdefault(row["question_id"], set()).add(row["split"])
    leaked = {
        question_id: splits
        for question_id, splits in question_splits.items()
        if len(splits) != 1
    }
    if leaked:
        raise ValueError(f"Question variants cross splits: {sorted(leaked)[:5]}")
    return dict(Counter(next(iter(splits)) for splits in question_splits.values()))


def split_manifest_sha256(mapping: dict[str, SplitName]) -> str:
    payload = json.dumps(
        mapping,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()
