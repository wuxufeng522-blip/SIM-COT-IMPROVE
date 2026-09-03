from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable
import json
import random
import re

from .config import DataConfig


@dataclass
class Example:
    example_id: str
    question: str
    clean_steps: list[str]
    observed_steps: list[str]
    answer: str
    is_noisy: bool = False
    noisy_step_index: int | None = None
    noise_type: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "Example":
        return cls(**value)


def _choose_operation(rng: random.Random, current: int) -> tuple[str, int, int]:
    candidates: list[tuple[str, int, int]] = []

    addend = rng.randint(2, 30)
    if current + addend <= 600:
        candidates.append(("+", addend, current + addend))

    if current > 4:
        subtrahend = rng.randint(1, min(30, current - 1))
        candidates.append(("-", subtrahend, current - subtrahend))

    factor = rng.randint(2, 5)
    if current * factor <= 600:
        candidates.append(("*", factor, current * factor))

    if not candidates:
        return "-", max(1, current // 3), current - max(1, current // 3)
    return rng.choice(candidates)


def _operation_phrase(operator: str, operand: int) -> str:
    if operator == "+":
        return f"adds {operand}"
    if operator == "-":
        return f"subtracts {operand}"
    return f"multiplies by {operand}"


def _make_clean_example(
    rng: random.Random,
    example_id: str,
    min_steps: int,
    max_steps: int,
) -> Example:
    start = rng.randint(2, 40)
    current = start
    steps: list[str] = []
    phrases: list[str] = []
    n_steps = rng.randint(min_steps, max_steps)

    for step_number in range(1, n_steps + 1):
        before = current
        operator, operand, current = _choose_operation(rng, current)
        phrases.append(_operation_phrase(operator, operand))
        display_operator = "×" if operator == "*" else operator
        steps.append(
            f"Step {step_number}: {before} {display_operator} {operand} = {current}."
        )

    action_text = ", then ".join(phrases)
    question = (
        f"A number machine starts at {start}, then {action_text}. "
        "What number is displayed after all operations?"
    )
    return Example(
        example_id=example_id,
        question=question,
        clean_steps=steps,
        observed_steps=list(steps),
        answer=str(current),
    )


_RESULT_PATTERN = re.compile(r"=\s*(-?\d+)\.$")


def _numeric_corruption(step: str, rng: random.Random) -> str:
    match = _RESULT_PATTERN.search(step)
    if match is None:
        raise ValueError(f"Cannot locate numeric result in step: {step}")
    original = int(match.group(1))
    delta = rng.choice([value for value in range(-9, 10) if value != 0])
    corrupted = original + delta
    return step[: match.start(1)] + str(corrupted) + step[match.end(1) :]


def _irrelevant_corruption(step_number: int, rng: random.Random) -> str:
    left = rng.randint(20, 80)
    right = rng.randint(2, 15)
    return f"Step {step_number}: {left} + {right} = {left + right}."


def _corrupt_example(example: Example, rng: random.Random) -> None:
    step_index = rng.randrange(len(example.clean_steps))
    noise_type = rng.choice(("numeric", "irrelevant", "repetition"))

    if noise_type == "numeric":
        corrupted = _numeric_corruption(example.clean_steps[step_index], rng)
    elif noise_type == "irrelevant":
        corrupted = _irrelevant_corruption(step_index + 1, rng)
    else:
        alternatives = [i for i in range(len(example.clean_steps)) if i != step_index]
        source_index = rng.choice(alternatives)
        corrupted = example.clean_steps[source_index]

    example.observed_steps[step_index] = corrupted
    example.is_noisy = True
    example.noisy_step_index = step_index
    example.noise_type = noise_type


def build_dataset(config: DataConfig) -> dict[str, list[Example]]:
    rng = random.Random(config.seed)
    total = config.train_size + config.val_size + config.test_size
    examples = [
        _make_clean_example(
            rng,
            example_id=f"arith-{index:05d}",
            min_steps=config.min_steps,
            max_steps=config.max_steps,
        )
        for index in range(total)
    ]

    train = examples[: config.train_size]
    val = examples[config.train_size : config.train_size + config.val_size]
    test = examples[config.train_size + config.val_size :]

    noise_rng = random.Random(config.seed + 1)
    noisy_count = round(config.train_size * config.noise_rate)
    for index in noise_rng.sample(range(config.train_size), noisy_count):
        _corrupt_example(train[index], noise_rng)

    return {"train": train, "val": val, "test": test}


def write_jsonl(path: str | Path, examples: Iterable[Example]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_dict(), ensure_ascii=False) + "\n")
    return sha256(target.read_bytes()).hexdigest()


def read_jsonl(path: str | Path) -> list[Example]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [Example.from_dict(json.loads(line)) for line in handle if line.strip()]


def dataset_summary(splits: dict[str, list[Example]]) -> dict:
    train = splits["train"]
    noise_counts: dict[str, int] = {}
    for example in train:
        if example.noise_type is not None:
            noise_counts[example.noise_type] = noise_counts.get(example.noise_type, 0) + 1
    return {
        "sizes": {name: len(items) for name, items in splits.items()},
        "train_noisy": sum(example.is_noisy for example in train),
        "train_noise_rate": sum(example.is_noisy for example in train) / len(train),
        "noise_types": noise_counts,
        "step_range": {
            "min": min(len(example.clean_steps) for values in splits.values() for example in values),
            "max": max(len(example.clean_steps) for values in splits.values() for example in values),
        },
    }


def materialize_dataset(config: DataConfig, output_dir: str | Path) -> dict:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = build_dataset(config)
    hashes = {
        name: write_jsonl(out / f"{name}.jsonl", examples)
        for name, examples in splits.items()
    }
    summary = dataset_summary(splits)
    manifest = {"config": asdict(config), "hashes": hashes, "summary": summary}
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest
