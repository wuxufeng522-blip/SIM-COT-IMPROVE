from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


@dataclass(frozen=True)
class DataConfig:
    train_size: int = 1200
    val_size: int = 200
    test_size: int = 200
    noise_rate: float = 0.20
    min_steps: int = 2
    max_steps: int = 4
    seed: int = 20260804


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "gpt2"
    max_question_tokens: int = 96
    max_step_tokens: int = 48
    max_answer_tokens: int = 24
    max_latent_steps: int = 4
    rank_clip: int = 100


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    warmup_updates: int = 400
    branch_updates: int = 800
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    checkpoint_every: int = 200
    lambda_step: float = 1.0
    max_reserved_memory_gb: float = 7.4
    hard_stop_hours: float = 10.0


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    work_dir: str = "work/overnight_poc"
    output_dir: str = "outputs/overnight_poc"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
