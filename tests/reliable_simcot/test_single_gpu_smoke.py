from __future__ import annotations

import torch

from reliable_simcot.official_adapter import OfficialExample
from reliable_simcot.single_gpu_smoke import (
    encode_smoke_example,
    tensorize_smoke_example,
    validate_alignment,
    validate_r004_for_resume,
)


class DummyTokenizer:
    eos_token_id = 99

    def encode(self, text: str, add_special_tokens: bool) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [10 + (ord(character) % 40) for character in text]


def test_encoded_smoke_example_aligns_latents_and_steps() -> None:
    example = OfficialExample(
        idx=3,
        question="q",
        steps=("<<1+1=2>>", "<<2+2=4>>", "<<4+4=8>>"),
        answer="8",
    )
    tokenizer = DummyTokenizer()
    token_ids = {
        "<|start-latent|>": 100,
        "<|end-latent|>": 101,
        "<|latent|>": 102,
    }
    encoded = encode_smoke_example(
        example,
        tokenizer,
        token_ids,
        latent_stage=5,
        c_thought=2,
    )
    batch = tensorize_smoke_example(encoded, device=torch.device("cpu"))
    alignment = validate_alignment(
        encoded,
        batch,
        latent_id=102,
        c_thought=2,
    )

    assert encoded.latent_tokens == 10
    assert alignment["latent_groups"] == 5
    assert alignment["real_supervised_groups"] == 3
    assert alignment["pseudo_filled_groups"] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"][0, :4] == -100).all()


def test_validate_r004_for_resume_checks_checkpoint_hash(tmp_path) -> None:
    import hashlib
    import pytest

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    metrics = {
        "run_id": "R004",
        "status": "PASS",
        "reload_consistent": True,
        "gate_passed": True,
        "checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest(),
    }
    validate_r004_for_resume(metrics, checkpoint)

    metrics["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        validate_r004_for_resume(metrics, checkpoint)
