from pathlib import Path

import torch

from reliable_simcot.semantic_conflict_scaling import (
    _bucket_slices,
    _latest_resume,
    auxiliary_target_steps,
    global_gradient_norm,
    state_dict_semantic_sha256,
    variable_tokenize_step_targets,
)
from reliable_simcot.m1_training import sha256_file


class Tokenizer:
    eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [len(text)]


def test_variable_targets_accept_four_and_five_real_steps():
    tokenizer = Tokenizer()
    assert len(variable_tokenize_step_targets(tokenizer, ["a"] * 4)) == 4
    assert len(variable_tokenize_step_targets(tokenizer, ["a"] * 5)) == 5


def test_auxiliary_target_selection_is_explicit():
    row = {
        "clean_steps": ["correct"],
        "semantic_conflict": {"steps": ["wrong"]},
    }
    assert auxiliary_target_steps(row, {}) == ["wrong"]
    assert auxiliary_target_steps(row, {"auxiliary_target": "official_clean_steps"}) == [
        "correct"
    ]


def test_small_last_bucket_is_merged():
    rows = [{"i": value} for value in range(528)]
    chunks = _bucket_slices(rows, 256)
    assert [len(chunk) for chunk in chunks] == [256, 272]


def test_semantic_state_hash_is_filename_independent(tmp_path: Path):
    state = {"weight": torch.arange(5, dtype=torch.float32)}
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    torch.save(state, left)
    torch.save(state, right)
    assert state_dict_semantic_sha256(left) == state_dict_semantic_sha256(right)


def test_external_bootstrap_resume_is_hash_checked(tmp_path: Path):
    model_path = tmp_path / "source" / "model.pt"
    optimizer_path = tmp_path / "source" / "optimizer.pt"
    model_path.parent.mkdir()
    torch.save({"weight": torch.arange(3, dtype=torch.float32)}, model_path)
    torch.save({"state": {}, "param_groups": []}, optimizer_path)
    config = {
        "milestone_examples": [8, 16],
        "gradient_accumulation_steps": 8,
        "work_root": "new_run",
        "bootstrap_resume": {
            "examples_seen": 8,
            "model_path": str(model_path.relative_to(tmp_path)),
            "model_file_sha256": sha256_file(model_path),
            "model_semantic_sha256": state_dict_semantic_sha256(model_path),
            "optimizer_path": str(optimizer_path.relative_to(tmp_path)),
            "optimizer_file_sha256": sha256_file(optimizer_path),
            "source_run_id": "old-run",
            "source_gate_status": "FAILED_EXACT_HASH",
        },
    }
    examples, loaded_model, loaded_optimizer, meta = _latest_resume(tmp_path, config)
    assert examples == 8
    assert loaded_model == model_path
    assert loaded_optimizer == optimizer_path
    assert meta["metric_history_start_update"] == 1
    assert not meta["metric_history_complete"]


def test_global_gradient_norm_does_not_modify_gradients():
    left = torch.nn.Parameter(torch.tensor([0.0]))
    right = torch.nn.Parameter(torch.tensor([0.0]))
    left.grad = torch.tensor([3.0])
    right.grad = torch.tensor([4.0])

    norm = global_gradient_norm([left, right])

    assert norm.item() == 5.0
    assert left.grad.item() == 3.0
    assert right.grad.item() == 4.0
