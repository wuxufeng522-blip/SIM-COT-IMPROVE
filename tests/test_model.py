import torch
from transformers import GPT2Config

from rsr_rd_simcot.model import SimCoTModel


def tiny_model() -> SimCoTModel:
    config = GPT2Config(
        vocab_size=97,
        n_positions=64,
        n_embd=32,
        n_layer=2,
        n_head=4,
        bos_token_id=1,
        eos_token_id=2,
    )
    return SimCoTModel.from_config(config)


def test_forward_shapes_and_gradient_flow() -> None:
    model = tiny_model()
    output = model(
        question_ids=torch.tensor([3, 4, 5, 6]),
        answer_ids=torch.tensor([7, 8]),
        step_ids=[torch.tensor([9, 10, 11]), torch.tensor([12, 13])],
        weights=torch.tensor([1.5, 0.5]),
    )

    assert output.step_losses.shape == (2,)
    assert output.latents.shape == (1, 2, 32)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert model.student.transformer.h[0].attn.c_attn.weight.grad is not None


def test_all_one_weights_match_mean_reduction() -> None:
    model = tiny_model()
    output = model(
        question_ids=torch.tensor([3, 4, 5]),
        answer_ids=torch.tensor([6]),
        step_ids=[torch.tensor([7, 8]), torch.tensor([9, 10, 11])],
        weights=torch.ones(2),
    )
    assert torch.allclose(output.step_loss, output.step_losses.mean())
