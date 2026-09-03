from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterator
import importlib.util
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SPECIAL_TOKENS = ("<|start-latent|>", "<|end-latent|>", "<|latent|>")


def validate_checkpoint_compatibility(
    missing_keys: list[str],
    unexpected_keys: list[str],
    *,
    allow_missing_auxiliary: bool,
) -> None:
    if not allow_missing_auxiliary:
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Checkpoint key mismatch: "
                f"missing={missing_keys}, unexpected={unexpected_keys}"
            )
        return
    if unexpected_keys or any(
        not key.startswith("expainable_llm.") for key in missing_keys
    ):
        raise ValueError(
            "Partial checkpoint key mismatch: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    if not missing_keys:
        raise ValueError("Expected a base-only checkpoint with missing auxiliary weights")


@dataclass(frozen=True)
class OfficialExample:
    idx: int
    question: str
    steps: tuple[str, ...]
    answer: str


def parse_icot_line(line: str, idx: int) -> OfficialExample:
    stripped = line.rstrip("\r\n")
    if "||" not in stripped or "##" not in stripped:
        raise ValueError(f"Malformed official data line {idx}")
    question, remainder = stripped.split("||", 1)
    # Match the released preprocessing/gsm_icot.py.  GSM8K source rows use
    # ``####`` before the answer, so rsplit("##", 1) would incorrectly leave
    # a synthetic ``##`` reasoning step behind.
    fields = remainder.split("##")
    reasoning, answer = fields[0], fields[-1]
    steps = tuple(reasoning.strip().split(" "))
    if not question or not steps or not answer.strip():
        raise ValueError(f"Empty field in official data line {idx}")
    return OfficialExample(
        idx=idx,
        question=question,
        steps=steps,
        answer=answer.strip(),
    )


def iter_icot_examples(path: str | Path) -> Iterator[OfficialExample]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if line.strip():
                yield parse_icot_line(line, idx)


def extract_answer_official(decoded_text: str) -> str:
    return decoded_text.split("#")[-1].replace(",", "").strip()


def build_tokenizer(base_model_dir: str | Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    added = tokenizer.add_tokens(list(SPECIAL_TOKENS))
    if added != len(SPECIAL_TOKENS):
        raise ValueError(f"Expected to add 3 latent tokens, added {added}")
    ids = {token: tokenizer.convert_tokens_to_ids(token) for token in SPECIAL_TOKENS}
    if len(set(ids.values())) != len(SPECIAL_TOKENS):
        raise ValueError("Latent special-token IDs are not unique")
    return tokenizer, ids


def load_official_module(official_coconut_dir: str | Path) -> ModuleType:
    module_path = Path(official_coconut_dir) / "coconut.py"
    if not module_path.is_file():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("official_simcot_coconut", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_model(
    *,
    official_coconut_dir: str | Path,
    base_model_dir: str | Path,
    checkpoint_path: str | Path,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    move_auxiliary_to_device: bool = False,
    allow_missing_auxiliary: bool = False,
):
    tokenizer, token_ids = build_tokenizer(base_model_dir)
    model_config = AutoConfig.from_pretrained(
        str(base_model_dir),
        local_files_only=True,
    )
    base_model = AutoModelForCausalLM.from_config(model_config)
    # The released SIM-CoT training code initializes a newly attached auxiliary
    # decoder from the pretrained language model.  Full SIM-CoT checkpoints
    # overwrite every auxiliary weight, so constructing from config is cheaper
    # there.  A pure Coconut checkpoint has no auxiliary weights, however, and
    # must take the pretrained initialization to remain faithful to the method.
    auxiliary_model = (
        AutoModelForCausalLM.from_pretrained(
            str(base_model_dir),
            local_files_only=True,
        )
        if allow_missing_auxiliary
        else AutoModelForCausalLM.from_config(model_config)
    )
    base_model.resize_token_embeddings(len(tokenizer))
    # Match the official run.py exactly: only the base model receives the
    # three latent-token rows.  The auxiliary decoder stays at GPT-2's
    # original 50,257-token vocabulary in the released checkpoint.

    official_module = load_official_module(official_coconut_dir)
    wrapper_config = SimpleNamespace(
        training_method="full",
        max_latent_stage=5,
        explain_mode="v1_aug",
        packing=False,
        w_prompt=False,
    )
    model = official_module.CoconutGPT_Same_Word_Embedding(
        base_model,
        auxiliary_model,
        tokenizer,
        token_ids["<|latent|>"],
        token_ids["<|start-latent|>"],
        token_ids["<|end-latent|>"],
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<<"),
        2,
        wrapper_config,
    )

    state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    incompatible = model.load_state_dict(state, strict=not allow_missing_auxiliary)
    validate_checkpoint_compatibility(
        incompatible.missing_keys,
        incompatible.unexpected_keys,
        allow_missing_auxiliary=allow_missing_auxiliary,
    )
    del state

    model.expainable_llm.to(
        device=device if move_auxiliary_to_device else torch.device("cpu"),
        dtype=dtype,
    )
    model.base_causallm.to(device=device, dtype=dtype)
    model.base_causallm.eval()
    return model, tokenizer, token_ids


def build_eval_tensors(
    example: OfficialExample,
    tokenizer,
    token_ids: dict[str, int],
    *,
    latent_tokens: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if latent_tokens <= 0:
        raise ValueError("latent_tokens must be positive")
    question_ids = tokenizer.encode(
        example.question + "\n",
        add_special_tokens=True,
    )
    tokens = (
        question_ids
        + [token_ids["<|start-latent|>"]]
        + [token_ids["<|latent|>"]] * latent_tokens
        + [token_ids["<|end-latent|>"]]
    )
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }


def summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot summarize zero predictions")
    correct = sum(bool(row["correct"]) for row in rows)
    return {
        "examples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
    }


def _load_completed_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    for expected_idx, row in enumerate(rows):
        if row.get("idx") != expected_idx:
            raise ValueError("Existing predictions are not a contiguous prefix")
    return rows


@torch.inference_mode()
def evaluate_checkpoint(
    *,
    model,
    tokenizer,
    token_ids: dict[str, int],
    dataset_path: str | Path,
    output_dir: str | Path,
    device: torch.device,
    latent_tokens: int,
    max_new_tokens: int,
    expected_accuracy: float,
    accuracy_tolerance: float,
    max_samples: int | None = None,
    resume: bool = False,
    flush_every: int = 25,
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    predictions_path = destination / "predictions.jsonl"
    existing = _load_completed_predictions(predictions_path) if resume else []
    if predictions_path.exists() and not resume:
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}; pass resume=True"
        )

    all_examples = iter_icot_examples(dataset_path)
    limit = max_samples if max_samples is not None else None
    rows = list(existing)
    started = time.perf_counter()
    mode = "a" if resume else "w"
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for example in all_examples:
            if limit is not None and example.idx >= limit:
                break
            if example.idx < len(existing):
                if existing[example.idx]["ground_truth"] != example.answer:
                    raise ValueError("Ground truth changed since the resumable run")
                continue

            tensors = build_eval_tensors(
                example,
                tokenizer,
                token_ids,
                latent_tokens=latent_tokens,
                device=device,
            )
            generated = model.generate(
                **tensors,
                max_new_tokens=max_new_tokens,
                synced_gpus=False,
            )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            prediction = extract_answer_official(decoded)
            row = {
                "idx": example.idx,
                "ground_truth": example.answer,
                "prediction": prediction,
                "correct": prediction == example.answer,
                "decoded_text": decoded,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if len(rows) == 1 or len(rows) % flush_every == 0:
                handle.flush()
                print(
                    f"evaluation: {len(rows)} examples, "
                    f"accuracy={summarize_predictions(rows)['accuracy']:.4f}",
                    flush=True,
                )

    summary = summarize_predictions(rows)
    summary.update(
        {
            "dataset_path": str(dataset_path),
            "ground_truth_source": "official dataset answer field",
            "latent_tokens": latent_tokens,
            "max_new_tokens": max_new_tokens,
            "expected_accuracy": expected_accuracy,
            "accuracy_tolerance": accuracy_tolerance,
            "full_evaluation": max_samples is None,
            "gate_passed": (
                abs(summary["accuracy"] - expected_accuracy) <= accuracy_tolerance
                if max_samples is None
                else None
            ),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "predictions_path": str(predictions_path),
            "peak_reserved_gb": (
                torch.cuda.max_memory_reserved(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    metrics_path = destination / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
