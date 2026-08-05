from __future__ import annotations

from pathlib import Path
from typing import Any
import argparse
import hashlib
import json
import time

import torch

from reliable_simcot.official_adapter import (
    OfficialExample,
    build_eval_tensors,
    load_official_model,
)
from reliable_simcot.ood_adapter import (
    extract_answer_number_official,
    load_ood_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the official GPT-2 Coconut+SIM-CoT checkpoint on OOD arithmetic tasks."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def project_path(root: Path, value: str) -> Path:
    target = (root / value).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes project root: {value}")
    return target


def read_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if any(row.get("idx") != idx for idx, row in enumerate(rows)):
        raise ValueError(f"Predictions are not a contiguous prefix: {path}")
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset_config(dataset_config: dict[str, Any], paths: list[Path]) -> None:
    hashes = dataset_config["source_sha256"]
    if len(hashes) != len(paths):
        raise ValueError("Each OOD source path must have one registered SHA-256")
    for path, expected_hash in zip(paths, hashes, strict=True):
        if sha256_file(path) != expected_hash:
            raise ValueError(f"OOD source SHA-256 mismatch: {path}")


@torch.inference_mode()
def evaluate_dataset(
    *,
    model,
    tokenizer,
    token_ids: dict[str, int],
    dataset_config: dict[str, Any],
    root: Path,
    device: torch.device,
    latent_tokens: int,
    max_new_tokens: int,
    max_samples: int | None,
    resume: bool,
    flush_every: int,
) -> dict[str, Any]:
    name = dataset_config["name"]
    paths = [project_path(root, value) for value in dataset_config["paths"]]
    validate_dataset_config(dataset_config, paths)
    examples = load_ood_examples(name, paths)
    if len(examples) != dataset_config["expected_examples"]:
        raise ValueError(
            f"{name} expected {dataset_config['expected_examples']} examples, "
            f"found {len(examples)}"
        )
    if max_samples is not None:
        examples = examples[:max_samples]

    output_dir = project_path(root, dataset_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    existing = read_predictions(predictions_path) if resume else []
    if predictions_path.exists() and not resume:
        raise FileExistsError(f"Predictions already exist: {predictions_path}")
    if len(existing) > len(examples):
        raise ValueError("Existing predictions exceed the requested evaluation set")
    for idx, row in enumerate(existing):
        if float(row["ground_truth"]) != examples[idx].answer:
            raise ValueError(f"{name} ground truth changed at example {idx}")

    rows = list(existing)
    started = time.perf_counter()
    with predictions_path.open("a" if resume else "w", encoding="utf-8", newline="\n") as handle:
        for example in examples[len(existing) :]:
            tensors = build_eval_tensors(
                OfficialExample(example.idx, example.question, (), str(example.answer)),
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
            generated_text = tokenizer.decode(
                generated[0, tensors["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            prediction = extract_answer_number_official(generated_text)
            row = {
                "idx": example.idx,
                "ground_truth": example.answer,
                "prediction": prediction,
                "correct": prediction == example.answer,
                "decoded_text": decoded,
                "generated_text": generated_text,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if len(rows) == 1 or len(rows) % flush_every == 0:
                handle.flush()
                correct = sum(bool(item["correct"]) for item in rows)
                print(
                    f"{name}: {len(rows)}/{len(examples)}, accuracy={correct / len(rows):.4f}",
                    flush=True,
                )

    correct = sum(bool(row["correct"]) for row in rows)
    accuracy = correct / len(rows)
    expected = dataset_config["expected_accuracy"]
    tolerance = dataset_config["accuracy_tolerance"]
    full_evaluation = max_samples is None
    metrics = {
        "dataset": name,
        "examples": len(rows),
        "correct": correct,
        "accuracy": accuracy,
        "expected_accuracy": expected,
        "accuracy_tolerance": tolerance,
        "full_evaluation": full_evaluation,
        "gate_passed": abs(accuracy - expected) <= tolerance if full_evaluation else None,
        "source_paths": [str(path) for path in paths],
        "ground_truth_source": "public dataset answer field",
        "answer_rule": "last numeric substring in generated continuation, matching released CODI/test.py",
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "predictions_path": str(predictions_path),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    provenance = json.loads(
        project_path(root, config["provenance_manifest"]).read_text(encoding="utf-8")
    )
    if provenance.get("status") != "PASS" or provenance.get("run_id") != "R001":
        raise ValueError("R001 provenance must pass before R003 evaluation")

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=project_path(root, config["official_source_dir"]),
        base_model_dir=project_path(root, config["base_model_dir"]),
        checkpoint_path=project_path(root, config["checkpoint_path"]),
        device=device,
        dtype=torch.float32,
    )
    metrics = []
    for dataset_config in config["datasets"]:
        metrics.append(
            evaluate_dataset(
                model=model,
                tokenizer=tokenizer,
                token_ids=token_ids,
                dataset_config=dataset_config,
                root=root,
                device=device,
                latent_tokens=config["latent_tokens"],
                max_new_tokens=config["max_new_tokens"],
                max_samples=args.max_samples,
                resume=args.resume,
                flush_every=config.get("flush_every", 25),
            )
        )
    summary = {
        "run_id": "R003",
        "status": (
            "PASS"
            if all(metric["gate_passed"] is True for metric in metrics)
            else "PARTIAL" if args.max_samples is not None else "FAIL"
        ),
        "datasets": metrics,
        "peak_reserved_gb": (
            torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else 0.0
        ),
    }
    summary_path = project_path(root, config["summary_path"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
