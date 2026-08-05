from __future__ import annotations

from pathlib import Path
import argparse
import json

import torch

from reliable_simcot.official_adapter import evaluate_checkpoint, load_official_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the official GPT-2 Coconut+SIM-CoT checkpoint on one GPU."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _project_path(root: Path, value: str) -> Path:
    target = (root / value).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes project root: {value}")
    return target


def main() -> None:
    args = parse_args()
    root = Path.cwd().resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    provenance_path = _project_path(root, config["provenance_manifest"])
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("status") != "PASS" or provenance.get("run_id") != "R001":
        raise ValueError("R001 provenance must pass before R002 evaluation")
    if config.get("ground_truth_source") != "dataset.answer":
        raise ValueError("Evaluation ground truth must be dataset.answer")

    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=_project_path(root, config["official_source_dir"]),
        base_model_dir=_project_path(root, config["base_model_dir"]),
        checkpoint_path=_project_path(root, config["checkpoint_path"]),
        device=device,
        dtype=torch.float32,
        allow_missing_auxiliary=config.get("allow_missing_auxiliary", False),
    )
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=_project_path(root, config["dataset_path"]),
        output_dir=_project_path(root, config["output_dir"]),
        device=device,
        latent_tokens=config["latent_tokens"],
        max_new_tokens=config["max_new_tokens"],
        expected_accuracy=config["expected_accuracy"],
        accuracy_tolerance=config["accuracy_tolerance"],
        max_samples=args.max_samples,
        resume=args.resume,
        flush_every=config.get("flush_every", 25),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
