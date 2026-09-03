from __future__ import annotations

from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402
from reliable_simcot.official_adapter import evaluate_checkpoint, load_official_model  # noqa: E402
from reliable_simcot.semantic_conflict_pilot import (  # noqa: E402
    prepare_semantic_conflict_data,
    run_semantic_conflict_training,
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_used_mib() -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def write_state(root: Path, config: dict, phase: str, status: str = "RUNNING", **extra) -> None:
    atomic_json(
        root / config["state_path"],
        {
            "schema_version": 1,
            "run_id": config["run_id"],
            "status": status,
            "phase": phase,
            "updated_unix": time.time(),
            **extra,
        },
    )


def evaluate(config: dict, root: Path, *, final: bool) -> dict:
    device = torch.device(config["device"])
    checkpoint = root / (
        config["checkpoint_output_path"] if final else config["checkpoint_path"]
    )
    output_dir = root / (config["final_eval_dir"] if final else config["base_eval_dir"])
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=False,
        allow_missing_auxiliary=not final,
    )
    model.base_causallm.eval()
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=root / config["gsm_test_path"],
        output_dir=output_dir,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=(output_dir / "predictions.jsonl").exists(),
        flush_every=int(config["flush_every"]),
    )
    del model
    torch.cuda.empty_cache()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the single-arm SIM-CoT semantic-conflict pilot")
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/semantic_conflict_pilot_v18.json",
    )
    args = parser.parse_args()
    root = ROOT
    config = load_config(root / args.config)
    (root / config["output_root"]).mkdir(parents=True, exist_ok=True)
    try:
        used = gpu_used_mib()
        if used > int(config["preflight_max_used_mib"]):
            raise RuntimeError(
                f"GPU preflight failed: {used} MiB used > {config['preflight_max_used_mib']} MiB"
            )
        write_state(root, config, "PREPARE_DATA", gpu_memory_used_mib=used)
        audit = prepare_semantic_conflict_data(config, project_root=root)

        smoke_metrics = root / config["train_output_dir"] / "smoke" / "metrics.json"
        if not smoke_metrics.exists():
            write_state(root, config, "GPU_SMOKE", manifest_sha256=audit["manifest_sha256"])
            run_semantic_conflict_training(
                config,
                project_root=root,
                max_updates=2,
                phase="smoke",
                save_checkpoint=False,
            )

        base_metrics_path = root / config["base_eval_dir"] / "metrics.json"
        write_state(root, config, "BASE_EVAL")
        base = (
            json.loads(base_metrics_path.read_text(encoding="utf-8"))
            if base_metrics_path.exists()
            else evaluate(config, root, final=False)
        )

        train_metrics_path = root / config["train_output_dir"] / "train" / "metrics.json"
        write_state(root, config, "SEMANTIC_CONFLICT_TRAIN", base_accuracy=base["accuracy"])
        training = (
            json.loads(train_metrics_path.read_text(encoding="utf-8"))
            if train_metrics_path.exists()
            else run_semantic_conflict_training(config, project_root=root)
        )

        final_metrics_path = root / config["final_eval_dir"] / "metrics.json"
        write_state(root, config, "FINAL_EVAL", base_accuracy=base["accuracy"])
        final = (
            json.loads(final_metrics_path.read_text(encoding="utf-8"))
            if final_metrics_path.exists()
            else evaluate(config, root, final=True)
        )
        effect_pp = 100.0 * (final["accuracy"] - base["accuracy"])
        result = {
            "schema_version": 1,
            "status": "COMPLETE",
            "base_accuracy": base["accuracy"],
            "semantic_conflict_accuracy": final["accuracy"],
            "change_pp": effect_pp,
            "harm_2pp": effect_pp <= -2.0,
            "harm_5pp": effect_pp <= -5.0,
            "train_examples": int(config["train_examples"]),
            "single_seed": int(config["seed"]),
            "starting_checkpoint_sha256": config["checkpoint_sha256"],
            "trained_checkpoint_sha256": training["checkpoint_sha256"],
            "disclosure": config["disclosure"],
        }
        atomic_json(root / config["output_root"] / "analysis.json", result)
        write_state(root, config, "COMPLETE", status="PASS", **result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_state(root, config, "ERROR", status="FAIL", error=repr(error))
        raise


if __name__ == "__main__":
    main()

