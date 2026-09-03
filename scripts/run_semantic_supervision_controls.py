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
    load_frozen_data,
    run_semantic_conflict_training,
)


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gpu_used_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def write_state(root: Path, config: dict, phase: str, run_status: str = "RUNNING", **extra) -> None:
    atomic_json(
        root / config["state_path"],
        {
            "schema_version": 1,
            "run_id": config["run_id"],
            "status": run_status,
            "phase": phase,
            "updated_unix": time.time(),
            **extra,
        },
    )


def verify_inputs(config: dict, root: Path) -> None:
    checks = (
        ("checkpoint_path", "checkpoint_sha256"),
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
        ("manifest_path", "manifest_file_sha256"),
        ("schedule_path", "schedule_file_sha256"),
        ("v18_error_checkpoint_path", "v18_error_checkpoint_sha256"),
    )
    for path_key, hash_key in checks:
        path = root / config[path_key]
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Frozen input missing or changed: {path}")
    manifest, schedule = load_frozen_data(config, root)
    if manifest["manifest_sha256"] != config["manifest_sha256"]:
        raise ValueError("Frozen manifest semantic hash changed")
    if schedule["schedule_sha256"] != config["schedule_sha256"]:
        raise ValueError("Frozen schedule semantic hash changed")


def evaluate_arm(config: dict, root: Path, arm: str) -> dict:
    device = torch.device(config["device"])
    checkpoint = root / config["work_root"] / arm / "checkpoint_final.pt"
    output_dir = root / config["output_root"] / arm / "eval"
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=False,
        allow_missing_auxiliary=False,
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
    parser = argparse.ArgumentParser(description="Run the v19 SIM-CoT supervision controls")
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/semantic_supervision_controls_v19.json",
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
        write_state(root, config, "VERIFY_INPUTS", gpu_memory_used_mib=used)
        verify_inputs(config, root)
        base = json.loads((root / config["v18_base_metrics_path"]).read_text(encoding="utf-8"))
        error_equal = json.loads(
            (root / config["v18_error_metrics_path"]).read_text(encoding="utf-8")
        )

        arm_results: dict[str, dict] = {}
        for arm in config["arm_order"]:
            smoke_path = root / config["output_root"] / arm / "train" / "smoke" / "metrics.json"
            if not smoke_path.exists():
                write_state(root, config, "GPU_SMOKE", current_arm=arm)
                run_semantic_conflict_training(
                    config,
                    project_root=root,
                    max_updates=2,
                    phase="smoke",
                    save_checkpoint=False,
                    arm=arm,
                )

            train_path = root / config["output_root"] / arm / "train" / "train" / "metrics.json"
            write_state(root, config, "TRAIN", current_arm=arm)
            training = (
                json.loads(train_path.read_text(encoding="utf-8"))
                if train_path.exists()
                else run_semantic_conflict_training(
                    config,
                    project_root=root,
                    phase="train",
                    save_checkpoint=True,
                    arm=arm,
                )
            )

            eval_path = root / config["output_root"] / arm / "eval" / "metrics.json"
            write_state(root, config, "EVAL", current_arm=arm)
            evaluation = (
                json.loads(eval_path.read_text(encoding="utf-8"))
                if eval_path.exists()
                else evaluate_arm(config, root, arm)
            )
            arm_results[arm] = {
                "accuracy": evaluation["accuracy"],
                "correct": evaluation["correct"],
                "checkpoint_sha256": training["checkpoint_sha256"],
                "peak_reserved_gb": training["peak_reserved_gb"],
                "elapsed_seconds": training["elapsed_seconds"],
            }

        accuracies = {
            "pure_coconut_base": base["accuracy"],
            "semantic_conflict_equal_v18": error_equal["accuracy"],
            **{arm: result["accuracy"] for arm, result in arm_results.items()},
        }
        clean_accuracy = accuracies["clean_official"]
        comparisons_pp = {
            "clean_vs_base": 100 * (clean_accuracy - accuracies["pure_coconut_base"]),
            "semantic_conflict_equal_vs_clean": 100
            * (accuracies["semantic_conflict_equal_v18"] - clean_accuracy),
            "redundant_equal_vs_clean": 100 * (accuracies["redundant_equal"] - clean_accuracy),
            "redundant_w01_vs_redundant_equal": 100
            * (accuracies["redundant_w01"] - accuracies["redundant_equal"]),
            "semantic_conflict_w01_vs_equal": 100
            * (accuracies["semantic_conflict_w01"] - accuracies["semantic_conflict_equal_v18"]),
            "semantic_conflict_w01_vs_clean": 100
            * (accuracies["semantic_conflict_w01"] - clean_accuracy),
        }
        analysis = {
            "schema_version": 1,
            "status": "COMPLETE",
            "accuracies": accuracies,
            "comparisons_pp": comparisons_pp,
            "arm_results": arm_results,
            "single_seed": int(config["seed"]),
            "train_examples_per_arm": int(config["train_examples"]),
            "disclosure": config["disclosure"],
        }
        atomic_json(root / config["analysis_path"], analysis)
        write_state(root, config, "COMPLETE", run_status="PASS", analysis_path=config["analysis_path"])
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_state(root, config, "ERROR", run_status="FAIL", error=repr(error))
        raise


if __name__ == "__main__":
    main()

