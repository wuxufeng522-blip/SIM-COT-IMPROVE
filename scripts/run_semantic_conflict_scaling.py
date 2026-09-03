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

from reliable_simcot.m1_training import atomic_json  # noqa: E402
from reliable_simcot.official_adapter import evaluate_checkpoint, load_official_model  # noqa: E402
from reliable_simcot.semantic_conflict_scaling import (  # noqa: E402
    _milestone_paths,
    prepare_scaling_data,
    run_scaling_training,
    state_dict_semantic_sha256,
)


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


def evaluate_milestone(config: dict, root: Path, examples: int) -> dict:
    model_path, _, _ = _milestone_paths(root, config, examples)
    bootstrap = config.get("bootstrap_resume")
    if bootstrap is not None and examples == int(bootstrap["examples_seen"]):
        model_path = root / bootstrap["model_path"]
    output_dir = root / config["output_root"] / "eval" / f"examples_{examples}"
    device = torch.device(config["device"])
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=root / config["official_source_dir"],
        base_model_dir=root / config["base_model_dir"],
        checkpoint_path=model_path,
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
    parser = argparse.ArgumentParser(description="Run reliable 32k semantic-conflict scaling")
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/semantic_conflict_scaling_v20.json",
    )
    args = parser.parse_args()
    root = ROOT
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    (root / config["output_root"]).mkdir(parents=True, exist_ok=True)
    try:
        dependency = config.get("depends_on")
        if dependency is not None:
            dependency_state = json.loads(
                (root / dependency["state_path"]).read_text(encoding="utf-8")
            )
            if (
                dependency_state.get("status") != dependency["expected_status"]
                or dependency_state.get("phase") != dependency["expected_phase"]
                or not (root / dependency["analysis_path"]).is_file()
            ):
                raise RuntimeError("Configured predecessor experiment is not complete")
        used = gpu_used_mib()
        if used > int(config["preflight_max_used_mib"]):
            raise RuntimeError(
                f"GPU preflight failed: {used} MiB used > {config['preflight_max_used_mib']} MiB"
            )
        write_state(root, config, "PREPARE_DATA", gpu_memory_used_mib=used)
        audit = prepare_scaling_data(config, project_root=root)
        source_semantic = state_dict_semantic_sha256(root / config["source_v18_checkpoint_path"])
        if source_semantic != config["source_v18_checkpoint_semantic_sha256"]:
            raise ValueError("v18 reproduction reference checkpoint changed")

        smoke_path = root / config["output_root"] / "smoke" / "metrics.json"
        if not smoke_path.exists():
            write_state(root, config, "GPU_SMOKE", manifest_sha256=audit["manifest_sha256"])
            run_scaling_training(
                config,
                project_root=root,
                max_updates=2,
                phase="smoke",
            )

        train_path = root / config["training_metrics_path"]
        write_state(root, config, "TRAIN", target_examples=int(config["unique_train_examples"]))
        training = (
            json.loads(train_path.read_text(encoding="utf-8"))
            if train_path.exists()
            else run_scaling_training(config, project_root=root)
        )

        base = json.loads(
            (root / config["source_base_eval_metrics_path"]).read_text(encoding="utf-8")
        )
        accuracies: dict[str, float] = {"0": float(base["accuracy"])}
        correct: dict[str, int] = {"0": int(base["correct"])}
        evaluation_examples = (8192, 16384, 32768)
        for examples in evaluation_examples:
            eval_path = root / config["output_root"] / "eval" / f"examples_{examples}" / "metrics.json"
            if (
                examples == 8192
                and config.get("bootstrap_resume") is None
                and config.get("auxiliary_target", "semantic_conflict_steps")
                == "semantic_conflict_steps"
            ):
                metrics = json.loads(
                    (root / config["source_v18_eval_metrics_path"]).read_text(encoding="utf-8")
                )
            else:
                write_state(root, config, "EVAL", current_milestone_examples=examples)
                metrics = (
                    json.loads(eval_path.read_text(encoding="utf-8"))
                    if eval_path.exists()
                    else evaluate_milestone(config, root, examples)
                )
            accuracies[str(examples)] = float(metrics["accuracy"])
            correct[str(examples)] = int(metrics["correct"])

        analysis = {
            "schema_version": 1,
            "status": "COMPLETE",
            "accuracies": accuracies,
            "correct": correct,
            "change_vs_base_pp": {
                key: 100 * (value - accuracies["0"])
                for key, value in accuracies.items()
                if key != "0"
            },
            "incremental_change_pp": {
                "16384_vs_8192": 100 * (accuracies["16384"] - accuracies["8192"]),
                "32768_vs_16384": 100 * (accuracies["32768"] - accuracies["16384"]),
            },
            "v18_reproduction_gate": config.get("v18_reproduction_gate_status", "PASS")
            == "PASS",
            "v18_reproduction_gate_status": config.get(
                "v18_reproduction_gate_status", "PASS"
            ),
            "unique_training_examples": int(config["unique_train_examples"]),
            "single_seed": int(config["seed"]),
            "auxiliary_target": config.get(
                "auxiliary_target", "semantic_conflict_steps"
            ),
            "training": training,
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
