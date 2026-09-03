from __future__ import annotations

from pathlib import Path
from statistics import mean, median, pstdev
import argparse
import json
import math
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from reliable_simcot.error_cancellation_evaluation import evaluate_arm  # noqa: E402
from reliable_simcot.error_cancellation_experiment import (  # noqa: E402
    checkpoint_path,
    load_manifest,
    load_schedule,
    run_training_arm,
    training_directory,
)
from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402


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


def write_state(root: Path, config: dict, phase: str, status: str = "RUNNING", **extra) -> None:
    atomic_json(
        root / config["state_path"],
        {
            "schema_version": 1,
            "run_id": config["pipeline_run_id"],
            "status": status,
            "phase": phase,
            "updated_unix": time.time(),
            **extra,
        },
    )


def verify_frozen_inputs(root: Path, config: dict) -> None:
    frozen_pairs = [
        ("checkpoint_path", "checkpoint_sha256"),
        ("gsm_train_path", "gsm_train_sha256"),
        ("gsm_test_path", "gsm_test_sha256"),
        ("manifest_path", "manifest_file_sha256"),
        ("schedule_path", "schedule_file_sha256"),
    ]
    if config.get("clipped_reference_analysis_path") is not None:
        frozen_pairs.append(
            ("clipped_reference_analysis_path", "clipped_reference_analysis_sha256")
        )
    for path_key, hash_key in frozen_pairs:
        path = root / config[path_key]
        if not path.is_file() or sha256_file(path) != config[hash_key]:
            raise ValueError(f"Frozen artifact missing or changed: {path}")
    if config.get("max_grad_norm") is not None:
        raise ValueError("This experiment requires max_grad_norm=null")
    if tuple(config["selected_arms"]) != ("C", "EW50"):
        raise ValueError("The paired pilot arms must remain C and EW50")
    load_manifest(config, root)
    load_schedule(config, root)


def training_is_complete(root: Path, config: dict, seed: int, arm: str) -> bool:
    metrics_path = training_directory(root, config, seed, arm) / "metrics.json"
    saved_path = checkpoint_path(root, config, seed, arm)
    if not metrics_path.exists() and not saved_path.exists():
        return False
    if not metrics_path.exists() or not saved_path.exists():
        raise RuntimeError(f"Partial training artifact for {seed}:{arm}; refusing overwrite")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if (
        metrics.get("status") != "PASS"
        or metrics.get("gradient_clipping_enabled") is not False
        or metrics.get("max_grad_norm") is not None
        or sha256_file(saved_path) != metrics.get("checkpoint_sha256")
    ):
        raise RuntimeError(f"Stale or invalid completed training artifact for {seed}:{arm}")
    return True


def evaluation_is_complete(root: Path, config: dict, seed: int, arm: str) -> bool:
    metrics_path = root / config["output_root"] / "eval" / f"seed_{seed}" / arm / "metrics.json"
    return metrics_path.is_file()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def analyze(root: Path, config: dict) -> dict:
    arms = tuple(config["selected_arms"])
    seeds = tuple(int(value) for value in config["seeds"])
    accuracies: dict[str, dict] = {}
    gradients: dict[str, dict] = {}
    for arm in arms:
        per_seed: dict[str, float] = {}
        norms: list[float] = []
        for seed in seeds:
            eval_path = (
                root / config["output_root"] / "eval" / f"seed_{seed}" / arm / "metrics.json"
            )
            train_path = training_directory(root, config, seed, arm) / "metrics.json"
            evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
            training = json.loads(train_path.read_text(encoding="utf-8"))
            per_seed[str(seed)] = float(evaluation["accuracy"])
            norms.extend(float(value) for value in training["preclip_gradient_norms"])
        values = list(per_seed.values())
        accuracies[arm] = {
            "per_seed": per_seed,
            "mean": mean(values),
            "population_sd": pstdev(values),
        }
        gradients[arm] = {
            "updates": len(norms),
            "median": median(norms),
            "p95": percentile(norms, 0.95),
            "maximum": max(norms),
            "fraction_above_previous_clip_threshold": mean(value > 1.0 for value in norms),
        }

    paired_effects = [
        100.0
        * (
            accuracies["EW50"]["per_seed"][str(seed)]
            - accuracies["C"]["per_seed"][str(seed)]
        )
        for seed in seeds
    ]
    comparison = {
        "effect_definition": "EW50_accuracy_minus_C_accuracy",
        "per_seed_pp": dict(zip(map(str, seeds), paired_effects)),
        "mean_effect_pp": mean(paired_effects),
        "harm_at_least_2pp": mean(paired_effects) <= -float(config["harm_practical_pp"]),
        "harm_at_least_5pp": mean(paired_effects) <= -float(config["severe_harm_pp"]),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "gradient_clipping_enabled": False,
        "max_grad_norm": None,
        "train_examples": int(config["train_examples"]),
        "updates_per_arm_seed": int(config["updates"]),
        "accuracies": accuracies,
        "gradient_norms": gradients,
        "no_clip_error_vs_clean": comparison,
        "disclosure": config["disclosure"],
    }
    base_metrics_path = root / config["base_eval_dir"] / "metrics.json"
    if base_metrics_path.is_file():
        result["starting_checkpoint_accuracy"] = float(
            json.loads(base_metrics_path.read_text(encoding="utf-8"))["accuracy"]
        )
    if config.get("clipped_reference_analysis_path") is not None:
        clipped = json.loads(
            (root / config["clipped_reference_analysis_path"]).read_text(encoding="utf-8")
        )
        result["clipped_reference"] = {
            "C_mean": float(clipped["accuracies"]["C"]["mean"]),
            "EW50_mean": float(clipped["accuracies"]["EW50"]["mean"]),
            "EW50_minus_C_pp": 100.0
            * (
                float(clipped["accuracies"]["EW50"]["mean"])
                - float(clipped["accuracies"]["C"]["mean"])
            ),
        }
    atomic_json(root / config["analysis_path"], result)
    return result


def write_report(root: Path, config: dict, analysis: dict) -> None:
    clean = analysis["accuracies"]["C"]
    error = analysis["accuracies"]["EW50"]
    effect = analysis["no_clip_error_vs_clean"]
    lines = [
        f"# {config.get('report_title', 'GSM8K 关闭梯度裁剪小样本错误步骤试跑')}",
        "",
        "本实验复用 v13 的 Codex 受控半合成严重单边冲突，不是自然教师噪声；最终答案标签保持正确。",
        "",
        "| 条件 | 三种子平均准确率 | 标准差 |",
        "|---|---:|---:|",
        f"| C：正确步骤 | {100*clean['mean']:.2f}% | {100*clean['population_sd']:.2f} pp |",
        f"| EW50：50%严重宽错误链 | {100*error['mean']:.2f}% | {100*error['population_sd']:.2f} pp |",
        "",
        f"关闭裁剪后 EW50−C：{effect['mean_effect_pp']:.2f} pp。",
        f"达到至少2 pp伤害：{'是' if effect['harm_at_least_2pp'] else '否'}；达到至少5 pp伤害：{'是' if effect['harm_at_least_5pp'] else '否'}。",
    ]
    if "starting_checkpoint_accuracy" in analysis:
        lines.append(
            f"训练起点准确率：{100*analysis['starting_checkpoint_accuracy']:.2f}%。"
        )
    if "clipped_reference" in analysis:
        lines.append(
            "已有裁剪=1.0对照中的 EW50−C："
            f"{analysis['clipped_reference']['EW50_minus_C_pp']:.2f} pp。"
        )
    report_path = root / config["report_path"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired 512-example SIM-CoT error supervision without gradient clipping"
    )
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/error_cancellation_gsm8k_v23_no_clip_pilot.json",
    )
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    (ROOT / config["output_root"]).mkdir(parents=True, exist_ok=True)
    try:
        used = gpu_used_mib()
        if used > int(config["preflight_max_used_mib"]):
            raise RuntimeError(
                f"GPU preflight failed: {used} MiB used > {config['preflight_max_used_mib']} MiB"
            )
        write_state(ROOT, config, "VERIFY_INPUTS", gpu_memory_used_mib=used)
        verify_frozen_inputs(ROOT, config)

        for seed in map(int, config["seeds"]):
            for arm in config["selected_arms"]:
                if not training_is_complete(ROOT, config, seed, arm):
                    write_state(ROOT, config, "TRAIN", current_seed=seed, current_arm=arm)
                    run_training_arm(
                        config,
                        arm=arm,
                        seed=seed,
                        project_root=ROOT,
                        formal=False,
                    )
                if not evaluation_is_complete(ROOT, config, seed, arm):
                    write_state(ROOT, config, "EVAL", current_seed=seed, current_arm=arm)
                    evaluate_arm(config, arm=arm, seed=seed, project_root=ROOT)

        write_state(ROOT, config, "ANALYZE")
        analysis = analyze(ROOT, config)
        write_report(ROOT, config, analysis)
        write_state(
            ROOT,
            config,
            "COMPLETE",
            status="PASS",
            analysis_path=config["analysis_path"],
            report_path=config["report_path"],
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    except Exception as error:
        write_state(ROOT, config, "ERROR", status="FAIL", error=repr(error))
        raise


if __name__ == "__main__":
    main()
