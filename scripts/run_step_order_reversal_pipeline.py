from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev, stdev
from typing import Any
import argparse
import gc
import json
import subprocess
import sys
import time
import traceback

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.full_conflict_data import (  # noqa: E402
    canonical_hash,
    load_frozen_schedule,
)
from reliable_simcot.full_conflict_evaluation import (  # noqa: E402
    exact_mcnemar,
    paired_bootstrap_damage_pp,
)
from reliable_simcot.full_conflict_experiment import (  # noqa: E402
    STEP_ORDER_REVERSAL_ARM,
    checkpoint_path,
    run_full_conflict_training,
    training_directory,
)
from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402
from reliable_simcot.official_adapter import (  # noqa: E402
    evaluate_checkpoint,
    load_official_model,
)


def _load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def _save_state(config: dict[str, Any], **updates: Any) -> dict[str, Any]:
    path = (ROOT / config["pipeline_state_path"]).resolve()
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "run_id": "SOR-V14",
            "started_unix": time.time(),
            "completed_training": [],
            "completed_evaluation": [],
        }
    if updates.get("status") in {"RUNNING", "PASS"}:
        for key in ("error_type", "error", "traceback"):
            state.pop(key, None)
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)
    return state


def _gpu_used_mib() -> int:
    process = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(process.stdout.splitlines()[0].strip())


def _preflight(config: dict[str, Any], label: str) -> None:
    limit = int(config["preflight_max_used_mib"])
    used = -1
    for attempt in range(1, 4):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        used = _gpu_used_mib()
        if used <= limit:
            print(f"GPU preflight {label}: {used} MiB <= {limit} MiB", flush=True)
            return
        if attempt < 3:
            print(
                f"GPU preflight {label}: transient {used} MiB > {limit} MiB; "
                f"cleanup retry {attempt}/2",
                flush=True,
            )
            time.sleep(2.0)
    raise RuntimeError(f"GPU preflight failed for {label}: {used} MiB > {limit} MiB")


def _data_audit(config: dict[str, Any]) -> dict[str, Any]:
    schedule_path = (ROOT / config["frozen_schedule_path"]).resolve()
    if sha256_file(schedule_path) != config["source_schedule_file_sha256"]:
        raise ValueError("Source schedule file SHA-256 mismatch")
    schedule = load_frozen_schedule(config, ROOT)
    if schedule["schedule_sha256"] != config["source_schedule_sha256"]:
        raise ValueError("Source schedule semantic SHA-256 mismatch")
    rows = schedule["train_entries"]
    failures: list[str] = []
    changed_positions: list[int] = []
    samples: list[dict[str, Any]] = []
    for row in rows:
        clean = tuple(row["clean_steps"])
        reversed_steps = tuple(reversed(clean))
        if len(clean) != 5:
            failures.append(f"{row['question_id']}:not_five_steps")
        if reversed_steps == clean:
            failures.append(f"{row['question_id']}:palindromic")
        if sorted(reversed_steps) != sorted(clean):
            failures.append(f"{row['question_id']}:step_multiset_changed")
        changed_positions.append(sum(a != b for a, b in zip(clean, reversed_steps, strict=True)))
        if len(samples) < 5:
            samples.append(
                {
                    "question_id": row["question_id"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "clean_steps": list(clean),
                    "reversed_steps": list(reversed_steps),
                }
            )
    result = {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "transformation": "reverse the five explicit step targets: [1,2,3,4,5] -> [5,4,3,2,1]",
        "source_schedule_sha256": schedule["schedule_sha256"],
        "source_schedule_file_sha256": sha256_file(schedule_path),
        "examples": len(rows),
        "exact_five_step_examples": sum(len(row["clean_steps"]) == 5 for row in rows),
        "non_palindromic_examples": sum(
            tuple(row["clean_steps"]) != tuple(reversed(row["clean_steps"])) for row in rows
        ),
        "mean_changed_positions": mean(changed_positions),
        "step_multiset_preserved": not any("step_multiset_changed" in item for item in failures),
        "question_answer_fields_untouched": True,
        "coverage": 1.0,
        "auxiliary_weight": 1.0,
        "failures": failures,
        "samples": samples,
        "official_test_opened": False,
    }
    result["audit_sha256"] = canonical_hash(result)
    atomic_json((ROOT / config["data_audit_path"]).resolve(), result)
    if result["status"] != "PASS":
        raise RuntimeError("Step-order reversal data audit failed")
    return result


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _baseline_audit(config: dict[str, Any]) -> dict[str, Any]:
    confirm_path = (ROOT / config["confirm_dataset_path"]).resolve()
    if sha256_file(confirm_path) != config["confirm_dataset_sha256"]:
        raise ValueError("Confirmation dataset SHA-256 mismatch")
    rows: dict[str, Any] = {}
    for seed in config["seeds"]:
        directory = (
            ROOT
            / config["baseline_output_root"]
            / "eval"
            / f"seed_{seed}"
            / config["baseline_arm"]
        ).resolve()
        metrics_path = directory / "metrics.json"
        predictions_path = directory / "predictions.jsonl"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = _read_predictions(predictions_path)
        expected = float(config["baseline_accuracy_by_seed"][str(seed)])
        if (
            metrics.get("schedule_sha256") != config["source_schedule_sha256"]
            or metrics.get("confirm_dataset_sha256") != config["confirm_dataset_sha256"]
            or len(predictions) != int(config["confirm_examples"])
            or float(metrics["accuracy"]) != expected
        ):
            raise ValueError(f"Frozen Clean baseline mismatch for seed {seed}")
        rows[str(seed)] = {
            "accuracy": metrics["accuracy"],
            "correct": metrics["correct"],
            "metrics_sha256": sha256_file(metrics_path),
            "predictions_sha256": sha256_file(predictions_path),
        }
    result = {
        "schema_version": 1,
        "status": "PASS",
        "arm": config["baseline_arm"],
        "per_seed": rows,
        "mean_accuracy": mean(row["accuracy"] for row in rows.values()),
        "confirm_dataset_sha256": config["confirm_dataset_sha256"],
        "source_schedule_sha256": config["source_schedule_sha256"],
        "official_test_opened": False,
    }
    if result["mean_accuracy"] != float(config["baseline_mean_accuracy"]):
        raise ValueError("Frozen Clean mean accuracy mismatch")
    atomic_json((ROOT / config["baseline_audit_path"]).resolve(), result)
    return result


def _training_complete(config: dict[str, Any], seed: int, audit_hash: str) -> bool:
    arm = config["arm"]
    metrics_path = training_directory(ROOT, config, seed, arm) / "metrics.json"
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if not metrics_path.exists() or not checkpoint.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == arm
            and metrics.get("schedule_sha256") == config["source_schedule_sha256"]
            and metrics.get("reversal_audit_sha256") == audit_hash
            and sha256_file(checkpoint) == metrics.get("checkpoint_sha256")
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _evaluation_directory(config: dict[str, Any], seed: int) -> Path:
    return (ROOT / config["output_root"] / "eval" / f"seed_{seed}" / config["arm"]).resolve()


def _evaluation_complete(config: dict[str, Any], seed: int, audit_hash: str) -> bool:
    directory = _evaluation_directory(config, seed)
    metrics_path = directory / "metrics.json"
    predictions_path = directory / "predictions.jsonl"
    if not metrics_path.exists() or not predictions_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        return (
            metrics.get("status") == "PASS"
            and metrics.get("seed") == seed
            and metrics.get("arm") == config["arm"]
            and metrics.get("reversal_audit_sha256") == audit_hash
            and len(_read_predictions(predictions_path)) == int(config["confirm_examples"])
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _train(config: dict[str, Any], seed: int, audit_hash: str) -> dict[str, Any]:
    _preflight(config, f"TRAIN:{seed}")
    metrics = run_full_conflict_training(
        config,
        arm=STEP_ORDER_REVERSAL_ARM,
        seed=seed,
        project_root=ROOT,
    )
    metrics.update(
        {
            "transformation": "reverse_steps_100",
            "reversal_audit_sha256": audit_hash,
            "question_answer_fields_untouched": True,
        }
    )
    atomic_json(training_directory(ROOT, config, seed, config["arm"]) / "metrics.json", metrics)
    if metrics["status"] != "PASS":
        raise RuntimeError(f"Training failed its memory gate for seed {seed}")
    return metrics


@torch.inference_mode()
def _evaluate(config: dict[str, Any], seed: int, audit_hash: str) -> dict[str, Any]:
    _preflight(config, f"EVAL:{seed}")
    arm = config["arm"]
    train_path = training_directory(ROOT, config, seed, arm) / "metrics.json"
    train_metrics = json.loads(train_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if sha256_file(checkpoint) != train_metrics["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint SHA-256 mismatch")
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(ROOT / config["official_source_dir"]).resolve(),
        base_model_dir=(ROOT / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    directory = _evaluation_directory(config, seed)
    predictions_path = directory / "predictions.jsonl"
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=(ROOT / config["confirm_dataset_path"]).resolve(),
        output_dir=directory,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=predictions_path.exists(),
        flush_every=int(config["flush_every"]),
    )
    metrics.update(
        {
            "status": "PASS",
            "run_id": f"SOR-EVAL-{seed}",
            "arm": arm,
            "seed": seed,
            "training_checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": config["source_schedule_sha256"],
            "confirm_dataset_sha256": config["confirm_dataset_sha256"],
            "reversal_audit_sha256": audit_hash,
            "official_test_opened": False,
        }
    )
    atomic_json(directory / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _analyze(config: dict[str, Any]) -> dict[str, Any]:
    per_seed: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        clean_directory = (
            ROOT
            / config["baseline_output_root"]
            / "eval"
            / f"seed_{seed}"
            / config["baseline_arm"]
        ).resolve()
        reversal_directory = _evaluation_directory(config, seed)
        clean_metrics = json.loads((clean_directory / "metrics.json").read_text(encoding="utf-8"))
        reversal_metrics = json.loads((reversal_directory / "metrics.json").read_text(encoding="utf-8"))
        clean_rows = _read_predictions(clean_directory / "predictions.jsonl")
        reversal_rows = _read_predictions(reversal_directory / "predictions.jsonl")
        if [row["ground_truth"] for row in clean_rows] != [row["ground_truth"] for row in reversal_rows]:
            raise ValueError(f"Paired ground truth mismatch for seed {seed}")
        clean_flags = [bool(row["correct"]) for row in clean_rows]
        reversal_flags = [bool(row["correct"]) for row in reversal_rows]
        clean_accuracy = float(clean_metrics["accuracy"])
        reversal_accuracy = float(reversal_metrics["accuracy"])
        per_seed.append(
            {
                "seed": seed,
                "clean_accuracy": clean_accuracy,
                "reversal_accuracy": reversal_accuracy,
                "damage_pp": 100.0 * (clean_accuracy - reversal_accuracy),
                "mcnemar": exact_mcnemar(clean_flags, reversal_flags),
                "paired_bootstrap": paired_bootstrap_damage_pp(
                    clean_flags,
                    reversal_flags,
                    samples=int(config["bootstrap_samples"]),
                    seed=seed,
                ),
            }
        )
    effects = [row["damage_pp"] for row in per_seed]
    mean_damage = mean(effects)
    half_width = 4.302652729911275 * stdev(effects) / len(effects) ** 0.5
    criteria = {
        "all_three_seeds_harmed": all(value > 0 for value in effects),
        "mean_damage_at_least_2pp": mean_damage >= float(config["min_damage_pp"]),
    }
    result = {
        "schema_version": 1,
        "status": "COMPLETE",
        "baseline_mean_accuracy": mean(row["clean_accuracy"] for row in per_seed),
        "baseline_population_sd": pstdev(row["clean_accuracy"] for row in per_seed),
        "reversal_mean_accuracy": mean(row["reversal_accuracy"] for row in per_seed),
        "reversal_population_sd": pstdev(row["reversal_accuracy"] for row in per_seed),
        "mean_damage_pp": mean_damage,
        "seed_level_t_ci95_low_pp": mean_damage - half_width,
        "seed_level_t_ci95_high_pp": mean_damage + half_width,
        "criteria": criteria,
        "confirmed_order_harm": all(criteria.values()),
        "per_seed": per_seed,
        "scope": "same frozen 512-example training schedule and 1024-example confirmation set as the 79.82% Clean baseline",
        "official_test_opened": False,
    }
    atomic_json((ROOT / config["analysis_path"]).resolve(), result)
    return result


def _write_report(config: dict[str, Any], analysis: dict[str, Any]) -> Path:
    lines = [
        "# SIM-CoT 步骤逆序 v14 实验报告",
        "",
        "本实验复用旧 full_conflict 实验的冻结 512 条训练样本、1024 条确认集、初始检查点、三种子和 64-update 调度。",
        "唯一处理是将所有训练样本的五个显式步骤从 [1,2,3,4,5] 逆序为 [5,4,3,2,1]；题目、步骤文本集合、最终答案标签和辅助损失权重均不变。",
        "",
        "| Seed | Clean | 步骤逆序 | Clean-逆序 (pp) | McNemar p |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['clean_accuracy']:.2%} | {row['reversal_accuracy']:.2%} | "
            f"{row['damage_pp']:.2f} | {row['mcnemar']['two_sided_exact_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            f"- Clean 三种子均值：{analysis['baseline_mean_accuracy']:.2%}",
            f"- 步骤逆序三种子均值：{analysis['reversal_mean_accuracy']:.2%}",
            f"- 平均伤害：{analysis['mean_damage_pp']:.2f} pp",
            f"- 种子层 95% t 区间：[{analysis['seed_level_t_ci95_low_pp']:.2f}, {analysis['seed_level_t_ci95_high_pp']:.2f}] pp",
            f"- 顺序伤害判定：{'通过' if analysis['confirmed_order_harm'] else '未通过'}",
            "",
            "注意：准确率来自旧实验冻结的 1024 条确认集，不是 GSM8K 官方测试集。逆序数据是确定性合成的顺序冲突，不是自然教师噪声。",
        ]
    )
    path = (ROOT / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SIM-CoT step-order reversal experiment")
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/step_order_reversal_v14.json",
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    if config["arm"] != STEP_ORDER_REVERSAL_ARM:
        raise ValueError("Configuration arm is not the registered reversal arm")
    try:
        _save_state(config, status="RUNNING", phase="AUDIT", current="data_and_baseline")
        data_audit = _data_audit(config)
        _baseline_audit(config)
        audit_hash = data_audit["audit_sha256"]
        for seed in config["seeds"]:
            key = f"{seed}:{config['arm']}"
            if not _training_complete(config, seed, audit_hash):
                _save_state(config, status="RUNNING", phase="TRAIN", current=key)
                _train(config, seed, audit_hash)
            else:
                print(f"skip completed training {key}", flush=True)
            state = _save_state(config)
            completed = list(state.get("completed_training", []))
            if key not in completed:
                completed.append(key)
                _save_state(config, completed_training=completed)

            if not _evaluation_complete(config, seed, audit_hash):
                _save_state(config, status="RUNNING", phase="EVAL", current=key)
                _evaluate(config, seed, audit_hash)
            else:
                print(f"skip completed evaluation {key}", flush=True)
            state = _save_state(config)
            completed = list(state.get("completed_evaluation", []))
            if key not in completed:
                completed.append(key)
                _save_state(config, completed_evaluation=completed)

        _save_state(config, status="RUNNING", phase="ANALYZE", current="paired_analysis")
        analysis = _analyze(config)
        report = _write_report(config, analysis)
        _save_state(
            config,
            status="PASS",
            phase="COMPLETE",
            current=None,
            report_path=str(report),
            completed_unix=time.time(),
        )
        print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        _save_state(
            config,
            status="FAIL",
            phase="STOPPED",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    main()
