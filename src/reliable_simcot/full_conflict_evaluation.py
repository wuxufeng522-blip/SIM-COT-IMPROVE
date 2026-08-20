from __future__ import annotations

from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any
import gc
import json
import math
import random

import torch

from .causal_evaluation import clean_dev_nll
from .full_conflict_data import load_frozen_schedule
from .full_conflict_experiment import (
    ALL_ARMS,
    MAIN_ARMS,
    CONDITIONAL_ARM,
    checkpoint_path,
    training_directory,
)
from .m1_training import atomic_json, sha256_file
from .official_adapter import (
    OfficialExample,
    evaluate_checkpoint,
    load_official_model,
)
from .oracle_weighting import grouped_auxiliary_loss, tokenize_step_targets
from .single_gpu_smoke import encode_smoke_example, tensorize_smoke_example


@torch.inference_mode()
def full_chain_preference_nll(
    model,
    tokenizer,
    token_ids: dict[str, int],
    config: dict[str, Any],
    *,
    schedule: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    entries = [row for row in schedule["train_entries"] if row["coverage_tier"] in {0, 1}]
    if len(entries) != int(config["preference_examples"]):
        raise ValueError("Frozen full-chain preference set is incomplete")
    clean_sum = corrupt_sum = 0.0
    clean_tokens = corrupt_tokens = 0
    model.base_causallm.eval()
    model.expainable_llm.eval()
    for number, entry in enumerate(entries, start=1):
        example = OfficialExample(
            int(entry["source_idx"]),
            entry["question"],
            tuple(entry["clean_steps"]),
            entry["answer"],
        )
        encoded = encode_smoke_example(
            example,
            tokenizer,
            token_ids,
            latent_stage=int(config["latent_stage"]),
            c_thought=int(config["c_thought"]),
        )
        batch = tensorize_smoke_example(encoded, device=device)
        clean_targets = tokenize_step_targets(tokenizer, example.steps)
        corrupt_targets = tokenize_step_targets(tokenizer, entry["full_chain"]["steps"])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            clean = grouped_auxiliary_loss(
                model,
                batch,
                clean_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=int(config["c_thought"]),
            )
            corrupt = grouped_auxiliary_loss(
                model,
                batch,
                corrupt_targets,
                (1.0,) * 5,
                latent_id=token_ids["<|latent|>"],
                c_thought=int(config["c_thought"]),
            )
        clean_sum += sum(clean["step_losses"].float().tolist())
        corrupt_sum += sum(corrupt["step_losses"].float().tolist())
        clean_tokens += sum(clean["token_counts"])
        corrupt_tokens += sum(corrupt["token_counts"])
        if number == 1 or number % int(config["nll_log_every"]) == 0 or number == len(entries):
            print(f"full-chain preference: {number}/{len(entries)}", flush=True)
    clean_nll = clean_sum / clean_tokens
    corrupt_nll = corrupt_sum / corrupt_tokens
    return {
        "examples": len(entries),
        "clean_step_token_nll": clean_nll,
        "full_conflict_step_token_nll": corrupt_nll,
        "full_minus_clean_nll": corrupt_nll - clean_nll,
        "interpretation": "positive means the model assigns lower NLL to the clean chain",
    }


def evaluate_full_conflict_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    resume: bool = False,
) -> dict[str, Any]:
    if arm not in ALL_ARMS or seed not in config["seeds"]:
        raise ValueError("Arm or seed is not preregistered")
    root = Path(project_root).resolve()
    schedule = load_frozen_schedule(config, root)
    train_path = training_directory(root, config, seed, arm) / "metrics.json"
    train_metrics = json.loads(train_path.read_text(encoding="utf-8"))
    if train_metrics.get("status") != "PASS" or train_metrics.get("schedule_sha256") != schedule[
        "schedule_sha256"
    ]:
        raise ValueError("Training metrics did not pass or use the frozen schedule")
    checkpoint = checkpoint_path(root, config, seed, arm)
    if sha256_file(checkpoint) != train_metrics["checkpoint_sha256"]:
        raise ValueError("Trained checkpoint SHA-256 mismatch")
    confirm_path = (root / config["confirm_dataset_path"]).resolve()
    device = torch.device(config["device"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model, tokenizer, token_ids = load_official_model(
        official_coconut_dir=(root / config["official_source_dir"]).resolve(),
        base_model_dir=(root / config["base_model_dir"]).resolve(),
        checkpoint_path=checkpoint,
        device=device,
        dtype=torch.float32,
        move_auxiliary_to_device=True,
    )
    clean_nll = clean_dev_nll(
        model,
        tokenizer,
        token_ids,
        {**config, "dev_examples": int(config["confirm_examples"])},
        dev_path=confirm_path,
        device=device,
    )
    preference = full_chain_preference_nll(
        model,
        tokenizer,
        token_ids,
        config,
        schedule=schedule,
        device=device,
    )
    output_dir = (root / config["output_root"] / "eval" / f"seed_{seed}" / arm).resolve()
    metrics = evaluate_checkpoint(
        model=model,
        tokenizer=tokenizer,
        token_ids=token_ids,
        dataset_path=confirm_path,
        output_dir=output_dir,
        device=device,
        latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
        max_new_tokens=int(config["max_new_tokens"]),
        expected_accuracy=0.0,
        accuracy_tolerance=1.0,
        resume=resume,
        flush_every=int(config["flush_every"]),
    )
    metrics.update(
        {
            "run_id": f"FC2-{seed}-{arm}",
            "arm": arm,
            "seed": seed,
            "training_run_id": train_metrics["run_id"],
            "checkpoint_sha256": train_metrics["checkpoint_sha256"],
            "schedule_sha256": schedule["schedule_sha256"],
            "confirm_dataset_sha256": sha256_file(confirm_path),
            "clean_nll": clean_nll,
            "chain_preference": preference,
            "official_test_opened": False,
        }
    )
    atomic_json(output_dir / "metrics.json", metrics)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


def _prediction_flags(path: Path) -> list[bool]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [bool(row["correct"]) for row in rows]


def exact_mcnemar(left: list[bool], right: list[bool]) -> dict[str, Any]:
    if len(left) != len(right) or not left:
        raise ValueError("McNemar inputs must be non-empty paired vectors")
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower = min(left_only, right_only)
        tail = sum(math.comb(discordant, k) for k in range(lower + 1)) / 2**discordant
        p_value = min(1.0, 2.0 * tail)
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "discordant": discordant,
        "two_sided_exact_p": p_value,
    }


def paired_bootstrap_damage_pp(
    clean: list[bool], treatment: list[bool], *, samples: int, seed: int
) -> dict[str, float]:
    if len(clean) != len(treatment) or not clean:
        raise ValueError("Bootstrap inputs must be non-empty paired vectors")
    rng = random.Random(seed)
    differences = [float(a) - float(b) for a, b in zip(clean, treatment, strict=True)]
    draws = []
    for _ in range(samples):
        draws.append(
            100.0
            * sum(differences[rng.randrange(len(differences))] for _ in differences)
            / len(differences)
        )
    draws.sort()
    return {
        "damage_pp": 100.0 * sum(differences) / len(differences),
        "ci95_low_pp": draws[int(0.025 * samples)],
        "ci95_high_pp": draws[min(samples - 1, int(0.975 * samples))],
    }


def _read_eval(root: Path, config: dict[str, Any], seed: int, arm: str) -> dict[str, Any]:
    path = root / config["output_root"] / "eval" / f"seed_{seed}" / arm / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "accuracy": [float(row["accuracy"]) for row in rows],
        "answer_nll": [float(row["clean_nll"]["answer_nll"]) for row in rows],
        "clean_step_token_nll": [float(row["clean_nll"]["step_token_nll"]) for row in rows],
        "full_minus_clean_nll": [
            float(row["chain_preference"]["full_minus_clean_nll"]) for row in rows
        ],
    }
    return {
        key: {"values": values, "mean": mean(values), "population_std": pstdev(values)}
        for key, values in fields.items()
    }


def analyze_25(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows = {
        arm: {seed: _read_eval(root, config, seed, arm) for seed in config["seeds"]}
        for arm in MAIN_ARMS
    }
    per_seed: list[dict[str, Any]] = []
    for seed in config["seeds"]:
        accuracy = {arm: float(rows[arm][seed]["accuracy"]) for arm in MAIN_ARMS}
        clean_flags = _prediction_flags(
            root / config["output_root"] / "eval" / f"seed_{seed}" / "clean_aux1" / "predictions.jsonl"
        )
        full_flags = _prediction_flags(
            root / config["output_root"] / "eval" / f"seed_{seed}" / "full_conflict_25" / "predictions.jsonl"
        )
        local_flags = _prediction_flags(
            root / config["output_root"] / "eval" / f"seed_{seed}" / "local_causal_25" / "predictions.jsonl"
        )
        per_seed.append(
            {
                "seed": seed,
                "accuracy": accuracy,
                "damage_full25_pp": 100 * (accuracy["clean_aux1"] - accuracy["full_conflict_25"]),
                "damage_local25_pp": 100 * (accuracy["clean_aux1"] - accuracy["local_causal_25"]),
                "mcnemar_clean_vs_full25": exact_mcnemar(clean_flags, full_flags),
                "mcnemar_clean_vs_local25": exact_mcnemar(clean_flags, local_flags),
                "bootstrap_clean_vs_full25": paired_bootstrap_damage_pp(
                    clean_flags,
                    full_flags,
                    samples=int(config["bootstrap_samples"]),
                    seed=int(seed),
                ),
            }
        )
    damage_full = mean(row["damage_full25_pp"] for row in per_seed)
    damage_local = mean(row["damage_local25_pp"] for row in per_seed)
    data_audit = json.loads((root / config["data_audit_path"]).read_text(encoding="utf-8"))
    sanity = json.loads((root / config["sanity_gate_path"]).read_text(encoding="utf-8"))
    gradient = json.loads((root / config["gradient_audit_path"]).read_text(encoding="utf-8"))
    criteria = {
        "mean_full25_damage_at_least_5pp": damage_full >= float(config["full25_min_damage_pp"]),
        "all_three_seeds_harmed": all(row["damage_full25_pp"] > 0 for row in per_seed),
        "increment_over_local_at_least_2pp": damage_full - damage_local
        >= float(config["full25_min_increment_over_local_pp"]),
        "all_engineering_audits_passed": data_audit.get("status") == "PASS"
        and sanity.get("gate_passed") is True
        and gradient.get("gate_passed") is True,
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "run_id": "FC300",
        "status": "PASS",
        "gate_passed": passed,
        "verdict": "PROCEED_TO_ORACLE_DESIGN" if passed else "UNLOCK_FULL_CONFLICT_50",
        "mean_damage_full25_pp": damage_full,
        "mean_damage_local25_pp": damage_local,
        "increment_over_local_pp": damage_full - damage_local,
        "criteria": criteria,
        "arm_summaries": {
            arm: _arm_summary([rows[arm][seed] for seed in config["seeds"]])
            for arm in MAIN_ARMS
        },
        "per_seed": per_seed,
        "official_test_opened": False,
    }
    atomic_json((root / config["analysis_25_path"]).resolve(), result)
    return result


def analyze_50(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    per_seed = []
    for seed in config["seeds"]:
        clean = _read_eval(root, config, seed, "clean_aux1")
        full = _read_eval(root, config, seed, CONDITIONAL_ARM)
        damage = 100 * (float(clean["accuracy"]) - float(full["accuracy"]))
        per_seed.append(
            {
                "seed": seed,
                "clean_accuracy": clean["accuracy"],
                "full50_accuracy": full["accuracy"],
                "damage_full50_pp": damage,
            }
        )
    mean_damage = mean(row["damage_full50_pp"] for row in per_seed)
    criteria = {
        "mean_full50_damage_at_least_5pp": mean_damage >= float(config["full50_min_damage_pp"]),
        "all_three_seeds_harmed": all(row["damage_full50_pp"] > 0 for row in per_seed),
    }
    passed = all(criteria.values())
    result = {
        "schema_version": 1,
        "run_id": "FC500",
        "status": "PASS",
        "gate_passed": passed,
        "verdict": "HIGH_DENSITY_ONLY" if passed else "STOP_WEIGHTING_DIRECTION",
        "mean_damage_full50_pp": mean_damage,
        "criteria": criteria,
        "per_seed": per_seed,
        "official_test_opened": False,
    }
    atomic_json((root / config["analysis_50_path"]).resolve(), result)
    return result


def write_chinese_report(
    config: dict[str, Any], *, project_root: str | Path, final: dict[str, Any]
) -> Path:
    root = Path(project_root).resolve()
    analysis25 = json.loads((root / config["analysis_25_path"]).read_text(encoding="utf-8"))
    lines = [
        "# SIM-CoT 全链强冲突一夜实验报告",
        "",
        f"最终判定：`{final['verdict']}`",
        "",
        "## 25% 主阶段",
        "",
        "| Seed | Clean EM | Local25 EM | Full25 EM | Clean-Full25 (pp) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in analysis25["per_seed"]:
        acc = row["accuracy"]
        lines.append(
            f"| {row['seed']} | {acc['clean_aux1']:.4f} | {acc['local_causal_25']:.4f} | "
            f"{acc['full_conflict_25']:.4f} | {row['damage_full25_pp']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"- 三种子平均 Full25 伤害：{analysis25['mean_damage_full25_pp']:.2f} pp",
            f"- 三种子平均 Local25 伤害：{analysis25['mean_damage_local25_pp']:.2f} pp",
            f"- Full25 相对 Local25 的增量伤害：{analysis25['increment_over_local_pp']:.2f} pp",
            f"- 25% 预注册门：{'通过' if analysis25['gate_passed'] else '未通过'}",
            "",
            "官方测试集未打开。错误数据是当前 Codex 任务编写的受约束强反事实压力处理，不能称为自然教师噪声。",
        ]
    )
    if (root / config["analysis_50_path"]).exists():
        analysis50 = json.loads((root / config["analysis_50_path"]).read_text(encoding="utf-8"))
        lines.extend(
            [
                "",
                "## 条件式 50% 阶段",
                "",
                f"三种子平均 Full50 伤害：{analysis50['mean_damage_full50_pp']:.2f} pp；"
                f"预注册门：{'通过' if analysis50['gate_passed'] else '未通过'}。",
            ]
        )
    path = (root / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
