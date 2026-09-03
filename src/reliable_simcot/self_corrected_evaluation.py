from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Callable
import gc
import json
import random
import re
import time

import torch

from .m1_training import atomic_json, sha256_file
from .official_adapter import (
    OfficialExample,
    build_eval_tensors,
    extract_answer_official,
    load_official_model,
)
from .prm800k_data import load_official_grader
from .self_corrected_experiment import (
    ARMS,
    checkpoint_path,
    load_manifest,
    load_training_schedule,
    training_directory,
)


def normalize_answer_for_em(value: str) -> str:
    text = value.strip()
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", text, flags=re.DOTALL)
    if boxed:
        text = boxed.group(1)
    text = text.strip().rstrip(".").strip().strip("$").strip()
    text = text.replace(",", "").replace(r"\!", "")
    return re.sub(r"\s+", "", text)


def _load_existing_predictions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for index, row in enumerate(rows):
        if row.get("idx") != index:
            raise ValueError("Existing MATH predictions are not a contiguous prefix")
    return rows


@torch.inference_mode()
def evaluate_self_corrected_arm(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    project_root: str | Path,
    resume: bool,
) -> dict[str, Any]:
    if arm not in ARMS or int(seed) not in [int(value) for value in config["seeds"]]:
        raise ValueError("Arm or seed is not frozen")
    root = Path(project_root).resolve()
    manifest = load_manifest(config, root)
    schedule = load_training_schedule(config, root)
    metrics_path = training_directory(root, config, seed, arm, sanity=False) / "metrics.json"
    train_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint_path(root, config, seed, arm)
    if (
        train_metrics.get("status") != "PASS"
        or train_metrics.get("schedule_sha256") != schedule["schedule_sha256"]
        or sha256_file(checkpoint) != train_metrics.get("checkpoint_sha256")
    ):
        raise ValueError("Training checkpoint or metrics do not match the frozen run")

    test_entries = manifest["test_entries"]
    dataset_family = str(config.get("dataset_family", "math")).lower()
    if len(test_entries) != int(config["test_examples"]):
        raise ValueError(f"Frozen official {dataset_family.upper()} test set is incomplete")
    output_dir = (
        root / config["output_root"] / "eval" / f"seed_{seed}" / arm
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    existing = _load_existing_predictions(predictions_path) if resume else []
    if predictions_path.exists() and not resume:
        raise FileExistsError(
            f"Predictions already exist at {predictions_path}; use resume=True"
        )
    for index, row in enumerate(existing):
        expected = test_entries[index]
        if (
            row.get("question_id") != expected["problem_id"]
            or row.get("ground_truth") != expected["answer"]
        ):
            raise ValueError("Frozen test data changed since evaluation began")

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
    model.base_causallm.eval()
    model.expainable_llm.eval()
    grade_answer = (
        None
        if dataset_family == "gsm8k"
        else load_official_grader((root / config["prm_repo_dir"]).resolve())
    )
    rows = list(existing)
    started = time.perf_counter()
    mode = "a" if resume else "w"
    with predictions_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for index, entry in enumerate(test_entries):
            if index < len(existing):
                continue
            example = OfficialExample(
                idx=index,
                question=entry["problem"],
                steps=(),
                answer=entry["answer"],
            )
            tensors = build_eval_tensors(
                example,
                tokenizer,
                token_ids,
                latent_tokens=int(config["latent_stage"]) * int(config["c_thought"]),
                device=device,
            )
            generated = model.generate(
                **tensors,
                max_new_tokens=int(config["max_new_tokens"]),
                synced_gpus=False,
            )
            decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
            prediction = extract_answer_official(decoded)
            normalized_em = normalize_answer_for_em(prediction) == normalize_answer_for_em(
                entry["answer"]
            )
            correct = (
                prediction == entry["answer"]
                if dataset_family == "gsm8k"
                else bool(grade_answer(prediction, entry["answer"]))
            )
            row = {
                "idx": index,
                "question_id": entry["problem_id"],
                "subject": entry.get("subject"),
                "level": entry.get("level"),
                "ground_truth": entry["answer"],
                "prediction": prediction,
                "correct": correct,
                "normalized_em": normalized_em,
                "decoded_text": decoded,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if len(rows) == 1 or len(rows) % int(config["flush_every"]) == 0:
                handle.flush()
                print(
                    f"evaluation {seed}:{arm}: {len(rows)}/{len(test_entries)}, "
                    f"accuracy={mean(float(item['correct']) for item in rows):.4f}",
                    flush=True,
                )

    peak = torch.cuda.max_memory_reserved(device) / 1024**3
    within_memory_limit = peak <= float(config["max_reserved_memory_gb"])
    result = {
        "schema_version": 1,
        "run_id": f"SC-EVAL-{seed}-{arm}",
        "status": "PASS" if within_memory_limit else "FAIL",
        "arm": arm,
        "seed": seed,
        "examples": len(rows),
        "correct": sum(bool(row["correct"]) for row in rows),
        "accuracy": mean(float(row["correct"]) for row in rows),
        "normalized_em": mean(float(row["normalized_em"]) for row in rows),
        "dataset_family": dataset_family,
        "ground_truth_source": f"frozen official {dataset_family.upper()} test answer field",
        "grader": (
            "official SIM-CoT GSM8K exact-match evaluator"
            if dataset_family == "gsm8k"
            else "official PRM800K MATH grader"
        ),
        "manifest_sha256": manifest["manifest_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "checkpoint_sha256": train_metrics["checkpoint_sha256"],
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_reserved_gb": peak,
        "memory_limit_gb": float(config["max_reserved_memory_gb"]),
        "within_memory_limit": within_memory_limit,
        "official_test_opened": True,
    }
    atomic_json(output_dir / "metrics.json", result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def hierarchical_paired_bootstrap(
    left: list[list[bool]],
    right: list[list[bool]],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    if not left or len(left) != len(right) or samples < 1:
        raise ValueError("Bootstrap inputs or sample count are invalid")
    question_count = len(left[0])
    if question_count < 1 or any(len(row) != question_count for row in (*left, *right)):
        raise ValueError("Bootstrap vectors must have equal non-zero lengths")
    per_seed_pp = [
        100.0 * mean(float(a) - float(b) for a, b in zip(lrow, rrow, strict=True))
        for lrow, rrow in zip(left, right, strict=True)
    ]
    estimate = mean(per_seed_pp)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _seed_draw in range(len(left)):
            seed_index = rng.randrange(len(left))
            for _question_draw in range(question_count):
                question_index = rng.randrange(question_count)
                total += float(left[seed_index][question_index]) - float(
                    right[seed_index][question_index]
                )
                count += 1
        draws.append(100.0 * total / count)
    draws.sort()
    return {
        "effect_pp": estimate,
        "ci95_low_pp": draws[int(0.025 * samples)],
        "ci95_high_pp": draws[min(samples - 1, int(0.975 * samples))],
        "per_seed_effect_pp": per_seed_pp,
        "positive_seed_count": sum(value > 0 for value in per_seed_pp),
    }


def _prediction_rows(root: Path, config: dict[str, Any], seed: int, arm: str) -> list[dict[str, Any]]:
    path = root / config["output_root"] / "eval" / f"seed_{seed}" / arm / "predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != int(config["test_examples"]):
        raise ValueError(f"Incomplete predictions for {seed}:{arm}")
    return rows


def _effect_decision(effect: dict[str, Any], practical_pp: float) -> dict[str, Any]:
    confirmed = (
        effect["effect_pp"] > 0
        and effect["positive_seed_count"] >= 2
        and effect["ci95_low_pp"] > 0
    )
    return {
        **effect,
        "confirmed": confirmed,
        "practically_meaningful": confirmed and effect["effect_pp"] >= practical_pp,
    }


def analyze_self_corrected(config: dict[str, Any], *, project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    seeds = [int(seed) for seed in config["seeds"]]
    predictions = {
        arm: {
            seed: _prediction_rows(root, config, seed, arm) for seed in seeds
        }
        for arm in ARMS
    }

    def flags(arm: str) -> list[list[bool]]:
        return [[bool(row["correct"]) for row in predictions[arm][seed]] for seed in seeds]

    arm_summaries: dict[str, Any] = {}
    for arm in ARMS:
        accuracies = [mean(float(row["correct"]) for row in predictions[arm][seed]) for seed in seeds]
        ems = [mean(float(row["normalized_em"]) for row in predictions[arm][seed]) for seed in seeds]
        arm_summaries[arm] = {
            "accuracy_values": accuracies,
            "accuracy_mean": mean(accuracies),
            "accuracy_population_std": pstdev(accuracies),
            "normalized_em_values": ems,
            "normalized_em_mean": mean(ems),
        }

    effects: dict[str, Any] = {}
    bootstrap_seed = int(config["bootstrap_seed"])
    samples = int(config["bootstrap_samples"])
    for family in ("solution", "misread"):
        for dose in (1, 2):
            equal_arm = f"{family}_n{dose}_equal"
            weighted_arm = f"{family}_n{dose}_w01"
            damage = hierarchical_paired_bootstrap(
                flags("clean"),
                flags(equal_arm),
                samples=samples,
                seed=bootstrap_seed + dose + (100 if family == "misread" else 0),
            )
            recovery = hierarchical_paired_bootstrap(
                flags(weighted_arm),
                flags(equal_arm),
                samples=samples,
                seed=bootstrap_seed + 10 + dose + (100 if family == "misread" else 0),
            )
            damage_decision = _effect_decision(damage, float(config["harm_practical_pp"]))
            recovery_decision = _effect_decision(
                recovery, float(config["recovery_practical_pp"])
            )
            recovery_fraction = (
                recovery["effect_pp"] / damage["effect_pp"]
                if damage["effect_pp"] > 0
                else None
            )
            recovery_decision["recovery_fraction"] = recovery_fraction
            recovery_decision["practically_meaningful"] = bool(
                recovery_decision["confirmed"]
                and recovery["effect_pp"] >= float(config["recovery_practical_pp"])
                and recovery_fraction is not None
                and recovery_fraction >= float(config["recovery_fraction_threshold"])
            )
            if damage_decision["confirmed"] and recovery_decision["confirmed"]:
                category = "HARM_AND_RECOVERY"
            elif damage_decision["confirmed"]:
                category = "HARM_NO_RECOVERY"
            elif recovery_decision["confirmed"]:
                category = "NO_HARM_WEIGHTING_GAIN"
            else:
                category = "NO_HARM_NO_RECOVERY"
            effects[f"{family}_n{dose}"] = {
                "damage": damage_decision,
                "recovery": recovery_decision,
                "category": category,
            }

        effects[f"{family}_dose_effect"] = hierarchical_paired_bootstrap(
            flags(f"{family}_n1_equal"),
            flags(f"{family}_n2_equal"),
            samples=samples,
            seed=bootstrap_seed + 200 + (100 if family == "misread" else 0),
        )
    for dose in (1, 2):
        effects[f"family_difference_n{dose}"] = hierarchical_paired_bootstrap(
            flags(f"solution_n{dose}_equal"),
            flags(f"misread_n{dose}_equal"),
            samples=samples,
            seed=bootstrap_seed + 400 + dose,
        )

    family_conclusions = {}
    for family in ("solution", "misread"):
        rows = [effects[f"{family}_n{dose}"] for dose in (1, 2)]
        harm = any(row["damage"]["confirmed"] for row in rows)
        recovery = any(
            row["damage"]["confirmed"] and row["recovery"]["confirmed"] for row in rows
        )
        family_conclusions[family] = (
            "HARM_AND_RECOVERY"
            if harm and recovery
            else "HARM_NO_RECOVERY"
            if harm
            else "NO_HARM_WEIGHTING_GAIN"
            if any(row["recovery"]["confirmed"] for row in rows)
            else "NO_HARM_NO_RECOVERY"
        )
    cross_family_same_direction = all(
        family_conclusions[family].startswith("HARM") for family in ("solution", "misread")
    )
    result = {
        "schema_version": 1,
        "status": "PASS",
        "arm_summaries": arm_summaries,
        "effects": effects,
        "family_conclusions": family_conclusions,
        "cross_family_harm_supported": cross_family_same_direction,
        "floor_effect_detected": arm_summaries["clean"]["accuracy_mean"]
        < float(config.get("minimum_clean_accuracy", 0.0)),
        "minimum_clean_accuracy": float(config.get("minimum_clean_accuracy", 0.0)),
        "bootstrap_samples": samples,
        "bootstrap_seed": bootstrap_seed,
        "official_test_opened": True,
        "generator_disclosure": (
            "The errors are Codex-authored controlled semi-synthetic strong conflicts, "
            "not natural teacher-noise samples or a natural-noise frequency estimate."
        ),
    }
    atomic_json((root / config["analysis_path"]).resolve(), result)
    return result


def write_self_corrected_report(
    config: dict[str, Any], *, project_root: str | Path, analysis: dict[str, Any]
) -> Path:
    root = Path(project_root).resolve()
    labels = {
        "clean": "Clean",
        "solution_n1_equal": "解法-N1-等权",
        "solution_n1_w01": "解法-N1-0.1",
        "solution_n2_equal": "解法-N2-等权",
        "solution_n2_w01": "解法-N2-0.1",
        "misread_n1_equal": "误读-N1-等权",
        "misread_n1_w01": "误读-N1-0.1",
        "misread_n2_equal": "误读-N2-等权",
        "misread_n2_w01": "误读-N2-0.1",
    }
    lines = [
        "# SIM-CoT 自纠强冲突完整析因实验报告",
        "",
        "## 准确率",
        "",
        "| 训练臂 | Seed 1 | Seed 2 | Seed 3 | 均值 | 标准化 EM |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        row = analysis["arm_summaries"][arm]
        values = row["accuracy_values"]
        lines.append(
            f"| {labels[arm]} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | "
            f"{row['accuracy_mean']:.4f} | {row['normalized_em_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 伤害与加权恢复",
            "",
            "| 条件 | 伤害 pp [95% CI] | 伤害确认 | 恢复 pp [95% CI] | 恢复确认 | 结论 |",
            "|---|---:|:---:|---:|:---:|---|",
        ]
    )
    for key in ("solution_n1", "solution_n2", "misread_n1", "misread_n2"):
        row = analysis["effects"][key]
        damage = row["damage"]
        recovery = row["recovery"]
        lines.append(
            f"| {key} | {damage['effect_pp']:.2f} [{damage['ci95_low_pp']:.2f}, "
            f"{damage['ci95_high_pp']:.2f}] | {'是' if damage['confirmed'] else '否'} | "
            f"{recovery['effect_pp']:.2f} [{recovery['ci95_low_pp']:.2f}, "
            f"{recovery['ci95_high_pp']:.2f}] | {'是' if recovery['confirmed'] else '否'} | "
            f"{row['category']} |"
        )
    lines.extend(
        [
            "",
            f"- 错误解法族结论：`{analysis['family_conclusions']['solution']}`",
            f"- 题意误读族结论：`{analysis['family_conclusions']['misread']}`",
            f"- 两族伤害方向均获确认：{'是' if analysis['cross_family_harm_supported'] else '否'}",
            f"- Clean 基线地板门：{'未通过' if analysis['floor_effect_detected'] else '通过'} "
            f"（冻结下限 {analysis['minimum_clean_accuracy']:.2%}）。",
            "",
            "本实验使用 Codex 编写的受约束、半合成强冲突，并不测量自然教师噪声的发生率或全部形态。",
        ]
    )
    path = (root / config["report_path"]).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
