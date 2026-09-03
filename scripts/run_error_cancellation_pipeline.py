from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import argparse
import json
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reliable_simcot.error_cancellation_evaluation import (  # noqa: E402
    analyze_stage1,
    analyze_stage2,
    clean_gate,
    evaluate_arm,
    evaluate_base,
)
from reliable_simcot.error_cancellation_experiment import (  # noqa: E402
    STAGE1_ARMS,
    STAGE2_PRIMARY_ARMS,
    checkpoint_path,
    load_manifest,
    load_schedule,
    run_equivalence_audit,
    run_memory_gate,
    run_sanity_gate,
    run_training_arm,
    training_directory,
)
from reliable_simcot.m1_training import atomic_json, sha256_file  # noqa: E402


def load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def state_path(config: dict[str, Any]) -> Path:
    return (ROOT / config["state_path"]).resolve()


def save_state(config: dict[str, Any], **updates: Any) -> None:
    path = state_path(config)
    state = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else {
            "schema_version": 1,
            "run_id": config["pipeline_run_id"],
            "started_unix": time.time(),
        }
    )
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)


def gpu_used_mib() -> int:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def wait_for_gpu(config: dict[str, Any], phase: str) -> None:
    limit = int(config["preflight_max_used_mib"])
    while True:
        used = gpu_used_mib()
        if used <= limit:
            save_state(
                config,
                status="RUNNING",
                phase=phase,
                gpu_memory_used_mib=used,
                preflight_max_used_mib=limit,
            )
            print(f"GPU preflight {phase}: {used} MiB <= {limit} MiB", flush=True)
            return
        save_state(
            config,
            status="WAITING_FOR_GPU",
            phase="PREFLIGHT",
            next_phase=phase,
            gpu_memory_used_mib=used,
            preflight_max_used_mib=limit,
        )
        print(f"GPU busy before {phase}: {used} MiB; waiting", flush=True)
        time.sleep(30)


def passed_artifact(path: Path) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "PASS" and payload.get("gate_passed") is True


def run_gate(
    config: dict[str, Any],
    phase: str,
    path_key: str,
    function: Callable[..., dict[str, Any]],
) -> None:
    path = (ROOT / config[path_key]).resolve()
    if passed_artifact(path):
        print(f"skip passed {phase}", flush=True)
        return
    if path.exists():
        raise RuntimeError(f"Existing non-PASS gate requires diagnosis: {path}")
    wait_for_gpu(config, phase)
    result = function(config, project_root=ROOT)
    if result.get("status") != "PASS" or not result.get("gate_passed"):
        raise RuntimeError(f"{phase} failed: {result}")


def training_complete(config: dict[str, Any], seed: int, arm: str) -> bool:
    metrics_path = training_directory(ROOT, config, seed, arm) / "metrics.json"
    checkpoint = checkpoint_path(ROOT, config, seed, arm)
    if not metrics_path.exists() or not checkpoint.exists():
        return False
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return (
        metrics.get("status") == "PASS"
        and metrics.get("schedule_sha256") == load_schedule(config, ROOT)["schedule_sha256"]
        and sha256_file(checkpoint) == metrics.get("checkpoint_sha256")
    )


def evaluation_complete(config: dict[str, Any], seed: int, arm: str) -> bool:
    path = ROOT / config["output_root"] / "eval" / f"seed_{seed}" / arm / "metrics.json"
    if not path.exists():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "PASS" and payload.get("examples") == int(
        config["test_examples"]
    )


def run_queue(config: dict[str, Any], arms: tuple[str, ...]) -> None:
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        for arm in arms:
            key = f"{seed}:{arm}"
            if not training_complete(config, seed, arm):
                wait_for_gpu(config, f"TRAIN:{key}")
                save_state(config, status="RUNNING", phase="TRAIN", current=key)
                run_training_arm(config, arm=arm, seed=seed, project_root=ROOT)
            if not evaluation_complete(config, seed, arm):
                wait_for_gpu(config, f"EVAL:{key}")
                save_state(config, status="RUNNING", phase="EVAL", current=key)
                evaluate_arm(config, arm=arm, seed=seed, project_root=ROOT)


def write_report(config: dict[str, Any], analysis: dict[str, Any]) -> None:
    lines = [
        f"# {config['experiment_name']} 错误抵消实验报告",
        "",
        "本实验使用 Codex 编写的确定性受控半合成冲突，不是自然教师噪声。",
        "",
        f"状态：{analysis['status']}",
        f"下一步：{analysis['next_action']}",
        "",
        "## 第一阶段准确率",
        "",
        "| 臂 | 三种子均值 | 标准差 |",
        "|---|---:|---:|",
    ]
    for arm, row in analysis["accuracies"].items():
        lines.append(f"| {arm} | {100*row['mean']:.2f}% | {100*row['population_sd']:.2f}pp |")
    primary = analysis["comparisons"]["primary_wide50"]
    lines.extend(
        [
            "",
            "## 预注册主比较",
            "",
            f"RW50−EW50：{primary['effect_pp']:.2f}pp，"
            f"95% CI [{primary['ci95_low_pp']:.2f}, {primary['ci95_high_pp']:.2f}]；"
            f"伤害门：{'PASS' if analysis['harm_gate_passed'] else 'FAIL'}。",
        ]
    )
    target = (ROOT / config["report_path"]).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/error_cancellation_gsm8k_v10.json",
    )
    parser.add_argument(
        "--continue-after-clean-gate-failure",
        action="store_true",
        help=(
            "Run non-clean arms as an explicitly exploratory deviation while "
            "preserving the failed preregistered Clean gate artifact."
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        data_audit = json.loads((ROOT / config["data_audit_path"]).read_text(encoding="utf-8"))
        if data_audit.get("status") != "PASS":
            raise RuntimeError("Configured data gate has not passed")
        manifest = load_manifest(config, ROOT)
        schedule = load_schedule(config, ROOT)
        save_state(
            config,
            status="RUNNING",
            phase="GATES",
            manifest_sha256=manifest["manifest_sha256"],
            schedule_sha256=schedule["schedule_sha256"],
        )
        run_gate(config, "EQUIVALENCE_AUDIT", "equivalence_audit_path", run_equivalence_audit)
        run_gate(config, "SANITY", "sanity_path", run_sanity_gate)
        run_gate(config, "FULL_SCHEDULE_MEMORY", "memory_gate_path", run_memory_gate)

        base_metrics = ROOT / config["base_eval_dir"] / "metrics.json"
        if not base_metrics.exists():
            wait_for_gpu(config, "BASE_EVAL")
            save_state(config, status="RUNNING", phase="BASE_EVAL")
            result = evaluate_base(config, project_root=ROOT)
            if result["status"] != "PASS":
                raise RuntimeError("Base evaluation exceeded the memory gate")

        run_queue(config, ("C",))
        clean = clean_gate(config, project_root=ROOT)
        if not clean["gate_passed"]:
            if not args.continue_after_clean_gate_failure:
                save_state(config, status="STOPPED", phase="CLEAN_GATE_FAILED", clean_gate=clean)
                print("Clean gate failed; non-clean arms were not started", flush=True)
                return
            save_state(
                config,
                status="RUNNING_EXPLORATORY",
                phase="EXPLORATORY_AFTER_CLEAN_GATE_FAILURE",
                clean_gate=clean,
                preregistered_clean_gate_deviation=True,
                deviation_authorization=(
                    "User explicitly requested continuation after the frozen "
                    "pipeline stopped at the Clean gate."
                ),
            )
            print(
                "Clean gate failed; continuing only as an explicitly exploratory run",
                flush=True,
            )

        run_queue(config, tuple(arm for arm in STAGE1_ARMS if arm != "C"))
        analysis = analyze_stage1(config, project_root=ROOT)
        analysis["clean_gate"] = clean
        analysis["preregistered_clean_gate_deviation"] = not clean["gate_passed"]
        analysis["interpretation_scope"] = config.get(
            "interpretation_scope",
            "confirmatory" if clean["gate_passed"] else "exploratory_after_failed_clean_gate",
        )
        atomic_json(ROOT / config["analysis_path"], analysis)
        if analysis["harm_gate_passed"]:
            run_queue(config, STAGE2_PRIMARY_ARMS)
            analysis = analyze_stage2(config, project_root=ROOT)
        else:
            analysis["status"] = "COMPLETE"
            atomic_json(ROOT / config["analysis_path"], analysis)
        write_report(config, analysis)
        save_state(
            config,
            status="PASS" if clean["gate_passed"] else "PASS_EXPLORATORY",
            phase="COMPLETE",
            completed_unix=time.time(),
            preregistered_clean_gate_deviation=not clean["gate_passed"],
        )
    except BaseException as error:
        save_state(
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
