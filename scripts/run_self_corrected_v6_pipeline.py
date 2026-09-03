from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
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

from reliable_simcot.m1_training import atomic_json  # noqa: E402
from reliable_simcot.self_corrected_experiment import (  # noqa: E402
    run_full_schedule_memory_gate,
    run_max_length_memory_gate,
    run_sanity_gate,
    run_weight_gradient_audit,
)


# This Windows/WDDM desktop has a measured idle graphics baseline around
# 1.0--1.2 GiB.  This is only an availability check; the frozen per-run
# torch.cuda.max_memory_reserved limit remains 7.4 GiB and is enforced by
# every gate, training arm, and evaluation arm.
PREFLIGHT_MAX_USED_MIB = 1536
PREFLIGHT_POLL_SECONDS = 30


def load_config(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).resolve().read_text(encoding="utf-8"))


def pipeline_state_path(config: dict[str, Any]) -> Path:
    return (ROOT / config["output_root"] / "pipeline_state.json").resolve()


def save_pipeline_state(config: dict[str, Any], **updates: Any) -> None:
    path = pipeline_state_path(config)
    state = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {
            "schema_version": 1,
            "run_id": str(config.get("pipeline_run_id", "SC-MEMORY-SAFE-PIPELINE")),
            "started_unix": time.time(),
        }
    )
    state.update(updates)
    state["updated_unix"] = time.time()
    atomic_json(path, state)


def gpu_memory_used_mib() -> int:
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
    first = completed.stdout.strip().splitlines()[0]
    return int(first.strip())


def wait_for_free_gpu(config: dict[str, Any], next_phase: str) -> None:
    while True:
        used = gpu_memory_used_mib()
        if used <= PREFLIGHT_MAX_USED_MIB:
            save_pipeline_state(
                config,
                status="RUNNING",
                phase=next_phase,
                gpu_memory_used_mib=used,
                preflight_max_used_mib=PREFLIGHT_MAX_USED_MIB,
            )
            print(
                f"GPU preflight passed for {next_phase}: {used} MiB used",
                flush=True,
            )
            return
        save_pipeline_state(
            config,
            status="WAITING_FOR_GPU",
            phase="PREFLIGHT",
            next_phase=next_phase,
            gpu_memory_used_mib=used,
            preflight_max_used_mib=PREFLIGHT_MAX_USED_MIB,
        )
        print(
            f"GPU busy: {used} MiB used; waiting for <= "
            f"{PREFLIGHT_MAX_USED_MIB} MiB before {next_phase}",
            flush=True,
        )
        time.sleep(PREFLIGHT_POLL_SECONDS)


def existing_gate_passed(path: Path) -> bool:
    if not path.is_file():
        return False
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("status") == "PASS" and payload.get("gate_passed") is True


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_gate(
    config: dict[str, Any],
    *,
    phase: str,
    path_key: str,
    function: Callable[..., dict[str, Any]],
) -> None:
    path = (ROOT / config[path_key]).resolve()
    if existing_gate_passed(path):
        print(f"skip passed gate {phase}: {path}", flush=True)
        return
    if path.exists():
        raise RuntimeError(
            f"Existing non-PASS gate artifact requires diagnosis; refusing to overwrite: {path}"
        )
    wait_for_free_gpu(config, phase)
    save_pipeline_state(config, status="RUNNING", phase=phase)
    result = function(config, project_root=ROOT)
    # Gate functions release their model objects before returning.  Empty the
    # allocator once more after those function-local tensor references are
    # gone so the next nvidia-smi preflight observes the true desktop baseline.
    release_cuda_cache()
    if result.get("status") != "PASS" or result.get("gate_passed") is not True:
        raise RuntimeError(f"{phase} failed: {result}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Queue the memory-safe v6 gates and full overnight experiment"
    )
    parser.add_argument(
        "--config",
        default="configs/reliable_simcot/self_corrected_strong_conflict_v6.json",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        run_gate(
            config,
            phase="SANITY",
            path_key="sanity_path",
            function=run_sanity_gate,
        )
        run_gate(
            config,
            phase="GRADIENT_AUDIT",
            path_key="gradient_audit_path",
            function=run_weight_gradient_audit,
        )
        run_gate(
            config,
            phase="MAX_LENGTH_MEMORY_GATE",
            path_key="max_length_memory_gate_path",
            function=run_max_length_memory_gate,
        )
        if config.get("full_schedule_memory_gate_path"):
            run_gate(
                config,
                phase="FULL_SCHEDULE_MEMORY_GATE",
                path_key="full_schedule_memory_gate_path",
                function=run_full_schedule_memory_gate,
            )
        wait_for_free_gpu(config, "OVERNIGHT")
        save_pipeline_state(config, status="RUNNING", phase="OVERNIGHT")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_self_corrected_overnight.py"),
                "--config",
                args.config,
            ],
            cwd=ROOT,
            check=True,
        )
        save_pipeline_state(
            config,
            status="PASS",
            phase="COMPLETE",
            completed_unix=time.time(),
        )
    except BaseException as error:
        save_pipeline_state(
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
