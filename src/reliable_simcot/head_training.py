from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import math
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .audit import sha256_file
from .calibration import apply_temperature, fit_temperature
from .corruptions import DEVELOPMENT_FAMILIES
from .detection_metrics import evaluate_detection, mean_equivalent_reliability_drop
from .reliability_head import DualReliabilityHead, paired_margin_loss


@dataclass(frozen=True)
class FeatureTable:
    rows: list[dict[str, Any]]
    z_previous: torch.Tensor
    z_current: torch.Tensor
    e_step: torch.Tensor

    def __post_init__(self) -> None:
        count = len(self.rows)
        if not (
            self.z_previous.shape[0]
            == self.z_current.shape[0]
            == self.e_step.shape[0]
            == count
        ):
            raise ValueError("Feature rows and tensors must align")


def load_feature_table(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str,
) -> FeatureTable:
    manifest_path = Path(manifest_path)
    if sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("Feature manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gate_passed") is not True or manifest.get("mode") != "full":
        raise ValueError("A passing full feature cache is required")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for descriptor in manifest["shards"]:
        path = Path(descriptor["path"])
        if sha256_file(path) != descriptor["sha256"]:
            raise ValueError(f"Feature shard SHA-256 mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        if len(shard) != descriptor["rows"]:
            raise ValueError(f"Feature shard row count mismatch: {path}")
        for record in shard:
            if record["variant_id"] in seen:
                raise ValueError("Duplicate variant ID in feature cache")
            seen.add(record["variant_id"])
            records.append(record)
    if len(records) != manifest["rows"]:
        raise ValueError("Feature manifest total row count mismatch")
    rows = [
        {key: value for key, value in record.items() if not torch.is_tensor(value)}
        for record in records
    ]
    return FeatureTable(
        rows=rows,
        z_previous=torch.stack([record["z_previous"] for record in records]),
        z_current=torch.stack([record["z_current"] for record in records]),
        e_step=torch.stack([record["e_step"] for record in records]),
    )


def balanced_classification_indices(
    rows: list[dict[str, Any]],
    eligible_indices: list[int],
    *,
    batch_size: int,
    rng: random.Random,
) -> list[int]:
    if batch_size <= 0 or batch_size % 4 != 0:
        raise ValueError("Classification batch size must be a positive multiple of four")
    invalid = [index for index in eligible_indices if rows[index]["y_valid"] == 0]
    useful = [
        index
        for index in eligible_indices
        if rows[index]["y_valid"] == 1 and rows[index]["y_utility"] == 1
    ]
    useless = [
        index
        for index in eligible_indices
        if rows[index]["y_valid"] == 1 and rows[index]["y_utility"] == 0
    ]
    if not invalid or not useful or not useless:
        raise ValueError("Balanced batches require invalid, useful, and useless groups")
    indices = (
        rng.choices(invalid, k=batch_size // 2)
        + rng.choices(useful, k=batch_size // 4)
        + rng.choices(useless, k=batch_size // 4)
    )
    rng.shuffle(indices)
    return indices


def paired_indices(
    rows: list[dict[str, Any]],
    eligible_indices: Iterable[int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    eligible = list(eligible_indices)
    clean_by_pair = {
        rows[index]["pair_id"]: index
        for index in eligible
        if rows[index]["family"] == "clean_original"
    }
    validity_pairs: list[tuple[int, int]] = []
    utility_pairs: list[tuple[int, int]] = []
    for index in eligible:
        row = rows[index]
        clean = clean_by_pair.get(row["pair_id"])
        if clean is None or clean == index:
            continue
        if row["y_valid"] == 0:
            validity_pairs.append((clean, index))
        elif row["y_utility"] == 0:
            utility_pairs.append((clean, index))
    if not validity_pairs or not utility_pairs:
        raise ValueError("Both validity and utility ranking pairs are required")
    return validity_pairs, utility_pairs


def _features(
    table: FeatureTable,
    indices: list[int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    index = torch.tensor(indices, dtype=torch.long)
    return (
        table.z_previous[index].to(device=device, dtype=torch.float32),
        table.z_current[index].to(device=device, dtype=torch.float32),
        table.e_step[index].to(device=device, dtype=torch.float32),
    )


def _predict(
    model: DualReliabilityHead,
    table: FeatureTable,
    indices: list[int],
    *,
    device: torch.device,
    batch_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    validity: list[np.ndarray] = []
    utility: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            output = model(*_features(table, batch_indices, device))
            validity.append(output.validity_logits.cpu().numpy())
            utility.append(output.utility_logits.cpu().numpy())
    return np.concatenate(validity), np.concatenate(utility)


def _sigmoid_numpy(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return sha256_file(path)


def train_lofo_fold(
    table: FeatureTable,
    *,
    heldout_family: str,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    if heldout_family not in DEVELOPMENT_FAMILIES:
        raise ValueError(f"Unknown held-out family: {heldout_family}")
    allowed_families = {
        "clean_original",
        "equivalent_positive",
        *[family for family in DEVELOPMENT_FAMILIES if family != heldout_family],
    }
    train_indices = [
        index
        for index, row in enumerate(table.rows)
        if row["split"] == "head_train" and row["family"] in allowed_families
    ]
    validation_indices = [
        index
        for index, row in enumerate(table.rows)
        if row["split"] == "head_validation" and row["family"] in allowed_families
    ]
    heldout_indices = [
        index
        for index, row in enumerate(table.rows)
        if row["split"] == "head_audit" and row["family"] == heldout_family
    ]
    heldout_pairs = {table.rows[index]["pair_id"] for index in heldout_indices}
    audit_indices = [
        index
        for index, row in enumerate(table.rows)
        if row["split"] == "head_audit"
        and (
            row["family"] == heldout_family
            or (row["family"] == "clean_original" and row["pair_id"] in heldout_pairs)
        )
    ]
    equivalent_indices = [
        index
        for index, row in enumerate(table.rows)
        if row["split"] == "head_audit"
        and row["family"] in {"clean_original", "equivalent_positive"}
    ]
    if not train_indices or not validation_indices or not audit_indices:
        raise ValueError("LOFO train/validation/audit sets must be non-empty")
    validity_pairs, utility_pairs = paired_indices(table.rows, train_indices)

    seed = config["seed"] + DEVELOPMENT_FAMILIES.index(heldout_family)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(config["device"])
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = DualReliabilityHead(
        hidden_size=table.z_current.shape[-1],
        projection_dim=config["projection_dim"],
        shared_hidden_dim=config["shared_hidden_dim"],
    ).to(device)
    if model.parameter_count > config["max_parameters"]:
        raise ValueError("Reliability head exceeds the registered parameter budget")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    best_auc = -math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = config["patience"]
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(config["epochs"]):
        model.train()
        running: list[float] = []
        for _ in range(config["steps_per_epoch"]):
            classification = balanced_classification_indices(
                table.rows,
                train_indices,
                batch_size=config["classification_batch_size"],
                rng=rng,
            )
            sampled_validity_pairs = rng.choices(
                validity_pairs, k=config["pair_batch_size"]
            )
            sampled_utility_pairs = rng.choices(
                utility_pairs, k=config["pair_batch_size"]
            )
            optimizer.zero_grad(set_to_none=True)
            classification_output = model(*_features(table, classification, device))
            y_valid = torch.tensor(
                [table.rows[index]["y_valid"] for index in classification],
                dtype=torch.float32,
                device=device,
            )
            utility_values = [
                -1 if table.rows[index]["y_utility"] is None else table.rows[index]["y_utility"]
                for index in classification
            ]
            y_utility = torch.tensor(utility_values, dtype=torch.float32, device=device)
            validity_bce = F.binary_cross_entropy_with_logits(
                classification_output.validity_logits, y_valid
            )
            valid_mask = y_valid.bool()
            utility_bce = F.binary_cross_entropy_with_logits(
                classification_output.utility_logits[valid_mask],
                y_utility[valid_mask],
            )

            validity_flat = [index for pair in sampled_validity_pairs for index in pair]
            validity_output = model(*_features(table, validity_flat, device))
            pair_positions = torch.arange(
                config["pair_batch_size"], device=device, dtype=torch.long
            )
            validity_rank = paired_margin_loss(
                validity_output.validity_probability,
                pair_positions * 2,
                pair_positions * 2 + 1,
                margin=config["margin"],
            )
            utility_flat = [index for pair in sampled_utility_pairs for index in pair]
            utility_output = model(*_features(table, utility_flat, device))
            utility_rank = paired_margin_loss(
                utility_output.utility_probability,
                pair_positions * 2,
                pair_positions * 2 + 1,
                margin=config["margin"],
            )
            loss = validity_bce + utility_bce + 0.5 * validity_rank + 0.5 * utility_rank
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite reliability-head loss")
            loss.backward()
            optimizer.step()
            running.append(float(loss.detach().cpu().item()))

        validation_v_logits, validation_u_logits = _predict(
            model, table, validation_indices, device=device
        )
        validation_rows = [table.rows[index] for index in validation_indices]
        validation_metrics = evaluate_detection(
            validation_rows,
            validity_scores=_sigmoid_numpy(validation_v_logits),
            utility_scores=_sigmoid_numpy(validation_u_logits),
        )
        validation_auc = validation_metrics["reliability"]["roc_auc"]
        if validation_auc is None:
            raise ValueError("Validation reliability AUC is undefined")
        history.append(
            {
                "epoch": epoch + 1,
                "mean_train_loss": float(np.mean(running)),
                "validation_reliability_auc": validation_auc,
            }
        )
        if validation_auc > best_auc + config["minimum_improvement"]:
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_state = deepcopy(model.state_dict())
            patience_left = config["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is None:
        raise RuntimeError("LOFO training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_v_logits, validation_u_logits = _predict(
        model, table, validation_indices, device=device
    )
    validation_rows = [table.rows[index] for index in validation_indices]
    validity_targets = torch.tensor(
        [row["y_valid"] for row in validation_rows], dtype=torch.float32
    )
    valid_mask = validity_targets.bool().numpy()
    utility_targets = torch.tensor(
        [row["y_utility"] for row in validation_rows if row["y_valid"] == 1],
        dtype=torch.float32,
    )
    validity_calibration = fit_temperature(
        torch.from_numpy(validation_v_logits), validity_targets
    )
    utility_calibration = fit_temperature(
        torch.from_numpy(validation_u_logits[valid_mask]), utility_targets
    )

    audit_v_logits, audit_u_logits = _predict(model, table, audit_indices, device=device)
    audit_v = _sigmoid_numpy(
        apply_temperature(
            torch.from_numpy(audit_v_logits), validity_calibration.temperature
        ).numpy()
    )
    audit_u = _sigmoid_numpy(
        apply_temperature(
            torch.from_numpy(audit_u_logits), utility_calibration.temperature
        ).numpy()
    )
    audit_rows = [table.rows[index] for index in audit_indices]
    audit_metrics = evaluate_detection(
        audit_rows,
        validity_scores=audit_v,
        utility_scores=audit_u,
    )

    equivalent_v_logits, equivalent_u_logits = _predict(
        model, table, equivalent_indices, device=device
    )
    equivalent_v = _sigmoid_numpy(
        apply_temperature(
            torch.from_numpy(equivalent_v_logits), validity_calibration.temperature
        ).numpy()
    )
    equivalent_u = _sigmoid_numpy(
        apply_temperature(
            torch.from_numpy(equivalent_u_logits), utility_calibration.temperature
        ).numpy()
    )
    equivalent_rows = [table.rows[index] for index in equivalent_indices]
    equivalent_drop = mean_equivalent_reliability_drop(
        equivalent_rows, equivalent_v * equivalent_u
    )

    checkpoint_path = output_dir / f"lofo_{heldout_family}.pt"
    checkpoint_sha = _save_checkpoint(
        checkpoint_path,
        {
            "heldout_family": heldout_family,
            "seed": seed,
            "model_config": {
                "hidden_size": model.hidden_size,
                "projection_dim": model.projection_dim,
                "shared_hidden_dim": model.shared_hidden_dim,
            },
            "state_dict": best_state,
            "validity_temperature": validity_calibration.temperature,
            "utility_temperature": utility_calibration.temperature,
        },
    )
    reloaded_payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    reloaded_model = DualReliabilityHead(**reloaded_payload["model_config"]).to(device)
    reloaded_model.load_state_dict(reloaded_payload["state_dict"])
    probe_indices = audit_indices[: min(64, len(audit_indices))]
    original_probe = _predict(model, table, probe_indices, device=device)
    reloaded_probe = _predict(reloaded_model, table, probe_indices, device=device)
    reload_max_abs_delta = max(
        float(np.max(np.abs(original - reloaded)))
        for original, reloaded in zip(original_probe, reloaded_probe, strict=True)
    )
    if reload_max_abs_delta > 1e-6:
        raise ValueError("Reloaded reliability head does not reproduce fixed-input logits")
    predictions_path = output_dir / f"lofo_{heldout_family}_predictions.jsonl"
    predictions_path.write_text(
        "".join(
            json.dumps(
                {
                    "variant_id": row["variant_id"],
                    "pair_id": row["pair_id"],
                    "family": row["family"],
                    "y_valid": row["y_valid"],
                    "y_utility": row["y_utility"],
                    "validity_probability": float(v_score),
                    "utility_probability": float(u_score),
                    "reliability_probability": float(v_score * u_score),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row, v_score, u_score in zip(audit_rows, audit_v, audit_u, strict=True)
        ),
        encoding="utf-8",
    )
    peak_reserved_gb = (
        torch.cuda.max_memory_reserved(device) / (1024**3)
        if device.type == "cuda"
        else 0.0
    )
    within_memory_limit = peak_reserved_gb <= config["max_reserved_memory_gb"]
    if not within_memory_limit:
        raise MemoryError("Reliability-head run exceeded the registered memory ceiling")
    return {
        "heldout_family": heldout_family,
        "seed": seed,
        "train_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "audit_rows": len(audit_indices),
        "validity_pairs": len(validity_pairs),
        "utility_pairs": len(utility_pairs),
        "parameter_count": model.parameter_count,
        "classification_batch_size": config["classification_batch_size"],
        "pair_batch_size_per_objective": config["pair_batch_size"],
        "gradient_accumulation_steps": 1,
        "peak_reserved_gb": peak_reserved_gb,
        "memory_limit_gb": config["max_reserved_memory_gb"],
        "within_memory_limit": within_memory_limit,
        "best_epoch": best_epoch,
        "best_validation_reliability_auc": best_auc,
        "history": history,
        "validity_calibration": asdict(validity_calibration),
        "utility_calibration": asdict(utility_calibration),
        "audit_metrics": audit_metrics,
        "equivalent_rewrite": equivalent_drop,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_reload_max_abs_delta": reload_max_abs_delta,
        "predictions_path": str(predictions_path),
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_lofo(
    config: dict[str, Any],
    *,
    project_root: str | Path,
    families: list[str] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()

    def project_path(value: str) -> Path:
        target = (root / value).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Path escapes project root: {value}")
        return target

    table = load_feature_table(
        project_path(config["feature_manifest_path"]),
        expected_manifest_sha256=config["feature_manifest_sha256"],
    )
    if any(
        tensor.requires_grad
        for tensor in (table.z_previous, table.z_current, table.e_step)
    ):
        raise ValueError("Frozen feature tensors must not require gradients")
    selected_families = list(DEVELOPMENT_FAMILIES) if families is None else families
    if not selected_families or any(
        family not in DEVELOPMENT_FAMILIES for family in selected_families
    ):
        raise ValueError("Requested LOFO families are invalid")
    output_dir = project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = [
        train_lofo_fold(
            table,
            heldout_family=family,
            config=config,
            output_dir=output_dir,
        )
        for family in selected_families
    ]
    aucs = [fold["audit_metrics"]["reliability"]["roc_auc"] for fold in folds]
    if any(auc is None for auc in aucs):
        raise ValueError("At least one LOFO reliability AUC is undefined")
    equivalent_drops = [fold["equivalent_rewrite"]["mean_drop"] for fold in folds]
    result = {
        "schema_version": 1,
        "run_id": config["run_id"],
        "registered_config_sha256": sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "feature_manifest_path": config["feature_manifest_path"],
        "feature_manifest_sha256": config["feature_manifest_sha256"],
        "status": "PASS",
        "families_run": selected_families,
        "all_registered_families_run": set(selected_families) == set(DEVELOPMENT_FAMILIES),
        "folds": folds,
        "macro_reliability_roc_auc": float(np.mean(aucs)),
        "worst_family_reliability_roc_auc": float(np.min(aucs)),
        "mean_equivalent_rewrite_drop": float(np.mean(equivalent_drops)),
        "thresholds": {
            "macro_reliability_roc_auc": config["macro_auc_threshold"],
            "worst_family_reliability_roc_auc": config["worst_family_auc_threshold"],
            "maximum_equivalent_rewrite_drop": config["maximum_equivalent_drop"],
        },
    }
    result["gate_evaluated"] = result["all_registered_families_run"]
    result["gate_passed"] = (
        result["gate_evaluated"]
        and result["macro_reliability_roc_auc"] >= config["macro_auc_threshold"]
        and result["worst_family_reliability_roc_auc"]
        >= config["worst_family_auc_threshold"]
        and result["mean_equivalent_rewrite_drop"] <= config["maximum_equivalent_drop"]
    )
    result["status"] = (
        "PASS"
        if not result["gate_evaluated"] or result["gate_passed"]
        else "FAIL"
    )
    output_path = output_dir / "lofo_metrics.json"
    temporary_output_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary_output_path.replace(output_path)
    return result
