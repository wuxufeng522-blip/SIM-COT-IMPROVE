from __future__ import annotations

from pathlib import Path
import csv
import json


def build_report(
    output_dir: str | Path,
    weight_stats: dict,
    equal_run: dict,
    weighted_run: dict,
    equal_metrics: dict,
    weighted_metrics: dict,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    same_updates = equal_run["completed_updates"] == weighted_run["completed_updates"]
    memory_ok = max(equal_run["peak_reserved_gb"], weighted_run["peak_reserved_gb"]) <= 7.4
    engineering_feasible = bool(
        same_updates
        and equal_run["finite"]
        and weighted_run["finite"]
        and memory_ok
        and weight_stats["roc_auc"] >= 0.65
        and weight_stats["mean_noisy_weight"] < weight_stats["mean_clean_weight"]
    )
    relative_nll_change = (
        equal_metrics["clean_step_nll"] - weighted_metrics["clean_step_nll"]
    ) / max(equal_metrics["clean_step_nll"], 1e-12)
    initially_effective = bool(
        engineering_feasible
        and weighted_metrics["answer_exact_match"] >= equal_metrics["answer_exact_match"]
        and relative_nll_change >= 0.03
    )
    if initially_effective:
        conclusion = "初步有效"
    elif engineering_feasible:
        conclusion = "工程可行，但未达到预注册质量收益"
    else:
        conclusion = "未达到工程可行判据"

    result = {
        "conclusion": conclusion,
        "engineering_feasible": engineering_feasible,
        "initially_effective": initially_effective,
        "relative_clean_step_nll_reduction": relative_nll_change,
        "weight_statistics": weight_stats,
        "equal_run": equal_run,
        "weighted_run": weighted_run,
        "equal_metrics": equal_metrics,
        "weighted_metrics": weighted_metrics,
    }
    (output / "final_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    rows = [
        ["answer_exact_match", equal_metrics["answer_exact_match"], weighted_metrics["answer_exact_match"]],
        ["clean_step_nll", equal_metrics["clean_step_nll"], weighted_metrics["clean_step_nll"]],
        ["step_token_accuracy", equal_metrics["step_token_accuracy"], weighted_metrics["step_token_accuracy"]],
        ["mean_pairwise_latent_l2", equal_metrics["mean_pairwise_latent_l2"], weighted_metrics["mean_pairwise_latent_l2"]],
    ]
    with (output / "metrics_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "equal", "weighted"])
        writer.writerows(rows)

    summary = f"""# RSR-RD Weighted SIM-CoT 一晚概念验证结果

## 结论

**{conclusion}**

## 预注册核心指标

- 污染步骤检测 ROC-AUC：{weight_stats['roc_auc']:.4f}
- 污染步骤平均权重：{weight_stats['mean_noisy_weight']:.4f}
- 干净步骤平均权重：{weight_stats['mean_clean_weight']:.4f}
- 等权组答案 EM：{equal_metrics['answer_exact_match']:.4f}
- 加权组答案 EM：{weighted_metrics['answer_exact_match']:.4f}
- 等权组干净步骤 NLL：{equal_metrics['clean_step_nll']:.4f}
- 加权组干净步骤 NLL：{weighted_metrics['clean_step_nll']:.4f}
- 干净步骤 NLL 相对降低：{relative_nll_change:.2%}

本结果来自单随机种子受控实验，不代表论文级统计结论。
"""
    (output / "RESULTS.md").write_text(summary, encoding="utf-8")
    return result
