# Output Manifest: Oracle Weighting 40% Noise

| Artifact | Purpose |
|---|---|
| `oracle_weighting_40pct_causal_report_2026-08-10.md` | Timestamped human-readable final report |
| `oracle_weighting_40pct_causal_report.md` | Pointer to the latest report |
| `o101_schedule_audit.json` | Frozen schedule distribution and SHA-256 audit |
| `o102_loss_parity.json` | Official-vs-custom all-one loss parity |
| `o102_sanity.json` | One-update sanity results for all three new arms |
| `o120_causal_analysis.json` | Machine-readable paired primary analysis |
| `noisy_equal/metrics.json` | O111 formal training metrics |
| `oracle_raw_0.1/metrics.json` | O112 formal training metrics |
| `oracle_normalized_0.1/metrics.json` | O113 formal training metrics |
| `eval/*/metrics.json` | Full official-test evaluation summaries |
| `eval/*/predictions.jsonl` | Per-question predictions used for paired statistics |

Training checkpoints and the full frozen schedule are under `work/reliable_simcot/oracle_weighting_40pct/`; they are intentionally not duplicated into the output directory.
