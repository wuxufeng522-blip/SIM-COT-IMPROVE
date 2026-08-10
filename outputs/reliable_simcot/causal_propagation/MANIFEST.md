# Output Manifest: Causal-Propagation Pilot

| Artifact | Purpose |
|---|---|
| `causal_pilot_report_2026-08-10.md` | Timestamped Chinese result report and claim boundary |
| `causal_pilot_report.md` | Pointer to the latest report |
| `calibration_gate.json` | Machine-readable preregistered damage gate and internal gate hash |
| `pilot_diagnostics.json` | Paired correctness transitions, NLL and exact McNemar diagnostics |
| `schedule_audit.json` | Frozen split/cell distributions and schedule SHA-256 without full entries |
| `readable_causal_examples.json` | One manually auditable example for each causal corruption family |
| `loss_parity.json` | Official-vs-custom all-one loss parity |
| `sanity_gate.json` | Six-arm one-update GPU sanity results |
| `pilot/*/metrics.json` | Four 64-update pilot training summaries |
| `pilot_eval/*/metrics.json` | Four complete frozen-dev evaluation summaries |
| `pilot_eval/*/predictions.jsonl` | Per-question dev predictions used for paired diagnostics |

The full frozen schedule, dev subset, and training checkpoints are under `work/reliable_simcot/causal_propagation/` and are intentionally not duplicated here. Formal outputs do not exist because the calibration gate failed before official-test access.
