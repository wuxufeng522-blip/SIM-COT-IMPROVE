# Output Manifest: Auxiliary-Gradient Leverage Experiment

| Artifact | Purpose |
|---|---|
| `gradient_leverage_report_2026-08-12.md` | Timestamped Chinese result report and claim boundary |
| `gradient_leverage_report.md` | Pointer to the latest report |
| `analysis.json` | Machine-readable preregistered gates, arm summaries and per-seed contrasts |
| `gradient_audit.json` | Eighteen-example, twelve-layer gradient norm/direction audit |
| `sanity_gate.json` | Five-arm objective, memory and post-hoc preclip-gradient audit |
| `confirm_audit.json` | Frozen 1,024-example confirm-set overlap and hash audit |
| `overnight_state.json` | End-to-end completion ledger; official-test flag remains false |
| `pilot/seed_*/*/metrics.json` | Fifteen 64-update training summaries |
| `eval/seed_*/*/metrics.json` | Fifteen complete confirm-set evaluation summaries |
| `eval/seed_*/*/predictions.jsonl` | Per-question predictions retained locally for paired diagnostics |

Full checkpoints and the frozen confirm dataset are under `work/reliable_simcot/gradient_leverage/` and are intentionally not duplicated here.
