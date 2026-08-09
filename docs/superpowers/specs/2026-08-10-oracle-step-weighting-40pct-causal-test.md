# Oracle Step Weighting at 40% Step Noise: Frozen Experiment Specification

**Date:** 2026-08-10  
**Status:** Approved for execution  
**Parent experiment:** `2026-08-09-oracle-step-weighting-causal-test.md`

## Research question

When the contaminated auxiliary-step rate is strong enough to measurably hurt SIM-CoT, does a perfect oracle that assigns weight `0.1` to every known contaminated step improve answer exact match on the official clean test set?

This is a causal mechanism test. It does not test a learned reliability detector and does not claim to reproduce natural teacher noise.

## Frozen comparison

All arms start from the same official checkpoint, see the same 2,048 training questions in the same order, use identical answer loss, optimizer, seed, update count, accumulation, and per-microbatch dropout seed. Only auxiliary-step text and weights differ.

| Arm | Auxiliary targets | Step weights |
|---|---|---|
| Prior clean reference (`O010`) | Five clean steps | `(1,1,1,1,1)` |
| `O111` noisy equal | Two contaminated steps, three clean steps | all one |
| `O112` oracle raw 0.1 | Same targets as `O111` | contaminated `0.1`, clean `1.0` |
| `O113` oracle normalized 0.1 | Same targets as `O111` | same 10:1 ratio, rescaled to mean one |

For two contaminated steps, raw weights are a permutation of `(0.1,0.1,1,1,1)`. Normalized weights are a permutation of `(0.15625,0.15625,1.5625,1.5625,1.5625)`.

## Noise construction

- Exactly two of the five supervised auxiliary steps are contaminated in every training example: 40% step contamination.
- The first contamination is copied byte-for-byte from the frozen 20% schedule.
- The second uses a different corruption family and a different step position from the first.
- The same deterministic corruption library and validity checks as the parent experiment are used.
- The 2,048 question indices, question order, and first contamination are immutable.
- The second family-position requests are deterministically balanced. Invalid requests may only be repaired by swapping requests between examples, preserving aggregate cell counts.
- The generated schedule is frozen by SHA-256 before training.

## Fixed training configuration

- Starting model: official SIM-CoT checkpoint 28
- Seed: `20260809`
- Updates: `256`
- Gradient accumulation: `8`
- Effective microbatches: `2,048`
- Auxiliary groups: `5`
- Latent stage: `5`; continuous thought width: `2`
- Precision: BF16
- Learning rate: `1e-4`
- Weight decay: `0.01`
- Gradient clipping: `1.0`
- Maximum reserved GPU memory: `7.4 GiB`
- Hardware target: RTX 4060 8 GB

The prior clean arm is reused rather than retrained because its data, initialization, update order, and hyperparameters are unchanged. This saves one formal training run without changing the comparison.

## Run order and gates

1. `O101`: build and audit the 40% frozen schedule.
2. `O102`: all-one loss parity plus one-update sanity for the three new arms.
3. `O111`-`O113`: train the three formal arms sequentially on one GPU.
4. `O120`: evaluate all new checkpoints on the full official test split and pair them with the prior clean predictions.

Sanity must pass before formal training. Evaluation ground truth must come from the official test dataset answer field, never from another model.

## Primary metrics and frozen interpretation

Let `A_clean`, `A_equal`, `A_raw`, and `A_norm` be paired answer EM values.

- Noise damage: `A_clean - A_equal`
- Raw oracle recovery: `A_raw - A_equal`
- Normalized oracle recovery: `A_norm - A_equal`
- Recovery ratio: raw recovery / noise damage, when damage is positive
- Paired uncertainty: exact McNemar test and paired bootstrap 95% CI

Decision rules:

- If noise damage is below `1.0` percentage point, report `INCONCLUSIVE_INSUFFICIENT_NOISE_DAMAGE`.
- Otherwise, raw oracle weighting must recover at least `1.0` percentage point and at least 50% of measured damage for a positive raw mechanism result.
- A positive normalized result strengthens selective-weighting evidence by controlling average auxiliary-loss scale.
- Statistical tests and confidence intervals are always reported; threshold labels are not substitutes for uncertainty.

## Success boundaries

A positive result establishes only that correct step-level downweighting can help under this controlled 40% synthetic-noise regime. A negative result weakens the proposed weighting mechanism under this training budget. Either result does not establish whether a learned head can recognize natural teacher errors.

## Budget

One schedule/sanity stage, three 256-update training arms, and three full-test evaluations: approximately 2 GPU-hours on the local RTX 4060, with a hard stop if a run exceeds twice the parent experiment's observed runtime or violates the memory gate.
