# Oracle Step-Weighting Causal Test

Date frozen: 2026-08-09 (Asia/Shanghai)

## Question

When a contaminated reasoning step is identified perfectly, does assigning that
step an auxiliary-loss weight of 0.1 improve clean reasoning accuracy compared
with training on the same contaminated steps at equal weight?

This experiment tests the weighting mechanism only. It does not test the failed
reliability detector and does not establish performance on natural teacher noise.

## Common setup

- Starting model: released official SIM-CoT checkpoint 28, already independently
  verified at 44.43% exact match on the clean 1,319-example GSM8K-Aug test set.
- Hardware: one RTX 4060 8 GB.
- Training source: official `train.txt` examples with at least five reasoning
  steps whose first five steps pass the local arithmetic equation checker.
- Supervision: the answer loss is always unchanged and equally weighted. Only
  the five auxiliary step-reconstruction losses are manipulated.
- Every selected base sequence and clean auxiliary sequence must fit GPT-2's
  1,024-token context window.
- Contamination: exactly one of the first five steps in every training example
  is replaced (20% step contamination). Families and positions are balanced and
  deterministic. The five development families are numeric error, operator
  relation error, dependency/order error, irrelevant-but-correct, and redundant
  repeat.
- Oracle label: the data generator records the replaced step. No model or
  reliability head predicts the label.
- One fixed training schedule, initialization, optimizer, seed, and sample order
  are shared by every arm.
- The dropout seed is reset from the frozen schedule position at every
  micro-batch, preventing different corrupted-step token lengths from changing
  later base-forward dropout masks across arms.
- Formal budget: 256 optimizer updates, micro-batch 1, gradient accumulation 8,
  BF16 autocast with FP32 parameters/optimizer, AdamW, learning rate 1e-4,
  weight decay 0.01, gradient norm clip 1.0, seed 20260809.

## Arms

| Arm | Auxiliary targets | Step weights | Purpose |
|---|---|---|---|
| clean | All five clean | all 1.0 | Measure drift from clean continuation |
| noisy_equal | One known contaminated step | all 1.0 | Measure damage caused by noise |
| oracle_raw_0.1 | Same contamination | clean 1.0, contaminated 0.1 | Test the user's proposed weighting |
| oracle_normalized_0.1 | Same contamination | 0.1/1.0 then normalize per example to mean 1 | Separate selective weighting from simply shrinking the total auxiliary gradient |

The raw arm has mean step weight 0.82. The normalized arm uses clean weight
1.219512 and contaminated weight 0.121951, keeping the mean at 1.0.

## Implementation gates

1. All-one custom grouped auxiliary loss must match the released wrapper loss on
   the same clean example within absolute tolerance 1e-4 in FP32.
2. The answer-loss term and base-model input must be byte-identical across all
   four arms for a scheduled example.
3. The noisy target and contaminated position must be byte-identical across the
   three noisy arms.
4. A one-update-per-arm CUDA sanity run must have finite loss/gradients and peak
   reserved memory no greater than 7.4 GB before the formal run.
5. Each saved checkpoint and the frozen schedule receive SHA-256 hashes.

Failure of a gate stops the formal run.

## Evaluation

- Primary endpoint: clean answer exact match on all 1,319 official test examples,
  using the dataset answer field as ground truth and the existing official answer
  extractor.
- Secondary endpoints: fixed held-out clean answer NLL and clean auxiliary-step
  NLL; training loss/gradient diagnostics; per-family and per-position audits.
- Paired analysis: prediction-level McNemar counts and paired bootstrap 95%
  confidence intervals for accuracy differences.

Definitions:

- Noise damage = accuracy(clean) - accuracy(noisy_equal).
- Oracle recovery = accuracy(oracle_raw_0.1) - accuracy(noisy_equal).
- Recovery ratio = Oracle recovery / Noise damage, only when Noise damage > 0.

## Frozen interpretation rule

- If Noise damage is less than 1 percentage point, report the mechanism test as
  inconclusive because the intervention did not create enough observable harm.
- A positive raw-weight mechanism result requires both:
  - Oracle recovery of at least 1 percentage point; and
  - Recovery ratio of at least 50%.
- If raw 0.1 passes but normalized 0.1 does not improve, interpret the result as
  compatible with generic auxiliary-loss attenuation rather than demonstrated
  selective step weighting.
- If normalized 0.1 also improves, the result is stronger evidence that correctly
  selecting the contaminated step matters.
- No outcome from this test rescues or validates the previously failed LOFO
  reliability-head gate.

## Run IDs

- `O001`: frozen contamination schedule and audit.
- `O002`: loss parity and one-update CUDA sanity gates.
- `O010`: clean formal continuation.
- `O011`: noisy equal-weight continuation.
- `O012`: oracle raw-0.1 continuation.
- `O013`: oracle normalized-0.1 continuation.
- `O020`: common clean evaluation and paired causal analysis.
