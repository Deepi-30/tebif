# Findings for mentors – TEBIF on PaySim (short-budget demo)

**Bottom line: in this run nothing works well enough to claim a result. TEBIF does not beat the baselines, and neither do the baselines beat chance by a meaningful margin.** Treat the numbers as a working pipeline, not as evidence for or against the method.

## Setup (one line each)
- PaySim only, 6.36M rows, 8,213 frauds. Account stream = every transaction where the account is sender **or** receiver (sender-only gives 1 event per account).
- Subset of 420k events (all fraud-touching accounts + sampled normal streams), chronological split by step (train ≤ 281 < val ≤ 355 < test), streams cut at split boundaries.
- Label: confirmed fraud involving the account within the next 24 simulated hours. Window: last 30 events.
- **Test set has only 150 positive points out of 83,914 (prevalence 0.18 %)**, validation has 142.
- **Limited budget: 1 seed per model, 5–7 epochs. The requested mean ± std over 3 seeds is not available.**

## Test results (seed 0, no error bars)
| Model | AUC-ROC | PR-AUC | F1 (val thr.) | Recall@85 % prec. | ms/event |
|---|---|---|---|---|---|
| LSTM | 0.562 | 0.0021 | 0.005 | 0.000 | 2.0 |
| Vanilla Transformer | 0.507 | 0.0031 | 0.020 | 0.000 | 5.2 |
| TEBIF (Transformer + Δt + time) | 0.563 | 0.0025 | 0.007 | 0.000 | 5.7 |
| TEBIF, Δt only | 0.566 | 0.0022 | 0.000 | 0.000 | 5.3 |
| TEBIF, cumulative time only | 0.559 | 0.0022 | 0.004 | 0.000 | 5.4 |
| *Random scorer* | *0.500* | *0.0018* | | | |

- PR-AUC of every model is at or barely above the random level, AUC 0.51–0.57. The ordering of models is within what one extra seed could change
  (a second LSTM seed gave AUC 0.569, PR-AUC 0.0023, versus 0.562 / 0.0021 for seed 0). **No time-encoding variant is distinguishable from the others.**
- The time-gap (Δt) idea is therefore **neither confirmed nor refuted**: PaySim's steps are artificial hours, and the signal is too weak to see an effect.
- No leakage alarm: highest AUC is 0.57 (threshold 0.99). Raw balances were excluded from the features; the feature-set re-runs were not done.

## Risk score and alerts (TEBIF seed 0, α = 0.5, threshold chosen on validation = 0.742)
- Test: 13 of 1,017 fraud accounts alerted **before** their fraud (recall 1.3 %); 225 of 29,438 normal accounts alerted (0.8 %); precision 5.5 % (fraud streams are 3.3 % of evaluated streams, so only ~1.6× the base rate, from just 13 hits).
- Median lead time 1 event (84 h), but this is not meaningful as early warning: nearly all alerts fire at the **first** event of a stream when it is a received DEBIT, i.e. the model has found one cheap pattern, not behaviour over time. The average R_t curves of fraud and normal accounts are almost identical.
- p_t is a score, not a calibrated probability (training used a 10× positive weight).

## Explainability
- Attention and occlusion are produced for every demo account, with plain-language summaries. With one-event windows they are trivially "the current event".
- Faithfulness (200 test windows): deleting the single most-attributed past event lowers p by 0.065 on average with occlusion order vs −0.002 random, and attention order 0.006, so occlusion is faithful at k = 1. For k = 3 and 5, deleting events *raises* p for all orderings (−0.22 at k = 5, including random), so the test is confounded by window length and says little. Attention is **not** more faithful than random here.

## What this means / what we would do next
1. Do not present these numbers as a performance claim. They show the pipeline runs end to end and that the task as defined is almost unlearnable on PaySim.
2. Why: fraud accounts mostly have no usable history (only 3,065 of 16,426 fraud event–account pairs have an earlier event), and in PaySim the mule who receives the fraudulent transfer is usually not the account that cashes out, so no account-level story links them.
3. Next steps if time allows: 3+ seeds with a longer epoch budget; run the `nobal` / `leaky` feature checks; try a label that is learnable from history (for example "will the next event be fraud" with receiver-side features), or a dataset with real long histories; calibrate p_t.

## Shortcuts taken and things that did not work
- 1 seed, ≤ 7 epochs; the first training batch (3 seeds, 18 runs) was stopped for time (too slow on 4 CPU cores).
- A time-based 60/20/20 split gave a 1.2k-event test set, so the split was moved to transaction-volume quantiles.
- Occlusion instead of SHAP/LIME; attention layer hand-written for speed; class weighting implemented as negative sub-sampling plus weight 10.
- Frontend verified by rendering screenshots only (no automated clicking; phone width not verified).
