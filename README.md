# TEBIF – simplified, reproducible version (short-budget demo)

Transformer-Based Explainable Behavioural Intelligence Framework for financial scam detection,
built only on the **synthetic PaySim** data set (no real customer data exists here).

> **Read this first – what this version is and is not**
> * It is a **short-budget demo**: **1 seed per model**, training capped at 5 epochs or early-stopped
>   at 7 or fewer. The requested "mean ± std over 3 seeds" is **not reported**; the code supports it
>   (`SEEDS` in `src/evaluate.py`, `--model all` in `train.py`), but the full 3-seed batch needs hours on this
>   4-core CPU and was stopped. Tables show no ± because a std from one seed would be fake.
> * All models score **near chance** (test AUC 0.51–0.57, PR-AUC ≈ prevalence). **No model, TEBIF included,
>   wins.** Differences between models are noise. See `findings.md`.
> * The feature-set / leakage re-runs were **not run** (see "Leakage").

## Idea
Each bank account is an ordered timeline of events. A model reads the last 30 events and outputs
p_t = score that the account will be involved in a confirmed fraud within the next 24 simulated hours
(PaySim `step` = 1 hour). A dynamic risk score `R_t = α·p_t + (1−α)·R_(t−1)` (α = 0.5) is updated after each event,
and an alert fires when `R_t ≥ τ` (τ chosen on the validation split).

## Setup and exact commands
```
pip install -r requirements.txt            # torch (CPU), numpy, pandas, scikit-learn, matplotlib
# data: data/PS_20174392719_1491204439457_log.csv  (PaySim; the file name is found with a glob)
python src/data_check.py                   # Step 0 report
python src/data.py                         # builds cache/events.npz, prints split statistics
# what was actually run (seed 0, epoch cap 5; early stopping on validation PR-AUC):
python src/train.py --model lstm        --seed 0 --epochs 5
python src/train.py --model vanilla     --seed 0 --epochs 5
python src/train.py --model tebif       --seed 0 --epochs 5
python src/train.py --model tebif_dt    --seed 0 --epochs 5
python src/train.py --model tebif_ctime --seed 0 --epochs 5
python src/evaluate.py                     # metrics table, risk score, alerts, lead time
python src/explain.py                      # attention / occlusion / faithfulness
python src/export.py                       # results/results.json + frontend/index.html
```
(Use `PYTHONIOENCODING=utf-8` on Windows consoles. The first four runs used the default cap of 20 epochs and
stopped by early stopping after 5 (LSTM, TEBIF Δt-only), 6 (vanilla) and 7 (TEBIF) epochs; `tebif_ctime` was run with the explicit cap of 5.
`python src/train.py --model all` would run the full 3-seed plan plus feature-set checks.)
Open `frontend/index.html` by double-click (no server, no internet; charts are plain SVG).

## Files
`src/data.py` pipeline · `src/models.py` LSTM / Transformer / TEBIF · `src/train.py` · `src/evaluate.py` (metrics + risk) ·
`src/explain.py` · `src/export.py` · `frontend/template.html` (+ generated `index.html`) · `results/` (metrics.json,
risk.json, faithfulness.json, results.json, PNG figures, `runs/` checkpoints and predictions) · `findings.md`.

## Step 0 – data check
* 6,362,620 rows, 11 columns, **8,213 frauds (0.129 %)**, `step` 1–743. Fraud only in TRANSFER and CASH_OUT.
* **Sender-only sequences are useless**: 99.85 % of senders appear once (mean 1.001 events).
  Therefore an account's stream = **all transactions where it is sender OR receiver**, ordered by step.
  Even so the median length is 1 (mean 1.40). Streams of ≥ 2 events: mean 8.8, median 6 over the full data.
  Merchants (`M…`) are excluded as accounts.
* Only 3,065 of 16,426 fraud (event, account) pairs have any earlier event in that account – the rest are an
  account's first event and cannot be predicted from history.

## Data pipeline (`src/data.py`) – design choices
* **Event** = one transaction seen from one account (every PaySim row gives a sender and a receiver event).
  Token = type × role (10 values). Continuous features: log amount, amount ÷ old balance (clipped at 5),
  relative balance change (clipped ±5). Δt = step difference to the previous event of the same stream (0 for the first).
  Δt is **not** an input feature; TEBIF gets it only through its time encoding.
* **Window** = last 30 events up to t, left-padded and masked. **Label** = 1 if a *later* event of the same
  account is a confirmed fraud ≤ 24 steps after t. Features use only events ≤ t.
* **Chronological split by step**: train ≤ 281, validation 282–355, test 356–743. The cut points are the 60 % / 80 %
  quantiles of PaySim *transactions* (not of time) because normal traffic is bursty: 95 % of it happens before
  step ≈ 445, while fraud is spread evenly. A 60/20/20 split of *time* left a test set of 1.2k events.
* **Streams are cut at split boundaries** (stream = account × split). A window never contains events from another split
  and a label's 24-step horizon must lie inside the split (points too close to a boundary are dropped as unlabelled).
  This is the strict reading of "no sequence may cross a split"; it also removes history for accounts that
  are active in two splits, which hurts (test positives are mostly at stream position 0).
* **Subset: 419,987 events (419,926 distinct PaySim rows, 77,285 accounts)**: every stream of every fraud-touching
  account, plus a random sample of normal streams drawn per split (60/20/20 of a 420k budget). Only accounts with ≥ 2 events
  per split are used. Consequence: prevalence in this subset is not the real-world prevalence.
  Streams: mean length 4.9, median 3 (after cutting at splits).
* Normalisation (z-score) and class weight are fitted on the **train split only**.
* Labelled points: train 231,514 (2,329 positives, 1.0 %), validation 58,664 (**142** positives, 0.24 %),
  test 83,914 (**150** positives, 0.18 %). Few test positives ⇒ every metric is very noisy.

## Models (`src/models.py`) – same data, optimiser, class weight, early stopping
* **TEBIF**: token embedding + projected continuous features + sinusoidal position encoding + sinusoidal encoding of Δt
  + sinusoidal encoding of cumulative time (time since the first event in the window); 2 pre-norm Transformer layers,
  4 heads, d = 64; the last (current) event is pooled into a sigmoid head. Last-layer attention weights are returned by `forward`.
* **Vanilla Transformer**: identical but without the two time encodings (positions only). **LSTM**: 2 layers, hidden 64, same inputs.
* **Ablations**: TEBIF with Δt encoding only; with cumulative-time encoding only. (The "no time encoding" ablation *is* the
  vanilla Transformer, because the architecture is otherwise identical.)
* Adam, lr 1e-3, batch 512. **Class imbalance**: each epoch uses all train positives plus a fresh random 10× sample of
  negatives with positive class weight 10 (≈ same expected gradient as weighting by the full-train n_neg/n_pos ≈ 98, 10× cheaper on CPU).
  Early stopping on validation PR-AUC (patience 4). Consequence: **p_t is a score, not a calibrated probability**
  (it is inflated by the 10× weight; that is why alert thresholds sit near 0.74).
* Attention is implemented by hand (not `nn.MultiheadAttention`) because the built-in layer made a training step ~3× slower on this CPU.

## Risk score and alerts (`evaluate.py`)
R_0 = p_0, R_t = 0.5·p_t + 0.5·R_(t−1). Threshold τ is chosen on validation to maximise stream-level F1 of "alert before the first fraud".
Primary model = TEBIF seed 0 (fixed beforehand, not chosen by test results). Lead time = events (and hours) between the first
alert and the first fraud event; streams whose very first event is fraud are excluded (nothing precedes them).

## Explainability (`explain.py`)
* **Attention**: last layer, mean over heads, from the current event to all events – "what the model attended to", not causation.
* **Occlusion (SHAP-style)**: remove one event, or blank one feature of one event, and measure the probability drop. Real SHAP was not used.
* **Faithfulness**: delete the top-k *past* events (k = 1, 3, 5) in occlusion order, attention order, or random order (average of 20);
  comprehensiveness = drop in p; plus insertion/deletion curves. The current event is never deleted. After deleting events, time gaps
  are recomputed from the remaining events. Pool: 200 test windows (alert events plus top-scored events, since there were too few alerts).
* Plain-language summaries are template text filled from these numbers (PaySim steps are hours, so gaps are in hours, not minutes).

## Leakage
* Step 0 probe: in fraud TRANSFER/CASH_OUT rows the origin balance is emptied in 97.6 % of cases vs 42.7 % in normal rows, and
  `oldbalanceDest = 0` in 65 % vs 14 %. Raw balances and an "emptied" flag are therefore **excluded**.
* The "safe" set still contains amount ÷ balance and relative balance change, which partly encode "account emptied" for the *current*
  event. They describe only events ≤ t, but they are not leak-free in spirit.
* The label looks at **future** events, so the current event's balances are not a direct label leak; no result exceeded AUC 0.57 (limit 0.99),
  so there is no sign of leakage. The planned `nobal` / `leaky` feature-set re-runs (code exists: `--feats`) were **not run**.

## Limitations and shortcuts (complete list)
1. 1 seed per model, ≤ 7 epochs: no std, no significance statements. `lstm_safe_s1` also exists in `results/runs` but is not used in the table (kept for a like-for-like comparison).
2. Everything is near chance; the data give very little learnable signal for this label (a gradient-boosting probe on hand features reached
   AUC 0.58 on val/test and 0.63 on held-out train streams). Test has only 150 positive points.
3. Alerts are dominated by one pattern: nearly all demo "alerts" fire at event 1 on a received DEBIT with no history. They are not early
   warnings of behaviour. At τ = 0.742 the test set has 13 of 1,017 fraud streams alerted early (recall 1.3 %), 225 of 29,438 normal streams alerted (0.8 %).
   Most fraud streams have their fraud more than 24 h after the previous events, or none before it.
4. Deletion test: deleting 5 past events *raises* p for every ordering (mean −0.22), so the test is confounded by window length (the model scores short
   windows higher). Only k = 1 shows occlusion > random (0.065 vs −0.002); attention-order ≈ random.
5. Subset sampling changes prevalence; stream cutting at splits removes history; the split is by transaction volume, not equal time.
6. Latency is measured on a single CPU thread, one window per call (LSTM 2.0 ms, Transformers 5–6 ms).
7. Recall at 85 % precision is 0 for every model (precision never reaches 85 %). The F1 values are < 0.02.
8. The frontend was checked by rendering it in headless Edge (screenshots) at desktop width; the play/slider interactions were not clicked through
   automatically, and phone width was not verified.
9. The raw CSV was moved from the project root into `data/`.
