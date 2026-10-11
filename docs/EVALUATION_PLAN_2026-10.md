# Evaluation plan, October 2026 (written and committed before the holdout is scored)

This file fixes, in advance, how the second evaluation will be run and judged. It is committed to
git before any model sees the holdout, so the rules cannot be adjusted to fit the result.

## Why a second evaluation

The first evaluation (run `mv5d-20261009-221045-8800915fff`) used public files that end in March
2021 and a hindsight-chosen list of 41 stocks, and found **no predictive edge**. It also had no
sector-ETF or Treasury-yield data, so two feature groups were never tested. This evaluation
uses current data from Yahoo Finance and FRED, includes those groups, and judges models on a
period no model in this repository has ever been scored on.

## Data

- Snapshot built by `backend/scripts/build_yahoo_snapshot.py`, on the owner's Mac.
- Daily bars from 1999-01-01 to the last completed session before the build.
- **Universe:** 99 large US stocks, approximately the S&P 100 at the end of 2021. The list was
  compiled from memory, not from an official constituent file.
- **Context:** SPY, QQQ, VIX, 11 SPDR sector ETFs, and FRED `DGS10` and `T10Y2Y`.
- The snapshot id and every file hash are recorded in its manifest. The evaluation refuses to
  run if any file changed.

## Splits (fixed now)

| Part | Period | Use |
|---|---|---|
| Development | 1999 → 2021-12-31 (minus the embargo) | Choosing models, settings and calibration |
| Walk-forward validation | 8 folds from 2010-01-01 to the end of development | Inside development only |
| **Holdout** | **2022-01-01 → the last row with a complete 5-session label** | Scored **once** |

`python -m app.forecast evaluate --snapshot <id> --holdout-start 2022-01-01 --first-validation-start 2010-01-01`

The 2016–2021 period, which was the first evaluation's holdout, is now ordinary development
data. The new holdout (2022 onwards) has not been used to score, select or tune anything.

## What is held fixed

- **Target:** the adjusted close 5 sessions later is higher than today's (unchanged).
- **Feature pipeline:** schema `marketvision-daily` 1.0.0 (unchanged). Feature groups are kept if
  they cover at least 90% of rows, as before.
- **Candidates and settings grids:** as in `app/forecast/models.py` and `train.py` at this commit.
- **Promotion rule:** version 1 in `app/forecast/promotion.py`, unchanged. All of it must pass:
  - Holdout Brier skill above 0, with its 95% calendar-week bootstrap interval above 0.
  - Holdout ROC-AUC interval above 0.50.
  - Positive skill in at least two thirds of holdout years.
  - Positive validation skill.
  - No miscalibrated probability band.
  - Every leakage check passes.
- **Seed:** 20261009. **Bootstrap:** 1,000 resamples.

## Procedure

1. Build the snapshot and record its id here.
2. Run the development stage only (`--stage development`), to check data coverage and the
   feature groups. If a data problem appears (a broken symbol, a coverage failure), it may be
   fixed by *dropping* that symbol, with the reason written here. Nothing about models,
   features or the rule changes in response to development results.
3. Run the full evaluation once. Whatever the verdict, it is reported as is.
4. A model is promoted only if it passes rule v1, and only by `python -m app.forecast promote`.

## What a result will and will not mean

- **Passing** would establish a statistical edge in 5-session direction on 2022–2026 for large
  US stocks. It would not establish profitable trading, which is the backtester's job, with
  costs.
- **Failing** means the application keeps serving the labelled base rate. That is the expected
  outcome for daily-bar features on large, liquid stocks, and it is a valid result.

## Record

- Snapshot id: **`yahoo-daily-3b1b2218e5af`**. Built 2026-10-11 on the owner's Mac, 1999-01-04 to
  2026-10-09: 97 stocks plus 14 context instruments and 2 FRED series. All 113 files were verified
  against the manifest after the transfer.
  - **Not downloaded (dropped, as the plan allows):** BK (Yahoo had no data under that ticker)
    and WBA (taken private in 2025, so Yahoo no longer serves it). WBA is a concrete case of the
    survivorship gap described above.
  - **Quality flags** (large one-day moves, kept): AAPL 1, AIG 4, C 1, MO 1, MS 1, VIX 14. Checked
    by date; all are real events:
    - AAPL 2000-09-29 (−52%, profit warning).
    - AIG, C and MS in Sept 2008 – Aug 2009 (the financial crisis).
    - MO 2008-03-31: the Philip Morris International spin-off, −69.9% in raw close but −2.5%
      in the adjusted close the targets use.
    - VIX spikes.

    *Correction:* the snapshot manifest's limits list says Yahoo does not adjust GE's spin-offs.
    That was written before checking, and it is wrong: GE's adjusted series shows no such jumps.
    The script text is corrected; the manifest of this snapshot keeps the original wording.
- Development-stage notes (run `mv5d-20261011-202506-e982a61a5a`, development only):
  - **Data:** 563,020 complete rows (2000-03-13 to 2026-10-02) from 97 stocks. All seven
    feature groups are used: price, volume, market, qqq, vix, sector and macro. Base rate 0.542.
    The holdout has 115,624 rows (2022-01-03 to 2026-10-02).
  - **All six leakage checks pass.** The shuffled-label AUC averages 0.495 over 5 shuffles.
  - **Validation:** no candidate beats the constant base rate.
    - Selected: `random_forest {max_depth: 4}`, Brier skill −0.00053, within-fold AUC 0.515.
    - Best logistic regression: −0.0024. Best gradient boosting: −0.0051. LightGBM: −0.0062.
  - Expected-return R² against the historical mean: −0.0017.
  - **No data problem was found, so nothing is changed.** The full run follows exactly as planned.
    Because validation skill is negative, requirement 5 of rule v1 is expected to fail. The
    holdout is still scored once, as committed.
- **Holdout run `mv5d-20261011-211129-e982a61a5a`, scored once on 2026-10-11 (0 earlier
  scorings of this holdout). Verdict: not promoted. No predictive edge was established.**

  | Holdout 2022-01-03 → 2026-10-02 (115,624 rows) | Accuracy | Majority | AUC | Brier | Skill vs base rate |
  |---|---|---|---|---|---|
  | Constant base rate (dev) | 0.5275 | 0.5275 | 0.500 | 0.24956 | 0 |
  | Trailing frequency | 0.5232 | 0.5275 | 0.503 | 0.25011 | −0.0022 |
  | Logistic regression | 0.5206 | 0.5275 | 0.510 | 0.24947 | +0.0004 |
  | **Random forest (selected on development)** | 0.5286 | 0.5275 | 0.503 | 0.24982 | −0.0010 |
  | Gradient boosting | 0.5261 | 0.5275 | 0.505 | 0.25047 | −0.0036 |
  | Hist gradient boosting | 0.5297 | 0.5275 | 0.505 | 0.25039 | −0.0033 |
  | LightGBM | 0.5277 | 0.5275 | 0.503 | 0.25067 | −0.0044 |

  - **Selected model after calibration (sigmoid):** Brier skill **−0.0027 [−0.0054, +0.0001]**,
    AUC **0.503 [0.481, 0.528]**.
  - **Rule v1:** only the "is a model" and leakage requirements pass. Holdout skill, the AUC
    interval, yearly skill (2 of 5 years positive: 2024 +0.0001, 2025 +0.0003), validation skill
    and calibration (2 bands off by more than 0.03) all fail.
  - Logistic regression's +0.0004 on the holdout is not a finding. It was not the model chosen
    on development data, picking it now would be choosing on the holdout, and the amount is far
    inside the noise.
  - **What changed underneath:** stocks rose over 5 sessions 54.5% of the time in 2000–2021 but
    52.8% in 2022–2026, with 49.9% in 2022 and 50.5% in 2026 so far. Every model learned the
    older, higher rate and over-stated the chance of a rise in the weak years. That is the
    calibration failure.
  - **Consequence:** the application keeps serving the labelled base rate. The served baseline
    is now this run's development base rate, 0.5454 (2000–2021), with its uncertainty table.
    `edge_established` stays `false`.
