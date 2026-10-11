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

- Snapshot id: *(filled in after the build)*
- Development-stage notes: *(filled in after step 2)*
- Holdout run id and verdict: *(filled in after step 3)*
