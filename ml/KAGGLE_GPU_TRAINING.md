# Kaggle GPU Auto-Tuning Workflow

Use Kaggle for heavier model tuning when local training becomes slow.

## 1. Create Kaggle Notebook

1. Open Kaggle.
2. Create a new notebook.
3. Enable GPU in notebook settings.
4. Upload this project folder or add it as a Kaggle dataset.

## 2. Install Dependencies

```bash
pip install -r ml/requirements.txt
```

## 3. Run Auto-Tuning

```bash
cd /kaggle/working/ai_stock_analysis_and_investment_portfolie/ml
python train_autotune.py --config config.example.json
```

The script will:

- download historical OHLCV data
- engineer technical features
- create bullish/bearish/sideways labels
- use time-series splits, not random splits
- run Optuna hyperparameter tuning
- train a final model
- save versioned model artifacts
- save metrics for backtesting review

## 4. Outputs

Artifacts are saved to:

```text
ml/artifacts/
```

Example files:

```text
model_20260510_221500.joblib
model_20260510_221500_metrics.json
training_dataset.parquet
```

## 5. Product Rule

Use trained ML models for probabilities and chart-path simulation.
Use Grok/OpenAI/Claude for explanation and reasoning.

Never present model output as guaranteed financial advice.
