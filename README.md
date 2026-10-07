# M5 Forecasting – Accuracy: hierarchical forecasting (Exercise 3)

Answers to Exercise 3 on the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) Walmart dataset, with code that runs on the real data.

| File | What it is |
|---|---|
| `exercise3_m5_forecasting.ipynb` | **Main notebook.** Written answers to Q1–Q4 plus the analysis behind them: hierarchy, EDA charts, backtest, models and submission. |
| `m5_vertex_custom_training.ipynb` | Runs the same pipeline as a **Vertex AI (Gemini Enterprise Agent Platform) custom training job**, adapted from Google's LightGBM sample. |
| `m5_core.py` | Shared pipeline module. It is written by the main notebook's `%%writefile` cell and packaged into the Vertex trainer. |
| `lgbm-model.ipynb` | My first LightGBM + Optuna attempt. The main notebook's Q4 lists what changed and why. |
| `get_started_vertex_training_lightgbm.ipynb` | Google's original Vertex AI LightGBM sample (Iris), used as the shell for the Vertex notebook. |

## Approach in one paragraph

Reproduce the WRMSSE metric over all 42,840 series of the 12-level hierarchy, and validate with rolling 28-day folds. Train one **global LightGBM (Tweedie loss)** on all 30,490 item-store series. Its features are *horizon-safe* (every sales feature lags at least 28 days, so there is no leakage and no recursion), plus prices/promotions, events, state-specific SNAP, calendar and parent-level demand. Benchmark against seasonal naive, ETS and a log-scale SARIMAX with exogenous regressors at the aggregate levels. Compare bottom-up with **WLS-reconciled + middle-out** forecasts, choose by holdout WRMSSE per level, then refit and forecast d_1942–1969.

## Running it

- **Kaggle:** attach the competition data and run `exercise3_m5_forecasting.ipynb` (default path `/kaggle/input/m5-forecasting-accuracy`).
- **Vertex AI Workbench / Colab / local:** set `M5_DATA_PATH` to a folder or `gs://` path with `sales_train_evaluation.csv`, `calendar.csv` and `sell_prices.csv`.
- **Quick pass:** `M5_FAST_MODE=1` uses a one-year training window and fewer boosting rounds.
- **Outputs:** charts go to `figures/`, plus `submission.csv` and `lgbm_m5.txt` (the model).
- **Vertex AI custom training:** open `m5_vertex_custom_training.ipynb` from this folder and fill in project, region and bucket.

Python ≥ 3.9 with `numpy`, `pandas`, `scipy`, `lightgbm`, `statsmodels`, `matplotlib`.
