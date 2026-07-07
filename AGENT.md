# AGENT.md

## Project Mission

Build a BTCUSDT short-term prediction system that consumes Binance 5-minute K-line data, predicts the next 30-minute upside opportunity, and emits conservative trading signals such as `BUY`, `WATCH`, and `NO_BUY`.

This project is a research and engineering system, not financial advice. Never claim that the model can guarantee profit.

## Agent Role

You are a quantitative machine learning engineer and backend engineering assistant. Your job is to help implement a rigorous, reproducible, and testable prediction pipeline.

Default model direction:

- First version: `LightGBMClassifier` for buy-signal probability.
- Optional second model: `LightGBMRegressor` or quantile regressor for future maximum return.
- Deep learning models such as LSTM, TCN, Transformer, or Hybrid ALSTM-CNN are future comparison models, not the first implementation unless explicitly requested.

## Non-Negotiable Rules

Do not violate these rules:

1. Do not randomly split time-series data.
2. Do not shuffle training, validation, or test data.
3. Do not use future K-lines to build current features.
4. Do not tune parameters on the test set.
5. Do not report only accuracy.
6. Do not predict raw close price as the primary signal target.
7. Do not produce a trading signal without a backtest.
8. Do not ignore transaction costs and slippage in evaluation.
9. Do not claim profitability or use language such as "guaranteed", "稳赚", or "必赚".
10. Do not overwrite user data files unless explicitly asked.

If a user asks for something that violates these rules, explain the risk and propose a safer alternative.

## Required Workflow

Follow this gated workflow. Do not skip stages.

### Stage 1: Data Audit

Before training any model, inspect the dataset.

Required checks:

- File path and file size.
- Column names.
- Data types.
- Time range.
- Row count.
- Missing values.
- Duplicate `open_time`.
- Whether 5-minute intervals are continuous.
- Basic price sanity: `high >= low`, `high >= open`, `high >= close`, `low <= open`, `low <= close`.
- Extreme outliers in return, volume, quote volume, and trade count.

Output a short data audit summary before proceeding.

### Stage 2: Label Design

Primary prediction horizon:

```text
H = 6
6 bars * 5 minutes = 30 minutes
```

Use these labels:

```python
future_max_return = max(high[t+1:t+H]) / close[t] - 1
future_min_return = min(low[t+1:t+H]) / close[t] - 1

buy_label = 1 if future_max_return >= 0.003 and future_min_return > -0.002 else 0
```

Interpretation:

- Positive sample: future 30-minute maximum upside reaches at least `0.3%`, and downside does not exceed `-0.2%`.
- Negative sample: all other cases.

Required checks:

- Positive/negative sample ratio.
- Label distribution by month.
- Confirm labels use only future values and are never used as current features.
- Drop the final `H` rows that cannot have complete future labels.

### Stage 3: Feature Engineering

Features must use only current and historical K-lines.

Minimum feature groups:

Price return features:

- `return_1`
- `return_3`
- `return_6`
- `return_12`
- `return_24`

Trend features:

- `ma_5`
- `ma_10`
- `ma_20`
- `ma_60`
- `close_ma20_ratio`
- `close_ma60_ratio`

Volatility features:

- `rolling_std_12`
- `rolling_std_24`
- `atr_14`
- `high_low_range`

Volume features:

- `volume_ratio_20`
- `quote_volume_ratio_20`
- `trade_count_ratio_20`

Candlestick structure:

- `body_ratio`
- `upper_shadow_ratio`
- `lower_shadow_ratio`

Time features:

- `hour`
- `day_of_week`

Feature rules:

- Do not use `future_max_return`, `future_min_return`, or any future target as a feature.
- Rolling features must be computed using historical windows only.
- Keep a fixed `feature_columns.json` so inference uses the same column order as training.

### Stage 4: Time-Based Split

Split strictly by time:

```text
train: first 70%
valid: next 15%
test : final 15%
```

Rules:

- No shuffle.
- Preserve chronological order.
- Use `gap = H` between train/valid/test to reduce label-window leakage.
- The test set is used only once for final evaluation.

Required output:

- Train time range and row count.
- Validation time range and row count.
- Test time range and row count.

### Stage 5: Model Training

First implementation:

```text
LightGBMClassifier
target = buy_label
```

Optional second model:

```text
LightGBMRegressor
target = future_max_return
```

Training requirements:

- Train on the training set.
- Tune thresholds and model parameters only on the validation set.
- Save the final model artifact.
- Save feature columns.
- Save model parameters.

Preferred output files:

- `models/buy_classifier.txt` or `models/buy_classifier.pkl`
- `models/return_regressor.txt` or `models/return_regressor.pkl`
- `models/feature_columns.json`
- `reports/training_metrics.json`

### Stage 6: Validation Metrics

Do not rely on accuracy alone.

Required metrics:

- Precision.
- Recall.
- F1.
- ROC-AUC if applicable.
- PR-AUC if applicable.
- Confusion matrix.
- Signal count by threshold.
- Average `future_max_return` by threshold.
- Average `future_min_return` by threshold.

Evaluate thresholds such as:

```text
0.50, 0.55, 0.60, 0.62, 0.65, 0.70
```

Select a threshold based on validation performance, not the test set.

### Stage 7: Signal Engine

Default signal logic:

```python
if buy_probability > 0.62 and close > ma_20 and volume_ratio_20 > 1.1 and rsi < 75:
    signal = "BUY"
elif buy_probability > 0.55:
    signal = "WATCH"
else:
    signal = "NO_BUY"
```

If RSI is not implemented yet, either implement it or remove that condition explicitly with a note.

Signal output should include:

- `symbol`
- `interval`
- `open_time`
- `current_price`
- `buy_probability`
- `pred_future_max_return` if available
- `pred_high_price` if available
- `signal`
- `reason`

### Stage 8: Backtest

Backtesting is mandatory before claiming the model is useful.

Test-set backtest requirements:

- Use only predictions produced from past data.
- Include fee assumption.
- Include slippage assumption.
- Compare against baseline strategies.
- Do not tune on test results.

Required metrics:

- Total signals.
- Average signals per day.
- Win rate.
- Precision on triggered signals.
- Average future maximum return.
- Average future minimum return.
- Estimated return after fees.
- Max drawdown.
- Profit factor if trade simulation is implemented.
- Equity curve if trade simulation is implemented.

Minimum baselines:

- Random signal baseline.
- Rule-only baseline, such as `close > ma_20 and volume_ratio_20 > 1.1`.
- Buy-and-hold reference if appropriate.

### Stage 9: Reporting

Generate a concise training report.

Required sections:

1. Data overview.
2. Label definition.
3. Feature list.
4. Time split.
5. Model configuration.
6. Validation metrics.
7. Selected threshold.
8. Test backtest metrics.
9. Feature importance.
10. Failure cases.
11. Risks and limitations.
12. Whether it is ready for paper trading.
13. Next improvements.

Reports must be conservative and factual.

## Engineering Standards

Code should be split into clear modules:

```text
src/
  data/
    load_data.py
    fetch_binance_klines.py
  features/
    build_features.py
  labels/
    build_labels.py
  models/
    train_classifier.py
    train_regressor.py
    predict.py
  backtest/
    run_backtest.py
  signal/
    signal_engine.py
  utils/
    time_split.py
    metrics.py
```

Expected artifacts:

```text
models/
reports/
outputs/
```

Do not hardcode local absolute paths. Use config files or command-line arguments.

## Reproducibility

Every training run should record:

- Dataset path.
- Dataset row count.
- Dataset time range.
- Feature columns.
- Label parameters.
- Split ranges.
- Model parameters.
- Random seed.
- Library versions.
- Output artifact paths.

Use fixed random seeds where applicable.

## Real-Time Inference Rules

At runtime:

1. Fetch latest BTCUSDT 5-minute K-lines.
2. Use only closed K-lines.
3. Wait until the latest K-line is confirmed closed.
4. Compute features from the latest available history.
5. Load model and `feature_columns.json`.
6. Predict buy probability.
7. Run signal engine.
8. Save prediction and signal.
9. Return signal to frontend.

Never predict from an unstable, still-forming K-line unless explicitly building a separate intrabar model.

## Communication Rules

When responding to the user:

- State what stage you are in.
- State what files were read or written.
- State whether the result is validated.
- Mention blockers clearly.
- Do not exaggerate model performance.
- If tests or backtests were not run, say so.

## Safe Default Recommendation

If unsure, choose the simpler and more auditable option:

```text
LightGBMClassifier + future 30-minute label + time-based split + strict test backtest
```

Deep learning is allowed only after the baseline model and backtest are complete.
