# BTCUSDT Short-Term Training Report

## Data Overview

- Dataset: `C:\ai\股票短期最高值预测\数据集\BTCUSDT_5m_2y.csv`
- Rows: 210241
- Time range: 2024-07-06 06:25:00 -> 2026-07-06 06:25:00
- Audit passed: True

## Label Definition

- Horizon: 6 bars
- Upside threshold: 0.002
- Positive rate: 39.0377%

## Feature List

- Feature count: 20

## Time Split

- Train: 147123 rows, 2024-07-06 11:20:00 -> 2025-11-29 07:30:00
- Valid: 31520 rows, 2025-11-29 08:05:00 -> 2026-03-18 18:40:00
- Test: 31521 rows, 2026-03-18 19:15:00 -> 2026-07-06 05:55:00

## Validation Metrics

- AUC: 0.7473676391571272
- Average Precision: 0.6490925374016413
- Brier Score: 0.21085282146500042
- BUY probability threshold: 0.62
- BUY return threshold: 0.0025

## Test Backtest

- Total BUY signals: 7000
- Precision on triggered signals: 0.5848571428571429
- Average future max return: 0.003406423310050747
- Average future min return: -0.0033399682713373628
- Average future close return: 0.00010112566299649465
- Average return after costs: -0.0022988743370035048
- Max consecutive losses: 72
- Max drawdown: -0.9999999055204263

## Feature Importance

- rolling_std_48: 462
- high_low_range: 383
- ma_60: 300
- rolling_std_12: 300
- rolling_std_24: 299
- return_24: 262
- ma_5: 249
- return_12: 232
- return_48: 228
- return_6: 220
- ma_10: 172
- ma_20: 146
- body_ratio: 129
- return_3: 102
- return_2: 90

## Risks and Limitations

- This report is factual research output, not financial advice.
- Test data was used only for final evaluation.
- Threshold sweep reports must not be used to tune default test thresholds.
- Fees and slippage are assumptions and may differ from live execution.
