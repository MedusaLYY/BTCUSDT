# BTCUSDT Short-Term Training Report

## Data Overview

- Dataset: `C:\ai\股票短期最高值预测\数据集\BTCUSDT_5m_2y.csv`
- Rows: 210241
- Time range: 2024-07-06 06:25:00 -> 2026-07-06 06:25:00
- Audit passed: True

## Label Definition

- Horizon: 6 bars
- Upside threshold: 0.003
- Drawdown threshold: -0.002
- Positive rate: 18.1483%

## Feature List

- Feature count: 24

## Time Split

- Train: 147123 rows, 2024-07-06 11:20:00 -> 2025-11-29 07:30:00
- Valid: 31520 rows, 2025-11-29 08:05:00 -> 2026-03-18 18:40:00
- Test: 31521 rows, 2026-03-18 19:15:00 -> 2026-07-06 05:55:00

## Validation Metrics

- ROC-AUC: 0.6849232266225195
- PR-AUC: 0.2920179989151467
- Selected threshold: 0.65

## Test Backtest

- Total BUY signals: 2
- Precision on triggered signals: 0.0
- Average future max return: 0.0018845252626886744
- Average future min return: -0.0024970203109236366
- Estimated return after costs: -0.0010309494746226517
- Max drawdown: -0.0011016509130277186

## Feature Importance

- hour: 306
- atr_14: 299
- day_of_week: 202
- ma_20: 191
- ma_60: 187
- rolling_std_24: 187
- ma_5: 186
- close_ma60_ratio: 181
- ma_10: 169
- rolling_std_12: 125
- high_low_range: 120
- return_24: 98
- close_ma20_ratio: 87
- return_6: 55
- return_12: 47

## Risks and Limitations

- This report is factual research output, not financial advice.
- Test data was used only for final evaluation.
- Fees and slippage are assumptions and may differ from live execution.

## Paper Trading Readiness

- Ready for paper trading: no
