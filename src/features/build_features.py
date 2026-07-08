from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_2",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "return_48",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_60",
    "rolling_std_12",
    "rolling_std_24",
    "rolling_std_48",
    "high_low_range",
    "volume_zscore_20",
    "trade_count_change_1",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "future_max_return",
    "future_min_return",
    "buy_label",
    "future_max_return_30m",
    "future_min_return_30m",
    "future_close_return_30m",
    "future_max_return_30m_from_next_open",
    "future_min_return_30m_from_next_open",
    "future_close_return_30m_from_next_open",
    "y_buy",
    "signal",
    "reason",
    "buy_probability",
    "predicted_max_return",
    "pred_future_max_return",
    "pred_high_price",
}


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build current-and-historical-only features and return fixed feature order."""
    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for features: {missing}")

    featured = df.copy().sort_values("open_time").reset_index(drop=True)
    close = featured["close"].astype(float)
    high = featured["high"].astype(float)
    low = featured["low"].astype(float)
    open_ = featured["open"].astype(float)

    for period in [1, 2, 3, 6, 12, 24, 48]:
        featured[f"return_{period}"] = close.pct_change(periods=period, fill_method=None)

    for window in [5, 10, 20, 60]:
        featured[f"ma_{window}"] = close.rolling(window=window).mean()

    returns_1 = featured["return_1"]
    for window in [12, 24, 48]:
        featured[f"rolling_std_{window}"] = returns_1.rolling(window=window).std()

    featured["high_low_range"] = _safe_div(high - low, close)

    volume_mean_20 = featured["volume"].rolling(window=20).mean()
    volume_std_20 = featured["volume"].rolling(window=20).std()
    featured["volume_zscore_20"] = _safe_div(featured["volume"] - volume_mean_20, volume_std_20)
    featured["trade_count_change_1"] = featured["trade_count"].pct_change(
        periods=1,
        fill_method=None,
    )

    candle_range = (high - low).replace(0, np.nan)
    upper_body = pd.concat([open_, close], axis=1).max(axis=1)
    lower_body = pd.concat([open_, close], axis=1).min(axis=1)
    featured["body_ratio"] = (close - open_).abs() / candle_range
    featured["upper_shadow_ratio"] = (high - upper_body) / candle_range
    featured["lower_shadow_ratio"] = (lower_body - low) / candle_range

    feature_columns = list(FEATURE_COLUMNS)
    forbidden = FORBIDDEN_FEATURE_COLUMNS.intersection(feature_columns)
    if forbidden:
        raise ValueError(f"target columns leaked into features: {sorted(forbidden)}")

    featured = featured.replace([np.inf, -np.inf], np.nan)
    featured = featured.dropna(subset=feature_columns).reset_index(drop=True)
    return featured, feature_columns


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)
