from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "return_24",
    "ma_5",
    "ma_10",
    "ma_20",
    "ma_60",
    "close_ma20_ratio",
    "close_ma60_ratio",
    "rolling_std_12",
    "rolling_std_24",
    "atr_14",
    "high_low_range",
    "volume_ratio_20",
    "quote_volume_ratio_20",
    "trade_count_ratio_20",
    "body_ratio",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "rsi_14",
    "hour",
    "day_of_week",
]

TARGET_COLUMNS = {"future_max_return", "future_min_return", "buy_label"}


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Build historical-only features and return the fixed feature order."""
    required = {
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for features: {missing}")

    featured = df.copy().reset_index(drop=True)
    close = featured["close"]
    high = featured["high"]
    low = featured["low"]
    open_ = featured["open"]

    for period in [1, 3, 6, 12, 24]:
        featured[f"return_{period}"] = close.pct_change(period)

    for window in [5, 10, 20, 60]:
        featured[f"ma_{window}"] = close.rolling(window=window).mean()

    featured["close_ma20_ratio"] = _safe_div(close, featured["ma_20"]) - 1
    featured["close_ma60_ratio"] = _safe_div(close, featured["ma_60"]) - 1
    featured["rolling_std_12"] = featured["return_1"].rolling(window=12).std()
    featured["rolling_std_24"] = featured["return_1"].rolling(window=24).std()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    featured["atr_14"] = true_range.rolling(window=14).mean()
    featured["high_low_range"] = _safe_div(high - low, close)

    featured["volume_ratio_20"] = _safe_div(
        featured["volume"], featured["volume"].rolling(window=20).mean()
    )
    featured["quote_volume_ratio_20"] = _safe_div(
        featured["quote_volume"], featured["quote_volume"].rolling(window=20).mean()
    )
    featured["trade_count_ratio_20"] = _safe_div(
        featured["trade_count"], featured["trade_count"].rolling(window=20).mean()
    )

    candle_range = (high - low).replace(0, np.nan)
    upper_body = pd.concat([open_, close], axis=1).max(axis=1)
    lower_body = pd.concat([open_, close], axis=1).min(axis=1)
    featured["body_ratio"] = (close - open_).abs() / candle_range
    featured["upper_shadow_ratio"] = (high - upper_body) / candle_range
    featured["lower_shadow_ratio"] = (lower_body - low) / candle_range
    featured["rsi_14"] = _rsi(close, window=14)

    featured["hour"] = featured["open_time"].dt.hour
    featured["day_of_week"] = featured["open_time"].dt.dayofweek

    feature_columns = list(FEATURE_COLUMNS)
    forbidden = TARGET_COLUMNS.intersection(feature_columns)
    if forbidden:
        raise ValueError(f"target columns leaked into features: {sorted(forbidden)}")

    featured = featured.replace([np.inf, -np.inf], np.nan)
    featured = featured.dropna(subset=feature_columns).reset_index(drop=True)
    return featured, feature_columns


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(100)
