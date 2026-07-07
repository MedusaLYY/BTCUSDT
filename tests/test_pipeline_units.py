from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.run_backtest import backtest_triggered_signals
from data.load_data import AuditError, audit_kline_data
from features.build_features import build_feature_frame
from labels.build_labels import add_future_labels
from signals.signal_engine import assign_signals
from utils.metrics import evaluate_thresholds, select_high_precision_threshold
from utils.time_split import time_based_split


def make_kline_frame(rows: int = 80) -> pd.DataFrame:
    open_time = pd.date_range("2026-01-01", periods=rows, freq="5min")
    close = pd.Series(100 + np.linspace(0, 4, rows) + np.sin(np.arange(rows) / 3))
    open_ = close.shift(1).fillna(close.iloc[0])
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.2
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.2
    volume = pd.Series(10 + (np.arange(rows) % 7), dtype=float)
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_.to_numpy(),
            "high": high.to_numpy(),
            "low": low.to_numpy(),
            "close": close.to_numpy(),
            "volume": volume.to_numpy(),
            "close_time": (open_time.view("int64") // 1_000_000) + 299_999,
            "quote_volume": (volume * close).to_numpy(),
            "trade_count": 100 + (np.arange(rows) % 5),
        }
    )


def test_audit_rejects_duplicate_open_time():
    df = make_kline_frame(10)
    df.loc[3, "open_time"] = df.loc[2, "open_time"]

    with pytest.raises(AuditError, match="duplicate open_time"):
        audit_kline_data(df, source_path=Path("sample.csv"))


def test_labels_drop_final_horizon_and_use_future_window():
    df = make_kline_frame(10)
    df.loc[1:6, "high"] = df.loc[0, "close"] * 1.004
    df.loc[1:6, "low"] = df.loc[0, "close"] * 0.999

    labeled = add_future_labels(
        df,
        horizon=6,
        upside_threshold=0.003,
        drawdown_threshold=-0.002,
    )

    assert len(labeled) == 4
    assert labeled.loc[0, "future_max_return"] >= 0.003
    assert labeled.loc[0, "future_min_return"] > -0.002
    assert labeled.loc[0, "buy_label"] == 1


def test_features_exclude_target_columns_and_keep_expected_order():
    labeled = add_future_labels(make_kline_frame(90), horizon=6)

    features, feature_columns = build_feature_frame(labeled)

    assert "future_max_return" not in feature_columns
    assert "future_min_return" not in feature_columns
    assert "buy_label" not in feature_columns
    assert feature_columns[:5] == [
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "return_24",
    ]
    assert features[feature_columns].isna().sum().sum() == 0


def test_time_split_preserves_order_and_gap():
    df = make_kline_frame(100)

    split = time_based_split(df, train_ratio=0.7, valid_ratio=0.15, gap=6)

    assert split.train["open_time"].is_monotonic_increasing
    assert split.valid["open_time"].is_monotonic_increasing
    assert split.test["open_time"].is_monotonic_increasing
    assert split.valid.index.min() - split.train.index.max() == 7
    assert split.test.index.min() - split.valid.index.max() == 7


def test_threshold_selection_prefers_precision_with_min_signal_count():
    y_true = pd.Series([1, 0, 1, 0, 1, 0])
    probabilities = pd.Series([0.91, 0.58, 0.72, 0.40, 0.62, 0.10])
    future_max = pd.Series([0.010, 0.001, 0.006, -0.001, 0.004, 0.002])
    future_min = pd.Series([-0.001, -0.003, -0.001, -0.002, -0.001, -0.002])

    rows = evaluate_thresholds(
        y_true,
        probabilities,
        future_max,
        future_min,
        thresholds=[0.5, 0.7, 0.9],
    )
    selected = select_high_precision_threshold(rows, min_signals=2)

    assert selected == 0.7


def test_signal_engine_and_backtest_apply_costs():
    df = make_kline_frame(4)
    df["ma_20"] = df["close"] - 1
    df["volume_ratio_20"] = 1.5
    df["rsi_14"] = 50
    df["buy_probability"] = [0.70, 0.56, 0.20, 0.80]
    df["pred_future_max_return"] = [0.004, 0.002, 0.001, 0.005]
    df["future_max_return"] = [0.005, 0.001, -0.001, 0.003]
    df["future_min_return"] = [-0.001, -0.003, -0.002, -0.001]
    df["buy_label"] = [1, 0, 0, 1]

    signaled = assign_signals(df, buy_threshold=0.62, watch_threshold=0.55)
    metrics = backtest_triggered_signals(
        signaled,
        fee_rate_per_side=0.001,
        slippage_rate_per_side=0.0002,
    )

    assert signaled["signal"].tolist() == ["BUY", "WATCH", "NO_BUY", "BUY"]
    assert metrics["total_signals"] == 2
    assert metrics["round_trip_cost"] == pytest.approx(0.0024)
    assert metrics["estimated_return_after_costs"] == pytest.approx(0.0032)


def test_backtest_average_signals_per_day_uses_full_window():
    df = make_kline_frame(4)
    df["signal"] = ["BUY", "BUY", "NO_BUY", "NO_BUY"]
    df["future_max_return"] = [0.005, 0.001, -0.001, 0.003]
    df["future_min_return"] = [-0.001, -0.003, -0.002, -0.001]
    df["buy_label"] = [1, 0, 0, 1]

    metrics = backtest_triggered_signals(
        df,
        fee_rate_per_side=0.001,
        slippage_rate_per_side=0.0002,
    )

    elapsed_days = (
        df["open_time"].max() - df["open_time"].min()
    ).total_seconds() / 86_400
    assert metrics["average_signals_per_day"] == pytest.approx(2 / elapsed_days)
