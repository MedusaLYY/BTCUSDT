from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.run_backtest import backtest_triggered_signals
from data.load_data import AuditError, audit_kline_data
from features.build_features import build_feature_frame
from labels.build_labels import add_future_labels
from signals.signal_engine import assign_signals
from utils.metrics import (
    build_threshold_sweep_report,
    classifier_probability_metrics,
    evaluate_thresholds,
    regression_metrics,
    select_high_precision_threshold,
)
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
    df = make_kline_frame(8)
    df.loc[0, "close"] = 100.0
    df.loc[1, "close"] = 99.9
    df.loc[0, "high"] = 150.0
    df.loc[1:6, "high"] = 100.2
    df.loc[1:6, "low"] = 99.9

    labeled = add_future_labels(
        df,
        horizon=6,
        upside_threshold=0.002,
    )

    assert len(labeled) == 2
    assert labeled.loc[0, "future_max_return_30m"] == pytest.approx(0.002)
    assert labeled.loc[0, "y_buy"] == 0
    assert labeled.loc[1, "future_max_return_30m"] > 0.002
    assert labeled.loc[1, "y_buy"] == 1


def test_labels_use_complete_future_window_without_skipna():
    df = make_kline_frame(8)
    df.loc[3, "high"] = np.nan

    labeled = add_future_labels(df, horizon=6, upside_threshold=0.002)

    assert len(labeled) == 2
    assert pd.isna(labeled.loc[0, "future_max_return_30m"])


def test_next_open_research_labels_use_next_open_entry_price():
    df = make_kline_frame(8)
    df.loc[0, "close"] = 100.0
    df.loc[1, "open"] = 102.0
    df.loc[1:6, "high"] = [103.0, 104.0, 103.5, 103.2, 102.7, 103.1]
    df.loc[1:6, "low"] = [101.0, 100.5, 100.0, 101.5, 101.7, 100.8]
    df.loc[6, "close"] = 103.0

    labeled = add_future_labels(df, horizon=6, upside_threshold=0.002)

    assert labeled.loc[0, "future_max_return_30m_from_next_open"] == pytest.approx(
        104.0 / 102.0 - 1
    )
    assert labeled.loc[0, "future_min_return_30m_from_next_open"] == pytest.approx(
        100.0 / 102.0 - 1
    )
    assert labeled.loc[0, "future_close_return_30m_from_next_open"] == pytest.approx(
        103.0 / 102.0 - 1
    )


def test_features_exclude_target_columns_and_keep_expected_order():
    labeled = add_future_labels(make_kline_frame(120), horizon=6)
    labeled["signal"] = "NO_BUY"
    labeled["reason"] = "research only"
    labeled["predicted_max_return"] = 0.0

    features, feature_columns = build_feature_frame(labeled)

    forbidden = {
        "future_max_return_30m",
        "future_min_return_30m",
        "future_close_return_30m",
        "future_max_return_30m_from_next_open",
        "future_min_return_30m_from_next_open",
        "future_close_return_30m_from_next_open",
        "y_buy",
        "signal",
        "reason",
        "predicted_max_return",
    }
    assert forbidden.isdisjoint(feature_columns)
    assert feature_columns[:7] == [
        "return_1",
        "return_2",
        "return_3",
        "return_6",
        "return_12",
        "return_24",
        "return_48",
    ]
    assert "volume_zscore_20" in feature_columns
    assert "trade_count_change_1" in feature_columns
    assert features[feature_columns].isna().sum().sum() == 0


def test_features_do_not_change_when_future_rows_change():
    raw = make_kline_frame(120)
    changed = raw.copy()
    target_time = raw.loc[80, "open_time"]
    changed.loc[81:, ["high", "low", "close", "volume", "trade_count"]] *= 5

    base_features, feature_columns = build_feature_frame(raw)
    changed_features, _ = build_feature_frame(changed)
    base_row = base_features.loc[base_features["open_time"] == target_time, feature_columns]
    changed_row = changed_features.loc[
        changed_features["open_time"] == target_time, feature_columns
    ]

    pd.testing.assert_frame_equal(
        base_row.reset_index(drop=True),
        changed_row.reset_index(drop=True),
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


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
    df["buy_probability"] = [0.62, 0.55, 0.62, 0.80]
    df["predicted_max_return"] = [0.0025, 0.0015, 0.0014, 0.005]
    df["future_max_return_30m_from_next_open"] = [0.005, 0.001, -0.001, 0.003]
    df["future_min_return_30m_from_next_open"] = [-0.001, -0.003, -0.002, -0.001]
    df["future_close_return_30m_from_next_open"] = [0.004, -0.001, 0.001, 0.005]
    df["y_buy"] = [1, 0, 0, 1]

    signaled = assign_signals(
        df,
        buy_probability_threshold=0.62,
        buy_return_threshold=0.0025,
        watch_probability_threshold=0.55,
        watch_return_threshold=0.0015,
    )
    metrics = backtest_triggered_signals(
        signaled,
        fee_rate_per_side=0.001,
        slippage_rate_per_side=0.0002,
    )

    assert signaled["signal"].tolist() == ["BUY", "WATCH", "NO_BUY", "BUY"]
    assert metrics["total_signals"] == 2
    assert metrics["round_trip_cost"] == pytest.approx(0.0024)
    assert metrics["average_estimated_return_after_costs"] == pytest.approx(0.0021)
    assert metrics["average_future_max_return"] == pytest.approx(0.004)
    assert metrics["average_future_min_return"] == pytest.approx(-0.001)
    assert metrics["average_future_close_return"] == pytest.approx(0.0045)
    assert metrics["max_consecutive_losses"] == 0


def test_backtest_average_signals_per_day_uses_full_window():
    df = make_kline_frame(4)
    df["signal"] = ["BUY", "BUY", "NO_BUY", "NO_BUY"]
    df["future_max_return_30m_from_next_open"] = [0.005, 0.001, -0.001, 0.003]
    df["future_min_return_30m_from_next_open"] = [-0.001, -0.003, -0.002, -0.001]
    df["future_close_return_30m_from_next_open"] = [0.005, -0.001, -0.002, 0.001]
    df["y_buy"] = [1, 0, 0, 1]

    metrics = backtest_triggered_signals(
        df,
        fee_rate_per_side=0.001,
        slippage_rate_per_side=0.0002,
    )

    elapsed_days = (
        df["open_time"].max() - df["open_time"].min()
    ).total_seconds() / 86_400
    assert metrics["average_signals_per_day"] == pytest.approx(2 / elapsed_days)
    assert metrics["max_consecutive_losses"] == 1


def test_probability_and_regression_metrics_include_calibration_and_error_analysis():
    y_true = pd.Series([1, 0, 1, 0, 1, 0, 0, 1, 0, 1])
    probabilities = pd.Series(
        [0.95, 0.10, 0.80, 0.20, 0.70, 0.30, 0.40, 0.60, 0.05, 0.55]
    )

    probability_metrics = classifier_probability_metrics(y_true, probabilities)

    assert probability_metrics["auc"] == pytest.approx(1.0)
    assert probability_metrics["average_precision"] == pytest.approx(1.0)
    assert probability_metrics["brier_score"] < 0.2
    assert probability_metrics["top_5pct_hit_rate"] == pytest.approx(1.0)
    assert probability_metrics["top_10pct_hit_rate"] == pytest.approx(1.0)
    assert probability_metrics["buy_signal_count"] == 3
    assert probability_metrics["buy_signal_rate"] == pytest.approx(0.3)
    assert probability_metrics["calibration_buckets"]

    regression = regression_metrics(
        pd.Series([0.001, 0.002, 0.003, 0.004]),
        pd.Series([0.0015, 0.0015, 0.0025, 0.0045]),
    )

    assert regression["mae"] == pytest.approx(0.0005)
    assert regression["rmse"] == pytest.approx(0.0005)
    assert regression["pearson_corr"] > 0.8
    assert "p50" in regression["absolute_error_quantiles"]


def test_threshold_sweep_report_compares_probability_and_return_thresholds():
    frame = pd.DataFrame(
        {
            "buy_probability": [0.70, 0.63, 0.58, 0.40],
            "predicted_max_return": [0.0030, 0.0026, 0.0018, 0.0040],
            "y_buy": [1, 0, 1, 0],
            "future_close_return_30m_from_next_open": [0.004, -0.001, 0.002, 0.005],
        }
    )

    sweep = build_threshold_sweep_report(
        frame,
        dataset_name="validation",
        probability_thresholds=[0.62],
        return_thresholds=[0.0025],
        round_trip_cost=0.0024,
    )

    assert sweep.to_dict("records") == [
        {
            "dataset": "validation",
            "buy_probability_threshold": 0.62,
            "predicted_max_return_threshold": 0.0025,
            "buy_count": 2,
            "buy_rate": 0.5,
            "hit_rate": 0.5,
            "average_return_after_costs": pytest.approx(-0.0009),
            "total_return_after_costs": pytest.approx(-0.0018),
            "max_drawdown": pytest.approx(-0.0034),
        }
    ]
