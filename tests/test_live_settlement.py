import json
from pathlib import Path

import pandas as pd
import pytest

from live.settle_live_predictions import (
    LIVE_PREDICTION_COLUMNS,
    append_live_prediction_record,
    compute_live_metrics,
    load_live_prediction_records,
    settle_live_predictions,
)


def make_future_klines() -> pd.DataFrame:
    open_time = pd.date_range("2026-07-08 10:00", periods=8, freq="5min")
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": [100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6, 100.7],
            "high": [100.2, 100.3, 100.5, 100.9, 100.7, 100.4, 100.8, 100.9],
            "low": [99.8, 99.9, 100.0, 100.1, 100.2, 100.0, 100.3, 100.5],
            "close": [100.0, 100.2, 100.4, 100.5, 100.3, 100.2, 100.7, 100.8],
            "volume": [10.0] * 8,
            "close_time": [0] * 8,
            "quote_volume": [1000.0] * 8,
            "trade_count": [100] * 8,
        }
    )


def make_pending_record() -> dict[str, object]:
    return {
        "symbol": "BTCUSDT",
        "interval": "5m",
        "open_time": "2026-07-08 10:00",
        "close": 100.0,
        "current_price": 100.0,
        "buy_probability": 0.63,
        "predicted_max_return": 0.0030,
        "pred_high_price": 100.3,
        "signal": "BUY",
        "reason": "test reason",
    }


def test_append_live_prediction_record_writes_pending_and_deduplicates(tmp_path):
    live_path = tmp_path / "live_predictions.csv"

    append_live_prediction_record(make_pending_record(), live_path=live_path)
    append_live_prediction_record(make_pending_record(), live_path=live_path)

    frame = pd.read_csv(live_path)
    assert frame.columns.tolist() == LIVE_PREDICTION_COLUMNS
    assert len(frame) == 1
    assert frame.loc[0, "settlement_status"] == "PENDING"
    assert pd.isna(frame.loc[0, "actual_future_max_return_30m"])
    assert pd.isna(frame.loc[0, "signal_hit"])


def test_settle_live_predictions_keeps_pending_until_six_future_bars_exist(tmp_path):
    live_path = tmp_path / "live_predictions.csv"
    metrics_path = tmp_path / "live_metrics.json"
    append_live_prediction_record(make_pending_record(), live_path=live_path)

    settle_live_predictions(
        live_path=live_path,
        metrics_path=metrics_path,
        kline_frame=make_future_klines().iloc[:6],
        now=pd.Timestamp("2026-07-08 10:35"),
    )

    frame = pd.read_csv(live_path)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert frame.loc[0, "settlement_status"] == "PENDING"
    assert metrics["settled_count"] == 0
    assert metrics["pending_count"] == 1


def test_settle_live_predictions_fills_actual_result_and_errors(tmp_path):
    live_path = tmp_path / "live_predictions.csv"
    metrics_path = tmp_path / "live_metrics.json"
    append_live_prediction_record(make_pending_record(), live_path=live_path)

    settle_live_predictions(
        live_path=live_path,
        metrics_path=metrics_path,
        kline_frame=make_future_klines(),
        now=pd.Timestamp("2026-07-08 10:35"),
    )

    frame = pd.read_csv(live_path)
    row = frame.iloc[0]
    actual = 100.9 / 100.0 - 1
    assert row["settlement_status"] == "SETTLED"
    assert row["settled_at"] == "2026-07-08 10:35:00"
    assert row["actual_future_max_return_30m"] == pytest.approx(actual)
    assert bool(row["actual_y_buy"]) is True
    assert bool(row["predicted_y_buy"]) is True
    assert bool(row["classification_hit"]) is True
    assert bool(row["signal_hit"]) is True
    assert row["return_error"] == pytest.approx(0.0030 - actual)
    assert row["return_abs_error"] == pytest.approx(abs(0.0030 - actual))


def test_compute_live_metrics_uses_only_settled_records():
    frame = pd.DataFrame(
        [
            {
                "open_time": "2026-07-08 10:00",
                "settlement_status": "SETTLED",
                "signal": "BUY",
                "classification_hit": True,
                "signal_hit": True,
                "return_error": -0.001,
                "return_abs_error": 0.001,
            },
            {
                "open_time": "2026-07-08 10:05",
                "settlement_status": "SETTLED",
                "signal": "WATCH",
                "classification_hit": False,
                "signal_hit": False,
                "return_error": 0.002,
                "return_abs_error": 0.002,
            },
            {
                "open_time": "2026-07-08 10:10",
                "settlement_status": "SETTLED",
                "signal": "NO_BUY",
                "classification_hit": True,
                "signal_hit": True,
                "return_error": 0.0005,
                "return_abs_error": 0.0005,
            },
            {
                "open_time": "2026-07-08 10:15",
                "settlement_status": "PENDING",
                "signal": "BUY",
                "classification_hit": False,
                "signal_hit": False,
                "return_error": 1.0,
                "return_abs_error": 1.0,
            },
        ]
    )

    metrics = compute_live_metrics(frame, now=pd.Timestamp("2026-07-08 10:20"))

    assert metrics["settled_count"] == 3
    assert metrics["pending_count"] == 1
    assert metrics["classification_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["signal_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["buy_signal_hit_rate"] == pytest.approx(1.0)
    assert metrics["watch_signal_hit_rate"] == pytest.approx(0.0)
    assert metrics["no_buy_correct_rate"] == pytest.approx(1.0)
    assert metrics["mean_return_error"] == pytest.approx(0.0005)
    assert metrics["mean_abs_return_error"] == pytest.approx((0.001 + 0.002 + 0.0005) / 3)
    assert metrics["last_30_signal_hit_rate"] == pytest.approx(2 / 3)
    assert metrics["last_7d_signal_hit_rate"] == pytest.approx(2 / 3)


def test_load_live_prediction_records_never_reads_test_predictions(tmp_path):
    live_path = tmp_path / "live_predictions.csv"
    pd.DataFrame(
        [
            {
                **{column: pd.NA for column in LIVE_PREDICTION_COLUMNS},
                "symbol": "BTCUSDT",
                "interval": "5m",
                "open_time": "2026-07-08 10:00",
                "current_price": 100.0,
                "signal": "NO_BUY",
                "buy_probability": 0.2,
                "predicted_max_return": 0.001,
                "pred_high_price": 100.1,
                "settlement_status": "PENDING",
            }
        ]
    ).to_csv(live_path, index=False)
    (tmp_path / "test_predictions.csv").write_text("not,a,valid,live,file\n", encoding="utf-8")

    records = load_live_prediction_records(live_path=live_path, limit=30, settle=False)

    assert len(records) == 1
    assert records[0]["settlement_status"] == "PENDING"
    assert records[0]["actual_future_max_return_30m"] is None


def test_load_live_prediction_records_treats_empty_live_file_as_empty(tmp_path):
    live_path = tmp_path / "live_predictions.csv"
    live_path.write_text("", encoding="utf-8")

    records = load_live_prediction_records(live_path=live_path, limit=30, settle=False)

    assert records == []
