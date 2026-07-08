from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from data.fetch_binance_klines import fetch_binance_spot_raw_klines
from features.build_features import build_feature_frame
from models.predict import predict_from_features
from signals.signal_engine import assign_signals


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_LATEST = PROJECT_ROOT / "runs" / "latest"
MODEL_DIR = RUNS_LATEST / "models"
REPORT_DIR = RUNS_LATEST / "reports"
OUTPUT_DIR = RUNS_LATEST / "outputs"


def build_latest_prediction_signal(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 500,
) -> dict[str, Any]:
    raw_klines = fetch_binance_spot_raw_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
    )
    frame = _raw_klines_to_feature_input(raw_klines, closed_only=True)
    featured, _ = build_feature_frame(frame)
    if featured.empty:
        raise ValueError("not enough closed kline history to build features")

    predicted = predict_from_features(
        classifier_path=MODEL_DIR / "buy_classifier.txt",
        regressor_path=MODEL_DIR / "return_regressor.txt",
        feature_columns_path=MODEL_DIR / "feature_columns.json",
        frame=featured,
    )
    signaled = assign_signals(predicted.tail(1), **_signal_thresholds())
    row = signaled.iloc[-1]
    reason = [item.strip() for item in str(row["reason"]).split(";") if item.strip()]

    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "openTime": _format_open_time(row["open_time"]),
        "currentPrice": float(row["close"]),
        "signal": str(row["signal"]),
        "buyProbability": float(row["buy_probability"]),
        "predReturn": float(row["predicted_max_return"]),
        "predHighPrice": float(row["pred_high_price"]),
        "reason": reason,
    }


def load_historical_signal_records(limit: int = 500) -> list[dict[str, Any]]:
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    frame = pd.read_csv(predictions_path).tail(limit).copy()
    records: list[dict[str, Any]] = []

    for index, row in frame.iterrows():
        actual_return = float(row["future_max_return_30m"])
        pred_return = float(row["predicted_max_return"])
        hit_status = "HIT" if actual_return >= pred_return * 0.72 else "MISS"
        records.append(
            {
                "id": f"HIST-{index}",
                "openTime": _format_open_time(row["open_time"]),
                "currentPrice": float(row["current_price"]),
                "signal": str(row["signal"]),
                "buyProbability": float(row["buy_probability"]),
                "predReturn": pred_return,
                "predHighPrice": float(row["pred_high_price"]),
                "actualFutureMaxReturn": actual_return,
                "hitStatus": hit_status,
            }
        )

    return records


def load_backtest_summary() -> dict[str, Any]:
    report = json.loads((REPORT_DIR / "backtest_report.json").read_text(encoding="utf-8"))
    metrics = report["model"]
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    counts = predictions["signal"].value_counts()

    return {
        "totalSignals": int(len(predictions)),
        "buySignals": int(counts.get("BUY", 0)),
        "watchSignals": int(counts.get("WATCH", 0)),
        "winRate": float(metrics["win_rate"]),
        "precision": float(metrics["precision_on_triggered_signals"]),
        "avgFutureMaxReturn": float(metrics["average_future_max_return"]),
        "avgFutureMinReturn": float(metrics["average_future_min_return"]),
        "avgSignalsPerDay": float(metrics["average_signals_per_day"]),
        "maxDrawdown": float(metrics["max_drawdown"]),
    }


def load_equity_curve() -> list[dict[str, Any]]:
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    predictions["date"] = pd.to_datetime(predictions["open_time"]).dt.strftime("%Y-%m-%d")
    buy_signals = predictions[predictions["signal"] == "BUY"].copy()
    buy_signals["net_return"] = (
        buy_signals["future_close_return_30m_from_next_open"] - 0.0024
    )
    daily_returns = buy_signals.groupby("date")["net_return"].sum().to_dict()
    dates = sorted(predictions["date"].unique())

    equity = 10000.0
    points: list[dict[str, Any]] = []
    for date in dates:
        equity *= 1 + float(daily_returns.get(date, 0.0))
        points.append({"time": date, "equity": round(equity, 2)})

    return points


def load_signal_distribution() -> list[dict[str, Any]]:
    predictions = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    counts = predictions["signal"].value_counts()

    return [
        {"signal": "BUY", "count": int(counts.get("BUY", 0))},
        {"signal": "WATCH", "count": int(counts.get("WATCH", 0))},
        {"signal": "NO_BUY", "count": int(counts.get("NO_BUY", 0))},
    ]


def load_precision_threshold() -> list[dict[str, Any]]:
    metrics = json.loads((REPORT_DIR / "training_metrics.json").read_text(encoding="utf-8"))
    rows = metrics["validation"]["threshold_metrics"]

    return [
        {
            "threshold": float(row["threshold"]),
            "precision": float(row["precision"]),
            "signalCount": int(row["signal_count"]),
        }
        for row in rows
    ]


def _raw_klines_to_feature_input(raw_klines: list[list[Any]], closed_only: bool) -> pd.DataFrame:
    now_ms = int(time.time() * 1000)
    rows: list[dict[str, Any]] = []
    for row in raw_klines:
        close_time = int(row[6])
        if closed_only and close_time >= now_ms:
            continue
        rows.append(
            {
                "open_time": pd.to_datetime(int(row[0]), unit="ms", utc=True)
                .tz_convert("Asia/Shanghai")
                .tz_localize(None),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
                "close_time": close_time,
                "quote_volume": float(row[7]),
                "trade_count": int(row[8]),
            }
        )

    return pd.DataFrame(rows)


def _signal_thresholds() -> dict[str, float]:
    metadata_path = MODEL_DIR / "model_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        thresholds = metadata["signal_thresholds"]
        return {
            "buy_probability_threshold": float(thresholds["buy_probability"]),
            "buy_return_threshold": float(thresholds["buy_return"]),
            "watch_probability_threshold": float(thresholds["watch_probability"]),
            "watch_return_threshold": float(thresholds["watch_return"]),
        }
    return {
        "buy_probability_threshold": 0.62,
        "buy_return_threshold": 0.0025,
        "watch_probability_threshold": 0.55,
        "watch_return_threshold": 0.0015,
    }


def _format_open_time(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")
