from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from data.fetch_binance_klines import fetch_binance_spot_raw_klines


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_LATEST = PROJECT_ROOT / "runs" / "latest"
LIVE_PREDICTIONS_PATH = RUNS_LATEST / "outputs" / "live_predictions.csv"
LIVE_METRICS_PATH = RUNS_LATEST / "reports" / "live_metrics.json"

BUY_PROBABILITY_THRESHOLD = 0.62
Y_BUY_THRESHOLD = 0.002
BUY_SIGNAL_RETURN_THRESHOLD = 0.0025
WATCH_SIGNAL_RETURN_THRESHOLD = 0.0015

LIVE_PREDICTION_COLUMNS = [
    "symbol",
    "interval",
    "open_time",
    "close",
    "current_price",
    "buy_probability",
    "predicted_max_return",
    "pred_high_price",
    "signal",
    "reason",
    "settlement_status",
    "settled_at",
    "actual_future_max_return_30m",
    "actual_y_buy",
    "predicted_y_buy",
    "classification_hit",
    "signal_hit",
    "return_error",
    "return_abs_error",
]
_LIVE_FILE_LOCK = threading.Lock()


def append_live_prediction_record(
    prediction: dict[str, Any],
    live_path: str | Path = LIVE_PREDICTIONS_PATH,
) -> pd.DataFrame:
    """Append a PENDING live prediction unless symbol/interval/open_time already exists."""
    path = Path(live_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LIVE_FILE_LOCK:
        frame = _read_live_frame(path)
        row = _pending_row(prediction)
        duplicate = (
            (frame["symbol"].astype(str) == str(row["symbol"]))
            & (frame["interval"].astype(str) == str(row["interval"]))
            & (frame["open_time"].astype(str) == str(row["open_time"]))
        )
        if not duplicate.any():
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            _write_live_frame(frame, path)
        return frame


def settle_live_predictions(
    live_path: str | Path = LIVE_PREDICTIONS_PATH,
    metrics_path: str | Path = LIVE_METRICS_PATH,
    kline_frame: pd.DataFrame | None = None,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Settle PENDING live predictions whose next six closed 5m bars are available."""
    live_file = Path(live_path)
    metrics_file = Path(metrics_path)
    with _LIVE_FILE_LOCK:
        frame = _read_live_frame(live_file)
        if now is None:
            now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)

        if not frame.empty and (frame["settlement_status"] == "PENDING").any():
            if kline_frame is None:
                kline_frame = _fetch_closed_kline_frame()
            frame = _settle_frame(frame, kline_frame, now)
            _write_live_frame(frame, live_file)

        metrics = compute_live_metrics(frame, now=now)
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        metrics_file.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics


def load_live_metrics(
    live_path: str | Path = LIVE_PREDICTIONS_PATH,
    metrics_path: str | Path = LIVE_METRICS_PATH,
) -> dict[str, Any]:
    """Settle pending rows and return live metrics."""
    return settle_live_predictions(live_path=live_path, metrics_path=metrics_path)


def load_live_prediction_records(
    live_path: str | Path = LIVE_PREDICTIONS_PATH,
    metrics_path: str | Path = LIVE_METRICS_PATH,
    limit: int = 30,
    settle: bool = True,
) -> list[dict[str, Any]]:
    """Settle pending rows and return latest live prediction records for the API."""
    if settle:
        settle_live_predictions(live_path=live_path, metrics_path=metrics_path)
    frame = _read_live_frame(live_path)
    if frame.empty:
        return []
    frame = frame.copy()
    frame["open_time_sort"] = pd.to_datetime(frame["open_time"])
    frame = frame.sort_values("open_time_sort").tail(limit)
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        records.append(
            {
                "id": f"LIVE-{index}",
                "openTime": _format_open_time(row["open_time"]),
                "currentPrice": _optional_float(row.get("current_price")),
                "signal": _optional_str(row.get("signal")),
                "buyProbability": _optional_float(row.get("buy_probability")),
                "predReturn": _optional_float(row.get("predicted_max_return")),
                "predHighPrice": _optional_float(row.get("pred_high_price")),
                "settlement_status": _optional_str(row.get("settlement_status")),
                "actual_future_max_return_30m": _optional_float(
                    row.get("actual_future_max_return_30m")
                ),
                "actual_y_buy": _optional_bool(row.get("actual_y_buy")),
                "classification_hit": _optional_bool(row.get("classification_hit")),
                "signal_hit": _optional_bool(row.get("signal_hit")),
                "return_abs_error": _optional_float(row.get("return_abs_error")),
            }
        )
    return records


def compute_live_metrics(
    frame: pd.DataFrame,
    now: pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Compute live accuracy metrics from SETTLED rows only."""
    normalized = _ensure_live_columns(frame.copy())
    if now is None:
        now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    normalized["open_time_dt"] = pd.to_datetime(normalized["open_time"], errors="coerce")
    settled = normalized[normalized["settlement_status"] == "SETTLED"].copy()
    pending_count = int((normalized["settlement_status"] == "PENDING").sum())

    return {
        "settled_count": int(len(settled)),
        "pending_count": pending_count,
        "classification_hit_rate": _bool_mean(settled["classification_hit"]),
        "signal_hit_rate": _bool_mean(settled["signal_hit"]),
        "buy_signal_hit_rate": _bool_mean(
            settled.loc[settled["signal"] == "BUY", "signal_hit"]
        ),
        "watch_signal_hit_rate": _bool_mean(
            settled.loc[settled["signal"] == "WATCH", "signal_hit"]
        ),
        "no_buy_correct_rate": _bool_mean(
            settled.loc[settled["signal"] == "NO_BUY", "signal_hit"]
        ),
        "mean_return_error": _numeric_mean(settled["return_error"]),
        "mean_abs_return_error": _numeric_mean(settled["return_abs_error"]),
        "last_30_signal_hit_rate": _last_n_hit_rate(settled, 30),
        "last_100_signal_hit_rate": _last_n_hit_rate(settled, 100),
        "last_500_signal_hit_rate": _last_n_hit_rate(settled, 500),
        "last_7d_signal_hit_rate": _last_days_hit_rate(settled, now, 7),
        "last_30d_signal_hit_rate": _last_days_hit_rate(settled, now, 30),
    }


def _settle_frame(
    frame: pd.DataFrame,
    kline_frame: pd.DataFrame,
    now: pd.Timestamp,
) -> pd.DataFrame:
    klines = kline_frame.copy()
    klines["open_time"] = pd.to_datetime(klines["open_time"])
    klines = klines.sort_values("open_time").reset_index(drop=True)
    for index, row in frame[frame["settlement_status"] == "PENDING"].iterrows():
        open_time = pd.Timestamp(row["open_time"])
        future = _future_six_bars(klines, open_time)
        if future is None:
            continue
        close = float(row["close"])
        actual = float(future["high"].astype(float).max() / close - 1)
        actual_y_buy = bool(actual > Y_BUY_THRESHOLD)
        predicted_y_buy = bool(float(row["buy_probability"]) >= BUY_PROBABILITY_THRESHOLD)
        signal_hit = _signal_hit(str(row["signal"]), actual)
        return_error = float(row["predicted_max_return"]) - actual
        frame.loc[index, "settlement_status"] = "SETTLED"
        frame.loc[index, "settled_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        frame.loc[index, "actual_future_max_return_30m"] = actual
        frame.loc[index, "actual_y_buy"] = actual_y_buy
        frame.loc[index, "predicted_y_buy"] = predicted_y_buy
        frame.loc[index, "classification_hit"] = predicted_y_buy == actual_y_buy
        frame.loc[index, "signal_hit"] = signal_hit
        frame.loc[index, "return_error"] = return_error
        frame.loc[index, "return_abs_error"] = abs(return_error)
    return frame


def _future_six_bars(klines: pd.DataFrame, open_time: pd.Timestamp) -> pd.DataFrame | None:
    expected = [open_time + pd.Timedelta(minutes=5 * offset) for offset in range(1, 7)]
    future = klines[klines["open_time"].isin(expected)].sort_values("open_time")
    if len(future) != 6:
        return None
    if future["open_time"].tolist() != expected:
        return None
    return future


def _signal_hit(signal: str, actual_future_max_return: float) -> bool | None:
    if signal == "BUY":
        return bool(actual_future_max_return >= BUY_SIGNAL_RETURN_THRESHOLD)
    if signal == "WATCH":
        return bool(actual_future_max_return >= WATCH_SIGNAL_RETURN_THRESHOLD)
    if signal == "NO_BUY":
        return bool(actual_future_max_return < Y_BUY_THRESHOLD)
    return None


def _pending_row(prediction: dict[str, Any]) -> dict[str, Any]:
    open_time = _format_open_time(prediction["open_time"])
    close = prediction.get("close", prediction.get("current_price"))
    return {
        "symbol": str(prediction.get("symbol", "BTCUSDT")).upper(),
        "interval": str(prediction.get("interval", "5m")),
        "open_time": open_time,
        "close": float(close),
        "current_price": float(prediction.get("current_price", close)),
        "buy_probability": float(prediction["buy_probability"]),
        "predicted_max_return": float(prediction["predicted_max_return"]),
        "pred_high_price": float(prediction.get("pred_high_price")),
        "signal": str(prediction["signal"]),
        "reason": str(prediction.get("reason", "")),
        "settlement_status": "PENDING",
        "settled_at": pd.NA,
        "actual_future_max_return_30m": pd.NA,
        "actual_y_buy": pd.NA,
        "predicted_y_buy": pd.NA,
        "classification_hit": pd.NA,
        "signal_hit": pd.NA,
        "return_error": pd.NA,
        "return_abs_error": pd.NA,
    }


def _read_live_frame(path: str | Path) -> pd.DataFrame:
    live_path = Path(path)
    if not live_path.exists():
        return pd.DataFrame(columns=LIVE_PREDICTION_COLUMNS)
    try:
        frame = pd.read_csv(live_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LIVE_PREDICTION_COLUMNS)
    return _ensure_live_columns(frame)


def _ensure_live_columns(frame: pd.DataFrame) -> pd.DataFrame:
    for column in LIVE_PREDICTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame.loc[:, LIVE_PREDICTION_COLUMNS].copy()
    for column in [
        "symbol",
        "interval",
        "open_time",
        "signal",
        "reason",
        "settlement_status",
        "settled_at",
        "actual_y_buy",
        "predicted_y_buy",
        "classification_hit",
        "signal_hit",
    ]:
        frame[column] = frame[column].astype("object")
    return frame


def _write_live_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    _ensure_live_columns(frame).to_csv(temp_path, index=False)
    temp_path.replace(path)


def _fetch_closed_kline_frame(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 1000,
) -> pd.DataFrame:
    raw_klines = fetch_binance_spot_raw_klines(symbol=symbol, interval=interval, limit=limit)
    now_ms = int(time.time() * 1000)
    rows: list[dict[str, Any]] = []
    for row in raw_klines:
        close_time = int(row[6])
        if close_time >= now_ms:
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


def _bool_mean(values: pd.Series) -> float | None:
    clean = values.dropna().map(_parse_bool).dropna()
    if clean.empty:
        return None
    return float(clean.astype(bool).mean())


def _numeric_mean(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _last_n_hit_rate(settled: pd.DataFrame, count: int) -> float | None:
    if settled.empty:
        return None
    ordered = settled.sort_values("open_time_dt").tail(count)
    return _bool_mean(ordered["signal_hit"])


def _last_days_hit_rate(
    settled: pd.DataFrame,
    now: pd.Timestamp,
    days: int,
) -> float | None:
    if settled.empty:
        return None
    cutoff = now - pd.Timedelta(days=days)
    recent = settled[settled["open_time_dt"] >= cutoff]
    return _bool_mean(recent["signal_hit"])


def _format_open_time(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M")


def _optional_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _optional_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    return _parse_bool(value)


def _optional_str(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _parse_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return None
    return bool(value)
