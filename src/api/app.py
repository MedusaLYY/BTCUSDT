from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from data.fetch_binance_klines import (
    fetch_binance_spot_klines,
    market_klines_to_dicts,
)
from api.realtime_signal import (
    build_latest_prediction_signal,
    load_backtest_summary,
    load_equity_curve,
    load_historical_signal_records,
    load_precision_threshold,
    load_signal_distribution,
)
from live.settle_live_predictions import (
    load_live_metrics,
    load_live_prediction_records,
)


app = FastAPI(title="BTCUSDT Short-Term Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "http://127.0.0.1:5174",
        "http://localhost:5174",
    ],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/klines")
def get_klines(
    symbol: str = Query(default="BTCUSDT", min_length=1, max_length=20),
    interval: str = Query(default="5m", min_length=1, max_length=4),
    limit: int = Query(default=500, ge=1, le=1000),
) -> list[dict[str, float | str]]:
    try:
        klines = fetch_binance_spot_klines(
            symbol=symbol,
            interval=interval,
            limit=limit,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"failed to fetch Binance klines: {error}",
        ) from error

    return market_klines_to_dicts(klines)


@app.get("/api/latest-signal")
def get_latest_signal() -> dict[str, object]:
    try:
        return build_latest_prediction_signal()
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"failed to build latest signal: {error}",
        ) from error


@app.get("/api/signals")
def get_signals() -> list[dict[str, object]]:
    return load_historical_signal_records()


@app.get("/api/live/metrics")
def get_live_metrics() -> dict[str, object]:
    return load_live_metrics()


@app.get("/api/live/predictions")
def get_live_predictions(
    limit: int = Query(default=30, ge=1, le=500),
) -> list[dict[str, object]]:
    return load_live_prediction_records(limit=limit)


@app.get("/api/backtest/summary")
def get_backtest_summary() -> dict[str, object]:
    return load_backtest_summary()


@app.get("/api/backtest/equity-curve")
def get_backtest_equity_curve() -> list[dict[str, object]]:
    return load_equity_curve()


@app.get("/api/backtest/signal-distribution")
def get_backtest_signal_distribution() -> list[dict[str, object]]:
    return load_signal_distribution()


@app.get("/api/backtest/precision-threshold")
def get_backtest_precision_threshold() -> list[dict[str, object]]:
    return load_precision_threshold()
