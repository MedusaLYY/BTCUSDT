from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


BINANCE_SPOT_BASE_URL = "https://api.binance.com"
DEFAULT_TIMEOUT_SECONDS = 12
SUPPORTED_INTERVALS = {
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
}


@dataclass(frozen=True)
class MarketKline:
    openTime: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_binance_spot_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 500,
    base_url: str = BINANCE_SPOT_BASE_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[MarketKline]:
    """Fetch public Binance Spot kline data and map it to the frontend contract."""
    payload = fetch_binance_spot_raw_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )
    return [parse_binance_kline(row) for row in payload]


def fetch_binance_spot_raw_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    limit: int = 500,
    base_url: str = BINANCE_SPOT_BASE_URL,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> list[list[Any]]:
    """Fetch raw Binance Spot kline arrays for feature engineering."""
    normalized_symbol = _normalize_symbol(symbol)
    normalized_interval = _validate_interval(interval)
    normalized_limit = _validate_limit(limit)
    query = urlencode(
        {
            "symbol": normalized_symbol,
            "interval": normalized_interval,
            "limit": normalized_limit,
        }
    )
    url = f"{base_url.rstrip('/')}/api/v3/klines?{query}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "btc-short-term-predictor/0.1"})

    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, list):
        raise ValueError("unexpected Binance kline response shape")

    return payload


def parse_binance_kline(row: list[Any]) -> MarketKline:
    """Convert Binance array kline format into the frontend OHLCV object."""
    if len(row) < 6:
        raise ValueError("Binance kline row has fewer than 6 fields")

    return MarketKline(
        openTime=_format_open_time(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
    )


def market_klines_to_dicts(klines: list[MarketKline]) -> list[dict[str, float | str]]:
    return [
        {
            "openTime": item.openTime,
            "open": item.open,
            "high": item.high,
            "low": item.low,
            "close": item.close,
            "volume": item.volume,
        }
        for item in klines
    ]


def _normalize_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not normalized or not normalized.replace("_", "").isalnum():
        raise ValueError("symbol must be an alphanumeric market symbol")
    return normalized.replace("_", "")


def _validate_interval(interval: str) -> str:
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(f"unsupported interval: {interval}")
    return interval


def _validate_limit(limit: int) -> int:
    if limit < 1:
        raise ValueError("limit must be greater than 0")
    return min(limit, 1000)


def _format_open_time(open_time_ms: int) -> str:
    timestamp = open_time_ms / 1000
    shanghai_time = datetime.fromtimestamp(timestamp, tz=ZoneInfo("Asia/Shanghai"))
    return shanghai_time.strftime("%Y-%m-%d %H:%M")
