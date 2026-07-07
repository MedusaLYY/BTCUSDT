from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
]


class AuditError(ValueError):
    """Raised when k-line data fails a non-negotiable audit check."""


@dataclass(frozen=True)
class DataAudit:
    source_path: str
    file_size_bytes: int | None
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    time_start: str | None
    time_end: str | None
    missing_values: dict[str, int]
    duplicate_open_time: int
    non_5m_intervals: int
    estimated_missing_bars: int
    price_sanity_violations: dict[str, int]
    outlier_quantiles: dict[str, dict[str, float]]
    errors: list[str]
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["passed"] = self.passed
        return payload


def load_kline_csv(path: str | Path) -> pd.DataFrame:
    """Load Binance k-line CSV data and normalize core column types."""
    csv_path = Path(path)
    df = pd.read_csv(csv_path)
    return normalize_kline_frame(df)


def normalize_kline_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with parsed times, numeric market columns, and time order."""
    normalized = df.copy()
    if "open_time" in normalized.columns:
        normalized["open_time"] = pd.to_datetime(
            normalized["open_time"], errors="coerce"
        )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
    ]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    if "open_time" in normalized.columns:
        normalized = normalized.sort_values("open_time").reset_index(drop=True)
    return normalized


def audit_kline_data(
    df: pd.DataFrame,
    source_path: str | Path,
    interval_minutes: int = 5,
    raise_on_error: bool = True,
) -> DataAudit:
    """Audit k-line data and optionally raise when hard checks fail."""
    source = Path(source_path)
    normalized = normalize_kline_frame(df)
    errors: list[str] = []
    warnings: list[str] = []

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in normalized.columns
    ]
    if missing_columns:
        errors.append(f"missing required columns: {missing_columns}")

    missing_values = {
        column: int(count)
        for column, count in normalized.isna().sum().items()
        if int(count) > 0
    }
    if missing_values:
        errors.append(f"missing values detected: {missing_values}")

    duplicate_open_time = 0
    non_5m_intervals = 0
    estimated_missing_bars = 0
    time_start: str | None = None
    time_end: str | None = None
    if "open_time" in normalized.columns:
        duplicate_open_time = int(normalized["open_time"].duplicated().sum())
        if duplicate_open_time:
            errors.append(f"duplicate open_time rows: {duplicate_open_time}")

        if not normalized["open_time"].isna().all():
            time_start = str(normalized["open_time"].min())
            time_end = str(normalized["open_time"].max())

        sorted_time = normalized["open_time"].dropna().sort_values().reset_index(
            drop=True
        )
        if len(sorted_time) > 1:
            interval_seconds = interval_minutes * 60
            diffs_seconds = sorted_time.astype("int64").diff().dropna() / 1_000_000
            bad_diffs = diffs_seconds[diffs_seconds != interval_seconds]
            non_5m_intervals = int(len(bad_diffs))
            estimated_missing_bars = int(
                ((bad_diffs / interval_seconds) - 1).clip(lower=0).sum()
            )
            if non_5m_intervals:
                errors.append(
                    "non-continuous 5-minute intervals: "
                    f"{non_5m_intervals}, estimated missing bars: "
                    f"{estimated_missing_bars}"
                )

    price_sanity_violations = _price_sanity_violations(normalized)
    for name, count in price_sanity_violations.items():
        if count:
            errors.append(f"price sanity violation {name}: {count}")

    outlier_quantiles = _outlier_quantiles(normalized)
    _add_outlier_warnings(outlier_quantiles, warnings)

    audit = DataAudit(
        source_path=str(source),
        file_size_bytes=source.stat().st_size if source.exists() else None,
        rows=len(normalized),
        columns=list(normalized.columns),
        dtypes={column: str(dtype) for column, dtype in normalized.dtypes.items()},
        time_start=time_start,
        time_end=time_end,
        missing_values=missing_values,
        duplicate_open_time=duplicate_open_time,
        non_5m_intervals=non_5m_intervals,
        estimated_missing_bars=estimated_missing_bars,
        price_sanity_violations=price_sanity_violations,
        outlier_quantiles=outlier_quantiles,
        errors=errors,
        warnings=warnings,
    )
    if raise_on_error and audit.errors:
        raise AuditError("; ".join(audit.errors))
    return audit


def _price_sanity_violations(df: pd.DataFrame) -> dict[str, int]:
    required = {"open", "high", "low", "close"}
    if not required.issubset(df.columns):
        return {
            "high_lt_low": 0,
            "high_lt_open": 0,
            "high_lt_close": 0,
            "low_gt_open": 0,
            "low_gt_close": 0,
        }
    return {
        "high_lt_low": int((df["high"] < df["low"]).sum()),
        "high_lt_open": int((df["high"] < df["open"]).sum()),
        "high_lt_close": int((df["high"] < df["close"]).sum()),
        "low_gt_open": int((df["low"] > df["open"]).sum()),
        "low_gt_close": int((df["low"] > df["close"]).sum()),
    }


def _outlier_quantiles(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    series_by_name: dict[str, pd.Series] = {}
    if "close" in df.columns:
        series_by_name["return_1"] = df["close"].pct_change()
    if {"high", "low", "close"}.issubset(df.columns):
        series_by_name["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    for column in ["volume", "quote_volume", "trade_count"]:
        if column in df.columns:
            series_by_name[column] = df[column]

    quantiles: dict[str, dict[str, float]] = {}
    for name, series in series_by_name.items():
        clean = series.replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            quantiles[name] = {}
            continue
        values = clean.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1])
        quantiles[name] = {
            str(key): float(value) for key, value in values.to_dict().items()
        }
    return quantiles


def _add_outlier_warnings(
    quantiles: dict[str, dict[str, float]], warnings: list[str]
) -> None:
    return_quantiles = quantiles.get("return_1", {})
    min_return = return_quantiles.get("0.0")
    max_return = return_quantiles.get("1.0")
    if min_return is not None and min_return < -0.05:
        warnings.append(f"extreme negative 5m return observed: {min_return:.4f}")
    if max_return is not None and max_return > 0.05:
        warnings.append(f"extreme positive 5m return observed: {max_return:.4f}")
