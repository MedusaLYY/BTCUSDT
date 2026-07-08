from __future__ import annotations

import pandas as pd


def add_future_labels(
    df: pd.DataFrame,
    horizon: int = 6,
    upside_threshold: float = 0.002,
    drawdown_threshold: float | None = None,
) -> pd.DataFrame:
    """Add 30-minute labels and research-only future return columns."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"open_time", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for labels: {missing}")

    labeled = df.copy().sort_values("open_time").reset_index(drop=True)
    future_highs = pd.concat(
        [labeled["high"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [labeled["low"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    future_high = future_highs.max(axis=1, skipna=False)
    future_low = future_lows.min(axis=1, skipna=False)
    future_close = labeled["close"].shift(-horizon)
    next_open = labeled["open"].shift(-1)

    labeled["future_max_return_30m"] = future_high / labeled["close"] - 1
    labeled["future_min_return_30m"] = future_low / labeled["close"] - 1
    labeled["future_close_return_30m"] = future_close / labeled["close"] - 1
    labeled["future_max_return_30m_from_next_open"] = future_high / next_open - 1
    labeled["future_min_return_30m_from_next_open"] = future_low / next_open - 1
    labeled["future_close_return_30m_from_next_open"] = future_close / next_open - 1
    labeled["y_buy"] = (
        labeled["future_max_return_30m"] > upside_threshold + 1e-12
    ).astype(int)

    return labeled.iloc[:-horizon].reset_index(drop=True)


def label_distribution_by_month(df: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize y-buy count and positive rate by open_time month."""
    if "open_time" not in df.columns or "y_buy" not in df.columns:
        raise ValueError("open_time and y_buy are required")
    grouped = df.groupby(df["open_time"].dt.to_period("M"))["y_buy"].agg(
        ["count", "mean"]
    )
    return [
        {"month": str(index), "count": int(row["count"]), "positive_rate": float(row["mean"])}
        for index, row in grouped.iterrows()
    ]
