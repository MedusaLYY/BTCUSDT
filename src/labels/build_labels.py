from __future__ import annotations

import pandas as pd


def add_future_labels(
    df: pd.DataFrame,
    horizon: int = 6,
    upside_threshold: float = 0.003,
    drawdown_threshold: float = -0.002,
) -> pd.DataFrame:
    """Add 30-minute future return targets and drop incomplete final rows."""
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    required = {"high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for labels: {missing}")

    labeled = df.copy().reset_index(drop=True)
    future_highs = pd.concat(
        [labeled["high"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [labeled["low"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    labeled["future_max_return"] = future_highs.max(axis=1) / labeled["close"] - 1
    labeled["future_min_return"] = future_lows.min(axis=1) / labeled["close"] - 1
    labeled["buy_label"] = (
        (labeled["future_max_return"] >= upside_threshold)
        & (labeled["future_min_return"] > drawdown_threshold)
    ).astype(int)

    return labeled.iloc[:-horizon].reset_index(drop=True)


def label_distribution_by_month(df: pd.DataFrame) -> list[dict[str, object]]:
    """Summarize buy-label count and positive rate by open_time month."""
    if "open_time" not in df.columns or "buy_label" not in df.columns:
        raise ValueError("open_time and buy_label are required")
    grouped = df.groupby(df["open_time"].dt.to_period("M"))["buy_label"].agg(
        ["count", "mean"]
    )
    return [
        {"month": str(index), "count": int(row["count"]), "positive_rate": float(row["mean"])}
        for index, row in grouped.iterrows()
    ]
