from __future__ import annotations

import numpy as np
import pandas as pd


def assign_signals(
    df: pd.DataFrame,
    buy_threshold: float,
    watch_threshold: float = 0.55,
) -> pd.DataFrame:
    """Assign conservative BUY/WATCH/NO_BUY labels from model probabilities."""
    required = {"buy_probability", "close", "ma_20", "volume_ratio_20", "rsi_14"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for signal engine: {missing}")

    signaled = df.copy()
    buy_condition = (
        (signaled["buy_probability"] > buy_threshold)
        & (signaled["close"] > signaled["ma_20"])
        & (signaled["volume_ratio_20"] > 1.1)
        & (signaled["rsi_14"] < 75)
    )
    watch_condition = signaled["buy_probability"] > watch_threshold

    signaled["signal"] = np.select(
        [buy_condition, watch_condition],
        ["BUY", "WATCH"],
        default="NO_BUY",
    )
    signaled["reason"] = [
        _reason(row, buy_threshold, watch_threshold)
        for _, row in signaled.iterrows()
    ]
    return signaled


def _reason(row: pd.Series, buy_threshold: float, watch_threshold: float) -> str:
    probability = float(row["buy_probability"])
    if row["signal"] == "BUY":
        return (
            f"probability {probability:.4f} > {buy_threshold:.2f}; "
            "close > ma_20; volume_ratio_20 > 1.1; rsi_14 < 75"
        )
    if row["signal"] == "WATCH":
        return f"probability {probability:.4f} > watch threshold {watch_threshold:.2f}"
    return f"probability {probability:.4f} <= watch threshold {watch_threshold:.2f}"
