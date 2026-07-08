from __future__ import annotations

import numpy as np
import pandas as pd


def assign_signals(
    df: pd.DataFrame,
    buy_probability_threshold: float = 0.62,
    buy_return_threshold: float = 0.0025,
    watch_probability_threshold: float = 0.55,
    watch_return_threshold: float = 0.0015,
    buy_threshold: float | None = None,
    watch_threshold: float | None = None,
) -> pd.DataFrame:
    """Assign BUY/WATCH/NO_BUY research labels from dual-model outputs."""
    if buy_threshold is not None:
        buy_probability_threshold = buy_threshold
    if watch_threshold is not None:
        watch_probability_threshold = watch_threshold

    required = {"buy_probability", "predicted_max_return"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for signal engine: {missing}")

    signaled = df.copy()
    buy_condition = (
        (signaled["buy_probability"] >= buy_probability_threshold)
        & (signaled["predicted_max_return"] >= buy_return_threshold)
    )
    watch_condition = (
        (signaled["buy_probability"] >= watch_probability_threshold)
        & (signaled["predicted_max_return"] >= watch_return_threshold)
    )

    signaled["signal"] = np.select(
        [buy_condition, watch_condition],
        ["BUY", "WATCH"],
        default="NO_BUY",
    )
    signaled["reason"] = [
        _reason(
            row,
            buy_probability_threshold=buy_probability_threshold,
            buy_return_threshold=buy_return_threshold,
            watch_probability_threshold=watch_probability_threshold,
            watch_return_threshold=watch_return_threshold,
        )
        for _, row in signaled.iterrows()
    ]
    return signaled


def _reason(
    row: pd.Series,
    buy_probability_threshold: float,
    buy_return_threshold: float,
    watch_probability_threshold: float,
    watch_return_threshold: float,
) -> str:
    probability = float(row["buy_probability"])
    predicted_return = float(row["predicted_max_return"])
    if row["signal"] == "BUY":
        return (
            f"buy_probability {probability:.4f} >= {buy_probability_threshold:.4f}; "
            f"predicted_max_return {predicted_return:.4f} >= {buy_return_threshold:.4f}"
        )
    if row["signal"] == "WATCH":
        return (
            f"buy_probability {probability:.4f} >= {watch_probability_threshold:.4f}; "
            f"predicted_max_return {predicted_return:.4f} >= {watch_return_threshold:.4f}"
        )
    return (
        f"buy_probability {probability:.4f} or predicted_max_return "
        f"{predicted_return:.4f} below WATCH thresholds"
    )
