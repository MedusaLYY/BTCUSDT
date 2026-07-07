from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_triggered_signals(
    df: pd.DataFrame,
    fee_rate_per_side: float,
    slippage_rate_per_side: float,
) -> dict[str, float | int | None]:
    """Evaluate BUY rows using future-window returns and round-trip costs."""
    required = {"signal", "future_max_return", "future_min_return", "buy_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns for backtest: {missing}")

    triggered = df[df["signal"] == "BUY"].copy()
    round_trip_cost = 2 * (fee_rate_per_side + slippage_rate_per_side)
    if triggered.empty:
        return {
            "total_signals": 0,
            "average_signals_per_day": 0.0,
            "win_rate": None,
            "precision_on_triggered_signals": None,
            "average_future_max_return": None,
            "average_future_min_return": None,
            "round_trip_cost": float(round_trip_cost),
            "estimated_return_after_costs": 0.0,
            "average_estimated_return_after_costs": None,
            "max_drawdown": 0.0,
            "profit_factor": None,
        }

    net_returns = triggered["future_max_return"].astype(float) - round_trip_cost
    return {
        "total_signals": int(len(triggered)),
        "average_signals_per_day": _signals_per_day(df, len(triggered)),
        "win_rate": float((net_returns > 0).mean()),
        "precision_on_triggered_signals": float(triggered["buy_label"].mean()),
        "average_future_max_return": float(triggered["future_max_return"].mean()),
        "average_future_min_return": float(triggered["future_min_return"].mean()),
        "round_trip_cost": float(round_trip_cost),
        "estimated_return_after_costs": float(net_returns.sum()),
        "average_estimated_return_after_costs": float(net_returns.mean()),
        "max_drawdown": float(_max_drawdown(net_returns)),
        "profit_factor": _profit_factor(net_returns),
    }


def buy_and_hold_reference(df: pd.DataFrame) -> dict[str, float]:
    """Return simple close-to-close buy-and-hold reference for the test window."""
    if df.empty:
        return {"return": 0.0}
    start = float(df["close"].iloc[0])
    end = float(df["close"].iloc[-1])
    return {"return": end / start - 1}


def make_random_baseline(
    df: pd.DataFrame,
    signal_count: int,
    seed: int,
) -> pd.DataFrame:
    """Create a random BUY baseline with the same number of signals."""
    baseline = df.copy()
    baseline["signal"] = "NO_BUY"
    if signal_count <= 0 or baseline.empty:
        return baseline
    rng = np.random.default_rng(seed)
    count = min(signal_count, len(baseline))
    selected = rng.choice(baseline.index.to_numpy(), size=count, replace=False)
    baseline.loc[selected, "signal"] = "BUY"
    return baseline


def make_rule_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Rule-only baseline from trend and volume filters."""
    baseline = df.copy()
    rule = (baseline["close"] > baseline["ma_20"]) & (
        baseline["volume_ratio_20"] > 1.1
    )
    baseline["signal"] = np.where(rule, "BUY", "NO_BUY")
    return baseline


def _signals_per_day(frame: pd.DataFrame, signal_count: int) -> float:
    if "open_time" not in frame.columns or frame.empty or signal_count == 0:
        return 0.0
    elapsed_days = (
        frame["open_time"].max() - frame["open_time"].min()
    ).total_seconds() / 86_400
    if elapsed_days <= 0:
        return float(signal_count)
    return float(signal_count / elapsed_days)


def _max_drawdown(returns: pd.Series) -> float:
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())


def _profit_factor(returns: pd.Series) -> float | None:
    gains = returns[returns > 0].sum()
    losses = returns[returns < 0].sum()
    if losses == 0:
        return None
    return float(gains / abs(losses))
