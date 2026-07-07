from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SplitData:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> dict[str, dict[str, object]]:
        return {
            name: _frame_summary(frame)
            for name, frame in [
                ("train", self.train),
                ("valid", self.valid),
                ("test", self.test),
            ]
        }


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    gap: int = 6,
) -> SplitData:
    """Split chronologically and drop gap rows at split boundaries."""
    if not abs((train_ratio + valid_ratio + test_ratio) - 1.0) < 1e-9:
        raise ValueError("train_ratio + valid_ratio + test_ratio must equal 1")
    if gap < 0:
        raise ValueError("gap must be non-negative")
    if "open_time" not in df.columns:
        raise ValueError("open_time is required for time-based split")

    ordered = df.sort_values("open_time")
    total_rows = len(ordered)
    train_cut = int(total_rows * train_ratio)
    valid_cut = int(total_rows * (train_ratio + valid_ratio))

    train = ordered.iloc[:train_cut].copy()
    valid = ordered.iloc[train_cut + gap : valid_cut].copy()
    test = ordered.iloc[valid_cut + gap :].copy()
    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "time split produced an empty partition; reduce gap or provide more data"
        )
    return SplitData(train=train, valid=valid, test=test)


def _frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(frame)),
        "time_start": str(frame["open_time"].min()),
        "time_end": str(frame["open_time"].max()),
    }
