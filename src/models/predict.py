from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd


AUTHORIZED_PASSTHROUGH_COLUMNS = {
    "symbol",
    "interval",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "current_price",
    "future_max_return_30m",
    "future_min_return_30m",
    "future_close_return_30m",
    "future_max_return_30m_from_next_open",
    "future_min_return_30m_from_next_open",
    "future_close_return_30m_from_next_open",
    "y_buy",
    "buy_probability",
    "predicted_max_return",
    "pred_high_price",
    "signal",
    "reason",
}


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_feature_frame(
    frame: pd.DataFrame,
    feature_columns_path: str | Path,
) -> pd.DataFrame:
    """Fail fast unless frame contains exactly the saved feature columns in order."""
    feature_columns = load_feature_columns(feature_columns_path)
    missing = [column for column in feature_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required feature columns: {missing}")

    unauthorized = [
        column
        for column in frame.columns
        if column not in feature_columns and column not in AUTHORIZED_PASSTHROUGH_COLUMNS
    ]
    if unauthorized:
        raise ValueError(f"unauthorized feature columns: {unauthorized}")

    present_feature_columns = [
        column for column in frame.columns if column in set(feature_columns)
    ]
    if present_feature_columns != feature_columns:
        raise ValueError(
            "feature column order mismatch: frame feature order does not match "
            "feature_columns.json"
        )
    return frame.loc[:, feature_columns]


def predict_from_features(
    classifier_path: str | Path,
    regressor_path: str | Path,
    feature_columns_path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict buy probability and future max return from a validated feature frame."""
    feature_matrix = validate_feature_frame(frame, feature_columns_path)
    classifier = _load_booster(classifier_path)
    regressor = _load_booster(regressor_path)
    predictions = frame.copy()
    predictions["buy_probability"] = classifier.predict(feature_matrix)
    predictions["predicted_max_return"] = regressor.predict(feature_matrix)
    predictions["pred_high_price"] = predictions["close"] * (
        1 + predictions["predicted_max_return"]
    )
    return predictions


def _load_booster(path: str | Path) -> lgb.Booster:
    """Load LightGBM models without passing non-ASCII paths to the native library."""
    model_text = Path(path).read_text(encoding="utf-8")
    return lgb.Booster(model_str=model_text)
