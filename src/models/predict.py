from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd


def load_feature_columns(path: str | Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def predict_from_features(
    classifier_path: str | Path,
    regressor_path: str | Path,
    feature_columns_path: str | Path,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Predict buy probability and future max return from a feature frame."""
    feature_columns = load_feature_columns(feature_columns_path)
    classifier = _load_booster(classifier_path)
    regressor = _load_booster(regressor_path)
    predictions = frame.copy()
    predictions["buy_probability"] = classifier.predict(predictions[feature_columns])
    predictions["pred_future_max_return"] = regressor.predict(
        predictions[feature_columns]
    )
    predictions["pred_high_price"] = predictions["close"] * (
        1 + predictions["pred_future_max_return"]
    )
    return predictions


def _load_booster(path: str | Path) -> lgb.Booster:
    """Load LightGBM models without passing non-ASCII paths to the native library."""
    model_text = Path(path).read_text(encoding="utf-8")
    return lgb.Booster(model_str=model_text)
