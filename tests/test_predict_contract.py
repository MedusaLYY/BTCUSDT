import json

import pandas as pd
import pytest

from models.predict import validate_feature_frame


def test_validate_feature_frame_requires_exact_saved_feature_order(tmp_path):
    feature_columns = ["return_1", "return_2", "volume_zscore_20"]
    feature_path = tmp_path / "feature_columns.json"
    feature_path.write_text(json.dumps(feature_columns), encoding="utf-8")

    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2026-01-01", periods=2, freq="5min"),
            "close": [100.0, 101.0],
            "return_1": [0.0, 0.01],
            "return_2": [0.0, 0.01],
            "volume_zscore_20": [0.1, 0.2],
        }
    )

    validated = validate_feature_frame(frame, feature_path)

    assert validated.columns.tolist() == feature_columns


def test_validate_feature_frame_fails_on_missing_reordered_or_extra_feature(tmp_path):
    feature_columns = ["return_1", "return_2", "volume_zscore_20"]
    feature_path = tmp_path / "feature_columns.json"
    feature_path.write_text(json.dumps(feature_columns), encoding="utf-8")

    missing = pd.DataFrame({"return_1": [0.0], "return_2": [0.0]})
    reordered = pd.DataFrame(
        {
            "return_2": [0.0],
            "return_1": [0.0],
            "volume_zscore_20": [0.0],
        }
    )
    extra = pd.DataFrame(
        {
            "return_1": [0.0],
            "return_2": [0.0],
            "volume_zscore_20": [0.0],
            "rogue_feature": [1.0],
        }
    )

    with pytest.raises(ValueError, match="missing required feature columns"):
        validate_feature_frame(missing, feature_path)
    with pytest.raises(ValueError, match="feature column order mismatch"):
        validate_feature_frame(reordered, feature_path)
    with pytest.raises(ValueError, match="unauthorized feature columns"):
        validate_feature_frame(extra, feature_path)
