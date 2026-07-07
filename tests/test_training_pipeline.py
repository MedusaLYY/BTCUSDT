import json

import numpy as np
import pandas as pd
import pytest
import yaml

from models.train_classifier import run_training


def test_run_training_writes_expected_artifacts(tmp_path):
    rows = 240
    rng = np.random.default_rng(7)
    open_time = pd.date_range("2026-01-01", periods=rows, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.08, rows))
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.45, rows)
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.25, rows)
    volume = rng.uniform(10, 30, rows)
    data = pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "close_time": (open_time.view("int64") // 1_000_000) + 299_999,
            "quote_volume": volume * close,
            "trade_count": rng.integers(100, 500, rows),
        }
    )
    dataset_path = tmp_path / "sample.csv"
    data.to_csv(dataset_path, index=False)

    config = {
        "dataset_path": str(dataset_path),
        "symbol": "BTCUSDT",
        "interval": "5m",
        "random_seed": 7,
        "label": {
            "horizon": 6,
            "upside_threshold": 0.002,
            "drawdown_threshold": -0.003,
        },
        "split": {
            "train_ratio": 0.7,
            "valid_ratio": 0.15,
            "test_ratio": 0.15,
            "gap": 6,
        },
        "cost": {
            "fee_rate_per_side": 0.001,
            "slippage_rate_per_side": 0.0002,
        },
        "thresholds": {
            "watch": 0.55,
            "buy_candidates": [0.5, 0.6, 0.7],
            "min_validation_signals": 1,
        },
        "classifier": {
            "objective": "binary",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 2,
            "class_weight": "balanced",
        },
        "regressor": {
            "objective": "regression",
            "n_estimators": 10,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "min_child_samples": 2,
        },
        "paths": {
            "models_dir": str(tmp_path / "models"),
            "reports_dir": str(tmp_path / "reports"),
            "outputs_dir": str(tmp_path / "outputs"),
        },
    }
    config_path = tmp_path / "training.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = run_training(config_path)

    assert result["selected_threshold"] in [0.5, 0.6, 0.7]
    assert (tmp_path / "models" / "buy_classifier.txt").exists()
    assert (tmp_path / "models" / "return_regressor.txt").exists()
    assert (tmp_path / "models" / "feature_columns.json").exists()
    assert (tmp_path / "reports" / "training_metrics.json").exists()
    assert (tmp_path / "reports" / "backtest_report.json").exists()
    assert (tmp_path / "outputs" / "test_predictions.csv").exists()

    metrics = json.loads(
        (tmp_path / "reports" / "training_metrics.json").read_text(encoding="utf-8")
    )
    assert metrics["validation"]["threshold_metrics"]
    assert metrics["test"]["backtest"]["model"]["round_trip_cost"] == pytest.approx(0.0024)
