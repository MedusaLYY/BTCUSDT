from __future__ import annotations

import argparse
import json
import platform
import shutil
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from lightgbm import LGBMClassifier

from backtest.run_backtest import (
    backtest_triggered_signals,
    buy_and_hold_reference,
    make_random_baseline,
    make_rule_baseline,
)
from data.load_data import audit_kline_data, load_kline_csv
from features.build_features import build_feature_frame
from labels.build_labels import add_future_labels, label_distribution_by_month
from models.train_regressor import train_return_regressor
from signals.signal_engine import assign_signals
from utils.metrics import (
    classifier_probability_metrics,
    evaluate_thresholds,
    regression_metrics,
    select_high_precision_threshold,
)
from utils.time_split import time_based_split


def run_training(config_path: str | Path) -> dict[str, Any]:
    """Run the full audit, train, validate, test, and reporting pipeline."""
    config_file = Path(config_path).resolve()
    config = _load_config(config_file)
    root = _project_root(config_file)
    random_seed = int(config.get("random_seed", 42))

    models_dir = _resolve_path(config["paths"]["models_dir"], root)
    reports_dir = _resolve_path(config["paths"]["reports_dir"], root)
    outputs_dir = _resolve_path(config["paths"]["outputs_dir"], root)
    for directory in [models_dir, reports_dir, outputs_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    dataset_path = _resolve_path(config["dataset_path"], root)
    raw = load_kline_csv(dataset_path)
    audit = audit_kline_data(raw, source_path=dataset_path)

    label_cfg = config["label"]
    labeled = add_future_labels(
        raw,
        horizon=int(label_cfg["horizon"]),
        upside_threshold=float(label_cfg["upside_threshold"]),
        drawdown_threshold=float(label_cfg["drawdown_threshold"]),
    )
    featured, feature_columns = build_feature_frame(labeled)

    split_cfg = config["split"]
    split = time_based_split(
        featured,
        train_ratio=float(split_cfg["train_ratio"]),
        valid_ratio=float(split_cfg["valid_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        gap=int(split_cfg["gap"]),
    )
    _require_two_classes(split.train["buy_label"], "train")

    classifier = train_buy_classifier(
        split.train,
        split.valid,
        feature_columns,
        params=config["classifier"],
        random_seed=random_seed,
    )
    regressor = train_return_regressor(
        split.train,
        split.valid,
        feature_columns,
        params=config["regressor"],
        random_seed=random_seed,
    )

    valid_prob = pd.Series(
        classifier.predict_proba(split.valid[feature_columns])[:, 1],
        index=split.valid.index,
    )
    test_prob = pd.Series(
        classifier.predict_proba(split.test[feature_columns])[:, 1],
        index=split.test.index,
    )
    valid_reg_pred = pd.Series(
        regressor.predict(split.valid[feature_columns]), index=split.valid.index
    )
    test_reg_pred = pd.Series(
        regressor.predict(split.test[feature_columns]), index=split.test.index
    )

    threshold_cfg = config["thresholds"]
    threshold_rows = evaluate_thresholds(
        split.valid["buy_label"],
        valid_prob,
        split.valid["future_max_return"],
        split.valid["future_min_return"],
        thresholds=threshold_cfg["buy_candidates"],
    )
    selected_threshold = select_high_precision_threshold(
        threshold_rows,
        min_signals=int(threshold_cfg.get("min_validation_signals", 50)),
    )

    test_predictions = _build_prediction_frame(
        split.test,
        test_prob,
        test_reg_pred,
        selected_threshold,
        watch_threshold=float(threshold_cfg.get("watch", 0.55)),
        symbol=str(config.get("symbol", "BTCUSDT")),
        interval=str(config.get("interval", "5m")),
    )
    test_predictions_path = outputs_dir / "test_predictions.csv"
    test_predictions.to_csv(test_predictions_path, index=False)

    cost_cfg = config["cost"]
    fee_rate = float(cost_cfg["fee_rate_per_side"])
    slippage_rate = float(cost_cfg["slippage_rate_per_side"])
    model_backtest = backtest_triggered_signals(
        test_predictions,
        fee_rate_per_side=fee_rate,
        slippage_rate_per_side=slippage_rate,
    )
    random_baseline = make_random_baseline(
        test_predictions,
        signal_count=int(model_backtest["total_signals"]),
        seed=random_seed,
    )
    rule_baseline = make_rule_baseline(test_predictions)
    backtest_report = {
        "model": model_backtest,
        "random_signal_baseline": backtest_triggered_signals(
            random_baseline,
            fee_rate_per_side=fee_rate,
            slippage_rate_per_side=slippage_rate,
        ),
        "rule_only_baseline": backtest_triggered_signals(
            rule_baseline,
            fee_rate_per_side=fee_rate,
            slippage_rate_per_side=slippage_rate,
        ),
        "buy_and_hold_reference": buy_and_hold_reference(test_predictions),
    }

    classifier_path = models_dir / "buy_classifier.txt"
    regressor_path = models_dir / "return_regressor.txt"
    feature_columns_path = models_dir / "feature_columns.json"
    _save_booster_model(classifier.booster_, classifier_path)
    _save_booster_model(regressor.booster_, regressor_path)
    _write_json(feature_columns_path, feature_columns)

    metrics = _build_metrics_payload(
        config=config,
        dataset_path=dataset_path,
        audit=audit.to_dict(),
        labeled=labeled,
        featured=featured,
        split=split,
        feature_columns=feature_columns,
        classifier=classifier,
        valid_prob=valid_prob,
        test_prob=test_prob,
        threshold_rows=threshold_rows,
        selected_threshold=selected_threshold,
        valid_reg_pred=valid_reg_pred,
        test_reg_pred=test_reg_pred,
        backtest_report=backtest_report,
        artifacts={
            "classifier": classifier_path,
            "regressor": regressor_path,
            "feature_columns": feature_columns_path,
            "test_predictions": test_predictions_path,
        },
    )
    metrics_path = reports_dir / "training_metrics.json"
    backtest_path = reports_dir / "backtest_report.json"
    report_path = reports_dir / "training_report.md"
    _write_json(metrics_path, metrics)
    _write_json(backtest_path, backtest_report)
    report_path.write_text(_render_markdown_report(metrics), encoding="utf-8")

    return {
        "selected_threshold": selected_threshold,
        "artifacts": {
            "classifier": str(classifier_path),
            "regressor": str(regressor_path),
            "feature_columns": str(feature_columns_path),
            "training_metrics": str(metrics_path),
            "backtest_report": str(backtest_path),
            "training_report": str(report_path),
            "test_predictions": str(test_predictions_path),
        },
    }


def train_buy_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
    random_seed: int,
) -> LGBMClassifier:
    """Train buy-label classifier without shuffling chronological data."""
    model_params = dict(params)
    model_params.setdefault("random_state", random_seed)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", -1)
    model = LGBMClassifier(**model_params)
    model.fit(
        train[feature_columns],
        train["buy_label"],
        eval_set=[(valid[feature_columns], valid["buy_label"])],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _build_prediction_frame(
    frame: pd.DataFrame,
    probabilities: pd.Series,
    predicted_returns: pd.Series,
    selected_threshold: float,
    watch_threshold: float,
    symbol: str,
    interval: str,
) -> pd.DataFrame:
    predictions = frame.copy()
    predictions["symbol"] = symbol
    predictions["interval"] = interval
    predictions["current_price"] = predictions["close"]
    predictions["buy_probability"] = probabilities.to_numpy()
    predictions["pred_future_max_return"] = predicted_returns.to_numpy()
    predictions["pred_high_price"] = predictions["close"] * (
        1 + predictions["pred_future_max_return"]
    )
    return assign_signals(
        predictions,
        buy_threshold=selected_threshold,
        watch_threshold=watch_threshold,
    )


def _build_metrics_payload(
    config: dict[str, Any],
    dataset_path: Path,
    audit: dict[str, Any],
    labeled: pd.DataFrame,
    featured: pd.DataFrame,
    split,
    feature_columns: list[str],
    classifier: LGBMClassifier,
    valid_prob: pd.Series,
    test_prob: pd.Series,
    threshold_rows: list[dict[str, object]],
    selected_threshold: float,
    valid_reg_pred: pd.Series,
    test_reg_pred: pd.Series,
    backtest_report: dict[str, Any],
    artifacts: dict[str, Path],
) -> dict[str, Any]:
    test_threshold_rows = evaluate_thresholds(
        split.test["buy_label"],
        test_prob,
        split.test["future_max_return"],
        split.test["future_min_return"],
        thresholds=[selected_threshold],
    )
    return {
        "dataset": {
            "path": str(dataset_path),
            "row_count": int(audit["rows"]),
            "time_start": audit["time_start"],
            "time_end": audit["time_end"],
            "audit": audit,
        },
        "label": {
            **config["label"],
            "labeled_rows": int(len(labeled)),
            "positive_count": int(labeled["buy_label"].sum()),
            "positive_rate": float(labeled["buy_label"].mean()),
            "distribution_by_month": label_distribution_by_month(labeled),
        },
        "features": {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "rows_after_feature_dropna": int(len(featured)),
        },
        "split": split.summary(),
        "model_config": {
            "classifier": config["classifier"],
            "regressor": config["regressor"],
            "random_seed": int(config.get("random_seed", 42)),
        },
        "validation": {
            "probability_metrics": classifier_probability_metrics(
                split.valid["buy_label"], valid_prob
            ),
            "threshold_metrics": threshold_rows,
            "selected_threshold": selected_threshold,
            "regression": regression_metrics(
                split.valid["future_max_return"], valid_reg_pred
            ),
        },
        "test": {
            "probability_metrics": classifier_probability_metrics(
                split.test["buy_label"], test_prob
            ),
            "threshold_metrics_at_selected_threshold": test_threshold_rows,
            "regression": regression_metrics(
                split.test["future_max_return"], test_reg_pred
            ),
            "backtest": backtest_report,
        },
        "feature_importance": _feature_importance(classifier, feature_columns),
        "runtime": _runtime_versions(),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "limitations": [
            "This is a research pipeline, not financial advice.",
            "The selected threshold is chosen on validation data only.",
            "Test backtest results must not be used for tuning this run.",
            "Signals use future-window labels for evaluation, not guaranteed trade fills.",
        ],
        "ready_for_paper_trading": _ready_for_paper_trading(backtest_report),
    }


def _feature_importance(
    classifier: LGBMClassifier, feature_columns: list[str]
) -> list[dict[str, Any]]:
    pairs = zip(feature_columns, classifier.feature_importances_)
    return [
        {"feature": feature, "importance": int(importance)}
        for feature, importance in sorted(pairs, key=lambda item: item[1], reverse=True)
    ]


def _ready_for_paper_trading(backtest_report: dict[str, Any]) -> bool:
    model = backtest_report["model"]
    return bool(
        model["total_signals"]
        and model["precision_on_triggered_signals"] is not None
        and model["precision_on_triggered_signals"] >= 0.5
        and model["estimated_return_after_costs"] > 0
    )


def _render_markdown_report(metrics: dict[str, Any]) -> str:
    validation = metrics["validation"]
    test = metrics["test"]
    model_backtest = test["backtest"]["model"]
    ready = "yes" if metrics["ready_for_paper_trading"] else "no"
    top_features = "\n".join(
        f"- {row['feature']}: {row['importance']}"
        for row in metrics["feature_importance"][:15]
    )
    return f"""# BTCUSDT Short-Term Training Report

## Data Overview

- Dataset: `{metrics['dataset']['path']}`
- Rows: {metrics['dataset']['row_count']}
- Time range: {metrics['dataset']['time_start']} -> {metrics['dataset']['time_end']}
- Audit passed: {metrics['dataset']['audit']['passed']}

## Label Definition

- Horizon: {metrics['label']['horizon']} bars
- Upside threshold: {metrics['label']['upside_threshold']}
- Drawdown threshold: {metrics['label']['drawdown_threshold']}
- Positive rate: {metrics['label']['positive_rate']:.4%}

## Feature List

- Feature count: {metrics['features']['feature_count']}

## Time Split

- Train: {metrics['split']['train']['rows']} rows, {metrics['split']['train']['time_start']} -> {metrics['split']['train']['time_end']}
- Valid: {metrics['split']['valid']['rows']} rows, {metrics['split']['valid']['time_start']} -> {metrics['split']['valid']['time_end']}
- Test: {metrics['split']['test']['rows']} rows, {metrics['split']['test']['time_start']} -> {metrics['split']['test']['time_end']}

## Validation Metrics

- ROC-AUC: {validation['probability_metrics']['roc_auc']}
- PR-AUC: {validation['probability_metrics']['pr_auc']}
- Selected threshold: {validation['selected_threshold']}

## Test Backtest

- Total BUY signals: {model_backtest['total_signals']}
- Precision on triggered signals: {model_backtest['precision_on_triggered_signals']}
- Average future max return: {model_backtest['average_future_max_return']}
- Average future min return: {model_backtest['average_future_min_return']}
- Estimated return after costs: {model_backtest['estimated_return_after_costs']}
- Max drawdown: {model_backtest['max_drawdown']}

## Feature Importance

{top_features}

## Risks and Limitations

- This report is factual research output, not financial advice.
- Test data was used only for final evaluation.
- Fees and slippage are assumptions and may differ from live execution.

## Paper Trading Readiness

- Ready for paper trading: {ready}
"""


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _project_root(config_path: Path) -> Path:
    if config_path.parent.name == "config":
        return config_path.parent.parent
    return config_path.parent


def _resolve_path(path_value: str, root: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else root / path


def _require_two_classes(series: pd.Series, split_name: str) -> None:
    if series.nunique() < 2:
        raise ValueError(f"{split_name} split must contain both buy_label classes")


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "lightgbm": lgb.__version__,
        "sklearn": sklearn.__version__,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _save_booster_model(booster: lgb.Booster, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.as_posix().encode("ascii")
        booster.save_model(str(path))
        return
    except UnicodeEncodeError:
        pass

    handle = tempfile.NamedTemporaryFile(
        mode="w", suffix=path.suffix, delete=False, encoding="utf-8"
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        booster.save_model(str(temp_path))
        shutil.move(str(temp_path), path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BTCUSDT prediction models.")
    parser.add_argument(
        "--config",
        default="config/training.yaml",
        help="Path to training YAML config.",
    )
    args = parser.parse_args()
    result = run_training(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
