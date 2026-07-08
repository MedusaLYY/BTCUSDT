from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: pd.Series,
    future_max_return: pd.Series,
    future_min_return: pd.Series,
    thresholds: Iterable[float],
) -> list[dict[str, object]]:
    """Evaluate classification and future-return summaries by probability threshold."""
    rows: list[dict[str, object]] = []
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    probs = pd.Series(probabilities).reset_index(drop=True)
    future_max = pd.Series(future_max_return).reset_index(drop=True)
    future_min = pd.Series(future_min_return).reset_index(drop=True)

    for threshold in thresholds:
        predicted = (probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
        signal_mask = predicted == 1
        rows.append(
            {
                "threshold": float(threshold),
                "signal_count": int(signal_mask.sum()),
                "precision": _safe_metric(precision_score, y, predicted),
                "recall": _safe_metric(recall_score, y, predicted),
                "f1": _safe_metric(f1_score, y, predicted),
                "true_positive": int(tp),
                "false_positive": int(fp),
                "true_negative": int(tn),
                "false_negative": int(fn),
                "avg_future_max_return": _masked_mean(future_max, signal_mask),
                "avg_future_min_return": _masked_mean(future_min, signal_mask),
            }
        )
    return rows


def select_high_precision_threshold(
    threshold_rows: list[dict[str, object]],
    min_signals: int = 50,
) -> float:
    """Select threshold on validation using precision first, then F1, then level."""
    eligible = [
        row for row in threshold_rows if int(row["signal_count"]) >= min_signals
    ]
    if not eligible:
        eligible = [row for row in threshold_rows if int(row["signal_count"]) > 0]
    if not eligible:
        return float(threshold_rows[-1]["threshold"])

    best = max(
        eligible,
        key=lambda row: (
            float(row["precision"]),
            float(row["f1"]),
            float(row["threshold"]),
        ),
    )
    return float(best["threshold"])


def classifier_probability_metrics(
    y_true: pd.Series,
    probabilities: pd.Series,
    buy_probability_threshold: float = 0.62,
) -> dict[str, object]:
    y = pd.Series(y_true).astype(int).reset_index(drop=True)
    probs = pd.Series(probabilities).astype(float).reset_index(drop=True)
    buy_mask = probs >= buy_probability_threshold

    if y.nunique() < 2:
        auc = None
        average_precision = None
    else:
        auc = float(roc_auc_score(y, probs))
        average_precision = float(average_precision_score(y, probs))

    return {
        "auc": auc,
        "roc_auc": auc,
        "average_precision": average_precision,
        "pr_auc": average_precision,
        "brier_score": float(brier_score_loss(y, probs)),
        "calibration_buckets": _calibration_buckets(y, probs),
        "top_5pct_hit_rate": _top_hit_rate(y, probs, pct=0.05),
        "top_10pct_hit_rate": _top_hit_rate(y, probs, pct=0.10),
        "buy_signal_count": int(buy_mask.sum()),
        "buy_signal_rate": float(buy_mask.mean()) if len(buy_mask) else 0.0,
    }


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, object]:
    actual = pd.Series(y_true).astype(float).reset_index(drop=True)
    predicted = pd.Series(y_pred).astype(float).reset_index(drop=True)
    error = predicted - actual
    absolute_error = error.abs()
    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(math.sqrt((error**2).mean())),
        "pearson_corr": _pearson_corr(actual, predicted),
        "mean_actual": float(actual.mean()),
        "mean_prediction": float(predicted.mean()),
        "error_quantiles": _quantiles(error),
        "absolute_error_quantiles": _quantiles(absolute_error),
        "prediction_quantile_error": _prediction_quantile_error(actual, predicted),
    }


def build_threshold_sweep_report(
    frame: pd.DataFrame,
    dataset_name: str,
    probability_thresholds: Iterable[float],
    return_thresholds: Iterable[float],
    round_trip_cost: float,
) -> pd.DataFrame:
    """Compare fixed probability and predicted-return threshold combinations."""
    required = {
        "buy_probability",
        "predicted_max_return",
        "y_buy",
        "future_close_return_30m_from_next_open",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns for threshold sweep: {missing}")

    rows: list[dict[str, object]] = []
    total_rows = len(frame)
    for probability_threshold in probability_thresholds:
        for return_threshold in return_thresholds:
            buy_mask = (frame["buy_probability"] >= probability_threshold) & (
                frame["predicted_max_return"] >= return_threshold
            )
            selected = frame.loc[buy_mask]
            net_returns = (
                selected["future_close_return_30m_from_next_open"].astype(float)
                - round_trip_cost
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "buy_probability_threshold": float(probability_threshold),
                    "predicted_max_return_threshold": float(return_threshold),
                    "buy_count": int(buy_mask.sum()),
                    "buy_rate": float(buy_mask.mean()) if total_rows else 0.0,
                    "hit_rate": _masked_mean(frame["y_buy"], buy_mask),
                    "average_return_after_costs": _series_mean(net_returns),
                    "total_return_after_costs": float(net_returns.sum()),
                    "max_drawdown": float(_max_drawdown(net_returns)),
                }
            )
    return pd.DataFrame(rows)


def _safe_metric(metric_fn, y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        return float(metric_fn(y_true, y_pred, zero_division=0))
    except TypeError:
        return float(metric_fn(y_true, y_pred))


def _masked_mean(values: pd.Series, mask: pd.Series) -> float | None:
    selected = pd.Series(values).reset_index(drop=True)[pd.Series(mask).reset_index(drop=True)]
    if selected.empty:
        return None
    value = selected.astype(float).mean()
    if np.isnan(value):
        return None
    return float(value)


def _series_mean(values: pd.Series) -> float | None:
    if values.empty:
        return None
    value = values.mean()
    if np.isnan(value):
        return None
    return float(value)


def _top_hit_rate(y: pd.Series, probabilities: pd.Series, pct: float) -> float | None:
    if y.empty:
        return None
    count = max(1, int(math.ceil(len(y) * pct)))
    selected_index = probabilities.sort_values(ascending=False).head(count).index
    return float(y.loc[selected_index].mean())


def _calibration_buckets(
    y: pd.Series,
    probabilities: pd.Series,
    bucket_count: int = 10,
) -> list[dict[str, object]]:
    bins = np.linspace(0.0, 1.0, bucket_count + 1)
    buckets = pd.cut(probabilities, bins=bins, include_lowest=True, right=True)
    report: list[dict[str, object]] = []
    for bucket, bucket_probs in probabilities.groupby(buckets, observed=False):
        if bucket_probs.empty:
            continue
        bucket_y = y.loc[bucket_probs.index]
        report.append(
            {
                "bucket": f"{float(bucket.left):.1f}-{float(bucket.right):.1f}",
                "count": int(len(bucket_probs)),
                "mean_predicted_probability": float(bucket_probs.mean()),
                "observed_positive_rate": float(bucket_y.mean()),
            }
        )
    return report


def _pearson_corr(actual: pd.Series, predicted: pd.Series) -> float | None:
    if len(actual) < 2 or actual.nunique(dropna=True) < 2 or predicted.nunique(dropna=True) < 2:
        return None
    value = actual.corr(predicted, method="pearson")
    if pd.isna(value):
        return None
    return float(value)


def _quantiles(values: pd.Series) -> dict[str, float]:
    quantiles = values.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        f"p{int(level * 100):02d}": float(value)
        for level, value in quantiles.items()
    }


def _prediction_quantile_error(
    actual: pd.Series,
    predicted: pd.Series,
    bucket_count: int = 5,
) -> list[dict[str, object]]:
    if predicted.nunique(dropna=True) < 2:
        return []
    buckets = pd.qcut(predicted, q=bucket_count, duplicates="drop")
    rows: list[dict[str, object]] = []
    for bucket, bucket_predicted in predicted.groupby(buckets, observed=False):
        if bucket_predicted.empty:
            continue
        bucket_actual = actual.loc[bucket_predicted.index]
        error = bucket_predicted - bucket_actual
        rows.append(
            {
                "bucket": str(bucket),
                "count": int(len(bucket_predicted)),
                "mean_prediction": float(bucket_predicted.mean()),
                "mean_actual": float(bucket_actual.mean()),
                "mae": float(error.abs().mean()),
            }
        )
    return rows


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    equity = (1 + returns).cumprod()
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return float(drawdown.min())
