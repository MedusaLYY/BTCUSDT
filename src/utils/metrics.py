from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
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
    """Evaluate classification and future-return summaries by threshold."""
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
    y_true: pd.Series, probabilities: pd.Series
) -> dict[str, float | None]:
    y = pd.Series(y_true).astype(int)
    probs = pd.Series(probabilities)
    if y.nunique() < 2:
        return {"roc_auc": None, "pr_auc": None}
    return {
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
    }


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    actual = pd.Series(y_true).astype(float)
    predicted = pd.Series(y_pred).astype(float)
    error = predicted - actual
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(math.sqrt((error**2).mean())),
        "mean_actual": float(actual.mean()),
        "mean_prediction": float(predicted.mean()),
    }


def _safe_metric(metric_fn, y_true: pd.Series, y_pred: pd.Series) -> float:
    try:
        return float(metric_fn(y_true, y_pred, zero_division=0))
    except TypeError:
        return float(metric_fn(y_true, y_pred))


def _masked_mean(values: pd.Series, mask: pd.Series) -> float | None:
    selected = values[mask]
    if selected.empty:
        return None
    value = selected.mean()
    if np.isnan(value):
        return None
    return float(value)
