from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd
from lightgbm import LGBMRegressor


def train_return_regressor(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
    random_seed: int,
) -> LGBMRegressor:
    """Train the 30-minute future maximum return regressor on chronological data."""
    model_params = dict(params)
    model_params.setdefault("random_state", random_seed)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("verbosity", -1)
    model = LGBMRegressor(**model_params)
    model.fit(
        train[feature_columns],
        train["future_max_return_30m"],
        eval_set=[(valid[feature_columns], valid["future_max_return_30m"])],
        eval_metric="l2",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model
