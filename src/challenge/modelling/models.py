from typing import Any, Dict, Tuple

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Try importing XGBoost and CatBoost
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def get_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of models to be used in the experiment.
    """
    models = {
        "LightGBM": LGBMClassifier(
            random_state=random_state, verbose=-1, class_weight="balanced"
        ),
        "LogisticRegression": LogisticRegression(
            random_state=random_state, max_iter=1000, n_jobs=-1, class_weight="balanced"
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            random_state=random_state, class_weight="balanced"
        ),
        "MLP": MLPClassifier(random_state=random_state, max_iter=1000),
    }

    if XGBClassifier:
        # Scale pos weight estimate: 500/10 = 50
        models["XGBoost"] = XGBClassifier(
            random_state=random_state, eval_metric="logloss", scale_pos_weight=50
        )

    return models


def get_custom_lgbm(random_state: int = 42) -> LGBMClassifier:
    """
    Returns a standard LightGBM classifier (formerly custom).
    Kept for backward compatibility.
    """
    return LGBMClassifier(
        random_state=random_state, verbose=-1, class_weight="balanced"
    )
