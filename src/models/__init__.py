"""模型訓練與推論。

對應 SPEC §7 的 `src/models/`。
"""

from src.models.candidates import Fitted, fit_catboost, fit_lightgbm, fit_xgboost
from src.models.compare import CandidateResult, run_comparison
from src.models.selection import NullImportance, null_importance
from src.models.train import (
    TrainResult,
    load_cohort_features,
    load_model_config,
    train_baseline,
)

__all__ = [
    "CandidateResult",
    "Fitted",
    "NullImportance",
    "TrainResult",
    "fit_catboost",
    "fit_lightgbm",
    "fit_xgboost",
    "load_cohort_features",
    "load_model_config",
    "null_importance",
    "run_comparison",
    "train_baseline",
]
