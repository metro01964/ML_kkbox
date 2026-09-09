"""特徵工程。

對應 SPEC §7 的 `src/features/`「特徵工程（純函式，可測試）」。
"""

from src.features.build import (
    CATEGORICAL,
    FeatureSet,
    assert_logs_match_cohort,
    build_features,
    feature_build_fingerprint,
)
from src.features.encoding import (
    IDENTIFIER_COLUMNS,
    TargetEncoder,
    assert_encoding_aligned,
    assert_encoding_is_oof,
    assert_not_identifier,
    fit_target_encoder,
    oof_target_encode,
)
from src.features.logs import (
    LOG_GROUPS,
    LOG_WINDOWS,
    assert_logs_within_cutoff,
    build_log_features,
    expected_log_columns,
    log_feature_group,
    narrow_logs,
    window_bounds,
)

__all__ = [
    "CATEGORICAL",
    "IDENTIFIER_COLUMNS",
    "LOG_GROUPS",
    "LOG_WINDOWS",
    "FeatureSet",
    "TargetEncoder",
    "assert_encoding_aligned",
    "assert_encoding_is_oof",
    "assert_logs_match_cohort",
    "assert_logs_within_cutoff",
    "assert_not_identifier",
    "build_features",
    "build_log_features",
    "expected_log_columns",
    "feature_build_fingerprint",
    "fit_target_encoder",
    "log_feature_group",
    "narrow_logs",
    "oof_target_encode",
    "window_bounds",
]
