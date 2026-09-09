"""Regression tests for leakage risks found during strict feature-pipeline review.

These tests are intentionally red against the current implementation.  Each test
describes the safety property the pipeline needs to enforce; a failure is evidence
that the corresponding boundary can currently be crossed.
"""

from __future__ import annotations

import ast
from pathlib import Path

import polars as pl
import pytest

from src.config import REPO_ROOT, Paths
from src.data import CohortSpec, build_cohort
from src.data.cohort import cohort_fingerprint
from src.features import build_features, build_log_features
from src.features.build import MISSING_CATEGORY
from src.features.logs import log_features_fingerprint
from src.fingerprint import write_with_fingerprint
from src.models.candidates import xgb_category_levels
from tests.conftest import NODATA, make_synthetic_cohort


@NODATA
def test_feature_builder_rejects_post_cutoff_log_features():
    """The public join boundary must not accept logs produced after cutoff."""
    cohort = make_synthetic_cohort()
    leaky_logs = pl.DataFrame(
        {
            "msno": ["u0"],
            "log_min_days_before": [-1],
            "log_has_logs": [1.0],
        }
    )

    with pytest.raises(AssertionError, match="紅線 2 違反"):
        build_features(cohort, leaky_logs)


@NODATA
def test_cohort_cache_hit_is_revalidated_before_use(tmp_path: Path):
    """A poisoned/stale cache must not bypass the transaction as-of guard."""
    paths = Paths(tmp_path)
    paths.interim.mkdir(parents=True)
    poisoned = make_synthetic_cohort().with_columns(
        pl.when(pl.col("msno") == "u0")
        .then(pl.lit(20170301))
        .otherwise(pl.col("last_tx"))
        .alias("last_tx")
    )
    # ⚠️ 假快取要帶**當前的程式版本指紋**，否則 `build_cohort()` 會先因為
    # 指紋不符而重算，根本走不到這條測試要驗的守門（見 src/fingerprint.py）。
    write_with_fingerprint(
        poisoned, paths.interim / "feb_cohort_asof.parquet", cohort_fingerprint()
    )

    with pytest.raises(AssertionError, match="紅線 1 違反"):
        build_cohort("feb", paths, verbose=False)


@NODATA
def test_log_feature_cache_hit_is_revalidated_before_use(tmp_path: Path):
    """A cached log table containing future activity must never be trusted blindly."""
    paths = Paths(tmp_path)
    paths.interim.mkdir(parents=True)
    # ⚠️ 假快取要帶當前的程式版本指紋，否則會先因指紋不符而重算，
    # 走不到這條測試要驗的紅線 2 守門（見 src/fingerprint.py）。
    write_with_fingerprint(
        pl.DataFrame(
            {
                "msno": ["u0"],
                "log_min_days_before": [-3],
                "log_has_logs": [1.0],
            }
        ),
        paths.interim / "feb_log_features.parquet",
        log_features_fingerprint(),
    )

    with pytest.raises(AssertionError, match="紅線 2 違反"):
        build_log_features("feb", paths, verbose=False)


@NODATA
def test_member_snapshot_is_masked_when_registration_is_after_cutoff():
    """Attributes that did not exist at prediction time must not enter X."""
    features = build_features(make_synthetic_cohort()).X

    # make_synthetic_cohort row u5 registers on 2017-03-30 but has a 2017-02-28 cutoff.
    row = features.row(5, named=True)
    assert row["days_since_registration"] is None
    assert row["city"] == MISSING_CATEGORY
    assert row["registered_via"] == MISSING_CATEGORY
    assert row["gender_code"] == MISSING_CATEGORY
    assert row["in_members"] == 0.0


@NODATA
def test_xgboost_category_dictionary_is_fit_on_training_data_only():
    """Holdout-only categories must not influence fitted preprocessing state."""
    train = pl.DataFrame({"city": [1, 2, MISSING_CATEGORY]})
    holdout = pl.DataFrame({"city": [99]})

    levels = xgb_category_levels(train, categorical=("city",))

    assert levels["city"] == [1, 2], (
        "XGBoost 類別字典納入了只存在 Mar holdout 的 city=99；"
        "這是以驗證集分布擬合前處理器的 transductive leakage"
    )

    # 這個性質現在由簽名保證：函式只收一個訓練 frame，想把 holdout 一起
    # 餵進去會 TypeError，而不是靜默忽略多餘的參數。
    with pytest.raises(TypeError):
        xgb_category_levels(train, holdout, categorical=("city",))


def _attribute_names(node: ast.AST) -> set[str]:
    return {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}


@NODATA
def test_feature_subset_winner_is_not_selected_by_mar_holdout_score():
    """Mar may evaluate a frozen choice, but must not choose that choice itself."""
    path = REPO_ROOT / "scripts" / "select_features.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    offenders: list[int] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "best" for t in node.targets):
            continue
        if "logloss" in _attribute_names(node.value):
            offenders.append(node.lineno)

    assert not offenders, (
        "selected_features 的 winner 直接依 Mar logloss 決定；Mar 同時被當成"
        f"選模集與最終成績單（scripts/select_features.py:{offenders}）"
    )


@NODATA
def test_cutoff_is_not_inferred_from_a_future_transaction(tmp_path: Path):
    """A row observed after expiry must not manufacture the as-of timestamp."""
    paths = Paths(tmp_path)
    paths.raw.mkdir(parents=True)

    tx_columns = {
        "msno": ["u0"],
        "payment_method_id": [1],
        "payment_plan_days": [30],
        "plan_list_price": [100],
        "actual_amount_paid": [100],
        "is_auto_renew": [1],
        "transaction_date": [20170101],
        "membership_expire_date": [20170131],
        "is_cancel": [0],
    }
    pl.DataFrame(tx_columns).write_csv(paths.raw / "transactions.csv")

    # This row was not observable at a February prediction time.  Nevertheless,
    # its February membership_expire_date currently creates u0's cutoff.
    future_tx = {**tx_columns}
    future_tx["transaction_date"] = [20170305]
    future_tx["membership_expire_date"] = [20170228]
    pl.DataFrame(future_tx).write_csv(paths.raw / "transactions_v2.csv")

    pl.DataFrame({"msno": ["u0"], "is_churn": [0]}).write_csv(paths.raw / "labels.csv")
    pl.DataFrame(
        {
            "msno": ["u0"],
            "city": [1],
            "bd": [29],
            "gender": ["male"],
            "registered_via": [7],
            "registration_init_time": [20160101],
        }
    ).write_csv(paths.raw / "members_v3.csv")

    spec = CohortSpec("future_cutoff", "labels.csv", 20170201, 20170228, "test")
    cohort = build_cohort(spec, paths, force=True, verbose=False)

    assert cohort.is_empty(), (
        "u0 只因 2017-03-05 才出現的交易而被賦予 2017-02-28 cutoff；"
        "現有 last_tx <= cutoff 守門無法偵測 cutoff 本身來自未來"
    )
