"""as-of 聚合的合成資料測試 —— 特別是「同日多筆交易」這個歧義。

## 為什麼一定要用合成資料

整段邏輯的正確性幾乎全在「最後交易日有多筆時取什麼值」，而那在真實資料上
只佔 1.30%（Feb 12,889 人）。跑真實資料驗不出它：分數會照常產生、守門會
照常通過，只有重建兩次比對才會看到 24 列悄悄變了。

合成資料可以把每一種情形各造一列，而且答案是手算的。

## 規則（C′，逐欄位判斷）

    最後交易日只有一筆                → 正常取值
    同日多筆、該欄位非 null 值只有一種 → 保留該值
    同日多筆、該欄位有兩種以上值      → 該欄位設 null

**逐欄位**是關鍵：同一天買兩筆同方案同金額、但其中一筆是取消，那
`last_plan_list_price` 一點都不模糊，`last_is_cancel` 才是真的沒有答案。
整列丟棄會扔掉一堆確定的資訊。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import LAST_TX_COLUMNS, aggregate_asof
from tests.conftest import NODATA

CUTOFF = 20170228


def _tx(rows: list[dict]) -> pl.LazyFrame:
    """造一批交易明細。未指定的欄位補上不影響判斷的預設值。"""
    default = {
        "cutoff": CUTOFF,
        "is_churn": 0,
        "is_cancel": 0,
        "is_auto_renew": 1,
        "actual_amount_paid": 149,
        "plan_list_price": 149,
        "payment_plan_days": 30,
        "payment_method_id": 41,
    }
    return pl.DataFrame([{**default, **r} for r in rows]).lazy()


def _one(rows: list[dict]) -> dict:
    """聚合成單一用戶的那一列。"""
    out = aggregate_asof(_tx(rows)).collect()
    assert out.height == 1, f"預期一位用戶，得到 {out.height}"
    return out.row(0, named=True)


@NODATA
def test_single_transaction_on_the_last_day_is_taken_as_is():
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170210, "is_cancel": 1, "actual_amount_paid": 99},
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 0, "actual_amount_paid": 149},
        ]
    )
    assert row["last_day_n_tx"] == 1
    assert row["last_day_has_conflict"] is False
    assert row["last_is_cancel"] == 0
    assert row["last_actual_amount_paid"] == 149


@NODATA
def test_same_day_duplicates_with_identical_values_are_kept():
    """同日多筆但值一樣 → 沒有歧義，全部保留。

    這是規則的第一半。若實作偷懶地「只要同日多筆就整列給 null」，這條會紅。
    """
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170220},
            {"msno": "u", "transaction_date": 20170220},
            {"msno": "u", "transaction_date": 20170220},
        ]
    )
    assert row["last_day_n_tx"] == 3
    assert row["last_day_has_conflict"] is False
    for col in LAST_TX_COLUMNS:
        assert row[f"last_{col}"] is not None, f"last_{col} 不該因為同日多筆就變 null"


@NODATA
def test_conflicting_is_cancel_nulls_only_that_column():
    """同日多筆且 `is_cancel` 衝突 → **只有** last_is_cancel 為 null。

    其餘五個欄位在這一天的值都一致，必須原封不動保留 —— 這就是 C′ 與
    「整列丟棄」的差別，也是本專案最強特徵（35.2% gain）的處理方式。
    """
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 0},
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 1},
        ]
    )
    assert row["last_day_n_tx"] == 2
    assert row["last_day_has_conflict"] is True
    assert row["last_is_cancel"] is None

    for col in LAST_TX_COLUMNS:
        if col == "is_cancel":
            continue
        assert row[f"last_{col}"] is not None, f"last_{col} 沒有衝突，不該被清掉"


@NODATA
def test_each_conflicting_column_is_nulled_independently():
    """兩個欄位各自衝突、其他欄位不受影響 —— 逐欄位判斷的完整版。"""
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 0, "payment_method_id": 41},
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 1, "payment_method_id": 38},
        ]
    )
    assert row["last_is_cancel"] is None
    assert row["last_payment_method_id"] is None
    assert row["last_actual_amount_paid"] == 149
    assert row["last_plan_list_price"] == 149
    assert row["last_payment_plan_days"] == 30
    assert row["last_is_auto_renew"] == 1


@NODATA
def test_transactions_after_cutoff_never_participate():
    """cutoff 之後的交易不能影響任何判斷 —— 紅線 1。

    這裡刻意讓「未來那筆」看起來像最後交易日、而且帶著會造成衝突的值。
    若截斷失效，last_tx 會變成 20170301、last_is_cancel 會變 null，兩者
    都是這條測試抓得到的。
    """
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 0},
            {"msno": "u", "transaction_date": 20170301, "is_cancel": 1, "actual_amount_paid": 0},
        ]
    )
    assert row["last_tx"] == 20170220
    assert row["n_tx"] == 1
    assert row["last_day_n_tx"] == 1
    assert row["last_day_has_conflict"] is False
    assert row["last_is_cancel"] == 0
    assert row["last_actual_amount_paid"] == 149


@NODATA
def test_cutoff_day_transactions_do_participate():
    """cutoff **當天**的交易要算進去 —— 截斷是 <=，不是 <。"""
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170210},
            {"msno": "u", "transaction_date": CUTOFF, "actual_amount_paid": 99},
        ]
    )
    assert row["last_tx"] == CUTOFF
    assert row["last_actual_amount_paid"] == 99


@NODATA
def test_order_independent_aggregates_are_unaffected():
    """n_tx / n_cancel_hist / mean_paid 不受同日歧義影響。

    它們對順序免疫，所以無論衝突與否都必須照算 —— 把它們一起清成 null
    會是過度反應，也會讓「歷史上取消過幾次」這種真實訊號憑空消失。
    """
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170210, "is_cancel": 1, "actual_amount_paid": 100},
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 0, "actual_amount_paid": 200},
            {"msno": "u", "transaction_date": 20170220, "is_cancel": 1, "actual_amount_paid": 300},
        ]
    )
    assert row["n_tx"] == 3
    assert row["n_cancel_hist"] == 2
    assert row["mean_paid"] == pytest.approx(200.0)
    assert row["first_tx"] == 20170210
    assert row["last_tx"] == 20170220
    # 金額也衝突（200 vs 300），所以該欄位為 null —— 但 mean_paid 照算。
    assert row["last_actual_amount_paid"] is None


@NODATA
def test_null_values_do_not_count_as_a_conflicting_value():
    """null 不算一種「值」：一筆有值、一筆缺值時保留那個值。

    缺值不是矛盾，是沒有資訊。把它當成第二種取值會讓大量本來確定的欄位
    無謂地變成 null。
    """
    row = _one(
        [
            {"msno": "u", "transaction_date": 20170220, "payment_method_id": 41},
            {"msno": "u", "transaction_date": 20170220, "payment_method_id": None},
        ]
    )
    assert row["last_payment_method_id"] == 41
    assert row["last_day_has_conflict"] is False


@NODATA
def test_multiple_users_are_independent():
    """一位用戶的衝突不能污染另一位 —— 遮罩必須是逐人的。"""
    out = (
        aggregate_asof(
            _tx(
                [
                    {"msno": "clean", "transaction_date": 20170220, "is_cancel": 1},
                    {"msno": "dirty", "transaction_date": 20170220, "is_cancel": 0},
                    {"msno": "dirty", "transaction_date": 20170220, "is_cancel": 1},
                ]
            )
        )
        .collect()
        .sort("msno")
    )

    rows = {r["msno"]: r for r in out.iter_rows(named=True)}
    assert rows["clean"]["last_is_cancel"] == 1
    assert rows["clean"]["last_day_has_conflict"] is False
    assert rows["dirty"]["last_is_cancel"] is None
    assert rows["dirty"]["last_day_has_conflict"] is True


@NODATA
def test_audit_columns_are_present_but_not_features():
    """稽核欄位要在 cohort 表裡，但不能出現在特徵矩陣。

    「最後一天有沒有衝突」很可能與流失相關（同日多筆常見於改方案、取消後
    重買），一旦當特徵就是拿資料品質瑕疵預測標籤 —— 會有效，但學到的是
    我們的管線而不是用戶行為。
    """
    from src.features import build_features

    cohort = aggregate_asof(_tx([{"msno": "u", "transaction_date": 20170220}])).collect()
    cohort = cohort.with_columns(
        pl.lit(None, dtype=pl.Int64).alias("city"),
        pl.lit(None, dtype=pl.Int64).alias("bd"),
        pl.lit(None, dtype=pl.Utf8).alias("gender"),
        pl.lit(None, dtype=pl.Int64).alias("registered_via"),
        pl.lit(20160101, dtype=pl.Int64).alias("registration_init_time"),
        pl.lit(False).alias("in_members"),
    )

    assert "last_day_n_tx" in cohort.columns
    assert "last_day_has_conflict" in cohort.columns

    fs = build_features(cohort)
    assert "last_day_n_tx" not in fs.X.columns
    assert "last_day_has_conflict" not in fs.X.columns
