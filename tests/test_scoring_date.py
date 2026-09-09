"""M6 · 固定評分日（Kaggle 管線的第三種 cutoff 規則）的測試。

全部手刻資料，標 nodata、在 CI 上實際執行。

## 這一組守的是「資料不夠，但每一條檢查都會通過」

Kaggle 測試集要預測 2017-04 到期的人，而交易與日誌**都只到 2017-03-31**。若沿用
`cutoff = 到期日 − 7 天`，77.64% 的測試用戶（到期日在 4/8 之後）的 cutoff 會落在
資料結束之後 —— 那時：

    紅線 1（last_tx <= cutoff）        必然通過（所有交易都比 cutoff 早）
    紅線 2（log_min_days_before >= 0） 必然通過（同上）
    cohort 錯置檢查                    通過（cutoff 確實落在宣告的區間內）
    欄位檢查                           通過（一欄都沒少）

而特徵已經被**資料集的結尾**截斷：`days_since_last_tx` 被放大、落在資料之外的
收聽窗口被模型讀成「這個人沒在聽歌」。擋它的是 `assert_data_covers_cutoffs()`。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import (
    APR_FIXED,
    FEB,
    FEB_FIXED,
    MAR_FIXED,
    MAR_T7,
    CohortSpec,
    assert_data_covers_cutoffs,
    assert_labels_are_real,
    build_cohort,
    cutoff_definition,
    cutoff_window,
    expected_columns,
)
from src.explain.reasons import meta
from src.features import build_features
from src.features.logs import MAX_WINDOW, window_bounds
from tests.conftest import NODATA, SLOW, make_synthetic_cohort

# 競賽資料實際涵蓋到的最後一天。**測試裡才寫死** —— 產線程式從資料算
# （`observed_transaction_end`），理由見該函式。
DATA_END = 20170331


@NODATA
def test_the_two_cutoff_rules_are_mutually_exclusive():
    """兩種規則同時給，「這個 cohort 的 cutoff 是什麼」就沒有答案。"""
    with pytest.raises(ValueError, match="互斥"):
        CohortSpec(
            "bad", "train.csv", 20170401, 20170430, "2017-05", lead_days=7, score_date=DATA_END
        )


@NODATA
def test_fixed_score_date_collapses_the_cutoff_window_to_a_point():
    """所有人同一天評分 → cutoff 區間退化成一個點，守門變成精確的相等檢查。"""
    assert cutoff_window(APR_FIXED) == (20170331, 20170331)
    assert cutoff_definition(APR_FIXED) == "fixed_score_date_20170331"
    assert cutoff_definition(FEB_FIXED) == "fixed_score_date_20170131"
    # 提前固定天數那兩種不受影響。
    assert cutoff_definition(FEB) == "expire_date"
    assert cutoff_definition(MAR_T7) == "expire_date_minus_7d"


@NODATA
def test_log_window_of_a_fixed_spec_does_not_use_the_expiry_month():
    """⚠️ 用到期區間去推收斂檔的範圍，上界會拉到到期月底 —— 那超出資料涵蓋。

    `apr_fixed` 的到期區間是 4/1~4/30，但它的收聽窗口只到評分日 20170331。
    差的那 30 天在資料裡一列都沒有，而收斂檔的檔名會宣稱它涵蓋到 4/30。
    """
    lo, hi = window_bounds((APR_FIXED,))
    assert hi == 20170331, "上界必須是評分日，不是到期月底"
    # 20170331 往前 90 天 = 20161231（3 月 30 天 + 2 月 28 + 1 月 31 = 89 → 1/1，再一天）。
    assert lo == 20161231, f"下界應該是評分日往前 {MAX_WINDOW} 天"
    assert hi <= DATA_END


@NODATA
def test_the_apr_cohort_at_minus_seven_days_is_refused():
    """⚠️ 這一條是本節的核心：**使用者指出的那個問題的迴歸測試**。

    4 月到期的人若用「到期日 − 7 天」，只有 4/1~4/7 到期的那 22% 的 cutoff 落在
    資料內。其餘的 cutoff 在 4/1 之後，而資料到 3/31 為止。
    """
    # 到期日 4/2、4/10、4/30 → cutoff 3/26、4/3、4/23
    cutoffs = pl.Series("cutoff", [20170326, 20170403, 20170423])
    with pytest.raises(AssertionError, match="超出資料涵蓋範圍"):
        assert_data_covers_cutoffs(
            cutoffs, data_end=DATA_END, spec_name="apr_t7", window_days=MAX_WINDOW
        )

    # 錯誤訊息要說得出「這不是洩漏，是資料不足」，否則讀者會去找洩漏。
    with pytest.raises(AssertionError, match="資料不足"):
        assert_data_covers_cutoffs(cutoffs, data_end=DATA_END, spec_name="apr_t7")


@NODATA
def test_a_cutoff_exactly_on_the_last_day_of_data_is_allowed():
    """固定評分日剛好等於資料最後一天 —— 那正是 `apr_fixed` 的設計，必須放行。"""
    cutoffs = pl.Series("cutoff", [DATA_END] * 5)
    assert_data_covers_cutoffs(cutoffs, data_end=DATA_END, spec_name="apr_fixed")


@NODATA
def test_the_guard_reports_how_many_rows_are_beyond():
    """訊息要帶「幾個人」與「最晚是哪一天」，否則不知道是手滑還是設計錯。"""
    cutoffs = pl.Series("cutoff", [20170301, 20170401, 20170402])
    with pytest.raises(AssertionError, match="2 位用戶"):
        assert_data_covers_cutoffs(cutoffs, data_end=DATA_END, spec_name="x")


@NODATA
def test_placeholder_labels_are_refused_for_training_and_evaluation():
    """Kaggle 測試集的 is_churn 全是 0，拿它算分數會得到看起來合理的垃圾。"""
    assert APR_FIXED.labels_are_real is False
    with pytest.raises(AssertionError, match="佔位值"):
        assert_labels_are_real(FEB_FIXED, APR_FIXED)
    # 真標籤的組合不該被擋。
    assert_labels_are_real(FEB_FIXED, MAR_FIXED)


@NODATA
def test_only_fixed_specs_declare_the_expire_date_column():
    """`expire_date` 是 `days_to_expire` 的來源，只有固定評分日的 spec 需要它。"""
    assert "expire_date" in expected_columns(APR_FIXED)
    assert "expire_date" not in expected_columns(MAR_T7)
    # 其餘欄位兩者相同。
    assert expected_columns(APR_FIXED) - {"expire_date"} == expected_columns(MAR_T7)


@NODATA
def test_days_to_expire_appears_only_when_the_cohort_has_a_fixed_score_date():
    """特徵集因評分規則而不同，那不是 bug（見 build_features 的說明）。

    提前固定天數的設計裡「還有幾天到期」是常數，加它只是多一欄零 gain。
    """
    cohort = make_synthetic_cohort()
    assert "days_to_expire" not in build_features(cohort).X.columns

    # 固定評分日：cutoff 全部同一天，到期日各自不同。
    fixed = cohort.with_columns(
        pl.lit(20170131).alias("cutoff"),
        pl.Series("expire_date", [20170201, 20170210, 20170228, 20170301] * 2),
    )
    X = build_features(fixed).X
    assert "days_to_expire" in X.columns
    assert X["days_to_expire"].to_list()[:4] == [1.0, 10.0, 28.0, 29.0]


@NODATA
def test_days_to_expire_keeps_negative_values():
    """到期日已經過了而還沒續訂 —— 那是強訊號，不是髒資料，不可轉成 null。

    （`_days_before_cutoff` 把負值轉 null 是另一回事：那些欄位的負值代表未來
    資訊，而這一欄的負值代表「已經過期」。）
    """
    cohort = make_synthetic_cohort().with_columns(
        pl.lit(20170228).alias("cutoff"),
        pl.lit(20170210).alias("expire_date"),
    )
    assert build_features(cohort).X["days_to_expire"].to_list()[0] == -18.0


@SLOW
def test_fixed_cohorts_keep_only_members_whose_expiry_is_visible_in_the_window(paths):
    """as-of 成員篩選：**看得出到期日不在目標月份的人要被排除**。

    這條釘住的是本地對照與 Kaggle 測試集的結構一致性。實測測試集（`apr_fixed`）
    100% 的成員在評分日當下就看得出到期日落在 4 月，而只用標籤檔的 cohort 只有
    89.5% —— 多出來的那 10.5% 是測試集不會有的人（到期日看起來在 90 天後、或已經
    過期一年），而他們最難預測，留著會讓本地對照系統性偏悲觀。

    ⚠️ 到期日**不唯一**（同日多筆衝突）的人要留下：那是真實的服務情境，Kaggle
    也要求給他們機率。
    """
    for spec in (MAR_FIXED, APR_FIXED):
        df = build_cohort(spec, paths, verbose=False)
        visible = df["expire_date"].drop_nulls()
        assert int(visible.min()) >= spec.expire_start, spec.name
        assert int(visible.max()) <= spec.expire_end, spec.name
        # 到期日不唯一的人留在 cohort 裡（不是被篩掉）。
        assert df["expire_date"].null_count() > 0, spec.name


@NODATA
def test_days_to_expire_has_a_reason_code_template():
    """新特徵沒有句型的話，原因碼會印出欄名 —— `meta()` 刻意不給預設值。"""
    m = meta("days_to_expire")
    assert m.noun and m.unit == "天"
    # 量測時點是「位移」：評分當下就看得到，不是到期日才發生的事。
    assert m.horizon == "位移"
