"""M6 · 提前評分（`cutoff = 到期日 − lead_days`）的切分測試。

SPEC §4.3 的 M6 追加：實務上挽回優惠要提前寄出才來得及，所以真正能上線的
模型是提前 7 天評分的那一個。這一份守的是**那個位移本身有沒有做對**。

全部用手刻的合成交易，因此標 nodata、在 CI 上會實際執行。真實資料上的分數
對照走 `scripts/lead_time.py`。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.data import (
    FEB,
    FEB_T7,
    MAR,
    MAR_T7,
    CohortSpec,
    aggregate_asof,
    assert_asof_respected,
    assert_cutoffs_within_window,
    cutoff_window,
)
from src.data.cohort import _shift_days_expr, _shift_yyyymmdd
from tests.conftest import NODATA


@NODATA
def test_yyyymmdd_is_shifted_as_a_date_not_as_an_integer():
    """`20170301 − 7` 用整數算會得到 20170294，那不是日期。

    這個坑在本專案已經出現兩次（`build.py` 的 `20170301 - 20170228 = 73`）。
    它不會報錯，只會產生垃圾特徵，所以每次都要釘住。
    """
    assert _shift_yyyymmdd(20170301, -7) == 20170222
    # 跨月
    assert _shift_yyyymmdd(20170201, -7) == 20170125
    # 跨年
    assert _shift_yyyymmdd(20170103, -7) == 20161227
    # 閏年：2016-02-29 存在，所以 3/1 往前 1 天是 2/29 而不是 2/28
    assert _shift_yyyymmdd(20160301, -1) == 20160229


@NODATA
@pytest.mark.parametrize(
    ("cutoff", "expected"),
    [
        (20170201, 20170125),  # 1 月：`month * 100 = 100` 還在 Int8 界內，**不會出事**
        (20170228, 20170221),  # 2 月：第一版在這裡壞掉（得到 20169965，差 256）
        (20170301, 20170222),  # 3 月：同上
        (20161231, 20161224),  # 12 月：溢位最嚴重的月份
        (20160301, 20160223),  # 閏年
    ],
)
def test_the_column_version_of_the_shift_survives_int8(cutoff, expected):
    """整欄版的位移必須與 Python 版一致。

    ⚠️ **這條測試存在的原因是它抓到過一個真的 bug。** 第一版寫成
    `year * 10000 + month * 100 + day`，而 polars 的 `dt.month()` 回傳
    **Int8** —— `2 * 100 = 200` 溢位成 −56，於是 20170228 往前 7 天得到
    20169965（差 256）。

    **1 月不會出事**（`1 * 100 = 100` 在界內），所以只用 20170201 當測資會
    通過。真實資料是靠 `assert_cutoffs_within_window()` 擋下來的 —— 而紅線 1
    的 `assert_asof_respected()` **通過了**，因為 cutoff 這個基準點自己是垃圾，
    所有以它為準的檢查都會成立（§7.11 註解裡預言過這個形狀）。
    """
    out = pl.DataFrame({"cutoff": [cutoff]}).with_columns(_shift_days_expr("cutoff", -7))
    assert out["cutoff"][0] == expected
    assert out["cutoff"][0] == _shift_yyyymmdd(cutoff, -7)


@NODATA
def test_the_t7_specs_shift_the_cutoff_window_but_not_the_labels():
    """提前評分動的是 cutoff，不是標籤 —— 標籤檔與到期區間都必須一樣。

    若 T−7 版本換了標籤或換了成員，「分數下降多少」就不是在量提前評分的代價，
    而是在量兩個不同的問題。
    """
    for base, shifted in ((FEB, FEB_T7), (MAR, MAR_T7)):
        assert shifted.label_file == base.label_file
        assert (shifted.expire_start, shifted.expire_end) == (base.expire_start, base.expire_end)
        assert base.lead_days == 0
        assert shifted.lead_days == 7

    assert cutoff_window(FEB) == (20170201, 20170228)
    assert cutoff_window(FEB_T7) == (20170125, 20170221)
    assert cutoff_window(MAR_T7) == (20170222, 20170324)


@NODATA
def test_the_window_guard_accepts_the_shifted_window_and_rejects_the_original():
    """`mar_t7` 的 cutoff 落在 0222~0324，用 `mar` 的區間檢查必須失敗。

    這是「拿錯 cohort」那條守門的延伸：兩個版本的 cutoff 區間不同，所以一張
    T−7 的表放進 T=0 的位置會被擋下來。
    """
    t7 = pl.DataFrame({"cutoff": [20170222, 20170301, 20170324]})

    assert_cutoffs_within_window(t7, MAR_T7)  # 不 raise
    with pytest.raises(AssertionError, match="cohort 錯置"):
        assert_cutoffs_within_window(t7, MAR)


@NODATA
def test_the_overlap_between_mar_t7_and_feb_is_documented_not_pretended_away():
    """⚠️ `mar_t7`（0222~0324）與 `feb`（0201~0228）**有 7 天重疊**。

    原本的守門靠「兩個到期區間不重疊」而具有決定性。加了 lead_days 之後，
    那句話只對 lead_days 相同的兩個 cohort 成立 —— 這條測試把重疊本身釘住，
    以免日後有人再依賴那個已經不成立的保證。

    對**完整**的錯置表守門仍然會叫（下面第二段），但那是因為其餘的列落在
    區間外，不是因為構造上不可能。
    """
    feb_lo, feb_hi = cutoff_window(FEB)
    t7_lo, t7_hi = cutoff_window(MAR_T7)
    assert t7_lo <= feb_hi and feb_lo <= t7_hi, "區間應該重疊，這條測試的前提是重疊存在"

    # 只含重疊那幾天的子集：蒙得過去（這就是被削弱的部分）
    sneaky = pl.DataFrame({"cutoff": [20170222, 20170228]})
    assert_cutoffs_within_window(sneaky, FEB)

    # 完整的 mar_t7 表：擋得下來
    full = pl.DataFrame({"cutoff": [20170222, 20170301, 20170324]})
    with pytest.raises(AssertionError):
        assert_cutoffs_within_window(full, FEB)


def _tx(rows: list[tuple[str, int, int, int]]) -> pl.DataFrame:
    """(msno, transaction_date, is_cancel, actual_amount_paid) → 交易明細。"""
    return pl.DataFrame(
        {
            "msno": [r[0] for r in rows],
            "transaction_date": [r[1] for r in rows],
            "is_cancel": [r[2] for r in rows],
            "actual_amount_paid": [r[3] for r in rows],
            "is_auto_renew": [1 for _ in rows],
            "plan_list_price": [149 for _ in rows],
            "payment_plan_days": [30 for _ in rows],
            "payment_method_id": [41 for _ in rows],
        }
    )


@NODATA
def test_the_expiry_day_cancellation_disappears_at_t7():
    """**這就是提前評分的代價，一列資料就看得出來。**

    u0 在到期日（0228）當天取消，另有一筆 0110 的正常交易。
    T=0 看得到那筆取消（`last_is_cancel = 1`）；T−7（cutoff 0221）看不到，
    最後一筆變成 0110 那筆，旗標翻回 0。

    M5 量到 `last_is_cancel` 佔投放名單解釋強度的 42.35%（§7.14）—— 這一列
    就是那 42.35% 消失的機制。
    """
    tx = _tx([("u0", 20170110, 0, 149), ("u0", 20170228, 1, 0)])

    for cutoff, expected_cancel, expected_n in ((20170228, 1.0, 2), (20170221, 0.0, 1)):
        joined = tx.with_columns(pl.lit(cutoff).alias("cutoff"), pl.lit(1).alias("is_churn"))
        out = aggregate_asof(joined.lazy()).collect()

        assert out["last_is_cancel"][0] == expected_cancel
        assert out["n_tx"][0] == expected_n
        assert_asof_respected(out)  # 紅線 1 在兩個時點都必須成立


@NODATA
def test_a_user_with_no_history_before_the_earlier_cutoff_drops_out():
    """到期前 7 天內才第一次交易的人，在 T−7 版本裡整個消失。

    ⚠️ **這不是 bug，是部署現實**：提前 7 天評分時，這個人還沒有可用的歷史。
    但它的後果是兩個版本的 cohort 成員**不同**，所以「分數下降多少」不能直接
    比 —— `scripts/lead_time.py` 因此另外在交集上比一次，並報出掉了幾個人。
    """
    tx = _tx([("u_new", 20170225, 0, 149)])

    at_expiry = aggregate_asof(
        tx.with_columns(pl.lit(20170228).alias("cutoff"), pl.lit(1).alias("is_churn")).lazy()
    ).collect()
    at_t7 = aggregate_asof(
        tx.with_columns(pl.lit(20170221).alias("cutoff"), pl.lit(1).alias("is_churn")).lazy()
    ).collect()

    assert at_expiry.height == 1
    assert at_t7.height == 0, "截斷之後一列不剩的人不該留在 cohort 裡"


@NODATA
def test_lead_days_zero_is_exactly_the_current_behaviour():
    """`lead_days = 0` 必須與加這個欄位之前完全一樣。

    這條是迴歸測試：M1–M5 的所有數字都建立在 T=0 的切分上，加一個預設值為 0
    的欄位不得改變它們。實際的端到端確認是重跑 `make explain` 看 Mar log loss
    仍是 0.15367。
    """
    spec = CohortSpec("x", "train.csv", 20170201, 20170228, "2017-03")
    assert spec.lead_days == 0
    assert cutoff_window(spec) == (spec.expire_start, spec.expire_end)
