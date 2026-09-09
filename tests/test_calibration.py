"""M4 校準診斷的純邏輯測試 —— 不需資料，CI 要求零 skip。

這些函式全部是「餵 y 與 p 進去就有答案」的純函式，因此可以用手刻的極端
案例逐條驗證。重點放在**會安靜出錯**的地方：分箱邊界、空箱、ECE 的加權、
以及「完美校準」與「完全失準」兩端是否落在預期的數值上。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation import (
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    max_calibration_error,
    reliability_curve,
)
from tests.conftest import NODATA


@NODATA
def test_brier_is_zero_for_perfect_predictions():
    """完美預測的 Brier = 0，完全相反的預測 = 1。兩端定錨。"""
    y = [0, 1, 0, 1]
    assert brier_score(y, [0.0, 1.0, 0.0, 1.0]) == pytest.approx(0.0)
    assert brier_score(y, [1.0, 0.0, 1.0, 0.0]) == pytest.approx(1.0)


@NODATA
def test_brier_of_constant_prediction_equals_variance():
    """對所有人預測基準率時，Brier 應等於標籤的變異數 p(1−p)。

    這是常數預測的理論值，拿來確認實作沒有偏差 —— 手算得出來的案例
    比「跑起來不會錯」有意義得多。
    """
    y = [0] * 90 + [1] * 10  # 基準率 0.1
    assert brier_score(y, [0.1] * 100) == pytest.approx(0.1 * 0.9)


@NODATA
def test_brier_rejects_length_mismatch():
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5])


@NODATA
def test_calibration_in_the_large_reports_signed_bias():
    """整體偏差必須帶正負號 —— ECE 取絕對值，會把方向資訊丟掉。

    低估與高估在業務上是相反的錯誤：低估讓投放門檻過於保守（漏掉該救的人），
    高估則會把預算撒在不會流失的人身上。摘要數字不能分不出這兩者。
    """
    y = [0] * 90 + [1] * 10  # 實際 10%
    out = calibration_in_the_large(y, [0.05] * 100)  # 預測 5%
    assert out["實際流失率"] == pytest.approx(0.10)
    assert out["平均預測"] == pytest.approx(0.05)
    assert out["偏差"] == pytest.approx(-0.05)
    assert out["相對偏差"] == pytest.approx(-0.5)


@NODATA
def test_quantile_binning_gives_equal_sized_bins():
    """等量分箱的每一箱樣本數應該一樣（整除時）。"""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(0, 1, 1000) < p).astype(int)

    curve = reliability_curve(y, p, n_bins=10, strategy="quantile")

    assert curve.height == 10
    assert curve["樣本數"].to_list() == [100] * 10


@NODATA
def test_uniform_binning_collapses_on_skewed_predictions():
    """等寬分箱在預測集中於低機率時會塌成少數幾箱 —— 這正是預設用等量的理由。

    這條測試的用意不是「等寬不好」，而是把它的失效條件釘住：本專案的
    預測有九成落在 0.02 以下（**但彼此不同**），等寬分箱會讓那九成全部
    擠進第一箱，等量分箱則能把它們攤開。
    """
    p = np.concatenate([np.linspace(1e-5, 0.02, 900), np.linspace(0.3, 0.9, 100)])
    y = np.zeros(1000, dtype=int)

    uniform = reliability_curve(y, p, n_bins=10, strategy="uniform")
    quantile = reliability_curve(y, p, n_bins=10, strategy="quantile")

    assert uniform["樣本數"].max() >= 900, "等寬分箱應該把九成樣本擠進同一箱"
    assert quantile["樣本數"].max() == 100, "等量分箱應該把它們均分成十箱"


@NODATA
def test_quantile_binning_cannot_split_tied_predictions():
    """等量分箱也有失效條件：**預測值相同的樣本無法被拆開**。

    900 筆預測值一模一樣時，相鄰分位數會相等，去重之後箱數就少於要求的
    數量。這不是 bug —— 沒有任何分箱法能把相同的值分到不同箱。

    釘住它是因為它會安靜地發生：呼叫端要 10 箱、拿到 2 箱，若沒察覺就會
    以為「模型在這個區間很平」，實際上是分箱塌了。所以 `reliability_curve`
    回傳的列數就是實際箱數，呼叫端看得到。
    """
    p = np.concatenate([np.full(900, 0.001), np.linspace(0.3, 0.9, 100)])
    y = np.zeros(1000, dtype=int)

    curve = reliability_curve(y, p, n_bins=10, strategy="quantile")

    assert curve.height < 10, "相同的預測值無法拆箱，箱數必然少於要求"
    assert curve["樣本數"].max() == 900


@NODATA
def test_reliability_curve_marks_thin_bins():
    """樣本不足的箱要保留並標記，不能默默丟掉。

    丟掉會讓圖看起來比實際乾淨，而「這一段沒有足夠資料下判斷」本身
    就是要呈現的結論。
    """
    p = np.linspace(0.01, 0.99, 50)
    y = (p > 0.5).astype(int)

    curve = reliability_curve(y, p, n_bins=5)

    assert curve.height == 5
    assert not curve["樣本足夠"].any(), "每箱只有 10 筆，全部都該標成樣本不足"


@NODATA
def test_reliability_curve_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="未知的分箱策略"):
        reliability_curve([0, 1], [0.1, 0.9], strategy="kmeans")


@NODATA
def test_perfectly_calibrated_data_has_near_zero_ece():
    """每箱的預測值等於該箱的實際頻率時，ECE 應趨近 0。"""
    # 四箱，每箱 1000 筆，預測值剛好等於該箱的真實比例。
    parts = []
    for rate in (0.1, 0.3, 0.6, 0.9):
        n_pos = int(1000 * rate)
        parts.append((np.full(1000, rate), np.array([1] * n_pos + [0] * (1000 - n_pos))))
    p = np.concatenate([a for a, _ in parts])
    y = np.concatenate([b for _, b in parts])

    curve = reliability_curve(y, p, n_bins=4, strategy="quantile")
    assert expected_calibration_error(curve) == pytest.approx(0.0, abs=1e-9)


@NODATA
def test_ece_is_weighted_by_bin_size():
    """ECE 必須依樣本數加權，不是各箱平均。

    不加權的話，一個只有 10 筆、誤差很大的箱，會跟一個 10 萬筆、誤差很小的
    箱有同樣的發言權 —— 那個數字就不再代表「隨機抽一個人平均差多少」。
    """
    curve = pl.DataFrame(
        {
            "樣本數": [9000, 1000],
            "平均預測": [0.10, 0.50],
            "實際流失率": [0.10, 0.60],  # 大箱誤差 0，小箱誤差 0.1
        }
    )
    # 加權：(9000*0 + 1000*0.1) / 10000 = 0.01；未加權會是 0.05
    assert expected_calibration_error(curve) == pytest.approx(0.01)


@NODATA
def test_ece_rejects_missing_columns():
    with pytest.raises(KeyError):
        expected_calibration_error(pl.DataFrame({"樣本數": [1], "平均預測": [0.1]}))


@NODATA
def test_mce_ignores_thin_bins():
    """MCE 只看樣本足夠的箱 —— 小箱的觀測頻率本身就在抖。"""
    curve = pl.DataFrame(
        {
            "樣本數": [50_000, 10],
            "平均預測": [0.10, 0.10],
            "實際流失率": [0.15, 0.90],  # 小箱誤差 0.8，但只有 10 筆
            "樣本足夠": [True, False],
        }
    )
    assert max_calibration_error(curve) == pytest.approx(0.05)
