"""M4 業務指標的純邏輯測試 —— 不需資料，CI 要求零 skip。

這一層把機率換算成錢，錯了不會有任何跡象：曲線照樣畫得出來、極大值照樣
存在，只是位置錯了，而「該投放給前幾 %」正是整個專案要交付的那個數字。

因此測試集中在三件事：

1. **恆等式** —— 期望曲線的極大值必然落在最後一個 `p > p*` 的人身上。
   這是代數結果不是經驗規律，可以用手算的案例逐點對。
2. **參數的可互換性** —— `r_save` 加倍與 `C_offer` 減半必須給出相同決策。
   這條若壞了，敏感度熱圖會長出假的結構。
3. **低估 → 門檻過於保守** —— SPEC §6.1 預告的方向，用合成資料釘住。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation import (
    campaign_curve,
    decision_threshold,
    expected_months,
    fixed_rule_point,
    optimal_point,
    sensitivity_grid,
)
from tests.conftest import NODATA


def _grouped(spec: list[tuple[float, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """依 (預測值, 人數, 實際流失率) 造資料。正例數取整，沒有抽樣雜訊。"""
    ys, ps = [], []
    for p, n, rate in spec:
        n_pos = int(round(n * rate))
        ys.append(np.array([1] * n_pos + [0] * (n - n_pos), dtype=np.float64))
        ps.append(np.full(n, p, dtype=np.float64))
    return np.concatenate(ys), np.concatenate(ps)


@NODATA
def test_threshold_is_cost_over_gain():
    """p* = C / (r × LTV)。手算：150 / (0.15 × 2000) = 0.5。"""
    assert decision_threshold(r_save=0.15, ltv_saved=2000, c_offer=150) == pytest.approx(0.5)


@NODATA
def test_threshold_rejects_zero_gain():
    """挽回收益為 0 時門檻是無限大，讓它報錯而不是回傳 inf。

    inf 會安靜地傳染到下游每一個計算，最後畫出一張全空的圖 —— 那看起來
    像「沒有人值得投放」，而不是「參數設錯了」。
    """
    with pytest.raises(ValueError):
        decision_threshold(r_save=0.0, ltv_saved=2000, c_offer=150)
    with pytest.raises(ValueError):
        decision_threshold(r_save=0.15, ltv_saved=0, c_offer=150)


@NODATA
def test_expected_months_is_inverse_of_hazard():
    """幾何存活：月流失率 6.39% → 期望續訂 15.6 個月。"""
    assert expected_months(0.063915) == pytest.approx(15.65, abs=0.01)
    with pytest.raises(ValueError):
        expected_months(0.0)
    with pytest.raises(ValueError):
        expected_months(1.5)


@NODATA
def test_expected_curve_peaks_exactly_at_the_threshold():
    """期望曲線的極大值落在最後一個 `p > p*` 的人身上 —— 這是恆等式。

    排在他後面的每個人期望收益都是負的，加進去只會讓總額變小。所以這條
    測試不是在驗「差不多對」，是在驗一個代數結果。
    """
    p = np.arange(1, 1001) / 1000.0  # 0.001 ~ 1.000，每人一個相異值
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=1000) < p).astype(np.float64)

    # gain = 1 × 1000 = 1000，C = 500.5 → p* = 0.5005 → 恰好 500 人達標
    curve = campaign_curve(y, p, r_save=1.0, ltv_saved=1000, c_offer=500.5, step=0.001)
    best = optimal_point(curve, by="期望模擬淨收益")

    assert decision_threshold(r_save=1.0, ltv_saved=1000, c_offer=500.5) == pytest.approx(0.5005)
    assert best["投放人數"] == 500
    assert best["門檻機率"] == pytest.approx(0.501)


@NODATA
def test_perfect_model_makes_expected_and_realized_identical():
    """p 等於真實標籤時，期望曲線與實際曲線必須逐點相同。

    兩條線的落差**只**能來自機率失準。這條測試把「沒有失準時落差為零」
    釘住，否則落差的解讀就沒有基準。
    """
    y = np.array([1.0] * 300 + [0.0] * 700)
    curve = campaign_curve(y, y, r_save=0.2, ltv_saved=1000, c_offer=50, step=0.01)

    assert np.allclose(curve["期望模擬淨收益"].to_numpy(), curve["標籤結算模擬淨收益"].to_numpy())


@NODATA
def test_underestimating_risk_makes_the_threshold_too_conservative():
    """低估流失風險 → 投放人數變少。這是 SPEC §6.1 預告的方向。

    同一批人、同一組業務參數，只把預測機率整體縮小到 0.72 倍（對應 §7.10
    量到的 −27.6%），期望規則選出來的人就變少 —— 而真實風險沒有變。
    **少投放的那些人裡面有真的會流失的，那就是低估的價格。**
    """
    truth = [(0.9, 200, 0.9), (0.6, 200, 0.6), (0.45, 200, 0.45), (0.2, 400, 0.2)]
    y, p_true = _grouped(truth)
    p_low = p_true * 0.72

    kwargs = {"r_save": 0.15, "ltv_saved": 2000, "c_offer": 150}  # p* = 0.5
    honest = optimal_point(campaign_curve(y, p_true, step=0.001, **kwargs), by="期望模擬淨收益")
    biased = optimal_point(campaign_curve(y, p_low, step=0.001, **kwargs), by="期望模擬淨收益")

    assert honest["投放人數"] == 400, "p* = 0.5，應選中 0.9 與 0.6 兩組"
    assert biased["投放人數"] == 200, "縮小 0.72 倍後 0.6 → 0.432，掉到門檻以下"
    assert biased["投放人數"] < honest["投放人數"]


@NODATA
def test_lift_is_precision_over_base_rate():
    """lift = 命中率 / 全體流失率。完美排序在 K = 流失率時達到 1/base。"""
    y, p = _grouped([(0.9, 100, 1.0), (0.1, 900, 0.0)])  # 前 100 人全部流失

    curve = campaign_curve(y, p, r_save=0.2, ltv_saved=1000, c_offer=50, step=0.01)
    at_10pct = curve.filter((pl.col("K") - 0.1).abs() < 1e-9).row(0, named=True)

    assert at_10pct["命中率"] == pytest.approx(1.0)
    assert at_10pct["lift"] == pytest.approx(10.0)  # base rate 0.1


@NODATA
def test_lift_is_one_when_everyone_is_targeted():
    """K = 100% 時 lift 必為 1 —— 投放給所有人就等於沒有排序。"""
    rng = np.random.default_rng(1)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < 0.3).astype(np.float64)

    curve = campaign_curve(y, p, r_save=0.2, ltv_saved=1000, c_offer=50, step=0.01)

    assert curve["lift"].to_numpy()[-1] == pytest.approx(1.0)
    assert curve["投放人數"].to_numpy()[-1] == 2000


@NODATA
def test_doubling_save_rate_equals_halving_offer_cost():
    """r_save 加倍與 C_offer 減半必須給出完全相同的決策。

    兩者只透過 p* = C/(r×LTV) 影響決策。這條若壞了，敏感度熱圖上會出現
    實際不存在的結構 —— 而熱圖的用途正是展示結論對假設的穩健性。
    """
    rng = np.random.default_rng(2)
    p = rng.uniform(size=5000)
    y = (rng.uniform(size=5000) < p).astype(np.float64)

    grid = sensitivity_grid(y, p, ltv_saved=2000, r_values=[0.10, 0.20], c_values=[100, 200])
    a = grid.filter((pl.col("r_save") == 0.10) & (pl.col("c_offer") == 100)).row(0, named=True)
    b = grid.filter((pl.col("r_save") == 0.20) & (pl.col("c_offer") == 200)).row(0, named=True)

    assert a["p*"] == pytest.approx(b["p*"])
    assert a["投放人數"] == b["投放人數"]


@NODATA
def test_sensitivity_grid_covers_every_combination():
    """網格必須是完整的笛卡兒積 —— 少一格會在熱圖上變成空白，不易察覺。"""
    y, p = _grouped([(0.8, 100, 0.8), (0.2, 100, 0.2)])
    grid = sensitivity_grid(y, p, ltv_saved=1000, r_values=[0.1, 0.2, 0.3], c_values=[50, 100])

    assert grid.height == 6
    assert grid["最佳投放比例"].max() <= 1.0
    assert grid["最佳投放比例"].min() >= 0.0


@NODATA
def test_fixed_rule_point_is_hand_computable():
    """不需模型的規則（例：全部投給新進用戶）。模型必須贏過它才有意義。

    手算：選中 200 人、其中 80 人流失、r×LTV = 200、C = 50
          → 80 × 200 − 200 × 50 = 16000 − 10000 = 6000
    """
    y = np.array([1.0] * 80 + [0.0] * 120 + [1.0] * 20 + [0.0] * 780)
    mask = np.array([True] * 200 + [False] * 800)

    point = fixed_rule_point(y, mask, r_save=0.2, ltv_saved=1000, c_offer=50, label="全部新進用戶")

    assert point["投放人數"] == 200
    assert point["命中數"] == 80
    assert point["標籤結算模擬淨收益"] == pytest.approx(6000.0)
    assert point["命中率"] == pytest.approx(0.4)
    assert point["lift"] == pytest.approx(4.0)  # base rate 0.1


@NODATA
def test_fixed_rule_rejects_empty_mask():
    with pytest.raises(ValueError, match="沒有選中任何人"):
        fixed_rule_point([0, 1], [False, False], r_save=0.2, ltv_saved=1000, c_offer=50, label="x")


@NODATA
def test_curve_rejects_bad_input():
    with pytest.raises(ValueError, match="長度不符"):
        campaign_curve([0, 1], [0.5], r_save=0.2, ltv_saved=1000, c_offer=50)
    with pytest.raises(ValueError):
        campaign_curve([], [], r_save=0.2, ltv_saved=1000, c_offer=50)
    with pytest.raises(ValueError, match="step"):
        campaign_curve([0, 1], [0.1, 0.9], r_save=0.2, ltv_saved=1000, c_offer=50, step=0.0)


@NODATA
def test_optimal_point_rejects_unknown_column():
    y, p = _grouped([(0.8, 100, 0.8), (0.2, 100, 0.2)])
    curve = campaign_curve(y, p, r_save=0.2, ltv_saved=1000, c_offer=50)
    with pytest.raises(KeyError):
        optimal_point(curve, by="不存在的欄位")
