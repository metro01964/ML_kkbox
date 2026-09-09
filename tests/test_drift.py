"""M6 · PSI 漂移監控的純邏輯測試。

全部手刻資料，標 nodata、在 CI 上實際執行。真實 cohort 上的數字走
`scripts/drift_report.py`。

守的是四件會讓監控**安靜失效**的事：

    箱界從當期算        分布完全相同時 PSI 不再是 0，漂移與分位數差混在一起
    旗標欄走分位數      0/1 欄的重複箱界把漂移壓成 0
    缺失被丟掉          「缺失率從 0 變成 30%」得到 PSI = 0
    epsilon 旗標常亮    永遠亮著的旗標等於沒有旗標
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.evaluation.drift import (
    EPSILON,
    MISSING_LABEL,
    UNSEEN_LABEL,
    apply_bins,
    band,
    fit_bins,
    noise_floor,
    psi,
    psi_from_proportions,
    psi_table,
    score_psi,
)
from tests.conftest import NODATA


@NODATA
def test_psi_matches_the_formula_by_hand():
    """兩箱、手算對照。PSI 的公式只有一種寫法，但符號很容易寫反。

    (0.5 − 0.6) × ln(0.5/0.6) + (0.5 − 0.4) × ln(0.5/0.4)
    = 0.018232 + 0.022314 = 0.040546
    """
    value, floored = psi_from_proportions(np.array([0.5, 0.5]), np.array([0.6, 0.4]))
    assert value == pytest.approx(0.040546, abs=1e-6)
    assert floored is False


@NODATA
def test_identical_distributions_give_zero():
    """完全相同 → 0。PSI 非負，所以這是它唯一的下界。"""
    ref = pl.Series("x", [float(i) for i in range(1000)])
    scheme = fit_bins(ref)
    value, floored, _ = psi(scheme, ref)
    assert value == pytest.approx(0.0, abs=1e-12)
    assert floored is False


@NODATA
def test_bigger_shifts_give_bigger_psi():
    """單調性：位移越大，PSI 越大。這條抓的是符號寫反或分母寫錯。"""
    rng = np.random.default_rng(0)
    ref = pl.Series("x", rng.normal(0, 1, 20_000))
    scheme = fit_bins(ref)
    values = [
        psi(scheme, pl.Series("x", rng.normal(shift, 1, 20_000)))[0] for shift in (0.1, 0.5, 1.0)
    ]
    assert values == sorted(values)
    assert values[0] < 0.1 < values[-1]


@NODATA
def test_bin_edges_come_from_the_reference_only():
    """箱界是一個**擬合出來的狀態**，只能從參考期算（紅線 5 的同一類東西）。

    ⚠️ 若箱界改從當期（或兩期合併）算，位移後的資料會再次被切成等量十箱 ——
    每箱都是 10%，PSI 因此接近 0：**一個真實的位移會回報「非常穩定」。**
    """
    ref = pl.Series("x", [float(i) for i in range(10_000)])
    scheme = fit_bins(ref)
    shifted = pl.Series("x", [float(i) + 20_000 for i in range(10_000)])

    # 箱界沒有改變 → 位移後的資料全部落在最後一箱。
    proportions = apply_bins(scheme, shifted)
    assert proportions[-2] == pytest.approx(1.0)  # 最後一個數值箱（-1 是缺失箱）

    # 對照：若箱界從當期算，同一份資料會被切成等量十箱、PSI 幾乎是 0。
    self_fitted = fit_bins(shifted)
    naive, _ = psi_from_proportions(self_fitted.reference, apply_bins(self_fitted, shifted))
    assert naive == pytest.approx(0.0, abs=1e-12)
    assert psi(scheme, shifted)[0] > 1.0


@NODATA
def test_binary_flags_are_binned_by_value_not_by_quantile():
    """0/1 旗標走離散那條，否則「免費方案佔比翻十倍」會得到 PSI ≈ 0。"""
    ref = pl.Series("is_free_plan", [0.0] * 970 + [1.0] * 30)
    scheme = fit_bins(ref)
    assert scheme.kind == "discrete"
    assert scheme.n_bins == 3  # 0 / 1 / 缺失

    current = pl.Series("is_free_plan", [0.0] * 700 + [1.0] * 300)
    value, floored, detail = psi(scheme, current)
    assert value > 0.25, "3% → 30% 應該是顯著漂移"
    assert floored is False
    # 貢獻最大的那一箱要是 1，營運才問得出「是哪一箱動了」。
    assert detail.sort("contribution", descending=True)["bin"][0] == "1"


@NODATA
def test_missingness_is_its_own_bin():
    """缺失率的變化必須被看到 —— 上游 join 壞掉就是這個形狀。"""
    ref = pl.Series("bd_clean", [float(i % 50 + 10) for i in range(1000)])
    scheme = fit_bins(ref)
    assert MISSING_LABEL in scheme.labels

    current = pl.Series(
        "bd_clean", [None if i < 300 else float(i % 50 + 10) for i in range(1000)], dtype=pl.Float64
    )
    value, floored, detail = psi(scheme, current)
    assert detail.filter(pl.col("bin") == MISSING_LABEL)["current"][0] == pytest.approx(0.3)
    # 參考期的缺失箱是 0、當期是 0.3 —— 單邊空箱，PSI 由 epsilon 決定。
    assert floored is True
    assert value > 1.0


@NODATA
def test_unseen_categories_get_their_own_bin_and_are_flagged():
    """訓練時沒見過的類別在推論時會落到缺失分支 —— 那件事要被監控看到。

    實測 Mar 有一個 Feb 沒有的 `last_payment_method_id`（§7.4）。
    """
    ref = pl.Series("last_payment_method_id", [41.0] * 500 + [36.0] * 500)
    scheme = fit_bins(ref, categorical=True)
    assert UNSEEN_LABEL in scheme.labels

    current = pl.Series("last_payment_method_id", [41.0] * 499 + [36.0] * 500 + [99.0])
    value, floored, detail = psi(scheme, current)
    unseen = detail.filter(pl.col("bin") == UNSEEN_LABEL)
    assert unseen["current"][0] == pytest.approx(0.001)
    assert floored is True
    assert np.isfinite(value)


@NODATA
def test_the_floored_flag_is_not_always_on():
    """⚠️ 一個永遠亮著的旗標等於沒有旗標（M5 在 `git_dirty` 上踩過這個坑）。

    `__missing__` 與 `__unseen__` 是結構性的箱：一個完全沒有缺失的欄位，兩邊
    都是 0，那不是「出現了參考期沒有的東西」。兩邊都空的箱要先丟掉。
    """
    ref = pl.Series("n_tx", [float(i % 40 + 1) for i in range(2000)])
    scheme = fit_bins(ref)
    _, floored, _ = psi(scheme, pl.Series("n_tx", [float(i % 40 + 1) for i in range(2000)]))
    assert floored is False

    cat = pl.Series("city", [1.0] * 500 + [13.0] * 500)
    cat_scheme = fit_bins(cat, categorical=True)
    _, cat_floored, _ = psi(cat_scheme, pl.Series("city", [1.0] * 400 + [13.0] * 600))
    assert cat_floored is False, "沒有新類別、沒有缺失時，旗標不該亮"


@NODATA
def test_psi_table_refuses_mismatched_columns():
    """欄位對不上時逐欄比 PSI 會比錯人，這種表沒有意義。"""
    a = pl.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 1.0, 2.0]})
    b = pl.DataFrame({"x": [1.0, 2.0, 3.0], "z": [1.0, 1.0, 2.0]})
    with pytest.raises(ValueError, match="欄位不一致"):
        psi_table(a, b)


@NODATA
def test_psi_table_reports_missing_rate_change_separately():
    """缺失率的變化單獨報一欄 —— PSI 把它跟其他變化混在一個數字裡。"""
    ref = pl.DataFrame({"a": [float(i) for i in range(500)]})
    cur = pl.DataFrame(
        {"a": [None if i < 100 else float(i) for i in range(500)]}, schema={"a": pl.Float64}
    )
    summary, detail = psi_table(ref, cur)
    row = summary.row(0, named=True)
    assert row["reference_missing"] == 0.0
    assert row["current_missing"] == pytest.approx(0.2)
    assert row["missing_delta"] == pytest.approx(0.2)
    assert detail.height == summary["n_bins"][0]


@NODATA
def test_noise_floor_is_small_when_there_is_no_drift():
    """把同一份資料切兩半 —— 這就是「完全沒有漂移」時 PSI 的樣子。

    它是判讀 0.1 / 0.25 的基準線：慣例門檻沒有樣本數修正，而 PSI 對小樣本會
    因為抽樣雜訊而變大。
    """
    rng = np.random.default_rng(7)
    frame = pl.DataFrame({"a": rng.normal(size=20_000), "b": rng.integers(0, 5, 20_000) * 1.0})
    floor = noise_floor(frame, rounds=3, seed=1)
    assert floor.height == 2
    assert floor["p95_all"][0] < 0.01, "兩萬列、無漂移，PSI 應該遠低於 0.1 的慣例門檻"
    assert (floor["psi_mean"] >= 0).all()


@NODATA
def test_noise_floor_grows_when_the_sample_is_small():
    """同樣沒有漂移，樣本數小時 PSI 會變大 —— 這正是慣例門檻沒有處理的事。"""
    rng = np.random.default_rng(11)
    big = pl.DataFrame({"a": rng.normal(size=20_000)})
    small = pl.DataFrame({"a": rng.normal(size=400)})
    assert (
        noise_floor(small, rounds=3, seed=2)["p95_all"][0]
        > (noise_floor(big, rounds=3, seed=2)["p95_all"][0])
    )


@NODATA
def test_score_psi_sees_a_shifted_score_distribution():
    """分數漂移是與特徵漂移不同的監控：61 個特徵微幅移動可能抵消，也可能疊加。"""
    rng = np.random.default_rng(3)
    ref = rng.beta(1.5, 20, 20_000)
    same, floored, _ = score_psi(ref, rng.beta(1.5, 20, 20_000))
    shifted, _, _ = score_psi(ref, rng.beta(2.5, 20, 20_000))
    assert same < 0.01
    assert shifted > same * 5
    assert floored is False


@NODATA
def test_bands_are_the_conventional_thresholds():
    """慣例分級。⚠️ 它是慣例不是檢定 —— 這條測試只釘住我們用的是哪一組。"""
    assert band(0.05) == "穩定"
    assert band(0.10) == "中度"
    assert band(0.24) == "中度"
    assert band(0.25) == "顯著"
    assert band(9.9) == "顯著"


@NODATA
def test_all_null_reference_raises_instead_of_guessing():
    """全是 null 的欄位定不出箱界，報錯比回一個 0 好 —— 後者讀起來像「很穩定」。"""
    with pytest.raises(ValueError, match="全是 null"):
        fit_bins(pl.Series("x", [None, None], dtype=pl.Float64))


@NODATA
def test_epsilon_is_recorded_because_it_decides_the_number():
    """單邊空箱時 PSI 由 epsilon 決定，所以換 epsilon 會換一個數字。"""
    ref, cur = np.array([1.0, 0.0]), np.array([0.9, 0.1])
    loose, _ = psi_from_proportions(ref, cur, epsilon=1e-3)
    tight, _ = psi_from_proportions(ref, cur, epsilon=EPSILON)
    assert tight > loose * 1.5, "epsilon 越小，同一個空箱的 PSI 越大"
