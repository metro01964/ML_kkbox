"""M4 校準器的純邏輯測試 —— 不需資料，CI 要求零 skip。

`test_calibration.py` 測的是「量測」，這一份測的是「修正」。重點放在三件
會安靜出錯的事：

1. **零區塊的下界** —— isotonic 輸出剛好 0，套到新資料上遇到正例就會踩到
   log loss 的 clip，單筆貢獻 34.5。它長得像「校準讓模型變差」，不像 bug。
2. **排序反轉被壓平** —— 這是選 isotonic 而不是 Platt 的唯一理由，
   所以要有一條測試釘住「它真的做到了」。
3. **同分的代價** —— 壓平就是製造同分，而 SPEC §6.2 的投放決策靠排序。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation import (
    calibration_in_the_large,
    fit_isotonic,
    log_loss,
    roc_auc,
    tie_profile,
)
from tests.conftest import NODATA


def _grouped(spec: list[tuple[float, int, float]]) -> tuple[np.ndarray, np.ndarray]:
    """依 (預測值, 人數, 實際流失率) 造一批標籤與預測。

    每組的正例數取整數，所以實際比率是精確的 —— 測試裡不該有抽樣雜訊，
    否則失敗時分不清是實作錯了還是這次 seed 運氣不好。
    """
    ys, ps = [], []
    for p, n, rate in spec:
        n_pos = int(round(n * rate))
        ys.append(np.array([1] * n_pos + [0] * (n - n_pos), dtype=np.float64))
        ps.append(np.full(n, p, dtype=np.float64))
    return np.concatenate(ys), np.concatenate(ps)


@NODATA
def test_isotonic_removes_a_known_monotone_bias():
    """實際流失率固定是預測值的一半時，校準後整體偏差應趨近 0。

    這是最基本的定錨：失準的形狀已知且單調，isotonic 必須把它修掉。
    """
    y, p = _grouped([(v / 10, 2000, v / 20) for v in range(1, 10)])

    cal = fit_isotonic(y, p)
    after = calibration_in_the_large(y, cal.apply(p))

    assert abs(after["相對偏差"]) < 0.01, "已知的單調失準應該被修掉"


@NODATA
def test_isotonic_flattens_a_rank_inversion():
    """排序反轉的兩段會被壓成同一個值 —— 這是選 isotonic 的唯一理由。

    M4 診斷在 D7 → D8 量到的就是這個形狀：平均預測升高，實際流失率反而
    下降。Platt 是單調平滑映射，只會平移整條曲線，這段反轉會原封不動
    留著；isotonic 強制單調，把它壓平成兩段的加權平均。
    """
    # 預測 0.10 的那群實際 30%，預測 0.20 的那群實際 10% —— 反過來了。
    y, p = _grouped([(0.10, 1000, 0.30), (0.20, 1000, 0.10)])

    cal = fit_isotonic(y, p)
    low, high = cal.apply([0.10, 0.20])

    assert low == pytest.approx(high), "反轉的兩段必須被壓平成同值"
    assert low == pytest.approx(0.20, abs=1e-6), "壓平後應是兩段的加權平均"


@NODATA
def test_floor_comes_from_the_size_of_the_zero_block():
    """下界由「零區塊有多大」決定，不是寫死的 epsilon。

    5000 筆全陰只能支持「機率低於千分之幾」，不能支持「機率是 0」。
    Jeffreys 後驗均值 0.5/(m+1) 把這句話寫成數字。
    """
    y, p = _grouped([(0.001, 5000, 0.0), (0.20, 1000, 0.20), (0.50, 1000, 0.60)])

    cal = fit_isotonic(y, p)

    assert cal.n_zero_block == 5000
    assert cal.floor == pytest.approx(0.5 / 5001)
    assert cal.apply([0.001])[0] == pytest.approx(cal.floor)


@NODATA
def test_floor_is_inert_when_every_block_has_positives():
    """沒有零區塊時下界是 0 —— 這個機制不會憑空改動任何預測。"""
    y, p = _grouped([(0.10, 1000, 0.05), (0.30, 1000, 0.30), (0.60, 1000, 0.70)])

    cal = fit_isotonic(y, p)

    assert cal.n_zero_block == 0
    assert cal.floor == 0.0


@NODATA
def test_zero_output_would_blow_up_log_loss():
    """把下界關掉，log loss 會被單一個正例炸掉 —— 這條測試就是那個代價。

    校準集裡 5000 筆全陰 → isotonic 對那段輸出 0。評估集同一段出現一個
    正例，clip 到 1e-15 之後單筆貢獻 −ln(1e-15) ≈ 34.5，除以 100 筆
    就是 0.345，比整個 M3 的改善幅度還大好幾倍。
    """
    y_fit, p_fit = _grouped([(0.001, 5000, 0.0), (0.20, 1000, 0.20), (0.50, 1000, 0.60)])

    with_floor = fit_isotonic(y_fit, p_fit)
    without_floor = fit_isotonic(y_fit, p_fit, floor=0.0)

    # 評估集：同一個低機率區間，但這次有一個人真的流失了。
    y_eval, p_eval = _grouped([(0.001, 100, 0.01)])

    loss_floored = log_loss(y_eval, with_floor.apply(p_eval))
    loss_raw = log_loss(y_eval, without_floor.apply(p_eval))

    assert np.isfinite(loss_raw) and np.isfinite(loss_floored)
    assert loss_raw > 0.3, "沒有下界時，單一個正例就足以讓 log loss 難看"
    assert loss_floored < loss_raw / 3, "下界應該把這個懲罰壓下來"


@NODATA
def test_out_of_range_predictions_clip_instead_of_extrapolate():
    """超出校準集範圍的預測取端點值，不外推。

    Mar cohort 的最高預測可能高於 Feb-sel 的最高預測。那一段沒有任何觀測
    支持，往上外推是憑空發明機率。
    """
    y, p = _grouped([(0.10, 1000, 0.05), (0.40, 1000, 0.40)])

    cal = fit_isotonic(y, p)

    assert cal.apply([0.99])[0] == pytest.approx(cal.apply([0.40])[0])
    assert cal.apply([0.001])[0] == pytest.approx(cal.apply([0.10])[0])


@NODATA
def test_calibration_is_monotone_non_decreasing():
    """輸出必須隨輸入單調不減 —— 這是 isotonic 的定義，也是它保住排序的基礎。"""
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(np.float64)

    cal = fit_isotonic(y, p)
    out = cal.apply(np.sort(p))

    assert np.all(np.diff(out) >= -1e-12)


@NODATA
def test_flattening_an_inversion_raises_auc_instead_of_lowering_it():
    """⚠️ 壓平排序反轉會讓 AUC **上升**，不是下降。

    直覺上「校準犧牲排序能力」（SPEC §7 的風險欄位就是這樣寫的），但那個
    直覺只在被壓平的區段原本排對了的時候成立。AUC 對同分各給一半，而反轉的
    配對原本得 0 分 —— 把它們壓成同分是 0 → 0.5，分數只會變高。

    用手算的數字釘住：預測 0.10 的 1000 人實際流失 30%，預測 0.20 的 1000 人
    實際流失 10%。四種配對加起來 AUC = 220000 / 640000 = 0.34375，遠低於
    亂猜。isotonic 把兩段壓成同一個值之後，全部同分 → 剛好 0.5。

    所以 Mar 上「校準後 AUC 變高」不能讀成模型變強了，只能讀成
    **原本那段排序是錯的，現在誠實地宣告不知道**。
    """
    y, p = _grouped([(0.10, 1000, 0.30), (0.20, 1000, 0.10)])

    cal = fit_isotonic(y, p)

    assert roc_auc(y, p) == pytest.approx(0.34375)
    assert roc_auc(y, cal.apply(p)) == pytest.approx(0.5)


@NODATA
def test_calibration_never_reverses_the_ordering():
    """AUC 的變化只能來自同分 —— 單調映射不會把任何一對的順序倒過來。

    這是上一條測試的另一半：方向可正可負，但成因只有一個。任何「校準把
    某兩個人的相對風險換位」的實作錯誤都會在這裡爆掉。
    """
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p * 0.5).astype(np.float64)

    cal = fit_isotonic(y, p)
    order = np.argsort(p, kind="stable")

    assert np.all(np.diff(cal.apply(p)[order]) >= -1e-12)


@NODATA
def test_auc_gives_tied_predictions_half_credit():
    """AUC 對同分各給一半 —— 這正是它能量出「壓平損失多少排序」的原因。

    全部同分 = 0.5（等於亂猜），完全分開 = 1.0。兩端定錨。
    """
    assert roc_auc([0, 1], [0.1, 0.9]) == pytest.approx(1.0)
    assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)


@NODATA
def test_tie_profile_measures_lost_resolution():
    """同分結構要量三個數字，其中「有同分對象的樣本佔比」最直觀。"""
    profile = tie_profile([0.1, 0.1, 0.1, 0.2, 0.3])

    assert profile["相異值數"] == 3
    assert profile["最大同分組佔比"] == pytest.approx(0.6)
    assert profile["有同分的樣本佔比"] == pytest.approx(0.6)


@NODATA
def test_isotonic_collapses_distinct_predictions_into_few_levels():
    """5000 個相異預測值 fit 完之後，剩下的風險等級應該少得多。

    這不是缺點也不是優點，是必然：PAVA 的輸出就是若干個同值區塊。
    釘住它是因為 SPEC §6.2 要「取前 K%」，而等級數就是那個 K 的解析度上限。
    """
    rng = np.random.default_rng(2)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, 5000) < p).astype(np.float64)

    cal = fit_isotonic(y, p)

    assert len(np.unique(p)) == 5000
    assert cal.n_levels < 500, "校準後的相異等級數應遠少於原始預測值個數"


@NODATA
def test_fit_rejects_single_class():
    """全陰的校準集會 fit 出常數映射，把整份預測抹平 —— 在源頭擋掉。"""
    with pytest.raises(ValueError, match="單一類別"):
        fit_isotonic([0, 0, 0], [0.1, 0.2, 0.3])


@NODATA
def test_fit_rejects_non_binary_labels():
    """標籤若是機率而不是 0/1，isotonic 照樣 fit 得出來，但意義完全不同。"""
    with pytest.raises(ValueError, match="只能是 0 或 1"):
        fit_isotonic([0.0, 0.5, 1.0], [0.1, 0.2, 0.3])


@NODATA
def test_fit_rejects_length_mismatch():
    with pytest.raises(ValueError, match="長度不符"):
        fit_isotonic([0, 1], [0.5])


@NODATA
def test_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        fit_isotonic([], [])


@NODATA
def test_apply_rejects_empty_input():
    y, p = _grouped([(0.1, 100, 0.1), (0.5, 100, 0.5)])
    cal = fit_isotonic(y, p)
    with pytest.raises(ValueError):
        cal.apply([])


@NODATA
def test_knots_are_monotone_and_within_bounds():
    """轉折點表是拿來畫映射曲線的，必須已經套過上下界。"""
    y, p = _grouped([(0.001, 3000, 0.0), (0.20, 1000, 0.20), (0.50, 1000, 0.60)])

    cal = fit_isotonic(y, p)
    knots = cal.knots

    assert knots.height >= 2
    assert knots["校準後"].min() >= cal.floor
    assert knots["校準後"].max() <= cal.ceiling
    assert np.all(np.diff(knots["校準後"].to_numpy()) >= -1e-12)
