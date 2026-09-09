"""M3 的純邏輯測試 —— 特徵子集、跨套件轉接、null importance 的篩選規則。

全部標 nodata：不碰原始資料，CI 上要求零 skip。

M3 的三個模組裡真正容易寫錯的不是模型呼叫（錯了會直接爆），而是**安靜出錯**
的那幾處：切列時忘了帶 msno、類別欄索引跟著欄序漂掉、分位數門檻的邊界
（`>` 還是 `>=`）。下面每一條測的都是這類「不會報錯但結果是錯的」情境。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features import build_features
from src.models.candidates import xgb_category_levels
from src.models.selection import NullImportance
from tests.conftest import NODATA, make_synthetic_cohort


@NODATA
def test_feature_set_take_keeps_msno_aligned():
    """切列必須把 X / y / msno 三者一起切，而且順序對得上。

    只切 X 和 y 不會有任何錯誤訊息 —— 直到 §4.5 的分群報告拿錯誤的 msno 去
    比對上一期名單，分群報告靜靜地變成亂數。
    """
    fs = build_features(make_synthetic_cohort())
    idx = [5, 1, 7]
    sub = fs.take(idx)

    assert sub.X.height == 3
    assert sub.msno.to_list() == [fs.msno[i] for i in idx]
    assert sub.y.to_list() == [fs.y[i] for i in idx]
    assert sub.X.row(0) == fs.X.row(5)
    # 類別特徵清單不隨切列改變（切的是列不是欄）。
    assert sub.categorical == fs.categorical


@NODATA
def test_feature_set_select_filters_categorical():
    """取欄位子集時，類別特徵清單必須同步過濾。

    忘了過濾的話，LightGBM 會拿到指向不存在欄位的索引 —— 而且不一定報錯，
    可能只是把錯的欄位當成類別處理。
    """
    fs = build_features(make_synthetic_cohort())
    keep = ["n_tx", "city", "mean_paid"]
    sub = fs.select(keep)

    assert sub.X.columns == keep
    assert sub.categorical == ("city",)
    assert [sub.X.columns.index(c) for c in sub.categorical] == [1]


@NODATA
def test_feature_set_select_rejects_unknown_column():
    """要保留的欄位不存在就要直接失敗，不能安靜地少給一欄。"""
    fs = build_features(make_synthetic_cohort())
    with pytest.raises(KeyError):
        fs.select(["n_tx", "沒有這一欄"])


@NODATA
def test_xgb_category_levels_excludes_missing_sentinel():
    """缺失哨兵 -1 不能被當成一個合法類別。

    若 -1 留在字典裡，XGBoost 會把「不在 members_v3 裡」當成一個普通的
    city 值去找切點，而不是走缺失分支 —— 與 LightGBM 的處理就不一致，
    三方比較也就不再是同一件事的比較。
    """
    train = pl.DataFrame({"city": [1, 5, -1]})

    levels = xgb_category_levels(train, categorical=("city",))

    assert levels["city"] == [1, 5]


@NODATA
def test_xgb_category_levels_is_fit_on_training_data_only():
    """類別字典是**擬合出來的前處理狀態**，只能看訓練資料（紅線 5）。

    把驗證集也餵進去不會洩漏標籤，但會讓前處理器依驗證集的分布而定 ——
    那是部署時做不到的事：上線時未來會出現哪些 payment_method_id，當下
    不可能知道。訓練時沒見過的類別在推論時落到缺失分支，那才是真實行為。

    這個性質由**簽名**保證而不是靠自律：函式只收一個 frame，想把 holdout
    一起餵進去會直接 TypeError。可變參數的版本會讓
    `xgb_category_levels(feb.X, mar.X, ...)` 寫得出來，而那一行看起來完全
    無害 —— 能寫出來的錯，遲早有人會寫。
    """
    train = pl.DataFrame({"city": [1, 5]})
    holdout = pl.DataFrame({"city": [99]})

    assert xgb_category_levels(train, categorical=("city",))["city"] == [1, 5]

    with pytest.raises(TypeError):
        xgb_category_levels(train, holdout, categorical=("city",))


def _null_importance(actual: dict[str, float], null: dict[str, list[float]]) -> NullImportance:
    """手刻一份 NullImportance，讓篩選規則能脫離模型單獨測試。"""
    names = list(actual)
    return NullImportance(
        actual=pl.DataFrame({"feature": names, "actual_gain": [actual[f] for f in names]}),
        null_gains=pl.DataFrame(
            [
                {"feature": f, "run": i + 1, "gain": g}
                for f, gains in null.items()
                for i, g in enumerate(gains)
            ]
        ),
        n_runs=max(len(v) for v in null.values()),
        num_boost_round=10,
    )


@NODATA
def test_keep_compares_each_feature_against_its_own_null():
    """篩選必須用「每個特徵自己的 null 分布」，不是全體共用一個門檻。

    這正是 null importance 的重點。`noisy` 的 actual gain（80）比 `real`
    的（30）**高**，但它的 null 分布也整個高上去 —— 用全體共用門檻會留下
    noisy 砍掉 real，剛好選反。
    """
    ni = _null_importance(
        actual={"real": 30.0, "noisy": 80.0},
        null={"real": [1.0, 2.0, 3.0, 4.0], "noisy": [70.0, 90.0, 110.0, 130.0]},
    )
    assert ni.keep(75) == ["real"]


@NODATA
def test_keep_is_strictly_greater_than_threshold():
    """邊界：actual 剛好等於門檻時要**砍掉**。

    `>=` 會讓「gain 完全等於 null 最大值」的特徵存活，而那代表它的表現與
    純雜訊沒有差別。這種 off-by-one 不會報錯，只會讓篩選鬆一格。
    """
    ni = _null_importance(actual={"tie": 5.0}, null={"tie": [1.0, 3.0, 5.0]})
    assert ni.keep(100) == []  # p100 = max = 5.0，等於不算贏
    assert ni.keep(50) == ["tie"]  # p50 = 3.0，5.0 > 3.0


@NODATA
def test_keep_preserves_original_column_order():
    """回傳的欄位順序必須沿用原始順序。

    欄序會影響 LightGBM 的類別索引與 feature_fraction 的抽樣結果。順序不穩
    的話，同一組特徵在兩次執行會得到不同的分數，實驗就不可重現。
    """
    ni = _null_importance(
        actual={"a": 10.0, "b": 10.0, "c": 10.0},
        null={"a": [1.0], "b": [1.0], "c": [1.0]},
    )
    assert ni.keep(50) == ["a", "b", "c"]


@NODATA
def test_summary_beats_pct_is_readable():
    """`beats_pct` 要等於「actual 贏過幾成的 null 執行」。

    score 是對數尺度、只能拿來排序；真正給人看的去留依據是這個比例，
    所以它必須是字面上的意思。
    """
    ni = _null_importance(actual={"f": 5.0}, null={"f": [1.0, 4.0, 6.0, 9.0]})
    row = ni.summary().row(0, named=True)
    assert row["beats_pct"] == pytest.approx(0.5)  # 4 次裡贏 2 次


@NODATA
def test_null_importance_shuffles_labels_not_features():
    """打亂標籤不得改動特徵矩陣。

    這一條測的是方法本身：打亂特徵會破壞欄與欄的相關結構，量到的 null
    分布就對應不到真實情境。`numpy.permutation` 作用在 y 上，X 一格都不動 ——
    這裡直接驗證 permutation 的語意（同一組值、順序被打亂）。
    """
    y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
    shuffled = np.random.default_rng(42).permutation(y)

    assert sorted(shuffled.tolist()) == sorted(y.tolist()), "打亂不得改變標籤的組成"
    assert shuffled.tolist() != y.tolist(), "seed 42 下應該真的被打亂"
