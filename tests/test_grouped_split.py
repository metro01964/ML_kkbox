"""M6 合併 cohort 切分的純邏輯測試 —— 不需資料，CI 要求零 skip。

`test_no_leakage.py` 測的是紅線 4 本身（守門會不會擋、合規的折法有沒有人跨邊）。
這一份測的是**周邊那些會安靜出錯的地方**：

  - 合併時欄序／dtype／類別宣告不一致 —— 直向 concat 是依位置對齊的，接錯了
    不會有人抱怨，只會讓 city 的值進到 registered_via 那一欄。
  - 佔全體的比例 vs 佔剩下的比例 —— 算錯只是某一段大小不對，沒有錯誤訊息。
  - 切分依不依賴列順序 —— `three_way_split` 的模組註解宣稱不依賴，那句話要有
    測試撐著，否則哪天有人把排序拿掉，症狀是「分數動了 0.0006，看起來像實驗
    有了效果」。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.models.grouped import (
    SEGMENTS,
    MergedCohorts,
    assert_groups_disjoint,
    fold_report,
    four_way_group_split,
    group_split,
    group_strata,
    grouped_folds,
    merge_cohorts,
    random_folds_violating_red_line_4,
)
from tests.conftest import NODATA, make_two_cohorts

SPLIT_CFG = {
    "fractions": {"train": 0.7, "es": 0.1, "sel": 0.1, "cal": 0.1},
    "seed": 20260810,
}


# ===========================================================================
# merge_cohorts —— 四種會安靜出錯的合併
# ===========================================================================


@NODATA
def test_merge_counts_shared_users():
    """合併後要說得出「幾個人跨期出現」—— 那是紅線 4 的曝險面大小。"""
    merged = merge_cohorts(make_two_cohorts(n_shared=150, n_only_a=50, n_only_b=50))

    assert merged.n_rows == 400
    assert merged.n_groups == 250
    assert merged.n_shared == 150
    assert merged.parts == ("feb", "mar")


@NODATA
def test_merge_keeps_cohort_out_of_the_feature_matrix():
    """出身欄不得混進 X。

    Apr cohort（要預測的那一批）沒有這個值。一旦它進了特徵，模型會學到一個
    部署時算不出來的東西，而訓練與驗證都不會有任何異狀。
    """
    parts = make_two_cohorts()
    merged = merge_cohorts(parts)

    assert "cohort" not in merged.fs.X.columns
    assert merged.cohort.len() == merged.n_rows
    assert merged.summary()["列數"].sum() == merged.n_rows


@NODATA
def test_merge_rejects_different_column_order():
    """欄序不同就拒絕 —— 直向 concat 是依位置對齊的。"""
    parts = make_two_cohorts()
    mar = parts["mar"]
    parts["mar"] = mar.select(list(reversed(mar.X.columns)))

    with pytest.raises(ValueError, match="欄位或順序不同"):
        merge_cohorts(parts)


@NODATA
def test_merge_rejects_different_dtype():
    """dtype 不同就拒絕 —— polars 會自動 upcast，類別欄會悄悄變型別。"""
    parts = make_two_cohorts()
    mar = parts["mar"]
    parts["mar"] = mar.__class__(
        X=mar.X.with_columns(pl.col("n_tx").cast(pl.Float64)),
        y=mar.y,
        msno=mar.msno,
        categorical=mar.categorical,
    )

    with pytest.raises(ValueError, match="dtype 不同"):
        merge_cohorts(parts)


@NODATA
def test_merge_rejects_placeholder_labels():
    """標籤是 null 的 cohort 不得併進訓練資料。

    Kaggle 測試集（apr_fixed）的 is_churn 就是這種。多數套件會把 null 當 0，
    也就是「這 90 萬人全部沒流失」—— 一個看起來完全正常的訓練過程。
    """
    parts = make_two_cohorts()
    mar = parts["mar"]
    parts["mar"] = mar.__class__(
        X=mar.X, y=mar.y.scatter(0, None), msno=mar.msno, categorical=mar.categorical
    )

    with pytest.raises(ValueError, match="標籤含"):
        merge_cohorts(parts)


@NODATA
def test_merge_needs_at_least_two_cohorts():
    """一個 cohort 沒有東西可以合併，也就沒有紅線 4 的問題 —— 直接拒絕比較誠實。"""
    with pytest.raises(ValueError, match="至少要兩個"):
        merge_cohorts({"feb": make_two_cohorts()["feb"]})


# ===========================================================================
# 四段切分
# ===========================================================================


@NODATA
def test_four_way_split_sizes_follow_the_declared_fractions():
    """設定檔寫的是佔**全體**的比例，切出來就要接近那個比例。

    容差放在人數上（±1.5%）：切的是人不是列，而每個人帶 1 或 2 列，
    所以列數的比例不會剛好等於宣告值 —— 這正是「群組切分」與「切列」的差別。
    """
    merged = merge_cohorts(make_two_cohorts())
    split = four_way_group_split(merged, SPLIT_CFG)

    for name in SEGMENTS:
        share = split.segment(name).msno.n_unique() / merged.n_groups
        assert abs(share - SPLIT_CFG["fractions"][name]) < 0.015, f"{name} 的人數比例偏掉了"


@NODATA
def test_four_way_split_preserves_churn_rate():
    """分層要生效：四段的流失率不能差太多。

    不分層的話，10% 的小塊裡正例數會有可觀波動，而 sel 段上「校準器有沒有
    幫助」的判斷就開始比運氣。
    """
    merged = merge_cohorts(make_two_cohorts())
    split = four_way_group_split(merged, SPLIT_CFG)

    overall = float(merged.fs.y.mean())
    for name in SEGMENTS:
        assert abs(float(split.segment(name).y.mean()) - overall) < 0.05


@NODATA
def test_four_way_split_does_not_depend_on_row_order():
    """把列打亂再切，每一段的人必須完全相同。

    `train_test_split` 依**位置**切：同一個 seed 餵進不同順序的資料會切出
    不同的人。合併是最容易改變列順序的一步（誰先誰後、有沒有排序），所以
    這裡的切分先按 msno 排出標準順序。這條測試就是那句話的憑據。
    """
    merged = merge_cohorts(make_two_cohorts())
    perm = np.random.default_rng(0).permutation(merged.n_rows)
    shuffled = MergedCohorts(
        fs=merged.fs.take(perm),
        cohort=merged.cohort[pl.Series(perm)],
        parts=merged.parts,
        n_shared=merged.n_shared,
    )

    a = four_way_group_split(merged, SPLIT_CFG)
    b = four_way_group_split(shuffled, SPLIT_CFG)
    for name in SEGMENTS:
        assert set(a.segment(name).msno) == set(b.segment(name).msno), f"{name} 換了一批人"


@NODATA
def test_four_way_split_rejects_fractions_that_do_not_sum_to_one():
    """四段加起來不是 1 就拒絕。

    少掉的那些人不會消失，他們會留在 train —— 於是 train 比宣告的大，
    而報表照樣印得出來。這種錯誤只有斷言擋得住。
    """
    merged = merge_cohorts(make_two_cohorts())
    bad = {"fractions": {"train": 0.7, "es": 0.1, "sel": 0.1, "cal": 0.2}, "seed": 1}

    with pytest.raises(ValueError, match="不是 1"):
        four_way_group_split(merged, bad)


@NODATA
def test_four_way_split_rejects_unknown_segment():
    """段名打錯要報錯，不能默默忽略。"""
    merged = merge_cohorts(make_two_cohorts())
    bad = {"fractions": {"train": 0.7, "es": 0.1, "sel": 0.1, "calib": 0.1}, "seed": 1}

    with pytest.raises(KeyError):
        four_way_group_split(merged, bad)


@NODATA
def test_group_split_keeps_users_whole():
    """兩塊切分也不能讓人跨邊 —— fold 內部那一小塊 early stopping 就是這一種。

    早停集同樣是「模型看得到的資料」。同一個人一半在訓練、一半在早停，
    停點就是看著自己選的，而那不會有任何跡象。
    """
    merged = merge_cohorts(make_two_cohorts())
    rest, take = group_split(merged.fs.msno, merged.fs.y, test_size=0.2, seed=7)

    assert len(rest) + len(take) == merged.n_rows
    assert_groups_disjoint(
        {
            "rest": merged.fs.msno[pl.Series(rest)],
            "take": merged.fs.msno[pl.Series(take)],
        }
    )


@NODATA
def test_group_strata_encodes_both_composition_and_label():
    """分層標籤要同時綁住「跨不跨期」與「流失幾次」。

    §4.5：重複用戶與新進用戶的流失率差 6.8 倍。只分層流失率而不分層組成，
    一塊裡新進用戶多幾個百分點，分數就跟著動 —— 那是切分的運氣。
    """
    merged = merge_cohorts(make_two_cohorts())
    groups, strata = group_strata(merged.fs.msno, merged.fs.y)

    assert groups.len() == merged.n_groups
    assert set(strata) == {"1-0", "1-1", "2-0", "2-1", "2-2"}
    # 「2-」開頭的就是跨期用戶，數量要對得上 n_shared
    assert sum(1 for s in strata if s.startswith("2-")) == merged.n_shared


# ===========================================================================
# KFold：合規與違規只差一個參數
# ===========================================================================


@NODATA
def test_grouped_and_random_folds_differ_only_in_crossing():
    """兩種折法的折數與涵蓋範圍相同，差別只在跨邊人數。

    這是紅線 4 要量的東西：**兩邊的程式碼與輸出形狀完全一樣**，只有一個
    看不見的性質不同。所以判斷不能靠讀程式碼，只能靠守門。
    """
    merged = merge_cohorts(make_two_cohorts())

    ok = fold_report(merged, grouped_folds(merged, n_splits=4, seed=0))
    bad = fold_report(merged, random_folds_violating_red_line_4(merged, n_splits=4, seed=0))

    assert ok.height == bad.height == 4
    assert ok["驗證列數"].sum() == bad["驗證列數"].sum() == merged.n_rows
    assert ok["跨邊人數"].sum() == 0
    assert bad["跨邊人數"].sum() > 0, "違規的折法沒有製造出跨邊，這個對照組沒有意義"


@NODATA
def test_guard_needs_at_least_two_segments():
    """只給一段時沒有東西可檢查 —— 要報錯，不能回傳「通過」。

    一個永遠通過的守門比沒有守門更糟：它會讓人以為已經檢查過了。
    """
    with pytest.raises(ValueError, match="至少要兩段"):
        assert_groups_disjoint({"train": pl.Series(["a", "b"])})
