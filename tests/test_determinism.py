"""可重現性的迴歸測試 —— 防止 §7.8 那個 bug 再回來。

那個 bug 的形狀值得記住，因為它不像 bug：

    build_cohort 的列順序每次重建都不同（polars 的 group_by 不保證順序）
      → train_test_split 依**位置**切分
      → 訓練集換一批人
      → Mar log loss 在 0.15853 ~ 0.15910 之間跳動

沒有任何東西會失敗。分數只是動了 0.0006，而那和我們想量的特徵效果同一個
量級 —— 於是「這個特徵有沒有用」量到的其實是重建快取的運氣。

因此這一份釘住兩件事，對應兩個獨立的失效點：

1. **順序本身**（`assert_rows_reproducible`）—— 含快取命中的路徑。
   修正之前產出的舊快取沒有洩漏，兩條既有守門都會通過，只有這條擋得住。
2. **切分不綁在列位置上**（`three_way_split` / `train_baseline`）——
   即使有人繞過了第 1 點，打亂順序也不該換掉任何一個人。

第 2 點的測試作法是**故意打亂**再切一次：這是 SPEC §5 對守門的要求 ——
只驗證「正常資料會通過」證明不了任何事，必須餵給它一份確實違規的輸入。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.data import assert_rows_reproducible
from src.features import FeatureSet
from src.models.tuning import three_way_split
from tests.conftest import NODATA

SPLIT_CFG = {"select_fraction": 0.15, "early_stopping_fraction": 0.176, "split_seed": 42}


def _cohort(msno: list[str]) -> pl.DataFrame:
    """最小的 cohort 表，只帶守門會看的欄位。"""
    return pl.DataFrame({"msno": msno, "cutoff": [20170228] * len(msno)})


def _feature_set(n: int = 2000, *, seed: int = 0) -> FeatureSet:
    """合成的 FeatureSet，msno 依字典序排好 —— 與 build_cohort 的產出一致。"""
    rng = np.random.default_rng(seed)
    msno = [f"user_{i:06d}" for i in range(n)]
    y = (rng.uniform(size=n) < 0.0639).astype(np.int8)  # 流失率貼近實測
    return FeatureSet(
        X=pl.DataFrame({"f0": rng.normal(size=n), "f1": rng.normal(size=n)}),
        y=pl.Series("is_churn", y),
        msno=pl.Series("msno", msno),
        categorical=(),
    )


@NODATA
def test_sorted_cohort_passes_the_guard():
    """正常的表要通過。單獨這一條證明不了守門有效 —— 見下面兩條違規案例。"""
    assert_rows_reproducible(_cohort(["a", "b", "c"]))


@NODATA
def test_unsorted_cohort_is_rejected():
    """未排序的表必須被擋下 —— 這就是修正之前的舊快取長的樣子。

    ⚠️ 它沒有洩漏：`assert_asof_respected` 與 `assert_cutoffs_within_window`
    都會通過。只有這條守門看得出它過時。
    """
    with pytest.raises(AssertionError, match="列順序不可重現"):
        assert_rows_reproducible(_cohort(["b", "a", "c"]))


@NODATA
def test_duplicate_msno_is_rejected():
    """同一個人兩列 → 他會同時出現在訓練與驗證集。"""
    with pytest.raises(AssertionError, match="msno 有重複"):
        assert_rows_reproducible(_cohort(["a", "a", "b"]))


@NODATA
def test_guard_requires_the_msno_column():
    with pytest.raises(KeyError):
        assert_rows_reproducible(pl.DataFrame({"cutoff": [1, 2]}))


@NODATA
def test_split_is_identical_when_rows_are_shuffled():
    """**核心迴歸測試**：打亂列順序，三段切分的成員必須一模一樣。

    這條若失敗，代表切分又綁回列位置上了 —— 而列順序來自另一個模組的
    `.sort("msno")`，一次 filter、一次 join、或一份舊快取就能讓它失效。
    """
    fs = _feature_set()
    rng = np.random.default_rng(7)
    shuffled = fs.take(rng.permutation(fs.X.height))

    a = three_way_split(fs, SPLIT_CFG)
    b = three_way_split(shuffled, SPLIT_CFG)

    for part_a, part_b, name in (
        (a.train, b.train, "train"),
        (a.es, b.es, "es"),
        (a.sel, b.sel, "sel"),
    ):
        assert sorted(part_a.msno.to_list()) == sorted(part_b.msno.to_list()), (
            f"{name} 的成員因列順序而改變"
        )


@NODATA
def test_split_members_do_not_overlap_and_cover_everyone():
    """三段互斥且完全覆蓋 —— 少了誰或誰重複出現都不會有錯誤訊息。"""
    fs = _feature_set()
    split = three_way_split(fs, SPLIT_CFG)

    train, es, sel = (set(p.msno.to_list()) for p in (split.train, split.es, split.sel))

    assert train & es == set()
    assert train & sel == set()
    assert es & sel == set()
    assert train | es | sel == set(fs.msno.to_list())


@NODATA
def test_same_seed_reproduces_the_same_split():
    """同 seed 兩次呼叫必須完全相同，包含順序。"""
    fs = _feature_set()
    a = three_way_split(fs, SPLIT_CFG)
    b = three_way_split(fs, SPLIT_CFG)

    assert a.train.msno.to_list() == b.train.msno.to_list()
    assert a.sel.msno.to_list() == b.sel.msno.to_list()


@NODATA
def test_different_seed_gives_a_different_split():
    """不同 seed 必須真的切出不同的人 —— 否則 configs/calibration.yaml 另立
    seed 以降低重疊的那整套理由就是空的。"""
    fs = _feature_set()
    a = three_way_split(fs, SPLIT_CFG)
    b = three_way_split(fs, {**SPLIT_CFG, "split_seed": 20260809})

    overlap = set(a.sel.msno.to_list()) & set(b.sel.msno.to_list())
    assert overlap != set(a.sel.msno.to_list()), "換 seed 應該換一批人"
    # 期望重疊約等於 sel 的比例（15%），給寬鬆的區間避免變成脆弱測試。
    assert 0.05 < len(overlap) / a.sel.X.height < 0.30


@NODATA
def test_split_keeps_the_churn_rate_in_every_part():
    """分層抽樣：三段的流失率都要貼近整體，否則選參數就是在比運氣。"""
    fs = _feature_set(n=20000)
    split = three_way_split(fs, SPLIT_CFG)
    overall = float(fs.y.mean())

    for part, name in ((split.train, "train"), (split.es, "es"), (split.sel, "sel")):
        assert float(part.y.mean()) == pytest.approx(overall, abs=0.005), f"{name} 的流失率偏離"
