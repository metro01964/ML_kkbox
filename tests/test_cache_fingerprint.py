"""快取的**程式版本指紋** —— §7.5 遺留至今的待辦。

## 這條守門補的洞

`build_cohort()` 與 `build_log_features()` 的快取命中路徑目前只檢查：

    欄位是不是 EXPECTED_COLUMNS 的超集
    紅線 1（last_tx <= cutoff）
    cohort 錯置
    列順序可重現

**四條全部通過，不代表這份快取是現在這版程式算出來的。** 改一個平滑常數、
改一條 filter、改 `_last_unambiguous` 的規則 —— 欄位一個都沒變，四條檢查
全綠，而快取裡是舊邏輯的產物。分數會變，但沒有任何東西會提醒你。

這正是 §7.11 那次事故的形狀：修好了程式，卻在舊快取上跑實驗。

## 指紋怎麼算

拿產生該快取的模組原始碼，**剝掉 docstring、忽略註解**，正規化成 AST 之後
雜湊。這個選擇是刻意的：

    邏輯改了（常數、條件、聚合式）→ AST 變 → 指紋變 → 自動重建
    只改註解或 docstring          → AST 不變 → 指紋不變 → 沿用快取

本專案的 docstring 佔了程式碼的一大半且經常改寫。若連它們一起雜湊，每次
補一句說明就要重掃 1.7 GB 交易檔 —— 那種守門會在第三天被人關掉。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.fingerprint import (
    FINGERPRINT_KEY,
    cache_is_current,
    logic_fingerprint,
    normalize_source,
    read_cache_fingerprint,
    write_with_fingerprint,
)
from tests.conftest import NODATA

BASE = '''
"""模組 docstring。"""
SMOOTHING = 100.0


def aggregate(x):
    """函式 docstring。"""
    # 一行註解
    return x * SMOOTHING
'''

ONLY_COMMENTS_CHANGED = '''
"""完全不同的模組 docstring，寫了三倍長的說明。"""
SMOOTHING = 100.0


def aggregate(x):
    """改寫過的函式 docstring。"""
    # 換一句註解，並且再加一行
    # 第二行註解
    return x * SMOOTHING
'''

LOGIC_CHANGED = '''
"""模組 docstring。"""
SMOOTHING = 50.0


def aggregate(x):
    """函式 docstring。"""
    # 一行註解
    return x * SMOOTHING
'''

FILTER_CHANGED = '''
"""模組 docstring。"""
SMOOTHING = 100.0


def aggregate(x):
    """函式 docstring。"""
    # 一行註解
    return x * SMOOTHING if x > 0 else 0
'''


@NODATA
def test_docstring_and_comment_changes_do_not_change_the_fingerprint():
    """只改說明文字不該觸發重建 —— 否則這條守門三天內就會被關掉。"""
    assert normalize_source(BASE) == normalize_source(ONLY_COMMENTS_CHANGED)
    assert logic_fingerprint(BASE) == logic_fingerprint(ONLY_COMMENTS_CHANGED)


@NODATA
def test_changing_a_constant_changes_the_fingerprint():
    """平滑常數 100 → 50：欄位一個都沒變，指紋必須變。"""
    assert logic_fingerprint(BASE) != logic_fingerprint(LOGIC_CHANGED)


@NODATA
def test_changing_a_condition_changes_the_fingerprint():
    """加一條 filter：同樣是「欄位不變、邏輯變了」。"""
    assert logic_fingerprint(BASE) != logic_fingerprint(FILTER_CHANGED)


@NODATA
def test_fingerprint_survives_a_parquet_roundtrip(tmp_path):
    path = tmp_path / "cache.parquet"
    df = pl.DataFrame({"msno": ["a", "b"], "v": [1, 2]})

    write_with_fingerprint(df, path, "deadbeef")

    assert read_cache_fingerprint(path) == "deadbeef"
    assert pl.read_parquet(path).equals(df), "指紋不得改動資料本身"
    assert FINGERPRINT_KEY in pl.read_parquet_metadata(path)


@NODATA
def test_stale_cache_is_detected(tmp_path):
    """**核心迴歸測試**：欄位相同、邏輯已改的快取必須被判為過時。

    這是 §7.5 遺留待辦的具體形態 —— 修正前的實作只比對欄位名稱，
    這種快取會完全通過。
    """
    path = tmp_path / "cache.parquet"
    write_with_fingerprint(pl.DataFrame({"msno": ["a"]}), path, logic_fingerprint(BASE))

    assert cache_is_current(path, logic_fingerprint(BASE))
    assert not cache_is_current(path, logic_fingerprint(LOGIC_CHANGED))


@NODATA
def test_cache_without_a_fingerprint_is_stale(tmp_path):
    """修正之前產生的快取沒有指紋 —— 一律視為過時，強制重建一次。

    「沒有指紋」與「指紋不符」要一視同仁：前者代表我們無從得知它是哪版
    程式算的，那跟知道它是舊版一樣不能用。
    """
    path = tmp_path / "legacy.parquet"
    pl.DataFrame({"msno": ["a"]}).write_parquet(path)

    assert read_cache_fingerprint(path) is None
    assert not cache_is_current(path, logic_fingerprint(BASE))


@NODATA
def test_missing_file_is_not_current(tmp_path):
    assert not cache_is_current(tmp_path / "nope.parquet", "whatever")


@NODATA
def test_fingerprint_accepts_modules_and_is_stable():
    """對真實模組取指紋必須成功、且同一次執行內穩定。"""
    from src.data import cohort
    from src.features import logs

    a = logic_fingerprint(cohort, logs)
    b = logic_fingerprint(cohort, logs)

    assert a == b
    assert len(a) == 16, "指紋長度固定，方便寫進 parquet metadata 與 log"
    # 模組順序不同就是不同的組合，不該互相覆蓋
    assert logic_fingerprint(cohort) != logic_fingerprint(cohort, logs)


@NODATA
def test_syntax_error_is_reported_not_swallowed():
    with pytest.raises(SyntaxError):
        logic_fingerprint("def broken(:\n    pass\n")
