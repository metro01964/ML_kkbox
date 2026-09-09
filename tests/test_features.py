"""特徵層測試。

全部使用 conftest.make_synthetic_cohort() 的手刻資料，因此標成 nodata，
在 CI 上也會實際執行。真實資料上的驗證另見 test_no_leakage.py 的 slow 測試。
"""

from __future__ import annotations

import polars as pl
import pytest

from src.features import CATEGORICAL, build_features
from tests.conftest import NODATA, make_synthetic_cohort

# 這些是 build_cohort 的輸出欄位，但**不得**出現在特徵矩陣裡。
# msno 是識別碼；其餘四個是原始日期。
BANNED_COLUMNS = {"msno", "cutoff", "first_tx", "last_tx", "registration_init_time"}


@pytest.fixture
def cohort() -> pl.DataFrame:
    return make_synthetic_cohort()


@NODATA
def test_no_identifier_or_raw_dates(cohort):
    """特徵矩陣不得含 msno 與任何原始日期欄位。

    原始日期為什麼危險：訓練集的 cutoff 全在 2017-02，測試集全在 2017-04，
    兩者沒有交集。模型會學到「cutoff < 20170301」這種在測試集永遠成立或
    永遠不成立的切點 —— 訓練分數漂亮，上線後那個分支形同不存在。
    """
    fs = build_features(cohort)
    leaked = BANNED_COLUMNS & set(fs.X.columns)
    assert not leaked, f"特徵矩陣含有不該出現的欄位：{sorted(leaked)}"


@NODATA
def test_date_difference_is_in_days(cohort):
    """日期相減必須是真正的天數，不是 YYYYMMDD 的十進位差。

    直接相減會得到 20170228 - 20160216 = 10012，那是垃圾但不會報錯，
    模型照樣訓練得起來，只是學到的東西沒有意義。

    ⚠️ 期望值刻意選在跨閏年的區間。2016 是閏年，u0 的 366 + 12 = 378 天裡
    包含 2016-02-29 —— 任何用「一年 365 天」近似的寫法都會少算一天。
    """
    fs = build_features(cohort)
    # 2016-02-16 → 2017-02-16 是 366 天（跨 2016-02-29），再 +12 天到 02-28
    assert fs.X["tenure_days"][0] == 378
    assert fs.X["days_since_last_tx"][1] == 0  # 同一天
    # 2015-01-01 → 2017-01-01：365 + 366（2016 閏年）
    assert fs.X["tenure_days"][4] == 731


@NODATA
def test_negative_duration_becomes_null(cohort):
    """註冊日晚於 cutoff 時，年資須為 null 而非負數。

    u5 的 registration_init_time = 20170330，cutoff = 20170228。實測真實
    資料有 8 位這樣的用戶，數量微不足道但那是未來資訊 —— members_v3 是
    快照，不受 as-of 截斷保護。
    """
    fs = build_features(cohort)
    assert fs.X["days_since_registration"][5] is None
    # 其餘沒問題的列不受影響（同樣跨閏年，378 天）
    assert fs.X["days_since_registration"][0] == 378


@NODATA
def test_zero_plan_days_does_not_produce_inf(cohort):
    """方案天數 0 時 price_per_day 必須是 null，不能是 inf 或 NaN。

    真實資料有 12,861 筆 payment_plan_days = 0。除以 0 在 float 裡不會
    拋例外，會安靜產生 inf，然後在 LightGBM 裡變成無法解釋的分裂點。
    """
    fs = build_features(cohort)
    assert fs.X["price_per_day"][5] is None  # u5 的 plan_days = 0
    for name, dtype in zip(fs.X.columns, fs.X.dtypes, strict=True):
        if dtype.is_float():
            assert not fs.X[name].is_infinite().any(), f"{name} 含有 inf"


@NODATA
def test_zero_paid_is_split_into_two_meanings(cohort):
    """「實付 0 元」要拆成免費方案與零收款兩個旗標。

    真實資料 122 萬筆實付 0 元裡，53.97% 的定價本來就是 0（免費試用），
    其餘是定價非 0 卻沒收到錢。兩者業務意義完全不同（SPEC §2.1）。
    """
    fs = build_features(cohort)
    # u1：定價 0、實付 0 → 免費方案
    assert fs.X["is_free_plan"][1] == 1.0
    assert fs.X["zero_collected"][1] == 0.0
    # u3：定價 149、實付 0 → 零收款
    assert fs.X["is_free_plan"][3] == 0.0
    assert fs.X["zero_collected"][3] == 1.0


@NODATA
def test_bd_outliers_become_null_but_keep_a_flag(cohort):
    """bd 的離群值轉 null，同時保留「原本是否有效」當獨立特徵。

    官方明示此欄含 -7000 ~ 2015 的離群值，cohort 內僅 39.18% 落在合理範圍。
    直接當數值特徵會毀掉模型；而「有沒有填有效年齡」本身有訊號。
    """
    fs = build_features(cohort)
    valid_expected = [0, 1, 0, 0, 1, 0, 1, 0]  # 只有 u1=25、u4=34、u6=45 合理
    assert fs.X["bd_valid"].to_list() == [float(v) for v in valid_expected]
    assert fs.X["bd_clean"][0] is None  # bd = 0
    assert fs.X["bd_clean"][2] is None  # bd = -7168
    assert fs.X["bd_clean"][5] is None  # bd = 2016
    assert fs.X["bd_clean"][1] == 25.0


@NODATA
def test_missing_categoricals_use_negative_sentinel(cohort):
    """類別特徵的缺失要填成負值。

    LightGBM 的約定是「類別特徵的所有負值視為缺失」。填 -1 不是拿 -1 當一個
    普通類別，而是明確告訴 LightGBM 這裡沒有值。實測 11.66% 的 cohort 用戶
    不在 members_v3 中，他們的 city / registered_via / gender 全落在這一類。
    """
    fs = build_features(cohort)
    assert fs.X["city"][3] == -1
    assert fs.X["registered_via"][3] == -1
    assert fs.X["gender_code"][1] == -1  # gender 為 null
    assert fs.X["gender_code"][0] == 0  # male
    assert fs.X["gender_code"][2] == 1  # female
    for col in CATEGORICAL:
        assert fs.X[col].null_count() == 0, f"{col} 仍有 null，LightGBM 會另外處理"


@NODATA
def test_row_order_and_labels_are_preserved(cohort):
    """特徵矩陣的列順序與標籤、msno 必須對得上。

    順序錯位是最難察覺的 bug —— 分數只會稍微變差，不會報錯。
    """
    fs = build_features(cohort)
    assert fs.X.height == cohort.height
    assert fs.y.to_list() == cohort["is_churn"].to_list()
    assert fs.msno.to_list() == cohort["msno"].to_list()


@NODATA
def test_missing_input_column_raises(cohort):
    """輸入缺欄位要明確報錯，不能默默產生少了特徵的矩陣。"""
    with pytest.raises(KeyError, match="缺少欄位"):
        build_features(cohort.drop("bd"))
