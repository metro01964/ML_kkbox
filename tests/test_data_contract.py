"""資料契約測試 —— SPEC §2.3 的斷言清單。

「契約」的意思是：這份資料**應該**長什麼樣，寫成程式自動檢查。

為什麼容忍度是 0：這是靜態的歷史資料集，2017 年就凍結了。筆數變了不可能是
「資料更新」，只可能是下載出錯、解壓不完整、或檔案被改動。允許誤差等於
允許錯誤悄悄通過。

數字來源：SPEC §2.1 / §2.3（主要電腦 2026-08-06 實測），本測試在備用電腦
上獨立重驗一次。兩台機器量到相同數字，才算真的驗證過。
"""

from __future__ import annotations

import polars as pl
import pytest

from tests.conftest import SLOW

# ---------------------------------------------------------------------------
# SPEC §2.1 的欄位定義。順序照 CSV 的實際順序。
# ---------------------------------------------------------------------------
SCHEMAS: dict[str, list[str]] = {
    "train.csv": ["msno", "is_churn"],
    "train_v2.csv": ["msno", "is_churn"],
    "transactions.csv": [
        "msno",
        "payment_method_id",
        "payment_plan_days",
        "plan_list_price",
        "actual_amount_paid",
        "is_auto_renew",
        "transaction_date",
        "membership_expire_date",
        "is_cancel",
    ],
    "members_v3.csv": [
        "msno",
        "city",
        "bd",
        "gender",
        "registered_via",
        "registration_init_time",
    ],
    "user_logs.csv": [
        "msno",
        "date",
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ],
}
SCHEMAS["transactions_v2.csv"] = SCHEMAS["transactions.csv"]
SCHEMAS["user_logs_v2.csv"] = SCHEMAS["user_logs.csv"]

# SPEC §2.3 斷言 2~6，加上 §2.0 的 user_logs（原本標「待測」，本機實測補上）。
ROW_COUNTS: dict[str, int] = {
    "train.csv": 992_931,
    "train_v2.csv": 970_960,
    "transactions.csv": 21_547_746,
    "transactions_v2.csv": 1_431_009,
    "members_v3.csv": 6_769_473,
    "user_logs.csv": 392_106_543,
    "user_logs_v2.csv": 18_396_362,
}

# SPEC §2.1 的標籤組成。
CHURN_COUNTS: dict[str, tuple[int, int, float]] = {
    # 檔名: (流失數, 續訂數, 流失率百分比)
    "train.csv": (63_471, 929_460, 6.3923),
    "train_v2.csv": (87_330, 883_630, 8.9942),
}


def _rows(lf: pl.LazyFrame) -> int:
    """算列數。用 streaming 才不會把 1.73 GB 讀進記憶體。"""
    return int(lf.select(pl.len()).collect(engine="streaming").item())


# ---------------------------------------------------------------------------
# 斷言 1：欄位名稱
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", sorted(SCHEMAS))
def test_columns_match_spec(raw_files, filename: str):
    """欄位漂移直接失敗（SPEC §2.3 斷言 1）。

    只讀 header，不讀資料，所以就算是 28 GB 的 user_logs 也是瞬間完成。
    """
    schema = raw_files(filename).collect_schema()
    assert schema.names() == SCHEMAS[filename], (
        f"{filename} 的欄位與 SPEC §2.1 不符。\n  實際 {schema.names()}\n  預期 {SCHEMAS[filename]}"
    )


@pytest.mark.parametrize("filename", ["train.csv", "train_v2.csv"])
def test_label_dtypes(raw_files, filename: str):
    """is_churn 必須是整數型別，msno 必須是字串。"""
    schema = raw_files(filename).collect_schema()
    assert schema["is_churn"].is_integer(), f"{filename} 的 is_churn 不是整數"
    assert schema["msno"] == pl.String, f"{filename} 的 msno 不是字串"


# ---------------------------------------------------------------------------
# 斷言 2~6：筆數，容忍度 0
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", ["train.csv", "train_v2.csv", "transactions_v2.csv"])
def test_row_counts_small(raw_files, filename: str):
    """小檔的筆數。這幾個掃很快，不標 slow。"""
    assert _rows(raw_files(filename)) == ROW_COUNTS[filename]


@SLOW
@pytest.mark.parametrize("filename", ["transactions.csv", "members_v3.csv"])
def test_row_counts_large(raw_files, filename: str):
    """中型檔的筆數（1.73 GB + 428 MB）。"""
    assert _rows(raw_files(filename)) == ROW_COUNTS[filename]


@SLOW
@pytest.mark.parametrize("filename", ["user_logs.csv", "user_logs_v2.csv"])
def test_row_counts_user_logs(raw_files, filename: str):
    """user_logs 的筆數。

    SPEC §2.0 這兩格原本標「待測」，本機實測補上：
        user_logs.csv     392,106,543 列
        user_logs_v2.csv   18,396,362 列
    掃 28 GB 約 27 秒（只解析 header 之外的最小欄位集）。
    """
    assert _rows(raw_files(filename)) == ROW_COUNTS[filename]


# ---------------------------------------------------------------------------
# 斷言 7：msno 無重複
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", ["train.csv", "train_v2.csv"])
def test_no_duplicate_msno(raw_files, filename: str):
    """一位用戶在一個 cohort 裡只能出現一次（SPEC §2.3 斷言 7）。

    有重複的話，訓練時同一個人會被計算兩次，等於偷偷加權。
    """
    df = raw_files(filename).select("msno").collect(engine="streaming")
    assert df.height - df["msno"].n_unique() == 0


# ---------------------------------------------------------------------------
# 斷言 8：is_churn 取值
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", ["train.csv", "train_v2.csv"])
def test_is_churn_is_binary(raw_files, filename: str):
    """is_churn 恰為 {0, 1}，不能有 null 或其他值（SPEC §2.3 斷言 8）。"""
    values = set(
        raw_files(filename).select("is_churn").collect(engine="streaming")["is_churn"].unique()
    )
    assert values == {0, 1}, f"{filename} 的 is_churn 取值為 {sorted(values)}"


# ---------------------------------------------------------------------------
# 斷言 9：紅線 3 的守門 —— members_v3 不含 expiration_date
# ---------------------------------------------------------------------------
def test_members_v3_has_no_expiration_date(raw_files):
    """members_v3 不得含 expiration_date（SPEC §2.3 斷言 9、紅線 3）。

    官方在 2017-11-13 發布 v3 就是為了移除這個快照欄位。它記錄的是
    「資料匯出當下」的到期日，對 2 月的樣本而言那是未來資訊。
    """
    names = raw_files("members_v3.csv").collect_schema().names()
    assert "expiration_date" not in names, (
        "members_v3.csv 含有 expiration_date —— 這是洩漏欄位，代表下載到的可能是舊版 members.csv。"
    )


# ---------------------------------------------------------------------------
# 斷言 10：兩期 msno 交集
# ---------------------------------------------------------------------------
def test_cohort_overlap(raw_files):
    """兩期交集 881,701 人（SPEC §2.3 斷言 10、§4.5）。

    這個數字是 SPEC §4.4 的核心：驗證集有 90.81% 的用戶模型訓練時看過。
    它不是傳統洩漏，但會讓驗證分數偏樂觀，也是紅線 4 存在的理由。
    """
    feb = raw_files("train.csv").select("msno").collect(engine="streaming")
    mar = raw_files("train_v2.csv").select("msno").collect(engine="streaming")
    overlap = mar.join(feb, on="msno", how="semi").height
    assert overlap == 881_701


# ---------------------------------------------------------------------------
# 斷言 11：流失率
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename", sorted(CHURN_COUNTS))
def test_churn_composition(raw_files, filename: str):
    """流失/續訂人數與流失率（SPEC §2.3 斷言 11）。

    斷言的是整數人數而不只是比率 —— 比率是浮點數，要挑容忍度；
    人數是整數，可以要求完全相等。
    """
    expected_churn, expected_stay, expected_pct = CHURN_COUNTS[filename]
    s = raw_files(filename).select("is_churn").collect(engine="streaming")["is_churn"]

    churn = int(s.sum())
    stay = s.len() - churn
    assert churn == expected_churn
    assert stay == expected_stay
    assert round(churn / s.len() * 100, 4) == expected_pct


# ---------------------------------------------------------------------------
# 斷言 12（新增）：members_v3 對 cohort 的覆蓋率
# ---------------------------------------------------------------------------
@SLOW
def test_members_coverage_of_feb_cohort(raw_files):
    """members_v3 查不到 115,770 位 Feb cohort 用戶（11.66%）。

    SPEC §2.1 目前寫「members 涵蓋 677 萬用戶，遠多於任一 cohort 的 99 萬，
    join 後不會有大量缺失」—— **實測不成立**，所以加這條斷言把真實情況鎖住。

    這不只是數字問題：那 11.66% 的用戶 city / bd / gender / registered_via
    全部是缺失，前處理不能假設 join 完就有值。而且「查不到」本身有訊號，
    那群人流失率 5.02%，低於整體 6.39%。
    """
    feb = raw_files("train.csv").select("msno").collect(engine="streaming")
    members = raw_files("members_v3.csv").select("msno").collect(engine="streaming")

    missing = feb.join(members, on="msno", how="anti").height
    assert missing == 115_770
    assert round(missing / feb.height * 100, 2) == 11.66
