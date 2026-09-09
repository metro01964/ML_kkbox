"""官方標籤產生器 `WSDMChurnLabeller.scala` 的 Polars 移植。

SPEC §2.1 把這支 Scala 列為「M3 進階選項」，用途是**自行建立更多歷史 cohort
的訓練標籤**。官方只發布了兩個月的標籤（`train.csv` = 2017-02 到期、
`train_v2.csv` = 2017-03 到期），而只有兩個月就只能做一次時間外驗證 ——
分數是好是壞，分不清是模型的本事還是那一個月剛好合拍。

## 為什麼是移植而不是直接跑官方程式

原程式依賴 Spark，本機沒有 Java / Scala / Spark。與其為了跑一次而裝一整套
JVM 生態，不如移植 —— 而且移植有一個官方程式沒有的好處：**可以驗證**。
`train.csv` 與 `train_v2.csv` 就是官方跑出來的答案，把它們當成迴歸測試，
對得上才代表這份移植可信。對不上就不能拿去生新的月份。

## ⚠️ 驗證結果：這份移植**還原不了官方標籤**（2026-08-08 實測）

把移植結果拿去對 `train.csv` / `train_v2.csv`，三種設定都對不上：

| 設定 | 候選人數（官方 992,931） | 標籤一致率 | 我方流失率（官方 6.39%） |
|---|---|---|---|
| 完全照抄（history = 1 月） | 856,144 | 98.77% | 3.33% |
| history 放寬到全歷史 | 879,537 | 98.36% | — |
| 名單用官方、只測標籤規則 | 990,836 | 97.91% | 5.11% |

Mar cohort 更差，一致率只有 **95.09%**。

**兩個具體的不一致**：

1. **候選名單差 11%。** 官方 Feb 名單裡有 111,358 人，在 1/31 當下的到期日
   並不落在 2 月 —— 其中 **85,398 人落在 3 月**。照這支 Scala 的定義，他們
   根本不該進 Feb cohort。
2. **流失率系統性偏低**（4.9~5.1% vs 官方 6.39%）。方向一致代表我們找到了
   官方認定不存在的續約，不是隨機噪音。

**結論：官方發布的 `train.csv` 不是由這支 sample Scala 產生的**，或它讀的
`wsdm_transactions_20170331.csv`（路徑裡有 `wsdm_fix`）與我們手上的
`transactions.csv` + `transactions_v2.csv` 不是同一份資料。

**因此這個模組目前不可用於產生新月份的標籤** —— 用它生出來的 2016-12 /
2017-01 標籤，與官方的 2017-02 / 2017-03 不是同一個定義，跨月穩定性比較
會變成拿不同尺規量不同的東西。保留程式碼與這份記錄，是為了讓下一個人不必
重做這五次實驗。

## 官方定義（逐條對應原始碼）

原程式的邏輯不是「到期後 30 天內有沒有新交易」那麼簡單，有三個地方會踩雷：

**一、`last_expire` 只看到期月的前一個月。**

    historyData = transactions where 20170101 <= transaction_date <= 20170131
    predictionCandidates = users whose last_expire in [20170201, 20170228]

也就是說 Feb cohort 的候選名單，是由**一月的交易**決定的。不是看全部歷史，
也不是看到期日當天為止 —— 這與 `src/data/cohort.py` 的 as-of 截斷是兩件
不同的事，不要混用。

**二、同日交易的排序鍵是字串串接後的字典序。**

    sig = plan_list_price + payment_plan_days + payment_method_id   ← 字串相加

`"100" + "30" + "41"` = `"1003041"`，然後比字典序。這在數值上沒有意義，
但那就是官方的行為，移植必須照抄 —— 我們要重現的是官方標籤，不是重新
設計一套更合理的。

排序規則（`calculateLastday` / `calculateRenewalGap` 共用同一個比較器）：

    1. transaction_date 遞增
    2. 同日 → sig **遞減**
    3. 同 sig → 兩筆都取消：expire 遞減
                兩筆都非取消：expire 遞增
                一取消一非取消：非取消在前
    4. 取排序後的**最後一筆**的 membership_expire_date

第 3 條看似要 pairwise 比較，其實可以化成排序鍵：混合的情況等價於
「is_cancel 遞增」，同 is_cancel 的情況等價於 expire 的方向排序。因此
把 expire 在取消時取負號，就能用單一組 sort key 表達（見 `_ordering`）。

**三、流失的定義是「gap >= 30」，而 gap 的起點會被取消往前拉。**

    走訪到期後的交易（同一個排序）：
        遇到 is_cancel == 1 且其 expire < 目前的到期日 → 把到期日**改早**
        遇到 is_cancel == 0 → gap = 該筆 transaction_date − 目前的到期日，停止
    沒有任何非取消交易 → gap = 9999

所以一位用戶在到期後先取消（把到期日拉早）再續訂，gap 是從**被拉早的那個
日子**算起，不是從原本的到期日。這會讓一些看起來「28 天內就續訂」的人被
判為流失。

    is_churn = (gap >= 30)
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from src.config import Paths

# 沒有任何非取消交易時的 gap 哨兵值，照抄原程式。
# 它大於 30，所以會被判為流失 —— 不必特別處理，讓它自然落進同一條規則。
NO_RENEWAL_GAP = 9999

# 流失門檻。原程式寫死在 `renewals.filter(col("gap") < 30)`。
CHURN_GAP_DAYS = 30

# 官方 CSV 的欄位，全部以字串讀入。
#
# 為什麼不讓 polars 推斷型別：排序鍵 sig 是三個欄位的**字串串接**，
# `plan_list_price` 若被推成整數，`100` 與 `"100"` 串出來的東西就不一樣
# （前導零、小數點都會變）。要重現官方行為，就得拿到官方看到的同一批字元。
TX_COLUMNS = (
    "msno",
    "payment_method_id",
    "payment_plan_days",
    "plan_list_price",
    "transaction_date",
    "membership_expire_date",
    "is_cancel",
)


@dataclass(frozen=True)
class LabelSpec:
    """一個月份的標籤定義。

    Attributes:
        name:          輸出檔名用。
        history_start: 決定 last_expire 的交易區間下界（含）。
        history_end:   上界（含）。原程式的 `historyCutoff`。
        expire_start:  候選名單的到期日下界（含）。
        expire_end:    上界（含）。

    官方的 Feb cohort 用 history=[20170101, 20170131]、expire=[20170201,
    20170228]，也就是「用前一個月的交易，找出這個月到期的人」。新月份沿用
    同一個相對關係。
    """

    name: str
    history_start: int
    history_end: int
    expire_start: int
    expire_end: int


# 官方已發布標籤的兩個月 —— 用來驗證這份移植，不是用來產生新標籤。
OFFICIAL_FEB = LabelSpec("feb", 20170101, 20170131, 20170201, 20170228)
OFFICIAL_MAR = LabelSpec("mar", 20170201, 20170228, 20170301, 20170331)

# 要新建的兩個月。
#
# ⚠️ 為什麼不往前再多做幾個月：Dec 2016 需要 2017-01 的交易才能判斷續約，
# 而 `transactions.csv` 的 transaction_date 起於 2015-01-01，往前做並無資料
# 上限；真正的限制是**收聽日誌**。`user_logs` 的窗口特徵要往前 90 天，
# 而 M2 的窄化快取只保留了 Feb/Mar cohort 需要的區間。多做一個月就要重掃
# 31.9 GB 一次。先做兩個月，確認穩定性的結論成不成立再決定要不要加。
DEC_2016 = LabelSpec("dec2016", 20161101, 20161130, 20161201, 20161231)
JAN_2017 = LabelSpec("jan2017", 20161201, 20161231, 20170101, 20170131)

NEW_COHORTS: dict[str, LabelSpec] = {s.name: s for s in (DEC_2016, JAN_2017)}


def scan_transactions_as_strings(paths: Paths) -> pl.LazyFrame:
    """兩個交易檔的聯集，**全部欄位以字串讀入**。

    與 `src.data.cohort.scan_transactions()` 刻意分開：那一支給特徵工程用，
    型別是推斷出來的；這一支給標籤重建用，必須拿到與官方相同的字元。
    兩支共用一份實作會讓其中一邊悄悄改變行為。
    """
    frames = [
        pl.scan_csv(paths.raw / name, infer_schema_length=0).select(TX_COLUMNS)
        for name in ("transactions.csv", "transactions_v2.csv")
    ]
    return pl.concat(frames)


def _ordering() -> list[pl.Expr]:
    """官方比較器的等價排序鍵。

    對應 Scala 的 `sortWith`：

        transaction_date 遞增
        → sig 遞減（sig = plan_list_price + payment_plan_days + payment_method_id，字串串接）
        → is_cancel 遞增（非取消在前）
        → 取消時 expire 遞減、非取消時 expire 遞增

    最後一條用「取消就取負號」壓成單一遞增鍵。expire 是 YYYYMMDD 的整數，
    取負數之後大小關係剛好反轉，與原程式的分支等價。
    """
    sig = (
        pl.col("plan_list_price") + pl.col("payment_plan_days") + pl.col("payment_method_id")
    ).alias("_sig")
    expire = pl.col("membership_expire_date").cast(pl.Int64)
    signed_expire = (
        pl.when(pl.col("is_cancel") == "1").then(-expire).otherwise(expire).alias("_expire_key")
    )
    return [
        pl.col("transaction_date"),
        sig,
        pl.col("is_cancel"),
        signed_expire,
    ]


_SORT_DESCENDING = [False, True, False, False]  # sig 遞減，其餘遞增


def last_expire(tx: pl.LazyFrame, spec: LabelSpec) -> pl.LazyFrame:
    """每位用戶在 history 區間內的「最後一筆交易」所宣告的到期日。

    對應 Scala 的 `calculateLastday`：先排序，取最後一筆的
    `membership_expire_date`。
    """
    history = tx.filter(
        (pl.col("transaction_date") >= str(spec.history_start))
        & (pl.col("transaction_date") <= str(spec.history_end))
    )
    keys = _ordering()
    return (
        history.with_columns(keys)
        .sort(
            ["msno", *[k.meta.output_name() for k in keys]], descending=[False, *_SORT_DESCENDING]
        )
        .group_by("msno", maintain_order=True)
        .agg(pl.col("membership_expire_date").last().alias("last_expire"))
    )


def renewal_gap(tx: pl.LazyFrame, candidates: pl.LazyFrame, spec: LabelSpec) -> pl.LazyFrame:
    """每位候選用戶到期後的續約間隔（天）。

    對應 Scala 的 `calculateRenewalGap`。走訪到期後的交易：取消會把「目前的
    到期日」往**早**拉，遇到第一筆非取消交易就以它結算 gap。

    Polars 沒有逐列帶狀態的迴圈，但這個狀態機可以拆成兩個聚合：

        A. 每位用戶第一筆非取消交易的位置與日期
        B. 排在它**之前**的取消交易中，最早的 expire

    `eff_expire = min(last_expire, B)`，`gap = A − eff_expire`。這與逐列
    走訪等價，因為原程式對到期日只做「取更早」這一種更新，順序不影響最小值。
    """
    future = tx.filter(pl.col("transaction_date") > str(spec.history_end))
    keys = _ordering()
    ordered = (
        candidates.join(future, on="msno", how="inner")
        .with_columns(keys)
        .sort(
            ["msno", *[k.meta.output_name() for k in keys]], descending=[False, *_SORT_DESCENDING]
        )
        .with_columns(pl.int_range(pl.len()).over("msno").alias("_idx"))
    )

    # A. 第一筆非取消交易。
    first_renewal = (
        ordered.filter(pl.col("is_cancel") == "0")
        .group_by("msno", maintain_order=True)
        .agg(
            pl.col("_idx").first().alias("_renewal_idx"),
            pl.col("transaction_date").first().alias("_renewal_date"),
        )
    )

    # B. 排在它之前的取消交易中最早的 expire。
    earliest_cancel = (
        ordered.filter(pl.col("is_cancel") == "1")
        .join(first_renewal, on="msno", how="inner")
        .filter(pl.col("_idx") < pl.col("_renewal_idx"))
        .group_by("msno", maintain_order=True)
        .agg(pl.col("membership_expire_date").min().alias("_cancel_expire"))
    )

    as_date = lambda c: pl.col(c).str.to_date("%Y%m%d")  # noqa: E731
    return (
        candidates.join(first_renewal, on="msno", how="left")
        .join(earliest_cancel, on="msno", how="left")
        .with_columns(
            pl.min_horizontal(
                "last_expire", pl.col("_cancel_expire").fill_null(pl.col("last_expire"))
            ).alias("_eff_expire")
        )
        .with_columns(
            pl.when(pl.col("_renewal_date").is_null())
            .then(pl.lit(NO_RENEWAL_GAP, dtype=pl.Int64))
            .otherwise(
                (as_date("_renewal_date") - as_date("_eff_expire")).dt.total_days().cast(pl.Int64)
            )
            .alias("gap")
        )
        .select("msno", "last_expire", "gap")
    )


def build_labels(spec: LabelSpec, paths: Paths, *, verbose: bool = True) -> pl.DataFrame:
    """重建某個月份的官方標籤。

    Returns:
        每位候選用戶一列，欄位 `msno` / `is_churn`（0/1）/ `gap` / `last_expire`。
        官方輸出只有前兩欄；`gap` 與 `last_expire` 保留下來供診斷，
        因為「為什麼這個人被判為流失」在除錯時比標籤本身有用。
    """

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    tx = scan_transactions_as_strings(paths)
    log(f"建立 {spec.name} 標籤（history {spec.history_start}~{spec.history_end}）...")

    candidates = last_expire(tx, spec).filter(
        (pl.col("last_expire") >= str(spec.expire_start))
        & (pl.col("last_expire") <= str(spec.expire_end))
    )

    out = (
        renewal_gap(tx, candidates, spec)
        .with_columns((pl.col("gap") >= CHURN_GAP_DAYS).cast(pl.Int8).alias("is_churn"))
        .select("msno", "is_churn", "gap", "last_expire")
        .collect(engine="streaming")
    )
    log(f"  候選 {out.height:,} 人，流失率 {out['is_churn'].mean():.4%}")
    return out
