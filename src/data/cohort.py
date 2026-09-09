"""建立 as-of cohort 特徵表 —— 本專案的防洩漏核心。

SPEC §4.3 的定義：

    cutoff(user) = membership_expire_date(user)
    所有進入特徵的 transaction_date 必須 <= cutoff

為什麼標籤會洩漏：`is_churn` 的定義是「到期後 30 天內沒有新的有效訂閱交易」。
而 transactions_v2.csv 涵蓋到 2017-03-31。對一個 2017-02 到期的用戶來說，
**他 3 月那筆交易就是答案本身**。讓它進特徵，CV 分數會漂亮到不真實，上線後全崩。

為什麼這段程式碼要獨立成模組，而不是留在 notebook 裡：

  1. **同一段邏輯要跑兩次。** M1 用 Feb cohort 訓練、Mar cohort 驗證。
     複製貼上再改日期，是紅線 1 最容易破功的地方 —— 改了一處忘了另一處，
     而且不會有任何錯誤訊息。
  2. **它可以被測試。** notebook 裡的程式碼寫不了 pytest。
  3. **截斷條件只寫在一個地方。** 要改就一起改。

本模組還內建了一個永遠會跑的守門檢查（見 build_cohort 末段）：算完之後
驗證沒有任何用戶的最後一筆交易晚於自己的 cutoff。這不是測試，是產線程式碼
的一部分 —— 洩漏一旦發生，寧可整支爆掉也不要靜靜地產出錯的表。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache

import polars as pl

from src.config import Paths, load_paths
from src.fingerprint import cache_is_current, logic_fingerprint, write_with_fingerprint


def _shift_yyyymmdd(yyyymmdd: int, days: int) -> int:
    """把 %Y%m%d 的整數往前／後移幾天。

    ⚠️ **不可以直接做整數加減。** `20170301 - 7 = 20170294`，那不是日期 ——
    YYYYMMDD 是十進位編碼不是天數。同一個坑在 `src/features/build.py` 也標了：
    `20170301 - 20170228 = 73`，不是 1 天。這種錯不會報錯，只會產生垃圾。
    """
    d = date(yyyymmdd // 10000, yyyymmdd // 100 % 100, yyyymmdd % 100) + timedelta(days=days)
    return d.year * 10000 + d.month * 100 + d.day


def _shift_days_expr(col: str, days: int) -> pl.Expr:
    """同上，但作用在一整欄上（走真正的日期型別，不是整數運算）。

    ## ⚠️ 為什麼用 strftime 而不是 year*10000 + month*100 + day

    第一版寫成那個算式，實測 20170228 往前 7 天得到 **20169965**（正解
    20170221，差 **256**）。成因是 **polars 的 `dt.month()` / `dt.day()` 回傳
    `Int8`**，而 `2 * 100 = 200` 在 Int8（上界 127）裡溢位成 −56：

        2017 * 10000            = 20170000
        (month 2) * 100 → −56   = 20169944
        + (day 21)              = 20169965

    **1 月不會出事**（`1 * 100 = 100` 還在界內），所以只用 20170201 當測資的
    測試會通過 —— 這也是原本的單元測試沒抓到的原因：它測的是 Python 版的
    `_shift_yyyymmdd`，而這一版一條測試都沒有。

    `dt.strftime("%Y%m%d")` 沒有任何算術，溢位無從發生。
    """
    shifted = pl.col(col).cast(pl.Int64).cast(pl.String).str.to_date("%Y%m%d") + pl.duration(
        days=days
    )
    return shifted.dt.strftime("%Y%m%d").cast(pl.Int64).alias(col)


@dataclass(frozen=True)
class CohortSpec:
    """一個到期月份的 cohort 定義。

    Attributes:
        name:         用於快取檔名。
        label_file:   官方標籤檔（raw/ 底下）。
        expire_start: 到期日區間下界（含），格式 %Y%m%d 的整數。
        expire_end:   到期日區間上界（含）。
        observation:  標籤的觀察期，僅供人閱讀與報表標示。
        lead_days:    **提前幾天評分**（M6，SPEC §4.3）。0 是現行版本
                      （`cutoff = 到期日`）；7 代表 `cutoff = 到期日 − 7 天`。
        score_date:   **固定評分日**（M6 的 Kaggle 管線）。給了就所有人同一天
                      評分：`cutoff = score_date`，於是提前天數隨到期日變動。
                      與 `lead_days` 互斥。
        labels_are_real: 標籤檔裡的 `is_churn` 是真的答案嗎。Kaggle 測試集的
                      `sample_submission_v2.csv` 全是 0（佔位值），標成 False
                      之後 `build_cohort()` 會把那一欄**填成 null**。

    ## 為什麼需要「固定評分日」這個第三種 cutoff 規則

    Kaggle 測試集要預測 2017-04 到期的人，而**交易與日誌都只到 2017-03-31**。
    於是 `cutoff = 到期日 − 7 天` 對 77.64% 的測試用戶（到期日在 4/8 之後）會
    落在資料結束之後 —— 那時 `transaction_date <= cutoff` 這條截斷變成空操作，
    特徵不是被我們的規則截斷，而是被**資料集的結尾**截斷：

        days_since_last_tx  最多被放大 23 天（那是上線模型 13.6% gain 的欄位）
        近 7 天收聽窗口      整段不存在，而模型把它讀成「這個人沒在聽歌」

    兩者都不會報錯，`assert_asof_respected()` 也會通過（`last_tx <= cutoff` 當然
    成立）。擋它的是 `assert_data_covers_cutoffs()`，見該函式。

    固定評分日把這件事變成一個誠實的設計：**在 2017-03-31 這一天，替所有下個月
    要到期的人評分。** 提前天數因此隨到期日變成 1~30 天，而每一欄特徵都 as-of
    一個資料完整的時點。那也是真實的部署節奏 —— 月底跑一次，寄出這個月的名單。

    ⚠️ 提前天數變動之後，「還有幾天到期」就是一個**有訊號而且看得到**的量
    （4/2 到期與 4/29 到期的人，在 3/31 的處境完全不同）。所以固定評分日的
    cohort 表多一欄 `expire_date`，而 `build_features()` 會據此多產生一個
    `days_to_expire` 特徵。提前固定天數的設計裡那一欄是常數（0 或 7），加它
    只是多一欄零 gain，所以那兩種 spec 不放。

    ## lead_days 改變的是 cutoff，不是標籤

    標籤永遠是「到期後 30 天內有沒有續訂」—— 那是我們要預測的結果，不會因為
    提前評分而改變。動的只有「評分那一刻看得到什麼」：提前 7 天，到期日當天
    的續訂或取消還沒發生，`last_is_cancel` 幾乎必然是 0（M5 量到它佔投放名單
    解釋強度的 42.35%，見 §7.14）。

    **這才是能上線的模型**：挽回優惠要提前寄出才來得及（§4.3）。
    """

    name: str
    label_file: str
    expire_start: int
    expire_end: int
    observation: str
    lead_days: int = 0
    score_date: int | None = None
    labels_are_real: bool = True

    def __post_init__(self) -> None:
        # 兩種 cutoff 規則同時給，「這個 cohort 的 cutoff 是什麼」就沒有答案 ——
        # 而下游只會拿到其中一個，安靜地。
        if self.score_date is not None and self.lead_days:
            raise ValueError(
                f"{self.name}：lead_days 與 score_date 互斥（"
                f"收到 lead_days={self.lead_days}、score_date={self.score_date}）"
            )

    @property
    def fixed_score_date(self) -> bool:
        """所有人同一天評分（提前天數隨到期日變動）。"""
        return self.score_date is not None


# SPEC §4.2 的時間外驗證切分：
#     訓練 Feb cohort（觀察期 2017-03）→ 驗證 Mar cohort（觀察期 2017-04）
# 兩者的時間結構同構，所以驗證分數可以外推到測試表現。
FEB = CohortSpec("feb", "train.csv", 20170201, 20170228, "2017-03")
MAR = CohortSpec("mar", "train_v2.csv", 20170301, 20170331, "2017-04")

# M6（§4.3）：同一批用戶、同一組標籤，只把評分時點提前 7 天。
#
# ⚠️ **cohort 成員可能因此變少。** 一位用戶若在到期前 7 天內才產生第一筆交易，
#    截斷之後他一列都不剩 —— 提前 7 天評分時，這個人還沒有可用的歷史。那不是
#    bug，是部署現實；`scripts/lead_time.py` 會把掉出去的人數報出來，並在交集
#    上另做一次同群比較。
FEB_T7 = CohortSpec("feb_t7", "train.csv", 20170201, 20170228, "2017-03", lead_days=7)
MAR_T7 = CohortSpec("mar_t7", "train_v2.csv", 20170301, 20170331, "2017-04", lead_days=7)

# M6 的 Kaggle 管線（見 CohortSpec 的說明）：**在上個月的最後一天，替這個月要
# 到期的所有人評分。** 提前天數因此是 1~30 天而不是固定 7 天。
#
# 三個 spec 的結構完全相同，所以 feb_fixed → mar_fixed 的分數是「該期待 Apr
# 提交拿到什麼」的本地估計 —— 那是本專案唯一能事先量到的東西（Apr 沒有標籤）。
#
# ⚠️ 評分日是「到期月份的前一天」：資料涵蓋到 2017-03-31，而 Apr cohort 的評分日
#    正好是那一天。這不是巧合而是這個設計的成立條件。
FEB_FIXED = CohortSpec("feb_fixed", "train.csv", 20170201, 20170228, "2017-03", score_date=20170131)
MAR_FIXED = CohortSpec(
    "mar_fixed", "train_v2.csv", 20170301, 20170331, "2017-04", score_date=20170228
)

# Kaggle 測試集。**沒有標籤** —— `sample_submission_v2.csv` 的 is_churn 全是 0
# 佔位值，`labels_are_real=False` 讓 build_cohort 把它填成 null，任何在本地算
# 分數的嘗試都會炸掉而不是回一個看起來合理的數字。
APR_FIXED = CohortSpec(
    "apr_fixed",
    "sample_submission_v2.csv",
    20170401,
    20170430,
    "2017-05",
    score_date=20170331,
    labels_are_real=False,
)

COHORTS: dict[str, CohortSpec] = {
    c.name: c for c in (FEB, MAR, FEB_T7, MAR_T7, FEB_FIXED, MAR_FIXED, APR_FIXED)
}


def cutoff_definition(spec: CohortSpec) -> str:
    """「這個 cohort 的 cutoff 是怎麼定的」的機器可讀字串。

    M5 的名單 manifest 與 M6 的模型 artifact 都必填這一欄（見
    `scripts/explain.py` 與 `src/serving/artifact.py`）—— 少了它，一個到期日
    當天評分的模型可以被當成能上線的模型部署出去，而回應看起來完全正常。

    寫成函式而不是兩邊各放一個常數：字串一旦不一致（`expire_date_minus_7d`
    vs `expire_date-7d`），比對這一欄的下游就會靜靜地認為兩份交付物不同源。
    """
    if spec.fixed_score_date:
        return f"fixed_score_date_{spec.score_date}"
    return "expire_date" if not spec.lead_days else f"expire_date_minus_{spec.lead_days}d"


def cutoff_window(spec: CohortSpec) -> tuple[int, int]:
    """這個 spec 的 cutoff 實際落在哪個區間。

    提前固定天數：到期區間往前移 `lead_days` 天。
    固定評分日：**單一天**（區間退化成一個點，於是守門變成精確的相等檢查）。
    """
    if spec.fixed_score_date:
        return (spec.score_date, spec.score_date)
    return (
        _shift_yyyymmdd(spec.expire_start, -spec.lead_days),
        _shift_yyyymmdd(spec.expire_end, -spec.lead_days),
    )


def expected_columns(spec: CohortSpec) -> frozenset[str]:
    """這個 spec 的 cohort 表**應該**有哪些欄位。

    固定評分日多一欄 `expire_date`（「還有幾天到期」的來源，見 CohortSpec）。
    寫成依 spec 而定的函式，是為了讓「少了那一欄」在快取命中時就被發現 ——
    否則 `build_features()` 會安靜地少產生一個特徵，而模型照樣訓練成功。
    """
    return EXPECTED_COLUMNS | ({"expire_date"} if spec.fixed_score_date else set())


@lru_cache(maxsize=4)
def observed_transaction_end(paths: Paths) -> int:
    """交易資料實際涵蓋到哪一天（兩個交易檔的 `transaction_date` 最大值）。

    **不寫死 20170331。** 那個值是資料的性質而不是我們的設定，寫死之後換一份
    資料（例如官方補了 4 月的交易）就會在一個沒有人會去看的常數裡過時。

    `lru_cache` 讓同一個 process 只掃一次（實測掃兩個檔的單一欄位約幾秒）。
    """
    scan = scan_transactions(paths).select(pl.col("transaction_date").max())
    return int(scan.collect(engine="streaming").item())


def assert_data_covers_cutoffs(
    cutoffs: pl.Series, *, data_end: int, spec_name: str, window_days: int = 0
) -> None:
    """守門：**沒有任何 cutoff 晚於資料實際涵蓋的最後一天。**

    ## 這條補的是「基準點合法，但基準點背後沒有資料」

    紅線 1（`assert_asof_respected`）驗的是 `last_tx <= cutoff`。cutoff 落在資料
    結束之後時，那個條件**必然成立** —— 所有交易都比 cutoff 早。於是截斷變成
    空操作，而特徵改由**資料集的結尾**決定：

        days_since_last_tx  被放大成「從最後一筆可見交易算到 cutoff」
        近 N 天的收聽窗口    整段落在資料之外，於是等於「完全沒有收聽紀錄」

    第二個尤其惡劣：`_attach_logs()` 對沒有紀錄的人填 `log_has_logs = 0`，而
    模型學到的意思是「這個人 90 天沒聽歌」（訓練資料裡 18% 的人是那樣）。
    一句**假的主張**，而它會讓機率往高風險偏。

    實測會踩到的地方：Kaggle 測試集若用 `cutoff = 到期日 − 7 天`，**77.64% 的
    用戶**（到期日在 4/8 之後）的 cutoff 會落在 20170331 之後。

    Args:
        window_days: 特徵要往回看幾天（收聽特徵的最長窗口）。目前只用於錯誤
            訊息 —— 上界是不是夠才是問題，下界不足由收斂檔的日期範圍負責。

    Raises:
        AssertionError: 有任何 cutoff 晚於 `data_end`。
    """
    if cutoffs.len() == 0:
        return
    latest = int(cutoffs.max())
    if latest <= data_end:
        return
    beyond = int((cutoffs > data_end).sum())
    raise AssertionError(
        f"{spec_name} 的 cutoff 超出資料涵蓋範圍：{beyond:,} 位用戶的 cutoff 晚於"
        f" {data_end}（最晚 {latest}）。\n"
        "  這不是洩漏，是**資料不足**：as-of 截斷會變成空操作，特徵改由資料集的"
        "結尾決定 ——\n"
        "  days_since_last_tx 被放大，而落在資料之外的收聽窗口會被模型讀成"
        f"「完全沒有收聽紀錄」（往回看 {window_days} 天）。\n"
        "  正解是改用固定評分日（見 CohortSpec.score_date），不是放行。"
    )


# 快取的 schema。欄位對不上就重算 —— 比要求使用者手動刪快取好，
# 因為「忘記刪快取所以看到舊結果」是很難察覺的錯誤。
EXPECTED_COLUMNS = frozenset(
    {
        "msno",
        "is_churn",
        "cutoff",
        "n_tx",
        "first_tx",
        "last_tx",
        "n_cancel_hist",
        "mean_paid",
        "last_is_cancel",
        "last_is_auto_renew",
        "last_actual_amount_paid",
        "last_plan_list_price",
        "last_payment_plan_days",
        "last_payment_method_id",
        "city",
        "bd",
        "gender",
        "registered_via",
        "registration_init_time",
        "in_members",
        # 稽核欄位（不進特徵矩陣，見 aggregate_asof）。列在這裡是為了讓
        # 修正之前產生的舊快取因欄位不符而自動重算 —— 那些快取的
        # last_* 值是用「隨便挑一筆」算出來的。
        "last_day_n_tx",
        "last_day_has_conflict",
    }
)


def scan_transactions(paths: Paths) -> pl.LazyFrame:
    """兩個交易檔的聯集，lazy。

    **兩個都要載入。** SPEC §5.1 的避雷清單：transactions_v2.csv 有 74.76%
    是 2017-03 的交易，它是增量更新檔不是完整歷史。只用 v2 等於完全沒有
    歷史特徵，而模型還是會訓練成功 —— 只是分數很差且找不出原因。
    """
    return pl.concat(
        [
            pl.scan_csv(paths.raw / "transactions.csv"),
            pl.scan_csv(paths.raw / "transactions_v2.csv"),
        ]
    )


# cutoff 之前「最後一筆交易」的時點值。這六個與 n_tx / mean_paid 那類聚合
# 有本質差別：後者對順序免疫，前者需要「哪一筆才算最後一筆」有定義。
# `membership_expire_date` 的哨兵值（SPEC §2.1 實測）。19700101 是 Unix epoch，
# 等同 null；20361015 是 2036 年。兩者都不是真的到期日，當日期算會產生垃圾特徵。
EXPIRY_SENTINELS = (19700101, 20361015)

LAST_TX_COLUMNS = (
    "is_cancel",
    "is_auto_renew",
    "actual_amount_paid",
    "plan_list_price",
    "payment_plan_days",
    "payment_method_id",
)


def _last_unambiguous(col: str, on_last_day: pl.Expr) -> pl.Expr:
    """最後交易日的取值；同日多筆而該欄位有衝突時給 null。

    ## 為什麼不是 sort_by("transaction_date").last()

    那個寫法假設「最後一筆」有定義。實測 **Feb cohort 有 1.30% 的用戶
    （12,889 人）在最後交易日當天有多筆交易**，而 transaction_date 只到日，
    沒有任何欄位能分出它們的先後 —— 誰是「最後一筆」在資料上就是未定義的。

    polars 於是每次重建挑到不同的那一筆：實測兩次重建之間，
    `last_actual_amount_paid` 有 24 列不同（最大差 1608）、`last_is_cancel`
    有 19 人翻面。**而 last_is_cancel 佔全模型 35.2% 的 gain。**
    Mar log loss 因此在 0.15821 ~ 0.15891 之間漂移（0.0007，約 1.5σ）。

    ## 逐欄位判斷，不是整列丟棄

    同日多筆不代表每個欄位都有歧義：一位用戶可能在同一天買了兩筆同方案、
    同金額的交易，那 `last_plan_list_price` 一點都不模糊；但若其中一筆是
    取消、另一筆不是，`last_is_cancel` 就真的沒有答案。

    因此規則是逐欄位的：**該欄位在最後交易日的非 null 取值只有一種就保留，
    有兩種以上就給 null**，讓 LightGBM 走缺失分支。實測衝突範圍：Feb 有
    0.92% 的用戶 `last_is_cancel` 取值不唯一，其餘欄位更少。

    取 null 而不是取眾數／最大值，是因為後兩者都是在編一個資料沒說的答案。
    「不知道」是這裡唯一誠實的值，而 LightGBM 原生支援它。
    """
    values = pl.col(col).filter(on_last_day).drop_nulls()
    # n_unique() <= 1 同時涵蓋兩種情形：唯一值（保留）與全 null（first() 給 null）。
    return pl.when(values.n_unique() <= 1).then(values.first()).otherwise(None).alias(f"last_{col}")


def aggregate_asof(joined: pl.LazyFrame, *, carry: tuple[str, ...] = ()) -> pl.LazyFrame:
    """每位用戶一列的 as-of 聚合。

    Args:
        joined: 交易明細，必須already含有每位用戶的 `cutoff` 欄位。
                **截斷在本函式內做**，呼叫端不需要先 filter。

    Returns:
        每位用戶一列，含順序無關的聚合、六個 `last_*` 時點值，以及兩個
        稽核欄位：

            last_day_n_tx          最後交易日當天有幾筆交易（1 = 沒有歧義）
            last_day_has_conflict  六個 last_* 之中是否有任何一個取值衝突

    稽核欄位**刻意不進特徵矩陣**（`build_features` 用白名單 select）。理由：
    「這個人的最後一天有沒有衝突」很可能與流失相關（同日多筆常見於改方案、
    取消後重買），一旦當特徵就是在用一個資料品質瑕疵預測標籤 —— 那會有效，
    但它學到的是我們的管線而不是用戶行為。留著是為了能回答「這批 null 是
    哪來的」，不是為了讓模型用。

    抽成獨立函式是為了讓它能被餵合成資料測試 —— 同 `assert_asof_respected`
    的理由。整段邏輯的正確性完全在「同日多筆時取什麼值」，而那在真實資料上
    只佔 1.3%，靠跑真實資料是驗不出來的。
    """
    # ⚠️ 這一行 filter 就是紅線 1 本身，必須在算最後交易日**之前**。
    truncated = joined.filter(pl.col("transaction_date") <= pl.col("cutoff"))

    # group_by 內的 max 是「該用戶的」最後交易日，所以這個遮罩自動是逐人的。
    on_last_day = pl.col("transaction_date") == pl.col("transaction_date").max()
    conflicts = [pl.col(c).filter(on_last_day).drop_nulls().n_unique() > 1 for c in LAST_TX_COLUMNS]

    return truncated.group_by("msno").agg(
        # is_churn 與 cutoff 對同一位用戶是常數（來自標籤檔與 cutoff 表的
        # join），所以 first() 在這裡與順序無關。
        pl.col("is_churn").first(),
        pl.col("cutoff").first(),
        # `carry` 是同樣「對一位用戶是常數」的欄位（目前只有 expire_date）。
        # 與 is_churn / cutoff 同理，first() 在這裡與順序無關。
        *[pl.col(c).first() for c in carry],
        # ---- 順序無關的聚合：不受同日多筆影響 ----
        pl.len().alias("n_tx"),
        pl.col("transaction_date").min().alias("first_tx"),
        pl.col("transaction_date").max().alias("last_tx"),
        pl.col("is_cancel").sum().alias("n_cancel_hist"),
        pl.col("actual_amount_paid").mean().alias("mean_paid"),
        # ---- 時點值：逐欄位判斷歧義 ----
        *[_last_unambiguous(c, on_last_day) for c in LAST_TX_COLUMNS],
        # ---- 稽核 ----
        pl.col("transaction_date").filter(on_last_day).len().alias("last_day_n_tx"),
        pl.any_horizontal(conflicts).alias("last_day_has_conflict"),
    )


def assert_asof_respected(df: pl.DataFrame) -> None:
    """紅線 1 守門：沒有任何用戶的最後一筆交易晚於自己的 cutoff。

    抽成獨立函式而不是寫在 build_cohort 裡面，是為了讓它可以被單獨測試。
    SPEC §5 要求每條紅線都要有「會失敗的測試」—— 測試必須能餵給它一張
    確實違規的表，確認它真的會 raise。只驗證「正常資料會通過」證明不了
    守門有效，因為一個永遠回傳 None 的空函式也會通過。

    Raises:
        AssertionError: 存在 last_tx > cutoff 的列。
    """
    missing = {"last_tx", "cutoff"} - set(df.columns)
    if missing:
        raise KeyError(f"缺少檢查所需的欄位：{sorted(missing)}")

    violations = df.filter(pl.col("last_tx") > pl.col("cutoff"))
    if violations.height:
        worst = violations.select((pl.col("last_tx") - pl.col("cutoff")).max().alias("d")).item()
        raise AssertionError(
            f"紅線 1 違反：{violations.height:,} 位用戶的特徵含 cutoff 之後的交易"
            f"（最嚴重的超出 cutoff 約 {worst} 天）。as-of 截斷失效，本表不可使用。"
        )


def assert_cutoffs_within_window(df: pl.DataFrame, spec: CohortSpec) -> None:
    """守門：每個 cutoff 都必須落在這個 cohort 宣告的到期區間內。

    **這條檢查補的是紅線 1 的盲點。**

    `assert_asof_respected()` 驗證的是「最後一筆交易不晚於 cutoff」—— 它假設
    cutoff 本身是對的。但 cutoff 可能整個來自另一個 cohort：把 Mar 的表放進
    `feb_cohort_asof.parquet`（共用 `interim/`、複製時改錯名、先跑 MAR 再改名），
    紅線 1 依然成立（Mar 的 last_tx 當然不晚於 Mar 的 cutoff），欄位檢查是
    超集比對也會通過。

    於是「二月到期的用戶」拿到三月的 cutoff，特徵裡就含了三月的交易 ——
    而 Feb 的標籤正是由三月的行為決定的。分數會變好，因此不會有人起疑。

    兩個 cohort 的到期區間不重疊，所以這個檢查對「拿錯 cohort」是決定性的：
    Mar 的 cutoff 全部落在 20170301~20170331，一條都進不了 Feb 的區間。

    ⚠️ **`lead_days` 讓區間跟著往前移，而這削弱了上一段的「決定性」。**

    比對的是 `cutoff` 落在哪，所以區間必須是「到期區間 − lead_days」：
    `mar_t7` 的 cutoff 全部落在 20170222~20170324。而那與 `feb` 宣告的
    20170201~20170228 **有 7 天重疊**（0222~0228）。

    對一張**完整**的錯置表，這個檢查仍然會叫（mar_t7 有大量列落在 0228 之後，
    整批進不了 feb 的區間）；但它不再是「由構造保證」的決定性檢查 —— 一個
    只含 0222~0228 那幾天的子集可以蒙過去。這一點寫出來，是因為原本那句
    「兩個區間不重疊」現在只對 lead_days 相同的兩個 cohort 成立。

    Raises:
        AssertionError: 存在落在區間外的 cutoff。
    """
    if "cutoff" not in df.columns:
        raise KeyError("缺少檢查所需的欄位 'cutoff'")

    lo_bound, hi_bound = cutoff_window(spec)
    outside = df.filter(~pl.col("cutoff").is_between(lo_bound, hi_bound))
    if outside.height:
        lo, hi = outside["cutoff"].min(), outside["cutoff"].max()
        lead = f"（到期 {spec.expire_start}~{spec.expire_end} 提前 {spec.lead_days} 天）"
        raise AssertionError(
            f"cohort 錯置：{outside.height:,} 列的 cutoff 落在 {spec.name} 宣告的 cutoff 區間"
            f"（{lo_bound}~{hi_bound}{lead if spec.lead_days else ''}）之外，"
            f"實際範圍 {lo}~{hi}。這份資料不屬於這個 cohort，本表不可使用。"
        )


def assert_labels_are_real(*specs: CohortSpec) -> None:
    """守門：這些 cohort 的標籤必須是真的答案，不是佔位值。

    Kaggle 測試集（`sample_submission_v2.csv`）的 `is_churn` 全是 0。拿它算
    log loss 會得到一個**看起來合理的數字**，而那個數字沒有任何意義。

    放在訓練／評估的入口（`src.models.train.load_cohort_features`），因為
    「用測試集當驗證集」是一個很容易手滑寫出來的錯 —— 只要 spec 名稱打錯一個字。

    Raises:
        AssertionError: 有任何 spec 的標籤是佔位值。
    """
    fake = [s.name for s in specs if not s.labels_are_real]
    if fake:
        raise AssertionError(
            f"{fake} 的標籤是佔位值（Kaggle 測試集），不能用於訓練或評估 —— "
            "本地算出來的任何分數都沒有意義。這個 cohort 只能用於產生提交檔。"
        )


def assert_rows_reproducible(df: pl.DataFrame) -> None:
    """守門：列順序必須是可重現的，也就是**依 msno 遞增且無重複**。

    這條補的是「快取過時」而不是「洩漏」—— §7.5 遺留的那一類問題。

    `build_cohort()` 末尾的 `.sort("msno")` 讓每次重建都得到相同的列順序，
    下游依位置切分的 `train_test_split` 才切得到同一批人。但那個保證只在
    **重算**的路徑上成立：快取命中時讀進來的是一個 parquet 檔，它可能由
    修正之前的程式產出。舊快取沒有洩漏（`assert_asof_respected` 與
    `assert_cutoffs_within_window` 都會通過），只是順序是亂的 —— 於是
    train / early stopping 換一批人，分數安靜地變動 0.0006 左右，和我們想
    量的特徵效果同一個量級。

    分數變動不會讓任何檢查失敗，只會讓人以為「這次實驗有效果」。所以這條
    守門必須跑在快取命中的路徑上，而不只是重算之後。

    Raises:
        AssertionError: 列順序不是依 msno 遞增，或 msno 有重複。
    """
    if "msno" not in df.columns:
        raise KeyError("缺少檢查所需的欄位 'msno'")
    if df.height == 0:
        return

    msno = df["msno"]
    if not msno.is_sorted():
        raise AssertionError(
            "列順序不可重現：本表未依 msno 排序。"
            '這通常代表快取由 `.sort("msno")` 修正之前的程式產出 —— '
            "它沒有洩漏，但依位置切分的下游會拿到不同的訓練集。"
            "請以 force=True 重建。"
        )
    if msno.n_unique() != msno.len():
        raise AssertionError(
            f"msno 有重複：{msno.len() - msno.n_unique():,} 列。"
            "cohort 的每位用戶只能一列，重複會讓同一個人同時進到訓練與驗證。"
        )


def cohort_fingerprint() -> str:
    """這份 cohort 快取對應的**程式版本**指紋。

    取自本模組的原始碼（剝掉 docstring 與註解，見 `src.fingerprint`）——
    `aggregate_asof`、`_last_unambiguous`、`LAST_TX_COLUMNS`、cutoff 的兩條
    filter 全都在這裡，改任何一個都會讓指紋改變。

    §7.5 遺留的待辦就是這個：欄位檢查看不出「舊版程式算出來的快取」。
    """
    import src.data.cohort as _self

    return logic_fingerprint(_self)


def build_cohort(
    spec: CohortSpec | str = FEB,
    paths: Paths | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """算出某個 cohort 每人一列的 as-of 特徵表。

    Args:
        spec:    CohortSpec，或 "feb" / "mar"。
        paths:   路徑設定，預設讀 configs/paths.yaml。
        force:   True 則忽略快取重算。
        verbose: 是否印進度。

    Returns:
        每位用戶一列。欄位見 EXPECTED_COLUMNS。

    Raises:
        AssertionError: 若產出的特徵含 cutoff 之後的交易（紅線 1 破功）。
    """
    if isinstance(spec, str):
        spec = COHORTS[spec]
    paths = (paths or load_paths()).ensure()

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    cache = paths.interim / f"{spec.name}_cohort_asof.parquet"
    fingerprint = cohort_fingerprint()

    if cache.exists() and not force and not cache_is_current(cache, fingerprint):
        # ⚠️ 欄位對得上不代表這份快取是現在這版程式算的。改一個常數、改一條
        # filter —— 欄位一個都沒變，下面四條守門全過，而內容是舊邏輯的產物。
        # 沒有指紋的快取（修正之前產生的）也走這一條，一律重算一次。
        log(f"快取 {cache.name} 的程式版本指紋不符（或缺少指紋），重算")

    elif cache.exists() and not force:
        cached = pl.read_parquet(cache)
        if expected_columns(spec) <= set(cached.columns):
            # ⚠️ **快取命中也要跑守門。**
            #
            # 快取是一個 parquet 檔，不是一個保證。它可能是舊版程式產出的、
            # 可能被手動覆蓋、可能來自另一台機器 —— 而唯一會發現的時機就是
            # 現在。只檢查欄位存在等於只驗證「形狀對」，形狀對而內容洩漏的
            # 表會一路通到模型裡，且分數會**變好**，因此不會有人起疑。
            #
            # 這一步的成本是掃兩欄比大小，相對於重算整個 cohort 微不足道。
            assert_asof_respected(cached)
            # 紅線 1 只檢查「last_tx <= cutoff」，對「這份資料屬於哪個 cohort」
            # 沒有意見。這一行補上那個盲點 —— 見 assert_cutoffs_within_window。
            assert_cutoffs_within_window(cached, spec)
            # 前兩條驗的是「有沒有洩漏」，這一條驗的是「順序可不可重現」。
            # 修正之前的舊快取兩條都會通過，只是順序亂的 —— 見
            # assert_rows_reproducible 的註解。
            assert_rows_reproducible(cached)
            log(f"讀取快取 {cache.name}（{cached.height:,} 列）")
            return cached
        log(f"快取 {cache.name} 的欄位與目前的 schema 不符，重算")

    log(f"建立 {spec.name} cohort（到期 {spec.expire_start}~{spec.expire_end}）...")

    labels = pl.read_csv(paths.raw / spec.label_file)
    tx = scan_transactions(paths)

    if not spec.labels_are_real:
        # ⚠️ **佔位標籤要換成 null，不能留著 0。**
        #
        # `sample_submission_v2.csv` 的 is_churn 全是 0（實測 907,471 列）。留著
        # 它，任何 `y.mean()` 都會回 0.0 —— 讀起來像「這個 cohort 流失率 0%」，
        # 而任何 log loss 都會算得出一個**看起來合理的數字**。null 讓那些嘗試
        # 直接壞掉，那才是正確的行為：這個 cohort 的答案在 Kaggle 手上。
        labels = labels.with_columns(pl.lit(None, dtype=pl.Int64).alias("is_churn"))
        log("  ⚠️ 這個 cohort 的標籤是佔位值，已填成 null（本地算不出分數）")

    if spec.fixed_score_date:
        # ---- 固定評分日（見 CohortSpec 的說明）----
        #
        # 成員直接來自標籤檔 —— 那是**官方宣告**「這些人在這個月到期」的名單，
        # 對 Kaggle 測試集更是唯一的定義（要交的就是那 907,471 列）。所以不從
        # 交易反推誰在 cohort 裡，只反推「他到期日是哪天」。
        #
        # ⚠️ **但成員還要再過一次「as-of 篩選」，否則本地對照會系統性偏悲觀。**
        #
        # 標籤檔是官方用**整個月**的交易決定的：一位 1/31 當下看起來 2019 年才到期
        # 的長約用戶，可能 2/10 取消而把到期日縮到 2/28；一位已經斷約一年的用戶，
        # 可能 2/10 回來續訂。這兩種人在 1/31 都不該進「二月要到期」的批次 ——
        # 而在 1/31 我們也**確實看不出**他們會進。
        #
        # 實測（觀測到的到期日是否落在目標月份內）：
        #
        #     apr_fixed（Kaggle 測試集）  100.00%   ← 官方就是 as-of 3/31 定義的
        #     feb_fixed（只用標籤檔）      89.45%
        #     mar_fixed（只用標籤檔）      89.69%
        #
        # 也就是說**測試集本身就是 as-of 定義的**，而只用標籤檔的本地 cohort 多了
        # 10.5% 測試集不會有的人 —— 而那些正是最難預測的（到期日看起來在 90 天後、
        # 或已經過期一年）。不篩掉，本地對照會低估自己、也就低估了 Apr 的預期分數。
        #
        # 判準：**看得出到期日不在目標月份的排除，看不出的（null）留下。**
        # 後者是真實的服務情境 —— 測試集有 6,580 位同日多筆而到期日不唯一的用戶，
        # Kaggle 一樣要求給他們機率（`days_to_expire` 為 null，模型原生處理缺失）。
        #
        # 剩下的一點不對稱：真正部署時批次由自己的訂閱名單定義，可能包含標籤檔裡
        # 沒有的人（他們的到期日在月中被交易改掉了）。那些人本地沒有標籤，量不了。
        #
        # ⚠️ 到期日必須以**評分日當下可觀測**的交易為準：`transaction_date <=
        # score_date`。少了那個 filter，一位在 4/5 才續訂、把到期日改成 4/25 的
        # 用戶，會讓我們在 3/31 就「知道」他 4/25 到期 —— 那是未來資訊。
        observable = tx.filter(pl.col("transaction_date") <= spec.score_date)
        # ⚠️ **不可以用 `max(membership_expire_date)`。**
        #
        # 「他現在的到期日」是**最後一筆交易宣告的那個**，不是歷來最大的那個。
        # 取 max 會被一筆遠期的離群值綁死：實測有用戶的交易宣告 2023-08-17
        # 到期，於是 `days_to_expire` 變成 2,330 天 —— 而那個人明明在 4 月到期
        # 的名單上。取 max 的版本讓 990,836 人裡的極端值一路傳到特徵裡。
        #
        # 同日多筆而宣告的到期日不一致時給 null，與六個 `last_*` 欄位共用
        # `_last_unambiguous()` 的規則（§7.11 的 C′）：「不知道」是唯一誠實的值。
        # ⚠️ 哨兵值要先變成 null，**不能當日期算**。
        #
        # SPEC §2.1 記錄了兩個：19700101（Unix epoch，等同 null）與 20361015。
        # 實測若不處理，最後一筆交易宣告 19700101 的用戶會得到
        # `days_to_expire = −17,197`（1970 到 2017 的天數）—— 一個看起來像資料
        # 而其實是「沒有值」的數字。
        #
        # 到期日的路徑（T=0 / T−7）不需要這一步：那邊的 cutoff 來自
        # `is_between(expire_start, expire_end)`，哨兵自然落在區間外。
        declared = (
            pl.when(pl.col("membership_expire_date").is_in(EXPIRY_SENTINELS))
            .then(None)
            .otherwise(pl.col("membership_expire_date"))
            .alias("membership_expire_date")
        )
        on_last_day = pl.col("transaction_date") == pl.col("transaction_date").max()
        expiry = (
            observable.with_columns(declared)
            .group_by("msno")
            .agg(_last_unambiguous("membership_expire_date", on_last_day).alias("expire_date"))
            .collect(engine="streaming")
        )
        cutoffs = (
            labels.select("msno")
            .with_columns(pl.lit(spec.score_date, dtype=pl.Int64).alias("cutoff"))
            .join(expiry, on="msno", how="left")
        )
        before = cutoffs.height
        cutoffs = cutoffs.filter(
            pl.col("expire_date").is_null()
            | pl.col("expire_date").is_between(spec.expire_start, spec.expire_end)
        )
        log(
            f"  固定評分日 {spec.score_date}：所有人同一天評分（提前天數隨到期日變動）\n"
            f"  as-of 篩選：{before - cutoffs.height:,} 人在評分日當下看得出到期日不在"
            f" {spec.expire_start}~{spec.expire_end}，排除"
            f"（保留 {cutoffs['expire_date'].null_count():,} 位到期日不唯一的）"
        )
        return _finish_cohort(spec, labels, cutoffs, tx, paths, cache, fingerprint, log)

    # ---- 步驟 1：每位用戶的 cutoff ----
    # cutoff = 落在本 cohort 到期區間內的 membership_expire_date。
    # 取 max 是因為同一個月內可能有多筆交易（例如月中改方案），最後那個
    # 才是真正的到期日。
    #
    # 第一個 filter 順帶擋掉了 SPEC §2.1 的兩個哨兵值：19700101（Unix epoch，
    # 等同 null）和 20361015（2036 年）都不在區間內，不會被選為 cutoff。
    #
    # ⚠️ **第二個 filter 是 as-of 截斷的一部分，不是資料清理。**
    #
    # 少了它，cutoff 本身就可能來自未來：實測有 158,766 筆交易的
    # transaction_date 晚於自己宣告的 membership_expire_date（95.93% 是取消
    # 紀錄，屬於補登慣例）。這種列在到期日當下**還不存在**，卻可以憑著它
    # 攜帶的到期日替用戶製造出一個 cutoff。
    #
    # 這是紅線 1 的守門抓不到的洩漏形式：`assert_asof_respected()` 檢查的是
    # 「last_tx <= cutoff」，而這裡出問題的是 **cutoff 這個基準點自己**。
    # 基準點錯了，所有以它為準的檢查都會通過，卻全部建立在未來資訊上。
    #
    # 判準：一筆交易在它自己宣告的到期日當下可觀測，等價於
    # `transaction_date <= membership_expire_date`。這個條件不循環（不依賴
    # 尚未算出的 cutoff），而且一旦成立，`transaction_date <= cutoff` 也自動
    # 成立。
    #
    # 代價：找不到任何「當下可觀測」交易的用戶會離開 cohort（實測 Feb 19 人、
    # Mar 1 人），另有 Feb 156 / Mar 148 人的 cutoff 值改變。這是正確的行為 ——
    # 部署時同樣算不出他們的 cutoff，硬留著等於假裝當時知道未來。
    cutoffs = (
        tx.filter(pl.col("membership_expire_date").is_between(spec.expire_start, spec.expire_end))
        .filter(pl.col("transaction_date") <= pl.col("membership_expire_date"))
        .group_by("msno")
        .agg(pl.col("membership_expire_date").max().alias("cutoff"))
        .collect(engine="streaming")
    )

    # ---- 步驟 1b：提前評分（M6，§4.3）----
    #
    # `lead_days = 7` 把每個人的 cutoff 往前移 7 天，於是步驟 2 的截斷會把
    # 到期前 7 天內的交易全部排除 —— 包含到期日當天那筆續訂或取消。
    #
    # ⚠️ **順序有意義：先選出 cohort 成員（用到期日），再移 cutoff。**
    # 反過來（先移再選）會讓「誰在這個 cohort 裡」也跟著變，兩個版本就不是
    # 同一批人，分數對照失去意義。成員仍可能在步驟 2 掉出去（截斷後一列不剩），
    # 那是另一回事，而且是部署現實 —— 那個人數要報出來，見 scripts/lead_time.py。
    if spec.lead_days:
        cutoffs = cutoffs.with_columns(_shift_days_expr("cutoff", -spec.lead_days))
        log(f"  提前 {spec.lead_days} 天評分：cutoff 已往前移（§4.3 的 M6 版本）")

    return _finish_cohort(spec, labels, cutoffs, tx, paths, cache, fingerprint, log)


def _finish_cohort(
    spec: CohortSpec,
    labels: pl.DataFrame,
    cutoffs: pl.DataFrame,
    tx: pl.LazyFrame,
    paths: Paths,
    cache,
    fingerprint: str,
    log,
) -> pl.DataFrame:
    """三種 cutoff 規則共用的後半段：聚合 → 接屬性 → 守門 → 快取。

    抽出來的理由是三條 cutoff 規則（到期日 / 提前固定天數 / 固定評分日）**只在
    「cutoff 怎麼來」這一段不同**。後半段複製三份的話，守門遲早只有其中一份是
    最新的 —— 而漏掉守門的那一條路徑不會有任何症狀。
    """
    cohort = labels.join(cutoffs, on="msno", how="inner")
    log(f"  標籤 {labels.height:,} 人 → 對得上 cutoff {cohort.height:,} 人")

    # ---- 資料涵蓋範圍的守門 ----
    #
    # 在聚合**之前**跑：cutoff 超出資料結尾時，截斷會變成空操作，而後面所有
    # 檢查都會通過（見 assert_data_covers_cutoffs）。
    # 收聽窗口的長度只進錯誤訊息。**在函式內 import** 避免循環依賴：
    # `src.features.logs` 在模組層 import 本模組。
    from src.features.logs import MAX_WINDOW

    assert_data_covers_cutoffs(
        cohort["cutoff"],
        data_end=observed_transaction_end(paths),
        spec_name=spec.name,
        window_days=MAX_WINDOW,
    )

    # ---- 步驟 2：as-of 聚合（含紅線 1 的截斷）----
    # 邏輯全在 aggregate_asof 裡，理由見該函式：同日多筆交易時「最後一筆」
    # 沒有定義，處理方式決定了 35.2% gain 的那個特徵長什麼樣。
    #
    # `expire_date` 只有固定評分日的 spec 有，它是 `days_to_expire` 的來源，
    # 對同一位用戶是常數，所以跟著 cutoff 一起 first() 帶過去。
    carry = ("expire_date",) if spec.fixed_score_date else ()
    asof = aggregate_asof(tx.join(cohort.lazy(), on="msno", how="inner"), carry=carry)

    # ---- 步驟 3：接上用戶屬性 ----
    # 用 left join 而不是 inner join：實測 11.66% 的 cohort 用戶不在
    # members_v3 裡，而「查不到」本身有訊號（那群人流失率 5.02%，低於
    # 整體 6.39%）。用 inner join 會把這 11.6 萬人整批丟掉。
    #
    # 只能用 members_v3.csv，不能用 members.csv（紅線 3）—— 後者含
    # expiration_date 快照欄位，官方發布 v3 就是為了移除那個洩漏欄位。
    out = (
        asof.join(pl.scan_csv(paths.raw / "members_v3.csv"), on="msno", how="left")
        .with_columns(pl.col("city").is_not_null().alias("in_members"))
        # ---- 依 msno 排序，讓列順序可重現 ----
        #
        # ⚠️ 這一行不是為了美觀，是**可重現性的必要條件**。
        #
        # polars 的 `group_by` 不保證輸出的列順序，實測同一份輸入重建兩次
        # 會得到不同的順序。而下游的 `train_test_split` 是**依位置**切分的
        # ——順序一變，train / early-stopping 就換一批人，分數跟著變。
        #
        # 實測這個效應的量級：同一份設定、同一份原始資料，只因重建快取，
        # M2 的 Mar log loss 在 0.15853 ~ 0.15910 之間跳動（差 0.00057，
        # 約 0.7 個 5-fold 標準差）。那和我們想量的特徵效果同一個量級 ——
        # 不固定順序，任何小於 1σ 的比較都是在量重建快取的運氣。
        #
        # 排 msno 而不是保留輸入順序：msno 是唯一鍵，排序結果與掃檔順序、
        # 執行緒數、polars 版本都無關。
        .sort("msno")
        .collect(engine="streaming")
    )

    # ---- 步驟 4：守門檢查 ----
    # 永遠會跑，不是只在測試裡。洩漏一旦發生，寧可整支爆掉也不要靜靜產出錯的表。
    assert_asof_respected(out)
    assert_cutoffs_within_window(out, spec)
    assert_rows_reproducible(out)

    write_with_fingerprint(out, cache, fingerprint)
    log(f"  完成 {out.height:,} 列 × {out.width} 欄，已快取 → {cache.name}")
    return out
