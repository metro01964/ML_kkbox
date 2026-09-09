"""把 as-of cohort 表轉成模型可用的特徵矩陣。

## 兩個核心設計決定

**一、本模組完全無狀態（stateless）。**

它不從資料計算任何統計量 —— 沒有平均數、沒有標準差、沒有類別頻率、沒有
target encoding。所有轉換都只用「這一列自己的值」和寫死的常數。

這是對紅線 5（「所有 imputation / scaling / encoding 統計量必須在 fold 內
計算」）**最強的回應：讓它無從違反**。一個不計算統計量的函式，不可能把
驗證集的資訊倒灌進訓練集。SPEC §5 註明「AI 產生的程式碼幾乎必犯」這條，
而最可靠的防法不是小心翼翼地在 fold 內 fit，是根本不需要 fit。

之所以做得到，是因為 LightGBM 本身就不需要 scaling、原生處理缺失值、
也原生處理類別特徵。M3 若引入 target encoding，那時才需要真正的 fold 內
pipeline，屆時紅線 6 的測試會派上用場。

**二、輸出不含任何原始日期。**

`cutoff`、`first_tx`、`last_tx`、`registration_init_time` 全部轉成「距離
cutoff 幾天」。理由是部署現實：訓練集的日期落在 2017-02，測試集在 2017-04，
兩者沒有交集。把原始日期餵進去，模型會學到「2017 年 2 月」這種在測試集
不存在的切點，訓練分數漂亮而測試崩盤。

⚠️ 日期相減必須先轉成真正的日期型別。`20170301 - 20170228 = 73`，不是 1 天
—— YYYYMMDD 是十進位編碼，不是天數。這個錯誤不會報錯，只會產生垃圾特徵。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from src.features.logs import assert_logs_within_cutoff
from src.fingerprint import logic_fingerprint

# LightGBM 原生支援的類別特徵。
#
# 依 LightGBM 的約定，類別特徵必須是非負整數，且**所有負值一律視為缺失**。
# 因此下方把 null 填成 -1 —— 這不是「用 -1 這個值代替缺失」，而是明確地
# 告訴 LightGBM 這裡是缺失。實測 11.66% 的 cohort 用戶不在 members_v3 中，
# 他們的 city / registered_via / gender 全部落在這一類。
CATEGORICAL: tuple[str, ...] = (
    "city",
    "registered_via",
    "last_payment_method_id",
    "gender_code",
)

MISSING_CATEGORY = -1

# gender 的編碼是**寫死的對照表**，不是從資料學來的，所以無狀態。
# 缺失獨立成一類：實測缺失者流失率 4.86%，而男 8.84% / 女 8.64% 兩者
# 幾乎沒有差異 —— 「有沒有填」比「填什麼」有訊號得多。
GENDER_CODES = {"male": 0, "female": 1}

# `members_v3.csv` 這份快照的時點。
#
# ⚠️ **這是本專案唯一一個已知、且無法從資料修復的洩漏。**
#
# 官方於 2017-11-13 發布 v3（目的是移除 members.csv 的 `expiration_date`
# 洩漏欄位，見紅線 3）。但整份檔案是**該時點的快照**，不是 as-of 各 cohort
# cutoff 的狀態。一位用戶若在 2017 年年中搬家或更新資料，Feb cohort 的特徵
# 會拿到 cutoff 之後才成立的值。
#
# 受影響的欄位與實測 gain 佔比（M2 模型，61 特徵）：
#
#     city            0.365%
#     registered_via  0.255%
#     bd              0.253%（bd_clean + bd_valid）
#     in_members      0.028%
#     gender          0.024%
#     ─────────────────────
#     合計            0.93%
#
# `registration_init_time`（0.43%）**不受影響** —— 註冊日不會事後變動。
#
# 為什麼修不掉：資料集沒有提供屬性的歷史版本，無從還原 cutoff 當下的值。
# 0.93% 是上界且很可能高估甚多（多數人的城市與性別本來就不會變）。
#
# 這個常數存在的目的是**讓時點被記錄下來**：一份沒有標註時點的快照，讀者
# 無從判斷屬性有多舊，也就無法評估風險。M6 的 MODEL_CARD 需要這個數字。
MEMBERS_SNAPSHOT_DATE = 20171113

# bd（年齡）的合理範圍。官方明示此欄含 -7168 ~ 2016 的離群值，
# cohort 內只有 39.18% 落在此區間（SPEC §5.1 要求做對照實驗，M3 處理）。
BD_MIN, BD_MAX = 10, 100


@dataclass(frozen=True)
class FeatureSet:
    """一個 cohort 的特徵矩陣與標籤。

    X 保持 polars DataFrame 而非 numpy，是為了讓特徵在訓練前仍可檢視
    （欄名、dtype、分布）。轉成 numpy 在訓練模組的邊界才做。
    """

    X: pl.DataFrame
    y: pl.Series
    msno: pl.Series
    categorical: tuple[str, ...] = CATEGORICAL

    @property
    def names(self) -> list[str]:
        return self.X.columns

    def select(self, columns: list[str]) -> FeatureSet:
        """取欄位子集，供消融實驗使用。

        `categorical` 會同步過濾 —— 忘了這一步的話，LightGBM 會拿到指向
        不存在欄位的類別索引，而且不一定會報錯，可能只是把錯的欄位當成
        類別特徵處理。
        """
        missing = set(columns) - set(self.X.columns)
        if missing:
            raise KeyError(f"要保留的欄位不存在：{sorted(missing)}")
        return FeatureSet(
            X=self.X.select(columns),
            y=self.y,
            msno=self.msno,
            categorical=tuple(c for c in self.categorical if c in columns),
        )

    def take(self, idx: Sequence[int] | pl.Series) -> FeatureSet:
        """取列的子集（供 early stopping 切分與 fold 切分使用）。

        M1 是先把 FeatureSet 轉成 numpy 才切列的。M3 不能那樣做 ——
        XGBoost 與 CatBoost 需要帶欄名與 dtype 的表格才分得出哪幾欄是類別
        特徵，一轉成無欄名的浮點矩陣就沒了。所以切列這件事往前挪到
        polars 這一層，各套件各自在自己的邊界轉換。

        `msno` 一起帶著走，否則子集就無法再做 §4.5 的分群回報。
        """
        i = pl.Series(idx) if not isinstance(idx, pl.Series) else idx
        return FeatureSet(
            X=self.X[i],
            y=self.y[i],
            msno=self.msno[i],
            categorical=self.categorical,
        )

    def __repr__(self) -> str:  # pragma: no cover - 只影響顯示
        return f"FeatureSet({self.X.height:,} 列 × {self.X.width} 特徵, 流失率 {self.y.mean():.4%})"


def _as_date(col: str) -> pl.Expr:
    """把 %Y%m%d 的整數欄位轉成日期型別。

    strict=False：無法解析的值回 null 而不是整支炸掉。SPEC §2.1 提到的哨兵值
    19700101 其實是合法日期（Unix epoch），會正常解析 —— 它在 cutoff 的計算
    階段就已被 cohort 的日期區間過濾掉，不會進到這裡。
    """
    return pl.col(col).cast(pl.Int64).cast(pl.String).str.to_date("%Y%m%d", strict=False)


def _days_before_cutoff(col: str, alias: str) -> pl.Expr:
    """該日期距離 cutoff 幾天。負值（日期晚於 cutoff）一律轉成 null。

    為什麼要擋負值：`registration_init_time` 來自 members_v3 快照，**不受
    as-of 截斷保護**。實測 Feb cohort 有 6 人、Mar cohort 有 2 人的註冊日
    晚於自己的 cutoff（例如一位有 21 筆交易的用戶「註冊」於 2017-03）。
    數量微不足道，但那是未來資訊，餵負數進模型等於開一個小洞給紅線 7。
    轉成 null 讓 LightGBM 當缺失處理，既不洩漏也不製造假訊號。
    """
    days = (_as_date("cutoff") - _as_date(col)).dt.total_days()
    return pl.when(days >= 0).then(days).otherwise(None).alias(alias)


def _registered_after_cutoff() -> pl.Expr:
    """這位用戶的 members_v3 資料在 cutoff 當下**還不存在**。

    `members_v3.csv` 是一份**快照**，不受 as-of 截斷保護（§4.3 的截斷只作用
    在 transactions 與 user_logs）。實測 Feb cohort 有 6 人、Mar cohort 有 2 人
    的 `registration_init_time` 晚於自己的 cutoff。

    對這幾個人而言，「他住哪個城市」「從哪個管道註冊」「性別是什麼」在
    評分時點通通不存在 —— 那筆會員資料是之後才建立的。把它們餵進模型，
    等於用未來才知道的事實預測過去。

    原本的處理只把 `days_since_registration` 轉成 null（因為天數會是負的，
    很顯眼），卻留下了同一列的其他欄位。**負數只是症狀，整列不該可見才是
    病因** —— 所以這個旗標一開，該用戶的**全部** members 屬性都退回缺失。

    null 一律視為「不晚於」：`registration_init_time` 為 null 代表查不到註冊
    日，那是缺資料，不是「註冊在未來」的證據，不該連帶把其他欄位也砍掉。
    """
    return (_as_date("registration_init_time") > _as_date("cutoff")).fill_null(False)


def build_features(df: pl.DataFrame, logs: pl.DataFrame | None = None) -> FeatureSet:
    """把 `src.data.build_cohort()` 的輸出轉成特徵矩陣。

    Args:
        df:   build_cohort 產生的 as-of 表，欄位見 `src.data.cohort.EXPECTED_COLUMNS`。
        logs: `src.features.build_log_features()` 的輸出（M2）。給了就以 left join
              併入。**必須是 left join** —— 實測 18.4% 的 cohort 用戶在 90 天窗口
              內沒有任何收聽紀錄，inner join 會把他們整批丟掉。

    Returns:
        FeatureSet。X 不含 `msno` 與任何原始日期欄位。

    Raises:
        KeyError: 輸入缺少必要欄位。
    """
    required = {
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
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"輸入缺少欄位：{sorted(missing)}")

    paid = pl.col("last_actual_amount_paid")
    price = pl.col("last_plan_list_price")
    plan_days = pl.col("last_payment_plan_days")

    # members_v3 的快照在 cutoff 當下還不存在的那幾位。見 _registered_after_cutoff。
    hidden = _registered_after_cutoff()

    def member_col(col: str) -> pl.Expr:
        """members_v3 來的欄位：快照不可見時退回 null。"""
        return pl.when(hidden).then(None).otherwise(pl.col(col))

    X = df.select(
        # --- 時間特徵：一律相對於 cutoff，不留原始日期 ---
        _days_before_cutoff("first_tx", "tenure_days"),
        _days_before_cutoff("last_tx", "days_since_last_tx"),
        _days_before_cutoff("registration_init_time", "days_since_registration"),
        # --- 交易史 ---
        # 實測交易史長度與流失率單調負相關：1-2 筆 26.52% → 25+ 筆 3.36%。
        pl.col("n_tx").cast(pl.Float64),
        pl.col("n_cancel_hist").cast(pl.Float64),
        (pl.col("n_cancel_hist") / pl.col("n_tx")).alias("cancel_rate"),
        pl.col("mean_paid"),
        # --- cutoff 之前最後一筆交易 ---
        # last_is_cancel 是實測最強的旗標（流失率 85.70% vs 4.26%），但它
        # 幾乎等於標籤：取消常發生在到期日當天，而 cutoff 就是到期日。
        # M6 的 cutoff = 到期日 − 7 天 版本會失去這個訊號，屆時分數必然下降。
        pl.col("last_is_cancel").cast(pl.Float64),
        pl.col("last_is_auto_renew").cast(pl.Float64),
        paid.cast(pl.Float64).alias("last_paid"),
        price.cast(pl.Float64).alias("last_price"),
        plan_days.cast(pl.Float64).alias("last_plan_days"),
        (price - paid).cast(pl.Float64).alias("discount_amount"),
        # 方案天數為 0 的交易實測有 12,861 筆，除法要擋掉否則產生 inf。
        pl.when(plan_days > 0).then(paid / plan_days).otherwise(None).alias("price_per_day"),
        # 這兩個旗標把「實付 0 元」拆成語意不同的兩件事（SPEC §2.1）：
        # 免費方案本來就收 0 元，與定價非 0 卻沒收到錢完全不同。
        (price == 0).cast(pl.Float64).alias("is_free_plan"),
        ((paid == 0) & (price > 0)).cast(pl.Float64).alias("zero_collected"),
        # --- 用戶屬性（全部來自 members_v3 快照，快照不可見時退回缺失）---
        # in_members 本身有訊號：查不到的那 11.66% 流失率 5.02%，低於整體 6.39%。
        # 註冊日晚於 cutoff 的人在此一律算作「查不到」—— 在評分時點，他們
        # 確實還不在 members_v3 裡。
        (~hidden & pl.col("in_members")).cast(pl.Float64).alias("in_members"),
        # bd 只有 39.18% 落在合理範圍。離群值不截斷也不補值，直接設 null 讓
        # LightGBM 走缺失分支；另外保留「原本是不是有效值」當獨立特徵。
        pl.when(member_col("bd").is_between(BD_MIN, BD_MAX))
        .then(member_col("bd"))
        .otherwise(None)
        .cast(pl.Float64)
        .alias("bd_clean"),
        member_col("bd")
        .is_between(BD_MIN, BD_MAX)
        .fill_null(False)
        .cast(pl.Float64)
        .alias("bd_valid"),
        # --- 類別特徵（LightGBM 原生處理，負值代表缺失）---
        member_col("city").fill_null(MISSING_CATEGORY).cast(pl.Int32).alias("city"),
        member_col("registered_via")
        .fill_null(MISSING_CATEGORY)
        .cast(pl.Int32)
        .alias("registered_via"),
        # 這一欄以前不可能是 null（總能挑到「最後一筆」交易），所以原本沒有
        # 填缺失。C′ 的逐欄位規則讓它會是 null 了：同一天多筆交易而付款方式
        # 不一致時，「最後用哪種付款方式」沒有答案（實測 Feb 少數幾百人）。
        #
        # 沒填的話 CatBoost 直接爆掉（`must be real number, not NoneType`）——
        # 它的 Pool 不接受類別欄有 None。LightGBM 反而不會叫，會安靜地把
        # NaN 當成一個獨立分支，於是三家吃到的東西不一樣，比較就不公平。
        pl.col("last_payment_method_id").fill_null(MISSING_CATEGORY).cast(pl.Int32),
        member_col("gender")
        .replace_strict(GENDER_CODES, default=MISSING_CATEGORY, return_dtype=pl.Int32)
        .alias("gender_code"),
    )

    # ---- 固定評分日的 cohort 多一個特徵：還有幾天到期 ----
    #
    # ⚠️ **有 `expire_date` 這一欄才產生它，而那一欄只有固定評分日的 spec 有。**
    #
    # 提前固定天數的設計（`cutoff = 到期日 − k`）裡，「還有幾天到期」是常數
    # （0 或 7），加它只是多一欄零 gain 的特徵。固定評分日（M6 的 Kaggle 管線）
    # 裡它是 1~30 天，而且**在評分時點看得到**：3/31 那天，一位 4/2 到期與一位
    # 4/29 到期的用戶處境完全不同，而模型沒有這一欄就分不出來。
    #
    # 負值有意義且刻意保留：到期日已經過了而還沒續訂 —— 那是強訊號，不是髒資料。
    # （`_days_before_cutoff` 擋負值是另一回事，那些欄位的負值代表未來資訊。）
    if "expire_date" in df.columns:
        # 從 df 算再接到 X 上（X 是 df.select 的輸出，列順序與長度相同）——
        # X 本身沒有原始日期欄位，那是本模組第二條設計決定的要求。
        X = X.with_columns(
            df.select(
                (_as_date("expire_date") - _as_date("cutoff"))
                .dt.total_days()
                .cast(pl.Float64)
                .alias("days_to_expire")
            ).to_series()
        )

    if logs is not None:
        X = _attach_logs(df["msno"], df["cutoff"], X, logs)

    return FeatureSet(X=X, y=df["is_churn"], msno=df["msno"])


def feature_build_fingerprint() -> str:
    """**這一版特徵轉換**的邏輯指紋（M6 用）。

    與 `cohort_fingerprint()` / `log_features_fingerprint()` 是三件不同的事：
    那兩個蓋的是「快取是哪一版程式算的」，這個蓋的是**本模組**，也就是
    「as-of 表 → 特徵矩陣」那一段。

    M6 的服務需要它，因為 artifact 裡的模型是用某一版 `build_features()` 的
    輸出訓練的，而服務用**現在這一版**把 payload 轉成特徵。兩版不同就是
    §7.11 的形狀（程式改了、模型還是舊的）—— 欄名與欄數可以完全一樣而語意
    已經變了（例如某一欄改了缺失的填法），沒有任何東西會抱怨。

    ⚠️ 只涵蓋本模組。收聽特徵那半段的邏輯在 `src.features.logs`，由
    `log_features_fingerprint()` 負責，artifact 兩個都記。
    """
    import src.features.build as _self

    return logic_fingerprint(_self)


def assert_logs_match_cohort(msno: pl.Series, cutoff: pl.Series, logs: pl.DataFrame) -> None:
    """守門：這份收聽特徵必須是用**同一批 cutoff** 算出來的。

    **這條檢查補的是紅線 2 的盲點。**

    `assert_logs_within_cutoff()` 檢查 `log_min_days_before >= 0` —— 但那是
    相對於**日誌自己那個** cutoff。一份用 Mar cutoff 算出來的收聽特徵，它的
    `log_min_days_before` 當然全部非負，所以紅線 2 必然放行。

    實測把 Mar 的收聽特徵接到 Feb 的 cohort 上：**71.89% 的用戶 join 得上**
    （兩期用戶重疊 90.81%），而那些人的 Mar cutoff 中位數比 Feb cutoff 晚
    **28 天**。於是「二月到期的用戶」的特徵含了到期後 28 天的收聽行為 ——
    而 Feb 的標籤正是由那段時間決定的。整個過程沒有任何一行程式碼會抱怨。

    因此 `build_log_features()` 在輸出裡帶了 `cutoff` 欄，這裡逐一比對。

    Raises:
        KeyError:       收聽特徵表沒有 `cutoff` 欄（舊版快取或手工組的表）。
        AssertionError: 有任何一位用戶的 cutoff 對不上。
    """
    if "cutoff" not in logs.columns:
        raise KeyError(
            "收聽特徵表缺少 cutoff 欄，無法驗證它屬於哪個 cohort。"
            "請以現行版本的 build_log_features() 重新產生（force=True）。"
        )

    ref = pl.DataFrame({"msno": msno, "cutoff": cutoff})
    joined = ref.join(
        logs.select("msno", pl.col("cutoff").alias("_log_cutoff")), on="msno", how="inner"
    )
    bad = joined.filter(pl.col("cutoff") != pl.col("_log_cutoff"))
    if bad.height:
        sample = bad.head(1).row(0, named=True)
        raise AssertionError(
            f"收聽特徵與 cohort 不符：{bad.height:,} 位用戶的 cutoff 對不上"
            f"（例：{sample['msno'][:12]}… cohort {sample['cutoff']}"
            f" vs 日誌 {sample['_log_cutoff']}）。"
            "這份收聽特徵是用別的 cohort 算的，本表不可使用。"
        )


def _attach_logs(
    msno: pl.Series, cutoff: pl.Series, X: pl.DataFrame, logs: pl.DataFrame
) -> pl.DataFrame:
    """把收聽特徵 left join 到交易特徵上。

    沒有收聽紀錄的用戶：`log_has_logs` 填 0，其餘 log 欄位保持 null 讓
    LightGBM 走缺失分支。**不補 0** —— 「沒有紀錄」和「聽了 0 秒」是不同的
    兩件事，補 0 會把前者偽裝成後者。

    實測「完全沒有紀錄」這件事本身**幾乎沒有訊號**（Feb 6.12% vs 6.45%，
    Mar 方向甚至相反），推測是自動續訂的休眠訂戶：不聽但錢照扣，所以不流失。
    保留 `log_has_logs` 讓模型自己決定要不要用。

    這一步仍然是無狀態的：每位用戶的 log 特徵只由他自己的日誌決定，與批次
    裡有哪些人無關，所以紅線 5 的無狀態測試依然成立。
    """
    if "msno" not in logs.columns:
        raise KeyError("收聽特徵表缺少 msno 欄位，無法 join")

    # ⚠️ **守門要放在這個公開邊界上，不能只放在 build_log_features() 裡。**
    #
    # `build_features(df, logs)` 是公開 API，logs 從哪來由呼叫端決定 ——
    # 可能是自己讀的 parquet、可能是消融實驗手動組出來的子集、可能是未來
    # 某條還沒寫的路徑。只在生產端守門，等於假設「所有人都會走那條路」。
    #
    # 紅線 1 的守門就是這樣設計的（`build_cohort()` 每次都跑），這裡讓紅線 2
    # 對齊同一個標準：**資料要進入模型之前，在最後一道門再確認一次。**
    assert_logs_within_cutoff(logs)
    assert_logs_match_cohort(msno, cutoff, logs)

    joined = (
        pl.DataFrame({"msno": msno})
        .join(logs, on="msno", how="left")
        .with_columns(pl.col("log_has_logs").fill_null(0.0))
    )
    if joined.height != X.height:
        raise ValueError(f"join 後列數改變（{X.height} → {joined.height}），收聽特徵有重複的 msno")

    # ⚠️ **horizontal concat 是依位置對齊的，不是依 msno。**
    #
    # 下面那行 `pl.concat(..., how="horizontal")` 把兩張表並排黏起來：第 i 列
    # 的交易特徵配第 i 列的收聽特徵。這假設 join 的輸出保持了左表的列順序。
    #
    # polars 的 left join 目前確實保持左表順序，但那是實作行為不是契約 ——
    # 換一個 engine、換一個版本、加一次 streaming，都可能重排。而一旦重排，
    # **每個人都會拿到別人的收聽特徵**：列數不變、欄位不變、沒有 null、
    # 沒有任何錯誤，模型照常訓練，分數只是變差。上面那個列數檢查完全看不出來。
    #
    # 這比 §7.8 的順序問題嚴重一個等級：那個是「換一批訓練資料」，這個是
    # 「特徵接錯人」。所以這裡直接驗證假設本身。
    if (joined["msno"] != msno).any():
        raise ValueError(
            "join 之後列順序改變，收聽特徵會接錯用戶。"
            "horizontal concat 依位置對齊，順序一變每個人都拿到別人的特徵。"
        )

    # `cutoff` 只是出身證明，驗證完就丟 —— 它不是特徵。留著會讓模型看到
    # 原始日期，而那正是本模組開頭第二條設計決定禁止的事。
    joined = joined.drop("msno", "cutoff")

    return pl.concat([X, joined], how="horizontal")
