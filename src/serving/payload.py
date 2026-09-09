"""M6 · 一筆 payload → 一列特徵矩陣。

## 這一層唯一的職責是「把 JSON 排成 build_features() 吃的形狀」

它**不算任何特徵**。日期差、折扣金額、`price_per_day`、缺失怎麼填、
`in_members` 怎麼推 —— 全部留在 `src.features.build.build_features()`，也就是
訓練時跑的那一份程式。

理由是這類專案最典型的線上／離線不一致：服務端為了「少一個依賴」自己算一遍
`days_since_last_tx`，某個地方少考慮一個邊界（例如 `20170301 - 20170228 = 73`
那個坑，本專案踩過兩次），於是線上的特徵與訓練時的特徵是兩件不同的東西。
症狀是機率偏掉，而 API 回 200、原因碼照樣通順、沒有任何一行程式會抱怨。

所以本模組只做四件事：

    1. 檢查欄位（少了就報，多了也報 —— 打錯的欄名會靜靜地變成缺失）
    2. 型別對齊（宣告 schema，不讓 polars 從 JSON 猜）
    3. 補兩個 payload 不該提供的欄位（見下）
    4. 把 payload 沒給的收聽特徵補成 null，並**回報這件事**

## payload 不提供 `is_churn` 與 `in_members`

`is_churn` 是**標籤**。`build_features()` 需要它是因為它同時回傳 y，而服務端
的 y 沒有意義 —— 這裡填 null 並且從不讀取（`FeatureSet.y` 直接丟掉）。讓呼叫端
傳標籤是在邀請一個「服務要求你告訴它答案」的介面。

`in_members` 是**推導欄位**：`build_cohort()` 的定義是「members_v3 裡查得到這
個人」，實作是 `city.is_not_null()`。讓呼叫端自己填，就可能出現「city 有值但
in_members = false」這種訓練資料裡不存在的組合。同一個量有兩個來源時，推導
只能有一份程式。

## ⚠️ 沒有收聽特徵不是「中性預設」

`_attach_logs()` 對「不在收聽特徵表裡」的人的處理是：`log_has_logs` 填 0，其餘
收聽欄位保持 null。實測（§7.14）那正是 **18.0% 的 cohort 用戶**的長相 ——
意思是「近 90 天完全沒有收聽紀錄」。

所以 payload 省略 `logs` 時，模型收到的不是「不知道」，是**一個主張**：這個人
90 天沒聽歌。那是一句話，而不是一個空值，所以回應裡一定要帶警告。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

from src.features.build import build_features
from src.features.logs import expected_log_columns

# --- payload 的欄位與型別 ------------------------------------------------------
#
# 宣告 schema 而不是讓 polars 從一列 JSON 推斷：一列裡的 `mean_paid: 149`
# 會被推成 Int64，而訓練時它是 Float64。多數運算會自動 upcast 所以看不出來，
# 但「多數」不是「全部」，而這種差異的症狀是某幾欄的值不同、不是報錯。

# 交易史（as-of 聚合的結果，來自 `aggregate_asof`）。
TRANSACTION_FIELDS: dict[str, Any] = {
    "cutoff": pl.Int64,
    "n_tx": pl.Int64,
    "first_tx": pl.Int64,
    "last_tx": pl.Int64,
    "n_cancel_hist": pl.Int64,
    "mean_paid": pl.Float64,
    # 這六個是「cutoff 之前最後一筆交易」的時點值。**可以是 null** ——
    # 同一天多筆而該欄位取值不唯一時，資料上就沒有答案（§7.11 的 C′ 規則）。
    "last_is_cancel": pl.Int64,
    "last_is_auto_renew": pl.Int64,
    "last_actual_amount_paid": pl.Float64,
    "last_plan_list_price": pl.Float64,
    "last_payment_plan_days": pl.Int64,
    "last_payment_method_id": pl.Int64,
}

# members_v3 的屬性。查不到這個人時全部是 null，而「查不到」本身有訊號
# （實測那 11.66% 的人流失率 5.02%，低於整體 6.39%）。
MEMBER_FIELDS: dict[str, Any] = {
    "city": pl.Int64,
    "bd": pl.Int64,
    "gender": pl.String,
    "registered_via": pl.Int64,
    "registration_init_time": pl.Int64,
}

COHORT_FIELDS: dict[str, Any] = {**TRANSACTION_FIELDS, **MEMBER_FIELDS}

# 這兩欄由本模組補，payload 不得提供（理由見模組開頭）。
DERIVED_FIELDS = ("is_churn", "in_members")

# 收聽特徵的欄位（`msno` / `cutoff` 是出身證明，由本模組填）。
LOG_FIELDS: tuple[str, ...] = tuple(sorted(expected_log_columns() - {"msno", "cutoff"}))


def _check_keys(given: Mapping[str, Any], allowed: dict[str, Any] | tuple[str, ...], what: str):
    """欄位檢查：少了要報，**多了也要報**。

    多的那一半是重點：打錯一個欄名（`last_is_cancle`）若被忽略，那一欄就變成
    缺失，而模型對缺失有意見 —— 它會給出一個完全合理的機率，只是回答的是
    另一個人的問題。
    """
    names = set(allowed) if isinstance(allowed, dict) else set(allowed)
    got = set(given)
    unknown = sorted(got - names)
    if unknown:
        raise ValueError(f"{what} 有不認識的欄位：{unknown}（可用欄位見 src/serving/payload.py）")
    return sorted(names - got)


def cohort_row(features: Mapping[str, Any], *, msno: str = "unknown") -> pl.DataFrame:
    """把 payload 的交易／會員欄位排成 `build_cohort()` 輸出的形狀（一列）。

    Raises:
        ValueError: 欄位缺少、多出，或提供了推導欄位。
    """
    for field in DERIVED_FIELDS:
        if field in features:
            raise ValueError(
                f"payload 不得提供 {field!r}："
                + (
                    "它是標籤，服務端不該要求呼叫端給答案。"
                    if field == "is_churn"
                    else "它由 city 推導（見 src/serving/payload.py）。"
                )
            )
    missing = _check_keys(features, COHORT_FIELDS, "payload 的 features")
    if missing:
        raise ValueError(f"payload 的 features 缺少欄位：{missing}")

    row = {
        name: pl.Series(name, [features[name]], dtype=dtype)
        for name, dtype in COHORT_FIELDS.items()
    }
    df = pl.DataFrame({"msno": pl.Series("msno", [msno], dtype=pl.String), **row})
    return df.with_columns(
        # 標籤位：`build_features()` 需要它才組得出 FeatureSet，但服務端的 y
        # 沒有意義，填 null 並且從不讀取。
        pl.lit(None, dtype=pl.Int64).alias("is_churn"),
        # 推導欄位，與 `build_cohort()` 步驟 3 的定義完全相同。
        pl.col("city").is_not_null().alias("in_members"),
    )


def logs_row(
    msno: str, cutoff: int, logs: Mapping[str, Any] | None
) -> tuple[pl.DataFrame, list[str]]:
    """把 payload 的收聽特徵排成 `build_log_features()` 輸出的形狀（一列）。

    `logs` 為 None 或空 dict 時，全部欄位是 null —— 於是 `_attach_logs()` 會把
    `log_has_logs` 填 0，也就是「近 90 天完全沒有收聽紀錄」。那是一個主張，
    所以會回一句警告（見模組開頭）。

    Returns:
        (一列的收聽特徵表, 警告清單)
    """
    warnings: list[str] = []
    given = dict(logs or {})
    missing = _check_keys(given, LOG_FIELDS, "payload 的 logs")

    if not given:
        warnings.append(
            "payload 沒有提供收聽特徵，模型因此看到「近 90 天完全沒有收聽紀錄」"
            "（訓練資料裡 18.0% 的用戶是這個樣子）。這是一個主張，不是中性預設 —— "
            "有收聽資料時請一併帶上，否則機率不可與離線名單相比。"
        )
    elif missing:
        warnings.append(
            f"收聽特徵有 {len(missing)} 欄未提供，模型看到的是缺失：{missing[:6]}"
            f"{'…' if len(missing) > 6 else ''}"
        )

    row: dict[str, pl.Series] = {
        "msno": pl.Series("msno", [msno], dtype=pl.String),
        "cutoff": pl.Series("cutoff", [cutoff], dtype=pl.Int64),
    }
    for name in LOG_FIELDS:
        value = given.get(name)
        # `log_has_logs` 在訓練時是 0/1 的浮點旗標；沒給就讓 `_attach_logs()`
        # 依它自己的規則填 0，不在這裡先填 —— 那個規則只能有一份。
        row[name] = pl.Series(name, [value], dtype=pl.Float64)
    return pl.DataFrame(row), warnings


def feature_row(
    features: Mapping[str, Any],
    *,
    logs: Mapping[str, Any] | None = None,
    with_logs: bool = True,
    msno: str = "unknown",
    feature_names: list[str] | None = None,
) -> tuple[pl.DataFrame, list[str]]:
    """payload → 一列特徵矩陣，**轉換全部由 `build_features()` 執行**。

    Args:
        features: 交易與會員欄位（`COHORT_FIELDS`）。
        logs: 收聽特徵（`LOG_FIELDS` 的子集）。
        with_logs: 這個模型吃不吃收聽特徵。由 artifact 的欄位清單決定，
            不由 payload 決定 —— 模型要幾欄就是幾欄。
        msno: 只用於錯誤訊息與 join，不是特徵。
        feature_names: artifact 記錄的欄位清單。給了就**依名字**重排成那個順序。

    ## ⚠️ 為什麼一定要傳 `feature_names`

    離線訓練時收聽特徵的欄序來自 `build_log_features()` 的輸出（也就是那張
    parquet 的欄序）；payload 這一邊的來源是一個 **JSON 物件，它沒有有意義的
    順序**。本模組只能挑一個順序（目前是欄名排序），而那與訓練時的順序不同。

    **而 CatBoost 的 `Pool` 依位置認特徵。** 欄名一模一樣、一欄不多一欄不少、
    型別全對，只是順序不同 —— 於是每一欄的值都餵給了別的特徵，模型照樣回一個
    0~1 的機率，原因碼照樣通順。這個 bug 是在真實資料的服務煙霧測試上抓到的：
    集合相同、順序不同，兩層守門的訊息都是「缺少 []、多出 []」。

    依**名字**重排是安全的（名字帶著身分），與「假設兩邊順序剛好一樣」是兩件
    不同的事。所以順序的唯一來源是 artifact，`score_rows()` 再驗一次。

    Returns:
        (一列的特徵矩陣, 警告清單)

    Raises:
        ValueError: 欄位不合，或收聽特徵的日期早於 cutoff 之後（紅線 2 守門）。
    """
    df = cohort_row(features, msno=msno)
    warnings: list[str] = []

    if not with_logs:
        if logs:
            raise ValueError("這個模型不吃收聽特徵，但 payload 提供了 logs")
        fs = build_features(df)
    else:
        logs_df, log_warnings = logs_row(msno, int(df["cutoff"][0]), logs)
        warnings += log_warnings
        # 紅線 2 的守門（`assert_logs_within_cutoff`）在 `build_features()` 裡跑，
        # 所以一筆 `log_min_days_before` 為負的 payload 會在這裡被擋下來 ——
        # 那是「到期後的收聽行為」，用它預測到期會不會續訂是用未來預測過去。
        fs = build_features(df, logs_df)

    # `build_features()` 對「註冊日晚於 cutoff」的人會把整列 members 屬性退回
    # 缺失（`_registered_after_cutoff`）。那是正確的行為，但**它是靜默的** ——
    # payload 明明給了 city，回應的原因碼卻說「查不到會員資料」。所以講出來。
    reg = features.get("registration_init_time")
    if reg is not None and int(reg) > int(features["cutoff"]):
        warnings.append(
            f"註冊日 {reg} 晚於 cutoff {features['cutoff']}，"
            "會員屬性（city / bd / gender / registered_via）在評分時點還不存在，"
            "已全部退回缺失（src/features/build.py 的 _registered_after_cutoff）。"
        )

    X = fs.X
    if feature_names is not None:
        missing = sorted(set(feature_names) - set(X.columns))
        extra = sorted(set(X.columns) - set(feature_names))
        if missing or extra:
            raise ValueError(
                "組出來的特徵與 artifact 的清單對不上（缺少 "
                f"{missing}，多出 {extra}）。"
                "這不是 payload 的問題 —— 現行程式算出來的欄位與模型訓練時不同，"
                "artifact 需要重新匯出。"
            )
        # 依名字重排（見 docstring）。順序的來源只能是 artifact。
        X = X.select(feature_names)
    return X, warnings


def feature_rows(
    users: Sequence[Mapping[str, Any]],
    *,
    with_logs: bool = True,
    feature_names: list[str] | None = None,
) -> tuple[pl.DataFrame, list[list[str]]]:
    """一批 payload → 一個特徵矩陣，**`build_features()` 只呼叫一次**。

    Args:
        users: 每個元素是 `{"id", "features", "logs"}`。
        with_logs / feature_names: 同 `feature_row()`。

    Returns:
        (N 列的特徵矩陣, 逐列的警告清單) —— 警告與 users 同序同長。

    ## 為什麼不是「呼叫 feature_row() N 次」

    那是第一版的寫法，而它慢了 4.5 倍。`build_features()` 的成本大部分是固定
    的（建 lazy 計畫、join、算衍生欄），與列數幾乎無關 —— 一列付一次，五十列
    也只付一次。50 列實測 213 ms → 48 ms。

    在 0.1 vCPU 的免費方案上那個差距乘以 16，也就是 3.4 秒變 0.8 秒，而使用者
    等的是那個數字。

    ⚠️ **輸出必須與逐列呼叫逐格相同**，否則批次名單與單筆查詢會給出不同的機率。
    `tests/test_serving.py` 有一條測試逐格比對兩條路徑。
    """
    if not users:
        raise ValueError("沒有要組特徵的 payload")

    cohorts: list[pl.DataFrame] = []
    logs_frames: list[pl.DataFrame] = []
    warnings: list[list[str]] = []

    for u in users:
        uid, features, logs = u["id"], u["features"], u.get("logs")
        row_warnings: list[str] = []
        cohorts.append(cohort_row(features, msno=uid))

        if not with_logs:
            if logs:
                raise ValueError(f"{uid}：這個模型不吃收聽特徵，但 payload 提供了 logs")
        else:
            frame, log_warnings = logs_row(uid, int(features["cutoff"]), logs)
            logs_frames.append(frame)
            row_warnings += log_warnings

        # 與 `feature_row()` 同一句話。`build_features()` 對「註冊日晚於 cutoff」
        # 的人靜默地把 members 屬性退回缺失，所以要講出來。
        reg = features.get("registration_init_time")
        if reg is not None and int(reg) > int(features["cutoff"]):
            row_warnings.append(
                f"註冊日 {reg} 晚於 cutoff {features['cutoff']}，"
                "會員屬性（city / bd / gender / registered_via）在評分時點還不存在，"
                "已全部退回缺失（src/features/build.py 的 _registered_after_cutoff）。"
            )
        warnings.append(row_warnings)

    cohort = pl.concat(cohorts, how="vertical")
    # 紅線 2 的守門在 `build_features()` 裡跑，所以整批一起驗 —— 任何一列的
    # 收聽紀錄晚於自己的 cutoff，整批就會被擋下來。
    fs = build_features(cohort, pl.concat(logs_frames, how="vertical") if with_logs else None)

    X = fs.X
    if feature_names is not None:
        missing = sorted(set(feature_names) - set(X.columns))
        extra = sorted(set(X.columns) - set(feature_names))
        if missing or extra:
            raise ValueError(
                "組出來的特徵與 artifact 的清單對不上（缺少 "
                f"{missing}，多出 {extra}）。"
                "這不是 payload 的問題 —— 現行程式算出來的欄位與模型訓練時不同，"
                "artifact 需要重新匯出。"
            )
        X = X.select(feature_names)
    return X, warnings
