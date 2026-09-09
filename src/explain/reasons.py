"""M5 · 把一筆 SHAP 歸因翻成一句人看得懂的流失原因。

SPEC §7 的 M5 驗收標準是「任一用戶可輸出 Top-3 流失原因」，例句是
「近 30 天活躍天數由 22 降至 4」這種營運讀得懂的話，不是 `log30_active_days
= 4.0, shap = +0.83`。這個模組就是那層翻譯。

## 每個特徵帶三件事，而不是只有一句話

    group     語意分組 —— 決定 Top-3 選的是三個「原因」還是三個「欄位」
    horizon   這個訊號在 M6 的 T−7 版本還在不在（見下）
    句型      中文名、單位、小數位、旗標的兩種說法

## horizon：為什麼原因碼要標註量測時點

M6 要交付 `cutoff = expire_date − 7d` 的版本（SPEC §7 M6）。同一位用戶在
T−7 評分時，**有些訊號還沒發生**：

    到期日  `last_is_cancel` —— 取消常發生在到期日當天，T−7 幾乎必然是 0。
            這是實測最強的旗標（流失率 85.70% vs 4.26%，見 src/features/build.py），
            所以它也是損失最大的一個。**這一句原因碼不可沿用到 T−7 版本。**

    位移    訊號還在，但量的是 7 天前的狀態。收聽窗口整體往前移（「近 30 天」
            指的是不同的 30 天）；交易類的 `last_*` 可能改指**另一筆交易**
            —— 到期日當天的續訂或取消會落在 cutoff 之後。

    快照    `members_v3` 的屬性，與 cutoff 幾乎無關。⚠️ 「幾乎」有兩個例外：
            這份快照本身的時點是 2017-11-13（§7.4 的已知洩漏），而註冊日晚於
            cutoff 的極少數人（Feb 6 位 / Mar 2 位）會整列退回缺失，那個判斷
            用到 cutoff（見 `_registered_after_cutoff`）。

## ⚠️ 標註不是開關

「T−7 不能用 `last_is_cancel`」的正確做法是**重訓一個 T−7 版本的模型**，
不是拿現在的模型推論時把那一欄遮掉。SHAP 的歸因是聯合的：遮掉一欄，它的
貢獻不會重新分配給其他特徵，只會讓 `sigmoid(base + Σ shap) == predict()`
這個恆等式破掉，破的量剛好是這個訊號的全部強度。

所以本模組**只標註、不過濾**。`expiry_dated_share()` 讓報表能回答「這份名單
的原因碼有多少比例撐不到 T−7 版本」—— 那是一個可以量的數字，M6 的分數下降
（SPEC 已預告「必然下降」）因此有了一個事前的估計。

## 對照數字是解釋用的，不是模型看到的

SPEC 的例句「由 22 降至 4」需要**兩個時點**，而特徵集裡沒有這個量 —— 只有
`log30_active_days`（一個絕對值）與 `log_trend_*`（比值）。可以誠實地補出來：
90 天窗口包含 30 天窗口，而活躍天數／秒數／次數是可加的，所以

    前 60 天平均每 30 天 = (log90 − log30) / 2

是純粹的既有特徵相減，不引入任何新資料。但它是**解釋用的脈絡**，模型看到的
是 `log30_active_days` 這一欄，排序也來自 SHAP 對那一欄的歸因。兩者在輸出裡
分別是 `value` 與句子裡的括號，不可混為一談。

⚠️ 只對可加的量做這件事。`unq`（不同歌曲數）不可加 —— 90 天聽過 100 首、
30 天聽過 40 首，不代表前 60 天聽了 60 首（同一首歌會重複）。比率類
（`completion` / `active_ratio` / `secs_per_active_day`）同理。7 天窗口不做，
因為「前 23 天換算成每 7 天」讀起來不像人話，而那個故事 `log_trend_7_30`
已經直接講了。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from src.features.build import GENDER_CODES, MISSING_CATEGORY
from src.features.logs import LOG_WINDOWS, MAX_WINDOW

# gender 的編碼表在 `src.features.build`，這裡只補中文。用它的鍵去查而不是自己
# 寫一份 {0: "男", 1: "女"} —— 那樣改了編碼表這裡不會有任何反應。
_GENDER_ZH = {"male": "男", "female": "女"}

# --- 語意分組 -----------------------------------------------------------------
#
# 收聽的四組沿用 M2 消融實驗的 R/F/I/T（`src.features.logs.log_feature_group`）
# —— 同一套分組驅動兩件事，「這個特徵屬於哪個家族」只有一個答案。
#
# 交易與會員這邊拆成五組。⚠️ 拆的方式會影響 Top-3：分組後取前 k 組，**大的組
# 天生容易累積較多貢獻**（9 欄相加 vs 2 欄相加）。所以價格類沒有全部塞成一個
# 「交易」，而是把「續訂設定」單獨拉出來 —— 它是營運真正能動的那個旋鈕。
G_RECENCY = "收聽近況"
G_FREQUENCY = "收聽頻率"
G_INTENSITY = "收聽強度"
G_TREND = "收聽趨勢"
G_TENURE = "訂閱資歷"
G_CANCEL = "取消紀錄"
G_RENEWAL = "續訂設定"
G_PLAN = "方案與金額"
G_MEMBER = "會員資料"

# --- 量測時點（見模組開頭）-----------------------------------------------------
HORIZON_EXPIRY = "到期日"
HORIZON_SHIFTED = "位移"
HORIZON_SNAPSHOT = "快照"

HORIZONS = (HORIZON_EXPIRY, HORIZON_SHIFTED, HORIZON_SNAPSHOT)


@dataclass(frozen=True)
class FeatureMeta:
    """一個特徵的分組、量測時點與句型。

    noun:     中文名詞片語。缺失時的句子是「{noun}缺失」，所以它必須是名詞。
    template: 覆寫預設句型（預設是「{noun} {v}{unit}」）。比值類需要它 ——
              「近 7 天聽歌時間是近 30 天日均的 0.12 倍」不是名詞加數字。
    flag:     0/1 旗標的兩種說法 `(值為 1 時, 值為 0 時)`。
    percent:  值是 0~1 的比率，印成百分比。
    category: 類別碼。`MISSING_CATEGORY` 印成「不明」。
    codes:    類別碼 → 中文（目前只有 gender）。
    context:  `(較寬窗口的欄名, 除數)`，用來補「前 60 天平均每 30 天」的對照。
    """

    group: str
    horizon: str
    noun: str
    unit: str = ""
    digits: int = 0
    template: str | None = None
    flag: tuple[str, str] | None = None
    percent: bool = False
    category: bool = False
    codes: dict[int, str] | None = None
    context: tuple[str, int] | None = None


# --- 收聽特徵：由 LOG_WINDOWS 推導，不寫死 --------------------------------------
#
# 理由與 `expected_log_columns()` 相同：加一個窗口（例如 60 天）之後，這裡會
# 自動長出對應的句型。寫死清單的話，新窗口的特徵會靜靜地沒有模板 —— 而
# `missing_metadata()` 的測試正是為了讓那件事失敗而不是靜默。
_LOG_SUFFIXES: dict[str, tuple[str, str, str, int, bool, bool]] = {
    # 後綴: (組, 名詞, 單位, 小數位, 是否百分比, 是否可加)
    "active_days": (G_FREQUENCY, "活躍天數", "天", 0, False, True),
    "active_ratio": (G_FREQUENCY, "活躍比例", "", 1, True, False),
    "secs": (G_INTENSITY, "聽歌時間", "秒", 0, False, True),
    "plays": (G_INTENSITY, "播放次數", "次", 0, False, True),
    "completed": (G_INTENSITY, "完播次數", "次", 0, False, True),
    "unq": (G_INTENSITY, "聽過的不同歌曲數", "首", 0, False, False),
    "completion": (G_INTENSITY, "完播率", "", 1, True, False),
    "secs_per_active_day": (G_INTENSITY, "活躍日平均聽歌時間", "秒", 0, False, False),
}

# 可加的量才做窗口相減的對照，而且只做 30 ↔ 90（見模組開頭）。
_CONTEXT_PAIR = (30, 90)


def _log_metadata() -> dict[str, FeatureMeta]:
    meta: dict[str, FeatureMeta] = {
        "log_min_days_before": FeatureMeta(
            G_RECENCY, HORIZON_SHIFTED, "最近一次收聽距到期日", "天"
        ),
        "log_max_days_before": FeatureMeta(
            G_RECENCY, HORIZON_SHIFTED, "最早一次收聽距到期日", "天"
        ),
        "log_has_logs": FeatureMeta(
            G_RECENCY,
            HORIZON_SHIFTED,
            "近 90 天的收聽紀錄",
            flag=("近 90 天有收聽紀錄", "近 90 天完全沒有收聽紀錄"),
        ),
        # 趨勢是比值，需要自己的句型。0.2 代表活躍度掉到五分之一。
        "log_trend_7_30": FeatureMeta(
            G_TREND,
            HORIZON_SHIFTED,
            "近 7 天聽歌時間對近 30 天日均的比值",
            "倍",
            digits=2,
            template="近 7 天聽歌時間是近 30 天日均的 {v} 倍",
        ),
        "log_trend_30_90": FeatureMeta(
            G_TREND,
            HORIZON_SHIFTED,
            "近 30 天聽歌時間對近 90 天日均的比值",
            "倍",
            digits=2,
            template="近 30 天聽歌時間是近 90 天日均的 {v} 倍",
        ),
        "log_trend_active_7_30": FeatureMeta(
            G_TREND,
            HORIZON_SHIFTED,
            "近 7 天活躍天數對近 30 天日均的比值",
            "倍",
            digits=2,
            template="近 7 天活躍天數是近 30 天日均的 {v} 倍",
        ),
    }
    narrow, wide = _CONTEXT_PAIR
    for w in LOG_WINDOWS:
        for suffix, (group, noun, unit, digits, percent, additive) in _LOG_SUFFIXES.items():
            context = None
            if additive and w == narrow and wide in LOG_WINDOWS:
                context = (f"log{wide}_{suffix}", (wide - narrow) // narrow)
            meta[f"log{w}_{suffix}"] = FeatureMeta(
                group,
                HORIZON_SHIFTED,
                f"近 {w} 天{noun}",
                unit,
                digits,
                percent=percent,
                context=context,
            )
    return meta


# --- 交易與會員特徵 ------------------------------------------------------------
FEATURES: dict[str, FeatureMeta] = {
    # 訂閱資歷。全部是「距 cutoff 幾天」或交易計數，cutoff 前移 7 天就整批位移。
    "tenure_days": FeatureMeta(G_TENURE, HORIZON_SHIFTED, "資歷（首次交易距到期日）", "天"),
    "days_since_last_tx": FeatureMeta(G_TENURE, HORIZON_SHIFTED, "距上一筆交易", "天"),
    "days_since_registration": FeatureMeta(G_TENURE, HORIZON_SHIFTED, "註冊至到期日", "天"),
    "n_tx": FeatureMeta(G_TENURE, HORIZON_SHIFTED, "歷史交易筆數", "筆"),
    # 只有固定評分日的 cohort 有這一欄（M6 的 Kaggle 管線，見 src/data/cohort.py）。
    # 量測時點標「位移」而不是「到期日」：它量的是**評分日與到期日的距離**，在
    # 評分當下完全看得到，不是到期日當天才發生的事。
    "days_to_expire": FeatureMeta(G_TENURE, HORIZON_SHIFTED, "距到期日還有", "天"),
    # 取消紀錄。
    #
    # ⚠️ `last_is_cancel` 是唯一標成「到期日」的特徵 —— 取消常發生在到期日當天，
    #    T−7 評分時那筆交易還沒發生，這句原因碼不可沿用（見模組開頭）。
    #    歷史取消次數不同：那是過去已經發生的事，T−7 一樣看得到（只是不含
    #    最後那一筆），所以標「位移」。
    "last_is_cancel": FeatureMeta(
        G_CANCEL,
        HORIZON_EXPIRY,
        "到期前最後一筆交易的取消旗標",
        flag=("到期前最後一筆交易是取消", "到期前最後一筆交易不是取消"),
    ),
    "n_cancel_hist": FeatureMeta(G_CANCEL, HORIZON_SHIFTED, "歷史取消次數", "次"),
    "cancel_rate": FeatureMeta(
        G_CANCEL, HORIZON_SHIFTED, "取消佔交易的比例", digits=1, percent=True
    ),
    # 續訂設定 —— 營運真正能動的旋鈕，所以單獨成組。
    "last_is_auto_renew": FeatureMeta(
        G_RENEWAL,
        HORIZON_SHIFTED,
        "自動續訂設定",
        flag=("已開啟自動續訂", "未開啟自動續訂"),
    ),
    "last_payment_method_id": FeatureMeta(
        G_RENEWAL, HORIZON_SHIFTED, "付款方式代碼", category=True
    ),
    # 方案與金額。
    "last_paid": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "最後一筆實付金額", "元"),
    "last_price": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "最後一筆方案定價", "元"),
    "last_plan_days": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "最後一筆方案天數", "天"),
    "discount_amount": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "最後一筆折扣金額", "元"),
    "price_per_day": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "日均單價", "元", digits=2),
    "mean_paid": FeatureMeta(G_PLAN, HORIZON_SHIFTED, "歷史平均實付金額", "元", digits=1),
    "is_free_plan": FeatureMeta(
        G_PLAN,
        HORIZON_SHIFTED,
        "免費方案旗標",
        flag=("使用免費方案（定價 0 元）", "非免費方案"),
    ),
    # SPEC §2.1：實付 0 元要拆成語意不同的兩件事，免費方案與沒收到錢不一樣。
    "zero_collected": FeatureMeta(
        G_PLAN,
        HORIZON_SHIFTED,
        "零收款旗標",
        flag=("定價非 0 元但實收 0 元", "有正常收款"),
    ),
    # 會員資料。全部來自 members_v3 快照（§7.4 的已知洩漏），與 cutoff 幾乎無關。
    "in_members": FeatureMeta(
        G_MEMBER,
        HORIZON_SNAPSHOT,
        "會員資料",
        flag=("查得到會員資料", "查不到會員資料"),
    ),
    "bd_clean": FeatureMeta(G_MEMBER, HORIZON_SNAPSHOT, "年齡", "歲"),
    "bd_valid": FeatureMeta(
        G_MEMBER,
        HORIZON_SNAPSHOT,
        "年齡欄有效性",
        flag=("年齡欄是合理值", "年齡欄無有效值"),
    ),
    "city": FeatureMeta(G_MEMBER, HORIZON_SNAPSHOT, "居住城市代碼", category=True),
    "registered_via": FeatureMeta(G_MEMBER, HORIZON_SNAPSHOT, "註冊管道代碼", category=True),
    "gender_code": FeatureMeta(
        G_MEMBER,
        HORIZON_SNAPSHOT,
        "性別",
        category=True,
        codes={code: _GENDER_ZH[name] for name, code in GENDER_CODES.items()},
    ),
    **_log_metadata(),
}


def meta(feature: str) -> FeatureMeta:
    """取一個特徵的中繼資料。

    Raises:
        KeyError: 沒有登記。**不給預設值** —— 一個沒有句型的特徵若默默印成
            欄名，名單裡就會出現「log60_secs 1234.0」這種營運看不懂的話，而且
            沒有任何東西會顯示它缺了模板。
    """
    if feature not in FEATURES:
        raise KeyError(f"特徵 {feature!r} 沒有登記句型與分組（見 src/explain/reasons.py）")
    return FEATURES[feature]


def feature_group(feature: str) -> str | None:
    """給 `top_contributors(groups=...)` 用的分組函式。未登記回 None（由該處報錯）。"""
    found = FEATURES.get(feature)
    return found.group if found else None


def missing_metadata(features) -> list[str]:
    """這些特徵裡有哪些還沒登記。供契約測試使用。"""
    return [f for f in features if f not in FEATURES]


def _format(value: float, m: FeatureMeta) -> str:
    if m.percent:
        return f"{value:.{m.digits}%}"
    return f"{value:,.{m.digits}f}"


def render_reason(
    feature: str,
    value: float | None,
    context_value: float | None = None,
    *,
    no_logs: bool = False,
) -> str:
    """一句原因碼。

    Args:
        feature: 特徵名。
        value: 該用戶在這個特徵上的值。`None` 代表模型看到的是缺失 ——
            那本身就是一種理由，不可以改印 0（與 `_attach_logs` 不補 0 同理）。
        context_value: 較寬窗口的值，用來補「前 60 天平均每 30 天」的對照。
        no_logs: 這位用戶的 `log_has_logs == 0`。見下方對缺失的處理。
    """
    m = meta(feature)

    if value is None or (isinstance(value, float) and np.isnan(value)):
        # ⚠️ **收聽欄位的缺失有一個唯一的成因，句子要講那個成因。**
        #
        # 實測（2026-08-10，Mar cohort）：`mar_log_features.parquet` 本身
        # **零 null**（796,298 列），而 cohort 有 970,959 人 —— 差的 174,661 人
        # （18.0%）根本不在那張表裡。所以特徵矩陣裡每一個 null 的收聽欄位都來自
        # `_attach_logs` 的 left join，意思是「近 90 天完全沒有收聽紀錄」。
        #
        # 印「近 14 天聽過的不同歌曲數缺失」在字面上沒錯，但讀起來像「14 天這個
        # 窗口有資料缺口」，而事實是這個人整個 90 天窗口一片空白 —— 那是**行為**
        # 而不是資料品質問題，兩者對營運的意思完全相反。
        #
        # 仍然依這一列實際的 `log_has_logs` 判斷，不假設上面那個不變量永遠成立：
        # 若哪天聚合真的產生了 null，`no_logs` 會是 False，句子就退回「缺失」。
        if no_logs and feature.startswith("log"):
            return f"近 {MAX_WINDOW} 天完全沒有收聽紀錄"
        return f"{m.noun}缺失"

    if m.flag is not None:
        return m.flag[0] if value >= 0.5 else m.flag[1]

    if m.category:
        code = int(value)
        if code == MISSING_CATEGORY:
            return f"{m.noun[:-2]}不明" if m.noun.endswith("代碼") else f"{m.noun}不明"
        if m.codes and code in m.codes:
            return f"{m.noun}為{m.codes[code]}"
        return f"{m.noun} {code}"

    # 數字與單位之間留一個空白（「149 元」而不是「149元」），與 SPEC / README
    # 全篇的中英數混排一致。百分比類的 unit 是空的，不會多出尾巴。
    shown = _format(value, m)
    tail = f" {m.unit}" if m.unit else ""
    sentence = m.template.format(v=shown) if m.template else f"{m.noun} {shown}{tail}"

    # 對照數字：前 (wide − narrow) 天換算成同樣 narrow 天的平均。
    if m.context is not None and context_value is not None and not np.isnan(context_value):
        narrow, wide = _CONTEXT_PAIR
        divisor = m.context[1]
        earlier = (context_value - value) / divisor
        if earlier >= 0:  # 90 天窗口含 30 天窗口，負值代表資料不一致，寧可不印
            sentence += f"（前 {wide - narrow} 天平均每 {narrow} 天 {_format(earlier, m)}{tail}）"
    return sentence


def add_reasons(top: pl.DataFrame, X: pl.DataFrame) -> pl.DataFrame:
    """把 `top_contributors()` 的輸出補上 `reason` / `horizon` / `expiry_dated`。

    ⚠️ **只標註，不過濾。** 到期日訊號的那幾句仍然留在輸出裡，帶著旗標 ——
    要不要用是 M6 的決定，而 M6 的正解是重訓 T−7 版本，不是遮欄位（見模組開頭）。

    Args:
        top: `src.explain.top_contributors()` 的長格式輸出。
        X: 產生歸因時用的同一份特徵矩陣（要拿對照欄的值）。
    """
    for col in ("row", "feature", "value"):
        if col not in top.columns:
            raise KeyError(f"輸入缺少欄位 {col!r}，這不像 top_contributors() 的輸出")

    unknown = missing_metadata(top["feature"].unique().to_list())
    if unknown:
        raise KeyError(f"這些特徵沒有登記句型：{sorted(unknown)}")

    # 對照欄先一次取出來（可能有 null → NaN）。
    needed = {
        FEATURES[f].context[0]
        for f in top["feature"].unique()
        if FEATURES[f].context is not None and FEATURES[f].context[0] in X.columns
    }
    context_cols = {
        c: X[c].cast(pl.Float64).to_numpy().astype(np.float64, copy=False) for c in needed
    }

    # 「這個人有沒有收聽紀錄」決定 null 的收聽欄位該講哪一句（見 `render_reason`）。
    # 沒有這一欄就一律當成「有紀錄」，句子退回保守的「缺失」。
    has_logs = (
        X["log_has_logs"].cast(pl.Float64).to_numpy() if "log_has_logs" in X.columns else None
    )

    reasons, horizons = [], []
    for row, feature, value in zip(
        top["row"].to_list(), top["feature"].to_list(), top["value"].to_list(), strict=True
    ):
        m = FEATURES[feature]
        ctx = None
        if m.context is not None and m.context[0] in context_cols:
            ctx = float(context_cols[m.context[0]][row])
        no_logs = bool(has_logs is not None and has_logs[row] == 0)
        reasons.append(render_reason(feature, value, ctx, no_logs=no_logs))
        horizons.append(m.horizon)

    return top.with_columns(
        pl.Series("reason", reasons, dtype=pl.String),
        pl.Series("horizon", horizons, dtype=pl.String),
        pl.Series("expiry_dated", [h == HORIZON_EXPIRY for h in horizons], dtype=pl.Boolean),
    )


# 呈現門檻：一句原因碼至少要有第 1 名的這個比例，才值得印給營運看。
#
# ## 為什麼需要它
#
# 實測（2026-08-10，48,853 人的名單）：第 2 句的貢獻中位數是第 1 名的 60.0%，
# 第 3 句只有 **7.5%**，而 26.2% 的人的第 3 句低於 5%。具體長相：
#
#     1. 到期前最後一筆交易是取消          +7.941
#     2. 近 14 天活躍日平均聽歌時間 790 秒  +0.093   ← 第 1 名的 1.2%
#     3. 近 90 天活躍天數 20 天             +0.075   ← 第 1 名的 0.9%
#
# 三句並列，營運會以為三件事都重要。**貢獻只有第 1 名 1/20 的東西不是原因，
# 是四捨五入的殘渣。**
#
# ## 為什麼是 0.05
#
# 這是一個**呈現判斷，不是統計檢定** —— SHAP 沒有提供「這個貢獻顯著嗎」的
# 分布。0.05 的選法是「同一份名單裡，被壓下的那些句子讀起來確實不像理由」，
# 而它的代價是可量的：26.2% 的人少一句。所以它不寫死在別處、也不放進
# `business.yaml`（那裡的參數會改變 p*，這個不會）—— 用 `--min-relative`
# 覆寫，並且**寫進 manifest**，讓「這份名單是用哪個門檻呈現的」有答案。
MIN_RELATIVE_SHARE = 0.05

# 稽核欄位裡「為什麼沒印」的值。null 代表印了。
SUPPRESSED_BY_FLOOR = "below_relative_floor"


def mark_display(
    reasons: pl.DataFrame, *, min_relative: float = MIN_RELATIVE_SHARE
) -> pl.DataFrame:
    """標出每一句該不該呈現給營運，**但不刪任何一列**。

    ## 兩層的分工

        營運呈現    只看 `displayed == True` 的句子（弱到不像理由的不印）
        底層稽核    保留全部候選 + 為什麼沒印，事後查得出當時的判斷

    刪掉被壓下的列會讓第二件事變成不可能：報表看起來很乾淨，而「這個人本來
    還有第三個理由、只是太弱」這個資訊消失了。這與 `expiry_dated` 只標註不
    過濾是同一個原則。

    Args:
        min_relative: 相對於該用戶第 1 名的門檻。0 代表全部呈現。

    Returns:
        原表加三欄：

            relative_to_top      `group_shap / 該用戶第 1 名的 group_shap`
            displayed            要不要印給營運
            suppression_reason   沒印的原因，印了則為 null

    ⚠️ **被壓下的一定是後綴。** 排名依 `group_shap` 遞減，所以
    `relative_to_top` 單調不增 —— 不可能出現「第 2 句沒印、第 3 句印了」。
    因此 `reason_1..3` 只會從尾端變空，不需要重新編號（`wide_reasons`
    依賴這個性質，`tests/test_explain.py` 釘住它）。
    """
    if "group_shap" not in reasons.columns or "rank" not in reasons.columns:
        raise KeyError("輸入缺少 group_shap / rank 欄，這不像 top_contributors() 的輸出")
    if min_relative < 0:
        raise ValueError(f"min_relative 不能為負：{min_relative}")

    top = reasons.filter(pl.col("rank") == 1).select("row", pl.col("group_shap").alias("_top"))
    out = reasons.join(top, on="row", how="left").with_columns(
        (pl.col("group_shap") / pl.col("_top")).alias("relative_to_top")
    )
    return (
        out.with_columns(
            (pl.col("relative_to_top") >= min_relative).alias("displayed"),
        )
        .with_columns(
            pl.when(pl.col("displayed"))
            .then(None)
            .otherwise(pl.lit(f"{SUPPRESSED_BY_FLOOR}({min_relative})"))
            .alias("suppression_reason")
        )
        .drop("_top")
    )


def display_impact(reasons: pl.DataFrame) -> dict:
    """呈現門檻壓掉了多少 —— 門檻的代價要跟門檻一起報。"""
    if "displayed" not in reasons.columns:
        raise KeyError("輸入缺少 displayed 欄，請先跑 mark_display()")
    people = reasons["row"].n_unique()
    suppressed = reasons.filter(~pl.col("displayed"))
    return {
        "候選句數": reasons.height,
        "呈現句數": int(reasons["displayed"].sum()),
        "壓下句數": suppressed.height,
        "受影響人數": suppressed["row"].n_unique(),
        "受影響人數比例": suppressed["row"].n_unique() / people if people else 0.0,
        "被壓下句子的相對貢獻中位數": (
            float(suppressed["relative_to_top"].median()) if suppressed.height else None
        ),
    }


def expiry_dated_share(reasons: pl.DataFrame) -> dict:
    """這批原因碼有多少撐不到 M6 的 T−7 版本。

    兩個分母是兩個不同的問題，都要回報：

        句數比例    145,644 句原因碼裡有幾句是到期日訊號
        受影響人數  有幾個人的 Top-3 裡至少有一句是（這個數字大得多，因為
                    一個人只要有一句就受影響）

    ⚠️ 這是「解釋有多少會消失」的估計，不是「分數會掉多少」。兩者方向一致但
    不是同一個量：SHAP 貢獻量與 log loss 的變化沒有換算關係。
    """
    if "expiry_dated" not in reasons.columns:
        raise KeyError("輸入缺少 expiry_dated 欄，請先跑 add_reasons()")
    if reasons.height == 0:
        raise ValueError("空的原因碼表無法計算比例")

    flagged = reasons.filter(pl.col("expiry_dated"))
    people = reasons["row"].n_unique()
    return {
        "原因碼句數": reasons.height,
        "到期日訊號句數": flagged.height,
        "句數比例": flagged.height / reasons.height,
        "有原因碼的人數": people,
        "受影響人數": flagged["row"].n_unique(),
        "受影響人數比例": flagged["row"].n_unique() / people,
        # ⚠️ 分母是**被選進 Top-3 的貢獻總和**，不是全部 61 個特徵的貢獻總和。
        # 前者才是「原因碼裡有多少比例來自到期日訊號」；後者是另一個問題。
        "到期日訊號佔 Top-3 貢獻的比例": (
            float(flagged["group_shap"].sum() / reasons["group_shap"].sum())
            if "group_shap" in reasons.columns
            else None
        ),
    }


def wide_reasons(reasons: pl.DataFrame, msno: pl.Series, *, k: int = 3) -> pl.DataFrame:
    """長格式 → 每人一列的 `reason_1..k`，**營運呈現用**。

    只帶 `displayed == True` 的句子（若已跑過 `mark_display()`）。被壓下的那些
    留在長格式的稽核表裡，不在這張表上 —— 這張是給人讀的，那張是給稽核用的。

    不足 k 句就留空 —— 補滿等於編造理由（見 `top_contributors` 的說明）。
    """
    if "displayed" in reasons.columns:
        reasons = reasons.filter(pl.col("displayed"))

    # row 的 dtype 要與 `top_contributors()` 的 Int64 一致，否則 join 會因型別
    # 不符而失敗（`with_row_index` 給的是 UInt32）。
    out = (
        pl.DataFrame({"msno": msno})
        .with_row_index("row")
        .with_columns(pl.col("row").cast(pl.Int64))
    )
    for rank in range(1, k + 1):
        part = reasons.filter(pl.col("rank") == rank).select(
            "row",
            pl.col("reason").alias(f"reason_{rank}"),
            pl.col("group").alias(f"group_{rank}"),
            pl.col("group_shap").round(4).alias(f"group_shap_{rank}"),
            pl.col("expiry_dated").alias(f"expiry_dated_{rank}"),
        )
        out = out.join(part, on="row", how="left")
    return out.drop("row")


# 稽核表的欄位順序。寫成常數是為了讓 CSV 的欄位順序有單一來源 ——
# 稽核檔的 schema 一變，下游的比對腳本就會對不上。
AUDIT_COLUMNS: tuple[str, ...] = (
    "msno",
    "rank",
    "group",
    "group_shap",
    "relative_to_top",
    "displayed",
    "suppression_reason",
    "feature",
    "feature_shap",
    "value",
    "reason",
    "horizon",
    "expiry_dated",
)


def audit_frame(reasons: pl.DataFrame, msno: pl.Series) -> pl.DataFrame:
    """長格式的稽核表：**每一句候選都在**，含沒印的與為什麼沒印。

    ⚠️ 唯一不在這裡的是**負貢獻**的特徵 —— 那不是「被壓下」，而是
    `top_contributors()` 依定義不把「降低風險的因素」當成投放理由（見該函式）。
    這個區別要留在文件裡，否則稽核者會以為表上就是全部 61 個特徵。
    """
    named = (
        pl.DataFrame({"msno": msno})
        .with_row_index("row")
        .with_columns(pl.col("row").cast(pl.Int64))
    )
    out = reasons.join(named, on="row", how="left")
    missing = [c for c in AUDIT_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(
            f"稽核表缺少欄位 {missing} —— 是不是漏跑了 add_reasons() 或 mark_display()？"
        )
    return out.select(AUDIT_COLUMNS).sort("msno", "rank")
