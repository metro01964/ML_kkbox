"""`/predict` 的具名範例 —— Swagger UI 上那個下拉選單。

## 為什麼需要

`/docs` 是這個服務唯一的介面（見 app.py 開頭）。沒有範例時，Swagger 的
「Try it out」只能依欄位型別產生骨架，整數全填 `0` —— 而 `cutoff` 是 `%Y%m%d`
的日期，`0` 不是合法值。訪客按下 Execute 得到的是錯誤，不是 Demo。

要自己填對，得先知道：日期格式、T−7 模型的 cutoff 是「到期日減 7 天」、
日期得落在模型訓練的區間、`last_payment_method_id` 填幾合理、什麼組合算高風險。
前四項文件有寫，最後一項要讀完整份 README。沒有人會做這件事。

## 資料是合成的

⚠️ **這裡沒有任何一列真實資料。** 依競賽規則，KKBox 資料與其衍生特徵不得散布
（見 MODEL_CARD.md 的授權聲明、deploy/serving.yaml 關閉 msno 介面的理由）。

每個原型的數值是**依 README 已公開的統計分佈手造**的：

    最後一筆已取消     流失率 85.70%   佔用戶 2.4%
    自動續訂關閉       流失率 32.25%   佔用戶 11.2%
    其餘               流失率  0.80%   佔用戶 86.4%
    首次到期的新客     流失率 39.84%   佔 Mar cohort 9.19%
    完全沒有收聽紀錄   訓練資料裡 18.0% 的用戶

## 衍生欄位由程式算，不手寫

38 個收聽欄位裡有 12 個是比率（completion、active_ratio、secs_per_active_day）
與 3 個是趨勢。手寫的話，「完播數 118 / 播放數 140」與「完播率 0.62」很容易
對不上 —— 那會讓模型看到自相矛盾的輸入，而 Demo 上看不出來。

所以這裡只手寫**基礎計數**，比率照 `src/features/logs.py` 的 `_derived()` 與
`_ratio()` 同一組公式算出來。公式改了而這裡沒跟上，
`tests/test_examples.py` 會失敗。
"""

from __future__ import annotations

from typing import Any

# src/features/logs.py 的 LOG_WINDOWS。在這裡重列一次而不是 import，是為了讓
# 這個模組不依賴特徵層 —— 對不上時由測試抓，不是由 import 順序決定。
WINDOWS = (7, 14, 30, 90)


def _log_block(
    per_window: dict[int, dict[str, float]],
    *,
    min_days_before: float,
    max_days_before: float,
) -> dict[str, float | None]:
    """由基礎計數展開成 38 欄。

    公式與 `src/features/logs.py` 一致：
        completion          = completed / plays          （plays 為 0 時 null）
        active_ratio        = active_days / w
        secs_per_active_day = secs / active_days         （active 為 0 時 null）
        trend_a_b           = (a / a_days) / (b / b_days)（分母為 0 時 null）
    """
    out: dict[str, float | None] = {}
    for w in WINDOWS:
        base = per_window[w]
        active, secs = base["active_days"], base["secs"]
        plays, completed, unq = base["plays"], base["completed"], base["unq"]
        out[f"log{w}_active_days"] = active
        out[f"log{w}_secs"] = secs
        out[f"log{w}_plays"] = plays
        out[f"log{w}_completed"] = completed
        out[f"log{w}_unq"] = unq
        out[f"log{w}_completion"] = round(completed / plays, 4) if plays > 0 else None
        out[f"log{w}_active_ratio"] = round(active / w, 4)
        out[f"log{w}_secs_per_active_day"] = round(secs / active, 1) if active > 0 else None

    def trend(num: str, num_days: int, den: str, den_days: int) -> float | None:
        per_day_den = out[den] / den_days  # type: ignore[operator]
        if not per_day_den > 0:
            return None
        return round((out[num] / num_days) / per_day_den, 4)  # type: ignore[operator]

    out["log_trend_7_30"] = trend("log7_secs", 7, "log30_secs", 30)
    out["log_trend_30_90"] = trend("log30_secs", 30, "log90_secs", 90)
    out["log_trend_active_7_30"] = trend("log7_active_days", 7, "log30_active_days", 30)
    out["log_min_days_before"] = min_days_before
    out["log_max_days_before"] = max_days_before
    out["log_has_logs"] = 1.0
    return out


# ---------------------------------------------------------------------------
# 五個原型
# ---------------------------------------------------------------------------
#
# cutoff 一律落在 20170125–20170221 —— artifact 的 cohort.cutoff_window。
# 落在區間外不會被拒絕，但那是模型沒見過的時點，機率不可信。

_CANCELLED = {
    "features": {
        "cutoff": 20170214,
        "n_tx": 9,
        "first_tx": 20160312,
        "last_tx": 20170208,
        "n_cancel_hist": 2,
        "mean_paid": 149.0,
        "last_is_cancel": 1,
        # ⚠️ 自動續訂必須是 1，不是 0。
        #
        # 這兩欄不是獨立的：**沒開自動續訂的人根本不需要取消**，時間到就自然
        # 結束。實測 Feb cohort 99.2 萬人，`已取消 × 自動續訂關` 這個組合
        # **一筆都不存在**（見 reports/figures/03_cancel_x_autorenew_heatmap.png）。
        #
        # 這裡原本寫 0，於是這個「最高風險」的範例是一個訓練資料裡零支撐的
        # 組合 —— 模型照樣回一個看起來很合理的 0.8253，沒有任何地方會報錯。
        # 改成 1（24,303 人、流失率 85.70% 的真實分群）之後是 0.9652，而且
        # 取消那句原因碼的強度從 +2.39 升到 +4.11：模型真正學過的是「主動
        # 取消一個原本會自動扣款的訂閱」，不是這個拼出來的組合。
        "last_is_auto_renew": 1,
        "last_actual_amount_paid": 0.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 40,
        "city": 5,
        "bd": 31,
        "gender": "male",
        "registered_via": 9,
        "registration_init_time": 20160310,
    },
    "logs": _log_block(
        {
            # 近一週完全停止，但 90 天前是活躍的 —— 典型的「先冷掉再退訂」
            7: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            14: {"active_days": 1, "secs": 900.0, "plays": 6, "completed": 3, "unq": 5},
            30: {"active_days": 6, "secs": 9800.0, "plays": 61, "completed": 38, "unq": 44},
            90: {"active_days": 41, "secs": 118000.0, "plays": 720, "completed": 480, "unq": 390},
        },
        min_days_before=12,
        max_days_before=88,
    ),
}

_AUTORENEW_OFF = {
    "features": {
        # 到期日 20170214，30 天方案 → 上一次續訂約 20170115（cutoff 前 23 天）。
        # 這三個日期要對得起來，否則 days_since_last_tx 會講出一個與方案週期
        # 矛盾的故事，而那正是模型最看重的特徵之一。
        "cutoff": 20170207,
        "n_tx": 14,
        "first_tx": 20151120,
        "last_tx": 20170115,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 0,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 38,
        "city": 13,
        "bd": 27,
        "gender": "female",
        "registered_via": 7,
        "registration_init_time": 20151118,
    },
    "logs": _log_block(
        {
            # 漸行漸遠：90 天前很活躍，近兩週剩零星，近一週只剩一天且完播率掉下來。
            # 這是「還沒退訂但已經不用了」的樣子。
            7: {"active_days": 1, "secs": 620.0, "plays": 9, "completed": 3, "unq": 8},
            14: {"active_days": 3, "secs": 2900.0, "plays": 26, "completed": 12, "unq": 22},
            30: {"active_days": 9, "secs": 14000.0, "plays": 104, "completed": 58, "unq": 81},
            90: {"active_days": 58, "secs": 141000.0, "plays": 910, "completed": 604, "unq": 520},
        },
        min_days_before=5,
        max_days_before=89,
    ),
}

_LOYAL = {
    "features": {
        # 到期日 20170208，30 天方案 → 上一次自動續訂 20170109。
        "cutoff": 20170201,
        "n_tx": 26,
        "first_tx": 20150118,
        "last_tx": 20170109,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 1,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 41,
        "city": 1,
        "bd": 35,
        "gender": "male",
        "registered_via": 9,
        "registration_init_time": 20150115,
    },
    "logs": _log_block(
        {
            # 幾乎天天聽，完播率高且四個窗口一致 —— 沒有衰退訊號
            7: {"active_days": 7, "secs": 29000.0, "plays": 140, "completed": 118, "unq": 96},
            14: {"active_days": 14, "secs": 57000.0, "plays": 275, "completed": 232, "unq": 180},
            30: {"active_days": 29, "secs": 122000.0, "plays": 590, "completed": 498, "unq": 340},
            90: {"active_days": 86, "secs": 361000.0, "plays": 1750, "completed": 1470, "unq": 820},
        },
        min_days_before=0,
        max_days_before=89,
    ),
}

_NEW_SUBSCRIBER = {
    "features": {
        # 首購後第一次到期。n_tx = 1 且 first_tx == last_tx 是這個族群的指紋。
        # 首購 20170108，30 天方案 → 到期 20170207，T−7 的 cutoff 是 20170131。
        "cutoff": 20170131,
        "n_tx": 1,
        "first_tx": 20170108,
        "last_tx": 20170108,
        "n_cancel_hist": 0,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 1,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 41,
        "city": 22,
        "bd": 24,
        "gender": None,  # 缺失率 65.43%，而且「沒填」本身有訊號
        "registered_via": 4,
        "registration_init_time": 20170102,
    },
    "logs": _log_block(
        {
            # 衝動購買型：買了之後試用幾天就沒再打開。只有 23 天歷史，
            # 所以 30 天與 90 天窗口的數字相同 —— 這本身就是「新客」的指紋。
            7: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            14: {"active_days": 0, "secs": 0.0, "plays": 0, "completed": 0, "unq": 0},
            30: {"active_days": 5, "secs": 5400.0, "plays": 38, "completed": 14, "unq": 35},
            90: {"active_days": 5, "secs": 5400.0, "plays": 38, "completed": 14, "unq": 35},
        },
        min_days_before=17,
        max_days_before=22,
    ),
}

_NO_LISTENING = {
    # 刻意不給 logs。訓練資料裡 18.0% 的用戶就是這個樣子，而回應會帶一句
    # 警告說明「省略不是中性預設，是一個主張」—— 那句警告本身值得被看到。
    "features": {
        # 到期日 20170223，30 天方案 → 上一次續訂 20170124。
        "cutoff": 20170216,
        "n_tx": 5,
        "first_tx": 20160820,
        "last_tx": 20170124,
        "n_cancel_hist": 1,
        "mean_paid": 149.0,
        "last_is_cancel": 0,
        "last_is_auto_renew": 0,
        "last_actual_amount_paid": 149.0,
        "last_plan_list_price": 149.0,
        "last_payment_plan_days": 30,
        "last_payment_method_id": 38,
        "city": 15,
        "bd": 0,  # 無效值。README：填 0 的人流失率 4.76%，比填了有效年齡的低
        "gender": None,
        "registered_via": 3,
        "registration_init_time": 20160818,
    },
}


# ⚠️ 說明文字裡引用的百分比一律是**分群的平均流失率**（來自 README 的 M0 分析），
# 不是「這個範例會輸出的機率」。兩者本來就不同 —— 模型看的是這 61 個特徵的組合，
# 不是這個人屬於哪一群。④ 就是刻意留著的反例：群體平均 39.84%，個體卻很低。
#
# 也刻意不把實際輸出的數字寫進說明。模型重新匯出後那些數字會變，而寫死的說明
# 不會跟著變 —— 一份與程式不同步的文件比沒有文件更糟。

OPENAPI_EXAMPLES: dict[str, dict[str, Any]] = {
    "cancelled": {
        "summary": "① 高風險：已經按過取消",
        "description": (
            "**這類人只佔全部用戶的 2.4%，但每 7 個就有 6 個會離開** —— "
            "而且他們一群人就佔了全部流失量的三分之一。\n\n"
            "兩個壞消息同時出現：主動按過取消、最近一週完全沒聽歌"
            "（雖然三個月前還很活躍）。五個範例裡機率最高的一個。\n\n"
            "⚠️ 注意他的自動續訂是**開著**的，這不是筆誤：沒開自動續訂的人"
            "根本不需要取消，時間到就自然結束。所以「已取消 × 自動續訂關」"
            "這個組合在 99.2 萬人裡一筆都不存在 —— 取消之所以是最強的訊號，"
            "正因為它是唯一需要用戶**主動做一件事**的狀態。\n\n"
            "⚠️ 另一個陷阱：取消這個動作常常就發生在到期當天。"
            "而真正能上線的模型必須**提前七天**判斷 —— 那時候還看不到這個訊號。"
            "提前七天的代價是準確度掉約 18%，這個數字專案裡有量。"
        ),
        "value": _CANCELLED,
    },
    "autorenew_off": {
        "summary": "② 中風險：還沒退訂，但已經不用了",
        "description": (
            "**這類人佔全部用戶的 11.2%，大約每 3 個有 1 個會離開。**"
            "因為人數多，他們佔了全部流失量的 56% —— 比 ① 那群還多。\n\n"
            "帳面上一切正常，沒按過取消。但自動續訂關著，而且最近一週只打開過一天，"
            "「整首聽完」的比例掉到剩三分之一。\n\n"
            "跟 ① 對照就看得出來：**光看一個開關不夠**，模型讀的是一整組行為。"
        ),
        "value": _AUTORENEW_OFF,
    },
    "loyal": {
        "summary": "③ 低風險：訂了兩年，每天都在聽",
        "description": (
            "**這類人佔全部用戶的 85.8%，1000 個裡只有 6 個會離開。**\n\n"
            "訂閱兩年、26 次扣款、自動續訂開著，而且不論看最近 7 天、14 天、30 天"
            "還是 90 天，聽歌的量都差不多 —— 沒有變冷的跡象。\n\n"
            "把這個跟 ① 輪流點一次，是最快看懂模型在做什麼的方法：機率會差上百倍。\n\n"
            "⚠️ 不過這群人不能因為風險低就忽略。他們人數實在太多，"
            "那 0.6% 加起來仍然是全部流失量的 8%。而且**用開關規則抓不出他們** —— "
            "這正是需要模型、而不是寫幾個 if 判斷的理由。"
        ),
        "value": _LOYAL,
    },
    "new_subscriber": {
        "summary": "④ 新客：整群很危險，不代表每個人都危險",
        "description": (
            "第一次訂閱剛好到期的人，只扣過一次款。"
            "**這類人平均每 5 個就有 2 個會離開，是老訂戶的近 7 倍。**\n\n"
            "但這一位跑出來的機率會**遠低於**那個平均值，因為他的自動續訂是開著的。\n\n"
            "**這個落差是故意留著的。** 模型看的是這個人的行為，"
            "不是「他被分在新客這一組」。如果照組別發優惠（「所有新客都發」），"
            "那就不需要機器學習了，寫個 if 判斷就好。模型的價值在於回答"
            "**同樣是新客，誰該發、誰不用發**。\n\n"
            "（他確實有風險訊號：買了之後只用五天，最近兩週完全沒打開。"
            "這些會出現在下方的原因排序裡，只是還壓不過自動續訂那一項。）"
        ),
        "value": _NEW_SUBSCRIBER,
    },
    "no_listening": {
        "summary": "⑤ 沒有收聽資料：看服務怎麼說實話",
        "description": (
            "**這一筆刻意不提供收聽紀錄**，訓練資料裡有 18% 的用戶就是這樣。\n\n"
            "重點不在機率，在下方跳出來的**警告訊息**：服務會告訴你"
            "「沒給收聽資料不等於中性，這等於在主張這個人 90 天沒聽過歌」，"
            "並附上那個 18% 讓你自己判斷這個假設合不合理。\n\n"
            "**一個會告訴你它假設了什麼的模型，比一個安靜給答案的模型有用。**\n\n"
            "這一筆的年齡欄位也是 0 —— 那是原始資料裡的無效值，服務不會替你猜一個。"
        ),
        "value": _NO_LISTENING,
    },
}


# ---------------------------------------------------------------------------
# Demo 批次：一批到期用戶
# ---------------------------------------------------------------------------
#
# ⚠️ **這裡同樣沒有任何一列真實資料**，理由與上面五個原型相同（競賽規則）。
#
# ## 為什麼要一批人
#
# 這個專案的第二項交付物是**投放名單**（M5：48,853 人 + 逐人原因碼）。單人介面
# 看得到機率與理由，營運視角看得到曲線與總數，但「名單本身」在 Demo 上一直是
# 看不到的 —— 而那正是交付給營運單位的東西。
#
# ## 組成刻意加重風險族群，但每一群的**行為**照 M0 的實測
#
# 真實 Mar cohort 依 p* 只有 3.83% 上榜。50 人照真實比例抽樣平均只有 2 人越過
# 門檻，那條線就讀不出來了。所以這裡調的是**各群的人數比例**，不是各群的
# 行為 —— 每個原型的旗標、方案、活躍度仍照 README 的 M0 分群統計：
#
#     已按過取消          流失率 85.70%   → 多數應該上榜
#     自動續訂關閉        流失率 32.25%   → 多數應該落在門檻下（p* = 48.4%）
#     自動續訂開、沒取消  流失率  0.60%   → 不應該有人上榜
#     首次到期的新客      流失率 39.84%   → 分散在門檻兩側
#     非月租方案          395 天 77.80%   → 高風險
#
# **那個「32.25% 的一群人多數不上榜」正是這一頁要講的事**：風險高與值得花錢是
# 兩件事，而分界線在 48.4%。如果調整比例時連行為一起調，這個論點就假了。
#
# ## 日期依訂閱週期推，不是隨機
#
# cutoff 是到期日前 7 天，上一次扣款在到期日前 `plan_days` 天，所以
# `days_since_last_tx = plan_days − 7`。這一欄是模型第三重要的特徵（15.5% gain），
# 隨機給的話會讓它講出一個與方案週期矛盾的故事 —— 而畫面上看不出來。
#
# ## 變異由 seed 固定
#
# 每次呼叫回同一批人。否則重新整理一次就換一批，截圖與說明對不起來，而
# 「模型每次給的答案不一樣」是這類 Demo 最容易被誤讀的地方。

BATCH_SEED = 20260818

# 評分時點落在 artifact 的 cohort.cutoff_window 內。
_CUTOFF_LO, _CUTOFF_HI = 20170125, 20170221

# README：`payment_plan_days` 30 天佔 97.10% 的用戶。非月租是獨立的一個原型，
# 不混進其他群 —— 混進去的話「非月租高風險」會蓋掉原型本身要示範的訊號。
_PLAN_NORMAL = ((30, 149.0), (30, 149.0), (30, 149.0), (30, 180.0), (30, 129.0))
_PLAN_EXOTIC = ((395, 1788.0), (90, 447.0), (7, 49.0))
_METHODS = (41, 40, 38, 36, 39, 34, 37)


def _shift(yyyymmdd: int, days: int) -> int:
    """日期加減，回 %Y%m%d。"""
    from datetime import date, timedelta

    d = date(yyyymmdd // 10000, yyyymmdd // 100 % 100, yyyymmdd % 100) + timedelta(days=days)
    return d.year * 10000 + d.month * 100 + d.day


def _pick_cutoff(rng) -> int:
    """在 cutoff_window 內挑一天。"""
    from datetime import date, timedelta

    lo = date(_CUTOFF_LO // 10000, _CUTOFF_LO // 100 % 100, _CUTOFF_LO % 100)
    hi = date(_CUTOFF_HI // 10000, _CUTOFF_HI // 100 % 100, _CUTOFF_HI % 100)
    d = lo + timedelta(days=rng.randint(0, (hi - lo).days))
    return d.year * 10000 + d.month * 100 + d.day


def _synth_logs(rng, intensity: float, trend: float) -> dict[str, float | None] | None:
    """由「活躍強度」與「近期趨勢」長出一組自洽的 38 欄收聽特徵。

    Args:
        intensity: 0~1，近 90 天的活躍比例。0 代表完全沒有紀錄（回 None）。
        trend: 近 7 天相對於 30 天日均的倍率。< 1 是在冷卻，> 1 是在升溫。

    四個窗口的 `active_days` 必須**隨窗口遞增**（它們是累計計數，不是各自獨立的
    數字）。手寫最容易錯的就是這件事，所以這裡由程式夾住；比率與趨勢一律交給
    `_log_block()` 用 `src/features/logs.py` 的同一組公式算。
    """
    if intensity <= 0:
        return None

    a90 = min(90, max(1, round(90 * intensity * rng.uniform(0.92, 1.08))))
    a30 = min(30, max(0, round(30 * intensity * rng.uniform(0.88, 1.12))))
    a7 = min(7, max(0, round(7 * intensity * trend * rng.uniform(0.85, 1.15))))
    a14 = min(14, max(a7, round(14 * intensity * (1.0 + trend) / 2)))
    a30 = max(a30, a14)  # 累計計數必須單調
    a90 = max(a90, a30)

    per_day_secs = rng.uniform(2600, 4600)
    plays_per_day = rng.uniform(14, 26)
    completion = rng.uniform(0.52, 0.86)
    unq_ratio = rng.uniform(0.58, 0.9)

    def block(active: int) -> dict[str, float]:
        secs = round(active * per_day_secs, 1)
        plays = round(active * plays_per_day)
        return {
            "active_days": active,
            "secs": secs,
            "plays": plays,
            "completed": round(plays * completion),
            "unq": round(plays * unq_ratio),
        }

    # 紅線 2：收聽紀錄不得晚於 cutoff，所以 min_days_before >= 0。
    if a7 > 0:
        min_before = rng.randint(0, 2)
    elif a30 > 0:
        min_before = rng.randint(8, 26)
    else:
        min_before = rng.randint(32, 84)

    return _log_block(
        {7: block(a7), 14: block(a14), 30: block(a30), 90: block(a90)},
        min_days_before=min_before,
        max_days_before=rng.randint(84, 89),
    )


# (人數, 標籤, cancel, auto_renew, 活躍強度, 近期趨勢, 年資月數, 方案組)
#
# ⚠️ `cancel=1` 一律配 `auto_renew=1` —— 沒開自動續訂的人不需要取消，那個組合在
# 99.2 萬人裡一筆都沒有。這裡是程式生成，所以那個約束由這張表**結構性地**保證，
# 不靠人工檢查（單筆範例那邊是靠註解提醒，踩過一次坑）。
_ARCHETYPES: tuple[tuple[int, str, int, int, tuple, tuple, tuple, str], ...] = (
    (8, "已按過取消", 1, 1, (0.0, 0.30), (0.0, 0.4), (6, 30), "normal"),
    (14, "自動續訂關閉", 0, 0, (0.10, 0.65), (0.3, 0.9), (4, 28), "normal"),
    (18, "自動續訂開、沒取消", 0, 1, (0.45, 1.0), (0.8, 1.25), (10, 30), "normal"),
    # README ④：新客整群平均 39.84%，但個體差很多。拆成兩組正是那一節的論點 ——
    # 「同樣是新客，誰該發、誰不用發」，照組別發優惠就不需要機器學習了。
    (3, "新客 · 自動續訂開", 0, 1, (0.0, 0.5), (0.0, 1.1), (1, 1), "normal"),
    (2, "新客 · 自動續訂關", 0, 0, (0.0, 0.4), (0.0, 0.9), (1, 1), "normal"),
    # README：無收聽紀錄者「多半是自動續訂的休眠訂戶 —— 不聽歌但錢照扣，所以
    # 不流失」。放在這裡是為了示範「沒有資料 ≠ 高風險」。
    (3, "休眠：完全沒有收聽紀錄", 0, 1, (0.0, 0.0), (1.0, 1.0), (8, 26), "normal"),
    (2, "非月租方案", 0, 0, (0.0, 0.45), (0.2, 0.8), (3, 14), "exotic"),
)


def build_demo_batch() -> list[dict[str, Any]]:
    """一批到期用戶（合成），給 Demo 頁丟進 `POST /predict/batch`。

    Returns:
        每人一個 `{"id", "segment", "features", "logs"}`。**不含機率** —— 分數
        一律由服務即時算，這一頁上沒有任何預先算好的數字（SPEC §7.14）。
    """
    import random

    rng = random.Random(BATCH_SEED)
    out: list[dict[str, Any]] = []
    seq = 0

    for count, label, cancel, auto_renew, act, trend, tenure, plan_grp in _ARCHETYPES:
        pool = _PLAN_EXOTIC if plan_grp == "exotic" else _PLAN_NORMAL
        for _ in range(count):
            seq += 1
            cutoff = _pick_cutoff(rng)
            months = rng.randint(*tenure)
            plan_days, list_price = pool[rng.randrange(len(pool))]
            is_new = months <= 1

            # cutoff = 到期日 − 7，上一次扣款在到期日 − plan_days
            #   → 正常續訂的 days_since_last_tx = plan_days − 7
            # 取消的人最後一筆是那筆取消，發生在扣款之後、到期日之前，所以更近。
            cycle_gap = max(1, plan_days - 7)
            gap = (
                rng.randint(1, max(2, cycle_gap - 1))
                if cancel
                else max(1, cycle_gap + rng.randint(-2, 2))
            )
            last_tx = _shift(cutoff, -gap)
            first_tx = last_tx if is_new else _shift(last_tx, -(months * plan_days))
            n_tx = 1 if is_new else max(2, months + rng.randint(-1, 2))

            logs = _synth_logs(rng, rng.uniform(*act), rng.uniform(*trend))
            has_bd = rng.random() > 0.45

            features: dict[str, Any] = {
                "cutoff": cutoff,
                "n_tx": n_tx,
                "first_tx": first_tx,
                "last_tx": last_tx,
                # 取消次數不能超過交易筆數，而且沒取消的人手上這筆不算 ——
                # 否則會長出「1 筆交易、取消佔比 100%，但最後一筆不是取消」
                # 這種讀起來合理、實際上不可能的列（實際生成過一位）。
                "n_cancel_hist": (
                    min(rng.randint(1, 3), n_tx)
                    if cancel
                    else min(rng.choice([0, 0, 0, 1]), n_tx - 1)
                ),
                "mean_paid": round(list_price * rng.uniform(0.92, 1.0), 1),
                "last_is_cancel": cancel,
                "last_is_auto_renew": auto_renew,
                # 取消不是一次收費，實付為 0。
                "last_actual_amount_paid": 0.0 if cancel else list_price,
                "last_plan_list_price": list_price,
                "last_payment_plan_days": plan_days,
                "last_payment_method_id": _METHODS[rng.randrange(len(_METHODS))],
                "city": rng.choice([1, 1, 5, 13, 15, 22, 4, 12]),
                # bd 為 0 是原始資料的無效值，不是年齡 —— 服務不替呼叫端猜。
                "bd": rng.randint(18, 46) if has_bd else 0,
                "gender": rng.choice(["male", "female"]) if rng.random() > 0.65 else None,
                "registered_via": rng.choice([9, 7, 3, 4]),
                "registration_init_time": _shift(first_tx, -rng.randint(0, 40)),
            }
            out.append({"id": f"u_{seq:04d}", "segment": label, "features": features, "logs": logs})

    return out
