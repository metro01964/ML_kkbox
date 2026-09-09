"""M4 · 把流失機率換算成投放決策（SPEC §6.1）。

    E[淨收益] = p_churn × r_save × LTV_saved − C_offer

## 三個參數只透過一個比值影響「該投放給誰」

投放一個人划不划算的條件是 `E > 0`，移項之後：

    p* = C_offer / (r_save × LTV_saved)

**門檻只取決於這一個數字。** 這件事有三個後果，決定了本模組的形狀：

1. `r_save` 加倍與 `C_offer` 減半是同一件事 —— 敏感度熱圖實際上是 `p*` 的
   等高線圖，很多 (r_save, C_offer) 組合會給出完全相同的決策。
2. 「LTV 該用營收還是毛利」這個爭論被吸收掉了：LTV 打三折等於 `C_offer`
   乘以 3.33，已經落在掃描區間裡，不必另開一個參數。
3. 期望淨收益曲線的極大值**必然**落在「最後一個 `p > p*` 的人」身上 ——
   因為排在他後面的每個人都讓總額變小。這是恆等式不是巧合，所以
   `campaign_curve()` 的極大值可以拿來驗證實作（見測試）。

## 兩條曲線都是模擬，差別只在「機率從哪來」

    期望模擬淨收益      p 用**模型預測** —— 模型以為會賺多少
    標籤結算模擬淨收益  p 用**真實標籤 y**（0/1）—— 把預測換成答案再算一次

⚠️ **兩條都不是真的賺到的錢，名字必須說清楚這件事。**

真實的只有 Mar 的流失標籤。`r_save`（挽回成功率）與 `LTV_saved` 仍然是假設：
本資料集沒有實驗組／對照組，**無法驗證投放是否真的改變了任何人的行為**
（SPEC §6.3 第一點）。所以第二條線叫「標籤結算模擬」而不是「實際」——
它回答的是「若挽回假設成立，且我們事先知道誰會流失，這組假設會算出多少」，
不是「我們賺到了多少」。

兩條線的落差量的是**機率誤差的代價**：模型低估流失風險 → 期望曲線比較悲觀
→ 極大值往左偏 → 投放門檻設得過於保守。SPEC §6.1 預告了這個方向。

⚠️ 標籤結算那條用到 Mar 的標籤，**只能事後回顧，不能拿來決定門檻** ——
部署時沒有它。

## 名單品質的三個對照組

SPEC §4.5 第 3 點：「若模型只是學會『新客風險高』，那挽回策略就退化成
『對所有新客發優惠』，不需要機器學習。」因此曲線要有對照：

    隨機排序        lift = 1 的水平線，證明排序本身有價值
    全部新進用戶    一個規則，不需要模型 —— 模型必須贏過它才有存在意義
    分群內排序      在同一群人裡面還能不能排出高低（這才是模型的價值）
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

# 每月天數。用 30.4（365.25 / 12）而不是 30，因為 `price_per_day` 是用實際
# 方案天數除出來的，換算回「月費」時多算或少算 1.3% 會直接進到 LTV。
DAYS_PER_MONTH = 30.4


def _as_array(values: Iterable) -> np.ndarray:
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(np.float64, copy=False)
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    return np.asarray(list(values), dtype=np.float64)


def decision_threshold(*, r_save: float, ltv_saved: float, c_offer: float) -> float:
    """投放門檻 `p* = C / (r × LTV)`。預測機率高於它才值得投放。

    Raises:
        ValueError: `r_save` 或 `ltv_saved` 為 0 —— 那代表「挽回一個人的收益
            是 0」，此時任何成本都不划算，門檻是無限大而不是一個數字。
            讓它報錯而不是回傳 inf，是因為 inf 會安靜地傳染到下游每個計算。
    """
    if r_save <= 0 or ltv_saved <= 0:
        raise ValueError(f"r_save（{r_save}）與 ltv_saved（{ltv_saved}）都必須為正")
    if c_offer < 0:
        raise ValueError(f"c_offer 不能為負：{c_offer}")
    return c_offer / (r_save * ltv_saved)


def expected_months(monthly_churn_rate: float) -> float:
    """挽回一位用戶之後，期望還會續訂幾個月。

        E[月數] = 1 / 月流失率

    這是幾何存活模型：假設每個月的流失風險相同（無記憶性）。

    ⚠️ **這個估計偏樂觀，而且方向已知。** 被挽回的人是我們特意挑出來的
    高風險用戶，他們往後的流失風險高於全體平均，實際續訂月數會比這個短。
    資料集沒有挽回實驗，無法校正 —— 只能標註方向。

    ⚠️ 傳入的流失率應該用**上一期 cohort** 的（部署時已知），不要用評估
    cohort 的：那是標籤，拿它設業務常數等於用到答案。
    """
    if not 0 < monthly_churn_rate < 1:
        raise ValueError(f"月流失率必須落在 (0, 1)：{monthly_churn_rate}")
    return 1.0 / monthly_churn_rate


def monthly_arpu(price_per_day: Iterable, days_per_month: float) -> float:
    """cohort 的平均月費，由 `price_per_day` 換算。

    用實付而非定價：定價是牌價，實付才是這位用戶真的貢獻的收入。
    null（同日多筆且金額衝突，見 §7.11）直接排除，不補值。

    Raises:
        ValueError: 全為 null，無法推估。
    """
    series = price_per_day if isinstance(price_per_day, pl.Series) else pl.Series(price_per_day)
    per_day = series.drop_nulls()
    if per_day.len() == 0:
        raise ValueError("price_per_day 全為 null，無法推估月費")
    return float(per_day.mean()) * days_per_month


@dataclass(frozen=True)
class Assumptions:
    """一組業務假設，以及由它們推導出來的投放門檻。

    ## 為什麼要有這個類別

    M4 的業務指標與 M5 的原因碼名單**必須用同一個 `p*`**，否則兩份交付物講的
    是不同的名單 —— 而它們都會印出「投放 4.8 萬人」這種看起來一致的句子。

    這是 `src.models.adopted` 同一個教訓的第二次：那次是「兩支腳本各自
    `load_model_config()`，於是校準結論是在一個不會上線的模型上得出的」。
    參數不是設定值而是**推導結果**時，推導只能有一份程式。
    """

    r_save: float
    c_offer: float
    monthly_arpu: float
    expected_months: float
    ltv_saved: float
    p_star: float
    months_source: str

    def summary(self) -> dict:
        """進 JSON 的欄位。兩支腳本共用這個方法，manifest 才逐欄可比。"""
        return {
            "r_save": self.r_save,
            "c_offer": self.c_offer,
            "ltv_saved": round(self.ltv_saved, 1),
            "monthly_arpu": round(self.monthly_arpu, 1),
            "expected_months": round(self.expected_months, 2),
            "months_source": self.months_source,
            "p_star": round(self.p_star, 4),
        }


def resolve_assumptions(
    biz: dict,
    *,
    price_per_day: Iterable,
    prior_churn_rate: float,
    months_source: str = "feb_churn_rate",
) -> Assumptions:
    """把 `configs/business.yaml` 加上 cohort 實測值，推成完整的一組假設。

    Args:
        biz: `business.yaml` 的 `[business]` 區段（`c_offer` / `r_save` /
            `days_per_month`）。
        price_per_day: **評估 cohort** 的日均單價，用來算月費。
        prior_churn_rate: **上一期**的月流失率。⚠️ 不可傳評估 cohort 的 ——
            那是標籤，拿它設業務常數等於用到答案（見 `expected_months`）。
        months_source: 這個流失率的來源，會寫進 manifest 供回溯。
    """
    for key in ("c_offer", "r_save", "days_per_month"):
        if key not in biz:
            raise KeyError(f"business 區段缺少 {key!r}")

    arpu = monthly_arpu(price_per_day, biz["days_per_month"])
    months = expected_months(prior_churn_rate)
    ltv = arpu * months
    return Assumptions(
        r_save=biz["r_save"],
        c_offer=biz["c_offer"],
        monthly_arpu=arpu,
        expected_months=months,
        ltv_saved=ltv,
        p_star=decision_threshold(r_save=biz["r_save"], ltv_saved=ltv, c_offer=biz["c_offer"]),
        months_source=months_source,
    )


def campaign_curve(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    r_save: float,
    ltv_saved: float,
    c_offer: float,
    step: float = 0.002,
) -> pl.DataFrame:
    """投放前 K% 的期望與標籤結算兩種模擬淨收益，K 從 `step` 掃到 1.0。

    Args:
        step: K 的解析度。預設 0.002 = 每 0.2 個百分點一個點（500 個點）。

    Returns:
        每個 K 一列：

            K / 投放人數 / 門檻機率 / 期望模擬淨收益 / 標籤結算模擬淨收益 /
            命中數 / 命中率 / lift

        「門檻機率」是該 K 之下最低的預測機率 —— 對應實際操作中的「發送
        名單的分數下限」，而不是一個抽象的比例。

    ⚠️ **同分的人會被任意切開。** 「前 K%」預設排序連續可切；預測機率有
    大量重複值時（例如套過 isotonic 之後），落在同分區塊裡的 K 沒有明確
    定義，這裡的作法是依排序後的位置硬切。未校準的預測有 84% 的相異值，
    影響很小；但這個假設本身要寫出來，因為它會安靜地成立或不成立。
    """
    y = _as_array(y_true)
    p = _as_array(y_pred)
    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法畫投放曲線")
    if not 0 < step <= 1:
        raise ValueError(f"step 必須落在 (0, 1]：{step}")

    n = y.size
    gain = r_save * ltv_saved  # 挽回一位「真的會流失」的用戶帶來的收益

    order = np.argsort(-p, kind="stable")  # 機率由高到低
    p_sorted, y_sorted = p[order], y[order]

    cum_p = np.cumsum(p_sorted)
    cum_y = np.cumsum(y_sorted)
    base_rate = float(y.mean())

    ks = np.arange(step, 1.0 + step / 2, step)
    idx = np.clip(np.round(ks * n).astype(np.int64), 1, n) - 1  # 取到第幾個人（0-based）
    sizes = idx + 1

    hits = cum_y[idx]
    precision = hits / sizes

    return pl.DataFrame(
        {
            "K": ks[: len(idx)],
            "投放人數": sizes,
            "門檻機率": p_sorted[idx],
            "期望模擬淨收益": cum_p[idx] * gain - sizes * c_offer,
            "標籤結算模擬淨收益": hits * gain - sizes * c_offer,
            "命中數": hits.astype(np.int64),
            "命中率": precision,
            "lift": precision / base_rate if base_rate else np.full(len(idx), np.nan),
        }
    )


def subset_calibration(y_true: Iterable, y_pred: Iterable, *, k: float) -> dict:
    """**投放名單自己的**校準偏差 —— 不是全體 cohort 的那一個。

    ## 為什麼需要這個函式

    「模型在 Mar 低估 27%」是一句關於**全體 cohort 平均預測流失率**的話。
    把它套到別的地方全都不成立：

        個別用戶        每個人的偏差不同
        個別風險區間    §7.10 的 reliability 曲線顯示各段差很大
        前 K% 名單      那是一個高機率子集，偏差與全體無關
        淨收益          `p × r × LTV − C` 只有第一項隨 p 縮放，
                        `C_offer` 是固定成本，不隨機率縮放

    最後一項最容易漏掉：機率低估 27%**不等於**淨收益低估 27%。

    這個函式讓報表可以誠實地說「這份名單自己的偏差是多少」，而不是把全體
    的數字借過來用。

    Args:
        k: 取預測機率最高的前 k 比例（0 < k <= 1）。

    Returns:
        人數 / 平均預測 / 實際流失率 / 偏差 / 相對偏差。
    """
    y = _as_array(y_true)
    p = _as_array(y_pred)
    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法計算子集校準")
    if not 0 < k <= 1:
        raise ValueError(f"k 必須落在 (0, 1]：{k}")

    order = np.argsort(-p, kind="stable")
    size = max(1, int(round(k * y.size)))
    idx = order[:size]

    actual, predicted = float(y[idx].mean()), float(p[idx].mean())
    return {
        "人數": size,
        "平均預測": predicted,
        "實際流失率": actual,
        "偏差": predicted - actual,
        "相對偏差": predicted / actual - 1 if actual else float("nan"),
    }


def optimal_point(curve: pl.DataFrame, *, by: str = "期望模擬淨收益") -> dict:
    """曲線的極大值那一列。

    Args:
        by: 依哪一欄取極大 —— 「期望模擬淨收益」是部署時唯一能用的依據，
            「標籤結算模擬淨收益」用到答案，只能事後回顧。
    """
    if by not in curve.columns:
        raise KeyError(f"找不到欄位 {by!r}。現有欄位：{curve.columns}")
    if curve.height == 0:
        raise ValueError("空的曲線沒有極大值")
    return curve.row(int(curve[by].arg_max()), named=True)


def fixed_rule_point(
    y_true: Iterable,
    mask: Iterable,
    *,
    r_save: float,
    ltv_saved: float,
    c_offer: float,
    label: str,
) -> dict:
    """一條**不需要模型**的規則的成績（例如「投放給所有新進用戶」）。

    模型必須贏過這種規則才有存在意義 —— SPEC §4.5 第 3 點。回傳格式與
    `campaign_curve()` 的一列相同，方便直接畫在同一張圖上。
    """
    y = _as_array(y_true)
    m = (
        np.asarray(list(mask), dtype=bool)
        if not isinstance(mask, np.ndarray)
        else mask.astype(bool)
    )
    if y.size != m.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，mask {m.size} 筆")
    size = int(m.sum())
    if size == 0:
        raise ValueError(f"規則「{label}」沒有選中任何人")

    hits = float(y[m].sum())
    base_rate = float(y.mean())
    precision = hits / size
    return {
        "規則": label,
        "K": size / y.size,
        "投放人數": size,
        "標籤結算模擬淨收益": hits * r_save * ltv_saved - size * c_offer,
        "命中數": int(hits),
        "命中率": precision,
        "lift": precision / base_rate if base_rate else float("nan"),
    }


def sensitivity_grid(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    ltv_saved: float,
    r_values: Sequence[float],
    c_values: Sequence[float],
) -> pl.DataFrame:
    """`r_save` × `C_offer` 網格上，最佳投放比例與該點的標籤結算模擬淨收益。

    每一格的決策都照部署時的作法算：門檻 `p* = C/(r×LTV)`，投放所有
    `p > p*` 的人。**標籤結算模擬淨收益**則把預測換成真實標籤再算一次 ——
    仍然是模擬（`r_save` 與 LTV 是假設），只是機率換成了答案。

    Returns:
        每格一列：r_save / c_offer / p* / 最佳投放比例 / 投放人數 /
        期望模擬淨收益 / 標籤結算模擬淨收益。
    """
    y = _as_array(y_true)
    p = _as_array(y_pred)
    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法做敏感度分析")

    order = np.argsort(-p, kind="stable")
    p_sorted, y_sorted = p[order], y[order]
    cum_p = np.concatenate([[0.0], np.cumsum(p_sorted)])
    cum_y = np.concatenate([[0.0], np.cumsum(y_sorted)])
    n = y.size

    rows = []
    for r in r_values:
        for c in c_values:
            star = decision_threshold(r_save=r, ltv_saved=ltv_saved, c_offer=c)
            # p_sorted 遞減，所以「有幾個人 p > p*」= 第一個不滿足的位置。
            size = int(np.searchsorted(-p_sorted, -star, side="left"))
            rows.append(
                {
                    "r_save": r,
                    "c_offer": c,
                    "p*": star,
                    "最佳投放比例": size / n,
                    "投放人數": size,
                    "期望模擬淨收益": cum_p[size] * r * ltv_saved - size * c,
                    "標籤結算模擬淨收益": cum_y[size] * r * ltv_saved - size * c,
                }
            )
    return pl.DataFrame(rows)
