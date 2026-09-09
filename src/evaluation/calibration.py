"""M4 · 機率校準的**診斷**工具（不 fit 任何東西）。

SPEC §6.2 把 reliability diagram 列為交付圖表之一，理由寫得很直白：
「沒有校準，上面那條曲線算出來的錢是假的。」§6.1 的決策公式是

    E[淨收益] = p_churn × r_save × LTV_saved − C_offer

`p_churn` 是**乘上去**的。實測 M2 模型在 Mar cohort 的平均預測是 6.49%，
實際流失率 8.99% —— 低估 28%。這個偏差會等比例地縮小每個人的期望收益，
讓投放門檻設得過於保守。

## 為什麼這個模組刻意不含 fit

校準器要 fit 在哪一塊資料，是 M4 最容易做錯的決定（見 `scripts/
calibration_report.py` 的模組註解）。把診斷與擬合分開，好處是**診斷可以先跑**：
先看失準長什麼形狀，再決定用 isotonic 還是 Platt。反過來先挑方法再看圖，
就變成用預設值決定方法。

因此本模組只有「量測」：Brier、reliability 曲線、ECE。全部是純函式，
餵 y 與 p 進去就有答案，不需要訓練資料、不會產生任何狀態。

## 分箱：為什麼預設是等量而不是等寬

流失率只有 9%，預測機率高度集中在 0 附近。等寬分箱（把 [0,1] 切成 10 段）
會讓九成以上的樣本落進第一箱，圖上只剩一個點有意義 —— 那張圖看起來很平，
但那是分箱造成的，不是模型準。

等量分箱（每箱樣本數相同）把解析度放在資料真正在的地方。代價是每箱的
寬度不一，x 軸不再等距 —— 所以畫圖時要用「該箱的平均預測值」當 x 座標，
不能用箱號。

兩種都提供，因為它們回答不同的問題：等寬看「整個機率尺度上哪裡失準」，
等量看「大多數樣本所在的區域準不準」。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import polars as pl

# 預設箱數。10 是慣例；箱太多會讓每箱樣本數不足、觀測頻率本身抖動，
# 圖上看到的鋸齒是抽樣雜訊而不是失準。
DEFAULT_BINS = 10

# 每箱至少要有這麼多樣本，否則該箱的觀測頻率不可信。
# 以 9% 的流失率計，1000 筆約含 90 個正例，頻率的標準誤約 0.9 個百分點。
MIN_BIN_COUNT = 1000


def brier_score(y_true: Iterable, y_pred: Iterable) -> float:
    """Brier score：預測機率與 0/1 標籤的均方誤差。

        Brier = mean((p - y)^2)

    與 log loss 的差別在於**懲罰的形狀**：log loss 對「非常有把握卻錯了」
    的懲罰是無上限的（趨近無限大），Brier 最多就是 1。所以 Brier 比較能
    反映「整體上偏差多少」，log loss 比較能抓出「有沒有災難性的過度自信」。

    兩個都報是因為它們會不一致 —— 一個把所有預測往中間壓的模型，Brier 可能
    變好而 log loss 變差。只看一個會被騙。
    """
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.asarray(list(y_pred), dtype=np.float64)
    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法計算 Brier score")
    return float(np.mean((p - y) ** 2))


def calibration_in_the_large(y_true: Iterable, y_pred: Iterable) -> dict[str, float]:
    """最粗但最重要的一個數字：平均預測 vs 實際發生率。

    SPEC §6.1 的公式對整體平均最敏感 —— 如果平均預測比實際低 28%，
    那麼**每一個人**的期望收益都被等比例縮小，投放門檻整條線都跟著偏。

    reliability diagram 上這對應「整條曲線往哪一邊偏移」，而不是形狀。
    兩者要分開看：形狀歪是排序問題，整體偏移是基準率問題，成因不同、
    解法也不同。
    """
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.asarray(list(y_pred), dtype=np.float64)
    actual, predicted = float(y.mean()), float(p.mean())
    return {
        "實際流失率": actual,
        "平均預測": predicted,
        "偏差": predicted - actual,
        "相對偏差": predicted / actual - 1 if actual else float("nan"),
    }


def reliability_curve(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    n_bins: int = DEFAULT_BINS,
    strategy: str = "quantile",
) -> pl.DataFrame:
    """把預測分箱，回報每箱的平均預測與實際發生率。

    Args:
        n_bins:   箱數。
        strategy: "quantile" 每箱樣本數相同（預設，理由見模組註解）；
                  "uniform" 把 [0, 1] 等寬切開。

    Returns:
        每箱一列：bin / 樣本數 / 平均預測 / 實際流失率 / 差距 / 下界 / 上界。
        **樣本數不足 MIN_BIN_COUNT 的箱會保留但標記出來** —— 直接丟掉會讓
        圖看起來比實際乾淨，而「這一段沒有足夠資料下判斷」本身就是結論。

    Raises:
        ValueError: 長度不符、空輸入、或未知的 strategy。
    """
    y = np.asarray(list(y_true), dtype=np.float64)
    p = np.asarray(list(y_pred), dtype=np.float64)
    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法畫 reliability 曲線")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        # 預測值高度集中時，相鄰分位數可能相同，會產生空箱。
        # 去重之後箱數可能少於要求的 n_bins —— 那是資料的事實，不是錯誤。
        edges = np.unique(edges)
    else:
        raise ValueError(f"未知的分箱策略 {strategy!r}，只接受 'quantile' 或 'uniform'")

    # right=False 讓區間是 [lo, hi)，最後一箱另外把上界包進來，
    # 否則預測值剛好等於最大值的那些樣本會落到一個不存在的箱。
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append(
            {
                "bin": b + 1,
                "下界": float(edges[b]),
                "上界": float(edges[b + 1]),
                "樣本數": n,
                "平均預測": float(p[mask].mean()),
                "實際流失率": float(y[mask].mean()),
                "差距": float(p[mask].mean() - y[mask].mean()),
                "樣本足夠": n >= MIN_BIN_COUNT,
            }
        )
    return pl.DataFrame(rows)


def expected_calibration_error(curve: pl.DataFrame) -> float:
    """ECE：各箱 |平均預測 − 實際| 以樣本數加權的平均。

    把 reliability diagram 壓成一個數字，方便在報表裡比較「校準前 vs 校準後」。

    ⚠️ **ECE 會隱藏方向。** 一個在低機率端高估、在高機率端低估的模型，
    兩邊的誤差不會互相抵銷（取了絕對值），但也看不出是哪一邊出問題。
    所以 ECE 只能當摘要，不能取代曲線本身 —— 這也是為什麼
    `calibration_in_the_large()` 要單獨報一個**帶正負號**的偏差。
    """
    required = {"樣本數", "平均預測", "實際流失率"}
    missing = required - set(curve.columns)
    if missing:
        raise KeyError(f"reliability 曲線缺少欄位：{sorted(missing)}")
    if curve.height == 0:
        raise ValueError("空的曲線無法計算 ECE")

    n = curve["樣本數"].to_numpy().astype(np.float64)
    gap = np.abs(curve["平均預測"].to_numpy() - curve["實際流失率"].to_numpy())
    return float((n * gap).sum() / n.sum())


def max_calibration_error(curve: pl.DataFrame) -> float:
    """MCE：樣本數足夠的箱裡，最大的 |平均預測 − 實際|。

    只看樣本足夠的箱：小箱的觀測頻率本身就在抖，用它當「最糟的一箱」
    量到的是抽樣雜訊。
    """
    usable = curve.filter(pl.col("樣本足夠")) if "樣本足夠" in curve.columns else curve
    if usable.height == 0:
        return float("nan")
    return float((usable["平均預測"] - usable["實際流失率"]).abs().max())
