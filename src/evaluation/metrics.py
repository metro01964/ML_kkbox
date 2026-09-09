"""評估指標。

兩條規定驅動了這個模組的存在：

**紅線 8** —— 本地評估必須套用官方的 clip(1e-15, 1-1e-15)，否則本地分數與
Kaggle LB 不可比。不 clip 的話，一個 p=0 但實際 y=1 的預測會讓 log loss
變成無限大，整個評估失去意義。把 clip 寫進函式裡，就沒有人會忘記。

**SPEC §4.5** —— M1 起每一次評估都必須回報三個數字，不能只報總分：

    log loss @ 全 Mar cohort         對應官方指標，headline
    log loss @ 重複用戶子群 (90.81%)   主力營收來源
    log loss @ 新進用戶子群 (9.19%)    高風險、高不確定性

兩群的流失率差 6.8 倍（5.87% vs 39.84%）。只報總分會被 90.81% 的低風險
用戶稀釋，掩蓋模型在新進用戶上的失效。
"""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

# 官方提交前的 clip 邊界（SPEC §3.1）。
EPS = 1e-15


def log_loss(y_true: Iterable, y_pred: Iterable, *, eps: float = EPS) -> float:
    """官方定義的 log loss（自然對數），含強制 clip。

        logloss = -(1/N) Σ [ y·ln(p) + (1-y)·ln(1-p) ]

    Args:
        y_true: 真實標籤，0 或 1。
        y_pred: 預測機率，[0, 1]。
        eps:    clip 邊界，預設為官方的 1e-15。

    Returns:
        平均 log loss。越小越好。

    Raises:
        ValueError: 長度不符或為空。
    """
    y = pl.Series("y", y_true, dtype=pl.Float64)
    p = pl.Series("p", y_pred, dtype=pl.Float64)

    if y.len() != p.len():
        raise ValueError(f"長度不符：y_true {y.len()} 筆，y_pred {p.len()} 筆")
    if y.len() == 0:
        raise ValueError("空的輸入無法計算 log loss")

    # 這一行就是紅線 8。少了它，p 剛好等於 0 或 1 時 ln 會發散成 inf。
    p = p.clip(eps, 1 - eps)

    return float((-(y * p.log() + (1 - y) * (1 - p).log())).mean())


def constant_log_loss(p_const: float, y_true: Iterable, *, eps: float = EPS) -> float:
    """對所有人預測同一個機率時的 log loss。

    這是 M1 的驗收門檻（SPEC §3.3）：用 Feb cohort 的流失率 6.3923% 對
    Mar cohort 做常數預測，實測 0.30746。打不贏它的模型沒有存在意義 ——
    因為那代表模型學到的東西還不如「大家風險都一樣」這個假設。
    """
    y = pl.Series("y", y_true, dtype=pl.Float64)
    return log_loss(y, pl.Series("p", [p_const] * y.len(), dtype=pl.Float64), eps=eps)


def roc_auc(y_true: Iterable, y_pred: Iterable) -> float:
    """AUC —— 只看排序，完全不看機率的絕對值。

    M1–M3 沒有用到它（官方指標是 log loss，排序好但機率歪的模型不該被獎勵）。
    M4 需要它的理由很具體：**校準與排序必須分開量**。

    isotonic 會把失準的區段壓平成同一個值，壓平就是製造同分，而 AUC 是唯一
    對同分有明確約定的指標：各給一半（等價於 ROC 曲線上走一段斜線）。log loss
    做不到這件事 —— 它同時受校準與排序影響，兩者混在同一個數字裡，改善了也
    說不出是哪一邊的功勞。

    ⚠️ **校準後 AUC 上升不代表排序變好。** 單調映射不會倒轉任何一對的順序，
    所以變化全部來自同分；而同分讓原本得 0 分的**反轉配對**變成 0.5 分。
    被壓平的那一段原本排錯了，AUC 就會上升 —— 那是「誠實地宣告不知道」，
    不是「學到了更多」。反過來，壓平原本排對的區段才會讓 AUC 下降。
    測試 `test_flattening_an_inversion_raises_auc_instead_of_lowering_it`
    用手算的數字釘住這件事。
    """
    from sklearn.metrics import roc_auc_score

    y = pl.Series("y", y_true, dtype=pl.Float64)
    p = pl.Series("p", y_pred, dtype=pl.Float64)
    if y.len() != p.len():
        raise ValueError(f"長度不符：y_true {y.len()} 筆，y_pred {p.len()} 筆")
    if y.n_unique() < 2:
        raise ValueError("只有單一類別時 AUC 沒有定義")
    return float(roc_auc_score(y.to_numpy(), p.to_numpy()))


def repeat_vs_new(msno: Iterable, previous_msno: Iterable) -> pl.Series:
    """把用戶標記成「重複用戶」或「新進用戶」（SPEC §4.5 的分群定義）。

    Args:
        msno:          本期 cohort 的用戶 ID。
        previous_msno: 上一期 cohort 的用戶 ID。

    Returns:
        與 msno 等長的 Series，值為「重複用戶」或「新進用戶」。

    ⚠️ **這個結果只能用於評估切分，絕對不能當特徵。**

    SPEC §5.1 避雷清單明列：「別讓『是否出現在上一期標籤檔』變成特徵。
    它在訓練集上威力驚人（6.8 倍差距），但測試集（Apr cohort）沒有對應的
    上一期標籤檔，部署時算不出來。」

    為什麼拿來切報表就沒問題：報表是事後診斷，不是模型輸入。模型看不到它，
    只有你在看模型表現時看得到。要在特徵裡表達「是不是新客」，必須改用
    as-of 的交易史長度（src.data.cohort 的 n_tx / first_tx）來代理。
    """
    prev = pl.Series("prev", previous_msno)
    return (
        pl.DataFrame({"msno": pl.Series("msno", msno)})
        .select(
            pl.when(pl.col("msno").is_in(prev))
            .then(pl.lit("重複用戶"))
            .otherwise(pl.lit("新進用戶"))
            .alias("segment")
        )
        .get_column("segment")
    )


def segment_report(
    df: pl.DataFrame,
    *,
    segment_col: str,
    y_col: str = "is_churn",
    p_col: str = "p_churn",
    overall_label: str = "全體",
) -> pl.DataFrame:
    """分群回報 log loss，最後一列是全體（SPEC §4.5 要求的三個數字）。

    Args:
        df:          含標籤、預測機率、分群欄位的表。
        segment_col: 分群欄位名。
        y_col:       標籤欄位名。
        p_col:       預測機率欄位名。
        overall_label: 全體那一列的標籤。

    Returns:
        每個分群一列，加上全體一列。欄位：
            分群 / 人數 / 佔比 / 實際流失率 / 平均預測機率 / log_loss

    「實際流失率」與「平均預測機率」並列是刻意的：兩者差很多就代表機率
    沒有校準。這是 M4 reliability diagram 的窮人版，但足以在 M1 就發現問題。
    """
    for col in (segment_col, y_col, p_col):
        if col not in df.columns:
            raise KeyError(f"找不到欄位 {col!r}。現有欄位：{df.columns}")

    total = df.height
    rows = []

    for (name,), part in sorted(df.group_by(segment_col), key=lambda kv: str(kv[0][0])):
        rows.append(
            {
                "分群": str(name),
                "人數": part.height,
                "佔比": part.height / total,
                "實際流失率": float(part[y_col].mean()),
                "平均預測機率": float(part[p_col].mean()),
                "log_loss": log_loss(part[y_col], part[p_col]),
            }
        )

    rows.append(
        {
            "分群": overall_label,
            "人數": total,
            "佔比": 1.0,
            "實際流失率": float(df[y_col].mean()),
            "平均預測機率": float(df[p_col].mean()),
            "log_loss": log_loss(df[y_col], df[p_col]),
        }
    )

    return pl.DataFrame(rows)
