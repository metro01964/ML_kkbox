"""M3 · Null importance 特徵篩選。

SPEC §7 的 M3 第二項交付物：「null importance 特徵篩選 + 篩選前後對照」。
動機來自 §7.3 的實測 —— 38 個收聽特徵只貢獻 10.99% 的 gain，「其中多數很
可能是雜訊」。這個模組把「可能」變成可驗證的數字。

## 為什麼不能直接砍 gain 低的特徵

gain 低不代表沒用，gain 高也不代表有用。**一個純粹的隨機欄位在
LightGBM 裡的 gain 幾乎不會是 0** —— 只要它有夠多的相異值，樹就總能在它
身上找到一些看似有利的切點，然後把雜訊擬合進去。高基數的特徵尤其嚴重：
`mean_paid`、`tenure_days` 這種連續欄位有幾十萬個候選切點，隨機也能切出
東西；`is_free_plan` 這種二值欄位只有一個切點，先天吃虧。

所以「gain > 某個門檻」這個規則同時偏袒高基數特徵、又無法分辨真訊號與
過擬合。**要判斷一個 gain 值大不大，得先知道「完全沒有訊號時它會有多大」。**

## Null importance 的作法

    1. 用真實標籤訓練一次        → 每個特徵的 actual gain
    2. 把標籤**整欄打亂**，重訓 N 次 → 每個特徵的 null gain 分布
    3. actual gain 明顯超出自己的 null 分布 → 這個特徵有真訊號

標籤打亂之後，特徵與標籤之間**所有**關聯都被破壞，但特徵本身的基數、
分布、與其他特徵的相關性全部保留。因此 null 分布量到的正是「這個特徵靠
過擬合能拿到多少 gain」。每個特徵跟**自己的**基準比，高基數的偏袒就被
抵銷掉了。

## 三個容易做錯的地方

**一、null 分布必須用固定輪數，不能 early stopping。** 標籤是亂的，
early stopping 會在第一輪就停，什麼都量不到。輪數取真實訓練找出來的
best_iteration，讓 null 與 actual 在同樣的預算下比較。

**二、全程只用 Feb cohort，Mar 一次都不能碰。** 用 Mar 的分數挑特徵、
再用 Mar 回報成績，回報的就是「在這批資料上挑得多好」而不是泛化能力。
篩選是訓練的一部分，必須關在訓練集裡。

**三、打亂的是標籤，不是特徵。** 打亂特徵欄會破壞欄與欄之間的相關結構，
量到的 null 分布就對應不到真實情境。`numpy` 的 `permutation` 作用在
y 上，X 一格都不動。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from src.evaluation import log_loss, repeat_vs_new, segment_report
from src.features import FeatureSet
from src.models.candidates import fit_lightgbm, to_lgb_arrays

# 篩選門檻的掃描範圍：保留 actual gain 超過自身 null 分布第 p 百分位的特徵。
#
# 100 代表「必須贏過所有 null 執行」，是最嚴格的；50 幾乎不篩。全部跑一遍
# 而不是只挑一個，是因為「該砍多少」沒有先驗答案，掃描出來的曲線本身就是
# 要寫進報告的結果 —— 它顯示分數對特徵數有多不敏感。
DEFAULT_PERCENTILES: tuple[int, ...] = (50, 75, 90, 95, 99, 100)


@dataclass
class NullImportance:
    """一次 null importance 分析的原始輸出。"""

    actual: pl.DataFrame  # feature / actual_gain
    null_gains: pl.DataFrame  # feature / run / gain（長格式，n_runs × n_features 列）
    n_runs: int
    num_boost_round: int

    def summary(self) -> pl.DataFrame:
        """每個特徵一列：actual gain、null 分布的分位數、以及分數。

        `score` 用的是 null importance 的慣用定義：

            score = log(1e-10 + actual_gain / (1 + null_gain 的第 75 百分位))

        取對數是為了把「贏過 100 倍」與「贏過 2 倍」壓到可讀的尺度；
        分母 +1 避免 null 幾乎為 0 的特徵得到無限大的分數。
        **score 只用來排序，絕對值沒有物理意義** —— 要判斷去留請看
        `beats_pct`（actual 贏過幾成的 null 執行），那個數字直接可讀。
        """
        stats = self.null_gains.group_by("feature").agg(
            pl.col("gain").median().alias("null_p50"),
            pl.col("gain").quantile(0.75).alias("null_p75"),
            pl.col("gain").quantile(0.95).alias("null_p95"),
            pl.col("gain").max().alias("null_max"),
        )
        beats = (
            self.null_gains.join(self.actual, on="feature")
            .group_by("feature")
            .agg((pl.col("gain") < pl.col("actual_gain")).mean().alias("beats_pct"))
        )
        return (
            self.actual.join(stats, on="feature")
            .join(beats, on="feature")
            .with_columns(
                (1e-10 + pl.col("actual_gain") / (1 + pl.col("null_p75"))).log().alias("score")
            )
            .sort("score", descending=True)
        )

    def keep(self, percentile: int) -> list[str]:
        """保留 actual gain 超過自身 null 分布第 `percentile` 百分位的特徵。

        用 numpy 的線性插值分位數（與 `summary()` 的 polars quantile 預設
        一致），percentile=100 等同於「超過所有 null 執行的最大值」。
        """
        q = (
            self.null_gains.group_by("feature")
            .agg(pl.col("gain").quantile(percentile / 100).alias("threshold"))
            .join(self.actual, on="feature")
        )
        kept = q.filter(pl.col("actual_gain") > pl.col("threshold"))["feature"].to_list()
        # 依原始欄位順序回傳，讓下游的特徵矩陣欄序穩定（欄序會影響 LightGBM
        # 的類別索引與隨機抽樣的結果，不固定就無法重現）。
        order = self.actual["feature"].to_list()
        return [f for f in order if f in set(kept)]


def _fit_gains(
    X: np.ndarray,
    y: np.ndarray,
    cat_idx: list[int],
    names: list[str],
    params: dict[str, Any],
    num_boost_round: int,
) -> dict[str, float]:
    """訓練一次、回傳每個特徵的 gain。null 與 actual 走的是同一條路徑。"""
    import lightgbm as lgb

    dtrain = lgb.Dataset(X, y, categorical_feature=cat_idx, free_raw_data=False)
    booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
    return dict(zip(names, booster.feature_importance("gain"), strict=True))


def null_importance(
    feb: FeatureSet,
    params: dict[str, Any],
    *,
    num_boost_round: int,
    n_runs: int,
    seed: int,
    verbose: bool = True,
) -> NullImportance:
    """在 Feb cohort 上做 null importance 分析。

    Args:
        num_boost_round: 固定輪數。取真實訓練的 best_iteration，讓 actual 與
                         null 在相同預算下比較。
        n_runs:          打亂標籤重訓幾次。20 次足以估到第 95 百分位；
                         要用第 99 百分位以上的門檻建議提高到 50 以上，
                         否則那個分位數其實是被最大值決定的。
        seed:            打亂標籤的隨機種子。

    Note:
        模型本身的 seed **不隨執行變動**（沿用 params 裡的固定值）。這是刻意的：
        兩次執行之間唯一的差別就只有「標籤有沒有被打亂」，量到的 null 分布
        才乾淨地對應「沒有真訊號時的 gain」。若連模型 seed 也變，分布裡就
        混進了 bagging 的隨機性，兩種來源分不開。
    """
    X, y, cat_idx = to_lgb_arrays(feb)
    names = feb.X.columns

    def log(msg: str) -> None:
        if verbose:
            print(msg, flush=True)

    log(f"真實標籤訓練（固定 {num_boost_round} 輪）...")
    actual = _fit_gains(X, y, cat_idx, names, params, num_boost_round)

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for run in range(1, n_runs + 1):
        # 只打亂 y。X 一格都不動，欄與欄之間的相關結構完整保留。
        y_shuffled = rng.permutation(y)
        gains = _fit_gains(X, y_shuffled, cat_idx, names, params, num_boost_round)
        rows.extend({"feature": f, "run": run, "gain": float(g)} for f, g in gains.items())
        log(f"  null run {run}/{n_runs}")

    return NullImportance(
        actual=pl.DataFrame({"feature": names, "actual_gain": [float(actual[f]) for f in names]}),
        null_gains=pl.DataFrame(rows),
        n_runs=n_runs,
        num_boost_round=num_boost_round,
    )


# ---------------------------------------------------------------------------
# 篩選前後對照
# ---------------------------------------------------------------------------
#
# ⚠️ **門檻是在 Feb 內部選的，Mar 只看兩次。**
#
# 一開始的寫法是「每個門檻都在 Mar 上評分，挑分數最低的那個」。那是
# selection bias：Mar 同時當了選模集與最終成績單，回報的數字就不是「這個
# 模型有多好」，而是「六個門檻裡最合這批資料胃口的那個有多好」—— 必然
# 偏樂觀，而且偏多少無從得知。
#
# 現在的作法與 `src/models/tuning.py` 一致，Feb 切成三塊：
#
#     train (70%)  訓練
#     es    (15%)  early stopping —— 決定停在第幾輪
#     sel   (15%)  選門檻 —— 決定砍到剩幾個特徵
#
# null importance 也只在 `train` 上做（不是整個 Feb），否則門檻的來源就
# 看過了 sel。門檻凍結之後，才在 Mar 上評估**兩次**：全特徵與贏家。
# 這兩次是 SPEC §7 M3 指定的「篩選前後對照」交付物，在看到分數之前就
# 決定要算哪兩個，因此不構成挑選。


@dataclass
class SelectionRun:
    """在某個門檻下重訓一次的結果。

    `sel_logloss` 是**選門檻用的**分數（Feb 內部）。`mar_logloss` 只有被
    凍結下來的那幾個 run 才會有值 —— 兩個欄位分開，是為了讓「哪個分數
    參與了決定」在型別上就看得出來，而不是靠讀程式碼推斷。
    """

    label: str
    # 給機器用的名字。MLflow 的 metric 名稱只接受英數與 _ - . 空白 /，
    # 中文的 label 直接送進去會被拒絕（實測 MlflowException）。
    key: str
    n_features: int
    best_iteration: int
    sel_logloss: float
    dropped: list[str]
    mar_logloss: float | None = None
    segments: pl.DataFrame | None = None

    def segment_score(self, name: str) -> float:
        if self.segments is None:
            return float("nan")
        row = self.segments.filter(pl.col("分群") == name)
        return float(row["log_loss"][0]) if row.height else float("nan")


def fit_subset(
    train: FeatureSet,
    es: FeatureSet,
    keep: list[str],
    params: dict[str, Any],
    train_cfg: dict,
):
    """用指定的特徵子集訓練一個 LightGBM。

    每個門檻都**重新 early stopping**：特徵少了之後最佳輪數本來就會變，
    沿用原本的輪數等於讓小特徵集背著別人的超參數上場。
    """
    return fit_lightgbm(train.select(keep), es.select(keep), params, train_cfg)


def evaluate_subset(
    train: FeatureSet,
    es: FeatureSet,
    sel: FeatureSet,
    keep: list[str],
    params: dict[str, Any],
    train_cfg: dict,
    *,
    label: str,
    key: str,
    all_features: list[str],
) -> SelectionRun:
    """訓練並在 **Feb 內部的 sel** 上評分。這是選門檻唯一看的數字。"""
    fitted = fit_subset(train, es, keep, params, train_cfg)
    sub_sel = sel.select(keep)
    return SelectionRun(
        label=label,
        key=key,
        n_features=len(keep),
        best_iteration=fitted.best_iteration,
        sel_logloss=log_loss(sub_sel.y, fitted.predict(sub_sel.X)),
        dropped=[c for c in all_features if c not in set(keep)],
    )


def score_on_mar(
    run: SelectionRun,
    train: FeatureSet,
    es: FeatureSet,
    mar: FeatureSet,
    feb: FeatureSet,
    keep: list[str],
    params: dict[str, Any],
    train_cfg: dict,
) -> SelectionRun:
    """把一個**已經凍結**的 run 拿到 Mar cohort 上評估（SPEC §4.5 分群回報）。

    分開成獨立函式，是為了讓「這一步發生在門檻決定之後」在呼叫順序上
    一目了然 —— 而不是藏在某個同時做訓練與選擇的迴圈裡。
    """
    fitted = fit_subset(train, es, keep, params, train_cfg)
    sub_mar = mar.select(keep)
    pred = fitted.predict(sub_mar.X)
    scored = pl.DataFrame(
        {
            "is_churn": sub_mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(sub_mar.msno, feb.msno),
        }
    )
    run.mar_logloss = log_loss(sub_mar.y, pred)
    run.segments = segment_report(scored, segment_col="segment")
    return run


def threshold_sweep_table(runs: list[SelectionRun]) -> pl.DataFrame:
    """各門檻在 **Feb-sel** 上的掃描結果 —— 選門檻的依據。"""
    ref = runs[0].sel_logloss
    return pl.DataFrame(
        [
            {
                "門檻": r.label,
                "特徵數": r.n_features,
                "輪數": r.best_iteration,
                "Feb-sel log loss": round(r.sel_logloss, 5),
                "相對全特徵": round(r.sel_logloss / ref - 1, 5),
            }
            for r in runs
        ]
    )


def before_after_table(runs: list[SelectionRun]) -> pl.DataFrame:
    """SPEC §7 M3 要求的「篩選前後對照」表。

    只含有 Mar 分數的那幾個 run（全特徵與凍結的贏家）。
    """
    scored = [r for r in runs if r.mar_logloss is not None]
    ref = scored[0].mar_logloss
    return pl.DataFrame(
        [
            {
                "階段": r.label,
                "特徵數": r.n_features,
                "輪數": r.best_iteration,
                "Feb-sel log loss": round(r.sel_logloss, 5),
                "Mar log loss": round(r.mar_logloss, 5),
                "相對全特徵": round(r.mar_logloss / ref - 1, 5),
                "重複用戶": round(r.segment_score("重複用戶"), 5),
                "新進用戶": round(r.segment_score("新進用戶"), 5),
            }
            for r in scored
        ]
    )
