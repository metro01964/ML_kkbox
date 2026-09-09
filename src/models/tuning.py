"""M3 · LightGBM 超參數隨機搜尋。

`configs/model_lgbm.yaml` 的開頭寫著「這組是**未調參的合理起點**，不是最佳解。
調參是 M3 的工作」。這個模組就是那件工作。

## 為什麼調參不能看 Mar 分數

Mar cohort 是本專案唯一的時間外估計，而且它要被外推到測試集（Apr）。
每看它一次，它就少一分「沒被最佳化過」的身分。跑 30 組超參數、挑 Mar 分數
最低的那組，回報的就不是「這個模型有多好」，而是「30 組裡最合這批資料
胃口的那組有多好」—— 那個數字必然偏樂觀，而且偏多少無從得知。

所以搜尋全程只用 Feb cohort，Mar 只在最後看**一次**。

## Feb 內部切成三塊，不是兩塊

    train (70%)  訓練
    es    (15%)  early stopping —— 決定停在第幾輪
    sel   (15%)  選超參數 —— 決定用哪一組參數

M1 只切兩塊，因為那時只有一個「看著分數做的決定」（停在第幾輪）。調參多了
第二個決定，如果兩個決定都看同一塊資料，那塊資料就同時被最佳化了兩次 ——
早停選在對它最有利的輪數、參數又選在對它最有利的組合，`sel` 分數會比真實
泛化能力好一截。

三塊都從 Feb 內部切、且**每個 trial 用同一批列**：不同 trial 若切到不同的
列，分數差異裡就混進了切分的運氣。

## 為什麼是隨機搜尋而不是網格

同樣的預算下，隨機搜尋在「只有少數幾個參數真的重要」的情況下明顯優於網格
（Bergstra & Bengio 2012）—— 網格會把預算浪費在不重要的維度上反覆取樣。
梯度提升正是這種情況：learning_rate 與 num_leaves 影響很大，
cat_smooth 之類的影響小得多。

貝氏最佳化（Optuna）會更有效率，但它多帶一個依賴、而且會讓「這組參數是
怎麼選出來的」變得難以複述。本專案的搜尋空間只有 9 維、預算只有幾十次，
隨機搜尋夠用，且一個 seed 就能完整重現。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from src.evaluation import log_loss
from src.features import FeatureSet
from src.models.candidates import fit_lightgbm

# 取整數的參數。從浮點數抽樣再取整，比為每個參數各寫一種抽樣邏輯簡單，
# 而且搜尋空間的設定檔只要宣告 type 就好。
INT_PARAMS = ("num_leaves", "min_data_in_leaf", "cat_smooth", "min_data_per_group")


@dataclass
class Trial:
    """一組超參數的搜尋結果。"""

    index: int
    params: dict[str, Any]
    best_iteration: int
    select_logloss: float  # 在 sel 那塊上的分數 —— 挑參數**只**看這個
    seconds: float
    sampled: dict[str, Any] = field(default_factory=dict)  # 只含這次抽樣到的值


@dataclass(frozen=True)
class ThreeWaySplit:
    """Feb cohort 的三段切分。每個 trial 共用同一組。"""

    train: FeatureSet
    es: FeatureSet
    sel: FeatureSet


def three_way_split(feb: FeatureSet, cfg: dict) -> ThreeWaySplit:
    """把 Feb cohort 切成 train / es / sel。

    先切出 sel，再從剩下的切出 es —— 兩次都分層抽樣。流失率只有 6.39%，
    不分層的話 15% 的小塊裡正例數量會有可觀波動，選參數就開始比運氣。

    ## 切分綁在 msno 上，不綁在列位置上

    `train_test_split` 是**依位置**切的：同一個 seed 餵進不同順序的資料，
    會切出不同的人。`build_cohort()` 末尾的 `.sort("msno")` 已經讓 cohort
    的順序固定，但那個保證住在另一個模組裡 —— 任何一次「先 filter 再切」、
    「join 完忘記排序」、或讀到修正之前的舊快取，都會讓這裡安靜地換一批
    訓練資料，而分數只動 0.0006 左右，看起來像實驗有了效果。

    因此這裡先自己按 msno 排出一個標準順序再切。cohort 已經排序時這是
    **恆等變換**（argsort 傳回 0..n-1），所有既有數字不受影響；順序一旦
    被上游改動，切分結果仍然不變。
    """
    canonical = feb.msno.arg_sort().to_numpy()
    y = feb.y.to_numpy()
    rest_idx, sel_idx = train_test_split(
        canonical,
        test_size=cfg["select_fraction"],
        random_state=cfg["split_seed"],
        stratify=y[canonical],
    )
    rest_y = feb.y.to_numpy()[rest_idx]
    tr_idx, es_idx = train_test_split(
        rest_idx,
        test_size=cfg["early_stopping_fraction"],
        random_state=cfg["split_seed"],
        stratify=rest_y,
    )
    return ThreeWaySplit(train=feb.take(tr_idx), es=feb.take(es_idx), sel=feb.take(sel_idx))


def sample_params(rng: np.random.Generator, space: dict[str, dict]) -> dict[str, Any]:
    """從搜尋空間抽一組參數。

    支援三種宣告：

        {type: uniform,    low: a, high: b}   線性均勻
        {type: loguniform, low: a, high: b}   對數均勻 —— 給跨數量級的參數
        {type: int,        low: a, high: b}   線性均勻後取整

    learning_rate 與正則化係數用 loguniform：它們的效果是乘性的，
    0.01→0.02 的差別遠大於 0.09→0.10，線性抽樣會把大半預算浪費在大值那端。
    """
    out: dict[str, Any] = {}
    for name, spec in space.items():
        kind = spec["type"]
        low, high = float(spec["low"]), float(spec["high"])
        if kind == "loguniform":
            value: Any = float(np.exp(rng.uniform(np.log(low), np.log(high))))
        elif kind == "uniform":
            value = float(rng.uniform(low, high))
        elif kind == "int":
            value = int(round(rng.uniform(low, high)))
        else:
            raise ValueError(f"未知的抽樣型別 {kind!r}（參數 {name}）")
        out[name] = int(value) if name in INT_PARAMS else value
    return out


def random_search(
    split: ThreeWaySplit,
    base_params: dict[str, Any],
    space: dict[str, dict],
    train_cfg: dict,
    *,
    n_trials: int,
    seed: int,
    verbose: bool = True,
) -> list[Trial]:
    """隨機搜尋。回傳所有 trial，依 sel 分數遞增排序。

    第 0 號 trial 固定是 `base_params` 本身（即 M1/M2 的設定），這樣「調參
    到底賺了多少」有一個同條件的對照，而不是跟記憶中的數字比。
    """
    rng = np.random.default_rng(seed)
    trials: list[Trial] = []

    for i in range(n_trials + 1):
        sampled = {} if i == 0 else sample_params(rng, space)
        params = {**base_params, **sampled}

        t0 = time.perf_counter()
        fitted = fit_lightgbm(split.train, split.es, params, train_cfg)
        score = log_loss(split.sel.y, fitted.predict(split.sel.X))
        secs = time.perf_counter() - t0

        trials.append(
            Trial(
                index=i,
                params=params,
                best_iteration=fitted.best_iteration,
                select_logloss=score,
                seconds=secs,
                sampled=sampled,
            )
        )
        if verbose:
            tag = "基準（M1/M2 設定）" if i == 0 else f"trial {i}/{n_trials}"
            print(
                f"  {tag}　sel {score:.5f}　{fitted.best_iteration} 輪　({secs:.0f} 秒)",
                flush=True,
            )

    return sorted(trials, key=lambda t: t.select_logloss)
