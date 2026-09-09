"""M4 第二步 · 機率校準器（isotonic）。

`calibration.py` 只量測、不 fit；這個模組是它的另一半 —— fit 一個把預測
機率映射到實際頻率的函式，並把**代價**一起量出來。

## 為什麼是 isotonic 而不是 Platt

不是因為 isotonic 通常比較好，是因為 Platt 的假設在這份資料上不成立。
M4 第一步的診斷發現 D7 → D8 出現排序反轉：平均預測從 0.64% 升到 1.68%，
實際流失率反而從 5.55% 降到 3.76%。Platt 是單調的 logistic 映射，只會把
整條曲線平移縮放，這段反轉會原封不動留著；isotonic 強制單調，至少能把
它壓平成同一個值。

## 校準器 fit 在哪一塊

Feb-sel —— 模型訓練與 early stopping 都沒看過的那 15%（切分來自
`configs/calibration.yaml`，seed 與 `tuning.yaml` 刻意不同，理由寫在該檔）。
fit 在 Mar 上會讓 reliability diagram 變漂亮，但那是拿唯一的時間外驗證集
調自己的成績單。

## 兩個會安靜出錯的地方

**一、isotonic 會輸出剛好 0。** PAVA 把相鄰的區段併成同值的區塊，最低的
那個區塊若一個正例都沒有，該區塊的值就是 0。套到新資料上，只要有一個正例
落進那個區間，log loss 就會踩到 `EPS = 1e-15` 的 clip，單筆貢獻 34.5 ——
一萬筆裡有一筆就足以讓整體 log loss 惡化 0.003。這比校準本身的收益還大，
而且它長得像「校準讓模型變差了」，不像 bug。

因此 `fit_isotonic` 會從**零區塊的實際大小**推出一個下界（見 `_edge_bounds`），
而不是寫死一個 epsilon：m 筆樣本裡觀測到 0 個正例，Jeffreys 後驗均值是
0.5/(m+1)，意思是「這 m 筆只能告訴你機率低於這個量級，不能告訴你它是 0」。
沒有零區塊時下界為 0，這個機制自動失效 —— 不會憑空改動任何預測。

**二、isotonic 會壓掉排序資訊。** 被壓平的區塊裡，所有人拿到完全相同的
機率。SPEC §6.2 的核心交付物是「投放給預測機率最高的前 K%」，如果門檻剛好
落在一個大區塊裡，「前 K%」就不再有定義。所以 `tie_profile()` 是這個模組的
一級 API，不是附屬診斷 —— 校準前後各量一次，代價要跟收益並排看。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

# Jeffreys 先驗 Beta(0.5, 0.5) 的參數。用它而不是 Laplace 的 +1，是因為
# Jeffreys 在極端比例（趨近 0 或 1）附近的覆蓋率明顯較好，而本專案的低機率端
# 正是這種情況：預測值有九成落在 2% 以下。
JEFFREYS = 0.5


def _as_array(values: Iterable) -> np.ndarray:
    """轉成 float64 的一維陣列，polars Series 走零複製的快路徑。

    `calibration.py` 用的是 `np.asarray(list(...))`，那對 97 萬列的 Mar cohort
    會先攤成一個 Python list。診斷腳本一次只跑幾回合，成本看不出來；校準要
    在同一份資料上反覆套用，就值得走 `to_numpy()`。
    """
    if isinstance(values, pl.Series):
        return values.to_numpy().astype(np.float64, copy=False)
    if isinstance(values, np.ndarray):
        return values.astype(np.float64, copy=False)
    return np.asarray(list(values), dtype=np.float64)


def _edge_bounds(fitted_values: np.ndarray) -> tuple[float, float]:
    """從零區塊與一區塊的大小推出下界與上界。

    isotonic 的輸出若剛好是 0，代表 PAVA 把最低的那一段併成一個「一個正例
    都沒有」的區塊。這個區塊有多大，決定了我們對它的機率能講到多細：

        m 筆樣本、0 個正例 → Jeffreys 後驗均值 0.5 / (m + 1)

    m 越大，下界越低（資料真的支持「這群人風險極低」）；m 很小則下界較高
    （只有幾百筆全陰，不足以宣稱風險是百萬分之一）。

    沒有零區塊時回傳 0.0 —— 下界失效，不會動到任何預測。上界同理。
    """
    n_zero = int((fitted_values <= 0.0).sum())
    n_one = int((fitted_values >= 1.0).sum())
    floor = JEFFREYS / (n_zero + 1) if n_zero else 0.0
    ceiling = 1.0 - JEFFREYS / (n_one + 1) if n_one else 1.0
    return floor, ceiling


@dataclass(frozen=True)
class IsotonicCalibrator:
    """fit 完成的 isotonic 映射，以及它在校準集上的形狀。

    Attributes:
        n_fit:    fit 用了幾筆。isotonic 的自由度隨樣本數成長，這個數字是
                  判讀「階梯是不是雜訊」的前提。
        floor:    輸出下界，由零區塊大小推出（見 `_edge_bounds`）。
        ceiling:  輸出上界，同理。
        n_zero_block: 校準集裡被映射到 0 的筆數。0 代表下界機制沒有啟動。
    """

    model: IsotonicRegression
    n_fit: int
    floor: float
    ceiling: float
    n_zero_block: int

    def apply(self, p: Iterable) -> np.ndarray:
        """把原始預測映射成校準後的機率。

        `out_of_bounds="clip"` 讓超出校準集範圍的預測取端點值。Mar cohort
        的最大預測值可能高於 Feb-sel 的最大值，外推沒有依據 —— 取端點是
        「不知道就別再往上猜」。
        """
        raw = _as_array(p)
        if raw.size == 0:
            raise ValueError("空的輸入無法套用校準器")
        return np.clip(self.model.predict(raw), self.floor, self.ceiling)

    @property
    def knots(self) -> pl.DataFrame:
        """映射的轉折點：原始預測 → 校準後機率。供繪圖與人工檢視。"""
        return pl.DataFrame(
            {
                "原始預測": self.model.X_thresholds_.astype(np.float64),
                "校準後": np.clip(
                    self.model.y_thresholds_.astype(np.float64), self.floor, self.ceiling
                ),
            }
        )

    @property
    def n_levels(self) -> int:
        """映射的相異輸出值個數 —— 校準後最多還能分出幾個風險等級。"""
        return int(np.unique(np.clip(self.model.y_thresholds_, self.floor, self.ceiling)).size)


def fit_isotonic(
    y_true: Iterable,
    y_pred: Iterable,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
) -> IsotonicCalibrator:
    """在保留集上 fit 一個 isotonic 校準器。

    Args:
        y_true:  保留集的真實標籤（0/1）。
        y_pred:  同一批人的**未校準**預測機率。
        floor:   手動指定輸出下界；預設由零區塊大小推出（見 `_edge_bounds`）。
        ceiling: 手動指定輸出上界；預設同理。

    Raises:
        ValueError: 長度不符、空輸入、標籤不是 0/1、或標籤只有單一類別。

    單一類別為什麼要擋：全陰的校準集會 fit 出一個把所有人映射到 0 的常數
    函式，套上去等於把整份預測抹平。它不會報錯，只會讓下游所有指標同時
    變得莫名其妙 —— 這種失效值得在源頭擋掉。
    """
    y = _as_array(y_true)
    p = _as_array(y_pred)

    if y.size != p.size:
        raise ValueError(f"長度不符：y_true {y.size} 筆，y_pred {p.size} 筆")
    if y.size == 0:
        raise ValueError("空的輸入無法 fit 校準器")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true 只能是 0 或 1")
    if len(np.unique(y)) < 2:
        raise ValueError("校準集只有單一類別，fit 出來的會是常數映射")

    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(p, y)

    on_fit = model.predict(p)
    auto_floor, auto_ceiling = _edge_bounds(on_fit)

    return IsotonicCalibrator(
        model=model,
        n_fit=int(y.size),
        floor=auto_floor if floor is None else float(floor),
        ceiling=auto_ceiling if ceiling is None else float(ceiling),
        n_zero_block=int((on_fit <= 0.0).sum()),
    )


def tie_profile(p: Iterable) -> dict[str, float]:
    """量化排序解析度：有多少人拿到一模一樣的機率。

    校準的代價就在這裡。isotonic 把失準的區段壓平成同一個值，區塊內的
    相對順序全部消失 —— 而 SPEC §6.2 的投放決策是「取預測機率最高的前 K%」，
    門檻若落在一個大區塊裡，那個 K% 要取誰就沒有依據了。

    Returns:
        相異值數 / 最大同分組佔比 / 有同分對象的樣本佔比。
        第三個數字最直觀：0.62 代表六成的人跟至少一個別人完全同分。
    """
    values = _as_array(p)
    if values.size == 0:
        raise ValueError("空的輸入無法計算同分結構")
    _, counts = np.unique(values, return_counts=True)
    return {
        "相異值數": int(counts.size),
        "最大同分組佔比": float(counts.max() / values.size),
        "有同分的樣本佔比": float(counts[counts > 1].sum() / values.size),
    }
