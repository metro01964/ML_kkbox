"""M5 · 逐位用戶的 SHAP 歸因（SPEC §7 M5）。

M5 要回答的是「**這一位**用戶為什麼被排在名單上」。M3 的 `Fitted.importance`
回答不了：gain 是整個模型的平均行為，同一個模型對每個人的理由不同。

## 三個單位問題，錯一個整張名單就讀錯

**一、SHAP 值在 log-odds 空間，不是機率。** 三家套件的貢獻值都定義在模型的
raw score 上，恆等式是

    sigmoid(base + Σ shap) == predict()

所以「`last_is_cancel` 貢獻 +2.3」的意思是**它把這個人的 log-odds 推高 2.3**，
不是「機率多 230%」。log-odds 到機率的換算不是線性的，同樣的 +2.3 在起點
0.05 與起點 0.5 的人身上會造成完全不同的機率變化 —— 這也是為什麼原因碼只能
排序、不該印「這一項讓他的流失率增加幾個百分點」。

**二、base value 不是 0，是「模型對所有人的共同起點」。** 一位用戶的分數
= 起點 + 他自己那 61 項偏離。原因碼講的是偏離，不是絕對水準。

**三、加總恆等式是可以驗的，所以一定要驗。** `assert_local_accuracy()` 把
`sigmoid(base + Σ shap)` 拿去跟 `Fitted.predict()` 對，逐列比。這條檢查抓的是
一整類不會報錯的錯：歸因用的模型與預測用的模型不同（輪數不同、特徵順序不同、
類別轉接寫了第二份），症狀都是「名單正確、原因碼講別人的事」。

理由與 §6.1 用曲線極大值驗 `campaign_curve()` 相同 —— 一個有恆等式的實作，
就不該只靠讀程式碼判斷它對不對。

## 分組：為什麼 Top-3 特徵不等於 Top-3 原因

61 個特徵裡有 4 個窗口 × 8 個收聽量（`log7_secs` / `log30_secs` / `log90_secs`
…）。它們高度相關，SHAP 會把同一件事的功勞拆給好幾欄 —— 於是純取 Top-3
特徵，很可能拿到「近 7 天聽歌時間低」「近 30 天聽歌時間低」「近 90 天聽歌
時間低」三句同義的話，而真正另一個原因（例如沒開自動續訂）被擠掉。

SHAP 的加法性讓分組是**精確的**：組內相加就是該組的總貢獻，不是近似。所以
`top_contributors(groups=...)` 先把貢獻依語意分組相加、取前 k 組，再從組內挑
貢獻最大的那一欄當代表去造句。分組本身是人的判斷（住在
`src/explain/reasons.py`），但「相加」這一步沒有近似。

⚠️ 代表欄與組總和是兩個數字，不可混用：`group_shap` 是「這個家族一起貢獻
多少」，`feature_shap` 是「代表那一欄自己貢獻多少」。組內可以有反向的成員，
所以前者不等於後者。

## 這一層不決定「印不印」

`top_contributors()` 挑出候選並排名，就到此為止。哪幾句真的呈現給營運，是
`src/explain/reasons.py::mark_display()` 的事 —— 分兩層是因為它們的失效方式
不同：這裡錯了是歸因錯，那裡錯了是呈現太寬鬆或太嚴格，而後者需要能事後稽核
「當時為什麼沒印」。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import polars as pl

# 加總恆等式的容差（機率空間的絕對誤差）。
#
# 為什麼在機率空間比而不是 log-odds：`Fitted.predict()` 回傳的就是機率，
# 要在 log-odds 比就得先 logit 回去，而 logit 在 p 接近 0 或 1 時會放大浮點
# 誤差 —— 實測本專案的機率分布最低到 1e-6 量級，logit 之後容差要訂到 1e-2
# 才不誤報，那就失去檢查的意義了。
#
# 1e-6 是「三家套件的 float32/float64 混用」允許的量級：CatBoost 的
# ShapValues 內部以 double 計算，LightGBM 的 pred_contrib 亦然，實測最大差
# 落在 1e-9 附近，離容差有三個數量級的餘裕。
LOCAL_ACCURACY_ATOL = 1e-6


def sigmoid(z: np.ndarray) -> np.ndarray:
    """log-odds → 機率。溢位安全（`exp` 只吃負數）。

    直接寫 `1 / (1 + exp(-z))` 在 z 很負時會 overflow 成 inf 並印出 RuntimeWarning
    —— 數值上仍然收斂到 0，但一個會在 log 裡噴警告的函式會讓真正的問題被忽略。
    """
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


@dataclass(frozen=True)
class Attribution:
    """一批用戶的 SHAP 歸因。

    values: `(列數, 特徵數)`，單位是 log-odds
    base:   `(列數,)`，模型對所有人的共同起點（同一個模型內為定值，但仍逐列
            保留 —— 三家套件都是逐列回傳，硬壓成純量等於假設它一定是定值）
    """

    features: tuple[str, ...]
    values: np.ndarray
    base: np.ndarray

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values 必須是二維，收到 {self.values.shape}")
        if self.values.shape[1] != len(self.features):
            raise ValueError(
                f"欄數不符：values 有 {self.values.shape[1]} 欄，"
                f"features 有 {len(self.features)} 個名字"
            )
        if self.base.shape != (self.values.shape[0],):
            raise ValueError(f"base 形狀應為 ({self.values.shape[0]},)，收到 {self.base.shape}")

    @property
    def n_rows(self) -> int:
        return int(self.values.shape[0])

    @property
    def raw(self) -> np.ndarray:
        """模型的 raw score（log-odds）= 起點 + 全部貢獻。"""
        return self.base + self.values.sum(axis=1)

    @property
    def probability(self) -> np.ndarray:
        """由歸因重建的機率。與 `Fitted.predict()` 應該逐列相等。"""
        return sigmoid(self.raw)

    def frame(self) -> pl.DataFrame:
        """歸因矩陣轉成有欄名的表，供人工檢視單一用戶。"""
        return pl.DataFrame({name: self.values[:, i] for i, name in enumerate(self.features)})


def attribute(fitted, X: pl.DataFrame) -> Attribution:
    """算這批列的 SHAP 歸因。

    Args:
        fitted: `src.models.candidates.Fitted`。用它的 `shap_values`，不碰
            模型本體 —— 類別特徵的轉接留在 `candidates.py`（見該模組開頭）。
        X: 特徵矩陣。**欄位順序必須與訓練時相同**，否則歸因會指到錯的欄名。

    Raises:
        ValueError: 輸出形狀不是 `(列數, 特徵數 + 1)`。
    """
    contrib = np.asarray(fitted.shap_values(X), dtype=np.float64)
    expected = (X.height, X.width + 1)
    if contrib.shape != expected:
        raise ValueError(
            f"{fitted.name} 的 SHAP 輸出形狀是 {contrib.shape}，預期 {expected}"
            "（最後一欄是 base value）"
        )
    return Attribution(tuple(X.columns), contrib[:, :-1], contrib[:, -1])


def local_accuracy_gap(fitted, X: pl.DataFrame, attr: Attribution) -> tuple[float, int]:
    """`sigmoid(base + Σ shap)` 與 `predict()` 的最大逐列差，以及發生在第幾列。"""
    gaps = np.abs(attr.probability - np.asarray(fitted.predict(X), dtype=np.float64))
    worst = int(np.argmax(gaps))
    return float(gaps[worst]), worst


def assert_local_accuracy(
    fitted, X: pl.DataFrame, attr: Attribution, *, atol: float = LOCAL_ACCURACY_ATOL
) -> float:
    """守門：歸因必須加得回這個模型的預測。

    這是 SHAP 的 local accuracy 性質，對精確 TreeSHAP 是恆等式而不是近似 ——
    所以不通過代表**歸因與預測不是同一個模型算的**，不是「誤差有點大」。

    Returns:
        實際的最大差（通過時仍回傳，供報表印出餘裕有多少）。

    Raises:
        AssertionError: 有任何一列超過容差。
    """
    gap, row = local_accuracy_gap(fitted, X, attr)
    if gap > atol:
        raise AssertionError(
            f"SHAP 加總對不上 {fitted.name} 的預測：第 {row} 列差 {gap:.3e}"
            f"（容差 {atol:.0e}）。\n"
            f"  歸因重建 {attr.probability[row]:.8f} vs predict {fitted.predict(X)[row]:.8f}\n"
            "  這不是精度問題 —— 精確 TreeSHAP 的加總是恆等式。可能的原因："
            "歸因與預測用了不同輪數的模型、欄位順序不一致、或類別特徵的轉接有兩份。"
        )
    return gap


def _group_index(
    features: tuple[str, ...], groups: Mapping[str, str] | Callable[[str], str] | None
) -> tuple[list[str], np.ndarray]:
    """把每個特徵對到一個組。`None` 代表不分組（每欄自成一組）。

    Raises:
        KeyError: 有特徵沒有對應的組。**刻意不預設吞掉** —— 新增一個特徵卻
            忘了給它分組，靜默的後果是它永遠不會出現在任何人的原因碼裡。
    """
    if groups is None:
        names = list(features)
        return names, np.arange(len(features))

    lookup = groups.get if isinstance(groups, Mapping) else groups
    assigned = []
    missing = []
    for f in features:
        g = lookup(f)
        if g is None:
            missing.append(f)
        else:
            assigned.append(g)
    if missing:
        raise KeyError(f"這些特徵沒有分組：{missing}")

    # 組的順序用「第一次出現」而非字母序 —— 字母序會讓組的編號隨改名而變動，
    # 而編號會進到輸出的排序裡。
    names: list[str] = []
    for g in assigned:
        if g not in names:
            names.append(g)
    idx = np.array([names.index(g) for g in assigned], dtype=np.int64)
    return names, idx


def top_contributors(
    attr: Attribution,
    X: pl.DataFrame,
    *,
    k: int = 3,
    groups: Mapping[str, str] | Callable[[str], str] | None = None,
    min_shap: float = 0.0,
) -> pl.DataFrame:
    """每一列取貢獻最大的前 k 個（組），長格式輸出。

    只取**推高風險**的方向（`shap > min_shap`）。挽回名單要的是「為什麼該打
    給他」，「他有開自動續訂所以風險較低」不是打電話的理由。

    ⚠️ **不足 k 個就給不足 k 個。** 一位用戶可能只有 1 個正貢獻的組，補到 3 個
    等於編造理由 —— 寧可讓那一格是空的。

    Args:
        k: 每列取幾個。
        groups: 特徵 → 語意組。給了就先組內相加再取前 k 組（見模組開頭）。
        min_shap: 門檻，預設 0 即「只要正貢獻」。

    Returns:
        每列每名次一列：

            row / rank / group / group_shap / feature / feature_shap / value

        `group_shap` 是**該組的總貢獻**，`feature_shap` 是代表那一欄自己的貢獻，
        兩者不相等（組內可以有反向成員）。`value` 是代表那一欄在該列的原始
        特徵值，缺失以 null 呈現 —— 「模型看到的是缺失」本身就是一種理由。
        `row` 是 X 的列位置（0-based），呼叫端自己接 msno。
    """
    if k < 1:
        raise ValueError(f"k 必須為正：{k}")
    if attr.n_rows != X.height:
        raise ValueError(f"列數不符：歸因 {attr.n_rows} 列，X {X.height} 列")
    if tuple(X.columns) != attr.features:
        raise ValueError("X 的欄位與歸因的特徵名不一致，歸因會指到錯的欄")

    # 非數值欄會在下面的 cast 裡安靜變成 null，於是每個原因碼都印「缺失」——
    # 看起來像資料品質問題，實際上是型別問題。擋在這裡。
    non_numeric = [c for c, dt in zip(X.columns, X.dtypes, strict=True) if not dt.is_numeric()]
    if non_numeric:
        raise ValueError(f"這些欄位不是數值型，無法作為原因碼的特徵值：{non_numeric}")

    group_names, gidx = _group_index(attr.features, groups)
    n_groups = len(group_names)

    # 組內相加。SHAP 的加法性讓這一步是精確的，不是近似。
    totals = np.zeros((attr.n_rows, n_groups), dtype=np.float64)
    for col, g in enumerate(gidx):
        totals[:, g] += attr.values[:, col]

    # 組內的代表欄：貢獻最大的那一個（不是絕對值最大 —— 要講的是推高風險的
    # 那一項）。組只有一欄時退化成它自己。
    members = [np.flatnonzero(gidx == g) for g in range(n_groups)]

    take = min(k, n_groups)
    # stable：同分時依組的編號（也就是特徵的出現順序）決定先後，不隨機。
    order = np.argsort(-totals, axis=1, kind="stable")[:, :take]

    values_f64 = X.with_columns(pl.all().cast(pl.Float64)).to_numpy()

    rows: list[dict] = []
    for r in range(attr.n_rows):
        rank = 0
        for g in order[r]:
            total = totals[r, g]
            if total <= min_shap:
                break  # 已依降序排列，後面只會更小
            cols = members[g]
            rep = cols[int(np.argmax(attr.values[r, cols]))]
            rank += 1
            v = values_f64[r, rep]
            rows.append(
                {
                    "row": r,
                    "rank": rank,
                    "group": group_names[g],
                    "group_shap": float(total),
                    "feature": attr.features[rep],
                    "feature_shap": float(attr.values[r, rep]),
                    "value": None if np.isnan(v) else float(v),
                }
            )

    schema = {
        "row": pl.Int64,
        "rank": pl.Int64,
        "group": pl.String,
        "group_shap": pl.Float64,
        "feature": pl.String,
        "feature_shap": pl.Float64,
        "value": pl.Float64,
    }
    return pl.DataFrame(rows, schema=schema)


def mean_abs_attribution(attr: Attribution) -> pl.DataFrame:
    """全體平均 |SHAP| —— 用來跟 `Fitted.importance` 的 gain 排名對照。

    兩者不該完全一致（gain 是分裂增益、這個是平均歸因量），但**排名前段落差
    太大就代表有問題**：例如歸因矩陣的欄位順序錯了。
    """
    return pl.DataFrame(
        {
            "feature": list(attr.features),
            "mean_abs_shap": np.abs(attr.values).mean(axis=0),
        }
    ).sort("mean_abs_shap", descending=True)
