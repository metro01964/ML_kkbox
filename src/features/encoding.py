"""Out-of-fold target encoding —— 紅線 6 的實作與守門。

SPEC §5 紅線 6：「Target encoding 必須 out-of-fold。`payment_method_id` 是
高基數類別，直接 target encode 會讓 CV 飆高、實測崩盤。」

## 為什麼「直接 target encode」會崩盤

Target encoding 是把類別換成「這個類別的平均流失率」。問題在於：**計算某一
列的編碼時，如果把那一列自己的標籤也算進去，編碼就含有標籤本身。**

極端但真實的情況：某個 `payment_method_id` 在訓練集只出現 1 次，那位用戶
流失了。它的編碼就是 1.0 —— 而 1.0 正好等於它的標籤。模型只要學會
「編碼 = 1.0 就預測流失」，訓練分數與 CV 分數都會漂亮得不像話。到了測試集，
那個類別的真實流失率可能是 0.06，模型整個崩掉。

實測 cohort 內 `last_payment_method_id` 有 33 種取值，長尾的幾種只出現
個位數次 —— 正是這個陷阱的溫床。

## 解法：每一列的編碼只能用「別人的標籤」

把訓練資料切成 K 折，第 i 折的編碼用**其餘 K−1 折**的標籤算。這樣一列的
編碼裡永遠不含自己的標籤，上面那個 1.0 就不會出現（該類別在其他折若沒出現
過，就退回全體先驗）。

驗證集與測試集則用**整個訓練集**擬合的編碼 —— 它們的標籤本來就不參與計算，
不需要再切折。

## 這個模組與紅線 5 的關係

紅線 5 要求「所有 imputation / scaling / encoding 統計量必須在 fold 內計算」，
而 `src/features/build.py` 的解法是**完全無狀態**：不算任何統計量，所以無從
違反。Target encoding 天生需要統計量，因此它**刻意不放進 build.py** ——
放進去就會讓那個無狀態保證（與守著它的紅線 5 測試）失效。

編碼是在訓練流程裡、切分之後才做的一步。這個分工讓兩條紅線都成立：
特徵建構仍然無狀態，需要統計量的部分被關在 fold 邊界內。

## 平滑（smoothing）

樣本數少的類別，其平均值不可信。用貝氏平滑往全體先驗拉：

    encoding = (該類別的正例數 + prior × m) / (該類別的列數 + m)

`m` 是「相當於幾筆先驗觀測」。出現 1 次的類別會幾乎完全退回先驗，出現
十萬次的類別則幾乎不受影響。這與紅線 6 是兩件不同的事 —— 平滑處理的是
**小樣本雜訊**，OOF 處理的是**標籤洩漏**，兩者都要做。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold

# 平滑強度的預設值：相當於 100 筆先驗觀測。與 build.py 的
# `min_data_in_leaf: 100` 同一個量級 —— 少於 100 筆的類別本來就不該被
# 單獨相信。
DEFAULT_SMOOTHING = 100.0

# OOF 的折數。與 SPEC §4.2 的 5-fold 一致，沒有另立一套。
DEFAULT_N_SPLITS = 5

# **永遠不得作為正式 target encoding 對象的欄位。**
#
# `msno` 是每列一個唯一值的識別碼。對它做 target encoding，每個「類別」的
# 樣本數都是 1，編碼值等於（平滑後的）該列自己的標籤 —— OOF 也救不了它：
# 每一折都看不到自己那個 msno，所以每一列都退回先驗，整欄變成常數，
# 一點資訊都沒有卻讓人以為做了特徵工程。
#
# 兩種結局都很糟，而且都不會報錯：naive 編碼把標籤直接餵進模型（E 變體），
# OOF 編碼則產生一整欄常數。因此把它列成黑名單，由 `assert_not_identifier`
# 在合規路徑上擋掉。違規對照組要用它是刻意的，走另一條明確標示的路徑。
IDENTIFIER_COLUMNS: frozenset[str] = frozenset({"msno"})


@dataclass(frozen=True)
class TargetEncoder:
    """在一批資料上擬合好的類別 → 編碼值對照表。

    frozen：擬合完就不該再變。要換一批資料就重新擬合一個，
    避免「這個 encoder 到底 fit 在哪批資料上」變成無法回答的問題。
    """

    mapping: dict[object, float]
    prior: float
    smoothing: float

    def transform(self, values: pl.Series) -> pl.Series:
        """套用編碼。**沒見過的類別一律回先驗**，不是 null。

        回先驗而不是 null：一個沒見過的付款方式，最合理的猜測就是「跟大家
        一樣」。給 null 會讓模型多走一條缺失分支，而那條分支學到的是
        「訓練時罕見」—— 那是關於資料集的事實，不是關於用戶的事實。
        """
        return values.replace_strict(
            self.mapping, default=self.prior, return_dtype=pl.Float64
        ).alias(f"{values.name}_te")


def fit_target_encoder(
    values: pl.Series,
    y: pl.Series,
    *,
    smoothing: float = DEFAULT_SMOOTHING,
) -> TargetEncoder:
    """用一批（且只有這一批）資料的標籤擬合編碼表。

    ⚠️ 傳進來的 `y` **必須全部屬於訓練集**。把驗證集的標籤混進來就是紅線 6
    的違反，而且不會有任何錯誤訊息 —— 只會得到一個好看的分數。
    """
    if values.len() != y.len():
        raise ValueError(f"長度不符：values {values.len()} 筆，y {y.len()} 筆")
    if values.len() == 0:
        raise ValueError("空的輸入無法擬合 target encoder")

    prior = float(y.mean())
    agg = (
        pl.DataFrame({"v": values, "y": y.cast(pl.Float64)})
        .group_by("v")
        .agg(pl.col("y").sum().alias("pos"), pl.len().alias("n"))
        .with_columns(
            ((pl.col("pos") + prior * smoothing) / (pl.col("n") + smoothing)).alias("enc")
        )
    )
    return TargetEncoder(
        mapping=dict(zip(agg["v"].to_list(), agg["enc"].to_list(), strict=True)),
        prior=prior,
        smoothing=smoothing,
    )


def oof_target_encode(
    values: pl.Series,
    y: pl.Series,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = 42,
    smoothing: float = DEFAULT_SMOOTHING,
) -> pl.Series:
    """對**訓練資料**做 out-of-fold 編碼：每一列的編碼只用其他折的標籤。

    Returns:
        與輸入等長、順序相同的編碼欄。

    這個函式就是紅線 6 本身。守門的測試
    `tests/test_no_leakage.py::test_red_line_6_target_encoding_is_oof`
    餵給它一份「每個類別只出現一次」的資料：naive 編碼會逐格等於標籤，
    OOF 編碼則必須全部退回先驗。分不出這兩者的實作會被那個測試擋下來。

    分層切折（StratifiedKFold）：流失率只有 6.39%，不分層的話某一折可能
    正例極少，那一折算出來的編碼會整體偏低。
    """
    if values.len() != y.len():
        raise ValueError(f"長度不符：values {values.len()} 筆，y {y.len()} 筆")

    y_np = y.to_numpy()
    out = np.empty(values.len(), dtype=np.float64)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fit_idx, apply_idx in skf.split(np.zeros(values.len()), y_np):
        # fit_idx 是「其他折」，apply_idx 是「這一折」。sklearn 的 split()
        # 回傳 (train, test)，這裡的語意剛好對得上：用 train 的標籤擬合，
        # 套到 test 上 —— test 那些列的標籤一次都沒被用到。
        enc = fit_target_encoder(values[fit_idx], y[fit_idx], smoothing=smoothing)
        out[apply_idx] = enc.transform(values[apply_idx]).to_numpy()

    return pl.Series(f"{values.name}_te", out)


def assert_not_identifier(column: str) -> None:
    """守門：合規的 target encoding 不得作用在唯一識別碼上。

    見 `IDENTIFIER_COLUMNS` 的說明 —— 對 `msno` 做 target encoding，naive 會
    洩漏標籤、OOF 會產生一整欄常數，兩種都不會報錯。

    ⚠️ 違規對照組（`scripts/target_encoding.py` 的 D、E）刻意需要繞過這條，
    它們走的是明確標示為「違規」的另一條路徑，且其分數不得進入任何結論。

    Raises:
        ValueError: 該欄位是唯一識別碼。
    """
    if column in IDENTIFIER_COLUMNS:
        raise ValueError(
            f"{column!r} 是唯一識別碼，不得作為正式 target encoding 特徵："
            "naive 編碼會等於該列自己的標籤，OOF 編碼則整欄退回先驗。"
        )


def assert_encoding_aligned(
    encoded: pl.Series,
    msno_before: pl.Series,
    msno_after: pl.Series,
) -> None:
    """守門：編碼前後的列順序與 msno 必須逐格對齊。

    編碼是「取一欄、算一欄、貼回去」，而貼回去是**依位置**的。中間只要有
    一次 sort、filter、join 或 group_by 改動了順序，每個人就會拿到別人的
    編碼值 —— 列數不變、沒有 null、沒有錯誤訊息，只有分數會變差。

    這與 `_attach_logs` 的 horizontal concat 是同一類問題（§7.11 第二個缺口），
    所以用同一種方式擋：直接驗證假設本身。

    Raises:
        AssertionError: 列數不符或 msno 順序改變。
    """
    if msno_before.len() != msno_after.len():
        raise AssertionError(f"編碼前後列數改變：{msno_before.len():,} → {msno_after.len():,}")
    if encoded.len() != msno_after.len():
        raise AssertionError(
            f"編碼欄長度（{encoded.len():,}）與資料列數（{msno_after.len():,}）不符"
        )
    if not msno_before.equals(msno_after):
        raise AssertionError(
            "編碼前後 msno 順序改變 —— 編碼值是依位置貼回去的，順序一變每個人都會拿到別人的編碼。"
        )


def assert_encoding_is_oof(
    encoded: pl.Series,
    y: pl.Series,
    *,
    tolerance: float = 1e-9,
) -> None:
    """守門：編碼欄不得逐格等於標籤。

    與紅線 1、2 的守門同構 —— 抽成獨立函式，才能餵它一份確實違規的輸入
    來證明它真的會擋。

    測法的理由：naive target encoding 最戲劇性的失敗就是**編碼 = 標籤**
    （每個類別只出現一次時必然如此）。這個檢查抓的是那個極端；比較細微的
    洩漏（類別出現數次）由 `oof_target_encode` 的結構保證，不是靠這個斷言。

    Raises:
        AssertionError: 編碼與標籤完全相同。
    """
    if encoded.len() != y.len():
        raise KeyError(f"長度不符：encoded {encoded.len()} 筆，y {y.len()} 筆")

    diff = (encoded.cast(pl.Float64) - y.cast(pl.Float64)).abs().max()
    assert diff is not None and float(diff) > tolerance, (
        "紅線 6 違反：target encoding 的值逐格等於標籤，"
        "代表編碼時用到了該列自己的標籤（沒有 out-of-fold）。"
    )
