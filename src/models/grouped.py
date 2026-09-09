"""M6 · 合併多個 cohort 之後的切分 —— 紅線 4。

SPEC §4.4 實測：881,693 位用戶同時出現在 Feb 與 Mar cohort，佔 Mar 的 **90.81%**。

在 §4.2 的 Feb → Mar 切分下這**不是**傳統意義的洩漏（兩邊的標籤來自不同的時間
窗，真實部署本來就是每月對同一批訂戶重複評分）。但一旦把兩個 cohort 併成一份
資料再隨機切分，它就是了：同一個人的 Feb 列進訓練、Mar 列進驗證，而他的城市、
註冊管道、慣用付款方式、年資在相隔一個月時幾乎不變，94.78% 的人連標籤都沒變。
模型只要記住特徵簽章就能在驗證集上拿分 —— 那個分數量到的是記憶力，不是泛化。

紅線 4 因此規定：**合併多 cohort 做 KFold 必須 `GroupKFold(groups=msno)`。**

## ⚠️ 上面那段是規定的理由，不是實測結果

M6 用違規對照組量過（§7.20）：違規那一臂每折有 **75.2% 的驗證集用戶模型已經
見過**，而兩臂的 log loss 只差 **0.008 倍折間標準差 —— 量不到**。三個原因：
`msno` 不是特徵、同一個人的兩列差一個月（不是重複列，5.22% 標籤還翻轉）、
深度 6 的樹沒有記住個人的容量。

這條紅線因此是一張**保險**：換成高容量模型、或對高基數的用戶級欄位做 target
encoding，同樣的重疊就會被兌現。遵守的成本是一行 `groups=msno`，而不遵守時
**沒有任何跡象會告訴你這次兌現了沒有** —— 所以規定照舊，只是理由要講準確。

## 守門守的是性質，不是「你呼叫了哪個類別」

`assert_groups_disjoint()` 檢查的是「沒有任何 msno 同時出現在兩段」。它不檢查
是誰產生了這組索引 —— 一個靜態檢查（有沒有 import GroupKFold）擋不住

    StratifiedGroupKFold(...).split(X, y, groups=df["city"])   # groups 傳錯欄

而那是這條紅線最可能的犯法方式：程式看起來完全正確，分數只是變好一點。
性質檢查兩種都擋得住，而且它在**正常路徑上每次都跑**，不是只在測試裡跑
（§7.5 的教訓：只寫在正常路徑上的守門，會被快取命中與公開 API 繞過）。

## 為什麼是 StratifiedGroupKFold 而不是 GroupKFold

紅線的字面要求是 `groups=msno`，`StratifiedGroupKFold` 滿足它 —— 它一樣不讓
任何群組跨折，只是在此之上盡量讓各折的正例率一致。而正例率一致在這裡是必要
的：合併後的流失率約 7.7%，純 `GroupKFold` 只看群組大小，各折的正例數會晃，
折間分數的差裡就混進了切分的運氣。

代價是 `StratifiedGroupKFold` 的分層是**近似**的（群組不可分割，無法同時
精確滿足兩個條件）。所以報告要印各折的實際正例率，而不是假設它們相等。

## 四段切分：每一段對應一個決定

§7.6 留下的那句「M6 重訓最終模型時應改採四段切分」。

    train  模型權重
    es     停在第幾輪
    sel    要不要採用校準器
    cal    fit 校準器

一塊資料只能承擔一個「看著分數做的決定」。§7.6 記錄的殘餘風險正是兩個決定
共用一塊：M3 用它挑門檻與超參數、M4 又用它 fit 校準器，於是那塊資料作為
「乾淨保留集」的身分已經被消耗過兩次。

反過來說，**切出來卻用不到的段等於白白縮小訓練集**，所以上面四段每一段都
有指定的用途，沒有備而不用的。

執行入口在 `scripts/final_model.py`。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split

from src.features import FeatureSet

# 四段的名字。順序是「報表印出來的順序」，切的順序見 HOLDOUTS。
SEGMENTS = ("train", "es", "sel", "cal")

# 實際切出來的三段；train 是剩下的那些人。
#
# 順序會影響「誰落在哪一段」（先切走的那批不參與後面的抽樣），所以它是配方的
# 一部分，不能隨手調換。cal 先切是因為它是最需要「從沒被任何決定看過」的一段。
HOLDOUTS = ("cal", "sel", "es")

# 四段比例加總允許的誤差。0.7 + 0.1 + 0.1 + 0.1 在 float64 下不精確等於 1。
FRACTION_TOLERANCE = 1e-9


@dataclass(frozen=True)
class MergedCohorts:
    """兩個以上 cohort 併成的一份訓練資料，外加每一列的出身。

    Attributes:
        fs:       合併後的特徵矩陣與標籤。
        cohort:   每一列來自哪個 cohort。**出身證明，不是特徵** —— 與
                  `build_log_features()` 帶的 `cutoff` 欄同一個地位。它絕不
                  能進 X：Apr cohort（要預測的那一批）沒有這個值，一旦進了
                  特徵，模型就會學到一個部署時不存在的東西。
        parts:    合併了哪些 cohort，依合併順序。
        n_shared: 出現在一個以上 cohort 的用戶數 —— 紅線 4 的曝險面大小。
    """

    fs: FeatureSet
    cohort: pl.Series
    parts: tuple[str, ...]
    n_shared: int

    @property
    def n_rows(self) -> int:
        return self.fs.X.height

    @property
    def n_groups(self) -> int:
        return self.fs.msno.n_unique()

    def summary(self) -> pl.DataFrame:
        """每個 cohort 貢獻了幾列、流失率多少。"""
        return (
            pl.DataFrame({"cohort": self.cohort, "is_churn": self.fs.y})
            .group_by("cohort")
            .agg(pl.len().alias("列數"), pl.col("is_churn").mean().alias("流失率"))
            .sort("cohort")
        )


@dataclass(frozen=True)
class FourWaySplit:
    """合併資料的四段切分。每一段的 msno 與其他三段完全不重疊。"""

    train: FeatureSet
    es: FeatureSet
    sel: FeatureSet
    cal: FeatureSet

    def segment(self, name: str) -> FeatureSet:
        return getattr(self, name)

    def summary(self) -> pl.DataFrame:
        """各段的列數、人數與流失率 —— 分層有沒有生效看這張表。"""
        return pl.DataFrame(
            {
                "段": list(SEGMENTS),
                "列數": [self.segment(s).X.height for s in SEGMENTS],
                "人數": [self.segment(s).msno.n_unique() for s in SEGMENTS],
                "流失率": [float(self.segment(s).y.mean()) for s in SEGMENTS],
            }
        )


def merge_cohorts(parts: Mapping[str, FeatureSet]) -> MergedCohorts:
    """把多個 cohort 的特徵矩陣直向接起來。

    合併前擋掉四種會安靜出錯的情況：

      1. **欄位或順序不同** —— 直向 concat 是依位置對齊的，欄序不同會把
         `city` 接到 `registered_via` 上，而兩者都是小整數，不會有人抱怨。
      2. **dtype 不同** —— polars 會自動 upcast，於是類別欄悄悄變成別的型別，
         下游的類別索引跟著失準。
      3. **categorical 宣告不同** —— 兩份資料對「哪幾欄是類別特徵」的認定
         不一致時，合併後那份的認定必然對其中一份是錯的。
      4. **標籤是佔位值** —— Kaggle 測試集的 `is_churn` 是 null（見
         `assert_labels_are_real`）。把它併進訓練集會得到一份含 null 標籤的
         資料，而多數套件會把 null 當 0，也就是「全部沒流失」。

    Raises:
        ValueError: 上述任一種情況。
    """
    if len(parts) < 2:
        raise ValueError(f"合併至少要兩個 cohort，只給了 {len(parts)} 個")

    items = list(parts.items())
    first_name, first = items[0]
    for name, fs in items[1:]:
        if fs.X.columns != first.X.columns:
            raise ValueError(
                f"{name} 與 {first_name} 的特徵欄位或順序不同，不能直向合併："
                f"{set(fs.X.columns) ^ set(first.X.columns) or '欄位相同但順序不同'}"
            )
        if fs.X.schema != first.X.schema:
            differing = [c for c in first.X.columns if fs.X.schema[c] != first.X.schema[c]]
            raise ValueError(f"{name} 與 {first_name} 的 dtype 不同：{differing}")
        if fs.categorical != first.categorical:
            raise ValueError(
                f"{name} 與 {first_name} 對類別特徵的宣告不同："
                f"{fs.categorical} vs {first.categorical}"
            )

    for name, fs in items:
        if fs.y.null_count():
            raise ValueError(
                f"{name} 的標籤含 {fs.y.null_count():,} 個 null —— "
                "沒有標籤的 cohort（Kaggle 測試集）不能併進訓練資料。"
            )
        if fs.msno.n_unique() != fs.msno.len():
            raise ValueError(f"{name} 內部就有重複的 msno，合併前的契約已經不成立")

    fs = FeatureSet(
        X=pl.concat([f.X for _, f in items], how="vertical"),
        y=pl.concat([f.y for _, f in items]),
        msno=pl.concat([f.msno for _, f in items]),
        categorical=first.categorical,
    )
    cohort = pl.concat(
        [pl.repeat(name, f.X.height, eager=True, dtype=pl.String) for name, f in items]
    ).rename("cohort")

    n_shared = int(
        pl.DataFrame({"msno": fs.msno, "cohort": cohort})
        .group_by("msno")
        .agg(pl.col("cohort").n_unique().alias("k"))
        .filter(pl.col("k") > 1)
        .height
    )

    return MergedCohorts(fs=fs, cohort=cohort, parts=tuple(parts), n_shared=n_shared)


def assert_groups_disjoint(segments: Mapping[str, pl.Series]) -> None:
    """紅線 4 守門：沒有任何 msno 同時出現在兩個切分段。

    Args:
        segments: 段名 → 該段的 msno。至少兩段。

    Raises:
        AssertionError: 有人跨段。
        ValueError: 給的段數少於兩段（那沒有東西可以檢查）。

    這個函式收的是 msno 而不是索引或 splitter 物件，因為要檢查的性質只跟
    「哪些人在哪一段」有關。用了什麼工具切的、切幾折、有沒有分層，都不影響
    這個判斷 —— 也因此繞不過去。
    """
    if len(segments) < 2:
        raise ValueError(f"至少要兩段才檢查得出跨段，只給了 {len(segments)} 段")

    frame = pl.concat(
        [
            pl.DataFrame(
                {
                    "msno": s.cast(pl.String).rename("msno"),
                    "segment": pl.repeat(name, s.len(), eager=True, dtype=pl.String),
                }
            )
            for name, s in segments.items()
        ]
    )
    crossing = (
        frame.unique()
        .group_by("msno")
        .agg(pl.col("segment").sort().alias("segments"))
        .filter(pl.col("segments").list.len() > 1)
        .sort("msno")
    )
    if crossing.height:
        sample = crossing.row(0, named=True)
        raise AssertionError(
            f"紅線 4 違反：{crossing.height:,} 位用戶同時出現在多個切分段"
            f"（例：{sample['msno'][:12]}… 出現在 {sample['segments']}）。"
            "合併 cohort 之後的切分必須綁 msno，否則同一個人的 Feb 列在訓練、"
            "Mar 列在驗證，模型記住特徵簽章就能拿分。"
        )


def group_strata(msno: pl.Series, y: pl.Series) -> tuple[pl.Series, np.ndarray]:
    """每位用戶一列，並給他一個分層標籤。

    Returns:
        (依 msno 排序的用戶清單, 對應的分層標籤)

    分層標籤是 `"{這個人有幾列}-{其中幾列流失}"`，例如 `2-1` 代表「跨兩期出現、
    其中一期流失」。它同時綁住兩件事：

      - **流失率**。合併後約 7.7%，不分層的話 10% 的小塊裡正例數會有可觀波動。
      - **重複／新進的組成**。§4.5 實測兩群的流失率差 6.8 倍，一塊裡新進用戶
        多幾個百分點分數就跟著動 —— 那是切分的運氣，不是模型的差別。而「有
        幾列」正好就是「跨不跨期」的直接編碼。

    用 msno 排序而不是用原始列順序，是為了讓切分與列順序無關（同
    `three_way_split` 的理由：`train_test_split` 依位置切，餵進不同順序的資料
    會切出不同的人）。
    """
    per_group = (
        pl.DataFrame({"msno": msno, "y": y})
        .group_by("msno")
        .agg(pl.len().alias("n"), pl.col("y").sum().alias("k"))
        .sort("msno")
    )
    strata = (per_group["n"].cast(pl.String) + "-" + per_group["k"].cast(pl.String)).to_numpy()
    return per_group["msno"], strata


def _rows_by_segment(
    msno: pl.Series, picked: Mapping[str, pl.Series], names: Sequence[str]
) -> dict[str, np.ndarray]:
    """把「哪個人在哪一段」對回列索引。

    這一步是**用 msno 對回去**，不是用位置 —— 位置對齊在這裡必然是錯的：
    一個人有一到兩列，群組索引與列索引的長度根本不同。
    """
    assign = pl.concat(
        [
            pl.DataFrame(
                {
                    "msno": s.rename("msno"),
                    "segment": pl.repeat(name, s.len(), eager=True, dtype=pl.String),
                }
            )
            for name, s in picked.items()
        ]
    )
    rows = pl.DataFrame({"msno": msno, "_row": np.arange(msno.len(), dtype=np.int64)})
    joined = rows.join(assign, on="msno", how="left")
    if joined.height != msno.len():
        raise AssertionError(
            f"對回列時列數改變（{msno.len():,} → {joined.height:,}）—— 分段表有重複的 msno"
        )
    if joined["segment"].null_count():
        raise AssertionError(f"{joined['segment'].null_count():,} 列沒有被分到任何一段")

    # 排序：join 之後的列序不保證，而「同一組索引每次都長一樣」讓後續的
    # 快取、比對與重現都少一個變數。
    idx_of = {
        name: np.sort(joined.filter(pl.col("segment") == name)["_row"].to_numpy()) for name in names
    }
    empty = [name for name, idx in idx_of.items() if len(idx) == 0]
    if empty:
        raise ValueError(f"這幾段一列都沒有：{empty}（比例設得太小？）")
    return idx_of


def group_split(
    msno: pl.Series, y: pl.Series, *, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """把列切成兩塊，同一個 msno 必定整組落在同一邊。

    Returns:
        (留下的列索引, 切出去的列索引)

    這是 `split_for_early_stopping()` 的群組版。合併資料上的每一次切分都要
    用這個 —— 包括 fold 內部那一小塊 early stopping：它同樣是「模型看得到
    的資料」，同一個人一半在訓練一半在早停，停點就是看著自己選的。
    """
    groups, strata = group_strata(msno, y)
    rest, take = train_test_split(
        np.arange(groups.len()), test_size=test_size, random_state=seed, stratify=strata
    )
    idx_of = _rows_by_segment(
        msno,
        {"rest": groups[pl.Series(rest)], "take": groups[pl.Series(take)]},
        ("rest", "take"),
    )
    assert_groups_disjoint({name: msno[pl.Series(idx)] for name, idx in idx_of.items()})
    return idx_of["rest"], idx_of["take"]


def four_way_group_split(merged: MergedCohorts, cfg: Mapping[str, Any]) -> FourWaySplit:
    """把合併資料切成 train / es / sel / cal，四段都綁 msno。

    Args:
        merged: `merge_cohorts()` 的輸出。
        cfg:    `configs/final_model.yaml` 的 `split` 區段（`fractions` 與 `seed`）。

    設定檔寫的是**佔全體**的比例，這裡自己換算成每一次抽樣的條件比例。
    `configs/calibration.yaml` 的 `early_stopping_fraction: 0.176` 是手算的
    條件比例（0.15 / 0.85）—— 那種數字沒有人覆核得動，而算錯了只會讓某一段
    大小不對，不會有任何錯誤訊息。

    Raises:
        KeyError:       設定缺段或缺 seed。
        ValueError:     四段比例加起來不是 1。
        AssertionError: 切完之後有人跨段（守門，見 `assert_groups_disjoint`）。
    """
    fractions = _validated_fractions(cfg)
    seed = int(cfg["seed"])
    groups, strata = group_strata(merged.fs.msno, merged.fs.y)

    # 一次切一段，每次都從「還沒被切走的人」裡抽。設定檔宣告的是佔全體的
    # 比例，條件比例（除以 remaining）由這裡算。
    remaining = 1.0
    rest = np.arange(groups.len())
    picked: dict[str, np.ndarray] = {}
    for name in HOLDOUTS:
        rest, take = train_test_split(
            rest,
            test_size=fractions[name] / remaining,
            random_state=seed,
            stratify=strata[rest],
        )
        picked[name] = take
        remaining -= fractions[name]
    picked["train"] = rest

    idx_of = _rows_by_segment(
        merged.fs.msno,
        {name: groups[pl.Series(idx)] for name, idx in picked.items()},
        SEGMENTS,
    )

    # 守門在正常路徑上跑。由建構方式看它是恆真的 —— 但「看起來恆真」正是
    # §7.5 那七個洩漏的共同外觀，而 join 多對一時它就不再恆真了。
    assert_groups_disjoint({name: merged.fs.msno[pl.Series(idx)] for name, idx in idx_of.items()})
    return FourWaySplit(**{name: merged.fs.take(idx_of[name]) for name in SEGMENTS})


def grouped_folds(
    merged: MergedCohorts, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """合併資料上**合規**的 KFold：`StratifiedGroupKFold(groups=msno)`。

    Returns:
        每折的 (訓練列索引, 驗證列索引)。

    Raises:
        AssertionError: 任何一折有人跨邊（守門每折都跑）。
    """
    return _folds(merged, StratifiedGroupKFold, n_splits=n_splits, seed=seed, grouped=True)


def random_folds_violating_red_line_4(
    merged: MergedCohorts, *, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """合併資料上**違規**的 KFold：不看 msno 的 `StratifiedKFold`。

    ⚠️ **這個函式故意違反紅線 4。** 它的唯一用途是量出違規的代價 —— 同一份
    資料、同樣的折數、同一組超參數，只換切分規則，兩邊 CV 分數的差就是這條
    紅線在這個資料與這個模型上擋掉的東西。

    **實測是量不到**（§7.20：0.008 倍折間標準差，而違規那一臂每折有 75.2% 的
    驗證集用戶模型已經見過）。那個「量不到」也只有量了才知道 —— 這正是這個
    函式存在的理由。

    與紅線 6 的處理方式相同（§7.4 用三個實測違規的控制組量出 target encoding
    的代價）：只宣稱「這樣做才對」而不量代價，讀者無從判斷這條規定值不值得
    遵守，而一條沒有代價的規定遲早會有人為了方便繞過去。

    函式名字長且刺眼是刻意的 —— 它不該出現在任何正式流程裡，
    `tests/test_no_leakage.py` 會確認守門確實抓得到它產生的切分。
    """
    return _folds(merged, StratifiedKFold, n_splits=n_splits, seed=seed, grouped=False)


def _folds(
    merged: MergedCohorts,
    splitter_cls: type,
    *,
    n_splits: int,
    seed: int,
    grouped: bool,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """兩種折法共用的軀幹 —— 差別只有「有沒有把 msno 當 groups 傳進去」。

    合規與違規只差一個參數，這件事本身值得寫成程式碼：紅線 4 被違反時，
    程式碼的外觀與合規版幾乎相同，不會有任何一行看起來可疑。
    """
    y = merged.fs.y.to_numpy()
    X = np.zeros((len(y), 1))  # splitter 只看形狀，不看內容
    kwargs = {}
    if grouped:
        # 字串當 groups 也可以，但 190 萬個字串的 unique 很慢；轉成類別碼
        # 是等價的（同一個 msno 得到同一個碼）。
        kwargs["groups"] = merged.fs.msno.cast(pl.Categorical).to_physical().to_numpy()

    splitter = splitter_cls(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for tr, va in splitter.split(X, y, **kwargs):
        if grouped:
            assert_groups_disjoint(
                {
                    "train": merged.fs.msno[pl.Series(tr)],
                    "valid": merged.fs.msno[pl.Series(va)],
                }
            )
        folds.append((tr, va))
    return folds


def fold_report(
    merged: MergedCohorts, folds: Sequence[tuple[np.ndarray, np.ndarray]]
) -> pl.DataFrame:
    """每折的驗證集大小、正例率，以及**跨邊人數**。

    跨邊人數是這張表的重點：合規的折法它必然是 0，違規的折法它就是那一折
    「模型在訓練時已經見過」的人數。報告要並排印兩張表，差別一眼可見。
    """
    rows = []
    for i, (tr, va) in enumerate(folds, 1):
        tr_msno = merged.fs.msno[pl.Series(tr)]
        va_msno = merged.fs.msno[pl.Series(va)]
        # implode()：polars 1.43 起，`is_in` 收同型別的 Series 是有歧義的
        # （逐列比對 vs 當成一個集合），要明確表態。這裡要的是後者。
        shared = int(va_msno.unique().is_in(tr_msno.unique().implode()).sum())
        rows.append(
            {
                "fold": i,
                "訓練列數": len(tr),
                "驗證列數": len(va),
                "驗證正例率": float(merged.fs.y[pl.Series(va)].mean()),
                "跨邊人數": shared,
                "跨邊佔驗證": shared / va_msno.n_unique(),
            }
        )
    return pl.DataFrame(rows)


def _validated_fractions(cfg: Mapping[str, Any]) -> dict[str, float]:
    if "fractions" not in cfg or "seed" not in cfg:
        raise KeyError("split 區段需要 fractions 與 seed")
    fractions = {k: float(v) for k, v in dict(cfg["fractions"]).items()}
    missing = [s for s in SEGMENTS if s not in fractions]
    if missing:
        raise KeyError(f"fractions 缺少這幾段：{missing}")
    extra = [k for k in fractions if k not in SEGMENTS]
    if extra:
        raise KeyError(f"fractions 有不認識的段：{extra}")
    total = sum(fractions.values())
    if abs(total - 1.0) > FRACTION_TOLERANCE:
        raise ValueError(f"四段比例加起來是 {total}，不是 1")
    return fractions
