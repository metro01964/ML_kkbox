"""M6 · PSI 漂移監控（SPEC §7 M6）。

服務上線之後唯一拿得到的東西是**輸入與輸出的分布** —— 標籤要等 30 天才知道
（`is_churn` 的定義是「到期後 30 天內沒有續訂」）。PSI（Population Stability
Index）量的就是「這個月進來的資料，跟模型訓練時看到的像不像」：

    PSI = Σ (p_i − q_i) × ln(p_i / q_i)

`p` 是**參考期**（訓練 cohort）每一箱的比例，`q` 是**當期**。它對稱、非負，
兩個分布完全相同時是 0。

## ⚠️ 0.1 / 0.25 這兩個門檻是慣例，不是統計檢定

業界慣用「< 0.1 穩定、0.1~0.25 中度、> 0.25 顯著」。這三句話沒有分布假設、
沒有樣本數修正 —— **同一個真實漂移，樣本數越大 PSI 不會變小，但樣本數越小
PSI 會因為抽樣雜訊而變大**。所以本模組另外提供 `noise_floor()`：把**同一個
cohort** 隨機切兩半算 PSI，得到「這個樣本數下、完全沒有漂移時」PSI 長什麼樣。
那個數字才是判讀的基準線。

理由與 §7.12 重量一次雜訊尺度相同：一個沒有比較基準的門檻，過與不過都只是
在讀慣例。

## 分箱：三種欄位分三種處理，而且**箱界只能從參考期算**

    連續值（n_unique > bins）   參考期的等量分位數當箱界
    離散值（n_unique <= bins）  直接用出現過的值當箱（0/1 旗標走這條）
    類別欄                      每個類別一箱

**箱界是一個擬合出來的狀態**，與紅線 5 說的 imputation / encoding 統計量同一
類東西。從當期或兩期合併算箱界，PSI 會被「當期自己的分布」影響 —— 那時它量
到的不再是漂移，而是「兩邊各自的分位數剛好差多少」，而分布完全相同時它也不
再保證是 0。

⚠️ **旗標欄一定要走離散那條。** `is_free_plan` 只有 0 與 1，等量分位數會產生
一堆重複的箱界，於是九成樣本落進同一箱、PSI 永遠接近 0 —— 一個「免費方案佔比
從 3% 變成 30%」的漂移會完全看不到。

## 缺失自成一箱，不丟掉

本專案的缺失是訊號（11.66% 的用戶不在 members_v3、18.0% 沒有收聽紀錄），而
**服務最可能遇到的漂移就是缺失率變了**：上游 join 壞掉、某個欄位停止供應。
把 null 丟掉再算 PSI，那種漂移會得到 0。

同理，類別欄在當期出現**參考期沒見過的值**時，那些列進一個獨立的 `unseen` 箱
並單獨回報。實測 Mar 有一個 Feb 沒有的 `last_payment_method_id`（§7.4）——
訓練時沒見過的類別在推論時會落到缺失分支，那件事本身就該被監控看到。

## PSI 是無上界的，所以「空箱」要講清楚

參考期某一箱是 0 而當期不是，`ln(p/q)` 是負無限。慣例是給一個下限
（`EPSILON`），於是 PSI 變成一個**由 epsilon 決定的大數字**。那個數字不可以
被當成「漂移的量」讀 —— 它只代表「出現了參考期沒有的東西」。因此本模組在
輸出裡帶 `epsilon_floored` 旗標，並把 epsilon 的值記進報告。

## 這個監控看不到什麼（M6 最重要的一句）

PSI 算的是**特徵與分數的分布**，它對「同樣的人、行為卻改變了」沒有意見。而
本專案已知的漂移正是那一種：Feb 的流失率 6.39% → Mar 8.99%（相對 +40%），
它讓 §7.10 的校準器失效。特徵分布若沒什麼變化而基準率變了那麼多，代表
**PSI 會在唯一真正傷到我們的那件事上回報「穩定」**。

所以 `scripts/drift_report.py` 把三個數字並排：特徵 PSI、分數 PSI、以及實際
的標籤漂移。第三個在部署時拿不到 —— 那正是要量它的理由。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# 預設箱數。10 是慣例（同 `src.evaluation.calibration` 的 reliability 分箱）。
DEFAULT_BINS = 10

# 空箱的比例下限。**這個值會直接決定「出現新類別」時 PSI 有多大**，所以它
# 進報告、也進每一列的 `epsilon_floored` 旗標。1e-6 對應「百萬列裡的一列」，
# 與本專案 cohort 的量級（97 萬）同階。
EPSILON = 1e-6

# 慣例門檻。寫成常數是為了讓「我們用的是慣例」這件事有一個明確的出處，
# 而不是散在各處的 0.25。判讀時要與 `noise_floor()` 一起看。
CONVENTIONAL_BANDS = ((0.10, "穩定"), (0.25, "中度"), (float("inf"), "顯著"))

MISSING_LABEL = "__missing__"
UNSEEN_LABEL = "__unseen__"


def band(value: float) -> str:
    """慣例分級。⚠️ 這是慣例不是檢定，見模組開頭。"""
    for upper, name in CONVENTIONAL_BANDS:
        if value < upper:
            return name
    return CONVENTIONAL_BANDS[-1][1]


@dataclass(frozen=True)
class Bins:
    """一個欄位的分箱方案，**擬合在參考期上**。

    Attributes:
        feature:   欄名。
        kind:      "quantile"（連續）/ "discrete"（少數取值）/ "category"。
        labels:    每一箱的名稱，含 `__missing__`（與類別欄的 `__unseen__`）。
        edges:     quantile 的內部箱界（不含 ±inf）。
        values:    discrete / category 的取值清單。
        reference: 參考期每一箱的比例，長度與 labels 相同、總和為 1。
        n_reference: 參考期的列數 —— 判讀 PSI 要知道它（見 noise_floor）。
    """

    feature: str
    kind: str
    labels: tuple[str, ...]
    reference: np.ndarray
    n_reference: int
    edges: tuple[float, ...] | None = None
    values: tuple[float, ...] | None = None

    @property
    def n_bins(self) -> int:
        return len(self.labels)


def _proportions(counts: np.ndarray) -> np.ndarray:
    total = counts.sum()
    if total == 0:
        raise ValueError("這一批一列都沒有，算不出分布")
    return counts / total


def fit_bins(
    series: pl.Series,
    *,
    bins: int = DEFAULT_BINS,
    categorical: bool = False,
) -> Bins:
    """在**參考期**上決定分箱方案（見模組開頭）。

    Args:
        series: 參考期的某一欄。
        bins: 連續值要幾箱。
        categorical: 是類別欄嗎（類別欄每個取值一箱，並保留 `unseen` 箱）。

    Raises:
        ValueError: 這一欄全是 null（沒有非缺失的值可以定箱界）。
    """
    name = series.name
    values = series.cast(pl.Float64)
    present = values.drop_nulls()
    if present.len() == 0:
        raise ValueError(f"{name} 在參考期全是 null，無法決定分箱")

    if categorical:
        cats = sorted(present.unique().to_list())
        labels = tuple([f"{c:g}" for c in cats] + [UNSEEN_LABEL, MISSING_LABEL])
        counts = np.array(
            [float((present == c).sum()) for c in cats] + [0.0, float(values.null_count())]
        )
        return Bins(
            name, "category", labels, _proportions(counts), values.len(), values=tuple(cats)
        )

    uniq = sorted(present.unique().to_list())
    if len(uniq) <= bins:
        # 旗標與少數取值的欄位。等量分位數在這裡會產生重複箱界，
        # 於是漂移被壓成 0（見模組開頭）。
        labels = tuple([f"{v:g}" for v in uniq] + [MISSING_LABEL])
        counts = np.array(
            [float((present == v).sum()) for v in uniq] + [float(values.null_count())]
        )
        return Bins(
            name, "discrete", labels, _proportions(counts), values.len(), values=tuple(uniq)
        )

    # 等量分位數。重複的箱界要去掉 —— 一個高度集中的欄位（例如 60% 是 0）
    # 會讓好幾個分位數相同，留著會產生寬度 0 的空箱。
    qs = [i / bins for i in range(1, bins)]
    edges = sorted({float(present.quantile(q, interpolation="linear")) for q in qs})
    counts, _ = np.histogram(present.to_numpy(), bins=[-np.inf, *edges, np.inf])
    labels = tuple(
        [f"(-inf, {edges[0]:g}]"]
        + [f"({edges[i - 1]:g}, {edges[i]:g}]" for i in range(1, len(edges))]
        + [f"({edges[-1]:g}, inf)", MISSING_LABEL]
    )
    counts = np.append(counts.astype(np.float64), float(values.null_count()))
    return Bins(name, "quantile", labels, _proportions(counts), values.len(), edges=tuple(edges))


def apply_bins(scheme: Bins, series: pl.Series) -> np.ndarray:
    """把當期的資料裝進參考期定好的箱，回傳比例。

    ⚠️ **不重新定箱界。** 那是本模組唯一一條不能違反的規則（見模組開頭）。
    """
    values = series.cast(pl.Float64)
    present = values.drop_nulls()
    missing = float(values.null_count())

    if scheme.kind == "quantile":
        counts, _ = np.histogram(present.to_numpy(), bins=[-np.inf, *(scheme.edges or ()), np.inf])
        counts = np.append(counts.astype(np.float64), missing)
        return _proportions(counts)

    known = list(scheme.values or ())
    counts = [float((present == v).sum()) for v in known]
    unseen = present.len() - sum(counts)
    if scheme.kind == "category":
        return _proportions(np.array(counts + [float(unseen), missing]))

    # discrete：參考期沒見過的取值也要有地方去，否則它們會靜靜消失。
    # 併進最後一箱會偽裝成「那個值變多了」，所以另開一箱 —— 但那會讓長度與
    # 參考期不同，因此這裡把它當成 unseen 處理並在 psi() 補一箱。
    return _proportions(np.array(counts + [float(unseen), missing]))


def psi_from_proportions(
    reference: np.ndarray, current: np.ndarray, *, epsilon: float = EPSILON
) -> tuple[float, bool]:
    """由兩組比例算 PSI。

    Returns:
        (PSI, 是否有任何一箱被 epsilon 墊高)。第二個值必須跟著第一個一起讀 ——
        它是 True 時，PSI 的大小由 epsilon 決定而不是由漂移的量決定。
    """
    if reference.shape != current.shape:
        raise ValueError(f"箱數不同：參考 {reference.shape}，當期 {current.shape}")
    floored = bool(((reference <= 0) | (current <= 0)).any())
    p = np.clip(reference, epsilon, None)
    q = np.clip(current, epsilon, None)
    return float(np.sum((p - q) * np.log(p / q))), floored


def psi(
    scheme: Bins, current: pl.Series, *, epsilon: float = EPSILON
) -> tuple[float, bool, pl.DataFrame]:
    """一個欄位的 PSI，以及逐箱的明細。

    明細是必要的，不是附加：PSI = 0.31 這個數字不會告訴你「是哪一箱動了」，
    而營運要問的下一個問題一定是那個。

    ## ⚠️ 兩邊都是空的箱要先丟掉，否則 `epsilon_floored` 會永遠是 True

    `__missing__` 與 `__unseen__` 這兩箱是**結構性**的（每個欄位都有，不管有沒有
    用到）。一個完全沒有缺失的欄位，兩邊的缺失箱都是 0 —— 那不是「出現了參考期
    沒有的東西」，它對 PSI 的貢獻是 0。不丟掉的話幾乎每一欄都會被標記，而
    一個永遠亮著的旗標等於沒有旗標（同 M5 的 `git_dirty`）。

    所以旗標的意思是精確的：**有某一箱只有一邊是空的**，此時 PSI 的大小由
    epsilon 決定，不可當成漂移的量來讀。
    """
    ref = scheme.reference
    cur = apply_bins(scheme, current)
    if scheme.kind == "discrete":
        # discrete 的 unseen 箱由 `apply_bins` 補在缺失箱之前；參考期必然是 0
        # （那些取值沒出現過），所以補一個 0 對齊長度。
        ref = np.insert(ref, ref.shape[0] - 1, 0.0)
        labels = (*scheme.labels[:-1], UNSEEN_LABEL, MISSING_LABEL)
    else:
        labels = scheme.labels

    keep = ~((ref <= 0) & (cur <= 0))
    value, floored = psi_from_proportions(ref[keep], cur[keep], epsilon=epsilon)
    detail = pl.DataFrame(
        {
            "feature": [scheme.feature] * len(labels),
            "bin": list(labels),
            "reference": ref,
            "current": cur,
            "contribution": [
                float(
                    (max(p, epsilon) - max(q, epsilon)) * np.log(max(p, epsilon) / max(q, epsilon))
                )
                for p, q in zip(ref, cur, strict=True)
            ],
        }
    )
    return value, floored, detail


def psi_table(
    reference: pl.DataFrame,
    current: pl.DataFrame,
    *,
    categorical: tuple[str, ...] = (),
    bins: int = DEFAULT_BINS,
    epsilon: float = EPSILON,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """整份特徵矩陣的 PSI，依大小遞減。

    Returns:
        (每欄一列的摘要, 逐箱明細)。摘要含缺失率的變化 —— 那是最容易解讀的
        一種漂移，而 PSI 把它跟其他變化混在一個數字裡。

    Raises:
        ValueError: 兩邊欄位不一致（比錯欄位的 PSI 沒有意義）。
    """
    if list(reference.columns) != list(current.columns):
        raise ValueError(
            "參考期與當期的欄位不一致，PSI 會逐欄比錯人。\n"
            f"  缺少：{sorted(set(reference.columns) - set(current.columns))}\n"
            f"  多出：{sorted(set(current.columns) - set(reference.columns))}"
        )

    rows, details = [], []
    for name in reference.columns:
        scheme = fit_bins(reference[name], bins=bins, categorical=name in categorical)
        value, floored, detail = psi(scheme, current[name], epsilon=epsilon)
        ref_missing = reference[name].null_count() / reference.height
        cur_missing = current[name].null_count() / current.height
        rows.append(
            {
                "feature": name,
                "psi": value,
                "band": band(value),
                "epsilon_floored": floored,
                "kind": scheme.kind,
                "n_bins": len(detail),
                "reference_missing": ref_missing,
                "current_missing": cur_missing,
                "missing_delta": cur_missing - ref_missing,
                "top_bin": detail.sort("contribution", descending=True)["bin"][0],
            }
        )
        details.append(detail)

    return pl.DataFrame(rows).sort("psi", descending=True), pl.concat(details)


def noise_floor(
    frame: pl.DataFrame,
    *,
    categorical: tuple[str, ...] = (),
    bins: int = DEFAULT_BINS,
    rounds: int = 5,
    seed: int = 42,
    epsilon: float = EPSILON,
) -> pl.DataFrame:
    """**完全沒有漂移**時 PSI 長什麼樣：同一個 cohort 隨機切兩半，比它自己。

    這是判讀 0.1 / 0.25 的基準線。慣例門檻沒有樣本數修正，而 PSI 對小樣本會
    因為抽樣雜訊而變大 —— 不先量這個，「0.08 算穩定」只是在讀慣例。

    切一半而不是 bootstrap：PSI 比較的是兩個獨立樣本，重抽會讓兩邊共用列，
    低估雜訊。

    Returns:
        每欄一列：`rounds` 次切分的 PSI 平均、最大，以及所有欄所有輪的
        95 百分位（欄位 `p95_all`，全表同一個值 —— 那是「整體雜訊地板」）。
    """
    if rounds < 2:
        raise ValueError(f"rounds 至少要 2：{rounds}")

    rng = np.random.default_rng(seed)
    per_feature: dict[str, list[float]] = {c: [] for c in frame.columns}
    for _ in range(rounds):
        mask = rng.permutation(frame.height)
        half = frame.height // 2
        a, b = frame[mask[:half]], frame[mask[half : 2 * half]]
        for name in frame.columns:
            scheme = fit_bins(a[name], bins=bins, categorical=name in categorical)
            value, _, _ = psi(scheme, b[name], epsilon=epsilon)
            per_feature[name].append(value)

    everything = [v for values in per_feature.values() for v in values]
    p95 = float(np.percentile(everything, 95))
    return pl.DataFrame(
        {
            "feature": list(per_feature),
            "psi_mean": [float(np.mean(v)) for v in per_feature.values()],
            "psi_max": [float(np.max(v)) for v in per_feature.values()],
            "rounds": [rounds] * len(per_feature),
            "p95_all": [p95] * len(per_feature),
        }
    ).sort("psi_max", descending=True)


def score_psi(
    reference: np.ndarray,
    current: np.ndarray,
    *,
    bins: int = DEFAULT_BINS,
    epsilon: float = EPSILON,
) -> tuple[float, bool, pl.DataFrame]:
    """**模型輸出**的 PSI（分數漂移）。

    與特徵 PSI 是兩個不同的監控，而且分數這個更接近我們真正在意的事：61 個
    特徵各自微幅移動，合起來可能讓分數大幅偏移，也可能互相抵消 —— 特徵 PSI
    看不出是哪一種。

    ⚠️ **參考期的分數不可以用訓練集的樣本內預測。** 樣本內的分布比實際更
    尖銳，於是「訓練集 vs 當期」的差異裡混進了「樣本內 vs 樣本外」，而那不是
    漂移。呼叫端要傳一塊模型沒有擬合過的資料（本專案用 Feb 內部那塊 early
    stopping 切分），這件事在報告裡要寫明。
    """
    ref = pl.Series("score", np.asarray(reference, dtype=np.float64))
    scheme = fit_bins(ref, bins=bins)
    return psi(scheme, pl.Series("score", np.asarray(current, dtype=np.float64)), epsilon=epsilon)
