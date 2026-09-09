"""user_logs 收聽行為聚合 —— M2 的核心。

## 問題規模

`user_logs` 兩檔合計 **410,502,905 列 / 31.9 GB**，是交易資料的 17.9 倍。
本機 RAM 31.1 GB，全量進記憶體不可能（SPEC §2.2）。

## 做法：先收斂，再聚合

關鍵洞察是 **SPEC §7 要的最長窗口只有 90 天**。因此 2015 年到 2016 年上半
的日誌完全用不到：

    Feb cohort  cutoff 2017-02-01 ~ 02-28  →  只需要 2016-11-03 起
    Mar cohort  cutoff 2017-03-01 ~ 03-31  →  只需要 2016-12-01 起
    兩者聯集                                   2016-11-03 ~ 2017-03-31

窗口外的列在掃描階段就被 predicate pushdown 丟掉，不會進到記憶體。收斂後
的結果存成 parquet，之後調整特徵定義時只要讀那份小檔，不必重掃 31.9 GB。

這是「先縮小再計算」的標準做法。反過來做 —— 先 join 再 filter —— 會需要
把 4 億列的 join 中間結果放進記憶體，那才是會 OOM 的寫法。

## as-of 截斷（紅線 2）

每位用戶的窗口是**相對於他自己的 cutoff**，不是統一日期。實作方式是先算

    days_before = cutoff − date

然後只保留 `days_before >= 0` 的列。與紅線 1 相同，守門檢查抽成
`assert_logs_within_cutoff()` 以便單獨測試。
"""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from src.config import Paths, load_paths
from src.data import COHORTS, CohortSpec, build_cohort
from src.fingerprint import cache_is_current, logic_fingerprint, write_with_fingerprint

# SPEC §7 指定的觀察窗口。
LOG_WINDOWS: tuple[int, ...] = (7, 14, 30, 90)
MAX_WINDOW = max(LOG_WINDOWS)

LOG_FILES = ("user_logs.csv", "user_logs_v2.csv")

# 播放次數的五個分桶。num_100 是完播（超過 98.5%），完播率的分子。
PLAY_COLUMNS = ("num_25", "num_50", "num_75", "num_985", "num_100")

# 收聽特徵的四個語意分組，用於消融實驗。
#
# 這是 RFM 框架套到訂閱行為上的版本：Recency（多久沒用）、Frequency（多常用）、
# Intensity（用得多深）、Trend（用量在漲還是在跌）。分組的用意不是整理，是要
# 回答一個具體問題 —— **M0 的 EDA 顯示活躍「水準」無法從 Feb 外推到 Mar，
# 只有「趨勢」兩邊方向一致。** 分組消融能把這個觀察變成量化結論。
LOG_GROUPS: tuple[str, ...] = ("recency", "frequency", "intensity", "trend")

_RECENCY = frozenset({"log_min_days_before", "log_max_days_before"})
_FREQUENCY_SUFFIX = ("_active_days", "_active_ratio")
_INTENSITY_SUFFIX = ("_secs", "_plays", "_unq", "_completed", "_completion", "_secs_per_active_day")


def log_feature_group(name: str) -> str | None:
    """把一個欄名歸類到 R/F/I/T。非收聽特徵回 None。

    用規則判斷而不是寫死清單：加了新窗口（例如 60 天）之後不必同步維護
    兩個地方，否則新特徵會靜靜地落在所有分組之外，消融實驗就漏掉它。
    """
    if not name.startswith("log"):
        return None
    if name in _RECENCY:
        return "recency"
    if name.startswith("log_trend"):
        return "trend"
    if name.endswith(_FREQUENCY_SUFFIX):
        return "frequency"
    if name.endswith(_INTENSITY_SUFFIX):
        return "intensity"
    return None  # log_has_logs 等不屬於任何行為分組的旗標


def expected_log_columns() -> frozenset[str]:
    """收聽特徵表**應該**有的欄位，由 `LOG_WINDOWS` 推導。

    刻意不寫死清單：加一個窗口（例如 60 天）之後，這個集合會自動變大，
    舊快取因此對不上而重算。寫死的話新窗口只會靜靜地不存在。
    """
    cols = {
        "msno",
        "cutoff",
        "log_min_days_before",
        "log_max_days_before",
        "log_has_logs",
        "log_trend_7_30",
        "log_trend_30_90",
        "log_trend_active_7_30",
    }
    for w in LOG_WINDOWS:
        cols |= {
            f"log{w}_active_days",
            f"log{w}_secs",
            f"log{w}_plays",
            f"log{w}_completed",
            f"log{w}_unq",
            f"log{w}_completion",
            f"log{w}_active_ratio",
            f"log{w}_secs_per_active_day",
        }
    return frozenset(cols)


def _to_date(yyyymmdd: int) -> date:
    s = str(yyyymmdd)
    return date(int(s[:4]), int(s[4:6]), int(s[6:]))


def _to_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day


def window_bounds(specs: tuple[CohortSpec, ...]) -> tuple[int, int]:
    """算出這些 cohort 合起來需要的日期範圍。

    下界是最早的 cutoff 再往前推 MAX_WINDOW 天；上界是最晚的 cutoff。
    範圍以外的日誌對任何特徵都沒有貢獻。

    ⚠️ **要扣掉 `lead_days`。** M6 的提前評分版本（§4.3）把 cutoff 往前移，
    它的 90 天窗口下界也跟著往前 —— 少扣這 7 天，`feb_t7` 最早那些人的窗口
    開頭會落在收斂檔之外，於是他們的收聽特徵少算 7 天。那不會報錯：紅線 2
    只檢查非負，缺資料它管不著（這也正是收斂檔把日期範圍寫進檔名的理由）。

    ⚠️ **固定評分日的 spec 不看到期區間。** 那種 cohort 所有人的 cutoff 都是
    `score_date`，所以窗口是 `[score_date − MAX_WINDOW, score_date]`。用到期區間
    去推會把上界拉到到期月底（例如 apr_fixed 的 20170430），而那超出資料涵蓋
    範圍 —— 收斂檔會被命名成一個「看起來更完整」的範圍，實際上多出來的那段
    一列資料都沒有。
    """

    def lower(s: CohortSpec) -> date:
        anchor = _to_date(s.score_date) if s.fixed_score_date else _to_date(s.expire_start)
        return anchor - timedelta(days=(0 if s.fixed_score_date else s.lead_days) + MAX_WINDOW)

    def upper(s: CohortSpec) -> int:
        if s.fixed_score_date:
            return s.score_date
        return _to_int(_to_date(s.expire_end) - timedelta(days=s.lead_days))

    return _to_int(min(lower(s) for s in specs)), max(upper(s) for s in specs)


def assert_logs_within_cutoff(df: pl.DataFrame) -> None:
    """紅線 2 守門：不得有任何 `user_logs.date > cutoff` 的日誌進入特徵。

    與紅線 1 的 `assert_asof_respected()` 同樣抽成獨立函式，理由也相同：
    SPEC §5 要求每條紅線都要有「會失敗的測試」，而測試必須能餵給它確實
    違規的輸入，確認它真的會 raise。

    檢查的是 `log_min_days_before`（最近一筆日誌距離 cutoff 幾天）。負值代表
    那筆日誌發生在到期日之後 —— 到期後的收聽行為是結果而不是原因，讓它進
    特徵等於用未來預測過去。

    Raises:
        AssertionError: 存在 log_min_days_before < 0 的列。
    """
    col = "log_min_days_before"
    if col not in df.columns:
        raise KeyError(f"缺少檢查所需的欄位 {col!r}")

    bad = df.filter(pl.col(col) < 0)
    if bad.height:
        worst = int(bad[col].min())
        raise AssertionError(
            f"紅線 2 違反：{bad.height:,} 位用戶的特徵含 cutoff 之後的日誌"
            f"（最嚴重的晚了 {-worst} 天）。as-of 截斷失效，本表不可使用。"
        )


def narrow_logs(
    paths: Paths | None = None,
    specs: tuple[CohortSpec, ...] | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """把 user_logs 收斂成只含需要的日期與用戶，寫成 parquet。

    這一步是整個 M2 記憶體策略的關鍵，也是唯一需要碰 31.9 GB 原始檔的地方。
    之後所有特徵計算都讀產出的 parquet。

    Returns:
        收斂後 parquet 的路徑。
    """
    paths = (paths or load_paths()).ensure()
    specs = specs or tuple(COHORTS.values())

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    # 檔名帶上實際的日期範圍。**快取命中的判斷不能只看「檔案在不在」** ——
    # 一份只為 Feb 收斂過的檔（上界 2017-02-28、只含 Feb 用戶）會被 Mar 的
    # 特徵計算直接沿用，而紅線 2 只檢查非負，缺資料它管不著。
    # 範圍寫進檔名之後，換一組 specs 自然就 miss。
    #
    # ⚠️ **日期範圍不夠，因為收斂還會依用戶過濾。**
    #
    # 下面的 semi-join 只留 `specs` 的標籤檔裡有的人。加一個新 cohort（M6 的
    # Kaggle 測試集帶進 907,471 位用戶）時，日期範圍可能**一天都沒變**（apr_fixed
    # 的評分日正好是 20170331，與原本的上界相同），於是舊檔命中 —— 而那份檔裡
    # 沒有新用戶的任何一列日誌。症狀是那批人全部變成「近 90 天沒有收聽紀錄」，
    # 而紅線 2 與欄位檢查都會通過。
    #
    # 所以檔名再帶一個**用戶集合的指紋**（標籤檔清單的雜湊）。舊的檔名不帶
    # 這一段，自然 miss 一次重收斂 —— 那是正確的代價。
    lo, hi = window_bounds(specs)
    who = hashlib.sha1("|".join(sorted({s.label_file for s in specs})).encode("utf-8")).hexdigest()[
        :8
    ]
    out = paths.interim / f"user_logs_window_{lo}_{hi}_{who}.parquet"
    if out.exists() and not force:
        log(f"讀取既有的收斂檔 {out.name}")
        return out

    span = (_to_date(hi) - _to_date(lo)).days + 1
    log(f"收斂 user_logs 至 {lo} ~ {hi}（{span} 天）...")

    missing = [f for f in LOG_FILES if not (paths.raw / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"缺少 {', '.join(missing)}。請執行 uv run python scripts/download.py --groups logs"
        )

    # 用戶集合：只留兩個 cohort 會用到的人。
    # user_logs 涵蓋的用戶遠多於任一 cohort，這一步能再砍掉一大塊。
    wanted = (
        pl.concat([pl.read_csv(paths.raw / s.label_file, columns=["msno"]) for s in specs])
        .unique()
        .lazy()
    )
    log(f"  目標用戶 {wanted.select(pl.len()).collect().item():,} 人")

    logs = pl.concat([pl.scan_csv(paths.raw / f) for f in LOG_FILES])

    # 用 semi-join 而不是 `is_in(msno_series)`：後者在 polars 新版會發出
    # ambiguous 的 DeprecationWarning（同型別集合無法分辨是集合成員判斷還是
    # 逐列比對），而 semi-join 語意明確，streaming 引擎也處理得比較好。
    #
    # 兩個 filter 都放在最前面，讓 predicate pushdown 在讀取階段就丟掉不要的
    # 列 —— 記憶體裡永遠不會同時存在超過一個 batch。順序反過來（先 join 再
    # filter）就得把 4 億列的中間結果放進記憶體，那才是會 OOM 的寫法。
    (
        logs.filter(pl.col("date").is_between(lo, hi))
        .join(wanted, on="msno", how="semi")
        .sink_parquet(out)
    )

    size_gb = out.stat().st_size / 1024**3
    log(f"  完成 → {out.name}（{size_gb:.2f} GiB）")
    return out


def log_features_fingerprint() -> str:
    """收聽特徵快取對應的程式版本指紋。

    涵蓋**本模組與 `src.data.cohort`**：窗口是相對於每位用戶的 cutoff 算的，
    所以 cutoff 的邏輯一改，這張表就過時了 —— 只看本模組會漏掉那一半。
    """
    import src.data.cohort as _cohort
    import src.features.logs as _self

    return logic_fingerprint(_self, _cohort)


# total_secs 換算成整數毫秒的倍率。見 _window_aggs 的說明。
SECS_SCALE = 1000


def _window_aggs(w: int) -> list[pl.Expr]:
    """單一窗口的聚合式。

    ⚠️ **`total_secs` 用整數毫秒相加，不直接對浮點求和。**

    浮點加法不可結合：(a+b)+c 與 a+(b+c) 會差最後幾個位元。polars 的
    group_by 是平行的，每次重建的分塊方式不保證一樣，於是同一份輸入算出來
    的 `log90_secs` 每次差約 1e-9。實測（scripts/verify_rebuild.py）三輪重建
    的收聽特徵內容指紋三個樣，差異全部落在這五個 `*_secs` 欄位上。

    1e-9 秒本身毫無意義，但 LightGBM 的直方圖分箱是有邊界的：一個剛好落在
    邊界上的值換邊，那棵樹就長得不一樣，early stopping 的停點也跟著變。
    §7.8 已經因為「重建快取就換一批分數」付過一次代價，這裡不留第二個入口。

    整數加法可交換也可結合，所以毫秒相加的結果與分塊方式無關。毫秒對「聽了
    多久」這件事遠超過需要的精度：90 天窗口的量級是 10^5 秒。

    其餘四欄是整數計數，本來就沒有這個問題。
    """
    inside = pl.col("days_before") < w
    return [
        # 有紀錄的天數。注意這不等於 w —— 沒打開 App 的日子不會有列。
        inside.sum().alias(f"log{w}_active_days"),
        (
            (pl.col("total_secs").filter(inside) * SECS_SCALE).round().cast(pl.Int64).sum()
            / SECS_SCALE
        ).alias(f"log{w}_secs"),
        pl.col("plays").filter(inside).sum().alias(f"log{w}_plays"),
        pl.col("num_100").filter(inside).sum().alias(f"log{w}_completed"),
        pl.col("num_unq").filter(inside).sum().alias(f"log{w}_unq"),
    ]


def _derived(w: int) -> list[pl.Expr]:
    """由聚合值算出的比率。除法一律擋分母為 0。"""
    plays = pl.col(f"log{w}_plays")
    active = pl.col(f"log{w}_active_days")
    secs = pl.col(f"log{w}_secs")
    return [
        # 完播率：播完的歌佔所有播放的比例。掐掉不聽是流失的前兆。
        pl.when(plays > 0)
        .then(pl.col(f"log{w}_completed") / plays)
        .otherwise(None)
        .alias(f"log{w}_completion"),
        # 活躍密度：這 w 天裡有幾成的日子有打開。
        (active / w).alias(f"log{w}_active_ratio"),
        # 有聽的那幾天平均聽多久。與 active_ratio 分開，因為「每天聽一點」
        # 和「偶爾聽很久」是不同的行為模式。
        pl.when(active > 0)
        .then(secs / active)
        .otherwise(None)
        .alias(f"log{w}_secs_per_active_day"),
    ]


def build_log_features(
    spec: CohortSpec | str,
    paths: Paths | None = None,
    *,
    force: bool = False,
    verbose: bool = True,
) -> pl.DataFrame:
    """算出某個 cohort 每人一列的收聽行為特徵。

    Args:
        spec:  CohortSpec 或 "feb" / "mar"。
        force: True 則忽略快取重算。

    Returns:
        每位用戶一列。含 msno 與所有 log 特徵。**只包含在窗口內有紀錄的
        用戶** —— 沒有紀錄的人不會出現，由呼叫端以 left join 處理。

    Raises:
        AssertionError: 產出的特徵含 cutoff 之後的日誌（紅線 2 破功）。
    """
    if isinstance(spec, str):
        spec = COHORTS[spec]
    paths = (paths or load_paths()).ensure()

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    cache = paths.interim / f"{spec.name}_log_features.parquet"
    fingerprint = log_features_fingerprint()

    if cache.exists() and not force and not cache_is_current(cache, fingerprint):
        # 指紋涵蓋本模組**與** src.data.cohort：收聽特徵的窗口是相對於每位
        # 用戶的 cutoff 算的，cutoff 的邏輯一改，這張表就過時了。
        log(f"快取 {cache.name} 的程式版本指紋不符（或缺少指紋），重算")

    elif cache.exists() and not force:
        cached = pl.read_parquet(cache)
        # ⚠️ **快取命中也要跑守門**，理由同 `build_cohort()`：快取是檔案不是
        # 保證。一張 `log_min_days_before` 為負的舊快取，代表裡面含有到期日
        # **之後**的收聽行為 —— 那是標籤的結果而不是原因，餵進模型分數會
        # 變好，因此不會有人察覺。
        #
        # ⚠️ **順序有意義：先驗洩漏，再驗過時。**
        #
        # 紅線 2 先跑 —— 一份含 cutoff 之後日誌的快取代表出了嚴重的事，
        # 必須整支中斷讓人看見，不能被「反正要重算」默默蓋過去。
        assert_logs_within_cutoff(cached)

        # 守門只讀 `log_min_days_before` 一欄，**不足以當 schema 檢查**。
        # 一份只有三欄的舊快取（例如改了 LOG_WINDOWS 之前產生的）會完整
        # 通過紅線 2，然後讓模型少掉 37 個特徵照樣訓練成功 —— 分數變差卻
        # 找不出原因。過時與洩漏是兩件事，處置也不同：過時重算即可。
        missing = expected_log_columns() - set(cached.columns)
        if missing:
            log(f"快取 {cache.name} 缺少欄位 {sorted(missing)}，重算")
        else:
            log(f"讀取快取 {cache.name}（{cached.height:,} 列）")
            return cached

    narrowed = narrow_logs(paths, verbose=verbose)

    # 每位用戶的 cutoff。從 cohort 表拿而不是自己重算，確保收聽特徵與交易
    # 特徵用的是**同一個** cutoff —— 兩處各自推導遲早會分歧。
    cutoffs = build_cohort(spec, paths, verbose=False).select("msno", "cutoff")
    log(f"建立 {spec.name} 收聽特徵（{cutoffs.height:,} 位用戶）...")

    date_col = pl.col("date").cast(pl.Int64).cast(pl.String).str.to_date("%Y%m%d")
    cutoff_col = pl.col("cutoff").cast(pl.Int64).cast(pl.String).str.to_date("%Y%m%d")

    per_user = (
        pl.scan_parquet(narrowed)
        .join(cutoffs.lazy(), on="msno", how="inner")
        .with_columns((cutoff_col - date_col).dt.total_days().alias("days_before"))
        # ---- 這一行就是紅線 2 ----
        # 只保留 cutoff 當天與之前的日誌。到期後的收聽行為是結果不是原因。
        .filter(pl.col("days_before") >= 0)
        .filter(pl.col("days_before") < MAX_WINDOW)
        .with_columns(pl.sum_horizontal(PLAY_COLUMNS).alias("plays"))
        .group_by("msno")
        .agg(
            # cutoff 一起帶出來。它不是特徵（`build_features` 會丟掉），
            # 而是**出身證明**：讓下游能驗證「這份收聽特徵是用哪個 cutoff
            # 算的」。少了它，一份用 Mar cutoff 算的日誌可以無聲無息地接到
            # Feb 的 cohort 上，而紅線 2 必然放行 —— 因為它檢查的是
            # log_min_days_before 相對於**自己那個** cutoff 是否非負。
            pl.col("cutoff").first(),
            pl.col("days_before").min().alias("log_min_days_before"),
            pl.col("days_before").max().alias("log_max_days_before"),
            *[e for w in LOG_WINDOWS for e in _window_aggs(w)],
        )
        .collect(engine="streaming")
    )

    out = per_user.with_columns(
        *[e for w in LOG_WINDOWS for e in _derived(w)],
        pl.lit(1.0).alias("log_has_logs"),
    )

    # 趨勢：短窗口的日均對長窗口的日均。
    # 比值而非迴歸斜率，因為比值不受單位影響、也不受離群日拖動，而且能直接
    # 讀成一句話：「最近一週的聽歌時間是最近一個月平均的幾倍」。
    # 0.2 代表活躍度掉到五分之一，那是很強的流失前兆。
    out = out.with_columns(
        _ratio("log7_secs", 7, "log30_secs", 30, "log_trend_7_30"),
        _ratio("log30_secs", 30, "log90_secs", 90, "log_trend_30_90"),
        _ratio("log7_active_days", 7, "log30_active_days", 30, "log_trend_active_7_30"),
    )

    assert_logs_within_cutoff(out)

    # ---- 依 msno 排序，理由與 build_cohort 相同 ----
    #
    # 實測（scripts/verify_rebuild.py，2026-08-10）：這張表的列順序**每次
    # 重建都不同** —— 又一個 group_by。它目前不影響任何分數，因為
    # `_attach_logs` 是以 cohort 的 msno 為左表 left join，輸出順序跟著左表；
    # 三輪重建的特徵矩陣指紋完全相同，證明了這件事。
    #
    # 那為什麼還要排：**「目前不影響」是一個關於呼叫端的假設，不是關於這張
    # 表的性質。** `build_log_features()` 是公開 API，任何一個直接拿它的順序
    # 用（zip、concat、依位置切分）的呼叫端都會踩到 §7.8 那個 bug 的翻版。
    # 排序的成本是一次 80 萬列的 sort，換掉整類問題。
    out = out.sort("msno")

    write_with_fingerprint(out, cache, fingerprint)
    covered = out.height / cutoffs.height
    log(f"  完成 {out.height:,} 列 × {out.width} 欄（覆蓋 {covered:.2%} 的 cohort 用戶）")
    log(f"  已快取 → {cache.name}")
    return out


def _ratio(num: str, num_days: int, den: str, den_days: int, alias: str) -> pl.Expr:
    """短窗口日均 ÷ 長窗口日均。分母為 0 時回 null。"""
    per_day_num = pl.col(num) / num_days
    per_day_den = pl.col(den) / den_days
    return pl.when(per_day_den > 0).then(per_day_num / per_day_den).otherwise(None).alias(alias)
