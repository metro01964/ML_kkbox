"""收聽特徵的消融實驗。

M2 的結果是 38 個收聽特徵只換來 3.15%。這支腳本回答兩個後續問題：

  1. **收聽行為到底有沒有獨立訊號？** 丟掉全部交易特徵單獨訓練一次。
     如果分數接近常數基準 0.30746，代表收聽資料對這題就是沒用，
     不必再投資 —— 這是 go / no-go 閘門。

  2. **哪一組有訊號？** 把收聽特徵拆成 Recency / Frequency / Intensity /
     Trend 四組，每次只加一組。M0 的 EDA 顯示活躍「水準」無法從 Feb 外推
     到 Mar、只有「趨勢」兩邊方向一致 —— 這裡把那個觀察變成數字。

## 為什麼用 only-one 而不是 leave-one-out

兩者回答不同問題：

    only-one（交易 + 單一組）     這組**本身**有多少獨立訊號
    leave-one-out（全部 − 一組）  移掉它會損失多少（給定其他組還在）

相關的特徵組在 leave-one-out 會互相掩護，看起來每組都不重要。要判斷
「哪組能外推」必須用 only-one。

leave-one-out 這裡刻意不做 —— 那回答的是「哪組可以砍」，而砍特徵是 M3 的
null importance 篩選該做的事，先做會重複。

## 不跑 5-fold

消融比較的是**時間外分數之間的差異**，而 5-fold 只估計 Feb 內部的變異數
（SPEC §4.2），對這個問題沒有貢獻，卻會讓每次實驗多花三倍時間。
確定最終設定之後再跑一次完整的 make train 補上標準差。

    uv run python scripts/ablation.py
"""

from __future__ import annotations

import sys
import time

import polars as pl

from src.config import load_paths
from src.data import FEB, build_cohort
from src.features import LOG_GROUPS, build_features, build_log_features, log_feature_group
from src.models.train import load_model_config, train_baseline


def build_variants(all_columns: list[str]) -> dict[str, list[str]]:
    """列出每個實驗要保留的欄位。"""
    txn = [c for c in all_columns if not c.startswith("log")]
    logs = [c for c in all_columns if c.startswith("log")]
    grouped = {g: [c for c in logs if log_feature_group(c) == g] for g in LOG_GROUPS}

    # 不屬於任何行為分組的收聽旗標（目前只有 log_has_logs）。
    ungrouped = [c for c in logs if log_feature_group(c) is None]

    variants = {
        "txn_only（= M1 對照）": txn,
        "log_only（無交易特徵）": logs,
    }
    for g in LOG_GROUPS:
        variants[f"txn + {g}"] = txn + grouped[g]
    variants["all（= M2 對照）"] = all_columns

    print("各組的特徵數：")
    print(f"  交易與用戶屬性  {len(txn)}")
    for g in LOG_GROUPS:
        print(f"  {g:14s}  {len(grouped[g])}")
    if ungrouped:
        print(f"  未分組          {len(ungrouped)}  {ungrouped}")
    print()
    return variants


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_model_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # 先建一次完整特徵表，只為了拿欄名清單。
    sample = build_features(
        build_cohort(FEB, paths, verbose=False),
        build_log_features(FEB, paths, verbose=False),
    )
    variants = build_variants(sample.X.columns)

    rows = []
    for i, (name, cols) in enumerate(variants.items(), 1):
        print(f"[{i}/{len(variants)}] {name}　（{len(cols)} 特徵）", flush=True)
        t0 = time.perf_counter()
        r = train_baseline(paths, cfg, keep_features=cols, verbose=False)
        secs = time.perf_counter() - t0

        seg = {row["分群"]: row["log_loss"] for row in r.segments.iter_rows(named=True)}
        rows.append(
            {
                "實驗": name,
                "特徵數": len(cols),
                "輪數": r.best_iteration,
                "Mar log loss": round(r.logloss, 5),
                "重複用戶": round(seg["重複用戶"], 5),
                "新進用戶": round(seg["新進用戶"], 5),
                "秒": round(secs),
            }
        )
        print(f"      → {r.logloss:.5f}　({secs:.0f} 秒)\n", flush=True)

    table = pl.DataFrame(rows)
    baseline = table.filter(pl.col("實驗").str.starts_with("txn_only"))["Mar log loss"][0]
    table = table.with_columns(((pl.col("Mar log loss") / baseline) - 1).alias("相對 M1"))

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(160)
    print("=" * 78)
    print("消融結果（全部以時間外 Mar cohort 分數比較）")
    print("=" * 78)
    print(table)

    log_only = table.filter(pl.col("實驗").str.starts_with("log_only"))["Mar log loss"][0]
    const = 0.30746
    print(f"\n常數基準 {const}")
    print(f"log_only {log_only:.5f}　—— 距離常數基準 {const - log_only:+.5f}")
    if log_only > const * 0.9:
        print("→ 收聽行為的獨立訊號很弱，不建議再投資 M2。")
    else:
        print("→ 收聽行為有獨立訊號，只是被交易特徵蓋住。值得往有效的分組加特徵。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
