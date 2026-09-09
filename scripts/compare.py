"""M3 · 三方模型比較的入口（SPEC §7 M3「LGBM / XGB / CatBoost 三方比較」）。

實作在 src/models/compare.py，這裡只負責被當成主程式執行、印報表。
分開的理由與 scripts/train.py 相同（見該檔說明）。

    uv run python scripts/compare.py
    make compare
"""

from __future__ import annotations

import sys

import polars as pl

from src.config import load_paths
from src.models.compare import (
    comparison_table,
    constant_baseline,
    load_comparison_config,
    log_comparison_to_mlflow,
    mean_ensemble,
    rank_agreement,
    run_comparison,
)


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_comparison_config()
    except FileNotFoundError as e:
        sys.exit(f"{e}\n請先執行 uv run python scripts/download.py")

    results, feb, mar = run_comparison(paths, cfg)
    baseline = constant_baseline(feb, mar)

    rows = list(results)
    if cfg.get("ensemble", {}).get("enabled", False) and len(results) > 1:
        rows.append(mean_ensemble(results, mar, feb))

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(170)

    print("\n" + "=" * 78)
    print("M3 · 三方比較（同一份特徵、同一個 Feb→Mar 切分、同一塊 early stopping 驗證集）")
    print("=" * 78)
    print(comparison_table(rows, baseline))
    print(f"\n常數基準 {baseline:.5f}（SPEC §3.3）")

    best = min(rows, key=lambda r: r.logloss)
    spread = max(r.logloss for r in results) - min(r.logloss for r in results)
    print(f"最佳：{best.name}　{best.logloss:.5f}")
    print(f"三家極差：{spread:.5f}（{spread / min(r.logloss for r in results):.2%}）")

    print("\n--- 特徵重要度排名一致性（三家的排名，依平均排名排序）---")
    print(rank_agreement(results))
    print(
        "\n注意：三家的 gain 定義不同（分裂增益 / total_gain / PredictionValuesChange），\n"
        "只有排名可比，絕對值不可比。"
    )

    run_ids = log_comparison_to_mlflow(rows, cfg, baseline)
    if run_ids:
        print(f"\nMLflow 已記錄 {len(run_ids)} 個 run（用 make mlflow 檢視）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
