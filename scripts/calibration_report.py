"""M4 第一步 · 機率校準的**診斷**（不 fit 校準器）。

SPEC §6.2 要求 reliability diagram，理由是「沒有校準，期望淨收益曲線算出來
的錢是假的」。這支腳本先把失準**量出來、畫出來**，再決定用什麼方法修。

## 為什麼先診斷再選方法

isotonic 與 Platt 修的是不同形狀的失準：

    Platt（sigmoid）   假設失準可以用一條 logistic 曲線描述 —— 適合
                       「整體過度自信 / 過度保守」這種單調而平滑的偏移
    Isotonic           只假設單調，形狀自由 —— 能修彎曲的失準，但自由度高，
                       小樣本時會過擬合成階梯

先挑方法再看圖，等於用預設值決定方法。所以這一步刻意不 fit 任何東西。

## 這張圖要分三群看

SPEC §4.5：新進用戶（9.2%）與重複用戶（90.8%）的流失率差 6.8 倍。整體
曲線會被 90.8% 的重複用戶主導，新進用戶那組的失準完全看不見 —— 而那正是
挽回名單最需要正確機率的族群。

## 模型用**正式採用的那一個**

⚠️ 本腳本一度用 LightGBM，而 §7.12 正式採用的是 CatBoost —— 於是「校準器
該不該上線」這個結論，是在一個不會上線的模型上得出的。現在一律走
`src.models.adopted`，超參數從 `configs/model_comparison.yaml` 讀，
腳本不自帶一份。

訓練用 Feb-train、early stopping 用 Feb-es，切分來自 `configs/calibration.yaml`。
**Feb-sel 這一塊刻意留著不用** —— 下一步的校準器要 fit 在它上面，現在先碰
它就等於提前用掉。**Mar 全程只在最後評估一次，標籤不參與任何決定。**

    uv run python scripts/calibration_report.py
    make calibrate
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import polars as pl

from src.config import load_paths
from src.evaluation import (
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    log_loss,
    max_calibration_error,
    reliability_curve,
    repeat_vs_new,
)
from src.models.adopted import ADOPTED_MODEL, fit_adopted
from src.models.train import load_cohort_features
from src.models.tuning import three_way_split

# 與 notebooks/eda_01_overview.py 相同的字體設定 —— Windows 的 matplotlib
# 預設字體沒有中文，不設會變成 □□□。
matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

SEGMENTS = ("全體", "重複用戶", "新進用戶")
COLORS = {"全體": "#333333", "重複用戶": "#1f77b4", "新進用戶": "#d62728"}


def load_split_config() -> dict:
    """校準用**自己的**三段切分（`configs/calibration.yaml`）。

    ⚠️ 早期版本直接讀 `configs/tuning.yaml`，於是同一批 15% 的用戶依序被用來
    挑 null importance 門檻、挑超參數、再 fit 校準器 —— 那一塊作為「乾淨保留
    集」的身分已經被消耗過兩次。現在用不同的 seed，期望重疊降到約 15%。
    理由與殘餘風險寫在該設定檔裡。
    """
    import yaml

    from src.config import REPO_ROOT

    path = REPO_ROOT / "configs" / "calibration.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "split" not in cfg:
        raise KeyError(f"{path} 缺少 [split] 區段")
    return cfg


def curves_by_segment(scored: pl.DataFrame, strategy: str) -> dict[str, pl.DataFrame]:
    """全體與兩個分群各一條 reliability 曲線。"""
    out = {"全體": reliability_curve(scored["is_churn"], scored["p_churn"], strategy=strategy)}
    for name in ("重複用戶", "新進用戶"):
        part = scored.filter(pl.col("segment") == name)
        if part.height:
            out[name] = reliability_curve(part["is_churn"], part["p_churn"], strategy=strategy)
    return out


def summary_table(scored: pl.DataFrame, curves: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """每個分群一列的校準摘要。"""
    rows = []
    for name in SEGMENTS:
        part = scored if name == "全體" else scored.filter(pl.col("segment") == name)
        if not part.height or name not in curves:
            continue
        large = calibration_in_the_large(part["is_churn"], part["p_churn"])
        rows.append(
            {
                "分群": name,
                "人數": part.height,
                "實際流失率": round(large["實際流失率"], 5),
                "平均預測": round(large["平均預測"], 5),
                "相對偏差": round(large["相對偏差"], 4),
                "log loss": round(log_loss(part["is_churn"], part["p_churn"]), 5),
                "Brier": round(brier_score(part["is_churn"], part["p_churn"]), 6),
                "ECE": round(expected_calibration_error(curves[name]), 5),
                "MCE": round(max_calibration_error(curves[name]), 5),
            }
        )
    return pl.DataFrame(rows)


def plot_reliability(curves: dict[str, pl.DataFrame], figdir, strategy: str) -> None:
    """兩張並排：左邊全範圍、右邊放大到低機率區。

    為什麼需要放大圖：九成的樣本擠在預測機率 0.2 以下，全範圍圖上那一段
    被壓成靠近原點的一小撮，看不出偏離對角線多少 —— 而那一小撮就是絕大
    多數的挽回名單。
    """
    fig, (ax_full, ax_log) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # --- 左：線性全範圍。看整體形狀與高風險端。---
    ax_full.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1, label="完美校準")
    for name, curve in curves.items():
        ax_full.plot(
            curve["平均預測"],
            curve["實際流失率"],
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=COLORS[name],
            label=name,
        )
    ax_full.set_xlim(0, 1)
    ax_full.set_ylim(0, 1)
    ax_full.set_title("線性尺度 · 全範圍", fontsize=11)
    ax_full.legend(fontsize=9, loc="upper left")

    # --- 右：對數雙軸。---
    #
    # 為什麼不用「放大到 0~0.2」：前五箱的平均預測都在 0.0001 附近，線性
    # 放大之後仍然疊在原點上，看不出任何東西。九成的樣本落在那個區間，
    # 而它們正是挽回名單裡「不必投放」的那群 —— 判斷模型有沒有把他們的
    # 風險估對，需要跨三個數量級的解析度，只有對數軸做得到。
    #
    # 對數軸上完美校準仍是一條直線（log y = log x），所以判讀方式不變：
    # 點在線下方 = 低估。
    floor = 1e-5  # 對數軸不能有 0；實測最小的觀測頻率是 1.1e-4，不受影響
    ax_log.plot([floor, 1], [floor, 1], color="#999999", linestyle="--", linewidth=1)
    for name, curve in curves.items():
        ax_log.plot(
            curve["平均預測"].clip(floor),
            curve["實際流失率"].clip(floor),
            marker="o",
            markersize=4,
            linewidth=1.6,
            color=COLORS[name],
        )
    ax_log.set_xscale("log")
    ax_log.set_yscale("log")
    ax_log.set_xlim(floor, 1)
    ax_log.set_ylim(floor, 1)
    ax_log.set_title("對數尺度 · 低機率端也看得見（九成樣本在此）", fontsize=11)

    # 刻度自己寫成百分比字串，不用 matplotlib 的預設科學記號。
    #
    # 兩個理由：一是中文字型（Microsoft JhengHei）缺 mathtext 指數用的負號
    # 字符，預設標籤會渲染成 `10¤1` 這種亂碼；二是「0.01%」對讀報告的人
    # 遠比「10⁻⁴」直觀，而這張圖的重點是給人看的。
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    labels = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]
    for axis in (ax_log.xaxis, ax_log.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())

    for ax in (ax_full, ax_log):
        ax.set_xlabel("平均預測機率")
        ax.set_ylabel("實際流失率")
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"Reliability diagram · Mar cohort · {ADOPTED_MODEL}（{strategy} 分箱）"
        "　點在對角線下方 = 低估流失風險",
        fontsize=12,
    )
    path = figdir / "09_reliability_diagram.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")


def plot_gap_by_decile(scored: pl.DataFrame, figdir) -> None:
    """依預測機率分十等分，畫「平均預測 vs 實際」的長條對照。

    reliability diagram 看的是形狀，這張看的是**每一段各有多少人、差多少**。
    投放決策是「取前 K%」，所以按分位數切才對得上業務動作。
    """
    curve = reliability_curve(scored["is_churn"], scored["p_churn"], n_bins=10)
    x = np.arange(curve.height)
    width = 0.4

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - width / 2, curve["平均預測"] * 100, width, label="平均預測", color="#7fb3d5")
    ax.bar(x + width / 2, curve["實際流失率"] * 100, width, label="實際流失率", color="#e59866")
    # 只標 D6 起的缺口：D1~D5 的實際流失率都低於 0.15%，長條在圖上是一條線，
    # 標上去只會變成一排壓在軸上的小字。缺口大到值得看的就是後五段。
    for i, row in enumerate(curve.iter_rows(named=True)):
        gap = row["實際流失率"] - row["平均預測"]
        if row["實際流失率"] < 0.005:
            continue
        ax.text(
            i,
            row["實際流失率"] * 100 + 1.5,
            f"低估 {gap * 100:.1f} pp",
            ha="center",
            fontsize=8.5,
            color="#a04000",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i + 1}" for i in x])
    ax.set_xlabel("預測機率十分位（D10 = 風險最高的 10%）")
    ax.set_ylabel("流失率 (%)")
    ax.set_ylim(0, 75)
    ax.set_title(
        "每個十分位的預測 vs 實際 —— 十段全部低估，D10 差 9.7 個百分點",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    path = figdir / "10_calibration_by_decile.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")


def main() -> int:
    try:
        paths = load_paths().ensure()
        cfg = load_split_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    feb, mar = load_cohort_features(paths, cfg)
    split = three_way_split(feb, cfg["split"])
    print(
        f"  Feb 三段切分：訓練 {split.train.X.height:,}"
        f" · early stopping {split.es.X.height:,}"
        f" · 保留給校準器 {split.sel.X.height:,}（本步驟不使用）\n"
    )

    print(f"訓練中（{ADOPTED_MODEL}，§7.12 正式採用的模型）...", flush=True)
    fitted = fit_adopted(split.train, split.es, all_train=feb)
    pred = fitted.predict(mar.X)
    print(f"  停在第 {fitted.best_iteration} 輪，Mar log loss {log_loss(mar.y, pred):.5f}")

    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(180)

    curves = curves_by_segment(scored, strategy="quantile")

    print("\n" + "=" * 78)
    print("校準診斷 · Mar cohort（校準器尚未套用）")
    print("=" * 78)
    print(summary_table(scored, curves))

    print("\n--- 全體 reliability 曲線（等量分箱，每箱約 9.7 萬人）---")
    print(
        curves["全體"].select(
            "bin",
            pl.col("平均預測").round(5),
            pl.col("實際流失率").round(5),
            pl.col("差距").round(5),
            "樣本數",
        )
    )

    print("\n產生圖表...")
    plot_reliability(curves, paths.figures, strategy="quantile")
    plot_gap_by_decile(scored, paths.figures)

    large = calibration_in_the_large(scored["is_churn"], scored["p_churn"])
    print("\n" + "=" * 78)
    print("讀法")
    print("=" * 78)
    print(
        f"整體：平均預測 {large['平均預測']:.4%} vs 實際 {large['實際流失率']:.4%}"
        f"　→ 相對偏差 {large['相對偏差']:+.2%}"
    )
    print(
        "\n這個偏差會等比例縮小 SPEC §6.1 的每一筆期望收益\n"
        "（E[淨收益] = p_churn × r_save × LTV − C_offer 是**乘上** p_churn），\n"
        "使投放門檻設得過於保守 —— 這就是 M4 要修的東西。\n"
        "\n⚠️ 下一步才 fit 校準器，而且要 fit 在 Feb-sel（模型與 early stopping\n"
        "   都沒看過的那一塊）。fit 在 Mar 上會讓這張圖變漂亮，但那是拿\n"
        "   驗證集調自己的成績單 —— M3 的 §7.5 第 6 項剛修掉同一個錯。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
