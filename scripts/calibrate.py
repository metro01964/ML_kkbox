"""M4 第二步 · fit isotonic 校準器，並把**收益與代價**一起量出來。

第一步（`scripts/calibration_report.py`）只量測不 fit，結論是選 isotonic ——
理由不是它通常較好，是 D7 → D8 的排序反轉讓 Platt 的單調平滑假設不成立。
這一支把校準器 fit 起來、套到 Mar，然後回答四個問題：

## 一、Feb-sel 上模型準不準（這決定其他三個問題怎麼讀）

第一步的 −27.7% 低估是在 **Mar** 上量的，但校準器只能 fit 在 **Feb-sel**。
兩者之間隔著一次 cohort 換月：Feb 的流失率 6.39%，Mar 是 8.99%，差 1.4 倍。

所以那個缺口至少有兩個成因，修得掉的只有第一個：

    機率曲線的形狀歪了      Feb-sel 上也看得到 → isotonic 修得掉
    Feb → Mar 的基準率漂移   Feb-sel 上看不到   → isotonic 修不掉

**因此本腳本先報 Feb-sel 的校準診斷，再報 Mar 的前後對照。** 如果 Feb-sel
上模型本來就準，那 fit 出來的映射會近似恆等，Mar 的低估會原封不動留著 ——
那不是失敗，是結論：那個缺口是分布漂移，得用別的方式處理，而且要寫進
SPEC §6.3 的限制。

## 二、校準修掉了多少（Mar 前後對照）

log loss / Brier / ECE / MCE / 整體偏差，全體與兩個分群各一組。

## 三、代價是什麼（排序解析度）

isotonic 把失準的區段壓平成同一個值，區塊內的相對順序全部消失。SPEC §6.2
的核心交付物是「投放給預測機率最高的前 K%」—— 門檻若落在一個大區塊裡，
那個 K% 要取誰就沒有依據了。所以 AUC 與同分結構要跟收益並排看。

⚠️ AUC **上升**不代表排序變好，見 `src/evaluation/metrics.py:roc_auc`。

## 四、還有多少是校準修得動的（洩漏對照組）

最後一列 `Mar-oracle` 是**故意違規**的：把校準器 fit 在 Mar 自己身上。它
不可上線，唯一的用途是當上界 —— 如果連它都修不掉剩下的偏差，那就代表
問題不在校準器選得好不好。`scripts/target_encoding.py` 用同樣的手法擺了
三個違規控制組，理由相同：把「不能這樣做」變成一個有數字的論證。

    uv run python scripts/calibrate.py
    make calibrate-fit
"""

from __future__ import annotations

import sys

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import polars as pl

from scripts.calibration_report import load_split_config
from src.config import load_paths
from src.evaluation import (
    brier_score,
    calibration_in_the_large,
    expected_calibration_error,
    fit_isotonic,
    log_loss,
    max_calibration_error,
    reliability_curve,
    repeat_vs_new,
    roc_auc,
    tie_profile,
)
from src.models.adopted import ADOPTED_MODEL, fit_adopted
from src.models.train import load_cohort_features
from src.models.tuning import three_way_split

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

SEGMENTS = ("全體", "重複用戶", "新進用戶")

# 對數軸的下界。實測最小的觀測頻率是 1.1e-4，不受影響；校準後的下界由零
# 區塊大小決定，也在這之上。
FLOOR = 1e-5


def metrics_row(label: str, y, p, *, with_auc: bool = True) -> dict:
    """一組 (標籤, 預測) 的完整指標。校準前後、各分群都走這一條路徑。"""
    curve = reliability_curve(y, p)
    large = calibration_in_the_large(y, p)
    row = {
        "分群": label,
        "人數": len(y),
        "實際流失率": round(large["實際流失率"], 5),
        "平均預測": round(large["平均預測"], 5),
        "相對偏差": round(large["相對偏差"], 4),
        "log loss": round(log_loss(y, p), 5),
        "Brier": round(brier_score(y, p), 6),
        "ECE": round(expected_calibration_error(curve), 5),
        "MCE": round(max_calibration_error(curve), 5),
        "箱數": curve.height,
    }
    row["AUC"] = round(roc_auc(y, p), 5) if with_auc else None
    return row


def comparison_table(scored: pl.DataFrame, p_cols: dict[str, str]) -> pl.DataFrame:
    """分群 × 階段的對照表。

    階段（校準前 / 校準後 / 洩漏對照）當作欄而不是分開三張表，是為了讓同一
    個分群的前後兩列相鄰 —— 要比的是「同一群人差多少」，不是「同一個指標
    在不同群裡差多少」。
    """
    rows = []
    for name in SEGMENTS:
        part = scored if name == "全體" else scored.filter(pl.col("segment") == name)
        if not part.height:
            continue
        for stage, col in p_cols.items():
            row = metrics_row(name, part["is_churn"], part[col])
            rows.append({"階段": stage, **row})
    return pl.DataFrame(rows)


def plot_before_after(scored: pl.DataFrame, cal, figdir) -> None:
    """三面板：全體 reliability、新進用戶 reliability、映射曲線本身。

    前兩張都用對數雙軸，理由與圖 09 相同 —— 九成樣本的預測落在 2% 以下，
    線性尺度上那九成疊成靠近原點的一撮，看不出偏離對角線多少。
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    ax_all, ax_new, ax_map = axes

    panels = [
        (ax_all, scored, "全體"),
        (ax_new, scored.filter(pl.col("segment") == "新進用戶"), "新進用戶"),
    ]
    for ax, part, title in panels:
        if not part.height:
            continue
        ax.plot(
            [FLOOR, 1], [FLOOR, 1], color="#999999", linestyle="--", linewidth=1, label="完美校準"
        )
        for col, label, color in (
            ("p_raw", "校準前", "#d62728"),
            ("p_cal", "校準後", "#1f77b4"),
        ):
            curve = reliability_curve(part["is_churn"], part[col])
            ax.plot(
                curve["平均預測"].clip(FLOOR),
                curve["實際流失率"].clip(FLOOR),
                marker="o",
                markersize=4,
                linewidth=1.6,
                color=color,
                label=f"{label}（{curve.height} 箱）",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(FLOOR, 1)
        ax.set_ylim(FLOOR, 1)
        ax.set_title(f"{title} · 點在對角線下方 = 低估", fontsize=11)
        ax.set_xlabel("平均預測機率")
        ax.set_ylabel("實際流失率")
        ax.legend(fontsize=8.5, loc="upper left")

    # --- 第三面板：映射本身。看得出哪一段被抬高、哪一段被壓平。---
    knots = cal.knots
    ax_map.plot(
        [FLOOR, 1], [FLOOR, 1], color="#999999", linestyle="--", linewidth=1, label="恆等映射"
    )
    ax_map.step(
        knots["原始預測"].clip(FLOOR),
        knots["校準後"].clip(FLOOR),
        where="post",
        color="#2ca02c",
        linewidth=1.6,
        label=f"isotonic（{cal.n_levels} 個等級）",
    )
    ax_map.set_xscale("log")
    ax_map.set_yscale("log")
    ax_map.set_xlim(FLOOR, 1)
    ax_map.set_ylim(FLOOR, 1)
    ax_map.set_title("校準映射：原始預測 → 校準後", fontsize=11)
    ax_map.set_xlabel("原始預測機率")
    ax_map.set_ylabel("校準後機率")
    ax_map.legend(fontsize=8.5, loc="upper left")

    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    labels = ["0.001%", "0.01%", "0.1%", "1%", "10%", "100%"]
    for ax in axes:
        # 刻度寫成百分比字串：中文字型缺 mathtext 指數的負號字符，預設標籤
        # 會渲染成 `10¤1`（與圖 09 同一個坑）。
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            axis.set_major_formatter(matplotlib.ticker.FixedFormatter(labels))
            axis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"isotonic 校準 · {ADOPTED_MODEL}（fit 在 Feb-sel，套到 Mar）· 校準前後與映射本身",
        fontsize=12,
    )
    path = figdir / "11_calibration_before_after.png"
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
        f" · 校準 {split.sel.X.height:,}\n"
    )

    print(f"訓練中（{ADOPTED_MODEL}，§7.12 正式採用的模型）...", flush=True)
    fitted = fit_adopted(split.train, split.es, all_train=feb)
    p_sel = fitted.predict(split.sel.X)
    print(f"  停在第 {fitted.best_iteration} 輪")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)

    # --- 一、校準器 fit 之前，先看 Feb-sel 準不準 ---------------------------
    print("\n" + "=" * 88)
    print("一、Feb-sel 的校準診斷（校準器 fit 在這一塊，所以先看它自己準不準）")
    print("=" * 88)
    sel_before = metrics_row("Feb-sel", split.sel.y, p_sel)
    print(pl.DataFrame([sel_before]))

    # --- 二、fit ------------------------------------------------------------
    cal = fit_isotonic(split.sel.y, p_sel)
    print("\n" + "=" * 88)
    print("二、fit 出來的映射")
    print("=" * 88)
    print(
        f"  樣本 {cal.n_fit:,} 筆　相異風險等級 {cal.n_levels:,} 個\n"
        f"  下界 {cal.floor:.3e}（零區塊 {cal.n_zero_block:,} 筆）　上界 {cal.ceiling:.6f}"
    )
    if cal.n_zero_block:
        print(
            "  ↑ 零區塊代表校準集裡有一段完全沒有正例。輸出 0 會讓 log loss 踩到\n"
            "    1e-15 的 clip（單筆 34.5），所以下界取 Jeffreys 後驗均值。"
        )

    # --- 三、套到 Mar -------------------------------------------------------
    p_raw = fitted.predict(mar.X)
    p_cal = cal.apply(p_raw)

    # 洩漏對照組：fit 在 Mar 自己身上。**不可上線**，只當上界。
    oracle = fit_isotonic(mar.y, p_raw)
    p_oracle = oracle.apply(p_raw)

    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_raw": p_raw,
            "p_cal": p_cal,
            "p_oracle": p_oracle,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )

    print("\n" + "=" * 88)
    print("三、Mar cohort：校準前 vs 校準後")
    print("=" * 88)
    table = comparison_table(
        scored,
        {"校準前": "p_raw", "校準後": "p_cal", "⚠️Mar-oracle": "p_oracle"},
    )
    # 拆成兩張表印，是因為 12 個欄位擠在同一行會被 polars 截成「…」，而被
    # 截掉的偏偏是 log loss 與相對偏差 —— 這一步要看的就是那兩欄。
    print("\n【校準品質】平均預測貼不貼近實際發生率")
    print(
        table.select(
            "階段", "分群", "人數", "實際流失率", "平均預測", "相對偏差", "ECE", "MCE", "箱數"
        )
    )
    print("\n【機率品質與排序】log loss 同時受兩者影響，AUC 只看排序")
    print(table.select("階段", "分群", "log loss", "Brier", "AUC"))

    # --- 四、代價：排序解析度 ------------------------------------------------
    print("\n" + "=" * 88)
    print("四、代價：校準壓掉了多少排序解析度")
    print("=" * 88)
    ties = pl.DataFrame(
        [
            {"階段": "校準前", **tie_profile(scored["p_raw"])},
            {"階段": "校準後", **tie_profile(scored["p_cal"])},
        ]
    )
    print(ties)

    at_floor = scored.filter(pl.col("p_cal") <= cal.floor)
    print(
        f"\n  踩到下界的有 {at_floor.height:,} 人（{at_floor.height / scored.height:.2%}），"
        f"其中 {int(at_floor['is_churn'].sum()):,} 人實際流失"
    )
    if at_floor.height:
        print(
            f"  這群人若被指派 0（沒有下界的話），單這一段就會讓 log loss 增加約 "
            f"{int(at_floor['is_churn'].sum()) * 34.5 / scored.height:.4f}"
        )

    # --- 五、log loss 為什麼變差：拆成下界區塊與其餘 -------------------------
    #
    # 「校準之後 log loss 變差」是這一步最需要解釋的結果。把總損失拆成兩段
    # （各段除以**總人數**，所以兩列相加等於整體 log loss），就看得出惡化
    # 集中在哪裡 —— 不必猜。
    print("\n" + "=" * 88)
    print("五、log loss 的惡化來自哪一段")
    print("=" * 88)
    parts = []
    for name, part in (
        ("下界區塊", at_floor),
        ("其餘", scored.filter(pl.col("p_cal") > cal.floor)),
    ):
        if not part.height:
            continue
        share = part.height / scored.height
        parts.append(
            {
                "區段": name,
                "人數": part.height,
                "佔比": round(share, 4),
                "實際流失率": round(float(part["is_churn"].mean()), 5),
                "校準前平均預測": round(float(part["p_raw"].mean()), 6),
                "校準後平均預測": round(float(part["p_cal"].mean()), 6),
                "貢獻(前)": round(log_loss(part["is_churn"], part["p_raw"]) * share, 5),
                "貢獻(後)": round(log_loss(part["is_churn"], part["p_cal"]) * share, 5),
            }
        )
    print(pl.DataFrame(parts))

    print("\n產生圖表...")
    plot_before_after(scored, cal, paths.figures)

    # --- 讀法 ---------------------------------------------------------------
    before = calibration_in_the_large(scored["is_churn"], scored["p_raw"])
    after = calibration_in_the_large(scored["is_churn"], scored["p_cal"])
    feb_rate, mar_rate = float(split.sel.y.mean()), float(mar.y.mean())

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    oracle_ece = expected_calibration_error(
        reliability_curve(scored["is_churn"], scored["p_oracle"])
    )
    raw_ece = expected_calibration_error(reliability_curve(scored["is_churn"], scored["p_raw"]))
    cal_ece = expected_calibration_error(reliability_curve(scored["is_churn"], scored["p_cal"]))

    print(
        f"Feb-sel 基準率 {feb_rate:.4%}　→　Mar 基準率 {mar_rate:.4%}"
        f"（漂移 {mar_rate / feb_rate - 1:+.1%}）\n"
        f"Feb-sel 相對偏差 {sel_before['相對偏差']:+.2%}"
        f"　→　校準器只看得到這個，也只修得掉這個\n"
        f"Mar 相對偏差 {before['相對偏差']:+.2%} → {after['相對偏差']:+.2%}"
        f"（修掉 {abs(before['相對偏差']) - abs(after['相對偏差']):+.2%}）\n"
        f"Mar ECE {raw_ece:.5f} → {cal_ece:.5f}　"
        f"洩漏對照組 {oracle_ece:.5f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
