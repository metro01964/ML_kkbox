"""M4 · 業務指標：期望模擬淨收益曲線與敏感度熱圖（SPEC §6.1、§6.2）。

SPEC §6.2 稱第一張圖是「整個專案的核心交付物」。它把模型輸出翻譯成一個
營運可以執行的決定：**這個月該投放給哪一批人。**

    E[淨收益] = p_churn × r_save × LTV_saved − C_offer

## 三個參數的地位完全不同

    LTV_saved   由資料推導 —— 月費實測 × 預期續訂月數（1 / 月流失率）
    r_save      純假設 —— 本資料集沒有實驗組，無法估計（SPEC §6.3）
    C_offer     業務決策 —— 不是估計值，是「我們打算送什麼」

決策門檻只取決於 `p* = C_offer / (r_save × LTV_saved)`，三者的不確定性因此
會互相吸收（LTV 打三折等於 C_offer 乘 3.33），敏感度熱圖掃的就是這件事。

## 預期續訂月數用**上一期**的流失率

`expected_months = 1 / 月流失率`。用 Feb 的 6.39% 而不是 Mar 的 8.99%：
Mar 是評估集的標籤，拿它設業務常數等於用到答案。兩者差距（15.6 vs 11.1
個月）落在 C_offer 的掃描區間裡，所以這個選擇不會左右結論 —— 但它會左右
基準情境的門檻，因此兩個都印出來。

## 為什麼要畫兩條曲線

    期望模擬淨收益      p 用**模型預測**算 —— 模型以為會賺多少
    標籤結算模擬淨收益  p 用**真實標籤 y**（0/1）算 —— 把預測換成答案再算一次

⚠️ **兩條都是模擬，都不是真的賺到的錢。** 真實的只有 Mar 的流失標籤；
`r_save` 與 `LTV_saved` 仍是假設，而且本資料集沒有實驗組／對照組，
**無法宣稱任何投放真的改變了行為**（SPEC §6.3 第一點）。

兩條線的落差量的是機率誤差的代價：低估流失風險 → 期望曲線悲觀 →
極大值往左偏 → 門檻設得過於保守。

⚠️ 標籤結算那條用到 Mar 的標籤，只能事後回顧。部署時沒有它。

## 用 CatBoost 而不是 LightGBM

§7.12 正式採用 CatBoost（Mar 0.15367，配對 8/8）。業務曲線必須用會上線的
那一個，模型與超參數一律走 `src.models.adopted`，腳本不自帶一份。

⚠️ 本腳本會量 CatBoost 在 Mar 全體的平均低估，但**那個數字不能往下套**到
個別用戶、風險區間、投放名單或淨收益 —— 投放名單自己的偏差另外報。

    uv run python scripts/business_value.py
    make eval
"""

from __future__ import annotations

import json
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.evaluation import (
    calibration_in_the_large,
    campaign_curve,
    expected_months,
    fixed_rule_point,
    log_loss,
    optimal_point,
    repeat_vs_new,
    resolve_assumptions,
    sensitivity_grid,
    subset_calibration,
)
from src.models.adopted import ADOPTED_MODEL, fit_adopted, load_adopted_config
from src.models.compare import split_for_early_stopping
from src.models.train import load_cohort_features

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110


def load_configs() -> dict:
    """業務參數。**模型超參數不在這裡** —— 由 `src.models.adopted` 從
    `configs/model_comparison.yaml` 讀，全專案只有那一份。"""
    biz_path = REPO_ROOT / "configs" / "business.yaml"
    if not biz_path.exists():
        raise FileNotFoundError(f"找不到 {biz_path}")
    biz = yaml.safe_load(biz_path.read_text(encoding="utf-8")) or {}
    for key in ("business", "sensitivity", "curve"):
        if key not in biz:
            raise KeyError(f"{biz_path} 缺少 [{key}] 區段")
    return biz


def campaign_table(scored: pl.DataFrame, biz: dict, ltv: float, step: float) -> pl.DataFrame:
    """全體與兩個分群各自的最佳投放點（依期望曲線決定，用標籤結算）。"""
    rows = []
    for name in ("全體", "重複用戶", "新進用戶"):
        part = scored if name == "全體" else scored.filter(pl.col("segment") == name)
        if not part.height:
            continue
        curve = campaign_curve(
            part["is_churn"],
            part["p_churn"],
            r_save=biz["r_save"],
            ltv_saved=ltv,
            c_offer=biz["c_offer"],
            step=step,
        )
        best = optimal_point(curve, by="期望模擬淨收益")
        rows.append(
            {
                "分群": name,
                "人數": part.height,
                "最佳投放比例": round(best["K"], 4),
                "投放人數": best["投放人數"],
                "門檻機率": round(best["門檻機率"], 4),
                "期望模擬淨收益": round(best["期望模擬淨收益"]),
                "標籤結算模擬淨收益": round(best["標籤結算模擬淨收益"]),
                "命中率": round(best["命中率"], 4),
                "lift": round(best["lift"], 2),
            }
        )
    return pl.DataFrame(rows)


def plot_curve(scored: pl.DataFrame, curve: pl.DataFrame, biz: dict, ltv: float, figdir) -> None:
    """核心交付圖：期望模擬淨收益 vs 投放比例。

    ## 為什麼左邊要放大

    曲線在 K 很小的地方就到頂，之後一路掉到 −1.2 億（投放給所有人，每人
    虧一個 C_offer）。畫全範圍的話，**唯一有決策意義的那一段會被壓在左緣
    幾個像素裡** —— 圖看起來很有戲劇性，卻回答不了「該投放給誰」。

    所以左邊放大到決策區間、右邊留全範圍當脈絡。兩張都要：只看放大圖會
    忘記代價有多陡，只看全範圍圖則看不出最佳點在哪。
    """
    fig, (ax, ax_full, ax_seg) = plt.subplots(1, 3, figsize=(17, 5))

    k = curve["K"].to_numpy() * 100
    exp = curve["期望模擬淨收益"].to_numpy() / 1e6
    act = curve["標籤結算模擬淨收益"].to_numpy() / 1e6

    base_rate = float(scored["is_churn"].mean())
    n = scored.height
    rand = (curve["K"].to_numpy() * n) * (base_rate * biz["r_save"] * ltv - biz["c_offer"]) / 1e6

    new_mask = (scored["segment"] == "新進用戶").to_numpy()
    rule = fixed_rule_point(
        scored["is_churn"],
        new_mask,
        r_save=biz["r_save"],
        ltv_saved=ltv,
        c_offer=biz["c_offer"],
        label="全部新進用戶",
    )

    i_exp, i_act = int(np.argmax(exp)), int(np.argmax(act))
    zoom = max(25.0, k[i_act] * 2.5)  # 決策區間：至少涵蓋兩個最佳點

    for axis, xmax, title in (
        # 「≤」在 Microsoft JhengHei 缺字會渲染成方框（與圖 09 的 mathtext 負號
        # 同一類問題），改用中文寫法。
        (ax, zoom, f"決策區間（K 在 {zoom:.0f}% 以內）"),
        (ax_full, 100.0, "全範圍 —— 代價有多陡"),
    ):
        axis.axhline(0, color="#999999", linewidth=1)
        axis.plot(k, exp, color="#1f77b4", linewidth=1.8, label="期望模擬淨收益（模型以為）")
        axis.plot(k, act, color="#d62728", linewidth=1.8, label="標籤結算模擬淨收益（事後結算）")
        axis.plot(k, rand, color="#999999", linestyle=":", linewidth=1.5, label="隨機排序")
        axis.plot(
            rule["K"] * 100, rule["標籤結算模擬淨收益"] / 1e6, "s", color="#2ca02c", markersize=8
        )
        axis.set_xlim(0, xmax)
        axis.set_xlabel("投放比例 K（%，依預測機率由高到低）")
        axis.set_ylabel("淨收益（百萬元）")
        axis.set_title(title, fontsize=11)
        axis.grid(alpha=0.25)

    # 放大圖的 y 範圍由決策區間內的資料決定，不被右半段的深谷拉扁。
    inside = k <= zoom
    lo = min(float(act[inside].min()), float(exp[inside].min()), rule["標籤結算模擬淨收益"] / 1e6)
    hi = max(float(act.max()), float(exp.max()))
    ax.set_ylim(lo - 0.5, hi + 1.6)

    for series, i, color, label, dy in (
        (exp, i_exp, "#1f77b4", "期望", -34),
        # 實際最佳點在曲線頂端，標註往上放會撞到子標題 —— 改放右下。
        (act, i_act, "#d62728", "實際", -6),
    ):
        ax.plot(k[i], series[i], "o", color=color, markersize=7)
        ax.annotate(
            f"{label}最佳　K={k[i]:.1f}%　{series[i]:.2f} 百萬",
            (k[i], series[i]),
            textcoords="offset points",
            xytext=(10, dy),
            fontsize=9,
            color=color,
        )
    ax.annotate(
        f"全部新進用戶（不需模型）\n{rule['標籤結算模擬淨收益'] / 1e6:.2f} 百萬",
        (rule["K"] * 100, rule["標籤結算模擬淨收益"] / 1e6),
        textcoords="offset points",
        xytext=(12, -6),
        fontsize=9,
        color="#2ca02c",
    )
    ax.legend(fontsize=8.5, loc="lower right")

    # --- 右：分群 ---
    for name, color in (("重複用戶", "#1f77b4"), ("新進用戶", "#d62728")):
        part = scored.filter(pl.col("segment") == name)
        c = campaign_curve(
            part["is_churn"],
            part["p_churn"],
            r_save=biz["r_save"],
            ltv_saved=ltv,
            c_offer=biz["c_offer"],
            step=0.002,
        )
        ax_seg.plot(
            c["K"].to_numpy() * 100,
            c["標籤結算模擬淨收益"].to_numpy() / 1e6,
            color=color,
            linewidth=1.8,
            label=f"{name}（{part.height:,} 人）",
        )
    ax_seg.axhline(0, color="#999999", linewidth=1)
    ax_seg.set_xlim(0, 60)
    ax_seg.set_ylim(-8, 4)
    ax_seg.set_xlabel("該分群內部的投放比例 K（%）")
    ax_seg.set_ylabel("標籤結算模擬淨收益（百萬元）")
    ax_seg.set_title("分群內部排序是否仍有價值（§4.5 第 3 點）", fontsize=11)
    ax_seg.legend(fontsize=9, loc="lower left")
    ax_seg.grid(alpha=0.25)

    fig.suptitle(
        f"期望模擬淨收益 vs 投放比例　r_save={biz['r_save']:.0%}"
        f"　C_offer={biz['c_offer']:.0f} 元　LTV={ltv:.0f} 元"
        f"　→ 門檻 p* = {biz['c_offer'] / (biz['r_save'] * ltv):.4f}",
        fontsize=12,
    )
    path = figdir / "12_expected_net_revenue.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")


def plot_sensitivity(grid: pl.DataFrame, biz: dict, figdir) -> None:
    """敏感度熱圖：r_save × C_offer 網格上的最佳投放比例。"""
    r_vals = sorted(grid["r_save"].unique().to_list())
    c_vals = sorted(grid["c_offer"].unique().to_list())
    k = np.zeros((len(c_vals), len(r_vals)))
    rev = np.zeros_like(k)
    for row in grid.iter_rows(named=True):
        i, j = c_vals.index(row["c_offer"]), r_vals.index(row["r_save"])
        k[i, j] = row["最佳投放比例"] * 100
        rev[i, j] = row["標籤結算模擬淨收益"] / 1e6

    fig, (ax_k, ax_r) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt, cmap in (
        (ax_k, k, "最佳投放比例 K（%）", "{:.1f}", "viridis"),
        (ax_r, rev, "照此門檻投放的標籤結算模擬淨收益（百萬元）", "{:+.1f}", "RdYlGn"),
    ):
        im = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(r_vals)), [f"{v:.0%}" for v in r_vals])
        ax.set_yticks(range(len(c_vals)), [f"{v:.0f}" for v in c_vals])
        ax.set_xlabel("r_save（挽回成功率）")
        ax.set_ylabel("C_offer（單次投放成本，元）")
        ax.set_title(title, fontsize=11)
        for i in range(len(c_vals)):
            for j in range(len(r_vals)):
                ax.text(
                    j,
                    i,
                    fmt.format(data[i, j]),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white" if cmap == "viridis" else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046)

    # 基準情境的位置
    for ax in (ax_k, ax_r):
        if biz["r_save"] in r_vals and biz["c_offer"] in c_vals:
            ax.plot(
                r_vals.index(biz["r_save"]),
                c_vals.index(biz["c_offer"]),
                marker="s",
                markersize=22,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
            )

    fig.suptitle(
        "參數敏感度：紅框為基準情境。決策只取決於 p* = C_offer / (r_save × LTV)",
        fontsize=12,
    )
    path = figdir / "13_sensitivity_heatmap.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")


def main() -> int:
    try:
        paths = load_paths().ensure()
        biz_cfg = load_configs()
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    biz, sens, curve_cfg = biz_cfg["business"], biz_cfg["sensitivity"], biz_cfg["curve"]
    _, train_cfg = load_adopted_config()

    feb, mar = load_cohort_features(paths, biz_cfg)
    train, es = split_for_early_stopping(feb, train_cfg)

    print(f"\n訓練中（{ADOPTED_MODEL}，§7.12 正式採用的模型）...", flush=True)
    fitted = fit_adopted(train, es, all_train=feb)
    pred = fitted.predict(mar.X)
    print(f"  停在第 {fitted.best_iteration} 輪　Mar log loss {log_loss(mar.y, pred):.5f}")

    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )

    # ---- LTV：月費由資料算，續訂月數用上一期的流失率 ----
    #
    # 推導走 `src.evaluation.resolve_assumptions`，**不在這裡自己算** ——
    # M5 的原因碼名單要用同一個 p*，兩支腳本各推一份遲早分歧，而分歧的症狀是
    # 兩份交付物都印出「投放 4.8 萬人」卻指著不同的名單。
    feb_rate, mar_rate = float(feb.y.mean()), float(mar.y.mean())
    assumptions = resolve_assumptions(
        biz, price_per_day=mar.X["price_per_day"], prior_churn_rate=feb_rate
    )
    arpu, months, ltv = assumptions.monthly_arpu, assumptions.expected_months, assumptions.ltv_saved
    ltv_mar = arpu * expected_months(mar_rate)

    pl.Config.set_tbl_rows(60)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)

    print("\n" + "=" * 88)
    print("一、LTV 的推導（只有「預期續訂月數」是假設）")
    print("=" * 88)
    print(
        f"  月費（Mar cohort 實付平均）      {arpu:>10.1f} 元\n"
        f"  預期續訂月數 = 1 / Feb 流失率     {months:>10.1f} 個月（Feb {feb_rate:.4%}）\n"
        f"  LTV_saved                        {ltv:>10.0f} 元\n"
        f"\n  ⚠️ 若改用 Mar 的流失率（{mar_rate:.4%}）會得到 {ltv_mar:.0f} 元 —— "
        f"但那是評估集的標籤，\n     拿它設業務常數等於用到答案。差距落在 C_offer 的掃描區間內。"
    )

    star = assumptions.p_star
    print(
        f"\n  投放門檻 p* = {biz['c_offer']:.0f} / ({biz['r_save']:.2f} × {ltv:.0f})"
        f" = **{star:.4f}**"
    )

    # ---- 模型的整體低估 ----
    #
    # ⚠️ 這個數字**只描述全體 cohort 的平均預測流失率**，不能往下套。
    #    往下套會錯在四個地方，最容易漏掉的是最後一個：
    #      個別用戶      每個人的偏差不同
    #      風險區間      §7.10 的 reliability 曲線顯示各段差很大
    #      前 K% 名單    那是高機率子集，偏差與全體無關（下面單獨報）
    #      淨收益        p × r × LTV − C 只有第一項隨 p 縮放，C_offer 是
    #                    固定成本，所以「機率低估 27%」≠「淨收益低估 27%」
    large = calibration_in_the_large(scored["is_churn"], scored["p_churn"])
    print(
        f"\n  ⚠️ {ADOPTED_MODEL} 在 **Mar 全體 cohort** 的平均預測流失率"
        f" {large['平均預測']:.4%} vs 實際 {large['實際流失率']:.4%}"
        f"（{large['相對偏差']:+.2%}）\n"
        "     這是一個關於**全體平均**的數字，不能套到個別用戶、個別風險區間、\n"
        "     投放名單，也不能套到淨收益（C_offer 是固定成本，不隨機率縮放）。\n"
        "     投放名單自己的偏差另外報，見第二節。"
    )

    # ---- 曲線 ----
    curve = campaign_curve(
        scored["is_churn"],
        scored["p_churn"],
        r_save=biz["r_save"],
        ltv_saved=ltv,
        c_offer=biz["c_offer"],
        step=curve_cfg["step"],
    )
    best_exp = optimal_point(curve, by="期望模擬淨收益")
    best_act = optimal_point(curve, by="標籤結算模擬淨收益")

    print("\n" + "=" * 88)
    print("二、期望模擬淨收益曲線（基準情境）")
    print("=" * 88)
    print(
        f"  依期望曲線（部署時唯一能用的依據）：投放前 {best_exp['K']:.1%}"
        f"（{best_exp['投放人數']:,} 人，門檻機率 {best_exp['門檻機率']:.4f}）\n"
        f"    期望模擬淨收益 {best_exp['期望模擬淨收益']:>12,.0f} 元"
        f"　標籤結算 {best_exp['標籤結算模擬淨收益']:>12,.0f} 元\n"
        f"    命中率 {best_exp['命中率']:.2%}　lift {best_exp['lift']:.2f}\n"
        f"\n  事後最佳（只能回顧，部署時不知道）：投放前 {best_act['K']:.1%}"
        f"（{best_act['投放人數']:,} 人）\n"
        f"    標籤結算模擬淨收益 {best_act['標籤結算模擬淨收益']:>12,.0f} 元"
    )
    gap = best_act["標籤結算模擬淨收益"] - best_exp["標籤結算模擬淨收益"]
    print(
        f"\n  **門檻偏保守的代價：{gap:,.0f} 元** —— 照期望曲線選門檻，"
        "在標籤結算下比事後最佳少這麼多。\n"
        "  方向與 SPEC §6.1 的預告一致：低估流失風險 → 門檻設得過於保守。"
    )

    # ---- 投放名單自己的偏差，以及兩條曲線在該點的差距 ----
    #
    # 這一段存在的唯一理由：**全體的 −27% 不能套到這份名單。** 名單是機率
    # 最高的一小撮，它的偏差要自己量。
    slice_cal = subset_calibration(scored["is_churn"], scored["p_churn"], k=best_exp["K"])
    exp_rev = best_exp["期望模擬淨收益"]
    set_rev = best_exp["標籤結算模擬淨收益"]

    print("\n" + "-" * 88)
    print(f"  模型選定的前 {best_exp['K']:.1%} 名單（{slice_cal['人數']:,} 人）自己的偏差")
    print("-" * 88)
    print(
        f"    平均預測 {slice_cal['平均預測']:.4%}"
        f"　實際流失率 {slice_cal['實際流失率']:.4%}"
        f"　相對偏差 **{slice_cal['相對偏差']:+.2%}**\n"
        f"      ↑ 與全體的 {large['相對偏差']:+.2%} **不同** —— 這正是"
        "「不能把全體偏差套到子集」的實證。\n"
        f"\n    期望模擬淨收益      {exp_rev:>14,.0f} 元\n"
        f"    標籤結算模擬淨收益  {set_rev:>14,.0f} 元\n"
        f"    差距                {set_rev - exp_rev:>+14,.0f} 元"
        f"（{set_rev / exp_rev - 1:+.2%}）\n"
        f"\n    ⚠️ 淨收益的差距（{set_rev / exp_rev - 1:+.2%}）與機率的相對偏差"
        f"（{slice_cal['相對偏差']:+.2%}）不相等，\n"
        "       因為 C_offer 是固定成本、不隨機率縮放。**兩者不可互相換算。**"
    )

    print("\n" + "=" * 88)
    print("三、分群（SPEC §4.5 第 3 點：模型必須贏過「對所有新客發優惠」）")
    print("=" * 88)
    print(campaign_table(scored, biz, ltv, curve_cfg["step"]))

    rule = fixed_rule_point(
        scored["is_churn"],
        (scored["segment"] == "新進用戶").to_numpy(),
        r_save=biz["r_save"],
        ltv_saved=ltv,
        c_offer=biz["c_offer"],
        label="全部新進用戶",
    )
    print(
        f"\n  不需模型的規則「{rule['規則']}」："
        f"投放 {rule['投放人數']:,} 人（{rule['K']:.1%}）"
        f"　標籤結算模擬淨收益 {rule['標籤結算模擬淨收益']:,.0f} 元　lift {rule['lift']:.2f}"
    )
    verdict = (
        "✅ 模型勝出"
        if best_exp["標籤結算模擬淨收益"] > rule["標籤結算模擬淨收益"]
        else "❌ 模型沒有贏過規則"
    )
    print(
        f"  模型排序在同樣的預算下標籤結算模擬淨收益 "
        f"{best_exp['標籤結算模擬淨收益']:,.0f} 元　→ {verdict}"
    )

    # ---- 敏感度 ----
    grid = sensitivity_grid(
        scored["is_churn"],
        scored["p_churn"],
        ltv_saved=ltv,
        r_values=sens["r_save"],
        c_values=sens["c_offer"],
    )
    print("\n" + "=" * 88)
    print("四、敏感度：最佳投放比例如何隨假設變動")
    print("=" * 88)
    print(
        grid.select(
            "r_save", "c_offer", pl.col("p*").round(4), pl.col("最佳投放比例").round(4), "投放人數"
        ).head(48)
    )

    positive = grid.filter(pl.col("標籤結算模擬淨收益") > 0)
    print(
        f"\n  48 組假設裡有 {positive.height} 組的標籤結算模擬淨收益為正"
        f"（{positive.height / grid.height:.0%}）。"
    )

    print("\n產生圖表...")
    plot_curve(scored, curve, biz, ltv, paths.figures)
    plot_sensitivity(grid, biz, paths.figures)

    # ---- 機器可讀的摘要 ----
    #
    # `scripts/rebaseline.py` 會把這份 JSON 併進 manifest，讓「這條曲線是用
    # 哪個模型、哪組假設、跑出什麼門檻」不必靠人去翻 log。完整 log 仍然保存，
    # 這份只是把最關鍵的欄位提出來。
    summary = {
        "model": ADOPTED_MODEL,
        "configs": ["configs/business.yaml", "configs/model_comparison.yaml"],
        "mar_log_loss": round(float(log_loss(mar.y, pred)), 5),
        "assumptions": assumptions.summary(),
        "optimum_by_expected": {
            "k": round(best_exp["K"], 4),
            "n_targeted": int(best_exp["投放人數"]),
            "threshold_probability": round(float(best_exp["門檻機率"]), 4),
            "expected_simulated_net": round(float(exp_rev)),
            "label_settled_simulated_net": round(float(set_rev)),
            "precision": round(float(best_exp["命中率"]), 4),
            "lift": round(float(best_exp["lift"]), 2),
        },
        "targeted_slice_calibration": {k: round(float(v), 6) for k, v in slice_cal.items()},
        "cohort_calibration": {k: round(float(v), 6) for k, v in large.items()},
        "figures": [
            "reports/figures/12_expected_net_revenue.png",
            "reports/figures/13_sensitivity_heatmap.png",
        ],
    }
    out_path = REPO_ROOT / "reports" / "business_value.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    摘要已存 → reports/{out_path.name}")

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    print(
        "  決策只取決於 p* = C_offer / (r_save × LTV)，三個參數的不確定性互相吸收。\n"
        "  基準情境門檻高，最佳投放比例小 —— **「多數情況不該大規模投放」本身就是結論**，\n"
        "  不是參數沒調好。\n"
        "\n  ⚠️ 所有金額用的是**未校準**機率（SPEC §7.10 決定不上線校準器），\n"
        "     且系統性偏低 —— 讀者知道它偏保守，曲線仍然可用。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
