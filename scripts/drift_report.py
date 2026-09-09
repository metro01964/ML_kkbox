"""M6 · PSI 漂移監控報告（SPEC §7 M6）。

    uv run python scripts/drift_report.py          # 用能上線的 T−7 artifact
    make drift

## 這支腳本問的是「上線之後，我們看得見什麼」

部署時**拿不到標籤** —— `is_churn` 的定義是「到期後 30 天內沒有續訂」，所以這個
月的答案要等下個月。監控只能看輸入與輸出的分布，那就是 PSI 的位置。

它**不重新訓練任何東西**：模型從 artifact 載入（`make artifact-t7` 產生的那一
份），這正是 artifact 存在的理由之一。

## 三個數字並排，因為第三個在部署時看不到

    特徵 PSI    61 欄各自的分布變了多少（參考期 = 訓練 cohort）
    分數 PSI    模型輸出的分布變了多少
    標籤漂移    實際流失率變了多少 ← **部署時拿不到，這裡拿得到**

⚠️ **這是本節唯一真正重要的比較。** 本專案已知的漂移是基準率：Feb 6.36% →
Mar 8.89%，而它讓 §7.10 的校準器失效、讓平均預測低估近三成。如果特徵 PSI 全部
落在「穩定」帶，那就等於說：**我們即將上線的監控，在唯一真正傷到我們的那件事上
會回報一切正常。** 那個結論要量出來寫進 MODEL_CARD，不是等它發生。

## 參考期的分數不可以用訓練集的樣本內預測

樣本內的分數分布比實際更尖銳（模型見過那些標籤）。拿它當參考期，「訓練 vs 當期」
的差異裡就混進了「樣本內 vs 樣本外」—— 而那不是漂移。

所以參考期用 **Feb 內部那塊 early stopping 切分**：它只決定停在第幾輪，沒有被
擬合。它不是完全的樣本外（停點是看著它挑的），這一點在報告裡寫明；本專案沒有
更好的選擇 —— 唯一真正乾淨的樣本外就是 Mar，而 Mar 是當期。

腳本同時印出「用樣本內當參考期」的版本作為對照，讓那個差距是可見的量而不是
一句提醒。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.data import COHORTS, build_cohort
from src.evaluation import (
    DEFAULT_BINS,
    EPSILON,
    band,
    calibration_in_the_large,
    noise_floor,
    psi_table,
    score_psi,
)
from src.features import CATEGORICAL, build_features, build_log_features, expected_log_columns
from src.models.adopted import load_adopted_config
from src.models.compare import split_for_early_stopping
from src.serving.artifact import load_artifact

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

OUT_PATH = REPO_ROOT / "reports" / "drift.json"

# 雜訊地板要切幾輪。5 輪 × 61 欄 = 305 個 PSI，足以定一個 95 百分位；
# 再多的成本是線性的而那條線不會動。
NOISE_ROUNDS = 5

# 表上印幾欄。61 欄全印沒有人會讀，而排序遞減讓前段就是全部的故事。
TOP_N = 15


def git_state() -> tuple[str, bool]:
    """SHA 與工作區狀態。**在寫出任何東西之前呼叫**（圖 16 進 git，M5 踩過）。"""

    def run(*args: str) -> str:
        return subprocess.run(
            args, capture_output=True, text=True, cwd=REPO_ROOT, check=False
        ).stdout.strip()

    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain"))


def serving_artifact_name() -> str:
    path = REPO_ROOT / "configs" / "serving.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(cfg.get("artifact", ""))


def load_business_config() -> dict:
    path = REPO_ROOT / "configs" / "business.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _cohort(name: str, paths):
    """建（或讀快取）某個 cohort 的 as-of 表。模型不重訓，但特徵要重建 ——
    監控看的就是特徵分布。"""
    return build_cohort(COHORTS[name], paths, verbose=False)


def print_table(summary: pl.DataFrame, floor_p95: float, n: int) -> None:
    pl.Config.set_tbl_rows(n + 2)
    pl.Config.set_tbl_width_chars(200)
    shown = summary.head(n).with_columns(
        pl.col("psi").round(4),
        (pl.col("psi") / floor_p95).round(1).alias("倍雜訊地板"),
        pl.col("reference_missing").round(4),
        pl.col("current_missing").round(4),
        pl.col("missing_delta").round(4),
    )
    print(
        shown.select(
            "feature",
            "psi",
            "band",
            "倍雜訊地板",
            "epsilon_floored",
            "kind",
            "missing_delta",
            "top_bin",
        )
    )


def plot_drift(
    summary: pl.DataFrame,
    floor_p95: float,
    scores: dict,
    labels: dict,
    figdir,
) -> str:
    """圖 16：特徵 PSI 的排行（含雜訊地板），以及分數分布 vs 標籤漂移。"""
    fig, (ax_f, ax_s) = plt.subplots(1, 2, figsize=(15, 5.4))

    top = summary.head(12)
    names = top["feature"].to_list()[::-1]
    values = top["psi"].to_numpy()[::-1]
    colors = ["#d62728" if f else "#1f77b4" for f in top["epsilon_floored"].to_list()[::-1]]
    ax_f.barh(names, values, color=colors)
    ax_f.set_xscale("log")
    # ⚠️ 對數刻度的預設刻度標籤走 mathtext，負指數會用 U+2212 MINUS SIGN ——
    # Microsoft JhengHei 缺這個字，會渲染成方框（圖 09 的負號、圖 12 的「≤」、
    # 圖 14 的「T−7」是同一類問題）。所以自己給 ASCII 的刻度標籤。
    ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    ax_f.set_xticks(ticks)
    ax_f.set_xticklabels(["0.00001", "0.0001", "0.001", "0.01", "0.1", "1"], fontsize=8)
    ax_f.set_xlim(min(values.min(), 1e-5) * 0.7, max(values.max(), 1.0) * 2.5)
    ax_f.axvline(floor_p95, color="#2ca02c", ls="--", lw=1.4, label=f"雜訊地板 p95 {floor_p95:.4f}")
    ax_f.axvline(0.10, color="#7f7f7f", ls=":", lw=1.2, label="慣例 0.10（中度）")
    ax_f.axvline(0.25, color="#333333", ls=":", lw=1.2, label="慣例 0.25（顯著）")
    ax_f.set_xlabel("PSI（對數刻度）")
    # 圖例只在真的有紅色的時候講紅色 —— 一句對應不到任何圖形的說明會讓讀者
    # 去找一個不存在的東西（實測 `last_payment_method_id` 排第 13，不在前 12）。
    note = "　紅色 = 有單邊空箱，數字由 epsilon 決定" if any(c == "#d62728" for c in colors) else ""
    ax_f.set_title(f"特徵 PSI 前 {top.height} 名{note}", fontsize=11)
    ax_f.tick_params(axis="y", labelsize=8)
    ax_f.legend(fontsize=8, loc="lower right")
    ax_f.grid(axis="x", alpha=0.25)

    # 分數分布：用參考期的十分位當箱，兩期並排。
    ref, cur = scores["reference_scores"], scores["current_scores"]
    edges = np.quantile(ref, np.linspace(0, 1, DEFAULT_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, bins=edges)[0] / ref.size
    cur_share = np.histogram(cur, bins=edges)[0] / cur.size
    x = np.arange(DEFAULT_BINS)
    ax_s.bar(x - 0.2, ref_share * 100, width=0.4, label="參考期（Feb ES 切分）", color="#1f77b4")
    ax_s.bar(x + 0.2, cur_share * 100, width=0.4, label="當期（Mar）", color="#ff7f0e")
    ax_s.set_xticks(x)
    ax_s.set_xticklabels([f"D{i + 1}" for i in x], fontsize=8)
    ax_s.set_xlabel("參考期分數的十分位")
    ax_s.set_ylabel("佔比（%）")
    ax_s.set_title(
        f"分數分布　PSI {scores['psi']:.4f}（{band(scores['psi'])}）　"
        f"而實際流失率 {labels['reference_churn']:.2%} → {labels['current_churn']:.2%}"
        f"（相對 {labels['relative_change']:+.1%}）",
        fontsize=11,
    )
    ax_s.legend(fontsize=8)
    ax_s.grid(axis="y", alpha=0.25)

    fig.suptitle(
        # 圖上不用 ⚠️（Microsoft JhengHei 缺這個字，會變方框）。
        "PSI 漂移監控　注意：慣例門檻沒有樣本數修正，要與雜訊地板一起讀；標籤漂移在部署時拿不到",
        fontsize=12,
    )
    path = figdir / "16_drift_psi.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")
    return f"reports/figures/{path.name}"


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 PSI 漂移監控報告")
    ap.add_argument(
        "--artifact",
        default=None,
        help="artifact 名稱（預設讀 configs/serving.yaml，也就是實際上線的那一份）",
    )
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS, help=f"箱數（預設 {DEFAULT_BINS}）")
    ap.add_argument(
        "--noise-rounds",
        type=int,
        default=NOISE_ROUNDS,
        help=f"雜訊地板切幾輪（預設 {NOISE_ROUNDS}）",
    )
    ap.add_argument("--top", type=int, default=TOP_N, help=f"表上印幾欄（預設 {TOP_N}）")
    args = ap.parse_args()

    sha, dirty = git_state()
    try:
        paths = load_paths().ensure()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這份報告無法用 SHA 回溯。")

    name = args.artifact or serving_artifact_name()
    art = load_artifact(name=name)
    train_spec, eval_spec = COHORTS[art.meta["cohort"]["train"]], COHORTS[art.eval_cohort]

    print(f"\n{'=' * 92}")
    print(
        f"PSI 漂移監控　artifact {art.directory.name}"
        f"（{art.meta['model']['name']}，cutoff {art.cutoff_definition}，"
        f"可上線 {art.deployable}）"
    )
    print(f"參考期 {train_spec.name}（訓練）→ 當期 {eval_spec.name}")
    print("=" * 92)
    for w in art.warnings:
        print(f"⚠️ {w}")

    # ---- 特徵 ----
    # 模型不重訓，但特徵要重建（監控看的就是特徵分布）。
    with_logs = art.uses_log_features
    reference = build_features(
        _cohort(train_spec.name, paths),
        build_log_features(train_spec, paths, verbose=False) if with_logs else None,
    )
    current = build_features(
        _cohort(eval_spec.name, paths),
        build_log_features(eval_spec, paths, verbose=False) if with_logs else None,
    )
    X_ref = reference.X.select(art.feature_names)
    X_cur = current.X.select(art.feature_names)
    print(f"\n參考期 {X_ref.height:,} 列 · 當期 {X_cur.height:,} 列 × {X_ref.width} 特徵")

    print("\n量雜訊地板（同一個 cohort 切兩半，完全沒有漂移時 PSI 長什麼樣）...", flush=True)
    floor = noise_floor(
        X_ref, categorical=CATEGORICAL, bins=args.bins, rounds=args.noise_rounds, epsilon=EPSILON
    )
    floor_p95 = float(floor["p95_all"][0])
    print(
        f"  {args.noise_rounds} 輪 × {X_ref.width} 欄：p95 = {floor_p95:.5f}"
        f"　最大 {float(floor['psi_max'].max()):.5f}（{floor['feature'][0]}）"
    )
    print(
        "  ⚠️ 慣例門檻 0.10 / 0.25 沒有樣本數修正。這個樣本數"
        f"（{X_ref.height // 2:,} vs {X_ref.height // 2:,}）下的無漂移基準是上面那個數字，\n"
        "     判讀時看「幾倍雜訊地板」比看慣例帶更有意義。"
    )

    print("\n算特徵 PSI ...", flush=True)
    summary, detail = psi_table(
        X_ref, X_cur, categorical=CATEGORICAL, bins=args.bins, epsilon=EPSILON
    )
    print(f"\n一、特徵 PSI 前 {args.top} 名")
    print_table(summary, floor_p95, args.top)

    bands = {name: int((summary["band"] == name).sum()) for name in ("穩定", "中度", "顯著")}
    floored = summary.filter(pl.col("epsilon_floored"))
    print(
        f"\n  慣例分級：穩定 {bands['穩定']} 欄 · 中度 {bands['中度']} 欄 · 顯著 {bands['顯著']} 欄"
        f"　（超過雜訊地板 p95 的有 {int((summary['psi'] > floor_p95).sum())} 欄）"
    )
    if floored.height:
        print(
            f"  ⚠️ {floored.height} 欄有單邊空箱（epsilon = {EPSILON:g} 決定了它們的數字大小，"
            f"不可當漂移的量讀）：{floored['feature'].to_list()[:6]}"
        )

    # ---- 分數 ----
    #
    # 參考期用 Feb 內部那塊 early stopping 切分（沒有被擬合），理由見模組開頭。
    _, train_cfg = load_adopted_config()
    _, es = split_for_early_stopping(reference, train_cfg)
    es_scores = np.asarray(art.fitted.predict(es.X.select(art.feature_names)), dtype=np.float64)
    in_sample = np.asarray(art.fitted.predict(X_ref), dtype=np.float64)
    cur_scores = np.asarray(art.fitted.predict(X_cur), dtype=np.float64)

    score_value, score_floored, score_detail = score_psi(
        es_scores, cur_scores, bins=args.bins, epsilon=EPSILON
    )
    in_sample_value, _, _ = score_psi(in_sample, cur_scores, bins=args.bins, epsilon=EPSILON)

    print(f"\n{'=' * 92}\n二、分數 PSI（模型輸出的分布）\n{'=' * 92}")
    print(
        f"  參考期 = Feb 內部 early stopping 切分（{es.X.height:,} 列，未被擬合）\n"
        f"  分數 PSI **{score_value:.4f}**（{band(score_value)}）"
        f"　= {score_value / floor_p95:.1f} 倍雜訊地板"
    )
    print(
        f"\n  對照：若拿樣本內預測當參考期，PSI = {in_sample_value:.4f}"
        f"（差 {in_sample_value - score_value:+.4f}）—— 那個差是"
        "「樣本內 vs 樣本外」而不是漂移，所以參考期不能用訓練集的預測。"
    )

    # ---- 標籤（部署時拿不到的那個）----
    ref_churn, cur_churn = float(reference.y.mean()), float(current.y.mean())
    cal = calibration_in_the_large(current.y, cur_scores)
    labels = {
        "reference_churn": ref_churn,
        "current_churn": cur_churn,
        "absolute_change": cur_churn - ref_churn,
        "relative_change": cur_churn / ref_churn - 1,
        "current_mean_prediction": float(cal["平均預測"]),
        "current_relative_bias": float(cal["相對偏差"]),
    }
    print(f"\n{'=' * 92}\n三、標籤漂移 —— **部署時拿不到，這裡拿得到**\n{'=' * 92}")
    print(
        f"  實際流失率 {ref_churn:.4%} → {cur_churn:.4%}"
        f"（相對 {labels['relative_change']:+.1%}）\n"
        f"  當期平均預測 {labels['current_mean_prediction']:.4%} vs 實際 {cur_churn:.4%}"
        f"（相對偏差 {labels['current_relative_bias']:+.2%}）"
    )
    print(
        "\n  ⚠️ 這是整個 M6 監控最重要的一行對照：\n"
        f"     特徵有 {bands['穩定']}/{summary.height} 欄落在慣例的「穩定」帶、"
        f"分數 PSI 是 {score_value:.4f}（{band(score_value)}），\n"
        f"     而實際流失率動了 {labels['relative_change']:+.1%} —— 那正是讓 §7.10 的校準器"
        "失效的那個漂移。\n"
        "     PSI 看得到「進來的人變了」，看不到「同一種人行為變了」。基準率要靠"
        "標籤到齊之後的回測，\n     或靠一個外生指標（例如當月的整體續訂率）來接。"
    )

    # ---- 投放後果 ----
    p_star = art.p_star
    ref_rate = float((es_scores > p_star).mean())
    cur_rate = float((cur_scores > p_star).mean())
    print(f"\n{'=' * 92}\n四、對投放的後果（p* = {p_star:.4f}）\n{'=' * 92}")
    print(
        f"  參考期有 {ref_rate:.2%} 的人超過門檻，當期 {cur_rate:.2%}"
        f"（相對 {cur_rate / ref_rate - 1:+.1%} —— 名單會變這麼多）"
    )

    print("\n產生圖表...")
    figure = plot_drift(
        summary,
        floor_p95,
        {"psi": score_value, "reference_scores": es_scores, "current_scores": cur_scores},
        labels,
        paths.figures,
    )

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "artifact": {
            "name": art.directory.name,
            "model_sha256_16": art.meta["model"]["sha256"][:16],
            "cutoff_definition": art.cutoff_definition,
            "lead_days": art.lead_days,
            "deployable": art.deployable,
        },
        "cohorts": {
            "reference": train_spec.name,
            "current": eval_spec.name,
            "n_reference": int(X_ref.height),
            "n_current": int(X_cur.height),
        },
        # ⚠️ 這三個參數都會改變下面每一個數字，所以它們跟結果放在一起。
        "settings": {
            "bins": args.bins,
            "epsilon": EPSILON,
            "noise_rounds": args.noise_rounds,
            "conventional_bands": [0.10, 0.25],
            "log_feature_columns": len(expected_log_columns() - {"msno", "cutoff"}),
        },
        "noise_floor": {
            "p95_all": round(floor_p95, 6),
            "max": round(float(floor["psi_max"].max()), 6),
            "worst_feature": floor["feature"][0],
            "how": "同一個 cohort 隨機切兩半算 PSI，重複 N 輪 —— 完全沒有漂移時的基準",
        },
        "features": {
            "bands": bands,
            "above_noise_floor": int((summary["psi"] > floor_p95).sum()),
            "epsilon_floored": floored["feature"].to_list(),
            "top": [
                {
                    k: (round(v, 6) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k != "n_bins"
                }
                for row in summary.head(args.top).iter_rows(named=True)
            ],
        },
        "score": {
            "psi": round(score_value, 6),
            "band": band(score_value),
            "epsilon_floored": score_floored,
            "reference": "feb inner early-stopping split（未被擬合，但停點看過它）",
            "in_sample_reference_psi": round(in_sample_value, 6),
            "deciles": [
                {k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()}
                for row in score_detail.iter_rows(named=True)
            ],
        },
        "labels": {k: round(v, 6) for k, v in labels.items()},
        "targeting": {
            "p_star": p_star,
            "reference_rate": round(ref_rate, 6),
            "current_rate": round(cur_rate, 6),
            "relative_change": round(cur_rate / ref_rate - 1, 6),
        },
        "figures": [figure],
        "how_to_read": [
            "0.10 / 0.25 是慣例，不是統計檢定 —— 沒有分布假設也沒有樣本數修正。"
            f"這個樣本數下的無漂移基準是 noise_floor.p95_all = {floor_p95:.5f}，"
            "所以「幾倍雜訊地板」比「落在哪一帶」更能判讀。",
            "epsilon_floored = true 的欄位有單邊空箱（例如出現了訓練時沒見過的類別、"
            f"或缺失率從 0 變成正的）。那時 PSI 的大小由 epsilon = {EPSILON:g} 決定，"
            "只能讀成「出現了參考期沒有的東西」，不能讀成漂移的量。",
            "分數 PSI 的參考期是 Feb 內部的 early stopping 切分，不是訓練集的樣本內預測 ——"
            f"用樣本內會得到 {in_sample_value:.4f}，那個差是「樣本內 vs 樣本外」不是漂移。",
            "⚠️ PSI 看不到基準率漂移。實際流失率動了"
            f"{labels['relative_change']:+.1%}，而特徵與分數的 PSI 沒有反映那個量級 ——"
            "這條限制要進 MODEL_CARD：監控回報「穩定」不等於模型還準。",
            "標籤漂移這一節在部署時拿不到（標籤要等到期後 30 天）。它在這裡是因為"
            "本專案有 Mar 的標籤 —— 那是一次可以量出「監控會漏掉什麼」的機會。",
        ],
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    報告已存 → reports/{OUT_PATH.name}")

    print("\n" + "=" * 92)
    print("讀法")
    print("=" * 92)
    for line in report["how_to_read"]:
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
