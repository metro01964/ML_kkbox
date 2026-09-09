"""M5 · 投放名單與流失原因碼（SPEC §7 M5）。

M4 交付的是「該投放給多少人」，這支交付的是**那些人是誰、為什麼是他們**。

    uv run python scripts/explain.py
    make explain

## 這份名單是一個函式的輸出，不是一份清單

輸出的 CSV **不是資產**，是一次執行的結果。它由三組輸入決定：

    模型        §7.12 採用的 CatBoost（誰風險高、為什麼）
    業務假設    configs/business.yaml 的 r_save / C_offer / LTV（切多深）
    快取        interim/*.parquet 的特徵（哪一版程式算出來的）

改業務假設，名單成員會變而**每個人的機率與原因碼一個字都不會變**；改模型或
特徵，三者全變。這個區別寫進 manifest 的讀法裡，否則下次名單變了，沒有人分得
出是假設動了還是模型動了。

因此：

  - CSV 每次重跑，不進 git（`*.gitignore` 的 `*.csv` 已經讓它進不去）
  - 進 git 的只有 `manifest.json`，而它**只放 provenance 與彙總** ——
    逐人的列是資料，資料不上 GitHub（競賽規則）
  - 下游不得快取這份 CSV。M6 的 `/predict` 要即時算，不是查表

## 名單的定義是 `p > p*`，不是「前 5%」

M4 報的最佳投放比例 5.0% 是期望曲線在 `curve.step = 0.002` 的網格上取極大值
—— 名單大小會受一個**畫圖解析度參數**影響。而 `src.evaluation.decision` 的
恆等式保證極大值必然落在「最後一個 `p > p*` 的人」身上，所以這裡直接用門檻：

    p* = C_offer / (r_save × LTV_saved)

名單因此只由三個業務假設決定，C_offer 從 150 改成 50 是一次除法，不必重跑
曲線。p* 的推導與 M4 共用 `resolve_assumptions()`，兩支腳本不各推一份。

## ⚠️ 原因碼的量測時點會限制它能用在哪

`last_is_cancel` 標成**到期日訊號**：取消常發生在到期日當天，而現行 cutoff
就是到期日。M6 要交付 `cutoff = 到期日 − 7 天` 的版本，那時這筆交易還沒發生
—— 這句原因碼不可沿用。

**正解是重訓 T−7 版本的模型，不是推論時把那一欄遮掉。** SHAP 的歸因是聯合的，
遮一欄會讓 `sigmoid(base + Σ shap) == predict()` 破掉，破的量剛好是這個訊號
的全部強度（流失率 75.07% vs 4.33%）。所以本腳本只標註與計量，不過濾。

manifest 因此有一個必填欄位 `cutoff_definition`。少了它，這份名單的原因碼被
拿到 T−7 版本使用時，沒有任何東西會擋。

## 兩份輸出：營運呈現與底層稽核

    targeting_list.csv   每人一列，**只有該呈現的句子**（弱到不像理由的不印）
    reasons_audit.csv    每人每句一列，**全部候選都在** + 為什麼沒印

分開的理由是它們回答不同的問題。營運要的是「打電話時講什麼」，三句並列會讓人
以為三件事都重要 —— 而實測第 3 句的貢獻中位數只有第 1 名的 7.5%。稽核要的是
「當時為什麼沒印」，那就不能刪任何一列。

呈現門檻是 `relative_to_top >= 0.05`（`--min-relative` 可覆寫），它是一個
**呈現判斷不是統計檢定**，所以門檻值與它壓掉多少都寫進 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.data import FEB, MAR, cutoff_definition
from src.data.cohort import cohort_fingerprint
from src.evaluation import log_loss, resolve_assumptions, subset_calibration
from src.explain import (
    MIN_RELATIVE_SHARE,
    add_reasons,
    assert_local_accuracy,
    attribute,
    audit_frame,
    display_impact,
    expiry_dated_share,
    feature_group,
    mark_display,
    mean_abs_attribution,
    top_contributors,
    wide_reasons,
)
from src.features.logs import log_features_fingerprint
from src.fingerprint import read_cache_fingerprint
from src.models.adopted import ADOPTED_MODEL, fit_adopted, load_adopted_config
from src.models.compare import split_for_early_stopping
from src.models.train import load_cohort_features

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

# 現行 cohort 的 cutoff 定義。M6 的 T−7 版本會是 "expire_date_minus_7d"，
# 屆時 `last_is_cancel` 的原因碼不可沿用（見模組開頭）。
#
# 由 spec 推導而不是寫死字串：M6 的模型 artifact 也必填這一欄，兩邊各寫一份
# 的話，字串一旦不一致（`expire_date-7d`）比對這一欄的下游就會靜靜地認為兩份
# 交付物不同源。
CUTOFF_DEFINITION = cutoff_definition(MAR)

TOP_K = 3
OUT_DIR = REPO_ROOT / "reports" / "explanations"

# 底層稽核表的檔名。營運名單是 targeting_list.csv，兩者刻意不同名 ——
# 「這個人只有一個理由」與「他的第三個理由太弱所以沒印」是不同的事，
# 只有稽核表分得出來。
AUDIT_NAME = "reasons_audit.csv"


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    return out.stdout.strip()


def git_dirty() -> bool:
    """工作區有沒有未提交的改動 —— 髒的工作區跑出來的名單無法用 SHA 回溯。"""
    out = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    return bool(out.stdout.strip())


def load_configs() -> dict:
    biz_path = REPO_ROOT / "configs" / "business.yaml"
    if not biz_path.exists():
        raise FileNotFoundError(f"找不到 {biz_path}")
    biz = yaml.safe_load(biz_path.read_text(encoding="utf-8")) or {}
    if "business" not in biz:
        raise KeyError(f"{biz_path} 缺少 [business] 區段")
    return biz


def cache_provenance(paths) -> dict:
    """名單的特徵是**哪一版程式**算出來的。

    ## 為什麼 git SHA 不夠

    特徵來自 `interim/*.parquet`。那些檔案可能是舊版程式產生的 —— §7.11 的
    事故正是這個形狀：程式改好了，快取還是舊的，而當時是靠人工 `force=True`
    才發現。SHA 只能證明「執行時的程式是這一版」，證明不了「餵進去的特徵是
    這一版算的」。

    所以兩個都記：快取檔上的指紋（實際產生它的那版程式）與現行程式的指紋。
    兩者不符就代表這份名單的特徵是舊邏輯的產物。
    """
    current = {"cohort": cohort_fingerprint(), "log_features": log_features_fingerprint()}
    caches = {}
    for spec in (FEB, MAR):
        caches[spec.name] = {
            "cohort": read_cache_fingerprint(paths.interim / f"{spec.name}_cohort_asof.parquet"),
            "log_features": read_cache_fingerprint(
                paths.interim / f"{spec.name}_log_features.parquet"
            ),
        }
    stale = [
        f"{cohort}.{kind}"
        for cohort, got in caches.items()
        for kind, value in got.items()
        if value != current[kind]
    ]
    return {"code": current, "caches": caches, "stale": stale}


def rank_distribution(reasons: pl.DataFrame, people: int) -> pl.DataFrame:
    """每人拿到幾句原因碼。不足 3 句是正常的，要看得出有多少人如此。"""
    per_person = reasons.group_by("row").agg(pl.len().alias("句數"))
    counts = (
        per_person.group_by("句數")
        .agg(pl.len().alias("人數"))
        .sort("句數", descending=True)
        .with_columns((pl.col("人數") / people).alias("佔比"))
    )
    zero = people - per_person.height
    if zero:
        counts = pl.concat(
            [
                counts,
                pl.DataFrame({"句數": [0], "人數": [zero], "佔比": [zero / people]}).cast(
                    counts.schema
                ),
            ]
        )
    return counts


def top1_groups(reasons: pl.DataFrame) -> pl.DataFrame:
    """Top-1 原因的組別分布 —— 名單整體「主要在抓什麼」。"""
    first = reasons.filter(pl.col("rank") == 1)
    return (
        first.group_by("group")
        .agg(pl.len().alias("人數"), pl.col("group_shap").mean().round(4).alias("平均貢獻"))
        .sort("人數", descending=True)
        .with_columns((pl.col("人數") / first.height).alias("佔比"))
    )


def plot_reasons(top1: pl.DataFrame, share: dict, counts: pl.DataFrame, figdir) -> str:
    """圖 14：名單在抓什麼，以及有多少解釋撐不到 T−7。"""
    fig, (ax_g, ax_h) = plt.subplots(1, 2, figsize=(14, 5))

    groups = top1["group"].to_list()[::-1]
    shares = (top1["佔比"].to_numpy() * 100)[::-1]
    ax_g.barh(groups, shares, color="#1f77b4")
    for i, v in enumerate(shares):
        ax_g.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9)
    ax_g.set_xlabel("佔名單的比例（%）")
    ax_g.set_title("Top-1 原因的組別分布 —— 這份名單主要在抓什麼", fontsize=11)
    ax_g.set_xlim(0, max(shares) * 1.18)
    ax_g.grid(axis="x", alpha=0.25)

    labels = ["原因碼句數", "受影響人數"]
    values = [share["句數比例"] * 100, share["受影響人數比例"] * 100]
    ax_h.bar(labels, values, color=["#d62728", "#ff7f0e"], width=0.55)
    for i, v in enumerate(values):
        ax_h.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax_h.set_ylabel("到期日訊號的佔比（%）")
    ax_h.set_ylim(0, max(values) * 1.25 + 4)
    # ⚠️ 標題不用 U+2212 MINUS SIGN（「T−7」）—— Microsoft JhengHei 缺這個字，
    # 會渲染成方框（與圖 09 的 mathtext 負號、圖 12 的「≤」同一類問題）。
    # 圖上一律用 ASCII 連字號，文件裡才用排版正確的減號。
    ax_h.set_title(
        "到期日訊號（last_is_cancel）—— M6 的 T-7 版本會失去的部分",
        fontsize=11,
    )
    ax_h.grid(axis="y", alpha=0.25)

    median_lines = "　".join(
        f"{row['句數']} 句 {row['佔比']:.1%}" for row in counts.iter_rows(named=True)
    )
    fig.suptitle(
        f"投放名單的原因碼結構　每人最多 {TOP_K} 句（{median_lines}）",
        fontsize=12,
    )
    path = figdir / "14_reason_codes.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")
    return f"reports/figures/{path.name}"


def print_samples(wide: pl.DataFrame, n: int) -> None:
    """幾位用戶的完整 Top-3，用讀得出來的格式。這就是 M5 的驗收標準。

    ## 為什麼取整份名單的等距位置，而不是機率最高的前 n 位

    第一次跑的時候印的是前 6 位，而**那 6 個人不代表名單**：機率 0.9999 的最
    頂端是同日交易衝突（§7.11）導致 `last_is_cancel` 與 `last_is_auto_renew`
    雙雙為 null 的極少數人（實測全名單只有 95 人的 Top-1 是「自動續訂設定
    缺失」）。於是樣本讀起來像「這份名單全是資料不明的人」，而真相是 92% 的人
    的 Top-1 是「未開啟自動續訂」或「到期前最後一筆交易是取消」這兩句。

    排序最頂端本來就是最極端的一小撮 —— 拿它當樣本會系統性地誤導。
    """
    if wide.height > n:
        idx = np.linspace(0, wide.height - 1, num=n).round().astype(np.int64)
        wide = wide[idx].with_columns(pl.Series("名單位置", idx + 1))
    for row in wide.head(n).iter_rows(named=True):
        flags = []
        for i in range(1, TOP_K + 1):
            if row[f"expiry_dated_{i}"]:
                flags.append(str(i))
        note = f"　⚠️ 第 {'、'.join(flags)} 句是到期日訊號，T−7 版本不可沿用" if flags else ""
        rank = f"第 {row['名單位置']:,} 位　" if "名單位置" in row else ""
        print(f"\n  {rank}{row['msno'][:16]}…　流失機率 {row['p_churn']:.4f}{note}")
        for i in range(1, TOP_K + 1):
            reason = row[f"reason_{i}"]
            if reason is None:
                continue
            print(f"    {i}. {reason}　（{row[f'group_{i}']}，貢獻 {row[f'group_shap_{i}']:+.3f}）")


def main() -> int:
    ap = argparse.ArgumentParser(description="M5 投放名單與流失原因碼")
    ap.add_argument("--sample", type=int, default=5, help="印幾位用戶的完整 Top-3（預設 5）")
    ap.add_argument("--msno", nargs="*", default=None, help="只解釋這幾位用戶（不受名單門檻限制）")
    ap.add_argument(
        "--min-relative",
        type=float,
        default=MIN_RELATIVE_SHARE,
        help=f"呈現門檻：一句原因碼至少要有第 1 名的幾成（預設 {MIN_RELATIVE_SHARE}，0 = 全印）",
    )
    args = ap.parse_args()

    # ⚠️ **git 狀態要在寫出任何東西之前抓。**
    #
    # 第一版把這兩行放在輸出段落，於是 `git_dirty` 永遠是 True —— 因為圖 14
    # 進 git，而 `plot_reasons()` 已經先把它改掉了。**這個旗標會自我實現：**
    # 每一次重跑都報「工作區是髒的」，於是它從一個警告退化成一行雜訊，讀者
    # 學會忽略它，而真正該被擋下的那次（帶著未提交的改動跑）就混了進來。
    #
    # 抓的是「這次執行開始時，程式碼是什麼狀態」—— 那才是「這份名單能不能用
    # 這個 SHA 回溯」的答案。本次執行自己產生的檔案不算污染。
    sha, dirty = git_sha(), git_dirty()

    try:
        paths = load_paths().ensure()
        biz_cfg = load_configs()
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這份名單無法用 SHA 回溯。")

    biz = biz_cfg["business"]
    _, train_cfg = load_adopted_config()

    feb, mar = load_cohort_features(paths, biz_cfg)
    train, es = split_for_early_stopping(feb, train_cfg)

    print(f"\n訓練中（{ADOPTED_MODEL}，§7.12 正式採用的模型）...", flush=True)
    fitted = fit_adopted(train, es, all_train=feb)
    pred = fitted.predict(mar.X)
    mar_loss = float(log_loss(mar.y, pred))
    print(f"  停在第 {fitted.best_iteration} 輪　Mar log loss {mar_loss:.5f}")

    assumptions = resolve_assumptions(
        biz, price_per_day=mar.X["price_per_day"], prior_churn_rate=float(feb.y.mean())
    )
    star = assumptions.p_star

    print("\n" + "=" * 88)
    print("一、名單 = 所有 p > p* 的人（門檻由業務假設決定，不由曲線解析度決定）")
    print("=" * 88)
    print(
        f"  p* = {assumptions.c_offer:.0f} / ({assumptions.r_save:.2f}"
        f" × {assumptions.ltv_saved:.0f}) = **{star:.4f}**"
    )

    if args.msno:
        selected = np.flatnonzero(np.isin(mar.msno.to_numpy(), np.array(args.msno)))
        if selected.size == 0:
            sys.exit(f"Mar cohort 裡找不到這些 msno：{args.msno}")
        print(f"  ⚠️ --msno 模式：只解釋指定的 {selected.size} 位，不套用名單門檻")
    else:
        selected = np.flatnonzero(pred > star)
        if selected.size == 0:
            sys.exit(f"沒有人的機率高於 p* = {star:.4f}，名單是空的（這本身可能就是結論）")
        # 名單依機率由高到低，營運要的是「先打誰」。
        selected = selected[np.argsort(-pred[selected], kind="stable")]
        k = selected.size / mar.X.height
        print(
            f"  名單 {selected.size:,} 人 / cohort {mar.X.height:,} 人 = {k:.2%}\n"
            f"  名單最低機率 {pred[selected].min():.4f}"
        )

    # ---- 歸因只算名單這些人 ----
    #
    # 全 cohort 的 SHAP 是 97 萬 × 62 的 float64（約 480 MB）。名單只有幾萬人，
    # 而 M5 要交付的正是那份名單的原因碼。單一用戶查詢走 --msno，M6 走同一個
    # `attribute()`，都不需要整份矩陣。
    X_sel = mar.X[selected]
    print("\n計算 SHAP 歸因...", flush=True)
    attr = attribute(fitted, X_sel)
    gap = assert_local_accuracy(fitted, X_sel, attr)
    print(f"  加總恆等式 sigmoid(base + Σshap) vs predict：最大差 {gap:.3e}（容差 1e-06）")

    top = top_contributors(attr, X_sel, k=TOP_K, groups=feature_group)
    # 兩層：`add_reasons` 造句與標量測時點，`mark_display` 決定哪幾句呈現。
    # 後者只加旗標不刪列 —— 稽核表要答得出「當時為什麼沒印」。
    reasons = mark_display(add_reasons(top, X_sel), min_relative=args.min_relative)

    msno_sel = mar.msno[selected]
    p_sel = pred[selected]

    wide = (
        wide_reasons(reasons, msno_sel, k=TOP_K)
        .with_columns(
            pl.Series("p_churn", p_sel).round(6),
            # 這一位的期望淨收益 = p × r × LTV − C。名單的定義是它為正。
            pl.Series(
                "expected_net",
                p_sel * assumptions.r_save * assumptions.ltv_saved - assumptions.c_offer,
            ).round(1),
        )
        .select(
            "msno",
            "p_churn",
            "expected_net",
            *[
                f"{prefix}_{i}"
                for i in range(1, TOP_K + 1)
                for prefix in ("reason", "group", "group_shap", "expiry_dated")
            ],
        )
    )
    audit = audit_frame(reasons, msno_sel)

    print("\n" + "=" * 88)
    print(f"二、名單樣本（沿名單等距取 {min(args.sample, wide.height)} 位，不是前 N 位）")
    print("=" * 88)
    print_samples(wide, args.sample)

    # ---- 彙總 ----
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(180)

    # 「每人幾句」與「主要在抓什麼」都問**營運看到的**那些句子，所以先過濾。
    # 稽核的數字另外報（第四節）—— 兩者分母不同，混用會兩邊都講不清。
    shown = reasons.filter(pl.col("displayed"))
    counts = rank_distribution(shown, wide.height)
    grouped_top1 = top1_groups(shown)

    # 分組前後對照：不分組時 Top-1 是哪個特徵，把它映回它所屬的組。
    # 若分組只是靠「組大」取勝，兩張表的排名會明顯不同 —— 這是自我檢查。
    plain = add_reasons(top_contributors(attr, X_sel, k=TOP_K), X_sel)
    plain_top1 = (
        plain.filter(pl.col("rank") == 1)
        .with_columns(
            pl.col("feature").map_elements(feature_group, return_dtype=pl.String).alias("所屬組")
        )
        .group_by("所屬組")
        .agg(pl.len().alias("人數"))
        .sort("人數", descending=True)
        .with_columns((pl.col("人數") / wide.height).alias("不分組佔比"))
    )

    print("\n" + "=" * 88)
    print("三、這份名單主要在抓什麼")
    print("=" * 88)
    print(grouped_top1)
    print("\n  不分組（純取 Top-1 特徵）時，該特徵所屬組的分布：")
    print(plain_top1)
    print(
        "\n  兩張表若排名相近，代表分組沒有改變結論、只是讓句子不重複；\n"
        "  差很多則要看是不是大的組靠成員數取勝（見 src/explain/reasons.py 的說明）。"
    )

    print("\n  每人實際呈現幾句原因碼：")
    print(counts)

    impact = display_impact(reasons)
    print("\n" + "=" * 88)
    print(f"四、呈現門檻（relative_to_top ≥ {args.min_relative}）壓掉了什麼")
    print("=" * 88)
    print(
        f"  候選 {impact['候選句數']:,} 句 → 呈現 {impact['呈現句數']:,} 句"
        f"（壓下 {impact['壓下句數']:,} 句）\n"
        f"  受影響人數：{impact['受影響人數']:,} 人"
        f"（{impact['受影響人數比例']:.2%}）—— 他們少一到兩句\n"
        f"  被壓下句子的相對貢獻中位數：{impact['被壓下句子的相對貢獻中位數']:.2%}"
        "（相對於該用戶的第 1 名）"
    )
    print(
        "\n  ⚠️ 這是**呈現判斷，不是統計檢定** —— SHAP 沒有提供「這個貢獻顯著嗎」的分布。\n"
        "     門檻值與它壓掉多少一起寫進 manifest，被壓下的每一句留在稽核表裡，\n"
        f"     帶著 suppression_reason —— 事後查得出當時為什麼沒印（{AUDIT_NAME}）。"
    )

    # 到期日訊號的比例分兩份報：營運看到的那些句子、以及全部候選。
    # 兩者分母不同，答的是不同的問題，任一單獨呈現都會被誤讀。
    share = expiry_dated_share(shown)
    share_all = expiry_dated_share(reasons)
    print("\n" + "=" * 88)
    print("五、有多少解釋撐不到 M6 的 T−7 版本")
    print("=" * 88)
    print(
        f"  【營運呈現的句子】到期日訊號 {share['到期日訊號句數']:,} / "
        f"{share['原因碼句數']:,} 句（{share['句數比例']:.2%}）\n"
        f"  受影響人數：{share['受影響人數']:,} / {share['有原因碼的人數']:,} 人"
        f"（{share['受影響人數比例']:.2%}）\n"
        f"  佔呈現貢獻總和：{share['到期日訊號佔 Top-3 貢獻的比例']:.2%}\n"
        f"\n  【全部候選（稽核）】句數比例 {share_all['句數比例']:.2%}"
        f"　受影響人數比例 {share_all['受影響人數比例']:.2%}"
        f"　佔貢獻總和 {share_all['到期日訊號佔 Top-3 貢獻的比例']:.2%}"
    )
    print(
        "\n  ⚠️ 這是「解釋有多少會消失」的估計，**不是「分數會掉多少」** ——\n"
        "     SHAP 貢獻量與 log loss 的變化之間沒有換算關係。\n"
        "  ⚠️ 正解是重訓 T−7 版本，不是推論時遮掉這一欄：遮一欄會讓加總恆等式\n"
        "     破掉，破的量剛好是這個訊號的全部強度。"
    )

    # 名單自己的校準偏差（--msno 模式沒有名單，跳過）
    slice_cal = None
    if not args.msno:
        slice_cal = subset_calibration(mar.y, pred, k=selected.size / mar.X.height)
        print(
            f"\n  名單自己的校準偏差：平均預測 {slice_cal['平均預測']:.4%}"
            f"　實際 {slice_cal['實際流失率']:.4%}"
            f"（相對 {slice_cal['相對偏差']:+.2%}）—— 全體的數字不可套用到這裡"
        )

    print("\n產生圖表...")
    figure = plot_reasons(grouped_top1, share, counts, paths.figures)

    # ---- 輸出：營運呈現一份、底層稽核一份 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / ("targeting_list.csv" if not args.msno else "explained_users.csv")
    wide.write_csv(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()[:16]
    print(f"    營運名單已存 → reports/explanations/{csv_path.name}（{wide.height:,} 列）")

    audit_path = OUT_DIR / AUDIT_NAME
    audit.write_csv(audit_path)
    audit_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()[:16]
    print(
        f"    稽核表已存 → reports/explanations/{audit_path.name}（{audit.height:,} 列，含未呈現）"
    )

    provenance = cache_provenance(paths)
    if provenance["stale"]:
        print(f"⚠️ 這些快取的程式版本指紋與現行程式不符：{provenance['stale']}")

    manifest = {
        # --- 這份名單是誰、在什麼時候、用哪一版程式算的 ---
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": sha,
        # 執行**開始時**的工作區狀態（見 main() 開頭）。本次執行自己寫出的
        # 圖與 CSV 不算污染，否則這個旗標永遠是 True。
        "git_dirty": dirty,
        "configs": ["configs/business.yaml", "configs/model_comparison.yaml"],
        "fingerprints": provenance,
        # --- ⚠️ 必填：cutoff 定義 ---
        #
        # 少了它，這份原因碼被拿到 M6 的 T−7 版本使用時沒有任何東西會擋，
        # 而 `last_is_cancel` 那一句在那個版本根本還沒發生。
        "cutoff_definition": CUTOFF_DEFINITION,
        "model": {
            "name": ADOPTED_MODEL,
            "best_iteration": int(fitted.best_iteration),
            "mar_log_loss": round(mar_loss, 5),
            "calibrated": False,  # §7.10 決定校準器不上線
        },
        "cohort": {"train": FEB.name, "eval": MAR.name, "n_eval": int(mar.X.height)},
        "assumptions": assumptions.summary(),
        "list": {
            "definition": "p_churn > p_star",
            "mode": "msno" if args.msno else "threshold",
            "n": int(wide.height),
            "k": round(wide.height / mar.X.height, 4),
            "min_probability": round(float(p_sel.min()), 6),
            "csv": f"reports/explanations/{csv_path.name}",
            "csv_sha256_16": digest,
            "csv_in_git": False,  # *.csv 在 .gitignore（競賽規則）
        },
        # --- 呈現門檻：營運看到的與稽核看到的不是同一組句子 ---
        #
        # ⚠️ 這是呈現判斷不是統計檢定，所以門檻值與它的代價都要記下來。少了
        # 這一段，同一份名單用不同門檻跑出來的兩個 CSV 無法分辨。
        "display": {
            "rule": "relative_to_top >= min_relative",
            "min_relative": args.min_relative,
            "impact": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in impact.items()},
            "audit_csv": f"reports/explanations/{audit_path.name}",
            "audit_csv_sha256_16": audit_digest,
            "audit_rows": int(audit.height),
            "audit_columns": list(audit.columns),
        },
        "attribution": {
            "method": "TreeSHAP（CatBoost 原生 ShapValues）",
            "space": "log-odds",
            "top_k": TOP_K,
            "local_accuracy_max_gap": float(f"{gap:.3e}"),
            "grouped": True,
        },
        # ⚠️ 這一段的數字全部是**營運呈現的那些句子**（`displayed == True`）。
        # 全部候選的版本另列在 `expiry_dated_all_candidates`，兩者分母不同。
        "reasons": {
            "n_sentences_displayed": int(shown.height),
            "sentences_per_person": {
                str(row["句數"]): int(row["人數"]) for row in counts.iter_rows(named=True)
            },
            "top1_groups": {
                row["group"]: round(row["佔比"], 4) for row in grouped_top1.iter_rows(named=True)
            },
            "expiry_dated": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in share.items()
            },
            "expiry_dated_all_candidates": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in share_all.items()
            },
        },
        "targeted_slice_calibration": (
            {k: round(float(v), 6) for k, v in slice_cal.items()} if slice_cal else None
        ),
        "global_mean_abs_shap_top10": {
            row["feature"]: round(row["mean_abs_shap"], 5)
            for row in mean_abs_attribution(attr).head(10).iter_rows(named=True)
        },
        "figures": [figure],
        # --- 讀法 ---
        #
        # 這幾句在 manifest 裡而不只在文件裡，是因為 manifest 會被單獨拿出去看。
        "how_to_read": [
            "這份名單是一個函式的輸出，不是一份清單。改 configs/business.yaml 的"
            " C_offer 會改變名單成員，但不會改變任何人的機率與原因碼；改模型或特徵"
            "則三者全變。",
            "CSV 不進 git，每次 make explain 重新產生。下游不得快取它 ——"
            " M6 的 /predict 要即時算，不是查表。",
            "兩份 CSV 的用途不同：targeting_list 是營運呈現（只有該印的句子），"
            f"{AUDIT_NAME} 是稽核（全部候選 + suppression_reason）。"
            "『這個人只有一個理由』與『他的第三個理由太弱所以沒印』是不同的事，"
            "只有稽核表分得出來。",
            f"cutoff_definition = {CUTOFF_DEFINITION}。標成到期日訊號的原因碼"
            "（last_is_cancel）不可沿用到 M6 的 T−7 版本，那個版本必須重訓。",
            "SHAP 的單位是 log-odds，不是機率。貢獻 +0.8 不等於流失率多 80%。",
        ],
    }
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    manifest 已存 → reports/explanations/{manifest_path.name}")

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    for line in manifest["how_to_read"]:
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
