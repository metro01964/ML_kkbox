"""M6 · 提前 7 天評分的代價（SPEC §4.3）。

    uv run python scripts/lead_time.py
    make lead-time

§4.3 的 M6 追加：實務上挽回優惠要提前寄出才來得及，所以**真正能上線的模型是
提前評分的那一個**。SPEC 預告「分數必然下降 —— 這個下降幅度要誠實寫進 README」。
這支腳本把那個幅度量出來。

## 動的只有 cutoff，不是標籤

標籤永遠是「到期後 30 天內有沒有續訂」。`lead_days = 7` 只改變「評分那一刻
看得到什麼」：到期日當天的續訂或取消還沒發生，於是 `last_is_cancel` 幾乎必然
是 0 —— 而 M5 量到它佔投放名單解釋強度的 **42.35%**（§7.14），M3 量到它佔
**35.2% 的 gain**（§7.2）。

## ⚠️ 兩個版本的 cohort 成員不同，所以要比兩次

提前 7 天評分時，「到期前 7 天內才第一次交易」的人還沒有可用的歷史 —— 截斷
之後他一列都不剩，整個掉出 cohort。那不是 bug 而是部署現實，但它的後果是
**兩份驗證集不是同一批人，log loss 不能直接比大小**。

所以報兩組數字：

    各自的驗證集    每個模型在自己的 cohort 上的分數（人數不同，僅供參考）
    共同子集        兩邊都在的那些人，同一組標籤、同一批人 —— **這才是代價**

同 §7.13 的教訓：分母不同的兩個比例不可互相替代，那就把兩個都寫出來。

## 這支腳本會觸發快取重建

`lead_days` 進了 `CohortSpec`，`src.data.cohort` 的原始碼因此改變 → cohort 與
收聽特徵的邏輯指紋全部變動 → 四份快取都會重算。**內容不變**（T=0 的路徑一行
邏輯都沒動），但第一次跑要多花幾分鐘。收斂檔的日期下界也往前 7 天，所以會多
產生一份 `user_logs_window_*.parquet`。
"""

from __future__ import annotations

import json
import subprocess
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.data import FEB, FEB_T7, MAR, MAR_T7, CohortSpec, cutoff_window
from src.evaluation import log_loss, repeat_vs_new
from src.models.adopted import ADOPTED_MODEL, fit_adopted, load_adopted_config
from src.models.compare import split_for_early_stopping
from src.models.train import load_cohort_features

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Microsoft YaHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["figure.dpi"] = 110

OUT_PATH = REPO_ROOT / "reports" / "lead_time.json"


def git_state() -> tuple[str, bool]:
    """SHA 與工作區狀態。**在寫出任何東西之前呼叫** —— 圖 15 進 git，
    寫完再問就會永遠回報「髒」（M5 踩過這個坑，見 scripts/explain.py）。"""

    def run(*args: str) -> str:
        return subprocess.run(
            args, capture_output=True, text=True, cwd=REPO_ROOT, check=False
        ).stdout.strip()

    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain"))


def load_configs() -> dict:
    path = REPO_ROOT / "configs" / "business.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # 只借用 features.use_logs（要不要含收聽特徵），業務參數與本腳本無關。
    return {"features": cfg.get("features", {"use_logs": True})}


def train_one(train_spec: CohortSpec, valid_spec: CohortSpec, cfg: dict, paths) -> dict:
    """訓練一個版本並回傳預測。兩個版本走完全相同的路徑，只有 spec 不同。"""
    _, train_cfg = load_adopted_config()
    lo, hi = cutoff_window(train_spec)

    print(
        f"\n{'=' * 88}\n{train_spec.name} → {valid_spec.name}"
        f"（提前 {train_spec.lead_days} 天，cutoff 落在 {lo}~{hi}）\n{'=' * 88}",
        flush=True,
    )
    tr, va = load_cohort_features(paths, cfg, train_spec=train_spec, valid_spec=valid_spec)
    inner_train, es = split_for_early_stopping(tr, train_cfg)

    fitted = fit_adopted(inner_train, es, all_train=tr)
    pred = fitted.predict(va.X)
    loss = float(log_loss(va.y, pred))
    print(f"  停在第 {fitted.best_iteration} 輪　{valid_spec.name} log loss {loss:.5f}")

    return {
        "spec": train_spec,
        "valid_spec": valid_spec,
        "fitted": fitted,
        "loss": loss,
        "scored": pl.DataFrame({"msno": va.msno, "y": va.y, "p": pred}),
        "train": tr,
        "valid": va,
    }


def dropped_profile(base_fs, lead_fs) -> dict:
    """提前評分之後掉出 cohort 的那些人是誰。

    ## 為什麼這一組數字比「掉了幾個人」重要得多

    掉出去的是「到期前 7 天內才第一次交易」的人。實測他們的流失率是
    **70.25%**（全體 8.99% 的 7.8 倍）—— 也就是說 T−7 模型**結構性地看不到
    風險最高的那一小撮人**。

    這不是可以修的 bug：那些人在評分時點真的還沒有任何歷史。它是一個部署
    約束，要寫進 MODEL_CARD：這批人需要另一個機制（在到期日當天補評一次，
    或用「首購後 N 天」觸發），不能靠 T−7 的名單接住。

    只報「掉了 0.16%」會讓人以為影響微不足道，那是錯的印象。
    """
    base = pl.DataFrame({"msno": base_fs.msno, "y": base_fs.y})
    kept = pl.DataFrame({"msno": lead_fs.msno})
    out = base.join(kept, on="msno", how="anti")
    return {
        "人數": out.height,
        "佔比": out.height / base.height,
        "流失率": float(out["y"].mean()) if out.height else None,
        "全體流失率": float(base["y"].mean()),
    }


def cancel_flag_profile(fs) -> dict:
    """`last_is_cancel` 在這個版本裡長什麼樣 —— 提前評分讓它幾乎消失。"""
    col = fs.X["last_is_cancel"]
    return {
        "為 1 的人數": int((col == 1.0).sum()),
        "為 1 的比例": float((col == 1.0).mean()),
        "缺失人數": int(col.null_count()),
    }


def gain_share(fitted, features: tuple[str, ...]) -> dict:
    imp = fitted.importance
    lookup = dict(zip(imp["feature"].to_list(), imp["gain_share"].to_list(), strict=True))
    return {f: round(float(lookup.get(f, 0.0)), 5) for f in features}


def common_subset(base: dict, lead: dict) -> pl.DataFrame:
    """兩個版本都在的那些人，各自的預測並排。

    Raises:
        ValueError: 同一位用戶在兩邊的標籤不一致 —— 那代表某一邊的標籤接錯了，
            而標籤是唯一真實的東西，錯了整個對照就沒有意義。
    """
    joined = base["scored"].join(
        lead["scored"].rename({"y": "y_lead", "p": "p_lead"}), on="msno", how="inner"
    )
    mismatch = joined.filter(pl.col("y") != pl.col("y_lead"))
    if mismatch.height:
        raise ValueError(
            f"{mismatch.height:,} 位用戶在兩個版本的標籤不同 —— 標籤不該隨 cutoff 改變，"
            "這代表某一邊接錯了人。"
        )
    return joined.drop("y_lead")


def plot_comparison(rows: pl.DataFrame, gains: dict, figdir) -> str:
    """圖 15：分群分數的變化，以及 gain 從哪裡搬到哪裡。"""
    fig, (ax_l, ax_g) = plt.subplots(1, 2, figsize=(14, 5))

    groups = rows["分群"].to_list()
    x = np.arange(len(groups))
    width = 0.38
    ax_l.bar(x - width / 2, rows["T=0"].to_numpy(), width, label="到期日評分", color="#1f77b4")
    ax_l.bar(x + width / 2, rows["T-7"].to_numpy(), width, label="提前 7 天評分", color="#d62728")
    for i, (a, b) in enumerate(zip(rows["T=0"], rows["T-7"], strict=True)):
        ax_l.text(i - width / 2, a, f"{a:.4f}", ha="center", va="bottom", fontsize=8)
        ax_l.text(i + width / 2, b, f"{b:.4f}", ha="center", va="bottom", fontsize=8)
    ax_l.set_xticks(x, groups)
    ax_l.set_ylabel("log loss（越低越好）")
    ax_l.set_title("同一批人、同一組標籤：提前 7 天評分的代價", fontsize=11)
    ax_l.legend(fontsize=9)
    ax_l.grid(axis="y", alpha=0.25)

    names = list(gains)[::-1]
    base = [gains[n]["T=0"] * 100 for n in names]
    lead = [gains[n]["T-7"] * 100 for n in names]
    y = np.arange(len(names))
    ax_g.barh(y - width / 2, base, width, label="到期日評分", color="#1f77b4")
    ax_g.barh(y + width / 2, lead, width, label="提前 7 天評分", color="#d62728")
    ax_g.set_yticks(y, names, fontsize=8)
    ax_g.set_xlabel("gain 佔比（%）")
    ax_g.set_title("訊號搬家：last_is_cancel 消失之後，模型改用什麼", fontsize=11)
    ax_g.legend(fontsize=9, loc="lower right")
    ax_g.grid(axis="x", alpha=0.25)

    fig.suptitle("提前評分的代價與訊號重分配（SPEC §4.3 的 M6 追加）", fontsize=12)
    path = figdir / "15_lead_time.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{path.name}")
    return f"reports/figures/{path.name}"


def main() -> int:
    sha, dirty = git_state()
    try:
        paths = load_paths().ensure()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這批數字無法用 SHA 回溯。")

    cfg = load_configs()
    base = train_one(FEB, MAR, cfg, paths)
    lead = train_one(FEB_T7, MAR_T7, cfg, paths)

    # ---- 一、成員的變化 ----
    print("\n" + "=" * 88)
    print("一、提前評分讓誰掉出 cohort")
    print("=" * 88)
    both = common_subset(base, lead)
    dropped = base["valid"].X.height - both.height
    drop_eval = dropped_profile(base["valid"], lead["valid"])
    drop_train = dropped_profile(base["train"], lead["train"])
    print(
        f"  {MAR.name} {base['valid'].X.height:,} 人　"
        f"{MAR_T7.name} {lead['valid'].X.height:,} 人　共同 {both.height:,} 人\n"
        f"  掉出去 {dropped:,} 人（{dropped / base['valid'].X.height:.3%}）"
        " —— 到期前 7 天內才第一次交易，提前評分時還沒有可用的歷史\n"
        f"  訓練集：{FEB.name} {base['train'].X.height:,} → "
        f"{FEB_T7.name} {lead['train'].X.height:,} 人"
    )
    print(
        f"\n  ⚠️ **掉出去的人流失率 {drop_eval['流失率']:.2%}**，"
        f"是全體 {drop_eval['全體流失率']:.2%} 的 "
        f"{drop_eval['流失率'] / drop_eval['全體流失率']:.1f} 倍。\n"
        f"     （訓練 cohort 那邊：{drop_train['人數']:,} 人，流失率 "
        f"{drop_train['流失率']:.2%}）\n"
        "     T−7 模型**結構性地看不到風險最高的那一小撮人** —— 那不是可以修的 bug，\n"
        "     是部署約束：這批人需要另一個機制（到期日當天補評一次，或用首購觸發），\n"
        "     不能靠 T−7 的名單接住。只報「掉了 0.16%」會給人錯的印象。"
    )

    # ---- 二、分數 ----
    print("\n" + "=" * 88)
    print("二、分數：各自的驗證集，以及共同子集")
    print("=" * 88)
    print(
        f"  各自的驗證集（人數不同，僅供參考）\n"
        f"    到期日評分      {base['loss']:.5f}（{base['valid'].X.height:,} 人）\n"
        f"    提前 7 天評分    {lead['loss']:.5f}（{lead['valid'].X.height:,} 人）"
    )

    common_base = float(log_loss(both["y"], both["p"]))
    common_lead = float(log_loss(both["y"], both["p_lead"]))
    delta = common_lead / common_base - 1
    print(
        f"\n  **共同子集（{both.height:,} 人，同一組標籤）—— 這才是代價**\n"
        f"    到期日評分      {common_base:.5f}\n"
        f"    提前 7 天評分    {common_lead:.5f}\n"
        f"    **退步 {delta:+.2%}**"
    )

    # ---- 三、分群（§4.5）----
    segment = repeat_vs_new(
        both["msno"], base["train"].msno
    )  # 以 T=0 的訓練集為「上一期是否出現過」的判準
    rows = []
    for name in ("全體", "重複用戶", "新進用戶"):
        mask = pl.Series([True] * both.height) if name == "全體" else (segment == name)
        part = both.filter(mask)
        if not part.height:
            continue
        rows.append(
            {
                "分群": name,
                "人數": part.height,
                "T=0": round(float(log_loss(part["y"], part["p"])), 5),
                "T-7": round(float(log_loss(part["y"], part["p_lead"])), 5),
            }
        )
    table = pl.DataFrame(rows).with_columns(
        ((pl.col("T-7") / pl.col("T=0") - 1) * 100).round(2).alias("退步 %")
    )
    print("\n" + "=" * 88)
    print("三、分群（SPEC §4.5）")
    print("=" * 88)
    print(table)
    print(
        "\n  ⚠️ 分群依「是否出現在 T=0 的訓練 cohort」判定，兩欄用的是同一個分群，所以差值是可比的。"
    )

    # ---- 四、訊號搬家 ----
    watch = (
        "last_is_cancel",
        "days_since_last_tx",
        "last_is_auto_renew",
        "last_payment_method_id",
        "last_price",
        "n_tx",
        "cancel_rate",
        "log_max_days_before",
    )
    g_base = gain_share(base["fitted"], watch)
    g_lead = gain_share(lead["fitted"], watch)
    gains = {name: {"T=0": g_base[name], "T-7": g_lead[name]} for name in watch}

    flag_base = cancel_flag_profile(base["valid"])
    flag_lead = cancel_flag_profile(lead["valid"])
    print("\n" + "=" * 88)
    print("四、訊號搬家：last_is_cancel 消失之後")
    print("=" * 88)
    print(
        f"  旗標本身（驗證集）：為 1 的比例 {flag_base['為 1 的比例']:.2%}"
        f" → {flag_lead['為 1 的比例']:.2%}\n"
        f"  它的 gain 佔比：{g_base['last_is_cancel']:.2%} → {g_lead['last_is_cancel']:.2%}"
    )
    print(
        pl.DataFrame(
            {
                "feature": list(watch),
                "T=0 gain%": [round(g_base[n] * 100, 2) for n in watch],
                "T-7 gain%": [round(g_lead[n] * 100, 2) for n in watch],
            }
        )
    )

    print("\n產生圖表...")
    figure = plot_comparison(table, gains, paths.figures)

    summary = {
        "git_sha": sha,
        "git_dirty": dirty,
        "model": ADOPTED_MODEL,
        "lead_days": FEB_T7.lead_days,
        "cutoff_windows": {s.name: list(cutoff_window(s)) for s in (FEB, MAR, FEB_T7, MAR_T7)},
        "best_iteration": {
            "t0": int(base["fitted"].best_iteration),
            "t7": int(lead["fitted"].best_iteration),
        },
        "cohort_sizes": {
            FEB.name: int(base["train"].X.height),
            MAR.name: int(base["valid"].X.height),
            FEB_T7.name: int(lead["train"].X.height),
            MAR_T7.name: int(lead["valid"].X.height),
            "common_eval": int(both.height),
            "dropped_from_eval": int(dropped),
        },
        "churn_rate": {
            FEB.name: round(float(base["train"].y.mean()), 6),
            MAR.name: round(float(base["valid"].y.mean()), 6),
            FEB_T7.name: round(float(lead["train"].y.mean()), 6),
            MAR_T7.name: round(float(lead["valid"].y.mean()), 6),
        },
        # ⚠️ 這一段比「掉了幾個人」重要得多，見 dropped_profile 的說明。
        "dropped_users": {
            "eval": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in drop_eval.items()},
            "train": {
                k: (round(v, 6) if isinstance(v, float) else v) for k, v in drop_train.items()
            },
        },
        "log_loss": {
            "own_eval_set": {"t0": round(base["loss"], 5), "t7": round(lead["loss"], 5)},
            # ⚠️ 這一組才可以互相比大小：同一批人、同一組標籤。
            "common_subset": {
                "t0": round(common_base, 5),
                "t7": round(common_lead, 5),
                "relative_change": round(delta, 4),
            },
        },
        "segments": table.to_dicts(),
        "last_is_cancel": {"t0": flag_base, "t7": flag_lead},
        "gain_share": gains,
        "figures": [figure],
        "how_to_read": [
            "兩個版本的驗證集不是同一批人（提前評分讓沒有歷史的人掉出去），"
            "所以 own_eval_set 的兩個數字不可互相比大小；要比就看 common_subset。",
            "標籤沒有改變 —— 動的只有 cutoff。退步的幅度就是「提前 7 天」的代價。",
            "這個退步是**應該發生的**：到期日當天的取消在部署時看不到，"
            "用它算出來的分數不是能上線的分數（SPEC §4.3）。",
            "⚠️ 掉出 cohort 的那批人流失率是全體的 7.8 倍 —— T−7 模型看不到風險最高的"
            "一小撮人。那是部署約束不是 bug，要寫進 MODEL_CARD 並用另一個機制接住。",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    摘要已存 → reports/{OUT_PATH.name}")

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    for line in summary["how_to_read"]:
        print(f"  · {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
