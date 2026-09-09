"""M6 · Apr cohort 的 Kaggle 提交檔（SPEC §3.3 的最後一條門檻）。

    uv run python scripts/predict_kaggle.py     # make kaggle

產出 `reports/kaggle/submission_<artifact>.csv`（907,471 列）與一份 manifest。
**不重訓、不自動提交** —— 提交是使用者自己的動作（見最後印出的指令）。

## 這支腳本存在的理由

§3.3 把「Apr cohort 的 Kaggle late submission 分數」列為 M6 的門檻，因為那是
**唯一與官方私榜（🥇 0.07974 / 第 20 名 0.10834）可比的數字**。本專案回報的所有
分數都算在 Mar cohort 上，跟排行榜不是同一個資料集。

## ⚠️ 本地算不出這份預測的分數，而那正是它要處理的事

`sample_submission_v2.csv` 的 `is_churn` 全是 0（佔位值）。`APR_FIXED` 因此標了
`labels_are_real=False`，`build_cohort()` 會把那一欄填成 **null** —— 任何在本地算
log loss 的嘗試都會壞掉，而不是回一個看起來合理的數字。

那麼「該期待幾分」從哪裡來？從**結構相同的本地對照**：`feb_fixed → mar_fixed`
（同一個固定評分日的設計、同一份特徵集）。那個分數寫在 artifact 的
`metrics.log_loss` 裡，本腳本會把它印出來當**事前登記的預期**。

## 為什麼一定要用固定評分日的 artifact

交易與日誌都只到 2017-03-31，而測試集要預測 4 月到期的人。`cutoff = 到期日 − 7 天`
對 **77.64%** 的測試用戶會落在資料結束之後 —— 那時 as-of 截斷變成空操作，特徵改由
資料集的結尾決定（見 `src/data/cohort.py` 的 `assert_data_covers_cutoffs`）。

所以只有 `--design fixed` 的 artifact 能用在這裡。本腳本會擋下其他的：特徵集不同
（`fixed` 多一欄 `days_to_expire`）本來就會讓 `assert_features_match()` 報錯，但
那個訊息講的是欄位，不是「你拿錯設計了」。

## 掉出 cohort 的人怎麼處理

Kaggle 要求 907,471 列都在。若有用戶在評分日之前**一筆交易都沒有**，他不會有特徵
（實測 0 人，但程式不能假設）。那時填**訓練 cohort 的流失率** —— 那是部署時唯一
知道的先驗，而且它讓 log loss 的損失可控。填 0 或 0.5 都是在編一個答案，而填的
人數與值都要寫進 manifest。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime

import numpy as np
import polars as pl

from src.config import REPO_ROOT, load_paths
from src.data import APR_FIXED, MAR_FIXED, build_cohort
from src.evaluation import EPSILON, band, score_psi
from src.features import build_features, build_log_features
from src.serving.artifact import load_artifact

OUT_DIR = REPO_ROOT / "reports" / "kaggle"
MANIFEST = REPO_ROOT / "reports" / "kaggle_submission.json"

# 這份 artifact 必須是**固定評分日這個設計**（見模組開頭）。
#
# ⚠️ 比的是設計的種類，不是 `cutoff_definition` 整個字串 —— 每個月的評分日本來
# 就不同（訓練用 20170131，套用到 4 月是 20170331）。第一版比了整個字串，於是
# 一份完全正確的 artifact 被擋下來（見 Artifact.scoring_design）。
REQUIRED_DESIGN = "fixed_score_date"


def git_state() -> tuple[str, bool]:
    def run(*args: str) -> str:
        return subprocess.run(
            args, capture_output=True, text=True, cwd=REPO_ROOT, check=False
        ).stdout.strip()

    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain"))


def features_for(spec, paths, artifact):
    """某個 cohort 的特徵矩陣，欄位對齊 artifact 記錄的順序。"""
    raw = build_cohort(spec, paths, verbose=False)
    logs = build_log_features(spec, paths, verbose=False) if artifact.uses_log_features else None
    fs = build_features(raw, logs)
    missing = sorted(set(artifact.feature_names) - set(fs.X.columns))
    if missing:
        sys.exit(
            f"{spec.name} 的特徵少了 {missing}。"
            "這通常代表 artifact 與現在的程式不是同一版（請重新匯出）。"
        )
    # 依名字重排 —— 順序的唯一來源是 artifact（見 src/serving/payload.py）。
    return raw, fs.X.select(artifact.feature_names)


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 產生 Kaggle 提交檔")
    ap.add_argument("--artifact", default="catboost_fixed", help="要用哪一份 artifact")
    ap.add_argument(
        "--skip-reference",
        action="store_true",
        help="不算與 mar_fixed 的分數漂移對照（省一次全 cohort 推論）",
    )
    args = ap.parse_args()

    sha, dirty = git_state()
    try:
        paths = load_paths().ensure()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這份提交檔無法用 SHA 回溯。")

    art = load_artifact(name=args.artifact)
    if art.scoring_design != REQUIRED_DESIGN:
        sys.exit(
            f"這份 artifact 的評分設計是 {art.scoring_design}"
            f"（{art.cutoff_definition}），而 Apr cohort 需要 {REQUIRED_DESIGN}。\n"
            "  交易與日誌只到 2017-03-31：『到期日 − 7 天』對 77.64% 的測試用戶會落在資料"
            "結束之後，\n  那時特徵是被資料集的結尾截斷的，不是被我們的規則截斷。\n"
            "  請先跑：uv run python scripts/export_model.py --design fixed"
        )

    print(f"\n{'=' * 92}")
    print(
        f"Apr cohort 推論　artifact {art.directory.name}"
        f"（{art.meta['model']['name']}，{art.cutoff_definition}）"
    )
    print(f"{'=' * 92}")
    for w in art.warnings:
        print(f"⚠️ {w}")

    # ---- 事前登記的預期 ----
    #
    # 先印，再算。順序有意義：分數出來之後才說「我本來就覺得會這樣」不是預先登記。
    proxy = art.meta["metrics"]
    lead = art.meta.get("effective_lead_days", {})
    print(
        f"\n【事前登記】結構相同的本地對照（{art.meta['cohort']['train']} → "
        f"{art.meta['cohort']['eval']}）log loss **{proxy['log_loss']}**"
        f"（常數基準 {proxy['constant_baseline']}）。\n"
        f"  Apr 的提交分數應該落在這個量級 —— 兩者的評分規則、特徵集、提前天數分布"
        f"（中位數 {lead.get('median')} 天）都相同，\n"
        "  差別只在 cohort 的月份與基準率。⚠️ 官方私榜的 0.10834 算在同一個 Apr "
        "cohort 上，所以那個比較是合法的；\n"
        "  而本專案 Mar 的 0.15367 / 0.17921 與它**不可比**（不同資料集，§3.3）。"
    )

    # ---- Apr cohort ----
    submission = pl.read_csv(paths.raw / APR_FIXED.label_file, columns=["msno"])
    raw, X = features_for(APR_FIXED, paths, art)
    print(f"\n測試名單 {submission.height:,} 人 → 有特徵的 {X.height:,} 人")

    pred = np.asarray(art.fitted.predict(X), dtype=np.float64)
    scored = pl.DataFrame({"msno": raw["msno"], "p": pred})

    # ---- 對齊提交檔的名單與順序 ----
    #
    # left join 之後仍是 null 的人 = 評分日之前一筆交易都沒有（見模組開頭）。
    fallback = float(art.meta["cohort"]["train_churn_rate"])
    out = submission.join(scored, on="msno", how="left")
    n_fallback = int(out["p"].null_count())
    out = out.with_columns(pl.col("p").fill_null(fallback).alias("is_churn")).select(
        "msno", "is_churn"
    )
    if n_fallback:
        print(
            f"  ⚠️ {n_fallback:,} 人沒有任何評分日之前的交易，填訓練 cohort 的流失率"
            f" {fallback:.4%}（那是部署時唯一知道的先驗）"
        )

    # ---- 提交檔的硬性檢查 ----
    #
    # 這些不是 nice-to-have：一份列數不對或含 null 的檔案，Kaggle 會拒收或給出
    # 一個無法解讀的分數，而那時已經用掉一次提交。
    assert out.height == submission.height, f"列數不符：{out.height} vs {submission.height}"
    assert out["is_churn"].null_count() == 0, "有 null 機率"
    assert float(out["is_churn"].min()) > 0.0 and float(out["is_churn"].max()) < 1.0, (
        "機率必須落在開區間 (0, 1) —— log loss 對 0 與 1 是無限大"
    )
    assert out["msno"].n_unique() == out.height, "msno 有重複"

    p = out["is_churn"].to_numpy()
    print(
        f"\n預測分布：平均 {p.mean():.4%}　中位數 {np.median(p):.4%}"
        f"　p99 {np.quantile(p, 0.99):.4%}　最大 {p.max():.4%}"
    )
    print(
        f"  訓練 cohort 的流失率 {fallback:.4%}"
        f"　評估 cohort 的實際流失率 {art.meta['cohort']['eval_churn_rate']:.4%}"
        "　（⚠️ 模型系統性低估，見 MODEL_CARD §5）"
    )
    above = float((p > art.p_star).mean())
    print(f"  超過投放門檻 p* = {art.p_star:.4f} 的比例 {above:.2%}")

    # ---- 唯一拿得到的品質檢查：分數分布 vs 本地對照 ----
    #
    # 沒有標籤，所以不能算分數。但可以問「這份預測的分布，跟結構相同的本地
    # cohort 像不像」—— 不像的話，要嘛 Apr 真的不一樣，要嘛管線有問題。
    drift = None
    if not args.skip_reference:
        print(f"\n與 {MAR_FIXED.name} 的分數分布對照 ...", flush=True)
        _, X_ref = features_for(MAR_FIXED, paths, art)
        ref = np.asarray(art.fitted.predict(X_ref), dtype=np.float64)
        value, floored, _ = score_psi(ref, pred, epsilon=EPSILON)
        drift = {"reference": MAR_FIXED.name, "score_psi": round(value, 6), "band": band(value)}
        print(
            f"  分數 PSI {value:.4f}（{band(value)}）"
            f"{'　⚠️ 有單邊空箱' if floored else ''}\n"
            f"  參考期平均 {ref.mean():.4%} → 當期 {p.mean():.4%}"
            f"（相對 {p.mean() / ref.mean() - 1:+.1%}）"
        )
        print(
            "  ⚠️ 這**不是**分數，是「分布像不像」。PSI 看不到基準率漂移"
            "（§7.17 實測），所以它正常不代表這份提交會準。"
        )

    # ---- 寫檔 ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"submission_{art.directory.name}.csv"
    out.write_csv(csv_path)
    digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()

    manifest = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_sha": sha,
        "git_dirty": dirty,
        "artifact": {
            "name": art.directory.name,
            "model_sha256_16": art.meta["model"]["sha256"][:16],
            "cutoff_definition": art.cutoff_definition,
            "trained_on": art.meta["cohort"]["train"],
            "n_features": art.meta["features"]["n"],
        },
        # ⚠️ 事前登記：這是**提交之前**寫下的預期，不是事後的解釋。
        "pre_registered_expectation": {
            "local_proxy": f"{art.meta['cohort']['train']} → {art.meta['cohort']['eval']}",
            "log_loss": proxy["log_loss"],
            "constant_baseline": proxy["constant_baseline"],
            "effective_lead_days": lead,
            "note": "Apr 提交的分數應與這個量級相當；官方私榜（第 20 名 0.10834）"
            "算在同一個 Apr cohort 上，所以與它的比較是合法的。",
        },
        "cohort": {
            "name": APR_FIXED.name,
            "score_date": APR_FIXED.score_date,
            "n_submission_rows": int(out.height),
            "n_with_features": int(X.height),
            "n_fallback": n_fallback,
            "fallback_value": round(fallback, 6),
            "labels_available": False,
        },
        "predictions": {
            "mean": round(float(p.mean()), 6),
            "median": round(float(np.median(p)), 6),
            "p99": round(float(np.quantile(p, 0.99)), 6),
            "max": round(float(p.max()), 6),
            "min": round(float(p.min()), 6),
            "above_p_star": round(above, 6),
            "p_star": art.p_star,
        },
        "score_drift_vs_local_reference": drift,
        "submission": {
            "csv": f"reports/kaggle/{csv_path.name}",
            "sha256": digest,
            "csv_in_git": False,  # *.csv 在 .gitignore（競賽規則）
        },
        "how_to_read": [
            "本地**算不出**這份預測的分數：Apr cohort 的標籤是 Kaggle 測試集，"
            "`sample_submission_v2.csv` 的 is_churn 全是 0 佔位值（已填成 null）。",
            "所以 `pre_registered_expectation` 是提交前寫下的預期，來自結構相同的"
            f"本地對照（{art.meta['cohort']['train']} → {art.meta['cohort']['eval']}）。",
            "分數 PSI 是「分布像不像」，不是分數。它正常不代表這份提交會準 ——"
            "§7.17 實測 PSI 對基準率漂移完全看不到。",
            "提交是使用者自己的動作。分數回來之後要寫進 README 的成果表與 §3.3，"
            "**無論結果如何** —— 那是這個門檻的意義。",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n  提交檔已存 → reports/kaggle/{csv_path.name}（{out.height:,} 列）")
    print(f"    sha256 {digest[:16]}…")
    print(f"  manifest 已存 → reports/{MANIFEST.name}")
    print("\n" + "=" * 92)
    print("提交（這一步請你自己做 —— 需要接受競賽規則，而那是一個帳號層級的動作）")
    print("=" * 92)
    print(
        f"  uv run kaggle competitions submit -c kkbox-churn-prediction-challenge \\\n"
        f'      -f "{csv_path}" -m "{art.directory.name} / {art.cutoff_definition}"\n'
        "\n  分數回來之後：填進 README 的成果表與 SPEC §3.3 的 M6 那一列，"
        "**無論結果如何**。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
