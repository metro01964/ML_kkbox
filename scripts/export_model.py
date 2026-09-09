"""M6 · 把採用的模型匯出成一份可載入的 artifact（SPEC §7 M6）。

    uv run python scripts/export_model.py                    # lead0：T=0（離線基準）
    uv run python scripts/export_model.py --design lead7     # T−7（提前固定 7 天）
    uv run python scripts/export_model.py --design fixed     # 固定評分日（Kaggle 管線）
    make artifact

服務、HF Spaces Demo、Apr cohort 的 Kaggle 推論管線都載這一份東西，沒有人重訓
（§7.12 的 CatBoost 一次約 4 分鐘）。

## 這支腳本的產出是「模型 + 它的身分證」

metadata 記什麼、為什麼要記那麼多，寫在 `src/serving/artifact.py` 的模組註解。
這裡只補匯出流程本身的兩個決定：

**一、訓練路徑與 M5 完全相同。** 同一份 `load_cohort_features()`、同一個
`split_for_early_stopping()`、同一個 `fit_adopted()`。不共用的話，服務端的模型
就不是報告裡那個模型 —— 而兩邊都會印出很像的分數。

**二、匯出之後立刻載回來對答案。** 存檔／載入是一條有恆等式的路徑：同一批列，
載回來的模型必須給出**完全相同**的機率（不是「差不多」——CatBoost 的原生格式是
無損的）。所以匯出的最後一步是重新載入、逐列比對，不符就非零離開。

理由與 §6.1 用曲線極大值驗 `campaign_curve()`、M5 逐列驗加總恆等式相同：一條
存了又載的路徑，失效方式是「機率安靜地偏掉」，那不能只靠讀程式碼判斷。

## ⚠️ `lead0` 不是能上線的模型

§4.3：挽回優惠要提前寄出才來得及。§7.15 實測共同子集退步 18.11%，而那個較差的
分數才是能上線的數字。`configs/serving.yaml` 預設載的是 `lead7` 那一份。

三種設計的 artifact 目錄名不同（`catboost_lead0d`/`catboost_lead7d`/`catboost_fixed`），
因為**它們是三個模型**，不是同一個模型的三個版本 —— cohort 成員、特徵集
（`fixed` 多一欄 `days_to_expire`）與提前天數都不同，**分數不可互相比大小**。

`fixed` 的存在理由是 Kaggle：交易與日誌都只到 2017-03-31，而測試集要預測 4 月
到期的人。`到期日 − 7 天` 對 77.64% 的測試用戶會落在資料結束之後（見
`src/data/cohort.py` 的 `assert_data_covers_cutoffs`）。固定評分日把「在 3/31
這一天替所有 4 月到期的人評分」寫成一個誠實的設計，而 `feb_fixed → mar_fixed`
的分數就是「該期待 Kaggle 給什麼」的本地估計。

## `--reuse`：只改業務假設時不必重訓

`p*` 由 `C_offer / (r_save × LTV_saved)` 推導，換一個 C_offer 不會改變任何人的
機率、也不會改變模型。`--reuse` 載回既有 artifact 的模型，重算假設與分數再寫
一次 metadata（約 1 分鐘，全部從快取讀）。

⚠️ 它**不能**用在特徵或程式改了之後 —— 那時 artifact 的邏輯指紋已經不符，
`load_artifact()` 會直接拒絕，這正是它該做的事。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.config import REPO_ROOT, load_paths
from src.data import (
    FEB,
    FEB_FIXED,
    FEB_T7,
    MAR,
    MAR_FIXED,
    MAR_T7,
    CohortSpec,
    cutoff_definition,
    cutoff_window,
)
from src.data.cohort import cohort_fingerprint
from src.evaluation import constant_log_loss, log_loss, resolve_assumptions
from src.features.logs import log_features_fingerprint
from src.fingerprint import read_cache_fingerprint
from src.models.adopted import ADOPTED_MODEL, fit_adopted, load_adopted_config
from src.models.compare import split_for_early_stopping
from src.models.train import load_cohort_features
from src.serving.artifact import load_artifact, save_artifact

# 匯出後的自我對答案要比幾列。全 97 萬列也只是幾秒，但 5,000 列已經足以抓到
# 「載回來的模型不是同一棵樹」——那種錯不會只影響某幾列。
ROUND_TRIP_ROWS = 5_000

# 存檔／載入必須是無損的，所以容差是 0：CatBoost 的 .cbm 存的是同一棵樹的
# 同一組葉子值，不是一個近似。訂一個 1e-9 的容差等於預先接受一個不該存在的差。
ROUND_TRIP_ATOL = 0.0

# 三種評分規則 → (訓練 cohort, 評估 cohort)。
#
#   lead0  cutoff = 到期日             離線基準，**不可上線**（§4.3）
#   lead7  cutoff = 到期日 − 7 天      提前固定天數
#   fixed  cutoff = 上個月最後一天      固定評分日，提前天數 1~30 天 —— Kaggle
#                                      測試集唯一做得到的設計（資料只到 3/31）
DESIGNS: dict[str, tuple[CohortSpec, CohortSpec]] = {
    "lead0": (FEB, MAR),
    "lead7": (FEB_T7, MAR_T7),
    "fixed": (FEB_FIXED, MAR_FIXED),
}

# artifact 目錄的預設名。lead 那兩個沿用既有的名字（`configs/serving.yaml` 與
# 文件都指著它們）—— 改名的代價是一堆文件要跟著改，而換到的只是一致的拼法。
ARTIFACT_SUFFIX = {"lead0": "lead0d", "lead7": "lead7d", "fixed": "fixed"}


def git_state() -> tuple[str, bool]:
    """SHA 與工作區狀態。artifact 不進 git，但它要說得出自己是哪一版程式做的。"""

    def run(*args: str) -> str:
        return subprocess.run(
            args, capture_output=True, text=True, cwd=REPO_ROOT, check=False
        ).stdout.strip()

    return run("git", "rev-parse", "HEAD"), bool(run("git", "status", "--porcelain"))


def load_business_config() -> dict:
    path = REPO_ROOT / "configs" / "business.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "business" not in cfg:
        raise KeyError(f"{path} 缺少 [business] 區段")
    return cfg


def serving_artifact_name() -> str:
    """`configs/serving.yaml` 現在指向哪一份 artifact —— 匯出完要提醒這件事。

    匯出了 T=0 卻忘了服務指向 T−7（或反過來）是一個安靜的錯誤：兩份 artifact
    都存在，服務照樣起得來，只是載的不是剛剛那一份。
    """
    path = REPO_ROOT / "configs" / "serving.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return str(cfg.get("artifact", "（未設定）"))


def cache_fingerprints(paths, specs: tuple[CohortSpec, ...]) -> dict:
    """訓練用的特徵是**哪一版程式**算出來的快取。

    與 M5 的 `cache_provenance()` 同一個理由（§7.11）：git SHA 只證明「執行時
    的程式是這一版」，證明不了「餵進去的特徵是這一版算的」。artifact 兩個都記。
    """
    current = {"cohort": cohort_fingerprint(), "log_features": log_features_fingerprint()}
    caches = {
        spec.name: {
            "cohort": read_cache_fingerprint(paths.interim / f"{spec.name}_cohort_asof.parquet"),
            "log_features": read_cache_fingerprint(
                paths.interim / f"{spec.name}_log_features.parquet"
            ),
        }
        for spec in specs
    }
    stale = [
        f"{name}.{kind}"
        for name, got in caches.items()
        for kind, value in got.items()
        if value != current[kind]
    ]
    return {"cache_code": current, "caches": caches, "stale": stale}


def verify_round_trip(directory, fitted, X, expected: np.ndarray) -> float:
    """載回來的模型必須給出完全相同的機率。

    Returns:
        最大逐列差。

    Raises:
        AssertionError: 超過容差（見 `ROUND_TRIP_ATOL`）。
    """
    # strict=True：這裡就是要確認「現行程式 + 這份 artifact」是一致的組合，
    # 而那正是服務啟動時會做的判斷。匯出當下就該過。
    reloaded = load_artifact(directory, strict=True)
    got = np.asarray(reloaded.fitted.predict(X), dtype=np.float64)
    gap = float(np.max(np.abs(got - expected)))
    if gap > ROUND_TRIP_ATOL:
        worst = int(np.argmax(np.abs(got - expected)))
        raise AssertionError(
            f"存檔／載入不是無損的：第 {worst} 列 {expected[worst]:.10f} → {got[worst]:.10f}"
            f"（最大差 {gap:.3e}，容差 {ROUND_TRIP_ATOL}）。\n"
            "  服務會用一個與離線不同的模型算分，而兩邊都不會報錯。"
        )
    return gap


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 匯出模型 artifact")
    ap.add_argument(
        "--design",
        default="lead0",
        choices=sorted(DESIGNS),
        help=(
            "評分規則。lead0 = 到期日當天（離線基準，不可上線）、"
            "lead7 = 提前 7 天、fixed = 固定評分日（提前 1~30 天，Kaggle 管線用）"
        ),
    )
    ap.add_argument("--name", default=None, help="artifact 目錄名（預設 <模型>_lead<N>d）")
    ap.add_argument("--out", default=None, help="輸出目錄（預設 <data_root>/artifacts/<name>）")
    ap.add_argument(
        "--reuse",
        action="store_true",
        help="不重訓，載回既有 artifact 的模型只重算業務假設與分數",
    )
    args = ap.parse_args()

    sha, dirty = git_state()
    try:
        paths = load_paths().ensure()
        biz_cfg = load_business_config()
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這份 artifact 無法用 SHA 回溯。")

    train_spec, valid_spec = DESIGNS[args.design]
    name = args.name or f"{ADOPTED_MODEL}_{ARTIFACT_SUFFIX[args.design]}"
    out_dir = paths.artifacts / name if args.out is None else Path(args.out)

    lo, hi = cutoff_window(train_spec)
    print(
        f"\n{'=' * 88}\n匯出 {name}：{train_spec.name} → {valid_spec.name}"
        f"（{cutoff_definition(train_spec)}，cutoff 落在 {lo}~{hi}）\n{'=' * 88}"
    )

    params, train_cfg = load_adopted_config()
    train, valid = load_cohort_features(
        paths, biz_cfg, train_spec=train_spec, valid_spec=valid_spec
    )

    if args.reuse:
        print(f"\n--reuse：載回 {out_dir} 的模型，不重訓 ...", flush=True)
        existing = load_artifact(out_dir, strict=True)
        fitted = existing.fitted
        if existing.feature_names != list(train.X.columns):
            sys.exit(
                "既有 artifact 的特徵清單與現在建出來的不一致，--reuse 不適用（請重新訓練匯出）。"
            )
    else:
        tr, es = split_for_early_stopping(train, train_cfg)
        print(
            f"\n訓練中（{ADOPTED_MODEL}，§7.12 正式採用的模型；"
            f"train {tr.X.height:,} · early stopping {es.X.height:,}）...",
            flush=True,
        )
        fitted = fit_adopted(tr, es, all_train=train)
        print(f"  停在第 {fitted.best_iteration} 輪")

    pred = np.asarray(fitted.predict(valid.X), dtype=np.float64)
    ll = float(log_loss(valid.y, pred))
    # 常數基準用**訓練 cohort 的流失率**，不是驗證 cohort 的 —— 後者是標籤。
    baseline = float(constant_log_loss(float(train.y.mean()), valid.y))
    print(f"  {valid_spec.name} log loss {ll:.5f}（常數基準 {baseline:.5f}）")

    assumptions = resolve_assumptions(
        biz_cfg["business"],
        price_per_day=valid.X["price_per_day"],
        prior_churn_rate=float(train.y.mean()),
        months_source=f"{train_spec.name}_churn_rate",
    )
    print(
        f"  p* = {assumptions.c_offer:.0f} / ({assumptions.r_save:.2f}"
        f" × {assumptions.ltv_saved:.0f}) = {assumptions.p_star:.4f}"
        f"　→ 名單 {(pred > assumptions.p_star).mean():.2%} 的人"
    )

    provenance = cache_fingerprints(paths, (train_spec, valid_spec))
    if provenance["stale"]:
        print(f"⚠️ 這些快取的程式版本指紋與現行程式不符：{provenance['stale']}")

    meta = {
        "cohort": {
            "train": train_spec.name,
            "eval": valid_spec.name,
            "design": args.design,
            "lead_days": train_spec.lead_days,
            "score_date": train_spec.score_date,
            # 必填。少了它，一個到期日當天評分的模型可以被當成能上線的模型
            # 部署出去（見 src/serving/artifact.py）。
            "cutoff_definition": cutoff_definition(train_spec),
            "cutoff_window": [lo, hi],
            "observation": valid_spec.observation,
            "n_train": int(train.X.height),
            "n_eval": int(valid.X.height),
            "train_churn_rate": round(float(train.y.mean()), 6),
            "eval_churn_rate": round(float(valid.y.mean()), 6),
        },
        # 固定評分日的**實際**提前天數（1~30 天不等）。這是那個設計的關鍵性質，
        # 不記下來就無法回答「這個模型平均提前多久評分」。
        "effective_lead_days": (
            {
                "min": int(valid.X["days_to_expire"].min()),
                "median": float(valid.X["days_to_expire"].median()),
                "max": int(valid.X["days_to_expire"].max()),
                "negative": int((valid.X["days_to_expire"] < 0).sum()),
            }
            if "days_to_expire" in valid.X.columns
            else {
                "min": train_spec.lead_days,
                "median": train_spec.lead_days,
                "max": train_spec.lead_days,
                "negative": 0,
            }
        ),
        "metrics": {
            "log_loss": round(ll, 5),
            "constant_baseline": round(baseline, 5),
            "improvement": round(1 - ll / baseline, 4),
            # ⚠️ 這是**這個 cohort 自己的**分數。T=0 與 T−7 的 cohort 成員不同，
            # 兩份 artifact 的這一欄不可直接比大小 —— 共同子集的比較在
            # reports/lead_time.json（§7.15）。
            "note": "同一 artifact 內部可比；跨 lead_days 的比較見 reports/lead_time.json",
        },
        "assumptions": assumptions.summary(),
        "hyperparameters": params,
        "training": train_cfg,
        "configs": ["configs/model_comparison.yaml", "configs/business.yaml"],
        "git": {"sha": sha, "dirty": dirty, "reused_model": bool(args.reuse)},
        "fingerprints": provenance,
        "limits": [
            "模型預測的是「會不會流失」，不是「投放優惠能不能改變他的行為」——"
            "後者需要 uplift modeling 與 A/B 實驗，本資料集沒有實驗組／對照組結構。",
            "機率未校準（§7.10：校準器讓每一項指標都變差，成因是 cohort 間的"
            "基準率漂移，校準器看不到它）。整體偏低估，用在金額上要記得這件事。",
            "members_v3.csv 是 2017-11-13 的快照，不是 as-of cutoff 的狀態 ——"
            "本專案唯一已知且無法修復的洩漏，影響上界 0.93% 的 gain。",
            f"cutoff_definition = {cutoff_definition(train_spec)}。"
            + (
                "到期日當天評分，挽回優惠來不及寄出，**不可上線**（§4.3）。"
                if args.design == "lead0"
                else (
                    "提前 7 天評分，這是能上線的版本；代價見 §7.15。"
                    if args.design == "lead7"
                    else "固定評分日：提前天數隨到期日變動（見 metadata 的"
                    " effective_lead_days），這是 Kaggle 測試集唯一做得到的設計。"
                )
            ),
        ],
    }

    full = save_artifact(
        fitted,
        out_dir,
        feature_names=list(train.X.columns),
        categorical=train.categorical,
        meta=meta,
    )
    print(f"\n  artifact 已存 → {out_dir}")
    print(f"    模型 {full['model']['file']}（sha256 {full['model']['sha256'][:16]}…）")
    print(f"    metadata artifact.json（{full['features']['n']} 特徵）")

    # ---- 存了就載回來對答案 ----
    n = min(ROUND_TRIP_ROWS, valid.X.height)
    print(f"\n載回來對答案（前 {n:,} 列）...", flush=True)
    try:
        gap = verify_round_trip(out_dir, fitted, valid.X.head(n), pred[:n])
    except AssertionError as e:
        print(f"\n❌ {e}")
        return 1
    print(f"  最大逐列差 {gap:.1e} —— 存檔／載入無損 ✅")

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    print(
        f"  · 這份 artifact 的 cutoff 定義是 {meta['cohort']['cutoff_definition']}，"
        f"{'**不可上線**（§4.3）' if args.design == 'lead0' else '可上線'}。\n"
        "  · 機率未校準，p* 由 configs/business.yaml 推導並存進 artifact ——"
        "服務端只比較，不重算（一列資料算不出 LTV）。\n"
        f"  · 起服務：make serve（載哪一份看 configs/serving.yaml，"
        f"現在指向 {serving_artifact_name()}）"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
