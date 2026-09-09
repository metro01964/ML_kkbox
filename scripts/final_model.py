"""M6 · 用全部帶標籤的資料重訓最終模型 —— 紅線 4 的觸發點（SPEC §7.4）。

    uv run python scripts/final_model.py             # 完整
    uv run python scripts/final_model.py --no-cv     # 跳過 CV（含違規對照組）
    make final

## 這支腳本存在的理由

到目前為止的三份 artifact（`catboost_lead0d` / `lead7d` / `fixed`）都只用
**單一 cohort** 訓練。上線前的慣例是把所有帶標籤的資料都用上：多一個 cohort
就是多 97 萬列，而 `mar_fixed` 在時間上離要預測的 Apr 更近。

SPEC §7.4 把紅線 4 的阻塞里程碑判在這一刻，理由就是這件事 —— **合併的那一秒，
90.81% 的重疊從「不是洩漏」變成「是洩漏」**：同一個人的 Feb 列進訓練、Mar 列
進驗證，而他的城市、註冊管道、慣用付款方式、年資在相隔一個月時幾乎不變。

## ⚠️ 合併之後就沒有時間外驗證集了

本專案只有兩個帶標籤的 cohort。兩個都拿去訓練，能報的分數就只剩合併資料
內部的 holdout 與 CV —— 那是**同分布**的估計。

    這份 artifact 的分數 **不可** 與 catboost_fixed 的 0.17342 比大小。

前者是「同一批人、同一種分布」，後者是「訓練 Feb、驗證 Mar」的時間外估計。
要回答「這個模型有多好」，該引用的仍然是後者；這份重訓買到的是**更多資料與
更近的時間**，不是一個更好的分數。真正的外部驗證只有一個：把 Apr cohort 的
預測交上 Kaggle（`scripts/predict_kaggle.py --artifact catboost_fixed_full`）。

## 四段切分，每一段一個決定（§7.6）

    train  模型權重
    es     停在第幾輪
    sel    要不要採用校準器
    cal    fit 校準器

§7.6 記錄的殘餘風險是「同一塊資料被兩個決定看過」。四段切分是那一節留下的
處置，M6 是它說的「重訓最終模型時」。

## 違規對照組

同一份合併資料、同樣的折數、同一組超參數，只把 `StratifiedGroupKFold` 換成
`StratifiedKFold`，兩邊 CV 分數的差就是紅線 4 擋掉的東西。

違規那一臂**不是特別寫的壞程式碼**：它用的是 `split_for_early_stopping()`，
M1 到 M3 一直在用、在單一 cohort 上完全正確的那個函式。這正是這條紅線的性質
—— 犯法的寫法與守法的寫法長得一模一樣，差別只在資料被合併過。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
import yaml

from scripts.export_model import (
    ROUND_TRIP_ROWS,
    cache_fingerprints,
    git_state,
    load_business_config,
    verify_round_trip,
)
from src.config import REPO_ROOT, load_paths
from src.data import COHORTS, CohortSpec, cutoff_definition
from src.evaluation import constant_log_loss, log_loss, resolve_assumptions
from src.evaluation.calibrator import fit_isotonic
from src.models.adopted import ADOPTED_MODEL, fit_adopted, load_adopted_config
from src.models.compare import split_for_early_stopping
from src.models.grouped import (
    SEGMENTS,
    MergedCohorts,
    fold_report,
    four_way_group_split,
    group_split,
    grouped_folds,
    merge_cohorts,
    random_folds_violating_red_line_4,
)
from src.models.train import load_cohort_features
from src.serving.artifact import save_artifact

DEFAULT_CONFIG = REPO_ROOT / "configs" / "final_model.yaml"
OUT_PATH = REPO_ROOT / "reports" / "final_model.json"

# 校準器的採用條件，**先寫下來再看數字**。
#
# §7.10 判定校準器不上線，成因是 cohort 之間的基準率漂移。合併重訓把那個漂移
# 吃進了訓練集，所以結論有可能反轉 —— 而「有可能反轉」正是事後才訂門檻最危險
# 的情境：看到 0.3% 的改善再決定「這樣算不算有幫助」，等於用結果決定標準。
CALIBRATION_ADOPT_THRESHOLD = 0.0  # sel 的 log loss 必須嚴格變好才算有幫助

# 兩臂的差要多少倍折間標準差才值得當成「效應」。
#
# 低於這個倍數時，正確的結論是「這個實驗量不到」，不是「等於 0」—— §7.12 的
# 配對 multi-seed 與 §7.17 的 PSI 雜訊地板都是同一件事：一個沒有比較基準的
# 差值，讀不出任何東西。
SIGMA_RESOLUTION = 1.0


def load_config(path=None) -> dict[str, Any]:
    """讀 configs/final_model.yaml，缺區段就直接失敗（同 `load_model_config`）。"""
    path = path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"找不到設定檔 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("cohorts", "split", "cv", "artifact"):
        if section not in cfg:
            raise KeyError(f"{path} 缺少 [{section}] 區段")
    return cfg


def load_parts(paths, cfg: dict[str, Any]) -> tuple[dict[str, Any], tuple[CohortSpec, ...]]:
    """把設定檔列的 cohort 各建一份特徵矩陣。

    **只支援剛好兩個 cohort**，而且刻意走 `load_cohort_features()` 那一條路 ——
    它同時載兩個 cohort、接同一份收聽特徵、並比對兩邊的欄位一致。自己另寫一個
    載入迴圈會讓「合併的兩份特徵完全同源」變成口頭承諾（§7.4 記錄過這個教訓：
    三個套件各自載一次資料，特徵集相同就只是宣稱）。

    本專案帶標籤的 cohort 只有兩個，所以這個限制目前不損失任何東西；真的要併
    第三個時，該改的是 `load_cohort_features()`，不是在這裡複製一份載入邏輯。
    """
    names = list(cfg["cohorts"])
    if len(names) != 2:
        raise ValueError(f"目前只支援合併兩個 cohort，設定檔列了 {len(names)} 個：{names}")
    unknown = [n for n in names if n not in COHORTS]
    if unknown:
        raise ValueError(f"不認識的 cohort：{unknown}")

    specs = tuple(COHORTS[n] for n in names)
    a, b = load_cohort_features(paths, cfg, train_spec=specs[0], valid_spec=specs[1])
    return {names[0]: a, names[1]: b}, specs


def fit_one_fold(
    merged: MergedCohorts,
    tr: np.ndarray,
    va: np.ndarray,
    train_cfg: dict,
    *,
    grouped: bool,
) -> tuple[float, int, float]:
    """一折：切早停集 → 訓練 → 在驗證折上算分。

    Returns:
        (log loss, 停在第幾輪, 花了幾秒)

    早停集的切法跟著那一臂走。合規的一臂用 `group_split()`，違規的一臂用
    `split_for_early_stopping()` —— 後者是 M1 至今一直在用的那個函式，在單一
    cohort 上完全正確。**違規不是寫了壞程式碼，是把對的程式碼用在合併資料上。**
    """
    fold_train, fold_valid = merged.fs.take(tr), merged.fs.take(va)
    if grouped:
        inner, es_idx = group_split(
            fold_train.msno,
            fold_train.y,
            test_size=train_cfg["inner_valid_fraction"],
            seed=train_cfg["inner_split_seed"],
        )
        tr_fs, es_fs = fold_train.take(inner), fold_train.take(es_idx)
    else:
        tr_fs, es_fs = split_for_early_stopping(fold_train, train_cfg)

    t0 = time.perf_counter()
    fitted = fit_adopted(tr_fs, es_fs, all_train=fold_train)
    pred = np.asarray(fitted.predict(fold_valid.X), dtype=np.float64)
    return float(log_loss(fold_valid.y, pred)), int(fitted.best_iteration), time.perf_counter() - t0


def run_arm(
    merged: MergedCohorts,
    folds: list[tuple[np.ndarray, np.ndarray]],
    train_cfg: dict,
    *,
    grouped: bool,
    label: str,
) -> dict[str, Any]:
    """跑完一臂的所有折，回傳分數與每折的細節。"""
    print(f"\n--- {label} ---", flush=True)
    print(fold_report(merged, folds))

    scores, iters = [], []
    for i, (tr, va) in enumerate(folds, 1):
        score, best_iter, secs = fit_one_fold(merged, tr, va, train_cfg, grouped=grouped)
        scores.append(score)
        iters.append(best_iter)
        print(
            f"  fold {i}/{len(folds)}　log loss {score:.5f}　{best_iter} 輪　({secs / 60:.1f} 分)",
            flush=True,
        )

    crossing = int(fold_report(merged, folds)["跨邊人數"].sum())
    return {
        "label": label,
        "grouped": grouped,
        "fold_scores": [round(s, 5) for s in scores],
        "best_iterations": iters,
        "mean": float(np.mean(scores)),
        # ddof=1：這幾折是母體的樣本，不是母體本身（同 CVResult.std）。
        "std": float(np.std(scores, ddof=1)),
        "crossing_users_total": crossing,
    }


def calibration_decision(fitted, split, *, threshold: float) -> dict[str, Any]:
    """在 cal 上 fit 校準器，在 sel 上判斷它有沒有幫助。

    兩段分開是重點：fit 與「決定要不要用」若看同一塊資料，校準器必然看起來
    有幫助 —— isotonic 在自己 fit 的那一塊上幾乎不可能變差。
    """
    cal_pred = np.asarray(fitted.predict(split.cal.X), dtype=np.float64)
    sel_pred = np.asarray(fitted.predict(split.sel.X), dtype=np.float64)

    calibrator = fit_isotonic(split.cal.y, cal_pred)
    sel_calibrated = np.asarray(calibrator.apply(sel_pred), dtype=np.float64)

    raw_ll = float(log_loss(split.sel.y, sel_pred))
    cal_ll = float(log_loss(split.sel.y, sel_calibrated))
    return {
        "fitted_on": "cal",
        "judged_on": "sel",
        "sel_logloss_raw": round(raw_ll, 5),
        "sel_logloss_calibrated": round(cal_ll, 5),
        "delta": round(cal_ll - raw_ll, 5),
        "sel_actual_rate": round(float(split.sel.y.mean()), 5),
        "sel_mean_pred_raw": round(float(sel_pred.mean()), 5),
        "sel_mean_pred_calibrated": round(float(sel_calibrated.mean()), 5),
        "adopted": bool(raw_ll - cal_ll > threshold),
        "threshold": threshold,
    }


def red_line_4_gap(arms: list[dict[str, Any]]) -> dict[str, Any]:
    """兩臂的差，以及**這個實驗量得到多小的差**。

    只印差值是不夠的：4 折的折間標準差本身就有 0.0006 的量級，一個 0.0001 的
    差讀不出任何東西。所以一併回報差值等於幾倍折間標準差 —— 同 §7.12 用 σ
    尺度讀 multi-seed、§7.17 用雜訊地板讀 PSI 的理由。
    """
    ok, bad = arms[0], arms[1]
    absolute = ok["mean"] - bad["mean"]
    pooled = float(np.sqrt((ok["std"] ** 2 + bad["std"] ** 2) / 2))
    return {
        "grouped_mean": round(ok["mean"], 6),
        "random_mean": round(bad["mean"], 6),
        "absolute": round(absolute, 6),
        "relative": round(ok["mean"] / bad["mean"] - 1, 5),
        "pooled_fold_std": round(pooled, 6),
        "in_fold_sigma": round(absolute / pooled, 3) if pooled else None,
        "below_resolution": bool(pooled and abs(absolute / pooled) < SIGMA_RESOLUTION),
        "sigma_resolution": SIGMA_RESOLUTION,
        "crossing_users_in_random_arm": bad["crossing_users_total"],
    }


def out_of_time_reference(paths, name: str = "catboost_fixed") -> dict[str, Any] | None:
    """既有單一 cohort artifact 的**時間外**分數，供對照。

    這是這份報告裡唯一一個時間外數字。合併重訓之後本地再也算不出新的 ——
    兩個帶標籤的 cohort 都進訓練集了。把它讀進來（而不是抄一個常數）是為了
    讓「哪個數字是時間外的」有出處可查。

    找不到就回 None：這份 artifact 是上一步的產物，不是本腳本的前置條件。
    """
    path = paths.artifacts / name / "artifact.json"
    if not path.exists():
        return None
    meta = json.loads(path.read_text(encoding="utf-8"))
    return {
        "artifact": name,
        "train": meta["cohort"]["train"],
        "eval": meta["cohort"]["eval"],
        "log_loss": meta["metrics"]["log_loss"],
        "constant_baseline": meta["metrics"]["constant_baseline"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="M6 合併 cohort 重訓最終模型（紅線 4）")
    ap.add_argument("--no-cv", action="store_true", help="跳過 CV 與違規對照組（省一半時間）")
    ap.add_argument("--name", default=None, help="artifact 目錄名（預設讀設定檔）")
    args = ap.parse_args()

    sha, dirty = git_state()
    try:
        cfg = load_config()
        paths = load_paths().ensure()
        biz_cfg = load_business_config()
        params, train_cfg = load_adopted_config()
    except (FileNotFoundError, KeyError, ValueError) as e:
        sys.exit(str(e))

    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這份 artifact 無法用 SHA 回溯。")

    name = args.name or str(cfg["artifact"])
    out_dir = paths.artifacts / name
    print(f"\n{'=' * 88}\n最終模型 {name}：合併 {' + '.join(cfg['cohorts'])} 重訓")
    print(f"（{ADOPTED_MODEL}，§7.12 正式採用的模型）\n{'=' * 88}")

    # ---- 1. 載入並合併 ----
    parts, specs = load_parts(paths, cfg)
    merged = merge_cohorts(parts)
    print("\n合併後：")
    print(merged.summary())
    print(
        f"  {merged.n_rows:,} 列 · {merged.n_groups:,} 人 · "
        f"跨期出現 {merged.n_shared:,} 人（佔 {merged.n_shared / merged.n_groups:.2%}）"
        f" · 流失率 {float(merged.fs.y.mean()):.4%}"
    )
    print(f"  ↑ 這 {merged.n_shared:,} 人就是紅線 4 的曝險面：隨機切分會讓他們同時進訓練與驗證。")

    # ---- 2. 四段切分 ----
    split = four_way_group_split(merged, cfg["split"])
    print("\n四段切分（每一段對應一個決定，四段都綁 msno）：")
    print(split.summary())

    # ---- 3. 訓練最終模型 ----
    print(
        f"\n訓練中（train {split.train.X.height:,} 列"
        f" · early stopping {split.es.X.height:,} 列）...",
        flush=True,
    )
    t0 = time.perf_counter()
    fitted = fit_adopted(split.train, split.es, all_train=merged.fs)
    train_secs = time.perf_counter() - t0
    print(f"  停在第 {fitted.best_iteration} 輪（{train_secs / 60:.1f} 分）")

    sel_pred = np.asarray(fitted.predict(split.sel.X), dtype=np.float64)
    sel_ll = float(log_loss(split.sel.y, sel_pred))
    sel_baseline = float(constant_log_loss(float(split.train.y.mean()), split.sel.y))
    print(f"  sel log loss {sel_ll:.5f}（常數基準 {sel_baseline:.5f}）")
    print("  ⚠️ 這是**同分布**的 holdout，不是時間外估計 —— 不可與 catboost_fixed 的分數比大小。")

    # ---- 4. 校準器：在 cal 上 fit，在 sel 上判斷 ----
    print("\n校準器（cal 上 fit → sel 上判斷）...", flush=True)
    calib = calibration_decision(fitted, split, threshold=CALIBRATION_ADOPT_THRESHOLD)
    print(
        f"  sel log loss　未校準 {calib['sel_logloss_raw']:.5f}"
        f" → 校準後 {calib['sel_logloss_calibrated']:.5f}（{calib['delta']:+.5f}）"
    )
    print(
        f"  sel 平均預測　未校準 {calib['sel_mean_pred_raw']:.4%}"
        f" → 校準後 {calib['sel_mean_pred_calibrated']:.4%}"
        f"　實際 {calib['sel_actual_rate']:.4%}"
    )
    if calib["adopted"]:
        print(
            "  ⚠️ 校準器在這份合併模型上**有幫助**，與 §7.10 的結論相反。\n"
            "     但這份 artifact 仍然存成未校準（save_artifact 硬寫 calibrated=False）——\n"
            "     要改成上線校準版，得同時改 artifact 與服務端，那是另一個決定。"
        )
    else:
        print("  → 不採用（與 §7.10 同向）。artifact 存的是未校準的機率。")

    # ---- 5. CV：合規 vs 違規 ----
    cv_cfg = cfg["cv"]
    arms: list[dict[str, Any]] = []
    if not args.no_cv:
        n_splits, cv_seed = int(cv_cfg["n_splits"]), int(cv_cfg["seed"])
        print(f"\n{'=' * 88}")
        print(f"{n_splits}-fold CV：合規（GroupKFold）vs 違規（隨機切分）")
        print("=" * 88)
        arms.append(
            run_arm(
                merged,
                grouped_folds(merged, n_splits=n_splits, seed=cv_seed),
                train_cfg,
                grouped=True,
                label=f"合規：StratifiedGroupKFold(groups=msno)　{n_splits} 折",
            )
        )
        if cv_cfg.get("violating_control", False):
            arms.append(
                run_arm(
                    merged,
                    random_folds_violating_red_line_4(merged, n_splits=n_splits, seed=cv_seed),
                    train_cfg,
                    grouped=False,
                    label=f"違規：StratifiedKFold（不看 msno）　{n_splits} 折",
                )
            )

    gap = red_line_4_gap(arms) if len(arms) == 2 else None
    if gap is not None:
        ok, bad = arms[0], arms[1]
        print(f"\n{'=' * 88}\n紅線 4 擋掉了多少\n{'=' * 88}")
        print(f"  合規（GroupKFold）    {ok['mean']:.5f} ± {ok['std']:.5f}")
        print(f"  違規（隨機切分）      {bad['mean']:.5f} ± {bad['std']:.5f}")
        per_fold = bad["crossing_users_total"] // max(len(bad["fold_scores"]), 1)
        print(f"  差　　　　　　　　    {gap['absolute']:+.6f}（{gap['relative']:+.3%}）")
        print(
            f"  折間標準差 {gap['pooled_fold_std']:.6f}"
            f" —— 這個差是它的 {abs(gap['in_fold_sigma']):.3f} 倍"
        )
        print(f"  違規那一臂每折平均有 {per_fold:,} 位驗證集用戶，模型在訓練時已經見過。")
        if abs(gap["in_fold_sigma"]) < SIGMA_RESOLUTION:
            print(
                "\n  ⚠️ **差距小於這個實驗的解析度。** 只能說「量不到」，不能說「等於 0」——\n"
                "     4 折給出的標準差就是這麼大。要量到更小的效應需要更多折或多 seed。"
            )

    # 唯一的時間外對照。合併之後本地再也算不出新的一個 —— 這件事本身
    # 比兩臂的差更值得看，所以印在同一個區塊裡。
    oot = out_of_time_reference(paths)
    if oot is not None:
        local = float(np.mean([a["mean"] for a in arms])) if arms else sel_ll
        print(f"\n  時間外對照（{oot['artifact']}：{oot['train']} → {oot['eval']}）")
        print(
            f"    時間外 {oot['log_loss']:.5f}　vs　本地同分布 {local:.5f}"
            f"（樂觀 {1 - local / oot['log_loss']:+.2%}）"
        )
        print("    ↑ 這個差與切分規則無關，是「合併之後沒有時間外驗證集」的代價。")

    # ---- 6. 匯出 artifact ----
    #
    # ⚠️ **LTV 用最近一期的流失率，不是合併後的平均**（SPEC §9 第 4 項，2026-08-12 定案）。
    #
    # `expected_months = 1 / 月流失率`，所以這個選擇直接決定 LTV，再決定 `p*`，
    # 最後決定投放名單多大 —— 而**理由與模型的排序能力完全無關**，它是一個業務假設。
    #
    # 原本傳 `merged.fs.y.mean()`，也就是 feb+mar 混合的 6.7682%。改用 mar_fixed
    # 單期的 7.6703%，因為兩期的流失率是**上升的**（Feb 6.39% → Mar 8.99%）：
    # 拿混合值當未來的預期，等於低估流失、高估 LTV、把門檻設得太低 ——
    # 也就是**投放給太多不該投放的人**。
    #
    # 這不違反 `resolve_assumptions` 的紅線（不可傳評估 cohort 的流失率）：
    # 合併模型把兩期都當訓練資料，mar 的標籤在訓練時就看得到，不是留出來的答案。
    # 它要服務的是 Apr，而 Apr 的流失率當然不知道 —— 最近一期的已知值就是 Mar。
    #
    # 代價要說清楚：單月比混合雜訊大。若下一期的流失率回落，這個門檻會偏保守。
    latest_name = cfg["cohorts"][-1]
    latest_churn = float(parts[latest_name].y.mean())
    print(
        f"\n  LTV 的流失率來源：{latest_name} 的 {latest_churn:.4%}"
        f"（合併平均是 {float(merged.fs.y.mean()):.4%}，刻意不用 —— 見程式碼註解）"
    )
    assumptions = resolve_assumptions(
        biz_cfg["business"],
        price_per_day=merged.fs.X["price_per_day"],
        prior_churn_rate=latest_churn,
        months_source=f"{latest_name}_churn_rate",
    )
    provenance = cache_fingerprints(paths, specs)
    if provenance["stale"]:
        print(f"\n⚠️ 這些快取的程式版本指紋與現行程式不符：{provenance['stale']}")

    definitions = [cutoff_definition(s) for s in specs]
    meta = {
        "cohort": {
            "train": "+".join(cfg["cohorts"]),
            # 沒有時間外驗證集可用 —— 這一欄講的是 sel 那一段，而它與訓練同分布。
            "eval": "merged_sel_block",
            "eval_is_out_of_time": False,
            "design": "fixed_merged",
            "lead_days": 0,
            # 兩個 cohort 的評分日不同，設計相同。字串以 `fixed_score_date` 開頭
            # 才會被 Artifact.scoring_design 認成固定評分日那個設計（Kaggle 推論
            # 腳本比的是種類不是整串，見 §7.16）。
            "cutoff_definition": "+".join(definitions),
            "cutoff_definitions": definitions,
            "observation": [s.observation for s in specs],
            "n_train": int(split.train.X.height),
            "n_merged_rows": int(merged.n_rows),
            "n_merged_users": int(merged.n_groups),
            "n_users_in_both_cohorts": int(merged.n_shared),
            "train_churn_rate": round(float(merged.fs.y.mean()), 6),
        },
        "split": {
            "kind": "four_way_group_split",
            "groups": "msno",
            "fractions": dict(cfg["split"]["fractions"]),
            "seed": int(cfg["split"]["seed"]),
            "sizes": {s: int(split.segment(s).X.height) for s in SEGMENTS},
            "decisions": {
                "train": "模型權重",
                "es": "停在第幾輪",
                "sel": "要不要採用校準器",
                "cal": "fit 校準器",
            },
        },
        "metrics": {
            "sel_log_loss": round(sel_ll, 5),
            "sel_constant_baseline": round(sel_baseline, 5),
            "sel_improvement": round(1 - sel_ll / sel_baseline, 4),
            "cv": [{k: v for k, v in arm.items() if k != "best_iterations"} for arm in arms],
            "red_line_4_gap": gap,
            # 唯一的時間外分數，來自上一份單一 cohort 的 artifact。留在 metadata
            # 裡是為了讓「這個模型有多好」在服務端也答得出來 —— 本 artifact 自己
            # 的每一個分數都是同分布的。
            "out_of_time_reference": oot,
            "note": (
                "這些分數全部算在**合併資料內部**，與訓練同分布，"
                "**不可**與 catboost_fixed 的時間外分數（feb_fixed → mar_fixed 0.17342）"
                "比大小。外部驗證只有 Kaggle late submission 一條路。"
            ),
        },
        "calibration": calib,
        "assumptions": assumptions.summary(),
        "hyperparameters": params,
        "training": train_cfg,
        "configs": [
            "configs/final_model.yaml",
            "configs/model_comparison.yaml",
            "configs/business.yaml",
        ],
        "git": {"sha": sha, "dirty": dirty, "reused_model": False},
        "fingerprints": provenance,
        "limits": [
            "**沒有時間外驗證集。** 兩個帶標籤的 cohort 都拿去訓練了，"
            "所有本地分數都是同分布估計，必然優於 §4.2 的時間外分數。",
            "模型預測的是「會不會流失」，不是「投放優惠能不能改變他的行為」——"
            "後者需要 uplift modeling 與 A/B 實驗，本資料集沒有實驗組／對照組結構。",
            "機率未校準（判斷見 metadata 的 calibration 區段，作法見 §7.10）。",
            "members_v3.csv 是 2017-11-13 的快照，不是 as-of cutoff 的狀態 ——"
            "本專案唯一已知且無法修復的洩漏，影響上界 0.93% 的 gain。",
            "固定評分日：提前天數隨到期日變動（1~30 天）。訓練資料橫跨兩個評分日"
            f"（{' 與 '.join(definitions)}），套用到 Apr 時是第三個。",
        ],
    }

    save_artifact(
        fitted,
        out_dir,
        feature_names=list(merged.fs.X.columns),
        categorical=merged.fs.categorical,
        meta=meta,
    )
    print(f"\n  artifact 已存 → {out_dir}")

    n = min(ROUND_TRIP_ROWS, split.sel.X.height)
    print(f"  載回來對答案（前 {n:,} 列）...", flush=True)
    try:
        diff = verify_round_trip(out_dir, fitted, split.sel.X.head(n), sel_pred[:n])
    except AssertionError as e:
        print(f"\n❌ {e}")
        return 1
    print(f"  最大逐列差 {diff:.1e} —— 存檔／載入無損 ✅")

    # ---- 7. 報告 ----
    report = {
        "artifact": name,
        "git_sha": sha,
        "git_dirty": dirty,
        "cohorts": list(cfg["cohorts"]),
        "merged": {
            "rows": int(merged.n_rows),
            "users": int(merged.n_groups),
            "users_in_both": int(merged.n_shared),
            "churn_rate": round(float(merged.fs.y.mean()), 6),
        },
        "split": meta["split"],
        "final_fit": {
            "best_iteration": int(fitted.best_iteration),
            "seconds": round(train_secs, 1),
            "sel_log_loss": round(sel_ll, 5),
            "sel_constant_baseline": round(sel_baseline, 5),
        },
        "calibration": calib,
        "cv_arms": arms,
        "red_line_4_gap": gap,
        # 這份報告裡唯一的時間外數字，來自上一步那份單一 cohort 的 artifact。
        "out_of_time_reference": oot,
        "how_to_read": [
            "本地所有分數都算在合併資料內部（同分布），不是時間外估計。"
            "要跟 M1–M6 的任何一個數字比大小之前，先確認那個數字算在哪個 cohort 上。",
            "red_line_4_gap 是這份報告的重點：兩臂只差一個切分規則，"
            "差多少就是「隨機切分讓模型看起來好多少」。"
            "**差值要跟 pooled_fold_std 一起讀** —— 低於一倍折間標準差時，"
            "結論是「這個實驗量不到」，不是「等於 0」。",
            "out_of_time_reference 與本地分數的差，量的是另一件事："
            "同分布估計比時間外估計樂觀多少。那個差與切分規則無關，"
            "是「合併之後失去時間外驗證集」的代價。",
            "校準器的 fit 與採用判斷分在 cal / sel 兩段 —— "
            "同一塊資料上判斷，isotonic 幾乎不可能變差，那個判斷沒有內容。",
        ],
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  摘要已存 → reports/{OUT_PATH.name}")

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    print(
        "  · 這份 artifact 用掉了全部帶標籤的資料，因此**本地沒有任何時間外分數**。\n"
        "  · 要外部驗證，只有一條路：\n"
        f"      uv run python scripts/predict_kaggle.py --artifact {name}\n"
        "    交上去的分數算在 Apr cohort 上，與 catboost_fixed 那次可直接比大小。\n"
        "  · 服務仍然指向 configs/serving.yaml 設定的 artifact —— 這支腳本不動它。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
