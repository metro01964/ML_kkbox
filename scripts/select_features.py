"""M3 · Null importance 特徵篩選的入口（SPEC §7 M3「篩選前後對照」）。

方法與三個容易做錯的地方寫在 src/models/selection.py 的模組註解裡。

流程：

    1. 全特徵在 Feb-train 上訓練一次 → Feb-sel 基準分數 + best_iteration
    2. 用 best_iteration 當固定預算，在 **Feb-train** 上跑 null importance
    3. 依不同門檻篩出特徵子集，每個子集在 **Feb-sel** 上評分
    4. 依 Feb-sel 凍結門檻
    5. Mar 只在最後登場，只算「篩選前」與「篩選後」兩個數字

    uv run python scripts/select_features.py
    make select

⚠️ **門檻在 Feb 內部選，Mar 只看兩次。**

早期版本讓每個門檻都在 Mar 上評分再挑最低的，那是 selection bias：Mar 同時
當了選模集與最終成績單，回報的數字就不是「這個模型有多好」，而是「六個門檻
裡最合這批資料胃口的那個有多好」。

現在 Feb 切成 train / es / sel 三塊，門檻由 sel 決定；凍結之後才在 Mar 上算
「篩選前」與「篩選後」—— 這兩個是 SPEC §7 指定的交付物，在看到分數之前就
決定要算哪兩個，因此不是挑選。

整條 sel 掃描曲線照樣印出來，讓「這些門檻其實差不多」這件事被看見。
"""

from __future__ import annotations

import json
import sys
import time

import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.models.selection import (
    before_after_table,
    evaluate_subset,
    null_importance,
    score_on_mar,
    threshold_sweep_table,
)
from src.models.train import load_cohort_features
from src.models.tuning import three_way_split

CONFIG = REPO_ROOT / "configs" / "feature_selection.yaml"


def load_config() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"找不到設定檔 {CONFIG}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    for section in ("model", "training", "split", "null_importance"):
        if section not in cfg:
            raise KeyError(f"{CONFIG} 缺少 [{section}] 區段")
    return cfg


def log_to_mlflow(cfg: dict, runs: list, chosen: str, n_kept: int, n_total: int) -> str | None:
    """把篩選結果記成一個 run。

    `sel_logloss_*` 是每個門檻都有的（選門檻的依據），`mar_logloss_*` 只有
    凍結下來的那兩個有 —— 這個不對稱是刻意的，讓 MLflow 上也看得出來
    「哪些分數參與了決定」。
    """
    import mlflow

    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return None

    backend = tracking.get("backend", "sqlite:///mlflow.db")
    if backend.startswith("sqlite:///") and not backend.startswith("sqlite:////"):
        backend = f"sqlite:///{(REPO_ROOT / backend.removeprefix('sqlite:///')).as_posix()}"
    mlflow.set_tracking_uri(backend)

    exp_name = tracking.get("experiment", "kkbox-churn")
    if mlflow.get_experiment_by_name(exp_name) is None:
        artifact_dir = REPO_ROOT / tracking.get("artifact_dir", "mlartifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f"{tracking.get('run_prefix', 'm3_select')}_{chosen}") as run:
        mlflow.log_params(
            {
                "null_runs": cfg["null_importance"]["n_runs"],
                "chosen_threshold": chosen,
                "chosen_by": "feb_sel_logloss",
                "n_features_kept": n_kept,
                "n_features_total": n_total,
            }
        )
        for r in runs:
            mlflow.log_metric(f"sel_logloss_{r.key}", r.sel_logloss)
            mlflow.log_metric(f"nfeat_{r.key}", r.n_features)
            if r.mar_logloss is not None:
                mlflow.log_metric(f"mar_logloss_{r.key}", r.mar_logloss)
        return run.info.run_id


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    params = dict(cfg["model"])
    train_cfg = cfg["training"]
    ni_cfg = cfg["null_importance"]

    feb, mar = load_cohort_features(paths, cfg)
    all_features = feb.X.columns

    # Feb 切成三塊：train 訓練、es 決定輪數、sel 決定門檻。Mar 全程不參與。
    split = three_way_split(feb, cfg["split"])
    print(
        f"  Feb 內部三段切分：訓練 {split.train.X.height:,}"
        f" · early stopping {split.es.X.height:,}"
        f" · 選門檻 {split.sel.X.height:,}\n"
    )

    # --- 1. 全特徵基準（同時取得 null importance 要用的固定輪數）-------------
    print("全特徵基準訓練中...", flush=True)
    t0 = time.perf_counter()
    base = evaluate_subset(
        split.train,
        split.es,
        split.sel,
        all_features,
        params,
        train_cfg,
        label="全特徵（未篩選）",
        key="all",
        all_features=all_features,
    )
    print(
        f"  → Feb-sel log loss {base.sel_logloss:.5f}　{base.n_features} 特徵　"
        f"停在第 {base.best_iteration} 輪　({time.perf_counter() - t0:.0f} 秒)"
    )

    # --- 2. Null importance（只在 train 上做，連 sel 都不看）-----------------
    rounds = ni_cfg.get("num_boost_round") or base.best_iteration
    n_runs = ni_cfg["n_runs"]
    print(
        f"\nNull importance：真實 1 次 + 打亂標籤 {n_runs} 次，"
        f"每次固定 {rounds} 輪（只用 Feb-train {split.train.X.height:,} 列）"
    )
    ni = null_importance(
        split.train,
        params,
        num_boost_round=rounds,
        n_runs=n_runs,
        seed=ni_cfg["seed"],
        verbose=True,
    )
    summary = ni.summary()

    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(170)
    print("\n" + "=" * 78)
    print("Null importance 摘要（依 score 遞減）")
    print("=" * 78)
    print(
        summary.select(
            "feature",
            pl.col("actual_gain").round(1),
            pl.col("null_p75").round(1),
            pl.col("null_max").round(1),
            pl.col("beats_pct").round(3),
            pl.col("score").round(3),
        )
    )
    never = summary.filter(pl.col("actual_gain") == 0)["feature"].to_list()
    if never:
        print(f"\n真實訓練中完全沒被用到的特徵 {len(never)} 個：{never}")

    # --- 3. 各門檻的掃描（分數來自 Feb-sel，Mar 尚未登場）--------------------
    runs = [base]
    kept_by_key: dict[str, list[str]] = {"all": all_features}
    for p in ni_cfg["percentiles"]:
        keep = ni.keep(p)
        label, key = f"> null p{p}", f"p{p}"
        if not keep:
            print(f"\n[{label}] 篩完沒有特徵留下，跳過。")
            continue
        kept_by_key[key] = keep
        print(f"\n[{label}] 保留 {len(keep)}/{len(all_features)} 個特徵，重訓中...", flush=True)
        r = evaluate_subset(
            split.train,
            split.es,
            split.sel,
            keep,
            params,
            train_cfg,
            label=label,
            key=key,
            all_features=all_features,
        )
        runs.append(r)
        print(f"  → Feb-sel log loss {r.sel_logloss:.5f}　停在第 {r.best_iteration} 輪")

    print("\n" + "=" * 78)
    print("門檻掃描（分數全部來自 Feb-sel，Mar 尚未參與）")
    print("=" * 78)
    print(threshold_sweep_table(runs))

    # --- 4. 凍結門檻 ---------------------------------------------------------
    # 依 Feb-sel 分數選，同分（差距 < 0.0001，遠小於 M1 實測的 5-fold 標準差
    # 0.00049）時選特徵少的那個：特徵少的模型好維護、推論快，M6 要上線的
    # 東西越簡單越好。
    winner = min(runs, key=lambda r: (round(r.sel_logloss, 4), r.n_features))
    print(f"\n凍結門檻：{winner.label}　{winner.n_features} 特徵")
    print(f"  依據：Feb-sel {winner.sel_logloss:.5f}（全特徵 {base.sel_logloss:.5f}）")
    if winner.dropped:
        print(f"  砍掉 {len(winner.dropped)} 個：{winner.dropped}")

    # --- 5. Mar 只在這裡登場，且只算「篩選前 / 篩選後」兩個數字 --------------
    print("\n在 Mar cohort 上評估（門檻已凍結，只算篩選前與篩選後）...")
    score_on_mar(base, split.train, split.es, mar, feb, all_features, params, train_cfg)
    if winner is not base:
        score_on_mar(
            winner,
            split.train,
            split.es,
            mar,
            feb,
            kept_by_key[winner.key],
            params,
            train_cfg,
        )

    print("\n" + "=" * 78)
    print("篩選前後對照（SPEC §7 M3 交付物）")
    print("=" * 78)
    print(before_after_table(runs))

    # --- 6. 存檔，供 M4 之後直接沿用 ------------------------------------------
    out = paths.interim / "m3_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    summary.write_csv(out / "null_importance.csv")
    ni.null_gains.write_csv(out / "null_importance_runs.csv")
    (out / "selected_features.json").write_text(
        json.dumps(
            {
                "threshold": winner.label,
                "chosen_by": "Feb-sel log loss（Mar 未參與選擇）",
                "n_features": winner.n_features,
                "features": kept_by_key[winner.key],
                "dropped": winner.dropped,
                "sel_logloss": winner.sel_logloss,
                "mar_logloss": winner.mar_logloss,
                "null_runs": n_runs,
                "null_num_boost_round": rounds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n已寫入 {out}")

    run_id = log_to_mlflow(cfg, runs, winner.key, winner.n_features, len(all_features))
    if run_id:
        print(f"MLflow run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
