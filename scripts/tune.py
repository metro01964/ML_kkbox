"""M3 · LightGBM 超參數隨機搜尋的入口。

方法寫在 src/models/tuning.py 的模組註解裡。一句話版本：**搜尋全程只用 Feb
cohort，Mar 只在最後看一次。**

    uv run python scripts/tune.py
    make tune

輸出：
    - 全部 trial 依 sel 分數排序的表
    - 最佳組合在 Mar cohort 上的分數（與基準設定同條件對照）
    - 最佳參數寫到 <interim>/m3_artifacts/best_params.yaml
"""

from __future__ import annotations

import sys
import time

import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.evaluation import log_loss, repeat_vs_new, segment_report
from src.models.candidates import fit_lightgbm
from src.models.train import load_cohort_features, load_model_config
from src.models.tuning import random_search, three_way_split

CONFIG = REPO_ROOT / "configs" / "tuning.yaml"


def load_config() -> dict:
    if not CONFIG.exists():
        raise FileNotFoundError(f"找不到設定檔 {CONFIG}")
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}
    for section in ("training", "split", "search", "search_space"):
        if section not in cfg:
            raise KeyError(f"{CONFIG} 缺少 [{section}] 區段")
    return cfg


def evaluate_on_mar(params: dict, split, feb, mar, train_cfg: dict) -> tuple[float, pl.DataFrame]:
    """用選定的參數重訓一次，在 Mar cohort 上評估（SPEC §4.5 分群回報）。

    重訓用的是 train + es（不含 sel）—— 與搜尋時完全相同的訓練資料，
    這樣 Mar 分數的差異才只來自超參數。
    """
    fitted = fit_lightgbm(split.train, split.es, params, train_cfg)
    pred = fitted.predict(mar.X)
    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )
    return log_loss(mar.y, pred), segment_report(scored, segment_col="segment")


def log_to_mlflow(cfg: dict, best, mar_logloss: float, base_mar: float) -> str | None:
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

    with mlflow.start_run(run_name=f"{tracking.get('run_prefix', 'm3_tune')}_best") as run:
        mlflow.log_params({f"lgb.{k}": v for k, v in best.params.items()})
        mlflow.log_param("trial_index", best.index)
        mlflow.log_param("best_iteration", best.best_iteration)
        mlflow.log_metrics(
            {
                "sel_logloss": best.select_logloss,
                "mar_logloss": mar_logloss,
                "mar_logloss_base_params": base_mar,
                "mar_gain_vs_base": base_mar - mar_logloss,
            }
        )
        return run.info.run_id


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    # 基準參數直接讀 configs/model_lgbm.yaml —— 對照組必須就是 M1/M2 跑的
    # 那一組，不在 tuning.yaml 複製一份（見該檔說明）。
    base_params = dict(load_model_config()["model"])
    train_cfg = cfg["training"]

    feb, mar = load_cohort_features(paths, cfg)
    split = three_way_split(feb, cfg["split"])
    print(
        f"  Feb 內部三段切分：訓練 {split.train.X.height:,}"
        f" · early stopping {split.es.X.height:,}"
        f" · 選參數 {split.sel.X.height:,}\n"
    )

    n_trials = cfg["search"]["n_trials"]
    print(f"隨機搜尋 {n_trials} 組 + 1 組基準設定（Mar cohort 全程不參與）...")
    t0 = time.perf_counter()
    trials = random_search(
        split,
        base_params,
        cfg["search_space"],
        train_cfg,
        n_trials=n_trials,
        seed=cfg["search"]["seed"],
        verbose=True,
    )
    print(f"\n搜尋完成，共 {time.perf_counter() - t0:.0f} 秒")

    # 表格的欄位由**搜尋空間**推導，不寫死參數名。
    #
    # 寫死會踩到一個安靜的陷阱：基準設定（configs/model_lgbm.yaml）不見得
    # 宣告了搜尋空間裡的每一個鍵 —— 例如它沒有 lambda_l1，因為它用的是
    # LightGBM 的預設值 0。直接用 t.params["lambda_l1"] 會在 trial 0 上
    # KeyError。用 .get 並在缺席時顯示 LightGBM 的預設值，讀表的人才知道
    # 那一格不是沒跑，而是「用了套件預設」。
    lgb_defaults = {"lambda_l1": 0.0, "lambda_l2": 0.0}
    searched = list(cfg["search_space"])

    def cell(t, key: str):
        v = t.params.get(key, lgb_defaults.get(key))
        return round(v, 4) if isinstance(v, float) else v

    table = pl.DataFrame(
        [
            {
                "排名": i,
                "trial": t.index,
                "sel log loss": round(t.select_logloss, 5),
                "輪數": t.best_iteration,
                **{k: cell(t, k) for k in searched},
            }
            for i, t in enumerate(trials, 1)
        ]
    )

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_width_chars(180)
    print("\n" + "=" * 78)
    print("隨機搜尋結果（依 Feb-sel 分數排序；trial 0 = M1/M2 的設定）")
    print("=" * 78)
    print(table.head(15))

    best = trials[0]
    base = next(t for t in trials if t.index == 0)
    print(f"\n最佳 trial {best.index}：sel {best.select_logloss:.5f}")
    print(f"基準 trial 0　　：sel {base.select_logloss:.5f}")
    print(f"sel 上的改善：{base.select_logloss - best.select_logloss:+.5f}")

    # --- Mar 只在這裡看一次 ---------------------------------------------------
    print("\n在 Mar cohort 上評估（本次搜尋唯一一次看時間外分數）...")
    best_mar, best_segments = evaluate_on_mar(best.params, split, feb, mar, train_cfg)
    base_mar, base_segments = evaluate_on_mar(base.params, split, feb, mar, train_cfg)

    print("\n--- 最佳參數 · Mar 分群回報（SPEC §4.5）---")
    print(best_segments)

    print("\n" + "=" * 78)
    print(f"基準設定　Mar log loss {base_mar:.5f}")
    print(f"調參之後　Mar log loss {best_mar:.5f}")
    print(f"實際改善　{base_mar - best_mar:+.5f}（{(best_mar / base_mar - 1):+.2%}）")
    print("=" * 78)
    print(
        "\n⚠️ sel 上的改善通常大於 Mar 上的改善，兩者的差距就是「挑了 31 組裡最合\n"
        "   Feb 胃口的那組」所帶來的樂觀偏誤。回報時以 Mar 的數字為準。"
    )

    out = paths.interim / "m3_artifacts"
    out.mkdir(parents=True, exist_ok=True)
    (out / "best_params.yaml").write_text(
        yaml.safe_dump(
            {
                "trial_index": best.index,
                "sel_logloss": float(best.select_logloss),
                "mar_logloss": float(best_mar),
                "best_iteration": best.best_iteration,
                "model": best.params,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    print(f"\n最佳參數已寫入 {out / 'best_params.yaml'}")

    run_id = log_to_mlflow(cfg, best, best_mar, base_mar)
    if run_id:
        print(f"MLflow run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
