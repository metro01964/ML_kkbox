"""M3 · LightGBM / XGBoost / CatBoost 三方比較。

SPEC §7 的 M3 交付物之一是「比較表」。這個模組負責產生它。

## 這個實驗要回答什麼

**不是**「哪一家最強」。三個梯度提升套件在表格資料上的差距通常是千分之幾，
選錯不會毀掉專案。真正要回答的是：

  1. **M2 的 0.15821 有多少是 LightGBM 的功勞？** 如果三家分數幾乎相同，
     代表分數由特徵決定，換模型不是有效的施力點 —— 那麼 M3 剩下的力氣就該
     花在特徵篩選（`src/models/selection.py`）而不是模型選擇。
  2. **三家犯的錯一樣嗎？** 若取平均後明顯變好，代表它們的錯誤不相關，
     集成有價值；若沒變好，代表三家學到的是同一件事。

第 2 點是設定檔裡 `ensemble.enabled` 的用意 —— 那一列不是為了刷分，是為了
測量「多樣性」這件事有沒有實體。

## 為什麼每家都要跑 §4.5 的分群回報

SPEC §4.5：「M1 起每一次評估都必須回報三個數字。」headline 分數會被 90.81%
的低風險重複用戶稀釋。三家的總分可能只差 0.001，但在 9.19% 的新進用戶上
差很多 —— 那才是挽回名單真正要處理的族群。只看總分會選錯模型。

執行入口：

    uv run python scripts/compare.py
    make compare
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml
from sklearn.model_selection import train_test_split

from src.config import REPO_ROOT, Paths
from src.data import FEB, MAR, CohortSpec
from src.evaluation import constant_log_loss, log_loss, repeat_vs_new, segment_report
from src.features import FeatureSet
from src.models.candidates import (
    Fitted,
    fit_catboost,
    fit_lightgbm,
    fit_xgboost,
    xgb_category_levels,
)
from src.models.train import load_cohort_features

DEFAULT_CONFIG = REPO_ROOT / "configs" / "model_comparison.yaml"

# 設定檔的鍵 → 建構函式。想加第四個套件只要在這裡多一列。
CANDIDATES = ("lightgbm", "xgboost", "catboost")


@dataclass
class CandidateResult:
    """一個候選模型在 Mar cohort 上的完整表現。"""

    name: str
    key: str
    best_iteration: int
    logloss: float
    segments: pl.DataFrame
    importance: pl.DataFrame
    seconds: float
    pred: np.ndarray

    def segment_score(self, label: str) -> float:
        row = self.segments.filter(pl.col("分群") == label)
        return float(row["log_loss"][0]) if row.height else float("nan")


def load_comparison_config(path: Path | None = None) -> dict[str, Any]:
    """讀 configs/model_comparison.yaml，缺區段就直接失敗。

    不套用預設值的理由同 `load_model_config`：靜默的預設值會讓「我改了設定
    但沒生效」變成極難察覺的問題。
    """
    path = path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"找不到比較設定檔 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("models", "training"):
        if section not in cfg:
            raise KeyError(f"{path} 缺少 [{section}] 區段")
    missing = [k for k in CANDIDATES if k not in cfg["models"]]
    if missing:
        raise KeyError(f"{path} 的 [models] 缺少：{missing}")
    return cfg


def split_for_early_stopping(feb: FeatureSet, train_cfg: dict) -> tuple[FeatureSet, FeatureSet]:
    """從 Feb cohort 內部切出 early stopping 用的一小塊。

    **三家共用同一次切分**：切的是「列的索引」，不是各套件各自轉換後的矩陣。
    如果讓三家各自呼叫 train_test_split，即使 seed 相同，只要輸入的 dtype 或
    列順序有任何差異就可能切出不同的列，比較就不再是同一批資料。

    分層抽樣（stratify）在流失率只有 6.39% 時是必要的：不分層的話，
    20% 的驗證集裡正例數量會有可觀的隨機波動，early stopping 的停點跟著晃。
    """
    idx = np.arange(feb.X.height)
    tr_idx, es_idx = train_test_split(
        idx,
        test_size=train_cfg["inner_valid_fraction"],
        random_state=train_cfg["inner_split_seed"],
        stratify=feb.y.to_numpy(),
    )
    return feb.take(tr_idx), feb.take(es_idx)


def evaluate(
    fitted: Fitted,
    key: str,
    mar: FeatureSet,
    feb: FeatureSet,
    seconds: float,
) -> CandidateResult:
    """在 Mar cohort 上評估，並產生 SPEC §4.5 要求的分群報告。"""
    pred = fitted.predict(mar.X)
    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )
    return CandidateResult(
        name=fitted.name,
        key=key,
        best_iteration=fitted.best_iteration,
        logloss=log_loss(mar.y, pred),
        segments=segment_report(scored, segment_col="segment"),
        importance=fitted.importance,
        seconds=seconds,
        pred=pred,
    )


def run_comparison(
    paths: Paths,
    cfg: dict[str, Any],
    *,
    train_spec: CohortSpec = FEB,
    valid_spec: CohortSpec = MAR,
    only: list[str] | None = None,
    verbose: bool = True,
) -> tuple[list[CandidateResult], FeatureSet, FeatureSet]:
    """三家各訓練一次，回傳結果清單與兩份特徵矩陣（供後續報表使用）。

    Args:
        train_spec / valid_spec: 訓練與驗證的 cohort。預設 Feb → Mar。
        only: 只跑這幾個候選（除錯用）。None 代表全跑。
    """

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    feb, mar = load_cohort_features(
        paths, cfg, train_spec=train_spec, valid_spec=valid_spec, verbose=verbose
    )
    train, es = split_for_early_stopping(feb, cfg["training"])
    log(f"  {train_spec.name} 內部切分：訓練 {train.X.height:,} · early stopping {es.X.height:,}")

    # XGBoost 的類別字典**只用訓練 cohort 擬合**（紅線 5：encoding 狀態必須
    # 在訓練資料內計算）。Mar 若出現 Feb 沒有的取值，推論時落到缺失分支 ——
    # 那正是部署時會發生的事，見 xgb_category_levels 的 docstring。
    levels = xgb_category_levels(feb.X, categorical=feb.categorical)

    keys = [k for k in CANDIDATES if only is None or k in only]
    results: list[CandidateResult] = []

    for i, key in enumerate(keys, 1):
        params = dict(cfg["models"][key])
        log(f"\n[{i}/{len(keys)}] {key} 訓練中...")
        t0 = time.perf_counter()

        if key == "lightgbm":
            fitted = fit_lightgbm(train, es, params, cfg["training"])
        elif key == "xgboost":
            fitted = fit_xgboost(train, es, params, cfg["training"], category_levels=levels)
        elif key == "catboost":
            fitted = fit_catboost(train, es, params, cfg["training"])
        else:  # pragma: no cover - CANDIDATES 已窮舉
            raise ValueError(f"未知的候選模型 {key}")

        secs = time.perf_counter() - t0
        r = evaluate(fitted, key, mar, feb, secs)
        results.append(r)
        log(
            f"      → {valid_spec.name} log loss {r.logloss:.5f}"
            f"　停在第 {r.best_iteration} 輪　({secs:.0f} 秒)"
        )

    return results, feb, mar


def mean_ensemble(
    results: list[CandidateResult], mar: FeatureSet, feb: FeatureSet
) -> CandidateResult:
    """三家預測值取算術平均。

    取平均而非 logit 平均：log loss 直接懲罰機率本身，而機率的算術平均仍是
    合法機率、且是「等權重相信三個模型」最直白的表述。logit 平均會放大
    極端預測，在需要校準的場景（M4）反而不利。
    """
    pred = np.mean([r.pred for r in results], axis=0)
    scored = pl.DataFrame(
        {
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )
    return CandidateResult(
        name="三者平均",
        key="ensemble_mean",
        best_iteration=0,
        logloss=log_loss(mar.y, pred),
        segments=segment_report(scored, segment_col="segment"),
        importance=pl.DataFrame({"feature": [], "gain": [], "gain_share": []}),
        seconds=sum(r.seconds for r in results),
        pred=pred,
    )


def comparison_table(results: list[CandidateResult], baseline: float) -> pl.DataFrame:
    """把結果整成 SPEC §7 M3 要求的比較表。"""
    ref = results[0].logloss  # 第一列（LightGBM）是參照點
    return pl.DataFrame(
        [
            {
                "模型": r.name,
                "輪數": r.best_iteration,
                "Mar log loss": round(r.logloss, 5),
                "相對 LightGBM": round(r.logloss / ref - 1, 5),
                "重複用戶": round(r.segment_score("重複用戶"), 5),
                "新進用戶": round(r.segment_score("新進用戶"), 5),
                "vs 常數基準": round(1 - r.logloss / baseline, 4),
                "秒": round(r.seconds),
            }
            for r in results
        ]
    )


def rank_agreement(results: list[CandidateResult], top_n: int = 15) -> pl.DataFrame:
    """三家的特徵重要度**排名**對照表。

    比的是排名不是 gain 值 —— 三家的 gain 定義不同（見 candidates 模組的
    模組註解），絕對值不可比。排名一致代表三家看到的是同一組訊號，
    這正是 null importance 篩選可以只用一個模型來做的前提。
    """
    frames = []
    for r in results:
        if r.importance.height == 0:
            continue
        frames.append(
            r.importance.with_row_index("rank", offset=1)
            .select("feature", pl.col("rank").alias(r.name))
            .with_columns(pl.col(r.name).cast(pl.Int32))
        )
    if not frames:
        return pl.DataFrame()

    table = frames[0]
    for f in frames[1:]:
        table = table.join(f, on="feature", how="full", coalesce=True)

    name_cols = [c for c in table.columns if c != "feature"]
    return (
        table.with_columns(pl.mean_horizontal(name_cols).alias("平均排名"))
        .sort("平均排名")
        .head(top_n)
    )


def log_comparison_to_mlflow(
    results: list[CandidateResult],
    cfg: dict[str, Any],
    baseline: float,
) -> list[str]:
    """每個候選模型各記一個 MLflow run。

    分開記而不是一個 run 記三組指標：MLflow 的比較介面是以 run 為單位的，
    三家各自成 run 才能在 UI 上並排、排序、畫圖。
    """
    import mlflow

    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return []

    backend = tracking.get("backend", "sqlite:///mlflow.db")
    if backend.startswith("sqlite:///") and not backend.startswith("sqlite:////"):
        rel = backend.removeprefix("sqlite:///")
        backend = f"sqlite:///{(REPO_ROOT / rel).as_posix()}"
    mlflow.set_tracking_uri(backend)

    exp_name = tracking.get("experiment", "kkbox-churn")
    if mlflow.get_experiment_by_name(exp_name) is None:
        artifact_dir = REPO_ROOT / tracking.get("artifact_dir", "mlartifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(exp_name)

    prefix = tracking.get("run_prefix", "m3_compare")
    run_ids = []
    for r in results:
        with mlflow.start_run(run_name=f"{prefix}_{r.key}") as run:
            mlflow.log_param("model", r.name)
            mlflow.log_param("features.use_logs", cfg.get("features", {}).get("use_logs", False))
            mlflow.log_param("best_iteration", r.best_iteration)
            if r.key in cfg["models"]:
                mlflow.log_params({f"{r.key}.{k}": v for k, v in cfg["models"][r.key].items()})
            mlflow.log_metrics(
                {
                    "mar_logloss": r.logloss,
                    "constant_baseline": baseline,
                    "improvement": 1 - r.logloss / baseline,
                    "train_seconds": r.seconds,
                }
            )
            for row in r.segments.iter_rows(named=True):
                tag = {"全體": "overall", "重複用戶": "repeat", "新進用戶": "new"}[row["分群"]]
                mlflow.log_metric(f"logloss_{tag}", row["log_loss"])
            run_ids.append(run.info.run_id)
    return run_ids


def constant_baseline(feb: FeatureSet, mar: FeatureSet) -> float:
    """SPEC §3.3 的常數基準（實測 0.30746），三家共用同一個分母。"""
    return constant_log_loss(float(feb.y.mean()), mar.y)
