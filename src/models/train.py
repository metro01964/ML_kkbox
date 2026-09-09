"""M1 · LightGBM baseline。

SPEC §4.2 的時間外驗證：

    訓練 Feb cohort（2017-02 到期，觀察期 2017-03）
    驗證 Mar cohort（2017-03 到期，觀察期 2017-04）

驗收門檻是 **Mar cohort 的 log loss < 0.30746**（SPEC §3.3），也就是「用 Feb
的流失率 6.3923% 對所有人做常數預測」的分數。打不贏它代表模型學到的東西
還不如「大家風險都一樣」這個假設。

## Early stopping 的驗證集為什麼不用 Mar

要停在第幾輪是一個**看著分數做的決定**。如果拿 Mar 來決定，回報的 Mar 分數
就已經被它自己影響過，會偏樂觀 —— 而那正是要跟 0.30746 比較、並外推到測試集
的數字。所以 early stopping 用 Feb 內部切出來的一小塊，Mar 全程不參與訓練，
保持乾淨的時間外身分。

在單一 cohort 內部做隨機分層切分是安全的：契約測試已驗證 `msno` 在一個
cohort 內不重複，且 as-of 截斷逐用戶計算，不存在時間洩漏。SPEC §4.2 也明文
允許「在 Feb cohort 內部另做 StratifiedKFold」。

執行入口在 `scripts/train.py`：

    uv run python scripts/train.py
    make train
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.config import REPO_ROOT, Paths, load_paths
from src.data import FEB, MAR, CohortSpec, assert_labels_are_real, build_cohort
from src.evaluation import constant_log_loss, log_loss, repeat_vs_new, segment_report
from src.features import FeatureSet, build_features, build_log_features
from src.models.candidates import to_lgb_arrays

DEFAULT_CONFIG = REPO_ROOT / "configs" / "model_lgbm.yaml"

# SPEC §3.3 的 M1 驗收門檻。寫成常數是為了讓程式自己判斷有沒有過關，
# 而不是印個數字讓人自己比對 —— 人會看漏。
M1_THRESHOLD = 0.30746


@dataclass
class TrainResult:
    """一次訓練的完整結果，供報表與後續步驟使用。"""

    booster: lgb.Booster
    best_iteration: int
    logloss: float
    baseline_logloss: float
    segments: pl.DataFrame
    importance: pl.DataFrame
    valid_pred: np.ndarray

    @property
    def beats_baseline(self) -> bool:
        return self.logloss < self.baseline_logloss

    @property
    def improvement(self) -> float:
        """相對常數基準的改善比例。"""
        return 1 - self.logloss / self.baseline_logloss


@dataclass
class CVResult:
    """Feb cohort 內部 5-fold 的結果。**只用於估計變異數，不用於模型選擇。**"""

    fold_scores: list[float]
    n_splits: int
    num_boost_round: int

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std(self) -> float:
        # ddof=1：這是樣本標準差。5 個 fold 是母體的樣本，不是母體本身。
        return float(np.std(self.fold_scores, ddof=1))


def load_model_config(path: Path | None = None) -> dict[str, Any]:
    """讀 configs/model_lgbm.yaml。

    找不到就直接失敗，不套用預設值 —— 靜默的預設值會讓「我改了設定但沒生效」
    這種問題極難察覺。
    """
    path = path or DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"找不到模型設定檔 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for section in ("model", "training"):
        if section not in cfg:
            raise KeyError(f"{path} 缺少 [{section}] 區段")
    return cfg


def load_cohort_features(
    paths: Paths,
    cfg: dict[str, Any],
    *,
    train_spec: CohortSpec = FEB,
    valid_spec: CohortSpec = MAR,
    keep_features: list[str] | None = None,
    verbose: bool = True,
) -> tuple[FeatureSet, FeatureSet]:
    """建好訓練與驗證兩份特徵矩陣。

    從 `train_baseline` 抽出來，讓 M3 的模型比較與特徵篩選共用同一段載入
    邏輯。三個套件如果各自載一次資料，「特徵集完全相同」這個前提就只是
    口頭承諾而不是程式保證。

    預設是 SPEC §4.2 的 Feb → Mar。兩個 spec 開放參數化是為了 `scripts/
    reverse_validation.py` 的反向驗證（Mar → Feb）—— **那不是第二個部署估計**，
    是「換一組 train/valid，M3 的結論還成不成立」的穩健性探測。理由寫在
    該腳本的模組註解裡。

    Raises:
        ValueError: 兩個 cohort 的特徵欄位不一致，或 train/valid 是同一個月。
    """
    if train_spec.name == valid_spec.name:
        raise ValueError(f"訓練與驗證不能是同一個 cohort（都是 {train_spec.name}）")
    # Kaggle 測試集的標籤是佔位值（全 0）。拿它當驗證集會算出一個看起來合理
    # 而毫無意義的分數 —— 只要 spec 名稱打錯一個字就會發生。
    assert_labels_are_real(train_spec, valid_spec)

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    use_logs = cfg.get("features", {}).get("use_logs", False)
    log(f"載入 cohort（收聽特徵：{'啟用' if use_logs else '停用'}）...")
    train_raw = build_cohort(train_spec, paths, verbose=False)
    valid_raw = build_cohort(valid_spec, paths, verbose=False)

    train_logs = valid_logs = None
    if use_logs:
        train_logs = build_log_features(train_spec, paths, verbose=False)
        valid_logs = build_log_features(valid_spec, paths, verbose=False)
        log(
            f"  收聽特徵覆蓋 {train_spec.name} {train_logs.height / train_raw.height:.2%}"
            f" · {valid_spec.name} {valid_logs.height / valid_raw.height:.2%}"
        )

    train = build_features(train_raw, train_logs)
    valid = build_features(valid_raw, valid_logs)

    if keep_features is not None:
        train = train.select(keep_features)
        valid = valid.select(keep_features)

    log(
        f"  訓練 {train_spec.name} {train.X.height:,} 列 × {train.X.width} 特徵，"
        f"流失率 {train.y.mean():.4%}"
    )
    log(
        f"  驗證 {valid_spec.name} {valid.X.height:,} 列 × {valid.X.width} 特徵，"
        f"流失率 {valid.y.mean():.4%}"
    )

    if train.X.columns != valid.X.columns:
        raise ValueError("兩個 cohort 的特徵欄位不一致，模型無法套用")

    return train, valid


def train_baseline(
    paths: Paths | None = None,
    config: dict[str, Any] | None = None,
    *,
    keep_features: list[str] | None = None,
    verbose: bool = True,
) -> TrainResult:
    """在 Feb cohort 上訓練，在 Mar cohort 上評估。

    Args:
        keep_features: 只保留這些特徵欄位。供消融實驗使用；None 代表全用。

    Returns:
        TrainResult，含 Mar cohort 的 log loss、分群報告與特徵重要度。
    """
    paths = paths or load_paths()
    cfg = config or load_model_config()
    params = dict(cfg["model"])
    train_cfg = cfg["training"]

    def log(msg: str = "") -> None:
        if verbose:
            print(msg, flush=True)

    # ---- 資料 ----
    feb, mar = load_cohort_features(paths, cfg, keep_features=keep_features, verbose=verbose)

    X_feb, y_feb, cat_idx = to_lgb_arrays(feb)
    X_mar, y_mar, _ = to_lgb_arrays(mar)

    # ---- Feb 內部切一小塊給 early stopping（Mar 全程不參與訓練）----
    #
    # 切分綁在 msno 上而不是列位置上：train_test_split 依位置切，同一個 seed
    # 餵進不同順序的資料會切出不同的人。cohort 已由 build_cohort 排序，所以
    # 下面的 argsort 是恆等變換、不改變任何既有分數；它的作用是讓「順序被
    # 上游改動」不再等於「換一批訓練資料」。理由詳見
    # src/models/tuning.py::three_way_split 與 src/data/cohort.py 的
    # assert_rows_reproducible。
    canonical = feb.msno.arg_sort().to_numpy()
    tr_idx, es_idx = train_test_split(
        canonical,
        test_size=train_cfg["inner_valid_fraction"],
        random_state=train_cfg["inner_split_seed"],
        stratify=y_feb[canonical],
    )
    X_tr, y_tr = X_feb[tr_idx], y_feb[tr_idx]
    X_es, y_es = X_feb[es_idx], y_feb[es_idx]
    log(f"  Feb 內部切分：訓練 {len(y_tr):,} · early stopping {len(y_es):,}")

    # ---- 訓練 ----
    log("\n訓練中...")
    dtrain = lgb.Dataset(X_tr, y_tr, categorical_feature=cat_idx, free_raw_data=False)
    des = lgb.Dataset(X_es, y_es, categorical_feature=cat_idx, reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=train_cfg["num_boost_round"],
        valid_sets=[des],
        valid_names=["feb_inner"],
        callbacks=[
            lgb.early_stopping(train_cfg["early_stopping_rounds"], verbose=verbose),
            lgb.log_evaluation(train_cfg["log_every_n"] if verbose else 0),
        ],
    )
    log(f"  最佳輪數 {booster.best_iteration}")

    # ---- 在 Mar cohort 上評估 ----
    pred = booster.predict(X_mar, num_iteration=booster.best_iteration)
    ll = log_loss(y_mar, pred)
    baseline = constant_log_loss(float(feb.y.mean()), y_mar)

    # ---- SPEC §4.5 要求的分群回報 ----
    scored = pl.DataFrame(
        {
            "msno": mar.msno,
            "is_churn": mar.y,
            "p_churn": pred,
            "segment": repeat_vs_new(mar.msno, feb.msno),
        }
    )
    segments = segment_report(scored, segment_col="segment")

    importance = (
        pl.DataFrame(
            {
                "feature": feb.X.columns,
                "gain": booster.feature_importance("gain"),
                "split": booster.feature_importance("split"),
            }
        )
        .with_columns((pl.col("gain") / pl.col("gain").sum()).alias("gain_share"))
        .sort("gain", descending=True)
    )

    return TrainResult(
        booster=booster,
        best_iteration=booster.best_iteration,
        logloss=ll,
        baseline_logloss=baseline,
        segments=segments,
        importance=importance,
        valid_pred=pred,
    )


def cross_validate_feb(
    feb: FeatureSet,
    params: dict[str, Any],
    *,
    num_boost_round: int,
    n_splits: int,
    seed: int,
    verbose: bool = True,
) -> CVResult:
    """在 Feb cohort 內部做 StratifiedKFold，估計分數的變異數。

    SPEC §4.2 對這個數字的定位：「用途**僅限**估計模型變異數（回報標準差）。
    模型選擇一律以時間外驗證分數為準。」所以它回答的是「這個分數穩不穩」，
    不是「這個模型好不好」。

    每個 fold 用固定的 num_boost_round，不各自 early stopping —— 要衡量的是
    「換一批資料分數差多少」，讓各 fold 自己找停點會把停點的變異也算進來，
    兩種來源就混在一起了。

    ⚠️ 這個分數會明顯**優於** Mar cohort 的時間外分數，而且那個差距不是
    bug。fold 內的訓練與驗證同屬 2017-02 到期的族群，分布相同；Mar cohort
    的流失率是 8.99% 而 Feb 是 6.39%，是不同的分布。SPEC §4.2 說得很清楚：
    「若兩者差距過大，該差距本身就是要寫進報告的發現（概念漂移），不是要
    調掉的問題。」
    """
    X, y, cat_idx = to_lgb_arrays(feb)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores: list[float] = []

    for i, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        dtrain = lgb.Dataset(X[tr_idx], y[tr_idx], categorical_feature=cat_idx)
        booster = lgb.train(params, dtrain, num_boost_round=num_boost_round)
        pred = booster.predict(X[va_idx])
        score = log_loss(y[va_idx], pred)
        scores.append(score)
        if verbose:
            print(f"  fold {i}/{n_splits}  log loss {score:.5f}", flush=True)

    return CVResult(fold_scores=scores, n_splits=n_splits, num_boost_round=num_boost_round)


def log_to_mlflow(
    result: TrainResult,
    cv: CVResult | None,
    cfg: dict[str, Any],
    paths: Paths,
) -> str | None:
    """把這次實驗記進 MLflow（SPEC §8）。

    在報表印出**之後**才呼叫：追蹤失敗不該讓辛苦訓練出來的數字消失在
    stack trace 裡。
    """
    import mlflow

    tracking = cfg.get("tracking", {})
    if not tracking.get("enabled", False):
        return None

    # 後端 URI。設定檔給的是相對路徑（sqlite:///mlflow.db），要展開成絕對路徑，
    # 否則 db 會落在「當下工作目錄」—— 從 repo 根目錄跑和從別處跑會產生兩份。
    backend = tracking.get("backend", "sqlite:///mlflow.db")
    if backend.startswith("sqlite:///") and not backend.startswith("sqlite:////"):
        rel = backend.removeprefix("sqlite:///")
        backend = f"sqlite:///{(REPO_ROOT / rel).as_posix()}"
    mlflow.set_tracking_uri(backend)

    # 用資料庫後端時，artifact 存放位置必須在建立 experiment 時指定 ——
    # 資料庫只存 metadata，模型檔與 CSV 仍然是檔案。
    exp_name = tracking.get("experiment", "kkbox-churn")
    if mlflow.get_experiment_by_name(exp_name) is None:
        artifact_dir = REPO_ROOT / tracking.get("artifact_dir", "mlartifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(exp_name, artifact_location=artifact_dir.as_uri())
    mlflow.set_experiment(exp_name)

    use_logs = cfg.get("features", {}).get("use_logs", False)
    run_name = tracking.get("run_name") or ("m2_with_logs" if use_logs else "m1_transactions_only")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("features.use_logs", use_logs)
        mlflow.log_params({f"lgb.{k}": v for k, v in cfg["model"].items()})
        mlflow.log_params({f"train.{k}": v for k, v in cfg["training"].items()})
        mlflow.log_param("best_iteration", result.best_iteration)
        mlflow.log_param("n_features", result.importance.height)

        mlflow.log_metrics(
            {
                "mar_logloss": result.logloss,
                "constant_baseline": result.baseline_logloss,
                "improvement": result.improvement,
            }
        )
        for row in result.segments.iter_rows(named=True):
            tag = {"全體": "overall", "重複用戶": "repeat", "新進用戶": "new"}[row["分群"]]
            mlflow.log_metric(f"logloss_{tag}", row["log_loss"])
            mlflow.log_metric(f"mean_pred_{tag}", row["平均預測機率"])
            mlflow.log_metric(f"actual_rate_{tag}", row["實際流失率"])

        if cv is not None:
            mlflow.log_metrics({"cv_mean": cv.mean, "cv_std": cv.std})
            for i, s in enumerate(cv.fold_scores, 1):
                mlflow.log_metric(f"cv_fold_{i}", s)
            # 時間外分數與 fold 內分數的差距 —— SPEC §4.2 要求寫進報告的發現。
            mlflow.log_metric("oot_minus_cv", result.logloss - cv.mean)

        # 模型與特徵重要度存成 artifact。用 save_model 而非 mlflow.lightgbm，
        # 產出的 txt 可以用 lgb.Booster(model_file=...) 直接載回，不綁 MLflow 版本。
        out = paths.interim / "m1_artifacts"
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / "lgbm_baseline.txt"
        imp_path = out / "feature_importance.csv"
        result.booster.save_model(str(model_path), num_iteration=result.best_iteration)
        result.importance.write_csv(imp_path)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(imp_path))

        return run.info.run_id


def _print_report(r: TrainResult) -> None:
    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(140)

    print("\n" + "=" * 70)
    print("M1 · LightGBM baseline 結果")
    print("=" * 70)
    print(f"最佳輪數              {r.best_iteration}")
    print(f"常數基準 log loss     {r.baseline_logloss:.5f}   ← SPEC §3.3 門檻")
    print(f"模型 log loss         {r.logloss:.5f}")
    print(f"改善                  {r.improvement:.2%}")
    print(f"驗收                  {'✅ 通過' if r.beats_baseline else '❌ 未達門檻'}")

    print("\n--- 分群回報（SPEC §4.5 要求）---")
    print(r.segments)

    print("\n--- 特徵重要度 Top 12（依 gain）---")
    print(r.importance.head(12))

    zero = r.importance.filter(pl.col("gain") == 0)
    if zero.height:
        print(f"\n完全沒被用到的特徵 {zero.height} 個：{zero['feature'].to_list()}")


def _print_cv(r: TrainResult, cv: CVResult) -> None:
    print(f"\n--- Feb 內部 {cv.n_splits}-fold（SPEC §7 要求回報標準差）---")
    for i, s in enumerate(cv.fold_scores, 1):
        print(f"  fold {i}  {s:.5f}")
    print(f"  平均 {cv.mean:.5f} ± {cv.std:.5f}")

    gap = r.logloss - cv.mean
    print(f"\n  時間外（Mar）{r.logloss:.5f}  −  fold 內（Feb）{cv.mean:.5f}  =  {gap:+.5f}")
    print(f"  差距是標準差的 {gap / cv.std:.0f} 倍。")
    print(
        "\n  這個差距不是 bug，是概念漂移的量化。fold 內的訓練與驗證同屬 2017-02\n"
        "  到期族群（流失率 6.39%），Mar cohort 是 8.99% 的另一個分布。\n"
        "  SPEC §4.2：「該差距本身就是要寫進報告的發現，不是要調掉的問題。」\n"
        "  ⚠️ 只報 fold 內分數會讓模型看起來好一倍 —— 那是最常見的自欺方式。"
    )


def main() -> int:
    try:
        cfg = load_model_config()
        paths = load_paths()
        result = train_baseline(paths, cfg)
    except FileNotFoundError as e:
        sys.exit(f"{e}\n請先執行 uv run python scripts/download.py")

    cv = None
    cv_cfg = cfg.get("cv", {})
    if cv_cfg.get("enabled", False):
        print(f"\n{cv_cfg['n_splits']}-fold 變異數估計（固定 {result.best_iteration} 輪）...")
        feb = build_features(build_cohort(FEB, paths, verbose=False))
        cv = cross_validate_feb(
            feb,
            dict(cfg["model"]),
            num_boost_round=result.best_iteration,
            n_splits=cv_cfg["n_splits"],
            seed=cv_cfg["seed"],
        )

    # 先印報表再記 MLflow：追蹤失敗不該讓訓練結果消失在 stack trace 裡。
    _print_report(result)
    if cv is not None:
        _print_cv(result, cv)

    run_id = log_to_mlflow(result, cv, cfg, paths)
    if run_id:
        print(f"\nMLflow run {run_id}　（用 uv run mlflow ui 檢視）")

    if not result.beats_baseline:
        print(f"\n❌ Mar cohort log loss {result.logloss:.5f} 未打敗基準 {M1_THRESHOLD}")
        return 1
    return 0
