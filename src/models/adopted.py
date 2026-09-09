"""正式採用的模型 —— 單一來源。

§7.12 的配對 multi-seed 選定 **CatBoost**（Mar 0.15367，8 個 seed 全勝，
反向驗證同向）。本模組的存在理由是讓「正式採用的是哪一個、用哪組參數」
只有一個答案。

## 為什麼需要這個模組

M4 一度出現這個狀況：校準診斷（`calibration_report.py`）與校準器
（`calibrate.py`）用 LightGBM，而業務指標（`business_value.py`）用 CatBoost。
**於是「校準器不該上線」這個結論，是在一個不會上線的模型上得出的。**
兩支腳本各自 `load_model_config()` 讀 `configs/model_lgbm.yaml`，看起來
完全正常，沒有任何東西會抱怨。

超參數也只能有一份來源。複製一份到別的設定檔，兩邊遲早不同步，而
「這條曲線是用哪組參數算的」屆時就沒有答案 —— 這正是 §7.4 的比較表
能成立的前提（三家吃同一份設定）。

因此：**採用哪個模型寫在這裡，參數一律從 `configs/model_comparison.yaml`
讀**，任何腳本都不得自帶一份。

## 換模型的時候

改 `ADOPTED_MODEL` 一個字，然後重跑 M4 的三支腳本（校準診斷、校準器、
業務指標）。`tests/test_m4_contracts.py` 會確認沒有腳本繞過這裡。
"""

from __future__ import annotations

from typing import Any

import yaml

from src.config import REPO_ROOT
from src.features import FeatureSet
from src.models.candidates import (
    Fitted,
    fit_catboost,
    fit_lightgbm,
    fit_xgboost,
    xgb_category_levels,
)

# §7.12 正式採用的模型。改這一行等於改變所有下游交付物。
ADOPTED_MODEL = "catboost"

OFFICIAL_CONFIG = REPO_ROOT / "configs" / "model_comparison.yaml"

_FITTERS = {"lightgbm": fit_lightgbm, "xgboost": fit_xgboost, "catboost": fit_catboost}


def load_adopted_config(model: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """回傳 (超參數, 訓練協定)，來源固定是 `configs/model_comparison.yaml`。

    Raises:
        FileNotFoundError: 設定檔不存在。
        KeyError: 設定檔缺少該模型或 training 區段。
    """
    if not OFFICIAL_CONFIG.exists():
        raise FileNotFoundError(f"找不到正式模型設定檔 {OFFICIAL_CONFIG}")
    cfg = yaml.safe_load(OFFICIAL_CONFIG.read_text(encoding="utf-8")) or {}

    key = model or ADOPTED_MODEL
    for section in ("models", "training"):
        if section not in cfg:
            raise KeyError(f"{OFFICIAL_CONFIG} 缺少 [{section}] 區段")
    if key not in cfg["models"]:
        raise KeyError(f"{OFFICIAL_CONFIG} 沒有 {key!r} 的超參數")

    return dict(cfg["models"][key]), dict(cfg["training"])


def fit_adopted(
    train: FeatureSet,
    es: FeatureSet,
    *,
    model: str | None = None,
    all_train: FeatureSet | None = None,
) -> Fitted:
    """用正式採用的模型與參數訓練一次。

    Args:
        train / es: 訓練與 early stopping 的特徵矩陣。
        model:      覆寫要用哪個模型（只給對照實驗用，正式流程別傳）。
        all_train:  XGBoost 需要的類別字典來源。**從整個訓練 cohort 算**，
                    而不是從切分後的 train —— 否則字典會隨切分變動。
                    CatBoost / LightGBM 不需要。

    Returns:
        `Fitted`，介面與 `src.models.candidates` 的三個 fitter 相同。
    """
    key = model or ADOPTED_MODEL
    params, train_cfg = load_adopted_config(key)

    extra: dict[str, Any] = {}
    if key == "xgboost":
        source = all_train if all_train is not None else train
        extra["category_levels"] = xgb_category_levels(source.X, categorical=source.categorical)

    return _FITTERS[key](train, es, params, train_cfg, **extra)
