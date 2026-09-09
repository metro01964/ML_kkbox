"""M6 · 服務層（SPEC §7 的 `src/serving/`）。

三個模組，分工是「載入 / 組特徵 / 算分」，FastAPI 的路由只負責 HTTP：

    artifact.py   模型 artifact 的存與載（含守門：模型檔與 metadata 必須同源）
    payload.py    一筆 payload → 特徵矩陣（**轉換一律走 build_features()**）
    score.py      特徵矩陣 → 機率 + 原因碼（與 M5 走同一條歸因路徑）
    app.py        FastAPI 路由

`app.py` 刻意不 import 到這裡 —— 它會拉進 fastapi，而 `scripts/export_model.py`
與 Kaggle 推論管線只需要前三個。
"""

from src.serving.artifact import (
    ARTIFACT_VERSION,
    ENV_ARTIFACT,
    META_FILE,
    Artifact,
    artifact_dir,
    load_artifact,
    save_artifact,
)
from src.serving.payload import (
    COHORT_FIELDS,
    MEMBER_FIELDS,
    cohort_row,
    feature_row,
)
from src.serving.score import Scored, score_rows

__all__ = [
    "ARTIFACT_VERSION",
    "COHORT_FIELDS",
    "ENV_ARTIFACT",
    "MEMBER_FIELDS",
    "META_FILE",
    "Artifact",
    "Scored",
    "artifact_dir",
    "cohort_row",
    "feature_row",
    "load_artifact",
    "save_artifact",
    "score_rows",
]
