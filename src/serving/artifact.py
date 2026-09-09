"""M6 · 模型 artifact 的存與載。

`/predict` 不能每次重訓 —— §7.12 採用的 CatBoost 訓練一次約 4 分鐘。所以要有
一份存下來的模型，而**存下來的東西必須帶著它自己的身分證**：

    模型檔      model.cbm（CatBoost 原生格式，載回來不綁 MLflow 版本）
    artifact.json  這個模型是誰、用什麼資料、哪一版程式、在什麼假設下算分

同一份 artifact 有三個消費者：本機服務、HF Spaces Demo、Apr cohort 的 Kaggle
推論管線。三者都不重訓。

## 為什麼 metadata 要記這麼多

一個只有 `model.cbm` 的目錄回答不了任何一個上線前必問的問題：

    這是 T=0 還是 T−7 的模型？   兩者分數差 18.11%，而只有後者能上線（§7.15）
    特徵欄位是哪 61 欄、什麼順序？ CatBoost 的 Pool 依位置認特徵，順序錯了
                                 不報錯只算錯
    p* 是多少？                  它由 cohort 的價格分布推導，不是設定值 ——
                                 服務端拿一列資料算不出來（見下）
    校準了嗎？                   §7.10 決定校準器不上線，這件事要跟著模型走
    這些特徵是哪一版程式算的？    §7.11 的形狀：程式改了、模型還是舊的

所以 metadata 是 artifact 的一部分，不是附註。少了 `cutoff_definition` 這一欄，
一個 T=0 的模型可以被當成能上線的模型部署出去，而 API 的回應看起來完全正常。

## ⚠️ p* 必須存進 artifact，不能在服務端現算

`p* = C_offer / (r_save × LTV_saved)`，而 `LTV_saved` 來自
`monthly_arpu(cohort 的 price_per_day)` —— 那是一個**在 cohort 上擬合出來的
統計量**。服務端只有一位用戶，拿他自己的日均單價去算，等於每個人有一條自己
的門檻線，「名單」這個概念就消失了。

所以 p* 與整組假設在匯出時算好、存進 artifact，服務只負責比較。改業務假設要
重新匯出（快，不必重訓，見 `scripts/export_model.py --reuse`）。

## 三道守門

**一、模型檔的 sha256。** metadata 與模型檔可能不同源：手動複製、兩次匯出交錯、
從別台機器拷一半。摘要不符就拒絕載入 —— 不然服務會用 A 模型算分、用 B 模型的
metadata 回報。

**二、模型檔自己記得的欄名。** 由 `catboost_fitted()` 比對（見該函式）。

**三、特徵程式的邏輯指紋。** artifact 記下匯出當時 `src.features.build` 與
`src.features.logs` 的指紋。載入時與現行程式比對，不符就**拒絕服務**：模型是
舊邏輯的特徵訓練出來的，而服務會用新邏輯把 payload 轉成特徵。這與 M5 的
`cache_provenance()` 是同一個教訓（§7.11），只是這次不只是報告，是擋下來 ——
一個算錯的機率會被拿去決定要不要寄優惠，而它看起來完全正常。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from src.features.build import feature_build_fingerprint
from src.features.logs import log_features_fingerprint
from src.models.candidates import FITTED_LOADERS, Fitted

# artifact 目錄的格式版本。改變 metadata 的必填欄位就要 +1，載入端才有辦法
# 對舊的 artifact 報一句看得懂的話，而不是在某個 KeyError 上炸掉。
ARTIFACT_VERSION = 1

META_FILE = "artifact.json"

# artifact 目錄的位置。Docker / HF Spaces 上沒有 `configs/paths.yaml` 指的
# data_root，需要一個不用寫檔案的覆寫管道（同 `DATA_ROOT` 的理由）。
ENV_ARTIFACT = "MODEL_ARTIFACT"

# 匯出端必須提供的區段。**不給預設值** —— 少了哪一段就代表這個 artifact
# 回答不了上面列的問題，靜靜地補一個空 dict 只會讓它變成一份看起來完整的
# 身分證。
REQUIRED_SECTIONS = ("cohort", "metrics", "assumptions", "hyperparameters", "training", "git")

# cohort 區段裡必填的兩欄。單獨列出來是因為它們決定「這個模型能不能上線」：
# lead_days = 0 的模型在到期日當天評分，挽回優惠來不及寄出（§4.3）。
REQUIRED_COHORT_KEYS = ("train", "eval", "lead_days", "cutoff_definition")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_fingerprints() -> dict[str, str]:
    """現行程式的特徵邏輯指紋。匯出時寫進 artifact，載入時比對。"""
    return {
        "feature_build": feature_build_fingerprint(),
        "log_features": log_features_fingerprint(),
    }


@dataclass(frozen=True)
class Artifact:
    """一份載好的 artifact：模型本體 + 它的身分證。"""

    directory: Path
    meta: dict[str, Any]
    fitted: Fitted
    # 載入時發現但不足以拒絕服務的事（例如 `git_dirty`）。app 會把它們放進
    # `/health`，而不是只寫在啟動日誌裡 —— 沒有人會去讀啟動日誌。
    warnings: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta["features"]["names"])

    @property
    def categorical(self) -> tuple[str, ...]:
        return tuple(self.meta["features"]["categorical"])

    @property
    def uses_log_features(self) -> bool:
        """這個模型吃不吃收聽特徵 —— 決定 payload 要不要組收聽那半段。"""
        return any(name.startswith("log") for name in self.feature_names)

    @property
    def lead_days(self) -> int:
        return int(self.meta["cohort"]["lead_days"])

    @property
    def cutoff_definition(self) -> str:
        return str(self.meta["cohort"]["cutoff_definition"])

    @property
    def eval_cohort(self) -> str:
        return str(self.meta["cohort"]["eval"])

    @property
    def p_star(self) -> float:
        return float(self.meta["assumptions"]["p_star"])

    @property
    def scoring_design(self) -> str:
        """評分規則的**種類**：`expire_date` / `lead_days` / `fixed_score_date`。

        ⚠️ **`cutoff_definition` 帶著日期，種類不帶。** 固定評分日的設計裡，每個
        月的評分日本來就不同（訓練是 `fixed_score_date_20170131`，套用到 4 月是
        `fixed_score_date_20170331`）—— 拿整個字串去比相等，會把「同一個設計的
        下一個月」誤判成「拿錯模型」。

        這個區分是實際踩到的：Kaggle 推論腳本第一版比的是完整字串，於是一份完全
        正確的 artifact 被擋下來。要比的是「這個模型期待的特徵是怎麼算出來的」，
        而那由種類決定（`fixed_score_date` 的模型多一欄 `days_to_expire`）。
        """
        definition = self.cutoff_definition
        if definition.startswith("fixed_score_date"):
            return "fixed_score_date"
        if definition.startswith("expire_date_minus"):
            return "lead_days"
        return "expire_date"

    @property
    def scores_at_expiry(self) -> bool:
        """這個模型是在**到期日當天**評分的嗎。

        ⚠️ **判準是 `cutoff_definition`，不是 `lead_days > 0`。** 固定評分日的
        設計（M6 的 Kaggle 管線）`lead_days` 是 0 而提前天數其實是 1~30 天 ——
        用 lead_days 判斷會把一個能上線的模型標成不能上線，而那種誤報的旗標
        會被學會忽略（M5 的 `git_dirty`）。
        """
        return self.cutoff_definition == "expire_date"

    @property
    def deployable(self) -> bool:
        """在到期日當天評分的模型**不是能上線的模型**（§4.3）。

        挽回優惠要提前寄出才來得及，所以到期日評分的版本無論分數多好都只是
        離線基準。這個旗標會出現在每一筆 `/predict` 的回應裡 —— 部署錯版本是
        一個不會有任何錯誤訊息的錯誤，只能靠回應自己講出來。
        """
        return not self.scores_at_expiry

    def summary(self) -> dict[str, Any]:
        """給 `/health` 與 `/model` 的摘要。不含 61 個欄名那種長清單。"""
        return {
            "artifact_version": self.meta["artifact_version"],
            "created_at": self.meta["created_at"],
            "model": self.meta["model"]["name"],
            "best_iteration": self.meta["model"]["best_iteration"],
            "model_sha256_16": self.meta["model"]["sha256"][:16],
            "n_features": self.meta["features"]["n"],
            "cohort": self.meta["cohort"],
            "metrics": self.meta["metrics"],
            "p_star": self.p_star,
            "calibrated": self.meta["model"]["calibrated"],
            "deployable": self.deployable,
            "git_sha": self.meta["git"]["sha"],
            "warnings": self.warnings,
        }


def artifact_dir(name: str | None = None, *, paths=None) -> Path:
    """artifact 目錄的位置。

    優先序：環境變數 `MODEL_ARTIFACT` > `data_root/artifacts/<name>`。

    環境變數放最前面是因為 Docker 與 HF Spaces 上沒有 data_root —— 那裡的
    artifact 是複製進映像檔的一個目錄，位置與本機無關。
    """
    env = os.environ.get(ENV_ARTIFACT)
    if env:
        return Path(env)
    if name is None:
        raise ValueError(
            f"沒有指定 artifact 名稱，也沒有設定環境變數 {ENV_ARTIFACT}，無從得知要載哪一個模型"
        )
    if paths is None:
        from src.config import load_paths

        paths = load_paths()
    return paths.artifacts / name


def save_artifact(
    fitted: Fitted,
    out_dir: Path,
    *,
    feature_names: list[str],
    categorical: tuple[str, ...],
    meta: dict[str, Any],
) -> dict[str, Any]:
    """把模型與它的身分證寫進 `out_dir`。

    Args:
        fitted: 訓練好的模型。存檔走 `Fitted.save`（見該欄位的註解）。
        out_dir: 目標目錄，不存在就建。**同名會覆寫** —— artifact 是「目前
            這一版模型」，不是版本庫；要留舊版就換一個名字。
        feature_names: 訓練時的欄位清單，**順序即為模型看到的順序**。
        categorical: 類別欄名。
        meta: 匯出端提供的網域資訊，必須含 `REQUIRED_SECTIONS` 那幾段。
            本函式再補上 provenance（版本、時間、模型摘要、特徵清單、指紋）。

    Returns:
        寫出去的完整 metadata。

    Raises:
        KeyError: 少了必填區段或必填欄位。
    """
    missing = [s for s in REQUIRED_SECTIONS if s not in meta]
    if missing:
        raise KeyError(f"artifact metadata 缺少必填區段：{missing}")
    missing_cohort = [k for k in REQUIRED_COHORT_KEYS if k not in meta["cohort"]]
    if missing_cohort:
        raise KeyError(
            f"artifact metadata 的 cohort 區段缺少 {missing_cohort} —— "
            "少了它，一個到期日當天評分的模型可以被當成能上線的模型部署出去。"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = fitted.save(out_dir)
    model_path = out_dir / saved["file"]

    full = {
        "artifact_version": ARTIFACT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": {
            "name": fitted.name,
            "best_iteration": int(fitted.best_iteration),
            # §7.10：校準器不上線。這件事必須跟著模型走，不是只寫在報告裡 ——
            # 服務端要能回答「這個機率校準過嗎」。
            "calibrated": False,
            "sha256": _sha256(model_path),
            **saved,
        },
        "features": {
            # ⚠️ 順序有意義（CatBoost 的 Pool 依位置認特徵），所以存 list 不存 set。
            "names": list(feature_names),
            "n": len(feature_names),
            "categorical": list(categorical),
        },
        "fingerprints": {"code": code_fingerprints(), **meta.get("fingerprints", {})},
        **{k: v for k, v in meta.items() if k != "fingerprints"},
    }

    (out_dir / META_FILE).write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return full


def load_artifact(
    source: Path | str | None = None,
    *,
    name: str | None = None,
    strict: bool = True,
) -> Artifact:
    """載回一份 artifact，並在載入時把該擋的擋下來。

    Args:
        source: artifact 目錄。None 則走 `artifact_dir(name)`。
        name: 當 `source` 是 None 且沒有設環境變數時，要載的 artifact 名稱。
        strict: 特徵程式的邏輯指紋與現行程式不符時是否拒絕載入。
            **預設拒絕**，理由見模組開頭第三道守門。`False` 只用於「我確實
            知道這一版程式與模型不同源」的離線比對，那時不符會進 `warnings`。

    Raises:
        FileNotFoundError: 目錄或檔案不存在。
        ValueError: 版本不符、摘要不符、載入器不認識這個格式。
        RuntimeError: 特徵程式的指紋不符（strict=True）。
    """
    directory = Path(source) if source is not None else artifact_dir(name)
    meta_path = directory / META_FILE
    if not meta_path.exists():
        raise FileNotFoundError(
            f"找不到 {meta_path}。\n"
            "請先匯出模型：uv run python scripts/export_model.py\n"
            f"（或用環境變數 {ENV_ARTIFACT} 指向已有的 artifact 目錄）"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    version = meta.get("artifact_version")
    if version != ARTIFACT_VERSION:
        raise ValueError(
            f"artifact 格式版本是 {version}，本程式支援 {ARTIFACT_VERSION}。請重新匯出。"
        )

    model_meta = meta["model"]
    model_path = directory / model_meta["file"]
    if not model_path.exists():
        raise FileNotFoundError(f"metadata 指向的模型檔不存在：{model_path}")

    # 守門一：模型檔與 metadata 必須同源。
    digest = _sha256(model_path)
    if digest != model_meta["sha256"]:
        raise ValueError(
            f"模型檔的 sha256 與 metadata 不符（{digest[:16]}… vs "
            f"{model_meta['sha256'][:16]}…）。這個目錄裡的模型與身分證不是同一次匯出的 —— "
            "服務會用一個模型算分、用另一個的 metadata 回報。請重新匯出。"
        )

    loader = FITTED_LOADERS.get(model_meta["format"])
    if loader is None:
        raise ValueError(
            f"不認識的模型格式 {model_meta['format']!r}（已知：{sorted(FITTED_LOADERS)}）"
        )

    warnings: list[str] = []

    # 守門三：特徵程式的邏輯指紋。
    stored = meta.get("fingerprints", {}).get("code", {})
    current = code_fingerprints()
    drifted = [k for k, v in current.items() if stored.get(k) != v]
    if drifted:
        detail = "、".join(
            f"{k}（artifact {stored.get(k, '無')} vs 現行 {current[k]}）" for k in drifted
        )
        message = (
            f"特徵程式的邏輯指紋不符：{detail}。"
            "模型是舊邏輯算出來的特徵訓練的，而服務會用現行邏輯轉換 payload —— "
            "欄名與欄數可以完全一樣而語意已經變了。請重新匯出模型。"
        )
        if strict:
            raise RuntimeError(message)
        warnings.append(message)

    if meta["git"].get("dirty"):
        warnings.append("匯出時工作區有未提交的改動 —— 這個模型無法用 git SHA 回溯。")

    fitted = loader(
        model_path,
        feature_names=list(meta["features"]["names"]),
        categorical=tuple(meta["features"]["categorical"]),
        best_iteration=int(model_meta["best_iteration"]),
    )
    return Artifact(directory=directory, meta=meta, fitted=fitted, warnings=warnings)


def assert_features_match(artifact: Artifact, X: pl.DataFrame) -> None:
    """守門：要拿去預測的表，欄位與順序必須與 artifact 記錄的一致。

    `catboost_fitted()` 的轉接層也會擋（那是最後一道），這裡多擋一次是為了
    **錯誤訊息發生在正確的層**：服務組錯 payload 與模型載錯，兩者的處置完全
    不同（前者回 4xx，後者是部署錯誤）。
    """
    expected = artifact.feature_names
    if list(X.columns) != expected:
        raise ValueError(
            "特徵欄位與 artifact 不一致。\n"
            f"  缺少：{sorted(set(expected) - set(X.columns))}\n"
            f"  多出：{sorted(set(X.columns) - set(expected))}\n"
            f"  順序是否相同：{list(X.columns) == expected}"
        )
