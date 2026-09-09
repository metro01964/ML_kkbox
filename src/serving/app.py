"""M6 · FastAPI 服務（SPEC §7 M6：「Docker 起服務，`/predict` 回傳機率與原因」）。

    uv run uvicorn src.serving.app:app --port 8000
    make serve

    GET  /health    服務活著嗎、載的是哪一份 artifact、有沒有警告
    GET  /model     完整的 artifact metadata（含 61 個欄名與業務假設）
    POST /predict   一位用戶的機率 + Top-3 原因碼
    POST /predict/batch  一批用戶 —— 誰該拿挽回優惠（同一條評分路徑）

## 兩種介面，因為它們回答不同的問題

    {"msno": "..."}                這個歷史 cohort 裡的某個人現在的分數與理由
    {"features": {...}, "logs": …}  任意一筆特徵的分數 —— 不需要本機資料

第一種是給營運與 Demo 用的（輸入一個 ID 就看得到結果），第二種才是**真正的
服務介面**：它不依賴任何本機快取，因此在 Docker 與 HF Spaces 上一樣能跑，而
且它逼著呼叫端把「模型看到什麼」講清楚。

⚠️ 兩者都**即時計算**機率與 SHAP，不查預先算好的名單（SPEC §7.14 的要求）。
msno 介面查的是**特徵**，那份特徵來自一個固定 cutoff 的歷史快照 —— 這件事寫在
每一筆回應的 `feature_source` 裡，因為「服務回了一個機率」與「那個機率是用今天
的資料算的」是兩件事，而後者在本專案不成立（沒有線上交易表）。

## 模型只載一次

§7.12 的 CatBoost 訓練一次約 4 分鐘。服務啟動時載一份 artifact（見
`src/serving/artifact.py`），之後每筆請求只做 predict + TreeSHAP。**載不起來就
不要啟動** —— 一個「活著但沒有模型」的服務會在第一筆請求時才失敗，而那通常是
在別人的 Demo 上。

## 這一層不做的事

不算特徵（`src/serving/payload.py` → `build_features()`）、不算原因碼
（`src/serving/score.py` → M5 的同一條路徑）、不決定 p*（匯出時算好存進
artifact）。路由只負責 HTTP 與錯誤碼的分類：

    422  payload 不合（欄位缺少／多出／型別不對）—— 呼叫端的問題
    404  msno 不在這個 cohort 裡
    503  msno 介面需要的本機快取不存在，或設定關閉了它
    500  特徵欄位與 artifact 不一致 —— **部署的問題**，不是呼叫端的
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import polars as pl
import yaml
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import REPO_ROOT, load_paths
from src.serving.artifact import Artifact, load_artifact
from src.serving.examples import OPENAPI_EXAMPLES, build_demo_batch
from src.serving.payload import LOG_FIELDS, feature_row, feature_rows
from src.serving.score import Scored, score_rows

CONFIG_PATH = REPO_ROOT / "configs" / "serving.yaml"

# 批次上限。免費方案是 0.1 vCPU / 512 MB，而 TreeSHAP 的成本隨列數線性成長 ——
# 沒有上限的話一個 50,000 列的請求會讓服務靜默地卡住好幾分鐘，那比回 422 更糟。
MAX_BATCH = 500


def load_serving_config(path=None) -> dict[str, Any]:
    """讀 `configs/serving.yaml`。找不到就失敗，不套用預設值。

    理由同 `src.models.train.load_model_config`：靜默的預設值會讓「我改了設定
    但沒生效」極難察覺。這裡更嚴重 —— 預設值會決定**載哪一個模型**。
    """
    path = path or CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"找不到服務設定檔 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "artifact" not in cfg:
        raise KeyError(f"{path} 缺少 artifact（要載哪一份模型）")
    return cfg


# ---------------------------------------------------------------------------
# 請求與回應
# ---------------------------------------------------------------------------
#
# 欄位刻意逐一寫出來而不是從 `COHORT_FIELDS` 動態生成：OpenAPI 的 /docs 是這個
# 服務的說明書（HF Spaces 上就是靠它試打），動態生成的模型在文件裡只會剩一個
# `object`。兩份清單會不同步，所以 `tests/test_serving.py` 有一條契約測試逐欄
# 比對 —— 加了特徵而忘了改這裡，測試會失敗而不是靜默。


class CohortFeatures(BaseModel):
    """一位用戶在 cutoff 當下的 as-of 交易史與會員屬性。

    等同 `build_cohort()` 輸出的一列（扣掉標籤與推導欄位）。**不是**模型的 61
    個特徵 —— 那些由 `build_features()` 算，服務端不重寫（見 payload.py）。
    """

    model_config = ConfigDict(extra="forbid")

    cutoff: int = Field(description="評分時點，%Y%m%d。T−7 模型請傳「到期日 − 7 天」")
    n_tx: int = Field(description="cutoff 之前的交易筆數")
    first_tx: int = Field(description="最早一筆交易日，%Y%m%d")
    last_tx: int = Field(description="cutoff 之前最後一筆交易日，%Y%m%d")
    n_cancel_hist: int = Field(description="歷史取消次數")
    mean_paid: float | None = Field(default=None, description="歷史平均實付金額")
    # 這六個可以是 null：同一天多筆而取值不唯一時，資料上沒有答案（§7.11）。
    last_is_cancel: int | None = Field(default=None, description="最後一筆是否取消（0/1）")
    last_is_auto_renew: int | None = Field(default=None, description="最後一筆是否自動續訂（0/1）")
    last_actual_amount_paid: float | None = None
    last_plan_list_price: float | None = None
    last_payment_plan_days: int | None = None
    last_payment_method_id: int | None = None
    # members_v3 查不到這個人時全部留空 —— 「查不到」本身有訊號，不要填假值。
    city: int | None = None
    bd: int | None = Field(default=None, description="年齡原始值，離群值不必自己清")
    gender: str | None = Field(default=None, description='"male" / "female" / null')
    registered_via: int | None = None
    registration_init_time: int | None = Field(default=None, description="註冊日，%Y%m%d")


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    msno: str | None = Field(default=None, description="查 cohort 快取取特徵（需本機資料）")
    features: CohortFeatures | None = None
    logs: dict[str, float | None] | None = Field(
        default=None,
        description=(
            "收聽特徵（欄名見 GET /model 的 features.names 裡 log 開頭那些）。"
            "省略等於告訴模型「近 90 天完全沒有收聽紀錄」，回應會帶警告。"
        ),
    )

    @model_validator(mode="after")
    def _exactly_one_interface(self) -> PredictRequest:
        if (self.msno is None) == (self.features is None):
            raise ValueError("請提供 msno 或 features 其中一個（不能同時、也不能都不給）")
        if self.msno is not None and self.logs is not None:
            raise ValueError("msno 介面的收聽特徵來自快取，不接受 logs")
        unknown = sorted(set(self.logs or {}) - set(LOG_FIELDS))
        if unknown:
            raise ValueError(f"logs 有不認識的欄位：{unknown}")
        return self


class Reason(BaseModel):
    rank: int
    group: str
    reason: str
    # 欄名帶單位。叫 impact 會被讀成「機率增加多少」，而那個換算不存在。
    group_shap_log_odds: float
    feature: str
    value: float | None
    horizon: str = Field(description="量測時點：到期日 / 位移 / 快照")
    expiry_dated: bool = Field(description="值落在 cutoff 當天。見回應的 warnings")
    relative_to_top: float
    suppression_reason: str | None = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    msno: str | None
    p_churn: float
    p_star: float = Field(description="投放門檻 C_offer / (r_save × LTV)，由 artifact 帶來")
    above_threshold: bool = Field(description="p > p*，也就是「該不該投放」")
    expected_net: float = Field(description="p × r_save × LTV − C_offer，單位元")
    reasons: list[Reason]
    suppressed: list[Reason] = Field(description="低於呈現門檻的候選，帶原因（同 M5 的稽核表）")
    model: dict[str, Any] = Field(description="這個機率是誰算的：模型、cutoff 定義、可否上線")
    feature_source: dict[str, Any]
    warnings: list[str]


class BatchUser(BaseModel):
    """批次裡的一位用戶。與 `PredictRequest` 的差別只有兩個。

    一是**沒有 msno 介面**：批次的用意是「一批還沒進過系統的人」，而 msno 查的
    是歷史快照裡已經存在的人，兩者是不同的問題。二是多一個 `id`，純粹帶回去讓
    呼叫端對得上自己的名單 —— 它不參與計算，也不會進特徵。
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="呼叫端自己的識別碼，原樣帶回")
    features: CohortFeatures
    logs: dict[str, float | None] | None = None

    @model_validator(mode="after")
    def _known_log_fields(self) -> BatchUser:
        unknown = sorted(set(self.logs or {}) - set(LOG_FIELDS))
        if unknown:
            raise ValueError(f"{self.id} 的 logs 有不認識的欄位：{unknown}")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    users: list[BatchUser] = Field(
        min_length=1,
        max_length=MAX_BATCH,
        description=f"一批到期用戶，最多 {MAX_BATCH} 人",
    )
    reasons: bool = Field(
        default=True,
        description=(
            "要不要附原因碼。**設 false 會快一個量級** —— TreeSHAP 佔整個請求 "
            "97.5% 的成本，而機率、決策與期望淨收益都不需要它。名單畫面靠這個做"
            "兩段式載入：先要機率把表畫出來，再要一次原因碼補上去。"
        ),
    )


class BatchItem(BaseModel):
    """批次結果的一列。刻意比 `PredictResponse` 薄。

    `model` / `feature_source` 這些逐筆重複的欄位提到批次層級去了 —— 500 個人
    各帶一份一模一樣的 metadata，是把回應撐大十倍去講同一句話。
    """

    id: str
    p_churn: float
    above_threshold: bool
    expected_net: float
    reasons: list[Reason]
    warnings: list[str]


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    n: int
    n_targeted: int = Field(description="p > p* 的人數，也就是名單長度")
    p_star: float
    # 前端要能在不重打分數的前提下移動門檻（拖假設滑桿），所以把三個假設帶出來。
    # p* = c_offer / (r_save × ltv_saved)，換算在呼叫端做，模型不必重跑。
    assumptions: dict[str, float]
    budget: float = Field(description="名單人數 × C_offer，也就是要花的錢")
    expected_net_total: float = Field(description="名單上每個人期望淨收益的總和")
    model: dict[str, Any]
    warnings: list[str] = Field(description="整批共通的警告（逐人的在各自的列裡）")
    users: list[BatchItem]


# ---------------------------------------------------------------------------
# 應用程式
# ---------------------------------------------------------------------------


class _State:
    """啟動時載一次的東西。artifact 必載，cohort 索引第一次用到才載。"""

    def __init__(self) -> None:
        self.cfg: dict[str, Any] = {}
        self.artifact: Artifact | None = None
        self.cohort: pl.DataFrame | None = None  # 特徵矩陣
        self.cohort_msno: pl.Series | None = None
        self.cohort_cutoff: pl.Series | None = None
        self.cohort_error: str | None = None


state = _State()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.cfg = load_serving_config()
    state.artifact = load_artifact(
        name=state.cfg["artifact"],
        strict=bool(state.cfg.get("strict_feature_fingerprint", True)),
    )
    summary = state.artifact.summary()
    print(
        f"已載入 artifact {state.artifact.directory}："
        f"{summary['model']} · {summary['n_features']} 特徵 · "
        f"cutoff {summary['cohort']['cutoff_definition']} · "
        f"p* {summary['p_star']:.4f} · 可上線 {summary['deployable']}"
    )
    for w in state.artifact.warnings:
        print(f"⚠️ {w}")
    yield


app = FastAPI(
    title="KKBox 訂閱流失預測",
    version="0.1.0",
    description=(
        "M6 的推論服務。`/predict` 回傳流失機率、是否該投放挽回優惠、以及 Top-3 "
        "中文原因碼（CatBoost 原生 TreeSHAP，每筆請求都驗加總恆等式）。\n\n"
        "⚠️ 模型預測的是「會不會流失」，**不是「投放優惠能不能改變他的行為」** —— "
        "後者需要 uplift modeling 與 A/B 實驗，本資料集不具備實驗組／對照組結構。"
    ),
    lifespan=lifespan,
)


def _artifact() -> Artifact:
    if state.artifact is None:  # pragma: no cover - lifespan 沒跑才會發生
        raise HTTPException(status_code=503, detail="artifact 還沒載入")
    return state.artifact


def _model_info(art: Artifact) -> dict[str, Any]:
    """每一筆回應都帶的「這個機率是誰算的」。

    ⚠️ `deployable` 與 `cutoff_definition` 一定要在這裡，不能只在 /health ——
    別人拿到的是回應，不是啟動日誌。
    """
    return {
        "name": art.meta["model"]["name"],
        "artifact": art.directory.name,
        "model_sha256_16": art.meta["model"]["sha256"][:16],
        "cutoff_definition": art.cutoff_definition,
        "lead_days": art.lead_days,
        "deployable": art.deployable,
        "calibrated": art.meta["model"]["calibrated"],
        "trained_on": art.meta["cohort"]["train"],
        "eval_log_loss": art.meta["metrics"].get("log_loss"),
    }


def _load_cohort() -> None:
    """把 artifact 的評估 cohort 特徵載進記憶體（msno 介面用，只做一次）。

    失敗的原因記在 `state.cohort_error` 而不是每次重試：缺資料是一個不會自己
    好起來的狀況，每筆請求重掃一次 34 GB 只是把它變慢。
    """
    art = _artifact()
    from src.data import COHORTS, build_cohort
    from src.features import build_features, build_log_features

    spec = COHORTS[art.eval_cohort]
    paths = load_paths()
    raw = build_cohort(spec, paths, verbose=False)
    logs = build_log_features(spec, paths, verbose=False) if art.uses_log_features else None
    fs = build_features(raw, logs)
    state.cohort, state.cohort_msno, state.cohort_cutoff = fs.X, fs.msno, raw["cutoff"]


def _cohort_row(msno: str) -> tuple[pl.DataFrame, dict[str, Any], list[str]]:
    art = _artifact()
    if not state.cfg.get("msno_lookup", {}).get("enabled", False):
        raise HTTPException(
            status_code=503,
            detail=(
                "這個部署關閉了 msno 介面（configs/serving.yaml 的 msno_lookup）。"
                "請改用 features payload —— 它不需要本機快取。"
            ),
        )
    if state.cohort is None and state.cohort_error is None:
        try:
            _load_cohort()
        except Exception as e:  # noqa: BLE001 - 任何原因都要變成一句看得懂的 503
            state.cohort_error = f"{type(e).__name__}: {e}"
    if state.cohort is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"msno 介面需要 {art.eval_cohort} cohort 的本機快取，載入失敗："
                f"{state.cohort_error}。請改用 features payload。"
            ),
        )

    idx = state.cohort_msno.to_frame().with_row_index("row").filter(pl.col("msno") == msno)
    if idx.height == 0:
        raise HTTPException(
            status_code=404,
            detail=(
                f"{art.eval_cohort} cohort 裡沒有這個 msno。這份快照是歷史 cohort，不是全體用戶。"
            ),
        )
    row = int(idx["row"][0])
    source = {
        "kind": "cohort_cache",
        "cohort": art.eval_cohort,
        "cutoff": int(state.cohort_cutoff[row]),
    }
    warnings = [
        f"特徵取自 {art.eval_cohort} cohort 的歷史快照（cutoff {source['cutoff']}），"
        "不是以「現在」為時點重算的 —— 真正的部署需要一份線上交易表，本專案沒有。"
        "機率與原因碼仍是即時計算的，不是查表。"
    ]
    return state.cohort[row], source, warnings


@app.get("/health", summary="服務與 artifact 狀態")
def health() -> dict[str, Any]:
    art = _artifact()
    return {
        "status": "ok",
        "artifact": art.directory.name,
        "msno_lookup": {
            "enabled": bool(state.cfg.get("msno_lookup", {}).get("enabled", False)),
            "loaded": state.cohort is not None,
            "error": state.cohort_error,
        },
        **art.summary(),
    }


@app.get("/model", summary="完整的 artifact metadata")
def model_card() -> dict[str, Any]:
    """整份 metadata：特徵欄名、業務假設、指紋、訓練與評估的 cohort。

    ⚠️ 這是**機器可讀的 provenance**，不是 `MODEL_CARD.md`。後者要講的是
    「這個模型不該被用來做什麼」，那不是 JSON 能承載的東西。
    """
    return _artifact().meta


@app.post("/predict", response_model=PredictResponse, summary="流失機率 + Top-3 原因碼")
def predict(
    # `openapi_examples` 而不是 `examples`：前者在 Swagger UI 上是**具名下拉選單**，
    # 後者只是塞進 schema 的無名清單。差別在訪客能不能一眼看懂有哪些情境可選。
    # 內容與合成理由見 src/serving/examples.py。
    req: Annotated[PredictRequest, Body(openapi_examples=OPENAPI_EXAMPLES)],
) -> PredictResponse:
    art = _artifact()
    reasons_cfg = state.cfg.get("reasons", {})

    if req.msno is not None:
        X, source, warnings = _cohort_row(req.msno)
        msno = req.msno
    else:
        msno = "payload"
        try:
            X, warnings = feature_row(
                req.features.model_dump(),
                logs=req.logs,
                with_logs=art.uses_log_features,
                msno=msno,
                # 欄序的唯一來源是 artifact —— payload 的 JSON 沒有順序，而
                # CatBoost 的 Pool 依位置認特徵（見 payload.feature_row）。
                feature_names=art.feature_names,
            )
        except (ValueError, KeyError, AssertionError) as e:
            # 呼叫端給的東西不合（含紅線 2 的守門：收聽紀錄晚於 cutoff）。
            raise HTTPException(status_code=422, detail=str(e)) from e
        source = {"kind": "payload", "cutoff": req.features.cutoff}

    try:
        scored: Scored = score_rows(
            art,
            X,
            msno=[msno],
            top_k=int(reasons_cfg.get("top_k", 3)),
            min_relative=float(reasons_cfg.get("min_relative", 0.05)),
            warnings=warnings,
        )[0]
    except ValueError as e:
        # 特徵欄位與 artifact 不一致 —— 這是部署錯了（artifact 與程式不同版），
        # 不是呼叫端的錯，所以是 500 而不是 422。
        raise HTTPException(status_code=500, detail=str(e)) from e

    return PredictResponse(
        msno=req.msno,
        p_churn=scored.p_churn,
        p_star=scored.p_star,
        above_threshold=scored.above_threshold,
        expected_net=scored.expected_net,
        reasons=[Reason(**r) for r in scored.reasons],
        suppressed=[Reason(**r) for r in scored.suppressed],
        model=_model_info(art),
        feature_source=source,
        warnings=scored.warnings,
    )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    summary="一批到期用戶 → 誰該拿挽回優惠",
)
def predict_batch(req: BatchRequest) -> BatchResponse:
    """把一批人一次評完，回傳每個人的機率、決策與原因碼。

    ## 為什麼要有這一支

    `/predict` 回答的是「這個人」，而營運要的是「這一批人裡誰該拿優惠」。逐一
    打 `/predict` 也能得到同一份名單，但那是 N 次 HTTP、N 次模型載入檢查、N 次
    TreeSHAP 初始化 —— 而 CatBoost 一次評 500 列與評 1 列的差別只有毫秒。

    ## 分數與單筆完全同源

    特徵組裝走 `feature_row()`、評分與原因碼走 `score_rows()`，與 `/predict`
    是**同一條路徑**（理由見 score.py 開頭）。批次不是另一套實作，只是把
    `score_rows()` 本來就支援的多列用起來 —— 那個函式從第一版就是收 N 列的。

    ## 錯誤碼

    任何一位的 payload 不合就整批 422，並指出是誰。批次成功一半是最難除錯的
    狀態：呼叫端拿到一份少了幾個人的名單，而少的是誰要自己比對。
    """
    art = _artifact()
    reasons_cfg = state.cfg.get("reasons", {})

    ids = [u.id for u in req.users]
    try:
        X, payload_warnings = feature_rows(
            [{"id": u.id, "features": u.features.model_dump(), "logs": u.logs} for u in req.users],
            with_logs=art.uses_log_features,
            feature_names=art.feature_names,
        )
    except (ValueError, KeyError, AssertionError) as e:
        # 指出是哪一位 —— 「批次裡有一筆不合」而不說是誰，等於沒講。
        # `cohort_row` / `logs_row` 的訊息帶欄名，`feature_rows` 帶 id。
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        scored = score_rows(
            art,
            X,
            msno=ids,
            top_k=int(reasons_cfg.get("top_k", 3)),
            min_relative=float(reasons_cfg.get("min_relative", 0.05)),
            with_reasons=req.reasons,
            # 逐人的 payload 警告不能當成整批共通的（那會讓沒問題的人也掛上
            # 別人的警告），所以這裡不傳，下面逐列併回去。
        )
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    users = [
        BatchItem(
            id=s.msno,
            p_churn=s.p_churn,
            above_threshold=s.above_threshold,
            expected_net=s.expected_net,
            reasons=[Reason(**r) for r in s.reasons],
            warnings=[*w, *s.warnings],
        )
        for w, s in zip(payload_warnings, scored, strict=True)
    ]

    a = art.meta["assumptions"]
    c_offer = float(a["c_offer"])
    targeted = [u for u in users if u.above_threshold]

    batch_warnings = list(art.warnings)
    n_no_logs = sum(1 for u in req.users if not u.logs)
    if art.uses_log_features and n_no_logs:
        batch_warnings.append(
            f"{n_no_logs} 位沒有提供收聽紀錄。省略不是中性預設 —— 那等於主張這些人"
            "近 90 天沒聽過歌（訓練資料裡 18.0% 的用戶確實如此）。"
        )

    return BatchResponse(
        n=len(users),
        n_targeted=len(targeted),
        p_star=art.p_star,
        assumptions={
            "c_offer": c_offer,
            "r_save": float(a["r_save"]),
            "ltv_saved": float(a["ltv_saved"]),
        },
        budget=round(len(targeted) * c_offer, 1),
        # 只加名單上的人。全體加總會把「不投放的人期望淨收益是負的」也算進來，
        # 而那些負數不會發生 —— 沒投放就沒有成本，也沒有挽回。
        expected_net_total=round(sum(u.expected_net for u in targeted), 1),
        model=_model_info(art),
        warnings=batch_warnings,
        users=users,
    )


# ---------------------------------------------------------------------------
# Demo 頁
# ---------------------------------------------------------------------------
#
# `/docs` 是給看得懂 API 的人用的；這一頁是給其他人用的。兩個都留著，README
# 並列兩個連結。
#
# 刻意不引入前端框架，也不掛 StaticFiles —— 就一個檔案、原生 JS、行內 SVG 畫圖。
# 理由是這個服務跑在 0.1 vCPU / 512 MB 的免費方案上，而一頁靜態 HTML 的維護成本
# 與失敗模式都遠低於一套建置流程。沒有外部 CDN，斷網也不會少半個字。

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEMO_CURVE_PATH = REPO_ROOT / "deploy" / "demo_curve.json"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/demo/examples", include_in_schema=False)
def demo_examples() -> dict[str, Any]:
    """`/docs` 下拉選單的那五個情境，給 Demo 頁拿去畫卡片與預填欄位。

    直接回 `OPENAPI_EXAMPLES`，所以兩邊永遠是同一份 —— 前端另外抄一份範例
    是這類頁面最典型的走鐘方式。
    """
    return OPENAPI_EXAMPLES


@app.get("/demo/batch", include_in_schema=False)
def demo_batch() -> dict[str, Any]:
    """Demo 頁的那一批到期用戶（合成），**未評分**。

    只回 payload，機率由頁面自己 POST 到 `/predict/batch` 取得 —— 這一頁上沒有
    任何預先算好的數字（SPEC §7.14）。回一份算好的名單會快一點，但那樣示範的
    就不是模型，是一個 JSON 檔。
    """
    users = build_demo_batch()
    return {
        "users": users,
        "note": (
            "這 50 位是合成資料，而且是**刻意加重風險族群**的抽樣，"
            "不是母體分佈 —— 真實 Mar cohort 依 p* 只有 3.83% 上榜。"
            "照真實比例抽 50 人平均只會有 2 人越過門檻，那條線就看不出來了。"
        ),
    }


@app.get("/demo/curve", include_in_schema=False)
def demo_curve() -> dict[str, Any]:
    """營運視角的聚合統計，由 `scripts/demo_curve.py` 預先算好。

    不即時計算：那需要整個 cohort 的特徵（3.0 GB 快取），而那份不進映像檔
    （見 deploy/serving.yaml 的授權說明）。這裡只有曲線與幾個彙總數字。
    """
    if not DEMO_CURVE_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"找不到 {DEMO_CURVE_PATH.name}。"
                "請先在有資料的機器上執行 `uv run python scripts/demo_curve.py`。"
            ),
        )
    return json.loads(DEMO_CURVE_PATH.read_text(encoding="utf-8"))
