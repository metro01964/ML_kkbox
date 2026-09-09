"""M6 · 服務層的測試（artifact 存載、payload 轉換、/predict）。

全部用手刻的合成 cohort 訓練一個 5 輪的迷你 CatBoost，因此標 nodata、在 CI 上
會實際執行。真實資料上的分數對照走 `scripts/export_model.py`（它自己會載回來
對答案）。

守的是四類「不會報錯的錯」：

    存檔／載入不是同一棵樹          載回來的機率必須逐列相同
    metadata 與模型檔不同源          改一個 byte 就要拒絕載入
    服務端自己算特徵                payload 路徑的輸出必須等於 build_features()
    payload 欄名打錯變成缺失        多一欄、少一欄都要報，不可靜默
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl
import pytest

from src.data import MAR, cutoff_definition
from src.features import build_features, build_log_features
from src.models.candidates import fit_catboost
from src.serving import artifact as artifact_mod
from src.serving.artifact import ARTIFACT_VERSION, load_artifact, save_artifact
from src.serving.payload import COHORT_FIELDS, LOG_FIELDS, cohort_row, feature_row
from src.serving.score import score_rows
from tests.conftest import NODATA, SLOW, make_synthetic_cohort

# 迷你模型的參數。5 輪、深度 2 —— 這些測試驗的是管線，不是分數。
TINY_PARAMS = {
    "loss_function": "Logloss",
    "depth": 2,
    "random_seed": 42,
    "allow_writing_files": False,
}
TINY_TRAINING = {"num_boost_round": 5, "early_stopping_rounds": 5}


def _synthetic_features():
    """合成 cohort → 特徵矩陣。**不含收聽特徵**（那一半由 payload 測試單獨驗）。"""
    return build_features(make_synthetic_cohort())


def _tiny_fitted(fs=None):
    fs = fs or _synthetic_features()
    # 8 列全部拿來訓練，early stopping 用同一份 —— 這裡不評估任何東西。
    return fit_catboost(fs, fs, TINY_PARAMS, TINY_TRAINING)


def _meta(fs, *, lead_days: int = 0) -> dict:
    """一份最小但**合法**的 metadata：必填區段都在。"""
    return {
        "cohort": {
            "train": "synthetic",
            "eval": "synthetic",
            "lead_days": lead_days,
            "cutoff_definition": "expire_date"
            if not lead_days
            else f"expire_date_minus_{lead_days}d",
            "n_train": fs.X.height,
        },
        "metrics": {"log_loss": 0.5},
        "assumptions": {"r_save": 0.15, "c_offer": 150.0, "ltv_saved": 2000.0, "p_star": 0.5},
        "hyperparameters": TINY_PARAMS,
        "training": TINY_TRAINING,
        "git": {"sha": "0" * 40, "dirty": False},
    }


@pytest.fixture
def saved_artifact(tmp_path):
    """存一份 artifact，回傳 (目錄, 訓練時的 Fitted, FeatureSet)。"""
    fs = _synthetic_features()
    fitted = _tiny_fitted(fs)
    save_artifact(
        fitted,
        tmp_path / "tiny",
        feature_names=list(fs.X.columns),
        categorical=fs.categorical,
        meta=_meta(fs),
    )
    return tmp_path / "tiny", fitted, fs


# ---------------------------------------------------------------------------
# artifact
# ---------------------------------------------------------------------------


@NODATA
def test_reloaded_model_predicts_bit_for_bit_the_same(saved_artifact):
    """存檔／載入是無損的，所以容差是 0 而不是「差不多」。

    這條抓的是整個 M6 最貴的一種錯：服務用一個與離線不同的模型算分。它不會
    報錯 —— 機率照樣在 0~1 之間、原因碼照樣通順。
    """
    directory, fitted, fs = saved_artifact
    art = load_artifact(directory)
    before = np.asarray(fitted.predict(fs.X), dtype=np.float64)
    after = np.asarray(art.fitted.predict(fs.X), dtype=np.float64)
    assert np.array_equal(before, after)


@NODATA
def test_reloaded_model_gives_the_same_shap_values(saved_artifact):
    """歸因也必須是同一棵樹算的 —— 否則機率對、原因碼講別人的事。"""
    directory, fitted, fs = saved_artifact
    art = load_artifact(directory)
    assert np.array_equal(fitted.shap_values(fs.X), art.fitted.shap_values(fs.X))


@NODATA
def test_tampered_model_file_is_refused(saved_artifact):
    """模型檔與 metadata 不同源時要拒絕載入。

    ⚠️ 沒有這道守門，服務會用 A 模型算分、用 B 模型的 metadata 回報 ——
    而 /health 看起來完全正常。
    """
    directory, _, _ = saved_artifact
    model_file = directory / "model.cbm"
    model_file.write_bytes(model_file.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="sha256"):
        load_artifact(directory)


@NODATA
def test_stale_feature_code_is_refused(saved_artifact, monkeypatch):
    """特徵程式改了而模型沒重訓 —— §7.11 的形狀，預設拒絕服務。"""
    directory, _, _ = saved_artifact
    monkeypatch.setattr(
        artifact_mod,
        "code_fingerprints",
        lambda: {"feature_build": "deadbeef", "log_features": "x"},
    )
    with pytest.raises(RuntimeError, match="邏輯指紋"):
        load_artifact(directory)
    # strict=False 時照樣載，但不符會進 warnings 而不是消失。
    art = load_artifact(directory, strict=False)
    assert any("邏輯指紋" in w for w in art.warnings)


@NODATA
def test_metadata_must_declare_the_cutoff_definition(tmp_path):
    """少了 cutoff_definition 就存不出去。

    這一欄決定「這個模型能不能上線」（§4.3）。給它一個預設值等於讓一個到期日
    當天評分的模型可以被當成能上線的模型部署出去。
    """
    fs = _synthetic_features()
    fitted = _tiny_fitted(fs)
    meta = _meta(fs)
    del meta["cohort"]["cutoff_definition"]
    with pytest.raises(KeyError, match="cutoff_definition"):
        save_artifact(
            fitted,
            tmp_path / "bad",
            feature_names=list(fs.X.columns),
            categorical=fs.categorical,
            meta=meta,
        )


@NODATA
def test_version_bump_is_not_silently_accepted(saved_artifact):
    """舊版 artifact 要報一句看得懂的話，不是在某個 KeyError 上炸掉。"""
    directory, _, _ = saved_artifact
    meta_path = directory / "artifact.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["artifact_version"] = ARTIFACT_VERSION + 1
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="格式版本"):
        load_artifact(directory)


@NODATA
def test_a_reordered_feature_frame_is_refused(saved_artifact):
    """CatBoost 的 Pool 依位置認特徵：欄序錯了不報錯只算錯，所以要擋。"""
    directory, _, fs = saved_artifact
    art = load_artifact(directory)
    shuffled = fs.X.select(list(reversed(fs.X.columns)))
    with pytest.raises(ValueError, match="順序|一致"):
        art.fitted.predict(shuffled)


@NODATA
def test_deployable_is_false_for_the_expiry_date_model(saved_artifact, tmp_path):
    """lead_days = 0 的模型不是能上線的模型（§4.3），旗標要說得出來。"""
    directory, fitted, fs = saved_artifact
    assert load_artifact(directory).deployable is False

    save_artifact(
        fitted,
        tmp_path / "t7",
        feature_names=list(fs.X.columns),
        categorical=fs.categorical,
        meta=_meta(fs, lead_days=7),
    )
    art = load_artifact(tmp_path / "t7")
    assert art.deployable is True
    assert art.cutoff_definition == "expire_date_minus_7d"


@NODATA
def test_cutoff_definition_has_one_source():
    """M5 的 manifest 與 M6 的 artifact 必須寫出同一個字串。"""
    assert cutoff_definition(MAR) == "expire_date"
    from src.data import MAR_T7

    assert cutoff_definition(MAR_T7) == "expire_date_minus_7d"


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------


def _payload_from(cohort: pl.DataFrame, row: int) -> dict:
    """把合成 cohort 的一列翻成 payload（扣掉標籤與推導欄位）。"""
    record = cohort[row].to_dicts()[0]
    return {name: record[name] for name in COHORT_FIELDS}


@NODATA
def test_payload_path_reproduces_build_features_exactly():
    """服務端不重寫特徵工程 —— 這條測試就是那句話的證明。

    payload 走 `feature_row()`，離線走 `build_features()`。兩者若不同，線上與
    離線的機率就不同，而 API 會回 200。
    """
    cohort = make_synthetic_cohort()
    offline = build_features(cohort)
    for row in range(cohort.height):
        online, _ = feature_row(_payload_from(cohort, row), with_logs=False)
        expected = offline.X[row]
        assert online.columns == expected.columns
        # frame_equal 會連 dtype 一起比 —— 型別漂掉（Int64 vs Float64）也算不同。
        assert online.equals(expected), f"第 {row} 列不同：\n{online}\n{expected}"


def _synthetic_logs(cohort: pl.DataFrame) -> pl.DataFrame:
    """一張最小的收聽特徵表，欄序**刻意不是**欄名排序。

    離線那一邊的欄序來自 `build_log_features()` 的輸出，payload 那一邊來自一個
    JSON 物件（沒有順序）。兩者不同才是常態 —— 這個函式把那個常態做出來。
    """
    columns = ["msno", "cutoff", *reversed(LOG_FIELDS)]
    return pl.DataFrame(
        {
            "msno": cohort["msno"],
            "cutoff": cohort["cutoff"],
            **{
                name: pl.Series(name, [1.0] * cohort.height, dtype=pl.Float64)
                for name in LOG_FIELDS
            },
        }
    ).select(columns)


@NODATA
def test_payload_columns_are_ordered_by_the_artifact_not_by_the_payload():
    """⚠️ 這條測試存在的原因是它抓到過一個真的 bug（真實資料的服務煙霧測試）。

    收聽特徵在離線那邊的欄序來自 parquet，在 payload 那邊來自 JSON —— 兩者不同。
    **而 CatBoost 的 Pool 依位置認特徵**：集合相同、順序不同，每一欄的值都餵給
    了別的特徵，模型照樣回一個 0~1 的機率、原因碼照樣通順。

    當時兩層守門的訊息都是「缺少 []、多出 []」，因為它們比的是集合。
    """
    cohort = make_synthetic_cohort()
    logs = _synthetic_logs(cohort)
    offline = build_features(cohort, logs)
    payload_logs = {name: 1.0 for name in LOG_FIELDS}

    unordered, _ = feature_row(_payload_from(cohort, 0), logs=payload_logs)
    assert set(unordered.columns) == set(offline.X.columns)
    # 沒有給 artifact 的清單時，順序本來就不保證與訓練時相同 —— 這正是那個 bug。
    ordered, _ = feature_row(
        _payload_from(cohort, 0), logs=payload_logs, feature_names=list(offline.X.columns)
    )
    assert ordered.columns == list(offline.X.columns)
    assert ordered.equals(offline.X[0])


@SLOW
def test_payload_path_matches_real_cohort_rows(paths, mar_cohort):
    """真實資料上的線上／離線等價 —— 合成資料驗不到的那一半。

    合成 cohort 沒有收聽特徵，所以欄序那個 bug 在 nodata 測試裡看不見（它是在
    真實資料的服務煙霧測試上抓到的）。這一條拿三位真人走完整條 payload 路徑，
    與離線的 `build_features()` 逐格比對，含 dtype、null 與**欄序**。
    """
    logs = build_log_features(MAR, paths, verbose=False)
    offline = build_features(mar_cohort, logs)
    names = list(offline.X.columns)

    have_logs = set(logs["msno"].to_list())
    msnos = mar_cohort["msno"].to_list()
    picked = [i for i, m in enumerate(msnos[:5000]) if m in have_logs][:3]
    assert picked, "前 5,000 位裡應該找得到有收聽紀錄的人"

    for i in picked:
        record = mar_cohort[i].to_dicts()[0]
        log_row = logs.filter(pl.col("msno") == msnos[i]).to_dicts()[0]
        online, _ = feature_row(
            {name: record[name] for name in COHORT_FIELDS},
            # 只帶有值的欄位：沒帶等於 null，與離線那一格的 null 同義。
            logs={k: v for k, v in log_row.items() if k in LOG_FIELDS and v is not None},
            feature_names=names,
            msno=msnos[i],
        )
        assert online.equals(offline.X[i]), f"第 {i} 列（{msnos[i][:12]}…）線上與離線不同"


@NODATA
def test_a_feature_set_mismatch_blames_the_artifact_not_the_caller():
    """欄位集合對不上是**匯出**的問題（程式與模型不同版），不是 payload 的問題。"""
    cohort = make_synthetic_cohort()
    with pytest.raises(ValueError, match="artifact 需要重新匯出"):
        feature_row(_payload_from(cohort, 0), with_logs=False, feature_names=["tenure_days"])


@NODATA
def test_a_misspelled_field_is_reported_not_ignored():
    """打錯的欄名若被忽略，那一欄就變成缺失 —— 模型會給一個很合理的錯答案。"""
    cohort = make_synthetic_cohort()
    payload = _payload_from(cohort, 0)
    payload["last_is_cancle"] = 1  # 打錯字
    with pytest.raises(ValueError, match="不認識的欄位"):
        cohort_row(payload)

    del payload["last_is_cancle"]
    del payload["n_tx"]
    with pytest.raises(ValueError, match="缺少欄位"):
        cohort_row(payload)


@NODATA
def test_payload_may_not_supply_the_label_or_derived_fields():
    """`is_churn` 是標籤、`in_members` 是推導欄位，兩者都不該由呼叫端給。"""
    cohort = make_synthetic_cohort()
    for field in ("is_churn", "in_members"):
        payload = _payload_from(cohort, 0) | {field: 1}
        with pytest.raises(ValueError, match=field):
            cohort_row(payload)


@NODATA
def test_in_members_is_derived_from_city_the_same_way_as_the_cohort_builder():
    """u5 不在 members_v3（city 為 null），推導欄位要跟 build_cohort 一致。"""
    cohort = make_synthetic_cohort()
    for row in range(cohort.height):
        payload = _payload_from(cohort, row)
        got = cohort_row(payload)["in_members"][0]
        assert got == (payload["city"] is not None)


@NODATA
def test_omitting_listening_features_is_reported_as_a_claim():
    """省略 logs 等於告訴模型「90 天沒聽歌」，那是一個主張，不是中性預設。"""
    cohort = make_synthetic_cohort()
    X, warnings = feature_row(_payload_from(cohort, 0), logs=None, with_logs=True)
    assert any("沒有收聽紀錄" in w for w in warnings)
    # 收聽欄位都在（模型要幾欄就是幾欄），而 log_has_logs 是 0 ——
    # 那正是 `_attach_logs()` 對「不在收聽特徵表裡」的人的處理。
    assert set(LOG_FIELDS) <= set(X.columns)
    assert X["log_has_logs"][0] == 0.0
    assert X["log30_secs"][0] is None


@NODATA
def test_listening_features_after_the_cutoff_are_refused():
    """紅線 2 的守門要在服務的邊界上也成立。

    `log_min_days_before < 0` 代表那筆日誌發生在到期日之後 —— 到期後的收聽
    行為是結果不是原因。payload 帶著它進來時要被擋下，而不是變成一個特徵。
    """
    cohort = make_synthetic_cohort()
    with pytest.raises(AssertionError, match="紅線 2"):
        feature_row(
            _payload_from(cohort, 0),
            logs={"log_min_days_before": -3.0, "log_has_logs": 1.0},
            with_logs=True,
        )


@NODATA
def test_hidden_member_snapshot_is_reported():
    """註冊日晚於 cutoff 時，會員屬性會退回缺失 —— 靜默的話回應會自相矛盾。"""
    cohort = make_synthetic_cohort()
    # 合成資料裡刻意有這樣一列（見 conftest 的說明）。
    rows = [
        r
        for r in range(cohort.height)
        if cohort["registration_init_time"][r] is not None
        and cohort["registration_init_time"][r] > cohort["cutoff"][r]
    ]
    assert rows, "合成 cohort 應該要有一列註冊日晚於 cutoff"
    _, warnings = feature_row(_payload_from(cohort, rows[0]), with_logs=False)
    assert any("註冊日" in w for w in warnings)


# ---------------------------------------------------------------------------
# 評分與原因碼
# ---------------------------------------------------------------------------


@NODATA
def test_score_rows_verifies_the_local_accuracy_identity(saved_artifact):
    """每一筆請求都驗加總恆等式：歸因必須加得回這個模型的預測。"""
    directory, _, fs = saved_artifact
    art = load_artifact(directory)
    scored = score_rows(art, fs.X, msno=list(fs.msno), verify=True)
    assert len(scored) == fs.X.height
    for s in scored:
        assert 0.0 <= s.p_churn <= 1.0
        assert s.above_threshold == (s.p_churn > s.p_star)
        # 原因碼依貢獻遞減，而且只有推高風險的方向。
        shaps = [r["group_shap_log_odds"] for r in s.reasons]
        assert shaps == sorted(shaps, reverse=True)
        assert all(v > 0 for v in shaps)
        assert all(r["reason"] for r in s.reasons)


@NODATA
def test_expiry_dated_warning_only_fires_on_the_t0_model(saved_artifact, tmp_path):
    """`expiry_dated` 在 T−7 的 artifact 上不代表「用不了」，不可誤報。

    誤報的旗標會被學會忽略 —— M5 在 `git_dirty` 上踩過這個坑。
    """
    directory, fitted, fs = saved_artifact
    save_artifact(
        fitted,
        tmp_path / "t7",
        feature_names=list(fs.X.columns),
        categorical=fs.categorical,
        meta=_meta(fs, lead_days=7),
    )
    t7 = score_rows(load_artifact(tmp_path / "t7"), fs.X, verify=False)
    assert not any("不可沿用" in w for s in t7 for w in s.warnings)


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------


@pytest.fixture
def client(saved_artifact, monkeypatch):
    """起一個載入迷你 artifact 的測試用服務（msno 介面關閉 —— 沒有本機資料）。"""
    pytest.importorskip("httpx", reason="starlette 的 TestClient 需要 httpx")
    from fastapi.testclient import TestClient

    from src.serving import app as app_mod

    directory, _, _ = saved_artifact
    monkeypatch.setenv("MODEL_ARTIFACT", str(directory))
    monkeypatch.setattr(
        app_mod,
        "load_serving_config",
        lambda path=None: {
            "artifact": "tiny",
            "msno_lookup": {"enabled": False},
            "reasons": {"top_k": 3, "min_relative": 0.05},
            "strict_feature_fingerprint": True,
        },
    )
    with TestClient(app_mod.app) as c:
        yield c


@NODATA
def test_health_reports_which_model_is_loaded(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["cohort"]["cutoff_definition"] == "expire_date"
    # T=0 的模型不可上線，而這件事要在 API 上看得到，不是只寫在報告裡。
    assert body["deployable"] is False


@NODATA
def test_predict_returns_probability_and_reasons(client):
    cohort = make_synthetic_cohort()
    body = client.post("/predict", json={"features": _payload_from(cohort, 1)}).json()
    assert 0.0 <= body["p_churn"] <= 1.0
    assert body["above_threshold"] == (body["p_churn"] > body["p_star"])
    assert body["model"]["cutoff_definition"] == "expire_date"
    assert body["model"]["deployable"] is False
    assert body["model"]["calibrated"] is False
    assert body["feature_source"] == {"kind": "payload", "cutoff": cohort["cutoff"][1]}
    for r in body["reasons"]:
        assert r["reason"] and r["horizon"]


@NODATA
def test_predict_rejects_both_interfaces_at_once(client):
    cohort = make_synthetic_cohort()
    r = client.post("/predict", json={"msno": "u0", "features": _payload_from(cohort, 0)})
    assert r.status_code == 422


@NODATA
def test_predict_rejects_unknown_fields(client):
    cohort = make_synthetic_cohort()
    payload = _payload_from(cohort, 0) | {"last_is_cancle": 1}
    assert client.post("/predict", json={"features": payload}).status_code == 422
    # 收聽特徵的欄名也一樣不可打錯。
    r = client.post(
        "/predict", json={"features": _payload_from(cohort, 0), "logs": {"log30_secz": 1.0}}
    )
    assert r.status_code == 422


@NODATA
def test_msno_interface_says_it_is_unavailable_rather_than_returning_nothing(client):
    """設定關閉時要回 503 並說清楚，不是安靜地回一個空結果。"""
    r = client.post("/predict", json={"msno": "u0"})
    assert r.status_code == 503
    assert "msno" in r.json()["detail"]


@NODATA
def test_request_model_covers_exactly_the_payload_fields():
    """pydantic 的欄位清單與 `COHORT_FIELDS` 必須逐欄相同。

    /docs 是這個服務的說明書，所以欄位是手寫的（動態生成在文件裡只剩一個
    `object`）。兩份清單就有不同步的可能，這條測試把它釘住。
    """
    from src.serving.app import CohortFeatures

    assert set(CohortFeatures.model_fields) == set(COHORT_FIELDS)


# ---------------------------------------------------------------------------
# /predict/batch
# ---------------------------------------------------------------------------
#
# 批次守的是一件事：**它不可以是第二套實作**。分數、原因碼、門檻只要有一條路徑
# 是批次自己寫的，線上就會有兩份答案，而兩邊都不會報錯（同 score.py 開頭的理由）。


@NODATA
def test_batch_scores_are_identical_to_scoring_one_by_one(client):
    """同一份 payload，走批次與走單筆必須**逐位元相同**。

    這是這一組測試的核心。批次只是把 `score_rows()` 本來就支援的多列用起來，
    一旦有人為了效能在批次路徑上抄一份簡化的評分，這裡就會失敗。
    """
    cohort = make_synthetic_cohort()
    users = [{"id": f"u{i}", "features": _payload_from(cohort, i)} for i in range(4)]

    batch = client.post("/predict/batch", json={"users": users}).json()
    assert batch["n"] == 4
    for u, got in zip(users, batch["users"], strict=True):
        one = client.post("/predict", json={"features": u["features"]}).json()
        assert got["id"] == u["id"]
        assert got["p_churn"] == one["p_churn"]
        assert got["above_threshold"] == one["above_threshold"]
        assert got["expected_net"] == one["expected_net"]
        assert [r["reason"] for r in got["reasons"]] == [r["reason"] for r in one["reasons"]]


@NODATA
def test_batch_preserves_the_caller_order(client):
    """回傳順序必須與送出順序相同 —— 呼叫端要靠位置對回自己的名單。"""
    cohort = make_synthetic_cohort()
    ids = ["z", "a", "m", "b"]
    users = [{"id": i, "features": _payload_from(cohort, n)} for n, i in enumerate(ids)]
    body = client.post("/predict/batch", json={"users": users}).json()
    assert [u["id"] for u in body["users"]] == ids


@NODATA
def test_one_bad_row_fails_the_whole_batch_and_names_it(client):
    """半成功是最難除錯的狀態：呼叫端拿到一份少了幾個人的名單，而少的是誰要自己比對。"""
    cohort = make_synthetic_cohort()
    users = [
        {"id": "ok", "features": _payload_from(cohort, 0)},
        {"id": "broken", "features": _payload_from(cohort, 1), "logs": {"log30_secz": 1.0}},
    ]
    r = client.post("/predict/batch", json={"users": users})
    assert r.status_code == 422
    assert "broken" in json.dumps(r.json(), ensure_ascii=False)


@NODATA
def test_batch_size_is_capped(client):
    """免費方案是 0.1 vCPU，而 TreeSHAP 隨列數線性成長 —— 沒有上限的話一個大請求
    會讓服務靜默地卡住好幾分鐘，那比回 422 更糟。
    """
    from src.serving.app import MAX_BATCH

    cohort = make_synthetic_cohort()
    one = _payload_from(cohort, 0)
    over = [{"id": f"u{i}", "features": one} for i in range(MAX_BATCH + 1)]
    assert client.post("/predict/batch", json={"users": over}).status_code == 422
    assert client.post("/predict/batch", json={"users": []}).status_code == 422


@NODATA
def test_batch_totals_only_count_the_targeted(client):
    """預算與期望淨收益只加名單上的人。

    全體加總會把「不投放的人期望淨收益是負的」也算進來，而那些負數不會發生 ——
    沒投放就沒有成本，也沒有挽回。這是一個會讓 ROI 看起來慘不忍睹的錯誤。
    """
    cohort = make_synthetic_cohort()
    users = [{"id": f"u{i}", "features": _payload_from(cohort, i)} for i in range(4)]
    body = client.post("/predict/batch", json={"users": users}).json()

    on = [u for u in body["users"] if u["above_threshold"]]
    assert body["n_targeted"] == len(on)
    assert body["budget"] == pytest.approx(len(on) * body["assumptions"]["c_offer"], abs=0.1)
    assert body["expected_net_total"] == pytest.approx(sum(u["expected_net"] for u in on), abs=0.1)


@NODATA
def test_batch_reports_the_assumptions_so_the_caller_can_move_the_threshold(client):
    """Demo 頁靠這三個假設在前端重算 p*（拖滑桿不重打 API），所以它們必須帶出來，
    而且要與 artifact 算出來的 p* 對得起來 —— 對不上就是那一頁在拿假數字畫圖。
    """
    cohort = make_synthetic_cohort()
    body = client.post(
        "/predict/batch", json={"users": [{"id": "u0", "features": _payload_from(cohort, 0)}]}
    ).json()
    a = body["assumptions"]
    assert a["c_offer"] / (a["r_save"] * a["ltv_saved"]) == pytest.approx(body["p_star"], abs=1e-9)


@NODATA
def test_batch_rejects_the_msno_interface(client):
    """批次是「一批還沒進系統的人」，msno 查的是歷史快照裡已經存在的人。"""
    cohort = make_synthetic_cohort()
    users = [{"id": "u0", "features": _payload_from(cohort, 0), "msno": "u0"}]
    assert client.post("/predict/batch", json={"users": users}).status_code == 422


# ---------------------------------------------------------------------------
# 批次的兩個效能修正
# ---------------------------------------------------------------------------
#
# 兩者都是「快但必須完全等價」的改動，所以守的是等價而不是速度。速度會隨機器變，
# 等價不會 —— 而一條快了十倍卻悄悄回不同機率的路徑，畫面上完全看不出來。


@NODATA
def test_feature_rows_matches_feature_row_cell_for_cell():
    """一次組 N 列必須與逐列組再疊起來逐格相同。

    `build_features()` 的成本大部分是固定的（建 lazy 計畫、join、算衍生欄），
    所以呼叫一次而不是 N 次快了 4.5 倍。但那個改寫只有在輸出完全相同時才成立
    —— 否則批次名單與單筆查詢會給出不同的機率。
    """
    from src.serving.payload import feature_rows

    cohort = make_synthetic_cohort()
    users = [
        {"id": f"u{i}", "features": _payload_from(cohort, i), "logs": None}
        for i in range(cohort.height)
    ]
    one_by_one = pl.concat(
        [feature_row(u["features"], logs=None, with_logs=False, msno=u["id"])[0] for u in users],
        how="vertical",
    )
    together, warnings = feature_rows(users, with_logs=False)

    assert together.equals(one_by_one)
    assert len(warnings) == len(users)


@NODATA
def test_feature_rows_keeps_warnings_with_their_own_row():
    """逐列的警告不可以混在一起 —— 一個人的問題不是另一個人的問題。"""
    from src.serving.payload import feature_rows

    cohort = make_synthetic_cohort()
    clean = _payload_from(cohort, 0)
    late = _payload_from(cohort, 1) | {"registration_init_time": 20991231}
    _, warnings = feature_rows(
        [{"id": "clean", "features": clean}, {"id": "late", "features": late}],
        with_logs=False,
    )
    assert warnings[0] == []
    assert any("註冊日" in w for w in warnings[1])


@NODATA
def test_skipping_reasons_does_not_change_a_single_probability(client):
    """`reasons=false` 只跳過 TreeSHAP，機率、決策與期望淨收益必須一字不差。

    TreeSHAP 佔一次批次請求 97.5% 的成本，所以名單畫面先要機率、再補原因碼。
    那個兩段式只有在兩段的機率相同時才成立 —— 不同的話，使用者拖過滑桿之後看到
    的名單會與最終名單不一致，而畫面上不會有任何提示。
    """
    cohort = make_synthetic_cohort()
    users = [{"id": f"u{i}", "features": _payload_from(cohort, i)} for i in range(4)]

    fast = client.post("/predict/batch", json={"users": users, "reasons": False}).json()
    full = client.post("/predict/batch", json={"users": users, "reasons": True}).json()

    assert [u["p_churn"] for u in fast["users"]] == [u["p_churn"] for u in full["users"]]
    assert [u["above_threshold"] for u in fast["users"]] == [
        u["above_threshold"] for u in full["users"]
    ]
    assert [u["expected_net"] for u in fast["users"]] == [u["expected_net"] for u in full["users"]]
    assert fast["n_targeted"] == full["n_targeted"]
    assert fast["expected_net_total"] == full["expected_net_total"]


@NODATA
def test_skipping_reasons_stays_silent_instead_of_claiming_there_are_none(client):
    """沒算原因碼與「這個人沒有推高風險的因素」是兩件事。

    後者是一句關於使用者的判斷。在還沒算的階段講它，是在報告一個沒有量過的結論。
    """
    cohort = make_synthetic_cohort()
    users = [{"id": f"u{i}", "features": _payload_from(cohort, i)} for i in range(4)]
    fast = client.post("/predict/batch", json={"users": users, "reasons": False}).json()

    for u in fast["users"]:
        assert u["reasons"] == []
        assert not any("沒有任何推高風險" in w for w in u["warnings"])

    full = client.post("/predict/batch", json={"users": users, "reasons": True}).json()
    assert any(u["reasons"] for u in full["users"])
