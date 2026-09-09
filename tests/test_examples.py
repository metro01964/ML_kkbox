"""`/docs` 的具名範例必須是自洽的。

範例是 Demo 的全部 —— 訪客不會讀 README，只會下拉選一個然後按 Execute。所以
一組算錯的範例比沒有範例更糟：它會安靜地展示一個矛盾的輸入，而畫面上看不出來。

這裡守三件事：

  1. **衍生欄位與基礎計數一致。** completion / active_ratio / secs_per_active_day
     與三個 trend 都是算出來的，公式在 src/features/logs.py。那邊改了而
     examples.py 沒跟上，這裡要失敗。
  2. **窗口是巢狀的。** 7 天的計數不可能大於 14 天的，以此類推。手寫數字時
     很容易破壞這個關係。
  3. **cutoff 落在模型見過的區間。** 落在外面不會被服務拒絕，但那是外插，
     機率不可信 —— Demo 不該示範一個不可信的用法。
"""

from __future__ import annotations

import pytest

from src.features.logs import LOG_WINDOWS
from src.serving.examples import OPENAPI_EXAMPLES, WINDOWS

ALL = list(OPENAPI_EXAMPLES.items())
WITH_LOGS = [(k, v) for k, v in ALL if v["value"].get("logs")]


def test_windows_match_the_feature_layer():
    """examples.py 重列了一份 WINDOWS，不是 import 的。對不上就在這裡爆。"""
    assert tuple(WINDOWS) == tuple(LOG_WINDOWS)


@pytest.mark.parametrize("key,ex", ALL)
def test_every_example_carries_a_summary_and_description(key, ex):
    assert ex["summary"].strip()
    assert len(ex["description"]) > 80, f"{key} 的說明太短，Demo 上等於沒說"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_derived_log_fields_match_their_formulas(key, ex):
    logs = ex["value"]["logs"]
    for w in WINDOWS:
        plays = logs[f"log{w}_plays"]
        active = logs[f"log{w}_active_days"]
        secs = logs[f"log{w}_secs"]
        completed = logs[f"log{w}_completed"]

        expected_completion = round(completed / plays, 4) if plays > 0 else None
        assert logs[f"log{w}_completion"] == expected_completion, f"{key} log{w}_completion"

        assert logs[f"log{w}_active_ratio"] == round(active / w, 4), f"{key} log{w}_active_ratio"

        expected_spad = round(secs / active, 1) if active > 0 else None
        assert logs[f"log{w}_secs_per_active_day"] == expected_spad, f"{key} log{w}_secs_per_day"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_trends_match_their_formula(key, ex):
    logs = ex["value"]["logs"]

    def expected(num: str, num_days: int, den: str, den_days: int):
        per_day_den = logs[den] / den_days
        if not per_day_den > 0:
            return None
        return round((logs[num] / num_days) / per_day_den, 4)

    assert logs["log_trend_7_30"] == expected("log7_secs", 7, "log30_secs", 30), key
    assert logs["log_trend_30_90"] == expected("log30_secs", 30, "log90_secs", 90), key
    assert logs["log_trend_active_7_30"] == expected(
        "log7_active_days", 7, "log30_active_days", 30
    ), key


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_windows_are_nested(key, ex):
    """7 ⊆ 14 ⊆ 30 ⊆ 90。短窗口的計數不可能超過長窗口。"""
    logs = ex["value"]["logs"]
    for metric in ("active_days", "secs", "plays", "completed", "unq"):
        values = [logs[f"log{w}_{metric}"] for w in WINDOWS]
        for short, long, sw, lw in zip(values, values[1:], WINDOWS, WINDOWS[1:], strict=False):
            assert short <= long, f"{key}: log{sw}_{metric}={short} > log{lw}_{metric}={long}"


@pytest.mark.parametrize("key,ex", WITH_LOGS)
def test_active_days_cannot_exceed_the_window(key, ex):
    logs = ex["value"]["logs"]
    for w in WINDOWS:
        assert logs[f"log{w}_active_days"] <= w, f"{key}: log{w}_active_days 超過 {w} 天"


@pytest.mark.parametrize("key,ex", ALL)
def test_transaction_dates_are_ordered(key, ex):
    f = ex["value"]["features"]
    assert f["first_tx"] <= f["last_tx"], f"{key}: first_tx 晚於 last_tx"
    assert f["last_tx"] <= f["cutoff"], f"{key}: last_tx 晚於 cutoff —— 那是紅線 1"
    if f.get("registration_init_time"):
        assert f["registration_init_time"] <= f["first_tx"], f"{key}: 註冊日晚於首次交易"


@pytest.mark.parametrize("key,ex", ALL)
def test_cutoff_is_inside_the_window_the_model_was_trained_on(key, ex):
    """artifact 的 cohort.cutoff_window。載不到 artifact 就跳過 —— CI 沒有它。"""
    try:
        from src.serving.app import load_serving_config
        from src.serving.artifact import load_artifact

        # 與 app 的 lifespan 同一條路徑：名稱來自 configs/serving.yaml，
        # 環境變數 MODEL_ARTIFACT 若有設會優先。不帶參數呼叫會 ValueError。
        art = load_artifact(name=load_serving_config()["artifact"])
    except Exception as exc:  # noqa: BLE001 - 環境問題，不是測試失敗
        pytest.skip(f"載不到 artifact：{type(exc).__name__}: {exc}")

    lo, hi = art.summary()["cohort"]["cutoff_window"]
    cutoff = ex["value"]["features"]["cutoff"]
    assert lo <= cutoff <= hi, f"{key}: cutoff {cutoff} 落在訓練區間 [{lo}, {hi}] 之外"


def test_the_demo_tells_a_coherent_story():
    """已取消的那位必須明顯高於忠誠用戶。

    不斷言絕對數值 —— 模型重新匯出後會變。斷言的是**排序**：如果這兩個
    反過來，Demo 展示的就是一個壞掉的模型，而那比沒有 Demo 糟糕得多。
    """
    try:
        from fastapi.testclient import TestClient

        from src.serving.app import app
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"起不了服務：{type(exc).__name__}")

    with TestClient(app) as client:
        scores = {}
        for key, ex in ALL:
            r = client.post("/predict", json=ex["value"])
            assert r.status_code == 200, f"{key} 回 {r.status_code}：{r.text[:200]}"
            scores[key] = r.json()["p_churn"]

    assert scores["cancelled"] > scores["loyal"] * 10, (
        f"已取消 {scores['cancelled']:.4f} 沒有明顯高於忠誠用戶 {scores['loyal']:.4f}"
    )
    assert scores["cancelled"] > scores["autorenew_off"] > scores["loyal"], (
        f"風險排序不符預期：{scores}"
    )


# ---------------------------------------------------------------------------
# Demo 批次
# ---------------------------------------------------------------------------
#
# 五個原型是手寫的，靠上面那幾條測試守；批次是**程式生成**的 50 列，所以要守的
# 是不同的東西：生成規則本身會不會長出一列自相矛盾的資料。手寫時看得到的矛盾，
# 生成時看不到 —— 它只會安靜地多出一個人。


@pytest.fixture(scope="module")
def batch():
    from src.serving.examples import build_demo_batch

    return build_demo_batch()


def test_batch_is_deterministic():
    """同一個 seed 必須回同一批人。

    每次重新整理就換一批的話，截圖、README 的數字與說明全部對不起來，而
    「模型每次給的答案不一樣」是這類 Demo 最容易被誤讀的地方。
    """
    from src.serving.examples import build_demo_batch

    assert build_demo_batch() == build_demo_batch()


def test_batch_ids_are_unique(batch):
    assert len({u["id"] for u in batch}) == len(batch)


@pytest.mark.parametrize("field", ["cutoff", "n_tx", "first_tx", "last_tx"])
def test_batch_rows_have_the_required_transaction_fields(batch, field):
    assert all(field in u["features"] for u in batch)


def test_batch_transaction_dates_are_ordered(batch):
    """first_tx ≤ last_tx ≤ cutoff。紅線 1：特徵不得由 cutoff 之後的事算出來。"""
    for u in batch:
        f = u["features"]
        assert f["first_tx"] <= f["last_tx"] <= f["cutoff"], u["id"]
        assert f["registration_init_time"] <= f["first_tx"], u["id"]


def test_batch_cutoffs_are_inside_the_training_window(batch):
    from src.serving.examples import _CUTOFF_HI, _CUTOFF_LO

    for u in batch:
        assert _CUTOFF_LO <= u["features"]["cutoff"] <= _CUTOFF_HI, u["id"]


def test_no_cancelled_row_has_auto_renew_off(batch):
    """`已取消 × 自動續訂關` 在 99.2 萬人裡一筆都沒有 —— 沒開自動扣款的人不需要
    取消，時間到就自然結束。單筆範例那邊踩過這個坑（見 OPENAPI_EXAMPLES 的註解），
    批次是生成的，所以由這條測試釘住。
    """
    for u in batch:
        f = u["features"]
        assert not (f["last_is_cancel"] == 1 and f["last_is_auto_renew"] == 0), u["id"]


def test_cancel_history_cannot_exceed_the_transaction_count(batch):
    """取消次數不能超過交易筆數，而且沒取消的人手上這筆不算。

    生成過一位「1 筆交易、取消佔比 100%，但最後一筆不是取消」—— 讀起來合理、
    實際上不可能，而畫面上只是多一句原因碼。
    """
    for u in batch:
        f = u["features"]
        ceiling = f["n_tx"] if f["last_is_cancel"] else f["n_tx"] - 1
        assert 0 <= f["n_cancel_hist"] <= ceiling, u["id"]


def test_batch_log_windows_are_nested_and_within_bounds(batch):
    """累計計數必須隨窗口遞增，且 active_days 不得超過窗口長度。"""
    for u in batch:
        L = u["logs"]
        if L is None:
            continue
        prev = -1
        for w in WINDOWS:
            active = L[f"log{w}_active_days"]
            assert active <= w, (u["id"], w)
            assert active >= prev, (u["id"], w)
            prev = active


def test_batch_listening_is_never_after_the_cutoff(batch):
    """紅線 2：收聽紀錄不得晚於 cutoff，所以 min_days_before 不能是負的。"""
    for u in batch:
        if u["logs"]:
            assert u["logs"]["log_min_days_before"] >= 0, u["id"]


def test_batch_derived_log_fields_match_their_formulas(batch):
    """與五個原型同一組公式 —— 生成的列一樣不准自相矛盾。"""
    for u in batch:
        L = u["logs"]
        if L is None:
            continue
        for w in WINDOWS:
            plays, completed = L[f"log{w}_plays"], L[f"log{w}_completed"]
            active, secs = L[f"log{w}_active_days"], L[f"log{w}_secs"]
            expected = round(completed / plays, 4) if plays > 0 else None
            assert L[f"log{w}_completion"] == expected, (u["id"], w)
            assert L[f"log{w}_active_ratio"] == round(active / w, 4), (u["id"], w)
            per_day = round(secs / active, 1) if active > 0 else None
            assert L[f"log{w}_secs_per_active_day"] == per_day, (u["id"], w)


def test_the_batch_covers_both_sides_of_the_threshold(batch):
    """名單的重點是**那條線**，所以兩邊都要有人。

    這條測試不看機率（那要載模型），看的是組成：至少要有一群 M0 實測流失率遠高
    於 p* 的人，和一群遠低於的。全部落在同一側的話，這一頁就沒有東西可講了。
    """
    segments = {u["segment"] for u in batch}
    assert any("取消" in s for s in segments)
    assert any("自動續訂開" in s for s in segments)
    # 「自動續訂關閉」那 14 位是這一頁的論點所在：M0 實測 32.25%，風險明顯偏高，
    # 但仍在 p* = 48.4% 之下 —— 風險高與值得花錢是兩件事。
    assert any(s == "自動續訂關閉" for s in segments)
