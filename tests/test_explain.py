"""M5 · 歸因層測試。

兩種測試，理由不同：

**一、純邏輯（假的 fitted）。** Top-k 的挑選規則、分組相加、缺失的呈現 ——
這些完全不需要模型，用手刻的歸因矩陣就能把規則釘死，失敗時一眼看出是哪一列。

**二、三個套件的形狀慣例（真的訓練一次）。** `Fitted.shap_values` 的契約是
「最後一欄是 base value、單位是 log-odds」，而那是三個套件各自的 API 慣例，
不是我們能決定的事。所以要真的各訓練一次小模型去驗 —— 用 conftest 的合成
cohort，不碰 34 GB 原始資料，因此仍然標 nodata、在 CI 上會實際執行。

⚠️ 第二類測試**只驗恆等式與形狀，不驗貢獻值非零**。合成資料只有 8 列，
XGBoost 在這種規模上可能一棵有效的樹都長不出來（實測全部貢獻為 0，機率
0.5）—— 那不是 bug，而恆等式在那種退化情況下依然成立，正好說明它驗的是
「歸因與預測是否同一個模型」而不是「模型有沒有學到東西」。
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from src.explain import (
    Attribution,
    assert_local_accuracy,
    attribute,
    mean_abs_attribution,
    sigmoid,
    top_contributors,
)
from tests.conftest import NODATA, make_synthetic_cohort

# 三欄的迷你歸因：前兩欄是同一件事的兩個窗口（會被分到同一組），第三欄獨立。
FEATURES = ("log7_secs", "log30_secs", "last_is_auto_renew")
GROUPS = {
    "log7_secs": "收聽量",
    "log30_secs": "收聽量",
    "last_is_auto_renew": "自動續訂",
}


def _attr(values: list[list[float]], base: float = -2.0) -> Attribution:
    v = np.array(values, dtype=np.float64)
    return Attribution(FEATURES, v, np.full(v.shape[0], base))


def _X(rows: list[list[float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {name: [r[i] for r in rows] for i, name in enumerate(FEATURES)},
        schema={name: pl.Float64 for name in FEATURES},
    )


def _fake_fitted(contrib: np.ndarray, probability: np.ndarray):
    """一個只有 `Fitted` 三個必要屬性的替身。

    用替身而不是真模型，是為了能製造「歸因與預測不一致」這種真模型做不出來的
    狀況 —— 守門的價值正在於擋住那件事，所以測試必須餵得出來。
    """
    return SimpleNamespace(
        name="Fake",
        predict=lambda X: probability,
        shap_values=lambda X: contrib,
    )


# ---------------------------------------------------------------------------
# 加總恆等式
# ---------------------------------------------------------------------------


@NODATA
def test_probability_is_rebuilt_from_base_plus_contributions():
    """`sigmoid(base + Σ shap)` 就是模型的機率 —— 手算一列釘住單位。

    base −2.0、三項貢獻合計 +1.0 → log-odds −1.0 → 機率 0.2689。
    這也是「貢獻 +1.0 不等於機率 +100%」的具體數字。
    """
    attr = _attr([[0.5, 0.3, 0.2]], base=-2.0)
    assert attr.raw[0] == pytest.approx(-1.0)
    assert attr.probability[0] == pytest.approx(0.26894142, abs=1e-8)


@NODATA
def test_sigmoid_does_not_overflow_on_extreme_log_odds():
    """z = −800 時 `1/(1+exp(-z))` 會 overflow 成 inf 並噴 RuntimeWarning。

    數值上仍收斂到 0，但會在 log 裡留下一行看起來很嚴重的警告 —— 而真正該被
    看見的警告就是這樣被淹掉的。
    """
    with np.errstate(over="raise"):
        out = sigmoid(np.array([-800.0, 0.0, 800.0]))
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.5)
    assert out[2] == pytest.approx(1.0)


@NODATA
def test_local_accuracy_passes_when_they_agree():
    contrib = np.array([[0.5, 0.3, 0.2, -2.0]])  # 最後一欄是 base
    fitted = _fake_fitted(contrib, sigmoid(np.array([-1.0])))
    X = _X([[10.0, 20.0, 1.0]])

    attr = attribute(fitted, X)
    assert assert_local_accuracy(fitted, X, attr) == pytest.approx(0.0, abs=1e-12)


@NODATA
def test_local_accuracy_catches_attribution_from_a_different_model():
    """歸因加不回預測 → 必須整支中斷。

    這條守門抓的是一整類**不會報錯**的錯：歸因與預測用了不同輪數的模型、
    欄位順序不一致、類別特徵的轉接寫了第二份。症狀都一樣 —— 名單完全正確、
    機率完全正確、原因碼張張可讀，只是講的是別人的事。
    """
    contrib = np.array([[0.5, 0.3, 0.2, -2.0]])  # → 機率 0.2689
    fitted = _fake_fitted(contrib, np.array([0.9]))  # 但模型說 0.9
    X = _X([[10.0, 20.0, 1.0]])

    attr = attribute(fitted, X)
    with pytest.raises(AssertionError, match="SHAP 加總對不上"):
        assert_local_accuracy(fitted, X, attr)


@NODATA
def test_attribute_rejects_output_without_the_base_column():
    """少了 base 那一欄就不能默默當成「剛好 61 欄」用。

    若放行，恆等式會差一個 base value（本專案約 −2.4 的 log-odds），機率整批
    偏掉 —— 而每一列都偏同一個量，看起來非常像「模型系統性低估」這種可以
    解釋得通的現象。
    """
    fitted = _fake_fitted(np.array([[0.5, 0.3, 0.2]]), np.array([0.5]))
    with pytest.raises(ValueError, match="base value"):
        attribute(fitted, _X([[10.0, 20.0, 1.0]]))


@NODATA
@pytest.mark.parametrize(
    ("values", "base", "match"),
    [
        (np.zeros((2, 2)), np.zeros(2), "欄數不符"),
        (np.zeros((2, 3)), np.zeros(3), "base 形狀"),
        (np.zeros(3), np.zeros(1), "必須是二維"),
    ],
)
def test_attribution_shape_contract(values, base, match):
    with pytest.raises(ValueError, match=match):
        Attribution(FEATURES, values, base)


# ---------------------------------------------------------------------------
# Top-k 的挑選規則
# ---------------------------------------------------------------------------


@NODATA
def test_only_risk_increasing_contributions_become_reasons():
    """負貢獻不是原因碼。

    挽回名單要的是「為什麼該打給他」。「他有開自動續訂，所以風險比較低」是
    真的，但那不是打電話的理由 —— 把它列進 Top-3 會讓營運讀到一句自相矛盾
    的話。
    """
    attr = _attr([[0.9, -1.5, 0.4]])
    out = top_contributors(attr, _X([[1.0, 2.0, 3.0]]), k=3)

    assert out["feature"].to_list() == ["log7_secs", "last_is_auto_renew"]
    assert out["rank"].to_list() == [1, 2]
    assert (out["group_shap"] > 0).all()


@NODATA
def test_fewer_than_k_reasons_is_allowed():
    """只有一個正貢獻就只給一個 —— 補到 3 個等於編造理由。"""
    attr = _attr([[0.7, -0.2, -0.9]])
    out = top_contributors(attr, _X([[1.0, 2.0, 3.0]]), k=3)
    assert out.height == 1


@NODATA
def test_no_reasons_when_nothing_pushes_the_risk_up():
    """全部負貢獻 → 一列都不給，而不是硬挑一個最不負的。"""
    attr = _attr([[-0.1, -0.2, -0.9]])
    assert top_contributors(attr, _X([[1.0, 2.0, 3.0]]), k=3).height == 0


@NODATA
def test_grouping_changes_which_reason_wins():
    """**Top-3 特徵不等於 Top-3 原因。**

    `log7_secs` 與 `log30_secs` 是同一件事的兩個窗口，SHAP 把功勞拆給兩欄
    （各 +0.3）。不分組時「沒開自動續訂」（+0.5）看起來是最大原因；分組相加
    之後「收聽量下滑」（+0.6）才是。

    SHAP 的加法性讓組內相加是精確的，所以這個重排不是近似造成的偏好，是把
    被拆散的功勞還原。
    """
    attr = _attr([[0.3, 0.3, 0.5]])
    X = _X([[1.0, 2.0, 0.0]])

    plain = top_contributors(attr, X, k=1)
    grouped = top_contributors(attr, X, k=1, groups=GROUPS)

    assert plain["feature"][0] == "last_is_auto_renew"
    assert grouped["group"][0] == "收聽量"
    assert grouped["group_shap"][0] == pytest.approx(0.6)
    # 組內同分時取先出現的那一欄（stable），不隨執行變動。
    assert grouped["feature"][0] == "log7_secs"


@NODATA
def test_group_total_and_representative_contribution_are_different_numbers():
    """組內可以有反向成員，所以 `group_shap`（組總和）≠ `feature_shap`（代表欄）。

    兩個數字混用會產生一句錯的話：報表若拿組總和當「這一欄的貢獻」，讀者
    會以為代表欄的影響比實際小。
    """
    attr = _attr([[0.8, -0.3, 0.1]])
    out = top_contributors(attr, _X([[1.0, 2.0, 3.0]]), k=1, groups=GROUPS)

    assert out["group"][0] == "收聽量"
    assert out["group_shap"][0] == pytest.approx(0.5)
    assert out["feature"][0] == "log7_secs"
    assert out["feature_shap"][0] == pytest.approx(0.8)


@NODATA
def test_a_missing_feature_value_stays_missing():
    """「模型看到的是缺失」本身就是一種理由，不可以補成 0。

    補 0 會讓原因碼寫出「近 7 天聽歌 0 秒」，而事實是這個人根本沒有紀錄 ——
    與 `_attach_logs` 不補 0 的理由完全相同。
    """
    attr = _attr([[0.9, 0.1, 0.1]])
    out = top_contributors(attr, _X([[None, 2.0, 3.0]]), k=1)
    assert out["value"][0] is None


@NODATA
def test_an_unmapped_feature_raises_instead_of_being_dropped():
    """新增特徵卻忘了給它分組 → 報錯。

    靜默的後果是那個特徵永遠不會出現在任何人的原因碼裡：名單照樣產生、
    每個人照樣有三句話，沒有任何東西會顯示少了一個家族。
    """
    attr = _attr([[0.3, 0.3, 0.5]])
    partial = {"log7_secs": "收聽量", "log30_secs": "收聽量"}
    with pytest.raises(KeyError, match="沒有分組"):
        top_contributors(attr, _X([[1.0, 2.0, 3.0]]), k=1, groups=partial)


@NODATA
def test_rows_are_independent():
    """兩列的排序互不影響 —— 逐列歸因不是全體平均。"""
    attr = _attr([[0.9, 0.1, 0.2], [0.1, 0.2, 0.9]])
    out = top_contributors(attr, _X([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), k=1)

    assert out["row"].to_list() == [0, 1]
    assert out["feature"].to_list() == ["log7_secs", "last_is_auto_renew"]
    assert out["value"].to_list() == [1.0, 6.0]


@NODATA
def test_mean_abs_attribution_uses_absolute_values():
    """全體排名要用 |SHAP|：一個「一半的人 +1、一半的人 −1」的特徵影響很大，
    但直接平均會得到 0，排到最後一名。"""
    attr = _attr([[1.0, 0.4, 0.0], [-1.0, 0.4, 0.0]])
    out = mean_abs_attribution(attr)
    assert out["feature"][0] == "log7_secs"
    assert out["mean_abs_shap"][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 三個套件的形狀慣例
# ---------------------------------------------------------------------------

TINY_TRAIN_CFG = {"num_boost_round": 8, "early_stopping_rounds": 4}


def _tiny_fitted(package: str):
    """在合成 cohort 上訓練一個迷你模型。

    超參數刻意不從 `configs/` 讀：這個測試驗的是套件 API 的形狀慣例，不是
    本專案的模型設定。綁上設定檔的話，調參會讓這條測試無關地變動。
    """
    from src.features import build_features
    from src.models.candidates import (
        fit_catboost,
        fit_lightgbm,
        fit_xgboost,
        xgb_category_levels,
    )

    fs = build_features(make_synthetic_cohort())
    train, es = fs.take([0, 1, 2, 3]), fs.take([4, 5, 6, 7])

    if package == "lightgbm":
        params = {
            "objective": "binary",
            "num_leaves": 2,
            "min_data_in_leaf": 1,
            "min_data_in_bin": 1,
            "learning_rate": 0.3,
            "verbose": -1,
            "seed": 0,
        }
        return fit_lightgbm(train, es, params, TINY_TRAIN_CFG), fs
    if package == "xgboost":
        params = {
            "objective": "binary:logistic",
            "max_depth": 2,
            "eta": 0.3,
            "tree_method": "hist",
        }
        levels = xgb_category_levels(train.X, categorical=train.categorical)
        return fit_xgboost(train, es, params, TINY_TRAIN_CFG, category_levels=levels), fs
    params = {"loss_function": "Logloss", "depth": 2, "learning_rate": 0.3, "random_seed": 0}
    return fit_catboost(train, es, params, TINY_TRAIN_CFG), fs


@NODATA
@pytest.mark.parametrize("package", ["lightgbm", "xgboost", "catboost"])
def test_every_backend_puts_the_base_value_in_the_last_column(package):
    """三家的 SHAP 輸出都是 `(列數, 特徵數 + 1)`，最後一欄是 base value。

    這個慣例是三個套件各自的 API 決定的，不是我們能選的 —— 所以要真的各跑
    一次。若某家換了慣例（例如 base 放第一欄），`attribute()` 的形狀檢查不會
    叫（欄數一樣），但恆等式會立刻破，因為 base 被當成某個特徵的貢獻。
    """
    fitted, fs = _tiny_fitted(package)

    raw = np.asarray(fitted.shap_values(fs.X))
    assert raw.shape == (fs.X.height, fs.X.width + 1)

    attr = attribute(fitted, fs.X)
    gap = assert_local_accuracy(fitted, fs.X, attr)
    assert gap < 1e-9, f"{package} 的加總誤差 {gap:.3e} 大得不像浮點誤差"


# ---------------------------------------------------------------------------
# 句型與量測時點（horizon）
# ---------------------------------------------------------------------------


@NODATA
def test_last_is_cancel_is_marked_as_an_expiry_dated_signal():
    """`last_is_cancel` 必須標成到期日訊號。

    取消常發生在到期日當天，而現行 cutoff 就是到期日。M6 要交付
    `cutoff = 到期日 − 7 天` 的版本，那時這筆交易還沒發生 —— **這句原因碼
    不可沿用**。它同時是實測最強的旗標（流失率 75.07% vs 4.33%），所以也是
    T−7 版本損失最大的一個。

    ⚠️ 標註不是「推論時把這一欄遮掉」。SHAP 的歸因是聯合的，遮一欄會讓
    `sigmoid(base + Σ shap) == predict()` 破掉，破的量剛好是這個訊號的強度。
    正解是重訓 T−7 版本的模型。
    """
    from src.explain import HORIZON_EXPIRY, meta

    assert meta("last_is_cancel").horizon == HORIZON_EXPIRY


@NODATA
def test_only_the_signal_that_may_not_exist_yet_is_expiry_dated():
    """到期日訊號只有 `last_is_cancel` 一個 —— 其餘不得誤標。

    歷史取消次數（`n_cancel_hist`）是已經發生的事，T−7 一樣看得到，只是不含
    最後那一筆 → 「位移」。把它一起標成到期日訊號，會讓「有多少解釋撐不到
    T−7」這個數字虛高，而那個數字是 M6 的決策依據之一。
    """
    from src.explain import FEATURES, HORIZON_EXPIRY

    expiry = {name for name, m in FEATURES.items() if m.horizon == HORIZON_EXPIRY}
    assert expiry == {"last_is_cancel"}


@NODATA
def test_every_feature_declares_a_known_horizon():
    """每個特徵都要有量測時點，而且只能是三種之一。

    打錯字（例如 "位移 " 多一個空白）會讓 `expiry_dated_share()` 的分組安靜地
    多一類，比例算出來仍然是個看起來合理的數字。
    """
    from src.explain import FEATURES, HORIZONS

    bad = {name: m.horizon for name, m in FEATURES.items() if m.horizon not in HORIZONS}
    assert not bad, f"未知的 horizon：{bad}"


@NODATA
def test_every_feature_in_the_matrix_has_a_template():
    """特徵矩陣的每一欄都要有句型與分組。

    交易與會員欄位由 `build_features()` 決定，收聽欄位由 `expected_log_columns()`
    推導 —— 兩邊都不寫死，所以新增一個窗口或一個特徵時，這條測試會先失敗，
    而不是讓名單裡出現「log60_secs 1234.0」這種營運看不懂的話。
    """
    from src.explain import missing_metadata
    from src.features import build_features
    from src.features.logs import expected_log_columns

    matrix_columns = build_features(make_synthetic_cohort()).X.columns
    log_columns = sorted(expected_log_columns() - {"msno", "cutoff"})

    missing = missing_metadata(list(matrix_columns) + log_columns)
    assert not missing, f"這些特徵沒有登記句型：{missing}"


@NODATA
def test_categorical_features_are_declared_as_categories():
    """`CATEGORICAL` 裡的欄位必須標成 category，否則 −1 會印成數字。

    `city −1.0` 讀起來像一個城市代碼，實際意思是「不在 members_v3 裡」。
    """
    from src.explain import meta
    from src.features import CATEGORICAL

    for name in CATEGORICAL:
        assert meta(name).category, f"{name} 是類別特徵但沒有標 category=True"


@NODATA
@pytest.mark.parametrize(
    ("feature", "value", "expected"),
    [
        # 旗標：講的是「值代表什麼狀態」，不是「這一欄叫什麼」
        ("last_is_auto_renew", 0.0, "未開啟自動續訂"),
        ("last_is_auto_renew", 1.0, "已開啟自動續訂"),
        ("last_is_cancel", 1.0, "到期前最後一筆交易是取消"),
        ("zero_collected", 1.0, "定價非 0 元但實收 0 元"),
        # 類別：−1 是缺失，不是一個代碼
        ("city", -1.0, "居住城市不明"),
        ("city", 13.0, "居住城市代碼 13"),
        ("gender_code", 1.0, "性別為女"),
        ("gender_code", -1.0, "性別不明"),
        # 數值與比率
        ("log30_completion", 0.1234, "近 30 天完播率 12.3%"),
        ("log_trend_7_30", 0.12, "近 7 天聽歌時間是近 30 天日均的 0.12 倍"),
        ("price_per_day", 3.3, "日均單價 3.30 元"),
        # 缺失：不可以印成 0
        ("price_per_day", None, "日均單價缺失"),
        ("bd_clean", None, "年齡缺失"),
    ],
)
def test_reason_sentences(feature, value, expected):
    """句子本身就是交付物的一部分，逐句釘住。"""
    from src.explain import render_reason

    assert render_reason(feature, value) == expected


@NODATA
def test_a_null_listening_feature_says_what_is_actually_true():
    """收聽欄位的 null 只有一個成因，句子要講那個成因。

    實測（2026-08-10）：`mar_log_features.parquet` 本身**零 null**（796,298 列），
    而 Mar cohort 有 970,959 人 —— 差的 174,661 人（18.0%）不在那張表裡。所以
    特徵矩陣裡每一個 null 的收聽欄位都來自 left join。

    「近 14 天聽過的不同歌曲數缺失」字面上沒錯，但讀起來像 14 天這個窗口有
    資料缺口；事實是這個人整個 90 天窗口一片空白 —— 那是**行為**而不是資料
    品質問題，對營運的意思完全相反。

    ⚠️ 仍然依該列實際的 `log_has_logs` 判斷，不假設那個不變量永遠成立。
    """
    from src.explain import render_reason

    assert render_reason("log14_unq", None, no_logs=True) == "近 90 天完全沒有收聽紀錄"
    assert render_reason("log14_unq", None, no_logs=False) == "近 14 天聽過的不同歌曲數缺失"
    # 交易欄位的缺失與收聽無關，不可以被這條規則波及。
    assert render_reason("price_per_day", None, no_logs=True) == "日均單價缺失"


@NODATA
def test_add_reasons_reads_has_logs_from_the_matrix():
    """`add_reasons` 要自己去 X 拿 `log_has_logs`，不是由呼叫端記得傳。"""
    from src.explain import add_reasons

    features = ("log30_active_days", "last_is_auto_renew", "log_has_logs")
    values = np.array([[0.9, 0.5, 0.1], [0.9, 0.5, 0.1]])
    attr = Attribution(features, values, np.full(2, -2.4))
    X = pl.DataFrame(
        {
            "log30_active_days": [None, 4.0],
            "last_is_auto_renew": [0.0, 0.0],
            "log_has_logs": [0.0, 1.0],
        }
    )

    out = add_reasons(top_contributors(attr, X, k=1), X)
    assert out["reason"].to_list() == ["近 90 天完全沒有收聽紀錄", "近 30 天活躍天數 4 天"]


@NODATA
def test_the_context_number_is_a_window_difference_not_a_new_feature():
    """SPEC 例句的「由 22 降至 4」用既有特徵相減補出來。

    90 天窗口包含 30 天窗口，活躍天數可加，所以「前 60 天平均每 30 天」
    = (log90 − log30) / 2。這裡 log90 = 40、log30 = 4 → 前 60 天平均 18 天。

    ⚠️ 對照數字是**解釋用的脈絡**，模型看到的是 `log30_active_days` 這一欄，
    排序也來自 SHAP 對那一欄的歸因。
    """
    from src.explain import render_reason

    sentence = render_reason("log30_active_days", 4.0, context_value=40.0)
    assert sentence == "近 30 天活躍天數 4 天（前 60 天平均每 30 天 18 天）"


@NODATA
def test_non_additive_features_get_no_context_clause():
    """不同歌曲數不可加 —— 90 天 100 首、30 天 40 首，不代表前 60 天 30 首。

    同一首歌會在兩個窗口重複出現，相減得到的是上界而不是數量。比率類同理。
    """
    from src.explain import meta

    assert meta("log30_unq").context is None
    assert meta("log30_completion").context is None
    assert meta("log30_active_ratio").context is None
    assert meta("log30_active_days").context is not None


@NODATA
def test_add_reasons_marks_but_does_not_filter():
    """到期日訊號要留在輸出裡帶旗標，不可以被過濾掉。

    過濾會讓報表看起來很乾淨，代價是「這份名單有多少解釋撐不到 T−7」變成
    不可量的問題 —— 而那正是 M6 需要的數字。
    """
    from src.explain import add_reasons, expiry_dated_share

    features = ("last_is_cancel", "last_is_auto_renew", "log30_active_days")
    values = np.array([[2.0, 0.5, 0.3], [0.0, 0.9, 0.4]])
    attr = Attribution(features, values, np.full(2, -2.4))
    X = pl.DataFrame(
        {
            "last_is_cancel": [1.0, 0.0],
            "last_is_auto_renew": [0.0, 0.0],
            "log30_active_days": [4.0, 2.0],
        }
    )

    out = add_reasons(top_contributors(attr, X, k=3), X)

    assert out.filter(pl.col("feature") == "last_is_cancel").height == 1
    assert out.filter(pl.col("expiry_dated")).height == 1
    assert "到期前最後一筆交易是取消" in out["reason"].to_list()

    share = expiry_dated_share(out)
    assert share["受影響人數"] == 1
    assert share["有原因碼的人數"] == 2


# ---------------------------------------------------------------------------
# 呈現門檻：營運看到的與稽核看到的不是同一組句子
# ---------------------------------------------------------------------------

DISPLAY_FEATURES = ("last_is_cancel", "log14_secs_per_active_day", "log90_active_days")


def _display_case(values: list[float]):
    """一位用戶，三個候選原因，貢獻由呼叫端指定。"""
    attr = Attribution(DISPLAY_FEATURES, np.array([values], dtype=np.float64), np.array([-2.4]))
    X = pl.DataFrame(
        {
            "last_is_cancel": [1.0],
            "log14_secs_per_active_day": [790.0],
            "log90_active_days": [20.0],
        }
    )
    from src.explain import add_reasons

    return add_reasons(top_contributors(attr, X, k=3), X), X


@NODATA
def test_the_relative_floor_suppresses_what_is_not_really_a_reason():
    """第 2、3 句只有第 1 名的 1% 時，不該並列呈現。

    實測的真實案例（2026-08-10）：

        1. 到期前最後一筆交易是取消          +7.941
        2. 近 14 天活躍日平均聽歌時間 790 秒  +0.093   ← 第 1 名的 1.2%
        3. 近 90 天活躍天數 20 天             +0.075   ← 第 1 名的 0.9%

    三句並列，營運會以為三件事都重要。**但被壓下的那兩列仍然留在表上**，
    帶著 `suppression_reason` —— 刪掉它們，「這個人本來還有第三個理由、只是
    太弱」這個資訊就永久消失了。
    """
    from src.explain import mark_display

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    out = mark_display(reasons, min_relative=0.05)

    assert out.height == 3, "被壓下的列不可以被刪掉"
    assert out["displayed"].to_list() == [True, False, False]
    assert out["relative_to_top"][0] == pytest.approx(1.0)
    assert out["relative_to_top"][1] == pytest.approx(0.093 / 7.941, rel=1e-6)
    assert out["suppression_reason"][0] is None
    assert out["suppression_reason"][1] == "below_relative_floor(0.05)"


@NODATA
def test_a_genuinely_multi_causal_user_keeps_all_three():
    """三個貢獻量級相近時，一句都不該被壓下 —— 門檻不是「只留第一名」。"""
    from src.explain import mark_display

    reasons, _ = _display_case([4.369, 3.442, 3.274])
    out = mark_display(reasons, min_relative=0.05)
    assert out["displayed"].to_list() == [True, True, True]


@NODATA
def test_suppression_is_always_a_suffix():
    """不可能出現「第 2 句沒印、第 3 句印了」。

    排名依 `group_shap` 遞減，所以 `relative_to_top` 單調不增。`wide_reasons`
    依賴這個性質才能不重新編號 —— 若哪天排序規則改了，這條會先失敗。
    """
    from src.explain import mark_display

    reasons, _ = _display_case([5.0, 0.2, 0.19])
    shown = mark_display(reasons, min_relative=0.05)["displayed"].to_list()
    assert shown == sorted(shown, reverse=True), f"呈現旗標不是後綴：{shown}"


@NODATA
def test_zero_floor_displays_everything():
    """`--min-relative 0` 要能還原成「全部呈現」，門檻才是可退出的。"""
    from src.explain import mark_display

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    assert mark_display(reasons, min_relative=0.0)["displayed"].all()


@NODATA
def test_the_operator_list_shows_only_displayed_reasons():
    """營運名單只印該印的；被壓下的那些不在這張表上。"""
    from src.explain import mark_display, wide_reasons

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    wide = wide_reasons(mark_display(reasons), pl.Series("msno", ["u0"]))

    assert wide["reason_1"][0] == "到期前最後一筆交易是取消"
    assert wide["reason_2"][0] is None
    assert wide["reason_3"][0] is None


@NODATA
def test_the_audit_table_keeps_every_candidate_with_the_five_columns():
    """稽核表要能回答「當時為什麼沒印」，所以五個欄位都必須在。"""
    from src.explain import AUDIT_COLUMNS, audit_frame, mark_display

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    audit = audit_frame(mark_display(reasons), pl.Series("msno", ["u0"]))

    for col in ("rank", "group_shap", "relative_to_top", "displayed", "suppression_reason"):
        assert col in audit.columns
    assert list(audit.columns) == list(AUDIT_COLUMNS), "稽核表的欄位順序要有單一來源"
    assert audit.height == 3, "稽核表保留全部候選"
    assert audit["msno"].to_list() == ["u0"] * 3


@NODATA
def test_audit_frame_refuses_a_table_that_never_went_through_mark_display():
    """少跑一步就報錯，不要產生一張沒有 displayed 欄的「稽核表」。"""
    from src.explain import audit_frame

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    with pytest.raises(KeyError, match="稽核表缺少欄位"):
        audit_frame(reasons, pl.Series("msno", ["u0"]))


@NODATA
def test_display_impact_reports_the_cost_of_the_floor():
    """門檻的代價要跟門檻一起報 —— 只報門檻值等於沒說它砍掉了什麼。"""
    from src.explain import display_impact, mark_display

    reasons, _ = _display_case([7.941, 0.093, 0.075])
    impact = display_impact(mark_display(reasons, min_relative=0.05))

    assert impact["候選句數"] == 3
    assert impact["呈現句數"] == 1
    assert impact["壓下句數"] == 2
    assert impact["受影響人數"] == 1


@NODATA
def test_wide_reasons_leaves_missing_ranks_empty():
    """不足 3 句就留空欄，不補字。

    補一句「無其他明顯原因」看起來友善，實際上是把「模型只找到一個理由」
    這個資訊蓋掉。
    """
    from src.explain import add_reasons, wide_reasons

    features = ("last_is_auto_renew", "log30_active_days")
    values = np.array([[0.9, -0.2]])
    attr = Attribution(features, values, np.full(1, -2.4))
    X = pl.DataFrame({"last_is_auto_renew": [0.0], "log30_active_days": [4.0]})

    wide = wide_reasons(add_reasons(top_contributors(attr, X, k=3), X), pl.Series("msno", ["u0"]))

    assert wide["reason_1"][0] == "未開啟自動續訂"
    assert wide["reason_2"][0] is None
    assert wide["reason_3"][0] is None


@NODATA
def test_the_adopted_model_is_the_one_that_gets_explained():
    """原因碼必須解釋**會上線的那個模型**（§7.12 的 CatBoost）。

    與 `test_m4_contracts.py` 第一條同一個理由：M4 一度出現「校準結論是在一個
    不會上線的模型上得出的」。原因碼比校準更容易犯這個錯 —— 用 LightGBM 解釋
    出來的三句話讀起來一樣通順，沒有任何東西會顯示它解釋的不是 CatBoost。
    """
    from src.models.adopted import ADOPTED_MODEL

    fitted, fs = _tiny_fitted(ADOPTED_MODEL)
    attr = attribute(fitted, fs.X)
    assert assert_local_accuracy(fitted, fs.X, attr) < 1e-9
