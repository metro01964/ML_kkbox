"""M4 的三條工程契約 —— 靜態掃描 + 純邏輯，不需資料。

這一份守的不是演算法，是**「文件與程式會不會各說各話」**。三條契約都來自
實際踩過的坑：

1. **校準流程必須用正式採用的模型。** M4 的校準診斷用 LightGBM 做，而
   §7.12 正式採用的是 CatBoost —— 於是「校準器不上線」這個結論是在一個
   不會上線的模型上得出的。兩支腳本各自 `load_model_config()` 讀
   `model_lgbm.yaml`，看起來完全正常。

2. **`實際淨收益` 這個名字會誤導。** 那條線用的 `r_save` 與 LTV 都是假設，
   只有 Mar 的流失標籤是真的。叫「實際」會讓人以為那是真的賺到的錢，
   而本專案沒有實驗組、無法宣稱任何投放效果。

3. **「所有金額都低估 27%」不成立。** 27% 是**全體 cohort 平均預測流失率**
   的相對偏差，不能套到個別用戶、個別風險區間、前 5% 名單，更不能套到
   扣掉固定成本之後的淨收益（成本項不隨機率縮放）。

靜態掃描的理由與紅線 3 相同：這類錯誤不會讓任何程式失敗，只會讓讀的人
得到錯的印象，所以要用「這個字串不准出現」來守。
"""

from __future__ import annotations

import re

import pytest

from src.config import REPO_ROOT
from tests.conftest import NODATA

CODE_DIRS = ("src", "scripts")
DOCS = ("SPEC.md", "README.md")


def _code_files():
    for d in CODE_DIRS:
        yield from sorted((REPO_ROOT / d).rglob("*.py"))


def _doc_files():
    for name in DOCS:
        path = REPO_ROOT / name
        if path.exists():
            yield path


@NODATA
def test_calibration_scripts_use_the_adopted_model():
    """校準流程不得自己挑模型 —— 必須走 `src.models.adopted`。

    §7.12 正式採用 CatBoost。校準器要決定的是「**會上線的那個模型**的機率
    準不準」，用另一個模型算出來的答案不能拿來下架或上線任何東西。
    """
    from src.models.adopted import ADOPTED_MODEL

    assert ADOPTED_MODEL == "catboost", "§7.12 正式採用的是 CatBoost"

    for name in ("calibration_report.py", "calibrate.py"):
        src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "fit_lightgbm" not in src, f"{name} 仍直接呼叫 fit_lightgbm，不是正式採用的模型"
        assert "adopted" in src, f"{name} 應透過 src.models.adopted 取得模型"


@NODATA
def test_adopted_params_are_not_duplicated():
    """超參數只能有一份來源（`configs/model_comparison.yaml`）。

    複製一份到別的設定檔，兩邊遲早不同步，而「這條曲線是用哪組參數算的」
    屆時就沒有答案。
    """
    import yaml

    from src.models.adopted import ADOPTED_MODEL, load_adopted_config

    official = yaml.safe_load(
        (REPO_ROOT / "configs" / "model_comparison.yaml").read_text(encoding="utf-8")
    )
    params, train_cfg = load_adopted_config()

    assert params == official["models"][ADOPTED_MODEL]
    assert train_cfg == official["training"]


@NODATA
def test_the_misleading_actual_revenue_name_is_gone():
    """`實際淨收益` 不得出現在程式或文件裡。

    真實的只有 Mar 的流失標籤；`r_save`（挽回成功率）與 LTV 都是假設，
    而且沒有實驗組可以驗證投放是否真的改變了行為。叫「實際」等於宣稱
    我們真的賺到了那筆錢。
    """
    offenders = []
    for path in list(_code_files()) + list(_doc_files()):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "實際淨收益" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, "仍在使用會誤導的「實際淨收益」：\n" + "\n".join(offenders)


@NODATA
def test_the_settled_revenue_name_is_used_instead():
    """替代名稱要真的被使用，而不是把舊名字刪掉就算。"""
    from src.evaluation import campaign_curve

    curve = campaign_curve(
        [0, 1, 1, 0],
        [0.1, 0.9, 0.8, 0.2],
        r_save=0.2,
        ltv_saved=1000,
        c_offer=50,
        step=0.5,
    )
    assert "標籤結算模擬淨收益" in curve.columns
    assert "實際淨收益" not in curve.columns


@NODATA
@pytest.mark.parametrize(
    "pattern",
    [
        r"所有金額都?系統性?偏低",
        r"所有金額.{0,6}低估",
        r"每一筆(收益|金額).{0,8}(等比例|按比例)",
        r"反向錯誤不會發生",
        r"等比例縮小每(一)?筆",
    ],
)
def test_the_overreaching_underestimate_claims_are_gone(pattern):
    """「低估 27% ⇒ 所有金額低估 27%」這個推論不成立，不得出現在文件裡。

    27% 是**全體 cohort 平均預測流失率**的相對偏差。它不能套到：

        個別用戶            每個人的偏差不同
        個別風險區間        §7.10 的 reliability 曲線顯示各段偏差差很大
        前 5% 投放名單      那是一個高機率子集，偏差與全體不同
        淨收益              C_offer 是固定成本，不隨機率縮放

    最後一項最容易被忽略：`p × r × LTV − C` 裡只有第一項隨 p 縮放，
    所以機率低估 27% 造成的淨收益偏差**不是** 27%。
    """
    rx = re.compile(pattern)
    offenders = []
    for path in _doc_files():
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if rx.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{i}　{line.strip()[:80]}")
    assert not offenders, f"仍有過度推論的敘述（{pattern}）：\n" + "\n".join(offenders)


@NODATA
def test_subset_calibration_is_available_for_the_targeted_slice():
    """必須能單獨報「前 K% 名單」自己的偏差，而不是套用全體的 27%。

    手算：8 人，前 25%（2 人）預測 0.9/0.8、實際 1/1 →
    平均預測 0.85、實際 1.0、相對偏差 −0.15。
    """
    from src.evaluation import subset_calibration

    y = [1, 1, 0, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]

    out = subset_calibration(y, p, k=0.25)

    assert out["人數"] == 2
    assert out["平均預測"] == pytest.approx(0.85)
    assert out["實際流失率"] == pytest.approx(1.0)
    assert out["相對偏差"] == pytest.approx(-0.15)


@NODATA
def test_subset_bias_differs_from_the_overall_bias():
    """釘住「不能把全體偏差套到子集」這件事本身。

    造一份低機率端低估、高機率端高估的資料：全體偏差接近 0，但前 25%
    的偏差是正的。若有人拿全體的數字去修子集，方向就錯了。
    """
    from src.evaluation import calibration_in_the_large, subset_calibration

    y = [1, 0, 0, 0, 1, 1, 1, 0]
    p = [0.95, 0.90, 0.05, 0.05, 0.60, 0.55, 0.50, 0.05]

    overall = calibration_in_the_large(y, p)
    top = subset_calibration(y, p, k=0.25)

    assert overall["相對偏差"] != pytest.approx(top["相對偏差"], abs=0.05)


@NODATA
def test_the_threshold_derivation_has_a_single_source():
    """M4 的業務曲線與 M5 的原因碼名單必須用同一個 `p*`。

    這是本檔第一條契約的同一個教訓再來一次。M4 一度出現「兩支腳本各自
    `load_model_config()`，於是校準結論是在一個不會上線的模型上得出的」；
    `p*` 更容易犯，因為它不是設定值而是**推導結果**（`C / (r × LTV)`，而 LTV
    又是 `月費 × 1/流失率` 推出來的）。兩支腳本各推一份，症狀會是兩份交付物
    都印出「投放 4.8 萬人」卻指著不同的名單。

    所以推導只能有一份程式：`src.evaluation.resolve_assumptions`。
    """
    for name in ("business_value.py", "explain.py"):
        src = (REPO_ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "resolve_assumptions" in src, f"{name} 沒有走共用的假設推導"
        assert "def monthly_arpu" not in src, f"{name} 自己實作了一份月費推導"


@NODATA
def test_the_assumption_summary_is_shared_so_manifests_are_comparable():
    """兩份 manifest 的 `assumptions` 區塊要逐欄可比，所以共用 `summary()`。"""
    from src.evaluation import resolve_assumptions

    assumptions = resolve_assumptions(
        {"c_offer": 150, "r_save": 0.15, "days_per_month": 30.4},
        price_per_day=[4.0, 5.0, None],
        prior_churn_rate=0.0639,
    )
    summary = assumptions.summary()

    assert list(summary) == [
        "r_save",
        "c_offer",
        "ltv_saved",
        "monthly_arpu",
        "expected_months",
        "months_source",
        "p_star",
    ]
    # 手算：月費 4.5 × 30.4 = 136.8；月數 1/0.0639 = 15.65；LTV = 2141.0
    # p* = 150 / (0.15 × 2140.99) = 0.4670
    assert summary["monthly_arpu"] == pytest.approx(136.8)
    assert summary["expected_months"] == pytest.approx(15.65, abs=0.01)
    assert summary["p_star"] == pytest.approx(0.467, abs=0.001)


@NODATA
def test_the_reason_code_list_records_its_cutoff_definition():
    """M5 的 manifest 必須記下 cutoff 定義。

    少了它，這份原因碼被拿到 M6 的 `cutoff = 到期日 − 7 天` 版本使用時，
    沒有任何東西會擋 —— 而 `last_is_cancel` 那一句在那個版本根本還沒發生。
    """
    from scripts.explain import CUTOFF_DEFINITION

    assert CUTOFF_DEFINITION == "expire_date"
    src = (REPO_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    assert '"cutoff_definition"' in src, "manifest 沒有寫入 cutoff_definition"


@NODATA
def test_the_dirty_flag_is_captured_before_anything_is_written():
    """`git_dirty` 必須在寫出任何檔案之前抓。

    圖 14 進 git，而 `plot_reasons()` 會覆寫它 —— 若在輸出段落才抓狀態，
    這個旗標**永遠是 True**，於是從一個警告退化成一行雜訊：讀者學會忽略它，
    真正該被擋下的那次（帶著未提交的改動跑）就混了進來。

    用靜態掃描而不是執行一次：這支腳本要訓練 CatBoost，跑一次五分鐘。

    ⚠️ 掃描要跳過註解行 —— 上面那段說明裡就寫著 `plot_reasons()`，不跳過的話
    測試會抓到自己的文件而不是程式。
    """
    src = (REPO_ROOT / "scripts" / "explain.py").read_text(encoding="utf-8")
    lines = [
        (i, line) for i, line in enumerate(src.splitlines()) if not line.lstrip().startswith("#")
    ]
    main_at = next(i for i, line in lines if line.startswith("def main("))
    capture = next(i for i, line in lines if "git_sha(), git_dirty()" in line)

    for writer in ("plot_reasons(", ".write_csv(", ".write_text("):
        first_write = next(i for i, line in lines if i > main_at and writer in line)
        assert capture < first_write, f"git 狀態在第 {first_write} 行的 {writer} 之後才抓"


@NODATA
def test_rebaseline_runs_the_business_step():
    """`make rebaseline` 宣稱重跑 M1–M4，就必須真的包含 M4 的業務指標。

    少了它，manifest 會顯示「M4 已重跑」而核心交付物根本沒動過。
    """
    from scripts.rebaseline import STEPS

    names = [s[0] for s in STEPS]
    assert "m4_business" in names, f"rebaseline 缺少 M4 業務指標步驟，現有：{names}"

    step = next(s for s in STEPS if s[0] == "m4_business")
    assert "scripts/business_value.py" in " ".join(step[2])
    assert step[3] == "configs/business.yaml"
