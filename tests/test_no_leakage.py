"""紅線測試 —— SPEC §5 的八條紅線。

SPEC §5 開宗明義：「以下任何一條被違反，該版本作廢重做。每一條都要有對應的
**會失敗的測試**。」

「會失敗的測試」的重點在於：光證明「現在的資料通過檢查」是不夠的，因為一個
永遠回傳 True 的空檢查也會通過。必須能餵給它一份確實違規的輸入，確認它真的
會擋下來。下面 test_red_line_1 就是這樣寫的。

M0 時八條裡只有 4 條寫得出來（1、3、7、8），另外 4 條依賴還不存在的程式碼 ——
為不存在的東西寫測試只能寫出假的通過，所以它們以 skip 保留在清單裡，每次
pytest 都提醒還欠幾條。M1 補上紅線 5、M3 補上 2 與 6，**M6 補上最後的紅線 4**
（合併 cohort 重訓最終模型時才真的觸發，見 SPEC §7.4 與 §7.20）。八條到齊。
"""

from __future__ import annotations

import math
import re

import polars as pl
import pytest

from src.config import REPO_ROOT
from src.data import assert_asof_respected
from src.evaluation import EPS, constant_log_loss, log_loss
from src.features import assert_logs_within_cutoff, build_features
from src.features.encoding import (
    assert_encoding_is_oof,
    fit_target_encoder,
    oof_target_encode,
)
from src.models.grouped import (
    SEGMENTS,
    assert_groups_disjoint,
    fold_report,
    four_way_group_split,
    grouped_folds,
    merge_cohorts,
    random_folds_violating_red_line_4,
)
from tests.conftest import NODATA, SLOW, make_synthetic_cohort, make_two_cohorts

# 測試觀察期的上界。SPEC 紅線 7：不得使用 2017-04 之後的任何資料。
MAX_ALLOWED_DATE = 20170430


# ===========================================================================
# 紅線 1 · 任何 transaction_date > cutoff 的交易不得進入特徵
# ===========================================================================
# 「到期日之後的交易就是標籤」—— 這是本題的頭號洩漏。


@NODATA
def test_red_line_1_guard_catches_violation():
    """守門函式必須能擋下違規的表。

    這是「會失敗的測試」本體：故意造一張 b 用戶交易日晚於 cutoff 的表，
    確認 assert_asof_respected 真的會 raise。如果哪天有人把守門邏輯改壞
    （例如比較符號寫反），這個測試會紅。
    """
    bad = pl.DataFrame(
        {
            "msno": ["a", "b"],
            "cutoff": [20170228, 20170228],
            "last_tx": [20170227, 20170301],  # b 的最後一筆交易在 cutoff 之後
        }
    )
    with pytest.raises(AssertionError, match="紅線 1 違反"):
        assert_asof_respected(bad)


@NODATA
def test_red_line_1_guard_accepts_clean_table():
    """乾淨的表要能通過，否則守門太嚴會擋掉正常流程。

    邊界值 cutoff == last_tx 必須通過：用戶在到期日當天交易是合法的，
    那筆資料在預測時點確實看得到。
    """
    good = pl.DataFrame(
        {
            "msno": ["a", "b"],
            "cutoff": [20170228, 20170216],
            "last_tx": [20170201, 20170216],  # b 剛好等於 cutoff，邊界值
        }
    )
    assert_asof_respected(good)  # 不應 raise


@NODATA
def test_red_line_1_guard_rejects_missing_columns():
    """欄位缺失要明確報錯，不能默默視為通過。"""
    with pytest.raises(KeyError):
        assert_asof_respected(pl.DataFrame({"msno": ["a"]}))


@SLOW
@pytest.mark.parametrize("fixture_name", ["feb_cohort", "mar_cohort"])
def test_red_line_1_real_cohorts_are_clean(request, fixture_name: str):
    """實際產出的兩個 cohort 特徵表都必須通過守門。"""
    df = request.getfixturevalue(fixture_name)
    assert_asof_respected(df)


# ===========================================================================
# 紅線 3 · 禁用 members.csv，只能用 members_v3.csv
# ===========================================================================
# members.csv 的 expiration_date 是快照欄位，記錄「資料匯出當下」的到期日。
# 官方在 2017-11-13 發布 v3 就是為了移除它。


@NODATA
def test_red_line_3_no_code_reads_members_csv():
    """程式碼裡不得出現對 members.csv 的引用。

    只比對「被引號包起來的檔名」，不比對純文字 —— 註解裡寫「不能用
    members.csv」是說明，不是引用，不該讓測試失敗。
    """
    pattern = re.compile(r"""["']members\.csv["']""")
    offenders = []

    for folder in ("src", "scripts", "notebooks", "tests"):
        for path in (REPO_ROOT / folder).rglob("*.py"):
            if pattern.search(path.read_text(encoding="utf-8")):
                offenders.append(str(path.relative_to(REPO_ROOT)))

    assert not offenders, (
        f"這些檔案引用了 members.csv（紅線 3 禁用，只能用 members_v3.csv）：{offenders}"
    )


def test_red_line_3_v3_lacks_the_leaky_column(raw_files):
    """members_v3.csv 不得含 expiration_date。

    與 test_data_contract.py 的斷言 9 重複是刻意的：契約測試檢查的是
    「資料長得對不對」，這裡檢查的是「紅線有沒有守住」。兩份清單各自完整，
    刪掉任一支都不會讓紅線失去守門。
    """
    assert "expiration_date" not in raw_files("members_v3.csv").collect_schema().names()


# ===========================================================================
# 紅線 7 · 不得使用 2017-04（測試觀察期）之後的任何資料
# ===========================================================================


@SLOW
@pytest.mark.parametrize("fixture_name", ["feb_cohort", "mar_cohort"])
def test_red_line_7_no_future_dates(request, fixture_name: str):
    """特徵表裡所有日期欄位都不得晚於 2017-04-30。

    涵蓋 cutoff、first_tx、last_tx 與 registration_init_time。最後那個是
    容易漏掉的一個 —— 它來自 members_v3，不受 as-of 截斷保護。
    """
    df = request.getfixturevalue(fixture_name)
    date_cols = ["cutoff", "first_tx", "last_tx", "registration_init_time"]

    for col in date_cols:
        worst = df[col].max()
        if worst is None:  # 整欄皆 null（例如全部不在 members_v3）
            continue
        assert worst <= MAX_ALLOWED_DATE, f"{col} 最大值 {worst} 晚於 {MAX_ALLOWED_DATE}"


@SLOW
def test_red_line_7_cutoff_inside_cohort_window(feb_cohort, mar_cohort):
    """cutoff 必須落在該 cohort 宣告的到期區間內。

    這條擋的是「cohort 定義被改壞」—— 例如日期區間打錯一位數，
    模型還是會訓練成功，只是訓練在錯的族群上。
    """
    for df, (lo, hi) in ((feb_cohort, (20170201, 20170228)), (mar_cohort, (20170301, 20170331))):
        assert df["cutoff"].min() >= lo
        assert df["cutoff"].max() <= hi


# ===========================================================================
# 紅線 8 · 本地評估必須套用官方的 clip(1e-15, 1-1e-15)
# ===========================================================================


@NODATA
def test_red_line_8_clip_prevents_infinity():
    """極端預測必須產生有限值。

    沒有 clip 的話 ln(0) = -∞，整個評估變成 inf，一筆極端錯誤就摧毀所有
    資訊。官方的 clip 把單筆懲罰上限定在 -ln(1e-15) ≈ 34.54。
    """
    # 低端：y=1 而 p=0，p 被 clip 成 EPS。
    loss_low = log_loss([1], [0.0])
    assert math.isfinite(loss_low)
    assert loss_low == pytest.approx(-math.log(EPS), rel=1e-9)
    assert loss_low == pytest.approx(34.5387763949, rel=1e-9)

    # 高端：y=0 而 p=1，p 被 clip 成 1-EPS。
    #
    # ⚠️ 這一側**不等於** -ln(EPS)。1-1e-15 在 float64 裡無法精確表示：
    # 1 附近的 double 間距是 2^-53 ≈ 1.11e-16，所以 1-1e-15 會落在距離 1
    # 約 9.992e-16 的那個 double 上，而不是剛好 1e-15。結果是
    #     -ln(1 - (1-EPS)) ≈ 34.5396   （低端是 34.5388）
    #
    # 官方計分（sklearn 的 log_loss）用完全相同的 clip(eps, 1-eps)，
    # 因此有完全相同的不對稱。這裡照抄而不「修正」—— 紅線 8 要求的是
    # 與官方一致，不是數值上最漂亮。改用 log1p 讓兩側對稱，反而會讓
    # 本地分數與 LB 對不起來，正好違背這條紅線的目的。
    loss_high = log_loss([0], [1.0])
    assert math.isfinite(loss_high)
    assert loss_high == pytest.approx(-math.log(1.0 - (1.0 - EPS)), rel=1e-12)

    # 兩側幾乎相同但不完全相同。差異只在 p 落到極端值時才出現，
    # 對實際模型輸出（p 通常在 0.001~0.999）沒有影響。
    assert abs(loss_high - loss_low) < 1e-3


@NODATA
def test_red_line_8_matches_hand_computation():
    """與手算的公式對照，確認實作沒寫錯。

    公式：logloss = -(1/N) Σ [ y·ln(p) + (1-y)·ln(1-p) ]
    """
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.8, 0.3]
    expected = -sum(
        yi * math.log(pi) + (1 - yi) * math.log(1 - pi) for yi, pi in zip(y, p, strict=True)
    ) / len(y)
    assert log_loss(y, p) == pytest.approx(expected, rel=1e-12)


@NODATA
def test_red_line_8_rejects_bad_input():
    """長度不符或空輸入要明確報錯，不能靜靜回傳一個數字。"""
    with pytest.raises(ValueError):
        log_loss([1, 0], [0.5])
    with pytest.raises(ValueError):
        log_loss([], [])


@SLOW
def test_m1_baseline_threshold(feb_cohort, mar_cohort):
    """M1 驗收門檻必須是 0.30746（SPEC §3.3）。

    用 Feb cohort 的流失率對 Mar cohort 做常數預測。這個數字是 M1 的
    及格線 —— 打不贏它的模型沒有存在意義。把它鎖進測試，之後就不會
    有人記錯門檻。
    """
    feb_rate = feb_cohort["is_churn"].mean()
    baseline = constant_log_loss(feb_rate, mar_cohort["is_churn"])
    assert round(baseline, 5) == 0.30746


# ===========================================================================
# M0 當時還寫不出來的四條 —— 紅線 2 / 4 / 5 / 6
# ===========================================================================
# 它們曾經以 `@pytest.mark.skip(reason=…)` 掛在這裡，讓每次 pytest 都列出
# 「還欠哪幾條、被哪個里程碑擋著」。M6 補上紅線 4 之後這份清單清空 ——
# 一條 skip 都不剩，八條紅線各自有一個餵違規資料會 raise 的守門測試。


@NODATA
def test_red_line_2_guard_catches_post_cutoff_logs():
    """守門函式必須擋下含 cutoff 之後日誌的特徵表。

    與紅線 1 同樣的測法：餵一張 u1 的最近日誌晚於自己 cutoff 的表，
    確認 assert_logs_within_cutoff 會 raise。

    `log_min_days_before` 是「最近一筆日誌距離 cutoff 幾天」，負值代表那筆
    日誌發生在到期日之後 —— 到期後的收聽行為是結果不是原因，讓它進特徵
    等於用未來預測過去。
    """
    bad = pl.DataFrame({"msno": ["u0", "u1"], "log_min_days_before": [0, -3]})
    with pytest.raises(AssertionError, match="紅線 2 違反"):
        assert_logs_within_cutoff(bad)


@NODATA
def test_red_line_2_guard_accepts_same_day_logs():
    """cutoff 當天的日誌必須通過（邊界值 0）。

    用戶在到期日當天聽歌是合法的，那筆資料在評分時點確實看得到。
    實測 Feb cohort 的 log_min_days_before 中位數就是 0 —— 多數人到期
    當天仍在使用，把 0 擋掉會誤殺一半以上的資料。
    """
    good = pl.DataFrame({"msno": ["u0", "u1"], "log_min_days_before": [0, 45]})
    assert_logs_within_cutoff(good)


@NODATA
def test_red_line_2_guard_rejects_missing_column():
    """缺欄位要明確報錯，不能默默視為通過。"""
    with pytest.raises(KeyError):
        assert_logs_within_cutoff(pl.DataFrame({"msno": ["u0"]}))


@SLOW
@pytest.mark.parametrize("cohort_name", ["feb", "mar"])
def test_red_line_2_real_log_features_are_clean(paths, cohort_name: str):
    """實際產出的兩個 cohort 收聽特徵表都必須通過守門。"""
    from src.features import build_log_features

    if not (paths.raw / "user_logs.csv").exists():
        pytest.skip("user_logs.csv 不存在，請執行 download.py --groups logs")
    assert_logs_within_cutoff(build_log_features(cohort_name, paths, verbose=False))


@NODATA
def test_red_line_4_guard_catches_crossing_msno():
    """守門函式必須擋下同一個 msno 跨段的切分。

    這是紅線 4 的「會失敗的測試」本體：手刻一組 c 同時在 train 與 valid 的
    切分，確認 `assert_groups_disjoint` 真的會 raise。

    守門收的是 msno 而不是 splitter 物件，因為要檢查的性質只跟「誰在哪一段」
    有關 —— 用什麼工具切的、切幾折都不影響這個判斷，也因此繞不過去。
    """
    with pytest.raises(AssertionError, match="紅線 4 違反"):
        assert_groups_disjoint(
            {
                "train": pl.Series(["a", "b", "c"]),
                "valid": pl.Series(["c", "d"]),  # c 兩邊都在
            }
        )


@NODATA
def test_red_line_4_guard_accepts_disjoint_split():
    """真正不重疊的切分要能通過，否則守門太嚴會擋掉正常流程。"""
    assert_groups_disjoint(
        {
            "train": pl.Series(["a", "b"]),
            "es": pl.Series(["c"]),
            "sel": pl.Series(["d"]),
            "cal": pl.Series(["e", "f"]),
        }
    )


@NODATA
def test_red_line_4_random_kfold_on_merged_cohorts_is_caught():
    """合併 cohort 後用不分群的 KFold —— 守門必須抓到。

    這條是紅線 4 的真正內容：它擋的不是一組手刻的壞索引，而是**最自然的
    那個寫法**。`StratifiedKFold` 在單一 cohort 內部完全正確（M1/M3 一直
    這樣用），把同一段程式碼套到合併資料上就變成洩漏，而程式碼看起來沒有
    任何可疑之處。
    """
    merged = merge_cohorts(make_two_cohorts())
    assert merged.n_shared > 0, "合成資料沒有跨期用戶，這個測試會空過"

    tr, va = random_folds_violating_red_line_4(merged, n_splits=4, seed=0)[0]
    with pytest.raises(AssertionError, match="紅線 4 違反"):
        assert_groups_disjoint(
            {
                "train": merged.fs.msno[pl.Series(tr)],
                "valid": merged.fs.msno[pl.Series(va)],
            }
        )


@NODATA
def test_red_line_4_grouped_kfold_keeps_every_user_on_one_side():
    """合規的折法：每一折都沒有人跨邊。

    `grouped_folds()` 自己每折都跑守門，所以這裡真正測的是「它沒有把守門
    關掉」以及「跨邊人數確實是 0」——後者用 `fold_report()` 獨立算一次，
    不共用守門的邏輯。兩者都通過才算數。
    """
    merged = merge_cohorts(make_two_cohorts())
    folds = grouped_folds(merged, n_splits=4, seed=0)

    report = fold_report(merged, folds)
    assert report["跨邊人數"].to_list() == [0, 0, 0, 0]
    # 每一列都要落在某一折的驗證集裡，否則「沒有人跨邊」可以靠少切幾折達成。
    assert sum(len(va) for _, va in folds) == merged.n_rows


@NODATA
def test_red_line_4_four_way_split_is_disjoint_by_msno():
    """§7.6 的四段切分：任兩段都不共用 msno，且四段涵蓋每一列。

    紅線 4 的字面規範是 KFold，但**同一個結構出現在四段切分上**：cal 與
    train 若共用同一個人，校準器就是在模型看過的人身上 fit 的。
    """
    merged = merge_cohorts(make_two_cohorts())
    split = four_way_group_split(
        merged,
        {"fractions": {"train": 0.7, "es": 0.1, "sel": 0.1, "cal": 0.1}, "seed": 20260810},
    )

    assert_groups_disjoint({s: split.segment(s).msno for s in SEGMENTS})
    assert sum(split.segment(s).X.height for s in SEGMENTS) == merged.n_rows
    # 跨期用戶必須整組落在同一段 —— 這才是四段切分與「隨機切列」的差別。
    assert sum(split.segment(s).msno.n_unique() for s in SEGMENTS) == merged.n_groups


@NODATA
def test_red_line_5_feature_builder_is_stateless():
    """特徵建構不得依賴整批資料的統計量。

    **測法**：對完整資料建一次特徵，再對其中一個子集建一次，比對相同那幾列
    的值是否逐格相同。

    為什麼這樣測得出來：任何「從資料學來的」轉換 —— `fillna(df.mean())`、
    標準化、類別頻率編碼、target encoding —— 算出來的統計量都會隨輸入的
    列集合而變。子集的平均數不等於全集的平均數，於是同一位用戶在兩次呼叫
    中會得到不同的特徵值，這個測試就會紅。

    反過來說，只要這個測試是綠的，紅線 5 就不可能被違反 —— 因為根本沒有
    跨列的統計量存在，也就沒有東西可以從驗證集倒灌進訓練集。

    這比「小心翼翼地在每個 fold 內 fit」可靠得多。SPEC §5 註明這條
    「AI 產生的程式碼幾乎必犯」，而最好的防法是讓它無從犯起。

    ⚠️ M3 若引入 target encoding，本測試會失敗 —— 那是正確的行為。屆時
    編碼必須移進 fold 內的 pipeline，並由紅線 6 的測試接手守門。
    """
    full = make_synthetic_cohort()
    subset_idx = [0, 2, 5, 7]

    from_full = build_features(full).X[subset_idx]
    from_subset = build_features(full[subset_idx]).X

    assert from_full.columns == from_subset.columns
    assert from_full.equals(from_subset), (
        "同一位用戶在完整資料與子集上算出不同的特徵值 —— "
        "代表特徵建構用到了跨列的統計量，違反紅線 5。"
    )


@NODATA
def test_red_line_6_target_encoding_is_oof():
    """target encoding 必須 out-of-fold。

    payment_method_id 是高基數類別（cohort 內實測 33 種），直接 target
    encode 會讓 CV 飆高、實測崩盤。

    **測法**：餵一份「每個類別只出現一次」的資料。這是洩漏的極端形式 ——
    naive 編碼下每一格的分母只有自己那一列，編碼值必然是自己的標籤（平滑
    前）。OOF 編碼則因為該類別在其他折從未出現，只能退回全體先驗。

    兩者的差別因此是**質的**而不是量的：naive 的編碼與標籤完全相關，
    OOF 的編碼是一個常數。分不出這兩者的實作會被這個測試擋下來。
    """
    n = 20
    values = pl.Series("cat", [f"c{i}" for i in range(n)])  # 每個類別各一列
    y = pl.Series("y", [i % 2 for i in range(n)])

    oof = oof_target_encode(values, y, n_splits=5, seed=0, smoothing=0.0)

    # 其他折從未見過這個類別 → 一律退回該折的先驗，不含自己的標籤。
    assert oof.n_unique() <= 5, f"OOF 編碼出現 {oof.n_unique()} 種值，疑似洩漏了自己的標籤"
    assert_encoding_is_oof(oof, y)


@NODATA
def test_red_line_6_guard_catches_naive_encoding():
    """守門必須擋下 naive（全表擬合）的 target encoding。

    與紅線 1、2 同樣的測法：真的算一份違規的編碼餵進去，確認它會 raise。
    光證明合規的資料會通過是不夠的 —— 一個永遠回傳 True 的空檢查也會通過。
    """
    n = 20
    values = pl.Series("cat", [f"c{i}" for i in range(n)])
    y = pl.Series("y", [i % 2 for i in range(n)])

    # smoothing=0：不往先驗拉，讓「每個類別只有自己一列」的洩漏完全暴露。
    naive = fit_target_encoder(values, y, smoothing=0.0).transform(values)
    assert naive.to_list() == y.cast(pl.Float64).to_list(), "naive 編碼應該逐格等於標籤"

    with pytest.raises(AssertionError, match="紅線 6 違反"):
        assert_encoding_is_oof(naive, y)


@NODATA
def test_red_line_6_smoothing_shrinks_rare_categories():
    """平滑必須把小樣本類別拉回先驗，且不影響大樣本類別。

    平滑與 OOF 是兩件不同的事（見 encoding 模組註解）：OOF 擋標籤洩漏，
    平滑擋小樣本雜訊。兩者都做才安全，所以兩者都要有測試。
    """
    # rare 出現 1 次（標籤 1），common 出現 100 次（標籤全 0）。先驗約 0.0099。
    values = pl.Series("cat", ["rare"] + ["common"] * 100)
    y = pl.Series("y", [1] + [0] * 100)

    enc = fit_target_encoder(values, y, smoothing=10.0)
    assert enc.mapping["rare"] < 0.15, "只出現一次的類別不該拿到接近 1.0 的編碼"
    assert enc.mapping["rare"] > enc.prior, "但它確實比先驗高一點，訊號不該被抹平"
    assert enc.mapping["common"] < enc.prior, "出現 100 次的類別幾乎不受平滑影響"
