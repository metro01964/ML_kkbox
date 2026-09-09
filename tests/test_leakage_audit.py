"""特徵工程管線的洩漏審查（2026-08-09）—— 六個漏洞，**已全數修正**。

原始版本的六條測試是紅的，每一條餵給管線一份會觸發該漏洞的輸入。修正之後
改寫成守門測試：**同樣的違規輸入，現在必須被擋下來**。

⚠️ 「餵違規輸入確認會 raise」是 SPEC §5 對每條紅線的要求，理由是只證明
「正常資料會通過」證明不了任何事 —— 一個永遠回傳 None 的空函式也會通過。

## 審查範圍與方法

逐一檢視 `src/data/cohort.py`、`src/features/build.py`、`src/features/logs.py`
的每一個「資料進入模型」的邊界，問同一個問題：

    **這個邊界能不能分辨『餵進來的東西屬於哪一個 cohort』？**

紅線 1（`assert_asof_respected`）與紅線 2（`assert_logs_within_cutoff`）都只
檢查**內部一致性** —— 「最後一筆交易不晚於 cutoff」「最近一筆日誌不晚於
cutoff」。兩者對「這個 cutoff 本身對不對」「這份日誌是不是別的 cohort 算的」
完全沒有意見。而那正是本次找到的四個漏洞的共同成因。

## 為什麼快取是重災區

三個模組都有快取，三個快取都只驗證「形狀對不對」：

    build_cohort        欄位是不是 EXPECTED_COLUMNS 的超集 + 紅線 1
    build_log_features  紅線 2（只讀一個欄位）
    narrow_logs         檔案存不存在

沒有任何一個記錄「這份快取是用什麼參數算出來的」。而快取檔名只帶 cohort
名稱，不帶 cutoff 區間、不帶窗口設定、不帶程式碼版本 —— 於是一份用錯參數
產生的快取可以完整通過所有守門，並且**分數會變好**，因此不會有人起疑。
"""

from __future__ import annotations

import polars as pl
import pytest
import yaml

from src.config import REPO_ROOT, Paths
from src.data import FEB, MAR, build_cohort
from src.data.cohort import cohort_fingerprint
from src.features import build_features, build_log_features
from src.features.logs import (
    LOG_WINDOWS,
    expected_log_columns,
    log_features_fingerprint,
    narrow_logs,
    window_bounds,
)
from src.fingerprint import write_with_fingerprint
from tests.conftest import NODATA, make_synthetic_cohort


@NODATA
def test_cohort_cache_must_belong_to_the_requested_cohort(tmp_path):
    """[高] `feb` 的快取檔裡放 Mar 的資料，目前完全通得過。

    **漏洞**：`build_cohort()` 命中快取時只檢查兩件事 —— 欄位是
    `EXPECTED_COLUMNS` 的超集、以及紅線 1（`last_tx <= cutoff`）。它**不檢查
    cutoff 是否落在該 cohort 宣告的到期區間內**。

    **為什麼是洩漏**：Mar 的 cutoff 落在三月，比 Feb 晚一個月。拿它去算 Feb
    的特徵，等於讓「二月到期的用戶」看到三月的交易 —— 而 Feb 的標籤正是由
    三月的行為決定的。這是紅線 1 的字面定義所允許的（`last_tx <= cutoff` 依然
    成立），卻是紅線 1 想擋的那件事。

    **怎麼發生**：兩台機器共用 `interim/`、複製檔案時改錯檔名、或是先跑
    `build_cohort(MAR)` 再手動改名。都不需要惡意，只要一次手滑。
    """
    paths = Paths(tmp_path)
    paths.interim.mkdir(parents=True)

    # 一張「三月」的 cohort 表：cutoff 在 2017-03-15，其餘欄位照舊。
    # 紅線 1 仍然成立（合成資料的 last_tx 全部早於 3/15），所以守門不會叫。
    mar_like = make_synthetic_cohort().with_columns(pl.lit(20170315).alias("cutoff"))
    # ⚠️ 假快取要帶**當前的程式版本指紋**，否則 `build_cohort()` 會先因為
    # 指紋不符而重算，根本走不到這條測試要驗的守門（見 src/fingerprint.py）。
    write_with_fingerprint(
        mar_like, paths.interim / "feb_cohort_asof.parquet", cohort_fingerprint()
    )

    with pytest.raises(AssertionError, match="cohort 錯置"):
        build_cohort(FEB, paths, verbose=False)


@NODATA
def test_build_features_rejects_log_features_from_another_cohort():
    """[高] 把 Mar 的收聽特徵接到 Feb 的 cohort 上，目前不會有任何抱怨。

    **漏洞**：`build_features(df, logs)` 只用 `msno` 做 left join，並呼叫
    `assert_logs_within_cutoff(logs)`。但那個守門檢查的是
    `log_min_days_before >= 0` —— **相對於這份日誌自己的 cutoff**。一份用
    Mar cutoff 算出來的日誌，它的 `log_min_days_before` 當然全部非負，
    所以守門必然放行。

    **為什麼是洩漏**：實測 90.81% 的用戶跨兩期出現，join 會成功接上絕大多數
    人。Feb 的特徵於是包含了「二月到期日之後、三月為止」的收聽行為 ——
    而那正是決定 Feb 標籤的那段時間。

    **需要的性質**：這個邊界必須能驗證日誌的出身。最小的作法是讓
    `build_log_features()` 在輸出裡帶一個 cohort 標記（或 cutoff 欄），
    由 `build_features()` 比對。目前兩者都沒有。
    """
    cohort = make_synthetic_cohort()  # cutoff 落在 2017-01 ~ 2017-03-01

    # 「另一個 cohort 算出來的」收聽特徵：對它自己的 cutoff 完全合法。
    foreign_logs = pl.DataFrame(
        {
            "msno": cohort["msno"],
            "log_min_days_before": [5] * cohort.height,
            "log_max_days_before": [80] * cohort.height,
            "log_has_logs": [1.0] * cohort.height,
        }
    )

    with pytest.raises((KeyError, ValueError, AssertionError)):
        build_features(cohort, foreign_logs)


@NODATA
def test_log_feature_cache_must_carry_the_full_feature_schema(tmp_path):
    """[中] 一份只有三欄的收聽特徵快取，目前會被原樣送進模型。

    **漏洞**：`build_log_features()` 命中快取時唯一的檢查是
    `assert_logs_within_cutoff()`，而那個函式只讀 `log_min_days_before` 一欄。
    少掉其餘 37 個特徵欄不會有任何錯誤。

    **後果**：`_attach_logs()` 會照樣 join，`build_features()` 產出一張少了
    37 欄的特徵表，而模型照樣訓練成功 —— 分數變差，但找不出原因。這是
    SPEC §2.1 註記過的同一種失敗模式（「只用 v2 等於完全沒有歷史特徵，而
    模型還是會訓練成功」）。

    **怎麼發生**：改了 `LOG_WINDOWS`（例如加一個 60 天窗口）之後忘了
    `force=True`。快取檔名不帶窗口設定，所以舊檔會被當成有效的。
    """
    paths = Paths(tmp_path)
    paths.interim.mkdir(parents=True)
    # ⚠️ 假快取要帶**當前的程式版本指紋**，否則 `build_cohort()` 會先因為
    # 指紋不符而重算，根本走不到這條測試要驗的守門（見 src/fingerprint.py）。
    write_with_fingerprint(
        pl.DataFrame({"msno": ["u0"], "log_min_days_before": [3], "log_has_logs": [1.0]}),
        paths.interim / "feb_log_features.parquet",
        log_features_fingerprint(),
    )

    # 快取被判定為缺欄位 → 走重算路徑 → 在 tmp_path 找不到原始 CSV 而 raise。
    # 這個 FileNotFoundError 正是「它拒絕沿用殘缺快取」的證據；若仍接受舊檔，
    # 函式會安靜地回傳那三欄。
    with pytest.raises(FileNotFoundError):
        build_log_features("feb", paths, verbose=False)

    # 期望的 schema 由 LOG_WINDOWS 推導，不是寫死清單 —— 加一個窗口之後
    # 這個集合會自動變大，舊快取因此對不上而重算。
    expected = expected_log_columns()
    for w in LOG_WINDOWS:
        assert f"log{w}_active_days" in expected
    assert "cutoff" in expected, "cutoff 是出身證明，必須在期望 schema 裡"


@NODATA
def test_narrow_logs_cache_is_keyed_by_the_requested_window(tmp_path):
    """[中] 收斂檔的快取只看「檔案存不存在」，不看它涵蓋哪段日期。

    **漏洞**：`narrow_logs()` 的快取路徑是固定的 `user_logs_window.parquet`，
    而快取命中的判斷只有 `out.exists()`。傳進來的 `specs` 在命中時**完全
    沒有被使用**。

    **後果**：一份只為 Feb 收斂過的檔案（日期上界 2017-02-28、只含 Feb 用戶）
    會被 Mar 的特徵計算直接沿用。Mar 用戶的日誌大量缺失、三月的日期整段
    不存在 —— 而 `assert_logs_within_cutoff` 只檢查非負，缺資料它管不著。

    **方向**：這一條造成的是「看得太少」而不是「看到未來」，因此嚴重性低於
    前兩條。但它與前兩條同源：**快取不記錄自己是用什麼參數算的**。
    """
    paths = Paths(tmp_path)
    paths.interim.mkdir(parents=True)

    # 假裝已有一份「只為 Feb 收斂」的檔：日期上界停在 2017-02-28。
    stale = paths.interim / "user_logs_window.parquet"
    pl.DataFrame({"msno": ["u0"], "date": [20161201]}).write_parquet(stale)

    # 檔名現在帶著實際的日期範圍，所以那個舊檔對 (FEB, MAR) 不再是命中 ——
    # 走重算路徑，在 tmp_path 找不到原始 CSV 而 raise。
    with pytest.raises(FileNotFoundError):
        narrow_logs(paths, specs=(FEB, MAR), verbose=False)

    # 而且不同的 specs 必須指向不同的檔名，否則兩者仍會互相覆蓋。
    lo_both, hi_both = window_bounds((FEB, MAR))
    lo_feb, hi_feb = window_bounds((FEB,))
    assert (lo_both, hi_both) != (lo_feb, hi_feb), "兩組 specs 的窗口應該不同"


@NODATA
def test_calibration_holdout_is_not_already_spent_on_other_choices():
    """[中] M4 的校準集，其實是 M3 已經用過兩次的那一塊。

    **漏洞**：`configs/feature_selection.yaml` 與 `configs/tuning.yaml` 的
    `[split]` 完全相同（同樣的 fraction、同樣的 seed），而
    `scripts/calibration_report.py` 直接讀 `tuning.yaml` 的切分。

    因此同一批 15% 的用戶依序被用來：

        1. 挑 null importance 的門檻（M3）
        2. 挑超參數（M3）
        3. fit 機率校準器（M4）

    **為什麼要在意**：M4 宣稱校準器 fit 在「模型與 early stopping 都沒看過的
    那一塊」。就模型權重而言這句話成立，但那一塊已經被用來做過兩個選擇 ——
    它作為「乾淨保留集」的身分已經被消耗過。§7.6 剛示範過同一類錯誤的代價。

    **需要的性質**：校準集必須是一塊**專屬**的切分，或至少 seed 與其他選擇
    步驟不同，讓三個決定看的不是同一批人。
    """

    def split_of(name: str) -> dict:
        path = REPO_ROOT / "configs" / name
        return (yaml.safe_load(path.read_text(encoding="utf-8")) or {})["split"]

    calibration = split_of("calibration.yaml")

    for name in ("feature_selection.yaml", "tuning.yaml"):
        assert calibration != split_of(name), (
            f"校準的切分與 {name} 完全相同 —— 同一批 15% 的用戶會被用來做三個不同的決定"
        )


@NODATA
def test_members_snapshot_vintage_is_declared():
    """[低] `members_v3` 是快照，但沒有任何地方記錄它的時點。

    **漏洞**：`build_features()` 只遮蔽「註冊日晚於 cutoff」的那幾位
    （Feb 6 人、Mar 2 人）。但 `city` / `gender` / `registered_via` / `bd`
    這些欄位反映的是**快照當下**的狀態，而快照是競賽後期才發布的
    （官方 2017-11-13 發布 v3）。

    一位用戶若在 2017 年 6 月搬家，Feb cohort 的特徵會拿到 6 月之後的城市 ——
    那是 cutoff 之後才成立的事實。

    **為什麼只列為低**：這個洩漏**無法從資料修復**（沒有屬性的歷史版本），
    量級也未知。但它必須被**寫下來**：一個沒有記錄時點的快照，讀者無從判斷
    「這些屬性有多舊」，也就無法評估風險。M6 的 MODEL_CARD 需要這個數字。

    **需要的性質**：`src/features/build.py` 宣告快照時點常數，並在文件中
    說明哪些欄位受影響。
    """
    from src.features import build

    assert hasattr(build, "MEMBERS_SNAPSHOT_DATE"), (
        "members_v3 的快照時點沒有被宣告；"
        "city / gender / registered_via / bd 反映的是快照當下而非 cutoff 當下的狀態"
    )
