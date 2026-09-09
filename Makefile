# KKBox 訂閱流失預測 —— 常用指令
#
# 每個目標都是單一 `uv run ...`，不使用 shell 內建指令（rm、cp、find 等），
# 這樣同一份 Makefile 在 Linux（CI）與 Windows 上行為一致。
#
# ⚠️ Windows 沒有內建 make。若未安裝，直接執行各目標底下的 uv 指令即可，
#    效果完全相同 —— 本檔案只是那些指令的集中索引。安裝方式見 README。

.DEFAULT_GOAL := help
.PHONY: help setup data data-all lint format test test-fast test-ci eda features ablation train compare select encode tune reverse calibrate calibrate-fit multi-seed verify-rebuild rebaseline mlflow clean eval explain lead-time artifact artifact-t7 artifact-fixed final serve drift kaggle

help:
	@echo "可用目標："
	@echo ""
	@echo "  setup      建立環境（取得 Python 3.12、建 .venv、editable 安裝本專案）"
	@echo "  data       下載 M0/M1 需要的資料（約 1.02 GB）"
	@echo "  data-all   下載全部競賽資料（8.95 GB，解壓後約 34 GB）"
	@echo ""
	@echo "  lint       ruff 檢查 + 格式檢查（不改檔案）"
	@echo "  format     ruff 自動排版（會改檔案）"
	@echo ""
	@echo "  test       全部測試（需要資料，約 30 秒）"
	@echo "  test-fast  跳過需掃大檔的測試"
	@echo "  test-ci    只跑不需資料的純邏輯測試，要求零 skip"
	@echo ""
	@echo "  eda        產生 EDA 圖表到 reports/figures/"
	@echo "  features   M2 收聽行為聚合（首次約 30 秒，之後讀快取）"
	@echo "  ablation   收聽特徵的分組消融實驗"
	@echo "  train      訓練並評估（含 5-fold 標準差與 MLflow 追蹤）"
	@echo ""
	@echo "  compare    M3 三方比較：LightGBM / XGBoost / CatBoost（約 6 分鐘）"
	@echo "  select     M3 null importance 特徵篩選 + 篩選前後對照（約 20 分鐘）"
	@echo "  encode     M3 target encoding 對照（紅線 6，約 1 分鐘）"
	@echo "  tune       M3 LightGBM 隨機搜尋 31 組（約 15 分鐘）"
	@echo "  reverse    反向時間外驗證 Mar→Feb，檢查比較結論的穩健性（約 12 分鐘）"
	@echo ""
	@echo "  calibrate  M4 校準診斷：reliability diagram + Brier/ECE（約 1 分鐘）"
	@echo "  calibrate-fit  M4 fit isotonic 校準器 + 校準前後對照（約 2 分鐘）"
	@echo ""
	@echo "  multi-seed     配對 multi-seed：量雜訊尺度 + 三家配對比較（約 35 分鐘）"
	@echo "  verify-rebuild 連續強制重建快取兩次，驗證誰在哪裡完全不變（約 6 分鐘）"
	@echo "  rebaseline     在固定基準上重跑 M1–M4 並留下完整紀錄（約 65 分鐘）"
	@echo ""
	@echo "  mlflow     開啟 MLflow UI 檢視實驗紀錄"
	@echo "  clean      清除 __pycache__ / .pytest_cache / .ruff_cache"
	@echo ""
	@echo "  eval       M4 業務指標：期望淨收益曲線 + 敏感度熱圖（約 5 分鐘）"
	@echo "  explain    M5 投放名單 + SHAP 流失原因碼（約 5 分鐘）"
	@echo "  lead-time  M6 提前 7 天評分的代價：兩版本共同子集比較（約 20 分鐘）"
	@echo "  artifact   M6 匯出模型 artifact（T=0，離線基準，約 5 分鐘）"
	@echo "  artifact-t7 M6 匯出 T−7 的 artifact —— **能上線的那一個**（約 5 分鐘）"
	@echo "  artifact-fixed M6 匯出固定評分日的 artifact（Kaggle 管線用，約 5 分鐘）"
	@echo "  serve      M6 起 FastAPI /predict（文件在 /docs）"
	@echo "  final      M6 合併 cohort 重訓最終模型 + 紅線 4 對照（約 55 分鐘）"
	@echo "  drift      M6 PSI 漂移監控報告 + 圖 16（約 3 分鐘，不重訓）"
	@echo "  kaggle     M6 產生 Apr cohort 的 Kaggle 提交檔（不自動提交）"

# --- 環境與資料 ------------------------------------------------------------

setup:
	uv sync

data:
	uv run python scripts/download.py

data-all:
	uv run python scripts/download.py --groups all

# --- 程式碼品質 ------------------------------------------------------------

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .

# --- 測試 ------------------------------------------------------------------

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

# CI 專用。GitHub Actions 上沒有原始資料（資料不進 Git），43 條測試會有 36 條
# skip —— 而一個全部 skip 的套件也會顯示綠燈。所以 CI 單獨跑標了 nodata 的
# 純邏輯測試，並要求它們全過。零 skip 的檢查在 ci.yml 裡執行。
test-ci:
	uv run pytest -m nodata

# --- 分析與訓練 ------------------------------------------------------------

eda:
	uv run python notebooks/eda_01_overview.py

# M2 收聽行為聚合。第一次要掃 31.9 GB 原始日誌（約 30 秒），之後讀快取。
features:
	uv run python scripts/features.py

# 收聽特徵的分組消融。七次訓練，約 2 分鐘。
ablation:
	uv run python scripts/ablation.py

# M1 baseline。訓練 Feb cohort、在 Mar cohort 評估，未達 SPEC §3.3 的
# 0.30746 門檻會回傳非零離開碼。
train:
	uv run python scripts/train.py

# --- M3 ---------------------------------------------------------------------

# 三方比較。三家拿到同一份特徵、同一個切分、同一塊 early stopping 驗證集。
# CatBoost 佔掉大部分時間（對稱樹 + ordered target statistics 較慢）。
compare:
	uv run python scripts/compare.py

# Null importance 特徵篩選。真實 1 次 + 打亂標籤 20 次 + 各門檻重訓一次。
select:
	uv run python scripts/select_features.py

# Target encoding 對照（含三個刻意違規的控制組，證明紅線 6 不是空話）。
encode:
	uv run python scripts/target_encoding.py

# 超參數隨機搜尋。搜尋全程只用 Feb cohort，Mar 只在最後看一次。
tune:
	uv run python scripts/tune.py

# 反向時間外驗證。⚠️ Mar→Feb 是時間倒流，不是部署估計，只用於檢查
# 「三方比較的排名」是不是單月雜訊。
reverse:
	uv run python scripts/reverse_validation.py

# --- M4 ---------------------------------------------------------------------

# 校準診斷。**不 fit 任何校準器** —— 先看失準的形狀，再決定用 isotonic
# 還是 Platt。產出 reports/figures/09、10 兩張圖。
calibrate:
	uv run python scripts/calibration_report.py

# 校準器本身。fit 在 Feb-sel，套到 Mar，並把代價（排序解析度）一起量出來。
# 最後一列 Mar-oracle 是**故意違規**的洩漏對照組，只當上界，不可上線。
calibrate-fit:
	uv run python scripts/calibrate.py

# 配對 multi-seed：8 個 seed × 3 家，三家共用同一次切分。量新的雜訊尺度，
# 並用配對差的 95% CI 判定三家高下（約 35 分鐘）。
multi-seed:
	uv run python scripts/multi_seed.py

# --- 可重現性 --------------------------------------------------------------

# 連續強制重建 cohort 與收聽特徵，比對「誰在哪裡」。§7.8 的修正加了
# .sort("msno")，但沒有人驗證過它 —— 這支補上，指紋存到 reports/。
verify-rebuild:
	uv run python scripts/verify_rebuild.py --rounds 2

# 在固定基準上重跑 M1–M4，每一步記錄 git SHA / 指令 / 設定 / seed / 耗時。
# ⚠️ 約 65 分鐘。只想確認快的那幾步用 --quick。
rebaseline:
	uv run python scripts/rebaseline.py

mlflow:
	uv run mlflow ui --backend-store-uri sqlite:///mlflow.db

clean:
	uv run python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path('.pytest_cache'), pathlib.Path('.ruff_cache')]]; print('已清除 __pycache__ / .pytest_cache / .ruff_cache')"

# --- M4 業務指標 -------------------------------------------------------------

# M4 業務指標。用 §7.12 正式採用的 CatBoost 產生機率，畫期望淨收益曲線與
# 敏感度熱圖。參數在 configs/business.yaml。約 5 分鐘。
eval:
	uv run python scripts/business_value.py

# --- M5 原因碼 ---------------------------------------------------------------

# M5 投放名單與流失原因碼。名單 = 所有 p > p* 的人，p* 與 M4 共用同一份推導。
#
# ⚠️ 輸出的 CSV **不是資產**，是這一次執行的結果（*.csv 在 .gitignore 裡，
#    進不了 git）。改 configs/business.yaml 的 C_offer，名單大小就會變 ——
#    那是正常的，它是函式的輸出。進 git 的只有 manifest.json，而它只放
#    provenance 與彙總，不放逐人的列（競賽規則：資料不進 Git）。
#
# 單一用戶查詢：uv run python scripts/explain.py --msno <msno> ...
explain:
	uv run python scripts/explain.py

# --- M6 -----------------------------------------------------------------------

# 提前 7 天評分的代價（SPEC §4.3）。訓練兩個版本（到期日 / 提前 7 天）並在
# **共同子集**上比較 —— 兩邊的 cohort 成員不同，各自驗證集的分數不可直接比。
#
# ⚠️ 第一次跑會重建四份快取（`lead_days` 進了 CohortSpec，邏輯指紋因此改變，
#    但 T=0 的內容不變），並多產生一份日期下界往前 7 天的收斂檔。約 20 分鐘。
lead-time:
	uv run python scripts/lead_time.py

# 匯出模型 artifact。服務、HF Spaces Demo、Kaggle 推論管線都載這一份，沒有人
# 重訓（CatBoost 一次約 4 分鐘）。
#
# ⚠️ 預設匯出的是 **T=0** 的版本，那是離線基準、**不可上線**（§4.3）。能上線的
#    是提前 7 天的那個，也是 configs/serving.yaml 預設載入的那個：
#
#        uv run python scripts/export_model.py --design lead7
#
#    第三種設計 `--design fixed`（固定評分日）是 Kaggle 管線用的：交易與日誌只到
#    2017-03-31，而測試集要預測 4 月到期的人 —— 「到期日 − 7 天」對 77.64% 的測試
#    用戶會落在資料結束之後（見 src/data/cohort.py 的 assert_data_covers_cutoffs）。
#
#    只改了 configs/business.yaml（C_offer / r_save）時加 --reuse，不必重訓。
artifact:
	uv run python scripts/export_model.py

artifact-t7:
	uv run python scripts/export_model.py --design lead7

artifact-fixed:
	uv run python scripts/export_model.py --design fixed

# 上線前的最終模型：合併 feb_fixed + mar_fixed 重訓（紅線 4 的觸發點，SPEC §7.20）。
#
# ⚠️ 合併之後**沒有時間外驗證集**了 —— 兩個帶標籤的 cohort 都進了訓練集。這份
#    artifact 的每一個本地分數都是同分布估計，**不可**與 catboost_fixed 的
#    0.17342 比大小。實測同分布 CV 樂觀 13.0%。
#
# 四段切分（train/es/sel/cal）全部綁 msno，並跑一組違規對照（不看 msno 的
# StratifiedKFold）量紅線 4 擋掉多少。9 次 CatBoost 訓練，約 55 分鐘；
# 只要模型不要對照組時加 --no-cv。
final:
	uv run python scripts/final_model.py

# 起 /predict 服務。載哪一份 artifact 看 configs/serving.yaml（預設 T−7）。
# 文件在 http://127.0.0.1:8000/docs —— OpenAPI 就是這個服務的說明書。
#
# ⚠️ artifact 不存在會**啟動失敗**，不是起一個沒有模型的服務 —— 後者會在第一筆
#    請求時才壞，而那通常是在別人的 Demo 上。先跑 make artifact-t7。
serve:
	uv run uvicorn src.serving.app:app --host 127.0.0.1 --port 8000

# PSI 漂移監控。**不重訓** —— 模型從 artifact 載入（那是 artifact 的用途之一）。
#
# 報三個數字：特徵 PSI、分數 PSI、以及實際的標籤漂移。第三個在部署時拿不到，
# 量它的目的是回答「這套監控會漏掉什麼」。實測答案是：**基準率漂移它看不到**。
drift:
	uv run python scripts/drift_report.py

# Apr cohort 的 Kaggle 提交檔（SPEC §3.3 的最後一條門檻）。**不自動提交** ——
# 提交要接受競賽規則，那是帳號層級的動作，腳本只印出指令。
#
# ⚠️ 需要 `--design fixed` 的 artifact：交易與日誌只到 2017-03-31，而測試集要預測
#    4 月到期的人。本地**算不出**這份預測的分數（測試集沒有標籤），所以腳本會先印出
#    結構相同的本地對照當「事前登記的預期」。
kaggle:
	uv run python scripts/predict_kaggle.py
