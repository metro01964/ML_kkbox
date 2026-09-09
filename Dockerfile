# KKBox 流失預測推論服務。
#
# 一份 Dockerfile 同時服務三個目標：
#
#   Koyeb / Render      平台注入 $PORT，CMD 會讀它
#   Hugging Face Space  固定 7860，正好是這裡的預設值
#   本機                docker run -p 8000:7860
#
# 放在 repo 根目錄而不是 deploy/：HF Spaces **要求** Dockerfile 在根目錄，
# 而 Koyeb/Render 也預設找這裡。少一個要在平台 UI 裡設定的欄位，少一個踩雷點。

FROM python:3.12-slim

# 非 root 使用者。
#
# HF Spaces 硬性要求（容器以 UID 1000 執行，寫不進 root 擁有的路徑）。
# Koyeb 與 Render 不強制，但沒有理由讓推論服務有 root 權限。
RUN useradd --create-home --uid 1000 app

# OpenMP 執行期函式庫。
#
# python:3.12-slim 沒有它，而 LightGBM 的 Linux wheel 假設系統有 —— import 時用
# ctypes 開原生 .so 會直接 OSError: libgomp.so.1: cannot open shared object file。
#
# 服務用的是 CatBoost，一行 LightGBM 都沒跑到。但 src/serving/artifact.py 匯入
# src/models/candidates.py，而 src/models/__init__.py 會把 compare.py → train.py
# 一路拉進來，train.py 在模組層級 `import lightgbm`。所以它照樣要能載入。
#
# 治本做法是把那個 import 改成延遲的（像 candidates.py 對 catboost 做的那樣），
# 這樣連 lightgbm 都不用進映像檔。沒有先做，是因為那要動已驗證的訓練程式碼，
# 而本機沒有 Docker 可以驗證修改後的映像檔。裝一個 200 KB 的系統套件風險低得多，
# 而且順帶保障 catboost 與 scikit-learn 可能的 OpenMP 需求 —— 它們在 slim 上
# 還沒跑到就先被 LightGBM 擋下了，不代表它們沒事。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 依賴先裝、程式碼後複製 —— 這樣改一行程式碼不會讓依賴層失效重裝。
COPY deploy/requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# catboost 單獨用 --no-deps 裝：它宣告的 matplotlib / plotly / graphviz 只服務
# 繪圖 API，載模型與 TreeSHAP 用不到。實測 import 與 load_model 都正常。
# 省下約 200 MB —— 在 512 MB 記憶體的免費方案上這不是小事。
# 它真正需要的 numpy / pandas / scipy / six 已經在 requirements.txt 裡。
RUN pip install --no-cache-dir --no-deps catboost==1.2.10

# 程式碼。只複製推論走得到的層 ——
# scripts/、tests/、notebooks/、reports/ 進映像檔沒有用處。
COPY src/ ./src/

# 容器專用設定（msno 介面關閉，理由見該檔）。
COPY deploy/serving.yaml ./configs/serving.yaml

# Demo 頁「營運視角」用的聚合統計（曲線 100 點 + 幾個彙總值，約 3 KB）。
# 由 scripts/demo_curve.py 在有資料的機器上預算好 —— 即時算需要 3.0 GB 的
# cohort 快取，而那份依授權不進映像檔。
COPY deploy/demo_curve.json ./deploy/demo_curve.json

# 模型 artifact（2.5 MB）。由 `make deploy-artifact` 從 <data_root>/artifacts
# 複製進 repo —— 映像檔的建置環境（Koyeb/Render/HF 的 builder）沒有 D 槽。
#
# 目錄名保留 `catboost_lead7d` 而不是簡化成 `artifact`：服務以目錄的 basename
# 當 artifact 名稱回報在 /health，叫 `artifact` 等於在 Demo 上宣告「這個模型沒有
# 名字」，也丟掉了名稱帶的資訊（lead7d 指的是 T−7 那個 cutoff 設計）。
COPY deploy/artifacts/catboost_lead7d/ ./catboost_lead7d/

# MODEL_ARTIFACT 優先於 data_root/artifacts/<name>（src/serving/artifact.py:207），
# 所以容器裡既不需要 configs/paths.yaml 也不需要 data_root 的目錄結構 ——
# 直接指到那一份 artifact 就好。這是 .gitignore 第 15 行預告過的用法。
ENV MODEL_ARTIFACT=/app/catboost_lead7d \
    PYTHONUNBUFFERED=1 \
    PORT=7860

USER app
EXPOSE 7860

# shell 形式而不是 exec 形式，因為要展開 ${PORT}。
# 平台有注入就用平台的，沒有就用 7860（HF Spaces 的固定值）。
CMD uvicorn src.serving.app:app --host 0.0.0.0 --port ${PORT}
