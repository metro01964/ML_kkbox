# 部署

`/predict` 推論服務的容器化與託管。M6 的最後一塊。

---

## 為什麼不是 Hugging Face Spaces

本專案原訂部署到 Hugging Face Spaces（**免費**）。**該敘述在 2026-08
已不成立**：HF 改為 Static Space 免費，Docker 與 Gradio Space 需要 PRO 訂閱
（$9/月），免費帳號只剩 2 個 ZeroGPU 的 Gradio Space 額度。

本專案改用容器託管平台。**規定的意圖（一個公開、點得進去的 Demo 連結）完全滿足**，
而且 `Dockerfile` 是平台中立的 —— 哪天訂了 PRO 要搬回 HF，同一份檔案直接用。

---

## 這個目錄有什麼

| 檔案 | 用途 |
|---|---|
| `requirements.txt` | 推論專用的依賴，**不是** `pyproject.toml` 的子集抄寫，是量出來的（見該檔說明） |
| `serving.yaml` | 容器內的服務設定。與 repo 根目錄那份差在 `msno_lookup` 關閉 |
| `artifacts/catboost_lead7d/` | 部署用的模型（2.5 MB）。`.gitignore` 對它開了例外，理由見該檔 |

`Dockerfile` 與 `.dockerignore` 在 repo 根目錄 —— HF Spaces 要求 Dockerfile 在根，
Koyeb / Render 也預設找那裡。

---

## 實測數字（2026-08-12，本機精簡環境）

| 項目 | 值 |
|---|---|
| 服務行程 RSS（載完模型） | 202 MB |
| 跑完一次推論後 | **217 MB** |
| 免費方案給的記憶體 | 512 MB |
| 冷啟動到 `/health` 回 200 | 約 1 秒 |

餘裕充足。這也是為什麼 `requirements.txt` 沒有為了再省 100 MB 去重構六個檔案的
模組層級 import —— 先量再決定。

---

## 部署到 Koyeb

免費方案 1 個服務、0.1 vCPU / 512 MB、**不休眠**，通常不要信用卡。

1. https://app.koyeb.com/ 用 GitHub 登入
2. **Create Web Service** → **GitHub** → 選這個 repo
3. Branch 選 `feat/m6-deploy`（合併回主線之後改成主線分支）
4. **Builder** 選 **Dockerfile**，路徑保持 `Dockerfile`（根目錄）
5. **Instance** 選 `Free`
6. **Exposed port** 填 `8000` — Koyeb 會注入 `$PORT`，`CMD` 會讀它
7. Deploy

健康檢查路徑設 `/health`（它不載 cohort 快取，很輕）。

## 部署到 Render

免費方案要綁信用卡，且閒置 15 分鐘後休眠（冷啟動約 50 秒）。

1. https://dashboard.render.com/ → **New** → **Web Service**
2. 接上 GitHub repo，Branch 選 `feat/m6-deploy`
3. **Language** 選 **Docker**
4. **Instance Type** 選 `Free`
5. Health Check Path 填 `/health`
6. Create Web Service

Render 會自動注入 `$PORT`，不需要額外設定。

## 之後搬回 Hugging Face（有 PRO 時）

只差兩件事，程式碼完全不用改：

1. Space 的 `README.md` 開頭要一段 YAML front-matter：

   ```yaml
   ---
   title: KKBox Churn Prediction
   emoji: 📉
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   pinned: false
   ---
   ```

2. 加 remote 後推上去：

   ```bash
   git remote add space https://huggingface.co/spaces/<帳號>/<space名稱>
   ```

`Dockerfile` 的 `PORT` 預設就是 7860（HF 的固定值），所以不用動。

---

## 本機驗證（需要 Docker）

```bash
docker build -t kkbox-churn .
```

```bash
docker run --rm -p 8000:7860 kkbox-churn
```

然後開 http://localhost:8000/docs

---

## Demo 怎麼用

服務沒有自訂前端，**用 FastAPI 自動產生的 `/docs`**（Swagger UI）。這是刻意的選擇：
它可以直接在網頁上填 payload、按 Execute、看回應，而且長得像技術文件而不是玩具。
`src/serving/app.py` 的欄位刻意逐一寫出而非動態生成，就是為了讓 `/docs` 讀得懂。

⚠️ **`msno` 介面在部署環境是關閉的。** 它讀的 cohort 快取（3.0 GB）是 KKBox 競賽
資料的衍生特徵，放上公開網站會與 `MODEL_CARD.md` 的授權聲明矛盾。Demo 只提供
`features` 介面，範例用合成資料。
