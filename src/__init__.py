"""KKBox 訂閱流失預測與挽回決策 —— 專案程式碼。

目錄對應 SPEC §7 的 repo 結構。目前只建了「現在真的有東西要放」的層：

    config.py       路徑與設定，跨層共用
    data/           資料載入與 as-of 切分（SPEC §4.3、紅線 1、2）
    evaluation/     指標與分群回報（紅線 8、SPEC §4.5）

SPEC 還列了 features/、models/、serving/，等 M2、M1、M6 要用時再建。
空目錄是雜訊，不先開。
"""

import sys as _sys

# ⚠️ **import 副作用，而且是刻意的。**
#
# `scripts/` 底下 21 個檔案都在輸出裡用 ✅ ⚠️ ❌ 標示狀態（數千處）。Python 在
# stdout 不是終端機時（管線、重導向、CI 擷取）會改用 locale 編碼 —— 在繁體中文
# Windows 上就是 cp950，而 cp950 編不出這些字元。
#
# 症狀非常誤導：`export_model.py` 已經把 artifact 寫好、round-trip 也驗證過了，
# 卻在印最後一行成功訊息時拋 UnicodeEncodeError，於是 `make artifact-t7` 回報
# exit 1。**一個成功的匯出看起來像失敗。**
#
# CI 跑 ubuntu-latest（UTF-8 預設）永遠碰不到，所以這個 bug 只會在讀者自己的
# Windows 機器上出現 —— 正好是最不該出事的地方。
#
# 放這裡而不是每個 script 各自呼叫：21 個 script 全部 `import src`，所以這一處
# 就覆蓋完；反過來要求每個新 script 記得呼叫，是遲早會漏的規則。代價是 `import
# src` 會動到行程的全域狀態，對函式庫來說不禮貌 —— 但 `src` 不是給外部 import
# 的函式庫，它的消費者只有這 21 個 script 與 FastAPI 服務。
for _stream in (_sys.stdout, _sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")
