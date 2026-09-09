"""路徑與環境設定。

專案裡任何需要知道「資料在哪」的程式都從這裡拿設定，不要各自去讀 YAML。

在這個模組出現之前，scripts/download.py 和 notebooks/eda_01_overview.py
各自實作了一份載入邏輯。兩份重複的程式碼會慢慢長歪 —— 例如 download.py
支援用環境變數 DATA_ROOT 覆寫、EDA 腳本沒有，於是同一份設定在兩支程式裡
解析出不同結果。這種 bug 很難查，因為兩邊「看起來」都對。

設計上刻意讓路徑是「算出來的」而不是「設定出來的」：paths.yaml 只填一個
data_root，archives / raw / interim 都由它推導。少一個可以填錯的欄位，
就少一種出錯的方式。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

# 用 __file__ 往上找 repo 根目錄，不依賴當前工作目錄。
# src/config.py → parents[0] 是 src/，parents[1] 是 repo 根目錄。
REPO_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = REPO_ROOT / "configs" / "paths.yaml"
EXAMPLE_PATH = REPO_ROOT / "configs" / "paths.example.yaml"


@dataclass(frozen=True)
class Paths:
    """資料與輸出位置。

    frozen=True 讓它建立後不可修改。設定值被程式中途改掉是很難追的 bug ——
    你在 A 檔案讀到的 data_root 和 B 檔案讀到的不一樣，而且沒有任何錯誤訊息。
    """

    data_root: Path
    sevenzip: str = ""

    @property
    def archives(self) -> Path:
        """Kaggle 下載的 .7z 壓縮檔。"""
        return self.data_root / "archives"

    @property
    def raw(self) -> Path:
        """解壓後的原始 CSV。SPEC §2.1 所指的位置。"""
        return self.data_root / "raw"

    @property
    def interim(self) -> Path:
        """中間結果（parquet 快取）。可以隨時整個刪掉重算，不算資產。"""
        return self.data_root / "interim"

    @property
    def artifacts(self) -> Path:
        """訓練好的模型 artifact（M6 的服務載入的東西）。

        **刻意不放在 `interim/`。** 那裡的東西「可以隨時整個刪掉重算」，而
        artifact 是一件交付物：服務、HF Spaces Demo、Kaggle 推論管線載入的
        就是它，刪掉等於服務起不來。兩者的生命週期不同，就不該共用一個目錄。

        也不放在 repo 裡：模型檔是二進位、每次重訓都變，git 存不動它
        （`*.cbm` 已在 .gitignore）。要帶去別的機器（Docker / HF Spaces）就
        整個目錄複製過去，並用環境變數 `MODEL_ARTIFACT` 指路 ——
        見 `src/serving/artifact.py`。
        """
        return self.data_root / "artifacts"

    @property
    def figures(self) -> Path:
        """圖表。注意這個在 repo 裡面，會進 git（SPEC §7 的 reports/figures/）。"""
        return REPO_ROOT / "reports" / "figures"

    def ensure(self) -> Paths:
        """建立輸出目錄。

        只建 interim 與 figures 這兩個「程式會寫入」的目錄。
        raw 與 archives 不建 —— 那是 scripts/download.py 的職責，
        在這裡默默建一個空的 raw/ 只會讓「資料還沒下載」這件事更難發現。
        """
        self.interim.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def load_paths() -> Paths:
    """讀 configs/paths.yaml，回傳 Paths。

    lru_cache 讓同一個 process 內只真正讀一次檔，之後都拿快取。設定檔在
    程式執行中途不會變，重複讀只是浪費。

    環境變數 DATA_ROOT 優先於設定檔 —— CI 沒有 paths.yaml（它不進 git），
    需要一個不用寫檔案的覆寫管道。
    """
    cfg: dict = {}
    if CONFIG_PATH.exists():
        cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}

    data_root = os.environ.get("DATA_ROOT") or cfg.get("data_root")
    if not data_root:
        raise FileNotFoundError(
            f"找不到資料路徑設定。請執行：\n"
            f"    copy configs\\paths.example.yaml configs\\paths.yaml\n"
            f"然後把 data_root 改成這台機器的實際路徑。\n"
            f"（或設定環境變數 DATA_ROOT。範本見 {EXAMPLE_PATH}）"
        )

    return Paths(data_root=Path(data_root), sevenzip=cfg.get("sevenzip") or "")
