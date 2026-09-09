"""訓練 M1 LightGBM baseline 的入口。

實作在 src/models/train.py，這裡只負責被當成主程式執行。

分開的理由：`python -m src.models.train` 會讓該模組被載入兩次（一次是
src/models/__init__.py 匯入它，一次是當主程式執行），Python 會發出
RuntimeWarning 並且模組層級的狀態會有兩份。放在 scripts/ 底下就沒有這個
問題，也與 scripts/download.py 的慣例一致。

    uv run python scripts/train.py
    make train
"""

import sys

from src.models.train import main

if __name__ == "__main__":
    sys.exit(main())
