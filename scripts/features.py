"""建立 user_logs 收聽行為聚合特徵（M2）。

    uv run python scripts/features.py
    uv run python scripts/features.py --force    # 忽略快取重算
    make features

第一次執行會掃描 31.9 GB 的原始日誌並產生收斂後的 parquet（約 30 秒），
之後只讀那份 2.2 GiB 的中間檔。可重複執行，中斷後重跑會沿用已完成的部分。
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_paths
from src.data import COHORTS
from src.features import build_log_features, narrow_logs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="忽略快取，全部重算")
    args = ap.parse_args()

    try:
        paths = load_paths()
    except FileNotFoundError as e:
        sys.exit(str(e))

    try:
        narrow_logs(paths, force=args.force)
        for name in COHORTS:
            feats = build_log_features(name, paths, force=args.force)
            cols = [c for c in feats.columns if c != "msno"]
            print(f"  {name}: {feats.height:,} 位用戶 × {len(cols)} 個收聽特徵\n")
    except FileNotFoundError as e:
        sys.exit(str(e))

    print(f"完成。中間檔位於 {paths.interim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
