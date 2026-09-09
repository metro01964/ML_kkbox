"""在固定基準上重跑 M1–M4，把每一步的執行條件與結果留成紀錄。

## 為什麼需要這支腳本，而不是手動跑一輪 make

§7.8 的教訓是「分數會因為看不見的原因變動」。要讓一組數字有資格當**基準**，
光是跑出來不夠，還要能回答：

    這個數字是哪個 commit 跑的？用哪個設定？哪個 seed？幾個特徵？
    跟上一次比差多少？

人工記錄這些會漏，而且漏掉的通常正是後來出問題的那一項。所以這裡把「執行」
與「記錄」綁在同一支程式裡：每一步都寫下 git SHA、指令、設定檔、seed、耗時
與完整 stdout，存到 `reports/rebaseline/`。

## 它不做什麼

**不下結論。** 這支腳本只負責跑與記錄；哪些舊結論仍然成立要人來判斷，因為
「差 0.0004 算不算變了」取決於當初那個結論是用多大的效果撐起來的。

**不重跑 M1–M3 的資料快取。** 快取由  的程式版本指紋自動
判斷是否過時；這支腳本只負責重跑實驗，不負責重建資料。

    uv run python scripts/rebaseline.py            # 全部
    uv run python scripts/rebaseline.py --quick    # 只跑幾分鐘內的
    make rebaseline
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.config import REPO_ROOT

# (名稱, 對應里程碑, 腳本, 設定檔, 約需分鐘, 是否列入 --quick, 摘要 JSON 或 None)
#
# ⚠️ **M4 有三步，不是兩步。** 這份清單一度只有校準診斷與校準器，於是
# `make rebaseline` 宣稱「重跑 M1–M4」而 §6.2 的核心交付物（期望模擬淨收益
# 曲線）根本沒動過 —— manifest 會顯示 M4 已重跑，那是假的。
STEPS: list[tuple[str, str, list[str], str, int, bool, str | None]] = [
    ("m1_m2_baseline", "M1 + M2", ["scripts/train.py"], "configs/model_lgbm.yaml", 2, True, None),
    (
        "m4_calibration_diag",
        "M4",
        ["scripts/calibration_report.py"],
        "configs/calibration.yaml",
        5,
        True,
        None,
    ),
    ("m4_calibrator", "M4", ["scripts/calibrate.py"], "configs/calibration.yaml", 5, True, None),
    (
        "m4_business",
        "M4",
        ["scripts/business_value.py"],
        "configs/business.yaml",
        6,
        True,
        "reports/business_value.json",
    ),
    ("m3_compare", "M3", ["scripts/compare.py"], "configs/model_comparison.yaml", 6, False, None),
    (
        "m3_reverse",
        "M3",
        ["scripts/reverse_validation.py"],
        "configs/model_comparison.yaml",
        12,
        False,
        None,
    ),
    ("m3_tune", "M3", ["scripts/tune.py"], "configs/tuning.yaml", 15, False, None),
    (
        "m3_select",
        "M3",
        ["scripts/select_features.py"],
        "configs/feature_selection.yaml",
        25,
        False,
        None,
    ),
]


def git_sha() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    return out.stdout.strip()


def git_dirty() -> bool:
    """工作區有沒有未提交的改動 —— 髒的工作區跑出來的數字無法回溯。"""
    out = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT, check=False
    )
    return bool(out.stdout.strip())


def run_step(
    name: str,
    milestone: str,
    cmd: list[str],
    config: str,
    outdir: Path,
    summary_json: str | None = None,
) -> dict:
    """跑一步，完整 stdout 存檔，回傳這一步的紀錄。"""
    log_path = outdir / f"{name}.log"
    full = [sys.executable, *cmd]
    print(f"\n{'=' * 78}\n{name}（{milestone}）  {' '.join(cmd)}\n{'=' * 78}", flush=True)

    # ⚠️ **編碼要兩端一起釘死，否則整份 stdout 會靜靜地變成 None。**
    #
    # 子行程的輸出接到 pipe 時，Windows 上的 Python 預設用 locale 的 cp950；
    # 父行程這邊 `text=True` 也用 cp950 解碼。兩邊看似一致，實際上腳本印出的
    # 某些字元（實測撞到 byte 0x89）在 cp950 裡不合法，讀取執行緒就整個炸掉 ——
    # 而 `subprocess.run` **不會因此失敗**：returncode 仍是 0，只是 proc.stdout
    # 變成 None。於是每一步都顯示 ok、每一份 log 都只有一個 "None"，而模型
    # 確實跑完了幾十分鐘。
    #
    # 修法是兩端都指定 utf-8：子行程用 PYTHONIOENCODING，父行程用 encoding。
    # errors="replace" 是最後一道防線 —— 寧可有幾個字變成 U+FFFD，也不要
    # 為了一個字元丟掉整份結果。
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    t0 = time.perf_counter()
    proc = subprocess.run(
        full,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        cwd=REPO_ROOT,
        check=False,
    )
    seconds = time.perf_counter() - t0

    if not proc.stdout:
        print("  ⚠️ 這一步沒有捕捉到任何 stdout —— 結果無法記錄，請檢查。", flush=True)

    log_path.write_text(
        f"$ {' '.join(full)}\n\n{proc.stdout}\n\n--- stderr ---\n{proc.stderr}",
        encoding="utf-8",
    )
    status = "ok" if proc.returncode == 0 else f"failed({proc.returncode})"
    print(f"  {status}　{seconds / 60:.1f} 分鐘　→ reports/rebaseline/{log_path.name}", flush=True)
    if proc.returncode != 0:
        print(f"  ⚠️ stderr 末段：\n{proc.stderr[-800:]}", flush=True)

    record = {
        "step": name,
        "milestone": milestone,
        "command": " ".join(cmd),
        "config": config,
        "status": status,
        "minutes": round(seconds / 60, 2),
        "log": f"reports/rebaseline/{log_path.name}",
    }

    # 腳本自己產生的機器可讀摘要（模型、假設、門檻、圖表路徑）併進 manifest ——
    # 「這一步跑出什麼」不必靠人去翻 log。
    if summary_json:
        path = REPO_ROOT / summary_json
        if path.exists():
            record["summary"] = json.loads(path.read_text(encoding="utf-8"))
        else:
            record["summary"] = None
            print(f"  ⚠️ 預期的摘要 {summary_json} 不存在", flush=True)

    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="只跑幾分鐘內完成的步驟")
    ap.add_argument("--only", nargs="*", help="只跑指定的步驟名稱")
    args = ap.parse_args()

    outdir = REPO_ROOT / "reports" / "rebaseline"
    outdir.mkdir(parents=True, exist_ok=True)

    sha, dirty = git_sha(), git_dirty()
    if dirty:
        print("⚠️ 工作區有未提交的改動 —— 這批數字無法用 SHA 回溯。建議先 commit。\n")

    steps = [s for s in STEPS if not args.quick or s[5]]
    if args.only:
        steps = [s for s in steps if s[0] in args.only]

    total = sum(s[4] for s in steps)
    print(f"git {sha[:7]}　{len(steps)} 步　預估約 {total} 分鐘\n")

    records = [run_step(n, m, c, cfg, outdir, js) for n, m, c, cfg, _, _, js in steps]

    manifest = {
        "git_sha": sha,
        "git_dirty": dirty,
        "steps": records,
        "total_minutes": round(sum(r["minutes"] for r in records), 1),
    }
    (outdir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    failed = [r["step"] for r in records if r["status"] != "ok"]
    print(f"\n{'=' * 78}")
    print(f"完成 {len(records)} 步，共 {manifest['total_minutes']} 分鐘　git {sha[:7]}")
    print("紀錄 → reports/rebaseline/manifest.json")
    if failed:
        print(f"❌ 失敗：{'、'.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
