"""下載 KKBox 競賽資料到本機。

資料絕對不進 Git（SPEC §2 · 競賽規則）。本腳本是唯一進 Git 的部分——
它記錄「怎麼拿到資料」，資料本身由每台機器各自下載一份。

前置條件：
  1. `~/.kaggle/kaggle.json`（Kaggle → Settings → API → Create New Token）
  2. 已在競賽頁點 Late Submission 並接受規則，否則 API 回 403
  3. `configs/paths.yaml` 存在（從 `configs/paths.example.yaml` 複製後改 data_root）

用法：
  uv run python scripts/download.py                    # core：M0 契約 + M1 baseline
  uv run python scripts/download.py --groups all       # 全部 8.95 GB
  uv run python scripts/download.py --groups core logs
  uv run python scripts/download.py --no-extract       # 只下載不解壓
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from src.config import load_paths

COMPETITION = "kkbox-churn-prediction-challenge"

# 官方檔案清單。壓縮大小為 2026-08-06 以 `kaggle competitions files` 實測。
# 大小用來驗證下載完整性：這是靜態的歷史資料集，數字變了就代表下載被截斷或
# 檔案被改動。容忍度 0 bytes，理由同 SPEC §2.3 對筆數的規定。
FILES: dict[str, tuple[str, int]] = {
    "train.csv.7z": ("core", 33_563_098),
    "train_v2.csv.7z": ("core", 32_818_991),
    "transactions.csv.7z": ("core", 707_508_779),
    "transactions_v2.csv.7z": ("core", 48_850_410),
    "members_v3.csv.7z": ("core", 242_308_558),
    "user_logs.csv.7z": ("logs", 7_136_060_375),
    "user_logs_v2.csv.7z": ("logs", 685_951_221),
    "sample_submission_v2.csv.7z": ("submit", 30_666_957),
    "sample_submission_zero.csv.7z": ("submit", 32_828_332),
    "WSDMChurnLabeller.scala": ("extra", 7_050),
}

GROUPS = {
    "core": "M0 資料契約 + M1 baseline（標籤、交易、用戶屬性）",
    "logs": "M2 每日收聽日誌，解壓後約 33 GB",
    "submit": "提交用的測試集用戶清單",
    "extra": "官方標籤產生器 WSDMChurnLabeller.scala（M3 選用）",
}


def human(n: float) -> str:
    """把 bytes 轉成人看得懂的單位。"""
    for unit in ("B", "KB", "MB"):
        if abs(n) < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} GB"


def find_7z(configured: str) -> Path | None:
    """找 7-Zip：設定檔優先，再試常見安裝位置，最後試 PATH。"""
    for candidate in (
        configured,
        r"C:\Program Files\7-Zip\7z.exe",
        r"C:\Program Files (x86)\7-Zip\7z.exe",
    ):
        if candidate and Path(candidate).exists():
            return Path(candidate)
    found = shutil.which("7z") or shutil.which("7za")
    return Path(found) if found else None


def kaggle_cmd() -> list[str]:
    """找 kaggle CLI。uv run 會把 .venv/Scripts 放進 PATH，但保險起見自己找一次。"""
    exe = shutil.which("kaggle")
    if exe:
        return [exe]
    name = "kaggle.exe" if os.name == "nt" else "kaggle"
    candidate = Path(sys.executable).parent / name
    if candidate.exists():
        return [str(candidate)]
    sys.exit("找不到 kaggle CLI。請先執行 uv sync。")


def download(name: str, archives: Path) -> Path:
    """下載單一檔案。已存在且大小正確就跳過，所以本腳本可重複執行。"""
    expected = FILES[name][1]
    dest = archives / name

    if dest.exists():
        actual = dest.stat().st_size
        if actual == expected:
            print(f"    已存在，跳過（{human(actual)}）")
            return dest
        print(f"    大小不符 {human(actual)} ≠ {human(expected)}，重新下載")
        dest.unlink()

    cmd = [*kaggle_cmd(), "competitions", "download", "-c", COMPETITION, "-f", name]
    subprocess.run([*cmd, "-p", str(archives)], check=True)

    # Kaggle CLI 有時會把單檔再包一層 .zip，解開後才是原始檔。
    wrapper = archives / f"{name}.zip"
    if wrapper.exists() and not dest.exists():
        shutil.unpack_archive(str(wrapper), str(archives))
        wrapper.unlink()

    if not dest.exists():
        raise FileNotFoundError(f"下載後找不到 {dest}")

    actual = dest.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{name} 大小不符：實際 {actual:,} bytes，預期 {expected:,} bytes。"
            " 下載可能被截斷，請重跑本腳本。"
        )
    print(f"    下載完成（{human(actual)}）")
    return dest


def extract(archive: Path, raw: Path, sevenzip: Path) -> None:
    """解壓到 raw/。已解壓就跳過。"""
    if archive.suffix != ".7z":
        target = raw / archive.name
        if not target.exists():
            shutil.copy2(archive, target)
            print("    複製到 raw/（非壓縮檔）")
        return

    target = raw / archive.name[: -len(".7z")]
    if target.exists():
        print(f"    已解壓，跳過（{human(target.stat().st_size)}）")
        return

    # 用 `e` 而不是 `x`：官方壓縮檔內部帶 data/churn_comp_refresh/ 這層目錄，
    # `x` 會把它一起還原，`e` 攤平成 raw/ 底下的單一檔案（SPEC §2.1 的位置）。
    subprocess.run(
        [str(sevenzip), "e", str(archive), f"-o{raw}", "-y", "-bso0", "-bsp1"],
        check=True,
    )
    if not target.exists():
        listing = subprocess.run(
            [str(sevenzip), "l", "-ba", str(archive)],
            capture_output=True,
            text=True,
        ).stdout
        raise FileNotFoundError(f"解壓後找不到 {target}。壓縮檔內容：\n{listing}")
    print(f"    解壓完成（{human(target.stat().st_size)}）")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--groups",
        nargs="+",
        default=["core"],
        choices=[*GROUPS, "all"],
        help="要下載的檔案群組（預設 core）。可用：" + " / ".join(GROUPS),
    )
    ap.add_argument("--no-extract", action="store_true", help="只下載 .7z，不解壓")
    args = ap.parse_args()

    # 路徑設定統一由 src.config 提供，不在這裡自己讀 YAML。
    try:
        paths = load_paths()
    except FileNotFoundError as e:
        sys.exit(str(e))
    archives, raw = paths.archives, paths.raw
    archives.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    groups = set(GROUPS) if "all" in args.groups else set(args.groups)
    # 由小到大排序：小檔先跑完，能及早發現憑證或權限問題。
    targets = sorted(
        (n for n, (g, _) in FILES.items() if g in groups),
        key=lambda n: FILES[n][1],
    )

    print(f"資料根目錄  {paths.data_root}")
    print(f"群組        {', '.join(sorted(groups))}")
    print(f"檔案        {len(targets)} 個，壓縮合計 {human(sum(FILES[n][1] for n in targets))}")
    print()

    sevenzip = None
    if not args.no_extract:
        sevenzip = find_7z(paths.sevenzip)
        if sevenzip is None:
            sys.exit("找不到 7-Zip。請安裝後在 configs/paths.yaml 指定，或加 --no-extract。")

    for i, name in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {name}  ({human(FILES[name][1])})")
        archive = download(name, archives)
        if sevenzip is not None:
            extract(archive, raw, sevenzip)

    print()
    print(f"完成。壓縮檔 {archives}")
    print(f"      CSV   {raw}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
