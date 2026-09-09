"""驗證「重建快取不會改變任何人的位置」—— 固定基準的前提。

§7.8 記錄過一個 bug：`build_cohort` 的輸出列順序每次重建都不同（polars 的
`group_by` 不保證順序），而下游的 `train_test_split` 依位置切分 —— 順序一變
就換一批訓練資料，M2 的 Mar log loss 在 0.15853 ~ 0.15910 之間跳動。修正是
在 `build_cohort` 末尾加 `.sort("msno")`。

**修正之後沒有人真的驗證過它。** 這支腳本補上：連續強制重建 N 次，每一輪
都對「誰在哪裡」取指紋並互相比對。

## 取哪些指紋

    cohort msno 順序      build_cohort 的直接產物，bug 的原點
    收聽特徵 msno 順序    另一個 group_by 的產物（join 用，順序理論上不影響）
    特徵矩陣 msno 順序    真正餵進模型的那一份
    train / es / sel      三段切分各自的成員名單（排序後取指紋，與順序無關）
    M1/M2 內部切分        train_baseline 的 early stopping 切分

前三個比對的是**順序**，後兩個比對的是**成員**。兩者要分開：成員一樣但順序
不同仍然會改變 LightGBM 的浮點加總次序；順序一樣但成員不同則是徹底換了資料。

## 為什麼 narrow_logs 不重建

`user_logs_window_*.parquet` 是從 30.5 GB 原始日誌收斂出來的中間檔，重建要
掃完整份 CSV。它以 msno + date 為鍵被 join，內部順序不會傳到特徵矩陣 ——
而那正是本腳本第三個指紋要驗證的事。要連它一起重建請自行加 --with-narrow。

    uv run python scripts/verify_rebuild.py --rounds 2
    make verify-rebuild
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from typing import Any

import polars as pl

from src.config import REPO_ROOT, load_paths
from src.data import COHORTS, build_cohort
from src.features import build_features, build_log_features, narrow_logs
from src.models.train import load_model_config
from src.models.tuning import three_way_split


def fingerprint(values) -> str:
    """一串 msno 的指紋。順序敏感 —— 這正是我們要比對的東西。"""
    s = pl.Series(values)
    h = hashlib.sha256()
    for v in s.to_list():
        h.update(str(v).encode())
        h.update(b"\x00")
    return h.hexdigest()[:16]


def members_fingerprint(values) -> str:
    """一群 msno 的**成員**指紋（先排序，所以與順序無關）。"""
    return fingerprint(sorted(pl.Series(values).to_list()))


def content_fingerprint(df: pl.DataFrame) -> str:
    """整張表的**逐位元**指紋 —— 值變了就會變，順序變了也會變。

    ⚠️ 這個指紋是後來才加的，而它的缺席讓第一版驗證得出了錯誤的結論。

    第一版只比 msno 的順序，三輪全綠，看起來「重建不改變任何東西」。實際上
    `last_actual_amount_paid` 有 24 列在重建之間變了值（最大差 1608）、
    `last_is_cancel` 有 19 人翻面 —— 順序指紋對這些完全無感，因為每個人
    還在原來的位置上，只是身上的數字換了。

    `hash_rows` 走的是值的位元表示，所以連 1e-9 的浮點差異都抓得到（收聽
    特徵的 `log*_secs` 確實有這種差異，來自平行加總的次序）。欄位先排序，
    避免欄序變動造成假警報。
    """
    ordered = df.select(sorted(df.columns))
    return hashlib.sha256(ordered.hash_rows(seed=0).to_numpy().tobytes()).hexdigest()[:16]


def load_split_cfgs() -> dict[str, dict]:
    """所有會切分資料的設定檔，各取一份 split 區段。"""
    import yaml

    out = {}
    for name in ("tuning", "calibration", "feature_selection"):
        path = REPO_ROOT / "configs" / f"{name}.yaml"
        if not path.exists():
            continue
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if "split" in cfg:
            out[name] = cfg["split"]
    return out


def snapshot(paths, cfg, *, force: bool, label: str, with_narrow: bool = False) -> dict[str, Any]:
    """跑一輪：（可選）強制重建，然後對每一層取指紋。"""
    print(f"\n{'=' * 78}\n{label}（force={force}）\n{'=' * 78}", flush=True)

    if force and with_narrow:
        print("  重建 narrow_logs（掃 30.5 GB 原始日誌）...", flush=True)
        narrow_logs(paths, force=True)

    out: dict[str, Any] = {"label": label, "force": force, "cohorts": {}}

    feature_sets = {}
    for name in COHORTS:
        print(f"  {name}: 重建 cohort..." if force else f"  {name}: 讀取快取...", flush=True)
        raw = build_cohort(name, paths, force=force, verbose=False)
        logs = build_log_features(name, paths, force=force, verbose=False)
        fs = build_features(raw, logs)
        feature_sets[name] = fs

        out["cohorts"][name] = {
            "列數": raw.height,
            "特徵數": fs.X.width,
            "流失率": round(float(fs.y.mean()), 6),
            "cohort_msno_順序": fingerprint(raw["msno"]),
            "收聽特徵_msno_順序": fingerprint(logs["msno"]),
            "特徵矩陣_msno_順序": fingerprint(fs.msno),
            # 內容指紋才抓得到「人沒動、值變了」—— 見 content_fingerprint。
            "cohort_內容": content_fingerprint(raw),
            "收聽特徵_內容": content_fingerprint(logs),
            "特徵矩陣_內容": content_fingerprint(fs.X),
            "cohort_已排序": bool(raw["msno"].is_sorted()),
            "同日多筆人數": int(raw["last_day_n_tx"].gt(1).sum()),
            "有衝突人數": int(raw["last_day_has_conflict"].sum()),
        }
        conflict = out["cohorts"][name]
        print(
            f"    {raw.height:,} 列 × {fs.X.width} 特徵　"
            f"同日多筆 {conflict['同日多筆人數']:,} · 有衝突 {conflict['有衝突人數']:,}",
            flush=True,
        )

    # ---- 切分：成員名單必須固定 ----
    feb = feature_sets["feb"]
    out["splits"] = {}
    for cfg_name, split_cfg in load_split_cfgs().items():
        split = three_way_split(feb, split_cfg)
        out["splits"][cfg_name] = {
            "seed": split_cfg["split_seed"],
            "train": {
                "n": split.train.X.height,
                "成員": members_fingerprint(split.train.msno),
                "順序": fingerprint(split.train.msno),
            },
            "es": {
                "n": split.es.X.height,
                "成員": members_fingerprint(split.es.msno),
                "順序": fingerprint(split.es.msno),
            },
            "sel": {
                "n": split.sel.X.height,
                "成員": members_fingerprint(split.sel.msno),
                "順序": fingerprint(split.sel.msno),
            },
        }
        print(
            f"  切分 {cfg_name}（seed {split_cfg['split_seed']}）："
            f"{split.train.X.height:,} / {split.es.X.height:,} / {split.sel.X.height:,}",
            flush=True,
        )

    # ---- M1/M2 的內部切分（train_baseline 用的那一個）----
    from sklearn.model_selection import train_test_split

    train_cfg = cfg["training"]
    canonical = feb.msno.arg_sort().to_numpy()
    tr_idx, es_idx = train_test_split(
        canonical,
        test_size=train_cfg["inner_valid_fraction"],
        random_state=train_cfg["inner_split_seed"],
        stratify=feb.y.to_numpy()[canonical],
    )
    out["m1_inner_split"] = {
        "seed": train_cfg["inner_split_seed"],
        "train": {"n": len(tr_idx), "成員": members_fingerprint(feb.msno[tr_idx])},
        "es": {"n": len(es_idx), "成員": members_fingerprint(feb.msno[es_idx])},
    }
    print(f"  M1/M2 內部切分：{len(tr_idx):,} / {len(es_idx):,}", flush=True)
    return out


def compare(rounds: list[dict]) -> bool:
    """逐項比對每一輪，回傳是否全部一致。"""
    base, ok = rounds[0], True
    print(f"\n{'=' * 78}\n比對（基準 = {base['label']}）\n{'=' * 78}")

    rows = []
    for other in rounds[1:]:
        for name in base["cohorts"]:
            for key in (
                "cohort_msno_順序",
                "收聽特徵_msno_順序",
                "特徵矩陣_msno_順序",
                "cohort_內容",
                "收聽特徵_內容",
                "特徵矩陣_內容",
            ):
                a, b = base["cohorts"][name][key], other["cohorts"][name][key]
                rows.append(
                    {"輪次": other["label"], "項目": f"{name}.{key}", "一致": a == b, "指紋": b}
                )
                ok &= a == b
        for cfg_name in base["splits"]:
            for part in ("train", "es", "sel"):
                a = base["splits"][cfg_name][part]["成員"]
                b = other["splits"][cfg_name][part]["成員"]
                rows.append(
                    {
                        "輪次": other["label"],
                        "項目": f"split[{cfg_name}].{part}.成員",
                        "一致": a == b,
                        "指紋": b,
                    }
                )
                ok &= a == b
        for part in ("train", "es"):
            a = base["m1_inner_split"][part]["成員"]
            b = other["m1_inner_split"][part]["成員"]
            rows.append(
                {"輪次": other["label"], "項目": f"m1_inner.{part}.成員", "一致": a == b, "指紋": b}
            )
            ok &= a == b

    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_width_chars(140)
    print(pl.DataFrame(rows))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=2, help="強制重建幾輪（預設 2）")
    ap.add_argument("--with-narrow", action="store_true", help="連 narrow_logs 一起重建（很慢）")
    args = ap.parse_args()

    try:
        paths = load_paths().ensure()
        cfg = load_model_config()
    except FileNotFoundError as e:
        sys.exit(str(e))

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    ).stdout.strip()

    # 第 0 輪讀現有快取 —— 這一輪的意義是「重建之後還跟重建之前一樣嗎」，
    # 也就是既有的所有分數還算不算數。
    rounds = [snapshot(paths, cfg, force=False, label="現有快取")]
    for i in range(1, args.rounds + 1):
        rounds.append(
            snapshot(paths, cfg, force=True, label=f"強制重建 #{i}", with_narrow=args.with_narrow)
        )

    ok = compare(rounds)

    record = {"git_sha": sha, "rounds": rounds, "all_identical": ok}
    out_path = REPO_ROOT / "reports" / "rebuild_fingerprints.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n指紋已存 → reports/{out_path.name}（git {sha}）")
    print("\n" + ("✅ 全部一致 —— 重建不改變任何人的位置" if ok else "❌ 有項目不一致，見上表"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
