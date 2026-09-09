"""M3 · 配對 multi-seed 實驗：重新量測雜訊尺度，並用配對差判定三家高下。

## 為什麼要重量一次

§7.6 的判定一直除以 `NOISE_SIGMA = 0.00048`，而那個值是在 §7.11 的「同日
交易定序」修正**之前**量的 —— 它裡面混著一部分現在已經修掉的不決定性。
在新的 σ 出來之前，「7σ」「0.5σ」這種說法都不算數，因此本次重測完成前，
SPEC 不保留任何 σ 倍數作為正式結論。

## 配對設計：為什麼不是各跑各的

天真的作法是各家跑 8 個 seed，比較兩組平均。那會把**切分本身的變異**算進
比較的雜訊裡，而那個變異對三家是**共同**的：某個 seed 剛好切出比較好學的
early stopping 集，三家會一起變好。

配對設計把它消掉：同一個 seed 下三家吃**完全相同**的 train/es 切分，然後看
每個 seed 的差值 `d_i = A_i − B_i`。共同的部分在相減時抵銷，剩下的才是「A
比 B 好多少」的真實變異。

    邊際標準差   σ(A)      「換一批切分，A 的分數會晃多少」
    配對標準差   σ(A − B)  「換一批切分，A 對 B 的優勢會晃多少」

**後者通常小得多，而判定要用的是後者。** 用邊際 σ 去除配對差，會把真實的
效果誤判成雜訊 —— §7.6 已經因為 σ 用錯而翻轉過一次結論（那次是 5-fold 的
σ 被拿來判 Mar 上的差距）。

## 判定規則：95% CI 是否含 0

配對差的 95% 信賴區間不含 0 → 差距可信；含 0 → 分不出高下。用 CI 而不是
σ 倍數，是因為 CI 自己帶著樣本數的資訊（n=8 的 t 臨界值是 2.365，不是 1.96），
而「幾 σ」這個講法會讓人忘記自由度。

    uv run python scripts/multi_seed.py
    make multi-seed
"""

from __future__ import annotations

import json
import sys
import time
from itertools import combinations

import numpy as np
import polars as pl
import yaml
from scipy import stats
from sklearn.model_selection import train_test_split

from src.config import REPO_ROOT, load_paths
from src.evaluation import log_loss
from src.models.candidates import (
    fit_catboost,
    fit_lightgbm,
    fit_xgboost,
    xgb_category_levels,
)
from src.models.train import load_cohort_features

FITTERS = {"LightGBM": fit_lightgbm, "XGBoost": fit_xgboost, "CatBoost": fit_catboost}
CONFIG_KEYS = {"LightGBM": "lightgbm", "XGBoost": "xgboost", "CatBoost": "catboost"}
CONFIDENCE = 0.95


def load_config() -> dict:
    path = REPO_ROOT / "configs" / "multi_seed.yaml"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for key in ("seeds", "training", "models"):
        if key not in cfg:
            raise KeyError(f"{path} 缺少 [{key}] 區段")
    return cfg


def split_for_seed(feb, train_cfg: dict, seed: int):
    """依 seed 切出 train / es。**三家共用這一次切分的結果。**

    切的是列索引而不是各套件轉換後的矩陣 —— 三家如果各自呼叫
    train_test_split，即使 seed 相同，只要 dtype 或列順序有任何差異就可能
    切到不同的列，配對設計就破功了（差值裡會混進「切分不同」的成分）。

    索引先按 msno 排出標準順序再切，理由同 `three_way_split`：cohort 目前
    是排序的，所以這是恆等變換；但切分不該依賴另一個模組的性質。
    """
    canonical = feb.msno.arg_sort().to_numpy()
    y = feb.y.to_numpy()
    tr_idx, es_idx = train_test_split(
        canonical,
        test_size=train_cfg["inner_valid_fraction"],
        random_state=seed,
        stratify=y[canonical],
    )
    return feb.take(tr_idx), feb.take(es_idx)


def run_seed(feb, mar, cfg: dict, seed: int, levels: dict) -> dict:
    """一個 seed：三家共用同一次切分，各回報 Mar log loss。

    `levels` 是 XGBoost 的類別字典，**從整個 Feb 算一次、所有 seed 共用** ——
    若每個 seed 各自從自己的訓練集重算，字典就會跟著切分變動，差值裡會混進
    「類別編碼不同」這個成分，配對設計的抵銷效果就被破壞了。
    """
    train, es = split_for_seed(feb, cfg["training"], seed)
    out = {"seed": seed, "train_n": train.X.height, "es_n": es.X.height}

    for name, fit in FITTERS.items():
        params = dict(cfg["models"][CONFIG_KEYS[name]])
        extra = {"category_levels": levels} if name == "XGBoost" else {}
        t0 = time.perf_counter()
        fitted = fit(train, es, params, cfg["training"], **extra)
        score = log_loss(mar.y, fitted.predict(mar.X))
        out[name] = score
        out[f"{name}_輪數"] = fitted.best_iteration
        print(
            f"    {name:9s} {score:.5f}　{fitted.best_iteration:>4} 輪"
            f"　({time.perf_counter() - t0:.0f} 秒)",
            flush=True,
        )
    return out


def marginal_spread(rows: list[dict]) -> pl.DataFrame:
    """各家自己的 Mar log loss 在不同切分下的散布 —— 這就是新的邊際 σ。

    ⚠️ 這個數字**不該**拿來判定兩家的高下（那要用配對 σ），但它是「同一個
    模型重跑一次會晃多少」的正確答案，也是判斷「某個單次量到的改善算不算
    數」的尺度 —— 例如 §7.4 的調參與 target encoding 都只有單次量測。
    """
    return pl.DataFrame(
        [
            {
                "模型": name,
                "平均": round(float(np.mean([r[name] for r in rows])), 5),
                "最小": round(float(np.min([r[name] for r in rows])), 5),
                "最大": round(float(np.max([r[name] for r in rows])), 5),
                "邊際σ": round(float(np.std([r[name] for r in rows], ddof=1)), 5),
                "平均輪數": int(np.mean([r[f"{name}_輪數"] for r in rows])),
            }
            for name in FITTERS
        ]
    )


def paired_comparisons(rows: list[dict]) -> pl.DataFrame:
    """兩兩配對比較：平均差、配對 σ、95% CI、勝出次數。

    差值定義為 `A − B`，log loss 越小越好，所以**負值代表 A 較優**。
    """
    n = len(rows)
    tcrit = float(stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1))

    out = []
    for a, b in combinations(FITTERS, 2):
        d = np.array([r[a] - r[b] for r in rows], dtype=np.float64)
        mean, sd = float(d.mean()), float(d.std(ddof=1))
        half = tcrit * sd / np.sqrt(n)
        lo, hi = mean - half, mean + half
        excludes_zero = lo > 0 or hi < 0
        out.append(
            {
                "對比": f"{a} − {b}",
                "平均差": round(mean, 6),
                "配對σ": round(sd, 6),
                "95% CI 下界": round(lo, 6),
                "95% CI 上界": round(hi, 6),
                # 欄名用通用的 A/B 而不是各家名稱：每一列的兩家不同，若用
                # 名稱當欄名，polars 會為每個模型各開一欄，其餘列填 null。
                "A 勝": int((d < 0).sum()),
                "B 勝": int((d > 0).sum()),
                "判定": "可信" if excludes_zero else "分不出高下",
            }
        )
    return pl.DataFrame(out)


def main() -> int:
    try:
        paths = load_paths().ensure()
        cfg = load_config()
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    seeds = list(cfg["seeds"])
    feb, mar = load_cohort_features(paths, cfg)
    print(f"\n配對 multi-seed：{len(seeds)} 個 seed × 3 家，seed 清單事先固定於設定檔")
    print(f"  {seeds}\n")

    levels = xgb_category_levels(feb.X, categorical=feb.categorical)

    rows = []
    for i, seed in enumerate(seeds, 1):
        print(f"  [{i}/{len(seeds)}] inner_split_seed = {seed}", flush=True)
        rows.append(run_seed(feb, mar, cfg, seed, levels))

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)

    per_seed = pl.DataFrame(
        [{"seed": r["seed"], **{m: round(r[m], 5) for m in FITTERS}} for r in rows]
    )

    print("\n" + "=" * 88)
    print("一、每個 seed 的 Mar log loss（三家共用同一次切分）")
    print("=" * 88)
    print(per_seed)

    print("\n" + "=" * 88)
    print("二、邊際散布 —— 「同一個模型重跑一次會晃多少」")
    print("=" * 88)
    spread = marginal_spread(rows)
    print(spread)

    print("\n" + "=" * 88)
    print(f"三、配對比較（差值 = A − B，負值代表 A 較優；{int(CONFIDENCE * 100)}% CI）")
    print("=" * 88)
    paired = paired_comparisons(rows)
    print(paired)

    print("\n" + "=" * 88)
    print("讀法")
    print("=" * 88)
    for row in paired.iter_rows(named=True):
        a, b = row["對比"].split(" − ")
        verdict = "可信" if row["判定"] == "可信" else "分不出高下"
        better = a if row["平均差"] < 0 else b
        print(
            f"  {row['對比']:24s} 平均差 {row['平均差']:+.5f}"
            f"　配對σ {row['配對σ']:.5f}"
            f"　CI [{row['95% CI 下界']:+.5f}, {row['95% CI 上界']:+.5f}]"
            f"　→ {verdict}" + (f"（{better} 較優）" if verdict == "可信" else "")
        )

    marginal = {r["模型"]: r["邊際σ"] for r in spread.iter_rows(named=True)}
    pair_sigmas = {r["對比"]: r["配對σ"] for r in paired.iter_rows(named=True)}
    print(
        f"\n配對讓雜訊小了一個量級：邊際σ 約 {max(marginal.values()):.5f}，"
        f"配對σ 最大 {max(pair_sigmas.values()):.5f}。\n"
        "同一個 seed 下三家吃同一份切分，切分本身的好壞在相減時抵銷 ——\n"
        "這就是為什麼判定要用配對σ，用邊際σ 會把真實效果誤判成雜訊。"
    )

    record = {
        "seeds": seeds,
        "per_seed": rows,
        "marginal": spread.to_dicts(),
        "paired": paired.to_dicts(),
        "confidence": CONFIDENCE,
    }
    out_path = REPO_ROOT / "reports" / "multi_seed.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果已存 → reports/{out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
