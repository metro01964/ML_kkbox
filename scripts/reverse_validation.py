"""反向時間外驗證：把 Feb → Mar 的比較，換成 Mar → Feb 再跑一次。

## ⚠️ 這**不是**第二個部署估計

真實部署是「用過去訓練、預測未來」。Mar → Feb 是時間倒流 —— 上線時不可能
拿三月的資料去預測二月。**這個實驗唯一能回答的是穩健性**：

    M3 的結論（哪個模型好、差距多大、集成有沒有用），
    換一組 train/valid 之後還成不成立？

如果 CatBoost 在兩個方向都贏，那個結論就不是「Mar 這個月剛好合它胃口」；
如果排名翻轉，代表 M3 的比較表其實是在量單月的雜訊，不能拿來選模型。

**回報時 headline 一律是 Feb → Mar 的 0.15781**，反向的分數只放在這張穩健性
表格裡，不進成果表、不與官方排行榜比較。

## 為什麼這個實驗不構成洩漏

擔心的點是：Mar cohort 的特徵（as-of 三月到期日）已經包含了決定 **Feb 標籤**
的那些交易（二月到期後 30 天內的續約）。訓練資料的特徵裡有驗證資料的答案，
聽起來像洩漏。

實際上不是，因為**評估時模型只看得到 Feb 的特徵**（as-of 二月到期日），
那份特徵對 Feb 標籤是乾淨的。訓練階段學到的是「特徵 → Mar 標籤」的映射，
Feb 標籤全程沒有進入訓練。

真正存在的偏誤是 SPEC §4.4 已經記錄過的那一個：**90.81% 的用戶跨兩期出現**，
而交集用戶有 94.78% 兩期標籤相同。模型只要記住特徵簽章就能拿分。這個偏誤
在正向與反向**同時存在且量級相當**，所以拿來做方向間的比較是公平的 ——
但兩個方向的絕對分數都因此偏樂觀，不可外推到測試集。

## 分群標籤的語意在反向時會變

`repeat_vs_new()` 比的是「驗證 cohort 的用戶有沒有出現在訓練 cohort」。
正向時那是「重複用戶 / 新進用戶」；反向時同一個計算的意思變成
「模型訓練時看過 / 沒看過」。數字仍然可比（都是在量重疊帶來的樂觀），
但報表上要標清楚，不能沿用「新進用戶」這個詞 —— 二月的「新進」要跟一月
比才算得出來，而我們沒有一月的 cohort。

    uv run python scripts/reverse_validation.py
    make reverse
"""

from __future__ import annotations

import sys

import polars as pl

from src.config import load_paths
from src.data import FEB, MAR
from src.models.compare import (
    constant_baseline,
    load_comparison_config,
    mean_ensemble,
    run_comparison,
)

# 兩個方向的顯示名稱。key 用 ASCII，供 MLflow 與檔名使用。
DIRECTIONS = (
    ("forward", "Feb → Mar（SPEC §4.2 正向）", FEB, MAR),
    ("reverse", "Mar → Feb（穩健性探測）", MAR, FEB),
)

# 雜訊尺度：**Mar log loss 在不同 inner split seed 下的標準差**。
#
# 為什麼需要它：光看「排名有沒有變」會把兩件完全不同的事混為一談 ——
# 「差 0.1% 的兩個選項換位」與「真正的翻轉」。前者本來就該換來換去，
# 後者才是結論不成立。所以判定一律拿差距去除以這個 σ，而不是比排名。
#
# **這個值換過兩次（0.00084 → 0.00048 → 0.00044），每一次的理由都不同。**
#
# 一、0.00084 是 Feb 內部 5-fold 的標準差 —— 它量的是「Feb 內部換一批
#     fold」，而判定的對象是 **Mar 的 log loss**，兩者不是同一個量。
#
# 二、0.00048 改量在 Mar 上（§7.8 的多 seed），指標對了。但它是**邊際**
#     標準差：「同一個模型重跑一次會晃多少」。而這裡要判的是「A 比 B 好
#     多少」，那是一個**配對差** —— 同一次切分下兩家一起變好或一起變差的
#     部分會抵銷，用邊際 σ 去除配對差會高估雜訊。
#
# 三、0.00044 來自 §7.12 的配對 multi-seed（8 個 seed，三家共用同一次切分），
#     取三組配對 σ 中最大的一組（XGBoost − CatBoost 的 0.000444）當保守值。
#     同一份實驗量到的邊際 σ 是 LightGBM 0.00026 / XGBoost 0.00037 /
#     CatBoost 0.00048 —— 舊值 0.00048 剛好等於 CatBoost 的邊際 σ，
#     **數字接近純屬巧合，它量的不是同一件事。**
#
# ⚠️ **單一切分的解析度有極限。** 以這個 σ 判定，LightGBM 對 XGBoost 的
# 0.00043 差距只有 0.97σ，落在雜訊裡；但 §7.12 的 8 個配對 seed 顯示那個
# 差距是真的（7/8 勝、95% CI 不含 0）。兩者不矛盾：**單次量測分不出來的
# 東西，重複量測分得出來。** 這個門檻回答的是「單一次比較能不能下結論」，
# 不是「這個差距存不存在」。
#
# ⚠️ 它仍然低估了「換一個月」的變異（只有兩個 cohort，估不出來）。
NOISE_SIGMA = 0.00044


def run_direction(paths, cfg, train_spec, valid_spec):
    """跑一個方向的三方比較 + 集成，回傳結果清單與常數基準。"""
    results, train, valid = run_comparison(paths, cfg, train_spec=train_spec, valid_spec=valid_spec)
    rows = list(results)
    if cfg.get("ensemble", {}).get("enabled", False) and len(results) > 1:
        rows.append(mean_ensemble(results, valid, train))
    return rows, constant_baseline(train, valid)


def direction_table(rows, baseline: float) -> pl.DataFrame:
    """單一方向的結果表。

    `相對最佳` 而非 `相對 LightGBM`：這張表要看的是排名與差距，而正向的
    參照點（LightGBM）在反向不見得還是參照點。用當下最佳者當分母，兩個
    方向的「差距」才是同一個尺度的東西。
    """
    best = min(r.logloss for r in rows)
    return pl.DataFrame(
        [
            {
                "模型": r.name,
                "輪數": r.best_iteration,
                "log loss": round(r.logloss, 5),
                "相對最佳": round(r.logloss / best - 1, 5),
                "訓練時看過": round(r.segment_score("重複用戶"), 5),
                "訓練時沒看過": round(r.segment_score("新進用戶"), 5),
                "vs 常數基準": round(1 - r.logloss / baseline, 4),
                "秒": round(r.seconds),
            }
            for r in rows
        ]
    )


def stability_table(by_direction: dict[str, list]) -> pl.DataFrame:
    """兩個方向的排名對照 —— 這張表才是本腳本的產出。"""
    ranks: dict[str, dict[str, int]] = {}
    scores: dict[str, dict[str, float]] = {}
    for key, rows in by_direction.items():
        ordered = sorted(rows, key=lambda r: r.logloss)
        for i, r in enumerate(ordered, 1):
            ranks.setdefault(r.name, {})[key] = i
            scores.setdefault(r.name, {})[key] = r.logloss

    names = list(ranks)
    return pl.DataFrame(
        [
            {
                "模型": n,
                "正向 log loss": round(scores[n]["forward"], 5),
                "正向排名": ranks[n]["forward"],
                "反向 log loss": round(scores[n]["reverse"], 5),
                "反向排名": ranks[n]["reverse"],
                "排名一致": "✅" if ranks[n]["forward"] == ranks[n]["reverse"] else "❌",
            }
            for n in names
        ]
    ).sort("正向排名")


def pairwise_stability(
    by_direction: dict[str, dict[str, float]], sigma: float = NOISE_SIGMA
) -> pl.DataFrame:
    """逐對比較兩個方向的差距，並以 σ 為單位判定。

    Args:
        by_direction: {"forward": {模型: log loss}, "reverse": {...}}
        sigma:        雜訊尺度，預設 NOISE_SIGMA。

    判定規則：

        可信  兩個方向的**方向一致**，且兩邊的差距都 >= 1σ
        雜訊  兩邊的差距都 < 1σ —— 這兩個選項分不出高下，排名換位是正常的
        存疑  其餘（方向一致但有一邊不到 1σ，或方向相反）

    「方向相反且都 >= 1σ」才是真正的結論翻轉，會落在存疑並在文字中點名。
    """
    names = list(by_direction["forward"])
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            fwd = by_direction["forward"][a] - by_direction["forward"][b]
            rev = by_direction["reverse"][a] - by_direction["reverse"][b]
            same_sign = (fwd < 0) == (rev < 0)
            big = abs(fwd) >= sigma and abs(rev) >= sigma
            small = abs(fwd) < sigma and abs(rev) < sigma
            verdict = "可信" if (same_sign and big) else ("雜訊" if small else "存疑")
            rows.append(
                {
                    "對比": f"{a} vs {b}",
                    "正向差距σ": round(fwd / sigma, 2),
                    "反向差距σ": round(rev / sigma, 2),
                    "方向一致": "✅" if same_sign else "❌",
                    "判定": verdict,
                }
            )
    return pl.DataFrame(rows)


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_comparison_config()
    except FileNotFoundError as e:
        sys.exit(f"{e}\n請先執行 uv run python scripts/download.py")

    pl.Config.set_tbl_rows(20)
    pl.Config.set_tbl_width_chars(170)

    by_direction: dict[str, list] = {}
    for key, label, train_spec, valid_spec in DIRECTIONS:
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        rows, baseline = run_direction(paths, cfg, train_spec, valid_spec)
        by_direction[key] = rows
        print()
        print(direction_table(rows, baseline))
        print(f"常數基準 {baseline:.5f}")

    print("\n" + "=" * 78)
    print("穩健性：兩個方向的排名是否一致")
    print("=" * 78)
    table = stability_table(by_direction)
    print(table)

    scores = {k: {r.name: r.logloss for r in rows} for k, rows in by_direction.items()}
    pairs = pairwise_stability(scores)
    # ⚠️ 這行標題原本把 σ 寫死成 0.00084，而實際用的是 NOISE_SIGMA = 0.00048
    # ——「換掉 σ」那次修正只改了常數，忘了改旁邊的字。報表上的數字與它自稱
    # 的單位不一致，而讀報表的人沒有辦法察覺。改成直接引用常數。
    print(f"\n--- 逐對比較（差距以 Mar log loss 的多 seed 標準差 σ={NOISE_SIGMA} 為單位）---")
    print(pairs)

    spread = {
        key: max(r.logloss for r in rows) - min(r.logloss for r in rows)
        for key, rows in by_direction.items()
    }
    print(
        f"\n三家極差　正向 {spread['forward']:.5f}（{spread['forward'] / NOISE_SIGMA:.1f}σ）"
        f" · 反向 {spread['reverse']:.5f}（{spread['reverse'] / NOISE_SIGMA:.1f}σ）"
    )

    trusted = pairs.filter(pl.col("判定") == "可信")["對比"].to_list()
    noise = pairs.filter(pl.col("判定") == "雜訊")["對比"].to_list()
    flipped = pairs.filter((pl.col("方向一致") == "❌") & (pl.col("判定") != "雜訊"))[
        "對比"
    ].to_list()

    print("\n" + "=" * 78)
    print("結論")
    print("=" * 78)
    if trusted:
        print(f"✅ 兩個方向都成立、且都超過 1σ 的比較：{'、'.join(trusted)}")
    if noise:
        print(f"⚪ 差距在雜訊範圍內、分不出高下：{'、'.join(noise)}")
    if flipped:
        print(f"❌ 方向翻轉且非雜訊：{'、'.join(flipped)}")
    if not trusted:
        print("❌ 沒有任何一組比較在兩個方向都站得住 —— 三方比較表不足以支撐選模決定。")

    print(
        "\n⚠️ 反向（Mar → Feb）是時間倒流，**不是部署估計**。headline 一律用\n"
        "   正向的 Feb → Mar 分數，反向只用於本表的穩健性判斷。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
