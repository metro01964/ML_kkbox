"""替線上 Demo 算一份營運視角的資料。

## 為什麼不直接用 reports/business_value.json

那份是 **lead0d（T=0）** 模型的：`p_star` 0.4868、`months_source` 為 feb 的流失率。
而部署上線的是 **lead7d（T−7）** —— `p_star` 0.4836，分數也不同（0.17921 vs 0.15367）。
拿另一個模型的營運數字裝在這個模型的 Demo 上，是最容易被抓到的那種不一致。

所以這裡直接載**部署中的那份 artifact**，用它自己的假設重算。

## 輸出的東西可以公開嗎

可以。寫出去的是**聚合統計**：一條 K% → 期望淨收益的曲線（約 500 點）、最佳投放
比例、名單人數、precision 與 lift。沒有任何一列用戶資料，也回推不出個人 ——
與 README 已經公開的分群流失率是同一類東西。

原始資料與 cohort 快取仍然不進映像檔（見 deploy/serving.yaml）。

用法：
    uv run python scripts/demo_curve.py
"""

from __future__ import annotations

import json

import numpy as np
import yaml

from src.config import REPO_ROOT, load_paths
from src.data import COHORTS, build_cohort
from src.evaluation import campaign_curve, optimal_point
from src.features import build_features, build_log_features
from src.serving.artifact import load_artifact

OUT_PATH = REPO_ROOT / "deploy" / "demo_curve.json"
SERVING_CONFIG = REPO_ROOT / "configs" / "serving.yaml"


def main() -> int:
    paths = load_paths()
    name = yaml.safe_load(SERVING_CONFIG.read_text(encoding="utf-8"))["artifact"]
    art = load_artifact(name=name)

    # artifact 自己記著它是拿哪個 cohort 評估的 —— 不要在這裡另外指定，
    # 那會變成兩份真相。
    eval_name = art.meta["cohort"]["eval"]
    if eval_name not in COHORTS:
        print(f"artifact 的評估 cohort {eval_name!r} 不在 COHORTS 裡，無法重算")
        return 1
    spec = COHORTS[eval_name]

    print(f"artifact {art.directory.name}（{art.cutoff_definition}）→ cohort {eval_name}")

    raw = build_cohort(spec, paths)
    logs = build_log_features(spec, paths)
    fs = build_features(raw, logs)
    pred = np.asarray(art.fitted.predict(fs.X), dtype=np.float64)
    y = np.asarray(fs.y, dtype=np.float64)

    a = art.meta["assumptions"]
    r_save, c_offer, ltv = a["r_save"], a["c_offer"], a["ltv_saved"]
    p_star = a["p_star"]

    curve = campaign_curve(y, pred, r_save=r_save, ltv_saved=ltv, c_offer=c_offer, step=0.002)
    best = optimal_point(curve, by="期望模擬淨收益")

    # 依 p* 這條固定門檻投放的話會是什麼結果 —— 那才是服務實際在做的事，
    # 曲線極大值只是「事後知道最佳點在哪」的對照。
    mask = pred > p_star
    n_at_p_star = int(mask.sum())
    hit_at_p_star = float(y[mask].mean()) if n_at_p_star else 0.0
    net_at_p_star = float((pred[mask] * r_save * ltv - c_offer).sum())

    k_col, net_col = "K", "期望模擬淨收益"
    payload = {
        "artifact": art.directory.name,
        "cutoff_definition": art.cutoff_definition,
        "cohort": eval_name,
        "n_users": int(len(y)),
        "churn_rate": round(float(y.mean()), 6),
        "log_loss": art.meta["metrics"]["log_loss"],
        "assumptions": {
            "r_save": r_save,
            "c_offer": c_offer,
            "ltv_saved": ltv,
            "p_star": p_star,
            "months_source": a.get("months_source"),
        },
        "at_p_star": {
            "n_targeted": n_at_p_star,
            "share": round(n_at_p_star / len(y), 4),
            "precision": round(hit_at_p_star, 4),
            "expected_net": round(net_at_p_star, 0),
        },
        "optimum": {
            "k": float(best[k_col]),
            "n_targeted": int(round(float(best[k_col]) * len(y))),
            "expected_net": round(float(best[net_col]), 0),
        },
        # 曲線只留畫圖需要的兩欄，並降到每 1% 一點 —— Demo 的折線圖用不到 0.2%
        # 的解析度，而檔案要進映像檔。
        "curve": [
            {"k": round(float(r[k_col]), 4), "net": round(float(r[net_col]), 0)}
            for r in curve.iter_rows(named=True)
            if abs(round(float(r[k_col]) * 100) - float(r[k_col]) * 100) < 1e-9
        ],
    }

    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  cohort {payload['n_users']:,} 人，流失率 {payload['churn_rate']:.4%}")
    print(
        f"  依 p* = {p_star} 投放：{n_at_p_star:,} 人"
        f"（{payload['at_p_star']['share']:.2%}）、命中率 {hit_at_p_star:.2%}、"
        f"期望淨收益 {net_at_p_star:,.0f} 元"
    )
    print(
        f"  曲線極大值在前 {payload['optimum']['k']:.1%}"
        f"（{payload['optimum']['n_targeted']:,} 人，{payload['optimum']['expected_net']:,.0f} 元）"
    )
    print(f"  寫出 {OUT_PATH.relative_to(REPO_ROOT)}（{len(payload['curve'])} 個曲線點）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
