"""M6 · 特徵矩陣 → 機率 + 原因碼。

`/predict` 與（未來的）Kaggle 推論管線共用這一層，理由與 `src.models.adopted`
相同：**同一個問題只能有一份答案**。服務端若自己接一遍 SHAP → 造句 → 門檻，
線上的原因碼與 M5 的名單就是兩套解釋，而兩邊都不會報錯。

所以這裡什麼都不新做，只是把 M5 那條路徑接起來：

    attribute()            CatBoost 原生 TreeSHAP（`Fitted.shap_values`）
    assert_local_accuracy() 加總恆等式，逐列驗
    top_contributors()     組內相加、取前 k 組
    add_reasons()          造句 + 標量測時點
    mark_display()         決定哪幾句呈現給營運，被壓下的留在稽核欄位裡

## ⚠️ 每一筆請求都驗加總恆等式

M5 的離線名單驗過（48,853 列，最大差 6.9e-15），但那證明的是**那一次執行**的
歸因與預測同源。服務是另一條路徑：模型從 `.cbm` 載回來、特徵由 payload 組成、
欄位順序由 artifact 的清單決定。這條路徑上「歸因指到錯的欄位」的失效方式完全
存在，症狀還是同一個 —— 機率正確、原因碼通順、講的是別人的事。

一列的驗證成本可以忽略（一次 predict + 一次相加），所以預設每次都驗。
`verify=False` 只留給批次推論的效能考量，而那時應該改成抽樣驗。

## `expiry_dated` 在兩種 artifact 上的意思不同

`src/explain/reasons.py` 把 `last_is_cancel` 標成「到期日訊號」，那個標註是
**相對於 T=0 的 cutoff** 定義的：在 `lead_days = 0` 的模型上，這句原因碼講的
是到期日當天發生的事，不可沿用到 T−7 版本（§7.14 量到它佔解釋強度 42.35%）。

而在 `lead_days = 7` 的 artifact 上，同一欄講的是「T−7 之前最後一筆交易」——
那在評分時點是**看得到**的，所以它不是一個「用不了」的解釋。

因此本模組**照實回報 `horizon` / `expiry_dated`，但只在 `lead_days = 0` 時多
一句警告**。把旗標當成「不可用」的同義詞會在 T−7 的部署上誤報，而誤報的旗標
會被學會忽略 —— M5 在 `git_dirty` 上踩過這個坑（見 `scripts/explain.py`）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl

from src.explain import (
    MIN_RELATIVE_SHARE,
    add_reasons,
    assert_local_accuracy,
    attribute,
    feature_group,
    mark_display,
    top_contributors,
)
from src.serving.artifact import Artifact, assert_features_match

TOP_K = 3


@dataclass(frozen=True)
class Scored:
    """一位用戶的評分結果。"""

    p_churn: float
    p_star: float
    above_threshold: bool
    expected_net: float
    # 營運呈現的句子（`displayed == True`），依貢獻遞減。
    reasons: list[dict[str, Any]] = field(default_factory=list)
    # 被呈現門檻壓下的候選 —— 不刪，帶著 suppression_reason（同 M5 的稽核表）。
    suppressed: list[dict[str, Any]] = field(default_factory=list)
    msno: str | None = None
    warnings: list[str] = field(default_factory=list)


def _reason_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": int(row["rank"]),
        "group": row["group"],
        "reason": row["reason"],
        # SHAP 的單位是 log-odds，欄名把這件事講出來 —— 叫 `impact` 會被讀成
        # 「機率增加多少」，而那個換算不存在（見 src/explain/attribution.py）。
        "group_shap_log_odds": round(float(row["group_shap"]), 4),
        "feature": row["feature"],
        "value": None if row["value"] is None else float(row["value"]),
        "horizon": row["horizon"],
        "expiry_dated": bool(row["expiry_dated"]),
        "relative_to_top": round(float(row["relative_to_top"]), 4),
    }


def score_rows(
    artifact: Artifact,
    X: pl.DataFrame,
    *,
    msno: list[str] | pl.Series | None = None,
    top_k: int = TOP_K,
    min_relative: float = MIN_RELATIVE_SHARE,
    verify: bool = True,
    with_reasons: bool = True,
    warnings: list[str] | None = None,
) -> list[Scored]:
    """算這幾列的機率與原因碼。

    Args:
        artifact: 載好的 artifact（模型 + 假設 + p*）。
        X: 特徵矩陣，欄位與順序必須與 artifact 一致（本函式會擋）。
        msno: 逐列的識別碼，只是帶回去，不參與計算。
        top_k / min_relative: 同 M5（每人最多幾句、呈現門檻）。
        verify: 是否逐列驗加總恆等式。見模組開頭。
        with_reasons: 要不要算原因碼。**False 會跳過 TreeSHAP**，那佔了這個
            函式 97.5% 的成本（50 列實測 739 ms / 758 ms，`predict` 只有 6 ms）。
            批次名單靠它做兩段式載入：先回機率把表畫出來，再補原因碼。
            關掉時沒有歸因，所以也沒有恆等式可驗 —— 那個保證由補原因碼的那一趟
            提供，不是被放棄。
        warnings: 上游（payload 組裝）已經產生的警告，會併進每一列的結果。

    Returns:
        每列一個 `Scored`。
    """
    assert_features_match(artifact, X)
    if X.height == 0:
        raise ValueError("沒有要評分的列")

    pred = np.asarray(artifact.fitted.predict(X), dtype=np.float64)

    by_row: dict[int, list[dict[str, Any]]] = {}
    if with_reasons:
        attr = attribute(artifact.fitted, X)
        if verify:
            assert_local_accuracy(artifact.fitted, X, attr)

        reasons = mark_display(
            add_reasons(top_contributors(attr, X, k=top_k, groups=feature_group), X),
            min_relative=min_relative,
        )
        for row in reasons.sort("row", "rank").iter_rows(named=True):
            by_row.setdefault(int(row["row"]), []).append(row)

    assumptions = artifact.meta["assumptions"]
    p_star = artifact.p_star
    r_save, ltv, c_offer = (
        float(assumptions["r_save"]),
        float(assumptions["ltv_saved"]),
        float(assumptions["c_offer"]),
    )
    ids = list(msno) if msno is not None else [None] * X.height
    base_warnings = list(warnings or [])

    out: list[Scored] = []
    for i in range(X.height):
        rows = by_row.get(i, [])
        shown = [_reason_dict(r) for r in rows if r["displayed"]]
        hidden = [
            {**_reason_dict(r), "suppression_reason": r["suppression_reason"]}
            for r in rows
            if not r["displayed"]
        ]
        row_warnings = list(base_warnings)
        if with_reasons and not shown:
            # 沒有任何正貢獻的組 —— 對低風險用戶很常見（`top_contributors()`
            # 依定義不把「降低風險的因素」當成投放理由）。講出來，否則一個
            # 空的 reasons 陣列讀起來像壞了。
            row_warnings.append(
                "這位用戶沒有任何推高風險的特徵組，因此沒有原因碼 —— "
                "那不是錯誤，是「找不到該打電話的理由」。"
            )
        if artifact.scores_at_expiry and any(r["expiry_dated"] for r in shown):
            row_warnings.append(
                "這個模型在到期日當天評分（cutoff_definition = expire_date），"
                "而標為 expiry_dated 的"
                "原因碼講的是到期日當天才發生的事 —— 提前寄挽回優惠時它還沒發生，"
                "不可沿用。能上線的版本是有提前量的那些 artifact（§4.3 / §7.15）。"
            )
        out.append(
            Scored(
                p_churn=float(pred[i]),
                p_star=p_star,
                above_threshold=bool(pred[i] > p_star),
                # 這一位的期望淨收益 = p × r_save × LTV − C_offer。名單的定義
                # 就是它為正（`p > p*` 與它同義，見 src/evaluation/decision.py）。
                expected_net=round(float(pred[i]) * r_save * ltv - c_offer, 1),
                reasons=shown,
                suppressed=hidden,
                msno=ids[i],
                warnings=row_warnings,
            )
        )
    return out
