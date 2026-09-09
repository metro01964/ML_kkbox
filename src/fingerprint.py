"""快取的程式版本指紋 —— §7.5 遺留的待辦。

## 它補的洞

`build_cohort()` 與 `build_log_features()` 的快取命中路徑本來只驗四件事：
欄位是不是 `EXPECTED_COLUMNS` 的超集、紅線 1、cohort 有沒有錯置、列順序可不
可重現。**四條全過，不代表這份快取是現在這版程式算出來的。**

改一個平滑常數、改一條 filter、改 `_last_unambiguous` 的規則 —— 欄位一個
都沒變，四條檢查全綠，而快取裡是舊邏輯的產物。分數會變，但沒有任何東西
會提醒你。§7.11 那次事故就是這個形狀：程式修好了，實驗卻跑在舊快取上，
而當時是靠人工 `force=True` 才發現的。

## 為什麼剝掉 docstring 與註解

指紋取自產生該快取的模組原始碼，但**先正規化成 AST**：

    邏輯改了（常數、條件、聚合式）→ AST 變 → 指紋變 → 自動重建
    只改註解或 docstring          → AST 不變 → 指紋不變 → 沿用快取

本專案的 docstring 佔了程式碼一大半而且經常改寫。若連它們一起雜湊，每補一
句說明就要重掃 1.7 GB 交易檔 —— **那種守門會在第三天被人關掉**，而一個被
關掉的守門比沒有守門更糟，因為大家以為它還在。

註解根本不進 AST，所以自動免疫；docstring 是 AST 節點，要明確剝除。

## 為什麼存在 parquet 的 metadata 裡

存成欄位會混進特徵矩陣；存成旁邊的 JSON 檔會在複製檔案時失散。parquet 的
key-value metadata 跟著檔案走，而且不影響讀出來的 DataFrame。
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import ModuleType

import polars as pl

# parquet metadata 的鍵名。加前綴避免與其他工具的鍵撞名。
FINGERPRINT_KEY = "kkbox_logic_fingerprint"

# 指紋長度。16 個十六進位字元（64 bit）—— 碰撞機率遠低於「有人手動改快取」
# 這種真實風險，而且短到可以直接印在 log 裡。
FINGERPRINT_CHARS = 16


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """把模組／類別／函式的 docstring 節點拿掉。"""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    return tree


def normalize_source(source: str) -> str:
    """原始碼 → 與註解／docstring 無關的正規表示。

    Raises:
        SyntaxError: 原始碼無法解析。讓它往上拋而不是回傳空字串 ——
            一個「解析失敗就給固定指紋」的實作會讓所有壞掉的版本共用
            同一個指紋，那比沒有指紋更危險。
    """
    tree = _strip_docstrings(ast.parse(source))
    return ast.dump(tree, annotate_fields=False, include_attributes=False)


def logic_fingerprint(*sources: ModuleType | str) -> str:
    """一組模組（或原始碼字串）的邏輯指紋。

    Args:
        *sources: 模組物件或原始碼字串。**順序有意義** —— 不同的組合就是
            不同的指紋，避免「多算了一個模組」與「少算了一個」撞在一起。

    Returns:
        16 個十六進位字元。
    """
    import inspect

    h = hashlib.sha256()
    for src in sources:
        text = src if isinstance(src, str) else inspect.getsource(src)
        h.update(normalize_source(text).encode())
        h.update(b"\x1e")  # 分隔符，讓 (A,B) 與 (AB) 不會雜湊成同一個
    return h.hexdigest()[:FINGERPRINT_CHARS]


def read_cache_fingerprint(path: Path) -> str | None:
    """讀一份 parquet 的指紋。沒有指紋（或檔案不存在）回 None。"""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pl.read_parquet_metadata(path).get(FINGERPRINT_KEY)
    except Exception:  # pragma: no cover - 壞檔一律當作沒有指紋
        return None


def write_with_fingerprint(df: pl.DataFrame, path: Path, fingerprint: str) -> None:
    """寫 parquet，並把指紋放進 key-value metadata。"""
    df.write_parquet(Path(path), metadata={FINGERPRINT_KEY: fingerprint})


def cache_is_current(path: Path, fingerprint: str) -> bool:
    """這份快取是不是現在這版程式算出來的。

    **「沒有指紋」與「指紋不符」一視同仁。** 修正之前產生的快取沒有指紋，
    我們無從得知它是哪一版算的 —— 那跟明確知道它是舊版一樣不能用。
    代價是升級後第一次執行必然重建一次，那是一次性的。
    """
    return read_cache_fingerprint(path) == fingerprint
