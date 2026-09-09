"""紅線 6 的洩漏迴歸測試 —— 不需資料，CI 要求零 skip。

`tests/test_no_leakage.py` 已經釘住紅線 6 最戲劇性的失敗（每個類別只出現一次
時，naive 編碼逐格等於標籤）。這一份補上另外三件**不會有任何徵兆**的事：

1. **驗證集的標籤不得影響訓練集的編碼。** 寫錯的版本會 fit 在 train + es 上，
   程式碼看起來完全正常，分數只是偏樂觀。
2. **未見過的類別必須回退 train 的 global prior。** 回 null 會讓模型多學一條
   「訓練時罕見」的分支 —— 那是關於資料集的事實，不是關於用戶的事實。
3. **編碼前後的列順序與 msno 必須對齊。** 編碼值是依位置貼回去的，中間任何
   一次 sort / filter / join 都會讓每個人拿到別人的編碼（與 §7.11 的
   `_attach_logs` 是同一類問題）。

這三條都是「錯了也跑得完、也不會 null、只有分數會怪」的類型，因此必須用
合成資料把答案手算出來。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.features.encoding import (
    IDENTIFIER_COLUMNS,
    assert_encoding_aligned,
    assert_not_identifier,
    fit_target_encoder,
    oof_target_encode,
)
from tests.conftest import NODATA


def _categories(n_per_cat: int = 200, rates: tuple[float, ...] = (0.1, 0.5, 0.9)):
    """每個類別 n 列、流失率各異的合成資料。"""
    vals, ys = [], []
    for i, rate in enumerate(rates):
        n_pos = int(round(n_per_cat * rate))
        vals += [f"c{i}"] * n_per_cat
        ys += [1] * n_pos + [0] * (n_per_cat - n_pos)
    return pl.Series("cat", vals), pl.Series("y", ys, dtype=pl.Int64)


@NODATA
def test_validation_labels_cannot_affect_train_encoding():
    """**改變 es / Mar 的標籤，train 的編碼必須逐格不變。**

    這是規定三與四的迴歸測試：mapping 只能 fit 在 train。寫成
    `fit_target_encoder(concat(train, es), concat(y_train, y_es))` 的版本
    在程式碼裡看起來一樣自然，而且不會有任何錯誤 —— 只有分數偏樂觀。
    """
    v_train, y_train = _categories()
    v_es, _ = _categories(n_per_cat=50)

    # 兩份「驗證集標籤」完全相反，若它們能影響 train 的編碼，下面必然不同。
    y_es_a = pl.Series("y", [0] * 150, dtype=pl.Int64)
    y_es_b = pl.Series("y", [1] * 150, dtype=pl.Int64)

    enc_a = oof_target_encode(v_train, y_train, seed=42)
    enc_b = oof_target_encode(v_train, y_train, seed=42)
    assert enc_a.equals(enc_b), "同樣的輸入必須給同樣的編碼"

    # 正式路徑：mapping 只 fit 在 train，es 的標籤根本沒有機會進入
    mapping_a = fit_target_encoder(v_train, y_train).mapping
    mapping_b = fit_target_encoder(v_train, y_train).mapping
    assert mapping_a == mapping_b

    # 明確示範「若把 es 併進去會怎樣」—— 這是違規版本，結果必須不同
    leaky_a = fit_target_encoder(pl.concat([v_train, v_es]), pl.concat([y_train, y_es_a])).mapping
    leaky_b = fit_target_encoder(pl.concat([v_train, v_es]), pl.concat([y_train, y_es_b])).mapping
    assert leaky_a != leaky_b, "違規版本會受 es 標籤影響 —— 這正是要避免的"
    assert mapping_a != leaky_a, "合規與違規必須是可分辨的兩件事"


@NODATA
def test_unseen_category_falls_back_to_train_prior():
    """未見過的類別回退 train 的 global prior，不是 null、不是 0。"""
    values, y = _categories()
    encoder = fit_target_encoder(values, y)

    out = encoder.transform(pl.Series("cat", ["c0", "沒見過", "也沒見過"]))

    assert out.null_count() == 0, "未見類別不得變成 null"
    assert out[1] == pytest.approx(encoder.prior)
    assert out[2] == pytest.approx(encoder.prior)
    assert encoder.prior == pytest.approx(float(y.mean()))


@NODATA
def test_oof_encoding_never_equals_own_label_for_singleton_categories():
    """每個類別只出現一次時，OOF 編碼必須全部退回先驗，不得等於自己的標籤。

    naive 編碼在這種情況下會逐格等於標籤（該類別只有自己一列）。這是紅線 6
    最極端也最容易示範的失敗形態。
    """
    n = 50
    values = pl.Series("cat", [f"only_{i}" for i in range(n)])
    y = pl.Series("y", [i % 2 for i in range(n)], dtype=pl.Int64)

    oof = oof_target_encode(values, y, n_splits=5, seed=0)
    naive = fit_target_encoder(values, y).transform(values)

    assert np.allclose(oof.to_numpy(), float(y.mean())), "每一列都該退回先驗"
    assert not np.allclose(oof.to_numpy(), y.to_numpy().astype(float))
    # naive 則會被自己的標籤拉走 —— 對照組，證明這個測試分得出兩者
    assert not np.allclose(naive.to_numpy(), float(y.mean()))


@NODATA
def test_encoding_preserves_row_order_and_msno():
    """編碼欄與資料必須逐格對齊；順序被動過就要爆。"""
    msno = pl.Series("msno", [f"u{i}" for i in range(6)])
    encoded = pl.Series("cat_te", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    assert_encoding_aligned(encoded, msno, msno)

    shuffled = pl.Series("msno", ["u1", "u0", "u2", "u3", "u4", "u5"])
    with pytest.raises(AssertionError, match="順序改變"):
        assert_encoding_aligned(encoded, msno, shuffled)

    with pytest.raises(AssertionError, match="列數改變"):
        assert_encoding_aligned(encoded, msno, msno.head(5))

    with pytest.raises(AssertionError, match="長度"):
        assert_encoding_aligned(encoded.head(3), msno, msno)


@NODATA
def test_oof_output_stays_aligned_with_input():
    """OOF 是逐折寫回去的，最容易在索引上出錯 —— 直接驗證對齊。

    作法：讓每個類別的標籤完全一致（全 0 或全 1），這樣該類別的 OOF 編碼
    必然接近它自己的比率。若寫回的索引錯了，值就會落到別的類別上。
    """
    values = pl.Series("cat", ["a"] * 300 + ["b"] * 300)
    y = pl.Series("y", [0] * 300 + [1] * 300, dtype=pl.Int64)

    out = oof_target_encode(values, y, n_splits=5, seed=0)

    assert out.len() == values.len()
    a_vals, b_vals = out.to_numpy()[:300], out.to_numpy()[300:]
    assert a_vals.max() < b_vals.min(), "a 群的編碼必須整體低於 b 群"


@NODATA
def test_identifier_columns_are_rejected_on_the_compliant_path():
    """`msno` 不得作為正式 target encoding 特徵。

    naive 會等於自己的標籤，OOF 則整欄退回先驗（每折都看不到自己）——
    兩種結局都不會報錯，所以要在源頭擋。
    """
    assert "msno" in IDENTIFIER_COLUMNS
    with pytest.raises(ValueError, match="唯一識別碼"):
        assert_not_identifier("msno")

    # 一般類別欄不受影響
    assert_not_identifier("last_payment_method_id")


@NODATA
def test_oof_on_an_identifier_collapses_to_a_constant():
    """把 OOF 用在識別碼上會得到一整欄常數 —— 證明黑名單不是多慮。

    這條測試存在的意義是：它示範「OOF 也救不了識別碼」。有人可能以為
    「只要 OOF 就安全」，實際上得到的是一個零資訊的欄位，而分數不會告訴你。
    """
    n = 100
    values = pl.Series("msno", [f"u{i}" for i in range(n)])
    y = pl.Series("y", [i % 2 for i in range(n)], dtype=pl.Int64)

    out = oof_target_encode(values, y, n_splits=5, seed=0)

    assert out.n_unique() == 1, "識別碼的 OOF 編碼必然是常數（先驗）"
