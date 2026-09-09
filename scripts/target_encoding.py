"""M3 · Target encoding 的**洩漏教學／對照實驗**（SPEC §5 紅線 6）。

> ⚠️ **這支腳本不參與選模。** 最終模型由 §7.12 的配對 multi-seed 決定
> （CatBoost，Mar 0.15367，8/8 勝）。**任何違規變體的分數再漂亮，都不得
> 改變模型選擇、不得寫入模型 artifact、不得進入 headline。**

紅線 6 說「target encoding 必須 out-of-fold，否則 CV 飆高、實測崩盤」。
這支腳本把那句話變成五個可比較的數字：

    A 原生類別（LightGBM 直接處理 last_payment_method_id）   ✅ 合規 · 對照
    B OOF TE · last_payment_method_id                        ✅ 合規
    C Naive TE · last_payment_method_id                      ⛔ 違規
    D Naive TE · msno，切分**之後**才編碼                     ⛔ 違規
    E Naive TE · msno，切分**之前**就編碼                     ⛔ 違規

**正式結論只比較 A 與 B。** C / D / E 是刻意寫錯的控制組，用來量出「違規在
真實資料上的代價」—— 單元測試證明守門擋得住合成資料，這裡證明違規在什麼
條件下會被分數抓到、什麼條件下不會。

## 合規路徑的五條規定（本腳本逐條落實）

    一、先切 train / es / sel，再做任何編碼
    二、train 用 5-fold OOF —— 每列的編碼只用其他折的標籤
    三、es / sel / Mar 的 mapping 只 fit 在 train
    四、smoothing、global prior、類別平均全部只從 train 算
    五、未見過的類別一律回退 train 的 global prior

第三條特別容易寫錯成「各自 fit 各自的」，那在程式碼裡看起來完全正常。

## 為什麼內部分數報 Feb-sel 而不是 Feb-es

`es` 是 early stopping 的依據 —— 停在第幾輪是**看著它**決定的，拿它當內部
分數會偏樂觀。`sel` 全程沒有被任何決定看過，是這裡唯一乾淨的內部估計。
兩個都印出來，差距本身就是「看著分數做決定」的代價。

## 三個違規變體為什麼要分成三個

它們是同一個錯誤在三種條件下的三種面貌，而這正是紅線 6 難懂的地方：

**C（低基數）**：`last_payment_method_id` 只有 33 種取值，每種平均兩萬多列。
把自己那一列的標籤算進去，只會讓類別平均值動 1/24000 —— 洩漏被大樣本稀釋
到量不出來。**違規在這裡幾乎不會被分數抓到。**

**D（高基數，切分後編碼）**：`msno` 每列一個唯一值，編碼幾乎就是自己的標籤。
但 es / sel / Mar 的 msno 從沒出現在 train 的編碼表裡，只能拿到先驗 ——
模型學到的規則在驗證集完全失效。**分數會當場崩掉。**

**E（高基數，切分前編碼）**：紅線 6 描述的經典錯誤 —— 先對整個 Feb cohort
做 target encoding，之後才切驗證集。驗證集的每一列也帶著自己的標籤，於是
**內部分數好得不像話、時間外崩盤**。C 與 E 的對比說明了為什麼門檻要訂在
「作法」而不是「分數有沒有變差」。

    uv run python scripts/target_encoding.py
    make encode
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import polars as pl
import yaml

from src.config import REPO_ROOT, load_paths
from src.evaluation import log_loss
from src.features import FeatureSet
from src.features.encoding import (
    assert_encoding_aligned,
    assert_encoding_is_oof,
    assert_not_identifier,
    fit_target_encoder,
    oof_target_encode,
)
from src.models.candidates import fit_lightgbm
from src.models.train import load_cohort_features, load_model_config
from src.models.tuning import three_way_split

TARGET_COL = "last_payment_method_id"


@dataclass(frozen=True)
class Variant:
    """一個實驗變體。

    `compliant` 不是註解而是控制流程：不合規的變體會跳過 `assert_not_identifier`
    與 OOF 守門（它們本來就是要違規），同時被排除在「最佳」的挑選之外。
    """

    label: str
    column: str | None
    mode: str  # none | oof | naive | naive_presplit
    compliant: bool


VARIANTS = (
    Variant("A 原生類別（對照）", None, "none", True),
    Variant(f"B OOF TE · {TARGET_COL}", TARGET_COL, "oof", True),
    Variant(f"C Naive TE · {TARGET_COL}", TARGET_COL, "naive", False),
    Variant("D Naive TE · msno（切分後）", "msno", "naive", False),
    Variant("E Naive TE · msno（切分前）", "msno", "naive_presplit", False),
)


def load_split_config() -> dict:
    """沿用 `configs/tuning.yaml` 的三段切分。

    不另開一份設定：篩選（§7.4 第二點）與調參（第四點）都用這一組，本實驗的
    基準因此與它們可比。兩處各寫一份 seed，遲早會不同步。
    """
    path = REPO_ROOT / "configs" / "tuning.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "split" not in cfg:
        raise KeyError(f"{path} 缺少 [split] 區段")
    return cfg["split"]


def with_encoded(fs: FeatureSet, col: str, encoded: pl.Series) -> FeatureSet:
    """把某欄換成它的編碼值，並把它從類別特徵清單裡移除。

    移除這一步不能忘：編碼後的欄位是連續的 [0, 1] 機率，若仍被宣告成類別
    特徵，LightGBM 會把每個不同的浮點數當成一個獨立類別 —— 那既沒有意義
    也會爆炸性地過擬合。

    貼回去是**依位置**的，所以貼之前先驗證 msno 沒有被動過。
    """
    assert_encoding_aligned(encoded, fs.msno, fs.msno)
    X = fs.X.drop(col) if col in fs.X.columns else fs.X
    return FeatureSet(
        X=X.with_columns(encoded.alias(f"{col}_te")),
        y=fs.y,
        msno=fs.msno,
        categorical=tuple(c for c in fs.categorical if c != col),
    )


def source_column(fs: FeatureSet, col: str) -> pl.Series:
    """取要編碼的原始欄。`msno` 不在特徵矩陣裡，要從 FeatureSet 另外拿。"""
    return fs.msno if col == "msno" else fs.X[col]


def prepare(
    feb: FeatureSet,
    mar: FeatureSet,
    split,
    variant: Variant,
) -> tuple[FeatureSet, FeatureSet, FeatureSet, FeatureSet]:
    """依變體產生 (train, es, sel, Mar) 四份特徵矩陣。"""
    col, mode = variant.column, variant.mode

    if mode == "none":
        return split.train, split.es, split.sel, mar

    if variant.compliant:
        # 合規路徑：識別碼一律擋掉（見 assert_not_identifier）。
        assert_not_identifier(col)

    if mode == "naive_presplit":
        # ⛔ 違規的關鍵在**順序**：先用整個 Feb（含之後會被切成 es / sel 的
        # 那些列）擬合編碼器，再切分。驗證集的每一列因此也帶著自己的標籤。
        src_feb = source_column(feb, col)
        encoder = fit_target_encoder(src_feb, feb.y)
        feb_v = with_encoded(feb, col, encoder.transform(src_feb))
        mar_v = with_encoded(mar, col, encoder.transform(source_column(mar, col)))
        leaked = three_way_split(feb_v, load_split_config())
        return leaked.train, leaked.es, leaked.sel, mar_v

    # ---- 合規順序：先切分，編碼器只看得到 train ----
    train, es, sel = split.train, split.es, split.sel
    src_train = source_column(train, col)

    if mode == "oof":
        enc_train = oof_target_encode(src_train, train.y)
        # 守門：OOF 的編碼不得逐格等於標籤。這是紅線 6 的核心斷言，
        # 放在產線路徑上而不只是測試裡。
        assert_encoding_is_oof(enc_train, train.y)
    elif mode == "naive":
        # ⛔ 違規：用 train 自己的標籤擬合再套回 train，每列的編碼含有自己的標籤。
        enc_train = fit_target_encoder(src_train, train.y).transform(src_train)
    else:  # pragma: no cover - VARIANTS 已窮舉
        raise ValueError(f"未知的模式 {mode}")

    # **es / sel / Mar 一律用「只 fit 在 train」的編碼器。**
    # 它們的標籤一次都沒有參與計算 —— 這就是規定三與四。
    # 未見過的類別由 TargetEncoder.transform 退回 train 的 global prior（規定五）。
    encoder = fit_target_encoder(src_train, train.y)
    return (
        with_encoded(train, col, enc_train),
        with_encoded(es, col, encoder.transform(source_column(es, col))),
        with_encoded(sel, col, encoder.transform(source_column(sel, col))),
        with_encoded(mar, col, encoder.transform(source_column(mar, col))),
    )


def main() -> int:
    try:
        paths = load_paths()
        cfg = load_model_config()
        split_cfg = load_split_config()
    except (FileNotFoundError, KeyError) as e:
        sys.exit(str(e))

    params, train_cfg = dict(cfg["model"]), cfg["training"]
    feb, mar = load_cohort_features(paths, cfg)
    split = three_way_split(feb, split_cfg)

    print(
        f"  Feb 三段切分：訓練 {split.train.X.height:,}"
        f" · early stopping {split.es.X.height:,}"
        f" · sel {split.sel.X.height:,}"
    )
    print("  五個變體共用同一批列、同一組超參數、同一個 seed —— 差異只有編碼方式\n")

    rows = []
    for i, v in enumerate(VARIANTS, 1):
        flag = "✅ 合規" if v.compliant else "⛔ 違規"
        print(f"[{i}/{len(VARIANTS)}] {v.label}　{flag}", flush=True)
        train_v, es_v, sel_v, mar_v = prepare(feb, mar, split, v)

        t0 = time.perf_counter()
        fitted = fit_lightgbm(train_v, es_v, params, train_cfg)
        secs = time.perf_counter() - t0

        # 三個分數都要報。只看內部會以為模型變強，只看時間外會以為它只是
        # 沒用而已 —— 洩漏的特徵是「內部好、時間外差」的剪刀差。
        row = {
            "變體": v.label,
            "合規": flag,
            "輪數": fitted.best_iteration,
            "Feb-sel": round(log_loss(sel_v.y, fitted.predict(sel_v.X)), 5),
            "Feb-es": round(log_loss(es_v.y, fitted.predict(es_v.X)), 5),
            "Mar 時間外": round(log_loss(mar_v.y, fitted.predict(mar_v.X)), 5),
            "秒": round(secs),
        }
        rows.append(row)
        print(
            f"      Feb-sel {row['Feb-sel']:.5f}　Feb-es {row['Feb-es']:.5f}"
            f"　Mar {row['Mar 時間外']:.5f}\n",
            flush=True,
        )

    table = pl.DataFrame(rows)
    ref_in, ref_out = table["Feb-sel"][0], table["Mar 時間外"][0]
    table = table.with_columns(
        (pl.col("Feb-sel") / ref_in - 1).round(4).alias("Feb-sel 相對 A"),
        (pl.col("Mar 時間外") / ref_out - 1).round(4).alias("時間外相對 A"),
    )

    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_width_chars(200)
    pl.Config.set_tbl_cols(20)
    print("=" * 88)
    print("Target encoding 對照（SPEC §5 紅線 6）—— 洩漏教學實驗，不參與選模")
    print("=" * 88)
    print(table)

    # ---- 正式結論只看合規的兩列 ----
    compliant = table.filter(pl.col("合規") == "✅ 合規")
    a, b = compliant.row(0, named=True), compliant.row(1, named=True)
    delta = b["Mar 時間外"] / a["Mar 時間外"] - 1

    print("\n" + "=" * 88)
    print("正式結論（只比較 A 與 B —— 違規變體不得進入任何結論）")
    print("=" * 88)
    print(
        f"  A 原生類別　Mar {a['Mar 時間外']:.5f}\n"
        f"  B OOF TE 　Mar {b['Mar 時間外']:.5f}　（相對 A {delta:+.2%}）"
    )
    print(
        "\n  ⚠️ 這是**單次量測**。§7.12 已示範單一切分的解析度極限"
        "（配對 σ 0.00044），\n"
        "     要採用 B 必須先照配對 multi-seed 協定重測。"
    )

    print("\n" + "=" * 88)
    print("違規控制組的代價（⛔ 不可部署、不得寫入 artifact、不得進 headline）")
    print("=" * 88)
    for row in table.filter(pl.col("合規") == "⛔ 違規").iter_rows(named=True):
        gap = row["Feb-sel"] - row["Mar 時間外"]
        print(
            f"  {row['變體']:32s} Feb-sel {row['Feb-sel']:.5f}"
            f"　Mar {row['Mar 時間外']:.5f}　剪刀差 {gap:+.5f}"
        )

    print(
        "\n讀法：\n"
        "  C 的違規**量不出來** —— 每個類別有兩萬多列，自己那一列的標籤只佔 1/24000。\n"
        "  D 切分後才編碼 → es/sel/Mar 拿不到編碼、只能吃先驗，early stopping 立刻停手。\n"
        "  E 切分前就編碼 → 內部分數好得不像話、時間外崩盤，這就是紅線 6 的剪刀差。\n"
        "\n結論：違規的代價由**欄位基數**與**編碼發生在切分的哪一側**決定，不是由\n"
        "「有沒有用 target encoding」決定。所以紅線 6 的門檻訂在作法上 —— 靠看分數\n"
        "來抓這個錯誤，在 C 那種情況下抓不到。\n"
        "\n⚠️ 本腳本不產生任何模型 artifact，也不參與選模。最終模型仍是 §7.12 的\n"
        "   CatBoost（Mar 0.15367，配對 multi-seed 8/8）。"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
