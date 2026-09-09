"""EDA 01 · 資料總覽與流失訊號探索

這支腳本回答 SPEC §5.1「避雷清單」裡列出的待答問題。每一段都對應一個
具體的建模決策 —— 不是隨便畫圖，是「不先看這個，後面會做錯」的東西。

怎麼在 PyCharm 跑：
  方法一（推薦，可以一段一段看）
      PyCharm 支援 `# %%` 儲存格標記。把游標放到某段裡，按
      Ctrl+Enter 就只跑那一段，圖會直接顯示在右側。
  方法二（整支跑完）
      右鍵 → Run 'eda_01_overview'。所有圖會存到 reports/figures/，
      最後一次全部跳出來。

第一次跑會比較久（要掃 2,154 萬列交易），之後會讀快取所以很快。
若想強制重算，把 FORCE_REBUILD 改成 True。

注意：本腳本只做「看資料」，不做特徵工程也不訓練模型。依 SPEC §7 的
repo 結構，notebooks/ 只放 EDA。
"""

# %%
# ============================================================================
# 0. 設定
# ============================================================================
import math

import matplotlib
import matplotlib.pyplot as plt
import polars as pl

from src.config import REPO_ROOT, load_paths
from src.data import FEB, build_cohort

# 路徑從 src.config 拿。這支腳本原本自己讀了一次 paths.yaml，跟
# scripts/download.py 的載入邏輯重複 —— 現在兩邊走同一份程式碼，
# 換電腦時也只有 configs/paths.yaml 一個地方要改。
PATHS = load_paths().ensure()
RAW = PATHS.raw
FIGDIR = PATHS.figures

FORCE_REBUILD = False

# Windows 上 matplotlib 預設字體沒有中文，不設定的話圖上的中文會變成 □□□。
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",  # 微軟正黑體，繁中優先
    "Microsoft YaHei",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False  # 負號也要用正常字體
matplotlib.rcParams["figure.dpi"] = 110

pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(180)


def save(fig, name: str) -> None:
    """存圖到 reports/figures/。圖不 close，這樣整支跑完時會一次全部顯示。"""
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"    圖已存 → reports/figures/{name}.png")


def pct(x: float) -> str:
    return f"{x:.2%}"


print(f"repo   {REPO_ROOT}")
print(f"raw    {RAW}")
print(f"圖表   {FIGDIR}")


# %%
# ============================================================================
# 1. 標籤總覽：兩個 cohort 的流失率
# ============================================================================
# 為什麼先看這個：SPEC §4.2 用 train.csv（2月到期）訓練、train_v2.csv（3月到期）
# 驗證。如果兩期的流失率差很多，代表有概念漂移，驗證分數不能直接當測試分數看。

train = pl.read_csv(RAW / "train.csv")
train_v2 = pl.read_csv(RAW / "train_v2.csv")

feb_rate = train["is_churn"].mean()
mar_rate = train_v2["is_churn"].mean()

print(f"train.csv    (Feb cohort)  {train.height:,} 列   流失率 {pct(feb_rate)}")
print(f"train_v2.csv (Mar cohort)  {train_v2.height:,} 列   流失率 {pct(mar_rate)}")
print(f"兩期相差 {(mar_rate / feb_rate - 1):.1%}  ← SPEC §2.1 說這是真實的概念漂移，不是雜訊")

# 常數預測的 log loss：把訓練集流失率套到每個人身上。
# 這是 SPEC §3.3 的 M1 驗收門檻 —— 打不贏它的模型沒有存在意義。
const_ll = -(mar_rate * math.log(feb_rate) + (1 - mar_rate) * math.log(1 - feb_rate))
print(f"\n用 Feb 流失率 {pct(feb_rate)} 對 Mar cohort 做常數預測 → log loss {const_ll:.5f}")
print("這就是 M1 要打敗的基準線。")

fig, ax = plt.subplots(figsize=(5, 3.6))
bars = ax.bar(
    ["Feb cohort\n(train.csv)", "Mar cohort\n(train_v2.csv)"],
    [feb_rate * 100, mar_rate * 100],
    color=["#4a7ba7", "#a74a4a"],
    width=0.55,
)
for b, v in zip(bars, [feb_rate, mar_rate], strict=True):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.15, pct(v), ha="center", fontsize=11)
ax.set_ylabel("流失率 (%)")
ax.set_title("兩期 cohort 流失率：訓練集 vs 驗證集")
ax.set_ylim(0, max(feb_rate, mar_rate) * 100 * 1.25)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "01_cohort_churn_rate")


# %%
# ============================================================================
# 2. 建立 as-of 特徵表（會快取，第一次比較久）
# ============================================================================
# 這是整個專案的防洩漏核心（SPEC §4.3、紅線 1）。
#
# cutoff(user) = 該用戶的 membership_expire_date
# 所有進特徵的 transaction_date 必須 <= cutoff。
#
# 為什麼：標籤定義是「到期後 30 天內有沒有新交易」。到期日之後的那筆交易
# 就是答案本身。讓它進特徵，CV 分數會漂亮到不真實，上線後全崩。
#
# 實作已經抽到 src/data/cohort.py。原因不是「整理程式碼」，是 M1 要用
# Feb cohort 訓練、Mar cohort 驗證 —— 同一段截斷邏輯要跑兩次。留在
# notebook 裡就得複製貼上改日期，那正是紅線 1 最容易破功的地方。
#
# 那個模組還內建了守門檢查：算完會驗證沒有任何用戶的最後一筆交易晚於
# 自己的 cutoff，違反就直接 raise。想看實作按住 Ctrl 點 build_cohort。

df = build_cohort(FEB, PATHS, force=FORCE_REBUILD)
print(f"\nas-of 特徵表 {df.height:,} 列 × {df.width} 欄")
print(df.columns)


# %%
# ============================================================================
# 3. 交易史長度 vs 流失率
# ============================================================================
# SPEC §4.5 預測「用戶歷史長度會是壓倒性主力特徵」。這裡驗證它。
#
# 注意 SPEC §4.5 的警告：不能用「有沒有出現在上一期標籤檔」來算這個特徵，
# 因為測試集（Apr cohort）沒有上一期標籤檔可查，部署時算不出來。要用
# as-of 的交易史長度來代理 —— 這正是下面在做的事。

hist = (
    df.with_columns(
        pl.when(pl.col("n_tx") <= 2)
        .then(pl.lit("1-2"))
        .when(pl.col("n_tx") <= 6)
        .then(pl.lit("3-6"))
        .when(pl.col("n_tx") <= 12)
        .then(pl.lit("7-12"))
        .when(pl.col("n_tx") <= 24)
        .then(pl.lit("13-24"))
        .otherwise(pl.lit("25+"))
        .alias("bin")
    )
    .group_by("bin")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
)
order = ["1-2", "3-6", "7-12", "13-24", "25+"]
hist = hist.with_columns(
    pl.col("bin").replace_strict({b: i for i, b in enumerate(order)}).alias("_o")
).sort("_o")
print(hist.drop("_o"))

fig, ax1 = plt.subplots(figsize=(7.5, 4))
ax1.bar(hist["bin"], hist["人數"], color="#c9d6e3", width=0.6, label="人數")
ax1.set_xlabel("cutoff 之前的交易筆數")
ax1.set_ylabel("人數", color="#5a6b7a")
ax1.ticklabel_format(axis="y", style="plain")

ax2 = ax1.twinx()
ax2.plot(hist["bin"], hist["流失率"] * 100, "o-", color="#a74a4a", lw=2.2, ms=8, label="流失率")
for x, y in zip(hist["bin"], hist["流失率"], strict=True):
    ax2.annotate(
        pct(y),
        (x, y * 100),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="#a74a4a",
        fontsize=10,
    )
ax2.set_ylabel("流失率 (%)", color="#a74a4a")
ax2.set_ylim(0, hist["流失率"].max() * 100 * 1.3)

ax1.set_title("交易史越長，流失率越低（單調關係）")
save(fig, "02_history_length_vs_churn")


# %%
# ============================================================================
# 4. is_cancel 與 is_auto_renew：兩個最強的旗標
# ============================================================================
# SPEC §5.1 要求：「把 is_cancel 當標籤或當強特徵前，先確認它與 is_churn
# 的實際關係。」這裡就是在做這件確認。
#
# 官方明示用戶可能因「換方案」而取消，所以 is_cancel=1 不等於流失。
# 下面會看到取消者裡確實有一部分沒流失 —— 那群人就是換方案的。

cross = (
    df.group_by("last_is_cancel", "last_is_auto_renew")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .sort("last_is_cancel", "last_is_auto_renew")
)
print(cross)

for col, label in [
    ("last_is_cancel", "最後一筆是否取消"),
    ("last_is_auto_renew", "最後一筆是否自動續訂"),
]:
    g = (
        df.group_by(col)
        .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
        .sort(col)
    )
    print(f"\n{label}")
    print(g)

# 熱圖：兩個旗標交叉之後的流失率。
#
# ⚠️ **每個旗標有三個取值，不是兩個。** 同一天有多筆交易而該欄位彼此衝突時，
# `_last_unambiguous()` 給 null（見 src/data/cohort.py）——「不知道」是那裡唯一
# 誠實的值。所以 null 必須在圖上有自己的一格，不能靜靜丟掉：實測 9,179 人的
# `last_is_cancel` 是 null，而他們的流失率 18.95% 恰好落在已取消（85.70%）與
# 未取消（4.26%）之間，正是「這格真的混著兩種人」的樣子。
#
# 這一段原本寫死成 2x2，會在 r["last_is_cancel"] 是 None 時以
# `TypeError: list indices must be integers` 掛掉 —— 也就是說 §7.11 的同日交易
# 修正之後，這張圖就再也沒有重畫成功過。改成由 AXIS 決定維度，之後再多一個
# 取值也不會炸。
#
# 空格用 nan 而不是 None —— matplotlib 的 imshow 不吃 object dtype，但認得 nan
# 並且會留白。而空格本身就是發現：沒開自動續訂就沒東西可取消，所以
# (已取消, 自動續訂關) 一筆都不存在。
AXIS = [0, 1, None]
CANCEL_LABEL = {0: "未取消", 1: "已取消", None: "同日衝突\n（不確定）"}
RENEW_LABEL = {0: "自動續訂 關", 1: "自動續訂 開", None: "同日衝突\n（不確定）"}

grid = [[float("nan")] * len(AXIS) for _ in AXIS]
counts = [[0] * len(AXIS) for _ in AXIS]
for r in cross.iter_rows(named=True):
    i = AXIS.index(r["last_is_cancel"])
    j = AXIS.index(r["last_is_auto_renew"])
    grid[i][j] = r["流失率"] * 100
    counts[i][j] = r["人數"]

fig, ax = plt.subplots(figsize=(7.6, 5.4))
im = ax.imshow(grid, cmap="Reds", vmin=0, vmax=100)
ax.set_xticks(range(len(AXIS)), [RENEW_LABEL[a] for a in AXIS])
ax.set_yticks(range(len(AXIS)), [CANCEL_LABEL[a] for a in AXIS])
for i in range(len(AXIS)):
    for j in range(len(AXIS)):
        if math.isnan(grid[i][j]):
            ax.text(j, i, "此組合\n不存在", ha="center", va="center", fontsize=9, color="#999999")
            continue
        ax.text(
            j,
            i,
            f"{grid[i][j]:.2f}%\n{counts[i][j]:,} 人",
            ha="center",
            va="center",
            fontsize=11,
            color="white" if grid[i][j] > 50 else "black",
        )
ax.set_title("流失率：最後一筆交易的兩個旗標交叉")
fig.colorbar(im, ax=ax, label="流失率 (%)", shrink=0.8)
save(fig, "03_cancel_x_autorenew_heatmap")


# %%
# ============================================================================
# 4b. 這兩個旗標各佔「全部流失人數」的多少？
# ============================================================================
# 上面看的是「每個分群裡有幾成流失」。這裡問的是反方向的問題：
# 「全部流失的人，是從哪些分群來的？」
#
# 這是給業務單位看的角度。分群流失率高但人數少，對總體影響有限；
# 要投放預算，看的是「這群人佔了多少流失量」。

seg = (
    df.with_columns(
        pl.when(pl.col("last_is_cancel") == 1)
        .then(pl.lit("A 已取消"))
        .when(pl.col("last_is_auto_renew") == 0)
        .then(pl.lit("B 自動續訂關閉"))
        .otherwise(pl.lit("C 自動續訂開啟且未取消"))
        .alias("segment")
    )
    .group_by("segment")
    .agg(
        pl.len().alias("人數"),
        pl.col("is_churn").sum().alias("流失人數"),
        pl.col("is_churn").mean().alias("流失率"),
    )
    .sort("segment")
)
total_churn = df["is_churn"].sum()
seg = seg.with_columns(
    (pl.col("人數") / df.height).alias("佔用戶比"),
    (pl.col("流失人數") / total_churn).alias("佔流失量比"),
)
print(seg)

print(f"""
→ 用兩個旗標就能把 {df.height:,} 人切成三群：
   A+B 合計 {seg["人數"][0] + seg["人數"][1]:,} 人
   （{(seg["人數"][0] + seg["人數"][1]) / df.height:.1%} 的用戶）
   卻涵蓋 {(seg["流失人數"][0] + seg["流失人數"][1]) / total_churn:.1%} 的流失量。

   這是給業務單位的第一句話：挽回預算不必撒在全體，
   盯住「已取消」與「自動續訂關閉」這兩群就好。

   但也要誠實說明限制：這兩個旗標是規則，不是模型。
   機器學習真正的價值在「同一分群內部還能不能排序」——
   C 群 {seg["人數"][2]:,} 人裡那 {seg["流失人數"][2]:,} 個流失者，
   規則找不出來，那才是模型要解的問題（SPEC §4.5 第 3 點）。
""")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.5, 4))
labels = [s.split(" ", 1)[1] for s in seg["segment"]]
colors = ["#a74a4a", "#d08a3e", "#4a7ba7"]

axa.bar(labels, seg["佔用戶比"] * 100, color=colors, width=0.55)
for i, v in enumerate(seg["佔用戶比"]):
    axa.text(i, v * 100 + 1.2, f"{v:.1%}", ha="center", fontsize=10)
axa.set_ylabel("佔全體用戶 (%)")
axa.set_title("三個分群的人數佔比")
axa.set_ylim(0, 100)
axa.tick_params(axis="x", labelsize=9)
axa.spines[["top", "right"]].set_visible(False)

axb.bar(labels, seg["佔流失量比"] * 100, color=colors, width=0.55)
for i, v in enumerate(seg["佔流失量比"]):
    axb.text(i, v * 100 + 1.2, f"{v:.1%}", ha="center", fontsize=10)
axb.set_ylabel("佔全部流失人數 (%)")
axb.set_title("三個分群的流失量佔比")
axb.set_ylim(0, 100)
axb.tick_params(axis="x", labelsize=9)
axb.spines[["top", "right"]].set_visible(False)

fig.suptitle("少數人貢獻多數流失：兩張圖要對照著看", fontsize=12)
save(fig, "03b_segment_churn_share")


# %%
# ============================================================================
# 5. bd（年齡）—— SPEC §5.1 要求做對照實驗的欄位
# ============================================================================
# 官方明示這個欄位含 -7000 ~ 2015 的離群值。直接當數值特徵會毀掉模型。
# 處理方式（截斷／分箱／視為缺失）必須做對照實驗，並在 README 記錄理由。
# 這裡先量清楚問題有多大。

bd = df["bd"]
valid = df.filter(pl.col("bd").is_between(10, 100))
print(f"bd 範圍       {bd.min()} ~ {bd.max()}")
print(f"落在 10~100   {valid.height:,} 人 ({valid.height / df.height:.2%})")
print(f"bd = 0        {df.filter(pl.col('bd') == 0).height:,} 人")
print(f"bd < 0        {df.filter(pl.col('bd') < 0).height:,} 人")
print(f"bd > 100      {df.filter(pl.col('bd') > 100).height:,} 人")
print(f"bd 為 null    {df.filter(pl.col('bd').is_null()).height:,} 人（不在 members_v3）")

# 關鍵問題：離群值那群人的流失率跟正常值那群一樣嗎？
# 如果不一樣，「bd 是不是離群值」本身就是一個有訊號的特徵。
print("\n依 bd 是否合理分群的流失率：")
print(
    df.with_columns(
        pl.when(pl.col("bd").is_null())
        .then(pl.lit("null（不在 members）"))
        .when(pl.col("bd").is_between(10, 100))
        .then(pl.lit("合理 10~100"))
        .otherwise(pl.lit("離群值"))
        .alias("bd_group")
    )
    .group_by("bd_group")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .sort("人數", descending=True)
)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.8))
axa.hist(valid["bd"].to_list(), bins=45, color="#4a7ba7", edgecolor="white", linewidth=0.4)
axa.set_xlabel("bd（年齡）")
axa.set_ylabel("人數")
axa.set_title(f"只看 10~100 歲：{valid.height:,} 人（{valid.height / df.height:.1%}）")
axa.spines[["top", "right"]].set_visible(False)

groups = ["bd = 0", "bd < 0", "bd > 100", "10~100", "null"]
vals = [
    df.filter(pl.col("bd") == 0).height,
    df.filter(pl.col("bd") < 0).height,
    df.filter(pl.col("bd") > 100).height,
    valid.height,
    df.filter(pl.col("bd").is_null()).height,
]
axb.barh(groups, vals, color=["#a74a4a", "#a74a4a", "#a74a4a", "#4a7ba7", "#999999"])
for i, v in enumerate(vals):
    axb.text(v + max(vals) * 0.01, i, f"{v:,}", va="center", fontsize=9)
axb.set_xlabel("人數")
axb.set_title("bd 的實際組成：多數不可用")
axb.spines[["top", "right"]].set_visible(False)
save(fig, "04_bd_age_distribution")


# %%
# ============================================================================
# 6. 缺失本身是訊號嗎？gender 與 members_v3 覆蓋率
# ============================================================================
# SPEC §5.1：「gender 缺失 65.43%，『缺失』本身可能就是一個有訊號的類別。」
# 這一段驗證這句話。順便驗證 SPEC §2.1 說「join 後不會有大量缺失」是否成立。

print("gender 分群：")
gen = (
    df.with_columns(pl.col("gender").fill_null("(缺失)"))
    .group_by("gender")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .sort("人數", descending=True)
)
print(gen)

print("\nmembers_v3 覆蓋率：")
mem = (
    df.group_by("in_members")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .sort("in_members")
)
print(mem)
n_missing = df.filter(~pl.col("in_members")).height
print(f"\n→ {n_missing:,} 人（{n_missing / df.height:.2%}）在 members_v3 查不到。")
print("  SPEC §2.1 寫「join 後不會有大量缺失」，實測不成立，這條要更正。")

fig, (axa, axb) = plt.subplots(1, 2, figsize=(11, 3.8))
axa.bar(gen["gender"], gen["流失率"] * 100, color="#4a7ba7", width=0.55)
for i, (v, n) in enumerate(zip(gen["流失率"], gen["人數"], strict=True)):
    axa.text(i, v * 100 + 0.1, f"{pct(v)}\n{n:,}人", ha="center", fontsize=9)
axa.set_ylabel("流失率 (%)")
axa.set_title("gender：缺失是不是一個有訊號的類別？")
axa.set_ylim(0, gen["流失率"].max() * 100 * 1.35)
axa.spines[["top", "right"]].set_visible(False)

labels = ["在 members_v3" if v else "查不到" for v in mem["in_members"]]
axb.bar(labels, mem["流失率"] * 100, color=["#4a7ba7", "#999999"], width=0.5)
for i, (v, n) in enumerate(zip(mem["流失率"], mem["人數"], strict=True)):
    axb.text(i, v * 100 + 0.1, f"{pct(v)}\n{n:,}人", ha="center", fontsize=9)
axb.set_ylabel("流失率 (%)")
axb.set_title("members_v3 覆蓋率與流失率")
axb.set_ylim(0, mem["流失率"].max() * 100 * 1.35)
axb.spines[["top", "right"]].set_visible(False)
save(fig, "05_missingness_as_signal")


# %%
# ============================================================================
# 7. payment_method_id —— 決定要不要 target encoding
# ============================================================================
# SPEC 紅線 6：payment_method_id 是高基數類別，直接 target encode 會讓 CV
# 飆高、實測崩盤，必須 out-of-fold。
#
# 先看它到底值不值得特別處理：如果各付款方式的流失率差很多，就值得；
# 如果差不多，就不用冒紅線 6 的風險。

pm = (
    df.group_by("last_payment_method_id")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .filter(pl.col("人數") >= 500)  # 人數太少的流失率不穩，先濾掉
    .sort("流失率", descending=True)
)
print(f"付款方式基數 {df['last_payment_method_id'].n_unique()} 種")
print(f"人數 >= 500 的有 {pm.height} 種：")
print(pm.head(20))

top = pm.head(15).reverse()
fig, ax = plt.subplots(figsize=(7.5, 5.5))
bars = ax.barh(
    [str(m) for m in top["last_payment_method_id"]],
    top["流失率"] * 100,
    color="#4a7ba7",
)
for b, v, n in zip(bars, top["流失率"], top["人數"], strict=True):
    ax.text(
        v * 100 + 0.4,
        b.get_y() + b.get_height() / 2,
        f"{pct(v)}  ({n:,}人)",
        va="center",
        fontsize=8.5,
    )
ax.axvline(feb_rate * 100, color="#a74a4a", ls="--", lw=1.5, label=f"整體流失率 {pct(feb_rate)}")
ax.set_xlabel("流失率 (%)")
ax.set_ylabel("payment_method_id")
ax.set_title("各付款方式的流失率差異（僅列人數 500 以上的前 15 名）")
ax.legend(loc="lower right")
ax.set_xlim(0, top["流失率"].max() * 100 * 1.35)
ax.spines[["top", "right"]].set_visible(False)
save(fig, "06_payment_method_churn")


# %%
# ============================================================================
# 8. payment_plan_days —— 月租戶 vs 其他方案
# ============================================================================
# SPEC §4.5 說新進用戶流失率 39.84%，包含「首次訂閱期滿的新客與非月租方案
# 到期者」。這裡拆開看非月租族群到底有多大、風險多高。

pd_ = (
    df.group_by("last_payment_plan_days")
    .agg(pl.len().alias("人數"), pl.col("is_churn").mean().alias("流失率"))
    .sort("人數", descending=True)
)
print(pd_.head(12))
n30 = pd_.filter(pl.col("last_payment_plan_days") == 30)["人數"][0]
print(f"\n30 天方案佔 {n30 / df.height:.2%}")

top = pd_.head(8)
fig, ax1 = plt.subplots(figsize=(8, 4))
xs = [str(d) for d in top["last_payment_plan_days"]]
ax1.bar(xs, top["人數"], color="#c9d6e3", width=0.6)
ax1.set_yscale("log")  # 30 天佔絕大多數，用對數軸才看得到其他方案
ax1.set_xlabel("payment_plan_days（方案天數）")
ax1.set_ylabel("人數（對數軸）", color="#5a6b7a")

ax2 = ax1.twinx()
ax2.plot(xs, top["流失率"] * 100, "o-", color="#a74a4a", lw=2, ms=7)
for x, v in zip(xs, top["流失率"], strict=True):
    ax2.annotate(
        pct(v),
        (x, v * 100),
        textcoords="offset points",
        xytext=(0, 9),
        ha="center",
        color="#a74a4a",
        fontsize=9,
    )
ax2.set_ylabel("流失率 (%)", color="#a74a4a")
ax2.set_ylim(0, top["流失率"].max() * 100 * 1.3)
ax1.set_title("方案天數分布與各自流失率")
save(fig, "07_plan_days_vs_churn")


# %%
# ============================================================================
# 9. 資料品質哨兵值
# ============================================================================
# SPEC §2.1 列了兩個哨兵值，前處理必須處理：
#   transactions.csv    membership_expire_date 最小值 19700101（Unix epoch，等同 null）
#   transactions_v2.csv membership_expire_date 最大值 20361015（2036 年，不合理）
# 不處理的話，任何「到期日 - 交易日」的日期運算都會產生垃圾特徵。

tx_lazy = pl.concat(
    [pl.scan_csv(RAW / "transactions.csv"), pl.scan_csv(RAW / "transactions_v2.csv")]
)
sentinel = tx_lazy.select(
    pl.len().alias("總列數"),
    (pl.col("membership_expire_date") == 19700101).sum().alias("到期日=19700101"),
    (pl.col("membership_expire_date") > 20180101).sum().alias("到期日>2018"),
    (pl.col("transaction_date") > pl.col("membership_expire_date")).sum().alias("交易日>到期日"),
    (pl.col("actual_amount_paid") == 0).sum().alias("實付=0"),
    (pl.col("actual_amount_paid") < pl.col("plan_list_price")).sum().alias("實付<定價"),
).collect(engine="streaming")
print("全量交易（transactions + transactions_v2）的哨兵值統計：")
for k, v in sentinel.row(0, named=True).items():
    print(f"  {k:20s} {v:>12,}")


# %%
# ============================================================================
# 10. 小結
# ============================================================================
def rate_where(col: str, val: int) -> str:
    return pct(df.filter(pl.col(col) == val)["is_churn"].mean())


print("=" * 74)
print("EDA 01 小結")
print("=" * 74)
print(f"""
1. 兩期流失率 {pct(feb_rate)} → {pct(mar_rate)}，常數基準 log loss = {const_ll:.5f}
   這是 M1 的驗收門檻（SPEC §3.3）。

2. 最強的兩個 as-of 旗標（都來自 cutoff 之前的最後一筆交易）：
     最後一筆已取消      流失率 {rate_where("last_is_cancel", 1)}
     最後一筆未取消      流失率 {rate_where("last_is_cancel", 0)}
     自動續訂關閉        流失率 {rate_where("last_is_auto_renew", 0)}
     自動續訂開啟        流失率 {rate_where("last_is_auto_renew", 1)}

   ⚠️ 「最後一筆已取消」這個訊號要小心。取消常常發生在到期日當天，而
      cutoff 就是到期日，所以它幾乎等於答案。SPEC §4.3 要求 M6 另做
      cutoff = 到期日 - 7 天 的版本，那個版本會失去這個訊號，分數必然
      下降 —— 而那才是誠實的部署分數。

3. 交易史長度與流失率單調負相關，印證 SPEC §4.5 的預測。

4. 待決事項（都要在 M1 之前想清楚）：
   - bd：只有 {valid.height / df.height:.1%} 落在 10~100，處理方式要做對照實驗
   - gender / members 缺失：缺失本身有訊號，不能隨便補值
   - payment_method_id：{df["last_payment_method_id"].n_unique()} 種，若要 target
     encoding 必須 out-of-fold（紅線 6）

5. SPEC 要更正：§2.1 寫 members_v3「join 後不會有大量缺失」，
   實測 {n_missing:,} 人（{n_missing / df.height:.2%}）查不到。
""")
print(f"所有圖表已存到 {FIGDIR}")

# 整支跑完時，一次顯示所有圖。用 PyCharm 的儲存格模式（Ctrl+Enter）逐段跑
# 的話，圖已經在右側顯示過了，這行不會有額外效果。
plt.show()
