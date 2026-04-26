"""
eda_and_visualize.py  –  Nykaa Hyperlocal Demand Forecaster
=============================================================
Rich EDA visualisations. Run this first to understand the data
before training Prophet models.

Outputs → plots/eda/
"""

import os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings("ignore")
os.makedirs("plots/eda", exist_ok=True)

# Nykaa brand colours
NYKAA_PINK   = "#FF3F6C"
NYKAA_DARK   = "#1C1C1C"
NYKAA_GREY   = "#F5F5F5"
PALETTE      = [NYKAA_PINK, "#FF8FAB", "#C9184A", "#590D22", "#FF6B9D"]

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/nykaa_beauty_sales.csv", parse_dates=["date"])
df["month"]     = df["date"].dt.month
df["year"]      = df["date"].dt.year
df["yearmonth"] = df["date"].dt.to_period("M").astype(str)

print("=" * 60)
print("  NYKAA DEMAND FORECASTER  –  EDA REPORT")
print("=" * 60)
print(f"\nDataset: {len(df):,} rows | {df['city'].nunique()} cities | {df['product'].nunique()} products")
print(f"\nOverall stockout rate: {df['stockout'].mean():.1%}")
print(f"\nStockout by city:\n{df.groupby('city')['stockout'].mean().sort_values(ascending=False).apply(lambda x: f'{x:.1%}').to_string()}")

# ── Plot 1: Monthly revenue trend by city ─────────────────────────────────────
monthly = df.groupby(["yearmonth","city"])["units_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(16, 5))
colors_city = dict(zip(df["city"].unique(), PALETTE))
for city in df["city"].unique():
    sub = monthly[monthly["city"]==city]
    ax.plot(range(len(sub)), sub["units_sold"], label=city,
            linewidth=2, color=colors_city[city], marker="o", markersize=2)

ticks = list(range(0, len(monthly["yearmonth"].unique()), 3))
ax.set_xticks(ticks)
ax.set_xticklabels(monthly["yearmonth"].unique()[ticks], rotation=45, ha="right")
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
ax.set_title("Monthly Total Units Sold  –  All Cities", fontsize=14, color=NYKAA_DARK)
ax.set_ylabel("Units Sold"); ax.legend(title="City", loc="upper left")
ax.axvspan(21, 22, alpha=0.15, color=NYKAA_PINK, label="Diwali zone")
plt.tight_layout()
plt.savefig("plots/eda/01_monthly_trend_by_city.png", dpi=130, bbox_inches="tight")
plt.close(); print("\n✅  Plot 1: Monthly trend by city")

# ── Plot 2: Festival demand spike heatmap ────────────────────────────────────
fest_df = df[df["festival"]!="None"].copy()
fest_df["festival_clean"] = fest_df["festival"].str.replace(r" \(nearby\)","",regex=True)
avg_boost = fest_df.groupby(["festival_clean","city"])["festival_boost"].mean().unstack(fill_value=1.0)
# Keep top 10 festivals by avg boost
top_fests = (fest_df.groupby("festival_clean")["festival_boost"]
             .mean().sort_values(ascending=False).head(10).index)
avg_boost = avg_boost.loc[avg_boost.index.isin(top_fests)]

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(avg_boost, annot=True, fmt=".2f", cmap="RdYlGn",
            linewidths=0.4, ax=ax, vmin=1.0, vmax=2.8,
            cbar_kws={"label": "Demand Multiplier (1.0 = baseline)"})
ax.set_title("Festival Demand Multiplier  ×  City\n(How much each festival boosts sales vs a normal day)",
             fontsize=13, color=NYKAA_DARK)
ax.set_xlabel("City"); ax.set_ylabel("Festival")
plt.tight_layout()
plt.savefig("plots/eda/02_festival_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 2: Festival heatmap")

# ── Plot 3: Stockout risk by city × product ───────────────────────────────────
pivot = df.groupby(["city","product"])["stockout"].mean().unstack()
fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(pivot, annot=True, fmt=".0%", cmap="YlOrRd",
            linewidths=0.4, ax=ax,
            cbar_kws={"format": mtick.PercentFormatter(xmax=1)})
ax.set_title("⚠️  Stockout Rate  –  City × Product  (Critical Supply Chain Risk)", fontsize=13, color=NYKAA_DARK)
plt.tight_layout()
plt.savefig("plots/eda/03_stockout_risk_heatmap.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 3: Stockout risk heatmap")

# ── Plot 4: Seasonal weather effect on Sunscreen & Moisturizer ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, product in zip(axes, ["Sunscreen SPF50", "Moisturizer"]):
    sub = df[df["product"]==product].groupby(["month","city"])["units_sold"].mean().reset_index()
    for city in CITIES if (CITIES := df["city"].unique()) else []:
        c = sub[sub["city"]==city]
        ax.plot(c["month"], c["units_sold"], marker="o", label=city,
                color=colors_city[city], linewidth=2)
    ax.set_title(f"{product}  –  Seasonal Demand by City", fontsize=12, color=NYKAA_DARK)
    ax.set_xlabel("Month"); ax.set_ylabel("Avg Daily Units")
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
                       rotation=30)
    ax.legend(fontsize=8)
plt.suptitle("Weather-Driven Demand  –  Key Insight: Jaipur/Chandigarh Sunscreen Peaks in April-May",
             fontsize=11, color=NYKAA_DARK)
plt.tight_layout()
plt.savefig("plots/eda/04_seasonal_weather_effect.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 4: Seasonal weather effect")

# ── Plot 5: Nykaa sale events impact ─────────────────────────────────────────
sale_days   = df[df["festival"].str.contains("Nykaa", na=False)]
normal_days = df[(df["festival"]=="None") & (df["is_weekend"]==0)]
fig, ax = plt.subplots(figsize=(10, 5))
products = df["product"].unique()
x = np.arange(len(products))
w = 0.35
sale_avg   = [sale_days[sale_days["product"]==p]["units_sold"].mean() for p in products]
normal_avg = [normal_days[normal_days["product"]==p]["units_sold"].mean() for p in products]
b1 = ax.bar(x - w/2, normal_avg, w, label="Normal weekday", color="#CCCCCC")
b2 = ax.bar(x + w/2, sale_avg,   w, label="Nykaa Sale Day", color=NYKAA_PINK)
ax.set_xticks(x); ax.set_xticklabels(products, rotation=10)
ax.set_title("Nykaa Sale Events vs Normal Days  –  Average Units Sold",
             fontsize=13, color=NYKAA_DARK)
ax.set_ylabel("Avg Units Sold")
ax.legend()
ax.bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
ax.bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
plt.tight_layout()
plt.savefig("plots/eda/05_nykaa_sale_impact.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 5: Nykaa sale events impact")

# ── Plot 6: Year-over-year growth per product ────────────────────────────────
yoy = df.groupby(["year","product"])["units_sold"].sum().reset_index()
fig, ax = plt.subplots(figsize=(10, 5))
for i, product in enumerate(df["product"].unique()):
    sub = yoy[yoy["product"]==product]
    ax.plot(sub["year"], sub["units_sold"], marker="o", label=product,
            color=PALETTE[i % len(PALETTE)], linewidth=2.5)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"{x/1000:.0f}k"))
ax.set_title("Year-over-Year Demand Growth by Product Category", fontsize=13, color=NYKAA_DARK)
ax.set_xlabel("Year"); ax.set_ylabel("Total Units")
ax.set_xticks([2022,2023,2024]); ax.legend()
plt.tight_layout()
plt.savefig("plots/eda/06_yoy_growth.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 6: YoY growth")

# ── Plot 7: Weekend vs weekday demand ────────────────────────────────────────
wk = df.groupby(["is_weekend","product"])["units_sold"].mean().reset_index()
wk["day_type"] = wk["is_weekend"].map({0:"Weekday",1:"Weekend"})
fig, ax = plt.subplots(figsize=(10, 4))
sns.barplot(data=wk, x="product", y="units_sold", hue="day_type",
            palette={"Weekday":"#AAAAAA","Weekend":NYKAA_PINK}, ax=ax)
ax.set_title("Weekend Uplift – Average Daily Units (All Cities)", fontsize=13, color=NYKAA_DARK)
ax.set_xlabel("Product"); ax.set_ylabel("Avg Units Sold"); ax.legend(title="Day Type")
plt.tight_layout()
plt.savefig("plots/eda/07_weekend_uplift.png", dpi=130, bbox_inches="tight")
plt.close(); print("✅  Plot 7: Weekend uplift")

print("\n✅  All 7 EDA charts saved to  plots/eda/")
