"""
forecast.py  –  Nykaa Hyperlocal Demand Forecaster
====================================================
Trains a Facebook Prophet model for every (city × product) pair.

External regressors fed into the model:
  - festival_boost   : how strong the festival demand lift is
  - weather_boost    : seasonal/climate effect for that product
  - is_weekend       : Saturday/Sunday binary flag

Model design choice – WHY PROPHET?
  Prophet is interpretable. You can show the decomposed components
  (trend + yearly seasonality + weekly seasonality + regressor effects)
  in a chart that a product manager or supply chain team can read.
  That's exactly what Nykaa's analysts need.

Outputs:
  forecasts/  → {City}_{Product}_forecast.csv  (15 files)
  plots/forecasts/ → forecast chart per model  (15 PNGs)
  reports/stockout_alert_report.csv  → actionable stockout warnings
  metrics.csv → MAE, RMSE, MAPE for model accuracy benchmarking
"""

import os, warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

os.makedirs("forecasts",       exist_ok=True)
os.makedirs("plots/forecasts", exist_ok=True)
os.makedirs("reports",         exist_ok=True)

# ── Nykaa Brand colours ───────────────────────────────────────────────────────
NYKAA_PINK = "#FF3F6C"
NYKAA_DARK = "#1C1C1C"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/nykaa_beauty_sales.csv", parse_dates=["date"])
CITIES   = sorted(df["city"].unique())
PRODUCTS = sorted(df["product"].unique())

FORECAST_DAYS = 90   # 3-month horizon — quarterly planning window

metrics_rows  = []
alert_rows    = []

print("=" * 70)
print("  NYKAA HYPERLOCAL DEMAND FORECASTER  –  Prophet Models")
print("  Cities:", ", ".join(CITIES))
print("=" * 70)

for city in CITIES:
    for product in PRODUCTS:

        # ── Prepare model dataframe ───────────────────────────────────────
        subset = (df[(df["city"]==city) & (df["product"]==product)]
                  .sort_values("date").reset_index(drop=True))

        model_df = subset.rename(columns={"date":"ds","units_sold":"y"})

        # ── Train / Test split (last 60 days held out) ────────────────────
        split_idx = len(model_df) - 60
        train_df  = model_df.iloc[:split_idx]
        test_df   = model_df.iloc[split_idx:]

        # ── Initialise Prophet ─────────────────────────────────────────────
        #
        # seasonality_mode = "multiplicative":
        #   Means festival/weather effects SCALE with underlying demand
        #   level, rather than adding a fixed offset. This is realistic —
        #   a Diwali boost on a high-demand month is bigger in absolute
        #   units than the same percentage boost on a slow month.
        #
        m = Prophet(
            yearly_seasonality       = True,
            weekly_seasonality       = True,
            daily_seasonality        = False,
            seasonality_mode         = "multiplicative",
            changepoint_prior_scale  = 0.05,   # conservative trend flexibility
            seasonality_prior_scale  = 10.0,
        )

        # Add external regressors
        # standardize=False keeps the original scale for interpretability
        m.add_regressor("festival_boost", standardize=False)
        m.add_regressor("weather_boost",  standardize=False)
        m.add_regressor("is_weekend",     standardize=False)

        m.fit(train_df[["ds","y","festival_boost","weather_boost","is_weekend"]])

        # ── Evaluate on test set ──────────────────────────────────────────
        test_pred = m.predict(
            test_df[["ds","festival_boost","weather_boost","is_weekend"]]
        )
        y_true = test_df["y"].values
        y_pred = test_pred["yhat"].clip(lower=0).values

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100

        metrics_rows.append({
            "city": city, "product": product,
            "MAE": round(mae,2), "RMSE": round(rmse,2), "MAPE%": round(mape,2)
        })
        print(f"  {city:12s} | {product:18s} | MAE={mae:5.1f}  RMSE={rmse:5.1f}  MAPE={mape:4.1f}%")

        # ── Build future dataframe ────────────────────────────────────────
        future = m.make_future_dataframe(periods=FORECAST_DAYS)

        # Regressors for historical period come from the known data
        full_reg_df = model_df[["ds","festival_boost","weather_boost","is_weekend"]].copy()

        # For future dates: use same-day-of-year pattern from historical data
        last_date   = model_df["ds"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta("1D"), periods=FORECAST_DAYS)

        future_regs = []
        for d in future_dates:
            # Weather: monthly average for this product+city
            wb = subset.loc[subset["date"].dt.month == d.month, "weather_boost"].mean()
            wb = float(wb) if not np.isnan(wb) else 1.0

            # Festival: same month-day from historical
            fb = subset.loc[
                (subset["date"].dt.month == d.month) &
                (subset["date"].dt.day   == d.day),
                "festival_boost"
            ].mean()
            fb = float(fb) if not np.isnan(fb) else 1.0

            iw = int(d.weekday() >= 5)
            future_regs.append({"ds": d, "festival_boost": fb,
                                 "weather_boost": wb, "is_weekend": iw})

        future_regs_df = pd.DataFrame(future_regs)
        all_regs = pd.concat([full_reg_df, future_regs_df], ignore_index=True)
        future_with_regs = future.merge(all_regs, on="ds", how="left")
        future_with_regs[["festival_boost","weather_boost","is_weekend"]] = \
            future_with_regs[["festival_boost","weather_boost","is_weekend"]].fillna(1.0)

        forecast = m.predict(future_with_regs)
        forecast["yhat"]       = forecast["yhat"].clip(lower=0).round()
        forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0).round()
        forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0).round()

        # ── Stockout alert: flag future days where yhat > 1.5× avg daily ──
        avg_daily = float(model_df["y"].mean())
        restock_threshold = avg_daily * 1.55

        future_only = forecast[forecast["ds"] > last_date].copy()
        high_risk   = future_only[future_only["yhat"] > restock_threshold]
        for _, row in high_risk.iterrows():
            alert_rows.append({
                "city":                 city,
                "product":              product,
                "date":                 row["ds"].strftime("%Y-%m-%d"),
                "forecast_demand":      int(row["yhat"]),
                "restock_threshold":    int(restock_threshold),
                "excess_units_needed":  int(row["yhat"] - restock_threshold),
                "upper_bound":          int(row["yhat_upper"]),
            })

        # ── Save forecast CSV ─────────────────────────────────────────────
        out_cols = ["ds","yhat","yhat_lower","yhat_upper","trend","weekly","yearly"]
        available = [c for c in out_cols if c in forecast.columns]
        fname = f"forecasts/{city}_{product.replace(' ','_')}_forecast.csv"
        forecast[available].to_csv(fname, index=False)

        # ── Plot ──────────────────────────────────────────────────────────
        fig = m.plot(forecast, figsize=(13, 5))
        fig.axes[0].set_title(
            f"Nykaa  |  {product}  –  {city}  |  90-Day Demand Forecast",
            fontsize=13, color=NYKAA_DARK, fontweight="bold"
        )
        fig.axes[0].set_xlabel("Date")
        fig.axes[0].set_ylabel("Daily Units Sold")
        # Add vertical line at forecast start
        fig.axes[0].axvline(x=last_date, color=NYKAA_PINK, linewidth=1.5,
                            linestyle="--", label="Forecast start")
        fig.axes[0].legend()
        plt.tight_layout()
        pfname = f"plots/forecasts/{city}_{product.replace(' ','_')}.png"
        plt.savefig(pfname, dpi=120, bbox_inches="tight")
        plt.close()

print("=" * 70)

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv("metrics.csv", index=False)

# ── Save stockout alert report ────────────────────────────────────────────────
alert_df = pd.DataFrame(alert_rows)
if not alert_df.empty:
    alert_df = alert_df.sort_values(["date","city","product"])
    alert_df.to_csv("reports/stockout_alert_report.csv", index=False)
    print(f"\n🚨  Stockout alerts found: {len(alert_df)} high-risk days")
    print(f"    Report saved → reports/stockout_alert_report.csv")
else:
    print("\n✅  No stockout risk days in forecast horizon.")

print(f"\n✅  Metrics saved   → metrics.csv")
print(f"✅  Forecasts saved → forecasts/  ({len(CITIES) * len(PRODUCTS)} files)")
print(f"✅  Plots saved     → plots/forecasts/")

print("\n── Model Accuracy Summary ──")
print(metrics_df.groupby("product")[["MAE","RMSE","MAPE%"]].mean().round(2).to_string())

print("\n── Top 5 High-Risk Stockout Alerts (next 90 days) ──")
if not alert_df.empty:
    print(alert_df.nlargest(5, "excess_units_needed")
          [["date","city","product","forecast_demand","restock_threshold","excess_units_needed"]]
          .to_string(index=False))
