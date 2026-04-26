"""
generate_data.py  –  Nykaa Hyperlocal Demand Forecaster
=========================================================
Generates a realistic synthetic dataset modelling Nykaa's
Tier-2 city demand for beauty products.

Cities chosen deliberately:
  - Bhopal     (MP)  : Nykaa's own cited growth city
  - Lucknow    (UP)  : mentioned in Nykaa Tier-2 reports
  - Jaipur     (RJ)  : growing Nykaa Luxe market
  - Chandigarh (PB)  : high per-capita spend tier-2
  - Coimbatore (TN)  : South India anchor

Products mirror Nykaa's top-selling categories:
  Skincare → Moisturizer, Sunscreen SPF50, Face Serum
  Makeup   → Lipstick, Foundation

External factors modelled:
  - 🎉 Indian festival calendar (city-weighted)
  - 🌦️ Season / weather (city-specific climate)
  - 🛍️  Nykaa sale events (Pink Friday, End of Season)
  - 📅 Weekends
  - 📈 Organic growth trend
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta

# ── Config ─────────────────────────────────────────────────────────────────────
CITIES = ["Bhopal", "Lucknow", "Jaipur", "Chandigarh", "Coimbatore"]
PRODUCTS = ["Lipstick", "Sunscreen SPF50", "Moisturizer", "Face Serum", "Foundation"]
START = date(2022, 1, 1)
END   = date(2024, 12, 31)
np.random.seed(2024)

# ── Festival calendar ─────────────────────────────────────────────────────────
# (month, day): (name, national_boost)
FESTIVALS = {
    (1, 14): ("Pongal / Makar Sankranti", 1.5),
    (1, 26): ("Republic Day",             1.2),
    (2, 14): ("Valentine's Day",          1.7),
    (3, 8):  ("Women's Day",              1.6),
    (3, 25): ("Holi",                     1.9),
    (4, 14): ("Baisakhi / Tamil New Year",1.4),
    (6, 17): ("Eid ul-Fitr",              1.7),
    (8, 15): ("Independence Day",         1.3),
    (8, 19): ("Raksha Bandhan",           1.5),
    (9, 22): ("Navratri Start",           2.0),
    (10,2):  ("Gandhi Jayanti",           1.1),
    (10,15): ("Dussehra",                 2.1),
    (11,1):  ("Diwali",                   2.6),
    (11,10): ("Nykaa Pink Friday Sale",   2.2),   # Nykaa's own sale event
    (11,13): ("Chhath Puja",              1.6),
    (12,12): ("Nykaa End of Season Sale", 1.8),
    (12,25): ("Christmas",               1.4),
    (12,31): ("New Year Eve",             1.6),
}

# ── City-level festival affinity (some cities celebrate certain festivals more) ─
# multiplier applied on top of national boost
CITY_FESTIVAL_AFFINITY = {
    # city: {festival_name_fragment: extra_multiplier}
    "Lucknow":    {"Eid": 1.3, "Diwali": 1.2, "Chhath": 1.4},
    "Jaipur":     {"Navratri": 1.3, "Diwali": 1.3, "Teej": 1.4, "Gangaur": 1.3},
    "Chandigarh": {"Baisakhi": 1.5, "Lohri": 1.4, "Diwali": 1.2},
    "Coimbatore": {"Pongal": 1.6, "Tamil New Year": 1.4, "Onam": 1.3},
    "Bhopal":     {"Diwali": 1.2, "Navratri": 1.2, "Eid": 1.2},
}

# ── Seasonal weather boost by city × product × month ─────────────────────────
# Jaipur/Chandigarh have extreme summers → high sunscreen
# Coimbatore is warm year-round → moderate sunscreen always
def weather_boost(city: str, product: str, month: int) -> float:
    hot_cities    = {"Jaipur", "Bhopal", "Lucknow"}
    humid_cities  = {"Coimbatore"}
    cold_cities   = {"Chandigarh"}

    summer = month in (3, 4, 5, 6)
    monsoon= month in (7, 8, 9)
    winter = month in (11, 12, 1)

    if product == "Sunscreen SPF50":
        if city in hot_cities:
            return {3:1.4,4:2.0,5:2.2,6:1.8,7:0.9,8:0.8,9:0.9,
                    10:0.9,11:0.7,12:0.6,1:0.6,2:0.8}.get(month, 1.0)
        if city in humid_cities:      # Coimbatore warm all year
            return 1.3 if summer else 1.1
        if city in cold_cities:       # Chandigarh
            return 2.3 if summer else 0.6
    if product == "Moisturizer":
        if city in cold_cities:
            return 1.9 if winter else (0.8 if summer else 1.0)
        return 1.6 if winter else (0.8 if summer else 1.0)
    if product == "Face Serum":
        return 1.3 if (winter or monsoon) else 1.0
    if product == "Lipstick":
        return 1.2 if winter else (0.9 if monsoon else 1.0)
    if product == "Foundation":
        return 1.0   # relatively stable

    return 1.0


def get_festival_boost(d: date, city: str) -> tuple:
    """Returns (boost, festival_name)"""
    key = (d.month, d.day)
    if key in FESTIVALS:
        name, boost = FESTIVALS[key]
        # City affinity check
        if city in CITY_FESTIVAL_AFFINITY:
            for frag, extra in CITY_FESTIVAL_AFFINITY[city].items():
                if frag.lower() in name.lower():
                    boost *= extra
        return boost, name
    # day before/after gets partial boost
    for delta in [-1, 1]:
        adj = d + timedelta(days=delta)
        k2 = (adj.month, adj.day)
        if k2 in FESTIVALS:
            name, boost = FESTIVALS[k2]
            partial = 1 + (boost - 1) * 0.45
            return partial, name + " (nearby)"
    return 1.0, "None"


def season_label(d: date, city: str) -> str:
    m = d.month
    if city == "Coimbatore":
        if m in (4,5,6):   return "Summer"
        if m in (7,8,9,10):return "Monsoon"
        return "Mild Winter"
    if m in (12,1,2):      return "Winter"
    if m in (3,4,5,6):     return "Summer"
    if m in (7,8,9):       return "Monsoon"
    return "Post-Monsoon"


PRODUCT_BASE = {
    "Lipstick":       48,
    "Sunscreen SPF50":32,
    "Moisturizer":    65,
    "Face Serum":     28,
    "Foundation":     38,
}

CITY_BASE = {
    "Bhopal":     1.00,
    "Lucknow":    1.18,
    "Jaipur":     1.25,
    "Chandigarh": 1.30,
    "Coimbatore": 1.15,
}


def generate():
    rows = []
    current = START
    total_days = (END - START).days + 1

    while current <= END:
        days_elapsed = (current - START).days
        trend = 1 + (days_elapsed / total_days) * 0.20   # 20% organic growth over 3 yrs

        for city in CITIES:
            fest_boost, fest_name = get_festival_boost(current, city)
            is_weekend = int(current.weekday() >= 5)
            season = season_label(current, city)

            for product in PRODUCTS:
                wb    = weather_boost(city, product, current.month)
                cb    = CITY_BASE[city]
                base  = PRODUCT_BASE[product]
                noise = np.random.normal(1.0, 0.07)

                demand = base * cb * wb * fest_boost * trend * noise
                demand += is_weekend * base * cb * 0.10
                demand = max(0, round(demand))

                # Stockout: demand exceeds warehouse replenishment threshold
                # Nykaa's rapid store capacity is roughly 1.5× average daily
                restock_threshold = base * cb * 1.55
                stockout = int(demand > restock_threshold)

                # Stockout risk score 0-100 (useful for dashboards)
                risk_score = min(100, round((demand / restock_threshold) * 100))

                rows.append({
                    "date":             current.isoformat(),
                    "city":             city,
                    "product":          product,
                    "category":         "Skincare" if product in ("Moisturizer","Sunscreen SPF50","Face Serum") else "Makeup",
                    "units_sold":       demand,
                    "festival":         fest_name,
                    "festival_boost":   round(fest_boost, 3),
                    "season":           season,
                    "weather_boost":    round(wb, 2),
                    "is_weekend":       is_weekend,
                    "stockout":         stockout,
                    "stockout_risk_pct":risk_score,
                })

        current += timedelta(days=1)

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/nykaa_beauty_sales.csv", index=False)
    print(f"✅  Dataset created  →  {len(df):,} rows  |  {df['city'].nunique()} cities  |  {df['product'].nunique()} products")
    print(f"\nStockout rate by city:\n{df.groupby('city')['stockout'].mean().round(3).to_string()}")
    print(f"\nSample rows:\n{df.head(5).to_string(index=False)}")
