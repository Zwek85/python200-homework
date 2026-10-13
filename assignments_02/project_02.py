import os

import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Task 1: Fetch the Data

url = "https://archive-api.open-meteo.com/v1/archive"
params = {
    "latitude": 45.52,       # City Chose: Portland, OR
    "longitude": -122.68,
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "daily": [
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
    ],
    "timezone": "America/Los_Angeles",
}

response = requests.get(url, params=params)
response.raise_for_status()

df = pd.DataFrame(response.json()["daily"])
df["date"] = pd.to_datetime(df["time"])
df = df.drop("time", axis=1)

print("Shape:", df.shape)
print(df.head())

# Task 2: Look at the Features

# 1. Summary statistics and missing values
print("\n--- describe() ---")
print(df.describe())

print("\n--- Missing values per column ---")
print(df.isna().sum())

# Missing-value decision: this is daily weather, so neighboring days are
# similar. If any values were missing, I would fill them by linear
# interpolation instead of dropping rows, because dropping a day would
# leave a gap in the time series and throw away the other columns for
# that day. If the check above shows 0 missing values, no action is needed.
if df.isna().any().any():
    df = df.sort_values("date").reset_index(drop=True)
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
    print("Missing values after interpolation:", df.isna().sum().sum())

# 2. Correlation of each numeric column with the daily high
print("\n--- Correlation with temperature_2m_max ---")
corr = (
    df.select_dtypes("number")
    .corr()["temperature_2m_max"]
    .drop("temperature_2m_max")
    .sort_values(ascending=False)
)
print(corr)

# The strongest relationship (by absolute value) is shown at the top or
# bottom of the sorted list above. I expect temperature_2m_min to be
# strongest, since warm days tend to have warm nights.
# precipitation_sum should be weakly negative (rainy days are cooler and
# cloudier), and wind_speed_10m_max should be weak.

# 3. Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

axes[0].hist(df["temperature_2m_max"], bins=30, edgecolor="black")
axes[0].set_xlabel("Daily high (°C)")
axes[0].set_ylabel("Number of days")
axes[0].set_title("Distribution of daily highs, Portland 2023")

axes[1].scatter(df["temperature_2m_min"], df["temperature_2m_max"], alpha=0.5)
axes[1].set_xlabel("Daily low (°C)")
axes[1].set_ylabel("Daily high (°C)")
axes[1].set_title("Daily low vs. daily high")

plt.tight_layout()
plt.savefig("outputs/task2_high_distribution_and_low_vs_high.png", dpi=150)
plt.close()

# What I see:
# - Histogram: the daily highs form a single hump (one peak) instead of
#   two clusters, so there is no clear split between a cool group of days
#   and a warm group.
# - Scatter: the daily low and daily high rise together in a roughly
#   straight line, so a linear relationship looks reasonable.
#   This suggests linear regression is a good starting model.


# Task 3: Engineer a Feature

df["is_summer"] = df["date"].dt.month.isin([6, 7, 8]).astype(int)

print("\n--- Summer vs. non-summer days ---")
print(df["is_summer"].value_counts().rename({1: "summer", 0: "non-summer"}))

# Do I expect is_summer to help? Yes. Summer in Portland is dry and warm,
# so the daily high is much higher in June-August than the rest of the
# year. The flag gives the model a simple way to capture that seasonal
# shift. However, temperature_2m_min already carries a lot of the same
# seasonal information (warm nights happen in summer), so the extra gain
# may be small. The flag is mainly useful for days where the low doesn't
# fully explain the high.

# Task 4: Baseline Model

X = df[["temperature_2m_min"]]   # 2D: sklearn expects a DataFrame/matrix
y = df["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Baseline model: temperature_2m_min -> temperature_2m_max ---")
print(f"Slope:     {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"RMSE:      {rmse:.3f} °C")
print(f"R²:        {r2:.3f}")

# RMSE in plain English: a typical prediction is off by about 4.4 °C.
# If the model predicts a high of 20 °C, the real high is usually within
# roughly 4 degrees of that, above or below. That's decent but not
# precise: the model explains about 74% of the variation in daily highs
# (R² = 0.741), so about a quarter is still unexplained.
#
# The slope of 1.42 means: for each 1 °C increase in the daily low, the
# model predicts the daily high goes up by about 1.4 °C. The slope is
# above 1 because the gap between day and night widens on warm days
# (clear, dry summer days heat up a lot during the day).

# Task 5: Full Model

baseline_r2 = r2         
baseline_rmse = rmse

feature_cols = ["temperature_2m_min", "precipitation_sum",
                "wind_speed_10m_max", "is_summer"]
X = df[feature_cols].values
y = df["temperature_2m_max"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

train_r2 = model.score(X_train, y_train)
test_r2 = model.score(X_test, y_test)
y_pred = model.predict(X_test)
full_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Full model (4 features) ---")
print(f"Train R²:  {train_r2:.3f}")
print(f"Test R²:   {test_r2:.3f}")
print(f"Test RMSE: {full_rmse:.3f} °C")
print(f"Baseline test R² was {baseline_r2:.3f}, RMSE {baseline_rmse:.3f} °C")

print("\n--- Coefficients ---")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:22s}: {coef:+.3f}")

# Comparison to baseline: test R² went from 0.741 to 0.866 and RMSE went
# from 4.38 to 3.15 °C. Adding three features cut the typical error by
# about 1.2 °C (roughly 28%) and explained about 12 more percentage points
# of the variation in daily highs. That is a clear improvement.

# Coefficients:
# - Raise the predicted high: temperature_2m_min (+1.21 per °C) and
#   is_summer (+3.04). Holding everything else fixed, a summer day is
#   predicted to be about 3 °C hotter than a non-summer day with the same
#   low, precipitation, and wind. is_summer helps even though the low
#   already carries seasonal information.
# - Lower the predicted high: precipitation_sum (-0.21 per mm of rain) and
#   wind_speed_10m_max (-0.10 per km/h). Rainy days are cloudier and
#   cooler, and windier days (often storm systems or cold fronts) tend to
#   be cooler.
# - Surprises: none of the signs are surprising. One thing to note is that
#   the temperature_2m_min slope dropped from 1.42 in the baseline to 1.21
#   here. That is because is_summer now absorbs part of the seasonal
#   effect that the low was picking up on its own. Coefficients are in
#   different units, so their sizes can't be compared directly.

# Train vs. test R²: train is 0.875 and test is 0.866, a gap of only
# 0.009. They are very close, so the model is generalizing well and is not
# overfitting. That makes sense with only 4 features and about


# Task 6: Evaluate and Summarize

# y_test and y_pred are still the full-model values from Task 5.
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test, y_pred, alpha=0.6)

lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")  # diagonal reference

ax.set_xlabel("Actual daily high (°C)")
ax.set_ylabel("Predicted daily high (°C)")
ax.set_title("Predicted vs Actual (Full Model)")
ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "task2_predicted_vs_actual_high.png", dpi=150)  # outputs/predicted_vs_actual_high.png
plt.close()

# Where does the model struggle? The error is not perfectly even.
# - Middle of the range (actual highs of about 5-13 °C, the cool wet
#   months): the dots hug the line on average but have the widest vertical
#   scatter. Many days are off by 3-6 °C either way, and this is where
#   most of the points are.
# - High end (actual highs above about 30 °C): the dots fall below the
#   line, so the model under-predicts the hottest days. For example, an
#   actual high of about 38 °C is predicted at about 32 °C. A linear model
#   tends to pull extreme values toward the average.
# - Biggest single miss: one day with an actual high of about 5 °C was
#   predicted at about -6 °C, an error of roughly 11 °C. Another day
#   (actual about 17.5 °C) was predicted near 11 °C.
# Overall the model is best in the warm-to-hot middle, tends to
# under-predict heat extremes, and has a few large one-off misses.

# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------
# Dataset: 365 days of Portland, OR weather for 2023 (one row per day).
# The test set is 73 days (20%); the other 292 days were used for training.
#
# Performance: the full model has a test RMSE of 3.15 °C and an R² of
# 0.866. A typical prediction is off by about 3 degrees Celsius (roughly
# 5-6 °F). If the model says 20 °C, the real high is usually somewhere
# between about 17 and 23 °C. The model explains about 87% of the
# day-to-day variation