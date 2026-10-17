# --- scikit-learn API ---

import numpy as np
from sklearn.linear_model import LinearRegression

# Q1
temp_min = np.array([1, 4, 8, 12, 16, 20]).reshape(-1, 1)
temp_max = np.array([8, 10, 15, 18, 23, 27])

# create
model = LinearRegression()

# fit
model.fit(temp_min, temp_max)

# predict (inputs must be 2D, so each value gets its own inner list)
predictions = model.predict(np.array([[6], [18]]))

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("Predicted high for a low of 6:", predictions[0])
print("Predicted high for a low of 18:", predictions[1])

# Q2
x = np.array([10, 20, 30, 40, 50])

print("Original shape:", x.shape)

x_2d = x.reshape(-1, 1)

print("Reshaped shape:", x_2d.shape)

# scikit-learn expects X to be 2D with shape (n_samples, n_features): each row
# is one sample and each column is one feature. A 1D array doesn't say whether
# it holds 5 samples with 1 feature or 1 sample with 5 features, so scikit-learn
# requires the extra dimension to remove that ambiguity. This also lets the
# same code work whether you have one feature or many.

# --- Linear Regression ---

import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

rng = np.random.default_rng(42)
num_days  = 120
temp_min  = rng.uniform(-5, 22, num_days)
is_rainy  = rng.integers(0, 2, num_days).astype(float)
temp_max  = 1.0 * temp_min + 6 - 4 * is_rainy + rng.normal(0, 2, num_days)

# Q1
plt.scatter(temp_min, temp_max, c=is_rainy, cmap="coolwarm")
plt.title("Daily High vs Low")
plt.xlabel("Low temperature")
plt.ylabel("High temperature")

out_path = Path(__file__).parent / "outputs" / "high_vs_low.png"
plt.savefig(out_path)
plt.close()
print("Saved plot to", out_path)

# What I see: the points rise from left to right, so warmer lows go with
# warmer highs. They also split into two bands. The red dots (rainy days)
# sit about 4 degrees below the blue dots (dry days) at the same low
# temperature. This tells me is_rainy affects the high temperature, so it
# should be used as a feature in the model along with temp_min.

# Q2
X = temp_min.reshape(-1, 1)
y = temp_max

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Q3
model = LinearRegression()
model.fit(X_train, y_train)

print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("R² on test set:", model.score(X_test, y_test))

# The slope is how much the daily high goes up for
# each 1 degree increase in the daily low. A slope close to 1 means that on a
# warmer-low day, the high is warmer by about the same amount. For example, if
# the low is 5 degrees higher, the predicted high is about 5 degrees higher too.

# Q4
X_full = np.column_stack([temp_min, is_rainy])

# new variable names so the Q2/Q3 split and model stay untouched
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)

r2_simple = model.score(X_test, y_test)               # Q3 model (low temp only)
r2_full = model_full.score(X_test_full, y_test_full)  # Q4 model (low temp + rain)

print("Q3 R² (temp_min only):", r2_simple)
print("Q4 R² (temp_min + is_rainy):", r2_full)

print("temp_min coefficient:", model_full.coef_[0])
print("is_rainy coefficient:", model_full.coef_[1])

# Does adding rain help? Yes. The R² went up compared to Q3, so the model
# explains the daily high better when it knows whether it rained.
#
# The is_rainy coefficient is how much the predicted high changes on a rainy
# day compared to a dry day, when the low temperature is the same. It should
# be about -3, meaning rainy days have a high about 3 degrees cooler than dry
# days with the same low. In practical terms, rain "pulls down" the high.

# Q5
y_pred_full = model_full.predict(X_test_full)

plt.scatter(y_pred_full, y_test_full)

# diagonal reference line: where predicted = actual
low = min(y_pred_full.min(), y_test_full.min())
high = max(y_pred_full.max(), y_test_full.max())
plt.plot([low, high], [low, high], color="red", linestyle="--")

plt.title("Predicted vs Actual")
plt.xlabel("Predicted high temperature")
plt.ylabel("Actual high temperature")

out_path = Path(__file__).parent / "outputs" / "predicted_vs_actual_warmup.png"
plt.savefig(out_path)
plt.close()
print("Saved plot to", out_path)

# A point above the diagonal means the actual value was higher than the
# model predicted, so the model underestimated. A point
# below the diagonal means the actual value was lower than predicted, so the
# model overestimated. The farther a point is from the
# diagonal, the bigger the mistake. If the points sit close to the line
# with no pattern, the model is doing a good job.