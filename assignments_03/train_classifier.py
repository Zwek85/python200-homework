# Part 2 Mini-Project: Weather Classifier (good day for running?)

import os
import requests
import pandas as pd

os.makedirs("outputs", exist_ok=True)

# ---------------------------------------------------------------
# Task 1: Fetch the data
# City chosen: Portland, Oregon (lat 45.52, lon -122.68)
# ---------------------------------------------------------------
URL = "https://archive-api.open-meteo.com/v1/archive"
PARAMS = {
    "latitude": 45.52,
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


def fetch_weather() -> pd.DataFrame:
    response = requests.get(URL, params=PARAMS, timeout=30)
    response.raise_for_status()
    df = pd.DataFrame(response.json()["daily"])
    df["date"] = pd.to_datetime(df["time"])
    return df.drop("time", axis=1)


if __name__ == "__main__":
    df = fetch_weather()
    print("Shape:", df.shape)
    print(df.head())

# ---------------------------------------------------------------
# Task 2: Engineer the label
# Thresholds: using the assignment defaults, no changes for Portland.
#   high between 7 and 26 C, low >= 0 C,
#   precipitation < 3 mm, max wind < 30 km/h
# ---------------------------------------------------------------
def add_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    good = (
        df["temperature_2m_max"].between(7, 26)
        & (df["temperature_2m_min"] >= 0)
        & (df["precipitation_sum"] < 3)
        & (df["wind_speed_10m_max"] < 30)
    )
    df["good_for_running"] = good.astype(int)
    return df


if __name__ == "__main__":
    df = fetch_weather()
    print("Shape:", df.shape)
    print(df.head())

    df = add_label(df)
    counts = df["good_for_running"].value_counts()
    print("\nClass balance:")
    print(f"Good days (1): {counts.get(1, 0)}")
    print(f"Skip days (0): {counts.get(0, 0)}")
    print(f"Fraction good: {df['good_for_running'].mean():.2f}")

    # Fraction of days that are good: 0.39 (143 good, 222 skip).
    # This seems reasonable for Portland. Most good days should fall in the dry
    # summer months and in mild spring and fall stretches, while the wet season
    # (roughly November through March) is filtered out by the 3 mm rain
    # threshold, and some hot July/August days by the 26 C cap.
    # Imbalance note: the split is about 39/61, which is only mildly imbalanced.
    # Accuracy is still meaningful, but I will also check precision, recall, and
    # F1 for the "good" class, since a model that always predicts "skip" would
    # already score about 61% accuracy.

# ---------------------------------------------------------------
# Task 3: Build and evaluate a KNN classifier
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

X = df[FEATURES]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Choose k with 5-fold cross-validation on the training data only
k_values = [1, 3, 5, 7, 9, 11, 15]
cv_scores = {}
for k in k_values:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_scores[k] = scores.mean()
    print(f"k={k:2d}  mean CV accuracy: {cv_scores[k]:.3f}")

best_k = max(cv_scores, key=cv_scores.get)
print(f"\nBest k: {best_k} (CV accuracy {cv_scores[best_k]:.3f})")

# Fit the chosen model on the full training set and evaluate on the test set
model = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=best_k)),
])
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"\nTest accuracy: {model.score(X_test, y_test):.3f}")
print(classification_report(y_test, y_pred, target_names=["skip (0)", "good (1)"]))

# ---------------------------------------------------------------
# Task 4: Build and evaluate a Logistic Regression classifier
# ---------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
print("\n--- Logistic Regression ---")
print(f"Test accuracy:  {accuracy_score(y_test, y_pred_log):.3f}")
print(f"Precision (good): {precision_score(y_test, y_pred_log):.3f}")
print(f"Recall (good):    {recall_score(y_test, y_pred_log):.3f}")
print(f"F1 (good):        {f1_score(y_test, y_pred_log):.3f}")
print(classification_report(y_test, y_pred_log, target_names=["skip (0)", "good (1)"]))

# Coefficients (on standardized features, so magnitudes are comparable)
coefs = log_model.named_steps["logreg"].coef_[0]
print("Feature coefficients:")
for name, coef in sorted(zip(FEATURES, coefs), key=lambda p: p[1]):
    print(f"  {name:22s} {coef:+.3f}")

# Interpretation: Precipitation has by far the largest coefficient (-3.22),
# so rain is the strongest push toward "skip". One standard deviation more
# precipitation lowers the log-odds of a good day more than any other change.
# Max temperature is next (-1.67): hotter highs push toward skip, which fits the
# 26 C cap that rules out hot summer days. Wind is also negative but weak
# (-0.42), so it matters little here, probably because few Portland days
# exceed the 30 km/h cutoff. Min temperature is positive (+0.95): warmer lows
# push toward "good", consistent with the low >= 0 rule that removes freezing
# nights. The max and min temperature coefficients have opposite signs because
# the two features are strongly correlated and partly cancel, so they should be
# read together rather than one at a time. A linear model also cannot capture
# "too cold AND too hot are both bad," so the temperature coefficients are a
# compromise rather than a clean rule.

# ---------------------------------------------------------------
# Task 5: Compare and choose
# ---------------------------------------------------------------
print("\n--- Model comparison (test set) ---")
for name, m in [("KNN", model), ("Logistic Regression", log_model)]:
    pred = m.predict(X_test)
    print(
        f"{name:20s} acc={accuracy_score(y_test, pred):.3f}  "
        f"prec={precision_score(y_test, pred):.3f}  "
        f"rec={recall_score(y_test, pred):.3f}  "
        f"f1={f1_score(y_test, pred):.3f}"
    )
# Comparison:
# Accuracy: KNN (k=7) scored 0.959 on the test set vs 0.836 for logistic
# regression, so KNN was higher. On 73 test days that is about 70 correct vs
# 61 correct, a gap of 9 days. That is too large to dismiss as noise.
#
# Metric choice: Accuracy alone is not what I care about most for a running app.
# A false positive (app says "good" on a bad day) sends the user out in rain
# or wind, which is the more annoying failure, so precision on the "good" class
# matters most to me. A false negative (app says "skip" on a good day) means a
# missed run, which is a smaller cost, but an over-cautious app loses trust, so
# recall still matters. F1 balances the two. KNN: precision 0.93, recall 0.97,
# F1 0.95. Logistic regression: precision 0.84, recall 0.72, F1 0.78. KNN is
# better on both, so there is no precision/recall tradeoff to agonize over.
#
# Deployment choice: I will deploy KNN (k=7). It is clearly more accurate, and
# the likely reason is that it can represent "good in the middle, bad at both
# ends" for temperature, which the label requires and a linear model cannot.
# The costs are real but small here: KNN stores all 292 training rows, predicts
# by computing distances to each, is harder to interpret than logistic
# regression's coefficients, and gives coarse probabilities (multiples of 1/7).
# At this data size, size and speed are negligible. If the app later needed
# smooth confidence scores or a much larger dataset, I would revisit logistic
# regression or a tree-based model.

final_model = model  # KNN pipeline, k=7

# ---------------------------------------------------------------
# Task 6: Save the model
# ---------------------------------------------------------------
import json
import platform
import joblib
import sklearn

os.makedirs("models", exist_ok=True)

MODEL_PATH = "models/weather_classifier.pkl"
METADATA_PATH = "models/weather_classifier_metadata.json"

joblib.dump(final_model, MODEL_PATH)

final_pred = final_model.predict(X_test)
metadata = {
    "python_version": platform.python_version(),
    "sklearn_version": sklearn.__version__,
    "feature_names": FEATURES,
    "label_rule": (
        "good_for_running = 1 when temperature_2m_max is between 7 and 26 C, "
        "temperature_2m_min >= 0 C, precipitation_sum < 3 mm, and "
        "wind_speed_10m_max < 30 km/h; otherwise 0"
    ),
    "city": "Portland, Oregon",
    "model_type": "KNeighborsClassifier (k=7) in a StandardScaler pipeline",
    "test_accuracy": round(float(accuracy_score(y_test, final_pred)), 4),
    "test_f1": round(float(f1_score(y_test, final_pred)), 4),
}

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved model to {MODEL_PATH}")
print(f"Saved metadata to {METADATA_PATH}")