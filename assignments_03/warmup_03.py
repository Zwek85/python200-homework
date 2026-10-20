import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from pathlib import Path
import joblib

df = pd.read_csv(Path(__file__).parent / "weather_classification.csv")
numeric = ["temperature_2m_max", "temperature_2m_min", "precipitation_sum", "wind_speed_10m_max"]
X = df[numeric]
y = df["good_for_running"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Preprocessing ---
 
# Q1
# Fit the scaler on X_train only so no information from the test set leaks into preprocessing.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
print("Q1: Mean of each scaled training column (should be ~0):")
print(pd.Series(X_train_scaled.mean(axis=0), index=numeric))

# Q2
encoder = OneHotEncoder(sparse_output=False)
season_encoded = encoder.fit_transform(df[["season"]])
 
print("Q2: Encoded season shape:", season_encoded.shape)
print("Q2: Category names:", encoder.get_feature_names_out())
# The single season column became one column per distinct season value (4 if the data has
# winter/spring/summer/fall), because one-hot encoding gives each category its own 0/1 column
# so the model doesn't treat the categories as ordered numbers.

# Q3
# Same split settings as the setup block, but with season included as a feature
feature_cols = numeric + ["season"]
X_full = df[feature_cols]
X_train_q3, X_test_q3, y_train_q3, y_test_q3 = train_test_split(
    X_full, y, test_size=0.2, random_state=42, stratify=y
)
 
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), ["season"]),
])
 
pipe = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=1000)),
])
 
pipe.fit(X_train_q3, y_train_q3)
print("Q3: Pipeline test accuracy:", accuracy_score(y_test_q3, pipe.predict(X_test_q3)))
# The pipeline automatically fits the scaler and encoder on the training data only, applies those
# same fitted transformations to the test data at predict time, and feeds the combined result to
# the model, so I don't have to scale, encode, and stack the columns by hand for each split.

 #--- KNN ---
 
# Q1
knn_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
 
knn_pipe.fit(X_train, y_train)
knn_pred = knn_pipe.predict(X_test)
 
print("KNN Q1: Test accuracy:", accuracy_score(y_test, knn_pred))
print("KNN Q1: Classification report:")
print(classification_report(y_test, knn_pred))

# Q2
k_values = [1, 3, 5, 7, 9, 11, 15, 21]
cv_results = {}
 
for k in k_values:
    k_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])
    scores = cross_val_score(k_pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_results[k] = scores.mean()
    print(f"KNN Q2: k={k}, mean 5-fold CV accuracy = {cv_results[k]:.4f}")
 
best_k = max(cv_results, key=cv_results.get)
print("KNN Q2: Best k by mean CV accuracy:", best_k)
# I would choose k = 11, because it has the highest mean CV accuracy on the training data (0.9417,
# just ahead of k=7 at 0.9396, so the top scores are very close). Very small k (like 1) overfits to noise, very large k oversmooths, and when
# scores are nearly tied I'd prefer the larger k for a more stable decision boundary.
 
# Q3
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
acc_raw = accuracy_score(y_test, knn_raw.predict(X_test))
 
knn_scaled = Pipeline([
    ("scaler", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
knn_scaled.fit(X_train, y_train)
acc_scaled = accuracy_score(y_test, knn_scaled.predict(X_test))
 
print("KNN Q3: Test accuracy without scaling:", acc_raw)
print("KNN Q3: Test accuracy with scaling:", acc_scaled)
# Yes, scaling helped: test accuracy rose from 0.925 without scaling to about 0.958 with it. Scaling matters for a distance-based
# model because KNN measures Euclidean distance, so features with larger numeric ranges (e.g.
# temperature vs. precipitation) dominate the distance and drown out features with smaller ranges.

 #--- Logistic Regression ---
 
# Q1
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000)),
])
lr_pipe.fit(X_train, y_train)
lr_pred = lr_pipe.predict(X_test)
 
# Use "good" as the positive class if labels are strings, otherwise fall back to 1
classes = list(lr_pipe.classes_)
pos_label = "good" if "good" in classes else 1
pos_idx = classes.index(pos_label)
print("LR: Classes (in model order):", classes, "| positive class:", pos_label)
 
print("LR Q1: Test accuracy:", accuracy_score(y_test, lr_pred))
print("LR Q1: Precision:", precision_score(y_test, lr_pred, pos_label=pos_label))
print("LR Q1: Recall:", recall_score(y_test, lr_pred, pos_label=pos_label))
print("LR Q1: F1:", f1_score(y_test, lr_pred, pos_label=pos_label))
 
# Q2
coefs = lr_pipe.named_steps["logreg"].coef_[0]
print(f"LR Q2: Coefficients (positive pushes toward class '{classes[1]}'):")
for name, coef in zip(numeric, coefs):
    print(f"  {name}: {coef:.4f}")
# Wind speed (-2.44) and precipitation (-1.89) push most strongly toward "skip", since good is the
# positive class (1) and these have the most negative coefficients. This matches my intuition:
# runners avoid windy, rainy days. The two temperature coefficients are small (-0.12 and +0.26),
# probably because the best running temperatures are mild rather than "higher is better", which a
# linear model can't capture well.
 
# Q3
probs = lr_pipe.predict_proba(X_test)[:, pos_idx]
print("LR Q3: P(good) for first five test days:", np.round(probs[:5], 2))
# A probability near 0.5 means the model is very uncertain about that day: the features sit close to
# the decision boundary, so it is essentially a coin flip between "good" and "skip". Probabilities near
# 0 or 1 mean the model is confident.
 
 # --- Evaluation ---
 
import os
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)
 
# Q1
# Order the labels as [skip, good] so the matrix matches display_labels regardless of how y is encoded
neg_label = [c for c in classes if c != pos_label][0]
cm = confusion_matrix(y_test, lr_pred, labels=[neg_label, pos_label])
tn, fp, fn, tp = cm.ravel()
print("Eval Q1: Confusion matrix (rows = actual, cols = predicted; order: skip, good):")
print(cm)
print(f"Eval Q1: False positives = {fp}, False negatives = {fn}")
 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["skip", "good"])
disp.plot()
plt.title("Logistic Regression Confusion Matrix")
plt.savefig("outputs/logreg_confusion_matrix.png", bbox_inches="tight")
plt.close()
# The model made 9 false positives (predicted "good" when the day was really "skip") and 7 false
# negatives (predicted "skip" when the day was really "good").
# A false positive means predicting "good" on a day that was really "skip", so a runner would be sent
# out in bad weather. A false negative means telling them to skip a day that was fine. For a running
# app, false positives probably matter more because they send people into bad conditions, while a
# false negative only costs a missed run.
 
# --- joblib ---
 
# Q1
pipeline = lr_pipe
joblib.dump(pipeline, "models/warmup_model.pkl")
 
loaded = joblib.load("models/warmup_model.pkl")
assert (loaded.predict(X_test) == pipeline.predict(X_test)).all()
print("Predictions match.")
# If I had saved only the LogisticRegression step, it would expect already-scaled inputs. Calling
# .predict() on raw data would feed it values on the wrong scale (e.g. temperatures in raw units
# instead of standardized ones), so the coefficients would be applied to the wrong numbers and the
# predictions would be silently wrong, with no error raised. Saving the whole pipeline keeps the
# fitted scaler's mean and std together with the model.