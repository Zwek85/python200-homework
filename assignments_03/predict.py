# Load the saved weather classifier and predict whether days are good for running.
# No training code here: no fit, no cross-validation, no API fetching.

import json
import joblib
import pandas as pd
import sklearn

MODEL_PATH = "models/weather_classifier.pkl"
METADATA_PATH = "models/weather_classifier_metadata.json"


def load_model():
    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    if metadata["sklearn_version"] != sklearn.__version__:
        print("Warning: scikit-learn version mismatch.")
        print(f"  saved with:  {metadata['sklearn_version']}")
        print(f"  running on:  {sklearn.__version__}")

    model = joblib.load(MODEL_PATH)
    return model, metadata


if __name__ == "__main__":
    model, metadata = load_model()
    features = metadata["feature_names"]

    # Key metadata
    print("Model metadata")
    print(f"  City:          {metadata['city']}")
    print(f"  Features:      {features}")
    print(f"  Test accuracy: {metadata['test_accuracy']}")
    print(f"  Test F1:       {metadata['test_f1']}")
    print()

    # Hypothetical days (temps in C, precipitation in mm, wind in km/h).
    # Row order matches `features`: temp_max, temp_min, precip, wind_max.
    descriptions = [
        "Clearly good: mild, dry, calm",
        "Clearly bad: freezing, heavy rain, windy",
        "Borderline: high just above the 26 C cap",
        "Borderline: rain just under the 3 mm cutoff",
        "Borderline: rain just over the 3 mm cutoff",
    ]
    rows = [
        [20.0, 9.0, 0.0, 12.0],
        [3.0, -4.0, 12.0, 45.0],
        [26.5, 14.0, 0.0, 10.0],
        [18.0, 8.0, 2.8, 20.0],
        [18.0, 8.0, 3.2, 20.0],
    ]
    days = pd.DataFrame(rows, columns=features)

    labels = model.predict(days)
    prob_good = model.predict_proba(days)[:, 1]

    print("Predictions")
    for i, desc in enumerate(descriptions):
        day = days.iloc[i]
        verdict = "good" if labels[i] == 1 else "skip"
        print(f"\n{desc}")
        for name in features:
            print(f"  {name}: {day[name]}")
        print(f"  Predicted label: {verdict}")
        print(f"  Confidence (P good): {prob_good[i]:.2f}")

# ---------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------
# 1. Borderline day: I picked the day with 3.2 mm of rain (just over the 3 mm
#    cutoff, so my label rule says "skip"). The model gave it P(good) = 0.71
#    and predicted "good," which disagrees with the rule. With k=7, that means
#    5 of the 7 nearest training days were good. I would call the model
#    moderately confident, but wrong here: KNN never learns the exact 3 mm
#    cutoff, it only sees that days with about 3 mm of rain, mild temperatures,
#    and light wind were mostly good. The 2.8 mm day got the same 0.71, which
#    shows the model cannot tell those two days apart. For a prediction of
#    0.52, I would not treat it as a firm answer. I would show the
#    probability instead of a hard label, or add a "maybe" band (for example,
#    0.35 to 0.65) and let the user decide. Since sending someone out on a bad
#    day is the more annoying error, I could also require P(good) above about
#    0.6 before saying "good."
#
# 2. If predict.py runs before train_classifier.py has ever run, the models/
#    folder and its two files do not exist, so the script fails with a
#    FileNotFoundError before it can load anything. The raw traceback does not
#    say what to do. load_model() now checks that both files exist first and
#    raises an error saying to run train_classifier.py first.
#
# 3. This script's example days are hardcoded in the __main__ block, and the
#    model paths are relative to the folder it is run from. A cloud pipeline
#    would need a function that takes a DataFrame of real daily weather
#    (validating that it has the four feature columns in the right units) and
#    returns labels and probabilities, with the model path configurable
#    instead of hardcoded.