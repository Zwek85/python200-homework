from pathlib import Path

from weather_model import WeatherClassifier

MODEL_PATH = Path(__file__).parent / "models" / "weather_classifier.pkl"

# Hypothetical days: (description, input features)
DAYS = [
    (
        "Clearly good",
        {
            "temperature_2m_max": 20.0,
            "temperature_2m_min": 11.0,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 8.0,
        },
    ),
    (
        "Clearly bad",
        {
            "temperature_2m_max": 1.0,
            "temperature_2m_min": -5.0,
            "precipitation_sum": 18.0,
            "wind_speed_10m_max": 45.0,
        },
    ),
    (
        "Borderline: cool and a little damp",
        {
            "temperature_2m_max": 11.0,
            "temperature_2m_min": 5.0,
            "precipitation_sum": 2.0,
            "wind_speed_10m_max": 20.0,
        },
    ),
    (
        "Borderline: warm but windy",
        {
            "temperature_2m_max": 24.0,
            "temperature_2m_min": 15.0,
            "precipitation_sum": 0.5,
            "wind_speed_10m_max": 32.0,
        },
    ),
]


def main() -> None:
    classifier = WeatherClassifier(MODEL_PATH)
    predictions = classifier.predict([features for _, features in DAYS])

    for (description, f), pred in zip(DAYS, predictions):
        print(
            f"{description}: "
            f"high {f['temperature_2m_max']}C, low {f['temperature_2m_min']}C, "
            f"rain {f['precipitation_sum']}mm, wind {f['wind_speed_10m_max']}km/h "
            f"-> {pred.label} (P(good) = {pred.probability:.2f})"
        )


# What if the guard were omitted?
# Importing this file from another script (say, to reuse a helper) would run
# main() immediately as a side effect: the model would load and all the
# predictions would print before the importing script did anything. The
# __name__ == "__main__" guard makes main() run only when this file is executed
# directly, not when it is imported.
if __name__ == "__main__":
    main()

# --- Reflection ---
#
# 1. Fixed feature order
# The model was trained on columns in one specific order, and a scikit-learn
# pipeline reads its input by position or by the column names it saw during
# training. If predict() used whatever order the caller's dict happened to have,
# a caller who wrote {"wind_speed_10m_max": 30, "temperature_2m_max": 20, ...}
# could have the wind value read as the temperature. The model would not
# raise an error. It would silently return wrong predictions. Selecting the
# four features in the same fixed order every time prevents that bug.
#
# 2. Tiny model built in a fixture
# Tests that depend on models/weather_classifier.pkl are fragile. They break if
# the file is missing, moved, or retrained (new weights can change a "clear"
# day's label), and they fail on a teammate's machine that doesn't have the
# file. A model built inside the fixture on synthetic data with a fixed random
# seed is the same every run, needs no outside files, and lets the test control
# what a "good" and a "bad" day mean. The tests then check my component's
# behavior (labels, ordering, errors), not the quality of one saved model.
#
# 3. Ready for Week 10, and what to add
# Ready: predict() takes a list of days and returns one Prediction per day in
# the same order, so a pipeline can send it a whole batch of database records
# in one call and match results back to rows. It also fails loudly with a
# ValueError that names the missing features instead of guessing.
# To add before production: validation of the values themselves, not just the
# keys. Database rows can contain None, NaN, or impossible numbers (a wind
# speed of -50), and right now those would reach the model and produce a
# prediction that looks normal. I would also add logging and a check that the
# loaded model's metadata (feature list and version) matches what the code
# expects.