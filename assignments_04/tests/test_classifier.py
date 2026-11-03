import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weather_model import Prediction, WeatherClassifier

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]

GOOD_DAY = {
    "temperature_2m_max": 20.0,
    "temperature_2m_min": 11.0,
    "precipitation_sum": 0.0,
    "wind_speed_10m_max": 8.0,
}

BAD_DAY = {
    "temperature_2m_max": 1.0,
    "temperature_2m_min": -5.0,
    "precipitation_sum": 18.0,
    "wind_speed_10m_max": 45.0,
}


@pytest.fixture
def classifier(tmp_path):
    """Build a tiny model on synthetic data, save it, and load a WeatherClassifier."""
    rng = np.random.default_rng(0)
    n = 300
    t_max = rng.uniform(-5, 35, n)
    X = pd.DataFrame(
        {
            "temperature_2m_max": t_max,
            "temperature_2m_min": t_max - rng.uniform(3, 10, n),
            "precipitation_sum": rng.uniform(0, 20, n),
            "wind_speed_10m_max": rng.uniform(0, 50, n),
        }
    )[FEATURES]
    # Good for running: warm enough, dry enough, not too windy
    y = (
        (X["temperature_2m_max"] >= 12)
        & (X["precipitation_sum"] < 2)
        & (X["wind_speed_10m_max"] < 25)
    ).astype(int)

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=50, random_state=0)),
        ]
    )
    model.fit(X, y)

    model_path = tmp_path / "tiny_model.pkl"
    joblib.dump(model, model_path)
    return WeatherClassifier(model_path)


# Behaviors 1 and 2: clearly good and clearly bad days get the right label
@pytest.mark.parametrize(
    "day, expected_label",
    [
        (GOOD_DAY, "good"),
        (BAD_DAY, "skip"),
    ],
)
def test_clear_days_get_expected_label(classifier, day, expected_label):
    result = classifier.predict([day])
    assert result[0].label == expected_label


# Behavior 3: probability is between 0 and 1
@pytest.mark.parametrize("day", [GOOD_DAY, BAD_DAY])
def test_probability_between_zero_and_one(classifier, day):
    result = classifier.predict([day])
    assert 0.0 <= result[0].probability <= 1.0


# Behavior 4: three days in, three Prediction objects out, in order
def test_three_days_return_three_predictions_in_order(classifier):
    result = classifier.predict([GOOD_DAY, BAD_DAY, GOOD_DAY])
    assert len(result) == 3
    assert all(isinstance(p, Prediction) for p in result)
    assert [p.label for p in result] == ["good", "skip", "good"]


# Behavior 5: empty list in, empty list out
def test_empty_list_returns_empty_list(classifier):
    assert classifier.predict([]) == []


# Behavior 6: a missing feature raises ValueError that names it
def test_missing_feature_raises_value_error(classifier):
    incomplete = {k: v for k, v in GOOD_DAY.items() if k != "precipitation_sum"}
    with pytest.raises(ValueError, match="precipitation_sum"):
        classifier.predict([incomplete])


# Behavior 7: a missing model file raises FileNotFoundError
def test_missing_model_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="No model file found"):
        WeatherClassifier(tmp_path / "does_not_exist.pkl")

 # Break check: I changed predict() to always return "skip". The tests
# test_clear_days_get_expected_label[good] and
# test_three_days_return_three_predictions_in_order failed, so they catch it.