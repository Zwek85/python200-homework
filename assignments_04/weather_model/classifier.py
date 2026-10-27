from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

FEATURES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
]


@dataclass
class Prediction:
    """The result of classifying one day's weather for running.

    Attributes:
        label: "good" if the day is suitable for running, otherwise "skip".
        probability: Probability, from 0 to 1, that the day is good for running.
    """

    label: str
    probability: float


class WeatherClassifier:
    """Predicts whether days are good for running using a saved model pipeline."""

    def __init__(self, model_path: str | Path) -> None:
        """Load a saved model pipeline from disk.

        Args:
            model_path: Path to a .pkl file saved with joblib.

        Raises:
            FileNotFoundError: If no file exists at model_path.
        """
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"No model file found at '{path}'. Check the path, or retrain "
                "and save the model with your Week 3 train_classifier.py."
            )
        self.model = joblib.load(path)

    def predict(self, days: list[dict]) -> list[Prediction]:
        """Predict running weather for a list of days.

        Args:
            days: One dict per day, each with the keys temperature_2m_max,
                temperature_2m_min, precipitation_sum, and wind_speed_10m_max.

        Returns:
            One Prediction per day, in the same order as the input. An empty
            list if days is empty.

        Raises:
            ValueError: If any day is missing a required feature.
        """
        if not days:
            return []

        missing = [
            feature
            for feature in FEATURES
            if any(feature not in day for day in days)
        ]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        X = pd.DataFrame([{f: day[f] for f in FEATURES} for day in days])[FEATURES]
        labels = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]

        return [
            Prediction(
                label="good" if label == 1 else "skip",
                probability=float(prob),
            )
            for label, prob in zip(labels, probabilities)
        ]