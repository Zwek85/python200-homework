import json
from pathlib import Path
from dataclasses import dataclass
import pytest

# --- Warmup Question 1 ---
class ModelInfo:
    def __init__(self, path):
        # Read and parse the file once, then keep the result in memory
        with open(path) as f:
            self.data = json.load(f)

    def feature_count(self) -> int:
        return len(self.data["features"])


# Create a small JSON file from Python
json_path = Path(__file__).parent / "sample_model_info.json"
with open(json_path, "w") as f:
    json.dump({"features": ["temperature", "humidity", "wind_speed"]}, f)

info = ModelInfo(json_path)
print("Feature count:", info.feature_count())

# Why load in __init__ instead of inside feature_count()?
# Reading and parsing a file is slow compared to reading from memory. Loading
# in __init__ does that work once, and every later call to feature_count() just
# uses the data already in self.data. If the method loaded the file itself, it
# would hit the disk and re-parse the JSON on every call, which wastes time when
# the method is called many times.

# --- Warmup Question 2 ---
@dataclass
class DayForecast:
    date: str
    high_c: float
    will_rain: bool

    def summary(self) -> str:
        rain = "rain expected" if self.will_rain else "no rain expected"
        return f"{self.date}: high {self.high_c}C, {rain}"


day1 = DayForecast("2023-06-15", 24.1, True)
day2 = DayForecast("2023-06-16", 27.8, False)

print(day1.summary())
print(day2.summary())

# What did @dataclass write for me?
# The __init__ method. I did not have to write
# def __init__(self, date, high_c, will_rain) and assign each field to self
# by hand. @dataclass also generated __repr__ and __eq__.

# --- Warmup Question 3 ---
def parse_temperature(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise ValueError(f"Invalid temperature: {text!r} is not a number")


def test_parse_temperature_valid():
    assert parse_temperature("24.1") == pytest.approx(24.1)


def test_parse_temperature_bad_input_raises():
    with pytest.raises(ValueError, match="temperature"):
        parse_temperature("hot")


# What does pytest.raises(ValueError) without match fail to catch?
# It cannot tell which ValueError was raised. Python's own float("hot") already
# raises a ValueError ("could not convert string to float"), so the test would
# pass even if my custom error handling were deleted or broken. Without match,
# the test misses a wrong or missing "temperature" message. (A different
# exception type, like TypeError, would not be caught at all, and the test
# would fail.)