import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from weatherkit import WeatherResponse

# A plain relative path like "weather_raw.json" is resolved against the
# directory the program is run FROM (the current working directory), not
# the directory this file lives in. If pytest is run from a different
# folder, the file would not be found. Building the path from __file__
# always points at this file's location, so it works from anywhere.
DATA_PATH = Path(__file__).parent.parent / "weather_raw.json"


def make_raw(**overrides) -> dict:
    """Return a small, valid response dict, with optional top-level overrides."""
    raw = {
        "latitude": 35.2,
        "longitude": -80.8,
        "timezone": "America/New_York",
        "elevation": 230.0,
        "hourly": {
            "time": ["2026-04-08T00:00", "2026-04-08T01:00", "2026-04-08T02:00"],
            "temperature_2m": [10.0, 11.5, 12.0],
            "precipitation": [0.0, 0.1, 0.0],
        },
    }
    raw.update(overrides)
    return raw


def test_real_file_validates_with_168_hours():
    with open(DATA_PATH) as f:
        raw = json.load(f)
    response = WeatherResponse.model_validate(raw)
    assert len(response.hourly.time) == 168


def test_latitude_out_of_range_raises():
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(make_raw(latitude=200.0))


def test_mismatched_list_lengths_raise():
    raw = make_raw()
    raw["hourly"]["precipitation"] = [0.0, 0.1]  # one element short
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)


def test_null_temperature_raises():
    raw = make_raw()
    raw["hourly"]["temperature_2m"] = [10.0, None, 12.0]
    with pytest.raises(ValidationError):
        WeatherResponse.model_validate(raw)