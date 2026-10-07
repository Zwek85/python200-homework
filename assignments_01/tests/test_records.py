import pytest

from weatherkit import HourlyReading, WeatherResponse, to_readings


@pytest.fixture
def response() -> WeatherResponse:
    """A small validated response with distinct values at every index."""
    return WeatherResponse.model_validate(
        {
            "latitude": 35.2,
            "longitude": -80.8,
            "timezone": "America/New_York",
            "elevation": 230.0,
            "hourly": {
                "time": [
                    "2026-04-08T00:00",
                    "2026-04-08T01:00",
                    "2026-04-08T02:00",
                    "2026-04-08T03:00",
                ],
                "temperature_2m": [10.0, 11.5, 12.0, 9.5],
                "precipitation": [0.0, 0.1, 0.4, 0.0],
            },
        }
    )


def test_one_reading_per_hour_in_order(response):
    readings = to_readings(response)
    assert len(readings) == 4
    assert readings[0].timestamp == "2026-04-08T00:00"
    assert readings[-1].timestamp == "2026-04-08T03:00"


def test_values_match_their_index(response):
    readings = to_readings(response)
    hourly = response.hourly
    for i, reading in enumerate(readings):
        assert reading.timestamp == hourly.time[i]
        assert reading.temperature_c == hourly.temperature_2m[i]
        assert reading.precipitation_mm == hourly.precipitation[i]


def test_identical_readings_are_equal():
    a = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    b = HourlyReading("2026-04-08T00:00", 10.0, 0.0)
    assert a == b