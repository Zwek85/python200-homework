from dataclasses import dataclass

from .schemas import WeatherResponse


@dataclass
class HourlyReading:
    """One hour of weather for a location.

    Attributes:
        timestamp: Local time of the observation as an ISO 8601 string,
            e.g. "2026-10-06T14:00".
        temperature_c: Air temperature at 2 m above ground, in degrees
            Celsius.
        precipitation_mm: Precipitation total for the hour, in millimetres.
    """

    timestamp: str
    temperature_c: float
    precipitation_mm: float


def to_readings(response: WeatherResponse) -> list[HourlyReading]:
    """Convert the columnar hourly block into one reading per hour.

    The API returns time, temperature, and precipitation as three parallel
    lists. Hour i is built by taking index i from each list. Order is
    preserved.

    Args:
        response: A validated WeatherResponse. Its hourly lists are already
            known to be the same length.

    Returns:
        A list of HourlyReading objects, one per hour, in the same order
        as the API's time list.
    """
    hourly = response.hourly
    return [
        HourlyReading(
            timestamp=t,
            temperature_c=temp,
            precipitation_mm=precip,
        )
        for t, temp, precip in zip(
            hourly.time, hourly.temperature_2m, hourly.precipitation
        )
    ]


# Why HourlyReading is a dataclass and WeatherResponse is a Pydantic model:
# the boundary is the point where data enters our program from outside, the
# raw JSON from the API. That data is untrusted: values can be missing, the
# wrong type, out of range, or the lists can be mismatched, so it needs
# Pydantic's validation and type coercion. Once WeatherResponse has
# validated it, and to_readings has converted it, everything is already
# known to be correct. Inside the boundary we only need a simple, fast
# container with named fields, which a dataclass gives us without paying for
# validation a second time. Validate once at the edge, then use plain
# dataclasses for trusted data in the rest of the code.