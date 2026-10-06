from dataclasses import dataclass, field, FrozenInstanceError

# --- Classes ---

# Q1
class Thermometer:
    """Stores temperature readings (Celsius) for one location."""

    def __init__(self, location, readings=None):
        self.location = location
        self.readings = list(readings) if readings else []

    def add(self, reading):
        self.readings.append(reading)

    def average(self):
        if not self.readings:
            return None
        return sum(self.readings) / len(self.readings)

    def hottest(self):
        if not self.readings:
            return None
        return max(self.readings)

    # Q2: added __repr__
    def __repr__(self):
        avg = self.average()
        avg_text = round(avg, 1) if avg is not None else None
        return (
            f"Thermometer(location={self.location!r}, "
            f"n_readings={len(self.readings)}, average={avg_text})"
        )


charlotte = Thermometer("Charlotte")
for r in [15.2, 18.9, 20.1, 20.2]:
    charlotte.add(r)

print("Q1 average:", charlotte.average())
print("Q1 hottest:", charlotte.hottest())

# average() must handle the empty case because sum([]) / len([]) is
# 0 / 0, which raises ZeroDivisionError. Without the check, a brand-new
# Thermometer with no readings would crash the program when someone 
# asked for its average. Returning None to say "no answer yet" instead.


# Q2
# (__repr__ is added to the Thermometer class above)

atlanta = Thermometer("Atlanta", [28.5, 31.2, 33.0, 29.8, 35.1])

print("Q2 single:", charlotte)
print("Q2 list:", [charlotte, atlanta])

# Without a __repr__, Python shows the default, something like
# <__main__.Thermometer object at 0x104a3c2d0>. That is just the class name
# and a memory address. But it doesn't tells you about the object's contents,
# so when debugging (or printing a list of objects) you can't tell which
# thermometer is which or what data it holds.


# Q3
class TemperatureAlert:
    """Flags readings above a threshold."""

    def __init__(self, threshold=30.0):
        self.threshold = threshold

    def breaches(self, thermometer):
        return [r for r in thermometer.readings if r > self.threshold]


default_alert = TemperatureAlert()
strict_alert = TemperatureAlert(threshold=33.0)

print("Q3 default (30.0):", default_alert.breaches(atlanta))
print("Q3 strict (33.0):", strict_alert.breaches(atlanta))

# The threshold describes the alert's rule, not what is being checked, so
# it belongs on TemperatureAlert. Configure it once, then call
# alert.breaches(thermo) with just the thermometer. With twenty thermometers
# you can loop over them with one alert object and be make sure that every one is
# judged by the same threshold, instead of passing (and possibly mistyping) the number twenty times.


# --- Dataclasses, Type Hints, and Docstrings ---

# Q1 (and Q2: the class is frozen=True, added for Q2)
@dataclass(frozen=True)
class Station:
    """A weather station with an ID, name, location, and elevation."""

    station_id: str
    name: str
    latitude: float
    longitude: float
    elevation: float


station_a = Station("S1", "Uptown", 35.2271, -80.8431, 229.0)
station_b = Station("S1", "Uptown", 35.2271, -80.8431, 229.0)
print("DC Q1 equal:", station_a == station_b)

# The result is True: a dataclass generates __eq__ that compares all fields
# in order, so two Stations with identical values are equal. With the
# original hand-written class there is no __eq__, so Python falls back to
# comparing identity (are they the same object in memory?). Two separate
# objects would give False even with identical field values.


# Q2
frozen_station = Station("S1", "Uptown", 35.2271, -80.8431, 229.0)
try:
    frozen_station.name = "Downtown"
except FrozenInstanceError as e:
    print("DC Q2 error:", e)

s1 = Station("S1", "Uptown", 35.2271, -80.8431, 229.0)
s2 = Station("S1", "Uptown", 35.2271, -80.8431, 229.0)
s3 = Station("S2", "Airport", 35.2140, -80.9431, 228.0)
print("DC Q2 set size:", len({s1, s2, s3}))

# Besides immutability, frozen=True (together with eq=True) makes the
# dataclass hashable: Python generates __hash__ from the fields. Hashable
# objects can go in sets and be used as dict keys. Here that means identical
# stations collapse into one entry in a set, which is why the length is 2.
# A non-frozen dataclass sets __hash__ to None, so adding it to a set raises
# TypeError.


# Q3
# First attempt:
#
#@dataclass
#class StationBatch:
#   region: str
#   stations: list[Station] = []

# ERROR:
# ValueError: mutable default <class 'list'> for field stations is not
# allowed: use default_factory
#
# Python refuses because a default value is created once, when the class is
# defined, and then shared by every instance. Every StationBatch would end up
# pointing at the same list, so adding a station to one batch would show up
# in all of them. default_factory calls list() fresh for each new instance,
# so each batch gets its own list.

@dataclass
class StationBatch:
    """A group of stations belonging to one region."""

    region: str
    stations: list[Station] = field(default_factory=list)

    def add(self, station: Station) -> None:
        """Add a station to this batch."""
        self.stations.append(station)

    def highest(self) -> Station | None:
        """Return the station with the greatest elevation, or None if empty."""
        if not self.stations:
            return None
        return max(self.stations, key=lambda s: s.elevation)


batch = StationBatch("Carolinas")
print("DC Q3 empty highest:", batch.highest())
batch.add(s1)
batch.add(s3)
print("DC Q3 highest:", batch.highest())

import pytest
from pydantic import BaseModel, Field, ValidationError, model_validator

# --- Pydantic ---

# Q1
class Reading(BaseModel):
    """One weather observation from a station."""

    station_id: str = Field(min_length=3)
    timestamp: str
    temperature_c: float = Field(ge=-90, le=60)
    humidity: float = Field(ge=0, le=100)

    # Q4: cross-field check (added for Q4)
    @model_validator(mode="after")
    def reject_failed_sensor(self):
        if self.humidity == 0.0 and self.temperature_c < -40:
            raise ValueError(
                "humidity of 0.0 with temperature below -40 suggests a failed sensor"
            )
        return self


good = Reading(
    station_id="CLT01",
    timestamp="2026-10-06T12:00:00",
    temperature_c=21.5,
    humidity=55.0,
)
print("PD Q1:", good)


# Q2
# Failure 1: missing required field (no timestamp)
try:
    Reading(station_id="CLT01", temperature_c=21.5, humidity=55.0)
except ValidationError as e:
    print("PD Q2 missing field:", e)

# Failure 2: temperature out of range
try:
    Reading(
        station_id="CLT01",
        timestamp="2026-10-06T12:00:00",
        temperature_c=150.0,
        humidity=55.0,
    )
except ValidationError as e:
    print("PD Q2 temperature 150:", e)

# Failure 3: humidity that is not a number
try:
    Reading(
        station_id="CLT01",
        timestamp="2026-10-06T12:00:00",
        temperature_c=21.5,
        humidity="very humid",
    )
except ValidationError as e:
    print("PD Q2 humidity text:", e)

# Coercion: a numeric string and an int are accepted
coerced = Reading(
    station_id="CLT01",
    timestamp="2026-10-06T12:00:00",
    temperature_c="21.5",
    humidity=40,
)
print("PD Q2 coerced:", coerced)
print("PD Q2 types:", type(coerced.temperature_c), type(coerced.humidity))

# Pydantic accepts "21.5" because, by default, it tries to convert the input
# to the declared type, and the text "21.5" can be read as the float 21.5.
# It rejects "very humid" because no sensible number can be made from that
# text, so the conversion fails. The rule: Pydantic will convert a value if
# the value clearly represents the target type, and refuses if it doesn't.


# Q3
try:
    Reading(station_id="ab", temperature_c="hot", humidity=50.0)
except ValidationError as e:
    errors = e.errors()
    for err in errors:
        print("PD Q3:", err["loc"], err["msg"])
    print("PD Q3 error count:", len(errors))

# Three errors were reported: station_id too short, timestamp missing, and
# temperature_c not a number. Reporting all of them at once is more useful
# than stopping at the first because you can fix every problem in one pass.
# If it stopped at the first error you would fix one, rerun, hit the next,
# and repeat, and with a big batch of data you would never see the full
# picture of what is wrong.


# Q4
valid_reading = Reading(
    station_id="CLT01",
    timestamp="2026-10-06T12:00:00",
    temperature_c=-50.0,
    humidity=20.0,
)
print("PD Q4 valid:", valid_reading)

try:
    Reading(
        station_id="CLT01",
        timestamp="2026-10-06T12:00:00",
        temperature_c=-50.0,
        humidity=0.0,
    )
except ValidationError as e:
    print("PD Q4 failed sensor:", e)

# Field constraints (min_length, ge, le) look at one field at a time, with no
# access to the other fields. This rule depends on two fields together:
# humidity being 0.0 AND temperature being below -40. Each value is fine on
# its own (0.0 humidity is allowed, -50 temperature is allowed), so only a
# model_validator, which runs after all fields are set and sees the whole
# object, can express it.


# --- pytest ---

# Q1
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert a temperature from Celsius to Fahrenheit."""
    return celsius * 9 / 4 + 32


def test_celsius_to_fahrenheit():
    assert celsius_to_fahrenheit(0) == 32
    assert celsius_to_fahrenheit(100) == 212
    assert celsius_to_fahrenheit(37) == pytest.approx(98.6)

# pytest.approx was necessary because floats can't be stored exactly in
# binary. 37 * 9 / 5 + 32 comes out as 98.60000000000001, not exactly 98.6,
# so a plain == fails. approx compares with a small tolerance instead.


# Q2
def mean(values: list[float]) -> float:
    """Return the average of a list of numbers.

    Raises ValueError if the list is empty.
    """
    if not values:
        raise ValueError("cannot compute the mean of an empty list")
    return sum(values) / len(values)


def test_mean_of_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean([])

# pytest.raises(ValueError) alone passes for ANY ValueError, even one raised
# for a completely different reason (a bug somewhere else, a different bad
# input). match= also checks the message, so the test only passes if the
# error is the one we meant, about an empty list.


# Q3
@pytest.mark.parametrize(
    "values, expected",
    [
        ([1, 2, 3], 2.0),
        ([5], 5.0),
        ([-1, -2, -3], -2.0),
        ([-5, 5], 0.0),
        ([1.5, 2.5], 2.0),
    ],
)
def test_mean_values(values, expected):
    assert mean(values) == pytest.approx(expected)

# Summary line from `pytest warmup_01.py -v`:
# ===== 7 passed in 6.27s =====
#
# One parametrized test is better than several near-identical functions
# because the test logic is written once, so there is less code to maintain
# and fix, adding a new case is one line, and pytest still reports each case
# separately so you can see exactly which input failed.


# Q4
# Deliberate break: change 9 / 5 to 9 / 4 in celsius_to_fahrenheit, run
# `pytest warmup_01.py -v`, and paste the failure output below. Then change
# it back to 9 / 5.
#
# ===== 7 passed in 0.93s  =====
#
# pytest showed the actual computed value next to the expected value (and the
# line of code that failed). That is more useful than a bare "assertion
# failed" because it tells you which assert broke and what the function
# really returned, so you can see how far off it is and start debugging
# immediately instead of adding print statements and rerunning.