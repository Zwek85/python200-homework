import pytest

from weatherkit import DailyAggregator, HourlyReading


@pytest.fixture
def readings() -> list[HourlyReading]:
    """Hand-built readings: two days of 3 hours, plus one day with 1 hour."""
    return [
        HourlyReading("2026-04-08T00:00", 10.0, 0.1),
        HourlyReading("2026-04-08T01:00", 14.0, 0.2),
        HourlyReading("2026-04-08T02:00", 12.0, 0.3),
        HourlyReading("2026-04-09T00:00", 5.0, 0.0),
        HourlyReading("2026-04-09T01:00", 8.0, 1.5),
        HourlyReading("2026-04-09T02:00", 6.0, 0.0),
        HourlyReading("2026-04-10T00:00", 20.0, 0.0),  # partial day
    ]


def test_groups_readings_into_two_days(readings):
    summaries = DailyAggregator(min_hours=3).summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09"]


def test_temp_max_and_min_correct(readings):
    summaries = DailyAggregator(min_hours=3).summarize(readings)
    assert summaries[0].temp_max == 14.0
    assert summaries[0].temp_min == 10.0
    assert summaries[1].temp_max == 8.0
    assert summaries[1].temp_min == 5.0


@pytest.mark.parametrize(
    "day_index, expected_precip",
    [
        (0, 0.6),
        (1, 1.5),
    ],
)
def test_precipitation_sum(readings, day_index, expected_precip):
    summaries = DailyAggregator(min_hours=3).summarize(readings)
    assert summaries[day_index].precipitation_sum == pytest.approx(expected_precip)


def test_short_day_dropped_and_reported(readings):
    aggregator = DailyAggregator(min_hours=3)
    dates = [s.date for s in aggregator.summarize(readings)]
    assert "2026-04-10" not in dates
    assert aggregator.incomplete_days(readings) == ["2026-04-10"]


def test_lower_min_hours_keeps_the_short_day(readings):
    summaries = DailyAggregator(min_hours=1).summarize(readings)
    assert [s.date for s in summaries] == ["2026-04-08", "2026-04-09", "2026-04-10"]
    assert summaries[2].hours_observed == 1


def test_summaries_sorted_by_date(readings):
    summaries = DailyAggregator(min_hours=3).summarize(list(reversed(readings)))
    dates = [s.date for s in summaries]
    assert dates == sorted(dates)


# Break-it check: I changed max(temps) to min(temps) in summarize.py.
# The test that failed was: ===== test_temp_max_and_min_correct  =====
# Then I changed it back and all tests passed again.