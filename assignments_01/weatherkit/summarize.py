from collections import defaultdict
from dataclasses import dataclass

from .records import HourlyReading


@dataclass
class DailySummary:
    """Summary of one calendar day of hourly weather readings.

    Attributes:
        date: The calendar date as "YYYY-MM-DD".
        temp_max: Highest hourly temperature of the day, in degrees Celsius.
        temp_min: Lowest hourly temperature of the day, in degrees Celsius.
        precipitation_sum: Total precipitation for the day, in millimetres.
        hours_observed: How many hourly readings went into this summary.
    """

    date: str
    temp_max: float
    temp_min: float
    precipitation_sum: float
    hours_observed: int

    def temp_range(self) -> float:
        """Return the difference between the day's max and min temperature."""
        return self.temp_max - self.temp_min


class DailyAggregator:
    """Groups hourly readings into daily summaries.

    Days with fewer than min_hours observations are not reported, because a
    maximum or minimum from a partial day can't be trusted.
    """

    def __init__(self, min_hours: int = 24) -> None:
        """Create an aggregator.

        Args:
            min_hours: Minimum number of hourly observations a day needs
                before it is reported.
        """
        self.min_hours = min_hours

    def _group_by_date(
        self, readings: list[HourlyReading]
    ) -> dict[str, list[HourlyReading]]:
        """Group readings by calendar date (first 10 characters of timestamp)."""
        groups: dict[str, list[HourlyReading]] = defaultdict(list)
        for reading in readings:
            groups[reading.timestamp[:10]].append(reading)
        return groups

    def summarize(self, readings: list[HourlyReading]) -> list[DailySummary]:
        """Build one summary per complete day, sorted by date.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            A DailySummary for each date with at least min_hours readings,
            sorted by date. Days with fewer readings are left out; use
            incomplete_days() to see which ones.
        """
        summaries: list[DailySummary] = []
        for date, day in self._group_by_date(readings).items():
            if len(day) < self.min_hours:
                continue
            temps = [r.temperature_c for r in day]
            summaries.append(
                DailySummary(
                    date=date,
                    temp_max=max(temps),
                    temp_min=min(temps),
                    # round() removes float noise like 2.4000000000000004
                    precipitation_sum=round(
                        sum(r.precipitation_mm for r in day), 2
                    ),
                    hours_observed=len(day),
                )
            )
        return sorted(summaries, key=lambda s: s.date)

    def incomplete_days(self, readings: list[HourlyReading]) -> list[str]:
        """List the dates that summarize() would drop.

        Args:
            readings: Hourly readings, in any order.

        Returns:
            The dates (sorted) that have fewer than min_hours readings.
        """
        return sorted(
            date
            for date, day in self._group_by_date(readings).items()
            if len(day) < self.min_hours
        )