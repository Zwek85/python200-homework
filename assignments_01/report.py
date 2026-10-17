import json
from pathlib import Path

from weatherkit import DailyAggregator, DailySummary, WeatherResponse, to_readings

DATA_PATH = Path(__file__).parent / "weather_raw.json"


def load_response(path: Path = DATA_PATH) -> WeatherResponse:
    """Load a weather JSON file and validate it into a WeatherResponse."""
    with open(path) as f:
        raw = json.load(f)
    return WeatherResponse.model_validate(raw)


def format_table(summaries: list[DailySummary]) -> str:
    """Return the daily summaries as a readable text table."""
    header = f"{'Date':<12}{'High (C)':>10}{'Low (C)':>10}{'Precip (mm)':>13}{'Range (C)':>11}"
    lines = [header, "-" * len(header)]
    for s in summaries:
        lines.append(
            f"{s.date:<12}{s.temp_max:>10.1f}{s.temp_min:>10.1f}"
            f"{s.precipitation_sum:>13.1f}{s.temp_range():>11.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    """Run the full pipeline and print the daily report."""
    response = load_response()
    readings = to_readings(response)

    aggregator = DailyAggregator()
    summaries = aggregator.summarize(readings)
    dropped = aggregator.incomplete_days(readings)

    print(f"Daily weather summary ({response.timezone})")
    print(format_table(summaries))

    if dropped:
        print(f"\nWARNING: dropped incomplete days: {', '.join(dropped)}")
    else:
        print("\nNo incomplete days were dropped.")


# Without this guard, main() would run at import time. If someone imported
# report.py just to reuse a helper such as load_response() or format_table(),
# Python would execute the whole module, so the file would load, validate,
# and print the full report as a side effect of the import. The guard makes
# main() run only when the file is run directly (python3 report.py), where
# __name__ is "__main__", and not when it is imported, where __name__ is
# "report".
if __name__ == "__main__":
    main()

# --- Reflection ---
#
# 1. I think rejecting the whole file is right in some cases. If the
# results are going to be saved somewhere important, like a database, then
# one missing temperature could make a wrong daily high or low and nobody
# would notice. It is better to get an error so we know there is a problem.
# But if I was just showing the data on a dashboard, I would rather keep
# going. One missing hour out of 168 should not stop the other 167 hours
# from being used. To allow this, I would change the schema so the lists
# can have None in them, like list[float | None]. I would also have to
# change the aggregator so it skips the None values, because max() and
# sum() would crash if they hit a None.
#
# 2. If the pipeline runs at noon, only about half of the day has happened,
# so the high, low, and rain total would only be for 12 hours. The numbers
# would look normal but they would be wrong, because the real high might
# come later in the afternoon. With min_hours=24 that day gets left out.
# incomplete_days() then gives me the date that was dropped, so I know what
# happened and can run it again later when the day is finished, instead of
# the day just disappearing.
#
# 3. Since weatherkit is a package, I can import it in other scripts with one
# line, like from weatherkit import DailyAggregator. In Week 10 I would not
# have to copy and paste the code into the new pipeline. I would use the
# same code that already has tests, and if I fix a bug in the package it is
# fixed everywhere that uses it.