from pydantic import BaseModel, Field, model_validator


class HourlyBlock(BaseModel):
    """The columnar hourly data from the API: three parallel lists.

    Hour i is made by taking index i from each list, so all three lists
    must be the same length.
    """

    time: list[str]
    temperature_2m: list[float]
    precipitation: list[float]

    @model_validator(mode="after")
    def check_equal_lengths(self):
        n_time = len(self.time)
        n_temp = len(self.temperature_2m)
        n_precip = len(self.precipitation)
        if not (n_time == n_temp == n_precip):
            raise ValueError(
                "hourly lists must be the same length, got "
                f"time={n_time}, temperature_2m={n_temp}, "
                f"precipitation={n_precip}"
            )
        return self


class WeatherResponse(BaseModel):
    """The top level of the Open-Meteo hourly response for one location."""

    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation: float
    hourly: HourlyBlock