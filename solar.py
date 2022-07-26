"""
solar.py
~~~~~~~~~~~~~
This module introduces a class for solar calculations aiming at calculating the solar zenith angle (SZA).
"""

import numpy as np
import pandas as pd
from timezonefinder import TimezoneFinder
from nptyping import NDArray, Shape
from typing import Any

# todo: make functions work with scalars where possible (e.g. solar dec)
class Location:
    """Class Location to compute earth-sun relationships at a location on a given date."""
    lat: float
    lon: float
    date: NDArray
    is_scalar: bool
    local_date: NDArray
    offset: float
    lt: NDArray
    lstm: float

    # approximate correction for atmospheric refraction at sunrise and sunse
    zenith_constant = 90.833

    def __init__(self, lat, lon, date):
        """
        :param lat: Latitude [degrees] of location, positive for North, negative for South
        :param lon: Longitude [degrees] of location, negative for East, positive for West
        :param date: naive timestamp for a date
        """

        # Latitude, Longitude [deg] and date [pd.Timestamp]
        self.lat = lat
        self.lon = lon
        self.date = np.asarray(date)

        # Flag when single value is passed, convert to array
        self.is_scalar = False
        if self.date.ndim == 0:
            self.date = self.date[None]
            self.is_scalar = True

        # Timezone-aware timestamp, UTC offset, local standard meridian time and local standard time
        self.local_date = self._localize_date()
        self.offset = self._utc_offset()
        self.lt = self._local_time()
        self.lstm = self._local_standard_time_meridian()
        self.lst = None

        # Day angle, equation of time, solar declination, hour angle, solar elevation, SZA, sunrise & sunset
        self.day_angle = None
        self.eot = None
        self.dec = None
        self.hra = None
        self.sa = None
        self.sza = None
        self.sunrise = None
        self.sunset = None

    def compute(self):
        """Public method coupling the calculations to derive Sun Zenith Angle (SZA)"""
        self.calc_day_angle()
        self.solar_declination()
        self.equation_of_time()
        self.local_solar_time()
        self.hour_angle()
        self.solar_elevation()
        self.solar_zenith_angle()
        self.calc_sunrise_sunset()

        # Solar Zenith & hour angle are masked by sunrise and sunset -> NaN at night time
        if not self.is_scalar:
            try:
                self.local_date < self.sunrise
            except TypeError:
                # On dates with time shift replace ambiguous time with NaN
                self.local_date = np.where(pd.isnull(self.local_date), np.nan, self.local_date)
            self.sza = np.where(self.local_date < self.sunrise, np.nan, self.sza)
            self.sza = np.where(self.local_date > self.sunset, np.nan, self.sza)
            self.hra = np.where(self.local_date < self.sunrise, np.nan, self.hra)
            self.hra = np.where(self.local_date > self.sunset, np.nan, self.hra)

    def _localize_date(self) -> NDArray:
        """Creates timezone-aware timestamp from Longitude and Latitude."""
        timezone_string = TimezoneFinder().timezone_at(lng=self.lon, lat=self.lat)
        return np.array([xi.tz_localize(timezone_string, ambiguous="NaT", nonexistent="NaT") for xi in self.date])

    def _utc_offset(self) -> float:
        """Calculates the offset between local time and UTC in hours."""
        return self.local_date[0].utcoffset().total_seconds() / 60 / 60

    def _local_time(self) -> NDArray:
        """Calculates the local time in minutes."""
        return np.array([xi.hour * 60 for xi in self.local_date])

    def _local_standard_time_meridian(self) -> float:
        """Calculates the reference meridian required for angle estimation."""
        return self.offset * 15

    def equation_of_time(self):
        """Calculates the equation of time expressing the relationship between sundial and standard time in minutes."""
        self.eot = (1440.0 / 2 / np.pi) * (
                0.000075
                + 0.001868 * np.cos(self.day_angle)
                - 0.032077 * np.sin(self.day_angle)
                - 0.014615 * np.cos(2.0 * self.day_angle)
                - 0.040849 * np.sin(2.0 * self.day_angle)
        )

    def local_solar_time(self):
        """Calculates the local solar time in minutes."""
        self.lst = (self.lt + 4
                    * (self.lon - self.lstm)
                    + self.eot)

    def calc_day_angle(self):
        """Calculates the day angle in radians."""
        self.day_angle = (2 * np.pi * (
                np.array([xi.dayofyear for xi in self.local_date]) - 1
        )
                          ) / 365

    def solar_declination(self):
        """Calculates the solar declination in radians with an accuracy of 0.25°, based on the Spencer 1971 method."""
        self.dec = (
                0.006918
                - 0.399912 * np.cos(self.day_angle)
                + 0.070257 * np.sin(self.day_angle)
                - 0.006758 * np.cos(2 * self.day_angle)
                + 0.000907 * np.sin(2 * self.day_angle)
                - 0.002697 * np.cos(3 * self.day_angle)
                + 0.001480 * np.sin(3 * self.day_angle)
        )

    def hour_angle(self):
        """Calculates the hour angle in radians."""
        self.hra = 15 * (self.lst / 60 - 12)

    def solar_elevation(self):
        """Calculates the solar elevation in radians."""
        self.sa = np.degrees(
            np.arcsin(
                np.sin(np.radians(self.lat)) * np.sin(self.dec)
                + np.cos(np.radians(self.lat)) * np.cos(self.dec) * np.cos(np.radians(self.hra))
            )
        )

    def solar_zenith_angle(self):
        """Calculates the solar zenith angle (SZA) in degrees."""
        self.sza = np.degrees(np.arccos((np.sin(np.radians(self.sa)))))

    def _solar_noon(self) -> NDArray:
        """Calculates the solar noon as a decimal [0,1] as share of 24 hours."""
        return (720 - 4 * self.lon - self.eot + self.offset * 60) / 1440

    def _sunrise_hour_angle(self) -> NDArray:
        """Calculates the hour angle at sunrise."""
        return np.degrees(
            np.arccos(
                np.cos(np.radians(Location.zenith_constant)) / (np.cos(np.radians(self.lat)) * np.cos(self.dec))
                - np.tan(np.radians(self.lat)) * np.tan(self.dec)
            )
        )

    def calc_sunrise_sunset(self):
        """Calculates sunrise at sunset as pd.Timestamp.

        Note that for e.g. days without night time at high latitudes, sunrise/sunset cannot be calculated. In this case,
        _sunrise_hour_angle() returns NaN or values greater than 1. We set sunrise to 1 minutes before the first time
        step and sunset to 1 minute after the last time step respectively in order to not mask out any night values."""

        solnoon = self._solar_noon()
        sunrise_hour_angle = self._sunrise_hour_angle()
        sunrise_decimal = 24 * (solnoon - sunrise_hour_angle * 4 / 1440)
        sunset_decimal = 24 * (solnoon + sunrise_hour_angle * 4 / 1440)

        if np.isnan(sunrise_decimal[0]) and sunrise_decimal[0] > 1 and sunset_decimal[0] > 24:
            self.sunrise = (self.local_date[0].replace(hour=int(sunrise_decimal[0]),
                                                       minute=int((sunrise_decimal[0] * 60) % 60),
                                                       second=int((sunrise_decimal[0] * 3600) % 60)
                                                       ))

            self.sunset = (self.local_date[0].replace(hour=int(sunset_decimal[0]),
                                                      minute=int((sunset_decimal[0] * 60) % 60),
                                                      second=int((sunset_decimal[0] * 3600) % 60)
                                                      ))

        else:
            # No day/nighttime
            self.sunrise = self.local_date[0] - pd.to_timedelta("1min")
            self.sunset = self.local_date[-1] + pd.to_timedelta("1min")
        """
        for enum, day in enumerate(self.local_date):
            try:
                int(sunrise_decimal[enum])
            except ValueError:
                # Skip times which are ambiguos because of DST
                sunrises.append(np.nan)
                sunsets.append(np.nan)
                continue
            sunrises.append(day.replace(hour=int(sunrise_decimal[enum]),
                                        minute=int((sunrise_decimal[enum] * 60) % 60),
                                        second=int((sunrise_decimal[enum] * 3600) % 60)
                                        ))
            sunsets.append(day.replace(hour=int(sunset_decimal[enum]),
                                       minute=int((sunset_decimal[enum] * 60) % 60),
                                       second=int((sunset_decimal[enum] * 3600) % 60)
                                       ))"""
        # self.sunrise = sunrises
        # self.sunset = sunsets


def hogan_sza_average(lat: float, lon: float, date: pd.Timestamp) -> float:
    """This function calculates an approximation of the daily average cosine sun zenith angle [RAD] based on an
    analytical solution by by Hogan et al. 2015.

    The Cosine of SZA is computed as the average over the time when the sun is above the horizon. The method works for
    any model time step with h_min and h_max expressing the hour angle at timestep t1 and t2. Since we are interested in
    the daily average, t1 refers to sunrise and t2 to sunset respectively. Therefore h is estimated by taking the
    minimum and maximum daily hour angle.

    :param lat: Latitude [deg]
    :param lon: Longitude [deg]
    ;
    Hogan, R. J., & Hirahara, S. (2016). Effect of solar zenith angle specification in models on mean shortwave
            fluxes and stratospheric temperatures. In Geophysical Research Letters (Vol. 43, Issue 1, pp. 482–488).
            American Geophysical Union (AGU). https://doi.org/10.1002/2015gl066868"""

    # The day timestamp is split into hours from 0H - 23H
    hourly_timesteps = pd.date_range(date, date + pd.to_timedelta("23H"), freq="H")
    hourly_timesteps.freq = None
    hourly_timesteps = hourly_timesteps.to_list()

    # Create Location object and compute relationships between site and the sun
    site = Location(lat, lon, hourly_timesteps)
    site.compute()

    # Minium and maximum hour angle for the day
    h_min = np.nanmin(site.hra)
    h_max = np.nanmax(site.hra)

    # Reduce declination to scalar value to prevent returning an array. Declination is constant over the day.
    declination = np.unique(site.dec)[0]

    # Analytical solution for the cosine of daily SZA by integrating Location.solar_zenith_angle() between [t1, t2]
    cos_mean_sza = np.sin(declination) * np.sin(np.radians(lat)) \
                   + (((np.cos(declination) * np.cos(np.radians(lat)))
                       * (np.sin(np.radians(h_max)) - np.sin(np.radians(h_min))))
                      / (np.radians(h_max) - np.radians(h_min)))

    return cos_mean_sza
