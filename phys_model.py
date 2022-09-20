"""
phys_model.py
~~~~~~~~~~~~~
This module contains all physical equations for the Priestley-Taylor and the Penman-Monteith Evaporation models.
Can be run as stand-alone for creating training target data by inverting PT or PM.
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

import solar
import constants
from timezonefinder import TimezoneFinder


def latent_heat_vaporization(ta, conversion_factor=1):
    """Calculates latent heat of vaporization from air temperature (Average 2.45).
        If J kg-1 is wanted, apply conversionf factor of 10**6
    Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p. 641). Kluwer Academic Publishers,
        Dordrecht, Netherlands.
    Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.

    :param ta: Air temperature [°C]
    :param conversion_factor: 1 for MJ kg-1, 10**6 for J kg -1
    :return: Latent heat of vaporization [MJ kg-1]
    """
    return (2.501 - 0.00237 * ta) * conversion_factor


def psychrometric_constant(air_pressure, air_temperature):
    """The ratio of specific heat (Cp) of moist air at constant pressure to latent heat (Lv) of vaporization of water.
       Average about 0.4 gwater/kgair K-1
       FAO Eq. 8 https://www.fao.org/3/x0490e/x0490e07.htm

    :param air_pressure: Atmospheric Pressure [kPa]
    :param air_temperature: Air Temperature [°C]
    :return: Psychrometric constant [kPa °C-1]
    """
    return ((constants.air_specific_heat_capacity * air_pressure)
            / (constants.molecular_water_air_ratio * latent_heat_vaporization(air_temperature)))


def latent_heat_to_evaporation(LE, ta, scale="1D"):
    """Converts latent heat flux (W m-2) to Evaporation (mm).
    :param LE: Latent Heat Flux [W m-2]
    :param ta: Air temperature [°C]
    :param scale: Temporal resolution [1D|1H] for conversion of kg m-2 s-1
    :return: Evaporation [mm]"""

    lam = latent_heat_vaporization(ta, conversion_factor=10**6)
    return LE / lam * pd.to_timedelta(scale).total_seconds()


def evaporation_to_latent_heat(ET, ta, scale="1D"):
    """Converts Evaporation (mm) to latent heat flux (W m-2).
    :param ET: vaporation [mm]
    :param ta: Air temperature [°C]
    :param scale: Temporal resolution [1D|1H] for conversion of Mj m-2 d-1
    :return: Latent Heat Flux [W m-2]"""

    lam = latent_heat_vaporization(ta, conversion_factor=10**6)
    return ET * lam / pd.to_timedelta(scale).total_seconds()


def aerodynamic_resistance(u, h, z):
    """Calculates aerodynamic resistance [s m-1]

    Lin, C., Gentine, P., Huang, Y., Guan, K., Kimm, H., & Zhou, S. (2018). Diel ecosystem conductance response to
        vapor pressure deficit is suboptimal and independent of soil moisture. Agricultural and Forest Meteorology,
        250–251, 24–34.

    :param u: wind speed at height z [m s-1]
    :param h: canopy height [m]
    :param z: height of wind/relative humidty sensor [m]
    :return:ga: Aerodynamic Resistance [s m-1]
    """

    d = (2 / 3) * h
    zm = z
    zh = z
    z0m = 0.1 * h
    z0h = 0.1 * z0m
    ga = (np.log((zm - d) / z0m) * np.log((zh - d) / z0h)) / (constants.karman ** 2 * u)
    return ga


def canopy_available_energy(netrad, LAI, SZA):
    """Computes the available energy in the canopy Based on Beer's Law

    :param netrad: Net radiation [W/m²]
    :param LAI: Leaf Area Index [-]
    :param SZA: Sun Zenith Angle
    :return: Ac: Canopy available energy
    """

    Ac = netrad * (1 - np.exp(-0.5 * LAI / np.cos(SZA)))
    return Ac


def net_radiation_canopy(netrad, LAI, SZA):
    """Calculates the net radiation of the canopy layer by partitioning measured net radiation using exponential
    function for Priestly-Taylor model.

    Anderson, M. (1997). A two-source time-integrated model for estimating surface fluxes using thermal infrared remote
        sensing. Remote Sensing of Environment, 60(2), 195–216.
    Norman, J. M., Kustas, W. P., Prueger, J. H., & Diak, G. R. (2000). Surface flux estimation using radiometric
        temperature: A dual-temperature-difference method to minimize measurement errors.
        In Water Resources Research (Vol. 36, Issue 8, pp. 2263–2274). American Geophysical Union (AGU)."""
    with np.errstate(invalid='ignore'):
        r_nc = netrad * (1 - np.exp(-constants.k * LAI / np.sqrt(2 * np.cos(np.radians(SZA)))))
    return r_nc


def slope_vapour_pressure_curve(ta):
    """Calculates the slope of the relationship between saturation vapour pressure and temperature.
    FAO Eq. 13: https://www.fao.org/3/x0490e/x0490e07.htm

    :param air_temperature: Air Temperature [°C]
    :return: Slope of saturation vapour pressure curve at air temperature [kPa °C-1]"""

    d = (4098 * (0.6108 * np.exp((17.27 * ta) / (ta + 237.3)))) / (ta + 237.3) ** 2
    return d


def pm_standard(gc, p, ta, VPD, netrad, LAI, SZA, u, h, z, ):
    """Computes Transpiration based on Two-Layer Penman-Monteith method.
    Canopy stomatal conductance gc is estimated from stomatal conductance gs by applying the "big-leaf" model.

    Leuning, R.; Zhang, Y.Q.; Rajaud, A.; Cleugh, H.; Tu, K. A simple surface conductance model to estimate regional
        evaporation using MODIS leaf area index and the Penman-Monteith equation. Water Resour. Res. 2008, 44.
    Ding, R., Kang, S., Du, T., Hao, X., & Zhang, Y. (2014). Scaling up stomatal conductance from leaf to canopy using
        a dual-leaf model for estimating crop evapotranspiration. PloS One, 9(4), e95584.

    :param gs: Canopy Conductance
    :param p: Atmospheric Pressure [kPa]
    :param ta: Air temperature [°C]
    :param VPD: Vapor Pressure Deficit
    :param netrad: Net radiation [W/m²]
    :param LAI: Leaf Area Index [-]
    :param SZA: Sun Zenith Angle
    :param u: wind speed at height z [m/s]
    :param h: canopy height [m]
    :param z: height of wind/relative humidty sensor [m]
    :return: T: Transpiration [m/W²]
    """

    ga = aerodynamic_resistance(u, h, z)
    Ac = canopy_available_energy(netrad, LAI, SZA)
    d = slope_vapour_pressure_curve(ta)
    gamma = psychrometric_constant(air_pressure=p, air_temperature=ta)
    cp = constants.air_specific_heat_capacity
    roh = constants.air_density
    T = (d * Ac + roh * cp * VPD * ga) / (d + gamma * (1 + ga / gc))
    return T


def pm_inverted(T, p, ta, VPD, netrad, LAI, SZA, u, h, z):
    """Inverted Penman-Monteith equation to calculate canopy conductance from given Transpiration.

    :param T: Transpiration [W/m²]
    :param p: Atmospheric Pressure [kPa]
    :param ta: Air temperature [°C]
    :param VPD: Vapor Pressure Deficit
    :param netrad: Net radiation [W/m²]
    :param LAI: Leaf Area Index [-]
    :param SZA: Sun Zenith Angle
    :param u: wind speed at height z [m/s]
    :param h: canopy height [m]
    :param z: height of wind/relative humidity sensor [m]
    :return: gc: Canopy conductance
    """

    ga = aerodynamic_resistance(u, h, z)
    Ac = net_radiation_canopy(netrad, LAI, SZA)
    d = slope_vapour_pressure_curve(ta)
    gamma = psychrometric_constant(air_pressure=p, air_temperature=ta)
    cp = constants.air_specific_heat_capacity
    roh = constants.air_density
    gc = ((T * ga * gamma) / (Ac * d + cp * ga * roh * VPD - T * (d + gamma))) * LAI
    return gc


def pt_standard(ta, p, netrad, LAI, SZA, alpha_c=1.26):
    """Priestley-Taylor model for Transpiration. If no PT-coefficient alpha_c is given, 1.26 is used as default value
    following Cammalleri et al. 2012.

    Gan, G., & Liu, Y. (2020). Inferring transpiration from evapotranspiration: A transpiration indicator using
        the Priestley-Taylor coefficient of wet environment. In Ecological Indicators (Vol. 110, p. 105853).
        Elsevier BV.
    Cammalleri, C. et al. (2012). Applications of a remote sensing-based two-source energy balance algorithm for mapping
        surface fluxes without in situ air temperature observations. In Remote Sensing of Environment, 124, pp. 502-515.

    :param: ta: Air temperature [°C]
    :param: p: Atmospheric Pressure [kPa]
    :param: netrad: Net radiation [W m-2]
    :param: LAI: Leaf Area Index [-]
    :param: SZA: Sun Zenith Angle
    :param: alpha_c: Priestly-Taylor coefficient
    :return: T: Transpiration [W m-2]
    """

    d = slope_vapour_pressure_curve(ta)
    gamma = psychrometric_constant(p, ta)
    R_nc = net_radiation_canopy(netrad, LAI, SZA)
    T = alpha_c * (d / (d + gamma)) * R_nc

    return T


def pt_inverted(ta, p, netrad, LAI, SZA, T):
    """Inverted Priestly-Taylor equation to calculate alpha_c from given Transpiration."""
    d = slope_vapour_pressure_curve(ta)
    gamma = psychrometric_constant(air_pressure=p, air_temperature=ta)
    R_nc = net_radiation_canopy(netrad, LAI, SZA)
    alpha_c = (T * (d + gamma)) / (d * R_nc)
    return alpha_c


if __name__ == "__main__":
    target = "transpiration"
    # Get list of site name codes
    sites = pd.read_csv("site_meta.csv", index_col=0)
    for site in list(sites.index):
        try:
            df = pd.read_csv(f"~/Projects/ml-trans/data/sapfluxnet_daily/{site}.csv", index_col=0, parse_dates=True)
        except FileNotFoundError:
            continue

        # Skip empty data frames
        df = df.dropna()
        if len(df) == 0:
            print(f"Data missing for site {site}")
            continue

        # Get latlon coordinates for site
        latitude = sites[sites.index == site]["lat"].item()
        longitude = sites[sites.index == site]["lon"].item()

        # Identify timezone string for the site for date localization in solar.py (e.g. Europe/Berlin)
        timezone_str = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)

        # Apply daily SZA averaging
        day_series = pd.Series(df.index)
        day_series = day_series.apply(lambda day: solar.hogan_sza_average(lat=latitude,
                                                                          lon=longitude,
                                                                          date=day,
                                                                          timezone=timezone_str))
        # Convert from cosine(SZA) [RAD] to SZA [deg]
        zenith_angle = np.degrees(np.arccos(day_series))
        zenith_angle.index = df.index

        # Inversion of Penman-Monteith model to calculate canopy conductance g_c
        try:
            gc = pm_inverted(T=evaporation_to_latent_heat(df[target], df["t2m"]), p=df["sp"], ta=df["t2m"],
                             VPD=df["vpd"], netrad=df["ssr"], LAI=df["LAI"],
                             SZA=zenith_angle, u=df["windspeed"], h=df["height"], z=df["height"])

        except KeyError:
            continue

        # Inversion of Priestley-Taylor model to calculate PT-coefficent alpha
        try:
            alpha = pt_inverted(ta=df["t2m"], p=df["sp"], netrad=df["ssr"], LAI=df["LAI"], SZA=zenith_angle,
                                T=evaporation_to_latent_heat(df[target], df["t2m"]))
        except KeyError:
            continue

        # Name generated data from inverted models and combine with original data frame
        alpha = alpha.rename("alpha")
        gc = gc.rename("gc")
        df = pd.concat([df, gc], axis=1)

        df = pd.concat([df, alpha], axis=1)
        df["alpha"].replace([np.inf, -np.inf], np.nan, inplace=True)
        df["alpha"].loc[df["alpha"] < 0] = np.nan
        df = df[evaporation_to_latent_heat(df[target], df["t2m"]) < df["ssr"]]
        df = df[(evaporation_to_latent_heat(df[target], df["t2m"]) / df["ssr"]) < 1]
        df = df.loc[np.abs(zscore(df.alpha, nan_policy="omit") < 3)]

        df.to_csv(f"/home/hannemam/Projects/ml-trans/data/param/{site}.csv")
