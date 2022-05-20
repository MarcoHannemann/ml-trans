import numpy as np
import pandas as pd

import constants

# todo: Scaling factor for LAI still implemented in Ac calculation!


def to_radiant(lat):
    """converts latitude in decimal degrees to radiants."""
    return np.pi / 180 * lat


def get_solar_declination(day_of_year):
    """Calculates the solar declination based on the day of year."""
    return 0.0409 * np.sin((2 * np.pi / 365) * day_of_year - 1.39)


def get_hour_angle(time):
    """Calculates the sun hour angle based on day time."""
    hourangle = pd.Series(data=time.hour, index=time)*15
    hourangle = hourangle.where(hourangle >= 0, hourangle+360)
    hourangle = hourangle.where(hourangle < 360, hourangle-360)
    return hourangle


def get_zenith_angle(solardeclination, hourangle, latitude):
    """Calculates the Solar Zenith Angle SZA.

    Bonan, Gordon: Ecological Climatology: Concepts and Applicotions. p. 61 eq. 4.1.
        Cambridge: Cambidge University Press, 2015."""
    return np.cos(np.sin(latitude) * np.sin(solardeclination) +
                  np.cos(latitude) * np.cos(solardeclination) * np.cos(hourangle))


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
    ga = (np.log((zm - d) / z0m) * np.log((zh - d) / z0h)) / constants.karmann ** 2 * u
    return ga


def canopy_available_energy(netrad, LAI, SZA):
    """Computes the available energy in the canopy Based on Beer's Law

    :param netrad: Net radiation [W/m²]
    :param LAI: Leaf Area Index [-]
    :param SZA: Sun Zenith Angle
    :return: Ac: Canopy available energy
    """

    Ac = netrad * ((1 - np.exp(-0.5 * LAI)) / np.cos(SZA))
    return Ac


def slope_vapour_pressure_curve(ta):
    """
    :param ta: Air Temperature [°C]
    :return: Slope of Vapor Pressure Curve
    """

    d = (4098 * (0.6018 * np.exp((17.27 * ta)/(ta+237.3)))) / (ta + 237.3)**2
    return d


def pm_standard(gs, ta, VPD, netrad, LAI, SZA, u, h, z,):
    """Computes Transpiration based on Two-Layer Penman-Monteith method.

    Leuning, R.; Zhang, Y.Q.; Rajaud, A.; Cleugh, H.; Tu, K. A simple surface conductance model to estimate regional
        evaporation using MODIS leaf area index and the Penman-Monteith equation. Water Resour. Res. 2008, 44.

    :param gs: Canopy Conductance
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
    Ac = canopy_available_energy(netrad, LAI*0.1, SZA)
    d = slope_vapour_pressure_curve(ta)
    gamma = constants.psychrometric_constant
    cp = constants.air_specific_heat_capacity
    roh = constants.air_density
    T = (d * Ac + roh * cp * VPD * ga) / (d + gamma * (1 + ga / gs))
    return T


def pm_inversed(T, ta, VPD, netrad, LAI, SZA, u, h, z):
    """Inverted Penman-Monteith equation to calculate canopy conductance from given Transpiration.

    :param T: Transpiration [W/m²]
    :param ta: Air temperature [°C]
    :param VPD: Vapor Pressure Deficit
    :param netrad: Net radiation [W/m²]
    :param LAI: Leaf Area Index [-]
    :param SZA: Sun Zenith Angle (Get from file)
    :param u: wind speed at height z [m/s]
    :param h: canopy height [m]
    :param z: height of wind/relative humidty sensor [m]
    :return: gs: Canopy conductance
    """

    ga = aerodynamic_resistance(u, h, z)
    Ac = canopy_available_energy(netrad, LAI*0.1, SZA)
    d = slope_vapour_pressure_curve(ta)
    gamma = constants.psychrometric_constant
    cp = constants.air_specific_heat_capacity
    roh = constants.air_density
    gs = (d * ga * T * gamma) / (Ac * d + cp * ga * roh * VPD - d * T * gamma)
    return gs