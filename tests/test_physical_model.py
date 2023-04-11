"""
test_physical_model.py
~~~~~~~~~~~~~
This module contains tests for the physical equations.
"""

import numpy as np

import phys_model


def test_latent_heat_of_vaporization():
    assert phys_model.latent_heat_vaporization(ta=25) == 2.44175


def test_psychrometric_constant():
    assert (
        round(
            phys_model.psychrometric_constant(air_pressure=100, air_temperature=25), 6
        )
        == 0.066699
    )


def test_slope_vapour_pressure_curve():
    """Allen R.G. et al. 1998: Crop evapotranspiration - Guidelines for computing crop water requirements -
    FAO Irrigation and drainage paper 56. Annex 2. Meteorological tables. Table 2.4 Slope of vapour pressure curve (D)
    for different temperatures (T)."""
    air_temperature = np.array([1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    expected = np.array([0.047, 0.061, 0.082, 0.110, 0.145, 0.189, 0.243, 0.311])
    result = phys_model.slope_vapour_pressure_curve(air_temperature)
    assert (result.round(3) == expected).all()


def test_latent_heat_to_evaporation():
    assert round(phys_model.latent_heat_to_evaporation(200, 25), 3) == 7.077
    assert round(phys_model.latent_heat_to_evaporation(400, 35), 3) == 14.293


def test_evaporation_to_latent_heat():
    assert round(phys_model.evaporation_to_latent_heat(5, 25), 3) == 141.305
    assert round(phys_model.evaporation_to_latent_heat(15, 35), 3) == 419.8
