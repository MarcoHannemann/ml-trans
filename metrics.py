"""
metrics.py
~~~~~~~~~~~~~~~~~~
This module contains functions for calculating basic metrics that can be wrapped around the model output.
"""

import numpy as np
from scipy.stats import linregress
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mae(y_true, y_pred):
    return round(mean_absolute_error(y_true, y_pred), 2)


def mse(y_true, y_pred):
    return round(mean_squared_error(y_true, y_pred), 2)


def r2(y_true, y_pred):
    return round(r2_score(y_true, y_pred), 2)


def linear_fit(y_true, y_pred, upper_lim):
    m, b, _, _, _ = linregress(y_pred, y_true)
    x = np.linspace(0, upper_lim, 1000)
    y = m * x + b
    return y, m, b
