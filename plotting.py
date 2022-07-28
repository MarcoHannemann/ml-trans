"""
plotting.py
~~~~~~~~~~~
Implementation of a scatter density plot for model training evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import metrics


def scatter_density_plot(df_train, df_test, df_val, title, density=True, upper_lim=10):
    """Creates a scatter plot with density visualization based on Gaussian KDE for training, testing and validation
    data of the neural network.

    :param df_train: Training data with true and predicted transpiration
    :param df_test: Testing data with true and predicted transpiration
    :param df_val: Validatio ndata with true and predicted transpiration
    :param title: Title of the scatter plot
    :param density: If True, points will be colored based on density estimated by KDE
    :param upper_lim: Upper limit of X/Y axes.
    """

    # scatter density plot using Gaussian KDE
    # https://stackoverflow.com/a/20107592/14234692 Answer by Joe Kington
    if density:
        xy_train = np.vstack([df_train['y_true'], df_train['y_pred']])
        xy_test = np.vstack([df_test['y_true'], df_test['y_pred']])
        xy_val = np.vstack([df_val['y_true'], df_val['y_pred']])
        z_training = gaussian_kde(xy_train)(xy_train)
        z_test = gaussian_kde(xy_test)(xy_test)
        z_val = gaussian_kde(xy_val)(xy_val)
    else:
        z_training = z_test = z_val = None
    fig, ax = plt.subplots(1, 3, figsize=(16, 8), sharex=True, sharey=True)
    cax1 = ax[0].scatter(df_train['y_true'], df_train['y_pred'], c=z_training, s=0.7)
    cax2 = ax[1].scatter(df_test['y_true'], df_test['y_pred'], c=z_test, s=0.7)
    cax3 = ax[2].scatter(df_val['y_true'], df_val['y_pred'], c=z_val, s=0.7)

    ax[0].plot([0, 1], [0, 1], transform=ax[0].transAxes)
    ax[1].plot([0, 1], [0, 1], transform=ax[1].transAxes)
    ax[2].plot([0, 1], [0, 1], transform=ax[2].transAxes)

    ax[0].set_xlabel('T Prediction training (W m-2)')
    ax[0].set_ylabel('T True training (W m-2)')
    ax[1].set_xlabel('T Prediction test (W m-2)')
    ax[1].set_ylabel('T True test (W m-2)')
    ax[2].set_xlabel('T Prediction val (W m-2)')
    ax[2].set_ylabel('T True val (W m-2)')
    ax[0].title.set_text('Training')
    ax[1].title.set_text('Test')
    ax[2].title.set_text('Validation')

    ax[0].set_ylim(0, upper_lim)
    ax[0].set_xlim(0, upper_lim)
    ax[1].set_ylim(0, upper_lim)
    ax[1].set_xlim(0, upper_lim)
    ax[2].set_ylim(0, upper_lim)
    ax[2].set_xlim(0, upper_lim)

    # linear regression
    x = np.linspace(0, upper_lim, 1000)
    y1, m1, b1 = metrics.linear_fit(df_train["y_true"], df_train["y_pred"], upper_lim=upper_lim)
    y2, m2, b2 = metrics.linear_fit(df_test["y_true"], df_test["y_pred"], upper_lim=upper_lim)
    y3, m3, b3 = metrics.linear_fit(df_val["y_true"], df_val["y_pred"], upper_lim=upper_lim)

    ax[0].plot(x, y1, ls='--', color='red')
    ax[1].plot(x, y2, ls='--', color='red')
    ax[2].plot(x, y3, ls='--', color='red')

    ax[0].text(x=upper_lim / 100, y=upper_lim * 0.95, va='top',
               s=f'N = {len(df_train)}'
                 f'\nMAE = {metrics.mae(df_train["y_true"], df_train["y_pred"])}'
                 f'\nR2 = {metrics.r2(df_train["y_true"], df_train["y_pred"])},'
                 f'\ny = {round(m1, 2)}x + {round(b1, 2)}')
    ax[1].text(x=upper_lim / 100, y=upper_lim * 0.95, va='top',
               s=f'N = {len(df_test)}'
                 f'\nMAE = {metrics.mae(df_test["y_true"], df_test["y_pred"])}'
                 f'\nR2 = {metrics.r2(df_test["y_true"], df_test["y_pred"])}'
                 f'\ny = {round(m2, 2)}x + {round(b2, 2)}')
    ax[2].text(x=upper_lim / 100, y=upper_lim * 0.95, va='top',
               s=f'N = {len(df_val)}'
                 f'\nMAE = {metrics.mae(df_val["y_true"], df_val["y_pred"])}'
                 f'\nR2 = {metrics.r2(df_val["y_true"], df_val["y_pred"])}'
                 f'\ny = {round(m3, 2)}x + {round(b3, 2)}')

    x0, x1 = ax[0].get_xlim()
    y0, y1 = ax[0].get_ylim()
    ax[0].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax[1].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    ax[2].set_aspect(abs(x1 - x0) / abs(y1 - y0))
    # colorbar based on probability distribution function
    # fig.colorbar(cax1)
    fig.suptitle(title)
    plt.show()
