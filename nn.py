#!/usr/bin/python3

"""
nn.py
~~~~~
nn.py is the core of ml-trans. It contains the source for the artificial neural network (ANN) and couples the
workflow steps: Loading -> Preprocessing -> Training -> Prediction -> Model evaluation.
"""

import os
import glob
import pickle
from datetime import datetime
import json
from typing import Union

import configparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from timezonefinder import TimezoneFinder

import load_model_data
import phys_model
import solar
import plotting
import metrics


# global to-do list
# todo: Involve flags in SAPFLUXNET data
# todo: involve SWVL2?
#  For example, Zeppel et al. (2008) reported that the transpira-
#  tion of Australian woodland was independent of the water content of
#  the top 80 cm of the soil profile, instead, water uptake has occurred
#  from depths of up to 3 m.


def predictions_to_dataframe(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Helper function that builds data frame out of two arrays.

    :param y_true: numpy array with true data, must be of shape (X, 1)
    :param y_pred: numpy array with predicted data
    :return: pandas data frame without NaN data
    """
    df = pd.DataFrame(
        np.concatenate((y_true, y_pred), axis=1), columns=["y_true", "y_pred"]
    )
    df = df.dropna(axis=0, how="any")
    return df


def calculate_aic(n, mse, n_params):
    aic = n * np.log(mse / n) + 2 * n_params
    return aic


def initialize_model(
        inp_shape: int = 11,
        activation: str = "relu",
        n_layers: int = 2,
        n_neurons: int = 32,
        dropout: Union[bool, float] = False,
) -> tf.keras.Model():
    """Creates a sequential model with tf.keras for regression problems. The parameters should be set in
    config/config.ini.

    :param inp_shape: Input shape of the input feature data. Equal to number of dataframe columns
    :param activation: Type of activation function
    :param n_layers: Number of hidden layers to be generated
    :param n_neurons: Number of neurons per hidden layer
    :param dropout: False or dropout rate
    :return: Compiled model ready to be fitted to training data
    """
    model_instance = tf.keras.Sequential()
    model_instance.add(tf.keras.Input(shape=(inp_shape,)))
    for i in range(0, n_layers):
        model_instance.add(tf.keras.layers.Dense(n_neurons, activation=activation))
        if dropout:
            model_instance.add(tf.keras.layers.Dropout(dropout))
    model_instance.add(tf.keras.layers.Dense(1))
    model_instance.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model_instance


def predict(
        trained_model: tf.keras.Model(), x: np.ndarray, y: Union[None, np.ndarray]) -> np.ndarray:
    """Uses trained model to make predictions based on input feature data and creates data frame with true values for
    comparison.

    :param trained_model: Compiled tf.keras model
    :param x: Input feature data. Must be of same dimension as compiled model
    :param y: True Y data for comparison
    :return: Data frame with and predicted data (y_pred) and true (y_pred) if target is known
    """
    pred = trained_model.predict(x)
    return predictions_to_dataframe(y, pred) if y is not None else pred


def predict_fluxnet(
        trained_model: tf.keras.Model(), target_var: str = "transpiration", freq: str = "1D") -> None:
    """Uses trained model to predict T at FLUXNET sites.

    :param trained_model: Compiled tf.keras model
    :param target_var: Target the model was trained on [ transpiration || gc || alpha ]
    :param freq: Temporal resolution of model [ 1D || 1H ]
    """

    # create empty dataframe for predictions period
    idx = pd.date_range("2002-07-04", "2015-12-31 22:00:00", freq=freq)
    predictions_all_stations = pd.DataFrame(index=idx)
    alpha_all_stations = pd.DataFrame(index=idx)

    # read transformed input data at FLUXNET sites
    files = sorted(glob.glob("output/fluxnet/*csv"))
    sitenames = [os.path.basename(file)[:-4] for file in files]
    for sitename, file in zip(sitenames, files):
        print(f"Predicting {sitename}...")
        # open transformed input data
        arr = np.array(pd.read_csv(file))

        # following try/except block for test cases only
        try:
            result = predict(trained_model, arr, y=None)
        except ValueError:
            print(f"IGBP not valid for {sitename}")
            continue
        series = pd.Series(result.flatten())

        try:
            series.index = idx
        except ValueError:
            print(f"Warning! File {file} cannot be reindexed, "
                  f"only has {len(series)} elements, new index has {len(idx)}!")
            continue

        # If target is canopy conductance, apply Penman-Monteith equation on predictions
        if target_var == "gc":
            df = pd.read_csv(f"data/fluxnet_hourly/{sitename}.csv", index_col=0, parse_dates=True)
            df = df["2002-07-04":"2015-12-31 22:00:00"].resample(freq).mean()
            gc = series.copy()
            gc.index = df.index
            series = phys_model.latent_heat_to_evaporation(
                phys_model.pm_standard(
                    gc=gc,
                    p=df["sp"],
                    ta=df["t2m"],
                    VPD=df["vpd"],
                    netrad=df["ssr"],
                    LAI=df["LAI"],
                    SZA=0,
                    u=df["height"],
                    h=df["height"],
                    z=df["height"],
                ),
                df["t2m"],
            )
            series.index = idx
            predictions_all_stations = pd.concat(
                [predictions_all_stations, series.rename(os.path.basename(file)[:-4])],
                axis=1,
            )
        # If target is alpha, apply Priestley-Taylor equation
        elif target_var == "alpha":

            # read hourly input data for FLUXNET sites and resample to set frequency
            df = pd.read_csv(f"{ext_path}{sitename}.csv", index_col=0, parse_dates=True)
            df = df["2002-07-04":"2015-12-31 22:00:00"].resample(freq).mean()

            # Get latitude and longitude coordinates for site for calculation of sun zenith angle (SZA)
            fluxnet_meta = pd.read_csv("FLX-site_info.csv", index_col=0, sep=";")
            latitude = fluxnet_meta[fluxnet_meta.index == sitename]["lat"].item()
            longitude = fluxnet_meta[fluxnet_meta.index == sitename]["lon"].item()

            # Identify timezone string for the site for date localization in solar.py (e.g. Europe/Berlin)
            timezone_str = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)

            # Apply daily SZA averaging
            day_series = pd.Series(df.index)
            day_series = day_series.apply(lambda day: solar.hogan_sza_average(lat=latitude,
                                                                              lon=longitude,
                                                                              date=day,
                                                                              timezone=timezone_str))
            # Convert from cosine(SZA) [RAD] to SZA [deg]
            sza = np.degrees(np.arccos(day_series))
            sza.index = df.index

            # Apply PT model on predicted alpha coefficients
            alpha = series.copy()
            alpha.index = df.index
            series = phys_model.latent_heat_to_evaporation(
                phys_model.pt_standard(
                    ta=df["t2m"],
                    p=df["sp"],
                    netrad=df["ssr"],
                    LAI=df["LAI"],
                    SZA=sza,
                    alpha_c=alpha,
                ),
                ta=df["t2m"],
                scale=freq,
            )
            series.index = idx

            # Calculate available energy (r_nc) in the canopy and set T = 0 if r_nc < 0
            r_nc = phys_model.net_radiation_canopy(netrad=df["ssr"], LAI=df["LAI"], SZA=sza)
            series.loc[(r_nc < 0)] = 0
            alpha.loc[(r_nc < 0)] = np.nan

            # Add site prediction to dataframe
            predictions_all_stations = pd.concat(
                [predictions_all_stations, series.rename(os.path.basename(file)[:-4])],
                axis=1, )

        # Append PT-coefficient prediction to data frame
        alpha_all_stations = pd.concat([alpha_all_stations, alpha.rename(os.path.basename(file)[:-4])],
                                       axis=1)
        series.plot(lw=0.3)
        plt.savefig(f"output/fluxnet_predictions/fig/{os.path.basename(file)[:-4]}")
        plt.clf()

    # Write out CSV with all FLUXNET predictions (Transpiration & coefficients)
    predictions_all_stations.to_csv(f"output/fluxnet_predictions/{model_time}-flx_predictions_{target_var}.csv")
    alpha_all_stations.to_csv(f"output/fluxnet_predictions/{model_time}-flx_coefficients_{target_var}.csv")


if __name__ == "__main__":
    # Get current timestamp of model execution and create folder for model outputs
    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.mkdir(f"models/{model_time}/")
    except FileExistsError:
        pass

    # read configuration file
    cp = configparser.ConfigParser(delimiters="=", converters={"list": lambda x: [i.strip() for i in x.split(",")]})
    cp.read("config/config.ini")

    # paths
    inp_path = cp["PATHS"]["training_data"]
    try:
        blacklist = cp.getboolean("PATHS", "blacklist")
    except ValueError:
        blacklist = cp["PATHS"]["blacklist"]
    ext_path = cp["PATHS"]["prediction_data"]

    # training settings
    retrain = cp.getboolean("TRAINING", "retrain")
    features = cp.getlist("TRAINING", "features")
    target = cp["TRAINING"]["target"]
    frequency = cp["TRAINING"]["frequency"]

    # model architecture
    layers = cp.getint("MODEL.ARCHITECTURE", "n_layers")
    neurons = cp.getint("MODEL.ARCHITECTURE", "n_neurons")
    try:
        dropout_rate = cp.getfloat("MODEL.ARCHITECTURE", "dropout_rate")
    except ValueError:
        dropout_rate = cp.getboolean("MODEL.ARCHITECTURE", "dropout_rate")
        pass
    early_stopping_epochs = cp.getint("MODEL.ARCHITECTURE", "early_stopping_epochs")
    act_fn = cp["MODEL.ARCHITECTURE"]["activation_fn"]

    # misc configs
    external_prediction = cp.getboolean("OTHER", "predict_fluxnet")

    # Retrain model if retrain is enabled
    if retrain:
        # load model data and create sequential model
        train_data, metadata = load_model_data.load(
            path_csv=inp_path,
            freq=frequency,
            features=features,
            timestamp=model_time,
            blacklist=blacklist,
            target=target,
            external_prediction=ext_path,
        )

        # Create sequential model from settings
        input_shape = train_data["Xtrain"].shape[1]
        model = initialize_model(
            inp_shape=input_shape,
            activation=act_fn,
            n_layers=layers,
            n_neurons=neurons,
            dropout=dropout_rate, )

        # Callbacks
        # Early Stopping if validation loss doesn't change within specified number of epochs
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_epochs,
            verbose=1,
#            min_delta=0.01
        )

        # Store model training checkpoints

        checkpoint_path = f"checkpoint/{model_time}/cp.ckpt"
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1)

        # train model
        model.fit(
            train_data["Xtrain"],
            train_data["Ytrain"],
            epochs=5000,
            batch_size=1000,
            callbacks=[es_callback, cp_callback],
            validation_data=(train_data["Xtest"], train_data["Ytest"]), )
        model_history = model.history
        # aic
        n_params = sum(
            tf.keras.backend.count_params(x) for x in model.trainable_weights)
        loss = model.history.history["loss"][-1]
        n = len(train_data["Ytrain"])
        aic = calculate_aic(n=n, mse=loss, n_params=n_params)
        print(aic)

        # Save trained model to disk
        model.save(f"models/{model_time}/model")

        # apply trained model on training data
        df_train = predict(model, train_data["Xtrain"], train_data["Ytrain"])
        df_test = predict(model, train_data["Xtest"], train_data["Ytest"])
        df_val = predict(model, train_data["Xval"], train_data["Yval"])

        try:
            os.mkdir(f"models/{model_time}")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"models/{model_time}/plots")
        except FileExistsError:
            pass

        # visualize model training results
        # Scatter density plot. Density should be disabled for hourly resolution, KDE needs too much computation power

        # axes scale, depends on target
        upper_lim = 20  # ** (math.ceil(math.log(train_data["Ytrain"].max(), 10)))
        plotting.scatter_density_plot(
            df_train,
            df_test,
            df_val,
            title=f"Target: {target}, {layers} Layers, {neurons} Neurons, "
                  f"Dropout: {dropout_rate}",
            time=model_time,
            density=True,
            upper_lim=upper_lim,

        )

        # training evolution plot
        plotting.plot_learning_curves(model_history, time=model_time)

        # set model metadata
        metadata["model"]["layers"] = layers
        metadata["model"]["neurons"] = neurons
        metadata["model"]["activation"] = act_fn
        metadata["model"]["dropout"] = dropout_rate
        metadata["model"]["early_stopping"] = early_stopping_epochs

        # calculate performance metrics for prediction on training data
        _, m1, b1 = metrics.linear_fit(
            df_train["y_true"], df_train["y_pred"], upper_lim=upper_lim)
        metadata["results"]["training"] = {
            "MAE": metrics.mae(df_train["y_true"], df_train["y_pred"]),
            "corr": metrics.r2(df_train["y_true"], df_train["y_pred"]),
            "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'", }

        _, m2, b12 = metrics.linear_fit(
            df_test["y_true"], df_test["y_pred"], upper_lim=upper_lim)
        metadata["results"]["testing"] = {
            "MAE": metrics.mae(df_test["y_true"], df_test["y_pred"]),
            "corr": metrics.r2(df_test["y_true"], df_test["y_pred"]),
            "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'", }

        _, m3, b3 = metrics.linear_fit(
            df_val["y_true"], df_val["y_pred"], upper_lim=upper_lim)
        metadata["results"]["validation"] = {
            "MAE": metrics.mae(df_val["y_true"], df_val["y_pred"]),
            "corr": metrics.r2(df_val["y_true"], df_val["y_pred"]),
            "fit": f"y = {round(m3, 2)}x + {round(b3, 2)}'", }

        metadata["results"]["cpk_path"] = f"checkpoint/{model_time}/"

        # write metadata to JSON
        with open(f"models/{model_time}/metadata.json", "w") as fp:
            json.dump(metadata, fp, indent=1)
    else:
        # load pretrained model
        model = tf.keras.models.load_model(f'{cp["PATHS"]["saved_model"]}model/')
        with open(f'{cp["PATHS"]["saved_model"]}pipeline.pickle', "rb") as pipeline_file:
            pipeline = pickle.load(pipeline_file)
        load_model_data.external_transform(features=features,
                                           ext_prediction=ext_path,
                                           freq=frequency,
                                           full_pipeline=pipeline)
    # Use model to predict T at FLUXNET sites
    if external_prediction:
        predict_fluxnet(model, target_var=target, freq=frequency)

