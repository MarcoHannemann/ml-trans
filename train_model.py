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
import tensorflow as tf

import load_model_data
import plotting
import metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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

def initialize_model(
        inp_shape: int = 11,
        activation: str = "relu",
        n_layers: int = 2,
        n_neurons: int = 32,
        dropout: Union[bool, float] = False,
        seed: int = 42,
        ) -> tf.keras.Model:
    """Creates a sequential model with tf.keras for regression problems. The parameters should be set in
    config/config.ini.

    :param inp_shape: Input shape of the input feature data. Equal to number of dataframe columns
    :param activation: Type of activation function
    :param n_layers: Number of hidden layers to be generated
    :param n_neurons: Number of neurons per hidden layer
    :param dropout: False or dropout rate
    :param seed: random seed for dropout layers
    :return: Compiled model ready to be fitted to training data
    """
    model_instance = tf.keras.Sequential()
    model_instance.add(tf.keras.Input(shape=(inp_shape,)))
    for _ in range(n_layers):
        model_instance.add(tf.keras.layers.Dense(n_neurons,
                                                 activation=activation,))
        if dropout:
            model_instance.add(tf.keras.layers.Dropout(rate=dropout,
                                                       seed=seed,))
    model_instance.add(tf.keras.layers.Dense(1))
    model_instance.compile(loss="mean_squared_error",
                           optimizer="adam",
                           metrics=["mse"],
                           )
    return model_instance


def predict(
        trained_model: tf.keras.Model(),
        x: np.ndarray,
        y: Union[None, np.ndarray]
        ) -> np.ndarray:
    """Uses trained model to make predictions based on input feature data and creates data frame with true values for
    comparison.

    :param trained_model: Compiled tf.keras model
    :param x: Input feature data. Must be of same dimension as compiled model
    :param y: True Y data for comparison
    :return: Data frame with and predicted data (y_pred) and true (y_pred) if target is known
    """
    pred = trained_model.predict(x)
    return predictions_to_dataframe(y, pred) if y is not None else pred


if __name__ == "__main__":
    np.random.seed(42)
    # Get current timestamp of model execution and create folder for model outputs
    model_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        os.mkdir(f"models/{model_time}/")
    except FileExistsError:
        pass

    # read configuration
    cp = configparser.ConfigParser(delimiters="=",
                                   converters={"list": lambda x: [i.strip() for i in x.split(",")]})
    cp.read("config/config.ini")

    # paths
    inp_path = cp["PATHS"]["training_data"]
    try:
        blacklist = cp.getboolean("PATHS", "blacklist")
    except ValueError:
        blacklist = cp["PATHS"]["blacklist"]

    # training settings
    features = cp.getlist("TRAINING", "features")
    target = cp["TRAINING"]["target"]
    frequency = cp["TRAINING"]["frequency"]

    # model architecture
    layers = cp.getint("MODEL.ARCHITECTURE", "n_layers")
    neurons = cp.getint("MODEL.ARCHITECTURE", "n_neurons")
    seed = cp.getint("MODEL.ARCHITECTURE", "seed")
    try:
        dropout_rate = cp.getfloat("MODEL.ARCHITECTURE", "dropout_rate")
    except ValueError:
        dropout_rate = cp.getboolean("MODEL.ARCHITECTURE", "dropout_rate")
        pass
    early_stopping_epochs = cp.getint("MODEL.ARCHITECTURE", "early_stopping_epochs")
    act_fn = cp["MODEL.ARCHITECTURE"]["activation_fn"]

    # load model data
    train_data, metadata = load_model_data.load(
            path_csv=inp_path,
            freq=frequency,
            features=features,
            timestamp=model_time,
            blacklist=blacklist,
            target=target,
            seed=seed,
        )

    # create sequential model with set configuration
    input_shape = train_data["Xtrain"].shape[1]
    model = initialize_model(
            inp_shape=input_shape,
            activation=act_fn,
            n_layers=layers,
            n_neurons=neurons,
            dropout=dropout_rate,
            seed=seed, )

    # Callbacks
    # Callback: Early Stopping if validation loss doesn't change within specified number of epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_epochs,
            verbose=1,
            min_delta=0.001
        )

    # Callback: Store model training checkpoints
    checkpoint_path = f"models/{model_time}/model/checkpoint/cp.ckpt"

    f"models/{model_time}/model/checkpoint"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1)

    # train model
    model.fit(
            train_data["Xtrain"],
            train_data["Ytrain"],
            epochs=5000,
            batch_size=1000,
            callbacks=[es_callback, cp_callback],
            validation_data=(train_data["Xval"], train_data["Yval"]), )

    # Save trained model to disk
    model.save(f"models/{model_time}/model")
    model_history = model.history
    with open(f"models/{model_time}/model/history.pkl", "wb") as history_file:
            pickle.dump(model_history, history_file)

    # apply trained model on training data
    df_train = predict(model, train_data["Xtrain"], train_data["Ytrain"])
    df_test = predict(model, train_data["Xtest"], train_data["Ytest"])
    df_val = predict(model, train_data["Xval"], train_data["Yval"])

    # visualize model training results
    os.mkdir(f"models/{model_time}/plots")
    # Scatter density plot. Density should be disabled for hourly resolution, KDE needs too much computation power
    # axes scale, depends on target
    upper_lim = 200  # ** (math.ceil(math.log(train_data["Ytrain"].max(), 10)))
    plotting.scatter_density_plot(
            df_train,
            df_test,
            df_val,
            target=target,
            title=f"Target: {target}, {layers} Layers, {neurons} Neurons, "
                  f"Dropout: {dropout_rate}",
            time=model_time,
            density=True,
        )

    # training evolution plot
    plotting.plot_learning_curves(model_history, time=model_time)

    # set metadata: model architecture
    metadata["model"]["layers"] = layers
    metadata["model"]["neurons"] = neurons
    metadata["model"]["activation"] = act_fn
    metadata["model"]["dropout"] = dropout_rate
    metadata["model"]["early_stopping"] = early_stopping_epochs

    # set metadata: performance metrics for prediction on training data
    _, m1, b1 = metrics.linear_fit(
            df_train["y_true"], df_train["y_pred"], upper_lim=upper_lim)
    metadata["results"]["training"] = {
            "MAE": metrics.mae(df_train["y_true"], df_train["y_pred"]),
            "MSE": metrics.mse(df_train["y_true"], df_train["y_pred"]),
            "corr": metrics.r2(df_train["y_true"], df_train["y_pred"]),
            "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'", }

    _, m2, b12 = metrics.linear_fit(
            df_test["y_true"], df_test["y_pred"], upper_lim=upper_lim)
    metadata["results"]["testing"] = {
            "MAE": metrics.mae(df_test["y_true"], df_test["y_pred"]),
            "MSE": metrics.mse(df_test["y_true"], df_test["y_pred"]),
            "corr": metrics.r2(df_test["y_true"], df_test["y_pred"]),
            "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'", }

    _, m3, b3 = metrics.linear_fit(
            df_val["y_true"], df_val["y_pred"], upper_lim=upper_lim)
    metadata["results"]["validation"] = {
            "MAE": metrics.mae(df_val["y_true"], df_val["y_pred"]),
            "MSE": metrics.mse(df_val["y_true"], df_val["y_pred"]),
            "corr": metrics.r2(df_val["y_true"], df_val["y_pred"]),
            "fit": f"y = {round(m3, 2)}x + {round(b3, 2)}'", }

    metadata["results"]["cpk_path"] = checkpoint_path

        # write metadata to JSON
    with open(f"models/{model_time}/metadata.json", "w") as fp:
            json.dump(metadata, fp, indent=1)
