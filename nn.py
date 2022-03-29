import os
import glob
from pickle import load
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

import load_model_data
import plotting


def predictions_to_dataframe(y_true, y_pred):
    """Helper function that builds data frame out of two arrays.

    :param y_true: numpy array with true data, must be of shape (X, 1)
    :param y_pred: numpy array with predicted data
    :return: pandas data frame without NaN data
    """
    df = pd.DataFrame(np.concatenate((y_true, y_pred), axis=1), columns=["y_true", "y_pred"])
    #df = pd.DataFrame(np.concatenate((y_true, y_pred), axis=1), columns=["y_true", "y_pred"])
    df = df.dropna(axis=0, how='any')
    return df


def load_training_data():
    """Reads train/test/validation data for training the neural network and converts the data to arrays.

    :return: Dictionary with the 3 input feature (X) and 3 prediction target (Y) arrays
    """
    X_train = np.array(pd.read_csv('../sfn_ann/output/ann/train_samples.csv'))
    Y_train = np.array(pd.read_csv('../sfn_ann/output/ann/train_lables.csv'))

    X_test = np.array(pd.read_csv('../sfn_ann/output/ann/test_samples.csv'))
    Y_test = np.array(pd.read_csv('../sfn_ann/output/ann/test_lables.csv'))

    X_val = np.array(pd.read_csv('../sfn_ann/output/ann/val_samples.csv'))
    Y_val = np.array(pd.read_csv('../sfn_ann/output/ann/val_lables.csv'))

    training_data = {"Xtrain": X_train, "Ytrain": Y_train,
                     "Xtest": X_test, "Ytest": Y_test,
                     "Xval": X_val, "Yval": Y_val}
    return training_data


def create_model(inp_shape=11, activation='relu', n_layers=2, n_neurons=32):
    """Creates a sequential model with tf.keras for regression problems.

    :param inp_shape: Input shape of the input feature data. Equal to number of dataframe columns
    :param activation: Type of activation function
    :param n_layers: Number of hidden layers to be generated
    :param n_neurons: Numer of neurons each hidden layer cotains
    :return: Compiled model ready to be fitted to training data
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(inp_shape,)))
    for i in range(0, n_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation=activation))

    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model


def predict(model, x, y):
    """Uses model to make predictions based on input feature data and creates data frame with true values for comparison0

    :param model: Compiled tf.keras model
    :param x: Input feature data. Must be of same dimension as compiled model
    :param y: True Y data for comparison
    :return: Data frame with true data (y_true) and predicted data (y_pred)
    """
    pred = model.predict(x)
    if y is not None:
        # case 1: "True" transpiration data is known (e.g. for Training data)
        return predictions_to_dataframe(y, pred)
    else:
        # case 2: External prediction, target transpiration unknown
        return pred


def predict_fluxnet(model):
    """Uses trained model to predict T at FLUXNET sites.

    :param model: Compiled tf.keras model
    """
    idx = pd.date_range('01-01-2002', '31-12-2007 23:00:00', freq='1D')
    predictions_all_stations = pd.DataFrame(index=idx)

    files = sorted(glob.glob("output/fluxnet/*csv"))
    for file in files:
        print(f"Predicting {os.path.basename(file)[:-4]}...")
        # open transformed input data
        arr = np.array(pd.read_csv(file))
        result = predict(model, arr, y=None)
        series = pd.Series(result.flatten())
        try:
            series.index = idx
        except ValueError:
            print(f"Warning! File {file} cannot be reindexed, "
                  f"only has {len(series)} elements, new index has {len(idx)}!")
            continue
        predictions_all_stations = pd.concat([predictions_all_stations,
                                              series.rename(os.path.basename(file)[:-4])],
                                             axis=1)
        series.plot(lw=0.3)
        plt.savefig(f'output/fluxnet_predictions/fig/{os.path.basename(file)[:-4]}')
        plt.clf()
    predictions_all_stations.to_csv('output/fluxnet_predictions/flx_predictions.csv')

ext_path = "data/fluxnet"
ext_path = None

# load data and set model options
features = ["t2m", "ssr", "swvl1", "vpd", "windspeed", "IGBP", "height"]
train_data = load_model_data.load(path_csv="data/sfn/", freq="1D", features=features,
                                  blacklist=True, external_prediction=ext_path)
n_layers = 5
n_neurons = 128
input_shape = train_data["Xtrain"].shape[1]
model = create_model(inp_shape=input_shape, activation="selu", n_layers=n_layers, n_neurons=n_neurons)

# Callbacks
# Early Stopping if validation loss doesn't change within specified number of epochs
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

# Store model parameters
model_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
checkpoint_path = f"checkpoint/{model_time}/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# retrain model
model.fit(train_data["Xtrain"], train_data["Ytrain"], epochs=10000, batch_size=1000, callbacks=[es_callback, cp_callback],
          validation_data=(train_data["Xtest"], train_data["Ytest"]))
# todo: load pretrained model
# checkpoint_path = "??"
# model.load_weights(checkpoint_path)

# apply trained model on training data
df_train = predict(model, train_data["Xtrain"], train_data["Ytrain"])
df_test = predict(model, train_data["Xtest"], train_data["Ytest"])
df_val = predict(model, train_data["Xval"], train_data["Yval"])

# visualize model results in a scatter plot for training, testing, validation
plotting.scatter_density_plot(df_train, df_test, df_val, title=f"{n_layers} Layers, {n_neurons} Neurons")

# Use model to predict T at FLUXNET sites
if ext_path:
    predict_fluxnet(model)
