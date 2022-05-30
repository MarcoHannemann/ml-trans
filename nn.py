import os
import glob
from datetime import datetime
import json
import math
from typing import Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import r2_score

import load_model_data
import phys_model
import plotting
import metrics


# global to-do list
# todo: Involve flags in SAPFLUXNET data
# todo: Full extraction of FPAR for SFN and FLX
# todo: Involve more SFN sites e.g. other PFTs
# todo: involve SWVL2?
#  For example, Zeppel et al. (2008) reported that the transpira-
#  tion of Australian woodland was independent of the water content of
#  the top 80 cm of the soil profile, instead, water uptake has occurred
#  from depths of up to 3 m.

def predictions_to_dataframe(y_true, y_pred):
    """Helper function that builds data frame out of two arrays.

    :param y_true: numpy array with true data, must be of shape (X, 1)
    :param y_pred: numpy array with predicted data
    :return: pandas data frame without NaN data
    """
    df = pd.DataFrame(np.concatenate((y_true, y_pred), axis=1), columns=["y_true", "y_pred"])
    df = df.dropna(axis=0, how='any')
    return df


def create_model(inp_shape: int = 11, activation: str = 'relu', n_layers: int = 2, n_neurons: int = 32,
                 dropout: Union[bool, float] = False) -> tf.keras.Model():
    """Creates a sequential model with tf.keras for regression problems.

    :param inp_shape: Input shape of the input feature data. Equal to number of dataframe columns
    :param activation: Type of activation function
    :param n_layers: Number of hidden layers to be generated
    :param n_neurons: Numer of neurons per hidden layer contains
    :param dropout: False or dropout rate
    :return: Compiled model ready to be fitted to training data
    """
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(inp_shape,)))
    for i in range(0, n_layers):
        model.add(tf.keras.layers.Dense(n_neurons, activation=activation))
        if dropout:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model


def predict(model, x, y):
    """Uses model to make predictions based on input feature data and creates data frame with true values for
    comparison.

    :param model: Compiled tf.keras model
    :param x: Input feature data. Must be of same dimension as compiled model
    :param y: True Y data for comparison
    :return: Data frame with and predicted data (y_pred) and true (y_pred) if target is known
    """
    pred = model.predict(x)
    return predictions_to_dataframe(y, pred) if y is not None else pred


def predict_fluxnet(model):
    """Uses trained model to predict T at FLUXNET sites.

    :param model: Compiled tf.keras model
    """
    idx = pd.date_range('2002-07-04', '31-12-2007 23:00:00', freq='1D')
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


# load data and set model options
features = ["t2m", "ssr", "swvl1", "vpd", "windspeed", "IGBP", "height", "LAI", "FPAR"]


layers = 5
neurons = 256
dropout_rate = False
act_fn = "selu"

ext_path = "data/fluxnet_lai"
#ext_path = None

# load model data and create sequential model
train_data, metadata = load_model_data.load(path_csv="data/param/", freq="1D", features=features,
                                            blacklist="whitelist.csv", target="transpiration",
                                            external_prediction=ext_path)
upper_lim = 25 # 10**(math.ceil(math.log(train_data["Ytrain"].max(), 10))) // 5
input_shape = train_data["Xtrain"].shape[1]
model = create_model(inp_shape=input_shape,
                     activation=act_fn,
                     n_layers=layers,
                     n_neurons=neurons,
                     dropout=dropout_rate)

# Callbacks
# Early Stopping if validation loss doesn't change within specified number of epochs
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=200)

# Store model parameters
model_time = datetime.now().strftime("%Y%m%d_%H:%M:%S")
checkpoint_path = f"checkpoint/{model_time}/cp.ckpt"
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# train model
model.fit(train_data["Xtrain"], train_data["Ytrain"], epochs=5000, batch_size=500, callbacks=[es_callback, cp_callback],
          validation_data=(train_data["Xtest"], train_data["Ytest"]))
# todo: load pretrained model
# checkpoint_path = "??"
# model.load_weights(checkpoint_path)
model.save(f"models/{model_time}")

# apply trained model on training data
df_train = predict(model, train_data["Xtrain"], train_data["Ytrain"])
df_test = predict(model, train_data["Xtest"], train_data["Ytest"])
df_val = predict(model, train_data["Xval"], train_data["Yval"])


"""x = train_data["untransformed"]["Xtrain"].reset_index()
T = phys_model.pm_standard(gc=df_train["y_pred"], ta=x["t2m"], VPD=x["vpd"], netrad=x["ssr"], LAI=x["LAI"], SZA=0, u=x["height"], h=x["height"], z=x["height"], )
c = pd.concat([x["transpiration"],
               phys_model.latent_heat_to_evaporation(T, train_data["untransformed"]["Xtrain"]["t2m"].to_numpy())],
              axis=1)
c.columns = ["true", "pred"]
print(r2_score(c["true"], c["pred"]))
c.plot(kind="scatter", x="pred", y="true", xlim=(0, 20), ylim=(0, 20))
plt.show()"""


# visualize model results in a scatter plot for training, testing, validation
plotting.scatter_density_plot(df_train, df_test, df_val,
                              title=f"{layers} Layers, {neurons} Neurons, Dropout: {dropout_rate}",
                              density=True,
                              upper_lim=upper_lim)

# write metadata to JSON
metadata["model"]["layers"] = layers
metadata["model"]["neurons"] = neurons
metadata["model"]["activation"] = act_fn
metadata["model"]["dropout"] = dropout_rate

_, m1, b1 = metrics.linear_fit(df_train["y_true"], df_train["y_pred"], upper_lim=upper_lim)
metadata["results"]["training"] = {"MAE": metrics.mae(df_train["y_true"], df_train["y_pred"]),
                                   "corr": metrics.r2(df_train["y_true"], df_train["y_pred"]),
                                   "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'"}

_, m2, b12 = metrics.linear_fit(df_test["y_true"], df_test["y_pred"], upper_lim=upper_lim)
metadata["results"]["testing"] = {"MAE": metrics.mae(df_test["y_true"], df_test["y_pred"]),
                                  "corr": metrics.r2(df_test["y_true"], df_test["y_pred"]),
                                  "fit": f"y = {round(m1, 2)}x + {round(b1, 2)}'"}

_, m3, b3 = metrics.linear_fit(df_val["y_true"], df_val["y_pred"], upper_lim=upper_lim)
metadata["results"]["validation"] = {"MAE": metrics.mae(df_val["y_true"], df_val["y_pred"]),
                                     "corr": metrics.r2(df_val["y_true"], df_val["y_pred"]),
                                     "fit": f"y = {round(m3, 2)}x + {round(b3, 2)}'"}

metadata["results"]["cpk_path"] = f"checkpoint/{model_time}/"

with open(f"models/{model_time}.json", "w") as fp:
    json.dump(metadata, fp, indent=1)

# Use model to predict T at FLUXNET sites
if ext_path:
    predict_fluxnet(model)
