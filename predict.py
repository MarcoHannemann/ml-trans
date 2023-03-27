import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from timezonefinder import TimezoneFinder

import solar
import phys_model

# With the seed set to 42, you can reproduce the results from the study
np.random.seed(42)

# First we need to load the deep learning model and transformation pipeline
model = tf.keras.models.load_model("models/model/")
with open("models/model/pipeline.pickle", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)

# We define the prediction site and the input features the model was trained for
sitename = "AU-Wom"
features = ["swvl1", "swvl2", "vpd", "windspeed", "IGBP", "height", "LAI", "FPAR"]


df = pd.read_csv(f"data/fluxnet/{sitename}.csv", index_col=0, parse_dates=True)
pft = df["IGBP"].iloc[0]

df = df.resample("1D").mean()
time_index = df.index


# Reassign IGBP which is lost after mean computation
df["IGBP"] = pft

# The auxilliary data is required for the PT equation, the input data is used for the neural network
aux_data = df[["t2m", "sp", "ssr", "LAI"]]
inp_data = df[features]

inp_data = pipeline.transform(inp_data)


alpha = pd.Series(model.predict(inp_data).flatten())
alpha.index = time_index

# Get latitude and longitude coordinates for site for calculation of sun zenith angle (SZA)
fluxnet_meta = pd.read_csv("data/FLX-site_info.csv", index_col=0, sep=";")
latitude = fluxnet_meta[fluxnet_meta.index == sitename]["lat"].item()
longitude = fluxnet_meta[fluxnet_meta.index == sitename]["lon"].item()

# Identify timezone string for the site for date localization in solar.py (e.g. Europe/Berlin)
timezone_str = TimezoneFinder().timezone_at(lng=longitude, lat=latitude)

# Apply daily SZA averaging
sza = pd.Series(time_index).apply(lambda day: solar.hogan_sza_average(lat=latitude,
                                                           lon=longitude,
                                                           date=day,
                                                           timezone=timezone_str))

# Convert from cosine(SZA) [RAD] to SZA [deg]
sza = np.degrees(np.arccos(sza))
sza.index = time_index

# Apply PT model on predicted alpha coefficients, convert from W m-2 to mm d-1
t = phys_model.latent_heat_to_evaporation(
    phys_model.pt_standard(
        ta=aux_data["t2m"],
        p=aux_data["sp"],
        netrad=aux_data["ssr"],
        LAI=aux_data["LAI"],
        SZA=sza,
        alpha_c=alpha,
    ),
    ta=aux_data["t2m"],
    scale="1D",
)


t.index = time_index
output = pd.DataFrame(data=[alpha, t]).T
output.columns = ["alpha_c", "transpiration"]
output.to_csv(f"output/{sitename}.csv")