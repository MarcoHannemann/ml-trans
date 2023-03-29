import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
from timezonefinder import TimezoneFinder

import load_model_data
import phys_model
import solar

# With the seed set to 42, you can reproduce the results from the study
np.random.seed(42)

# First we need to load the deep learning model and transformation pipeline
model = tf.keras.models.load_model("models/model/")
with open("models/model/pipeline.pickle", "rb") as pipeline_file:
    pipeline = pickle.load(pipeline_file)
features = ["swvl1", "swvl2", "vpd", "windspeed", "IGBP", "height", "LAI", "FPAR"]
# We define the prediction site and the input features the model was trained for
fluxnet_data = load_model_data.load_external(
    path="data/fluxnet2/", features=features, freq="1D"
)

for sitename, data in fluxnet_data.items():
    pft = data["IGBP"].iloc[0]
    time_index = data.index

    # Reassign IGBP which is lost after mean computation
    data["IGBP"] = pft

    # The auxilliary data is required for the PT equation, the input data is used for the neural network
    aux_data = data[["t2m", "sp", "ssr", "LAI"]]
    inp_data = data[features]

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
    sza = pd.Series(time_index).apply(
        lambda day: solar.hogan_sza_average(
            lat=latitude, lon=longitude, date=day, timezone=timezone_str
        )
    )

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
