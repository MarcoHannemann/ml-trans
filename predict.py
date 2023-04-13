import pickle
from datetime import datetime
from importlib.metadata import version

import numpy as np
import pandas as pd
import tensorflow as tf
from timezonefinder import TimezoneFinder

import load_model_data
import phys_model
import solar


def predict_fluxnet(
    trained_model, transformation_pipeline, features, flux_data: dict, output_fmt: str = "CSV"
):
    df_t = pd.DataFrame(index=pd.date_range("2002-07-04", "2015-12-31"))
    df_alpha = pd.DataFrame(index=pd.date_range("2002-07-04", "2015-12-31"))
    for sitename, data in flux_data.items():
        print(f"Predicting {sitename}")
        pft = data["IGBP"].iloc[0]
        time_index = data.index

        # Reassign IGBP which is lost after mean computation
        data["IGBP"] = pft

        # The auxilliary data is required for the PT equation, the input data is used for the neural network
        aux_data = data[["t2m", "sp", "ssr", "LAI"]]
        inp_data = data[features]

        inp_data = transformation_pipeline.transform(inp_data)

        alpha = pd.Series(trained_model.predict(inp_data).flatten())
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
        transpiration_pt = phys_model.latent_heat_to_evaporation(
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

        transpiration_pt.index = time_index
        df_t = pd.concat([df_t, transpiration_pt.rename(sitename)], axis=1)
        df_alpha = pd.concat([df_alpha, alpha.rename(sitename)], axis=1)
        if output_fmt == "CSV":
            output = pd.DataFrame(data=[alpha, transpiration_pt]).T
            output.columns = ["alpha_c", "transpiration"]
            output.to_csv(f"output/{sitename}.csv")

    if output_fmt == "netCDF":
        write_netcdf(df_t, df_alpha)
    return df_t, df_alpha


def write_netcdf(ml_predictions, ml_coeffs):
    idx_t = pd.MultiIndex.from_product(
        [list(ml_predictions.columns), list(ml_predictions.index)],
        names=["station", "time"],
    )
    col_t = ["transpiration"]
    df_t = pd.DataFrame(np.array(ml_predictions.T).flatten(), idx_t, col_t)
    ds_t = df_t.to_xarray()

    idx_c = pd.MultiIndex.from_product(
        [list(ml_coeffs.columns), list(ml_coeffs.index)], names=["station", "time"]
    )
    col_c = ["PT_coefficient"]
    df_c = pd.DataFrame(np.array(ml_coeffs.T).flatten(), idx_c, col_c)
    ds_c = df_c.to_xarray()
    ds_t["PT_coefficient"] = ds_c["PT_coefficient"]
    ds = ds_t.copy()

    ds["transpiration"] = ds["transpiration"].transpose("time", "station")

    ds["time"].attrs["long_name"] = "time"
    ds["station"].attrs["long_name"] = "FLUXNET Tier 1 site ID"
    ds["station"].attrs["cf_role"] = "timeseries_id"
    ds["transpiration"].attrs["long_name"] = "Evaporation from vegetation transpiration"
    ds["transpiration"].attrs["short_name"] = "Transpiration"
    ds["transpiration"].attrs["standard_name"] = "transpiration_flux"
    ds["transpiration"].attrs["units"] = "kg m-2 d-1"
    ds["transpiration"].attrs["coordinates"] = "lat lon"
    # ds["transpiration"].attrs["_FillValue"] = np.array(-9999.).astype(np.float64)

    ds["PT_coefficient"].attrs["long_name"] = "Priestley-Taylor coefficient"
    ds["PT_coefficient"].attrs["short_name"] = "\N{GREEK SMALL LETTER ALPHA}"
    ds["PT_coefficient"].attrs["units"] = "1"
    # ds["PT_coefficient"].attrs["_FillValue"] = np.array(-9999.).astype(np.float64)
    ds["PT_coefficient"] = ds["PT_coefficient"].transpose("time", "station")
    fluxnet_meta = pd.read_csv("data/FLX-site_info.csv", index_col=0, sep=";")
    lats = fluxnet_meta.loc[ds["station"].values].lat
    lats.index = lats.index.rename("station")
    ds["lat"] = lats
    ds["lat"].attrs["standard_name"] = "latitude"
    ds["lat"].attrs["long_name"] = "station latitude"
    ds["lat"].attrs["units"] = "degrees_north"

    lons = fluxnet_meta.loc[ds["station"].values].lon
    lons.index = lons.index.rename("station")
    ds["lon"] = lons
    ds["lon"].attrs["standard_name"] = "longitude"
    ds["lon"].attrs["long_name"] = "station longitude"
    ds["lon"].attrs["units"] = "degrees_east"

    ds = ds.fillna(-9999.0)
    ds.attrs = {
        "title": "Daily transpiration from SAP-ANN",
        "institution": "Helmholtz-Centre for Environmental Research - UFZ GmbH",
        "author": "Marco Hannemann",
        "email": "marco.hannemann@ufz.de",
        "source": "model",
        "featureType": "timeSeries",
        "references": "Hannemann et al. 2023",
        "Conventions": "CF-1.8",
        "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} with netCDF4 {version('netCDF4')}",
    }

    ds = ds.reindex(
        {index_name: ds.indexes[index_name] for index_name in ["time", "station"]}
    )
    ds.to_netcdf(
        "output/hybrid_T.nc",
        unlimited_dims=["time"],
        encoding={
            "transpiration": {"dtype": np.float64, "_FillValue": -9999.0},
            "PT_coefficient": {"dtype": np.float64, "_FillValue": -9999.0},
            "time": {"dtype": "int32"},
            "lat": {"dtype": np.float32, "_FillValue": None},
            "lon": {"dtype": np.float32, "_FillValue": None},
            "station": {"_FillValue": None, "dtype": "unicode"},
        },
    )


if __name__ == "__main__":
    # With the seed set to 42, you can reproduce the results from the study
    np.random.seed(42)

    # First we need to load the model and data transformation pipeline
    model = tf.keras.models.load_model("models/20230412_092554/model/")
    with open("models/20230412_092554/pipeline.pickle", "rb") as pipeline_file:
        pipeline = pickle.load(pipeline_file)
    input_variables = ["swvl1", "swvl2", "vpd", "windspeed", "IGBP", "height", "LAI", "FPAR"]

    # We define the prediction site and the input features the model was trained for
    fluxnet_data = load_model_data.load_external(
        path="data/fluxnet2/", features=input_variables, freq="1D"
    )
    t, a = predict_fluxnet(model, pipeline, input_variables, fluxnet_data, output_fmt="netCDF")
