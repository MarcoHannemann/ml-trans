"""
load_model_data.py
~~~~~~~~~~~~~~~~~~
This module contains the data reading and preprocessing steps. It is imported by nn.py and handles loading, filtering,
transforming and storing input and output data.
"""

import os
import glob
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# todo: filter out negative transpiration and nightime values
# todo: add doy as columns for split data so that the doy is known after scaling
# todo: check sklearn.preprocessing.RobustScaler for outlier detection
# filter out night time values before


def load_tabular(path: str, features: list, target: str, freq: str) -> dict:
    """
    Loads comma seperates value files containing time series for a single location into a data dictionary.

    :param path: point location CSV directory
    :param features: list of input features
    :param target: name of target variable
    :param freq: Temporal resolution. Currently only "1D" supported
    :return: data: data dictionary with structure {basename: pd.DataFrame}
    """

    csv_files = glob.glob(f"{path}*.csv")

    # Drop columns which are not selected as target
    targets = ["transpiration", "gc", "alpha", "con"]
    targets.remove(target)

    # read CSV file for each site and store to dictionary, drop rows containing NaN
    data = {}
    for csv_file in csv_files:
        sitename = os.path.splitext(os.path.basename(csv_file))[0]
        data[sitename] = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        # drop alternative target columns and NaN
        try:
            data[sitename].drop(columns=targets, inplace=True)
        except KeyError:
            pass
        data[sitename] = drop_features(data[sitename], features, [target])
        data[sitename].dropna(how="any", inplace=True)

        # Extract IGBP plant functional type
        try:
            igbp = data[sitename]["IGBP"].iloc[0]
        except IndexError:
            print(f"WARNING: {sitename} contains empty dataframe or is missing variable.")
            del data[sitename]
            continue

        # Filter data below 0
        data[sitename].loc[(data[sitename]['vpd'] < 0)] = np.nan
        data[sitename].loc[(data[sitename][target] < 0)] = np.nan
        if ["ssr"] in features:
            data[sitename].loc[(data[sitename]['ssr'] < 0)] = np.nan

        if "tr_ca" in list(data[sitename].columns):
            data[sitename].drop(columns=["tr_ca"], inplace=True)
        data[sitename].dropna(how="any", inplace=True)
        data[sitename] = data[sitename].resample(freq).mean()
        data[sitename].dropna(how="any", inplace=True)
        data[sitename]["IGBP"] = igbp

    data_new = {}

    # Check for each site if all features are available
    for sitename, df in data.items():
        if isfeature(df, features):
            data_new[sitename] = df
    return data_new


def filter_short_timeseries(data: dict, length=365) -> dict:
    """Removes sites with time series shorter than specified length. Applied to not involve sites which don't catch
    their full seasonal cycle.

    :param data: dictionary containing site dataframes
    :param length: minimum amount of timesteps in time series
    :return: new dictionary with short time series sites removed
    """
    data_filtered = {}
    for site, df in data.items():
        if len(df) > length:
            data_filtered[site] = df
    return data_filtered


def dict_to_df(data: dict) -> pd.DataFrame:
    """Summarizes all sites and convert dict to single dataframe. Temporal spatiotemporal information will be lost.

    :param data: dictionary containing site dataframes
    :return: single concatenated DataFrame
    """
    data = pd.concat(data.values())
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filters out nighttime data and Transpiration lower zero

    :param data: training data
    :return: data: training data
    """
    data = data.loc[data["vpd"] > 0]
    data = data.loc[data["ssr"] > 50]
    #data = data.loc[data["alpha"] > 50]
    return data.reset_index(drop=True)


def isfeature(df, features):
    """Check if specified features are available in data."""
    for feature in features:
        if feature not in df.columns:
            return False
    return True


def drop_features(df: pd.DataFrame, features: list, target: list) -> pd.DataFrame:
    """Drops features to be excluded."""
    return df[features + target]


def split_data(data: pd.DataFrame, target="transpiration", random_state=42) -> tuple:
    """Splits the data into training, testing, validation.


    :param data: DataFrame containing whole dataset
    :param target: name of target variable
    :param random_state: Used to make model reproducable
    :return: tuple containg train, test, val data for input x and target y
    """
    x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=target), data[target],
                                                        train_size=0.7,)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.66, random_state=random_state)

    return x_train, x_test, x_val, y_train, y_test, y_val


def transform_data(x_train: np.array, x_test: np.array, x_val: np.array,
                   y_train: np.array, y_test: np.array, y_val: np.array,
                   features: list, timestamp: str, ext_prediction: str = None, freq: str = "1D") -> dict:
    """Transforms and fits data. Includes normalization and encoding.

    :param ext_prediction: If path is specified, external locations are transformed for predictions
    :param x_train: Training input data
    :param x_test: Testing input data
    :param x_val: Validation input data
    :param y_train: Training target
    :param y_test: Testing target
    :param y_val: Validation target
    :param features: List of feature names incoroporated in model
    :param timestamp: Date and Time of model run
    :param ext_prediction: Path to directory with external sites (CSV)
    :param freq: Temporal resolution 1D | 1H
    :return: dictionary containing transformed training data and untransformed samples
    """

    # divide input features based on scale (interval, nominal)
    cat_attributes = ["IGBP"]
    features.remove("IGBP")
    num_attributes = features

    # Pipeline for scaling and encoding. Scaling is performed after train_test_split to avoid data leakage.
    # We use OneHotEncoder() for the categorical feature PFT, since we want to avoid any ranking in Land Cover Classes.
    # Instead of IGBP = {1..n}, we end up with one feature for each IGBP class set to 0 or 1.

    num_pipeline = Pipeline([
        ('minmax_scaler', StandardScaler())
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attributes),
        ('cat', OneHotEncoder(), cat_attributes)
    ])

    # Store transformation pipeline to disk
    with open(f"models/{timestamp}/pipeline.pickle", "wb") as output_file:
        pickle.dump(full_pipeline, output_file)

    # apply pipeline, fit only to training data
    df_train = full_pipeline.fit_transform(x_train)
    df_test = full_pipeline.transform(x_test)
    df_val = full_pipeline.transform(x_val)

    # Extract new feature names from OneHotEncoder().categories_ ('IGBP' -> ['DBF, 'EBF', ...])
    new_categories = full_pipeline.transformers_[1][1].categories_[0].tolist()

    # convert numpy array back to data frame
    df_train = pd.DataFrame(df_train, columns=num_attributes + new_categories)
    df_train.columns = df_train.columns.map(''.join)
    df_test = pd.DataFrame(df_test, columns=num_attributes + new_categories)
    df_test.columns = df_test.columns.map(''.join)
    df_val = pd.DataFrame(df_val, columns=num_attributes + new_categories)
    df_val.columns = df_val.columns.map(''.join)

    # write out transformed data to CSV for analysis purpose (not used in neural network)
    df_train.to_csv('output/training/train_samples.csv', index=False)
    df_test.to_csv('output/training/test_samples.csv', index=False)
    df_val.to_csv('output/training/val_samples.csv', index=False)

    y_train.to_csv('output/training/train_lables.csv', index=False)
    y_test.to_csv('output/training/test_lables.csv', index=False)
    y_val.to_csv('output/training/val_lables.csv', index=False)

    # if external prediction is activated, external input features are transformed here
    if ext_prediction is not None:
        ext_data = load_external(ext_prediction, features=features, freq=freq)
        for sitename, df in ext_data.items():
            df_transformed = df.reset_index(drop=True)
            df_transformed = pd.DataFrame(full_pipeline.transform(df_transformed),
                                          columns=num_attributes + new_categories)
            df_transformed.to_csv(f"output/fluxnet/{sitename}.csv", index=False)

    return {"Xtrain": np.array(df_train), "Ytrain": np.expand_dims(np.array(y_train), axis=1),
            "Xtest": np.array(df_test), "Ytest": np.expand_dims(np.array(y_test), axis=1),
            "Xval": np.array(df_val), "Yval": np.expand_dims(np.array(y_val), axis=1),
            "untransformed": {"Xtrain": x_train, "Xtest": x_test, "Xval": x_val}}


def load_external(path: str, features: list, freq: str = "1D") -> dict:
    """Loads tabular data for prediction outside of training. Files must contain all variables involved in training.
    :param path: Path to input features for prediction
    :param features: Input features for prediction
    :param freq: Temporal resolution (1D|1H)
    :return fitered_data: Dictionary with input features for each site
    """
    csv_files = sorted(glob.glob(f"{path}/*.csv"))
    ext_data = {}
    for csv_file in csv_files:
        sitename = os.path.splitext(os.path.basename(csv_file))[0]
        ext_data[sitename] = pd.read_csv(csv_file, index_col=0, parse_dates=True)

        try:
            igbp = ext_data[sitename]["IGBP"].dropna().iloc[0]
        except IndexError:
            print(f"WARNING: {sitename} contains empty dataframe or is missing IGBP.")
            del ext_data[sitename]
            continue
        # todo: No preprocessing done before resampling (e.g. filter invalid data)
        # Check if all features are present in FLUXNET samples
        if not all(x in list(ext_data[sitename].columns) for x in features):
            continue

        # Resample data to specifiec temporal resolution
        ext_data[sitename] = ext_data[sitename].resample(freq).mean()

        # Set IGBP column to PFT
        ext_data[sitename]["IGBP"] = igbp

        # Drop features not used in training
        ext_data[sitename] = drop_features(ext_data[sitename], features+["IGBP"], target=[])

    # Filter data by PFT and time period
    filtered_data = {}
    for sitename, df in ext_data.items():
        if df["IGBP"].unique().item() in ['EBF', 'ENF', 'DBF',  'SAV', 'MF', 'DNF']:
            # 2002-07-04: Start of MODIS data
            filtered_data[sitename] = df["2002-07-04": "2015-12-31"]
        else:
            continue
    return filtered_data


def load(path_csv: str, freq: str, features: list, timestamp: str, blacklist=False, target="transpiration",
         external_prediction: str = None, ) -> tuple:
    """Loads the data from passed path and does preprocessing for the neural network.

    :param external_prediction: If path is specified, external locations are transformed for prediction
    :param path_csv: Directory containing CSV for point locaions
    :param freq: Temporal resolution. Currently only "1D" supported
    :param features: List with input variables to be used
    :param timestamp: Date and Time of model run
    :param blacklist: If True, sites specified in metadata are removed
    :param target: Name of target variable (transpiration|gc|alpha)
    :return train_data: Dictionary with preprocessed training data
    """

    # Create metadata dictionary
    keys = ["site_info", "setup", "model", "results"]
    metadata = {key: {} for key in keys}

    # load tabular point data
    sfn_data = load_tabular(path_csv, features, target=target, freq=freq)

    # filter out time series shorter than 1 year so site covers at least one full seasonal cycle
    # sfn_data = filter_short_timeseries(sfn_data)

    # optional: Read a "blacklist/whitelist" to exclude sites from training
    if isinstance(blacklist, str):
        site_selection = pd.read_csv(blacklist, index_col='si_code')
        site_selection = list(site_selection.loc[site_selection['exclude'] == 0].index)
        forbidden_sites = [site for site in sfn_data.keys() if site in site_selection]
        for site in forbidden_sites:
            del sfn_data[site]

    # Store metadata
    metadata["site_info"]["n_sites"] = len(sfn_data)
    metadata["site_info"]["sitenames"] = sorted(list(sfn_data.keys()))
    # metadata["site_info"]["ecosystems"] = [arr.item() for arr in np.unique(([data["IGBP"].unique()
    #                                                                        for data in sfn_data.values()]))]
    metadata["setup"]["frequency"] = freq
    metadata["setup"]["target"] = target
    metadata["setup"]["features"] = features

    # Convert dictionary of sites to single dataframe. Geographic information will be lost from here.
    sfn_data = dict_to_df(sfn_data)
    # Filter out data (e.g. T < 0 mm, net radiation < 50 W/m2)
    #sfn_data = filter_data(sfn_data)

    # Shuffle data randomly and split into training, testing, validation. Temporal information will be lost from here.
    x_train, x_test, x_val, y_train, y_test, y_val = split_data(sfn_data, target=target)

    # Scale and encode data for neural network
    train_data = transform_data(x_train, x_test, x_val, y_train, y_test, y_val,
                                features, timestamp=timestamp, ext_prediction=external_prediction, freq=freq)
    return train_data, metadata
