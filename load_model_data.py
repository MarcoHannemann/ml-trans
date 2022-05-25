import os
import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# todo: filter out negative transpiration and nightime values


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

    # read CSV file for each site and store to dictionary, drop rows containing NaN
    data = {}
    for csv_file in csv_files:
        sitename = os.path.splitext(os.path.basename(csv_file))[0]
        data[sitename] = pd.read_csv(csv_file, index_col=0, parse_dates=True).dropna(how="any")
        try:
            igbp = data[sitename]["IGBP"].iloc[0]
        except IndexError:
            print(f"WARNING: {sitename} contains empty dataframe or is missing IGBP/canopy height.")
            del data[sitename]
            continue
        data[sitename].loc[(data[sitename]['vpd'] < 0)] = np.nan
        data[sitename].loc[(data[sitename][target] < 0)] = np.nan
        data[sitename].loc[(data[sitename]['ssr'] < 0)] = np.nan
        if "tr_ca" in list(data[sitename].columns):
            data[sitename].drop(columns=["tr_ca"], inplace=True)
        data[sitename].dropna(how="any", inplace=True)
        data[sitename] = data[sitename].resample(freq).mean()
        data[sitename].dropna(how="any", inplace=True)
        data[sitename]["IGBP"] = igbp

    data_new = {}
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
    data = pd.concat(data.values()).reset_index(drop=True)
    return data


def filter_data(data: pd.DataFrame) -> pd.DataFrame:
    """Filters out nighttime data and Transpiration lower zero

    :param data: training data
    :return: data: training data
    """
    data = data.loc[data["vpd"] > 0]
    data = data.loc[data["ssr"] > 50]
    data = data.loc[data["tr"] > 0.01]
    return data.reset_index(drop=True)


def drop_features(features):
    """Drop features if specified."""
    pass


def isfeature(df, features):
    """Check if specified features are available in data."""
    for feature in features:
        if feature not in df.columns:
            return False
    return True


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
                   features: list, ext_prediction: str = None, freq: str = "1D") -> dict:
    """Transforms and fits data. Includes normalization and encoding.

    :param ext_prediction: If path is specified, external locations are transformed for predictions
    :param x_train: Training input data
    :param x_test: Testing input data
    :param x_val: Validation input data
    :param y_train: Training target
    :param y_test: Testing target
    :param y_val: Validation target
    :param features: List of feature names incoroporated in model
    :param ext_prediction: Path to directory with external sites (CSV)
    :param freq: Temporal resolution 1D | 1H
    :return: dictionary containing transformed training data
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
        ext_data = load_external(ext_prediction, freq=freq)
        for sitename, df in ext_data.items():
            df_transformed = df.reset_index(drop=True)
            df_transformed = pd.DataFrame(full_pipeline.transform(df_transformed), columns=num_attributes + new_categories)
            df_transformed.to_csv(f"output/fluxnet/{sitename}.csv", index=False)

    return {"Xtrain": np.array(df_train), "Ytrain": np.expand_dims(np.array(y_train), axis=1),
            "Xtest": np.array(df_test), "Ytest": np.expand_dims(np.array(y_test), axis=1),
            "Xval": np.array(df_val), "Yval": np.expand_dims(np.array(y_val), axis=1)}


def load_external(path: str, freq: str = "1D") -> dict:
    """Loads tabular data for prediction outside of training. Files must contain all variables involved in training."""
    csv_files = sorted(glob.glob(f"{path}/*.csv"))
    ext_data = {}
    for csv_file in csv_files:
        sitename = os.path.splitext(os.path.basename(csv_file))[0]
        ext_data[sitename] = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        try:
            igbp = ext_data[sitename]["IGBP"].iloc[0]
        except IndexError:
            print(f"WARNING: {sitename} contains empty dataframe or is missing IGBP.")
            del ext_data[sitename]
            continue
        # todo: No preprocessing done before resampling (e.g. filter invalid data)
        ext_data[sitename] = ext_data[sitename].resample(freq).mean()
        ext_data[sitename]["IGBP"] = igbp
    filtered_data = {}
    for sitename, df in ext_data.items():
        if df["IGBP"].unique().item() in ["ENF", "DBF", "EBF", "MF"]:
            filtered_data[sitename] = df
        else:
            continue
    return filtered_data


def load(path_csv: str, freq: str, features: list, blacklist=False, target="transpiration", external_prediction: str = None) -> tuple:
    """Loads the data from passed path and does preprocessing for the neural network. Should be called from external
    script.

    :param external_prediction: If path is specified, external locations are transformed for prediction
    :param path_csv: Directory containing CSV for point locaions
    :param freq: Temporal resolution. Currently only "1D" supported
    :param features: List with input variables to be used
    :param blacklist: If True, sites specified in metadata are removed
    :param target: Name of target variable
    :return train_data: Dictionary with preprocessed training data
    """
    keys = ["site_info", "setup", "model", "results"]
    metadata = {key:{} for key in keys}
    # load tabular point data
    sfn_data = load_tabular(path_csv, features, target=target, freq=freq)

    # filter out time series shorter than 1 year so site covers at least one full seasonal cycle
    #sfn_data = filter_short_timeseries(sfn_data)

    # optional: Read a "blacklist/whitelist" to exclude sites from training
    if isinstance(blacklist, str):
        site_selection = pd.read_csv(blacklist, index_col='si_code')
        site_selection = list(site_selection.loc[site_selection['exclude'] == 0].index)
        forbidden_sites = [site for site in sfn_data.keys() if site in site_selection]
        for site in forbidden_sites:
            del sfn_data[site]
    metadata["site_info"]["n_sites"] = len(sfn_data)
    metadata["site_info"]["sitenames"] = sorted(list(sfn_data.keys()))
    #metadata["site_info"]["ecosystems"] = [arr.item() for arr in np.unique(([data["IGBP"].unique()
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
                                features, ext_prediction=external_prediction, freq=freq)
    return train_data, metadata
