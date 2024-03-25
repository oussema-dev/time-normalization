"""This module provides methods for the normalization strategies used in this project
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler


def zero_padding(ap, ml, v):
    """this method replaces the nan values in the provided dataframes with zeroes

    Parameters:
    ap (dataframe): grf antero-posterior data
    ml (dataframe): grf medio-lateral data
    v (dataframe): grf vert data

    Returns:
    ap (dataframe): zero padded grf antero-posterior data
    ml (dataframe): zero padded grf medio-lateral data
    v (dataframe): zero padded grf vert data
    """

    ap.fillna(0, inplace=True)
    ml.fillna(0, inplace=True)
    v.fillna(0, inplace=True)

    return ap, ml, v


def normalize_by_body_mass(ap, ml, v):
    """this method divides the data values in each row by the participant's body mass

    Parameters:
    ap (dataframe): grf antero-posterior data
    ml (dataframe): grf medio-lateral data
    v (dataframe): grf vert data

    Returns:
    ap (dataframe): normalized grf antero-posterior data
    ml (dataframe): normalized grf medio-lateral data
    v (dataframe): normalized grf vert data
    """

    # Normalize all data points by dividing each feature values by the subject body_mass
    # Get the index of 'body_mass' column
    body_mass_index = ap.columns.get_loc("BODY_MASS")
    # Define the columns to be scaled
    columns_to_scale_ap = ap.columns[4:body_mass_index].tolist()
    columns_to_scale_ml = ml.columns[4:body_mass_index].tolist()
    columns_to_scale_v = v.columns[4:body_mass_index].tolist()

    # Divide every value in each row of the specified columns by the 'body_mass' value of that same row
    ap[columns_to_scale_ap] = ap[columns_to_scale_ap].div(ap["BODY_MASS"], axis=0)
    ml[columns_to_scale_ml] = ml[columns_to_scale_ml].div(ml["BODY_MASS"], axis=0)
    v[columns_to_scale_v] = v[columns_to_scale_v].div(v["BODY_MASS"], axis=0)

    return ap, ml, v


def min_max_scale(ap, ml, v):
    """This method applices a min-max scaler to the data

    Parameters:
    ap (dataframe): grf antero-posterior data
    ml (dataframe): grf medio-lateral data
    v (dataframe): grf vert data

    Returns:
    ap (dataframe): scaled grf antero-posterior data
    ml (dataframe): scaled grf medio-lateral data
    v (dataframe): scaled grf vert data
    """

    X_data1 = ap.iloc[:, 4:-2]
    X_data2 = ml.iloc[:, 4:-2]
    X_data3 = v.iloc[:, 4:-2]

    scaler = MinMaxScaler()

    scaled_ap = pd.DataFrame(scaler.fit_transform(X_data1), columns=X_data1.columns)
    scaled_ml = pd.DataFrame(scaler.fit_transform(X_data2), columns=X_data2.columns)
    scaled_v = pd.DataFrame(scaler.fit_transform(X_data3), columns=X_data3.columns)

    concatenated_ap = pd.concat([ap.iloc[:, :4], scaled_ap, ap.iloc[:, -2:]], axis=1)
    concatenated_ml = pd.concat([ml.iloc[:, :4], scaled_ml, ml.iloc[:, -2:]], axis=1)
    concatenated_v = pd.concat([v.iloc[:, :4], scaled_v, v.iloc[:, -2:]], axis=1)

    return concatenated_ap, concatenated_ml, concatenated_v


def interpolate_row(row):
    """a method for interpolating a row to 100 values

    Parameters:
    row (dataframe record): the row to be interpolated to 100 frames

    Returns:
    interpolated_row (dataframe record): the interpolated row
    """
    non_nan_indices = np.where(~np.isnan(row))[0]
    if len(non_nan_indices) <= 1:
        return np.full(100, np.nan)
    interpolated_row = np.interp(
        np.linspace(0, 1, 100),
        np.linspace(0, 1, len(non_nan_indices)),
        row[non_nan_indices],
    )
    return interpolated_row


def interpolate(ap, ml, v):
    """a method for interpolating the data to 100 frames for each row

    Parameters:
    X_data1 (dataframe): the first channel to be interpolated to 100 frames
    X_data2 (dataframe): the second channel to be interpolated to 100 frames
    X_data3 (dataframe): the third channel to be interpolated to 100 frames

    Returns:
    X_data1 (dataframe): the interpolated first channel
    X_data2 (dataframe): the interpolated second channel
    X_data3 (dataframe): the interpolated third channel
    """
    X_data1 = ap.iloc[:, 4:-2]
    X_data2 = ml.iloc[:, 4:-2]
    X_data3 = v.iloc[:, 4:-2]

    # Interpolate data and shrink it to 100 frames
    data_array_1 = X_data1.to_numpy()
    data_array_2 = X_data2.to_numpy()
    data_array_3 = X_data3.to_numpy()

    interpolated_data_1 = np.apply_along_axis(interpolate_row, axis=1, arr=data_array_1)
    interpolated_data_2 = np.apply_along_axis(interpolate_row, axis=1, arr=data_array_2)
    interpolated_data_3 = np.apply_along_axis(interpolate_row, axis=1, arr=data_array_3)

    X_data1 = pd.DataFrame(interpolated_data_1)
    X_data2 = pd.DataFrame(interpolated_data_2)
    X_data3 = pd.DataFrame(interpolated_data_3)

    concatenated_ap = pd.concat([ap.iloc[:, :4], X_data1, ap.iloc[:, -2:]], axis=1)
    concatenated_ml = pd.concat([ml.iloc[:, :4], X_data2, ml.iloc[:, -2:]], axis=1)
    concatenated_v = pd.concat([v.iloc[:, :4], X_data3, v.iloc[:, -2:]], axis=1)

    return concatenated_ap, concatenated_ml, concatenated_v
