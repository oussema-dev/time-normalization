"""This module provides methods for the normalization strategies used in this project
"""

import numpy as np
import pandas as pd


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
    ap = ap.div(ap["BODY_MASS"], axis=0)
    ml = ml.div(ml["BODY_MASS"], axis=0)
    v = v.div(v["BODY_MASS"], axis=0)
    return ap, ml, v


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


def interpolate(X_data1, X_data2, X_data3):
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

    return X_data1, X_data2, X_data3
