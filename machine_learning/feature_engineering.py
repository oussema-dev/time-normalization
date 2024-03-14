"""This module provides the feature extraction function
"""

import pandas as pd


def extract_features(df):
    """This is the feature extraction method it calculates statistical features

    Parameters:
    df (dataframe): dataframe from which to extract the features

    Returns:
    result_df (dataframe): dataframe with extracted features
    """

    # Identify the indices of 'SESSION_ID' and 'SEX' columns
    session_id_index = df.columns.get_loc("SESSION_ID")
    body_mass_index = df.columns.get_loc("BODY_MASS")

    # Get the columns between 'SESSION_ID' and 'BODY_MASS'
    data_columns = df.columns[session_id_index + 1 : body_mass_index]

    # Define a function to calculate statistical features for each row
    def calculate_features(row):
        row_data = row[data_columns]
        features = {
            "min_value": row_data.min(),
            "max_value": row_data.max(),
            "mean_value": row_data.mean(),
            "std_deviation": row_data.std(),
            "median_value": row_data.median(),
            "75th_percentile": row_data.quantile(0.75),
            "range": row_data.max() - row_data.min(),
        }
        return pd.Series(features)

    # Apply the function to each row
    row_features = df.apply(calculate_features, axis=1)

    # Concatenate the original DataFrame with the calculated features
    result_df = pd.concat(
        [
            df[["SUBJECT_ID", "SESSION_ID"]],
            row_features,
            df[["BODY_MASS", "SEX"]],
        ],
        axis=1,
    )

    return result_df
