"""This module provides different utility functions for data processing
"""

import urllib.request
import os
import shutil
import pandas as pd
from config import (
    data_fld,
    logs_fld,
    downloaded_data_fld,
    processed_data_fld,
    raw_data_fld,
    excluded_participants,
)


def create_folders():
    """This method creates necessary folders for the project

    Parameters:
    None

    Returns:
    None
    """

    try:
        # Remove data folder and its contents
        print("Deleting existing data...")
        shutil.rmtree(data_fld)
        shutil.rmtree(logs_fld)
    except FileNotFoundError:
        print("Dataset folder not found")

    # Create folders
    print("Creating folders...")
    os.mkdir(data_fld)
    os.mkdir(logs_fld)
    os.mkdir(downloaded_data_fld)
    os.mkdir(processed_data_fld)
    os.mkdir(raw_data_fld)


def download_gutenburg_db():
    """loads Gutenburg gait database,
    available at: https://doi.org/10.6084/m9.figshare.c.5311538.v1
    """

    # Download participant metadata
    url_meta = "https://springernature.figshare.com/ndownloader/files/26576495"
    file_name_meta = "GRF_metadata.csv"
    get_gutenburg(url_meta, file_name_meta)

    # Download raw grf vert for right leg
    url_GRF_F_V_RAW_right = (
        "https://springernature.figshare.com/ndownloader/files/26576491"
    )
    file_GRF_F_V_RAW_right = "GRF_F_V_RAW_right.csv"
    get_gutenburg(url_GRF_F_V_RAW_right, file_GRF_F_V_RAW_right)

    # Download raw grf medio-lateral for right leg
    url_GRF_F_ML_RAW_right = (
        "https://springernature.figshare.com/ndownloader/files/26576487"
    )
    file_GRF_F_ML_RAW_right = "GRF_F_ML_RAW_right.csv"
    get_gutenburg(url_GRF_F_ML_RAW_right, file_GRF_F_ML_RAW_right)

    # Download raw grf antero-posterior for right leg
    url_GRF_F_AP_RAW_right = (
        "https://springernature.figshare.com/ndownloader/files/26576469"
    )
    file_GRF_F_AP_RAW_right = "GRF_F_AP_RAW_right.csv"
    get_gutenburg(url_GRF_F_AP_RAW_right, file_GRF_F_AP_RAW_right)

    # Download processed grf vert for right leg
    url_GRF_F_V_PRO_right = (
        "https://springernature.figshare.com/ndownloader/files/26576493"
    )
    file_GRF_F_V_PRO_right = "GRF_F_V_PRO_right.csv"
    get_gutenburg(url_GRF_F_V_PRO_right, file_GRF_F_V_PRO_right)

    # Download processed grf medio-lateral for right leg
    url_GRF_F_ML_PRO_right = (
        "https://springernature.figshare.com/ndownloader/files/26576481"
    )
    file_GRF_F_ML_PRO_right = "GRF_F_ML_PRO_right.csv"
    get_gutenburg(url_GRF_F_ML_PRO_right, file_GRF_F_ML_PRO_right)

    # Download processed grf antero-posterior for right leg
    url_GRF_F_AP_PRO_right = (
        "https://springernature.figshare.com/ndownloader/files/26576473"
    )
    file_GRF_F_AP_PRO_right = "GRF_F_AP_PRO_right.csv"
    get_gutenburg(url_GRF_F_AP_PRO_right, file_GRF_F_AP_PRO_right)

    print("Gutenberg dataset downloaded sucessfully")


def get_gutenburg(url, file_name):
    """download gutenburg dataset"""

    if not os.path.exists(downloaded_data_fld):
        os.makedirs(downloaded_data_fld)

    file_pth = os.path.join(downloaded_data_fld, file_name)
    if os.path.exists(file_pth):
        print("Previously downloaded {} to {}".format(file_name, file_pth))
    else:
        print("Downloading {} from {}".format(file_name, url))
        urllib.request.urlretrieve(url, file_pth)


def process_data():
    print("Processing data...")
    process_file("GRF_F_AP_PRO_right.csv")
    process_file("GRF_F_AP_RAW_right.csv")
    process_file("GRF_F_ML_PRO_right.csv")
    process_file("GRF_F_ML_RAW_right.csv")
    process_file("GRF_F_V_PRO_right.csv")
    process_file("GRF_F_V_RAW_right.csv")
    print("Data processed sucessfully")


def process_file(file_name):
    """this method add the sex and body mass columns to the provided
    dataframe and removes the participants with ambiguous sex data

    Parameters:
    file_name (string): name of the csv file to be processed

    Returns:
    None
    """
    file = pd.read_csv(os.path.join(downloaded_data_fld, file_name))
    metadata = pd.read_csv(os.path.join(downloaded_data_fld, "GRF_metadata.csv"))

    # Create a dictionary mapping the unique identifiers to the 'BODY_MASS' column from the metadata
    body_mass_mapping = metadata.set_index(["DATASET_ID", "SUBJECT_ID", "SESSION_ID"])[
        "BODY_MASS"
    ].to_dict()

    # Create a dictionary mapping the unique identifiers to the 'SEX' column from the metadata
    sex_mapping = metadata.set_index(["DATASET_ID", "SUBJECT_ID", "SESSION_ID"])[
        "SEX"
    ].to_dict()

    # Map the 'BODY_MASS' column from the metadata file to each data file based on matching identifiers
    file["BODY_MASS"] = file.set_index(
        ["DATASET_ID", "SUBJECT_ID", "SESSION_ID"]
    ).index.map(body_mass_mapping)

    # Map the 'SEX' column from the metadata file to each data file based on matching identifiers
    file["SEX"] = file.set_index(["DATASET_ID", "SUBJECT_ID", "SESSION_ID"]).index.map(
        sex_mapping
    )

    # Exclude participoants with ambiguous sex column (542 rows to be excluded)

    file = file[~file["SUBJECT_ID"].isin(excluded_participants)]

    index = file_name.find("PRO")
    if index != -1:
        new_file_path = os.path.join(processed_data_fld, file_name)

    else:
        new_file_path = os.path.join(raw_data_fld, file_name)

    file.to_csv(new_file_path, index=False)


def load_data(data_type):
    """this method loads the dataframes according to the data type

    Parameters:
    data_type (string): type of the data to be loaded

    Returns:
    ap (dataframe): grf antero-posterior data
    ml (dataframe): grf medio-lateral data
    v (dataframe): grf vert data
    """

    if data_type == "PRO":
        ap = pd.read_csv(os.path.join(processed_data_fld, "GRF_F_AP_PRO_right.csv"))
        ml = pd.read_csv(os.path.join(processed_data_fld, "GRF_F_ML_PRO_right.csv"))
        v = pd.read_csv(os.path.join(processed_data_fld, "GRF_F_V_PRO_right.csv"))
    else:
        ap = pd.read_csv(os.path.join(raw_data_fld, "GRF_F_AP_RAW_right.csv"))
        ml = pd.read_csv(os.path.join(raw_data_fld, "GRF_F_ML_RAW_right.csv"))
        v = pd.read_csv(os.path.join(raw_data_fld, "GRF_F_V_RAW_right.csv"))
    return ap, ml, v


def separate_data(dataframe):
    """this method separates the output and the subject id columns

    Parameters:
    dataframe (dataframe): dataframe from which to extract the columns

    Returns:
    y (list): the output column
    subject (list): the subject id column
    """
    # Separate the output 'SEX' column
    y = dataframe["SEX"]

    # Separate the column 'SUBJECT_ID' to be used for splitting
    subject = dataframe["SUBJECT_ID"]
    return y, subject


def drop_extra_columns(ap, ml, v):
    """this method drops all columns from the dataframe execpt
    the data values needed to train the model

    Parameters:
    ap (dataframe): grf antero-posterior data
    ml (dataframe): grf medio-lateral data
    v (dataframe): grf vert data

    Returns:
    X_data1 (dataframe): the processed first channel
    X_data2 (dataframe): the processed second channel
    X_data3 (dataframe): the processed third channel
    """
    # Drop the first four and last 2 columns from each DataFrame to only keep the data values
    X_data1 = ap.iloc[:, 4:-2]
    X_data2 = ml.iloc[:, 4:-2]
    X_data3 = v.iloc[:, 4:-2]

    return X_data1, X_data2, X_data3
