"""This is a config file that stores hard-coded variables
"""

import os

# Folder names
data_fld = os.path.join("data") + os.path.sep
logs_fld = os.path.join("logs") + os.path.sep
downloaded_data_fld = os.path.join(data_fld, "downloaded data") + os.path.sep
processed_data_fld = os.path.join(data_fld, "processed data") + os.path.sep
raw_data_fld = os.path.join(data_fld, "raw data") + os.path.sep

# Participants with ambiguous sex data
excluded_participants = [
    24,
    27,
    41,
    47,
    49,
    51,
    54,
    55,
    57,
    74,
    75,
    80,
    82,
    85,
    90,
    105,
    106,
    111,
    116,
    121,
    124,
    125,
    126,
    128,
    143,
    144,
    348,
    349,
    350,
]
