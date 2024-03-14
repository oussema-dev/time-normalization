import numpy as np
import random


def subject_wise_split_tensor(
    x, y, participant, subject_wise=True, test_size=0.10, random_state=42
):
    """Split data into train and test sets via an inter-subject scheme, see:
    Shah, V., Flood, M. W., Grimm, B., & Dixon, P. C. (2022). Generalizability of deep learning models for predicting outdoor irregular walking surfaces.
    Journal of Biomechanics, 139, 111159. https://doi.org/10.1016/j.jbiomech.2022.111159

    Arguments:
        x: nd.array, feature space
        y: nd.array, label class
        participant: nd.array, participant associated with each row in x and y
        subject_wise: bool, choices {True, False}, default = True. True = subject-wise split approach, False random-split
        test_size: float, number between 0 and 1. Default = 0.10. percentage spilt for test set.
        random_state: int. default = 42. Seed selector for numpy random number generator.
    Returns:
        x_train: nd.array, train set for feature space
        x_test: nd.array, test set for feature space
        y_train: nd.array, train set label class
        y_test: nd.array, test set label class
        subject_train: nd.array[string], train set for participants by row of input data
        subjects_test: nd.array[string[, test set for participants by row of input data
    """
    if type(participant) == list:
        participant = np.asarray(participant, dtype=np.float32)

    np.random.seed(random_state)
    if subject_wise:
        # Extract unique participants
        uniq_parti = np.unique(participant)
        # Calculate the number of participants for the test set
        num = np.round(uniq_parti.shape[0] * test_size).astype("int64")
        np.random.shuffle(uniq_parti)
        extract = uniq_parti[0:num]
        test_index = np.array([], dtype="int64")
        for j in extract:
            test_index = np.append(test_index, np.where(participant == j)[0])
        # Remove test set indices from all indices to get train set indices
        train_index = np.delete(np.arange(len(participant)), test_index)
        # Shuffle the train and test indices
        np.random.shuffle(test_index)
        np.random.shuffle(train_index)

    else:
        index = np.arange(len(participant)).astype("int64")
        np.random.shuffle(index)
        num = np.round(participant.shape[0] * test_size).astype("int64")
        # Select train and test set indices
        test_index = index[0:num]
        train_index = index[num:]

    # Extract train and test sets based on the computed indices
    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    subject_train = participant[train_index]
    subject_test = participant[test_index]

    return x_train, x_test, y_train, y_test, subject_train, subject_test


def subject_wise_split_tabular(df, random_state):
    """Split data into train and test sets via an inter-subject scheme

    Arguments:
        df: dataframe, data to be split
        random_state: int. Seed selector for numpy random number generator.
    Returns:
        X_train: nd.array, train set for feature space
        X_test: nd.array, test set for feature space
        y_train: nd.array, train set label class
        y_test: nd.array, test set label class
    """

    features = [
        "min_value",
        "max_value",
        "mean_value",
        "std_deviation",
        "median_value",
        "75th_percentile",
        "range",
    ]
    participants = df["SUBJECT_ID"].unique()
    random.seed(random_state)
    random.shuffle(participants)

    # Assign a certain percentage of participants to the training set and the remaining to the test set
    train_percentage = 0.8
    num_train = int(train_percentage * len(participants))

    train_participants = participants[:num_train]
    test_participants = participants[num_train:]

    # Use the participant list to filter the dataset into training and testing sets
    train_set = df[df["SUBJECT_ID"].isin(train_participants)]
    test_set = df[df["SUBJECT_ID"].isin(test_participants)]

    # Split the data into features (X) and labels (y)
    X_train, y_train = train_set[features], train_set["SEX"]
    X_test, y_test = test_set[features], test_set["SEX"]

    return X_train, X_test, y_train, y_test
