"""This module is used to initiate the training model
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM


def initiate_model(model_type, X_train):
    """This method initiates the model

    Parameters:
    model_type (str): type of the model to be initiated
    X_train (tensor): training data

    Returns:
    model (dkeras model): initiated model
    """

    if model_type == "cnn":
        model = Sequential()
        model.add(
            Conv1D(
                32,
                kernel_size=3,
                activation="relu",
                input_shape=(X_train.shape[1], X_train.shape[2]),
            )
        )
        model.add(MaxPooling1D(pool_size=2))
        model.add(Conv1D(64, kernel_size=3, activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())

        # Dense layers for classification
        model.add(Dense(128, activation="relu"))
        model.add(Dense(2, activation="softmax"))

    else:
        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=2, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model
